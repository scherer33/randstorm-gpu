mod cuda;

use bitcoin::address::Address;
use bitcoin::hashes::Hash;
use clap::Parser;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use std::str::FromStr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use randstorm::prng::{XorShift128Plus, SecureRandom};
use randstorm::ec_table::{EcTable, NLIMBS, Fe, fe_add, fe_mul, fe_sub};

const FE_ZERO: Fe = [0u32; NLIMBS];
const FE_ONE: Fe = {
    let mut a = [0u32; NLIMBS];
    a[0] = 1;
    a
};
const FE_TWO: Fe = {
    let mut a = [0u32; NLIMBS];
    a[0] = 2;
    a
};
const FE_THREE: Fe = {
    let mut a = [0u32; NLIMBS];
    a[0] = 3;
    a
};
const FE_FOUR: Fe = {
    let mut a = [0u32; NLIMBS];
    a[0] = 4;
    a
};
const FE_EIGHT: Fe = {
    let mut a = [0u32; NLIMBS];
    a[0] = 8;
    a
};

#[derive(Parser, Debug)]
#[command(name = "randstorm")]
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

    #[arg(long, short, default_value = "randstorm_results.txt")]
    output: String,

    #[arg(long, default_value_t = false)]
    no_cuda: bool,

    #[arg(short, long, default_value_t = 0)]
    threads: usize,
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
    let threads = if args.threads == 0 {
        rayon::current_num_threads()
    } else {
        args.threads
    };

    eprintln!("randstorm — Batch Inversion Pipeline");
    eprintln!("Target: {}  Hash160: {}", args.target, hex::encode(target_h160));
    eprintln!("Seeds: {} → {}  (total: {})", args.start, args.end, total);

    let base_pool = {
        let mut prng = XorShift128Plus::new(args.v8seed);
        let mut pool = [0u8; 256];
        let mut p = 0;
        while p < 256 {
            let t = prng.next_u16();
            pool[p] = (t >> 8) as u8;
            p += 1;
            pool[p] = t as u8;
            p += 1;
        }
        pool
    };

    eprintln!("Precomputing EC table (256 points)...");
    let ec_table = EcTable::precompute();
    eprintln!("EC table ready ({} KB)", std::mem::size_of::<EcTable>() / 1024);

    let cuda = if !args.no_cuda {
        cuda::CudaContext::try_init()
    } else {
        None
    };

    if let Some(cuda_ctx) = cuda {
        if !cuda_ctx.upload_ec_table(&ec_table) {
            eprintln!("Failed to upload EC table to GPU, falling back to CPU");
            run_cpu(args, base_pool, ec_table, target_h160, total, threads);
        } else {
            eprintln!("[CUDA] EC table uploaded to constant memory");
            run_gpu(args, base_pool, ec_table, target_h160, total, cuda_ctx);
        }
    } else {
        eprintln!("Mode: CPU ({} threads)", threads);
        run_cpu(args, base_pool, ec_table, target_h160, total, threads);
    }
}

fn run_gpu(
    args: Args,
    base_pool: [u8; 256],
    _ec_table: EcTable,
    target_h160: [u8; 20],
    total: u64,
    cuda: cuda::CudaContext,
) {
    let batch = args.batch_size;
    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} seed:{msg} ({per_sec} keys/s, ETA: {eta})")
            .unwrap()
            .progress_chars("#>- "),
    );

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
            sr.get_bytes(&mut key_buf[i * 32..(i + 1) * 32]);
        }

        // GPU Phase 1: private keys → Jacobian (Z values returned to CPU)
        let z_values: &[Fe] = unsafe {
            if cuda
                .pk_to_jacobian(&key_buf[..n * 32], n, &mut z_buf)
                .is_none()
            {
                eprintln!("GPU Phase 1 failed");
                break;
            }
            std::slice::from_raw_parts(z_buf.as_ptr() as *const Fe, n)
        };

        // CPU: batch inversion (1 modular inverse + O(N) multiplications)
        let z_invs = randstorm::ec_table::batch_invert(z_values);

        let z_inv_flat: &[u32] =
            unsafe { std::slice::from_raw_parts(z_invs.as_ptr() as *const u32, n * NLIMBS) };

        // GPU Phase 2: Jacobian + Z-inverse → affine → hash160
        if cuda
            .jacobian_to_hash160(z_inv_flat, n, &mut h160_results)
            .is_some()
        {
            for i in 0..n {
                if h160_results[i] == target_h160 {
                    let key_hex = key_buf[i * 32..(i + 1) * 32]
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<String>();
                    let seed = current + i as u64;
                    eprintln!("\n\nMATCH! Seed: {} Key: {}", seed, key_hex);

                    if let Ok(mut f) = std::fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(&args.output)
                    {
                        use std::io::Write;
                        let _ = writeln!(f, "{},{},{}", seed, key_hex, args.target);
                    }
                    eprintln!("Result saved to {}", args.output);

                    *found_key.lock().unwrap() = Some(key_hex);
                    found.store(true, Ordering::Relaxed);
                    break;
                }
            }
        } else {
            eprintln!("GPU Phase 2 failed");
            break;
        }

        pb.set_message(format!("{}", current));
        current += n as u64;
        pb.inc(n as u64);
    }

    pb.finish_with_message("Done");
    print_result(&found_key);
}

fn run_cpu(
    args: Args,
    base_pool: [u8; 256],
    ec_table: EcTable,
    target_h160: [u8; 20],
    total: u64,
    threads: usize,
) {
    let batch = args.batch_size;
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("Failed to build thread pool");

    let pb = ProgressBar::new(total);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} seed:{msg} ({per_sec} keys/s, ETA: {eta})")
            .unwrap()
            .progress_chars("#>- "),
    );

    let found = Arc::new(AtomicBool::new(false));
    let found_key = Arc::new(Mutex::new(None::<String>));
    let ec_table = Arc::new(ec_table);
    let base_pool = Arc::new(base_pool);
    let output = Arc::new(args.output.clone());
    let target_arg = Arc::new(args.target.clone());

    let mut current = args.start;

    while current < args.end && !found.load(Ordering::Relaxed) {
        let batch_end = (current + batch as u64).min(args.end);
        let n = (batch_end - current) as usize;

        // Generate all private keys for this batch
        let mut key_buf = vec![0u8; n * 32];
        for i in 0..n {
            let mut sr = SecureRandom::from_pool(&base_pool, current + i as u64);
            sr.get_bytes(&mut key_buf[i * 32..(i + 1) * 32]);
        }

        // CPU: ec_table scalar mult → jacobian → batch_invert → affine → hash160
        // Use rayon for parallel scalar multiplication
        let jacobians: Vec<(Fe, Fe, Fe)> = pool.install(|| {
            (0..n)
                .into_par_iter()
                .map(|i| {
                    let key = &key_buf[i * 32..(i + 1) * 32];
                    ec_table_scalar_mult(&ec_table, key)
                })
                .collect()
        });

        // Extract Z values for batch inversion
        let z_values: Vec<Fe> = jacobians.iter().map(|(_, _, z)| *z).collect();

        // Batch inversion (1 modular inverse + O(N) multiplications)
        let z_invs = randstorm::ec_table::batch_invert(&z_values);

        // Compute hash160 for each key
    let matches: Vec<(usize, String)> = pool.install(|| {
        (0..n)
            .into_par_iter()
            .enumerate()
            .filter_map(|(i, _)| {
                if found.load(Ordering::Relaxed) {
                    return None;
                }

                let (jx, jy, _) = &jacobians[i];
                let zi = &z_invs[i];

                let h160 = jacobian_to_h160(jx, jy, zi);
                if h160 == target_h160 {
                    let key_hex = key_buf[i * 32..(i + 1) * 32]
                        .iter()
                        .map(|b| format!("{:02x}", b))
                        .collect::<String>();
                    Some((i, key_hex))
                } else {
                    None
                }
            })
            .collect()
    });

        for (i, key_hex) in matches {
            let seed = current + i as u64;
            eprintln!("\n\nMATCH! Seed: {} Key: {}", seed, key_hex);

            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(&*output)
            {
                use std::io::Write;
                let _ = writeln!(f, "{},{},{}", seed, key_hex, &*target_arg);
            }
            eprintln!("Result saved to {}", &*output);

            *found_key.lock().unwrap() = Some(key_hex);
            found.store(true, Ordering::Relaxed);
        }

        pb.set_message(format!("{}", current));
        current += n as u64;
        pb.inc(n as u64);
    }

    pb.finish_with_message("Done");
    print_result(&found_key);
}

/// Table-based scalar multiplication on CPU using 32-bit limb precomputed EC table
/// Returns Jacobian coordinates (X, Y, Z) as 8-limb field elements
fn ec_table_scalar_mult(table: &EcTable, key: &[u8]) -> (Fe, Fe, Fe) {
    let mut qx = FE_ZERO;
    let mut qy = FE_ZERO;
    let mut qz = FE_ZERO;

    for &byte in key {
        if !fe_is_zero(&qz) {
            for _ in 0..8 {
                let (dx, dy, dz) = jacobian_double(&qx, &qy, &qz);
                qx = dx; qy = dy; qz = dz;
            }
        }

        if byte == 0 {
            continue;
        }

        let offset = byte as usize * NLIMBS;
        let tx: [u32; NLIMBS] = {
            let mut t = [0u32; NLIMBS];
            for l in 0..NLIMBS {
                t[l] = table.x[offset + NLIMBS - 1 - l];
            }
            t
        };
        let ty: [u32; NLIMBS] = {
            let mut t = [0u32; NLIMBS];
            for l in 0..NLIMBS {
                t[l] = table.y[offset + NLIMBS - 1 - l];
            }
            t
        };

        if fe_is_zero(&qz) {
            qx = tx;
            qy = ty;
            qz = FE_ONE;
        } else {
            let (rx, ry, rz) = jacobian_add_affine(&qx, &qy, &qz, &tx, &ty);
            qx = rx; qy = ry; qz = rz;
        }
    }

    (qx, qy, qz)
}

fn fe_is_zero(a: &Fe) -> bool {
    a.iter().all(|&x| x == 0)
}

/// Jacobian point doubling on CPU (32-bit limbs)
fn jacobian_double(x: &Fe, y: &Fe, z: &Fe) -> (Fe, Fe, Fe) {
    // Standard Jacobian doubling:
    // A = Y^2, B = 4*X*A, C = 8*A^2
    // M = 3*X^2 + a*Z^4 (a=0 for secp256k1)
    // X3 = M^2 - 2*B
    // Y3 = M*(B - X3) - C
    // Z3 = 2*Y*Z

    let a = fe_mul(y, y);
    let b = fe_mul(&fe_mul(&FE_FOUR, x), &a);
    let c = fe_mul(&fe_mul(&FE_EIGHT, &a), &a);
    let m = fe_mul(&FE_THREE, &fe_mul(x, x));

    let two_b = fe_add(&b, &b);
    let rx = fe_sub(&fe_mul(&m, &m), &two_b);
    let ry = fe_sub(&fe_mul(&m, &fe_sub(&b, &rx)), &c);
    let rz = fe_mul(&FE_TWO, &fe_mul(y, z));

    (rx, ry, rz)
}

/// Jacobian + Affine point addition on CPU (32-bit limbs)
/// T has Z=1 (from precomputed table)
fn jacobian_add_affine(jx: &Fe, jy: &Fe, jz: &Fe, tx: &Fe, ty: &Fe) -> (Fe, Fe, Fe) {
    // Standard Jacobian + Affine addition:
    // U2 = X2 * Z1^2
    // S2 = Y2 * Z1^3
    // H = U2 - X1
    // R = S2 - Y1
    // X3 = R^2 - H^3 - 2*X1*H^2
    // Y3 = R*(X1*H^2 - X3) - Y1*H^3
    // Z3 = Z1 * H

    let z1_sq = fe_mul(jz, jz);
    let z1_cu = fe_mul(&z1_sq, jz);

    let u2 = fe_mul(tx, &z1_sq);
    let s2 = fe_mul(ty, &z1_cu);

    let h = fe_sub(&u2, jx);
    let r = fe_sub(&s2, jy);

    let h_sq = fe_mul(&h, &h);
    let h_cu = fe_mul(&h_sq, &h);

    let x1_h_sq = fe_mul(jx, &h_sq);
    let two_x1_h_sq = fe_add(&x1_h_sq, &x1_h_sq);

    let rx = fe_sub(&fe_sub(&fe_mul(&r, &r), &h_cu), &two_x1_h_sq);
    let ry = fe_sub(&fe_mul(&r, &fe_sub(&x1_h_sq, &rx)), &fe_mul(jy, &h_cu));
    let rz = fe_mul(jz, &h);

    (rx, ry, rz)
}

/// Convert Jacobian + Z-inverse → affine → compressed pubkey → hash160
fn jacobian_to_h160(jx: &Fe, jy: &Fe, zi: &Fe) -> [u8; 20] {
    let zi2 = fe_mul(zi, zi);
    let zi3 = fe_mul(&zi2, zi);

    let x = fe_mul(jx, &zi2);
    let y = fe_mul(jy, &zi3);

    let mut pubkey = [0u8; 33];
    pubkey[0] = if y[0] & 1 != 0 { 0x03 } else { 0x02 };
    for b in 0..32 {
        let limb = (31 - b) / 4;
        let shift = ((31 - b) % 4) * 8;
        pubkey[1 + b] = (x[limb] >> shift) as u8;
    }

    let h160 = bitcoin::hashes::hash160::Hash::hash(&pubkey);
    *h160.as_byte_array()
}

fn print_result(found_key: &Arc<Mutex<Option<String>>>) {
    let guard = found_key.lock().unwrap();
    if let Some(key) = guard.as_ref() {
        println!("Private Key: {}", key);
    } else {
        println!("No match found.");
    }
}
