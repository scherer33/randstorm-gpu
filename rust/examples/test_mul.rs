use randstorm::{fe_mul, Fe};

fn fe_to_bytes(f: &[u32; 8]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for i in 0..32 {
        let limb = (31 - i) / 4;
        let shift = ((31 - i) % 4) * 8;
        b[i] = (f[limb] >> shift) as u8;
    }
    b
}

fn fe_hex(f: &[u32; 8]) -> String {
    hex::encode(fe_to_bytes(f))
}

fn main() {
    // Test: compute 2^256 mod p using fe_mul repeatedly
    // 2^256 mod p should equal 977
    let p: [u64; 4] = [
        0xFFFFFFFEFFFFFC2F,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
        0xFFFFFFFFFFFFFFFF,
    ];
    
    let two: Fe = { let mut a = [0u32; 8]; a[0] = 2; a };
    
    // Compute 2^2, 2^4, 2^8, ... 2^256 by squaring
    let mut val = two;
    for i in 1..=256 {
        val = fe_mul(&val, &val);
        if i <= 10 || i >= 250 {
            println!("2^{} mod p = {}", 1 << i, fe_hex(&val));
        } else if i == 11 {
            println!("... skipping ...");
        }
    }
    
    println!();
    println!("2^256 mod p = {}", fe_hex(&val));
    println!("Expected    = 00000000000000000000000000000000000000000000000000000000000003d1");
    
    // Also test: compute 2^8 = 256 by repeated squaring
    let mut v = two;
    for _ in 0..7 {
        v = fe_mul(&v, &v);
    }
    println!();
    println!("2^8 = {} (expected 100)", fe_hex(&v));
    
    // Simple chain: a*b*c where a=3, b=5, c=7 = 105
    let three: Fe = { let mut a = [0u32; 8]; a[0] = 3; a };
    let five: Fe = { let mut a = [0u32; 8]; a[0] = 5; a };
    let seven: Fe = { let mut a = [0u32; 8]; a[0] = 7; a };
    let r = fe_mul(&fe_mul(&three, &five), &seven);
    println!("3*5*7 mod p = {} (expected 105)", u64::from(r[0]) | (u64::from(r[1]) << 32));
}
