//! CPU-side: EC table precomputation + batch Jacobian→Affine inversion

use bitcoin::PublicKey;
use bitcoin::secp256k1::{Secp256k1, SecretKey};

pub const NLIMBS: usize = 8;

pub struct EcTable {
    pub x: [u32; 256 * NLIMBS],
    pub y: [u32; 256 * NLIMBS],
}

impl EcTable {
    pub fn precompute() -> Self {
        let secp = Secp256k1::new();
        let mut table = EcTable {
            x: [0u32; 256 * NLIMBS],
            y: [0u32; 256 * NLIMBS],
        };

        for idx in 0u16..256 {
            let mut scalar_bytes = [0u8; 32];
            scalar_bytes[30] = (idx >> 8) as u8;
            scalar_bytes[31] = idx as u8;

            if let Ok(sk) = SecretKey::from_slice(&scalar_bytes) {
                let spk = bitcoin::secp256k1::PublicKey::from_secret_key(&secp, &sk);
                let pk = PublicKey::new(spk);
                let serialized = pk.inner.serialize_uncompressed();
                for limb in 0..NLIMBS {
                    let off = 1 + limb * 4;
                    table.x[idx as usize * NLIMBS + limb] = u32::from_be_bytes([
                        serialized[off], serialized[off+1], serialized[off+2], serialized[off+3],
                    ]);
                    let off = 33 + limb * 4;
                    table.y[idx as usize * NLIMBS + limb] = u32::from_be_bytes([
                        serialized[off], serialized[off+1], serialized[off+2], serialized[off+3],
                    ]);
                }
            }
        }

        table
    }

}

/// Fe (field element) = 8 × u32
type Fe = [u32; NLIMBS];

const P32: Fe = [
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF,
];

fn fe_is_zero(a: &Fe) -> bool {
    a.iter().all(|&x| x == 0)
}

fn fe_cmp(a: &Fe, b: &Fe) -> std::cmp::Ordering {
    for i in (0..NLIMBS).rev() {
        if a[i] > b[i] { return std::cmp::Ordering::Greater; }
        if a[i] < b[i] { return std::cmp::Ordering::Less; }
    }
    std::cmp::Ordering::Equal
}

fn fe_sub(a: &Fe, b: &Fe) -> Fe {
    let mut r = [0u32; NLIMBS];
    let mut borrow: i64 = 0;
    for i in 0..NLIMBS {
        borrow += a[i] as i64 - b[i] as i64;
        r[i] = borrow as u32;
        borrow >>= 32;
    }
    if borrow < 0 {
        let mut carry: u64 = 0;
        for i in 0..NLIMBS {
            carry += r[i] as u64 + P32[i] as u64;
            r[i] = carry as u32;
            carry >>= 32;
        }
    }
    r
}

fn fe_mul(a: &Fe, b: &Fe) -> Fe {
    let mut d = [0u64; 16];
    for i in 0..NLIMBS {
        for j in 0..NLIMBS {
            let prod = a[i] as u64 * b[j] as u64;
            d[i + j] += prod;
            let mut carry = d[i + j] >> 32;
            d[i + j] &= 0xFFFFFFFF;
            let mut k = i + j + 1;
            while carry > 0 && k < 16 {
                d[k] += carry;
                carry = d[k] >> 32;
                d[k] &= 0xFFFFFFFF;
                k += 1;
            }
        }
    }

    let mut t = [0u64; 9];
    for i in 0..8 { t[i] = d[i]; }

    for i in 0..8 {
        let val = d[8 + i];
        if val == 0 { continue; }
        let vlo = (val & 0xFFFFFFFF) << 32;
        let vhi = val >> 32;
        let p977 = val * 977;

        t[i] += vlo; if t[i] >> 32 > 0 { t[i + 1] += t[i] >> 32; t[i] &= 0xFFFFFFFF; }
        if i + 1 < 9 { t[i + 1] += vhi; if t[i + 1] >> 32 > 0 && i + 2 < 9 { t[i + 2] += t[i + 1] >> 32; t[i + 1] &= 0xFFFFFFFF; } }
        t[i] += p977;
        let mut carry = t[i] >> 32; t[i] &= 0xFFFFFFFF;
        let mut k = i + 1;
        while carry > 0 && k < 9 { t[k] += carry; carry = t[k] >> 32; t[k] &= 0xFFFFFFFF; k += 1; }
    }

    if t[8] > 0 {
        let val = t[8]; t[8] = 0;
        let vlo = (val & 0xFFFFFFFF) << 32;
        let vhi = val >> 32;
        let p977 = val * 977;
        t[0] += vlo; if t[0] >> 32 > 0 { t[1] += t[0] >> 32; t[0] &= 0xFFFFFFFF; }
        t[1] += vhi; if t[1] >> 32 > 0 { t[2] += t[1] >> 32; t[1] &= 0xFFFFFFFF; }
        t[0] += p977;
        let mut carry = t[0] >> 32; t[0] &= 0xFFFFFFFF;
        let mut k = 1;
        while carry > 0 && k < 9 { t[k] += carry; carry = t[k] >> 32; t[k] &= 0xFFFFFFFF; k += 1; }
    }

    let mut r: Fe = [0u32; NLIMBS];
    for i in 0..NLIMBS { r[i] = t[i] as u32; }

    if fe_cmp(&r, &P32) != std::cmp::Ordering::Less {
        let mut borrow: i64 = 0;
        for i in 0..NLIMBS {
            borrow += r[i] as i64 - P32[i] as i64;
            r[i] = borrow as u32;
            borrow >>= 32;
        }
    }
    r
}

fn fe_inv(a: &Fe) -> Fe {
    let mut u = *a;
    let mut v = P32;
    let mut x1 = [0u32; NLIMBS]; x1[0] = 1;
    let mut x2 = [0u32; NLIMBS];
    for _ in 0..600 {
        while fe_cmp(&u, &v) != std::cmp::Ordering::Less && !fe_is_zero(&v) {
            u = fe_sub(&u, &v);
            x1 = fe_sub(&x1, &x2);
        }
        if fe_is_zero(&v) { break; }
        std::mem::swap(&mut u, &mut v);
        std::mem::swap(&mut x1, &mut x2);
    }
    x1
}

/// Batch inversion: given z[0..N], compute 1/z[i] for all i
/// Uses single modular inverse + O(N) multiplications
pub fn batch_invert(z_values: &[Fe]) -> Vec<Fe> {
    let n = z_values.len();
    if n == 0 { return vec![]; }

    // Forward: cumulative products
    let mut cumprod = vec![[0u32; NLIMBS]; n];
    cumprod[0] = z_values[0];
    for i in 1..n {
        cumprod[i] = fe_mul(&cumprod[i - 1], &z_values[i]);
    }

    // Single modular inverse
    let inv_n = fe_inv(&cumprod[n - 1]);

    // Backward: extract individual inverses
    let mut invs = vec![[0u32; NLIMBS]; n];
    let mut running = inv_n;
    for i in (1..n).rev() {
        invs[i] = fe_mul(&running, &cumprod[i - 1]);
        running = fe_mul(&running, &z_values[i]);
    }
    invs[0] = running;

    invs
}
