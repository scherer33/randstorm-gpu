pub mod prng;
pub mod ec_table;

pub use ec_table::{EcTable, Fe, NLIMBS, fe_add, fe_mul, fe_sub, fe_inv, batch_invert};
use bitcoin::hashes::Hash;

pub fn ec_table_scalar_mult_test(table: &EcTable, key: &[u8]) -> (Fe, Fe, Fe) {
    const FE_ZERO: Fe = [0u32; 8];
    const FE_ONE: Fe = {
        let mut a = [0u32; 8];
        a[0] = 1;
        a
    };

    let mut rx = FE_ZERO;
    let mut ry = FE_ZERO;
    let mut rz = FE_ZERO;

    for &byte in key {
        if !fe_is_zero_test(&rz) {
            for _ in 0..8 {
                let (dx, dy, dz) = jacobian_double(&rx, &ry, &rz);
                rx = dx; ry = dy; rz = dz;
            }
        }

        if byte == 0 {
            continue;
        }

        let offset = byte as usize * 8;
        let mut tx = [0u32; 8];
        let mut ty = [0u32; 8];
        for i in 0..8 {
            tx[i] = table.x[offset + 8 - 1 - i];
            ty[i] = table.y[offset + 8 - 1 - i];
        }

        if fe_is_zero_test(&rz) {
            rx = tx;
            ry = ty;
            rz = FE_ONE;
        } else {
            let (ax, ay, az) = jacobian_add_affine(&rx, &ry, &rz, &tx, &ty);
            rx = ax; ry = ay; rz = az;
        }
    }

    (rx, ry, rz)
}

fn fe_is_zero_test(a: &Fe) -> bool {
    a.iter().all(|&x| x == 0)
}

const FE_TWO: Fe = { let mut a = [0u32; 8]; a[0] = 2; a };
const FE_THREE: Fe = { let mut a = [0u32; 8]; a[0] = 3; a };
const FE_FOUR: Fe = { let mut a = [0u32; 8]; a[0] = 4; a };
const FE_EIGHT: Fe = { let mut a = [0u32; 8]; a[0] = 8; a };

fn jacobian_double(x: &Fe, y: &Fe, z: &Fe) -> (Fe, Fe, Fe) {
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

fn jacobian_add_affine(jx: &Fe, jy: &Fe, jz: &Fe, tx: &Fe, ty: &Fe) -> (Fe, Fe, Fe) {
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

pub fn batch_invert_test(z_values: &[Fe]) -> Vec<Fe> {
    batch_invert(z_values)
}

pub fn jacobian_to_h160_test(jx: &Fe, jy: &Fe, zi: &Fe) -> [u8; 20] {
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
