use randstorm::{EcTable, fe_mul, fe_add, fe_sub, fe_inv};

fn fe_to_bytes(f: &[u32; 8]) -> [u8; 32] {
    let mut b = [0u8; 32];
    for i in 0..32 {
        let limb = (31 - i) / 4;
        let shift = ((31 - i) % 4) * 8;
        b[i] = (f[limb] >> shift) as u8;
    }
    b
}

fn fe_to_u64(f: &[u32; 8]) -> u64 {
    // Only for small values
    f[0] as u64 | ((f[1] as u64) << 32)
}

fn fe_hex(f: &[u32; 8]) -> String {
    hex::encode(fe_to_bytes(f))
}

const FE_TWO: [u32; 8] = { let mut a = [0u32; 8]; a[0] = 2; a };
const FE_THREE: [u32; 8] = { let mut a = [0u32; 8]; a[0] = 3; a };
const FE_FOUR: [u32; 8] = { let mut a = [0u32; 8]; a[0] = 4; a };
const FE_EIGHT: [u32; 8] = { let mut a = [0u32; 8]; a[0] = 8; a };

fn jacobian_double(x: &[u32; 8], y: &[u32; 8], z: &[u32; 8]) -> ([u32; 8], [u32; 8], [u32; 8]) {
    let a = fe_mul(y, y);      // A = Y^2
    let b = fe_mul(&fe_mul(&FE_FOUR, x), &a);  // B = 4*X*Y^2
    let c = fe_mul(&fe_mul(&FE_EIGHT, &a), &a); // C = 8*Y^4
    let m = fe_mul(&FE_THREE, &fe_mul(x, x));   // M = 3*X^2
    let two_b = fe_add(&b, &b);                  // 2*B
    let rx = fe_sub(&fe_mul(&m, &m), &two_b);   // X3 = M^2 - 2*B
    let ry = fe_sub(&fe_mul(&m, &fe_sub(&b, &rx)), &c); // Y3 = M*(B - X3) - C
    let rz = fe_mul(&FE_TWO, &fe_mul(y, z));     // Z3 = 2*Y*Z
    (rx, ry, rz)
}

fn main() {
    let ec_table = EcTable::precompute();

    // G point in LE limbs (from table entry scalar=1, reversed)
    let g_x: [u32; 8] = {
        let mut t = [0u32; 8];
        for i in 0..8 { t[i] = ec_table.x[1 * 8 + 8 - 1 - i]; }
        t
    };
    let g_y: [u32; 8] = {
        let mut t = [0u32; 8];
        for i in 0..8 { t[i] = ec_table.y[1 * 8 + 8 - 1 - i]; }
        t
    };
    let g_z: [u32; 8] = { let mut a = [0u32; 8]; a[0] = 1; a };

    println!("G_x = {}", fe_hex(&g_x));
    println!("G_y = {}", fe_hex(&g_y));

    // Step by step
    let a = fe_mul(&g_y, &g_y);
    println!("A = Y^2 = {}", fe_hex(&a));
    
    let b = fe_mul(&fe_mul(&FE_FOUR, &g_x), &a);
    println!("B = 4*X*A = {}", fe_hex(&b));
    
    let m = fe_mul(&FE_THREE, &fe_mul(&g_x, &g_x));
    println!("M = 3*X^2 = {}", fe_hex(&m));

    let m2 = fe_mul(&m, &m);
    println!("M^2 = {}", fe_hex(&m2));

    let two_b = fe_add(&b, &b);
    println!("2*B = {}", fe_hex(&two_b));

    let rx = fe_sub(&m2, &two_b);
    println!("X' = M^2 - 2*B = {}", fe_hex(&rx));
    println!("Expected X' = 7d152c041ea8e1dc2191843d1fa9db55b68f88fef695e2c791d40444b365afc2");

    let rz = fe_mul(&FE_TWO, &fe_mul(&g_y, &g_z));
    println!("Z' = {}", fe_hex(&rz));
    println!("Expected Z' = 9075b4ee4d4788cabb49f7f81c221151fa2f68914d0aa833388fa11ff621a970");

    // Convert to affine
    let zi = fe_inv(&rz);
    let zi2 = fe_mul(&zi, &zi);
    let x_aff = fe_mul(&rx, &zi2);
    println!("Affine X = {}", fe_hex(&x_aff));
    println!("Expected  = c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5");

    // Test simple multiplication: 3 * 5 mod p
    let three: [u32; 8] = { let mut a = [0u32; 8]; a[0] = 3; a };
    let five: [u32; 8] = { let mut a = [0u32; 8]; a[0] = 5; a };
    let result = fe_mul(&three, &five);
    println!();
    println!("3*5 mod p = {} (expected 15)", fe_to_u64(&result));

    // Test larger: 4 * Gx mod p
    let four_gx = fe_mul(&FE_FOUR, &g_x);
    let expected = "0xe6f999fbe772eeb156818a573a1c2c1c0a6ff36cb738a36567ca056d5be06231";
    println!("4*Gx mod p = {}", fe_hex(&four_gx));
    println!("Expected    = {}", expected);
    println!("Match: {}", fe_hex(&four_gx) == expected);
}
