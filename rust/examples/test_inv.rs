use randstorm::{fe_mul, fe_inv, Fe};

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

fn from_hex(s: &str) -> Fe {
    let bytes = hex::decode(s).unwrap();
    let mut f = [0u32; 8];
    for i in 0..8 {
        let off = (7 - i) * 4;
        f[i] = u32::from_be_bytes([bytes[off], bytes[off+1], bytes[off+2], bytes[off+3]]);
    }
    f
}

fn main() {
    // Test 1: inv(2) should be (p+1)/2
    let two: Fe = { let mut a = [0u32; 8]; a[0] = 2; a };
    let inv2 = fe_inv(&two);
    println!("inv(2) = {}", fe_hex(&inv2));
    
    // inv(2) * 2 should be 1
    let check = fe_mul(&two, &inv2);
    println!("inv(2) * 2 = {}", fe_hex(&check));
    println!("Should be 1: {}", check[0] == 1 && check[1..].iter().all(|&x| x == 0));

    // Test 2: inv(Z') for the doubling test
    let z = from_hex("9075b4ee4d4788cabb49f7f81c221151fa2f68914d0aa833388fa11ff621a970");
    let zi = fe_inv(&z);
    println!("inv(Z') = {}", fe_hex(&zi));
    println!("Expected = b7e31a064ed74d314de79011c5f0a46ac155602353dc3d340fbeaeec9767a6a6");

    let check_z = fe_mul(&z, &zi);
    println!("Z * inv(Z) = {}", fe_hex(&check_z));
    println!("Should be 1: {}", check_z[0] == 1 && check_z[1..].iter().all(|&x| x == 0));

    // Test 3: inv(5)
    let five: Fe = { let mut a = [0u32; 8]; a[0] = 5; a };
    let inv5 = fe_inv(&five);
    let check5 = fe_mul(&five, &inv5);
    println!();
    println!("inv(5) = {}", fe_hex(&inv5));
    println!("5 * inv(5) = {}", fe_hex(&check5));
    println!("Should be 1: {}", check5[0] == 1 && check5[1..].iter().all(|&x| x == 0));
}
