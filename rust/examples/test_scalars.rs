use bitcoin::secp256k1::{PublicKey, Secp256k1, SecretKey};

fn main() {
    let secp = Secp256k1::new();

    // Test: double the generator point (scalar 2 = 1+1)
    let mut s1 = [0u8; 32]; s1[31] = 1;
    let sk1 = SecretKey::from_slice(&s1).unwrap();
    let pk1 = PublicKey::from_secret_key(&secp, &sk1);
    println!("G (scalar 1): {}", hex::encode(pk1.serialize()));

    // Scalar 2 = 2*G
    let mut s2 = [0u8; 32]; s2[31] = 2;
    let sk2 = SecretKey::from_slice(&s2).unwrap();
    let pk2 = PublicKey::from_secret_key(&secp, &sk2);
    println!("2G (scalar 2): {}", hex::encode(pk2.serialize()));

    // Scalar 3 = 3*G
    let mut s3 = [0u8; 32]; s3[31] = 3;
    let sk3 = SecretKey::from_slice(&s3).unwrap();
    let pk3 = PublicKey::from_secret_key(&secp, &sk3);
    println!("3G (scalar 3): {}", hex::encode(pk3.serialize()));

    // Scalar 4 = 4*G
    let mut s4 = [0u8; 32]; s4[31] = 4;
    let sk4 = SecretKey::from_slice(&s4).unwrap();
    let pk4 = PublicKey::from_secret_key(&secp, &sk4);
    println!("4G (scalar 4): {}", hex::encode(pk4.serialize()));

    // Scalar 255
    let mut s255 = [0u8; 32]; s255[31] = 255;
    let sk255 = SecretKey::from_slice(&s255).unwrap();
    let pk255 = PublicKey::from_secret_key(&secp, &sk255);
    println!("255G: {}", hex::encode(pk255.serialize()));

    // Test ec_table with these scalars
    let ec_table = randstorm::EcTable::precompute();

    // Scalar 2 via ec_table: key = [0,0,...,0,2]
    let key2 = &mut [0u8; 32]; key2[31] = 2;
    let (rx2, ry2, rz2) = randstorm::ec_table_scalar_mult_test(&ec_table, key2);
    
    fn fe_to_bytes(f: &[u32; 8]) -> [u8; 32] {
        let mut b = [0u8; 32];
        for i in 0..32 {
            let limb = (31 - i) / 4;
            let shift = ((31 - i) % 4) * 8;
            b[i] = (f[limb] >> shift) as u8;
        }
        b
    }

    let z_invs2 = randstorm::batch_invert_test(&[rz2]);
    let zi2 = &z_invs2[0];
    let x2_aff = randstorm::fe_mul(&rx2, &randstorm::fe_mul(zi2, zi2));
    let x2_bytes = fe_to_bytes(&x2_aff);
    let y2_odd = randstorm::fe_mul(&ry2, &randstorm::fe_mul(zi2, &randstorm::fe_mul(zi2, zi2)))[0] & 1;
    let prefix2 = if y2_odd != 0 { "03" } else { "02" };
    println!();
    println!("ec_table scalar 2: {}{}", prefix2, hex::encode(x2_bytes));
    println!("reference scalar 2: {}", hex::encode(pk2.serialize()));

    // Scalar 3
    let key3 = &mut [0u8; 32]; key3[31] = 3;
    let (rx3, ry3, rz3) = randstorm::ec_table_scalar_mult_test(&ec_table, key3);
    let z_invs3 = randstorm::batch_invert_test(&[rz3]);
    let zi3 = &z_invs3[0];
    let x3_aff = randstorm::fe_mul(&rx3, &randstorm::fe_mul(zi3, zi3));
    let x3_bytes = fe_to_bytes(&x3_aff);
    let y3_odd = randstorm::fe_mul(&ry3, &randstorm::fe_mul(zi3, &randstorm::fe_mul(zi3, zi3)))[0] & 1;
    let prefix3 = if y3_odd != 0 { "03" } else { "02" };
    println!("ec_table scalar 3: {}{}", prefix3, hex::encode(x3_bytes));
    println!("reference scalar 3: {}", hex::encode(pk3.serialize()));

    // Scalar 255
    let key255 = &mut [0u8; 32]; key255[31] = 255;
    let (rx255, ry255, rz255) = randstorm::ec_table_scalar_mult_test(&ec_table, key255);
    let z_invs255 = randstorm::batch_invert_test(&[rz255]);
    let zi255 = &z_invs255[0];
    let x255_aff = randstorm::fe_mul(&rx255, &randstorm::fe_mul(zi255, zi255));
    let x255_bytes = fe_to_bytes(&x255_aff);
    let y255_odd = randstorm::fe_mul(&ry255, &randstorm::fe_mul(zi255, &randstorm::fe_mul(zi255, zi255)))[0] & 1;
    let prefix255 = if y255_odd != 0 { "03" } else { "02" };
    println!("ec_table scalar 255: {}{}", prefix255, hex::encode(x255_bytes));
    println!("reference scalar 255: {}", hex::encode(pk255.serialize()));

    // Test 2-byte scalar: 256 = 0x0100
    let mut s256 = [0u8; 32]; s256[30] = 1; // byte 30 = 1, byte 31 = 0 → scalar = 256
    let sk256 = SecretKey::from_slice(&s256).unwrap();
    let pk256 = PublicKey::from_secret_key(&secp, &sk256);
    println!();
    println!("256G reference: {}", hex::encode(pk256.serialize()));

    let (rx256, ry256, rz256) = randstorm::ec_table_scalar_mult_test(&ec_table, &s256);
    let z_invs256 = randstorm::batch_invert_test(&[rz256]);
    let zi256 = &z_invs256[0];
    let x256_aff = randstorm::fe_mul(&rx256, &randstorm::fe_mul(zi256, zi256));
    let x256_bytes = fe_to_bytes(&x256_aff);
    let y256_odd = randstorm::fe_mul(&ry256, &randstorm::fe_mul(zi256, &randstorm::fe_mul(zi256, zi256)))[0] & 1;
    let prefix256 = if y256_odd != 0 { "03" } else { "02" };
    println!("256G ec_table:  {}{}", prefix256, hex::encode(x256_bytes));
    println!("Match: {}", hex::encode(x256_bytes) == hex::encode(&pk256.serialize()[1..]));

    // Test full vulnerable key
    let key_hex = "421874e8761bddd15c23a28c02045d9039e986234a966c763ef720bc4200d395";
    let key_bytes = hex::decode(key_hex).unwrap();
    let sk_full = SecretKey::from_slice(&key_bytes).unwrap();
    let pk_full = PublicKey::from_secret_key(&secp, &sk_full);
    println!();
    println!("Full key reference: {}", hex::encode(pk_full.serialize()));

    let key_arr: [u8; 32] = key_bytes.try_into().unwrap();
    let (rxf, ryf, rzf) = randstorm::ec_table_scalar_mult_test(&ec_table, &key_arr);
    let z_invsf = randstorm::batch_invert_test(&[rzf]);
    let zif = &z_invsf[0];
    let xf_aff = randstorm::fe_mul(&rxf, &randstorm::fe_mul(zif, zif));
    let xf_bytes = fe_to_bytes(&xf_aff);
    let yf_odd = randstorm::fe_mul(&ryf, &randstorm::fe_mul(zif, &randstorm::fe_mul(zif, zif)))[0] & 1;
    let prefixf = if yf_odd != 0 { "03" } else { "02" };
    println!("Full key ec_table:  {}{}", prefixf, hex::encode(xf_bytes));
    println!("Match: {}", hex::encode(xf_bytes) == hex::encode(&pk_full.serialize()[1..]));
}
