use bitcoin::secp256k1::{PublicKey, Secp256k1, SecretKey};
use bitcoin::PublicKey as BtcPubKey;
use bitcoin::hashes::{hash160, Hash};

fn main() {
    let secp = Secp256k1::new();

    // Test: scalar 1 = generator point
    let mut scalar_bytes = [0u8; 32];
    scalar_bytes[31] = 1;
    let sk = SecretKey::from_slice(&scalar_bytes).unwrap();
    let pk = PublicKey::from_secret_key(&secp, &sk);
    let compressed = pk.serialize();
    println!("Scalar 1 compressed: {}", hex::encode(compressed));

    // Test: scalar 2
    scalar_bytes = [0u8; 32];
    scalar_bytes[31] = 2;
    let sk2 = SecretKey::from_slice(&scalar_bytes).unwrap();
    let pk2 = PublicKey::from_secret_key(&secp, &sk2);
    println!("Scalar 2 compressed: {}", hex::encode(pk2.serialize()));

    // Test: scalar 255
    scalar_bytes = [0u8; 32];
    scalar_bytes[31] = 255;
    let sk255 = SecretKey::from_slice(&scalar_bytes).unwrap();
    let pk255 = PublicKey::from_secret_key(&secp, &sk255);
    println!("Scalar 255 compressed: {}", hex::encode(pk255.serialize()));

    // Full key test
    let test_key = "421874e8761bddd15c23a28c02045d9039e986234a966c763ef720bc4200d395";
    let key_bytes = hex::decode(test_key).unwrap();
    let sk_full = SecretKey::from_slice(&key_bytes).unwrap();
    let pk_full = PublicKey::from_secret_key(&secp, &sk_full);
    let btc_pk = BtcPubKey::new(pk_full);
    let h160 = hash160::Hash::hash(&pk_full.serialize());
    println!("Full key hash160: {}", hex::encode(h160.as_byte_array()));
    println!("Full key compressed: {}", hex::encode(pk_full.serialize()));
}
