use bitcoin::address::Address;
use bitcoin::hashes::{hash160, Hash};
use bitcoin::secp256k1::{PublicKey, Secp256k1, SecretKey};
use bitcoin::PublicKey as BtcPubKey;

struct XorShift128Plus { s0: u64, s1: u64 }
impl XorShift128Plus {
    fn new(seed: u64) -> Self { let mut s = Self{s0:0,s1:0}; s.seed(seed); s }
    fn seed(&mut self, seed: u64) {
        self.s0 = Self::splitmix(seed); self.s1 = Self::splitmix(self.s0);
        for _ in 0..10 { self.next_raw(); }
    }
    fn next_raw(&mut self) -> u64 {
        let mut s1 = self.s0; let s0 = self.s1; self.s0 = s0;
        s1 ^= s1 << 23; s1 ^= s0; s1 ^= s1 >> 17; s1 ^= s0 >> 26; self.s1 = s1;
        s1.wrapping_add(s0)
    }
    fn next_u16(&mut self) -> u16 { (self.next_raw() >> 37) as u16 }
    fn splitmix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
}

struct ArcFour { i: u8, j: u8, s: [u8; 256] }
impl ArcFour {
    fn new(key: &[u8]) -> Self {
        let mut s = [0u8; 256];
        for (i, v) in s.iter_mut().enumerate() { *v = i as u8; }
        let mut j = 0u8;
        for i in 0..256u16 {
            j = j.wrapping_add(s[i as usize]).wrapping_add(key[i as usize % key.len()]);
            s.swap(i as usize, j as usize);
        }
        ArcFour { i: 0, j: 0, s }
    }
    fn next(&mut self) -> u8 {
        self.i = self.i.wrapping_add(1);
        self.j = self.j.wrapping_add(self.s[self.i as usize]);
        self.s.swap(self.i as usize, self.j as usize);
        let idx = self.s[self.i as usize].wrapping_add(self.s[self.j as usize]);
        self.s[idx as usize]
    }
}

struct SecureRandom { arc4: ArcFour }
impl SecureRandom {
    fn from_pool(pool: &[u8; 256], seed_time: u64) -> Self {
        let mut p = *pool;
        p[0] ^= seed_time as u8; p[1] ^= (seed_time>>8) as u8;
        p[2] ^= (seed_time>>16) as u8; p[3] ^= (seed_time>>24) as u8;
        SecureRandom { arc4: ArcFour::new(&p) }
    }
    fn get_bytes(&mut self, buf: &mut [u8]) {
        for b in buf.iter_mut() { *b = self.arc4.next(); }
    }
}

fn fe_to_hex(f: &[u32; 8]) -> String {
    f.iter().map(|l| format!("{:08x}", l)).collect::<Vec<_>>().join("")
}

fn main() {
    let test_seed: u64 = 1389000000000;

    let mut prng_x = XorShift128Plus::new(0xDEADBEEF);
    let mut pool = [0u8; 256];
    let mut p = 0;
    while p < 256 {
        let t = prng_x.next_u16();
        pool[p] = (t >> 8) as u8; p += 1;
        pool[p] = t as u8; p += 1;
    }

    let mut sr = SecureRandom::from_pool(&pool, test_seed);
    let mut key_bytes = [0u8; 32];
    sr.get_bytes(&mut key_bytes);
    let key_hex: String = key_bytes.iter().map(|b| format!("{:02x}", b)).collect();

    // Reference
    let secp = Secp256k1::new();
    let sk = SecretKey::from_slice(&key_bytes).unwrap();
    let pk = PublicKey::from_secret_key(&secp, &sk);
    let compressed = pk.serialize();
    let h160_ref = hash160::Hash::hash(&compressed);
    let btc_pk = BtcPubKey::new(pk);
    let addr = Address::p2pkh(&btc_pk, bitcoin::Network::Bitcoin);

    println!("=== libsecp256k1 (reference) ===");
    println!("Privatekey: {}", key_hex);
    println!("Compressed: {}", hex::encode(compressed));
    println!("Hash160:    {}", hex::encode(h160_ref.as_byte_array()));
    println!("Address:    {}", addr);

    // EC table
    let ec_table = randstorm::EcTable::precompute();

    // Verify table entry for scalar=1
    let mut table_x_bytes = [0u8; 32];
    let mut table_y_bytes = [0u8; 32];
    for l in 0..8 {
        let x_val = ec_table.x[1 * 8 + l];
        let y_val = ec_table.y[1 * 8 + l];
        table_x_bytes[l*4..l*4+4].copy_from_slice(&x_val.to_be_bytes());
        table_y_bytes[l*4..l*4+4].copy_from_slice(&y_val.to_be_bytes());
    }
    let gen_x = "79be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    println!();
    println!("=== Table entry scalar=1 ===");
    println!("Table X: {}", hex::encode(table_x_bytes));
    println!("Ref gen: {}", gen_x);
    println!("X match: {}", hex::encode(table_x_bytes) == gen_x);

    // Scalar=1 test: Z should be 1, so affine = jacobian
    let mut scalar1 = [0u8; 32];
    scalar1[31] = 1;
    let (rx1, ry1, rz1) = randstorm::ec_table_scalar_mult_test(&ec_table, &scalar1);
    println!();
    println!("=== Scalar=1 ec_table mult ===");
    println!("Jacobian X (LE limbs): {}", fe_to_hex(&rx1));
    println!("Jacobian Y (LE limbs): {}", fe_to_hex(&ry1));
    println!("Jacobian Z (LE limbs): {}", fe_to_hex(&rz1));

    // Convert X from LE limbs to bytes
    let mut x_bytes = [0u8; 32];
    for b in 0..32 {
        let limb = (31 - b) / 4;
        let shift = ((31 - b) % 4) * 8;
        x_bytes[b] = (rx1[limb] >> shift) as u8;
    }
    let mut y_bytes = [0u8; 32];
    for b in 0..32 {
        let limb = (31 - b) / 4;
        let shift = ((31 - b) % 4) * 8;
        y_bytes[b] = (ry1[limb] >> shift) as u8;
    }
    let prefix_str = if ry1[0] & 1 != 0 { "03" } else { "02" };
    let gen_compressed_ref = "0279be667ef9dcbbac55a06295ce870b07029bfcdb2dce28d959f2815b16f81798";
    println!("Computed: {}{}", prefix_str, hex::encode(x_bytes));
    println!("Reference: {}", gen_compressed_ref);
    println!("X bytes match:    {}", hex::encode(x_bytes) == &gen_compressed_ref[2..]);

    // Full key test with batch_invert
    let (jx, jy, jz) = randstorm::ec_table_scalar_mult_test(&ec_table, &key_bytes);
    println!();
    println!("=== Full key ec_table ===");
    println!("Z is zero: {}", jz.iter().all(|&x| x == 0));

    if !jz.iter().all(|&x| x == 0) {
        let z_invs = randstorm::batch_invert_test(&[jz]);
        let zi = &z_invs[0];
        let h160_custom = randstorm::jacobian_to_h160_test(&jx, &jy, zi);
        println!("Hash160:    {}", hex::encode(h160_custom));
        println!("Expected:   {}", hex::encode(h160_ref.as_byte_array()));
        println!("Match:      {}", h160_ref.as_byte_array() == &h160_custom);
    }
}
