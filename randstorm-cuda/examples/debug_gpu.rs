use bitcoin::secp256k1::{Secp256k1, SecretKey, PublicKey};
use bitcoin::hashes::{hash160, Hash};
use randstorm_cuda::ec_table::{EcTable, batch_invert, NLIMBS};

type Fe = [u32; NLIMBS];

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

fn fe_mul(a: &Fe, b: &Fe) -> Fe {
    let p32: Fe = [0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF];
    let mut d = [0u64; 16];
    for i in 0..8 {
        for j in 0..8 {
            let prod = a[i] as u64 * b[j] as u64;
            d[i+j] += prod;
            let mut carry = d[i+j] >> 32;
            d[i+j] &= 0xFFFFFFFF;
            let mut k = i+j+1;
            while carry > 0 && k < 16 { d[k] += carry; carry = d[k] >> 32; d[k] &= 0xFFFFFFFF; k += 1; }
        }
    }
    let mut t = [0u64; 9];
    for i in 0..8 { t[i] = d[i]; }
    for i in 0..8 {
        let val = d[8+i];
        if val == 0 { continue; }
        let vlo = (val & 0xFFFFFFFF) << 32;
        let vhi = val >> 32;
        let p977 = val * 977;
        t[i] += vlo; if t[i]>>32 > 0 { t[i+1] += t[i]>>32; t[i] &= 0xFFFFFFFF; }
        if i+1 < 9 { t[i+1] += vhi; if t[i+1]>>32>0 && i+2<9 { t[i+2] += t[i+1]>>32; t[i+1] &= 0xFFFFFFFF; } }
        t[i] += p977;
        let mut carry = t[i] >> 32; t[i] &= 0xFFFFFFFF;
        let mut k = i+1;
        while carry > 0 && k < 9 { t[k] += carry; carry = t[k]>>32; t[k] &= 0xFFFFFFFF; k += 1; }
    }
    if t[8] > 0 {
        let val = t[8]; t[8] = 0;
        let vlo = (val & 0xFFFFFFFF) << 32; let vhi = val >> 32; let p977 = val * 977;
        t[0] += vlo; if t[0]>>32>0 { t[1]+=t[0]>>32; t[0]&=0xFFFFFFFF; }
        t[1] += vhi; if t[1]>>32>0 { t[2]+=t[1]>>32; t[1]&=0xFFFFFFFF; }
        t[0] += p977;
        let mut carry = t[0]>>32; t[0]&=0xFFFFFFFF;
        for k in 1..9 { if carry==0 {break;} t[k] += carry; carry = t[k]>>32; t[k]&=0xFFFFFFFF; }
    }
    let mut r: Fe = [0u32; 8];
    for i in 0..8 { r[i] = t[i] as u32; }
    // mod p
    for i in (0..8).rev() { if r[i] < p32[i] { return r; } if r[i] > p32[i] { break; } }
    let mut borrow: i64 = 0;
    for i in 0..8 { borrow += r[i] as i64 - p32[i] as i64; r[i] = borrow as u32; borrow >>= 32; }
    r
}

fn main() {
    let seed = 1394413261000u64;
    let v8seed: u64 = 0xDEADBEEF;

    let base_pool = {
        let mut prng = XorShift128Plus::new(v8seed);
        let mut pool = [0u8; 256];
        let mut p = 0;
        while p < 256 {
            let t = prng.next_u16();
            pool[p] = (t>>8) as u8; p+=1;
            pool[p] = t as u8; p+=1;
        }
        pool
    };

    let mut key = [0u8; 32];
    SecureRandom::from_pool(&base_pool, seed).get_bytes(&mut key);

    // CPU reference
    let ref_secp = Secp256k1::new();
    let ref_sk = SecretKey::from_slice(&key).unwrap();
    let ref_pk = PublicKey::from_secret_key(&ref_secp, &ref_sk);
    let ref_serialized = ref_pk.serialize();
    let ref_h160 = hash160::Hash::hash(&ref_serialized);

    eprintln!("=== CPU Reference ===");
    eprintln!("Private key: {}", hex::encode(key));
    eprintln!("Pubkey (compressed): {}", hex::encode(ref_serialized));
    eprintln!("Hash160: {}", hex::encode(ref_h160.as_byte_array()));

    // Get uncompressed pubkey for X,Y verification
    let ref_uncompressed = ref_pk.serialize_uncompressed();
    let ref_x: Fe = {
        let mut x = [0u32; 8];
        for i in 0..8 { let off = 1+i*4; x[7-i] = u32::from_be_bytes([ref_uncompressed[off],ref_uncompressed[off+1],ref_uncompressed[off+2],ref_uncompressed[off+3]]); }
        x
    };
    let ref_y: Fe = {
        let mut y = [0u32; 8];
        for i in 0..8 { let off = 33+i*4; y[7-i] = u32::from_be_bytes([ref_uncompressed[off],ref_uncompressed[off+1],ref_uncompressed[off+2],ref_uncompressed[off+3]]); }
        y
    };
    eprintln!("Ref X (u32 LE): {:?}", ref_x);
    eprintln!("Ref Y (u32 LE): {:?}", ref_y);

    // GPU pipeline
    let ec_table = EcTable::precompute();
    let cuda = randstorm_cuda::cuda::CudaContext::try_init().unwrap();
    cuda.upload_ec_table(&ec_table);

    // Phase 1: privkey -> Jacobian
    let mut z_buf = vec![0u32; 8];
    cuda.pk_to_jacobian(&key, 1, &mut z_buf).unwrap();

    // Read back all Jacobian
    let mut jpx = vec![0u32; 8];
    let mut jpy = vec![0u32; 8];
    let mut jpz = vec![0u32; 8];
    cuda.read_jacobian(1, &mut jpx, &mut jpy, &mut jpz).unwrap();

    eprintln!("\n=== GPU Jacobian ===");
    eprintln!("JX: {:?}", &jpx[..]);
    eprintln!("JY: {:?}", &jpy[..]);
    eprintln!("JZ: {:?}", &jpz[..]);

    // Convert GPU Jacobian to affine on CPU using batch inversion
    let z_vals: &[Fe] = unsafe { std::slice::from_raw_parts(jpz.as_ptr() as *const Fe, 1) };
    let z_invs: Vec<Fe> = batch_invert(z_vals);
    let z_inv = z_invs[0];

    eprintln!("\n=== CPU-side Affine from GPU Jacobian ===");
    // Affine X = JX * Z^-2, Y = JY * Z^-3
    let zi2 = fe_mul(&z_inv, &z_inv);
    let zi3 = fe_mul(&zi2, &z_inv);
    let jx: Fe = [jpx[0],jpx[1],jpx[2],jpx[3],jpx[4],jpx[5],jpx[6],jpx[7]];
    let jy: Fe = [jpy[0],jpy[1],jpy[2],jpy[3],jpy[4],jpy[5],jpy[6],jpy[7]];
    let ax = fe_mul(&jx, &zi2);
    let ay = fe_mul(&jy, &zi3);

    eprintln!("Affine X: {:?}", ax);
    eprintln!("Affine Y: {:?}", ay);
    eprintln!("Ref    X: {:?}", ref_x);
    eprintln!("Ref    Y: {:?}", ref_y);
    eprintln!("X match: {}", ax == ref_x);
    eprintln!("Y match: {}", ay == ref_y);

    // GPU Phase 2
    let z_inv_flat: &[u32] = unsafe { std::slice::from_raw_parts(z_invs.as_ptr() as *const u32, 8) };
    let mut h160_res = vec![[0u8; 20]; 1];
    cuda.jacobian_to_hash160(z_inv_flat, 1, &mut h160_res).unwrap();
    eprintln!("\nGPU hash160: {}", hex::encode(h160_res[0]));
    eprintln!("CPU hash160: {}", hex::encode(ref_h160.as_byte_array()));
    eprintln!("Hash160 match: {}", h160_res[0] == *ref_h160.as_byte_array());
}
