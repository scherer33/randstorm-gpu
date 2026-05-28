use randstorm_cuda::ec_table::{EcTable, batch_invert, NLIMBS};
type Fe = [u32; NLIMBS];

fn main() {
    // k = 1 (little endian, MSB)
    let mut key = [0u8; 32];
    key[31] = 1;   // k=1 in big-endian

    let ref_uncompressed = {
        use bitcoin::secp256k1::{Secp256k1, SecretKey, PublicKey};
        let s = Secp256k1::new();
        let sk = SecretKey::from_slice(&key).unwrap();
        let pk = PublicKey::from_secret_key(&s, &sk);
        pk.serialize_uncompressed()
    };
    let ref_x: Fe = { let mut x=[0u32;8]; for i in 0..8 { let o=1+i*4; x[7-i]=u32::from_be_bytes([ref_uncompressed[o],ref_uncompressed[o+1],ref_uncompressed[o+2],ref_uncompressed[o+3]]); } x };
    let ref_y: Fe = { let mut y=[0u32;8]; for i in 0..8 { let o=33+i*4; y[7-i]=u32::from_be_bytes([ref_uncompressed[o],ref_uncompressed[o+1],ref_uncompressed[o+2],ref_uncompressed[o+3]]); } y };

    eprintln!("k=1, expected pubkey X: {:?}", ref_x);
    eprintln!("k=1, expected pubkey Y: {:?}", ref_y);
    eprintln!("Generator X should be: [3052483029, 383564184, 2232051793, 47214946, 3466164231, 1436254869, 4193635244, 2042525214]");

    let ec = EcTable::precompute();
    // Check table[1]
    eprintln!("\ntable[1].x[0..]: {:?}", &ec.x[8..16]);
    eprintln!("Expected GX:      [3052483029, 383564184, 2232051793, 47214946, 3466164231, 1436254869, 4193635244, 2042525214]");

    let cuda = randstorm_cuda::cuda::CudaContext::try_init().unwrap();
    cuda.upload_ec_table(&ec);

    let mut z_buf = vec![0u32; 8];
    cuda.pk_to_jacobian(&key, 1, &mut z_buf).unwrap();

    let mut jpx = vec![0u32; 8];
    let mut jpy = vec![0u32; 8];
    let mut jpz = vec![0u32; 8];
    cuda.read_jacobian(1, &mut jpx, &mut jpy, &mut jpz).unwrap();

    eprintln!("\nGPU Jacobian for k=1:");
    eprintln!("JX: {:?}", &jpx[..]);
    eprintln!("JY: {:?}", &jpy[..]);
    eprintln!("JZ: {:?}", &jpz[..]);
    eprintln!("Z should be [1,0,0,0,0,0,0,0] (affine, Z=1)");

    // Convert to affine using batch inv
    let z_vals: &[Fe] = unsafe { std::slice::from_raw_parts(jpz.as_ptr() as *const Fe, 1) };
    let z_invs = batch_invert(z_vals);
    eprintln!("Z-inverse: {:?}", z_invs[0]);

    // Manually verify Z * Z_inv = 1
    let mut prod = [0u64; 16];
    for i in 0..8 { for j in 0..8 { prod[i+j] += jpz[i] as u64 * z_invs[0][j] as u64; } }
    eprintln!("Z * Z_inv result array (should be 1 + k*P): len={} entries", 16);
    for i in 0..16 { eprintln!("  lim{}: 0x{:016x}", i, prod[i]); }
}
