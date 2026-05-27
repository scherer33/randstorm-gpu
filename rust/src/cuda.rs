#[cfg(cuda_available)]
mod cuda_impl {
    use libloading::Library;

    #[repr(C)]
    struct CudaResult {
        code: i32,
    }

    type Secp256k1Hash160KernelFn =
        unsafe extern "C" fn(privkeys: *const u8, hash160_out: *mut u8, count: u32) -> CudaResult;

    type Secp256k1KernelFn =
        unsafe extern "C" fn(privkeys: *const u8, pubkeys: *mut u8, count: u32) -> CudaResult;

    pub struct CudaContext {
        _lib: Library,
        secp256k1_hash160_fn: Secp256k1Hash160KernelFn,
        secp256k1_fn: Secp256k1KernelFn,
    }

    impl CudaContext {
        pub fn try_init() -> Option<Self> {
            let so_paths = [
                std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|d| d.join("cuda_kernels.so"))),
                option_env!("OUT_DIR").map(|d| std::path::PathBuf::from(d).join("cuda_kernels.so")),
            ];

            let mut lib = None;
            for path in &so_paths {
                if let Some(p) = path {
                    if p.exists() {
                        eprintln!("[CUDA] Loading from: {}", p.display());
                        match unsafe { Library::new(p) } {
                            Ok(l) => { lib = Some(l); break; }
                            Err(e) => eprintln!("[CUDA] Failed to load {}: {}", p.display(), e),
                        }
                    }
                }
            }

            let lib = lib?;

            let secp256k1_hash160_fn: Secp256k1Hash160KernelFn = unsafe {
                match lib.get::<Secp256k1Hash160KernelFn>(b"randstorm_secp256k1_hash160_kernel\0") {
                    Ok(s) => {
                        let p: Secp256k1Hash160KernelFn = *s.into_raw();
                        p
                    }
                    Err(e) => {
                        eprintln!("[CUDA] randstorm_secp256k1_hash160_kernel not found: {e}");
                        return None;
                    }
                }
            };

            let secp256k1_fn: Secp256k1KernelFn = unsafe {
                match lib.get::<Secp256k1KernelFn>(b"randstorm_secp256k1_kernel\0") {
                    Ok(s) => {
                        let p: Secp256k1KernelFn = *s.into_raw();
                        p
                    }
                    Err(e) => {
                        eprintln!("[CUDA] randstorm_secp256k1_kernel not found: {e}");
                        return None;
                    }
                }
            };

            eprintln!("[CUDA] All kernels loaded successfully");
            Some(CudaContext { _lib: lib, secp256k1_hash160_fn, secp256k1_fn })
        }

        pub fn secp256k1_hash160_batch(
            &self, private_keys: &[u8], count: usize,
        ) -> Option<Vec<[u8; 20]>> {
            if count == 0 || private_keys.len() < count * 32 { return None; }
            let mut hash160s = vec![[0u8; 20]; count];
            unsafe {
                let result = (self.secp256k1_hash160_fn)(
                    private_keys.as_ptr(),
                    hash160s.as_mut_ptr() as *mut u8,
                    count as u32,
                );
                if result.code != 0 {
                    eprintln!("[CUDA] kernel failed: code {}", result.code);
                    return None;
                }
            }
            Some(hash160s)
        }

        pub fn batch_size(&self) -> usize { 16384 }
    }
}

#[cfg(cuda_available)]
pub use cuda_impl::CudaContext;

#[cfg(not(cuda_available))]
pub struct CudaContext;

#[cfg(not(cuda_available))]
impl CudaContext {
    pub fn try_init() -> Option<Self> { None }
}
