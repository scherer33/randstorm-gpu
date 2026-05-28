#[cfg(cuda_available)]
mod cuda_impl {
    use libloading::Library;

    #[repr(C)] pub struct CudaResult { pub code: i32 }

    type FnUpload = unsafe extern "C" fn(x: *const u32, y: *const u32) -> CudaResult;
    type FnInit   = unsafe extern "C" fn(max: u64) -> CudaResult;
    type FnFree   = unsafe extern "C" fn() -> CudaResult;
    type FnPK2JP  = unsafe extern "C" fn(keys: *const u8, z_out: *mut u32, count: u32) -> CudaResult;
    type FnJP2H160= unsafe extern "C" fn(z_inv: *const u32, h160: *mut u8, count: u32) -> CudaResult;
    type FnReadJP = unsafe extern "C" fn(x: *mut u32, y: *mut u32, z: *mut u32, count: u32) -> CudaResult;

    pub struct CudaContext {
        _lib: Library,
        _upload_fn: FnUpload,
        _free_fn: FnFree,
        pk_to_jp_fn: FnPK2JP,
        jp_to_h160_fn: FnJP2H160,
        read_jp_fn: FnReadJP,
    }

    impl CudaContext {
        pub fn try_init() -> Option<Self> {
            let so_paths = [
                std::env::current_exe().ok().and_then(|p| p.parent().map(|d| d.join("cuda_kernels.so"))),
                option_env!("OUT_DIR").map(|d| std::path::PathBuf::from(d).join("cuda_kernels.so")),
            ];
            let mut lib = None;
            for p in &so_paths {
                if let Some(p) = p {
                    if p.exists() {
                        match unsafe { Library::new(p) } { Ok(l) => { lib = Some(l); break; } _ => {} }
                    }
                }
            }
            let lib = lib?;

            let upload: FnUpload = unsafe { let s = lib.get::<FnUpload>(b"randstorm_upload_ec_table\0").ok()?; *s.into_raw() };
            let init: FnInit = unsafe { let s = lib.get::<FnInit>(b"randstorm_gpu_init\0").ok()?; *s.into_raw() };
            let free: FnFree = unsafe { let s = lib.get::<FnFree>(b"randstorm_gpu_free\0").ok()?; *s.into_raw() };
            let pk_jp: FnPK2JP = unsafe { let s = lib.get::<FnPK2JP>(b"randstorm_pk_to_jacobian\0").ok()?; *s.into_raw() };
            let jp_h160: FnJP2H160 = unsafe { let s = lib.get::<FnJP2H160>(b"randstorm_jacobian_to_hash160\0").ok()?; *s.into_raw() };
            let read_jp: FnReadJP = unsafe { let s = lib.get::<FnReadJP>(b"randstorm_read_jacobian\0").ok()?; *s.into_raw() };

            unsafe { init(32768); }
            eprintln!("[CUDA] GPU ready (32768 keys)");
            Some(CudaContext { _lib: lib, _upload_fn: upload, _free_fn: free, pk_to_jp_fn: pk_jp, jp_to_h160_fn: jp_h160, read_jp_fn: read_jp })
        }

        pub fn upload_ec_table(&self, t: &crate::ec_table::EcTable) -> bool {
            unsafe { let r = (self._upload_fn)(t.x.as_ptr(), t.y.as_ptr()); r.code == 0 }
        }

        pub fn pk_to_jacobian(&self, keys: &[u8], count: usize, z_out: &mut [u32]) -> Option<()> {
            if count == 0 { return Some(()); }
            if z_out.len() < count * 8 { return None; }
            unsafe {
                let r = (self.pk_to_jp_fn)(keys.as_ptr(), z_out.as_mut_ptr(), count as u32);
                if r.code != 0 { return None; }
            }
            Some(())
        }

        pub fn jacobian_to_hash160(&self, z_inv: &[u32], count: usize, h160_out: &mut [[u8; 20]]) -> Option<()> {
            if count == 0 { return Some(()); }
            if z_inv.len() < count * 8 { return None; }
            if h160_out.len() < count { return None; }
            unsafe {
                let r = (self.jp_to_h160_fn)(z_inv.as_ptr(), h160_out.as_mut_ptr() as *mut u8, count as u32);
                if r.code != 0 { return None; }
            }
            Some(())
        }

        pub fn read_jacobian(&self, count: usize, jpx: &mut [u32], jpy: &mut [u32], jpz: &mut [u32]) -> Option<()> {
            if count == 0 { return Some(()); }
            if jpx.len() < count * 8 || jpy.len() < count * 8 || jpz.len() < count * 8 { return None; }
            unsafe {
                let r = (self.read_jp_fn)(jpx.as_mut_ptr(), jpy.as_mut_ptr(), jpz.as_mut_ptr(), count as u32);
                if r.code != 0 { return None; }
            }
            Some(())
        }

    }

    impl Drop for CudaContext {
        fn drop(&mut self) { unsafe { (self._free_fn)(); } }
    }
}

#[cfg(cuda_available)]
pub use cuda_impl::CudaContext;

#[cfg(not(cuda_available))]
pub struct CudaContext;
#[cfg(not(cuda_available))]
impl CudaContext {
    pub fn try_init() -> Option<Self> { None }
    #[allow(unused)] pub fn upload_ec_table(&self, _: &crate::ec_table::EcTable) -> bool { false }
    #[allow(unused)] pub fn pk_to_jacobian(&self, _: &[u8], _: usize, _: &mut [u32]) -> Option<()> { None }
    #[allow(unused)] pub fn jacobian_to_hash160(&self, _: &[u32], _: usize, _: &mut [[u8; 20]]) -> Option<()> { None }
}
