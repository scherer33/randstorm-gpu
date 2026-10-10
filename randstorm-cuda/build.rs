use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let cuda_home = env::var("CUDA_HOME")
        .or_else(|_| env::var("CUDA_PATH"))
        .unwrap_or_else(|_| "/opt/cuda".to_string());

    let nvcc = Path::new(&cuda_home).join("bin/nvcc");
    if !nvcc.exists() {
        println!("cargo:warning=nvcc not found at {}, building CPU-only", nvcc.display());
        return;
    }

    let test_result = Command::new(&nvcc).arg("--version").output();
    if test_result.is_err() {
        println!("cargo:warning=nvcc test failed, building CPU-only");
        return;
    }

    println!("cargo:warning=CUDA found, compiling kernels with nvcc");

    let out_dir = env::var("OUT_DIR").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let cuda_include = Path::new(&cuda_home).join("include");
    let cuda_lib = Path::new(&cuda_home).join("lib64");

    // Compile the unified CUDA kernel file as a single compilation unit.
    // This avoids __constant__ memory conflicts between separate .cu files.
    // The host-side wrapper functions (extern "C") use <<<>>> syntax, which
    // is the correct way to launch __global__ kernels (raw function pointers don't work).
    let src = "kernel/cuda_kernels.cu";
    let obj = format!("{}/cuda_kernels.o", out_dir);

    let result = Command::new(&nvcc)
        .args(&[
            "-O3",
            "-arch=sm_75",
            "--generate-code=arch=compute_75,code=[sm_75,compute_75]",
            "-Xcompiler", "-fPIC,-fvisibility=default,-mcmodel=large,-fno-reorder-blocks-and-partition",
            &format!("-I{}", cuda_include.display()),
            &format!("-I{}/kernel", manifest_dir),
            "-c",
            src,
            "-o",
            &obj,
        ])
        .output()
        .expect("Failed to run nvcc");

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        println!("cargo:warning=CUDA kernel compilation failed: {}", stderr);
        println!("cargo:warning=Building CPU-only mode");
        return;
    }

    // Link into shared library
    let so_path = format!("{}/cuda_kernels.so", out_dir);

    // Export function names (host-side wrappers, not __global__ kernels)
    let export_kernels = [
        "randstorm_upload_ec_table",
        "randstorm_gpu_init",
        "randstorm_gpu_free",
        "randstorm_pk_to_jacobian",
        "randstorm_read_jacobian",
        "randstorm_jacobian_to_hash160",
    ];

    let version_script = format!("{}/kernel/export.map", manifest_dir);

    // Try linking with g++ first, fall back to nvcc
    let vs_arg = format!("-Wl,--version-script={}", version_script);
    let cuda_lib_str = cuda_lib.display().to_string();
    let link_args = vec![
        "-shared", "-o", &so_path,
        &obj,
        "-L", &cuda_lib_str,
        "-lcudart",
        &vs_arg,
        "-Wl,-z,notext",
    ];

    let link_result = Command::new("g++")
        .args(&link_args)
        .output();

    let so_exists = match link_result {
        Ok(r) if r.status.success() => true,
        _ => {
            // g++ failed, try without version-script
            let simple_args = vec![
                "-shared", "-o", &so_path,
                &obj,
                "-L", &cuda_lib_str,
                "-lcudart",
                "-Wl,-z,notext",
            ];
            let gpp2 = Command::new("g++").args(&simple_args).output();
            match gpp2 {
                Ok(r) if r.status.success() => true,
                _ => {
                    let nvcc_args = vec![
                        "-shared", "-o", &so_path,
                        &obj,
                        "-L", &cuda_lib_str,
                        "-lcudart",
                    ];
                    let nvcc_result = Command::new(&nvcc).args(&nvcc_args).output();
                    match nvcc_result {
                        Ok(r) if r.status.success() => true,
                        _ => {
                            println!("cargo:warning=CUDA kernel linking failed");
                            false
                        }
                    }
                }
            }
        }
    };

    if !so_exists {
        println!("cargo:warning=CUDA kernel linking failed, building CPU-only mode");
        return;
    }

    // Verify: check if kernel symbols are in .dynsym (use -W for wide output to avoid truncation)
    let verify = Command::new("readelf")
        .args(["-W", "--dyn-syms", &so_path])
        .output();
    if let Ok(r) = verify {
        let out = String::from_utf8_lossy(&r.stdout);
        let found = export_kernels.iter().filter(|k| out.contains(**k)).count();
        if found == export_kernels.len() {
            println!("cargo:warning=All {}/{} kernel symbols exported OK", found, export_kernels.len());
        } else {
            println!("cargo:warning=.dynsym has {}/{} kernel symbols (may be truncated, checking with nm)", found, export_kernels.len());
            let nm_verify = Command::new("nm")
                .args(["-D", &so_path])
                .output();
            if let Ok(nm_r) = nm_verify {
                let nm_out = String::from_utf8_lossy(&nm_r.stdout);
                let nm_found = export_kernels.iter().filter(|k| nm_out.contains(**k)).count();
                println!("cargo:warning=nm -D confirms {}/{} kernel symbols", nm_found, export_kernels.len());
            }
        }
    }

    // Copy the .so next to where the binary will be so it can be found at runtime
    let target_dir = std::path::Path::new("target/release");
    if target_dir.exists() {
        let _ = std::fs::copy(&so_path, target_dir.join("cuda_kernels.so"));
    }

    // Enable cuda_available cfg flag
    println!("cargo:rustc-cfg=cuda_available");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rustc-check-cfg=cfg(cuda_available)");
}
