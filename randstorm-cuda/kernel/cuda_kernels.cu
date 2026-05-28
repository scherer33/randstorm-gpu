// RandStorm CUDA kernels — unified compilation unit
// Split pipeline: Phase 1 (GPU) → CPU batch inversion → Phase 2 (GPU)
// Avoids per-thread fe_inv (slow on GPU) by doing single modular inverse on CPU
#include "sha256.cuh"
#include "ripemd160.cuh"
#include "secp256k1_32.cuh"

__device__ __constant__ uint32_t d_ec_table_x[256 * NLIMBS];
__device__ __constant__ uint32_t d_ec_table_y[256 * NLIMBS];

// ============================================================
// Phase 1: Private keys → Jacobian points (table-based scalar mult)
// ============================================================
__global__ void privkey_to_jacobian_kernel(
    const uint8_t* private_keys,
    uint32_t* jpx, uint32_t* jpy, uint32_t* jpz,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count) return;

    const uint8_t* key = private_keys + idx * 32;
    jp R;
    jp_mul_table(key, d_ec_table_x, d_ec_table_y, &R);

    uint32_t* out_x = jpx + idx * NLIMBS;
    uint32_t* out_y = jpy + idx * NLIMBS;
    uint32_t* out_z = jpz + idx * NLIMBS;
    for(int i=0; i<NLIMBS; i++) {
        out_x[i] = R.x[i];
        out_y[i] = R.y[i];
        out_z[i] = R.z[i];
    }
}

// ============================================================
// Phase 2: Jacobian + pre-computed Z-inverse → Affine → Hash160
// Z-inverse is computed on CPU using batch inversion (1 modular inv + O(N) mults)
// Each thread: load Jacobian + z_inv → affine → compressed → SHA256 → RIPEMD160
// ============================================================
__global__ void jacobian_to_hash160_kernel(
    const uint32_t* jpx, const uint32_t* jpy,
    const uint32_t* z_inv,
    uint8_t* h160,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count) return;

    // Load pre-computed Z-inverse
    fe zi;
    const uint32_t* zi_src = z_inv + idx * NLIMBS;
    for(int i=0; i<NLIMBS; i++) zi[i] = zi_src[i];

    // Load Jacobian X and Y
    fe jx, jy;
    const uint32_t* jx_src = jpx + idx * NLIMBS;
    const uint32_t* jy_src = jpy + idx * NLIMBS;
    for(int i=0; i<NLIMBS; i++) { jx[i] = jx_src[i]; jy[i] = jy_src[i]; }

    // Compute affine: X = JX * Z^-2, Y = JY * Z^-3
    // zi2 = zi^2, zi3 = zi^3 = zi2 * zi
    fe zi2, zi3, x, y;
    fe_mul(zi, zi, zi2);      // zi2 = Z^-2
    fe_mul(zi2, zi, zi3);     // zi3 = Z^-3
    fe_mul(jx, zi2, x);       // X = JX / Z^2
    fe_mul(jy, zi3, y);       // Y = JY / Z^3

    // Compressed pubkey
    uint8_t pubkey[33];
    pubkey[0] = (y[0] & 1) ? 0x03 : 0x02;
    for(int b=0; b<32; b++) {
        int limb = (31-b)/4;
        int shift = ((31-b)%4)*8;
        pubkey[1+b] = (uint8_t)(x[limb] >> shift);
    }

    // Hash160
    uint8_t sha_result[32];
    sha256_device(pubkey, 33, sha_result);
    ripemd160_device(sha_result, 32, h160 + idx * 20);
}

extern "C" struct CudaResult { int32_t code; };

#define CUDA_BLOCK_SIZE 256

extern "C" CudaResult randstorm_upload_ec_table(
    const uint32_t* table_x,
    const uint32_t* table_y
) {
    cudaError_t err;
    err = cudaMemcpyToSymbol(d_ec_table_x, table_x, 256 * NLIMBS * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) { CudaResult r; r.code = (int32_t)err; return r; }
    err = cudaMemcpyToSymbol(d_ec_table_y, table_y, 256 * NLIMBS * sizeof(uint32_t), 0, cudaMemcpyHostToDevice);
    if(err != cudaSuccess) { CudaResult r; r.code = (int32_t)err; return r; }
    CudaResult r; r.code = 0; return r;
}

static uint8_t *g_d_keys = nullptr;
static uint32_t *g_d_jpx = nullptr;
static uint32_t *g_d_jpy = nullptr;
static uint32_t *g_d_jpz = nullptr;
static uint8_t *g_d_h160 = nullptr;
static size_t g_alloc_size = 0;

extern "C" CudaResult randstorm_gpu_init(size_t max_count) {
    cudaError_t err;
    g_alloc_size = max_count;
    err = cudaMalloc(&g_d_keys, max_count * 32);
    if(err) { CudaResult r; r.code=(int32_t)err; return r; }
    err = cudaMalloc(&g_d_jpx, max_count * 32);
    if(err) { cudaFree(g_d_keys); CudaResult r; r.code=(int32_t)err; return r; }
    err = cudaMalloc(&g_d_jpy, max_count * 32);
    if(err) { cudaFree(g_d_jpx); cudaFree(g_d_keys); CudaResult r; r.code=(int32_t)err; return r; }
    err = cudaMalloc(&g_d_jpz, max_count * 32);
    if(err) { cudaFree(g_d_jpy); cudaFree(g_d_jpx); cudaFree(g_d_keys); CudaResult r; r.code=(int32_t)err; return r; }
    err = cudaMalloc(&g_d_h160, max_count * 20);
    if(err) { cudaFree(g_d_jpz); cudaFree(g_d_jpy); cudaFree(g_d_jpx); cudaFree(g_d_keys); CudaResult r; r.code=(int32_t)err; return r; }
    CudaResult r; r.code = 0; return r;
}

extern "C" CudaResult randstorm_gpu_free() {
    cudaFree(g_d_keys); cudaFree(g_d_jpx); cudaFree(g_d_jpy); cudaFree(g_d_jpz); cudaFree(g_d_h160);
    g_d_keys=nullptr; g_d_jpx=nullptr; g_d_jpy=nullptr; g_d_jpz=nullptr; g_d_h160=nullptr;
    g_alloc_size = 0;
    CudaResult r; r.code = 0; return r;
}

// Phase 1: Upload private keys → run Jacobian kernel → download Z values to CPU
extern "C" CudaResult randstorm_pk_to_jacobian(
    const uint8_t* private_keys,
    uint32_t* z_cpu,
    uint32_t count
) {
    if(count > g_alloc_size) { CudaResult r; r.code=-1; return r; }
    cudaError_t err;

    // Upload keys
    err = cudaMemcpyAsync(g_d_keys, private_keys, (size_t)count * 32, cudaMemcpyHostToDevice, 0);
    if(err != cudaSuccess) { CudaResult r; r.code=(int32_t)err; return r; }

    // Run Jacobian kernel
    int grid = (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    privkey_to_jacobian_kernel<<<grid, CUDA_BLOCK_SIZE>>>(g_d_keys, g_d_jpx, g_d_jpy, g_d_jpz, count);

    // Download Z values to CPU (for batch inversion)
    err = cudaMemcpyAsync(z_cpu, g_d_jpz, (size_t)count * 32, cudaMemcpyDeviceToHost, 0);
    if(err != cudaSuccess) { CudaResult r; r.code=(int32_t)err; return r; }

    err = cudaDeviceSynchronize();
    CudaResult r; r.code = (err == cudaSuccess) ? 0 : (int32_t)err;
    return r;
}

// Phase 2: Upload Z-inverse → run affine+hash160 kernel → download hash160 to CPU
// Debug: read back all Jacobian coordinates from GPU Phase 1
extern "C" CudaResult randstorm_read_jacobian(
    uint32_t* jpx_cpu, uint32_t* jpy_cpu, uint32_t* jpz_cpu, uint32_t count
) {
    if(count > g_alloc_size) { CudaResult r; r.code=-1; return r; }
    cudaError_t err;
    err = cudaMemcpyAsync(jpx_cpu, g_d_jpx, (size_t)count * 32, cudaMemcpyDeviceToHost, 0);
    if(err != cudaSuccess) { CudaResult r; r.code=(int32_t)err; return r; }
    err = cudaMemcpyAsync(jpy_cpu, g_d_jpy, (size_t)count * 32, cudaMemcpyDeviceToHost, 0);
    if(err != cudaSuccess) { CudaResult r; r.code=(int32_t)err; return r; }
    err = cudaMemcpyAsync(jpz_cpu, g_d_jpz, (size_t)count * 32, cudaMemcpyDeviceToHost, 0);
    if(err != cudaSuccess) { CudaResult r; r.code=(int32_t)err; return r; }
    err = cudaDeviceSynchronize();
    CudaResult r; r.code = (err == cudaSuccess) ? 0 : (int32_t)err;
    return r;
}

extern "C" CudaResult randstorm_jacobian_to_hash160(
    const uint32_t* z_inv_cpu,
    uint8_t* h160_cpu,
    uint32_t count
) {
    if(count > g_alloc_size) { CudaResult r; r.code=-1; return r; }
    cudaError_t err;

    // Upload Z-inverse (reuse g_d_jpz buffer)
    err = cudaMemcpyAsync(g_d_jpz, z_inv_cpu, (size_t)count * 32, cudaMemcpyHostToDevice, 0);
    if(err != cudaSuccess) { CudaResult r; r.code=(int32_t)err; return r; }

    // Run affine+hash160 kernel
    int grid = (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    jacobian_to_hash160_kernel<<<grid, CUDA_BLOCK_SIZE>>>(g_d_jpx, g_d_jpy, g_d_jpz, g_d_h160, count);

    // Download hash160 results
    err = cudaMemcpyAsync(h160_cpu, g_d_h160, (size_t)count * 20, cudaMemcpyDeviceToHost, 0);
    if(err != cudaSuccess) { CudaResult r; r.code=(int32_t)err; return r; }

    err = cudaDeviceSynchronize();
    CudaResult r; r.code = (err == cudaSuccess) ? 0 : (int32_t)err;
    return r;
}
