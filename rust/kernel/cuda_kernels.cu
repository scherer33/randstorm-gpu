// Unified CUDA kernel file — single compilation unit to avoid __constant__ memory conflicts.
// Also provides extern "C" host-side wrapper functions that launch kernels with <<<>>> syntax,
// since __global__ functions cannot be called via raw function pointers from dlsym.

#include "sha256.cuh"
#include "ripemd160.cuh"
#include "keccak256.cuh"
#include "base58check.cuh"
#include "secp256k1.cuh"

// ============================================================
// SHA-256 batch kernel
// ============================================================
__global__ void sha256_batch_kernel_impl(
    const uint8_t* input_data,
    const uint32_t* input_offsets,
    const uint32_t* input_lengths,
    uint8_t* output,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* msg = input_data + input_offsets[idx];
    uint32_t len = input_lengths[idx];
    uint8_t* out = output + idx * 32;

    sha256_device(msg, len, out);
}

// ============================================================
// Hash160 batch kernel (SHA-256 then RIPEMD-160)
// ============================================================
__global__ void hash160_batch_kernel_impl(
    const uint8_t* pubkey_data,
    uint8_t* output,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* pubkey = pubkey_data + idx * 33;
    uint8_t* out = output + idx * 20;

    uint8_t sha_result[32];
    sha256_device(pubkey, 33, sha_result);
    ripemd160_device(sha_result, 32, out);
}

// ============================================================
// Pubkey-to-hash160 kernel (same as hash160_batch but different name)
// ============================================================
__global__ void pubkey_to_hash160_kernel_impl(
    const uint8_t* pubkey_data,
    uint8_t* hash160_output,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* pubkey = pubkey_data + idx * 33;
    uint8_t* out = hash160_output + idx * 20;

    uint8_t sha_result[32];
    sha256_device(pubkey, 33, sha_result);
    ripemd160_device(sha_result, 32, out);
}

// ============================================================
// Keccak-256 batch kernel (Ethereum address derivation)
// Input: uncompressed pubkeys (64 bytes each, without 0x04 prefix)
// Output: 32-byte Keccak-256 digests
// ============================================================
__global__ void keccak256_batch_kernel_impl(
    const uint8_t* pubkey_data,
    uint8_t* output,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* pubkey = pubkey_data + idx * 64;
    uint8_t* out = output + idx * 32;

    keccak256_device(pubkey, 64, out);
}

// ============================================================
// Multi-hash batch kernel (BTC hash160 + ETH keccak256)
// Input: uncompressed pubkeys (65 bytes each, 0x04 || X || Y)
// Output: hash160 (20 bytes each) + keccak256 digest (32 bytes each)
// BTC: compresses to 33 bytes, then SHA-256 + RIPEMD-160
// ETH: Keccak-256 of 64 bytes (X || Y, without 0x04 prefix)
// ============================================================
__global__ void multi_hash_batch_kernel_impl(
    const uint8_t* pubkey_data,
    uint8_t* hash160_output,
    uint8_t* keccak256_output,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Input is 65-byte uncompressed pubkeys (0x04 || X || Y)
    const uint8_t* pubkey = pubkey_data + idx * 65;
    uint8_t* h160_out = hash160_output + idx * 20;
    uint8_t* kec_out = keccak256_output + idx * 32;

    // BTC: hash160 of the compressed form.
    // We need compressed pubkey (33 bytes) for BTC addresses.
    // From uncompressed (0x04 || X || Y): compressed = (prefix_byte || X)
    // where prefix_byte = 0x02 if Y is even, 0x03 if Y is odd.
    uint8_t prefix = (pubkey[64] & 1) ? 0x03 : 0x02;  // Y parity determines prefix
    uint8_t compressed[33];
    compressed[0] = prefix;
    for (int i = 0; i < 32; i++) compressed[1 + i] = pubkey[1 + i];  // X coords

    uint8_t sha_result[32];
    sha256_device(compressed, 33, sha_result);
    ripemd160_device(sha_result, 32, h160_out);

    // ETH: Keccak-256 of 64 bytes (X || Y, without 0x04 prefix)
    keccak256_device(pubkey + 1, 64, kec_out);
}

// ============================================================
// PBKDF2-HMAC-SHA512 kernel (inline to avoid separate CU __constant__ conflict)
// ============================================================

// SHA-512 constants
__device__ __constant__ uint64_t SHA512_K[80] = {
    0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
    0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
    0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
    0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
    0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
    0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
    0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
    0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
    0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
    0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
    0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
    0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
    0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
    0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
    0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
    0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
    0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
    0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
    0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
    0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
};

__device__ uint64_t rotr64(uint64_t x, uint64_t n) {
    return (x >> n) | (x << (64 - n));
}

__device__ void sha512_transform(uint64_t state[8], const uint8_t block[128]) {
    uint64_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint64_t)block[i*8] << 56) | ((uint64_t)block[i*8+1] << 48) |
               ((uint64_t)block[i*8+2] << 40) | ((uint64_t)block[i*8+3] << 32) |
               ((uint64_t)block[i*8+4] << 24) | ((uint64_t)block[i*8+5] << 16) |
               ((uint64_t)block[i*8+6] << 8)  | (uint64_t)block[i*8+7];
    }
    for (int i = 16; i < 80; i++) {
        uint64_t s0 = rotr64(w[i-15], 1) ^ rotr64(w[i-15], 8) ^ (w[i-15] >> 7);
        uint64_t s1 = rotr64(w[i-2], 19) ^ rotr64(w[i-2], 61) ^ (w[i-2] >> 6);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }

    uint64_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint64_t e = state[4], f = state[5], g = state[6], h = state[7];

    for (int i = 0; i < 80; i++) {
        uint64_t S1 = rotr64(e, 14) ^ rotr64(e, 18) ^ rotr64(e, 41);
        uint64_t ch = (e & f) ^ (~e & g);
        uint64_t temp1 = h + S1 + ch + SHA512_K[i] + w[i];
        uint64_t S0 = rotr64(a, 28) ^ rotr64(a, 34) ^ rotr64(a, 39);
        uint64_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint64_t temp2 = S0 + maj;
        h = g; g = f; f = e; e = d + temp1;
        d = c; c = b; b = a; a = temp1 + temp2;
    }

    state[0] += a; state[1] += b; state[2] += c; state[3] += d;
    state[4] += e; state[5] += f; state[6] += g; state[7] += h;
}

__device__ void sha512_device(const uint8_t* msg, uint32_t msg_len, uint8_t digest[64]) {
    uint64_t state[8] = {
        0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
        0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
        0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
        0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
    };

    uint8_t block[128] = {0};
    uint32_t remaining = msg_len;

    uint32_t full_blocks = msg_len / 128;
    for (uint32_t i = 0; i < full_blocks; i++) {
        sha512_transform(state, msg + i * 128);
    }
    remaining = msg_len % 128;

    for (uint32_t i = 0; i < remaining; i++) {
        block[i] = msg[full_blocks * 128 + i];
    }
    block[remaining] = 0x80;

    if (remaining >= 112) {
        sha512_transform(state, block);
        uint8_t block2[128] = {0};
        uint64_t bit_len = (uint64_t)msg_len * 8;
        for (int i = 0; i < 8; i++) {
            block2[120 + i] = (uint8_t)(bit_len >> (56 - i * 8));
        }
        sha512_transform(state, block2);
    } else {
        uint64_t bit_len = (uint64_t)msg_len * 8;
        for (int i = 0; i < 8; i++) {
            block[120 + i] = (uint8_t)(bit_len >> (56 - i * 8));
        }
        sha512_transform(state, block);
    }

    for (int i = 0; i < 8; i++) {
        digest[i*8]   = (uint8_t)(state[i] >> 56);
        digest[i*8+1] = (uint8_t)(state[i] >> 48);
        digest[i*8+2] = (uint8_t)(state[i] >> 40);
        digest[i*8+3] = (uint8_t)(state[i] >> 32);
        digest[i*8+4] = (uint8_t)(state[i] >> 24);
        digest[i*8+5] = (uint8_t)(state[i] >> 16);
        digest[i*8+6] = (uint8_t)(state[i] >> 8);
        digest[i*8+7] = (uint8_t)(state[i]);
    }
}

__device__ void hmac_sha512_device(
    const uint8_t* key, uint32_t key_len,
    const uint8_t* msg, uint32_t msg_len,
    uint8_t digest[64]
) {
    uint8_t k_pad[128] = {0};

    if (key_len > 128) {
        sha512_device(key, key_len, k_pad);
        key_len = 64;
    } else {
        for (uint32_t i = 0; i < key_len; i++) k_pad[i] = key[i];
    }

    uint8_t k_ipad[128];
    for (int i = 0; i < 128; i++) k_ipad[i] = k_pad[i] ^ 0x36;

    // Stack-allocated inner message (max 128 + msg_len, BIP39 salt is always <= 140)
    uint8_t inner[256];
    for (int i = 0; i < 128; i++) inner[i] = k_ipad[i];
    for (uint32_t i = 0; i < msg_len; i++) inner[128 + i] = msg[i];
    uint8_t inner_hash[64];
    sha512_device(inner, 128 + msg_len, inner_hash);

    uint8_t k_opad[128];
    for (int i = 0; i < 128; i++) k_opad[i] = k_pad[i] ^ 0x5c;

    uint8_t outer_msg[128 + 64];
    for (int i = 0; i < 128; i++) outer_msg[i] = k_opad[i];
    for (int i = 0; i < 64; i++) outer_msg[128 + i] = inner_hash[i];
    sha512_device(outer_msg, 192, digest);
}

__global__ void pbkdf2_hmac_sha512_kernel_impl(
    const uint8_t* mnemonic_data,
    const uint32_t* mnemonic_offsets,
    uint8_t* seed_output,
    uint32_t count,
    uint32_t iterations
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    const uint8_t* mnemonic = mnemonic_data + mnemonic_offsets[idx];
    uint32_t mnemonic_len = mnemonic_offsets[idx + 1] - mnemonic_offsets[idx];
    uint8_t* out = seed_output + idx * 64;

    const uint8_t salt[] = "mnemonic";
    uint32_t salt_len = 8;

    uint8_t salt_block[12];
    for (uint32_t i = 0; i < salt_len; i++) salt_block[i] = salt[i];
    salt_block[salt_len] = 0;
    salt_block[salt_len+1] = 0;
    salt_block[salt_len+2] = 0;
    salt_block[salt_len+3] = 1;

    uint8_t u[64];
    hmac_sha512_device(mnemonic, mnemonic_len, salt_block, salt_len + 4, u);

    uint8_t t[64];
    for (int i = 0; i < 64; i++) t[i] = u[i];

    for (uint32_t iter = 1; iter < iterations; iter++) {
        hmac_sha512_device(mnemonic, mnemonic_len, u, 64, u);
        for (int i = 0; i < 64; i++) t[i] ^= u[i];
    }

    for (int i = 0; i < 64; i++) out[i] = t[i];
}

// ============================================================
// Host-side wrapper functions (extern "C" for dlsym from Rust)
// Uses GPU device memory + async copies for better throughput.
// Pinned host memory when possible for faster DMA transfers.
// ============================================================

struct CudaResult {
    int32_t code;
};

#define CUDA_BLOCK_SIZE 256

extern "C" CudaResult pbkdf2_hmac_sha512_kernel(
    const uint8_t* mnemonic_data,
    const uint32_t* mnemonic_offsets,
    uint8_t* seed_output,
    uint32_t count,
    uint32_t iterations
) {
    int grid = (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    pbkdf2_hmac_sha512_kernel_impl<<<grid, CUDA_BLOCK_SIZE>>>(
        mnemonic_data, mnemonic_offsets, seed_output, count, iterations
    );
    cudaError_t err = cudaDeviceSynchronize();
    CudaResult r;
    r.code = (err == cudaSuccess) ? 0 : (int32_t)err;
    return r;
}

extern "C" CudaResult hash160_batch_kernel(
    const uint8_t* pubkey_data,
    uint8_t* output,
    uint32_t count
) {
    int grid = (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    hash160_batch_kernel_impl<<<grid, CUDA_BLOCK_SIZE>>>(pubkey_data, output, count);
    cudaError_t err = cudaDeviceSynchronize();
    CudaResult r;
    r.code = (err == cudaSuccess) ? 0 : (int32_t)err;
    return r;
}

extern "C" CudaResult pubkey_to_hash160_kernel(
    const uint8_t* pubkey_data,
    uint8_t* hash160_output,
    uint32_t count
) {
    cudaError_t err;

    // Allocate GPU device memory
    uint8_t* d_pubkeys = nullptr;
    uint8_t* d_hash160s = nullptr;

    err = cudaMalloc(&d_pubkeys, (size_t)count * 33);
    if (err != cudaSuccess) { CudaResult r; r.code = (int32_t)err; return r; }
    err = cudaMalloc(&d_hash160s, (size_t)count * 20);
    if (err != cudaSuccess) { cudaFree(d_pubkeys); CudaResult r; r.code = (int32_t)err; return r; }

    // Try to use pinned host memory for faster DMA transfers
    uint8_t* h_pubkeys_pinned = nullptr;
    bool using_pinned = (cudaMallocHost(&h_pubkeys_pinned, (size_t)count * 33) == cudaSuccess);

    uint8_t* src_pubkeys;
    if (using_pinned) {
        // Copy input data to pinned buffer for faster GPU transfer
        memcpy(h_pubkeys_pinned, pubkey_data, (size_t)count * 33);
        src_pubkeys = h_pubkeys_pinned;
    } else {
        src_pubkeys = (uint8_t*)pubkey_data;
    }

    // Async copy to GPU, launch kernel, async copy back
    cudaMemcpyAsync(d_pubkeys, src_pubkeys, (size_t)count * 33, cudaMemcpyHostToDevice, 0);
    int grid = (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    pubkey_to_hash160_kernel_impl<<<grid, CUDA_BLOCK_SIZE>>>(d_pubkeys, d_hash160s, count);
    cudaMemcpyAsync(hash160_output, d_hash160s, (size_t)count * 20, cudaMemcpyDeviceToHost, 0);

    err = cudaDeviceSynchronize();

    if (using_pinned) {
        cudaFreeHost(h_pubkeys_pinned);
    }
    cudaFree(d_pubkeys);
    cudaFree(d_hash160s);

    CudaResult r;
    r.code = (err == cudaSuccess) ? 0 : (int32_t)err;
    return r;
}

// ============================================================
// Keccak-256 batch kernel wrapper (for ETH address derivation)
// Input: 64-byte uncompressed pubkeys (without 0x04 prefix)
// Output: 32-byte Keccak-256 digests
// ============================================================
extern "C" CudaResult keccak256_batch_kernel(
    const uint8_t* pubkey_data,
    uint8_t* output,
    uint32_t count
) {
    cudaError_t err;

    uint8_t* d_pubkeys = nullptr;
    uint8_t* d_output = nullptr;

    err = cudaMalloc(&d_pubkeys, (size_t)count * 64);
    if (err != cudaSuccess) { CudaResult r; r.code = (int32_t)err; return r; }
    err = cudaMalloc(&d_output, (size_t)count * 32);
    if (err != cudaSuccess) { cudaFree(d_pubkeys); CudaResult r; r.code = (int32_t)err; return r; }

    // Pinned host memory
    uint8_t* h_pubkeys_pinned = nullptr;
    bool using_pinned = (cudaMallocHost(&h_pubkeys_pinned, (size_t)count * 64) == cudaSuccess);

    uint8_t* src_pubkeys;
    if (using_pinned) {
        memcpy(h_pubkeys_pinned, pubkey_data, (size_t)count * 64);
        src_pubkeys = h_pubkeys_pinned;
    } else {
        src_pubkeys = (uint8_t*)pubkey_data;
    }

    cudaMemcpyAsync(d_pubkeys, src_pubkeys, (size_t)count * 64, cudaMemcpyHostToDevice, 0);
    int grid = (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    keccak256_batch_kernel_impl<<<grid, CUDA_BLOCK_SIZE>>>(d_pubkeys, d_output, count);
    cudaMemcpyAsync(output, d_output, (size_t)count * 32, cudaMemcpyDeviceToHost, 0);

    err = cudaDeviceSynchronize();

    if (using_pinned) {
        cudaFreeHost(h_pubkeys_pinned);
    }
    cudaFree(d_pubkeys);
    cudaFree(d_output);

    CudaResult r;
    r.code = (err == cudaSuccess) ? 0 : (int32_t)err;
    return r;
}

// ============================================================
// Multi-hash batch kernel wrapper (BTC hash160 + ETH keccak256)
// Input: 65-byte uncompressed pubkeys (WITH 0x04 prefix)
// Output: hash160 (20 bytes each) + keccak256 (32 bytes each)
// ============================================================
extern "C" CudaResult multi_hash_batch_kernel(
    const uint8_t* pubkey_data,
    uint8_t* hash160_output,
    uint8_t* keccak256_output,
    uint32_t count
) {
    cudaError_t err;

    uint8_t* d_pubkeys = nullptr;
    uint8_t* d_hash160s = nullptr;
    uint8_t* d_keccak256s = nullptr;

    err = cudaMalloc(&d_pubkeys, (size_t)count * 65);
    if (err != cudaSuccess) { CudaResult r; r.code = (int32_t)err; return r; }
    err = cudaMalloc(&d_hash160s, (size_t)count * 20);
    if (err != cudaSuccess) { cudaFree(d_pubkeys); CudaResult r; r.code = (int32_t)err; return r; }
    err = cudaMalloc(&d_keccak256s, (size_t)count * 32);
    if (err != cudaSuccess) { cudaFree(d_pubkeys); cudaFree(d_hash160s); CudaResult r; r.code = (int32_t)err; return r; }

    // Pinned host memory for input
    uint8_t* h_pubkeys_pinned = nullptr;
    bool using_pinned = (cudaMallocHost(&h_pubkeys_pinned, (size_t)count * 65) == cudaSuccess);

    uint8_t* src_pubkeys;
    if (using_pinned) {
        memcpy(h_pubkeys_pinned, pubkey_data, (size_t)count * 65);
        src_pubkeys = h_pubkeys_pinned;
    } else {
        src_pubkeys = (uint8_t*)pubkey_data;
    }

    // Pinned host memory for outputs
    uint8_t* h_hash160_pinned = nullptr;
    bool using_pinned_h160 = (cudaMallocHost(&h_hash160_pinned, (size_t)count * 20) == cudaSuccess);

    uint8_t* h_keccak_pinned = nullptr;
    bool using_pinned_kec = (cudaMallocHost(&h_keccak_pinned, (size_t)count * 32) == cudaSuccess);

    cudaMemcpyAsync(d_pubkeys, src_pubkeys, (size_t)count * 65, cudaMemcpyHostToDevice, 0);
    int grid = (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    multi_hash_batch_kernel_impl<<<grid, CUDA_BLOCK_SIZE>>>(d_pubkeys, d_hash160s, d_keccak256s, count);
    cudaMemcpyAsync(hash160_output, d_hash160s, (size_t)count * 20, cudaMemcpyDeviceToHost, 0);
    cudaMemcpyAsync(keccak256_output, d_keccak256s, (size_t)count * 32, cudaMemcpyDeviceToHost, 0);

    err = cudaDeviceSynchronize();

    if (using_pinned) cudaFreeHost(h_pubkeys_pinned);
    if (using_pinned_h160) cudaFreeHost(h_hash160_pinned);
    if (using_pinned_kec) cudaFreeHost(h_keccak_pinned);
    cudaFree(d_pubkeys);
    cudaFree(d_hash160s);
    cudaFree(d_keccak256s);

    CudaResult r;
    r.code = (err == cudaSuccess) ? 0 : (int32_t)err;
    return r;
}

// ============================================================
// RandStorm: secp256k1 + hash160 combined pipeline
// Pipeline: private keys → compressed pubkeys → hash160, all on GPU
// ============================================================
extern "C" CudaResult randstorm_secp256k1_hash160_kernel(
    const uint8_t* private_keys,
    uint8_t* hash160_output,
    uint32_t count
) {
    cudaError_t err;
    uint8_t *d_privkeys=nullptr, *d_pubkeys=nullptr, *d_hash160s=nullptr;

    err = cudaMalloc(&d_privkeys, (size_t)count * 32);
    if(err != cudaSuccess) { CudaResult r; r.code = (int32_t)err; return r; }
    err = cudaMalloc(&d_pubkeys, (size_t)count * 33);
    if(err != cudaSuccess) { cudaFree(d_privkeys); CudaResult r; r.code = (int32_t)err; return r; }
    err = cudaMalloc(&d_hash160s, (size_t)count * 20);
    if(err != cudaSuccess) { cudaFree(d_privkeys); cudaFree(d_pubkeys); CudaResult r; r.code = (int32_t)err; return r; }

    cudaMemcpyAsync(d_privkeys, private_keys, (size_t)count * 32, cudaMemcpyHostToDevice, 0);

    int grid = (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    secp256k1_batch_kernel_impl<<<grid, CUDA_BLOCK_SIZE>>>(d_privkeys, d_pubkeys, count);
    hash160_batch_kernel_impl<<<grid, CUDA_BLOCK_SIZE>>>(d_pubkeys, d_hash160s, count);

    cudaMemcpyAsync(hash160_output, d_hash160s, (size_t)count * 20, cudaMemcpyDeviceToHost, 0);
    err = cudaDeviceSynchronize();

    cudaFree(d_privkeys); cudaFree(d_pubkeys); cudaFree(d_hash160s);
    CudaResult r; r.code = (err == cudaSuccess) ? 0 : (int32_t)err;
    return r;
}

// Secp256k1-only: private keys → compressed pubkeys
extern "C" CudaResult randstorm_secp256k1_kernel(
    const uint8_t* private_keys,
    uint8_t* pubkeys,
    uint32_t count
) {
    cudaError_t err;
    uint8_t *d_privkeys=nullptr, *d_pubkeys=nullptr;

    err = cudaMalloc(&d_privkeys, (size_t)count * 32);
    if(err != cudaSuccess) { CudaResult r; r.code = (int32_t)err; return r; }
    err = cudaMalloc(&d_pubkeys, (size_t)count * 33);
    if(err != cudaSuccess) { cudaFree(d_privkeys); CudaResult r; r.code = (int32_t)err; return r; }

    cudaMemcpyAsync(d_privkeys, private_keys, (size_t)count * 32, cudaMemcpyHostToDevice, 0);

    int grid = (count + CUDA_BLOCK_SIZE - 1) / CUDA_BLOCK_SIZE;
    secp256k1_batch_kernel_impl<<<grid, CUDA_BLOCK_SIZE>>>(d_privkeys, d_pubkeys, count);

    cudaMemcpyAsync(pubkeys, d_pubkeys, (size_t)count * 33, cudaMemcpyDeviceToHost, 0);
    err = cudaDeviceSynchronize();

    cudaFree(d_privkeys); cudaFree(d_pubkeys);
    CudaResult r; r.code = (err == cudaSuccess) ? 0 : (int32_t)err;
    return r;
}
