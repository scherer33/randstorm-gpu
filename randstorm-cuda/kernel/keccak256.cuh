// Keccak-256 CUDA device functions (Ethereum-style, domain 0x01 padding)
// Based on NIST FIPS 202 Keccak specification (public domain).
// Reference: bradley-cao/cuda-keccak256 for Ethereum padding convention.
#ifndef KECCAK256_CUH
#define KECCAK256_CUH

#include <cstdint>

// Keccak-f[1600] round constants
__device__ __constant__ uint64_t KECCAK_RC[24] = {
    0x0000000000000001ULL, 0x0000000000008082ULL,
    0x800000000000808AULL, 0x8000000080008000ULL,
    0x000000000000808BULL, 0x0000000080000001ULL,
    0x8000000080008081ULL, 0x8000000000008009ULL,
    0x000000000000008AULL, 0x0000000000000088ULL,
    0x0000000080008009ULL, 0x000000008000000AULL,
    0x000000008000808BULL, 0x800000000000008BULL,
    0x8000000000008089ULL, 0x8000000000008003ULL,
    0x8000000000008002ULL, 0x8000000000000080ULL,
    0x000000000000800AULL, 0x800000008000000AULL,
    0x8000000080008081ULL, 0x8000000000008080ULL,
    0x0000000080000001ULL, 0x8000000080008008ULL
};

// Rotation offsets for rho step, indexed by state[5*y + x]
// Standard Keccak rho offsets from FIPS 202
__device__ __constant__ uint8_t KECCAK_ROT[25] = {
     0,  1, 62, 28, 27,
    36, 44,  6, 55, 20,
     3, 10, 43, 25, 39,
    41, 45, 15, 21,  8,
    18,  2, 61, 56, 14
};

static __device__ __forceinline__ uint64_t keccak_rotl64(uint64_t x, uint8_t n) {
    return (x << n) | (x >> (64 - n));
}

// Keccak-f[1600] permutation: 24 rounds on the 5x5 array of 64-bit lanes
static __device__ void keccak_f1600(uint64_t state[25]) {
    for (int round = 0; round < 24; round++) {
        // Theta step
        uint64_t C[5], D[5];
        for (int x = 0; x < 5; x++) {
            C[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
        }
        for (int x = 0; x < 5; x++) {
            D[x] = C[(x + 4) % 5] ^ keccak_rotl64(C[(x + 1) % 5], 1);
        }
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                state[5 * y + x] ^= D[x];
            }
        }

        // Rho and Pi steps (combined)
        uint64_t temp[25];
        for (int x = 0; x < 5; x++) {
            for (int y = 0; y < 5; y++) {
                int idx = 5 * y + x;
                int new_idx = 5 * ((2 * x + 3 * y) % 5) + y;  // π: (x,y) → (y, 2x+3y)
                temp[new_idx] = keccak_rotl64(state[idx], KECCAK_ROT[idx]);
            }
        }
        for (int i = 0; i < 25; i++) {
            state[i] = temp[i];
        }

        // Chi step
        for (int y = 0; y < 5; y++) {
            uint64_t row[5];
            for (int x = 0; x < 5; x++) {
                row[x] = state[5 * y + x];
            }
            for (int x = 0; x < 5; x++) {
                state[5 * y + x] = row[x] ^ ((~row[(x + 1) % 5]) & row[(x + 2) % 5]);
            }
        }

        // Iota step
        state[0] ^= KECCAK_RC[round];
    }
}

/// Keccak-256 device function — Ethereum-style (domain separator 0x01).
/// For Ethereum address derivation: hash 64-byte pubkey (X+Y without 0x04 prefix),
/// then take bytes [12..32] of the 32-byte digest.
///
/// Handles messages up to 135 bytes (fits in single 136-byte rate block for Keccak-256).
/// This covers our use case: 64-byte uncompressed pubkey.
static __device__ void keccak256_device(const uint8_t* msg, uint32_t msg_len, uint8_t digest[32]) {
    uint64_t state[25] = {0};

    // Absorb: load message bytes into state (little-endian, lane-interleaved)
    // Keccak-256 rate = 1088 bits = 136 bytes = 17 lanes of 64 bits
    uint32_t rate_bytes = 136;
    uint32_t rate_lanes = rate_bytes / 8;

    // Full lanes
    uint32_t full_lanes = msg_len / 8;
    for (uint32_t i = 0; i < full_lanes && i < rate_lanes; i++) {
        uint64_t lane = 0;
        for (int j = 0; j < 8; j++) {
            lane |= (uint64_t)msg[i * 8 + j] << (j * 8);
        }
        state[i] ^= lane;
    }

    // Remaining bytes in last partial lane
    uint32_t remaining = msg_len % 8;
    if (remaining > 0) {
        uint64_t lane = 0;
        uint32_t offset = full_lanes * 8;
        for (uint32_t j = 0; j < remaining; j++) {
            lane |= (uint64_t)msg[offset + j] << (j * 8);
        }
        // Ethereum padding: domain separator 0x01 after message, then 0x80 at end
        lane |= (uint64_t)0x01 << (remaining * 8);
        // If this is the last lane and remaining+1 < 8, we can set 0x80 in same lane
        // Otherwise 0x80 goes in the next lane
        if (remaining + 1 < 8) {
            lane |= (uint64_t)0x80 << (7 * 8);  // 0x80 at byte 7 of this lane
            state[full_lanes] ^= lane;
        } else {
            state[full_lanes] ^= lane;
            // 0x80 in the next lane
            if (full_lanes + 1 < rate_lanes) {
                state[full_lanes + 1] ^= (uint64_t)0x80 << (7 * 8);
            }
        }
    } else {
        // Message length is exact multiple of 8 bytes
        // Padding: 0x01 at byte position msg_len, 0x80 at byte 135 (last byte of rate)
        state[full_lanes] ^= (uint64_t)0x01;  // 0x01 at byte 0 of this lane
        state[rate_lanes - 1] ^= (uint64_t)0x80 << (7 * 8);  // 0x80 at last byte of rate
    }

    // Permute
    keccak_f1600(state);

    // Squeeze: output 32 bytes (4 lanes of 64 bits, little-endian)
    for (int i = 0; i < 4; i++) {
        uint64_t lane = state[i];
        for (int j = 0; j < 8; j++) {
            digest[i * 8 + j] = (uint8_t)(lane >> (j * 8));
        }
    }
}

#endif // KECCAK256_CUH
