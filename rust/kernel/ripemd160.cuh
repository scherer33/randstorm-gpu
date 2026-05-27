// RIPEMD-160 CUDA device functions (header-only for inclusion)
#ifndef RIPEMD160_CUH
#define RIPEMD160_CUH

#include <cstdint>

__device__ __constant__ uint32_t RIPEMD160_KL[5] = {
    0x00000000, 0x5A827999, 0x6ED9EBA1, 0x8F1BBCDC, 0xA953FD4E
};
__device__ __constant__ uint32_t RIPEMD160_KR[5] = {
    0x50A28BE6, 0x5C4DD124, 0x6D703EF3, 0x7A6D76E9, 0x00000000
};

__device__ __constant__ uint8_t RIPEMD160_RL[80] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
    3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
    1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
    4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
};

__device__ __constant__ uint8_t RIPEMD160_RR[80] = {
    5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
    6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
    15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
    8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
    12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
};

__device__ __constant__ uint8_t RIPEMD160_SL[80] = {
    11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
    7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
    11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
    11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
    9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
};

__device__ __constant__ uint8_t RIPEMD160_SR[80] = {
    8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
    9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
    9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
    15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
    8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
};

static __device__ uint32_t ripemd_rotl32(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32 - n));
}

static __device__ uint32_t ripemd_f(uint32_t j, uint32_t x, uint32_t y, uint32_t z) {
    if (j < 16) return x ^ y ^ z;
    if (j < 32) return (x & y) | (~x & z);
    if (j < 48) return (x | ~y) ^ z;
    if (j < 64) return (x & z) | (y & ~z);
    return x ^ (y | ~z);
}

static __device__ void ripemd160_device(const uint8_t* msg, uint32_t msg_len, uint8_t digest[20]) {
    uint32_t h0 = 0x67452301, h1 = 0xEFCDAB89, h2 = 0x98BADCFE;
    uint32_t h3 = 0x10325476, h4 = 0xC3D2E1F0;

    uint8_t block[64] = {0};
    for (uint32_t i = 0; i < msg_len && i < 55; i++) {
        block[i] = msg[i];
    }
    block[msg_len] = 0x80;
    uint64_t bit_len = (uint64_t)msg_len * 8;
    for (int i = 0; i < 8; i++) {
        block[56 + i] = (uint8_t)(bit_len >> (i * 8));
    }

    uint32_t X[16];
    for (int i = 0; i < 16; i++) {
        X[i] = ((uint32_t)block[i*4]) | ((uint32_t)block[i*4+1] << 8) |
               ((uint32_t)block[i*4+2] << 16) | ((uint32_t)block[i*4+3] << 24);
    }

    uint32_t al = h0, bl = h1, cl = h2, dl = h3, el = h4;
    uint32_t ar = h0, br = h1, cr = h2, dr = h3, er = h4;

    for (int j = 0; j < 80; j++) {
        int rnd = j / 16;
        uint32_t t = ripemd_rotl32((al + ripemd_f(j, bl, cl, dl) + X[RIPEMD160_RL[j]] + RIPEMD160_KL[rnd]), RIPEMD160_SL[j]) + el;
        al = el; el = dl; dl = ripemd_rotl32(cl, 10); cl = bl; bl = t;

        t = ripemd_rotl32((ar + ripemd_f(79-j, br, cr, dr) + X[RIPEMD160_RR[j]] + RIPEMD160_KR[rnd]), RIPEMD160_SR[j]) + er;
        ar = er; er = dr; dr = ripemd_rotl32(cr, 10); cr = br; br = t;
    }

    uint32_t t = h1 + cl + dr;
    h1 = h2 + dl + er;
    h2 = h3 + el + ar;
    h3 = h4 + al + br;
    h4 = h0 + bl + cr;
    h0 = t;

    uint32_t h[5] = {h0, h1, h2, h3, h4};
    for (int i = 0; i < 5; i++) {
        digest[i*4]   = (uint8_t)(h[i]);
        digest[i*4+1] = (uint8_t)(h[i] >> 8);
        digest[i*4+2] = (uint8_t)(h[i] >> 16);
        digest[i*4+3] = (uint8_t)(h[i] >> 24);
    }
}

#endif // RIPEMD160_CUH
