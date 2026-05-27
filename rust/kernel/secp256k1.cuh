// secp256k1 CUDA kernel: modular arithmetic + scalar multiplication
// 256-bit numbers as 4x64-bit limbs (little-endian)
// Fast reduction using p = 2^256 - 2^32 - 977
#ifndef SECP256K1_CUH
#define SECP256K1_CUH

#include <cstdint>

static __device__ const uint64_t P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL
};

static __device__ const uint64_t GX[4] = {
    0x16F81798B59F2815ULL, 0x02CEDCE28D959F28ULL,
    0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL
};

static __device__ const uint64_t GY[4] = {
    0x0FFB10D4B8C47D08ULL, 0xDA4FBFC0E1108A8FULL,
    0x9C47D08FFB10D4B8ULL, 0x483ADA7726A3C465ULL
};

#define U256_ZERO(r) do{(r)[0]=0;(r)[1]=0;(r)[2]=0;(r)[3]=0;}while(0)
#define U256_IS_ZERO(r) (((r)[0]|(r)[1]|(r)[2]|(r)[3])==0)

static __device__ int u256_cmp(const uint64_t a[4], const uint64_t b[4]) {
    for(int i=3; i>=0; i--) {
        if(a[i] > b[i]) return 1;
        if(a[i] < b[i]) return -1;
    }
    return 0;
}

// a + b → r, with carry handling and mod p reduction
static __device__ void u256_add(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t sum[4], carry=0;
    for(int i=0; i<4; i++) {
        uint64_t t = a[i];
        sum[i] = t + b[i] + carry;
        carry = (sum[i] < t) || (carry && sum[i] == t) ? 1 : 0;
    }
    // Fast reduction: if overflow or >= P, subtract P
    if(carry || u256_cmp(sum, P) >= 0) {
        uint64_t borrow=0;
        for(int i=0; i<4; i++) {
            uint64_t t = sum[i];
            r[i] = t - P[i] - borrow;
            borrow = (borrow && t == 0) || (t < P[i] + borrow) ? 1 : 0;
        }
    } else {
        for(int i=0; i<4; i++) r[i] = sum[i];
    }
}

// a - b → r mod p (handles underflow by adding P)
static __device__ void u256_sub(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    uint64_t borrow=0;
    for(int i=0; i<4; i++) {
        uint64_t t = a[i];
        r[i] = t - b[i] - borrow;
        borrow = (borrow && t == 0) || (t < b[i] + borrow) ? 1 : 0;
    }
    if(borrow) {
        uint64_t carry=0;
        for(int i=0; i<4; i++) {
            uint64_t t = r[i];
            r[i] = t + P[i] + carry;
            carry = (r[i] < t) || (r[i] < P[i] + carry) ? 1 : 0;
        }
    }
}

// a * b → r mod p (schoolbook + fast reduction using p = 2^256 - 2^32 - 977)
static __device__ void u256_mul(const uint64_t a[4], const uint64_t b[4], uint64_t r[4]) {
    // 512-bit product d[8] using schoolbook 4x4
    // Use __uint128_t for 64×64→128 (supported in CUDA device code)
    __uint128_t d[8] = {0};
    for(int i=0; i<4; i++) {
        __uint128_t carry = 0;
        for(int j=0; j<4; j++) {
            __uint128_t prod = d[i+j] + (__uint128_t)a[i] * (__uint128_t)b[j] + carry;
            d[i+j] = prod & 0xFFFFFFFFFFFFFFFFULL;
            carry = prod >> 64;
        }
        d[i+4] = carry;
    }

    // Fast reduction: d mod p, where p = 2^256 - 0x1000003D1
    // d = d_lo + d_hi*2^256, and 2^256 ≡ 0x1000003D1 (mod p)
    // So: r ≡ d_lo + d_hi * 0x1000003D1  (mod p)
    // 0x1000003D1 = 2^32 + 977

    uint64_t t[5] = {(uint64_t)d[0], (uint64_t)d[1], (uint64_t)d[2], (uint64_t)d[3], 0};

    for(int i=0; i<4; i++) {
        uint64_t val = (uint64_t)d[4+i];
        if(val == 0) continue;
        // Add val<<32 at position i*64
        uint64_t vlo = val << 32, vhi = val >> 32;
        uint64_t carry = 0, sum;
        sum = t[i] + vlo; carry=(sum<t[i]||sum<vlo)?1:0; t[i]=sum;
        for(int k=i+1; k<5 && (carry||vhi); k++) {
            if(k==i+1){sum=t[k]+vhi+carry;carry=(sum<t[k])?1:0;t[k]=sum;vhi=0;}
            else{sum=t[k]+carry;carry=(sum<t[k])?1:0;t[k]=sum;}
        }
        // Add val*977 at position i*64
        __uint128_t prod = (__uint128_t)val * 977ULL;
        uint64_t plo=(uint64_t)prod, phi=(uint64_t)(prod>>64);
        carry=0;
        sum=t[i]+plo; carry=(sum<t[i])?1:0; t[i]=sum;
        for(int k=i+1; k<5 && (carry||phi); k++) {
            if(k==i+1){sum=t[k]+phi+carry;carry=(sum<t[k])?1:0;t[k]=sum;phi=0;}
            else{sum=t[k]+carry;carry=(sum<t[k])?1:0;t[k]=sum;}
        }
    }

    // Reduce overflow t[4] again
    if(t[4]) {
        uint64_t v = t[4];
        uint64_t vlo=v<<32, vhi=v>>32, carry, sum;
        sum=t[0]+vlo; carry=(sum<t[0])?1:0; t[0]=sum;
        sum=t[1]+vhi+carry; carry=(sum<t[1])?1:0; t[1]=sum;
        if(carry){t[2]++; if(t[2]==0)t[3]++;}

        __uint128_t pr = (__uint128_t)v * 977ULL;
        uint64_t pl=(uint64_t)pr, ph=(uint64_t)(pr>>64);
        sum=t[0]+pl; carry=(sum<t[0])?1:0; t[0]=sum;
        sum=t[1]+ph+carry; carry=(sum<t[1])?1:0; t[1]=sum;
        if(carry){t[2]++; if(t[2]==0)t[3]++;}
        t[4]=0;
    }

    r[0]=t[0]; r[1]=t[1]; r[2]=t[2]; r[3]=t[3];

    if(u256_cmp(r, P) >= 0) {
        uint64_t b=0;
        for(int i=0; i<4; i++) {
            uint64_t t=r[i];
            r[i]=t-P[i]-b;
            b=(b && t==0) || (t < P[i]+b) ? 1 : 0;
        }
    }
}

static __device__ void u256_sq(const uint64_t a[4], uint64_t r[4]) {
    u256_mul(a, a, r);
}

// Jacobian point
typedef struct { uint64_t x[4], y[4], z[4]; } JP;

// P = 2P (Jacobian doubling, a=0)
static __device__ void jp_double(const JP* P, JP* R) {
    if(U256_IS_ZERO(P->y)) { U256_ZERO(R->x);U256_ZERO(R->y);U256_ZERO(R->z); return; }
    uint64_t A[4],B[4],C[4],D[4],E[4],F[4],T[4];
    u256_sq(P->x, A);                         // A = X^2
    u256_sq(P->y, B);                         // B = Y^2
    u256_sq(B, C);                             // C = B^2
    u256_add(P->x, B, T); u256_sq(T, T);      // (X+B)^2
    u256_sub(T, A, D); u256_sub(D, C, D);     // D = (X+B)^2 - A - C
    u256_add(D, D, D);                         // D = 2*D
    u256_add(A, A, T); u256_add(T, A, E);     // E = 3*A
    u256_sq(E, F);                             // F = E^2
    u256_add(D, D, T);                         // 2*D
    u256_sub(F, T, R->x);                      // X3 = F - 2*D
    u256_sub(D, R->x, T);                      // D - X3
    u256_mul(E, T, T);                         // E*(D-X3)
    u256_add(C, C, A); u256_add(A, A, C);
    u256_add(C, C, C);                         // C = 8*C
    u256_sub(T, C, R->y);                      // Y3 = E*(D-X3) - 8*C
    u256_mul(P->y, P->z, T);                  // Y*Z
    u256_add(T, T, R->z);                      // Z3 = 2*Y*Z
}

// R = P + Q (general Jacobian addition)
static __device__ void jp_add(const JP* P, const JP* Q, JP* R) {
    if(U256_IS_ZERO(P->z)) { *R = *Q; return; }
    if(U256_IS_ZERO(Q->z)) { *R = *P; return; }

    uint64_t z1z1[4],z2z2[4],u1[4],u2[4],s1[4],s2[4],h[4],rr[4];
    u256_sq(P->z, z1z1); u256_sq(Q->z, z2z2);
    u256_mul(P->x, z2z2, u1); u256_mul(Q->x, z1z1, u2);
    u256_mul(P->y, z2z2, s1); u256_mul(s1, Q->z, s1);
    u256_mul(Q->y, z1z1, s2); u256_mul(s2, P->z, s2);
    u256_sub(u2, u1, h); u256_sub(s2, s1, rr);

    if(U256_IS_ZERO(h)) {
        if(U256_IS_ZERO(rr)) { jp_double(P, R); return; }
        U256_ZERO(R->x); U256_ZERO(R->y); U256_ZERO(R->z); return;
    }

    uint64_t h2[4],h3[4],u1h2[4],tmp[4];
    u256_sq(h, h2); u256_mul(h, h2, h3);
    u256_mul(u1, h2, u1h2);
    u256_sq(rr, tmp);                     // r^2
    u256_sub(tmp, h3, tmp);              // r^2 - H^3
    uint64_t D[4]; u256_add(u1h2, u1h2, D); // 2*U1*H^2
    u256_sub(tmp, D, R->x);              // X3 = r^2 - H^3 - 2*U1*H^2
    u256_sub(u1h2, R->x, tmp);           // U1*H^2 - X3
    u256_mul(rr, tmp, tmp);              // r*(U1*H^2 - X3)
    u256_mul(s1, h3, D);                // S1*H^3
    u256_sub(tmp, D, R->y);             // Y3
    u256_mul(h, P->z, tmp);
    u256_mul(tmp, Q->z, R->z);           // Z3
}

// Scalar = k (big-endian 32 bytes), P = base point → R = k*P
// Simple double-and-add, MSB first
static __device__ void jp_scalar_mult(const uint8_t k[32], const JP* base, JP* R) {
    U256_ZERO(R->x); U256_ZERO(R->y); U256_ZERO(R->z); // infinity
    JP Q = *base;
    int started = 0;
    for(int b=0; b<32; b++) {
        uint8_t byte = k[31 - b];
        for(int bit=7; bit>=0; bit--) {
            if(started) { JP tmp = *R; jp_double(&tmp, R); }
            if(byte & (1<<bit)) {
                if(started) { JP tmp = *R; jp_add(&tmp, &Q, R); }
                else { *R = Q; started = 1; }
            }
        }
    }
}

// Modular inverse via extended Euclidean (a^(p-2) mod p doesn't fit GPU well)
static __device__ void u256_inv(const uint64_t a[4], uint64_t r[4]) {
    uint64_t u[4], v[4], x1[4], x2[4];
    // u=a, v=p, x1=1, x2=0
    for(int i=0; i<4; i++) { u[i]=a[i]; v[i]=P[i]; }
    x1[0]=1; x1[1]=x1[2]=x1[3]=0;
    x2[0]=x2[1]=x2[2]=x2[3]=0;

    int iter=0;
    while(!U256_IS_ZERO(v) && iter < 600) {
        while(u256_cmp(u, v) >= 0 && !U256_IS_ZERO(v)) {
            u256_sub(u, v, u);
            u256_sub(x1, x2, x1);
        }
        // swap u↔v and x1↔x2
        uint64_t tmp[4];
        for(int i=0;i<4;i++){tmp[i]=u[i];u[i]=v[i];v[i]=tmp[i];}
        for(int i=0;i<4;i++){tmp[i]=x1[i];x1[i]=x2[i];x2[i]=tmp[i];}
        iter++;
    }
    for(int i=0; i<4; i++) r[i] = x1[i];
}

// Jacobian → compressed pubkey (33 bytes)
static __device__ void jp_to_compressed(const JP* P, uint8_t out[33]) {
    if(U256_IS_ZERO(P->z)) { for(int i=0;i<33;i++)out[i]=0; return; }

    uint64_t zi[4], zi2[4];
    u256_inv(P->z, zi);
    u256_sq(zi, zi2);

    uint64_t x[4], y[4];
    u256_mul(P->x, zi2, x);
    u256_mul(P->y, zi2, y);
    u256_mul(y, zi, y);

    out[0] = (y[0] & 1) ? 0x03 : 0x02;
    for(int i=0; i<32; i++) {
        int limb = (31 - i) / 8;
        int shift = ((31 - i) % 8) * 8;
        out[1 + i] = (uint8_t)(x[limb] >> shift);
    }
}

// Batch kernel: private_keys → compressed pubkeys (33 bytes each)
__global__ void secp256k1_batch_kernel_impl(
    const uint8_t* private_keys,
    uint8_t* pubkeys,
    uint32_t count
) {
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= count) return;

    JP G;
    G.x[0]=GX[0];G.x[1]=GX[1];G.x[2]=GX[2];G.x[3]=GX[3];
    G.y[0]=GY[0];G.y[1]=GY[1];G.y[2]=GY[2];G.y[3]=GY[3];
    G.z[0]=1;G.z[1]=G.z[2]=G.z[3]=0;

    JP R;
    jp_scalar_mult(private_keys + idx * 32, &G, &R);
    jp_to_compressed(&R, pubkeys + idx * 33);
}

#endif
