// secp256k1 CUDA kernel — 32-bit limb arithmetic (fast on consumer GPUs)
// 256-bit numbers stored as 8 × uint32_t (little-endian)
// Fast reduction exploits p = 2^256 - 2^32 - 977
// Jacobian coordinates, no per-key modular inverse (deferred to batch inversion)
#ifndef SECP256K1_32_CUH
#define SECP256K1_32_CUH

#include <cstdint>

#define NLIMBS 8

static __device__ const uint32_t P32[NLIMBS] = {
    0xFFFFFC2F, 0xFFFFFFFE, 0xFFFFFFFF, 0xFFFFFFFF,
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

static __device__ const uint32_t GX32[NLIMBS] = {
    0xB59F2815, 0x16F81798, 0xD959F282, 0x02CEDCE2,
    0xCE870B07, 0x55A06295, 0xF9DCBBAC, 0x79BE667E
};

static __device__ const uint32_t GY32[NLIMBS] = {
    0x8C47D08F, 0x0FFB10D4, 0xE1108A8F, 0xDA4FBFC0,
    0x8FFB10D4, 0x9C47D08F, 0xA3C4655D, 0x483ADA77
};

typedef uint32_t fe[NLIMBS];
typedef struct { fe x, y, z; } jp;

// u32_cmp: returns 1 if a>b, -1 if a<b, 0 if equal
static __device__ int fe_cmp(const fe a, const fe b) {
    for(int i=NLIMBS-1; i>=0; i--) {
        if(a[i] > b[i]) return 1;
        if(a[i] < b[i]) return -1;
    }
    return 0;
}

static __device__ int fe_is_zero(const fe a) {
    for(int i=0; i<NLIMBS; i++) if(a[i]) return 0;
    return 1;
}

// fe_add: r = (a + b) mod p
static __device__ void fe_add(const fe a, const fe b, fe r) {
    uint64_t sum = 0;
    for(int i=0; i<NLIMBS; i++) {
        sum += (uint64_t)a[i] + b[i];
        r[i] = (uint32_t)sum;
        sum >>= 32;
    }
    if(sum || fe_cmp(r, P32) >= 0) {
        int64_t diff = 0;
        for(int i=0; i<NLIMBS; i++) {
            diff += (int64_t)r[i] - (int64_t)P32[i];
            r[i] = (uint32_t)(diff);
            diff >>= 32;
        }
    }
}

// fe_sub: r = (a - b) mod p
static __device__ void fe_sub(const fe a, const fe b, fe r) {
    int64_t diff = 0;
    for(int i=0; i<NLIMBS; i++) {
        diff += (int64_t)a[i] - (int64_t)b[i];
        r[i] = (uint32_t)diff;
        diff >>= 32;
    }
    if(diff) {
        uint64_t sum = 0;
        for(int i=0; i<NLIMBS; i++) {
            sum += (uint64_t)r[i] + P32[i];
            r[i] = (uint32_t)sum;
            sum >>= 32;
        }
    }
}

// fe_mul: r = a * b mod p (schoolbook 8×8 + fast reduction)
static __device__ void fe_mul(const fe a, const fe b, fe r) {
    uint64_t d[16] = {0};
    for(int i=0; i<NLIMBS; i++) {
        for(int j=0; j<NLIMBS; j++) {
            uint64_t prod = (uint64_t)a[i] * (uint64_t)b[j];
            uint64_t sum = d[i+j] + prod;
            d[i+j] = (uint32_t)sum;
            d[i+j+1] += sum >> 32;
            for(int k=i+j+1; k<15 && (d[k] >> 32); k++) {
                d[k+1] += d[k] >> 32;
                d[k] = (uint32_t)d[k];
            }
        }
    }

    // Fast reduction: p = 2^256 - 2^32 - 977
    uint64_t t[9] = {d[0], d[1], d[2], d[3], d[4], d[5], d[6], d[7], 0};

    for(int i=0; i<8; i++) {
        uint64_t val = d[8+i];
        if(val == 0) continue;

        uint64_t vlo = (val & 0xFFFFFFFFULL) << 32;
        uint64_t vhi = val >> 32;
        uint64_t p977 = val * 977ULL;

        t[i] += vlo;
        if(t[i] >> 32) { t[i+1] += t[i] >> 32; t[i] &= 0xFFFFFFFFULL; }

        if(i+1 < 9) { t[i+1] += vhi; if(t[i+1] >> 32 && i+2 < 9) { t[i+2] += t[i+1] >> 32; t[i+1] &= 0xFFFFFFFFULL; }}

        t[i] += p977;
        uint64_t carry = t[i] >> 32; t[i] &= 0xFFFFFFFFULL;
        for(int k=i+1; k<9 && carry; k++) {
            t[k] += carry;
            carry = t[k] >> 32; t[k] &= 0xFFFFFFFFULL;
        }
    }

    if(t[8]) {
        uint64_t val = t[8]; t[8] = 0;
        uint64_t vlo = (val & 0xFFFFFFFFULL) << 32;
        uint64_t vhi = val >> 32;
        uint64_t p977 = val * 977ULL;

        t[0] += vlo; if(t[0]>>32){t[1]+=t[0]>>32;t[0]&=0xFFFFFFFFULL;}
        t[1] += vhi; if(t[1]>>32){t[2]+=t[1]>>32;t[1]&=0xFFFFFFFFULL;}
        t[0] += p977;
        uint64_t carry = t[0] >> 32; t[0] &= 0xFFFFFFFFULL;
        for(int k=1; k<9 && carry; k++) {
            t[k] += carry;
            carry = t[k] >> 32; t[k] &= 0xFFFFFFFFULL;
        }
    }

    for(int i=0; i<NLIMBS; i++) r[i] = (uint32_t)t[i];

    if(fe_cmp(r, P32) >= 0) {
        int64_t diff = 0;
        for(int i=0; i<NLIMBS; i++) {
            diff += (int64_t)r[i] - (int64_t)P32[i];
            r[i] = (uint32_t)diff;
            diff >>= 32;
        }
    }
}

// fe_sq: r = a^2 mod p
static __device__ void fe_sq(const fe a, fe r) {
    fe_mul(a, a, r);
}

// fe_set: r = a
static __device__ void fe_set(fe r, const fe a) {
    for(int i=0; i<NLIMBS; i++) r[i] = a[i];
}

// fe_inv: r = 1/a mod p (extended Euclidean, max ~600 iterations)
static __device__ void fe_inv(const fe a, fe r) {
    fe u, v, x1, x2;
    for(int i=0; i<NLIMBS; i++) { u[i]=a[i]; v[i]=P32[i]; }
    x1[0]=1; for(int i=1;i<NLIMBS;i++) x1[i]=0;
    for(int i=0;i<NLIMBS;i++) x2[i]=0;

    int iter=0;
    while(!fe_is_zero(v) && iter < 600) {
        while(fe_cmp(u, v) >= 0 && !fe_is_zero(v)) {
            fe_sub(u, v, u);
            fe_sub(x1, x2, x1);
        }
        fe tmp;
        for(int i=0;i<NLIMBS;i++) tmp[i]=u[i];
        for(int i=0;i<NLIMBS;i++) u[i]=v[i];
        for(int i=0;i<NLIMBS;i++) v[i]=tmp[i];
        for(int i=0;i<NLIMBS;i++) tmp[i]=x1[i];
        for(int i=0;i<NLIMBS;i++) x1[i]=x2[i];
        for(int i=0;i<NLIMBS;i++) x2[i]=tmp[i];
        iter++;
    }
    for(int i=0; i<NLIMBS; i++) r[i] = x1[i];
}

// fe_set_one: r = 1
static __device__ void fe_set_one(fe r) {
    r[0] = 1; for(int i=1; i<NLIMBS; i++) r[i] = 0;
}

// fe_set_zero: r = 0
static __device__ void fe_set_zero(fe r) {
    for(int i=0; i<NLIMBS; i++) r[i] = 0;
}

// ============================================================
// Point operations in Jacobian coordinates (a=0 for secp256k1)
// ============================================================

// R = 2*P (Jacobian doubling, a=0)
static __device__ void jp_double_32(const jp* P, jp* R) {
    if(fe_is_zero(P->y)) {
        fe_set_zero(R->x); fe_set_zero(R->y); fe_set_zero(R->z); return;
    }
    fe t1, t2, t3, t4;
    fe A, B, C, D, E, F;

    fe_sq(P->x, A);
    fe_sq(P->y, B);
    fe_sq(B, C);
    fe_add(P->x, B, t1);
    fe_sq(t1, t1);
    fe_sub(t1, A, t1);
    fe_sub(t1, C, D);
    fe_add(D, D, D);
    fe_add(A, A, t1); fe_add(t1, A, E);
    fe_sq(E, F);
    fe_add(D, D, t1);
    fe_sub(F, t1, R->x);
    fe_sub(D, R->x, t1);
    fe_mul(E, t1, t1);
    fe_add(C, C, t2); fe_add(t2, t2, t3); fe_add(t3, t3, C);
    fe_sub(t1, C, R->y);
    fe_mul(P->y, P->z, t1);
    fe_add(t1, t1, R->z);
}

// R = P + Q (Jacobian addition)
static __device__ void jp_add_32(const jp* P, const jp* Q, jp* R) {
    if(fe_is_zero(P->z)) { *R = *Q; return; }
    if(fe_is_zero(Q->z)) { *R = *P; return; }

    fe z1z1, z2z2, u1, u2, s1, s2, h, r_val;
    fe_sq(P->z, z1z1); fe_sq(Q->z, z2z2);
    fe_mul(P->x, z2z2, u1); fe_mul(Q->x, z1z1, u2);
    fe_mul(P->y, z2z2, s1); fe_mul(s1, Q->z, s1);
    fe_mul(Q->y, z1z1, s2); fe_mul(s2, P->z, s2);
    fe_sub(u2, u1, h); fe_sub(s2, s1, r_val);

    if(fe_is_zero(h)) {
        if(fe_is_zero(r_val)) { jp_double_32(P, R); return; }
        fe_set_zero(R->x); fe_set_zero(R->y); fe_set_zero(R->z); return;
    }

    fe h2, h3, u1h2, tmp;
    fe_sq(h, h2); fe_mul(h, h2, h3);
    fe_mul(u1, h2, u1h2);
    fe_sq(r_val, tmp);
    fe_sub(tmp, h3, tmp);
    fe t2d; fe_add(u1h2, u1h2, t2d);
    fe_sub(tmp, t2d, R->x);
    fe_sub(u1h2, R->x, tmp);
    fe_mul(r_val, tmp, tmp);
    fe_mul(s1, h3, t2d);
    fe_sub(tmp, t2d, R->y);
    fe_mul(h, P->z, tmp);
    fe_mul(tmp, Q->z, R->z);
}

// ============================================================
// 8-bit scalar multiplication
// Process bytes MSB-first, double 8x per byte, add table entry
// ============================================================
static __device__ void jp_mul_table(
    const uint8_t k[32],
    const uint32_t table_x[256 * NLIMBS],
    const uint32_t table_y[256 * NLIMBS],
    jp* R
) {
    fe_set_zero(R->x); fe_set_zero(R->y); fe_set_zero(R->z);
    jp T;

    for(int b=0; b<32; b++) {
        uint8_t idx = k[b];

        if(!fe_is_zero(R->z)) {
            for(int d=0; d<8; d++) {
                jp tmp = *R;
                jp_double_32(&tmp, R);
            }
        }

        if(idx == 0) continue;

        int offset = (int)idx * NLIMBS;
        for(int i=0; i<NLIMBS; i++) {
            T.x[i] = table_x[offset + NLIMBS - 1 - i];
            T.y[i] = table_y[offset + NLIMBS - 1 - i];
        }
        fe_set_one(T.z);

        if(fe_is_zero(R->z)) {
            *R = T;
        } else {
            jp tmp = *R;
            jp_add_32(&tmp, &T, R);
        }
    }
}

#endif // SECP256K1_32_CUH
