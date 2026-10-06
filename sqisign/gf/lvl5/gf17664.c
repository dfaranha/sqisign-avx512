#include <string.h>
#include "gf17664.h"

// sqi_fp_asm.S
void fp_add(gf17664 *out, const gf17664 *a, const gf17664 *b);
void fp_sub(gf17664 *out, const gf17664 *a, const gf17664 *b);
void fp_mul(gf17664 *out, const gf17664 *a, const gf17664 *b);
void fp_sqr(gf17664 *out, const gf17664 *a);

__extension__ typedef unsigned __int128 u128;

#define NLIMBS 11
#define NBYTES 84

// q = 17 * 2^664 - 1
static const gf17664 Q = { .arr = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
                                    0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
                                    0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x0000000010FFFFFF } };
// R^2 mod q
static const gf17664 R2 = { .arr = { 0xD3B597795B3D1F00, 0x0F0F0F0F0F0F0FF1, 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F,
                                     0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F,
                                     0x0F0F0F0F0F0F0F0F, 0x0F0F0F0F0F0F0F0F, 0x000000000F0F0F0F } };
// 1/2 and 1/3, Montgomery form
static const gf17664 TWO_INV = { .arr = { 0x0000000787878787, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0000000009000000 } };
static const gf17664 THREE_INV = { .arr = { 0xAAAAAAAFAFAFAFAF, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
                                            0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA,
                                            0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA, 0x000000000BAAAAAA } };
// The integer 1 (not in Montgomery form): multiplying by it leaves Montgomery form
static const gf17664 ONE_RAW = { .arr = { 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };

const gf17664 gf17664_ZERO = { .arr = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };
const gf17664 gf17664_ONE = { .arr = { 0x0000000F0F0F0F0F, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x0000000001000000 } };
const gf17664 gf17664_MINUS_ONE = { .arr = { 0xFFFFFFF0F0F0F0F0, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
                                             0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
                                             0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
                                             0xFFFFFFFFFFFFFFFF, 0x000000000FFFFFFF } };

static uint64_t
mask64(uint32_t ctl)
{
    return (uint64_t)(int64_t)(int32_t)ctl;
}

// w < 2^728 (12 limbs) -> d = w mod q, with d < 2^669.
// Writing w = h*2^664 + l and h = 17*s + r, we have w = s + r*2^664 + l mod q
// because 17*2^664 = 1 mod q; that sum is below 2^669.
static void
partial_reduce(gf17664 *d, const uint64_t *w)
{
    uint64_t h = (w[10] >> 24) | (w[11] << 40);
    uint64_t s = h / 17, r = h - 17 * s;

    u128 cc = (u128)w[0] + s;
    d->arr[0] = (uint64_t)cc;
    for (int i = 1; i < NLIMBS - 1; i++) {
        cc = (cc >> 64) + w[i];
        d->arr[i] = (uint64_t)cc;
    }
    d->arr[10] = (uint64_t)(cc >> 64) + (w[10] & 0xFFFFFF) + (r << 24);
}

// m <- canonical integer value of a, in [0, q)
static void
canonical(uint64_t *m, const gf17664 *a)
{
    gf17664 t;
    uint64_t s[NLIMBS];
    unsigned char b = 0;

    fp_mul(&t, a, &ONE_RAW); // out of Montgomery form; t < 2^669 < 2q
    for (int i = 0; i < NLIMBS; i++) {
        u128 u = (u128)t.arr[i] - Q.arr[i] - b;
        s[i] = (uint64_t)u;
        b = (u >> 64) & 1;
    }
    uint64_t keep = 0 - (uint64_t)b; // borrow means t < q
    for (int i = 0; i < NLIMBS; i++)
        m[i] = (t.arr[i] & keep) | (s[i] & ~keep);
}

void
gf17664_neg(gf17664 *d, const gf17664 *a)
{
    fp_sub(d, &gf17664_ZERO, a);
}

void
gf17664_half(gf17664 *d, const gf17664 *a)
{
    fp_mul(d, a, &TWO_INV);
}

void
gf17664_div3(gf17664 *d, const gf17664 *a)
{
    fp_mul(d, a, &THREE_INV);
}

void
gf17664_mul_small(gf17664 *d, const gf17664 *a, uint32_t x)
{
    uint64_t w[NLIMBS + 1];
    u128 cc = 0;

    for (int i = 0; i < NLIMBS; i++) {
        cc += (u128)a->arr[i] * x;
        w[i] = (uint64_t)cc;
        cc >>= 64;
    }
    w[NLIMBS] = (uint64_t)cc;
    partial_reduce(d, w);
}

void
gf17664_set_small(gf17664 *d, uint32_t x)
{
    gf17664 t = gf17664_ZERO;

    t.arr[0] = x;
    fp_mul(d, &t, &R2);
}

void
gf17664_select(gf17664 *d, const gf17664 *a0, const gf17664 *a1, uint32_t ctl)
{
    uint64_t m = mask64(ctl);

    for (int i = 0; i < NLIMBS; i++)
        d->arr[i] = a0->arr[i] ^ (m & (a0->arr[i] ^ a1->arr[i]));
}

void
gf17664_cswap(gf17664 *a, gf17664 *b, uint32_t ctl)
{
    uint64_t m = mask64(ctl);

    for (int i = 0; i < NLIMBS; i++) {
        uint64_t t = m & (a->arr[i] ^ b->arr[i]);
        a->arr[i] ^= t;
        b->arr[i] ^= t;
    }
}

// Since values are below 2^669 < 2q, zero is represented by either 0 or q.
uint32_t
gf17664_iszero(const gf17664 *a)
{
    uint64_t r0 = 0, rq = 0;

    for (int i = 0; i < NLIMBS; i++) {
        r0 |= a->arr[i];
        rq |= a->arr[i] ^ Q.arr[i];
    }
    uint32_t z0 = (uint32_t)((r0 | (0 - r0)) >> 63) - 1;
    uint32_t zq = (uint32_t)((rq | (0 - rq)) >> 63) - 1;
    return z0 | zq;
}

uint32_t
gf17664_equals(const gf17664 *a, const gf17664 *b)
{
    gf17664 t;

    fp_sub(&t, a, b);
    return gf17664_iszero(&t);
}

// d <- a^(2^k - 1), k >= 1
static void
pow_ones(gf17664 *d, const gf17664 *a, int k)
{
    gf17664 x = *a;
    int e = 1;

    for (int i = 62 - __builtin_clzll((unsigned long long)k); i >= 0; i--) {
        gf17664 y = x;
        for (int j = 0; j < e; j++)
            fp_sqr(&y, &y);
        fp_mul(&x, &y, &x);
        e <<= 1;
        if ((k >> i) & 1) {
            fp_sqr(&x, &x);
            fp_mul(&x, &x, a);
            e += 1;
        }
    }
    *d = x;
}

// (q - 3)/4 = 17*2^662 - 1 = 16*2^662 + (2^662 - 1)
void
gf17664_exp3div4(gf17664 *a)
{
    gf17664 m, z;

    pow_ones(&m, a, 662);
    fp_mul(&z, &m, a); // a^(2^662)
    for (int i = 0; i < 4; i++)
        fp_sqr(&z, &z); // a^(16*2^662)
    fp_mul(a, &z, &m);
}

uint32_t
gf17664_invert(gf17664 *d, const gf17664 *a)
{
    gf17664 a0 = *a, t = *a;
    uint32_t nz = ~gf17664_iszero(&a0);

    gf17664_exp3div4(&t);
    fp_sqr(&t, &t);
    fp_sqr(&t, &t);
    fp_mul(d, &t, &a0); // a^(q-2)
    return nz;
}

int32_t
gf17664_legendre(const gf17664 *a)
{
    gf17664 t = *a;

    gf17664_exp3div4(&t);
    fp_sqr(&t, &t);
    fp_mul(&t, &t, a); // a^((q-1)/2)
    uint32_t one = gf17664_equals(&t, &gf17664_ONE);
    uint32_t minus_one = gf17664_equals(&t, &gf17664_MINUS_ONE);
    return (int32_t)(one & 1) - (int32_t)(minus_one & 1);
}

uint32_t
gf17664_sqrt(gf17664 *d, const gf17664 *a)
{
    gf17664 a0 = *a, y = *a, y2, ny;
    uint64_t m[NLIMBS];

    // y = a^((q+1)/4); since q = 3 mod 4, y^2 = a if a is a square, and -a otherwise
    gf17664_exp3div4(&y);
    fp_mul(&y, &y, &a0);
    fp_sqr(&y2, &y);
    uint32_t ok = gf17664_equals(&y2, &a0);

    canonical(m, &y);
    fp_sub(&ny, &gf17664_ZERO, &y);
    gf17664_select(d, &y, &ny, -(uint32_t)(m[0] & 1));
    return ok;
}

void
gf17664_encode(void *dst, const gf17664 *a)
{
    uint64_t m[NLIMBS];
    uint8_t *buf = dst;

    canonical(m, a);
    for (int i = 0; i < NBYTES; i++)
        buf[i] = (uint8_t)(m[i >> 3] >> (8 * (i & 7)));
}

uint32_t
gf17664_decode(gf17664 *d, const void *src)
{
    const uint8_t *buf = src;
    gf17664 v = gf17664_ZERO;
    unsigned char b = 0;

    for (int i = 0; i < NBYTES; i++)
        v.arr[i >> 3] |= (uint64_t)buf[i] << (8 * (i & 7));
    for (int i = 0; i < NLIMBS; i++) {
        u128 u = (u128)v.arr[i] - Q.arr[i] - b;
        b = (u >> 64) & 1;
    }
    uint32_t ok = (uint32_t)0 - (uint32_t)b; // borrow means v < q

    for (int i = 0; i < NLIMBS; i++)
        v.arr[i] &= mask64(ok);
    fp_mul(d, &v, &R2);
    return ok;
}

// Horner over 88-byte blocks, most significant first: acc <- acc*2^704 + block
void
gf17664_decode_reduce(gf17664 *d, const void *src, size_t len)
{
    const uint8_t *buf = src;
    const size_t B = 8 * NLIMBS;
    gf17664 acc = gf17664_ZERO;

    if (len == 0) {
        *d = acc;
        return;
    }
    size_t nb = (len + B - 1) / B;
    for (size_t blk = nb; blk-- > 0;) {
        size_t off = blk * B;
        size_t n = (off + B <= len) ? B : len - off;
        uint64_t w[NLIMBS + 1];
        gf17664 limb;

        memset(w, 0, sizeof w);
        for (size_t i = 0; i < n; i++)
            w[i >> 3] |= (uint64_t)buf[off + i] << (8 * (i & 7));
        partial_reduce(&limb, w);
        if (blk == nb - 1) {
            acc = limb;
        } else {
            gf17664 sh;
            fp_mul(&sh, &acc, &R2); // acc * 2^704 mod q
            fp_add(&acc, &sh, &limb);
        }
    }
    fp_mul(d, &acc, &R2); // to Montgomery form
}
