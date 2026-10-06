#include <string.h>
#include "gf3324.h"

// sqi_fp_asm.S
void fp_add(gf3324 *out, const gf3324 *a, const gf3324 *b);
void fp_sub(gf3324 *out, const gf3324 *a, const gf3324 *b);
void fp_mul(gf3324 *out, const gf3324 *a, const gf3324 *b);
void fp_sqr(gf3324 *out, const gf3324 *a);

__extension__ typedef unsigned __int128 u128;

#define NLIMBS 6
#define NBYTES 41

// q = 3 * 2^324 - 1
static const gf3324 Q = { .arr = { 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
                                   0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x000000000000002F } };
// R^2 mod q
static const gf3324 R2 = { .arr = { 0xC71C71C71C71C71C, 0x5571C71C71C71C71, 0x5555555555555555,
                                    0x5555555555555555, 0x5555555555555555, 0x0000000000000015 } };
// 1/2 and 1/3, Montgomery form
static const gf3324 TWO_INV = { .arr = { 0x02AAAAAAAAAAAAAA, 0x0000000000000000, 0x0000000000000000,
                                         0x0000000000000000, 0x0000000000000000, 0x0000000000000020 } };
static const gf3324 THREE_INV = { .arr = { 0x571C71C71C71C71C, 0x5555555555555555, 0x5555555555555555,
                                           0x5555555555555555, 0x5555555555555555, 0x0000000000000015 } };
// The integer 1 (not in Montgomery form): multiplying by it leaves Montgomery form
static const gf3324 ONE_RAW = { .arr = { 1, 0, 0, 0, 0, 0 } };

const gf3324 gf3324_ZERO = { .arr = { 0, 0, 0, 0, 0, 0 } };
const gf3324 gf3324_ONE = { .arr = { 0x0555555555555555, 0x0000000000000000, 0x0000000000000000,
                                     0x0000000000000000, 0x0000000000000000, 0x0000000000000010 } };
const gf3324 gf3324_MINUS_ONE = { .arr = { 0xFAAAAAAAAAAAAAAA, 0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF,
                                           0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0x000000000000001F } };

static uint64_t
mask64(uint32_t ctl)
{
    return (uint64_t)(int64_t)(int32_t)ctl;
}

// w < 2^388 (7 limbs) -> d = w mod q, with d < 2^326.
// Writing w = h*2^324 + l and h = 3*s + r, we have w = s + r*2^324 + l mod q
// because 3*2^324 = 1 mod q; that sum is below 2^326.
static void
partial_reduce(gf3324 *d, const uint64_t *w)
{
    uint64_t h = (w[5] >> 4) | (w[6] << 60);
    uint64_t s = h / 3, r = h - 3 * s;

    u128 cc = (u128)w[0] + s;
    d->arr[0] = (uint64_t)cc;
    for (int i = 1; i < NLIMBS - 1; i++) {
        cc = (cc >> 64) + w[i];
        d->arr[i] = (uint64_t)cc;
    }
    d->arr[5] = (uint64_t)(cc >> 64) + (w[5] & 0xF) + (r << 4);
}

// m <- canonical integer value of a, in [0, q)
static void
canonical(uint64_t *m, const gf3324 *a)
{
    gf3324 t;
    uint64_t s[NLIMBS];
    unsigned char b = 0;

    fp_mul(&t, a, &ONE_RAW); // out of Montgomery form; t < 2^326 < 2q
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
gf3324_neg(gf3324 *d, const gf3324 *a)
{
    fp_sub(d, &gf3324_ZERO, a);
}

void
gf3324_half(gf3324 *d, const gf3324 *a)
{
    fp_mul(d, a, &TWO_INV);
}

void
gf3324_div3(gf3324 *d, const gf3324 *a)
{
    fp_mul(d, a, &THREE_INV);
}

void
gf3324_mul_small(gf3324 *d, const gf3324 *a, uint32_t x)
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
gf3324_set_small(gf3324 *d, uint32_t x)
{
    gf3324 t = gf3324_ZERO;

    t.arr[0] = x;
    fp_mul(d, &t, &R2);
}

void
gf3324_select(gf3324 *d, const gf3324 *a0, const gf3324 *a1, uint32_t ctl)
{
    uint64_t m = mask64(ctl);

    for (int i = 0; i < NLIMBS; i++)
        d->arr[i] = a0->arr[i] ^ (m & (a0->arr[i] ^ a1->arr[i]));
}

void
gf3324_cswap(gf3324 *a, gf3324 *b, uint32_t ctl)
{
    uint64_t m = mask64(ctl);

    for (int i = 0; i < NLIMBS; i++) {
        uint64_t t = m & (a->arr[i] ^ b->arr[i]);
        a->arr[i] ^= t;
        b->arr[i] ^= t;
    }
}

// Since values are below 2^326 < 2q, zero is represented by either 0 or q.
uint32_t
gf3324_iszero(const gf3324 *a)
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
gf3324_equals(const gf3324 *a, const gf3324 *b)
{
    gf3324 t;

    fp_sub(&t, a, b);
    return gf3324_iszero(&t);
}

// d <- a^(2^k - 1), k >= 1
static void
pow_ones(gf3324 *d, const gf3324 *a, int k)
{
    gf3324 x = *a;
    int e = 1;

    for (int i = 62 - __builtin_clzll((unsigned long long)k); i >= 0; i--) {
        gf3324 y = x;
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

// (q - 3)/4 = 3*2^322 - 1 = 2*2^322 + (2^322 - 1)
void
gf3324_exp3div4(gf3324 *a)
{
    gf3324 m, z;

    pow_ones(&m, a, 322);
    fp_mul(&z, &m, a); // a^(2^322)
    fp_sqr(&z, &z);    // a^(2^323)
    fp_mul(a, &z, &m);
}

uint32_t
gf3324_invert(gf3324 *d, const gf3324 *a)
{
    gf3324 a0 = *a, t = *a;
    uint32_t nz = ~gf3324_iszero(&a0);

    gf3324_exp3div4(&t);
    fp_sqr(&t, &t);
    fp_sqr(&t, &t);
    fp_mul(d, &t, &a0); // a^(q-2)
    return nz;
}

int32_t
gf3324_legendre(const gf3324 *a)
{
    gf3324 t = *a;

    gf3324_exp3div4(&t);
    fp_sqr(&t, &t);
    fp_mul(&t, &t, a); // a^((q-1)/2)
    uint32_t one = gf3324_equals(&t, &gf3324_ONE);
    uint32_t minus_one = gf3324_equals(&t, &gf3324_MINUS_ONE);
    return (int32_t)(one & 1) - (int32_t)(minus_one & 1);
}

uint32_t
gf3324_sqrt(gf3324 *d, const gf3324 *a)
{
    gf3324 a0 = *a, y = *a, y2, ny;
    uint64_t m[NLIMBS];

    // y = a^((q+1)/4); since q = 3 mod 4, y^2 = a if a is a square, and -a otherwise
    gf3324_exp3div4(&y);
    fp_mul(&y, &y, &a0);
    fp_sqr(&y2, &y);
    uint32_t ok = gf3324_equals(&y2, &a0);

    canonical(m, &y);
    fp_sub(&ny, &gf3324_ZERO, &y);
    gf3324_select(d, &y, &ny, -(uint32_t)(m[0] & 1));
    return ok;
}

void
gf3324_encode(void *dst, const gf3324 *a)
{
    uint64_t m[NLIMBS];
    uint8_t *buf = dst;

    canonical(m, a);
    for (int i = 0; i < NBYTES; i++)
        buf[i] = (uint8_t)(m[i >> 3] >> (8 * (i & 7)));
}

uint32_t
gf3324_decode(gf3324 *d, const void *src)
{
    const uint8_t *buf = src;
    gf3324 v = gf3324_ZERO;
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

// Horner over 48-byte blocks, most significant first: acc <- acc*2^384 + block
void
gf3324_decode_reduce(gf3324 *d, const void *src, size_t len)
{
    const uint8_t *buf = src;
    const size_t B = 8 * NLIMBS;
    gf3324 acc = gf3324_ZERO;

    if (len == 0) {
        *d = acc;
        return;
    }
    size_t nb = (len + B - 1) / B;
    for (size_t blk = nb; blk-- > 0;) {
        size_t off = blk * B;
        size_t n = (off + B <= len) ? B : len - off;
        uint64_t w[NLIMBS + 1];
        gf3324 limb;

        memset(w, 0, sizeof w);
        for (size_t i = 0; i < n; i++)
            w[i >> 3] |= (uint64_t)buf[off + i] << (8 * (i & 7));
        partial_reduce(&limb, w);
        if (blk == nb - 1) {
            acc = limb;
        } else {
            gf3324 sh;
            fp_mul(&sh, &acc, &R2); // acc * 2^384 mod q
            fp_add(&acc, &sh, &limb);
        }
    }
    fp_mul(d, &acc, &R2); // to Montgomery form
}
