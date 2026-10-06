/*
 * GF(q) for q = 3 * 2^324 - 1 (326 bits, 6 limbs).
 *
 * Elements are kept in Montgomery representation with R = 2^384 and are
 * only partially reduced (value < 2^326, not necessarily < q), which is the
 * domain of the fp_add/fp_sub/fp_mul/fp_sqr kernels in sqi_fp_asm.S.
 * The algorithms follow the generic sat64 backend of the SQIsign v3
 * reference implementation, specialised to this prime.
 *
 * Boolean values are uint32_t equal to 0xFFFFFFFF (true) or 0 (false).
 */

#ifndef gf3324_h__
#define gf3324_h__

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>
#include <stdint.h>
#include "tutil.h"

    typedef union
    {
        struct
        {
            uint64_t v0;
            uint64_t v1;
            uint64_t v2;
            uint64_t v3;
            uint64_t v4;
            uint64_t v5;
        };
        digit_t arr[6];
    } gf3324;

    extern const gf3324 gf3324_ZERO;
    extern const gf3324 gf3324_ONE;
    extern const gf3324 gf3324_MINUS_ONE;

    void gf3324_neg(gf3324 *d, const gf3324 *a);
    void gf3324_half(gf3324 *d, const gf3324 *a);
    void gf3324_div3(gf3324 *d, const gf3324 *a);
    void gf3324_mul_small(gf3324 *d, const gf3324 *a, uint32_t x);
    void gf3324_set_small(gf3324 *d, uint32_t x);

    // d <- a1 if ctl == 0xFFFFFFFF, d <- a0 if ctl == 0
    void gf3324_select(gf3324 *d, const gf3324 *a0, const gf3324 *a1, uint32_t ctl);
    // swap *a and *b if ctl == 0xFFFFFFFF
    void gf3324_cswap(gf3324 *a, gf3324 *b, uint32_t ctl);

    uint32_t gf3324_iszero(const gf3324 *a);
    uint32_t gf3324_equals(const gf3324 *a, const gf3324 *b);

    // a <- a^((q-3)/4)
    void gf3324_exp3div4(gf3324 *a);

    // d <- 1/a; returns 0xFFFFFFFF, or 0 (and d = 0) if a is zero
    uint32_t gf3324_invert(gf3324 *d, const gf3324 *a);

    // 0 for zero, +1 for a non-zero square, -1 for a non-square
    int32_t gf3324_legendre(const gf3324 *a);

    // If a is a square, d <- sqrt(a) and returns 0xFFFFFFFF; otherwise
    // d <- sqrt(-a) and returns 0. The root written into d is the one whose
    // integer representation in [0, q) is even.
    uint32_t gf3324_sqrt(gf3324 *d, const gf3324 *a);

    // Exactly 41 bytes, little-endian, canonical value in [0, q)
    void gf3324_encode(void *dst, const gf3324 *a);
    // Decodes exactly 41 bytes; a non-canonical input yields d = 0 and returns 0
    uint32_t gf3324_decode(gf3324 *d, const void *src);
    // Little-endian integer of len bytes, reduced modulo q
    void gf3324_decode_reduce(gf3324 *d, const void *src, size_t len);

#ifdef __cplusplus
}
#endif

#endif
