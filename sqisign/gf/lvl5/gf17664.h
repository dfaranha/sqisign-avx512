/*
 * GF(q) for q = 17 * 2^664 - 1 (669 bits, 11 limbs).
 *
 * Elements are kept in Montgomery representation with R = 2^704 and are
 * only partially reduced (value < 2^669, not necessarily < q), which is the
 * domain of the fp_add/fp_sub/fp_mul/fp_sqr kernels in sqi_fp_asm.S.
 * The algorithms follow the generic sat64 backend of the SQIsign v3
 * reference implementation, specialised to this prime.
 *
 * Boolean values are uint32_t equal to 0xFFFFFFFF (true) or 0 (false).
 */

#ifndef gf17664_h__
#define gf17664_h__

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
            uint64_t v6;
            uint64_t v7;
            uint64_t v8;
            uint64_t v9;
            uint64_t v10;
        };
        digit_t arr[11];
    } gf17664;

    extern const gf17664 gf17664_ZERO;
    extern const gf17664 gf17664_ONE;
    extern const gf17664 gf17664_MINUS_ONE;

    void gf17664_neg(gf17664 *d, const gf17664 *a);
    void gf17664_half(gf17664 *d, const gf17664 *a);
    void gf17664_div3(gf17664 *d, const gf17664 *a);
    void gf17664_mul_small(gf17664 *d, const gf17664 *a, uint32_t x);
    void gf17664_set_small(gf17664 *d, uint32_t x);

    // d <- a1 if ctl == 0xFFFFFFFF, d <- a0 if ctl == 0
    void gf17664_select(gf17664 *d, const gf17664 *a0, const gf17664 *a1, uint32_t ctl);
    // swap *a and *b if ctl == 0xFFFFFFFF
    void gf17664_cswap(gf17664 *a, gf17664 *b, uint32_t ctl);

    uint32_t gf17664_iszero(const gf17664 *a);
    uint32_t gf17664_equals(const gf17664 *a, const gf17664 *b);

    // a <- a^((q-3)/4)
    void gf17664_exp3div4(gf17664 *a);

    // d <- 1/a; returns 0xFFFFFFFF, or 0 (and d = 0) if a is zero
    uint32_t gf17664_invert(gf17664 *d, const gf17664 *a);

    // 0 for zero, +1 for a non-zero square, -1 for a non-square
    int32_t gf17664_legendre(const gf17664 *a);

    // If a is a square, d <- sqrt(a) and returns 0xFFFFFFFF; otherwise
    // d <- sqrt(-a) and returns 0. The root written into d is the one whose
    // integer representation in [0, q) is even.
    uint32_t gf17664_sqrt(gf17664 *d, const gf17664 *a);

    // Exactly 84 bytes, little-endian, canonical value in [0, q)
    void gf17664_encode(void *dst, const gf17664 *a);
    // Decodes exactly 84 bytes; a non-canonical input yields d = 0 and returns 0
    uint32_t gf17664_decode(gf17664 *d, const void *src);
    // Little-endian integer of len bytes, reduced modulo q
    void gf17664_decode_reduce(gf17664 *d, const void *src, size_t len);

#ifdef __cplusplus
}
#endif

#endif
