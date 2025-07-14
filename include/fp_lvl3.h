/**
 *   The SQISign lvl3 prime is p = 65 * 2^376 - 1
 *   It is 383 bits, so the number of 51-bit limbs needed is 383/51 < 8
 *   
 *   Formats: a x b = a vectors of b Fp limbs
 */

#ifndef _FP_LVL3_H
#define _FP_LVL3_H

#include <stdint.h>

#include "intrinsics_avx512.h"

static const uint64_t LIMB_LEN = 51;
static const uint64_t LIMB_MASK = 0x7ffffffffffff;
static const uint64_t LAST_LIMB_MASK = 0x3ffffff;
// p in radix 2^51
static const uint64_t p[8] = { 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x207ffff };

static const uint64_t p_plus_1_top = 0x2080000;
// 2*p in radix 2^51
static const uint64_t p2[8] = { 0x7fffffffffffe, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x40fffff };

// We use the Montgomery representation with R = 2^408
// R2 is R^2 mod p, which is needed to convert to the representation.
static const uint64_t R2[8] = { 0x3732f52771f91, 0x7e07e07e07e2, 0x40fc0fc0fc0fc, 0x781f81f81f81f, 0x3f03f03f03f03, 0x7e07e07e07e0, 0x40fc0fc0fc0fc, 0xf9f81f };

// ======================================================================================================
//                                                 8x1
// ======================================================================================================

// 8x1 = 8 Fp elements, 1 limb per vector.

void fp_setzero_8x1(__m512i* res);

void fp_copy_8x1(__m512i* res, __m512i* a);

/**
 * @param a - Fp element stored in 51-bit limbs
 * @result 8 copies of a, stored in res
 */
void fp_duplicate_8x1(__m512i* res, const uint64_t* a);

/**
 *   Returns an 8-bit mask which is 1 iff the i-th Fp elements are equal
 */
int fp_is_equal_8x1(__m512i* a, __m512i* b);

/**
 *  res = a + b (mod p)
 */
void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 *  res = a - b (mod p)
 */
void fp_sub_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 *  res = -a (mod p)
 */
void fp_neg_8x1(__m512i* res, __m512i* a);

/**
 * If the i-th bit of add_mask is 1, res[i] = a[i] + b[i] mod p, else res[i] = a[i] - b[i] mod p
 */
void fp_mixaddsub_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

/**
 *  res = a * b (mod p)
 */
void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a^2 (mod p)
 */
void fp_sqr_8x1(__m512i* res, __m512i* a);

/**
 * res = a * R (mod p)
 */
void fp_tomont_8x1(__m512i* res, __m512i* a);

/**
 * res = a * R^(-1) (mod p)
 */
void fp_frommont_8x1(__m512i*res, __m512i* a);

// ======================================================================================================
//                                                 4x2
// ======================================================================================================

/*  4x2 = 4 Fp elements a, b, c, d, interleaved as follows:
 *  v0 = (a0, a4, b0, b4, c0, c4, d0, d4)
 *  v1 = (a1, a5, b1, b5, c1, c5, d1, d5)
 *  v2 = (a2, a6, b2, a6, c2, c6, d2, d6)
 *  v3 = (a3, a7, b3, b7, c3, c7, d3, d7)
 */

void fp_setzero_4x2(__m512i* res);
void fp_copy_4x2(__m512i* res, __m512i* a);


/**
 * @param a - Fp element stored in 51-bit limbs
 * @result 4 copies of a, stored in res
 */
void fp_duplicate_4x2(__m512i* res, const uint64_t* a);

/**
 *   Returns a 4-bit mask which is 1 iff the i-th Fp elements are equal
 */
int fp_is_equal_4x2(__m512i* a, __m512i* b);

void fp_add_4x2(__m512i* res, __m512i* a, __m512i* b);

void fp_sub_4x2(__m512i* res, __m512i* a, __m512i* b);

void fp_neg_4x2(__m512i* res, __m512i* a);

/**
 *  Denote the 4 Fp elements stored in a by a0, a1, a2, a3, and in b by b0, b1, b2, b3.
 *  Computes a0 - b0, a1 + b1, a2 - b2, a3 + b3.
 */
 void fp_mixsubadd_4x2(__m512i* res, __m512i* a, __m512i* b);

void fp_mul_4x2(__m512i* res, __m512i* a, __m512i* b);

void fp_sqr_4x2(__m512i* res, __m512i* a);

void fp_tomont_4x2(__m512i* res, __m512i* a);

void fp_frommont_4x2(__m512i*res, __m512i* a);

#endif