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

#endif