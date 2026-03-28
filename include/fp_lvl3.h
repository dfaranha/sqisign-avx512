/**
 *   The SQISign lvl3 prime is p = 65 * 2^376 - 1
 *   It is 383 bits, so the number of 48-bit limbs needed is 383/48 < 8
 *   
 *   Formats: a x b = a vectors of b Fp limbs
 */

#ifndef _FP_LVL3_H
#define _FP_LVL3_H

#include <stdint.h>

#include "intrinsics_avx512.h"

#include "sqi_fp_lvl3.h"

#define FP_NUM_64LIMBS 6
#define FP_NUM_AVXLIMBS 8

static const uint64_t LIMB_LEN = 48;
static const uint64_t LIMB_ALIGN = 52 - LIMB_LEN;
static const uint64_t LIMB_MASK = 0xffffffffffff;
static const uint64_t LAST_LIMB_MASK = 0x7fffffffffff;    ///////// IS THIS USED SOMEWHERE?
// p in radix 2^48
static const uint64_t p[8] = { 0xffffffffffff, 0xffffffffffff, 0xffffffffffff, 0xffffffffffff, 0xffffffffffff, 0xffffffffffff, 0xffffffffffff, 0x40ffffffffff };

static const uint64_t p_plus_1_top = 0x410000000000;
// 2*p in radix 2^48
static const uint64_t p2[8] = { 0xfffffffffffe, 0xffffffffffff, 0xffffffffffff, 0xffffffffffff, 0xffffffffffff, 0xffffffffffff, 0xffffffffffff, 0x81ffffffffff };

// We use the Montgomery representation with R = 2^(9*48) = 2^384
// R2 is R^2 mod p, which is needed to convert to the representation.
static const uint64_t R[8] = {
    0xF03F03F03F03, 0x03, 0x0, 0x0, 
    0x0, 0x0, 0x0, 0x3D0000000000
};

static const uint64_t R2[8] = { 
    0x7B0FD07F000F, 0x732F52771F91, 0xF03F03F03F13, 0xF03F03F03F03, 
    0xF03F03F03F03, 0xF03F03F03F03, 0xF03F03F03F03, 0x213F03F03F03 
};

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
 *   Returns an 8-bit mask where the i-th bit is 1 iff the i-th Fp elements are equal
 */
int fp_is_equal_8x1(__m512i* a, __m512i* b);

/**
 *   Returns an 8-bit mask where the i-th bit is 1 iff the i-th Fp elements are equal
 */
int fp_is_zero_8x1(__m512i* a);

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
 * If add_mask[i]=1 res[i] = a[i] + b[i], else res[i] = a[i] - b[i]
 */
void fp_mixaddsub_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

/**
 *  res = a * b (mod p), multiprecision multiplication using schoolbook
 */
void fp_mul_8x1_school(__m512i* res, __m512i* a, __m512i* b);

/**
 *  res = a * b (mod p), multiprecision multiplication using Karatsuba
 */
void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a^2 (mod p)
 */
void fp_sqr_8x1(__m512i* res, __m512i* a);

/**
 * res = a * 3^(-1) (mod p)
 */
void fp_div3_8x1(__m512i* res, __m512i* src);

/**
 * res = a * R (mod p)
 */
void fp_tomont_8x1(__m512i* res, __m512i* a);

/**
 * res = a * R^(-1) (mod p)
 */
void fp_frommont_8x1(__m512i* res, __m512i* a);

void fp_8x1_to_unvectorized(fp_t* res, __m512i* src);
// helper function, also used in fp2 conversions.
void fp_48_to_64_bit_limbs_8x1(fp_t* res, __m512i* src);
void fp_48_to_64_bit_limbs_8x1_only_even_idx(fp_t* res, __m512i* src);

void fp_unvectorized_to_8x1(__m512i* res, fp_t* src);
// helper function, also used in fp2 conversions.
void fp_64_to_48_bit_limbs_8x1(__m512i* res, fp_t* src);
void fp_64_to_48_bit_limbs_8x1_only_even_idx(__m512i* res, fp_t* src);

/**
 *  res = a + b, multiprecision addition, no correction, no carry propagation
 */
void mp_add_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 *  res = a - b, multiprecision subtraction, no correction, no carry propagation
 */
void mp_sub_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 *  res = res - a - b, double-length multiprecision subtraction, no correction
 */
void mp_dblsubx2_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 *  res = res - a, double-length subtraction/addition, no correction
 */
void mp_subx2_8x1(__m512i* res, __m512i* a);

void mp_mixaddsubx2_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

/**
 *  res = (a+p) - b, no correction. When the inputs are fully reduced, this is guaranteed >= 0
 */
void mp_add_p_sub_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 *  res = 2*a, no correction
 */
void mp_dbl_8x1(__m512i* res, __m512i* a);

/**
 *  res = a * b, multiprecision multiplication using Karatsuba
 */
void mp_mul_8x1_karat(__m512i* res, __m512i* a, __m512i* b);

/**
 *  res = a (mod p), Montgomery reduction
 */
void rdc_8x1(__m512i* res, __m512i* a);

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

/**
 *   Returns a 4-bit mask which is 1 iff the i-th Fp element is zero
 */
int fp_is_zero_4x2(__m512i* a);

void fp_add_4x2(__m512i* res, __m512i* a, __m512i* b);

void fp_sub_4x2(__m512i* res, __m512i* a, __m512i* b);

void fp_neg_4x2(__m512i* res, __m512i* a);

/**
 *  Denote the 4 Fp elements stored in a by a0, a1, a2, a3, and in b by b0, b1, b2, b3.
 *  Computes a0 - b0, a1 + b1, a2 - b2, a3 + b3.
 */
 void fp_mixaddsub_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

void mp_add_4x2(__m512i* res, __m512i* a, __m512i* b);
void mp_add_p_sub_4x2(__m512i* res, __m512i* a, __m512i* b);
void mp_mixaddsubx2_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);
void mp_dblsubx2_4x2(__m512i* res, __m512i* a, __m512i* b);
void mp_subx2_4x2(__m512i* res, __m512i* a);
void mp_mul_4x2(__m512i* res, __m512i* a, __m512i* b);
void rdc_4x2(__m512i* res, __m512i* a);

void fp_mul_4x2(__m512i* res, __m512i* a, __m512i* b);
void fp_sqr_4x2(__m512i* res, __m512i* a);

void fp_tomont_4x2(__m512i* res, __m512i* a);
void fp_frommont_4x2(__m512i* res, __m512i* a);

void fp_48_to_64_bit_limbs_4x2(fp_t* res, __m512i* src);
void fp_64_to_48_bit_limbs_4x2(__m512i* res, fp_t* src);

void fp_4x2_to_unvectorized(fp_t* res, __m512i* src);
void fp_unvectorized_to_4x2(__m512i* res, fp_t* src);

#endif