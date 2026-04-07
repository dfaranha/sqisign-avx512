#ifndef _FP_H
#define _FP_H

#include <stdint.h>
#include "intrinsics_avx512.h"

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
 * res = a^(-1) (mod p)
 */
void fp_inv_8x1(__m512i* res, __m512i* src);

/**
 * res = a * R (mod p)
 */
void fp_tomont_8x1(__m512i* res, __m512i* a);

/**
 * res = a * R^(-1) (mod p)
 */
void fp_frommont_8x1(__m512i* res, __m512i* a);

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

void rand_fp_8x1(__m512i* res);

void fp_64_to_limbs_8x1(__m512i* res, uint64_t* unvec);
void fp_64_to_limbs_8x1_only_even_idx(__m512i* res, uint64_t* unvec);
void fp_limbs_to_64_8x1(uint64_t* res, __m512i* src);
void fp_limbs_to_64_8x1_only_even_idx(uint64_t* res, __m512i* src);

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

void fp_div3_4x2(__m512i* res, __m512i* a);
void fp_inv_4x2(__m512i* res, __m512i* src);

void rand_fp_4x2(__m512i* res);

void fp_64_to_limbs_4x2(__m512i* res, uint64_t* unvec);
void fp_limbs_to_64_4x2(uint64_t* res, __m512i* src);

// ======================================================================================================
//                                                 2x4
// ======================================================================================================

/*  2x4 = 2 Fp elements a, b, c, d, interleaved as follows:
 *  v0 = (a0, a2, a4, a6, b0, b2, b4, b6)
 *  v1 = (a1, a3, a5, a7, b1, b3, b5, b7)
 */

void fp_setzero_2x4(__m512i* res);
void fp_copy_2x4(__m512i* res, __m512i* a);


/**
 * @param a - Fp element stored in 51-bit limbs
 * @result 2 copies of a, stored in res
 */
void fp_duplicate_2x4(__m512i* res, const uint64_t* a);

/**
 *   Returns a 2-bit mask which is 1 iff the i-th Fp elements are equal
 */
int fp_is_equal_2x4(__m512i* a, __m512i* b);

/**
 *   Returns a 2-bit mask which is 1 iff the i-th Fp element is zero
 */
int fp_is_zero_2x4(__m512i* a);

/**
 * Copies the i-th Fp element from a1 to res if the i-th bit of s is set, else it copies from a0.
 */
void fp_select_2x4(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);

void fp_add_2x4(__m512i* res, __m512i* a, __m512i* b);
void fp_sub_2x4(__m512i* res, __m512i* a, __m512i* b);
void fp_neg_2x4(__m512i* res, __m512i* a);
void fp_mixaddsub_2x4(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

void fp_mul_2x4(__m512i* res, __m512i* a, __m512i* b);
void fp_sqr_2x4(__m512i* res, __m512i* a);

void fp_tomont_2x4(__m512i* res, __m512i* a);
void fp_frommont_2x4(__m512i* res, __m512i* a);

void fp_div3_2x4(__m512i* res, __m512i* a);
void fp_inv_2x4(__m512i* res, __m512i* a);
void fp_sqrt_2x4(__m512i* res, __m512i* a);
void fp_exp3div4_2x4(__m512i* res, __m512i* a);

void fp_4x2_to_2x4(__m512i* res1, __m512i* res2, __m512i* src);
void fp_2x4_to_4x2(__m512i* res, __m512i* src1, __m512i* src2);

void rand_fp_2x4(__m512i* res);

void fp_64_to_limbs_2x4(__m512i* res, uint64_t* unvec);
void fp_limbs_to_64_2x4(uint64_t* res, __m512i* src);

// ====================================================================================================
//                                Conversions between Fp representations                               
// ====================================================================================================

// Given src = (a, b, c, d, e, f, g, h),
// it computes res1 = (a, c, e, g) and res2 = (b, d, f, h).
void fp_8x1_to_4x2(__m512i* res1, __m512i* res2, __m512i* src);

// Given src1 = (a1, b1, c1, d1) and src2 = (a2, b2, c2, d2),
// it computes res = (a1, a2, b1, b2, c1, c2, d1, d2).
void fp_4x2_to_8x1(__m512i* res, __m512i* src1, __m512i* src2);

// Given src=(a, b, c, d), it computes res1 = (a, c) and res2 = (b, d).
// We choose this interleaving pattern for the Fp2 2x2x2 sqrt,
// where res1 = re and res2 = im.
void fp_4x2_to_2x4(__m512i* res1, __m512i* res2, __m512i* src);
void fp_2x4_to_4x2(__m512i* res, __m512i* src1, __m512i* src2);

// ====================================================================================================
//                                      Unvectorized Fp inversion                                     
// ====================================================================================================

void fp_inv_64(uint64_t* res, const uint64_t* src);

#endif