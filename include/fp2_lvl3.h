#ifndef _FP2_LVL3_H
#define _FP2_LVL3_H

#include <stdint.h>

#include "intrinsics_avx512.h"
#include "sqi_fp2.h"


/**
 * F_(p^2) = F_p[X]/<X^2+1>
 */

// ======================================================================================================
//                                                 8x1x1
// ======================================================================================================

/**
 * In the 8x1x1 format, we store 8 Fp2 elements of the form a0 + i* a1 in 16 vectors. 
 * The first 8 vectors store the a0, the next 8 store a1
 */

void fp2_setzero_8x1x1(__m512i* res);
void fp2_copy_8x1x1(__m512i* res, __m512i* a);

/**
 *   Returns an 8-bit mask. The i-th bit is 1 iff the i-th Fp2 elements are equal
 */
int fp2_is_equal_8x1x1(__m512i* a, __m512i* b);

/**
 *   Returns an 8-bit mask. The i-th bit is 1 iff the i-th Fp2 element is zero
 */
int fp2_is_zero_8x1x1(__m512i* a);

void fp2_add_8x1x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sub_8x1x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_neg_8x1x1(__m512i* res, __m512i* a);

/**
 * add_mask_re has 8-bits, one per Fp2 element in the 8x1x1.
 * If add_mask_re[i]=1, re(res[i]) = re(a[i]) + re(b[i]), else re(res[i]) = re(a[i]) - re(b[i])
 */
void fp2_mixaddsub_8x1x1(__m512i* res, __m512i* a, __m512i* b, 
  const __mmask8 add_mask_re, const __mmask8 add_mask_im);

void fp2_mul_8x1x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sqr_8x1x1(__m512i* res, __m512i* a);

void fp2_div3_8x1x1(__m512i* res, __m512i* src);

/**
 * Multipication by i.
 */
void fp2_muli_8x1x1(__m512i* res, __m512i* a);

void fp2_tomont_8x1x1(__m512i* res, __m512i* a);

void fp2_frommont_8x1x1(__m512i*res, __m512i* a);

// ======================================================================================================
//                                                 4x1x2
// ======================================================================================================

/**
 * In the 4x1x2 format, we store 4 Fp2 elements a0 + a1 * i in 8 vectors. 
 * The first 4 vectors store the a0, the next 4 store the a1.
 * The Fp coefficients are stored in the 4x2 format.
 */

void fp2_setzero_4x1x2(__m512i* res);
void fp2_copy_4x1x2(__m512i* res, __m512i* a);

/**
 *   Returns a 4-bit mask. The ith bit is 1 iff the i-th Fp2 elements are equal
 */
 int fp2_is_equal_4x1x2(__m512i* a, __m512i* b);

 /**
 *   Returns a 4-bit mask. The ith bit is 1 iff the i-th Fp2 element is zero
 */
 int fp2_is_zero_4x1x2(__m512i* a);

 void fp2_add_4x1x2(__m512i* res, __m512i* a, __m512i* b);
 
 void fp2_sub_4x1x2(__m512i* res, __m512i* a, __m512i* b);
 
 void fp2_neg_4x1x2(__m512i* res, __m512i* a);
 
 void fp2_mul_4x1x2(__m512i* res, __m512i* a, __m512i* b);
 
 void fp2_sqr_4x1x2(__m512i* res, __m512i* a);
 
 void fp2_tomont_4x1x2(__m512i* res, __m512i* a);
 
 void fp2_frommont_4x1x2(__m512i*res, __m512i* a);

// ======================================================================================================
//                                                 4x2x1
// ======================================================================================================

/**
 * In the 4x2x1 format, we store 4 Fp2 elements a, b, c, d in 8 vectors. 
 * We store their Fp coefficients in the 8x1 Fp format
 * The i-th vector contains the i-th limb of a0, a1, b0, b1, c0, c1, d0, d1
 */

void fp2_setzero_4x2x1(__m512i* res);
void fp2_copy_4x2x1(__m512i* res, __m512i* a);

/**
 *   Returns a 4-bit mask. The ith bit is 1 iff the i-th Fp2 elements are equal
 */
int fp2_is_equal_4x2x1(__m512i* a, __m512i* b);

/**
 *   Returns a 4-bit mask. The ith bit is 1 iff the i-th Fp2 element is zero
 */
int fp2_is_zero_4x2x1(__m512i* a);

void fp2_add_4x2x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sub_4x2x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_neg_4x2x1(__m512i* res, __m512i* a);

/**
 * add_mask has 8-bits, one per Fp element in the 4x2x1.
 * If add_mask[i]=1, res[i] = a[i] + b[i], else res[i] = a[i] - b[i]
 */
void fp2_mixaddsub_4x2x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

void fp2_mul_4x2x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sqr_4x2x1(__m512i* res, __m512i* a);

void fp2_muli_4x2x1(__m512i* res, __m512i* a);

void fp2_div3_4x2x1(__m512i* res, __m512i* src);

void fp2_tomont_4x2x1(__m512i* res, __m512i* a);

void fp2_frommont_4x2x1(__m512i*res, __m512i* a);

// ======================================================================================================
//                                                 2x2x2
// ======================================================================================================

/**
 *  In the 2x2x2 format, we store two Fp2 elements a, b in 4 vectors.
 *  We store their coefficients in the 4x2 Fp format.
 *  The i-th vector contains a_(0,i) a_(0,i+4), a_(1,i), a_(1,i+4), b_(0,i), b_(0,i+4), b_(1,i), b_(1,i+4)
 */

void fp2_setzero_2x2x2(__m512i* res);
void fp2_copy_2x2x2(__m512i* res, __m512i* a);

/**
 *   Returns a 2-bit mask. The ith bit is 1 iff the i-th Fp2 elements are equal
 */
int fp2_is_equal_2x2x2(__m512i* a, __m512i* b);

/**
 *   Returns a 2-bit mask. The ith bit is 1 iff the i-th Fp2 element is zero
 */
int fp2_is_zero_2x2x2(__m512i* a);

void fp2_add_2x2x2(__m512i* res, __m512i* a, __m512i* b);

void fp2_sub_2x2x2(__m512i* res, __m512i* a, __m512i* b);

void fp2_neg_2x2x2(__m512i* res, __m512i* a);

void fp2_mul_2x2x2(__m512i* res, __m512i* a, __m512i* b);

void fp2_sqr_2x2x2(__m512i* res, __m512i* a);

void fp2_tomont_2x2x2(__m512i* res, __m512i* a);

void fp2_frommont_2x2x2(__m512i*res, __m512i* a);

// ======================================================================================================
//                                                 2x4x1
// ======================================================================================================

/**
 *  In the 2x4x1 format, we store two Fp2 elements a, b in 8 vectors
 *  The i-th vector contains the i-th limb of a0, 0, a1, 0, b0, 0, b1, 0
 *  This representation is wasteful, but allows using the 8x1 Fp multiplier.
 */

 void fp2_setzero_2x4x1(__m512i* res);
 void fp2_copy_2x4x1(__m512i* res, __m512i* a);
 
 /**
  *   Returns a 2-bit mask. The ith bit is 1 iff the i-th Fp2 elements are equal
  */
int fp2_is_equal_2x4x1(__m512i* a, __m512i* b);

/**
  *   Returns a 2-bit mask. The ith bit is 1 iff the i-th Fp2 element is zero
  */
int fp2_is_zero_2x4x1(__m512i* a);

void fp2_add_2x4x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sub_2x4x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_neg_2x4x1(__m512i* res, __m512i* a);

/**
 * add_mask has 8-bits. Bit 2*i is for the i-th Fp element in the 2x4x1.
 * If add_mask[i]=1, res[i] = a[i] + b[i], else res[i] = a[i] - b[i]
 */
void fp2_mixaddsub_2x4x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

void fp2_mul_2x4x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sqr_2x4x1(__m512i* res, __m512i* a);

void fp2_tomont_2x4x1(__m512i* res, __m512i* a);

void fp2_frommont_2x4x1(__m512i* res, __m512i* a);

// ======================================================================================================
//                                      Conversions between representations
// ======================================================================================================

void fp2_unvectorized_to_8x1x1(__m512i* res, fp2_t* src);
void fp2_unvectorized_to_4x2x1(__m512i* res, fp2_t* src);
void fp2_8x1x1_to_unvectorized(fp2_t* res, __m512i* src);
void fp2_4x2x1_to_unvectorized(fp2_t* res, __m512i* src);

/**
 * Converts an Fp2 8x1x1 to two 4x2x1's.
 * src = (x_1, x_2, y_1, y_2, ...)
 * res1 = (x_1, y_1, ...), res2 = (x_2, y_2, ...)
 */
void fp2_8x1x1_to_4x2x1(__m512i* res1, __m512i* res2, __m512i* src);

/**
 * Converts two Fp2 4x2x1's to an 8x1x1.
 * src_i = (x_i, y_i, z_i, t_i)
 * res = (x_1, x_2, y_1, y_2, ...)
 */
void fp2_4x2x1_to_8x1x1(__m512i* res, __m512i* src1, __m512i* src2);

// More specialized conversion functions:
// TODO: Go through vectorized code and see if we can use these.

// Given a in 4x2x1, outputs (a, a) in 8x1x1.
void fp2_4x2x1_duplicate_to_8x1x1(__m512i* res, __m512i* src);
// Only extracts first 4x2x1
void fp2_8x1x1_to_first_4x2x1(__m512i* res, __m512i* src);


void fp2_4x2x1_to_2x4x1(__m512i* res, __m512i* src);
void fp2_2x4x1_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2);

#endif