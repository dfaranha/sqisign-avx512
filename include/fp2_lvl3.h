#ifndef _FP2_LVL3_H
#define _FP2_LVL3_H

#include <stdint.h>

#include "intrinsics_avx512.h"


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

void fp2_add_8x1x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sub_8x1x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_neg_8x1x1(__m512i* res, __m512i* a);

void fp2_mul_8x1x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sqr_8x1x1(__m512i* res, __m512i* a);

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

void fp2_add_4x2x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sub_4x2x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_neg_4x2x1(__m512i* res, __m512i* a);

void fp2_mul_4x2x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sqr_4x2x1(__m512i* res, __m512i* a);

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

void fp2_add_2x4x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sub_2x4x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_neg_2x4x1(__m512i* res, __m512i* a);

void fp2_mul_2x4x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sqr_2x4x1(__m512i* res, __m512i* a);

void fp2_tomont_2x4x1(__m512i* res, __m512i* a);

void fp2_frommont_2x4x1(__m512i* res, __m512i* a);

// ======================================================================================================
//                                      Conversions between representations
// ======================================================================================================

/**
 * Converts between the array representation and the vectorized representations.
 * In the array representation, we store the Fp2 elements one after the other, in one big array.
 * Each Fp2 element is stored in twelve 64-bit integers, with the first six storing the real coefficient.
 */
void fp2_array_to_8x1x1(__m512i* res, uint64_t* src);
void fp2_array_to_4x2x1(__m512i* res, uint64_t* src);

void fp2_8x1x1_to_array(uint64_t* res, __m512i* src);
void fp2_4x2x1_to_array(uint64_t* res, __m512i* src);

/**
 * Converts an Fp2 8x1x1 to two 4x2x1's.
 */
void fp2_8x1x1_to_4x2x1(__m512i* res, __m512i* src);

/**
 * Converts two Fp2 4x2x1's to an 8x1x1.
 */
void fp2_4x2x1_to_8x1x1(__m512i* res, __m512i* src);

#endif