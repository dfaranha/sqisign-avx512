#ifndef _FP2_LVL3_H
#define _FP2_LVL3_H




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

#endif