#ifndef _FP2_LVL3_H
#define _FP2_LVL3_H


/**
 * F_(p^2) = F_p[X]/<X^2+1>
 * 
 * In the 8x1x1 format, we store 8 Fp2 elements of the form a0 + i* a1 in 16 vectors. 
 * The first 8 vectors store the a0, the next 8 store a1
 */

void fp2_setzero_8x1x1(__m512i* res);
void fp2_copy_8x1x1(__m512i* res, __m512i* a);

/**
 *   Returns an 8-bit mask which is 1 iff the i-th Fp2 elements are equal
 */
int fp2_is_equal_8x1x1(__m512i* a, __m512i* b);

void fp2_add_8x1x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_sub_8x1x1(__m512i* res, __m512i* a, __m512i* b);

void fp2_neg_8x1x1(__m512i* res, __m512i* a);

 void fp2_mul_8x1x1(__m512i* res, __m512i* a, __m512i* b);

 void fp2_sqr_8x1x1(__m512i* res, __m512i* a);
 
 void fp2_tomont_8x1x1(__m512i* res, __m512i* a);
 
 void fp2_frommont_8x1x1(__m512i*res, __m512i* a);

#endif