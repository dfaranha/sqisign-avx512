#ifndef _FP2_H
#define _FP2_H

#include <stdint.h>
#include "fp.h"
#include "intrinsics_avx512.h"

/*
 * Fp2 = Fp[X]/<X^2 + 1>
 * Each element is of the form a_re + a_im * i, with a_re, a_im in Fp and i^2 = -1
 */

// ====================================================================================================
//                                 Conversions between representations                                
// ====================================================================================================

/**
 * Given (a, b, c, d, e, f, g, h) in 8x1x1,it computes (a, c, e, g) and (b, d, f, h) in 4x2x1.
 */
void fp2_8x1x1_to_4x2x1(__m512i* res1, __m512i* res2, __m512i* src);

/**
 * Given (a1, b1, c1, d1) and (a2, b2, c2, d2) in 4x2x1,
 * it computes (a1, a2, b1, b2, c1, c2, d1, d2) in 8x1x1.
 */
void fp2_4x2x1_to_8x1x1(__m512i* res, __m512i* src1, __m512i* src2);

/**
 * Only extracts the first 4x2x1, i.e. from (a, b, c, d, e, f, g, h) in 8x1x1 it outputs (a, c, e, g) in 4x2x1.
 */
void fp2_8x1x1_to_first_4x2x1(__m512i* res, __m512i* src);

/**
 * Computes 4x2x1_to_8x1x1 with both inputs as src.
 */
void fp2_4x2x1_duplicate_to_8x1x1(__m512i* res, __m512i* src);

/**
 * Given (a, b, c, d) in 4x2x1 it computes (a, b) and (c, d) in 2x2x2.
 */
void fp2_4x2x1_to_2x2x2(__m512i* res1, __m512i* res2, __m512i* src);

/**
 * Given (a1, b1) and (a2, b2) in 2x2x2, it computes (a1, b1, a2, b2) in 4x2x1.
 */
void fp2_2x2x2_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2);

/**
 * Given (a, b, c, d) in 4x2x1 it computes (a, b) and (c, d) in 2x4x1.
 */
void fp2_4x2x1_to_2x4x1(__m512i* res1, __m512i* res2, __m512i* src);

/**
 * Given (a1, b1) and (a2, b2) in 2x4x1, it computes (a1, b1, a2, b2) in 4x2x1.
 */
void fp2_2x4x1_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2);

/**
 * Given (a, b) in 2x2x2 it computes (a, b) in 2x4x1.
 */
void fp2_2x2x2_to_2x4x1(__m512i* res, __m512i* src);

/**
 * Given (a, b) in 2x4x1 it computes (a, b) in 2x2x2.
 */
void fp2_2x4x1_to_2x2x2(__m512i* res, __m512i* src);

// ====================================================================================================
//                                                8x1x1                                               
// ====================================================================================================

/*
 * The 8x1x1 format stores eight Fp2 elements in 2 * NUM_LIMBS vectors.
 * The first NUM_LIMBS vectors store the real Fp coefficient in the 8x1 format, the latter store the imaginary coefficient in 8x1.
 */

/**
 * Sets all the Fp2 elements to 0.
 */
void fp2_setzero_8x1x1(__m512i* res);

/**
 * If bits[i] = 1, the i-th Fp2 element is set to 1 (in Montgomery repr.)
 */
void fp2_set_binary_8x1x1(__m512i* res, __mmask8 bits);

/**
 * Copies a to res
 */
void fp2_copy_8x1x1(__m512i* res, __m512i* a);

/**
 * Returns 8-bit mask where the i-th bit is 1 iff the i-th Fp2 elements are equal.
 */
__mmask8 fp2_is_equal_8x1x1(__m512i* a, __m512i* b);

/**
 * Returns 8-bit mask where the i-th bit is 1 iff the i-th Fp2 element is zero.
 */
__mmask8 fp2_is_zero_8x1x1(__m512i* a);

/**
 * Copies the i-th Fp2 element from a1 to res if the i-th bit of s is set, else it copies from a0.
 */
void fp2_select_8x1x1(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);

/**
 * res = a + b
 */
static inline void fp2_add_8x1x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_add_8x1(res, a, b);
    fp_add_8x1(res+8, a+8, b+8);
}

/**
 * res = a - b
 */
static inline void fp2_sub_8x1x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_sub_8x1(res, a, b);
    fp_sub_8x1(res+8, a+8, b+8);
}

/**
 * res = -a
 */
static inline void fp2_neg_8x1x1(__m512i* res, __m512i* a)
{
    fp_neg_8x1(res, a);
    fp_neg_8x1(res+8, a+8);
}

/**
 * Mixed addition and subtraction.
 * If add_mask_re[i]=1, re(res[i]) = re(a[i]) + re(b[i]), else re(res[i]) = re(a[i]) - re(b[i])
 * Similarly, add_mask_im controls the operation applied to the imaginary coefficients.
 */
static inline void fp2_mixaddsub_8x1x1(__m512i* res, __m512i* a, __m512i* b,
    const __mmask8 add_mask_re, const __mmask8 add_mask_im)
{
    fp_mixaddsub_8x1(res, a, b, add_mask_re);
    fp_mixaddsub_8x1(res+8, a+8, b+8, add_mask_im);
}

/**
 * res = a * b
 */
static inline void fp2_mul_8x1x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i tt1[16], tt2[16], tt3[16];
    
    mp_add_8x1(tt3, a, a + 8);
    mp_add_8x1(tt3 + 8, b, b + 8);
    mp_mul_8x1(tt1, a, b);
    mp_mul_8x1(tt2, a + 8, b + 8);
    mp_mul_8x1(tt3, tt3, tt3 + 8);
    mp_dblsubx2_8x1(tt3, tt1, tt2);
    mp_subx2_8x1(tt1, tt2);
    rdc_8x1(res + 8, tt3);
    rdc_8x1(res, tt1);
}

/**
 * res = a^2
 */
static inline void fp2_sqr_8x1x1(__m512i* res, __m512i* a)
{
    __m512i s1[8], s2[8], s3[8];
    
    mp_add_8x1(s1, a, a + 8);
    mp_add_p_sub_8x1(s2, a, a + 8);
    mp_add_8x1(s3, a, a);
    fp_mul_8x1(res, s1, s2);
    fp_mul_8x1(res + 8, s3, a + 8);
}

/**
 * Multiplication by i.
 */
void fp2_muli_8x1x1(__m512i* res, __m512i* a);

/**
 * Complex conjugation. res = a_re - a_im * i
 */
void fp2_conjugate_8x1x1(__m512i* res, __m512i* a);

/**
 * res = a / 2
 */
void fp2_div2_8x1x1(__m512i* res, __m512i* src);

/**
 * res = a / 3
 */
void fp2_div3_8x1x1(__m512i* res, __m512i* src);

/**
 * res = a^-1
 */
void fp2_inv_8x1x1(__m512i* res, __m512i* src);

/**
 * res = sqrt(a)
 */
void fp2_sqrt_8x1x1(__m512i* res, __m512i* a);

/**
 * Converts a to the Montgomery representation. res = a * R
 */
void fp2_tomont_8x1x1(__m512i* res, __m512i* a);

/**
 * Converts a back from the Montgomery representation. res = a * R^-1
 */
void fp2_frommont_8x1x1(__m512i* res, __m512i* a);

/**
 * Samples a random Fp2 element.
 */
void rand_fp2_8x1x1(__m512i* res);

// ====================================================================================================
//                                                4x1x2                                               
// ====================================================================================================

/*
 * The 4x1x2 format stores four Fp2 elements in 2 * N4x2 vectors.
 * The first N4x2 vectors store the real Fp coefficient in the 4x2 format, and the latter store the imaginary coefficient in 4x2.
 */

/**
 * Sets all the Fp2 elements to 0.
 */
void fp2_setzero_4x1x2(__m512i* res);

/**
 * If bits[i] = 1, the i-th Fp2 element is set to 1 (in Montgomery repr.)
 */
void fp2_set_binary_4x1x2(__m512i* res, __mmask8 bits);

/**
 * Copies a to res
 */
void fp2_copy_4x1x2(__m512i* res, __m512i* a);

/**
 * Returns 4-bit mask where the i-th bit is 1 iff the i-th Fp2 elements are equal.
 */
__mmask8 fp2_is_equal_4x1x2(__m512i* a, __m512i* b);

/**
 * Returns 4-bit mask where the i-th bit is 1 iff the i-th Fp2 element is zero.
 */
__mmask8 fp2_is_zero_4x1x2(__m512i* a);

/**
 * Copies the i-th Fp2 element from a1 to res if the i-th bit of s is set, else it copies from a0.
 */
void fp2_select_4x1x2(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);

/**
 * res = a + b
 */
static inline void fp2_add_4x1x2(__m512i* res, __m512i* a, __m512i* b)
{
    fp_add_4x2(res, a, b);
    fp_add_4x2(res+4, a+4, b+4);
}

/**
 * res = a - b
 */
static inline void fp2_sub_4x1x2(__m512i* res, __m512i* a, __m512i* b)
{
    fp_sub_4x2(res, a, b);
    fp_sub_4x2(res+4, a+4, b+4);
}

/**
 * res = -a
 */
static inline void fp2_neg_4x1x2(__m512i* res, __m512i* a)
{
    fp_neg_4x2(res, a);
    fp_neg_4x2(res+4, a+4);
}

/**
 * Mixed addition and subtraction.
 * If add_mask_re[i]=1, re(res[i]) = re(a[i]) + re(b[i]), else re(res[i]) = re(a[i]) - re(b[i])
 * Similarly, add_mask_im controls the operation applied to the imaginary coefficients.
 */
static inline void fp2_mixaddsub_4x1x2(__m512i* res, __m512i* a, __m512i* b,
    const __mmask8 add_mask_re, const __mmask8 add_mask_im)
{
    fp_mixaddsub_4x2(res, a, b, add_mask_re);
    fp_mixaddsub_4x2(res+4, a+4, b+4, add_mask_im);
}

/**
 * res = a * b
 */
static inline void fp2_mul_4x1x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t1[13], t2[13], t3[13];
    
    mp_add_4x2(t3, a, a + 4);
    mp_add_4x2(t3 + 4, b, b + 4);
    mp_mul_4x2(t1, a, b);
    mp_mul_4x2(t2, a+4, b+4);
    mp_mul_4x2(t3, t3, t3+4);
    mp_dblsubx2_4x2(t3, t1, t2);
    mp_subx2_4x2(t1, t2);
    rdc_4x2(res+4, t3);
    rdc_4x2(res, t1);
}

/**
 * res = a^2
 */
static inline void fp2_sqr_4x1x2(__m512i* res, __m512i* a)
{
    __m512i s1[4], s2[4], s3[4];
    
    mp_add_4x2(s1, a, a + 4);
    mp_add_p_sub_4x2(s2, a, a + 4);
    mp_add_4x2(s3, a, a);
    fp_mul_4x2(res + 4, s3, a + 4);
    fp_mul_4x2(res, s1, s2);
}

/**
 * Multiplication by i.
 */
void fp2_muli_4x1x2(__m512i* res, __m512i* a);

/**
 * Complex conjugation. res = a_re - a_im * i
 */
void fp2_conjugate_4x1x2(__m512i* res, __m512i* a);

/**
 * res = a / 2
 */
void fp2_div2_4x1x2(__m512i* res, __m512i* src);

/**
 * res = a / 3
 */
void fp2_div3_4x1x2(__m512i* res, __m512i* src);

/**
 * res = a^-1
 */
void fp2_inv_4x1x2(__m512i* res, __m512i* src);

/**
 * res = sqrt(a)
 */
void fp2_sqrt_4x1x2(__m512i* res, __m512i* a);

/**
 * Converts a to the Montgomery representation. res = a * R
 */
void fp2_tomont_4x1x2(__m512i* res, __m512i* a);

/**
 * Converts a back from the Montgomery representation. res = a * R^-1
 */
void fp2_frommont_4x1x2(__m512i* res, __m512i* a);

/**
 * Samples a random Fp2 element.
 */
void rand_fp2_4x1x2(__m512i* res);

// ====================================================================================================
//                                                4x2x1                                               
// ====================================================================================================

/*
 * The 4x2x1 format stores four Fp2 elements a, b, c, d in NUM_LIMBS vectors.
 * It stores (a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im) in 8x1.
 */

/**
 * Sets all the Fp2 elements to 0.
 */
void fp2_setzero_4x2x1(__m512i* res);

/**
 * If bits[i] = 1, the i-th Fp2 element is set to 1 (in Montgomery repr.)
 */
void fp2_set_binary_4x2x1(__m512i* res, __mmask8 bits);

/**
 * Copies a to res
 */
void fp2_copy_4x2x1(__m512i* res, __m512i* a);

/**
 * Returns 4-bit mask where the i-th bit is 1 iff the i-th Fp2 elements are equal.
 */
__mmask8 fp2_is_equal_4x2x1(__m512i* a, __m512i* b);

/**
 * Returns 4-bit mask where the i-th bit is 1 iff the i-th Fp2 element is zero.
 */
__mmask8 fp2_is_zero_4x2x1(__m512i* a);

/**
 * Copies the i-th Fp2 element from a1 to res if the i-th bit of s is set, else it copies from a0.
 */
void fp2_select_4x2x1(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);

/**
 * res = a + b
 */
static inline void fp2_add_4x2x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_add_8x1(res, a, b);
}

/**
 * res = a - b
 */
static inline void fp2_sub_4x2x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_sub_8x1(res, a, b);
}

/**
 * res = -a
 */
static inline void fp2_neg_4x2x1(__m512i* res, __m512i* a)
{
    fp_neg_8x1(res, a);
}

/**
 * Mixed addition and subtraction. The operation applied is determined by add_mask.
 * Bit 2*i and 2*i+1 determine the operation for the real and imaginary coefficient of the i-th Fp2 element.
 */
static inline void fp2_mixaddsub_4x2x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    fp_mixaddsub_8x1(res, a, b, add_mask);
}

/**
 * res = a * b
 */
static inline void fp2_mul_4x2x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0[16], t1[16];
    for (__mmask8 i = 0; i < 8; i++) {
        t0[i] = VSHUFFLE(a[i], 0x44); // a0, a0
        t1[i] = VSHUFFLE(a[i], 0xee); // a1, a1
        t1[i + 8] = VSHUFFLE(b[i], 0x4e); // b0, b1
    }
    mp_mul_8x1(t0, t0, b);
    mp_mul_8x1(t1, t1, t1 + 8);
    mp_mixaddsubx2_8x1(t0, t0, t1, 0xaa);
    rdc_8x1(res, t0);
}

/**
 * res = a^2
 */
static inline void fp2_sqr_4x2x1(__m512i* res, __m512i* a)
{
    __m512i s1[8], s2[8], s3[8];
    __mmask16 mask = 0b0011001100110011;
    for (__mmask8 i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0
        s2[i] = VSHUFFLEMASKZ(mask, a[i], 0x0e); // 0, a1
        s3[i] = VSHUFFLE(a[i], 0x4e); // a0, a1
    }
    mp_add_8x1(s1, s1, s3);
    mp_add_p_sub_8x1(s2, a, s2);
    fp_mul_8x1(res, s1, s2);
}

/**
 * Computes res1 = a1 * b1 and res2 = a2 * b2 using 8x1x1.
 */
static inline void fp2_mulx2_4x2x1(__m512i* res1, __m512i* res2, __m512i* a1, __m512i* b1, __m512i* a2, __m512i* b2)
{
    __m512i t0[16], t1[16];
    
    fp2_4x2x1_to_8x1x1(t0, a1, a2);
    fp2_4x2x1_to_8x1x1(t1, b1, b2);
    fp2_mul_8x1x1(t0, t0, t1);
    fp2_8x1x1_to_4x2x1(res1, res2, t0);
}

/**
 * Computes res1 = a1^2 and res2 = a2^2 using 8x1x1.
 */
static inline void fp2_sqrx2_4x2x1(__m512i* res1, __m512i* res2, __m512i* a1, __m512i* a2)
{
    __m512i t[16];
    
    fp2_4x2x1_to_8x1x1(t, a1, a2);
    fp2_sqr_8x1x1(t, t);
    fp2_8x1x1_to_4x2x1(res1, res2, t);
}

/**
 * Multiplication by i.
 */
void fp2_muli_4x2x1(__m512i* res, __m512i* a);

/**
 * Complex conjugation. res = a_re - a_im * i
 */
void fp2_conjugate_4x2x1(__m512i* res, __m512i* a);

/**
 * res = a / 2
 */
void fp2_div2_4x2x1(__m512i* res, __m512i* src);

/**
 * res = a / 3
 */
void fp2_div3_4x2x1(__m512i* res, __m512i* src);

/**
 * res = a^-1
 */
void fp2_inv_4x2x1(__m512i* res, __m512i* src);

/**
 * res = sqrt(a)
 */
void fp2_sqrt_4x2x1(__m512i* res, __m512i* a);

/**
 * Converts a to the Montgomery representation. res = a * R
 */
void fp2_tomont_4x2x1(__m512i* res, __m512i* a);

/**
 * Converts a back from the Montgomery representation. res = a * R^-1
 */
void fp2_frommont_4x2x1(__m512i* res, __m512i* a);

/**
 * Samples a random Fp2 element.
 */
void rand_fp2_4x2x1(__m512i* res);

// ====================================================================================================
//                                                2x2x2                                               
// ====================================================================================================

/*
 * The 2x2x2 format stores two Fp2 elements a and b in N4x2 vectors.
 * It stores (a_re, a_im, b_re, b_im) in 4x2.
 */

/**
 * Sets all the Fp2 elements to 0.
 */
void fp2_setzero_2x2x2(__m512i* res);

/**
 * If bits[i] = 1, the i-th Fp2 element is set to 1 (in Montgomery repr.)
 */
void fp2_set_binary_2x2x2(__m512i* res, __mmask8 bits);

/**
 * Copies a to res
 */
void fp2_copy_2x2x2(__m512i* res, __m512i* a);

/**
 * Returns 2-bit mask where the i-th bit is 1 iff the i-th Fp2 elements are equal.
 */
__mmask8 fp2_is_equal_2x2x2(__m512i* a, __m512i* b);

/**
 * Returns 2-bit mask where the i-th bit is 1 iff the i-th Fp2 element is zero.
 */
__mmask8 fp2_is_zero_2x2x2(__m512i* a);

/**
 * Copies the i-th Fp2 element from a1 to res if the i-th bit of s is set, else it copies from a0.
 */
void fp2_select_2x2x2(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);

/**
 * res = a + b
 */
static inline void fp2_add_2x2x2(__m512i* res, __m512i* a, __m512i* b)
{
    fp_add_4x2(res, a, b);
}

/**
 * res = a - b
 */
static inline void fp2_sub_2x2x2(__m512i* res, __m512i* a, __m512i* b)
{
    fp_sub_4x2(res, a, b);
}

/**
 * res = -a
 */
static inline void fp2_neg_2x2x2(__m512i* res, __m512i* a)
{
    fp_neg_4x2(res, a);
}

/**
 * Mixed addition and subtraction. The operation applied is determined by add_mask.
 * Bit 2*i and 2*i+1 determine the operation for the real and imaginary coefficient of the i-th Fp2 element.
 */
static inline void fp2_mixaddsub_2x2x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    fp_mixaddsub_4x2(res, a, b, add_mask);
}

/**
 * res = a * b
 */
static inline void fp2_mul_2x2x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0[13], t1[13];
    
    for (__mmask8 i = 0; i < 4; i++) {
        t0[i] = VPERM256(a[i], 0x44); // a0, a0
        t1[i] = VPERM256(a[i], 0xee); // a1, a1
        t1[i + 4] = VPERM256(b[i], 0x4e); // b0, b1
    }
    mp_mul_4x2(t0, t0, b);
    mp_mul_4x2(t1, t1, t1+4);
    mp_mixaddsubx2_4x2(t0, t0, t1, 0xcc);
    rdc_4x2(res, t0);
}

/**
 * res = a^2
 */
static inline void fp2_sqr_2x2x2(__m512i* res, __m512i* a)
{
    __m512i s1[4], s2[4];
    
    for (__mmask8 i = 0; i < 4; i++) {
        s1[i] = VPERM256(a[i], 0x44); // a0, a0
        s2[i] = VPERM256(a[i], 0x4e); // a1, a0
    }
    mp_add_4x2(s1, s1, s2);
    for (__mmask8 i = 0; i < 4; i++) {
        s2[i] = VPERM256MASKZ(0x33, a[i], 0x4e); // a1, 0
    }
    mp_add_p_sub_4x2(s2, a, s2); // a0 - a1, a1
    fp_mul_4x2(res, s1, s2);
}

/**
 * Multiplication by i.
 */
void fp2_muli_2x2x2(__m512i* res, __m512i* a);

/**
 * Complex conjugation. res = a_re - a_im * i
 */
void fp2_conjugate_2x2x2(__m512i* res, __m512i* a);

/**
 * res = a / 2
 */
void fp2_div2_2x2x2(__m512i* res, __m512i* src);

/**
 * res = a / 3
 */
void fp2_div3_2x2x2(__m512i* res, __m512i* src);

/**
 * res = a^-1
 */
void fp2_inv_2x2x2(__m512i* res, __m512i* src);

/**
 * res = sqrt(a)
 */
void fp2_sqrt_2x2x2(__m512i* res, __m512i* a);

/**
 * Converts a to the Montgomery representation. res = a * R
 */
void fp2_tomont_2x2x2(__m512i* res, __m512i* a);

/**
 * Converts a back from the Montgomery representation. res = a * R^-1
 */
void fp2_frommont_2x2x2(__m512i* res, __m512i* a);

/**
 * Samples a random Fp2 element.
 */
void rand_fp2_2x2x2(__m512i* res);

// ====================================================================================================
//                                                2x4x1                                               
// ====================================================================================================

/*
 * The 2x4x1 format stores two Fp2 elements a and b in NUM_LIMBS vectors.
 * It stores (a_re, 0, a_im, 0, b_re, 0, b_im, 0) in 8x1.
 */

/**
 * Sets all the Fp2 elements to 0.
 */
void fp2_setzero_2x4x1(__m512i* res);

/**
 * If bits[i] = 1, the i-th Fp2 element is set to 1 (in Montgomery repr.)
 */
void fp2_set_binary_2x4x1(__m512i* res, __mmask8 bits);

/**
 * Copies a to res
 */
void fp2_copy_2x4x1(__m512i* res, __m512i* a);

/**
 * Returns 2-bit mask where the i-th bit is 1 iff the i-th Fp2 elements are equal.
 */
__mmask8 fp2_is_equal_2x4x1(__m512i* a, __m512i* b);

/**
 * Returns 2-bit mask where the i-th bit is 1 iff the i-th Fp2 element is zero.
 */
__mmask8 fp2_is_zero_2x4x1(__m512i* a);

/**
 * Copies the i-th Fp2 element from a1 to res if the i-th bit of s is set, else it copies from a0.
 */
void fp2_select_2x4x1(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);

/**
 * res = a + b
 */
static inline void fp2_add_2x4x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_add_8x1(res, a, b);
}

/**
 * res = a - b
 */
static inline void fp2_sub_2x4x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_sub_8x1(res, a, b);
}

/**
 * res = -a
 */
static inline void fp2_neg_2x4x1(__m512i* res, __m512i* a)
{
    fp_neg_8x1(res, a);
}

/**
 * Mixed addition and subtraction. The operation applied is determined by the even index bits of add_mask.
 * Bit 0 and 2 determine the operation applied to the real and imaginary coefficient of the first Fp2 element,bit 4 and 6 determine it for the second Fp2 element.
 */
static inline void fp2_mixaddsub_2x4x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    fp_mixaddsub_8x1(res, a, b, add_mask);
}

/**
 * res = a * b
 */
static inline void fp2_mul_2x4x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i s1[8], s2[8];
    for (__mmask8 i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0, a1, a1
        s2[i] = VPERM256(b[i], 0x28); // b0, b1, b1, b0
    }
    fp_mul_8x1(s1, s1, s2);
    for (__mmask8 i = 0; i < 8; i++) {
        s2[i] = VPERM256MASKZ(0x55, s1[i], 0x32);
        s1[i] = VPERM256MASKZ(0x55, s1[i], 0x10);
    }
    fp_mixaddsub_8x1(res, s1, s2, 0xcc);
}

/**
 * res = a^2
 */
static inline void fp2_sqr_2x4x1(__m512i* res, __m512i* a)
{
    __m512i s1[8], s2[8];
    for (__mmask8 i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0, a1, a1
        s2[i] = VPERM256(a[i], 0x28); // a0, a1, a1, a0
    }
    fp_mul_8x1(s1, s1, s2);
    for (__mmask8 i = 0; i < 8; i++) {
        s2[i] = VPERM256MASKZ(0x55, s1[i], 0x32);
        s1[i] = VPERM256MASKZ(0x55, s1[i], 0x10);
    }
    fp_mixaddsub_8x1(res, s1, s2, 0xcc);
}

/**
 * Multiplication by i.
 */
void fp2_muli_2x4x1(__m512i* res, __m512i* a);

/**
 * Complex conjugation. res = a_re - a_im * i
 */
void fp2_conjugate_2x4x1(__m512i* res, __m512i* a);

/**
 * res = a / 2
 */
void fp2_div2_2x4x1(__m512i* res, __m512i* src);

/**
 * res = a / 3
 */
void fp2_div3_2x4x1(__m512i* res, __m512i* src);

/**
 * res = a^-1
 */
void fp2_inv_2x4x1(__m512i* res, __m512i* src);

/**
 * res = sqrt(a)
 */
void fp2_sqrt_2x4x1(__m512i* res, __m512i* a);

/**
 * Converts a to the Montgomery representation. res = a * R
 */
void fp2_tomont_2x4x1(__m512i* res, __m512i* a);

/**
 * Converts a back from the Montgomery representation. res = a * R^-1
 */
void fp2_frommont_2x4x1(__m512i* res, __m512i* a);

/**
 * Samples a random Fp2 element.
 */
void rand_fp2_2x4x1(__m512i* res);

#endif
