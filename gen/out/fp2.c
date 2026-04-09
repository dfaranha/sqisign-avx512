#include "fp.h"
#include "fp_consts.h"
#include "fp2.h"
#include "utils.h"

// ====================================================================================================
//                                                8x1x1                                               
// ====================================================================================================

void fp2_setzero_8x1x1(__m512i* res)
{
    fp_setzero_8x1(res);
    fp_setzero_8x1(res+8);
}

void fp2_copy_8x1x1(__m512i* res, __m512i* a)
{
    fp_copy_8x1(res, a);
    fp_copy_8x1(res+8, a+8);
}

int fp2_is_equal_8x1x1(__m512i* a, __m512i* b)
{
    int lo = fp_is_equal_8x1(a, b);
    int hi = fp_is_equal_8x1(a+8, b+8);
    return lo & hi;
}

int fp2_is_zero_8x1x1(__m512i* a)
{
    int lo = fp_is_zero_8x1(a);
    int hi = fp_is_zero_8x1(a+8);
    return lo & hi;
}

void fp2_add_8x1x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_add_8x1(res, a, b);
    fp_add_8x1(res+8, a+8, b+8);
}

void fp2_sub_8x1x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_sub_8x1(res, a, b);
    fp_sub_8x1(res+8, a+8, b+8);
}

void fp2_neg_8x1x1(__m512i* res, __m512i* a)
{
    fp_neg_8x1(res, a);
    fp_neg_8x1(res+8, a+8);
}

void fp2_mixaddsub_8x1x1(__m512i* res, __m512i* a, __m512i* b,
    const __mmask8 add_mask_re, const __mmask8 add_mask_im)
{
    fp_mixaddsub_8x1(res, a, b, add_mask_re);
    fp_mixaddsub_8x1(res+8, a+8, b+8, add_mask_im);
}

void fp2_mul_8x1x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i tt1[16], tt2[16], tt3[16];
    
    mp_add_8x1(tt3, a, a + 8);
    mp_add_8x1(tt3 + 8, b, b + 8);
    mp_mul_8x1_karat(tt1, a, b);
    mp_mul_8x1_karat(tt2, a + 8, b + 8);
    mp_mul_8x1_karat(tt3, tt3, tt3 + 8);
    mp_dblsubx2_8x1(tt3, tt1, tt2);
    mp_subx2_8x1(tt1, tt2);
    rdc_8x1(res + 8, tt3);
    rdc_8x1(res, tt1);
}

void fp2_sqr_8x1x1(__m512i* res, __m512i* a)
{
    __m512i s1[8], s2[8], s3[8];
    
    mp_add_8x1(s1, a, a + 8);
    mp_add_p_sub_8x1(s2, a, a + 8);
    mp_add_8x1(s3, a, a);
    fp_mul_8x1(res, s1, s2);
    fp_mul_8x1(res + 8, s3, a + 8);
}

void fp2_muli_8x1x1(__m512i* res, __m512i* a)
{
    __m512i t[8];
    
    fp_neg_8x1(t, a + 8);
    fp_copy_8x1(res + 8, a);
    fp_copy_8x1(res, t);
}

void fp2_div3_8x1x1(__m512i* res, __m512i* src)
{
    fp_div3_8x1(res, src);
    fp_div3_8x1(res + 8, src + 8);
}

void fp2_inv_8x1x1(__m512i* res, __m512i* a)
{
    __m512i t0[8], t1[8];
    fp_sqr_8x1(t0, a);
    fp_sqr_8x1(t1, a + 8);
    fp_add_8x1(t0, t0, t1);
    fp_inv_8x1(t0, t0);
    fp_mul_8x1(t1, a + 8, t0);
    fp_mul_8x1(res, a, t0);
    fp_neg_8x1(res + 8, t1);
}

void fp2_tomont_8x1x1(__m512i* res, __m512i* a)
{
    fp_tomont_8x1(res, a);
    fp_tomont_8x1(res + 8, a + 8);
}

void fp2_frommont_8x1x1(__m512i*res, __m512i* a)
{
    fp_frommont_8x1(res, a);
    fp_frommont_8x1(res + 8, a + 8);
}

void rand_fp2_8x1x1(__m512i* res)
{
    rand_fp_8x1(res);
    rand_fp_8x1(res+8);
}

// ====================================================================================================
//                                                4x1x2                                               
// ====================================================================================================

void fp2_setzero_4x1x2(__m512i* res)
{
    fp_setzero_4x2(res);
    fp_setzero_4x2(res+4);
}

void fp2_copy_4x1x2(__m512i* res, __m512i* a)
{
    fp_copy_4x2(res, a);
    fp_copy_4x2(res+4, a+4);
}

int fp2_is_equal_4x1x2(__m512i* a, __m512i* b)
{
    int lo = fp_is_equal_4x2(a, b);
    int hi = fp_is_equal_4x2(a+4, b+4);
    return lo & hi;
}

int fp2_is_zero_4x1x2(__m512i* a)
{
    int lo = fp_is_zero_4x2(a);
    int hi = fp_is_zero_4x2(a+4);
    return lo & hi;
}

void fp2_add_4x1x2(__m512i* res, __m512i* a, __m512i* b)
{
    fp_add_4x2(res, a, b);
    fp_add_4x2(res+4, a+4, b+4);
}

void fp2_sub_4x1x2(__m512i* res, __m512i* a, __m512i* b)
{
    fp_sub_4x2(res, a, b);
    fp_sub_4x2(res+4, a+4, b+4);
}

void fp2_neg_4x1x2(__m512i* res, __m512i* a)
{
    fp_neg_4x2(res, a);
    fp_neg_4x2(res+4, a+4);
}

void fp2_mixaddsub_4x1x2(__m512i* res, __m512i* a, __m512i* b,
    const __mmask8 add_mask_re, const __mmask8 add_mask_im)
{
    fp_mixaddsub_4x2(res, a, b, add_mask_re);
    fp_mixaddsub_4x2(res+4, a+4, b+4, add_mask_im);
}

void fp2_mul_4x1x2(__m512i* res, __m512i* a, __m512i* b)
{
    t1[13], t2[13], t3[13];
    
    mp_add_4x2(t3, a, a + 4);
    fp_add_4x2(t3 + 4, b, b + 4);
    mp_mul_4x2(t1, a, b);
    mp_mul_4x2(t2, a+4, b+4);
    mp_mul_4x2(t3, t3, t3+4);
    mp_dblsubx2_4x2(t3, t1, t2);
    mp_subx2_4x2(t1, t2);
    rdc_4x2(res+4, t3);
    rdc_4x2(res, t1);
}

void fp2_sqr_4x1x2(__m512i* res, __m512i* a)
{
    __m512i s1[4], s2[4], s3[4];
    
    mp_add_4x2(s1, a, a + 4);
    mp_app_p_sub_4x2(s2, a, a + 4);
    mp_add_4x2(s3, a, a);
    fp_mul_4x2(res + 4, s3, a + 4);
    fp_mul_4x2(res, s1, s2);
}

void fp2_muli_4x1x2(__m512i* res, __m512i* a)
{
    __m512i t[4];
    
    fp_neg_4x2(t, a + 4);
    fp_copy_4x2(res + 4, a);
    fp_copy_4x2(res, t);
}

void fp2_div3_4x1x2(__m512i* res, __m512i* src)
{
    fp_div3_4x2(res, src);
    fp_div3_4x2(res + 4, src + 4);
}

void fp2_inv_4x1x2(__m512i* res, __m512i* a)
{
    __m512i t0[4], t1[4];
    fp_sqr_4x2(t0, a);
    fp_sqr_4x2(t1, a + 4);
    fp_add_4x2(t0, t0, t1);
    fp_inv_4x2(t0, t0);
    fp_mul_4x2(t1, a + 4, t0);
    fp_mul_4x2(res, a, t0);
    fp_neg_4x2(res + 4, t1);
}

void fp2_tomont_4x1x2(__m512i* res, __m512i* a)
{
    fp_tomont_4x2(res, a);
    fp_tomont_4x2(res + 4, a + 4);
}

void fp2_frommont_4x1x2(__m512i*res, __m512i* a)
{
    fp_frommont_4x2(res, a);
    fp_frommont_4x2(res + 4, a + 4);
}

void rand_fp2_4x1x2(__m512i* res)
{
    rand_fp_4x2(res);
    rand_fp_4x2(res + 4);
}

// ====================================================================================================
//                                                4x2x1                                               
// ====================================================================================================

void fp2_setzero_4x2x1(__m512i* res)
{
    fp_setzero_8x1(res);
}

void fp2_copy_4x2x1(__m512i* res, __m512i* a)
{
    fp_copy_8x1(res, a);
}

int fp2_is_equal_4x2x1(__m512i* a, __m512i* b)
{
    int mask = fp_is_equal_8x1(a, b);
    // a = b iff both halves are equal
    return (mask & (mask >> 1) & 0x1)
        | ((mask >> 1) & (mask >> 2) & 0x2)
        | ((mask >> 2) & (mask >> 3) & 0x4)
        | ((mask >> 3) & (mask >> 4) & 0x8);
}

int fp2_is_zero_4x2x1(__m512i* a)
{
    int mask = fp_is_zero_8x1(a);
    // a = 0 iff both halves are 0
    return (mask & (mask >> 1) & 0x1)
        | ((mask >> 1) & (mask >> 2) & 0x2)
        | ((mask >> 2) & (mask >> 3) & 0x4)
        | ((mask >> 3) & (mask >> 4) & 0x8);
}

void fp2_add_4x2x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_add_8x1(res, a, b);
}

void fp2_sub_4x2x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_sub_8x1(res, a, b);
}

void fp2_neg_4x2x1(__m512i* res, __m512i* a)
{
    fp_neg_8x1(res, a);
}

void fp2_mixaddsub_4x2x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    fp_mixaddsub_8x1(res, a, b, add_mask);
}

void fp2_mul_4x2x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0[16], t1[16];
    for (int i = 0; i < 8; i++) {
        t0[i] = VSHUFFLE(a[i], 0x44); // a0, a0
        t1[i] = VSHUFFLE(a[i], 0xee); // a1, a1
        t1[i + 8] = VSHUFFLE(b[i], 0x4e); // b0, b1
    }
    mp_mul_8x1_karat(t0, t0, b);
    mp_mul_8x1_karat(t1, t1, t1 + 8);
    mp_mixaddsubx2_8x1(res, t0, t1, 0xaa);
    rdc_8x1(res, t0);
}

void fp2_sqr_4x2x1(__m512i* res, __m512i* a)
{
    __m512i s1[8], s2[8], s3[8];
    __mmask16 mask = 0b0011001100110011;
    for (int i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0
        s2[i] = VSHUFFLEMASKZ(mask, a[i], 0x0e); // 0, a1
        s3[i] = VSHUFFLE(a[i], 0x4e); // a0, a1
    }
    mp_add_8x1(s1, s1, s3);
    mp_add_p_sub_8x1(s2, a, s2);
    fp_mul_8x1(res, s1, s2);
}

// res1 = a1 * b1, res2 = a2 * b2
void fp2_mulx2_4x2x1(__m512i* res1, __m512i* res2, __m512i* a1, __m512i* b1, __m512i* a2, __m512i* b2)
{
    __m512i t0[16], t1[16];
    
    fp2_4x2x1_to_8x1x1(t0, a1, a2);
    fp2_4x2x1_to_8x1x1(t1, b1, b2);
    fp2_mul_8x1x1(t0, t0, t1);
    fp2_8x1x1_to_4x2x1(res1, res2, t0);
}

// res1 = a1^2, res2 = a2^a
void fp2_sqrx2_4x2x1(__m512i* res1, __m512i* res2, __m512i* a1, __m512i* a2)
{
    __m512i t[16];
    
    fp2_4x2x1_to_8x1x1(t, a1, a2);
    fp2_sqr_8x1x1(t, t);
    fp2_8x1x1_to_4x2x1(res1, res2, t);
}

void fp2_muli_4x2x1(__m512i* res, __m512i* a)
{
    __m512i t[8];
    
    fp_setzero_8x1(t);
    fp_mixaddsub_8x1(t, t, a, 0x55);
    for (int i = 0; i < 8; i++) {
        res[i] = VSHUFFLE(t[i], 0x4e);
    }
}

void fp2_div3_4x2x1(__m512i* res, __m512i* src)
{
    fp_div3_8x1(res, src);
}

void fp2_inv_4x2x1(__m512i* res, __m512i* a)
{
    __m512i re[4], im[4], t0[4], t1[4];
    fp_8x1_to_4x2(re, im, a);
    
    fp_sqr_4x2(t0, re);
    fp_sqr_4x2(t1, im);
    fp_add_4x2(t0, t0, t1);
    fp_inv_4x2(t0, t0);
    fp_mul_4x2(re, re, t0);
    fp_mul_4x2(im, im, t0);
    fp_neg_4x2(im, im);
    
    fp_4x2_to_8x1(res, re, im);
}

void fp2_tomont_4x2x1(__m512i* res, __m512i* a)
{
    fp_tomont_8x1(res, a);
}

void fp2_frommont_4x2x1(__m512i*res, __m512i* a)
{
    fp_frommont_8x1(res, a);
}

void rand_fp2_4x2x1(__m512i* res)
{
    rand_fp_8x1(res);
}

// ====================================================================================================
//                                                2x2x2                                               
// ====================================================================================================

void fp2_setzero_2x2x2(__m512i* res)
{
    fp_setzero_4x2(res);
}

void fp2_copy_2x2x2(__m512i* res, __m512i* a)
{
    fp_copy_4x2(res, a);
}

int fp2_is_equal_2x2x2(__m512i* a, __m512i* b)
{
    int mask = fp_is_equal_4x2(a, b);
    return (mask & (mask >> 1) & 0x1 ) | ((mask >> 1) & (mask >> 2) & 0x2); 
}

int fp2_is_zero_2x2x2(__m512i* a)
{
    int mask = fp_is_zero_4x2(a);
    return (mask & (mask >> 1) & 0x1 ) | ((mask >> 1) & (mask >> 2) & 0x2);
}

void fp2_add_2x2x2(__m512i* res, __m512i* a, __m512i* b)
{
    fp_add_4x2(res, a, b);
}

void fp2_sub_2x2x2(__m512i* res, __m512i* a, __m512i* b)
{
    fp_sub_4x2(res, a, b);
}

void fp2_neg_2x2x2(__m512i* res, __m512i* a)
{
    fp_neg_4x2(res, a);
}

void fp2_mixaddsub_2x2x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    fp_mixaddsub_4x2(res, a, b, add_mask);
}

void fp2_mul_2x2x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0[13], s2[13], s3[13];
    
    for (int i = 0; i < 4; i++) {
        t0[i] = VPERM256(a[i], 0x44); // a0, a0
        t1[i] = VPERM256(a[i], 0xee); // a1, a1
        t1[i + 4] = VPERM256(b[i], 0x4e); // b0, b1
    }
    mp_mul_4x2(t0, t0, b);
    mp_mul_4x2(t1, t1, t1+4);
    mp_mixaddsubx2_4x2(t0, t0, t1, 0xcc);
    rdc_4x2(res, t0);
}

void fp2_sqr_2x2x2(__m512i* res, __m512i* a)
{
    __m512i s1[4], s2[4];
    
    for (int i = 0; i < 4; i++) {
        s1[i] = VPERM256(a[i], 0x44); // a0, a0
        s2[i] = VPERM256(a[i], 0x4e); // a1, a0
    }
    mp_add_4x2(s1, s1, s2);
    for (int i = 0; i < 4; i++) {
        s2[i] = VPERM256MASKZ(0x33, a[i], 0x4e); // a1, 0
    }
    mp_add_p_sub_4x2(s2, a, s2); // a0 - a1, a1
    fp_mul_4x2(res, s1, s2);
}

void fp2_muli_2x2x2(__m512i* res, __m512i* a)
{
    __m512i t[4];
    
    fp_setzero_4x2(t);
    fp_mixaddsub_4x2(t, t, a, 0x33);
    for (int i = 0; i < 4; i++) {
        res[i] = VPERM256(t[i], 0x4e);
    }
}

void fp2_div3_2x2x2(__m512i* res, __m512i* src)
{
    fp_div3_4x2(res, src);
}

void fp2_sqrt_2x2x2(__m512i* res, __m512i* a)
{
    __m512i re[2], im[2], x0[2], x1[2], t0[2], t1[2];
    __mmask8 is_zero;
    
    // From "Optimized One-Dimensional SQIsign Verification on Intel and
    // Cortex-M4" by Aardal et al: https://eprint.iacr.org/2024/1563
    
    fp_4x2_to_2x4(re, im, a);
    
    // x0 = \delta = sqrt(a0^2 + a1^2)
    fp_sqr_2x4(x0, re);
    fp_sqr_2x4(x1, im);
    fp_add_2x4(x0, x0, x1);
    fp_sqrt_2x4(x0, x0);
    // If a1 = 0, there is a risk of \delta = -a0, which makes x0 = 0 below.
    // In that case, we restore the value \delta = a0.
    is_zero = fp_is_zero_2x4(im);
    fp_select_2x4(x0, x0, re, is_zero);
    // x0 = \delta + a0, t0 = 2 * x0.
    fp_add_2x4(x0, x0, re);
    fp_add_2x4(t0, x0, x0);
    // x1 = t0^(p-3)/4.
    fp_copy_2x4(x1, t0);
    fp_exp3div4_2x4(x1, x1);
    // x0 = x0 * x1, x1 = x1 * a1, t1 = (2x0)^2.
    fp_mul_2x4(x0, x0, x1);
    fp_mul_2x4(x1, x1, im);
    fp_add_2x4(t1, x0, x0);
    fp_sqr_2x4(t1, t1);
    // If t1 = t0, return x0 + x1*i, otherwise x1 - x0*i.
    fp_sub_2x4(t0, t0, t1);
    is_zero = fp_is_zero_2x4(t0);
    fp_neg_2x4(t1, x0);
    fp_copy_2x4(t0, x1);
    fp_select_2x4(re, t0, x0, is_zero);
    fp_select_2x4(im, t1, x1, is_zero);
    
    fp_2x4_to_4x2(res, re, im);
}

void fp2_inv_2x2x2(__m512i* res, __m512i* a)
{
    __m512i re[2], im[2], t0[2], t1[2];
    fp_4x2_to_2x4(re, im, a);
    
    fp_sqr_2x4(t0, re);
    fp_sqr_2x4(t1, im);
    fp_add_2x4(t0, t0, t1);
    fp_inv_2x4(t0, t0);
    fp_mul_2x4(re, re, t0);
    fp_mul_2x4(im, im, t0);
    fp_neg_2x4(im, im);
    
    fp_2x4_to_4x2(res, re, im);
}

void fp2_tomont_2x2x2(__m512i* res, __m512i* a)
{
    fp_tomont_4x2(res, a);
}

void fp2_frommont_2x2x2(__m512i*res, __m512i* a)
{
    fp_frommont_4x2(res, a);
}

void rand_fp2_2x2x2(__m512i* res)
{
    rand_fp_4x2(res);
}

// ====================================================================================================
//                                                2x4x1                                               
// ====================================================================================================

void fp2_setzero_2x4x1(__m512i* res)
{
    fp_setzero_8x1(res);
}

void fp2_copy_2x4x1(__m512i* res, __m512i* a)
{
    fp_copy_8x1(res, a);
}

int fp2_is_equal_2x4x1(__m512i* a, __m512i* b)
{
    int mask = fp_is_equal_8x1(a, b);
    return (mask & (mask >> 2) & 0x1)
        | ((mask >> 3) & (mask >> 5) & 0x2);
}

int fp2_is_zero_2x4x1(__m512i* a)
{
    int mask = fp_is_zero_8x1(a);
    return (mask & (mask >> 2) & 0x1)
        | ((mask >> 3) & (mask >> 5) & 0x2);
}

void fp2_add_2x4x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_add_8x1(res, a, b);
}

void fp2_sub_2x4x1(__m512i* res, __m512i* a, __m512i* b)
{
    fp_sub_8x1(res, a, b);
}

void fp2_neg_2x4x1(__m512i* res, __m512i* a)
{
    fp_neg_8x1(res, a);
}

void fp2_mixaddsub_2x4x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    fp_mixaddsub_8x1(res, a, b, add_mask);
}

void fp2_mul_2x4x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i s1[8], s2[8];
    for (int i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0, a1, a1
        s2[i] = VPERM256(b[i], 0x28); // b0, b1, b1, b0
    }
    fp_mul_8x1(s1, s1, s2);
    for (int i = 0; i < 8; i++) {
        s2[i] = VPERM256MASKZ(0x55, s1[i], 0x32);
        s1[i] = VPERM256MASKZ(0x55, s1[i], 0x10);
    }
    fp_mixaddsub_8x1(res, s1, s2, 0xcc);
}

void fp2_sqr_2x4x1(__m512i* res, __m512i* a)
{
    __m512i s1[8], s2[8];
    for (int i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0, a1, a1
        s2[i] = VPERM256(a[i], 0x28); // a0, a1, a1, a0
    }
    fp_mul_8x1(s1, s1, s2);
    for (int i = 0; i < 8; i++) {
        s2[i] = VPERM256MASKZ(0x55, s1[i], 0x32);
        s1[i] = VPERM256MASKZ(0x55, s1[i], 0x10);
    }
    fp_mixaddsub_8x1(res, s1, s2, 0xcc);
}

void fp2_muli_2x4x1(__m512i* res, __m512i* a)
{
    __m512i t[8];
    
    fp_setzero_8x1(t);
    fp_mixaddsub_8x1(t, t, a, 0x33);
    for (int i = 0; i < 8; i++) {
        res[i] = VPERM256(t[i], 0x4e);
    }
}

void fp2_div3_2x4x1(__m512i* res, __m512i* src)
{
    fp_div3_8x1(res, src);
}

void fp2_inv_2x4x1(__m512i* res, __m512i* src)
{
    __m512i re[4], im[4], zero[4], t0[2], t1[2];
    
    fp_8x1_to_4x2(re, zero, src); // re = (a_re, a_im, b_re, b_im), zero = (0, 0, 0, 0)
    fp_4x2_to_2x4(re, im, re);
    
    fp_sqr_2x4(t0, re);
    fp_sqr_2x4(t1, im);
    fp_add_2x4(t0, t0, t1);
    fp_inv_2x4(t0, t0);
    fp_mul_2x4(re, re, t0);
    fp_mul_2x4(im, im, t0);
    fp_neg_2x4(im, im);
    
    fp_2x4_to_4x2(re, re, im);
    fp_4x2_to_8x1(res, re, zero);
}

void fp2_tomont_2x4x1(__m512i* res, __m512i* a)
{
    fp_tomont_8x1(res, a);
}

void fp2_frommont_2x4x1(__m512i*res, __m512i* a)
{
    fp_frommont_8x1(res, a);
}

void rand_fp2_2x4x1(__m512i* res)
{
    rand_fp_8x1(res);
    for(int i = 0; i < 8; i++) {
        res[i] = VBLEND(0x55, VZERO(), res[i]);
    }
}

// ====================================================================================================
//                               Conversions between Fp2 representations                              
// ====================================================================================================

void fp2_8x1x1_to_4x2x1(__m512i* res1, __m512i* res2, __m512i* src)
{
    for (int i = 0; i < 8; i++) {
        __m512i re = src[i], im = src[i + 8];
        res1[i] = VUNPACKLO(re, im);
        res2[i] = VUNPACKHI(re, im);
    }
}

// Only extracts first 4x2x1
void fp2_8x1x1_to_first_4x2x1(__m512i* res, __m512i* src)
{
    for (int i = 0; i < 8; i++) {
        res[i] = VUNPACKLO(src[i], src[i + 8]);
    }
}

void fp2_4x2x1_to_8x1x1(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i idx_re = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
    __m512i idx_im = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
    for (int i = 0; i < 8; i++) {
        __m512i s1 = src1[i], s2 = src2[i];
        res[i] = VBLENDPERM(idx_re, s1, s2);
        res[i+8] = VBLENDPERM(idx_im, s1, s2);
    }
}

void fp2_4x2x1_duplicate_to_8x1x1(__m512i* res, __m512i* src)
{
    __m512i idx_re = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
    __m512i idx_im = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
    for (int i = 0; i < 8; i++) {
        __m512i s = src[i];
        res[i] = VBLENDPERM(idx_re, s, s);
        res[i+8] = VBLENDPERM(idx_im, s, s);
    }
}

void fp2_4x2x1_to_2x2x2(__m512i* res1, __m512i* res2, __m512i* src)
{
    __m512i s, t;
    __m512i idx1 = VSET(0, 8, 1, 9, 2, 10, 3, 11);
    __m512i idx2 = VSET(4, 12, 5, 13, 6, 14, 7, 15);
    for (int i = 0; i < 4; i++) {
        s = src[i], t = src[i+4];
        res1[i] = VBLENDPERM(idx1, s, t);
        res2[i] = VBLENDPERM(idx2, s, t);
    }
}

void fp2_2x2x2_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i s, t;
    __m512i idx1 = VSET(0, 2, 4, 6, 8, 10, 12, 14);
    __m512i idx2 = VSET(1, 3, 5, 7, 9, 11, 13, 15);
    for (int i = 0; i < 4; i++) {
        s = src1[i], t = src2[i];
        res[i] = VBLENDPERM(idx1, s, t);
        res[i+4] = VBLENDPERM(idx2, s, t);
    }
}

// Maps (a,b,c,d) in 4x2x1 to (a,b) and (c,d) in 2x4x1
void fp2_4x2x1_to_2x4x1(__m512i* res1, __m512i* res2, __m512i* src)
{
    __m512i idx1 = VSET(0, 0, 1, 1, 2, 2, 3, 3);
    __m512i idx2 = VSET(4, 4, 5, 5, 6, 6, 7, 7);
    for (int i = 0; i < 8; i++) {
        __m512i t = src[i];
        res1[i] = VPERM512MASKZ(0x55, t, idx1);
        res2[i] = VPERM512MASKZ(0x55, t, idx2);
    }
}

void fp2_2x4x1_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i idx = VSET(0, 2, 4, 6, 8, 10, 12, 14);
    for (int i = 0; i < 8; i++) {
        res[i] = VBLENDPERM(idx, src1[i], src2[i]);
    }
}

void fp2_2x2x2_to_2x4x1(__m512i* res, __m512i* src)
{
    __m512i idx1 = VSET(0, 0, 2, 2, 4, 4, 6, 6);
    __m512i idx2 = VSET(1, 1, 3, 3, 5, 5, 7, 7);
    for (int i = 0; i < 4; i++) {
        __m512i t = src[i];
        res[i] = VPERM512MASKZ(0x55, t, idx1);
        res[i+4] = VPERM512MASKZ(0x55, t, idx2);
    }
}

void fp2_2x4x1_to_2x2x2(__m512i* res, __m512i* src)
{
    __m512i idx = VSET(0, 8, 2, 10, 4, 12, 6, 14);
    for (int i = 0; i < 4; i++) {
        res[i] = VBLENDPERM(idx, src[i], src[i+4]);
    }
}

