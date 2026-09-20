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
    fp_setzero_8x1(res+5);
}

void fp2_set_binary_8x1x1(__m512i* res, __mmask8 bits)
{
    fp_set_binary_8x1(res, bits);
    fp_setzero_8x1(res + 5);
}

void fp2_copy_8x1x1(__m512i* res, __m512i* a)
{
    fp_copy_8x1(res, a);
    fp_copy_8x1(res+5, a+5);
}

__mmask8 fp2_is_equal_8x1x1(__m512i* a, __m512i* b)
{
    __mmask8 lo = fp_is_equal_8x1(a, b);
    __mmask8 hi = fp_is_equal_8x1(a+5, b+5);
    return lo & hi;
}

__mmask8 fp2_is_zero_8x1x1(__m512i* a)
{
    __mmask8 lo = fp_is_zero_8x1(a);
    __mmask8 hi = fp_is_zero_8x1(a+5);
    return lo & hi;
}

void fp2_select_8x1x1(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    fp_select_8x1(res, a0, a1, s);
    fp_select_8x1(res + 5, a0 + 5, a1 + 5, s);
}

void fp2_muli_8x1x1(__m512i* res, __m512i* a)
{
    __m512i t[5];
    
    fp_neg_8x1(t, a + 5);
    fp_copy_8x1(res + 5, a);
    fp_copy_8x1(res, t);
}

void fp2_conjugate_8x1x1(__m512i* res, __m512i* a)
{
    fp_copy_8x1(res, a);
    fp_neg_8x1(res + 5, a + 5);
}

void fp2_div2_8x1x1(__m512i* res, __m512i* src)
{
    fp_div2_8x1(res, src);
    fp_div2_8x1(res + 5, src + 5);
}

void fp2_div3_8x1x1(__m512i* res, __m512i* src)
{
    fp_div3_8x1(res, src);
    fp_div3_8x1(res + 5, src + 5);
}

void fp2_inv_8x1x1(__m512i* res, __m512i* a)
{
    __m512i t0[5], t1[5];
    fp_sqr_8x1(t0, a);
    fp_sqr_8x1(t1, a + 5);
    fp_add_8x1(t0, t0, t1);
    fp_inv_8x1(t0, t0);
    fp_mul_8x1(t1, a + 5, t0);
    fp_mul_8x1(res, a, t0);
    fp_neg_8x1(res + 5, t1);
}

void fp2_sqrt_8x1x1(__m512i* res, __m512i* a)
{
    __m512i x0[5], x1[5], t0[5], t1[5];
    __mmask8 is_zero;
    
    // From "Optimized One-Dimensional SQIsign Verification on Intel and
    // Cortex-M4" by Aardal et al: https://eprint.iacr.org/2024/1563
    
    // x0 = \delta = sqrt(a0^2 + a1^2)
    fp_sqr_8x1(x0, a);
    fp_sqr_8x1(x1, a + 5);
    fp_add_8x1(x0, x0, x1);
    fp_sqrt_8x1(x0, x0);
    // If a1 = 0, there is a risk of \delta = -a0, which makes x0 = 0 below.
    // In that case, we restore the value \delta = a0.
    is_zero = fp_is_zero_8x1(a + 5);
    fp_select_8x1(x0, x0, a, is_zero);
    // x0 = \delta + a0, t0 = 2 * x0.
    fp_add_8x1(x0, x0, a);
    fp_add_8x1(t0, x0, x0);
    // x1 = t0^(p-3)/4.
    fp_copy_8x1(x1, t0);
    fp_exp3div4_8x1(x1, x1);
    // x0 = x0 * x1, x1 = x1 * a1, t1 = (2x0)^2.
    fp_mul_8x1(x0, x0, x1);
    fp_mul_8x1(x1, x1, a + 5);
    fp_add_8x1(t1, x0, x0);
    fp_sqr_8x1(t1, t1);
    // If t1 = t0, return x0 + x1*i, otherwise x1 - x0*i.
    fp_sub_8x1(t0, t0, t1);
    is_zero = fp_is_zero_8x1(t0);
    fp_neg_8x1(t1, x0);
    fp_copy_8x1(t0, x1);
    fp_select_8x1(res, t0, x0, is_zero);
    fp_select_8x1(res + 5, t1, x1, is_zero);
}

void fp2_tomont_8x1x1(__m512i* res, __m512i* a)
{
    fp_tomont_8x1(res, a);
    fp_tomont_8x1(res + 5, a + 5);
}

void fp2_frommont_8x1x1(__m512i*res, __m512i* a)
{
    fp_frommont_8x1(res, a);
    fp_frommont_8x1(res + 5, a + 5);
}

void rand_fp2_8x1x1(__m512i* res)
{
    rand_fp_8x1(res);
    rand_fp_8x1(res+5);
}

// ====================================================================================================
//                                                4x1x2                                               
// ====================================================================================================

void fp2_setzero_4x1x2(__m512i* res)
{
    fp_setzero_4x2(res);
    fp_setzero_4x2(res+3);
}

void fp2_set_binary_4x1x2(__m512i* res, __mmask8 bits)
{
    fp_set_binary_4x2(res, bits);
    fp_setzero_8x1(res + 3);
}

void fp2_copy_4x1x2(__m512i* res, __m512i* a)
{
    fp_copy_4x2(res, a);
    fp_copy_4x2(res+3, a+3);
}

__mmask8 fp2_is_equal_4x1x2(__m512i* a, __m512i* b)
{
    __mmask8 lo = fp_is_equal_4x2(a, b);
    __mmask8 hi = fp_is_equal_4x2(a+3, b+3);
    return lo & hi;
}

__mmask8 fp2_is_zero_4x1x2(__m512i* a)
{
    __mmask8 lo = fp_is_zero_4x2(a);
    __mmask8 hi = fp_is_zero_4x2(a+3);
    return lo & hi;
}

void fp2_select_4x1x2(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    fp_select_4x2(res, a0, a1, s);
    fp_select_4x2(res + 3, a0 + 3, a1 + 3, s);
}

void fp2_muli_4x1x2(__m512i* res, __m512i* a)
{
    __m512i t[3];
    
    fp_neg_4x2(t, a + 3);
    fp_copy_4x2(res + 3, a);
    fp_copy_4x2(res, t);
}

void fp2_conjugate_4x1x2(__m512i* res, __m512i* a)
{
    fp_copy_4x2(res, a);
    fp_neg_4x2(res + 3, a + 3);
}

void fp2_div2_4x1x2(__m512i* res, __m512i* src)
{
    fp_div2_4x2(res, src);
    fp_div2_4x2(res + 3, src + 3);
}

void fp2_div3_4x1x2(__m512i* res, __m512i* src)
{
    fp_div3_4x2(res, src);
    fp_div3_4x2(res + 3, src + 3);
}

void fp2_inv_4x1x2(__m512i* res, __m512i* a)
{
    __m512i t0[3], t1[3];
    fp_sqr_4x2(t0, a);
    fp_sqr_4x2(t1, a + 3);
    fp_add_4x2(t0, t0, t1);
    fp_inv_4x2(t0, t0);
    fp_mul_4x2(t1, a + 3, t0);
    fp_mul_4x2(res, a, t0);
    fp_neg_4x2(res + 3, t1);
}

void fp2_sqrt_4x1x2(__m512i* res, __m512i* a)
{
    __m512i x0[3], x1[3], t0[3], t1[3];
    __mmask8 is_zero;
    
    // From "Optimized One-Dimensional SQIsign Verification on Intel and
    // Cortex-M4" by Aardal et al: https://eprint.iacr.org/2024/1563
    
    // x0 = \delta = sqrt(a0^2 + a1^2)
    fp_sqr_4x2(x0, a);
    fp_sqr_4x2(x1, a + 3);
    fp_add_4x2(x0, x0, x1);
    fp_sqrt_4x2(x0, x0);
    // If a1 = 0, there is a risk of \delta = -a0, which makes x0 = 0 below.
    // In that case, we restore the value \delta = a0.
    is_zero = fp_is_zero_4x2(a + 3);
    fp_select_4x2(x0, x0, a, is_zero);
    // x0 = \delta + a0, t0 = 2 * x0.
    fp_add_4x2(x0, x0, a);
    fp_add_4x2(t0, x0, x0);
    // x1 = t0^(p-3)/4.
    fp_copy_4x2(x1, t0);
    fp_exp3div4_4x2(x1, x1);
    // x0 = x0 * x1, x1 = x1 * a1, t1 = (2x0)^2.
    fp_mul_4x2(x0, x0, x1);
    fp_mul_4x2(x1, x1, a + 3);
    fp_add_4x2(t1, x0, x0);
    fp_sqr_4x2(t1, t1);
    // If t1 = t0, return x0 + x1*i, otherwise x1 - x0*i.
    fp_sub_4x2(t0, t0, t1);
    is_zero = fp_is_zero_4x2(t0);
    fp_neg_4x2(t1, x0);
    fp_copy_4x2(t0, x1);
    fp_select_4x2(res, t0, x0, is_zero);
    fp_select_4x2(res + 3, t1, x1, is_zero);
}

void fp2_tomont_4x1x2(__m512i* res, __m512i* a)
{
    fp_tomont_4x2(res, a);
    fp_tomont_4x2(res + 3, a + 3);
}

void fp2_frommont_4x1x2(__m512i*res, __m512i* a)
{
    fp_frommont_4x2(res, a);
    fp_frommont_4x2(res + 3, a + 3);
}

void rand_fp2_4x1x2(__m512i* res)
{
    rand_fp_4x2(res);
    rand_fp_4x2(res + 3);
}

// ====================================================================================================
//                                                4x2x1                                               
// ====================================================================================================

void fp2_setzero_4x2x1(__m512i* res)
{
    fp_setzero_8x1(res);
}

void fp2_set_binary_4x2x1(__m512i* res, __mmask8 bits)
{
    __mmask8 bits_8x1 = (bits & 0x1) | ((bits & 0x2) << 1) | ((bits & 0x4) << 2) | ((bits & 0x8) << 3);
    fp_set_binary_8x1(res, bits_8x1);
}

void fp2_copy_4x2x1(__m512i* res, __m512i* a)
{
    fp_copy_8x1(res, a);
}

__mmask8 fp2_is_equal_4x2x1(__m512i* a, __m512i* b)
{
    __mmask8 mask = fp_is_equal_8x1(a, b);
    // a = b iff both halves are equal
    return (mask & (mask >> 1) & 0x1)
        | ((mask >> 1) & (mask >> 2) & 0x2)
        | ((mask >> 2) & (mask >> 3) & 0x4)
        | ((mask >> 3) & (mask >> 4) & 0x8);
}

__mmask8 fp2_is_zero_4x2x1(__m512i* a)
{
    __mmask8 mask = fp_is_zero_8x1(a);
    // a = 0 iff both halves are 0
    return (mask & (mask >> 1) & 0x1)
        | ((mask >> 1) & (mask >> 2) & 0x2)
        | ((mask >> 2) & (mask >> 3) & 0x4)
        | ((mask >> 3) & (mask >> 4) & 0x8);
}

void fp2_select_4x2x1(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    __mmask8 t = (s & 0x1) | ((s & 0x1) << 1) | ((s & 0x2) << 1) | ((s & 0x2) << 2) | ((s & 0x4) << 2) | ((s & 0x4) << 3) | ((s & 0x8) << 3) | ((s & 0x8) << 4);
    fp_select_8x1(res, a0, a1, t);
}

void fp2_muli_4x2x1(__m512i* res, __m512i* a)
{
    __m512i t[5];
    
    fp_setzero_8x1(t);
    fp_mixaddsub_8x1(t, t, a, 0x55);
    for (__mmask8 i = 0; i < 5; i++) {
        res[i] = VSHUFFLE(t[i], 0x4e);
    }
}

void fp2_conjugate_4x2x1(__m512i* res, __m512i* a)
{
    __m512i zero[5];
    fp_setzero_8x1(zero);
    
    fp_mixaddsub_8x1(res, zero, a, 0x55);
}

void fp2_div2_4x2x1(__m512i* res, __m512i* src)
{
    fp_div2_8x1(res, src);
}

void fp2_div3_4x2x1(__m512i* res, __m512i* src)
{
    fp_div3_8x1(res, src);
}

void fp2_inv_4x2x1(__m512i* res, __m512i* a)
{
    __m512i re[3], im[3], t0[3], t1[3];
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

void fp2_sqrt_4x2x1(__m512i* res, __m512i* a)
{
    __m512i re[3], im[3], x0[3], x1[3], t0[3], t1[3];
    __mmask8 is_zero;
    
    // From "Optimized One-Dimensional SQIsign Verification on Intel and
    // Cortex-M4" by Aardal et al: https://eprint.iacr.org/2024/1563
    
    fp_8x1_to_4x2(re, im, a);
    
    // x0 = \delta = sqrt(a0^2 + a1^2)
    fp_sqr_4x2(x0, re);
    fp_sqr_4x2(x1, im);
    fp_add_4x2(x0, x0, x1);
    fp_sqrt_4x2(x0, x0);
    // If a1 = 0, there is a risk of \delta = -a0, which makes x0 = 0 below.
    // In that case, we restore the value \delta = a0.
    is_zero = fp_is_zero_4x2(im);
    fp_select_4x2(x0, x0, re, is_zero);
    // x0 = \delta + a0, t0 = 2 * x0.
    fp_add_4x2(x0, x0, re);
    fp_add_4x2(t0, x0, x0);
    // x1 = t0^(p-3)/4.
    fp_copy_4x2(x1, t0);
    fp_exp3div4_4x2(x1, x1);
    // x0 = x0 * x1, x1 = x1 * a1, t1 = (2x0)^2.
    fp_mul_4x2(x0, x0, x1);
    fp_mul_4x2(x1, x1, im);
    fp_add_4x2(t1, x0, x0);
    fp_sqr_4x2(t1, t1);
    // If t1 = t0, return x0 + x1*i, otherwise x1 - x0*i.
    fp_sub_4x2(t0, t0, t1);
    is_zero = fp_is_zero_4x2(t0);
    fp_neg_4x2(t1, x0);
    fp_copy_4x2(t0, x1);
    fp_select_4x2(re, t0, x0, is_zero);
    fp_select_4x2(im, t1, x1, is_zero);
    
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

void fp2_set_binary_2x2x2(__m512i* res, __mmask8 bits)
{
    __mmask8 bits_4x2 = (bits & 0x1) | ((bits & 0x2) << 1);
    fp_set_binary_4x2(res, bits_4x2);
}

void fp2_copy_2x2x2(__m512i* res, __m512i* a)
{
    fp_copy_4x2(res, a);
}

__mmask8 fp2_is_equal_2x2x2(__m512i* a, __m512i* b)
{
    __mmask8 mask = fp_is_equal_4x2(a, b);
    return (mask & (mask >> 1) & 0x1 ) | ((mask >> 1) & (mask >> 2) & 0x2); 
}

__mmask8 fp2_is_zero_2x2x2(__m512i* a)
{
    __mmask8 mask = fp_is_zero_4x2(a);
    return (mask & (mask >> 1) & 0x1 ) | ((mask >> 1) & (mask >> 2) & 0x2);
}

void fp2_select_2x2x2(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    __mmask8 t = (s & 0x1) | ((s & 0x1) << 1) | ((s & 0x2) << 1) | ((s & 0x2) << 2);
    fp_select_4x2(res, a0, a1, t);
}

void fp2_muli_2x2x2(__m512i* res, __m512i* a)
{
    __m512i t[3];
    
    fp_setzero_4x2(t);
    fp_mixaddsub_4x2(t, t, a, 0x33);
    for (__mmask8 i = 0; i < 3; i++) {
        res[i] = VPERM256(t[i], 0x4e);
    }
}

void fp2_conjugate_2x2x2(__m512i* res, __m512i* a)
{
    __m512i zero[3];
    fp_setzero_4x2(zero);
    
    fp_mixaddsub_4x2(res, zero, a, 0x33);
}

void fp2_div2_2x2x2(__m512i* res, __m512i* src)
{
    fp_div2_4x2(res, src);
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

void fp2_set_binary_2x4x1(__m512i* res, __mmask8 bits)
{
    __mmask8 bits_8x1 = (bits & 0x1) | ((bits & 0x2) << 3);
    fp_set_binary_8x1(res, bits_8x1);
}

void fp2_copy_2x4x1(__m512i* res, __m512i* a)
{
    fp_copy_8x1(res, a);
}

__mmask8 fp2_is_equal_2x4x1(__m512i* a, __m512i* b)
{
    __mmask8 mask = fp_is_equal_8x1(a, b);
    return (mask & (mask >> 2) & 0x1)
        | ((mask >> 3) & (mask >> 5) & 0x2);
}

__mmask8 fp2_is_zero_2x4x1(__m512i* a)
{
    __mmask8 mask = fp_is_zero_8x1(a);
    return (mask & (mask >> 2) & 0x1)
        | ((mask >> 3) & (mask >> 5) & 0x2);
}

void fp2_select_2x4x1(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    __mmask8 t = (s & 0x1) | ((s & 0x1) << 2) | ((s & 0x2) << 3) | ((s & 0x2) << 5);
    fp_select_8x1(res, a0, a1, t);
}

void fp2_muli_2x4x1(__m512i* res, __m512i* a)
{
    __m512i t[5];
    
    fp_setzero_8x1(t);
    fp_mixaddsub_8x1(t, t, a, 0x33);
    for (__mmask8 i = 0; i < 5; i++) {
        res[i] = VPERM256(t[i], 0x4e);
    }
}

void fp2_conjugate_2x4x1(__m512i* res, __m512i* a)
{
    __m512i zero[5];
    fp_setzero_8x1(zero);
    
    fp_mixaddsub_8x1(res, zero, a, 0x33);
}

void fp2_div2_2x4x1(__m512i* res, __m512i* src)
{
    fp_div2_8x1(res, src);
}

void fp2_div3_2x4x1(__m512i* res, __m512i* src)
{
    fp_div3_8x1(res, src);
}

void fp2_inv_2x4x1(__m512i* res, __m512i* src)
{
    __m512i re[3], im[3], zero[3], t0[2], t1[2];
    
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

void fp2_sqrt_2x4x1(__m512i* res, __m512i* a)
{
    __m512i re[3], im[2], x0[2], x1[2], t0[2], t1[2];
    __m512i zero[3];
    __mmask8 is_zero;
    
    // From "Optimized One-Dimensional SQIsign Verification on Intel and
    // Cortex-M4" by Aardal et al: https://eprint.iacr.org/2024/1563
    
    fp_8x1_to_4x2(re, zero, a);
    fp_4x2_to_2x4(re, im, re);
    
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
    fp_4x2_to_8x1(res, res, zero);
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
    for(__mmask8 i = 0; i < 5; i++) {
        res[i] = VBLEND(0x55, VZERO(), res[i]);
    }
}

// ====================================================================================================
//                               Conversions between Fp2 representations                              
// ====================================================================================================

void fp2_8x1x1_to_4x2x1(__m512i* res1, __m512i* res2, __m512i* src)
{
    for (__mmask8 i = 0; i < 5; i++) {
        __m512i re = src[i], im = src[i + 5];
        res1[i] = VUNPACKLO(re, im);
        res2[i] = VUNPACKHI(re, im);
    }
}

// Only extracts first 4x2x1
void fp2_8x1x1_to_first_4x2x1(__m512i* res, __m512i* src)
{
    for (__mmask8 i = 0; i < 5; i++) {
        res[i] = VUNPACKLO(src[i], src[i + 5]);
    }
}

void fp2_4x2x1_to_8x1x1(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i idx_re = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
    __m512i idx_im = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
    for (__mmask8 i = 0; i < 5; i++) {
        __m512i s1 = src1[i], s2 = src2[i];
        res[i] = VBLENDPERM(idx_re, s1, s2);
        res[i+5] = VBLENDPERM(idx_im, s1, s2);
    }
}

void fp2_4x2x1_duplicate_to_8x1x1(__m512i* res, __m512i* src)
{
    __m512i idx_re = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
    __m512i idx_im = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);
    for (__mmask8 i = 0; i < 5; i++) {
        __m512i s = src[i];
        res[i] = VBLENDPERM(idx_re, s, s);
        res[i+5] = VBLENDPERM(idx_im, s, s);
    }
}

void fp2_4x2x1_to_2x2x2(__m512i* res1, __m512i* res2, __m512i* src)
{
    __m512i s, t;
    __m512i idx1 = VSET(0, 8, 1, 9, 2, 10, 3, 11);
    __m512i idx2 = VSET(4, 12, 5, 13, 6, 14, 7, 15);
    for (__mmask8 i = 0; i < 2; i++) {
        s = src[i], t = src[i+3];
        res1[i] = VBLENDPERM(idx1, s, t);
        res2[i] = VBLENDPERM(idx2, s, t);
    }
    s = src[2];
    res1[2] = VBLENDPERM(idx1, s, VZERO());
    res2[2] = VBLENDPERM(idx2, s, VZERO());
}

void fp2_2x2x2_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i s, t;
    __m512i idx1 = VSET(0, 2, 4, 6, 8, 10, 12, 14);
    __m512i idx2 = VSET(1, 3, 5, 7, 9, 11, 13, 15);
    for (__mmask8 i = 0; i < 2; i++) {
        s = src1[i], t = src2[i];
        res[i] = VBLENDPERM(idx1, s, t);
        res[i+3] = VBLENDPERM(idx2, s, t);
    }
    res[2] = VBLENDPERM(idx1, src1[2], src2[2]);
}

// Maps (a,b,c,d) in 4x2x1 to (a,b) and (c,d) in 2x4x1
void fp2_4x2x1_to_2x4x1(__m512i* res1, __m512i* res2, __m512i* src)
{
    __m512i idx1 = VSET(0, 0, 1, 1, 2, 2, 3, 3);
    __m512i idx2 = VSET(4, 4, 5, 5, 6, 6, 7, 7);
    for (__mmask8 i = 0; i < 5; i++) {
        __m512i t = src[i];
        res1[i] = VPERM512MASKZ(0x55, t, idx1);
        res2[i] = VPERM512MASKZ(0x55, t, idx2);
    }
}

void fp2_2x4x1_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i idx = VSET(0, 2, 4, 6, 8, 10, 12, 14);
    for (__mmask8 i = 0; i < 5; i++) {
        res[i] = VBLENDPERM(idx, src1[i], src2[i]);
    }
}

void fp2_2x2x2_to_2x4x1(__m512i* res, __m512i* src)
{
    __m512i idx1 = VSET(0, 0, 2, 2, 4, 4, 6, 6);
    __m512i idx2 = VSET(1, 1, 3, 3, 5, 5, 7, 7);
    for (__mmask8 i = 0; i < 2; i++) {
        __m512i t = src[i];
        res[i] = VPERM512MASKZ(0x55, t, idx1);
        res[i+3] = VPERM512MASKZ(0x55, t, idx2);
    }
    res[2] = VPERM512MASKZ(0x55, src[2], idx1);
}

void fp2_2x4x1_to_2x2x2(__m512i* res, __m512i* src)
{
    __m512i idx = VSET(0, 8, 2, 10, 4, 12, 6, 14);
    for (__mmask8 i = 0; i < 2; i++) {
        res[i] = VBLENDPERM(idx, src[i], src[i+3]);
    }
    res[2] = VBLENDPERM(idx, src[2], VZERO());
}

