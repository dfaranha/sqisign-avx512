#include "fp_lvl3.h"
#include "fp2_lvl3.h"

// ======================================================================================================
//                                                 8x1x1
// ======================================================================================================

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

void fp2_mul_8x1x1v2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i s1[8], s2[8], t1[8], t2[8], t3[8];
    fp_add_8x1(s1, a, a+8);
    fp_add_8x1(s2, b, b+8);
    fp_mul_8x1_school(t1, a, b);
    fp_mul_8x1_school(t2, a+8, b+8);
    fp_mul_8x1_school(t3, s1, s2);
    fp_sub_8x1(t3, t3, t1);
    fp_sub_8x1(res+8, t3, t2);
    fp_sub_8x1(res, t1, t2);
}

void fp2_mul_8x1x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i s1[8], s2[8], tt1[16], tt2[16], tt3[16];
    mp_add_8x1(s1, a, a + 8);
    mp_add_8x1(s2, b, b + 8);
    mp_mul_8x1_karat(tt1, a, b);
    mp_mul_8x1_karat(tt2, a + 8, b + 8);
    mp_mul_8x1_karat(tt3, s1, s2);
    mp_dblsubx2_8x1(tt3, tt1, tt2);
    mp_subx2_8x1(tt1, tt2);
    rdc_8x1(res + 8, tt3);
    rdc_8x1(res, tt1);
}

void fp2_sqr_8x1x1(__m512i* res, __m512i* a)
{
    __m512i s1[8], s2[8], s3[8];
    mp_add_8x1(s1, a, a + 8);
    fp_sub_8x1(s2, a, a + 8);           ///////// THIS COULD BE OPTIMIZED FOR LIMB SIZE 48
    mp_add_8x1(s3, a, a);
    fp_mul_8x1(res, s1, s2);
    fp_mul_8x1(res + 8, s3, a + 8);
}

void fp2_muli_8x1x1(__m512i* res, __m512i* a)
{
    __m512i t[8];

    fp_neg_8x1(t, a+8);
    for (int i = 0; i < 8; i++) {
        res[i+8] = a[i];
    }
    for(int i = 0; i < 8; i++) {
        res[i] = t[i];
    }
}

void fp2_div3_8x1x1(__m512i* res, __m512i* src)
{
    fp_div3_8x1(res, src);
    fp_div3_8x1(res+8, src+8);
}
 
void fp2_tomont_8x1x1(__m512i* res, __m512i* a)
{
    fp_tomont_8x1(res, a);
    fp_tomont_8x1(res+8, a+8);
}

void fp2_frommont_8x1x1(__m512i*res, __m512i* a)
{
    fp_frommont_8x1(res, a);
    fp_frommont_8x1(res+8, a+8);
}

// ======================================================================================================
//                                                 4x1x2
// ======================================================================================================

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

void fp2_mul_4x1x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i s1[4], s2[4], t1[4], t2[4], t3[4];
    fp_add_4x2(s1, a, a+4);
    fp_add_4x2(s2, b, b+4);
    fp_mul_4x2(t1, a, b);
    fp_mul_4x2(t2, a+4, b+4);
    fp_mul_4x2(t3, s1, s2);
    fp_sub_4x2(t3, t3, t1);
    fp_sub_4x2(res+4, t3, t2);
    fp_sub_4x2(res, t1, t2);
}

void fp2_sqr_4x1x2(__m512i* res, __m512i* a)
{
    fp2_mul_4x1x2(res, a, a);
}
 
void fp2_tomont_4x1x2(__m512i* res, __m512i* a)
{
    fp_tomont_4x2(res, a);
    fp_tomont_4x2(res+4, a+4);
}

void fp2_frommont_4x1x2(__m512i*res, __m512i* a)
{
    fp_frommont_4x2(res, a);
    fp_frommont_4x2(res+4, a+4);
}

// ======================================================================================================
//                                                 4x2x1
// ======================================================================================================

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
    int lo = fp_is_equal_4x2(a, b);
    int hi = fp_is_equal_4x2(a+4, b+4);
    return lo & hi;
}

int fp2_is_zero_4x2x1(__m512i* a)
{
    int lo = fp_is_zero_4x2(a);
    int hi = fp_is_zero_4x2(a+4);
    return lo & hi;
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
    __m512i s1[8], s2[8], s3[8], t1[8], t2[8], t3[8];
    for(int i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0
        s2[i] = VSHUFFLE(a[i], 0xee); // a1, a1
        s3[i] = VSHUFFLE(b[i], 0x4e); // b0, b1
    }
    fp_mul_8x1(t1, s1, b);
    fp_mul_8x1(t2, s2, s3);
    fp_mixaddsub_8x1(res, t1, t2, 0xaa);
}

void fp2_sqr_4x2x1(__m512i* res, __m512i* a)
{
    __mmask16 k = 0b0011001100110011;
    __m512i s1[8], s2[8], s3[8], s4[8], t1[8], t2[8];
    for(int i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0
        s2[i] = VSHUFFLEMASKZ(k, a[i], 0x0e); // 0, a1
		s3[i] = VSHUFFLE(a[i], 0x4e); // a0, a1
    }
    fp_add_8x1(t1, s1, s3);
    fp_sub_8x1(t2, a, s2);
    fp_mul_8x1(res, t1, t2);
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

// res1 = a1 * b1, res2 = a2 * b2
void fp2_sqrx2_4x2x1(__m512i* res1, __m512i* res2, __m512i* a1, __m512i* a2)
{
    __m512i t[16];
    fp2_4x2x1_to_8x1x1(t, a1, a2);
    fp2_sqr_8x1x1(t, t);
    fp2_8x1x1_to_4x2x1(res1, res2, t);
}

void fp2_div3_4x2x1(__m512i* res, __m512i* src)
{
    fp_div3_8x1(res, src);
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
 
void fp2_tomont_4x2x1(__m512i* res, __m512i* a)
{
    fp_tomont_8x1(res, a);
}

void fp2_frommont_4x2x1(__m512i*res, __m512i* a)
{
    fp_frommont_8x1(res, a);
}

// ======================================================================================================
//                                                 2x2x2
// ======================================================================================================

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
    __mmask8 mask, acc, res;
    acc = 0xff;
    for(int i = 0; i < 4; i++){
        mask = VCMPEQ(a[i], b[i]);
        acc = acc & mask;
    }
    res = (acc & (acc >> 1) & (acc >> 2) & (acc >> 3) & 0x1) 
        | ((acc >> 3) & (acc >> 4) & (acc >> 5) & (acc >> 6) & 0x2);
    return res;
}

int fp2_is_zero_2x2x2(__m512i* a)
{
    __m512i zero = VZERO();
    __mmask8 mask, acc, res;
    acc = 0xff;
    for(int i = 0; i < 4; i++){
        mask = VCMPEQ(a[i], zero);
        acc = acc & mask;
    }
    res = (acc & (acc >> 1) & (acc >> 2) & (acc >> 3) & 0x1) 
        | ((acc >> 3) & (acc >> 4) & (acc >> 5) & (acc >> 6) & 0x2);
    return res;
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

void fp2_mul_2x2x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i s1[4], s2[4], s3[4], t1[4], t2[4];
    for(int i = 0; i < 4; i++) {
        s1[i] = VPERM256(a[i], 0x44); // a0, a0
        s2[i] = VPERM256(a[i], 0xee); // a1, a1
        s3[i] = VPERM256(b[i], 0x4e); // b1, b0
    }
    fp_mul_4x2(t1, s1, b);
    fp_mul_4x2(t2, s2, s3);
    fp_mixaddsub_4x2(res, t1, t2, 0xcc);
}

void fp2_sqr_2x2x2(__m512i* res, __m512i* a)
{
    __m512i t0[4], t1[4];
    for (int i = 0; i < 4; i++) {
        t0[i] = VPERM256(a[i], 0x44); // a0, a0
        t1[i] = VPERM256(a[i], 0x4e); // a1, a0
    }
    fp_add_4x2(t0, t0, t1); // a0 + a1, a0 + a0
    for (int i = 0; i < 4; i++) {
        t1[i] = VPERM256MASKZ(0x33, a[i], 0x4e); // a1, 0
    }
    fp_sub_4x2(t1, a, t1); // a0 - a1, a1
    fp_mul_4x2(res, t0, t1);
}
 
void fp2_tomont_2x2x2(__m512i* res, __m512i* a)
{
    fp_tomont_4x2(res, a);
}

void fp2_frommont_2x2x2(__m512i*res, __m512i* a)
{
    fp_frommont_4x2(res, a);
}

// ======================================================================================================
//                                                 2x4x1
// ======================================================================================================

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
    __mmask8 mask, acc, res;
    acc = 0xff;
    for(int i = 0; i < 4; i++){
        mask = VCMPEQ(a[i], b[i]);
        acc = acc & mask;
    }
    res = (acc & (acc >> 2) & 0x1) 
        | ((acc >> 3) & (acc >> 5) & 0x2);
    return res;
}

int fp2_is_zero_2x4x1(__m512i* a)
{
    __m512i zero = VZERO();
    __mmask8 mask, acc, res;
    acc = 0xff;
    for(int i = 0; i < 4; i++){
        mask = VCMPEQ(a[i], zero);
        acc = acc & mask;
    }
    res = (acc & (acc >> 2) & 0x1) 
        | ((acc >> 3) & (acc >> 5) & 0x2);
    return res;
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
    __m512i s1[8], s2[8], t1[8], t2[8];
    for(int i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44);   // a0, a0, a1, a1
        s2[i] = VPERM256(b[i], 0x28);   // b0, b1, b1, b0
    }
    fp_mul_8x1(t1, s1, s2);
    for(int i = 0; i < 8; i++) {
        t2[i] = VPERM256MASKZ(0x55, t1[i], 0x32);
        t1[i] = VPERM256MASKZ(0x55, t1[i], 0x10);
    }
    fp_mixaddsub_8x1(res, t1, t2, 0xcc);
}

void fp2_sqr_2x4x1(__m512i* res, __m512i* a)
{
    __m512i s1[8], s2[8], t1[8], t2[8];
    for(int i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44);   // a0, a0, a1, a1
        s2[i] = VPERM256(a[i], 0x28);   // a0, a1, a1, a0
    }
    fp_mul_8x1(t1, s1, s2);
    for(int i = 0; i < 8; i++) {
        t2[i] = VPERM256MASKZ(0x55, t1[i], 0x32);
        t1[i] = VPERM256MASKZ(0x55, t1[i], 0x10);
    }
    fp_mixaddsub_8x1(res, t1, t2, 0xcc);
}
 
void fp2_tomont_2x4x1(__m512i* res, __m512i* a)
{
    fp_tomont_8x1(res, a);
}

void fp2_frommont_2x4x1(__m512i*res, __m512i* a)
{
    fp_frommont_8x1(res, a);
}

// ======================================================================================================
//                                      Conversions between representations
// ======================================================================================================

void fp2_unvectorized_to_8x1x1(__m512i* res, fp2_t* src)
{
    fp_t coeffs_re[8], coeffs_im[8];
    for (int i = 0; i < 8; i++) {
        fp_encode(coeffs_re[i].arr, &(src+i)->re);
        fp_encode(coeffs_im[i].arr, &(src+i)->im);
    }

    fp_64_to_48_bit_limbs_8x1(res, coeffs_re);
    fp_64_to_48_bit_limbs_8x1(res+8, coeffs_im);
    fp2_tomont_8x1x1(res, res);
}

void fp2_8x1x1_to_unvectorized(fp2_t* res, __m512i* src)
{
    fp_t coeffs_re[8], coeffs_im[8];
    __m512i src_frommont[16];

    fp2_frommont_8x1x1(src_frommont, src);
    fp_48_to_64_bit_limbs_8x1(coeffs_re, src_frommont);
    fp_48_to_64_bit_limbs_8x1(coeffs_im, src_frommont+8);

    for (int i = 0; i < 8; i++) {
        fp_decode(&(res+i)->re, (const void *) coeffs_re[i].arr);
        fp_decode(&(res+i)->im, (const void *) coeffs_im[i].arr);
    }
}

void fp2_unvectorized_to_4x2x1(__m512i* res, fp2_t* src)
{
    fp_t coeffs[8];
    for (int i = 0; i < 4; i++) {
        fp_encode(coeffs[2*i].arr, &src[i].re);
        fp_encode(coeffs[2*i+1].arr, &src[i].im);
    }

    fp_64_to_48_bit_limbs_8x1(res, coeffs);
    fp2_tomont_4x2x1(res, res);
}

void fp2_unvectorized_to_2x4x1(__m512i* res, fp2_t* src)
{
    fp_t coeffs[4];
    for (int i = 0; i < 2; i++) {
        fp_encode(coeffs[2*i].arr, &src[i].re);
        fp_encode(coeffs[2*i+1].arr, &src[i].im);
    }

    fp_64_to_48_bit_limbs_8x1_only_even_idx(res, coeffs);
    fp2_tomont_2x4x1(res, res);
}

void fp2_4x2x1_to_unvectorized(fp2_t* res, __m512i* src)
{
    fp_t coeffs[8];
    __m512i src_frommont[8];

    fp2_frommont_4x2x1(src_frommont, src);
    fp_48_to_64_bit_limbs_8x1(coeffs, src_frommont);

    for (int i = 0; i < 4; i++) {
        fp_decode(&res[i].re, (const void *) coeffs[2*i].arr);
        fp_decode(&res[i].im, (const void *) coeffs[2*i+1].arr);
    }
}

void fp2_2x4x1_to_unvectorized(fp2_t* res, __m512i* src)
{
    fp_t coeffs[4];
    __m512i src_frommont[8];

    fp2_frommont_2x4x1(src_frommont, src);
    fp_48_to_64_bit_limbs_8x1_only_even_idx(coeffs, src_frommont);

    for (int i = 0; i < 2; i++) {
        fp_decode(&res[i].re, (const void *) coeffs[2*i].arr);
        fp_decode(&res[i].im, (const void *) coeffs[2*i+1].arr);
    }
}

void fp2_8x1x1_to_4x2x1(__m512i* res1, __m512i* res2, __m512i* src)
{
    __m512i re, im;
    for (int i = 0; i < 8; i++) {
        re = src[i];
        im = src[i+8];
        res1[i] = VUNPACKLO(re, im);
        res2[i] = VUNPACKHI(re, im);
    }
}

// Only extracts first 4x2x1
void fp2_8x1x1_to_first_4x2x1(__m512i* res, __m512i* src)
{
    for (int i = 0; i < 8; i++) {
        res[i] = VUNPACKLO(src[i], src[i+8]);
    }
}

void fp2_4x2x1_to_8x1x1(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i s1, s2;
    __m512i idx_re = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
    __m512i idx_im = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);

    for (int i = 0; i < 8; i++) {
        s1 = src1[i];
        s2 = src2[i];
        res[i] = VBLENDPERM(idx_re, s1, s2);
        res[i+8] = VBLENDPERM(idx_im, s1, s2);
    }
}

void fp2_4x2x1_duplicate_to_8x1x1(__m512i* res, __m512i* src)
{
    __m512i s;
    __m512i idx_re = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);
    __m512i idx_im = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);

    for (int i = 0; i < 8; i++) {
        s = src[i];
        res[i] = VBLENDPERM(idx_re, s, s);
        res[i+8] = VBLENDPERM(idx_im, s, s);
    }
}

// Assumes that the input and output arrays do not overlap.
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


// Alternate implementation, mapping (a,b,c,d) in 4x2x1 to (a,c) and (b,d) in 2x4x1
// void fp2_4x2x1_to_2x4x1(__m512i* res, __m512i* src)
// {
//     for (int i = 0; i < 8; i++) {
//         res[i] = VPERM256MASKZ(0x55, src[i], 0x50); // a, b, c, d -> a, 0, b, 0
//         res[i+8] = VPERM256MASKZ(0x55, src[i], 0xfa); // a, b, c, d -> c, 0, d, 0
//     }
// }

// void fp2_2x4x1_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2)
// {
//     __m512i indices = VSET(
//             0x0000000100000000, 0x0000000500000004, 0x0000001100000010, 0x0000001500000014,
//             0x0000000900000008, 0x0000000d0000000c, 0x0000001900000018, 0x0000001d0000001c
//         );
//     for (int i = 0; i < 8; i++) {
//         res[i] = VBLENDPERM32(0xffff, src1[i], src2[i], indices);
//     }
// }