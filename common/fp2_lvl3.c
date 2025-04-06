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

void fp2_mul_8x1x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i s1[8], s2[8], t1[8], t2[8], t3[8];
    fp_add_8x1(s1, a, a+8);
    fp_add_8x1(s2, b, b+8);
    fp_mul_8x1(t1, a, b);
    fp_mul_8x1(t2, a+8, b+8);
    fp_mul_8x1(t3, s1, s2);
    fp_sub_8x1(t3, t3, t1);
    fp_sub_8x1(res+8, t3, t2);
    fp_sub_8x1(res, t1, t2);
}

void fp2_sqr_8x1x1(__m512i* res, __m512i* a)
{
    fp2_mul_8x1x1(res, a, a);
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

void fp2_mul_4x2x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i s1[8], s2[8], s3[8], t1[8], t2[8];
    for(int i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0
        s2[i] = VSHUFFLE(a[i], 0xee); // a1, a1
        s3[i] = VSHUFFLE(b[i], 0x4e); // b1, b0
    }
    fp_mul_8x1(t1, s1, b);
    fp_mul_8x1(t2, s2, s3);
    fp_mixsubadd_8x1(res, t1, t2);
}

void fp2_sqr_4x2x1(__m512i* res, __m512i* a)
{
    fp2_mul_4x2x1(res, a, a);
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
        s1[i] = VPERM(a[i], 0x44); // a0, a0
        s2[i] = VPERM(a[i], 0xee); // a1, a1
        s3[i] = VPERM(b[i], 0x4e); // b1, b0
    }
    fp_mul_4x2(t1, s1, b);
    fp_mul_4x2(t2, s2, s3);
    fp_mixsubadd_4x2(res, t1, t2);
}

void fp2_sqr_2x2x2(__m512i* res, __m512i* a)
{
    fp2_mul_2x2x2(res, a, a);
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

void fp2_mul_2x4x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i s1[8], s2[8], t1[8], t2[8];
    for(int i = 0; i < 8; i++) {
        s1[i] = VSHUFFLE(a[i], 0x44);   // a0, a0, a1, a1
        s2[i] = VPERM(b[i], 0x28);      // b0, b1, b1, b0
    }
    fp_mul_8x1(t1, s1, s2);
    for(int i = 0; i < 8; i++) {
        t2[i] = VPERMMASKZ(0x55, t1[i], 0x32);
        t1[i] = VPERMMASKZ(0x55, t1[i], 0x10);
    }
    fp_mixsubsubaddadd_8x1(res, t1, t2);
}

void fp2_sqr_2x4x1(__m512i* res, __m512i* a)
{
    fp2_mul_2x4x1(res, a, a);
}
 
void fp2_tomont_2x4x1(__m512i* res, __m512i* a)
{
    fp_tomont_8x1(res, a);
}

void fp2_frommont_2x4x1(__m512i*res, __m512i* a)
{
    fp_frommont_8x1(res, a);
}