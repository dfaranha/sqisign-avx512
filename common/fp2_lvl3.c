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
    fp_mixaddsub_8x1(res, t1, t2, 0xaa);
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
        s1[i] = VPERM256(a[i], 0x44); // a0, a0
        s2[i] = VPERM256(a[i], 0xee); // a1, a1
        s3[i] = VPERM256(b[i], 0x4e); // b1, b0
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
        s2[i] = VPERM256(b[i], 0x28);      // b0, b1, b1, b0
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

// ======================================================================================================
//                                      Conversions between representations
// ======================================================================================================

void fp2_array_to_8x1x1(__m512i* res, uint64_t* src)
{
    uint64_t t[8];
    //Splitting into 51-bit limbs. Outer loop for each Fp coefficient
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 8; i++) {
            t[i] = src[12*i+6*j] & LIMB_MASK;
        }
        res[8*j] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
        for (int i = 0; i < 8; i++) {
            t[i] = ((src[12*i+6*j] >> 51) & 0x1ffff) | ((src[12*i+6*j+1] & 0x3fffffffff) << 13); 
        }
        res[8*j+1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
        for (int i = 0; i < 8; i++) {
            t[i] = ((src[12*i+6*j+1] >> 38) & 0x3ffffff) | ((src[12*i+6*j+2] & 0x1ffffff) << 26); 
        }
        res[8*j+2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
        for (int i = 0; i < 8; i++) {
            t[i] = ((src[12*i+6*j+2] >> 25) & 0x7fffffffff) | ((src[12*i+6*j+3] & 0xfff) << 39); 
        }
        res[8*j+3] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
        for (int i = 0; i < 8; i++) {
            t[i] = (src[12*i+6*j+3] >> 12) & LIMB_MASK;
        }
        res[8*j+4] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
        for (int i = 0; i < 8; i++) {
            t[i] = ((src[12*i+6*j+3] >> 63) & 0x1) | ((src[12*i+6*j+4] & 0x3ffffffffffff) << 1); 
        }
        res[8*j+5] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
        for (int i = 0; i < 8; i++) {
            t[i] = ((src[12*i+6*j+4] >> 50) & 0x3fff) | ((src[12*i+6*j+5] & 0x1fffffffff) << 14); 
        }
        res[8*j+6] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
        for (int i = 0; i < 8; i++) {
            t[i] = (src[12*i+6*j+5] >> 37) & 0x7ffffff;
        }
        res[8*j+7] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    }
}

void fp2_array_to_4x2x1(__m512i* res, uint64_t* src)
{
    uint64_t t[8];
    for (int i = 0; i < 8; i++) {
        t[i] = src[6*i] & LIMB_MASK;
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((src[6*i] >> 51) & 0x1ffff) | ((src[6*i+1] & 0x3fffffffff) << 13); 
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((src[6*i+1] >> 38) & 0x3ffffff) | ((src[6*i+2] & 0x1ffffff) << 26); 
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((src[6*i+2] >> 25) & 0x7fffffffff) | ((src[6*i+3] & 0xfff) << 39); 
    }
    res[3] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (src[6*i+3] >> 12) & LIMB_MASK;
    }
    res[4] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((src[6*i+3] >> 63) & 0x1) | ((src[6*i+4] & 0x3ffffffffffff) << 1); 
    }
    res[5] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((src[6*i+4] >> 50) & 0x3fff) | ((src[6*i+5] & 0x1fffffffff) << 14); 
    }
    res[6] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (src[6*i+5] >> 37) & 0x7ffffff;
    }
    res[7] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

void fp2_8x1x1_to_array(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    //Merging 51-bit limbs to 64-bit limbs. Outer loop for each Fp coefficient
    for (int j = 0; j < 2; j++) {
        VSTORE(t, src[8*j]);
        for (int i = 0; i < 8; i++) {
            res[12*i+6*j] = t[i];
        } 
        VSTORE(t, src[8*j+1]);
        for (int i = 0; i < 8; i++) {
            res[12*i+6*j] |= (t[i] & 0x1fff) << 51;
            res[12*i+6*j+1] = (t[i] >> 13) & 0x3fffffffff;
        } 
        VSTORE(t, src[8*j+2]);
        for (int i = 0; i < 8; i++) {
            res[12*i+6*j+1] |= (t[i] & 0x3ffffff) << 38;
            res[12*i+6*j+2] = (t[i] >> 26) & 0x1ffffff;
        }
        VSTORE(t, src[8*j+3]);
        for (int i = 0; i < 8; i++) {
            res[12*i+6*j+2] |= (t[i] & 0x7fffffffff) << 25;
            res[12*i+6*j+3] = (t[i] >> 39) & 0xfff;
        }
        VSTORE(t, src[8*j+4]);
        for (int i = 0; i < 8; i++) {
            res[12*i+6*j+3] |= t[i] << 12;
        }
        VSTORE(t, src[8*j+5]);
        for (int i = 0; i < 8; i++) {
            res[12*i+6*j+3] |= (t[i] & 0x1) << 63;
            res[12*i+6*j+4] = (t[i] >> 1) & 0x3ffffffffffff;
        }
        VSTORE(t, src[8*j+6]);
        for (int i = 0; i < 8; i++) {
            res[12*i+6*j+4] |= (t[i] & 0x3fff) << 50;
            res[12*i+6*j+5] = (t[i] >> 14) & 0x1fffffffff;
        }
        VSTORE(t, src[8*j+7]);
        for (int i = 0; i < 8; i++) {
            res[12*i+6*j+5] |= t[i] << 37;
        }
    }

}
void fp2_4x2x1_to_array(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    VSTORE(t, src[0]);
    for (int i = 0; i < 8; i++) {
        res[6*i] = t[i];
    } 
    VSTORE(t, src[1]);
    for (int i = 0; i < 8; i++) {
        res[6*i] |= (t[i] & 0x1fff) << 51;
        res[6*i+1] = (t[i] >> 13) & 0x3fffffffff;
    } 
    VSTORE(t, src[2]);
    for (int i = 0; i < 8; i++) {
        res[6*i+1] |= (t[i] & 0x3ffffff) << 38;
        res[6*i+2] = (t[i] >> 26) & 0x1ffffff;
    }
    VSTORE(t, src[3]);
    for (int i = 0; i < 8; i++) {
        res[6*i+2] |= (t[i] & 0x7fffffffff) << 25;
        res[6*i+3] = (t[i] >> 39) & 0xfff;
    }
    VSTORE(t, src[4]);
    for (int i = 0; i < 8; i++) {
        res[6*i+3] |= t[i] << 12;
    }
    VSTORE(t, src[5]);
    for (int i = 0; i < 8; i++) {
        res[6*i+3] |= (t[i] & 0x1) << 63;
        res[6*i+4] = (t[i] >> 1) & 0x3ffffffffffff;
    }
    VSTORE(t, src[6]);
    for (int i = 0; i < 8; i++) {
        res[6*i+4] |= (t[i] & 0x3fff) << 50;
        res[6*i+5] = (t[i] >> 14) & 0x1fffffffff;
    }
    VSTORE(t, src[7]);
    for (int i = 0; i < 8; i++) {
        res[6*i+5] |= t[i] << 37;
    }
}

void fp2_8x1x1_to_4x2x1(__m512i* res, __m512i* src)
{
    __m512i t_re, t_im;

    for (int i = 0; i < 8; i++) {
        t_re = VSHUFFLE(src[i], 0x4e); // a, b, c, d, ... -> b, a, d, c, ...
        t_im = VSHUFFLE(src[i+8], 0x4e);
        res[i] = VBLEND(0xaa, src[i], t_im); 
        res[i+8] = VBLEND(0xaa, t_re, src[i+8]);
    }
}

void fp2_4x2x1_to_8x1x1(__m512i* res, __m512i* src)
{
    __m512i t1, t2;

    for (int i = 0; i < 8; i++) {
        t1 = VSHUFFLE(src[i], 0x4e); // a, b, c, d, ... -> b, a, d, c, ...
        t2 = VSHUFFLE(src[i+8], 0x4e);
        res[i] = VBLEND(0xaa, src[i], t2); 
        res[i+8] = VBLEND(0xaa, t1, src[i+8]);
    }
}