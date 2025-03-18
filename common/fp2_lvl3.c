#include "fp_lvl3.h"
#include "fp2_lvl3.h"

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