#include <stdio.h>
#include <stdint.h>
#include "fp.h"
#include "fp_consts.h"
#include "fp_unvec.h"
#include "fp2.h"
#include "utils.h"

void fp_8x1_to_unvectorized(fp_t* res, __m512i* src)
{
    __m512i s[NUM_LIMBS];
    uint64_t res_arr[8*NWORDS64];

    fp_frommont_8x1(s, src);
    fp_limbs_to_64_8x1(res_arr, s);

    // Convert to Montgomery representation
    for (int i = 0; i < 8; i++) {
        fp_decode(res+i, (const void *) (res_arr + i*NWORDS64));
    }
}

void fp_unvectorized_to_8x1(__m512i* res, fp_t* src)
{
    uint64_t src_arr[8*NWORDS64] = { 0 };

    // Convert from Montgomery representation
    for (int i = 0; i < 8; i++) {
        fp_encode(src_arr + i*NWORDS64, src+i);
    }

    fp_64_to_limbs_8x1(res, src_arr);
    fp_tomont_8x1(res, res);
}

void fp_4x2_to_unvectorized(fp_t* res, __m512i* src)
{
    __m512i s[N4x2];
    uint64_t res_arr[4*NWORDS64];

    fp_frommont_4x2(s, src);
    fp_limbs_to_64_4x2(res_arr, s);

    // Convert to Montgomery representation
    for (int i = 0; i < 4; i++) {
        fp_decode(res+i, (const void *) (res_arr + i*NWORDS64));
    }
}

void fp_unvectorized_to_4x2(__m512i* res, fp_t* src)
{
    fp_t coeffs[4];
    uint64_t src_arr[4*NWORDS64] = { 0 };

    // Convert from Montgomery representation
    for (int i = 0; i < 4; i++) {
        fp_encode(src_arr + i*NWORDS64, src+i);
    }

    fp_64_to_limbs_4x2(res, src_arr);
    fp_tomont_4x2(res, res);
}

void fp_2x4_to_unvectorized(fp_t* res, __m512i* src)
{
    __m512i s[N2x4];
    uint64_t res_arr[2*NWORDS64];

    fp_frommont_2x4(s, src);
    fp_limbs_to_64_2x4(res_arr, s);

    // Convert to Montgomery representation
    for (int i = 0; i < 2; i++) {
        fp_decode(res+i, (const void *) (res_arr + i*NWORDS64));
    }
}

void fp_unvectorized_to_2x4(__m512i* res, fp_t* src)
{
    fp_t coeffs[2];
    uint64_t src_arr[2*NWORDS64] = { 0 };

    // Convert from Montgomery representation
    for (int i = 0; i < 2; i++) {
        fp_encode(src_arr + i*NWORDS64, src+i);
    }

    fp_64_to_limbs_2x4(res, src_arr);
    fp_tomont_2x4(res, res);
}

void fp2_unvectorized_to_8x1x1(__m512i* res, fp2_t* src)
{
    uint64_t re[8*NWORDS64] = { 0 }, im[8*NWORDS64] = { 0 };
    for (int i = 0; i < 8; i++) {
        fp_encode(re + i*NWORDS64, &(src+i)->re);
        fp_encode(im + i*NWORDS64, &(src+i)->im);
    }

    fp_64_to_limbs_8x1(res, re);
    fp_64_to_limbs_8x1(res + NUM_LIMBS, im);
    fp2_tomont_8x1x1(res, res);
}

void fp2_8x1x1_to_unvectorized(fp2_t* res, __m512i* src)
{
    __m512i s[2*NUM_LIMBS];
    uint64_t re[8*NWORDS64], im[8*NWORDS64];

    fp2_frommont_8x1x1(s, src);

    fp_limbs_to_64_8x1(re, s);
    fp_limbs_to_64_8x1(im, s + NUM_LIMBS);

    for (int i = 0; i < 8; i++) {
        fp_decode(&(res+i)->re, (const void *) (re + i*NWORDS64));
        fp_decode(&(res+i)->im, (const void *) (im + i*NWORDS64));
    }
}

void fp2_unvectorized_to_4x2x1(__m512i* res, fp2_t* src)
{
    uint64_t coeffs[8*NWORDS64] = { 0 };
    for (int i = 0; i < 4; i++) {
        fp_encode(coeffs + (2*i)*NWORDS64, &(src+i)->re);
        fp_encode(coeffs + (2*i+1)*NWORDS64, &(src+i)->im);
    }

    fp_64_to_limbs_8x1(res, coeffs);
    fp2_tomont_4x2x1(res, res);
}

void fp2_4x2x1_to_unvectorized(fp2_t* res, __m512i* src)
{
    __m512i s[NUM_LIMBS];
    uint64_t coeffs[8*NWORDS64];

    fp_frommont_8x1(s, src);
    fp_limbs_to_64_8x1(coeffs, s);

    for (int i = 0; i < 4; i++) {
        fp_decode(&res[i].re, (const void *) (coeffs + (2*i)*NWORDS64));
        fp_decode(&res[i].im, (const void *) (coeffs + (2*i+1)*NWORDS64));
    }
}

void fp2_unvectorized_to_2x2x2(__m512i* res, fp2_t* src)
{
    uint64_t coeffs[4*NWORDS64] = { 0 };
    for (int i = 0; i < 2; i++) {
        fp_encode(coeffs + (2*i)*NWORDS64, &(src+i)->re);
        fp_encode(coeffs + (2*i+1)*NWORDS64, &(src+i)->im);
    }

    fp_64_to_limbs_4x2(res, coeffs);
    fp_tomont_4x2(res, res);
}

void fp2_2x2x2_to_unvectorized(fp2_t* res, __m512i* src)
{
    __m512i s[N4x2];
    uint64_t coeffs[4*NWORDS64];

    fp_frommont_4x2(s, src);
    fp_limbs_to_64_4x2(coeffs, s);

    for (int i = 0; i < 2; i++) {
        fp_decode(&res[i].re, (const void *) (coeffs + (2*i)*NWORDS64));
        fp_decode(&res[i].im, (const void *) (coeffs + (2*i+1)*NWORDS64));
    }
}

void fp2_unvectorized_to_2x4x1(__m512i* res, fp2_t* src)
{
    uint64_t coeffs[4*NWORDS64] = { 0 };
    for (int i = 0; i < 2; i++) {
        fp_encode(coeffs + (2*i)*NWORDS64, &(src+i)->re);
        fp_encode(coeffs + (2*i+1)*NWORDS64, &(src+i)->im);
    }

    fp_64_to_limbs_8x1_only_even_idx(res, coeffs);
    fp2_tomont_2x4x1(res, res);
}

void fp2_2x4x1_to_unvectorized(fp2_t* res, __m512i* src)
{
    __m512i s[NUM_LIMBS];
    uint64_t coeffs[4*NWORDS64];

    fp2_frommont_2x4x1(s, src);
    fp_limbs_to_64_8x1_only_even_idx(coeffs, s);

    for (int i = 0; i < 2; i++) {
        fp_decode(&res[i].re, (const void *) (coeffs + (2*i)*NWORDS64));
        fp_decode(&res[i].im, (const void *) (coeffs + (2*i+1)*NWORDS64));
    }
}