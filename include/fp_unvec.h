#ifndef _FP_UNVEC_H
#define _FP_UNVEC_H

#include <stdint.h>
#include "intrinsics_avx512.h"
#include "sqi_fp_lvl3.h"
#include "sqi_fp2.h"

void fp_8x1_to_unvectorized(fp_t* res, __m512i* src);
void fp_unvectorized_to_8x1(__m512i* res, fp_t* src);

void fp_4x2_to_unvectorized(fp_t* res, __m512i* src);
void fp_unvectorized_to_4x2(__m512i* res, fp_t* src);

void fp_2x4_to_unvectorized(fp_t* res, __m512i* src);
void fp_unvectorized_to_2x4(__m512i* res, fp_t* src);

void fp2_unvectorized_to_8x1x1(__m512i* res, fp2_t* src);
void fp2_8x1x1_to_unvectorized(fp2_t* res, __m512i* src);
void fp2_unvectorized_to_4x2x1(__m512i* res, fp2_t* src);
void fp2_4x2x1_to_unvectorized(fp2_t* res, __m512i* src);
void fp2_unvectorized_to_2x2x2(__m512i* res, fp2_t* src);
void fp2_2x2x2_to_unvectorized(fp2_t* res, __m512i* src);
void fp2_unvectorized_to_2x4x1(__m512i* res, fp2_t* src);
void fp2_2x4x1_to_unvectorized(fp2_t* res, __m512i* src);

#endif