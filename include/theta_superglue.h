#ifndef THETA_SUPERGLUE_H
#define THETA_SUPERGLUE_H

#include "intrinsics_avx512.h"
#include "sqi_ec.h"
#include "sqi_hd.h"

/**
 * Algorithm 8.39 of the SQIsign spec.
 * Everything except matrix is in 4x2x1.
 */
bool superglue_codomain(__m512i* res_OA, __m512i* A24_12, int *position,
		bool *sign_flip, fp2_t dual_theta_null_square[4], fp2_t elements[6],
		__m512i* x_8, __m512i* z_8);

void superglue_eval_pair(__m512i *imP_Q, __m512i *A_12, __m512i *xP,
		__m512i *yP, __m512i *zP, __m512i *T8x, __m512i *T8y, __m512i *T8z,
		__m512i *ab, int position, bool sign_flip,
		fp2_t dual_theta_null_square[4], fp2_t elements[6]);
#endif