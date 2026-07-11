#ifndef THETA_SUPERGLUE_H
#define THETA_SUPERGLUE_H

#include "intrinsics_avx512.h"
#include "sqi_ec.h"
#include "sqi_hd.h"

typedef struct {
    __m512i C0[2 * NUM_LIMBS]; // 8x1x1 multiplier of the first 8M block
    __m512i C1[NUM_LIMBS];     // 4x2x1 broadcast: B_inv (pos 4) / e5 (pos 1|7)
    __m512i J[2 * NUM_LIMBS];  // 8x1x1 (b, b, a, a, b, b, a, a) from ab
    int position;
    bool sign_flip;
} superglue_precomp_t;

/**
 * Algorithm 8.39 of the SQIsign spec.
 * Everything except matrix is in 4x2x1.
 */
bool superglue_codomain(__m512i* res_OA, __m512i* A24_12,
        superglue_precomp_t *K, __m512i* x_8, __m512i* z_8);

void superglue_eval_pair(__m512i *imP_Q, __m512i *A_12, __m512i *xP,
        __m512i *yP, __m512i *zP, __m512i *T8x, __m512i *T8y, __m512i *T8z,
        const superglue_precomp_t *K);
#endif