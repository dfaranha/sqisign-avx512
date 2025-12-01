#ifndef THETA_GLUING_H
#define THETA_GLUING_H

#include "intrinsics_avx512.h"

/**
 * Helper function:
 *
 * Subroutine to compute the matrix in action_by_translation
 * Uses an alternative projective representation to avoid all inversions.

 * With inverses precomputed we need five calls to "pairwise_mul" so there
 * is an extra cost of a single avx_sqr call
 * 
 * Called _compute_matrix in the python script
 * TODO: Explain what inputs and outpus are.
 */
void gluing_compute_matrix_4x2x1(
    __m512i* g1, __m512i* g2, __m512i* h1, __m512i* h2, __m512i* pi, //outputs, name as in change of basis
    __m512i* x, __m512i* z, __m512i* u, __m512i* w, __m512i* det //inputs
);


#endif