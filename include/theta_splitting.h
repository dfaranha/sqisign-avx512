#ifndef THETA_SPLITTING_H
#define THETA_SPLITTING_H

#include "intrinsics_avx512.h"

#define NUM_EVEN_PAIRS 10
static const int EVEN_I[NUM_EVEN_PAIRS] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3 };
static const int EVEN_J[NUM_EVEN_PAIRS] = { 0, 1, 2, 3, 0, 2, 0, 1, 0, 3 };

/**
 * Algorithm 8.41 of the SQIsign spec v2.0.1, modified to output the index in the list of even pairs (i,j)
 * In case zero or multiple pairs are found, it outputs -1.
 * @param null The theta null point
 */
int splitting_get_index(__m512i* null);

/**
 * Algorithm 8.42 of the SQIsign spec v2.0.1
 * Samples a random isomorphism matrix.
 * The matrix is represented by two AVX_8x1x1 vectors, one for the first two rows, and the other as the last two rows
 * Entries:
 * 0: 0
 * 1: 1
 * 3: -1
 * 5: i
 * 7: -i
 * First bit encodes whether it is 0 or not, second if it is negative, third if it is imaginary.
 *
 * TODO: At the moment, this is not randomized
 * @param matrix The output is a 4x4 matrix over Fp2 stored as two 8x1x1.
 * @param null The theta null point
 */
void splitting_isom_matrix(int* matrix, __m512i* null);

void splitting_apply_isomorphism(__m512i* res, int* matrix, __m512i* null);

void splitting_theta_to_product(__m512i* product, __m512i* null);

/**
 * Given a theta null point OA in a compatible form with splitting, compute
 * P1, P2 in E1 x E2 on the corresponding product of Montgomery curves.
 *
 * TODO: can we use parallelism here?
 * @param mont_points The output is (X_1, X_2, Z_1, Z_2) in the 4x2x1 format, where Pi = (Xi, Zi)
 * @param theta_point The theta point in the 4x2x1 format
 * @param null The theta null point
 */
void splitting_theta_point_to_mont(__m512i* mont_points, __m512i* theta_point, __m512i* null);

#endif