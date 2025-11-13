#ifndef THETA_SPLITTING_H
#define THETA_SPLITTING_H

#include "intrinsics_avx512.h"

#define NUM_EVEN_PAIRS 10
static const int EVEN_I[NUM_EVEN_PAIRS] = { 0, 0, 0, 0, 1, 1, 2, 2, 3, 3 };
static const int EVEN_J[NUM_EVEN_PAIRS] = { 0, 1, 2, 3, 0, 2, 0, 1, 0, 3 };

/**
 * Encodes a 4x4 matrix A  with entries in {0, 1, -1, i, -i}
 * The i-th mask of each array encodes the i-th column of A.
 * Bit 2*j and 2*k+1 encodes A_(i,j).
 * is_zero: If A_(i,j) = 0, both bits are 1
 * is_negative: If A_(i,j) is negative, both bits are 1
 * is_imaginary: If A_(i,j) is imaginary, both bits are 1
 * 
 * The reason why the bits 2*j and 2*j+1 encode the same value is that it is a convenient format
 * for multiplying one or two points by the matrix.
 * One point: In the 4x2x1 format the limbs 2*j and 2*j+1 are both for the j-th Fp2 element.
 * Two points: They are stored in the 8x1x1 format as (x, x', y, y', ...),
 *             so the same operation should be applied to limb 2*j and 2*j+1 for j = 0, ..., 3.
 */
typedef struct splitting_matrix_t
{
    __mmask8 is_zero[4];
    __mmask8 is_negative[4];
    __mmask8 is_imaginary[4];
} splitting_matrix_t;

static const splitting_matrix_t SPLITTING_MATRICES[10] = {
    // case (0,0): {1,5,1,5, 1,7,3,5, 1,5,3,7, 3,5,3,5}
    {
        { 0x00, 0x00, 0x00, 0x00 }, // zero
        { 0xc0, 0x0c, 0xfc, 0x30 }, // negative
        { 0x00, 0xff, 0x00, 0xff }  // imaginary
    },

    // case (0,1): {1,0,0,0, 0,0,0,1, 0,0,1,0, 0,3,0,0}
    {
        { 0xfc, 0x3f, 0xcf, 0xf3 },
        { 0x00, 0xc0, 0x00, 0x00 },
        { 0x00, 0x00, 0x00, 0x00 }
    },

    // case (0,2): {1,0,0,0, 0,1,0,0, 0,0,0,1, 0,0,3,0}
    {
        { 0xfc, 0xf3, 0x3f, 0xcf },
        { 0x00, 0x00, 0xc0, 0x00 },
        { 0x00, 0x00, 0x00, 0x00 }
    },

    // case (0,3): {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,3}
    {
        { 0xfc, 0xf3, 0xcf, 0x3f },
        { 0x00, 0x00, 0x00, 0xc0 },
        { 0x00, 0x00, 0x00, 0x00 }
    },

    // case (1,0): {1,1,1,1, 1,3,3,1, 1,1,3,3, 3,1,3,1}
    {
        { 0x00, 0x00, 0x00, 0x00 },
        { 0xc0, 0x0c, 0xfc, 0x30 },
        { 0x00, 0x00, 0x00, 0x00 }
    },

    // case (1,2): {1,0,0,0, 0,1,0,0, 0,0,0,1, 0,0,1,0}
    {
        { 0xfc, 0xf3, 0x3f, 0xcf },
        { 0x00, 0x00, 0x00, 0x00 },
        { 0x00, 0x00, 0x00, 0x00 }
    },

    // case (2,0): {1,1,1,1, 1,3,1,3, 1,3,3,1, 3,3,1,1}
    {
        { 0x00, 0x00, 0x00, 0x00 },
        { 0xc0, 0xfc, 0x30, 0x0c },
        { 0x00, 0x00, 0x00, 0x00 }
    },

    // case (2,1): {1,1,1,1, 1,3,1,3, 1,3,3,1, 1,1,3,3}
    {
        { 0x00, 0x00, 0x00, 0x00 },
        { 0x00, 0x3c, 0xf0, 0xcc },
        { 0x00, 0x00, 0x00, 0x00 }
    },

    // case (3,0): {1,1,1,1, 1,3,1,3, 1,1,3,3, 3,1,1,3}
    {
        { 0x00, 0x00, 0x00, 0x00 },
        { 0xc0, 0x0c, 0x30, 0xfc },
        { 0x00, 0x00, 0x00, 0x00 }
    },

    // case (3,3): {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1}
    {
        { 0xfc, 0xf3, 0xcf, 0x3f },
        { 0x00, 0x00, 0x00, 0x00 },
        { 0x00, 0x00, 0x00, 0x00 }
    }
};

static const splitting_matrix_t RAND_MATRICES[6] = {
    {
        { 0xfc, 0xf3, 0xcf, 0x3f },
        { 0x00, 0x00, 0x00, 0x00 },
        { 0x00, 0x00, 0x00, 0x00 }
    },
    {
        { 0x3f, 0xcf, 0xf3, 0xfc },
        { 0x00, 0x00, 0x00, 0x00 },
        { 0x00, 0x00, 0x00, 0x00 }
    },
    {
        { 0x00, 0x00, 0x00, 0x00 },
        { 0x00, 0xcc, 0xf0, 0x3c },
        { 0x00, 0x00, 0x00, 0x00 }
    },
    {
        { 0x00, 0x00, 0x00, 0x00 },
        { 0x3c, 0x0f, 0x33, 0x00 },
        { 0x00, 0x00, 0x00, 0x00 }
    },
    {
        { 0x00, 0x00, 0x00, 0x00 },
        { 0x03, 0x0c, 0x30, 0xc0 },
        { 0x3c, 0xc3, 0xc3, 0x3c }
    },
    {
        { 0x00, 0x00, 0x00, 0x00 },
        { 0xc0, 0x30, 0x0c, 0x03 },
        { 0x3c, 0xc3, 0xc3, 0x3c }
    }
};

/**
 * Algorithm 8.41 of the SQIsign spec v2.0.1, modified to output the index in the list of even pairs (i,j)
 * In case zero or multiple pairs are found, it outputs -1.
 * @param null The theta null point
 */
int splitting_get_index(__m512i* null);
int splitting_get_index_old(__m512i* null);

/**
 * Algorithm 8.42 of the SQIsign spec v2.0.1
 * Samples an isomorphism matrix.
 *
 * @param matrix The output is a 4x4 matrix over Fp2.
 * @param null The theta null point
 */
void splitting_isom_matrix(splitting_matrix_t* matrix, __m512i* null);

void splitting_rand_matrix(splitting_matrix_t* matrix);

void splitting_apply_isomorphism_single(__m512i* res, __m512i* point, const splitting_matrix_t* matrix);

void splitting_apply_isomorphism_pair(__m512i* res, __m512i* point, const splitting_matrix_t* matrix);

void splitting_theta_to_product(__m512i* product, __m512i* null);
void splitting_theta_to_product_old(__m512i* product, __m512i* null);

/**
 * Given a theta null point OA in a compatible form with splitting, compute
 * P1, P2 in E1 x E2 on the corresponding product of Montgomery curves.
 *
 * @param mont_points The output is (X_1, X_2, Z_1, Z_2) in the 4x2x1 format, where Pi = (Xi, Zi)
 * @param theta_point The theta point in the 4x2x1 format
 * @param null The theta null point
 */
void splitting_theta_point_to_mont_single(__m512i* mont_points, __m512i* theta_point, __m512i* null);

/**
 * Given a theta null point OA in a compatible form with splitting, compute
 * P1, P2 in E1 x E2 on the corresponding product of Montgomery curves.
 *
 * TODO: can we use parallelism here?
 * @param mont_points The output is the Montgomery points interleaved as
                      (X_1, X'_1, X_2, X'_2, Z_1, Z'_1, Z_2, Z'_2), where 
                      (X_1, Z_1) and (X_2, Z_2) correspond to the theta point (X, Y, Z, T).
 * @param theta_point The theta points (X, Y, Z, T) and (X', Y', Z', T') interleaved 
                      in the 8x1x1 format.
 * @param null The theta null point
 */
void splitting_theta_point_to_mont_pair(__m512i* mont_points, __m512i* theta_point, __m512i* null);
void splitting_theta_point_to_mont_pair_old(__m512i* mont_points, __m512i* theta_point, __m512i* null);


// Internal functions
void splitting_duplicate_pair_in_array(__m512i* res, __m512i* point, const int index, const int arraylen);
void splitting_mul_matrix_column_single(__m512i* res, __m512i* dup, const splitting_matrix_t* matrix, const int index);
void splitting_mul_matrix_column_pair(__m512i* res, __m512i* dup, const splitting_matrix_t* matrix, const int index);


#endif