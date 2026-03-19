#ifndef THETA_GLUING_H
#define THETA_GLUING_H

#include "intrinsics_avx512.h"
#include "sqi_ec.h"
#include "sqi_hd.h"

/**
 * Helper function:
 *
 * Subroutine to compute the matrix in action_by_translation
 * Uses an alternative projective representation to avoid all inversions.

 * With inverses precomputed we need five calls to "pairwise_mul" so there
 * is an extra cost of a single avx_sqr call
 * 
 * Called _compute_matrix in the python script
 * TODO: Explain what inputs and outputs are.
 */
void gluing_compute_matrix_4x2x1(
    __m512i* g1, __m512i* g2, __m512i* h1, __m512i* h2, __m512i* pi, //outputs, name as in change of basis
    __m512i* x_4, __m512i* z_4, __m512i* u, __m512i* w, __m512i* det //inputs
);

/**
 * Algorithm 8.36 of the SQISign spec.
 * Compute the translation matrix induced by the generator of our order 4
 * isotropic basis <(P1,P2)x(Q1,Q2)> in E_1 x E_2.
 */
void gluing_action_by_translation_4x2x1(
    __m512i* g1, __m512i* g2, __m512i* h1, __m512i* h2, __m512i* pi,
    __m512i* A24, __m512i* x_4, __m512i* z_4
);

/**
 * Algorithm 8.37 of the SQIsign spec.
 * Compute the change of matrix that transform the standard product
 * theta structure into the theta structure induced by the 4-symplectic
 * basis <(0,Q2_4),(P1_4,0)> + <(P1_4,P2_4),(Q1_4,Q2_4)>.
 *
 * Input: Pi, Qi are four torsion points on (E1 x E2)
 * Returns: Matrix: the 4x4 change of basis matrix as two 8x1x1 vectors
 */
void gluing_change_of_basis_4x2x1(
    __m512i* res_matrix01, __m512i* res_matrix23, 
    __m512i* A24, __m512i* x_4, __m512i* z_4
);

/**
 * Internal helper function for matrix multiplication.
 * Outputs res_P = P0 + P1 + P2 + P3.
 */
void gluing_internal_sum(__m512i* res_P, __m512i* P0P1, __m512i* P2P3);


/**
 * Internal helper function for matrix multiplication.
 * Outputs res_P = P0 + P1 + P2 + P3 and res_Q = Q0 + Q1 + Q2 + Q3.
 */
void gluing_internal_sum_pair(
    __m512i* res_PQ, 
    __m512i* P0Q1, __m512i* Q0P1, __m512i* P2Q3, __m512i* Q2P3
);

/**
 * Compute M * P = res_P for the 4x4 change of basis matrix. 
 * The output is in the 4x2x1 format.
 */
void gluing_apply_matrix(
    __m512i* res_P, 
    __m512i* matrix01, __m512i* matrix23, __m512i* P
);

/**
 * Compute M * P = res_P  and M * Q = res_Q for the 4x4 change of basis
 * matrix. The output is (P,Q) in the 8x1x1 format.
 */
void gluing_apply_matrix_pair(
    __m512i* res_PQ, 
    __m512i* matrix01, __m512i* matrix23, __m512i* PQ
);

/**
 * Algorithm 8.38 of the SQISign spec.
 * TODO: Not sure about the 4x2x1 in the name.
 */
void gluing_product_to_theta_pair(
    __m512i* res_P12, // 8x1x1
    __m512i* x, __m512i* z, __m512i* matrix1, __m512i* matrix2
);

/**
 * Algorithm 8.39 of the SQIsign spec.
 * Everything except matrix is in 4x2x1.
 */
bool gluing_codomain(
    __m512i* res_OA, __m512i* res_OA_dual_inv, __m512i* res_J_prime, __m512i* res_matrix01, __m512i* res_matrix23,
    __m512i* A24_12, __m512i* x_8, __m512i* z_8, const bool verify
);

/**
 * Accepts a pair of points (P1, P2) and (Q1, Q2) on E1 x E2
 * xP = (xP1, xQ1, xP2, xQ2)
 */
void gluing_eval_pair(
    __m512i* res, //8x1x1
    __m512i* A_12,  __m512i* xP, __m512i* yP, __m512i* zP,
    __m512i* xT_8, __m512i* yT_8, __m512i* zT_8,
    __m512i* J_prime, __m512i* matrix01, __m512i* matrix23
);

/**
 * Here we convert from 64-bit to 51-bit limbs
 */
void gluing_eval_product_to_theta(
    __m512i* res,
    ec_point_t* P1, ec_point_t* P2, __m512i* matrix01, __m512i* matrix23
);

/**
 * Here we convert from 64-bit to 51-bit limbs
 * TODO: Not sure if the order of the theta coords are correct!!!
 */
void gluing_eval_product_to_theta_pair(
    __m512i* res,
    ec_point_t* P1, ec_point_t* P2, ec_point_t* Q1, ec_point_t* Q2, 
    __m512i* matrix01, __m512i* matrix23
);

/**
 * Evaluation of couple points of the form (P1, 0) or (0, P2)
 * which are the only forms needed for SQIsign.
 */
bool gluing_eval_special(
    __m512i* res,
    theta_couple_point_t* P12, __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23, const bool verify
);

/**
 * Evaluation of couple points of the form (P1, 0) or (0, P2)
 * which are the only forms needed for SQIsign.
 */
bool gluing_eval_special_pair(
    __m512i* res,
    theta_couple_point_t* P12, theta_couple_point_t* Q12,
    __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23, const bool verify
);

bool gluing_eval_special_list_pair(
    __m512i* res_pts, theta_couple_point_t* pts, __m512i* inv, 
    __m512i* matrix01, __m512i* matrix23, const int num_pts, const bool verify
);

#endif