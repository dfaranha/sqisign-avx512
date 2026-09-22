#ifndef THETA_GLUING_H
#define THETA_GLUING_H

#include "intrinsics_avx512.h"
#include "sqi_ec.h"
#include "sqi_hd.h"

/**
 * Compute the three symmetric elements above the 4-torsion basis of each curve
 * of the elliptic product.
 *
 * g1, g2, g3 are packed as (s00, s01) of the first curve followed by (s00, s01)
 * of the second one, gs holds g1.s10 of the first curve and g2.s10 of the
 * second, and deltas is (delta_1 * delta_2, delta_1, delta_2, -).
 */
void gluing_symmetric_elements_4x2x1(
    __m512i* g1, __m512i* g2, __m512i* g3, __m512i* gs, __m512i* deltas,
    __m512i* x_4, __m512i* z_4
);

/**
 * Compute the change of basis matrix that transforms the standard product
 * theta structure into the theta structure induced by the 4-symplectic
 * basis <(P1_4, P2_4), (Q1_4, Q2_4)>, from the symmetric elements above it.
 *
 * Input: the 4-torsion basis (P1_4, Q1_4, P2_4, Q2_4) in 4x2x1.
 * Returns: the 4x4 change of basis matrix as two 8x1x1 vectors.
 */
void gluing_change_of_basis_4x2x1(
    __m512i* res_matrix01, __m512i* res_matrix23, __m512i* x_4, __m512i* z_4
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
 * As the above, for a kernel that is already x-only. res_J_pair may be NULL,
 * otherwise it receives the inverse dual images of both kernel points in
 * 8x1x1, which is what the barycentric evaluation scales by.
 */
bool gluing_codomain_x_only(
    __m512i* res_OA, __m512i* res_OA_dual_inv, __m512i* res_J_prime, __m512i* res_J_pair,
    __m512i* res_matrix01, __m512i* res_matrix23,
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
 * Same, from the ladder state P = ([m]G, [m-1]G) and the sum Dp and difference
 * Dm of the coordinates of the generators, which are the translation points.
 */
void gluing_eval_pair_bary(
    __m512i* res, //8x1x1
    __m512i* xe, __m512i* ze, __m512i* xo, __m512i* zo, __m512i* Dp, __m512i* Dm,
    __m512i* J_pair, __m512i* matrix01, __m512i* matrix23, const bool last_hadamard
);

/**
 * Evaluation of a pair of points from their addition components, whatever
 * representation these were computed from.
 */
void gluing_eval_components_pair(
    __m512i* res, //8x1x1
    __m512i* u, __m512i* v, __m512i* w,
    __m512i* J_pair, __m512i* matrix01, __m512i* matrix23, const bool last_hadamard
);

/**
 * Projective inverse (yzw : xzw : xyw : xyz) of a pair of theta points.
 */
void theta_invert_pair(__m512i* res, __m512i* P);

// P1_P2 is (X1, Z1, X2, Z2) in 4x2x1
void gluing_eval_product_to_theta(
    __m512i* res,
    __m512i* P1_P2, __m512i* matrix01, __m512i* matrix23
);

/**
 * Here we convert from 64-bit to 51-bit limbs
 */
void gluing_eval_product_to_theta_unvec(
    __m512i* res,
    ec_point_t* P1, ec_point_t* P2, __m512i* matrix01, __m512i* matrix23
);

// P1_P2 and Q1_Q2 are pairs of points in 4x2x1 as (X1, Z1, X2, Z2)
void gluing_eval_product_to_theta_pair(
    __m512i* res,
    __m512i* P1_P2, __m512i* Q1_Q2, 
    __m512i* matrix01, __m512i* matrix23
);

/**
 * Here we convert from 64-bit to 51-bit limbs
 */
void gluing_eval_product_to_theta_pair_unvec(
    __m512i* res,
    ec_point_t* P1, ec_point_t* P2, ec_point_t* Q1, ec_point_t* Q2, 
    __m512i* matrix01, __m512i* matrix23
);

// P1_P2 = (X1, Z1, X2, Z2) in 4x2x1
bool gluing_eval_special(
    __m512i* res,
    __m512i* P1_P2, __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23, const bool verify
);
/**
 * Evaluation of couple points of the form (P1, 0) or (0, P2)
 * which are the only forms needed for SQIsign.
 */
bool gluing_eval_special_unvec(
    __m512i* res,
    theta_couple_point_t* P12, __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23, const bool verify
);

// P1_P2 and Q1_Q2 are pairs of points in 4x2x1 as (X1, Z1, X2, Z2)
bool gluing_eval_special_pair(
    __m512i* res,
    __m512i* P1_P2, __m512i* Q1_Q2,
    __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23, const bool verify
);
/**
 * Evaluation of couple points of the form (P1, 0) or (0, P2)
 * which are the only forms needed for SQIsign.
 */
bool gluing_eval_special_pair_unvec(
    __m512i* res,
    theta_couple_point_t* P12, theta_couple_point_t* Q12,
    __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23, const bool verify
);

bool gluing_eval_special_list_pair(
    __m512i* res_pts, __m512i* pts, __m512i* inv, 
    __m512i* matrix01, __m512i* matrix23, const int num_pts, const bool verify
);
bool gluing_eval_special_list_pair_unvec(
    __m512i* res_pts, theta_couple_point_t* pts, __m512i* inv, 
    __m512i* matrix01, __m512i* matrix23, const int num_pts, const bool verify
);

#endif