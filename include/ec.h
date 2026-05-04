#ifndef EC_H
#define EC_H

#include "intrinsics_avx512.h"

#include "sqi_ec.h"
#include "sqi_fp_constants.h"

// TODO replace digit_t with scalar_t where appropriate

typedef digit_t scalar_t[NWORDS_ORDER];
typedef scalar_t scalar_mtx_2x2_t[2][2];

// TODO: Probably want to create structs to group elements together.
// Next goal is to implement division by 3 in fp and fp2. 
// For now just precompute the inverse of 3 mod p, we can optimize later.

/**
 * Converts four x-only points in the ec_point_t type to two 4x2x1 vectors.
 * @param res_x Output: The x-cooordinates in the 4x2x1 format.
 * @param res_z Output: The z-coordinates in the 4x2x1 format.
 * @param src The four points in the ec_point_t format.
 */
void ec_point_unvectorized_to_4x4x2x1(__m512i* res_x, __m512i* res_z, ec_point_t* src);

/**
 * Converts four x-only points in the 4x2x1 format to the ec_point_t type.
 * @param res Output: The four points in the ec_point_t format.
 * @param x The x-cooordinates in the 4x2x1 format.
 * @param z The z-coordinates in the 4x2x1 format.
 */
void ec_point_4x4x2x1_to_unvectorized(ec_point_t* res, __m512i* x, __m512i* z);


/**
 * Double four x-only points simultaneously in the 4x2x1 format.
 * @param res_x Output: x-coordinates
 * @param res_z Output: z-coordinates
 * @param A24 The curves
 * @param x The input x-coordinates
 * @param z The input z-coordinates
 */
void ec_double_4x4x2x1(__m512i* res_x, __m512i* res_z, __m512i* A24, __m512i* x, __m512i* z);
/**
 * Converts each Jacobian x-only point (jac_x, jac_y, jac_z) to its Montgomery representation (x : z).
 *
 * TODO: Called gluing_kernel_to_x_only in python script, but the name/description 
 * does not seem to be accurate, no Y coord. EDIT: Think transformation doesn't need Y coord. 
 *
 * @param x Output: x coordinate of the 4 Montgomery points.
 * @param z Output: z coordinate of the 4 Montgomery points.
 * @param jac_x x coordinate of the 4 input points.
 * @param jac_z z coordinate of the 4 input points.
 */
void ec_jac_to_xz_4x4x2x1(__m512i* x, __m512i* z, __m512i* jac_x, __m512i* jac_z);

// TODO: Documentation

void ec_jac_to_ws_4x4x2x1(
    __m512i* res_x, __m512i* res_y, __m512i* res_z, __m512i* res_t, __m512i* res_A12_over_3,
    __m512i* x, __m512i* y, __m512i* z, __m512i* A12
);

void ec_ws_to_jac_4x4x2x1(
    __m512i* res_x, __m512i* res_y, __m512i* res_z,
    __m512i* x, __m512i* y, __m512i* z, __m512i* A12_over_3
);

void ec_double_ws_4x4x2x1(
    __m512i* res_x, __m512i* res_y, __m512i* res_z, __m512i* res_t,
    __m512i* x, __m512i* y, __m512i* z, __m512i* t
);

void ec_double_ws_iter_4x4x2x1(
    __m512i* res_x, __m512i* res_y, __m512i* res_z, __m512i* res_t,
    __m512i* x, __m512i* y, __m512i* z, __m512i* t, const int n
);

/**
 * Algorithm 8.13 of the SQIsign spec v2.0
 * Given distinct Jacobian points P, Q and the Montgomery coefficient A,
 * compute u, v, w such that x(P + Q) = (u - v : w) and 
 * x(P - Q) = (u + x : w).
 */
void ec_add_components_4x4x2x1(
    __m512i* res_u, __m512i* res_v, __m512i* res_w, 
    __m512i* A, __m512i* xP, __m512i* yP, __m512i* zP, 
    __m512i* xQ, __m512i* yQ, __m512i* zQ
);


// ========================= 1x2x4x1  ==================================================

void ec_point_unvec_to_1x2x4x1(__m512i* res, ec_point_t* P);

void ec_point_1x2x4x1_to_unvec(ec_point_t* res, __m512i* P);

/**
 *  P = (0 : 1) in 2x4x1
 */
void ec_set_zero_1x2x4x1(__m512i* P);

/**
 * Verify that a pair of Montgomery A-coefficients in 2x4x1 are valid, i.e. that A^2-4 != 0
 */
bool ec_curve_verif_A_1x2x4x1(__m512i* A1_A2);

void ec_double_1x2x4x1(__m512i* res, __m512i* P, __m512i* A24);
void ec_double_iter_1x2x4x1(__m512i* res, __m512i* P, __m512i* A24, const int n);

// Assumes n1 >= n2, inputs in 2x4x1
void ec_double_basis_pair_1x2x4x1(
    __m512i* res_B1, 
    __m512i* res_B2, 
    __m512i* B1, 
    __m512i* B2, 
    __m512i* A24_norm1,
    __m512i* A24_norm2,
    const int n1,
    const int n2
);

/**
 *  Assumes res != PmQ_A24
 *  @param P_Q The (x,z) points P and Q stored in 4x2x1
 *  @param PmQ_A24 P-Q and A24 stored in 4x2x1
 *  @param res Output: 2*P and P+Q stored in 4x2x1
 */
void ec_double_add_1x4x2x1(__m512i* res, __m512i* P_Q, __m512i* PmQ_A24);

/**
 *  Assumes res != PmQ_A24
 *  @param P_Q The (x,z) points P and Q stored in 4x2x1
 *  @param PmQ_A24 P-Q and the normalized A coefficient stored in the first 3 components of a 4x2x1
 *  @param res Output: 2*P and P+Q stored in 4x2x1
 */
void ec_double_add_normA_1x4x2x1(__m512i* res, __m512i* P_Q, __m512i* PmQ_A);

/**
 * Computes two double_add's in parallel.
 * The inputs and output is in the 8x1x1 format (P.x, PQ.x, P.z, PQ.z, Q.x, A24.x, Q.z, A24.z)
 * Assumes that the output vector is not the same as one of the inputs.
 *
 * @param P1_Q1_PQ1_A24 Points P1, Q1, P1-Q1 on E1 and the curve's A24
 * @param P2_Q2_PQ2_A24 Points P2, Q2, P2-Q2 on E2 and the curve's A24
 * @param R1_R2_S1_S2 Output: R1 = 2*P1, R2 = P1 + Q1 and S1 = 2*P2, S2 = P2 + Q2
 */
void ec_double_add_pair_8x1x1(__m512i* R1_R2_S1_S2, __m512i* P1_Q1_PQ1_A24, __m512i* P2_Q2_PQ2_A24);

/**
 *  @param P The (x,z) point in 2x4x1
 *  @param A24 The curve's A24 point in 2x4x1
 *  @param m The scalar as an array of digits
 *  @param mbits The bit length of the scalar m
 *  @param res Output: [m]*P in 2x4x1
 */
void ec_scalar_mul_1x2x4x1(__m512i* res, __m512i* P, __m512i* A24, const digit_t* m, const unsigned int mbits);

/**
 *  @param P The (x,z) point in 2x4x1
 *  @param A The normalized A coefficient of the curve. Stored in a 2x4x1 where the second component is ignored.
 *  @param m The scalar as an array of digits
 *  @param mbits The bit length of the scalar m
 *  @param res Output: [m]*P in 2x4x1
 */
void ec_scalar_mul_normA_1x2x4x1(__m512i* res, __m512i* P, __m512i* A, const digit_t* m, const unsigned int mbits);

// TODO: Have consistent ordering of parameters.

/**
 *  @param A24 The A24 of the curve in 2x4x1
 *  @param basis The basis P, Q stored with three points x(P), x(Q), x(P-Q) as consecutive 2x4x1's
 *  @param m The scalar as an array of NWORDS_ORDER digits
 *  @param res Output: P + [m]*Q in 2x4x1
 */
bool ec_ladder_3pt_1x2x4x1(__m512i* res, __m512i* A24, __m512i* basis, const digit_t *m);

/**
 *  @param A The normalized A coefficient of the curve. Stored in a 2x4x1 where the second component is ignored.
 *  @param basis The basis P, Q stored with three points x(P), x(Q), x(P-Q) as consecutive 2x4x1's
 *  @param m The scalar as an array of NWORDS_ORDER digits
 *  @param res Output: P + [m]*Q in 2x4x1
 */
bool ec_ladder_3pt_normA_1x2x4x1(__m512i* res, __m512i* A, __m512i* basis, const digit_t *m);

/**
 *  @param scalarP The scalar for P as an array of NWORDS_ORDER digits
 *  @param scalarQ The scalar for Q as an array of NWORDS_ORDER digits
 *  @param A24 The A24 of the curve in 2x4x1
 *  @param basis The basis P, Q stored with three points x(P), x(Q), x(P-Q) as consecutive 2x4x1's
 *  @param res Output: [scalarP]*P + [scalarQ]*Q in 2x4x1
 */
bool ec_biscalar_mul_1x2x4x1(
    __m512i* res, 
    const digit_t *scalarP,
    const digit_t *scalarQ,
    __m512i* A24,
    __m512i* basis
);

/**
 *  @param scalarP The scalar for P as an array of NWORDS_ORDER digits
 *  @param scalarQ The scalar for Q as an array of NWORDS_ORDER digits
 *  @param A The normalized A coefficient of the curve. Stored in a 2x4x1 where the second component is ignored.
 *  @param A24 The A24 point of the curve in 2x4x1
 *  @param basis The basis P, Q stored with three points x(P), x(Q), x(P-Q) as consecutive 2x4x1's
 *  @param res Output: [scalarP]*P + [scalarQ]*Q in 2x4x1
 */
bool ec_biscalar_mul_normA_1x2x4x1(
    __m512i* res, 
    const digit_t *scalarP,
    const digit_t *scalarQ,
    __m512i* A,
    __m512i* A24,
    __m512i* basis
);

/**
 * Computes res1 = mat[0,0]*P + mat[1,0]*Q and res2 = mat[0,1]*P + mat[1,1]*Q
 */
bool ec_biscalar_mul_pair_1x2x4x1(
    __m512i* res1,
    __m512i* res2,
    scalar_mtx_2x2_t* mat,
    __m512i* A24,
    __m512i* basis
);

// Uses the normalized A-coefficient.
bool ec_biscalar_mul_pair_normA_1x2x4x1(
    __m512i* res1,
    __m512i* res2,
    scalar_mtx_2x2_t *mat,
    __m512i* A,
    __m512i* A24,
    __m512i* basis
);

// Debugging functions
bool test_point_order_twof_1x2x4x1(__m512i* P, __m512i* A24, unsigned int f);
bool test_basis_order_twof_1x2x4x1(__m512i* B, __m512i* A24, unsigned int f);

// ==================== 2x2x2 ===================================================

void ec_point_unvec_to_1x2x2x2(__m512i* res, ec_point_t* P);
void ec_point_1x2x2x2_to_unvec(ec_point_t* res, __m512i* P);

/**
 *  P = (0 : 1) in 2x2x2
 */
void ec_set_zero_1x2x2x2(__m512i* P);

void ec_double_1x2x2x2(__m512i* res, __m512i* P, __m512i* A24);
void ec_double_iter_1x2x2x2(__m512i* res, __m512i* P, __m512i* A24, const int n);

// Assumes n1 >= n2, inputs in 2x4x1
void ec_double_basis_pair_1x2x2x2(
    __m512i* res_B1, 
    __m512i* res_B2, 
    __m512i* B1, 
    __m512i* B2, 
    __m512i* A24_norm1,
    __m512i* A24_norm2,
    const int n1,
    const int n2
);

/**
 *  @param P The (x,z) point in 2x2x2
 *  @param A24 The curve's A24 point in 2x2x2
 *  @param m The scalar as an array of digits
 *  @param mbits The bit length of the scalar m
 *  @param res Output: [m]*P in 2x2x2
 */
void ec_scalar_mul_1x2x2x2(__m512i* res, __m512i* P, __m512i* A24, const digit_t* m, const unsigned int mbits);

/**
 *  @param P The (x,z) point in 2x2x2
 *  @param A The normalized A coefficient of the curve. Stored in a 2x2x2 where the second component is ignored.
 *  @param m The scalar as an array of digits
 *  @param mbits The bit length of the scalar m
 *  @param res Output: [m]*P in 2x2x2
 */
void ec_scalar_mul_normA_1x2x2x2(__m512i* res, __m512i* P, __m512i* A, const digit_t* m, const unsigned int mbits);

/**
 *  @param P1 (x,z) point in 2x2x2 on the curve E1
 *  @param P2 (x,z) point in 2x2x2 on the curve E2
 *  @param A1 The normalized A coefficient of the curve E1. Stored in a 2x2x2 where the second component is ignored.
 *  @param A1 The normalized A coefficient of the curve E2. Stored in a 2x2x2 where the second component is ignored.
 *  @param m The scalar as an array of digits
 *  @param mbits The bit length of the scalar m
 *  @param res1 Output: [m]*P1 in 2x2x2
 *  @param res2 Output: [m]*P2 in 2x2x2
 */
void ec_scalar_mul_pair_normA_2x2x2(__m512i* res1, __m512i* res2, __m512i* P1, __m512i* P2, __m512i* A1, __m512i* A2, const digit_t* m, const unsigned int mbits);

/**
 *  @param A24 The A24 of the curve in 2x2x2
 *  @param basis The basis P, Q stored with three points x(P), x(Q), x(P-Q) as consecutive 2x2x2's
 *  @param m The scalar as an array of NWORDS_ORDER digits
 *  @param res Output: P + [m]*Q in 2x2x2
 */
bool ec_ladder_3pt_1x2x2x2(__m512i* res, __m512i* A24, __m512i* basis, const digit_t *m);

/**
 *  @param A The normalized A coefficient of the curve. Stored in a 2x2x2 where the second component is ignored.
 *  @param basis The basis P, Q stored with three points x(P), x(Q), x(P-Q) as consecutive 2x2x2's
 *  @param m The scalar as an array of NWORDS_ORDER digits
 *  @param res Output: P + [m]*Q in 2x2x2
 */
bool ec_ladder_3pt_normA_1x2x2x2(__m512i* res, __m512i* A, __m512i* basis, const digit_t *m);

/**
 *  @param scalarP The scalar for P as an array of NWORDS_ORDER digits
 *  @param scalarQ The scalar for Q as an array of NWORDS_ORDER digits
 *  @param A24 The A24 of the curve in 2x2x2
 *  @param basis The basis P, Q stored with three points x(P), x(Q), x(P-Q) as consecutive 2x2x2's
 *  @param res Output: [scalarP]*P + [scalarQ]*Q in 2x2x2
 */
bool ec_biscalar_mul_1x2x2x2(
    __m512i* res, 
    const digit_t *scalarP,
    const digit_t *scalarQ,
    __m512i* A24,
    __m512i* basis
);

/**
 *  @param scalarP The scalar for P as an array of NWORDS_ORDER digits
 *  @param scalarQ The scalar for Q as an array of NWORDS_ORDER digits
 *  @param A The normalized A coefficient of the curve. Stored in a 2x2x2 where the second component is ignored.
 *  @param A24 The A24 point of the curve in 2x2x2
 *  @param basis The basis P, Q stored with three points x(P), x(Q), x(P-Q) as consecutive 2x2x2's
 *  @param res Output: [scalarP]*P + [scalarQ]*Q in 2x2x2
 */
bool ec_biscalar_mul_normA_1x2x2x2(
    __m512i* res, 
    const digit_t *scalarP,
    const digit_t *scalarQ,
    __m512i* A,
    __m512i* A24,
    __m512i* basis
);

/**
 * Computes res1 = mat[0,0]*P + mat[1,0]*Q and res2 = mat[0,1]*P + mat[1,1]*Q
 */
bool ec_biscalar_mul_pair_1x2x2x2(
    __m512i* res1,
    __m512i* res2,
    scalar_mtx_2x2_t* mat,
    __m512i* A24,
    __m512i* basis
);

// Uses the normalized A-coefficient.
bool ec_biscalar_mul_pair_normA_1x2x2x2(
    __m512i* res1,
    __m512i* res2,
    scalar_mtx_2x2_t *mat,
    __m512i* A,
    __m512i* A24,
    __m512i* basis
);

void ec_normalize_A24_pair_2x2x2(__m512i* res1, __m512i* res2, __m512i* A24_1, __m512i* A24_2);
void ec_normA24_to_normA_1x2x2x2(__m512i* res, __m512i* normA24);
void ec_normA_to_normA24_1x2x2x2(__m512i* res, __m512i* normA);
void ec_normalize_curve_2x2x2(__m512i* res_A24, __m512i* res_A, __m512i* A24);
void ec_normalize_curve_SQI_2x2x2(__m512i* res_A24, __m512i* res_A, __m512i* A24);
void ec_normalize_curve_pair_2x2x2(__m512i* res_A24_1, __m512i* res_A24_2, __m512i* res_A_1, __m512i* res_A_2, __m512i* A24_1, __m512i* A24_2);
bool ec_curve_verify_normA_pair_2x2x2(__m512i* normA1, __m512i* normA2);

// j = (256 * (A^2 - 3C^2)^3) / (C^4 * (A^2 - 4C^2))
void ec_j_inv_2x2x2(__m512i* res, __m512i* AC);

void ec_proj_diff_1x2x2x2(__m512i* res_PmQ, __m512i* P, __m512i* Q, __m512i* normA);
void ec_proj_diff_SQI_1x2x2x2(__m512i* res_PmQ, __m512i* P, __m512i* Q, __m512i* normA);
// P, Q and normA are pairs of points in 4x2x1
// P = (P1.x, P1.z, P2.x, P2.z)
void ec_proj_diff_pair_4x2x1(__m512i* res_PmQ, __m512i* P, __m512i* Q, __m512i* normA);
void ec_curve_to_basis_from_hint_1x2x2x2(__m512i* res_basis, __m512i* normA, const uint8_t hint_pk);
void ec_curve_to_basis_from_hint_SQI_1x2x2x2(__m512i* res_basis, __m512i* normA, const uint8_t hint);
void ec_curve_to_basis_from_hint_2x2x2x2(__m512i* res_basis1, __m512i* res_basis2, __m512i* normA1, __m512i* normA2, const uint8_t hint1, const uint8_t hint2);

// Recover y-coordinate of a normalized x point on the Montgomery curve y^2 = x^3 + Ax^2 + x
// Returns (x, y, 1, -) in 4x2x1
// Note: There is essentially no parallelization, second slot unused. We only use the pair version
bool ec_recover_y_2x2x2(__m512i* res, __m512i* normX, __m512i* normA, const bool verify);
// Lifts a basis x(P), x(Q), x(P-Q) assuming the curve's A coefficient and x(P) are normalized.
// Outputs res_P = (xP, yP, 1) and res_Q = (xQ, yQ, zQ) in Jacobia coordinates.
bool lift_basis_normalized_2x2x2(__m512i* res_P, __m512i* res_Q, __m512i* basis, __m512i* normA, const bool verify);

// Takes as input a normalized x-only point x1 on the curve with normalized A-coefficient A1, and x2 on A2.
// It returns the y coordinate of each point, as (y_1, y_2) in 2x2x2.
bool ec_recover_y_pair_2x2x2(__m512i* res, __m512i* normX1, __m512i* normX2, __m512i* normA1, __m512i* normA2, const bool verify);
// Lifts a pair of bases x(P), x(Q), x(P-Q) assuming the curve's A coefficient and x(P) are normalized.
// Outputs each basis P, Q in Jacobian coordinates, as res_x = (P1.X, Q1.X, P2.X, Q2.X) etc.
bool lift_basis_pair_normalized_2x2x2(
    __m512i* res_x, __m512i* res_y, __m512i* res_z,
    __m512i* basis1, __m512i* basis2, __m512i* normA1, __m512i* normA2, 
    const bool verify
);

// Debugging functions
bool test_point_order_twof_1x2x2x2(__m512i* P, __m512i* A24, unsigned int f);
bool test_basis_order_twof_1x2x2x2(__m512i* B, __m512i* A24, unsigned int f);

#endif