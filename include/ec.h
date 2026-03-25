#ifndef EC_H
#define EC_H

#include "intrinsics_avx512.h"

#include "sqi_ec.h"
#include "sqi_fp_constants_lvl3.h"

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

/**
 *  Assumes res != PmQ_A24
 *  @param P_Q The (x,z) points P and Q stored in 4x2x1
 *  @param PmQ_A24 P-Q and A24 stored in 4x2x1
 *  @param res Output: 2*P and P+Q stored in 4x2x1
 */
void ec_double_add_1x2x4x1(__m512i* res, __m512i* P_Q, __m512i* PmQ_A24);

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

// TODO: Have consistent ordering of parameters.

/**
 *  @param A24 The A24 of the curve in 2x4x1
 *  @param basis The basis P, Q stored with three points x(P), x(Q), x(P-Q) as consecutive 2x4x1's
 *  @param m The scalar as an array of NWORDS_ORDER digits
 *  @param res Output: P + [m]*Q in 2x4x1
 */
bool ec_ladder_3pt_1x2x4x1(__m512i* res, __m512i* A24, __m512i* basis, const digit_t *m);

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
 * Computes res1 = mat[0,0]*P + mat[1,0]*Q and res2 = mat[0,1]*P + mat[1,1]*Q
 */
bool ec_biscalar_mul_pair_1x2x4x1(
    __m512i* res1,
    __m512i* res2,
    scalar_mtx_2x2_t* mat,
    __m512i* A24,
    __m512i* basis
);

// ========================= Debugging functions =================================================

bool test_point_order_twof_1x2x4x1(__m512i* P, __m512i* A24, unsigned int f);
bool test_basis_order_twof_1x2x4x1(__m512i* B, __m512i* A24, unsigned int f);




// TODO: Delete
void ec_diff_add_1x2x4x1(__m512i* P_Q, __m512i* PmQ_A24);

#endif