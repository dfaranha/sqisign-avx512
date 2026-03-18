#ifndef EC_H
#define EC_H

#include "intrinsics_avx512.h"

#include "sqi_ec.h"

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


// ==== For 1D isogeny ====
// TODO: Names might not be consistent. ec_double_4x2x1 doubles 4 points at once,
// while ec_double_1x2x4x1 doubles only one point.

void ec_point_unvec_to_1x2x4x1(__m512i* res, ec_point_t* P);

void ec_point_1x2x4x1_to_unvec(ec_point_t* res, __m512i* P);

void ec_double_1x2x4x1(__m512i* res, __m512i* P, __m512i* A24);
void ec_double_iter_1x2x4x1(__m512i* res, __m512i* P, __m512i* A24, const int n);

/**
 *  @param P_Q The (x,z) points P and Q stored in 4x2x1
 *  @param PmQ_A24 P-Q and A24 stored in 4x2x1
 *  @param res Output: 2*P and P+Q stored in 4x2x1
 */
void ec_double_add_1x2x4x1(__m512i* res, __m512i* P_Q, __m512i* PmQ_A24);

/**
 *  @param A24 The A24 of the curve in 2x4x1
 *  @param basis The basis P, Q stored with three points x(P), x(Q), x(P-Q) as consecutive 2x4x1's
 *  @param m The scalar
 *  @param res Output: P + [m]*Q in 2x4x1
 */
bool ec_ladder_3pt_1x2x4x1(__m512i* res, __m512i* A24, __m512i* basis, const digit_t *m);

// ========================= Debugging functions =================================================

bool test_point_order_twof_1x2x4x1(__m512i* P, __m512i* A24, unsigned int f);
bool test_basis_order_twof_1x2x4x1(__m512i* B, __m512i* A24, unsigned int f);

#endif