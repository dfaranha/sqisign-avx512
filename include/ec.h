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
void ec_point_unvectorized_to_4x2x1(__m512i* res_x, __m512i* res_z, ec_point_t* src);

/**
 * Converts four x-only points in the 4x2x1 format to the ec_point_t type.
 * @param res Output: The four points in the ec_point_t format.
 * @param x The x-cooordinates in the 4x2x1 format.
 * @param z The z-coordinates in the 4x2x1 format.
 */
void ec_point_4x2x1_to_unvectorized(ec_point_t* res, __m512i* x, __m512i* z);


/**
 * Double four x-only points simultaneously in the 4x2x1 format.
 * @param res_x Output: x-coordinates
 * @param res_z Output: z-coordinates
 * @param A24 The curves
 * @param x The input x-coordinates
 * @param z The input z-coordinates
 */
void ec_double_4x2x1(__m512i* res_x, __m512i* res_z, __m512i* A24, __m512i* x, __m512i* z);

/**
 * Converts each Jacobian x-only point (jac_x : jac_z) to its Montgomery representation (x : z).
 *
 * TODO: Called gluing_kernel_to_x_only in python script, but the name/description 
 * does not seem to be accurate, no Y coord. 
 *
 * @param x Output: x coordinate of the 4 Montgomery points.
 * @param z Output: z coordinate of the 4 Montgomery points.
 * @param jac_x x coordinate of the 4 input points.
 * @param jac_z z coordinate of the 4 input points.
 */
void jac_to_xz_4x2x1(__m512i* x, __m512i* z, __m512i* jac_x, __m512i* jac_z);


#endif