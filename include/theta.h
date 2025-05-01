#ifndef _THETA_H
#define _THETA_H

#include "intrinsics_avx512.h"

// ======================================================================================================
//                                                 8x1x1
// ======================================================================================================

/**
 * Here we store two theta points (a, b, c, d) and (e, f, g, h) simultaneously in the 8x1x1 Fp2 format.
 * There are 16 vectors, the first 8 contain the real coefficients, the latter the imaginary coefficients.
 */

/** 
 * Given the theta points a=((x, y, z, w), (x', y', z', w')),
 * res = ((x^2, y^2, z^2, w^2), (x'^2, y'^2, z'^2, w'^2)).
 */
void theta_sqr_8x1x1(__m512i* res, __m512i* a);

/**
 * Given the theta points in a and b, res will be the pairwise product of each Fp2 coordinate.
 */
void theta_mul_8x1x1(__m512i* res, __m512i* a, __m512i* b);
 
/**
 * res is the Hadamard transform of the two theta points stored in a.
 */
void theta_had_8x1x1(__m512i* res, __m512i* a);


// ======================================================================================================
//                                                 4x2x1
// ======================================================================================================

/**
 * The "naive" approach:
 * A point in theta coordinates is represented by 4 Fp2 coordinates (a,b,c,d).
 * Here, we store them using the 4x2x1 Fp2 format.
 * There are 8 vectors, where the i-th vector stores the i-th limb of a0, a1, b0, b1, c0, c1, d0, d1.
 * Here a = a0 + a1 * i, etc.
 */

/** 
 * Given the theta point a = (x, y, z, w),
 * res = (x^2, y^2, z^2, w^2).
 */
void theta_sqr_4x2x1(__m512i* res, __m512i* a);

/**
 * Given the theta points a = (x_a, y_a, z_a, w_a) and b = (x_b, y_b, z_b, w_b),
 * res = (x_a * x_b, y_a * y_b, z_a * z_b, w_a * w_b).
 */
void theta_mul_4x2x1(__m512i* res, __m512i* a, __m512i* b);

/**
 * Given the theta point a = (x, y, z, w), compute its Hadamard transform.
 * res = (x', y', z', w'), with
 * x' = x + y + z + w
 * y' = x + y - z - w
 * z' = x - y + z - w
 * w' = x - y - z + w
 */
void theta_had_4x2x1(__m512i* res, __m512i* a);

#endif
