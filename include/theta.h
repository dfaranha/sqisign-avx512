#ifndef _THETA_H
#define _THETA_H

#include <stdbool.h>

#include "intrinsics_avx512.h"

// ======================================================================================================
//                                              Single    
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
void theta_sqr_single(__m512i* res, __m512i* a);

/**
 * Given the theta points a = (x_a, y_a, z_a, w_a) and b = (x_b, y_b, z_b, w_b),
 * res = (x_a * x_b, y_a * y_b, z_a * z_b, w_a * w_b).
 */
void theta_mul_single(__m512i* res, __m512i* a, __m512i* b);

/**
 * Given the theta point a = (x, y, z, w), compute its Hadamard transform.
 * res = (x', y', z', w'), with
 * x' = x + y + z + w
 * y' = x + y - z - w
 * z' = x - y + z - w
 * w' = x - y - z + w
 */
void theta_had_single(__m512i* res, __m512i* a);

/**
 * Computes the constants c1 and c2 needed for doubling wrt the theta null point.
 */
void theta_precomp(__m512i* c1, __m512i* c2, __m512i* null);

/**
 * Double the theta point a, using the precomputed constants c1 and c2.
 */
void theta_double_single(__m512i* res, __m512i* a, __m512i* c1, __m512i* c2);

/**
 * Compute the dual theta null point of the codomain of the (2,2)-isogeny, as well as its inverse.
 * @param res Output. The dual theta null point of the codomain, (1 : beta : gamma : delta).
 * @param res_inv Output. The coordinate-wise inverse of the dual theta null point, (1 : beta^-1 : gamma^-1 : delta^-1).
 * @param k1, k2 The theta points defining the kernel of the (2,2)-isogeny.
 * @param h1 Flag for initally applying Hadamard to k1 and k2.
 * @param h2 Flag for applying Hadamard to the result at the end.
 */
void theta_codomain_single(__m512i* res, __m512i* res_inv, __m512i* k1, __m512i* k2, bool h1, bool h2);

/**
 * Evaluate a theta point through a (2,2)-isogeny given the precomputed constants.
 * @param a The theta point to be evaluated.
 * @param inv The inverse of the dual theta null point, (1 : beta^-1 : gamma^-1 : delta^-1).
 * @param h1 Flag for initially applying Hamard to a.
 * @param h2 Flag for applying Hadamard to res at the end.
 */
void theta_eval_single(__m512i* res, __m512i* a, __m512i* inv, bool h1, bool h2);

/**
 * Compute the (2^n, 2^n)-isogeny chain defined by the theta points k1 and k2, and outputs the codomain.
 * There is no gluing step, we assume the points are already in theta-coordinates.
 * @param null The theta null point of the domain.
 * @param a, b The points of order 2^(n+2) defining the isogeny.
 * @param n The length of the isogeny chain.
 */
void theta_isogeny_chain_single(__m512i* null, __m512i* a, __m512i* b, const int n);

// ======================================================================================================
//                                                 Pair
// ======================================================================================================

/**
 * Here we store two theta points (a, b, c, d) and (e, f, g, h) simultaneously in the 8x1x1 Fp2 format.
 * There are 16 vectors, the first 8 contain the real coefficients, the latter the imaginary coefficients.
 * In each vector the pattern is a, e, b, f, c, g, d, h.
 */

/** 
 * Given the theta points a=((x, y, z, w), (x', y', z', w')),
 * res = ((x^2, y^2, z^2, w^2), (x'^2, y'^2, z'^2, w'^2)).
 */
void theta_sqr_pair(__m512i* res, __m512i* a);

/**
* Multiplies the pair of theta points in a with the (single) theta point c.
*/
void theta_mul_pair(__m512i* res, __m512i* a, __m512i* c);

/**
* res is the Hadamard transform of the two theta points stored in a.
*/
void theta_had_pair(__m512i* res, __m512i* a);

// This assumes a non-interleaved format
void theta_had_pair_alt(__m512i* res, __m512i* a);

/**
 * Double the pair of theta points in a, using the precomputed constants c1 and c2.
 */
void theta_double_pair(__m512i* res, __m512i* a, __m512i* c1, __m512i* c2);

/**
 * Evaluate a pair of theta points through a (2,2)-isogeny given the precomputed constants.
 * @param a The pair of theta points to be evaluated.
 * @param inv The inverse of the dual theta null point, (1 : alpha^-1 : beta^-1 : delta^-1), stored in the 4x2x1 format.
 * @param h1 Flag for initially applying the Hamard to a.
 * @param h2 Flag for applying Hadamard to the result at the end.
 */
void theta_eval_pair(__m512i* res, __m512i* a, __m512i* inv, bool h1, bool h2);

#endif
