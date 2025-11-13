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
 * Compute the (2^n, 2^n)-isogeny chain defined by the theta points a and b, and outputs the codomain.
 * There is no gluing step, we assume the points are already in theta-coordinates.
 * @param prod Output. The codomain product E3 x E4
 * @param eval_pts Output. The evaluation of the points in pts.
 * @param null The theta null point of the domain.
 * @param ker_a, ker_b The points of order 2^(n+2) defining the isogeny.
 * @param pts The points to be evalutated
 * @param n The length of the isogeny chain.
 * @param num_pts The number of points to be evaluated
 */
void theta_isogeny_chain_single(__m512i* prod, __m512i* eval_pts, __m512i* null, 
    __m512i* ker_a, __m512i* ker_b, __m512i* pts, const int n, const int num_pts, 
    const bool randomize);

// ======================================================================================================
//                                                 Pair
// ======================================================================================================

/**
 * Here we (sometimes) store two theta points (a, b, c, d) and (e, f, g, h) simultaneously in the 8x1x1 Fp2 format.
 * There are 16 vectors, the first 8 contain the real coefficients, the latter the imaginary coefficients.
 * In each vector the pattern is a, e, b, f, c, g, d, h.
 */

/**
 * Compute the (2^n, 2^n)-isogeny chain defined by the theta points a and b, and outputs the codomain.
 * There is no gluing step, we assume the points are already in theta-coordinates.
 * @param prod Output. The codomain product E3 x E4
 * @param eval_pts Output. The evaluation of the points in pts.
 * @param null The theta null point of the domain in the 4x2x1 format.
 * @param ker The points of order 2^(n+2) defining the isogeny, stored in the 8x1x1 format.
 * @param pts The points to be evalutated. 
 * @param n The length of the isogeny chain.
 * @param num_pts The number of points to be evaluated
 */
void theta_isogeny_chain_pair(__m512i* codom, __m512i* eval_pts, __m512i* null, __m512i* ker, 
    __m512i* pts, const int n, const int num_pts, const bool randomize);

#endif
