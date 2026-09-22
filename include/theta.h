#ifndef _THETA_H
#define _THETA_H

#include <stdbool.h>
#include "intrinsics_avx512.h"
#include "sqi_hd.h"
#include "sqi_ec.h"

// ====================================================================
//                 Helper functions for isogeny chains    
// ====================================================================

// Normalizes in place
void theta_normalize_chain_input_unvec(
    theta_couple_curve_t* E12, theta_kernel_couple_points_t* ker
);
void theta_normalize_chain_input(__m512i* A24_1, __m512i* A24_2, __m512i* ker_basis1, __m512i* ker_basis2);


int theta_compute_product_doubles(
    __m512i* A_12, __m512i* strat_points_x, __m512i* strat_points_y, __m512i* strat_points_z,
    uint16_t* orders, int space
);

// ====================================================================
//  Isogeny chains where we have the 2^(n+2) torsion   
// ====================================================================

/**
 * Computes the (2^n,2^n) isogeny E1 x E2 -> E3 x E4 defined by the kernel, and evaluates pts through it.
 * Assumes that the kernel has the 2^(n+2) torsion basis, so that sqrt's can be avoided in the final two codomains.
 * @param res_A_34 Output: The AC coeffients of the codomain as (A3, C3, A4, C4) in 4x2x1
 * @param res_pts_34 Output: The evaluated pts as a list of (X3, Z3, X4, Z4) in 4x2x1
 * @param ker_basis1 The kernel basis for E1, as a list of three points (X, Z) in 2x2x2
 * @param ker_basis2 The kernel basis for E2, as a list of three points (X, Z) in 2x2x2
 * @param normA_1 The normalized Montgomery A-coefficient of E1 as (A1, 1) in 2x2x2
 * @param normA_2 The normalized Montgomery A-coefficient of E2 as (A2, 1) in 2x2x2
 * @param normA24_1 The normalized A24-coefficient of E1 as (A24_1, 1) in 2x2x2
 * @param normA24_2 The normalized A24_coefficient of E2 as (A24_2, 1) in 2x2x2
 * @param pts The points to be evaluated, as a list of (X1, Z1, X2, Z2) in 4x2x1
 * @param n The length of the (2^n, 2^n)-isogeny
 * @param num_pts The number of points to be evaluated
 * @param randomize When set to true, a random representative is sampled for the codomain
 * @param verify This flag enables additional checks during verification.
 */
bool theta_isogeny_chain_vec(
    __m512i* res_A_34, 
    __m512i* res_pts_34,
    __m512i* ker_basis1,
    __m512i* ker_basis2,
    __m512i* normA_1,
    __m512i* normA_2, 
    __m512i* normA24_1,
    __m512i* normA24_2,
    __m512i* pts, 
    const int n, 
    const int num_pts, 
    const bool randomize,
    const bool verify
);

/**
 * As theta_isogeny_chain_vec, with the first step computed from x-only points:
 * the kernel is doubled with a ladder keeping [2^m - 1]G alongside [2^m]G and
 * pushed through the gluing with barycentric coordinates, so neither the
 * Montgomery A-coefficient nor a square root is needed. This is the chain
 * SQIsign verification runs, with no evaluation points and verify set.
 * @param ker_basis1 The kernel basis for E1, only x(P) and x(Q) are read
 * @param ker_basis2 The kernel basis for E2, only x(P) and x(Q) are read
 */
bool theta_isogeny_chain_bary_vec(
    __m512i* res_A_34, 
    __m512i* res_pts_34,
    __m512i* ker_basis1,
    __m512i* ker_basis2,
    __m512i* normA24_1,
    __m512i* normA24_2, 
    __m512i* pts, 
    const int n, 
    const int num_pts, 
    const bool randomize,
    const bool verify
);

bool theta_isogeny_chain_bary(
    theta_couple_curve_t* res_E34, 
    theta_couple_point_t* res_pts, 
    theta_couple_curve_t* E12, 
    theta_kernel_couple_points_t* ker,
    theta_couple_point_t* pts, 
    const int n, 
    const int num_pts, 
    const bool randomize,
    const bool verify
);

bool theta_isogeny_chain(
    theta_couple_curve_t* res_E34, 
    theta_couple_point_t* res_pts, 
    theta_couple_curve_t* E12, 
    theta_kernel_couple_points_t* ker,
    theta_couple_point_t* pts, 
    const int n, 
    const int num_pts, 
    const bool randomize,
    const bool verify
);


// ====================================================================
//  Isogeny chains where we have the 2^n torsion only (requires sqrts)   
// ====================================================================

/**
 * Computes the (2^n,2^n) isogeny E1 x E2 -> E3 x E4 defined by the kernel, and evaluates pts through it.
 * Assumes ker_basis1 and ker_basis2 defines the kernel (with no extra torsion information).
 * @param res_A_34 Output: The AC coeffients of the codomain as (A3, C3, A4, C4) in 4x2x1
 * @param res_pts_34 Output: The evaluated pts as a list of (X3, Z3, X4, Z4) in 4x2x1
 * @param ker_basis1 The kernel basis for E1, as a list of three points (X, Z) in 2x2x2
 * @param ker_basis2 The kernel basis for E2, as a list of three points (X, Z) in 2x2x2
 * @param normA_1 The normalized Montgomery A-coefficient of E1 as (A1, 1) in 2x2x2
 * @param normA_2 The normalized Montgomery A-coefficient of E2 as (A2, 1) in 2x2x2
 * @param normA24_1 The normalized A24-coefficient of E1 as (A24_1, 1) in 2x2x2
 * @param normA24_2 The normalized A24_coefficient of E2 as (A24_2, 1) in 2x2x2
 * @param pts The points to be evaluated, as a list of (X1, Z1, X2, Z2) in 4x2x1
 * @param n The length of the (2^n, 2^n)-isogeny
 * @param num_pts The number of points to be evaluated
 * @param randomize When set to true, a random representative is sampled for the codomain
 * @param verify This flag enables additional checks during verification.
 */
bool theta_isogeny_chain_no_torsion_vec(
    __m512i* res_A_34, 
    __m512i* res_pts_34,
    __m512i* ker_basis1,
    __m512i* ker_basis2,
    __m512i* normA_1,
    __m512i* normA_2, 
    __m512i* normA24_1,
    __m512i* normA24_2,
    __m512i* pts, 
    const int n, 
    const int num_pts, 
    const bool randomize,
    const bool verify
);

bool theta_isogeny_chain_no_torsion(
    theta_couple_curve_t* res_E34, 
    theta_couple_point_t* res_pts, 
    theta_couple_curve_t* E12, 
    theta_kernel_couple_points_t* ker,
    theta_couple_point_t* pts, 
    const int n, 
    const int num_pts, 
    const bool randomize,
    const bool verify
);

#endif
