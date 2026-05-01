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
void theta_normalize_chain_input(
    theta_couple_curve_t* E12, theta_kernel_couple_points_t* ker
);

int theta_compute_product_doubles(
    __m512i* A_12, __m512i* strat_points_x, __m512i* strat_points_y, __m512i* strat_points_z,
    uint16_t* orders, int space
);

// ====================================================================
//  Isogeny chains where we have the 2^(n+2) torsion   
// ====================================================================

bool theta_isogeny_chain_single(
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

/** FILL OUT NOW
 */
bool theta_isogeny_chain_pair_vec(
    __m512i* res_A_34, 
    __m512i* res_pts_34,
    __m512i* ker_basis1,
    __m512i* ker_basis2,
    __m512i* normA_1,
    __m512i* normA_2, 
    __m512i* normA24_1,
    __m512i* normA24_2,
    theta_couple_point_t* pts, 
    const int n, 
    const int num_pts, 
    const bool randomize,
    const bool verify
);

bool theta_isogeny_chain_pair(
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

bool theta_isogeny_chain_ver(
    theta_couple_curve_t* res_E34, 
    theta_couple_curve_t* E12, 
    theta_kernel_couple_points_t* ker,
    const int n
);


// ====================================================================
//  Isogeny chains where we have the 2^n torsion only (requires sqrts)   
// ====================================================================

bool theta_isogeny_chain_no_torsion_single(
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

bool theta_isogeny_chain_no_torsion_pair(
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
