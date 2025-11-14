#include <stdio.h> //TODO: Remove

#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "theta.h"
#include "theta_gluing.h"
#include "theta_middle.h"
#include "theta_splitting.h"

// ======================================================================================================
//                                               Single
// ======================================================================================================

void theta_isogeny_chain_single(__m512i* codom, __m512i* eval_pts, __m512i* null, 
    __m512i* ker_a, __m512i* ker_b, __m512i* pts, const int n, const int num_pts, 
    const bool randomize)
{
    bool h1, h2;
    //int isom_matrix[16];
    splitting_matrix_t isom_matrix, rand_matrix;
    __m512i c1[8], c2[8], p1[8], p2[8], k1[8], k2[8], codom_null[8], inv[8];
    fp2_copy_4x2x1(p1, ker_a);
    fp2_copy_4x2x1(p2, ker_b);
    fp2_copy_4x2x1(codom_null, null);

    // Initialize eval_pts. 8 is the number of __m512i vectors in a fp2 4x2x1 element.
    // TODO: copy impl might be inefficient.
    for (int i = 0; i < num_pts; i++) {
        fp2_copy_4x2x1(eval_pts+8*i, pts+8*i);
    }

    for (int i = 0; i < n; i++) {
        // Precompute constants for doubling
        theta_precomp(c1, c2, codom_null);

        // Double the kernel points of order 2^(n+2) to get points of order 2^(1 + 2) = 8
        // TODO: Can get rid of this copy
        fp2_copy_4x2x1(k1, p1);
        fp2_copy_4x2x1(k2, p2);
        for (int j = 0; j < n - i - 1; j++) {
            theta_double_single(k1, k1, c1, c2);
            theta_double_single(k2, k2, c1, c2);
        }

        // For the ultimate and penultimate steps we do extra / less Hadamards
        // to avoid zeros from the product structure.
        if (i == n-1) {
            h1 = true; h2 = false;
        } else if (i == n-2) {
            h1 = false; h2 = false;
        } else {
            h1 = false; h2 = true;
        }
        
        // Compute the codomain of the kernel
        theta_codomain_single(codom_null, inv, k1, k2, h1, h2);

        // Push the kernel through the images
        theta_eval_single(p1, p1, inv, h1, h2);
        theta_eval_single(p2, p2, inv, h1, h2);

        // Eval points
        for (int j = 0; j < num_pts; j++) {
            theta_eval_single(eval_pts+8*j, eval_pts+8*j, inv, h1, h2);
        }
        
    }

    // Splitting
    splitting_isom_matrix(&isom_matrix, codom_null);
    splitting_apply_isomorphism_single(codom_null, codom_null, &isom_matrix);

    for (int i = 0; i < num_pts; i++) {
        splitting_apply_isomorphism_single(eval_pts+8*i, eval_pts+8*i, &isom_matrix);
    }

    if (randomize) {
        splitting_rand_matrix(&rand_matrix);
        splitting_apply_isomorphism_single(codom_null, codom_null, &rand_matrix);
        for (int i = 0; i < num_pts; i++) {
            splitting_apply_isomorphism_single(eval_pts+8*i, eval_pts+8*i, &rand_matrix);
        }
    }

    splitting_theta_to_product(codom, codom_null);
    for (int i = 0; i < num_pts; i++) {
        splitting_theta_point_to_mont_single(eval_pts+8*i, eval_pts+8*i, codom_null);
    }
}

// ======================================================================================================
//                                                 Pair
// ======================================================================================================

void theta_isogeny_chain_pair(__m512i* codom, __m512i* eval_pts, __m512i* null, __m512i* ker, 
    __m512i* pts, const int n, const int num_pts, const bool randomize)
{
    splitting_matrix_t isom_matrix, rand_matrix;
    __m512i c1[8], c2[8], p[16], k[16], codom_null[8], inv[8];
    bool h1, h2;
    bool num_pts_is_odd = (num_pts & 0x1);
    int num_pair_pts = num_pts / 2;

    fp2_copy_8x1x1(p, ker);
    fp2_copy_4x2x1(codom_null, null);
    for (int i = 0; i < num_pts; i++) {
        fp2_copy_4x2x1(eval_pts+8*i, pts+8*i);
    }

    for (int i = 0; i < n; i++) {
        // Precompute constants for doubling
        theta_precomp(c1, c2, codom_null);

        // Double the kernel points of order 2^(n+2) to get points of order 2^(1 + 2) = 8
        // TODO: Can get rid of this copy
        fp2_copy_8x1x1(k, p);

        for (int j = 0; j < n - i - 1; j++) {
            theta_double_pair(k, k, c1, c2);
        }

        // For the ultimate and penultimate steps we do extra / less Hadamards
        // to avoid zeros from the product structure.
        if (i == n-1) {
            h1 = true; h2 = false;
        } else if (i == n-2) {
            h1 = false; h2 = false;
        } else {
            h1 = false; h2 = true;
        }
        
        // Compute the codomain of the kernel
        theta_codomain_pair(codom_null, inv, k, h1, h2);

        // Push the kernel through the images
        theta_eval_pair(p, p, inv, h1, h2);

        // Eval points
        for (int j = 0; j < num_pair_pts; j++) {
            theta_eval_pair(eval_pts+16*j, eval_pts+16*j, inv, h1, h2);
        }
        if (num_pts_is_odd) {
            theta_eval_single(eval_pts+8*(num_pts-1), eval_pts+8*(num_pts-1), inv, h1, h2);
        }
    }

    // Splitting
    splitting_isom_matrix(&isom_matrix, codom_null);
    splitting_apply_isomorphism_single(codom_null, codom_null, &isom_matrix);
    for (int i = 0; i < num_pair_pts; i++) {
        splitting_apply_isomorphism_pair(eval_pts+16*i, eval_pts+16*i, &isom_matrix);
    }
    if (num_pts_is_odd) {
        splitting_apply_isomorphism_single(eval_pts+8*(num_pts-1), eval_pts+8*(num_pts-1), &isom_matrix);
    }

    if (randomize) {
        splitting_rand_matrix(&rand_matrix);
        splitting_apply_isomorphism_single(codom_null, codom_null, &rand_matrix);
        for (int i = 0; i < num_pair_pts; i++) {
            splitting_apply_isomorphism_pair(eval_pts+16*i, eval_pts+16*i, &rand_matrix);
        }
        if (num_pts_is_odd) {
            splitting_apply_isomorphism_single(eval_pts+8*(num_pts-1), eval_pts+8*(num_pts-1), &rand_matrix);
        }
    }

    splitting_theta_to_product(codom, codom_null);
    for (int i = 0; i < num_pair_pts; i++) {
        splitting_theta_point_to_mont_pair(eval_pts+16*i, eval_pts+16*i, codom_null);
    }
    if (num_pts_is_odd) {
        splitting_theta_point_to_mont_single(eval_pts+8*(num_pts-1), eval_pts+8*(num_pts-1), codom_null);
    }
}