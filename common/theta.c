#include <assert.h>
#include <stdio.h> //TODO: Remove

#include "ec.h"
#include "fp.h"
#include "fp_consts.h"
#include "fp_unvec.h"
#include "fp2.h"
#include "theta.h"
#include "theta_gluing.h"
#include "theta_middle.h"
#include "theta_splitting.h"
#include "utils.h"
#include "sqi_test_extras.h"
#include "strategy.h"
#include "sqi_ec_params.h"

// ====================================================================
//                 Helper functions for isogeny chains    
// ====================================================================

// Relative cost of theta_double_pair and theta_eval_pair. Only the ratio matters.
// Tuned by measurement: an evaluation costs more than its multiplications alone,
// since theta_eval_list_pair reloads each node from strat_theta_pair.
#define THETA_DBL_COST 10
#define THETA_EVAL_COST 6

// Splits for the (2,2)-chain, built once. A chain is never longer than the
// available even torsion.
static const uint16_t *
theta_strategy(void)
{
    static uint16_t split[TORSION_EVEN_POWER + 1];
    static bool ready = false;

    if (!ready) {
        strategy_splits(split, TORSION_EVEN_POWER, THETA_DBL_COST, THETA_EVAL_COST);
        ready = true;
    }
    return split;
}

// TODO: Merge with vectorized verif impl, or remove and make interface assume normalized?
void theta_normalize_chain_input_unvec(
    theta_couple_curve_t* E12, theta_kernel_couple_points_t* ker
)
{
    fp2_t inverses[4];
    fp2_copy(&inverses[0], &E12->E1.C);
    fp2_copy(&inverses[1], &E12->E2.C);
    fp2_copy(&inverses[2], &ker->T1.P1.z);
    fp2_copy(&inverses[3], &ker->T1.P2.z);

    fp2_batched_inv(inverses, 4);

    // Set Z-values to 1
    fp2_set_one(&E12->E1.C);
    fp2_set_one(&E12->E2.C);
    fp2_set_one(&ker->T1.P1.z);
    fp2_set_one(&ker->T1.P2.z);

    // Multiply by the inverses
    fp2_mul(&E12->E1.A, &E12->E1.A, &inverses[0]);
    fp2_mul(&E12->E2.A, &E12->E2.A, &inverses[1]);
    fp2_mul(&ker->T1.P1.x, &ker->T1.P1.x, &inverses[2]);
    fp2_mul(&ker->T1.P2.x, &ker->T1.P2.x, &inverses[3]);

    // Compute A24 in place
    fp2_add_one(&E12->E1.A24.x, &E12->E1.A); 
    fp2_add_one(&E12->E1.A24.x, &E12->E1.A24.x);
    fp_copy(&E12->E1.A24.x.im, &E12->E1.A.im);
    fp2_half(&E12->E1.A24.x, &E12->E1.A24.x); 
    fp2_half(&E12->E1.A24.x, &E12->E1.A24.x); 
    fp2_set_one(&E12->E1.A24.z);

    // TODO: potentially remove me, for the aux curve
    // This doesnt need to be done, but it might do after more AVX things?
    fp2_add_one(&E12->E2.A24.x, &E12->E2.A); 
    fp2_add_one(&E12->E2.A24.x, &E12->E2.A24.x); 
    fp_copy(&E12->E2.A24.x.im, &E12->E2.A.im);
    fp2_half(&E12->E2.A24.x, &E12->E2.A24.x); 
    fp2_half(&E12->E2.A24.x, &E12->E2.A24.x);
    fp2_set_one(&E12->E2.A24.z);
}

void theta_normalize_chain_input(__m512i* A24_1, __m512i* A24_2, __m512i* ker_basis1, __m512i* ker_basis2)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], idx0, idx1;
    fp2_2x2x2_to_4x2x1(t0, A24_1, A24_2); // (A24_1, C24_1, A24_2, C24_2)
    fp2_2x2x2_to_4x2x1(t1, ker_basis1, ker_basis2); // (P1.x, P1.z, P2.x, P2.z)

    // t0 = (A24_1, A24_1, P_chall.x, P_aux.x)
    // t1 = (C24_1, C24_2, P_chall.z, P_aux.z)
    idx0 = VSET(0, 1, 4, 5, 8, 9, 12, 13);
    idx1 = VSET(2, 3, 6, 7, 10, 11, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        __m512i s0 = t0[i], s1 = t1[i];
        t0[i] = VBLENDPERM(idx0, s0, s1);
        t1[i] = VBLENDPERM(idx1, s0, s1);
    }

    fp2_inv_4x2x1(t1, t1);
    fp2_mul_4x2x1(t0, t0, t1);

    // t0 = (A24_1, 1, A24_2, 1)
    // t1 = (P_chall.x, 1, P_aux.x, 1)
    fp2_set_binary_4x2x1(t1, 0b1111);
    idx0 = VSET(0, 1, 8, 9, 2, 3, 8, 9);
    idx1 = VSET(4, 5, 8, 9, 6, 7, 8, 9);
    for (int i = 0; i < NUM_LIMBS; i++) {
        __m512i s0 = t0[i], s1 = t1[i];
        t0[i] = VBLENDPERM(idx0, s0, s1);
        t1[i] = VBLENDPERM(idx1, s0, s1);
    }

    fp2_4x2x1_to_2x2x2(A24_1, A24_2, t0);
    fp2_4x2x1_to_2x2x2(ker_basis1, ker_basis2, t1);
}

int theta_compute_product_doubles(
    __m512i* A_12, __m512i* strat_points_x, __m512i* strat_points_y, __m512i* strat_points_z,
    uint16_t* orders, int space
)
{
    __m512i A12_over_3[NUM_LIMBS], strat_points_t[space * NUM_LIMBS];
    // When we start doubling, only the first element is defined. We convert this
    // from the Montgomery to the Weierstrass model

    // TODO: This conversion should be possible in place. Double check that it works.
    // TODO: Standardize naming for Montgomery coeff, should be A_12 not A12.
    ec_jac_to_ws_4x4x2x1(strat_points_x, strat_points_y, strat_points_z, strat_points_t,
        A12_over_3, strat_points_x, strat_points_y, strat_points_z, A_12);
    
    // Now we store within strat_points the Weierstrass coordinates instead.
    // In C I guess this is slightly more annoying as we need different allocation?
    // In fact, we could just store X, Y, Z in here and have U as an extra AXV blob
    // which we keep somewhere in the stack simply within the loop, as we don't use
    // it for the conversion back in the end anyway.
    // strat_points[0] = X, Y, Z, U

    // All doubling is performed in the Weierstrass model without conversions
    int k = 0, n;
    while (orders[k] != 1)
    {
        k++;
        // TODO: need to balance this in the C code to find the optional
        // splitting of the chain, as eval is more expensive than doubling
        if (orders[k - 1] >= 16) { n = orders[k - 1] / 2; }
        else { n = orders[k - 1] - 1; }
            
        ec_double_ws_iter_4x4x2x1(
            strat_points_x + k*NUM_LIMBS, strat_points_y + k*NUM_LIMBS, 
            strat_points_z + k*NUM_LIMBS, strat_points_t + k*NUM_LIMBS,
            strat_points_x + (k-1)*NUM_LIMBS, strat_points_y + (k-1)*NUM_LIMBS,
            strat_points_z + (k-1)*NUM_LIMBS, strat_points_t + (k-1)*NUM_LIMBS,
            n
        );
        orders[k] = orders[k - 1] - n;
    }

    // Now we must convert the Weierstrass model back to the Montgomery model
    // Which is done only on the points we have doubled so far.
    for (int i = 0; i <= k; i++)
    {
        ec_ws_to_jac_4x4x2x1(
            strat_points_x + i*NUM_LIMBS, strat_points_y + i*NUM_LIMBS, strat_points_z + i*NUM_LIMBS,
            strat_points_x + i*NUM_LIMBS, strat_points_y + i*NUM_LIMBS, strat_points_z + i*NUM_LIMBS,
            A12_over_3
        );
    }

    return k;
}

// ====================================================================
//  Isogeny chains where we have the 2^(n+2) torsion    
// ====================================================================

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
)
{
    __m512i A_12[NUM_LIMBS], A24_12[NUM_LIMBS], ker_x[NUM_LIMBS], ker_y[NUM_LIMBS], ker_z[NUM_LIMBS], OA[NUM_LIMBS], OA_dual_inv[2*NUM_LIMBS];
    __m512i J[NUM_LIMBS], matrix01[2*NUM_LIMBS], matrix23[2*NUM_LIMBS], T8_x[NUM_LIMBS], T8_y[NUM_LIMBS], T8_z[NUM_LIMBS];
    __m512i eval_pts_theta[num_pts * NUM_LIMBS];
    __m512i t0[N4x2], t1[N4x2], idx;
    __m512i c1[2*NUM_LIMBS], c2[2*NUM_LIMBS];
    splitting_matrix_t isom_matrix, rand_matrix;
    bool h1, h2, success = true;

    // Lift each basis x(P), x(Q), x(P-Q) to P, Q in Jacobian coordinates.
    // We pack the coordinates into three blobs
    // ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X), etc.
    success &= lift_basis_pair_normalized_2x2x2(ker_x, ker_y, ker_z, ker_basis1, ker_basis2, normA_1, normA_2, verify);
    
    // Packing curve coefficients
    // A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    idx = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(idx, normA_1[i]);
        t1[i] = VPERM512(idx, normA_2[i]);
    }
    fp2_2x2x2_to_4x2x1(A_12, t0, t1);
    // A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(idx, normA24_1[i]);
        t1[i] = VPERM512(idx, normA24_2[i]);
    }
    fp2_2x2x2_to_4x2x1(A24_12, t0, t1);

    // Setting up the strategy
    const uint16_t *split = theta_strategy();
    int space = strategy_space(split, n);

    uint16_t orders[space];
    __m512i strat_points_x[space*NUM_LIMBS], strat_points_y[space*NUM_LIMBS], strat_points_z[space*NUM_LIMBS];
    // TODO: ker_x, ker_y, ker_z never used again, remove copies.
    orders[0] = n;
    fp2_copy_4x2x1(strat_points_x, ker_x);
    fp2_copy_4x2x1(strat_points_y, ker_y);
    fp2_copy_4x2x1(strat_points_z, ker_z);

    int current = theta_compute_product_doubles(
        A_12, strat_points_x, strat_points_y, strat_points_z, orders, space
    );

    success &= gluing_codomain(
        OA, OA_dual_inv, J, matrix01, matrix23, A24_12, 
        strat_points_x + current*NUM_LIMBS, strat_points_z + current*NUM_LIMBS,
        verify
    );

    // Package the point we use for translation
    // T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    // T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    // T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    idx = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        int j = current*NUM_LIMBS + i;
        T8_x[i] = VPERM512(idx, strat_points_x[j]);
        T8_y[i] = VPERM512(idx, strat_points_y[j]);
        T8_z[i] = VPERM512(idx, strat_points_z[j]);
    }

    // Push kernel though the gluing isogeny
    // 4 Jacobian points for the kernel become 2 theta points
    // TODO: Is current = space-1 here? And even if it is not, is it better to use 8*space?
    __m512i strat_theta_pair[space * 2*NUM_LIMBS];
    for (int i = 0; i < current; i++) {
        gluing_eval_pair(
            strat_theta_pair + i*2*NUM_LIMBS, //8x1x1
            A_12, strat_points_x + i*NUM_LIMBS, strat_points_y + i*NUM_LIMBS, strat_points_z + i*NUM_LIMBS,
            T8_x, T8_y, T8_z, J, matrix01, matrix23
        );
        orders[i]--;
    }
    current--;

    success &= gluing_eval_special_list_pair(
        eval_pts_theta, pts, OA_dual_inv, 
        matrix01, matrix23, num_pts, verify
    );

    // Step Two: Compute the rest of the chain
    int step = 0;
    bool codom_verify = verify;
    while (current >= 0 && orders[current] != 0) {
        // Precompute constants for doubling, unused when this step has none
        if (orders[current] != 1) {
            theta_precomp_pair(c1, c2, OA);
        }

        while (orders[current] != 1) {
            current++;
            assert(current < space);

            int num_dbls = split[orders[current-1]];
            theta_double_iter_pair(
                strat_theta_pair + current * 2*NUM_LIMBS, 
                strat_theta_pair + (current-1) * 2*NUM_LIMBS, 
                c1, c2, num_dbls
            );
            orders[current] = orders[current-1] - num_dbls;
        }

        if (step == n - 2) {
            h1 = true; h2 = false;
            codom_verify = false;
        } else if (step == n - 3) {
            h1 = false; h2 = false;
        } else {
            h1 = false; h2 = true;
        }
        
        // Compute the codomain of the kernel
        success &= theta_codomain_pair(
            OA, OA_dual_inv, strat_theta_pair + current * 2*NUM_LIMBS, 
            h1, h2, codom_verify
        );

        // Push the kernel through the images
        // we have current number of pairs, so 2*current number of points
        theta_eval_list_pair(strat_theta_pair, OA_dual_inv, h1, h2, 2*current);
        for (int i = 0; i < current; i++) {
            orders[i]--;
        }
        current--;

        // Eval points
        theta_eval_list_pair(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);
        step++;
    }

    // Step Three: Splitting
    success &= splitting_isom_matrix(&isom_matrix, OA, verify);
    splitting_apply_isomorphism_single(OA, OA, &isom_matrix);
    splitting_apply_isomorphism_list_pair(eval_pts_theta, &isom_matrix, num_pts);

    if (randomize) {
        splitting_rand_matrix(&rand_matrix);
        splitting_apply_isomorphism_single(OA, OA, &rand_matrix);
        splitting_apply_isomorphism_list_pair(eval_pts_theta, &rand_matrix, num_pts);
    }

    success &= splitting_theta_to_product(res_A_34, OA, verify);
    success &= splitting_theta_point_to_mont_list_pair(res_pts_34, eval_pts_theta, OA, num_pts, verify);

    return success;
}

/**
 * As theta_isogeny_chain_vec, with the first step computed from x-only points
 * as in the SQIsign v3 reference: the kernel is doubled with a ladder which
 * keeps [2^m - 1]G alongside [2^m]G, and every stored point is pushed through
 * the gluing with the barycentric coordinates of ([2^m]G, G). No Montgomery
 * A-coefficient and no square root are needed.
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
)
{
    __m512i A24_12[NUM_LIMBS], OA[NUM_LIMBS], OA_dual_inv[2*NUM_LIMBS];
    __m512i J[NUM_LIMBS], J_pair[2*NUM_LIMBS], matrix01[2*NUM_LIMBS], matrix23[2*NUM_LIMBS];
    __m512i gluing_eval_res[2*NUM_LIMBS];
    __m512i eval_pts_theta[(num_pts + 1) * NUM_LIMBS]; // + 1 as verification asks for none
    __m512i xD[NUM_LIMBS], zD[NUM_LIMBS], Dp[NUM_LIMBS], Dm[NUM_LIMBS], one[NUM_LIMBS];
    __m512i t0[N4x2], t1[N4x2], u0[NUM_LIMBS], u1[NUM_LIMBS], idx;
    __m512i c1[2*NUM_LIMBS], c2[2*NUM_LIMBS];
    splitting_matrix_t isom_matrix, rand_matrix;
    bool h1, h2, success = true;

    // A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)
    idx = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(idx, normA24_1[i]);
        t1[i] = VPERM512(idx, normA24_2[i]);
    }
    fp2_2x2x2_to_4x2x1(A24_12, t0, t1);

    // The four generators are the differences of the four ladders, and are
    // normalized together. The basis point x(P - Q) is never needed.
    fp2_2x2x2_to_4x2x1(u0, ker_basis1, ker_basis1 + N4x2); // (P1.X, P1.Z, Q1.X, Q1.Z)
    fp2_2x2x2_to_4x2x1(u1, ker_basis2, ker_basis2 + N4x2); // (P2.X, P2.Z, Q2.X, Q2.Z)
    idx = VSET(0, 1, 4, 5, 8, 9, 12, 13);
    for (int i = 0; i < NUM_LIMBS; i++) {
        xD[i] = VBLENDPERM(idx, u0[i], u1[i]);
    }
    idx = VSET(2, 3, 6, 7, 10, 11, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        zD[i] = VBLENDPERM(idx, u0[i], u1[i]);
    }
    fp2_inv_4x2x1(zD, zD);
    fp2_mul_4x2x1(xD, xD, zD);

    fp2_setzero_4x2x1(one);
    one[0] = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    fp2_tomont_4x2x1(one, one);
    fp2_add_4x2x1(Dp, xD, one);
    fp2_sub_4x2x1(Dm, xD, one);

    // Setting up the strategy
    const uint16_t *split = theta_strategy();
    int space = strategy_space(split, n);

    uint16_t orders[space];
    __m512i strat_xe[space*NUM_LIMBS], strat_ze[space*NUM_LIMBS];
    __m512i strat_xo[space*NUM_LIMBS], strat_zo[space*NUM_LIMBS];

    // The ladder starts at ([1]G, [0]G), the odd point being the identity
    orders[0] = n;
    fp2_copy_4x2x1(strat_xe, xD);
    fp2_copy_4x2x1(strat_ze, one);
    fp2_copy_4x2x1(strat_xo, one);
    fp2_setzero_4x2x1(strat_zo);

    // Step One: ladder the kernel, storing the intermediate results
    int current = 0;
    while (orders[current] != 1) {
        current++;
        assert(current < space);

        int num_dbls = orders[current-1] >= 16 ? orders[current-1] / 2 : orders[current-1] - 1;
        ec_gluing_ladder_4x4x2x1(
            strat_xe + current*NUM_LIMBS, strat_ze + current*NUM_LIMBS,
            strat_xo + current*NUM_LIMBS, strat_zo + current*NUM_LIMBS, A24_12,
            strat_xe + (current-1)*NUM_LIMBS, strat_ze + (current-1)*NUM_LIMBS,
            strat_xo + (current-1)*NUM_LIMBS, strat_zo + (current-1)*NUM_LIMBS,
            xD, num_dbls
        );
        orders[current] = orders[current-1] - num_dbls;
    }

    success &= gluing_codomain_x_only(
        OA, OA_dual_inv, J, J_pair, matrix01, matrix23, A24_12,
        strat_xe + current*NUM_LIMBS, strat_ze + current*NUM_LIMBS, verify
    );

    __m512i strat_theta_pair[space * 2*NUM_LIMBS];

    // The generators are translated by the eight torsion above them, which is
    // the only point whose image is known, and everything else by the
    // generators themselves.
    gluing_eval_pair_bary(
        gluing_eval_res,
        strat_xe + current*NUM_LIMBS, strat_ze + current*NUM_LIMBS,
        strat_xo + current*NUM_LIMBS, strat_zo + current*NUM_LIMBS,
        Dp, Dm, J_pair, matrix01, matrix23, false
    );
    theta_invert_pair(J_pair, gluing_eval_res);
    theta_had_pair(strat_theta_pair, gluing_eval_res);
    orders[0]--;

    // Push the rest of the kernel through the gluing isogeny
    for (int i = 1; i < current; i++) {
        gluing_eval_pair_bary(
            strat_theta_pair + i*2*NUM_LIMBS,
            strat_xe + i*NUM_LIMBS, strat_ze + i*NUM_LIMBS,
            strat_xo + i*NUM_LIMBS, strat_zo + i*NUM_LIMBS,
            Dp, Dm, J_pair, matrix01, matrix23, true
        );
        orders[i]--;
    }
    current--;

    success &= gluing_eval_special_list_pair(
        eval_pts_theta, pts, OA_dual_inv, 
        matrix01, matrix23, num_pts, verify
    );

    // Step Two: Compute the rest of the chain
    int step = 0;
    bool codom_verify = verify;
    while (current >= 0 && orders[current] != 0) {
        // Precompute constants for doubling, unused when this step has none
        if (orders[current] != 1) {
            theta_precomp_pair(c1, c2, OA);
        }

        while (orders[current] != 1) {
            current++;
            assert(current < space);

            int num_dbls = split[orders[current-1]];
            theta_double_iter_pair(
                strat_theta_pair + current * 2*NUM_LIMBS, 
                strat_theta_pair + (current-1) * 2*NUM_LIMBS, 
                c1, c2, num_dbls
            );
            orders[current] = orders[current-1] - num_dbls;
        }

        if (step == n - 2) {
            h1 = true; h2 = false;
            codom_verify = false;
        } else if (step == n - 3) {
            h1 = false; h2 = false;
        } else {
            h1 = false; h2 = true;
        }
        
        // Compute the codomain of the kernel
        success &= theta_codomain_pair(
            OA, OA_dual_inv, strat_theta_pair + current * 2*NUM_LIMBS, 
            h1, h2, codom_verify
        );

        // Push the kernel through the images
        // we have current number of pairs, so 2*current number of points
        theta_eval_list_pair(strat_theta_pair, OA_dual_inv, h1, h2, 2*current);
        for (int i = 0; i < current; i++) {
            orders[i]--;
        }
        current--;

        // Eval points
        theta_eval_list_pair(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);
        step++;
    }

    // Step Three: Splitting
    success &= splitting_isom_matrix(&isom_matrix, OA, verify);
    splitting_apply_isomorphism_single(OA, OA, &isom_matrix);
    splitting_apply_isomorphism_list_pair(eval_pts_theta, &isom_matrix, num_pts);

    if (randomize) {
        splitting_rand_matrix(&rand_matrix);
        splitting_apply_isomorphism_single(OA, OA, &rand_matrix);
        splitting_apply_isomorphism_list_pair(eval_pts_theta, &rand_matrix, num_pts);
    }

    success &= splitting_theta_to_product(res_A_34, OA, verify);
    success &= splitting_theta_point_to_mont_list_pair(res_pts_34, eval_pts_theta, OA, num_pts, verify);

    return success;
}

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
)
{
    fp2_t unvec[4];
    __m512i basis1[3*N4x2], basis2[3*N4x2], normA_1[N4x2], normA_2[N4x2], normA24_1[N4x2], normA24_2[N4x2];
    __m512i A_34[NUM_LIMBS], eval_pts_mont[num_pts * NUM_LIMBS];
    
    ec_point_t AC;
    ec_point_unvec_to_1x2x2x2(basis1, &ker->T1.P1);
    ec_point_unvec_to_1x2x2x2(basis1 + N4x2, &ker->T2.P1);
    ec_point_unvec_to_1x2x2x2(basis1 + 2*N4x2, &ker->T1m2.P1);
    ec_point_unvec_to_1x2x2x2(basis2, &ker->T1.P2);
    ec_point_unvec_to_1x2x2x2(basis2 + N4x2, &ker->T2.P2);
    ec_point_unvec_to_1x2x2x2(basis2 + 2*N4x2, &ker->T1m2.P2);

    fp2_copy(&AC.x, &E12->E1.A);
    fp2_set_one(&AC.z);
    ec_point_unvec_to_1x2x2x2(normA_1, &AC);
    fp2_copy(&AC.x, &E12->E2.A);
    fp2_set_one(&AC.z);
    ec_point_unvec_to_1x2x2x2(normA_2, &AC);
    ec_point_unvec_to_1x2x2x2(normA24_1, &E12->E1.A24);
    ec_point_unvec_to_1x2x2x2(normA24_2, &E12->E2.A24);

    for (int i = 0; i < num_pts; i++) {
        fp2_copy(unvec, &(pts + i)->P1.x);
        fp2_copy(unvec+1, &(pts + i)->P1.z);
        fp2_copy(unvec+2, &(pts + i)->P2.x);
        fp2_copy(unvec+3, &(pts + i)->P2.z);
        fp2_unvectorized_to_4x2x1(eval_pts_mont + i*NUM_LIMBS, unvec);
    }

    bool ret = theta_isogeny_chain_vec(
        A_34, eval_pts_mont, 
        basis1, basis2, normA_1, normA_2, normA24_1, normA24_2, eval_pts_mont, 
        n, num_pts, randomize, verify
    );

    fp2_4x2x1_to_unvectorized(unvec, A_34);
    ec_curve_init(&res_E34->E1);
    ec_curve_init(&res_E34->E2);
    res_E34->E1.A = unvec[0];
    res_E34->E1.C = unvec[1];
    res_E34->E2.A = unvec[2];
    res_E34->E2.C = unvec[3];

    for (int i = 0; i < num_pts; i++) {
        fp2_4x2x1_to_unvectorized(unvec, eval_pts_mont + i*NUM_LIMBS);
        fp2_copy(&(res_pts+i)->P1.x, unvec);
        fp2_copy(&(res_pts+i)->P1.z, unvec+1);
        fp2_copy(&(res_pts+i)->P2.x, unvec+2);
        fp2_copy(&(res_pts+i)->P2.z, unvec+3);
    }

    return ret;
}

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
)
{
    fp2_t unvec[4];
    __m512i basis1[3*N4x2], basis2[3*N4x2], normA24_1[N4x2], normA24_2[N4x2];
    __m512i A_34[NUM_LIMBS], eval_pts_mont[num_pts * NUM_LIMBS];
    
    ec_point_unvec_to_1x2x2x2(basis1, &ker->T1.P1);
    ec_point_unvec_to_1x2x2x2(basis1 + N4x2, &ker->T2.P1);
    ec_point_unvec_to_1x2x2x2(basis1 + 2*N4x2, &ker->T1m2.P1);
    ec_point_unvec_to_1x2x2x2(basis2, &ker->T1.P2);
    ec_point_unvec_to_1x2x2x2(basis2 + N4x2, &ker->T2.P2);
    ec_point_unvec_to_1x2x2x2(basis2 + 2*N4x2, &ker->T1m2.P2);

    ec_point_unvec_to_1x2x2x2(normA24_1, &E12->E1.A24);
    ec_point_unvec_to_1x2x2x2(normA24_2, &E12->E2.A24);

    for (int i = 0; i < num_pts; i++) {
        fp2_copy(unvec, &(pts + i)->P1.x);
        fp2_copy(unvec+1, &(pts + i)->P1.z);
        fp2_copy(unvec+2, &(pts + i)->P2.x);
        fp2_copy(unvec+3, &(pts + i)->P2.z);
        fp2_unvectorized_to_4x2x1(eval_pts_mont + i*NUM_LIMBS, unvec);
    }

    bool ret = theta_isogeny_chain_bary_vec(
        A_34, eval_pts_mont, 
        basis1, basis2, normA24_1, normA24_2, eval_pts_mont, 
        n, num_pts, randomize, verify
    );

    fp2_4x2x1_to_unvectorized(unvec, A_34);
    ec_curve_init(&res_E34->E1);
    ec_curve_init(&res_E34->E2);
    res_E34->E1.A = unvec[0];
    res_E34->E1.C = unvec[1];
    res_E34->E2.A = unvec[2];
    res_E34->E2.C = unvec[3];

    for (int i = 0; i < num_pts; i++) {
        fp2_4x2x1_to_unvectorized(unvec, eval_pts_mont + i*NUM_LIMBS);
        fp2_copy(&(res_pts+i)->P1.x, unvec);
        fp2_copy(&(res_pts+i)->P1.z, unvec+1);
        fp2_copy(&(res_pts+i)->P2.x, unvec+2);
        fp2_copy(&(res_pts+i)->P2.z, unvec+3);
    }

    return ret;
}

// ====================================================================
//  Isogeny chains where we have the 2^n torsion only (requires sqrts)   
// ====================================================================

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
)
{
    __m512i A_12[NUM_LIMBS], A24_12[NUM_LIMBS], ker_x[NUM_LIMBS], ker_y[NUM_LIMBS], ker_z[NUM_LIMBS], OA[NUM_LIMBS], OA_dual_inv[2*NUM_LIMBS];
    __m512i J[NUM_LIMBS], matrix01[2*NUM_LIMBS], matrix23[2*NUM_LIMBS], T8_x[NUM_LIMBS], T8_y[NUM_LIMBS], T8_z[NUM_LIMBS];
    __m512i eval_pts_theta[num_pts * NUM_LIMBS];
    __m512i t0[N4x2], t1[N4x2], idx;
    __m512i c1[2*NUM_LIMBS], c2[2*NUM_LIMBS];
    splitting_matrix_t isom_matrix, rand_matrix;
    bool h1, h2, success = true;

    // Lift each basis x(P), x(Q), x(P-Q) to P, Q in Jacobian coordinates.
    // We pack the coordinates into three blobs
    // ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X), etc.
    success &= lift_basis_pair_normalized_2x2x2(ker_x, ker_y, ker_z, ker_basis1, ker_basis2, normA_1, normA_2, verify);
    
    // Packing curve coefficients
    // A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    idx = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(idx, normA_1[i]);
        t1[i] = VPERM512(idx, normA_2[i]);
    }
    fp2_2x2x2_to_4x2x1(A_12, t0, t1);
    // A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(idx, normA24_1[i]);
        t1[i] = VPERM512(idx, normA24_2[i]);
    }
    fp2_2x2x2_to_4x2x1(A24_12, t0, t1);

    // Setting up the strategy
    const uint16_t *split = theta_strategy();
    int space = strategy_space(split, n);

    uint16_t orders[space];
    __m512i strat_points_x[space*NUM_LIMBS], strat_points_y[space*NUM_LIMBS], strat_points_z[space*NUM_LIMBS];
    // TODO: ker_x, ker_y, ker_z never used again, remove copies.
    orders[0] = n-2;
    fp2_copy_4x2x1(strat_points_x, ker_x);
    fp2_copy_4x2x1(strat_points_y, ker_y);
    fp2_copy_4x2x1(strat_points_z, ker_z);

    int current = theta_compute_product_doubles(
        A_12, strat_points_x, strat_points_y, strat_points_z, orders, space
    );

    success &= gluing_codomain(
        OA, OA_dual_inv, J, matrix01, matrix23, A24_12, 
        strat_points_x + current*NUM_LIMBS, strat_points_z + current*NUM_LIMBS,
        verify
    );

    // Package the point we use for translation
    // T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    // T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    // T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    idx = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        int j = current*NUM_LIMBS + i;
        T8_x[i] = VPERM512(idx, strat_points_x[j]);
        T8_y[i] = VPERM512(idx, strat_points_y[j]);
        T8_z[i] = VPERM512(idx, strat_points_z[j]);
    }

    // Push kernel though the gluing isogeny
    // 4 Jacobian points for the kernel become 2 theta points
    // TODO: Is current = space-1 here? And even if it is not, is it better to use 8*space?
    __m512i strat_theta_pair[space * 2*NUM_LIMBS];
    for (int i = 0; i < current; i++) {
        gluing_eval_pair(
            strat_theta_pair + i*2*NUM_LIMBS, //8x1x1
            A_12, strat_points_x + i*NUM_LIMBS, strat_points_y + i*NUM_LIMBS, strat_points_z + i*NUM_LIMBS,
            T8_x, T8_y, T8_z, J, matrix01, matrix23
        );
        orders[i]--;
    }
    current--;

    success &= gluing_eval_special_list_pair(
        eval_pts_theta, pts, OA_dual_inv, 
        matrix01, matrix23, num_pts, verify
    );

    // Step Two: Compute the rest of the chain
    h1 = false; h2 = true;
    while (current >= 0 && orders[current] != 0) {
        // Precompute constants for doubling, unused when this step has none
        if (orders[current] != 1) {
            theta_precomp_pair(c1, c2, OA);
        }

        while (orders[current] != 1) {
            current++;
            assert(current < space);

            int num_dbls = split[orders[current-1]];
            theta_double_iter_pair(
                strat_theta_pair + current*(2*NUM_LIMBS), 
                strat_theta_pair + (current-1)*(2*NUM_LIMBS), 
                c1, c2, num_dbls
            );
            orders[current] = orders[current-1] - num_dbls;
        }

        // Compute the codomain of the kernel
        success &= theta_codomain_pair(
            OA, OA_dual_inv, strat_theta_pair + current*(2*NUM_LIMBS), 
            h1, h2, verify
        );

        // Push the kernel through the images
        // we have current number of pairs, so 2*current number of points
        theta_eval_list_pair(strat_theta_pair, OA_dual_inv, h1, h2, 2*current);
        for (int i = 0; i < current; i++) {
            orders[i]--;
        }
        current--;

        // Eval points
        theta_eval_list_pair(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);
    }

    // Step Three:
    // we now do the last two steps outside of the strategy. As we do not
    // have the torsion above we require square roots to compute the last
    // two codomains.
    
    //TODO: We can save 30 cycles by using theta_eval_pair_old here, as we only need OA_inv in 4x2x1.
    fp2_8x1x1_to_4x2x1(strat_theta_pair, strat_theta_pair + NUM_LIMBS, strat_theta_pair);
    if (n > 3) {
        fp2_8x1x1_to_first_4x2x1(OA_dual_inv, OA_dual_inv);
        theta_eval_single(strat_theta_pair, strat_theta_pair, OA_dual_inv, h1, h2);
    }

    h1 = false; h2 = false;
    theta_codomain_with_four_torsion(OA_dual_inv, OA, strat_theta_pair, h1, h2);
    // In order to use pair eval
    fp2_4x2x1_duplicate_to_8x1x1(OA_dual_inv, OA_dual_inv);
    theta_eval_list_pair(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);

    h1 = true; h2 = false;
    theta_codomain_with_no_torsion(OA_dual_inv, OA, h1, h2);
    // In order to use pair eval
    fp2_4x2x1_duplicate_to_8x1x1(OA_dual_inv, OA_dual_inv);
    theta_eval_list_pair(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);


    // Step Four: Splitting
    success &= splitting_isom_matrix(&isom_matrix, OA, verify);
    splitting_apply_isomorphism_single(OA, OA, &isom_matrix);
    splitting_apply_isomorphism_list_pair(eval_pts_theta, &isom_matrix, num_pts);

    if (randomize) {
        splitting_rand_matrix(&rand_matrix);
        splitting_apply_isomorphism_single(OA, OA, &rand_matrix);
        splitting_apply_isomorphism_list_pair(eval_pts_theta, &rand_matrix, num_pts);
    }

    success &= splitting_theta_to_product(res_A_34, OA, verify);
    success &= splitting_theta_point_to_mont_list_pair(res_pts_34, eval_pts_theta, OA, num_pts, verify);

    return success;
}

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
)
{
    fp2_t unvec[4];
    __m512i basis1[3*N4x2], basis2[3*N4x2], normA_1[N4x2], normA_2[N4x2], normA24_1[N4x2], normA24_2[N4x2];
    __m512i A_34[NUM_LIMBS], eval_pts_mont[num_pts * NUM_LIMBS];
    
    ec_point_t AC;
    ec_point_unvec_to_1x2x2x2(basis1, &ker->T1.P1);
    ec_point_unvec_to_1x2x2x2(basis1 + N4x2, &ker->T2.P1);
    ec_point_unvec_to_1x2x2x2(basis1 + 2*N4x2, &ker->T1m2.P1);
    ec_point_unvec_to_1x2x2x2(basis2, &ker->T1.P2);
    ec_point_unvec_to_1x2x2x2(basis2 + N4x2, &ker->T2.P2);
    ec_point_unvec_to_1x2x2x2(basis2 + 2*N4x2, &ker->T1m2.P2);

    fp2_copy(&AC.x, &E12->E1.A);
    fp2_set_one(&AC.z);
    ec_point_unvec_to_1x2x2x2(normA_1, &AC);
    fp2_copy(&AC.x, &E12->E2.A);
    fp2_set_one(&AC.z);
    ec_point_unvec_to_1x2x2x2(normA_2, &AC);
    ec_point_unvec_to_1x2x2x2(normA24_1, &E12->E1.A24);
    ec_point_unvec_to_1x2x2x2(normA24_2, &E12->E2.A24);

    for (int i = 0; i < num_pts; i++) {
        fp2_copy(unvec, &(pts + i)->P1.x);
        fp2_copy(unvec+1, &(pts + i)->P1.z);
        fp2_copy(unvec+2, &(pts + i)->P2.x);
        fp2_copy(unvec+3, &(pts + i)->P2.z);
        fp2_unvectorized_to_4x2x1(eval_pts_mont + i*NUM_LIMBS, unvec);
    }

    bool ret = theta_isogeny_chain_no_torsion_vec(
        A_34, eval_pts_mont, 
        basis1, basis2, normA_1, normA_2, normA24_1, normA24_2, eval_pts_mont, 
        n, num_pts, randomize, verify
    );

    fp2_4x2x1_to_unvectorized(unvec, A_34);
    ec_curve_init(&res_E34->E1);
    ec_curve_init(&res_E34->E2);
    res_E34->E1.A = unvec[0];
    res_E34->E1.C = unvec[1];
    res_E34->E2.A = unvec[2];
    res_E34->E2.C = unvec[3];

    for (int i = 0; i < num_pts; i++) {
        fp2_4x2x1_to_unvectorized(unvec, eval_pts_mont + i*NUM_LIMBS);
        fp2_copy(&(res_pts+i)->P1.x, unvec);
        fp2_copy(&(res_pts+i)->P1.z, unvec+1);
        fp2_copy(&(res_pts+i)->P2.x, unvec+2);
        fp2_copy(&(res_pts+i)->P2.z, unvec+3);
    }

    return ret;
}
