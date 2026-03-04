#include <stdio.h> //TODO: Remove

#include "ec.h"
#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "theta.h"
#include "theta_gluing.h"
#include "theta_middle.h"
#include "theta_splitting.h"
#include "utils.h"
#include "sqi_test_extras.h"

// ====================================================================
//                 Helper functions for isogeny chains    
// ====================================================================

// TODO: Do I need to normalize points?
// TODO: Should we normalize eval points?
void theta_normalize_chain_input(
    theta_couple_curve_t* E12, theta_kernel_couple_points_t* ker
)
{
    //TODO batched inv
    ec_normalize_curve(&E12->E1);
    ec_normalize_curve(&E12->E2);
    // ec_normalize_point(&B1->P);
    // ec_normalize_point(&B1->Q);
    // ec_normalize_point(&B1->PmQ);
    // ec_normalize_point(&B2->P);
    // ec_normalize_point(&B2->Q);
    // ec_normalize_point(&B2->PmQ);
}

int theta_compute_product_doubles(
    __m512i* A_12, __m512i* strat_points_x, __m512i* strat_points_y, __m512i* strat_points_z,
    uint16_t* orders, int space
)
{
    __m512i A12_over_3[8], strat_points_t[8*space];
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
            strat_points_x + (8*k), strat_points_y + (8*k), 
            strat_points_z + (8*k), strat_points_t + (8*k),
            strat_points_x + (8*(k-1)), strat_points_y + (8*(k-1)),
            strat_points_z + (8*(k-1)), strat_points_t + (8*(k-1)),
            n
        );
        orders[k] = orders[k - 1] - n;
    }

    // Now we must convert the Weierstrass model back to the Montgomery model
    // Which is done only on the points we have doubled so far.
    for (int i = 0; i <= k; i++)
    {
        ec_ws_to_jac_4x4x2x1(
            strat_points_x + (8*i), strat_points_y + (8*i), strat_points_z + (8*i),
            strat_points_x + (8*i), strat_points_y + (8*i), strat_points_z + (8*i),
            A12_over_3
        );
    }

    return k;
}

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
)
{
    __m512i A_12[8], A24_12[8], ker_x[8], ker_y[8], ker_z[8], OA[8], OA_dual_inv[8];
    __m512i J[8], matrix01[16], matrix23[16], T8_x[8], T8_y[8], T8_z[8], gluing_eval_res[16];
    __m512i eval_pts_theta[8*num_pts], idx;

    // TODO: Copied from middle. Which variables can be merged?
    bool h1, h2, success = true;
    splitting_matrix_t isom_matrix, rand_matrix;
    __m512i c1[8], c2[8], p1[8], p2[8], k1[8], k2[8];

    fp2_t unvec1[4];

    // TODO: Is it enough to just normalize curves? 
    theta_normalize_chain_input(E12, ker);   

    // lift the basis
    theta_couple_jac_point_t xyT1, xyT2;
    ec_basis_t bas1 = { .P = ker->T1.P1, .Q = ker->T2.P1, .PmQ = ker->T1m2.P1 };
    ec_basis_t bas2 = { .P = ker->T1.P2, .Q = ker->T2.P2, .PmQ = ker->T1m2.P2 };

    success &= lift_basis(&xyT1.P1, &xyT2.P1, &bas1, &E12->E1, verify);
    success &= lift_basis(&xyT1.P2, &xyT2.P2, &bas2, &E12->E2, verify);

    // We are now in the position to convert from 64 to 51 bit limbs
    // For the curve we need to pack the Montgomery coefficient for
    // arithmetic.
    // A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    unvec1[0] = E12->E1.A; unvec1[1] = E12->E1.A; unvec1[2] = E12->E2.A; unvec1[3] = E12->E2.A;
    fp2_unvectorized_to_4x2x1(A_12, unvec1);
    // A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)
    unvec1[0] = E12->E1.A24.x; unvec1[1] = E12->E1.A24.x; unvec1[2] = E12->E2.A24.x; unvec1[3] = E12->E2.A24.x;
    fp2_unvectorized_to_4x2x1(A24_12, unvec1);

    // For the Jacobian points, we pack the coordinates into three blobs
    // ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X)
    unvec1[0] = xyT1.P1.x; unvec1[1] = xyT2.P1.x; unvec1[2] = xyT1.P2.x; unvec1[3] = xyT2.P2.x;
    fp2_unvectorized_to_4x2x1(ker_x, unvec1);
    // ker_Y = AVX_4x2x1(P1.Y, Q1.Y, P2.Y, Q2.Y)
    unvec1[0] = xyT1.P1.y; unvec1[1] = xyT2.P1.y; unvec1[2] = xyT1.P2.y; unvec1[3] = xyT2.P2.y;
    fp2_unvectorized_to_4x2x1(ker_y, unvec1);
    // ker_Z = AVX_4x2x1(P1.Z, Q1.Z, P2.Z, Q2.Z)
    unvec1[0] = xyT1.P1.z; unvec1[1] = xyT2.P1.z; unvec1[2] = xyT1.P2.z; unvec1[3] = xyT2.P2.z;
    fp2_unvectorized_to_4x2x1(ker_z, unvec1);

    // Setting up the strategy
    int space = 1;
    for (unsigned i = 1; i < n; i *= 2) // n.bit_length + 1
        ++space;

    uint16_t orders[space];
    __m512i strat_points_x[8*space], strat_points_y[8*space], strat_points_z[8*space];
    // TODO: ker_x, ker_y, ker_z never used again, remove copies.
    orders[0] = n;
    fp2_copy_4x2x1(strat_points_x, ker_x);
    fp2_copy_4x2x1(strat_points_y, ker_y);
    fp2_copy_4x2x1(strat_points_z, ker_z);

    int current = theta_compute_product_doubles(
        A_12, strat_points_x, strat_points_y, strat_points_z, orders, space
    );

    success &= gluing_codomain(
        OA, OA_dual_inv, J, matrix01, matrix23,
        A24_12, strat_points_x + (8*current), strat_points_z + (8*current),
        verify
    );

    // Package the point we use for translation
    // T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    // T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    // T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    idx = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    for (int i = 0; i <  8; i++) {
        int j = 8*current + i;
        T8_x[i] = VPERM512(idx, strat_points_x[j]);
        T8_y[i] = VPERM512(idx, strat_points_y[j]);
        T8_z[i] = VPERM512(idx, strat_points_z[j]);
    }

    // Push kernel though the gluing isogeny
    // 4 Jacobian points for the kernel become 2 theta points
    // TODO: Is current = space-1 here? And even if it is not, is it better to use 8*space?
    __m512i strat_theta1[8*space], strat_theta2[8*space];
    for (int i = 0; i < current; i++) {
        gluing_eval_pair(
            gluing_eval_res, //8x1x1
            A_12, strat_points_x + (8*i), strat_points_y + (8*i), strat_points_z + (8*i),
            T8_x, T8_y, T8_z, J, matrix01, matrix23
        );
        fp2_8x1x1_to_4x2x1( // Since we have a 4x2x1 chain and we don't have gluing_eval_single
            strat_theta1 + (8*i), strat_theta2 + (8*i), gluing_eval_res
        );
        orders[i]--;
    }
    current--;

    // Push other points through gluing isogeny
    for (int i = 0; i < num_pts; i++) {
        success &= gluing_eval_special(
            eval_pts_theta + (8*i),
            pts + i, OA_dual_inv, matrix01, matrix23, verify
        );
    }

    // Step Two: Compute the rest of the chain
    int step = 0;
    while (current >= 0 && orders[current] != 0) {
        // Precompute constants for doubling
        theta_precomp_single(c1, c2, OA);
        
        while (orders[current] != 1) {
            current++;

            int num_dbls = orders[current-1] / 2;
            theta_double_iter_single(
                strat_theta1 + (8*current), 
                strat_theta1 + (8*(current-1)), 
                c1, c2, num_dbls
            );
            theta_double_iter_single(
                strat_theta2 + (8*current), 
                strat_theta2 + (8*(current-1)), 
                c1, c2, num_dbls
            );
            orders[current] = orders[current-1] - num_dbls;
        }

        if (step == n - 1) {
            h1 = true; h2 = false;
        } else if (step == n - 2) {
            h1 = false; h2 = false;
        } else {
            h1 = false; h2 = true;
        }

        // Compute the codomain of the kernel
        theta_codomain_single(
            OA, OA_dual_inv, 
            strat_theta1 + (8*current), strat_theta2 + (8*current),
            h1, h2
        );

        // Push the kernel through the images
        theta_eval_list_single(strat_theta1, OA_dual_inv, h1, h2, current);
        theta_eval_list_single(strat_theta2, OA_dual_inv, h1, h2, current);
        for (int i = 0; i < current; i++) {
            orders[i]--;
        }
        current--;

        // Eval points
        theta_eval_list_single(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);
        step++;
    }

    // Step Three: Splitting
    success &= splitting_isom_matrix(&isom_matrix, OA, verify);
    splitting_apply_isomorphism_single(OA, OA, &isom_matrix);
    splitting_apply_isomorphism_list_single(eval_pts_theta, &isom_matrix, num_pts);

    if (randomize) {
        splitting_rand_matrix(&rand_matrix);
        splitting_apply_isomorphism_single(OA, OA, &rand_matrix);
        splitting_apply_isomorphism_list_single(eval_pts_theta, &rand_matrix, num_pts);
    }

    success &= splitting_theta_to_product(res_E34, OA, verify);
    success &= splitting_theta_point_to_mont_list_single(res_pts, eval_pts_theta, OA, num_pts, verify);

    return success;
}

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
)
{
    __m512i A_12[8], A24_12[8], ker_x[8], ker_y[8], ker_z[8], OA[8], OA_dual_inv[16];
    __m512i J[8], matrix01[16], matrix23[16], T8_x[8], T8_y[8], T8_z[8], gluing_eval_res[16];
    __m512i eval_pts_theta[8*num_pts], idx;

    // TODO: Copied from middle. Which variables can be merged?
    bool h1, h2, success = true;
    splitting_matrix_t isom_matrix, rand_matrix;
    __m512i c1[16], c2[16], p1[8], p2[8], k1[8], k2[8];

    fp2_t unvec1[4];

    // TODO: Is it enough to just normalize curves? 
    theta_normalize_chain_input(E12, ker);   

    // lift the basis
    theta_couple_jac_point_t xyT1, xyT2;
    ec_basis_t bas1 = { .P = ker->T1.P1, .Q = ker->T2.P1, .PmQ = ker->T1m2.P1 };
    ec_basis_t bas2 = { .P = ker->T1.P2, .Q = ker->T2.P2, .PmQ = ker->T1m2.P2 };

    success &= lift_basis(&xyT1.P1, &xyT2.P1, &bas1, &E12->E1, verify);
    success &= lift_basis(&xyT1.P2, &xyT2.P2, &bas2, &E12->E2, verify);
    // printf("lift %d\n", success);
    // We are now in the position to convert from 64 to 51 bit limbs
    // For the curve we need to pack the Montgomery coefficient for
    // arithmetic.
    // A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    unvec1[0] = E12->E1.A; unvec1[1] = E12->E1.A; unvec1[2] = E12->E2.A; unvec1[3] = E12->E2.A;
    fp2_unvectorized_to_4x2x1(A_12, unvec1);
    // A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)
    unvec1[0] = E12->E1.A24.x; unvec1[1] = E12->E1.A24.x; unvec1[2] = E12->E2.A24.x; unvec1[3] = E12->E2.A24.x;
    fp2_unvectorized_to_4x2x1(A24_12, unvec1);

    // For the Jacobian points, we pack the coordinates into three blobs
    // ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X)
    unvec1[0] = xyT1.P1.x; unvec1[1] = xyT2.P1.x; unvec1[2] = xyT1.P2.x; unvec1[3] = xyT2.P2.x;
    fp2_unvectorized_to_4x2x1(ker_x, unvec1);
    // ker_Y = AVX_4x2x1(P1.Y, Q1.Y, P2.Y, Q2.Y)
    unvec1[0] = xyT1.P1.y; unvec1[1] = xyT2.P1.y; unvec1[2] = xyT1.P2.y; unvec1[3] = xyT2.P2.y;
    fp2_unvectorized_to_4x2x1(ker_y, unvec1);
    // ker_Z = AVX_4x2x1(P1.Z, Q1.Z, P2.Z, Q2.Z)
    unvec1[0] = xyT1.P1.z; unvec1[1] = xyT2.P1.z; unvec1[2] = xyT1.P2.z; unvec1[3] = xyT2.P2.z;
    fp2_unvectorized_to_4x2x1(ker_z, unvec1);

    // Setting up the strategy
    int space = 1;
    for (unsigned i = 1; i < n; i *= 2) // n.bit_length + 1
        ++space;

    uint16_t orders[space];
    __m512i strat_points_x[8*space], strat_points_y[8*space], strat_points_z[8*space];
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
        strat_points_x + (8*current), strat_points_z + (8*current),
        verify
    );
    // printf("gluing_codom %d\n", success);

    // Package the point we use for translation
    // T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    // T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    // T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    idx = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    for (int i = 0; i <  8; i++) {
        int j = 8*current + i;
        T8_x[i] = VPERM512(idx, strat_points_x[j]);
        T8_y[i] = VPERM512(idx, strat_points_y[j]);
        T8_z[i] = VPERM512(idx, strat_points_z[j]);
    }

    // Push kernel though the gluing isogeny
    // 4 Jacobian points for the kernel become 2 theta points
    // TODO: Is current = space-1 here? And even if it is not, is it better to use 8*space?
    __m512i strat_theta_pair[16*space];
    for (int i = 0; i < current; i++) {
        gluing_eval_pair(
            strat_theta_pair + 16*i, //8x1x1
            A_12, strat_points_x + (8*i), strat_points_y + (8*i), strat_points_z + (8*i),
            T8_x, T8_y, T8_z, J, matrix01, matrix23
        );
        orders[i]--;
    }
    current--;

    success &= gluing_eval_special_list_pair(
        eval_pts_theta, pts, OA_dual_inv, 
        matrix01, matrix23, num_pts, verify
    );
    // printf("gluing eval %d\n", success);

    // Step Two: Compute the rest of the chain
    int step = 0;
    bool codom_verify = verify;
    while (current >= 0 && orders[current] != 0) {
        // Precompute constants for doubling
        theta_precomp_pair(c1, c2, OA);
        
        while (orders[current] != 1) {
            current++;

            int num_dbls = orders[current-1] / 2;
            theta_double_iter_pair(
                strat_theta_pair + (16*current), 
                strat_theta_pair + (16*(current-1)), 
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
            OA, OA_dual_inv, strat_theta_pair + (16*current), 
            h1, h2, codom_verify
        );
        // printf("%d-th codomain %d\n", step, success);

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

    success &= splitting_theta_to_product(res_E34, OA, verify);
    success &= splitting_theta_point_to_mont_list_pair(res_pts, eval_pts_theta, OA, num_pts, verify);
    // printf("splitting %d\n", success);
    return success;
}

bool theta_isogeny_chain_ver(
    theta_couple_curve_t* res_E34, 
    theta_couple_curve_t* E12, 
    theta_kernel_couple_points_t* ker,
    const int n
)
{
    theta_couple_point_t pts[0], res_pts[0];
    int num_pts = 0;
    bool randomize = false, verify = true;
    return theta_isogeny_chain_single(
        res_E34, res_pts, E12, ker, pts, n, num_pts, randomize, verify
    );
}

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
)
{
    __m512i A_12[8], A24_12[8], ker_x[8], ker_y[8], ker_z[8], OA[8], OA_dual_inv[8];
    __m512i J[8], matrix01[16], matrix23[16], T8_x[8], T8_y[8], T8_z[8], gluing_eval_res[16];
    __m512i eval_pts_theta[8*num_pts], idx;

    // TODO: Copied from middle. Which variables can be merged?
    bool h1, h2, success = true;
    splitting_matrix_t isom_matrix, rand_matrix;
    __m512i c1[8], c2[8], p1[8], p2[8], k1[8], k2[8];
    fp2_t unvec1[4];

    // TODO: Is it enough to just normalize curves? 
    theta_normalize_chain_input(E12, ker);   

    // lift the basis
    theta_couple_jac_point_t xyT1, xyT2;

    ec_basis_t bas1 = { .P = ker->T1.P1, .Q = ker->T2.P1, .PmQ = ker->T1m2.P1 };
    ec_basis_t bas2 = { .P = ker->T1.P2, .Q = ker->T2.P2, .PmQ = ker->T1m2.P2 };
    
    success &= lift_basis(&xyT1.P1, &xyT2.P1, &bas1, &E12->E1, verify);
    success &= lift_basis(&xyT1.P2, &xyT2.P2, &bas2, &E12->E2, verify);

    // We are now in the position to convert from 64 to 51 bit limbs
    // For the curve we need to pack the Montgomery coefficient for
    // arithmetic.
    // A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    unvec1[0] = E12->E1.A; unvec1[1] = E12->E1.A; unvec1[2] = E12->E2.A; unvec1[3] = E12->E2.A;
    fp2_unvectorized_to_4x2x1(A_12, unvec1);
    // A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)
    unvec1[0] = E12->E1.A24.x; unvec1[1] = E12->E1.A24.x; unvec1[2] = E12->E2.A24.x; unvec1[3] = E12->E2.A24.x;
    fp2_unvectorized_to_4x2x1(A24_12, unvec1);

    // For the Jacobian points, we pack the coordinates into three blobs
    // ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X)
    unvec1[0] = xyT1.P1.x; unvec1[1] = xyT2.P1.x; unvec1[2] = xyT1.P2.x; unvec1[3] = xyT2.P2.x;
    fp2_unvectorized_to_4x2x1(ker_x, unvec1);
    // ker_Y = AVX_4x2x1(P1.Y, Q1.Y, P2.Y, Q2.Y)
    unvec1[0] = xyT1.P1.y; unvec1[1] = xyT2.P1.y; unvec1[2] = xyT1.P2.y; unvec1[3] = xyT2.P2.y;
    fp2_unvectorized_to_4x2x1(ker_y, unvec1);
    // ker_Z = AVX_4x2x1(P1.Z, Q1.Z, P2.Z, Q2.Z)
    unvec1[0] = xyT1.P1.z; unvec1[1] = xyT2.P1.z; unvec1[2] = xyT1.P2.z; unvec1[3] = xyT2.P2.z;
    fp2_unvectorized_to_4x2x1(ker_z, unvec1);

    // Setting up the strategy
    int space = 1;
    for (unsigned i = 1; i < n; i *= 2) // n.bit_length + 1
        ++space;

    uint16_t orders[space];
    __m512i strat_points_x[8*space], strat_points_y[8*space], strat_points_z[8*space];
    // TODO: ker_x, ker_y, ker_z never used again, remove copies.
    orders[0] = n - 2;
    fp2_copy_4x2x1(strat_points_x, ker_x);
    fp2_copy_4x2x1(strat_points_y, ker_y);
    fp2_copy_4x2x1(strat_points_z, ker_z);

    int current = theta_compute_product_doubles(
        A_12, strat_points_x, strat_points_y, strat_points_z, orders, space
    );

    // Compute the gluing codomain
    success &= gluing_codomain(
        OA, OA_dual_inv, J, matrix01, matrix23, A24_12, 
        strat_points_x + (8*current), strat_points_z + (8*current),
        verify
    );

    // Package the point we use for translation
    // T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    // T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    // T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    idx = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    for (int i = 0; i <  8; i++) {
        int j = 8*current + i;
        T8_x[i] = VPERM512(idx, strat_points_x[j]);
        T8_y[i] = VPERM512(idx, strat_points_y[j]);
        T8_z[i] = VPERM512(idx, strat_points_z[j]);
    }

    // Push kernel though the gluing isogeny
    // 4 Jacobian points for the kernel become 2 theta points
    // TODO: Is current = space-1 here? And even if it is not, is it better to use 8*space?
    __m512i strat_theta1[8*space], strat_theta2[8*space];
    for (int i = 0; i < current; i++) {
        gluing_eval_pair(
            gluing_eval_res, //8x1x1
            A_12, strat_points_x + (8*i), strat_points_y + (8*i), strat_points_z + (8*i),
            T8_x, T8_y, T8_z, J, matrix01, matrix23
        );
        fp2_8x1x1_to_4x2x1( // Since we have a 4x2x1 chain and we don't have gluing_eval_single
            strat_theta1 + (8*i), strat_theta2 + (8*i), gluing_eval_res
        );
        orders[i]--;
    }

    // Push other points through gluing isogeny
    for (int i = 0; i < num_pts; i++) {
        success &= gluing_eval_special(
            eval_pts_theta + (8*i),
            pts + i, OA_dual_inv, matrix01, matrix23, verify
        );
    }
    current--;

    // Step Two:
    // Compute all but the last two steps of the isogeny without square roots
    h1 = false; h2 = true;
    while (current >= 0 && orders[current] != 0) {
        // Precompute constants for doubling
        theta_precomp_single(c1, c2, OA);
        
        while (orders[current] != 1) {
            current++;

            int num_dbls = orders[current-1] / 2;
            theta_double_iter_single(
                strat_theta1 + (8*current), 
                strat_theta1 + (8*(current-1)), 
                c1, c2, num_dbls
            );
            theta_double_iter_single(
                strat_theta2 + (8*current), 
                strat_theta2 + (8*(current-1)), 
                c1, c2, num_dbls
            );
            orders[current] = orders[current-1] - num_dbls;
        }

        // Compute the codomain of the kernel
        theta_codomain_single(
            OA, OA_dual_inv, 
            strat_theta1 + (8*current), strat_theta2 + (8*current),
            h1, h2
        );
        
        // Push the kernel through the images
        theta_eval_list_single(strat_theta1, OA_dual_inv, h1, h2, current);
        theta_eval_list_single(strat_theta2, OA_dual_inv, h1, h2, current);
        for (int i = 0; i < current; i++) {
            orders[i]--;
        }
        current--;

        // Eval points
        theta_eval_list_single(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);
    }

    // Step Three:
    // we now do the last two steps outside of the strategy. As we do not
    // have the torsion above we require square roots to compute the last
    // two codomains.
    
    if (n > 3) {
        theta_eval_single(strat_theta1, strat_theta1, OA_dual_inv, h1, h2);
    }

    h1 = false; h2 = false;
    theta_codomain_with_four_torsion(OA_dual_inv, OA, strat_theta1, h1, h2);
    theta_eval_list_single(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);

    h1 = true; h2 = false;
    theta_codomain_with_no_torsion(OA_dual_inv, OA, h1, h2);
    theta_eval_list_single(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);

    // Step Four: Splitting
    success &= splitting_isom_matrix(&isom_matrix, OA, verify);
    splitting_apply_isomorphism_single(OA, OA, &isom_matrix);
    splitting_apply_isomorphism_list_single(eval_pts_theta, &isom_matrix, num_pts);

    if (randomize) {
        splitting_rand_matrix(&rand_matrix);
        splitting_apply_isomorphism_single(OA, OA, &rand_matrix);
        splitting_apply_isomorphism_list_single(eval_pts_theta, &rand_matrix, num_pts);
    }

    success &= splitting_theta_to_product(res_E34, OA, verify);
    success &= splitting_theta_point_to_mont_list_single(res_pts, eval_pts_theta, OA, num_pts, verify);

    return success;
}

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
)
{
    __m512i A_12[8], A24_12[8], ker_x[8], ker_y[8], ker_z[8], OA[8], OA_dual_inv[16];
    __m512i J[8], matrix01[16], matrix23[16], T8_x[8], T8_y[8], T8_z[8], gluing_eval_res[16];
    __m512i eval_pts_theta[8*num_pts], idx;

    // TODO: Copied from middle. Which variables can be merged?
    bool h1, h2, success = true;
    splitting_matrix_t isom_matrix, rand_matrix;
    __m512i c1[16], c2[16], p1[8], p2[8], k1[8], k2[8];

    fp2_t unvec1[4];

    // TODO: Is it enough to just normalize curves? 
    theta_normalize_chain_input(E12, ker);   

    // lift the basis
    theta_couple_jac_point_t xyT1, xyT2;
    ec_basis_t bas1 = { .P = ker->T1.P1, .Q = ker->T2.P1, .PmQ = ker->T1m2.P1 };
    ec_basis_t bas2 = { .P = ker->T1.P2, .Q = ker->T2.P2, .PmQ = ker->T1m2.P2 };

    success &= lift_basis(&xyT1.P1, &xyT2.P1, &bas1, &E12->E1, verify);
    success &= lift_basis(&xyT1.P2, &xyT2.P2, &bas2, &E12->E2, verify);

    // We are now in the position to convert from 64 to 51 bit limbs
    // For the curve we need to pack the Montgomery coefficient for
    // arithmetic.
    // A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    unvec1[0] = E12->E1.A; unvec1[1] = E12->E1.A; unvec1[2] = E12->E2.A; unvec1[3] = E12->E2.A;
    fp2_unvectorized_to_4x2x1(A_12, unvec1);
    // A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)
    unvec1[0] = E12->E1.A24.x; unvec1[1] = E12->E1.A24.x; unvec1[2] = E12->E2.A24.x; unvec1[3] = E12->E2.A24.x;
    fp2_unvectorized_to_4x2x1(A24_12, unvec1);

    // For the Jacobian points, we pack the coordinates into three blobs
    // ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X)
    unvec1[0] = xyT1.P1.x; unvec1[1] = xyT2.P1.x; unvec1[2] = xyT1.P2.x; unvec1[3] = xyT2.P2.x;
    fp2_unvectorized_to_4x2x1(ker_x, unvec1);
    // ker_Y = AVX_4x2x1(P1.Y, Q1.Y, P2.Y, Q2.Y)
    unvec1[0] = xyT1.P1.y; unvec1[1] = xyT2.P1.y; unvec1[2] = xyT1.P2.y; unvec1[3] = xyT2.P2.y;
    fp2_unvectorized_to_4x2x1(ker_y, unvec1);
    // ker_Z = AVX_4x2x1(P1.Z, Q1.Z, P2.Z, Q2.Z)
    unvec1[0] = xyT1.P1.z; unvec1[1] = xyT2.P1.z; unvec1[2] = xyT1.P2.z; unvec1[3] = xyT2.P2.z;
    fp2_unvectorized_to_4x2x1(ker_z, unvec1);

    // Setting up the strategy
    int space = 1;
    for (unsigned i = 1; i < n; i *= 2) // n.bit_length + 1
        ++space;

    uint16_t orders[space];
    __m512i strat_points_x[8*space], strat_points_y[8*space], strat_points_z[8*space];
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
        strat_points_x + (8*current), strat_points_z + (8*current),
        verify
    );

    // Package the point we use for translation
    // T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    // T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    // T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    idx = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    for (int i = 0; i <  8; i++) {
        int j = 8*current + i;
        T8_x[i] = VPERM512(idx, strat_points_x[j]);
        T8_y[i] = VPERM512(idx, strat_points_y[j]);
        T8_z[i] = VPERM512(idx, strat_points_z[j]);
    }

    // Push kernel though the gluing isogeny
    // 4 Jacobian points for the kernel become 2 theta points
    // TODO: Is current = space-1 here? And even if it is not, is it better to use 8*space?
    __m512i strat_theta_pair[16*space];
    for (int i = 0; i < current; i++) {
        gluing_eval_pair(
            strat_theta_pair + 16*i, //8x1x1
            A_12, strat_points_x + (8*i), strat_points_y + (8*i), strat_points_z + (8*i),
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
    h1 = false; h2 = true;
    while (current >= 0 && orders[current] != 0) {
        // Precompute constants for doubling
        theta_precomp_pair(c1, c2, OA);
        
        while (orders[current] != 1) {
            current++;

            int num_dbls = orders[current-1] / 2;
            theta_double_iter_pair(
                strat_theta_pair + (16*current), 
                strat_theta_pair + (16*(current-1)), 
                c1, c2, num_dbls
            );
            orders[current] = orders[current-1] - num_dbls;
        }

        // Compute the codomain of the kernel
        success &= theta_codomain_pair(
            OA, OA_dual_inv, strat_theta_pair + (16*current), 
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
        step++;
    }

    // Step Three:
    // we now do the last two steps outside of the strategy. As we do not
    // have the torsion above we require square roots to compute the last
    // two codomains.
    
    //TODO: We can save 30 cycles by using theta_eval_pair_old here, as we only need OA_inv in 4x2x1.
    fp2_8x1x1_to_4x2x1(strat_theta_pair, strat_theta_pair + 8, strat_theta_pair);
    if (n > 3) {
        fp2_8x1x1_to_first_4x2x1(OA_dual_inv, OA_dual_inv);
        theta_eval_single(strat_theta_pair, strat_theta_pair, OA_dual_inv, h1, h2);
    }

    
    h1 = false; h2 = false;
    theta_codomain_with_four_torsion(OA_dual_inv, OA, strat_theta_pair, h1, h2);
    theta_eval_list_pair_old(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);

    h1 = true; h2 = false;
    theta_codomain_with_no_torsion(OA_dual_inv, OA, h1, h2);
    theta_eval_list_pair_old(eval_pts_theta, OA_dual_inv, h1, h2, num_pts);


    // Step Four: Splitting
    success &= splitting_isom_matrix(&isom_matrix, OA, verify);
    splitting_apply_isomorphism_single(OA, OA, &isom_matrix);
    splitting_apply_isomorphism_list_pair(eval_pts_theta, &isom_matrix, num_pts);

    if (randomize) {
        splitting_rand_matrix(&rand_matrix);
        splitting_apply_isomorphism_single(OA, OA, &rand_matrix);
        splitting_apply_isomorphism_list_pair(eval_pts_theta, &rand_matrix, num_pts);
    }

    success &= splitting_theta_to_product(res_E34, OA, verify);
    success &= splitting_theta_point_to_mont_list_pair(res_pts, eval_pts_theta, OA, num_pts, verify);

    return success;
}
