#include <stdio.h>

#include "ec.h"
#include "fp_consts.h"
#include "fp_unvec.h"
#include "fp2.h"
#include "test.h"
#include "utils.h"
#include "sqi_test_extras.h"

bool test_ec(int iterations)
{   
    // ======== Setup =========================
    __m512i xP[NUM_LIMBS], zP[NUM_LIMBS], xQ[NUM_LIMBS], zQ[NUM_LIMBS], xR[NUM_LIMBS], zR[NUM_LIMBS], A24[NUM_LIMBS]; 
    __m512i xz0[NUM_LIMBS], xz1[NUM_LIMBS], xz2[NUM_LIMBS];
    fp2_t A24_unvec[4], x_unvec[4], z_unvec[4], scalar;
    ec_point_t P[4], Q[4], R[4], A24_point;
    for (int i = 0; i < 4; i++) {
        ec_point_init(P+i);
        ec_point_init(Q+i);
        ec_point_init(R+i);
    }
    int passed = 1;
    
    // Working with fixed curve of j-invariant 1728
    ec_curve_t curve;
    ec_curve_init(&curve);
    fp2_set_zero(&(curve.A));
    fp2_set_one(&(curve.C));
    ec_curve_normalize_A24(&curve);

    for (int i = 0; i < 4; i++) {
        fp2_copy(A24_unvec+i, &curve.A24.x);
    }
    fp2_unvectorized_to_4x2x1(A24, A24_unvec);

    printf("EC tests:\n");
    // =======================================

    // Conversions between 4x2x1 and unvectorized
    for (int i = 0; i < iterations && passed; i++) {
        // Sample random values
        for (int j = 0; j < 4; j++) {
            ec_random_test(P+j, &curve);            
        }
        rand_fp2_4x2x1(xQ);
        rand_fp2_4x2x1(zQ);

        // To 4x2x1 and back
        ec_point_unvectorized_to_4x4x2x1(xP, zP, P);
        ec_point_4x4x2x1_to_unvectorized(R, xP, zP);
        for (int j = 0; j < 4 && passed; j++) {
            passed = ec_is_equal(P+j, R+j);
        }
        if (passed == 0) { break; }

        // To unvectorized and back
        ec_point_4x4x2x1_to_unvectorized(Q, xQ, zQ);
        ec_point_unvectorized_to_4x4x2x1(xP, zP, Q);
        passed = (fp2_is_equal_4x2x1(xP, xQ) == 0xf) 
            && (fp2_is_equal_4x2x1(zP, zQ) == 0xf);
    }
    print_test_res("unvectorized <-> 4x4x2x1", passed);
    if (passed == 0) {
        return false;
    }

    // Conversions between 1x2x4x1 and unvectorized
    for (int i = 0; i < iterations && passed; i++) {
        // Sample random values
        ec_random_test(P, &curve);  
        rand_fp2_2x4x1(xz0);

        // To 1x2x4x1 and back
        ec_point_unvec_to_1x2x4x1(xz1, P);
        ec_point_1x2x4x1_to_unvec(Q, xz1);
        passed = ec_is_equal(P, Q);
    
        if (passed == 0) { break; }

        // To unvectorized and back
        ec_point_1x2x4x1_to_unvec(Q, xz0);
        ec_point_unvec_to_1x2x4x1(xz1, Q);
        passed = (fp2_is_equal_2x4x1(xz0, xz1) == 0x3);
    }
    print_test_res("unvectorized <-> 1x2x4x1", passed);
    if (passed == 0) {
        return false;
    }

    // Doubling 4x4x2x1
    for (int i = 0; i < iterations && passed; i++) {
        // Sample random points
        for (int j = 0; j < 4; j++) {
            ec_random_test(P+j, &curve);            
        }
        ec_point_unvectorized_to_4x4x2x1(xP, zP, P);

        // Double
        ec_double_4x4x2x1(xQ, zQ, A24, xP, zP);
        for (int j = 0; j < 4; j++) {
            xDBL_A24(R+j, P+j, &curve.A24, true);
        }

        // Compare
        ec_point_4x4x2x1_to_unvectorized(Q, xQ, zQ);
        // TODO remove when code has been merged:
        //ec_point_unvectorized_to_4x2x1(xR, zR, R);
        //printf("this works when merging: %x %x\n", fp2_is_equal_4x2x1(xQ, xR), fp2_is_equal_4x2x1(zQ, zR));
        for(int j = 0; j < 4 && passed; j++) {
            passed = ec_is_equal(Q+j, R+j);
        }
    }
    print_test_res("double 4x4x2x1", passed);
    if (passed == 0) {
        return false;
    }

    // double 1x2x4x1
    for (int i = 0; i < iterations && passed; i++) {
        // Sample random point
        ec_random_test(P, &curve);            
        ec_point_unvec_to_1x2x4x1(xz0, P);

        // Randomize projective repr of A24
        fp2_set_zero(&scalar);
        while (fp2_is_zero(&scalar)) {
            fp2_random_test(&scalar);
        }
        fp2_mul(&A24_point.x, &scalar, &curve.A24.x);
        fp2_mul(&A24_point.z, &scalar, &curve.A24.z);
        ec_point_unvec_to_1x2x4x1(xz1, &A24_point);

        // Double
        ec_double_1x2x4x1(xz2, xz0, xz1);
        xDBL_A24(Q, P, &curve.A24, true);

        // Compare
        ec_point_1x2x4x1_to_unvec(R, xz2);
        passed = ec_is_equal(Q, R);
    }
    print_test_res("double 1x2x4x1", passed);
    if (passed == 0) {
        return false;
    }

    // double 1x2x2x2
    for (int i = 0; i < iterations && passed; i++) {
        // Sample random point
        ec_random_test(P, &curve);            
        ec_point_unvec_to_1x2x2x2(xz0, P);

        // Randomize projective repr of A24
        fp2_set_zero(&scalar);
        while (fp2_is_zero(&scalar)) {
            fp2_random_test(&scalar);
        }
        fp2_mul(&A24_point.x, &scalar, &curve.A24.x);
        fp2_mul(&A24_point.z, &scalar, &curve.A24.z);
        ec_point_unvec_to_1x2x2x2(xz1, &A24_point);

        // Double
        ec_double_1x2x2x2(xz2, xz0, xz1);
        xDBL_A24(Q, P, &curve.A24, true);

        // Compare
        ec_point_1x2x2x2_to_unvec(R, xz2);
        passed = ec_is_equal(Q, R);
    }
    print_test_res("double 1x2x2x2", passed);
    if (passed == 0) {
        return false;
    }

    return true;
}