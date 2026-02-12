#include <stdio.h>

#include "ec.h"
#include "fp2_lvl3.h"
#include "test.h"
#include "utils.h"
#include "sqi_test_extras.h"

bool test_ec(int iterations)
{   
    // ======== Setup =========================
    __m512i xP[8], zP[8], xQ[8], zQ[8], xR[8], zR[8], A24[8];
    fp2_t A24_unvec[4], x_unvec[4], z_unvec[4];
    ec_point_t P[4], Q[4], R[4];
    for (int i = 0; i < 4; i++) {
        ec_point_init(P+i);
        ec_point_init(Q+i);
        ec_point_init(R+i);
    }
    int passed = 1;
    
    // Working with fixed curve of j-invariant 1728
    ec_curve_t curve;
    ec_curve_init(&curve);
    fp2_set_small(&(curve.A), 0);
    fp2_set_small(&(curve.C), 1);
    ec_curve_normalize_A24(&curve);

    for (int i = 0; i < 4; i++) {
        fp2_copy(A24_unvec+i, &curve.A24.x);
    }
    fp2_unvectorized_to_4x2x1(A24, A24_unvec);

    printf("EC 4x2x1 tests:\n");
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
        ec_point_unvectorized_to_4x2x1(xP, zP, P);
        ec_point_4x2x1_to_unvectorized(R, xP, zP);
        for (int j = 0; j < 4 && passed; j++) {
            passed = ec_is_equal(P+j, R+j);
        }
        if (passed == 0) { break; }

        // To unvectorized and back
        ec_point_4x2x1_to_unvectorized(Q, xQ, zQ);
        ec_point_unvectorized_to_4x2x1(xP, zP, Q);
        passed = (fp2_is_equal_4x2x1(xP, xQ) == 0xf) 
            && (fp2_is_equal_4x2x1(zP, zQ) == 0xf);
    }
    print_test_res("unvectorized <-> 4x2x1", passed);
    if (passed == 0) {
        return false;
    }

    // Doubling
    for (int i = 0; i < iterations && passed; i++) {
        // Sample random points
        for (int j = 0; j < 4; j++) {
            ec_random_test(P+j, &curve);            
        }
        ec_point_unvectorized_to_4x2x1(xP, zP, P);

        // Double
        ec_double_4x2x1(xQ, zQ, A24, xP, zP);
        for (int j = 0; j < 4; j++) {
            xDBL_A24(R+j, P+j, &curve.A24, true);
        }

        // Compare
        ec_point_4x2x1_to_unvectorized(Q, xQ, zQ);
        // TODO remove when code has been merged:
        //ec_point_unvectorized_to_4x2x1(xR, zR, R);
        //printf("this works when merging: %x %x\n", fp2_is_equal_4x2x1(xQ, xR), fp2_is_equal_4x2x1(zQ, zR));
        for(int j = 0; j < 4 && passed; j++) {
            passed = ec_is_equal(Q+j, R+j);
        }
    }
    print_test_res("double", passed);
    if (passed == 0) {
        return false;
    }

    return true;
}