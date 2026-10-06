#include <stdio.h>

#include "fp.h"
#include "fp2.h"
#include "fp_consts.h"
#include "test.h"
#include "theta_middle.h"
#include "utils.h"

bool test_theta_middle(int iterations)
{
    __m512i a[2*NUM_LIMBS], b[2*NUM_LIMBS], c[2*NUM_LIMBS], d[2*NUM_LIMBS], e[2*NUM_LIMBS];
    __m512i f[2*NUM_LIMBS];
    int passed = 1;

    // ======================================================================================================
    //                                                 Single
    // ======================================================================================================

    printf("Theta middle tests:\n");

    // Hadamard
    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        
        // H^2 is mult by 4
        theta_had_single(d, a);
        theta_had_single(e, d);
        fp2_add_4x2x1(f, a, a);
        fp2_add_4x2x1(f, f, f);

        if (fp2_is_equal_4x2x1(e, f) != 0xf) {
            passed = 0;
            break;
        }

        // In place
        fp2_copy_4x2x1(d, a);
        theta_had_single(d, d);
        theta_had_single(e, a);

        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // Linear H*a + H*b = H*(a+b)
        theta_had_single(d, a);
        theta_had_single(e, b);
        fp2_add_4x2x1(d, d, e);
        fp2_add_4x2x1(f, a, b);
        theta_had_single(f, f);

        if (fp2_is_equal_4x2x1(d, f) != 0xf) {
            passed = 0;
            break;
        }
    }

    print_test_res("Hadamard single", passed);
    if (passed == 0) {
        return false;
    }

    // ======================================================================================================
    //                                                 Pair
    // ======================================================================================================

    // Hadamard
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        
        // H^2 is mult by 4
        theta_had_pair(d, a);
        theta_had_pair(e, d);
        fp2_add_8x1x1(f, a, a);
        fp2_add_8x1x1(f, f, f);

        if (fp2_is_equal_8x1x1(e, f) != 0xff) {
            passed = 0;
            break;
        }

        // In place
        fp2_copy_8x1x1(d, a);
        theta_had_pair(d, d);
        theta_had_pair(e, a);

        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // Linear H*a + H*b = H*(a+b)
        theta_had_pair(d, a);
        theta_had_pair(e, b);
        fp2_add_8x1x1(d, d, e);
        fp2_add_8x1x1(f, a, b);
        theta_had_pair(f, f);

        if (fp2_is_equal_8x1x1(d, f) != 0xff) {
            passed = 0;
            break;
        }

        // Pair matches single
        theta_had_pair(d, a);
        fp2_8x1x1_to_4x2x1(e, e + NUM_LIMBS, a);
        theta_had_single(e, e);
        theta_had_single(e + NUM_LIMBS, e + NUM_LIMBS);
        fp2_4x2x1_to_8x1x1(f, e, e + NUM_LIMBS);

        if (fp2_is_equal_8x1x1(d, f) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("Hadamard pair", passed);
    if (passed == 0) {
        return false;
    }

    // mul
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);

        // Pair matches single
        theta_mul_pair(d, a, b);
        fp2_8x1x1_to_4x2x1(e, e + NUM_LIMBS, a);
        theta_mul_single(f, e, b);
        theta_mul_single(f + NUM_LIMBS, e + NUM_LIMBS, b);
        fp2_4x2x1_to_8x1x1(e, f, f + NUM_LIMBS);

        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("mul", passed);
    if (passed == 0) {
        return false;
    }

    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(c);

        // Pair matches single
        fp2_8x1x1_to_4x2x1(e, e + NUM_LIMBS, a);
        theta_double_single(f, e, b, c);
        theta_double_single(f + NUM_LIMBS, e + NUM_LIMBS, b, c);
        fp2_4x2x1_to_8x1x1(e, f, f + NUM_LIMBS);

        fp2_4x2x1_to_8x1x1(b, b, b);
        fp2_4x2x1_to_8x1x1(c, c, c);
        theta_double_pair(d, a, b, c);
        
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("double", passed);
    if (passed == 0) {
        return false;
    }

    // TODO: Test is not using an actual dual theta point
    // eval
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);
        bool h1 = false;
        bool h2 = true;

        // Pair matches single
        fp2_8x1x1_to_4x2x1(e, e + NUM_LIMBS, a);
        theta_eval_single(f, e, b, h1, h2);
        theta_eval_single(f + NUM_LIMBS, e + NUM_LIMBS, b, h1, h2);
        fp2_4x2x1_to_8x1x1(e, f, f + NUM_LIMBS);

        fp2_4x2x1_to_8x1x1(b, b, b);
        theta_eval_pair(d, a, b, h1, h2);

        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("eval", passed);
    if (passed == 0) {
        return false;
    }

    return true;
}