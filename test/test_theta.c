#include <stdio.h>

#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "test.h"
#include "theta.h"
#include "utils.h"

bool test_theta(int iterations)
{
    __m512i a[16], b[16], c[16], d[16], e[16], f[16], zero[16], one[16], t;
    fp2_setzero_8x1x1(zero);
    fp2_setzero_8x1x1(one);
    one[0] = VSETALL(1);
    int passed = 1;

    printf("Theta single tests:\n");

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
    print_test_res("Hadamard", passed);
    if (passed == 0) {
        return false;
    }

    printf("Theta pair tests:\n");

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
        fp2_8x1x1_to_4x2x1(e, a);
        theta_had_single(e, e);
        theta_had_single(e+8, e+8);
        fp2_4x2x1_to_8x1x1(f, e);

        if (fp2_is_equal_8x1x1(d, f) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("Hadamard", passed);
    if (passed == 0) {
        return false;
    }

    return true;
}