#include <stdbool.h>
#include <stdio.h>

#include "fp.h"
#include "fp2.h"
#include "test.h"
#include "theta_splitting.h"
#include "utils.h"

#include "sqi_fp2.h"
#include "sqi_ec.h"

bool test_theta_splitting(int iterations)
{
    __m512i a[16], b[16], c[16], d[16], e[16], f[16], g[16];
    int passed = 1;

    printf("Theta splitting tests:\n");

    // splitting isomorphism
    splitting_matrix_t matrix = {
        { 0xff, 0x00, 0x00, 0x00 }, // zero
        { 0x0f, 0xcc, 0x00, 0x00 }, // negative
        { 0x03, 0xf0, 0x00, 0x00 }  // imaginary
    };

    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(a);

        // Multiplication by first column should be all 0's
        splitting_mul_matrix_column_single(b, a, &matrix, 0);
        if (fp2_is_zero_4x2x1(b) != 0xf) {
            passed = 0;
            break;
        }

        fp2_copy_4x2x1(b, a); // testing in-place operation
        splitting_mul_matrix_column_single(b, b, &matrix, 1);
        int correct_mul_1 = fp2_is_equal_4x2x1(b, a) == 0x1;
        fp2_neg_4x2x1(c, a);
        int correct_mul_minus1 = fp2_is_equal_4x2x1(b, c) == 0x2;
        fp2_muli_4x2x1(c, a);
        int correct_mul_i = fp2_is_equal_4x2x1(b, c) == 0x4;
        fp2_neg_4x2x1(c, c);
        int correct_mul_minusi = fp2_is_equal_4x2x1(b, c) == 0x8;
        passed = correct_mul_1 & correct_mul_minus1 & correct_mul_i & correct_mul_minus1;
        if (passed == 0) {
            break;
        }
    }
    print_test_res("splitting matrix mult single", passed);
    if (passed == 0) {
        return false;
    }

    // splitting isomorphism
    for (int i = 0; i < iterations && passed; i++) {
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        fp2_4x2x1_to_8x1x1(c, a, b);

        for(int j = 0; j < 4 && passed; j++) {
            splitting_mul_matrix_column_single(d, a, &matrix, j);
            splitting_mul_matrix_column_single(e, b, &matrix, j);
            fp2_4x2x1_to_8x1x1(f, d, e);
            splitting_mul_matrix_column_pair(g, c, &matrix, j);
            passed = fp2_is_equal_8x1x1(f, g) == 0xff;
        }
    }
    print_test_res("splitting matrix mult pair", passed);
    if (passed == 0) {
        return false;
    }

    return true;
}