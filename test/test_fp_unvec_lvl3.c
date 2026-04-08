#include <stdio.h>

#include "fp.h"
#include "fp_consts.h"
#include "fp_unvec_lvl3.h"
#include "fp2.h"
#include "fp2_kat.h"
#include "test.h"
#include "utils.h"
#include "sqi_test_extras.h"

bool test_fp_unvec_lvl3(int iterations)
{
    __m512i a[2*NUM_LIMBS], b[2*NUM_LIMBS], c[2*NUM_LIMBS], d[2*NUM_LIMBS], e[2*NUM_LIMBS], f[2*NUM_LIMBS], t;
    fp2_t unvec1[8], unvec2[8];
    int passed = 1;

    printf("Fp2 vec <-> SQIsign unvec tests:\n");
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 8; j++) {
            fp2_random_test(unvec1+j);
        }
        rand_fp2_8x1x1(a);

        // Converting back and forth does nothin
        fp2_unvectorized_to_8x1x1(d, unvec1);
        fp2_8x1x1_to_unvectorized(unvec2, d);

        for (int j = 0; j < 8; j++) {
            if (fp2_is_equal(unvec1+j, unvec2+j) == 0) { passed = 0; }
        }
        if (passed == 0) {
            printf("To vectorized and back is not identity!\n");
            break;
        }

        fp2_8x1x1_to_unvectorized(unvec2, a);
        fp2_unvectorized_to_8x1x1(d, unvec2);
        if (fp2_is_equal_8x1x1(a, d) != 0xff) {
            passed = 0;
            printf("To unvectorized and back is not identity!\n");
            break;
        }

        // Top 13 bits of each AVX limb is 0
        t = VSETALL(0xfff8000000000000);
        fp2_unvectorized_to_8x1x1(d, unvec1);
        for (int j = 0; j < 2*NUM_LIMBS; j++) {
            e[j] = VAND(d[j], t);
        }
        if (fp2_is_zero_8x1x1(e) != 0xff) {
            passed = 0;
            break;
        }
        
        // Conversion commutes with arithmetic
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        fp2_mul_8x1x1(c, a, b);

        fp2_8x1x1_to_unvectorized(unvec1, a);
        fp2_8x1x1_to_unvectorized(unvec2, b);
        for (int j = 0; j < 8; j++) {
            fp2_mul(unvec1+j, unvec1+j, unvec2+j);
        }
        fp2_unvectorized_to_8x1x1(d, unvec1);

        if (fp2_is_equal_8x1x1(c, d) != 0xff) {
            printf("The conversion does not commute with mul!\n");
            passed = 0;
            break;
        }

        for (int j = 0; j < 8; j++) {
            fp2_random_test(unvec1+j);
            fp2_random_test(unvec2+j);
        }

        fp2_unvectorized_to_8x1x1(d, unvec1);
        fp2_unvectorized_to_8x1x1(e, unvec2);

        for (int j = 0; j < 8; j++) {
            fp2_mul(unvec1+j, unvec1+j, unvec2+j);
        }
        fp2_mul_8x1x1(f, d, e);
        fp2_8x1x1_to_unvectorized(unvec2, f);
        
        for (int j = 0; j < 8; j++) {
            if (fp2_is_equal(unvec1+j, unvec2+j) == 0) {
                passed = 0;
                printf("The conversion does not commute with mul!\n");
                break;
            }
        }
    }
    print_test_res("unvectorized <-> 8x1x1", passed);
    if (passed == 0) {
        return false;
    }

    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 4; j++) {
            fp2_random_test(&unvec1[j]);
        }
        rand_fp2_4x2x1(a);

        // Converting back and forth does nothin
        fp2_unvectorized_to_4x2x1(d, unvec1);
        fp2_4x2x1_to_unvectorized(unvec2, d);

        for (int j = 0; j < 4; j++) {
            if (fp2_is_equal(&unvec1[j], &unvec2[j]) == 0) { passed = 0; }
        }
        if (passed == 0) {
            printf("To vectorized and back is not identity!\n");
            break;
        }

        fp2_4x2x1_to_unvectorized(unvec2, a);
        fp2_unvectorized_to_4x2x1(d, unvec2);

        if (fp2_is_equal_4x2x1(a, d) != 0xf) {
            passed = 0;
            printf("To unvectorized and back is not identity!\n");
            break;
        }

        // Top 13 bits of each AVX limb is 0
        t = VSETALL(0xfff8000000000000);
        fp2_unvectorized_to_4x2x1(d, unvec1);
        for (int j = 0; j < NUM_LIMBS; j++) {
            e[j] = VAND(d[j], t);
        }
        if (fp2_is_zero_4x2x1(e) != 0xf) {
            passed = 0;
            break;
        }
        
        // Conversion commutes with arithmetic
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        fp2_mul_4x2x1(c, a, b);

        fp2_4x2x1_to_unvectorized(unvec1, a);
        fp2_4x2x1_to_unvectorized(unvec2, b);
        for (int i = 0; i < 4; i++) {
            fp2_mul(&unvec1[i], &unvec1[i], &unvec2[i]);
        }
        fp2_unvectorized_to_4x2x1(d, unvec1);

        if (fp2_is_equal_4x2x1(c, d) != 0xf) {
            passed = 0;
            printf("The conversion does not commute with mul!\n");
            break;
        }

        for (int j = 0; j < 4; j++) {
            fp2_random_test(&unvec1[j]);
            fp2_random_test(&unvec2[j]);
        }

        fp2_unvectorized_to_4x2x1(d, unvec1);
        fp2_unvectorized_to_4x2x1(e, unvec2);

        for (int j = 0; j < 4; j++) {
            fp2_mul(&unvec1[j], &unvec1[j], &unvec2[j]);
        }
        fp2_mul_4x2x1(f, d, e);
        fp2_4x2x1_to_unvectorized(unvec2, f);
        
        for (int j = 0; j < 4; j++) {
            if (fp2_is_equal(&unvec1[j], &unvec2[j]) == 0) {
                passed = 0;
                printf("The conversion does not commute with mul!\n");
                break;
            }
        }
    }
    print_test_res("unvectorized <-> 4x2x1", passed);
    if (passed == 0) {
        return false;
    }

    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 2; j++) {
            fp2_random_test(&unvec1[j]);
        }
        rand_fp2_2x2x2(a);

        // Converting back and forth does nothin
        fp2_unvectorized_to_2x2x2(d, unvec1);
        fp2_2x2x2_to_unvectorized(unvec2, d);

        for (int j = 0; j < 2; j++) {
            if (fp2_is_equal(&unvec1[j], &unvec2[j]) == 0) { passed = 0; }
        }
        if (passed == 0) {
            printf("To vectorized and back is not identity!\n");
            break;
        }

        fp2_2x2x2_to_unvectorized(unvec2, a);
        fp2_unvectorized_to_2x2x2(d, unvec2);

        if (fp2_is_equal_2x2x2(a, d) != 0x3) {
            passed = 0;
            printf("To unvectorized and back is not identity!\n");
            break;
        }

        // Top 13 bits of each AVX limb is 0
        t = VSETALL(0xfff8000000000000);
        fp2_unvectorized_to_2x2x2(d, unvec1);
        for (int j = 0; j < N4x2; j++) {
            e[j] = VAND(d[j], t);
        }
        if (fp2_is_zero_2x2x2(e) != 0x3) {
            passed = 0;
            break;
        }
        
        // Conversion commutes with arithmetic
        rand_fp2_2x2x2(a);
        rand_fp2_2x2x2(b);
        fp2_mul_2x2x2(c, a, b);

        fp2_2x2x2_to_unvectorized(unvec1, a);
        fp2_2x2x2_to_unvectorized(unvec2, b);
        for (int i = 0; i < 2; i++) {
            fp2_mul(&unvec1[i], &unvec1[i], &unvec2[i]);
        }
        fp2_unvectorized_to_2x2x2(d, unvec1);

        if (fp2_is_equal_2x2x2(c, d) != 0x3) {
            passed = 0;
            printf("The conversion does not commute with mul!\n");
            break;
        }

        for (int j = 0; j < 2; j++) {
            fp2_random_test(&unvec1[j]);
            fp2_random_test(&unvec2[j]);
        }

        fp2_unvectorized_to_2x2x2(d, unvec1);
        fp2_unvectorized_to_2x2x2(e, unvec2);

        for (int j = 0; j < 2; j++) {
            fp2_mul(&unvec1[j], &unvec1[j], &unvec2[j]);
        }
        fp2_mul_2x2x2(f, d, e);
        fp2_2x2x2_to_unvectorized(unvec2, f);
        
        for (int j = 0; j < 2; j++) {
            if (fp2_is_equal(&unvec1[j], &unvec2[j]) == 0) {
                passed = 0;
                printf("The conversion does not commute with mul!\n");
                break;
            }
        }
    }
    print_test_res("unvectorized <-> 2x2x2", passed);
    if (passed == 0) {
        return false;
    }

    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 2; j++) {
            fp2_random_test(&unvec1[j]);
        }
        rand_fp2_2x4x1(a);

        // Converting back and forth does nothin
        fp2_unvectorized_to_2x4x1(d, unvec1);
        fp2_2x4x1_to_unvectorized(unvec2, d);

        for (int j = 0; j < 2; j++) {
            if (fp2_is_equal(&unvec1[j], &unvec2[j]) == 0) { passed = 0; }
        }
        if (passed == 0) {
            printf("To vectorized and back is not identity!\n");
            break;
        }

        fp2_2x4x1_to_unvectorized(unvec2, a);
        fp2_unvectorized_to_2x4x1(d, unvec2);

        if (fp2_is_equal_2x4x1(a, d) != 0x3) {
            passed = 0;
            printf("To unvectorized and back is not identity!\n");
            break;
        }

        // Top 13 bits of each AVX limb is 0
        t = VSETALL(0xfff8000000000000);
        fp2_unvectorized_to_2x4x1(d, unvec1);
        for (int j = 0; j < NUM_LIMBS; j++) {
            e[j] = VAND(d[j], t);
        }
        if (fp2_is_zero_2x4x1(e) != 0x3) {
            passed = 0;
            break;
        }
        
        // Conversion commutes with arithmetic
        rand_fp2_2x4x1(a);
        rand_fp2_2x4x1(b);
        fp2_mul_2x4x1(c, a, b);

        fp2_2x4x1_to_unvectorized(unvec1, a);
        fp2_2x4x1_to_unvectorized(unvec2, b);
        for (int i = 0; i < 2; i++) {
            fp2_mul(&unvec1[i], &unvec1[i], &unvec2[i]);
        }
        fp2_unvectorized_to_2x4x1(d, unvec1);

        if (fp2_is_equal_2x4x1(c, d) != 0x3) {
            passed = 0;
            printf("The conversion does not commute with mul!\n");
            break;
        }

        for (int j = 0; j < 2; j++) {
            fp2_random_test(&unvec1[j]);
            fp2_random_test(&unvec2[j]);
        }

        fp2_unvectorized_to_2x4x1(d, unvec1);
        fp2_unvectorized_to_2x4x1(e, unvec2);

        for (int j = 0; j < 2; j++) {
            fp2_mul(&unvec1[j], &unvec1[j], &unvec2[j]);
        }
        fp2_mul_2x4x1(f, d, e);
        fp2_2x4x1_to_unvectorized(unvec2, f);
        
        for (int j = 0; j < 2; j++) {
            if (fp2_is_equal(&unvec1[j], &unvec2[j]) == 0) {
                passed = 0;
                printf("The conversion does not commute with mul!\n");
                break;
            }
        }
    }
    print_test_res("unvectorized <-> 2x4x1", passed);
    if (passed == 0) {
        return false;
    }
}