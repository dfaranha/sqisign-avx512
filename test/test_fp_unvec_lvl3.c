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
    fp_t unvec1[8], unvec2[8];
    fp2_t unvec1_fp2[8], unvec2_fp2[8];
    int passed = 1;

    int fail = 0;
for (int iter = 0; iter < 10; iter++) {
    fp_t unvec_orig[4], unvec_rt[4];
    uint64_t raw_in[4*NWORDS64], raw_out[4*NWORDS64];
    __m512i packed[N4x2], monted[N4x2], demonted[N4x2], zero4[N4x2];
    
    for (int j = 0; j < 4; j++)
        fp_random_test(unvec_orig + j);
    
    for (int i = 0; i < 4; i++)
        fp_encode(raw_in + i*NWORDS64, unvec_orig + i);
    fp_64_to_limbs_4x2(packed, raw_in);
    fp_tomont_4x2(monted, packed);
    fp_frommont_4x2(demonted, monted);
    fp_setzero_4x2(zero4);
    fp_add_4x2(demonted, demonted, zero4);
    fp_limbs_to_64_4x2(raw_out, demonted);
    
    // Check raw first
    for (int i = 0; i < 4; i++)
        for (int k = 0; k < NWORDS64; k++)
            if (raw_in[i*NWORDS64+k] != raw_out[i*NWORDS64+k])
                printf("RAW fail iter %d elem %d word %d: %016lx vs %016lx\n",
                       iter, i, k, raw_in[i*NWORDS64+k], raw_out[i*NWORDS64+k]);
    
    // Then check after decode
    for (int i = 0; i < 4; i++) {
        fp_decode(unvec_rt + i, (const void*)(raw_out + i*NWORDS64));
        if (fp_is_equal(unvec_orig + i, unvec_rt + i) == 0)
            printf("FP fail iter %d elem %d\n", iter, i);
    }
}

    printf("Fp vec <-> SQIsign unvec tests:\n");

    // 8x1 <-> unvectorized
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 8; j++) {
            fp_random_test(unvec1+j);
        }
        rand_fp_8x1(a);

        // Converting back and forth does nothin
        fp_unvectorized_to_8x1(d, unvec1);
        fp_8x1_to_unvectorized(unvec2, d);

        for (int j = 0; j < 8; j++) {
            if (fp_is_equal(unvec1+j, unvec2+j) == 0) { passed = 0; }
        }
        if (passed == 0) {
            break;
        }

        fp_8x1_to_unvectorized(unvec2, a);
        fp_unvectorized_to_8x1(d, unvec2);
        if (fp_is_equal_8x1(a, d) != 0xff) {
            passed = 0;
            break;
        }
        
        // Conversion commutes with arithmetic
        rand_fp_8x1(a);
        rand_fp_8x1(b);
        fp_mul_8x1(c, a, b);

        fp_8x1_to_unvectorized(unvec1, a);
        fp_8x1_to_unvectorized(unvec2, b);
        for (int j = 0; j < 8; j++) {
            fp_mul(unvec1+j, unvec1+j, unvec2+j);
        }
        fp_unvectorized_to_8x1(d, unvec1);

        if (fp_is_equal_8x1(c, d) != 0xff) {
            passed = 0;
            break;
        }

        for (int j = 0; j < 8; j++) {
            fp_random_test(unvec1+j);
            fp_random_test(unvec2+j);
        }

        fp_unvectorized_to_8x1(d, unvec1);
        fp_unvectorized_to_8x1(e, unvec2);

        for (int j = 0; j < 8; j++) {
            fp_mul(unvec1+j, unvec1+j, unvec2+j);
        }
        fp_mul_8x1(f, d, e);
        fp_8x1_to_unvectorized(unvec2, f);
        
        for (int j = 0; j < 8; j++) {
            if (fp_is_equal(unvec1+j, unvec2+j) == 0) {
                passed = 0;
                break;
            }
        }
    }
    print_test_res("unvectorized <-> 8x1", passed);
    if (passed == 0) {
        return false;
    }

    // 4x2 <-> unvectorized
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 8; j++) {
            fp_random_test(unvec1+j);
        }
        rand_fp_4x2(a);

        // Converting back and forth does nothin
        fp_unvectorized_to_4x2(d, unvec1);
        fp_4x2_to_unvectorized(unvec2, d);

        for (int j = 0; j < 4; j++) {
            if (fp_is_equal(unvec1+j, unvec2+j) == 0) { passed = 0; }
        }
        if (passed == 0) {
            printf("unvec -> 4x2 -> unvec != id\n");
            break;
        }

        fp_4x2_to_unvectorized(unvec2, a);
        fp_unvectorized_to_4x2(d, unvec2);

        if (fp_is_equal_4x2(a, d) != 0xf) {
            passed = 0;
            printf("4x2 -> unvec -> 4x2 != id\n");
            break;
        }
        
        // Conversion commutes with arithmetic
        rand_fp_4x2(a);
        rand_fp_4x2(b);
        fp_mul_4x2(c, a, b);

        fp_4x2_to_unvectorized(unvec1, a);
        fp_4x2_to_unvectorized(unvec2, b);
        for (int j = 0; j < 4; j++) {
            fp_mul(unvec1+j, unvec1+j, unvec2+j);
        }
        fp_unvectorized_to_4x2(d, unvec1);

        if (fp_is_equal_4x2(c, d) != 0xf) {
            passed = 0;
            break;
        }

        for (int j = 0; j < 4; j++) {
            fp_random_test(unvec1+j);
            fp_random_test(unvec2+j);
        }

        fp_unvectorized_to_4x2(d, unvec1);
        fp_unvectorized_to_4x2(e, unvec2);

        for (int j = 0; j < 4; j++) {
            fp_mul(unvec1+j, unvec1+j, unvec2+j);
        }
        fp_mul_4x2(f, d, e);
        fp_4x2_to_unvectorized(unvec2, f);
        
        for (int j = 0; j < 4; j++) {
            if (fp_is_equal(unvec1+j, unvec2+j) == 0) {
                passed = 0;
                break;
            }
        }
    }
    print_test_res("unvectorized <-> 4x2", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 vec <-> SQIsign unvec tests:\n");
    for (int i = 0; i < iterations; i++) {
        for (int j = 0; j < 8; j++) {
            fp2_random_test(unvec1_fp2+j);
        }
        rand_fp2_8x1x1(a);

        // Converting back and forth does nothin
        fp2_unvectorized_to_8x1x1(d, unvec1_fp2);
        fp2_8x1x1_to_unvectorized(unvec2_fp2, d);

        for (int j = 0; j < 8; j++) {
            if (fp2_is_equal(unvec1_fp2+j, unvec2_fp2+j) == 0) { passed = 0; }
        }
        if (passed == 0) {
            printf("To vectorized and back is not identity!\n");
            break;
        }

        fp2_8x1x1_to_unvectorized(unvec2_fp2, a);
        fp2_unvectorized_to_8x1x1(d, unvec2_fp2);
        if (fp2_is_equal_8x1x1(a, d) != 0xff) {
            passed = 0;
            printf("To unvectorized and back is not identity!\n");
            break;
        }

        // Conversion commutes with arithmetic
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        fp2_mul_8x1x1(c, a, b);

        fp2_8x1x1_to_unvectorized(unvec1_fp2, a);
        fp2_8x1x1_to_unvectorized(unvec2_fp2, b);
        for (int j = 0; j < 8; j++) {
            fp2_mul(unvec1_fp2+j, unvec1_fp2+j, unvec2_fp2+j);
        }
        fp2_unvectorized_to_8x1x1(d, unvec1_fp2);

        if (fp2_is_equal_8x1x1(c, d) != 0xff) {
            printf("The conversion does not commute with mul!\n");
            passed = 0;
            break;
        }

        for (int j = 0; j < 8; j++) {
            fp2_random_test(unvec1_fp2+j);
            fp2_random_test(unvec2_fp2+j);
        }

        fp2_unvectorized_to_8x1x1(d, unvec1_fp2);
        fp2_unvectorized_to_8x1x1(e, unvec2_fp2);

        for (int j = 0; j < 8; j++) {
            fp2_mul(unvec1_fp2+j, unvec1_fp2+j, unvec2_fp2+j);
        }
        fp2_mul_8x1x1(f, d, e);
        fp2_8x1x1_to_unvectorized(unvec2_fp2, f);
        
        for (int j = 0; j < 8; j++) {
            if (fp2_is_equal(unvec1_fp2+j, unvec2_fp2+j) == 0) {
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
            fp2_random_test(&unvec1_fp2[j]);
        }
        rand_fp2_4x2x1(a);

        // Converting back and forth does nothin
        fp2_unvectorized_to_4x2x1(d, unvec1_fp2);
        fp2_4x2x1_to_unvectorized(unvec2_fp2, d);

        for (int j = 0; j < 4; j++) {
            if (fp2_is_equal(&unvec1_fp2[j], &unvec2_fp2[j]) == 0) { passed = 0; }
        }
        if (passed == 0) {
            printf("To vectorized and back is not identity!\n");
            break;
        }

        fp2_4x2x1_to_unvectorized(unvec2_fp2, a);
        fp2_unvectorized_to_4x2x1(d, unvec2_fp2);

        if (fp2_is_equal_4x2x1(a, d) != 0xf) {
            passed = 0;
            printf("To unvectorized and back is not identity!\n");
            break;
        }
        
        // Conversion commutes with arithmetic
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        fp2_mul_4x2x1(c, a, b);

        fp2_4x2x1_to_unvectorized(unvec1_fp2, a);
        fp2_4x2x1_to_unvectorized(unvec2_fp2, b);
        for (int i = 0; i < 4; i++) {
            fp2_mul(&unvec1_fp2[i], &unvec1_fp2[i], &unvec2_fp2[i]);
        }
        fp2_unvectorized_to_4x2x1(d, unvec1_fp2);

        if (fp2_is_equal_4x2x1(c, d) != 0xf) {
            passed = 0;
            printf("The conversion does not commute with mul!\n");
            break;
        }

        for (int j = 0; j < 4; j++) {
            fp2_random_test(&unvec1_fp2[j]);
            fp2_random_test(&unvec2_fp2[j]);
        }

        fp2_unvectorized_to_4x2x1(d, unvec1_fp2);
        fp2_unvectorized_to_4x2x1(e, unvec2_fp2);

        for (int j = 0; j < 4; j++) {
            fp2_mul(&unvec1_fp2[j], &unvec1_fp2[j], &unvec2_fp2[j]);
        }
        fp2_mul_4x2x1(f, d, e);
        fp2_4x2x1_to_unvectorized(unvec2_fp2, f);
        
        for (int j = 0; j < 4; j++) {
            if (fp2_is_equal(&unvec1_fp2[j], &unvec2_fp2[j]) == 0) {
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
            fp2_random_test(&unvec1_fp2[j]);
        }
        rand_fp2_2x2x2(a);

        // Converting back and forth does nothin
        fp2_unvectorized_to_2x2x2(d, unvec1_fp2);
        fp2_2x2x2_to_unvectorized(unvec2_fp2, d);

        for (int j = 0; j < 2; j++) {
            if (fp2_is_equal(&unvec1_fp2[j], &unvec2_fp2[j]) == 0) { passed = 0; }
        }
        if (passed == 0) {
            printf("To vectorized and back is not identity!\n");
            break;
        }

        fp2_2x2x2_to_unvectorized(unvec2_fp2, a);
        fp2_unvectorized_to_2x2x2(d, unvec2_fp2);

        if (fp2_is_equal_2x2x2(a, d) != 0x3) {
            passed = 0;
            printf("To unvectorized and back is not identity!\n");
            break;
        }
        
        // Conversion commutes with arithmetic
        rand_fp2_2x2x2(a);
        rand_fp2_2x2x2(b);
        fp2_mul_2x2x2(c, a, b);

        fp2_2x2x2_to_unvectorized(unvec1_fp2, a);
        fp2_2x2x2_to_unvectorized(unvec2_fp2, b);
        for (int i = 0; i < 2; i++) {
            fp2_mul(&unvec1_fp2[i], &unvec1_fp2[i], &unvec2_fp2[i]);
        }
        fp2_unvectorized_to_2x2x2(d, unvec1_fp2);

        if (fp2_is_equal_2x2x2(c, d) != 0x3) {
            passed = 0;
            printf("The conversion does not commute with mul!\n");
            break;
        }

        for (int j = 0; j < 2; j++) {
            fp2_random_test(&unvec1_fp2[j]);
            fp2_random_test(&unvec2_fp2[j]);
        }

        fp2_unvectorized_to_2x2x2(d, unvec1_fp2);
        fp2_unvectorized_to_2x2x2(e, unvec2_fp2);

        for (int j = 0; j < 2; j++) {
            fp2_mul(&unvec1_fp2[j], &unvec1_fp2[j], &unvec2_fp2[j]);
        }
        fp2_mul_2x2x2(f, d, e);
        fp2_2x2x2_to_unvectorized(unvec2_fp2, f);
        
        for (int j = 0; j < 2; j++) {
            if (fp2_is_equal(&unvec1_fp2[j], &unvec2_fp2[j]) == 0) {
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
            fp2_random_test(&unvec1_fp2[j]);
        }
        rand_fp2_2x4x1(a);

        // Converting back and forth does nothin
        fp2_unvectorized_to_2x4x1(d, unvec1_fp2);
        fp2_2x4x1_to_unvectorized(unvec2_fp2, d);

        for (int j = 0; j < 2; j++) {
            if (fp2_is_equal(&unvec1_fp2[j], &unvec2_fp2[j]) == 0) { passed = 0; }
        }
        if (passed == 0) {
            printf("To vectorized and back is not identity!\n");
            break;
        }

        fp2_2x4x1_to_unvectorized(unvec2_fp2, a);
        fp2_unvectorized_to_2x4x1(d, unvec2_fp2);

        if (fp2_is_equal_2x4x1(a, d) != 0x3) {
            passed = 0;
            printf("To unvectorized and back is not identity!\n");
            break;
        }
        
        // Conversion commutes with arithmetic
        rand_fp2_2x4x1(a);
        rand_fp2_2x4x1(b);
        fp2_mul_2x4x1(c, a, b);

        fp2_2x4x1_to_unvectorized(unvec1_fp2, a);
        fp2_2x4x1_to_unvectorized(unvec2_fp2, b);
        for (int i = 0; i < 2; i++) {
            fp2_mul(&unvec1_fp2[i], &unvec1_fp2[i], &unvec2_fp2[i]);
        }
        fp2_unvectorized_to_2x4x1(d, unvec1_fp2);

        if (fp2_is_equal_2x4x1(c, d) != 0x3) {
            passed = 0;
            printf("The conversion does not commute with mul!\n");
            break;
        }

        for (int j = 0; j < 2; j++) {
            fp2_random_test(&unvec1_fp2[j]);
            fp2_random_test(&unvec2_fp2[j]);
        }

        fp2_unvectorized_to_2x4x1(d, unvec1_fp2);
        fp2_unvectorized_to_2x4x1(e, unvec2_fp2);

        for (int j = 0; j < 2; j++) {
            fp2_mul(&unvec1_fp2[j], &unvec1_fp2[j], &unvec2_fp2[j]);
        }
        fp2_mul_2x4x1(f, d, e);
        fp2_2x4x1_to_unvectorized(unvec2_fp2, f);
        
        for (int j = 0; j < 2; j++) {
            if (fp2_is_equal(&unvec1_fp2[j], &unvec2_fp2[j]) == 0) {
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

    return true;
}