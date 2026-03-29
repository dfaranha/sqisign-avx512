#include <stdio.h>

#include "fp_lvl3.h"
#include "test.h"
#include "utils.h"

#include "sqi_test_extras.h"

bool test_fp(int iterations)
{
    __m512i a[8], b[8], c[8], d[8], e[8], f[8], zero[8], one[8], t;
    fp_t unvec1[8], unvec2[8];
    fp_setzero_8x1(zero);
    fp_setzero_8x1(one);
    one[0] = VSETALL(1);
    int passed = 1;

    printf("Fp lvl3 8x1 tests:\n");

    // Equality test
    for(int i = 0; i < iterations; i++){
        rand_fp_8x1(a);
        if (fp_is_equal_8x1(a, a) != 0xff) {
            passed = 0;
            break;
        }
        fp_copy_8x1(b, a);
        if (fp_is_equal_8x1(a, b) != 0xff) {
            passed = 0;
            break;
        }

        // We check that making one change in each limb has the intended effect
        // b will be a copy of a, except that one limb of a is different
        // c will start out as a, but gradually be changed limb by limb.
        // d is the collection of vectors that we use to change each limb
        fp_copy_8x1(c, a);
        d[0] = VSET(0x1, 0, 0, 0, 0, 0, 0, 0);
        d[1] = VSET(0, 0x10, 0, 0, 0, 0, 0, 0);
        d[2] = VSET(0, 0, 0x100, 0, 0, 0, 0, 0);
        d[3] = VSET(0, 0, 0, 0x1000, 0, 0, 0, 0);
        d[4] = VSET(0, 0, 0, 0, 0x100000, 0, 0, 0);
        d[5] = VSET(0, 0, 0, 0, 0, 0x1000000, 0, 0);
        d[6] = VSET(0, 0, 0, 0, 0, 0, 0x100000000, 0);
        d[7] = VSET(0, 0, 0, 0, 0, 0, 0, 0x4000000000000);

        int mask_b = 0xfe;
        int mask_c = 0xfe;
        int multiplier = 1;
        int i_fp = 0, j_limb = 0;
        while(i_fp < 8) {
            b[j_limb] = VXOR(b[j_limb], d[i_fp]);
            c[j_limb] = VXOR(c[j_limb], d[i_fp]);
            int valid = (fp_is_equal_8x1(a, b) == mask_b) & (fp_is_equal_8x1(a,c) == mask_c);
            if (valid == 0) {
                passed = 0;
                break;
            }

            b[j_limb] = a[j_limb];
            if(j_limb == 7) {
                j_limb = 0;
                i_fp += 1;
                multiplier *= 2;
                mask_b = 0xff - multiplier; // b should be equal to a, except at index i_fp
                mask_c -= multiplier; // c should only be equal to a at the fp elements we haven't touched yet.
            }
            else {
                j_limb += 1;
            }
        }
        if (passed == 0) {
            break;
        }
    }
    print_test_res("equality", passed);
    if (passed == 0) {
        return false;
    }

    // Addition test
    for(int i = 0; i < iterations; i++) {
        rand_fp_8x1(a);
        rand_fp_8x1(b);
        rand_fp_8x1(c);

        // a + 0 == a
        fp_add_8x1(d, a, zero);
        if(fp_is_equal_8x1(a, d) != 0xff) {
            passed = 0;
            break;
        }

        // a + 1 != a
        fp_add_8x1(e, a, one);
        if (fp_is_equal_8x1(a, e) != 0) {
            passed = 0;
            break;
        }

        // a + b == b + a
        fp_add_8x1(d, a, b);
        fp_add_8x1(e, b, a);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // (a + b) + c = a + (b + c)
        fp_add_8x1(d, a, b);
        fp_add_8x1(d, d, c);
        fp_add_8x1(e, b, c);
        fp_add_8x1(e, e, a);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("addition", passed);
    if (passed == 0) {
        return false;
    }

    // Subtraction test
    for(int i = 0; i < iterations; i++) {
        rand_fp_8x1(a);
        rand_fp_8x1(b);
        rand_fp_8x1(c);

        // a - 0 == a
        fp_sub_8x1(d, a, zero);
        if(fp_is_equal_8x1(a, d) != 0xff) {
            passed = 0;
            break;
        }

        // a - 1 != a
        fp_sub_8x1(e, a, one);
        if (fp_is_equal_8x1(a, e) != 0) {
            passed = 0;
            break;
        }

        // a - a == 0
        fp_sub_8x1(d, a, a);
        if (fp_is_zero_8x1(d) != 0xff) {
            passed = 0;
            break;
        }

        // (a - b) - c = a - (b + c)
        fp_sub_8x1(d, a, b);
        fp_sub_8x1(d, d, c);
        fp_add_8x1(e, b, c);
        fp_sub_8x1(e, a, e);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("subtraction", passed);
    if (passed == 0) {
        return false;
    }

    // Negation tests
    for(int i = 0; i < iterations; i++) {
        rand_fp_8x1(a);
        rand_fp_8x1(b);

        // (-a) == 0 - a
        fp_neg_8x1(d, a);
        fp_sub_8x1(e, zero, a);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // -(-a) == a
        fp_neg_8x1(d, a);
        fp_neg_8x1(e, d);
        if (fp_is_equal_8x1(e, a) != 0xff) {
            passed = 0;
            break;
        }

        // a + (-a) == 0
        fp_neg_8x1(d, a);
        fp_add_8x1(e, a, d);
        if (fp_is_zero_8x1(e) != 0xff) {
            passed = 0;
            break;
        }

        // a - b == (-b) + a
        fp_sub_8x1(d, a, b);
        fp_neg_8x1(e, b);
        fp_add_8x1(f, e, a);
        if (fp_is_equal_8x1(d, f) != 0xff) {
            passed = 0;
            break;
        }

        // a-b == -(b-a)
        fp_sub_8x1(d, a, b);
        fp_sub_8x1(e, b, a);
        fp_neg_8x1(f, e);
        if (fp_is_equal_8x1(d, f) != 0xff) {
            passed = 0;
            break;
        }

        // -0 == 0
        fp_neg_8x1(d, zero);
        if (fp_is_zero_8x1(d) != 0xff) {
            passed = 0;
            break;
        }

    }
    print_test_res("negation", passed);
    if (passed == 0) {
        return false;
    }

    // Mixed subtraction & addition
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp_8x1(a);
        rand_fp_8x1(b);

        fp_mixaddsub_8x1(d, a, b, 0xaa);
        fp_sub_8x1(e, a, b);
        fp_add_8x1(f, a, b);
        if(fp_is_equal_8x1(d, e) != 0x55 || fp_is_equal_8x1(d, f) != 0xaa) {
            passed = 0;
            break;
        }

        fp_mixaddsub_8x1(d, a, b, 0xcc);
        fp_sub_8x1(e, a, b);
        fp_add_8x1(f, a, b);
        if(fp_is_equal_8x1(d, e) != 0x33 || fp_is_equal_8x1(d, f) != 0xcc) {
            passed = 0;
            break;
        }
    }
    print_test_res("mixed add sub", passed);
    if (passed == 0) {
        return false;
    }

    // Multiplication tests
    // Using Montgomery constant
    __m512i vR[8], vR2[8];
    fp_duplicate_8x1(vR, R);
    fp_duplicate_8x1(vR2, R2);

    // Montgomery representation of 1 is R mod p
    fp_tomont_8x1(d, one);
    if (fp_is_equal_8x1(d, vR) != 0xff) {
        passed = 0;
    }

    // Montgomery representation of R is R2 mod p
    fp_tomont_8x1(d, vR);
    if (fp_is_equal_8x1(d, vR2) != 0xff) {
        passed = 0;
    }

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp_8x1(a);
        rand_fp_8x1(b);
        rand_fp_8x1(c);

        //  a * 0  == 0
        fp_mul_8x1(d, a, zero);
        if (fp_is_zero_8x1(d) != 0xff) {
            passed = 0;
            break;
        }

        // (a * R)/R == a
        fp_mul_8x1(d, a, vR);
        if (fp_is_equal_8x1(d, a) != 0xff) {
            passed = 0;
            break;
        }

        // Transforms cancel
        fp_tomont_8x1(d, a);
        fp_frommont_8x1(e, d);
        if (fp_is_equal_8x1(a, e) != 0xff) {
            passed = 0;
            break;
        }

        fp_frommont_8x1(d, a);
        fp_tomont_8x1(e, d);
        if (fp_is_equal_8x1(a, e) != 0xff) {
            passed = 0;
            break;
        }

        // a * b == b * a
        fp_mul_8x1(d, a, b);
        fp_mul_8x1(e, b, a);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // a * (b * c) == (a * b) * c
        fp_mul_8x1(d, b, c);
        fp_mul_8x1(d, d, a);
        fp_mul_8x1(e, a, b);
        fp_mul_8x1(e, e, c);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // a * (b+c) == a*b + a*c
        fp_add_8x1(d, b, c);
        fp_mul_8x1(d, d, a);
        fp_mul_8x1(e, a, b),
        fp_mul_8x1(f, a, c);
        fp_add_8x1(e, e, f);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

    }
    print_test_res("multiplication", passed);
    if (passed == 0) {
        return false;
    }

    // Squaring tests
    fp_sqr_8x1(d, vR);
    if (fp_is_equal_8x1(d, vR) != 0xff) {
        passed = 0;
    }

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp_8x1(a);
        rand_fp_8x1(b);
        rand_fp_8x1(c);

        // 0^2 == 0
        fp_sqr_8x1(d, zero);
        if (fp_is_zero_8x1(d) != 0xff) {
            passed = 0;
            break;
        }

        // a^2 = a * a
        fp_sqr_8x1(d, a);
        fp_mul_8x1(e, a, a);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // (a+b)^2 = a^2 + 2ab + b^2
        fp_add_8x1(d, a, b);
        fp_sqr_8x1(d, d);
        fp_sqr_8x1(e, a);
        fp_sqr_8x1(f, b);
        fp_add_8x1(e, e, f);
        fp_mul_8x1(f, a, b);
        fp_add_8x1(f, f, f);
        fp_add_8x1(e, e, f);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }

    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

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

        // Top 16 bits of each AVX limb is 0
        t = VSETALL(0xffff000000000000);
        fp_unvectorized_to_8x1(d, unvec1);
        for (int j = 0; j < 8; j++) {
            e[j] = VAND(d[j], t);
        }
        if (fp_is_zero_8x1(e) != 0xff) {
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

    // Division by 3
    for (int i = 0; i < iterations; i++) {
        rand_fp_8x1(a);
        fp_copy_8x1(b, a);

        fp_div3_8x1(b, b);
        fp_add_8x1(c, b, b);
        fp_add_8x1(c, c, b);

        if (fp_is_equal_8x1(a, c) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 3", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp lvl3 4x2 tests:\n");
    one[0] = VSET(1, 0, 1, 0, 1, 0, 1, 0);

    // Equality test
    for(int i = 0; i < iterations; i++){
        rand_fp_4x2(a);
        if (fp_is_equal_4x2(a, a) != 0xf) {
            passed = 0;
            break;
        }
        fp_copy_4x2(b, a);
        if (fp_is_equal_4x2(a, b) != 0xf) {
            passed = 0;
            break;
        }

        // We check that making one change in each limb has the intended effect
        // b will be a copy of a, except that one limb of a is different
        // c will start out as a, but gradually be changed limb by limb.
        // d is the collection of vectors that we use to change each limb
        fp_copy_4x2(c, a);
        d[0] = VSET(0x1, 0, 0, 0, 0, 0, 0, 0);
        d[1] = VSET(0, 0x10, 0, 0, 0, 0, 0, 0);
        d[2] = VSET(0, 0, 0x100, 0, 0, 0, 0, 0);
        d[3] = VSET(0, 0, 0, 0x1000, 0, 0, 0, 0);
        d[4] = VSET(0, 0, 0, 0, 0x100000, 0, 0, 0);
        d[5] = VSET(0, 0, 0, 0, 0, 0x1000000, 0, 0);
        d[6] = VSET(0, 0, 0, 0, 0, 0, 0x100000000, 0);
        d[7] = VSET(0, 0, 0, 0, 0, 0, 0, 0x4000000000000);

        int mask_b = 0xe;
        int mask_c = 0xe;
        int multiplier = 1;
        int i_vec = 0, j_limb = 0;
        while(i_vec < 8) {
            b[j_limb] = VXOR(b[j_limb], d[i_vec]);
            c[j_limb] = VXOR(c[j_limb], d[i_vec]);

            int valid = (fp_is_equal_4x2(a, b) == mask_b) & (fp_is_equal_4x2(a,c) == mask_c);
            if (valid == 0) {
                passed = 0;
                break;
            }

            b[j_limb] = a[j_limb];
            if(j_limb == 3) {
                j_limb = 0;
                i_vec += 1;
                if(i_vec % 2 == 0)
                {
                    multiplier *= 2;
                    mask_b = 0xf - multiplier; // b should be equal to a, except at index i_vec
                    mask_c -= multiplier; // c should only be equal to a at the fp elements we haven't touched yet.
                }
            }
            else {
                j_limb += 1;
            }
        }
        if (passed == 0) {
            break;
        }
    }
    print_test_res("equality", passed);
    if (passed == 0) {
        return false;
    }

    // Addition test
    for(int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);
        rand_fp_4x2(b);
        rand_fp_4x2(c);

        // a + 0 == a
        fp_add_4x2(d, a, zero);
        if(fp_is_equal_4x2(a, d) != 0xf) {
            passed = 0;
            break;
        }

        // a + 1 != a
        fp_add_4x2(e, a, one);
        if (fp_is_equal_4x2(a, e) != 0) {
            passed = 0;
            break;
        }

        // a + b == b + a
        fp_add_4x2(d, a, b);
        fp_add_4x2(e, b, a);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // (a + b) + c = a + (b + c)
        fp_add_4x2(d, a, b);
        fp_add_4x2(d, d, c);
        fp_add_4x2(e, b, c);
        fp_add_4x2(e, e, a);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("addition", passed);
    if (passed == 0) {
        return false;
    }

    // Subtraction test
    for(int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);
        rand_fp_4x2(b);
        rand_fp_4x2(c);

        // a - 0 == a
        fp_sub_4x2(d, a, zero);
        if(fp_is_equal_4x2(a, d) != 0xf) {
            passed = 0;
            break;
        }

        // a - 1 != a
        fp_sub_4x2(e, a, one);
        if (fp_is_equal_4x2(a, e) != 0) {
            passed = 0;
            break;
        }

        // a - a == 0
        fp_sub_4x2(d, a, a);
        if (fp_is_zero_4x2(d) != 0xf) {
            passed = 0;
            break;
        }

        // (a - b) - c = a - (b + c)
        fp_sub_4x2(d, a, b);
        fp_sub_4x2(d, d, c);
        fp_add_4x2(e, b, c);
        fp_sub_4x2(e, a, e);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("subtraction", passed);
    if (passed == 0) {
        return false;
    }

    // Negation tests
    for(int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);
        rand_fp_4x2(b);

        // (-a) == 0 - a
        fp_neg_4x2(d, a);
        fp_sub_4x2(e, zero, a);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // -(-a) == a
        fp_neg_4x2(d, a);
        fp_neg_4x2(e, d);
        if (fp_is_equal_4x2(e, a) != 0xf) {
            passed = 0;
            break;
        }

        // a + (-a) == 0
        fp_neg_4x2(d, a);
        fp_add_4x2(e, a, d);
        if (fp_is_zero_4x2(e) != 0xf) {
            passed = 0;
            break;
        }

        // a - b == (-b) + a
        fp_sub_4x2(d, a, b);
        fp_neg_4x2(e, b);
        fp_add_4x2(f, e, a);
        if (fp_is_equal_4x2(d, f) != 0xf) {
            passed = 0;
            break;
        }

        // a-b == -(b-a)
        fp_sub_4x2(d, a, b);
        fp_sub_4x2(e, b, a);
        fp_neg_4x2(f, e);
        if (fp_is_equal_4x2(d, f) != 0xf) {
            passed = 0;
            break;
        }

        // -0 == 0
        fp_neg_4x2(d, zero);
        if (fp_is_zero_4x2(d) != 0xf) {
            passed = 0;
            break;
        }

    }
    print_test_res("negation", passed);
    if (passed == 0) {
        return false;
    }

    // Mixed subtraction & addition
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp_4x2(a);
        rand_fp_4x2(b);

        fp_mixaddsub_4x2(d, a, b, 0xcc);
        fp_sub_4x2(e, a, b);
        fp_add_4x2(f, a, b);
        if(fp_is_equal_4x2(d, e) != 0x5 || fp_is_equal_4x2(d, f) != 0xa) {
            passed = 0;
            break;
        }
    }
    print_test_res("mixed add sub", passed);
    if (passed == 0) {
        return false;
    }

    // Multiplication tests
    // Using Montgomery constant
    fp_duplicate_4x2(vR, R);
    fp_duplicate_4x2(vR2, R2);

    // Montgomery representation of 1 is R mod p
    fp_tomont_4x2(d, one);
    if (fp_is_equal_4x2(d, vR) != 0xf) {
		print_fp_4x2_index(d, 0);
		print_fp_4x2_index(vR, 0);
        passed = 0;
    }

    // Montgomery representation of R is R2 mod p
    fp_tomont_4x2(d, vR);
    if (fp_is_equal_4x2(d, vR2) != 0xf) {
		print_fp_4x2_index(d, 0);
		print_fp_4x2_index(vR2, 0);
        passed = 0;
    }

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp_4x2(a);
        rand_fp_4x2(b);
        rand_fp_4x2(c);

        //  a * 0  == 0
        fp_mul_4x2(d, a, zero);
        if (fp_is_zero_4x2(d) != 0xf) {
            passed = 0;
            break;
        }

        // (a * R)/R == a
        fp_mul_4x2(d, a, vR);
        if (fp_is_equal_4x2(d, a) != 0xf) {
            passed = 0;
            break;
        }

        // Transforms cancel
        fp_tomont_4x2(d, a);
        fp_frommont_4x2(e, d);
        if (fp_is_equal_4x2(a, e) != 0xf) {
            //passed = 0;
            break;
        }

        fp_frommont_4x2(d, a);
        fp_tomont_4x2(e, d);
        if (fp_is_equal_4x2(a, e) != 0xf) {
            //passed = 0;
            break;
        }

        // a * b == b * a
        fp_mul_4x2(d, a, b);
        fp_mul_4x2(e, b, a);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            //passed = 0;
            break;
        }

        // a * (b * c) == (a * b) * c
        fp_mul_4x2(d, b, c);
        fp_mul_4x2(d, d, a);
        fp_mul_4x2(e, a, b);
        fp_mul_4x2(e, e, c);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            //passed = 0;
            break;
        }

        // a * (b+c) == a*b + a*c
        fp_add_4x2(d, b, c);
        fp_mul_4x2(d, d, a);
        fp_mul_4x2(e, a, b),
        fp_mul_4x2(f, a, c);
        fp_add_4x2(e, e, f);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            //passed = 0;
            break;
        }

    }
    print_test_res("multiplication", passed);
    if (passed == 0) {
        return false;
    }

    // Squaring tests
    fp_sqr_4x2(d, vR);
    if (fp_is_equal_4x2(d, vR) != 0xf) {
        passed = 0;
    }

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp_4x2(a);
        rand_fp_4x2(b);
        rand_fp_4x2(c);

        // 0^2 == 0
        fp_sqr_4x2(d, zero);
        if (fp_is_zero_4x2(d) != 0xf) {
            passed = 0;
            break;
        }

        // a^2 = a * a
        fp_sqr_4x2(d, a);
        fp_mul_4x2(e, a, a);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // (a+b)^2 = a^2 + 2ab + b^2
        fp_add_4x2(d, a, b);
        fp_sqr_4x2(d, d);
        fp_sqr_4x2(e, a);
        fp_sqr_4x2(f, b);
        fp_add_4x2(e, e, f);
        fp_mul_4x2(f, a, b);
        fp_add_4x2(f, f, f);
        fp_add_4x2(e, e, f);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            break;
        }
    }

    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 3
    for (int i = 0; i < iterations; i++) {
        rand_fp_8x1(a);
        fp_copy_8x1(b, a);

        fp_div3_8x1(b, b);
        fp_add_8x1(c, b, b);
        fp_add_8x1(c, c, b);

        if (fp_is_equal_8x1(a, c) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 3", passed);
    if (passed == 0) {
        return false;
    }

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

        // Top 13 bits of each AVX limb is 0
        t = VSETALL(0xfff8000000000000);
        fp_unvectorized_to_8x1(d, unvec1);
        for (int j = 0; j < 8; j++) {
            e[j] = VAND(d[j], t);
        }
        if (fp_is_zero_8x1(e) != 0xff) {
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
            break;
        }

        fp_4x2_to_unvectorized(unvec2, a);
        fp_unvectorized_to_4x2(d, unvec2);

        if (fp_is_equal_4x2(a, d) != 0xf) {
            passed = 0;
            break;
        }

        // Top 13 bits of each AVX limb is 0
        t = VSETALL(0xfff8000000000000);
        fp_unvectorized_to_8x1(d, unvec1);
        for (int j = 0; j < 4; j++) {
            e[j] = VAND(d[j], t);
        }
        if (fp_is_zero_4x2(e) != 0xf) {
            passed = 0;
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

    printf("Fp lvl3 2x4 tests:\n");

    // Addition
    for (int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);
        rand_fp_4x2(b);
        fp_4x2_to_2x4(c, c+2, a);
        fp_4x2_to_2x4(d, d+2, b);
        fp_add_2x4(c, c, d);
        fp_add_2x4(c+2, c+2, d+2);
        fp_2x4_to_4x2(c, c, c+2);
        fp_add_4x2(e, a, b);

        if (fp_is_equal_4x2(c, e) != 0xf) {
            print_fp_4x2_index(c, 0);
            print_fp_4x2_index(e, 0);
            passed = 0;
            break;
        }
    }
    print_test_res("addition", passed);
    if (passed == 0) {
        return false;
    }

    // Subtraction
    for (int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);
        rand_fp_4x2(b);
        fp_4x2_to_2x4(c, c+2, a);
        fp_4x2_to_2x4(d, d+2, b);
        fp_sub_2x4(c, c, d);
        fp_sub_2x4(c+2, c+2, d+2);
        fp_2x4_to_4x2(c, c, c+2);
        fp_sub_4x2(e, a, b);

        if (fp_is_equal_4x2(c, e) != 0xf) {
            print_fp_4x2_index(c, 0);
            print_fp_4x2_index(e, 0);
            passed = 0;
            break;
        }
    }
    print_test_res("subtraction", passed);
    if (passed == 0) {
        return false;
    }

    // Negation
    for (int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);

        fp_4x2_to_2x4(c, c+2, a);
        fp_neg_2x4(c, c);
        fp_neg_2x4(c+2, c+2);
        fp_2x4_to_4x2(c, c, c+2);
        fp_neg_4x2(e, a);

        if (fp_is_equal_4x2(c, e) != 0xf) {
            print_fp_4x2_index(c, 0);
            print_fp_4x2_index(e, 0);
            passed = 0;
            break;
        }
    }
    print_test_res("negation", passed);
    if (passed == 0) {
        return false;
    }

    // Mixed add sub
    for (int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);
        rand_fp_4x2(b);
        fp_4x2_to_2x4(c, c+2, a);
        fp_4x2_to_2x4(d, d+2, b);
        fp_mixaddsub_2x4(c, c, d, 0xff);
        fp_mixaddsub_2x4(c+2, c+2, d+2, 0x0);
        fp_2x4_to_4x2(c, c, c+2);
        fp_mixaddsub_4x2(e, a, b, 0x33);

        if (fp_is_equal_4x2(c, e) != 0xf) {
            print_fp_4x2_index(c, 0);
            print_fp_4x2_index(e, 0);
            passed = 0;
            break;
        }
    }
    print_test_res("mix add sub", passed);
    if (passed == 0) {
        return false;
    }

    // Multiplication tests
    // Using Montgomery constant
    one[0] = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    fp_duplicate_2x4(vR, R);
    fp_duplicate_2x4(vR2, R2);

    // Montgomery representation of 1 is R mod p
    fp_tomont_2x4(d, one);
    if (fp_is_equal_2x4(d, vR) != 0x3) {
        passed = 0;
    }

    // Montgomery representation of R is R2 mod p
    fp_tomont_2x4(d, vR);
    if (fp_is_equal_2x4(d, vR2) != 0x3) {
        passed = 0;
    }

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp_2x4(a);
        rand_fp_2x4(b);
        rand_fp_2x4(c);

        //  a * 0  == 0
        fp_mul_2x4(d, a, zero);
        if (fp_is_zero_2x4(d) != 0x3) {
            passed = 0;
            break;
        }

        // (a * R)/R == a
        fp_mul_2x4(d, a, vR);
        if (fp_is_equal_2x4(d, a) != 0x3) {
            passed = 0;
            break;
        }

        // Transforms cancel
        fp_tomont_2x4(d, a);
        fp_frommont_2x4(e, d);
        if (fp_is_equal_2x4(a, e) != 0x3) {
            passed = 0;
            break;
        }

        fp_frommont_2x4(d, a);
        fp_tomont_2x4(e, d);
        if (fp_is_equal_2x4(a, e) != 0x3) {
            passed = 0;
            break;
        }

        // a * b == b * a
        fp_mul_2x4(d, a, b);
        fp_mul_2x4(e, b, a);
        if (fp_is_equal_2x4(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // a * (b * c) == (a * b) * c
        fp_mul_2x4(d, b, c);
        fp_mul_2x4(d, d, a);
        fp_mul_2x4(e, a, b);
        fp_mul_2x4(e, e, c);
        if (fp_is_equal_2x4(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // a * (b+c) == a*b + a*c
        fp_add_2x4(d, b, c);
        fp_mul_2x4(d, d, a);
        fp_mul_2x4(e, a, b),
        fp_mul_2x4(f, a, c);
        fp_add_2x4(e, e, f);
        if (fp_is_equal_2x4(d, e) != 0x3) {
            passed = 0;
            break;
        }

    }
    print_test_res("multiplication", passed);
    if (passed == 0) {
        return false;
    }

    // Squaring
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp_2x4(a);

        //  a^2  == a * a
        fp_sqr_2x4(c, a);
        fp_mul_2x4(d, a, a);
        if (fp_is_equal_2x4(c,d) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    // Square root
    for (int i = 0; i < iterations; i++) {
        rand_fp_2x4(a);
        
        fp_sqr_2x4(b, a);
        fp_sqrt_2x4(c, b);
        fp_sqr_2x4(d, c);

        if (fp_is_equal_2x4(b, d) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("sqrt", passed);
    if (passed == 0) {
        return false;
    }

    for (int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);

        // back and forth does nothing
        fp_4x2_to_2x4(b, b+2, a);
        fp_2x4_to_4x2(b, b, b+2);

        if (fp_is_equal_4x2(b, a) != 0xf) {
            passed = 0;
            break;
        }

        rand_fp_2x4(a);
        rand_fp_2x4(a+2);
        fp_2x4_to_4x2(b, a, a+2);
        fp_4x2_to_2x4(b, b+2, b);

        if (fp_is_equal_2x4(b, a) != 0x3 || fp_is_equal_2x4(b+2, a+2) != 0x3) {
            passed = 0;
            break;
        }

        rand_fp_4x2(a);
        rand_fp_4x2(b);
        fp_mul_4x2(c, a, b);
        fp_4x2_to_2x4(d, d+2, a);
        fp_4x2_to_2x4(e, e+2, b);
        fp_mul_2x4(d, d, e);
        fp_mul_2x4(d+2, d+2, e+2);
        fp_2x4_to_4x2(d, d, d+2);

        if (fp_is_equal_4x2(c, d) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("2x4 <-> 4x2", passed);
    if (passed == 0) {
        return false;
    }
    
    return true;
}
