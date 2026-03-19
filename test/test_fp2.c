#include <stdio.h>

#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "test.h"
#include "utils.h"
#include "sqi_test_extras.h"

bool test_fp2(int iterations)
{
    __m512i a[16], b[16], c[16], d[16], e[16], f[16], zero[16], one[16], t;
    fp2_setzero_8x1x1(zero);
    fp2_setzero_8x1x1(one);

    //TODO: This is not one in the other representations, fix the tests below
    one[0] = VSETALL(1);
    fp2_tomont_8x1x1(one, one);

    int passed = 1;

    printf("Fp2 lvl3 8x1x1 tests:\n");

    // Equality test
    for(int i = 0; i < iterations; i++){
        rand_fp2_8x1x1(a);
        if (fp2_is_equal_8x1x1(a, a) != 0xff) {
            passed = 0;
            break;
        }
        fp2_copy_8x1x1(b, a);
        if (fp2_is_equal_8x1x1(a, b) != 0xff) {
            passed = 0;
            break;
        }

        fp2_copy_8x1x1(c, a);
        t = VSET(0, 0, 1, 0, 0, 0, 0, 0);
        c[9] = VADD(c[9], t);
        if (fp2_is_equal_8x1x1(a,c) != 0xfb) {
            passed = 0;
            break;
        }
    }
    print_test_res("equality", passed);
    if (passed == 0) {
        return false;
    }

    // Addition test
    for(int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        rand_fp2_8x1x1(c);

        // a + 0 == a
        fp2_add_8x1x1(d, a, zero);
        if(fp2_is_equal_8x1x1(a, d) != 0xff) {
            passed = 0;
            break;
        }

        // a + 1 != a
        fp2_add_8x1x1(e, a, one);
        if (fp2_is_equal_8x1x1(a, e) != 0) {
            passed = 0;
            break;
        }

        // a + b == b + a
        fp2_add_8x1x1(d, a, b);
        fp2_add_8x1x1(e, b, a);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // (a + b) + c = a + (b + c)
        fp2_add_8x1x1(d, a, b);
        fp2_add_8x1x1(d, d, c);
        fp2_add_8x1x1(e, b, c);
        fp2_add_8x1x1(e, e, a);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
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
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        rand_fp2_8x1x1(c);

        // a - 0 == a
        fp2_sub_8x1x1(d, a, zero);
        if(fp2_is_equal_8x1x1(a, d) != 0xff) {
            passed = 0;
            break;
        }

        // a - 1 != a
        fp2_sub_8x1x1(e, a, one);
        if (fp2_is_equal_8x1x1(a, e) != 0) {
            passed = 0;
            break;
        }

        // a - a == 0
        fp2_sub_8x1x1(d, a, a);
        if (fp2_is_zero_8x1x1(d) != 0xff) {
            passed = 0;
            break;
        }

        // (a - b) - c = a - (b + c)
        fp2_sub_8x1x1(d, a, b);
        fp2_sub_8x1x1(d, d, c);
        fp2_add_8x1x1(e, b, c);
        fp2_sub_8x1x1(e, a, e);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
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
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        rand_fp2_8x1x1(c);

        // (-a) == 0 - a
        fp2_neg_8x1x1(d, a);
        fp2_sub_8x1x1(e, zero, a);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // -(-a) == a
        fp2_neg_8x1x1(d, a);
        fp2_neg_8x1x1(e, d);
        if (fp2_is_equal_8x1x1(e, a) != 0xff) {
            passed = 0;
            break;
        }

        // a + (-a) == 0
        fp2_neg_8x1x1(d, a);
        fp2_add_8x1x1(e, a, d);
        if (fp2_is_zero_8x1x1(e) != 0xff) {
            passed = 0;
            break;
        }

        // a - b == (-b) + a
        fp2_sub_8x1x1(d, a, b);
        fp2_neg_8x1x1(e, b);
        fp2_add_8x1x1(f, e, a);
        if (fp2_is_equal_8x1x1(d, f) != 0xff) {
            passed = 0;
            break;
        }

        // a-b == -(b-a)
        fp2_sub_8x1x1(d, a, b);
        fp2_sub_8x1x1(e, b, a);
        fp2_neg_8x1x1(f, e);
        if (fp2_is_equal_8x1x1(d, f) != 0xff) {
            passed = 0;
            break;
        }

        // -0 == 0
        fp2_neg_8x1x1(d, zero);
        if (fp2_is_zero_8x1x1(d) != 0xff) {
            passed = 0;
            break;
        }

    }
    print_test_res("negation", passed);
    if (passed == 0) {
        return false;
    }

    // Multiplication tests
    // Using Montgomery constant
    uint64_t R[8] = { 0xF03F03F03F03, 0x03, 0x0, 0x0, 0x0, 0x0, 0x0, 0x3D0000000000 };
    __m512i vR[16];
    fp2_setzero_8x1x1(vR);
    fp_duplicate_8x1(vR, R);

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        rand_fp2_8x1x1(c);

        //  a * 0  == 0
        fp2_mul_8x1x1(d, a, zero);
        if (fp2_is_zero_8x1x1(d) != 0xff) {
            passed = 0;
            break;
        }

        // (a * R)/R == a
        fp2_mul_8x1x1(d, a, vR);
        if (fp2_is_equal_8x1x1(d, a) != 0xff) {
            passed = 0;
            break;
        }

        // Transforms cancel
        fp2_tomont_8x1x1(d, a);
        fp2_frommont_8x1x1(e, d);
        if (fp2_is_equal_8x1x1(a, e) != 0xff) {
            passed = 0;
            break;
        }

        fp2_frommont_8x1x1(d, a);
        fp2_tomont_8x1x1(e, d);
        if (fp2_is_equal_8x1x1(a, e) != 0xff) {
            passed = 0;
            break;
        }

        // a * b == b * a
        fp2_mul_8x1x1(d, a, b);
        fp2_mul_8x1x1(e, b, a);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // a * (b * c) == (a * b) * c
        fp2_mul_8x1x1(d, b, c);
        fp2_mul_8x1x1(d, d, a);
        fp2_mul_8x1x1(e, a, b);
        fp2_mul_8x1x1(e, e, c);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // a * (b+c) == a*b + a*c
        fp2_add_8x1x1(d, b, c);
        fp2_mul_8x1x1(d, d, a);
        fp2_mul_8x1x1(e, a, b),
        fp2_mul_8x1x1(f, a, c);
        fp2_add_8x1x1(e, e, f);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

    }
    print_test_res("multiplication", passed);
    if (passed == 0) {
        return false;
    }

    // Squaring tests
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        rand_fp2_8x1x1(c);

        // 0^2 == 0
        fp2_sqr_8x1x1(d, zero);
        if (fp2_is_zero_8x1x1(d) != 0xff) {
            passed = 0;
            break;
        }

        // a^2 = a * a
        fp2_sqr_8x1x1(d, a);
        fp2_mul_8x1x1(e, a, a);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // (a+b)^2 = a^2 + 2ab + b^2
        fp2_add_8x1x1(d, a, b);
        fp2_sqr_8x1x1(d, d);
        fp2_sqr_8x1x1(e, a);
        fp2_sqr_8x1x1(f, b);
        fp2_add_8x1x1(e, e, f);
        fp2_mul_8x1x1(f, a, b);
        fp2_add_8x1x1(f, f, f);
        fp2_add_8x1x1(e, e, f);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }

    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    // Multiplication by i
    
    // Preparing i
    fp2_setzero_8x1x1(a);
    a[8] = VSETALL(1);
    fp2_tomont_8x1x1(a, a);

    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(b);
        
        fp2_muli_8x1x1(c, b);
        fp2_mul_8x1x1(d, a, b);

        if (fp2_is_equal_8x1x1(c, d) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("mult by i", passed);
    if (passed == 0) {
        return false;
    }

/*

    // Division by 3
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        fp2_copy_8x1x1(b, a);

        fp2_div3_8x1x1(b, b);
        fp2_add_8x1x1(c, b, b);
        fp2_add_8x1x1(c, c, b);

        if (fp2_is_equal_8x1x1(a, c) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 3", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 lvl3 4x1x2 tests:\n");

    // Equality test
    for(int i = 0; i < iterations; i++){
        rand_fp2_4x1x2(a);
        if (fp2_is_equal_4x1x2(a, a) != 0xf) {
            passed = 0;
            break;
        }
        fp2_copy_4x1x2(b, a);
        if (fp2_is_equal_4x1x2(a, b) != 0xf) {
            passed = 0;
            break;
        }

        fp2_copy_4x1x2(c, a);
        t = VSET(0, 0, 1, 0, 0, 0, 0, 0);
        c[5] = VADD(c[5], t);
        if (fp2_is_equal_4x1x2(a,c) != 0xd) {
            passed = 0;
            break;
        }
    }
    print_test_res("equality", passed);
    if (passed == 0) {
        return false;
    }

    // Addition test
    for(int i = 0; i < iterations; i++) {
        rand_fp2_4x1x2(a);
        rand_fp2_4x1x2(b);
        rand_fp2_4x1x2(c);

        // a + 0 == a
        fp2_add_4x1x2(d, a, zero);
        if(fp2_is_equal_4x1x2(a, d) != 0xf) {
            passed = 0;
            break;
        }

        // a + 1 != a
        fp2_add_4x1x2(e, a, one);
        if (fp2_is_equal_4x1x2(a, e) != 0) {
            passed = 0;
            break;
        }

        // a + b == b + a
        fp2_add_4x1x2(d, a, b);
        fp2_add_4x1x2(e, b, a);
        if (fp2_is_equal_4x1x2(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // (a + b) + c = a + (b + c)
        fp2_add_4x1x2(d, a, b);
        fp2_add_4x1x2(d, d, c);
        fp2_add_4x1x2(e, b, c);
        fp2_add_4x1x2(e, e, a);
        if (fp2_is_equal_4x1x2(d, e) != 0xf) {
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
        rand_fp2_4x1x2(a);
        rand_fp2_4x1x2(b);
        rand_fp2_4x1x2(c);

        // a - 0 == a
        fp2_sub_4x1x2(d, a, zero);
        if(fp2_is_equal_4x1x2(a, d) != 0xf) {
            passed = 0;
            break;
        }

        // a - 1 != a
        fp2_sub_4x1x2(e, a, one);
        if (fp2_is_equal_4x1x2(a, e) != 0) {
            passed = 0;
            break;
        }

        // a - a == 0
        fp2_sub_4x1x2(d, a, a);
        if (fp2_is_zero_4x1x2(d) != 0xf) {
            passed = 0;
            break;
        }

        // (a - b) - c = a - (b + c)
        fp2_sub_4x1x2(d, a, b);
        fp2_sub_4x1x2(d, d, c);
        fp2_add_4x1x2(e, b, c);
        fp2_sub_4x1x2(e, a, e);
        if (fp2_is_equal_4x1x2(d, e) != 0xf) {
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
        rand_fp2_4x1x2(a);
        rand_fp2_4x1x2(b);
        rand_fp2_4x1x2(c);

        // (-a) == 0 - a
        fp2_neg_4x1x2(d, a);
        fp2_sub_4x1x2(e, zero, a);
        if (fp2_is_equal_4x1x2(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // -(-a) == a
        fp2_neg_4x1x2(d, a);
        fp2_neg_4x1x2(e, d);
        if (fp2_is_equal_4x1x2(e, a) != 0xf) {
            passed = 0;
            break;
        }

        // a + (-a) == 0
        fp2_neg_4x1x2(d, a);
        fp2_add_4x1x2(e, a, d);
        if (fp2_is_zero_4x1x2(e) != 0xf) {
            passed = 0;
            break;
        }

        // a - b == (-b) + a
        fp2_sub_4x1x2(d, a, b);
        fp2_neg_4x1x2(e, b);
        fp2_add_4x1x2(f, e, a);
        if (fp2_is_equal_4x1x2(d, f) != 0xf) {
            passed = 0;
            break;
        }

        // a-b == -(b-a)
        fp2_sub_4x1x2(d, a, b);
        fp2_sub_4x1x2(e, b, a);
        fp2_neg_4x1x2(f, e);
        if (fp2_is_equal_4x1x2(d, f) != 0xf) {
            passed = 0;
            break;
        }

        // -0 == 0
        fp2_neg_4x1x2(d, zero);
        if (fp2_is_zero_4x1x2(d) != 0xf) {
            passed = 0;
            break;
        }

    }
    print_test_res("negation", passed);
    if (passed == 0) {
        return false;
    }

    // Multiplication tests
    // Using Montgomery constant
    fp2_setzero_4x1x2(vR);
    fp_duplicate_4x2(vR, R);

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_4x1x2(a);
        rand_fp2_4x1x2(b);
        rand_fp2_4x1x2(c);

        //  a * 0  == 0
        fp2_mul_4x1x2(d, a, zero);
        if (fp2_is_zero_4x1x2(d) != 0xf) {
            passed = 0;
            break;
        }

        // (a * R)/R == a
        fp2_mul_4x1x2(d, a, vR);
        if (fp2_is_equal_4x1x2(d, a) != 0xf) {
            passed = 0;
            break;
        }

        // Transforms cancel
        fp2_tomont_4x1x2(d, a);
        fp2_frommont_4x1x2(e, d);
        if (fp2_is_equal_4x1x2(a, e) != 0xf) {
            passed = 0;
            break;
        }

        fp2_frommont_4x1x2(d, a);
        fp2_tomont_4x1x2(e, d);
        if (fp2_is_equal_4x1x2(a, e) != 0xf) {
            passed = 0;
            break;
        }

        // a * b == b * a
        fp2_mul_4x1x2(d, a, b);
        fp2_mul_4x1x2(e, b, a);
        if (fp2_is_equal_4x1x2(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // a * (b * c) == (a * b) * c
        fp2_mul_4x1x2(d, b, c);
        fp2_mul_4x1x2(d, d, a);
        fp2_mul_4x1x2(e, a, b);
        fp2_mul_4x1x2(e, e, c);
        if (fp2_is_equal_4x1x2(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // a * (b+c) == a*b + a*c
        fp2_add_4x1x2(d, b, c);
        fp2_mul_4x1x2(d, d, a);
        fp2_mul_4x1x2(e, a, b),
        fp2_mul_4x1x2(f, a, c);
        fp2_add_4x1x2(e, e, f);
        if (fp2_is_equal_4x1x2(d, e) != 0xf) {
            passed = 0;
            break;
        }

    }
    print_test_res("multiplication", passed);
    if (passed == 0) {
        return false;
    }

    // Squaring tests
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_4x1x2(a);
        rand_fp2_4x1x2(b);
        rand_fp2_4x1x2(c);

        // 0^2 == 0
        fp2_sqr_4x1x2(d, zero);
        if (fp2_is_zero_4x1x2(d) != 0xf) {
            passed = 0;
            break;
        }

        // a^2 = a * a
        fp2_sqr_4x1x2(d, a);
        fp2_mul_4x1x2(e, a, a);
        if (fp2_is_equal_4x1x2(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // (a+b)^2 = a^2 + 2ab + b^2
        fp2_add_4x1x2(d, a, b);
        fp2_sqr_4x1x2(d, d);
        fp2_sqr_4x1x2(e, a);
        fp2_sqr_4x1x2(f, b);
        fp2_add_4x1x2(e, e, f);
        fp2_mul_4x1x2(f, a, b);
        fp2_add_4x1x2(f, f, f);
        fp2_add_4x1x2(e, e, f);
        if (fp2_is_equal_4x1x2(d, e) != 0xf) {
            passed = 0;
            break;
        }
    }

    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 lvl3 4x2x1 tests:\n");

    // Equality test
    for(int i = 0; i < iterations; i++){
        rand_fp2_4x2x1(a);
        if (fp2_is_equal_4x2x1(a, a) != 0xf) {
            passed = 0;
            break;
        }
        fp2_copy_4x2x1(b, a);
        if (fp2_is_equal_4x2x1(a, b) != 0xf) {
            passed = 0;
            break;
        }

        fp2_copy_4x2x1(c, a);
        t = VSET(0, 0, 1, 0, 0, 0, 0, 0);
        c[5] = VADD(c[5], t);
        if (fp2_is_equal_4x2x1(a,c) != 0xd) {
            passed = 0;
            break;
        }
    }
    print_test_res("equality", passed);
    if (passed == 0) {
        return false;
    }

    // Addition test
    for(int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(c);

        // a + 0 == a
        fp2_add_4x2x1(d, a, zero);
        if(fp2_is_equal_4x2x1(a, d) != 0xf) {
            passed = 0;
            break;
        }

        // a + 1 != a
        fp2_add_4x2x1(e, a, one);
        if (fp2_is_equal_4x2x1(a, e) != 0) {
            passed = 0;
            break;
        }

        // a + b == b + a
        fp2_add_4x2x1(d, a, b);
        fp2_add_4x2x1(e, b, a);
        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // (a + b) + c = a + (b + c)
        fp2_add_4x2x1(d, a, b);
        fp2_add_4x2x1(d, d, c);
        fp2_add_4x2x1(e, b, c);
        fp2_add_4x2x1(e, e, a);
        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
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
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(c);

        // a - 0 == a
        fp2_sub_4x2x1(d, a, zero);
        if(fp2_is_equal_4x2x1(a, d) != 0xf) {
            passed = 0;
            break;
        }

        // a - 1 != a
        fp2_sub_4x2x1(e, a, one);
        if (fp2_is_equal_4x2x1(a, e) != 0) {
            passed = 0;
            break;
        }

        // a - a == 0
        fp2_sub_4x2x1(d, a, a);
        if (fp2_is_zero_4x2x1(d) != 0xf) {
            passed = 0;
            break;
        }

        // (a - b) - c = a - (b + c)
        fp2_sub_4x2x1(d, a, b);
        fp2_sub_4x2x1(d, d, c);
        fp2_add_4x2x1(e, b, c);
        fp2_sub_4x2x1(e, a, e);
        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
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
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(c);

        // (-a) == 0 - a
        fp2_neg_4x2x1(d, a);
        fp2_sub_4x2x1(e, zero, a);
        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // -(-a) == a
        fp2_neg_4x2x1(d, a);
        fp2_neg_4x2x1(e, d);
        if (fp2_is_equal_4x2x1(e, a) != 0xf) {
            passed = 0;
            break;
        }

        // a + (-a) == 0
        fp2_neg_4x2x1(d, a);
        fp2_add_4x2x1(e, a, d);
        if (fp2_is_zero_4x2x1(e) != 0xf) {
            passed = 0;
            break;
        }

        // a - b == (-b) + a
        fp2_sub_4x2x1(d, a, b);
        fp2_neg_4x2x1(e, b);
        fp2_add_4x2x1(f, e, a);
        if (fp2_is_equal_4x2x1(d, f) != 0xf) {
            passed = 0;
            break;
        }

        // a-b == -(b-a)
        fp2_sub_4x2x1(d, a, b);
        fp2_sub_4x2x1(e, b, a);
        fp2_neg_4x2x1(f, e);
        if (fp2_is_equal_4x2x1(d, f) != 0xf) {
            passed = 0;
            break;
        }

        // -0 == 0
        fp2_neg_4x2x1(d, zero);
        if (fp2_is_zero_4x2x1(d) != 0xf) {
            passed = 0;
            break;
        }

    }
    print_test_res("negation", passed);
    if (passed == 0) {
        return false;
    }

    // Multiplication tests
    // Using Montgomery constant
    fp2_setzero_4x2x1(vR);
    for(int i = 0; i < 8; i++) {
        vR[i] = VSET(R[i], 0, R[i], 0, R[i], 0, R[i], 0);
    }

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(c);

        //  a * 0  == 0
        fp2_mul_4x2x1(d, a, zero);
        if (fp2_is_zero_4x2x1(d) != 0xf) {
            passed = 0;
            break;
        }

        // (a * R)/R == a
        fp2_mul_4x2x1(d, a, vR);
        if (fp2_is_equal_4x2x1(d, a) != 0xf) {
            passed = 0;
            break;
        }

        // Transforms cancel
        fp2_tomont_4x2x1(d, a);
        fp2_frommont_4x2x1(e, d);
        if (fp2_is_equal_4x2x1(a, e) != 0xf) {
            passed = 0;
            break;
        }

        fp2_frommont_4x2x1(d, a);
        fp2_tomont_4x2x1(e, d);
        if (fp2_is_equal_4x2x1(a, e) != 0xf) {
            passed = 0;
            break;
        }

        // a * b == b * a
        fp2_mul_4x2x1(d, a, b);
        fp2_mul_4x2x1(e, b, a);
        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // a * (b * c) == (a * b) * c
        fp2_mul_4x2x1(d, b, c);
        fp2_mul_4x2x1(d, d, a);
        fp2_mul_4x2x1(e, a, b);
        fp2_mul_4x2x1(e, e, c);
        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // a * (b+c) == a*b + a*c
        fp2_add_4x2x1(d, b, c);
        fp2_mul_4x2x1(d, d, a);
        fp2_mul_4x2x1(e, a, b),
        fp2_mul_4x2x1(f, a, c);
        fp2_add_4x2x1(e, e, f);
        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
            passed = 0;
            break;
        }

    }
    print_test_res("multiplication", passed);
    if (passed == 0) {
        return false;
    }

    // Squaring tests
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(c);

        // 0^2 == 0
        fp2_sqr_4x2x1(d, zero);
        if (fp2_is_zero_4x2x1(d) != 0xf) {
            passed = 0;
            break;
        }

        // a^2 = a * a
        fp2_sqr_4x2x1(d, a);
        fp2_mul_4x2x1(e, a, a);
        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // (a+b)^2 = a^2 + 2ab + b^2
        fp2_add_4x2x1(d, a, b);
        fp2_sqr_4x2x1(d, d);
        fp2_sqr_4x2x1(e, a);
        fp2_sqr_4x2x1(f, b);
        fp2_add_4x2x1(e, e, f);
        fp2_mul_4x2x1(f, a, b);
        fp2_add_4x2x1(f, f, f);
        fp2_add_4x2x1(e, e, f);
        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
            passed = 0;
            break;
        }
    }

    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    // Multiplication by i
    
    // Preparing i
    fp2_setzero_4x2x1(a);
    a[0] = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    fp2_tomont_4x2x1(a, a);

    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(b);
        
        fp2_muli_4x2x1(c, b);
        fp2_mul_4x2x1(d, a, b);

        if (fp2_is_equal_4x2x1(c, d) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("mult by i", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 3
    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(a);
        fp2_copy_4x2x1(b, a);

        fp2_div3_4x2x1(b, b);
        fp2_add_4x2x1(c, b, b);
        fp2_add_4x2x1(c, c, b);

        if (fp2_is_equal_4x2x1(a, c) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 3", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 lvl3 2x2x2 tests:\n");

    // Equality test
    for(int i = 0; i < iterations; i++){
        rand_fp2_2x2x2(a);
        if (fp2_is_equal_2x2x2(a, a) != 0x3) {
            passed = 0;
            break;
        }
        fp2_copy_2x2x2(b, a);
        if (fp2_is_equal_2x2x2(a, b) != 0x3) {
            passed = 0;
            break;
        }

        fp2_copy_2x2x2(c, a);
        t = VSET(0, 0, 1, 0, 0, 0, 0, 0);
        c[2] = VADD(c[2], t);
        if (fp2_is_equal_2x2x2(a,c) != 0x2) {
            passed = 0;
            break;
        }

        fp2_copy_2x2x2(c, a);
        t = VSET(0, 0, 0, 0, 0, 0, 1, 0);
        c[0] = VADD(c[0], t);
        if (fp2_is_equal_2x2x2(a,c) != 0x1) {
            passed = 0;
            break;
        }
    }
    print_test_res("equality", passed);
    if (passed == 0) {
        return false;
    }

    // Addition test
    for(int i = 0; i < iterations; i++) {
        rand_fp2_2x2x2(a);
        rand_fp2_2x2x2(b);
        rand_fp2_2x2x2(c);

        // a + 0 == a
        fp2_add_2x2x2(d, a, zero);
        if(fp2_is_equal_2x2x2(a, d) != 0x3) {
            passed = 0;
            break;
        }

        // a + 1 != a
        fp2_add_2x2x2(e, a, one);
        if (fp2_is_equal_2x2x2(a, e) != 0) {
            passed = 0;
            break;
        }

        // a + b == b + a
        fp2_add_2x2x2(d, a, b);
        fp2_add_2x2x2(e, b, a);
        if (fp2_is_equal_2x2x2(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // (a + b) + c = a + (b + c)
        fp2_add_2x2x2(d, a, b);
        fp2_add_2x2x2(d, d, c);
        fp2_add_2x2x2(e, b, c);
        fp2_add_2x2x2(e, e, a);
        if (fp2_is_equal_2x2x2(d, e) != 0x3) {
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
        rand_fp2_2x2x2(a);
        rand_fp2_2x2x2(b);
        rand_fp2_2x2x2(c);

        // a - 0 == a
        fp2_sub_2x2x2(d, a, zero);
        if(fp2_is_equal_2x2x2(a, d) != 0x3) {
            passed = 0;
            break;
        }

        // a - 1 != a
        fp2_sub_2x2x2(e, a, one);
        if (fp2_is_equal_2x2x2(a, e) != 0) {
            passed = 0;
            break;
        }

        // a - a == 0
        fp2_sub_2x2x2(d, a, a);
        if (fp2_is_zero_2x2x2(d) != 0x3) {
            passed = 0;
            break;
        }

        // (a - b) - c = a - (b + c)
        fp2_sub_2x2x2(d, a, b);
        fp2_sub_2x2x2(d, d, c);
        fp2_add_2x2x2(e, b, c);
        fp2_sub_2x2x2(e, a, e);
        if (fp2_is_equal_2x2x2(d, e) != 0x3) {
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
        rand_fp2_2x2x2(a);
        rand_fp2_2x2x2(b);
        rand_fp2_2x2x2(c);

        // (-a) == 0 - a
        fp2_neg_2x2x2(d, a);
        fp2_sub_2x2x2(e, zero, a);
        if (fp2_is_equal_2x2x2(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // -(-a) == a
        fp2_neg_2x2x2(d, a);
        fp2_neg_2x2x2(e, d);
        if (fp2_is_equal_2x2x2(e, a) != 0x3) {
            passed = 0;
            break;
        }

        // a + (-a) == 0
        fp2_neg_2x2x2(d, a);
        fp2_add_2x2x2(e, a, d);
        if (fp2_is_zero_2x2x2(e) != 0x3) {
            passed = 0;
            break;
        }

        // a - b == (-b) + a
        fp2_sub_2x2x2(d, a, b);
        fp2_neg_2x2x2(e, b);
        fp2_add_2x2x2(f, e, a);
        if (fp2_is_equal_2x2x2(d, f) != 0x3) {
            passed = 0;
            break;
        }

        // a-b == -(b-a)
        fp2_sub_2x2x2(d, a, b);
        fp2_sub_2x2x2(e, b, a);
        fp2_neg_2x2x2(f, e);
        if (fp2_is_equal_2x2x2(d, f) != 0x3) {
            passed = 0;
            break;
        }

        // -0 == 0
        fp2_neg_2x2x2(d, zero);
        if (fp2_is_zero_2x2x2(d) != 0x3) {
            passed = 0;
            break;
        }

    }
    print_test_res("negation", passed);
    if (passed == 0) {
        return false;
    }

    // Multiplication tests
    // Using Montgomery constant
    fp2_setzero_2x2x2(vR);
    for(int i = 0; i < 4; i++) {
        vR[i] = VSET(R[i], R[i+4], 0, 0, R[i], R[i+4], 0, 0);
    }

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_2x2x2(a);
        rand_fp2_2x2x2(b);
        rand_fp2_2x2x2(c);

        //  a * 0  == 0
        fp2_mul_2x2x2(d, a, zero);
        if (fp2_is_zero_2x2x2(d) != 0x3) {
            passed = 0;
            break;
        }

        // (a * R)/R == a
        fp2_mul_2x2x2(d, a, vR);
        if (fp2_is_equal_2x2x2(d, a) != 0x3) {
            passed = 0;
            break;
        }

        // Transforms cancel
        fp2_tomont_2x2x2(d, a);
        fp2_frommont_2x2x2(e, d);
        if (fp2_is_equal_2x2x2(a, e) != 0x3) {
            passed = 0;
            break;
        }

        fp2_frommont_2x2x2(d, a);
        fp2_tomont_2x2x2(e, d);
        if (fp2_is_equal_2x2x2(a, e) != 0x3) {
            passed = 0;
            break;
        }

        // a * b == b * a
        fp2_mul_2x2x2(d, a, b);
        fp2_mul_2x2x2(e, b, a);
        if (fp2_is_equal_2x2x2(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // a * (b * c) == (a * b) * c
        fp2_mul_2x2x2(d, b, c);
        fp2_mul_2x2x2(d, d, a);
        fp2_mul_2x2x2(e, a, b);
        fp2_mul_2x2x2(e, e, c);
        if (fp2_is_equal_2x2x2(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // a * (b+c) == a*b + a*c
        fp2_add_2x2x2(d, b, c);
        fp2_mul_2x2x2(d, d, a);
        fp2_mul_2x2x2(e, a, b),
        fp2_mul_2x2x2(f, a, c);
        fp2_add_2x2x2(e, e, f);
        if (fp2_is_equal_2x2x2(d, e) != 0x3) {
            passed = 0;
            break;
        }

    }
    print_test_res("multiplication", passed);
    if (passed == 0) {
        return false;
    }

    // Squaring tests
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_2x2x2(a);
        rand_fp2_2x2x2(b);
        rand_fp2_2x2x2(c);

        // 0^2 == 0
        fp2_sqr_2x2x2(d, zero);
        if (fp2_is_zero_2x2x2(d) != 0x3) {
            passed = 0;
            break;
        }

        // a^2 = a * a
        fp2_sqr_2x2x2(d, a);
        fp2_mul_2x2x2(e, a, a);
        if (fp2_is_equal_2x2x2(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // (a+b)^2 = a^2 + 2ab + b^2
        fp2_add_2x2x2(d, a, b);
        fp2_sqr_2x2x2(d, d);
        fp2_sqr_2x2x2(e, a);
        fp2_sqr_2x2x2(f, b);
        fp2_add_2x2x2(e, e, f);
        fp2_mul_2x2x2(f, a, b);
        fp2_add_2x2x2(f, f, f);
        fp2_add_2x2x2(e, e, f);
        if (fp2_is_equal_2x2x2(d, e) != 0x3) {
            passed = 0;
            break;
        }
    }

    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 lvl3 2x4x1 tests:\n");

    // Equality test
    for(int i = 0; i < iterations; i++){
        rand_fp2_2x4x1(a);
        if (fp2_is_equal_2x4x1(a, a) != 0x3) {
            passed = 0;
            break;
        }
        fp2_copy_2x4x1(b, a);
        if (fp2_is_equal_2x4x1(a, b) != 0x3) {
            passed = 0;
            break;
        }

        fp2_copy_2x4x1(c, a);
        t = VSET(0, 0, 1, 0, 0, 0, 0, 0);
        c[2] = VADD(c[2], t);
        if (fp2_is_equal_2x4x1(a,c) != 0x2) {
            passed = 0;
            break;
        }

        fp2_copy_2x4x1(c, a);
        t = VSET(0, 0, 0, 0, 0, 0, 1, 0);
        c[0] = VADD(c[0], t);
        if (fp2_is_equal_2x4x1(a,c) != 0x1) {
            passed = 0;
            break;
        }
    }
    print_test_res("equality", passed);
    if (passed == 0) {
        return false;
    }

    // Addition test
    for(int i = 0; i < iterations; i++) {
        rand_fp2_2x4x1(a);
        rand_fp2_2x4x1(b);
        rand_fp2_2x4x1(c);

        // a + 0 == a
        fp2_add_2x4x1(d, a, zero);
        if(fp2_is_equal_2x4x1(a, d) != 0x3) {
            passed = 0;
            break;
        }

        // a + 1 != a
        fp2_add_2x4x1(e, a, one);
        if (fp2_is_equal_2x4x1(a, e) != 0) {
            passed = 0;
            break;
        }

        // a + b == b + a
        fp2_add_2x4x1(d, a, b);
        fp2_add_2x4x1(e, b, a);
        if (fp2_is_equal_2x4x1(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // (a + b) + c = a + (b + c)
        fp2_add_2x4x1(d, a, b);
        fp2_add_2x4x1(d, d, c);
        fp2_add_2x4x1(e, b, c);
        fp2_add_2x4x1(e, e, a);
        if (fp2_is_equal_2x4x1(d, e) != 0x3) {
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
        rand_fp2_2x4x1(a);
        rand_fp2_2x4x1(b);
        rand_fp2_2x4x1(c);

        // a - 0 == a
        fp2_sub_2x4x1(d, a, zero);
        if(fp2_is_equal_2x4x1(a, d) != 0x3) {
            passed = 0;
            break;
        }

        // a - 1 != a
        fp2_sub_2x4x1(e, a, one);
        if (fp2_is_equal_2x4x1(a, e) != 0) {
            passed = 0;
            break;
        }

        // a - a == 0
        fp2_sub_2x4x1(d, a, a);
        if (fp2_is_zero_2x4x1(d) != 0x3) {
            passed = 0;
            break;
        }

        // (a - b) - c = a - (b + c)
        fp2_sub_2x4x1(d, a, b);
        fp2_sub_2x4x1(d, d, c);
        fp2_add_2x4x1(e, b, c);
        fp2_sub_2x4x1(e, a, e);
        if (fp2_is_equal_2x4x1(d, e) != 0x3) {
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
        rand_fp2_2x4x1(a);
        rand_fp2_2x4x1(b);
        rand_fp2_2x4x1(c);

        // (-a) == 0 - a
        fp2_neg_2x4x1(d, a);
        fp2_sub_2x4x1(e, zero, a);
        if (fp2_is_equal_2x4x1(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // -(-a) == a
        fp2_neg_2x4x1(d, a);
        fp2_neg_2x4x1(e, d);
        if (fp2_is_equal_2x4x1(e, a) != 0x3) {
            passed = 0;
            break;
        }

        // a + (-a) == 0
        fp2_neg_2x4x1(d, a);
        fp2_add_2x4x1(e, a, d);
        if (fp2_is_zero_2x4x1(e) != 0x3) {
            passed = 0;
            break;
        }

        // a - b == (-b) + a
        fp2_sub_2x4x1(d, a, b);
        fp2_neg_2x4x1(e, b);
        fp2_add_2x4x1(f, e, a);
        if (fp2_is_equal_2x4x1(d, f) != 0x3) {
            passed = 0;
            break;
        }

        // a-b == -(b-a)
        fp2_sub_2x4x1(d, a, b);
        fp2_sub_2x4x1(e, b, a);
        fp2_neg_2x4x1(f, e);
        if (fp2_is_equal_2x4x1(d, f) != 0x3) {
            passed = 0;
            break;
        }

        // -0 == 0
        fp2_neg_2x4x1(d, zero);
        if (fp2_is_zero_2x4x1(d) != 0x3) {
            passed = 0;
            break;
        }

    }
    print_test_res("negation", passed);
    if (passed == 0) {
        return false;
    }

    // Multiplication tests
    // Using Montgomery constant
    fp2_setzero_2x4x1(vR);
    for(int i = 0; i < 8; i++) {
        vR[i] = VSET(R[i], 0, 0, 0, R[i], 0, 0, 0);
    }

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_2x4x1(a);
        rand_fp2_2x4x1(b);
        rand_fp2_2x4x1(c);

        //  a * 0  == 0
        fp2_mul_2x4x1(d, a, zero);
        if (fp2_is_zero_2x4x1(d) != 0x3) {
            passed = 0;
            break;
        }

        // (a * R)/R == a
        fp2_mul_2x4x1(d, a, vR);
        if (fp2_is_equal_2x4x1(d, a) != 0x3) {
            passed = 0;
            break;
        }

        // Transforms cancel
        fp2_tomont_2x4x1(d, a);
        fp2_frommont_2x4x1(e, d);
        if (fp2_is_equal_2x4x1(a, e) != 0x3) {
            passed = 0;
            break;
        }

        fp2_frommont_2x4x1(d, a);
        fp2_tomont_2x4x1(e, d);
        if (fp2_is_equal_2x4x1(a, e) != 0x3) {
            passed = 0;
            break;
        }

        // a * b == b * a
        fp2_mul_2x4x1(d, a, b);
        fp2_mul_2x4x1(e, b, a);
        if (fp2_is_equal_2x4x1(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // a * (b * c) == (a * b) * c
        fp2_mul_2x4x1(d, b, c);
        fp2_mul_2x4x1(d, d, a);
        fp2_mul_2x4x1(e, a, b);
        fp2_mul_2x4x1(e, e, c);
        if (fp2_is_equal_2x4x1(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // a * (b+c) == a*b + a*c
        fp2_add_2x4x1(d, b, c);
        fp2_mul_2x4x1(d, d, a);
        fp2_mul_2x4x1(e, a, b),
        fp2_mul_2x4x1(f, a, c);
        fp2_add_2x4x1(e, e, f);
        if (fp2_is_equal_2x4x1(d, e) != 0x3) {
            passed = 0;
            break;
        }

    }
    print_test_res("multiplication", passed);
    if (passed == 0) {
        return false;
    }

    // Squaring tests
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_2x4x1(a);
        rand_fp2_2x4x1(b);
        rand_fp2_2x4x1(c);

        // 0^2 == 0
        fp2_sqr_2x4x1(d, zero);
        if (fp2_is_zero_2x4x1(d) != 0x3) {
            passed = 0;
            break;
        }

        // a^2 = a * a
        fp2_sqr_2x4x1(d, a);
        fp2_mul_2x4x1(e, a, a);
        if (fp2_is_equal_2x4x1(d, e) != 0x3) {
            passed = 0;
            break;
        }

        // (a+b)^2 = a^2 + 2ab + b^2
        fp2_add_2x4x1(d, a, b);
        fp2_sqr_2x4x1(d, d);
        fp2_sqr_2x4x1(e, a);
        fp2_sqr_2x4x1(f, b);
        fp2_add_2x4x1(e, e, f);
        fp2_mul_2x4x1(f, a, b);
        fp2_add_2x4x1(f, f, f);
        fp2_add_2x4x1(e, e, f);
        if (fp2_is_equal_2x4x1(d, e) != 0x3) {
            passed = 0;
            break;
        }
    }

    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 lvl3 conversion tests:\n");

    fp2_t unvec1[8], unvec2[8];
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
            break;
        }

        fp2_8x1x1_to_unvectorized(unvec2, a);
        fp2_unvectorized_to_8x1x1(d, unvec2);
        if (fp2_is_equal_8x1x1(a, d) != 0xff) {
            passed = 0;
            break;
        }

        // Top 13 bits of each AVX limb is 0
        t = VSETALL(0xfff8000000000000);
        fp2_unvectorized_to_8x1x1(d, unvec1);
        for (int j = 0; j < 16; j++) {
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
            break;
        }

        fp2_4x2x1_to_unvectorized(unvec2, a);
        fp2_unvectorized_to_4x2x1(d, unvec2);

        if (fp2_is_equal_4x2x1(a, d) != 0xf) {
            passed = 0;
            break;
        }

        // Top 13 bits of each AVX limb is 0
        t = VSETALL(0xfff8000000000000);
        fp2_unvectorized_to_4x2x1(d, unvec1);
        for (int j = 0; j < 8; j++) {
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
        rand_fp2_2x4x1(a);

        // Converting back and forth does nothin
        fp2_unvectorized_to_2x4x1(d, unvec1);
        fp2_2x4x1_to_unvectorized(unvec2, d);

        for (int j = 0; j < 2; j++) {
            if (fp2_is_equal(&unvec1[j], &unvec2[j]) == 0) { passed = 0; }
        }
        if (passed == 0) {
            break;
        }

        fp2_2x4x1_to_unvectorized(unvec2, a);
        fp2_unvectorized_to_2x4x1(d, unvec2);

        if (fp2_is_equal_2x4x1(a, d) != 0x3) {
            passed = 0;
            break;
        }

        // Top 13 bits of each AVX limb is 0
        t = VSETALL(0xfff8000000000000);
        fp2_unvectorized_to_2x4x1(d, unvec1);
        for (int j = 0; j < 8; j++) {
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
                break;
            }
        }
    }
    print_test_res("unvectorized <-> 2x4x1", passed);
    if (passed == 0) {
        return false;
    }

    for(int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(b+8);

        // Conversion back and forth is identity
        fp2_8x1x1_to_4x2x1(d, d+8, a);
        fp2_4x2x1_to_8x1x1(e, d, d+8);
        if (fp2_is_equal_8x1x1(e, a) != 0xff) {
            passed = 0;
            break;
        }

        fp2_4x2x1_to_8x1x1(f, b, b+8);
        fp2_8x1x1_to_4x2x1(e, e+8, f);
        if (fp2_is_equal_4x2x1(e, b) != 0xf || fp2_is_equal_4x2x1(e+8, b+8) != 0xf) {
            passed = 0;
            break;
        }

        // Top 13 bits of each limb is 0
        t = VSETALL(0xfff8000000000000);
        for(int j = 0; j < 16; j++) {
            d[j] = VAND(d[j], t);
            f[j] = VAND(f[j], t);
        }
        if (fp2_is_zero_8x1x1(d) != 0xff || fp2_is_zero_8x1x1(f) != 0xff) {
            passed = 0;
            break;
        }

        // In place conversion back and forth is identity
        rand_fp2_8x1x1(c);
        fp2_copy_8x1x1(d, c);
        fp2_8x1x1_to_4x2x1(c, c+8, c);
        fp2_4x2x1_to_8x1x1(c, c, c+8);
        if (fp2_is_equal_8x1x1(c, d) != 0xff) {
            passed = 0;
            break;
        }
        rand_fp2_4x2x1(c);
        rand_fp2_4x2x1(c+8);
        fp2_copy_4x2x1(d, c);
        fp2_copy_4x2x1(d+8, c+8);
        fp2_4x2x1_to_8x1x1(c, c, c+8);
        fp2_8x1x1_to_4x2x1(c, c+8, c);
        if (fp2_is_equal_4x2x1(c, d) != 0xf || fp2_is_equal_4x2x1(c+8, d+8) != 0xf) {
            passed = 0;
            break;
        }

        // Conversion commutes with arithmetic
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        fp2_mul_8x1x1(c, a, b);

        fp2_8x1x1_to_4x2x1(d, d+8, a);
        fp2_8x1x1_to_4x2x1(e, e+8, b);
        fp2_mul_4x2x1(f, d, e);
        fp2_mul_4x2x1(f+8, d+8, e+8);
        fp2_4x2x1_to_8x1x1(f, f, f+8);

        if (fp2_is_equal_8x1x1(c, f) != 0xff) {
            passed = 0;
            break;
        }

        rand_fp2_4x2x1(a); rand_fp2_4x2x1(a+8);
        rand_fp2_4x2x1(b); rand_fp2_4x2x1(b+8);
        fp2_mul_4x2x1(c, a, b);
        fp2_mul_4x2x1(c+8, a+8, b+8);

        fp2_4x2x1_to_8x1x1(d, a, a+8);
        fp2_4x2x1_to_8x1x1(e, b, b+8);
        fp2_mul_8x1x1(f, d, e);
        fp2_8x1x1_to_4x2x1(f, f+8, f);

        if (fp2_is_equal_4x2x1(c, f) != 0xf || fp2_is_equal_4x2x1(c+8, f+8) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("8x1x1 <-> 4x2x1", passed);
    if (passed == 0) {
        return false;
    }

    for(int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(a);
        rand_fp2_2x4x1(b);
        rand_fp2_2x4x1(b+8);

        // Conversion back and forth is identity
        fp2_4x2x1_to_2x4x1(d, a);
        fp2_2x4x1_to_4x2x1(e, d, d+8);
        if (fp2_is_equal_4x2x1(e, a) != 0xf) {
            passed = 0;
            break;
        }

        fp2_2x4x1_to_4x2x1(f, b, b+8);
        fp2_4x2x1_to_2x4x1(e, f);
        if (fp2_is_equal_2x4x1(e, b) != 0x3 || fp2_is_equal_2x4x1(e+8, b+8) != 0x3) {
            passed = 0;
            break;
        }

        // Top 13 bits of each limb is 0
        t = VSETALL(0xfff8000000000000);
        for(int j = 0; j < 8; j++) {
            d[j] = VAND(d[j], t);
            f[j] = VAND(f[j], t);
        }
        if (fp2_is_zero_4x2x1(d) != 0xf || fp2_is_zero_4x2x1(f) != 0xf) {
            passed = 0;
            break;
        }

        // Conversion commutes with arithmetic
        fp2_mul_4x2x1(c, a, b);

        fp2_4x2x1_to_2x4x1(d, a);
        fp2_4x2x1_to_2x4x1(e, b);
        fp2_mul_2x4x1(f, d, e);
        fp2_mul_2x4x1(f+8, d+8, e+8);
        fp2_2x4x1_to_4x2x1(f, f, f+8);

        if (fp2_is_equal_4x2x1(c, f) != 0xf) {
            passed = 0;
            break;
        }

        rand_fp2_2x4x1(a); rand_fp2_2x4x1(a+8);
        rand_fp2_2x4x1(b); rand_fp2_2x4x1(b+8);
        fp2_mul_2x4x1(c, a, b);
        fp2_mul_2x4x1(c+8, a+8, b+8);

        fp2_2x4x1_to_4x2x1(d, a, a+8);
        fp2_2x4x1_to_4x2x1(e, b, b+8);
        fp2_mul_4x2x1(f, d, e);
        fp2_4x2x1_to_2x4x1(e, f);

        if (fp2_is_equal_2x4x1(c, e) != 0x3 || fp2_is_equal_2x4x1(c+8, e+8) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("4x2x1 <-> 2x4x1", passed);
    if (passed == 0) {
        return false;
    }*/

    return true;
}
