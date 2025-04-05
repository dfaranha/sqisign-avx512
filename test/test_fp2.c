#include <stdio.h>

#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "test.h"
#include "utils.h"

bool test_fp2(int iterations)
{
    __m512i a[16], b[16], c[16], d[16], e[16], f[16], zero[16], one[16], t;
    fp2_setzero_8x1x1(zero);
    fp2_setzero_8x1x1(one);
    one[0] = VSETALL(1);
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
        if (fp2_is_equal_8x1x1(d, zero) != 0xff) {
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
        if (fp2_is_equal_8x1x1(e, zero) != 0xff) {
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
        if (fp2_is_equal_8x1x1(d, zero) != 0xff) {
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
    uint64_t R[8] = { 0x3f03f03, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x1e80000 };
    __m512i vR[16];
    fp2_setzero_8x1x1(vR);
    fp_duplicate_8x1(vR, R);

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        rand_fp2_8x1x1(c);

        //  a * 0  == 0
        fp2_mul_8x1x1(d, a, zero);
        if (fp2_is_equal_8x1x1(d, zero) != 0xff) {
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
        if (fp2_is_equal_8x1x1(d, zero) != 0xff) {
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
        if (fp2_is_equal_4x1x2(d, zero) != 0xf) {
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
        if (fp2_is_equal_4x1x2(e, zero) != 0xf) {
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
        if (fp2_is_equal_4x1x2(d, zero) != 0xf) {
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
        if (fp2_is_equal_4x1x2(d, zero) != 0xf) {
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
        if (fp2_is_equal_4x1x2(d, zero) != 0xf) {
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
        if (fp2_is_equal_4x2x1(d, zero) != 0xf) {
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
        if (fp2_is_equal_4x2x1(e, zero) != 0xf) {
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
        if (fp2_is_equal_4x2x1(d, zero) != 0xf) {
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
        if (fp2_is_equal_4x2x1(d, zero) != 0xf) {
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
        if (fp2_is_equal_4x2x1(d, zero) != 0xf) {
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
}