#include <stdio.h>

#include "fp.h"
#include "test.h"
#include "utils.h"
#include "fp_kat.h"

bool test_fp(int iterations)
{
    __m512i a[NUM_LIMBS], b[NUM_LIMBS], c[NUM_LIMBS], d[NUM_LIMBS], e[NUM_LIMBS], f[NUM_LIMBS];
    __m512i g[NUM_LIMBS], h[NUM_LIMBS], zero[NUM_LIMBS], one[NUM_LIMBS], t;
    __m512i vR[NUM_LIMBS], vR2[NUM_LIMBS];
    uint64_t a_64[8*NWORDS64];
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
            printf("a != a\n");
            break;
        }
        fp_copy_8x1(b, a);
        if (fp_is_equal_8x1(a, b) != 0xff) {
            passed = 0;
            printf("copy of a != a\n");
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
            printf("a + 0 != a\n");
            break;
        }

        // a + 1 != a
        fp_add_8x1(e, a, one);
        if (fp_is_equal_8x1(a, e) != 0) {
            passed = 0;
            printf("a + 1 = a\n");
            break;
        }

        // a + b == b + a
        fp_add_8x1(d, a, b);
        fp_add_8x1(e, b, a);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            printf("a + b != b + a\n");
            break;
        }

        // (a + b) + c = a + (b + c)
        fp_add_8x1(d, a, b);
        fp_add_8x1(d, d, c);
        fp_add_8x1(e, b, c);
        fp_add_8x1(e, e, a);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            printf("(a + b) + c != a + (b + c)\n");
            break;
        }
    }
    for (int i = 0; i < NUM_FP_KAT; i++) {
        __m512i* A = a_add_8x1[i];
        __m512i* B = b_add_8x1[i];
        __m512i* C = c_add_8x1[i];

        fp_add_8x1(c, A, B);
        if (fp_is_equal_8x1(c, C) != 0xff) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
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
            printf("a - 0 != a\n");
            break;
        }

        // a - 1 != a
        fp_sub_8x1(e, a, one);
        if (fp_is_equal_8x1(a, e) != 0) {
            passed = 0;
            printf("a - 1 = a\n");
            break;
        }

        // a - a == 0
        fp_sub_8x1(d, a, a);
        if (fp_is_zero_8x1(d) != 0xff) {
            passed = 0;
            printf("a - a != 0\n");
            break;
        }

        // (a - b) - c = a - (b + c)
        fp_sub_8x1(d, a, b);
        fp_sub_8x1(d, d, c);
        fp_add_8x1(e, b, c);
        fp_sub_8x1(e, a, e);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            printf("(a-b) - c != a - (b+c)\n");
            break;
        }
    }
    for (int i = 0; i < NUM_FP_KAT; i++) {
        __m512i* A = a_sub_8x1[i];
        __m512i* B = b_sub_8x1[i];
        __m512i* C = c_sub_8x1[i];

        fp_sub_8x1(c, A, B);
        if (fp_is_equal_8x1(c, C) != 0xff) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
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
            printf("-a != 0 - a\n");
            break;
        }

        // -(-a) == a
        fp_neg_8x1(d, a);
        fp_neg_8x1(e, d);
        if (fp_is_equal_8x1(e, a) != 0xff) {
            passed = 0;
            printf("-(-a) != a\n");
            break;
        }

        // a + (-a) == 0
        fp_neg_8x1(d, a);
        fp_add_8x1(e, a, d);
        if (fp_is_zero_8x1(e) != 0xff) {
            passed = 0;
            printf("a + (-a) != 0\n");
            break;
        }

        // a - b == (-b) + a
        fp_sub_8x1(d, a, b);
        fp_neg_8x1(e, b);
        fp_add_8x1(f, e, a);
        if (fp_is_equal_8x1(d, f) != 0xff) {
            passed = 0;
            printf("a - b != (-b) + a\n");
            break;
        }

        // a-b == -(b-a)
        fp_sub_8x1(d, a, b);
        fp_sub_8x1(e, b, a);
        fp_neg_8x1(f, e);
        if (fp_is_equal_8x1(d, f) != 0xff) {
            passed = 0;
            printf("a-b != -(b-a)\n");
            break;
        }

        // -0 == 0
        fp_neg_8x1(d, zero);
        if (fp_is_zero_8x1(d) != 0xff) {
            passed = 0;
            printf("-0 != 0\n");
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
    fp_duplicate_8x1(vR, R);
    fp_duplicate_8x1(vR2, R2);

    // Montgomery representation of 1 is R mod p
    fp_tomont_8x1(d, one);
    if (fp_is_equal_8x1(d, vR) != 0xff) {
        printf("The Montgomery representation of 1 is not R mod p\n");
        passed = 0;
    }

    // Montgomery representation of R is R2 mod p
    fp_tomont_8x1(d, vR);
    if (fp_is_equal_8x1(d, vR2) != 0xff) {
        printf("The Montgomery representation of R is not R^2 mod p\n");
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
            printf("a * 0 != 0\n");
            break;
        }

        // (a * R)/R == a
        fp_mul_8x1(d, a, vR);
        if (fp_is_equal_8x1(d, a) != 0xff) {
            printf("(a * R)/ R != a\n");
            passed = 0;
            break;
        }

        // Transforms cancel
        fp_tomont_8x1(d, a);
        fp_frommont_8x1(e, d);
        if (fp_is_equal_8x1(a, e) != 0xff) {
            passed = 0;
            printf("tomont then frommont != identity\n");
            break;
        }

        fp_frommont_8x1(d, a);
        fp_tomont_8x1(e, d);
        if (fp_is_equal_8x1(a, e) != 0xff) {
            printf("frommon then tomont != identity\n");
            passed = 0;
            break;
        }

        // a * b == b * a
        fp_mul_8x1(d, a, b);
        fp_mul_8x1(e, b, a);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            printf("a * b != b * a\n");
            break;
        }

        // a * (b * c) == (a * b) * c
        fp_mul_8x1(d, b, c);
        fp_mul_8x1(d, d, a);
        fp_mul_8x1(e, a, b);
        fp_mul_8x1(e, e, c);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            printf("a * (b * c) != (a * b) * c\n");
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
            printf("a * (b+c) != a*b + a*c\n");
            break;
        }

    }
    for (int i = 0; i < NUM_FP_KAT; i++) {
        __m512i* A = a_mul_8x1[i];
        __m512i* B = b_mul_8x1[i];
        __m512i* C = c_mul_8x1[i];

        fp_mul_8x1(c, A, B);
        if (fp_is_equal_8x1(c, C) != 0xff) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
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
        printf("R^2 / R != R mod p\n");
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
            printf("0^2 != 0\n");
            break;
        }

        // a^2 = a * a
        fp_sqr_8x1(d, a);
        fp_mul_8x1(e, a, a);
        if (fp_is_equal_8x1(d, e) != 0xff) {
            passed = 0;
            printf("a^2 != a * a\n");
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
            printf("(a+b)^2 != a^2 + 2ab + b^2\n");
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

    // Square root
    for (int i = 0; i < iterations; i++) {
        rand_fp_8x1(a);
        
        fp_sqr_8x1(b, a);
        fp_sqrt_8x1(c, b);
        fp_sqr_8x1(d, c);

        if (fp_is_equal_8x1(b, d) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("sqrt", passed);
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
            printf("a != a\n");
            break;
        }
        fp_copy_4x2(b, a);
        if (fp_is_equal_4x2(a, b) != 0xf) {
            passed = 0;
            printf("copy of a != a\n");
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
            printf("a + 0 != a\n");
            break;
        }

        // a + 1 != a
        fp_add_4x2(e, a, one);
        if (fp_is_equal_4x2(a, e) != 0) {
            passed = 0;
            printf("a + 1 = a\n");
            break;
        }

        // a + b == b + a
        fp_add_4x2(d, a, b);
        fp_add_4x2(e, b, a);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            printf("a + b != b + a\n");
            break;
        }

        // (a + b) + c = a + (b + c)
        fp_add_4x2(d, a, b);
        fp_add_4x2(d, d, c);
        fp_add_4x2(e, b, c);
        fp_add_4x2(e, e, a);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            printf("(a+b) + c != a + (b+c)\n");
            break;
        }
    }
    for (int i = 0; i < NUM_FP_KAT; i++) {
        __m512i* A = a_add_4x2[i];
        __m512i* B = b_add_4x2[i];
        __m512i* C = c_add_4x2[i];

        fp_add_4x2(c, A, B);
        if (fp_is_equal_4x2(c, C) != 0xf) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
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
            printf("a - 0 != 0\n");
            break;
        }

        // a - 1 != a
        fp_sub_4x2(e, a, one);
        if (fp_is_equal_4x2(a, e) != 0) {
            passed = 0;
            printf("a - 1 != a\n");
            break;
        }

        // a - a == 0
        fp_sub_4x2(d, a, a);
        if (fp_is_zero_4x2(d) != 0xf) {
            passed = 0;
            printf("a - a != 0\n");
            break;
        }

        // (a - b) - c = a - (b + c)
        fp_sub_4x2(d, a, b);
        fp_sub_4x2(d, d, c);
        fp_add_4x2(e, b, c);
        fp_sub_4x2(e, a, e);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            printf("(a - b) - c != a - (b+c)\n");
            break;
        }
    }
    for (int i = 0; i < NUM_FP_KAT; i++) {
        __m512i* A = a_sub_4x2[i];
        __m512i* B = b_sub_4x2[i];
        __m512i* C = c_sub_4x2[i];

        fp_sub_4x2(c, A, B);
        if (fp_is_equal_4x2(c, C) != 0xf) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
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
            printf("(-a) != 0 - a\n");
            break;
        }

        // -(-a) == a
        fp_neg_4x2(d, a);
        fp_neg_4x2(e, d);
        if (fp_is_equal_4x2(e, a) != 0xf) {
            passed = 0;
            printf("-(-a) != a\n");
            break;
        }

        // a + (-a) == 0
        fp_neg_4x2(d, a);
        fp_add_4x2(e, a, d);
        if (fp_is_zero_4x2(e) != 0xf) {
            passed = 0;
            printf("a + (-a) != 0\n");
            break;
        }

        // a - b == (-b) + a
        fp_sub_4x2(d, a, b);
        fp_neg_4x2(e, b);
        fp_add_4x2(f, e, a);
        if (fp_is_equal_4x2(d, f) != 0xf) {
            passed = 0;
            printf("a - b != (-b) + a\n");
            break;
        }

        // a-b == -(b-a)
        fp_sub_4x2(d, a, b);
        fp_sub_4x2(e, b, a);
        fp_neg_4x2(f, e);
        if (fp_is_equal_4x2(d, f) != 0xf) {
            passed = 0;
            printf("a-b != -(b-a)\n");
            break;
        }

        // -0 == 0
        fp_neg_4x2(d, zero);
        if (fp_is_zero_4x2(d) != 0xf) {
            passed = 0;
            printf("-0 != 0\n");
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
        passed = 0;
        printf("Montgomery representation of 1 is not R mod p\n");
    }

    // Montgomery representation of R is R2 mod p
    fp_tomont_4x2(d, vR);
    if (fp_is_equal_4x2(d, vR2) != 0xf) {
        passed = 0;
        printf("Montgomery representation of R is not R^2 mod p\n");
    }

    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp_4x2(a);
        rand_fp_4x2(b);
        rand_fp_4x2(c);

        //  a * 0  == 0
        fp_mul_4x2(d, a, zero);
        if (fp_is_zero_4x2(d) != 0xf) {
            passed = 0;
            printf("a * 0 != 0\n");
            break;
        }

        // (a * R)/R == a
        fp_mul_4x2(d, a, vR);
        if (fp_is_equal_4x2(d, a) != 0xf) {
            passed = 0;
            printf("(a * R)/R != a\n");
            break;
        }

        // Transforms cancel
        fp_tomont_4x2(d, a);
        fp_frommont_4x2(e, d);
        if (fp_is_equal_4x2(a, e) != 0xf) {
            passed = 0;
            printf("tomont then frommont is not identity\n");
            break;
        }

        fp_frommont_4x2(d, a);
        fp_tomont_4x2(e, d);
        if (fp_is_equal_4x2(a, e) != 0xf) {
            passed = 0;
            printf("frommont then tomont is not identity\n");
            break;
        }

        // a * b == b * a
        fp_mul_4x2(d, a, b);
        fp_mul_4x2(e, b, a);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            printf("a * b != b * a\n");
            break;
        }

        // a * (b * c) == (a * b) * c
        fp_mul_4x2(d, b, c);
        fp_mul_4x2(d, d, a);
        fp_mul_4x2(e, a, b);
        fp_mul_4x2(e, e, c);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            printf("a * (b*c) != (a*b) * c\n");
            break;
        }

        // a * (b+c) == a*b + a*c
        fp_add_4x2(d, b, c);
        fp_mul_4x2(d, d, a);
        fp_mul_4x2(e, a, b),
        fp_mul_4x2(f, a, c);
        fp_add_4x2(e, e, f);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            printf("a * (b+c) != a*b + a*c\n");
            break;
        }

    }
    for (int i = 0; i < NUM_FP_KAT; i++) {
        __m512i* A = a_mul_4x2[i];
        __m512i* B = b_mul_4x2[i];
        __m512i* C = c_mul_4x2[i];

        fp_mul_4x2(c, A, B);
        if (fp_is_equal_4x2(c, C) != 0xf) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
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
            printf("0^2 != 0\n");
            break;
        }

        // a^2 = a * a
        fp_sqr_4x2(d, a);
        fp_mul_4x2(e, a, a);
        if (fp_is_equal_4x2(d, e) != 0xf) {
            passed = 0;
            printf("a^2 != a * a\n");
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
            printf("(a+b)^2 != a^2 + 2ab + b^2\n");
            break;
        }
    }

    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 3
    for (int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);
        fp_copy_4x2(b, a);

        fp_div3_4x2(b, b);
        fp_add_4x2(c, b, b);
        fp_add_4x2(c, c, b);

        if (fp_is_equal_4x2(a, c) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 3", passed);
    if (passed == 0) {
        return false;
    }

    // Square root
    for (int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);
        
        fp_sqr_4x2(b, a);
        fp_sqrt_4x2(c, b);
        fp_sqr_4x2(d, c);

        if (fp_is_equal_4x2(b, d) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("sqrt", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp lvl3 2x4 tests:\n");

    // Equality test
    for(int i = 0; i < iterations; i++){
        rand_fp_2x4(a);
        if (fp_is_equal_2x4(a, a) != 0x3) {
            passed = 0;
            printf("a != a\n");
            break;
        }
        fp_copy_2x4(b, a);
        if (fp_is_equal_2x4(a, b) != 0x3) {
            passed = 0;
            printf("copy of a != a\n");
            break;
        }
    }
    print_test_res("equality", passed);
    if (passed == 0) {
        return false;
    }

    // Addition test
    for(int i = 0; i < iterations; i++) {
        rand_fp_2x4(a);
        rand_fp_2x4(b);
        rand_fp_2x4(c);

        // a + 0 == a
        fp_add_2x4(d, a, zero);
        if(fp_is_equal_2x4(a, d) != 0x3) {
            passed = 0;
            printf("a + 0 != a\n");
            break;
        }

        // a + 1 != a
        fp_add_2x4(e, a, one);
        if (fp_is_equal_2x4(a, e) != 0) {
            passed = 0;
            printf("a + 1 = a\n");
            break;
        }

        // a + b == b + a
        fp_add_2x4(d, a, b);
        fp_add_2x4(e, b, a);
        if (fp_is_equal_2x4(d, e) != 0x3) {
            passed = 0;
            printf("a + b != b + a\n");
            break;
        }

        // (a + b) + c = a + (b + c)
        fp_add_2x4(d, a, b);
        fp_add_2x4(d, d, c);
        fp_add_2x4(e, b, c);
        fp_add_2x4(e, e, a);
        if (fp_is_equal_2x4(d, e) != 0x3) {
            passed = 0;
            printf("(a+b) + c != a + (b+c)\n");
            break;
        }
    }
    for (int i = 0; i < NUM_FP_KAT; i++) {
        __m512i* A = a_add_2x4[i];
        __m512i* B = b_add_2x4[i];
        __m512i* C = c_add_2x4[i];

        fp_add_2x4(c, A, B);
        if (fp_is_equal_2x4(c, C) != 0x3) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
            break;
        }
    }
    print_test_res("addition", passed);
    if (passed == 0) {
        return false;
    }

    // Subtraction test
    for(int i = 0; i < iterations; i++) {
        rand_fp_2x4(a);
        rand_fp_2x4(b);
        rand_fp_2x4(c);

        // a - 0 == a
        fp_sub_2x4(d, a, zero);
        if(fp_is_equal_2x4(a, d) != 0x3) {
            passed = 0;
            printf("a - 0 != 0\n");
            break;
        }

        // a - 1 != a
        fp_sub_2x4(e, a, one);
        if (fp_is_equal_2x4(a, e) != 0) {
            passed = 0;
            printf("a - 1 != a\n");
            break;
        }

        // a - a == 0
        fp_sub_2x4(d, a, a);
        if (fp_is_zero_2x4(d) != 0x3) {
            passed = 0;
            printf("a - a != 0\n");
            break;
        }

        // (a - b) - c = a - (b + c)
        fp_sub_2x4(d, a, b);
        fp_sub_2x4(d, d, c);
        fp_add_2x4(e, b, c);
        fp_sub_2x4(e, a, e);
        if (fp_is_equal_2x4(d, e) != 0x3) {
            passed = 0;
            printf("(a - b) - c != a - (b+c)\n");
            break;
        }
    }
    for (int i = 0; i < NUM_FP_KAT; i++) {
        __m512i* A = a_sub_2x4[i];
        __m512i* B = b_sub_2x4[i];
        __m512i* C = c_sub_2x4[i];

        fp_sub_2x4(c, A, B);
        if (fp_is_equal_2x4(c, C) != 0x3) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
            break;
        }
    }
    print_test_res("subtraction", passed);
    if (passed == 0) {
        return false;
    }

    // Negation tests
    for(int i = 0; i < iterations; i++) {
        rand_fp_2x4(a);
        rand_fp_2x4(b);

        // (-a) == 0 - a
        fp_neg_2x4(d, a);
        fp_sub_2x4(e, zero, a);
        if (fp_is_equal_2x4(d, e) != 0x3) {
            passed = 0;
            printf("(-a) != 0 - a\n");
            break;
        }

        // -(-a) == a
        fp_neg_2x4(d, a);
        fp_neg_2x4(e, d);
        if (fp_is_equal_2x4(e, a) != 0x3) {
            passed = 0;
            printf("-(-a) != a\n");
            break;
        }

        // a + (-a) == 0
        fp_neg_2x4(d, a);
        fp_add_2x4(e, a, d);
        if (fp_is_zero_2x4(e) != 0x3) {
            passed = 0;
            printf("a + (-a) != 0\n");
            break;
        }

        // a - b == (-b) + a
        fp_sub_2x4(d, a, b);
        fp_neg_2x4(e, b);
        fp_add_2x4(f, e, a);
        if (fp_is_equal_2x4(d, f) != 0x3) {
            passed = 0;
            printf("a - b != (-b) + a\n");
            break;
        }

        // a-b == -(b-a)
        fp_sub_2x4(d, a, b);
        fp_sub_2x4(e, b, a);
        fp_neg_2x4(f, e);
        if (fp_is_equal_2x4(d, f) != 0x3) {
            passed = 0;
            printf("a-b != -(b-a)\n");
            break;
        }

        // -0 == 0
        fp_neg_2x4(d, zero);
        if (fp_is_zero_2x4(d) != 0x3) {
            passed = 0;
            printf("-0 != 0\n");
            break;
        }

    }
    print_test_res("negation", passed);
    if (passed == 0) {
        return false;
    }

    // Mixed subtraction & addition
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp_2x4(a);
        rand_fp_2x4(b);

        fp_mixaddsub_2x4(d, a, b, 0xf);
        fp_sub_2x4(e, a, b);
        fp_add_2x4(f, a, b);
        if(fp_is_equal_2x4(d, e) != 0x2 || fp_is_equal_2x4(d, f) != 0x1) {
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
    fp_setzero_2x4(one);
    one[0] = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    fp_duplicate_2x4(vR, R);
    fp_duplicate_2x4(vR2, R2);

    // Montgomery representation of 1 is R mod p
    fp_tomont_2x4(d, one);
    if (fp_is_equal_2x4(d, vR) != 0x3) {
        printf("Montgomery representation of 1 is not R mod p\n");
        passed = 0;
    }

    // Montgomery representation of R is R2 mod p
    fp_tomont_2x4(d, vR);
    if (fp_is_equal_2x4(d, vR2) != 0x3) {
        printf("Montgomery representation of R is not R^2 mod p\n");
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
            printf("a * 0 != 0\n");
            break;
        }

        // (a * R)/R == a
        fp_mul_2x4(d, a, vR);
        if (fp_is_equal_2x4(d, a) != 0x3) {
            passed = 0;
            printf("(a * R)/R != a\n");
            break;
        }

        // Transforms cancel
        fp_tomont_2x4(d, a);
        fp_frommont_2x4(e, d);
        if (fp_is_equal_2x4(a, e) != 0x3) {
            passed = 0;
            printf("tomont then frommont is not identity\n");
            break;
        }

        fp_frommont_2x4(d, a);
        fp_tomont_2x4(e, d);
        if (fp_is_equal_2x4(a, e) != 0x3) {
            passed = 0;
            printf("frommont then tomont is not identity\n");
            break;
        }

        // a * b == b * a
        fp_mul_2x4(d, a, b);
        fp_mul_2x4(e, b, a);
        if (fp_is_equal_2x4(d, e) != 0x3) {
            passed = 0;
            printf("a * b != b * a\n");
            break;
        }

        // a * (b * c) == (a * b) * c
        fp_mul_2x4(d, b, c);
        fp_mul_2x4(d, d, a);
        fp_mul_2x4(e, a, b);
        fp_mul_2x4(e, e, c);
        if (fp_is_equal_2x4(d, e) != 0x3) {
            passed = 0;
            printf("a * (b * c) != (a * b) * c\n");
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
            printf("a * (b+c) != a*b + a*c\n");
            break;
        }
    }
    for (int i = 0; i < NUM_FP_KAT; i++) {
        __m512i* A = a_mul_2x4[i];
        __m512i* B = b_mul_2x4[i];
        __m512i* C = c_mul_2x4[i];

        fp_mul_2x4(c, A, B);
        if (fp_is_equal_2x4(c, C) != 0x3) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
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
            printf("a^2  != a * a\n");
            break;
        }
    }
    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 3
    for (int i = 0; i < iterations; i++) {
        rand_fp_2x4(a);
        fp_copy_2x4(b, a);

        fp_div3_2x4(b, b);
        fp_add_2x4(c, b, b);
        fp_add_2x4(c, c, b);

        if (fp_is_equal_2x4(a, c) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 3", passed);
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
    
    printf("Conversions between Fp representations\n");

    for (int i = 0; i < iterations; i++) {
        rand_fp_8x1(a);

        // back and forth does nothing
        fp_8x1_to_4x2(b, c, a);
        fp_4x2_to_8x1(d, b, c);

        if (fp_is_equal_8x1(d, a) != 0xff) {
            passed = 0;
            printf("8x1 -> 4x2 -> 8x1 is not identity!\n");
            break;
        }

        rand_fp_4x2(a);
        rand_fp_4x2(b);
        fp_4x2_to_8x1(c, a, b);
        fp_8x1_to_4x2(d, e, c);

        if (fp_is_equal_4x2(d, a) != 0xf || fp_is_equal_4x2(e, b) != 0xf) {
            passed = 0;
            printf("4x2 -> 8x1 -> 4x2 is not identity!\n");
            break;
        }

        rand_fp_8x1(a);
        rand_fp_8x1(b);
        fp_mul_8x1(c, a, b);
        fp_8x1_to_4x2(d, e, a);
        fp_8x1_to_4x2(f, g, b);
        fp_mul_4x2(d, d, f);
        fp_mul_4x2(e, e, g);
        fp_4x2_to_8x1(d, d, e);

        if (fp_is_equal_8x1(c, d) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("8x1 <-> 4x2", passed);
    if (passed == 0) {
        return false;
    }

    for (int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);

        // back and forth does nothing
        fp_4x2_to_2x4(b, c, a);
        fp_2x4_to_4x2(b, b, c);

        if (fp_is_equal_4x2(b, a) != 0xf) {
            passed = 0;
            break;
        }

        rand_fp_2x4(a);
        rand_fp_2x4(b);
        fp_2x4_to_4x2(c, a, b);
        fp_4x2_to_2x4(c, d, c);

        if (fp_is_equal_2x4(c, a) != 0x3 || fp_is_equal_2x4(d, b) != 0x3) {
            passed = 0;
            break;
        }

        rand_fp_4x2(a);
        rand_fp_4x2(b);
        fp_mul_4x2(c, a, b);
        fp_4x2_to_2x4(d, f, a);
        fp_4x2_to_2x4(e, g, b);
        fp_mul_2x4(d, d, e);
        fp_mul_2x4(f, f, g);
        fp_2x4_to_4x2(d, d, f);

        if (fp_is_equal_4x2(c, d) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("2x4 <-> 4x2", passed);
    if (passed == 0) {
        return false;
    }

    // 8x1 <-> uint64_t
    for (int i = 0; i < iterations; i++) {
        rand_fp_8x1(a);

        // back and forth does nothing
        fp_limbs_to_64_8x1(a_64, a);
        fp_64_to_limbs_8x1(b, a_64);

        if (fp_is_equal_8x1(b, a) != 0xff) {
            passed = 0;
            break;
        }

        // back and forth on even idx does nothing
        fp_limbs_to_64_8x1_only_even_idx(a_64, a);
        fp_64_to_limbs_8x1_only_even_idx(b, a_64);

        if ((fp_is_equal_8x1(b, a) & 0x55) != 0x55) {
            passed = 0;
            break;
        }
    }
    print_test_res("8x1 <-> 64", passed);
    if (passed == 0) {
        return false;
    }

    // 4x2 <-> uint64_t
    for (int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);

        // back and forth does nothing
        fp_limbs_to_64_4x2(a_64, a);
        fp_64_to_limbs_4x2(b, a_64);

        if (fp_is_equal_4x2(b, a) != 0xf) {
            passed = 0;
            break;
        }

    }
    print_test_res("4x2 <-> 64", passed);
    if (passed == 0) {
        return false;
    }

    // 2x4 <-> uint64_t
    for (int i = 0; i < iterations; i++) {
        rand_fp_2x4(a);

        // back and forth does nothing
        fp_limbs_to_64_2x4(a_64, a);
        fp_64_to_limbs_2x4(b, a_64);

        if (fp_is_equal_2x4(b, a) != 0x3) {
            passed = 0;
            break;
        }

    }
    print_test_res("2x4 <-> 64", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp inversion tests\n");

    // Inverse
    for (int i = 0; i < iterations; i++) {
        rand_fp_2x4(a);

        if (fp_is_zero_2x4(a) != 0)
            continue;
        
        fp_inv_2x4(b, a);
        fp_mul_2x4(c, b, a);

        if (fp_is_equal_2x4(c, vR) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("2x4", passed);
    if (passed == 0) {
        return false;
    }

    //Batched inversion
    fp_duplicate_4x2(vR, R);
    for (int i = 0; i < iterations; i++) {
        rand_fp_4x2(a);
        if (fp_is_zero_4x2(a) != 0) { continue; }

        fp_inv_4x2(b, a);
        fp_mul_4x2(c, a, b);

        if (fp_is_equal_4x2(c, vR) != 0xf) {
            passed = 0;
            printf("a * a^-1 != 1 mod p\n");
            break;
        }
    }
    print_test_res("4x2", passed);
    if (passed == 0) {
        return false;
    }

    //Batched inversion
    fp_duplicate_8x1(vR, R);
    for (int i = 0; i < iterations; i++) {
        rand_fp_8x1(a);
        if (fp_is_zero_8x1(a) != 0) { continue; }

        fp_inv_8x1(b, a);
        fp_mul_8x1(c, a, b);

        if (fp_is_equal_8x1(c, vR) != 0xff) {
            passed = 0;
            printf("a * a^-1 != 1 mod p\n");
            break;
        }
    }
    print_test_res("inversion", passed);
    if (passed == 0) {
        return false;
    }
    
    return true;
}
