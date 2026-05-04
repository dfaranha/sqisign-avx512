#include <stdio.h>

#include "fp.h"
#include "fp_consts.h"
#include "fp2.h"
#include "fp2_kat.h"
#include "test.h"
#include "utils.h"

bool test_fp2(int iterations)
{
    __m512i a[2*NUM_LIMBS], b[2*NUM_LIMBS], c[2*NUM_LIMBS], d[2*NUM_LIMBS], e[2*NUM_LIMBS], f[2*NUM_LIMBS];
    __m512i g[2*NUM_LIMBS], h[2*NUM_LIMBS], zero[2*NUM_LIMBS], vR[2*NUM_LIMBS], t;
    fp2_setzero_8x1x1(zero);
    fp2_setzero_8x1x1(vR);
    fp_duplicate_8x1(vR, R);

    int passed = 1;

    printf("Fp2 8x1x1 tests:\n");

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
        c[NUM_LIMBS+1] = VADD(c[NUM_LIMBS+1], t);
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
        fp2_add_8x1x1(e, a, vR);
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_add_8x1x1[i];
        __m512i* B = b_add_8x1x1[i];
        __m512i* C = c_add_8x1x1[i];

        fp2_add_8x1x1(c, A, B);
        if (fp2_is_equal_8x1x1(c, C) != 0xff) {
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
        fp2_sub_8x1x1(e, a, vR);
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_sub_8x1x1[i];
        __m512i* B = b_sub_8x1x1[i];
        __m512i* C = c_sub_8x1x1[i];

        fp2_sub_8x1x1(c, A, B);
        if (fp2_is_equal_8x1x1(c, C) != 0xff) {
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

    // Multiplication
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        rand_fp2_8x1x1(c);

        //  a * 0  == 0
        fp2_mul_8x1x1(d, a, zero);
        if (fp2_is_zero_8x1x1(d) != 0xff) {
            passed = 0;
            printf("a * 0 != 0\n");
            break;
        }

        // (a * R)/R == a
        fp2_mul_8x1x1(d, a, vR);
        if (fp2_is_equal_8x1x1(d, a) != 0xff) {
            printf("(a * R)/R != a\n");
            passed = 0;
            break;
        }

        // Transforms cancel
        fp2_tomont_8x1x1(d, a);
        fp2_frommont_8x1x1(e, d);
        if (fp2_is_equal_8x1x1(a, e) != 0xff) {
            passed = 0;
            printf("tomont -> frommont != id\n");
            break;
        }

        fp2_frommont_8x1x1(d, a);
        fp2_tomont_8x1x1(e, d);
        if (fp2_is_equal_8x1x1(a, e) != 0xff) {
            passed = 0;
            printf("frommont -> tomont != id\n");
            break;
        }

        // a * b == b * a
        fp2_mul_8x1x1(d, a, b);
        fp2_mul_8x1x1(e, b, a);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            printf("a * b != b * a\n");
            break;
        }

        // a * (b * c) == (a * b) * c
        fp2_mul_8x1x1(d, b, c);
        fp2_mul_8x1x1(d, d, a);
        fp2_mul_8x1x1(e, a, b);
        fp2_mul_8x1x1(e, e, c);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            printf("a * (b * c) != (a * b) * c\n");
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
            printf("a * (b+c) != a*b + a*c\n");
            break;
        }

    }
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_mul_8x1x1[i];
        __m512i* B = b_mul_8x1x1[i];
        __m512i* C = c_mul_8x1x1[i];

        fp2_mul_8x1x1(c, A, B);
        if (fp2_is_equal_8x1x1(c, C) != 0xff) {
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
    for(int i = 0; i < iterations && passed != 0; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        rand_fp2_8x1x1(c);

        // 0^2 == 0
        fp2_sqr_8x1x1(d, zero);
        if (fp2_is_zero_8x1x1(d) != 0xff) {
            passed = 0;
            printf("0^2 != 0\n");
            break;
        }

        // a^2 = a * a
        fp2_sqr_8x1x1(d, a);
        fp2_mul_8x1x1(e, a, a);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            printf("a^2 != a * a\n");
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
            printf("(a+b)^2 != a^2 + 2ab + b^2\n");
            break;
        }
    }
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_sqr_8x1x1[i];
        __m512i* C = c_sqr_8x1x1[i];

        fp2_sqr_8x1x1(c, A);
        if (fp2_is_equal_8x1x1(c, C) != 0xff) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
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
    a[NUM_LIMBS] = VSETALL(1);
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

    // Conjugation
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        
        // conjugating twice = identity
        fp2_conjugate_8x1x1(c, a);
        fp2_conjugate_8x1x1(c, c);

        if (fp2_is_equal_8x1x1(a, c) != 0xff) {
            passed = 0;
            printf("conjugating twice is not identity!\n");
            break;
        }

        // conjugate(a) + conjugate(b) = conjugate(a+b)
        fp2_conjugate_8x1x1(c, a);
        fp2_conjugate_8x1x1(d, b);
        fp2_add_8x1x1(c, c, d);
        fp2_add_8x1x1(d, a, b);
        fp2_conjugate_8x1x1(d, d);
        if (fp2_is_equal_8x1x1(c, d) != 0xff) {
            passed = 0;
            printf("conjugate(a) + conjugate(b) != conjugate(a+b)\n");
            break;
        }

        // conjugate(a)*conjugate(b) = conjugate(a*b)
        fp2_conjugate_8x1x1(c, a);
        fp2_conjugate_8x1x1(d, b);
        fp2_mul_8x1x1(c, c, d);
        fp2_mul_8x1x1(d, a, b);
        fp2_conjugate_8x1x1(d, d);
        if (fp2_is_equal_8x1x1(c, d) != 0xff) {
            passed = 0;
            printf("conjugate(a)*conjugate(b) != conjugate(a*b)\n");
            break;
        }

        // a + conjugate(a) = 2*re(a)
        fp2_conjugate_8x1x1(c, a);
        fp2_add_8x1x1(c, c, a);
        fp_add_8x1(d, a, a);
        if (fp_is_equal_8x1(c, d) != 0xff || fp_is_zero_8x1(c + NUM_LIMBS) != 0xff)
        {
            passed = 0;
            printf("a + conjugate(a) != 2*re(a)\n");
            break;
        }
    }
    print_test_res("conjugation", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 2
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        fp2_copy_8x1x1(b, a);

        fp2_div2_8x1x1(b, b);
        fp2_add_8x1x1(c, b, b);

        if (fp2_is_equal_8x1x1(a, c) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 2", passed);
    if (passed == 0) {
        return false;
    }

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

    // Inverse
    fp2_inv_8x1x1(a, zero);
    if (fp2_is_zero_8x1x1(a) != 0xff) {
        passed = 0;
        printf("Inv does not output 0 on input 0\n");
    }

    for (int i = 0; i < iterations && passed; i++) {
        // a * a^-1 == 1
        rand_fp2_8x1x1(a);

        if (fp2_is_zero_8x1x1(a))
            continue;

        fp2_inv_8x1x1(b, a);
        fp2_mul_8x1x1(b, b, a);

        if (fp2_is_equal_8x1x1(b, vR) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("inv", passed);
    if (passed == 0) {
        return false;
    }

    // Square root
    fp2_sqrt_8x1x1(a, zero);
    if (fp2_is_zero_8x1x1(a) != 0xff) {
        passed = 0;
        printf("sqrt(0) != 0\n");
    }

    for (int i = 0; i < iterations && passed; i++) {
        rand_fp2_8x1x1(a);
        
        fp2_sqr_8x1x1(b, a);
        fp2_sqrt_8x1x1(c, b);
        fp2_sqr_8x1x1(d, c);

        if (fp2_is_equal_8x1x1(b, d) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("sqrt", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 4x1x2 tests:\n");
    fp2_setzero_4x1x2(vR);
    fp_duplicate_4x2(vR, R);

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
        c[N4x2 + 1] = VADD(c[N4x2 + 1], t);
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
        fp2_add_4x1x2(e, a, vR);
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_add_4x1x2[i];
        __m512i* B = b_add_4x1x2[i];
        __m512i* C = c_add_4x1x2[i];

        fp2_add_4x1x2(c, A, B);
        if (fp2_is_equal_4x1x2(c, C) != 0xf) {
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
        fp2_sub_4x1x2(e, a, vR);
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_sub_4x1x2[i];
        __m512i* B = b_sub_4x1x2[i];
        __m512i* C = c_sub_4x1x2[i];

        fp2_sub_4x1x2(c, A, B);
        if (fp2_is_equal_4x1x2(c, C) != 0xf) {
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

    // Multiplication
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_mul_4x1x2[i];
        __m512i* B = b_mul_4x1x2[i];
        __m512i* C = c_mul_4x1x2[i];

        fp2_mul_4x1x2(c, A, B);
        if (fp2_is_equal_4x1x2(c, C) != 0xf) {
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_sqr_4x1x2[i];
        __m512i* C = c_sqr_4x1x2[i];

        fp2_sqr_4x1x2(c, A);
        if (fp2_is_equal_4x1x2(c, C) != 0xf) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
            break;
        }
    }
    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    // Preparing i
    fp2_setzero_4x1x2(a);
    a[N4x2] = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    fp2_tomont_4x1x2(a, a);

    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x1x2(b);
        
        fp2_muli_4x1x2(c, b);
        fp2_mul_4x1x2(d, a, b);

        if (fp2_is_equal_4x1x2(c, d) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("mult by i", passed);
    if (passed == 0) {
        return false;
    }

    // Conjugation
    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x1x2(a);
        rand_fp2_4x1x2(b);
        
        // conjugating twice = identity
        fp2_conjugate_4x1x2(c, a);
        fp2_conjugate_4x1x2(c, c);

        if (fp2_is_equal_4x1x2(a, c) != 0xf) {
            passed = 0;
            printf("conjugating twice is not identity!\n");
            break;
        }

        // conjugate(a) + conjugate(b) = conjugate(a+b)
        fp2_conjugate_4x1x2(c, a);
        fp2_conjugate_4x1x2(d, b);
        fp2_add_4x1x2(c, c, d);
        fp2_add_4x1x2(d, a, b);
        fp2_conjugate_4x1x2(d, d);
        if (fp2_is_equal_4x1x2(c, d) != 0xf) {
            passed = 0;
            printf("conjugate(a) + conjugate(b) != conjugate(a+b)\n");
            break;
        }

        // conjugate(a)*conjugate(b) = conjugate(a*b)
        fp2_conjugate_4x1x2(c, a);
        fp2_conjugate_4x1x2(d, b);
        fp2_mul_4x1x2(c, c, d);
        fp2_mul_4x1x2(d, a, b);
        fp2_conjugate_4x1x2(d, d);
        if (fp2_is_equal_4x1x2(c, d) != 0xf) {
            passed = 0;
            printf("conjugate(a)*conjugate(b) != conjugate(a*b)\n");
            break;
        }

        // a + conjugate(a) = 2*re(a)
        fp2_conjugate_4x1x2(c, a);
        fp2_add_4x1x2(c, c, a);
        fp_add_4x2(d, a, a);
        if (fp_is_equal_4x2(c, d) != 0xf || fp_is_zero_4x2(c + N4x2) != 0xf)
        {
            passed = 0;
            printf("a + conjugate(a) != 2*re(a)\n");
            break;
        }
    }
    print_test_res("conjugation", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 2
    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x1x2(a);
        fp2_copy_4x1x2(b, a);

        fp2_div2_4x1x2(b, b);
        fp2_add_4x1x2(c, b, b);

        if (fp2_is_equal_4x1x2(a, c) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 2", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 3
    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x1x2(a);
        fp2_copy_4x1x2(b, a);

        fp2_div3_4x1x2(b, b);
        fp2_add_4x1x2(c, b, b);
        fp2_add_4x1x2(c, c, b);

        if (fp2_is_equal_4x1x2(a, c) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 3", passed);
    if (passed == 0) {
        return false;
    }

    // Inverse
    fp2_inv_4x1x2(a, zero);
    if (fp2_is_zero_4x1x2(a) != 0xf) {
        passed = 0;
        printf("Inv does not output 0 on input 0\n");
    }

    for (int i = 0; i < iterations && passed; i++) {
        rand_fp2_4x1x2(a);

        if (fp2_is_zero_4x1x2(a))
            continue;
        
        fp2_inv_4x1x2(b, a);
        fp2_mul_4x1x2(b, b, a);

        if (fp2_is_equal_4x1x2(b, vR) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("inv", passed);
    if (passed == 0) {
        return false;
    }

    // Square root
    fp2_sqrt_4x1x2(a, zero);
    if (fp2_is_zero_4x1x2(a) != 0xf) {
        passed = 0;
        printf("sqrt(0) != 0\n");
    }

    for (int i = 0; i < iterations && passed; i++) {
        rand_fp2_4x1x2(a);
        
        fp2_sqr_4x1x2(b, a);
        fp2_sqrt_4x1x2(c, b);
        fp2_sqr_4x1x2(d, c);

        if (fp2_is_equal_4x1x2(b, d) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("sqrt", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 4x2x1 tests:\n");
    fp2_setzero_4x2x1(vR);
    for(int i = 0; i < NUM_LIMBS; i++) {
        vR[i] = VSET(R[i], 0, R[i], 0, R[i], 0, R[i], 0);
    }

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
        c[NUM_LIMBS-1] = VADD(c[NUM_LIMBS-1], t);
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
        fp2_add_4x2x1(e, a, vR);
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_add_4x2x1[i];
        __m512i* B = b_add_4x2x1[i];
        __m512i* C = c_add_4x2x1[i];

        fp2_add_4x2x1(c, A, B);
        if (fp2_is_equal_4x2x1(c, C) != 0xf) {
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
        fp2_sub_4x2x1(e, a, vR);
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_sub_4x2x1[i];
        __m512i* B = b_sub_4x2x1[i];
        __m512i* C = c_sub_4x2x1[i];

        fp2_sub_4x2x1(c, A, B);
        if (fp2_is_equal_4x2x1(c, C) != 0xf) {
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

    // Multiplication
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_mul_4x2x1[i];
        __m512i* B = b_mul_4x2x1[i];
        __m512i* C = c_mul_4x2x1[i];

        fp2_mul_4x2x1(c, A, B);
        if (fp2_is_equal_4x2x1(c, C) != 0xf) {
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_sqr_4x2x1[i];
        __m512i* C = c_sqr_4x2x1[i];

        fp2_sqr_4x2x1(c, A);
        if (fp2_is_equal_4x2x1(c, C) != 0xf) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
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

    // Conjugation
    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        
        // conjugating twice = identity
        fp2_conjugate_4x2x1(c, a);
        fp2_conjugate_4x2x1(c, c);

        if (fp2_is_equal_4x2x1(a, c) != 0xf) {
            passed = 0;
            printf("conjugating twice is not identity!\n");
            break;
        }

        // conjugate(a) + conjugate(b) = conjugate(a+b)
        fp2_conjugate_4x2x1(c, a);
        fp2_conjugate_4x2x1(d, b);
        fp2_add_4x2x1(c, c, d);
        fp2_add_4x2x1(d, a, b);
        fp2_conjugate_4x2x1(d, d);
        if (fp2_is_equal_4x2x1(c, d) != 0xf) {
            passed = 0;
            printf("conjugate(a) + conjugate(b) != conjugate(a+b)\n");
            break;
        }

        // conjugate(a)*conjugate(b) = conjugate(a*b)
        fp2_conjugate_4x2x1(c, a);
        fp2_conjugate_4x2x1(d, b);
        fp2_mul_4x2x1(c, c, d);
        fp2_mul_4x2x1(d, a, b);
        fp2_conjugate_4x2x1(d, d);
        if (fp2_is_equal_4x2x1(c, d) != 0xf) {
            passed = 0;
            printf("conjugate(a)*conjugate(b) != conjugate(a*b)\n");
            break;
        }

        // a + conjugate(a) = 2*re(a)
        fp2_conjugate_4x2x1(c, a);
        fp2_add_4x2x1(c, c, a);
        fp2_add_4x2x1(d, a, a);
        if (fp_is_equal_8x1(c, d) & 0x55 != 0x55 || fp_is_zero_8x1(c) & 0xaa != 0xaa)
        {
            passed = 0;
            printf("a + conjugate(a) != 2*re(a)\n");
            break;
        }
    }
    print_test_res("conjugation", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 2
    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(a);
        fp2_copy_4x2x1(b, a);

        fp2_div2_4x2x1(b, b);
        fp2_add_4x2x1(c, b, b);

        if (fp2_is_equal_4x2x1(a, c) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 2", passed);
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

    // Inverse
    fp2_inv_4x2x1(a, zero);
    if (fp2_is_zero_4x2x1(a) != 0xf) {
        passed = 0;
        printf("Inv does not output 0 on input 0\n");
    }

    for (int i = 0; i < iterations && passed; i++) {
        rand_fp2_4x2x1(a);

        if (fp2_is_zero_4x2x1(a))
            continue;
        
        fp2_inv_4x2x1(b, a);
        fp2_mul_4x2x1(b, b, a);

        if (fp2_is_equal_4x2x1(b, vR) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("inv", passed);
    if (passed == 0) {
        return false;
    }

    // Square root
    fp2_sqrt_4x2x1(a, zero);
    if (fp2_is_zero_4x2x1(a) != 0xf) {
        passed = 0;
        printf("sqrt(0) != 0\n");
    }

    for (int i = 0; i < iterations && passed; i++) {
        rand_fp2_4x2x1(a);
        
        fp2_sqr_4x2x1(b, a);
        fp2_sqrt_4x2x1(c, b);
        fp2_sqr_4x2x1(d, c);

        if (fp2_is_equal_4x2x1(b, d) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("sqrt", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 2x2x2 tests:\n");
    
    fp2_setzero_2x2x2(vR);
    int rem4x2 = NUM_LIMBS % 2;
    for(int i = 0; i < N4x2 - rem4x2; i++) {
        vR[i] = VSET(R[i], R[i+N4x2], 0, 0, R[i], R[i+N4x2], 0, 0);
    }
    if (rem4x2) {
        vR[N4x2-1] = VSET(R[N4x2-1], 0, 0, 0, R[N4x2-1], 0, 0, 0);
    }

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
        c[N4x2-1] = VADD(c[N4x2-1], t);
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
        fp2_add_2x2x2(e, a, vR);
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_add_2x2x2[i];
        __m512i* B = b_add_2x2x2[i];
        __m512i* C = c_add_2x2x2[i];

        fp2_add_2x2x2(c, A, B);
        if (fp2_is_equal_2x2x2(c, C) != 0x3) {
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
        fp2_sub_2x2x2(e, a, vR);
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_sub_2x2x2[i];
        __m512i* B = b_sub_2x2x2[i];
        __m512i* C = c_sub_2x2x2[i];

        fp2_sub_2x2x2(c, A, B);
        if (fp2_is_equal_2x2x2(c, C) != 0x3) {
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

    // Multiplication
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_mul_2x2x2[i];
        __m512i* B = b_mul_2x2x2[i];
        __m512i* C = c_mul_2x2x2[i];

        fp2_mul_2x2x2(c, A, B);
        if (fp2_is_equal_2x2x2(c, C) != 0x3) {
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_sqr_2x2x2[i];
        __m512i* C = c_sqr_2x2x2[i];

        fp2_sqr_2x2x2(c, A);
        if (fp2_is_equal_2x2x2(c, C) != 0x3) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
            break;
        }
    }
    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    // Preparing i
    fp2_setzero_2x2x2(a);
    a[0] = VSET(0, 0, 1, 0, 0, 0, 1, 0);
    fp2_tomont_2x2x2(a, a);

    for (int i = 0; i < iterations; i++) {
        rand_fp2_2x2x2(b);
        
        fp2_muli_2x2x2(c, b);
        fp2_mul_2x2x2(d, a, b);

        if (fp2_is_equal_2x2x2(c, d) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("mult by i", passed);
    if (passed == 0) {
        return false;
    }

    // Conjugation
    for (int i = 0; i < iterations; i++) {
        rand_fp2_2x2x2(a);
        rand_fp2_2x2x2(b);
        
        // conjugating twice = identity
        fp2_conjugate_2x2x2(c, a);
        fp2_conjugate_2x2x2(c, c);

        if (fp2_is_equal_2x2x2(a, c) != 0x3) {
            passed = 0;
            printf("conjugating twice is not identity!\n");
            break;
        }

        // conjugate(a) + conjugate(b) = conjugate(a+b)
        fp2_conjugate_2x2x2(c, a);
        fp2_conjugate_2x2x2(d, b);
        fp2_add_2x2x2(c, c, d);
        fp2_add_2x2x2(d, a, b);
        fp2_conjugate_2x2x2(d, d);
        if (fp2_is_equal_2x2x2(c, d) != 0x3) {
            passed = 0;
            printf("conjugate(a) + conjugate(b) != conjugate(a+b)\n");
            break;
        }

        // conjugate(a)*conjugate(b) = conjugate(a*b)
        fp2_conjugate_2x2x2(c, a);
        fp2_conjugate_2x2x2(d, b);
        fp2_mul_2x2x2(c, c, d);
        fp2_mul_2x2x2(d, a, b);
        fp2_conjugate_2x2x2(d, d);
        if (fp2_is_equal_2x2x2(c, d) != 0x3) {
            passed = 0;
            printf("conjugate(a)*conjugate(b) != conjugate(a*b)\n");
            break;
        }

        // a + conjugate(a) = 2*re(a)
        fp2_conjugate_2x2x2(c, a);
        fp2_add_2x2x2(c, c, a);
        fp2_add_2x2x2(d, a, a);
        if (fp_is_equal_4x2(c, d) & 0x5 != 0x5 || fp_is_zero_4x2(c) & 0xa != 0xa)
        {
            passed = 0;
            printf("a + conjugate(a) != 2*re(a)\n");
            break;
        }
    }
    print_test_res("conjugation", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 2
    for (int i = 0; i < iterations; i++) {
        rand_fp2_2x2x2(a);
        fp2_copy_2x2x2(b, a);

        fp2_div2_2x2x2(b, b);
        fp2_add_2x2x2(c, b, b);

        if (fp2_is_equal_2x2x2(a, c) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 2", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 3
    for (int i = 0; i < iterations; i++) {
        rand_fp2_2x2x2(a);
        fp2_copy_2x2x2(b, a);

        fp2_div3_2x2x2(b, b);
        fp2_add_2x2x2(c, b, b);
        fp2_add_2x2x2(c, c, b);

        if (fp2_is_equal_2x2x2(a, c) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 3", passed);
    if (passed == 0) {
        return false;
    }

    // Inverse
    fp2_inv_2x2x2(a, zero);
    if (fp2_is_zero_2x2x2(a) != 0x3) {
        passed = 0;
        printf("Inv does not output 0 on input 0\n");
    }

    for (int i = 0; i < iterations && passed; i++) {
        rand_fp2_2x2x2(a);

        if (fp2_is_zero_2x2x2(a))
            continue;
        
        fp2_inv_2x2x2(b, a);
        fp2_mul_2x2x2(b, b, a);

        if (fp2_is_equal_2x2x2(b, vR) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("inv", passed);
    if (passed == 0) {
        return false;
    }

    // Square root
    fp2_sqrt_2x2x2(a, zero);
    if (fp2_is_zero_2x2x2(a) != 0x3) {
        passed = 0;
        printf("sqrt(0) != 0\n");
    }

    for (int i = 0; i < iterations && passed; i++) {
        rand_fp2_2x2x2(a);
        
        fp2_sqr_2x2x2(b, a);
        fp2_sqrt_2x2x2(c, b);
        fp2_sqr_2x2x2(d, c);

        if (fp2_is_equal_2x2x2(b, d) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("sqrt", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 2x4x1 tests:\n");

    fp2_setzero_2x4x1(vR);
    for(int i = 0; i < NUM_LIMBS; i++) {
        vR[i] = VSET(R[i], 0, 0, 0, R[i], 0, 0, 0);
    }

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
        c[NUM_LIMBS-1] = VADD(c[NUM_LIMBS-1], t);
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
        fp2_add_2x4x1(e, a, vR);
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_add_2x4x1[i];
        __m512i* B = b_add_2x4x1[i];
        __m512i* C = c_add_2x4x1[i];

        fp2_add_2x4x1(c, A, B);
        if (fp2_is_equal_2x4x1(c, C) != 0x3) {
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
        fp2_sub_2x4x1(e, a, vR);
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_sub_2x4x1[i];
        __m512i* B = b_sub_2x4x1[i];
        __m512i* C = c_sub_2x4x1[i];

        fp2_sub_2x4x1(c, A, B);
        if (fp2_is_equal_2x4x1(c, C) != 0x3) {
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

    // Multiplication
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_mul_2x4x1[i];
        __m512i* B = b_mul_2x4x1[i];
        __m512i* C = c_mul_2x4x1[i];

        fp2_mul_2x4x1(c, A, B);
        if (fp2_is_equal_2x4x1(c, C) != 0x3) {
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
    for (int i = 0; i < NUM_FP2_KAT; i++) {
        __m512i* A = a_sqr_2x4x1[i];
        __m512i* C = c_sqr_2x4x1[i];

        fp2_sqr_2x4x1(c, A);
        if (fp2_is_equal_2x4x1(c, C) != 0x3) {
            passed = 0;
            printf("Failed KAT #%d\n", i);
            break;
        }
    }
    print_test_res("squaring", passed);
    if (passed == 0) {
        return false;
    }

    // Preparing i
    fp2_setzero_2x4x1(a);
    a[0] = VSET(0, 0, 1, 0, 0, 0, 1, 0);
    fp2_tomont_2x4x1(a, a);

    for (int i = 0; i < iterations; i++) {
        rand_fp2_2x4x1(b);
        
        fp2_muli_2x4x1(c, b);
        fp2_mul_2x4x1(d, a, b);

        if (fp2_is_equal_2x4x1(c, d) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("mult by i", passed);
    if (passed == 0) {
        return false;
    }

    // Conjugation
    for (int i = 0; i < iterations; i++) {
        rand_fp2_2x4x1(a);
        rand_fp2_2x4x1(b);
        
        // conjugating twice = identity
        fp2_conjugate_2x4x1(c, a);
        fp2_conjugate_2x4x1(c, c);

        if (fp2_is_equal_2x4x1(a, c) != 0x3) {
            passed = 0;
            printf("conjugating twice is not identity!\n");
            break;
        }

        // conjugate(a) + conjugate(b) = conjugate(a+b)
        fp2_conjugate_2x4x1(c, a);
        fp2_conjugate_2x4x1(d, b);
        fp2_add_2x4x1(c, c, d);
        fp2_add_2x4x1(d, a, b);
        fp2_conjugate_2x4x1(d, d);
        if (fp2_is_equal_2x4x1(c, d) != 0x3) {
            passed = 0;
            printf("conjugate(a) + conjugate(b) != conjugate(a+b)\n");
            break;
        }

        // conjugate(a)*conjugate(b) = conjugate(a*b)
        fp2_conjugate_2x4x1(c, a);
        fp2_conjugate_2x4x1(d, b);
        fp2_mul_2x4x1(c, c, d);
        fp2_mul_2x4x1(d, a, b);
        fp2_conjugate_2x4x1(d, d);
        if (fp2_is_equal_2x4x1(c, d) != 0x3) {
            passed = 0;
            printf("conjugate(a)*conjugate(b) != conjugate(a*b)\n");
            break;
        }

        // a + conjugate(a) = 2*re(a)
        fp2_conjugate_2x4x1(c, a);
        fp2_add_2x4x1(c, c, a);
        fp2_add_2x4x1(d, a, a);
        if (fp_is_equal_8x1(c, d) & 0x33 != 0x33 || fp_is_zero_8x1(c) & 0xcc != 0xcc)
        {
            passed = 0;
            printf("a + conjugate(a) != 2*re(a)\n");
            break;
        }
    }
    print_test_res("conjugation", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 2
    for (int i = 0; i < iterations; i++) {
        rand_fp2_2x4x1(a);
        fp2_copy_2x4x1(b, a);

        fp2_div2_2x4x1(b, b);
        fp2_add_2x4x1(c, b, b);

        if (fp2_is_equal_2x4x1(a, c) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 2", passed);
    if (passed == 0) {
        return false;
    }

    // Division by 3
    for (int i = 0; i < iterations; i++) {
        rand_fp2_2x4x1(a);
        fp2_copy_2x4x1(b, a);

        fp2_div3_2x4x1(b, b);
        fp2_add_2x4x1(c, b, b);
        fp2_add_2x4x1(c, c, b);

        if (fp2_is_equal_2x4x1(a, c) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("division by 3", passed);
    if (passed == 0) {
        return false;
    }

    // Inverse
    fp2_inv_2x4x1(a, zero);
    if (fp2_is_zero_2x4x1(a) != 0x3) {
        passed = 0;
        printf("Inv does not output 0 on input 0\n");
    }

    for (int i = 0; i < iterations; i++) {
        rand_fp2_2x4x1(a);

        if (fp2_is_zero_2x4x1(a))
            continue;
        
        fp2_inv_2x4x1(b, a);
        fp2_mul_2x4x1(b, b, a);

        if (fp2_is_equal_2x4x1(b, vR) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("inv", passed);
    if (passed == 0) {
        return false;
    }

    // Square root
    fp2_sqrt_2x4x1(a, zero);
    if (fp2_is_zero_2x4x1(a) != 0x3) {
        passed = 0;
        printf("sqrt(0) != 0\n");
    }

    for (int i = 0; i < iterations && passed; i++) {
        rand_fp2_2x4x1(a);
        
        fp2_sqr_2x4x1(b, a);
        fp2_sqrt_2x4x1(c, b);
        fp2_sqr_2x4x1(d, c);

        if (fp2_is_equal_2x4x1(b, d) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("sqrt", passed);
    if (passed == 0) {
        return false;
    }

    printf("Fp2 conversion tests:\n");

    for(int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(c);

        // Conversion back and forth is identity
        fp2_8x1x1_to_4x2x1(d, e, a);
        fp2_4x2x1_to_8x1x1(d, d, e);
        if (fp2_is_equal_8x1x1(d, a) != 0xff) {
            passed = 0;
            break;
        }

        fp2_4x2x1_to_8x1x1(d, b, c);
        fp2_8x1x1_to_4x2x1(e, f, d);
        if (fp2_is_equal_4x2x1(e, b) != 0xf || fp2_is_equal_4x2x1(f, c) != 0xf) {
            passed = 0;
            break;
        }

        // Conversion commutes with arithmetic
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        fp2_mul_8x1x1(c, a, b);

        fp2_8x1x1_to_4x2x1(d, e, a);
        fp2_8x1x1_to_4x2x1(f, g, b);
        fp2_mul_4x2x1(d, d, f);
        fp2_mul_4x2x1(e, e, g);
        fp2_4x2x1_to_8x1x1(d, d, e);

        if (fp2_is_equal_8x1x1(c, d) != 0xff) {
            passed = 0;
            break;
        }

        rand_fp2_4x2x1(a); rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(c); rand_fp2_4x2x1(d);
        fp2_mul_4x2x1(e, a, c);
        fp2_mul_4x2x1(f, b, d);

        fp2_4x2x1_to_8x1x1(a, a, b);
        fp2_4x2x1_to_8x1x1(c, c, d);
        fp2_mul_8x1x1(g, a, c);
        fp2_8x1x1_to_4x2x1(g, h, g);

        if (fp2_is_equal_4x2x1(e, g) != 0xf || fp2_is_equal_4x2x1(f, h) != 0xf) {
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
        rand_fp2_2x2x2(b);
        rand_fp2_2x2x2(c);

        // Conversion back and forth is identity
        fp2_4x2x1_to_2x2x2(d, e, a);
        fp2_2x2x2_to_4x2x1(d, d, e);
        if (fp2_is_equal_4x2x1(d, a) != 0xf) {
            passed = 0;
            printf("4x2x1 -> 2x2x2 -> 4x2x1 is not identity!\n");
            break;
        }

        fp2_2x2x2_to_4x2x1(f, b, c);
        fp2_4x2x1_to_2x2x2(g, h, f);
        if (fp2_is_equal_2x2x2(g, b) != 0x3 || fp2_is_equal_2x2x2(h, c) != 0x3) {
            passed = 0;
            printf("2x2x2 -> 4x2x1 -> 2x2x2 is not identity!\n");
            break;
        }

        // Top 13 bits of each limb is 0
        t = VSETALL(0xfff8000000000000);
        for(int j = 0; j < NUM_LIMBS; j++) {
            d[j] = VAND(d[j], t);
            f[j] = VAND(f[j], t);
        }
        if (fp2_is_zero_4x2x1(d) != 0xf || fp2_is_zero_4x2x1(f) != 0xf) {
            passed = 0;
            break;
        }

        // Conversion commutes with arithmetic
        rand_fp2_4x2x1(b);
        fp2_mul_4x2x1(c, a, b);

        fp2_4x2x1_to_2x2x2(d, e, a);
        fp2_4x2x1_to_2x2x2(f, g, b);
        fp2_mul_2x2x2(d, d, f);
        fp2_mul_2x2x2(e, e, g);
        fp2_2x2x2_to_4x2x1(d, d, e);

        if (fp2_is_equal_4x2x1(c, d) != 0xf) {
            passed = 0;
            printf("Conversions do not commute with mul!\n");
            break;
        }

        rand_fp2_2x2x2(a); rand_fp2_2x2x2(b);
        rand_fp2_2x2x2(c); rand_fp2_2x2x2(d);
        fp2_mul_2x2x2(e, a, c);
        fp2_mul_2x2x2(f, b, d);

        fp2_2x2x2_to_4x2x1(g, a, b);
        fp2_2x2x2_to_4x2x1(h, c, d);
        fp2_mul_4x2x1(g, g, h);
        fp2_4x2x1_to_2x2x2(g, h, g);

        if (fp2_is_equal_2x2x2(e, g) != 0x3 || fp2_is_equal_2x2x2(f, h) != 0x3) {
            passed = 0;
            printf("Conversions do not commute with mul!\n");
            break;
        }
    }
    print_test_res("4x2x1 <-> 2x2x2", passed);
    if (passed == 0) {
        return false;
    }

    for(int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(a);
        rand_fp2_2x4x1(b);
        rand_fp2_2x4x1(c);

        // Conversion back and forth is identity
        fp2_4x2x1_to_2x4x1(d, e, a);
        fp2_2x4x1_to_4x2x1(d, d, e);
        if (fp2_is_equal_4x2x1(d, a) != 0xf) {
            passed = 0;
            break;
        }

        fp2_2x4x1_to_4x2x1(f, b, c);
        fp2_4x2x1_to_2x4x1(g, h, f);
        if (fp2_is_equal_2x4x1(g, b) != 0x3 || fp2_is_equal_2x4x1(h, c) != 0x3) {
            passed = 0;
            break;
        }

        // Conversion commutes with arithmetic
        rand_fp2_4x2x1(b);
        fp2_mul_4x2x1(c, a, b);

        fp2_4x2x1_to_2x4x1(d, e, a);
        fp2_4x2x1_to_2x4x1(f, g, b);
        fp2_mul_2x4x1(d, d, f);
        fp2_mul_2x4x1(e, e, g);
        fp2_2x4x1_to_4x2x1(d, d, e);

        if (fp2_is_equal_4x2x1(c, d) != 0xf) {
            passed = 0;
            break;
        }

        rand_fp2_2x4x1(a); rand_fp2_2x4x1(b);
        rand_fp2_2x4x1(c); rand_fp2_2x4x1(d);
        fp2_mul_2x4x1(e, a, c);
        fp2_mul_2x4x1(f, b, d);

        fp2_2x4x1_to_4x2x1(g, a, b);
        fp2_2x4x1_to_4x2x1(h, c, d);
        fp2_mul_4x2x1(g, g, h);
        fp2_4x2x1_to_2x4x1(g, h, g);

        if (fp2_is_equal_2x4x1(e, g) != 0x3 || fp2_is_equal_2x4x1(f, h) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("4x2x1 <-> 2x4x1", passed);
    if (passed == 0) {
        return false;
    }

    for(int i = 0; i < iterations; i++) {
        rand_fp2_2x4x1(a);
        rand_fp2_2x2x2(b);

        // Conversion back and forth is identity
        fp2_2x4x1_to_2x2x2(d, a);
        fp2_2x2x2_to_2x4x1(e, d);
        if (fp2_is_equal_2x4x1(e, a) != 0x3) {
            passed = 0;
            break;
        }

        fp2_2x2x2_to_2x4x1(f, b);
        fp2_2x4x1_to_2x2x2(e, f);
        if (fp2_is_equal_2x2x2(e, b) != 0x3) {
            passed = 0;
            break;
        }

        // Conversion commutes with arithmetic
        rand_fp2_2x4x1(b);
        fp2_mul_2x4x1(c, a, b);

        fp2_2x4x1_to_2x2x2(d, a);
        fp2_2x4x1_to_2x2x2(e, b);
        fp2_mul_2x2x2(f, d, e);
        fp2_2x2x2_to_2x4x1(f, f);

        if (fp2_is_equal_2x4x1(c, f) != 0x3) {
            passed = 0;
            break;
        }

        rand_fp2_2x2x2(a);
        rand_fp2_2x2x2(b);
        fp2_mul_2x2x2(c, a, b);

        fp2_2x2x2_to_2x4x1(d, a);
        fp2_2x2x2_to_2x4x1(e, b);
        fp2_mul_2x4x1(f, d, e);
        fp2_2x4x1_to_2x2x2(e, f);

        if (fp2_is_equal_2x2x2(c, e) != 0x3) {
            passed = 0;
            break;
        }
    }
    print_test_res("2x2x2 <-> 2x4x1", passed);
    if (passed == 0) {
        return false;
    }

    return true;
}
