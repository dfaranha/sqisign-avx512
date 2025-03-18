#include <stdio.h>

#include "fp_lvl3.h"
#include "test.h"
#include "utils.h"

bool test_fp(int iterations)
{
    __m512i a[8], b[8], c[8], d[8], e[8], f[8], zero[8], one[8], t;
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
        while(i_fp < 8 && j_limb < 8) {
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
        if (fp_is_equal_8x1(d, zero) != 0xff) {
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
        rand_fp_8x1(c);

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
        if (fp_is_equal_8x1(e, zero) != 0xff) {
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
        if (fp_is_equal_8x1(d, zero) != 0xff) {
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
        if (fp_is_equal_8x1(d, zero) != 0xff) {
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
        if (fp_is_equal_8x1(d, zero) != 0xff) {
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
    
    return true;
}