#include <stdio.h>

#include "fp_lvl3.h"
#include "test.h"
#include "utils.h"

bool test_fp(int iterations)
{
    __m512i a[8], b[8], c[8], d[8], e[8], f[8], zero[8], t;
    fp_setzero_8x1(zero);
    int passed = 1;

    // Equality test
    for(int i = 0; i < iterations; i++){
        rand_fp_8x1(a);
        if (fp_is_equal_8x1(a, a) != 0xff) {
            passed = 0;
            break;
        }
        fp_copy_8x1(b, a);
        fp_copy_8x1(c, a);
        if (fp_is_equal_8x1(a, b) != 0xff) {
            passed = 0;
            break;
        }

        // We check that making one change in each limb has the intended effect
        // b will be a copy of a, except that one limb of a is different
        // c will start out as a, but gradually be changed limb by limb.
        // d is the collection of vectors that we use to change each limb

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
    print_test_res("Fp lvl3 8x1 - equality tests", passed);
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
        fp_setzero_8x1(d);
        d[0] = VSETALL(1);
        fp_add_8x1(e, a, d);
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
    print_test_res("Fp lvl3 8x1 - addition tests", passed);
    if (passed == 0) {
        return false;
    }
    
    return true;
}