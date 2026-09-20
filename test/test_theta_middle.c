#include <stdio.h>

#include "fp.h"
#include "fp2.h"
#include "fp_consts.h"
#include "test.h"
#include "theta_middle.h"
#include "utils.h"

// Under SUPERGLUE the pair functions store a pair as (a1, b1, c1, d1, a2, b2,
// c2, d2) rather than interleaved, so fp2_{8x1x1_to_4x2x1,4x2x1_to_8x1x1} do
// not split and join it. These do.
#ifdef SUPERGLUE
static void pair_split(__m512i* res1, __m512i* res2, __m512i* src)
{
    const __m512i idx_lo = VSET(0, 8, 1, 9, 2, 10, 3, 11);
    const __m512i idx_hi = VSET(4, 12, 5, 13, 6, 14, 7, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res1[i] = VBLENDPERM(idx_lo, src[i], src[i + NUM_LIMBS]);
        res2[i] = VBLENDPERM(idx_hi, src[i], src[i + NUM_LIMBS]);
    }
}

static void pair_join(__m512i* res, __m512i* src1, __m512i* src2)
{
    const __m512i idx_re = VSET(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i idx_im = VSET(1, 3, 5, 7, 9, 11, 13, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        __m512i re = VBLENDPERM(idx_re, src1[i], src2[i]);
        res[i + NUM_LIMBS] = VBLENDPERM(idx_im, src1[i], src2[i]);
        res[i] = re;
    }
}
#else
#define pair_split fp2_8x1x1_to_4x2x1
#define pair_join fp2_4x2x1_to_8x1x1
#endif

bool test_theta_middle(int iterations)
{
    __m512i a[2*NUM_LIMBS], b[2*NUM_LIMBS], c[2*NUM_LIMBS], d[2*NUM_LIMBS], e[2*NUM_LIMBS];
    __m512i f[2*NUM_LIMBS], g[2*NUM_LIMBS];
    int passed = 1;

    // ======================================================================================================
    //                                                 Single
    // ======================================================================================================

    printf("Theta middle tests:\n");

    // Hadamard
    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        
        // H^2 is mult by 4
        theta_had_single(d, a);
        theta_had_single(e, d);
        fp2_add_4x2x1(f, a, a);
        fp2_add_4x2x1(f, f, f);

        if (fp2_is_equal_4x2x1(e, f) != 0xf) {
            passed = 0;
            break;
        }

        // In place
        fp2_copy_4x2x1(d, a);
        theta_had_single(d, d);
        theta_had_single(e, a);

        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // Linear H*a + H*b = H*(a+b)
        theta_had_single(d, a);
        theta_had_single(e, b);
        fp2_add_4x2x1(d, d, e);
        fp2_add_4x2x1(f, a, b);
        theta_had_single(f, f);

        if (fp2_is_equal_4x2x1(d, f) != 0xf) {
            passed = 0;
            break;
        }
    }

    print_test_res("Hadamard single", passed);
    if (passed == 0) {
        return false;
    }

    // ======================================================================================================
    //                                                 Pair
    // ======================================================================================================

    // Hadamard
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        
        // H^2 is mult by 4
        theta_had_pair(d, a);
        theta_had_pair(e, d);
        fp2_add_8x1x1(f, a, a);
        fp2_add_8x1x1(f, f, f);

        if (fp2_is_equal_8x1x1(e, f) != 0xff) {
            passed = 0;
            break;
        }

        // In place
        fp2_copy_8x1x1(d, a);
        theta_had_pair(d, d);
        theta_had_pair(e, a);

        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // Linear H*a + H*b = H*(a+b)
        theta_had_pair(d, a);
        theta_had_pair(e, b);
        fp2_add_8x1x1(d, d, e);
        fp2_add_8x1x1(f, a, b);
        theta_had_pair(f, f);

        if (fp2_is_equal_8x1x1(d, f) != 0xff) {
            passed = 0;
            break;
        }

        // Pair matches single
        theta_had_pair(d, a);
        fp2_8x1x1_to_4x2x1(e, e + NUM_LIMBS, a);
        theta_had_single(e, e);
        theta_had_single(e + NUM_LIMBS, e + NUM_LIMBS);
        fp2_4x2x1_to_8x1x1(f, e, e + NUM_LIMBS);

        if (fp2_is_equal_8x1x1(d, f) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("Hadamard pair", passed);
    if (passed == 0) {
        return false;
    }

    // mul
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);

        // Pair matches single
        theta_mul_pair(d, a, b);
        fp2_8x1x1_to_4x2x1(e, e + NUM_LIMBS, a);
        theta_mul_single(f, e, b);
        theta_mul_single(f + NUM_LIMBS, e + NUM_LIMBS, b);
        fp2_4x2x1_to_8x1x1(e, f, f + NUM_LIMBS);

        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("mul", passed);
    if (passed == 0) {
        return false;
    }

    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(c);

        // Pair matches single
        pair_split(e, e + NUM_LIMBS, a);
        theta_double_single(f, e, b, c);
        theta_double_single(f + NUM_LIMBS, e + NUM_LIMBS, b, c);
        pair_join(e, f, f + NUM_LIMBS);

        pair_join(b, b, b);
        pair_join(c, c, c);
        theta_double_pair(d, a, b, c);
        
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("double", passed);
    if (passed == 0) {
        return false;
    }

    // TODO: Test is not using an actual dual theta point
    // eval
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);
        bool h1 = false;
        bool h2 = true;

        // Pair matches single
        fp2_8x1x1_to_4x2x1(e, e + NUM_LIMBS, a);
        theta_eval_single(f, e, b, h1, h2);
        theta_eval_single(f + NUM_LIMBS, e + NUM_LIMBS, b, h1, h2);
        fp2_4x2x1_to_8x1x1(e, f, f + NUM_LIMBS);

        fp2_4x2x1_to_8x1x1(b, b, b);
        theta_eval_pair(d, a, b, h1, h2);

        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("eval", passed);
    if (passed == 0) {
        return false;
    }

    return true;
}