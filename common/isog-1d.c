#include "isog-1d.h"
#include "fp_consts.h"
#include "fp_unvec_lvl3.h"
#include "fp2.h"
#include "ec.h"
#include <assert.h>
#include "sqi_test_extras.h"

// ============================== 2x4x1 ================================================

void two_isog_codom_1x2x4x1(__m512i* res_A24, __m512i* res_eval_data, __m512i* ker)
{
    __m512i t[8], idx0, idx1;
    __mmask8 mask;

    fp2_sqr_2x4x1(res_A24, ker);

    // A24_prime = C24_prime - A24_prime
    // TODO: Wastes a subtraction
    idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    mask = 0xf;
    for (int i = 0; i < 8; i++) {
        t[i] = VPERM512(idx0, res_A24[i]);
        res_A24[i] = VPERM512MASKZ(mask, res_A24[i], idx1);
    }
    fp2_sub_2x4x1(res_A24, t, res_A24);

    // res_eval_data = (Kx + Kz, Kx - Kz)
    for (int i = 0; i < 8; i++) {
        res_eval_data[i] = VPERM512(idx1, ker[i]);
        t[i] = VPERM512(idx0, ker[i]);
    }
    fp2_mixaddsub_2x4x1(res_eval_data, res_eval_data, t, mask);
}

void two_isog_eval_1x2x4x1(__m512i* Q, __m512i* eval_data, __m512i* P)
{
    __m512i t0[8], t1[8], idx0, idx1;
    __mmask8 mask;
    
    // t0 = (Px - Pz, Px + Pz)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    mask = 0xf0;
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }
    fp2_mixaddsub_2x4x1(t0, t0, t1, mask);

    fp2_mul_2x4x1(t1, t0, eval_data);

    mask = 0xf;
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, t1[i]);
        t1[i] = VPERM512(idx1, t1[i]);
    }
    fp2_mixaddsub_2x4x1(t1, t0, t1, mask);

    fp2_mul_2x4x1(Q, P, t1);
}

void four_isog_codom_1x2x4x1(__m512i* res_A24, __m512i* res_eval_data, __m512i* ker)
{
    __m512i t0[8], t1[8], idx0, idx1;
    __mmask8 mask;

    // res_eval_data[0] = (c1, c2) = (x - z, x + z)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    mask = 0xf0;
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, ker[i]);
        t1[i] = VPERM512(idx1, ker[i]);
    }
    fp2_mixaddsub_2x4x1(res_eval_data, t0, t1, mask);

    // t0 = (x^2, z^2)
    // res_A24 = (x^4, z^4)
    fp2_sqr_2x4x1(t0, ker);
    fp2_sqr_2x4x1(res_A24, t0);

    // t1 = (res_A24[1] - res_A24[0], t0[1] + t0[1])
    idx0 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
    for (int i = 0; i < 8; i++) {
        t1[i] = VBLENDPERM(idx0, res_A24[i], t0[i]);
        t0[i] = VBLENDPERM(idx1, res_A24[i], t0[i]);
    }
    fp2_mixaddsub_2x4x1(t1, t1, t0, mask);

    // res_A24[0] = t1[0]
    // res_eval_data[1] = (c2, c0), c0 = t1[1] + t1[1]
    idx0 = VSET(8, 9, 10, 11, 4, 5, 6, 7);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    mask = 0xf0;
    for (int i = 0; i < 8; i++) {
        res_A24[i] = VBLENDPERM(idx0, res_A24[i], t1[i]);
        t0[i] = VBLENDPERM(idx1, res_eval_data[i], t1[i]); // (c2, t1[1])
        t1[i] = VBLENDPERMMASKZ(mask, res_eval_data[i], t1[i], idx1); // (0, t1[1])
    }
    
    fp2_add_2x4x1(res_eval_data+8, t0, t1);
}

void four_isog_eval_1x2x4x1(__m512i* Q, __m512i* eval_data, __m512i* P)
{
    __m512i t0[8], t1[8], t2[8], idx0, idx1;
    __mmask8 mask;

    // t0 = (Px + Pz, Px - Pz)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    mask = 0xf;
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }
    fp2_mixaddsub_2x4x1(t0, t0, t1, mask);

    // t1 = (t0[0] * c1, t0[0] * t0[1])
    idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VBLENDPERM(idx1, eval_data[i], t0[i]);
    }
    fp2_mul_2x4x1(t1, t1, t2);

    // t0 = (t0[1] * c2, t1[1] * c0)
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < 8; i++) {
        t0[i] = VBLENDPERM(idx1, t0[i], t1[i]);
    }
    fp2_mul_2x4x1(t0, t0, eval_data+8);

    // t1 = (t1[0] + t0[0], t1[0] - t0[0])^2
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512(idx0, t1[i]);
        t2[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_2x4x1(t1, t1, t2, mask);
    fp2_sqr_2x4x1(t1, t1);

    // t0 = (t0[1] + t1[0], t0[1] - t1[1])
    idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_2x4x1(t0, t0, t1, mask);

    fp2_mul_2x4x1(Q, t0, t1);
}

// Purely in 1x2x4x1
bool isog_chain_1d_1x2x4x1(
    __m512i* res_A24,
    __m512i *A24,
    __m512i *ker,
    const unsigned int isog_len,
    __m512i *pts,
    const unsigned int num_pts,
    const bool verify
)
{
    int space = 1;
    for (int i = 1; i < isog_len; i *= 2) {
        space++;
    }

    __m512i strat[8*space], eval_data[16], test[8];
    uint16_t orders[space];
    orders[0] = isog_len;
    int current = 0;
    fp2_copy_2x4x1(strat, ker);
    fp2_copy_2x4x1(res_A24, A24);
    
    for (int j = 0; j < isog_len/2; j++) {
        assert(current >= 0);
        assert(orders[current] >= 1);

        // Strategy
        while (orders[current] != 2) {
            assert(orders[current] >= 3);
            // A new split will be added
            current++;
            fp2_copy_2x4x1(strat+8*current, strat + 8*(current-1));
            // if we copied from the very first element, then we perform one additional doubling
            int num_dbls = (orders[current - 1] / 4) * 2 + orders[current - 1] % 2;
            orders[current] = orders[current-1] - num_dbls;
            while (num_dbls--) {
                ec_double_1x2x4x1(strat+8*current, strat+8*current, res_A24);
            }
        }

        if (verify) {
            // P != 0
            int zmask = fp2_is_zero_2x4x1(strat + 8*current);
            int z_is_zero = (zmask & 0x2) == 0x2;
            if (z_is_zero) { return false; }

            // Q= 2 * P != 0 and XQ != 0 (disallow singular)
            ec_double_1x2x4x1(test, strat + 8*current, res_A24);
            zmask = fp2_is_zero_2x4x1(test);
            // if Q.x = 0 or Q.z = 0
            if (zmask) { return false; }

            // order must be 4
            ec_double_1x2x4x1(test, test, res_A24);
            zmask = fp2_is_zero_2x4x1(test);
            z_is_zero = (zmask & 0x2) == 0x2;
            if (!z_is_zero) { return false; }
        }

        // Evaluate 4-isogeny
        four_isog_codom_1x2x4x1(res_A24, eval_data, strat + 8*current);
         
        for (int i = 0; i < current; i++) {
            four_isog_eval_1x2x4x1(strat + 8*i, eval_data, strat + 8*i);
            orders[i] -= 2;
        }
        for (int i = 0; i < num_pts; i++) {
            four_isog_eval_1x2x4x1(pts + 8*i, eval_data, pts + 8*i);
        }
        current--;
    }
    assert(isog_len % 2 ? !current : current == -1);
    if (isog_len % 2 == 1) { // last 2-isogeny
        if (verify) {
            // P != 0 and P.X != 0 (disallow singular)
            int zmask = fp2_is_zero_2x4x1(strat);
            if (zmask) { return false; }

            // order must be 2
            ec_double_1x2x4x1(test, strat, A24);
            zmask = fp2_is_zero_2x4x1(test);
            int z_is_zero = (zmask & 0x2) == 0x2;
            if (!z_is_zero) { return false; }
        }

        two_isog_codom_1x2x4x1(res_A24, eval_data, strat);
        for (int i = 0; i < num_pts; i++) {
            two_isog_eval_1x2x4x1(pts+8*i, eval_data, pts+8*i);
        }
    }

    return true;
}

// Does the unvec <--> 1x2x4x1 conversion
bool isog_chain_1d_using_2x4x1(
    ec_curve_t* res_E2,
    ec_curve_t* E1,
    ec_point_t* ker,
    const unsigned int isog_len,
    ec_point_t* pts,
    const unsigned int num_pts,
    const bool verify
)
{
    __m512i A24[8], ker_vec[8], pts_vec[8*num_pts];
    ec_point_unvec_to_1x2x4x1(A24, &E1->A24);
    ec_point_unvec_to_1x2x4x1(ker_vec, ker);
    for (int i = 0; i < num_pts; i++) {
        ec_point_unvec_to_1x2x4x1(pts_vec + 8*i, pts+i);
    }

    bool success = isog_chain_1d_1x2x4x1(A24, A24, ker_vec, isog_len, pts_vec, num_pts, verify);

    ec_point_1x2x4x1_to_unvec(&res_E2->A24, A24);
    for (int i = 0; i < num_pts; i++) {
        ec_point_1x2x4x1_to_unvec(pts+i, pts_vec+8*i);
    }

    // Output curve in the form (A:C)
    A24_to_AC(res_E2, &res_E2->A24);
    res_E2->is_A24_computed_and_normalized = false;

    return success;
}

// ============================== 2x2x2 ================================================

void two_isog_codom_1x2x2x2(__m512i* res_A24, __m512i* res_eval_data, __m512i* ker)
{
    __m512i t[4], idx0, idx1;
    __mmask8 mask;

    fp2_sqr_2x2x2(res_A24, ker);

    // A24_prime = C24_prime - A24_prime
    idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    mask = 0xf;
    for (int i = 0; i < 4; i++) {
        t[i] = VPERM512(idx0, res_A24[i]);
        res_A24[i] = VPERM512MASKZ(mask, res_A24[i], idx1);
    }
    fp2_sub_2x2x2(res_A24, t, res_A24);

    // res_eval_data = (Kx + Kz, Kx - Kz)
    for (int i = 0; i < 4; i++) {
        res_eval_data[i] = VPERM512(idx1, ker[i]);
        t[i] = VPERM512(idx0, ker[i]);
    }
    fp2_mixaddsub_2x2x2(res_eval_data, res_eval_data, t, mask);
}

void two_isog_eval_1x2x2x2(__m512i* Q, __m512i* eval_data, __m512i* P)
{
    __m512i t0[4], t1[4], idx0, idx1;
    __mmask8 mask;
    
    // t0 = (Px - Pz, Px + Pz)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    mask = 0xf0;
    for (int i = 0; i < 4; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }
    fp2_mixaddsub_2x2x2(t0, t0, t1, mask);

    fp2_mul_2x2x2(t1, t0, eval_data);

    mask = 0xf;
    for (int i = 0; i < 4; i++) {
        t0[i] = VPERM512(idx0, t1[i]);
        t1[i] = VPERM512(idx1, t1[i]);
    }
    fp2_mixaddsub_2x2x2(t1, t0, t1, mask);

    fp2_mul_2x2x2(Q, P, t1);
}

void four_isog_codom_1x2x2x2(__m512i* res_A24, __m512i* res_eval_data, __m512i* ker)
{
    __m512i t0[4], t1[4], idx0, idx1;
    __mmask8 mask;

    // res_eval_data[0] = (c1, c2) = (x - z, x + z)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    mask = 0xf0;
    for (int i = 0; i < 4; i++) {
        t0[i] = VPERM512(idx0, ker[i]);
        t1[i] = VPERM512(idx1, ker[i]);
    }
    fp2_mixaddsub_2x2x2(res_eval_data, t0, t1, mask);

    // t0 = (x^2, z^2)
    // res_A24 = (x^4, z^4)
    fp2_sqr_2x2x2(t0, ker);
    fp2_sqr_2x2x2(res_A24, t0);

    // t1 = (res_A24[1] - res_A24[0], t0[1] + t0[1])
    idx0 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
    for (int i = 0; i < 4; i++) {
        t1[i] = VBLENDPERM(idx0, res_A24[i], t0[i]);
        t0[i] = VBLENDPERM(idx1, res_A24[i], t0[i]);
    }
    fp2_mixaddsub_2x2x2(t1, t1, t0, mask);

    // res_A24[0] = t1[0]
    // res_eval_data[1] = (c2, c0), c0 = t1[1] + t1[1]
    idx0 = VSET(8, 9, 10, 11, 4, 5, 6, 7);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    mask = 0xf0;
    for (int i = 0; i < 4; i++) {
        res_A24[i] = VBLENDPERM(idx0, res_A24[i], t1[i]);
        t0[i] = VBLENDPERM(idx1, res_eval_data[i], t1[i]); // (c2, t1[1])
        t1[i] = VBLENDPERMMASKZ(mask, res_eval_data[i], t1[i], idx1); // (0, t1[1])
    }
    
    fp2_add_2x2x2(res_eval_data+4, t0, t1);
}

void four_isog_eval_1x2x2x2(__m512i* Q, __m512i* eval_data, __m512i* P)
{
    __m512i t0[4], t1[4], t2[4], idx0, idx1;
    __mmask8 mask;

    // t0 = (Px + Pz, Px - Pz)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    mask = 0xf;
    for (int i = 0; i < 4; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }
    fp2_mixaddsub_2x2x2(t0, t0, t1, mask);

    // t1 = (t0[0] * c1, t0[0] * t0[1])
    idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
    for (int i = 0; i < 4; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VBLENDPERM(idx1, eval_data[i], t0[i]);
    }
    fp2_mul_2x2x2(t1, t1, t2);

    // t0 = (t0[1] * c2, t1[1] * c0)
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < 4; i++) {
        t0[i] = VBLENDPERM(idx1, t0[i], t1[i]);
    }
    fp2_mul_2x2x2(t0, t0, eval_data+4);

    // t1 = (t1[0] + t0[0], t1[0] - t0[0])^2
    for (int i = 0; i < 4; i++) {
        t1[i] = VPERM512(idx0, t1[i]);
        t2[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_2x2x2(t1, t1, t2, mask);
    fp2_sqr_2x2x2(t1, t1);

    // t0 = (t0[1] + t1[0], t0[1] - t1[1])
    idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < 4; i++) {
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_2x2x2(t0, t0, t1, mask);

    fp2_mul_2x2x2(Q, t0, t1);
}

void four_isog_eval_pair_4x2x1(__m512i* Q, __m512i* eval_data, __m512i* P)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS], idx0, idx1;
    __mmask8 mask;

    // t0 = (Px + Pz, Px - Pz)
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    mask = 0x33;
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask);

    // t1 = (t0[0] * c1, t0[0] * t0[1])
    idx1 = VSET(0, 1, 10, 11, 4, 5, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VBLENDPERM(idx1, eval_data[i], t0[i]);
    }
    fp2_mul_4x2x1(t1, t1, t2);

    // t0 = (t0[1] * c2, t1[1] * c0)
    idx1 = VSET(2, 3, 10, 11, 6, 7, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx1, t0[i], t1[i]);
    }
    fp2_mul_4x2x1(t0, t0, eval_data + NUM_LIMBS);

    // t1 = (t1[0] + t0[0], t1[0] - t0[0])^2
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t1[i]);
        t2[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_4x2x1(t1, t1, t2, mask);
    fp2_sqr_4x2x1(t1, t1);

    // t0 = (t0[1] + t1[0], t0[1] - t1[1])
    idx0 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask);

    fp2_mul_4x2x1(Q, t0, t1);
}

// (x1, z1, x2, z2) and (x3, z3, x4, z4) -> (x1, x3, z1, z3, x2, x4, z2, z4)
void four_isog_eval_quad_8x1x1(__m512i* Q, __m512i* eval_data, __m512i* P)
{
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS], idx0, idx1;
    __mmask8 mask;

    // t0 = (Px + Pz, Px - Pz)
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    mask = 0x33;
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }
    fp2_mixaddsub_8x1x1(t0, t0, t1, mask, mask);

    // t1 = (t0[0] * c1, t0[0] * t0[1])
    idx1 = VSET(0, 1, 10, 11, 4, 5, 14, 15);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VBLENDPERM(idx1, eval_data[i], t0[i]);
    }
    fp2_mul_8x1x1(t1, t1, t2);

    // t0 = (t0[1] * c2, t1[1] * c0)
    idx1 = VSET(2, 3, 10, 11, 6, 7, 14, 15);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx1, t0[i], t1[i]);
    }
    fp2_mul_8x1x1(t0, t0, eval_data + 2*NUM_LIMBS);

    // t1 = (t1[0] + t0[0], t1[0] - t0[0])^2
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t1[i]);
        t2[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_8x1x1(t1, t1, t2, mask, mask);
    fp2_sqr_8x1x1(t1, t1);

    // t0 = (t0[1] + t1[0], t0[1] - t1[1])
    idx0 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_8x1x1(t0, t0, t1, mask, mask);

    fp2_mul_8x1x1(Q, t0, t1);
}

bool isog_chain_1d_1x2x2x2(
    __m512i* res_A24,
    __m512i *A24,
    __m512i *ker,
    const unsigned int isog_len,
    __m512i *pts,
    const unsigned int num_pts,
    const bool verify
)
{
    uint64_t total = 0;
    uint64_t total2 = 0;

    int space = 1;
    for (int i = 1; i < isog_len; i *= 2) {
        space++;
    }
    __m512i strat[space * N4x2], test[N4x2];
    __m512i eval_data[2 * N4x2], eval_data_pair[2*NUM_LIMBS], eval_data_quad[4*NUM_LIMBS];
    uint16_t orders[space];
    orders[0] = isog_len;
    int current = 0;
    fp2_copy_2x2x2(strat, ker);
    fp2_copy_2x2x2(res_A24, A24);

    // Setting up pts
    int num_pts_quad = (num_pts) / 4; 
    int num_pts_rem = num_pts % 4;
    int num_pts_trip = num_pts_rem == 3 ? 1 : 0;
    int num_vec_coupled = (num_pts_quad + num_pts_trip) * 2*NUM_LIMBS;
    if (num_pts_rem == 2) {
        num_vec_coupled += NUM_LIMBS;
    } else if (num_pts_rem == 1) {
        num_vec_coupled += N4x2;
    }
    __m512i pts_coupled[num_vec_coupled], t[2*NUM_LIMBS];
    for (int i = 0; i < num_pts_quad; i++) {
        fp2_2x2x2_to_4x2x1(t, pts + (4*i)*N4x2, pts + (4*i+1)*N4x2);
        fp2_2x2x2_to_4x2x1(t + NUM_LIMBS, pts + (4*i+2)*N4x2, pts + (4*i+3)*N4x2);
        fp2_4x2x1_to_8x1x1(pts_coupled + i*2*NUM_LIMBS, t, t + NUM_LIMBS);
    }
    if (num_pts_rem == 3) {
        fp2_2x2x2_to_4x2x1(t, pts + (4*num_pts_quad)*N4x2, pts + (4*num_pts_quad + 1)*N4x2);
        fp2_2x2x2_to_4x2x1(t + NUM_LIMBS, pts + (4*num_pts_quad + 2)*N4x2, pts + (4*num_pts_quad + 2)*N4x2); // wasted slot
        fp2_4x2x1_to_8x1x1(pts_coupled + num_pts_quad*2*NUM_LIMBS, t, t + NUM_LIMBS);
    } else if (num_pts_rem == 2) {
        fp2_2x2x2_to_4x2x1(pts_coupled + num_pts_quad*2*NUM_LIMBS, pts + (4*num_pts_quad)*N4x2, pts + (4*num_pts_quad + 1)*N4x2);
    } else if (num_pts_rem == 1) {
        fp2_copy_2x2x2(pts_coupled + num_pts_quad*2*NUM_LIMBS, pts + (4*num_pts_quad)*N4x2);
    }
    
    for (int j = 0; j < isog_len/2; j++) {
        assert(current >= 0);
        assert(orders[current] >= 1);

        // Strategy
        while (orders[current] != 2) {
            assert(orders[current] >= 3);
            current++;
            fp2_copy_2x2x2(strat + current*N4x2, strat + (current-1)*N4x2);
            int num_dbls = (orders[current - 1] / 4) * 2 + orders[current - 1] % 2;
            orders[current] = orders[current-1] - num_dbls;
            while (num_dbls--) {
                ec_double_1x2x2x2(strat + current*N4x2, strat + current*N4x2, res_A24);
            }
        }

        if (verify) {
            int zmask = fp2_is_zero_2x2x2(strat + current*N4x2);
            int z_is_zero = (zmask & 0x2) == 0x2;
            if (z_is_zero) { return false; }

            ec_double_1x2x2x2(test, strat + current*N4x2, res_A24);
            zmask = fp2_is_zero_2x2x2(test);
            if (zmask) { return false; }

            ec_double_1x2x2x2(test, test, res_A24);
            zmask = fp2_is_zero_2x2x2(test);
            z_is_zero = (zmask & 0x2) == 0x2;
            if (!z_is_zero) { return false; }
        }

        four_isog_codom_1x2x2x2(res_A24, eval_data, strat + current*N4x2);
        
        fp2_2x2x2_to_4x2x1(eval_data_pair, eval_data, eval_data);
        fp2_2x2x2_to_4x2x1(eval_data_pair + NUM_LIMBS, eval_data + N4x2, eval_data + N4x2);
        fp2_4x2x1_to_8x1x1(eval_data_quad, eval_data_pair, eval_data_pair);
        fp2_4x2x1_to_8x1x1(eval_data_quad + 2*NUM_LIMBS, eval_data_pair + NUM_LIMBS, eval_data_pair + NUM_LIMBS);

        // Eval strat points
        // We need some of these as 2x2x2 next iteration, so for simplicity we convert back and forth each time
        int nquad = current / 4;
        int rem = current % 4;
        for (int i = 0; i < nquad; i++) {
            fp2_2x2x2_to_4x2x1(t, strat + N4x2*(4*i), strat + N4x2*(4*i+1));
            fp2_2x2x2_to_4x2x1(t+NUM_LIMBS, strat + N4x2*(4*i+2), strat + N4x2*(4*i+3));
            fp2_4x2x1_to_8x1x1(t, t, t+NUM_LIMBS);
            four_isog_eval_quad_8x1x1(t, eval_data_quad, t);
            fp2_8x1x1_to_4x2x1(t, t+NUM_LIMBS, t);
            fp2_4x2x1_to_2x2x2(strat + N4x2*(4*i), strat + N4x2*(4*i+1), t);
            fp2_4x2x1_to_2x2x2(strat + N4x2*(4*i+2), strat + N4x2*(4*i+3), t+NUM_LIMBS);
        }
        if (rem == 3) {
            fp2_2x2x2_to_4x2x1(t, strat + N4x2*(4*nquad), strat + N4x2*(4*nquad+1));
            fp2_2x2x2_to_4x2x1(t+NUM_LIMBS, strat + N4x2*(4*nquad+2), strat + N4x2*(4*nquad+2));
            fp2_4x2x1_to_8x1x1(t, t, t+NUM_LIMBS);
            four_isog_eval_quad_8x1x1(t, eval_data_quad, t);
            fp2_8x1x1_to_4x2x1(t, t+NUM_LIMBS, t);
            fp2_4x2x1_to_2x2x2(strat + N4x2*(4*nquad), strat + N4x2*(4*nquad+1), t);
            fp2_4x2x1_to_2x2x2(strat + N4x2*(4*nquad+2), t, t+NUM_LIMBS);
        } else if (rem == 2) {
            fp2_2x2x2_to_4x2x1(t, strat + N4x2*(4*nquad), strat + N4x2*(4*nquad+1));
            four_isog_eval_pair_4x2x1(t, eval_data_pair, t);
            fp2_4x2x1_to_2x2x2(strat + N4x2*(4*nquad), strat + N4x2*(4*nquad+1), t);
        } else if (rem == 1) {
            four_isog_eval_1x2x2x2(strat + N4x2*(4*nquad), eval_data, strat + N4x2*(4*nquad));
        }

        for (int i = 0; i < current; i++) {
            orders[i] -= 2;
        }

        // Eval pts
        for (int i = 0; i < num_pts_quad + num_pts_trip; i++) {
            int j = i*2*NUM_LIMBS;
            four_isog_eval_quad_8x1x1(pts_coupled + j, eval_data_quad, pts_coupled + j);
        }
        int j = num_pts_quad*2*NUM_LIMBS;
        if (num_pts_rem == 2) {
            four_isog_eval_pair_4x2x1(pts_coupled + j, eval_data_pair, pts_coupled + j);
        } else if (num_pts_rem == 1) {
            four_isog_eval_1x2x2x2(pts_coupled + j, eval_data, pts_coupled + j);
        }
        current--;
    }
    for (int i = 0; i < num_pts_quad; i++) {
        fp2_8x1x1_to_4x2x1(t, t+NUM_LIMBS, pts_coupled + i*2*NUM_LIMBS);
        fp2_4x2x1_to_2x2x2(pts + (4*i)*N4x2, pts + (4*i+1)*N4x2, t);
        fp2_4x2x1_to_2x2x2(pts + (4*i+2)*N4x2, pts + (4*i+3)*N4x2, t + NUM_LIMBS);
    }
    if (num_pts_rem == 3) {
        fp2_8x1x1_to_4x2x1(t, t+NUM_LIMBS, pts_coupled + num_pts_quad*2*NUM_LIMBS);
        fp2_4x2x1_to_2x2x2(pts + (4*num_pts_quad)*N4x2, pts + (4*num_pts_quad+1)*N4x2, t);
        fp2_4x2x1_to_2x2x2(pts + (4*num_pts_quad+2)*N4x2, t, t + NUM_LIMBS);
    } else if (num_pts_rem == 2) {
        fp2_4x2x1_to_2x2x2(pts + (4*num_pts_quad)*N4x2, pts + (4*num_pts_quad+1)*N4x2, pts_coupled + num_pts_quad*2*NUM_LIMBS);
    } else if (num_pts_rem == 1) {
        fp2_copy_2x2x2(pts + (4*num_pts_quad)*N4x2, pts_coupled + num_pts_quad*2*NUM_LIMBS);
    }

    assert(isog_len % 2 ? !current : current == -1);
    if (isog_len % 2 == 1) { // last 2-isogeny
        if (verify) {
            int zmask = fp2_is_zero_2x2x2(strat);
            if (zmask) { return false; }

            ec_double_1x2x2x2(test, strat, A24);
            zmask = fp2_is_zero_2x2x2(test);
            int z_is_zero = (zmask & 0x2) == 0x2;
            if (!z_is_zero) { return false; }
        }

        two_isog_codom_1x2x2x2(res_A24, eval_data, strat);
        for (int i = 0; i < num_pts; i++) {
            two_isog_eval_1x2x2x2(pts + i*N4x2, eval_data, pts + i*N4x2);
        }
    }

    return true;
}

// Does the unvec <--> 1x2x2x2 conversion
bool isog_chain_1d(
    ec_curve_t* res_E2,
    ec_curve_t* E1,
    ec_point_t* ker,
    const unsigned int isog_len,
    ec_point_t* pts,
    const unsigned int num_pts,
    const bool verify
)
{
    __m512i A24[4], ker_vec[4], pts_vec[4*num_pts];
    ec_point_unvec_to_1x2x2x2(A24, &E1->A24);
    ec_point_unvec_to_1x2x2x2(ker_vec, ker);
    for (int i = 0; i < num_pts; i++) {
        ec_point_unvec_to_1x2x2x2(pts_vec + 4*i, pts+i);
    }

    bool success = isog_chain_1d_1x2x2x2(A24, A24, ker_vec, isog_len, pts_vec, num_pts, verify);

    ec_point_1x2x2x2_to_unvec(&res_E2->A24, A24);
    for (int i = 0; i < num_pts; i++) {
        ec_point_1x2x2x2_to_unvec(pts+i, pts_vec+4*i);
    }

    // Output curve in the form (A:C)
    A24_to_AC(res_E2, &res_E2->A24);
    res_E2->is_A24_computed_and_normalized = false;

    return success;
}