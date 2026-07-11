#include "ec.h"
#include "fp2.h"
#include "fp_consts.h"
#include "fp_unvec.h"
#include "theta_superglue.h"
#include "theta_middle.h" // TODO: move had, sqr, mul to theta.h? Since they are "core" functions.
#include "intrinsics_avx512.h"

#include "utils.h"
#include "stdio.h"
#include "assert.h"
#include "sqi_test_extras.h"

// ============================================================================
// MOVE TO theta_superglue.h:
//
// All step constants for superglue evaluation, in vectorized form. Filled
// once per isogeny step by superglue_codomain; consumed read-only by every
// superglue_eval_pair call. This replaces the scalar
// (position, sign_flip, dual_theta_null_square[4], elements[6], ab) interface,
// which forced superglue_evaluation_on_position to rebuild its vector
// constants (12 fp2_copy + 3 conversions) twice per evaluated point.
//
// typedef struct {
//     __m512i C0[2 * NUM_LIMBS]; // 8x1x1 multiplier of the first 8M block
//     __m512i C1[NUM_LIMBS];     // 4x2x1 broadcast: B_inv (pos 4) / e5 (pos 1|7)
//     __m512i J[2 * NUM_LIMBS];  // 8x1x1 (b, b, a, a, b, b, a, a) from ab
//     int position;
//     bool sign_flip;
// } superglue_precomp_t;
// ============================================================================

// Decode a 4-bit zero-flag mask (bit k = element k is zero) into a position,
// with the same priority order as the old scalar find_pos_early(e1, e0, e3, e2).
static int find_pos_from_mask(unsigned mask) {
    if (mask & 0x2)
        return 0;
    if (mask & 0x1)
        return 1;
    if (mask & 0x8)
        return 2;
    if (mask & 0x4)
        return 3;
    return -1;
}

bool compute_symmetric_position(int *pos_1, int *pos_2, __m512i* x_4, __m512i* z_4) {
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    fp2_t rs[8];
    __mmask8 zero_mask;
    // The positions we will find
    *pos_1 = *pos_2 = -1;

    // T0 = AVX_8x1x1(x1, x2, x3, x4, x1, x2, x3, x4)
    // T1 = AVX_8x1x1(z1, z2, z3, z4, z1, z2, z3, z4)
    // T2 = avx_811_add_sub(T0, T1)
    // r1, r2, r3, r4, s1, s2, s3, s4 = T2
    fp2_4x2x1_to_8x1x1(t0, x_4, x_4);
    fp2_4x2x1_to_8x1x1(t1, z_4, z_4);
    fp2_mixaddsub_8x1x1(t2, t0, t1, 0x55, 0x55);

    // For SQIsign, pos_2 is always 2 or 3, but pos_1 is unbounded. We can
    // return early here if we're in the lucky 2/3 cases. All eight zero
    // tests happen in one vectorized call; the unvectorize is only needed
    // on the slow path below.
    zero_mask = fp2_is_zero_8x1x1(t2);
    *pos_1 = find_pos_from_mask(zero_mask & 0xf);
    *pos_2 = find_pos_from_mask((zero_mask >> 4) & 0xf);

    if (*pos_1 != -1)
        return true;

    // If we've reached here, we need to do extra work to find pos_1
    // at a cost of 2M and 2a. Cold path: staging kept scalar, only the
    // final zero test is vectorized.
    fp2_8x1x1_to_unvectorized(rs, t2);
    // ir1 = i * r1
    fp2_mul_i(&rs[4], &rs[0], 1);
    fp2_copy(&rs[0], &rs[4]);

    //T0 = AVX_2x4x1(ir1, r2)
    //T1 = AVX_2x4x1(s2, s1)
    //T2 = avx_241_mul(T0, T1)
    fp2_copy(&rs[4], &rs[1]);
    fp2_unvectorized_to_2x4x1(t0, &rs[3]);
    fp2_copy(&rs[1], &rs[2]);
    fp2_unvectorized_to_2x4x1(t1, &rs[0]);
    fp2_mul_2x4x1(t2, t0, t1);

    //t1, t2 = T2
    //T0 = AVX_2x4x1(t1, t1)
    //T1 = AVX_2x4x1(t2, t2)
    //u1, u2 = avx_241_add_sub(T0, T1)
    fp2_2x4x1_to_unvectorized(rs, t2);
    fp2_copy(&rs[2], &rs[1]);
    fp2_copy(&rs[3], &rs[1]);
    fp2_unvectorized_to_2x4x1(t1, &rs[2]);
    fp2_copy(&rs[1], &rs[0]);
    fp2_unvectorized_to_2x4x1(t0, &rs[0]);
    fp2_mixaddsub_2x4x1(t2, t0, t1, 0x25);

    // Find the position in the case where neither P or Q is above (0 : 1)
    zero_mask = fp2_is_zero_2x4x1(t2);
    if (zero_mask & 0x1)
        *pos_1 = 4;
    else if (zero_mask & 0x2)
        *pos_1 = 5;
    else {
        printf("No vanishing coordinate found in 4-torsion");
        return false;
    }
    return true;
}

// Left scalar deliberately: pure memory management (copies, negs, zero-sets;
// no muls or adds) with six divergent placement patterns; runs once per gluing.
void compute_symmetric_elements(fp2_t g1[3], fp2_t g2[3], fp2_t g3[2],
        fp2_t *r1, fp2_t *r2, fp2_t *delta, int pos) {
    fp2_t ir1, ir2;
    // Almost no arithmetic here, just memory management and a sign flip
    //ir1, ir2 = zeta * r1, zeta * r2

    fp2_mul_i(&ir1, r1, 1);
    fp2_mul_i(&ir2, r2, 1);

    switch(pos) {
        case 0:
            fp2_set_zero(&g1[0]);
            fp2_copy(&g1[1], delta);
            fp2_copy(&g1[2], delta);
            fp2_copy(&g2[0], r1);
            fp2_neg(&g2[1], r2);
            fp2_copy(&g2[2], r2);
            fp2_copy(&g3[0], &ir2);
            fp2_neg(&g3[1], &ir1);
            fp2_copy(&g3[2], &ir1);
            break;
        case 1:
            fp2_set_zero(&g1[0]);
            fp2_neg(&g1[1], delta);
            fp2_neg(&g1[2], delta);
            fp2_copy(&g2[0], r1);
            fp2_neg(&g2[1], r2);
            fp2_copy(&g2[2], r2);
            fp2_neg(&g3[0], &ir2);
            fp2_copy(&g3[1], &ir1);
            fp2_neg(&g3[2], &ir1);
            break;
        case 2:
            fp2_copy(&g1[0], r1);
            fp2_neg(&g1[1], r2);
            fp2_copy(&g1[2], r2);
            fp2_set_zero(&g2[0]);
            fp2_copy(&g2[1], delta);
            fp2_copy(&g2[2], delta);
            fp2_neg(&g3[0], &ir2);
            fp2_copy(&g3[1], &ir1);
            fp2_neg(&g3[2], &ir1);
            break;
        case 3:
            fp2_copy(&g1[0], r1);
            fp2_neg(&g1[1], r2);
            fp2_copy(&g1[2], r2);
            fp2_set_zero(&g2[0]);
            fp2_neg(&g2[1], delta);
            fp2_neg(&g2[2], delta);
            fp2_copy(&g3[0], &ir2);
            fp2_neg(&g3[1], &ir1);
            fp2_copy(&g3[2], &ir1);
            break;
        case 4:
            fp2_copy(&g1[0], r1);
            fp2_neg(&g1[1], r2);
            fp2_copy(&g1[2], r2);
            fp2_neg(&g2[0], &ir2);
            fp2_copy(&g2[1], &ir1);
            fp2_neg(&g2[2], &ir1);
            fp2_set_zero(&g3[0]);
            fp2_neg(&g3[1], delta);
            fp2_neg(&g3[2], delta);
            break;
        case 5:
            fp2_copy(&g1[0], r1);
            fp2_neg(&g1[1], r2);
            fp2_copy(&g1[2], r2);
            fp2_copy(&g2[0], &ir2);
            fp2_neg(&g2[1], &ir1);
            fp2_copy(&g2[2], &ir1);
            fp2_set_zero(&g3[0]);
            fp2_copy(&g3[1], delta);
            fp2_copy(&g3[2], delta);
            break;
    }
}

bool compute_superglue_coefficients(int *position, fp2_t dual_theta_null_square[4],
        fp2_t elements[6], __m512i* x_4, __m512i* z_4) {
    int pos_1, pos_2;
    fp2_t t[8], u[8], v[4], s9[4], sq[4];
    fp2_t g0_1[3], g1_1[3], g2_1[3], g0_2[3], g1_2[3], g2_2[3];
    fp2_t ga[4], gb[4], arr8[8];
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    __m512i A4[NUM_LIMBS], B4[NUM_LIMBS], V4[NUM_LIMBS], S5v[NUM_LIMBS], S9v[NUM_LIMBS];
    __m512i Sum4[NUM_LIMBS], Diff4[NUM_LIMBS], E8[NUM_LIMBS], O8[NUM_LIMBS];

    // Compute the positions of each element
    // 2M + 10a
    compute_symmetric_position(&pos_1, &pos_2, x_4, z_4);
#ifndef NDEBUG
    // printf("\npos %d %d\n", pos_1, pos_2);
    assert(pos_2 == 2 || pos_2 == 3);
#endif

    // x1, x2, x3, _ = X_4 ; z1, z2, z3, _ = Z_4
    // four_torsion = (x2, z2, x3, z3) if pos_1 in {0, 1} else (x1, z1, x3, z3)
    // 8M (2M wasted)
    // T0 = AVX_8x1x1(x1', z1', x2', z2', x1', x2', 0, 0)
    // T1 = AVX_8x1x1(x1', z1', x2', z2', z1', z2', 0, 0)
    // T2 = avx_811_mul(T0, T1)
    // Staged directly from x_4/z_4 (4x2x1) with two-source maskz permutes:
    // the pos_1 branch only selects which x/z element plays x1'/z1', so it
    // reduces to a choice of index constants. No scalar extraction needed.
    {
        const bool low = (pos_1 == 0 || pos_1 == 1);   // x1' = x2 : x1
        const __m512i i0r = low ? VSET(2, 10, 4, 12, 2, 4, 0, 0)
                                : VSET(0, 8, 4, 12, 0, 4, 0, 0);
        const __m512i i0i = low ? VSET(3, 11, 5, 13, 3, 5, 0, 0)
                                : VSET(1, 9, 5, 13, 1, 5, 0, 0);
        const __m512i i1r = low ? VSET(2, 10, 4, 12, 10, 12, 0, 0)
                                : VSET(0, 8, 4, 12, 8, 12, 0, 0);
        const __m512i i1i = low ? VSET(3, 11, 5, 13, 11, 13, 0, 0)
                                : VSET(1, 9, 5, 13, 9, 13, 0, 0);
        for (int i = 0; i < NUM_LIMBS; i++) {
            t0[i]             = VBLENDPERMMASKZ(0x3f, x_4[i], z_4[i], i0r);
            t0[i + NUM_LIMBS] = VBLENDPERMMASKZ(0x3f, x_4[i], z_4[i], i0i);
            t1[i]             = VBLENDPERMMASKZ(0x3f, x_4[i], z_4[i], i1r);
            t1[i + NUM_LIMBS] = VBLENDPERMMASKZ(0x3f, x_4[i], z_4[i], i1i);
        }
    }
    fp2_mul_8x1x1(t2, t0, t1);
    // t2 = (s1, s2, s3, s4, r2, r4, 0, 0)

    // 8a (2a wasted)
    // T0 = AVX_8x1x1(s1, r2, s3, r4, s1, s3, 0, 0) = t2 elements (0,4,2,5,0,2)
    // T1 = AVX_8x1x1(s2, r2, s4, r4, s2, s4, 0, 0) = t2 elements (1,4,3,5,1,3)
    // coefficients = avx_811_add_sub(T0, T1) -- staged by two self-permutes
    {
        const __m512i iC0 = VSET(0, 4, 2, 5, 0, 2, 0, 0);
        const __m512i iC1 = VSET(1, 4, 3, 5, 1, 3, 0, 0);
        for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
            t0[i] = VPERM512MASKZ(0x3f, t2[i], iC0);
            t1[i] = VPERM512MASKZ(0x3f, t2[i], iC1);
        }
    }
    fp2_mixaddsub_8x1x1(t2, t0, t1, 0x0f, 0x0f);
    // r1, r2, r3, r4, delta_1, delta_2, _, _ = coefficients
    // Extracted to scalars only for compute_symmetric_elements, whose
    // placement logic stays scalar.
    fp2_8x1x1_to_unvectorized(t, t2);

    // g0_1, g1_1, g2_1 = compute_symmetric_elements(r1, r2, delta_1, pos_1)
    // g0_2, g1_2, g2_2 = compute_symmetric_elements(r3, r4, delta_2, pos_2)
    compute_symmetric_elements(g0_1, g1_1, g2_1, &t[0], &t[1], &t[4], pos_1);
    compute_symmetric_elements(g0_2, g1_2, g2_2, &t[2], &t[3], &t[5], pos_2);

    *position = 3 * (pos_1 >> 1) + (pos_2 >> 1);
#ifndef NDEBUG
    // Generically there's 9 elements here for the Q basis, but we actually only need 7
    assert(fp2_is_zero(&g1_2[0]));
    assert(fp2_is_equal(&g1_2[1], &g1_2[2]));
    // Handle special cases for positions
    assert(*position == 1 || *position == 4 || *position == 7);
#endif

    /* 8M
    T0 = AVX_8x1x1(
        g0_1[0], g0_1[0], g0_1[1], g0_1[1], delta_1, g1_1[0], g1_1[1], g1_1[1]
    )
    T1 = AVX_8x1x1(
        g0_2[0], g0_2[1], g0_2[0], g0_2[1], delta_2, g1_2[1], g1_2[0], g1_2[1]
    )
    T2 = avx_811_mul(T0, T1)
    Scalar-built: the g arrays come out of the scalar placement logic. */
    fp2_copy(&u[0], &g0_1[0]);
    fp2_copy(&u[1], &g0_1[0]);
    fp2_copy(&u[2], &g0_1[1]);
    fp2_copy(&u[3], &g0_1[1]);
    fp2_copy(&u[4], &t[4]);
    fp2_copy(&u[5], &g1_1[0]);
    fp2_copy(&u[6], &g1_1[1]);
    fp2_copy(&u[7], &g1_1[1]);
    fp2_unvectorized_to_8x1x1(t0, u);
    fp2_copy(&u[0], &g0_2[0]);
    fp2_copy(&u[1], &g0_2[1]);
    fp2_copy(&u[2], &g0_2[0]);
    fp2_copy(&u[3], &g0_2[1]);
    fp2_copy(&u[4], &t[5]);
    fp2_copy(&u[5], &g1_2[1]);
    fp2_copy(&u[6], &g1_2[0]);
    fp2_copy(&u[7], &g1_2[1]);
    fp2_unvectorized_to_8x1x1(t1, u);
    fp2_mul_8x1x1(t2, t0, t1);

    // 4a: v[j] = T2[j] + T2[j+4], fully in-vector: repack the two element
    // quadruples of t2 as 4x2x1 (one two-source permute each) and add.
    {
        const __m512i iP = VSET(0, 8, 1, 9, 2, 10, 3, 11);
        const __m512i iQ = VSET(4, 12, 5, 13, 6, 14, 7, 15);
        for (int i = 0; i < NUM_LIMBS; i++) {
            E8[i] = VBLENDPERM(iP, t2[i], t2[i + NUM_LIMBS]);
            O8[i] = VBLENDPERM(iQ, t2[i], t2[i + NUM_LIMBS]);
        }
    }
    fp_add_8x1(V4, E8, O8);            // V4 = (v0, v1, v2, v3)
    // NOTE: t2 stays live -- element 4 (S0[0]) feeds the S10 broadcast below.

    // 4M as one vector mul
    // S1 = AVX_4x2x1(g2_1[0], g2_1[0], g2_1[1], g2_1[1])
    // S2 = AVX_4x2x1(g2_2[0], g2_2[1], g2_2[0], g2_2[1])
    // S3 = avx_421_mul(S1, S2);  v = avx_421_sub(v, S3)
    fp2_copy(&ga[0], &g2_1[0]); fp2_copy(&ga[1], &g2_1[0]);
    fp2_copy(&ga[2], &g2_1[1]); fp2_copy(&ga[3], &g2_1[1]);
    fp2_copy(&gb[0], &g2_2[0]); fp2_copy(&gb[1], &g2_2[1]);
    fp2_copy(&gb[2], &g2_2[0]); fp2_copy(&gb[3], &g2_2[1]);
    fp2_unvectorized_to_4x2x1(A4, ga);
    fp2_unvectorized_to_4x2x1(B4, gb);
    fp2_mul_4x2x1(B4, A4, B4);
    fp_sub_8x1(V4, V4, B4);            // fp2 sub in 4x2x1 is lanewise Fp sub

    // 4M as one vector mul
    // S4 = AVX_4x2x1(g1_2[0], g1_2[2], g0_1[0], g0_1[2])
    // S5 = AVX_4x2x1(v[0], v[1], v[0], v[2]) -- per-limb self-permute of V4
    fp2_copy(&ga[0], &g1_2[0]); fp2_copy(&ga[1], &g1_2[1]);
    fp2_copy(&ga[2], &g0_1[0]); fp2_copy(&ga[3], &g0_1[2]);
    fp2_unvectorized_to_4x2x1(A4, ga);
    const __m512i idxS5 = VSET(0, 1, 2, 3, 0, 1, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        S5v[i] = VPERM512(idxS5, V4[i]);
    }
    fp2_mul_4x2x1(B4, S5v, A4);        // B4 = S6

    // 2a: R2 = (S6_0 + S6_1, S6_2 + S6_3), in-vector: swap adjacent element
    // slots (128-bit chunks within each 256-bit half) and add. R2_0 lands in
    // element slots 0-1, R2_1 in slots 2-3.
    for (int i = 0; i < NUM_LIMBS; i++) {
        S5v[i] = VPERM256(B4[i], 0x4E);
    }
    fp_add_8x1(S5v, B4, S5v);          // S5v = (R2_0, R2_0, R2_1, R2_1)

    // These first two are optional, but fit in a 4-way
    // 4M as one vector mul
    // S7 = AVX_4x2x1(g0_1[0], g0_1[2], delta_1, delta_2)  -- scalar-built
    // S8 = AVX_4x2x1(v[1], v[3], R2_0, R2_1)              -- one permute
    fp2_copy(&ga[0], &g0_1[0]); fp2_copy(&ga[1], &g0_1[2]);
    fp2_copy(&ga[2], &t[4]);    fp2_copy(&ga[3], &t[5]);
    fp2_unvectorized_to_4x2x1(A4, ga);
    const __m512i idxS8 = VSET(2, 3, 6, 7, 8, 9, 12, 13);
    for (int i = 0; i < NUM_LIMBS; i++) {
        B4[i] = VBLENDPERM(idxS8, V4[i], S5v[i]);
    }
    fp2_mul_4x2x1(S9v, A4, B4);        // S9v kept live: S11 + position 4

    // 4M, broadcast mul: S10 = S0[0] x4, taken straight from element 4 of
    // the 8M product -- one permute, no scalar round trip.
    const __m512i idxBC = VSET(4, 12, 4, 12, 4, 12, 4, 12);
    for (int i = 0; i < NUM_LIMBS; i++) {
        A4[i] = VBLENDPERM(idxBC, t2[i], t2[i + NUM_LIMBS]);
    }
    fp2_mul_4x2x1(V4, V4, A4);         // V4 = (v0, v1, v2, v3), finalized

    // Compute dual_theta_null_square for later.
    // S11 = AVX_4x2x1(S9_2, S9_3, v0, 0), staged in-vector; 3S batched;
    // only the three squares come out as scalars for the irregular combine.
    const __m512i idxS11 = VSET(4, 5, 6, 7, 8, 9, 0, 0);
    for (int i = 0; i < NUM_LIMBS; i++) {
        B4[i] = VBLENDPERMMASKZ(0x3f, S9v[i], V4[i], idxS11);
    }
    fp2_sqr_4x2x1(B4, B4);   // use fp2_mul_4x2x1(B4, B4, B4) if no sqr variant
    fp2_4x2x1_to_unvectorized(sq, B4);   // sq = (t4, t5, t6, _)

    // 3a: dual_theta_null_square = (t4 + t5, t6 - t4, t6 - t5, 0)
    fp2_add(&dual_theta_null_square[0], &sq[0], &sq[1]);
    fp2_sub(&dual_theta_null_square[1], &sq[2], &sq[0]);
    fp2_sub(&dual_theta_null_square[2], &sq[2], &sq[1]);
    fp2_set_zero(&dual_theta_null_square[3]);

    bool sign_flip = true;
    if (*position == 1 || *position == 7) {
        // case 1 | 7:
        // Standard case: all four coordinates are nonzero
        // 6M as one 8-way mul (2 wasted lanes); T3/T4 staged straight from
        // V4 with single-source permutes -- no scalar extraction on this path.
        // T3 = AVX_8x1x1(v0, v2, v0, v1, v0, v1, ., .)
        // T4 = AVX_8x1x1(v1, v3, v2, v3, v3, v2, ., .)
        const __m512i iT3r = VSET(0, 4, 0, 2, 0, 2, 0, 0);
        const __m512i iT3i = VSET(1, 5, 1, 3, 1, 3, 1, 1);
        const __m512i iT4r = VSET(2, 6, 4, 6, 6, 4, 2, 2);
        const __m512i iT4i = VSET(3, 7, 5, 7, 7, 5, 3, 3);
        for (int i = 0; i < NUM_LIMBS; i++) {
            t0[i]             = VPERM512(iT3r, V4[i]);
            t0[i + NUM_LIMBS] = VPERM512(iT3i, V4[i]);
            t1[i]             = VPERM512(iT4r, V4[i]);
            t1[i + NUM_LIMBS] = VPERM512(iT4i, V4[i]);
        }
        fp2_mul_8x1x1(t2, t0, t1);
        // t2 lanes 0-5 = T5[0..5] of the original comment; lanes 6,7 junk.

        // T8 = avx_811_add_sub over adjacent pairs = even/odd split + add/sub
        fp2_8x1x1_to_4x2x1(E8, O8, t2);
        fp_add_8x1(Sum4, E8, O8);    // (T8[0], T8[1], T8[2], .)
        fp_sub_8x1(Diff4, E8, O8);   // (T8[4], T8[5], T8[6], .)

        // elements = [T8[0], T8[4], T8[1], T8[5], T8[2], T8[6]] -- exactly
        // the even/odd interleave that fp2_4x2x1_to_8x1x1 implements.
        fp2_4x2x1_to_8x1x1(t2, Sum4, Diff4);
        fp2_8x1x1_to_unvectorized(arr8, t2);
        for (int k = 0; k < 6; k++)
            fp2_copy(&elements[k], &arr8[k]);
        sign_flip = false;
    } else if (*position == 4) {
        // Special case: one coordinate vanishes, sign_flip encodes which
        // square root branch to take. Rarely hot (1/3 of calls): the scalar
        // extractions of V4 and S9 now happen only here.
        fp2_t sgl_val;
        fp2_4x2x1_to_unvectorized(v, V4);
        fp2_4x2x1_to_unvectorized(s9, S9v);
        // 1a + 1M: sgl_val = (S9_0 + S9_1) * delta_2
        fp2_add(&sgl_val, &s9[0], &s9[1]);
        fp2_mul(&sgl_val, &sgl_val, &t[5]);

        // 4M + 3a
        // S16 = AVX_4x2x1(v[0], S9_3, v[0], v[0])
        // S17 = AVX_4x2x1(v[1], sgl_val, v[1], v[2])
        // S18 = avx_421_mul(S16, S17)
        fp2_mul(&t[0], &v[0], &v[1]);
        fp2_mul(&t[1], &s9[3], &sgl_val);
        fp2_mul(&t[2], &v[0], &v[1]);
        fp2_mul(&t[3], &v[0], &v[2]);

        // S19 = AVX_4x2x1(S18[0], 0, S18[0], S18[2])
        // S20 = AVX_4x2x1(S18[1], 0, S18[1], S18[3])
        // S21 = avx_421_add_sub(S19, S20)
        fp2_add(&v[0], &t[0], &t[1]);
        fp2_set_zero(&v[1]);
        fp2_sub(&v[2], &t[0], &t[1]);
        fp2_sub(&v[3], &t[2], &t[3]);
        // elements = [S21[0], S21[2], 0, 0, 0, 0]
        fp2_copy(&elements[0], &v[0]);
        fp2_copy(&elements[1], &v[2]);
        fp2_copy(&elements[2], &v[1]);
        fp2_copy(&elements[3], &v[1]);
        fp2_copy(&elements[4], &v[1]);
        fp2_copy(&elements[5], &v[1]);

        //sign_flip = S21[3] != 0
        sign_flip = !fp2_is_zero(&v[3]);
    } else {
        //case _:
        //    raise ValueError("unsupported case for SQIsign verification")
        printf("ValueError(unsupported case for SQIsign verification\n");
    }

    return sign_flip;
}

// Build the vectorized step constants from the scalar precomputation output.
// Runs ONCE per isogeny step (superglue_codomain calls it); afterwards no
// evaluation touches dual_theta_null_square/elements again.
// pos 4:   C0 = AVX_8x1x1(A_inv, e0, e1, C_inv, C_inv, e1, e0, A_inv)
//          C1 = AVX_4x2x1(B_inv x4)   -- only lanes 0,1 are ever read
// pos 1|7: C0 = AVX_8x1x1(A|C, e0, e2, e4, B_inv, e3, C|A, e1)
//          C1 = AVX_4x2x1(e5 x4)      -- only lanes 6,7 are ever read
void superglue_precompute_consts(superglue_precomp_t *K, int position,
        bool sign_flip, fp2_t dual_theta_null_square[4], fp2_t elements[6])
{
    fp2_t c8[8], ce[4];

    K->position = position;
    K->sign_flip = sign_flip;

    if (position == 4) {
        fp2_copy(&c8[0], &dual_theta_null_square[0]);
        fp2_copy(&c8[1], &elements[0]);
        fp2_copy(&c8[2], &elements[1]);
        fp2_copy(&c8[3], &dual_theta_null_square[2]);
        fp2_copy(&c8[4], &dual_theta_null_square[2]);
        fp2_copy(&c8[5], &elements[1]);
        fp2_copy(&c8[6], &elements[0]);
        fp2_copy(&c8[7], &dual_theta_null_square[0]);
        fp2_unvectorized_to_8x1x1(K->C0, c8);
        for (int k = 0; k < 4; k++)
            fp2_copy(&ce[k], &dual_theta_null_square[1]);
        fp2_unvectorized_to_4x2x1(K->C1, ce);
    } else if (position == 1 || position == 7) {
        fp2_copy(&c8[0], (position == 7) ? &dual_theta_null_square[2]
                                         : &dual_theta_null_square[0]);
        fp2_copy(&c8[1], &elements[0]);
        fp2_copy(&c8[2], &elements[2]);
        fp2_copy(&c8[3], &elements[4]);
        fp2_copy(&c8[4], &dual_theta_null_square[1]);
        fp2_copy(&c8[5], &elements[3]);
        fp2_copy(&c8[6], (position == 7) ? &dual_theta_null_square[0]
                                         : &dual_theta_null_square[2]);
        fp2_copy(&c8[7], &elements[1]);
        fp2_unvectorized_to_8x1x1(K->C0, c8);
        for (int k = 0; k < 4; k++)
            fp2_copy(&ce[k], &elements[5]);
        fp2_unvectorized_to_4x2x1(K->C1, ce);
    }
    // K->J is filled by superglue_precompute_eval_J once ab is available.
}

// Precompute the J multiplier for superglue_eval_pair from the ab output of
// superglue_codomain (stored in A24_12). Called once per step at the end of
// superglue_codomain -- hoists 2*NUM_LIMBS port-5 permutes out of the
// per-point path.
// a, b = ab;  J = AVX_8x1x1(b, b, a, a, b, b, a, a)
void superglue_precompute_eval_J(superglue_precomp_t *K, __m512i *ab)
{
    const __m512i idx2 = VSET(4, 12, 0, 8, 4, 12, 0, 8);
    const __m512i idx3 = VSET(6, 14, 2, 10, 6, 14, 2, 10);
    for (int i = 0; i < NUM_LIMBS; i++) {
        K->J[i] = VPERM512(idx2, ab[i]);
        K->J[i + NUM_LIMBS] = VPERM512(idx3, ab[i]);
    }
}

// Signature change: s[8] (scalar fp2_t) is replaced by S in 8x1x1 layout,
// and (position, dual_theta_null_square, elements) by the precomputed K --
// the constant staging that used to run here (twice per evaluated point)
// now lives in superglue_precompute_consts, once per step.
bool superglue_evaluation_on_position(__m512i *_p, __m512i *S,
        const superglue_precomp_t *K)
{
    __m512i T0[2 * NUM_LIMBS], T1[2 * NUM_LIMBS], T2[2 * NUM_LIMBS];
    __m512i E[NUM_LIMBS], O[NUM_LIMBS], R[NUM_LIMBS];
    __m512i *C0 = (__m512i *)K->C0;   // read-only; casts for the fp2_* API
    const __m512i *C1 = K->C1;
    const int position = K->position;
    // S = AVX_8x1x1(s0, s1, s2, s3, s4, s5, s6, s7)

    if (position == 4) {
        // 8M
        // T0 = AVX_8x1x1(s1, s5, s1, s5, s1, s5, s1, s5)
        // T2 = avx_811_mul(T0, C0)
        const __m512i idxS = VSET(1, 5, 1, 5, 1, 5, 1, 5);
        for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
            T0[i] = VPERM512(idxS, S[i]);
        }
        fp2_mul_8x1x1(T2, T0, C0);

        // 4a: adjacent-pair sums via even/odd split
        // R = AVX_4x2x1(r0, r1, r2, r3), rj = T2[2j] + T2[2j+1]
        fp2_8x1x1_to_4x2x1(E, O, T2);
        fp_add_8x1(R, E, O);

        // 8M (2 wasted), restaged so the final pair-sum IS the output:
        // T0 = AVX_8x1x1(s0, s4, s7, 0, s0, s4, s6, 0)
        // T1 = AVX_8x1x1(r0, r1, B_inv, ., r2, r3, B_inv, .)
        // T2 = (m0, m1, s7*B, 0, m2, m3, s6*B, 0)
        const __m512i idxT0  = VSET(0, 4, 7, 0, 0, 4, 6, 0);  // maskz zeroes lanes 3,7
        const __m512i idxT1r = VSET(0, 2, 8, 0, 4, 6, 8, 0);  // 8 -> C1 lane 0 = B.re
        const __m512i idxT1i = VSET(1, 3, 9, 1, 5, 7, 9, 1);  // 9 -> C1 lane 1 = B.im
        for (__mmask8 i = 0; i < NUM_LIMBS; i++) {
            T0[i]             = VPERM512MASKZ(0x77, S[i], idxT0);
            T0[i + NUM_LIMBS] = VPERM512MASKZ(0x77, S[i + NUM_LIMBS], idxT0);
            T1[i]             = VBLENDPERM(idxT1r, R[i], C1[i]);
            T1[i + NUM_LIMBS] = VBLENDPERM(idxT1i, R[i], C1[i]);
        }
        fp2_mul_8x1x1(T2, T0, T1);

        // final 4a doubles as the return packaging:
        // _p = AVX_4x2x1(m0 + m1, s7*B_inv, m2 + m3, s6*B_inv)
        fp2_8x1x1_to_4x2x1(E, O, T2);
        fp_add_8x1(_p, E, O);
    } else if (position == 1 || position == 7) {
        // 8M
        // T0 = AVX_8x1x1(s1, s5, s1, s5, s0, s4, s1, s5)
        // T2 = avx_811_mul(T0, C0) = (t2_0 .. t2_7)
        const __m512i idxS = VSET(1, 5, 1, 5, 0, 4, 1, 5);
        for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
            T0[i] = VPERM512(idxS, S[i]);
        }
        fp2_mul_8x1x1(T2, T0, C0);

        // 4a: R = AVX_4x2x1(t5_0, t5_1, ., .), then park e5 in the unused
        // slot 3 (lanes 6,7) so the next staging needs only two sources.
        fp2_8x1x1_to_4x2x1(E, O, T2);
        fp_add_8x1(R, E, O);
        for (__mmask8 i = 0; i < NUM_LIMBS; i++) {
            R[i] = VBLEND(0xC0, R[i], C1[i]);
        }

        // 8M (1 wasted), restaged; the pos-7 output reversal lives in the indices:
        // pos 1: T0 = (s0, s4, s3, s3, s2, s2, s6, 0)
        //        T1 = (t5_0, t5_1, t2_4, t2_5, t2_6, t2_7, e5, .)
        // pos 7: T0 = (s2, s2, s3, s3, s0, s4, s6, 0)
        //        T1 = (t2_6, t2_7, t2_4, t2_5, t5_0, t5_1, e5, .)
        const __m512i idxT6 = (position == 7)
            ? VSET(2, 2, 3, 3, 0, 4, 6, 0)
            : VSET(0, 4, 3, 3, 2, 2, 6, 0);
        const __m512i idxT7r = (position == 7)
            ? VSET(14, 15, 12, 13, 0, 2, 6, 0)
            : VSET(0, 2, 12, 13, 14, 15, 6, 0);
        const __m512i idxT7i = (position == 7)
            ? VSET(14, 15, 12, 13, 1, 3, 7, 1)
            : VSET(1, 3, 12, 13, 14, 15, 7, 1);
        for (__mmask8 i = 0; i < NUM_LIMBS; i++) {
            T0[i]             = VPERM512MASKZ(0x7f, S[i], idxT6);
            T0[i + NUM_LIMBS] = VPERM512MASKZ(0x7f, S[i + NUM_LIMBS], idxT6);
            T1[i]             = VBLENDPERM(idxT7r, R[i], T2[i]);
            T1[i + NUM_LIMBS] = VBLENDPERM(idxT7i, R[i], T2[i + NUM_LIMBS]);
        }
        fp2_mul_8x1x1(T2, T0, T1);
        // pos 1: T2 = (t8_0, t8_1, t8_2, t8_3, t8_4, t8_5, s6*e5, 0)
        // pos 7: T2 = (t8_4, t8_5, t8_2, t8_3, t8_0, t8_1, s6*e5, 0)

        // final 4a doubles as the return packaging:
        // pos 1: _p = AVX_4x2x1(b0, b1, b2, s6*e5)
        // pos 7: _p = AVX_4x2x1(b2, b1, b0, s6*e5)
        fp2_8x1x1_to_4x2x1(E, O, T2);
        fp_add_8x1(_p, E, O);
    } else {
        // All other cases wont be touched by SQIsign
        printf("ValueError(evaluation failed due to unsupported case\n");
        return false;
    }

    return true;
}

// Shared tail of the two evaluation-pair variants. On entry:
//   T2 = products of the 4M+4S block (elements t0_0 .. t0_7 pre-Hadamard)
//   T3 = products of the 8M block   (elements t1_0 .. t1_7)
// Both are consumed/overwritten.
static void superglue_evaluation_finish(__m512i *p1, __m512i *p2,
        __m512i *T2, __m512i *T3, const superglue_precomp_t *K)
{
    __m512i T0[2 * NUM_LIMBS], T1[2 * NUM_LIMBS];
    __m512i u[2 * NUM_LIMBS], v[2 * NUM_LIMBS], w[2 * NUM_LIMBS];

    // 8 adds -- final Hadamard, restaged so outputs land grouped for packaging:
    // T2 := (t0[0], t0[1], t0[4], t0[5] | t0[2], t0[3], t0[6], t0[7])
    //     = (data_1 lanes 0-3          | data_2 lanes 0-3)
    const __m512i idx7 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    const __m512i idx8 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
        T0[i] = VPERM512(idx7, T2[i]);
        T1[i] = VPERM512(idx8, T2[i]);
    }
    fp2_mixaddsub_8x1x1(T2, T0, T1, 0x33, 0x33);

    // 4M as one 8-way mul (lanes 0,1,4,5 are don't-care):
    // T0 = (., ., t1[2], t0[4], ., ., t1[6], t0[6])
    // T1 = (., ., t1[3], t0[5], ., ., t1[7], t0[7])
    // w  = (., ., s2[0], s2[1], ., ., s2[2], s2[3])
    const __m512i idxA = VSET(0, 1, 2, 10, 4, 5, 6, 14);
    const __m512i idxB = VSET(0, 1, 11, 3, 4, 5, 15, 7);
    for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
        T0[i] = VBLENDPERM(idxA, T3[i], T2[i]);
        T1[i] = VBLENDPERM(idxB, T2[i], T3[i]);
    }
    fp2_mul_8x1x1(w, T0, T1);

    // 10 scalar adds -> 3 vector adds + 2 blends:
    // u := (2t1[0], 2t1[1], 4s2[0], s2[1] | 2t1[4], 2t1[5], 4s2[2], s2[3])
    //    = (data_1 lanes 4-7             | data_2 lanes 4-7)
    fp_add_8x1(u, T3, T3);                                        // 2*t1
    fp_add_8x1(u + NUM_LIMBS, T3 + NUM_LIMBS, T3 + NUM_LIMBS);
    fp_add_8x1(v, w, w);                                          // 2*s2
    fp_add_8x1(v + NUM_LIMBS, w + NUM_LIMBS, w + NUM_LIMBS);
    fp_add_8x1(v, v, v);                                          // 4*s2
    fp_add_8x1(v + NUM_LIMBS, v + NUM_LIMBS, v + NUM_LIMBS);
    for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
        v[i] = VBLEND(0x88, v[i], w[i]);   // lanes 3,7 back to 1*s2
        u[i] = VBLEND(0xCC, u[i], v[i]);   // splice s2 lanes into the 2*t1 lanes
    }

    // Package:
    // data_1 = AVX_8x1x1(T2[0], T2[1], T2[4], T2[5], T4[0], T4[1], T5[0], S2[1])
    // data_2 = AVX_8x1x1(T2[2], T2[3], T2[6], T2[7], T4[3], T4[4], T5[1], S2[3])
    //        = (low 256 bits of T2 | low 256 of u) and (high 256 | high 256)
    for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
        T0[i] = VPERM256HALF(T2[i], u[i], 0x44);
        T1[i] = VPERM256HALF(T2[i], u[i], 0xEE);
    }

    // This could also be batched, but when case is 1 | 7 we do a 8 way then 7 way mul
    // when case is 4 we do an 8 way then 6 way mul, we could only save time with batched
    // additions and the code would be so much harder to read.
    superglue_evaluation_on_position(p1, T0, K);
    superglue_evaluation_on_position(p2, T1, K);
}

void superglue_evaluation_pair(__m512i *p1, __m512i *p2, __m512i *U, __m512i *V,
        __m512i *W, const superglue_precomp_t *K)
{
    __m512i u[2 * NUM_LIMBS], v[2 * NUM_LIMBS], w[2 * NUM_LIMBS];
    __m512i T0[2 * NUM_LIMBS], T1[2 * NUM_LIMBS], T2[2 * NUM_LIMBS], T3[2 * NUM_LIMBS];

    // u1, u3, u2, u4 = U
    // v1, v3, v2, v4 = V
    // w1, w3, w2, w4 = W

    // Branch-free conditional negation of elements 0,1 of W (qword lanes 0-3).
    fp2_neg_4x2x1(T0, W);
    for (__mmask8 i = 0; i < NUM_LIMBS; i++) {
        W[i] = VBLEND(K->sign_flip ? 0x0f : 0x00, W[i], T0[i]);
    }

    // 8 adds
    // T0 = AVX_8x1x1(u1, u2, u3, u4, u1, u2, u3, u4)
    // T1 = AVX_8x1x1(v1, v2, v3, v4, v1, v2, v3, v4)
    // T2 = avx_811_add_sub(T0, T1)
    const __m512i idx0 = VSET(0, 4, 2, 6, 0, 4, 2, 6);
    const __m512i idx1 = VSET(1, 5, 3, 7, 1, 5, 3, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        u[i] = VPERM512(idx0, U[i]);
        u[i + NUM_LIMBS] = VPERM512(idx1, U[i]);
        v[i] = VPERM512(idx0, V[i]);
        v[i + NUM_LIMBS] = VPERM512(idx1, V[i]);
        w[i] = VPERM512(idx0, W[i]);
        w[i + NUM_LIMBS] = VPERM512(idx1, W[i]);
    }
    fp2_mixaddsub_8x1x1(T2, u, v, 0x0f, 0x0f);

    // 4M + 4S
    // T0 = AVX_8x1x1(T2[0], T2[1], w1, w2, T2[2], T2[3], w3, w4)
    // T1 = AVX_8x1x1(T2[4], T2[5], w1, w2, T2[6], T2[7], w3, w4)
    // T2 = avx_811_mul(T0, T1)
    const __m512i idx2 = VSET(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i idx3 = VSET(4, 5, 8, 9, 6, 7, 10, 11);
    for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
        T0[i] = VBLENDPERM(idx2, T2[i], w[i]);
        T1[i] = VBLENDPERM(idx3, T2[i], w[i]);
    }
    fp2_mul_8x1x1(T2, T0, T1);

    // 8M
    // T0 = AVX_8x1x1(u1, u2, v1, w2, u3, u4, v3, w4)
    // T1 = AVX_8x1x1(w1, w2, v2, w1, w3, w4, v4, w3)
    // T3 = avx_811_mul(T0, T1)          -- T3 holds t1[0..7], kept vectorized
    const __m512i idx4 = VSET(0, 1, 8, 0, 2, 3, 10, 0);
    const __m512i idx5 = VSET(0, 1, 2, 9, 4, 5, 6, 11);
    const __m512i idx6 = VSET(0, 1, 9, 0, 2, 3, 11, 2);
    for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
        T0[i] = VBLENDPERM(idx4, u[i], v[i]);
        T0[i] = VBLENDPERM(idx5, T0[i], w[i]);
        T1[i] = VBLENDPERM(idx6, w[i], v[i]);
    }
    fp2_mul_8x1x1(T3, T1, T0);

    superglue_evaluation_finish(p1, p2, T2, T3, K);
}

// Specialization for the codomain computation, where V = 0 identically:
//  * the initial add/sub degenerates (u +- 0 = u, and the idx0/idx1
//    deinterleave already duplicates the four elements across both halves,
//    so its output is just the deinterleaved U) -- saves 8 adds and a third
//    of the deinterleave permutes;
//  * the 4M+4S block's two operands coincide, so it is one 8-way squaring;
//  * the two lanes of the 8M block that read V become zeros via maskz staging.
void superglue_evaluation_pair_codomain(__m512i *p1, __m512i *p2, __m512i *U,
        __m512i *W, const superglue_precomp_t *K)
{
    __m512i u[2 * NUM_LIMBS], w[2 * NUM_LIMBS];
    __m512i T0[2 * NUM_LIMBS], T1[2 * NUM_LIMBS], T2[2 * NUM_LIMBS], T3[2 * NUM_LIMBS];

    // Branch-free conditional negation of elements 0,1 of W (qword lanes 0-3).
    fp2_neg_4x2x1(T0, W);
    for (__mmask8 i = 0; i < NUM_LIMBS; i++) {
        W[i] = VBLEND(K->sign_flip ? 0x0f : 0x00, W[i], T0[i]);
    }

    const __m512i idx0 = VSET(0, 4, 2, 6, 0, 4, 2, 6);
    const __m512i idx1 = VSET(1, 5, 3, 7, 1, 5, 3, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        u[i] = VPERM512(idx0, U[i]);
        u[i + NUM_LIMBS] = VPERM512(idx1, U[i]);
        w[i] = VPERM512(idx0, W[i]);
        w[i + NUM_LIMBS] = VPERM512(idx1, W[i]);
    }

    // 4M + 4S with V = 0: both operands equal
    // T0 = AVX_8x1x1(u1, u2, w1, w2, u3, u4, w3, w4); T2 = avx_811_sqr(T0)
    const __m512i idx2 = VSET(0, 1, 8, 9, 2, 3, 10, 11);
    for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
        T0[i] = VBLENDPERM(idx2, u[i], w[i]);
    }
    fp2_sqr_8x1x1(T2, T0);   // use fp2_mul_8x1x1(T2, T0, T0) if no sqr variant

    // 8M with V = 0: the V lanes (2 and 6) become 0*0
    // T0 = AVX_8x1x1(u1, u2, 0, w2, u3, u4, 0, w4)
    // T1 = AVX_8x1x1(w1, w2, 0, w1, w3, w4, 0, w3)
    const __m512i idx4 = VSET(0, 1, 0, 9, 2, 3, 0, 11);
    const __m512i idx5 = VSET(0, 1, 0, 0, 2, 3, 0, 2);
    for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
        T0[i] = VBLENDPERMMASKZ(0xBB, u[i], w[i], idx4);
        T1[i] = VPERM512MASKZ(0xBB, w[i], idx5);
    }
    fp2_mul_8x1x1(T3, T1, T0);

    superglue_evaluation_finish(p1, p2, T2, T3, K);
}

// TODO: Output (OA_inv, OA_inv) in 8x1x1 for special eval.
// Signature change: (position, sign_flip, dual_theta_null_square, elements)
// outputs are replaced by the precomputation bundle K, which also absorbs J.
// The scalar arrays are now locals -- they exist only long enough to be
// baked into K->C0/K->C1.
bool superglue_codomain(__m512i* res_OA, __m512i* A24_12,
        superglue_precomp_t *K, __m512i* x_8, __m512i* z_8) {
    __m512i x_8_[2*NUM_LIMBS], z_8_[2*NUM_LIMBS], x_4[NUM_LIMBS], z_4[NUM_LIMBS];
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS];
    __m512i p1[NUM_LIMBS], p2[NUM_LIMBS];
    fp2_t dual_theta_null_square[4], elements[6];
    int position;
    bool sign_flip;

    // X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)
    ec_jac_to_xz_4x4x2x1(x_8_, z_8_, x_8, z_8);

    // Double points to recover 4-torsion for change of basis
    ec_double_4x4x2x1(x_4, z_4, A24_12, x_8_, z_8_);

    sign_flip = compute_superglue_coefficients(&position, dual_theta_null_square, elements, x_4, z_4);
#ifndef NDEBUG
    // fp2_print("dtns0 = ", &dual_theta_null_square[0]);
    // fp2_print("dtns1 = ", &dual_theta_null_square[1]);
    // fp2_print("dtns2 = ", &dual_theta_null_square[2]);
    // fp2_print("dtns3 = ", &dual_theta_null_square[3]);
    // fp2_print("elem0 = ", &elements[0]);
    // fp2_print("elem1 = ", &elements[1]);
    // fp2_print("elem2 = ", &elements[2]);
    // fp2_print("elem3 = ", &elements[3]);
    // fp2_print("elem4 = ", &elements[4]);
    // fp2_print("elem5 = ", &elements[5]);
#endif

    // Bake all step constants into vector form, once. Every evaluation from
    // here on (including the two inside this function) reads only K.
    superglue_precompute_consts(K, position, sign_flip,
            dual_theta_null_square, elements);

    // (u, v, w) = (x_8, 0, z_8): V vanishes on the codomain path, so use the
    // specialized variant and pass the torsion coordinates directly -- both
    // arrays are dead after this call (W is modified in place).
    // 16M + 4S + 16a (was 20M + 24a)
    superglue_evaluation_pair_codomain(p1, p2, x_8_, z_8_, K);

    // 4M
    // T0 = AVX_4x2x1(A1[0], A1[1], A1[0], A1[1])
    // T1 = AVX_4x2x1(A2[0], A2[0], A2[2], A2[2])
    // T2 = avx_421_mul(T0, T1)
    const __m512i idx1v = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    const __m512i idx2v = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx1v, p1[i]);
        t1[i] = VPERM512(idx2v, p2[i]);
    }
    fp2_mul_4x2x1(t0, t0, t1);
    // ABCD = AVX_4x2x1(T2[0], T2[1], T2[2], 0)
    for (int i = 0; i < NUM_LIMBS; i++) {
        res_OA[i] = MASKZ(0x3f, t0[i]);
    }

    // 2M
    // T0 = AVX_2x4x1(A1[0], A1[2])
    // T1 = AVX_2x4x1(T2[3], ABCD[1])
    // ab = avx_241_mul(T0, T1)
    const __m512i idx4v = VSET(0, 0, 1, 0, 4, 0, 5, 0);
    const __m512i idx5v = VSET(6, 0, 7, 0, 2, 0, 3, 0);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512MASKZ(0x55, p1[i], idx4v);
        t0[i] = VPERM512MASKZ(0x55, t0[i], idx5v);
    }
    fp2_mul_2x4x1(A24_12, t1, t0);

    // ab is now known: finish the precomputation bundle.
    superglue_precompute_eval_J(K, A24_12);

    // 8a
    //codomain = hadamard(ABCD)
    theta_had_single(res_OA, res_OA);

    return true;
}

// Signature change: (ab, position, sign_flip, dual_theta_null_square,
// elements) collapse into the single precomputation bundle K.
void superglue_eval_pair(__m512i *imP_Q, __m512i *A_12, __m512i *xP,
        __m512i *yP, __m512i *zP, __m512i *T8x, __m512i *T8y, __m512i *T8z,
        const superglue_precomp_t *K) {
    __m512i u[NUM_LIMBS], v[NUM_LIMBS], w[NUM_LIMBS];
    __m512i p1[2*N4x2], p2[2*N4x2];

    // 16S + 48M + 28a
    // U, V, W = four_add_components(A_12, P, T8)
    ec_add_components_4x4x2x1(u, v, w, A_12, xP, yP, zP, T8x, T8y, T8z);

    // 20M + 24a
    superglue_evaluation_pair(p1, p2, u, v, w, K);

    // imP_Q = avx_421_to_811(P1, P2) -- P1's elements land in lanes 0-3 and
    // P2's in lanes 4-7 (grouped, NOT the interleaved order of
    // fp2_4x2x1_to_8x1x1), matching theta_had_pair_alt and the J layout.
    const __m512i idx0 = VSET(0, 2, 4, 6, 8, 10, 12, 14);
    const __m512i idx1 = VSET(1, 3, 5, 7, 9, 11, 13, 15);
    for (__mmask8 i = 0; i < NUM_LIMBS; i++) {
        __m512i s1 = p1[i], s2 = p2[i];
        imP_Q[i] = VBLENDPERM(idx0, s1, s2);
        imP_Q[i+NUM_LIMBS] = VBLENDPERM(idx1, s1, s2);
    }

    // 8M -- J = AVX_8x1x1(b, b, a, a, b, b, a, a), precomputed once per step
    fp2_mul_8x1x1(imP_Q, imP_Q, (__m512i *)K->J);

    // 16a
    //imP_Q = hadamard_pair(imP_Q)
    theta_had_pair_alt(imP_Q, imP_Q);
}