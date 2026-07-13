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

// Conditional per-lane negation over one fp 8x1 limb set:
// lanes set in `neg` become p - x, others pass through.
// Same mixaddsub-with-zero idiom as fp2_conjugate_4x2x1; res == a is safe
// (the codebase already aliases mixaddsub operands, cf. fp2_muli_4x2x1).
static void fp_cneg_8x1(__m512i *res, __m512i *a, const __mmask8 neg)
{
    __m512i zero[NUM_LIMBS];
    fp_setzero_8x1(zero);
    fp_mixaddsub_8x1(res, zero, a, (__mmask8)~neg);
}

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

// The products the scalar tail computes are (i*r1*s2, r2*s1) with
// (r1, r2, s1, s2) = t2 lanes (0, 1, 2, 3), and t2 is still vector-resident
// when the path begins. Both 2x4x1 operands are therefore one two-source
// blend-permute of t2 each; the i-multiplication folds into the permute
// (re <- im source, im <- re source) plus a one-lane fp_cneg_8x1, exactly
// the idiom validated in compute_superglue_coefficients.
// 2x4x1 lane map: element A re/im at lanes 0/2, element B at 4/6; odd
// lanes zero (staged with maskz 0x55, matching fp2_4x2x1_to_2x4x1).
// ============================================================================
bool compute_symmetric_position(int *pos_1, int *pos_2, __m512i* x_4, __m512i* z_4) {
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    __mmask8 zero_mask;
    *pos_1 = *pos_2 = -1;
 
    // ---- unchanged fast path ---------------------------------------------
    fp2_4x2x1_to_8x1x1(t0, x_4, x_4);
    fp2_4x2x1_to_8x1x1(t1, z_4, z_4);
    fp2_mixaddsub_8x1x1(t2, t0, t1, 0x55, 0x55);
 
    zero_mask = fp2_is_zero_8x1x1(t2);
    *pos_1 = find_pos_from_mask(zero_mask & 0xf);
    *pos_2 = find_pos_from_mask((zero_mask >> 4) & 0xf);
 
    if (*pos_1 != -1)
        return true;
 
    // ---- cold path (position 7, ~1/3 of calls), now fully in-vector ------
    // T0 = 2x4x1(i*r1, r2): re(i*r1) = -im(r1) -> idx 8+0 with cneg lane 0;
    //                        im(i*r1) =  re(r1) -> idx 0.
    // T1 = 2x4x1(s2, s1)
    // T2 = avx_241_mul(T0, T1) = (i*r1*s2, r2*s1)
    {
        const __m512i iA = VSET(8, 0, 0, 0, 1, 0, 9, 0);
        const __m512i iB = VSET(3, 0, 11, 0, 2, 0, 10, 0);
        for (int i = 0; i < NUM_LIMBS; i++) {
            t0[i] = VBLENDPERMMASKZ(0x55, t2[i], t2[i + NUM_LIMBS], iA);
            t1[i] = VBLENDPERMMASKZ(0x55, t2[i], t2[i + NUM_LIMBS], iB);
        }
        fp_cneg_8x1(t0, t0, 0x01);
    }
    fp2_mul_2x4x1(t2, t0, t1);
 
    // u1, u2 = (P0 + P1, P0 - P1): two self-permutes + the original
    // mixaddsub (mask 0x25 kept verbatim; bit 5 falls on a zero lane).
    {
        const __m512i iX = VSET(0, 0, 2, 0, 0, 0, 2, 0);   // (P0, P0)
        const __m512i iY = VSET(4, 0, 6, 0, 4, 0, 6, 0);   // (P1, P1)
        for (int i = 0; i < NUM_LIMBS; i++) {
            t0[i] = VPERM512MASKZ(0x55, t2[i], iX);
            t1[i] = VPERM512MASKZ(0x55, t2[i], iY);
        }
    }
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
 
bool compute_superglue_coefficients(superglue_precomp_t *K,
        __m512i *x_4, __m512i *z_4)
{
    int pos_1, pos_2, position;
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS];
    __m512i co[2*NUM_LIMBS];   // coefficients (r1, r2, r3, r4, d1, d2, 0, 0), 8x1x1 -- live throughout
    __m512i P8[2*NUM_LIMBS];   // product of the 8M block, 8x1x1 -- element 4 feeds S10
    __m512i A4[NUM_LIMBS], B4[NUM_LIMBS], V4[NUM_LIMBS], S5v[NUM_LIMBS], S9v[NUM_LIMBS];
    __m512i Sum4[NUM_LIMBS], Diff4[NUM_LIMBS], E8[NUM_LIMBS], O8[NUM_LIMBS], D4[NUM_LIMBS];
    bool sign_flip = true;
 
    // Compute the positions of each element
    // 2M + 10a
    compute_symmetric_position(&pos_1, &pos_2, x_4, z_4);
#ifndef NDEBUG
    assert(pos_2 == 2 || pos_2 == 3);
#endif
 
    // ---- unchanged: four-torsion products --------------------------------
    // x1, x2, x3, _ = X_4 ; z1, z2, z3, _ = Z_4
    // 8M (2M wasted)
    // T0 = AVX_8x1x1(x1', z1', x2', z2', x1', x2', 0, 0)
    // T1 = AVX_8x1x1(x1', z1', x2', z2', z1', z2', 0, 0)
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
    fp2_mul_8x1x1(co, t0, t1);
    // co = (s1, s2, s3, s4, r2, r4, 0, 0)
 
    // 8a (2a wasted): coefficients = adjacent add/sub
    {
        const __m512i iC0 = VSET(0, 4, 2, 5, 0, 2, 0, 0);
        const __m512i iC1 = VSET(1, 4, 3, 5, 1, 3, 0, 0);
        for (__mmask8 i = 0; i < 2 * NUM_LIMBS; i++) {
            t0[i] = VPERM512MASKZ(0x3f, co[i], iC0);
            t1[i] = VPERM512MASKZ(0x3f, co[i], iC1);
        }
    }
    fp2_mixaddsub_8x1x1(co, t0, t1, 0x0f, 0x0f);
    // co = (r1, r2, r3, r4, delta_1, delta_2, 0, 0)
    // Element map for all tables below: 0=r1 1=r2 2=r3 3=r4 4=d1 5=d2.
    // NO scalar extraction: co stays live and every operand is staged from it.
 
    position = 3 * (pos_1 >> 1) + (pos_2 >> 1);
    K->position = position;
#ifndef NDEBUG
    // The former asserts on g1_2 (Z-pattern of kernel 2) hold by construction
    // now: pos_2 in {2,3} selects the (0, +-d2, +-d2) row of the tables.
    assert(position == 1 || position == 4 || position == 7);
#endif
    const bool p2lo = (pos_2 == 2);   // sign of kernel-2's Z/I patterns
 
    // ---- former compute_symmetric_elements + 8M block --------------------
    // T0 = (g0_1[0], g0_1[0], g0_1[1], g0_1[1], d1, g1_1[0], g1_1[1], g1_1[1])
    // per pos_1:
    //   0: ( 0,  0,  d1,  d1, d1,  r1, -r2, -r2)
    //   1: ( 0,  0, -d1, -d1, d1,  r1, -r2, -r2)
    //   2: (r1, r1, -r2, -r2, d1,   0,  d1,  d1)
    //   3: (r1, r1, -r2, -r2, d1,   0, -d1, -d1)
    //   4: (r1, r1, -r2, -r2, d1,-ir2, ir1, ir1)
    //   5: (r1, r1, -r2, -r2, d1, ir2,-ir1,-ir1)
    // 8x1x1 staging: idx < 8 reads the plain element, idx >= 8 reads the
    // i-swapped source (re half: co-im limbs; im half: co-re limbs); the
    // re/im halves need separate neg masks because +-i*x negates only one.
    {
        static const __mmask8 keepT0[6] = {0xFC, 0xFC, 0xDF, 0xDF, 0xFF, 0xFF};
        static const __mmask8 nrT0[6]   = {0xC0, 0xCC, 0x0C, 0xCC, 0xCC, 0x2C};
        static const __mmask8 niT0[6]   = {0xC0, 0xCC, 0x0C, 0xCC, 0x2C, 0xCC};
        const __m512i iT0 = (pos_1 < 2) ? VSET(0, 0, 4, 4, 4, 0, 1, 1)
                          : (pos_1 < 4) ? VSET(0, 0, 1, 1, 4, 0, 4, 4)
                                        : VSET(0, 0, 1, 1, 4, 9, 8, 8);
        // T1 = (g0_2[0], g0_2[1], g0_2[0], g0_2[1], d2, g1_2[1], g1_2[0], g1_2[1])
        //    = (r3, -r4, r3, -r4, d2, +-d2, 0, +-d2)     (+ if pos_2 == 2)
        const __m512i iT1 = VSET(2, 3, 2, 3, 5, 5, 0, 5);
        const __mmask8 nT1 = p2lo ? 0x0A : 0xAA;
 
        for (int i = 0; i < NUM_LIMBS; i++) {
            t0[i]             = VBLENDPERMMASKZ(keepT0[pos_1], co[i], co[i + NUM_LIMBS], iT0);
            t0[i + NUM_LIMBS] = VBLENDPERMMASKZ(keepT0[pos_1], co[i + NUM_LIMBS], co[i], iT0);
            t1[i]             = VBLENDPERMMASKZ(0xBF, co[i], co[i + NUM_LIMBS], iT1);
            t1[i + NUM_LIMBS] = VBLENDPERMMASKZ(0xBF, co[i + NUM_LIMBS], co[i], iT1);
        }
        fp_cneg_8x1(t0,             t0,             nrT0[pos_1]);
        fp_cneg_8x1(t0 + NUM_LIMBS, t0 + NUM_LIMBS, niT0[pos_1]);
        fp_cneg_8x1(t1,             t1,             nT1);
        fp_cneg_8x1(t1 + NUM_LIMBS, t1 + NUM_LIMBS, nT1);
    }
    fp2_mul_8x1x1(P8, t0, t1);
    // P8 element 4 = d1*d2 = S0[0]; feeds the S10 broadcast below.
 
    // 4a: v[j] = P8[j] + P8[j+4] -- repack element quadruples as 4x2x1, add.
    {
        const __m512i iP = VSET(0, 8, 1, 9, 2, 10, 3, 11);
        const __m512i iQ = VSET(4, 12, 5, 13, 6, 14, 7, 15);
        for (int i = 0; i < NUM_LIMBS; i++) {
            E8[i] = VBLENDPERM(iP, P8[i], P8[i + NUM_LIMBS]);
            O8[i] = VBLENDPERM(iQ, P8[i], P8[i + NUM_LIMBS]);
        }
    }
    fp_add_8x1(V4, E8, O8);            // V4 = (v0, v1, v2, v3)
 
    // ---- S1/S2 block: 4M as one vector mul -------------------------------
    // A4 = (g2_1[0], g2_1[0], g2_1[1], g2_1[1]) per pos_1:
    //   0,3: ( ir2,  ir2, -ir1, -ir1)     1,2: (-ir2, -ir2,  ir1,  ir1)
    //   4:   (   0,    0,  -d1,  -d1)     5:   (   0,    0,   d1,   d1)
    // B4 = (g2_2[0], g2_2[1], g2_2[0], g2_2[1])
    //    = (-ir4, ir3, -ir4, ir3) if pos_2 == 2, negated if pos_2 == 3.
    // 4x2x1 i-staging: re slot <- 8+e (im source), im slot <- e (re source).
    {
        static const __mmask8 keepS1[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xF0, 0xF0};
        static const __mmask8 negS1[6]  = {0xA5, 0x5A, 0x5A, 0xA5, 0xF0, 0x00};
        const __m512i iS1 = (pos_1 < 4) ? VSET(9, 1, 9, 1, 8, 0, 8, 0)
                                        : VSET(0, 0, 0, 0, 4, 12, 4, 12);
        const __m512i iS2 = VSET(11, 3, 10, 2, 11, 3, 10, 2);
        const __mmask8 nS2 = p2lo ? 0x66 : 0x99;
        for (int i = 0; i < NUM_LIMBS; i++) {
            A4[i] = VBLENDPERMMASKZ(keepS1[pos_1], co[i], co[i + NUM_LIMBS], iS1);
            B4[i] = VBLENDPERM(iS2, co[i], co[i + NUM_LIMBS]);
        }
        fp_cneg_8x1(A4, A4, negS1[pos_1]);
        fp_cneg_8x1(B4, B4, nS2);
    }
    fp2_mul_4x2x1(B4, A4, B4);
    fp_sub_8x1(V4, V4, B4);            // fp2 sub in 4x2x1 is lanewise Fp sub
 
    // ---- S4/S5/S6 block: 4M as one vector mul ----------------------------
    // A4 = (g1_2[0], g1_2[1], g0_1[0], g0_1[2])
    //    = (0, +-d2,   0, +-d1)  if pos_1 < 2   (d2: + iff pos_2==2; d1: + iff pos_1==0)
    //    = (0, +-d2,  r1,   r2)  otherwise
    {
        const __m512i iS4 = (pos_1 < 2) ? VSET(0, 0, 5, 13, 0, 0, 4, 12)
                                        : VSET(0, 0, 5, 13, 0, 8, 1, 9);
        const __mmask8 keep = (pos_1 < 2) ? 0xCC : 0xFC;
        const __mmask8 neg = (__mmask8)((p2lo ? 0x00 : 0x0C) | ((pos_1 == 1) ? 0xC0 : 0x00));
        for (int i = 0; i < NUM_LIMBS; i++) {
            A4[i] = VBLENDPERMMASKZ(keep, co[i], co[i + NUM_LIMBS], iS4);
        }
        fp_cneg_8x1(A4, A4, neg);
    }
    // S5 = (v0, v1, v0, v2) -- per-limb self-permute of V4 (unchanged)
    const __m512i idxS5 = VSET(0, 1, 2, 3, 0, 1, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        S5v[i] = VPERM512(idxS5, V4[i]);
    }
    fp2_mul_4x2x1(B4, S5v, A4);        // B4 = S6
 
    // 2a: R2 fold (unchanged)
    for (int i = 0; i < NUM_LIMBS; i++) {
        S5v[i] = VPERM256(B4[i], 0x4E);
    }
    fp_add_8x1(S5v, B4, S5v);          // S5v = (R2_0, R2_0, R2_1, R2_1)
 
    // ---- S7/S8 block: 4M as one vector mul -------------------------------
    // A4 = (g0_1[0], g0_1[2], d1, d2)
    //    = (0, +-d1, d1, d2) if pos_1 < 2 (+ iff pos_1==0), else (r1, r2, d1, d2)
    {
        const __m512i iS7 = (pos_1 < 2) ? VSET(0, 0, 4, 12, 4, 12, 5, 13)
                                        : VSET(0, 8, 1, 9, 4, 12, 5, 13);
        const __mmask8 keep = (pos_1 < 2) ? 0xFC : 0xFF;
        const __mmask8 neg = (pos_1 == 1) ? 0x0C : 0x00;
        for (int i = 0; i < NUM_LIMBS; i++) {
            A4[i] = VBLENDPERMMASKZ(keep, co[i], co[i + NUM_LIMBS], iS7);
        }
        fp_cneg_8x1(A4, A4, neg);
    }
    const __m512i idxS8 = VSET(2, 3, 6, 7, 8, 9, 12, 13);
    for (int i = 0; i < NUM_LIMBS; i++) {
        B4[i] = VBLENDPERM(idxS8, V4[i], S5v[i]);
    }
    fp2_mul_4x2x1(S9v, A4, B4);        // S9v kept live: S11 + position 4
 
    // ---- S10: broadcast mul (unchanged, source is now P8) ----------------
    const __m512i idxBC = VSET(4, 12, 4, 12, 4, 12, 4, 12);
    for (int i = 0; i < NUM_LIMBS; i++) {
        A4[i] = VBLENDPERM(idxBC, P8[i], P8[i + NUM_LIMBS]);
    }
    fp2_mul_4x2x1(V4, V4, A4);         // V4 = (v0, v1, v2, v3), finalized
 
    // ---- S11 + dual_theta_null_square, fully in-vector -------------------
    // squares Bsq = (t4, t5, t6, .) = ((S9_2)^2, (S9_3)^2, v0^2, .)
    const __m512i idxS11 = VSET(4, 5, 6, 7, 8, 9, 0, 0);
    for (int i = 0; i < NUM_LIMBS; i++) {
        B4[i] = VBLENDPERMMASKZ(0x3f, S9v[i], V4[i], idxS11);
    }
    fp2_sqr_4x2x1(B4, B4);   // use fp2_mul_4x2x1(B4, B4, B4) if no sqr variant
    // D4 = (t4 + t5, t6 - t4, t6 - t5, 0): two self-permutes + one mixaddsub.
    {
        const __m512i iDa = VSET(0, 1, 4, 5, 4, 5, 0, 0);   // (t4, t6, t6, .)
        const __m512i iDb = VSET(2, 3, 0, 1, 2, 3, 0, 0);   // (t5, t4, t5, .)
        for (int i = 0; i < NUM_LIMBS; i++) {
            A4[i] = VPERM512(iDa, B4[i]);
            S5v[i] = VPERM512(iDb, B4[i]);   // S5v dead, reuse as scratch
        }
        fp_mixaddsub_8x1(D4, A4, S5v, 0x03); // add elt 0, sub elts 1, 2
        for (int i = 0; i < NUM_LIMBS; i++) {
            D4[i] = MASKZ(0x3f, D4[i]);      // element 3 = 0
        }
    }
 
    // ---- per-position finish: stage K->C0 / K->C1 directly ---------------
    if (position == 1 || position == 7) {
        // Standard case: all four coordinates nonzero.
        // 6M as one 8-way mul (2 wasted lanes) -- unchanged staging.
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
        fp2_mul_8x1x1(t0, t0, t1);   // lanes 0-5 = T5[0..5]; 6,7 junk
 
        // T8 = add/sub over adjacent pairs = even/odd split + add/sub
        fp2_8x1x1_to_4x2x1(E8, O8, t0);
        fp_add_8x1(Sum4, E8, O8);    // (e0, e2, e4, .)  [T8_0, T8_1, T8_2]
        fp_sub_8x1(Diff4, E8, O8);   // (e1, e3, e5, .)  [T8_4, T8_5, T8_6]
 
        // C0 = (A|C, e0, e2, e4, B, e3, C|A, e1) with (A,B,C) = (D0,D1,D2):
        //   pos 1: (D0, e0, e2, e4, D1, e3, D2, e1)
        //   pos 7: (D2, e0, e2, e4, D1, e3, D0, e1)
        // staged 8x1x1 in two blend-permutes: Sum/Diff first, D second.
        {
            const __m512i iSFr = VSET(0, 0, 2, 4, 0, 10, 0, 8);
            const __m512i iSFi = VSET(0, 1, 3, 5, 0, 11, 0, 9);
            const __m512i iXDr = (position == 1)
                ? VSET(8, 1, 2, 3, 10, 5, 12, 7)
                : VSET(12, 1, 2, 3, 10, 5, 8, 7);
            const __m512i iXDi = (position == 1)
                ? VSET(9, 1, 2, 3, 11, 5, 13, 7)
                : VSET(13, 1, 2, 3, 11, 5, 9, 7);
            const __m512i iC1 = VSET(4, 5, 4, 5, 4, 5, 4, 5); // e5 = Diff4 elt 2
            for (int i = 0; i < NUM_LIMBS; i++) {
                K->C0[i]             = VBLENDPERM(iSFr, Sum4[i], Diff4[i]);
                K->C0[i + NUM_LIMBS] = VBLENDPERM(iSFi, Sum4[i], Diff4[i]);
                K->C0[i]             = VBLENDPERM(iXDr, K->C0[i], D4[i]);
                K->C0[i + NUM_LIMBS] = VBLENDPERM(iXDi, K->C0[i + NUM_LIMBS], D4[i]);
                K->C1[i]             = VPERM512(iC1, Diff4[i]);
            }
        }
        sign_flip = false;
    } else if (position == 4) {
        // Special case: one coordinate vanishes. Now vectorized:
        //   sgl   = (s9_0 + s9_1) * d2
        //   t0    = v0*v1,  t1 = s9_3 * sgl,  t3 = v0*v2
        //   e0    = t0 + t1,  e1 = t0 - t1,  sign_flip = (t0 - t3) != 0
        __m512i P4[NUM_LIMBS], Q4[NUM_LIMBS], G[NUM_LIMBS];
 
        // M1 = (v0, v0, s9_3, .) * (v1, v2, d2, .)   -- 4M, one lane idle
        const __m512i iM1a = VSET(0, 1, 0, 1, 14, 15, 0, 0);
        const __m512i iM1bA = VSET(2, 3, 4, 5, 0, 0, 0, 0);   // (v1, v2, ., .)
        const __m512i iM1bB = VSET(0, 1, 2, 3, 13, 0, 0, 0);  // splice re(d2)
        const __m512i iM1bC = VSET(0, 1, 2, 3, 4, 13, 0, 0);  // splice im(d2)
        for (int i = 0; i < NUM_LIMBS; i++) {
            A4[i] = VBLENDPERMMASKZ(0x3f, V4[i], S9v[i], iM1a);
            B4[i] = VPERM512(iM1bA, V4[i]);
            B4[i] = VBLENDPERM(iM1bB, B4[i], co[i]);
            B4[i] = VBLENDPERMMASKZ(0x3f, B4[i], co[i + NUM_LIMBS], iM1bC);
        }
        fp2_mul_4x2x1(P4, A4, B4);   // P4 = (v0*v1, v0*v2, s9_3*d2, 0)
 
        // q = s9_0 + s9_1 (lands in element slots 0 and 1)
        for (int i = 0; i < NUM_LIMBS; i++) {
            Q4[i] = VPERM256(S9v[i], 0x4E);
        }
        fp_add_8x1(Q4, Q4, S9v);
 
        // t1 = (s9_3*d2) * q -- 1 live lane; keep 4x2x1 for uniformity
        // (a 2x4x1 mul would halve the waste if you care on this branch)
        const __m512i iQa = VSET(4, 5, 0, 0, 0, 0, 0, 0);
        for (int i = 0; i < NUM_LIMBS; i++) {
            A4[i] = VPERM512(iQa, P4[i]);
        }
        fp2_mul_4x2x1(Q4, A4, Q4);   // Q4 elt 0 = t1; other lanes junk
 
        // G = (t0 + t1, t0 - t1, t0 - t3, 0)
        const __m512i iX = VSET(0, 1, 0, 1, 0, 1, 0, 0);
        const __m512i iY = VSET(0, 1, 0, 1, 10, 11, 0, 0);   // (t1, t1, t3, .)
        for (int i = 0; i < NUM_LIMBS; i++) {
            A4[i] = VPERM512MASKZ(0x3f, P4[i], iX);
            B4[i] = VBLENDPERMMASKZ(0x3f, Q4[i], P4[i], iY);
        }
        fp_mixaddsub_8x1(G, A4, B4, 0x03);
 
        // sign_flip = (v0*v1 != v0*v2)
        sign_flip = !((fp2_is_zero_4x2x1(G) >> 2) & 0x1);
 
        // C0 = (D0, e0, e1, D2, D2, e1, e0, D0), C1 = D1 broadcast
        {
            const __m512i iGDr = VSET(8, 0, 2, 12, 12, 2, 0, 8);
            const __m512i iGDi = VSET(9, 1, 3, 13, 13, 3, 1, 9);
            const __m512i iC1 = VSET(2, 3, 2, 3, 2, 3, 2, 3);  // D1 = D4 elt 1
            for (int i = 0; i < NUM_LIMBS; i++) {
                K->C0[i]             = VBLENDPERM(iGDr, G[i], D4[i]);
                K->C0[i + NUM_LIMBS] = VBLENDPERM(iGDi, G[i], D4[i]);
                K->C1[i]             = VPERM512(iC1, D4[i]);
            }
        }
    } else {
        printf("ValueError(unsupported case for SQIsign verification\n");
        K->sign_flip = sign_flip;
        return false;
    }
 
    K->sign_flip = sign_flip;
    return true;
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
// ============================================================================
// superglue_codomain: updated head. The scalar locals and the
// superglue_precompute_consts call are gone; everything else is unchanged.
// ============================================================================
bool superglue_codomain(__m512i* res_OA, __m512i* A24_12,
        superglue_precomp_t *K, __m512i* x_8, __m512i* z_8) {
    __m512i x_8_[2*NUM_LIMBS], z_8_[2*NUM_LIMBS], x_4[NUM_LIMBS], z_4[NUM_LIMBS];
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS];
    __m512i p1[NUM_LIMBS], p2[NUM_LIMBS];
    bool success;
 
    // X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)
    ec_jac_to_xz_4x4x2x1(x_8_, z_8_, x_8, z_8);
 
    // Double points to recover 4-torsion for change of basis
    ec_double_4x4x2x1(x_4, z_4, A24_12, x_8_, z_8_);
 
    // Fills K->position, K->sign_flip, K->C0, K->C1 -- all in vector form,
    // no scalar staging anywhere on this path.
    success = compute_superglue_coefficients(K, x_4, z_4);
 
    // (u, v, w) = (x_8, 0, z_8): V vanishes on the codomain path.
    // 16M + 4S + 16a
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

    return success;
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
