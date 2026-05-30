#include "theta_middle.h"
#include "fp_unvec.h"
#include "fp_consts.h"
#include "fp2.h"

// ======================================================================================================
//                                               Single
// ======================================================================================================

void theta_sqr_single(__m512i* res, __m512i* a) 
{
    fp2_sqr_4x2x1(res, a);
}

void theta_mul_single(__m512i* res, __m512i* a, __m512i* b) 
{
    fp2_mul_4x2x1(res, a, b);
}

// TODO: Could we merge splitting with mixaddsub and use less tmp registers?
void theta_had_single(__m512i* res, __m512i* a) 
{
    __m512i t[NUM_LIMBS];
    __m512i pqpq = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    __m512i rsrs = VSET(4, 5, 6, 7, 4, 5, 6, 7);

    // Split
    for (int i = 0; i < NUM_LIMBS; i++) {
        t[i] = VPERM256(a[i], 0x44); //aacc
        res[i] = VPERM256(a[i], 0xee); //bbdd
    }

    // add sub add sub
    fp2_mixaddsub_4x2x1(res, t, res, 0x33); // p, q, r, s = a + b, a - b, c + d, c - d
    
    // Split 
    for (int i = 0; i < NUM_LIMBS; i++) {
        t[i] = VPERM512(pqpq, res[i]);
        res[i] = VPERM512(rsrs, res[i]);
    }

    // add add sub sub
    fp2_mixaddsub_4x2x1(res, t, res, 0xf); // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
}

void theta_precomp_single(__m512i* c1, __m512i* c2, __m512i* null)
{
    __m512i t1[NUM_LIMBS], t2[NUM_LIMBS], idx1, idx2;

    theta_sqr_single(t1, null);
    theta_had_single(t1, t1); // t1 = (x', y', z', w')

    idx1 = VSET(
        0x0000001100000010, 0x0000001300000012, 0x0000001900000018, 0x0000001b0000001a,
        0x0000000100000000, 0x0000000300000002, 0x0000000900000008, 0x0000000b0000000a
    );
    idx2 = VSET(
        0x0000001500000014, 0x0000001700000016, 0x0000001d0000001c, 0x0000001f0000001e,
        0x0000000500000004, 0x0000000700000006, 0x0000000d0000000c, 0x0000000f0000000e
    );
    for (int i = 0; i < NUM_LIMBS; i++) {
        c1[i] = VPERM256(t1[i], 0x4e);                      // (y', x', w', z')
        c2[i] = VPERM256(null[i], 0x4e);                    // (y, x, w, z)
        t2[i] = VBLENDPERM32(0xffff, null[i], t1[i], idx2);   // (y', w', y, w)
        t1[i] = VBLENDPERM32(0xffff, null[i], t1[i], idx1);   // (x', z', x, z)
    }

    fp2_mul_4x2x1(t2, t1, t2);

    idx1 = VSET(2, 3, 2, 3, 0, 1, 0, 1);
    idx2 = VSET(6, 7, 6, 7, 4, 5, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx1, t2[i]);
        t2[i] = VPERM512(idx2, t2[i]);
    }

    fp2_mulx2_4x2x1(c1, c2, c1, t1, c2, t2);
}

void theta_double_single(__m512i* res, __m512i* a, __m512i* c1, __m512i* c2)
{
    theta_sqr_single(res, a);
    theta_had_single(res, res);
    theta_sqr_single(res, res);
    theta_mul_single(res, res, c1);
    theta_had_single(res, res);
    theta_mul_single(res, res, c2);
}

void theta_double_iter_single(__m512i* res, __m512i* a, __m512i* c1, __m512i* c2, const int n)
{
    if (n == 0) {
        fp2_copy_4x2x1(res, a);
        return;
    }

    theta_double_single(res, a, c1, c2);
    for (int i = 0; i < n-1; i++) {
        theta_double_single(res, res, c1, c2);
    }
}


// Idea 2: Investigate the possibility of 1x8x1 Fp2 mul? Or convert to non-vectorized rep and do a single Fp mul?
// Warning: Modifies k1 and k2
void theta_codomain_single(__m512i* res, __m512i* res_inv, __m512i* k1, __m512i* k2, const bool h1, const bool h2) 
{
    __m512i t1[2*NUM_LIMBS], t2[2*NUM_LIMBS], P[NUM_LIMBS], Q[NUM_LIMBS], idx, idx2;

    fp2_copy_4x2x1(P, k1);
    fp2_copy_4x2x1(Q, k2);

    //TODO: Would it be more efficient to use the pair operations here?
    if (h1) {
        theta_had_single(P, P);
        theta_had_single(Q, Q);
    }

    theta_sqr_single(P, P);
    theta_sqr_single(Q, Q);
    theta_had_single(P, P);
    theta_had_single(Q, Q);

    // Input: ki = (xi, yi, zi, wi) in the 4x2x1 format
    // t1 = (x1, y1, z2, w2)
    // t2 = (y2, x2, w2, z2)
    idx = VSET(
        0x0000000100000000, 0x0000000300000002, 0x0000000500000004, 0x0000000700000006,
        0x0000001900000018, 0x0000001b0000001a, 0x0000001d0000001c, 0x0000001f0000001e
    );
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VPERM256(Q[i], 0x4e);
        t1[i] = VBLENDPERM32(0xffff, P[i], Q[i], idx);
    }

    // t1 = (x1 * y2, y1 * x2, z2 * w2, w2 * z2) = (s1, s2, s3, s3)
    // TODO: Maybe better to put s3 first?
    fp2_mul_4x2x1(t1, t1, t2);

    // t2 = (s1, s2, s1, s2)
    idx = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VPERM512(idx, t1[i]);
    }

    // res
    fp2_mul_4x2x1(res, Q, t2);

    // res_inv
    fp2_4x2x1_to_2x4x1(t2, t2 + NUM_LIMBS, t1); // t2[8:15] = (s3, s3)
    idx = VSET(2, 2, 3, 3, 0, 0, 1, 1);
    idx2 = VSET(6, 6, 7, 7, 4, 4, 5, 5);
    for (int i = 0; i < NUM_LIMBS; i++) { // t1[0:7] = (y1, x1), t1[8:15] = (w_res, z_res)
        t1[i] = VPERM512MASKZ(0x55, P[i], idx);
        t1[i + NUM_LIMBS] = VPERM512MASKZ(0x55, res[i], idx2);
    }
    fp2_mul_2x4x1(t1, t1, t2 + NUM_LIMBS);
    fp2_2x4x1_to_4x2x1(res_inv, t1, t1 + NUM_LIMBS);

    if (h2) {
        theta_had_single(res, res);
    }
}

void theta_codomain_with_four_torsion(__m512i* res_inv, __m512i* null, __m512i* ker, const bool h1, const bool h2)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS], t3[2*N4x2];
    __m512i idx0, idx1, idx2, idx3;

    if (h1) {
        theta_had_single(t0, ker);
        theta_sqr_single(t0, t0);
        theta_had_single(t0, t0);

        theta_had_single(t1, null);
        theta_sqr_single(t1, t1);
        theta_had_single(t1, t1);
    } else {
        theta_sqr_single(t0, ker);
        theta_had_single(t0, t0);

        theta_sqr_single(t1, null);
        theta_had_single(t1, t1);
    }
    // TODO: two wasted squarings? Or needed for had?
    // t0 = (xAB, _, xCD, _)
    // t1 = (AA, BB, CC, DD)

    // After this no longer need: CC
    // Still need: xAB, xCD, AA, BB, DD

    // t2 = (AA, AA, xAB, BB)
    // t3 = (BB, CC, CC, DD)
    idx0 = VSET(8, 9, 8, 9, 0, 1, 10, 11);
    idx1 = VSET(2, 3, 4, 5, 4, 5, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t3[i] = VPERM512(idx1, t1[i]);
    }
    // t2 = (AABB, AACC, xABCC, BBDD)
    fp2_mul_4x2x1(t2, t2, t3);

    // After this no longer need: AABB, AACC
    // Still need: xAB, xCD, AA, BB, DD, xABCC, BBDD

    // t2 = (AB, AC, xABCC, BBDD)
    fp2_4x2x1_to_2x2x2(t3, t3 + N4x2, t2);
    fp2_sqrt_2x2x2(t3, t3);
    fp2_2x2x2_to_4x2x1(t2, t3, t3 + N4x2);

    // x A^3 BC as a projective factor
    // A = xAB * AA * AC
    // B = AB * AC * xAB
    // C = xAB * AA * CC
    // D = xCD * AB * AA

    // x A^2 B^3 C^2 D^2 as a projective factor
    // A_inv = xAB * BB * CC * DD
    // B_inv = xAB * AB * CC * DD
    // C_inv = xAB * AC * BB * DD
    // D_inv = AB * BB * AC * xCD

    // No longer need: CC
    // After this no longer need: xAB, xCD, BB, DD, AC
    // Still need: AA, AB, xABCC, BBDD

    // t0 = (xAB, AB, xCD, BB)
    // t3 = (AC, DD, AB, AC)
    idx0 = VSET(0, 1, 8, 9, 4, 5, 0, 0);
    idx1 = VSET(0, 1, 2, 3, 4, 5, 10, 11);
    idx2 = VSET(10, 11, 6, 7, 8, 9, 10, 11);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], t2[i]);
        t0[i] = VBLENDPERM(idx1, t0[i], t1[i]);
        t3[i] = VBLENDPERM(idx2, t1[i], t2[i]);
    }
    // t0 = (xABAC, ABDD, xCDAB, BBAC)
    fp2_mul_4x2x1(t0, t0, t3);

    // No longer need: xAB, xCD, BB, DD, CC, AC
    // After this no longer need: AA, AB
    // Still need: xABCC, BBDD in t2 and xABAC, ABDD, xCDAB, BBAC in t0

    // null = (AA, AB, AA, AA)
    // t1 = (xABAC, xABAC, xABCC, xCDAB)
    idx0 = VSET(0, 1, 8, 9, 0, 1, 0, 1);
    idx1 = VSET(0, 1, 0, 1, 12, 13, 4, 5);
    // res_inv = (xABCC, xABCC, xABAC, xCDAB)
    // t3 = (BBDD, ABDD, BBDD, BBAC)
    idx2 = VSET(12, 13, 12, 13, 0, 1, 4, 5);
    idx3 = VSET(14, 15, 2, 3, 14, 15, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        null[i] = VBLENDPERM(idx0, t1[i], t2[i]);
        t1[i] = VBLENDPERM(idx1, t0[i], t2[i]);
        res_inv[i] = VBLENDPERM(idx2, t0[i], t2[i]);
        t3[i] = VBLENDPERM(idx3, t0[i], t2[i]);
    }
    fp2_mulx2_4x2x1(null, res_inv, null, t1, res_inv, t3);

    if (h2) {
        theta_had_single(null, null);
    }
}

void theta_codomain_with_no_torsion(__m512i* res_inv, __m512i* null, const bool h1, const bool h2)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS];
    __m512i idx0, idx1, idx2, idx3;

    // t0 = (AA, BB, CC, DD)
    if (h1) {
        theta_had_single(null, null);
    }
    fp2_sqr_4x2x1(t0, null);
    theta_had_single(t0, t0);

    // This calls pairwise_multiplication twice for a total of
    // 8M where as in serial we only need 7, so 1 M is technically
    // redundant here.
    // A_inv = BB * CC * DD
    // B_inv = AB * CC * DD
    // C_inv = BB * AC * DD
    // D_inv = BB * CC * AD

    // null = (AA, AA, AA, AA)
    // t1 = (AA, BB, CC, DD)
    idx0 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    idx1 = VSET(0, 1, 2, 3, 4, 5, 6, 7);
    // res_inv = (BB, CC, BB, BB)
    // t2 =      (CC, DD, DD, CC)
    idx2 = VSET(2, 3, 4, 5, 2, 3, 2, 3);
    idx3 = VSET(4, 5, 6, 7, 6, 7, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        null[i] = VPERM512(idx0, t0[i]);
        t1[i] = VPERM512(idx1, t0[i]);
        res_inv[i] = VPERM512(idx2, t0[i]);
        t2[i] = VPERM512(idx3, t0[i]);
    }
    fp2_mulx2_4x2x1(null, res_inv, null, t1, res_inv, t2);

    // null = (AA, AB, AC, AD)
    // TODO: Wasting one sqrt, but using it to set up null without last permute to insert AA
    fp2_sqrt_4x2x1(null, null);

    // t0 = (DD, AB, AC, AD)
    idx0 = VSET(6, 7, 10, 11, 12, 13, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], null[i]);
    }
    fp2_mul_4x2x1(res_inv, res_inv, t0);
   
    if (h2) {
        theta_had_single(null, null);
    }
}

void theta_eval_single(__m512i* res, __m512i* a, __m512i* inv, const bool h1, const bool h2)
{
    if (h1) {
        theta_had_single(res, a);
    }
    theta_sqr_single(res, res);
    theta_had_single(res, res);
    theta_mul_single(res, res, inv);
    if (h2) {
        theta_had_single(res, res);
    }
}

void theta_eval_list_single(__m512i* pts, __m512i* inv, const bool h1, const bool h2, const int num_pts)
{
    for (int i = 0; i < num_pts; i++) {
        theta_eval_single(pts + i*NUM_LIMBS, pts + i*NUM_LIMBS, inv, h1, h2);
    }
}

// ======================================================================================================
//                                                 Pair
// ======================================================================================================

void theta_sqr_pair(__m512i* res, __m512i* a) 
{
    fp2_sqr_8x1x1(res, a);
}

void theta_mul_pair(__m512i* res, __m512i* a, __m512i* c) 
{
    __m512i t[2*NUM_LIMBS];
    fp2_4x2x1_to_8x1x1(t, c, c);
    fp2_mul_8x1x1(res, a, t);
}

void theta_had_pair(__m512i* res, __m512i* a) 
{
    __m512i t[2*NUM_LIMBS];
    __m512i pqpq = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    __m512i rsrs = VSET(4, 5, 6, 7, 4, 5, 6, 7);

    // Split
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t[i] = VPERM256(a[i], 0x44); //aacc
        res[i] = VPERM256(a[i], 0xee); //bbdd
    }

    // add sub add sub
    // p, q, r, s = a + b, a - b, c + d, c - d
    fp2_mixaddsub_8x1x1(res, t, res, 0x33, 0x33); 

    // Split 
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t[i] = VPERM512(pqpq, res[i]);
        res[i] = VPERM512(rsrs, res[i]);
    }

    // add add sub sub
    // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
    fp2_mixaddsub_8x1x1(res, t, res, 0xf, 0xf);
}

// Like 1 cycle faster, but assumes we store the pairs as (a1, b1, c1, d1, a2, b2, c2, d2)
void theta_had_pair_alt(__m512i* res, __m512i* a) 
{
    __m512i t[2*NUM_LIMBS];

    // Split
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t[i] = VSHUFFLE(a[i], 0x44); //aacc
        res[i] = VSHUFFLE(a[i], 0xee); //bbdd
    }

    // add sub add sub
    // p, q, r, s = a + b, a - b, c + d, c - d
    fp2_mixaddsub_8x1x1(res, t, res, 0x55, 0x55);

    // Split 
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t[i] = VPERM256(res[i], 0x44);
        res[i] = VPERM256(res[i], 0xee);
    }

    // add add sub sub
    // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
    fp2_mixaddsub_8x1x1(res, t, res, 0x33, 0x33);
}

void theta_precomp_pair(__m512i* c1, __m512i* c2, __m512i* null)
{
    __m512i t1_t2[2*NUM_LIMBS], idx1, idx2, idx_re, idx_im;

    theta_sqr_single(t1_t2, null);
    theta_had_single(t1_t2, t1_t2); // t1 = (x', y', z', w')

    idx1 = VSET(
        0x0000001100000010, 0x0000001300000012, 0x0000001900000018, 0x0000001b0000001a,
        0x0000000100000000, 0x0000000300000002, 0x0000000900000008, 0x0000000b0000000a
    );
    idx2 = VSET(
        0x0000001500000014, 0x0000001700000016, 0x0000001d0000001c, 0x0000001f0000001e,
        0x0000000500000004, 0x0000000700000006, 0x0000000d0000000c, 0x0000000f0000000e
    );
    idx_re = VSET(2, 10, 0, 8, 6, 14, 4, 12);
    idx_im = VSET(3, 11, 1, 9, 7, 15, 5, 13);
    for (int i = 0; i < NUM_LIMBS; i++) {
        // Interleave (y', x', w', z') and (y, x, w, z)
        c1[i] = VBLENDPERM(idx_re, t1_t2[i], null[i]);
        c1[i + NUM_LIMBS] = VBLENDPERM(idx_im, t1_t2[i], null[i]);
        t1_t2[i + NUM_LIMBS] = VBLENDPERM32(0xffff, null[i], t1_t2[i], idx2);   // (y', w', y, w)
        t1_t2[i] = VBLENDPERM32(0xffff, null[i], t1_t2[i], idx1);   // (x', z', x, z)
    }

    fp2_mul_4x2x1(t1_t2 + NUM_LIMBS, t1_t2, t1_t2 + NUM_LIMBS); // t2 = (a, b, c, d)

    // Interleave (b, b, a, a) and (d, d, c, c)
    idx_re = VSET(2, 6, 2, 6, 0, 4, 0, 4);
    idx_im = VSET(3, 7, 3, 7, 1, 5, 1, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1_t2[i] = VPERM512(idx_re, t1_t2[i + NUM_LIMBS]);
        t1_t2[i + NUM_LIMBS] = VPERM512(idx_im, t1_t2[i + NUM_LIMBS]);
    }

    // Result is c1 and c2 interleaved
    fp2_mul_8x1x1(t1_t2, t1_t2, c1);

    // Output (c1, c1) and (c2, c2)
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        c1[i] = VUNPACKLO(t1_t2[i], t1_t2[i]);
        c2[i] = VUNPACKHI(t1_t2[i], t1_t2[i]);
    }
	//fp2_4x2x1_to_8x1x1(c1, c1, c1);
	//fp2_4x2x1_to_8x1x1(c2, c2, c2);
#ifdef SUPERGLUE
	fp2_t t[16];
	fp2_8x1x1_to_unvectorized(t, c1);
	fp2_cswap(&t[1], &t[2], -1);
	fp2_cswap(&t[2], &t[4], -1);
	fp2_cswap(&t[3], &t[6], -1);
	fp2_cswap(&t[5], &t[6], -1);
	fp2_unvectorized_to_8x1x1(c1, t);
	fp2_8x1x1_to_unvectorized(t, c2);
	fp2_cswap(&t[1], &t[2], -1);
	fp2_cswap(&t[2], &t[4], -1);
	fp2_cswap(&t[3], &t[6], -1);
	fp2_cswap(&t[5], &t[6], -1);
	fp2_unvectorized_to_8x1x1(c2, t);
#endif
}

void theta_double_pair(__m512i* res, __m512i* a, __m512i* c1, __m512i* c2)
{
    theta_sqr_pair(res, a);
#ifdef SUPERGLUE
	theta_had_pair_alt(res, res);
#else
	theta_had_pair(res, res);
#endif
	theta_sqr_pair(res, res);
	fp2_mul_8x1x1(res, res, c1);
#ifdef SUPERGLUE
	theta_had_pair_alt(res, res);
#else
	theta_had_pair(res, res);
#endif
	fp2_mul_8x1x1(res, res, c2);
}

void theta_double_iter_pair(__m512i* res, __m512i* a, __m512i* c1, __m512i* c2, const int n)
{
    if (n == 0) { 
        fp2_copy_8x1x1(res, a);
        return; 
    }

    theta_double_pair(res, a, c1, c2);
    for (int i = 0; i < n-1; i++) {
        theta_double_pair(res, res, c1, c2);
    }
}

bool check_isotropic_pair(__m512i* PQ, __m512i* inv)
{
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS];

    // PQ = (P0, Q0, P1, Q1, P2, Q2, P3, Q3)
    // t0 = (inv0, inv0, inv1, inv1, inv2, inv2, inv3, inv3)
    fp2_mul_8x1x1(t0, PQ, inv);

#ifndef SUPERGLUE
    // assert t0[0] == t0[2] & t0[4] == t0[6] & 
    //        t0[1] == t0[5] & t0[3] == t0[7]
    __m512i idx0 = VSET(0, 8, 4, 12, 1, 9, 3, 11);
    __m512i idx1 = VSET(2, 10, 6, 14, 5, 13, 7, 15);
#else
    __m512i idx0 = VSET(0, 8, 2, 10, 4, 12, 5, 13);
    __m512i idx1 = VSET(1, 9, 3, 11, 6, 14, 7, 15);
#endif
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i], t0[i + NUM_LIMBS]);
        t1[i + NUM_LIMBS] = VBLENDPERM(idx1, t0[i], t0[i + NUM_LIMBS]);
    }
    
    return fp2_is_equal_4x2x1(t1, t1 + NUM_LIMBS) == 0xf;
}

bool theta_codomain_pair(__m512i* res, __m512i* res_inv, __m512i* k, const bool h1, const bool h2, const bool verify)
{
    __m512i PQ[2*NUM_LIMBS], k_split[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS], idx1, idx2, idx3;

    fp2_copy_8x1x1(PQ, k);

    if (h1) {
#ifndef SUPERGLUE
        theta_had_pair(PQ, PQ);
#else
        theta_had_pair_alt(PQ, PQ);
#endif
    }

    theta_sqr_pair(PQ, PQ);
#ifndef SUPERGLUE
    theta_had_pair(PQ, PQ);
#else
    theta_had_pair_alt(PQ, PQ);
#endif

    if (verify) {
        // xA, xB, yC, yD = ker_P
        // zA, wB, zC, wD = ker_Q
        // assert all but yC and yD are nonzero
        int nz_mask = ~fp2_is_zero_8x1x1(PQ);
        if (nz_mask & 0xaf != 0xaf) {
            return false;
        }
    }

#ifndef SUPERGLUE
    // t1 = (x1, x2, z2, z2)
    // t2 = (y2, y1, w2, w2)
    idx1 = VSET(0, 8, 1, 9, 5, 13, 5, 13);
    idx2 = VSET(3, 11, 2, 10, 7, 15, 7, 15);
#else
    // t1 = (x1, x2, z2, w2)
    // t2 = (y2, y1, w2, w2)
    idx1 = VSET(0, 8, 4, 12, 6, 14, 6, 14);
    idx2 = VSET(5, 13, 1, 9, 7, 15, 7, 15); 
#endif
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VBLENDPERM(idx1, PQ[i], PQ[i + NUM_LIMBS]);
        t2[i] = VBLENDPERM(idx2, PQ[i], PQ[i + NUM_LIMBS]);
    }

    // t1 = (x1 * y2, y1 * x2, z2 * w2, w2 * z2) = (s1, s2, s3, s3)
    fp2_mul_4x2x1(t1, t1, t2);

#ifndef SUPERGLUE
    // Computing res and res_inv simultenously with 8x1x1 mul (faster than 4x2x1 mul + 2x4x1 mul)
    // t2 = (s1, s2, s1, s2) and (s3, s3, s2, s1) interleaved
    idx2 = VSET(0, 4, 2, 4, 0, 2, 2, 0);
    idx3 = VSET(1, 5, 3, 5, 1, 3, 3, 1);
    // t1 = (x2, y2, z2, w2) and (y1, z1, w2, z2) interleaved
    idx1 = VSET(1, 2, 3, 0, 5, 7, 7, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VPERM512(idx2, t1[i]);
        t2[i + NUM_LIMBS] = VPERM512(idx3, t1[i]);
        t1[i] = VPERM512(idx1, PQ[i]);
        t1[i + NUM_LIMBS] = VPERM512(idx1, PQ[i + NUM_LIMBS]);
    }
#else
    idx2 = VSET(0, 2, 0, 2, 4, 4, 0, 0);
    idx3 = VSET(1, 3, 1, 3, 5, 5, 1, 1);
    idx1 = VSET(4, 5, 6, 7, 1, 0, 0, 0);
    for (int i = 0; i < 8; i++) {
        t2[i] = VPERM512(idx2, t1[i]);
        t2[i+NUM_LIMBS] = VPERM512(idx3, t1[i]);
        t1[i] = VPERM512(idx1, PQ[i]);
        t1[i+NUM_LIMBS] = VPERM512(idx1, PQ[i+NUM_LIMBS]);
    }
#endif
    fp2_mul_8x1x1(t1, t1, t2);

    // Preparing res in 4x2x1 and (res_inv, res_inv) in 8x1x1
#ifndef SUPERGLUE
    fp2_8x1x1_to_first_4x2x1(res, t1);
    idx1 = VSET(1, 1, 3, 3, 5, 5, 7, 7);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        res_inv[i] = VPERM512(idx1, t1[i]);
    }
#else
	idx1 = VSET(4, 5, 3, 2, 12, 13, 11, 10);
    for (int i = 0; i < 16; i++) {
		res_inv[i] = VPERM512(idx1, t1[i]);
    }
	idx1 = VSET(0, 8, 1, 9, 2, 10, 3, 11);
	for (int i = 0; i < 8; i++) {
		res[i] = VBLENDPERM(idx1, t1[i], t1[i+NUM_LIMBS]);
	}
#endif

    if (verify) {
        bool is_isotropic = check_isotropic_pair(PQ, res_inv);
        if (!is_isotropic) {
            return false;
        }
    }

    if (h2) {
        theta_had_single(res, res);
    }
#ifdef SUPERGLUE
	idx1 = VSET(4, 5, 3, 2, 4, 5, 3, 2);
	idx2 = VSET(12, 13, 11, 10, 12, 13, 11, 10);
	for (int i = 0; i < 8; i++) {
		res_inv[i] = VBLENDPERM(idx1, t1[i], t1[i+8]);
		res_inv[i+8] = VBLENDPERM(idx2, t1[i], t1[i+8]);
	}
#endif
	return true;
}

void theta_eval_pair(__m512i* res, __m512i* a, __m512i* inv, const bool h1, const bool h2)
{
    if (h1) {
#ifndef SUPERGLUE
        theta_had_pair(res, a);
#else
		theta_had_pair_alt(res, a);
#endif
    }
    theta_sqr_pair(res, res);
#ifndef SUPERGLUE
    theta_had_pair(res, res);
#else
	theta_had_pair_alt(res, res);
#endif
    fp2_mul_8x1x1(res, res, inv);
    if (h2) {
#ifndef SUPERGLUE
        theta_had_pair(res, res);
#else
		theta_had_pair_alt(res, res);
#endif
    }
}


void theta_eval_list_pair(__m512i* pts, __m512i* inv, const bool h1, const bool h2, const int num_pts)
{
    bool num_pts_is_odd = (num_pts & 0x1);
    int num_pair_pts = num_pts / 2;
    __m512i t[NUM_LIMBS];

    for (int i = 0; i < num_pair_pts; i++) {
        theta_eval_pair(pts + i*(2*NUM_LIMBS), pts + i*(2*NUM_LIMBS), inv, h1, h2);
    }
    if (num_pts_is_odd) {
        fp2_8x1x1_to_first_4x2x1(t, inv);
        theta_eval_single(pts + (num_pts-1)*NUM_LIMBS, pts + (num_pts-1)*NUM_LIMBS, t, h1, h2);
    }
}
