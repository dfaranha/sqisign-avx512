#include "theta_middle.h"

#include "fp2_lvl3.h"

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
    __m512i t[8];
    __m512i pqpq = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    __m512i rsrs = VSET(4, 5, 6, 7, 4, 5, 6, 7);

    // Split
    for (int i = 0; i < 8; i++) {
        t[i] = VPERM256(a[i], 0x44); //aacc
        res[i] = VPERM256(a[i], 0xee); //bbdd
    }

    // add sub add sub
    fp2_mixaddsub_4x2x1(res, t, res, 0x33); // p, q, r, s = a + b, a - b, c + d, c - d
    
    // Split 
    for (int i = 0; i < 8; i++) {
        t[i] = VPERM512(pqpq, res[i]);
        res[i] = VPERM512(rsrs, res[i]);
    }

    // add add sub sub
    fp2_mixaddsub_4x2x1(res, t, res, 0xf); // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
}

void theta_precomp(__m512i* c1, __m512i* c2, __m512i* null)
{
    __m512i t1[8], t2[8], idx1, idx2;

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
    for (int i = 0; i < 8; i++) {
        c1[i] = VPERM256(t1[i], 0x4e);                      // (y', x', w', z')
        c2[i] = VPERM256(null[i], 0x4e);                    // (y, x, w, z)
        t2[i] = VBLENDPERM32(0xffff, null[i], t1[i], idx2);   // (y', w', y, w)
        t1[i] = VBLENDPERM32(0xffff, null[i], t1[i], idx1);   // (x', z', x, z)
    }

    fp2_mul_4x2x1(t2, t1, t2);

    idx1 = VSET(2, 3, 2, 3, 0, 1, 0, 1);
    idx2 = VSET(6, 7, 6, 7, 4, 5, 4, 5);
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512(idx1, t2[i]);
        t2[i] = VPERM512(idx2, t2[i]);
    }

    fp2_mul_4x2x1(c1, c1, t1);
    fp2_mul_4x2x1(c2, c2, t2);
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
    __m512i t1[16], t2[16], P[8], Q[8], idx, idx2;

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
    for (int i = 0; i < 8; i++) {
        t2[i] = VPERM256(Q[i], 0x4e);
        t1[i] = VBLENDPERM32(0xffff, P[i], Q[i], idx);
    }

    // t1 = (x1 * y2, y1 * x2, z2 * w2, w2 * z2) = (s1, s2, s3, s3)
    // TODO: Maybe better to put s3 first?
    fp2_mul_4x2x1(t1, t1, t2);

    // t2 = (s1, s2, s1, s2)
    idx = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < 8; i++) {
        t2[i] = VPERM512(idx, t1[i]);
    }

    // res
    fp2_mul_4x2x1(res, Q, t2);

    // res_inv
    fp2_4x2x1_to_2x4x1(t2, t1); // t2[8:15] = (s3, s3)
    idx = VSET(2, 2, 3, 3, 0, 0, 1, 1);
    idx2 = VSET(6, 6, 7, 7, 4, 4, 5, 5);
    for (int i = 0; i < 8; i++) { // t1[0:7] = (y1, x1), t1[8:15] = (w_res, z_res)
        t1[i] = VPERM512MASKZ(0x55, P[i], idx);
        t1[i+8] = VPERM512MASKZ(0x55, res[i], idx2);
    }
    fp2_mul_2x4x1(t1, t1, t2+8);
    fp2_2x4x1_to_4x2x1(res_inv, t1, t1+8);

    if (h2) {
        theta_had_single(res, res);
    }
}

void theta_codomain_with_four_torsion(__m512i* res_inv, __m512i* null, __m512i* ker, const bool h1, const bool h2)
{
    __m512i t0[8], t1[8], t2[8];
    fp2_t unvec0[4], unvec1[4], unvec2[4], unvec3[4], t_unvec[4], AB, AC;

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
    //xAB, _, xCD, _ = T0
    //AA, BB, CC, DD = T1
    fp2_4x2x1_to_unvectorized(unvec0, t0);
    fp2_4x2x1_to_unvectorized(unvec1, t1);

    // T0 = AVX_4x2x1(AA, AA, xAB, BB)
    t_unvec[0] = unvec1[0]; t_unvec[1] = unvec1[0]; t_unvec[2] = unvec0[0]; t_unvec[3] = unvec1[1];
    fp2_unvectorized_to_4x2x1(t0, t_unvec);
    // T1 = AVX_4x2x1(BB, CC, CC, DD)
    t_unvec[0] = unvec1[1]; t_unvec[1] = unvec1[2]; t_unvec[2] = unvec1[2]; t_unvec[3] = unvec1[3];
    fp2_unvectorized_to_4x2x1(t1, t_unvec);
    //AABB, AACC, xABCC, BBDD = pairwise_multiplication(T0, T1)
    fp2_mul_4x2x1(t0, t0, t1);
    fp2_4x2x1_to_unvectorized(unvec2, t0);

    // TODO: here are two fp2 sqrt, do we want to do something
    //       in parallel here, or convert from 64-bit for the last
    //       two steps where we need sqrts?
    fp2_copy(&AB, unvec2);
    fp2_copy(&AC, unvec2+1);
    fp2_sqrt(&AB);
    fp2_sqrt(&AC);

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

    // T0 = AVX_4x2x1(xAB, AB, xCD, BB)
    t_unvec[0] = unvec0[0]; t_unvec[1] = AB; t_unvec[2] = unvec0[2]; t_unvec[3] = unvec1[1];
    fp2_unvectorized_to_4x2x1(t0, t_unvec);
    // T1 = AVX_4x2x1(AC, DD, AB, AC)
    t_unvec[0] = AC; t_unvec[1] = unvec1[3]; t_unvec[2] = AB; t_unvec[3] = AC;
    fp2_unvectorized_to_4x2x1(t1, t_unvec);
    //xABAC, ABDD, xCDAB, BBAC = pairwise_multiplication(T0, T1)
    fp2_mul_4x2x1(t0, t0, t1);
    fp2_4x2x1_to_unvectorized(unvec3, t0);

    // OB = AVX_4x2x1(AA, AB, AA, AA)
    t_unvec[0] = unvec1[0]; t_unvec[1] = AB; t_unvec[2] = unvec1[0]; t_unvec[3] = unvec1[0];
    fp2_unvectorized_to_4x2x1(null, t_unvec);
    //T0 = AVX_4x2x1(xABAC, xABAC, xABCC, xCDAB)
    t_unvec[0] = unvec3[0]; t_unvec[1] = unvec3[0]; t_unvec[2] = unvec2[2]; t_unvec[3] = unvec3[2];
    fp2_unvectorized_to_4x2x1(t0, t_unvec);
    fp2_mul_4x2x1(null, null, t0);

    // inv = AVX_4x2x1(xABCC, xABCC, xABAC, xCDAB)
    t_unvec[0] = unvec2[2]; t_unvec[1] = unvec2[2]; t_unvec[2] = unvec3[0]; t_unvec[3] = unvec3[2];
    fp2_unvectorized_to_4x2x1(res_inv, t_unvec);
    // T0 = AVX_4x2x1(BBDD, ABDD, BBDD, BBAC)
    t_unvec[0] = unvec2[3]; t_unvec[1] = unvec3[1]; t_unvec[2] = unvec2[3]; t_unvec[3] = unvec3[3];
    fp2_unvectorized_to_4x2x1(t0, t_unvec);
    fp2_mul_4x2x1(res_inv, res_inv, t0);

    if (h2) {
        theta_had_single(null, null);
    }
}

void theta_codomain_with_no_torsion(__m512i* res_inv, __m512i* null, const bool h1, const bool h2)
{
    __m512i t0[8], t1[8];
    fp2_t unvec[4], t_unvec[4], AB, AC, AD;
    
    if (h1) {
        theta_had_single(null, null);
    }
    fp2_sqr_4x2x1(t0, null);
    theta_had_single(t0, t0);

    // TODO: here are three fp2 sqrt, do we want to do something
    //       in parallel here, or convert from 64-bit for the last
    //       two steps where we need sqrts?
    // AA, BB, CC, DD = T0
    fp2_4x2x1_to_unvectorized(unvec, t0);
    fp2_mul(&AB, unvec, unvec + 1);
    fp2_mul(&AC, unvec, unvec + 2);
    fp2_mul(&AD, unvec, unvec + 3);
    fp2_sqrt(&AB);
    fp2_sqrt(&AC);
    fp2_sqrt(&AD);
    // OB = AVX_4x2x1(AA, AB, AC, AD)
    t_unvec[0] = unvec[0]; t_unvec[1] = AB; t_unvec[2] = AC; t_unvec[3] = AD;
    fp2_unvectorized_to_4x2x1(null, t_unvec);

    // This calls pairwise_multiplication twice for a total of
    // 8M where as in serial we only need 7, so 1 M is technically
    // redundant here.
    // A_inv = BB * CC * DD
    // B_inv = AB * CC * DD
    // C_inv = BB * AC * DD
    // D_inv = BB * CC * AD
    
    // inv = AVX_4x2x1(BB, AB, BB, BB)
    t_unvec[0] = unvec[1]; t_unvec[1] = AB; t_unvec[2] = unvec[1]; t_unvec[3] = unvec[1];
    fp2_unvectorized_to_4x2x1(res_inv, t_unvec);
    // T0 = AVX_4x2x1(CC, CC, AC, CC)
    t_unvec[0] = unvec[2]; t_unvec[1] = unvec[2]; t_unvec[2] = AC; t_unvec[3] = unvec[2];
    fp2_unvectorized_to_4x2x1(t0, t_unvec);
    fp2_mul_4x2x1(res_inv, res_inv, t0);
    
    // T0 = AVX_4x2x1(DD, DD, DD, AD)
    t_unvec[0] = unvec[3]; t_unvec[1] = unvec[3]; t_unvec[2] = unvec[3]; t_unvec[3] = AD;
    fp2_unvectorized_to_4x2x1(t0, t_unvec);
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
        theta_eval_single(pts + 8*i, pts + 8*i, inv, h1, h2);
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
    __m512i t[16];
    fp2_4x2x1_to_8x1x1(t, c, c);
    fp2_mul_8x1x1(res, a, t);
}

void theta_had_pair(__m512i* res, __m512i* a) 
{
    __m512i t[16];
    __m512i pqpq = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    __m512i rsrs = VSET(4, 5, 6, 7, 4, 5, 6, 7);

    // Split
    for (int i = 0; i < 16; i++) {
        t[i] = VPERM256(a[i], 0x44); //aacc
        res[i] = VPERM256(a[i], 0xee); //bbdd
    }

    // add sub add sub
    // p, q, r, s = a + b, a - b, c + d, c - d
    fp2_mixaddsub_8x1x1(res, t, res, 0x33, 0x33); 

    // Split 
    for (int i = 0; i < 16; i++) {
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
    __m512i t[16];

    // Split
    for (int i = 0; i < 16; i++) {
        t[i] = VSHUFFLE(a[i], 0x44); //aacc
        res[i] = VSHUFFLE(a[i], 0xee); //bbdd
    }

    // add sub add sub
    // p, q, r, s = a + b, a - b, c + d, c - d
    fp2_mixaddsub_8x1x1(res, t, res, 0x55, 0x55);

    // Split 
    for (int i = 0; i < 16; i++) {
        t[i] = VPERM256(res[i], 0x44);
        res[i] = VPERM256(res[i], 0xee);
    }

    // add add sub sub
    // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
    fp2_mixaddsub_8x1x1(res, t, res, 0x33, 0x33);
}

void theta_double_pair(__m512i* res, __m512i* a, __m512i* c1, __m512i* c2)
{
    theta_sqr_pair(res, a);
    theta_had_pair(res, res);
    theta_sqr_pair(res, res);
    theta_mul_pair(res, res, c1);
    theta_had_pair(res, res);
    theta_mul_pair(res, res, c2);
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

void theta_codomain_pair(__m512i* res, __m512i* res_inv, __m512i* k, const bool h1, const bool h2)
{
    __m512i PQ[16], k_split[16], t1[16], t2[16], idx, idx2;

    fp2_copy_8x1x1(PQ, k);

    if (h1) {
        theta_had_pair(PQ, PQ);
    }

    theta_sqr_pair(PQ, PQ);
    theta_had_pair(PQ, PQ);

    // Input: k = (k1, k2) = (xi, yi, zi, wi)_{i=1,2}
    // TODO: Is this necessary?
    fp2_8x1x1_to_4x2x1(k_split, k_split+8, PQ);

    // t1 = (x1, y1, z2, w2)
    // t2 = (y2, x2, w2, z2)
    idx = VSET(
        0x0000000100000000, 0x0000000300000002, 0x0000000500000004, 0x0000000700000006,
        0x0000001900000018, 0x0000001b0000001a, 0x0000001d0000001c, 0x0000001f0000001e
    );
    for (int i = 0; i < 8; i++) {
        t2[i] = VPERM256(k_split[i+8], 0x4e);
        t1[i] = VBLENDPERM32(0xffff, k_split[i], k_split[i+8], idx);
    }

    // t1 = (x1 * y2, y1 * x2, z2 * w2, w2 * z2) = (s1, s2, s3, s3)
    // TODO: Maybe better to put s3 first?
    fp2_mul_4x2x1(t1, t1, t2);

    // t2 = (s1, s2, s1, s2)
    idx = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < 8; i++) {
        t2[i] = VPERM512(idx, t1[i]);
    }

    // res
    fp2_mul_4x2x1(res, k_split+8, t2);

    // res_inv
    fp2_4x2x1_to_2x4x1(t2, t1); // t2[8:15] = (s3, s3)
    idx = VSET(2, 2, 3, 3, 0, 0, 1, 1);
    idx2 = VSET(6, 6, 7, 7, 4, 4, 5, 5);
    for (int i = 0; i < 8; i++) { // t1[0:7] = (y1, x1), t1[8:15] = (w_res, z_res)
        t1[i] = VPERM512MASKZ(0x55, k_split[i], idx);
        t1[i+8] = VPERM512MASKZ(0x55, res[i], idx2);
    }
    fp2_mul_2x4x1(t1, t1, t2+8);
    fp2_2x4x1_to_4x2x1(res_inv, t1, t1+8);

    if (h2) {
        theta_had_single(res, res);
    }
}

void theta_eval_pair(__m512i* res, __m512i* a, __m512i* inv, const bool h1, const bool h2)
{
    if (h1) {
        theta_had_pair(res, a);
    }
    theta_sqr_pair(res, res);
    theta_had_pair(res, res);
    theta_mul_pair(res, res, inv);
    if (h2) {
        theta_had_pair(res, res);
    }
}

void theta_eval_list_pair(__m512i* pts, __m512i* inv, const bool h1, const bool h2, const int num_pts)
{
    bool num_pts_is_odd = (num_pts & 0x1);
    int num_pair_pts = num_pts / 2;

    for (int i = 0; i < num_pair_pts; i++) {
        theta_eval_pair(pts + 16*i, pts + 16*i, inv, h1, h2);
    }
    if (num_pts_is_odd) {
        theta_eval_single(pts+8*(num_pts-1), pts+8*(num_pts-1), inv, h1, h2);
    }
}

