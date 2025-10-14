#include <stdio.h> //TODO: Remove

#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "theta.h"
#include "theta_splitting.h"

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
    fp_mixaddsub_8x1(res, t, res, 0x33); // p, q, r, s = a + b, a - b, c + d, c - d
    
    // Split 
    for (int i = 0; i < 8; i++) {
        t[i] = VPERM512(pqpq, res[i]);
        res[i] = VPERM512(rsrs, res[i]);
    }

    // add add sub sub
    fp_mixaddsub_8x1(res, t, res, 0xf); // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
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
        t2[i] = VBLENDPERM(0xffff, null[i], t1[i], idx2);   // (y', w', y, w)
        t1[i] = VBLENDPERM(0xffff, null[i], t1[i], idx1);   // (x', z', x, z)
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

// Idea 2: Investigate the possibility of 1x8x1 Fp2 mul? Or convert to non-vectorized rep and do a single Fp mul?
// Warning: Modifies k1 and k2
void theta_codomain_single(__m512i* res, __m512i* res_inv, __m512i* k1, __m512i* k2, bool h1, bool h2) 
{
    __m512i t1[16], t2[16], idx, idx2;

    //TODO: Would it be more efficient to use the pair operations here?
    if (h1) {
        theta_had_single(k1, k1);
        theta_had_single(k2, k2);
    }

    theta_sqr_single(k1, k1);
    theta_sqr_single(k2, k2);
    theta_had_single(k1, k1);
    theta_had_single(k2, k2);

    // Input: ki = (xi, yi, zi, wi) in the 4x2x1 format
    // t1 = (x1, y1, z2, w2)
    // t2 = (y2, x2, w2, z2)
    idx = VSET(
        0x0000000100000000, 0x0000000300000002, 0x0000000500000004, 0x0000000700000006,
        0x0000001900000018, 0x0000001b0000001a, 0x0000001d0000001c, 0x0000001f0000001e
    );
    for (int i = 0; i < 8; i++) {
        t2[i] = VPERM256(k2[i], 0x4e);
        t1[i] = VBLENDPERM(0xffff, k1[i], k2[i], idx);
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
    fp2_mul_4x2x1(res, k2, t2);

    // res_inv
    fp2_4x2x1_to_2x4x1(t2, t1); // t2[8:15] = (s3, s3)
    idx = VSET(2, 2, 3, 3, 0, 0, 1, 1);
    idx2 = VSET(6, 6, 7, 7, 4, 4, 5, 5);
    for (int i = 0; i < 8; i++) { // t1[0:7] = (y1, x1), t1[8:15] = (w_res, z_res)
        t1[i] = VPERM512MASKZ(0x55, k1[i], idx);
        t1[i+8] = VPERM512MASKZ(0x55, res[i], idx2);
    }
    fp2_mul_2x4x1(t1, t1, t2+8);
    fp2_2x4x1_to_4x2x1(res_inv, t1, t1+8);

    if (h2) {
        theta_had_single(res, res);
    }
}

void theta_eval_single(__m512i* res, __m512i* a, __m512i* inv, bool h1, bool h2)
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

void theta_isogeny_chain_single(__m512i* prod, __m512i* null, __m512i* a, __m512i* b, const int n)
{
    bool h1, h2;
    int isom_matrix[16];
    __m512i c1[8], c2[8], p1[8], p2[8], k1[8], k2[8], codom_null[8], inv[8];
    fp2_copy_4x2x1(p1, a);
    fp2_copy_4x2x1(p2, b);
    fp2_copy_4x2x1(codom_null, null);

    for (int i = 0; i < n; i++) {
        // Precompute constants for doubling
        theta_precomp(c1, c2, codom_null);

        // Double the kernel points of order 2^(n+2) to get points of order 2^(1 + 2) = 8
        // TODO: Can get rid of this copy
        fp2_copy_4x2x1(k1, p1);
        fp2_copy_4x2x1(k2, p2);
        for (int j = 0; j < n - i - 1; j++) {
            theta_double_single(k1, k1, c1, c2);
            theta_double_single(k2, k2, c1, c2);
        }

        // For the ultimate and penultimate steps we do extra / less Hadamards
        // to avoid zeros from the product structure.
        if (i == n-1) {
            h1 = true; h2 = false;
        } else if (i == n-2) {
            h1 = false; h2 = false;
        } else {
            h1 = false; h2 = true;
        }
        
        // Compute the codomain of the kernel
        theta_codomain_single(codom_null, inv, k1, k2, h1, h2);

        // Push the kernel through the images
        theta_eval_single(p1, p1, inv, h1, h2);
        theta_eval_single(p2, p2, inv, h1, h2);
    }

    // Splitting
    splitting_isom_matrix(isom_matrix, codom_null);
    splitting_apply_isomorphism(codom_null, isom_matrix, codom_null);
    splitting_theta_to_product(prod, codom_null);
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
    fp_mixaddsub_8x1(res, t, res, 0x33); // p, q, r, s = a + b, a - b, c + d, c - d
    fp_mixaddsub_8x1(res+8, t+8, res+8, 0x33);

    // Split 
    for (int i = 0; i < 16; i++) {
        t[i] = VPERM512(pqpq, res[i]);
        res[i] = VPERM512(rsrs, res[i]);
    }

    // add add sub sub
    fp_mixaddsub_8x1(res, t, res, 0xf); // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
    fp_mixaddsub_8x1(res+8, t+8, res+8, 0xf);
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
    fp_mixaddsub_8x1(res, t, res, 0x55); // p, q, r, s = a + b, a - b, c + d, c - d
    fp_mixaddsub_8x1(res+8, t+8, res+8, 0x55);

    // Split 
    for (int i = 0; i < 16; i++) {
        t[i] = VPERM256(res[i], 0x44);
        res[i] = VPERM256(res[i], 0xee);
    }

    // add add sub sub
    fp_mixaddsub_8x1(res, t, res, 0x33); // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
    fp_mixaddsub_8x1(res+8, t+8, res+8, 0x33);
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

void theta_codomain_pair(__m512i* res, __m512i* res_inv, __m512i* k, bool h1, bool h2)
{
    __m512i k_split[16], t1[16], t2[16], idx, idx2;

    //TODO: Would it be more efficient to use the pair operations here?
    if (h1) {
        theta_had_pair(k, k);
    }

    theta_sqr_pair(k, k);
    theta_had_pair(k, k);

    // Input: k = (k1, k2) = (xi, yi, zi, wi)_{i=1,2}
    // TODO: Is this necessary?
    fp2_8x1x1_to_4x2x1(k_split, k);

    // t1 = (x1, y1, z2, w2)
    // t2 = (y2, x2, w2, z2)
    idx = VSET(
        0x0000000100000000, 0x0000000300000002, 0x0000000500000004, 0x0000000700000006,
        0x0000001900000018, 0x0000001b0000001a, 0x0000001d0000001c, 0x0000001f0000001e
    );
    for (int i = 0; i < 8; i++) {
        t2[i] = VPERM256(k_split[i+8], 0x4e);
        t1[i] = VBLENDPERM(0xffff, k_split[i], k_split[i+8], idx);
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

void theta_eval_pair(__m512i* res, __m512i* a, __m512i* inv, bool h1, bool h2)
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

void theta_isogeny_chain_pair(__m512i* prod, __m512i* null, __m512i* ab, const int n)
{
    bool h1, h2;
    int isom_matrix[16];
    __m512i c1[8], c2[8], p[16], k[16], codom_null[8], inv[8];
    fp2_copy_8x1x1(p, ab);
    fp2_copy_4x2x1(codom_null, null);

    for (int i = 0; i < n; i++) {
        // Precompute constants for doubling
        theta_precomp(c1, c2, codom_null);

        // Double the kernel points of order 2^(n+2) to get points of order 2^(1 + 2) = 8
        // TODO: Can get rid of this copy
        fp2_copy_8x1x1(k, p);

        for (int j = 0; j < n - i - 1; j++) {
            theta_double_pair(k, k, c1, c2);
        }

        // For the ultimate and penultimate steps we do extra / less Hadamards
        // to avoid zeros from the product structure.
        if (i == n-1) {
            h1 = true; h2 = false;
        } else if (i == n-2) {
            h1 = false; h2 = false;
        } else {
            h1 = false; h2 = true;
        }
        
        // Compute the codomain of the kernel
        theta_codomain_pair(codom_null, inv, k, h1, h2);

        // Push the kernel through the images
        theta_eval_pair(p, p, inv, h1, h2);
    }

    // Splitting
    splitting_isom_matrix(isom_matrix, codom_null);
    splitting_apply_isomorphism(codom_null, isom_matrix, codom_null);
    splitting_theta_to_product(prod, codom_null);
}