#include "ec.h"
#include "fp2_lvl3.h"
#include "theta_gluing.h"
#include "theta_middle.h" // TODO: move had, sqr, mul to theta.h? Since they are "core" functions.
#include "intrinsics_avx512.h"

#include <assert.h>
#include "utils.h"
#include "sqi_test_extras.h"

// Done (TODO delete this comment) from 11654 to 1904 cycles.
// TODO: With the faster field arithmetic, the simpler pure 4x2x1 version might be just as efficient.
void gluing_compute_matrix_4x2x1(
    __m512i* g1, __m512i* h1, __m512i* g2, __m512i* h2, __m512i* pi,
    __m512i* x_4, __m512i* z_4, __m512i* u, __m512i* w, __m512i* det
)
{
    __m512i z_sqr[8], a[8], b[8], c[8], d[8];
    
    // TODO: Worth it to use fp2_4x2x1_duplicate_to_8x1x1?
    fp2_mulx2_4x2x1(c, pi, u, z_4, det, z_4);

    fp2_sub_4x2x1(c, c, det);
    fp2_mul_4x2x1(c, c, x_4);
    fp2_sqr_4x2x1(z_sqr, z_4);

    fp2_mulx2_4x2x1(d, b, u, z_sqr, w, z_sqr);
    fp2_neg_4x2x1(a, d);
    fp2_neg_4x2x1(b, b);

    // Compute transpose:
    __m512i idx_lo = _mm512_set_epi64(13, 12, 5, 4, 9, 8, 1, 0);    
    __m512i idx_hi = _mm512_set_epi64(15, 14, 7, 6, 11, 10, 3, 2);
    __m512i ab_lo, ab_hi, cd_lo, cd_hi;

    for (int i = 0; i < 8; i++) {
        // Step 1: 
        // lo: (a0, b0, a2, b2)
        // hi: (a1, b1, a3, b3)
        ab_lo = VBLENDPERM(idx_lo, a[i], b[i]);
        ab_hi = VBLENDPERM(idx_hi, a[i], b[i]);
        cd_lo = VBLENDPERM(idx_lo, c[i], d[i]);
        cd_hi = VBLENDPERM(idx_hi, c[i], d[i]);

        // Step 2:
        // (a0, b0, c0, d0)
        g1[i] = VPERM256HALF(ab_lo, cd_lo, 0x44);
        // (a1, b1, c1, d1)
        h1[i] = VPERM256HALF(ab_hi, cd_hi, 0x44);
        // (a2, b2, c2, d2)
        g2[i] = VPERM256HALF(ab_lo, cd_lo, 0xee);
        // (a3, b3, c3, d3)
        h2[i] = VPERM256HALF(ab_hi, cd_hi, 0xee);
    }
}

// Done (TODO delete this comment) 4052 -> 4002 cycles.
void gluing_action_by_translation_4x2x1(
    __m512i* g1, __m512i* g2, __m512i* h1, __m512i* h2, __m512i* pi,
    __m512i* A24, __m512i* x_4, __m512i* z_4
)
{
    __m512i u[8], w[8], t0[8], t1[8];

    ec_double_4x2x1(u, w, A24, x_4, z_4);
    fp2_mulx2_4x2x1(t0, t1, w, x_4, u, z_4);
    fp2_sub_4x2x1(t0, t0, t1); // det
    gluing_compute_matrix_4x2x1(g1, h1, g2, h2, pi, x_4, z_4, u, w, t0);
}

/** TODO: Micro-optimizations possible:
 * We can merge the shuffling and the 4x2x1_to_8x1x1 comps.
 * Might be able to keep some values in 8x1x1 longer. 
 */
void gluing_change_of_basis_4x2x1(
    __m512i* res_matrix01, __m512i* res_matrix23, 
    __m512i* A24, __m512i* x_4, __m512i* z_4
)
{
    /**
    Giac Notes:

    Computes N projectively which removes 21M + 1I from a batched inversion
    but costs 4S and 34 multiplications (so swaps one inversion for 13M + 1S)

    For AXV, 34 is not a multiple of four, so this adds 9 calls to avx_pairwise_mul
    and one call to avx_sqr.

    An interesting thing to explore is `avx_pairwise_mul` where one of the AVX blobs
    is four copies of the same element. This happens for 6 of the 9 total calls for
    avx_pairwise_mul.
    */
    __m512i g1[8], h1[8], g2[8], h2[8], pi[8];
    __m512i s0[8], s1[8], s2[8], s3[8], s4[8];
    __m512i s5[8], s6[8], s7[8], s8[8], s9[8];
    __m512i t0[8], t1[8], t2[8], idx0, idx1, idx2, idx3;
    __mmask8 mask;

    gluing_action_by_translation_4x2x1(g1, g2, h1, h2, pi, A24, x_4, z_4);

    // Computation of S0
    // T0 = AVX_4x2x1(PI[0], PI[2], PI[0], PI[1])
    // T1 = AVX_4x2x1(PI[1], PI[3], PI[2], PI[3])
    // S0 = avx_421_mul(T0, T1)
    idx0 = VSET(0, 1, 4, 5, 0, 1, 2, 3);
    idx1 = VSET(2, 3, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < 8; i++) {
        s0[i] = VPERM512(idx0, pi[i]);
        t0[i] = VPERM512(idx1, pi[i]);
    }
    fp2_mul_4x2x1(s0, s0, t0);

    // Computation of S1 in 2x4x1
    // T0 = AVX_4x2x1(PI[0], S0[0], 1, 1)
    // T1 = AVX_4x2x1(PI[3], S0[1], 1, 1)
    // S1 = avx_421_mul(T0, T1)
    idx0 = VSET(0, 0, 1, 1, 8, 8, 9, 9);
    idx1 = VSET(6, 6, 7, 7, 10, 10, 11, 11);
    mask = 0x55;
    for (int i = 0; i < 8; i++) {
        s1[i] = VBLENDPERMMASKZ(mask, pi[i], s0[i], idx0);
        t0[i] = VBLENDPERMMASKZ(mask, pi[i], s0[i], idx1);
    }
    fp2_mul_2x4x1(s1, s1, t0);

    // Computation of S2
    // T0 = AVX_4x2x1(G1[0], G1[2], G2[0], G2[2])
    // T1 = AVX_4x2x1(H1[0], H1[0], H2[0], H2[0])
    // T2 = AVX_4x2x1(G1[1], G1[3], G2[1], G2[3])
    // T3 = AVX_4x2x1(H1[2], H1[2], H2[2], H2[2])
    // S2, T2 = avx_421_pair_mul(T0, T1, T2, T3)
    // S2 = avx_421_add(S2, T2)
    idx0 = VSET(0, 1, 4, 5, 8, 9, 12, 13);
    idx1 = VSET(0, 1, 0, 1, 8, 9, 8, 9);
    idx2 = VSET(2, 3, 6, 7, 10, 11, 14, 15);
    idx3 = VSET(4, 5, 4, 5, 12, 13, 12, 13);
    for (int i = 0; i < 8; i++) {
        s2[i] = VBLENDPERM(idx0, g1[i], g2[i]);
        t0[i] = VBLENDPERM(idx1, h1[i], h2[i]);
        t1[i] = VBLENDPERM(idx2, g1[i], g2[i]);
        t2[i] = VBLENDPERM(idx3, h1[i], h2[i]);
    }
    fp2_mulx2_4x2x1(s2, t2, s2, t0, t1, t2);
    fp2_add_4x2x1(s2, s2, t2);

    // Computation of S2 (continued), S3, S4
    // T0 = AVX_4x2x1(S0[1], S0[1], S0[0], S0[0])
    // S3 = AVX_4x2x1(G1[0], G1[0], G1[2], G1[2])
    // T1 = AVX_4x2x1(G2[0], G2[2], G2[0], G2[2])
    // S2, S4 = avx_421_pair_mul(S2, T0, S3, T1)
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 0, 1, 0, 1);
    idx2 = VSET(0, 1, 4, 5, 0, 1, 4, 5);
    for (int i = 0; i < 8; i++) {
        s3[i] = VPERM512(idx0, g1[i]);
        t0[i] = VPERM512(idx1, s0[i]);
        t1[i] = VPERM512(idx2, g2[i]);
    }
    fp2_mulx2_4x2x1(s2, s4, s2, t0, s3, t1);

    // Computation of S4 (continued), S5
    // T0 = AVX_4x2x1(H1[0], H1[0], H1[2], H1[2])
    // S5 = AVX_4x2x1(H2[0], H2[2], H2[0], H2[2])
    // # TODO: every element of T2 is the same, can we use this?
    // T1 = AVX_4x2x1(S0[3], S0[3], S0[3], S0[3])
    // S4, T1 = avx_421_pair_mul(S4, T1, T0, S5)
    idx0 = VSET(0, 1, 4, 5, 0, 1, 4, 5);
    idx1 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx2 = VSET(6, 7, 6, 7, 6, 7, 6, 7);
    for (int i = 0; i < 8; i++) {
        s5[i] = VPERM512(idx0, h2[i]);
        t0[i] = VPERM512(idx1, h1[i]);
        t1[i] = VPERM512(idx2, s0[i]);
    }
    fp2_mulx2_4x2x1(s4, t1, s4, t1, t0, s5);

    // TODO: every element of T2 is the same, can we use this?
    // T0 = AVX_4x2x1(S0[2], S0[2], S0[2], S0[2])
    // T1 = avx_421_mul(T1, T0)
    // S4 = avx_421_add(S4, T1)
    idx0 = VSET(4, 5, 4, 5, 4, 5, 4, 5);
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, s0[i]);
    }
    fp2_mul_4x2x1(t1, t1, t0);
    fp2_add_4x2x1(s4, s4, t1);

    // TODO: this could be optimised as we need only one Fp2 add?
    // T0 = AVX_4x2x1(S1[1], 0, 0, 0)
    // S4 = avx_421_add(S4, T0)
    idx0 = VSET(4, 6, 4, 6, 4, 6, 4, 6); // Since s1 is in 2x4x1.
    mask = 0x3;
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512MASKZ(mask, s1[i], idx0);
    }
    fp2_add_4x2x1(s4, s4, t0);

    // TODO: every element of T1 is the same, can we use this?
    // T0 = AVX_4x2x1(S1[1], S1[1], S1[1], S1[1])
    // T1 = AVX_4x2x1(S2[0], S2[0], S2[1], S2[1])
    // T2 = AVX_4x2x1(S2[2], S2[3], S2[2], S2[3])
    // S4, T4 = avx_421_pair_mul(S4, T0, T1, T2)
    // S4 = avx_421_add(S4, T4)
    idx1 = VSET(0, 1, 0, 1, 2, 3, 2, 3);
    idx2 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, s1[i]);
        t1[i] = VPERM512(idx1, s2[i]);
        t2[i] = VPERM512(idx2, s2[i]);
    }
    fp2_mulx2_4x2x1(s4, t1, s4, t0, t1, t2);
    fp2_add_4x2x1(s4, s4, t1);

    // Computation of S6
    // S6 = AVX_4x2x1(S4[0], S4[0], S4[2], S4[2])
    // T0 = AVX_4x2x1(S4[1], S4[1], S4[3], S4[3])
    // T1 = AVX_4x2x1(H2[1], H2[3], H2[1], H2[3])
    // S6, T2 = avx_421_pair_mul(S6, S5, T0, T1)
    // S6 = avx_421_add(S6, T2)
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    idx2 = VSET(2, 3, 6, 7, 2, 3, 6, 7);
    for (int i = 0; i < 8; i++) {
        s6[i] = VPERM512(idx0, s4[i]);
        t0[i] = VPERM512(idx1, s4[i]);
        t1[i] = VPERM512(idx2, h2[i]);
    }
    fp2_mulx2_4x2x1(s6, t0, s6, s5, t0, t1);
    fp2_add_4x2x1(s6, s6, t0);

    // Computation of S7, S8
    // S7 = AVX_4x2x1(S4[0], S4[1], S4[0], S4[1])
    // S8 = AVX_4x2x1(G1[1], G1[1], G1[3], G1[3])
    // T0 = AVX_4x2x1(S4[2], S4[3], S4[2], S4[3])
    // S7, T1 = avx_421_pair_mul(S7, S3, S8, T0)
    // S7 = avx_421_add(S7, T1)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    idx2 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < 8; i++) {
        s7[i] = VPERM512(idx0, s4[i]);
        s8[i] = VPERM512(idx1, g1[i]);
        t0[i] = VPERM512(idx2, s4[i]);
    }
    fp2_mulx2_4x2x1(s7, t0, s7, s3, s8, t0);
    fp2_add_4x2x1(s7, s7, t0);

    // Computation of S9
    // S9 = AVX_4x2x1(S6[0], S6[1], S6[0], S6[1])
    // T0 = AVX_4x2x1(S6[2], S6[3], S6[2], S6[3])
    // S9, T1 = avx_421_pair_mul(S9, S3, S8, T0)
    // S9 = avx_421_add(S9, T1)
    for (int i = 0; i < 8; i++) {
        s9[i] = VPERM512(idx0, s6[i]);
        t0[i] = VPERM512(idx2, s6[i]);
    }
    fp2_mulx2_4x2x1(s9, t0, s9, s3, s8, t0);
    fp2_add_4x2x1(s9, s9, t0);

    // TODO: every element of T0 is the same, can we use this?
    // T0 = AVX_4x2x1(S1[0], S1[0], S1[0], S1[0])
    // S4 = avx_421_mul(S4, T0)
    idx0 = VSET(0, 2, 0, 2, 0, 2, 0, 2); //s1 is in 2x4x1
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, s1[i]);
    }
    fp2_mul_4x2x1(s4, s4, t0);

    // T0 = AVX_4x2x1(PI[0], PI[0], PI[0], PI[0])
    // T1 = AVX_4x2x1(PI[3], PI[3], PI[3], PI[3])
    // S6, S7 = avx_421_pair_mul(S6, T0, S7, T1)
    idx0 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    idx1 = VSET(6, 7, 6, 7, 6, 7, 6, 7);
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, pi[i]);
        t1[i] = VPERM512(idx1, pi[i]);
    }
    fp2_mulx2_4x2x1(s6, s7, s6, t0, s7, t1);

    // return (
    // AVX_8x1x1(S4[0], S4[1], S4[2], S4[3], S6[0], S6[1], S6[2], S6[3]),
    // AVX_8x1x1(S7[0], S7[1], S7[2], S7[3], S9[0], S9[1], S9[2], S9[3]),
    // )
    fp2_4x2x1_to_8x1x1(res_matrix01, s4, s6);
    fp2_4x2x1_to_8x1x1(res_matrix23, s7, s9);
}

// Done (TODO delete this comment) from 9122 to 108 cycles.
void gluing_internal_sum(__m512i* res_P, __m512i* P0P1, __m512i* P2P3)
{
    __m512i t0[16], t1[16], idx0, idx1;

    // Pa0, Pa1, Pb0, Pb1, Pc0, Pc1, Pd0, Pd1 = P0P1
    // Pa2, Pa3, Pb2, Pb3, Pc2, Pc3, Pd2, Pd3 = P2P3

    // Pa0 + Pb0, Pc0 + Pd0, Pa1 + Pb1, Pc1 + Pd1, ... :
    idx0 = VSET(0, 4, 1, 5, 8, 12, 9, 13);
    idx1 = VSET(2, 6, 3, 7, 10, 14, 11, 15);
    for (int i = 0; i < 16; i++) {
        t0[i] = VBLENDPERM(idx0, P0P1[i], P2P3[i]);
        t1[i] = VBLENDPERM(idx1, P0P1[i], P2P3[i]);
    }
    fp2_add_8x1x1(t0, t0, t1);

    // Pa0 + Pb0 + Pc0 + Pd0, Pa1 + Pb1 + Pc1 + Pd1, ...
    fp2_8x1x1_to_4x2x1(t0, t1, t0);
    fp2_add_4x2x1(res_P, t0, t1);
}

// Done (TODO delete this comment)
void gluing_internal_sum_pair(
    __m512i* res_PQ, 
    __m512i* P0Q1, __m512i* Q0P1, __m512i* P2Q3, __m512i* Q2P3
)
{
    __m512i t0[16], t1[16], t2[16], t3[16], idx0, idx1;
    
    // Pa0, Qa1, Pb0, Qb1, Pc0, Qc1, Pd0, Qd1 = P0Q1
    // Qa0, Pa1, Qb0, Pb1, Qc0, Pc1, Qd0, Pd1 = Q0P1
    // Pa2, Qa3, Pb2, Qb3, Pc2, Qc3, Pd2, Qd3 = P2Q3
    // Qa2, Pa3, Qb2, Pb3, Qc2, Pc3, Qd2, Pd3 = Q2P3

    // t0 = Pa0 + Pb0, Pc0 + Pd0, Qa0 + Qb0, Qc0 + Qd0, Pa1 + Pb1, ...
    // t2 = Pa2 + Pb2, Pc2 + Pd2, ...
    idx0 = VSET(0, 4, 8, 12, 9, 13, 1, 5);
    idx1 = VSET(2, 6, 10, 14, 11, 15, 3, 7);
    for (int i = 0; i < 16; i++) {
        t0[i] = VBLENDPERM(idx0, P0Q1[i], Q0P1[i]);
        t1[i] = VBLENDPERM(idx1, P0Q1[i], Q0P1[i]);
        t2[i] = VBLENDPERM(idx0, P2Q3[i], Q2P3[i]);
        t3[i] = VBLENDPERM(idx1, P2Q3[i], Q2P3[i]);
    }
    fp2_add_8x1x1(t0, t0, t1);
    fp2_add_8x1x1(t2, t2, t3);

    // Pa0 + Pb0 + Pc0 + Pd0, Qa0 + Qb0 + Qc0 + Qd0, ...
    idx0 = VSET(0, 2, 4, 6, 8, 10, 12, 14);
    idx1 = VSET(1, 3, 5, 7, 9, 11, 13, 15);
    for (int i = 0; i < 16; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i], t2[i]);
        t3[i] = VBLENDPERM(idx1, t0[i], t2[i]);
    }
    fp2_add_8x1x1(res_PQ, t1, t3);    
}

// Done (TODO delete this comment)
void gluing_apply_matrix(
    __m512i* res_P, 
    __m512i* matrix01, __m512i* matrix23, __m512i* P
)
{
    __m512i P0P1[16], P2P3[16];

    fp2_4x2x1_duplicate_to_8x1x1(P2P3, P);
    fp2_mul_8x1x1(P0P1, matrix01, P2P3);
    fp2_mul_8x1x1(P2P3, matrix23, P2P3);

    gluing_internal_sum(res_P, P0P1, P2P3);
}

// Done (TODO delete this comment)
void gluing_apply_matrix_pair(
    __m512i* res_PQ, 
    __m512i* matrix01, __m512i* matrix23, __m512i* PQ
)
{
    __m512i QP[16], P0Q1[16], Q0P1[16], P2Q3[16], Q2P3[16];

    // Swap P || Q to Q || P
    for (int i = 0; i < 16; i++) {
        QP[i] = VSHUFFLE(PQ[i], 0x4e);
    }

    // Compute first row mul P, second row mul Q
    fp2_mul_8x1x1(P0Q1, matrix01, PQ);

    // Compute first row mul Q, second row mul P
    fp2_mul_8x1x1(Q0P1, matrix01, QP);

    // Compute third row mul P, fourth row mul Q
    fp2_mul_8x1x1(P2Q3, matrix23, PQ);

    // Compute third row mul Q, fourth row mul P
    fp2_mul_8x1x1(Q2P3, matrix23, QP);

    gluing_internal_sum_pair(res_PQ, P0Q1, Q0P1, P2Q3, Q2P3);
}

// Done (TODO delete this comment)
void gluing_product_to_theta_pair(
    __m512i* res_P12,
    __m512i* x, __m512i* z, __m512i* matrix1, __m512i* matrix2
)
{
    __m512i t0[16], t1[16], idx0_re, idx0_im, idx1_re, idx1_im;

    // t0 = AVX_8x1x1(X1, X2, X1, X2, Z1, Z2, Z1, Z2)
    // t1 = AVX_8x1x1(X3, X4, Z3, Z4, X3, X4, Z3, Z4)
    idx0_re = VSET(0, 2, 0, 2, 8, 10, 8, 10);
    idx0_im = VSET(1, 3, 1, 3, 9, 11, 9, 11);
    idx1_re = VSET(4, 6, 12, 14, 4, 6, 12, 14);
    idx1_im = VSET(5, 7, 13, 15, 5, 7, 13, 15);

    for (int i = 0; i < 8; i++) {
        t0[i] = VBLENDPERM(idx0_re, x[i], z[i]);
        t0[i+8] = VBLENDPERM(idx0_im, x[i], z[i]);
        t1[i] = VBLENDPERM(idx1_re, x[i], z[i]);
        t1[i+8] = VBLENDPERM(idx1_im, x[i], z[i]);
    }

    fp2_mul_8x1x1(res_P12, t0, t1);
    gluing_apply_matrix_pair(res_P12, matrix1, matrix2, res_P12);
}

// TODO: Decide how to implement last assert.
// TODO: Output (OA_inv, OA_inv) in 8x1x1 for special eval.
void gluing_codomain(
    __m512i* res_OA, __m512i* res_OA_dual_inv, __m512i* res_J_prime, __m512i* res_matrix01, __m512i* res_matrix23,
    __m512i* A24_12, __m512i* x_8, __m512i* z_8
)
{
    __m512i x_8_[8], z_8_[8], x_4[8], z_4[8], P12[16];
    __m512i t0[8], t1[8], t2[8], t3[8], idx0, idx1, idx2;
    __mmask8 mask;

    // X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)
    ec_jac_to_xz_4x2x1(x_8_, z_8_, x_8, z_8);

    // Double points to recover 4-torsion for change of basis
    ec_double_4x2x1(x_4, z_4, A24_12, x_8_, z_8_);

    // Compute the change of basis for our theta structure
    gluing_change_of_basis_4x2x1(res_matrix01, res_matrix23, A24_12, x_4, z_4);

    // Lift both 8-torsion points above the generator of our basis in the theta model.
    gluing_product_to_theta_pair(P12, x_8_, z_8_, res_matrix01, res_matrix23);

    // compute their dual-image.
    theta_sqr_pair(P12, P12);
    theta_had_pair(P12, P12);

    // X1, X2, Y1, Y2, Z1, Z2, W1, W2 = P12
    // TODO: Wrap asserts in #ifdefined, so they don't impact benchmarks
    int zmask = fp2_is_zero_8x1x1(P12);

    // assert that W1 = 0 and W2 = 0;
    int last_is_zero = ((zmask & 0x40) >> 6) & ((zmask & 0x80) >> 7);
    assert(last_is_zero);

    // assert that x1, x2, y1, z2 != 0
    zmask = ~zmask;
    int projective_factors_not_zero = 
        (zmask & 0x1) &
        ((zmask & 0x2) >> 1) &
        ((zmask & 0x4) >> 2) &
        ((zmask & 0x20) >> 5);
    assert(projective_factors_not_zero);

    // ============================================================
    // Just like in the generic codomain there's a bit of ugliness
    // here and the below is my best effort to compute:
    //
    // A = X1 * X2
    // B = Y1 * X2
    // C = X1 * Z2

    // A_inv = Y1 * Z2
    // B_inv = C
    // C_inv = B

    // x = Y1 * C
    // y = Z1 * B
    //
    // Which requires 6 multiplications and we can do it by doing
    // two calls to pairwise_multiplication with the last element
    // of t1 and t2 being 1 (so we only need 3/4 multiplications
    // from this function)
    // ============================================================

    // Computing B, C in 2x4x1
    idx0 = VSET(2, 2, 10, 10, 0, 0, 8, 8); // (Y1, X1)
    idx1 = VSET(1, 1, 9, 9, 5, 5, 13, 13); // (X2, Z2)
    mask = 0x55;
    for (int i = 0; i < 8; i++) {
        t0[i] = VBLENDPERMMASKZ(mask, P12[i], P12[i+8], idx0);
        t1[i] = VBLENDPERMMASKZ(mask, P12[i], P12[i+8], idx1);
        
    }
    fp2_mul_2x4x1(t2, t0, t1); // t2 = (B, C)
    
    // Computing A, A_inv, x, y
    // We use that t0 has both the real and imaginary components of
    // X1 and Y1, and t1 has both for X2 and Z2.
    idx0 = VSET(4, 6, 0, 2, 12, 14, 8, 10); // (X1, Y1, C, B)
    idx1 = VSET(0, 2, 4, 6, 10, 10, 12, 12); // (X2, Z2, re(Y1), re(Z1))
    idx2 = VSET(0, 1, 2, 3, 4, 10, 6, 12); // (X2, Z2, Y1, Z1)
    for (int i = 0; i < 8; i++) {
        t3[i] = VBLENDPERM(idx0, t0[i], t2[i]);
        t0[i] = VBLENDPERM(idx1, t1[i], P12[i]);
        t0[i] = VBLENDPERM(idx2, t0[i], P12[i+8]);
    }
    fp2_mul_4x2x1(t0, t0, t3); // t0 = (A, A_inv, x, y)

    // TODO: Needed?
    // if (Y1 * B_inv != x) or (X2 * A_inv != Z2 * C_inv):
    //    raise ValueError("input kernel is not isotropic")
    // fp2_mul(t_unvec, P12_unvec+2, ABCD_unvec+2);
    // fp2_mul(t_unvec+1, P12_unvec+1, inv_unvec);
    // fp2_mul(t_unvec+2, P12_unvec+5, ABCD_unvec+1);
    // int ker_is_isotropic = fp2_is_equal(t_unvec, inv_unvec+1) 
    //     & fp2_is_equal(t_unvec+1, t_unvec+2);
    // assert(ker_is_isotropic);

    // OA = hadamard(AVX_4x2x1(A, B, C, 0))
    idx0 = VSET(0, 1, 8, 10, 12, 14, 0, 0);
    // OA_dual_inv = AVX_4x2x1(A_inv, C, B, 0)
    idx1 = VSET(2, 3, 12, 14, 8, 10, 0, 0);
    mask = 0x3f;
    // J_prime = AVX_4x2x1(y, y, x, x)
    idx2 = VSET(6, 7, 6, 7, 4, 5, 4, 5);
    for (int i = 0; i < 8; i++) {
        res_OA[i] = VBLENDPERMMASKZ(mask, t0[i], t2[i], idx0);
        res_OA_dual_inv[i] = VBLENDPERMMASKZ(mask, t0[i], t2[i], idx1);
        res_J_prime[i] = VPERM512(idx2, t0[i]);
    }
    theta_had_single(res_OA, res_OA);
}

// TODO: Haven't investigated the possibility of micro-optimizations.
void gluing_eval_pair(
    __m512i* res,
    __m512i* A_12,  __m512i* xP, __m512i* yP, __m512i* zP,
    __m512i* xT_8, __m512i* yT_8, __m512i* zT_8,
    __m512i* J_prime, __m512i* matrix01, __m512i* matrix23
)
{
    __m512i u[8], v[8], w[8], t0[16], t1[16], t2[16];
    __m512i idx0_re, idx0_im, idx1_re, idx1_im;
    __mmask8 mask;

    // Add components
    // u1, v1, w1 = add_components(E1, P1, T1_8)
    // u3, v3, w3 = add_components(E1, Q1, T1_8)
    // u2, v2, w2 = add_components(E2, P2, T2_8)
    // u4, v4, w4 = add_components(E2, Q2, T2_8)
    ec_add_components_4x2x1(u, v, w, A_12, xP, yP, zP, xT_8, yT_8, zT_8);

    // ===============================================================
    // Given:
    // u1, u3, u2, u4 = U
    // v1, v3, v2, v4 = V
    // w1, w3, w2, w4 = W
    // 
    // Compute:
    // U1 = AVX_4x2x1(u1 * u2 + v1 * v2, u1 * w2, w1 * u2, w1 * w2)
    // U2 = AVX_4x2x1(u3 * u4 + v3 * v4, u3 * w4, w3 * u4, w3 * w4)
    // V1 = AVX_4x2x1(v1 * u2 + u1 * v2, v1 * w2, w1 * v2, 0)
    // V2 = AVX_4x2x1(v3 * u4 + u3 * v4, v3 * w4, w3 * v4, 0)
    // ===============================================================

    // t0 = (u1u2, u3u4, u1w2, u3w4, w1u2, w3u4, w1w2, w3w4)
    idx0_re = VSET(0, 2, 0, 2, 8, 10, 8, 10); // (u1, u3, u1, u3, w1, w3, w1, w3)
    idx0_im = VSET(1, 3, 1, 3, 9, 11, 9, 11);
    idx1_re = VSET(4, 6, 12, 14, 4, 6, 12, 14); // (u2, u4, w2, w4, u2, u4, w2, w4)
    idx1_im = VSET(5, 7, 13, 15, 5, 7, 13, 15);
    for (int i = 0; i < 8; i++) {
        t0[i] = VBLENDPERM(idx0_re, u[i], w[i]);
        t0[i+8] = VBLENDPERM(idx0_im, u[i], w[i]);
        t1[i] = VBLENDPERM(idx1_re, u[i], w[i]);
        t1[i+8] = VBLENDPERM(idx1_im, u[i], w[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    // t1 = (v1u2, v3u4, u1v2, u3v4, v1w2, v3w4, w1v2, w3v4)
    idx0_re = VSET(0, 2, 4, 6, 0, 2, 4, 6); // (v1, v3, v2, v4, v1, v3, v2, v4)
    idx0_im = VSET(1, 3, 5, 7, 1, 3, 5, 7);
    idx1_re = VSET(4, 6, 0, 2, 12, 14, 8, 10); // (u2, u4, u1, u3, w2, w4, w1, w3)
    idx1_im = VSET(5, 7, 1, 3, 13, 15, 9, 11);
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512(idx0_re, v[i]);
        t1[i+8] = VPERM512(idx0_im, v[i]);
        t2[i] = VBLENDPERM(idx1_re, u[i], w[i]);
        t2[i+8] = VBLENDPERM(idx1_im, u[i], w[i]);
    }
    fp2_mul_8x1x1(t1, t1, t2);

    // t2[0:7] = (v1v2, v3v4) in 2x4x1
    idx0_re = VSET(0, 0, 1, 1, 2, 2, 3, 3);
    idx1_re = VSET(4, 4, 5, 5, 6, 6, 7, 7);
    mask = 0x55;
    for (int i = 0; i < 8; i++) {
        t2[i] = VPERM512MASKZ(mask, v[i], idx0_re);
        t2[i+8] = VPERM512MASKZ(mask, v[i], idx1_re);
    }
    fp2_mul_2x4x1(t2, t2, t2+8);

    //t2[8:15] = (u1u2, u3u4, v1u2, v3u4)
    idx0_re = VSET(0, 0, 1, 1, 8, 8, 9, 9); // (u1u2, u3u4, v1u2, v3u4)
    idx0_im = VSET(0, 8, 2, 9, 4, 5, 6, 7); // for im of u1u2 & u3u4
    idx1_im = VSET(0, 1, 2, 3, 4, 8, 6, 9); // for im of v1u2 & v3u4
    for (int i = 0; i < 8; i++) {
        t2[i+8] = VBLENDPERM(idx0_re, t0[i], t1[i]);
        t2[i+8] = VBLENDPERM(idx0_im, t2[i+8], t0[i+8]);
        t2[i+8] = VBLENDPERM(idx1_im, t2[i+8], t1[i+8]);
    }
    // t2[0:7] = (v1v2, v3v4, u1v2, u3v4)
    idx0_re = VSET(0, 2, 4, 6, 10, 10, 11, 11); // (v1v2, v3v4, re(u1v2), re(u3v4))
    idx0_im = VSET(0, 1, 2, 3, 4, 10, 6, 11); // (v1v2, v3v4, u1v2, u3v4)
    for (int i = 0; i < 8; i++) {
        t2[i] = VBLENDPERM(idx0_re, t2[i], t1[i]);
        t2[i] = VBLENDPERM(idx0_im, t2[i], t1[i+8]);
    }

    // t2[0:7] = (u1u2+v1v2, u3u4+v3v4, v1u2+u1v2, v3u4+u3v4)
    fp2_add_4x2x1(t2, t2, t2+8);

    // U12 = (u1u2+v1v2, u3u4+v3v4, u1w2, u3w4, w1u2, w3u4, w1w2, w3w4)
    // V12 = (v1u2+u1v2, v3u4+u3v4, v1w2, v3w4, w1v2, w3v4, 0, 0)
    idx0_re = VSET(8, 10, 2, 3, 4, 5, 6, 7);
    idx0_im = VSET(9, 11, 2, 3, 4, 5, 6, 7);
    idx1_re = VSET(12, 14, 4, 5, 6, 7, 0, 0);
    idx1_im = VSET(13, 15, 4, 5, 6, 7, 0, 0);
    mask = 0x3f;
    for (int i = 0; i < 8; i++) {
        t0[i] = VBLENDPERM(idx0_re, t0[i], t2[i]);
        t0[i+8] = VBLENDPERM(idx0_im, t0[i+8], t2[i]);
        t1[i] = VBLENDPERMMASKZ(mask, t1[i], t2[i], idx1_re);
        t1[i+8] = VBLENDPERMMASKZ(mask, t1[i+8], t2[i], idx1_im);
    }

    // Apply the isomorphism by N on U1, U2, V1, V2
    gluing_apply_matrix_pair(t0, matrix01, matrix23, t0);
    gluing_apply_matrix_pair(t1, matrix01, matrix23, t1);

    fp2_sqr_8x1x1(t0, t0);
    fp2_sqr_8x1x1(t1, t1);

    // Hadamard of component wise subtraction of U, V
    fp2_sub_8x1x1(res, t0, t1);

    theta_had_pair(res, res);

    // Scaling by (y : y : x : x)
    theta_mul_pair(res, res, J_prime);

    // Final Hadamard
    theta_had_pair(res, res);
}

// Done (TODO delete this comment)
void gluing_eval_product_to_theta(
    __m512i* res,
    ec_point_t* P1, ec_point_t* P2, __m512i* matrix01, __m512i* matrix23
)
{
    __m512i t0[8];
    fp2_t unvec0[4], unvec1[4];

    // P = AVX_4x2x1(X1, X1, Z1, Z1)
    unvec0[0] = P1->x; unvec0[1] = P1->x; unvec0[2] = P1->z; unvec0[3] = P1->z;
    // T0 = AVX_4x2x1(X2, Z2, X2, Z2)
    unvec1[0] = P2->x; unvec1[1] = P2->z; unvec1[2] = P2->x; unvec1[3] = P2->z;
    fp2_unvectorized_to_4x2x1(res, unvec0);
    fp2_unvectorized_to_4x2x1(t0, unvec1);

    fp2_mul_4x2x1(res, res, t0);
    gluing_apply_matrix(res, matrix01, matrix23, res);
}

// Done (TODO delete this comment)
void gluing_eval_product_to_theta_pair(
    __m512i* res,
    ec_point_t* P1, ec_point_t* P2, ec_point_t* Q1, ec_point_t* Q2, 
    __m512i* matrix01, __m512i* matrix23
)
{
    __m512i t0[16];
    fp2_t unvec0[8], unvec1[8];
    // PQ = AVX_8x1x1(X1, X3, X1, X3, Z1, Z3, Z1, Z3)
    unvec0[0] = P1->x; unvec0[1] = Q1->x; unvec0[2] = P1->x; unvec0[3] = Q1->x;
    unvec0[4] = P1->z; unvec0[5] = Q1->z; unvec0[6] = P1->z; unvec0[7] = Q1->z;
    // T0 = AVX_8x1x1(X2, X4, Z2, Z4, X2, X4, Z2, Z4)
    unvec1[0] = P2->x; unvec1[1] = Q2->x; unvec1[2] = P2->z; unvec1[3] = Q2->z;
    unvec1[4] = P2->x; unvec1[5] = Q2->x; unvec1[6] = P2->z; unvec1[7] = Q2->z;
    fp2_unvectorized_to_8x1x1(res, unvec0);
    fp2_unvectorized_to_8x1x1(t0, unvec1);  
    
    fp2_mul_8x1x1(res, res, t0);
    gluing_apply_matrix_pair(res, matrix01, matrix23, res);
}

// Done (TODO delete this comment)
void gluing_eval_special(
    __m512i* res,
    theta_couple_point_t* P12, __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23
)
{
    assert(fp2_is_zero(&P12->P1.z) || fp2_is_zero(&P12->P2.z));
    gluing_eval_product_to_theta(res, &P12->P1, &P12->P2, matrix01, matrix23);
    fp2_sqr_4x2x1(res, res);
    theta_had_single(res, res);

    assert(fp2_is_zero_4x2x1(res) & 0x8); // this is the special case we must have
    fp2_mul_4x2x1(res, res, OA_inv);
    theta_had_single(res, res);
}

// Done (TODO delete this comment)
void gluing_eval_special_pair(
    __m512i* res,
    theta_couple_point_t* P12, theta_couple_point_t* Q12,
    __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23
)
{
    assert(fp2_is_zero(&P12->P1.z) || fp2_is_zero(&P12->P2.z));
    assert(fp2_is_zero(&Q12->P1.z) || fp2_is_zero(&Q12->P2.z));
    gluing_eval_product_to_theta_pair(
        res, &P12->P1, &P12->P2, &Q12->P1, &Q12->P2, matrix01, matrix23
    );
    fp2_sqr_8x1x1(res, res);
    theta_had_pair(res, res);

    // w1 = 0 and w2 = 0:
    assert((fp2_is_zero_8x1x1(res) & 0xc0) == 0xc0 ); 
    theta_mul_pair(res, res, OA_inv);
    theta_had_pair(res, res);
}

// Done (TODO delete this comment)
void gluing_eval_special_list_pair(
    __m512i* res_pts, theta_couple_point_t* pts, __m512i* inv, 
    __m512i* matrix01, __m512i* matrix23, const int num_pts
)
{
    bool num_pts_is_odd = num_pts & 0x1;
    int num_pair_pts = num_pts / 2;

    for (int i = 0; i < num_pair_pts; i++) {
        gluing_eval_special_pair(
            res_pts + (16*i), pts + (2*i), pts + (2*i+1), 
            inv, matrix01, matrix23
        );
    }
    if (num_pts_is_odd) {
        gluing_eval_special(
            res_pts + 8*(num_pts-1), pts + (num_pts-1),
            inv, matrix01, matrix23
        );
    }
}