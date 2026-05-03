#include "ec.h"
#include "fp_consts.h"
#include "fp_unvec.h"
#include "fp2.h"
#include "theta_gluing.h"
#include "theta_middle.h" // TODO: move had, sqr, mul to theta.h? Since they are "core" functions.
#include "intrinsics_avx512.h"

#include "utils.h"
#include "sqi_test_extras.h"

// Done (TODO delete this comment) from 11654 to 1904 cycles.
// TODO: With the faster field arithmetic, the simpler pure 4x2x1 version might be just as efficient.
void gluing_compute_matrix_4x2x1(
    __m512i* g1, __m512i* h1, __m512i* g2, __m512i* h2, __m512i* pi,
    __m512i* x_4, __m512i* z_4, __m512i* u, __m512i* w, __m512i* det
)
{
    __m512i z_sqr[NUM_LIMBS], a[NUM_LIMBS], b[NUM_LIMBS], c[NUM_LIMBS], d[NUM_LIMBS];
    
    // TODO: Worth it to use fp2_4x2x1_duplicate_to_8x1x1?
    fp2_mulx2_4x2x1(c, pi, u, z_4, det, z_4);

    fp2_sub_4x2x1(c, c, det);
    fp2_mul_4x2x1(c, c, x_4);
    fp2_sqr_4x2x1(z_sqr, z_4);

    fp2_mulx2_4x2x1(d, b, u, z_sqr, w, z_sqr);
    fp2_neg_4x2x1(a, d);
    fp2_neg_4x2x1(b, b);

    // Compute transpose:
    __m512i idx_lo = VSET(0, 1, 8, 9, 4, 5, 12, 13);    
    __m512i idx_hi = VSET(2, 3, 10, 11, 6, 7, 14, 15);
    __m512i ab_lo, ab_hi, cd_lo, cd_hi;

    for (int i = 0; i < NUM_LIMBS; i++) {
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
    __m512i u[NUM_LIMBS], w[NUM_LIMBS], t0[NUM_LIMBS], t1[NUM_LIMBS];

    ec_double_4x4x2x1(u, w, A24, x_4, z_4);
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
    __m512i g1[NUM_LIMBS], h1[NUM_LIMBS], g2[NUM_LIMBS], h2[NUM_LIMBS], pi[NUM_LIMBS];
    __m512i s0[NUM_LIMBS], s1[NUM_LIMBS], s2[NUM_LIMBS], s3[NUM_LIMBS], s4[NUM_LIMBS];
    __m512i s5[NUM_LIMBS], s6[NUM_LIMBS], s7[NUM_LIMBS], s8[NUM_LIMBS], s9[NUM_LIMBS];
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS], idx0, idx1, idx2, idx3;
    __mmask8 mask;

    gluing_action_by_translation_4x2x1(g1, g2, h1, h2, pi, A24, x_4, z_4);

    // Computation of S0
    // T0 = AVX_4x2x1(PI[0], PI[2], PI[0], PI[1])
    // T1 = AVX_4x2x1(PI[1], PI[3], PI[2], PI[3])
    // S0 = avx_421_mul(T0, T1)
    idx0 = VSET(0, 1, 4, 5, 0, 1, 2, 3);
    idx1 = VSET(2, 3, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, s0[i]);
    }
    fp2_mul_4x2x1(t1, t1, t0);
    fp2_add_4x2x1(s4, s4, t1);

    // TODO: this could be optimised as we need only one Fp2 add?
    // T0 = AVX_4x2x1(S1[1], 0, 0, 0)
    // S4 = avx_421_add(S4, T0)
    idx0 = VSET(4, 6, 4, 6, 4, 6, 4, 6); // Since s1 is in 2x4x1.
    mask = 0x3;
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    for (int i = 0; i < NUM_LIMBS; i++) {
        s9[i] = VPERM512(idx0, s6[i]);
        t0[i] = VPERM512(idx2, s6[i]);
    }
    fp2_mulx2_4x2x1(s9, t0, s9, s3, s8, t0);
    fp2_add_4x2x1(s9, s9, t0);

    // TODO: every element of T0 is the same, can we use this?
    // T0 = AVX_4x2x1(S1[0], S1[0], S1[0], S1[0])
    // S4 = avx_421_mul(S4, T0)
    idx0 = VSET(0, 2, 0, 2, 0, 2, 0, 2); //s1 is in 2x4x1
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, s1[i]);
    }
    fp2_mul_4x2x1(s4, s4, t0);

    // T0 = AVX_4x2x1(PI[0], PI[0], PI[0], PI[0])
    // T1 = AVX_4x2x1(PI[3], PI[3], PI[3], PI[3])
    // S6, S7 = avx_421_pair_mul(S6, T0, S7, T1)
    idx0 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    idx1 = VSET(6, 7, 6, 7, 6, 7, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], idx0, idx1;

    // Pa0, Pa1, Pb0, Pb1, Pc0, Pc1, Pd0, Pd1 = P0P1
    // Pa2, Pa3, Pb2, Pb3, Pc2, Pc3, Pd2, Pd3 = P2P3

    // Pa0 + Pb0, Pc0 + Pd0, Pa1 + Pb1, Pc1 + Pd1, ... :
    idx0 = VSET(0, 4, 1, 5, 8, 12, 9, 13);
    idx1 = VSET(2, 6, 3, 7, 10, 14, 11, 15);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
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
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS], t3[2*NUM_LIMBS], idx0, idx1;
    
    // Pa0, Qa1, Pb0, Qb1, Pc0, Qc1, Pd0, Qd1 = P0Q1
    // Qa0, Pa1, Qb0, Pb1, Qc0, Pc1, Qd0, Pd1 = Q0P1
    // Pa2, Qa3, Pb2, Qb3, Pc2, Qc3, Pd2, Qd3 = P2Q3
    // Qa2, Pa3, Qb2, Pb3, Qc2, Pc3, Qd2, Pd3 = Q2P3

    // t0 = Pa0 + Pb0, Pc0 + Pd0, Qa0 + Qb0, Qc0 + Qd0, Pa1 + Pb1, ...
    // t2 = Pa2 + Pb2, Pc2 + Pd2, ...
    idx0 = VSET(0, 4, 8, 12, 9, 13, 1, 5);
    idx1 = VSET(2, 6, 10, 14, 11, 15, 3, 7);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
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
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i], t2[i]);
        t3[i] = VBLENDPERM(idx1, t0[i], t2[i]);
    }
    fp2_add_8x1x1(res_PQ, t1, t3);    
}

void gluing_apply_matrix(
    __m512i* res_P, 
    __m512i* matrix01, __m512i* matrix23, __m512i* P
)
{
    __m512i P0P1[2*NUM_LIMBS], P2P3[2*NUM_LIMBS];

    fp2_4x2x1_duplicate_to_8x1x1(P2P3, P);
    fp2_mul_8x1x1(P0P1, matrix01, P2P3);
    fp2_mul_8x1x1(P2P3, matrix23, P2P3);

    gluing_internal_sum(res_P, P0P1, P2P3);
}

void gluing_apply_matrix_pair(
    __m512i* res_PQ, 
    __m512i* matrix01, __m512i* matrix23, __m512i* PQ
)
{
    __m512i QP[2*NUM_LIMBS], P0Q1[2*NUM_LIMBS], Q0P1[2*NUM_LIMBS], P2Q3[2*NUM_LIMBS], Q2P3[2*NUM_LIMBS];

    // Swap P || Q to Q || P
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
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

void gluing_product_to_theta_pair(
    __m512i* res_P12,
    __m512i* x, __m512i* z, __m512i* matrix1, __m512i* matrix2
)
{
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], idx0_re, idx0_im, idx1_re, idx1_im;

    // t0 = AVX_8x1x1(X1, X2, X1, X2, Z1, Z2, Z1, Z2)
    // t1 = AVX_8x1x1(X3, X4, Z3, Z4, X3, X4, Z3, Z4)
    idx0_re = VSET(0, 2, 0, 2, 8, 10, 8, 10);
    idx0_im = VSET(1, 3, 1, 3, 9, 11, 9, 11);
    idx1_re = VSET(4, 6, 12, 14, 4, 6, 12, 14);
    idx1_im = VSET(5, 7, 13, 15, 5, 7, 13, 15);

    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0_re, x[i], z[i]);
        t0[i + NUM_LIMBS] = VBLENDPERM(idx0_im, x[i], z[i]);
        t1[i] = VBLENDPERM(idx1_re, x[i], z[i]);
        t1[i + NUM_LIMBS] = VBLENDPERM(idx1_im, x[i], z[i]);
    }

    fp2_mul_8x1x1(res_P12, t0, t1);
    gluing_apply_matrix_pair(res_P12, matrix1, matrix2, res_P12);
}

// TODO: Output (OA_inv, OA_inv) in 8x1x1 for special eval.
bool gluing_codomain(
    __m512i* res_OA, __m512i* res_OA_dual_inv, __m512i* res_J_prime, __m512i* res_matrix01, __m512i* res_matrix23,
    __m512i* A24_12, __m512i* x_8, __m512i* z_8, const bool verify
)
{
    __m512i x_8_[NUM_LIMBS], z_8_[NUM_LIMBS], x_4[NUM_LIMBS], z_4[NUM_LIMBS], P12[2*NUM_LIMBS];
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS], t3[NUM_LIMBS], idx0, idx1, idx2;
    __mmask8 mask;

    // X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)
    ec_jac_to_xz_4x4x2x1(x_8_, z_8_, x_8, z_8);

    // Double points to recover 4-torsion for change of basis
    ec_double_4x4x2x1(x_4, z_4, A24_12, x_8_, z_8_);

    // Compute the change of basis for our theta structure
    gluing_change_of_basis_4x2x1(res_matrix01, res_matrix23, A24_12, x_4, z_4);

    // Lift both 8-torsion points above the generator of our basis in the theta model.
    gluing_product_to_theta_pair(P12, x_8_, z_8_, res_matrix01, res_matrix23);

    // compute their dual-image.
    theta_sqr_pair(P12, P12);
    theta_had_pair(P12, P12);

    if (verify) {
        // X1, X2, Y1, Y2, Z1, Z2, W1, W2 = P12
        int zmask = fp2_is_zero_8x1x1(P12);

        // assert that W1 = 0 and W2 = 0;
        int last_is_zero = ((zmask & 0x40) >> 6) & ((zmask & 0x80) >> 7);
        if (!last_is_zero) { return false; }

        // assert that x1, x2, y1, z2 != 0
        zmask = ~zmask;
        int projective_factors_not_zero = 
            (zmask & 0x1) &
            ((zmask & 0x2) >> 1) &
            ((zmask & 0x4) >> 2) &
            ((zmask & 0x20) >> 5);
        if (!projective_factors_not_zero) { return false; }
    }

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
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERMMASKZ(mask, P12[i], P12[i + NUM_LIMBS], idx0);
        t1[i] = VBLENDPERMMASKZ(mask, P12[i], P12[i + NUM_LIMBS], idx1);
        
    }
    fp2_mul_2x4x1(t2, t0, t1); // t2 = (B, C)
    
    // Computing A, A_inv, x, y
    // We use that t0 has both the real and imaginary components of
    // X1 and Y1, and t1 has both for X2 and Z2.
    idx0 = VSET(4, 6, 0, 2, 12, 14, 8, 10); // (X1, Y1, C, B)
    idx1 = VSET(0, 2, 4, 6, 10, 10, 12, 12); // (X2, Z2, re(Y1), re(Z1))
    idx2 = VSET(0, 1, 2, 3, 4, 10, 6, 12); // (X2, Z2, Y1, Z1)
    for (int i = 0; i < NUM_LIMBS; i++) {
        t3[i] = VBLENDPERM(idx0, t0[i], t2[i]);
        t0[i] = VBLENDPERM(idx1, t1[i], P12[i]);
        t0[i] = VBLENDPERM(idx2, t0[i], P12[i + NUM_LIMBS]);
    }
    fp2_mul_4x2x1(t0, t0, t3); // t0 = (A, A_inv, x, y)

    if (verify) {
        // if (Y1 * B_inv != x) or (X2 * A_inv != Z2 * C_inv):
        //    raise ValueError("input kernel is not isotropic")
        idx0 = VSET(2, 10, 1, 9, 5, 13, 0, 0); // (Y1, X2, Z2, -)
        idx1 = VSET(12, 14, 2, 3, 8, 10, 0, 0); // (C, A_inv, B, -)
        for (int i = 0; i < NUM_LIMBS; i++) {
            t1[i] = VBLENDPERM(idx0, P12[i], P12[i + NUM_LIMBS]);
            t3[i] = VBLENDPERM(idx1, t0[i], t2[i]);
        }
        fp2_mul_4x2x1(t1, t1, t3); // ( Y1 * C, X2 * A_inv, Z2 * B, -)
        idx0 = VSET(4, 5, 12, 13, 0, 0, 0, 0); // (Z2 * B, x, -, -)
        for(int i = 0; i < NUM_LIMBS; i++) {
            t3[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        }
        int eqmask = fp2_is_equal_4x2x1(t1, t3);
        int ker_is_isotropic = ((eqmask & 0x2) >> 1) & eqmask;
        
        if (!ker_is_isotropic) { return false; }
    }

    // OA = hadamard(AVX_4x2x1(A, B, C, 0))
    idx0 = VSET(0, 1, 8, 10, 12, 14, 0, 0);
    // OA_dual_inv = AVX_4x2x1(A_inv, C, B, 0)
    idx1 = VSET(2, 3, 12, 14, 8, 10, 0, 0);
    mask = 0x3f;
    // J_prime = AVX_4x2x1(y, y, x, x)
    idx2 = VSET(6, 7, 6, 7, 4, 5, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res_OA[i] = VBLENDPERMMASKZ(mask, t0[i], t2[i], idx0);
        res_OA_dual_inv[i] = VBLENDPERMMASKZ(mask, t0[i], t2[i], idx1);
        res_J_prime[i] = VPERM512(idx2, t0[i]);
    }
    theta_had_single(res_OA, res_OA);

    return true;
}

// TODO: Haven't investigated the possibility of micro-optimizations.
void gluing_eval_pair(
    __m512i* res,
    __m512i* A_12,  __m512i* xP, __m512i* yP, __m512i* zP,
    __m512i* xT_8, __m512i* yT_8, __m512i* zT_8,
    __m512i* J_prime, __m512i* matrix01, __m512i* matrix23
)
{
    __m512i u[NUM_LIMBS], v[NUM_LIMBS], w[NUM_LIMBS], t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    __m512i idx0_re, idx0_im, idx1_re, idx1_im;
    __mmask8 mask;

    // Add components
    // u1, v1, w1 = add_components(E1, P1, T1_8)
    // u3, v3, w3 = add_components(E1, Q1, T1_8)
    // u2, v2, w2 = add_components(E2, P2, T2_8)
    // u4, v4, w4 = add_components(E2, Q2, T2_8)
    ec_add_components_4x4x2x1(u, v, w, A_12, xP, yP, zP, xT_8, yT_8, zT_8);

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
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0_re, u[i], w[i]);
        t0[i + NUM_LIMBS] = VBLENDPERM(idx0_im, u[i], w[i]);
        t1[i] = VBLENDPERM(idx1_re, u[i], w[i]);
        t1[i + NUM_LIMBS] = VBLENDPERM(idx1_im, u[i], w[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    // t1 = (v1u2, v3u4, u1v2, u3v4, v1w2, v3w4, w1v2, w3v4)
    idx0_re = VSET(0, 2, 4, 6, 0, 2, 4, 6); // (v1, v3, v2, v4, v1, v3, v2, v4)
    idx0_im = VSET(1, 3, 5, 7, 1, 3, 5, 7);
    idx1_re = VSET(4, 6, 0, 2, 12, 14, 8, 10); // (u2, u4, u1, u3, w2, w4, w1, w3)
    idx1_im = VSET(5, 7, 1, 3, 13, 15, 9, 11);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0_re, v[i]);
        t1[i + NUM_LIMBS] = VPERM512(idx0_im, v[i]);
        t2[i] = VBLENDPERM(idx1_re, u[i], w[i]);
        t2[i + NUM_LIMBS] = VBLENDPERM(idx1_im, u[i], w[i]);
    }
    fp2_mul_8x1x1(t1, t1, t2);

    // t2[0:7] = (v1v2, v3v4) in 2x4x1
    idx0_re = VSET(0, 0, 1, 1, 2, 2, 3, 3);
    idx1_re = VSET(4, 4, 5, 5, 6, 6, 7, 7);
    mask = 0x55;
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VPERM512MASKZ(mask, v[i], idx0_re);
        t2[i + NUM_LIMBS] = VPERM512MASKZ(mask, v[i], idx1_re);
    }
    fp2_mul_2x4x1(t2, t2, t2 + NUM_LIMBS);

    //t2[8:15] = (u1u2, u3u4, v1u2, v3u4)
    idx0_re = VSET(0, 0, 1, 1, 8, 8, 9, 9); // (u1u2, u3u4, v1u2, v3u4)
    idx0_im = VSET(0, 8, 2, 9, 4, 5, 6, 7); // for im of u1u2 & u3u4
    idx1_im = VSET(0, 1, 2, 3, 4, 8, 6, 9); // for im of v1u2 & v3u4
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i + NUM_LIMBS] = VBLENDPERM(idx0_re, t0[i], t1[i]);
        t2[i + NUM_LIMBS] = VBLENDPERM(idx0_im, t2[i + NUM_LIMBS], t0[i + NUM_LIMBS]);
        t2[i + NUM_LIMBS] = VBLENDPERM(idx1_im, t2[i + NUM_LIMBS], t1[i + NUM_LIMBS]);
    }
    // t2[0:7] = (v1v2, v3v4, u1v2, u3v4)
    idx0_re = VSET(0, 2, 4, 6, 10, 10, 11, 11); // (v1v2, v3v4, re(u1v2), re(u3v4))
    idx0_im = VSET(0, 1, 2, 3, 4, 10, 6, 11); // (v1v2, v3v4, u1v2, u3v4)
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx0_re, t2[i], t1[i]);
        t2[i] = VBLENDPERM(idx0_im, t2[i], t1[i + NUM_LIMBS]);
    }

    // t2[0:7] = (u1u2+v1v2, u3u4+v3v4, v1u2+u1v2, v3u4+u3v4)
    fp2_add_4x2x1(t2, t2, t2 + NUM_LIMBS);

    // U12 = (u1u2+v1v2, u3u4+v3v4, u1w2, u3w4, w1u2, w3u4, w1w2, w3w4)
    // V12 = (v1u2+u1v2, v3u4+u3v4, v1w2, v3w4, w1v2, w3v4, 0, 0)
    idx0_re = VSET(8, 10, 2, 3, 4, 5, 6, 7);
    idx0_im = VSET(9, 11, 2, 3, 4, 5, 6, 7);
    idx1_re = VSET(12, 14, 4, 5, 6, 7, 0, 0);
    idx1_im = VSET(13, 15, 4, 5, 6, 7, 0, 0);
    mask = 0x3f;
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0_re, t0[i], t2[i]);
        t0[i + NUM_LIMBS] = VBLENDPERM(idx0_im, t0[i + NUM_LIMBS], t2[i]);
        t1[i] = VBLENDPERMMASKZ(mask, t1[i], t2[i], idx1_re);
        t1[i + NUM_LIMBS] = VBLENDPERMMASKZ(mask, t1[i + NUM_LIMBS], t2[i], idx1_im);
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

// P1_P2 is (X1, Z1, X2, Z2) in 4x2x1
void gluing_eval_product_to_theta(
    __m512i* res,
    __m512i* P1_P2, __m512i* matrix01, __m512i* matrix23
)
{
    __m512i t0[NUM_LIMBS], idx0, idx1;

    // res = (X1, X1, Z1, Z1)
    // t0 = (X2, Z2, X2, Z2)
    idx0 = VSET(0, 1, 0, 1, 2, 3, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res[i] = VPERM512(idx0, P1_P2[i]);
        t0[i] = VPERM512(idx1, P1_P2[i]);
    }

    fp2_mul_4x2x1(res, res, t0);
    gluing_apply_matrix(res, matrix01, matrix23, res);
}

void gluing_eval_product_to_theta_unvec(
    __m512i* res,
    ec_point_t* P1, ec_point_t* P2, __m512i* matrix01, __m512i* matrix23
)
{
    fp2_t unvec[4];
    __m512i t[NUM_LIMBS];
    unvec[0] = P1->x; unvec[1] = P1->z; unvec[2] = P2->x; unvec[3] = P2->z;
    fp2_unvectorized_to_4x2x1(t, unvec);

    gluing_eval_product_to_theta(res, t, matrix01, matrix23);
}

// P1_P2 and Q1_Q2 are pairs of points in 4x2x1 as (X1, Z1, X2, Z2)
void gluing_eval_product_to_theta_pair(
    __m512i* res,
    __m512i* P1_P2, __m512i* Q1_Q2, 
    __m512i* matrix01, __m512i* matrix23
)
{
    __m512i t0[2*NUM_LIMBS], idx0, idx1, idx2, idx3;

    // res = (X1, X3, X1, X3, Z1, Z3, Z1, Z3)
    // t0 = (X2, X4, Z2, Z4, X2, X4, Z2, Z4)
    idx0 = VSET(0, 8, 0, 8, 2, 10, 2, 10);
    idx1 = VSET(1, 9, 1, 9, 3, 11, 3, 11);
    idx2 = VSET(4, 12, 6, 14, 4, 12, 6, 14);
    idx3 = VSET(5, 13, 7, 15, 5, 13, 7, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res[i] = VBLENDPERM(idx0, P1_P2[i], Q1_Q2[i]);
        res[i + NUM_LIMBS] = VBLENDPERM(idx1, P1_P2[i], Q1_Q2[i]);
        t0[i] = VBLENDPERM(idx2, P1_P2[i], Q1_Q2[i]);
        t0[i + NUM_LIMBS] = VBLENDPERM(idx3, P1_P2[i], Q1_Q2[i]);
    }
    
    fp2_mul_8x1x1(res, res, t0);
    gluing_apply_matrix_pair(res, matrix01, matrix23, res);
}


void gluing_eval_product_to_theta_pair_unvec(
    __m512i* res,
    ec_point_t* P1, ec_point_t* P2, ec_point_t* Q1, ec_point_t* Q2, 
    __m512i* matrix01, __m512i* matrix23
)
{
    fp2_t unvec[4];
    __m512i P1_P2[NUM_LIMBS], Q1_Q2[NUM_LIMBS];

    unvec[0] = P1->x; 
    unvec[1] = P1->z; 
    unvec[2] = P2->x; 
    unvec[3] = P2->z; 
    fp2_unvectorized_to_4x2x1(P1_P2, unvec);

    unvec[0] = Q1->x;
    unvec[1] = Q1->z;
    unvec[2] = Q2->x;
    unvec[3] = Q2->z;
    fp2_unvectorized_to_4x2x1(Q1_Q2, unvec);

    gluing_eval_product_to_theta_pair(res, P1_P2, Q1_Q2, matrix01, matrix23);
}

bool gluing_eval_special(
    __m512i* res,
    __m512i* P1_P2, __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23, const bool verify
)
{
    // Check that the points are of the form (P1, 0) or (0, P2)
    if (verify) {
        __mmask8 nzmask = ~fp2_is_zero_4x2x1(P1_P2);
        if (nzmask & 0xa != 0xa) { // P1.z != 0 and P2.z != 0
            return false;
        }
    }
    gluing_eval_product_to_theta(res, P1_P2, matrix01, matrix23);
    fp2_sqr_4x2x1(res, res);
    theta_had_single(res, res);

    // check that w = 0
    if (verify) {
        if (fp2_is_zero_4x2x1(res) & 0x8 != 0x8) { return false; }
    }
    fp2_mul_4x2x1(res, res, OA_inv);
    theta_had_single(res, res);

    return true;
}

bool gluing_eval_special_unvec(
    __m512i* res,
    theta_couple_point_t* P12, __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23, const bool verify
)
{
    fp2_t unvec[4];
    __m512i t[NUM_LIMBS];
    unvec[0] = P12->P1.x; unvec[1] = P12->P1.z; unvec[2] = P12->P2.x; unvec[3] = P12->P2.z;
    fp2_unvectorized_to_4x2x1(t, unvec);

    return gluing_eval_special(res, t, OA_inv, matrix01, matrix23, verify);
}

bool gluing_eval_special_pair(
    __m512i* res,
    __m512i* P1_P2, __m512i* Q1_Q2,
    __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23, const bool verify
)
{
    // check that pair of points are of the form (P1, 0) or (0, P2)
    if (verify) {
        __mmask8 nzmask = ~fp2_is_zero_4x2x1(P1_P2);
        if (nzmask & 0xa != 0xa) { // P1.z != 0 and P2.z != 0
            return false;
        }
        nzmask = ~fp2_is_zero_4x2x1(Q1_Q2);
        if (nzmask & 0xa != 0xa) { // Q1.z != 0 and Q2.z != 0
            return false;
        }
    }
    gluing_eval_product_to_theta_pair(res, P1_P2, Q1_Q2, matrix01, matrix23);
    fp2_sqr_8x1x1(res, res);
    theta_had_pair(res, res);

    // check that w1 = 0 and w2 = 0:
    if (verify) {
        if (fp2_is_zero_8x1x1(res) & 0xc0 != 0xc0) { return false; }
    }
    theta_mul_pair(res, res, OA_inv);
    theta_had_pair(res, res);

    return true;
}

bool gluing_eval_special_pair_unvec(
    __m512i* res,
    theta_couple_point_t* P12, theta_couple_point_t* Q12,
    __m512i* OA_inv, __m512i* matrix01, __m512i* matrix23, const bool verify
)
{
    fp2_t unvec[4];
    __m512i P1_P2[NUM_LIMBS], Q1_Q2[NUM_LIMBS];

    unvec[0] = P12->P1.x; 
    unvec[1] = P12->P1.z; 
    unvec[2] = P12->P2.x; 
    unvec[3] = P12->P2.z; 
    fp2_unvectorized_to_4x2x1(P1_P2, unvec);

    unvec[0] = Q12->P1.x;
    unvec[1] = Q12->P1.z;
    unvec[2] = Q12->P2.x;
    unvec[3] = Q12->P2.z;
    fp2_unvectorized_to_4x2x1(Q1_Q2, unvec);

    return gluing_eval_special_pair(res, P1_P2, Q1_Q2, OA_inv, matrix01, matrix23, verify);
}

bool gluing_eval_special_list_pair(
    __m512i* res_pts, __m512i* pts, __m512i* inv, 
    __m512i* matrix01, __m512i* matrix23, const int num_pts, const bool verify
)
{
    bool success = true;
    bool num_pts_is_odd = num_pts & 0x1;
    int num_pair_pts = num_pts / 2;

    for (int i = 0; i < num_pair_pts; i++) {
        success &= gluing_eval_special_pair(
            res_pts + i*(2*NUM_LIMBS), pts + (2*i)*NUM_LIMBS, pts + (2*i+1)*NUM_LIMBS, 
            inv, matrix01, matrix23, verify
        );
    }
    if (num_pts_is_odd) {
        success &= gluing_eval_special(
            res_pts + (num_pts-1)*NUM_LIMBS, pts + (num_pts-1)*NUM_LIMBS,
            inv, matrix01, matrix23, verify
        );
    }
    return success;
}

bool gluing_eval_special_list_pair_unvec(
    __m512i* res_pts, theta_couple_point_t* pts, __m512i* inv, 
    __m512i* matrix01, __m512i* matrix23, const int num_pts, const bool verify
)
{
    bool success = true;
    bool num_pts_is_odd = num_pts & 0x1;
    int num_pair_pts = num_pts / 2;

    for (int i = 0; i < num_pair_pts; i++) {
        success &= gluing_eval_special_pair_unvec(
            res_pts + i*(2*NUM_LIMBS), pts + (2*i), pts + (2*i+1), 
            inv, matrix01, matrix23, verify
        );
    }
    if (num_pts_is_odd) {
        success &= gluing_eval_special_unvec(
            res_pts + (num_pts-1)*NUM_LIMBS, pts + (num_pts-1),
            inv, matrix01, matrix23, verify
        );
    }
    return success;
}