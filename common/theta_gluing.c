#include "ec.h"
#include "fp_consts.h"
#include "fp_unvec.h"
#include "fp2.h"
#include "theta_gluing.h"
#include "theta_middle.h" // TODO: move had, sqr, mul to theta.h? Since they are "core" functions.
#include "intrinsics_avx512.h"

#include "utils.h"
#include "sqi_test_extras.h"

// Branchless equality mask, all ones iff a == b. The positions below are
// derived from secret torsion points, so they are never branched on.
static inline uint32_t ct_eq_mask(uint32_t a, uint32_t b)
{
    uint32_t d = a ^ b;
    return ((d | (0u - d)) >> 31) - 1u;
}

// All ones iff bit k of m is set.
static inline uint32_t ct_bit_mask(__mmask8 m, int k)
{
    return 0u - (uint32_t)((m >> k) & 1);
}

/**
 * Compute the three symmetric elements above the 4-torsion basis of each curve
 * of the elliptic product.
 *
 * g1, g2, g3 are packed as (s00, s01) of the first curve followed by (s00, s01)
 * of the second one. The entry s10 equals +/- s01, and gs holds the only two
 * the caller needs: g1.s10 of the first curve and g2.s10 of the second.
 * deltas is (delta_1 * delta_2, delta_1, delta_2, -).
 */
void gluing_symmetric_elements_4x2x1(
    __m512i* g1, __m512i* g2, __m512i* g3, __m512i* gs, __m512i* deltas,
    __m512i* x_4, __m512i* z_4
)
{
    __m512i r[NUM_LIMBS], s[NUM_LIMBS], xz[NUM_LIMBS], dr[NUM_LIMBS], rd[NUM_LIMBS];
    __m512i ea[NUM_LIMBS], eb[NUM_LIMBS], ec[NUM_LIMBS];
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS], t3[NUM_LIMBS], idx0, idx1;
    __mmask8 zr, zs, zc, swap12, swap23, s10_sign;
    uint32_t pos[2], m12[2], m23[2];

    fp2_add_4x2x1(r, x_4, z_4); // (x1 + z1, x2 + z2) per curve
    fp2_sub_4x2x1(s, x_4, z_4); // (x1 - z1, x2 - z2) per curve

    // t0 = ((x1 + z1)(x2 - z2), (x1 - z1)(x2 + z2)) per curve
    idx0 = VSET(0, 1, 8, 9, 4, 5, 12, 13);
    idx1 = VSET(10, 11, 2, 3, 14, 15, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, r[i], s[i]);
        t1[i] = VBLENDPERM(idx1, r[i], s[i]);
    }
    fp2_mul_4x2x1(t0, t0, t1);

    // t3 = (iP + Q, iP - Q) per curve, which vanishes when it is the sum of the
    // two torsion points that lies above (0 : 1)
    fp2_muli_4x2x1(t1, t0);
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VPERM512(idx0, t1[i]);
        t3[i] = VPERM512(idx1, t0[i]);
    }
    fp2_mixaddsub_4x2x1(t3, t2, t3, 0x33);

    zr = fp2_is_zero_4x2x1(r);
    zs = fp2_is_zero_4x2x1(s);
    zc = fp2_is_zero_4x2x1(t3);

    // pos in 0..5 records which of T1, T2, T1 + T2 lies above (0 : 1)
    for (int c = 0; c < 2; c++) {
        uint32_t p = 255u;
        p ^= (p ^ 0u) & ct_bit_mask(zs, 2*c);
        p ^= (p ^ 1u) & ct_bit_mask(zr, 2*c);
        p ^= (p ^ 2u) & ct_bit_mask(zs, 2*c + 1);
        p ^= (p ^ 3u) & ct_bit_mask(zr, 2*c + 1);
        p ^= (p ^ 4u) & ct_bit_mask(zc, 2*c);
        p ^= (p ^ 5u) & ct_bit_mask(zc, 2*c + 1);
        pos[c] = p;
        m12[c] = ct_eq_mask(p >> 1, 0u);
        m23[c] = ct_eq_mask(p >> 1, 2u);
    }

    // Everything below is built from the torsion point that is not the one
    // above (0 : 1), so pick it out of the basis of each curve.
    uint32_t e0 = m12[0] & 1u;
    uint32_t e1 = 2u + (m12[1] & 1u);
    idx0 = VSET(2*e0, 2*e0 + 1, 8 + 2*e0, 9 + 2*e0, 2*e1, 2*e1 + 1, 8 + 2*e1, 9 + 2*e1);
    for (int i = 0; i < NUM_LIMBS; i++) {
        xz[i] = VBLENDPERM(idx0, x_4[i], z_4[i]);
    }

    // dr = (x^2 - z^2, x^2 + z^2) = (delta, r1) per curve
    fp2_sqr_4x2x1(t0, xz);
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VPERM512(idx1, t0[i]);
    }
    fp2_mixaddsub_4x2x1(dr, t1, t2, 0xcc);

    // t2 = (r2_1, r2_2, delta_1 * delta_2, -) with r2 = 2xz
    fp2_add_4x2x1(t0, xz, xz);
    idx0 = VSET(0, 1, 4, 5, 8, 9, 8, 9);
    idx1 = VSET(2, 3, 6, 7, 12, 13, 12, 13);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i], dr[i]);
        t2[i] = VBLENDPERM(idx1, xz[i], dr[i]);
    }
    fp2_mul_4x2x1(t2, t1, t2);

    idx0 = VSET(4, 5, 8, 9, 12, 13, 12, 13);
    idx1 = VSET(2, 3, 8, 9, 6, 7, 10, 11);
    for (int i = 0; i < NUM_LIMBS; i++) {
        deltas[i] = VBLENDPERM(idx0, t2[i], dr[i]);
        rd[i] = VBLENDPERM(idx1, dr[i], t2[i]); // (r1, r2) per curve
    }

    // The three elements before the conditional swaps:
    // ea = (-r1, r2), eb = (0, +/- delta), ec = (+/- i r2, -/+ i r1)
    fp2_neg_4x2x1(t0, rd);
    idx0 = VSET(0, 1, 10, 11, 4, 5, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        ea[i] = VBLENDPERM(idx0, t0[i], rd[i]);
    }

    fp2_neg_4x2x1(t0, dr);
    uint32_t a0 = 8u - 8u*(pos[0] & 1u);
    uint32_t a1 = 12u - 8u*(pos[1] & 1u);
    idx0 = VSET(0, 0, a0, a0 + 1, 0, 0, a1, a1 + 1);
    for (int i = 0; i < NUM_LIMBS; i++) {
        eb[i] = VBLENDPERMMASKZ(0xcc, dr[i], t0[i], idx0);
    }

    fp2_muli_4x2x1(t0, rd);
    fp2_neg_4x2x1(t1, t0);
    uint32_t f0 = (ct_eq_mask(pos[0], 0u) | ct_eq_mask(pos[0], 3u) | ct_eq_mask(pos[0], 4u)) & 1u;
    uint32_t f1 = (ct_eq_mask(pos[0], 1u) | ct_eq_mask(pos[0], 2u) | ct_eq_mask(pos[0], 5u)) & 1u;
    uint32_t f2 = (ct_eq_mask(pos[1], 0u) | ct_eq_mask(pos[1], 3u) | ct_eq_mask(pos[1], 4u)) & 1u;
    uint32_t f3 = (ct_eq_mask(pos[1], 1u) | ct_eq_mask(pos[1], 2u) | ct_eq_mask(pos[1], 5u)) & 1u;
    uint32_t b0 = 10u - 8u*f0, b1 = 8u - 8u*f1, b2 = 14u - 8u*f2, b3 = 12u - 8u*f3;
    idx0 = VSET(b0, b0 + 1, b1, b1 + 1, b2, b2 + 1, b3, b3 + 1);
    for (int i = 0; i < NUM_LIMBS; i++) {
        ec[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }

    // Order the three elements according to the positions
    swap12 = (__mmask8)((m12[0] & 0x3u) | (m12[1] & 0xcu));
    swap23 = (__mmask8)((m23[0] & 0x3u) | (m23[1] & 0xcu));
    fp2_select_4x2x1(g1, ea, eb, swap12);
    fp2_select_4x2x1(t0, eb, ec, swap23);
    fp2_select_4x2x1(g2, t0, ea, swap12);
    fp2_select_4x2x1(g3, ec, eb, swap23);

    // s10 is +s01 for the element built from delta and -s01 for the other two
    idx0 = VSET(2, 3, 14, 15, 2, 3, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, g1[i], g2[i]);
    }
    fp2_neg_4x2x1(t1, t0);
    s10_sign = (__mmask8)((m12[0] & 0x1u) | (~(m12[1] | m23[1]) & 0x2u));
    fp2_select_4x2x1(gs, t1, t0, s10_sign);
}

/**
 * Compute the change of basis matrix that transforms the standard product
 * theta structure into the theta structure induced by the 4-symplectic basis
 * <(P1_4, P2_4), (Q1_4, Q2_4)>, from the symmetric elements above that basis.
 *
 * Input: the 4-torsion basis (P1_4, Q1_4, P2_4, Q2_4) in 4x2x1.
 * Returns: the 4x4 change of basis matrix as two 8x1x1 vectors.
 */
void gluing_change_of_basis_4x2x1(
    __m512i* res_matrix01, __m512i* res_matrix23, __m512i* x_4, __m512i* z_4
)
{
    __m512i g1[NUM_LIMBS], g2[NUM_LIMBS], g3[NUM_LIMBS], gs[NUM_LIMBS], deltas[NUM_LIMBS];
    __m512i l1[NUM_LIMBS], l2[NUM_LIMBS], a1[NUM_LIMBS], a2[NUM_LIMBS], a4[NUM_LIMBS];
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS], t3[NUM_LIMBS];
    __m512i row0[NUM_LIMBS], row1[NUM_LIMBS], row2[NUM_LIMBS], row3[NUM_LIMBS];
    __m512i idxl, idxr, idx0, idx1, idx2, idx3;

    gluing_symmetric_elements_4x2x1(g1, g2, g3, gs, deltas, x_4, z_4);

    // First row: the pairwise products of the symmetric elements of both curves
    idxl = VSET(0, 1, 0, 1, 2, 3, 2, 3);
    idxr = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        l1[i] = VPERM512(idxl, g1[i]);
        l2[i] = VPERM512(idxl, g2[i]);
        a1[i] = VPERM512(idxr, g2[i]);
        t0[i] = VPERM512(idxr, g1[i]);
    }
    fp2_mulx2_4x2x1(t0, t1, l1, t0, l2, a1);
    fp2_add_4x2x1(row0, t0, t1);

    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idxl, g3[i]);
        t1[i] = VPERM512(idxr, g3[i]);
    }
    fp2_mul_4x2x1(t0, t0, t1);
    fp2_sub_4x2x1(row0, row0, t0);

    // The first entry also picks up delta_1 * delta_2
    idx0 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512MASKZ(0x3, deltas[i], idx0);
    }
    fp2_add_4x2x1(row0, row0, t0);

    // Second and third rows: act on the first one by g2 of the second curve
    // and by g1 of the first curve respectively
    idx0 = VSET(2, 3, 12, 13, 2, 3, 12, 13);
    idx1 = VSET(0, 1, 0, 1, 8, 9, 8, 9);
    idx2 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx3 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < NUM_LIMBS; i++) {
        a2[i] = VBLENDPERM(idx0, gs[i], g2[i]);
        a4[i] = VBLENDPERM(idx1, gs[i], g1[i]);
        t0[i] = VPERM512(idx2, row0[i]);
        t1[i] = VPERM512(idx3, row0[i]);
    }
    fp2_mulx2_4x2x1(t0, t1, a1, t0, l1, t1);

    idx0 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VPERM512(idx0, row0[i]);
        t3[i] = VPERM512(idx1, row0[i]);
    }
    fp2_mulx2_4x2x1(t2, t3, a2, t2, a4, t3);
    fp2_mixaddsub_4x2x1(row1, t0, t2, 0x33);
    fp2_mixaddsub_4x2x1(row2, t1, t3, 0x0f);

    // Fourth row: act on the second one by g1 of the first curve
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx3, row1[i]);
        t1[i] = VPERM512(idx1, row1[i]);
    }
    fp2_mulx2_4x2x1(t0, t1, l1, t0, a4, t1);
    fp2_mixaddsub_4x2x1(row3, t0, t1, 0x0f);

    // Realign the projective factors of the first three rows
    idx0 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    idx1 = VSET(2, 3, 2, 3, 2, 3, 2, 3);
    idx2 = VSET(4, 5, 4, 5, 4, 5, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, deltas[i]);
        t1[i] = VPERM512(idx1, deltas[i]);
        t2[i] = VPERM512(idx2, deltas[i]);
    }
    fp2_mulx2_4x2x1(row0, row1, row0, t0, row1, t1);
    fp2_mul_4x2x1(row2, row2, t2);

    fp2_4x2x1_to_8x1x1(res_matrix01, row0, row1);
    fp2_4x2x1_to_8x1x1(res_matrix23, row2, row3);
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
    __m512i x_8_[NUM_LIMBS], z_8_[NUM_LIMBS];

    // X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)
    ec_jac_to_xz_4x4x2x1(x_8_, z_8_, x_8, z_8);

    return gluing_codomain_x_only(
        res_OA, res_OA_dual_inv, res_J_prime, NULL, res_matrix01, res_matrix23,
        A24_12, x_8_, z_8_, verify
    );
}

// As the above, for a kernel that is already x-only. res_J_pair may be NULL,
// otherwise it receives the inverse dual images of both kernel points in
// 8x1x1, which is what the barycentric evaluation scales by.
bool gluing_codomain_x_only(
    __m512i* res_OA, __m512i* res_OA_dual_inv, __m512i* res_J_prime, __m512i* res_J_pair,
    __m512i* res_matrix01, __m512i* res_matrix23,
    __m512i* A24_12, __m512i* x_8_, __m512i* z_8_, const bool verify
)
{
    __m512i x_4[NUM_LIMBS], z_4[NUM_LIMBS], P12[2*NUM_LIMBS];
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS], t3[NUM_LIMBS], idx0, idx1, idx2;
    __mmask8 mask;

    // Double points to recover 4-torsion for change of basis
    ec_double_4x4x2x1(x_4, z_4, A24_12, x_8_, z_8_);

    // Compute the change of basis for our theta structure
    gluing_change_of_basis_4x2x1(res_matrix01, res_matrix23, x_4, z_4);

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

    if (res_J_pair != NULL) {
        // The image of the second kernel point is (D : B : D : B) up to the
        // factor Z2 shared by both, one multiplication as B is recomputed.
        idx0 = VSET(0, 8, 2, 10, 0, 8, 2, 10); // (X1, Y1, X1, Y1)
        idx1 = VSET(3, 11, 1, 9, 3, 11, 1, 9); // (Y2, X2, Y2, X2)
        for (int i = 0; i < NUM_LIMBS; i++) {
            t1[i] = VBLENDPERM(idx0, P12[i], P12[i + NUM_LIMBS]);
            t3[i] = VBLENDPERM(idx1, P12[i], P12[i + NUM_LIMBS]);
        }
        fp2_mul_4x2x1(t1, t1, t3);
        fp2_4x2x1_to_8x1x1(res_J_pair, res_J_prime, t1);
    }

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
    __m512i u[NUM_LIMBS], v[NUM_LIMBS], w[NUM_LIMBS], t0[2*NUM_LIMBS];

    // Add components
    // u1, v1, w1 = add_components(E1, P1, T1_8)
    // u3, v3, w3 = add_components(E1, Q1, T1_8)
    // u2, v2, w2 = add_components(E2, P2, T2_8)
    // u4, v4, w4 = add_components(E2, Q2, T2_8)
    ec_add_components_4x4x2x1(u, v, w, A_12, xP, yP, zP, xT_8, yT_8, zT_8);

    // Both points are translated by the same eight torsion point
    fp2_4x2x1_to_8x1x1(t0, J_prime, J_prime);

    gluing_eval_components_pair(res, u, v, w, t0, matrix01, matrix23, true);
}

// Same, from the ladder state P = ([m]G, [m-1]G) and the sum Dp and difference
// Dm of the coordinates of the generators, which are the translation points.
void gluing_eval_pair_bary(
    __m512i* res,
    __m512i* xe, __m512i* ze, __m512i* xo, __m512i* zo, __m512i* Dp, __m512i* Dm,
    __m512i* J_pair, __m512i* matrix01, __m512i* matrix23, const bool last_hadamard
)
{
    __m512i u[NUM_LIMBS], v[NUM_LIMBS], w[NUM_LIMBS];

    ec_bary_components_4x4x2x1(u, v, w, xe, ze, xo, zo, Dp, Dm);

    gluing_eval_components_pair(res, u, v, w, J_pair, matrix01, matrix23, last_hadamard);
}

// Evaluation of a pair of points from their addition components, whatever
// representation these were computed from.
void gluing_eval_components_pair(
    __m512i* res, __m512i* u, __m512i* v, __m512i* w,
    __m512i* J_pair, __m512i* matrix01, __m512i* matrix23, const bool last_hadamard
)
{
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    __m512i idx0_re, idx0_im, idx1_re, idx1_im;
    __mmask8 mask;

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

    // Scaling by the inverse dual image of the translation point
    fp2_mul_8x1x1(res, res, J_pair);

    // Final Hadamard
    if (last_hadamard) {
        theta_had_pair(res, res);
    }
}

// Projective inverse (yzw : xzw : xyw : xyz) of a pair of theta points.
void theta_invert_pair(__m512i* res, __m512i* P)
{
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], idx0, idx1;

    // t0 = (xy, zw) per point, half the lanes are wasted
    idx0 = VSET(0, 1, 4, 5, 0, 1, 4, 5); // (x, z) per point
    idx1 = VSET(2, 3, 6, 7, 2, 3, 6, 7); // (y, w) per point
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    idx0 = VSET(2, 3, 0, 1, 6, 7, 4, 5); // (y, x, w, z) per point
    idx1 = VSET(2, 3, 2, 3, 0, 1, 0, 1); // (zw, zw, xy, xy) per point
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, P[i]);
        t0[i] = VPERM512(idx1, t0[i]);
    }
    fp2_mul_8x1x1(res, t0, t1);
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
        if ((nzmask & 0xa) != 0xa) { // P1.z != 0 and P2.z != 0
            return false;
        }
    }
    gluing_eval_product_to_theta(res, P1_P2, matrix01, matrix23);
    fp2_sqr_4x2x1(res, res);
    theta_had_single(res, res);

    // check that w = 0
    if (verify) {
        if ((fp2_is_zero_4x2x1(res) & 0x8) != 0x8) { return false; }
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
        if ((nzmask & 0xa) != 0xa) { // P1.z != 0 and P2.z != 0
            return false;
        }
        nzmask = ~fp2_is_zero_4x2x1(Q1_Q2);
        if ((nzmask & 0xa) != 0xa) { // Q1.z != 0 and Q2.z != 0
            return false;
        }
    }
    gluing_eval_product_to_theta_pair(res, P1_P2, Q1_Q2, matrix01, matrix23);
    fp2_sqr_8x1x1(res, res);
    theta_had_pair(res, res);

    // check that w1 = 0 and w2 = 0:
    if (verify) {
        if ((fp2_is_zero_8x1x1(res) & 0xc0) != 0xc0) { return false; }
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