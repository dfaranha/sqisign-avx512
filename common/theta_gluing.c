#include "ec.h"
#include "fp2_lvl3.h"
#include "theta_gluing.h"
#include "theta_middle.h" // TODO: move had, sqr, mul to theta.h? Since they are "core" functions.

#include <assert.h>
#include "utils.h"
#include "sqi_test_extras.h"

//TODO improve implementation, remove the need to unvectorize.
void gluing_compute_matrix_4x2x1(
    __m512i* g1, __m512i* h1, __m512i* g2, __m512i* h2, __m512i* pi,
    __m512i* x_4, __m512i* z_4, __m512i* u, __m512i* w, __m512i* det
)
{
    // TODO: Copy comment on how this comp is performed.

    __m512i z_sqr[8], a[8], b[8], c[8], d[8];
    fp2_t a_un[4], b_un[4], c_un[4], d_un[4], out_un[4];

    fp2_sqr_4x2x1(z_sqr, z_4);
    fp2_mul_4x2x1(d, u, z_sqr);
    fp2_mul_4x2x1(b, w, z_sqr);
    fp2_neg_4x2x1(b, b);
    fp2_mul_4x2x1(c, u, z_4);
    fp2_sub_4x2x1(c, c, det);
    fp2_mul_4x2x1(c, c, x_4);
    fp2_neg_4x2x1(a, d);
    // Observe that this doesn't depend on the previous computation:
    fp2_mul_4x2x1(pi, z_4, det);

    fp2_4x2x1_to_unvectorized(a_un, a);
    fp2_4x2x1_to_unvectorized(b_un, b);
    fp2_4x2x1_to_unvectorized(c_un, c);
    fp2_4x2x1_to_unvectorized(d_un, d);

    out_un[0] = a_un[0]; out_un[1] = b_un[0]; out_un[2] = c_un[0]; out_un[3] = d_un[0];
    fp2_unvectorized_to_4x2x1(g1, out_un);
    out_un[0] = a_un[1]; out_un[1] = b_un[1]; out_un[2] = c_un[1]; out_un[3] = d_un[1];
    fp2_unvectorized_to_4x2x1(h1, out_un);
    out_un[0] = a_un[2]; out_un[1] = b_un[2]; out_un[2] = c_un[2]; out_un[3] = d_un[2];
    fp2_unvectorized_to_4x2x1(g2, out_un);
    out_un[0] = a_un[3]; out_un[1] = b_un[3]; out_un[2] = c_un[3]; out_un[3] = d_un[3];
    fp2_unvectorized_to_4x2x1(h2, out_un);
}

void gluing_action_by_translation_4x2x1(
    __m512i* g1, __m512i* g2, __m512i* h1, __m512i* h2, __m512i* pi,
    __m512i* A24, __m512i* x_4, __m512i* z_4
)
{
    __m512i u[8], w[8], t0[8], t1[8];

    ec_double_4x2x1(u, w, A24, x_4, z_4);
    fp2_mul_4x2x1(t0, w, x_4);
    fp2_mul_4x2x1(t1, u, z_4);
    fp2_sub_4x2x1(t0, t0, t1); // det

    gluing_compute_matrix_4x2x1(g1, h1, g2, h2, pi, x_4, z_4, u, w, t0);
}

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
    __m512i t[8], t0[8], t1[8], t2[8], t3[8], t4[8];
    __m512i a0[8], a1[8], a2[8], a3[8];
    fp2_t g1_unvec[4], h1_unvec[4], g2_unvec[4], h2_unvec[4], pi_unvec[4];
    fp2_t d_prod0[4], d_prod1[4], a0_unvec[4], a1_unvec[4], a2_unvec[4], a3_unvec[4];
    fp2_t t_unvec[4], t_unvec0[4], t_unvec1[4], res_unvec[8];

    gluing_action_by_translation_4x2x1(g1, g2, h1, h2, pi, A24, x_4, z_4);

    // Extract out the projective factors we need to remove all inversions
    // dg1, dh1, dg2, dh2 = pi
    fp2_4x2x1_to_unvectorized(pi_unvec, pi);

    // Marius: I'll just unvectorize the rest immediately:
    fp2_4x2x1_to_unvectorized(g1_unvec, g1);
    fp2_4x2x1_to_unvectorized(h1_unvec, h1);
    fp2_4x2x1_to_unvectorized(g2_unvec, g2);
    fp2_4x2x1_to_unvectorized(h2_unvec, h2);

    // Precomputation we need the values for various parts, which doesn't
    // feel all that nice... Currently consumes 2 avx_pairwise_mul for 6 mul
    // dg1 * dh1
    // dg2 * dh2
    // dg1 * dg2
    // dh1 * dh2
    // dg1 * dh2
    // dg1 * dh1 * dg2 * dh2

    // T0 = AVX_4x2x1(dg1, dg2, dg1, dh1)
    t_unvec0[0] = pi_unvec[0]; t_unvec0[1] = pi_unvec[2]; t_unvec0[2] = pi_unvec[0]; t_unvec0[3] = pi_unvec[1];
    // T1 = AVX_4x2x1(dh1, dh2, dg2, dh2)
    t_unvec1[0] = pi_unvec[1]; t_unvec1[1] = pi_unvec[3]; t_unvec1[2] = pi_unvec[2]; t_unvec1[3] = pi_unvec[3];
    fp2_unvectorized_to_4x2x1(t0, t_unvec0);
    fp2_unvectorized_to_4x2x1(t1, t_unvec1);
    
    // dg1h1, dg2h2, dg1g2, dh1h2 = avx_pairwise_mul(T0, T1)
    fp2_mul_4x2x1(t0, t0, t1);
    fp2_4x2x1_to_unvectorized(d_prod0, t0);

    // TODO: redundant computation here!
    // T0 = AVX_4x2x1(dg1, dg1h1, 1, 1)
    t_unvec0[0] = pi_unvec[0]; t_unvec0[1] = d_prod0[0]; fp2_set_one(t_unvec0+2); fp2_set_one(t_unvec0+3);
    // T1 = AVX_4x2x1(dh2, dg2h2, 1, 1)
    t_unvec1[0] = pi_unvec[3]; t_unvec1[1] = d_prod0[1]; fp2_set_one(t_unvec1+2); fp2_set_one(t_unvec1+3);
    fp2_unvectorized_to_4x2x1(t0, t_unvec0);
    fp2_unvectorized_to_4x2x1(t1, t_unvec1);

    // dg1h2, dg1g2h1h2, _, _ = avx_pairwise_mul(T0, T1)
    fp2_mul_4x2x1(t0, t0, t1);
    fp2_4x2x1_to_unvectorized(d_prod1, t0);

    // Compute the numerator of ti as standard
    // T0 = AVX_4x2x1(g1[0], g1[2], g2[0], g2[2])
    t_unvec0[0] = g1_unvec[0]; t_unvec0[1] = g1_unvec[2]; t_unvec0[2] = g2_unvec[0]; t_unvec0[3] = g2_unvec[2];
    // T1 = AVX_4x2x1(h1[0], h1[0], h2[0], h2[0])
    t_unvec1[0] = h1_unvec[0]; t_unvec1[1] = h1_unvec[0]; t_unvec1[2] = h2_unvec[0]; t_unvec1[3] = h2_unvec[0];
    fp2_unvectorized_to_4x2x1(t0, t_unvec0);
    fp2_unvectorized_to_4x2x1(t1, t_unvec1);
    fp2_mul_4x2x1(t, t0, t1);
    // T0 = AVX_4x2x1(g1[1], g1[3], g2[1], g2[3])
    t_unvec0[0] = g1_unvec[1]; t_unvec0[1] = g1_unvec[3]; t_unvec0[2] = g2_unvec[1]; t_unvec0[3] = g2_unvec[3];
    // T1 = AVX_4x2x1(h1[2], h1[2], h2[2], h2[2])
    t_unvec1[0] = h1_unvec[2]; t_unvec1[1] = h1_unvec[2]; t_unvec1[2] = h2_unvec[2]; t_unvec1[3] = h2_unvec[2];
    fp2_unvectorized_to_4x2x1(t0, t_unvec0);
    fp2_unvectorized_to_4x2x1(t1, t_unvec1);
    fp2_mul_4x2x1(t2, t0, t1);
    fp2_add_4x2x1(t, t, t2);

    // Ensure correct projective factor for ti
    // T2 = AVX_4x2x1(dg2h2, dg2h2, dg1h1, dg1h1)
    t_unvec0[0] = d_prod0[1]; t_unvec0[1] = d_prod0[1]; t_unvec0[2] = d_prod0[0]; t_unvec0[3] = d_prod0[0];
    fp2_unvectorized_to_4x2x1(t2, t_unvec0);
    fp2_mul_4x2x1(t, t, t2);
    // t1, t2, t3, t4 = T
    fp2_4x2x1_to_unvectorized(t_unvec, t);

    // Compute A0 = a, b, c, d
    // We divide out a projective factor of 1 / (dg1 * dh1 * dg2 * dh2)^2
    // T0 = AVX_4x2x1(g1[0], g1[0], g1[2], g1[2])
    t_unvec0[0] = g1_unvec[0]; t_unvec0[1] = g1_unvec[0]; t_unvec0[2] = g1_unvec[2]; t_unvec0[3] = g1_unvec[2];
    // T1 = AVX_4x2x1(g2[0], g2[2], g2[0], g2[2])
    t_unvec1[0] = g2_unvec[0]; t_unvec1[1] = g2_unvec[2]; t_unvec1[2] = g2_unvec[0]; t_unvec1[3] = g2_unvec[2];
    fp2_unvectorized_to_4x2x1(t0, t_unvec0);
    fp2_unvectorized_to_4x2x1(t1, t_unvec1);
    fp2_mul_4x2x1(a0, t0, t1);
    // T2 = AVX_4x2x1(dh1h2, dh1h2, dh1h2, dh1h2)
    t_unvec0[0] = d_prod0[3]; t_unvec0[1] = d_prod0[3]; t_unvec0[2] = d_prod0[3]; t_unvec0[3] = d_prod0[3];
    fp2_unvectorized_to_4x2x1(t2, t_unvec0);
    // TODO: every element of T2 is the same, can we use this?
    fp2_mul_4x2x1(a0, a0, t2);

    // T1 = AVX_4x2x1(h1[0], h1[0], h1[2], h1[2])
    t_unvec0[0] = h1_unvec[0]; t_unvec0[1] = h1_unvec[0]; t_unvec0[2] = h1_unvec[2]; t_unvec0[3] = h1_unvec[2];
    // T2 = AVX_4x2x1(h2[0], h2[2], h2[0], h2[2])
    t_unvec1[0] = h2_unvec[0]; t_unvec1[1] = h2_unvec[2]; t_unvec1[2] = h2_unvec[0]; t_unvec1[3] = h2_unvec[2];
    fp2_unvectorized_to_4x2x1(t1, t_unvec0); // Marius: why not t0, t1? A: See below "Recall: T0 = ..."
    fp2_unvectorized_to_4x2x1(t2, t_unvec1);
    fp2_mul_4x2x1(t3, t1, t2);
    // T1 = AVX_4x2x1(dg1g2, dg1g2, dg1g2, dg1g2)
    // TODO: every element of T1 is the same, can we use this?
    t_unvec0[0] = d_prod0[2]; t_unvec0[1] = d_prod0[2]; t_unvec0[2] = d_prod0[2]; t_unvec0[3] = d_prod0[2];
    fp2_unvectorized_to_4x2x1(t1, t_unvec0);
    fp2_mul_4x2x1(t3, t1, t3);
    fp2_add_4x2x1(a0, a0, t3);
    
    // TODO: this could be optimised?
    // T1 = AVX_4x2x1(dg1g2h1h2, 0, 0, 0)
    t_unvec0[0] = d_prod1[1]; fp2_set_zero(t_unvec0+1); fp2_set_zero(t_unvec0+2); fp2_set_zero(t_unvec0+3); 
    fp2_unvectorized_to_4x2x1(t1, t_unvec0);
    fp2_add_4x2x1(a0, a0, t1);
    
    // T1 = AVX_4x2x1(dg1g2h1h2, dg1g2h1h2, dg1g2h1h2, dg1g2h1h2)
    // TODO: every element of T1 is the same, can we use this?
    t_unvec0[0] = d_prod1[1]; t_unvec0[1] = d_prod1[1]; t_unvec0[2] = d_prod1[1]; t_unvec0[3] = d_prod1[1];
    fp2_unvectorized_to_4x2x1(t1, t_unvec0);
    fp2_mul_4x2x1(a0, a0, t1);
    
    // T1 = AVX_4x2x1(t1, t1, t2, t2)
    t_unvec0[0] = t_unvec[0]; t_unvec0[1] = t_unvec[0]; t_unvec0[2] = t_unvec[1]; t_unvec0[3] = t_unvec[1];
    // T3 = AVX_4x2x1(t3, t4, t3, t4)
    t_unvec1[0] = t_unvec[2]; t_unvec1[1] = t_unvec[3]; t_unvec1[2] = t_unvec[2]; t_unvec1[3] = t_unvec[3];
    fp2_unvectorized_to_4x2x1(t1, t_unvec0);
    fp2_unvectorized_to_4x2x1(t3, t_unvec1);
    fp2_mul_4x2x1(t4, t1, t3);
    fp2_add_4x2x1(a0, a0, t4);
    // a, b, c, d = A0
    fp2_4x2x1_to_unvectorized(a0_unvec, a0);

    // Compute A1 = a1, b1, c1, d1
    // We divide out a projective factor of 1 / (dg1 * dh1 * dg2 * dh2)^2 * dh2
    // A1 = AVX_4x2x1(a, a, c, c)
    t_unvec0[0] = a0_unvec[0]; t_unvec0[1] = a0_unvec[0]; t_unvec0[2] = a0_unvec[2]; t_unvec0[3] = a0_unvec[2];
    fp2_unvectorized_to_4x2x1(a1, t_unvec0);
    fp2_mul_4x2x1(a1, a1, t2); // Recall: T2 = AVX_4x2x1(h2[0], h2[2], h2[0], h2[2])
    // T1 = AVX_4x2x1(b, b, d, d)
    t_unvec0[0] = a0_unvec[1]; t_unvec0[1] = a0_unvec[1]; t_unvec0[2] = a0_unvec[3]; t_unvec0[3] = a0_unvec[3];
    // T2 = AVX_4x2x1(h2[1], h2[3], h2[1], h2[3])
    t_unvec1[0] = h2_unvec[1]; t_unvec1[1] = h2_unvec[3]; t_unvec1[2] = h2_unvec[1]; t_unvec1[3] = h2_unvec[3];
    fp2_unvectorized_to_4x2x1(t1, t_unvec0);
    fp2_unvectorized_to_4x2x1(t2, t_unvec1);
    fp2_mul_4x2x1(t3, t1, t2);
    fp2_add_4x2x1(a1, a1, t3);
    // a1, b1, c1, d1 = A1
    fp2_4x2x1_to_unvectorized(a1_unvec, a1);

    // Compute A2 = a2, b2, c2, d2
    // We divide out a projective factor of 1 / (dg1 * dh1 * dg2 * dh2)^2 * dg1
    // A2 = AVX_4x2x1(a, b, a, b)
    t_unvec0[0] = a0_unvec[0]; t_unvec0[1] = a0_unvec[1]; t_unvec0[2] = a0_unvec[0]; t_unvec0[3] = a0_unvec[1];
    fp2_unvectorized_to_4x2x1(a2, t_unvec0);
    fp2_mul_4x2x1(a2, a2, t0); // Recall: T0 = AVX_4x2x1(g1[0], g1[0], g1[2], g1[2])
    // T1 = AVX_4x2x1(g1[1], g1[1], g1[3], g1[3])
    t_unvec0[0] = g1_unvec[1]; t_unvec0[1] = g1_unvec[1]; t_unvec0[2] = g1_unvec[3]; t_unvec0[3] = g1_unvec[3];
    // T2 = AVX_4x2x1(c, d, c, d)
    t_unvec1[0] = a0_unvec[2]; t_unvec1[1] = a0_unvec[3]; t_unvec1[2] = a0_unvec[2]; t_unvec1[3] = a0_unvec[3];
    fp2_unvectorized_to_4x2x1(t1, t_unvec0);
    fp2_unvectorized_to_4x2x1(t2, t_unvec1);
    fp2_mul_4x2x1(t3, t1, t2);
    fp2_add_4x2x1(a2, a2, t3);
    // a2, b2, c2, d2 = A2
    fp2_4x2x1_to_unvectorized(a2_unvec, a2);

    // Compute A3 = a3, b3, c3, d3
    // We divide out a projective factor of 1 / (dg1 * dh1 * dg2 * dh2)^2 * dg1 * dh2
    // A3 = AVX_4x2x1(a1, b1, a1, b1)
    t_unvec0[0] = a1_unvec[0]; t_unvec0[1] = a1_unvec[1]; t_unvec0[2] = a1_unvec[0]; t_unvec0[3] = a1_unvec[1];
    fp2_unvectorized_to_4x2x1(a3, t_unvec0);
    fp2_mul_4x2x1(a3, a3, t0); // Recall: T0 = AVX_4x2x1(g1[0], g1[0], g1[2], g1[2])
    // T2 = AVX_4x2x1(c1, d1, c1, d1)
    t_unvec0[0] = a1_unvec[2]; t_unvec0[1] = a1_unvec[3]; t_unvec0[2] = a1_unvec[2]; t_unvec0[3] = a1_unvec[3];
    fp2_unvectorized_to_4x2x1(t2, t_unvec0);
    fp2_mul_4x2x1(t3, t1, t2); // Recall: T1 = AVX_4x2x1(g1[1], g1[1], g1[3], g1[3])
    fp2_add_4x2x1(a3, a3, t3);
    // a3, b3, c3, d3 = A3
    fp2_4x2x1_to_unvectorized(a3_unvec, a3);

    // We now need to make sure the entire matrix has a projective factor of
    // 1 / (dg1 * dh1 * dg2 * dh2)^2 * dg1 * dh2
    // TODO: every element of T0 is the same, can we use this?
    // T0 = AVX_4x2x1(dg1h2, dg1h2, dg1h2, dg1h2)
    t_unvec0[0] = d_prod1[0]; t_unvec0[1] = d_prod1[0]; t_unvec0[2] = d_prod1[0]; t_unvec0[3] = d_prod1[0];
    fp2_unvectorized_to_4x2x1(t0, t_unvec0);
    // a, b, c, d = avx_pairwise_mul(A0, T0)
    fp2_mul_4x2x1(a0, a0, t0);
    fp2_4x2x1_to_unvectorized(a0_unvec, a0);
    // T0 = AVX_4x2x1(dg1, dg1, dg1, dg1) Again all entries same
    t_unvec0[0] = pi_unvec[0]; t_unvec0[1] = pi_unvec[0]; t_unvec0[2] = pi_unvec[0]; t_unvec0[3] = pi_unvec[0];
    fp2_unvectorized_to_4x2x1(t0, t_unvec0);
    // a1, b1, c1, d1 = avx_pairwise_mul(A1, T0)
    fp2_mul_4x2x1(a1, a1, t0);
    fp2_4x2x1_to_unvectorized(a1_unvec, a1);
    // T0 = AVX_4x2x1(dh2, dh2, dh2, dh2)
    t_unvec0[0] = pi_unvec[3]; t_unvec0[1] = pi_unvec[3]; t_unvec0[2] = pi_unvec[3]; t_unvec0[3] = pi_unvec[3];
    fp2_unvectorized_to_4x2x1(t0, t_unvec0);
    // a2, b2, c2, d2 = avx_pairwise_mul(A2, T0)
    fp2_mul_4x2x1(a2, a2, t0);
    fp2_4x2x1_to_unvectorized(a2_unvec, a2);

    // Result (following the AVX interleaving pattern)
    // AVX_8x1x1(a0, a1, b0, b1, c0, c1, d0, d1)
    res_unvec[0] = a0_unvec[0]; res_unvec[1] = a1_unvec[0]; res_unvec[2] = a0_unvec[1]; res_unvec[3] = a1_unvec[1];
    res_unvec[4] = a0_unvec[2]; res_unvec[5] = a1_unvec[2]; res_unvec[6] = a0_unvec[3]; res_unvec[7] = a1_unvec[3]; 
    fp2_unvectorized_to_8x1x1(res_matrix01, res_unvec);
    // AVX_8x1x1(a2, a3, b2, b3, c2, c3, d2, d3)
    res_unvec[0] = a2_unvec[0]; res_unvec[1] = a3_unvec[0]; res_unvec[2] = a2_unvec[1]; res_unvec[3] = a3_unvec[1];
    res_unvec[4] = a2_unvec[2]; res_unvec[5] = a3_unvec[2]; res_unvec[6] = a2_unvec[3]; res_unvec[7] = a3_unvec[3]; 
    fp2_unvectorized_to_8x1x1(res_matrix23, res_unvec);
}

void gluing_internal_sum(__m512i* res_P, __m512i* P0P1, __m512i* P2P3)
{
    __m512i sum[8], t[8]; //TODO use res as sum
    fp2_t P0P1_unvec[8], P2P3_unvec[8], t_unvec[4];

    // Pa0, Pa1, Pb0, Pb1, Pc0, Pc1, Pd0, Pd1 = A
    fp2_8x1x1_to_unvectorized(P0P1_unvec, P0P1);
    // Pa2, Pa3, Pb2, Pb3, Pc2, Pc3, Pd2, Pd3 = B
    fp2_8x1x1_to_unvectorized(P2P3_unvec, P2P3);

    // Compute the sum of the elements for each of the 8x1x1 blobs
    // a = a0 + b0 + c0 + d0
    // b = a1 + b1 + c1 + d1
    // c = a2 + b2 + c2 + d2
    // d = a3 + b3 + c3 + d3

    // C = AVX_4x2x1(a0, a1, a2, a3)
    t_unvec[0] = P0P1_unvec[0]; t_unvec[1] = P0P1_unvec[1]; t_unvec[2] = P2P3_unvec[0]; t_unvec[3] = P2P3_unvec[1];
    fp2_unvectorized_to_4x2x1(sum, t_unvec);
    //tmp = AVX_4x2x1(b0, b1, b2, b3)
    t_unvec[0] = P0P1_unvec[2]; t_unvec[1] = P0P1_unvec[3]; t_unvec[2] = P2P3_unvec[2]; t_unvec[3] = P2P3_unvec[3];
    fp2_unvectorized_to_4x2x1(t, t_unvec);
    fp2_add_4x2x1(sum, sum, t);
    
    // tmp = AVX_4x2x1(c0, c1, c2, c3)
    t_unvec[0] = P0P1_unvec[4]; t_unvec[1] = P0P1_unvec[5]; t_unvec[2] = P2P3_unvec[4]; t_unvec[3] = P2P3_unvec[5];
    fp2_unvectorized_to_4x2x1(t, t_unvec);
    fp2_add_4x2x1(sum, sum, t);

    // tmp = AVX_4x2x1(d0, d1, d2, d3)
    t_unvec[0] = P0P1_unvec[6]; t_unvec[1] = P0P1_unvec[7]; t_unvec[2] = P2P3_unvec[6]; t_unvec[3] = P2P3_unvec[7];
    fp2_unvectorized_to_4x2x1(t, t_unvec);
    fp2_add_4x2x1(res_P, sum, t);
}

void gluing_internal_sum_pair(
    __m512i* res_PQ, 
    __m512i* P0Q1, __m512i* Q0P1, __m512i* P2Q3, __m512i* Q2P3
)
{
    __m512i sum[16], t[16];
    fp2_t P0Q1_unvec[8], Q0P1_unvec[8], P2Q3_unvec[8], Q2P3_unvec[8], t_unvec[8];

    // Pa0, Qa1, Pb0, Qb1, Pc0, Qc1, Pd0, Qd1 = A
    // Pa0, Pb0, Pc0, Pd0, Qa1, Qb1, Qc1, Qd1 = A OLD
    fp2_8x1x1_to_unvectorized(P0Q1_unvec, P0Q1);
    // Qa0, Pa1, Qb0, Pb1, Qc0, Pc1, Qd0, Pd1 = B
    // Qa0, Qb0, Qc0, Qd0, Pa1, Pb1, Pc1, Pd1 = B OLD
    fp2_8x1x1_to_unvectorized(Q0P1_unvec, Q0P1);
    // Pa2, Qa3, Pb2, Qb3, Pc2, Qc3, Pd2, Qd3 = C
    // Pa2, Pb2, Pc2, Pd2, Qa3, Qb3, Qc3, Qd3 = C OLD
    fp2_8x1x1_to_unvectorized(P2Q3_unvec, P2Q3);
    // Qa2, Pa3, Qb2, Pb3, Qc2, Pc3, Qd2, Pd3 = D
    // Qa2, Qb2, Qc2, Qd2, Pa3, Pb3, Pc3, Pd3 = D OLD
    fp2_8x1x1_to_unvectorized(Q2P3_unvec, Q2P3);

    // Compute the sum of the elements for each of the 8x1x1 blobs
    // P:  a = Pa0 + Pb0 + Pc0 + Pd0
    //     b = Pa1 + Pb1 + Pc1 + Pd1
    //     c = Pa2 + Pb2 + Pc2 + Pd2
    //     d = Pa3 + Pb3 + Pc3 + Pd3
    //
    // Q:  a = Qa0 + Qb0 + Qc0 + Qd0
    //     b = Qa1 + Qb1 + Qc1 + Qd1
    //     c = Qa2 + Qb2 + Qc2 + Qd2
    //     d = Qa3 + Qb3 + Qc3 + Qd3

    // E = AVX_8x1x1(Pa0, Pa1, Pa2, Pa3, Qa0, Qa1, Qa2, Qa3)
    t_unvec[0] = P0Q1_unvec[0]; t_unvec[1] = Q0P1_unvec[1]; t_unvec[2] = P2Q3_unvec[0]; t_unvec[3] = Q2P3_unvec[1];
    t_unvec[4] = Q0P1_unvec[0]; t_unvec[5] = P0Q1_unvec[1]; t_unvec[6] = Q2P3_unvec[0]; t_unvec[7] = P2Q3_unvec[1];
    fp2_unvectorized_to_8x1x1(sum, t_unvec);

    // tmp = AVX_8x1x1(Pb0, Pb1, Pb2, Pb3, Qb0, Qb1, Qb2, Qb3)
    t_unvec[0] = P0Q1_unvec[2]; t_unvec[1] = Q0P1_unvec[3]; t_unvec[2] = P2Q3_unvec[2]; t_unvec[3] = Q2P3_unvec[3];
    t_unvec[4] = Q0P1_unvec[2]; t_unvec[5] = P0Q1_unvec[3]; t_unvec[6] = Q2P3_unvec[2]; t_unvec[7] = P2Q3_unvec[3];
    fp2_unvectorized_to_8x1x1(t, t_unvec);
    fp2_add_8x1x1(sum, sum, t);

    // tmp = AVX_8x1x1(Pc0, Pc1, Pc2, Pc3, Qc0, Qc1, Qc2, Qc3)
    t_unvec[0] = P0Q1_unvec[4]; t_unvec[1] = Q0P1_unvec[5]; t_unvec[2] = P2Q3_unvec[4]; t_unvec[3] = Q2P3_unvec[5];
    t_unvec[4] = Q0P1_unvec[4]; t_unvec[5] = P0Q1_unvec[5]; t_unvec[6] = Q2P3_unvec[4]; t_unvec[7] = P2Q3_unvec[5];
    fp2_unvectorized_to_8x1x1(t, t_unvec);
    fp2_add_8x1x1(sum, sum, t);

    // tmp = AVX_8x1x1(Pd0, Pd1, Pd2, Pd3, Qd0, Qd1, Qd2, Qd3)
    t_unvec[0] = P0Q1_unvec[6]; t_unvec[1] = Q0P1_unvec[7]; t_unvec[2] = P2Q3_unvec[6]; t_unvec[3] = Q2P3_unvec[7];
    t_unvec[4] = Q0P1_unvec[6]; t_unvec[5] = P0Q1_unvec[7]; t_unvec[6] = Q2P3_unvec[6]; t_unvec[7] = P2Q3_unvec[7];
    fp2_unvectorized_to_8x1x1(t, t_unvec);
    fp2_add_8x1x1(res_PQ, sum, t);

    // TODO: Order of fp2 coords is wrong. For now I manually rearrange them
    fp2_t t_unvec0[8], t_unvec1[8];
    fp2_8x1x1_to_unvectorized(t_unvec0, res_PQ);
    t_unvec1[0] = t_unvec0[0]; t_unvec1[1] = t_unvec0[4]; t_unvec1[2] = t_unvec0[1]; t_unvec1[3] = t_unvec0[5];
    t_unvec1[4] = t_unvec0[2]; t_unvec1[5] = t_unvec0[6]; t_unvec1[6] = t_unvec0[3]; t_unvec1[7] = t_unvec0[7];
    fp2_unvectorized_to_8x1x1(res_PQ, t_unvec1);    
}

void gluing_apply_matrix(
    __m512i* res_P, 
    __m512i* matrix01, __m512i* matrix23, __m512i* P
)
{
    __m512i P0P1[16], P2P3[16];

    theta_mul_pair(P0P1, matrix01, P);
    theta_mul_pair(P2P3, matrix23, P);

    gluing_internal_sum(res_P, P0P1, P2P3);
}

// TODO: If we always call this function with 
// input PQ = the output PQ, maybe we can optimize.
void gluing_apply_matrix_pair(
    __m512i* res_PQ, 
    __m512i* matrix01, __m512i* matrix23, __m512i* PQ
)
{
    __m512i QP[16], P0Q1[16], Q0P1[16], P2Q3[16], Q2P3[16];
    fp2_t PQ_unvec[8], t_unvec[8];

    // Swap P || Q in a doubled point to Q || P
    fp2_8x1x1_to_unvectorized(PQ_unvec, PQ);
    t_unvec[0] = PQ_unvec[1]; t_unvec[1] = PQ_unvec[0]; t_unvec[2] = PQ_unvec[3]; t_unvec[3] = PQ_unvec[2];
    t_unvec[4] = PQ_unvec[5]; t_unvec[5] = PQ_unvec[4]; t_unvec[6] = PQ_unvec[7]; t_unvec[7] = PQ_unvec[6];
    fp2_unvectorized_to_8x1x1(QP, t_unvec);

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
    __m512i t0[16], t1[16];
    fp2_t x_unvec[4], z_unvec[4], t_unvec0[8], t_unvec1[8];
    fp2_4x2x1_to_unvectorized(x_unvec, x);
    fp2_4x2x1_to_unvectorized(z_unvec, z);

    // t1 = AVX_8x1x1(X1, X1, Z1, Z1, X2, X2, Z2, Z2)
    t_unvec0[0] = x_unvec[0]; t_unvec0[1] = x_unvec[0]; t_unvec0[2] = z_unvec[0]; t_unvec0[3] = z_unvec[0];
    t_unvec0[4] = x_unvec[1]; t_unvec0[5] = x_unvec[1]; t_unvec0[6] = z_unvec[1]; t_unvec0[7] = z_unvec[1];
    fp2_unvectorized_to_8x1x1(t0, t_unvec0);
    // t2 = AVX_8x1x1(X3, Z3, X3, Z3, X4, Z4, X4, Z4)
    t_unvec1[0] = x_unvec[2]; t_unvec1[1] = z_unvec[2]; t_unvec1[2] = x_unvec[2]; t_unvec1[3] = z_unvec[2];
    t_unvec1[4] = x_unvec[3]; t_unvec1[5] = z_unvec[3]; t_unvec1[6] = x_unvec[3]; t_unvec1[7] = z_unvec[3];
    fp2_unvectorized_to_8x1x1(t1, t_unvec1);
    
    fp2_mul_8x1x1(res_P12, t0, t1);

    // TODO: Order of fp2 coords is wrong. For now I manually rearrange them
    fp2_8x1x1_to_unvectorized(t_unvec0, res_P12);
    t_unvec1[0] = t_unvec0[0]; t_unvec1[1] = t_unvec0[4]; t_unvec1[2] = t_unvec0[1]; t_unvec1[3] = t_unvec0[5];
    t_unvec1[4] = t_unvec0[2]; t_unvec1[5] = t_unvec0[6]; t_unvec1[6] = t_unvec0[3]; t_unvec1[7] = t_unvec0[7];
    fp2_unvectorized_to_8x1x1(res_P12, t_unvec1);

    gluing_apply_matrix_pair(res_P12, matrix1, matrix2, res_P12);
}

void gluing_codomain(
    __m512i* res_OA, __m512i* res_OA_dual_inv, __m512i* res_J_prime, __m512i* res_matrix01, __m512i* res_matrix23,
    __m512i* A24_12, __m512i* x_8, __m512i* z_8
)
{
    __m512i x_8_[8], z_8_[8], x_4[8], z_4[8], P12[16], t0[8], t1[8];
    fp2_t P12_unvec[8], ABCD_unvec[4], inv_unvec[4], t_unvec[4];

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
    fp2_8x1x1_to_unvectorized(P12_unvec, P12);
    int last_is_not_zero = fp2_is_zero(P12_unvec+6) 
        & fp2_is_zero(P12_unvec+7);
    // TODO: Wrap asserts in #ifdefined, so they don't impact benchmarks
    assert(last_is_not_zero);

    // TODO: Why do we not check the other coordinates?
    int projective_factors_not_zero = !(
        fp2_is_zero(P12_unvec) | fp2_is_zero(P12_unvec+1) 
        | fp2_is_zero(P12_unvec+2) | fp2_is_zero(P12_unvec+5)
    );
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

    // x = X1 * A_inv
    // y = Z1 * C_inv
    //
    // Which requires 6 multiplications and we can do it by doing
    // two calls to pairwise_multiplication with the last element
    // of t1 and t2 being 1 (so we only need 3/4 multiplications
    // from this function)
    // ============================================================

    // t1 = AVX_4x2x1(X1, Y1, X1, 1)
    t_unvec[0] = P12_unvec[0]; t_unvec[1] = P12_unvec[2]; t_unvec[2] = P12_unvec[0]; fp2_set_one(t_unvec+3);
    fp2_unvectorized_to_4x2x1(t0, t_unvec);
    // t2 = AVX_4x2x1(X2, X2, Z2, 1)
    t_unvec[0] = P12_unvec[1]; t_unvec[1] = P12_unvec[1]; t_unvec[2] = P12_unvec[5]; fp2_set_one(t_unvec+3);
    fp2_unvectorized_to_4x2x1(t1, t_unvec);

    // A, B, C, _ = pairwise_multiplication_3(t1, t2)
    fp2_mul_4x2x1(t0, t0, t1);
    fp2_4x2x1_to_unvectorized(ABCD_unvec, t0);

    // t1 = AVX_4x2x1(Y1, C, B, 1)
    t_unvec[0] = P12_unvec[2]; t_unvec[1] = ABCD_unvec[2]; t_unvec[2] = ABCD_unvec[1]; fp2_set_one(t_unvec+3);
    fp2_unvectorized_to_4x2x1(t0, t_unvec);
    // t2 = AVX_4x2x1(Z2, Y1, Z1, 1)
    t_unvec[0] = P12_unvec[5]; t_unvec[1] = P12_unvec[2]; t_unvec[2] = P12_unvec[4]; fp2_set_one(t_unvec+3);
    fp2_unvectorized_to_4x2x1(t1, t_unvec);
    // A_inv, x, y, _ = pairwise_multiplication_3(t1, t2)
    // B_inv = C
    // C_inv = B
    fp2_mul_4x2x1(t0, t0, t1);
    fp2_4x2x1_to_unvectorized(inv_unvec, t0);
    
    //if (Y1 * B_inv != x) or (X2 * A_inv != Z2 * C_inv):
    //    raise ValueError("input kernel is not isotropic")
    fp2_mul(t_unvec, P12_unvec+2, ABCD_unvec+2);
    fp2_mul(t_unvec+1, P12_unvec+1, inv_unvec);
    fp2_mul(t_unvec+2, P12_unvec+5, ABCD_unvec+1);
    int ker_is_isotropic = fp2_is_equal(t_unvec, inv_unvec+1) 
        & fp2_is_equal(t_unvec+1, t_unvec+2);
    assert(ker_is_isotropic);

    // OA = hadamard(AVX_4x2x1(A, B, C, 0))
    t_unvec[0] = ABCD_unvec[0]; t_unvec[1] = ABCD_unvec[1]; t_unvec[2] = ABCD_unvec[2]; fp2_set_zero(t_unvec+3);
    fp2_unvectorized_to_4x2x1(res_OA, t_unvec);
    theta_had_single(res_OA, res_OA);
    // OA_dual_inv = AVX_4x2x1(A_inv, B_inv, C_inv, 0)
    t_unvec[0] = inv_unvec[0]; t_unvec[1] = ABCD_unvec[2]; t_unvec[2] = ABCD_unvec[1]; fp2_set_zero(t_unvec+3);
    fp2_unvectorized_to_4x2x1(res_OA_dual_inv, t_unvec);
    // J_prime = AVX_4x2x1(y, y, x, x)
    t_unvec[0] = inv_unvec[2]; t_unvec[1] = inv_unvec[2]; t_unvec[2] = inv_unvec[1]; t_unvec[3] = inv_unvec[1];
    fp2_unvectorized_to_4x2x1(res_J_prime, t_unvec);
}

void gluing_eval_pair(
    __m512i* res,
    __m512i* A_12,  __m512i* xP, __m512i* yP, __m512i* zP,
    __m512i* xT_8, __m512i* yT_8, __m512i* zT_8,
    __m512i* J_prime, __m512i* matrix01, __m512i* matrix23
)
{
    __m512i u[8], v[8], w[8], t0[16], t1[16];
    fp2_t u_unvec[4], v_unvec[4], w_unvec[4], t_unvec0[8], t_unvec1[8];
    fp2_t mul_unvec0[8], mul_unvec1[8], mul_unvec2[4], add_unvec[4];

    // Add components
    // u1, v1, w1 = add_components(E1, P1, T1_8)
    // u3, v3, w3 = add_components(E1, Q1, T1_8)
    // u2, v2, w2 = add_components(E2, P2, T2_8)
    // u4, v4, w4 = add_components(E2, Q2, T2_8)
    ec_add_components_4x2x1(u, v, w, A_12, xP, yP, zP, xT_8, yT_8, zT_8);

    // ===============================================================
    // We want to compute four additions and 9 multiplications as best
    // as we can...
    //
    // U1 = AVX_4x2x1(u1 * u2 + v1 * v2, u1 * w2, w1 * u2, w1 * w2)
    // U2 = AVX_4x2x1(u3 * u4 + v3 * v4, u3 * w4, w3 * u4, w3 * w4)
    // V1 = AVX_4x2x1(v1 * u2 + u1 * v2, v1 * w2, w1 * v2, 0)
    // V2 = AVX_4x2x1(v3 * u4 + u3 * v4, v3 * w4, w3 * v4, 0)
    // ===============================================================

    // u1, u3, u2, u4 = U
    // v1, v3, v2, v4 = V
    // w1, w3, w2, w4 = W
    fp2_4x2x1_to_unvectorized(u_unvec, u);
    fp2_4x2x1_to_unvectorized(v_unvec, v);
    fp2_4x2x1_to_unvectorized(w_unvec, w);

    // TT0 = AVX_8x1x1(u1, u1, w1, w1, u3, u3, w3, w3)
    t_unvec0[0] = u_unvec[0]; t_unvec0[1] = u_unvec[0]; t_unvec0[2] = w_unvec[0]; t_unvec0[3] = w_unvec[0];
    t_unvec0[4] = u_unvec[1]; t_unvec0[5] = u_unvec[1]; t_unvec0[6] = w_unvec[1]; t_unvec0[7] = w_unvec[1];
    // TT1 = AVX_8x1x1(u2, w2, u2, w2, u4, w4, u4, w4)
    t_unvec1[0] = u_unvec[2]; t_unvec1[1] = w_unvec[2]; t_unvec1[2] = u_unvec[2]; t_unvec1[3] = w_unvec[2];
    t_unvec1[4] = u_unvec[3]; t_unvec1[5] = w_unvec[3]; t_unvec1[6] = u_unvec[3]; t_unvec1[7] = w_unvec[3];
    fp2_unvectorized_to_8x1x1(t0, t_unvec0);
    fp2_unvectorized_to_8x1x1(t1, t_unvec1);
    // u1u2, u1w2, w1u2, w1w2, u3u4, u3w4, w3u4, w3w4 = avx_wide_pairwise_mul(TT0, TT1)
    fp2_mul_8x1x1(t0, t0, t1);
    fp2_8x1x1_to_unvectorized(mul_unvec0, t0);

    // TT0 = AVX_8x1x1(v1, u1, v1, w1, v3, u3, v3, w3)
    t_unvec0[0] = v_unvec[0]; t_unvec0[1] = u_unvec[0]; t_unvec0[2] = v_unvec[0]; t_unvec0[3] = w_unvec[0];
    t_unvec0[4] = v_unvec[1]; t_unvec0[5] = u_unvec[1]; t_unvec0[6] = v_unvec[1]; t_unvec0[7] = w_unvec[1];
    // TT1 = AVX_8x1x1(u2, v2, w2, v2, u4, v4, w4, v4)
    t_unvec1[0] = u_unvec[2]; t_unvec1[1] = v_unvec[2]; t_unvec1[2] = w_unvec[2]; t_unvec1[3] = v_unvec[2];
    t_unvec1[4] = u_unvec[3]; t_unvec1[5] = v_unvec[3]; t_unvec1[6] = w_unvec[3]; t_unvec1[7] = v_unvec[3];
    fp2_unvectorized_to_8x1x1(t0, t_unvec0);
    fp2_unvectorized_to_8x1x1(t1, t_unvec1);
    //v1u2, u1v2, v1w2, w1v2, v3u4, u3v4, v3w4, w3v4 = avx_wide_pairwise_mul(TT0, TT1)
    fp2_mul_8x1x1(t0, t0, t1);
    fp2_8x1x1_to_unvectorized(mul_unvec1, t0);

    // T0 = AVX_4x2x1(v1, v3, 0, 0)
    t_unvec0[0] = v_unvec[0]; t_unvec0[1] = v_unvec[1]; fp2_set_zero(t_unvec0+2); fp2_set_zero(t_unvec0+3);
    // T1 = AVX_4x2x1(v2, v4, 0, 0)
    t_unvec1[0] = v_unvec[2]; t_unvec1[1] = v_unvec[3]; fp2_set_zero(t_unvec1+2); fp2_set_zero(t_unvec1+3);
    fp2_unvectorized_to_4x2x1(t0, t_unvec0);
    fp2_unvectorized_to_4x2x1(t1, t_unvec1);
    //v1v2, v3v4, _, _ = avx_pairwise_mul(T0, T1)  // Two redundant mul here.
    fp2_mul_4x2x1(t0, t0, t1);
    fp2_4x2x1_to_unvectorized(mul_unvec2, t0);

    // T0 = AVX_4x2x1(u1u2, u3u4, v1u2, v3u4)
    t_unvec0[0] = mul_unvec0[0]; t_unvec0[1] = mul_unvec0[4]; t_unvec0[2] = mul_unvec1[0]; t_unvec0[3] = mul_unvec1[4];
    // T1 = AVX_4x2x1(v1v2, v3v4, u1v2, u3v4)
    t_unvec1[0] = mul_unvec2[0]; t_unvec1[1] = mul_unvec2[1]; t_unvec1[2] = mul_unvec1[1]; t_unvec1[3] = mul_unvec1[5];
    fp2_unvectorized_to_4x2x1(t0, t_unvec0);
    fp2_unvectorized_to_4x2x1(t1, t_unvec1);
    // u1u2_v1v2, u3u4_v3v4, v1u2_u1v2, v3u4_u3v4 = avx_add(T0, T1)
    fp2_add_4x2x1(t0, t0, t1);
    fp2_4x2x1_to_unvectorized(add_unvec, t0);

    // TODO: See the ipad notes for an overview of which values are not needed at this point (the inputs to the previous add are no longer needed)
    // U12 = AVX_8x1x1(u1u2_v1v2, u1w2, w1u2, w1w2, u3u4_v3v4, u3w4, w3u4, w3w4)
    t_unvec0[0] = add_unvec[0]; t_unvec0[1] = mul_unvec0[1]; t_unvec0[2] = mul_unvec0[2]; t_unvec0[3] = mul_unvec0[3];
    t_unvec0[4] = add_unvec[1]; t_unvec0[5] = mul_unvec0[5]; t_unvec0[6] = mul_unvec0[6]; t_unvec0[7] = mul_unvec0[7];
    // V12 = AVX_8x1x1(v1u2_u1v2, v1w2, w1v2, 0, v3u4_u3v4, v3w4, w3v4, 0)
    t_unvec1[0] = add_unvec[2]; t_unvec1[1] = mul_unvec1[2]; t_unvec1[2] = mul_unvec1[3]; fp2_set_zero(t_unvec1+3);
    t_unvec1[4] = add_unvec[3]; t_unvec1[5] = mul_unvec1[6]; t_unvec1[6] = mul_unvec1[7]; fp2_set_zero(t_unvec1+7);
    fp2_unvectorized_to_8x1x1(t0, t_unvec0);
    fp2_unvectorized_to_8x1x1(t1, t_unvec1);

    // TODO: Order of fp2 coords is wrong. For now I manually rearrange them
    fp2_8x1x1_to_unvectorized(t_unvec0, t0);
    t_unvec1[0] = t_unvec0[0]; t_unvec1[1] = t_unvec0[4]; t_unvec1[2] = t_unvec0[1]; t_unvec1[3] = t_unvec0[5];
    t_unvec1[4] = t_unvec0[2]; t_unvec1[5] = t_unvec0[6]; t_unvec1[6] = t_unvec0[3]; t_unvec1[7] = t_unvec0[7];
    fp2_unvectorized_to_8x1x1(t0, t_unvec1);
    fp2_8x1x1_to_unvectorized(t_unvec0, t1);
    t_unvec1[0] = t_unvec0[0]; t_unvec1[1] = t_unvec0[4]; t_unvec1[2] = t_unvec0[1]; t_unvec1[3] = t_unvec0[5];
    t_unvec1[4] = t_unvec0[2]; t_unvec1[5] = t_unvec0[6]; t_unvec1[6] = t_unvec0[3]; t_unvec1[7] = t_unvec0[7];
    fp2_unvectorized_to_8x1x1(t1, t_unvec1);

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

void gluing_eval_product_to_theta_pair(
    __m512i* res,
    ec_point_t* P1, ec_point_t* P2, ec_point_t* Q1, ec_point_t* Q2, 
    __m512i* matrix01, __m512i* matrix23
)
{
    __m512i t0[16];
    fp2_t unvec0[8], unvec1[8];
    // PQ = AVX_8x1x1(X1, X1, Z1, Z1, X3, X3, Z3, Z3)
    unvec0[0] = P1->x; unvec0[1] = P1->x; unvec0[2] = P1->z; unvec0[3] = P1->z;
    unvec0[4] = Q1->x; unvec0[5] = Q1->x; unvec0[6] = Q1->z; unvec0[7] = Q1->z;
    // T0 = AVX_8x1x1(X2, Z2, X2, Z2, X4, Z4, X4, Z4)
    unvec1[0] = P2->x; unvec1[1] = P2->z; unvec1[2] = P2->x; unvec1[3] = P2->z;
    unvec1[4] = Q2->x; unvec1[5] = Q2->z; unvec1[6] = Q2->x; unvec1[7] = Q2->z;
    fp2_unvectorized_to_8x1x1(res, unvec0);
    fp2_unvectorized_to_8x1x1(t0, unvec1);  
    
    fp2_mul_8x1x1(res, res, t0);

    // TODO: Order of fp2 coords is wrong. For now I manually rearrange them
    fp2_t t_unvec0[8], t_unvec1[8];
    fp2_8x1x1_to_unvectorized(t_unvec0, res);
    t_unvec1[0] = t_unvec0[0]; t_unvec1[1] = t_unvec0[4]; t_unvec1[2] = t_unvec0[1]; t_unvec1[3] = t_unvec0[5];
    t_unvec1[4] = t_unvec0[2]; t_unvec1[5] = t_unvec0[6]; t_unvec1[6] = t_unvec0[3]; t_unvec1[7] = t_unvec0[7];
    fp2_unvectorized_to_8x1x1(res, t_unvec1);  

    gluing_apply_matrix_pair(res, matrix01, matrix23, res);
}

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

    assert(fp2_is_zero_8x1x1(res) & 0x40); // w1 = 0
    assert(fp2_is_zero_8x1x1(res) & 0x80); // w2 = 0
    theta_mul_pair(res, res, OA_inv);
    theta_had_pair(res, res);
}

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