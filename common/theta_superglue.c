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

static int find_pos_early(fp2_t *s, fp2_t *r, fp2_t *s_other, fp2_t *r_other) {
	if (fp2_is_zero(s))
		return 0;
	else if (fp2_is_zero(r))
		return 1;
	else if (fp2_is_zero(s_other))
		return 2;
	else if (fp2_is_zero(r_other))
		return 3;
	else
		return -1;
}

bool compute_symmetric_position(int *pos_1, int *pos_2, __m512i* x_4, __m512i* z_4) {
	__m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
	fp2_t rs[8];
	// The positions we will find
	*pos_1 = *pos_2 = -1;

	// T0 = AVX_8x1x1(x1, x2, x3, x4, x1, x2, x3, x4)
    // T1 = AVX_8x1x1(z1, z2, z3, z4, z1, z2, z3, z4)
    // T2 = avx_811_add_sub(T0, T1)
	// r1, r2, r3, r4, s1, s2, s3, s4 = T2
	fp2_4x2x1_to_8x1x1(t0, x_4, x_4);
	fp2_4x2x1_to_8x1x1(t1, z_4, z_4);
	fp2_mixaddsub_8x1x1(t2, t0, t1, 0x55, 0x55);
	fp2_8x1x1_to_unvectorized(rs, t2);

    // For SQIsign, pos_2 is always 2 or 3, but pos_1
    // is unbounded. We can return early here if we're
    // in the lucky 2/3 cases
    *pos_1 = find_pos_early(&rs[1], &rs[0], &rs[3], &rs[2]);
    *pos_2 = find_pos_early(&rs[5], &rs[4], &rs[7], &rs[6]);

    if (*pos_1 != -1)
        return true;

    // If we've reached here, we need to do extra work to find pos_1
    // at a cost of 2M and 2a
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
	// TODO: optimize with fp2_is_zero_2x4x1(t2)
	fp2_2x4x1_to_unvectorized(rs, t2);

    // Find the position in the case where neither P or Q is above (0 : 1)
    if (fp2_is_zero(&rs[0]))
        *pos_1 = 4;
    else if (fp2_is_zero(&rs[1]))
        *pos_1 = 5;
    else {
		printf("No vanishing coordinate found in 4-torsion");
		return false;
	}
    return true;
}

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
	fp2_t t[8], u[8], v[8], g0_1[3], g1_1[3], g2_1[3], g0_2[3], g1_2[3], g2_2[3];
	__m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];

    // Compute the positions of each element
    // 2M + 10a
    compute_symmetric_position(&pos_1, &pos_2, x_4, z_4);
#ifndef NDEBUG
	// printf("\npos %d %d\n", pos_1, pos_2);
	assert(pos_2 == 2 || pos_2 == 3);
#endif

    // Memory management for what we send into compute_matrix_coefficients
    //n based on pos_1 and pos_2

	// x1, x2, x3, _ = X_4
	// z1, z2, z3, _ = Z_4
	// if pos_1 == 0 or pos_1 == 1:
	//    four_torsion = AVX_4x2x1(x2, z2, x3, z3)
	// else:
	//  four_torsion = AVX_4x2x1(x1, z1, x3, z3)
	fp2_4x2x1_to_unvectorized(t, x_4);
	fp2_4x2x1_to_unvectorized(t+4, z_4);
	if (pos_1 == 0 || pos_1 == 1) {
		fp2_copy(&u[0], &t[1]);
		fp2_copy(&u[1], &t[5]);
	} else {
		fp2_copy(&u[0], &t[0]);
		fp2_copy(&u[1], &t[4]);
	}
	fp2_copy(&u[2], &t[2]);
	fp2_copy(&u[3], &t[6]);
	fp2_set_zero(&u[6]);
	fp2_set_zero(&u[7]);
	// x1, z1, x2, z2 = four_torsion
    // 2M + 4S + 6a
	// 8M (2M wasted)
	// T0 = AVX_8x1x1(x1, z1, x2, z2, x1, x2, 0, 0)
    // T1 = AVX_8x1x1(x1, z1, x2, z2, z1, z2, 0, 0)
	// T2 = avx_811_mul(T0, T1)
	fp2_copy(&u[4], &u[0]);
	fp2_copy(&u[5], &u[2]);
	fp2_unvectorized_to_8x1x1(t0, u);
	fp2_copy(&u[4], &u[1]);
	fp2_copy(&u[5], &u[3]);
	fp2_unvectorized_to_8x1x1(t1, u);
	fp2_mul_8x1x1(t2, t0, t1);

	// s1, s2, s3, s4, r2, r4, _, _ = T2
	fp2_8x1x1_to_unvectorized(t, t2);
	fp2_copy(&u[0], &t[0]);
	fp2_copy(&u[1], &t[4]);
	fp2_copy(&u[2], &t[2]);
	fp2_copy(&u[3], &t[5]);
	fp2_copy(&u[4], &t[0]);
	fp2_copy(&u[5], &t[2]);
	// 8a (2a wasted)
	// T0 = AVX_8x1x1(s1, r2, s3, r4, s1, s3, 0, 0)
	fp2_unvectorized_to_8x1x1(t0, u);

	fp2_copy(&u[0], &t[1]);
	fp2_copy(&u[2], &t[3]);
	fp2_copy(&u[4], &t[1]);
	fp2_copy(&u[5], &t[3]);
    // T1 = AVX_8x1x1(s2, r2, s4, r4, s2, s4, 0, 0)
	// avx_811_add_sub(T0, T1)
	fp2_unvectorized_to_8x1x1(t1, u);
	fp2_mixaddsub_8x1x1(t2, t0, t1, 0x0f, 0x0f);
	fp2_8x1x1_to_unvectorized(t, t2);

    // Compute the unique elements which we use to file the symmetric elements
    // 2M + 4S + 6a
    //coefficients = t2 = compute_matrix_coefficients(four_torsion)

    // TODO: this can be done in parallel, but there is no arithmetic, only
    // memory management so maybe it's not worth it
    // r1, r2, r3, r4, delta_1, delta_2, _, _ = coefficients
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
	// fp2_print("g0_1[0] = ", &g0_1[0]);
	// fp2_print("g0_1[1] = ", &g0_1[1]);
	// fp2_print("g0_1[2] = ", &g0_1[2]);
	// fp2_print("g1_1[0] = ", &g1_1[0]);
	// fp2_print("g1_1[1] = ", &g1_1[1]);
	// fp2_print("g1_1[2] = ", &g1_1[2]);
	// fp2_print("g2_1[0] = ", &g2_1[0]);
	// fp2_print("g2_1[1] = ", &g2_1[1]);
	// fp2_print("g2_1[2] = ", &g2_1[2]);
	// fp2_print("g0_2[0] = ", &g0_2[0]);
	// fp2_print("g0_2[1] = ", &g0_2[1]);
	// fp2_print("g0_2[2] = ", &g0_2[2]);
	// fp2_print("g1_2[0] = ", &g1_2[0]);
	// fp2_print("g1_2[1] = ", &g1_2[1]);
	// fp2_print("g1_2[2] = ", &g1_2[2]);
	// fp2_print("g2_2[0] = ", &g2_2[0]);
	// fp2_print("g2_2[1] = ", &g2_2[1]);
	// fp2_print("g2_2[2] = ", &g2_2[2]);
#endif

	/* 8M
	T0 = AVX_8x1x1(
        g0_1[0], g0_1[0], g0_1[1], g0_1[1], delta_1, g1_1[0], g1_1[1], g1_1[1]
    )
    T1 = AVX_8x1x1(
        g0_2[0], g0_2[1], g0_2[0], g0_2[1], delta_2, g1_2[1], g1_2[0], g1_2[1]
    )
    T2 = avx_811_mul(T0, T1) */
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

    // 4a
    //v = AVX_4x2x1(T2[0], T2[1], T2[2], T2[3])
    //S0 = AVX_4x2x1(T2[4], T2[5], T2[6], T2[7])
    //v = avx_421_add(v, S0)
	fp2_8x1x1_to_unvectorized(u, t2);
	fp2_add(&v[0], &u[0], &u[4]);
	fp2_add(&v[1], &u[1], &u[5]);
	fp2_add(&v[2], &u[2], &u[6]);
	fp2_add(&v[3], &u[3], &u[7]);

    // 4M
    // S1 = AVX_4x2x1(g2_1[0], g2_1[0], g2_1[1], g2_1[1])
    // S2 = AVX_4x2x1(g2_2[0], g2_2[1], g2_2[0], g2_2[1])
    // S3 = avx_421_mul(S1, S2)
    // v = avx_421_sub(v, S3)
	fp2_mul(&t[0], &g2_1[0], &g2_2[0]);
	fp2_mul(&t[1], &g2_1[0], &g2_2[1]);
	fp2_mul(&t[2], &g2_1[1], &g2_2[0]);
	fp2_mul(&t[3], &g2_1[1], &g2_2[1]);
	fp2_sub(&v[0], &v[0], &t[0]);
	fp2_sub(&v[1], &v[1], &t[1]);
	fp2_sub(&v[2], &v[2], &t[2]);
	fp2_sub(&v[3], &v[3], &t[3]);

    // 4M
    // S4 = AVX_4x2x1(g1_2[0], g1_2[2], g0_1[0], g0_1[2])
    // S5 = AVX_4x2x1(v[0], v[1], v[0], v[2])
    // S6 = avx_421_mul(S4, S5)
	fp2_mul(&v[4], &v[0], &g1_2[0]);
	fp2_mul(&v[5], &v[1], &g1_2[1]);
	fp2_mul(&v[6], &v[0], &g0_1[0]);
	fp2_mul(&v[7], &v[2], &g0_1[2]);

    // 2a
    // R0 = AVX_2x4x1(S6[0], S6[2])
    // R1 = AVX_2x4x1(S6[1], S6[3])
    // R2 = avx_241_add(R0, R1)
	fp2_add(&v[4], &v[4], &v[5]);
	fp2_add(&v[5], &v[6], &v[7]);

    // These first two are optional, but fit in a 4-way
    // 4M
    // S7 = AVX_4x2x1(g0_1[0], g0_1[2], delta_1, delta_2)
    // S8 = AVX_4x2x1(v[1], v[3], R2[0], R2[1])
    // S9 = avx_421_mul(S7, S8)
	fp2_mul(&t[0], &v[1], &g0_1[0]);
	fp2_mul(&t[1], &v[3], &g0_1[2]);
	fp2_mul(&t[2], &v[4], &t[4]);
	fp2_mul(&t[3], &v[5], &t[5]);

    // This only needs to be done 1/3 of the time
    fp2_t sgl_val;
	fp2_set_zero(&sgl_val);
    if (*position == 4) {
        // Another 1a + 1M
        // R3 = AVX_2x4x1(S9[0], 0)
        // R4 = AVX_2x4x1(S9[1], 0)
        // R5 = avx_241_add(R3, R4)
        // R6 = AVX_2x4x1(delta_2, 0)
        // R7 = avx_241_mul(R5, R6)
        // sgl_val = R7[0]
		fp2_add(&sgl_val, &t[0], &t[1]);
		fp2_mul(&sgl_val, &sgl_val, &t[5]);
	}

    // 4M
    // S10 = AVX_4x2x1(S0[0], S0[0], S0[0], S0[0])
    // v = avx_421_mul(v, S10)
	fp2_mul(&v[0], &v[0], &u[4]);
	fp2_mul(&v[1], &v[1], &u[4]);
	fp2_mul(&v[2], &v[2], &u[4]);
	fp2_mul(&v[3], &v[3], &u[4]);

    // Compute dual_theta_null_square for later
    // 3S
    // S11 = AVX_4x2x1(S9[2], S9[3], v[0], 0)
    // S12 = avx_421_sqr(S11)
	fp2_sqr(&t[4], &t[2]);
	fp2_sqr(&t[5], &t[3]);
	fp2_sqr(&t[6], &v[0]);
	fp2_set_zero(&t[7]);

    // 3a
    // S13 = AVX_4x2x1(S12[0], 0, S12[2], S12[2])
    // S14 = AVX_4x2x1(S12[1], 0, S12[0], S12[1])
    // S15 = avx_421_add_sub(S13, S14)
	fp2_add(&t[0], &t[4], &t[5]);
	fp2_sub(&t[2], &t[6], &t[4]);
	fp2_sub(&t[4], &t[6], &t[5]);

    // dual_theta_null_square = [S15[0], S15[2], S15[3], 0]
	fp2_copy(&dual_theta_null_square[0], &t[0]);
	fp2_copy(&dual_theta_null_square[1], &t[2]);
	fp2_copy(&dual_theta_null_square[2], &t[4]);
	fp2_set_zero(&dual_theta_null_square[3]);

	bool sign_flip = true;
	if (*position == 1 || *position == 7) {
        // case 1 | 7:
		// Standard case: all four coordinates are nonzero
		// 6M + 6a
		// T3 = AVX_8x1x1(v[0], v[2], v[0], v[1], v[0], v[1], 0, 0)
		// T4 = AVX_8x1x1(v[1], v[3], v[2], v[3], v[3], v[2], 0, 0)
		// T5 = avx_811_mul(T3, T4)
		fp2_mul(&t[0], &v[0], &v[1]);
		fp2_mul(&t[1], &v[2], &v[3]);
		fp2_mul(&t[2], &v[0], &v[2]);
		fp2_mul(&t[3], &v[1], &v[3]);
		fp2_mul(&t[4], &v[0], &v[3]);
		fp2_mul(&t[5], &v[1], &v[2]);

		// T6 = AVX_8x1x1(T5[0], T5[2], T5[4], 0, T5[0], T5[2], T5[4], 0)
		// T7 = AVX_8x1x1(T5[1], T5[3], T5[5], 0, T5[1], T5[3], T5[5], 0)
		// T8 = avx_811_add_sub(T6, T7)
		fp2_add(&v[0], &t[0], &t[1]);
		fp2_add(&v[1], &t[2], &t[3]);
		fp2_add(&v[2], &t[4], &t[5]);
		fp2_sub(&v[4], &t[0], &t[1]);
		fp2_sub(&v[5], &t[2], &t[3]);
		fp2_sub(&v[6], &t[4], &t[5]);

		// elements = [T8[0], T8[4], T8[1], T8[5], T8[2], T8[6]]
		fp2_copy(&elements[0], &v[0]);
		fp2_copy(&elements[1], &v[4]);
		fp2_copy(&elements[2], &v[1]);
		fp2_copy(&elements[3], &v[5]);
		fp2_copy(&elements[4], &v[2]);
		fp2_copy(&elements[5], &v[6]);
		sign_flip = false;
	} else if (*position == 4) {
		// Special case: one coordinate vanishes, sign_flip encodes
		// which square root branch to take
		// 4M + 3a
		// S16 = AVX_4x2x1(v[0], S9[3], v[0], v[0])
		// S17 = AVX_4x2x1(v[1], sgl_val, v[1], v[2])
		// S18 = avx_421_mul(S16, S17)
		fp2_mul(&t[0], &v[0], &v[1]);
		fp2_mul(&t[1], &t[3], &sgl_val);
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
        //	raise ValueError("unsupported case for SQIsign verification")
		printf("ValueError(unsupported case for SQIsign verification\n");
	}

	return sign_flip;
}

bool superglue_evaluation_on_position(fp2_t *p, fp2_t s[8], int position, fp2_t dual_theta_null_square[4], fp2_t elements[6]) {
	fp2_t t2[8], t3[4], t8[8];
	// s0, s1, s2, s3, s4, s5, s6, s7 = data
    // A_inv, B_inv, C_inv, _ = dual_theta_null_square

	if (position == 4) {
		//e0, e1, _, _, _, _ = elements
		// 8M
		// T0 = AVX_8x1x1(s1, s5, s1, s5, s1, s5, s1, s5)
		// T1 = AVX_8x1x1(A_inv, e0, e1, C_inv, C_inv, e1, e0, A_inv)
		// T2 = avx_811_mul(T0, T1)
		fp2_mul(&t2[0], &s[1], &dual_theta_null_square[0]);
		fp2_mul(&t2[1], &s[5], &elements[0]);
		fp2_mul(&t2[2], &s[1], &elements[1]);
		fp2_mul(&t2[3], &s[5], &dual_theta_null_square[2]);
		fp2_mul(&t2[4], &s[1], &dual_theta_null_square[2]);
		fp2_mul(&t2[5], &s[5], &elements[1]);
		fp2_mul(&t2[6], &s[1], &elements[0]);
		fp2_mul(&t2[7], &s[5], &dual_theta_null_square[0]);
		// 4a
		// T0 = AVX_4x2x1(T2[0], T2[2], T2[4], T2[6])
		// T1 = AVX_4x2x1(T2[1], T2[3], T2[5], T2[7])
		// T2 = avx_421_add(T0, T1)
		fp2_add(&t2[0], &t2[0], &t2[1]);
		fp2_add(&t2[1], &t2[2], &t2[3]);
		fp2_add(&t2[2], &t2[4], &t2[5]);
		fp2_add(&t2[3], &t2[6], &t2[7]);

		// 8M (2 wasted)
		// T0 = AVX_8x1x1(s0, s4, s0, s4, s6, s7, 0, 0)
		// T1 = AVX_8x1x1(T2[0], T2[1], T2[2], T2[3], B_inv, B_inv, 0, 0)
		// T2 = avx_811_mul(T0, T1)
		fp2_mul(&t2[0], &s[0], &t2[0]);
		fp2_mul(&t2[1], &s[4], &t2[1]);
		fp2_mul(&t2[2], &s[0], &t2[2]);
		fp2_mul(&t2[3], &s[4], &t2[3]);
		fp2_mul(&t2[4], &s[6], &dual_theta_null_square[1]);
		fp2_mul(&t2[5], &s[7], &dual_theta_null_square[1]);

		// 4a
		// T0 = AVX_2x4x1(T2[0], T2[2])
		// T1 = AVX_2x4x1(T2[1], T2[3])
		// T3 = avx_241_add(T0, T1)
		fp2_add(&t3[0], &t2[0], &t2[1]);
		fp2_add(&t3[1], &t2[2], &t2[3]);

		// 	return AVX_4x2x1(T3[0], T2[5], T3[1], T2[4])
		fp2_copy(&p[0], &t3[0]);
		fp2_copy(&p[1], &t2[5]);
		fp2_copy(&p[2], &t3[1]);
		fp2_copy(&p[3], &t2[4]);
	} else if (position == 1 || position == 7) {
		// Cases 1 and 7 are nearly identical
        // e0, e1, e2, e3, e4, e5 = elements

		// 8M
		// T0 = AVX_8x1x1(s1, s5, s1, s5, s0, s4, s1, s5)
		// For positions 1, 7 only the role of A_inv and C_inv
		// and then X, Z are reversed.
		// if position == 7:
			// T1 = AVX_8x1x1(C_inv, e0, e2, e4, B_inv, e3, A_inv, e1)
		// else:
			// T1 = AVX_8x1x1(A_inv, e0, e2, e4, B_inv, e3, C_inv, e1)
		// T2 = avx_811_mul(T0, T1)
		if (position == 7) {
			fp2_mul(&t2[0], &s[1], &dual_theta_null_square[2]);
			fp2_mul(&t2[1], &s[5], &elements[0]);
			fp2_mul(&t2[2], &s[1], &elements[2]);
			fp2_mul(&t2[3], &s[5], &elements[4]);
			fp2_mul(&t2[4], &s[0], &dual_theta_null_square[1]);
			fp2_mul(&t2[5], &s[4], &elements[3]);
			fp2_mul(&t2[6], &s[1], &dual_theta_null_square[0]);
			fp2_mul(&t2[7], &s[5], &elements[1]);
		} else {
			fp2_mul(&t2[0], &s[1], &dual_theta_null_square[0]);
			fp2_mul(&t2[1], &s[5], &elements[0]);
			fp2_mul(&t2[2], &s[1], &elements[2]);
			fp2_mul(&t2[3], &s[5], &elements[4]);
			fp2_mul(&t2[4], &s[0], &dual_theta_null_square[1]);
			fp2_mul(&t2[5], &s[4], &elements[3]);
			fp2_mul(&t2[6], &s[1], &dual_theta_null_square[2]);
			fp2_mul(&t2[7], &s[5], &elements[1]);
		}

		// 4a
		// T3 = AVX_2x4x1(T2[0], T2[2])
		// T4 = AVX_2x4x1(T2[1], T2[3])
		// T5 = avx_241_add(T3, T4)
		fp2_add(&t3[0], &t2[0], &t2[1]);
		fp2_add(&t3[1], &t2[2], &t2[3]);

		// 8M (1 wasted)
		// T6 = AVX_8x1x1(s0, s4, s3, s3, s2, s2, s6, 0)
		// T7 = AVX_8x1x1(T5[0], T5[1], T2[4], T2[5], T2[6], T2[7], e5, 0)
		// T8 = avx_811_mul(T6, T7)
		fp2_mul(&t8[0], &s[0], &t3[0]);
		fp2_mul(&t8[1], &s[4], &t3[1]);
		fp2_mul(&t8[2], &s[3], &t2[4]);
		fp2_mul(&t8[3], &s[3], &t2[5]);
		fp2_mul(&t8[4], &s[2], &t2[6]);
		fp2_mul(&t8[5], &s[2], &t2[7]);
		fp2_mul(&t8[6], &s[6], &elements[5]);
		fp2_set_zero(&t8[7]);

		// 4a (1 wasted)
		// T9 = AVX_4x2x1(T8[0], T8[2], T8[4], 0)
		// Ta = AVX_4x2x1(T8[1], T8[3], T8[5], 0)
		// Tb = avx_421_add(T9, Ta)
		fp2_add(&t3[0], &t8[0], &t8[1]);
		fp2_add(&t3[1], &t8[2], &t8[3]);
		fp2_add(&t3[2], &t8[4], &t8[5]);
		fp2_set_zero(&t3[3]);

		// For positions 1, 7 only the role of A_inv and C_inv
		// and then X, Z are reversed.
		//if position == 7:
			//return AVX_4x2x1(Tb[2], Tb[1], Tb[0], T8[6])
		//else:
			//return AVX_4x2x1(Tb[0], Tb[1], Tb[2], T8[6])
		if (position == 7) {
			fp2_copy(&p[0], &t3[2]);
			fp2_copy(&p[1], &t3[1]);
			fp2_copy(&p[2], &t3[0]);
			fp2_copy(&p[3], &t8[6]);
		} else {
			fp2_copy(&p[0], &t3[0]);
			fp2_copy(&p[1], &t3[1]);
			fp2_copy(&p[2], &t3[2]);
			fp2_copy(&p[3], &t8[6]);
		}
	} else {
        // All other cases wont be touched by SQIsign
        // case _:
        //		raise ValueError("evaluation failed due to unsupported case")
		printf("ValueError(evaluation failed due to unsupported case\n");
		return false;
	}
	return true;
}

void superglue_evaluation_pair(fp2_t *p1, fp2_t *p2, __m512i *U, __m512i *V,
		__m512i *W, int position, bool sign_flip, fp2_t *dual_theta_null_square,
		fp2_t *elements) {
	fp2_t u[4], v[4], w[4], t0[8], t1[8], t2[8], t4[8], s2[4];
    // TODO: for codomain vi is zero, we might find a nice optimization here?
    // u1, u3, u2, u4 = U
    // v1, v3, v2, v4 = V
    // w1, w3, w2, w4 = W
	fp2_4x2x1_to_unvectorized(u, U);
	fp2_4x2x1_to_unvectorized(v, V);
	fp2_4x2x1_to_unvectorized(w, W);
	fp2_cswap(&u[1], &u[2], -1);
	fp2_cswap(&v[1], &v[2], -1);
	fp2_cswap(&w[1], &w[2], -1);

    if (sign_flip) {
		fp2_neg(&w[0], &w[0]);
		fp2_neg(&w[2], &w[2]);
	}

    // 8 adds
    // T0 = AVX_8x1x1(u1, u2, u3, u4, u1, u2, u3, u4)
    // T1 = AVX_8x1x1(v1, v2, v3, v4, v1, v2, v3, v4)
    // T2 = avx_811_add_sub(T0, T1)
	fp2_add(&t2[0], &u[0], &v[0]);
	fp2_add(&t2[1], &u[1], &v[1]);
	fp2_add(&t2[2], &u[2], &v[2]);
	fp2_add(&t2[3], &u[3], &v[3]);
	fp2_sub(&t2[4], &u[0], &v[0]);
	fp2_sub(&t2[5], &u[1], &v[1]);
	fp2_sub(&t2[6], &u[2], &v[2]);
	fp2_sub(&t2[7], &u[3], &v[3]);

    // 4M + 4S
    // T0 = AVX_8x1x1(T2[0], T2[1], w1, w2, T2[2], T2[3], w3, w4)
    // T1 = AVX_8x1x1(T2[4], T2[5], w1, w2, T2[6], T2[7], w3, w4)
    // T2 = avx_811_mul(T0, T1)
	fp2_mul(&t2[0], &t2[0], &t2[4]);
	fp2_mul(&t2[1], &t2[1], &t2[5]);
	fp2_mul(&t2[4], &t2[2], &t2[6]);
	fp2_mul(&t2[5], &t2[3], &t2[7]);
	fp2_mul(&t2[2], &w[0], &w[0]);
	fp2_mul(&t2[3], &w[1], &w[1]);
	fp2_mul(&t2[6], &w[2], &w[2]);
	fp2_mul(&t2[7], &w[3], &w[3]);

	// 8 adds
    // T0 = AVX_8x1x1(T2[0], T2[1], T2[4], T2[5], T2[0], T2[1], T2[4], T2[5])
    // T1 = AVX_8x1x1(T2[2], T2[3], T2[6], T2[7], T2[2], T2[3], T2[6], T2[7])
    // T2 = avx_811_add_sub(T0, T1)
	fp2_add(&t0[0], &t2[0], &t2[2]);
	fp2_add(&t0[1], &t2[1], &t2[3]);
	fp2_add(&t0[2], &t2[4], &t2[6]);
	fp2_add(&t0[3], &t2[5], &t2[7]);
	fp2_sub(&t0[4], &t2[0], &t2[2]);
	fp2_sub(&t0[5], &t2[1], &t2[3]);
	fp2_sub(&t0[6], &t2[4], &t2[6]);
	fp2_sub(&t0[7], &t2[5], &t2[7]);

    // 8M
    // T0 = AVX_8x1x1(u1, u2, v1, w2, u3, u4, v3, w4)
    // T1 = AVX_8x1x1(w1, w2, v2, w1, w3, w4, v4, w3)
    // T3 = avx_811_mul(T0, T1)
	fp2_mul(&t1[0], &u[0], &w[0]);
	fp2_mul(&t1[1], &u[1], &w[1]);
	fp2_mul(&t1[2], &v[0], &v[1]);
	fp2_mul(&t1[3], &w[1], &w[0]);
	fp2_mul(&t1[4], &u[2], &w[2]);
	fp2_mul(&t1[5], &u[3], &w[3]);
	fp2_mul(&t1[6], &v[2], &v[3]);
	fp2_mul(&t1[7], &w[3], &w[2]);

    // 4M (s7, r7 only used for case 4 though)
    // S0 = AVX_4x2x1(T3[2], T2[5], T3[6], T2[7])
    // S1 = AVX_4x2x1(T3[3], T2[4], T3[7], T2[6])
    // S2 = avx_421_mul(S0, S1)
	fp2_mul(&s2[0], &t1[2], &t1[3]);
	fp2_mul(&s2[1], &t0[5], &t0[4]);
	fp2_mul(&s2[2], &t1[6], &t1[7]);
	fp2_mul(&s2[3], &t0[7], &t0[6]);

    // 8 adds (2 wasted)
    //T0 = AVX_8x1x1(T3[0], T3[1], S2[0], T3[4], T3[5], S2[2], 0, 0)
    //T4 = avx_811_add(T0, T0)
	fp2_add(&t4[0], &t1[0], &t1[0]);
	fp2_add(&t4[1], &t1[1], &t1[1]);
	fp2_add(&t4[2], &s2[0], &s2[0]);
	fp2_add(&t4[3], &t1[4], &t1[4]);
	fp2_add(&t4[4], &t1[5], &t1[5]);
	fp2_add(&t4[5], &s2[2], &s2[2]);
	fp2_set_zero(&t4[6]);
	fp2_set_zero(&t4[7]);

    // 2 adds
    // T0 = AVX_2x4x1(T4[2], T4[5])
    // T5 = avx_241_add(T0, T0)
	fp2_add(&t4[6], &t4[2], &t4[2]);
	fp2_add(&t4[7], &t4[5], &t4[5]);

	fp2_t data_1[8], data_2[8];
    // Package data to send to superglue_evaluation_on_position
    // data_1 = AVX_8x1x1(T2[0], T2[1], T2[4], T2[5], T4[0], T4[1], T5[0], S2[1])
    // data_2 = AVX_8x1x1(T2[2], T2[3], T2[6], T2[7], T4[3], T4[4], T5[1], S2[3])
	fp2_copy(&data_1[0], &t0[0]);
	fp2_copy(&data_1[1], &t0[1]);
	fp2_copy(&data_1[2], &t0[4]);
	fp2_copy(&data_1[3], &t0[5]);
	fp2_copy(&data_1[4], &t4[0]);
	fp2_copy(&data_1[5], &t4[1]);
	fp2_copy(&data_1[6], &t4[6]);
	fp2_copy(&data_1[7], &s2[1]);
	fp2_copy(&data_2[0], &t0[2]);
	fp2_copy(&data_2[1], &t0[3]);
	fp2_copy(&data_2[2], &t0[6]);
	fp2_copy(&data_2[3], &t0[7]);
	fp2_copy(&data_2[4], &t4[3]);
	fp2_copy(&data_2[5], &t4[4]);
	fp2_copy(&data_2[6], &t4[7]);
	fp2_copy(&data_2[7], &s2[3]);

	// This could also be batched, but when case is 1 | 7 we do a 8 way then 7 way mul
    // when case is 4 we do an 8 way then 6 way mul, we could only save time with batched
    // additions and the code would be so much harder to read.#
    superglue_evaluation_on_position(p1, data_1, position, dual_theta_null_square, elements);
    superglue_evaluation_on_position(p2, data_2, position, dual_theta_null_square, elements);
}

// TODO: Output (OA_inv, OA_inv) in 8x1x1 for special eval.
bool superglue_codomain(__m512i* res_OA, __m512i* A24_12, int *position,
		bool *sign_flip, fp2_t dual_theta_null_square[4], fp2_t elements[6],
		__m512i* x_8, __m512i* z_8) {
	fp2_t p1[4], p2[4], t[4];
    __m512i x_8_[2*NUM_LIMBS], z_8_[2*NUM_LIMBS], x_4[NUM_LIMBS], z_4[NUM_LIMBS];
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    __mmask8 mask;

    // X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)
    ec_jac_to_xz_4x4x2x1(x_8_, z_8_, x_8, z_8);

    // Double points to recover 4-torsion for change of basis
    ec_double_4x4x2x1(x_4, z_4, A24_12, x_8_, z_8_);

	*sign_flip = compute_superglue_coefficients(position, dual_theta_null_square, elements, x_4, z_4);
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

	// Pack the (u, v, w) coordinates directly from the 8 torsion
	fp2_copy_8x1x1(t0, x_8_);
	fp2_setzero_4x2x1(t1);
	fp2_copy_8x1x1(t2, z_8_);

    // 20M + 24a
    superglue_evaluation_pair(p1, p2, t0, t1, t2, *position, *sign_flip, dual_theta_null_square, elements);

    // 4M
    // T0 = AVX_4x2x1(A1[0], A1[1], A1[0], A1[1])
    // T1 = AVX_4x2x1(A2[0], A2[0], A2[2], A2[2])
    // T2 = avx_421_mul(T0, T1)
	fp2_mul(&t[0], &p1[0], &p2[0]);
	fp2_mul(&t[1], &p1[1], &p2[0]);
	fp2_mul(&t[2], &p1[0], &p2[2]);
	fp2_set_zero(&t[3]);

    // ABCD = AVX_4x2x1(T2[0], T2[1], T2[2], 0)
	fp2_unvectorized_to_4x2x1(res_OA, t);

	// 2M
    // T0 = AVX_2x4x1(A1[0], A1[2])
    // T1 = AVX_2x4x1(T2[3], ABCD[1])
    // ab = avx_241_mul(T0, T1)
	fp2_mul(&t[3], &p1[1], &p2[2]);
	fp2_mul(&t[0], &p1[0], &t[3]);
	fp2_mul(&t[1], &p1[2], &t[1]);
	fp2_unvectorized_to_2x4x1(A24_12, t);

    // 8a
    //codomain = hadamard(ABCD)
	theta_had_single(res_OA, res_OA);
	
    return true;
}

void superglue_eval_pair(__m512i *imP_Q, __m512i *A_12, __m512i *xP,
		__m512i *yP, __m512i *zP, __m512i *T8x, __m512i *T8y, __m512i *T8z,
		__m512i *ab, int position, bool sign_flip,
		fp2_t dual_theta_null_square[4], fp2_t elements[6]) {
	__m512i u[NUM_LIMBS], v[NUM_LIMBS], w[NUM_LIMBS], t[2 * NUM_LIMBS];
	fp2_t p1[8];
    // 16S + 48M + 28a
    // U, V, W = four_add_components(A_12, P, T8)
	ec_add_components_4x4x2x1(u, v, w, A_12, xP, yP, zP, T8x, T8y, T8z);

    // Unpack the insane precomputation
    //position, sign_flip, dual_theta_null_square, elements = precomputation

    // 20M + 24a
    superglue_evaluation_pair(p1, p1+4, u, v, w, position, sign_flip, dual_theta_null_square, elements);

    // imP_Q = avx_421_to_811(P1, P2)
    // a, b = ab
    // J = AVX_8x1x1(b, b, a, a, b, b, a, a)
    // 8M
    // imP_Q = avx_811_mul(imP_Q, J)
	fp2_unvectorized_to_8x1x1(imP_Q, p1);

	__m512i idx0 = VSET(4, 12, 0, 8, 4, 12, 0, 8);
    __m512i idx1 = VSET(6, 14, 2, 10, 6, 14, 2, 10);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t[i] = VBLENDPERM(idx0, ab[i], ab[i]);
        t[i + NUM_LIMBS] = VBLENDPERM(idx1, ab[i], ab[i]);
    }
	fp2_mul_8x1x1(imP_Q, imP_Q, t);

	// 16a
    //imP_Q = hadamard_pair(imP_Q)
	theta_had_pair_alt(imP_Q, imP_Q);
}
