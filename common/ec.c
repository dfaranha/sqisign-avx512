#include <assert.h>
#include <stdio.h>

#include "ec.h"
#include "fp_consts.h"
#include "fp_unvec.h"
#include "fp2.h"
#include "mp.h"
#include "sqi_ec_params.h"

#include "utils.h"

void ec_point_unvectorized_to_4x4x2x1(__m512i* res_x, __m512i* res_z, ec_point_t* src)
{
    fp2_t unvec_x[4], unvec_z[4];
    for (int i = 0; i < 4; i++) {
        fp2_copy(unvec_x+i, &((src+i)->x));
        fp2_copy(unvec_z+i, &((src+i)->z));
    }
    fp2_unvectorized_to_4x2x1(res_x, unvec_x);
    fp2_unvectorized_to_4x2x1(res_z, unvec_z);
}


void ec_point_4x4x2x1_to_unvectorized(ec_point_t* res, __m512i* x, __m512i* z)
{
    fp2_t unvec_x[4], unvec_z[4];
    fp2_4x2x1_to_unvectorized(unvec_x, x);
    fp2_4x2x1_to_unvectorized(unvec_z, z);
    for (int i = 0; i < 4; i++) {
        fp2_copy(&((res+i)->x), unvec_x+i);
        fp2_copy(&((res+i)->z), unvec_z+i);
    }
}

// Assumes A24 is normalized.
void ec_double_4x4x2x1(__m512i* res_x, __m512i* res_z, __m512i* A24, __m512i* x, __m512i* z)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS], t3[NUM_LIMBS];
    __m512i s0[2*NUM_LIMBS], s1[2*NUM_LIMBS];

    fp2_4x2x1_duplicate_to_8x1x1(s0, x);
    fp2_4x2x1_duplicate_to_8x1x1(s1, z);
    fp2_mixaddsub_8x1x1(s0, s0, s1, 0x55, 0x55); // s0 = (x+z, x-z, x+z, x-z, ...)
    fp2_sqr_8x1x1(s0, s0);
    fp2_8x1x1_to_4x2x1(t0, t1, s0);

    fp2_sub_4x2x1(t2, t0, t1);
    fp2_mul_4x2x1(t3, t2, A24);
    fp2_add_4x2x1(t3, t3, t1);

    fp2_mulx2_4x2x1(res_x, res_z, t0, t1, t2, t3);
}

// Assumes A24 is normalized.
void ec_double_8x1x1x1(__m512i* res_x, __m512i* res_z, __m512i* A24, __m512i* x, __m512i* z)
{
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    fp2_add_8x1x1(t0, res_x, res_z);
    fp2_sub_8x1x1(t1, res_x, res_z);
    fp2_sqr_8x1x1(t0, t0);
    fp2_sqr_8x1x1(t1, t1);

    fp2_sub_8x1x1(t2, t0, t1);
    fp2_mul_8x1x1(res_z, t2, A24);
    fp2_add_8x1x1(res_z, res_z, t1);
    fp2_mul_8x1x1(res_z, res_z, t2);
    fp2_mul_8x1x1(res_x, t0, t1);
}

// TODO: in SQIsign jac_to_xz has more code to handle the point at infty
void ec_jac_to_xz_4x4x2x1(__m512i* x, __m512i* z, __m512i* jac_x, __m512i* jac_z)
{
    fp2_copy_4x2x1(x, jac_x);
    fp2_sqr_4x2x1(z, jac_z);
}

void ec_jac_to_ws_4x4x2x1(
    __m512i* res_x, __m512i* res_y, __m512i* res_z, __m512i* res_t, __m512i* res_A12_over_3,
    __m512i* x, __m512i* y, __m512i* z, __m512i* A12
)
{
    __m512i a[NUM_LIMBS], one[NUM_LIMBS], tmp1[NUM_LIMBS], tmp2[NUM_LIMBS];

    fp2_copy_4x2x1(res_y, y);
    fp2_copy_4x2x1(res_z, z);
    fp2_div3_4x2x1(res_A12_over_3, A12);

    //TODO: inefficient way to set one. Have dedicated function?
    fp2_setzero_4x2x1(one);
    one[0] = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    fp2_tomont_4x2x1(one, one);

    // a = 1 - A^2/3
    // res_x = x + z^2 * A/3
    fp2_sqr_4x2x1(tmp1, z);
    // TODO: Worth it to use fp2_4x2x1_duplicate_to_8x1x1?
    fp2_mulx2_4x2x1(tmp2, a, tmp1, res_A12_over_3, A12, res_A12_over_3);
    fp2_sub_4x2x1(a, one, a);
    fp2_add_4x2x1(res_x, x, tmp2);

    // res_T = a * z^4
    fp2_sqr_4x2x1(tmp1, tmp1);
    fp2_mul_4x2x1(res_t, a, tmp1);
}

void ec_ws_to_jac_4x4x2x1(
    __m512i* res_x, __m512i* res_y, __m512i* res_z,
    __m512i* x, __m512i* y, __m512i* z, __m512i* A12_over_3
)
{
    __m512i tmp[NUM_LIMBS];
    fp2_copy_4x2x1(res_y, y);
    fp2_copy_4x2x1(res_z, z);

    //res_x = x - z^2 * A/3
    fp2_sqr_4x2x1(tmp, z);
    fp2_mul_4x2x1(tmp, tmp, A12_over_3);
    fp2_sub_4x2x1(res_x, x, tmp);
}

void ec_double_ws_4x4x2x1(
    __m512i* res_x, __m512i* res_y, __m512i* res_z, __m512i* res_t,
    __m512i* x, __m512i* y, __m512i* z, __m512i* t
)
{
    __m512i xx[NUM_LIMBS], aa[NUM_LIMBS], m[NUM_LIMBS], s[NUM_LIMBS];
    __m512i u[NUM_LIMBS], t0[NUM_LIMBS], t1[NUM_LIMBS];

    fp2_sqrx2_4x2x1(xx, t0, x, y);
    fp2_add_4x2x1(aa, t0, t0);
    fp2_add_4x2x1(t0, x, aa);
    fp2_sqrx2_4x2x1(aa, t0, aa, t0);
    fp2_add_4x2x1(u, aa, aa);
    fp2_sub_4x2x1(t0, t0, xx);
    fp2_sub_4x2x1(s, t0, aa);
    fp2_add_4x2x1(t0, xx, xx);
    fp2_add_4x2x1(t0, t0, xx);
    fp2_add_4x2x1(m, t0, t);

    fp2_mulx2_4x2x1(t0, t1, m, m, y, z);
    fp2_add_4x2x1(res_z, t1, t1);
    fp2_add_4x2x1(t1, s, s);
    fp2_sub_4x2x1(res_x, t0, t1);
    fp2_sub_4x2x1(t0, s, res_x);
    fp2_mulx2_4x2x1(t0, t1, t0, m, u, t);
    fp2_sub_4x2x1(res_y, t0, u);
    fp2_add_4x2x1(res_t, t1, t1);
}
void ec_double_ws_iter_4x4x2x1(
    __m512i* res_x, __m512i* res_y, __m512i* res_z, __m512i* res_t,
    __m512i* x, __m512i* y, __m512i* z, __m512i* t, const int n
)
{
    if (n == 0) {
        fp2_copy_4x2x1(res_x, x);
        fp2_copy_4x2x1(res_y, y);
        fp2_copy_4x2x1(res_z, z);
        fp2_copy_4x2x1(res_t, t);
        return;
    }
    
    ec_double_ws_4x4x2x1(res_x, res_y, res_z, res_t, x, y, z, t);
    for (int i = 0; i < n-1; i++) {
        ec_double_ws_4x4x2x1(res_x, res_y, res_z, res_t, res_x, res_y, res_z, res_t);
    }
}

void ec_add_components_4x4x2x1(
    __m512i* res_u, __m512i* res_v, __m512i* res_w, 
    __m512i* A, __m512i* xP, __m512i* yP, __m512i* zP, 
    __m512i* xQ, __m512i* yQ, __m512i* zQ
)
{
    __m512i zQ_zP[2*NUM_LIMBS], tP_tQ[2*NUM_LIMBS], t1_t0[2*NUM_LIMBS];
    __m512i t2_t3[2*NUM_LIMBS], t4_t5[2*NUM_LIMBS], t6_t7[2*NUM_LIMBS];

    // Compute the slope dy / dx for the case P != Q
    fp2_4x2x1_to_8x1x1(zQ_zP, zQ, zP);
    fp2_4x2x1_to_8x1x1(tP_tQ, xP, xQ);
    fp2_sqr_8x1x1(t1_t0, zQ_zP);
    fp2_mul_8x1x1(t2_t3, t1_t0, tP_tQ);
    fp2_4x2x1_to_8x1x1(tP_tQ, yP, yQ);
    fp2_mul_8x1x1(t4_t5, zQ_zP, tP_tQ);
    fp2_mul_8x1x1(t4_t5, t4_t5, t1_t0);

    // t0 = t0 * t1, t1 = t4 * t5:
    for(int i = 0; i < 2*NUM_LIMBS; i++) {
        t6_t7[i] = VUNPACKLO(t1_t0[i], t4_t5[i]); // (t1, t4)
        t1_t0[i] = VUNPACKHI(t1_t0[i], t4_t5[i]); // (t0, t5)
    }
    fp2_mul_8x1x1(t6_t7, t6_t7, t1_t0); // (t0, t1)
    fp2_8x1x1_to_4x2x1(t1_t0+NUM_LIMBS, t1_t0, t6_t7);
    fp2_add_4x2x1(res_v, t1_t0, t1_t0);

    fp2_sqr_8x1x1(t4_t5, t4_t5);

    fp2_8x1x1_to_4x2x1(t2_t3, t2_t3+NUM_LIMBS, t2_t3);
    fp2_8x1x1_to_4x2x1(t4_t5, t4_t5+NUM_LIMBS, t4_t5);
    fp2_add_4x2x1(t4_t5, t4_t5, t4_t5+NUM_LIMBS);
    fp2_add_4x2x1(t4_t5+NUM_LIMBS, t2_t3, t2_t3+NUM_LIMBS);
    fp2_add_4x2x1(t2_t3, t2_t3+NUM_LIMBS, t2_t3+NUM_LIMBS);
    fp2_sub_4x2x1(t2_t3, t4_t5+NUM_LIMBS, t2_t3);

    fp2_mulx2_4x2x1(t2_t3, t1_t0, t2_t3, t2_t3, A, t1_t0+NUM_LIMBS);
    fp2_add_4x2x1(t1_t0, t1_t0, t4_t5+NUM_LIMBS);
    // TODO: Worth it to use fp2_4x2x1_duplicate_to_8x1x1 with t2_t3?
    fp2_mulx2_4x2x1(t1_t0, res_w, t1_t0, t2_t3, t2_t3, t1_t0+NUM_LIMBS);
    fp2_sub_4x2x1(res_u, t4_t5, t1_t0);
}

// ============================== 2x4x1 ======================================0

void ec_point_unvec_to_1x2x4x1(__m512i* res, ec_point_t* P)
{
    fp2_t xz[2] = { P->x, P->z };
    fp2_unvectorized_to_2x4x1(res, xz);
}

void ec_point_1x2x4x1_to_unvec(ec_point_t* res, __m512i* P)
{
    fp2_t xz[2];
    fp2_2x4x1_to_unvectorized(xz, P);
    fp2_copy(&res->x, xz);
    fp2_copy(&res->z, xz+1);
}

// TODO: Do we want to make this more efficient without a mul?
void ec_set_zero_1x2x4x1(__m512i* P)
{
    fp2_set_binary_2x4x1(P, 0b1);
}

// TODO: Do we want to make this more efficient without a mul?
// TODO maybe setzero is redundant and there is a way to init to zero?
bool ec_curve_verif_A_1x2x4x1(__m512i* A1_A2)
{
    __m512i t[NUM_LIMBS];
    fp2_setzero_2x4x1(t);
    t[0] = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    fp2_tomont_2x4x1(t, t);

    fp2_add_2x4x1(t, t, t); // t = 2
    if (fp2_is_equal_2x4x1(A1_A2, t))
        return false;

    fp2_neg_2x4x1(t, t);
    return fp2_is_equal_2x4x1(A1_A2, t) != 0;
}

// TODO: Can we not use this algorithm to create a 8x1x1 algorithm
//       for doubling 4 points in theta isog?
// Does not assume A24 has been normalized.
void ec_double_1x2x4x1(__m512i* res, __m512i* P, __m512i* A24)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS], t3[NUM_LIMBS], idx0, idx1;
    __mmask8 mask;

    // t0 = (X, X), t1 = (Z, Z)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }

    // t0 = (X + Z, X - Z)
    // t1 = t0^2
    mask = 0xf;
    fp2_mixaddsub_2x4x1(t0, t0, t1, mask);
    fp2_sqr_2x4x1(t1, t0);

    // t0 = (t1[0] + t1[1], t1[0] - t1[1])
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, t1[i]);
        t2[i] = VPERM512(idx1, t1[i]);
    }
    fp2_mixaddsub_2x4x1(t0, t0, t2, mask);

    // t2 = (A24 * t0[1], C24 * t1[1])
    idx0 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }
    fp2_mul_2x4x1(t2, t2, A24);

    // TODO: Wasting one addition here.
    // t3 = (-, t2[0] + t2[1])
    idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t3[i] = VPERM512(idx0, t2[i]);
    }
    fp2_add_2x4x1(t3, t3, t2);
    
    // res_X = t1[0] * t2[1], res_Z = t0[1] * t3[1]
    idx0 = VSET(8, 9, 10, 11, 4, 5, 6, 7);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t1[i] = VBLENDPERM(idx1, t2[i], t3[i]);
    }
    fp2_mul_2x4x1(res, t0, t1);
}

// TODO: This is not constant time, but also not used other than verification. Should we change it?
void ec_double_iter_1x2x4x1(__m512i* res, __m512i* P, __m512i* A24, const int n)
{
    if (n == 0) {
        fp2_copy_2x4x1(res, P);
        return;
    }
    
    ec_double_1x2x4x1(res, P, A24);
    for (int i = 0; i < n-1; i++) {
        ec_double_1x2x4x1(res, res, A24);
    }   
}

// Assumes n1 >= n2, inputs in 2x4x1
void ec_double_basis_pair_1x2x4x1(
    __m512i* res_B1, 
    __m512i* res_B2, 
    __m512i* B1, 
    __m512i* B2, 
    __m512i* A24_norm1,
    __m512i* A24_norm2,
    const int n1,
    const int n2
)
{
    __m512i x[2*NUM_LIMBS], z[2*NUM_LIMBS], t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], A24x8[2*NUM_LIMBS];
    __m512i idx0, idx1, idx2;
    __mmask8 mask;

    assert(n1 >= n2);

    // pair first components and second components in a 2x4x1
    // TODO: Define a gadget for this (observe same pattern for 2x2x2 except +N4x2 instead of +NUM_LIMBS)
    idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        x[i] = VBLENDPERM(idx0, B1[i], B1[i+NUM_LIMBS]);
        z[i] = VBLENDPERM(idx1, B1[i], B1[i+NUM_LIMBS]);
        x[i+NUM_LIMBS] = VBLENDPERM(idx0, B1[i+2*NUM_LIMBS], B1[i+2*NUM_LIMBS]); //wasteful
        z[i+NUM_LIMBS] = VBLENDPERM(idx1, B1[i+2*NUM_LIMBS], B1[i+2*NUM_LIMBS]);
        t0[i] = VBLENDPERM(idx0, B2[i], B2[i+NUM_LIMBS]);
        t1[i] = VBLENDPERM(idx1, B2[i], B2[i+NUM_LIMBS]);
        t0[i+NUM_LIMBS] = VBLENDPERM(idx0, B2[i+2*NUM_LIMBS], B2[i+2*NUM_LIMBS]);
        t1[i+NUM_LIMBS] = VBLENDPERM(idx1, B2[i+2*NUM_LIMBS], B2[i+2*NUM_LIMBS]);
    }
    fp2_2x4x1_to_4x2x1(x, x, x+NUM_LIMBS);
    fp2_2x4x1_to_4x2x1(x+NUM_LIMBS, t0, t0+NUM_LIMBS);
    fp2_2x4x1_to_4x2x1(z, z, z+NUM_LIMBS);
    fp2_2x4x1_to_4x2x1(z+NUM_LIMBS, t1, t1+NUM_LIMBS);
    fp2_4x2x1_to_8x1x1(x, x, x+NUM_LIMBS);
    fp2_4x2x1_to_8x1x1(z, z, z+NUM_LIMBS);

    idx0 = VSET(0, 8, 0, 8, 0, 8, 0, 8);
    idx1 = VSET(2, 10, 2, 10, 2, 10, 2, 10);
    for (int i = 0; i < NUM_LIMBS; i++) {
        A24x8[i] = VBLENDPERM(idx0, A24_norm1[i], A24_norm2[i]);
        A24x8[i+NUM_LIMBS] = VBLENDPERM(idx1, A24_norm1[i], A24_norm2[i]);
    }

    for (int i = 0; i < n2; i++) {
        ec_double_8x1x1x1(x, z, A24x8, x, z);
    }

    fp2_8x1x1_to_4x2x1(x, x+NUM_LIMBS, x);
    fp2_8x1x1_to_4x2x1(z, z+NUM_LIMBS, z);
    fp2_8x1x1_to_first_4x2x1(A24x8, A24x8);
    for (int i = 0; i < n1 - n2; i++) {
        ec_double_4x4x2x1(x, z, A24x8, x, z);
    }

    idx0 = VSET(0, 0, 1, 1, 8, 8, 9, 9);
    idx1 = VSET(2, 2, 3, 3, 10, 10, 11, 11);
    idx2 = VSET(4, 4, 5, 5, 12, 12, 13, 13);
    mask = 0x55;
    for (int i = 0; i < NUM_LIMBS; i++) {
        res_B1[i] = VBLENDPERMMASKZ(mask, x[i], z[i], idx0);
        res_B1[i + NUM_LIMBS] = VBLENDPERMMASKZ(mask, x[i], z[i], idx1);
        res_B1[i + 2*NUM_LIMBS] = VBLENDPERMMASKZ(mask, x[i], z[i], idx2);
        res_B2[i] = VBLENDPERMMASKZ(mask, x[i+NUM_LIMBS], z[i+NUM_LIMBS], idx0);
        res_B2[i + NUM_LIMBS] = VBLENDPERMMASKZ(mask, x[i+NUM_LIMBS], z[i+NUM_LIMBS], idx1);
        res_B2[i + 2*NUM_LIMBS] = VBLENDPERMMASKZ(mask, x[i+NUM_LIMBS], z[i+NUM_LIMBS], idx2);
    }
}

// TODO: Name is slightly confusing, all operations are 4x2x1
void ec_double_add_1x4x2x1(__m512i* res, __m512i* P_Q, __m512i* PmQ_A24)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS];
    __m512i idx0, idx1;
    __mmask8 mask;

    // t0 = (P.x, Q.x, P.x, Q.x)
    // t1 = (P.z, Q.z, P.z, Q.z)
    idx0 = VSET(0, 1, 4, 5, 0, 1, 4, 5);
    idx1 = VSET(2, 3, 6, 7, 2, 3, 6, 7);
    mask = 0xf;
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, P_Q[i]);
        t1[i] = VPERM512(idx1, P_Q[i]);
    }
    // t0 = (P.x + P.z, Q.x + Q.z, P.x - P.z, Q.x - Q.z)
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask);

    // t1 = (t0[0], t0[2], t0[0], t0[2])
    // t0 = (t0[0], t0[2], t0[3], t0[1])
    idx1 = VSET(0, 1, 4, 5, 6, 7, 2, 3);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t0[i] = VPERM512(idx1, t0[i]);
    }
    fp2_mul_4x2x1(t0, t0, t1);

    // t1 = (t0[0], t0[2], t0[0], t0[2])
    // t2 = (t0[1], t0[3], t0[1], t0[3])
    idx1 = VSET(2, 3, 6, 7, 2, 3, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VPERM512(idx1, t0[i]);
    }
    fp2_mixaddsub_4x2x1(t1, t1, t2, mask);

    // Moving t0[0], t1[2] to res to free them after the next mul
    idx0 = VSET(0, 1, 12, 13, 0, 0, 0, 0);
    idx1 = VSET(0, 1, 2, 3, 10, 11, 8, 9);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        res[i] = VBLENDPERM(idx1, res[i], PmQ_A24[i]);
    }

    // t0 = (t0[1], t1[2], t1[1], t1[3])
    // t1 = (PmQ_A24[3], PmQ_A24[2], t1[1], t1[3])
    idx0 = VSET(2, 3, 12, 13, 10, 11, 14, 15);
    idx1 = VSET(14, 15, 12, 13, 2, 3, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t1[i] = VBLENDPERM(idx1, t1[i], PmQ_A24[i]);
    }
    fp2_mul_4x2x1(t0, t0, t1);

    // t1 = (t0[0], t0[0], t0[2], t0[3])
    // t2 = (0    , t0[1], 0    , 0    )
    idx0 = VSET(0, 1, 0, 1, 4, 5, 6, 7);
    idx1 = VSET(0, 0, 2, 3, 0, 0, 0, 0);
    mask = 0xc;
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VPERM512MASKZ(mask, t0[i], idx1);
    }
    fp2_add_4x2x1(t0, t1, t2);

    fp2_mul_4x2x1(res, res, t0);
}

// TODO: Create a VPERM512 and VBLENDPERM for loop macro? We do it so often. Just input pattern (and len of array?)

// TODO: Simplify 8x1x1 to not have this weird interleaving pattern? I think the interleaving doesn't improve efficiency much, if at all.

void ec_double_add_pair_8x1x1(__m512i* R1_R2_S1_S2, __m512i* P1_Q1_PQ1_A24, __m512i* P2_Q2_PQ2_A24)
{
    // Note: Python stores elements in the format
    //       (P.x, P.z, Q.x, Q.z, PQ.x, PQ.z, A24.x, A24.z)
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    __m512i idx0, idx1, idx2;
    __mmask8 mask;

    // t0 = (P1.x, Q1.x, P2.x, Q2.x, P1.x, Q1.x, P2.x, Q2.x)
    //         +     +     +     +     -     -     -     -
    // t1 = (P1.z, Q1.z, P2.z, Q2.z, P1.z, Q1.z, P2.z, Q2.z)    
    idx0 = VSET(0, 4, 8, 12, 0, 4, 8, 12);
    idx1 = VSET(2, 6, 10, 14, 2, 6, 10, 14);
    mask = 0xf;
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, P1_Q1_PQ1_A24[i], P2_Q2_PQ2_A24[i]);
        t1[i] = VBLENDPERM(idx1, P1_Q1_PQ1_A24[i], P2_Q2_PQ2_A24[i]);
    }
    fp2_mixaddsub_8x1x1(t0, t0, t1, mask, mask);

    // 8x2x1 mul with four squares hidden inside
    // t1 = (t0[0], t0[4], t0[5], t0[1], t0[2], t0[6], t0[7], t0[3])
    // t0 = (t0[0], t0[4], t0[0], t0[4], t0[2], t0[6], t0[2], t0[6])
    idx0 = VSET(0, 4, 0, 4, 2, 6, 2, 6);
    idx1 = VSET(0, 4, 5, 1, 2, 6, 7, 3);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx1, t0[i]);
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    // t1 = (t0[0], t0[2], t0[4], t0[6], t0[0], t0[2], t0[4], t0[6])
    // t2 = (t0[1], t0[3], t0[5], t0[7], t0[1], t0[3], t0[5], t0[7])
    idx0 = VSET(0, 2, 4, 6, 0, 2, 4, 6);
    idx1 = VSET(1, 3, 5, 7, 1, 3, 5, 7);
    mask = 0xf;
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VPERM512(idx1, t0[i]);
    }
    fp2_mixaddsub_8x1x1(t1, t1, t2, mask, mask);

    // Moving values from t0 and t1 to res to free them after the next mul
    // res = (t0[0], t0[4], t1[4], t1[6], PQ1.z, PQ2.z, PQ1.x, PQ2.x)
    idx0 = VSET(0, 4, 12, 14, 0, 0, 0, 0);
    idx1 = VSET(0, 1, 2, 3, 11, 0, 9, 0);
    idx2 = VSET(0, 1, 2, 3, 4, 11, 6, 9);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        R1_R2_S1_S2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        R1_R2_S1_S2[i] = VBLENDPERM(idx1, R1_R2_S1_S2[i], P1_Q1_PQ1_A24[i]);
        R1_R2_S1_S2[i] = VBLENDPERM(idx2, R1_R2_S1_S2[i], P2_Q2_PQ2_A24[i]);
    }

    // t0 = (t0[1], t1[4], t1[1], t1[5], t0[5], t1[6], t1[3], t1[7])
    // t1 = (A24_1.z, A24_1.x, t1[1], t1[5], A24_2.z, A24_2.x, t1[3], t1[7])
    idx0 = VSET(1, 12, 9, 13, 5, 14, 11, 15);
    idx1 = VSET(15, 13, 1, 5, 0, 0, 3, 7);
    idx2 = VSET(0, 1, 2, 3, 15, 13, 6, 7);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t1[i] = VBLENDPERM(idx1, t1[i], P1_Q1_PQ1_A24[i]);
        t1[i] = VBLENDPERM(idx2, t1[i], P2_Q2_PQ2_A24[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    // C: (P.x, PQ.x, P.z, PQ.z, Q.x, A24.x, Q.z, A24.z)
    // P: (P.x, P.z, Q.x, Q.z, PQ.x, PQ.z, A24.x, A24.z)

    // t1 = (t0[0], t0[4], t0[0], t0[4], t0[2], t0[6], t0[3], t0[7])
    // t0 = (0    , 0    , t0[1], t0[5], 0    , 0    , 0    , 0    )

    mask = 0xc; 
    idx0 = VSET(0, 4, 0, 4, 2, 6, 3, 7);
    idx1 = VSET(0, 0, 1, 5, 0, 0, 0, 0);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t0[i] = VPERM512MASKZ(mask, t0[i], idx1);
    }
    fp2_add_8x1x1(t0, t0, t1);
    fp2_mul_8x1x1(R1_R2_S1_S2, R1_R2_S1_S2, t0);
}

// Requires normalized A coordinate
void ec_double_add_normA_1x4x2x1(__m512i* res, __m512i* P_Q, __m512i* PmQ_A)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS];
    __m512i idx0, idx1;
    __mmask8 mask0, mask1;

    // First had
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    mask0 = 0x33;
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, P_Q[i]);
        t1[i] = VPERM512(idx1, P_Q[i]);
    }
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask0);

    idx0 = VSET(0, 1, 2, 3, 2, 3, 0, 1);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mul_4x2x1(t0, t0, t1);

    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx1, t0[i]);
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask0);
    fp2_sqr_4x2x1(t1, t0);

    idx0 = VSET(0, 1, 10, 11, 12, 13, 14, 15);
    idx1 = VSET(2, 3, 12, 13, 10, 11, 8, 9);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res[i] = VBLENDPERM(idx1, t0[i], PmQ_A[i]);
        t0[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }
    fp2_mul_4x2x1(t0, t0, res);

    // TODO: Figure out how to do this last part efficiently in C:
    idx0 = VSET(0, 1, 8, 9, 12, 13, 14, 15);
    idx1 = VSET(2, 3, 10, 11, 0, 0, 0, 0);
    mask0 = 0xf; mask1 = 0xc;
    for (int i = 0; i < NUM_LIMBS; i++) {
        res[i] = VBLENDPERM(idx0, t1[i], t0[i]);
        t1[i] = VBLENDPERMMASKZ(mask0, t1[i], t0[i], idx1);
        t0[i] = VBLENDPERMMASKZ(mask1, t1[i], t0[i], idx0);
    }
    fp2_mixaddsub_4x2x1(res, res, t1, mask1);
    fp2_add_4x2x1(res, res, t0);
}

void ec_double_add_pair_normA_8x1x1(__m512i* R1_R2_S1_S2, __m512i* P1_Q1_PQ1_A, __m512i* P2_Q2_PQ2_A)
{
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    __m512i idx0, idx1, idx2;
    __mmask8 mask0, mask1;

    idx0 = VSET(0, 0, 4, 4, 8, 8, 12, 12);
    idx1 = VSET(2, 2, 6, 6, 10, 10, 14, 14);
    mask0 = 0x55;
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, P1_Q1_PQ1_A[i], P2_Q2_PQ2_A[i]);
        t1[i] = VBLENDPERM(idx1, P1_Q1_PQ1_A[i], P2_Q2_PQ2_A[i]);
    }
    fp2_mixaddsub_8x1x1(t0, t0, t1, mask0, mask0);

    idx0 = VSET(0, 1, 1, 0, 4, 5, 5, 4);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    idx0 = VSET(0, 0, 2, 2, 4, 4, 6, 6);
    idx1 = VSET(1, 1, 3, 3, 5, 5, 7, 7);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx1, t0[i]);
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_8x1x1(t0, t0, t1, mask0, mask0);
    
    fp2_sqr_8x1x1(t1, t0);

    idx0 = VSET(0, 9, 10, 11, 4, 13, 14, 15);
    idx1 = VSET(1, 13, 11, 9, 5, 0, 0, 0);
    idx2 = VSET(0, 1, 2, 3, 4, 13, 11, 9);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx1, t0[i], P1_Q1_PQ1_A[i]);
        t2[i] = VBLENDPERM(idx2, t2[i], P2_Q2_PQ2_A[i]);
        t0[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }
    fp2_mul_8x1x1(t0, t0, t2);

    // TODO: figure out how to do this efficiently in C
    idx0 = VSET(0, 4, 8, 12, 10, 14, 11, 15);
    idx1 = VSET(1, 5, 9, 13, 0, 0, 0, 0);
    mask0 = 0xf; mask1 = 0xc;
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        R1_R2_S1_S2[i] = VBLENDPERM(idx0, t1[i], t0[i]);
        t1[i] = VBLENDPERMMASKZ(mask0, t1[i], t0[i], idx1);
        t0[i] = VBLENDPERMMASKZ(mask1, t1[i], t0[i], idx0);
    }
    fp2_mixaddsub_8x1x1(R1_R2_S1_S2, R1_R2_S1_S2, t1, mask1, mask1);
    fp2_add_8x1x1(R1_R2_S1_S2, R1_R2_S1_S2, t0);
}

static void ec_scalar_mul_inner_1x4x2x1(__m512i* X0_X1, __m512i* XD_A24, const digit_t* m, const unsigned int mbits) {
    unsigned int bit;
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3);

    for (int i = mbits - 1; i >= 0; i--) {
        bit = (m[i >> LOG2RADIX] >> (i & (RADIX - 1))) & 1;

        if (bit) { // swap
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }

        ec_double_add_1x4x2x1(X0_X1, X0_X1, XD_A24);

        if (bit) { // swap
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }
    }
}

static void ec_scalar_mul_normA_inner_1x4x2x1(__m512i* X0_X1, __m512i* XD_A, const digit_t* m, int from_bit, int to_bit) {
    unsigned int bit;
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3);

    for (int i = to_bit; i >= from_bit; i--) {
        bit = (m[i >> LOG2RADIX] >> (i & (RADIX - 1))) & 1;

        if (bit) { // swap
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }

        ec_double_add_normA_1x4x2x1(X0_X1, X0_X1, XD_A);

        if (bit) { // swap
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }
    }
}

static void ec_scalar_mul_pair_normA_inner_8x1x1(__m512i* X0_X1_XD_A, __m512i* Y0_Y1_YD_A, const digit_t* m, const unsigned int mbits) {
    __m512i R1_R2_S1_S2[2*NUM_LIMBS], idx0, idx1;
    unsigned int bit;

    for (int i = mbits - 1; i >= 0; i--) {
        bit = (m[i >> LOG2RADIX] >> (i & (RADIX - 1))) & 1;

        if (bit) { // swap X0 and X1
            idx0 = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int k = 0; k < 2*NUM_LIMBS; k++) {
                X0_X1_XD_A[k] = VPERM512(idx0, X0_X1_XD_A[k]);
                Y0_Y1_YD_A[k] = VPERM512(idx0, Y0_Y1_YD_A[k]);
            }
        }

        ec_double_add_pair_normA_8x1x1(R1_R2_S1_S2, X0_X1_XD_A, Y0_Y1_YD_A);

        if (!bit) { // X0_X1 = R1_R2, Y0_Y1 = S1_S2
            idx0 = VSET(8, 1, 10, 3, 12, 5, 14, 7);
            idx1 = VSET(9, 1, 11, 3, 13, 5, 15, 7);
            for (int k = 0; k < 2*NUM_LIMBS; k++) {
                X0_X1_XD_A[k] = VBLENDPERM(idx0, X0_X1_XD_A[k], R1_R2_S1_S2[k]);
                Y0_Y1_YD_A[k] = VBLENDPERM(idx1, Y0_Y1_YD_A[k], R1_R2_S1_S2[k]);
            }
            
        } else { // swap
            idx0 = VSET(12, 1, 14, 3, 8, 5, 10, 7);
            idx1 = VSET(13, 1, 15, 3, 9, 5, 11, 7);
            for (int k = 0; k < 2*NUM_LIMBS; k++) {
                X0_X1_XD_A[k] = VBLENDPERM(idx0, X0_X1_XD_A[k], R1_R2_S1_S2[k]);
                Y0_Y1_YD_A[k] = VBLENDPERM(idx1, Y0_Y1_YD_A[k], R1_R2_S1_S2[k]);
            }
        }
    }
}

void ec_scalar_mul_1x2x4x1(__m512i* res, __m512i* P, __m512i* A24, const digit_t* m, const unsigned int mbits)
{
    __m512i X0_X1[NUM_LIMBS], XD_A24[NUM_LIMBS];
    if (mbits == 0) {
        fp2_copy_2x4x1(res, P);
        return;
    }

    // X0 = P, X1 = 0, XD = P
    ec_set_zero_1x2x4x1(X0_X1);
    fp2_2x4x1_to_4x2x1(X0_X1, X0_X1, P);
    fp2_2x4x1_to_4x2x1(XD_A24, P, A24);

    ec_scalar_mul_inner_1x4x2x1(X0_X1, XD_A24, m, mbits);

    // Ignore X1 and output X0
    fp2_4x2x1_to_2x4x1(res, XD_A24, X0_X1);
}

void ec_scalar_mul_normA_1x2x4x1(__m512i* res, __m512i* P, __m512i* A, const digit_t* m, const unsigned int mbits)
{
    __m512i X0_X1[NUM_LIMBS], XD_A[NUM_LIMBS];
    if (mbits == 0) {
        fp2_copy_2x4x1(res, P);
        return;
    }

    // X0 = 0, X1 = P, XD = P
    ec_set_zero_1x2x4x1(X0_X1);
    fp2_2x4x1_to_4x2x1(X0_X1, X0_X1, P);
    fp2_2x4x1_to_4x2x1(XD_A, P, A);

    ec_scalar_mul_normA_inner_1x4x2x1(X0_X1, XD_A, m, 0, mbits-1);

    // Ignore X1 and output X0
    fp2_4x2x1_to_2x4x1(res, XD_A, X0_X1);
}

static void ec_ladder_3pt_inner_1x4x2x1(__m512i* X0_X1, __m512i* XD_A24, const digit_t* m)
{
    __m512i X0_XD[NUM_LIMBS], X1_A24[NUM_LIMBS];
    __m512i idx_first1_second2 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
    __m512i idx_first1_first2 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    __m512i idx_second1_second2 = VSET(4, 5, 6, 7, 12, 13, 14, 15);

    digit_t t;
    for (int i = 0; i < NWORDS_ORDER; i++) {
        t = 1;
        for (int j = 0; j < 64; j++) {
            uint64_t bit_set = (t & m[i]);
            if (bit_set) {
                ec_double_add_1x4x2x1(X0_X1, X0_X1, XD_A24);
            } else {
                // Swap X1 and XD
                for (int k = 0; k < NUM_LIMBS; k++) {
                    X0_XD[k] = VBLENDPERM(idx_first1_first2, X0_X1[k], XD_A24[k]);
                    X1_A24[k] = VBLENDPERM(idx_second1_second2, X0_X1[k], XD_A24[k]);
                }
                
                ec_double_add_1x4x2x1(X0_XD, X0_XD, X1_A24);

                // Update X0 and XD
                for (int k = 0; k < NUM_LIMBS; k++) {
                    X0_X1[k] = VBLENDPERM(idx_first1_second2, X0_XD[k], X0_X1[k]);
                    XD_A24[k] = VBLENDPERM(idx_second1_second2, X0_XD[k], XD_A24[k]);
                }
            }
            t <<= 1;
        }
    }
}

static void ec_ladder_3pt_normA_inner_1x4x2x1(__m512i* X0_X1, __m512i* XD_A, const digit_t* m)
{
    __m512i X0_XD[NUM_LIMBS], X1_A[NUM_LIMBS];
     __m512i idx_first1_second2 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
    __m512i idx_first1_first2 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    __m512i idx_second1_second2 = VSET(4, 5, 6, 7, 12, 13, 14, 15);

    digit_t t;
    for (int i = 0; i < NWORDS_ORDER; i++) {
        t = 1;
        for (int j = 0; j < 64; j++) {
            uint64_t bit_set = (t & m[i]);
            if (bit_set) {
                ec_double_add_normA_1x4x2x1(X0_X1, X0_X1, XD_A);
            } else {
                // Swap X1 and XD
                for (int k = 0; k < NUM_LIMBS; k++) {
                    X0_XD[k] = VBLENDPERM(idx_first1_first2, X0_X1[k], XD_A[k]);
                    X1_A[k] = VBLENDPERM(idx_second1_second2, X0_X1[k], XD_A[k]);
                }
                
                ec_double_add_normA_1x4x2x1(X0_XD, X0_XD, X1_A);

                // Update X0 and XD
                for (int k = 0; k < NUM_LIMBS; k++) {
                    X0_X1[k] = VBLENDPERM(idx_first1_second2, X0_XD[k], X0_X1[k]);
                    XD_A[k] = VBLENDPERM(idx_second1_second2, X0_XD[k], XD_A[k]);
                }
            }
            t <<= 1;
        }
    }
}

bool ec_ladder_3pt_1x2x4x1(__m512i* res, __m512i* A24, __m512i* basis, const digit_t* m)
{
    __m512i X0_X1[NUM_LIMBS], XD_A24[NUM_LIMBS];

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x4x1(basis + 2*NUM_LIMBS)) {
        return false;
    }

    // X0 = Q, X1 = P, XD = PmQ
    fp2_2x4x1_to_4x2x1(X0_X1, basis + NUM_LIMBS, basis);
    fp2_2x4x1_to_4x2x1(XD_A24, basis + 2*NUM_LIMBS, A24);

    ec_ladder_3pt_inner_1x4x2x1(X0_X1, XD_A24, m);

    // We output X1 and ignore X0
    fp2_4x2x1_to_2x4x1(XD_A24, res, X0_X1);

    return true;
}

bool ec_ladder_3pt_normA_1x2x4x1(__m512i* res, __m512i* A, __m512i* basis, const digit_t *m)
{
    __m512i X0_X1[NUM_LIMBS], XD_A[NUM_LIMBS];

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x4x1(basis+2*NUM_LIMBS)) {
        return false;
    }

    // X0 = Q, X1 = P, XD = PmQ
    fp2_2x4x1_to_4x2x1(X0_X1, basis+NUM_LIMBS, basis);
    fp2_2x4x1_to_4x2x1(XD_A, basis+2*NUM_LIMBS, A);

    ec_ladder_3pt_normA_inner_1x4x2x1(X0_X1, XD_A, m);

    // We output X1 and ignore X0
    fp2_4x2x1_to_2x4x1(XD_A, res, X0_X1);

    return true;
}

/**
 * Internal helper function for ec_biscalar mul.
 * Given x(P), x(Q) and x(P-Q) it computes x(P+Q). It then sets x(Q), x(P-Q) := x(P+Q), x(Q).
 * A24 is not needed, but is only an artifact of how PmQ is stored in the biscalar ladder.
 * @param P_Q P and Q in 4x2x1
 * @param PmQ_A24 PmQ and A24 in 4x2x1
 */
static void ec_diff_add_1x2x4x1(__m512i* P_Q, __m512i* PmQ_A24)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], idx0, idx1;
    __mmask8 mask;

    // t0 = (P.x, P.x, Q.x, Q.x)
    // t1 = (P.z, P.z, Q.z, Q.z)
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, P_Q[i]);
        t1[i] = VPERM512(idx1, P_Q[i]);
    }
    // t0 = (P.x + P.z, P.x - P.z, Q.x + Q.z, Q.x - Q.z)
    mask = 0x33;
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask);

    // t1 = (t0[3], t0[2])
    // t0 = (t0[0], t0[1])
    idx0 = VSET(6, 6, 7, 7, 4, 4, 5, 5);
    idx1 = VSET(0, 0, 1, 1, 2, 2, 3, 3);
    mask = 0x55;
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512MASKZ(mask, t0[i], idx0);
        t0[i] = VPERM512MASKZ(mask, t0[i], idx1);
    }
    fp2_mul_2x4x1(t0, t0, t1);

    // t1 = (t0[1], t0[1])
    // t0 = (t0[0], t0[0])
    idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t0[i] = VPERM512(idx1, t0[i]);
    }
    mask = 0xf;
    fp2_mixaddsub_2x4x1(t0, t0, t1, mask);
    fp2_sqr_2x4x1(t0, t0);

    // t1 = (PmQ.z, PmQ.x)
    idx0 = VSET(2, 2, 3, 3, 0, 0, 1, 1);
    mask = 0x55;
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512MASKZ(mask, PmQ_A24[i], idx0);
    }
    fp2_mul_2x4x1(t0, t0, t1); // Output t0 = ((P+Q).x, (P+Q).z)

    // Updating input
    // PmQ_A24 = (Q, A24), P_Q = (P, P+Q)
    idx0 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    idx1 = VSET(0, 1, 2, 3, 8, 10, 12, 14);
    for (int i = 0; i < NUM_LIMBS; i++) {
        PmQ_A24[i] = VBLENDPERM(idx0, P_Q[i], PmQ_A24[i]);
        P_Q[i] = VBLENDPERM(idx1, P_Q[i], t0[i]);
    }
}

// TODO: Can we optimize ec_double_add to modify in place?
// TODO: At the moment I don't take the number of bits as input. 
//       Potentially, the scalars could be much smaller than NWORDS_ORDER.
// TODO: Some swaps are unecessary with a different arrangement of inputs.
bool ec_biscalar_mul_1x2x4x1(
    __m512i* res, 
    const digit_t *scalarP,
    const digit_t *scalarQ,
    __m512i* A24,
    __m512i* basis
)
{
    __m512i X0_X1[NUM_LIMBS], XD_A24[NUM_LIMBS], t0[NUM_LIMBS], t1[NUM_LIMBS];
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3), idx0, idx1;
    digit_t s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER], t_arr[NWORDS_ORDER];
    digit_t* s0 = s0_arr, *s1 = s1_arr, *t = t_arr;

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x4x1(basis + 2*NUM_LIMBS)) {
        return false;
    }

    fp2_2x4x1_to_4x2x1(X0_X1, basis, basis + NUM_LIMBS);
    fp2_2x4x1_to_4x2x1(XD_A24, basis + 2*NUM_LIMBS, A24);
    mp_copy(s0, scalarP, NWORDS_ORDER);
    mp_copy(s1, scalarQ, NWORDS_ORDER);

    while (!mp_is_zero(s0, NWORDS_ORDER)) {
        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t* tmp = s0;
            s0 = s1;
            s1 = tmp;
            
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }
        // Now s0 <= s1

        // t = 4 * s0
        mp_copy(t, s0, NWORDS_ORDER);
        digit_t overflow_bits = t[NWORDS_ORDER - 1] >> (RADIX-2);
        mp_shiftl(t, 2, NWORDS_ORDER); 
        bool s0_even = mp_is_even(s0, NWORDS_ORDER);
        bool s1_even = mp_is_even(s1, NWORDS_ORDER);

        // if the shift overflows, 4*s0 is definitely larger than s1.
        if (overflow_bits || mp_compare(s1, t, NWORDS_ORDER) <= 0) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_diff_add_1x2x4x1(X0_X1, XD_A24);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s0_even == s1_even) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);
            mp_shiftr(s1, 1, NWORDS_ORDER);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_double_add_1x4x2x1(X0_X1, X0_X1, XD_A24);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s1_even) {
            mp_shiftr(s1, 1, NWORDS_ORDER);

            idx0 = VSET(4, 5, 6, 7, 8, 9, 10, 11); // second first
            idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            for (int k = 0; k < NUM_LIMBS; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A24[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A24[k]);
            }
            ec_double_add_1x4x2x1(t0, t0, t1);

            // Update X1 and XD
            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VBLENDPERM(idx0, X0_X1[k], t0[k]);
                XD_A24[k] = VBLENDPERM(idx1, t0[k], XD_A24[k]);
            }
        } else {
            mp_shiftr(s0, 1, NWORDS_ORDER);

            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A24[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A24[k]);
            }
            ec_double_add_1x4x2x1(t0, t0, t1);

            // Update X0 and XD
            idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VBLENDPERM(idx0, t0[k], X0_X1[k]);
                XD_A24[k] = VBLENDPERM(idx1, t0[k], XD_A24[k]);
            }
        }
    }

    // We ignore X0, and continue with X1 in 2x4x1
    fp2_4x2x1_to_2x4x1(XD_A24, X0_X1, X0_X1);

    while (mp_is_even(s1, NWORDS_ORDER)) {
        mp_shiftr(s1, 1, NWORDS_ORDER);
        ec_double_1x2x4x1(X0_X1, X0_X1, A24);
    }
    unsigned int s1_nbits = mp_bitlength(s1, NWORDS_ORDER);

    // res = [s_1]*X1
    ec_scalar_mul_1x2x4x1(res, X0_X1, A24, s1, s1_nbits);

    return true;
}

bool ec_biscalar_mul_normA_1x2x4x1(
    __m512i* res, 
    const digit_t *scalarP,
    const digit_t *scalarQ,
    __m512i* A,
    __m512i* A24,
    __m512i* basis
)
{
    __m512i X0_X1[NUM_LIMBS], XD_A[NUM_LIMBS], t0[NUM_LIMBS], t1[NUM_LIMBS];
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3), idx0, idx1;
    digit_t s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER], t_arr[NWORDS_ORDER];
    digit_t* s0 = s0_arr, *s1 = s1_arr, *t = t_arr;

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x4x1(basis + 2*NUM_LIMBS)) {
        return false;
    }

    fp2_2x4x1_to_4x2x1(X0_X1, basis, basis + NUM_LIMBS);
    fp2_2x4x1_to_4x2x1(XD_A, basis + 2*NUM_LIMBS, A);
    
    mp_copy(s0, scalarP, NWORDS_ORDER);
    mp_copy(s1, scalarQ, NWORDS_ORDER);

    while (!mp_is_zero(s0, NWORDS_ORDER)) {
        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t* tmp = s0;
            s0 = s1;
            s1 = tmp;
            
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }
        // Now s0 <= s1

        // t = 4 * s0
        mp_copy(t, s0, NWORDS_ORDER);
        digit_t overflow_bits = t[NWORDS_ORDER - 1] >> (RADIX-2);
        mp_shiftl(t, 2, NWORDS_ORDER); 
        bool s0_even = mp_is_even(s0, NWORDS_ORDER);
        bool s1_even = mp_is_even(s1, NWORDS_ORDER);

        // if the shift overflows, 4*s0 is definitely larger than s1.
        if (overflow_bits || mp_compare(s1, t, NWORDS_ORDER) <= 0) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_diff_add_1x2x4x1(X0_X1, XD_A);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s0_even == s1_even) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);
            mp_shiftr(s1, 1, NWORDS_ORDER);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_double_add_normA_1x4x2x1(X0_X1, X0_X1, XD_A);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s1_even) {
            mp_shiftr(s1, 1, NWORDS_ORDER);

            idx0 = VSET(4, 5, 6, 7, 8, 9, 10, 11); // second first
            idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            for (int k = 0; k < NUM_LIMBS; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A[k]);
            }
            ec_double_add_normA_1x4x2x1(t0, t0, t1);

            // Update X1 and XD
            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VBLENDPERM(idx0, X0_X1[k], t0[k]);
                XD_A[k] = VBLENDPERM(idx1, t0[k], XD_A[k]);
            }
        } else {
            mp_shiftr(s0, 1, NWORDS_ORDER);

            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A[k]);
            }
            ec_double_add_normA_1x4x2x1(t0, t0, t1);

            // Update X0 and XD
            idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VBLENDPERM(idx0, t0[k], X0_X1[k]);
                XD_A[k] = VBLENDPERM(idx1, t0[k], XD_A[k]);
            }
        }
    }

    // We ignore X0, and continue with X1 in 2x4x1
    fp2_4x2x1_to_2x4x1(XD_A, X0_X1, X0_X1);

    while (mp_is_even(s1, NWORDS_ORDER)) {
        mp_shiftr(s1, 1, NWORDS_ORDER);
        ec_double_1x2x4x1(X0_X1, X0_X1, A24);
    }
    unsigned int s1_nbits = mp_bitlength(s1, NWORDS_ORDER);

    // res = [s_1]*X1
    ec_scalar_mul_normA_1x2x4x1(res, X0_X1, A, s1, s1_nbits);

    return true;
}

static void ec_biscalar_mul_pair_inner_8x1x1(
    __m512i* X0_X1_XD_A24,
    __m512i* Y0_Y1_YD_A24,
    digit_t** r0_ptr,
    digit_t** r1_ptr,
    digit_t** s0_ptr,
    digit_t** s1_ptr
)
{
    __m512i R1_R2_S1_S2[2*NUM_LIMBS], idx;
    __m512i *P1_Q1_PQ1_A24 = X0_X1_XD_A24 + 2*NUM_LIMBS, *P2_Q2_PQ2_A24 = Y0_Y1_YD_A24 + 2*NUM_LIMBS;
    digit_t *r0 = *r0_ptr, *r1 = *r1_ptr, *s0 = *s0_ptr, *s1 = *s1_ptr;
    int case_r, case_s;

    while (true) {
        if (mp_compare(r0, r1, NWORDS_ORDER) > 0) { // r0 > r1
            digit_t *tmp = r0;
            r0 = r1;
            r1 = tmp;

            // swapping X0 and X1
            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                X0_X1_XD_A24[i] = VPERM512(idx, X0_X1_XD_A24[i]);
            }
        }

        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t *tmp = s0;
            s0 = s1;
            s1 = tmp;

            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                Y0_Y1_YD_A24[i] = VPERM512(idx, Y0_Y1_YD_A24[i]);
            }
        }

        bool r0_even = mp_is_even(r0, NWORDS_ORDER);
        bool r1_even = mp_is_even(r1, NWORDS_ORDER);

        if (r0_even == r1_even) {
            case_r = 0;
            mp_sub(r1, r1, r0, NWORDS_ORDER);
            mp_shiftr(r1, 1, NWORDS_ORDER);

            // P1 = X1, Q1 = X0, PQ1 = XD
            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P1_Q1_PQ1_A24[i] = VPERM512(idx, X0_X1_XD_A24[i]);
            }
        } else if (r1_even) {
            case_r = 1;
            mp_shiftr(r1, 1, NWORDS_ORDER);

            // P1 = X1, Q1 = XD, PQ1 = X0
            idx = VSET(4, 0, 6, 2, 1, 5, 3, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P1_Q1_PQ1_A24[i] = VPERM512(idx, X0_X1_XD_A24[i]);
            }
        } else {
            case_r = 2;
            mp_shiftr(r0, 1, NWORDS_ORDER);

            // P1 = X0, Q1 = XD, PQ1 = X1
            idx = VSET(0, 4, 2, 6, 1, 5, 3, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P1_Q1_PQ1_A24[i] = VPERM512(idx, X0_X1_XD_A24[i]);
            }
        }

        bool s0_even = mp_is_even(s0, NWORDS_ORDER);
        bool s1_even = mp_is_even(s1, NWORDS_ORDER);

        if (s0_even == s1_even) {
            case_s = 0;
            mp_sub(s1, s1, s0, NWORDS_ORDER);
            mp_shiftr(s1, 1, NWORDS_ORDER);

            // P2 = Y1, Q2 = Y0, PQ2 = YD
            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P2_Q2_PQ2_A24[i] = VPERM512(idx, Y0_Y1_YD_A24[i]);
            }
        } else if(s1_even) {
            case_s = 1;
            mp_shiftr(s1, 1, NWORDS_ORDER);

            // P2 = Y1, Q2 = YD, PQ2 = Y0
            idx = VSET(4, 0, 6, 2, 1, 5, 3, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P2_Q2_PQ2_A24[i] = VPERM512(idx, Y0_Y1_YD_A24[i]);
            }
        } else {
            case_s = 2;
            mp_shiftr(s0, 1, NWORDS_ORDER);

            // P2 = Y0, Q2 = YD, PQ2 = Y1
            idx = VSET(0, 4, 2, 6, 1, 5, 3, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P2_Q2_PQ2_A24[i] = VPERM512(idx, Y0_Y1_YD_A24[i]);
            }
        }

        ec_double_add_pair_8x1x1(R1_R2_S1_S2, P1_Q1_PQ1_A24, P2_Q2_PQ2_A24);

        if (case_r == 0) {
            // X0 = R2, X1 = R1
            idx = VSET(4, 9, 6, 11, 0, 13, 2, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                X0_X1_XD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A24[i]);
            }
        } else if (case_r == 1) {
            // X1 = R1, XD = R2
            idx = VSET(8, 4, 10, 6, 0, 13, 2, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                X0_X1_XD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A24[i]);
            }
        } else if (case_r == 2) {
            // X0 = R1, XD = R2
            idx = VSET(0, 4, 2, 6, 12, 13, 14, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                X0_X1_XD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A24[i]);
            }
        }

        if (case_s == 0) {
            // Y0 = S2, Y1 = S1
            idx = VSET(5, 9, 7, 11, 1, 13, 3, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                Y0_Y1_YD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], Y0_Y1_YD_A24[i]);
            }
        } else if (case_s == 1) {
            // Y1 = S1, YD = S2
            idx = VSET(8, 5, 10, 7, 1, 13, 3, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                Y0_Y1_YD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], Y0_Y1_YD_A24[i]);
            }
        } else if (case_s == 2) {
            // Y0 = S1, YD = S2
            idx = VSET(1, 5, 3, 7, 12, 13, 14, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                Y0_Y1_YD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], Y0_Y1_YD_A24[i]);
            }
        }

        // If either r0 = 0 or s0 = 0, we must break the chain
        if (mp_is_zero(r0, NWORDS_ORDER) || mp_is_zero(s0, NWORDS_ORDER)) {
            break;
        }
    }

    *r0_ptr = r0;
    *r1_ptr = r1;
    *s0_ptr = s0;
    *s1_ptr = s1;
}

static void ec_biscalar_mul_pair_normA_inner_8x1x1(
    __m512i* X0_X1_XD_A,
    __m512i* Y0_Y1_YD_A,
    digit_t** r0_ptr,
    digit_t** r1_ptr,
    digit_t** s0_ptr,
    digit_t** s1_ptr
)
{
    __m512i R1_R2_S1_S2[2*NUM_LIMBS], idx;
    __m512i *P1_Q1_PQ1_A = X0_X1_XD_A + 2*NUM_LIMBS, *P2_Q2_PQ2_A = Y0_Y1_YD_A + 2*NUM_LIMBS;
    digit_t *r0 = *r0_ptr, *r1 = *r1_ptr, *s0 = *s0_ptr, *s1 = *s1_ptr;
    int case_r, case_s;

    while (true) {
        if (mp_compare(r0, r1, NWORDS_ORDER) > 0) { // r0 > r1
            digit_t *tmp = r0;
            r0 = r1;
            r1 = tmp;

            // swapping X0 and X1
            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                X0_X1_XD_A[i] = VPERM512(idx, X0_X1_XD_A[i]);
            }
        }

        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t *tmp = s0;
            s0 = s1;
            s1 = tmp;

            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                Y0_Y1_YD_A[i] = VPERM512(idx, Y0_Y1_YD_A[i]);
            }
        }

        bool r0_even = mp_is_even(r0, NWORDS_ORDER);
        bool r1_even = mp_is_even(r1, NWORDS_ORDER);

        if (r0_even == r1_even) {
            case_r = 0;
            mp_sub(r1, r1, r0, NWORDS_ORDER);
            mp_shiftr(r1, 1, NWORDS_ORDER);

            // P1 = X1, Q1 = X0, PQ1 = XD
            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P1_Q1_PQ1_A[i] = VPERM512(idx, X0_X1_XD_A[i]);
            }
        } else if (r1_even) {
            case_r = 1;
            mp_shiftr(r1, 1, NWORDS_ORDER);

            // P1 = X1, Q1 = XD, PQ1 = X0
            idx = VSET(4, 0, 6, 2, 1, 5, 3, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P1_Q1_PQ1_A[i] = VPERM512(idx, X0_X1_XD_A[i]);
            }
        } else {
            case_r = 2;
            mp_shiftr(r0, 1, NWORDS_ORDER);

            // P1 = X0, Q1 = XD, PQ1 = X1
            idx = VSET(0, 4, 2, 6, 1, 5, 3, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P1_Q1_PQ1_A[i] = VPERM512(idx, X0_X1_XD_A[i]);
            }
        }

        bool s0_even = mp_is_even(s0, NWORDS_ORDER);
        bool s1_even = mp_is_even(s1, NWORDS_ORDER);

        if (s0_even == s1_even) {
            case_s = 0;
            mp_sub(s1, s1, s0, NWORDS_ORDER);
            mp_shiftr(s1, 1, NWORDS_ORDER);

            // P2 = Y1, Q2 = Y0, PQ2 = YD
            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P2_Q2_PQ2_A[i] = VPERM512(idx, Y0_Y1_YD_A[i]);
            }
        } else if(s1_even) {
            case_s = 1;
            mp_shiftr(s1, 1, NWORDS_ORDER);

            // P2 = Y1, Q2 = YD, PQ2 = Y0
            idx = VSET(4, 0, 6, 2, 1, 5, 3, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P2_Q2_PQ2_A[i] = VPERM512(idx, Y0_Y1_YD_A[i]);
            }
        } else {
            case_s = 2;
            mp_shiftr(s0, 1, NWORDS_ORDER);

            // P2 = Y0, Q2 = YD, PQ2 = Y1
            idx = VSET(0, 4, 2, 6, 1, 5, 3, 7);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                P2_Q2_PQ2_A[i] = VPERM512(idx, Y0_Y1_YD_A[i]);
            }
        }

        ec_double_add_pair_normA_8x1x1(R1_R2_S1_S2, P1_Q1_PQ1_A, P2_Q2_PQ2_A);

        if (case_r == 0) {
            // X0 = R2, X1 = R1
            idx = VSET(4, 9, 6, 11, 0, 13, 2, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                X0_X1_XD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A[i]);
            }
        } else if (case_r == 1) {
            // X1 = R1, XD = R2
            idx = VSET(8, 4, 10, 6, 0, 13, 2, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                X0_X1_XD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A[i]);
            }
        } else if (case_r == 2) {
            // X0 = R1, XD = R2
            idx = VSET(0, 4, 2, 6, 12, 13, 14, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                X0_X1_XD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A[i]);
            }
        }

        if (case_s == 0) {
            // Y0 = S2, Y1 = S1
            idx = VSET(5, 9, 7, 11, 1, 13, 3, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                Y0_Y1_YD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], Y0_Y1_YD_A[i]);
            }
        } else if (case_s == 1) {
            // Y1 = S1, YD = S2
            idx = VSET(8, 5, 10, 7, 1, 13, 3, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                Y0_Y1_YD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], Y0_Y1_YD_A[i]);
            }
        } else if (case_s == 2) {
            // Y0 = S1, YD = S2
            idx = VSET(1, 5, 3, 7, 12, 13, 14, 15);
            for (int i = 0; i < 2*NUM_LIMBS; i++) {
                Y0_Y1_YD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], Y0_Y1_YD_A[i]);
            }
        }

        // If either r0 = 0 or s0 = 0, we must break the chain
        if (mp_is_zero(r0, NWORDS_ORDER) || mp_is_zero(s0, NWORDS_ORDER)) {
            break;
        }
    }

    *r0_ptr = r0;
    *r1_ptr = r1;
    *s0_ptr = s0;
    *s1_ptr = s1;
}

bool ec_biscalar_mul_pair_1x2x4x1(
    __m512i* res1,
    __m512i* res2,
    scalar_mtx_2x2_t *mat,
    __m512i* A24,
    __m512i* basis
)
{
    digit_t r0_arr[NWORDS_ORDER], r1_arr[NWORDS_ORDER], s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER];
    digit_t *r0 = r0_arr, *r1 = r1_arr, *s0 = s0_arr, *s1 = s1_arr; 
    __m512i X0_X1_XD_A24[4*NUM_LIMBS], Y0_Y1_YD_A24[4*NUM_LIMBS]; // Need 32 consecutive vectors for single biscalars at the end

    // 8x1x1 layout = (X0.x, XD.x, X0.z, XD.z, X1.x, A24.x, X1.z, A24.z)
    fp2_2x4x1_to_4x2x1(X0_X1_XD_A24, basis, basis + NUM_LIMBS);
    fp2_2x4x1_to_4x2x1(X0_X1_XD_A24 + NUM_LIMBS, basis + 2*NUM_LIMBS, A24);
    fp2_4x2x1_to_8x1x1(X0_X1_XD_A24, X0_X1_XD_A24, X0_X1_XD_A24 + NUM_LIMBS);
    fp2_copy_8x1x1(Y0_Y1_YD_A24, X0_X1_XD_A24);

    mp_copy(r0, (*mat)[0][0], NWORDS_ORDER);
    mp_copy(r1, (*mat)[1][0], NWORDS_ORDER);
    mp_copy(s0, (*mat)[0][1], NWORDS_ORDER);
    mp_copy(s1, (*mat)[1][1], NWORDS_ORDER);

    ec_biscalar_mul_pair_inner_8x1x1(X0_X1_XD_A24, Y0_Y1_YD_A24, &r0, &r1, &s0, &s1);

    fp2_8x1x1_to_4x2x1(X0_X1_XD_A24, X0_X1_XD_A24 + 2*NUM_LIMBS, X0_X1_XD_A24);
    fp2_4x2x1_to_2x4x1(X0_X1_XD_A24, X0_X1_XD_A24 + NUM_LIMBS, X0_X1_XD_A24);
    fp2_4x2x1_to_2x4x1(X0_X1_XD_A24 + 2*NUM_LIMBS, X0_X1_XD_A24 + 3*NUM_LIMBS, X0_X1_XD_A24 + 2*NUM_LIMBS);
    fp2_8x1x1_to_4x2x1(Y0_Y1_YD_A24, Y0_Y1_YD_A24 + 2*NUM_LIMBS, Y0_Y1_YD_A24);
    fp2_4x2x1_to_2x4x1(Y0_Y1_YD_A24, Y0_Y1_YD_A24 + NUM_LIMBS, Y0_Y1_YD_A24);
    fp2_4x2x1_to_2x4x1(Y0_Y1_YD_A24 + 2*NUM_LIMBS, Y0_Y1_YD_A24 + 3*NUM_LIMBS, Y0_Y1_YD_A24 + 2*NUM_LIMBS);

    bool succ1 = ec_biscalar_mul_1x2x4x1(res1, r0, r1, A24, X0_X1_XD_A24);
    bool succ2 = ec_biscalar_mul_1x2x4x1(res2, s0, s1, A24, Y0_Y1_YD_A24);

    return succ1 && succ2;
}

bool ec_biscalar_mul_pair_normA_1x2x4x1(
    __m512i* res1,
    __m512i* res2,
    scalar_mtx_2x2_t *mat,
    __m512i* A,
    __m512i* A24,
    __m512i* basis
)
{
    digit_t r0_arr[NWORDS_ORDER], r1_arr[NWORDS_ORDER], s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER];
    digit_t *r0 = r0_arr, *r1 = r1_arr, *s0 = s0_arr, *s1 = s1_arr; 
    __m512i X0_X1_XD_A[4*NUM_LIMBS], Y0_Y1_YD_A[4*NUM_LIMBS]; // Need consecutive vectors for single biscalars at the end

    // 8x1x1 layout = (X0.x, XD.x, X0.z, XD.z, X1.x, A.x, X1.z, -)
    fp2_2x4x1_to_4x2x1(X0_X1_XD_A, basis, basis + NUM_LIMBS);
    fp2_2x4x1_to_4x2x1(X0_X1_XD_A + NUM_LIMBS, basis + 2*NUM_LIMBS, A);
    fp2_4x2x1_to_8x1x1(X0_X1_XD_A, X0_X1_XD_A, X0_X1_XD_A + NUM_LIMBS);
    fp2_copy_8x1x1(Y0_Y1_YD_A, X0_X1_XD_A);

    mp_copy(r0, (*mat)[0][0], NWORDS_ORDER);
    mp_copy(r1, (*mat)[1][0], NWORDS_ORDER);
    mp_copy(s0, (*mat)[0][1], NWORDS_ORDER);
    mp_copy(s1, (*mat)[1][1], NWORDS_ORDER);

    ec_biscalar_mul_pair_normA_inner_8x1x1(X0_X1_XD_A, Y0_Y1_YD_A, &r0, &r1, &s0, &s1);

    fp2_8x1x1_to_4x2x1(X0_X1_XD_A, X0_X1_XD_A + 2*NUM_LIMBS, X0_X1_XD_A);
    fp2_4x2x1_to_2x4x1(X0_X1_XD_A, X0_X1_XD_A + NUM_LIMBS, X0_X1_XD_A);
    fp2_4x2x1_to_2x4x1(X0_X1_XD_A + 2*NUM_LIMBS, X0_X1_XD_A + 3*NUM_LIMBS, X0_X1_XD_A + 2*NUM_LIMBS);
    fp2_8x1x1_to_4x2x1(Y0_Y1_YD_A, Y0_Y1_YD_A + 2*NUM_LIMBS, Y0_Y1_YD_A);
    fp2_4x2x1_to_2x4x1(Y0_Y1_YD_A, Y0_Y1_YD_A + NUM_LIMBS, Y0_Y1_YD_A);
    fp2_4x2x1_to_2x4x1(Y0_Y1_YD_A + 2*NUM_LIMBS, Y0_Y1_YD_A + 3*NUM_LIMBS, Y0_Y1_YD_A + 2*NUM_LIMBS);

    bool succ1 = ec_biscalar_mul_normA_1x2x4x1(res1, r0, r1, A, A24, X0_X1_XD_A);
    bool succ2 = ec_biscalar_mul_normA_1x2x4x1(res2, s0, s1, A, A24, Y0_Y1_YD_A);

    return succ1 && succ2;
}

// Debugging functions

bool test_point_order_twof_1x2x4x1(__m512i* P, __m512i* A24, unsigned int f)
{
    __m512i Q[NUM_LIMBS];
    fp2_copy_2x4x1(Q, P);

    int z_is_zero = fp2_is_zero_2x4x1(Q) & 0x2;
    if (z_is_zero || f == 0) {
        return f == 0 && z_is_zero;
    }

    ec_double_iter_1x2x4x1(Q, Q, A24, f-1);
    z_is_zero = fp2_is_zero_2x4x1(Q) & 0x2;
    if (z_is_zero) {
        return false;
    }

    ec_double_1x2x4x1(Q, Q, A24);
    z_is_zero = fp2_is_zero_2x4x1(Q) & 0x2;
    return z_is_zero;
}

bool test_basis_order_twof_1x2x4x1(__m512i* B, __m512i* A24, unsigned int f)
{
    return 
        test_point_order_twof_1x2x4x1(B, A24, f) &
        test_point_order_twof_1x2x4x1(B + NUM_LIMBS, A24, f) &
        test_point_order_twof_1x2x4x1(B + 2*NUM_LIMBS, A24, f);
}

// ==================== 2x2x2 ===================================================

void ec_point_unvec_to_1x2x2x2(__m512i* res, ec_point_t* P)
{
    fp2_t xz[2] = { P->x, P->z };
    fp2_unvectorized_to_2x2x2(res, xz);
}

void ec_point_1x2x2x2_to_unvec(ec_point_t* res, __m512i* P)
{
    fp2_t xz[2];
    fp2_2x2x2_to_unvectorized(xz, P);
    fp2_copy(&res->x, xz);
    fp2_copy(&res->z, xz+1);
}

void ec_set_zero_1x2x2x2(__m512i* P)
{
    fp2_set_binary_2x2x2(P, 0b01);
}

// Does not assume A24 has been normalized.
void ec_double_1x2x2x2(__m512i* res, __m512i* P, __m512i* A24)
{
    __m512i t0[N4x2], t1[N4x2], t2[N4x2], t3[N4x2], idx0, idx1;
    __mmask8 mask;

    // t0 = (X, X), t1 = (Z, Z)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }

    // t0 = (X + Z, X - Z)
    // t1 = t0^2
    mask = 0xf;
    fp2_mixaddsub_2x2x2(t0, t0, t1, mask);
    fp2_sqr_2x2x2(t1, t0);

    // t0 = (t1[0] + t1[1], t1[0] - t1[1])
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(idx0, t1[i]);
        t2[i] = VPERM512(idx1, t1[i]);
    }
    fp2_mixaddsub_2x2x2(t0, t0, t2, mask);

    // t2 = (A24 * t0[1], C24 * t1[1])
    idx0 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < N4x2; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }
    fp2_mul_2x2x2(t2, t2, A24);

    // TODO: Wasting one addition here.
    // t3 = (-, t2[0] + t2[1])
    // Maybe we can implement 1x8x1 Fp2?
    idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
    for (int i = 0; i < N4x2; i++) {
        t3[i] = VPERM512(idx0, t2[i]);
    }
    fp2_add_2x2x2(t3, t3, t2);
    
    // res_X = t1[0] * t2[1], res_Z = t0[1] * t3[1]
    idx0 = VSET(8, 9, 10, 11, 4, 5, 6, 7);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t1[i] = VBLENDPERM(idx1, t2[i], t3[i]);
    }
    fp2_mul_2x2x2(res, t0, t1);
}

void ec_double_iter_1x2x2x2(__m512i* res, __m512i* P, __m512i* A24, const int n)
{
    if (n == 0) {
        fp2_copy_2x2x2(res, P);
        return;
    }
    
    ec_double_1x2x2x2(res, P, A24);
    for (int i = 0; i < n-1; i++) {
        ec_double_1x2x2x2(res, res, A24);
    }   
}

// Assumes n1 >= n2, inputs in 2x2x2
void ec_double_basis_pair_1x2x2x2(
    __m512i* res_B1, 
    __m512i* res_B2, 
    __m512i* B1, 
    __m512i* B2, 
    __m512i* A24_norm1,
    __m512i* A24_norm2,
    const int n1,
    const int n2
)
{
    __m512i x[2*NUM_LIMBS], z[2*NUM_LIMBS], t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], A24x8[2*NUM_LIMBS];
    __m512i idx0, idx1, idx2, idx3;

    assert(n1 >= n2);

    // Preparing input in 8x1x1
    fp2_2x2x2_to_4x2x1(t0, B1, B1 + N4x2);
    fp2_2x2x2_to_4x2x1(t0 + NUM_LIMBS, B1 + 2*N4x2, A24_norm1);
    fp2_2x2x2_to_4x2x1(t1, B2, B2 + N4x2);
    fp2_2x2x2_to_4x2x1(t1 + NUM_LIMBS, B2 + 2*N4x2, A24_norm2);

    idx0 = VSET(0, 1, 4, 5, 8, 9, 0, 0); // (xP, xQ, xPmQ, -)
    idx1 = VSET(2, 3, 6, 7, 10, 11, 0, 0); // (zP, zQ, zPmQ, -)
    idx2 = VSET(4, 5, 4, 5, 4, 5, 4, 5); // (A24, A24, A24, A24)
    for (int i = 0; i < NUM_LIMBS; i++) {
        x[i] = VBLENDPERM(idx0, t0[i], t0[i+NUM_LIMBS]);
        x[i + NUM_LIMBS] = VBLENDPERM(idx0, t1[i], t1[i+NUM_LIMBS]);
        z[i] = VBLENDPERM(idx1, t0[i], t0[i+NUM_LIMBS]);
        z[i + NUM_LIMBS] = VBLENDPERM(idx1, t1[i], t1[i+NUM_LIMBS]);
        A24x8[i] = VPERM512(idx2, t0[i + NUM_LIMBS]);
        A24x8[i + NUM_LIMBS] = VPERM512(idx2, t1[i + NUM_LIMBS]);
    }
    fp2_4x2x1_to_8x1x1(x, x, x + NUM_LIMBS);
    fp2_4x2x1_to_8x1x1(z, z, z + NUM_LIMBS);
    fp2_4x2x1_to_8x1x1(A24x8, A24x8, A24x8 + NUM_LIMBS);
    
    for (int i = 0; i < n2; i++) {
        ec_double_8x1x1x1(x, z, A24x8, x, z);
    }

    fp2_8x1x1_to_4x2x1(x, x + NUM_LIMBS, x);
    fp2_8x1x1_to_4x2x1(z, z + NUM_LIMBS, z);
    fp2_8x1x1_to_first_4x2x1(A24x8, A24x8);
    for (int i = 0; i < n1 - n2; i++) {
        ec_double_4x4x2x1(x, z, A24x8, x, z);
    }

    fp2_4x2x1_to_2x2x2(x, t0, x);
    fp2_4x2x1_to_2x2x2(x + NUM_LIMBS, t0 + NUM_LIMBS, x + NUM_LIMBS);
    fp2_4x2x1_to_2x2x2(z, t1, z);
    fp2_4x2x1_to_2x2x2(z + NUM_LIMBS, t1 + NUM_LIMBS, z + NUM_LIMBS);

    // Coupling first and second components
    idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for(int i = 0; i < N4x2; i++) {
        res_B1[i] = VBLENDPERM(idx0, x[i], z[i]);
        res_B1[i + N4x2] = VBLENDPERM(idx1, x[i], z[i]);
        res_B1[i + 2*N4x2] = VBLENDPERM(idx0, t0[i], t1[i]);
        res_B2[i] = VBLENDPERM(idx0, x[i+NUM_LIMBS], z[i+NUM_LIMBS]);
        res_B2[i + N4x2] = VBLENDPERM(idx1, x[i+NUM_LIMBS], z[i+NUM_LIMBS]);
        res_B2[i + 2*N4x2] = VBLENDPERM(idx0, t0[i+NUM_LIMBS], t1[i+NUM_LIMBS]);
    }
}

void ec_scalar_mul_1x2x2x2(__m512i* res, __m512i* P, __m512i* A24, const digit_t* m, const unsigned int mbits)
{
    __m512i X0_X1[NUM_LIMBS], XD_A24[NUM_LIMBS];
    if (mbits == 0) {
        fp2_copy_2x2x2(res, P);
        return;
    }

    // X0 = P, X1 = 0, XD = P
    ec_set_zero_1x2x2x2(X0_X1);
    fp2_2x2x2_to_4x2x1(X0_X1, X0_X1, P);
    fp2_2x2x2_to_4x2x1(XD_A24, P, A24);

    ec_scalar_mul_inner_1x4x2x1(X0_X1, XD_A24, m, mbits);

    // Ignore X1 and output X0
    fp2_4x2x1_to_2x2x2(res, XD_A24, X0_X1);
}

void ec_scalar_mul_normA_1x2x2x2(__m512i* res, __m512i* P, __m512i* A, const digit_t* m, const unsigned int mbits)
{
    __m512i X0_X1[NUM_LIMBS], XD_A[NUM_LIMBS];
    if (mbits == 0) {
        fp2_copy_2x2x2(res, P);
        return;
    }

    // X0 = 0, X1 = P, XD = P
    ec_set_zero_1x2x2x2(X0_X1);
    fp2_2x2x2_to_4x2x1(X0_X1, X0_X1, P);
    fp2_2x2x2_to_4x2x1(XD_A, P, A);

    ec_scalar_mul_normA_inner_1x4x2x1(X0_X1, XD_A, m, 0, mbits-1);

    // Ignore X1 and output X0
    fp2_4x2x1_to_2x2x2(res, XD_A, X0_X1);
}

// res1 = [m]*P1, res2 = [m]*P2
void ec_scalar_mul_pair_normA_2x2x2(__m512i* res1, __m512i* res2, __m512i* P1, __m512i* P2, __m512i* A1, __m512i* A2, const digit_t* m, const unsigned int mbits)
{
    __m512i X0_X1_XD_A[2*NUM_LIMBS], Y0_Y1_YD_A[2*NUM_LIMBS], zero[N4x2];
    if (mbits == 0) {
        fp2_copy_2x2x2(res1, P1);
        fp2_copy_2x2x2(res2, P2);
        return;
    }

    // X0 = 0, X1 = P, XD = P
    ec_set_zero_1x2x2x2(zero);
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A, zero, P1);
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A + NUM_LIMBS, P1, A1);
    fp2_4x2x1_to_8x1x1(X0_X1_XD_A, X0_X1_XD_A, X0_X1_XD_A + NUM_LIMBS);

    fp2_2x2x2_to_4x2x1(Y0_Y1_YD_A, zero, P2);
    fp2_2x2x2_to_4x2x1(Y0_Y1_YD_A + NUM_LIMBS, P2, A2);
    fp2_4x2x1_to_8x1x1(Y0_Y1_YD_A, Y0_Y1_YD_A, Y0_Y1_YD_A + NUM_LIMBS);

    ec_scalar_mul_pair_normA_inner_8x1x1(X0_X1_XD_A, Y0_Y1_YD_A, m, mbits);

    // Output X0
    fp2_8x1x1_to_4x2x1(X0_X1_XD_A, X0_X1_XD_A + NUM_LIMBS, X0_X1_XD_A);
    fp2_4x2x1_to_2x2x2(res1, X0_X1_XD_A + NUM_LIMBS, X0_X1_XD_A);
    
    fp2_8x1x1_to_4x2x1(Y0_Y1_YD_A, Y0_Y1_YD_A + NUM_LIMBS, Y0_Y1_YD_A);
    fp2_4x2x1_to_2x2x2(res2, Y0_Y1_YD_A + NUM_LIMBS, Y0_Y1_YD_A);
}

bool ec_ladder_3pt_1x2x2x2(__m512i* res, __m512i* A24, __m512i* basis, const digit_t* m)
{
    __m512i X0_X1[NUM_LIMBS], XD_A24[NUM_LIMBS];

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x2x2(basis + 2*N4x2)) {
        return false;
    }

    // X0 = Q, X1 = P, XD = PmQ
    fp2_2x2x2_to_4x2x1(X0_X1, basis + N4x2, basis);
    fp2_2x2x2_to_4x2x1(XD_A24, basis + 2*N4x2, A24);

    ec_ladder_3pt_inner_1x4x2x1(X0_X1, XD_A24, m);

    // We output X1 and ignore X0
    fp2_4x2x1_to_2x2x2(XD_A24, res, X0_X1);

    return true;
}

bool ec_ladder_3pt_normA_1x2x2x2(__m512i* res, __m512i* A, __m512i* basis, const digit_t *m)
{
    __m512i X0_X1[NUM_LIMBS], XD_A[NUM_LIMBS];

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x2x2(basis + 2*N4x2)) {
        return false;
    }

    // X0 = Q, X1 = P, XD = PmQ
    fp2_2x2x2_to_4x2x1(X0_X1, basis + N4x2, basis);
    fp2_2x2x2_to_4x2x1(XD_A, basis + 2*N4x2, A);

    ec_ladder_3pt_normA_inner_1x4x2x1(X0_X1, XD_A, m);

    // We output X1 and ignore X0
    fp2_4x2x1_to_2x2x2(XD_A, res, X0_X1);

    return true;
}

/**
 * Internal helper function for ec_biscalar mul.
 * Given x(P), x(Q) and x(P-Q) it computes x(P+Q). It then sets x(Q), x(P-Q) := x(P+Q), x(Q).
 * A24 is not needed, but is only an artifact of how PmQ is stored in the biscalar ladder.
 * @param P_Q P and Q in 4x2x1
 * @param PmQ_A24 PmQ and A24 in 4x2x1
 */
static void ec_diff_add_1x2x2x2(__m512i* P_Q, __m512i* PmQ_A24)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], r0[N4x2], r1[N4x2];
    __m512i idx0, idx1;
    __mmask8 mask;

    // t0 = (P.x, P.x, Q.x, Q.x),  
    // t1 = (P.z, P.z, Q.z, Q.z)
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, P_Q[i]);
        t1[i] = VPERM512(idx1, P_Q[i]);
    }
    // t0 = (P.x + P.z, P.x - P.z, Q.x - Q.z, Q.x + Q.z)
    mask = 0xc3;
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask);

    // r0 = (P.x + P.z, P.x - P.z)
    // r1 = (Q.x - Q.z, Q.x + Q.z)
    fp2_4x2x1_to_2x2x2(r0, r1, t0);
    fp2_mul_2x2x2(r0, r0, r1);

    // r0 = (r0[0], r0[0]),  t1 = (r0[1], r0[1])
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int j = 0; j < N4x2; j++) {
        r1[j] = VPERM512(idx1, r0[j]);
        r0[j] = VPERM512(idx0, r0[j]);
    }

    fp2_mixaddsub_2x2x2(r0, r0, r1, 0x0f);
    fp2_sqr_2x2x2(r0, r0);

    // Extract (PmQ.z, PmQ.x) from PmQ_A24 into r1
    // Suptoptimal to handle any NUM_LIMBS % 2
    fp2_4x2x1_to_2x2x2(r1, t0, PmQ_A24);
    idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
    for (int i = 0; i < N4x2; i++) {
        r1[i] = VPERM512(idx0, r1[i]);
    }

    fp2_mul_2x2x2(r0, r0, r1); // P + Q

    // Updating input
    // PmQ_A24 = (Q, A24), P_Q = (P, P+Q)
    idx0 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        PmQ_A24[i] = VBLENDPERM(idx0, P_Q[i], PmQ_A24[i]);
    }

    // Supoptimal, to have code be independent of NUM_LIMBS % 2
    fp2_4x2x1_to_2x2x2(r1, t0, P_Q);
    fp2_2x2x2_to_4x2x1(P_Q, r1, r0);
}

// TODO: Can we optimize ec_double_add to modify in place?
// TODO: At the moment I don't take the number of bits as input. 
//       Potentially, the scalars could be much smaller than NWORDS_ORDER.
// TODO: Some swaps are unecessary with a different arrangement of inputs.
bool ec_biscalar_mul_1x2x2x2(
    __m512i* res, 
    const digit_t *scalarP,
    const digit_t *scalarQ,
    __m512i* A24,
    __m512i* basis
)
{
    __m512i X0_X1[NUM_LIMBS], XD_A24[NUM_LIMBS], t0[NUM_LIMBS], t1[NUM_LIMBS];
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3), idx0, idx1;
    digit_t s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER], t_arr[NWORDS_ORDER];
    digit_t* s0 = s0_arr, *s1 = s1_arr, *t = t_arr;

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x2x2(basis + 2*N4x2)) {
        return false;
    }

    fp2_2x2x2_to_4x2x1(X0_X1, basis, basis + N4x2);
    fp2_2x2x2_to_4x2x1(XD_A24, basis + 2*N4x2, A24);
    mp_copy(s0, scalarP, NWORDS_ORDER);
    mp_copy(s1, scalarQ, NWORDS_ORDER);

    while (!mp_is_zero(s0, NWORDS_ORDER)) {
        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t* tmp = s0;
            s0 = s1;
            s1 = tmp;
            
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }
        // Now s0 <= s1

        // t = 4 * s0
        mp_copy(t, s0, NWORDS_ORDER);
        digit_t overflow_bits = t[NWORDS_ORDER - 1] >> (RADIX-2);
        mp_shiftl(t, 2, NWORDS_ORDER); 
        bool s0_even = mp_is_even(s0, NWORDS_ORDER);
        bool s1_even = mp_is_even(s1, NWORDS_ORDER);

        // if the shift overflows, 4*s0 is definitely larger than s1.
        if (overflow_bits || mp_compare(s1, t, NWORDS_ORDER) <= 0) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_diff_add_1x2x2x2(X0_X1, XD_A24);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s0_even == s1_even) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);
            mp_shiftr(s1, 1, NWORDS_ORDER);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_double_add_1x4x2x1(X0_X1, X0_X1, XD_A24);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s1_even) {
            mp_shiftr(s1, 1, NWORDS_ORDER);

            idx0 = VSET(4, 5, 6, 7, 8, 9, 10, 11); // second first
            idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            for (int k = 0; k < NUM_LIMBS; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A24[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A24[k]);
            }
            ec_double_add_1x4x2x1(t0, t0, t1);

            // Update X1 and XD
            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VBLENDPERM(idx0, X0_X1[k], t0[k]);
                XD_A24[k] = VBLENDPERM(idx1, t0[k], XD_A24[k]);
            }
        } else {
            mp_shiftr(s0, 1, NWORDS_ORDER);

            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A24[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A24[k]);
            }
            ec_double_add_1x4x2x1(t0, t0, t1);

            // Update X0 and XD
            idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VBLENDPERM(idx0, t0[k], X0_X1[k]);
                XD_A24[k] = VBLENDPERM(idx1, t0[k], XD_A24[k]);
            }
        }
    }

    // We ignore X0, and continue with X1 in 2x2x2
    fp2_4x2x1_to_2x2x2(XD_A24, X0_X1, X0_X1);

    while (mp_is_even(s1, NWORDS_ORDER)) {
        mp_shiftr(s1, 1, NWORDS_ORDER);
        ec_double_1x2x2x2(X0_X1, X0_X1, A24);
    }
    unsigned int s1_nbits = mp_bitlength(s1, NWORDS_ORDER);

    // res = [s_1]*X1
    ec_scalar_mul_1x2x2x2(res, X0_X1, A24, s1, s1_nbits);

    return true;
}

bool ec_biscalar_mul_normA_1x2x2x2(
    __m512i* res, 
    const digit_t *scalarP,
    const digit_t *scalarQ,
    __m512i* A,
    __m512i* A24,
    __m512i* basis
)
{
    __m512i X0_X1[NUM_LIMBS], XD_A[NUM_LIMBS], t0[NUM_LIMBS], t1[NUM_LIMBS];
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3), idx0, idx1;
    digit_t s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER], t_arr[NWORDS_ORDER];
    digit_t* s0 = s0_arr, *s1 = s1_arr, *t = t_arr;

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x2x2(basis + 2*N4x2)) {
        return false;
    }

    fp2_2x2x2_to_4x2x1(X0_X1, basis, basis + N4x2);
    fp2_2x2x2_to_4x2x1(XD_A, basis + 2*N4x2, A);
    mp_copy(s0, scalarP, NWORDS_ORDER);
    mp_copy(s1, scalarQ, NWORDS_ORDER);

    while (!mp_is_zero(s0, NWORDS_ORDER)) {
        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t* tmp = s0;
            s0 = s1;
            s1 = tmp;
            
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }
        // Now s0 <= s1

        // t = 4 * s0
        mp_copy(t, s0, NWORDS_ORDER);
        digit_t overflow_bits = t[NWORDS_ORDER - 1] >> (RADIX-2);
        mp_shiftl(t, 2, NWORDS_ORDER); 
        bool s0_even = mp_is_even(s0, NWORDS_ORDER);
        bool s1_even = mp_is_even(s1, NWORDS_ORDER);

        // if the shift overflows, 4*s0 is definitely larger than s1.
        if (overflow_bits || mp_compare(s1, t, NWORDS_ORDER) <= 0) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_diff_add_1x2x2x2(X0_X1, XD_A);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s0_even == s1_even) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);
            mp_shiftr(s1, 1, NWORDS_ORDER);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_double_add_normA_1x4x2x1(X0_X1, X0_X1, XD_A);

            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s1_even) {
            mp_shiftr(s1, 1, NWORDS_ORDER);

            idx0 = VSET(4, 5, 6, 7, 8, 9, 10, 11); // second first
            idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            for (int k = 0; k < NUM_LIMBS; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A[k]);
            }
            ec_double_add_normA_1x4x2x1(t0, t0, t1);

            // Update X1 and XD
            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VBLENDPERM(idx0, X0_X1[k], t0[k]);
                XD_A[k] = VBLENDPERM(idx1, t0[k], XD_A[k]);
            }
        } else {
            mp_shiftr(s0, 1, NWORDS_ORDER);

            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A[k]);
            }
            ec_double_add_normA_1x4x2x1(t0, t0, t1);

            // Update X0 and XD
            idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < NUM_LIMBS; k++) {
                X0_X1[k] = VBLENDPERM(idx0, t0[k], X0_X1[k]);
                XD_A[k] = VBLENDPERM(idx1, t0[k], XD_A[k]);
            }
        }
    }

    // We ignore X0, and continue with X1 in 2x4x1
    fp2_4x2x1_to_2x2x2(XD_A, X0_X1, X0_X1);

    while (mp_is_even(s1, NWORDS_ORDER)) {
        mp_shiftr(s1, 1, NWORDS_ORDER);
        ec_double_1x2x2x2(X0_X1, X0_X1, A24);
    }
    unsigned int s1_nbits = mp_bitlength(s1, NWORDS_ORDER);

    // res = [s_1]*X1
    ec_scalar_mul_normA_1x2x2x2(res, X0_X1, A, s1, s1_nbits);

    return true;
}

bool ec_biscalar_mul_pair_1x2x2x2(
    __m512i* res1,
    __m512i* res2,
    scalar_mtx_2x2_t *mat,
    __m512i* A24,
    __m512i* basis
)
{
    digit_t r0_arr[NWORDS_ORDER], r1_arr[NWORDS_ORDER], s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER];
    digit_t *r0 = r0_arr, *r1 = r1_arr, *s0 = s0_arr, *s1 = s1_arr; 
    __m512i X0_X1_XD_A24[4*NUM_LIMBS], Y0_Y1_YD_A24[4*NUM_LIMBS], t[2*NUM_LIMBS];
    // TODO: Do we still need 4*NUM_LIMBS?

    // 8x1x1 layout = (X0.x, XD.x, X0.z, XD.z, X1.x, A24.x, X1.z, A24.z)
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A24, basis, basis + N4x2);
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A24 + NUM_LIMBS, basis + 2*N4x2, A24);
    fp2_4x2x1_to_8x1x1(X0_X1_XD_A24, X0_X1_XD_A24, X0_X1_XD_A24 + NUM_LIMBS);
    fp2_copy_8x1x1(Y0_Y1_YD_A24, X0_X1_XD_A24);

    mp_copy(r0, (*mat)[0][0], NWORDS_ORDER);
    mp_copy(r1, (*mat)[1][0], NWORDS_ORDER);
    mp_copy(s0, (*mat)[0][1], NWORDS_ORDER);
    mp_copy(s1, (*mat)[1][1], NWORDS_ORDER);

    ec_biscalar_mul_pair_inner_8x1x1(X0_X1_XD_A24, Y0_Y1_YD_A24, &r0, &r1, &s0, &s1);

    fp2_8x1x1_to_4x2x1(t, t + NUM_LIMBS, X0_X1_XD_A24);
    fp2_4x2x1_to_2x2x2(X0_X1_XD_A24, X0_X1_XD_A24 + N4x2, t);
    fp2_4x2x1_to_2x2x2(X0_X1_XD_A24 + 2*N4x2, X0_X1_XD_A24 + 3*N4x2, t + NUM_LIMBS);
    fp2_8x1x1_to_4x2x1(t, t + NUM_LIMBS, Y0_Y1_YD_A24);
    fp2_4x2x1_to_2x2x2(Y0_Y1_YD_A24, Y0_Y1_YD_A24 + N4x2, t);
    fp2_4x2x1_to_2x2x2(Y0_Y1_YD_A24 + 2*N4x2, Y0_Y1_YD_A24 + 3*N4x2, t + NUM_LIMBS);

    bool succ1 = ec_biscalar_mul_1x2x2x2(res1, r0, r1, A24, X0_X1_XD_A24);
    bool succ2 = ec_biscalar_mul_1x2x2x2(res2, s0, s1, A24, Y0_Y1_YD_A24);

    return succ1 && succ2;
}

bool ec_biscalar_mul_pair_normA_1x2x2x2(
    __m512i* res1,
    __m512i* res2,
    scalar_mtx_2x2_t *mat,
    __m512i* A,
    __m512i* A24,
    __m512i* basis
)
{
    digit_t r0_arr[NWORDS_ORDER], r1_arr[NWORDS_ORDER], s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER];
    digit_t *r0 = r0_arr, *r1 = r1_arr, *s0 = s0_arr, *s1 = s1_arr; 
    __m512i X0_X1_XD_A[4*NUM_LIMBS], Y0_Y1_YD_A[4*NUM_LIMBS], t[2*NUM_LIMBS];
    // TODO: Do we still need 4*NUM_LIMBS?

    // 8x1x1 layout = (X0.x, XD.x, X0.z, XD.z, X1.x, A.x, X1.z, -)
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A, basis, basis + N4x2);
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A + NUM_LIMBS, basis + 2*N4x2, A);
    fp2_4x2x1_to_8x1x1(X0_X1_XD_A, X0_X1_XD_A, X0_X1_XD_A + NUM_LIMBS);
    fp2_copy_8x1x1(Y0_Y1_YD_A, X0_X1_XD_A);

    mp_copy(r0, (*mat)[0][0], NWORDS_ORDER);
    mp_copy(r1, (*mat)[1][0], NWORDS_ORDER);
    mp_copy(s0, (*mat)[0][1], NWORDS_ORDER);
    mp_copy(s1, (*mat)[1][1], NWORDS_ORDER);

    ec_biscalar_mul_pair_normA_inner_8x1x1(X0_X1_XD_A, Y0_Y1_YD_A, &r0, &r1, &s0, &s1);

    fp2_8x1x1_to_4x2x1(t, t + NUM_LIMBS, X0_X1_XD_A);
    fp2_4x2x1_to_2x2x2(X0_X1_XD_A, X0_X1_XD_A + N4x2, t);
    fp2_4x2x1_to_2x2x2(X0_X1_XD_A + 2*N4x2, X0_X1_XD_A + 3*N4x2, t+NUM_LIMBS);
    fp2_8x1x1_to_4x2x1(t, t + NUM_LIMBS, Y0_Y1_YD_A);
    fp2_4x2x1_to_2x2x2(Y0_Y1_YD_A, Y0_Y1_YD_A + N4x2, t);
    fp2_4x2x1_to_2x2x2(Y0_Y1_YD_A + 2*N4x2, Y0_Y1_YD_A + 3*N4x2, t + NUM_LIMBS);

    bool succ1 = ec_biscalar_mul_normA_1x2x2x2(res1, r0, r1, A, A24, X0_X1_XD_A);
    bool succ2 = ec_biscalar_mul_normA_1x2x2x2(res2, s0, s1, A, A24, Y0_Y1_YD_A);

    return succ1 && succ2;
}

// Faster than SQI variant
void ec_normalize_curve_2x2x2(__m512i* res_A24, __m512i* res_A, __m512i* A24)
{
    __m512i t0[N4x2], t1[N4x2], t2[N4x2];
    const __m512i idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    const __m512i idx1 = VSET(0, 1, 2, 3, 8, 9, 10, 11);

    // t0 = (A24^-1, C24^-1), wasted computation of A24^-1
    fp2_inv_2x2x2(t0, A24);

    // res_A24 = (A24 / C24, 1)
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mul_2x2x2(t2, t0, A24);

    // A = 4*A24 - 2 = 2*(2*A24 - 1)
    fp2_add_2x2x2(t1, t2, t2);
    fp2_set_binary_2x2x2(t0, 0b11);
    fp2_sub_2x2x2(t1, t1, t0);
    fp2_add_2x2x2(t1, t1, t1);

    for (int i = 0; i < N4x2; i++) {
        res_A[i] = VBLENDPERM(idx1, t1[i], t0[i]);
    }
    fp2_copy_2x2x2(res_A24, t2);
}

// a > b for canonical Fp2 elements stored as (re, im), ordered by im then re.
static bool fp2_64_is_greater(const uint64_t *a, const uint64_t *b)
{
    for (int i = 2*NWORDS64 - 1; i >= 0; i--) {
        if (a[i] != b[i]) {
            return a[i] > b[i];
        }
    }
    return false;
}

// Largest of the six isomorphic A = -2 (a^4 + b^4) / (a^4 - b^4), taken over the
// pairs (a, b), (a + b, a - b), (i*a + b, a + i*b) and their negations, ordered
// by (im, re). ab is (a, b) in 2x2x2, res_normA is (A, 1). False if a^4 == b^4.
bool ec_normalized_A_from_pair_2x2x2(__m512i* res_normA, __m512i* ab)
{
    __m512i t0[N4x2], t1[N4x2], a01[N4x2], b01[N4x2], a22[N4x2], b22[N4x2];
    __m512i rowA[NUM_LIMBS], rowB[NUM_LIMBS], num[NUM_LIMBS], den[NUM_LIMBS];
    uint64_t pos[6*NWORDS64], neg[6*NWORDS64], enc[4*NWORDS64];

    const __m512i dup_lo = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    const __m512i dup_hi = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    const __m512i swap   = VSET(4, 5, 6, 7, 0, 1, 2, 3);
    const __m512i idx_a  = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    const __m512i idx_b  = VSET(4, 5, 6, 7, 12, 13, 14, 15);

    // t0 = (a, a), t1 = (b, b), then t0 = (a + b, a - b) = (a1, b1)
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(dup_lo, ab[i]);
        t1[i] = VPERM512(dup_hi, ab[i]);
    }
    fp2_mixaddsub_2x2x2(t0, t0, t1, 0x0f);

    // t1 = i*(a, b) + (b, a) = (i*a + b, a + i*b) = (a2, b2)
    fp2_muli_2x2x2(t1, ab);
    for (int i = 0; i < N4x2; i++) {
        a22[i] = VPERM512(swap, ab[i]);
    }
    fp2_add_2x2x2(t1, t1, a22);

    // The fourth slot repeats the third, to fill the vector and keep it invertible.
    for (int i = 0; i < N4x2; i++) {
        a01[i] = VBLENDPERM(idx_a, ab[i], t0[i]); // (a0, a1)
        b01[i] = VBLENDPERM(idx_b, ab[i], t0[i]); // (b0, b1)
        a22[i] = VPERM512(dup_lo, t1[i]);         // (a2, a2)
        b22[i] = VPERM512(dup_hi, t1[i]);         // (b2, b2)
    }
    fp2_2x2x2_to_4x2x1(rowA, a01, a22);
    fp2_2x2x2_to_4x2x1(rowB, b01, b22);

    // Fourth powers of both rows
    fp2_sqrx2_4x2x1(rowA, rowB, rowA, rowB);
    fp2_sqrx2_4x2x1(rowA, rowB, rowA, rowB);

    // A_k = -2 (a_k^4 + b_k^4) / (a_k^4 - b_k^4)
    fp2_add_4x2x1(num, rowA, rowB);
    fp2_add_4x2x1(num, num, num);
    fp2_neg_4x2x1(num, num);
    fp2_sub_4x2x1(den, rowA, rowB);
    if (fp2_is_zero_4x2x1(den) & 0x7) {
        return false;
    }
    fp2_inv_4x2x1(den, den);
    fp2_mul_4x2x1(num, num, den);

    // Decode the three coefficients and their negations to canonical integers
    fp2_4x2x1_to_2x2x2(t0, t1, num);
    fp2_frommont_2x2x2(t0, t0);
    fp2_frommont_2x2x2(t1, t1);
    fp_limbs_to_64_4x2(pos, t0);
    fp_limbs_to_64_4x2(enc, t1);
    memcpy(pos + 4*NWORDS64, enc, 2*NWORDS64*sizeof(uint64_t));

    fp2_neg_4x2x1(num, num);
    fp2_4x2x1_to_2x2x2(t0, t1, num);
    fp2_frommont_2x2x2(t0, t0);
    fp2_frommont_2x2x2(t1, t1);
    fp_limbs_to_64_4x2(neg, t0);
    fp_limbs_to_64_4x2(enc, t1);
    memcpy(neg + 4*NWORDS64, enc, 2*NWORDS64*sizeof(uint64_t));

    const uint64_t *best = pos;
    for (int i = 1; i < 3; i++) {
        if (fp2_64_is_greater(pos + 2*i*NWORDS64, best)) {
            best = pos + 2*i*NWORDS64;
        }
    }
    for (int i = 0; i < 3; i++) {
        if (fp2_64_is_greater(neg + 2*i*NWORDS64, best)) {
            best = neg + 2*i*NWORDS64;
        }
    }

    // res_normA = (A_max, 1)
    memcpy(enc, best, 2*NWORDS64*sizeof(uint64_t));
    memset(enc + 2*NWORDS64, 0, 2*NWORDS64*sizeof(uint64_t));
    enc[2*NWORDS64] = 1;
    fp_64_to_limbs_4x2(res_normA, enc);
    fp_tomont_4x2(res_normA, res_normA);

    return true;
}

// Slower than previous
void ec_normalize_curve_SQI_2x2x2(__m512i* res_A24, __m512i* res_A, __m512i* A24)
{
    fp2_t unvec[2];
    __m512i one[N4x2];
    const __m512i idx = VSET(0, 1, 2, 3, 8, 9, 10, 11);

    fp2_2x2x2_to_unvectorized(unvec, A24);
    fp2_inv(unvec+1);
    fp2_mul(unvec, unvec, unvec+1);
    fp2_set_one(unvec+1);
    fp2_unvectorized_to_2x2x2(res_A24, unvec);

    // A = 4*A24 - 2 = 2*(2*A24 - 1)
    // faster to avoid another unvec -> 2x2x2
    fp2_set_binary_2x2x2(one, 0b01);
    fp2_add_2x2x2(res_A, res_A24, res_A24);
    fp2_sub_2x2x2(res_A, res_A, one);
    fp2_add_2x2x2(res_A, res_A, res_A);
    for (int i = 0; i < N4x2; i++) {
        res_A[i] = VBLENDPERM(idx, res_A[i], one[i]);
    }
}

void ec_normalize_curve_pair_2x2x2(__m512i* res_A24_1, __m512i* res_A24_2, __m512i* res_A_1, __m512i* res_A_2, __m512i* A24_1, __m512i* A24_2)
{
    __m512i t0[N4x2], t1[N4x2];
    const __m512i idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    const __m512i idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);

    // t0 = (A24_1, A24_2), t1 = (C24_1, C24_2)
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx0, A24_1[i], A24_2[i]);
        t1[i] = VBLENDPERM(idx1, A24_1[i], A24_2[i]);
    }

    // t0 = (A24_1 / C24_1, A24_2 / C24_2)
    fp2_inv_2x2x2(t1, t1);
    fp2_mul_2x2x2(t0, t0, t1);

    // Output normalized A24
    fp2_set_binary_2x2x2(t1, 0b11);
    for (int i = 0; i < N4x2; i++) {
        res_A24_1[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        res_A24_2[i] = VBLENDPERM(idx1, t0[i], t1[i]);
    }

    // A = 4*A24 - 2 = 2*(2*A24 - 1)
    fp2_add_2x2x2(t0, t0, t0);
    fp2_sub_2x2x2(t0, t0, t1);
    fp2_add_2x2x2(t0, t0, t0);

    // Output normalized A
    for (int i = 0; i < N4x2; i++) {
        res_A_1[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        res_A_2[i] = VBLENDPERM(idx1, t0[i], t1[i]);
    }
}

void ec_normalize_A24_pair_2x2x2(__m512i* res1, __m512i* res2, __m512i* A24_1, __m512i* A24_2)
{
    __m512i A24[N4x2], C24[N4x2];
    const __m512i idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    const __m512i idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);

    // A24 = (A24_1, A24_2), C24 = (C24_1, C24_2)
    for (int i = 0; i < N4x2; i++) {
        A24[i] = VBLENDPERM(idx0, A24_1[i], A24_2[i]);
        C24[i] = VBLENDPERM(idx1, A24_1[i], A24_2[i]);
    }

    fp2_inv_2x2x2(C24, C24);
    fp2_mul_2x2x2(A24, A24, C24);

    // res1 = (A24_1 / C24_1, 1), res2 = (A24_2 / C24_2, 1)
    fp2_set_binary_2x2x2(C24, 0b11);
    for (int i = 0; i < N4x2; i++) {
        res1[i] = VBLENDPERM(idx0, A24[i], C24[i]);
        res2[i] = VBLENDPERM(idx1, A24[i], C24[i]);
    }
}

// A = 4*A24 - 2
void ec_normA24_to_normA_1x2x2x2(__m512i* res, __m512i* normA24)
{
    __m512i two[N4x2], normA[N4x2];
    const __m512i idx = VSET(0, 1, 2, 3, 12, 13, 14, 15);

    fp2_set_binary_2x2x2(two, 0b01);
    fp2_add_2x2x2(two, two, two);

    fp2_add_2x2x2(normA, normA24, normA24);
    fp2_add_2x2x2(normA, normA, normA);
    fp2_sub_2x2x2(normA, normA, two);

    // Assuming normA24 is normalized, we exploit that it's second component is 1.
    for (int i = 0; i < N4x2; i++) {
        res[i] = VBLENDPERM(idx, normA[i], normA24[i]);
    }
}

// A24 = (A + 2)/4
void ec_normA_to_normA24_1x2x2x2(__m512i* res, __m512i* normA)
{
    __m512i two[N4x2], normA24[N4x2];
    const __m512i idx = VSET(0, 1, 2, 3, 12, 13, 14, 15);

    fp2_set_binary_2x2x2(two, 0b01);
    fp2_add_2x2x2(two, two, two);

    fp2_add_2x2x2(normA24, normA, two);
    fp2_div2_2x2x2(normA24, normA24);
    fp2_div2_2x2x2(normA24, normA24);

    // Assuming normA is normalized, we exploit that it's second component is 1.
    for (int i = 0; i < N4x2; i++) {
        res[i] = VBLENDPERM(idx, normA24[i], normA[i]);
    }
}

// Verifies the normalized Montgomery A-coefficient is valid for each curve (A^2-4 != 0, i.e. A != +- 2)
bool ec_curve_verify_normA_pair_2x2x2(__m512i* normA1, __m512i* normA2)
{
    __m512i A1_A2[N4x2], two[N4x2];
    const __m512i idx = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    for (int i = 0; i < N4x2; i++) {
        A1_A2[i] = VBLENDPERM(idx, normA1[i], normA2[i]);
    }
    fp2_set_binary_2x2x2(two, 0b11);
    fp2_add_2x2x2(two, two, two);
    if (fp2_is_equal_2x2x2(A1_A2, two) != 0) {
        return false;
    }
    fp2_neg_2x2x2(two, two);
    return fp2_is_equal_2x2x2(A1_A2, two) == 0;
}

// j = (256 * (A^2 - 3C^2)^3) / (C^4 * (A^2 - 4C^2))
void ec_j_inv_2x2x2(__m512i* res, __m512i* AC)
{
    __m512i t0[N4x2], t1[N4x2], t2[N4x2], t3[N4x2];
    __m512i idx0, idx1;

    fp2_sqr_2x2x2(t0, AC); // A^2, C^2

    // t1 = (C^2, 1);
    fp2_set_binary_2x2x2(t1, 0b01);
    idx0 = VSET(4, 5, 6, 7, 8, 9, 10, 11);
    for (int i = 0; i < N4x2; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }

    // t2 = (2C^2, 2)
    fp2_add_2x2x2(t2, t1, t1);

    // t0 = (A^2, 2)
    // t2 = (A^2 - 2C^2, 4)
    idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
    for(int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], t2[i]);
    }
    fp2_mixaddsub_2x2x2(t2, t0, t2, 0xf0);

    // t0 = (C^2, 4)
    // t2 = (A^2 - 3C^2, 8)
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx0, t1[i], t2[i]);
    }
    fp2_mixaddsub_2x2x2(t2, t2, t0, 0xf0);

    // t0 = (C^2, 8)
    // t1 = (A^2 - 4C^2, 16)
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], t2[i]);
    }
    fp2_mixaddsub_2x2x2(t1, t2, t0, 0xf0);

    // t0 = (C^2, (A^2 - 3C^2))^2
    idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], t2[i]);
    }
    fp2_sqr_2x2x2(t0, t0);

    // t2 = (16, A^2 - 3C^2)
    // t3 = (16, (A^2 - 3C^2)^2)
    idx0 = VSET(4, 5, 6, 7, 8, 9, 10, 11);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < N4x2; i++) {
        t2[i] = VBLENDPERM(idx0, t1[i], t2[i]);
        t3[i] = VBLENDPERM(idx1, t1[i], t0[i]);
    }
    // t2 = (256, (A^2 - 3C^2)^3)
    fp2_mul_2x2x2(t2, t2, t3);

    // t0 = (256, C^4)
    // t1 = ((A^2 - 3C^2)^3, A^2 - 4C^2)
    idx0 = VSET(8, 9, 10, 11, 0, 1, 2, 3);
    idx1 = VSET(12, 13, 14, 15, 0, 1, 2, 3);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], t2[i]);
        t1[i] = VBLENDPERM(idx1, t1[i], t2[i]);
    }
    // t0 = (256 * (A^2 - 3C^2)^3, C^4 * (A^2 - 4C^2)) = (nominator, denominator)
    fp2_mul_2x2x2(t0, t0, t1);
    
    // t1 = (denominator^-1, denominator^-1)
    // res = (j_inv, 1)
    fp2_inv_2x2x2(t1, t0);
    idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < N4x2; i++) {
        t1[i] = VPERM512(idx0, t1[i]);
    }
    fp2_mul_2x2x2(res, t0, t1);
}

// Deterministic x(P - Q) for a pair of curves. P and Q are
// (P1.x, P1.z, P2.x, P2.z) in 4x2x1 and normA is (A1, 1, A2, 1).
void ec_proj_diff_pair_4x2x1(__m512i* res_PmQ, __m512i* P, __m512i* Q, __m512i* normA)
{
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    __m512i b_xz[NUM_LIMBS], b_xxzz[NUM_LIMBS], lam[NUM_LIMBS];
    __m512i u[NUM_LIMBS], v[NUM_LIMBS], delta[N4x2], tmp[N4x2];
    __m512i idx0, idx1;

    // In 8x1x1 we have t0 = (P1.x, P1.x, P1.z, P1.z, P2.x, P2.x, P2.z, P2.z)
    // and t1 = (Q1.x, Q1.z, Q1.z, Q1.x, Q2.x, Q2.z, Q2.z, Q2.x), so that the
    // product holds (t0, t4, t1, t5) for each of the two curves.
    fp2_4x2x1_to_8x1x1(t0, P, P);
    idx0 = VSET(0, 2, 2, 0, 4, 6, 6, 4);
    idx1 = VSET(1, 3, 3, 1, 5, 7, 7, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, Q[i]);
        t1[i+NUM_LIMBS] = VPERM512(idx1, Q[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    // Hadamard, giving (t0 + t1, t4 + t5, t0 - t1, t4 - t5) per curve
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VPERM512(idx1, t0[i]);
    }
    fp2_mixaddsub_8x1x1(t1, t1, t2, 0x33, 0x33);

    // ((t0 + t1)(t4 + t5), (t0 - t1)^2, t4 t5, (t4 - t5)^2) per curve
    idx0 = VSET(9, 10, 3, 11, 13, 14, 7, 15);
    idx1 = VSET(8, 10, 1, 11, 12, 14, 5, 15);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t1[i] = VBLENDPERM(idx1, t0[i], t1[i]);
    }
    fp2_mul_8x1x1(t1, t1, t2);

    // t1 = ((t0+t1)(t4+t5), t4 t5) per curve, t1 + NUM_LIMBS = (b_xx, b_zz)
    fp2_8x1x1_to_4x2x1(t1, t1 + NUM_LIMBS, t1);

    // b_xz = (t0 + t1)(t4 + t5) + 2 A t4 t5, held in slots 0, 1 and 2, 3
    idx0 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    idx1 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        u[i] = VPERM512(idx0, t1[i]);     // (t4 t5, t4 t5) per curve
        v[i] = VPERM512(idx1, normA[i]);  // (A, A) per curve
        b_xz[i] = VPERM512(idx1, t1[i]);  // ((t0+t1)(t4+t5), same) per curve
    }
    fp2_mul_4x2x1(u, u, v);
    fp2_add_4x2x1(b_xz, b_xz, u);
    fp2_add_4x2x1(b_xz, b_xz, u);

    // lam = conj(b_zz), broadcast over both slots of each curve
    fp2_conjugate_4x2x1(u, t1 + NUM_LIMBS);
    for (int i = 0; i < NUM_LIMBS; i++) {
        lam[i] = VPERM512(idx0, u[i]);
    }
    fp2_mulx2_4x2x1(b_xz, b_xxzz, b_xz, lam, t1 + NUM_LIMBS, lam);

    // delta^2 = b_xz^2 - b_xx b_zz, gathered into the first two slots
    idx0 = VSET(0, 1, 8, 9, 4, 5, 12, 13);
    idx1 = VSET(0, 1, 10, 11, 4, 5, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        u[i] = VBLENDPERM(idx0, b_xz[i], b_xxzz[i]);
        v[i] = VBLENDPERM(idx1, b_xz[i], b_xxzz[i]);
    }
    fp2_mul_4x2x1(u, u, v);
    idx0 = VSET(0, 1, 4, 5, 0, 1, 4, 5);
    idx1 = VSET(2, 3, 6, 7, 2, 3, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        v[i] = VPERM512(idx1, u[i]);
        u[i] = VPERM512(idx0, u[i]);
    }
    fp2_sub_4x2x1(u, u, v);

    // Both square roots fit in one 2x2x2 call
    fp2_4x2x1_to_2x2x2(delta, tmp, u);
    fp2_sqrt_2x2x2(delta, delta);
    fp2_2x2x2_to_4x2x1(v, delta, delta);

    // res_PmQ = (b_xz - delta, b_zz) per curve
    idx0 = VSET(0, 1, 4, 5, 0, 1, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        u[i] = VPERM512(idx0, b_xz[i]);
    }
    fp2_sub_4x2x1(u, u, v);
    idx0 = VSET(0, 1, 10, 11, 2, 3, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res_PmQ[i] = VBLENDPERM(idx0, u[i], b_xxzz[i]);
    }
}

// Single curve version, computed with the pair routine on a duplicated input.
void ec_proj_diff_1x2x2x2(__m512i* res_PmQ, __m512i* P, __m512i* Q, __m512i* normA)
{
    __m512i Pv[NUM_LIMBS], Qv[NUM_LIMBS], Av[NUM_LIMBS], res[NUM_LIMBS], tmp[N4x2];

    fp2_2x2x2_to_4x2x1(Pv, P, P);
    fp2_2x2x2_to_4x2x1(Qv, Q, Q);
    fp2_2x2x2_to_4x2x1(Av, normA, normA);
    ec_proj_diff_pair_4x2x1(res, Pv, Qv, Av);
    fp2_4x2x1_to_2x2x2(res_PmQ, tmp, res);
}

void ec_proj_diff_SQI_1x2x2x2(__m512i* res_PmQ, __m512i* P, __m512i* Q, __m512i* normA)
{
    ec_proj_diff_1x2x2x2(res_PmQ, P, Q, normA);
}

// Assums hint_P != 0
// Outputs P and Q (the cofactor has not been cleared)
void ec_curve_to_basis_from_hint_1x2x2x2_inner_set_P_Q(__m512i* res_P, __m512i* res_Q, __m512i* normA, const uint8_t hint)
{
    __m512i t0[N4x2], t1[N4x2], t2[N4x2], idx0, idx1;

    uint8_t hint_A = hint & 0x1;
    uint8_t hint_P = hint / 2;

    if (!hint_P) { // TODO, call unvectorized code
        printf("Not implemented!\n");
        return;
    }

    if (hint_A) {
        // xP = -A, zP = 1 + hint_P*i
        // xQ = A - A*zP = A*(1-zP) = (-A)*(zP-1) = (-A)*(hintP*i), zQ = zP

        // t0 = (xP, .), t1 = (zP-1, zP)
        fp2_neg_2x2x2(t0, normA);
        fp2_setzero_2x2x2(t1);
        t1[0] = VSET(0, 0, hint_P, 0, 1, 0, hint_P, 0);
        fp2_tomont_2x2x2(t1, t1);

        // t2 = (xQ, .)
        fp2_mul_2x2x2(t2, t0, t1);

        idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
        for (int i = 0; i < N4x2; i++) {
            res_P[i] = VBLENDPERM(idx0, t0[i], t1[i]);
            res_Q[i] = VBLENDPERM(idx0, t2[i], t1[i]);
        }
    } else {
        // xP = hint_P * A, zP = 1
        // xQ = -(xP + A) = (hint_P+1)*(-A), zQ = 1

        // t0 = (A, -A)
        fp2_neg_2x2x2(t0, normA);
        idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
        for (int i = 0; i < N4x2; i++) {
            t0[i] = VBLENDPERM(idx0, normA[i], t0[i]);
        }
        // t1 = (hint_P, hint_P + 1)
        fp2_setzero_2x2x2(t1);
        t1[0] = VSET(hint_P, 0, 0, 0, hint_P+1, 0, 0, 0);
        fp2_tomont_2x2x2(t1, t1);

        // t0 := t0 * t1 = (xP, xQ)
        fp2_mul_2x2x2(t0, t0, t1);

        // P = (t0[0], normA[1])
        // Q = (t0[1], normA[1])
        idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
        idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
        for (int i = 0; i < N4x2; i++) {
            res_P[i] = VBLENDPERM(idx0, t0[i], normA[i]);
            res_Q[i] = VBLENDPERM(idx1, t0[i], normA[i]);
        }
    }
}

void ec_curve_to_basis_from_hint_1x2x2x2(__m512i* res_basis, __m512i* normA, const uint8_t hint)
{
    ec_curve_to_basis_from_hint_1x2x2x2_inner_set_P_Q(res_basis, res_basis+2*N4x2, normA, hint);

    ec_scalar_mul_pair_normA_2x2x2(res_basis, res_basis + 2*N4x2, res_basis, res_basis + 2*N4x2, normA, normA, COFACTOR, COFACTOR_NBITS);

    // compute PmQ, set PmQ to Q to ensure Q above (0,0)
    ec_proj_diff_1x2x2x2(res_basis+N4x2, res_basis, res_basis+2*N4x2, normA);
}

// n doublings of (P1, Q1) on E1 and (P2, Q2) on E2, points in 2x2x2 and curves as (A, 1).
static void ec_double_iter_PQ_pair_2x2x2(__m512i* P1, __m512i* Q1, __m512i* P2, __m512i* Q2,
                                         __m512i* normA1, __m512i* normA2, const int n)
{
    __m512i x[NUM_LIMBS], z[NUM_LIMBS], A24[NUM_LIMBS], t0[NUM_LIMBS], t1[NUM_LIMBS];
    __m512i A24_1[N4x2], A24_2[N4x2];

    const __m512i idx_x = VSET(0, 1, 4, 5, 8, 9, 12, 13);
    const __m512i idx_z = VSET(2, 3, 6, 7, 10, 11, 14, 15);
    const __m512i idx_A24 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    const __m512i idx_B1 = VSET(0, 1, 8, 9, 2, 3, 10, 11);
    const __m512i idx_B2 = VSET(4, 5, 12, 13, 6, 7, 14, 15);

    ec_normA_to_normA24_1x2x2x2(A24_1, normA1);
    ec_normA_to_normA24_1x2x2x2(A24_2, normA2);

    // x = (P1.x, Q1.x, P2.x, Q2.x), z likewise, A24 = (A24_1, A24_1, A24_2, A24_2)
    fp2_2x2x2_to_4x2x1(t0, P1, Q1);
    fp2_2x2x2_to_4x2x1(t1, P2, Q2);
    fp2_2x2x2_to_4x2x1(A24, A24_1, A24_2);
    for (int i = 0; i < NUM_LIMBS; i++) {
        x[i] = VBLENDPERM(idx_x, t0[i], t1[i]);
        z[i] = VBLENDPERM(idx_z, t0[i], t1[i]);
        A24[i] = VPERM512(idx_A24, A24[i]);
    }

    for (int i = 0; i < n; i++) {
        ec_double_4x4x2x1(x, z, A24, x, z);
    }

    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx_B1, x[i], z[i]);
        t1[i] = VBLENDPERM(idx_B2, x[i], z[i]);
    }
    fp2_4x2x1_to_2x2x2(P1, Q1, t0);
    fp2_4x2x1_to_2x2x2(P2, Q2, t1);
}

// Doubling of two points on one curve, each in 2x2x2, with the normalized A24
// in both slots of a24. Every multiplication runs with both slots live: the two
// points share the final one, which is what a 4-slot format cannot do for a pair.
static void ec_double_pair_1x2x2x2(__m512i* P, __m512i* Q, __m512i* a24)
{
    __m512i sP[N4x2], sQ[N4x2], cP[N4x2], cQ[N4x2], rP[N4x2], rQ[N4x2], t0[N4x2], t1[N4x2];

    const __m512i dup_lo = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    const __m512i dup_hi = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    const __m512i lo_hi = VSET(0, 1, 2, 3, 12, 13, 14, 15);
    const __m512i hi_hi = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    const __m512i lo_lo = VSET(0, 1, 2, 3, 8, 9, 10, 11);

    // s = ((x + z)^2, (x - z)^2) = (A, B) per point
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(dup_lo, P[i]);
        t1[i] = VPERM512(dup_hi, P[i]);
    }
    fp2_mixaddsub_2x2x2(t0, t0, t1, 0xf);
    fp2_sqr_2x2x2(sP, t0);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(dup_lo, Q[i]);
        t1[i] = VPERM512(dup_hi, Q[i]);
    }
    fp2_mixaddsub_2x2x2(t0, t0, t1, 0xf);
    fp2_sqr_2x2x2(sQ, t0);

    // c = (C, C) with C = A - B
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(dup_lo, sP[i]);
        t1[i] = VPERM512(dup_hi, sP[i]);
    }
    fp2_sub_2x2x2(cP, t0, t1);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(dup_lo, sQ[i]);
        t1[i] = VPERM512(dup_hi, sQ[i]);
    }
    fp2_sub_2x2x2(cQ, t0, t1);

    // (A, C) * (B, A24) = (X', C*A24)
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(lo_hi, sP[i], cP[i]);
        t1[i] = VBLENDPERM(hi_hi, sP[i], a24[i]);
    }
    fp2_mul_2x2x2(rP, t0, t1);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(lo_hi, sQ[i], cQ[i]);
        t1[i] = VBLENDPERM(hi_hi, sQ[i], a24[i]);
    }
    fp2_mul_2x2x2(rQ, t0, t1);

    // Z' = C * (C*A24 + B), both points in the one multiplication
    fp2_add_2x2x2(sP, rP, sP);
    fp2_add_2x2x2(sQ, rQ, sQ);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(hi_hi, sP[i], sQ[i]);
        t1[i] = VBLENDPERM(lo_hi, cP[i], cQ[i]);
    }
    fp2_mul_2x2x2(t0, t0, t1);

    for (int i = 0; i < N4x2; i++) {
        P[i] = VBLENDPERM(lo_lo, rP[i], t0[i]);
        Q[i] = VBLENDPERM(lo_hi, rQ[i], t0[i]);
    }
}

// n doublings of P and Q on a single curve, points in 2x2x2 and the curve as (A, 1).
static void ec_double_iter_PQ_2x2x2(__m512i* P, __m512i* Q, __m512i* normA, const int n)
{
    __m512i a24[N4x2];
    const __m512i dup_lo = VSET(0, 1, 2, 3, 0, 1, 2, 3);

    ec_normA_to_normA24_1x2x2x2(a24, normA);
    for (int i = 0; i < N4x2; i++) {
        a24[i] = VPERM512(dup_lo, a24[i]);
    }

    for (int i = 0; i < n; i++) {
        ec_double_pair_1x2x2x2(P, Q, a24);
    }
}

// Basis of E[2^e] from a hint. The difference point is only deterministic at the
// order it is computed at, so clear the cofactor to 2^e before computing it.
void ec_curve_to_basis_from_hint_SQI_1x2x2x2(__m512i* res_basis, __m512i* normA, const uint8_t hint, const int e)
{
    assert(e <= TORSION_EVEN_POWER);

    ec_curve_to_basis_from_hint_1x2x2x2_inner_set_P_Q(res_basis, res_basis+2*N4x2, normA, hint);

    // Clear the 2-power with plain doublings, then the odd part with the ladder.
    ec_double_iter_PQ_2x2x2(res_basis, res_basis + 2*N4x2, normA, TORSION_EVEN_POWER - e);
    ec_scalar_mul_pair_normA_2x2x2(res_basis, res_basis + 2*N4x2, res_basis, res_basis + 2*N4x2, normA, normA, COFACTOR, COFACTOR_NBITS);

    // compute PmQ, set PmQ to Q to ensure Q above (0,0)
    ec_proj_diff_SQI_1x2x2x2(res_basis+N4x2, res_basis, res_basis+2*N4x2, normA);
}

void ec_curve_to_basis_from_hint_2x2x2x2(__m512i* res_basis1, __m512i* res_basis2, __m512i* normA1, __m512i* normA2, const uint8_t hint1, const uint8_t hint2, const int e)
{
    __m512i P[NUM_LIMBS], Q[NUM_LIMBS], A[NUM_LIMBS];

    assert(e <= TORSION_EVEN_POWER);

    // Faster to do separately
    ec_curve_to_basis_from_hint_1x2x2x2_inner_set_P_Q(res_basis1, res_basis1+2*N4x2, normA1, hint1);
    ec_curve_to_basis_from_hint_1x2x2x2_inner_set_P_Q(res_basis2, res_basis2+2*N4x2, normA2, hint2);

    // Clear the 2-power with plain doublings, then the odd part with the ladder.
    ec_double_iter_PQ_pair_2x2x2(res_basis1, res_basis1 + 2*N4x2, res_basis2, res_basis2 + 2*N4x2, normA1, normA2, TORSION_EVEN_POWER - e);
    ec_scalar_mul_pair_normA_2x2x2(res_basis1, res_basis1 + 2*N4x2, res_basis1, res_basis1 + 2*N4x2, normA1, normA1, COFACTOR, COFACTOR_NBITS);
    ec_scalar_mul_pair_normA_2x2x2(res_basis2, res_basis2 + 2*N4x2, res_basis2, res_basis2 + 2*N4x2, normA2, normA2, COFACTOR, COFACTOR_NBITS);

    // compute PmQ, set PmQ to Q to ensure Q above (0,0)
    fp2_2x2x2_to_4x2x1(P, res_basis1, res_basis2);
    fp2_2x2x2_to_4x2x1(Q, res_basis1 + 2*N4x2, res_basis2 + 2*N4x2);
    fp2_2x2x2_to_4x2x1(A, normA1, normA2);
    ec_proj_diff_pair_4x2x1(Q, P, Q, A);
    fp2_4x2x1_to_2x2x2(res_basis1 + N4x2, res_basis2 + N4x2, Q);
}

// Recover y-coordinate of a normalized x point on the Montgomery curve y^2 = x^3 + Ax^2 + x
// Returns (x, y, 1, -) in 4x2x1.
// Note: There is essentially no parallelization, second slot unused. We only use the pair version
bool ec_recover_y_2x2x2(__m512i* res, __m512i* normX, __m512i* normA, const bool verify)
{ 
    __m512i t0[N4x2], t1[N4x2], idx;
    bool is_square = true;
    
    fp2_add_2x2x2(t0, normX, normA); // x + A
    fp2_sqr_2x2x2(t1, normX); // x^2
    fp2_mul_2x2x2(t0, t0, t1); // x^3 + Ax^2
    fp2_add_2x2x2(t1, t0, normX); // x^2 + Ax + x

    fp2_sqrt_2x2x2(res, t1);

    if (verify) {
        fp2_sqr_2x2x2(t0, res);
        is_square = (fp2_is_equal_2x2x2(t0, t1) & 0x1) == 0x1;
    }

    // t0 = (x, y), t1 = (1, 0)
    idx = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx, normX[i], res[i]);
    }
    fp2_set_binary_2x2x2(t1, 0b01);

    // res = (x, y, z, 0)
    fp2_2x2x2_to_4x2x1(res, t0, t1);

    return is_square;
}

// Lifts a basis x(P), x(Q), x(P-Q) assuming the curve's A coefficient and x(P) are normalized.
// Outputs res_P = (xP, yP, 1) and res_Q = (xQ, yQ, zQ) in Jacobian coordinates.
bool lift_basis_normalized_2x2x2(__m512i* res_P, __m512i* res_Q, __m512i* basis, __m512i* normA, const bool verify)
{
    __m512i P_2x2x2[2*N4x2], t0[N4x2], t1[N4x2], t2[N4x2], t3[N4x2];
    __m512i idx0, idx1;

    bool is_sqr = ec_recover_y_2x2x2(res_P, basis, normA, verify);
    fp2_4x2x1_to_2x2x2(P_2x2x2, P_2x2x2 + N4x2, res_P); // (x,y), (z, 1)

    // Algorithm of Okeya-Sakurai to recover y.Q in the Montgomery model

    // t0 = (P.x * Q.x, P.x * Q.z)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VPERM512(idx0, P_2x2x2[i]);
    }
    fp2_mul_2x2x2(t0, t0, basis + N4x2);

    // t1 = (Q.x, Q.x)
    // t2 = (t0[1], t0[1])
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < N4x2; i++) {
        t1[i] = VPERM512(idx0, basis[i + N4x2]);
        t2[i] = VPERM512(idx1, t0[i]);
    }
    // t1 = (Q.x + t0[0], Q.x - t0[0])
    fp2_mixaddsub_2x2x2(t1, t1, t2, 0xf);

    // t0 = (t0[0], A)
    // t2 = (Q.z, A)
    idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    idx1 = VSET(4, 5, 6, 7, 8, 9, 10, 11);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], normA[i]);
        t2[i] = VBLENDPERM(idx1, basis[i+N4x2], normA[i]);
    }
    // t0 = (t0[0] + Q.z, 2*A)
    fp2_add_2x2x2(t0, t0, t2);

    // t2 = (t1[1], t0[1])
    // t3 = (PmQ.x, Q.z)
    idx0 = VSET(12, 13, 14, 15, 4, 5, 6, 7);
    idx1 = VSET(8, 9, 10, 11, 4, 5, 6, 7);
    for (int i = 0; i < N4x2; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t3[i] = VBLENDPERM(idx1, basis[i + N4x2], basis[i + 2*N4x2]);
    }
    fp2_mul_2x2x2(t2, t2, t3);
    
    // t1[0] += t2[1]
    idx0 = VSET(4, 5, 6, 7, 0, 0, 0, 0);
    for (int i = 0; i < N4x2; i++) {
        t3[i] = VPERM512MASKZ(0xf, t2[i], idx0); // (t2[1], 0)
    }
    fp2_add_2x2x2(t1, t1, t3);

    // t0[1] = Q.z
    // t3 = (t1[0], t2[1])
    idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], basis[i + N4x2]);
        t3[i] = VBLENDPERM(idx0, t1[i], t2[i]);
    }
    fp2_mul_2x2x2(t0, t0, t3);

    // t3 = (t0[1], P.y)
    // t0 = (t0[0], P.y)
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < N4x2; i++) {
        t3[i] = VBLENDPERM(idx1, t0[i], P_2x2x2[i]);
        t0[i] = VBLENDPERM(idx0, t0[i], P_2x2x2[i]);
    }
    fp2_mixaddsub_2x2x2(t0, t0, t3, 0xf0);

    // t3 = (PmQ.z, Q.z)
    for (int i = 0; i < N4x2; i++) {
        t3[i] = VBLENDPERM(idx1, basis[i + 2*N4x2], basis[i + N4x2]);
    }
    fp2_mul_2x2x2(t0, t0, t3);

    // t1 = (t1[1], PmQ.z)
    // t2 = (t2[0], t0[1])
    idx0 = VSET(8, 9, 10, 11, 4, 5, 6, 7);
    for (int i = 0; i < N4x2; i++) {
        t1[i] = VBLENDPERM(idx1, t1[i], basis[i + 2*N4x2]);
        t2[i] = VBLENDPERM(idx0, t0[i], t2[i]);
    }
    fp2_mul_2x2x2(t1, t1, t2);
    
    // t0 = (res_yQ, -)
    fp2_sub_2x2x2(t0, t1, t0);

    // t1 = (res_xQ, res_zQ)
    idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < N4x2; i++) {
        t1[i] = VPERM512(idx0, t1[i]);
    }
    fp2_mul_2x2x2(t1, t1, basis + N4x2);

    // Transforming to Jacobian coordinates

    // t2 = (-, res_zQ^2)
    fp2_copy_2x2x2(t2, t1);
    fp2_sqr_2x2x2(t2, t2);

    // t0 = (res_xQ, res_yQ)
    // t1 = (res_zQ, res_zQ^2)
    idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    for (int i = 0; i < N4x2; i++) {
        t0[i] = VBLENDPERM(idx0, t1[i], t0[i]);
        t1[i] = VBLENDPERM(idx1, t1[i], t2[i]);
    }
    fp2_mul_2x2x2(t0, t0, t1);

    // Convert to 4x2x1
    fp2_2x2x2_to_4x2x1(res_Q, t0, t1);

    return is_sqr;
}

// Takes as input a normalized x-only point x1 on the curve with normalized A-coefficient A1, and x2 on A2.
// It returns the y coordinate of each point, as (y_1, y_2) in 2x2x2.
bool ec_recover_y_pair_2x2x2(__m512i* res, __m512i* normX1, __m512i* normX2, __m512i* normA1, __m512i* normA2, const bool verify)
{ 
    __m512i t0[N4x2], t1[N4x2], X1_X2[N4x2], idx;
    bool is_square = true;

    // X1_X2 = (X1, X2), t0 = (A1, A2)
    idx = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    for (int i = 0; i < N4x2; i++) {
        X1_X2[i] = VBLENDPERM(idx, normX1[i], normX2[i]);
        t0[i] = VBLENDPERM(idx, normA1[i], normA2[i]);
    }

    fp2_add_2x2x2(t0, t0, X1_X2); // x + A
    fp2_sqr_2x2x2(t1, X1_X2); // x^2
    fp2_mul_2x2x2(t0, t0, t1); // x^3 + Ax^2
    fp2_add_2x2x2(t1, t0, X1_X2); // x^2 + Ax + x

    fp2_sqrt_2x2x2(res, t1);

    if (verify) {
        fp2_sqr_2x2x2(t0, res);
        is_square = (fp2_is_equal_2x2x2(t0, t1) & 0x3) == 0x3;
    }

    return is_square;
}

// Lifts a pair of bases x(P), x(Q), x(P-Q) assuming the curve's A coefficient and x(P) are normalized.
// Outputs each basis (P, Q) in Jacobian coordinates, as res_x = (P1.X, Q1.X, P2.X, Q2.X) etc.
bool lift_basis_pair_normalized_2x2x2(
    __m512i* res_x, __m512i* res_y, __m512i* res_z,
    __m512i* basis1, __m512i* basis2, __m512i* normA1, __m512i* normA2, 
    const bool verify
)
{
    __m512i yP_2x2x2[N4x2], xP_yP[NUM_LIMBS], xQ_zQ[NUM_LIMBS], xPmQ_zPmQ[NUM_LIMBS], A1_A2[NUM_LIMBS];
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], t2[NUM_LIMBS], t3[2*N4x2];
    __m512i idx0, idx1;

    bool is_sqr = ec_recover_y_pair_2x2x2(yP_2x2x2, basis1, basis2, normA1, normA2, verify);

    // Algorithm of Okeya-Sakurai to recover y.Q in the Montgomery model
    // Preparing values in 4x2x1
    idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    for (int i = 0; i < N4x2; i++) {
        xP_yP[i] = VBLENDPERM(idx0, basis1[i], basis2[i]); // (P1.x, P2.x)
    }
    fp2_2x2x2_to_4x2x1(xP_yP, xP_yP, yP_2x2x2); // (P1.x, P2.x, P1.y, P2.y)
    fp2_2x2x2_to_4x2x1(xQ_zQ, basis1 + N4x2, basis2 + N4x2); // (Q1.x, Q1.z, Q2.x, Q2.z)
    fp2_2x2x2_to_4x2x1(xPmQ_zPmQ, basis1 + 2*N4x2, basis2 + 2*N4x2); // (PmQ1.x, PmQ1.z, PmQ2.x, PmQ2.z)
    fp2_2x2x2_to_4x2x1(A1_A2, normA1, normA2); // (A1, 1, A2, 1)


    // t0 = (P.x * Q.x, P.x * Q.z)
    idx0 = VSET(0, 1, 0, 1, 2, 3, 2, 3);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, xP_yP[i]);
    }
    fp2_mul_4x2x1(t0, t0, xQ_zQ);

    // t1 = (Q.x, Q.x)
    // t2 = (t0[1], t0[1])
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, xQ_zQ[i]);
        t2[i] = VPERM512(idx1, t0[i]);
    }
    // t1 = (Q.x + t0[0], Q.x - t0[0])
    fp2_mixaddsub_4x2x1(t1, t1, t2, 0x33);

    // t0 = (t0[0], A)
    // t2 = (Q.z, A)
    idx0 = VSET(0, 1, 8, 9, 4, 5, 12, 13);
    idx1 = VSET(2, 3, 8, 9, 6, 7, 12, 13);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], A1_A2[i]);
        t2[i] = VBLENDPERM(idx1, xQ_zQ[i], A1_A2[i]);
    }
    // t0 = (t0[0] + Q.z, 2*A)
    fp2_add_4x2x1(t0, t0, t2);

    // t2 = (t1[1], t0[1])
    // t3 = (PmQ.x, Q.z)
    idx0 = VSET(10, 11, 2, 3, 14, 15, 6, 7);
    idx1 = VSET(8, 9, 2, 3, 12, 13, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t3[i] = VBLENDPERM(idx1, xQ_zQ[i], xPmQ_zPmQ[i]);
    }
    fp2_mul_4x2x1(t2, t2, t3);
    
    // t1[0] += t2[1] 
    idx0 = VSET(2, 3, 0, 0, 6, 7, 0, 0);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t3[i] = VPERM512MASKZ(0x33, t2[i], idx0); // (t2[1], 0)
    }
    fp2_add_4x2x1(t1, t1, t3);

    // t0[1] = Q.z
    // t3 = (t1[0], t2[1])
    idx0 = VSET(0, 1, 10, 11, 4, 5, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], xQ_zQ[i]);
        t3[i] = VBLENDPERM(idx0, t1[i], t2[i]);
    }
    fp2_mul_4x2x1(t0, t0, t3);

    // t3 = (t0[1], P.y)
    // t0 = (t0[0], P.y)
    idx1 = VSET(2, 3, 12, 13, 6, 7, 14, 15);
    idx0 = VSET(0, 1, 12, 13, 4, 5, 14, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t3[i] = VBLENDPERM(idx1, t0[i], xP_yP[i]);
        t0[i] = VBLENDPERM(idx0, t0[i], xP_yP[i]);
    }
    fp2_mixaddsub_4x2x1(t0, t0, t3, 0xcc);

    // t3 = (PmQ.z, Q.z)
    idx0 = VSET(10, 11, 2, 3, 14, 15, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t3[i] = VBLENDPERM(idx0, xQ_zQ[i], xPmQ_zPmQ[i]);
    }
    fp2_mul_4x2x1(t0, t0, t3);

    // t1 = (t1[1], PmQ.z)
    // t2 = (t2[0], t0[1])
    idx0 = VSET(2, 3, 10, 11, 6, 7, 14, 15);
    idx1 = VSET(8, 9, 2, 3, 12, 13, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VBLENDPERM(idx0, t1[i], xPmQ_zPmQ[i]);
        t2[i] = VBLENDPERM(idx1, t0[i], t2[i]);
    }
    fp2_mul_4x2x1(t1, t1, t2);
    
    // t0 = (res_yQ, -)
    fp2_sub_4x2x1(t0, t1, t0);

    // t1 = (res_xQ, res_zQ)
    idx0 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t1[i]);
    }
    fp2_mul_4x2x1(t1, t1, xQ_zQ);
    
    // Transforming to Jacobian coordinates
    // t2 = (-, res_zQ^2)
    // In the pair version, we transform to 2x2x2 and square both
    idx0 = VSET(2, 3, 6, 7, 2, 3, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VPERM512(idx0, t1[i]);
    }
    fp2_4x2x1_to_2x2x2(t3, t3 + N4x2, t2);
    fp2_sqr_2x2x2(t3 + N4x2, t3 + N4x2);
    fp2_2x2x2_to_4x2x1(t2, t3, t3 + N4x2); // (res_zQ1, res_zQ2, res_zQ1^2, res_zQ2^2)

    // t0 = (res_xQ1, res_xQ2, res_yQ1, res_yQ2)
    idx0 = VSET(0, 1, 4, 5, 8, 9, 12, 13);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, t1[i], t0[i]);
    }
    fp2_mul_4x2x1(t0, t0, t2);

    // Rearrange to output format
    // res_y = (P1.x, res_xQ1, P2.x, res_xQ2)
    idx0 = VSET(0, 1, 8, 9, 2, 3, 10, 11);
    // res_y = (P1.y, res_yQ1, P2.y, res_yQ2)
    idx1 = VSET(4, 5, 12, 13, 6, 7, 14, 15);
    // res_z = (1, res_zQ1, 1, res_zQ2)
    fp2_set_binary_4x2x1(t3, 0b1111);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res_x[i] = VBLENDPERM(idx0, xP_yP[i], t0[i]);
        res_y[i] = VBLENDPERM(idx1, xP_yP[i], t0[i]);
        res_z[i] = VBLENDPERM(idx0, t3[i], t2[i]);
    }

    return is_sqr;
}

// Debugging functions

bool test_point_order_twof_1x2x2x2(__m512i* P, __m512i* A24, unsigned int f)
{
    __m512i Q[N4x2];
    fp2_copy_2x2x2(Q, P);

    int z_is_zero = fp2_is_zero_2x2x2(Q) & 0x2;
    if (z_is_zero || f == 0) {
        return f == 0 && z_is_zero;
    }

    ec_double_iter_1x2x2x2(Q, Q, A24, f-1);
    z_is_zero = fp2_is_zero_2x2x2(Q) & 0x2;
    if (z_is_zero) {
        return false;
    }

    ec_double_1x2x2x2(Q, Q, A24);
    z_is_zero = fp2_is_zero_2x2x2(Q) & 0x2;
    return z_is_zero;
}

bool test_basis_order_twof_1x2x2x2(__m512i* B, __m512i* A24, unsigned int f)
{
    return 
        test_point_order_twof_1x2x2x2(B, A24, f) &
        test_point_order_twof_1x2x2x2(B + N4x2, A24, f) &
        test_point_order_twof_1x2x2x2(B + 2*N4x2, A24, f);
}

// =============================== Jacobian arithmetic ====================================================


void jac_mul_pair(__m512i* res, __m512i* P, __m512i* normA, const digit_t* m0, const digit_t* m1)
{
    __m512i X0_X1_XD_A[2*NUM_LIMBS], Y0_Y1_YD_A[2*NUM_LIMBS], R1_R2_S1_S2[2*NUM_LIMBS], t[2*N4x2];
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS];
    __m512i idx0, idx1, idx2, idx3;
    digit_t s0[NWORDS_ORDER], s1[NWORDS_ORDER];

    mp_copy(s0, m0, NWORDS_ORDER);
    mp_copy(s1, m1, NWORDS_ORDER);

    // Convert from Jacobian to x-only Montgomery
    // t = (xP, zP) = (X, Z^2)
    fp2_4x2x1_to_2x2x2(t, t + N4x2, P);
    fp2_sqr_2x2x2(t + N4x2, t + N4x2);
    idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    for (int i = 0; i < N4x2; i++) {
        t[i] = VBLENDPERM(idx0, t[i], t[i+N4x2]);
    }

    // x-only Montgomery ladder
    // We don't know that s0 and s1 have the same bit-length,
    // so we do the longer one on its own first
    int len0 = mp_bitlength(s0, NWORDS_ORDER);
    int len1 = mp_bitlength(s1, NWORDS_ORDER);
    int len = len0;

    // X0 = 0, X1 = P, XD = P
    ec_set_zero_1x2x2x2(X0_X1_XD_A);
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A, X0_X1_XD_A, t);
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A + NUM_LIMBS, t, normA);
    fp2_copy_4x2x1(t, X0_X1_XD_A); // save for later, t = (1, 0, xP, zP)

    if (len0 != len1) {
        int from_bit, to_bit;
        digit_t* s;
        if (len0 < len1) {
            from_bit = len0; to_bit = len1-1; s = s1; 
            len = len0;
        } else {
            from_bit = len1; to_bit = len0-1; s = s0; 
            len = len1;
        }

        // output is X0_X1
        ec_scalar_mul_normA_inner_1x4x2x1(X0_X1_XD_A, X0_X1_XD_A + NUM_LIMBS, s, from_bit, to_bit);
    }

    // Convert to 8x1x1 for joint ladder
    if (len0 < len1) {
        fp2_4x2x1_to_8x1x1(Y0_Y1_YD_A, X0_X1_XD_A, X0_X1_XD_A + NUM_LIMBS);
        fp2_4x2x1_to_8x1x1(X0_X1_XD_A, t, X0_X1_XD_A + NUM_LIMBS);
    } else if (len0 > len1) {
        fp2_4x2x1_to_8x1x1(Y0_Y1_YD_A, t, X0_X1_XD_A + NUM_LIMBS);
        fp2_4x2x1_to_8x1x1(X0_X1_XD_A, X0_X1_XD_A, X0_X1_XD_A + NUM_LIMBS);
    } else {
        fp2_4x2x1_to_8x1x1(X0_X1_XD_A, X0_X1_XD_A, X0_X1_XD_A + NUM_LIMBS);
        fp2_copy_8x1x1(Y0_Y1_YD_A, X0_X1_XD_A);
    }

    // joint ladder
    for (int i = len-1; i >= 0; i--) {
        unsigned int bit0 = (s0[i >> LOG2RADIX] >> (i & (RADIX - 1))) & 1;
        unsigned int bit1 = (s1[i >> LOG2RADIX] >> (i & (RADIX - 1))) & 1;

        // swap X0 and X1
        if (bit0) { 
            idx0 = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int k = 0; k < 2*NUM_LIMBS; k++) {
                X0_X1_XD_A[k] = VPERM512(idx0, X0_X1_XD_A[k]);
            }
        }
        if (bit1) {
            idx0 = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int k = 0; k < 2*NUM_LIMBS; k++) {
                Y0_Y1_YD_A[k] = VPERM512(idx0, Y0_Y1_YD_A[k]);
            }
        }

        ec_double_add_pair_normA_8x1x1(R1_R2_S1_S2, X0_X1_XD_A, Y0_Y1_YD_A);

        if (!bit0) { // X0_X1 = R1_R2
            idx0 = VSET(8, 1, 10, 3, 12, 5, 14, 7);
        } else { // swap
            idx0 = VSET(12, 1, 14, 3, 8, 5, 10, 7);
        }
        for (int k = 0; k < 2*NUM_LIMBS; k++) {
            X0_X1_XD_A[k] = VBLENDPERM(idx0, X0_X1_XD_A[k], R1_R2_S1_S2[k]);
        }

        if (!bit1) { // Y0_Y1 = S1_S2
            idx0 = VSET(9, 1, 11, 3, 13, 5, 15, 7);
            
        } else { // swap
            idx0 = VSET(13, 1, 15, 3, 9, 5, 11, 7);
        }
        for (int k = 0; k < 2*NUM_LIMBS; k++) {
            Y0_Y1_YD_A[k] = VBLENDPERM(idx0, Y0_Y1_YD_A[k], R1_R2_S1_S2[k]);
        }
    }

    // R1_R2 = (X0, Z0, X1, Z1)
    idx0 = VSET(0, 8, 2, 10, 4, 12, 6, 14);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        R1_R2_S1_S2[i] = VBLENDPERM(idx0, X0_X1_XD_A[i], Y0_Y1_YD_A[i]);
    }


    // We will now compute the result in Jacobian coordinates.
    // Most of the work is in recovering the y-coordinates.

    // T0 = (PX, PY, PZ, -) = (X * Z, Y, Z * zP, -)
    idx0 = VSET(12, 13, 0, 1, 6, 7, 2, 3);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, t[i], P[i]); // (Z, 1, zP, 0)
    }
    fp2_mul_4x2x1(t0, t0, P);

    // Pair computation in 8x1x1
    // T0 = (PX, PX, PZ, PZ)
    idx0 = VSET(0, 0, 0, 0, 4, 4, 4, 4);
    idx1 = VSET(1, 1, 1, 1, 5, 5, 5, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i + NUM_LIMBS] = VPERM512(idx1, t0[i]);
        t0[i] = VPERM512(idx0, t0[i]);
    }
    // T1 = (X0, Z0, X0, Z0) for each res
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, R1_R2_S1_S2[i]);
    }
    // T0 = (tt, t1, t2, t3)
    fp2_mul_8x1x1(t0, t0, t1);

    // Registers:
    // T0 = (tt, t1, t2, t3)

    // T1 = (tt, A, t1, Y)
    idx0 = VSET(0, 1, 13, 13, 2, 3, 0, 0);
    idx1 = VSET(0, 1, 2, 3, 4, 5, 10, 10);
    idx2 = VSET(0, 1, 2, 3, 4, 5, 11, 11);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i], X0_X1_XD_A[i]);
        t1[i+NUM_LIMBS] = VBLENDPERM(idx0, t0[i+NUM_LIMBS], X0_X1_XD_A[i+NUM_LIMBS]);
        t1[i] = VBLENDPERM(idx1, t1[i], P[i]);
        t1[i+NUM_LIMBS] = VBLENDPERM(idx2, t1[i+NUM_LIMBS], P[i]);
    }

    // T2 = (t3, A, t2, Y)
    idx0 = VSET(6, 7, 13, 13, 4, 5, 0, 0);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], X0_X1_XD_A[i]);
        t2[i+NUM_LIMBS] = VBLENDPERM(idx0, t0[i+NUM_LIMBS], X0_X1_XD_A[i+NUM_LIMBS]);
        t2[i] = VBLENDPERM(idx1, t2[i], P[i]);
        t2[i+NUM_LIMBS] = VBLENDPERM(idx2, t2[i+NUM_LIMBS], P[i]);
    }
    fp2_add_8x1x1(t1, t1, t2);

    // Registers:
    // T0 = (-, t1, t2, t3)
    // T1 = (t0, t4, t5, v0)

    // T2 = (t3, t3), (t4, v0)
    idx0 = VSET(6, 14, 6, 14, 7, 15, 7, 15);
    idx1 = VSET(2, 10, 6, 14, 3, 11, 7, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], t0[i+NUM_LIMBS]);
        t2[i+NUM_LIMBS] = VBLENDPERM(idx1, t1[i], t1[i+NUM_LIMBS]);
    }
    fp2_mul_4x2x1(t2, t2, t2+NUM_LIMBS);

    // Registers:
    // T0 = (-, t1, t2, t3)
    // T1 = (t0, -, t5, -)
    // T2 = (t4, v0)

    // T1 = (t0, 0, t5, 0)
    // T2 = (0, t4, t4, v0)
    idx0 = VSET(0, 0, 0, 4, 0, 4, 2, 6);
    idx1 = VSET(0, 0, 1, 5, 1, 5, 3, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM256MASKZ(0x33, t1[i], 0xe4);
        t1[i+NUM_LIMBS] = VPERM256MASKZ(0x33, t1[i+NUM_LIMBS], 0xe4);
        t2[i+NUM_LIMBS] = VPERM512MASKZ(0xfc, t2[i], idx1);
        t2[i] = VPERM512MASKZ(0xfc, t2[i], idx0);
    }
    fp2_add_8x1x1(t1, t1, t2);

    // Registers:
    // T0 = (-, t1, t2, t3)
    // T1 = (t0, t4, t5, v0)

    // T2 = (t5, t3, -, Z1)
    idx0 = VSET(12, 13, 6, 7, 0, 0, 0, 0);
    idx1 = VSET(0, 1, 2, 3, 4, 5, 14, 15);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t2[i] = VBLENDPERM(idx1, t2[i], R1_R2_S1_S2[i]);
    }
    fp2_mul_8x1x1(t1, t1, t2);

    // Registers:
    // T0 = (-, t1, t2, -)
    // T1 = (t5, t6, -, v0)

    // T0 = (v0, t1, t5, v0)
    // T2 = (0,  t2, t6, 0)
    idx0 = VSET(14, 15, 2, 3, 8, 9, 14, 15);
    idx2 = VSET(0, 0, 4, 5, 10, 11, 0, 0);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t2[i] = VBLENDPERMMASKZ(0x3c, t0[i], t1[i], idx2);
        t0[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }
    fp2_sub_8x1x1(t0, t0, t2);

    // Registers:
    // T0 = (v0, t0, u0, v0)

    // T1 = (X0, t0, Z1, Z0)
    idx0 = VSET(8, 9, 2, 3, 14, 15, 10, 11);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i], R1_R2_S1_S2[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    // Registers:
    // T0 = (RX, t0, RY, RZ) 

    // T0 = (RX, t0), (RY, RZ)
    idx0 = VSET(0, 8, 2, 10, 1, 9, 3, 11);
    idx1 = VSET(4, 12, 6, 14, 5, 13, 7, 15);
    for (int i = 0; i < NUM_LIMBS; i++) {
        __m512i r0 = t0[i], r1 = t0[i+NUM_LIMBS];
        t0[i] = VBLENDPERM(idx0, r0, r1);
        t0[i+NUM_LIMBS] = VBLENDPERM(idx1, r0, r1);
    }
    
    // T1 = (RZ, X1)
    idx0 = VSET(2, 3, 12, 0, 6, 7, 13, 0);
    idx1 = VSET(0, 1, 2, 12, 4, 5, 6, 13);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i+NUM_LIMBS], R1_R2_S1_S2[i]);
        t1[i] = VBLENDPERM(idx1, t1[i], R1_R2_S1_S2[i+NUM_LIMBS]);
    }
    fp2_mul_4x2x1(t0, t0, t1);

    // Registers:
    // T0 = (RX, t0), (RY, RZ)

    // T1 = (t0, 0)
    idx0 = VSET(2, 3, 0, 0, 6, 7, 0, 0);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512MASKZ(0x33, t0[i], idx0);
    }
    fp2_sub_4x2x1(t0+NUM_LIMBS, t0+NUM_LIMBS, t1);
    fp2_sqr_4x2x1(t1, t0+NUM_LIMBS);

    // Registers:
    // T0 = (RX, -), (RY, RZ)
    // T1 = (-, t0)
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VPERM512MASKZ(0x33, t1[i], idx0);
    }

    fp2_mul_4x2x1(t1, t1, t0+NUM_LIMBS);

    // T0 = (RX, -), (-, RZ)
    // T1 = (RY, 0)

    // res = (RX, RY, RZ, 0)
    idx0 = VSET(0, 4, 8, 12, 0, 0, 10, 14);
    idx1 = VSET(1, 5, 9, 13, 0, 0, 11, 15);

    idx2 = VSET(0, 1, 2, 3, 10, 14, 6, 7);
    idx3 = VSET(0, 1, 2, 3, 11, 15, 6, 7);

    for (int i = 0; i < NUM_LIMBS; i++) {
        res[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        res[i+NUM_LIMBS] = VBLENDPERM(idx1, t0[i], t1[i]);
        res[i] = VBLENDPERM(idx2, res[i], t0[i+NUM_LIMBS]);
        res[i+NUM_LIMBS] = VBLENDPERM(idx3, res[i+NUM_LIMBS], t0[i+NUM_LIMBS]);
    }
}

void jac_add_pair(__m512i* res, __m512i* P1_P2, __m512i* Q1_Q2, __m512i* normA)
{
    __m512i A_dup[2*NUM_LIMBS];
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS], t3[2*NUM_LIMBS], t4[2*NUM_LIMBS];
    __m512i idx0, idx1, idx2, idx3;

    fp2_2x2x2_to_4x2x1(A_dup, normA, normA);
    fp2_4x2x1_to_8x1x1(A_dup, A_dup, A_dup);

    // T0 = (zP, zQ, xP, zP)
    // T1 = (zP, zQ, xP, zQ)
    idx0 = VSET(4, 5, 12, 13, 0, 1, 4, 5);
    idx1 = VSET(4, 5, 12, 13, 0, 1, 12, 13);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, P1_P2[i], Q1_Q2[i]);
        t1[i] = VBLENDPERM(idx1, P1_P2[i], Q1_Q2[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    // Registers:
    // T0 = (t0, t1, t4, t5)

    // T1 = (t1, t0, t0, t1)
    // T2 = (zQ, zP, xQ, xP)
    idx0 = VSET(2, 3, 0, 1, 0, 1, 2, 3);
    idx1 = VSET(12, 13, 4, 5, 8, 9, 0, 1);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VBLENDPERM(idx1, P1_P2[i], Q1_Q2[i]);
    }
    fp2_mul_8x1x1(t1, t1, t2);

    // Registers:
    // T0 = (t0, -, t4, t5)
    // T1 = (v1, t2, u2, u1)

    // T2 = (v1, t2, A, t5)
    // T3 = (yP, yQ, xP, t5)
    idx0 = VSET(0, 1, 2, 3, 0, 0, 14, 15);
    idx1 = VSET(0, 1, 2, 3, 8, 8, 6, 7);

    idx2 = VSET(2, 3, 10, 11, 0, 1, 0, 0);
    idx3 = VSET(0, 1, 2, 3, 4, 5, 14, 15);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx0, t1[i], t0[i]);
        t2[i] = VBLENDPERM(idx1, t2[i], A_dup[i]);
        t3[i] = VBLENDPERM(idx2, P1_P2[i], Q1_Q2[i]);
        t3[i] = VBLENDPERM(idx3, t3[i], t0[i]);
    }
    fp2_mul_8x1x1(t2, t2, t3);

    // Registers:
    // T0 = (t0, -, t4, t5)
    // T1 = (-, -, u2, u1)
    // T2 = (v1, t2, t3, t8)

    // T3 = (t2, u2, yP, t3)
    // T4 = (v1, u1, yP, t3)
    idx0 = VSET(10, 11, 4, 5, 0, 0, 12, 13);
    idx1 = VSET(0, 1, 2, 3, 10, 11, 6, 7);

    idx2 = VSET(8, 9, 6, 7, 0, 0, 12, 13);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t3[i] = VBLENDPERM(idx0, t1[i], t2[i]);
        t4[i] = VBLENDPERM(idx2, t1[i], t2[i]);
        t3[i] = VBLENDPERM(idx1, t3[i], P1_P2[i]);
        t4[i] = VBLENDPERM(idx1, t4[i], P1_P2[i]);
    }
    fp2_mixaddsub_8x1x1(t3, t3, t4, 0xf0, 0xf0);

    // Registers:
    // T0 = (t0, -, t4, t5)
    // T1 = (-, -, u2, u1)
    // T2 = (v1, -, -, t8)
    // T3 = (dy, dx, t7, t3)

    // In preparation for select, putting dx in correct position
    // T2 = (v1, dy, dx, t8)
    idx0 = VSET(0, 1, 8, 9, 10, 11, 6, 7);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t2[i] = VBLENDPERM(idx0, t2[i], t3[i]);
    }
    // if dx = 0 and dy = 0: dx = t7
    __mmask8 zmask = fp2_is_zero_8x1x1(t2);
    zmask = (zmask & (zmask << 2) & 0x30);
    fp2_select_8x1x1(t2, t2, t3, zmask);

    // Registers:
    // T0 = (t0, -, t4, t5)
    // T1 = (-, -, u2, u1)
    // T2 = (v1, dy, dx, t8)
    // T3 = (-, -, -, t3)

    // T1 = (t0, t4, u2, u1)
    // T3 = (t3, t4, u1, 0 )
    idx0 = VSET(0, 1, 4, 5, 12, 13, 14, 15);
    idx1 = VSET(14, 15, 4, 5, 6, 7, 0, 0);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t3[i] = VBLENDPERMMASKZ(0x3f, t1[i], t3[i], idx1);
    }
    fp2_add_8x1x1(t1, t1, t3);

    // Registers:
    // T0 = (t0, -, t4, t5)
    // T1 = (t3, t9, u2, u1)
    // T2 = (v1, dy, dx, t8)

    // T3 = (t0, dx, t8, t5)
    // T4 = (t3, dx, A, dx)
    idx0 = VSET(0, 1, 12, 13, 14, 15, 6, 7);
    idx1 = VSET(0, 1, 12, 13, 0, 0, 12, 13);
    idx2 = VSET(0, 1, 2, 3, 8, 8, 6, 7);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t3[i] = VBLENDPERM(idx0, t0[i], t2[i]);
        t4[i] = VBLENDPERM(idx1, t1[i], t2[i]);
        t4[i] = VBLENDPERM(idx2, t4[i], A_dup[i]);
    }
    fp2_mul_8x1x1(t3, t3, t4);

    // Registers:
    // T0 = (-, - , t4, -)
    // T1 = (-, t9, u2, u1)
    // T2 = (v1, dy, dx, -)
    // T3 = (t3, t6, xR, zR)

    // T4 = (t9, 0, u2, 0)
    idx0 = VSET(2, 3, 0, 0, 4, 5, 0, 0);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t4[i] = VPERM512MASKZ(0x33, t1[i], idx0);
    }
    fp2_add_8x1x1(t3, t3, t4);

    // Registers:
    // T0 = (-, - , t4, -)
    // T1 = (-, -, -, u1)
    // T2 = (v1, dy, dx, -)
    // T3 = (t3, t6, xR, zR)

    // T4 = (t4, 0, 0, 0)
    idx0 = VSET(4, 5, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t4[i] = VPERM512MASKZ(0x3, t0[i], idx0);
    }
    fp2_add_4x2x1(t3, t3, t4);

    // Registers:
    // T1 = (-, -, -, u1)
    // T2 = (v1, dy, dx, -)
    // T3 = (t3, t6, xR, zR)
    // T4 = (t4, 0, 0, 0)

    // T0 = (t6, zQ, t6, t6)
    // T1 = (xR, t3, u1, dx)
    idx0 = VSET(2, 3, 12, 13, 2, 3, 2, 3);
    idx1 = VSET(12, 13, 8, 9, 6, 7, 0, 0);
    idx2 = VSET(0, 1, 2, 3, 4, 5, 12, 13);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VBLENDPERM(idx0, t3[i], Q1_Q2[i]);
        t1[i] = VBLENDPERM(idx1, t1[i], t3[i]);
        t1[i] = VBLENDPERM(idx2, t1[i], t2[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    // Registers:
    // T0 = (xR, t3, yR, t4)
    // T2 = (v1, dy, -, -)
    // T3 = (-, -, -, zR)

    // If dx == 0 and dy == 0: dy = t3
    fp2_select_8x1x1(t2, t2, t0, zmask >> 2);

    // T1 = (dy, t4)
    // T2 = (dy, v1)
    idx0 = VSET(0, 0, 6, 14, 0, 0, 7, 15);
    idx1 = VSET(10, 0, 2, 3, 11, 0, 6, 7);
    idx2 = VSET(0, 10, 2, 3, 4, 11, 6, 7);

    idx3 = VSET(2, 10, 0, 8, 3, 11, 1, 9);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i], t0[i+NUM_LIMBS]);
        t1[i] = VBLENDPERM(idx1, t1[i], t2[i]);
        t1[i] = VBLENDPERM(idx2, t1[i], t2[i+NUM_LIMBS]);
        t2[i] = VBLENDPERM(idx3, t2[i], t2[i+NUM_LIMBS]);
    }
    fp2_mul_4x2x1(t1, t1, t2);

    // Registers:
    // T0 = (xR, -, yR, -)
    // T1 = (t3, t4)
    // T2 = (dy, -)
    // T3 = (-, -, -, zR)

    // T4 = (xR, 0), (yR, 0)
    idx0 = VSET(0, 8, 0, 0, 1, 9, 0, 0);
    idx1 = VSET(4, 12, 0, 0, 5, 13, 0, 0);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t4[i] = VBLENDPERMMASKZ(0x33, t0[i], t0[i+NUM_LIMBS], idx0);
        t4[i+NUM_LIMBS] = VBLENDPERMMASKZ(0x33, t0[i], t0[i+NUM_LIMBS], idx1);
    }
    fp2_sub_4x2x1(t4, t1, t4);
    fp2_sub_4x2x1(t4+NUM_LIMBS, t4+NUM_LIMBS, t4);

    // Registers:
    // T2 = (dy, -)
    // T3 = (-, -, -, zR)
    // T4 = (xR, t4), (yR, -)

    fp2_mul_4x2x1(t4+NUM_LIMBS, t4+NUM_LIMBS, t2);

    // Registers:
    // T3 = (-, -, -, zR)
    // T4 = (xR, t4), (yR, -)

    // T0 = (t4, -)
    idx0 = VSET(2, 3, 0, 0, 6, 7, 0, 0);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, t4[i]);
    }
    fp2_sub_4x2x1(t4+NUM_LIMBS, t4+NUM_LIMBS, t0);

    // T3 = (-, -, -, zR)
    // T4 = (xR, -), (yR, -)

    // res = (xR, yR, zR, 0)
    idx0 = VSET(0, 4, 8, 12, 0, 0, 0, 0);
    idx1 = VSET(1, 5, 9, 13, 0, 0, 0, 0);
    idx2 = VSET(0, 1, 2, 3, 14, 15, 0, 0);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res[i] = VBLENDPERM(idx0, t4[i], t4[i+NUM_LIMBS]);
        res[i+NUM_LIMBS] = VBLENDPERM(idx1, t4[i], t4[i+NUM_LIMBS]);
        res[i] = VBLENDPERMMASKZ(0x3f, res[i], t3[i], idx2);
        res[i+NUM_LIMBS] = VBLENDPERMMASKZ(0x3f, res[i+NUM_LIMBS], t3[i+NUM_LIMBS], idx2);
    }
    
    // If zP = 0: R= Q. If zQ = 0: R = P.
    __mmask8 zmask_P = fp2_is_zero_8x1x1(P1_P2) & 0x30;
    __mmask8 zmask_Q = fp2_is_zero_8x1x1(Q1_Q2) & 0x30;
    zmask_P = (zmask_P >> 4) | (zmask_P >> 2) | zmask_P;
    zmask_Q = (zmask_Q >> 4) | (zmask_Q >> 2) | zmask_Q;
    fp2_select_8x1x1(res, res, Q1_Q2, zmask_P);
    fp2_select_8x1x1(res, res, P1_P2, zmask_Q);
}