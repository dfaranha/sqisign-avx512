#include "ec.h"
#include "fp2_lvl3.h"
#include "mp.h"

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
    __m512i t0[8], t1[8], t2[8], t3[8];
    __m512i s0[16], s1[16];

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
    __m512i t0[16], t1[16], t2[16];
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
    __m512i a[8], one[8], tmp1[8], tmp2[8];

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
    __m512i tmp[8];
    fp2_copy_4x2x1(res_y, y);
    fp2_copy_4x2x1(res_z, z);

    //res_x = x - z^2 * A/3
    fp2_sqr_4x2x1(tmp, z);
    fp2_mul_4x2x1(tmp, tmp, A12_over_3);
    fp2_sub_4x2x1(res_x, x, tmp);
}

// Done (TODO delete this comment)
void ec_double_ws_4x4x2x1(
    __m512i* res_x, __m512i* res_y, __m512i* res_z, __m512i* res_t,
    __m512i* x, __m512i* y, __m512i* z, __m512i* t
)
{
    __m512i xx[8], aa[8], m[8], s[8], u[8], t0[8], t1[8];

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

// Done (TODO delete this comment)
void ec_add_components_4x4x2x1(
    __m512i* res_u, __m512i* res_v, __m512i* res_w, 
    __m512i* A, __m512i* xP, __m512i* yP, __m512i* zP, 
    __m512i* xQ, __m512i* yQ, __m512i* zQ
)
{
    __m512i zQ_zP[16], tP_tQ[16], t1_t0[16], t2_t3[16], t4_t5[16], t6_t7[16];
    // Compute the slope dy / dx for the case P != Q
    fp2_4x2x1_to_8x1x1(zQ_zP, zQ, zP);
    fp2_4x2x1_to_8x1x1(tP_tQ, xP, xQ);
    fp2_sqr_8x1x1(t1_t0, zQ_zP);
    fp2_mul_8x1x1(t2_t3, t1_t0, tP_tQ);
    fp2_4x2x1_to_8x1x1(tP_tQ, yP, yQ);
    fp2_mul_8x1x1(t4_t5, zQ_zP, tP_tQ);
    fp2_mul_8x1x1(t4_t5, t4_t5, t1_t0);

    // t0 = t0 * t1, t1 = t4 * t5:
    for(int i = 0; i < 16; i++) {
        t6_t7[i] = VUNPACKLO(t1_t0[i], t4_t5[i]); // (t1, t4)
        t1_t0[i] = VUNPACKHI(t1_t0[i], t4_t5[i]); // (t0, t5)
    }
    fp2_mul_8x1x1(t6_t7, t6_t7, t1_t0); // (t0, t1)
    fp2_8x1x1_to_4x2x1(t1_t0+8, t1_t0, t6_t7);
    fp2_add_4x2x1(res_v, t1_t0, t1_t0);

    fp2_sqr_8x1x1(t4_t5, t4_t5);

    fp2_8x1x1_to_4x2x1(t2_t3, t2_t3+8, t2_t3);
    fp2_8x1x1_to_4x2x1(t4_t5, t4_t5+8, t4_t5);
    fp2_add_4x2x1(t4_t5, t4_t5, t4_t5+8);
    fp2_add_4x2x1(t4_t5+8, t2_t3, t2_t3+8);
    fp2_add_4x2x1(t2_t3, t2_t3+8, t2_t3+8);
    fp2_sub_4x2x1(t2_t3, t4_t5+8, t2_t3);

    fp2_mulx2_4x2x1(t2_t3, t1_t0, t2_t3, t2_t3, A, t1_t0+8);
    fp2_add_4x2x1(t1_t0, t1_t0, t4_t5+8);
    // TODO: Worth it to use fp2_4x2x1_duplicate_to_8x1x1 with t2_t3?
    fp2_mulx2_4x2x1(t1_t0, res_w, t1_t0, t2_t3, t2_t3, t1_t0+8);
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
    fp2_setzero_2x4x1(P);
    P[0] = VSET(1, 0, 0, 0, 0, 0, 0, 0);
    fp2_tomont_2x4x1(P, P);
}

// TODO: Do we want to make this more efficient without a mul?
// TODO maybe setzero is redundant and there is a way to init to zero?
bool ec_curve_verif_A_1x2x4x1(__m512i* A1_A2)
{
    __m512i t[8];
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
    __m512i t0[8], t1[8], t2[8], t3[8], idx0, idx1;
    __mmask8 mask;

    // t0 = (X, X), t1 = (Z, Z)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }

    // t0 = (X + Z, X - Z)
    // t1 = t0^2
    mask = 0xf;
    fp2_mixaddsub_2x4x1(t0, t0, t1, mask);
    fp2_sqr_2x4x1(t1, t0);

    // t0 = (t1[0] + t1[1], t1[0] - t1[1])
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, t1[i]);
        t2[i] = VPERM512(idx1, t1[i]);
    }
    fp2_mixaddsub_2x4x1(t0, t0, t2, mask);

    // t2 = (A24 * t0[1], C24 * t1[1])
    idx0 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < 8; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }
    fp2_mul_2x4x1(t2, t2, A24);

    // TODO: Wasting one addition here.
    // t3 = (-, t2[0] + t2[1])
    // Maybe we can implement 1x8x1 Fp2?
    idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
    for (int i = 0; i < 8; i++) {
        t3[i] = VPERM512(idx0, t2[i]);
    }
    fp2_add_2x4x1(t3, t3, t2);
    
    // res_X = t1[0] * t2[1], res_Z = t0[1] * t3[1]
    idx0 = VSET(8, 9, 10, 11, 4, 5, 6, 7);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < 8; i++) {
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
    __m512i x[16], z[16], t0[16], t1[16], A24x8[16];
    __m512i idx0, idx1, idx2;
    __mmask8 mask;

    assert(n1 >= n2);

    // pair first components and second components in a 2x4x1
    // TODO: Define a gadget for this (observe same pattern for 2x2x2 except +4 instead of +8)
    idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < 8; i++) {
        x[i] = VBLENDPERM(idx0, B1[i], B1[i+8]);
        z[i] = VBLENDPERM(idx1, B1[i], B1[i+8]);
        x[i+8] = VBLENDPERM(idx0, B1[i+16], B1[i+16]); //wasteful
        z[i+8] = VBLENDPERM(idx1, B1[i+16], B1[i+16]);
        t0[i] = VBLENDPERM(idx0, B2[i], B2[i+8]);
        t1[i] = VBLENDPERM(idx1, B2[i], B2[i+8]);
        t0[i+8] = VBLENDPERM(idx0, B2[i+16], B2[i+16]);
        t1[i+8] = VBLENDPERM(idx1, B2[i+16], B2[i+16]);
    }
    fp2_2x4x1_to_4x2x1(x, x, x+8);
    fp2_2x4x1_to_4x2x1(x+8, t0, t0+8);
    fp2_2x4x1_to_4x2x1(z, z, z+8);
    fp2_2x4x1_to_4x2x1(z+8, t1, t1+8);
    fp2_4x2x1_to_8x1x1(x, x, x+8);
    fp2_4x2x1_to_8x1x1(z, z, z+8);

    idx0 = VSET(0, 8, 0, 8, 0, 8, 0, 8);
    idx1 = VSET(2, 10, 2, 10, 2, 10, 2, 10);
    for (int i = 0; i < 8; i++) {
        A24x8[i] = VBLENDPERM(idx0, A24_norm1[i], A24_norm2[i]);
        A24x8[i+8] = VBLENDPERM(idx1, A24_norm1[i], A24_norm2[i]);
    }

    for (int i = 0; i < n2; i++) {
        ec_double_8x1x1x1(x, z, A24x8, x, z);
    }

    fp2_8x1x1_to_4x2x1(x, x+8, x);
    fp2_8x1x1_to_4x2x1(z, z+8, z);
    fp2_8x1x1_to_first_4x2x1(A24x8, A24x8);
    for (int i = 0; i < n1 - n2; i++) {
        ec_double_4x4x2x1(x, z, A24x8, x, z);
    }

    idx0 = VSET(0, 0, 1, 1, 8, 8, 9, 9);
    idx1 = VSET(2, 2, 3, 3, 10, 10, 11, 11);
    idx2 = VSET(4, 4, 5, 5, 12, 12, 13, 13);
    mask = 0x55;
    for (int i = 0; i < 8; i++) {
        res_B1[i] = VBLENDPERMMASKZ(mask, x[i], z[i], idx0);
        res_B1[i+8] = VBLENDPERMMASKZ(mask, x[i], z[i], idx1);
        res_B1[i+16] = VBLENDPERMMASKZ(mask, x[i], z[i], idx2);
        res_B2[i] = VBLENDPERMMASKZ(mask, x[i+8], z[i+8], idx0);
        res_B2[i+8] = VBLENDPERMMASKZ(mask, x[i+8], z[i+8], idx1);
        res_B2[i+16] = VBLENDPERMMASKZ(mask, x[i+8], z[i+8], idx2);
    }
}

// TODO: Name is slightly confusing, all operations are 4x2x1
void ec_double_add_1x4x2x1(__m512i* res, __m512i* P_Q, __m512i* PmQ_A24)
{
    __m512i t0[8], t1[8], t2[8];
    __m512i idx0, idx1;
    __mmask8 mask;

    // t0 = (P.x, Q.x, P.x, Q.x)
    // t1 = (P.z, Q.z, P.z, Q.z)
    idx0 = VSET(0, 1, 4, 5, 0, 1, 4, 5);
    idx1 = VSET(2, 3, 6, 7, 2, 3, 6, 7);
    mask = 0xf;
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, P_Q[i]);
        t1[i] = VPERM512(idx1, P_Q[i]);
    }
    // t0 = (P.x + P.z, Q.x + Q.z, P.x - P.z, Q.x - Q.z)
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask);

    // t1 = (t0[0], t0[2], t0[0], t0[2])
    // t0 = (t0[0], t0[2], t0[3], t0[1])
    idx1 = VSET(0, 1, 4, 5, 6, 7, 2, 3);
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t0[i] = VPERM512(idx1, t0[i]);
    }
    fp2_mul_4x2x1(t0, t0, t1);

    // t1 = (t0[0], t0[2], t0[0], t0[2])
    // t2 = (t0[1], t0[3], t0[1], t0[3])
    idx1 = VSET(2, 3, 6, 7, 2, 3, 6, 7);
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VPERM512(idx1, t0[i]);
    }
    fp2_mixaddsub_4x2x1(t1, t1, t2, mask);

    // Moving t0[0], t1[2] to res to free them after the next mul
    idx0 = VSET(0, 1, 12, 13, 0, 0, 0, 0);
    idx1 = VSET(0, 1, 2, 3, 10, 11, 8, 9);
    for (int i = 0; i < 8; i++) {
        res[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        res[i] = VBLENDPERM(idx1, res[i], PmQ_A24[i]);
    }

    // t0 = (t0[1], t1[2], t1[1], t1[3])
    // t1 = (PmQ_A24[3], PmQ_A24[2], t1[1], t1[3])
    idx0 = VSET(2, 3, 12, 13, 10, 11, 14, 15);
    idx1 = VSET(14, 15, 12, 13, 2, 3, 6, 7);
    for (int i = 0; i < 8; i++) {
        t0[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        t1[i] = VBLENDPERM(idx1, t1[i], PmQ_A24[i]);
    }
    fp2_mul_4x2x1(t0, t0, t1);

    // t1 = (t0[0], t0[0], t0[2], t0[3])
    // t2 = (0    , t0[1], 0    , 0    )
    idx0 = VSET(0, 1, 0, 1, 4, 5, 6, 7);
    idx1 = VSET(0, 0, 2, 3, 0, 0, 0, 0);
    mask = 0xc;
    for (int i = 0; i < 8; i++) {
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
    __m512i t0[16], t1[16], t2[16];
    __m512i idx0, idx1, idx2;
    __mmask8 mask;

    // t0 = (P1.x, Q1.x, P2.x, Q2.x, P1.x, Q1.x, P2.x, Q2.x)
    //         +     +     +     +     -     -     -     -
    // t1 = (P1.z, Q1.z, P2.z, Q2.z, P1.z, Q1.z, P2.z, Q2.z)    
    idx0 = VSET(0, 4, 8, 12, 0, 4, 8, 12);
    idx1 = VSET(2, 6, 10, 14, 2, 6, 10, 14);
    mask = 0xf;
    for (int i = 0; i < 16; i++) {
        t0[i] = VBLENDPERM(idx0, P1_Q1_PQ1_A24[i], P2_Q2_PQ2_A24[i]);
        t1[i] = VBLENDPERM(idx1, P1_Q1_PQ1_A24[i], P2_Q2_PQ2_A24[i]);
    }
    fp2_mixaddsub_8x1x1(t0, t0, t1, mask, mask);

    // 8x2x1 mul with four squares hidden inside
    // t1 = (t0[0], t0[4], t0[5], t0[1], t0[2], t0[6], t0[7], t0[3])
    // t0 = (t0[0], t0[4], t0[0], t0[4], t0[2], t0[6], t0[2], t0[6])
    idx0 = VSET(0, 4, 0, 4, 2, 6, 2, 6);
    idx1 = VSET(0, 4, 5, 1, 2, 6, 7, 3);
    for (int i = 0; i < 16; i++) {
        t1[i] = VPERM512(idx1, t0[i]);
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    // t1 = (t0[0], t0[2], t0[4], t0[6], t0[0], t0[2], t0[4], t0[6])
    // t2 = (t0[1], t0[3], t0[5], t0[7], t0[1], t0[3], t0[5], t0[7])
    idx0 = VSET(0, 2, 4, 6, 0, 2, 4, 6);
    idx1 = VSET(1, 3, 5, 7, 1, 3, 5, 7);
    mask = 0xf;
    for (int i = 0; i < 16; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t2[i] = VPERM512(idx1, t0[i]);
    }
    fp2_mixaddsub_8x1x1(t1, t1, t2, mask, mask);

    // Moving values from t0 and t1 to res to free them after the next mul
    // res = (t0[0], t0[4], t1[4], t1[6], PQ1.z, PQ2.z, PQ1.x, PQ2.x)
    idx0 = VSET(0, 4, 12, 14, 0, 0, 0, 0);
    idx1 = VSET(0, 1, 2, 3, 11, 0, 9, 0);
    idx2 = VSET(0, 1, 2, 3, 4, 11, 6, 9);
    for (int i = 0; i < 16; i++) {
        R1_R2_S1_S2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
        R1_R2_S1_S2[i] = VBLENDPERM(idx1, R1_R2_S1_S2[i], P1_Q1_PQ1_A24[i]);
        R1_R2_S1_S2[i] = VBLENDPERM(idx2, R1_R2_S1_S2[i], P2_Q2_PQ2_A24[i]);
    }

    // t0 = (t0[1], t1[4], t1[1], t1[5], t0[5], t1[6], t1[3], t1[7])
    // t1 = (A24_1.z, A24_1.x, t1[1], t1[5], A24_2.z, A24_2.x, t1[3], t1[7])
    idx0 = VSET(1, 12, 9, 13, 5, 14, 11, 15);
    idx1 = VSET(15, 13, 1, 5, 0, 0, 3, 7);
    idx2 = VSET(0, 1, 2, 3, 15, 13, 6, 7);
    for (int i = 0; i < 16; i++) {
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
    for (int i = 0; i < 16; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t0[i] = VPERM512MASKZ(mask, t0[i], idx1);
    }
    fp2_add_8x1x1(t0, t0, t1);
    fp2_mul_8x1x1(R1_R2_S1_S2, R1_R2_S1_S2, t0);
}

// Requires normalized A coordinate 4* (A24 / C4) - 2
void ec_double_add_normA_1x4x2x1(__m512i* res, __m512i* P_Q, __m512i* PmQ_A)
{
    __m512i t0[8], t1[8];
    __m512i idx0, idx1;
    __mmask8 mask0, mask1;

    // First had
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    mask0 = 0x33;
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, P_Q[i]);
        t1[i] = VPERM512(idx1, P_Q[i]);
    }
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask0);

    idx0 = VSET(0, 1, 2, 3, 2, 3, 0, 1);
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mul_4x2x1(t0, t0, t1);

    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512(idx1, t0[i]);
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask0);
    fp2_sqr_4x2x1(t1, t0);

    idx0 = VSET(0, 1, 10, 11, 12, 13, 14, 15);
    idx1 = VSET(2, 3, 12, 13, 10, 11, 8, 9);
    for (int i = 0; i < 8; i++) {
        res[i] = VBLENDPERM(idx1, t0[i], PmQ_A[i]);
        t0[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }
    fp2_mul_4x2x1(t0, t0, res);

    // TODO: Figure out how to do this last part efficiently in C:
    idx0 = VSET(0, 1, 8, 9, 12, 13, 14, 15);
    idx1 = VSET(2, 3, 10, 11, 0, 0, 0, 0);
    mask0 = 0xf; mask1 = 0xc;
    for (int i = 0; i < 8; i++) {
        res[i] = VBLENDPERM(idx0, t1[i], t0[i]);
        t1[i] = VBLENDPERMMASKZ(mask0, t1[i], t0[i], idx1);
        t0[i] = VBLENDPERMMASKZ(mask1, t1[i], t0[i], idx0);
    }
    fp2_mixaddsub_4x2x1(res, res, t1, mask1);
    fp2_add_4x2x1(res, res, t0);
}

void ec_double_add_pair_normA_8x1x1(__m512i* R1_R2_S1_S2, __m512i* P1_Q1_PQ1_A, __m512i* P2_Q2_PQ2_A)
{
    __m512i t0[16], t1[16], t2[16];
    __m512i idx0, idx1, idx2;
    __mmask8 mask0, mask1;

    idx0 = VSET(0, 0, 4, 4, 8, 8, 12, 12);
    idx1 = VSET(2, 2, 6, 6, 10, 10, 14, 14);
    mask0 = 0x55;
    for (int i = 0; i < 16; i++) {
        t0[i] = VBLENDPERM(idx0, P1_Q1_PQ1_A[i], P2_Q2_PQ2_A[i]);
        t1[i] = VBLENDPERM(idx1, P1_Q1_PQ1_A[i], P2_Q2_PQ2_A[i]);
    }
    fp2_mixaddsub_8x1x1(t0, t0, t1, mask0, mask0);

    idx0 = VSET(0, 1, 1, 0, 4, 5, 5, 4);
    for (int i = 0; i < 16; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);

    idx0 = VSET(0, 0, 2, 2, 4, 4, 6, 6);
    idx1 = VSET(1, 1, 3, 3, 5, 5, 7, 7);
    for (int i = 0; i < 16; i++) {
        t1[i] = VPERM512(idx1, t0[i]);
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp2_mixaddsub_8x1x1(t0, t0, t1, mask0, mask0);
    
    fp2_sqr_8x1x1(t1, t0);

    idx0 = VSET(0, 9, 10, 11, 4, 13, 14, 15);
    idx1 = VSET(1, 13, 11, 9, 5, 0, 0, 0);
    idx2 = VSET(0, 1, 2, 3, 4, 13, 11, 9);
    for (int i = 0; i < 16; i++) {
        t2[i] = VBLENDPERM(idx1, t0[i], P1_Q1_PQ1_A[i]);
        t2[i] = VBLENDPERM(idx2, t2[i], P2_Q2_PQ2_A[i]);
        t0[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }
    fp2_mul_8x1x1(t0, t0, t2);

    // TODO: figure out how to do this efficiently in C
    idx0 = VSET(0, 4, 8, 12, 10, 14, 11, 15);
    idx1 = VSET(1, 5, 9, 13, 0, 0, 0, 0);
    mask0 = 0xf; mask1 = 0xc;
    for (int i = 0; i < 16; i++) {
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
            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }

        ec_double_add_1x4x2x1(X0_X1, X0_X1, XD_A24);

        if (bit) { // swap
            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }
    }
}

static void ec_scalar_mul_normA_inner_1x4x2x1(__m512i* X0_X1, __m512i* XD_A, const digit_t* m, const unsigned int mbits) {
    unsigned int bit;
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3);

    for (int i = mbits - 1; i >= 0; i--) {
        bit = (m[i >> LOG2RADIX] >> (i & (RADIX - 1))) & 1;

        if (bit) { // swap
            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }

        ec_double_add_normA_1x4x2x1(X0_X1, X0_X1, XD_A);

        if (bit) { // swap
            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        }
    }
}

void ec_scalar_mul_1x2x4x1(__m512i* res, __m512i* P, __m512i* A24, const digit_t* m, const unsigned int mbits)
{
    __m512i X0_X1[8], XD_A24[8];
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
    __m512i X0_X1[8], XD_A[8];
    if (mbits == 0) {
        fp2_copy_2x4x1(res, P);
        return;
    }

    // X0 = P, X1 = 0, XD = P
    ec_set_zero_1x2x4x1(X0_X1);
    fp2_2x4x1_to_4x2x1(X0_X1, X0_X1, P);
    fp2_2x4x1_to_4x2x1(XD_A, P, A);

    ec_scalar_mul_normA_inner_1x4x2x1(X0_X1, XD_A, m, mbits);

    // Ignore X1 and output X0
    fp2_4x2x1_to_2x4x1(res, XD_A, X0_X1);
}

static void ec_ladder_3pt_inner_1x4x2x1(__m512i* X0_X1, __m512i* XD_A24, const digit_t* m)
{
    __m512i X0_XD[8], X1_A24[8];
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
                for (int k = 0; k < 8; k++) {
                    X0_XD[k] = VBLENDPERM(idx_first1_first2, X0_X1[k], XD_A24[k]);
                    X1_A24[k] = VBLENDPERM(idx_second1_second2, X0_X1[k], XD_A24[k]);
                }
                
                ec_double_add_1x4x2x1(X0_XD, X0_XD, X1_A24);

                // Update X0 and XD
                for (int k = 0; k < 8; k++) {
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
    __m512i X0_XD[8], X1_A[8];
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
                for (int k = 0; k < 8; k++) {
                    X0_XD[k] = VBLENDPERM(idx_first1_first2, X0_X1[k], XD_A[k]);
                    X1_A[k] = VBLENDPERM(idx_second1_second2, X0_X1[k], XD_A[k]);
                }
                
                ec_double_add_normA_1x4x2x1(X0_XD, X0_XD, X1_A);

                // Update X0 and XD
                for (int k = 0; k < 8; k++) {
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
    __m512i X0_X1[8], XD_A24[8];

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x4x1(basis+16)) {
        return false;
    }

    // X0 = Q, X1 = P, XD = PmQ
    fp2_2x4x1_to_4x2x1(X0_X1, basis+8, basis);
    fp2_2x4x1_to_4x2x1(XD_A24, basis+16, A24);

    ec_ladder_3pt_inner_1x4x2x1(X0_X1, XD_A24, m);

    // We output X1 and ignore X0
    fp2_4x2x1_to_2x4x1(XD_A24, res, X0_X1);

    return true;
}

bool ec_ladder_3pt_normA_1x2x4x1(__m512i* res, __m512i* A, __m512i* basis, const digit_t *m)
{
    __m512i X0_X1[8], XD_A[8];

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x4x1(basis+16)) {
        return false;
    }

    // X0 = Q, X1 = P, XD = PmQ
    fp2_2x4x1_to_4x2x1(X0_X1, basis+8, basis);
    fp2_2x4x1_to_4x2x1(XD_A, basis+16, A);

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
    __m512i t0[8], t1[8], idx0, idx1;
    __mmask8 mask;

    // t0 = (P.x, P.x, Q.x, Q.x)
    // t1 = (P.z, P.z, Q.z, Q.z)
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < 8; i++) {
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
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512MASKZ(mask, t0[i], idx0);
        t0[i] = VPERM512MASKZ(mask, t0[i], idx1);
    }
    fp2_mul_2x4x1(t0, t0, t1);

    // t1 = (t0[1], t0[1])
    // t0 = (t0[0], t0[0])
    idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512(idx0, t0[i]);
        t0[i] = VPERM512(idx1, t0[i]);
    }
    mask = 0xf;
    fp2_mixaddsub_2x4x1(t0, t0, t1, mask);
    fp2_sqr_2x4x1(t0, t0);

    // t1 = (PmQ.z, PmQ.x)
    idx0 = VSET(2, 2, 3, 3, 0, 0, 1, 1);
    mask = 0x55;
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512MASKZ(mask, PmQ_A24[i], idx0);
    }
    fp2_mul_2x4x1(t0, t0, t1); // Output t0 = ((P+Q).x, (P+Q).z)

    // Updating input
    // PmQ_A24 = (Q, A24), P_Q = (P, P+Q)
    idx0 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    idx1 = VSET(0, 1, 2, 3, 8, 10, 12, 14);
    for (int i = 0; i < 8; i++) {
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
    __m512i X0_X1[8], XD_A24[8], t0[8], t1[8];
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3), idx0, idx1;
    digit_t s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER], t_arr[NWORDS_ORDER];
    digit_t* s0 = s0_arr, *s1 = s1_arr, *t = t_arr;

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x4x1(basis+16)) {
        return false;
    }

    fp2_2x4x1_to_4x2x1(X0_X1, basis, basis+8);
    fp2_2x4x1_to_4x2x1(XD_A24, basis+16, A24);
    mp_copy(s0, scalarP, NWORDS_ORDER);
    mp_copy(s1, scalarQ, NWORDS_ORDER);

    while (!mp_is_zero(s0, NWORDS_ORDER)) {
        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t* tmp = s0;
            s0 = s1;
            s1 = tmp;
            
            for (int k = 0; k < 8; k++) {
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

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_diff_add_1x2x4x1(X0_X1, XD_A24);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s0_even == s1_even) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);
            mp_shiftr(s1, 1, NWORDS_ORDER);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_double_add_1x4x2x1(X0_X1, X0_X1, XD_A24);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s1_even) {
            mp_shiftr(s1, 1, NWORDS_ORDER);

            idx0 = VSET(4, 5, 6, 7, 8, 9, 10, 11); // second first
            idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            for (int k = 0; k < 8; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A24[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A24[k]);
            }
            ec_double_add_1x4x2x1(t0, t0, t1);

            // Update X1 and XD
            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VBLENDPERM(idx0, X0_X1[k], t0[k]);
                XD_A24[k] = VBLENDPERM(idx1, t0[k], XD_A24[k]);
            }
        } else {
            mp_shiftr(s0, 1, NWORDS_ORDER);

            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A24[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A24[k]);
            }
            ec_double_add_1x4x2x1(t0, t0, t1);

            // Update X0 and XD
            idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
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
    __m512i X0_X1[8], XD_A[8], t0[8], t1[8];
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3), idx0, idx1;
    digit_t s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER], t_arr[NWORDS_ORDER];
    digit_t* s0 = s0_arr, *s1 = s1_arr, *t = t_arr;

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x4x1(basis+16)) {
        return false;
    }

    fp2_2x4x1_to_4x2x1(X0_X1, basis, basis+8);
    fp2_2x4x1_to_4x2x1(XD_A, basis+16, A);
    mp_copy(s0, scalarP, NWORDS_ORDER);
    mp_copy(s1, scalarQ, NWORDS_ORDER);

    while (!mp_is_zero(s0, NWORDS_ORDER)) {
        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t* tmp = s0;
            s0 = s1;
            s1 = tmp;
            
            for (int k = 0; k < 8; k++) {
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

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_diff_add_1x2x4x1(X0_X1, XD_A);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s0_even == s1_even) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);
            mp_shiftr(s1, 1, NWORDS_ORDER);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_double_add_normA_1x4x2x1(X0_X1, X0_X1, XD_A);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s1_even) {
            mp_shiftr(s1, 1, NWORDS_ORDER);

            idx0 = VSET(4, 5, 6, 7, 8, 9, 10, 11); // second first
            idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            for (int k = 0; k < 8; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A[k]);
            }
            ec_double_add_normA_1x4x2x1(t0, t0, t1);

            // Update X1 and XD
            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VBLENDPERM(idx0, X0_X1[k], t0[k]);
                XD_A[k] = VBLENDPERM(idx1, t0[k], XD_A[k]);
            }
        } else {
            mp_shiftr(s0, 1, NWORDS_ORDER);

            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A[k]);
            }
            ec_double_add_normA_1x4x2x1(t0, t0, t1);

            // Update X0 and XD
            idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
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
    __m512i R1_R2_S1_S2[16], idx;
    __m512i *P1_Q1_PQ1_A24 = X0_X1_XD_A24+16, *P2_Q2_PQ2_A24 = Y0_Y1_YD_A24+16;
    digit_t *r0 = *r0_ptr, *r1 = *r1_ptr, *s0 = *s0_ptr, *s1 = *s1_ptr;
    int case_r, case_s;

    while (true) {
        if (mp_compare(r0, r1, NWORDS_ORDER) > 0) { // r0 > r1
            digit_t *tmp = r0;
            r0 = r1;
            r1 = tmp;

            // swapping X0 and X1
            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 16; i++) {
                X0_X1_XD_A24[i] = VPERM512(idx, X0_X1_XD_A24[i]);
            }
        }

        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t *tmp = s0;
            s0 = s1;
            s1 = tmp;

            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 16; i++) {
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
            for (int i = 0; i < 16; i++) {
                P1_Q1_PQ1_A24[i] = VPERM512(idx, X0_X1_XD_A24[i]);
            }
        } else if (r1_even) {
            case_r = 1;
            mp_shiftr(r1, 1, NWORDS_ORDER);

            // P1 = X1, Q1 = XD, PQ1 = X0
            idx = VSET(4, 0, 6, 2, 1, 5, 3, 7);
            for (int i = 0; i < 16; i++) {
                P1_Q1_PQ1_A24[i] = VPERM512(idx, X0_X1_XD_A24[i]);
            }
        } else {
            case_r = 2;
            mp_shiftr(r0, 1, NWORDS_ORDER);

            // P1 = X0, Q1 = XD, PQ1 = X1
            idx = VSET(0, 4, 2, 6, 1, 5, 3, 7);
            for (int i = 0; i < 16; i++) {
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
            for (int i = 0; i < 16; i++) {
                P2_Q2_PQ2_A24[i] = VPERM512(idx, Y0_Y1_YD_A24[i]);
            }
        } else if(s1_even) {
            case_s = 1;
            mp_shiftr(s1, 1, NWORDS_ORDER);

            // P2 = Y1, Q2 = YD, PQ2 = Y0
            idx = VSET(4, 0, 6, 2, 1, 5, 3, 7);
            for (int i = 0; i < 16; i++) {
                P2_Q2_PQ2_A24[i] = VPERM512(idx, Y0_Y1_YD_A24[i]);
            }
        } else {
            case_s = 2;
            mp_shiftr(s0, 1, NWORDS_ORDER);

            // P2 = Y0, Q2 = YD, PQ2 = Y1
            idx = VSET(0, 4, 2, 6, 1, 5, 3, 7);
            for (int i = 0; i < 16; i++) {
                P2_Q2_PQ2_A24[i] = VPERM512(idx, Y0_Y1_YD_A24[i]);
            }
        }

        ec_double_add_pair_8x1x1(R1_R2_S1_S2, P1_Q1_PQ1_A24, P2_Q2_PQ2_A24);

        if (case_r == 0) {
            // X0 = R2, X1 = R1
            idx = VSET(4, 9, 6, 11, 0, 13, 2, 15);
            for (int i = 0; i < 16; i++) {
                X0_X1_XD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A24[i]);
            }
        } else if (case_r == 1) {
            // X1 = R1, XD = R2
            idx = VSET(8, 4, 10, 6, 0, 13, 2, 15);
            for (int i = 0; i < 16; i++) {
                X0_X1_XD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A24[i]);
            }
        } else if (case_r == 2) {
            // X0 = R1, XD = R2
            idx = VSET(0, 4, 2, 6, 12, 13, 14, 15);
            for (int i = 0; i < 16; i++) {
                X0_X1_XD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A24[i]);
            }
        }

        if (case_s == 0) {
            // Y0 = S2, Y1 = S1
            idx = VSET(5, 9, 7, 11, 1, 13, 3, 15);
            for (int i = 0; i < 16; i++) {
                Y0_Y1_YD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], Y0_Y1_YD_A24[i]);
            }
        } else if (case_s == 1) {
            // Y1 = S1, YD = S2
            idx = VSET(8, 5, 10, 7, 1, 13, 3, 15);
            for (int i = 0; i < 16; i++) {
                Y0_Y1_YD_A24[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], Y0_Y1_YD_A24[i]);
            }
        } else if (case_s == 2) {
            // Y0 = S1, YD = S2
            idx = VSET(1, 5, 3, 7, 12, 13, 14, 15);
            for (int i = 0; i < 16; i++) {
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
    __m512i R1_R2_S1_S2[16], idx;
    __m512i *P1_Q1_PQ1_A = X0_X1_XD_A+16, *P2_Q2_PQ2_A = Y0_Y1_YD_A+16;
    digit_t *r0 = *r0_ptr, *r1 = *r1_ptr, *s0 = *s0_ptr, *s1 = *s1_ptr;
    int case_r, case_s;

    while (true) {
        if (mp_compare(r0, r1, NWORDS_ORDER) > 0) { // r0 > r1
            digit_t *tmp = r0;
            r0 = r1;
            r1 = tmp;

            // swapping X0 and X1
            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 16; i++) {
                X0_X1_XD_A[i] = VPERM512(idx, X0_X1_XD_A[i]);
            }
        }

        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t *tmp = s0;
            s0 = s1;
            s1 = tmp;

            idx = VSET(4, 1, 6, 3, 0, 5, 2, 7);
            for (int i = 0; i < 16; i++) {
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
            for (int i = 0; i < 16; i++) {
                P1_Q1_PQ1_A[i] = VPERM512(idx, X0_X1_XD_A[i]);
            }
        } else if (r1_even) {
            case_r = 1;
            mp_shiftr(r1, 1, NWORDS_ORDER);

            // P1 = X1, Q1 = XD, PQ1 = X0
            idx = VSET(4, 0, 6, 2, 1, 5, 3, 7);
            for (int i = 0; i < 16; i++) {
                P1_Q1_PQ1_A[i] = VPERM512(idx, X0_X1_XD_A[i]);
            }
        } else {
            case_r = 2;
            mp_shiftr(r0, 1, NWORDS_ORDER);

            // P1 = X0, Q1 = XD, PQ1 = X1
            idx = VSET(0, 4, 2, 6, 1, 5, 3, 7);
            for (int i = 0; i < 16; i++) {
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
            for (int i = 0; i < 16; i++) {
                P2_Q2_PQ2_A[i] = VPERM512(idx, Y0_Y1_YD_A[i]);
            }
        } else if(s1_even) {
            case_s = 1;
            mp_shiftr(s1, 1, NWORDS_ORDER);

            // P2 = Y1, Q2 = YD, PQ2 = Y0
            idx = VSET(4, 0, 6, 2, 1, 5, 3, 7);
            for (int i = 0; i < 16; i++) {
                P2_Q2_PQ2_A[i] = VPERM512(idx, Y0_Y1_YD_A[i]);
            }
        } else {
            case_s = 2;
            mp_shiftr(s0, 1, NWORDS_ORDER);

            // P2 = Y0, Q2 = YD, PQ2 = Y1
            idx = VSET(0, 4, 2, 6, 1, 5, 3, 7);
            for (int i = 0; i < 16; i++) {
                P2_Q2_PQ2_A[i] = VPERM512(idx, Y0_Y1_YD_A[i]);
            }
        }

        ec_double_add_pair_normA_8x1x1(R1_R2_S1_S2, P1_Q1_PQ1_A, P2_Q2_PQ2_A);

        if (case_r == 0) {
            // X0 = R2, X1 = R1
            idx = VSET(4, 9, 6, 11, 0, 13, 2, 15);
            for (int i = 0; i < 16; i++) {
                X0_X1_XD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A[i]);
            }
        } else if (case_r == 1) {
            // X1 = R1, XD = R2
            idx = VSET(8, 4, 10, 6, 0, 13, 2, 15);
            for (int i = 0; i < 16; i++) {
                X0_X1_XD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A[i]);
            }
        } else if (case_r == 2) {
            // X0 = R1, XD = R2
            idx = VSET(0, 4, 2, 6, 12, 13, 14, 15);
            for (int i = 0; i < 16; i++) {
                X0_X1_XD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], X0_X1_XD_A[i]);
            }
        }

        if (case_s == 0) {
            // Y0 = S2, Y1 = S1
            idx = VSET(5, 9, 7, 11, 1, 13, 3, 15);
            for (int i = 0; i < 16; i++) {
                Y0_Y1_YD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], Y0_Y1_YD_A[i]);
            }
        } else if (case_s == 1) {
            // Y1 = S1, YD = S2
            idx = VSET(8, 5, 10, 7, 1, 13, 3, 15);
            for (int i = 0; i < 16; i++) {
                Y0_Y1_YD_A[i] = VBLENDPERM(idx, R1_R2_S1_S2[i], Y0_Y1_YD_A[i]);
            }
        } else if (case_s == 2) {
            // Y0 = S1, YD = S2
            idx = VSET(1, 5, 3, 7, 12, 13, 14, 15);
            for (int i = 0; i < 16; i++) {
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
    __m512i X0_X1_XD_A24[32], Y0_Y1_YD_A24[32]; // Need 32 consecutive vectors for single biscalars at the end

    // 8x1x1 layout = (X0.x, XD.x, X0.z, XD.z, X1.x, A24.x, X1.z, A24.z)
    fp2_2x4x1_to_4x2x1(X0_X1_XD_A24, basis, basis+8);
    fp2_2x4x1_to_4x2x1(X0_X1_XD_A24+8, basis+16, A24);
    fp2_4x2x1_to_8x1x1(X0_X1_XD_A24, X0_X1_XD_A24, X0_X1_XD_A24+8);
    fp2_copy_8x1x1(Y0_Y1_YD_A24, X0_X1_XD_A24);

    mp_copy(r0, (*mat)[0][0], NWORDS_ORDER);
    mp_copy(r1, (*mat)[1][0], NWORDS_ORDER);
    mp_copy(s0, (*mat)[0][1], NWORDS_ORDER);
    mp_copy(s1, (*mat)[1][1], NWORDS_ORDER);

    ec_biscalar_mul_pair_inner_8x1x1(X0_X1_XD_A24, Y0_Y1_YD_A24, &r0, &r1, &s0, &s1);

    fp2_8x1x1_to_4x2x1(X0_X1_XD_A24, X0_X1_XD_A24+16, X0_X1_XD_A24);
    fp2_4x2x1_to_2x4x1(X0_X1_XD_A24, X0_X1_XD_A24+8, X0_X1_XD_A24);
    fp2_4x2x1_to_2x4x1(X0_X1_XD_A24+16, X0_X1_XD_A24+24, X0_X1_XD_A24+16);
    fp2_8x1x1_to_4x2x1(Y0_Y1_YD_A24, Y0_Y1_YD_A24+16, Y0_Y1_YD_A24);
    fp2_4x2x1_to_2x4x1(Y0_Y1_YD_A24, Y0_Y1_YD_A24+8, Y0_Y1_YD_A24);
    fp2_4x2x1_to_2x4x1(Y0_Y1_YD_A24+16, Y0_Y1_YD_A24+24, Y0_Y1_YD_A24+16);

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
    __m512i X0_X1_XD_A[32], Y0_Y1_YD_A[32]; // Need 32 consecutive vectors for single biscalars at the end

    // 8x1x1 layout = (X0.x, XD.x, X0.z, XD.z, X1.x, A.x, X1.z, -)
    fp2_2x4x1_to_4x2x1(X0_X1_XD_A, basis, basis+8);
    fp2_2x4x1_to_4x2x1(X0_X1_XD_A+8, basis+16, A);
    fp2_4x2x1_to_8x1x1(X0_X1_XD_A, X0_X1_XD_A, X0_X1_XD_A+8);
    fp2_copy_8x1x1(Y0_Y1_YD_A, X0_X1_XD_A);

    mp_copy(r0, (*mat)[0][0], NWORDS_ORDER);
    mp_copy(r1, (*mat)[1][0], NWORDS_ORDER);
    mp_copy(s0, (*mat)[0][1], NWORDS_ORDER);
    mp_copy(s1, (*mat)[1][1], NWORDS_ORDER);

    ec_biscalar_mul_pair_normA_inner_8x1x1(X0_X1_XD_A, Y0_Y1_YD_A, &r0, &r1, &s0, &s1);

    fp2_8x1x1_to_4x2x1(X0_X1_XD_A, X0_X1_XD_A+16, X0_X1_XD_A);
    fp2_4x2x1_to_2x4x1(X0_X1_XD_A, X0_X1_XD_A+8, X0_X1_XD_A);
    fp2_4x2x1_to_2x4x1(X0_X1_XD_A+16, X0_X1_XD_A+24, X0_X1_XD_A+16);
    fp2_8x1x1_to_4x2x1(Y0_Y1_YD_A, Y0_Y1_YD_A+16, Y0_Y1_YD_A);
    fp2_4x2x1_to_2x4x1(Y0_Y1_YD_A, Y0_Y1_YD_A+8, Y0_Y1_YD_A);
    fp2_4x2x1_to_2x4x1(Y0_Y1_YD_A+16, Y0_Y1_YD_A+24, Y0_Y1_YD_A+16);

    bool succ1 = ec_biscalar_mul_normA_1x2x4x1(res1, r0, r1, A, A24, X0_X1_XD_A);
    bool succ2 = ec_biscalar_mul_normA_1x2x4x1(res2, s0, s1, A, A24, Y0_Y1_YD_A);

    return succ1 && succ2;
}

// Debugging functions

bool test_point_order_twof_1x2x4x1(__m512i* P, __m512i* A24, unsigned int f)
{
    __m512i Q[8];
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
        test_point_order_twof_1x2x4x1(B+8, A24, f) &
        test_point_order_twof_1x2x4x1(B+16, A24, f);
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
    fp2_setzero_2x2x2(P);
    P[0] = VSET(1, 0, 0, 0, 0, 0, 0, 0);
    fp2_tomont_2x2x2(P, P);
}

// Does not assume A24 has been normalized.
void ec_double_1x2x2x2(__m512i* res, __m512i* P, __m512i* A24)
{
    __m512i t0[4], t1[4], t2[4], t3[4], idx0, idx1;
    __mmask8 mask;

    // t0 = (X, X), t1 = (Z, Z)
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < 4; i++) {
        t0[i] = VPERM512(idx0, P[i]);
        t1[i] = VPERM512(idx1, P[i]);
    }

    // t0 = (X + Z, X - Z)
    // t1 = t0^2
    mask = 0xf;
    fp2_mixaddsub_2x2x2(t0, t0, t1, mask);
    fp2_sqr_2x2x2(t1, t0);

    // t0 = (t1[0] + t1[1], t1[0] - t1[1])
    for (int i = 0; i < 4; i++) {
        t0[i] = VPERM512(idx0, t1[i]);
        t2[i] = VPERM512(idx1, t1[i]);
    }
    fp2_mixaddsub_2x2x2(t0, t0, t2, mask);

    // t2 = (A24 * t0[1], C24 * t1[1])
    idx0 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < 4; i++) {
        t2[i] = VBLENDPERM(idx0, t0[i], t1[i]);
    }
    fp2_mul_2x2x2(t2, t2, A24);

    // TODO: Wasting one addition here.
    // t3 = (-, t2[0] + t2[1])
    // Maybe we can implement 1x8x1 Fp2?
    idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
    for (int i = 0; i < 4; i++) {
        t3[i] = VPERM512(idx0, t2[i]);
    }
    fp2_add_2x2x2(t3, t3, t2);
    
    // res_X = t1[0] * t2[1], res_Z = t0[1] * t3[1]
    idx0 = VSET(8, 9, 10, 11, 4, 5, 6, 7);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < 4; i++) {
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
    __m512i x[16], z[16], t0[16], t1[16], A24x8[16];
    __m512i idx0, idx1, idx2, idx3;

    assert(n1 >= n2);

    idx0 = VSET(0, 8, 0, 0, 0, 0, 0, 0);
    idx1 = VSET(1, 9, 0, 0, 0, 0, 0, 0);
    idx2 = VSET(2, 10, 0, 0, 0, 0, 0, 0);
    idx3 = VSET(3, 11, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < 4; i++) {
        x[i] = VBLENDPERM(idx0, B1[i], B2[i]);
        x[i+4] = VBLENDPERM(idx1, B1[i], B2[i]);
        x[i+8] = VBLENDPERM(idx2, B1[i], B2[i]);
        x[i+12] = VBLENDPERM(idx3, B1[i], B2[i]);

        t0[i] = VBLENDPERM(idx0, B1[i+4], B2[i+4]);
        t0[i+4] = VBLENDPERM(idx1, B1[i+4], B2[i+4]);
        t0[i+8] = VBLENDPERM(idx2, B1[i+4], B2[i+4]);
        t0[i+12] = VBLENDPERM(idx3, B1[i+4], B2[i+4]);

        t1[i] = VBLENDPERM(idx0, B1[i+8], B2[i+8]);
        t1[i+4] = VBLENDPERM(idx1, B1[i+8], B2[i+8]);
        t1[i+8] = VBLENDPERM(idx2, B1[i+8], B2[i+8]);
        t1[i+12] = VBLENDPERM(idx3, B1[i+8], B2[i+8]);
    }
    idx0 = VSET(0, 1, 8, 9, 0, 0, 0, 0);
    idx1 = VSET(0, 1, 2, 3, 8, 9, 0, 0);
    for (int i = 0; i < 16; i++) {
        x[i] = VBLENDPERM(idx0, x[i], t0[i]);
        x[i] = VBLENDPERM(idx1, x[i], t1[i]);
    }
    idx0 = VSET(4, 12, 0, 0, 0, 0, 0, 0);
    idx1 = VSET(5, 13, 0, 0, 0, 0, 0, 0);
    idx2 = VSET(6, 14, 0, 0, 0, 0, 0, 0);
    idx3 = VSET(7, 15, 0, 0, 0, 0, 0, 0);
    for (int i = 0; i < 4; i++) {
        z[i] = VBLENDPERM(idx0, B1[i], B2[i]);
        z[i+4] = VBLENDPERM(idx1, B1[i], B2[i]);
        z[i+8] = VBLENDPERM(idx2, B1[i], B2[i]);
        z[i+12] = VBLENDPERM(idx3, B1[i], B2[i]);

        t0[i] = VBLENDPERM(idx0, B1[i+4], B2[i+4]);
        t0[i+4] = VBLENDPERM(idx1, B1[i+4], B2[i+4]);
        t0[i+8] = VBLENDPERM(idx2, B1[i+4], B2[i+4]);
        t0[i+12] = VBLENDPERM(idx3, B1[i+4], B2[i+4]);

        t1[i] = VBLENDPERM(idx0, B1[i+8], B2[i+8]);
        t1[i+4] = VBLENDPERM(idx1, B1[i+8], B2[i+8]);
        t1[i+8] = VBLENDPERM(idx2, B1[i+8], B2[i+8]);
        t1[i+12] = VBLENDPERM(idx3, B1[i+8], B2[i+8]);
    }
    idx0 = VSET(0, 1, 8, 9, 0, 0, 0, 0);
    idx1 = VSET(0, 1, 2, 3, 8, 9, 0, 0);
    for (int i = 0; i < 16; i++) {
        z[i] = VBLENDPERM(idx0, z[i], t0[i]);
        z[i] = VBLENDPERM(idx1, z[i], t1[i]);
    }
    idx0 = VSET(0, 8, 0, 8, 0, 8, 0, 8);
    idx1 = VSET(1, 9, 1, 9, 1, 9, 1, 9);
    idx2 = VSET(2, 10, 2, 10, 2, 10, 2, 10);
    idx3 = VSET(3, 11, 3, 11, 3, 11, 3, 11);
    for (int i = 0; i < 4; i++) {
        A24x8[i] = VBLENDPERM(idx0, A24_norm1[i], A24_norm2[i]);
        A24x8[i+4] = VBLENDPERM(idx1, A24_norm1[i], A24_norm2[i]);
        A24x8[i+8] = VBLENDPERM(idx2, A24_norm1[i], A24_norm2[i]);
        A24x8[i+12] = VBLENDPERM(idx3, A24_norm1[i], A24_norm2[i]);
    }

    for (int i = 0; i < n2; i++) {
        ec_double_8x1x1x1(x, z, A24x8, x, z);
    }

    fp2_8x1x1_to_4x2x1(x, x+8, x);
    fp2_8x1x1_to_4x2x1(z, z+8, z);
    fp2_8x1x1_to_first_4x2x1(A24x8, A24x8);
    for (int i = 0; i < n1 - n2; i++) {
        ec_double_4x4x2x1(x, z, A24x8, x, z);
    }

    fp2_4x2x1_to_2x2x2(x, x+4, x);
    fp2_4x2x1_to_2x2x2(x+8, x+12, x+8);
    fp2_4x2x1_to_2x2x2(z, z+4, z);
    fp2_4x2x1_to_2x2x2(z+8, z+12, z+8);

    // Coupling first and second components
    idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for(int i = 0; i < 4; i++) {
        res_B1[i] = VBLENDPERM(idx0, x[i], z[i]);
        res_B1[i+4] = VBLENDPERM(idx1, x[i], z[i]);
        res_B1[i+8] = VBLENDPERM(idx0, x[i+4], z[i+4]);
        res_B2[i] = VBLENDPERM(idx0, x[i+8], z[i+8]);
        res_B2[i+4] = VBLENDPERM(idx1, x[i+8], z[i+8]);
        res_B2[i+8] = VBLENDPERM(idx0, x[i+12], z[i+12]);
    }
}

void ec_scalar_mul_1x2x2x2(__m512i* res, __m512i* P, __m512i* A24, const digit_t* m, const unsigned int mbits)
{
    __m512i X0_X1[8], XD_A24[8];
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
    __m512i X0_X1[8], XD_A[8];
    if (mbits == 0) {
        fp2_copy_2x2x2(res, P);
        return;
    }

    // X0 = P, X1 = 0, XD = P
    ec_set_zero_1x2x2x2(X0_X1);
    fp2_2x2x2_to_4x2x1(X0_X1, X0_X1, P);
    fp2_2x2x2_to_4x2x1(XD_A, P, A);

    ec_scalar_mul_normA_inner_1x4x2x1(X0_X1, XD_A, m, mbits);

    // Ignore X1 and output X0
    fp2_4x2x1_to_2x2x2(res, XD_A, X0_X1);
}

bool ec_ladder_3pt_1x2x2x2(__m512i* res, __m512i* A24, __m512i* basis, const digit_t* m)
{
    __m512i X0_X1[8], XD_A24[8];

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x2x2(basis+8)) {
        return false;
    }

    // X0 = Q, X1 = P, XD = PmQ
    fp2_2x2x2_to_4x2x1(X0_X1, basis+4, basis);
    fp2_2x2x2_to_4x2x1(XD_A24, basis+8, A24);

    ec_ladder_3pt_inner_1x4x2x1(X0_X1, XD_A24, m);

    // We output X1 and ignore X0
    fp2_4x2x1_to_2x2x2(XD_A24, res, X0_X1);

    return true;
}

bool ec_ladder_3pt_normA_1x2x2x2(__m512i* res, __m512i* A, __m512i* basis, const digit_t *m)
{
    __m512i X0_X1[8], XD_A[8];

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x2x2(basis+8)) {
        return false;
    }

    // X0 = Q, X1 = P, XD = PmQ
    fp2_2x2x2_to_4x2x1(X0_X1, basis+4, basis);
    fp2_2x2x2_to_4x2x1(XD_A, basis+8, A);

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
    __m512i t0[8], t1[8];
    __m512i u0[4], u1[4];
    __m512i idx0, idx1;
    __mmask8 mask;

    // t0 = (P.x, P.x, Q.x, Q.x),  
    // t1 = (P.z, P.z, Q.z, Q.z)
    idx0 = VSET(0, 1, 0, 1, 4, 5, 4, 5);
    idx1 = VSET(2, 3, 2, 3, 6, 7, 6, 7);
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx0, P_Q[i]);
        t1[i] = VPERM512(idx1, P_Q[i]);
    }
    // t0 = (P.x + P.z, P.x - P.z, Q.x + Q.z, Q.x - Q.z)
    mask = 0x33;
    fp2_mixaddsub_4x2x1(t0, t0, t1, mask);

    // t1 = (P.x+P.z, P.x-P.z)
    // t1+4 = (s3, s2) = (Q.x-Q.z, Q.x+Q.z)
    idx0 = VSET(8, 0, 9, 1, 10, 2, 11, 3);
    idx1 = VSET(14, 6, 15, 7, 12, 4, 13, 5);
    for (int j = 0; j < 4; j++) {
        t1[j] = VBLENDPERM(idx0, t0[j+4], t0[j]);
        t0[j] = VBLENDPERM(idx1, t0[j+4], t0[j]);
    }
    fp2_mul_2x2x2(t0, t0, t1);

    // t0 = (t0[0], t0[0]),  t1 = (t0[1], t0[1])
    idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int j = 0; j < 4; j++) {
        t1[j] = VPERM512(idx1, t0[j]);
        t0[j] = VPERM512(idx0, t0[j]);
    }

    fp2_mixaddsub_2x2x2(t0, t0, t1, 0x0f);
    fp2_sqr_2x2x2(t0, t0);

    // Extract (PmQ.z, PmQ.x) from PmQ_A24 into t1
    idx0 = VSET(2, 10, 3, 11, 0, 8, 1, 9);
    for (int j = 0; j < 4; j++) {
        t1[j] = VBLENDPERM(idx0, PmQ_A24[j], PmQ_A24[j+4]);
    }
    fp2_mul_2x2x2(t0, t0, t1); // P + Q

    // Updating input
    // PmQ_A24 = (Q, A24), P_Q = (P, P+Q)
    idx0 = VSET(4, 5, 6, 7, 12, 13, 14, 15);
    for (int i = 0; i < 8; i++) {
        PmQ_A24[i] = VBLENDPERM(idx0, P_Q[i], PmQ_A24[i]);
    }

    idx0 = VSET(0, 1, 2, 3, 8, 10, 12, 14);
    idx1 = VSET(0, 1, 2, 3, 9, 11, 13, 15);
    for (int j = 0; j < 4; j++) {
        P_Q[j] = VBLENDPERM(idx0, P_Q[j],   t0[j]);
        P_Q[j+4] = VBLENDPERM(idx1, P_Q[j+4], t0[j]);
    }
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
    __m512i X0_X1[8], XD_A24[8], t0[8], t1[8];
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3), idx0, idx1;
    digit_t s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER], t_arr[NWORDS_ORDER];
    digit_t* s0 = s0_arr, *s1 = s1_arr, *t = t_arr;

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x2x2(basis+8)) {
        return false;
    }

    fp2_2x2x2_to_4x2x1(X0_X1, basis, basis+4);
    fp2_2x4x1_to_4x2x1(XD_A24, basis+8, A24);
    mp_copy(s0, scalarP, NWORDS_ORDER);
    mp_copy(s1, scalarQ, NWORDS_ORDER);

    while (!mp_is_zero(s0, NWORDS_ORDER)) {
        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t* tmp = s0;
            s0 = s1;
            s1 = tmp;
            
            for (int k = 0; k < 8; k++) {
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

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_diff_add_1x2x2x2(X0_X1, XD_A24);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s0_even == s1_even) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);
            mp_shiftr(s1, 1, NWORDS_ORDER);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_double_add_1x4x2x1(X0_X1, X0_X1, XD_A24);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s1_even) {
            mp_shiftr(s1, 1, NWORDS_ORDER);

            idx0 = VSET(4, 5, 6, 7, 8, 9, 10, 11); // second first
            idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            for (int k = 0; k < 8; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A24[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A24[k]);
            }
            ec_double_add_1x4x2x1(t0, t0, t1);

            // Update X1 and XD
            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VBLENDPERM(idx0, X0_X1[k], t0[k]);
                XD_A24[k] = VBLENDPERM(idx1, t0[k], XD_A24[k]);
            }
        } else {
            mp_shiftr(s0, 1, NWORDS_ORDER);

            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A24[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A24[k]);
            }
            ec_double_add_1x4x2x1(t0, t0, t1);

            // Update X0 and XD
            idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
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
    __m512i X0_X1[8], XD_A[8], t0[8], t1[8];
    __m512i idx_swap = VSET(4, 5, 6, 7, 0, 1, 2, 3), idx0, idx1;
    digit_t s0_arr[NWORDS_ORDER], s1_arr[NWORDS_ORDER], t_arr[NWORDS_ORDER];
    digit_t* s0 = s0_arr, *s1 = s1_arr, *t = t_arr;

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x2x2(basis+8)) {
        return false;
    }

    fp2_2x2x2_to_4x2x1(X0_X1, basis, basis+4);
    fp2_2x2x2_to_4x2x1(XD_A, basis+8, A);
    mp_copy(s0, scalarP, NWORDS_ORDER);
    mp_copy(s1, scalarQ, NWORDS_ORDER);

    while (!mp_is_zero(s0, NWORDS_ORDER)) {
        if (mp_compare(s0, s1, NWORDS_ORDER) > 0) { // s0 > s1
            digit_t* tmp = s0;
            s0 = s1;
            s1 = tmp;
            
            for (int k = 0; k < 8; k++) {
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

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_diff_add_1x2x2x2(X0_X1, XD_A);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s0_even == s1_even) {
            mp_sub(s1, s1, s0, NWORDS_ORDER);
            mp_shiftr(s1, 1, NWORDS_ORDER);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }

            ec_double_add_normA_1x4x2x1(X0_X1, X0_X1, XD_A);

            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VPERM512(idx_swap, X0_X1[k]);
            }
        } else if (s1_even) {
            mp_shiftr(s1, 1, NWORDS_ORDER);

            idx0 = VSET(4, 5, 6, 7, 8, 9, 10, 11); // second first
            idx1 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            for (int k = 0; k < 8; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A[k]);
            }
            ec_double_add_normA_1x4x2x1(t0, t0, t1);

            // Update X1 and XD
            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
                X0_X1[k] = VBLENDPERM(idx0, X0_X1[k], t0[k]);
                XD_A[k] = VBLENDPERM(idx1, t0[k], XD_A[k]);
            }
        } else {
            mp_shiftr(s0, 1, NWORDS_ORDER);

            idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11); // first first
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
                t0[k] = VBLENDPERM(idx0, X0_X1[k], XD_A[k]);
                t1[k] = VBLENDPERM(idx1, X0_X1[k], XD_A[k]);
            }
            ec_double_add_normA_1x4x2x1(t0, t0, t1);

            // Update X0 and XD
            idx0 = VSET(0, 1, 2, 3, 12, 13, 14, 15); // first second
            idx1 = VSET(4, 5, 6, 7, 12, 13, 14, 15); // second second
            for (int k = 0; k < 8; k++) {
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
    __m512i X0_X1_XD_A24[32], Y0_Y1_YD_A24[32]; // Need 32 consecutive vectors for single biscalars at the end

    // 8x1x1 layout = (X0.x, XD.x, X0.z, XD.z, X1.x, A24.x, X1.z, A24.z)
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A24, basis, basis+4);
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A24+8, basis+8, A24);
    fp2_4x2x1_to_8x1x1(X0_X1_XD_A24, X0_X1_XD_A24, X0_X1_XD_A24+8);
    fp2_copy_8x1x1(Y0_Y1_YD_A24, X0_X1_XD_A24);

    mp_copy(r0, (*mat)[0][0], NWORDS_ORDER);
    mp_copy(r1, (*mat)[1][0], NWORDS_ORDER);
    mp_copy(s0, (*mat)[0][1], NWORDS_ORDER);
    mp_copy(s1, (*mat)[1][1], NWORDS_ORDER);

    ec_biscalar_mul_pair_inner_8x1x1(X0_X1_XD_A24, Y0_Y1_YD_A24, &r0, &r1, &s0, &s1);

    fp2_8x1x1_to_4x2x1(X0_X1_XD_A24, X0_X1_XD_A24+8, X0_X1_XD_A24);
    fp2_4x2x1_to_2x2x2(X0_X1_XD_A24, X0_X1_XD_A24+4, X0_X1_XD_A24);
    fp2_4x2x1_to_2x2x2(X0_X1_XD_A24+8, X0_X1_XD_A24+12, X0_X1_XD_A24+8);
    fp2_8x1x1_to_4x2x1(Y0_Y1_YD_A24, Y0_Y1_YD_A24+8, Y0_Y1_YD_A24);
    fp2_4x2x1_to_2x2x2(Y0_Y1_YD_A24, Y0_Y1_YD_A24+4, Y0_Y1_YD_A24);
    fp2_4x2x1_to_2x2x2(Y0_Y1_YD_A24+8, Y0_Y1_YD_A24+12, Y0_Y1_YD_A24+8);

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
    __m512i X0_X1_XD_A[32], Y0_Y1_YD_A[32]; // Need 32 consecutive vectors for single biscalars at the end

    // 8x1x1 layout = (X0.x, XD.x, X0.z, XD.z, X1.x, A.x, X1.z, -)
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A, basis, basis+4);
    fp2_2x2x2_to_4x2x1(X0_X1_XD_A+8, basis+8, A);
    fp2_4x2x1_to_8x1x1(X0_X1_XD_A, X0_X1_XD_A, X0_X1_XD_A+8);
    fp2_copy_8x1x1(Y0_Y1_YD_A, X0_X1_XD_A);

    mp_copy(r0, (*mat)[0][0], NWORDS_ORDER);
    mp_copy(r1, (*mat)[1][0], NWORDS_ORDER);
    mp_copy(s0, (*mat)[0][1], NWORDS_ORDER);
    mp_copy(s1, (*mat)[1][1], NWORDS_ORDER);

    ec_biscalar_mul_pair_normA_inner_8x1x1(X0_X1_XD_A, Y0_Y1_YD_A, &r0, &r1, &s0, &s1);

    fp2_8x1x1_to_4x2x1(X0_X1_XD_A, X0_X1_XD_A+8, X0_X1_XD_A);
    fp2_4x2x1_to_2x2x2(X0_X1_XD_A, X0_X1_XD_A+4, X0_X1_XD_A);
    fp2_4x2x1_to_2x2x2(X0_X1_XD_A+8, X0_X1_XD_A+12, X0_X1_XD_A+8);
    fp2_8x1x1_to_4x2x1(Y0_Y1_YD_A, Y0_Y1_YD_A+8, Y0_Y1_YD_A);
    fp2_4x2x1_to_2x2x2(Y0_Y1_YD_A, Y0_Y1_YD_A+4, Y0_Y1_YD_A);
    fp2_4x2x1_to_2x2x2(Y0_Y1_YD_A+8, Y0_Y1_YD_A+12, Y0_Y1_YD_A+8);

    bool succ1 = ec_biscalar_mul_normA_1x2x2x2(res1, r0, r1, A, A24, X0_X1_XD_A);
    bool succ2 = ec_biscalar_mul_normA_1x2x2x2(res2, s0, s1, A, A24, Y0_Y1_YD_A);

    return succ1 && succ2;
}

// Debugging functions

bool test_point_order_twof_1x2x2x2(__m512i* P, __m512i* A24, unsigned int f)
{
    __m512i Q[4];
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
        test_point_order_twof_1x2x2x2(B+4, A24, f) &
        test_point_order_twof_1x2x2x2(B+8, A24, f);
}

void difference_point_normA_1x2x2x2(ec_point_t *PQ, const ec_point_t *P, const ec_point_t *Q, const ec_curve_t *curve)
{
    // Given P,Q in projective x-only, computes a deterministic choice for (P-Q)
    // Based on Proposition 3 of https://eprint.iacr.org/2017/518.pdf

    fp2_t Bxx, Bxz, Bzz, t0, t1;

    fp2_mul(&t0, &P->x, &Q->x);
    fp2_mul(&t1, &P->z, &Q->z);
    fp2_sub(&Bxx, &t0, &t1);
    fp2_sqr(&Bxx, &Bxx);
    fp2_mul(&Bxx, &Bxx, &curve->C); // C*(P.x*Q.x-P.z*Q.z)^2
    fp2_add(&Bxz, &t0, &t1);
    fp2_mul(&t0, &P->x, &Q->z);
    fp2_mul(&t1, &P->z, &Q->x);
    fp2_add(&Bzz, &t0, &t1);
    fp2_mul(&Bxz, &Bxz, &Bzz); // (P.x*Q.x+P.z*Q.z)(P.x*Q.z+P.z*Q.x)
    fp2_sub(&Bzz, &t0, &t1);
    fp2_sqr(&Bzz, &Bzz);
    fp2_mul(&Bzz, &Bzz, &curve->C); // C*(P.x*Q.z-P.z*Q.x)^2
    fp2_mul(&Bxz, &Bxz, &curve->C); // C*(P.x*Q.x+P.z*Q.z)(P.x*Q.z+P.z*Q.x)
    fp2_mul(&t0, &t0, &t1);
    fp2_mul(&t0, &t0, &curve->A);
    fp2_add(&t0, &t0, &t0);
    fp2_add(&Bxz, &Bxz, &t0); // C*(P.x*Q.x+P.z*Q.z)(P.x*Q.z+P.z*Q.x) + 2*A*P.x*Q.z*P.z*Q.x

    // To ensure that the denominator is a fourth power in Fp, we normalize by
    // C*C_bar^2*(P.z)_bar^2*(Q.z)_bar^2
    fp_copy(&t0.re, &curve->C.re);
    fp_neg(&t0.im, &curve->C.im);
    fp2_sqr(&t0, &t0);
    fp2_mul(&t0, &t0, &curve->C);
    fp_copy(&t1.re, &P->z.re);
    fp_neg(&t1.im, &P->z.im);
    fp2_sqr(&t1, &t1);
    fp2_mul(&t0, &t0, &t1);
    fp_copy(&t1.re, &Q->z.re);
    fp_neg(&t1.im, &Q->z.im);
    fp2_sqr(&t1, &t1);
    fp2_mul(&t0, &t0, &t1);
    fp2_mul(&Bxx, &Bxx, &t0);
    fp2_mul(&Bxz, &Bxz, &t0);
    fp2_mul(&Bzz, &Bzz, &t0);

    // Solving quadratic equation
    fp2_sqr(&t0, &Bxz);
    fp2_mul(&t1, &Bxx, &Bzz);
    fp2_sub(&t0, &t0, &t1);
    // No need to check if t0 is square, as per the entangled basis algorithm.
    fp2_sqrt(&t0);
    fp2_add(&PQ->x, &Bxz, &t0);
    fp2_copy(&PQ->z, &Bzz);
}