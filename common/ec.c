#include "ec.h"
#include "fp2_lvl3.h"
#include "sqi_fp_constants_lvl3.h"

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

// TODO: Might be that 2x2x2 is faster?
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

// TODO: Name is slightly confusing, all operations are 4x2x1
void ec_double_add_1x2x4x1(__m512i* res, __m512i* P_Q, __m512i* PmQ_A24)
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

bool ec_ladder_3pt_1x2x4x1(__m512i* res, __m512i* A24, __m512i* basis, const digit_t *m)
{
    __m512i X0_X1[8], XD_A24[8], X0_XD[8], X1_A24[8];
    __m512i idx_first1_second2 = VSET(0, 1, 2, 3, 12, 13, 14, 15);
    __m512i idx_first1_first2 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    __m512i idx_second1_second2 = VSET(4, 5, 6, 7, 12, 13, 14, 15);

    // Formulas are not valid if PmQ.x = 0 or PmQ.z = 0
    if (fp2_is_zero_2x4x1(basis+16)) {
        return false;
    }

    // X0 = Q, X1 = P, XD = PmQ
    fp2_2x4x1_to_4x2x1(X0_X1, basis+8, basis);
    fp2_2x4x1_to_4x2x1(XD_A24, basis+16, A24);

    digit_t t;
    for (int i = 0; i < NWORDS_ORDER; i++) {
        t = 1;
        for (int j = 0; j < 64; j++) {
            uint64_t bit_set = (t & m[i]);
            if (bit_set) {
                ec_double_add_1x2x4x1(X0_X1, X0_X1, XD_A24);
            } else {
                // Swap X1 and XD
                for (int k = 0; k < 8; k++) {
                    X0_XD[k] = VBLENDPERM(idx_first1_first2, X0_X1[k], XD_A24[k]);
                    X1_A24[k] = VBLENDPERM(idx_second1_second2, X0_X1[k], XD_A24[k]);
                }
                
                ec_double_add_1x2x4x1(X0_XD, X0_XD, X1_A24);

                // Update X0 and XD
                for (int k = 0; k < 8; k++) {
                    X0_X1[k] = VBLENDPERM(idx_first1_second2, X0_XD[k], X0_X1[k]);
                    XD_A24[k] = VBLENDPERM(idx_second1_second2, X0_XD[k], XD_A24[k]);
                }
            }
            t <<= 1;
        }
    }

    // We output X1 and ignore X0
    fp2_4x2x1_to_2x4x1(XD_A24, res, X0_X1);

    return true;
}