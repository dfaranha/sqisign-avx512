#include "ec.h"
#include "fp2_lvl3.h"

void ec_point_unvectorized_to_4x2x1(__m512i* res_x, __m512i* res_z, ec_point_t* src)
{
    fp2_t unvec_x[4], unvec_z[4];
    for (int i = 0; i < 4; i++) {
        fp2_copy(unvec_x+i, &((src+i)->x));
        fp2_copy(unvec_z+i, &((src+i)->z));
    }
    fp2_unvectorized_to_4x2x1(res_x, unvec_x);
    fp2_unvectorized_to_4x2x1(res_z, unvec_z);
}


void ec_point_4x2x1_to_unvectorized(ec_point_t* res, __m512i* x, __m512i* z)
{
    fp2_t unvec_x[4], unvec_z[4];
    fp2_4x2x1_to_unvectorized(unvec_x, x);
    fp2_4x2x1_to_unvectorized(unvec_z, z);
    for (int i = 0; i < 4; i++) {
        fp2_copy(&((res+i)->x), unvec_x+i);
        fp2_copy(&((res+i)->z), unvec_z+i);
    }
}

// Done (TODO delete this comment) 1582 -> 1530 cycles.
void ec_double_4x2x1(__m512i* res_x, __m512i* res_z, __m512i* A24, __m512i* x, __m512i* z)
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

// It could be that this is just as fast with the optimized field arithmetic.
void ec_double_4x2x1_old(__m512i* res_x, __m512i* res_z, __m512i* A24, __m512i* x, __m512i* z)
{
    __m512i t0[8], t1[8], t2[8];
    
    fp2_add_4x2x1(t0, x, z);
    fp2_sqr_4x2x1(t0, t0);
    fp2_sub_4x2x1(t1, x, z);
    fp2_sqr_4x2x1(t1, t1);
    fp2_sub_4x2x1(t2, t0, t1);
    fp2_mul_4x2x1(res_x, t0, t1);
    fp2_mul_4x2x1(t0, t2, A24);
    fp2_add_4x2x1(t0, t0, t1);
    fp2_mul_4x2x1(res_z, t0, t2);
}

// TODO: in SQIsign jac_to_xz has more code to handle the point at infty
void ec_jac_to_xz_4x2x1(__m512i* x, __m512i* z, __m512i* jac_x, __m512i* jac_z)
{
    fp2_copy_4x2x1(x, jac_x);
    fp2_sqr_4x2x1(z, jac_z);
}

void ec_jac_to_ws_4x2x1(
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

void ec_ws_to_jac_4x2x1(
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

void ec_double_ws_4x2x1(
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
void ec_double_ws_iter_4x2x1(
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
    
    ec_double_ws_4x2x1(res_x, res_y, res_z, res_t, x, y, z, t);
    for (int i = 0; i < n-1; i++) {
        ec_double_ws_4x2x1(res_x, res_y, res_z, res_t, res_x, res_y, res_z, res_t);
    }
}

void ec_add_components_4x2x1_old(
    __m512i* res_u, __m512i* res_v, __m512i* res_w, 
    __m512i* A, __m512i* xP, __m512i* yP, __m512i* zP, 
    __m512i* xQ, __m512i* yQ, __m512i* zQ
)
{
    __m512i t0[8], t1[8], t2[8], t3[8], t4[8], t5[8], t6[8];

    // Compute the slope dy / dx for the case P != Q
    fp2_sqrx2_4x2x1(t0, t1, zP, zQ);
    fp2_mulx2_4x2x1(t2, t3, t1, xP, t0, xQ);
    fp2_mulx2_4x2x1(t4, t5, yP, zQ, zP, yQ);
    fp2_mulx2_4x2x1(t4, t5, t4, t1, t5, t0);
    fp2_mulx2_4x2x1(t0, t6, t0, t1, t4, t5);
    fp2_add_4x2x1(res_v, t6, t6);
    fp2_sqrx2_4x2x1(t4, t5, t4, t5);
    fp2_add_4x2x1(t4, t4, t5);
    fp2_add_4x2x1(t5, t2, t3);
    fp2_add_4x2x1(t6, t3, t3);
    fp2_sub_4x2x1(t6, t5, t6);
    fp2_mulx2_4x2x1(t6, t1, t6, t6, A, t0);
    fp2_add_4x2x1(t1, t1, t5);
    fp2_mulx2_4x2x1(t1, res_w, t1, t6, t6, t0);
    fp2_sub_4x2x1(res_u, t4, t1);
}

void ec_add_components_4x2x1(
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