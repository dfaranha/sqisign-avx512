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

void ec_double_4x2x1(__m512i* res_x, __m512i* res_z, __m512i* A24, __m512i* x, __m512i* z)
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
    fp2_mul_4x2x1(a, A12, res_A12_over_3);
    fp2_sub_4x2x1(a, one, a);

    // res_x = x + z^2 * A/3
    fp2_sqr_4x2x1(tmp1, z);
    fp2_mul_4x2x1(tmp2, tmp1, res_A12_over_3);
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

    fp2_sqr_4x2x1(xx, x);
    fp2_sqr_4x2x1(t0, y);
    fp2_add_4x2x1(aa, t0, t0);
    fp2_add_4x2x1(t0, x, aa);
    fp2_sqr_4x2x1(aa, aa);
    fp2_add_4x2x1(u, aa, aa);
    fp2_sqr_4x2x1(t0, t0);
    fp2_sub_4x2x1(t0, t0, xx);
    fp2_sub_4x2x1(s, t0, aa);
    fp2_add_4x2x1(t0, xx, xx);
    fp2_add_4x2x1(t0, t0, xx);
    fp2_add_4x2x1(m, t0, t);
    fp2_sqr_4x2x1(t0, m);
    fp2_add_4x2x1(t1, s, s);
    fp2_sub_4x2x1(res_x, t0, t1);
    fp2_sub_4x2x1(t0, s, res_x);
    fp2_mul_4x2x1(t0, t0, m);
    fp2_sub_4x2x1(res_y, t0, u);
    fp2_mul_4x2x1(t0, y, z);
    fp2_add_4x2x1(res_z, t0, t0);
    fp2_mul_4x2x1(t0, u, t);
    fp2_add_4x2x1(res_t, t0, t0);
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

void ec_add_components_4x2x1(
    __m512i* res_u, __m512i* res_v, __m512i* res_w, 
    __m512i* A, __m512i* xP, __m512i* yP, __m512i* zP, 
    __m512i* xQ, __m512i* yQ, __m512i* zQ
)
{
    __m512i t0[8], t1[8], t2[8], t3[8], t4[8], t5[8], t6[8];

    // Compute the slope dy / dx for the case P != Q
    fp2_sqr_4x2x1(t0, zP);
    fp2_sqr_4x2x1(t1, zQ);
    fp2_mul_4x2x1(t2, t1, xP);
    fp2_mul_4x2x1(t3, t0, xQ);
    fp2_mul_4x2x1(t4, yP, zQ);
    fp2_mul_4x2x1(t4, t4, t1);
    fp2_mul_4x2x1(t5, zP, yQ);
    fp2_mul_4x2x1(t5, t5, t0);
    fp2_mul_4x2x1(t0, t0, t1);
    fp2_mul_4x2x1(t6, t4, t5);
    fp2_add_4x2x1(res_v, t6, t6);
    fp2_sqr_4x2x1(t4, t4);
    fp2_sqr_4x2x1(t5, t5);
    fp2_add_4x2x1(t4, t4, t5);
    fp2_add_4x2x1(t5, t2, t3);
    fp2_add_4x2x1(t6, t3, t3);
    fp2_sub_4x2x1(t6, t5, t6);
    fp2_sqr_4x2x1(t6, t6);
    fp2_mul_4x2x1(t1, A, t0);
    fp2_add_4x2x1(t1, t1, t5);
    fp2_mul_4x2x1(t1, t1, t6);
    fp2_sub_4x2x1(res_u, t4, t1);
    fp2_mul_4x2x1(res_w, t6, t0);
}