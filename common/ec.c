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

void jac_to_xz_4x2x1(__m512i* x, __m512i* z, __m512i* jac_x, __m512i* jac_z)
{
    fp2_copy_4x2x1(x, jac_x);
    fp2_sqr_4x2x1(z, jac_z);
}