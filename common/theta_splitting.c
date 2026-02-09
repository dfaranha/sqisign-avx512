#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "theta.h"
#include "theta_splitting.h"
#include "utils.h"

#include "sqi_fp2.h"

#include <string.h>

int splitting_get_index(__m512i* null)
{
    __m512i tmp1[8], tmp2[8], tmp3[8], tmp4[8], idx1, idx2, idx3, idx4;
    __mmask8 add_mask;
    int is_zero, index = 0, num_found = 0, t;

    // === (0, 0), (1, 0), (2, 0), (3, 0) === 
    fp2_sqr_4x2x1(tmp1, null);
    
    idx1 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    idx2 = VSET(2, 3, 2, 3, 2, 3, 2, 3);
    idx3 = VSET(4, 5, 4, 5, 4, 5, 4, 5);
    idx4 = VSET(6, 7, 6, 7, 6, 7, 6, 7);
    for (int i = 0; i < 8; i++) {
        tmp2[i] = VPERM512(idx2, tmp1[i]);
        tmp3[i] = VPERM512(idx3, tmp1[i]);
        tmp4[i] = VPERM512(idx4, tmp1[i]);
        tmp1[i] = VPERM512(idx1, tmp1[i]);
    }
    add_mask = 0x33; // +, -, +, -
    fp2_mixaddsub_4x2x1(tmp1, tmp1, tmp2, add_mask);
    add_mask = 0x0f; // +, +, -, -
    fp2_mixaddsub_4x2x1(tmp1, tmp1, tmp3, add_mask);
    add_mask = 0xc3; // +, -, -, +
    fp2_mixaddsub_4x2x1(tmp1, tmp1, tmp4, add_mask);
    
    is_zero = fp2_is_zero_4x2x1(tmp1);
    t = is_zero & 0x1; // (0,0 ) has index 0
    num_found += t;
    t = (is_zero >> 1) & 0x1;
    num_found += t;
    index = index*(1-t) + 4*t; // (1, 0) has index 4
    t = (is_zero >> 2) & 0x1;
    num_found += t;
    index = index*(1-t) + 6*t; // (2, 0) has index 6
    t = (is_zero >> 3) & 0x1;
    num_found += t;
    index = index*(1-t) + 8*t; // (3, 0) has index 8

    // === (0, 1), (0, 2), (2, 1), (1, 2) === 
    idx1 = VSET(0, 1, 0, 1, 2, 3, 4, 5);
    idx2 = VSET(2, 3, 4, 5, 6, 7, 6, 7);
    for (int i = 0; i < 8; i++) {
        tmp1[i] = VPERM512(idx1, null[i]);
        tmp2[i] = VPERM512(idx2, null[i]);
    }
    fp2_mul_4x2x1(tmp1, tmp1, tmp2);
    
    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx2 = VSET(6, 7, 4, 5, 6, 7, 5, 5);
    for (int i = 0; i < 8; i++) {
        tmp2[i] = VPERM512(idx2, tmp1[i]);
        tmp1[i] = VPERM512(idx1, tmp1[i]);
    }
    add_mask = 0x0f;
    fp2_mixaddsub_4x2x1(tmp1, tmp1, tmp2, add_mask);
    is_zero = fp2_is_zero_4x2x1(tmp1);
    t = is_zero & 0x1;
    num_found += t;
    index = index*(1-t) + t; // (0, 1) has index 1
    t = (is_zero >> 1) & 0x1;
    num_found += t;
    index = index*(1-t) + 2*t; // (0, 2) has index 2
    t = (is_zero >> 2) & 0x1;
    num_found += t;
    index = index*(1-t) + 7*t; // (2, 1) has index 7
    t = (is_zero >> 3) & 0x1;
    num_found += t;
    index = index*(1-t) + 5*t; // (1, 2) has index 5

    // === (0, 3), (3, 3) ===
    // preparing 2x4x1 directly
    idx1 = VSET(0, 0, 1, 1, 2, 2, 3, 3);
    idx2 = VSET(6, 6, 7, 7, 4, 4, 5, 5); // if this was not swapped we could use standard conv.
    for (int i = 0; i < 8; i++) {
        tmp1[i] = VPERM512MASKZ(0x55, null[i], idx1);
        tmp2[i] = VPERM512MASKZ(0x55, null[i], idx2);
    }

    fp2_mul_2x4x1(tmp1, tmp1, tmp2);
    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx2 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < 8; i++) {
        tmp2[i] = VPERM512MASKZ(0x55, tmp1[i], idx2);
        tmp1[i] = VPERM512MASKZ(0x55, tmp1[i], idx1);
    }
    add_mask = 0x0f;
    fp2_mixaddsub_2x4x1(tmp1, tmp1, tmp2, add_mask);
    is_zero = fp2_is_zero_2x4x1(tmp1);
    t = is_zero & 0x1;
    num_found += t;
    index = index*(1-t) + 3*t;
    t = (is_zero >> 1) & 0x1;
    num_found += t;
    index = index*(1-t) + 9*t;

    if (num_found != 1) {
        return -1;
    } 
    return index;
} 

void splitting_isom_matrix(splitting_matrix_t* matrix, __m512i* null)
{
    int index = splitting_get_index(null);
    *matrix = SPLITTING_MATRICES[index];
}

void splitting_rand_matrix(splitting_matrix_t* matrix)
{
    // sampling uniformly at random from [0,5]
    int index = rand_in_range(0, 5, 3); 
    *matrix = RAND_MATRICES[index];
}

// res = (point[2*index], point[2*index+1], point[2*index], point[2*index+1], ...)
void splitting_duplicate_pair_in_array(__m512i* res, __m512i* point, const int index, const int arraylen) {
    __m512i indices = VSET(2*index, 2*index+1 , 2*index, 2*index+1, 2*index, 2*index+1, 2*index, 2*index+1);
    for (int i = 0; i < arraylen; i++) {
        res[i] = VPERM512(indices, point[i]);
    }
}

void splitting_mul_matrix_column_single(__m512i* res, __m512i* dup, const splitting_matrix_t* matrix, const int index)
{
    __m512i tmp[8];
    
    // Negate re when matrix entry is -1 or -i. Negate im when -1 or i.
    __mmask8 negate_mask = matrix->is_negative[index] ^ (0xaa & matrix->is_imaginary[index]);
    fp2_neg_4x2x1(tmp, dup);
    
    for (int i = 0; i < 8; i++) {
        res[i] = VBLEND(negate_mask, dup[i], tmp[i]);

        // conditional swap of re and im if multiplying by imaginary
        tmp[i] = VSHUFFLE(res[i], 0x4e);
        res[i] = VBLEND(matrix->is_imaginary[index], res[i], tmp[i]);

        // conditionally set to 0
        res[i] = VBLEND(matrix->is_zero[index], res[i], VZERO());
    }

}

// Let vec(a_1), ..., vec(a_4) be the columns of the matrix A.
// A * (x, y, z, t)^t = x * vec(a_1) + y * vec(a_2) + z * vec(a_3) + t * vec(a_4)
// To compute x*vec(a_1), we create a vector (x, x, x, x)^t, 
// then do coordinate-wise mult with vec(a_1).
void splitting_apply_isomorphism_single(__m512i* res, __m512i* point, const splitting_matrix_t* matrix)
{
    __m512i vec[8], point_copy[8], indices;
    __mmask8 negate_mask;
    fp2_copy_4x2x1(point_copy, point); // in case res = point

    splitting_duplicate_pair_in_array(res, point_copy, 0, 8);
    splitting_mul_matrix_column_single(res, res, matrix, 0);

    for (int j = 1; j < 4; j++) {
        splitting_duplicate_pair_in_array(vec, point_copy, j, 8);
        splitting_mul_matrix_column_single(vec, vec, matrix, j);
        fp2_add_4x2x1(res, res, vec);
    }
}

void splitting_apply_isomorphism_list_single(__m512i* pts, const splitting_matrix_t* matrix, const int num_pts)
{
    for (int i = 0; i < num_pts; i++) {
        splitting_apply_isomorphism_single(pts + 8*i, pts + 8*i, matrix);
    }
}

// dup = (x, x', x, x', x, x', x, x') in the 8x1x1 format.
// Multiplies the index-th column of the matrix A by x and x'.
// res = (a_(1, i) * x, a_(1, i) * x', ...)
void splitting_mul_matrix_column_pair(__m512i* res, __m512i* dup, const splitting_matrix_t* matrix, const int index)
{
    __m512i tmp[8];
    __mmask8 negate_mask;
    
    // Negate re when matrix entry is -1 or -i.
    negate_mask = matrix->is_negative[index];
    fp_neg_8x1(tmp, dup);
    for (int i = 0; i < 8; i++) {
        res[i] = VBLEND(negate_mask, dup[i], tmp[i]);

        // conditionally set to 0
        res[i] = VBLEND(matrix->is_zero[index], res[i], VZERO());
    }

    // Negate im when -1 or i.
    negate_mask = matrix->is_negative[index] ^ matrix->is_imaginary[index];
    fp_neg_8x1(tmp, dup+8);
    for (int i = 0; i < 8; i++) {
        res[i+8] = VBLEND(negate_mask, dup[i+8], tmp[i]);

        // conditionally set to 0
        res[i+8] = VBLEND(matrix->is_zero[index], res[i+8], VZERO());

        // conditionally swap re and im if multiplying by imaginary
        tmp[i] = res[i];
        res[i] = VBLEND(matrix->is_imaginary[index], res[i], res[i+8]);
        res[i+8] = VBLEND(matrix->is_imaginary[index], res[i+8], tmp[i]);
    }
}

// Multiplies the points by the matrix A.
// The points are stored in the 8x1x1 format as (x, x', y, y', z, z', t, t')
// To multply x and x' by the first column of A, we create a vector (x, x', x, x', x, x', x, x')
void splitting_apply_isomorphism_pair(__m512i* res, __m512i* points, const splitting_matrix_t* matrix)
{
    __m512i vec[16], points_copy[16], indices;
    __mmask8 negate_mask;
    fp2_copy_8x1x1(points_copy, points); // in case res = point

    splitting_duplicate_pair_in_array(res, points_copy, 0, 16);
    splitting_mul_matrix_column_pair(res, res, matrix, 0);

    for (int j = 1; j < 4; j++) {
        splitting_duplicate_pair_in_array(vec, points_copy, j, 16);
        splitting_mul_matrix_column_pair(vec, vec, matrix, j);
        fp2_add_8x1x1(res, res, vec);
    }
}

void splitting_apply_isomorphism_list_pair(__m512i* pts, const splitting_matrix_t* matrix, const int num_pts)
{
    bool num_pts_is_odd = (num_pts & 0x1);
    int num_pair_pts = num_pts / 2;

    for (int i = 0; i < num_pair_pts; i++) {
        splitting_apply_isomorphism_pair(pts + 16*i, pts + 16*i, matrix);
    }
    if (num_pts_is_odd) {
        splitting_apply_isomorphism_single(
            pts+8*(num_pts-1), pts+8*(num_pts-1), matrix
        );
    }
}

void splitting_theta_to_product_old(__m512i* product, __m512i* null) 
{    
    __m512i tmp[8], idx1, idx2;
    __mmask8 mask = 0x0f;
    
    // x = a^4, y = b^4, z = c^4
    // TODO: Wastes one squaring, can we make use of it? I think the answer is no...
    fp2_sqr_4x2x1(tmp, null);
    fp2_sqr_4x2x1(tmp, tmp);

    // (x, x, x, x)
    idx1 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    // (z, y, z, y)
    idx2 = VSET(4, 5, 2, 3, 4, 5, 2, 3);
    for (int i = 0; i < 8; i++) {
        product[i] = VPERM512(idx1, tmp[i]);
        tmp[i] = VPERM512(idx2, tmp[i]);
    }
    
    // (x+z, x+y, x-z, x-y)
    fp2_mixaddsub_4x2x1(product, product, tmp, mask);

    // (-2*(x+y), -2*(x+y), -, -)
    fp2_add_4x2x1(tmp, product, product);
    fp2_neg_4x2x1(tmp, tmp);
    
    // (-2*(x+y), -2*(x+y), x-z, x-y)
    for (int i = 0; i < 8; i++) {
        product[i] = VBLEND(mask, product[i], tmp[i]);
    }
}

void splitting_theta_to_product(theta_couple_curve_t* res, __m512i* null) 
{    
    __m512i t0[8], t1[8], idx0, idx1;
    fp2_t unvec[4];
    __mmask8 mask = 0x0f;
    
    // x = a^4, y = b^4, z = c^4
    // TODO: Wastes one squaring, can we make use of it? I think the answer is no...
    fp2_sqr_4x2x1(t0, null);
    fp2_sqr_4x2x1(t0, t0);

    // (z, y, z, y)
    idx0 = VSET(4, 5, 2, 3, 4, 5, 2, 3);
    // (x, x, x, x)
    idx1 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    for (int i = 0; i < 8; i++) {
        t1[i] = VPERM512(idx1, t0[i]);
        t0[i] = VPERM512(idx0, t0[i]);
    }
    
    // (x+z, x+y, x-z, x-y)
    fp2_mixaddsub_4x2x1(t1, t1, t0, mask);

    // (-2*(x+z), -2*(x+y), -, -)
    fp2_add_4x2x1(t0, t1, t1);
    fp2_neg_4x2x1(t0, t0);
    
    // (-2*(x+z), -2*(x+y), x-z, x-y)
    for (int i = 0; i < 8; i++) {
        t0[i] = VBLEND(mask, t1[i], t0[i]);
    }
    
    // Finally, we convert to the unvectorized 64-bit repr.
    fp2_4x2x1_to_unvectorized(unvec, t0);
    ec_curve_init(&res->E1);
    ec_curve_init(&res->E2);
    res->E1.A = unvec[0];
    res->E2.A = unvec[1];
    res->E1.C = unvec[2];
    res->E2.C = unvec[3];
}

void splitting_theta_point_to_mont_single_old(__m512i* mont_points, __m512i* theta_point, __m512i* null)
{
    __m512i tmp[8], idx1, idx2;
    __mmask8 add_mask;
    
    idx1 = VSET(4, 5, 2, 3, 0, 1, 0, 1); // (c, b, a, a)
    idx2 = VSET(0, 1, 0, 1, 4, 5, 2, 3); // (x, x, z, y)
    for (int i = 0; i < 8; i++) {
        tmp[i] = VPERM512(idx1, null[i]);
        mont_points[i] = VPERM512(idx2, theta_point[i]);
    }
    fp2_mul_4x2x1(mont_points, mont_points, tmp); // (c*x, b*x, a*z, a*y)

    idx1 = VSET(4, 5, 6, 7, 0, 1, 2, 3);    // (a*z, a*y, c*x, b*x)
    for (int i = 0; i < 8; i++) {
        tmp[i] = VPERM512(idx1, mont_points[i]);
    }

    // (a*z + c*x, a*y + b*x, a*x - c*x, a*y - b*x)
    add_mask = 0x0f;
    fp2_mixaddsub_4x2x1(mont_points, mont_points, tmp, add_mask);
}

void splitting_theta_point_to_mont_list_single_old(__m512i* pts, __m512i* null, const int num_pts)
{
    for (int i = 0; i < num_pts; i++) {
        splitting_theta_point_to_mont_single_old(pts + 8*i, pts + 8*i, null);
    }
}

void splitting_theta_point_to_mont_pair_old(__m512i* mont_points, __m512i* theta_points, __m512i* null)
{
    __m512i tmp[16], idx;
    __mmask8 add_mask;

    idx = VSET(4, 5, 2, 3, 0, 1, 0, 1); // (c, c, b, a)
    for (int i = 0; i < 8; i++) {
        tmp[i] = VPERM512(idx, null[i]);
    }
    fp2_4x2x1_to_8x1x1(tmp, tmp, tmp); // (c, c, c, c, b, b, a, a)
    
    idx = VSET(0, 1, 0, 1, 4, 5, 2, 3); // (x, x', x, x' z, z', y, y')
    for (int i = 0; i < 16; i++) {
        mont_points[i] = VPERM512(idx, theta_points[i]);
    }

    // (c*x, c*x', b*x, b*x', a*z, a*z', a*y, a*y')
    fp2_mul_8x1x1(mont_points, mont_points, tmp); 

    // (a*z, a*z', a*y, a*y', c*x, c*x', b*x, b*x')
    idx = VSET(4, 5, 6, 7, 0, 1, 2, 3);    
    for (int i = 0; i < 16; i++) {
        tmp[i] = VPERM512(idx, mont_points[i]);
    }

    // (a*z + c*x, a*z' + c*x', a*y + b*x, a*y' + b*x', 
    //  a*x - c*x, a*x' - c*x', a*y - b*x, a*y' - b*x')
    add_mask = 0x0f;
    fp2_mixaddsub_8x1x1(mont_points, mont_points, tmp, add_mask, add_mask);
}

void splitting_theta_point_to_mont_list_pair_old(__m512i* pts, __m512i* null, const int num_pts)
{
    bool num_pts_is_odd = (num_pts & 0x1);
    int num_pair_pts = num_pts / 2;

    for (int i = 0; i < num_pair_pts; i++) {
        splitting_theta_point_to_mont_pair_old(pts + 16*i, pts + 16*i, null);
    }
    if (num_pts_is_odd) {
        splitting_theta_point_to_mont_single_old(
            pts+8*(num_pts-1), pts+8*(num_pts-1), null
        );
    }
}

// TODO: The python script has additional cases not handled here
void splitting_theta_point_to_mont_single(
    theta_couple_point_t* mont_pair, __m512i* theta_point, __m512i* null
)
{
    fp2_t OA_unvec[4], P_unvec[4], x1, x2, z1, z2, tmp;
    fp2_4x2x1_to_unvectorized(OA_unvec, null);
    fp2_4x2x1_to_unvectorized(P_unvec, theta_point);

    if (fp2_is_zero(P_unvec) && fp2_is_zero(P_unvec+2)) {
        fp2_mul(&x1, OA_unvec, P_unvec+3);
        fp2_copy(&z1, &x1);
        fp2_mul(&tmp, OA_unvec+2, P_unvec+1);
        fp2_add(&x1, &x1, &tmp);
        fp2_sub(&z1, &z1, &tmp);
    } else {
        fp2_mul(&x1, OA_unvec, P_unvec+2);
        fp2_copy(&z1, &x1);
        fp2_mul(&tmp, OA_unvec+2, P_unvec);
        fp2_add(&x1, &x1, &tmp);
        fp2_sub(&z1, &z1, &tmp);
    }

    if (fp2_is_zero(P_unvec) && fp2_is_zero(P_unvec+1)) {
        fp2_mul(&x2, OA_unvec, P_unvec+3);
        fp2_copy(&z2, &x2);
        fp2_mul(&tmp, OA_unvec+1, P_unvec+2);
        fp2_add(&x2, &x2, &tmp);
        fp2_sub(&z2, &z2, &tmp);
    } else {
        fp2_mul(&x2, OA_unvec, P_unvec+1);
        fp2_copy(&z2, &x2);
        fp2_mul(&tmp, OA_unvec+1, P_unvec);
        fp2_add(&x2, &x2, &tmp);
        fp2_sub(&z2, &z2, &tmp);
    }
    mont_pair->P1.x = x1;
    mont_pair->P2.x = x2;
    mont_pair->P1.z = z1;
    mont_pair->P2.z = z2;
    // __m512i t0[8], t1[8], idx0, idx1;
    // fp2_t unvec[4];
    // __mmask8 add_mask;
    
    // idx0 = VSET(4, 5, 2, 3, 0, 1, 0, 1); // (c, b, a, a)
    // idx1 = VSET(0, 1, 0, 1, 4, 5, 2, 3); // (x, x, z, y)
    // for (int i = 0; i < 8; i++) {
    //     t0[i] = VPERM512(idx0, null[i]);
    //     t1[i] = VPERM512(idx1, theta_point[i]);
    // }
    // fp2_mul_4x2x1(t1, t1, t0); // (c*x, b*x, a*z, a*y)

    // idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);    // (a*z, a*y, c*x, b*x)
    // for (int i = 0; i < 8; i++) {
    //     t0[i] = VPERM512(idx0, t1[i]);
    // }

    // // (a*z + c*x, a*y + b*x, a*x - c*x, a*y - b*x)
    // add_mask = 0x0f;
    // fp2_mixaddsub_4x2x1(t1, t1, t0, add_mask);

    // // Convert to unvectorized 64-bit repr.
    // fp2_4x2x1_to_unvectorized(unvec, t1);
    // mont_pair->P1.x = unvec[0];
    // mont_pair->P2.x = unvec[1];
    // mont_pair->P1.z = unvec[2];
    // mont_pair->P2.z = unvec[3];
}

void splitting_theta_point_to_mont_list_single(
    theta_couple_point_t* mont_pts, __m512i* theta_pts, 
    __m512i* null, const int num_pts
)
{
    for (int i = 0; i < num_pts; i++) {
        splitting_theta_point_to_mont_single(mont_pts + i, theta_pts + 8*i, null);
    }
}

void splitting_theta_point_to_mont_pair(
    theta_couple_point_t* mont_pair1, theta_couple_point_t* mont_pair2, 
    __m512i* theta_pair, __m512i* null
)
{
    __m512i t0[16], t1[16], idx;
    fp2_t unvec[8];
    __mmask8 add_mask;

    idx = VSET(4, 5, 2, 3, 0, 1, 0, 1); // (c, c, b, a)
    for (int i = 0; i < 8; i++) {
        t0[i] = VPERM512(idx, null[i]);
    }
    fp2_4x2x1_to_8x1x1(t0, t0, t0); // (c, c, c, c, b, b, a, a)
    
    idx = VSET(0, 1, 0, 1, 4, 5, 2, 3); // (x, x', x, x' z, z', y, y')
    for (int i = 0; i < 16; i++) {
        t1[i] = VPERM512(idx, theta_pair[i]);
    }

    // (c*x, c*x', b*x, b*x', a*z, a*z', a*y, a*y')
    fp2_mul_8x1x1(t1, t1, t0); 

    // (a*z, a*z', a*y, a*y', c*x, c*x', b*x, b*x')
    idx = VSET(4, 5, 6, 7, 0, 1, 2, 3);    
    for (int i = 0; i < 16; i++) {
        t0[i] = VPERM512(idx, t1[i]);
    }

    // (a*z + c*x, a*z' + c*x', a*y + b*x, a*y' + b*x', 
    //  a*x - c*x, a*x' - c*x', a*y - b*x, a*y' - b*x')
    add_mask = 0x0f;
    fp2_mixaddsub_8x1x1(t1, t1, t0, add_mask, add_mask);

    // Convert to 64-bit unvectorized repr.
    fp2_8x1x1_to_unvectorized(unvec, t1);
    mont_pair1->P1.x = unvec[0];
    mont_pair2->P1.x = unvec[1];
    mont_pair1->P2.x = unvec[2];
    mont_pair2->P2.x = unvec[3];
    mont_pair1->P1.z = unvec[4];
    mont_pair2->P1.z = unvec[5];
    mont_pair1->P2.z = unvec[6];
    mont_pair2->P2.z = unvec[7];
}

void splitting_theta_point_to_mont_list_pair(
    theta_couple_point_t* mont_pts, __m512i* theta_pts, 
    __m512i* null, const int num_pts
)
{
    bool num_pts_is_odd = (num_pts & 0x1);
    int num_pair_pts = num_pts / 2;

    for (int i = 0; i < num_pair_pts; i++) {
        splitting_theta_point_to_mont_pair(
            mont_pts + 2*i, mont_pts + 2*i+1, theta_pts + 16*i, null
        );
    }
    if (num_pts_is_odd) {
        splitting_theta_point_to_mont_single(
            mont_pts + (num_pts-1), theta_pts+8*(num_pts-1), null
        );
    }
}