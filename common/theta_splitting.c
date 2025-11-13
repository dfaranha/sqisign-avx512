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

void splitting_theta_to_product(__m512i* product, __m512i* null) 
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

void splitting_theta_point_to_mont_single(__m512i* mont_points, __m512i* theta_point, __m512i* null)
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

void splitting_theta_point_to_mont_pair(__m512i* mont_points, __m512i* theta_points, __m512i* null)
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