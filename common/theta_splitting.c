#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "theta.h"
#include "theta_splitting.h"

#include "sqi_fp2.h"

#include <string.h>

bool splitting_chi(int i, int j)
{
    for (int k = 0; k < NUM_EVEN_PAIRS; k++) {
        if (i == EVEN_I[k] && j == EVEN_J[k]) {
            return true;
        }
    }
    return false;
}

int splitting_get_index(__m512i* null) 
{
    int count = 0;
    int index = -1;
    int i, j;
    fp2_t null_unvec[4];
    fp2_t sum, term;
    fp2_4x2x1_to_unvectorized(null_unvec, null);

    for (int k = 0; k < NUM_EVEN_PAIRS; k++) {
        i = EVEN_I[k];
        j = EVEN_J[k];
        fp2_set_zero(&sum);
        for (int l = 0; l < 4; l++) {
            fp2_mul(&term, null_unvec + l, null_unvec + (j ^ l));
            if (splitting_chi(i, l)) {
                fp2_add(&sum, &sum, &term);
            }
            else {
                fp2_sub(&sum, &sum, &term);
            }
        }
        if (fp2_is_zero(&sum)) {
            index = k;
            count += 1;
        }
    }

    if (count != 1) {
        return -1;
    }
    return index;
}

void splitting_isom_matrix(splitting_matrix_t* matrix, __m512i* null)
{
    int index = splitting_get_index(null);
    *matrix = SPLITTING_MATRICES[index];
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

void splitting_apply_isomorphism_old(__m512i* res, __m512i* null, int* matrix)
{
    // TODO: Atm we unvectorize multiple times, to match the avx512 interface of the methods.
    //       This is suboptimal, but maybe we want to get rid of the unvectorized code anyways?
    fp2_t null_unvec[4], res_unvec[4], tmp;
    fp2_4x2x1_to_unvectorized(null_unvec, null);

    for (int i = 0; i < 4; i++) {
        fp2_set_zero(res_unvec+i);
        for (int j = 0; j < 4; j++) {
            int matrix_entry = matrix[4*i + j];
            if (matrix_entry == 1) {
                fp2_add(res_unvec+i, res_unvec+i, null_unvec+j);
            }
            if (matrix_entry == 3) {
                fp2_sub(res_unvec+i, res_unvec+i, null_unvec+j);
            }
            if (matrix_entry == 5) {
                fp2_mul_i(&tmp, null_unvec+j, 1);
                fp2_add(res_unvec+i, res_unvec+i, &tmp);
            }
            if (matrix_entry == 7) {
                fp2_mul_i(&tmp, null_unvec+j, 1);
                fp2_sub(res_unvec+i, res_unvec+i, &tmp);
            }
        }
    }
    fp2_unvectorized_to_4x2x1(res, res_unvec);
}

void splitting_theta_to_product(__m512i* product, __m512i* null) 
{
    // TODO: Atm we unvectorize multiple times, to match the avx512 interface of the methods.
    //       This is suboptimal, but maybe we want to get rid of the unvectorized code anyways?
    fp2_t null_unvec[4], product_unvec[4], x, y, z;
    fp2_4x2x1_to_unvectorized(null_unvec, null);
    
    // x = a^4, y = b^4, z = c^4
    fp2_sqr(&x, &null_unvec[0]);
    fp2_sqr(&x, &x);
    fp2_sqr(&y, &null_unvec[1]);
    fp2_sqr(&y, &y);
    fp2_sqr(&z, &null_unvec[2]);
    fp2_sqr(&z, &z);

    // A1 = -2*(x+z), C2 = x - z
    fp2_add(product_unvec, &x, &z);
    fp2_add(product_unvec, product_unvec, product_unvec);
    fp2_neg(product_unvec, product_unvec);
    fp2_sub(product_unvec+1, &x, &z);

    // A2 = -2*(x+y), C2 = x - y
    fp2_add(product_unvec+2, &x, &y);
    fp2_add(product_unvec+2, product_unvec+2, product_unvec+2);
    fp2_neg(product_unvec+2, product_unvec+2);
    fp2_sub(product_unvec+3, &x, &y);

    //TODO: Python code checks that a * d = b * c and that C1, C2 != 0.

    fp2_unvectorized_to_4x2x1(product, product_unvec);
}

void splitting_theta_point_to_mont_single(__m512i* mont_points, __m512i* theta_point, __m512i* null)
{
    fp2_t theta_point_unvec[4], null_unvec[4], mont_points_unvec[4], m1, m2, m3, m4;
    fp2_4x2x1_to_unvectorized(theta_point_unvec, theta_point); // theta = (x, y, z, _)
    fp2_4x2x1_to_unvectorized(null_unvec, null); // null = (a, b, c, _)

    fp2_mul(&m1, null_unvec, theta_point_unvec+1); // a*y
    fp2_mul(&m2, null_unvec, theta_point_unvec+2); // a*z
    fp2_mul(&m3, null_unvec+1, theta_point_unvec); // b*x
    fp2_mul(&m4, null_unvec+2, theta_point_unvec); // c*x

    fp2_add(mont_points_unvec, &m2, &m4);   // X1 = a * z + c * x
    fp2_add(mont_points_unvec+1, &m1, &m3); // X2 = a * y + b * x
    fp2_sub(mont_points_unvec+2, &m2, &m4); // Z1 = a * z - c * x
    fp2_sub(mont_points_unvec+3, &m1, &m3); // Z2 = a * y  - b * x

    fp2_unvectorized_to_4x2x1(mont_points, mont_points_unvec);
}

void splitting_theta_point_to_mont_pair(__m512i* mont_points, __m512i* theta_points, __m512i* null)
{
    fp2_t theta_points_unvec[8], null_unvec[4], mont_points_unvec[8], tmp[8];
    // theta = (x, x', y, y', z, z', _, _)
    fp2_8x1x1_to_unvectorized(theta_points_unvec, theta_points);
    // null = (a, b, c, _)
    fp2_4x2x1_to_unvectorized(null_unvec, null); 

    fp2_mul(tmp, null_unvec, theta_points_unvec+2); // a*y
    fp2_mul(tmp+1, null_unvec, theta_points_unvec+3);
    fp2_mul(tmp+2, null_unvec, theta_points_unvec+4); // a*z
    fp2_mul(tmp+3, null_unvec, theta_points_unvec+5);
    fp2_mul(tmp+4, null_unvec+1, theta_points_unvec); // b*x
    fp2_mul(tmp+5, null_unvec+1, theta_points_unvec+1);
    fp2_mul(tmp+6, null_unvec+2, theta_points_unvec); // c*x
    fp2_mul(tmp+7, null_unvec+2, theta_points_unvec+1);
    

    fp2_add(mont_points_unvec, tmp+2, tmp+6);   // X1 = a * z + c * x
    fp2_add(mont_points_unvec+1, tmp+3, tmp+7);
    fp2_add(mont_points_unvec+2, tmp, tmp+4);   // X2 = a * y + b * x
    fp2_add(mont_points_unvec+3, tmp+1, tmp+5);
    fp2_sub(mont_points_unvec+4, tmp+2, tmp+6); // Z1 = a * z - c * x
    fp2_sub(mont_points_unvec+5, tmp+3, tmp+7);
    fp2_sub(mont_points_unvec+6, tmp, tmp+4);   // Z2 = a * y  - b * x
    fp2_sub(mont_points_unvec+7, tmp+1, tmp+5);

    fp2_unvectorized_to_8x1x1(mont_points, mont_points_unvec);
}