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

void splitting_isom_matrix(int* matrix, __m512i* null)
{
    int index = splitting_get_index(null);
    switch (index) {
        case 0:
            memcpy(matrix, (int[16]) {1, 5, 1, 5, 1, 7, 3, 5, 1, 5, 3, 7, 3, 5, 3, 5}, sizeof(int)*16);
            break;
        case 1:
            memcpy(matrix, (int[16]) {1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 3, 0, 0}, sizeof(int)*16);
            break;
        case 2:
            memcpy(matrix, (int[16]) {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 3, 0}, sizeof(int)*16);
            break;
        case 3:
            memcpy(matrix, (int[16]) {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 3}, sizeof(int)*16);
            break;
        case 4:
            memcpy(matrix, (int[16]) {1, 1, 1, 1, 1, 3, 3, 1, 1, 1, 3, 3, 3, 1, 3, 1}, sizeof(int)*16);
            break;
        case 5:
            memcpy(matrix, (int[16]) {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0}, sizeof(int)*16);
            break;
        case 6:
            memcpy(matrix, (int[16]) {1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 3, 1, 3, 3, 1, 1}, sizeof(int)*16);
            break;
        case 7:
            memcpy(matrix, (int[16]) {1, 1, 1, 1, 1, 3, 1, 3, 1, 3, 3, 1, 1, 1, 3, 3}, sizeof(int)*16);
            break;
        case 8:
            memcpy(matrix, (int[16]) {1, 1, 1, 1, 1, 3, 1, 3, 1, 1, 3, 3, 3, 1, 1, 3}, sizeof(int)*16);
            break;
        default: //TODO should be case 9, default should throw error. It is this way for now to make the benchmarks with nonsense input not crash.
            memcpy(matrix, (int[16]) {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}, sizeof(int)*16);
        
    }
}

void splitting_apply_isomorphism(__m512i* res, int* matrix, __m512i* null)
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
                fp2_muli(&tmp, null_unvec+j);
                fp2_add(res_unvec+i, res_unvec+i, &tmp);
            }
            if (matrix_entry == 7) {
                fp2_muli(&tmp, null_unvec+j);
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