#include "fp.h"
#include "fp_consts.h"
#include "fp_unvec.h"
#include "fp2.h"
#include "ec.h"
#include "theta.h"
#include "theta_splitting.h"
#include "utils.h"

#include "sqi_fp2.h"

#include <string.h>

int splitting_get_index(__m512i* null)
{
    __m512i tmp1[NUM_LIMBS], tmp2[NUM_LIMBS], tmp3[NUM_LIMBS], tmp4[NUM_LIMBS];
    __m512i idx1, idx2, idx3, idx4;
    __mmask8 add_mask;
    int is_zero, index = 0, num_found = 0, t;

    // === (0, 0), (1, 0), (2, 0), (3, 0) === 
    fp2_sqr_4x2x1(tmp1, null);
    
    idx1 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    idx2 = VSET(2, 3, 2, 3, 2, 3, 2, 3);
    idx3 = VSET(4, 5, 4, 5, 4, 5, 4, 5);
    idx4 = VSET(6, 7, 6, 7, 6, 7, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    for (int i = 0; i < NUM_LIMBS; i++) {
        tmp1[i] = VPERM512(idx1, null[i]);
        tmp2[i] = VPERM512(idx2, null[i]);
    }
    fp2_mul_4x2x1(tmp1, tmp1, tmp2);
    
    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx2 = VSET(6, 7, 4, 5, 6, 7, 5, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    // preparing 2x2x2 directly
    
    // tmp1 = (a, b), tmp2 = (d, c)
    fp2_4x2x1_to_2x2x2(tmp1, tmp2, null);
    idx1 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
    for (int i = 0; i < N4x2; i++) {
        tmp2[i] = VPERM512(idx1, tmp2[i]);
    }
    fp2_mul_2x2x2(tmp1, tmp1, tmp2);

    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    idx2 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
    for (int i = 0; i < N4x2; i++) {
        tmp2[i] = VPERM512(idx2, tmp1[i]);
        tmp1[i] = VPERM512(idx1, tmp1[i]);
    }
    add_mask = 0x0f;
    fp2_mixaddsub_2x2x2(tmp1, tmp1, tmp2, add_mask);
    is_zero = fp2_is_zero_2x2x2(tmp1);
    t = is_zero & 0x1;
    num_found += t;
    index = index*(1-t) + 3*t;
    t = (is_zero >> 1) & 0x1;
    num_found += t;
    index = index*(1-t) + 9*t;

    // printf("num_found %d\n", num_found);
    if (num_found != 1) {
        return -1;
    } 
    return index;
} 

bool splitting_isom_matrix(splitting_matrix_t* matrix, __m512i* null, bool verify)
{
    int index = splitting_get_index(null);
    if (index == -1) {
        return false;
    }
    *matrix = SPLITTING_MATRICES[index];
    return true;
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
    __m512i tmp[NUM_LIMBS];
    
    // Negate re when matrix entry is -1 or -i. Negate im when -1 or i.
    __mmask8 negate_mask = matrix->is_negative[index] ^ (0xaa & matrix->is_imaginary[index]);
    fp2_neg_4x2x1(tmp, dup);
    
    for (int i = 0; i < NUM_LIMBS; i++) {
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
    __m512i vec[NUM_LIMBS], point_copy[NUM_LIMBS], indices;
    __mmask8 negate_mask;
    fp2_copy_4x2x1(point_copy, point); // in case res = point

    splitting_duplicate_pair_in_array(res, point_copy, 0, NUM_LIMBS);
    splitting_mul_matrix_column_single(res, res, matrix, 0);

    for (int j = 1; j < 4; j++) {
        splitting_duplicate_pair_in_array(vec, point_copy, j, NUM_LIMBS);
        splitting_mul_matrix_column_single(vec, vec, matrix, j);
        fp2_add_4x2x1(res, res, vec);
    }
}

void splitting_apply_isomorphism_list_single(__m512i* pts, const splitting_matrix_t* matrix, const int num_pts)
{
    for (int i = 0; i < num_pts; i++) {
        splitting_apply_isomorphism_single(pts + i*NUM_LIMBS, pts + i*NUM_LIMBS, matrix);
    }
}

// dup = (x, x', x, x', x, x', x, x') in the 8x1x1 format.
// Multiplies the index-th column of the matrix A by x and x'.
// res = (a_(1, i) * x, a_(1, i) * x', ...)
void splitting_mul_matrix_column_pair(__m512i* res, __m512i* dup, const splitting_matrix_t* matrix, const int index)
{
    __m512i tmp[NUM_LIMBS];
    __mmask8 negate_mask;
    
    // Negate re when matrix entry is -1 or -i.
    negate_mask = matrix->is_negative[index];
    fp_neg_8x1(tmp, dup);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res[i] = VBLEND(negate_mask, dup[i], tmp[i]);

        // conditionally set to 0
        res[i] = VBLEND(matrix->is_zero[index], res[i], VZERO());
    }

    // Negate im when -1 or i.
    negate_mask = matrix->is_negative[index] ^ matrix->is_imaginary[index];
    fp_neg_8x1(tmp, dup + NUM_LIMBS);
    for (int i = 0; i < NUM_LIMBS; i++) {
        res[i + NUM_LIMBS] = VBLEND(negate_mask, dup[i + NUM_LIMBS], tmp[i]);

        // conditionally set to 0
        res[i + NUM_LIMBS] = VBLEND(matrix->is_zero[index], res[i + NUM_LIMBS], VZERO());

        // conditionally swap re and im if multiplying by imaginary
        tmp[i] = res[i];
        res[i] = VBLEND(matrix->is_imaginary[index], res[i], res[i + NUM_LIMBS]);
        res[i + NUM_LIMBS] = VBLEND(matrix->is_imaginary[index], res[i + NUM_LIMBS], tmp[i]);
    }
}

// Multiplies the points by the matrix A.
// The points are stored in the 8x1x1 format as (x, x', y, y', z, z', t, t')
// To multply x and x' by the first column of A, we create a vector (x, x', x, x', x, x', x, x')
void splitting_apply_isomorphism_pair(__m512i* res, __m512i* points, const splitting_matrix_t* matrix)
{
    __m512i vec[2*NUM_LIMBS], points_copy[2*NUM_LIMBS], indices;
    __mmask8 negate_mask;
    fp2_copy_8x1x1(points_copy, points); // in case res = point

    splitting_duplicate_pair_in_array(res, points_copy, 0, 2*NUM_LIMBS);
    splitting_mul_matrix_column_pair(res, res, matrix, 0);

    for (int j = 1; j < 4; j++) {
        splitting_duplicate_pair_in_array(vec, points_copy, j, 2*NUM_LIMBS);
        splitting_mul_matrix_column_pair(vec, vec, matrix, j);
        fp2_add_8x1x1(res, res, vec);
    }
}

void splitting_apply_isomorphism_list_pair(__m512i* pts, const splitting_matrix_t* matrix, const int num_pts)
{
    bool num_pts_is_odd = (num_pts & 0x1);
    int num_pair_pts = num_pts / 2;

    for (int i = 0; i < num_pair_pts; i++) {
        splitting_apply_isomorphism_pair(pts + i*(2*NUM_LIMBS), pts + i*(2*NUM_LIMBS), matrix);
    }
    if (num_pts_is_odd) {
        splitting_apply_isomorphism_single(
            pts + (num_pts-1)*NUM_LIMBS, pts + (num_pts-1)*NUM_LIMBS, matrix
        );
    }
}

bool splitting_theta_to_product(__m512i* res, __m512i* null, const bool verify) 
{    
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], idx0;

    // assert a * d == b * c
    if (verify) {
        // t0 = (a, b), t1 = (d, c)
        fp2_4x2x1_to_2x2x2(t0, t1, null); 
        idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
        for (int i = 0; i < N4x2; i++) {
            t1[i] = VPERM512(idx0, t1[i]);
        }
        fp2_mul_2x2x2(t0, t0, t1);
        idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
        for (int i = 0; i < N4x2; i++) {
            t1[i] = VPERM512(idx0, t0[i]);
        }
        
        bool equal = (fp2_is_equal_2x2x2(t0, t1) & 0x1) == 0x1;
        if (!equal) { return false; }
    }
    
    // Normalized curves from the null point pairs (a, c) and (a, b).
    fp2_4x2x1_to_2x2x2(t0, t1, null);   // t0 = (a, b), t1 = (c, d)

    idx0 = VSET(0, 1, 2, 3, 8, 9, 10, 11);
    for (int i = 0; i < N4x2; i++) {
        t1[i] = VBLENDPERM(idx0, t0[i], t1[i]);   // (a, c)
    }

    if (!ec_normalized_A_from_pair_2x2x2(t1, t1) || !ec_normalized_A_from_pair_2x2x2(t0, t0)) {
        return false;   // failed to split to a product of elliptic curves
    }

    // res = (A1, 1, A2, 1)
    fp2_2x2x2_to_4x2x1(res, t1, t0);

    return true;
}

bool splitting_theta_to_product_unvec(theta_couple_curve_t* res, __m512i* null, const bool verify) 
{    
    fp2_t unvec[4];
    __m512i t0[NUM_LIMBS];
    bool ret = splitting_theta_to_product(t0, null, verify);
    
    // Finally, we convert to the unvectorized 64-bit repr.
    fp2_4x2x1_to_unvectorized(unvec, t0);
    ec_curve_init(&res->E1);
    ec_curve_init(&res->E2);
    res->E1.A = unvec[0];
    res->E1.C = unvec[1];
    res->E2.A = unvec[2];
    res->E2.C = unvec[3];

    return ret;
}

// Outputs the pair of montgomery points (X1, Z1, X2, Z2) in 4x2x1
bool splitting_theta_point_to_mont_single(
    __m512i* res, __m512i* theta_point, __m512i* null, const bool verify
)
{
    __m512i t0[NUM_LIMBS], t1[NUM_LIMBS], idx0, idx1;
    __mmask8 mask;

    // Check P != (0, 0, 0, 0) and x * w == y * z
    __mmask8 zmask = fp2_is_zero_4x2x1(theta_point);
    if (verify) {
        if (zmask == 0xf) { return false; }

        // t0 = (x, y), t1 = (z, w)
        fp2_4x2x1_to_2x2x2(t0, t1, theta_point); 
        idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
        for (int i = 0; i < N4x2; i++) {
            t1[i] = VPERM512(idx0, t1[i]);
        }
        fp2_mul_2x2x2(t0, t0, t1);
        idx0 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
        for (int i = 0; i < N4x2; i++) {
            t1[i] = VPERM512(idx0, t0[i]);
        }
        
        bool equal = (fp2_is_equal_2x2x2(t0, t1) & 0x1) == 0x1;
        if (!equal) { return false; }
    }

    // Computing (X1, Z1, X2, Z2) depending on these flags, in constant-time
    __mmask8 xz_zero = (zmask >> 2) & zmask & 0x1;
    __mmask8 xy_zero = (zmask >> 1) & zmask & 0x1;
    mask = xz_zero | (xz_zero << 1) | (xy_zero << 2) | (xy_zero << 3);
    
    // t0 = (a, a, a, a)
    // t1[0,1] = (w, w) if (x = 0 and z = 0) else (z, z)
    // t1[2,3] = (w, w) if (x = 0 and y = 0) else (y, y)
    // We compute t1 first
    idx0 = VSET(4, 5, 4, 5, 2, 3, 2, 3);
    idx1 = VSET(6, 7, 6, 7, 6, 7, 6, 7);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, theta_point[i]);
        t1[i] = VPERM512(idx1, theta_point[i]);
    }
    fp2_select_4x2x1(t1, t0, t1, mask);
    idx0 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, null[i]);
    }
    fp2_mul_4x2x1(res, t0, t1);

    // t0 = (c, c, b, b)
    // t1[0,1] = (y, y) if (x = 0 and z = 0) else (x, x)
    // t1[2,3] = (z, z) if (x = 0 and y = 0) else (x, x)
    idx1 = VSET(2, 3, 2, 3, 4, 5, 4, 5);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, theta_point[i]);
        t1[i] = VPERM512(idx1, theta_point[i]);
    }
    fp2_select_4x2x1(t1, t0, t1, mask);
    idx0 = VSET(4, 5, 4, 5, 2, 3, 2, 3);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, null[i]);
    }
    fp2_mul_4x2x1(t0, t0, t1);
    fp2_mixaddsub_4x2x1(res, res, t0, 0x33);

    return true;
}

bool splitting_theta_point_to_mont_single_unvec(
    theta_couple_point_t* mont_pair, __m512i* theta_point, __m512i* null, const bool verify
)
{
    fp2_t u[4];
    __m512i t[NUM_LIMBS];

    bool ret = splitting_theta_point_to_mont_single(t, theta_point, null, verify);

    fp2_4x2x1_to_unvectorized(u, t);
    fp2_copy(&mont_pair->P1.x, u);
    fp2_copy(&mont_pair->P1.z, u+1);
    fp2_copy(&mont_pair->P2.x, u+2);
    fp2_copy(&mont_pair->P2.z, u+3);

    return ret;
}

// Outputs pair of points (X3, Z3, X4, Z4) in 4x2x1.
bool splitting_theta_point_to_mont_pair(
    __m512i* mont_pair1, __m512i* mont_pair2, __m512i* theta_pair, __m512i* null, const bool verify
)
{
    __m512i t0[2*NUM_LIMBS], t1[2*NUM_LIMBS], t2[2*NUM_LIMBS], idx0, idx1;
    __mmask8 mask;

    // check that points are not zero and that xi * wi == yi * zi
    __mmask8 zmask = fp2_is_zero_8x1x1(theta_pair);
    if (verify) {
        bool P_zero = zmask & 0x55 == 0x55;
        bool Q_zero = zmask & 0xaa == 0xaa;
        if (P_zero || Q_zero) {
            return false;
        }

        idx0 = VSET(0, 8, 1, 9, 2, 10, 3, 11);
        idx1 = VSET(6, 14, 7, 15, 4, 12, 5, 13);
        for (int i = 0; i < NUM_LIMBS; i++) {
            t0[i] = VBLENDPERM(idx0, theta_pair[i], theta_pair[i + NUM_LIMBS]);
            t1[i] = VBLENDPERM(idx1, theta_pair[i], theta_pair[i + NUM_LIMBS]);
        }
        fp2_mul_4x2x1(t0, t0, t1);
        idx0 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
        idx1 = VSET(4, 5, 6, 7, 4, 5, 6, 7);
        for (int i = 0; i < NUM_LIMBS; i++) {
            t1[i] = VPERM512(idx1, t0[i]);
            t0[i] = VPERM512(idx0, t0[i]);
        }
        if (fp2_is_equal_4x2x1(t0, t1) != 0xf) { return false; }
    }

    // Computing output depending on these flags, in constant-time
    // Recall that points are stored as (X1, X2, Y1, Y2, Z1, Z2, W1, W2)
    __mmask8 xz_zero1 = (zmask >> 4) & zmask & 0x1;
    __mmask8 xy_zero1 = (zmask >> 2) & zmask & 0x1;
    __mmask8 xz_zero2 = (zmask >> 5) & (zmask >> 1) & 0x1;
    __mmask8 xy_zero2 = (zmask >> 3) & (zmask >> 1) & 0x1;

    mask = xz_zero1 | xz_zero2 | (xz_zero1 << 1) | (xz_zero2 << 1) 
        | (xy_zero1 << 2) | (xy_zero2 << 2) | (xy_zero1 << 3) | (xy_zero2 << 3);

    // We do this (but interleaved in 8x1x1):
    // t0 = (a, a, a, a)
    // t1[0,1] = (w, w) if (x = 0 and z = 0) else (z, z)
    // t1[2,3] = (w, w) if (x = 0 and y = 0) else (y, y)
    idx0 = VSET(4, 5, 4, 5, 2, 3, 2, 3);
    idx1 = VSET(6, 7, 6, 7, 6, 7, 6, 7);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, theta_pair[i]);
        t1[i] = VPERM512(idx1, theta_pair[i]);
    }
    fp2_select_8x1x1(t1, t0, t1, mask);
    idx0 = VSET(0, 0, 0, 0, 0, 0, 0, 0);
    idx1 = VSET(1, 1, 1, 1, 1, 1, 1, 1);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, null[i]);
        t0[i + NUM_LIMBS] = VPERM512(idx1, null[i]);
    }
    fp2_mul_8x1x1(t2, t0, t1);

    // t0 = (c, c, b, b)
    // t1[0,1] = (y, y) if (x = 0 and z = 0) else (x, x)
    // t1[2,3] = (z, z) if (x = 0 and y = 0) else (x, x)
    idx0 = VSET(0, 1, 0, 1, 0, 1, 0, 1);
    idx1 = VSET(2, 3, 2, 3, 4, 5, 4, 5);
    for (int i = 0; i < 2*NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, theta_pair[i]);
        t1[i] = VPERM512(idx1, theta_pair[i]);
    }
    fp2_select_8x1x1(t1, t0, t1, mask);
    idx0 = VSET(4, 4, 4, 4, 2, 2, 2, 2);
    idx1 = VSET(5, 5, 5, 5, 3, 3, 3, 3);
    for (int i = 0; i < NUM_LIMBS; i++) {
        t0[i] = VPERM512(idx0, null[i]);
        t0[i + NUM_LIMBS] = VPERM512(idx1, null[i]);
    }
    fp2_mul_8x1x1(t0, t0, t1);
    fp2_mixaddsub_8x1x1(t2, t2, t0, 0x33, 0x33);

    fp2_8x1x1_to_4x2x1(mont_pair1, mont_pair2, t2);

    return true;
}

bool splitting_theta_point_to_mont_pair_unvec(
    theta_couple_point_t* mont_pair1, theta_couple_point_t* mont_pair2, 
    __m512i* theta_pair, __m512i* null, const bool verify
)
{
    fp2_t unvec[4];
    __m512i t1[NUM_LIMBS], t2[NUM_LIMBS];

    bool ret = splitting_theta_point_to_mont_pair(t1, t2, theta_pair, null, verify);

    fp2_4x2x1_to_unvectorized(unvec, t1);
    mont_pair1->P1.x = unvec[0];
    mont_pair1->P1.z = unvec[1];
    mont_pair1->P2.x = unvec[2];
    mont_pair1->P2.z = unvec[3];

    fp2_4x2x1_to_unvectorized(unvec, t2);
    mont_pair2->P1.x = unvec[0];
    mont_pair2->P1.z = unvec[1];
    mont_pair2->P2.x = unvec[2];
    mont_pair2->P2.z = unvec[3];

    return ret;
}

bool splitting_theta_point_to_mont_list_single(
    theta_couple_point_t* mont_pts, __m512i* theta_pts, 
    __m512i* null, const int num_pts, const bool verify
)
{
    bool success = true;
    for (int i = 0; i < num_pts; i++) {
        success &= splitting_theta_point_to_mont_single_unvec(mont_pts + i, theta_pts + i*NUM_LIMBS, null, verify);
    }
    return success;
}

bool splitting_theta_point_to_mont_list_pair(
    __m512i* mont_pts, __m512i* theta_pts, 
    __m512i* null, const int num_pts, const bool verify
)
{
    bool success = true;
    bool num_pts_is_odd = (num_pts & 0x1);
    int num_pair_pts = num_pts / 2;

    for (int i = 0; i < num_pair_pts; i++) {
        success &= splitting_theta_point_to_mont_pair(
            mont_pts + (2*i)*NUM_LIMBS, mont_pts + (2*i+1)*NUM_LIMBS, theta_pts + i*(2*NUM_LIMBS), null, verify
        );
    }
    if (num_pts_is_odd) {
        success &= splitting_theta_point_to_mont_single(
            mont_pts + (num_pts-1)*NUM_LIMBS, theta_pts + (num_pts-1)*NUM_LIMBS, null, verify
        );
    }
    return success;
}

bool splitting_theta_point_to_mont_list_pair_unvec(
    theta_couple_point_t* mont_pts, __m512i* theta_pts, 
    __m512i* null, const int num_pts, const bool verify
)
{
    bool success = true;
    bool num_pts_is_odd = (num_pts & 0x1);
    int num_pair_pts = num_pts / 2;

    for (int i = 0; i < num_pair_pts; i++) {
        success &= splitting_theta_point_to_mont_pair_unvec(
            mont_pts + 2*i, mont_pts + 2*i+1, theta_pts + i*(2*NUM_LIMBS), null, verify
        );
    }
    if (num_pts_is_odd) {
        success &= splitting_theta_point_to_mont_single_unvec(
            mont_pts + (num_pts-1), theta_pts + (num_pts-1)*NUM_LIMBS, null, verify
        );
    }
    return success;
}
