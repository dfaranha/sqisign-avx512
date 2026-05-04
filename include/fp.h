#ifndef _FP_H
#define _FP_H

#include <stdint.h>
#include "intrinsics_avx512.h"

// ====================================================================================================
//                                Conversions between Fp representations                               
// ====================================================================================================

/**
 * Given src = (a, b, c, d, e, f, g, h) in 8x1
 * it computes res1 = (a, c, e, g) and res2 = (b, d, f, h) in 4x2.
 */
void fp_8x1_to_4x2(__m512i* res1, __m512i* res2, __m512i* src);

/**
 * Given src1 = (a1, b1, c1, d1) and src2 = (a2, b2, c2, d2) in 4x2,
 * it computes res = (a1, a2, b1, b2, c1, c2, d1, d2) in 8x1.
 */
void fp_4x2_to_8x1(__m512i* res, __m512i* src1, __m512i* src2);

/**
 * Given src=(a, b, c, d) in 4x2, it computes res1 = (a, c) and res2 = (b, d) in 2x4.
 * We choose this interleaving pattern for the Fp2 2x2x2 sqrt,
 * where res1 = re and res2 = im.
 */
void fp_4x2_to_2x4(__m512i* res1, __m512i* res2, __m512i* src);

/**
 * Given src1 = (a1, b1) and src2 = (a2, b2) in 2x4,
 * it computes res = (a1, a2, b1, b2) in 4x2.
 */
void fp_2x4_to_4x2(__m512i* res, __m512i* src1, __m512i* src2);

// ====================================================================================================
//                                      Unvectorized Fp inversion                                     
// ====================================================================================================

/**
 * Unvectorized inversion mod p using 64-bit arithmetic.
 * It uses an internal Montgomery representation with R_64 = 2^(64*NWORDS64).
 */
void fp_inv_64(uint64_t* res, const uint64_t* src);

// ====================================================================================================
//                                                 8x1                                                
// ====================================================================================================

/*
 * In the 8x1 format, we store eight Fp elements in NUM_LIMBS vectors.
 * The i-th vector stores the i-th limb of each vector.
 */

/**
 * Sets all the Fp elements to 0.
 */
void fp_setzero_8x1(__m512i* res);

/**
 * If bits[i] = 1, the i-th Fp element is set to 1 (in Montgomery repr.)
 */
void fp_set_binary_8x1(__m512i* res, __mmask8 bits);

/**
 * Copies a to res.
 */
void fp_copy_8x1(__m512i* res, __m512i* a);

/**
 * Sets each Fp element in res to a.
 * @param a The 51-bit limbs of an Fp element stored in a uint64_t array
 * @param res Output: 8 copies of a stored in 8x1.
 */
void fp_duplicate_8x1(__m512i* res, const uint64_t* a);

/**
 * Returns 8-bit mask where the i-th bit is 1 iff the i-th Fp elements are equal.
 */
__mmask8 fp_is_equal_8x1(__m512i* a, __m512i* b);

/**
 * Returns 8-bit mask where the i-th bit is 1 iff the i-th Fp element is zero.
 */
__mmask8 fp_is_zero_8x1(__m512i* a);

/**
 * Copies the i-th Fp element from a1 to res if the i-th bit of s is set, else it copies from a0.
 */
void fp_select_8x1(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);

/**
 * res = a + b (mod p)
 */
void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a - b (mod p)
 */
void fp_sub_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = -a (mod p)
 */
void fp_neg_8x1(__m512i* res, __m512i* a);

/**
 * Mixed addition and subtraction.
 * If add_mask[i]=1, res[i] = a[i] + b[i] (mod p), else res[i] = a[i] - b[i] (mod p).
 */
void fp_mixaddsub_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

/**
 * Reduces from [0, 2p) to [0, p)
 */
void fp_normalize_8x1(__m512i* res, __m512i* a);

/**
 * res = a * b (mod p)
 */
void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a^2 (mod p)
 */
void fp_sqr_8x1(__m512i* res, __m512i* a);

/**
 * res = a/2 (mod p)
 */
void fp_div2_8x1(__m512i* res, __m512i* src);

/**
 * res = a/3 (mod p)
 */
void fp_div3_8x1(__m512i* res, __m512i* src);

/**
 * res = a^(-1) (mod p). Input 0 yields 0.
 */
void fp_inv_8x1(__m512i* res, __m512i* src);

/**
 * res = sqrt(a) (mod p)
 */
void fp_sqrt_8x1(__m512i* res, __m512i* src);

/**
 * res = a^((p-3)/4) mod p
 */
void fp_exp3div4_8x1(__m512i* res, __m512i* a);

/**
 * Converts a to the Montgomery representation. res = a * R (mod p)
 */
void fp_tomont_8x1(__m512i* res, __m512i* a);

/**
 * Converts a back from the Montgomery representation. res = a * R^(-1) (mod p)
 */
void fp_frommont_8x1(__m512i* res, __m512i* a);

/**
 * res = a + b, multiprecision addition, no correction, no carry propagation
 */
void mp_add_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = res - a - b, double-length multiprecision subtraction, no correction, but propagates carry
 */
void mp_dblsubx2_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = res - a, double-length multiprecision subtraction, with carry propagation.
 * We always add p*R to make the result non-negative.
 */
void mp_subx2_8x1(__m512i* res, __m512i* a);

/**
 * double-length multiprecision mixed add sub, with carry propagation.
 * We always add p*R to make the result non-negative.
 */
void mp_mixaddsubx2_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

/**
 * (a + 2p) - b, multiprecision with no correction , but carry propagation.
 */
void mp_add_p_sub_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a * b, multiprecision multiplication.
 */
void mp_mul_8x1(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a (mod p), Montgomery reduction.
 */
void rdc_8x1(__m512i* res, __m512i* a);

/**
 * Sampled a random Fp 8x1 element.
 */
void rand_fp_8x1(__m512i* res);

/**
 * Converts from Fp 8x1 with 51-bit limbs to an array of 8*NWORDS64 64-bit words.
 * The first NWORDS64 words encode the first Fp element, and so on.
 * The function does not change the Montgomery representation.
 */
void fp_limbs_to_64_8x1(uint64_t* res, __m512i* src);

/**
 * It takes as input an arry of 8*NWORDS64 64-bit words, where the first NWORDS64 words encode the first Fp element, and so on.
 * It converts the input to an Fp 8x1 element.
 */
void fp_64_to_limbs_8x1(__m512i* res, uint64_t* unvec);

/**
 * The same as fp_limbs_to_64_8x1, except that it only extract every other Fp element (index 0, 2, 4, 6).
 */
void fp_limbs_to_64_8x1_only_even_idx(uint64_t* res, __m512i* src);

/**
 * The same as fp_64_to_limbs_8x1, except that it only takes an array of 4 Fp elements a, b, c, d in unvectorized form, 
 * and converts it to an Fp 8x1 of the form (a, 0, b, 0, c, 0, d, 0).
 */
void fp_64_to_limbs_8x1_only_even_idx(__m512i* res, uint64_t* unvec);

// ====================================================================================================
//                                                 4x2                                                
// ====================================================================================================

/*
 * In the 4x2 format, we store four Fp elements a, b, c, d in N4x2 vectors.The i-th vector stores (a_i, a_(i + N4x2), b_i, b_(i + N4x2), c_i, c_(i + N4x2), d_i, d_(i + N4x2)).
 */

/**
 * Sets all the Fp elements to 0.
 */
void fp_setzero_4x2(__m512i* res);

/**
 * If bits[i] = 1, the i-th Fp element is set to 1 (in Montgomery repr.)
 */
void fp_set_binary_4x2(__m512i* res, __mmask8 bits);

/**
 * Copies a to res.
 */
void fp_copy_4x2(__m512i* res, __m512i* a);

/**
 * Sets each Fp element in res to a.
 * @param a The 51-bit limbs of an Fp element stored in a uint64_t array
 * @param res Output: 4 copies of a stored in 4x2.
 */
void fp_duplicate_4x2(__m512i* res, const uint64_t* a);

/**
 * Returns 4-bit mask where the i-th bit is 1 iff the i-th Fp elements are equal.
 */
__mmask8 fp_is_equal_4x2(__m512i* a, __m512i* b);

/**
 * Returns 4-bit mask where the i-th bit is 1 iff the i-th Fp element is zero.
 */
__mmask8 fp_is_zero_4x2(__m512i* a);

/**
 * Copies the i-th Fp element from a1 to res if the i-th bit of s is set, else it copies from a0.
 */
void fp_select_4x2(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);

/**
 * res = a + b (mod p)
 */
void fp_add_4x2(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a - b (mod p)
 */
void fp_sub_4x2(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = -a (mod p)
 */
void fp_neg_4x2(__m512i* res, __m512i* a);

/**
 * Mixed addition and subtraction.
 * If add_mask[i]=1, res[i] = a[i] + b[i] (mod p), else res[i] = a[i] - b[i] (mod p).
 */
void fp_mixaddsub_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

/**
 * Reduces from [0, 2p) to [0, p)
 */
void fp_normalize_4x2(__m512i* res, __m512i* a);

/**
 * res = a * b (mod p)
 */
void fp_mul_4x2(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a^2 (mod p)
 */
void fp_sqr_4x2(__m512i* res, __m512i* a);

/**
 * res = a/2 (mod p)
 */
void fp_div2_4x2(__m512i* res, __m512i* src);

/**
 * res = a/3 (mod p)
 */
void fp_div3_4x2(__m512i* res, __m512i* src);

/**
 * res = a^(-1) (mod p). Input 0 yields 0.
 */
void fp_inv_4x2(__m512i* res, __m512i* src);

/**
 * res = sqrt(a) (mod p)
 */
void fp_sqrt_4x2(__m512i* res, __m512i* src);

/**
 * res = a^((p-3)/4) mod p
 */
void fp_exp3div4_4x2(__m512i* res, __m512i* a);

/**
 * Converts a to the Montgomery representation. res = a * R (mod p)
 */
void fp_tomont_4x2(__m512i* res, __m512i* a);

/**
 * Converts a back from the Montgomery representation. res = a * R^(-1) (mod p)
 */
void fp_frommont_4x2(__m512i* res, __m512i* a);

/**
 * res = a + b, multiprecision addition, no correction, no carry propagation
 */
void mp_add_4x2(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = res - a - b, double-length multiprecision subtraction, no correction or carry propagation
 */
void mp_dblsubx2_4x2(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = res - a, double-length multiprecision subtraction, without carry propagation.
 * We always add p*R to make the result non-negative.
 */
void mp_subx2_4x2(__m512i* res, __m512i* a);

/**
 * double-length multiprecision mixed add sub, without carry propagation.
 * We always add p*R to make the result non-negative.
 */
void mp_mixaddsubx2_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

/**
 * (a + 2p) - b, multiprecision with no correction or carry propagation.
 */
void mp_add_p_sub_4x2(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a * b, multiprecision multiplication.
 */
void mp_mul_4x2(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a (mod p), Montgomery reduction.
 */
void rdc_4x2(__m512i* res, __m512i* a);

/**
 * Sampled a random Fp 4x2 element.
 */
void rand_fp_4x2(__m512i* res);

/**
 * Converts from Fp 4x2 with 51-bit limbs to an array of 4*NWORDS64 64-bit words.
 * The first NWORDS64 words encode the first Fp element, and so on.
 * The function does not change the Montgomery representation.
 */
void fp_limbs_to_64_4x2(uint64_t* res, __m512i* src);

/**
 * It takes as input an arry of 4*NWORDS64 64-bit words, where the first NWORDS64 words encode the first Fp element, and so on.
 * It converts the input to an Fp 8x1 element.
 */
void fp_64_to_limbs_4x2(__m512i* res, uint64_t* unvec);

// ====================================================================================================
//                                                 2x4                                                
// ====================================================================================================

/*
 * In the 2x4 format, we store two Fp element a and b in N2x4 vectors.The i-th vector stores (a_i, a_(i + N2x4), a_(i + 2*N2x4), a_(i + 3*N2x4), b_i, b_(i + N2x4), b_(i + 2*N2x4), b_(i + 3*N2x4)).
 */

/**
 * Sets all the Fp elements to 0.
 */
void fp_setzero_2x4(__m512i* res);

/**
 * If bits[i] = 1, the i-th Fp element is set to 1 (in Montgomery repr.)
 */
void fp_set_binary_2x4(__m512i* res, __mmask8 bits);

/**
 * Copies a to res.
 */
void fp_copy_2x4(__m512i* res, __m512i* a);

/**
 * Sets each Fp element in res to a.
 * @param a The 51-bit limbs of an Fp element stored in a uint64_t array
 * @param res Output: 2 copies of a stored in 2x4.
 */
void fp_duplicate_2x4(__m512i* res, const uint64_t* a);

/**
 * Returns 2-bit mask where the i-th bit is 1 iff the i-th Fp elements are equal.
 */
__mmask8 fp_is_equal_2x4(__m512i* a, __m512i* b);

/**
 * Returns 2-bit mask where the i-th bit is 1 iff the i-th Fp element is zero.
 */
__mmask8 fp_is_zero_2x4(__m512i* a);

/**
 * Copies the i-th Fp element from a1 to res if the i-th bit of s is set, else it copies from a0.
 */
void fp_select_2x4(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);

/**
 * res = a + b (mod p)
 */
void fp_add_2x4(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a - b (mod p)
 */
void fp_sub_2x4(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = -a (mod p)
 */
void fp_neg_2x4(__m512i* res, __m512i* a);

/**
 * Mixed addition and subtraction.
 * If add_mask[i]=1, res[i] = a[i] + b[i] (mod p), else res[i] = a[i] - b[i] (mod p).
 */
void fp_mixaddsub_2x4(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);

/**
 * Reduces from [0, 2p) to [0, p)
 */
void fp_normalize_2x4(__m512i* res, __m512i* a);

/**
 * res = a * b (mod p)
 */
void fp_mul_2x4(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a^2 (mod p)
 */
void fp_sqr_2x4(__m512i* res, __m512i* a);

/**
 * res = a/2 (mod p)
 */
void fp_div2_2x4(__m512i* res, __m512i* src);

/**
 * res = a/3 (mod p)
 */
void fp_div3_2x4(__m512i* res, __m512i* src);

/**
 * res = a^(-1) (mod p). Input 0 yields 0.
 */
void fp_inv_2x4(__m512i* res, __m512i* src);

/**
 * res = sqrt(a) (mod p)
 */
void fp_sqrt_2x4(__m512i* res, __m512i* src);

/**
 * res = a^((p-3)/4) mod p
 */
void fp_exp3div4_2x4(__m512i* res, __m512i* a);

/**
 * Converts a to the Montgomery representation. res = a * R (mod p)
 */
void fp_tomont_2x4(__m512i* res, __m512i* a);

/**
 * Converts a back from the Montgomery representation. res = a * R^(-1) (mod p)
 */
void fp_frommont_2x4(__m512i* res, __m512i* a);

/**
 * res = a * b, multiprecision multiplication.
 */
void mp_mul_2x4(__m512i* res, __m512i* a, __m512i* b);

/**
 * res = a (mod p), Montgomery reduction.
 */
void rdc_2x4(__m512i* res, __m512i* a);

/**
 * Sampled a random Fp 2x4 element.
 */
void rand_fp_2x4(__m512i* res);

/**
 * Converts from Fp 2x4 with 51-bit limbs to an array of 2*NWORDS64 64-bit words.
 * The first NWORDS64 words encode the first Fp element, and so on.
 * The function does not change the Montgomery representation.
 */
void fp_limbs_to_64_2x4(uint64_t* res, __m512i* src);

/**
 * It takes as input an arry of 2*NWORDS64 64-bit words, where the first NWORDS64 words encode the first Fp element, and so on.
 * It converts the input to an Fp 8x1 element.
 */
void fp_64_to_limbs_2x4(__m512i* res, uint64_t* unvec);

#endif
