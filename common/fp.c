#include <stdio.h>
#include <immintrin.h>

#include "fp.h"
#include "fp_consts.h"
#include "utils.h"

// ====================================================================================================
//                                                 8x1                                                
// ====================================================================================================

void fp_setzero_8x1(__m512i* res)
{
    res[0] = VZERO();
    res[1] = VZERO();
    res[2] = VZERO();
    res[3] = VZERO();
    res[4] = VZERO();
    res[5] = VZERO();
    res[6] = VZERO();
    res[7] = VZERO();
    res[8] = VZERO();
    res[9] = VZERO();
    res[10] = VZERO();
}

void fp_copy_8x1(__m512i* res, __m512i* a)
{
    res[0] = a[0];
    res[1] = a[1];
    res[2] = a[2];
    res[3] = a[3];
    res[4] = a[4];
    res[5] = a[5];
    res[6] = a[6];
    res[7] = a[7];
    res[8] = a[8];
    res[9] = a[9];
    res[10] = a[10];
}

void fp_duplicate_8x1(__m512i* res, const uint64_t* a)
{
    res[0] = VSETALL(a[0]);
    res[1] = VSETALL(a[1]);
    res[2] = VSETALL(a[2]);
    res[3] = VSETALL(a[3]);
    res[4] = VSETALL(a[4]);
    res[5] = VSETALL(a[5]);
    res[6] = VSETALL(a[6]);
    res[7] = VSETALL(a[7]);
    res[8] = VSETALL(a[8]);
    res[9] = VSETALL(a[9]);
    res[10] = VSETALL(a[10]);
}

int fp_is_equal_8x1(__m512i* a, __m512i* b)
{
    __m512i zero[NUM_LIMBS], norm_a[NUM_LIMBS], norm_b[NUM_LIMBS];
    __mmask8 acc;
    fp_setzero_8x1(zero);
    
    // Normalize both to [0, p-1] by adding zero
    fp_add_8x1(norm_a, a, zero);
    fp_add_8x1(norm_b, b, zero);
    
    acc = 0xff;
    acc &= VCMPEQ(norm_a[0], norm_b[0]);
    acc &= VCMPEQ(norm_a[1], norm_b[1]);
    acc &= VCMPEQ(norm_a[2], norm_b[2]);
    acc &= VCMPEQ(norm_a[3], norm_b[3]);
    acc &= VCMPEQ(norm_a[4], norm_b[4]);
    acc &= VCMPEQ(norm_a[5], norm_b[5]);
    acc &= VCMPEQ(norm_a[6], norm_b[6]);
    acc &= VCMPEQ(norm_a[7], norm_b[7]);
    acc &= VCMPEQ(norm_a[8], norm_b[8]);
    acc &= VCMPEQ(norm_a[9], norm_b[9]);
    acc &= VCMPEQ(norm_a[10], norm_b[10]);
    return acc;
}

int fp_is_zero_8x1(__m512i* a)
{
    __m512i zero[NUM_LIMBS], norm_a[NUM_LIMBS];
    __mmask8 acc;
    fp_setzero_8x1(zero);
    
    // Normalize a to [0, p-1] by adding zero
    fp_add_8x1(norm_a, a, zero);
    
    acc = 0xff;
    acc &= VCMPEQ(norm_a[0], zero[0]);
    acc &= VCMPEQ(norm_a[1], zero[1]);
    acc &= VCMPEQ(norm_a[2], zero[2]);
    acc &= VCMPEQ(norm_a[3], zero[3]);
    acc &= VCMPEQ(norm_a[4], zero[4]);
    acc &= VCMPEQ(norm_a[5], zero[5]);
    acc &= VCMPEQ(norm_a[6], zero[6]);
    acc &= VCMPEQ(norm_a[7], zero[7]);
    acc &= VCMPEQ(norm_a[8], zero[8]);
    acc &= VCMPEQ(norm_a[9], zero[9]);
    acc &= VCMPEQ(norm_a[10], zero[10]);
    return acc;
}

void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[NUM_LIMBS], sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSETALL(1);
    __m512i p_top = VSETALL(P_PLUS_1_TOP);
    
    // Add limbs
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    t[3] = VADD(a[3], b[3]);
    t[4] = VADD(a[4], b[4]);
    t[5] = VADD(a[5], b[5]);
    t[6] = VADD(a[6], b[6]);
    t[7] = VADD(a[7], b[7]);
    t[8] = VADD(a[8], b[8]);
    t[9] = VADD(a[9], b[9]);
    t[10] = VADD(a[10], b[10]);
    
    // Subtract p, using special shape of p
    t[0] = VADD(t[0], p_bot);
    t[10] = VSUB(t[10], p_top);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t[10], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[10] = VADD(t[10], VAND(p_top, sign_mask));
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
}

void fp_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[NUM_LIMBS], sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSETALL(1);
    __m512i p_top = VSETALL(P_PLUS_1_TOP);
    
    // Subtract limbs
    t[0] = VSUB(a[0], b[0]);
    t[1] = VSUB(a[1], b[1]);
    t[2] = VSUB(a[2], b[2]);
    t[3] = VSUB(a[3], b[3]);
    t[4] = VSUB(a[4], b[4]);
    t[5] = VSUB(a[5], b[5]);
    t[6] = VSUB(a[6], b[6]);
    t[7] = VSUB(a[7], b[7]);
    t[8] = VSUB(a[8], b[8]);
    t[9] = VSUB(a[9], b[9]);
    t[10] = VSUB(a[10], b[10]);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t[10], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[10] = VADD(t[10], VAND(p_top, sign_mask));
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
}

void fp_neg_8x1(__m512i* res, __m512i* a)
{
    __m512i t[NUM_LIMBS], sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSETALL(1);
    __m512i p_top = VSETALL(P_PLUS_1_TOP);
    __m512i zero = VZERO();
    
    // 0 - a
    t[0] = VSUB(zero, a[0]);
    t[1] = VSUB(zero, a[1]);
    t[2] = VSUB(zero, a[2]);
    t[3] = VSUB(zero, a[3]);
    t[4] = VSUB(zero, a[4]);
    t[5] = VSUB(zero, a[5]);
    t[6] = VSUB(zero, a[6]);
    t[7] = VSUB(zero, a[7]);
    t[8] = VSUB(zero, a[8]);
    t[9] = VSUB(zero, a[9]);
    t[10] = VSUB(zero, a[10]);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t[10], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[10] = VADD(t[10], VAND(p_top, sign_mask));
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
}

void fp_mixaddsub_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[NUM_LIMBS], sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSETALL(1);
    __m512i p_top = VSETALL(P_PLUS_1_TOP);
    const __mmask8 sub_mask = ~add_mask;
    
    t[0] = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t[1] = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t[2] = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    t[3] = VSUBMASK(a[3], sub_mask, a[3], b[3]);
    t[4] = VSUBMASK(a[4], sub_mask, a[4], b[4]);
    t[5] = VSUBMASK(a[5], sub_mask, a[5], b[5]);
    t[6] = VSUBMASK(a[6], sub_mask, a[6], b[6]);
    t[7] = VSUBMASK(a[7], sub_mask, a[7], b[7]);
    t[8] = VSUBMASK(a[8], sub_mask, a[8], b[8]);
    t[9] = VSUBMASK(a[9], sub_mask, a[9], b[9]);
    t[10] = VSUBMASK(a[10], sub_mask, a[10], b[10]);
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    t[2] = VADDMASK(t[2], add_mask, a[2], b[2]);
    t[3] = VADDMASK(t[3], add_mask, a[3], b[3]);
    t[4] = VADDMASK(t[4], add_mask, a[4], b[4]);
    t[5] = VADDMASK(t[5], add_mask, a[5], b[5]);
    t[6] = VADDMASK(t[6], add_mask, a[6], b[6]);
    t[7] = VADDMASK(t[7], add_mask, a[7], b[7]);
    t[8] = VADDMASK(t[8], add_mask, a[8], b[8]);
    t[9] = VADDMASK(t[9], add_mask, a[9], b[9]);
    t[10] = VADDMASK(t[10], add_mask, a[10], b[10]);
    
    // Subtract p, using special shape of p
    t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);
    t[10] = VSUBMASK(t[10], add_mask, t[10], p_top);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t[10], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[10] = VADD(t[10], VAND(p_top, sign_mask));
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
}

// Single-length multiprecision addition, no correction, no carry propagation
void mp_add_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[NUM_LIMBS];
    
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    t[3] = VADD(a[3], b[3]);
    t[4] = VADD(a[4], b[4]);
    t[5] = VADD(a[5], b[5]);
    t[6] = VADD(a[6], b[6]);
    t[7] = VADD(a[7], b[7]);
    t[8] = VADD(a[8], b[8]);
    t[9] = VADD(a[9], b[9]);
    t[10] = VADD(a[10], b[10]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
}

// Single-length multiprecision addition, no correction, no carry propagation
void mp_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[NUM_LIMBS];
    
    t[0] = VSUB(a[0], b[0]);
    t[1] = VSUB(a[1], b[1]);
    t[2] = VSUB(a[2], b[2]);
    t[3] = VSUB(a[3], b[3]);
    t[4] = VSUB(a[4], b[4]);
    t[5] = VSUB(a[5], b[5]);
    t[6] = VSUB(a[6], b[6]);
    t[7] = VSUB(a[7], b[7]);
    t[8] = VSUB(a[8], b[8]);
    t[9] = VSUB(a[9], b[9]);
    t[10] = VSUB(a[10], b[10]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
}

// Double-length multiprecision subtraction res = res - a - b, no correction
void mp_dblsubx2_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[2*NUM_LIMBS];
    __m512i limb_mask = VSETALL(LIMB_MASK);
    
    t[0] = VSUB(res[0], a[0]);
    t[1] = VSUB(res[1], a[1]);
    t[2] = VSUB(res[2], a[2]);
    t[3] = VSUB(res[3], a[3]);
    t[4] = VSUB(res[4], a[4]);
    t[5] = VSUB(res[5], a[5]);
    t[6] = VSUB(res[6], a[6]);
    t[7] = VSUB(res[7], a[7]);
    t[8] = VSUB(res[8], a[8]);
    t[9] = VSUB(res[9], a[9]);
    t[10] = VSUB(res[10], a[10]);
    t[11] = VSUB(res[11], a[11]);
    t[12] = VSUB(res[12], a[12]);
    t[13] = VSUB(res[13], a[13]);
    t[14] = VSUB(res[14], a[14]);
    t[15] = VSUB(res[15], a[15]);
    t[16] = VSUB(res[16], a[16]);
    t[17] = VSUB(res[17], a[17]);
    t[18] = VSUB(res[18], a[18]);
    t[19] = VSUB(res[19], a[19]);
    t[20] = VSUB(res[20], a[20]);
    t[21] = VSUB(res[21], a[21]);
    
    t[0] = VSUB(t[0], b[0]);
    t[1] = VSUB(t[1], b[1]);
    t[2] = VSUB(t[2], b[2]);
    t[3] = VSUB(t[3], b[3]);
    t[4] = VSUB(t[4], b[4]);
    t[5] = VSUB(t[5], b[5]);
    t[6] = VSUB(t[6], b[6]);
    t[7] = VSUB(t[7], b[7]);
    t[8] = VSUB(t[8], b[8]);
    t[9] = VSUB(t[9], b[9]);
    t[10] = VSUB(t[10], b[10]);
    t[11] = VSUB(t[11], b[11]);
    t[12] = VSUB(t[12], b[12]);
    t[13] = VSUB(t[13], b[13]);
    t[14] = VSUB(t[14], b[14]);
    t[15] = VSUB(t[15], b[15]);
    t[16] = VSUB(t[16], b[16]);
    t[17] = VSUB(t[17], b[17]);
    t[18] = VSUB(t[18], b[18]);
    t[19] = VSUB(t[19], b[19]);
    t[20] = VSUB(t[20], b[20]);
    t[21] = VSUB(t[21], b[21]);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[11] = VADD(t[11], VSRA(t[10], LIMB_LEN));
    t[12] = VADD(t[12], VSRA(t[11], LIMB_LEN));
    t[13] = VADD(t[13], VSRA(t[12], LIMB_LEN));
    t[14] = VADD(t[14], VSRA(t[13], LIMB_LEN));
    t[15] = VADD(t[15], VSRA(t[14], LIMB_LEN));
    t[16] = VADD(t[16], VSRA(t[15], LIMB_LEN));
    t[17] = VADD(t[17], VSRA(t[16], LIMB_LEN));
    t[18] = VADD(t[18], VSRA(t[17], LIMB_LEN));
    t[19] = VADD(t[19], VSRA(t[18], LIMB_LEN));
    t[20] = VADD(t[20], VSRA(t[19], LIMB_LEN));
    t[21] = VADD(t[21], VSRA(t[20], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    t[10] = VAND(t[10], limb_mask);
    t[11] = VAND(t[11], limb_mask);
    t[12] = VAND(t[12], limb_mask);
    t[13] = VAND(t[13], limb_mask);
    t[14] = VAND(t[14], limb_mask);
    t[15] = VAND(t[15], limb_mask);
    t[16] = VAND(t[16], limb_mask);
    t[17] = VAND(t[17], limb_mask);
    t[18] = VAND(t[18], limb_mask);
    t[19] = VAND(t[19], limb_mask);
    t[20] = VAND(t[20], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
    res[11] = t[11];
    res[12] = t[12];
    res[13] = t[13];
    res[14] = t[14];
    res[15] = t[15];
    res[16] = t[16];
    res[17] = t[17];
    res[18] = t[18];
    res[19] = t[19];
    res[20] = t[20];
    res[21] = t[21];
}

// Double-length multiprecision subtraction res = res - a
void mp_subx2_8x1(__m512i* res, __m512i* a)
{
    __m512i t[2*NUM_LIMBS];
    __m512i limb_mask = VSETALL(LIMB_MASK);
    
    t[0] = VSUB(res[0], a[0]);
    t[1] = VSUB(res[1], a[1]);
    t[2] = VSUB(res[2], a[2]);
    t[3] = VSUB(res[3], a[3]);
    t[4] = VSUB(res[4], a[4]);
    t[5] = VSUB(res[5], a[5]);
    t[6] = VSUB(res[6], a[6]);
    t[7] = VSUB(res[7], a[7]);
    t[8] = VSUB(res[8], a[8]);
    t[9] = VSUB(res[9], a[9]);
    t[10] = VSUB(res[10], a[10]);
    t[11] = VSUB(res[11], a[11]);
    t[12] = VSUB(res[12], a[12]);
    t[13] = VSUB(res[13], a[13]);
    t[14] = VSUB(res[14], a[14]);
    t[15] = VSUB(res[15], a[15]);
    t[16] = VSUB(res[16], a[16]);
    t[17] = VSUB(res[17], a[17]);
    t[18] = VSUB(res[18], a[18]);
    t[19] = VSUB(res[19], a[19]);
    t[20] = VSUB(res[20], a[20]);
    t[21] = VSUB(res[21], a[21]);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[11] = VADD(t[11], VSRA(t[10], LIMB_LEN));
    t[12] = VADD(t[12], VSRA(t[11], LIMB_LEN));
    t[13] = VADD(t[13], VSRA(t[12], LIMB_LEN));
    t[14] = VADD(t[14], VSRA(t[13], LIMB_LEN));
    t[15] = VADD(t[15], VSRA(t[14], LIMB_LEN));
    t[16] = VADD(t[16], VSRA(t[15], LIMB_LEN));
    t[17] = VADD(t[17], VSRA(t[16], LIMB_LEN));
    t[18] = VADD(t[18], VSRA(t[17], LIMB_LEN));
    t[19] = VADD(t[19], VSRA(t[18], LIMB_LEN));
    t[20] = VADD(t[20], VSRA(t[19], LIMB_LEN));
    t[21] = VADD(t[21], VSRA(t[20], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    t[10] = VAND(t[10], limb_mask);
    t[11] = VAND(t[11], limb_mask);
    t[12] = VAND(t[12], limb_mask);
    t[13] = VAND(t[13], limb_mask);
    t[14] = VAND(t[14], limb_mask);
    t[15] = VAND(t[15], limb_mask);
    t[16] = VAND(t[16], limb_mask);
    t[17] = VAND(t[17], limb_mask);
    t[18] = VAND(t[18], limb_mask);
    t[19] = VAND(t[19], limb_mask);
    t[20] = VAND(t[20], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
    res[11] = t[11];
    res[12] = t[12];
    res[13] = t[13];
    res[14] = t[14];
    res[15] = t[15];
    res[16] = t[16];
    res[17] = t[17];
    res[18] = t[18];
    res[19] = t[19];
    res[20] = t[20];
    res[21] = t[21];
}

void mp_mixaddsubx2_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[2*NUM_LIMBS];
    __m512i limb_mask = VSETALL(LIMB_MASK);
    const __mmask8 sub_mask = ~add_mask;
    
    t[0] = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t[1] = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t[2] = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    t[3] = VSUBMASK(a[3], sub_mask, a[3], b[3]);
    t[4] = VSUBMASK(a[4], sub_mask, a[4], b[4]);
    t[5] = VSUBMASK(a[5], sub_mask, a[5], b[5]);
    t[6] = VSUBMASK(a[6], sub_mask, a[6], b[6]);
    t[7] = VSUBMASK(a[7], sub_mask, a[7], b[7]);
    t[8] = VSUBMASK(a[8], sub_mask, a[8], b[8]);
    t[9] = VSUBMASK(a[9], sub_mask, a[9], b[9]);
    t[10] = VSUBMASK(a[10], sub_mask, a[10], b[10]);
    t[11] = VSUBMASK(a[11], sub_mask, a[11], b[11]);
    t[12] = VSUBMASK(a[12], sub_mask, a[12], b[12]);
    t[13] = VSUBMASK(a[13], sub_mask, a[13], b[13]);
    t[14] = VSUBMASK(a[14], sub_mask, a[14], b[14]);
    t[15] = VSUBMASK(a[15], sub_mask, a[15], b[15]);
    t[16] = VSUBMASK(a[16], sub_mask, a[16], b[16]);
    t[17] = VSUBMASK(a[17], sub_mask, a[17], b[17]);
    t[18] = VSUBMASK(a[18], sub_mask, a[18], b[18]);
    t[19] = VSUBMASK(a[19], sub_mask, a[19], b[19]);
    t[20] = VSUBMASK(a[20], sub_mask, a[20], b[20]);
    t[21] = VSUBMASK(a[21], sub_mask, a[21], b[21]);
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    t[2] = VADDMASK(t[2], add_mask, a[2], b[2]);
    t[3] = VADDMASK(t[3], add_mask, a[3], b[3]);
    t[4] = VADDMASK(t[4], add_mask, a[4], b[4]);
    t[5] = VADDMASK(t[5], add_mask, a[5], b[5]);
    t[6] = VADDMASK(t[6], add_mask, a[6], b[6]);
    t[7] = VADDMASK(t[7], add_mask, a[7], b[7]);
    t[8] = VADDMASK(t[8], add_mask, a[8], b[8]);
    t[9] = VADDMASK(t[9], add_mask, a[9], b[9]);
    t[10] = VADDMASK(t[10], add_mask, a[10], b[10]);
    t[11] = VADDMASK(t[11], add_mask, a[11], b[11]);
    t[12] = VADDMASK(t[12], add_mask, a[12], b[12]);
    t[13] = VADDMASK(t[13], add_mask, a[13], b[13]);
    t[14] = VADDMASK(t[14], add_mask, a[14], b[14]);
    t[15] = VADDMASK(t[15], add_mask, a[15], b[15]);
    t[16] = VADDMASK(t[16], add_mask, a[16], b[16]);
    t[17] = VADDMASK(t[17], add_mask, a[17], b[17]);
    t[18] = VADDMASK(t[18], add_mask, a[18], b[18]);
    t[19] = VADDMASK(t[19], add_mask, a[19], b[19]);
    t[20] = VADDMASK(t[20], add_mask, a[20], b[20]);
    t[21] = VADDMASK(t[21], add_mask, a[21], b[21]);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[8] = VADD(t[8], VSRA(t[7], LIMB_LEN));
    t[9] = VADD(t[9], VSRA(t[8], LIMB_LEN));
    t[10] = VADD(t[10], VSRA(t[9], LIMB_LEN));
    t[11] = VADD(t[11], VSRA(t[10], LIMB_LEN));
    t[12] = VADD(t[12], VSRA(t[11], LIMB_LEN));
    t[13] = VADD(t[13], VSRA(t[12], LIMB_LEN));
    t[14] = VADD(t[14], VSRA(t[13], LIMB_LEN));
    t[15] = VADD(t[15], VSRA(t[14], LIMB_LEN));
    t[16] = VADD(t[16], VSRA(t[15], LIMB_LEN));
    t[17] = VADD(t[17], VSRA(t[16], LIMB_LEN));
    t[18] = VADD(t[18], VSRA(t[17], LIMB_LEN));
    t[19] = VADD(t[19], VSRA(t[18], LIMB_LEN));
    t[20] = VADD(t[20], VSRA(t[19], LIMB_LEN));
    t[21] = VADD(t[21], VSRA(t[20], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    t[9] = VAND(t[9], limb_mask);
    t[10] = VAND(t[10], limb_mask);
    t[11] = VAND(t[11], limb_mask);
    t[12] = VAND(t[12], limb_mask);
    t[13] = VAND(t[13], limb_mask);
    t[14] = VAND(t[14], limb_mask);
    t[15] = VAND(t[15], limb_mask);
    t[16] = VAND(t[16], limb_mask);
    t[17] = VAND(t[17], limb_mask);
    t[18] = VAND(t[18], limb_mask);
    t[19] = VAND(t[19], limb_mask);
    t[20] = VAND(t[20], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
    res[11] = t[11];
    res[12] = t[12];
    res[13] = t[13];
    res[14] = t[14];
    res[15] = t[15];
    res[16] = t[16];
    res[17] = t[17];
    res[18] = t[18];
    res[19] = t[19];
    res[20] = t[20];
    res[21] = t[21];
}

// Computes res = a + p - b without any carry propagation or conditional masks
void mp_add_p_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[NUM_LIMBS];
    __m512i p_bot = VSETALL(LIMB_MASK);
    __m512i p_top = VSETALL(P_PLUS_1_TOP-1);
    
    // Add p
    t[0] = VADD(a[0], p_bot);
    t[1] = VADD(a[1], p_bot);
    t[2] = VADD(a[2], p_bot);
    t[3] = VADD(a[3], p_bot);
    t[4] = VADD(a[4], p_bot);
    t[5] = VADD(a[5], p_bot);
    t[6] = VADD(a[6], p_bot);
    t[7] = VADD(a[7], p_bot);
    t[8] = VADD(a[8], p_bot);
    t[9] = VADD(a[9], p_bot);
    t[10] = VADD(a[10], p_top);
    
    t[0] = VSUB(t[0], b[0]);
    t[1] = VSUB(t[1], b[1]);
    t[2] = VSUB(t[2], b[2]);
    t[3] = VSUB(t[3], b[3]);
    t[4] = VSUB(t[4], b[4]);
    t[5] = VSUB(t[5], b[5]);
    t[6] = VSUB(t[6], b[6]);
    t[7] = VSUB(t[7], b[7]);
    t[8] = VSUB(t[8], b[8]);
    t[9] = VSUB(t[9], b[9]);
    t[10] = VSUB(t[10], b[10]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
}

void mp_dbl_8x1(__m512i* res, __m512i* a)
{
    __m512i t[NUM_LIMBS];
    
    t[0] = VSL(a[0], 1);
    t[1] = VSL(a[1], 1);
    t[2] = VSL(a[2], 1);
    t[3] = VSL(a[3], 1);
    t[4] = VSL(a[4], 1);
    t[5] = VSL(a[5], 1);
    t[6] = VSL(a[6], 1);
    t[7] = VSL(a[7], 1);
    t[8] = VSL(a[8], 1);
    t[9] = VSL(a[9], 1);
    t[10] = VSL(a[10], 1);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
}

void mp_mul_8x1_karat(__m512i* res, __m512i* a, __m512i* b)
{
    const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
    const __m512i a6 = a[6], a7 = a[7], a8 = a[8], a9 = a[9], a10 = a[10];
    const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5];
    const __m512i b6 = b[6], b7 = b[7], b8 = b[8], b9 = b[9], b10 = b[10];
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    __m512i t16, t17, t18, t19, t20, t21;
    __m512i ta0, ta1, ta2, ta3, ta4, ta5;
    __m512i tb0, tb1, tb2, tb3, tb4, tb5;
    __m512i m0, m1, m2, m3, m4, m5, m6, m7, m8, m9, m10, m11;
    const __m512i zero = VZERO();
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    // a_low * b_low
    t0 = VMULADDL(zero, a0, b0);
    t1 = VMULADDH(zero, a0, b0);
    t1 = VSL(t1, LIMB_ALIGN);
    
    t1 = VMULADDL(t1, a0, b1);
    t2 = VMULADDH(zero, a0, b1);
    t1 = VMULADDL(t1, a1, b0);
    t2 = VMULADDH(t2, a1, b0);
    t2 = VSL(t2, LIMB_ALIGN);
    
    t2 = VMULADDL(t2, a0, b2);
    t3 = VMULADDH(zero, a0, b2);
    t2 = VMULADDL(t2, a1, b1);
    t3 = VMULADDH(t3, a1, b1);
    t2 = VMULADDL(t2, a2, b0);
    t3 = VMULADDH(t3, a2, b0);
    t3 = VSL(t3, LIMB_ALIGN);
    
    t3 = VMULADDL(t3, a0, b3);
    t4 = VMULADDH(zero, a0, b3);
    t3 = VMULADDL(t3, a1, b2);
    t4 = VMULADDH(t4, a1, b2);
    t3 = VMULADDL(t3, a2, b1);
    t4 = VMULADDH(t4, a2, b1);
    t3 = VMULADDL(t3, a3, b0);
    t4 = VMULADDH(t4, a3, b0);
    t4 = VSL(t4, LIMB_ALIGN);
    
    t4 = VMULADDL(t4, a0, b4);
    t5 = VMULADDH(zero, a0, b4);
    t4 = VMULADDL(t4, a1, b3);
    t5 = VMULADDH(t5, a1, b3);
    t4 = VMULADDL(t4, a2, b2);
    t5 = VMULADDH(t5, a2, b2);
    t4 = VMULADDL(t4, a3, b1);
    t5 = VMULADDH(t5, a3, b1);
    t4 = VMULADDL(t4, a4, b0);
    t5 = VMULADDH(t5, a4, b0);
    t5 = VSL(t5, LIMB_ALIGN);
    
    t5 = VMULADDL(t5, a0, b5);
    t6 = VMULADDH(zero, a0, b5);
    t5 = VMULADDL(t5, a1, b4);
    t6 = VMULADDH(t6, a1, b4);
    t5 = VMULADDL(t5, a2, b3);
    t6 = VMULADDH(t6, a2, b3);
    t5 = VMULADDL(t5, a3, b2);
    t6 = VMULADDH(t6, a3, b2);
    t5 = VMULADDL(t5, a4, b1);
    t6 = VMULADDH(t6, a4, b1);
    t5 = VMULADDL(t5, a5, b0);
    t6 = VMULADDH(t6, a5, b0);
    t6 = VSL(t6, LIMB_ALIGN);
    
    t6 = VMULADDL(t6, a1, b5);
    t7 = VMULADDH(zero, a1, b5);
    t6 = VMULADDL(t6, a2, b4);
    t7 = VMULADDH(t7, a2, b4);
    t6 = VMULADDL(t6, a3, b3);
    t7 = VMULADDH(t7, a3, b3);
    t6 = VMULADDL(t6, a4, b2);
    t7 = VMULADDH(t7, a4, b2);
    t6 = VMULADDL(t6, a5, b1);
    t7 = VMULADDH(t7, a5, b1);
    t7 = VSL(t7, LIMB_ALIGN);
    
    t7 = VMULADDL(t7, a2, b5);
    t8 = VMULADDH(zero, a2, b5);
    t7 = VMULADDL(t7, a3, b4);
    t8 = VMULADDH(t8, a3, b4);
    t7 = VMULADDL(t7, a4, b3);
    t8 = VMULADDH(t8, a4, b3);
    t7 = VMULADDL(t7, a5, b2);
    t8 = VMULADDH(t8, a5, b2);
    t8 = VSL(t8, LIMB_ALIGN);
    
    t8 = VMULADDL(t8, a3, b5);
    t9 = VMULADDH(zero, a3, b5);
    t8 = VMULADDL(t8, a4, b4);
    t9 = VMULADDH(t9, a4, b4);
    t8 = VMULADDL(t8, a5, b3);
    t9 = VMULADDH(t9, a5, b3);
    t9 = VSL(t9, LIMB_ALIGN);
    
    t9 = VMULADDL(t9, a4, b5);
    t10 = VMULADDH(zero, a4, b5);
    t9 = VMULADDL(t9, a5, b4);
    t10 = VMULADDH(t10, a5, b4);
    t10 = VSL(t10, LIMB_ALIGN);
    
    t10 = VMULADDL(t10, a5, b5);
    t11 = VMULADDH(zero, a5, b5);
    t11 = VSL(t11, LIMB_ALIGN);
    
    // a_hi * b_hi
    t12 = VMULADDL(zero, a6, b6);
    t13 = VMULADDH(zero, a6, b6);
    t13 = VSL(t13, LIMB_ALIGN);
    
    t13 = VMULADDL(t13, a6, b7);
    t14 = VMULADDH(zero, a6, b7);
    t13 = VMULADDL(t13, a7, b6);
    t14 = VMULADDH(t14, a7, b6);
    t14 = VSL(t14, LIMB_ALIGN);
    
    t14 = VMULADDL(t14, a6, b8);
    t15 = VMULADDH(zero, a6, b8);
    t14 = VMULADDL(t14, a7, b7);
    t15 = VMULADDH(t15, a7, b7);
    t14 = VMULADDL(t14, a8, b6);
    t15 = VMULADDH(t15, a8, b6);
    t15 = VSL(t15, LIMB_ALIGN);
    
    t15 = VMULADDL(t15, a6, b9);
    t16 = VMULADDH(zero, a6, b9);
    t15 = VMULADDL(t15, a7, b8);
    t16 = VMULADDH(t16, a7, b8);
    t15 = VMULADDL(t15, a8, b7);
    t16 = VMULADDH(t16, a8, b7);
    t15 = VMULADDL(t15, a9, b6);
    t16 = VMULADDH(t16, a9, b6);
    t16 = VSL(t16, LIMB_ALIGN);
    
    t16 = VMULADDL(t16, a6, b10);
    t17 = VMULADDH(zero, a6, b10);
    t16 = VMULADDL(t16, a7, b9);
    t17 = VMULADDH(t17, a7, b9);
    t16 = VMULADDL(t16, a8, b8);
    t17 = VMULADDH(t17, a8, b8);
    t16 = VMULADDL(t16, a9, b7);
    t17 = VMULADDH(t17, a9, b7);
    t16 = VMULADDL(t16, a10, b6);
    t17 = VMULADDH(t17, a10, b6);
    t17 = VSL(t17, LIMB_ALIGN);
    
    t17 = VMULADDL(t17, a7, b10);
    t18 = VMULADDH(zero, a7, b10);
    t17 = VMULADDL(t17, a8, b9);
    t18 = VMULADDH(t18, a8, b9);
    t17 = VMULADDL(t17, a9, b8);
    t18 = VMULADDH(t18, a9, b8);
    t17 = VMULADDL(t17, a10, b7);
    t18 = VMULADDH(t18, a10, b7);
    t18 = VSL(t18, LIMB_ALIGN);
    
    t18 = VMULADDL(t18, a8, b10);
    t19 = VMULADDH(zero, a8, b10);
    t18 = VMULADDL(t18, a9, b9);
    t19 = VMULADDH(t19, a9, b9);
    t18 = VMULADDL(t18, a10, b8);
    t19 = VMULADDH(t19, a10, b8);
    t19 = VSL(t19, LIMB_ALIGN);
    
    t19 = VMULADDL(t19, a9, b10);
    t20 = VMULADDH(zero, a9, b10);
    t19 = VMULADDL(t19, a10, b9);
    t20 = VMULADDH(t20, a10, b9);
    t20 = VSL(t20, LIMB_ALIGN);
    
    t20 = VMULADDL(t20, a10, b10);
    t21 = VMULADDH(zero, a10, b10);
    t21 = VSL(t21, LIMB_ALIGN);
    
    // ta = a_low + a_hi
    ta0 = VADD(a0, a6);
    ta1 = VADD(a1, a7);
    ta2 = VADD(a2, a8);
    ta3 = VADD(a3, a9);
    ta4 = VADD(a4, a10);
    ta5 = a5;
    
    // tb = b_low + b_hi
    tb0 = VADD(b0, b6);
    tb1 = VADD(b1, b7);
    tb2 = VADD(b2, b8);
    tb3 = VADD(b3, b9);
    tb4 = VADD(b4, b10);
    tb5 = b5;
    
    // m = ta * tb
    m0 = VMULADDL(zero, ta0, tb0);
    m1 = VMULADDH(zero, ta0, tb0);
    m1 = VSL(m1, LIMB_ALIGN);
    
    m1 = VMULADDL(m1, ta0, tb1);
    m2 = VMULADDH(zero, ta0, tb1);
    m1 = VMULADDL(m1, ta1, tb0);
    m2 = VMULADDH(m2, ta1, tb0);
    m2 = VSL(m2, LIMB_ALIGN);
    
    m2 = VMULADDL(m2, ta0, tb2);
    m3 = VMULADDH(zero, ta0, tb2);
    m2 = VMULADDL(m2, ta1, tb1);
    m3 = VMULADDH(m3, ta1, tb1);
    m2 = VMULADDL(m2, ta2, tb0);
    m3 = VMULADDH(m3, ta2, tb0);
    m3 = VSL(m3, LIMB_ALIGN);
    
    m3 = VMULADDL(m3, ta0, tb3);
    m4 = VMULADDH(zero, ta0, tb3);
    m3 = VMULADDL(m3, ta1, tb2);
    m4 = VMULADDH(m4, ta1, tb2);
    m3 = VMULADDL(m3, ta2, tb1);
    m4 = VMULADDH(m4, ta2, tb1);
    m3 = VMULADDL(m3, ta3, tb0);
    m4 = VMULADDH(m4, ta3, tb0);
    m4 = VSL(m4, LIMB_ALIGN);
    
    m4 = VMULADDL(m4, ta0, tb4);
    m5 = VMULADDH(zero, ta0, tb4);
    m4 = VMULADDL(m4, ta1, tb3);
    m5 = VMULADDH(m5, ta1, tb3);
    m4 = VMULADDL(m4, ta2, tb2);
    m5 = VMULADDH(m5, ta2, tb2);
    m4 = VMULADDL(m4, ta3, tb1);
    m5 = VMULADDH(m5, ta3, tb1);
    m4 = VMULADDL(m4, ta4, tb0);
    m5 = VMULADDH(m5, ta4, tb0);
    m5 = VSL(m5, LIMB_ALIGN);
    
    m5 = VMULADDL(m5, ta0, tb5);
    m6 = VMULADDH(zero, ta0, tb5);
    m5 = VMULADDL(m5, ta1, tb4);
    m6 = VMULADDH(m6, ta1, tb4);
    m5 = VMULADDL(m5, ta2, tb3);
    m6 = VMULADDH(m6, ta2, tb3);
    m5 = VMULADDL(m5, ta3, tb2);
    m6 = VMULADDH(m6, ta3, tb2);
    m5 = VMULADDL(m5, ta4, tb1);
    m6 = VMULADDH(m6, ta4, tb1);
    m5 = VMULADDL(m5, ta5, tb0);
    m6 = VMULADDH(m6, ta5, tb0);
    m6 = VSL(m6, LIMB_ALIGN);
    
    m6 = VMULADDL(m6, ta1, tb5);
    m7 = VMULADDH(zero, ta1, tb5);
    m6 = VMULADDL(m6, ta2, tb4);
    m7 = VMULADDH(m7, ta2, tb4);
    m6 = VMULADDL(m6, ta3, tb3);
    m7 = VMULADDH(m7, ta3, tb3);
    m6 = VMULADDL(m6, ta4, tb2);
    m7 = VMULADDH(m7, ta4, tb2);
    m6 = VMULADDL(m6, ta5, tb1);
    m7 = VMULADDH(m7, ta5, tb1);
    m7 = VSL(m7, LIMB_ALIGN);
    
    m7 = VMULADDL(m7, ta2, tb5);
    m8 = VMULADDH(zero, ta2, tb5);
    m7 = VMULADDL(m7, ta3, tb4);
    m8 = VMULADDH(m8, ta3, tb4);
    m7 = VMULADDL(m7, ta4, tb3);
    m8 = VMULADDH(m8, ta4, tb3);
    m7 = VMULADDL(m7, ta5, tb2);
    m8 = VMULADDH(m8, ta5, tb2);
    m8 = VSL(m8, LIMB_ALIGN);
    
    m8 = VMULADDL(m8, ta3, tb5);
    m9 = VMULADDH(zero, ta3, tb5);
    m8 = VMULADDL(m8, ta4, tb4);
    m9 = VMULADDH(m9, ta4, tb4);
    m8 = VMULADDL(m8, ta5, tb3);
    m9 = VMULADDH(m9, ta5, tb3);
    m9 = VSL(m9, LIMB_ALIGN);
    
    m9 = VMULADDL(m9, ta4, tb5);
    m10 = VMULADDH(zero, ta4, tb5);
    m9 = VMULADDL(m9, ta5, tb4);
    m10 = VMULADDH(m10, ta5, tb4);
    m10 = VSL(m10, LIMB_ALIGN);
    
    m10 = VMULADDL(m10, ta5, tb5);
    m11 = VMULADDH(zero, ta5, tb5);
    m11 = VSL(m11, LIMB_ALIGN);
    
    // m = (ta * tb) - (a_low * b_low) - (a_hi * b_hi)
    m0 = VSUB(m0, VADD(t0, t12));
    m1 = VSUB(m1, VADD(t1, t13));
    m2 = VSUB(m2, VADD(t2, t14));
    m3 = VSUB(m3, VADD(t3, t15));
    m4 = VSUB(m4, VADD(t4, t16));
    m5 = VSUB(m5, VADD(t5, t17));
    m6 = VSUB(m6, VADD(t6, t18));
    m7 = VSUB(m7, VADD(t7, t19));
    m8 = VSUB(m8, VADD(t8, t20));
    m9 = VSUB(m9, VADD(t9, t21));
    m10 = VSUB(m10, t10);
    m11 = VSUB(m11, t11);
    
    // t += m * 2^(limbs_low)
    t6 = VADD(t6, m0);
    t7 = VADD(t7, m1);
    t8 = VADD(t8, m2);
    t9 = VADD(t9, m3);
    t10 = VADD(t10, m4);
    t11 = VADD(t11, m5);
    t12 = VADD(t12, m6);
    t13 = VADD(t13, m7);
    t14 = VADD(t14, m8);
    t15 = VADD(t15, m9);
    t16 = VADD(t16, m10);
    t17 = VADD(t17, m11);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7; res[8] = t8; res[9] = t9; res[10] = t10; res[11] = t11;
    res[12] = t12; res[13] = t13; res[14] = t14; res[15] = t15; res[16] = t16; res[17] = t17;
    res[18] = t18; res[19] = t19; res[20] = t20; res[21] = t21;
}

void rdc_8x1(__m512i* res, __m512i* a)
{
    // Montgomery reduction with R = 2^n mod p, with n = ((num_limbs + 1) * limb_len)
    // low(a) = (a_0, a_1, ... , a_num_limbs)
    // m = -p^(-1) mod 2^n = p + 2 = c * 2^e + 1
    // The output of this reduction is (a + (low(a) * m mod 2^n )*p) / 2^n
    
    const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
    const __m512i a6 = a[6], a7 = a[7], a8 = a[8], a9 = a[9], a10 = a[10], a11 = a[11];
    const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15], a16 = a[16], a17 = a[17];
    const __m512i a18 = a[18], a19 = a[19], a20 = a[20], a21 = a[21];
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12;
    __m512i s10, s11, hi;
    __mmask8 msb_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i top = VSETALL(P_PLUS_1_TOP);
    const __m512i zero = VZERO();
    const __m512i one = VSETALL(1);
    
    // s = low(a) * (2^521 + 1) mod 2^((11+1)*48)
    // s0 = a0, ..., s9 = a9
    s10 = VADD(a10, VAND(VSL(a0, 41), limb_mask));
    s11 = VADD(a11, VSRA(a0, 7));
    s11 = VADD(s11, VAND(VSL(a1, 41), limb_mask));
    s11 = VADD(s11, VSRA(s10, 48));
    s10 = VAND(s10, limb_mask);
    s11 = VAND(s11, limb_mask);
    
    // Computing a + s*(p+1) - s = a + s*2^521 - s
    // As the division by 2^n cancels the lower limbs, we don't need to compute - s.
    t0 = VAND(VSL(a0, 41), limb_mask);
    hi = VSRA(a0, 7);
    
    t1 = VADD(hi, VAND(VSL(a1, 41), limb_mask));
    hi = VSRA(a1, 7);
    hi = VADD(hi, VSRA(t1, 48));
    t1 = VAND(t1, limb_mask);
    
    t2 = VADD(hi, VAND(VSL(a2, 41), limb_mask));
    hi = VSRA(a2, 7);
    hi = VADD(hi, VSRA(t2, 48));
    t2 = VAND(t2, limb_mask);
    
    t3 = VADD(hi, VAND(VSL(a3, 41), limb_mask));
    hi = VSRA(a3, 7);
    hi = VADD(hi, VSRA(t3, 48));
    t3 = VAND(t3, limb_mask);
    
    t4 = VADD(hi, VAND(VSL(a4, 41), limb_mask));
    hi = VSRA(a4, 7);
    hi = VADD(hi, VSRA(t4, 48));
    t4 = VAND(t4, limb_mask);
    
    t5 = VADD(hi, VAND(VSL(a5, 41), limb_mask));
    hi = VSRA(a5, 7);
    hi = VADD(hi, VSRA(t5, 48));
    t5 = VAND(t5, limb_mask);
    
    t6 = VADD(hi, VAND(VSL(a6, 41), limb_mask));
    hi = VSRA(a6, 7);
    hi = VADD(hi, VSRA(t6, 48));
    t6 = VAND(t6, limb_mask);
    
    t7 = VADD(hi, VAND(VSL(a7, 41), limb_mask));
    hi = VSRA(a7, 7);
    hi = VADD(hi, VSRA(t7, 48));
    t7 = VAND(t7, limb_mask);
    
    t8 = VADD(hi, VAND(VSL(a8, 41), limb_mask));
    hi = VSRA(a8, 7);
    hi = VADD(hi, VSRA(t8, 48));
    t8 = VAND(t8, limb_mask);
    
    t9 = VADD(hi, VAND(VSL(a9, 41), limb_mask));
    hi = VSRA(a9, 7);
    hi = VADD(hi, VSRA(t9, 48));
    t9 = VAND(t9, limb_mask);
    
    t10 = VADD(hi, VAND(VSL(s10, 41), limb_mask));
    hi = VSRA(s10, 7);
    hi = VADD(hi, VSRA(t10, 48));
    t10 = VAND(t10, limb_mask);
    
    t11 = VADD(hi, VAND(VSL(s11, 41), limb_mask));
    t12 = VSRA(s11, 7);
    t12 = VADD(t12, VSRA(t11, 48));
    t11 = VAND(t11, limb_mask);
    
    // t += a (ignoring lower limbs)
    t0 = VADD(t0, a10);
    t1 = VADD(t1, a11);
    t2 = VADD(t2, a12);
    t3 = VADD(t3, a13);
    t4 = VADD(t4, a14);
    t5 = VADD(t5, a15);
    t6 = VADD(t6, a16);
    t7 = VADD(t7, a17);
    t8 = VADD(t8, a18);
    t9 = VADD(t9, a19);
    t10 = VADD(t10, a20);
    t11 = VADD(t11, a21);
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t8 = VADD(t8, VSRA(t7, LIMB_LEN));
    t9 = VADD(t9, VSRA(t8, LIMB_LEN));
    t10 = VADD(t10, VSRA(t9, LIMB_LEN));
    t11 = VADD(t11, VSRA(t10, LIMB_LEN));
    t12 = VADD(t12, VSRA(t11, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    t7 = VAND(t7, limb_mask);
    t8 = VAND(t8, limb_mask);
    t9 = VAND(t9, limb_mask);
    t10 = VAND(t10, limb_mask);
    t11 = VAND(t11, limb_mask);
    
    // After dividing by 2^n, the result is in t2, ..., t12
    // The result can be 522 bits.
    // We do not propagate the carry here. The bottom limb can have 49 bits.
    res[0] = t2; res[1] = t3; res[2] = t4; res[3] = t5; res[4] = t6; res[5] = t7;
    res[6] = t8; res[7] = t9; res[8] = t10; res[9] = t11; res[10] = t12;
}

void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[22];
    __m512i t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16;
    __m512i t17, t18, t19, t20, t21;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    mp_mul_8x1_karat(t, a, b);
    
    // Carry propagation
    t1 = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t2 = VADD(t[2], VSRA(t1, LIMB_LEN));
    t3 = VADD(t[3], VSRA(t2, LIMB_LEN));
    t4 = VADD(t[4], VSRA(t3, LIMB_LEN));
    t5 = VADD(t[5], VSRA(t4, LIMB_LEN));
    t6 = VADD(t[6], VSRA(t5, LIMB_LEN));
    t7 = VADD(t[7], VSRA(t6, LIMB_LEN));
    t8 = VADD(t[8], VSRA(t7, LIMB_LEN));
    t9 = VADD(t[9], VSRA(t8, LIMB_LEN));
    t10 = VADD(t[10], VSRA(t9, LIMB_LEN));
    t11 = VADD(t[11], VSRA(t10, LIMB_LEN));
    t12 = VADD(t[12], VSRA(t11, LIMB_LEN));
    t13 = VADD(t[13], VSRA(t12, LIMB_LEN));
    t14 = VADD(t[14], VSRA(t13, LIMB_LEN));
    t15 = VADD(t[15], VSRA(t14, LIMB_LEN));
    t16 = VADD(t[16], VSRA(t15, LIMB_LEN));
    t17 = VADD(t[17], VSRA(t16, LIMB_LEN));
    t18 = VADD(t[18], VSRA(t17, LIMB_LEN));
    t19 = VADD(t[19], VSRA(t18, LIMB_LEN));
    t20 = VADD(t[20], VSRA(t19, LIMB_LEN));
    t[21] = VADD(t[21], VSRA(t20, LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t1, limb_mask);
    t[2] = VAND(t2, limb_mask);
    t[3] = VAND(t3, limb_mask);
    t[4] = VAND(t4, limb_mask);
    t[5] = VAND(t5, limb_mask);
    t[6] = VAND(t6, limb_mask);
    t[7] = VAND(t7, limb_mask);
    t[8] = VAND(t8, limb_mask);
    t[9] = VAND(t9, limb_mask);
    t[10] = VAND(t10, limb_mask);
    t[11] = VAND(t11, limb_mask);
    t[12] = VAND(t12, limb_mask);
    t[13] = VAND(t13, limb_mask);
    t[14] = VAND(t14, limb_mask);
    t[15] = VAND(t15, limb_mask);
    t[16] = VAND(t16, limb_mask);
    t[17] = VAND(t17, limb_mask);
    t[18] = VAND(t18, limb_mask);
    t[19] = VAND(t19, limb_mask);
    t[20] = VAND(t20, limb_mask);
    
    rdc_8x1(res, t);
}

void fp_sqr_8x1(__m512i* res, __m512i* a)
{
    fp_mul_8x1(res, a, a);
}

void fp_div3_8x1(__m512i* res, __m512i* src)
{
    __m512i t[11];
    fp_duplicate_8x1(t, INV3);
    fp_mul_8x1(res, src, t);
}

// Batched inversion invoking a single call to fp_inv_64.
void fp_inv_8x1(__m512i* res, __m512i* a)
{
    uint64_t r_64[18];
    __m512i u[11], t0[6], t1[6], s0[3], s1[3], r[3];
    __m512i idx0, idx1;
    
    fp_8x1_to_4x2(t0, t1, a);
    // t0 = (a0a1, a2a3, a4a5, a6a7)
    fp_mul_4x2(t0, t0, t1);
    fp_4x2_to_2x4(s0, s1, t0);
    // s0 = (a0a1a2a3, a4a5a6a7)
    fp_mul_2x4(s0, s0, s1);
    idx1 = VSET(4, 5, 6, 7, 0, 0, 0, 0);
    for (int i = 0; i < 3; i++) {
        s1[i] = VPERM512(idx1, s0[i]);
    }
    // r = a0a1a2a3a4a5a6a7 (wastes a mul)
    fp_mul_2x4(r, s0, s1);
    
    // Unvectorized inversion of r
    fp_limbs_to_64_2x4(r_64, r);
    fp_inv_64(r_64, r_64);
    fp_64_to_limbs_2x4(r, r_64);
    
    // s0 = (inv_0123, inv_4567)
    idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < 3; i++) {
        s0[i] = VPERM512(idx0, s0[i]);
        s1[i] = VPERM512(idx1, r[i]);
    }
    fp_mul_2x4(s0, s0, s1);
    // t0 = (inv01, inv23, inv45, inv67)
    fp_2x4_to_4x2(t1, s0, s0); // (inv_0123, inv_0123, inv_4567, inv_4567)
    idx0 = VSET(2, 3, 0, 1, 6, 7, 4, 5);
    for (int i = 0; i < 6; i++) {
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp_mul_4x2(t0, t0, t1);
    // res = (inv0, inv1, inv2, ...)
    fp_4x2_to_8x1(u, t0, t0);
    idx0 = VSET(1, 0, 3, 2, 5, 4, 7, 6);
    for (int i = 0; i < 11; i++) {
        res[i] = VPERM512(idx0, a[i]);
    }
    fp_mul_8x1(res, res, u);
}

void fp_tomont_8x1(__m512i* res, __m512i* a)
{
    __m512i vR2[NUM_LIMBS];
    fp_duplicate_8x1(vR2, R2);
    fp_mul_8x1(res, a, vR2);
}

void fp_frommont_8x1(__m512i* res, __m512i* a)
{
    __m512i one[NUM_LIMBS];
    fp_setzero_8x1(one);
    one[0] = VSETALL(1);
    fp_mul_8x1(res, a, one);
}

// Assumes unvec is an array of 8 * NWORDS64 64-bit ints
void fp_64_to_limbs_8x1(__m512i* res, uint64_t* unvec)
{
    uint64_t t[8];
    
    for (int i = 0; i < 8; i++) {
        t[i] = unvec[NWORDS64 * i] & LIMB_MASK;
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NWORDS64 * i] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 1] & 0xffffffffULL) << 16);
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NWORDS64 * i + 1] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 2] & 0xffffULL) << 32);
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (unvec[NWORDS64 * i + 2] >> 16) & LIMB_MASK;
    }
    res[3] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = unvec[NWORDS64 * i + 3] & LIMB_MASK;
    }
    res[4] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NWORDS64 * i + 3] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 4] & 0xffffffffULL) << 16);
    }
    res[5] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NWORDS64 * i + 4] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 5] & 0xffffULL) << 32);
    }
    res[6] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (unvec[NWORDS64 * i + 5] >> 16) & LIMB_MASK;
    }
    res[7] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = unvec[NWORDS64 * i + 6] & LIMB_MASK;
    }
    res[8] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NWORDS64 * i + 6] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 7] & 0xffffffffULL) << 16);
    }
    res[9] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (unvec[NWORDS64 * i + 7] >> 32) & LIMB_MASK;
    }
    res[10] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

// Assumes unvec is an array of 4 * NWORDS64 64-bit ints
void fp_64_to_limbs_8x1_only_even_idx(__m512i* res, uint64_t* unvec)
{
    uint64_t t[4];
    
    for (int i = 0; i < 4; i++) {
        t[i] = unvec[NWORDS64 * i] & LIMB_MASK;
    }
    res[0] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NWORDS64 * i] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 1] & 0xffffffffULL) << 16);
    }
    res[1] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NWORDS64 * i + 1] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 2] & 0xffffULL) << 32);
    }
    res[2] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (unvec[NWORDS64 * i + 2] >> 16) & LIMB_MASK;
    }
    res[3] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = unvec[NWORDS64 * i + 3] & LIMB_MASK;
    }
    res[4] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NWORDS64 * i + 3] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 4] & 0xffffffffULL) << 16);
    }
    res[5] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NWORDS64 * i + 4] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 5] & 0xffffULL) << 32);
    }
    res[6] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (unvec[NWORDS64 * i + 5] >> 16) & LIMB_MASK;
    }
    res[7] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = unvec[NWORDS64 * i + 6] & LIMB_MASK;
    }
    res[8] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NWORDS64 * i + 6] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 7] & 0xffffffffULL) << 16);
    }
    res[9] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (unvec[NWORDS64 * i + 7] >> 32) & LIMB_MASK;
    }
    res[10] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
}

void fp_limbs_to_64_8x1(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[11], zero[11];
    // Normalize input
    fp_setzero_8x1(zero);
    fp_add_8x1(s, src, zero);
    
    
    VSTORE(t, s[0]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i] = t[i] & LIMB_MASK;
    }
    VSTORE(t, s[1]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i] |= (t[i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 1] = (t[i] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, s[2]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 1] |= (t[i] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 2] = (t[i] >> 32) & 0xffffULL;
    }
    VSTORE(t, s[3]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 2] |= (t[i] & LIMB_MASK) << 16;
    }
    VSTORE(t, s[4]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 3] = t[i] & LIMB_MASK;
    }
    VSTORE(t, s[5]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 3] |= (t[i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 4] = (t[i] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, s[6]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 4] |= (t[i] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 5] = (t[i] >> 32) & 0xffffULL;
    }
    VSTORE(t, s[7]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 5] |= (t[i] & LIMB_MASK) << 16;
    }
    VSTORE(t, s[8]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 6] = t[i] & LIMB_MASK;
    }
    VSTORE(t, s[9]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 6] |= (t[i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 7] = (t[i] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, s[10]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 7] |= (t[i] & LIMB_MASK) << 32;
    }
}

void fp_limbs_to_64_8x1_only_even_idx(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[11], zero[11];
    // Normalize input
    fp_setzero_8x1(zero);
    fp_add_8x1(s, src, zero);
    
    
    VSTORE(t, s[0]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i] = t[2*i] & LIMB_MASK;
    }
    VSTORE(t, s[1]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i] |= (t[2*i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 1] = (t[2*i] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, s[2]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 1] |= (t[2*i] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 2] = (t[2*i] >> 32) & 0xffffULL;
    }
    VSTORE(t, s[3]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 2] |= (t[2*i] & LIMB_MASK) << 16;
    }
    VSTORE(t, s[4]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 3] = t[2*i] & LIMB_MASK;
    }
    VSTORE(t, s[5]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 3] |= (t[2*i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 4] = (t[2*i] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, s[6]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 4] |= (t[2*i] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 5] = (t[2*i] >> 32) & 0xffffULL;
    }
    VSTORE(t, s[7]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 5] |= (t[2*i] & LIMB_MASK) << 16;
    }
    VSTORE(t, s[8]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 6] = t[2*i] & LIMB_MASK;
    }
    VSTORE(t, s[9]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 6] |= (t[2*i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 7] = (t[2*i] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, s[10]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 7] |= (t[2*i] & LIMB_MASK) << 32;
    }
}

void rand_fp_8x1(__m512i* res)
{
    __m512i zero[11];
    __m512i mask = VSETALL(LIMB_MASK);
    fp_setzero_8x1(zero);
    
    for (int i = 0; i < 10; i++) {
        res[i] = VAND(rand_512i(), mask);
    }
    // Last limb is shorter
    mask = VSETALL(LAST_LIMB_MASK);
    res[10] = VAND(rand_512i(), mask);
    
    // Reduction
    fp_add_8x1(res, res, zero);
}

// ====================================================================================================
//                                                 4x2                                                
// ====================================================================================================

void fp_setzero_4x2(__m512i* res)
{
    res[0] = VZERO();
    res[1] = VZERO();
    res[2] = VZERO();
    res[3] = VZERO();
    res[4] = VZERO();
    res[5] = VZERO();
}

void fp_copy_4x2(__m512i* res, __m512i* a)
{
    res[0] = a[0];
    res[1] = a[1];
    res[2] = a[2];
    res[3] = a[3];
    res[4] = a[4];
    res[5] = a[5];
}

void fp_duplicate_4x2(__m512i* res, const uint64_t* a)
{
    res[0] = VSET(a[0], a[6], a[0], a[6], a[0], a[6], a[0], a[6]);
    res[1] = VSET(a[1], a[7], a[1], a[7], a[1], a[7], a[1], a[7]);
    res[2] = VSET(a[2], a[8], a[2], a[8], a[2], a[8], a[2], a[8]);
    res[3] = VSET(a[3], a[9], a[3], a[9], a[3], a[9], a[3], a[9]);
    res[4] = VSET(a[4], a[10], a[4], a[10], a[4], a[10], a[4], a[10]);
    res[5] = VSET(a[5], 0, a[5], 0, a[5], 0, a[5], 0);
}

int fp_is_equal_4x2(__m512i* a, __m512i* b)
{
    __m512i zero[N4x2], norm_a[N4x2], norm_b[N4x2];
    __mmask8 acc;
    fp_setzero_4x2(zero);
    
    // Normalize both to [0, p-1] by adding zero
    fp_add_4x2(norm_a, a, zero);
    fp_add_4x2(norm_b, b, zero);
    
    acc = 0xff;
    acc &= VCMPEQ(norm_a[0], norm_b[0]);
    acc &= VCMPEQ(norm_a[1], norm_b[1]);
    acc &= VCMPEQ(norm_a[2], norm_b[2]);
    acc &= VCMPEQ(norm_a[3], norm_b[3]);
    acc &= VCMPEQ(norm_a[4], norm_b[4]);
    acc &= VCMPEQ(norm_a[5], norm_b[5]);
    
    // a = b iff both halves are equal
    return (acc & (acc >> 1) & 0x1)
        | ((acc >> 1) & (acc >> 2) & 0x2)
        | ((acc >> 2) & (acc >> 3) & 0x4)
        | ((acc >> 3) & (acc >> 4) & 0x8);
}

int fp_is_zero_4x2(__m512i* a)
{
    __m512i zero[N4x2], norm_a[N4x2];
    __mmask8 acc;
    fp_setzero_4x2(zero);
    
    // Normalize a to [0, p-1] by adding zero
    fp_add_4x2(norm_a, a, zero);
    
    acc = 0xff;
    acc &= VCMPEQ(norm_a[0], zero[0]);
    acc &= VCMPEQ(norm_a[1], zero[1]);
    acc &= VCMPEQ(norm_a[2], zero[2]);
    acc &= VCMPEQ(norm_a[3], zero[3]);
    acc &= VCMPEQ(norm_a[4], zero[4]);
    acc &= VCMPEQ(norm_a[5], zero[5]);
    
    // a = 0 iff both halves are 0
    return (acc & (acc >> 1) & 0x1)
        | ((acc >> 1) & (acc >> 2) & 0x2)
        | ((acc >> 2) & (acc >> 3) & 0x4)
        | ((acc >> 3) & (acc >> 4) & 0x8);
}

void fp_add_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[6], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    
    // Add limbs
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    t[3] = VADD(a[3], b[3]);
    t[4] = VADD(a[4], b[4]);
    t[5] = VADD(a[5], b[5]);
    
    // Subtract p, using special shape of p
    t[0] = VADD(t[0], p_bot);
    t[4] = VSUB(t[4], p_top);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0x55, t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADDMASK(t[5], 0x55, t[5], VSRA(t[4], LIMB_LEN));
    carry = VSRA(t[5], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0xaa, t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VANDMASK(t[4], 0x55, t[4], limb_mask);
    t[5] = VANDMASK(t[5], 0x55, t[5], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[4], 63);
    t[4] = VADD(t[4], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0x55, t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADDMASK(t[5], 0x55, t[5], VSRA(t[4], LIMB_LEN));
    carry = VSRA(t[5], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0xaa, t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VANDMASK(t[4], 0x55, t[4], limb_mask);
    t[5] = VANDMASK(t[5], 0x55, t[5], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
}

void fp_sub_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[6], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    
    // Subtract limbs
    t[0] = VSUB(a[0], b[0]);
    t[1] = VSUB(a[1], b[1]);
    t[2] = VSUB(a[2], b[2]);
    t[3] = VSUB(a[3], b[3]);
    t[4] = VSUB(a[4], b[4]);
    t[5] = VSUB(a[5], b[5]);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0x55, t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADDMASK(t[5], 0x55, t[5], VSRA(t[4], LIMB_LEN));
    carry = VSRA(t[5], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0xaa, t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VANDMASK(t[4], 0x55, t[4], limb_mask);
    t[5] = VANDMASK(t[5], 0x55, t[5], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[4], 63);
    t[4] = VADD(t[4], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0x55, t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADDMASK(t[5], 0x55, t[5], VSRA(t[4], LIMB_LEN));
    carry = VSRA(t[5], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0xaa, t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VANDMASK(t[4], 0x55, t[4], limb_mask);
    t[5] = VANDMASK(t[5], 0x55, t[5], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
}

void fp_neg_4x2(__m512i* res, __m512i* a)
{
    __m512i t[6], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    const __m512i zero = VZERO();
    
    // Subtract limbs
    t[0] = VSUB(zero, a[0]);
    t[1] = VSUB(zero, a[1]);
    t[2] = VSUB(zero, a[2]);
    t[3] = VSUB(zero, a[3]);
    t[4] = VSUB(zero, a[4]);
    t[5] = VSUB(zero, a[5]);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0x55, t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADDMASK(t[5], 0x55, t[5], VSRA(t[4], LIMB_LEN));
    carry = VSRA(t[5], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0xaa, t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VANDMASK(t[4], 0x55, t[4], limb_mask);
    t[5] = VANDMASK(t[5], 0x55, t[5], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[4], 63);
    t[4] = VADD(t[4], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0x55, t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADDMASK(t[5], 0x55, t[5], VSRA(t[4], LIMB_LEN));
    carry = VSRA(t[5], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0xaa, t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VANDMASK(t[4], 0x55, t[4], limb_mask);
    t[5] = VANDMASK(t[5], 0x55, t[5], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
}

void fp_mixaddsub_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[6], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    const __mmask8 sub_mask = ~add_mask;
    
    t[0] = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t[1] = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t[2] = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    t[3] = VSUBMASK(a[3], sub_mask, a[3], b[3]);
    t[4] = VSUBMASK(a[4], sub_mask, a[4], b[4]);
    t[5] = VSUBMASK(a[5], sub_mask, a[5], b[5]);
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    t[2] = VADDMASK(t[2], add_mask, a[2], b[2]);
    t[3] = VADDMASK(t[3], add_mask, a[3], b[3]);
    t[4] = VADDMASK(t[4], add_mask, a[4], b[4]);
    t[5] = VADDMASK(t[5], add_mask, a[5], b[5]);
    
    // a + b - p
    t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);
    t[4] = VSUBMASK(t[4], add_mask, t[4], p_top);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0x55, t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADDMASK(t[5], 0x55, t[5], VSRA(t[4], LIMB_LEN));
    carry = VSRA(t[5], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0xaa, t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VANDMASK(t[4], 0x55, t[4], limb_mask);
    t[5] = VANDMASK(t[5], 0x55, t[5], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[4], 63);
    t[4] = VADD(t[4], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0x55, t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADDMASK(t[5], 0x55, t[5], VSRA(t[4], LIMB_LEN));
    carry = VSRA(t[5], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADDMASK(t[4], 0xaa, t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VANDMASK(t[4], 0x55, t[4], limb_mask);
    t[5] = VANDMASK(t[5], 0x55, t[5], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
}

// Single-length multiprecision addition, no correction, no carry propagation
void mp_add_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[6];
    
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    t[3] = VADD(a[3], b[3]);
    t[4] = VADD(a[4], b[4]);
    t[5] = VADD(a[5], b[5]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
}

void mp_add_p_sub_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[6];
    const __m512i p_bod = VSETALL(LIMB_MASK);
    const __m512i p_top = VSET(LIMB_MASK, P_PLUS_1_TOP-1, LIMB_MASK, P_PLUS_1_TOP-1, LIMB_MASK, P_PLUS_1_TOP-1, LIMB_MASK, P_PLUS_1_TOP-1);
    
    t[0] = VSUB(VADD(a[0], p_bod), b[0]);
    t[1] = VSUB(VADD(a[1], p_bod), b[1]);
    t[2] = VSUB(VADD(a[2], p_bod), b[2]);
    t[3] = VSUB(VADD(a[3], p_bod), b[3]);
    t[4] = VSUB(VADD(a[4], p_top), b[4]);
    t[5] = VADDMASK(a[5], 0x55, a[5], p_bod);
    t[5] = VSUBMASK(t[5], 0x55, t[5], b[5]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
}

void mp_mixaddsubx2_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[18];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __mmask8 sub_mask = ~add_mask;
    
    t[0] = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t[1] = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t[2] = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    t[3] = VSUBMASK(a[3], sub_mask, a[3], b[3]);
    t[4] = VSUBMASK(a[4], sub_mask, a[4], b[4]);
    t[5] = VSUBMASK(a[5], sub_mask, a[5], b[5]);
    t[6] = VSUBMASK(a[6], sub_mask, a[6], b[6]);
    t[7] = VSUBMASK(a[7], sub_mask, a[7], b[7]);
    t[8] = VSUBMASK(a[8], sub_mask, a[8], b[8]);
    t[9] = VSUBMASK(a[9], sub_mask, a[9], b[9]);
    t[10] = VSUBMASK(a[10], sub_mask, a[10], b[10]);
    t[11] = VSUBMASK(a[11], sub_mask, a[11], b[11]);
    t[12] = VSUBMASK(a[12], sub_mask, a[12], b[12]);
    t[13] = VSUBMASK(a[13], sub_mask, a[13], b[13]);
    t[14] = VSUBMASK(a[14], sub_mask, a[14], b[14]);
    t[15] = VSUBMASK(a[15], sub_mask, a[15], b[15]);
    t[16] = VSUBMASK(a[16], sub_mask, a[16], b[16]);
    t[17] = VSUBMASK(a[17], sub_mask, a[17], b[17]);
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    t[2] = VADDMASK(t[2], add_mask, a[2], b[2]);
    t[3] = VADDMASK(t[3], add_mask, a[3], b[3]);
    t[4] = VADDMASK(t[4], add_mask, a[4], b[4]);
    t[5] = VADDMASK(t[5], add_mask, a[5], b[5]);
    t[6] = VADDMASK(t[6], add_mask, a[6], b[6]);
    t[7] = VADDMASK(t[7], add_mask, a[7], b[7]);
    t[8] = VADDMASK(t[8], add_mask, a[8], b[8]);
    t[9] = VADDMASK(t[9], add_mask, a[9], b[9]);
    t[10] = VADDMASK(t[10], add_mask, a[10], b[10]);
    t[11] = VADDMASK(t[11], add_mask, a[11], b[11]);
    t[12] = VADDMASK(t[12], add_mask, a[12], b[12]);
    t[13] = VADDMASK(t[13], add_mask, a[13], b[13]);
    t[14] = VADDMASK(t[14], add_mask, a[14], b[14]);
    t[15] = VADDMASK(t[15], add_mask, a[15], b[15]);
    t[16] = VADDMASK(t[16], add_mask, a[16], b[16]);
    t[17] = VADDMASK(t[17], add_mask, a[17], b[17]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
    res[11] = t[11];
    res[12] = t[12];
    res[13] = t[13];
    res[14] = t[14];
    res[15] = t[15];
    res[16] = t[16];
    res[17] = t[17];
}

// Double-length multiprecision subtraction res = res - a - b, no correction
void mp_dblsubx2_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[18];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    t[0] = VSUB(res[0], a[0]);
    t[1] = VSUB(res[1], a[1]);
    t[2] = VSUB(res[2], a[2]);
    t[3] = VSUB(res[3], a[3]);
    t[4] = VSUB(res[4], a[4]);
    t[5] = VSUB(res[5], a[5]);
    t[6] = VSUB(res[6], a[6]);
    t[7] = VSUB(res[7], a[7]);
    t[8] = VSUB(res[8], a[8]);
    t[9] = VSUB(res[9], a[9]);
    t[10] = VSUB(res[10], a[10]);
    t[11] = VSUB(res[11], a[11]);
    t[12] = VSUB(res[12], a[12]);
    t[13] = VSUB(res[13], a[13]);
    t[14] = VSUB(res[14], a[14]);
    t[15] = VSUB(res[15], a[15]);
    t[16] = VSUB(res[16], a[16]);
    t[17] = VSUB(res[17], a[17]);
    
    t[0] = VSUB(t[0], b[0]);
    t[1] = VSUB(t[1], b[1]);
    t[2] = VSUB(t[2], b[2]);
    t[3] = VSUB(t[3], b[3]);
    t[4] = VSUB(t[4], b[4]);
    t[5] = VSUB(t[5], b[5]);
    t[6] = VSUB(t[6], b[6]);
    t[7] = VSUB(t[7], b[7]);
    t[8] = VSUB(t[8], b[8]);
    t[9] = VSUB(t[9], b[9]);
    t[10] = VSUB(t[10], b[10]);
    t[11] = VSUB(t[11], b[11]);
    t[12] = VSUB(t[12], b[12]);
    t[13] = VSUB(t[13], b[13]);
    t[14] = VSUB(t[14], b[14]);
    t[15] = VSUB(t[15], b[15]);
    t[16] = VSUB(t[16], b[16]);
    t[17] = VSUB(t[17], b[17]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
    res[11] = t[11];
    res[12] = t[12];
    res[13] = t[13];
    res[14] = t[14];
    res[15] = t[15];
    res[16] = t[16];
    res[17] = t[17];
}

// Double-length multiprecision subtraction res = res - a
void mp_subx2_4x2(__m512i* res, __m512i* a)
{
    __m512i t[18];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    t[0] = VSUB(res[0], a[0]);
    t[1] = VSUB(res[1], a[1]);
    t[2] = VSUB(res[2], a[2]);
    t[3] = VSUB(res[3], a[3]);
    t[4] = VSUB(res[4], a[4]);
    t[5] = VSUB(res[5], a[5]);
    t[6] = VSUB(res[6], a[6]);
    t[7] = VSUB(res[7], a[7]);
    t[8] = VSUB(res[8], a[8]);
    t[9] = VSUB(res[9], a[9]);
    t[10] = VSUB(res[10], a[10]);
    t[11] = VSUB(res[11], a[11]);
    t[12] = VSUB(res[12], a[12]);
    t[13] = VSUB(res[13], a[13]);
    t[14] = VSUB(res[14], a[14]);
    t[15] = VSUB(res[15], a[15]);
    t[16] = VSUB(res[16], a[16]);
    t[17] = VSUB(res[17], a[17]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
    res[11] = t[11];
    res[12] = t[12];
    res[13] = t[13];
    res[14] = t[14];
    res[15] = t[15];
    res[16] = t[16];
    res[17] = t[17];
}

// Algorithm 2 from "Finite Field Arithmetic Using AVX-512 For Isogeny-Based Cryptography" by Orisaka, Aranha and López
void mp_mul_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[17], sl[6], sh[5];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i zero = VZERO();
    
    // Duplicating limbs of b with shuffles
    // If b[0] = {a_0, a_6, b_0, b_6, ...} then sl0 = {a_0, a_0, b_0, b_0, ...}
    sl[0] = VSHUFFLE(b[0], 0x44);
    sl[1] = VSHUFFLE(b[1], 0x44);
    sl[2] = VSHUFFLE(b[2], 0x44);
    sl[3] = VSHUFFLE(b[3], 0x44);
    sl[4] = VSHUFFLE(b[4], 0x44);
    sl[5] = VSHUFFLE(b[5], 0x44);
    // If b[0] = {a_0, a_6, b_0, b_6, ...} then sh0 = {a_6, a_6, b_6, b_6, ...}
    sh[0] = VSHUFFLE(b[0], 0xee);
    sh[1] = VSHUFFLE(b[1], 0xee);
    sh[2] = VSHUFFLE(b[2], 0xee);
    sh[3] = VSHUFFLE(b[3], 0xee);
    sh[4] = VSHUFFLE(b[4], 0xee);
    
    // Schoolbook
    t[0] = VMULADDL(zero, a[0], sl[0]);
    t[1] = VMULADDH(zero, a[0], sl[0]);
    t[1] = VSL(t[1], LIMB_ALIGN);
    
    t[1] = VMULADDL(t[1], a[0], sl[1]);
    t[2] = VMULADDH(zero, a[0], sl[1]);
    t[1] = VMULADDL(t[1], a[1], sl[0]);
    t[2] = VMULADDH(t[2], a[1], sl[0]);
    t[2] = VSL(t[2], LIMB_ALIGN);
    
    t[2] = VMULADDL(t[2], a[0], sl[2]);
    t[3] = VMULADDH(zero, a[0], sl[2]);
    t[2] = VMULADDL(t[2], a[1], sl[1]);
    t[3] = VMULADDH(t[3], a[1], sl[1]);
    t[2] = VMULADDL(t[2], a[2], sl[0]);
    t[3] = VMULADDH(t[3], a[2], sl[0]);
    t[3] = VSL(t[3], LIMB_ALIGN);
    
    t[3] = VMULADDL(t[3], a[0], sl[3]);
    t[4] = VMULADDH(zero, a[0], sl[3]);
    t[3] = VMULADDL(t[3], a[1], sl[2]);
    t[4] = VMULADDH(t[4], a[1], sl[2]);
    t[3] = VMULADDL(t[3], a[2], sl[1]);
    t[4] = VMULADDH(t[4], a[2], sl[1]);
    t[3] = VMULADDL(t[3], a[3], sl[0]);
    t[4] = VMULADDH(t[4], a[3], sl[0]);
    t[4] = VSL(t[4], LIMB_ALIGN);
    
    t[4] = VMULADDL(t[4], a[0], sl[4]);
    t[5] = VMULADDH(zero, a[0], sl[4]);
    t[4] = VMULADDL(t[4], a[1], sl[3]);
    t[5] = VMULADDH(t[5], a[1], sl[3]);
    t[4] = VMULADDL(t[4], a[2], sl[2]);
    t[5] = VMULADDH(t[5], a[2], sl[2]);
    t[4] = VMULADDL(t[4], a[3], sl[1]);
    t[5] = VMULADDH(t[5], a[3], sl[1]);
    t[4] = VMULADDL(t[4], a[4], sl[0]);
    t[5] = VMULADDH(t[5], a[4], sl[0]);
    t[5] = VSL(t[5], LIMB_ALIGN);
    
    t[5] = VMULADDL(t[5], a[0], sl[5]);
    t[6] = VMULADDH(zero, a[0], sl[5]);
    t[5] = VMULADDL(t[5], a[1], sl[4]);
    t[6] = VMULADDH(t[6], a[1], sl[4]);
    t[5] = VMULADDL(t[5], a[2], sl[3]);
    t[6] = VMULADDH(t[6], a[2], sl[3]);
    t[5] = VMULADDL(t[5], a[3], sl[2]);
    t[6] = VMULADDH(t[6], a[3], sl[2]);
    t[5] = VMULADDL(t[5], a[4], sl[1]);
    t[6] = VMULADDH(t[6], a[4], sl[1]);
    t[5] = VMULADDL(t[5], a[5], sl[0]);
    t[6] = VMULADDH(t[6], a[5], sl[0]);
    t[6] = VSL(t[6], LIMB_ALIGN);
    
    t[6] = VMULADDL(t[6], a[0], sh[0]);
    t[7] = VMULADDH(zero, a[0], sh[0]);
    t[6] = VMULADDL(t[6], a[1], sl[5]);
    t[7] = VMULADDH(t[7], a[1], sl[5]);
    t[6] = VMULADDL(t[6], a[2], sl[4]);
    t[7] = VMULADDH(t[7], a[2], sl[4]);
    t[6] = VMULADDL(t[6], a[3], sl[3]);
    t[7] = VMULADDH(t[7], a[3], sl[3]);
    t[6] = VMULADDL(t[6], a[4], sl[2]);
    t[7] = VMULADDH(t[7], a[4], sl[2]);
    t[6] = VMULADDL(t[6], a[5], sl[1]);
    t[7] = VMULADDH(t[7], a[5], sl[1]);
    t[7] = VSL(t[7], LIMB_ALIGN);
    
    t[7] = VMULADDL(t[7], a[0], sh[1]);
    t[8] = VMULADDH(zero, a[0], sh[1]);
    t[7] = VMULADDL(t[7], a[1], sh[0]);
    t[8] = VMULADDH(t[8], a[1], sh[0]);
    t[7] = VMULADDL(t[7], a[2], sl[5]);
    t[8] = VMULADDH(t[8], a[2], sl[5]);
    t[7] = VMULADDL(t[7], a[3], sl[4]);
    t[8] = VMULADDH(t[8], a[3], sl[4]);
    t[7] = VMULADDL(t[7], a[4], sl[3]);
    t[8] = VMULADDH(t[8], a[4], sl[3]);
    t[7] = VMULADDL(t[7], a[5], sl[2]);
    t[8] = VMULADDH(t[8], a[5], sl[2]);
    t[8] = VSL(t[8], LIMB_ALIGN);
    
    t[8] = VMULADDL(t[8], a[0], sh[2]);
    t[9] = VMULADDH(zero, a[0], sh[2]);
    t[8] = VMULADDL(t[8], a[1], sh[1]);
    t[9] = VMULADDH(t[9], a[1], sh[1]);
    t[8] = VMULADDL(t[8], a[2], sh[0]);
    t[9] = VMULADDH(t[9], a[2], sh[0]);
    t[8] = VMULADDL(t[8], a[3], sl[5]);
    t[9] = VMULADDH(t[9], a[3], sl[5]);
    t[8] = VMULADDL(t[8], a[4], sl[4]);
    t[9] = VMULADDH(t[9], a[4], sl[4]);
    t[8] = VMULADDL(t[8], a[5], sl[3]);
    t[9] = VMULADDH(t[9], a[5], sl[3]);
    t[9] = VSL(t[9], LIMB_ALIGN);
    
    t[9] = VMULADDL(t[9], a[0], sh[3]);
    t[10] = VMULADDH(zero, a[0], sh[3]);
    t[9] = VMULADDL(t[9], a[1], sh[2]);
    t[10] = VMULADDH(t[10], a[1], sh[2]);
    t[9] = VMULADDL(t[9], a[2], sh[1]);
    t[10] = VMULADDH(t[10], a[2], sh[1]);
    t[9] = VMULADDL(t[9], a[3], sh[0]);
    t[10] = VMULADDH(t[10], a[3], sh[0]);
    t[9] = VMULADDL(t[9], a[4], sl[5]);
    t[10] = VMULADDH(t[10], a[4], sl[5]);
    t[9] = VMULADDL(t[9], a[5], sl[4]);
    t[10] = VMULADDH(t[10], a[5], sl[4]);
    t[10] = VSL(t[10], LIMB_ALIGN);
    
    t[10] = VMULADDL(t[10], a[0], sh[4]);
    t[11] = VMULADDH(zero, a[0], sh[4]);
    t[10] = VMULADDL(t[10], a[1], sh[3]);
    t[11] = VMULADDH(t[11], a[1], sh[3]);
    t[10] = VMULADDL(t[10], a[2], sh[2]);
    t[11] = VMULADDH(t[11], a[2], sh[2]);
    t[10] = VMULADDL(t[10], a[3], sh[1]);
    t[11] = VMULADDH(t[11], a[3], sh[1]);
    t[10] = VMULADDL(t[10], a[4], sh[0]);
    t[11] = VMULADDH(t[11], a[4], sh[0]);
    t[10] = VMULADDL(t[10], a[5], sl[5]);
    t[11] = VMULADDH(t[11], a[5], sl[5]);
    t[11] = VSL(t[11], LIMB_ALIGN);
    
    t[11] = VMULADDL(t[11], a[1], sh[4]);
    t[12] = VMULADDH(zero, a[1], sh[4]);
    t[11] = VMULADDL(t[11], a[2], sh[3]);
    t[12] = VMULADDH(t[12], a[2], sh[3]);
    t[11] = VMULADDL(t[11], a[3], sh[2]);
    t[12] = VMULADDH(t[12], a[3], sh[2]);
    t[11] = VMULADDL(t[11], a[4], sh[1]);
    t[12] = VMULADDH(t[12], a[4], sh[1]);
    t[11] = VMULADDL(t[11], a[5], sh[0]);
    t[12] = VMULADDH(t[12], a[5], sh[0]);
    t[12] = VSL(t[12], LIMB_ALIGN);
    
    t[12] = VMULADDL(t[12], a[2], sh[4]);
    t[13] = VMULADDH(zero, a[2], sh[4]);
    t[12] = VMULADDL(t[12], a[3], sh[3]);
    t[13] = VMULADDH(t[13], a[3], sh[3]);
    t[12] = VMULADDL(t[12], a[4], sh[2]);
    t[13] = VMULADDH(t[13], a[4], sh[2]);
    t[12] = VMULADDL(t[12], a[5], sh[1]);
    t[13] = VMULADDH(t[13], a[5], sh[1]);
    t[13] = VSL(t[13], LIMB_ALIGN);
    
    t[13] = VMULADDL(t[13], a[3], sh[4]);
    t[14] = VMULADDH(zero, a[3], sh[4]);
    t[13] = VMULADDL(t[13], a[4], sh[3]);
    t[14] = VMULADDH(t[14], a[4], sh[3]);
    t[13] = VMULADDL(t[13], a[5], sh[2]);
    t[14] = VMULADDH(t[14], a[5], sh[2]);
    t[14] = VSL(t[14], LIMB_ALIGN);
    
    t[14] = VMULADDL(t[14], a[4], sh[4]);
    t[15] = VMULADDH(zero, a[4], sh[4]);
    t[14] = VMULADDL(t[14], a[5], sh[3]);
    t[15] = VMULADDH(t[15], a[5], sh[3]);
    t[15] = VSL(t[15], LIMB_ALIGN);
    
    t[15] = VMULADDL(t[15], a[5], sh[4]);
    t[16] = VMULADDH(zero, a[5], sh[4]);
    t[16] = VSL(t[16], LIMB_ALIGN);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
    res[11] = t[11];
    res[12] = t[12];
    res[13] = t[13];
    res[14] = t[14];
    res[15] = t[15];
    res[16] = t[16];
    res[17] = zero;
}

// Algorithm 3 from "Finite Field Arithmetic Using AVX-512 For Isogeny-Based Cryptography"
// by Orisaka, Aranha and López
// Modifies a
void rdc_4x2(__m512i* res, __m512i* a)
{
    __m512i carry, hi, shuf;
    const __m512i top = VSETALL(P_PLUS_1_TOP);
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i zero = VZERO();
    const __m512i one = VSETALL(1);
    
    // Montgomery reduction loop
    // using the fact that -p^{-1} mod 2^(limb_len) = 1 + c* 2^f mod 2^(limb_len) = 1 as f > limb_len
    carry = VSRA(a[0], LIMB_LEN);
    a[0] = VAND(a[0], limb_mask);
    a[1] = VADD(a[1], carry);
    // a = a + u_i * p * 2^(limb_len * i)
    // To multiply by p, we multiply by (p+1) and subtract one.
    // However, since we will drop the lower limbs when we divide later, we don't need to subtract.
    shuf = VSHUFFLEMASKZ(0xcccc, a[0], 0x4e);
    a[4] = VMULADDL(a[4], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[5] = VADD(a[5], hi);
    // Folding step
    a[6] = VADDMASK(a[6], 0x55, a[6], VSHUFFLE(a[0], 0x4e));
    
    carry = VSRA(a[1], LIMB_LEN);
    a[1] = VAND(a[1], limb_mask);
    a[2] = VADD(a[2], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[1], 0x4e);
    a[5] = VMULADDL(a[5], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[6] = VADD(a[6], hi);
    a[7] = VADDMASK(a[7], 0x55, a[7], VSHUFFLE(a[1], 0x4e));
    
    carry = VSRA(a[2], LIMB_LEN);
    a[2] = VAND(a[2], limb_mask);
    a[3] = VADD(a[3], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[2], 0x4e);
    a[6] = VMULADDL(a[6], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[7] = VADD(a[7], hi);
    a[8] = VADDMASK(a[8], 0x55, a[8], VSHUFFLE(a[2], 0x4e));
    
    carry = VSRA(a[3], LIMB_LEN);
    a[3] = VAND(a[3], limb_mask);
    a[4] = VADD(a[4], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[3], 0x4e);
    a[7] = VMULADDL(a[7], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[8] = VADD(a[8], hi);
    a[9] = VADDMASK(a[9], 0x55, a[9], VSHUFFLE(a[3], 0x4e));
    
    carry = VSRA(a[4], LIMB_LEN);
    a[4] = VAND(a[4], limb_mask);
    a[5] = VADD(a[5], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[4], 0x4e);
    a[8] = VMULADDL(a[8], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[9] = VADD(a[9], hi);
    a[10] = VADDMASK(a[10], 0x55, a[10], VSHUFFLE(a[4], 0x4e));
    
    carry = VSRA(a[5], LIMB_LEN);
    a[5] = VAND(a[5], limb_mask);
    a[6] = VADD(a[6], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[5], 0x4e);
    a[9] = VMULADDL(a[9], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[10] = VADD(a[10], hi);
    a[11] = VADDMASK(a[11], 0x55, a[11], VSHUFFLE(a[5], 0x4e));
    
    carry = VSRA(a[6], LIMB_LEN);
    a[6] = VAND(a[6], limb_mask);
    a[7] = VADD(a[7], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[6], 0x4e);
    a[10] = VMULADDL(a[10], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[11] = VADD(a[11], hi);
    a[12] = VADDMASK(a[12], 0x55, a[12], VSHUFFLE(a[6], 0x4e));
    
    carry = VSRA(a[7], LIMB_LEN);
    a[7] = VAND(a[7], limb_mask);
    a[8] = VADD(a[8], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[7], 0x4e);
    a[11] = VMULADDL(a[11], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[12] = VADD(a[12], hi);
    a[13] = VADDMASK(a[13], 0x55, a[13], VSHUFFLE(a[7], 0x4e));
    
    carry = VSRA(a[8], LIMB_LEN);
    a[8] = VAND(a[8], limb_mask);
    a[9] = VADD(a[9], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[8], 0x4e);
    a[12] = VMULADDL(a[12], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[13] = VADD(a[13], hi);
    a[14] = VADDMASK(a[14], 0x55, a[14], VSHUFFLE(a[8], 0x4e));
    
    carry = VSRA(a[9], LIMB_LEN);
    a[9] = VAND(a[9], limb_mask);
    a[10] = VADD(a[10], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[9], 0x4e);
    a[13] = VMULADDL(a[13], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[14] = VADD(a[14], hi);
    a[15] = VADDMASK(a[15], 0x55, a[15], VSHUFFLE(a[9], 0x4e));
    
    carry = VSRA(a[10], LIMB_LEN);
    a[10] = VAND(a[10], limb_mask);
    a[11] = VADD(a[11], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[10], 0x4e);
    a[14] = VMULADDL(a[14], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[15] = VADD(a[15], hi);
    a[16] = VADDMASK(a[16], 0x55, a[16], VSHUFFLE(a[10], 0x4e));
    
    carry = VSRA(a[11], LIMB_LEN);
    a[11] = VAND(a[11], limb_mask);
    a[12] = VADD(a[12], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[11], 0x4e);
    a[15] = VMULADDL(a[15], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[16] = VADD(a[16], hi);
    a[17] = VADDMASK(a[17], 0x55, a[17], VSHUFFLE(a[11], 0x4e));
    
    // Result is now in a[12], ..., a[17]
    // Carry propagation
    a[13] = VADDMASK(a[13], 0x55, a[13], VSRA(a[12], LIMB_LEN));
    a[14] = VADDMASK(a[14], 0x55, a[14], VSRA(a[13], LIMB_LEN));
    a[15] = VADDMASK(a[15], 0x55, a[15], VSRA(a[14], LIMB_LEN));
    a[16] = VADDMASK(a[16], 0x55, a[16], VSRA(a[15], LIMB_LEN));
    a[17] = VADDMASK(a[17], 0x55, a[17], VSRA(a[16], LIMB_LEN));
    carry = VSRA(a[17], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    a[12] = VADDMASK(a[12], 0xaa, a[12], carry);
    a[13] = VADDMASK(a[13], 0xaa, a[13], VSRA(a[12], LIMB_LEN));
    a[14] = VADDMASK(a[14], 0xaa, a[14], VSRA(a[13], LIMB_LEN));
    a[15] = VADDMASK(a[15], 0xaa, a[15], VSRA(a[14], LIMB_LEN));
    a[16] = VADDMASK(a[16], 0xaa, a[16], VSRA(a[15], LIMB_LEN));
    a[12] = VAND(a[12], limb_mask);
    a[13] = VAND(a[13], limb_mask);
    a[14] = VAND(a[14], limb_mask);
    a[15] = VAND(a[15], limb_mask);
    a[16] = VANDMASK(a[16], 0x55, a[16], limb_mask);
    a[17] = VANDMASK(a[17], 0x55, a[17], limb_mask);
    
    // The result can be 522 bits.
    // Output result: final carry propagation is left for the final result.
    res[0] = a[12];
    res[1] = a[13];
    res[2] = a[14];
    res[3] = a[15];
    res[4] = a[16];
    res[5] = a[17];
}

void fp_mul_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[18];
    
    mp_mul_4x2(t, a, b);
    rdc_4x2(res, t);
}

void fp_sqr_4x2(__m512i* res, __m512i* a)
{
    fp_mul_4x2(res, a, a);
}

void fp_div3_4x2(__m512i* res, __m512i* src)
{
    __m512i t[6];
    fp_duplicate_4x2(t, INV3);
    fp_mul_4x2(res, src, t);
}

// Batched inversion.
void fp_inv_4x2(__m512i* res, __m512i* a)
{
    uint64_t r_64[18];
    __m512i s0[3], s1[3], r[3];
    __m512i t0[6], t1[6];
    __m512i idx0, idx1;
    
    // s0 = (a0*a1, a2*a3)
    fp_4x2_to_2x4(s0, s1, a);
    fp_mul_2x4(s0, s0, s1);
    
    // r = a0a1a2a3 (wastes one mul)
    idx1 = VSET(4, 5, 6, 7, 0, 0, 0, 0);
    for (int i = 0; i < 3; i++) {
        s1[i] = VPERM512(idx1, s0[i]);
    }
    fp_mul_2x4(r, s0, s1);
    
    // Computing r^-1 mod p
    fp_limbs_to_64_2x4(r_64, r);
    fp_inv_64(r_64, r_64);
    fp_64_to_limbs_2x4(r, r_64);
    
    // s0 = (inv_01, inv_23)
    idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < 3; i++) {
        s0[i] = VPERM512(idx0, s0[i]);
        s1[i] = VPERM512(idx1, r[i]);
    }
    fp_mul_2x4(s0, s0, s1);
    
    // res = (inv0, inv1, inv2, inv3)
    fp_2x4_to_4x2(t0, s0, s0); // (inv_01, inv_01, inv_23, inv_23)
    idx1 = VSET(2, 3, 0, 1, 6, 7, 4, 5);
    for (int i = 0; i < 6; i++) {
        t1[i] = VPERM512(idx1, a[i]);
    }
    fp_mul_4x2(res, t0, t1);
}

void fp_tomont_4x2(__m512i* res, __m512i* a)
{
    __m512i vR2[6];
    fp_duplicate_4x2(vR2, R2);
    fp_mul_4x2(res, a, vR2);
}

void fp_frommont_4x2(__m512i* res, __m512i* a)
{
    __m512i one[6];
    fp_setzero_4x2(one);
    one[0] = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    fp_mul_4x2(res, a, one);
}

// Assumes unvec is an array of 4 * NWORDS64 64-bit ints
void fp_64_to_limbs_4x2(__m512i* res, uint64_t* unvec)
{
    uint64_t t[8];
    
    for (int i = 0; i < 4; i++) {
        t[2*i] = unvec[NWORDS64 * i] & LIMB_MASK;
        t[2*i+1] = ((unvec[NWORDS64 * i + 4] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 5] & 0xffffULL) << 32);
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = ((unvec[NWORDS64 * i] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 1] & 0xffffffffULL) << 16);
        t[2*i+1] = (unvec[NWORDS64 * i + 5] >> 16) & LIMB_MASK;
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = ((unvec[NWORDS64 * i + 1] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 2] & 0xffffULL) << 32);
        t[2*i+1] = unvec[NWORDS64 * i + 6] & LIMB_MASK;
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = (unvec[NWORDS64 * i + 2] >> 16) & LIMB_MASK;
        t[2*i+1] = ((unvec[NWORDS64 * i + 6] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 7] & 0xffffffffULL) << 16);
    }
    res[3] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = unvec[NWORDS64 * i + 3] & LIMB_MASK;
        t[2*i+1] = (unvec[NWORDS64 * i + 7] >> 32) & LIMB_MASK;
    }
    res[4] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = ((unvec[NWORDS64 * i + 3] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 4] & 0xffffffffULL) << 16);
        t[2*i+1] = 0;
    }
    res[5] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

// Assumes src is fully redduced
void fp_limbs_to_64_4x2(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[6], zero[6];
    // Normalize input
    fp_setzero_4x2(zero);
    fp_add_4x2(s, src, zero);
    
    
    VSTORE(t, s[0]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i] = t[2*i] & LIMB_MASK;
        
        res[NWORDS64 * i + 4] = (t[2*i+1] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 5] = (t[2*i+1] >> 32) & 0xffffULL;
    }
    VSTORE(t, s[1]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i] |= (t[2*i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 1] = (t[2*i] >> 16) & 0xffffffffULL;
        
        res[NWORDS64 * i + 5] |= (t[2*i+1] & LIMB_MASK) << 16;
    }
    VSTORE(t, s[2]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 1] |= (t[2*i] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 2] = (t[2*i] >> 32) & 0xffffULL;
        
        res[NWORDS64 * i + 6] = t[2*i+1] & LIMB_MASK;
    }
    VSTORE(t, s[3]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 2] |= (t[2*i] & LIMB_MASK) << 16;
        
        res[NWORDS64 * i + 6] |= (t[2*i+1] & 0xffffULL) << 48;
        res[NWORDS64 * i + 7] = (t[2*i+1] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, s[4]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 3] = t[2*i] & LIMB_MASK;
        
        res[NWORDS64 * i + 7] |= (t[2*i+1] & LIMB_MASK) << 32;
    }
    VSTORE(t, s[5]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 3] |= (t[2*i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 4] |= (t[2*i] >> 16) & 0xffffffffULL;
        
    }
}

void rand_fp_4x2(__m512i* res)
{
    __m512i zero[6];
    __m512i mask = VSETALL(LIMB_MASK);
    fp_setzero_4x2(zero);
    
    for (int i = 0; i < 4; i++) {
        res[i] = VAND(rand_512i(), mask);
    }
    
    // Last limb is shorter
    mask = VSET(LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK);
    res[4] = VAND(rand_512i(), mask);
    
    mask = VSET(LIMB_MASK, 0, LIMB_MASK, 0, LIMB_MASK, 0, LIMB_MASK, 0);
    res[5] = VAND(rand_512i(), mask);
    
    // Reduction
    fp_add_4x2(res, res, zero);
}

// ====================================================================================================
//                                                 2x4                                                
// ====================================================================================================

void fp_setzero_2x4(__m512i* res)
{
    res[0] = VZERO();
    res[1] = VZERO();
    res[2] = VZERO();
}

void fp_copy_2x4(__m512i* res, __m512i* a)
{
    res[0] = a[0];
    res[1] = a[1];
    res[2] = a[2];
}

void fp_duplicate_2x4(__m512i* res, const uint64_t* a)
{
    res[0] = VSET(a[0], a[3], a[6], a[9], a[0], a[3], a[6], a[9]);
    res[1] = VSET(a[1], a[4], a[7], a[10], a[1], a[4], a[7], a[10]);
    res[2] = VSET(a[2], a[5], a[8], 0, a[2], a[5], a[8], 0);
}

int fp_is_equal_2x4(__m512i* a, __m512i* b)
{
    __m512i zero[3], norm_a[3], norm_b[3];
    fp_setzero_2x4(zero);
    
    // Normalize both to [0, p-1] by adding zero
    fp_add_2x4(norm_a, a, zero);
    fp_add_2x4(norm_b, b, zero);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], norm_b[0]);
    acc &= VCMPEQ(norm_a[1], norm_b[1]);
    acc &= VCMPEQ(norm_a[2], norm_b[2]);
    
    return (acc & (acc >> 1) & (acc >> 2) & (acc >> 3) & 0x1)
        | ((acc >> 3) & (acc >> 4) & (acc >> 5) & (acc >> 6) & 0x2);
}

int fp_is_zero_2x4(__m512i* a)
{
    __m512i zero[3], norm_a[3];
    fp_setzero_2x4(zero);
    
    // Normalize a to [0, p-1] by adding zero
    fp_add_2x4(norm_a, a, zero);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], zero[0]);
    acc &= VCMPEQ(norm_a[1], zero[1]);
    acc &= VCMPEQ(norm_a[2], zero[2]);
    
    return (acc & (acc >> 1) & (acc >> 2) & (acc >> 3) & 0x1)
        | ((acc >> 3) & (acc >> 4) & (acc >> 5) & (acc >> 6) & 0x2);
}

void fp_select_2x4(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    __m512i t[3];
    __mmask8 bit0 = s & 0x1, bit1 = (s >> 1) & 0x1;
    __mmask8 mask = bit0 | (bit0 << 1) | (bit0 << 2) | (bit0 << 3)
        | (bit1 << 4) | (bit1 << 5) | (bit1 << 6) | (bit1 << 7);
    
    t[0] = VBLEND(mask, a0[0], a1[0]);
    t[1] = VBLEND(mask, a0[1], a1[1]);
    t[2] = VBLEND(mask, a0[2], a1[2]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void fp_add_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[3], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, 0, P_PLUS_1_TOP, 0, 0, 0, P_PLUS_1_TOP);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6)
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // Broadcasts the sign bits from lanes 3 and 7 to their respective elements
    const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);
    
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    
    // Subtract p using the special shape of p
    t[0] = VADD(t[0], p_bot);
    t[2] = VSUB(t[2], p_top);
    
    // Carry propagation (3 full cross-lane ripples to cross all 4 lanes)
    // Straight ripple through the vectors
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    // Cross-lane ripple from the last vector back to the first
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple to finish the exact (4 * n24) limb sequence
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t[2], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[2] = VADD(t[2], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (3 full cross-lane ripples to cross all 4 lanes)
    // Straight ripple through the vectors
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    // Cross-lane ripple from the last vector back to the first
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple to finish the exact (4 * n24) limb sequence
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void fp_sub_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[3], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, 0, P_PLUS_1_TOP, 0, 0, 0, P_PLUS_1_TOP);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6)
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // Broadcasts the sign bits from lanes 3 and 7 to their respective elements
    const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);
    
    t[0] = VSUB(a[0], b[0]);
    t[1] = VSUB(a[1], b[1]);
    t[2] = VSUB(a[2], b[2]);
    
    // Carry propagation (3 full cross-lane ripples to cross all 4 lanes)
    // Straight ripple through the vectors
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    // Cross-lane ripple from the last vector back to the first
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple to finish the exact (4 * n24) limb sequence
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t[2], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[2] = VADD(t[2], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (3 full cross-lane ripples to cross all 4 lanes)
    // Straight ripple through the vectors
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    // Cross-lane ripple from the last vector back to the first
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple to finish the exact (4 * n24) limb sequence
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void fp_neg_2x4(__m512i* res, __m512i* a)
{
    __m512i t[3], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, 0, P_PLUS_1_TOP, 0, 0, 0, P_PLUS_1_TOP);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6)
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // Broadcasts the sign bits from lanes 3 and 7 to their respective elements
    const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);
    const __m512i zero = VZERO();
    
    t[0] = VSUB(zero, a[0]);
    t[1] = VSUB(zero, a[1]);
    t[2] = VSUB(zero, a[2]);
    
    // Carry propagation (3 full cross-lane ripples to cross all 4 lanes)
    // Straight ripple through the vectors
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    // Cross-lane ripple from the last vector back to the first
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple to finish the exact (4 * n24) limb sequence
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t[2], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[2] = VADD(t[2], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (3 full cross-lane ripples to cross all 4 lanes)
    // Straight ripple through the vectors
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    // Cross-lane ripple from the last vector back to the first
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple to finish the exact (4 * n24) limb sequence
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void fp_mixaddsub_2x4(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[3], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, 0, P_PLUS_1_TOP, 0, 0, 0, P_PLUS_1_TOP);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6)
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // Broadcasts the sign bits from lanes 3 and 7 to their respective elements
    const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);
    const __mmask8 sub_mask = ~add_mask;
    
    t[0] = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t[1] = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t[2] = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    t[2] = VADDMASK(t[2], add_mask, a[2], b[2]);
    // Subtract p using the special shape of p
    t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);
    t[2] = VSUBMASK(t[2], add_mask, t[2], p_top);
    
    // Carry propagation (3 full cross-lane ripples to cross all 4 lanes)
    // Straight ripple through the vectors
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    // Cross-lane ripple from the last vector back to the first
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple to finish the exact (4 * n24) limb sequence
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t[2], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[2] = VADD(t[2], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (3 full cross-lane ripples to cross all 4 lanes)
    // Straight ripple through the vectors
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    // Cross-lane ripple from the last vector back to the first
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    carry = VSRA(t[2], LIMB_LEN);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple to finish the exact (4 * n24) limb sequence
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VADD(t[2], carry);
    t[2] = VANDMASK(t[2], 0x77, t[2], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void mp_mul_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[14];
    __m512i B[11];
    const __m512i zero = VZERO();
    
    // Create a vector for each limb of b
    B[0] = VPERM256(b[0], 0x0);
    B[1] = VPERM256(b[1], 0x0);
    B[2] = VPERM256(b[2], 0x0);
    B[3] = VPERM256(b[0], 0x55);
    B[4] = VPERM256(b[1], 0x55);
    B[5] = VPERM256(b[2], 0x55);
    B[6] = VPERM256(b[0], 0xaa);
    B[7] = VPERM256(b[1], 0xaa);
    B[8] = VPERM256(b[2], 0xaa);
    B[9] = VPERM256(b[0], 0xff);
    B[10] = VPERM256(b[1], 0xff);
    
    // Schoolbook
    t[0] = VMULADDL(zero, a[0], B[0]);
    t[1] = VMULADDH(zero, a[0], B[0]);
    t[1] = VSL(t[1], LIMB_ALIGN);
    
    t[1] = VMULADDL(t[1], a[0], B[1]);
    t[2] = VMULADDH(zero, a[0], B[1]);
    t[1] = VMULADDL(t[1], a[1], B[0]);
    t[2] = VMULADDH(t[2], a[1], B[0]);
    t[2] = VSL(t[2], LIMB_ALIGN);
    
    t[2] = VMULADDL(t[2], a[0], B[2]);
    t[3] = VMULADDH(zero, a[0], B[2]);
    t[2] = VMULADDL(t[2], a[1], B[1]);
    t[3] = VMULADDH(t[3], a[1], B[1]);
    t[2] = VMULADDL(t[2], a[2], B[0]);
    t[3] = VMULADDH(t[3], a[2], B[0]);
    t[3] = VSL(t[3], LIMB_ALIGN);
    
    t[3] = VMULADDL(t[3], a[0], B[3]);
    t[4] = VMULADDH(zero, a[0], B[3]);
    t[3] = VMULADDL(t[3], a[1], B[2]);
    t[4] = VMULADDH(t[4], a[1], B[2]);
    t[3] = VMULADDL(t[3], a[2], B[1]);
    t[4] = VMULADDH(t[4], a[2], B[1]);
    t[4] = VSL(t[4], LIMB_ALIGN);
    
    t[4] = VMULADDL(t[4], a[0], B[4]);
    t[5] = VMULADDH(zero, a[0], B[4]);
    t[4] = VMULADDL(t[4], a[1], B[3]);
    t[5] = VMULADDH(t[5], a[1], B[3]);
    t[4] = VMULADDL(t[4], a[2], B[2]);
    t[5] = VMULADDH(t[5], a[2], B[2]);
    t[5] = VSL(t[5], LIMB_ALIGN);
    
    t[5] = VMULADDL(t[5], a[0], B[5]);
    t[6] = VMULADDH(zero, a[0], B[5]);
    t[5] = VMULADDL(t[5], a[1], B[4]);
    t[6] = VMULADDH(t[6], a[1], B[4]);
    t[5] = VMULADDL(t[5], a[2], B[3]);
    t[6] = VMULADDH(t[6], a[2], B[3]);
    t[6] = VSL(t[6], LIMB_ALIGN);
    
    t[6] = VMULADDL(t[6], a[0], B[6]);
    t[7] = VMULADDH(zero, a[0], B[6]);
    t[6] = VMULADDL(t[6], a[1], B[5]);
    t[7] = VMULADDH(t[7], a[1], B[5]);
    t[6] = VMULADDL(t[6], a[2], B[4]);
    t[7] = VMULADDH(t[7], a[2], B[4]);
    t[7] = VSL(t[7], LIMB_ALIGN);
    
    t[7] = VMULADDL(t[7], a[0], B[7]);
    t[8] = VMULADDH(zero, a[0], B[7]);
    t[7] = VMULADDL(t[7], a[1], B[6]);
    t[8] = VMULADDH(t[8], a[1], B[6]);
    t[7] = VMULADDL(t[7], a[2], B[5]);
    t[8] = VMULADDH(t[8], a[2], B[5]);
    t[8] = VSL(t[8], LIMB_ALIGN);
    
    t[8] = VMULADDL(t[8], a[0], B[8]);
    t[9] = VMULADDH(zero, a[0], B[8]);
    t[8] = VMULADDL(t[8], a[1], B[7]);
    t[9] = VMULADDH(t[9], a[1], B[7]);
    t[8] = VMULADDL(t[8], a[2], B[6]);
    t[9] = VMULADDH(t[9], a[2], B[6]);
    t[9] = VSL(t[9], LIMB_ALIGN);
    
    t[9] = VMULADDL(t[9], a[0], B[9]);
    t[10] = VMULADDH(zero, a[0], B[9]);
    t[9] = VMULADDL(t[9], a[1], B[8]);
    t[10] = VMULADDH(t[10], a[1], B[8]);
    t[9] = VMULADDL(t[9], a[2], B[7]);
    t[10] = VMULADDH(t[10], a[2], B[7]);
    t[10] = VSL(t[10], LIMB_ALIGN);
    
    t[10] = VMULADDL(t[10], a[0], B[10]);
    t[11] = VMULADDH(zero, a[0], B[10]);
    t[10] = VMULADDL(t[10], a[1], B[9]);
    t[11] = VMULADDH(t[11], a[1], B[9]);
    t[10] = VMULADDL(t[10], a[2], B[8]);
    t[11] = VMULADDH(t[11], a[2], B[8]);
    t[11] = VSL(t[11], LIMB_ALIGN);
    
    t[11] = VMULADDL(t[11], a[1], B[10]);
    t[12] = VMULADDH(zero, a[1], B[10]);
    t[11] = VMULADDL(t[11], a[2], B[9]);
    t[12] = VMULADDH(t[12], a[2], B[9]);
    t[12] = VSL(t[12], LIMB_ALIGN);
    
    t[12] = VMULADDL(t[12], a[2], B[10]);
    t[13] = VMULADDH(zero, a[2], B[10]);
    t[13] = VSL(t[13], LIMB_ALIGN);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
    res[9] = t[9];
    res[10] = t[10];
    res[11] = t[11];
    res[12] = t[12];
    res[13] = t[13];
}

// Modfies a. Its length must be 23.
void rdc_2x4(__m512i* res, __m512i* a)
{
    __m512i carry, lo, hi, shuf;
    const __m512i zero = VZERO();
    const __m512i one = VSETALL(1);
    const __m512i p_top = VSETALL(P_PLUS_1_TOP);
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    __mmask8 msb_mask;
    
    // Initialize the upper accumulators to zero
    a[14] = zero;
    a[15] = zero;
    a[16] = zero;
    a[17] = zero;
    a[18] = zero;
    a[19] = zero;
    a[20] = zero;
    a[21] = zero;
    a[22] = zero;
    
    // Move each limb one step to the left, for each half
    const __m512i idx = VSET(1, 2, 3, 0, 5, 6, 7, 0);
    
    // Generalized Montgomery reduction loop
    carry = VSRA(a[0], LIMB_LEN);
    a[0] = VAND(a[0], limb_mask);
    a[1] = VADD(a[1], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[0], 0x4e);
    a[7] = VMULADDL(a[7], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[8] = VADD(a[8], VSL(hi, LIMB_ALIGN));
    a[3] = VADD(a[3], VPERM512MASKZ(0x77, a[0], idx));
    
    carry = VSRA(a[1], LIMB_LEN);
    a[1] = VAND(a[1], limb_mask);
    a[2] = VADD(a[2], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[1], 0x4e);
    a[8] = VMULADDL(a[8], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[9] = VADD(a[9], VSL(hi, LIMB_ALIGN));
    a[4] = VADD(a[4], VPERM512MASKZ(0x77, a[1], idx));
    
    carry = VSRA(a[2], LIMB_LEN);
    a[2] = VAND(a[2], limb_mask);
    a[3] = VADD(a[3], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[2], 0x4e);
    a[9] = VMULADDL(a[9], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[10] = VADD(a[10], VSL(hi, LIMB_ALIGN));
    a[5] = VADD(a[5], VPERM512MASKZ(0x77, a[2], idx));
    
    carry = VSRA(a[3], LIMB_LEN);
    a[3] = VAND(a[3], limb_mask);
    a[4] = VADD(a[4], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[3], 0x4e);
    a[10] = VMULADDL(a[10], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[11] = VADD(a[11], VSL(hi, LIMB_ALIGN));
    a[6] = VADD(a[6], VPERM512MASKZ(0x77, a[3], idx));
    
    carry = VSRA(a[4], LIMB_LEN);
    a[4] = VAND(a[4], limb_mask);
    a[5] = VADD(a[5], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[4], 0x4e);
    a[11] = VMULADDL(a[11], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[12] = VADD(a[12], VSL(hi, LIMB_ALIGN));
    a[7] = VADD(a[7], VPERM512MASKZ(0x77, a[4], idx));
    
    carry = VSRA(a[5], LIMB_LEN);
    a[5] = VAND(a[5], limb_mask);
    a[6] = VADD(a[6], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[5], 0x4e);
    a[12] = VMULADDL(a[12], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[13] = VADD(a[13], VSL(hi, LIMB_ALIGN));
    a[8] = VADD(a[8], VPERM512MASKZ(0x77, a[5], idx));
    
    carry = VSRA(a[6], LIMB_LEN);
    a[6] = VAND(a[6], limb_mask);
    a[7] = VADD(a[7], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[6], 0x4e);
    a[13] = VMULADDL(a[13], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[14] = VADD(a[14], VSL(hi, LIMB_ALIGN));
    a[9] = VADD(a[9], VPERM512MASKZ(0x77, a[6], idx));
    
    carry = VSRA(a[7], LIMB_LEN);
    a[7] = VAND(a[7], limb_mask);
    a[8] = VADD(a[8], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[7], 0x4e);
    a[14] = VMULADDL(a[14], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[15] = VADD(a[15], VSL(hi, LIMB_ALIGN));
    a[10] = VADD(a[10], VPERM512MASKZ(0x77, a[7], idx));
    
    carry = VSRA(a[8], LIMB_LEN);
    a[8] = VAND(a[8], limb_mask);
    a[9] = VADD(a[9], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[8], 0x4e);
    a[15] = VMULADDL(a[15], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[16] = VADD(a[16], VSL(hi, LIMB_ALIGN));
    a[11] = VADD(a[11], VPERM512MASKZ(0x77, a[8], idx));
    
    carry = VSRA(a[9], LIMB_LEN);
    a[9] = VAND(a[9], limb_mask);
    a[10] = VADD(a[10], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[9], 0x4e);
    a[16] = VMULADDL(a[16], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[17] = VADD(a[17], VSL(hi, LIMB_ALIGN));
    a[12] = VADD(a[12], VPERM512MASKZ(0x77, a[9], idx));
    
    carry = VSRA(a[10], LIMB_LEN);
    a[10] = VAND(a[10], limb_mask);
    a[11] = VADD(a[11], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[10], 0x4e);
    a[17] = VMULADDL(a[17], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[18] = VADD(a[18], VSL(hi, LIMB_ALIGN));
    a[13] = VADD(a[13], VPERM512MASKZ(0x77, a[10], idx));
    
    carry = VSRA(a[11], LIMB_LEN);
    a[11] = VAND(a[11], limb_mask);
    a[12] = VADD(a[12], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[11], 0x4e);
    a[18] = VMULADDL(a[18], shuf, p_top);
    hi = VMULADDH(zero, shuf, p_top);
    a[19] = VADD(a[19], VSL(hi, LIMB_ALIGN));
    a[14] = VADD(a[14], VPERM512MASKZ(0x77, a[11], idx));
    
    // Propagate carries
    carry = VSRA(a[12], LIMB_LEN);
    a[12] = VAND(a[12], limb_mask);
    a[13] = VADD(a[13], carry);
    a[15] = VADD(a[15], VPERM512MASKZ(0x77, a[12], idx));
    
    carry = VSRA(a[13], LIMB_LEN);
    a[13] = VAND(a[13], limb_mask);
    a[14] = VADD(a[14], carry);
    a[16] = VADD(a[16], VPERM512MASKZ(0x77, a[13], idx));
    
    carry = VSRA(a[14], LIMB_LEN);
    a[14] = VAND(a[14], limb_mask);
    a[15] = VADD(a[15], carry);
    a[17] = VADD(a[17], VPERM512MASKZ(0x77, a[14], idx));
    
    carry = VSRA(a[15], LIMB_LEN);
    a[15] = VAND(a[15], limb_mask);
    a[16] = VADD(a[16], carry);
    a[18] = VADD(a[18], VPERM512MASKZ(0x77, a[15], idx));
    
    carry = VSRA(a[16], LIMB_LEN);
    a[16] = VAND(a[16], limb_mask);
    a[17] = VADD(a[17], carry);
    a[19] = VADD(a[19], VPERM512MASKZ(0x77, a[16], idx));
    
    carry = VSRA(a[17], LIMB_LEN);
    a[17] = VAND(a[17], limb_mask);
    a[18] = VADD(a[18], carry);
    a[20] = VADD(a[20], VPERM512MASKZ(0x77, a[17], idx));
    
    carry = VSRA(a[18], LIMB_LEN);
    a[18] = VAND(a[18], limb_mask);
    a[19] = VADD(a[19], carry);
    a[21] = VADD(a[21], VPERM512MASKZ(0x77, a[18], idx));
    
    carry = VSRA(a[19], LIMB_LEN);
    a[19] = VAND(a[19], limb_mask);
    a[20] = VADD(a[20], carry);
    a[22] = VADD(a[22], VPERM512MASKZ(0x77, a[19], idx));
    
    carry = VSRA(a[20], LIMB_LEN);
    a[20] = VAND(a[20], limb_mask);
    a[21] = VADD(a[21], carry);
    
    carry = VSRA(a[21], LIMB_LEN);
    a[21] = VAND(a[21], limb_mask);
    a[22] = VADD(a[22], carry);
    
    // Final reduction step to get the result down to the strict bit boundary
    // Pack result from pos 0 of the accumulators back into 2x4 format
    const __m512i idx_pack = VSET(0, 1, 8, 9, 4, 5, 12, 13);
    lo = VUNPACKLO(a[12], a[15]);
    hi = VUNPACKLO(a[18], a[21]);
    res[0] = VBLENDPERM(idx_pack, lo, hi);
    
    lo = VUNPACKLO(a[13], a[16]);
    hi = VUNPACKLO(a[19], a[22]);
    res[1] = VBLENDPERM(idx_pack, lo, hi);
    
    lo = VUNPACKLO(a[14], a[17]);
    hi = VUNPACKLO(a[20], a[23]);
    res[2] = VBLENDPERM(idx_pack, lo, hi);
}

void fp_mul_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[23];
    
    mp_mul_2x4(t, a, b);
    rdc_2x4(res, t);
}

void fp_sqr_2x4(__m512i* res, __m512i* a)
{
    __m512i t[23];
    
    mp_mul_2x4(t, a, a);
    rdc_2x4(res, t);
}

void fp_tomont_2x4(__m512i* res, __m512i* a)
{
    __m512i vR2[3];
    fp_duplicate_2x4(vR2, R2);
    fp_mul_2x4(res, a, vR2);
}

void fp_frommont_2x4(__m512i* res, __m512i* a)
{
    __m512i one[3];
    fp_setzero_2x4(one);
    one[0] = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    fp_mul_2x4(res, a, one);
}

void fp_div3_2x4(__m512i* res, __m512i* src)
{
    __m512i t[3];
    fp_duplicate_2x4(t, INV3);
    fp_mul_2x4(res, src, t);
}

// Batched inversion.
void fp_inv_2x4(__m512i* res, __m512i* a)
{
    uint64_t r_64[18];
    __m512i s[3], r[3];
    __m512i idx0, idx1;
    
    // r = a0a1 (wastes one mul)
    idx1 = VSET(4, 5, 6, 7, 0, 0, 0, 0);
    for (int i = 0; i < 3; i++) {
        r[i] = VPERM512(idx1, a[i]);
    }
    fp_mul_2x4(r, r, a);
    
    // Computing r^-1 mod p
    fp_limbs_to_64_2x4(r_64, r);
    fp_inv_64(r_64, r_64);
    fp_64_to_limbs_2x4(r, r_64);
    
    // res = (inv_0, inv_1)
    idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);
    idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    for (int i = 0; i < 3; i++) {
        s[i] = VPERM512(idx0, a[i]);
        r[i] = VPERM512(idx1, r[i]);
    }
    fp_mul_2x4(res, r, s);
    
}

void fp_sqrt_2x4(__m512i* res, __m512i* a)
{
    // Candidate root is a^((p+1)/4), with (p+1)/4 = 1*2^519
    __m512i t[3];
    
    fp_sqr_2x4(t, a);
    for (int i = 0; i < 517; i++) {
        fp_sqr_2x4(t, t);
    }
    fp_sqr_2x4(res, t);
}

// Cost: 518 squarings, 12 multiplications
void fp_exp3div4_2x4(__m512i* res, __m512i* a)
{
    __m512i v__10[3];
    __m512i v__11[3];
    __m512i v__1100[3];
    __m512i v__1111[3];
    __m512i v__11110000[3];
    __m512i v__11111111[3];
    __m512i v_x129[3];
    __m512i v_x130[3];
    __m512i v_x16[3];
    __m512i v_x259[3];
    __m512i v_x260[3];
    __m512i v_x32[3];
    __m512i v_x64[3];
    __m512i v_x65[3];
    
    fp_sqr_2x4(v__10, a);
    fp_mul_2x4(v__11, a, v__10);
    fp_sqr_2x4(v__1100, v__11);
    fp_sqr_2x4(v__1100, v__1100);
    fp_mul_2x4(v__1111, v__11, v__1100);
    fp_sqr_2x4(v__11110000, v__1111);
    fp_sqr_2x4(v__11110000, v__11110000);
    fp_sqr_2x4(v__11110000, v__11110000);
    fp_sqr_2x4(v__11110000, v__11110000);
    fp_mul_2x4(v__11111111, v__1111, v__11110000);
    fp_sqr_2x4(v_x16, v__11111111);
    for (int i = 0; i < 7; i++) fp_sqr_2x4(v_x16, v_x16);
    fp_mul_2x4(v_x16, v_x16, v__11111111);
    fp_sqr_2x4(v_x32, v_x16);
    for (int i = 0; i < 15; i++) fp_sqr_2x4(v_x32, v_x32);
    fp_mul_2x4(v_x32, v_x32, v_x16);
    fp_sqr_2x4(v_x64, v_x32);
    for (int i = 0; i < 31; i++) fp_sqr_2x4(v_x64, v_x64);
    fp_mul_2x4(v_x64, v_x64, v_x32);
    fp_sqr_2x4(v_x65, v_x64);
    fp_mul_2x4(v_x65, v_x65, a);
    fp_sqr_2x4(v_x129, v_x65);
    for (int i = 0; i < 63; i++) fp_sqr_2x4(v_x129, v_x129);
    fp_mul_2x4(v_x129, v_x129, v_x64);
    fp_sqr_2x4(v_x130, v_x129);
    fp_mul_2x4(v_x130, v_x130, a);
    fp_sqr_2x4(v_x259, v_x130);
    for (int i = 0; i < 128; i++) fp_sqr_2x4(v_x259, v_x259);
    fp_mul_2x4(v_x259, v_x259, v_x129);
    fp_sqr_2x4(v_x260, v_x259);
    fp_mul_2x4(v_x260, v_x260, a);
    fp_sqr_2x4(res, v_x260);
    for (int i = 0; i < 258; i++) fp_sqr_2x4(res, res);
    fp_mul_2x4(res, res, v_x259);
}

// Assumes unvec is an array of 2 * NWORDS64 64-bit ints
void fp_64_to_limbs_2x4(__m512i* res, uint64_t* unvec)
{
    uint64_t t[8];
    
    for (int i = 0; i < 2; i++) {
        t[4*i] = unvec[NWORDS64 * i] & LIMB_MASK;
        t[4*i + 1] = (unvec[NWORDS64 * i + 2] >> 16) & LIMB_MASK;
        t[4*i + 2] = ((unvec[NWORDS64 * i + 4] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 5] & 0xffffULL) << 32);
        t[4*i+3] = ((unvec[NWORDS64 * i + 6] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 7] & 0xffffffffULL) << 16);
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 2; i++) {
        t[4*i] = ((unvec[NWORDS64 * i] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 1] & 0xffffffffULL) << 16);
        t[4*i + 1] = unvec[NWORDS64 * i + 3] & LIMB_MASK;
        t[4*i + 2] = (unvec[NWORDS64 * i + 5] >> 16) & LIMB_MASK;
        t[4*i+3] = ((unvec[NWORDS64 * i + 7] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 8] & 0xffffULL) << 32);
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 2; i++) {
        t[4*i] = ((unvec[NWORDS64 * i + 1] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 2] & 0xffffULL) << 32);
        t[4*i + 1] = ((unvec[NWORDS64 * i + 3] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 4] & 0xffffffffULL) << 16);
        t[4*i + 2] = unvec[NWORDS64 * i + 6] & LIMB_MASK;
        t[4*i+3] = (unvec[NWORDS64 * i + 8] >> 16) & LIMB_MASK;
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

void fp_limbs_to_64_2x4(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[3], zero[3];
    // Normalize input
    fp_setzero_2x4(zero);
    fp_add_2x4(s, src, zero);
    
    VSTORE(t, s[0]);
    for (int i = 0; i < 2; i++) {
        res[NWORDS64 * i] = t[4*i] & LIMB_MASK;
        res[NWORDS64 * i + 2] = (t[4*i + 1] & LIMB_MASK) << 16;
        res[NWORDS64 * i + 4] = (t[4*i + 2] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 5] = (t[4*i + 2] >> 32) & 0xffffULL;
        res[NWORDS64 * i + 6] = (t[4*i+3] & 0xffffULL) << 48;
        res[NWORDS64 * i + 7] = (t[4*i+3] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, s[1]);
    for (int i = 0; i < 2; i++) {
        res[NWORDS64 * i] |= (t[4*i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 1] = (t[4*i] >> 16) & 0xffffffffULL;
        res[NWORDS64 * i + 3] = t[4*i + 1] & LIMB_MASK;
        res[NWORDS64 * i + 5] |= (t[4*i + 2] & LIMB_MASK) << 16;
        res[NWORDS64 * i + 7] |= (t[4*i+3] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 8] = (t[4*i+3] >> 32) & 0xffffULL;
    }
    VSTORE(t, s[2]);
    for (int i = 0; i < 2; i++) {
        res[NWORDS64 * i + 1] |= (t[4*i] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 2] |= (t[4*i] >> 32) & 0xffffULL;
        res[NWORDS64 * i + 3] |= (t[4*i + 1] & 0xffffULL) << 48;
        res[NWORDS64 * i + 4] |= (t[4*i + 1] >> 16) & 0xffffffffULL;
        res[NWORDS64 * i + 6] |= t[4*i + 2] & LIMB_MASK;
        res[NWORDS64 * i + 8] |= (t[4*i+3] & LIMB_MASK) << 16;
    }
}

void rand_fp_2x4(__m512i* res)
{
    __m512i zero[3];
    __m512i mask = VSETALL(LIMB_MASK);
    fp_setzero_2x4(zero);
    
    for (int i = 0; i < 2; i++) {
        res[i] = VAND(rand_512i(), mask);
    }
    
    // Last limb is shorter
    mask = VSET(LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK);
    res[2] = VAND(rand_512i(), mask);
    
    // Reduction
    fp_add_2x4(res, res, zero);
}

// ====================================================================================================
//                                Conversions between Fp representations                               
// ====================================================================================================

// Given src = (a, b, c, d, e, f, g, h),
// it computes res1 = (a, c, e, g) and res2 = (b, d, f, h).
void fp_8x1_to_4x2(__m512i* res1, __m512i* res2, __m512i* src)
{
    __m512i s, t;
    const __m512i idx1 = VSET(0, 8, 2, 10, 4, 12, 6, 14);
    const __m512i idx2 = VSET(1, 9, 3, 11, 5, 13, 7, 15);
    for (int i = 0; i < 5; i++) {
        s = src[i], t = src[i + 6];
        res1[i] = VBLENDPERM(idx1, s, t);
        res2[i] = VBLENDPERM(idx2, s, t);
    }
    s = src[5];
    res1[5] = VBLENDPERM(idx1, s, VZERO());
    res2[5] = VBLENDPERM(idx2, s, VZERO());
}

// Given src1 = (a1, b1, c1, d1) and src2 = (a2, b2, c2, d2),
// it computes res = (a1, a2, b1, b2, c1, c2, d1, d2).
void fp_4x2_to_8x1(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i s, t;
    const __m512i idx1 = VSET(0, 8, 2, 10, 4, 12, 6, 14);
    const __m512i idx2 = VSET(1, 9, 3, 11, 5, 13, 7, 15);
    for (int i = 0; i < 5; i++) {
        s = src1[i], t = src2[i];
        res[i] = VBLENDPERM(idx1, s, t);
        res[i + 6] = VBLENDPERM(idx2, s, t);
    }
    s = src1[5], t = src2[5];
    res[5] = VBLENDPERM(idx1, s, t);
}

// Given src=(a, b, c, d), it computes res1 = (a, c) and res2 = (b, d).
// We choose this interleaving pattern for the Fp2 2x2x2 sqrt,
// where res1 = re and res2 = im.
void fp_4x2_to_2x4(__m512i* res1, __m512i* res2, __m512i* src)
{
    __m512i idx1 = VSET(0, 8, 1, 9, 4, 12, 5, 13);
    __m512i idx2 = VSET(2, 10, 3, 11, 6, 14, 7, 15);
    for (int i = 0; i < 3; i++) {
        __m512i s = src[i], t = src[i+3];
        res1[i] = VBLENDPERM(idx1, s, t);
        res2[i] = VBLENDPERM(idx2, s, t);
    }
}

void fp_2x4_to_4x2(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i idx1 = VSET(0, 2, 8, 10, 4, 6, 12, 14);
    __m512i idx2 = VSET(1, 3, 9, 11, 5, 7, 13, 15);
    for (int i = 0; i < 3; i++) {
        __m512i s = src1[i], t = src2[i];
        res[i] = VBLENDPERM(idx1, s, t);
        res[i+3] = VBLENDPERM(idx2, s, t);
    }
}

// ====================================================================================================
//                                      Unvectorized Fp inversion                                     
// ====================================================================================================

#define adc(cc, a, b, d) _addcarry_u64(cc, a, b, (unsigned long long *)(void *)d)
#define sbb(cc, a, b, d) _subborrow_u64(cc, a, b, (unsigned long long *)(void *)d)
#define lzcnt(x) ((x) == 0 ? 64 : (int64_t)__builtin_clzll(x))

// Outputs 0xffffffffffffffff if src is negative, else 0
uint64_t sgnw(const uint64_t src)
{
    return (uint64_t)(*(int64_t *)&src >> 63);
}

// Assumes input is fully reduced.
uint64_t is_zero64(const uint64_t* src)
{
    uint64_t t = src[0];
    t |= src[1];
    t |= src[2];
    t |= src[3];
    t |= src[4];
    t |= src[5];
    t |= src[6];
    t |= src[7];
    t |= src[8];
    
    t = (t | -t) >> 63; // 0 if t = 0, 1 otherwise
    return t-1;
}

// If mask == 0xffffffffffffffff, a1 is copied to res
void select64(uint64_t* res, const uint64_t* a0, const uint64_t* a1, const uint64_t mask)
{
    res[0] = a0[0] ^ (mask & (a0[0] ^ a1[0]));
    res[1] = a0[1] ^ (mask & (a0[1] ^ a1[1]));
    res[2] = a0[2] ^ (mask & (a0[2] ^ a1[2]));
    res[3] = a0[3] ^ (mask & (a0[3] ^ a1[3]));
    res[4] = a0[4] ^ (mask & (a0[4] ^ a1[4]));
    res[5] = a0[5] ^ (mask & (a0[5] ^ a1[5]));
    res[6] = a0[6] ^ (mask & (a0[6] ^ a1[6]));
    res[7] = a0[7] ^ (mask & (a0[7] ^ a1[7]));
    res[8] = a0[8] ^ (mask & (a0[8] ^ a1[8]));
}

// (lo, hi) = x*y as integers
void umul_64(uint64_t* lo, uint64_t* hi, const uint64_t x, const uint64_t y)
{
    unsigned __int128 t;
    t = (unsigned __int128)(x) * (unsigned __int128)(y);
    *lo = (uint64_t) t;
    *hi = (uint64_t)(t >> 64);
}

// (lo, hi) = x*y + z as integers
void umul_add_64(uint64_t* lo, uint64_t* hi, const uint64_t x, const uint64_t y, const uint64_t z)
{
    unsigned __int128 t;
    t = (unsigned __int128)(x) * (unsigned __int128)(y) + (unsigned __int128)(uint64_t)(z);
*lo = (uint64_t) t;
*hi = (uint64_t)(t >> 64);
}

// (lo, hi) = x1*y1 + x2*y2 as integers
void umulx2_64(uint64_t* lo, uint64_t* hi, const uint64_t x1, const uint64_t y1, const uint64_t x2, const uint64_t y2)
{
    unsigned __int128 t;
    t = (unsigned __int128)(x1) * (unsigned __int128)(y1) + (unsigned __int128)(x2) * (unsigned __int128)(y2);
    *lo = (uint64_t) t;
    *hi = (uint64_t)(t >> 64);
}

// (lo, hi) = x1*y1 + x2*y2 + z as integers
void umulx2_add_64(uint64_t* lo, uint64_t* hi, const uint64_t x1,
    const uint64_t y1, const uint64_t x2, const uint64_t y2, const uint64_t z)
{
    unsigned __int128 t;
    t = (unsigned __int128)(x1) * (unsigned __int128)(y1) + (unsigned __int128)(x2) * (unsigned __int128)(y2)
        + (unsigned __int128)(uint64_t)(z);
    *lo = (uint64_t) t;
    *hi = (uint64_t)(t >> 64);
}

// Assumes input is fully reduced.
void neg64(uint64_t* res, const uint64_t* src)
{
    uint64_t t[9];
    unsigned char cc;
    const uint64_t nz = ~is_zero64(src);
    
    // p - a
    cc = sbb(0, (uint64_t) 0xffffffffffffffff, src[0], t);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[1], t+1);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[2], t+2);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[3], t+3);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[4], t+4);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[5], t+5);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[6], t+6);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[7], t+7);
    cc = sbb(cc, (uint64_t) 0x1ff, src[8], t+8);
    
    res[0] = t[0] & nz;
    res[1] = t[1] & nz;
    res[2] = t[2] & nz;
    res[3] = t[3] & nz;
    res[4] = t[4] & nz;
    res[5] = t[5] & nz;
    res[6] = t[6] & nz;
    res[7] = t[7] & nz;
    res[8] = t[8] & nz;
}

// res = f*u + g*v mod p
// Assumes f,g < 2^62
void lin64(uint64_t* res, const uint64_t* u, const uint64_t* v, uint64_t f, uint64_t g)
{
    uint64_t t[9], tu[9], tv[9], hi;
    unsigned char cc;
    
    // If f is negative, set f := abs(f) and negate u
    uint64_t sf = sgnw(f);
    f = (f ^ sf) - sf;
    neg64(tu, u);
    select64(tu, u, tu, sf);
    
    // If g is negative, set g := abs(g) and negate v
    uint64_t sg = sgnw(g);
    g = (g ^ sg) - sg;
    neg64(tv, v);
    select64(tv, v, tv, sg);
    // t = f*u + g*v over the integers
    umulx2_64(t, &hi, tu[0], f, tv[0], g);
    umulx2_add_64(t+1, &hi, tu[1], f, tv[1], g, hi);
    umulx2_add_64(t+2, &hi, tu[2], f, tv[2], g, hi);
    umulx2_add_64(t+3, &hi, tu[3], f, tv[3], g, hi);
    umulx2_add_64(t+4, &hi, tu[4], f, tv[4], g, hi);
    umulx2_add_64(t+5, &hi, tu[5], f, tv[5], g, hi);
    umulx2_add_64(t+6, &hi, tu[6], f, tv[6], g, hi);
    umulx2_add_64(t+7, &hi, tu[7], f, tv[7], g, hi);
    umulx2_add_64(t+8, &hi, tu[8], f, tv[8], g, hi);
    
    // Reduction
    // t = L + 2^e H
    // Since p = c* 2^e - 1, 2^e = 1/c mod p
    // We fold the output as t = L + 2^e * (H mod c) + 2^e * floor(H/c)
    
    // Compute H in (h0, h1) and L in (t[0], ..., t[8]):
    uint64_t h0 = (t[8] >> 9) | (hi << 55);
    uint64_t h1 = (hi >> 9);
    t[8] &= 0x00000000000001FFULL;
    
    // Divide H by c
    __uint128_t H_full = ((__uint128_t)h1 << 64) | h0;
    uint64_t quo = (uint64_t)(H_full / 0x1ULL);
    uint64_t rem = (uint64_t)(H_full % 0x1ULL);
    
    // t = L + quo + rem * 2^e
    cc = adc(0, t[0], quo, &t[0]);
    cc = adc(cc, t[1], 0, &t[1]);
    cc = adc(cc, t[2], 0, &t[2]);
    cc = adc(cc, t[3], 0, &t[3]);
    cc = adc(cc, t[4], 0, &t[4]);
    cc = adc(cc, t[5], 0, &t[5]);
    cc = adc(cc, t[6], 0, &t[6]);
    cc = adc(cc, t[7], 0, &t[7]);
    adc(cc, t[8], rem << 9, &t[8]);
    
    // Finally, if t > p, we subtract p to reduce fully.
    cc = sbb(0, t[0], 0xffffffffffffffffULL, tu);
    cc = sbb(cc, t[1], 0xffffffffffffffffULL, tu + 1);
    cc = sbb(cc, t[2], 0xffffffffffffffffULL, tu + 2);
    cc = sbb(cc, t[3], 0xffffffffffffffffULL, tu + 3);
    cc = sbb(cc, t[4], 0xffffffffffffffffULL, tu + 4);
    cc = sbb(cc, t[5], 0xffffffffffffffffULL, tu + 5);
    cc = sbb(cc, t[6], 0xffffffffffffffffULL, tu + 6);
    cc = sbb(cc, t[7], 0xffffffffffffffffULL, tu + 7);
    cc = sbb(cc, t[8], 0x1ffULL, tu + 8);
    // If no borrow (t >= p), use tu; otherwise keep t
    uint64_t mask = -(uint64_t)cc;
    select64(res, tu, t, mask);
}

// d = abs(floor((a*fc + b*gc) / 2^31))
// Coefficients fc, gc are signed, |fc|,|gc| <= 2^31.
// Returns 0xFFFFFFFFFFFFFFFF if a*fc + b*gc < 0, else 0.
uint64_t lindiv31abs(uint64_t* d, const uint64_t* a, const uint64_t* b, uint64_t fc, uint64_t gc)
{
    uint64_t ta[10], tb[10], t[10], hi;
    unsigned char cc;
    
    // fc = abs(fc), conditionally negate a
    uint64_t sf = sgnw(fc);
    fc = (fc ^ sf) - sf;
    cc = sbb(0, a[0] ^ sf, sf, &ta[0]);
    cc = sbb(cc, a[1] ^ sf, sf, &ta[1]);
    cc = sbb(cc, a[2] ^ sf, sf, &ta[2]);
    cc = sbb(cc, a[3] ^ sf, sf, &ta[3]);
    cc = sbb(cc, a[4] ^ sf, sf, &ta[4]);
    cc = sbb(cc, a[5] ^ sf, sf, &ta[5]);
    cc = sbb(cc, a[6] ^ sf, sf, &ta[6]);
    cc = sbb(cc, a[7] ^ sf, sf, &ta[7]);
    cc = sbb(cc, a[8] ^ sf, sf, &ta[8]);
    (void)sbb(cc, 0, 0, &ta[9]);
    
    // gc = abs(gc), conditionally negate b
    uint64_t sg = sgnw(gc);
    gc = (gc ^ sg) - sg;
    cc = sbb(0, b[0] ^ sg, sg, &tb[0]);
    cc = sbb(cc, b[1] ^ sg, sg, &tb[1]);
    cc = sbb(cc, b[2] ^ sg, sg, &tb[2]);
    cc = sbb(cc, b[3] ^ sg, sg, &tb[3]);
    cc = sbb(cc, b[4] ^ sg, sg, &tb[4]);
    cc = sbb(cc, b[5] ^ sg, sg, &tb[5]);
    cc = sbb(cc, b[6] ^ sg, sg, &tb[6]);
    cc = sbb(cc, b[7] ^ sg, sg, &tb[7]);
    cc = sbb(cc, b[8] ^ sg, sg, &tb[8]);
    (void)sbb(cc, 0, 0, &tb[9]);
    
    // Compute ta*fc + tb*gc as a signed integer
    umulx2_64(&t[0], &hi, ta[0], fc, tb[0], gc);
    umulx2_add_64(&t[1], &hi, ta[1], fc, tb[1], gc, hi);
    umulx2_add_64(&t[2], &hi, ta[2], fc, tb[2], gc, hi);
    umulx2_add_64(&t[3], &hi, ta[3], fc, tb[3], gc, hi);
    umulx2_add_64(&t[4], &hi, ta[4], fc, tb[4], gc, hi);
    umulx2_add_64(&t[5], &hi, ta[5], fc, tb[5], gc, hi);
    umulx2_add_64(&t[6], &hi, ta[6], fc, tb[6], gc, hi);
    umulx2_add_64(&t[7], &hi, ta[7], fc, tb[7], gc, hi);
    umulx2_add_64(&t[8], &hi, ta[8], fc, tb[8], gc, hi);
    t[9] = hi - (ta[9] & fc) - (tb[9] & gc);
    
    // Right-shift by 31 bits
    t[0] = (t[0] >> 31) | (t[1] << 33);
    t[1] = (t[1] >> 31) | (t[2] << 33);
    t[2] = (t[2] >> 31) | (t[3] << 33);
    t[3] = (t[3] >> 31) | (t[4] << 33);
    t[4] = (t[4] >> 31) | (t[5] << 33);
    t[5] = (t[5] >> 31) | (t[6] << 33);
    t[6] = (t[6] >> 31) | (t[7] << 33);
    t[7] = (t[7] >> 31) | (t[8] << 33);
    t[8] = (t[8] >> 31) | (t[9] << 33);
    
    // If result is negative, negate it
    uint64_t s = sgnw(t[9]);
    cc = sbb(0, t[0] ^ s, s, &d[0]);
    cc = sbb(cc, t[1] ^ s, s, &d[1]);
    cc = sbb(cc, t[2] ^ s, s, &d[2]);
    cc = sbb(cc, t[3] ^ s, s, &d[3]);
    cc = sbb(cc, t[4] ^ s, s, &d[4]);
    cc = sbb(cc, t[5] ^ s, s, &d[5]);
    cc = sbb(cc, t[6] ^ s, s, &d[6]);
    cc = sbb(cc, t[7] ^ s, s, &d[7]);
    (void)sbb(cc, t[8] ^ s, s, &d[8]);
    
    return s;
}

// Montgomery multiplication: res = a * b / 2^576 mod p
// p = 1 * 2^521 - 1
void fp_mul_64(uint64_t* res, const uint64_t* a, const uint64_t* b)
{
    const uint64_t P[9] = {
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0x1ffULL
    };
    
    uint64_t e[18], lo, hi;
    uint64_t s[9], h[10];
    unsigned char cc;
    
    // Schoolbook multiplication: e = a * b
    umul_64(&e[0], &hi, a[0], b[0]);
    umul_add_64(&e[1], &hi, a[0], b[1], hi);
    umul_add_64(&e[2], &hi, a[0], b[2], hi);
    umul_add_64(&e[3], &hi, a[0], b[3], hi);
    umul_add_64(&e[4], &hi, a[0], b[4], hi);
    umul_add_64(&e[5], &hi, a[0], b[5], hi);
    umul_add_64(&e[6], &hi, a[0], b[6], hi);
    umul_add_64(&e[7], &hi, a[0], b[7], hi);
    umul_add_64(&e[8], &hi, a[0], b[8], hi);
    e[9] = hi;
    
    umul_64(&lo, &hi, a[1], b[0]);
    cc = adc(0, e[1], lo, &e[1]);
    umul_add_64(&lo, &hi, a[1], b[1], hi);
    cc = adc(cc, e[2], lo, &e[2]);
    umul_add_64(&lo, &hi, a[1], b[2], hi);
    cc = adc(cc, e[3], lo, &e[3]);
    umul_add_64(&lo, &hi, a[1], b[3], hi);
    cc = adc(cc, e[4], lo, &e[4]);
    umul_add_64(&lo, &hi, a[1], b[4], hi);
    cc = adc(cc, e[5], lo, &e[5]);
    umul_add_64(&lo, &hi, a[1], b[5], hi);
    cc = adc(cc, e[6], lo, &e[6]);
    umul_add_64(&lo, &hi, a[1], b[6], hi);
    cc = adc(cc, e[7], lo, &e[7]);
    umul_add_64(&lo, &hi, a[1], b[7], hi);
    cc = adc(cc, e[8], lo, &e[8]);
    umul_add_64(&lo, &hi, a[1], b[8], hi);
    cc = adc(cc, e[9], lo, &e[9]);
    e[10] = hi + (uint64_t)cc;
    
    umul_64(&lo, &hi, a[2], b[0]);
    cc = adc(0, e[2], lo, &e[2]);
    umul_add_64(&lo, &hi, a[2], b[1], hi);
    cc = adc(cc, e[3], lo, &e[3]);
    umul_add_64(&lo, &hi, a[2], b[2], hi);
    cc = adc(cc, e[4], lo, &e[4]);
    umul_add_64(&lo, &hi, a[2], b[3], hi);
    cc = adc(cc, e[5], lo, &e[5]);
    umul_add_64(&lo, &hi, a[2], b[4], hi);
    cc = adc(cc, e[6], lo, &e[6]);
    umul_add_64(&lo, &hi, a[2], b[5], hi);
    cc = adc(cc, e[7], lo, &e[7]);
    umul_add_64(&lo, &hi, a[2], b[6], hi);
    cc = adc(cc, e[8], lo, &e[8]);
    umul_add_64(&lo, &hi, a[2], b[7], hi);
    cc = adc(cc, e[9], lo, &e[9]);
    umul_add_64(&lo, &hi, a[2], b[8], hi);
    cc = adc(cc, e[10], lo, &e[10]);
    e[11] = hi + (uint64_t)cc;
    
    umul_64(&lo, &hi, a[3], b[0]);
    cc = adc(0, e[3], lo, &e[3]);
    umul_add_64(&lo, &hi, a[3], b[1], hi);
    cc = adc(cc, e[4], lo, &e[4]);
    umul_add_64(&lo, &hi, a[3], b[2], hi);
    cc = adc(cc, e[5], lo, &e[5]);
    umul_add_64(&lo, &hi, a[3], b[3], hi);
    cc = adc(cc, e[6], lo, &e[6]);
    umul_add_64(&lo, &hi, a[3], b[4], hi);
    cc = adc(cc, e[7], lo, &e[7]);
    umul_add_64(&lo, &hi, a[3], b[5], hi);
    cc = adc(cc, e[8], lo, &e[8]);
    umul_add_64(&lo, &hi, a[3], b[6], hi);
    cc = adc(cc, e[9], lo, &e[9]);
    umul_add_64(&lo, &hi, a[3], b[7], hi);
    cc = adc(cc, e[10], lo, &e[10]);
    umul_add_64(&lo, &hi, a[3], b[8], hi);
    cc = adc(cc, e[11], lo, &e[11]);
    e[12] = hi + (uint64_t)cc;
    
    umul_64(&lo, &hi, a[4], b[0]);
    cc = adc(0, e[4], lo, &e[4]);
    umul_add_64(&lo, &hi, a[4], b[1], hi);
    cc = adc(cc, e[5], lo, &e[5]);
    umul_add_64(&lo, &hi, a[4], b[2], hi);
    cc = adc(cc, e[6], lo, &e[6]);
    umul_add_64(&lo, &hi, a[4], b[3], hi);
    cc = adc(cc, e[7], lo, &e[7]);
    umul_add_64(&lo, &hi, a[4], b[4], hi);
    cc = adc(cc, e[8], lo, &e[8]);
    umul_add_64(&lo, &hi, a[4], b[5], hi);
    cc = adc(cc, e[9], lo, &e[9]);
    umul_add_64(&lo, &hi, a[4], b[6], hi);
    cc = adc(cc, e[10], lo, &e[10]);
    umul_add_64(&lo, &hi, a[4], b[7], hi);
    cc = adc(cc, e[11], lo, &e[11]);
    umul_add_64(&lo, &hi, a[4], b[8], hi);
    cc = adc(cc, e[12], lo, &e[12]);
    e[13] = hi + (uint64_t)cc;
    
    umul_64(&lo, &hi, a[5], b[0]);
    cc = adc(0, e[5], lo, &e[5]);
    umul_add_64(&lo, &hi, a[5], b[1], hi);
    cc = adc(cc, e[6], lo, &e[6]);
    umul_add_64(&lo, &hi, a[5], b[2], hi);
    cc = adc(cc, e[7], lo, &e[7]);
    umul_add_64(&lo, &hi, a[5], b[3], hi);
    cc = adc(cc, e[8], lo, &e[8]);
    umul_add_64(&lo, &hi, a[5], b[4], hi);
    cc = adc(cc, e[9], lo, &e[9]);
    umul_add_64(&lo, &hi, a[5], b[5], hi);
    cc = adc(cc, e[10], lo, &e[10]);
    umul_add_64(&lo, &hi, a[5], b[6], hi);
    cc = adc(cc, e[11], lo, &e[11]);
    umul_add_64(&lo, &hi, a[5], b[7], hi);
    cc = adc(cc, e[12], lo, &e[12]);
    umul_add_64(&lo, &hi, a[5], b[8], hi);
    cc = adc(cc, e[13], lo, &e[13]);
    e[14] = hi + (uint64_t)cc;
    
    umul_64(&lo, &hi, a[6], b[0]);
    cc = adc(0, e[6], lo, &e[6]);
    umul_add_64(&lo, &hi, a[6], b[1], hi);
    cc = adc(cc, e[7], lo, &e[7]);
    umul_add_64(&lo, &hi, a[6], b[2], hi);
    cc = adc(cc, e[8], lo, &e[8]);
    umul_add_64(&lo, &hi, a[6], b[3], hi);
    cc = adc(cc, e[9], lo, &e[9]);
    umul_add_64(&lo, &hi, a[6], b[4], hi);
    cc = adc(cc, e[10], lo, &e[10]);
    umul_add_64(&lo, &hi, a[6], b[5], hi);
    cc = adc(cc, e[11], lo, &e[11]);
    umul_add_64(&lo, &hi, a[6], b[6], hi);
    cc = adc(cc, e[12], lo, &e[12]);
    umul_add_64(&lo, &hi, a[6], b[7], hi);
    cc = adc(cc, e[13], lo, &e[13]);
    umul_add_64(&lo, &hi, a[6], b[8], hi);
    cc = adc(cc, e[14], lo, &e[14]);
    e[15] = hi + (uint64_t)cc;
    
    umul_64(&lo, &hi, a[7], b[0]);
    cc = adc(0, e[7], lo, &e[7]);
    umul_add_64(&lo, &hi, a[7], b[1], hi);
    cc = adc(cc, e[8], lo, &e[8]);
    umul_add_64(&lo, &hi, a[7], b[2], hi);
    cc = adc(cc, e[9], lo, &e[9]);
    umul_add_64(&lo, &hi, a[7], b[3], hi);
    cc = adc(cc, e[10], lo, &e[10]);
    umul_add_64(&lo, &hi, a[7], b[4], hi);
    cc = adc(cc, e[11], lo, &e[11]);
    umul_add_64(&lo, &hi, a[7], b[5], hi);
    cc = adc(cc, e[12], lo, &e[12]);
    umul_add_64(&lo, &hi, a[7], b[6], hi);
    cc = adc(cc, e[13], lo, &e[13]);
    umul_add_64(&lo, &hi, a[7], b[7], hi);
    cc = adc(cc, e[14], lo, &e[14]);
    umul_add_64(&lo, &hi, a[7], b[8], hi);
    cc = adc(cc, e[15], lo, &e[15]);
    e[16] = hi + (uint64_t)cc;
    
    umul_64(&lo, &hi, a[8], b[0]);
    cc = adc(0, e[8], lo, &e[8]);
    umul_add_64(&lo, &hi, a[8], b[1], hi);
    cc = adc(cc, e[9], lo, &e[9]);
    umul_add_64(&lo, &hi, a[8], b[2], hi);
    cc = adc(cc, e[10], lo, &e[10]);
    umul_add_64(&lo, &hi, a[8], b[3], hi);
    cc = adc(cc, e[11], lo, &e[11]);
    umul_add_64(&lo, &hi, a[8], b[4], hi);
    cc = adc(cc, e[12], lo, &e[12]);
    umul_add_64(&lo, &hi, a[8], b[5], hi);
    cc = adc(cc, e[13], lo, &e[13]);
    umul_add_64(&lo, &hi, a[8], b[6], hi);
    cc = adc(cc, e[14], lo, &e[14]);
    umul_add_64(&lo, &hi, a[8], b[7], hi);
    cc = adc(cc, e[15], lo, &e[15]);
    umul_add_64(&lo, &hi, a[8], b[8], hi);
    cc = adc(cc, e[16], lo, &e[16]);
    e[17] = hi + (uint64_t)cc;
    
    // Montgomery reduction
    // m = -1/p mod 2^(64 * n) = 1 * 2^521 + 1) 
    // s = low(e) * m mod 2^(64 * n)
    for (int i = 0; i < 9; i++) s[i] = e[i];
    s[8] += (uint64_t)((e[0] & 0x7FFFFFFFFFFFFFULL) * 1ULL) << 9;
    
    // h = s * (1 << 9)
    umul_64(&h[0], &hi, s[0], 512ULL);
    umul_add_64(&h[1], &hi, s[1], 512ULL, hi);
    umul_add_64(&h[2], &hi, s[2], 512ULL, hi);
    umul_add_64(&h[3], &hi, s[3], 512ULL, hi);
    umul_add_64(&h[4], &hi, s[4], 512ULL, hi);
    umul_add_64(&h[5], &hi, s[5], 512ULL, hi);
    umul_add_64(&h[6], &hi, s[6], 512ULL, hi);
    umul_add_64(&h[7], &hi, s[7], 512ULL, hi);
    umul_add_64(&h[8], &hi, s[8], 512ULL, hi);
    h[9] = hi;
    
    // e += s*p = s*c*2^f - s
    // Step A: e -= s
    cc = 0;
    for (int i = 0; i < 9; i++) cc = sbb(cc, e[i], s[i], &e[i]);
    for (int i = 9; i < 18; i++) cc = sbb(cc, e[i], 0, &e[i]);
    
    // Step B: e += h at word position 8
    cc = 0;
    for (int i = 0; i <= 9; i++) cc = adc(cc, e[8 + i], h[i], &e[8 + i]);
    
    // Conditional subtraction of p for full reduction
    uint64_t r[9];
    cc = 0;
    for (int i = 0; i < 9; i++) cc = sbb(cc, e[9 + i], P[i], &r[i]);
    uint64_t mask = -(uint64_t)cc;
    for (int i = 0; i < 9; i++) res[i] = (e[9 + i] & mask) | (r[i] & ~mask);
}

// Implements inversion mod p using a constant-time extended binary GCD algorithm
// by Thomas Pornin: https://eprint.iacr.org/2020/972.
// This code is based on the version implented in the SQIsign submission.
// It asssumes the input is normalized and in the Montgomery representation with R.
void fp_inv_64(uint64_t* res, const uint64_t* src)
{
    uint64_t a[9], b[9], u[9], v[9];
    uint64_t xa, xb, f0, g0, f1, g1;
    
    // Initialize a = src, b = p, u = 1 * R, v = 0
    for (int i = 0; i < 9; i++) {
        a[i] = src[i];
        b[i] = P_64[i];
        u[i] = R_64[i];
        v[i] = 0;
    }
    
    // Main loop: 32 outer iterations of k-1=31 inner halvings
    for (int i = 0; i < 32; i++) {
        // Get 64-bit approximations xa, xb of a, b
        uint64_t tnzm = 0, tnza = 0, tnzb = 0, snza = 0, snzb = 0;
        uint64_t found = 0;
        for (int k = 8; k >= 1; k--) {
            uint64_t mk = a[k] | b[k];
            uint64_t nz = sgnw(mk | -mk) & ~found;
            found |= nz;
            tnzm |= mk & nz;
            tnza |= a[k] & nz;
            tnzb |= b[k] & nz;
            snza |= a[k-1] & nz;
            snzb |= b[k-1] & nz;
        }
        
        int64_t s = lzcnt(tnzm);
        uint64_t sm = (uint64_t)((31 - s) >> 63);
        tnza ^= sm & (tnza ^ ((tnza << 32) | (snza >> 32)));
        tnzb ^= sm & (tnzb ^ ((tnzb << 32) | (snzb >> 32)));
        s -= 32 & sm;
        tnza <<= s;
        tnzb <<= s;
        
        uint64_t tzx = ~found;
        tnza |= a[0] & tzx;
        tnzb |= b[0] & tzx;
        xa = (a[0] & 0x7FFFFFFF) | (tnza & 0xFFFFFFFF80000000);
        xb = (b[0] & 0x7FFFFFFF) | (tnzb & 0xFFFFFFFF80000000);
        
        // 31 inner iterations
        uint64_t fg0 = (uint64_t)1;
        uint64_t fg1 = (uint64_t)1 << 32;
        for (int j = 0; j < 31; j++) {
            uint64_t a_odd, swap, t0, t1, t2;
            unsigned char cc;
            a_odd = -(xa & 1);
            cc = sbb(0, xa, xb, &t0);
            (void)sbb(cc, 0, 0, &swap);
            swap &= a_odd;
            t1 = swap & (xa ^ xb);
            xa ^= t1;
            xb ^= t1;
            t2 = swap & (fg0 ^ fg1);
            fg0 ^= t2;
            fg1 ^= t2;
            xa -= a_odd & xb;
            fg0 -= a_odd & fg1;
            xa >>= 1;
            fg1 <<= 1;
        }
        fg0 += 0x7FFFFFFF7FFFFFFF;
        fg1 += 0x7FFFFFFF7FFFFFFF;
        f0 = (fg0 & 0xFFFFFFFF) - (uint64_t)0x7FFFFFFF;
        g0 = (fg0 >> 32) - (uint64_t)0x7FFFFFFF;
        f1 = (fg1 & 0xFFFFFFFF) - (uint64_t)0x7FFFFFFF;
        g1 = (fg1 >> 32) - (uint64_t)0x7FFFFFFF;
        
        // Update a, b, u, v
        uint64_t na[9], nb[9], nu[9], nv[9];
        uint64_t nega = lindiv31abs(na, a, b, f0, g0);
        uint64_t negb = lindiv31abs(nb, a, b, f1, g1);
        f0 = (f0 ^ nega) - nega;
        g0 = (g0 ^ nega) - nega;
        f1 = (f1 ^ negb) - negb;
        g1 = (g1 ^ negb) - negb;
        lin64(nu, u, v, f0, g0);
        lin64(nv, u, v, f1, g1);
        for (int k = 0; k < 9; k++) {
            a[k] = na[k]; b[k] = nb[k]; u[k] = nu[k]; v[k] = nv[k];
        }
    }
    
    // Final scalar loop: 49 iterations
    xa = a[0]; xb = b[0];
    f0 = 1; g0 = 0; f1 = 0; g1 = 1;
    for (int j = 0; j < 49; j++) {
        uint64_t a_odd, swap, t0, t1, t2, t3;
        unsigned char cc;
        a_odd = -(xa & 1);
        cc = sbb(0, xa, xb, &t0);
        (void)sbb(cc, 0, 0, &swap);
        swap &= a_odd;
        t1 = swap & (xa ^ xb);
        xa ^= t1;
        xb ^= t1;
        t2 = swap & (f0 ^ f1);
        f0 ^= t2;
        f1 ^= t2;
        t3 = swap & (g0 ^ g1);
        g0 ^= t3;
        g1 ^= t3;
        xa -= a_odd & xb;
        f0 -= a_odd & f1;
        g0 -= a_odd & g1;
        xa >>= 1;
        f1 <<= 1;
        g1 <<= 1;
    }
    lin64(v, u, v, f1, g1);
    
    // Correction: Given the input src = x * R mod p,
    //  v now holds 2^num_iterations * x^-1 mod p
    // We multiply by a precomputed constant to obtain x^-1 * R mod p
    fp_mul_64(res, v, INV_CONST_64);
}

