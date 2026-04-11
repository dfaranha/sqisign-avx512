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
}

void fp_copy_8x1(__m512i* res, __m512i* a)
{
    res[0] = a[0];
    res[1] = a[1];
    res[2] = a[2];
    res[3] = a[3];
    res[4] = a[4];
}

void fp_duplicate_8x1(__m512i* res, const uint64_t* a)
{
    res[0] = VSETALL(a[0]);
    res[1] = VSETALL(a[1]);
    res[2] = VSETALL(a[2]);
    res[3] = VSETALL(a[3]);
    res[4] = VSETALL(a[4]);
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
    return acc;
}

void fp_select_8x1(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    res[0] = VBLEND(s, a0[0], a1[0]);
    res[1] = VBLEND(s, a0[1], a1[1]);
    res[2] = VBLEND(s, a0[2], a1[2]);
    res[3] = VBLEND(s, a0[3], a1[3]);
    res[4] = VBLEND(s, a0[4], a1[4]);
}

void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4;
    __m512i sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSETALL(1);
    const __m512i p_top = VSETALL(P_PLUS_1_TOP);
    
    // Add limbs
    t0 = VADD(a[0], b[0]);
    t1 = VADD(a[1], b[1]);
    t2 = VADD(a[2], b[2]);
    t3 = VADD(a[3], b[3]);
    t4 = VADD(a[4], b[4]);
    
    // Subtract p, using special shape of p
    t0 = VADD(t0, p_bot);
    t4 = VSUB(t4, p_top);
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t4, 63);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    t4 = VADD(t4, VAND(p_top, sign_mask));
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4;
}

void fp_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4;
    __m512i sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSETALL(1);
    const __m512i p_top = VSETALL(P_PLUS_1_TOP);
    
    // Subtract limbs
    t0 = VSUB(a[0], b[0]);
    t1 = VSUB(a[1], b[1]);
    t2 = VSUB(a[2], b[2]);
    t3 = VSUB(a[3], b[3]);
    t4 = VSUB(a[4], b[4]);
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t4, 63);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    t4 = VADD(t4, VAND(p_top, sign_mask));
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4;
}

void fp_neg_8x1(__m512i* res, __m512i* a)
{
    __m512i t0, t1, t2, t3, t4;
    __m512i sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSETALL(1);
    const __m512i p_top = VSETALL(P_PLUS_1_TOP);
    const __m512i zero = VZERO();
    
    // 0 - a
    t0 = VSUB(zero, a[0]);
    t1 = VSUB(zero, a[1]);
    t2 = VSUB(zero, a[2]);
    t3 = VSUB(zero, a[3]);
    t4 = VSUB(zero, a[4]);
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t4, 63);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    t4 = VADD(t4, VAND(p_top, sign_mask));
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4;
}

void fp_mixaddsub_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t0, t1, t2, t3, t4;
    __m512i sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSETALL(1);
    const __m512i p_top = VSETALL(P_PLUS_1_TOP);
    const __mmask8 sub_mask = ~add_mask;
    
    t0 = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t1 = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t2 = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    t3 = VSUBMASK(a[3], sub_mask, a[3], b[3]);
    t4 = VSUBMASK(a[4], sub_mask, a[4], b[4]);
    
    t0 = VADDMASK(t0, add_mask, a[0], b[0]);
    t1 = VADDMASK(t1, add_mask, a[1], b[1]);
    t2 = VADDMASK(t2, add_mask, a[2], b[2]);
    t3 = VADDMASK(t3, add_mask, a[3], b[3]);
    t4 = VADDMASK(t4, add_mask, a[4], b[4]);
    
    // Subtract p, using special shape of p
    t0 = VADDMASK(t0, add_mask, t0, p_bot);
    t4 = VSUBMASK(t4, add_mask, t4, p_top);
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t4, 63);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    t4 = VADD(t4, VAND(p_top, sign_mask));
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4;
}

// Single-length multiprecision addition, no correction, no carry propagation
void mp_add_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[NUM_LIMBS];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    t[3] = VADD(a[3], b[3]);
    t[4] = VADD(a[4], b[4]);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
}

// Single-length multiprecision addition, no correction, no carry propagation
void mp_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[NUM_LIMBS];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    t[0] = VSUB(a[0], b[0]);
    t[1] = VSUB(a[1], b[1]);
    t[2] = VSUB(a[2], b[2]);
    t[3] = VSUB(a[3], b[3]);
    t[4] = VSUB(a[4], b[4]);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
}

// Double-length multiprecision subtraction res = res - a - b, no correction
void mp_dblsubx2_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[2*NUM_LIMBS];
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
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    
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
}

// Double-length multiprecision subtraction res = res - a
void mp_subx2_8x1(__m512i* res, __m512i* a)
{
    __m512i t[2*NUM_LIMBS];
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
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    
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
}

void mp_mixaddsubx2_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[2*NUM_LIMBS];
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
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    t[7] = VAND(t[7], limb_mask);
    t[8] = VAND(t[8], limb_mask);
    
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
}

// Computes res = a + p - b without any carry propagation or conditional masks
void mp_add_p_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[NUM_LIMBS];
    const __m512i p_bot = VSETALL(1);
    const __m512i p_top = VSETALL(P_PLUS_1_TOP);
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    // Add p
    t[0] = VSUB(VSUB(a[0], p_bot), b[0]);
    t[1] = VSUB(a[1], b[1]);
    t[2] = VSUB(a[2], b[2]);
    t[3] = VSUB(a[3], b[3]);
    t[4] = VSUB(VADD(a[4], p_top), b[4]);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
}

void mp_mul_8x1_karat(__m512i* res, __m512i* a, __m512i* b)
{
    const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4];
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9;
    __m512i ta0, ta1, ta2;
    __m512i tb0, tb1, tb2;
    __m512i m0, m1, m2, m3, m4, m5;
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
    
    t3 = VMULADDL(t3, a1, b2);
    t4 = VMULADDH(zero, a1, b2);
    t3 = VMULADDL(t3, a2, b1);
    t4 = VMULADDH(t4, a2, b1);
    t4 = VSL(t4, LIMB_ALIGN);
    
    t4 = VMULADDL(t4, a2, b2);
    t5 = VMULADDH(zero, a2, b2);
    t5 = VSL(t5, LIMB_ALIGN);
    
    // a_hi * b_hi
    t6 = VMULADDL(zero, a3, b3);
    t7 = VMULADDH(zero, a3, b3);
    t7 = VSL(t7, LIMB_ALIGN);
    
    t7 = VMULADDL(t7, a3, b4);
    t8 = VMULADDH(zero, a3, b4);
    t7 = VMULADDL(t7, a4, b3);
    t8 = VMULADDH(t8, a4, b3);
    t8 = VSL(t8, LIMB_ALIGN);
    
    t8 = VMULADDL(t8, a4, b4);
    t9 = VMULADDH(zero, a4, b4);
    t9 = VSL(t9, LIMB_ALIGN);
    
    // ta = a_low + a_hi
    ta0 = VADD(a0, a3);
    ta1 = VADD(a1, a4);
    ta2 = a2;
    
    // tb = b_low + b_hi
    tb0 = VADD(b0, b3);
    tb1 = VADD(b1, b4);
    tb2 = b2;
    
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
    
    m3 = VMULADDL(m3, ta1, tb2);
    m4 = VMULADDH(zero, ta1, tb2);
    m3 = VMULADDL(m3, ta2, tb1);
    m4 = VMULADDH(m4, ta2, tb1);
    m4 = VSL(m4, LIMB_ALIGN);
    
    m4 = VMULADDL(m4, ta2, tb2);
    m5 = VMULADDH(zero, ta2, tb2);
    m5 = VSL(m5, LIMB_ALIGN);
    
    // m = (ta * tb) - (a_low * b_low) - (a_hi * b_hi)
    m0 = VSUB(m0, VADD(t0, t6));
    m1 = VSUB(m1, VADD(t1, t7));
    m2 = VSUB(m2, VADD(t2, t8));
    m3 = VSUB(m3, VADD(t3, t9));
    m4 = VSUB(m4, t4);
    m5 = VSUB(m5, t5);
    
    // t += m * 2^(limbs_low)
    t3 = VADD(t3, m0);
    t4 = VADD(t4, m1);
    t5 = VADD(t5, m2);
    t6 = VADD(t6, m3);
    t7 = VADD(t7, m4);
    t8 = VADD(t8, m5);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7; res[8] = t8; res[9] = t9;
}

void rdc_8x1(__m512i* res, __m512i* a)
{
    // Montgomery reduction with R = 2^n mod p, with n = ((num_limbs) * limb_len)
    // low(a) = (a_0, a_1, ... , a_num_limbs)
    // m = -p^(-1) mod 2^n = p + 2 = c * 2^e + 1
    // The output of this reduction is (a + (low(a) * m mod 2^n )*p) / 2^n
    
    const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
    const __m512i a6 = a[6], a7 = a[7], a8 = a[8], a9 = a[9];
    __m512i t0, t1, t2, t3, t4, t5;
    __mmask8 msb_mask;
    const __m512i top = VSETALL(P_PLUS_1_TOP);
    const __m512i bot = VSETALL(1);
    const __m512i two_bot = VSETALL(2);
    const __m512i two_top = VSETALL(2*P_PLUS_1_TOP);
    __m512i s4, hi, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i zero = VZERO();
    const __m512i one = VSETALL(1);
    
    // s = low(a) * m mod 2^n, n = 5*51
    // s0 = a0, ..., s3 = a3
    s4 = VAND(VMULADDL(a4, a0, top), limb_mask);
    
    // Computing a + s*p as a + s*(p+1) - s.
    // As the division by 2^n cancels the lower limbs, we don't need to compute - s.
    t0 = VMULADDL(zero, a0, top);
    hi = VMULADDH(zero, a0, top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t1 = VMULADDL(hi, a1, top);
    hi = VMULADDH(zero, a1, top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t2 = VMULADDL(hi, a2, top);
    hi = VMULADDH(zero, a2, top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t3 = VMULADDL(hi, a3, top);
    hi = VMULADDH(zero, a3, top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t4 = VMULADDL(hi, s4, top);
    hi = VMULADDH(zero, s4, top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t5 = hi;
    
    // t += a (ignoring lower limbs)
    t0 = VADD(t0, a4);
    t1 = VADD(t1, a5);
    t2 = VADD(t2, a6);
    t3 = VADD(t3, a7);
    t4 = VADD(t4, a8);
    t5 = VADD(t5, a9);
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    
    // After dividing by 2^n, the result is in t1, ..., t5
    // Reduce from [-p, 2p) to [0, p)
    // If negative add p, if non-negative subtract p
    // Implemented as: always subtract p, then add 2p if was negative
    
    sign_mask = VSRA(t5, 63);
    t1 = VADD(t1, one);
    t5 = VSUB(t5, top);
    t1 = VSUB(t1, VAND(two_bot, sign_mask));
    t5 = VADD(t5, VAND(two_top, sign_mask));
    
    // Carry propagation
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    
    // If still negative, add p
    sign_mask = VSRA(t5, 63);
    t1 = VSUB(t1, VAND(one, sign_mask));
    t5 = VADD(t5, VAND(top, sign_mask));
    
    // Carry propagation
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    
    res[0] = t1; res[1] = t2; res[2] = t3; res[3] = t4; res[4] = t5;
}

void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[10];
    __m512i t1, t2, t3, t4, t5, t6, t7, t8, t9;
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
    t[9] = VADD(t[9], VSRA(t8, LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t1, limb_mask);
    t[2] = VAND(t2, limb_mask);
    t[3] = VAND(t3, limb_mask);
    t[4] = VAND(t4, limb_mask);
    t[5] = VAND(t5, limb_mask);
    t[6] = VAND(t6, limb_mask);
    t[7] = VAND(t7, limb_mask);
    t[8] = VAND(t8, limb_mask);
    
    rdc_8x1(res, t);
}

void fp_sqr_8x1(__m512i* res, __m512i* a)
{
    fp_mul_8x1(res, a, a);
}

void fp_div3_8x1(__m512i* res, __m512i* src)
{
    __m512i t[5];
    fp_duplicate_8x1(t, INV3);
    fp_mul_8x1(res, src, t);
}

// Batched inversion invoking a single call to fp_inv_64.
void fp_inv_8x1(__m512i* res, __m512i* a)
{
    uint64_t r_64[8];
    __m512i u[5], t0[3], t1[3], s0[2], s1[2], r[2];
    __m512i idx0, idx1;
    
    fp_8x1_to_4x2(t0, t1, a);
    // t0 = (a0a1, a2a3, a4a5, a6a7)
    fp_mul_4x2(t0, t0, t1);
    fp_4x2_to_2x4(s0, s1, t0);
    // s0 = (a0a1a2a3, a4a5a6a7)
    fp_mul_2x4(s0, s0, s1);
    idx1 = VSET(4, 5, 6, 7, 0, 0, 0, 0);
    for (int i = 0; i < 2; i++) {
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
    for (int i = 0; i < 2; i++) {
        s0[i] = VPERM512(idx0, s0[i]);
        s1[i] = VPERM512(idx1, r[i]);
    }
    fp_mul_2x4(s0, s0, s1);
    // t0 = (inv01, inv23, inv45, inv67)
    fp_2x4_to_4x2(t1, s0, s0); // (inv_0123, inv_0123, inv_4567, inv_4567)
    idx0 = VSET(2, 3, 0, 1, 6, 7, 4, 5);
    for (int i = 0; i < 3; i++) {
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp_mul_4x2(t0, t0, t1);
    // res = (inv0, inv1, inv2, ...)
    fp_4x2_to_8x1(u, t0, t0);
    idx0 = VSET(1, 0, 3, 2, 5, 4, 7, 6);
    for (int i = 0; i < 5; i++) {
        res[i] = VPERM512(idx0, a[i]);
    }
    fp_mul_8x1(res, res, u);
}

void fp_sqrt_8x1(__m512i* res, __m512i* a)
{
    // Candidate root is a^((p+1)/4), with (p+1)/4 = 5*2^246
    __m512i t[5];
    
    fp_sqr_8x1(t, a);
    for (int i = 1; i < 2; i++) {
        fp_sqr_8x1(t, t);
    }
    fp_mul_8x1(t, t, a);
    for (int i = 0; i < 245; i++) {
        fp_sqr_8x1(t, t);
    }
    fp_sqr_8x1(res, t);
}

// Cost: 244 squarings, 18 multiplications
void fp_exp3div4_8x1(__m512i* res, __m512i* a)
{
    __m512i v__10[5];
    __m512i v__11[5];
    __m512i v__110[5];
    __m512i v__111[5];
    __m512i v__1110[5];
    __m512i v__11100[5];
    __m512i v__111000[5];
    __m512i v_i11[5];
    __m512i v_i18[5];
    __m512i v_i204[5];
    __m512i v_i35[5];
    __m512i v_i69[5];
    __m512i v_i70[5];
    __m512i v_i72[5];
    __m512i v_x57[5];
    __m512i v_x62[5];
    __m512i v_x64[5];
    
    fp_sqr_8x1(v__10, a);
    fp_mul_8x1(v__11, a, v__10);
    fp_sqr_8x1(v__110, v__11);
    fp_mul_8x1(v__111, a, v__110);
    fp_sqr_8x1(v__1110, v__111);
    fp_sqr_8x1(v__11100, v__1110);
    fp_sqr_8x1(v__111000, v__11100);
    fp_sqr_8x1(v_i11, v__111000);
    fp_sqr_8x1(v_i11, v_i11);
    fp_sqr_8x1(v_i11, v_i11);
    fp_mul_8x1(v_i11, v_i11, v__111000);
    fp_sqr_8x1(v_i18, v_i11);
    for (int i = 0; i < 5; i++) fp_sqr_8x1(v_i18, v_i18);
    fp_mul_8x1(v_i18, v_i18, v_i11);
    fp_sqr_8x1(v_i35, v_i18);
    fp_sqr_8x1(v_i35, v_i35);
    fp_mul_8x1(v_i35, v_i35, v__11100);
    for (int i = 0; i < 13; i++) fp_sqr_8x1(v_i35, v_i35);
    fp_mul_8x1(v_i35, v_i35, v_i18);
    fp_sqr_8x1(v_x57, v_i35);
    for (int i = 0; i < 26; i++) fp_sqr_8x1(v_x57, v_x57);
    fp_mul_8x1(v_x57, v_x57, v_i35);
    fp_mul_8x1(v_x57, v_x57, v__111);
    fp_sqr_8x1(v_i69, v_x57);
    fp_sqr_8x1(v_i69, v_i69);
    fp_sqr_8x1(v_i69, v_i69);
    fp_sqr_8x1(v_i69, v_i69);
    fp_mul_8x1(v_i69, v_i69, v__1110);
    fp_mul_8x1(v_i70, v__11, v_i69);
    fp_mul_8x1(v_x62, v_i69, v_i70);
    fp_mul_8x1(v_i72, v_i70, v_x62);
    fp_mul_8x1(v_x64, v_x62, v_i72);
    fp_mul_8x1(v_x64, v_x64, v_i72);
    fp_mul_8x1(v_i204, v_x62, v_x64);
    for (int i = 0; i < 63; i++) fp_sqr_8x1(v_i204, v_i204);
    fp_mul_8x1(v_i204, v_i204, v_x64);
    for (int i = 0; i < 64; i++) fp_sqr_8x1(v_i204, v_i204);
    fp_mul_8x1(v_i204, v_i204, v_x64);
    fp_sqr_8x1(res, v_i204);
    for (int i = 0; i < 56; i++) fp_sqr_8x1(res, res);
    fp_mul_8x1(res, res, v_x57);
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
        t[i] = ((unvec[NWORDS64 * i] >> 51) & 0x1fffULL) | ((unvec[NWORDS64 * i + 1] & 0x3fffffffffULL) << 13);
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NWORDS64 * i + 1] >> 38) & 0x3ffffffULL) | ((unvec[NWORDS64 * i + 2] & 0x1ffffffULL) << 26);
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NWORDS64 * i + 2] >> 25) & 0x7fffffffffULL) | ((unvec[NWORDS64 * i + 3] & 0xfffULL) << 39);
    }
    res[3] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (unvec[NWORDS64 * i + 3] >> 12) & LIMB_MASK;
    }
    res[4] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
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
        t[i] = ((unvec[NWORDS64 * i] >> 51) & 0x1fffULL) | ((unvec[NWORDS64 * i + 1] & 0x3fffffffffULL) << 13);
    }
    res[1] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NWORDS64 * i + 1] >> 38) & 0x3ffffffULL) | ((unvec[NWORDS64 * i + 2] & 0x1ffffffULL) << 26);
    }
    res[2] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NWORDS64 * i + 2] >> 25) & 0x7fffffffffULL) | ((unvec[NWORDS64 * i + 3] & 0xfffULL) << 39);
    }
    res[3] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (unvec[NWORDS64 * i + 3] >> 12) & LIMB_MASK;
    }
    res[4] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
}

void fp_limbs_to_64_8x1(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[5], zero[5];
    // Normalize input
    fp_setzero_8x1(zero);
    fp_add_8x1(s, src, zero);
    
    
    VSTORE(t, s[0]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i] = t[i] & LIMB_MASK;
    }
    VSTORE(t, s[1]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i] |= (t[i] & 0x1fffULL) << 51;
        res[NWORDS64 * i + 1] = (t[i] >> 13) & 0x3fffffffffULL;
    }
    VSTORE(t, s[2]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 1] |= (t[i] & 0x3ffffffULL) << 38;
        res[NWORDS64 * i + 2] = (t[i] >> 26) & 0x1ffffffULL;
    }
    VSTORE(t, s[3]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 2] |= (t[i] & 0x7fffffffffULL) << 25;
        res[NWORDS64 * i + 3] = (t[i] >> 39) & 0xfffULL;
    }
    VSTORE(t, s[4]);
    for (int i = 0; i < 8; i++) {
        res[NWORDS64 * i + 3] |= (t[i] & LIMB_MASK) << 12;
    }
}

void fp_limbs_to_64_8x1_only_even_idx(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[5], zero[5];
    // Normalize input
    fp_setzero_8x1(zero);
    fp_add_8x1(s, src, zero);
    
    
    VSTORE(t, s[0]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i] = t[2*i] & LIMB_MASK;
    }
    VSTORE(t, s[1]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i] |= (t[2*i] & 0x1fffULL) << 51;
        res[NWORDS64 * i + 1] = (t[2*i] >> 13) & 0x3fffffffffULL;
    }
    VSTORE(t, s[2]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 1] |= (t[2*i] & 0x3ffffffULL) << 38;
        res[NWORDS64 * i + 2] = (t[2*i] >> 26) & 0x1ffffffULL;
    }
    VSTORE(t, s[3]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 2] |= (t[2*i] & 0x7fffffffffULL) << 25;
        res[NWORDS64 * i + 3] = (t[2*i] >> 39) & 0xfffULL;
    }
    VSTORE(t, s[4]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 3] |= (t[2*i] & LIMB_MASK) << 12;
    }
}

void rand_fp_8x1(__m512i* res)
{
    __m512i zero[5];
    __m512i mask = VSETALL(LIMB_MASK);
    fp_setzero_8x1(zero);
    
    for (int i = 0; i < 4; i++) {
        res[i] = VAND(rand_512i(), mask);
    }
    // Last limb is shorter
    mask = VSETALL(LAST_LIMB_MASK);
    res[4] = VAND(rand_512i(), mask);
    
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
}

void fp_copy_4x2(__m512i* res, __m512i* a)
{
    res[0] = a[0];
    res[1] = a[1];
    res[2] = a[2];
}

void fp_duplicate_4x2(__m512i* res, const uint64_t* a)
{
    res[0] = VSET(a[0], a[3], a[0], a[3], a[0], a[3], a[0], a[3]);
    res[1] = VSET(a[1], a[4], a[1], a[4], a[1], a[4], a[1], a[4]);
    res[2] = VSET(a[2], 0, a[2], 0, a[2], 0, a[2], 0);
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
    acc &= VCMPEQ(norm_a[2], norm_b[2]) | 0xaa;
    
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
    acc &= VCMPEQ(norm_a[2], zero[2]) | 0xaa;
    
    // a = 0 iff both halves are 0
    return (acc & (acc >> 1) & 0x1)
        | ((acc >> 1) & (acc >> 2) & 0x2)
        | ((acc >> 2) & (acc >> 3) & 0x4)
        | ((acc >> 3) & (acc >> 4) & 0x8);
}

void fp_select_4x2(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    __m512i t[3];
    __mmask8 mask = (s & 0x1) | ((s & 0x1) << 1)
        | ((s & 0x2) << 1) | ((s & 0x2) << 2)
        | ((s & 0x4) << 2) | ((s & 0x4) << 3)
        | ((s & 0x8) << 3) | ((s & 0x8) << 4);
    
    t[0] = VBLEND(mask, a0[0], a1[0]);
    t[1] = VBLEND(mask, a0[1], a1[1]);
    t[2] = VBLEND(mask, a0[2], a1[2]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void fp_add_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[3], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    
    // Add limbs
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    
    // Subtract p, using special shape of p
    t[0] = VADD(t[0], p_bot);
    t[1] = VSUB(t[1], p_top);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    carry = VSRA(t[2], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VANDMASK(t[1], 0x55, t[1], limb_mask);
    t[2] = VANDMASK(t[2], 0x55, t[2], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[1], 63);
    t[1] = VADD(t[1], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    carry = VSRA(t[2], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VANDMASK(t[1], 0x55, t[1], limb_mask);
    t[2] = VANDMASK(t[2], 0x55, t[2], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void fp_sub_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[3], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    
    // Subtract limbs
    t[0] = VSUB(a[0], b[0]);
    t[1] = VSUB(a[1], b[1]);
    t[2] = VSUB(a[2], b[2]);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    carry = VSRA(t[2], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VANDMASK(t[1], 0x55, t[1], limb_mask);
    t[2] = VANDMASK(t[2], 0x55, t[2], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[1], 63);
    t[1] = VADD(t[1], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    carry = VSRA(t[2], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VANDMASK(t[1], 0x55, t[1], limb_mask);
    t[2] = VANDMASK(t[2], 0x55, t[2], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void fp_neg_4x2(__m512i* res, __m512i* a)
{
    __m512i t[3], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    const __m512i zero = VZERO();
    
    // Subtract limbs
    t[0] = VSUB(zero, a[0]);
    t[1] = VSUB(zero, a[1]);
    t[2] = VSUB(zero, a[2]);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    carry = VSRA(t[2], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VANDMASK(t[1], 0x55, t[1], limb_mask);
    t[2] = VANDMASK(t[2], 0x55, t[2], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[1], 63);
    t[1] = VADD(t[1], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    carry = VSRA(t[2], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VANDMASK(t[1], 0x55, t[1], limb_mask);
    t[2] = VANDMASK(t[2], 0x55, t[2], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void fp_mixaddsub_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[3], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    const __mmask8 sub_mask = ~add_mask;
    
    t[0] = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t[1] = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t[2] = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    t[2] = VADDMASK(t[2], add_mask, a[2], b[2]);
    
    // a + b - p
    t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);
    t[1] = VSUBMASK(t[1], add_mask, t[1], p_top);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    carry = VSRA(t[2], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VANDMASK(t[1], 0x55, t[1], limb_mask);
    t[2] = VANDMASK(t[2], 0x55, t[2], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[1], 63);
    t[1] = VADD(t[1], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    carry = VSRA(t[2], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VANDMASK(t[1], 0x55, t[1], limb_mask);
    t[2] = VANDMASK(t[2], 0x55, t[2], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

// Single-length multiprecision addition, no correction, no carry propagation
void mp_add_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[3];
    
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void mp_add_p_sub_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[3];
    const __m512i p_bod = VSETALL(LIMB_MASK);
    const __m512i p_top = VSET(LIMB_MASK, P_PLUS_1_TOP-1, LIMB_MASK, P_PLUS_1_TOP-1, LIMB_MASK, P_PLUS_1_TOP-1, LIMB_MASK, P_PLUS_1_TOP-1);
    
    t[0] = VSUB(VADD(a[0], p_bod), b[0]);
    t[1] = VSUB(VADD(a[1], p_top), b[1]);
    t[2] = VADDMASK(a[2], 0x55, a[2], p_bod);
    t[2] = VSUBMASK(t[2], 0x55, t[2], b[2]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
}

void mp_mixaddsubx2_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[9];
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
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    t[2] = VADDMASK(t[2], add_mask, a[2], b[2]);
    t[3] = VADDMASK(t[3], add_mask, a[3], b[3]);
    t[4] = VADDMASK(t[4], add_mask, a[4], b[4]);
    t[5] = VADDMASK(t[5], add_mask, a[5], b[5]);
    t[6] = VADDMASK(t[6], add_mask, a[6], b[6]);
    t[7] = VADDMASK(t[7], add_mask, a[7], b[7]);
    t[8] = VADDMASK(t[8], add_mask, a[8], b[8]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
}

// Double-length multiprecision subtraction res = res - a - b, no correction
void mp_dblsubx2_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[9];
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
    
    t[0] = VSUB(t[0], b[0]);
    t[1] = VSUB(t[1], b[1]);
    t[2] = VSUB(t[2], b[2]);
    t[3] = VSUB(t[3], b[3]);
    t[4] = VSUB(t[4], b[4]);
    t[5] = VSUB(t[5], b[5]);
    t[6] = VSUB(t[6], b[6]);
    t[7] = VSUB(t[7], b[7]);
    t[8] = VSUB(t[8], b[8]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
}

// Double-length multiprecision subtraction res = res - a
void mp_subx2_4x2(__m512i* res, __m512i* a)
{
    __m512i t[9];
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
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = t[8];
}

// Algorithm 2 from "Finite Field Arithmetic Using AVX-512 For Isogeny-Based Cryptography" by Orisaka, Aranha and López
void mp_mul_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[8], sl[3], sh[2];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i zero = VZERO();
    
    // Duplicating limbs of b with shuffles
    // If b[0] = {a_0, a_3, b_0, b_3, ...} then sl0 = {a_0, a_0, b_0, b_0, ...}
    sl[0] = VSHUFFLE(b[0], 0x44);
    sl[1] = VSHUFFLE(b[1], 0x44);
    sl[2] = VSHUFFLE(b[2], 0x44);
    // If b[0] = {a_0, a_3, b_0, b_3, ...} then sh0 = {a_3, a_3, b_3, b_3, ...}
    sh[0] = VSHUFFLE(b[0], 0xee);
    sh[1] = VSHUFFLE(b[1], 0xee);
    
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
    
    t[3] = VMULADDL(t[3], a[0], sh[0]);
    t[4] = VMULADDH(zero, a[0], sh[0]);
    t[3] = VMULADDL(t[3], a[1], sl[2]);
    t[4] = VMULADDH(t[4], a[1], sl[2]);
    t[3] = VMULADDL(t[3], a[2], sl[1]);
    t[4] = VMULADDH(t[4], a[2], sl[1]);
    t[4] = VSL(t[4], LIMB_ALIGN);
    
    t[4] = VMULADDL(t[4], a[0], sh[1]);
    t[5] = VMULADDH(zero, a[0], sh[1]);
    t[4] = VMULADDL(t[4], a[1], sh[0]);
    t[5] = VMULADDH(t[5], a[1], sh[0]);
    t[4] = VMULADDL(t[4], a[2], sl[2]);
    t[5] = VMULADDH(t[5], a[2], sl[2]);
    t[5] = VSL(t[5], LIMB_ALIGN);
    
    t[5] = VMULADDL(t[5], a[1], sh[1]);
    t[6] = VMULADDH(zero, a[1], sh[1]);
    t[5] = VMULADDL(t[5], a[2], sh[0]);
    t[6] = VMULADDH(t[6], a[2], sh[0]);
    t[6] = VSL(t[6], LIMB_ALIGN);
    
    t[6] = VMULADDL(t[6], a[2], sh[1]);
    t[7] = VMULADDH(zero, a[2], sh[1]);
    t[7] = VSL(t[7], LIMB_ALIGN);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
    res[8] = zero;
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
    __m512i sign_mask;
    const __m512i bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i two_bot = VSET(2, 0, 2, 0, 2, 0, 2, 0);
    const __m512i two_top = VADD(top, top);
    
    // Montgomery reduction loop
    // using the fact that -p^{-1} mod 2^(limb_len) = 1 + c* 2^f mod 2^(limb_len) = 1 as f > limb_len
    carry = VSRA(a[0], LIMB_LEN);
    a[0] = VAND(a[0], limb_mask);
    a[1] = VADD(a[1], carry);
    // a = a + u_i * p * 2^(limb_len * i)
    // To multiply by p, we multiply by (p+1) and subtract one.
    // However, since we will drop the lower limbs when we divide later, we don't need to subtract.
    shuf = VSHUFFLEMASKZ(0xcccc, a[0], 0x4e);
    a[1] = VMULADDL(a[1], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[2] = VADD(a[2], hi);
    // Folding step
    a[3] = VADDMASK(a[3], 0x55, a[3], VSHUFFLE(a[0], 0x4e));
    
    carry = VSRA(a[1], LIMB_LEN);
    a[1] = VAND(a[1], limb_mask);
    a[2] = VADD(a[2], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[1], 0x4e);
    a[2] = VMULADDL(a[2], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[3] = VADD(a[3], hi);
    a[4] = VADDMASK(a[4], 0x55, a[4], VSHUFFLE(a[1], 0x4e));
    
    carry = VSRA(a[2], LIMB_LEN);
    a[2] = VAND(a[2], limb_mask);
    a[3] = VADD(a[3], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[2], 0x4e);
    a[3] = VMULADDL(a[3], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[4] = VADD(a[4], hi);
    a[5] = VADDMASK(a[5], 0x55, a[5], VSHUFFLE(a[2], 0x4e));
    
    carry = VSRA(a[3], LIMB_LEN);
    a[3] = VAND(a[3], limb_mask);
    a[4] = VADD(a[4], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[3], 0x4e);
    a[4] = VMULADDL(a[4], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[5] = VADD(a[5], hi);
    a[6] = VADDMASK(a[6], 0x55, a[6], VSHUFFLE(a[3], 0x4e));
    
    carry = VSRA(a[4], LIMB_LEN);
    a[4] = VAND(a[4], limb_mask);
    a[5] = VADD(a[5], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[4], 0x4e);
    a[5] = VMULADDL(a[5], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[6] = VADD(a[6], hi);
    a[7] = VADDMASK(a[7], 0x55, a[7], VSHUFFLE(a[4], 0x4e));
    
    // Result is now in a[5], ..., a[7]
    // Carry propagation
    a[6] = VADDMASK(a[6], 0x55, a[6], VSRA(a[5], LIMB_LEN));
    a[7] = VADDMASK(a[7], 0x55, a[7], VSRA(a[6], LIMB_LEN));
    carry = VSRA(a[7], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    a[5] = VADDMASK(a[5], 0xaa, a[5], carry);
    a[6] = VADDMASK(a[6], 0xaa, a[6], VSRA(a[5], LIMB_LEN));
    a[5] = VAND(a[5], limb_mask);
    a[6] = VANDMASK(a[6], 0x55, a[6], limb_mask);
    a[7] = VANDMASK(a[7], 0x55, a[7], limb_mask);
    
    // The result is in [0, 2p).
    // Reduce from [-p, 2p) to [0, p)
    
    sign_mask = VSRA(a[6], 63);
    a[5] = VADD(a[5], bot);
    a[6] = VSUBMASK(a[6], 0xaa, a[6], top);
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    a[5] = VSUB(a[5], VAND(two_bot, sign_mask));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    a[6] = VADDMASK(a[6], 0xaa, a[6], VAND(sign_mask, two_top));
    
    // Carry propagation
    a[6] = VADDMASK(a[6], 0x55, a[6], VSRA(a[5], LIMB_LEN));
    a[7] = VADDMASK(a[7], 0x55, a[7], VSRA(a[6], LIMB_LEN));
    carry = VSRA(a[7], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    a[5] = VADDMASK(a[5], 0xaa, a[5], carry);
    a[6] = VADDMASK(a[6], 0xaa, a[6], VSRA(a[5], LIMB_LEN));
    a[5] = VAND(a[5], limb_mask);
    a[6] = VANDMASK(a[6], 0x55, a[6], limb_mask);
    a[7] = VANDMASK(a[7], 0x55, a[7], limb_mask);
    
    // If still negative, add p
    sign_mask = VSRA(a[6], 63);
    a[6] = VADDMASK(a[6], 0xaa, a[6], VAND(sign_mask, top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    a[5] = VSUB(a[5], VAND(bot, sign_mask));
    
    // Carry propagation
    a[6] = VADDMASK(a[6], 0x55, a[6], VSRA(a[5], LIMB_LEN));
    a[7] = VADDMASK(a[7], 0x55, a[7], VSRA(a[6], LIMB_LEN));
    carry = VSRA(a[7], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    a[5] = VADDMASK(a[5], 0xaa, a[5], carry);
    a[6] = VADDMASK(a[6], 0xaa, a[6], VSRA(a[5], LIMB_LEN));
    a[5] = VAND(a[5], limb_mask);
    a[6] = VANDMASK(a[6], 0x55, a[6], limb_mask);
    a[7] = VANDMASK(a[7], 0x55, a[7], limb_mask);
    
    res[0] = a[5];
    res[1] = a[6];
    res[2] = a[7];
}

void fp_mul_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[9];
    
    mp_mul_4x2(t, a, b);
    rdc_4x2(res, t);
}

void fp_sqr_4x2(__m512i* res, __m512i* a)
{
    fp_mul_4x2(res, a, a);
}

void fp_div3_4x2(__m512i* res, __m512i* src)
{
    __m512i t[3];
    fp_duplicate_4x2(t, INV3);
    fp_mul_4x2(res, src, t);
}

// Batched inversion.
void fp_inv_4x2(__m512i* res, __m512i* a)
{
    uint64_t r_64[8];
    __m512i s0[2], s1[2], r[2];
    __m512i t0[3], t1[3];
    __m512i idx0, idx1;
    
    // s0 = (a0*a1, a2*a3)
    fp_4x2_to_2x4(s0, s1, a);
    fp_mul_2x4(s0, s0, s1);
    
    // r = a0a1a2a3 (wastes one mul)
    idx1 = VSET(4, 5, 6, 7, 0, 0, 0, 0);
    for (int i = 0; i < 2; i++) {
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
    for (int i = 0; i < 2; i++) {
        s0[i] = VPERM512(idx0, s0[i]);
        s1[i] = VPERM512(idx1, r[i]);
    }
    fp_mul_2x4(s0, s0, s1);
    
    // res = (inv0, inv1, inv2, inv3)
    fp_2x4_to_4x2(t0, s0, s0); // (inv_01, inv_01, inv_23, inv_23)
    idx1 = VSET(2, 3, 0, 1, 6, 7, 4, 5);
    for (int i = 0; i < 3; i++) {
        t1[i] = VPERM512(idx1, a[i]);
    }
    fp_mul_4x2(res, t0, t1);
}

void fp_sqrt_4x2(__m512i* res, __m512i* a)
{
    // Candidate root is a^((p+1)/4), with (p+1)/4 = 5*2^246
    __m512i t[3];
    
    fp_sqr_4x2(t, a);
    for (int i = 1; i < 2; i++) {
        fp_sqr_4x2(t, t);
    }
    fp_mul_4x2(t, t, a);
    for (int i = 0; i < 245; i++) {
        fp_sqr_4x2(t, t);
    }
    fp_sqr_4x2(res, t);
}

// Cost: 244 squarings, 18 multiplications
void fp_exp3div4_4x2(__m512i* res, __m512i* a)
{
    __m512i v__10[3];
    __m512i v__11[3];
    __m512i v__110[3];
    __m512i v__111[3];
    __m512i v__1110[3];
    __m512i v__11100[3];
    __m512i v__111000[3];
    __m512i v_i11[3];
    __m512i v_i18[3];
    __m512i v_i204[3];
    __m512i v_i35[3];
    __m512i v_i69[3];
    __m512i v_i70[3];
    __m512i v_i72[3];
    __m512i v_x57[3];
    __m512i v_x62[3];
    __m512i v_x64[3];
    
    fp_sqr_4x2(v__10, a);
    fp_mul_4x2(v__11, a, v__10);
    fp_sqr_4x2(v__110, v__11);
    fp_mul_4x2(v__111, a, v__110);
    fp_sqr_4x2(v__1110, v__111);
    fp_sqr_4x2(v__11100, v__1110);
    fp_sqr_4x2(v__111000, v__11100);
    fp_sqr_4x2(v_i11, v__111000);
    fp_sqr_4x2(v_i11, v_i11);
    fp_sqr_4x2(v_i11, v_i11);
    fp_mul_4x2(v_i11, v_i11, v__111000);
    fp_sqr_4x2(v_i18, v_i11);
    for (int i = 0; i < 5; i++) fp_sqr_4x2(v_i18, v_i18);
    fp_mul_4x2(v_i18, v_i18, v_i11);
    fp_sqr_4x2(v_i35, v_i18);
    fp_sqr_4x2(v_i35, v_i35);
    fp_mul_4x2(v_i35, v_i35, v__11100);
    for (int i = 0; i < 13; i++) fp_sqr_4x2(v_i35, v_i35);
    fp_mul_4x2(v_i35, v_i35, v_i18);
    fp_sqr_4x2(v_x57, v_i35);
    for (int i = 0; i < 26; i++) fp_sqr_4x2(v_x57, v_x57);
    fp_mul_4x2(v_x57, v_x57, v_i35);
    fp_mul_4x2(v_x57, v_x57, v__111);
    fp_sqr_4x2(v_i69, v_x57);
    fp_sqr_4x2(v_i69, v_i69);
    fp_sqr_4x2(v_i69, v_i69);
    fp_sqr_4x2(v_i69, v_i69);
    fp_mul_4x2(v_i69, v_i69, v__1110);
    fp_mul_4x2(v_i70, v__11, v_i69);
    fp_mul_4x2(v_x62, v_i69, v_i70);
    fp_mul_4x2(v_i72, v_i70, v_x62);
    fp_mul_4x2(v_x64, v_x62, v_i72);
    fp_mul_4x2(v_x64, v_x64, v_i72);
    fp_mul_4x2(v_i204, v_x62, v_x64);
    for (int i = 0; i < 63; i++) fp_sqr_4x2(v_i204, v_i204);
    fp_mul_4x2(v_i204, v_i204, v_x64);
    for (int i = 0; i < 64; i++) fp_sqr_4x2(v_i204, v_i204);
    fp_mul_4x2(v_i204, v_i204, v_x64);
    fp_sqr_4x2(res, v_i204);
    for (int i = 0; i < 56; i++) fp_sqr_4x2(res, res);
    fp_mul_4x2(res, res, v_x57);
}

void fp_tomont_4x2(__m512i* res, __m512i* a)
{
    __m512i vR2[3];
    fp_duplicate_4x2(vR2, R2);
    fp_mul_4x2(res, a, vR2);
}

void fp_frommont_4x2(__m512i* res, __m512i* a)
{
    __m512i one[3];
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
        t[2*i+1] = ((unvec[NWORDS64 * i + 2] >> 25) & 0x7fffffffffULL) | ((unvec[NWORDS64 * i + 3] & 0xfffULL) << 39);
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = ((unvec[NWORDS64 * i] >> 51) & 0x1fffULL) | ((unvec[NWORDS64 * i + 1] & 0x3fffffffffULL) << 13);
        t[2*i+1] = (unvec[NWORDS64 * i + 3] >> 12) & LIMB_MASK;
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = ((unvec[NWORDS64 * i + 1] >> 38) & 0x3ffffffULL) | ((unvec[NWORDS64 * i + 2] & 0x1ffffffULL) << 26);
        t[2*i+1] = 0;
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

// Assumes src is fully redduced
void fp_limbs_to_64_4x2(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[3], zero[3];
    // Normalize input
    fp_setzero_4x2(zero);
    fp_add_4x2(s, src, zero);
    
    
    VSTORE(t, s[0]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i] = t[2*i] & LIMB_MASK;
        
        res[NWORDS64 * i + 2] = (t[2*i+1] & 0x7fffffffffULL) << 25;
        res[NWORDS64 * i + 3] = (t[2*i+1] >> 39) & 0xfffULL;
    }
    VSTORE(t, s[1]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i] |= (t[2*i] & 0x1fffULL) << 51;
        res[NWORDS64 * i + 1] = (t[2*i] >> 13) & 0x3fffffffffULL;
        
        res[NWORDS64 * i + 3] |= (t[2*i+1] & LIMB_MASK) << 12;
    }
    VSTORE(t, s[2]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 1] |= (t[2*i] & 0x3ffffffULL) << 38;
        res[NWORDS64 * i + 2] |= (t[2*i] >> 26) & 0x1ffffffULL;
        
    }
}

void rand_fp_4x2(__m512i* res)
{
    __m512i zero[3];
    __m512i mask = VSETALL(LIMB_MASK);
    fp_setzero_4x2(zero);
    
    for (int i = 0; i < 1; i++) {
        res[i] = VAND(rand_512i(), mask);
    }
    
    // Last limb is shorter
    mask = VSET(LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK);
    res[1] = VAND(rand_512i(), mask);
    
    mask = VSET(LIMB_MASK, 0, LIMB_MASK, 0, LIMB_MASK, 0, LIMB_MASK, 0);
    res[2] = VAND(rand_512i(), mask);
    
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
}

void fp_copy_2x4(__m512i* res, __m512i* a)
{
    res[0] = a[0];
    res[1] = a[1];
}

void fp_duplicate_2x4(__m512i* res, const uint64_t* a)
{
    res[0] = VSET(a[0], a[2], a[4], 0, a[0], a[2], a[4], 0);
    res[1] = VSET(a[1], a[3], 0, 0, a[1], a[3], 0, 0);
}

int fp_is_equal_2x4(__m512i* a, __m512i* b)
{
    __m512i zero[2], norm_a[2], norm_b[2];
    fp_setzero_2x4(zero);
    
    // Normalize both to [0, p-1] by adding zero
    fp_add_2x4(norm_a, a, zero);
    fp_add_2x4(norm_b, b, zero);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], norm_b[0]) | 0x88;
    acc &= VCMPEQ(norm_a[1], norm_b[1]) | 0xcc;
    
    return (acc & (acc >> 1) & (acc >> 2) & (acc >> 3) & 0x1)
        | ((acc >> 3) & (acc >> 4) & (acc >> 5) & (acc >> 6) & 0x2);
}

int fp_is_zero_2x4(__m512i* a)
{
    __m512i zero[2], norm_a[2];
    fp_setzero_2x4(zero);
    
    // Normalize a to [0, p-1] by adding zero
    fp_add_2x4(norm_a, a, zero);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], zero[0]) | 0x88;
    acc &= VCMPEQ(norm_a[1], zero[1]) | 0xcc;
    
    return (acc & (acc >> 1) & (acc >> 2) & (acc >> 3) & 0x1)
        | ((acc >> 3) & (acc >> 4) & (acc >> 5) & (acc >> 6) & 0x2);
}

void fp_select_2x4(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    __m512i t[2];
    __mmask8 bit0 = s & 0x1, bit1 = (s >> 1) & 0x1;
    __mmask8 mask = bit0 | (bit0 << 1) | (bit0 << 2) | (bit0 << 3)
        | (bit1 << 4) | (bit1 << 5) | (bit1 << 6) | (bit1 << 7);
    
    t[0] = VBLEND(mask, a0[0], a1[0]);
    t[1] = VBLEND(mask, a0[1], a1[1]);
    
    res[0] = t[0];
    res[1] = t[1];
}

void fp_add_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[2], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, P_PLUS_1_TOP, 0, 0, 0, P_PLUS_1_TOP, 0);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // For duplicating sign bit in each half
    const __m512i idx_sign = VSET(2, 2, 2, 2, 6, 6, 6, 6);
    
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    
    // Subtract p using the special shape of p
    t[0] = VADD(t[0], p_bot);
    t[0] = VSUB(t[0], p_top);
    
    // Carry propagation (2 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    
    // If the result is negative, add p back
    sign_mask = VSRA(t[1], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[0] = VADD(t[0], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (2 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
}

void fp_sub_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[2], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, P_PLUS_1_TOP, 0, 0, 0, P_PLUS_1_TOP, 0);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // For duplicating sign bit in each half
    const __m512i idx_sign = VSET(2, 2, 2, 2, 6, 6, 6, 6);
    
    t[0] = VSUB(a[0], b[0]);
    t[1] = VSUB(a[1], b[1]);
    
    // Carry propagation (2 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t[1], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[0] = VADD(t[0], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (2 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
}

void fp_neg_2x4(__m512i* res, __m512i* a)
{
    __m512i t[2], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, P_PLUS_1_TOP, 0, 0, 0, P_PLUS_1_TOP, 0);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // For duplicating sign bit in each half
    const __m512i idx_sign = VSET(2, 2, 2, 2, 6, 6, 6, 6);
    const __m512i zero = VZERO();
    
    t[0] = VSUB(zero, a[0]);
    t[1] = VSUB(zero, a[1]);
    
    // Carry propagation (2 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t[1], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[0] = VADD(t[0], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (2 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
}

void fp_mixaddsub_2x4(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[2], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, P_PLUS_1_TOP, 0, 0, 0, P_PLUS_1_TOP, 0);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // For duplicating sign bit in each half
    const __m512i idx_sign = VSET(2, 2, 2, 2, 6, 6, 6, 6);
    const __mmask8 sub_mask = ~add_mask;
    
    t[0] = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t[1] = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    // Subtract p using the special shape of p
    t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);
    t[0] = VSUBMASK(t[0], add_mask, t[0], p_top);
    
    // Carry propagation (2 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t[1], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[0] = VADD(t[0], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (2 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0xbb, t[1], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
}

void mp_mul_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[7];
    __m512i B[5];
    const __m512i zero = VZERO();
    
    // Create a vector for each limb of b
    B[0] = VPERM256(b[0], 0x0);
    B[1] = VPERM256(b[1], 0x0);
    B[2] = VPERM256(b[0], 0x55);
    B[3] = VPERM256(b[1], 0x55);
    B[4] = VPERM256(b[0], 0xaa);
    
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
    t[3] = VSL(t[3], LIMB_ALIGN);
    
    t[3] = VMULADDL(t[3], a[0], B[3]);
    t[4] = VMULADDH(zero, a[0], B[3]);
    t[3] = VMULADDL(t[3], a[1], B[2]);
    t[4] = VMULADDH(t[4], a[1], B[2]);
    t[4] = VSL(t[4], LIMB_ALIGN);
    
    t[4] = VMULADDL(t[4], a[0], B[4]);
    t[5] = VMULADDH(zero, a[0], B[4]);
    t[4] = VMULADDL(t[4], a[1], B[3]);
    t[5] = VMULADDH(t[5], a[1], B[3]);
    t[5] = VSL(t[5], LIMB_ALIGN);
    
    t[5] = VMULADDL(t[5], a[1], B[4]);
    t[6] = VMULADDH(zero, a[1], B[4]);
    t[6] = VSL(t[6], LIMB_ALIGN);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
}

// Modifies a. Its length must be at least 13.
void rdc_2x4(__m512i* res, __m512i* a)
{
    __m512i carry, lo, hi, shuf;
    const __m512i zero = VZERO();
    const __m512i p_top_all = VSETALL(P_PLUS_1_TOP);
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i sign_mask;
    const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, P_PLUS_1_TOP, 0, 0, 0, P_PLUS_1_TOP, 0);
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    const __m512i idx_sign = VSET(2, 2, 2, 2, 6, 6, 6, 6);
    
    // Initialize the upper accumulators to zero
    a[7] = zero;
    a[8] = zero;
    a[9] = zero;
    a[10] = zero;
    a[11] = zero;
    a[12] = zero;
    
    // Move each limb one step to the left, for each half
    const __m512i idx = VSET(1, 2, 3, 0, 5, 6, 7, 0);
    
    // Generalized Montgomery reduction loop
    carry = VSRA(a[0], LIMB_LEN);
    a[0] = VAND(a[0], limb_mask);
    a[1] = VADD(a[1], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[0], 0x4e);
    a[2] = VMULADDL(a[2], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[3] = VADD(a[3], VSL(hi, LIMB_ALIGN));
    a[2] = VADD(a[2], VPERM512MASKZ(0x33, a[0], idx));
    
    carry = VSRA(a[1], LIMB_LEN);
    a[1] = VAND(a[1], limb_mask);
    a[2] = VADD(a[2], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[1], 0x4e);
    a[3] = VMULADDL(a[3], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[4] = VADD(a[4], VSL(hi, LIMB_ALIGN));
    a[3] = VADD(a[3], VPERM512MASKZ(0x33, a[1], idx));
    
    carry = VSRA(a[2], LIMB_LEN);
    a[2] = VAND(a[2], limb_mask);
    a[3] = VADD(a[3], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[2], 0x4e);
    a[4] = VMULADDL(a[4], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[5] = VADD(a[5], VSL(hi, LIMB_ALIGN));
    a[4] = VADD(a[4], VPERM512MASKZ(0x33, a[2], idx));
    
    carry = VSRA(a[3], LIMB_LEN);
    a[3] = VAND(a[3], limb_mask);
    a[4] = VADD(a[4], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[3], 0x4e);
    a[5] = VMULADDL(a[5], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[6] = VADD(a[6], VSL(hi, LIMB_ALIGN));
    a[5] = VADD(a[5], VPERM512MASKZ(0x33, a[3], idx));
    
    carry = VSRA(a[4], LIMB_LEN);
    a[4] = VAND(a[4], limb_mask);
    a[5] = VADD(a[5], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[4], 0x4e);
    a[6] = VMULADDL(a[6], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[7] = VADD(a[7], VSL(hi, LIMB_ALIGN));
    a[6] = VADD(a[6], VPERM512MASKZ(0x33, a[4], idx));
    
    // Propagate carries
    carry = VSRA(a[5], LIMB_LEN);
    a[5] = VAND(a[5], limb_mask);
    a[6] = VADD(a[6], carry);
    a[7] = VADD(a[7], VPERM512MASKZ(0x33, a[5], idx));
    
    carry = VSRA(a[6], LIMB_LEN);
    a[6] = VAND(a[6], limb_mask);
    a[7] = VADD(a[7], carry);
    a[8] = VADD(a[8], VPERM512MASKZ(0x33, a[6], idx));
    
    carry = VSRA(a[7], LIMB_LEN);
    a[7] = VAND(a[7], limb_mask);
    a[8] = VADD(a[8], carry);
    a[9] = VADD(a[9], VPERM512MASKZ(0x33, a[7], idx));
    
    carry = VSRA(a[8], LIMB_LEN);
    a[8] = VAND(a[8], limb_mask);
    a[9] = VADD(a[9], carry);
    a[10] = VADD(a[10], VPERM512MASKZ(0x33, a[8], idx));
    
    carry = VSRA(a[9], LIMB_LEN);
    a[9] = VAND(a[9], limb_mask);
    a[10] = VADD(a[10], carry);
    
    // Pack result from pos 0 of the accumulators back into 2x4 format
    const __m512i idx_pack = VSET(0, 1, 8, 9, 4, 5, 12, 13);
    lo = VUNPACKLO(a[5], a[7]);
    hi = VUNPACKLO(a[9], a[11]);
    res[0] = VBLENDPERM(idx_pack, lo, hi);
    
    lo = VUNPACKLO(a[6], a[8]);
    hi = VUNPACKLO(a[10], a[12]);
    res[1] = VBLENDPERM(idx_pack, lo, hi);
    
    res[0] = VBLEND(0x77, zero, res[0]);
    res[1] = VBLEND(0x33, zero, res[1]);
    
    // Reduce from [0, 2p) to [0, p)
    // Subtract p
    res[0] = VADD(res[0], p_bot);
    res[0] = VSUB(res[0], p_top);
    
    // Carry propagation (2 cross-lane ripples + final straight ripple)
    carry = VSRA(res[0], LIMB_LEN);
    res[0] = VAND(res[0], limb_mask);
    res[1] = VADD(res[1], carry);
    carry = VSRA(res[1], LIMB_LEN);
    res[1] = VANDMASK(res[1], 0xbb, res[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    res[0] = VADD(res[0], carry);
    carry = VSRA(res[0], LIMB_LEN);
    res[0] = VAND(res[0], limb_mask);
    res[1] = VADD(res[1], carry);
    carry = VSRA(res[1], LIMB_LEN);
    res[1] = VANDMASK(res[1], 0xbb, res[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    res[0] = VADD(res[0], carry);
    // Final straight ripple
    carry = VSRA(res[0], LIMB_LEN);
    res[0] = VAND(res[0], limb_mask);
    res[1] = VADD(res[1], carry);
    res[1] = VANDMASK(res[1], 0xbb, res[1], limb_mask);
    
    // If negative, add p back
    sign_mask = VSRA(res[1], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    res[0] = VADD(res[0], VAND(p_top, sign_mask));
    res[0] = VSUB(res[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (2 cross-lane ripples + final straight ripple)
    carry = VSRA(res[0], LIMB_LEN);
    res[0] = VAND(res[0], limb_mask);
    res[1] = VADD(res[1], carry);
    carry = VSRA(res[1], LIMB_LEN);
    res[1] = VANDMASK(res[1], 0xbb, res[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    res[0] = VADD(res[0], carry);
    carry = VSRA(res[0], LIMB_LEN);
    res[0] = VAND(res[0], limb_mask);
    res[1] = VADD(res[1], carry);
    carry = VSRA(res[1], LIMB_LEN);
    res[1] = VANDMASK(res[1], 0xbb, res[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    res[0] = VADD(res[0], carry);
    // Final straight ripple
    carry = VSRA(res[0], LIMB_LEN);
    res[0] = VAND(res[0], limb_mask);
    res[1] = VADD(res[1], carry);
    res[1] = VANDMASK(res[1], 0xbb, res[1], limb_mask);
    
}

void fp_mul_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[13];
    
    mp_mul_2x4(t, a, b);
    rdc_2x4(res, t);
}

void fp_sqr_2x4(__m512i* res, __m512i* a)
{
    __m512i t[13];
    
    mp_mul_2x4(t, a, a);
    rdc_2x4(res, t);
}

void fp_tomont_2x4(__m512i* res, __m512i* a)
{
    __m512i vR2[2];
    fp_duplicate_2x4(vR2, R2);
    fp_mul_2x4(res, a, vR2);
}

void fp_frommont_2x4(__m512i* res, __m512i* a)
{
    __m512i one[2];
    fp_setzero_2x4(one);
    one[0] = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    fp_mul_2x4(res, a, one);
}

void fp_div3_2x4(__m512i* res, __m512i* src)
{
    __m512i t[2];
    fp_duplicate_2x4(t, INV3);
    fp_mul_2x4(res, src, t);
}

// Batched inversion.
void fp_inv_2x4(__m512i* res, __m512i* a)
{
    uint64_t r_64[8];
    __m512i s[2], r[2];
    __m512i idx0, idx1;
    
    // r = a0a1 (wastes one mul)
    idx1 = VSET(4, 5, 6, 7, 0, 0, 0, 0);
    for (int i = 0; i < 2; i++) {
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
    for (int i = 0; i < 2; i++) {
        s[i] = VPERM512(idx0, a[i]);
        r[i] = VPERM512(idx1, r[i]);
    }
    fp_mul_2x4(res, r, s);
    
}

void fp_sqrt_2x4(__m512i* res, __m512i* a)
{
    // Candidate root is a^((p+1)/4), with (p+1)/4 = 5*2^246
    __m512i t[2];
    
    fp_sqr_2x4(t, a);
    for (int i = 1; i < 2; i++) {
        fp_sqr_2x4(t, t);
    }
    fp_mul_2x4(t, t, a);
    for (int i = 0; i < 245; i++) {
        fp_sqr_2x4(t, t);
    }
    fp_sqr_2x4(res, t);
}

// Cost: 244 squarings, 18 multiplications
void fp_exp3div4_2x4(__m512i* res, __m512i* a)
{
    __m512i v__10[2];
    __m512i v__11[2];
    __m512i v__110[2];
    __m512i v__111[2];
    __m512i v__1110[2];
    __m512i v__11100[2];
    __m512i v__111000[2];
    __m512i v_i11[2];
    __m512i v_i18[2];
    __m512i v_i204[2];
    __m512i v_i35[2];
    __m512i v_i69[2];
    __m512i v_i70[2];
    __m512i v_i72[2];
    __m512i v_x57[2];
    __m512i v_x62[2];
    __m512i v_x64[2];
    
    fp_sqr_2x4(v__10, a);
    fp_mul_2x4(v__11, a, v__10);
    fp_sqr_2x4(v__110, v__11);
    fp_mul_2x4(v__111, a, v__110);
    fp_sqr_2x4(v__1110, v__111);
    fp_sqr_2x4(v__11100, v__1110);
    fp_sqr_2x4(v__111000, v__11100);
    fp_sqr_2x4(v_i11, v__111000);
    fp_sqr_2x4(v_i11, v_i11);
    fp_sqr_2x4(v_i11, v_i11);
    fp_mul_2x4(v_i11, v_i11, v__111000);
    fp_sqr_2x4(v_i18, v_i11);
    for (int i = 0; i < 5; i++) fp_sqr_2x4(v_i18, v_i18);
    fp_mul_2x4(v_i18, v_i18, v_i11);
    fp_sqr_2x4(v_i35, v_i18);
    fp_sqr_2x4(v_i35, v_i35);
    fp_mul_2x4(v_i35, v_i35, v__11100);
    for (int i = 0; i < 13; i++) fp_sqr_2x4(v_i35, v_i35);
    fp_mul_2x4(v_i35, v_i35, v_i18);
    fp_sqr_2x4(v_x57, v_i35);
    for (int i = 0; i < 26; i++) fp_sqr_2x4(v_x57, v_x57);
    fp_mul_2x4(v_x57, v_x57, v_i35);
    fp_mul_2x4(v_x57, v_x57, v__111);
    fp_sqr_2x4(v_i69, v_x57);
    fp_sqr_2x4(v_i69, v_i69);
    fp_sqr_2x4(v_i69, v_i69);
    fp_sqr_2x4(v_i69, v_i69);
    fp_mul_2x4(v_i69, v_i69, v__1110);
    fp_mul_2x4(v_i70, v__11, v_i69);
    fp_mul_2x4(v_x62, v_i69, v_i70);
    fp_mul_2x4(v_i72, v_i70, v_x62);
    fp_mul_2x4(v_x64, v_x62, v_i72);
    fp_mul_2x4(v_x64, v_x64, v_i72);
    fp_mul_2x4(v_i204, v_x62, v_x64);
    for (int i = 0; i < 63; i++) fp_sqr_2x4(v_i204, v_i204);
    fp_mul_2x4(v_i204, v_i204, v_x64);
    for (int i = 0; i < 64; i++) fp_sqr_2x4(v_i204, v_i204);
    fp_mul_2x4(v_i204, v_i204, v_x64);
    fp_sqr_2x4(res, v_i204);
    for (int i = 0; i < 56; i++) fp_sqr_2x4(res, res);
    fp_mul_2x4(res, res, v_x57);
}

// Assumes unvec is an array of 2 * NWORDS64 64-bit ints
void fp_64_to_limbs_2x4(__m512i* res, uint64_t* unvec)
{
    uint64_t t[8];
    
    for (int i = 0; i < 2; i++) {
        t[4*i] = unvec[NWORDS64 * i] & LIMB_MASK;
        t[4*i + 1] = ((unvec[NWORDS64 * i + 1] >> 38) & 0x3ffffffULL) | ((unvec[NWORDS64 * i + 2] & 0x1ffffffULL) << 26);
        t[4*i + 2] = (unvec[NWORDS64 * i + 3] >> 12) & LIMB_MASK;
        t[4*i+3] = 0;
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 2; i++) {
        t[4*i] = ((unvec[NWORDS64 * i] >> 51) & 0x1fffULL) | ((unvec[NWORDS64 * i + 1] & 0x3fffffffffULL) << 13);
        t[4*i + 1] = ((unvec[NWORDS64 * i + 2] >> 25) & 0x7fffffffffULL) | ((unvec[NWORDS64 * i + 3] & 0xfffULL) << 39);
        t[4*i + 2] = 0;
        t[4*i+3] = 0;
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

void fp_limbs_to_64_2x4(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[2], zero[2];
    // Normalize input
    fp_setzero_2x4(zero);
    fp_add_2x4(s, src, zero);
    
    VSTORE(t, s[0]);
    for (int i = 0; i < 2; i++) {
        res[NWORDS64 * i] = t[4*i] & LIMB_MASK;
        res[NWORDS64 * i + 1] = (t[4*i + 1] & 0x3ffffffULL) << 38;
        res[NWORDS64 * i + 2] = (t[4*i + 1] >> 26) & 0x1ffffffULL;
        res[NWORDS64 * i + 3] = (t[4*i + 2] & LIMB_MASK) << 12;
    }
    VSTORE(t, s[1]);
    for (int i = 0; i < 2; i++) {
        res[NWORDS64 * i] |= (t[4*i] & 0x1fffULL) << 51;
        res[NWORDS64 * i + 1] |= (t[4*i] >> 13) & 0x3fffffffffULL;
        res[NWORDS64 * i + 2] |= (t[4*i + 1] & 0x7fffffffffULL) << 25;
        res[NWORDS64 * i + 3] |= (t[4*i + 1] >> 39) & 0xfffULL;
    }
}

void rand_fp_2x4(__m512i* res)
{
    __m512i zero[2];
    __m512i mask;
    fp_setzero_2x4(zero);
    
    mask = VSET(LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, 0, LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, 0);
    res[0] = VAND(rand_512i(), mask);
    mask = VSET(LIMB_MASK, LIMB_MASK, 0, 0, LIMB_MASK, LIMB_MASK, 0, 0);
    res[1] = VAND(rand_512i(), mask);
    
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
    for (int i = 0; i < 2; i++) {
        s = src[i], t = src[i + 3];
        res1[i] = VBLENDPERM(idx1, s, t);
        res2[i] = VBLENDPERM(idx2, s, t);
    }
    s = src[2];
    res1[2] = VBLENDPERM(idx1, s, VZERO());
    res2[2] = VBLENDPERM(idx2, s, VZERO());
}

// Given src1 = (a1, b1, c1, d1) and src2 = (a2, b2, c2, d2),
// it computes res = (a1, a2, b1, b2, c1, c2, d1, d2).
void fp_4x2_to_8x1(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i s, t;
    const __m512i idx1 = VSET(0, 8, 2, 10, 4, 12, 6, 14);
    const __m512i idx2 = VSET(1, 9, 3, 11, 5, 13, 7, 15);
    for (int i = 0; i < 2; i++) {
        s = src1[i], t = src2[i];
        res[i] = VBLENDPERM(idx1, s, t);
        res[i + 3] = VBLENDPERM(idx2, s, t);
    }
    s = src1[2], t = src2[2];
    res[2] = VBLENDPERM(idx1, s, t);
}

// Given src=(a, b, c, d), it computes res1 = (a, c) and res2 = (b, d).
// We choose this interleaving pattern for the Fp2 2x2x2 sqrt,
// where res1 = re and res2 = im.
void fp_4x2_to_2x4(__m512i* res1, __m512i* res2, __m512i* src)
{
    const __m512i idx_lo_ac = VSET(0, 8, 0, 0, 4, 12, 0, 0);
    const __m512i idx_lo_bd = VSET(2, 10, 0, 0, 6, 14, 0, 0);
    const __m512i idx_hi_ac = VSET(0, 0, 1, 9, 0, 0, 5, 13);
    const __m512i idx_hi_bd = VSET(0, 0, 3, 11, 0, 0, 7, 15);
    // For the single-source hi (lanes 2,6 only)
    const __m512i perm_ac = VSET(0, 0, 1, 0, 0, 0, 5, 0);
    const __m512i perm_bd = VSET(0, 0, 3, 0, 0, 0, 7, 0);
    // For the last vector (src[n24-1] paired with src[0])
    const __m512i idx_last_ac = VSET(0, 9, 0, 0, 4, 13, 0, 0);
    const __m512i idx_last_bd = VSET(2, 11, 0, 0, 6, 15, 0, 0);
    __m512i t[3], lo, hi;
    
    t[0] = src[0];
    t[1] = src[1];
    t[2] = src[2];
    
    lo = VBLENDPERM(idx_lo_ac, t[0], t[2]);
    hi = VPERM512MASKZ(0x44, t[1], perm_ac);
    res1[0] = VBLEND(0xcc, lo, hi);
    lo = VBLENDPERM(idx_lo_bd, t[0], t[2]);
    hi = VPERM512MASKZ(0x44, t[1], perm_bd);
    res2[0] = VBLEND(0xcc, lo, hi);
    
    lo = VBLENDPERM(idx_last_ac, t[1], t[0]);
    hi = VPERM512MASKZ(0x44, t[2], perm_ac);
    res1[1] = VBLEND(0xcc, lo, hi);
    lo = VBLENDPERM(idx_last_bd, t[1], t[0]);
    hi = VPERM512MASKZ(0x44, t[2], perm_bd);
    res2[1] = VBLEND(0xcc, lo, hi);
}

void fp_2x4_to_4x2(__m512i* res, __m512i* src1, __m512i* src2)
{
    const __m512i idx_0 = VSET(0, 0, 8, 8, 4, 4, 12, 12);
    const __m512i idx_1 = VSET(1, 1, 9, 9, 5, 5, 13, 13);
    const __m512i idx_2 = VSET(2, 2, 10, 10, 6, 6, 14, 14);
    __m512i t1[2], t2[2], even, odd;
    
    t1[0] = src1[0], t2[0] = src2[0];
    t1[1] = src1[1], t2[1] = src2[1];
    
    even = VBLENDPERM(idx_0, t1[0], t2[0]);
    odd  = VBLENDPERM(idx_1, t1[1], t2[1]);
    res[0] = VBLEND(0xaa, even, odd);
    
    even = VBLENDPERM(idx_0, t1[1], t2[1]);
    odd  = VBLENDPERM(idx_2, t1[0], t2[0]);
    res[1] = VBLEND(0xaa, even, odd);
    
    even = VBLENDPERM(idx_1, t1[0], t2[0]);
    odd  = VBLENDPERM(idx_2, t1[1], t2[1]);
    res[2] = VBLEND(0xaa, even, odd);
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
    uint64_t t[4];
    unsigned char cc;
    const uint64_t nz = ~is_zero64(src);
    
    // p - a
    cc = sbb(0, (uint64_t) 0xffffffffffffffff, src[0], t);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[1], t+1);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[2], t+2);
    cc = sbb(cc, (uint64_t) 0x4ffffffffffffff, src[3], t+3);
    
    res[0] = t[0] & nz;
    res[1] = t[1] & nz;
    res[2] = t[2] & nz;
    res[3] = t[3] & nz;
}

// res = f*u + g*v mod p
// Assumes f,g < 2^62
void lin64(uint64_t* res, const uint64_t* u, const uint64_t* v, uint64_t f, uint64_t g)
{
    uint64_t t[4], tu[4], tv[4], hi;
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
    
    // Reduction
    // t = L + 2^e H
    // Since p = c* 2^e - 1, 2^e = 1/c mod p
    // We fold the output as t = L + 2^e * (H mod c) + 2^e * floor(H/c)
    
    // Compute H in (h0, h1) and L in (t[0], ..., t[3]):
    uint64_t h0 = (t[3] >> 56) | (hi << 8);
    uint64_t h1 = (hi >> 56);
    t[3] &= 0x00FFFFFFFFFFFFFFULL;
    
    // Divide H by c
    __uint128_t H_full = ((__uint128_t)h1 << 64) | h0;
    uint64_t quo = (uint64_t)(H_full / 0x5ULL);
    uint64_t rem = (uint64_t)(H_full % 0x5ULL);
    
    // t = L + quo + rem * 2^e
    cc = adc(0, t[0], quo, &t[0]);
    cc = adc(cc, t[1], 0, &t[1]);
    cc = adc(cc, t[2], 0, &t[2]);
    adc(cc, t[3], rem << 56, &t[3]);
    
    // Finally, if t > p, we subtract p to reduce fully.
    cc = sbb(0, t[0], 0xffffffffffffffffULL, tu);
    cc = sbb(cc, t[1], 0xffffffffffffffffULL, tu + 1);
    cc = sbb(cc, t[2], 0xffffffffffffffffULL, tu + 2);
    cc = sbb(cc, t[3], 0x4ffffffffffffffULL, tu + 3);
    // If no borrow (t >= p), use tu; otherwise keep t
    uint64_t mask = -(uint64_t)cc;
    select64(res, tu, t, mask);
}

// d = abs(floor((a*fc + b*gc) / 2^31))
// Coefficients fc, gc are signed, |fc|,|gc| <= 2^31.
// Returns 0xFFFFFFFFFFFFFFFF if a*fc + b*gc < 0, else 0.
uint64_t lindiv31abs(uint64_t* d, const uint64_t* a, const uint64_t* b, uint64_t fc, uint64_t gc)
{
    uint64_t ta[5], tb[5], t[5], hi;
    unsigned char cc;
    
    // fc = abs(fc), conditionally negate a
    uint64_t sf = sgnw(fc);
    fc = (fc ^ sf) - sf;
    cc = sbb(0, a[0] ^ sf, sf, &ta[0]);
    cc = sbb(cc, a[1] ^ sf, sf, &ta[1]);
    cc = sbb(cc, a[2] ^ sf, sf, &ta[2]);
    cc = sbb(cc, a[3] ^ sf, sf, &ta[3]);
    (void)sbb(cc, 0, 0, &ta[4]);
    
    // gc = abs(gc), conditionally negate b
    uint64_t sg = sgnw(gc);
    gc = (gc ^ sg) - sg;
    cc = sbb(0, b[0] ^ sg, sg, &tb[0]);
    cc = sbb(cc, b[1] ^ sg, sg, &tb[1]);
    cc = sbb(cc, b[2] ^ sg, sg, &tb[2]);
    cc = sbb(cc, b[3] ^ sg, sg, &tb[3]);
    (void)sbb(cc, 0, 0, &tb[4]);
    
    // Compute ta*fc + tb*gc as a signed integer
    umulx2_64(&t[0], &hi, ta[0], fc, tb[0], gc);
    umulx2_add_64(&t[1], &hi, ta[1], fc, tb[1], gc, hi);
    umulx2_add_64(&t[2], &hi, ta[2], fc, tb[2], gc, hi);
    umulx2_add_64(&t[3], &hi, ta[3], fc, tb[3], gc, hi);
    t[4] = hi - (ta[4] & fc) - (tb[4] & gc);
    
    // Right-shift by 31 bits
    t[0] = (t[0] >> 31) | (t[1] << 33);
    t[1] = (t[1] >> 31) | (t[2] << 33);
    t[2] = (t[2] >> 31) | (t[3] << 33);
    t[3] = (t[3] >> 31) | (t[4] << 33);
    
    // If result is negative, negate it
    uint64_t s = sgnw(t[4]);
    cc = sbb(0, t[0] ^ s, s, &d[0]);
    cc = sbb(cc, t[1] ^ s, s, &d[1]);
    cc = sbb(cc, t[2] ^ s, s, &d[2]);
    (void)sbb(cc, t[3] ^ s, s, &d[3]);
    
    return s;
}

// Montgomery multiplication: res = a * b / 2^256 mod p
// p = 5 * 2^248 - 1
void fp_mul_64(uint64_t* res, const uint64_t* a, const uint64_t* b)
{
    const uint64_t P[4] = {
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0x4ffffffffffffffULL
    };
    
    uint64_t e[8], lo, hi;
    uint64_t s[4], h[5];
    unsigned char cc;
    
    // Schoolbook multiplication: e = a * b
    umul_64(&e[0], &hi, a[0], b[0]);
    umul_add_64(&e[1], &hi, a[0], b[1], hi);
    umul_add_64(&e[2], &hi, a[0], b[2], hi);
    umul_add_64(&e[3], &hi, a[0], b[3], hi);
    e[4] = hi;
    
    umul_64(&lo, &hi, a[1], b[0]);
    cc = adc(0, e[1], lo, &e[1]);
    umul_add_64(&lo, &hi, a[1], b[1], hi);
    cc = adc(cc, e[2], lo, &e[2]);
    umul_add_64(&lo, &hi, a[1], b[2], hi);
    cc = adc(cc, e[3], lo, &e[3]);
    umul_add_64(&lo, &hi, a[1], b[3], hi);
    cc = adc(cc, e[4], lo, &e[4]);
    e[5] = hi + (uint64_t)cc;
    
    umul_64(&lo, &hi, a[2], b[0]);
    cc = adc(0, e[2], lo, &e[2]);
    umul_add_64(&lo, &hi, a[2], b[1], hi);
    cc = adc(cc, e[3], lo, &e[3]);
    umul_add_64(&lo, &hi, a[2], b[2], hi);
    cc = adc(cc, e[4], lo, &e[4]);
    umul_add_64(&lo, &hi, a[2], b[3], hi);
    cc = adc(cc, e[5], lo, &e[5]);
    e[6] = hi + (uint64_t)cc;
    
    umul_64(&lo, &hi, a[3], b[0]);
    cc = adc(0, e[3], lo, &e[3]);
    umul_add_64(&lo, &hi, a[3], b[1], hi);
    cc = adc(cc, e[4], lo, &e[4]);
    umul_add_64(&lo, &hi, a[3], b[2], hi);
    cc = adc(cc, e[5], lo, &e[5]);
    umul_add_64(&lo, &hi, a[3], b[3], hi);
    cc = adc(cc, e[6], lo, &e[6]);
    e[7] = hi + (uint64_t)cc;
    
    // Montgomery reduction
    // m = -1/p mod 2^(64 * n) = 5 * 2^248 + 1) 
    // s = low(e) * m mod 2^(64 * n)
    for (int i = 0; i < 4; i++) s[i] = e[i];
    s[3] += (uint64_t)((e[0] & 0xFFULL) * 5ULL) << 56;
    
    // h = s * (5 << 56)
    umul_64(&h[0], &hi, s[0], 360287970189639680ULL);
    umul_add_64(&h[1], &hi, s[1], 360287970189639680ULL, hi);
    umul_add_64(&h[2], &hi, s[2], 360287970189639680ULL, hi);
    umul_add_64(&h[3], &hi, s[3], 360287970189639680ULL, hi);
    h[4] = hi;
    
    // e += s*p = s*c*2^f - s
    // Step A: e -= s
    cc = 0;
    for (int i = 0; i < 4; i++) cc = sbb(cc, e[i], s[i], &e[i]);
    for (int i = 4; i < 8; i++) cc = sbb(cc, e[i], 0, &e[i]);
    
    // Step B: e += h at word position 3
    cc = 0;
    for (int i = 0; i <= 4; i++) cc = adc(cc, e[3 + i], h[i], &e[3 + i]);
    
    // Conditional subtraction of p for full reduction
    uint64_t r[4];
    cc = 0;
    for (int i = 0; i < 4; i++) cc = sbb(cc, e[4 + i], P[i], &r[i]);
    uint64_t mask = -(uint64_t)cc;
    for (int i = 0; i < 4; i++) res[i] = (e[4 + i] & mask) | (r[i] & ~mask);
}

// Implements inversion mod p using a constant-time extended binary GCD algorithm
// by Thomas Pornin: https://eprint.iacr.org/2020/972.
// This code is based on the version implented in the SQIsign submission.
// It asssumes the input is normalized and in the Montgomery representation with R.
void fp_inv_64(uint64_t* res, const uint64_t* src)
{
    uint64_t a[4], b[4], u[4], v[4];
    uint64_t xa, xb, f0, g0, f1, g1;
    
    // Initialize a = src, b = p, u = 1 * R, v = 0
    for (int i = 0; i < 4; i++) {
        a[i] = src[i];
        b[i] = P_64[i];
        u[i] = R_64[i];
        v[i] = 0;
    }
    
    // Main loop: 15 outer iterations of k-1=31 inner halvings
    for (int i = 0; i < 15; i++) {
        // Get 64-bit approximations xa, xb of a, b
        uint64_t tnzm = 0, tnza = 0, tnzb = 0, snza = 0, snzb = 0;
        uint64_t found = 0;
        for (int k = 3; k >= 1; k--) {
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
        uint64_t na[4], nb[4], nu[4], nv[4];
        uint64_t nega = lindiv31abs(na, a, b, f0, g0);
        uint64_t negb = lindiv31abs(nb, a, b, f1, g1);
        f0 = (f0 ^ nega) - nega;
        g0 = (g0 ^ nega) - nega;
        f1 = (f1 ^ negb) - negb;
        g1 = (g1 ^ negb) - negb;
        lin64(nu, u, v, f0, g0);
        lin64(nv, u, v, f1, g1);
        for (int k = 0; k < 4; k++) {
            a[k] = na[k]; b[k] = nb[k]; u[k] = nu[k]; v[k] = nv[k];
        }
    }
    
    // Final scalar loop: 36 iterations
    xa = a[0]; xb = b[0];
    f0 = 1; g0 = 0; f1 = 0; g1 = 1;
    for (int j = 0; j < 36; j++) {
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

