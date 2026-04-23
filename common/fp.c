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
}

// If bits[i] = 1, the i-th Fp element is set to 1 in the Montgomery representation.
void fp_set_binary_8x1(__m512i* res, __mmask8 bits)
{
    uint64_t m0 = -(bits & 0x1);
    uint64_t m1 = -((bits >> 1) & 0x1);
    uint64_t m2 = -((bits >> 2) & 0x1);
    uint64_t m3 = -((bits >> 3) & 0x1);
    uint64_t m4 = -((bits >> 4) & 0x1);
    uint64_t m5 = -((bits >> 5) & 0x1);
    uint64_t m6 = -((bits >> 6) & 0x1);
    uint64_t m7 = -((bits >> 7) & 0x1);
    
    res[0] = VSET(R[0] & m0, R[0] & m1, R[0] & m2, R[0] & m3, R[0] & m4, R[0] & m5, R[0] & m6, R[0] & m7);
    res[1] = VSET(R[1] & m0, R[1] & m1, R[1] & m2, R[1] & m3, R[1] & m4, R[1] & m5, R[1] & m6, R[1] & m7);
    res[2] = VSET(R[2] & m0, R[2] & m1, R[2] & m2, R[2] & m3, R[2] & m4, R[2] & m5, R[2] & m6, R[2] & m7);
    res[3] = VSET(R[3] & m0, R[3] & m1, R[3] & m2, R[3] & m3, R[3] & m4, R[3] & m5, R[3] & m6, R[3] & m7);
    res[4] = VSET(R[4] & m0, R[4] & m1, R[4] & m2, R[4] & m3, R[4] & m4, R[4] & m5, R[4] & m6, R[4] & m7);
    res[5] = VSET(R[5] & m0, R[5] & m1, R[5] & m2, R[5] & m3, R[5] & m4, R[5] & m5, R[5] & m6, R[5] & m7);
    res[6] = VSET(R[6] & m0, R[6] & m1, R[6] & m2, R[6] & m3, R[6] & m4, R[6] & m5, R[6] & m6, R[6] & m7);
    res[7] = VSET(R[7] & m0, R[7] & m1, R[7] & m2, R[7] & m3, R[7] & m4, R[7] & m5, R[7] & m6, R[7] & m7);
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
}

int fp_is_equal_8x1(__m512i* a, __m512i* b)
{
    __m512i norm_a[NUM_LIMBS], norm_b[NUM_LIMBS];
    fp_normalize_8x1(norm_a, a);
    fp_normalize_8x1(norm_b, b);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], norm_b[0]);
    acc &= VCMPEQ(norm_a[1], norm_b[1]);
    acc &= VCMPEQ(norm_a[2], norm_b[2]);
    acc &= VCMPEQ(norm_a[3], norm_b[3]);
    acc &= VCMPEQ(norm_a[4], norm_b[4]);
    acc &= VCMPEQ(norm_a[5], norm_b[5]);
    acc &= VCMPEQ(norm_a[6], norm_b[6]);
    acc &= VCMPEQ(norm_a[7], norm_b[7]);
    return acc;
}

int fp_is_zero_8x1(__m512i* a)
{
    __m512i norm_a[NUM_LIMBS];
    const __m512i zero = VZERO();
    
    // Normalize a to [0, p)
    fp_normalize_8x1(norm_a, a);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], zero);
    acc &= VCMPEQ(norm_a[1], zero);
    acc &= VCMPEQ(norm_a[2], zero);
    acc &= VCMPEQ(norm_a[3], zero);
    acc &= VCMPEQ(norm_a[4], zero);
    acc &= VCMPEQ(norm_a[5], zero);
    acc &= VCMPEQ(norm_a[6], zero);
    acc &= VCMPEQ(norm_a[7], zero);
    return acc;
}

void fp_select_8x1(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    res[0] = VBLEND(s, a0[0], a1[0]);
    res[1] = VBLEND(s, a0[1], a1[1]);
    res[2] = VBLEND(s, a0[2], a1[2]);
    res[3] = VBLEND(s, a0[3], a1[3]);
    res[4] = VBLEND(s, a0[4], a1[4]);
    res[5] = VBLEND(s, a0[5], a1[5]);
    res[6] = VBLEND(s, a0[6], a1[6]);
    res[7] = VBLEND(s, a0[7], a1[7]);
}

void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    __m512i sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSETALL(2);
    const __m512i p_top = VSETALL(2*P_PLUS_1_TOP);
    
    // Add limbs
    t0 = VADD(a[0], b[0]);
    t1 = VADD(a[1], b[1]);
    t2 = VADD(a[2], b[2]);
    t3 = VADD(a[3], b[3]);
    t4 = VADD(a[4], b[4]);
    t5 = VADD(a[5], b[5]);
    t6 = VADD(a[6], b[6]);
    t7 = VADD(a[7], b[7]);
    
    // Subtract 2p, using special shape of p
    t0 = VADD(t0, p_bot);
    t7 = VSUB(t7, p_top);
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    
    // If result is negative, we undo the subtraction by 2p
    sign_mask = VSRA(t7, 63);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    t7 = VADD(t7, VAND(p_top, sign_mask));
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7;
}

void fp_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    __m512i sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSETALL(2);
    const __m512i p_top = VSETALL(2*P_PLUS_1_TOP);
    
    // Subtract limbs
    t0 = VSUB(a[0], b[0]);
    t1 = VSUB(a[1], b[1]);
    t2 = VSUB(a[2], b[2]);
    t3 = VSUB(a[3], b[3]);
    t4 = VSUB(a[4], b[4]);
    t5 = VSUB(a[5], b[5]);
    t6 = VSUB(a[6], b[6]);
    t7 = VSUB(a[7], b[7]);
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    
    // If result is negative, we add 2p
    sign_mask = VSRA(t7, 63);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    t7 = VADD(t7, VAND(p_top, sign_mask));
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7;
}

void fp_neg_8x1(__m512i* res, __m512i* a)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    __m512i sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i zero = VZERO();
    const __m512i p_bot = VSETALL(2);
    const __m512i p_top = VSETALL(2*P_PLUS_1_TOP);
    
    // 0 - a
    t0 = VSUB(zero, a[0]);
    t1 = VSUB(zero, a[1]);
    t2 = VSUB(zero, a[2]);
    t3 = VSUB(zero, a[3]);
    t4 = VSUB(zero, a[4]);
    t5 = VSUB(zero, a[5]);
    t6 = VSUB(zero, a[6]);
    t7 = VSUB(zero, a[7]);
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    
    // If result is negative, we add 2p
    sign_mask = VSRA(t7, 63);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    t7 = VADD(t7, VAND(p_top, sign_mask));
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7;
}

void fp_mixaddsub_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    __m512i sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSETALL(2);
    const __m512i p_top = VSETALL(2*P_PLUS_1_TOP);
    const __mmask8 sub_mask = ~add_mask;
    
    t0 = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t1 = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t2 = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    t3 = VSUBMASK(a[3], sub_mask, a[3], b[3]);
    t4 = VSUBMASK(a[4], sub_mask, a[4], b[4]);
    t5 = VSUBMASK(a[5], sub_mask, a[5], b[5]);
    t6 = VSUBMASK(a[6], sub_mask, a[6], b[6]);
    t7 = VSUBMASK(a[7], sub_mask, a[7], b[7]);
    
    t0 = VADDMASK(t0, add_mask, a[0], b[0]);
    t1 = VADDMASK(t1, add_mask, a[1], b[1]);
    t2 = VADDMASK(t2, add_mask, a[2], b[2]);
    t3 = VADDMASK(t3, add_mask, a[3], b[3]);
    t4 = VADDMASK(t4, add_mask, a[4], b[4]);
    t5 = VADDMASK(t5, add_mask, a[5], b[5]);
    t6 = VADDMASK(t6, add_mask, a[6], b[6]);
    t7 = VADDMASK(t7, add_mask, a[7], b[7]);
    
    // Subtract 2p, using special shape of p
    t0 = VADDMASK(t0, add_mask, t0, p_bot);
    t7 = VSUBMASK(t7, add_mask, t7, p_top);
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    
    // If the result is negative, we add 2p
    sign_mask = VSRA(t7, 63);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    t7 = VADD(t7, VAND(p_top, sign_mask));
    
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7;
}

// Reduces from [0, 2p) to [0, p)
void fp_normalize_8x1(__m512i* res, __m512i* a)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    __m512i sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSETALL(1);
    const __m512i p_top = VSETALL(P_PLUS_1_TOP);
    
    // Subtract p, using the special shape of p
    t0 = VADD(a[0], p_bot);
    t1 = a[1];
    t2 = a[2];
    t3 = a[3];
    t4 = a[4];
    t5 = a[5];
    t6 = a[6];
    t7 = VSUB(a[7], p_top);
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t7, 63);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    t7 = VADD(t7, VAND(p_top, sign_mask));
    // Carry propagation
    t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7;
}

// Single-length multiprecision addition, no correction, no carry propagation
void mp_add_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    
    t0 = VADD(a[0], b[0]);
    t1 = VADD(a[1], b[1]);
    t2 = VADD(a[2], b[2]);
    t3 = VADD(a[3], b[3]);
    t4 = VADD(a[4], b[4]);
    t5 = VADD(a[5], b[5]);
    t6 = VADD(a[6], b[6]);
    t7 = VADD(a[7], b[7]);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7;
}

// Double-length multiprecision subtraction res = res - a - b, no correction
void mp_dblsubx2_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    t0 = VSUB(res[0], a[0]);
    t1 = VSUB(res[1], a[1]);
    t2 = VSUB(res[2], a[2]);
    t3 = VSUB(res[3], a[3]);
    t4 = VSUB(res[4], a[4]);
    t5 = VSUB(res[5], a[5]);
    t6 = VSUB(res[6], a[6]);
    t7 = VSUB(res[7], a[7]);
    t8 = VSUB(res[8], a[8]);
    t9 = VSUB(res[9], a[9]);
    t10 = VSUB(res[10], a[10]);
    t11 = VSUB(res[11], a[11]);
    t12 = VSUB(res[12], a[12]);
    t13 = VSUB(res[13], a[13]);
    t14 = VSUB(res[14], a[14]);
    t15 = VSUB(res[15], a[15]);
    
    t0 = VSUB(t0, b[0]);
    t1 = VSUB(t1, b[1]);
    t2 = VSUB(t2, b[2]);
    t3 = VSUB(t3, b[3]);
    t4 = VSUB(t4, b[4]);
    t5 = VSUB(t5, b[5]);
    t6 = VSUB(t6, b[6]);
    t7 = VSUB(t7, b[7]);
    t8 = VSUB(t8, b[8]);
    t9 = VSUB(t9, b[9]);
    t10 = VSUB(t10, b[10]);
    t11 = VSUB(t11, b[11]);
    t12 = VSUB(t12, b[12]);
    t13 = VSUB(t13, b[13]);
    t14 = VSUB(t14, b[14]);
    t15 = VSUB(t15, b[15]);
    
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
    t13 = VADD(t13, VSRA(t12, LIMB_LEN));
    t14 = VADD(t14, VSRA(t13, LIMB_LEN));
    t15 = VADD(t15, VSRA(t14, LIMB_LEN));
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
    t12 = VAND(t12, limb_mask);
    t13 = VAND(t13, limb_mask);
    t14 = VAND(t14, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7; res[8] = t8; res[9] = t9; res[10] = t10; res[11] = t11;
    res[12] = t12; res[13] = t13; res[14] = t14; res[15] = t15;
}

// Double-length multiprecision subtraction res = res - a
void mp_subx2_8x1(__m512i* res, __m512i* a)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    const __m512i p_bot = VSETALL(1);
    const __m512i p_top = VSETALL(P_PLUS_1_TOP);
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    t0 = VSUB(res[0], a[0]);
    t1 = VSUB(res[1], a[1]);
    t2 = VSUB(res[2], a[2]);
    t3 = VSUB(res[3], a[3]);
    t4 = VSUB(res[4], a[4]);
    t5 = VSUB(res[5], a[5]);
    t6 = VSUB(res[6], a[6]);
    t7 = VSUB(res[7], a[7]);
    t8 = VSUB(res[8], a[8]);
    t9 = VSUB(res[9], a[9]);
    t10 = VSUB(res[10], a[10]);
    t11 = VSUB(res[11], a[11]);
    t12 = VSUB(res[12], a[12]);
    t13 = VSUB(res[13], a[13]);
    t14 = VSUB(res[14], a[14]);
    t15 = VSUB(res[15], a[15]);
    
    // We can get result in [0, p*R) by always adding p * 2^(num_limbs * limb_len)
    t8 = VSUB(t8, p_bot);
    t15 = VADD(t15, p_top);
    
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
    t13 = VADD(t13, VSRA(t12, LIMB_LEN));
    t14 = VADD(t14, VSRA(t13, LIMB_LEN));
    t15 = VADD(t15, VSRA(t14, LIMB_LEN));
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
    t12 = VAND(t12, limb_mask);
    t13 = VAND(t13, limb_mask);
    t14 = VAND(t14, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7; res[8] = t8; res[9] = t9; res[10] = t10; res[11] = t11;
    res[12] = t12; res[13] = t13; res[14] = t14; res[15] = t15;
}

void mp_mixaddsubx2_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    const __m512i p_bot = VSETALL(1);
    const __m512i p_top = VSETALL(P_PLUS_1_TOP);
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __mmask8 sub_mask = ~add_mask;
    
    t0 = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t1 = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t2 = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    t3 = VSUBMASK(a[3], sub_mask, a[3], b[3]);
    t4 = VSUBMASK(a[4], sub_mask, a[4], b[4]);
    t5 = VSUBMASK(a[5], sub_mask, a[5], b[5]);
    t6 = VSUBMASK(a[6], sub_mask, a[6], b[6]);
    t7 = VSUBMASK(a[7], sub_mask, a[7], b[7]);
    t8 = VSUBMASK(a[8], sub_mask, a[8], b[8]);
    t9 = VSUBMASK(a[9], sub_mask, a[9], b[9]);
    t10 = VSUBMASK(a[10], sub_mask, a[10], b[10]);
    t11 = VSUBMASK(a[11], sub_mask, a[11], b[11]);
    t12 = VSUBMASK(a[12], sub_mask, a[12], b[12]);
    t13 = VSUBMASK(a[13], sub_mask, a[13], b[13]);
    t14 = VSUBMASK(a[14], sub_mask, a[14], b[14]);
    t15 = VSUBMASK(a[15], sub_mask, a[15], b[15]);
    
    // We can get result in [0, p*R) by always adding p * 2^(num_limbs * limb_len)
    t8 = VSUB(t8, p_bot);
    t15 = VADD(t15, p_top);
    
    t0 = VADDMASK(t0, add_mask, a[0], b[0]);
    t1 = VADDMASK(t1, add_mask, a[1], b[1]);
    t2 = VADDMASK(t2, add_mask, a[2], b[2]);
    t3 = VADDMASK(t3, add_mask, a[3], b[3]);
    t4 = VADDMASK(t4, add_mask, a[4], b[4]);
    t5 = VADDMASK(t5, add_mask, a[5], b[5]);
    t6 = VADDMASK(t6, add_mask, a[6], b[6]);
    t7 = VADDMASK(t7, add_mask, a[7], b[7]);
    t8 = VADDMASK(t8, add_mask, a[8], b[8]);
    t9 = VADDMASK(t9, add_mask, a[9], b[9]);
    t10 = VADDMASK(t10, add_mask, a[10], b[10]);
    t11 = VADDMASK(t11, add_mask, a[11], b[11]);
    t12 = VADDMASK(t12, add_mask, a[12], b[12]);
    t13 = VADDMASK(t13, add_mask, a[13], b[13]);
    t14 = VADDMASK(t14, add_mask, a[14], b[14]);
    t15 = VADDMASK(t15, add_mask, a[15], b[15]);
    
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
    t13 = VADD(t13, VSRA(t12, LIMB_LEN));
    t14 = VADD(t14, VSRA(t13, LIMB_LEN));
    t15 = VADD(t15, VSRA(t14, LIMB_LEN));
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
    t12 = VAND(t12, limb_mask);
    t13 = VAND(t13, limb_mask);
    t14 = VAND(t14, limb_mask);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7; res[8] = t8; res[9] = t9; res[10] = t10; res[11] = t11;
    res[12] = t12; res[13] = t13; res[14] = t14; res[15] = t15;
}

// Computes res = a + 2p - b without any carry propagation or conditional masks
void mp_add_p_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    const __m512i p_bot = VSETALL(2*LIMB_MASK);
    const __m512i p_top = VSETALL(2*P_PLUS_1_TOP-2);
    
    t0 = VSUB(VADD(a[0], p_bot), b[0]);
    t1 = VSUB(VADD(a[1], p_bot), b[1]);
    t2 = VSUB(VADD(a[2], p_bot), b[2]);
    t3 = VSUB(VADD(a[3], p_bot), b[3]);
    t4 = VSUB(VADD(a[4], p_bot), b[4]);
    t5 = VSUB(VADD(a[5], p_bot), b[5]);
    t6 = VSUB(VADD(a[6], p_bot), b[6]);
    t7 = VSUB(VADD(a[7], p_top), b[7]);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7;
}

void mp_mul_8x1_karat(__m512i* res, __m512i* a, __m512i* b)
{
    const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
    const __m512i a6 = a[6], a7 = a[7];
    const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3], b4 = b[4], b5 = b[5];
    const __m512i b6 = b[6], b7 = b[7];
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    __m512i ta0, ta1, ta2, ta3;
    __m512i tb0, tb1, tb2, tb3;
    __m512i m0, m1, m2, m3, m4, m5, m6, m7;
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
    
    t4 = VMULADDL(t4, a1, b3);
    t5 = VMULADDH(zero, a1, b3);
    t4 = VMULADDL(t4, a2, b2);
    t5 = VMULADDH(t5, a2, b2);
    t4 = VMULADDL(t4, a3, b1);
    t5 = VMULADDH(t5, a3, b1);
    t5 = VSL(t5, LIMB_ALIGN);
    
    t5 = VMULADDL(t5, a2, b3);
    t6 = VMULADDH(zero, a2, b3);
    t5 = VMULADDL(t5, a3, b2);
    t6 = VMULADDH(t6, a3, b2);
    t6 = VSL(t6, LIMB_ALIGN);
    
    t6 = VMULADDL(t6, a3, b3);
    t7 = VMULADDH(zero, a3, b3);
    t7 = VSL(t7, LIMB_ALIGN);
    
    // a_hi * b_hi
    t8 = VMULADDL(zero, a4, b4);
    t9 = VMULADDH(zero, a4, b4);
    t9 = VSL(t9, LIMB_ALIGN);
    
    t9 = VMULADDL(t9, a4, b5);
    t10 = VMULADDH(zero, a4, b5);
    t9 = VMULADDL(t9, a5, b4);
    t10 = VMULADDH(t10, a5, b4);
    t10 = VSL(t10, LIMB_ALIGN);
    
    t10 = VMULADDL(t10, a4, b6);
    t11 = VMULADDH(zero, a4, b6);
    t10 = VMULADDL(t10, a5, b5);
    t11 = VMULADDH(t11, a5, b5);
    t10 = VMULADDL(t10, a6, b4);
    t11 = VMULADDH(t11, a6, b4);
    t11 = VSL(t11, LIMB_ALIGN);
    
    t11 = VMULADDL(t11, a4, b7);
    t12 = VMULADDH(zero, a4, b7);
    t11 = VMULADDL(t11, a5, b6);
    t12 = VMULADDH(t12, a5, b6);
    t11 = VMULADDL(t11, a6, b5);
    t12 = VMULADDH(t12, a6, b5);
    t11 = VMULADDL(t11, a7, b4);
    t12 = VMULADDH(t12, a7, b4);
    t12 = VSL(t12, LIMB_ALIGN);
    
    t12 = VMULADDL(t12, a5, b7);
    t13 = VMULADDH(zero, a5, b7);
    t12 = VMULADDL(t12, a6, b6);
    t13 = VMULADDH(t13, a6, b6);
    t12 = VMULADDL(t12, a7, b5);
    t13 = VMULADDH(t13, a7, b5);
    t13 = VSL(t13, LIMB_ALIGN);
    
    t13 = VMULADDL(t13, a6, b7);
    t14 = VMULADDH(zero, a6, b7);
    t13 = VMULADDL(t13, a7, b6);
    t14 = VMULADDH(t14, a7, b6);
    t14 = VSL(t14, LIMB_ALIGN);
    
    t14 = VMULADDL(t14, a7, b7);
    t15 = VMULADDH(zero, a7, b7);
    t15 = VSL(t15, LIMB_ALIGN);
    
    // ta = a_low + a_hi
    ta0 = VADD(a0, a4);
    ta1 = VADD(a1, a5);
    ta2 = VADD(a2, a6);
    ta3 = VADD(a3, a7);
    
    // tb = b_low + b_hi
    tb0 = VADD(b0, b4);
    tb1 = VADD(b1, b5);
    tb2 = VADD(b2, b6);
    tb3 = VADD(b3, b7);
    
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
    
    m4 = VMULADDL(m4, ta1, tb3);
    m5 = VMULADDH(zero, ta1, tb3);
    m4 = VMULADDL(m4, ta2, tb2);
    m5 = VMULADDH(m5, ta2, tb2);
    m4 = VMULADDL(m4, ta3, tb1);
    m5 = VMULADDH(m5, ta3, tb1);
    m5 = VSL(m5, LIMB_ALIGN);
    
    m5 = VMULADDL(m5, ta2, tb3);
    m6 = VMULADDH(zero, ta2, tb3);
    m5 = VMULADDL(m5, ta3, tb2);
    m6 = VMULADDH(m6, ta3, tb2);
    m6 = VSL(m6, LIMB_ALIGN);
    
    m6 = VMULADDL(m6, ta3, tb3);
    m7 = VMULADDH(zero, ta3, tb3);
    m7 = VSL(m7, LIMB_ALIGN);
    
    // m = (ta * tb) - (a_low * b_low) - (a_hi * b_hi)
    m0 = VSUB(m0, VADD(t0, t8));
    m1 = VSUB(m1, VADD(t1, t9));
    m2 = VSUB(m2, VADD(t2, t10));
    m3 = VSUB(m3, VADD(t3, t11));
    m4 = VSUB(m4, VADD(t4, t12));
    m5 = VSUB(m5, VADD(t5, t13));
    m6 = VSUB(m6, VADD(t6, t14));
    m7 = VSUB(m7, VADD(t7, t15));
    
    // t += m * 2^(limbs_low)
    t4 = VADD(t4, m0);
    t5 = VADD(t5, m1);
    t6 = VADD(t6, m2);
    t7 = VADD(t7, m3);
    t8 = VADD(t8, m4);
    t9 = VADD(t9, m5);
    t10 = VADD(t10, m6);
    t11 = VADD(t11, m7);
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7; res[8] = t8; res[9] = t9; res[10] = t10; res[11] = t11;
    res[12] = t12; res[13] = t13; res[14] = t14; res[15] = t15;
}

void rdc_8x1(__m512i* res, __m512i* a)
{
    // Montgomery reduction with R = 2^n mod p, with n = ((num_limbs + 1) * limb_len)
    // low(a) = (a_0, a_1, ... , a_num_limbs)
    // m = -p^(-1) mod 2^n = p + 2 = c * 2^e + 1
    // The output of this reduction is (a + (low(a) * m mod 2^n )*p) / 2^n
    
    const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4], a5 = a[5];
    const __m512i a6 = a[6], a7 = a[7], a8 = a[8], a9 = a[9], a10 = a[10], a11 = a[11];
    const __m512i a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9;
    __mmask8 msb_mask;
    __m512i s7, s8, hi;
    const __m512i top = VSETALL(P_PLUS_1_TOP);
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i zero = VZERO();
    const __m512i one = VSETALL(1);
    
    // s = low(a) * m mod 2^n, n = (8+1)*48
    // s0 = a0, ..., s6 = a6
    s7 = VMULADDL(a7, a0, top);
    s8 = VMULADDH(zero, a0, top);
    s8 = VADD(a8, VSL(s8, LIMB_ALIGN));
    s8 = VMULADDL(s8, a1, top);
    s8 = VADD(s8, VSRA(s7, LIMB_LEN));
    s7 = VAND(s7, limb_mask);
    s8 = VAND(s8, limb_mask);
    
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
    
    t4 = VMULADDL(hi, a4, top);
    hi = VMULADDH(zero, a4, top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t5 = VMULADDL(hi, a5, top);
    hi = VMULADDH(zero, a5, top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t6 = VMULADDL(hi, a6, top);
    hi = VMULADDH(zero, a6, top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t7 = VMULADDL(hi, s7, top);
    hi = VMULADDH(zero, s7, top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t8 = VMULADDL(hi, s8, top);
    t9 = VMULADDH(zero, s8, top);
    t9 = VSL(t9, LIMB_ALIGN);
    
    // t += a (ignoring lower limbs)
    t0 = VADD(t0, a7);
    t1 = VADD(t1, a8);
    t2 = VADD(t2, a9);
    t3 = VADD(t3, a10);
    t4 = VADD(t4, a11);
    t5 = VADD(t5, a12);
    t6 = VADD(t6, a13);
    t7 = VADD(t7, a14);
    t8 = VADD(t8, a15);
    
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
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VAND(t3, limb_mask);
    t4 = VAND(t4, limb_mask);
    t5 = VAND(t5, limb_mask);
    t6 = VAND(t6, limb_mask);
    t7 = VAND(t7, limb_mask);
    t8 = VAND(t8, limb_mask);
    
    // After dividing by 2^n, the result is in t2, ..., t9
    res[0] = t2; res[1] = t3; res[2] = t4; res[3] = t5; res[4] = t6; res[5] = t7;
    res[6] = t8; res[7] = t9;
}

void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[16];
    __m512i t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
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
    t[15] = VADD(t[15], VSRA(t14, LIMB_LEN));
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
    
    rdc_8x1(res, t);
}

void fp_sqr_8x1(__m512i* res, __m512i* a)
{
    fp_mul_8x1(res, a, a);
}

void fp_div2_8x1(__m512i* res, __m512i* src)
{
    const __m512i a0 = src[0], a1 = src[1], a2 = src[2], a3 = src[3], a4 = src[4], a5 = src[5];
    const __m512i a6 = src[6], a7 = src[7];
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    const __m512i one = VSETALL(1);
    const __m512i cnst = VSETALL(P_PLUS_1_DIV2_TOP);
    __m512i odd_mask;
    
    // odd_mask = all 1s if a is odd, else 0
    odd_mask = VSUB(VZERO(), VAND(a0, one));
    
    // a >> 1
    t0 = VOR(VSRL(a0, 1), VSL(VAND(a1, one), 47));
    t1 = VOR(VSRL(a1, 1), VSL(VAND(a2, one), 47));
    t2 = VOR(VSRL(a2, 1), VSL(VAND(a3, one), 47));
    t3 = VOR(VSRL(a3, 1), VSL(VAND(a4, one), 47));
    t4 = VOR(VSRL(a4, 1), VSL(VAND(a5, one), 47));
    t5 = VOR(VSRL(a5, 1), VSL(VAND(a6, one), 47));
    t6 = VOR(VSRL(a6, 1), VSL(VAND(a7, one), 47));
    t7 = VSRL(a7, 1);
    
    // Conditionally add (p+1)/2 = c * 2^(e-1)
    t7 = VADD(t7, VAND(cnst, odd_mask));
    
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4; res[5] = t5;
    res[6] = t6; res[7] = t7;
}

void fp_div3_8x1(__m512i* res, __m512i* src)
{
    __m512i t[8];
    fp_duplicate_8x1(t, INV3);
    fp_mul_8x1(res, src, t);
}

// Batched inversion invoking a single call to fp_inv_64.
void fp_inv_8x1(__m512i* res, __m512i* a)
{
    uint64_t r_64[12];
    __m512i u[8], t0[4], t1[4], s0[2], s1[2], r[2];
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
    for (int i = 0; i < 4; i++) {
        t0[i] = VPERM512(idx0, t0[i]);
    }
    fp_mul_4x2(t0, t0, t1);
    // res = (inv0, inv1, inv2, ...)
    fp_4x2_to_8x1(u, t0, t0);
    idx0 = VSET(1, 0, 3, 2, 5, 4, 7, 6);
    for (int i = 0; i < 8; i++) {
        res[i] = VPERM512(idx0, a[i]);
    }
    fp_mul_8x1(res, res, u);
}

void fp_sqrt_8x1(__m512i* res, __m512i* a)
{
    // Candidate root is a^((p+1)/4), with (p+1)/4 = 65*2^374
    __m512i t[8];
    
    fp_sqr_8x1(t, a);
    for (int i = 1; i < 6; i++) {
        fp_sqr_8x1(t, t);
    }
    fp_mul_8x1(t, t, a);
    for (int i = 0; i < 373; i++) {
        fp_sqr_8x1(t, t);
    }
    fp_sqr_8x1(res, t);
}

// Cost: 376 squarings, 24 multiplications
void fp_exp3div4_8x1(__m512i* res, __m512i* a)
{
    __m512i v__10[8];
    __m512i v__100[8];
    __m512i v__101[8];
    __m512i v__111[8];
    __m512i v__1110[8];
    __m512i v__11100[8];
    __m512i v__111000[8];
    __m512i v__1110000[8];
    __m512i v_i12[8];
    __m512i v_i131[8];
    __m512i v_i132[8];
    __m512i v_i133[8];
    __m512i v_i139[8];
    __m512i v_i143[8];
    __m512i v_i144[8];
    __m512i v_i19[8];
    __m512i v_i36[8];
    __m512i v_i68[8];
    __m512i v_x122[8];
    __m512i v_x125[8];
    
    fp_sqr_8x1(v__10, a);
    fp_sqr_8x1(v__100, v__10);
    fp_mul_8x1(v__101, a, v__100);
    fp_mul_8x1(v__111, v__10, v__101);
    fp_sqr_8x1(v__1110, v__111);
    fp_sqr_8x1(v__11100, v__1110);
    fp_sqr_8x1(v__111000, v__11100);
    fp_sqr_8x1(v__1110000, v__111000);
    fp_sqr_8x1(v_i12, v__1110000);
    fp_sqr_8x1(v_i12, v_i12);
    fp_sqr_8x1(v_i12, v_i12);
    fp_mul_8x1(v_i12, v_i12, v__1110000);
    fp_sqr_8x1(v_i19, v_i12);
    for (int i = 0; i < 5; i++) fp_sqr_8x1(v_i19, v_i19);
    fp_mul_8x1(v_i19, v_i19, v_i12);
    fp_sqr_8x1(v_i36, v_i19);
    fp_sqr_8x1(v_i36, v_i36);
    fp_mul_8x1(v_i36, v_i36, v__111000);
    for (int i = 0; i < 13; i++) fp_sqr_8x1(v_i36, v_i36);
    fp_mul_8x1(v_i36, v_i36, v_i19);
    fp_sqr_8x1(v_i68, v_i36);
    fp_sqr_8x1(v_i68, v_i68);
    fp_mul_8x1(v_i68, v_i68, v__111000);
    for (int i = 0; i < 28; i++) fp_sqr_8x1(v_i68, v_i68);
    fp_mul_8x1(v_i68, v_i68, v_i36);
    fp_sqr_8x1(v_i131, v_i68);
    fp_mul_8x1(v_i131, v_i131, v__11100);
    for (int i = 0; i < 59; i++) fp_sqr_8x1(v_i131, v_i131);
    fp_mul_8x1(v_i131, v_i131, v_i68);
    fp_mul_8x1(v_i131, v_i131, v__101);
    fp_mul_8x1(v_i132, v__111, v_i131);
    fp_mul_8x1(v_i133, v__1110, v_i132);
    fp_mul_8x1(v_x122, v_i131, v_i133);
    fp_sqr_8x1(v_i139, v_x122);
    fp_mul_8x1(v_i139, v_i139, v_x122);
    fp_sqr_8x1(v_i139, v_i139);
    fp_mul_8x1(v_i139, v_i139, v_x122);
    fp_mul_8x1(v_i139, v_i139, v_i133);
    fp_mul_8x1(v_x125, v_i132, v_i139);
    fp_sqr_8x1(v_i143, v_x125);
    fp_mul_8x1(v_i143, v_i143, v_x125);
    fp_mul_8x1(v_i143, v_i143, v_i139);
    fp_mul_8x1(v_i144, v_x122, v_i143);
    fp_sqr_8x1(res, v_i144);
    for (int i = 0; i < 127; i++) fp_sqr_8x1(res, res);
    fp_mul_8x1(res, res, v_i144);
    fp_mul_8x1(res, res, v_i143);
    for (int i = 0; i < 125; i++) fp_sqr_8x1(res, res);
    fp_mul_8x1(res, res, v_x125);
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
}

void fp_limbs_to_64_8x1(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[8];
    fp_normalize_8x1(s, src);
    
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
}

void fp_limbs_to_64_8x1_only_even_idx(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[8];
    fp_normalize_8x1(s, src);
    
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
}

void rand_fp_8x1(__m512i* res)
{
    __m512i zero[8];
    __m512i mask = VSETALL(LIMB_MASK);
    fp_setzero_8x1(zero);
    
    for (int i = 0; i < 7; i++) {
        res[i] = VAND(rand_512i(), mask);
    }
    // Last limb is shorter
    mask = VSETALL(LAST_LIMB_MASK);
    res[7] = VAND(rand_512i(), mask);
    
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
}

// If bits[i] = 1, the i-th Fp element is set to 1 in the Montgomery representation.
void fp_set_binary_4x2(__m512i* res, __mmask8 bits)
{
    uint64_t m0 = -(bits & 0x1);
    uint64_t m1 = -((bits >> 1) & 0x1);
    uint64_t m2 = -((bits >> 2) & 0x1);
    uint64_t m3 = -((bits >> 3) & 0x1);
    
    res[0] = VSET(R[0] & m0, R[4] & m0, R[0] & m1, R[4] & m1, R[0] & m2, R[4] & m2, R[0] & m3, R[4] & m3);
    res[1] = VSET(R[1] & m0, R[5] & m0, R[1] & m1, R[5] & m1, R[1] & m2, R[5] & m2, R[1] & m3, R[5] & m3);
    res[2] = VSET(R[2] & m0, R[6] & m0, R[2] & m1, R[6] & m1, R[2] & m2, R[6] & m2, R[2] & m3, R[6] & m3);
    res[3] = VSET(R[3] & m0, R[7] & m0, R[3] & m1, R[7] & m1, R[3] & m2, R[7] & m2, R[3] & m3, R[7] & m3);
}

void fp_copy_4x2(__m512i* res, __m512i* a)
{
    res[0] = a[0];
    res[1] = a[1];
    res[2] = a[2];
    res[3] = a[3];
}

void fp_duplicate_4x2(__m512i* res, const uint64_t* a)
{
    res[0] = VSET(a[0], a[4], a[0], a[4], a[0], a[4], a[0], a[4]);
    res[1] = VSET(a[1], a[5], a[1], a[5], a[1], a[5], a[1], a[5]);
    res[2] = VSET(a[2], a[6], a[2], a[6], a[2], a[6], a[2], a[6]);
    res[3] = VSET(a[3], a[7], a[3], a[7], a[3], a[7], a[3], a[7]);
}

int fp_is_equal_4x2(__m512i* a, __m512i* b)
{
    __m512i norm_a[N4x2], norm_b[N4x2];
    fp_normalize_4x2(norm_a, a);
    fp_normalize_4x2(norm_b, b);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], norm_b[0]);
    acc &= VCMPEQ(norm_a[1], norm_b[1]);
    acc &= VCMPEQ(norm_a[2], norm_b[2]);
    acc &= VCMPEQ(norm_a[3], norm_b[3]);
    
    // a = b iff both halves are equal
    return (acc & (acc >> 1) & 0x1)
        | ((acc >> 1) & (acc >> 2) & 0x2)
        | ((acc >> 2) & (acc >> 3) & 0x4)
        | ((acc >> 3) & (acc >> 4) & 0x8);
}

int fp_is_zero_4x2(__m512i* a)
{
    __m512i norm_a[N4x2];
    const __m512i zero = VZERO();
    
    // Normalize a to [0, p)
    fp_normalize_4x2(norm_a, a);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], zero);
    acc &= VCMPEQ(norm_a[1], zero);
    acc &= VCMPEQ(norm_a[2], zero);
    acc &= VCMPEQ(norm_a[3], zero);
    
    // a = 0 iff both halves are 0
    return (acc & (acc >> 1) & 0x1)
        | ((acc >> 1) & (acc >> 2) & 0x2)
        | ((acc >> 2) & (acc >> 3) & 0x4)
        | ((acc >> 3) & (acc >> 4) & 0x8);
}

void fp_select_4x2(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)
{
    __m512i t[4];
    __mmask8 mask = (s & 0x1) | ((s & 0x1) << 1)
        | ((s & 0x2) << 1) | ((s & 0x2) << 2)
        | ((s & 0x4) << 2) | ((s & 0x4) << 3)
        | ((s & 0x8) << 3) | ((s & 0x8) << 4);
    
    t[0] = VBLEND(mask, a0[0], a1[0]);
    t[1] = VBLEND(mask, a0[1], a1[1]);
    t[2] = VBLEND(mask, a0[2], a1[2]);
    t[3] = VBLEND(mask, a0[3], a1[3]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

void fp_add_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[4], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(2, 0, 2, 0, 2, 0, 2, 0);
    const __m512i p_top = VSET(0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP);
    
    // Add limbs
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    t[3] = VADD(a[3], b[3]);
    
    // Subtract 2p, using special shape of p
    t[0] = VADD(t[0], p_bot);
    t[3] = VSUB(t[3], p_top);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    carry = VSRA(t[3], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VANDMASK(t[3], 0x55, t[3], limb_mask);
    
    // If result is negative, we undo the subtraction by 2p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[3], 63);
    t[3] = VADD(t[3], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    carry = VSRA(t[3], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VANDMASK(t[3], 0x55, t[3], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

void fp_sub_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[4], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(2, 0, 2, 0, 2, 0, 2, 0);
    const __m512i p_top = VSET(0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP);
    
    // Subtract limbs
    t[0] = VSUB(a[0], b[0]);
    t[1] = VSUB(a[1], b[1]);
    t[2] = VSUB(a[2], b[2]);
    t[3] = VSUB(a[3], b[3]);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    carry = VSRA(t[3], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VANDMASK(t[3], 0x55, t[3], limb_mask);
    
    // If result is negative, we add 2p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[3], 63);
    t[3] = VADD(t[3], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    carry = VSRA(t[3], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VANDMASK(t[3], 0x55, t[3], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

void fp_neg_4x2(__m512i* res, __m512i* a)
{
    __m512i t[4], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i zero = VZERO();
    const __m512i p_bot = VSET(2, 0, 2, 0, 2, 0, 2, 0);
    const __m512i p_top = VSET(0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP);
    
    // Subtract limbs
    t[0] = VSUB(zero, a[0]);
    t[1] = VSUB(zero, a[1]);
    t[2] = VSUB(zero, a[2]);
    t[3] = VSUB(zero, a[3]);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    carry = VSRA(t[3], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VANDMASK(t[3], 0x55, t[3], limb_mask);
    
    // If result is negative, we add 2p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[3], 63);
    t[3] = VADD(t[3], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    carry = VSRA(t[3], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VANDMASK(t[3], 0x55, t[3], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

void fp_mixaddsub_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[4], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(2, 0, 2, 0, 2, 0, 2, 0);
    const __m512i p_top = VSET(0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP, 0, 2*P_PLUS_1_TOP);
    const __mmask8 sub_mask = ~add_mask;
    
    t[0] = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t[1] = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t[2] = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    t[3] = VSUBMASK(a[3], sub_mask, a[3], b[3]);
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    t[2] = VADDMASK(t[2], add_mask, a[2], b[2]);
    t[3] = VADDMASK(t[3], add_mask, a[3], b[3]);
    
    // We subtract 2p, using the special shape of p
    t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);
    t[3] = VSUBMASK(t[3], add_mask, t[3], p_top);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    carry = VSRA(t[3], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VANDMASK(t[3], 0x55, t[3], limb_mask);
    
    // If result is negative, we add 2p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[3], 63);
    t[3] = VADD(t[3], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    carry = VSRA(t[3], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VANDMASK(t[3], 0x55, t[3], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

void fp_normalize_4x2(__m512i* res, __m512i* a)
{
    __m512i t[4], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    
    // Subtract p, using special shape of p
    t[0] = VADD(a[0], p_bot);
    t[1] = a[1];
    t[2] = a[2];
    t[3] = VSUB(a[3], p_top);
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    carry = VSRA(t[3], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VANDMASK(t[3], 0x55, t[3], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    sign_mask = VSRA(t[3], 63);
    t[3] = VADD(t[3], VAND(sign_mask, p_top));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    carry = VSRA(t[3], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t[0] = VADDMASK(t[0], 0xaa, t[0], carry);
    t[1] = VADDMASK(t[1], 0xaa, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0xaa, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0xaa, t[3], VSRA(t[2], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VANDMASK(t[3], 0x55, t[3], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

// Single-length multiprecision addition, no correction, no carry propagation
void mp_add_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[4];
    
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    t[3] = VADD(a[3], b[3]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

void mp_add_p_sub_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[4];
    const __m512i p_bod = VSETALL(2*LIMB_MASK);
    const __m512i p_top = VSET(2*LIMB_MASK, 2*P_PLUS_1_TOP-2, 2*LIMB_MASK, 2*P_PLUS_1_TOP-2, 2*LIMB_MASK, 2*P_PLUS_1_TOP-2, 2*LIMB_MASK, 2*P_PLUS_1_TOP-2);
    
    t[0] = VSUB(VADD(a[0], p_bod), b[0]);
    t[1] = VSUB(VADD(a[1], p_bod), b[1]);
    t[2] = VSUB(VADD(a[2], p_bod), b[2]);
    t[3] = VSUB(VADD(a[3], p_top), b[3]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

void mp_mixaddsubx2_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[12];
    // We can get subtraction result in [0, p*R) by always adding p * 2^(num_limbs * limb_len)
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
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
    
    t[8] = VSUBMASK(t[8], sub_mask, t[8], p_bot);
    t[11] = VADDMASK(t[8], sub_mask, t[11], p_top);
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
}

// Double-length multiprecision subtraction res = res - a - b, no correction
void mp_dblsubx2_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[12];
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
}

// Double-length multiprecision subtraction res = res - a
void mp_subx2_4x2(__m512i* res, __m512i* a)
{
    __m512i t[12];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    // We can get result in [0, p*R) by always adding p * 2^(num_limbs * limb_len)
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    
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
    
    t[8] = VSUB(t[8], p_bot);
    t[11] = VADD(t[11], p_top);
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
}

// Algorithm 2 from "Finite Field Arithmetic Using AVX-512 For Isogeny-Based Cryptography" by Orisaka, Aranha and López
void mp_mul_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[12], sl[4], sh[4];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i zero = VZERO();
    
    // Duplicating limbs of b with shuffles
    // If b[0] = {a_0, a_4, b_0, b_4, ...} then sl0 = {a_0, a_0, b_0, b_0, ...}
    sl[0] = VSHUFFLE(b[0], 0x44);
    sl[1] = VSHUFFLE(b[1], 0x44);
    sl[2] = VSHUFFLE(b[2], 0x44);
    sl[3] = VSHUFFLE(b[3], 0x44);
    // If b[0] = {a_0, a_4, b_0, b_4, ...} then sh0 = {a_4, a_4, b_4, b_4, ...}
    sh[0] = VSHUFFLE(b[0], 0xee);
    sh[1] = VSHUFFLE(b[1], 0xee);
    sh[2] = VSHUFFLE(b[2], 0xee);
    sh[3] = VSHUFFLE(b[3], 0xee);
    
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
    
    t[4] = VMULADDL(t[4], a[0], sh[0]);
    t[5] = VMULADDH(zero, a[0], sh[0]);
    t[4] = VMULADDL(t[4], a[1], sl[3]);
    t[5] = VMULADDH(t[5], a[1], sl[3]);
    t[4] = VMULADDL(t[4], a[2], sl[2]);
    t[5] = VMULADDH(t[5], a[2], sl[2]);
    t[4] = VMULADDL(t[4], a[3], sl[1]);
    t[5] = VMULADDH(t[5], a[3], sl[1]);
    t[5] = VSL(t[5], LIMB_ALIGN);
    
    t[5] = VMULADDL(t[5], a[0], sh[1]);
    t[6] = VMULADDH(zero, a[0], sh[1]);
    t[5] = VMULADDL(t[5], a[1], sh[0]);
    t[6] = VMULADDH(t[6], a[1], sh[0]);
    t[5] = VMULADDL(t[5], a[2], sl[3]);
    t[6] = VMULADDH(t[6], a[2], sl[3]);
    t[5] = VMULADDL(t[5], a[3], sl[2]);
    t[6] = VMULADDH(t[6], a[3], sl[2]);
    t[6] = VSL(t[6], LIMB_ALIGN);
    
    t[6] = VMULADDL(t[6], a[0], sh[2]);
    t[7] = VMULADDH(zero, a[0], sh[2]);
    t[6] = VMULADDL(t[6], a[1], sh[1]);
    t[7] = VMULADDH(t[7], a[1], sh[1]);
    t[6] = VMULADDL(t[6], a[2], sh[0]);
    t[7] = VMULADDH(t[7], a[2], sh[0]);
    t[6] = VMULADDL(t[6], a[3], sl[3]);
    t[7] = VMULADDH(t[7], a[3], sl[3]);
    t[7] = VSL(t[7], LIMB_ALIGN);
    
    t[7] = VMULADDL(t[7], a[0], sh[3]);
    t[8] = VMULADDH(zero, a[0], sh[3]);
    t[7] = VMULADDL(t[7], a[1], sh[2]);
    t[8] = VMULADDH(t[8], a[1], sh[2]);
    t[7] = VMULADDL(t[7], a[2], sh[1]);
    t[8] = VMULADDH(t[8], a[2], sh[1]);
    t[7] = VMULADDL(t[7], a[3], sh[0]);
    t[8] = VMULADDH(t[8], a[3], sh[0]);
    t[8] = VSL(t[8], LIMB_ALIGN);
    
    t[8] = VMULADDL(t[8], a[1], sh[3]);
    t[9] = VMULADDH(zero, a[1], sh[3]);
    t[8] = VMULADDL(t[8], a[2], sh[2]);
    t[9] = VMULADDH(t[9], a[2], sh[2]);
    t[8] = VMULADDL(t[8], a[3], sh[1]);
    t[9] = VMULADDH(t[9], a[3], sh[1]);
    t[9] = VSL(t[9], LIMB_ALIGN);
    
    t[9] = VMULADDL(t[9], a[2], sh[3]);
    t[10] = VMULADDH(zero, a[2], sh[3]);
    t[9] = VMULADDL(t[9], a[3], sh[2]);
    t[10] = VMULADDH(t[10], a[3], sh[2]);
    t[10] = VSL(t[10], LIMB_ALIGN);
    
    t[10] = VMULADDL(t[10], a[3], sh[3]);
    t[11] = VMULADDH(zero, a[3], sh[3]);
    t[11] = VSL(t[11], LIMB_ALIGN);
    
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
    res[12] = zero;
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
    
    // Montgomery reduction loop
    // using the fact that -p^{-1} mod 2^(limb_len) = 1 + c* 2^f mod 2^(limb_len) = 1 as f > limb_len
    carry = VSRA(a[0], LIMB_LEN);
    a[0] = VAND(a[0], limb_mask);
    a[1] = VADD(a[1], carry);
    // a = a + u_i * p * 2^(limb_len * i)
    // To multiply by p, we multiply by (p+1) and subtract one.
    // However, since we will drop the lower limbs when we divide later, we don't need to subtract.
    shuf = VSHUFFLEMASKZ(0xcccc, a[0], 0x4e);
    a[3] = VMULADDL(a[3], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[4] = VADD(a[4], hi);
    // Folding step
    a[4] = VADDMASK(a[4], 0x55, a[4], VSHUFFLE(a[0], 0x4e));
    
    carry = VSRA(a[1], LIMB_LEN);
    a[1] = VAND(a[1], limb_mask);
    a[2] = VADD(a[2], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[1], 0x4e);
    a[4] = VMULADDL(a[4], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[5] = VADD(a[5], hi);
    a[5] = VADDMASK(a[5], 0x55, a[5], VSHUFFLE(a[1], 0x4e));
    
    carry = VSRA(a[2], LIMB_LEN);
    a[2] = VAND(a[2], limb_mask);
    a[3] = VADD(a[3], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[2], 0x4e);
    a[5] = VMULADDL(a[5], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[6] = VADD(a[6], hi);
    a[6] = VADDMASK(a[6], 0x55, a[6], VSHUFFLE(a[2], 0x4e));
    
    carry = VSRA(a[3], LIMB_LEN);
    a[3] = VAND(a[3], limb_mask);
    a[4] = VADD(a[4], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[3], 0x4e);
    a[6] = VMULADDL(a[6], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[7] = VADD(a[7], hi);
    a[7] = VADDMASK(a[7], 0x55, a[7], VSHUFFLE(a[3], 0x4e));
    
    carry = VSRA(a[4], LIMB_LEN);
    a[4] = VAND(a[4], limb_mask);
    a[5] = VADD(a[5], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[4], 0x4e);
    a[7] = VMULADDL(a[7], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[8] = VADD(a[8], hi);
    a[8] = VADDMASK(a[8], 0x55, a[8], VSHUFFLE(a[4], 0x4e));
    
    carry = VSRA(a[5], LIMB_LEN);
    a[5] = VAND(a[5], limb_mask);
    a[6] = VADD(a[6], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[5], 0x4e);
    a[8] = VMULADDL(a[8], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[9] = VADD(a[9], hi);
    a[9] = VADDMASK(a[9], 0x55, a[9], VSHUFFLE(a[5], 0x4e));
    
    carry = VSRA(a[6], LIMB_LEN);
    a[6] = VAND(a[6], limb_mask);
    a[7] = VADD(a[7], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[6], 0x4e);
    a[9] = VMULADDL(a[9], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[10] = VADD(a[10], hi);
    a[10] = VADDMASK(a[10], 0x55, a[10], VSHUFFLE(a[6], 0x4e));
    
    carry = VSRA(a[7], LIMB_LEN);
    a[7] = VAND(a[7], limb_mask);
    a[8] = VADD(a[8], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[7], 0x4e);
    a[10] = VMULADDL(a[10], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[11] = VADD(a[11], hi);
    a[11] = VADDMASK(a[11], 0x55, a[11], VSHUFFLE(a[7], 0x4e));
    
    carry = VSRA(a[8], LIMB_LEN);
    a[8] = VAND(a[8], limb_mask);
    a[9] = VADD(a[9], carry);
    shuf = VSHUFFLEMASKZ(0xcccc, a[8], 0x4e);
    a[11] = VMULADDL(a[11], shuf, top);
    hi = VMULADDH(zero, shuf, top);
    hi = VSL(hi, LIMB_ALIGN);
    a[12] = VADD(a[12], hi);
    a[12] = VADDMASK(a[12], 0x55, a[12], VSHUFFLE(a[8], 0x4e));
    
    // Result is now in a[9], ..., a[12]
    // Carry propagation
    a[10] = VADDMASK(a[10], 0x55, a[10], VSRA(a[9], LIMB_LEN));
    a[11] = VADDMASK(a[11], 0x55, a[11], VSRA(a[10], LIMB_LEN));
    a[12] = VADDMASK(a[12], 0x55, a[12], VSRA(a[11], LIMB_LEN));
    carry = VSRA(a[12], LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    a[9] = VADDMASK(a[9], 0xaa, a[9], carry);
    a[10] = VADDMASK(a[10], 0xaa, a[10], VSRA(a[9], LIMB_LEN));
    a[11] = VADDMASK(a[11], 0xaa, a[11], VSRA(a[10], LIMB_LEN));
    a[12] = VADDMASK(a[12], 0xaa, a[12], VSRA(a[11], LIMB_LEN));
    a[9] = VAND(a[9], limb_mask);
    a[10] = VAND(a[10], limb_mask);
    a[11] = VAND(a[11], limb_mask);
    a[12] = VANDMASK(a[12], 0x55, a[12], limb_mask);
    
    res[0] = a[9];
    res[1] = a[10];
    res[2] = a[11];
    res[3] = a[12];
}

void fp_mul_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[13];
    
    mp_mul_4x2(t, a, b);
    rdc_4x2(res, t);
}

void fp_sqr_4x2(__m512i* res, __m512i* a)
{
    fp_mul_4x2(res, a, a);
}

void fp_div2_4x2(__m512i* res, __m512i* src)
{
    const __m512i a0 = src[0], a1 = src[1], a2 = src[2], a3 = src[3];
    __m512i t0, t1, t2, t3;
    const __m512i one = VSETALL(1);
    const __m512i cnst = VSET(0, P_PLUS_1_DIV2_TOP, 0, P_PLUS_1_DIV2_TOP, 0, P_PLUS_1_DIV2_TOP, 0, P_PLUS_1_DIV2_TOP);
    __m512i odd_mask;
    
    // odd_mask = all 1s if a is odd, else 0
    odd_mask = VSHUFFLE(a0, 0x44); // (a[0], a[0])
    odd_mask = VSUB(VZERO(), VAND(odd_mask, one));
    
    // a >> 1
    t0 = VOR(VSRL(a0, 1), VSL(VAND(a1, one), 47));
    t1 = VOR(VSRL(a1, 1), VSL(VAND(a2, one), 47));
    t2 = VOR(VSRL(a2, 1), VSL(VAND(a3, one), 47));
    t3 = VSRL(a3, 1);
    t3 = VADDMASK(t3, 0x55, t3, VSL(VAND(VSHUFFLE(a0, 0xee), one), 47));
    
    // Conditionally add (p+1)/2 = c * 2^(e-1)
    t3 = VADD(t3, VAND(cnst, odd_mask));
    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3;
}

void fp_div3_4x2(__m512i* res, __m512i* src)
{
    __m512i t[4];
    fp_duplicate_4x2(t, INV3);
    fp_mul_4x2(res, src, t);
}

// Batched inversion.
void fp_inv_4x2(__m512i* res, __m512i* a)
{
    uint64_t r_64[12];
    __m512i s0[2], s1[2], r[2];
    __m512i t0[4], t1[4];
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
    for (int i = 0; i < 4; i++) {
        t1[i] = VPERM512(idx1, a[i]);
    }
    fp_mul_4x2(res, t0, t1);
}

void fp_sqrt_4x2(__m512i* res, __m512i* a)
{
    // Candidate root is a^((p+1)/4), with (p+1)/4 = 65*2^374
    __m512i t[4];
    
    fp_sqr_4x2(t, a);
    for (int i = 1; i < 6; i++) {
        fp_sqr_4x2(t, t);
    }
    fp_mul_4x2(t, t, a);
    for (int i = 0; i < 373; i++) {
        fp_sqr_4x2(t, t);
    }
    fp_sqr_4x2(res, t);
}

// Cost: 376 squarings, 24 multiplications
void fp_exp3div4_4x2(__m512i* res, __m512i* a)
{
    __m512i v__10[4];
    __m512i v__100[4];
    __m512i v__101[4];
    __m512i v__111[4];
    __m512i v__1110[4];
    __m512i v__11100[4];
    __m512i v__111000[4];
    __m512i v__1110000[4];
    __m512i v_i12[4];
    __m512i v_i131[4];
    __m512i v_i132[4];
    __m512i v_i133[4];
    __m512i v_i139[4];
    __m512i v_i143[4];
    __m512i v_i144[4];
    __m512i v_i19[4];
    __m512i v_i36[4];
    __m512i v_i68[4];
    __m512i v_x122[4];
    __m512i v_x125[4];
    
    fp_sqr_4x2(v__10, a);
    fp_sqr_4x2(v__100, v__10);
    fp_mul_4x2(v__101, a, v__100);
    fp_mul_4x2(v__111, v__10, v__101);
    fp_sqr_4x2(v__1110, v__111);
    fp_sqr_4x2(v__11100, v__1110);
    fp_sqr_4x2(v__111000, v__11100);
    fp_sqr_4x2(v__1110000, v__111000);
    fp_sqr_4x2(v_i12, v__1110000);
    fp_sqr_4x2(v_i12, v_i12);
    fp_sqr_4x2(v_i12, v_i12);
    fp_mul_4x2(v_i12, v_i12, v__1110000);
    fp_sqr_4x2(v_i19, v_i12);
    for (int i = 0; i < 5; i++) fp_sqr_4x2(v_i19, v_i19);
    fp_mul_4x2(v_i19, v_i19, v_i12);
    fp_sqr_4x2(v_i36, v_i19);
    fp_sqr_4x2(v_i36, v_i36);
    fp_mul_4x2(v_i36, v_i36, v__111000);
    for (int i = 0; i < 13; i++) fp_sqr_4x2(v_i36, v_i36);
    fp_mul_4x2(v_i36, v_i36, v_i19);
    fp_sqr_4x2(v_i68, v_i36);
    fp_sqr_4x2(v_i68, v_i68);
    fp_mul_4x2(v_i68, v_i68, v__111000);
    for (int i = 0; i < 28; i++) fp_sqr_4x2(v_i68, v_i68);
    fp_mul_4x2(v_i68, v_i68, v_i36);
    fp_sqr_4x2(v_i131, v_i68);
    fp_mul_4x2(v_i131, v_i131, v__11100);
    for (int i = 0; i < 59; i++) fp_sqr_4x2(v_i131, v_i131);
    fp_mul_4x2(v_i131, v_i131, v_i68);
    fp_mul_4x2(v_i131, v_i131, v__101);
    fp_mul_4x2(v_i132, v__111, v_i131);
    fp_mul_4x2(v_i133, v__1110, v_i132);
    fp_mul_4x2(v_x122, v_i131, v_i133);
    fp_sqr_4x2(v_i139, v_x122);
    fp_mul_4x2(v_i139, v_i139, v_x122);
    fp_sqr_4x2(v_i139, v_i139);
    fp_mul_4x2(v_i139, v_i139, v_x122);
    fp_mul_4x2(v_i139, v_i139, v_i133);
    fp_mul_4x2(v_x125, v_i132, v_i139);
    fp_sqr_4x2(v_i143, v_x125);
    fp_mul_4x2(v_i143, v_i143, v_x125);
    fp_mul_4x2(v_i143, v_i143, v_i139);
    fp_mul_4x2(v_i144, v_x122, v_i143);
    fp_sqr_4x2(res, v_i144);
    for (int i = 0; i < 127; i++) fp_sqr_4x2(res, res);
    fp_mul_4x2(res, res, v_i144);
    fp_mul_4x2(res, res, v_i143);
    for (int i = 0; i < 125; i++) fp_sqr_4x2(res, res);
    fp_mul_4x2(res, res, v_x125);
}

void fp_tomont_4x2(__m512i* res, __m512i* a)
{
    __m512i vR2[4];
    fp_duplicate_4x2(vR2, R2);
    fp_mul_4x2(res, a, vR2);
}

void fp_frommont_4x2(__m512i* res, __m512i* a)
{
    __m512i one[4];
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
        t[2*i+1] = unvec[NWORDS64 * i + 3] & LIMB_MASK;
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = ((unvec[NWORDS64 * i] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 1] & 0xffffffffULL) << 16);
        t[2*i+1] = ((unvec[NWORDS64 * i + 3] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 4] & 0xffffffffULL) << 16);
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = ((unvec[NWORDS64 * i + 1] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 2] & 0xffffULL) << 32);
        t[2*i+1] = ((unvec[NWORDS64 * i + 4] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 5] & 0xffffULL) << 32);
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = (unvec[NWORDS64 * i + 2] >> 16) & LIMB_MASK;
        t[2*i+1] = (unvec[NWORDS64 * i + 5] >> 16) & LIMB_MASK;
    }
    res[3] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

void fp_limbs_to_64_4x2(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[4];
    fp_normalize_4x2(s, src);
    
    VSTORE(t, s[0]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i] = t[2*i] & LIMB_MASK;
        
        res[NWORDS64 * i + 3] = t[2*i+1] & LIMB_MASK;
    }
    VSTORE(t, s[1]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i] |= (t[2*i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 1] = (t[2*i] >> 16) & 0xffffffffULL;
        
        res[NWORDS64 * i + 3] |= (t[2*i+1] & 0xffffULL) << 48;
        res[NWORDS64 * i + 4] = (t[2*i+1] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, s[2]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 1] |= (t[2*i] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 2] = (t[2*i] >> 32) & 0xffffULL;
        
        res[NWORDS64 * i + 4] |= (t[2*i+1] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 5] = (t[2*i+1] >> 32) & 0xffffULL;
    }
    VSTORE(t, s[3]);
    for (int i = 0; i < 4; i++) {
        res[NWORDS64 * i + 2] |= (t[2*i] & LIMB_MASK) << 16;
        
        res[NWORDS64 * i + 5] |= (t[2*i+1] & LIMB_MASK) << 16;
    }
}

void rand_fp_4x2(__m512i* res)
{
    __m512i zero[4];
    __m512i mask = VSETALL(LIMB_MASK);
    fp_setzero_4x2(zero);
    
    for (int i = 0; i < 3; i++) {
        res[i] = VAND(rand_512i(), mask);
    }
    
    // Last limb is shorter
    mask = VSET(LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK);
    res[3] = VAND(rand_512i(), mask);
    
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

// If bits[i] = 1, the i-th Fp element is set to 1 in the Montgomery representation.
void fp_set_binary_2x4(__m512i* res, __mmask8 bits)
{
    uint64_t m0 = -(bits & 0x1);
    uint64_t m1 = -((bits >> 1) & 0x1);
    
    res[0] = VSET(R[0] & m0, R[2] & m0, R[4] & m0, R[6] & m0, R[0] & m1, R[2] & m1, R[4] & m1, R[6] & m1);
    res[1] = VSET(R[1] & m0, R[3] & m0, R[5] & m0, R[7] & m0, R[1] & m1, R[3] & m1, R[5] & m1, R[7] & m1);
}

void fp_copy_2x4(__m512i* res, __m512i* a)
{
    res[0] = a[0];
    res[1] = a[1];
}

void fp_duplicate_2x4(__m512i* res, const uint64_t* a)
{
    res[0] = VSET(a[0], a[2], a[4], a[6], a[0], a[2], a[4], a[6]);
    res[1] = VSET(a[1], a[3], a[5], a[7], a[1], a[3], a[5], a[7]);
}

int fp_is_equal_2x4(__m512i* a, __m512i* b)
{
    __m512i norm_a[2], norm_b[2];
    fp_normalize_2x4(norm_a, a);
    fp_normalize_2x4(norm_b, b);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], norm_b[0]);
    acc &= VCMPEQ(norm_a[1], norm_b[1]);
    
    return (acc & (acc >> 1) & (acc >> 2) & (acc >> 3) & 0x1)
        | ((acc >> 3) & (acc >> 4) & (acc >> 5) & (acc >> 6) & 0x2);
}

int fp_is_zero_2x4(__m512i* a)
{
    __m512i norm_a[2];
    const __m512i zero = VZERO();
    
    fp_normalize_2x4(norm_a, a);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], zero);
    acc &= VCMPEQ(norm_a[1], zero);
    
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
    const __m512i p_bot = VSET(2, 0, 0, 0, 2, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, 0, 2*P_PLUS_1_TOP, 0, 0, 0, 2*P_PLUS_1_TOP);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // For duplicating sign bit in each half
    const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);
    
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    
    // Subtract 2p using the special shape of p
    t[0] = VADD(t[0], p_bot);
    t[1] = VSUB(t[1], p_top);
    
    // Carry propagation (3 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    
    // If the result is negative, add 2p back
    sign_mask = VSRA(t[1], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[1] = VADD(t[1], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (3 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
}

void fp_sub_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[2], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(2, 0, 0, 0, 2, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, 0, 2*P_PLUS_1_TOP, 0, 0, 0, 2*P_PLUS_1_TOP);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // For duplicating sign bit in each half
    const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);
    
    t[0] = VSUB(a[0], b[0]);
    t[1] = VSUB(a[1], b[1]);
    
    // Carry propagation (3 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    
    // If the result is negative, we add 2p
    sign_mask = VSRA(t[1], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[1] = VADD(t[1], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (3 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
}

void fp_neg_2x4(__m512i* res, __m512i* a)
{
    __m512i t[2], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(2, 0, 0, 0, 2, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, 0, 2*P_PLUS_1_TOP, 0, 0, 0, 2*P_PLUS_1_TOP);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // For duplicating sign bit in each half
    const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);
    const __m512i zero = VZERO();
    
    t[0] = VSUB(zero, a[0]);
    t[1] = VSUB(zero, a[1]);
    
    // Carry propagation (3 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    
    // If the result is negative, we add 2p
    sign_mask = VSRA(t[1], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[1] = VADD(t[1], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (3 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
}

void fp_mixaddsub_2x4(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[2], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(2, 0, 0, 0, 2, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, 0, 2*P_PLUS_1_TOP, 0, 0, 0, 2*P_PLUS_1_TOP);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // For duplicating sign bit in each half
    const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);
    const __mmask8 sub_mask = ~add_mask;
    
    t[0] = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t[1] = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    // Subtract 2p using the special shape of p
    t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);
    t[1] = VSUBMASK(t[1], add_mask, t[1], p_top);
    
    // Carry propagation (3 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    
    // If the result is negative, we add 2p
    sign_mask = VSRA(t[1], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[1] = VADD(t[1], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (3 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
}

void fp_normalize_2x4(__m512i* res, __m512i* a)
{
    __m512i t[2], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, 0, P_PLUS_1_TOP, 0, 0, 0, P_PLUS_1_TOP);
    // Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation
    const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);
    // For duplicating sign bit in each half
    const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);
    
    // Subtract p using the special shape of p
    t[0] = VADD(a[0], p_bot);
    t[1] = VSUB(a[1], p_top);
    
    // Carry propagation (3 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    
    // If the result is negative, add p back
    sign_mask = VSRA(t[1], 63);
    sign_mask = VPERM512(idx_sign, sign_mask);
    t[1] = VADD(t[1], VAND(p_top, sign_mask));
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    
    // Carry propagation (3 cross-lane ripples + final straight ripple)
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    carry = VSRA(t[1], LIMB_LEN);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    carry = VPERM512MASKZ(0xee, carry, idx_shift);
    t[0] = VADD(t[0], carry);
    // Final straight ripple
    carry = VSRA(t[0], LIMB_LEN);
    t[0] = VAND(t[0], limb_mask);
    t[1] = VADD(t[1], carry);
    t[1] = VANDMASK(t[1], 0x77, t[1], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
}

void mp_mul_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[10];
    __m512i B[8];
    const __m512i zero = VZERO();
    
    // Create a vector for each limb of b
    B[0] = VPERM256(b[0], 0x0);
    B[1] = VPERM256(b[1], 0x0);
    B[2] = VPERM256(b[0], 0x55);
    B[3] = VPERM256(b[1], 0x55);
    B[4] = VPERM256(b[0], 0xaa);
    B[5] = VPERM256(b[1], 0xaa);
    B[6] = VPERM256(b[0], 0xff);
    B[7] = VPERM256(b[1], 0xff);
    
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
    
    t[5] = VMULADDL(t[5], a[0], B[5]);
    t[6] = VMULADDH(zero, a[0], B[5]);
    t[5] = VMULADDL(t[5], a[1], B[4]);
    t[6] = VMULADDH(t[6], a[1], B[4]);
    t[6] = VSL(t[6], LIMB_ALIGN);
    
    t[6] = VMULADDL(t[6], a[0], B[6]);
    t[7] = VMULADDH(zero, a[0], B[6]);
    t[6] = VMULADDL(t[6], a[1], B[5]);
    t[7] = VMULADDH(t[7], a[1], B[5]);
    t[7] = VSL(t[7], LIMB_ALIGN);
    
    t[7] = VMULADDL(t[7], a[0], B[7]);
    t[8] = VMULADDH(zero, a[0], B[7]);
    t[7] = VMULADDL(t[7], a[1], B[6]);
    t[8] = VMULADDH(t[8], a[1], B[6]);
    t[8] = VSL(t[8], LIMB_ALIGN);
    
    t[8] = VMULADDL(t[8], a[1], B[7]);
    t[9] = VMULADDH(zero, a[1], B[7]);
    t[9] = VSL(t[9], LIMB_ALIGN);
    
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

// Modifies a. Its length must be at least 17.
void rdc_2x4(__m512i* res, __m512i* a)
{
    __m512i carry, lo, hi, shuf;
    const __m512i zero = VZERO();
    const __m512i p_top_all = VSETALL(P_PLUS_1_TOP);
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(2, 0, 0, 0, 2, 0, 0, 0);
    const __m512i p_top = VSET(0, 0, 0, 2*P_PLUS_1_TOP, 0, 0, 0, 2*P_PLUS_1_TOP);
    
    // Initialize the upper accumulators to zero
    a[10] = zero;
    a[11] = zero;
    a[12] = zero;
    a[13] = zero;
    a[14] = zero;
    a[15] = zero;
    a[16] = zero;
    
    // Move each limb one step to the left, for each half
    const __m512i idx = VSET(1, 2, 3, 0, 5, 6, 7, 0);
    
    // Generalized Montgomery reduction loop
    carry = VSRA(a[0], LIMB_LEN);
    a[0] = VAND(a[0], limb_mask);
    a[1] = VADD(a[1], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[0], 0x4e);
    a[5] = VMULADDL(a[5], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[6] = VADD(a[6], VSL(hi, LIMB_ALIGN));
    a[2] = VADD(a[2], VPERM512MASKZ(0x77, a[0], idx));
    
    carry = VSRA(a[1], LIMB_LEN);
    a[1] = VAND(a[1], limb_mask);
    a[2] = VADD(a[2], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[1], 0x4e);
    a[6] = VMULADDL(a[6], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[7] = VADD(a[7], VSL(hi, LIMB_ALIGN));
    a[3] = VADD(a[3], VPERM512MASKZ(0x77, a[1], idx));
    
    carry = VSRA(a[2], LIMB_LEN);
    a[2] = VAND(a[2], limb_mask);
    a[3] = VADD(a[3], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[2], 0x4e);
    a[7] = VMULADDL(a[7], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[8] = VADD(a[8], VSL(hi, LIMB_ALIGN));
    a[4] = VADD(a[4], VPERM512MASKZ(0x77, a[2], idx));
    
    carry = VSRA(a[3], LIMB_LEN);
    a[3] = VAND(a[3], limb_mask);
    a[4] = VADD(a[4], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[3], 0x4e);
    a[8] = VMULADDL(a[8], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[9] = VADD(a[9], VSL(hi, LIMB_ALIGN));
    a[5] = VADD(a[5], VPERM512MASKZ(0x77, a[3], idx));
    
    carry = VSRA(a[4], LIMB_LEN);
    a[4] = VAND(a[4], limb_mask);
    a[5] = VADD(a[5], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[4], 0x4e);
    a[9] = VMULADDL(a[9], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[10] = VADD(a[10], VSL(hi, LIMB_ALIGN));
    a[6] = VADD(a[6], VPERM512MASKZ(0x77, a[4], idx));
    
    carry = VSRA(a[5], LIMB_LEN);
    a[5] = VAND(a[5], limb_mask);
    a[6] = VADD(a[6], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[5], 0x4e);
    a[10] = VMULADDL(a[10], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[11] = VADD(a[11], VSL(hi, LIMB_ALIGN));
    a[7] = VADD(a[7], VPERM512MASKZ(0x77, a[5], idx));
    
    carry = VSRA(a[6], LIMB_LEN);
    a[6] = VAND(a[6], limb_mask);
    a[7] = VADD(a[7], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[6], 0x4e);
    a[11] = VMULADDL(a[11], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[12] = VADD(a[12], VSL(hi, LIMB_ALIGN));
    a[8] = VADD(a[8], VPERM512MASKZ(0x77, a[6], idx));
    
    carry = VSRA(a[7], LIMB_LEN);
    a[7] = VAND(a[7], limb_mask);
    a[8] = VADD(a[8], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[7], 0x4e);
    a[12] = VMULADDL(a[12], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[13] = VADD(a[13], VSL(hi, LIMB_ALIGN));
    a[9] = VADD(a[9], VPERM512MASKZ(0x77, a[7], idx));
    
    carry = VSRA(a[8], LIMB_LEN);
    a[8] = VAND(a[8], limb_mask);
    a[9] = VADD(a[9], carry);
    shuf = VSHUFFLEMASKZ(0x0c0c, a[8], 0x4e);
    a[13] = VMULADDL(a[13], shuf, p_top_all);
    hi = VMULADDH(zero, shuf, p_top_all);
    a[14] = VADD(a[14], VSL(hi, LIMB_ALIGN));
    a[10] = VADD(a[10], VPERM512MASKZ(0x77, a[8], idx));
    
    // Propagate carries
    carry = VSRA(a[9], LIMB_LEN);
    a[9] = VAND(a[9], limb_mask);
    a[10] = VADD(a[10], carry);
    a[11] = VADD(a[11], VPERM512MASKZ(0x77, a[9], idx));
    
    carry = VSRA(a[10], LIMB_LEN);
    a[10] = VAND(a[10], limb_mask);
    a[11] = VADD(a[11], carry);
    a[12] = VADD(a[12], VPERM512MASKZ(0x77, a[10], idx));
    
    carry = VSRA(a[11], LIMB_LEN);
    a[11] = VAND(a[11], limb_mask);
    a[12] = VADD(a[12], carry);
    a[13] = VADD(a[13], VPERM512MASKZ(0x77, a[11], idx));
    
    carry = VSRA(a[12], LIMB_LEN);
    a[12] = VAND(a[12], limb_mask);
    a[13] = VADD(a[13], carry);
    a[14] = VADD(a[14], VPERM512MASKZ(0x77, a[12], idx));
    
    carry = VSRA(a[13], LIMB_LEN);
    a[13] = VAND(a[13], limb_mask);
    a[14] = VADD(a[14], carry);
    a[15] = VADD(a[15], VPERM512MASKZ(0x77, a[13], idx));
    
    carry = VSRA(a[14], LIMB_LEN);
    a[14] = VAND(a[14], limb_mask);
    a[15] = VADD(a[15], carry);
    a[16] = VADD(a[16], VPERM512MASKZ(0x77, a[14], idx));
    
    carry = VSRA(a[15], LIMB_LEN);
    a[15] = VAND(a[15], limb_mask);
    a[16] = VADD(a[16], carry);
    
    // Pack result from pos 0 of the accumulators back into 2x4 format
    const __m512i idx_pack = VSET(0, 1, 8, 9, 4, 5, 12, 13);
    lo = VUNPACKLO(a[9], a[11]);
    hi = VUNPACKLO(a[13], a[15]);
    res[0] = VBLENDPERM(idx_pack, lo, hi);
    
    lo = VUNPACKLO(a[10], a[12]);
    hi = VUNPACKLO(a[14], a[16]);
    res[1] = VBLENDPERM(idx_pack, lo, hi);
    
    
}

void fp_mul_2x4(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[17];
    
    mp_mul_2x4(t, a, b);
    rdc_2x4(res, t);
}

void fp_sqr_2x4(__m512i* res, __m512i* a)
{
    __m512i t[17];
    
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

void fp_div2_2x4(__m512i* res, __m512i* src)
{
    const __m512i a0 = src[0], a1 = src[1];
    __m512i t0, t1;
    const __m512i one = VSETALL(1);
    const __m512i cnst = VSET(0, 0, 0, P_PLUS_1_DIV2_TOP, 0, 0, 0, P_PLUS_1_DIV2_TOP);
    __m512i odd_mask;
    
    // odd_mask = all 1s if a is odd, else 0
    odd_mask = VPERM256(a0, 0x00); // (a[0], a[0], a[0], a[0])
    odd_mask = VSUB(VZERO(), VAND(odd_mask, one));
    
    // a >> 1
    t0 = VSRL(a0, 1);
    t0 = VADD(t0, VSL(VAND(a1, one), 47));
    t1 = VSRL(a1, 1);
    t1 = VADDMASK(t1, 0x77, t1, VPERM256(VSL(VAND(a0, one), 47), 0x39));
    
    // Conditionally add (p+1)/2 = c * 2^(e-1)
    t1 = VADD(t1, VAND(cnst, odd_mask));
    
    res[0] = t0; res[1] = t1;
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
    uint64_t r_64[12];
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
    // Candidate root is a^((p+1)/4), with (p+1)/4 = 65*2^374
    __m512i t[2];
    
    fp_sqr_2x4(t, a);
    for (int i = 1; i < 6; i++) {
        fp_sqr_2x4(t, t);
    }
    fp_mul_2x4(t, t, a);
    for (int i = 0; i < 373; i++) {
        fp_sqr_2x4(t, t);
    }
    fp_sqr_2x4(res, t);
}

// Cost: 376 squarings, 24 multiplications
void fp_exp3div4_2x4(__m512i* res, __m512i* a)
{
    __m512i v__10[2];
    __m512i v__100[2];
    __m512i v__101[2];
    __m512i v__111[2];
    __m512i v__1110[2];
    __m512i v__11100[2];
    __m512i v__111000[2];
    __m512i v__1110000[2];
    __m512i v_i12[2];
    __m512i v_i131[2];
    __m512i v_i132[2];
    __m512i v_i133[2];
    __m512i v_i139[2];
    __m512i v_i143[2];
    __m512i v_i144[2];
    __m512i v_i19[2];
    __m512i v_i36[2];
    __m512i v_i68[2];
    __m512i v_x122[2];
    __m512i v_x125[2];
    
    fp_sqr_2x4(v__10, a);
    fp_sqr_2x4(v__100, v__10);
    fp_mul_2x4(v__101, a, v__100);
    fp_mul_2x4(v__111, v__10, v__101);
    fp_sqr_2x4(v__1110, v__111);
    fp_sqr_2x4(v__11100, v__1110);
    fp_sqr_2x4(v__111000, v__11100);
    fp_sqr_2x4(v__1110000, v__111000);
    fp_sqr_2x4(v_i12, v__1110000);
    fp_sqr_2x4(v_i12, v_i12);
    fp_sqr_2x4(v_i12, v_i12);
    fp_mul_2x4(v_i12, v_i12, v__1110000);
    fp_sqr_2x4(v_i19, v_i12);
    for (int i = 0; i < 5; i++) fp_sqr_2x4(v_i19, v_i19);
    fp_mul_2x4(v_i19, v_i19, v_i12);
    fp_sqr_2x4(v_i36, v_i19);
    fp_sqr_2x4(v_i36, v_i36);
    fp_mul_2x4(v_i36, v_i36, v__111000);
    for (int i = 0; i < 13; i++) fp_sqr_2x4(v_i36, v_i36);
    fp_mul_2x4(v_i36, v_i36, v_i19);
    fp_sqr_2x4(v_i68, v_i36);
    fp_sqr_2x4(v_i68, v_i68);
    fp_mul_2x4(v_i68, v_i68, v__111000);
    for (int i = 0; i < 28; i++) fp_sqr_2x4(v_i68, v_i68);
    fp_mul_2x4(v_i68, v_i68, v_i36);
    fp_sqr_2x4(v_i131, v_i68);
    fp_mul_2x4(v_i131, v_i131, v__11100);
    for (int i = 0; i < 59; i++) fp_sqr_2x4(v_i131, v_i131);
    fp_mul_2x4(v_i131, v_i131, v_i68);
    fp_mul_2x4(v_i131, v_i131, v__101);
    fp_mul_2x4(v_i132, v__111, v_i131);
    fp_mul_2x4(v_i133, v__1110, v_i132);
    fp_mul_2x4(v_x122, v_i131, v_i133);
    fp_sqr_2x4(v_i139, v_x122);
    fp_mul_2x4(v_i139, v_i139, v_x122);
    fp_sqr_2x4(v_i139, v_i139);
    fp_mul_2x4(v_i139, v_i139, v_x122);
    fp_mul_2x4(v_i139, v_i139, v_i133);
    fp_mul_2x4(v_x125, v_i132, v_i139);
    fp_sqr_2x4(v_i143, v_x125);
    fp_mul_2x4(v_i143, v_i143, v_x125);
    fp_mul_2x4(v_i143, v_i143, v_i139);
    fp_mul_2x4(v_i144, v_x122, v_i143);
    fp_sqr_2x4(res, v_i144);
    for (int i = 0; i < 127; i++) fp_sqr_2x4(res, res);
    fp_mul_2x4(res, res, v_i144);
    fp_mul_2x4(res, res, v_i143);
    for (int i = 0; i < 125; i++) fp_sqr_2x4(res, res);
    fp_mul_2x4(res, res, v_x125);
}

// Assumes unvec is an array of 2 * NWORDS64 64-bit ints
void fp_64_to_limbs_2x4(__m512i* res, uint64_t* unvec)
{
    uint64_t t[8];
    
    for (int i = 0; i < 2; i++) {
        t[4*i] = unvec[NWORDS64 * i] & LIMB_MASK;
        t[4*i + 1] = ((unvec[NWORDS64 * i + 1] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 2] & 0xffffULL) << 32);
        t[4*i + 2] = unvec[NWORDS64 * i + 3] & LIMB_MASK;
        t[4*i+3] = ((unvec[NWORDS64 * i + 4] >> 32) & 0xffffffffULL) | ((unvec[NWORDS64 * i + 5] & 0xffffULL) << 32);
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 2; i++) {
        t[4*i] = ((unvec[NWORDS64 * i] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 1] & 0xffffffffULL) << 16);
        t[4*i + 1] = (unvec[NWORDS64 * i + 2] >> 16) & LIMB_MASK;
        t[4*i + 2] = ((unvec[NWORDS64 * i + 3] >> 48) & 0xffffULL) | ((unvec[NWORDS64 * i + 4] & 0xffffffffULL) << 16);
        t[4*i+3] = (unvec[NWORDS64 * i + 5] >> 16) & LIMB_MASK;
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

void fp_limbs_to_64_2x4(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    __m512i s[2];
    fp_normalize_2x4(s, src);
    
    VSTORE(t, s[0]);
    for (int i = 0; i < 2; i++) {
        res[NWORDS64 * i] = t[4*i] & LIMB_MASK;
        res[NWORDS64 * i + 1] = (t[4*i + 1] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 2] = (t[4*i + 1] >> 32) & 0xffffULL;
        res[NWORDS64 * i + 3] = t[4*i + 2] & LIMB_MASK;
        res[NWORDS64 * i + 4] = (t[4*i+3] & 0xffffffffULL) << 32;
        res[NWORDS64 * i + 5] = (t[4*i+3] >> 32) & 0xffffULL;
    }
    VSTORE(t, s[1]);
    for (int i = 0; i < 2; i++) {
        res[NWORDS64 * i] |= (t[4*i] & 0xffffULL) << 48;
        res[NWORDS64 * i + 1] |= (t[4*i] >> 16) & 0xffffffffULL;
        res[NWORDS64 * i + 2] |= (t[4*i + 1] & LIMB_MASK) << 16;
        res[NWORDS64 * i + 3] |= (t[4*i + 2] & 0xffffULL) << 48;
        res[NWORDS64 * i + 4] |= (t[4*i + 2] >> 16) & 0xffffffffULL;
        res[NWORDS64 * i + 5] |= (t[4*i+3] & LIMB_MASK) << 16;
    }
}

void rand_fp_2x4(__m512i* res)
{
    __m512i zero[2];
    __m512i mask;
    fp_setzero_2x4(zero);
    
    mask = VSET(LIMB_MASK, LIMB_MASK, LIMB_MASK, LIMB_MASK, LIMB_MASK, LIMB_MASK, LIMB_MASK, LIMB_MASK);
    res[0] = VAND(rand_512i(), mask);
    mask = VSET(LIMB_MASK, LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK);
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
    for (int i = 0; i < 4; i++) {
        s = src[i], t = src[i + 4];
        res1[i] = VBLENDPERM(idx1, s, t);
        res2[i] = VBLENDPERM(idx2, s, t);
    }
}

// Given src1 = (a1, b1, c1, d1) and src2 = (a2, b2, c2, d2),
// it computes res = (a1, a2, b1, b2, c1, c2, d1, d2).
void fp_4x2_to_8x1(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i s, t;
    const __m512i idx1 = VSET(0, 8, 2, 10, 4, 12, 6, 14);
    const __m512i idx2 = VSET(1, 9, 3, 11, 5, 13, 7, 15);
    for (int i = 0; i < 4; i++) {
        s = src1[i], t = src2[i];
        res[i] = VBLENDPERM(idx1, s, t);
        res[i + 4] = VBLENDPERM(idx2, s, t);
    }
}

// Given src=(a, b, c, d), it computes res1 = (a, c) and res2 = (b, d).
// We choose this interleaving pattern for the Fp2 2x2x2 sqrt,
// where res1 = re and res2 = im.
void fp_4x2_to_2x4(__m512i* res1, __m512i* res2, __m512i* src)
{
    __m512i idx1 = VSET(0, 8, 1, 9, 4, 12, 5, 13);
    __m512i idx2 = VSET(2, 10, 3, 11, 6, 14, 7, 15);
    for (int i = 0; i < 2; i++) {
        __m512i s = src[i], t = src[i+2];
        res1[i] = VBLENDPERM(idx1, s, t);
        res2[i] = VBLENDPERM(idx2, s, t);
    }
}

void fp_2x4_to_4x2(__m512i* res, __m512i* src1, __m512i* src2)
{
    __m512i idx1 = VSET(0, 2, 8, 10, 4, 6, 12, 14);
    __m512i idx2 = VSET(1, 3, 9, 11, 5, 7, 13, 15);
    for (int i = 0; i < 2; i++) {
        __m512i s = src1[i], t = src2[i];
        res[i] = VBLENDPERM(idx1, s, t);
        res[i+2] = VBLENDPERM(idx2, s, t);
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
    uint64_t t[6];
    unsigned char cc;
    const uint64_t nz = ~is_zero64(src);
    
    // p - a
    cc = sbb(0, (uint64_t) 0xffffffffffffffff, src[0], t);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[1], t+1);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[2], t+2);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[3], t+3);
    cc = sbb(cc, (uint64_t) 0xffffffffffffffff, src[4], t+4);
    cc = sbb(cc, (uint64_t) 0x40ffffffffffffff, src[5], t+5);
    
    res[0] = t[0] & nz;
    res[1] = t[1] & nz;
    res[2] = t[2] & nz;
    res[3] = t[3] & nz;
    res[4] = t[4] & nz;
    res[5] = t[5] & nz;
}

// res = f*u + g*v mod p
// Assumes f,g < 2^62
void lin64(uint64_t* res, const uint64_t* u, const uint64_t* v, uint64_t f, uint64_t g)
{
    uint64_t t[6], tu[6], tv[6], hi;
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
    
    // Reduction
    // t = L + 2^e H
    // Since p = c* 2^e - 1, 2^e = 1/c mod p
    // We fold the output as t = L + 2^e * (H mod c) + 2^e * floor(H/c)
    
    // Compute H in (h0, h1) and L in (t[0], ..., t[5]):
    uint64_t h0 = (t[5] >> 56) | (hi << 8);
    uint64_t h1 = (hi >> 56);
    t[5] &= 0x00FFFFFFFFFFFFFFULL;
    
    // Divide H by c
    __uint128_t H_full = ((__uint128_t)h1 << 64) | h0;
    uint64_t quo = (uint64_t)(H_full / 0x41ULL);
    uint64_t rem = (uint64_t)(H_full % 0x41ULL);
    
    // t = L + quo + rem * 2^e
    cc = adc(0, t[0], quo, &t[0]);
    cc = adc(cc, t[1], 0, &t[1]);
    cc = adc(cc, t[2], 0, &t[2]);
    cc = adc(cc, t[3], 0, &t[3]);
    cc = adc(cc, t[4], 0, &t[4]);
    adc(cc, t[5], rem << 56, &t[5]);
    
    // Finally, if t > p, we subtract p to reduce fully.
    cc = sbb(0, t[0], 0xffffffffffffffffULL, tu);
    cc = sbb(cc, t[1], 0xffffffffffffffffULL, tu + 1);
    cc = sbb(cc, t[2], 0xffffffffffffffffULL, tu + 2);
    cc = sbb(cc, t[3], 0xffffffffffffffffULL, tu + 3);
    cc = sbb(cc, t[4], 0xffffffffffffffffULL, tu + 4);
    cc = sbb(cc, t[5], 0x40ffffffffffffffULL, tu + 5);
    // If no borrow (t >= p), use tu; otherwise keep t
    uint64_t mask = -(uint64_t)cc;
    select64(res, tu, t, mask);
}

// d = abs(floor((a*fc + b*gc) / 2^31))
// Coefficients fc, gc are signed, |fc|,|gc| <= 2^31.
// Returns 0xFFFFFFFFFFFFFFFF if a*fc + b*gc < 0, else 0.
uint64_t lindiv31abs(uint64_t* d, const uint64_t* a, const uint64_t* b, uint64_t fc, uint64_t gc)
{
    uint64_t ta[7], tb[7], t[7], hi;
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
    (void)sbb(cc, 0, 0, &ta[6]);
    
    // gc = abs(gc), conditionally negate b
    uint64_t sg = sgnw(gc);
    gc = (gc ^ sg) - sg;
    cc = sbb(0, b[0] ^ sg, sg, &tb[0]);
    cc = sbb(cc, b[1] ^ sg, sg, &tb[1]);
    cc = sbb(cc, b[2] ^ sg, sg, &tb[2]);
    cc = sbb(cc, b[3] ^ sg, sg, &tb[3]);
    cc = sbb(cc, b[4] ^ sg, sg, &tb[4]);
    cc = sbb(cc, b[5] ^ sg, sg, &tb[5]);
    (void)sbb(cc, 0, 0, &tb[6]);
    
    // Compute ta*fc + tb*gc as a signed integer
    umulx2_64(&t[0], &hi, ta[0], fc, tb[0], gc);
    umulx2_add_64(&t[1], &hi, ta[1], fc, tb[1], gc, hi);
    umulx2_add_64(&t[2], &hi, ta[2], fc, tb[2], gc, hi);
    umulx2_add_64(&t[3], &hi, ta[3], fc, tb[3], gc, hi);
    umulx2_add_64(&t[4], &hi, ta[4], fc, tb[4], gc, hi);
    umulx2_add_64(&t[5], &hi, ta[5], fc, tb[5], gc, hi);
    t[6] = hi - (ta[6] & fc) - (tb[6] & gc);
    
    // Right-shift by 31 bits
    t[0] = (t[0] >> 31) | (t[1] << 33);
    t[1] = (t[1] >> 31) | (t[2] << 33);
    t[2] = (t[2] >> 31) | (t[3] << 33);
    t[3] = (t[3] >> 31) | (t[4] << 33);
    t[4] = (t[4] >> 31) | (t[5] << 33);
    t[5] = (t[5] >> 31) | (t[6] << 33);
    
    // If result is negative, negate it
    uint64_t s = sgnw(t[6]);
    cc = sbb(0, t[0] ^ s, s, &d[0]);
    cc = sbb(cc, t[1] ^ s, s, &d[1]);
    cc = sbb(cc, t[2] ^ s, s, &d[2]);
    cc = sbb(cc, t[3] ^ s, s, &d[3]);
    cc = sbb(cc, t[4] ^ s, s, &d[4]);
    (void)sbb(cc, t[5] ^ s, s, &d[5]);
    
    return s;
}

// Montgomery multiplication: res = a * b / 2^384 mod p
// p = 65 * 2^376 - 1
void fp_mul_64(uint64_t* res, const uint64_t* a, const uint64_t* b)
{
    const uint64_t P[6] = {
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0xffffffffffffffffULL,
        0x40ffffffffffffffULL
    };
    
    uint64_t e[12], lo, hi;
    uint64_t s[6], h[7];
    unsigned char cc;
    
    // Schoolbook multiplication: e = a * b
    umul_64(&e[0], &hi, a[0], b[0]);
    umul_add_64(&e[1], &hi, a[0], b[1], hi);
    umul_add_64(&e[2], &hi, a[0], b[2], hi);
    umul_add_64(&e[3], &hi, a[0], b[3], hi);
    umul_add_64(&e[4], &hi, a[0], b[4], hi);
    umul_add_64(&e[5], &hi, a[0], b[5], hi);
    e[6] = hi;
    
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
    e[7] = hi + (uint64_t)cc;
    
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
    e[8] = hi + (uint64_t)cc;
    
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
    e[9] = hi + (uint64_t)cc;
    
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
    e[10] = hi + (uint64_t)cc;
    
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
    e[11] = hi + (uint64_t)cc;
    
    // Montgomery reduction
    // m = -1/p mod 2^(64 * n) = 65 * 2^376 + 1) 
    // s = low(e) * m mod 2^(64 * n)
    for (int i = 0; i < 6; i++) s[i] = e[i];
    s[5] += (uint64_t)((e[0] & 0xFFULL) * 65ULL) << 56;
    
    // h = s * (65 << 56)
    umul_64(&h[0], &hi, s[0], 4683743612465315840ULL);
    umul_add_64(&h[1], &hi, s[1], 4683743612465315840ULL, hi);
    umul_add_64(&h[2], &hi, s[2], 4683743612465315840ULL, hi);
    umul_add_64(&h[3], &hi, s[3], 4683743612465315840ULL, hi);
    umul_add_64(&h[4], &hi, s[4], 4683743612465315840ULL, hi);
    umul_add_64(&h[5], &hi, s[5], 4683743612465315840ULL, hi);
    h[6] = hi;
    
    // e += s*p = s*c*2^f - s
    // Step A: e -= s
    cc = 0;
    for (int i = 0; i < 6; i++) cc = sbb(cc, e[i], s[i], &e[i]);
    for (int i = 6; i < 12; i++) cc = sbb(cc, e[i], 0, &e[i]);
    
    // Step B: e += h at word position 5
    cc = 0;
    for (int i = 0; i <= 6; i++) cc = adc(cc, e[5 + i], h[i], &e[5 + i]);
    
    // Conditional subtraction of p for full reduction
    uint64_t r[6];
    cc = 0;
    for (int i = 0; i < 6; i++) cc = sbb(cc, e[6 + i], P[i], &r[i]);
    uint64_t mask = -(uint64_t)cc;
    for (int i = 0; i < 6; i++) res[i] = (e[6 + i] & mask) | (r[i] & ~mask);
}

// Implements inversion mod p using a constant-time extended binary GCD algorithm
// by Thomas Pornin: https://eprint.iacr.org/2020/972.
// This code is based on the version implented in the SQIsign submission.
// It asssumes the input is normalized and in the Montgomery representation with R.
void fp_inv_64(uint64_t* res, const uint64_t* src)
{
    uint64_t a[6], b[6], u[6], v[6];
    uint64_t xa, xb, f0, g0, f1, g1;
    
    // Initialize a = src, b = p, u = 1 * R, v = 0
    for (int i = 0; i < 6; i++) {
        a[i] = src[i];
        b[i] = P_64[i];
        u[i] = R_64[i];
        v[i] = 0;
    }
    
    // Main loop: 23 outer iterations of k-1=31 inner halvings
    for (int i = 0; i < 23; i++) {
        // Get 64-bit approximations xa, xb of a, b
        uint64_t tnzm = 0, tnza = 0, tnzb = 0, snza = 0, snzb = 0;
        uint64_t found = 0;
        for (int k = 5; k >= 1; k--) {
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
        uint64_t na[6], nb[6], nu[6], nv[6];
        uint64_t nega = lindiv31abs(na, a, b, f0, g0);
        uint64_t negb = lindiv31abs(nb, a, b, f1, g1);
        f0 = (f0 ^ nega) - nega;
        g0 = (g0 ^ nega) - nega;
        f1 = (f1 ^ negb) - negb;
        g1 = (g1 ^ negb) - negb;
        lin64(nu, u, v, f0, g0);
        lin64(nv, u, v, f1, g1);
        for (int k = 0; k < 6; k++) {
            a[k] = na[k]; b[k] = nb[k]; u[k] = nu[k]; v[k] = nv[k];
        }
    }
    
    // Final scalar loop: 52 iterations
    xa = a[0]; xb = b[0];
    f0 = 1; g0 = 0; f1 = 0; g1 = 1;
    for (int j = 0; j < 52; j++) {
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

