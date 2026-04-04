#include <stdio.h>
#include <stdint.h>

#include "fp_lvl3.h"
#include "intrinsics_avx512.h"
#include "utils.h"

#define LIMB_LEN 51
#define NUM_LIMBS 10
#define LIMB_LEN 52 - LIMB_LEN
#define LIMB_MASK 0x7ffffffffffffULL
#define LAST_LIMB_MASK 0x3fffffffffffULL

#define P_PLUS_1_TOP 0x360000000000ULL

// Montgomery constant R = 2^(11*51) mod p
static const uint64_t R[NUM_LIMBS] = {
    0x7684bda12f684, 0x25, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0,
    0x0, 0x280000000000
};

// R2 = R^2 mod p
static const uint64_t R2[NUM_LIMBS] = {
    0xbf08c77657ca, 0x514b802cf301c, 0x97b425ed0f19, 0x4bda12f684bda,
    0x5ed097b425ed0, 0x7684bda12f684, 0x3425ed097b425, 0x212f684bda12f,
    0x97b425ed097b, 0x21a12f684bda
};

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
    
    // Subtract p, using special shape of p
    t[0] = VADD(t[0], p_bot);
    t[9] = VSUB(t[9], p_top);
    
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
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t[9], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[9] = VADD(t[9], VAND(p_top, sign_mask));
    
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
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t[9], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[9] = VADD(t[9], VAND(p_top, sign_mask));
    
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
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t[9], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[9] = VADD(t[9], VAND(p_top, sign_mask));
    
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
    
    // Subtract p, using special shape of p
    t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);
    t[9] = VSUBMASK(t[9], add_mask, t[9], p_top);
    
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
    
    // If the result is negative, we add p
    sign_mask = VSRA(t[9], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[9] = VADD(t[9], VAND(p_top, sign_mask));
    
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

