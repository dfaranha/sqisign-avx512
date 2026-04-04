#include <stdio.h>
#include <stdint.h>

#include "fp_lvl3.h"
#include "intrinsics_avx512.h"
#include "utils.h"

#define LIMB_LEN 48
#define NUM_LIMBS 8
#define LIMB_LEN 52 - LIMB_LEN
#define NUM_LIMBS_64 6

#define N4x2 4
#define N2x4 2

#define LIMB_MASK 0xffffffffffffULL
#define LAST_LIMB_MASK 0x7fffffffffffULL

#define P_PLUS_1_TOP 0x410000000000ULL

// Montgomery constant R = 2^(9*48) mod p
static const uint64_t R[NUM_LIMBS] = {
    0xf03f03f03f03, 0x3, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x3d0000000000
};

// R2 = R^2 mod p
static const uint64_t R2[NUM_LIMBS] = {
    0x7b0fd07f000f, 0x732f52771f91, 0xf03f03f03f13, 0xf03f03f03f03,
    0xf03f03f03f03, 0xf03f03f03f03, 0xf03f03f03f03, 0x213f03f03f03
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
    
    // Subtract p, using special shape of p
    t[0] = VADD(t[0], p_bot);
    t[7] = VSUB(t[7], p_top);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t[7], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[7] = VADD(t[7], VAND(p_top, sign_mask));
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
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
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t[7], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[7] = VADD(t[7], VAND(p_top, sign_mask));
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
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
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    
    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t[7], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[7] = VADD(t[7], VAND(p_top, sign_mask));
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
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
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    t[2] = VADDMASK(t[2], add_mask, a[2], b[2]);
    t[3] = VADDMASK(t[3], add_mask, a[3], b[3]);
    t[4] = VADDMASK(t[4], add_mask, a[4], b[4]);
    t[5] = VADDMASK(t[5], add_mask, a[5], b[5]);
    t[6] = VADDMASK(t[6], add_mask, a[6], b[6]);
    t[7] = VADDMASK(t[7], add_mask, a[7], b[7]);
    
    // Subtract p, using special shape of p
    t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);
    t[7] = VSUBMASK(t[7], add_mask, t[7], p_top);
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    
    // If the result is negative, we add p
    sign_mask = VSRA(t[7], 63);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));
    t[7] = VADD(t[7], VAND(p_top, sign_mask));
    
    // Carry propagation
    t[1] = VADD(t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADD(t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADD(t[3], VSRA(t[2], LIMB_LEN));
    t[4] = VADD(t[4], VSRA(t[3], LIMB_LEN));
    t[5] = VADD(t[5], VSRA(t[4], LIMB_LEN));
    t[6] = VADD(t[6], VSRA(t[5], LIMB_LEN));
    t[7] = VADD(t[7], VSRA(t[6], LIMB_LEN));
    t[0] = VAND(t[0], limb_mask);
    t[1] = VAND(t[1], limb_mask);
    t[2] = VAND(t[2], limb_mask);
    t[3] = VAND(t[3], limb_mask);
    t[4] = VAND(t[4], limb_mask);
    t[5] = VAND(t[5], limb_mask);
    t[6] = VAND(t[6], limb_mask);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
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
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
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
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
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
}

// Computes res = a + p - b without any carry propagation or conditional masks
void mp_add_p_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[NUM_LIMBS];
    __m512i p_bot = VSETALL(1);
    __m512i p_top = VSETALL(P_PLUS_1_TOP);
    
    // Add p
    t[0] = VSUB(a[0], p_bot);
    t[7] = VADD(a[7], p_top);
    
    t[0] = VSUB(t[0], b[0]);
    t[1] = VSUB(a[1], b[1]);
    t[2] = VSUB(a[2], b[2]);
    t[3] = VSUB(a[3], b[3]);
    t[4] = VSUB(a[4], b[4]);
    t[5] = VSUB(a[5], b[5]);
    t[6] = VSUB(a[6], b[6]);
    t[7] = VSUB(t[7], b[7]);
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
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
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
}

void mp_mul_8x1_karat(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[16], ta[4], tb[4], m[8];
    const __m512i zero = VZERO();
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    // a_low * b_low
    t[0] = VMULADDL(zero, a[0], b[0]);
    t[1] = VMULADDH(zero, a[0], b[0]);
    t[1] = VSL(t[1], LIMB_ALIGN);
    
    t[1] = VMULADDL(t[1], a[0], b[1]);
    t[2] = VMULADDH(zero, a[0], b[1]);
    t[1] = VMULADDL(t[1], a[1], b[0]);
    t[2] = VMULADDH(t[2], a[1], b[0]);
    t[2] = VSL(t[2], LIMB_ALIGN);
    
    t[2] = VMULADDL(t[2], a[0], b[2]);
    t[3] = VMULADDH(zero, a[0], b[2]);
    t[2] = VMULADDL(t[2], a[1], b[1]);
    t[3] = VMULADDH(t[3], a[1], b[1]);
    t[2] = VMULADDL(t[2], a[2], b[0]);
    t[3] = VMULADDH(t[3], a[2], b[0]);
    t[3] = VSL(t[3], LIMB_ALIGN);
    
    t[3] = VMULADDL(t[3], a[0], b[3]);
    t[4] = VMULADDH(zero, a[0], b[3]);
    t[3] = VMULADDL(t[3], a[1], b[2]);
    t[4] = VMULADDH(t[4], a[1], b[2]);
    t[3] = VMULADDL(t[3], a[2], b[1]);
    t[4] = VMULADDH(t[4], a[2], b[1]);
    t[3] = VMULADDL(t[3], a[3], b[0]);
    t[4] = VMULADDH(t[4], a[3], b[0]);
    t[4] = VSL(t[4], LIMB_ALIGN);
    
    t[4] = VMULADDL(t[4], a[1], b[3]);
    t[5] = VMULADDH(zero, a[1], b[3]);
    t[4] = VMULADDL(t[4], a[2], b[2]);
    t[5] = VMULADDH(t[5], a[2], b[2]);
    t[4] = VMULADDL(t[4], a[3], b[1]);
    t[5] = VMULADDH(t[5], a[3], b[1]);
    t[5] = VSL(t[5], LIMB_ALIGN);
    
    t[5] = VMULADDL(t[5], a[2], b[3]);
    t[6] = VMULADDH(zero, a[2], b[3]);
    t[5] = VMULADDL(t[5], a[3], b[2]);
    t[6] = VMULADDH(t[6], a[3], b[2]);
    t[6] = VSL(t[6], LIMB_ALIGN);
    
    t[6] = VMULADDL(t[6], a[3], b[3]);
    t[7] = VMULADDH(zero, a[3], b[3]);
    t[7] = VSL(t[7], LIMB_ALIGN);
    
    // a_hi * b_hi
    t[8] = VMULADDL(zero, a[4], b[4]);
    t[9] = VMULADDH(zero, a[4], b[4]);
    t[9] = VSL(t[9], LIMB_ALIGN);
    
    t[9] = VMULADDL(t[9], a[4], b[5]);
    t[10] = VMULADDH(zero, a[4], b[5]);
    t[9] = VMULADDL(t[9], a[5], b[4]);
    t[10] = VMULADDH(t[10], a[5], b[4]);
    t[10] = VSL(t[10], LIMB_ALIGN);
    
    t[10] = VMULADDL(t[10], a[4], b[6]);
    t[11] = VMULADDH(zero, a[4], b[6]);
    t[10] = VMULADDL(t[10], a[5], b[5]);
    t[11] = VMULADDH(t[11], a[5], b[5]);
    t[10] = VMULADDL(t[10], a[6], b[4]);
    t[11] = VMULADDH(t[11], a[6], b[4]);
    t[11] = VSL(t[11], LIMB_ALIGN);
    
    t[11] = VMULADDL(t[11], a[4], b[7]);
    t[12] = VMULADDH(zero, a[4], b[7]);
    t[11] = VMULADDL(t[11], a[5], b[6]);
    t[12] = VMULADDH(t[12], a[5], b[6]);
    t[11] = VMULADDL(t[11], a[6], b[5]);
    t[12] = VMULADDH(t[12], a[6], b[5]);
    t[11] = VMULADDL(t[11], a[7], b[4]);
    t[12] = VMULADDH(t[12], a[7], b[4]);
    t[12] = VSL(t[12], LIMB_ALIGN);
    
    t[12] = VMULADDL(t[12], a[5], b[7]);
    t[13] = VMULADDH(zero, a[5], b[7]);
    t[12] = VMULADDL(t[12], a[6], b[6]);
    t[13] = VMULADDH(t[13], a[6], b[6]);
    t[12] = VMULADDL(t[12], a[7], b[5]);
    t[13] = VMULADDH(t[13], a[7], b[5]);
    t[13] = VSL(t[13], LIMB_ALIGN);
    
    t[13] = VMULADDL(t[13], a[6], b[7]);
    t[14] = VMULADDH(zero, a[6], b[7]);
    t[13] = VMULADDL(t[13], a[7], b[6]);
    t[14] = VMULADDH(t[14], a[7], b[6]);
    t[14] = VSL(t[14], LIMB_ALIGN);
    
    t[14] = VMULADDL(t[14], a[7], b[7]);
    t[15] = VMULADDH(zero, a[7], b[7]);
    t[15] = VSL(t[15], LIMB_ALIGN);
    
    // ta = a_low + a_hi
    ta[0] = VADD(a[0], a[4]);
    ta[1] = VADD(a[1], a[5]);
    ta[2] = VADD(a[2], a[6]);
    ta[3] = VADD(a[3], a[7]);
    
    // tb = b_low + b_hi
    tb[0] = VADD(b[0], b[4]);
    tb[1] = VADD(b[1], b[5]);
    tb[2] = VADD(b[2], b[6]);
    tb[3] = VADD(b[3], b[7]);
    
    // m = ta * tb
    m[0] = VMULADDL(zero, ta[0], tb[0]);
    m[1] = VMULADDH(zero, ta[0], tb[0]);
    m[1] = VSL(m[1], LIMB_ALIGN);
    
    m[1] = VMULADDL(m[1], ta[0], tb[1]);
    m[2] = VMULADDH(zero, ta[0], tb[1]);
    m[1] = VMULADDL(m[1], ta[1], tb[0]);
    m[2] = VMULADDH(m[2], ta[1], tb[0]);
    m[2] = VSL(m[2], LIMB_ALIGN);
    
    m[2] = VMULADDL(m[2], ta[0], tb[2]);
    m[3] = VMULADDH(zero, ta[0], tb[2]);
    m[2] = VMULADDL(m[2], ta[1], tb[1]);
    m[3] = VMULADDH(m[3], ta[1], tb[1]);
    m[2] = VMULADDL(m[2], ta[2], tb[0]);
    m[3] = VMULADDH(m[3], ta[2], tb[0]);
    m[3] = VSL(m[3], LIMB_ALIGN);
    
    m[3] = VMULADDL(m[3], ta[0], tb[3]);
    m[4] = VMULADDH(zero, ta[0], tb[3]);
    m[3] = VMULADDL(m[3], ta[1], tb[2]);
    m[4] = VMULADDH(m[4], ta[1], tb[2]);
    m[3] = VMULADDL(m[3], ta[2], tb[1]);
    m[4] = VMULADDH(m[4], ta[2], tb[1]);
    m[3] = VMULADDL(m[3], ta[3], tb[0]);
    m[4] = VMULADDH(m[4], ta[3], tb[0]);
    m[4] = VSL(m[4], LIMB_ALIGN);
    
    m[4] = VMULADDL(m[4], ta[1], tb[3]);
    m[5] = VMULADDH(zero, ta[1], tb[3]);
    m[4] = VMULADDL(m[4], ta[2], tb[2]);
    m[5] = VMULADDH(m[5], ta[2], tb[2]);
    m[4] = VMULADDL(m[4], ta[3], tb[1]);
    m[5] = VMULADDH(m[5], ta[3], tb[1]);
    m[5] = VSL(m[5], LIMB_ALIGN);
    
    m[5] = VMULADDL(m[5], ta[2], tb[3]);
    m[6] = VMULADDH(zero, ta[2], tb[3]);
    m[5] = VMULADDL(m[5], ta[3], tb[2]);
    m[6] = VMULADDH(m[6], ta[3], tb[2]);
    m[6] = VSL(m[6], LIMB_ALIGN);
    
    m[6] = VMULADDL(m[6], ta[3], tb[3]);
    m[7] = VMULADDH(zero, ta[3], tb[3]);
    m[7] = VSL(m[7], LIMB_ALIGN);
    
    // m = (ta * tb) - (a_low * b_low) - (a_hi * b_hi)
    m[0] = VSUB(m[0], VADD(t[0], t[8]));
    m[1] = VSUB(m[1], VADD(t[1], t[9]));
    m[2] = VSUB(m[2], VADD(t[2], t[10]));
    m[3] = VSUB(m[3], VADD(t[3], t[11]));
    m[4] = VSUB(m[4], VADD(t[4], t[12]));
    m[5] = VSUB(m[5], VADD(t[5], t[13]));
    m[6] = VSUB(m[6], VADD(t[6], t[14]));
    m[7] = VSUB(m[7], VADD(t[7], t[15]));
    
    // t += m * 2^(limbs_low)
    t[4] = VADD(t[4], m[0]);
    t[5] = VADD(t[5], m[1]);
    t[6] = VADD(t[6], m[2]);
    t[7] = VADD(t[7], m[3]);
    t[8] = VADD(t[8], m[4]);
    t[9] = VADD(t[9], m[5]);
    t[10] = VADD(t[10], m[6]);
    t[11] = VADD(t[11], m[7]);
    
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
}

void rdc_8x1(__m512i* res, __m512i* a)
{
    // Montgomery reduction with R = 2^n mod p, with n = ((num_limbs+1) * limb_len)
    // low(a) = (a_0, a_1, ... , a_num_limbs)
    // m = -p^(-1) mod 2^n = p + 2 = c * 2^e + 1
    // The output of this reduction is (a + (low(a) * m mod 2^n )*p) / 2^432
    
    __m512i t[10], s7, s8, hi, msb_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i top = VSETALL(P_PLUS_1_TOP);
    const __m512i zero = VZERO();
    const __m512i one = VSETALL(1);
    
    // s = low(a) * m mod 2^n = (a0, a1) * m mod 2^n
    // s0 = a0, ..., s6 = a6
    s7 = VMULADDL(a[7], a[0], top);
    s8 = VMULADDH(zero, a[0], top);
    s8 = VADD(a[8], VSL(s8, LIMB_ALIGN));
    s8 = VMULADDL(s8, a[1], top);
    s8 = VADD(s8, VSR(s7, LIMB_LEN));
    s7 = VAND(s7, limb_mask);
    s8 = VAND(s8, limb_mask);
    
    // Computing a + s*p as a + s*(p+1) - s.
    // As the division by 2^n cancels the lower limbs, we don't need to compute - s.
    t[0] = VMULADDL(zero, a[0], top);
    hi = VMULADDH(zero, a[0], top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t[1] = VMULADDL(hi, a[1], top);
    hi = VMULADDH(zero, a[1], top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t[2] = VMULADDL(hi, a[2], top);
    hi = VMULADDH(zero, a[2], top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t[3] = VMULADDL(hi, a[3], top);
    hi = VMULADDH(zero, a[3], top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t[4] = VMULADDL(hi, a[4], top);
    hi = VMULADDH(zero, a[4], top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t[5] = VMULADDL(hi, a[5], top);
    hi = VMULADDH(zero, a[5], top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t[6] = VMULADDL(hi, a[6], top);
    hi = VMULADDH(zero, a[6], top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t[7] = VMULADDL(hi, s7, top);
    hi = VMULADDH(zero, s7, top);
    hi = VSL(hi, LIMB_ALIGN);
    
    t[8] = VMULADDL(hi, s8, top);
    t[9] = VMULADDH(zero, s8, top);
    t[9] = VSL(t[9], LIMB_ALIGN);
    
    // t += a (ignoring lower limbs)
    t[0] = VADD(t[0], a[7]);
    t[1] = VADD(t[1], a[8]);
    t[2] = VADD(t[2], a[9]);
    t[3] = VADD(t[3], a[10]);
    t[4] = VADD(t[4], a[11]);
    t[5] = VADD(t[5], a[12]);
    t[6] = VADD(t[6], a[13]);
    t[7] = VADD(t[7], a[14]);
    t[8] = VADD(t[8], a[15]);
    
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
    
    // After dividing by 2^n, the result is in t[2], ..., t[9]
    // The result can be 384 bits.
    // If so, we do a final subtraction by p to bring it down to 383 bits.
    hi = VSL(t[9], 16);
    msb_mask = VSIGN(hi);
    t[9] = VSUBMASK(t[9], msb_mask, t[9], top);
    t[2] = VADDMASK(t[2], msb_mask, [t2], one);
    
    // We do not propogate the carry here. The bottom limb can have 49 bits.
    res[0] = t[2];
    res[1] = t[3];
    res[2] = t[4];
    res[3] = t[5];
    res[4] = t[6];
    res[5] = t[7];
    res[6] = t[8];
    res[7] = t[9];
}

void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[16];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    
    mp_mul_8x1_karat(t, a, b);
    
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
    
    rdc_8x1(res, t);
}

void fp_sqr_8x1(__m512i* res, __m512i* a)
{
    fp_mul_8x1(res, a, a);
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

// Assumes unvec is an array of 8 * NUM_LIMBS_64 64-bit ints
void fp_64_to_limbs_8x1(__m512i* res, uint64_t* unvec)
{
    uint64_t t[8];
    
    for (int i = 0; i < 8; i++) {
        t[i] = unvec[NUM_LIMBS_64 * i] & LIMB_MASK;
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NUM_LIMBS_64 * i] >> 48) & 0xffffULL) | ((unvec[NUM_LIMBS_64 * i + 1] & 0xffffffffULL) << 16);
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NUM_LIMBS_64 * i + 1] >> 32) & 0xffffffffULL) | ((unvec[NUM_LIMBS_64 * i + 2] & 0xffffULL) << 32);
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (unvec[NUM_LIMBS_64 * i + 2 >> 16) & LIMB_MASK;
    }
    res[3] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = unvec[NUM_LIMBS_64 * i + 3] & LIMB_MASK;
    }
    res[4] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NUM_LIMBS_64 * i + 3] >> 48) & 0xffffULL) | ((unvec[NUM_LIMBS_64 * i + 4] & 0xffffffffULL) << 16);
    }
    res[5] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((unvec[NUM_LIMBS_64 * i + 4] >> 32) & 0xffffffffULL) | ((unvec[NUM_LIMBS_64 * i + 5] & 0xffffULL) << 32);
    }
    res[6] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (unvec[NUM_LIMBS_64 * i + 5 >> 16) & LIMB_MASK;
    }
    res[7] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

// Assumes unvec is an array of 4 * NUM_LIMBS_64 64-bit ints
void fp_64_to_limbs_8x1_only_even_idx(__m512i* res, uint64_t* unvec)
{
    uint64_t t[4];
    
    for (int i = 0; i < 4; i++) {
        t[i] = unvec[NUM_LIMBS_64 * i] & LIMB_MASK;
    }
    res[0] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NUM_LIMBS_64 * i] >> 48) & 0xffffULL) | ((unvec[NUM_LIMBS_64 * i + 1] & 0xffffffffULL) << 16);
    }
    res[1] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NUM_LIMBS_64 * i + 1] >> 32) & 0xffffffffULL) | ((unvec[NUM_LIMBS_64 * i + 2] & 0xffffULL) << 32);
    }
    res[2] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (unvec[NUM_LIMBS_64 * i + 2] >> 16) & LIMB_MASK;
    }
    res[3] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = unvec[NUM_LIMBS_64 * i + 3] & LIMB_MASK;
    }
    res[4] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NUM_LIMBS_64 * i + 3] >> 48) & 0xffffULL) | ((unvec[NUM_LIMBS_64 * i + 4] & 0xffffffffULL) << 16);
    }
    res[5] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((unvec[NUM_LIMBS_64 * i + 4] >> 32) & 0xffffffffULL) | ((unvec[NUM_LIMBS_64 * i + 5] & 0xffffULL) << 32);
    }
    res[6] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (unvec[NUM_LIMBS_64 * i + 5] >> 16) & LIMB_MASK;
    }
    res[7] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
}

void fp_limbs_to_64_8x1(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    
    VSTORE(t, src[0]);
    for (int i = 0; i < 8; i++) {
        res[NUM_LIMBS_64 * i] = t[i] & LIMB_MASK;
    }
    VSTORE(t, src[1]);
    for (int i = 0; i < 8; i++) {
        res[NUM_LIMBS_64 * i] |= (t[i] & 0xffffULL) << 48;
        res[NUM_LIMBS_64 * i + 1] = (t[i] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, src[2]);
    for (int i = 0; i < 8; i++) {
        res[NUM_LIMBS_64 * i + 1] |= (t[i] & 0xffffffffULL) << 32;
        res[NUM_LIMBS_64 * i + 2] = (t[i] >> 32) & 0xffffULL;
    }
    VSTORE(t, src[3]);
    for (int i = 0; i < 8; i++) {
        res[NUM_LIMBS_64 * i + 2] |= (t[i] & LIMB_MASK) << 16;
    }
    VSTORE(t, src[4]);
    for (int i = 0; i < 8; i++) {
        res[NUM_LIMBS_64 * i + 3] = t[i] & LIMB_MASK;
    }
    VSTORE(t, src[5]);
    for (int i = 0; i < 8; i++) {
        res[NUM_LIMBS_64 * i + 3] |= (t[i] & 0xffffULL) << 48;
        res[NUM_LIMBS_64 * i + 4] = (t[i] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, src[6]);
    for (int i = 0; i < 8; i++) {
        res[NUM_LIMBS_64 * i + 4] |= (t[i] & 0xffffffffULL) << 32;
        res[NUM_LIMBS_64 * i + 5] = (t[i] >> 32) & 0xffffULL;
    }
    VSTORE(t, src[7]);
    for (int i = 0; i < 8; i++) {
        res[NUM_LIMBS_64 * i + 5] |= (t[i] & LIMB_MASK) << 16;
    }
}

void fp_limbs_to_64_8x1_only_even_idx(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    
    VSTORE(t, src[0]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i] = t[2*i] & LIMB_MASK;
    }
    VSTORE(t, src[1]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i] |= (t[2*i] & 0xffffULL) << 48;
        res[NUM_LIMBS_64 * i + 1] = (t[2*i] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, src[2]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i + 1] |= (t[2*i] & 0xffffffffULL) << 32;
        res[NUM_LIMBS_64 * i + 2] = (t[2*i] >> 32) & 0xffffULL;
    }
    VSTORE(t, src[3]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i + 2] |= (t[2*i] & LIMB_MASK) << 16;
    }
    VSTORE(t, src[4]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i + 3] = t[2*i] & LIMB_MASK;
    }
    VSTORE(t, src[5]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i + 3] |= (t[2*i] & 0xffffULL) << 48;
        res[NUM_LIMBS_64 * i + 4] = (t[2*i] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, src[6]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i + 4] |= (t[2*i] & 0xffffffffULL) << 32;
        res[NUM_LIMBS_64 * i + 5] = (t[2*i] >> 32) & 0xffffULL;
    }
    VSTORE(t, src[7]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i + 5] |= (t[2*i] & LIMB_MASK) << 16;
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
    
    // a = 0 iff both halves are equal
    return (acc & (acc >> 1) & 0x1)
        | ((acc >> 1) & (acc >> 2) & 0x2)
        | ((acc >> 2) & (acc >> 3) & 0x4)
        | ((acc >> 3) & (acc >> 4) & 0x8);
}

void fp_add_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[4], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    
    // Add limbs
    t[0] = VADD(a[0], b[0]);
    t[1] = VADD(a[1], b[1]);
    t[2] = VADD(a[2], b[2]);
    t[3] = VADD(a[3], b[3]);
    
    // Subtract p, using special shape of p
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

void fp_sub_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[4], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    
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

void fp_neg_4x2(__m512i* res, __m512i* a)
{
    __m512i t[4], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    const __m512i zero = VZERO();
    
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

void fp_mixaddsub_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[4], carry, sign_mask;
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    const __m512i p_top = VSET(0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP, 0, P_PLUS_1_TOP);
    const __mmask8 sub_mask = ~add_mask;
    
    t[0] = VSUBMASK(a[0], sub_mask, a[0], b[0]);
    t[1] = VSUBMASK(a[1], sub_mask, a[1], b[1]);
    t[2] = VSUBMASK(a[2], sub_mask, a[2], b[2]);
    t[3] = VSUBMASK(a[3], sub_mask, a[3], b[3]);
    
    t[0] = VADDMASK(t[0], add_mask, a[0], b[0]);
    t[1] = VADDMASK(t[1], add_mask, a[1], b[1]);
    t[2] = VADDMASK(t[2], add_mask, a[2], b[2]);
    t[3] = VADDMASK(t[3], add_mask, a[3], b[3]);
    
    // a + b - p
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
    const __m512i p_bod = VSETALL(LIMB_MASK);
    const __m512i p_top = VSET(LIMB_MASK, P_PLUS_1_TOP-1, LIMB_MASK, P_PLUS_1_TOP-1, LIMB_MASK, P_PLUS_1_TOP-1, LIMB_MASK, P_PLUS_1_TOP-1);
    
    t[0] = VSUB(VADD(a[0], p_bod), b[0]);
    t[1] = VSUB(VADD(a[1], p_bod), b[1]);
    t[2] = VSUB(VADD(a[2], p_bod), b[2]);
    t[3] = VSUB(VADD(a[3], p_top), b[3])
    
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

void mp_mixaddsubx2_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[12];
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

// Algorithm 3 from "Finite Field Arithmetic Using AVX-512 For Isogeny-Based Cryptography" by Orisaka, Aranha and López
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
    
    // The result can be 384 bits.
    // If so, we do a final subtraction by p to bring it down to 383 bits.
    hi = VSL(a[12], 16);
    __mmask8 msb_mask = VSIGN(hi);
    __mmask8 mask_odd  = msb_mask & 0xaa;
    __mmask8 mask_even = mask_odd >> 1;
    a[12] = VSUBMASK(a[12], mask_odd, a[12], top);
    a[9]  = VADDMASK(a[9], mask_even, a[9], one);
    
    // Output result: final carry propagation is left for the final result.
    //                Bottom limb can have up to 49 bits (because of the +1).
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

// Assumes unvec is an array of 4 * NUM_LIMBS_64 64-bit ints
void fp_64_to_limbs_4x2(__m512i* res, uint64_t* unvec)
{
    uint64_t t[8];
    
    for (int i = 0; i < 4; i++) {
        t[2*i] = unvec[NUM_LIMBS_64 * i] & LIMB_MASK;
        t[2*i+1] = unvec[NUM_LIMBS_64 * i + 3] & LIMB_MASK;
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = ((unvec[NUM_LIMBS_64 * i] >> 48) & 0xffffULL) | ((unvec[NUM_LIMBS_64 * i + 1] & 0xffffffffULL) << 16);
        t[2*i+1] = ((unvec[NUM_LIMBS_64 * i + 3] >> 48) & 0xffffULL) | ((unvec[NUM_LIMBS_64 * i + 4] & 0xffffffffULL) << 16);
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = ((unvec[NUM_LIMBS_64 * i + 1] >> 32) & 0xffffffffULL) | ((unvec[NUM_LIMBS_64 * i + 2] & 0xffffULL) << 32);
        t[2*i+1] = ((unvec[NUM_LIMBS_64 * i + 4] >> 32) & 0xffffffffULL) | ((unvec[NUM_LIMBS_64 * i + 5] & 0xffffULL) << 32);
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = (unvec[NUM_LIMBS_64 * i + 2] >> 16) & LIMB_MASK;
        t[2*i+1] = (unvec[NUM_LIMBS_64 * i + 5] >> 16) & LIMB_MASK;
    }
    res[3] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

void fp_limbs_to_64_4x2(uint64_t* res, __m512i* src)
{
    uint64_t t[8];
    
    VSTORE(t, src[0]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i] = t[2*i] & LIMB_MASK;
        
        res[NUM_LIMBS_64 * i + 3] = t[2*i+1] & LIMB_MASK;
    }
    VSTORE(t, src[1]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i] |= (t[2*i] & 0xffffULL) << 48;
        res[NUM_LIMBS_64 * i + 1] = (t[2*i] >> 16) & 0xffffffffULL;
        
        res[NUM_LIMBS_64 * i + 3] |= (t[2*i+1] & 0xffffULL) << 48;
        res[NUM_LIMBS_64 * i + 4] = (t[2*i+1] >> 16) & 0xffffffffULL;
    }
    VSTORE(t, src[2]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i + 1] |= (t[2*i] & 0xffffffffULL) << 32;
        res[NUM_LIMBS_64 * i + 2] = (t[2*i] >> 32) & 0xffffULL;
        
        res[NUM_LIMBS_64 * i + 4] |= (t[2*i+1] & 0xffffffffULL) << 32;
        res[NUM_LIMBS_64 * i + 5] = (t[2*i+1] >> 32) & 0xffffULL;
    }
    VSTORE(t, src[3]);
    for (int i = 0; i < 4; i++) {
        res[NUM_LIMBS_64 * i + 2] |= (t[2*i] & LIMB_MASK) << 16;
        
        res[NUM_LIMBS_64 * i + 5] |= (t[2*i+1] & LIMB_MASK) << 16;
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
    __m512i zero[2], norm_a[2], norm_b[2];
    fp_setzero_2x4(zero);
    
    // Normalize both to [0, p-1] by adding zero
    fp_add_2x4(norm_a, a, zero);
    fp_add_2x4(norm_b, b, zero);
    
    __mmask8 acc = 0xff;
    acc &= VCMPEQ(norm_a[0], norm_b[0]);
    acc &= VCMPEQ(norm_a[1], norm_b[1]);
    
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
    acc &= VCMPEQ(norm_a[0], zero[0]);
    acc &= VCMPEQ(norm_a[1], zero[1]);
    
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

