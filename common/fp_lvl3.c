#include <stdio.h>
#include "fp_lvl3.h"
#include "utils.h"

// ======================================================================================================
//                                                 8x1
// ======================================================================================================

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
    for(int i = 0; i < 8; i++) {
        res[i] = VSETALL(a[i]);
    }
}

// Assumes that elements are fully reduced mod p, so there is only one representative.
int fp_is_equal_8x1(__m512i* a, __m512i* b)
{
    __mmask8 mask, acc;
    acc = 0xFF;
    for(int i = 0; i < 8; i++){
        mask = VCMPEQ(a[i], b[i]);
        acc = acc & mask;
    }
    return acc;
}

// Assumes that elements are fully reduced mod p, so there is only one representative.
int fp_is_zero_8x1(__m512i* a)
{
    __mmask8 mask, acc;
    __m512i zero = VZERO();
    acc = 0xFF;
    for(int i = 0; i < 8; i++){
        mask = VCMPEQ(a[i], zero);
        acc = acc & mask;
    }
    return acc;
}

// Following AVX512 SIKE paper
// Currently the arithmetic is mod p, 
// I think they do mod 2p to not have to do the final step of the Montgomery reduction
void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSETALL(1);
    __m512i p_top = VSETALL(p_plus_1_top);
    
    // Add 48-bit limbs
    t0 = VADD(a[0], b[0]);
    t1 = VADD(a[1], b[1]);
    t2 = VADD(a[2], b[2]);
    t3 = VADD(a[3], b[3]);
    t4 = VADD(a[4], b[4]);
    t5 = VADD(a[5], b[5]);
    t6 = VADD(a[6], b[6]);
    t7 = VADD(a[7], b[7]);

    // Subtract p, using special shape of p
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

    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t7, 63);
    
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    t7 = VADD(t7, VAND(p_top, sign_mask));

    // For some reason this was slower:
    // __mmask8 sign_mask = VSIGN(t7);
    // t0 = VSUBMASK(t0, sign_mask, t0, p_bot);
    // t7 = VADDMASK(t7, sign_mask, t7, p_top);

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

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
}

void fp_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSETALL(1);
    __m512i p_top = VSETALL(p_plus_1_top);

    // a - b
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

    // If result is negative, we add p
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

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
}


void fp_neg_8x1(__m512i* res, __m512i* a)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSETALL(1);
    __m512i p_top = VSETALL(p_plus_1_top);
    __m512i zero = VZERO();

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

    // If result is negative, we add p
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

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
}

// TODO: Possible to only do final reduction at the end of Fp2 mul, not here?
// Used in Fp2 4x2x1
void fp_mixaddsub_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)                      ////////////// NOT REVIEWED YET
{
    __m512i t[8], sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSETALL(1);
    __m512i p_top = VSETALL(p_plus_1_top);
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

    // a + b - p
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

    // Output result
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
    res[4] = t[4];
    res[5] = t[5];
    res[6] = t[6];
    res[7] = t[7];
}

void fp_mul_8x1_school(__m512i* res, __m512i* a, __m512i* b)                                                 ////////////// NOT REVIEWED YET
{
    //Likely not needed to use t15, but for now we keep things simple:
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15; 
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15;
    __m512i s7, hi, cnst, quo, rem;
    __m512i zero = VZERO();
    __m512i limb_mask = VSETALL(LIMB_MASK);

    // Part 1: Schoolbook multiplication
    // We have 51 bit inputs. For each ti, we can potentially add one term for each a[i]
    // 8=2^3, so we get that the ti will grow to 54 bits, well within what we can fit.
    // In conclusion, we don't need to handle intermediate carries.

    // i + j = 0
    t0 = VMULADDL(zero, a[0], b[0]);
    t1 = VMULADDH(zero, a[0], b[0]);
    t1 = VSL(t1, 1);

    // i + j = 1
    t1 = VMULADDL(t1, a[0], b[1]);
    t2 = VMULADDH(zero, a[0], b[1]);
    t1 = VMULADDL(t1, a[1], b[0]);
    t2 = VMULADDH(t2, a[1], b[0]);
    t2 = VSL(t2, 1);

    // 2
    t2 = VMULADDL(t2, a[0], b[2]);
    t3 = VMULADDH(zero, a[0], b[2]);
    t2 = VMULADDL(t2, a[1], b[1]);
    t3 = VMULADDH(t3, a[1], b[1]);
    t2 = VMULADDL(t2, a[2], b[0]);
    t3 = VMULADDH(t3, a[2], b[0]);
    t3 = VSL(t3, 1);

    // 3
    t3 = VMULADDL(t3, a[0], b[3]);
    t4 = VMULADDH(zero, a[0], b[3]);
    t3 = VMULADDL(t3, a[1], b[2]);
    t4 = VMULADDH(t4, a[1], b[2]);
    t3 = VMULADDL(t3, a[2], b[1]);
    t4 = VMULADDH(t4, a[2], b[1]);
    t3 = VMULADDL(t3, a[3], b[0]);
    t4 = VMULADDH(t4, a[3], b[0]);
    t4 = VSL(t4, 1);

    // 4
    t4 = VMULADDL(t4, a[0], b[4]);
    t5 = VMULADDH(zero, a[0], b[4]);
    t4 = VMULADDL(t4, a[1], b[3]);
    t5 = VMULADDH(t5, a[1], b[3]);
    t4 = VMULADDL(t4, a[2], b[2]);
    t5 = VMULADDH(t5, a[2], b[2]);
    t4 = VMULADDL(t4, a[3], b[1]);
    t5 = VMULADDH(t5, a[3], b[1]);
    t4 = VMULADDL(t4, a[4], b[0]);
    t5 = VMULADDH(t5, a[4], b[0]);
    t5 = VSL(t5, 1);

    // 5
    t5 = VMULADDL(t5, a[0], b[5]);
    t6 = VMULADDH(zero, a[0], b[5]);
    t5 = VMULADDL(t5, a[1], b[4]);
    t6 = VMULADDH(t6, a[1], b[4]);
    t5 = VMULADDL(t5, a[2], b[3]);
    t6 = VMULADDH(t6, a[2], b[3]);
    t5 = VMULADDL(t5, a[3], b[2]);
    t6 = VMULADDH(t6, a[3], b[2]);
    t5 = VMULADDL(t5, a[4], b[1]);
    t6 = VMULADDH(t6, a[4], b[1]);
    t5 = VMULADDL(t5, a[5], b[0]);
    t6 = VMULADDH(t6, a[5], b[0]);
    t6 = VSL(t6, 1);

    // 6
    t6 = VMULADDL(t6, a[0], b[6]);
    t7 = VMULADDH(zero, a[0], b[6]);
    t6 = VMULADDL(t6, a[1], b[5]);
    t7 = VMULADDH(t7, a[1], b[5]);
    t6 = VMULADDL(t6, a[2], b[4]);
    t7 = VMULADDH(t7, a[2], b[4]);
    t6 = VMULADDL(t6, a[3], b[3]);
    t7 = VMULADDH(t7, a[3], b[3]);
    t6 = VMULADDL(t6, a[4], b[2]);
    t7 = VMULADDH(t7, a[4], b[2]);
    t6 = VMULADDL(t6, a[5], b[1]);
    t7 = VMULADDH(t7, a[5], b[1]);
    t6 = VMULADDL(t6, a[6], b[0]);
    t7 = VMULADDH(t7, a[6], b[0]);
    t7 = VSL(t7, 1);

    // 7
    t7 = VMULADDL(t7, a[0], b[7]);
    t8 = VMULADDH(zero, a[0], b[7]);
    t7 = VMULADDL(t7, a[1], b[6]);
    t8 = VMULADDH(t8, a[1], b[6]);
    t7 = VMULADDL(t7, a[2], b[5]);
    t8 = VMULADDH(t8, a[2], b[5]);
    t7 = VMULADDL(t7, a[3], b[4]);
    t8 = VMULADDH(t8, a[3], b[4]);
    t7 = VMULADDL(t7, a[4], b[3]);
    t8 = VMULADDH(t8, a[4], b[3]);
    t7 = VMULADDL(t7, a[5], b[2]);
    t8 = VMULADDH(t8, a[5], b[2]);
    t7 = VMULADDL(t7, a[6], b[1]);
    t8 = VMULADDH(t8, a[6], b[1]);
    t7 = VMULADDL(t7, a[7], b[0]);
    t8 = VMULADDH(t8, a[7], b[0]);
    t8 = VSL(t8, 1);

    // 8
    t8 = VMULADDL(t8, a[1], b[7]);
    t9 = VMULADDH(zero, a[1], b[7]);
    t8 = VMULADDL(t8, a[2], b[6]);
    t9 = VMULADDH(t9, a[2], b[6]);
    t8 = VMULADDL(t8, a[3], b[5]);
    t9 = VMULADDH(t9, a[3], b[5]);
    t8 = VMULADDL(t8, a[4], b[4]);
    t9 = VMULADDH(t9, a[4], b[4]);
    t8 = VMULADDL(t8, a[5], b[3]);
    t9 = VMULADDH(t9, a[5], b[3]);
    t8 = VMULADDL(t8, a[6], b[2]);
    t9 = VMULADDH(t9, a[6], b[2]);
    t8 = VMULADDL(t8, a[7], b[1]);
    t9 = VMULADDH(t9, a[7], b[1]);
    t9 = VSL(t9, 1);

    // 9
    t9 = VMULADDL(t9, a[2], b[7]);
    t10 = VMULADDH(zero, a[2], b[7]);
    t9 = VMULADDL(t9, a[3], b[6]);
    t10 = VMULADDH(t10, a[3], b[6]);
    t9 = VMULADDL(t9, a[4], b[5]);
    t10 = VMULADDH(t10, a[4], b[5]);
    t9 = VMULADDL(t9, a[5], b[4]);
    t10 = VMULADDH(t10, a[5], b[4]);
    t9 = VMULADDL(t9, a[6], b[3]);
    t10 = VMULADDH(t10, a[6], b[3]);
    t9 = VMULADDL(t9, a[7], b[2]);
    t10 = VMULADDH(t10, a[7], b[2]);
    t10 = VSL(t10, 1);

    // 10
    t10 = VMULADDL(t10, a[3], b[7]);
    t11 = VMULADDH(zero, a[3], b[7]);
    t10 = VMULADDL(t10, a[4], b[6]);
    t11 = VMULADDH(t11, a[4], b[6]);
    t10 = VMULADDL(t10, a[5], b[5]);
    t11 = VMULADDH(t11, a[5], b[5]);
    t10 = VMULADDL(t10, a[6], b[4]);
    t11 = VMULADDH(t11, a[6], b[4]);
    t10 = VMULADDL(t10, a[7], b[3]);
    t11 = VMULADDH(t11, a[7], b[3]);
    t11 = VSL(t11, 1);

    // 11
    t11 = VMULADDL(t11, a[4], b[7]);
    t12 = VMULADDH(zero, a[4], b[7]);
    t11 = VMULADDL(t11, a[5], b[6]);
    t12 = VMULADDH(t12, a[5], b[6]);
    t11 = VMULADDL(t11, a[6], b[5]);
    t12 = VMULADDH(t12, a[6], b[5]);
    t11 = VMULADDL(t11, a[7], b[4]);
    t12 = VMULADDH(t12, a[7], b[4]);
    t12 = VSL(t12, 1);

    // 12
    t12 = VMULADDL(t12, a[5], b[7]);
    t13 = VMULADDH(zero, a[5], b[7]);
    t12 = VMULADDL(t12, a[6], b[6]);
    t13 = VMULADDH(t13, a[6], b[6]);
    t12 = VMULADDL(t12, a[7], b[5]);
    t13 = VMULADDH(t13, a[7], b[5]);
    t13 = VSL(t13, 1);

    // 13
    t13 = VMULADDL(t13, a[6], b[7]);
    t14 = VMULADDH(zero, a[6], b[7]);
    t13 = VMULADDL(t13, a[7], b[6]);
    t14 = VMULADDH(t14, a[7], b[6]);
    t14 = VSL(t14, 1);

    // 14
    t14 = VMULADDL(t14, a[7], b[7]); // Will be 52 bits
    t15 = VZERO();

    // Carry propagation TODO: Can we delay further?
    // Result is 766 = 15 * 51 + 1 bits, so t15 just holds one bit
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

    // Part 2: Montgomery reduction
    // low(t) = (t0, t1, ..., t7) is 8*51 = 408 bits
    // We will do Montgomery reduction with R = 2^408.
    // m = -p^(-1) mod 2^408 = 65 * 2^376 + 1
    // The output of this reduction is (t + (low(t) * m mod 2^408)*p) / 2^408
    
    // We begin by computing m * low mod 2^408. Note that 376 = 7 * 51 + 19.
    // Then m * low = (t0 * 65 * 2^376) + low mod 2^408.
    // Thus, we only need to update the top limb.
    // s0 = t0, s1 = t1, ..., s6 = t6
    cnst = VSETALL((uint64_t) 65 << 19);
    s7 = VMULADDL(t7, t0, cnst);
    s7 = VAND(s7, limb_mask);

    // r = s*p
    // We exploit the special shape of p, computing r = s * (p+1) - s = s * (65 * 2^376) - s
    // s is 408 bits, (p+1) is 7 + 376 = 383 bits, so r will be 791 = 15 * 51 + 26 bits
    r7 = VMULADDL(zero, t0, cnst);
    hi = VMULADDH(zero, t0, cnst);
    hi = VSL(hi, 1);
    r8 = VMULADDL(hi, t1, cnst);
    hi = VMULADDH(zero, t1, cnst);
    hi = VSL(hi, 1);
    r9 = VMULADDL(hi, t2, cnst);
    hi = VMULADDH(zero, t2, cnst);
    hi = VSL(hi, 1);
    r10 = VMULADDL(hi, t3, cnst);
    hi = VMULADDH(zero, t3, cnst);
    hi = VSL(hi, 1);
    r11 = VMULADDL(hi, t4, cnst);
    hi = VMULADDH(zero, t4, cnst);
    hi = VSL(hi, 1);
    r12 = VMULADDL(hi, t5, cnst);
    hi = VMULADDH(zero, t5, cnst);
    hi = VSL(hi, 1);
    r13 = VMULADDL(hi, t6, cnst);
    hi = VMULADDH(zero, t6, cnst);
    hi = VSL(hi, 1);
    r14 = VMULADDL(hi, s7, cnst);
    r15 = VMULADDH(zero, s7, cnst);
    r15 = VSL(r15, 1);

    r0 = VSUB(zero, t0);
    r1 = VSUB(zero, t1);
    r2 = VSUB(zero, t2);
    r3 = VSUB(zero, t3);
    r4 = VSUB(zero, t4);
    r5 = VSUB(zero, t5);
    r6 = VSUB(zero, t6);
    r7 = VSUB(r7, s7);

    // t = t + r
    // Before the addition, we have t < 2^766 and r < 2^791, so t+r < 2^792
    // 792 = 15 * 51 + 27
    t0 = VADD(t0, r0);
    t1 = VADD(t1, r1);
    t2 = VADD(t2, r2);
    t3 = VADD(t3, r3);
    t4 = VADD(t4, r4);
    t5 = VADD(t5, r5);
    t6 = VADD(t6, r6);
    t7 = VADD(t7, r7);
    t8 = VADD(t8, r8);
    t9 = VADD(t9, r9);
    t10 = VADD(t10, r10);
    t11 = VADD(t11, r11);
    t12 = VADD(t12, r12);
    t13 = VADD(t13, r13);
    t14 = VADD(t14, r14);
    t15 = VADD(t15, r15);

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

    // We divide by R = 2^408 by dropping t0-t7.
    // The result is in t8 - t15, and it is 7 * 51 + 27 = 384 bits
    // We do a final reduction step, to get the result down to 383 bits

    // Split into an 8-bit top part and 376-bit bottom part
    hi = VSRA(t15, 19); 
    cnst = VSETALL(0x7ffff);
    t15 = VAND(t15, cnst);

    // 65 * 2^376 = 1 mod p, hence hi mod p = floor(h/65) + (h mod 65)*2^376
    cnst = VSETALL(0xfc1);
    quo = VMULADDL(zero, hi, cnst);
    quo = VSRA(quo, 18);
    cnst = VSETALL(65); // again suboptimal, not doing -65
    rem = VMULADDL(zero, quo, cnst);
    rem = VSUB(hi, rem);

    // Adding hi mod p to 376-bit bottom part
    // quo < floor(2^8 / 65) <= 3 and rem <= 64
    // Hence quo + rem * 2^376 <= 3 + 2^382.
    // Adding this to 376-bit bottom part, we get a result that is at most 383 bits.
    t8 = VADD(t8, quo);
    t15 = VADD(t15, VSL(rem, 19));

    // Carry propagation
    t9 = VADD(t9, VSRA(t8, LIMB_LEN));
    t10 = VADD(t10, VSRA(t9, LIMB_LEN));
    t11 = VADD(t11, VSRA(t10, LIMB_LEN));
    t12 = VADD(t12, VSRA(t11, LIMB_LEN));
    t13 = VADD(t13, VSRA(t12, LIMB_LEN));
    t14 = VADD(t14, VSRA(t13, LIMB_LEN));
    t15 = VADD(t15, VSRA(t14, LIMB_LEN));
    t8 = VAND(t8, limb_mask);
    t9 = VAND(t9, limb_mask);
    t10 = VAND(t10, limb_mask);
    t11 = VAND(t11, limb_mask);
    t12 = VAND(t12, limb_mask);
    t13 = VAND(t13, limb_mask);
    t14 = VAND(t14, limb_mask);

    // Output result
    res[0] = t8;
    res[1] = t9;
    res[2] = t10;
    res[3] = t11;
    res[4] = t12;
    res[5] = t13;
    res[6] = t14;
    res[7] = t15;
}

void fp_sqr_8x1(__m512i* res, __m512i* a)
{
    fp_mul_8x1(res, a, a);
}

// TODO: Optimize
void fp_div3_8x1(__m512i* res, __m512i* src)
{
    fp_t tmp[8];
    fp_8x1_to_unvectorized(tmp, src);
    for (int i = 0; i < 8; i++) {
        fp_div3(tmp+i, tmp+i);
    }
    fp_unvectorized_to_8x1(res, tmp);
}

void fp_tomont_8x1(__m512i* res, __m512i* a)
{
    __m512i vR2[8];
    fp_duplicate_8x1(vR2, R2);
    fp_mul_8x1(res, a, vR2);
}

void fp_frommont_8x1(__m512i* res, __m512i* a)
{
    __m512i one[8];
    fp_setzero_8x1(one);
    one[0] = VSETALL(1);
    fp_mul_8x1(res, a, one);
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

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
}

// Single-length multiprecision subtraction, no correction, no carry propagation
void mp_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;

    t0 = VSUB(a[0], b[0]);
    t1 = VSUB(a[1], b[1]);
    t2 = VSUB(a[2], b[2]);
    t3 = VSUB(a[3], b[3]);
    t4 = VSUB(a[4], b[4]);
    t5 = VSUB(a[5], b[5]);
    t6 = VSUB(a[6], b[6]);
    t7 = VSUB(a[7], b[7]);

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
}

// Double-length multiprecision subtraction res = res - a - b, no correction
void mp_dblsubx2_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    __m512i limb_mask = VSETALL(LIMB_MASK);

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

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
    res[8] = t8;
    res[9] = t9;
    res[10] = t10;
    res[11] = t11;
    res[12] = t12;
    res[13] = t13;
    res[14] = t14;
    res[15] = t15;
}

// Double-length multiprecision subtraction res = res - a
void mp_subx2_8x1(__m512i* res, __m512i* a)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    __m512i limb_mask = VSETALL(LIMB_MASK);

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

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
    res[8] = t8;
    res[9] = t9;
    res[10] = t10;
    res[11] = t11;
    res[12] = t12;
    res[13] = t13;
    res[14] = t14;
    res[15] = t15;
}

// Double-length multiprecision
void mp_mixaddsubx2_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    __m512i limb_mask = VSETALL(LIMB_MASK);
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

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
    res[8] = t8;
    res[9] = t9;
    res[10] = t10;
    res[11] = t11;
    res[12] = t12;
    res[13] = t13;
    res[14] = t14;
    res[15] = t15;
}

// Computes res = a + p - b without any carry propagation or conditional masks
void mp_add_p_sub_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i p_bod = VSETALL(LIMB_MASK);
    __m512i p_top  = VSETALL((65ULL << 40) - 1);
    __m512i t[8];

    t[0] = VSUB(VADD(a[0], p_bod), b[0]);
    t[1] = VSUB(VADD(a[1], p_bod), b[1]);
    t[2] = VSUB(VADD(a[2], p_bod), b[2]);
    t[3] = VSUB(VADD(a[3], p_bod), b[3]);
    t[4] = VSUB(VADD(a[4], p_bod), b[4]);
    t[5] = VSUB(VADD(a[5], p_bod), b[5]);
    t[6] = VSUB(VADD(a[6], p_bod), b[6]);
    t[7] = VSUB(VADD(a[7], p_top), b[7]);

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
    res[0] = VSL(a[0], 1);
    res[1] = VSL(a[1], 1);
    res[2] = VSL(a[2], 1);
    res[3] = VSL(a[3], 1);
    res[4] = VSL(a[4], 1);
    res[5] = VSL(a[5], 1);
    res[6] = VSL(a[6], 1);
    res[7] = VSL(a[7], 1);
}

// Multiprecision multiplication using Karatsuba
void mp_mul_8x1_karat(__m512i* res, __m512i* a, __m512i* b)
{
    const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3];
    const __m512i a4 = a[4], a5 = a[5], a6 = a[6], a7 = a[7];
    const __m512i b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    const __m512i b4 = b[4], b5 = b[5], b6 = b[6], b7 = b[7];
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i ta0, ta1, ta2, ta3;
    __m512i tb0, tb1, tb2, tb3;
    __m512i t0 = VZERO(), t1 = VZERO(), t2 = VZERO(), t3 = VZERO();
    __m512i t4 = VZERO(), t5 = VZERO(), t6 = VZERO(), t7 = VZERO();
    __m512i t8 = VZERO(), t9 = VZERO(), t10 = VZERO(), t11 = VZERO();
    __m512i t12 = VZERO(), t13 = VZERO(), t14 = VZERO(), t15 = VZERO();
    __m512i s0 = VZERO(), s1 = VZERO(), s2 = VZERO(), s3 = VZERO();
    __m512i s4 = VZERO(), s5 = VZERO(), s6 = VZERO(), s7 = VZERO();
    __m512i s8 = VZERO(), s9 = VZERO(), s10 = VZERO(), s11 = VZERO();
    __m512i s12 = VZERO(), s13 = VZERO(), s14 = VZERO(), s15 = VZERO();
    __m512i m0 = VZERO(), m1 = VZERO(), m2 = VZERO(), m3 = VZERO();
    __m512i m4 = VZERO(), m5 = VZERO(), m6 = VZERO(), m7 = VZERO();

    // tL(t0-t7) <- aL(a0-a3) * bL(b0-b4)

    t0 = VMULADDL(t0, a0, b0);
    s0 = VMULADDH(s0, a0, b0);
    s0 = VSL(s0, LIMB_ALIGN);

    t1 = VMULADDL(s0, a0, b1);
    t1 = VMULADDL(t1, a1, b0);
    s1 = VMULADDH(s1, a0, b1);
    s1 = VMULADDH(s1, a1, b0);
    s1 = VSL(s1, LIMB_ALIGN);

    t2 = VMULADDL(s1, a0, b2);
    t2 = VMULADDL(t2, a1, b1);
    t2 = VMULADDL(t2, a2, b0);
    s2 = VMULADDH(s2, a0, b2);
    s2 = VMULADDH(s2, a1, b1);
    s2 = VMULADDH(s2, a2, b0);
    s2 = VSL(s2, LIMB_ALIGN);

    t3 = VMULADDL(s2, a0, b3);
    t3 = VMULADDL(t3, a1, b2);
    t3 = VMULADDL(t3, a2, b1);
    t3 = VMULADDL(t3, a3, b0);
    s3 = VMULADDH(s3, a0, b3);
    s3 = VMULADDH(s3, a1, b2);
    s3 = VMULADDH(s3, a2, b1);
    s3 = VMULADDH(s3, a3, b0);
    s3 = VSL(s3, LIMB_ALIGN);

    t4 = VMULADDL(s3, a1, b3);
    t4 = VMULADDL(t4, a2, b2);
    t4 = VMULADDL(t4, a3, b1);
    s4 = VMULADDH(s4, a1, b3);
    s4 = VMULADDH(s4, a2, b2);
    s4 = VMULADDH(s4, a3, b1);
    s4 = VSL(s4, LIMB_ALIGN);

    t5 = VMULADDL(s4, a2, b3);
    t5 = VMULADDL(t5, a3, b2);
    s5 = VMULADDH(s5, a2, b3);
    s5 = VMULADDH(s5, a3, b2);
    s5 = VSL(s5, LIMB_ALIGN);

    t6 = VMULADDL(s5, a3, b3);
    s6 = VMULADDH(s6, a3, b3);
    s6 = VSL(s6, LIMB_ALIGN);
    t7 = s6;

    // tH(t8-t15) <- aH(a4-a7) * bH(b4-b7)

    t8 = VMULADDL(t8, a4, b4);
    s8 = VMULADDH(s8, a4, b4);
    s8 = VSL(s8, LIMB_ALIGN);

    t9 = VMULADDL(s8, a4, b5);
    t9 = VMULADDL(t9, a5, b4);
    s9 = VMULADDH(s9, a4, b5);
    s9 = VMULADDH(s9, a5, b4);
    s9 = VSL(s9, LIMB_ALIGN);

    t10 = VMULADDL(s9, a4, b6);
    t10 = VMULADDL(t10, a5, b5);
    t10 = VMULADDL(t10, a6, b4);
    s10 = VMULADDH(s10, a4, b6);
    s10 = VMULADDH(s10, a5, b5);
    s10 = VMULADDH(s10, a6, b4);
    s10 = VSL(s10, LIMB_ALIGN);

    t11 = VMULADDL(s10, a4, b7);
    t11 = VMULADDL(t11, a5, b6);
    t11 = VMULADDL(t11, a6, b5);
    t11 = VMULADDL(t11, a7, b4);
    s11 = VMULADDH(s11, a4, b7);
    s11 = VMULADDH(s11, a5, b6);
    s11 = VMULADDH(s11, a6, b5);
    s11 = VMULADDH(s11, a7, b4);
    s11 = VSL(s11, LIMB_ALIGN);

    t12 = VMULADDL(s11, a5, b7);
    t12 = VMULADDL(t12, a6, b6);
    t12 = VMULADDL(t12, a7, b5);
    s12 = VMULADDH(s12, a5, b7);
    s12 = VMULADDH(s12, a6, b6);
    s12 = VMULADDH(s12, a7, b5);
    s12 = VSL(s12, LIMB_ALIGN);

    t13 = VMULADDL(s12, a6, b7);
    t13 = VMULADDL(t13, a7, b6);
    s13 = VMULADDH(s13, a6, b7);
    s13 = VMULADDH(s13, a7, b6);
    s13 = VSL(s13, LIMB_ALIGN);

    t14 = VMULADDL(s13, a7, b7);
    s14 = VMULADDH(s14, a7, b7);
    s14 = VSL(s14, LIMB_ALIGN);
    t15 = s14;

    // ta(ta0-ta3) <- aL(a0-a3) + aH(a4-a7)
    ta0 = VADD(a0, a4);
    ta1 = VADD(a1, a5);
    ta2 = VADD(a2, a6);
    ta3 = VADD(a3, a7);

    // tb(tb0-tb3) <- bL(b0-b3) + bH(b4-b7)
    tb0 = VADD(b0, b4);
    tb1 = VADD(b1, b5);
    tb2 = VADD(b2, b6);
    tb3 = VADD(b3, b7);

    // tM(m0-m7) <- ta(ta0-ta3) * tb(tb0-tb3)
    s0 = s1 = s2 = s3 = s4 = s5 = s6 = s7 = VZERO();

    m0 = VMULADDL(m0, ta0, tb0);
    s0 = VMULADDH(s0, ta0, tb0);
    s0 = VSL(s0, LIMB_ALIGN);

    m1 = VMULADDL(s0, ta0, tb1);
    m1 = VMULADDL(m1, ta1, tb0);
    s1 = VMULADDH(s1, ta0, tb1);
    s1 = VMULADDH(s1, ta1, tb0);
    s1 = VSL(s1, LIMB_ALIGN);

    m2 = VMULADDL(s1, ta0, tb2);
    m2 = VMULADDL(m2, ta1, tb1);
    m2 = VMULADDL(m2, ta2, tb0);
    s2 = VMULADDH(s2, ta0, tb2);
    s2 = VMULADDH(s2, ta1, tb1);
    s2 = VMULADDH(s2, ta2, tb0);
    s2 = VSL(s2, LIMB_ALIGN);

    m3 = VMULADDL(s2, ta0, tb3);
    m3 = VMULADDL(m3, ta1, tb2);
    m3 = VMULADDL(m3, ta2, tb1);
    m3 = VMULADDL(m3, ta3, tb0);
    s3 = VMULADDH(s3, ta0, tb3);
    s3 = VMULADDH(s3, ta1, tb2);
    s3 = VMULADDH(s3, ta2, tb1);
    s3 = VMULADDH(s3, ta3, tb0);
    s3 = VSL(s3, LIMB_ALIGN);

    m4 = VMULADDL(s3, ta1, tb3);
    m4 = VMULADDL(m4, ta2, tb2);
    m4 = VMULADDL(m4, ta3, tb1);
    s4 = VMULADDH(s4, ta1, tb3);
    s4 = VMULADDH(s4, ta2, tb2);
    s4 = VMULADDH(s4, ta3, tb1);
    s4 = VSL(s4, LIMB_ALIGN);

    m5 = VMULADDL(s4, ta2, tb3);
    m5 = VMULADDL(m5, ta3, tb2);
    s5 = VMULADDH(s5, ta2, tb3);
    s5 = VMULADDH(s5, ta3, tb2);
    s5 = VSL(s5, LIMB_ALIGN);

    m6 = VMULADDL(s5, ta3, tb3);
    s6 = VMULADDH(s6, ta3, tb3);
    s6 = VSL(s6, LIMB_ALIGN);
    m7 = s6;

    // tM(m0-m7) <- (ta * tb) - tL - tH
    m0 = VSUB(m0, VADD(t0, t8));
    m1 = VSUB(m1, VADD(t1, t9));
    m2 = VSUB(m2, VADD(t2, t10));
    m3 = VSUB(m3, VADD(t3, t11));
    m4 = VSUB(m4, VADD(t4, t12));
    m5 = VSUB(m5, VADD(t5, t13));
    m6 = VSUB(m6, VADD(t6, t14));
    m7 = VSUB(m7, VADD(t7, t15));

    // t = t + tM
    t4 = VADD(t4, m0);
    t5 = VADD(t5, m1);
    t6 = VADD(t6, m2);
    t7 = VADD(t7, m3);
    t8 = VADD(t8, m4);
    t9 = VADD(t9, m5);
    t10 = VADD(t10, m6);
    t11 = VADD(t11, m7);

    // Output result
    // Carry propagation is delayed to mp_dblsubx2_8x1 and mp_subx2_8x1 during fp2mul 

    res[0] = t0; res[1] = t1; res[2] = t2; res[3] = t3; res[4] = t4;
    res[5] = t5; res[6] = t6; res[7] = t7; res[8] = t8; res[9] = t9;
    res[10] = t10; res[11] = t11;
    res[12] = t12; res[13] = t13;
    res[14] = t14; res[15] = t15;
}

// Montgomery reduction
void rdc_8x1(__m512i* res, __m512i* a)
{
    const __m512i a0 = a[0], a1 = a[1], a2 = a[2], a3 = a[3], a4 = a[4];
    const __m512i a5 = a[5], a6 = a[6], a7 = a[7], a8 = a[8], a9 = a[9];
    const __m512i a10 = a[10], a11 = a[11], a12 = a[12], a13 = a[13], a14 = a[14], a15 = a[15];
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16;
    __m512i s7, s8, hi, cnst;
    __m512i zero = VZERO();
    __m512i one = VSETALL(1);
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __mmask8 msb_mask;

    // Montgomery reduction with R = 2^432
    // low(a) = (a0, a1, ... , a7, a8) is 9*48 = 432 bits
    // m = -p^(-1) mod 2^432 = 65 * 2^376 + 1
    // The output of this reduction is (a + (low(a) * m mod 2^432)*p) / 2^432 = \lfloor (a + (low(a) * m mod 2^432) * 65 * 2^376) / 2^432 \rfloor

    // We begin by computing m * low mod 2^432. Note that 376 = 7 * 48 + 40.
    // Then m * low = (a0:a1 * 65 * 2^376) + low mod 2^432.
    // Thus, we only need to update the top two limbs.
    // s0 = a0, s1 = a1, ... , s6 = a6
    cnst = VSETALL((uint64_t)65 << 40);
    s7 = VMULADDL(a7, a0, cnst);
    s8 = VMULADDH(zero, a0, cnst);
    s8 = VADD(a8, VSL(s8, LIMB_ALIGN));
    s8 = VMULADDL(s8, a1, cnst);
    s8 = VADD(s8, VSRA(s7, LIMB_LEN));
    s7 = VAND(s7, limb_mask);
    s8 = VAND(s8, limb_mask);

    // r <- s * (p+1)
    r7 = VMULADDL(zero, a0, cnst);
    hi = VMULADDH(zero, a0, cnst);
    hi = VSL(hi, LIMB_ALIGN);
    r8 = VMULADDL(hi, a1, cnst);
    hi = VMULADDH(zero, a1, cnst);
    hi = VSL(hi, LIMB_ALIGN);
    r9 = VMULADDL(hi, a2, cnst);
    hi = VMULADDH(zero, a2, cnst);
    hi = VSL(hi, LIMB_ALIGN);
    r10 = VMULADDL(hi, a3, cnst);
    hi = VMULADDH(zero, a3, cnst);
    hi = VSL(hi, LIMB_ALIGN);
    r11 = VMULADDL(hi, a4, cnst);
    hi = VMULADDH(zero, a4, cnst);
    hi = VSL(hi, LIMB_ALIGN);
    r12 = VMULADDL(hi, a5, cnst);
    hi = VMULADDH(zero, a5, cnst);
    hi = VSL(hi, LIMB_ALIGN);
    r13 = VMULADDL(hi, a6, cnst);
    hi = VMULADDH(zero, a6, cnst);
    hi = VSL(hi, LIMB_ALIGN);
    r14 = VMULADDL(hi, s7, cnst);
    hi = VMULADDH(zero, s7, cnst);
    hi = VSL(hi, LIMB_ALIGN);
    r15 = VMULADDL(hi, s8, cnst);
    r16 = VMULADDH(zero, s8, cnst);
    r16 = VSL(r16, LIMB_ALIGN);

    // r <- a + r
    r7 = VADD(a7, r7);
    r8 = VADD(a8, r8);
    r9 = VADD(a9, r9);
    r10 = VADD(a10, r10);
    r11 = VADD(a11, r11);
    r12 = VADD(a12, r12);
    r13 = VADD(a13, r13);
    r14 = VADD(a14, r14);
    r15 = VADD(a15, r15);

    // Carry propagation
    r8 = VADD(r8, VSRA(r7, LIMB_LEN));
    r9 = VADD(r9, VSRA(r8, LIMB_LEN));
    r10 = VADD(r10, VSRA(r9, LIMB_LEN));
    r11 = VADD(r11, VSRA(r10, LIMB_LEN));
    r12 = VADD(r12, VSRA(r11, LIMB_LEN));
    r13 = VADD(r13, VSRA(r12, LIMB_LEN));
    r14 = VADD(r14, VSRA(r13, LIMB_LEN));
    r15 = VADD(r15, VSRA(r14, LIMB_LEN));
    r16 = VADD(r16, VSRA(r15, LIMB_LEN));
    r9 = VAND(r9, limb_mask);
    r10 = VAND(r10, limb_mask);
    r11 = VAND(r11, limb_mask);
    r12 = VAND(r12, limb_mask);
    r13 = VAND(r13, limb_mask);
    r14 = VAND(r14, limb_mask);
    r15 = VAND(r15, limb_mask);

    // The result is in r9:r16 after dividing by R = 2^432
    // Final correction to get the result down to 383 bits:
    // If bit383 = 1 then mask is all 1's, else it's all 0's
    // If bit383 = 1 then subtract by (p + 1) - 1 => r - (65*2^376) + 1 
    hi = VSL(r16, 16);                      // 64 - (384 - (7*48)) = 16
    msb_mask = VSIGN(hi);
    VSUBMASK(r16, msb_mask, r16, cnst);
    VADDMASK(r9, msb_mask, r9, one);

    // Output result: final carry propagation is left for the final result. 
    //                Bottom limb can have up to 49 bits.
    res[0] = r9;
    res[1] = r10;
    res[2] = r11;
    res[3] = r12;
    res[4] = r13;
    res[5] = r14;
    res[6] = r15;
    res[7] = r16;
}

void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    const __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i t[16];
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;

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
    t15 = VADD(t[15], VSRA(t14, LIMB_LEN));;
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
    t[15] = t15;

    rdc_8x1(res, t);
}

// Montgomery reduction with R = 2^(9*48), loop-based.
// Doesn't require carry propagation before being called.
void rdc_loop_8x1(__m512i* res, __m512i* a)
{
    __m512i zero = VZERO();
    __m512i one = VSETALL(1);
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i cnst = VSETALL((uint64_t)65 << 40);
    __m512i carry, hi;
    __mmask8 msb_mask;

    // p' = -p^{-1} mod 2^48 = 1, so u_i = a[i] mod 2^48.
    // A += u_i * p * 2^(48*i), where p = 65*2^376 - 1.
    // The -u_i subtraction from limb i is skipped since limbs 0-8 are discarded.
    for (int i = 0; i < 9; i++) {
        carry = VSRA(a[i], LIMB_LEN);
        a[i] = VAND(a[i], limb_mask);  // u_i
        a[i+1] = VADD(a[i+1], carry);

        // Add u_i * (65 << 40) to limb i+7, overflow to limb i+8
        a[i+7] = VMULADDL(a[i+7], a[i], cnst);
        hi = VMULADDH(zero, a[i], cnst);
        a[i+8] = VADD(a[i+8], VSL(hi, LIMB_ALIGN));
    }

    // Carry propagation for result in a[9]..a[15], a16
    a[10] = VADD(a[10], VSRA(a[9], LIMB_LEN));
    a[11] = VADD(a[11], VSRA(a[10], LIMB_LEN));
    a[12] = VADD(a[12], VSRA(a[11], LIMB_LEN));
    a[13] = VADD(a[13], VSRA(a[12], LIMB_LEN));
    a[14] = VADD(a[14], VSRA(a[13], LIMB_LEN));
    a[15] = VADD(a[15], VSRA(a[14], LIMB_LEN));
    a[16] = VADD(a[16], VSRA(a[15], LIMB_LEN));
    a[9] = VAND(a[9], limb_mask);
    a[10] = VAND(a[10], limb_mask);
    a[11] = VAND(a[11], limb_mask);
    a[12] = VAND(a[12], limb_mask);
    a[13] = VAND(a[13], limb_mask);
    a[14] = VAND(a[14], limb_mask);
    a[15] = VAND(a[15], limb_mask);

    // Final correction: 384 -> 383 bits
    // If bit 383 is set, subtract (p+1) and add 1
    hi = VSL(a[16], 16);  // bit 47 of limb 7 -> bit 63
    msb_mask = VSIGN(hi);
    a[16] = VSUBMASK(a[16], msb_mask, a[16], cnst);
    a[9] = VADDMASK(a[9], msb_mask, a[9], one);

    res[0] = a[9];
    res[1] = a[10];
    res[2] = a[11];
    res[3] = a[12];
    res[4] = a[13];
    res[5] = a[14];
    res[6] = a[15];
    res[7] = a[16];
}

// Simplified fp_mul — no carry propagation between multiplication and reduction
// but it is 2 cycles slower
void fp_mul_loop_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[17];
    mp_mul_8x1_karat(t, a, b);
    t[16] = VZERO();
    rdc_loop_8x1(res, t);
}

void fp_8x1_to_unvectorized(fp_t* res, __m512i* src)
{
    fp_t coeffs[8];
    __m512i src_frommont[8];

    fp_frommont_8x1(src_frommont, src);
    fp_48_to_64_bit_limbs_8x1(coeffs, src_frommont);

    // Convert to Montgomery representation
    for (int i = 0; i < 8; i++) {
        fp_decode(res+i, (const void *) coeffs[i].arr);
    }
}

void fp_48_to_64_bit_limbs_8x1(fp_t* res, __m512i* src)
{
    uint64_t t[8];
    VSTORE(t, src[0]);
    for (int i = 0; i < 8; i++) {
        (res+i)->arr[0] = t[i] & LIMB_MASK;
    } 
    VSTORE(t, src[1]);
    for (int i = 0; i < 8; i++) {
        (res+i)->arr[0] |= (t[i] & 0xffff) << 48;
        (res+i)->arr[1] = (t[i] >> 16) & 0xffffffffffff;
    } 
    VSTORE(t, src[2]);
    for (int i = 0; i < 8; i++) {
        (res+i)->arr[1] |= (t[i] & 0xffffffff) << 32;
        (res+i)->arr[2] = (t[i] >> 32) & 0xffffffff;
    }
    VSTORE(t, src[3]);
    for (int i = 0; i < 8; i++) {
        (res+i)->arr[2] |= (t[i] & 0xffffffffffff) << 16;
        (res+i)->arr[3] = (t[i] >> 48) & 0xffff;
    }
    VSTORE(t, src[4]);
    for (int i = 0; i < 8; i++) {
        (res+i)->arr[3] |= t[i] & 0xffffffffffff;
    }
    VSTORE(t, src[5]);
    for (int i = 0; i < 8; i++) {
        (res+i)->arr[3] |= (t[i] & 0xffff) << 48;
        (res+i)->arr[4] = (t[i] >> 16) & 0xffffffffffff;
    }
    VSTORE(t, src[6]);
    for (int i = 0; i < 8; i++) {
        (res+i)->arr[4] |= (t[i] & 0xffffffff) << 32;
        (res+i)->arr[5] = (t[i] >> 32) & 0xffffffff;
    }
    VSTORE(t, src[7]);
    for (int i = 0; i < 8; i++) {
        (res+i)->arr[5] |= t[i] << 16;
    }
}

void fp_48_to_64_bit_limbs_8x1_only_even_idx(fp_t* res, __m512i* src)
{
    uint64_t t[8];
    VSTORE(t, src[0]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[0] = t[2*i] & LIMB_MASK;
    } 
    VSTORE(t, src[1]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[0] |= (t[2*i] & 0xffff) << 48;
        (res+i)->arr[1] = (t[2*i] >> 16) & 0xffffffffffff;
    } 
    VSTORE(t, src[2]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[1] |= (t[2*i] & 0xffffffff) << 32;
        (res+i)->arr[2] = (t[2*i] >> 32) & 0xffffffff;
    }
    VSTORE(t, src[3]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[2] |= (t[2*i] & 0xffffffffffff) << 16;
        (res+i)->arr[3] = (t[2*i] >> 48) & 0xffff;
    }
    VSTORE(t, src[4]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[3] |= t[2*i] & 0xffffffffffff;
    }
    VSTORE(t, src[5]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[3] |= (t[2*i] & 0xffff) << 48;
        (res+i)->arr[4] = (t[2*i] >> 16) & 0xffffffffffff;
    }
    VSTORE(t, src[6]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[4] |= (t[2*i] & 0xffffffff) << 32;
        (res+i)->arr[5] = (t[2*i] >> 32) & 0xffffffff;
    }
    VSTORE(t, src[7]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[5] |= t[2*i] << 16;
    }
}

void fp_unvectorized_to_8x1(__m512i* res, fp_t* src)
{
    fp_t coeffs[8];

    // Convert from Montgomery representation
    for (int i = 0; i < 8; i++) {
        fp_encode(coeffs[i].arr, src+i);
    }

    fp_64_to_48_bit_limbs_8x1(res, coeffs);
    fp_tomont_8x1(res, res);
}

void fp_64_to_48_bit_limbs_8x1(__m512i* res, fp_t* src)
{
    uint64_t t[8];

    for (int i = 0; i < 8; i++) {
        t[i] = (src+i)->arr[0] & LIMB_MASK;
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (((src+i)->arr[0] >> 48) & 0xffff) | (((src+i)->arr[1] & 0xffffffff) << 16); 
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (((src+i)->arr[1] >> 32) & 0xffffffff) | (((src+i)->arr[2] & 0xffff) << 32); 
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (((src+i)->arr[2] >> 16) & 0xffffffffffff);
    }
    res[3] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((src+i)->arr[3]) & LIMB_MASK;
    }
    res[4] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (((src+i)->arr[3] >> 48) & 0xffff) | (((src+i)->arr[4] & 0xffffffff) << 16); 
    }
    res[5] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = (((src+i)->arr[4] >> 32) & 0xffffffff) | (((src+i)->arr[5] & 0xffff) << 32); 
    }
    res[6] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 8; i++) {
        t[i] = ((src+i)->arr[5] >> 16) & LIMB_MASK;
    }
    res[7] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}

void fp_64_to_48_bit_limbs_8x1_only_even_idx(__m512i* res, fp_t* src)
{
    uint64_t t[4];

    for (int i = 0; i < 4; i++) {
        t[i] = (src+i)->arr[0] & LIMB_MASK;
    }
    res[0] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (((src+i)->arr[0] >> 48) & 0xffff) | (((src+i)->arr[1] & 0xffffffff) << 16); 
    }
    res[1] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (((src+i)->arr[1] >> 32) & 0xffffffff) | (((src+i)->arr[2] & 0xffff) << 32); 
    }
    res[2] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (((src+i)->arr[2] >> 16) & 0xffffffffffff);
    }
    res[3] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((src+i)->arr[3]) & LIMB_MASK;
    }
    res[4] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (((src+i)->arr[3] >> 48) & 0xffff) | (((src+i)->arr[4] & 0xffffffff) << 16); 
    }
    res[5] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = (((src+i)->arr[4] >> 32) & 0xffffffff) | (((src+i)->arr[5] & 0xffff) << 32); 
    }
    res[6] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
    for (int i = 0; i < 4; i++) {
        t[i] = ((src+i)->arr[5] >> 16) & LIMB_MASK;
    }
    res[7] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);
}

// ======================================================================================================
//                                                 4x2
// ======================================================================================================

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
    for(int i = 0; i < 4; i++) {
        res[i] = VSET(a[i], a[i+4], a[i], a[i+4], a[i], a[i+4], a[i], a[i+4]);
    }
}

// Assumes that elements are fully reduced mod p, so there is only one representative.
int fp_is_equal_4x2(__m512i* a, __m512i* b)
{
    __mmask8 mask, acc, res;
    acc = 0xff;
    for(int i = 0; i < 4; i++){
        mask = VCMPEQ(a[i], b[i]);
        acc = acc & mask;
    }
    // a = b iff both halfs are equal
    res = (acc & (acc >> 1) & 0x1) 
        | ((acc >> 1) & (acc >> 2) & 0x2) 
        | ((acc >> 2) & (acc >> 3) & 0x4) 
        | ((acc >> 3) & (acc >> 4) & 0x8);
    return res;
}

// Assumes that elements are fully reduced mod p, so there is only one representative.
int fp_is_zero_4x2(__m512i* a)
{
    __m512i zero = VZERO();
    __mmask8 mask, acc, res;
    acc = 0xff;
    for(int i = 0; i < 4; i++){
        mask = VCMPEQ(a[i], zero);
        acc = acc & mask;
    }
    // a = b iff both halfs are equal
    res = (acc & (acc >> 1) & 0x1) 
        | ((acc >> 1) & (acc >> 2) & 0x2) 
        | ((acc >> 2) & (acc >> 3) & 0x4) 
        | ((acc >> 3) & (acc >> 4) & 0x8);
    return res;
}

void fp_add_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, carry, sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    __m512i p_top = VSET(0, p_plus_1_top, 0, p_plus_1_top, 0, p_plus_1_top, 0, p_plus_1_top);
    
    // Add 51-bit limbs
    t0 = VADD(a[0], b[0]);
    t1 = VADD(a[1], b[1]);
    t2 = VADD(a[2], b[2]);
    t3 = VADD(a[3], b[3]);

    // Subtract p, using special shape of p
    t0 = VADD(t0, p_bot);
    t3 = VSUB(t3, p_top);

    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t1 = VADDMASK(t1, 0x55, t1, VSRA(t0, LIMB_LEN));
    t2 = VADDMASK(t2, 0x55, t2, VSRA(t1, LIMB_LEN));
    t3 = VADDMASK(t3, 0x55, t3, VSRA(t2, LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
    carry = VSRA(t3, LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xaa, t0, carry); 
    t1 = VADDMASK(t1, 0xaa, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xaa, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xaa, t3, VSRA(t2, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VANDMASK(t3, 0x55, t3, limb_mask);

    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t3, 63); // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    t3 = VADD(t3, VAND(p_top, sign_mask));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    
    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t1 = VADDMASK(t1, 0x55, t1, VSRA(t0, LIMB_LEN));
    t2 = VADDMASK(t2, 0x55, t2, VSRA(t1, LIMB_LEN));
    t3 = VADDMASK(t3, 0x55, t3, VSRA(t2, LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
    carry = VSRA(t3, LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xaa, t0, carry); 
    t1 = VADDMASK(t1, 0xaa, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xaa, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xaa, t3, VSRA(t2, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VANDMASK(t3, 0x55, t3, limb_mask);

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
}

void fp_sub_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, carry, sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    __m512i p_top = VSET(0, p_plus_1_top, 0, p_plus_1_top, 0, p_plus_1_top, 0, p_plus_1_top);

    // Subtract 51-bit limbs
    t0 = VSUB(a[0], b[0]);
    t1 = VSUB(a[1], b[1]);
    t2 = VSUB(a[2], b[2]);
    t3 = VSUB(a[3], b[3]);

    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t1 = VADDMASK(t1, 0x55, t1, VSRA(t0, LIMB_LEN));
    t2 = VADDMASK(t2, 0x55, t2, VSRA(t1, LIMB_LEN));
    t3 = VADDMASK(t3, 0x55, t3, VSRA(t2, LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
    carry = VSRA(t3, LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xaa, t0, carry); 
    t1 = VADDMASK(t1, 0xaa, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xaa, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xaa, t3, VSRA(t2, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VANDMASK(t3, 0x55, t3, limb_mask);

    // If result is negative, we add p
    sign_mask = VSRA(t3, 63); // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    t3 = VADD(t3, VAND(p_top, sign_mask));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    
    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t1 = VADDMASK(t1, 0x55, t1, VSRA(t0, LIMB_LEN));
    t2 = VADDMASK(t2, 0x55, t2, VSRA(t1, LIMB_LEN));
    t3 = VADDMASK(t3, 0x55, t3, VSRA(t2, LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
    carry = VSRA(t3, LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xaa, t0, carry); 
    t1 = VADDMASK(t1, 0xaa, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xaa, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xaa, t3, VSRA(t2, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VANDMASK(t3, 0x55, t3, limb_mask);

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
}

void fp_neg_4x2(__m512i* res, __m512i* a)
{
    __m512i t0, t1, t2, t3, carry, sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    __m512i p_top = VSET(0, p_plus_1_top, 0, p_plus_1_top, 0, p_plus_1_top, 0, p_plus_1_top);
    __m512i zero = VZERO();

    // Subtract 51-bit limbs
    t0 = VSUB(zero, a[0]);
    t1 = VSUB(zero, a[1]);
    t2 = VSUB(zero, a[2]);
    t3 = VSUB(zero, a[3]);

    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t1 = VADDMASK(t1, 0x55, t1, VSRA(t0, LIMB_LEN));
    t2 = VADDMASK(t2, 0x55, t2, VSRA(t1, LIMB_LEN));
    t3 = VADDMASK(t3, 0x55, t3, VSRA(t2, LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
    carry = VSRA(t3, LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xaa, t0, carry); 
    t1 = VADDMASK(t1, 0xaa, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xaa, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xaa, t3, VSRA(t2, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VANDMASK(t3, 0x55, t3, limb_mask);

    // If result is negative, we add p
    sign_mask = VSRA(t3, 63); // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    t3 = VADD(t3, VAND(p_top, sign_mask));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    
    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t1 = VADDMASK(t1, 0x55, t1, VSRA(t0, LIMB_LEN));
    t2 = VADDMASK(t2, 0x55, t2, VSRA(t1, LIMB_LEN));
    t3 = VADDMASK(t3, 0x55, t3, VSRA(t2, LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
    carry = VSRA(t3, LIMB_LEN);
    carry = VSHUFFLEMASKZ(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xaa, t0, carry); 
    t1 = VADDMASK(t1, 0xaa, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xaa, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xaa, t3, VSRA(t2, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VANDMASK(t3, 0x55, t3, limb_mask);

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
}

//was sub_add with submask 0x33 and addmask 0xcc
void fp_mixaddsub_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t[8], sign_mask, carry;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);
    __m512i p_top = VSET(0, p_plus_1_top, 0, p_plus_1_top, 0, p_plus_1_top, 0, p_plus_1_top);
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
    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
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

    // If result is negative, we add p
    sign_mask = VSRA(t[3], 63); // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    t[3] = VADD(t[3], VAND(p_top, sign_mask));
    sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);
    t[0] = VSUB(t[0], VAND(p_bot, sign_mask));

    // Carry propagation
    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t[1] = VADDMASK(t[1], 0x55, t[1], VSRA(t[0], LIMB_LEN));
    t[2] = VADDMASK(t[2], 0x55, t[2], VSRA(t[1], LIMB_LEN));
    t[3] = VADDMASK(t[3], 0x55, t[3], VSRA(t[2], LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
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

    // Output result
    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

// Single-length multiprecision addition, no correction, no carry propagation
void mp_add_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3;

    t0 = VADD(a[0], b[0]);
    t1 = VADD(a[1], b[1]);
    t2 = VADD(a[2], b[2]);
    t3 = VADD(a[3], b[3]);

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
}

// Single-length multiprecision subtraction, no correction, no carry propagation
void mp_add_p_sub_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i p_bod = VSETALL(LIMB_MASK);
    uint64_t top_limb = (65ULL << 40) - 1;
    __m512i p_top  = VSET(LIMB_MASK, top_limb, LIMB_MASK, top_limb, LIMB_MASK, top_limb, LIMB_MASK, top_limb);
    __m512i t[4];

    t[0] = VSUB(VADD(a[0], p_bod), b[0]);
    t[1] = VSUB(VADD(a[1], p_bod), b[1]);
    t[2] = VSUB(VADD(a[2], p_bod), b[2]);
    t[3] = VSUB(VADD(a[3], p_top), b[3]);

    res[0] = t[0];
    res[1] = t[1];
    res[2] = t[2];
    res[3] = t[3];
}

// Double-length multiprecision
void mp_mixaddsubx2_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
    __m512i limb_mask = VSETALL(LIMB_MASK);
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

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
    res[8] = t8;
    res[9] = t9;
    res[10] = t10;
    res[11] = t11;
}

// Double-length multiprecision subtraction res = res - a - b, no correction
void mp_dblsubx2_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
    __m512i limb_mask = VSETALL(LIMB_MASK);

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

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
    res[8] = t8;
    res[9] = t9;
    res[10] = t10;
    res[11] = t11;
}

// Double-length multiprecision subtraction res = res - a
void mp_subx2_4x2(__m512i* res, __m512i* a)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
    __m512i limb_mask = VSETALL(LIMB_MASK);

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

    // Output result
    res[0] = t0;
    res[1] = t1;
    res[2] = t2;
    res[3] = t3;
    res[4] = t4;
    res[5] = t5;
    res[6] = t6;
    res[7] = t7;
    res[8] = t8;
    res[9] = t9;
    res[10] = t10;
    res[11] = t11;
}

void mp_mul_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t[12];
    __m512i sl0, sl1, sl2, sl3, sh0, sh1, sh2, sh3;
    __m512i zero = VZERO();

    // Algorithm 2 from "Finite Field Arithmetic Using AVX-512 For Isogeny-Based Cryptography" by Orisaka, Aranha and López

    // Duplicating limbs of b with shuffles:
    // If b[0] = {a_0, a_4, b_0, b_4, ...} then sl0 = {a_0, a_0, b_0, b_0, ...}
    sl0 = VSHUFFLE(b[0], 0x44);
    sl1 = VSHUFFLE(b[1], 0x44);
    sl2 = VSHUFFLE(b[2], 0x44);
    sl3 = VSHUFFLE(b[3], 0x44);
    //// If b[0] = {a_0, a_4, b_0, b_4, ...} then sh0 = {a_4, a_4, b_4, b_4, ...}
    sh0 = VSHUFFLE(b[0], 0xEE);
    sh1 = VSHUFFLE(b[1], 0xEE);
    sh2 = VSHUFFLE(b[2], 0xEE);
    sh3 = VSHUFFLE(b[3], 0xEE);

    // left i+j = 0 right i+j = 4
    t[0] = VMULADDL(zero, a[0], sl0);
    t[1] = VMULADDH(zero, a[0], sl0);
    t[1] = VSL(t[1], LIMB_ALIGN);

    // left i+j = 1 
    t[1] = VMULADDL(t[1], a[0], sl1);
    t[2] = VMULADDH(zero, a[0], sl1);
    t[1] = VMULADDL(t[1], a[1], sl0);
    t[2] = VMULADDH(t[2], a[1], sl0);
    t[2] = VSL(t[2], LIMB_ALIGN);

    // left i+j = 2
    t[2] = VMULADDL(t[2], a[0], sl2);
    t[3] = VMULADDH(zero, a[0], sl2);
    t[2] = VMULADDL(t[2], a[1], sl1);
    t[3] = VMULADDH(t[3], a[1], sl1);
    t[2] = VMULADDL(t[2], a[2], sl0);
    t[3] = VMULADDH(t[3], a[2], sl0);
    t[3] = VSL(t[3], LIMB_ALIGN);

    // left i+j = 3
    t[3] = VMULADDL(t[3], a[0], sl3);
    t[4] = VMULADDH(zero, a[0], sl3);
    t[3] = VMULADDL(t[3], a[1], sl2);
    t[4] = VMULADDH(t[4], a[1], sl2);
    t[3] = VMULADDL(t[3], a[2], sl1);
    t[4] = VMULADDH(t[4], a[2], sl1);
    t[3] = VMULADDL(t[3], a[3], sl0);
    t[4] = VMULADDH(t[4], a[3], sl0);
    t[4] = VSL(t[4], LIMB_ALIGN);

    // left i+j = 4
    t[4] = VMULADDL(t[4], a[0], sh0);
    t[5] = VMULADDH(zero, a[0], sh0);
    t[4] = VMULADDL(t[4], a[1], sl3);
    t[5] = VMULADDH(t[5], a[1], sl3);
    t[4] = VMULADDL(t[4], a[2], sl2);
    t[5] = VMULADDH(t[5], a[2], sl2);
    t[4] = VMULADDL(t[4], a[3], sl1);
    t[5] = VMULADDH(t[5], a[3], sl1);
    t[5] = VSL(t[5], LIMB_ALIGN);

    // left i+j = 5
    t[5] = VMULADDL(t[5], a[0], sh1);
    t[6] = VMULADDH(zero, a[0], sh1);
    t[5] = VMULADDL(t[5], a[1], sh0);
    t[6] = VMULADDH(t[6], a[1], sh0);
    t[5] = VMULADDL(t[5], a[2], sl3);
    t[6] = VMULADDH(t[6], a[2], sl3);
    t[5] = VMULADDL(t[5], a[3], sl2);
    t[6] = VMULADDH(t[6], a[3], sl2);
    t[6] = VSL(t[6], LIMB_ALIGN);

    // left i+j = 6
    t[6] = VMULADDL(t[6], a[0], sh2);
    t[7] = VMULADDH(zero, a[0], sh2);
    t[6] = VMULADDL(t[6], a[1], sh1);
    t[7] = VMULADDH(t[7], a[1], sh1);
    t[6] = VMULADDL(t[6], a[2], sh0);
    t[7] = VMULADDH(t[7], a[2], sh0);
    t[6] = VMULADDL(t[6], a[3], sl3);
    t[7] = VMULADDH(t[7], a[3], sl3);
    t[7] = VSL(t[7], LIMB_ALIGN);

    // left i+j = 7
    t[7] = VMULADDL(t[7], a[0], sh3);
    t[8] = VMULADDH(zero, a[0], sh3);
    t[7] = VMULADDL(t[7], a[1], sh2);
    t[8] = VMULADDH(t[8], a[1], sh2);
    t[7] = VMULADDL(t[7], a[2], sh1);
    t[8] = VMULADDH(t[8], a[2], sh1);
    t[7] = VMULADDL(t[7], a[3], sh0);
    t[8] = VMULADDH(t[8], a[3], sh0);
    t[8] = VSL(t[8], LIMB_ALIGN);

    // left i+j = 8
    t[8] = VMULADDL(t[8], a[1], sh3);
    t[9] = VMULADDH(zero, a[1], sh3);
    t[8] = VMULADDL(t[8], a[2], sh2);
    t[9] = VMULADDH(t[9], a[2], sh2);
    t[8] = VMULADDL(t[8], a[3], sh1);
    t[9] = VMULADDH(t[9], a[3], sh1);
    t[9] = VSL(t[9], LIMB_ALIGN);
    
    // left i+j = 9
    t[9] = VMULADDL(t[9], a[2], sh3);
    t[10] = VMULADDH(zero, a[2], sh3);
    t[9] = VMULADDL(t[9], a[3], sh2);
    t[10] = VMULADDH(t[10], a[3], sh2);
    t[10] = VSL(t[10], LIMB_ALIGN);

    // left i+j = 10
    t[10] = VMULADDL(t[10], a[3], sh3);
    t[11] = VMULADDH(zero, a[3], sh3);
    t[11] = VSL(t[11], LIMB_ALIGN);

    // Output
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

    // For Montgomery reduction with R = 2^(9*48) mod p
    res[12] = VZERO();
}

// Modifies a
void rdc_4x2(__m512i* res, __m512i* a)
{
    __m512i carry, hi, shuf, quo, rem;
    __m512i zero = VZERO();
    __m512i cnst = VSETALL((uint64_t) 65 << 40); // top limb of (p+1)
    __m512i limb_mask = VSETALL(LIMB_MASK);

    for(int i = 0; i < 9; i++) {
        // using the fact that -p^{-1} mod 2^48 = 1 + 65* 2^376 mod 2^48 = 1
        carry = VSRA(a[i], LIMB_LEN);
        a[i] = VAND(a[i], limb_mask); // Left slot is u_i = t_i mod 2^48
        a[i+1] = VADD(a[i+1], carry);

        // t = t + u_i * p * 2^(48*i) 
        // p = (p+1) - 1 = 2^(48*7) * (65 << 40) -1
        shuf = VSHUFFLEMASKZ(0xcccc, a[i], 0x4e); // shuf = {0, u_i, 0, ...}
        a[i+3] = VMULADDL(a[i+3], shuf, cnst);
        hi = VMULADDH(zero, shuf, cnst);
        hi = VSL(hi, LIMB_ALIGN);
        a[i+4] = VADD(a[i+4], hi);
        //a[i] = VSHUFFLEMASKZ(0xcc, a[i], 0xe4); // Would need to do t - 2^(48*i)*u_i, but we will drop vector when dividing, so can skip
        a[i+4] = VADDMASK(a[i+4], 0x55, a[i+4], VSHUFFLE(a[i], 0x4e));
    }

    // Result is now in a[9] to a[12]
    // Propogating carry
    a[10] = VADDMASK(a[10], 0x55, a[10], VSRA(a[9], LIMB_LEN));
    a[11] = VADDMASK(a[11], 0x55, a[11], VSRA(a[10], LIMB_LEN));
    a[12] = VADDMASK(a[12], 0x55, a[12], VSRA(a[11], LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
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

    // The result is in a[9] - a[12], and it is 8 * 48 = 384 bits
    // We do a final reduction step, to get the result down to 383 bits    
    // Bit 383 is bit 47 of limb 7 (odd lanes of t[12])
    hi = VSL(a[12], 16);                        
    __mmask8 msb_mask = VSIGN(hi); 
    __mmask8 mask_odd  = msb_mask & 0xaa;
    __mmask8 mask_even = mask_odd >> 1; 

    // If bit383 = 1, then res = res - p = res - (p+1) + 1
    __m512i one = VSETALL(1);
    a[12] = VSUBMASK(a[12], mask_odd, a[12], cnst);
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

void fp_48_to_64_bit_limbs_4x2(fp_t* res, __m512i* src)
{
    uint64_t t[8];
    VSTORE(t, src[0]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[0] = t[2*i] & LIMB_MASK;
        (res+i)->arr[3] = t[2*i+1] & 0xffffffffffff;
    } 
    VSTORE(t, src[1]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[0] |= (t[2*i] & 0xffff) << 48;
        (res+i)->arr[1] = (t[2*i] >> 16) & 0xffffffffffff;

        (res+i)->arr[3] |= (t[2*i+1] & 0xffff) << 48;
        (res+i)->arr[4] = (t[2*i+1] >> 16) & 0xffffffffffff;
    } 
    VSTORE(t, src[2]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[1] |= (t[2*i] & 0xffffffff) << 32;
        (res+i)->arr[2] = (t[2*i] >> 32) & 0xffffffff;

        (res+i)->arr[4] |= (t[2*i+1] & 0xffffffff) << 32;
        (res+i)->arr[5] = (t[2*i+1] >> 32) & 0xffffffff;
    }
    VSTORE(t, src[3]);
    for (int i = 0; i < 4; i++) {
        (res+i)->arr[2] |= (t[2*i] & 0xffffffffffff) << 16;
        (res+i)->arr[3] |= (t[2*i] >> 48) & 0xffff;

        (res+i)->arr[5] |= t[2*i+1] << 16;
    }
}

void fp_64_to_48_bit_limbs_4x2(__m512i* res, fp_t* src)
{
    uint64_t t[8];

    for (int i = 0; i < 4; i++) {
        t[2*i] = (src+i)->arr[0] & LIMB_MASK;
        t[2*i+1] = ((src+i)->arr[3]) & LIMB_MASK;
    }
    res[0] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = (((src+i)->arr[0] >> 48) & 0xffff) | (((src+i)->arr[1] & 0xffffffff) << 16); 
        t[2*i+1] = (((src+i)->arr[3] >> 48) & 0xffff) | (((src+i)->arr[4] & 0xffffffff) << 16); 
    }
    res[1] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = (((src+i)->arr[1] >> 32) & 0xffffffff) | (((src+i)->arr[2] & 0xffff) << 32); 
        t[2*i+1] = (((src+i)->arr[4] >> 32) & 0xffffffff) | (((src+i)->arr[5] & 0xffff) << 32); 
    }
    res[2] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
    for (int i = 0; i < 4; i++) {
        t[2*i] = (((src+i)->arr[2] >> 16) & 0xffffffffffff);
        t[2*i+1] = ((src+i)->arr[5] >> 16) & LIMB_MASK;
    }
    res[3] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);
}



void fp_4x2_to_unvectorized(fp_t* res, __m512i* src)
{
    fp_t coeffs[4];
    __m512i src_frommont[4];

    fp_frommont_4x2(src_frommont, src);
    fp_48_to_64_bit_limbs_4x2(coeffs, src_frommont);

    // Convert to Montgomery representation
    for (int i = 0; i < 4; i++) {
        fp_decode(res+i, (const void *) coeffs[i].arr);
    }
}

void fp_unvectorized_to_4x2(__m512i* res, fp_t* src)
{
    fp_t coeffs[4];

    // Convert from Montgomery representation
    for (int i = 0; i < 4; i++) {
        fp_encode(coeffs[i].arr, src+i);
    }

    fp_64_to_48_bit_limbs_4x2(res, coeffs);
    fp_tomont_4x2(res, res);
}
