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

// Following AVX512 SIKE paper
// Currently the arithmetic is mod p, 
// I think they do mod 2p to not have to do the final step of the Montgomery reduction
void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, sign_mask;
    __m512i limb_mask = VSETALL(LIMB_MASK);
    __m512i p_bot = VSETALL(1);
    __m512i p_top = VSETALL(p_plus_1_top);
    
    // Add 51-bit limbs
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

    // Propogate carries
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

    // Propogate the carry
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

    // Propogate carries
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

    // Propogate the carry
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

    // Propogate the carry
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

    // Propogate the carry
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

void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b)
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

    // Propogate the carry TODO: Can we delay further?
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

    // Propogate the carry
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

    // Propogate carry
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

void fp_tomont_8x1(__m512i* res, __m512i* a)
{
    __m512i vR2[8];
    fp_duplicate_8x1(vR2, R2);
    fp_mul_8x1(res, a, vR2);
}

void fp_frommont_8x1(__m512i*res, __m512i* a)
{
    __m512i one[8];
    fp_setzero_8x1(one);
    one[0] = VSETALL(1);
    fp_mul_8x1(res, a, one);
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
    carry = VSHUFFLEMASK(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xAA, t0, carry); 
    t1 = VADDMASK(t1, 0xAA, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xAA, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xAA, t3, VSRA(t2, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VANDMASK(t3, 0x55, t3, limb_mask);

    // If result is negative, we undo the subtraction by p
    sign_mask = VSRA(t3, 63); // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    t3 = VADD(t3, VAND(p_top, sign_mask));
    sign_mask = VSHUFFLEMASK(0xffff, sign_mask, 0x4e);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    
    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t1 = VADDMASK(t1, 0x55, t1, VSRA(t0, LIMB_LEN));
    t2 = VADDMASK(t2, 0x55, t2, VSRA(t1, LIMB_LEN));
    t3 = VADDMASK(t3, 0x55, t3, VSRA(t2, LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
    carry = VSRA(t3, LIMB_LEN);
    carry = VSHUFFLEMASK(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xAA, t0, carry); 
    t1 = VADDMASK(t1, 0xAA, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xAA, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xAA, t3, VSRA(t2, LIMB_LEN));
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
    carry = VSHUFFLEMASK(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xAA, t0, carry); 
    t1 = VADDMASK(t1, 0xAA, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xAA, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xAA, t3, VSRA(t2, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VANDMASK(t3, 0x55, t3, limb_mask);

    // If result is negative, we add p
    sign_mask = VSRA(t3, 63); // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    t3 = VADD(t3, VAND(p_top, sign_mask));
    sign_mask = VSHUFFLEMASK(0xffff, sign_mask, 0x4e);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    
    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t1 = VADDMASK(t1, 0x55, t1, VSRA(t0, LIMB_LEN));
    t2 = VADDMASK(t2, 0x55, t2, VSRA(t1, LIMB_LEN));
    t3 = VADDMASK(t3, 0x55, t3, VSRA(t2, LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
    carry = VSRA(t3, LIMB_LEN);
    carry = VSHUFFLEMASK(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xAA, t0, carry); 
    t1 = VADDMASK(t1, 0xAA, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xAA, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xAA, t3, VSRA(t2, LIMB_LEN));
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
    carry = VSHUFFLEMASK(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xAA, t0, carry); 
    t1 = VADDMASK(t1, 0xAA, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xAA, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xAA, t3, VSRA(t2, LIMB_LEN));
    t0 = VAND(t0, limb_mask);
    t1 = VAND(t1, limb_mask);
    t2 = VAND(t2, limb_mask);
    t3 = VANDMASK(t3, 0x55, t3, limb_mask);

    // If result is negative, we add p
    sign_mask = VSRA(t3, 63); // sign_mask = {0, sign_mask1, 0, sign_mask2, ...}
    t3 = VADD(t3, VAND(p_top, sign_mask));
    sign_mask = VSHUFFLEMASK(0xffff, sign_mask, 0x4e);
    t0 = VSUB(t0, VAND(p_bot, sign_mask));
    
    // TODO: Carry propagation could be suboptimal, recomputing shifts
    t1 = VADDMASK(t1, 0x55, t1, VSRA(t0, LIMB_LEN));
    t2 = VADDMASK(t2, 0x55, t2, VSRA(t1, LIMB_LEN));
    t3 = VADDMASK(t3, 0x55, t3, VSRA(t2, LIMB_LEN));
    //Need to move carry to second word. Shifting is faster than permuting (1 cycle vs 3)
    carry = VSRA(t3, LIMB_LEN);
    carry = VSHUFFLEMASK(0xcccc, carry, 0x4e);
    t0 = VADDMASK(t0, 0xAA, t0, carry); 
    t1 = VADDMASK(t1, 0xAA, t1, VSRA(t0, LIMB_LEN)); 
    t2 = VADDMASK(t2, 0xAA, t2, VSRA(t1, LIMB_LEN)); 
    t3 = VADDMASK(t3, 0xAA, t3, VSRA(t2, LIMB_LEN));
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

void fp_mul_4x2(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11;
    __m512i sl0, sl1, sl2, sl3, sh0, sh1, sh2, sh3;
    __m512i zero = VZERO();
    __m512i limb_mask = VSETALL(LIMB_MASK);


    // Part 1: Integer multiplication
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
    t0 = VMULADDL(zero, a[0], sl0);
    t1 = VMULADDH(zero, a[0], sl0);
    t1 = VSL(t1, 1);

    // left i+j = 1 
    t1 = VMULADDL(t1, a[0], sl1);
    t2 = VMULADDH(zero, a[0], sl1);
    t1 = VMULADDL(t1, a[1], sl0);
    t2 = VMULADDH(t2, a[1], sl0);
    t2 = VSL(t2, 1);

    // left i+j = 2
    t2 = VMULADDL(t2, a[0], sl2);
    t3 = VMULADDH(zero, a[0], sl2);
    t2 = VMULADDL(t2, a[1], sl1);
    t3 = VMULADDH(t3, a[1], sl1);
    t2 = VMULADDL(t2, a[2], sl0);
    t3 = VMULADDH(t3, a[2], sl0);
    t3 = VSL(t3, 1);

    // left i+j = 3
    t3 = VMULADDL(t3, a[0], sl3);
    t4 = VMULADDH(zero, a[0], sl3);
    t3 = VMULADDL(t3, a[1], sl2);
    t4 = VMULADDH(t4, a[1], sl2);
    t3 = VMULADDL(t3, a[2], sl1);
    t4 = VMULADDH(t4, a[2], sl1);
    t3 = VMULADDL(t3, a[3], sl0);
    t4 = VMULADDH(t4, a[3], sl0);
    t4 = VSL(t4, 1);

    // left i+j = 4
    t4 = VMULADDL(t4, a[0], sh0);
    t5 = VMULADDH(zero, a[0], sh0);
    t4 = VMULADDL(t4, a[1], sl3);
    t5 = VMULADDH(t5, a[1], sl3);
    t4 = VMULADDL(t4, a[2], sl2);
    t5 = VMULADDH(t5, a[2], sl2);
    t4 = VMULADDL(t4, a[3], sl1);
    t5 = VMULADDH(t5, a[3], sl1);
    t5 = VSL(t5, 1);

    // left i+j = 5
    t5 = VMULADDL(t5, a[0], sh1);
    t6 = VMULADDH(zero, a[0], sh1);
    t5 = VMULADDL(t5, a[1], sh0);
    t6 = VMULADDH(t6, a[1], sh0);
    t5 = VMULADDL(t5, a[2], sl3);
    t6 = VMULADDH(t6, a[2], sl3);
    t5 = VMULADDL(t5, a[3], sl2);
    t6 = VMULADDH(t6, a[3], sl2);
    t6 = VSL(t6, 1);

    // left i+j = 6
    t6 = VMULADDL(t6, a[0], sh2);
    t7 = VMULADDH(zero, a[0], sh2);
    t6 = VMULADDL(t6, a[1], sh1);
    t7 = VMULADDH(t7, a[1], sh1);
    t6 = VMULADDL(t6, a[2], sh0);
    t7 = VMULADDH(t7, a[2], sh0);
    t6 = VMULADDL(t6, a[3], sl3);
    t7 = VMULADDH(t7, a[3], sl3);
    t7 = VSL(t7, 1);

    // left i+j = 7
    t7 = VMULADDL(t7, a[0], sh3);
    t8 = VMULADDH(zero, a[0], sh3);
    t7 = VMULADDL(t7, a[1], sh2);
    t8 = VMULADDH(t8, a[1], sh2);
    t7 = VMULADDL(t7, a[2], sh1);
    t8 = VMULADDH(t8, a[2], sh1);
    t7 = VMULADDL(t7, a[3], sh0);
    t8 = VMULADDH(t8, a[3], sh0);
    t8 = VSL(t8, 1);

    // left i+j = 8
    t8 = VMULADDL(t8, a[1], sh3);
    t9 = VMULADDH(zero, a[1], sh3);
    t8 = VMULADDL(t8, a[2], sh2);
    t9 = VMULADDH(t9, a[2], sh2);
    t8 = VMULADDL(t8, a[3], sh1);
    t9 = VMULADDH(t9, a[3], sh1);
    t9 = VSL(t9, 1);
    
    // left i+j = 9
    t9 = VMULADDL(t9, a[2], sh3);
    t10 = VMULADDH(zero, a[2], sh3);
    t9 = VMULADDL(t9, a[3], sh2);
    t10 = VMULADDH(t10, a[3], sh2);
    t10 = VSL(t10, 1);

    // left i+j = 10
    t10 = VMULADDL(t10, a[3], sh3);
    t11 = VMULADDH(zero, a[3], sh3);

    // Part 2: Montgomery reduction

    // TODO
}