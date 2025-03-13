#include <stdio.h>
#include "fp_lvl3.h"

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

// Assumes that elements are fully reduced mod p, so there is only one representative.
int fp_is_equal_8x1(__m512i* a, __m512i* b)
{
    __mmask8 mask, acc, res;
    acc = 0xFF;
    for(int i = 0; i < 8; i++){
        mask = VCMPEQ(a[i], b[i]);
        acc = acc & mask;
    }
    return acc;
}

//IDEA: mask_mov to not have to add p again?

// Following AVX512 SIKE paper
// Currently the arithmetic is mod p, 
// I think they do mod 2p to not have to do the final step of the Montgomery reduction,
// but we have a special prime, so this is probably not needed?
void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b)
{
    __m512i t0, t1, t2, t3, t4, t5, t6, t7;
    __m512i p_0, p_1, p_2, p_3, p_4, p_5, p_6, p_7;
    __m512i limb_mask, sign_mask;
    limb_mask = VSETALL(LIMB_MASK);
    
    // Add 51-bit limbs
    t0 = VADD(a[0], b[0]);
    t1 = VADD(a[1], b[1]);
    t2 = VADD(a[2], b[2]);
    t3 = VADD(a[3], b[3]);
    t4 = VADD(a[4], b[4]);
    t5 = VADD(a[5], b[5]);
    t6 = VADD(a[6], b[6]);
    t7 = VADD(a[7], b[7]);

    // // Load p
    // p_0 = VSETALL(p[0]);
    // p_1 = VSETALL(p[1]);
    // p_2 = VSETALL(p[2]);
    // p_3 = VSETALL(p[3]);
    // p_4 = VSETALL(p[4]);
    // p_5 = VSETALL(p[5]);
    // p_6 = VSETALL(p[6]);
    // p_7 = VSETALL(p[7]);

    // // Subtract p
    // t0 = VSUB(t0, p_0);
    // t1 = VSUB(t1, p_1);
    // t2 = VSUB(t2, p_2);
    // t3 = VSUB(t3, p_3);
    // t4 = VSUB(t4, p_4);
    // t5 = VSUB(t5, p_5);
    // t6 = VSUB(t6, p_6);
    // t7 = VSUB(t7, p_7);

    // // Propogate carries
    // t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    // t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    // t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    // t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    // t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    // t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    // t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    // t0 = VAND(t0, limb_mask);
    // t1 = VAND(t1, limb_mask);
    // t2 = VAND(t2, limb_mask);
    // t3 = VAND(t3, limb_mask);
    // t4 = VAND(t4, limb_mask);
    // t5 = VAND(t5, limb_mask);
    // t6 = VAND(t6, limb_mask);

    // // If result is negative, we undo the subtraction by p
    // sign_mask = VSRA(t7, 63);
    
    // t0 = VADD(t0, VAND(p_0, sign_mask));
    // t1 = VADD(t1, VAND(p_1, sign_mask));
    // t2 = VADD(t2, VAND(p_2, sign_mask));
    // t3 = VADD(t3, VAND(p_3, sign_mask));
    // t4 = VADD(t4, VAND(p_4, sign_mask));
    // t5 = VADD(t5, VAND(p_5, sign_mask));
    // t6 = VADD(t6, VAND(p_6, sign_mask));
    // t7 = VADD(t7, VAND(p_7, sign_mask));

    // // Propogate the carry
    // t1 = VADD(t1, VSRA(t0, LIMB_LEN));
    // t2 = VADD(t2, VSRA(t1, LIMB_LEN));
    // t3 = VADD(t3, VSRA(t2, LIMB_LEN));
    // t4 = VADD(t4, VSRA(t3, LIMB_LEN));
    // t5 = VADD(t5, VSRA(t4, LIMB_LEN));
    // t6 = VADD(t6, VSRA(t5, LIMB_LEN));
    // t7 = VADD(t7, VSRA(t6, LIMB_LEN));
    // t0 = VAND(t0, limb_mask);
    // t1 = VAND(t1, limb_mask);
    // t2 = VAND(t2, limb_mask);
    // t3 = VAND(t3, limb_mask);
    // t4 = VAND(t4, limb_mask);
    // t5 = VAND(t5, limb_mask);
    // t6 = VAND(t6, limb_mask);

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