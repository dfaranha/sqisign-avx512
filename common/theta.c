#include <stdio.h> //TODO: Remove

#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "theta.h"

// ======================================================================================================
//                                                 4x2x1
// ======================================================================================================

void theta_sqr_single(__m512i* res, __m512i* a) 
{
    fp2_sqr_4x2x1(res, a);
}

void theta_mul_single(__m512i* res, __m512i* a, __m512i* b) 
{
    fp2_mul_4x2x1(res, a, b);
}

// TODO: Could we merge splitting with mixaddsub and use less tmp registers?
void theta_had_single(__m512i* res, __m512i* a) 
{
    __m512i t[8];
    __m512i pqpq = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    __m512i rsrs = VSET(4, 5, 6, 7, 4, 5, 6, 7);

    // Split
    for (int i = 0; i < 8; i++) {
        t[i] = VPERM256(a[i], 0x44); //aacc
        res[i] = VPERM256(a[i], 0xee); //bbdd
    }

    // add sub add sub
    fp_mixaddsub_8x1(res, t, res, 0x33); // p, q, r, s = a + b, a - b, c + d, c - d
    
    // Split 
    for (int i = 0; i < 8; i++) {
        t[i] = VPERM512(pqpq, res[i]);
        res[i] = VPERM512(rsrs, res[i]);
    }

    // add add sub sub
    fp_mixaddsub_8x1(res, t, res, 0xf); // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
}

// ======================================================================================================
//                                                 8x1x1
// ======================================================================================================

void theta_sqr_pair(__m512i* res, __m512i* a) 
{
    fp2_sqr_8x1x1(res, a);
}

void theta_mul_pair(__m512i* res, __m512i* a, __m512i* b) 
{
    fp2_mul_8x1x1(res, a, b);
}

void theta_had_pair(__m512i* res, __m512i* a) 
{
    __m512i t[16];
    __m512i pqpq = VSET(0, 1, 2, 3, 0, 1, 2, 3);
    __m512i rsrs = VSET(4, 5, 6, 7, 4, 5, 6, 7);

    // Split
    for (int i = 0; i < 16; i++) {
        t[i] = VPERM256(a[i], 0x44); //aacc
        res[i] = VPERM256(a[i], 0xee); //bbdd
    }

    // add sub add sub
    fp_mixaddsub_8x1(res, t, res, 0x33); // p, q, r, s = a + b, a - b, c + d, c - d
    fp_mixaddsub_8x1(res+8, t+8, res+8, 0x33);

    // Split 
    for (int i = 0; i < 16; i++) {
        t[i] = VPERM512(pqpq, res[i]);
        res[i] = VPERM512(rsrs, res[i]);
    }

    // add add sub sub
    fp_mixaddsub_8x1(res, t, res, 0xf); // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
    fp_mixaddsub_8x1(res+8, t+8, res+8, 0xf);
}

// Like 1 cycle faster, but assumes we store the pairs as (a1, b1, c1, d1, a2, b2, c2, d2)
void theta_had_pair_alt(__m512i* res, __m512i* a) 
{
    __m512i t[16];

    // Split
    for (int i = 0; i < 16; i++) {
        t[i] = VSHUFFLE(a[i], 0x44); //aacc
        res[i] = VSHUFFLE(a[i], 0xee); //bbdd
    }

    // add sub add sub
    fp_mixaddsub_8x1(res, t, res, 0x55); // p, q, r, s = a + b, a - b, c + d, c - d
    fp_mixaddsub_8x1(res+8, t+8, res+8, 0x55);

    // Split 
    for (int i = 0; i < 16; i++) {
        t[i] = VPERM256(res[i], 0x44);
        res[i] = VPERM256(res[i], 0xee);
    }

    // add add sub sub
    fp_mixaddsub_8x1(res, t, res, 0x33); // p + r, q + s, p - r, q - s = a + b + c + d, a - b + c - d, a + b - c - d, a - b - c + d
    fp_mixaddsub_8x1(res+8, t+8, res+8, 0x33);
}