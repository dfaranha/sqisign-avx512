#ifndef _FP_LVL3_H
#define _FP_LVL3_H

#include "intrinsics_avx512.h"

/**
*   The SQISign lvl3 prime is p = 65 * 2^376 - 1
*   It is 383 bits, so the number of 51-bit limbs needed is 383/51 < 8
*   
*   Formats: a x b = a vectors of b Fp limbs
*/


/**
    c = a + b (mod p)
*/
void fp_add_1x8(__m512i a, __m512i b, __m512i c);

#endif