/**
*   The SQISign lvl3 prime is p = 65 * 2^376 - 1
*   It is 383 bits, so the number of 51-bit limbs needed is 383/51 < 8
*   
*   Formats: a x b = a vectors of b Fp limbs
*/

#ifndef _FP_LVL3_H
#define _FP_LVL3_H

#include <stdint.h>
#include "intrinsics_avx512.h"

static const uint64_t LIMB_LEN = 51;
static const uint64_t LIMB_MASK = 0x7ffffffffffff;
static const uint64_t LAST_LIMB_MASK = 0x3ffffff;
// p in radix 2^51
static const uint64_t p[8] = { 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x207ffff };
// 2*p in radix 2^51
static const uint64_t p2[8] = { 0x7fffffffffffe, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x7ffffffffffff, 0x40fffff };


void fp_setzero_8x1(__m512i* res);

void fp_copy_8x1(__m512i* res, __m512i* a);

/**
*   returns an 8-bit mask which is 1 iff the i-th Fp element is 1
*/
int fp_is_equal_8x1(__m512i* a, __m512i* b);

/**
*    res = a + b (mod 2p)
*/
void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b);

#endif