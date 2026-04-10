#ifndef _FP_CONSTS_H
#define _FP_CONSTS_H

#include <stdint.h>

#define LIMB_LEN 51
#define NUM_LIMBS 5
#define LIMB_ALIGN 52 - LIMB_LEN
#define NWORDS64 4

#define N4x2 3
#define N2x4 2

#define LIMB_MASK 0x7ffffffffffffULL
#define LAST_LIMB_MASK 0x7fffffffffffULL

#define P_PLUS_1_TOP 0x500000000000ULL

// Montgomery constant R = 2^(6*51) mod p
static const uint64_t R[NUM_LIMBS] =
{
    0x19, 0x0, 0x0, 0x0,
    0x300000000000
};

// R2 = R^2 mod p
static const uint64_t R2[NUM_LIMBS] =
{
    0x4cccccccccf5c, 0x1999999999999, 0x3333333333333, 0x6666666666666,
    0xccccccccccc
};

// 3^-1 * R mod p
static const uint64_t INV3[NUM_LIMBS] =
{
    0x555555555555d, 0x2aaaaaaaaaaaa, 0x5555555555555, 0x2aaaaaaaaaaaa,
    0x455555555555
};

// p in 64-bit repr.
static const uint64_t P_64[NWORDS64] =
{
    0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0x4ffffffffffffff
};

// R in 64-bit repr.
static const uint64_t R_64[NWORDS64] =
{
    0x19, 0x0, 0x0, 0x300000000000000
};

// Constant used to correct output of fp_inv64
static const uint64_t INV_CONST_64[NWORDS64] =
{
    0x400, 0x0, 0x0, 0x0
};

#endif
