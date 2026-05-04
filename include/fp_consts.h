#ifndef _FP_CONSTS_H
#define _FP_CONSTS_H

#include <stdint.h>

#define LIMB_LEN 51
#define NUM_LIMBS 10
#define LIMB_ALIGN 52 - LIMB_LEN
#define NWORDS64 8

#define N4x2 5
#define N2x4 3

#define LIMB_MASK 0x7ffffffffffffULL
#define LAST_LIMB_MASK 0x3fffffffffffULL

#define P_PLUS_1_TOP 0x360000000000ULL
#define P_PLUS_1_DIV2_TOP 0x1b0000000000ULL

// Montgomery constant R = 2^(11*51) mod p
static const uint64_t R[NUM_LIMBS] =
{
    0x7684bda12f684, 0x25, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0,
    0x0, 0x280000000000
};

// R2 = R^2 mod p
static const uint64_t R2[NUM_LIMBS] =
{
    0xbf08c77657ca, 0x514b802cf301c, 0x97b425ed0f19, 0x4bda12f684bda,
    0x5ed097b425ed0, 0x7684bda12f684, 0x3425ed097b425, 0x212f684bda12f,
    0x97b425ed097b, 0x21a12f684bda
};

// 3^-1 * R mod p
static const uint64_t INV3[NUM_LIMBS] =
{
    0x7cd6e9e06522c, 0x5555555555561, 0x2aaaaaaaaaaaa, 0x5555555555555,
    0x2aaaaaaaaaaaa, 0x5555555555555, 0x2aaaaaaaaaaaa, 0x5555555555555,
    0x2aaaaaaaaaaaa, 0xd5555555555
};

// Cofactor c of p = c*2^e - 1
#define COFACTOR_NBITS 5
#define COFACTOR_NWORDS64 1
static const uint64_t COFACTOR[COFACTOR_NWORDS64] =
{
    0x1b
};

// p in 64-bit repr.
static const uint64_t P_64[NWORDS64] =
{
    0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
    0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0x1afffffffffffff
};

// R in 64-bit repr.
static const uint64_t R_64[NWORDS64] =
{
    0x12f684bda12f684, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x140000000000000
};

// Constant used to correct output of fp_inv64
static const uint64_t INV_CONST_64[NWORDS64] =
{
    0x0, 0x1, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0
};

#endif
