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

// Montgomery constant R = 2^(11*51) mod p
static const uint64_t R[NUM_LIMBS] =
{
    0x25, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0,
    0x0, 0x320000000000
};

// R2 = R^2 mod p
static const uint64_t R2[NUM_LIMBS] =
{
    0x97b425ed0f19, 0x4bda12f684bda, 0x5ed097b425ed0, 0x7684bda12f684,
    0x3425ed097b425, 0x212f684bda12f, 0x97b425ed097b, 0x4bda12f684bda,
    0x5ed097b425ed0, 0x224bda12f684
};

// 3^-1 * R mod p
static const uint64_t INV3[NUM_LIMBS] =
{
    0x5555555555561, 0x2aaaaaaaaaaaa, 0x5555555555555, 0x2aaaaaaaaaaaa,
    0x5555555555555, 0x2aaaaaaaaaaaa, 0x5555555555555, 0x2aaaaaaaaaaaa,
    0x5555555555555, 0x34aaaaaaaaaa
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
    0x25, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x190000000000000
};

// Constant used to correct output of fp_inv64
static const uint64_t INV_CONST_64[NWORDS64] =
{
    0x2000, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0
};

#endif
