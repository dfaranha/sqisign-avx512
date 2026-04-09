#ifndef _FP_CONSTS_H
#define _FP_CONSTS_H

#include <stdint.h>

#define LIMB_LEN 48
#define NUM_LIMBS 11
#define LIMB_ALIGN 52 - LIMB_LEN
#define NWORDS64 9

#define N4x2 6
#define N2x4 3

#define LIMB_MASK 0xffffffffffffULL
#define LAST_LIMB_MASK 0x1ffffffffffULL

#define P_PLUS_1_TOP 0x20000000000ULL

// Montgomery constant R = 2^(12*48) mod p
static const uint64_t R[NUM_LIMBS] =
{
    0x0, 0x80, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0
};

// R2 = R^2 mod p
static const uint64_t R2[NUM_LIMBS] =
{
    0x0, 0x0, 0x4000, 0x0,
    0x0, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0
};

// 3^-1 * R mod p
static const uint64_t INV3[NUM_LIMBS] =
{
    0x555555555555, 0xaaaaaaaaaad5, 0xaaaaaaaaaaaa, 0xaaaaaaaaaaaa,
    0xaaaaaaaaaaaa, 0xaaaaaaaaaaaa, 0xaaaaaaaaaaaa, 0xaaaaaaaaaaaa,
    0xaaaaaaaaaaaa, 0xaaaaaaaaaaaa, 0xaaaaaaaaaa
};

// p in 64-bit repr.
static const uint64_t P_64[NWORDS64] =
{
    0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
    0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
    0x1ff
};

// R in 64-bit repr.
static const uint64_t R_64[NWORDS64] =
{
    0x80000000000000, 0x0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0,
    0x0
};

// Constant used to correct output of fp_inv64
static const uint64_t INV_CONST_64[NWORDS64] =
{
    0x0, 0x800000000000, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x0,
    0x0
};

#endif
