#ifndef _FP_CONSTS_H
#define _FP_CONSTS_H

#include <stdint.h>

#define LIMB_LEN 49
#define NUM_LIMBS 8
#define LIMB_ALIGN 52 - LIMB_LEN
#define NWORDS64 6

#define N4x2 4
#define N2x4 2

#define LIMB_MASK 0x1ffffffffffffULL
#define LAST_LIMB_MASK 0xffffffffffULL

#define P_PLUS_1_TOP 0x8200000000ULL

// Montgomery constant R = 2^(9*49) mod p
static const uint64_t R[NUM_LIMBS] =
{
    0x7e07e07e07e0, 0x3f0, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x4000000000
};

// R2 = R^2 mod p
static const uint64_t R2[NUM_LIMBS] =
{
    0x141fc003e0bc1, 0xa4ee3f22f61f, 0x103f03f13732f, 0x181f81f81f81f,
    0x1c0fc0fc0fc0f, 0x1e07e07e07e07, 0x1f03f03f03f03, 0x1d81f81f81
};

// 3^-1 * R mod p
static const uint64_t INV3[NUM_LIMBS] =
{
    0x17f57f57f57f5, 0xaaaaaaaaabfa, 0x1555555555555, 0xaaaaaaaaaaaa,
    0x1555555555555, 0xaaaaaaaaaaaa, 0x1555555555555, 0x40aaaaaaaa
};

// p in 64-bit repr.
static const uint64_t P_64[NWORDS64] =
{
    0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
    0xffffffffffffffff, 0x40ffffffffffffff
};

// R in 64-bit repr.
static const uint64_t R_64[NWORDS64] =
{
    0x7e07e07e07e07e0, 0x0, 0x0, 0x0,
    0x0, 0x2000000000000000
};

// Constant used to correct output of fp_inv64
static const uint64_t INV_CONST_64[NWORDS64] =
{
    0x1000000000000000, 0x0, 0x0, 0x0,
    0x0, 0x0
};

#endif
