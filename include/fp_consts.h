#ifndef _FP_CONSTS_H
#define _FP_CONSTS_H

#include <stdint.h>

#define LIMB_LEN 48
#define NUM_LIMBS 8
#define LIMB_ALIGN 52 - LIMB_LEN
#define NWORDS64 6

#define N4x2 4
#define N2x4 2

#define LIMB_MASK 0xffffffffffffULL
#define LAST_LIMB_MASK 0x7fffffffffffULL

#define P_PLUS_1_TOP 0x410000000000ULL
#define P_PLUS_1_DIV2_TOP 0x208000000000ULL

// Montgomery constant R = 2^(8*48) mod p
static const uint64_t R[NUM_LIMBS] =
{
    0xf03f03f03f03, 0x3, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x3d0000000000
};

// R2 = R^2 mod p
static const uint64_t R2[NUM_LIMBS] =
{
    0x7b0fd07f000f, 0x732f52771f91, 0xf03f03f03f13, 0xf03f03f03f03,
    0xf03f03f03f03, 0xf03f03f03f03, 0xf03f03f03f03, 0x213f03f03f03
};

// 3^-1 * R mod p
static const uint64_t INV3[NUM_LIMBS] =
{
    0xfabfabfabfab, 0xaaaaaaaaaaab, 0xaaaaaaaaaaaa, 0xaaaaaaaaaaaa,
    0xaaaaaaaaaaaa, 0xaaaaaaaaaaaa, 0xaaaaaaaaaaaa, 0x3faaaaaaaaaa
};

// Cofactor c of p = c*2^e - 1
#define COFACTOR_NBITS 7
#define COFACTOR_NWORDS64 1
static const uint64_t COFACTOR[COFACTOR_NWORDS64] =
{
    0x41
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
    0x3f03f03f03f03, 0x0, 0x0, 0x0,
    0x0, 0x3d00000000000000
};

// Constant used to correct output of fp_inv64
static const uint64_t INV_CONST_64[NWORDS64] =
{
    0x8000000000000, 0x0, 0x0, 0x0,
    0x0, 0x0
};

#endif
