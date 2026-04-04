#ifndef _FP_CONSTS_H
#define _FP_CONSTS_H

#include <stdint.h>

#define LIMB_LEN 48
#define NUM_LIMBS 8
#define LIMB_LEN 52 - LIMB_LEN
#define NUM_LIMBS_64 6

#define N4x2 4
#define N2x4 2

#define LIMB_MASK 0xffffffffffffULL
#define LAST_LIMB_MASK 0x7fffffffffffULL

#define P_PLUS_1_TOP 0x410000000000ULL

// Montgomery constant R = 2^(9*48) mod p
static const uint64_t R[NUM_LIMBS] = {
    0xf03f03f03f03, 0x3, 0x0, 0x0,
    0x0, 0x0, 0x0, 0x3d0000000000
};

// R2 = R^2 mod p
static const uint64_t R2[NUM_LIMBS] = {
    0x7b0fd07f000f, 0x732f52771f91, 0xf03f03f03f13, 0xf03f03f03f03,
    0xf03f03f03f03, 0xf03f03f03f03, 0xf03f03f03f03, 0x213f03f03f03
};

#endif
