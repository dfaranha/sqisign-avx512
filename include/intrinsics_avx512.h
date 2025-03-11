// Here we can define custom intrinsics for AVX512 instructions that we use

#ifndef _INTRINSICS_AVX512_H
#define _INTRINSICS_AVX512_H

#include <immintrin.h>

#define VMULADDL(A, B, C) _mm512_madd52lo_epu64(A, B, C)

#endif