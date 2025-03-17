// Here we can define custom intrinsics for AVX512 instructions that we use

#ifndef _INTRINSICS_AVX512_H
#define _INTRINSICS_AVX512_H

#include <immintrin.h>

// Arithmetic
#define VADD(A,B) _mm512_add_epi64(A, B) // DST = A + B
#define VADDMASK(SRC, mask, A, B) _mm512_mask_add_epi64(SRC, mask, A, B) // If mask[i] = 1, DST[i] = A[i] + B[i]. Else, DST[i] = SRC[i]
#define VSUB(A,B) _mm512_sub_epi64(A, B) // DST = A - B
#define VSUBMASK(SRC, mask, A, B) _mm512_mask_sub_epi64(SRC, mask, A, B) // If mask[i] = 1, DST[i] = A[i] - B[i]. Else, DST[i] = SRC[i]
#define VMULADDL(A, B, C) _mm512_madd52lo_epu64(A, B, C) // DST = A + LOW(B * C)
#define VMULADDH(A, B, C) _mm512_madd52hi_epu64(A, B, C) // DST = A + HIGH(B * C)

// Logical operators
#define VAND(A, B) _mm512_and_epi64(A, B) // DST = A AND B
#define VOR(A, B) _mm512_or_epi64(A, B) // DST = A OR B
#define VXOR(A, B) _mm512_xor_epi64(A, B) // DST = A XOR B
#define VSRA(A, n) _mm512_srai_epi64(A, n) // DST = A >> n (arithmetic right shift)
#define VSL(A, n) _mm512_slli_epi64(A, n) // DST = AA << n

// Comparison
#define VCMPEQ(A, B) _mm512_cmpeq_epi64_mask(A, B) // DST is an 8-bit mask, where the i-th bit is 1 if the i-th limbs are equal
#define VSIGN(A) _mm512_movepi64_mask(A) // DST is an 8-bit mask, where the i-th bit is 1 if the i-th limb has top bit 1

// Assignment
#define VSET(A, B, C, D, E, F, G, H) _mm512_set_epi64(H, G, F, E, D, C, B, A) // DST = { A, B, C, D, E, F, G, H}
#define VSETALL(a) _mm512_set1_epi64(a) // DST is filled with 8 copies of a
#define VZERO() _mm512_setzero_si512() // DST = 0

#endif