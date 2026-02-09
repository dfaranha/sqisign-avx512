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
#define VANDMASK(SRC, mask, A, B) _mm512_mask_and_epi64(SRC, mask, A, B) // If mask[i] = 1, DST[i] = A[i] AND B[i]. Else, DST[i] = SRC[i]
#define VOR(A, B) _mm512_or_epi64(A, B) // DST = A OR B
#define VXOR(A, B) _mm512_xor_epi64(A, B) // DST = A XOR B
#define VSRA(A, n) _mm512_srai_epi64(A, n) // DST = A >> n (arithmetic right shift)
#define VSRAMASKZ(mask, A, n) _mm512_maskz_srai_epi64(mask, A, n) // If mask[i] = 1, DST[i] = A[i] >> n, else DST[i] = 0
#define VSL(A, n) _mm512_slli_epi64(A, n) // DST = AA << n

// Comparison
#define VCMPEQ(A, B) _mm512_cmpeq_epi64_mask(A, B) // DST is an 8-bit mask, where the i-th bit is 1 if the i-th limbs are equal
#define VSIGN(A) _mm512_movepi64_mask(A) // DST is an 8-bit mask, where the i-th bit is 1 if the i-th limb has top bit 1
#define VTEST(A, B) _mm512_test_epi64_mask(A, B) // DST is an 8-bit mask, where the i-th bit is 1 if A[i] & B[i] is non-zero.

// Assignment
#define VSET(A, B, C, D, E, F, G, H) _mm512_set_epi64(H, G, F, E, D, C, B, A) // DST = { A, B, C, D, E, F, G, H}
#define VSETALL(a) _mm512_set1_epi64(a) // DST is filled with 8 copies of a
#define VZERO() _mm512_setzero_si512() // DST = 0
#define VSTORE(a, B) _mm512_storeu_epi64(a, B) // Store the 512-bit vector B as 8 64-bit integers in the array a.

// Permutations & Shuffles
#define VSHUFFLE(A, pattern) _mm512_shuffle_epi32(A, pattern) // Shuffles 32-bit limbs within each 128-bit lane, according to pattern. The first two bits of pattern specify where the first limb goes, etc.
#define VSHUFFLEMASKZ(mask, A, pattern) _mm512_maskz_shuffle_epi32(mask, A, pattern) // If mask[i] = 1, the i-th 32-bit limb of the output is the result, else it is 0.
#define VPERM256(A, pattern) _mm512_permutex_epi64(A, pattern) // Shuffles 64-bit limbs within each 256-bit lane. Latency is 3 VS 1 for shuffles, but throughput is 1 for both.
#define VPERM256MASKZ(mask, A, pattern) _mm512_maskz_permutex_epi64(mask, A, pattern) // If mask[i] = 1, the i-th 64-bit limb of the output is the result, else it is 0.
#define VPERM256HALF(A, B, pattern) _mm512_shuffle_i64x2(A, B, pattern) // the 3 lsb of pattern[i] determine index, the 4-th bit is 1 if the source is B and 0 if the source is A.
#define VPERM512(indices, A) _mm512_permutexvar_epi64(indices, A)  // Shuffles 64-bit limbs within the entire 512-bit vector. DST[i] = A[indices[i]]
#define VPERM512MASKZ(mask, A, indices) _mm512_maskz_permutexvar_epi64(mask, indices, A)
#define VBLEND(mask, A, B) _mm512_mask_blend_epi64(mask, A, B) // If mask[i] = 1, DST[i] = B[i]. Else, DST[i] = A[i].
#define VBLENDPERM(indices, A, B) _mm512_permutex2var_epi64(A, indices, B) // the 3 lsb of indices[i] determine index, the 4-th bit is 1 if the source is B and 0 if the source is A.

/**
 * Operates on 32-bit limbs. If mask[i] = 0, DST[i] = A[i]. Else, it parses the i-th 32-bit limb of indices. 
 * The 4 least significant bits determine the index of the 32-bit limb from the src. The 5th bit determines if the src is A (0) or B (1).
 */
#define VBLENDPERM32(mask, A, B, indices) _mm512_mask_permutex2var_epi32(A, mask, indices, B)   
#endif