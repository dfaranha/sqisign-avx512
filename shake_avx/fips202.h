// SPDX-License-Identifier: Apache-2.0

#ifndef FIPS202_H
#define FIPS202_H

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include "intrinsics_avx512.h"

#define SHAKE128_RATE 168
#define SHAKE256_RATE 136
#define SHA3_256_RATE 136
#define SHA3_384_RATE 104
#define SHA3_512_RATE 72

#define PQC_SHAKEINCCTX_U64WORDS 26
#define PQC_SHAKECTX_U64WORDS 25

#define PQC_SHAKEINCCTX_BYTES (sizeof(uint64_t) * 26)
#define PQC_SHAKECTX_BYTES (sizeof(uint64_t) * 25)

// Context for incremental API
typedef struct {
    uint64_t ctx[PQC_SHAKEINCCTX_U64WORDS];
} shake128incctx;

// Context for non-incremental API
typedef struct {
    uint64_t ctx[PQC_SHAKECTX_U64WORDS];
} shake128ctx;

// Context for incremental API
typedef struct {
    uint64_t ctx[PQC_SHAKEINCCTX_U64WORDS];
} shake256incctx;

// Context for non-incremental API
typedef struct {
    uint64_t ctx[PQC_SHAKECTX_U64WORDS];
} shake256ctx;

// Context for incremental API
typedef struct {
    uint64_t ctx[PQC_SHAKEINCCTX_U64WORDS];
} sha3_256incctx;

// Context for incremental API
typedef struct {
    uint64_t ctx[PQC_SHAKEINCCTX_U64WORDS];
} sha3_384incctx;

// Context for incremental API
typedef struct {
    uint64_t ctx[PQC_SHAKEINCCTX_U64WORDS];
} sha3_512incctx;

/* Initialize the state and absorb the provided input.
 *
 * This function does not support being called multiple times
 * with the same state.
 */
void shake128_absorb(shake128ctx *state, const uint8_t *input, size_t inlen);
/* Squeeze output out of the sponge.
 *
 * Supports being called multiple times
 */
void shake128_squeezeblocks(uint8_t *output, size_t nblocks, shake128ctx *state);
/* Free the state */
void shake128_ctx_release(shake128ctx *state);
/* Copy the state. */
void shake128_ctx_clone(shake128ctx *dest, const shake128ctx *src);

/* Initialize incremental hashing API */
void shake128_inc_init(shake128incctx *state);
/* Absorb more information into the XOF.
 *
 * Can be called multiple times.
 */
void shake128_inc_absorb(shake128incctx *state, const uint8_t *input, size_t inlen);
/* Finalize the XOF for squeezing */
void shake128_inc_finalize(shake128incctx *state);
/* Squeeze output out of the sponge.
 *
 * Supports being called multiple times
 */
void shake128_inc_squeeze(uint8_t *output, size_t outlen, shake128incctx *state);
/* Copy the context of the SHAKE128 XOF */
void shake128_inc_ctx_clone(shake128incctx *dest, const shake128incctx *src);
/* Free the context of the SHAKE128 XOF */
void shake128_inc_ctx_release(shake128incctx *state);

/* Initialize the state and absorb the provided input.
 *
 * This function does not support being called multiple times
 * with the same state.
 */
void shake256_absorb(shake256ctx *state, const uint8_t *input, size_t inlen);
/* Squeeze output out of the sponge.
 *
 * Supports being called multiple times
 */
void shake256_squeezeblocks(uint8_t *output, size_t nblocks, shake256ctx *state);
/* Free the context held by this XOF */
void shake256_ctx_release(shake256ctx *state);
/* Copy the context held by this XOF */
void shake256_ctx_clone(shake256ctx *dest, const shake256ctx *src);

/* Initialize incremental hashing API */
void shake256_inc_init(shake256incctx *state);
void shake256_inc_absorb(shake256incctx *state, const uint8_t *input, size_t inlen);
/* Prepares for squeeze phase */
void shake256_inc_finalize(shake256incctx *state);
/* Squeeze output out of the sponge.
 *
 * Supports being called multiple times
 */
void shake256_inc_squeeze(uint8_t *output, size_t outlen, shake256incctx *state);
/* Copy the state */
void shake256_inc_ctx_clone(shake256incctx *dest, const shake256incctx *src);
/* Free the state */
void shake256_inc_ctx_release(shake256incctx *state);

/* One-stop SHAKE128 call */
void shake128(uint8_t *output, size_t outlen,
              const uint8_t *input, size_t inlen);

/* One-stop SHAKE256 call */
void shake256(uint8_t *output, size_t outlen,
              const uint8_t *input, size_t inlen);

/* Initialize the incremental hashing state */
void sha3_256_inc_init(sha3_256incctx *state);
/* Absorb blocks into SHA3 */
void sha3_256_inc_absorb(sha3_256incctx *state, const uint8_t *input, size_t inlen);
/* Obtain the output of the function and free `state` */
void sha3_256_inc_finalize(uint8_t *output, sha3_256incctx *state);
/* Copy the context */
void sha3_256_inc_ctx_clone(sha3_256incctx *dest, const sha3_256incctx *src);
/* Release the state, don't use if `_finalize` has been used */
void sha3_256_inc_ctx_release(sha3_256incctx *state);

void sha3_256(uint8_t *output, const uint8_t *input, size_t inlen);

/* Initialize the incremental hashing state */
void sha3_384_inc_init(sha3_384incctx *state);
/* Absorb blocks into SHA3 */
void sha3_384_inc_absorb(sha3_384incctx *state, const uint8_t *input, size_t inlen);
/* Obtain the output of the function and free `state` */
void sha3_384_inc_finalize(uint8_t *output, sha3_384incctx *state);
/* Copy the context */
void sha3_384_inc_ctx_clone(sha3_384incctx *dest, const sha3_384incctx *src);
/* Release the state, don't use if `_finalize` has been used */
void sha3_384_inc_ctx_release(sha3_384incctx *state);

/* One-stop SHA3-384 shop */
void sha3_384(uint8_t *output, const uint8_t *input, size_t inlen);

/* Initialize the incremental hashing state */
void sha3_512_inc_init(sha3_512incctx *state);
/* Absorb blocks into SHA3 */
void sha3_512_inc_absorb(sha3_512incctx *state, const uint8_t *input, size_t inlen);
/* Obtain the output of the function and free `state` */
void sha3_512_inc_finalize(uint8_t *output, sha3_512incctx *state);
/* Copy the context */
void sha3_512_inc_ctx_clone(sha3_512incctx *dest, const sha3_512incctx *src);
/* Release the state, don't use if `_finalize` has been used */
void sha3_512_inc_ctx_release(sha3_512incctx *state);

/* One-stop SHA3-512 shop */
void sha3_512(uint8_t *output, const uint8_t *input, size_t inlen);



// Keccak function implemented in AVX512 by Roberto Camal:
#define ALIGN __attribute__ ((aligned (32)))
#define KECCAK256

typedef unsigned long long uInt256[4];
typedef unsigned long long uInt512[8];

// we only use the inner keccak function
void keccakF(__m512i* x, int rnd);

// function for repeated hash in SQIsign calling keccakF
// SHAKE256, single-block in / single-block out, fresh state.
// Constraints: inlen ≤ 135, outlen ≤ 136.
// inline void shake256_oneblock(uint8_t *out, size_t outlen, const uint8_t *in, size_t inlen) {
//     uint64_t block[17] = {0};
//     memcpy(block, in, inlen);
//     ((uint8_t *)block)[inlen]    = 0x1F;
//     ((uint8_t *)block)[136 - 1] |= 0x80;

//     __m512i x[5];
//     x[0] = VLOADMASKZ(0x1F, (const __m512i *)(block +  0));
//     x[1] = VLOADMASKZ(0x1F, (const __m512i *)(block +  5));
//     x[2] = VLOADMASKZ(0x1F, (const __m512i *)(block + 10));
//     x[3] = VLOADMASKZ(0x03, (const __m512i *)(block + 15));
//     x[4] = VZERO();

//     keccakF(x, 24);

//     uint64_t outbuf[17];
//     VSTOREMASKZ((__m512i *)(outbuf +  0), 0x1F, x[0]);
//     VSTOREMASKZ((__m512i *)(outbuf +  5), 0x1F, x[1]);
//     VSTOREMASKZ((__m512i *)(outbuf + 10), 0x1F, x[2]);
//     VSTOREMASKZ((__m512i *)(outbuf + 15), 0x03, x[3]);
//     memcpy(out, outbuf, outlen);
// }

// specialized versions:
// lvl1
static inline void shake256_oneblock_32(uint8_t *out, const uint8_t *in) {
    __m512i x[5];

    // 32 bytes = 4 full qwords, all in x[0] lanes 0..3.
    // Domain 0x1F at byte 32 = qword 4 = x[0] lane 4.
    uint64_t msg[4];
    memcpy(msg, in, 32);
    x[0] = _mm512_set_epi64(0, 0, 0, 0x1F,
                            msg[3], msg[2], msg[1], msg[0]);
    x[1] = _mm512_setzero_si512();
    x[2] = _mm512_setzero_si512();
    x[3] = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0x8000000000000000ULL, 0);
    x[4] = _mm512_setzero_si512();

    keccakF(x, 24);

    // Squeeze 32 bytes: x[0] lanes 0..3.
    _mm512_mask_storeu_epi64((__m512i *)out, 0x0F, x[0]);
}

// lvl3
static inline void shake256_oneblock_48(uint8_t *out, const uint8_t *in) {
    __m512i x[5];

    // x[0]: load 5 qwords (40 bytes) of message.
    x[0] = _mm512_maskz_loadu_epi64(0x1F, (const __m512i *)in);

    // x[1] lane 0: last 8 bytes of message. lane 1: 0x1F domain byte. rest: 0.
    uint64_t tail;
    memcpy(&tail, in + 40, 8);
    // Build x[1] as { tail, 0x1F, 0, 0, 0, 0, 0, 0 }.
    // _mm512_set_epi64 takes args from highest lane to lowest.
    x[1] = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0x000000000000001FULL, tail);

    x[2] = _mm512_setzero_si512();

    // x[3] lane 1: 0x8000000000000000. lane 0 and 2..4: 0.
    x[3] = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0x8000000000000000ULL, 0);

    x[4] = _mm512_setzero_si512();

    keccakF(x, 24);

    // Squeeze 48 bytes: 5 qwords from x[0] + 1 qword from x[1] lane 0.
    _mm512_mask_storeu_epi64((__m512i *)out, 0x1F, x[0]);
    uint64_t out_tail = _mm_cvtsi128_si64(_mm512_castsi512_si128(x[1]));
    memcpy(out + 40, &out_tail, 8);
}

// lvl5
static inline void shake256_oneblock_64(uint8_t *out, const uint8_t *in) {
    __m512i x[5];

    // 64 bytes = 8 full qwords. x[0] = qwords 0..4 (40 bytes), x[1] lanes 0..2 = qwords 5..7 (24 bytes).
    // Domain 0x1F at byte 64 = qword 8 = x[1] lane 3.
    x[0] = _mm512_maskz_loadu_epi64(0x1F, (const __m512i *)in);
    uint64_t tail[3];
    memcpy(tail, in + 40, 24);
    x[1] = _mm512_set_epi64(0, 0, 0, 0, 0x1F,
                            tail[2], tail[1], tail[0]);
    x[2] = _mm512_setzero_si512();
    x[3] = _mm512_set_epi64(0, 0, 0, 0, 0, 0, 0x8000000000000000ULL, 0);
    x[4] = _mm512_setzero_si512();

    keccakF(x, 24);

    // Squeeze 64 bytes: x[0] lanes 0..4 + x[1] lanes 0..2.
    _mm512_mask_storeu_epi64((__m512i *)out,        0x1F, x[0]);
    _mm512_mask_storeu_epi64((__m512i *)(out + 40), 0x07, x[1]);
}

#if SECURITY_BITS == 128
  #define shake256_oneblock shake256_oneblock_32
#elif SECURITY_BITS == 192
  #define shake256_oneblock shake256_oneblock_48
#elif SECURITY_BITS == 256
  #define shake256_oneblock shake256_oneblock_64
#endif

#endif
