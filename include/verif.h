#ifndef VERIF_H
#define VERIF_H

#include "ec.h"
#include "fp_consts.h"
#include "intrinsics_avx512.h"
#include "tutil.h"
#include <stdbool.h>

typedef unsigned char byte_t;

typedef struct public_key_2x2x2
{
    __m512i A[N4x2]; // the normalized A-coefficient of the curve
    __m512i A24[N4x2]; // the normalized A24-coeffficient of the curve
    uint8_t hint;
} public_key_2x2x2_t;

typedef struct signature_2x2x2
{
    __m512i A_aux[N4x2]; // the normalized A-coefficient of the auxiliary curve
    __m512i A24_aux[N4x2]; // the normalized A24-coefficient of the auxilliary curve
    uint8_t backtracking;
    uint8_t two_resp_length;
    scalar_mtx_2x2_t mat_Bchall_can_to_B_chall; // the matrix of the desired basis
    scalar_t chall_coeff;
    uint8_t hint_aux;
    uint8_t hint_chall;
} signature_2x2x2_t;

/**
 * @brief Verification
 *
 * @param sig signature
 * @param pk public key
 * @param m message
 * @param l size
 * @returns 1 if the signature verifies, 0 otherwise
 */
bool protocols_verify(
    signature_2x2x2_t* sig,
    public_key_2x2x2_t* pk, 
    const unsigned char* m, 
    size_t l
);

// Decodes the input and calls protocols_verify
bool verify(
    const unsigned char *m,
    unsigned long long mlen,
    const unsigned char *sig,
    unsigned long long siglen,
    const unsigned char *pk
);
#endif
