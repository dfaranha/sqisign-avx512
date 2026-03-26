#ifndef VERIF_222_H
#define VERIF_222_H

#include "sqi_verif_util.h"

/**
 * @brief Verification
 *
 * @param sig signature
 * @param pk public key
 * @param m message
 * @param l size
 * @returns 1 if the signature verifies, 0 otherwise
 */
bool protocols_verify_2x2x2(signature_t* sig, const public_key_t* pk, const unsigned char* m, size_t l);

// Decodes the input and calls protocols_verify
bool verify_2x2x2(const unsigned char *m,
               unsigned long long mlen,
               const unsigned char *sig,
               unsigned long long siglen,
               const unsigned char *pk);
#endif
