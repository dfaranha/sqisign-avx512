#ifndef VERIF_H
#define VERIF_H

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
int protocols_verify(signature_t* sig, const public_key_t* pk, const unsigned char* m, size_t l);

#endif
