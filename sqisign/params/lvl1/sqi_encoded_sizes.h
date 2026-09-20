#ifndef SQI_ENCODED_SIZES_H
#define SQI_ENCODED_SIZES_H

// NOT YET PORTED TO SQIsign v3: these are the v2 parameters. v3 moves this
// level to a different prime, for which no field arithmetic exists here yet.
#define SQIsign_PRIME_NAME "p248_5"

#define SECURITY_BITS 128
#define SQIsign_response_length 126
#define HASH_ITERATIONS 64
#define FP_ENCODED_BYTES 32
#define FP2_ENCODED_BYTES 64
#define EC_CURVE_ENCODED_BYTES 64
#define EC_POINT_ENCODED_BYTES 64
#define EC_BASIS_ENCODED_BYTES 192
#define PUBLICKEY_BYTES 65
#define SECRETKEY_BYTES 353
#define SIGNATURE_BYTES 148

#define CHALLENGE_BYTES (SECURITY_BITS / 8)
#define RESPONSE_BYTES ((SQIsign_response_length + 9) / 8)

#endif