#ifndef SQI_ENCODED_SIZES_H
#define SQI_ENCODED_SIZES_H

// NOT YET PORTED TO SQIsign v3: these are the v2 parameters. v3 moves this
// level to a different prime, for which no field arithmetic exists here yet.
#define SQIsign_PRIME_NAME "p376_65"

#define SECURITY_BITS 192
#define SQIsign_response_length 192
#define HASH_ITERATIONS 256
#define FP_ENCODED_BYTES 48
#define FP2_ENCODED_BYTES 96
#define EC_CURVE_ENCODED_BYTES 96
#define EC_POINT_ENCODED_BYTES 96
#define EC_BASIS_ENCODED_BYTES 288
#define PUBLICKEY_BYTES 97
#define SECRETKEY_BYTES 529
#define SIGNATURE_BYTES 224

#define CHALLENGE_BYTES (SECURITY_BITS / 8)
#define RESPONSE_BYTES ((SQIsign_response_length + 9) / 8)

#endif