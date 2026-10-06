#ifndef SQI_ENCODED_SIZES_H
#define SQI_ENCODED_SIZES_H

// SQIsign v3, NIST level 1: p = 3 * 2^324 - 1
#define SQIsign_PRIME_NAME "p324_3"

#define SECURITY_BITS 128
#define SQIsign_response_length 196
#define FP_ENCODED_BYTES 41
#define FP2_ENCODED_BYTES 82
#define EC_CURVE_ENCODED_BYTES 82
#define EC_POINT_ENCODED_BYTES 82
#define EC_BASIS_ENCODED_BYTES 246
#define PUBLICKEY_BYTES 83
#define SECRETKEY_BYTES 270
#define SIGNATURE_BYTES 200

// Encoded length of the challenge scalar and of each of the four
// basis-change coefficients carried by a signature.
#define CHALLENGE_BYTES (SECURITY_BITS / 8)
#define RESPONSE_BYTES ((SQIsign_response_length + 9) / 8)

#endif
