#ifndef SQI_ENCODED_SIZES_H
#define SQI_ENCODED_SIZES_H

// SQIsign v3, NIST level 5: p = 17 * 2^664 - 1
#define SQIsign_PRIME_NAME "p664_17"

#define SECURITY_BITS 256
#define SQIsign_response_length 400
#define FP_ENCODED_BYTES 84
#define FP2_ENCODED_BYTES 168
#define EC_CURVE_ENCODED_BYTES 168
#define EC_POINT_ENCODED_BYTES 168
#define EC_BASIS_ENCODED_BYTES 504
#define PUBLICKEY_BYTES 169
#define SECRETKEY_BYTES 549
#define SIGNATURE_BYTES 406

// Encoded length of the challenge scalar and of each of the four
// basis-change coefficients carried by a signature.
#define CHALLENGE_BYTES (SECURITY_BITS / 8)
#define RESPONSE_BYTES ((SQIsign_response_length + 9) / 8)

#endif
