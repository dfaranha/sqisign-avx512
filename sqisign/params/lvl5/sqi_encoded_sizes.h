#ifndef SQI_ENCODED_SIZES_H
#define SQI_ENCODED_SIZES_H

// SQIsign v3, NIST level 3: p = 27 * 2^500 - 1
#define SQIsign_PRIME_NAME "p500_27"

#define SECURITY_BITS 192
#define SQIsign_response_length 302
#define FP_ENCODED_BYTES 64
#define FP2_ENCODED_BYTES 128
#define EC_CURVE_ENCODED_BYTES 128
#define EC_POINT_ENCODED_BYTES 128
#define EC_BASIS_ENCODED_BYTES 384
#define PUBLICKEY_BYTES 129
#define SECRETKEY_BYTES 417
#define SIGNATURE_BYTES 306

// Encoded length of the challenge scalar and of each of the four
// basis-change coefficients carried by a signature.
#define CHALLENGE_BYTES (SECURITY_BITS / 8)
#define RESPONSE_BYTES ((SQIsign_response_length + 9) / 8)

#endif
