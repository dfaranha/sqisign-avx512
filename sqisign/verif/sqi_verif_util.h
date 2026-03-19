#ifndef SQI_VERIF_H
#define SQI_VERIF_H

#include "sqi_ec.h"
#include "sqi_fp_lvl3.h"
#include "sqi_fp2.h"
#include "sqi_fp_constants_lvl3.h"

typedef digit_t scalar_t[NWORDS_ORDER];
typedef scalar_t scalar_mtx_2x2_t[2][2];

typedef struct signature
{
    fp2_t E_aux_A; // the Montgomery A-coefficient for the auxiliary curve
    uint8_t backtracking;
    uint8_t two_resp_length;
    scalar_mtx_2x2_t mat_Bchall_can_to_B_chall; // the matrix of the desired basis
    scalar_t chall_coeff;
    uint8_t hint_aux;
    uint8_t hint_chall;
} signature_t;

typedef struct public_key
{
    ec_curve_t curve; // the normalized A-coefficient of the Montgomery curve
    uint8_t hint_pk;
} public_key_t;

/*************************** Functions *****************************/

void public_key_init(public_key_t* pk);

void hash_to_challenge(scalar_t* scalar,
                       const public_key_t* pk,
                       const ec_curve_t* com_curve,
                       const unsigned char* message,
                       size_t length);


/*************************** Encoding *****************************/

/** @defgroup encoding Encoding and decoding functions
 * @{
 */

/**
 * @brief Encodes a signature as a byte array
 *
 * @param enc : Byte array to encode the signature in
 * @param sig : Signature to encode
 */
void signature_to_bytes(unsigned char* enc, const signature_t* sig);

/**
 * @brief Decodes a signature from a byte array
 *
 * @param sig : Structure to decode the signature in
 * @param enc : Byte array to decode
 */
void signature_from_bytes(signature_t* sig, const unsigned char* enc);

/**
 * @brief Encodes a public key as a byte array
 *
 * @param enc : Byte array to encode the public key in
 * @param pk : Public key to encode
 */
unsigned char* public_key_to_bytes(unsigned char* enc, const public_key_t* pk);

/**
 * @brief Decodes a public key from a byte array
 *
 * @param pk : Structure to decode the public key in
 * @param enc : Byte array to decode
 */
const unsigned char* public_key_from_bytes(public_key_t* pk, const unsigned char* enc);

#endif