#include "sqi_verif_util.h"
#include "sqi_encoded_sizes.h"
#include "sqi_ec_params.h"
#include "fips202.h"
#include "mp.h"
#include "tutil.h"

// compute the challenge as the hash of the message, the public key and the
// Montgomery coefficient of the commitment curve
void hash_to_challenge(
    scalar_t* scalar,
    const byte_t* pk_com_enc,
    const unsigned char* message,
    size_t length
)
{
    // We hash H("SQI" + pk + A_com + message). The message is absorbed
    // directly into SHAKE to avoid copying it into a buffer.
    shake256incctx ctx;

    memset(*scalar, 0, NWORDS_ORDER * sizeof(digit_t));

    shake256_inc_init(&ctx);
    shake256_inc_absorb(&ctx, (const uint8_t *)"SQI", 3);
    shake256_inc_absorb(&ctx, pk_com_enc, PUBLICKEY_BYTES + FP2_ENCODED_BYTES);
    shake256_inc_absorb(&ctx, message, length);
    shake256_inc_finalize(&ctx);
    shake256_inc_squeeze((void *)(*scalar), CHALLENGE_BYTES, &ctx);
}
