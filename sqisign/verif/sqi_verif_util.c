#include "sqi_verif_util.h"
#include "sqi_encoded_sizes_lvl3.h"
#include "sqi_ec_params_lvl3.h"
#include "fips202.h"
#include "mp.h"
#include "tutil.h"


// compute the challenge as the hash of the message and the commitment curve and public key
void hash_to_challenge(
    scalar_t* scalar,
    const byte_t* pk_com_enc,
    const unsigned char* message,
    size_t length
)
{
    // We hash H(pk + j(E_com) + message)
    // First we encode the public key to bytes and append the j-invariant of
    // the commitment curve. We append the message later directly into SHAKE
    // to avoid having to allocate and copying the message into the buffer. 
    {
        // The type scalar_t represents an element of GF(p), which is about
        // 2*lambda bits, where lambda = 128, 192 or 256, according to the
        // security level. Thus, the variable scalar should have enough memory
        // for the values produced by SHAKE256 in the intermediate iterations.

        shake256incctx ctx;

        size_t hash_bytes = ((2 * SECURITY_BITS) + 7) / 8;
        size_t limbs = (hash_bytes + sizeof(digit_t) - 1) / sizeof(digit_t);
        size_t bits = (2 * SECURITY_BITS) % RADIX;
        digit_t mask = ((digit_t)-1) >> ((RADIX - bits) % RADIX);

        shake256_inc_init(&ctx);
        shake256_inc_absorb(&ctx, pk_com_enc, PUBLICKEY_BYTES + FP2_ENCODED_BYTES);
        shake256_inc_absorb(&ctx, message, length);
        shake256_inc_finalize(&ctx);
        shake256_inc_squeeze((void *)(*scalar), hash_bytes, &ctx);
        (*scalar)[limbs - 1] &= mask;
        for (int i = 2; i < HASH_ITERATIONS; i++) {
            shake256_inc_init(&ctx);
            shake256_inc_absorb(&ctx, (void *)(*scalar), hash_bytes);
            shake256_inc_finalize(&ctx);
            shake256_inc_squeeze((void *)(*scalar), hash_bytes, &ctx);
            (*scalar)[limbs - 1] &= mask;
        }
        shake256_inc_init(&ctx);
        shake256_inc_absorb(&ctx, (void *)(*scalar), hash_bytes);
        shake256_inc_finalize(&ctx);

        hash_bytes = ((TORSION_EVEN_POWER - SQIsign_response_length) + 7) / 8;
        limbs = (hash_bytes + sizeof(digit_t) - 1) / sizeof(digit_t);
        bits = (TORSION_EVEN_POWER - SQIsign_response_length) % RADIX;
        mask = ((digit_t)-1) >> ((RADIX - bits) % RADIX);

        memset(*scalar, 0, NWORDS_ORDER * sizeof(digit_t));
        shake256_inc_squeeze((void *)(*scalar), hash_bytes, &ctx);
        (*scalar)[limbs - 1] &= mask;

        mp_mod_2exp(*scalar, SECURITY_BITS, NWORDS_ORDER);
    }
}