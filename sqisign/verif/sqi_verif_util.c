#include "sqi_verif_util.h"
#include "sqi_encoded_sizes_lvl3.h"
#include "sqi_ec_params_lvl3.h"
#include "fips202.h"
#include "mp.h"
#include "tutil.h"
#include <assert.h>

typedef unsigned char byte_t;

void
public_key_init(public_key_t *pk)
{
    ec_curve_init(&pk->curve);
}

// compute the challenge as the hash of the message and the commitment curve and public key
void
hash_to_challenge(scalar_t *scalar,
                  const public_key_t *pk,
                  const ec_curve_t *com_curve,
                  const unsigned char *message,
                  size_t length)
{
    // We hash H(pk + j(E_com) + message)
    // First we encode the public key to bytes and append the j-invariant of
    // the commitment curve. We append the message later directly into SHAKE
    // to avoid having to allocate and copying the message into the buffer. 
    unsigned char buf[PUBLICKEY_BYTES + FP2_ENCODED_BYTES];

    // First encode the public key as bytes
    public_key_to_bytes(buf, pk);
    {
        // Compute the j-invariant of the commitment and include bytes into buffer 
        fp2_t j_com;
        ec_j_inv(&j_com, com_curve);
        fp2_encode(buf + PUBLICKEY_BYTES, &j_com);
    }

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
#ifdef TARGET_BIG_ENDIAN
        mask = BSWAP_DIGIT(mask);
#endif

        shake256_inc_init(&ctx);
        shake256_inc_absorb(&ctx, buf, sizeof(buf));
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
#ifdef TARGET_BIG_ENDIAN
        mask = BSWAP_DIGIT(mask);
#endif

        memset(*scalar, 0, NWORDS_ORDER * sizeof(digit_t));
        shake256_inc_squeeze((void *)(*scalar), hash_bytes, &ctx);
        (*scalar)[limbs - 1] &= mask;

#ifdef TARGET_BIG_ENDIAN
        for (int i = 0; i < NWORDS_ORDER; i++)
            (*scalar)[i] = BSWAP_DIGIT((*scalar)[i]);
#endif

        mp_mod_2exp(*scalar, SECURITY_BITS, NWORDS_ORDER);
    }
}

// digits

static void
encode_digits(byte_t *enc, const digit_t *x, size_t nbytes)
{
#ifdef TARGET_BIG_ENDIAN
    const size_t ndigits = nbytes / sizeof(digit_t);
    const size_t rem = nbytes % sizeof(digit_t);

    for (size_t i = 0; i < ndigits; i++)
        ((digit_t *)enc)[i] = BSWAP_DIGIT(x[i]);
    if (rem) {
        digit_t ld = BSWAP_DIGIT(x[ndigits]);
        memcpy(enc + ndigits * sizeof(digit_t), (byte_t *)&ld, rem);
    }
#else
    memcpy(enc, (const byte_t *)x, nbytes);
#endif
}

static void
decode_digits(digit_t *x, const byte_t *enc, size_t nbytes, size_t ndigits)
{
    assert(nbytes <= ndigits * sizeof(digit_t));
    memcpy((byte_t *)x, enc, nbytes);
    memset((byte_t *)x + nbytes, 0, ndigits * sizeof(digit_t) - nbytes);

#ifdef TARGET_BIG_ENDIAN
    for (size_t i = 0; i < ndigits; i++)
        x[i] = BSWAP_DIGIT(x[i]);
#endif
}

// fp2_t

static byte_t *
fp2_to_bytes(byte_t *enc, const fp2_t *x)
{
    fp2_encode(enc, x);
    return enc + FP2_ENCODED_BYTES;
}

static const byte_t *
fp2_from_bytes(fp2_t *x, const byte_t *enc)
{
    fp2_decode(x, enc);
    return enc + FP2_ENCODED_BYTES;
}

// curves and points

static byte_t *
proj_to_bytes(byte_t *enc, const fp2_t *x, const fp2_t *z)
{
    assert(!fp2_is_zero(z));
    fp2_t tmp = *z;
    fp2_inv(&tmp);
#ifndef NDEBUG
    {
        fp2_t chk;
        fp2_mul(&chk, z, &tmp);
        fp2_t one;
        fp2_set_one(&one);
        assert(fp2_is_equal(&chk, &one));
    }
#endif
    fp2_mul(&tmp, x, &tmp);
    enc = fp2_to_bytes(enc, &tmp);
    return enc;
}

static const byte_t *
proj_from_bytes(fp2_t *x, fp2_t *z, const byte_t *enc)
{
    enc = fp2_from_bytes(x, enc);
    fp2_set_one(z);
    return enc;
}

static byte_t *
ec_curve_to_bytes(byte_t *enc, const ec_curve_t *curve)
{
    return proj_to_bytes(enc, &curve->A, &curve->C);
}

static const byte_t *
ec_curve_from_bytes(ec_curve_t *curve, const byte_t *enc)
{
    memset(curve, 0, sizeof(*curve));
    return proj_from_bytes(&curve->A, &curve->C, enc);
}

static byte_t *
ec_point_to_bytes(byte_t *enc, const ec_point_t *point)
{
    return proj_to_bytes(enc, &point->x, &point->z);
}

static const byte_t *
ec_point_from_bytes(ec_point_t *point, const byte_t *enc)
{
    return proj_from_bytes(&point->x, &point->z, enc);
}

static byte_t *
ec_basis_to_bytes(byte_t *enc, const ec_basis_t *basis)
{
    enc = ec_point_to_bytes(enc, &basis->P);
    enc = ec_point_to_bytes(enc, &basis->Q);
    enc = ec_point_to_bytes(enc, &basis->PmQ);
    return enc;
}

static const byte_t *
ec_basis_from_bytes(ec_basis_t *basis, const byte_t *enc)
{
    enc = ec_point_from_bytes(&basis->P, enc);
    enc = ec_point_from_bytes(&basis->Q, enc);
    enc = ec_point_from_bytes(&basis->PmQ, enc);
    return enc;
}

// public API

byte_t *
public_key_to_bytes(byte_t *enc, const public_key_t *pk)
{
#ifndef NDEBUG
    const byte_t *const start = enc;
#endif
    enc = ec_curve_to_bytes(enc, &pk->curve);
    *enc++ = pk->hint_pk;
    assert(enc - start == PUBLICKEY_BYTES);
    return enc;
}

const byte_t *
public_key_from_bytes(public_key_t *pk, const byte_t *enc)
{
#ifndef NDEBUG
    const byte_t *const start = enc;
#endif
    enc = ec_curve_from_bytes(&pk->curve, enc);
    pk->hint_pk = *enc++;
    assert(enc - start == PUBLICKEY_BYTES);
    return enc;
}

void
signature_to_bytes(byte_t *enc, const signature_t *sig)
{
#ifndef NDEBUG
    byte_t *const start = enc;
#endif

    enc = fp2_to_bytes(enc, &sig->E_aux_A);

    *enc++ = sig->backtracking;
    *enc++ = sig->two_resp_length;

    size_t nbytes = (SQIsign_response_length + 9) / 8;
    encode_digits(enc, sig->mat_Bchall_can_to_B_chall[0][0], nbytes);
    enc += nbytes;
    encode_digits(enc, sig->mat_Bchall_can_to_B_chall[0][1], nbytes);
    enc += nbytes;
    encode_digits(enc, sig->mat_Bchall_can_to_B_chall[1][0], nbytes);
    enc += nbytes;
    encode_digits(enc, sig->mat_Bchall_can_to_B_chall[1][1], nbytes);
    enc += nbytes;

    nbytes = SECURITY_BITS / 8;
    encode_digits(enc, sig->chall_coeff, nbytes);
    enc += nbytes;

    *enc++ = sig->hint_aux;
    *enc++ = sig->hint_chall;

    assert(enc - start == SIGNATURE_BYTES);
}

void
signature_from_bytes(signature_t *sig, const byte_t *enc)
{
#ifndef NDEBUG
    const byte_t *const start = enc;
#endif

    enc = fp2_from_bytes(&sig->E_aux_A, enc);

    sig->backtracking = *enc++;
    sig->two_resp_length = *enc++;

    size_t nbytes = (SQIsign_response_length + 9) / 8;
    decode_digits(sig->mat_Bchall_can_to_B_chall[0][0], enc, nbytes, NWORDS_ORDER);
    enc += nbytes;
    decode_digits(sig->mat_Bchall_can_to_B_chall[0][1], enc, nbytes, NWORDS_ORDER);
    enc += nbytes;
    decode_digits(sig->mat_Bchall_can_to_B_chall[1][0], enc, nbytes, NWORDS_ORDER);
    enc += nbytes;
    decode_digits(sig->mat_Bchall_can_to_B_chall[1][1], enc, nbytes, NWORDS_ORDER);
    enc += nbytes;

    nbytes = SECURITY_BITS / 8;
    decode_digits(sig->chall_coeff, enc, nbytes, NWORDS_ORDER);
    enc += nbytes;

    sig->hint_aux = *enc++;
    sig->hint_chall = *enc++;

    assert(enc - start == SIGNATURE_BYTES);
}
