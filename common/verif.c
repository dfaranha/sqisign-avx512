#include <assert.h>
#include <stdio.h>

#include "verif.h"
#include "mp.h"
#include "sqi_hd.h"
#include "sqi_encoded_sizes.h"
#include "sqi_ec_params.h"
#include "fp_unvec.h"
#include "fp_consts.h"
#include "fp2.h"
#include "ec.h"
#include "isog-1d.h"
#include "theta.h"
#include "sqi_test_extras.h"
#include "utils.h"

#include "sqi_verif_util.h"
#include "mp.h"
#include "tutil.h"

#define NBYTES_FP2 8*2*NWORDS64
#define NBYTES_PK NBYTES_FP2 + 1

// Torsion the challenge and response chains run on; each is fed a kernel with
// HD_extra_torsion more than its length.
#define CHALL_TORSION (SECURITY_BITS + HD_extra_torsion)
#define RESP_TORSION (SQIsign_response_length + HD_extra_torsion)

static void decode_digits(digit_t *x, const byte_t *enc, size_t nbytes, size_t ndigits)
{
    assert(nbytes <= ndigits * sizeof(digit_t));
    memcpy((byte_t *)x, enc, nbytes);
    memset((byte_t *)x + nbytes, 0, ndigits * sizeof(digit_t) - nbytes);
}

static void public_key_from_bytes_2x2x2(public_key_2x2x2_t* pk, const byte_t* enc)
{
    uint64_t A_unvec[4*NWORDS64];

    memcpy((byte_t *) A_unvec, enc, NBYTES_FP2);
    enc += NBYTES_FP2;

    memset((byte_t *) (A_unvec + 2*NWORDS64), 0, NBYTES_FP2);
    A_unvec[2*NWORDS64] = 1; // C = 1
    fp_64_to_limbs_4x2(pk->A, A_unvec);
    fp_tomont_4x2(pk->A, pk->A);

    ec_normA_to_normA24_1x2x2x2(pk->A24, pk->A);

    pk->hint = *enc;
}

static void public_key_to_bytes_2x2x2(byte_t *enc, public_key_2x2x2_t *pk)
{
    __m512i A[N4x2];
    uint64_t A_unvec[4*NWORDS64];

    fp2_frommont_2x2x2(A, pk->A);
    fp_limbs_to_64_4x2(A_unvec, A);
    memcpy(enc, (byte_t *) A_unvec, NBYTES_FP2);
    enc += NBYTES_FP2;

    *enc = pk->hint;
}

// A_aux, four basis change coefficients, the challenge scalar, two basis hints.
static void signature_from_bytes_2x2x2(signature_2x2x2_t* sig, const byte_t* enc)
{
    uint64_t A_unvec[4*NWORDS64];

    memcpy((byte_t *) A_unvec, enc, NBYTES_FP2);
    enc += NBYTES_FP2;

    memset((byte_t *) (A_unvec + 2*NWORDS64), 0, NBYTES_FP2);
    A_unvec[2*NWORDS64] = 1; // C = 1
    fp_64_to_limbs_4x2(sig->A_aux, A_unvec);
    fp_tomont_4x2(sig->A_aux, sig->A_aux);

    ec_normA_to_normA24_1x2x2x2(sig->A24_aux, sig->A_aux);

    decode_digits(sig->mat_Bchall_can_to_B_chall[0][0], enc, RESPONSE_BYTES, NWORDS_ORDER);
    enc += RESPONSE_BYTES;
    decode_digits(sig->mat_Bchall_can_to_B_chall[0][1], enc, RESPONSE_BYTES, NWORDS_ORDER);
    enc += RESPONSE_BYTES;
    decode_digits(sig->mat_Bchall_can_to_B_chall[1][0], enc, RESPONSE_BYTES, NWORDS_ORDER);
    enc += RESPONSE_BYTES;
    decode_digits(sig->mat_Bchall_can_to_B_chall[1][1], enc, RESPONSE_BYTES, NWORDS_ORDER);
    enc += RESPONSE_BYTES;

    decode_digits(sig->chall_coeff, enc, CHALLENGE_BYTES, NWORDS_ORDER);
    enc += CHALLENGE_BYTES;

    sig->hint_aux = *enc++;
    sig->hint_chall = *enc++;
}

static void A_com_to_bytes_2x2x2(byte_t *enc, __m512i* A_com)
{
    __m512i A[N4x2];
    uint64_t unvec[4*NWORDS64];

    fp2_frommont_2x2x2(A, A_com);
    fp_limbs_to_64_4x2(unvec, A);
    memcpy(enc, (byte_t *) unvec, NBYTES_FP2);
}

static void get_hash_bytes(byte_t *enc, public_key_2x2x2_t* pk, __m512i* A_com)
{
    public_key_to_bytes_2x2x2(enc, pk);
    A_com_to_bytes_2x2x2(enc + NBYTES_PK, A_com);
}

// Check that the basis change matrix elements are canonical
// representatives modulo 2^(SQIsign_response_length + 2).
static bool check_canonical_basis_change_matrix(const signature_2x2x2_t *sig)
{
    // This works as long as all values in sig->mat_Bchall_can_to_B_chall are
    // positive integers.
    bool ret = true;
    scalar_t aux;

    memset(aux, 0, NWORDS_ORDER * sizeof(digit_t));
    aux[0] = 0x1;
    multiple_mp_shiftl(aux, RESP_TORSION, NWORDS_ORDER);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (mp_compare(aux, sig->mat_Bchall_can_to_B_chall[i][j], NWORDS_ORDER) <= 0) {
                ret = false;
            }
        }
    }

    return ret;
}

// Challenge isogeny with kernel [4](P + [chall_coeff]Q). The chain leaves a point
// of order four above the codomain, which normalizes the challenge curve.
static bool compute_challenge_verify(__m512i* res_A24_chall, __m512i* res_A_chall, const signature_2x2x2_t *sig, public_key_2x2x2_t* pk)
{
    __m512i ker[N4x2], A24[N4x2], A[N4x2], bas_Epk[3*N4x2], four_torsion[N4x2];

    fp2_copy_2x2x2(A, pk->A);
    fp2_copy_2x2x2(A24, pk->A24);

    // Compute basis for E_pk from hint
    ec_curve_to_basis_from_hint_SQI_1x2x2x2(bas_Epk, A, pk->hint, CHALL_TORSION);

    if (!ec_ladder_3pt_normA_1x2x2x2(ker, A, bas_Epk, sig->chall_coeff)) {
        return false;
    }

    if (!isog_chain_1d_1x2x2x2(A24, A24, ker, SECURITY_BITS, NULL, 0, true, four_torsion)) {
        return false;
    }

    if (!ec_normalized_A_from_pair_2x2x2(res_A_chall, four_torsion)) {
        return false;
    }
    ec_normA_to_normA24_1x2x2x2(res_A24_chall, res_A_chall);

    return true;
}

// SQIsign verification
bool protocols_verify(signature_2x2x2_t* sig, public_key_2x2x2_t* pk, const unsigned char* m, size_t l)
{
    __m512i A24_chall[N4x2], A_chall[N4x2], B_chall_can[3*N4x2], B_aux_can[3*N4x2], A_com[N4x2], A_tmp[N4x2];
    __m512i A_34[NUM_LIMBS];
    __m512i ker_chall[2*N4x2];

    if (!check_canonical_basis_change_matrix(sig))
        return false;

    if (!ec_curve_verify_normA_pair_2x2x2(pk->A, sig->A_aux)) {
        return false;
    }

    // computation of the challenge
    if (!compute_challenge_verify(A24_chall, A_chall, sig, pk)) {
        return false;
    }

    // Canonical bases, built at the order the (2,2)-chain needs.
    ec_curve_to_basis_from_hint_2x2x2x2(B_chall_can, B_aux_can, A_chall, sig->A_aux, sig->hint_chall, sig->hint_aux, RESP_TORSION);

#ifndef NDEBUG
    if (!test_basis_order_twof_1x2x2x2(B_chall_can, A24_chall, RESP_TORSION)) {
        printf("canonical basis has wrong order, expect something to fail\n");
        return false;
    }
#endif

    // Both curves are already normalized, this normalizes x(P) on each of them
    theta_normalize_chain_input(A24_chall, sig->A24_aux, B_chall_can, B_aux_can);
    ec_normA24_to_normA_1x2x2x2(A_chall, A24_chall);

    // Apply the change of basis on the challenge curve. The gluing is x-only,
    // so this is a biscalar ladder and no basis is ever lifted.
    if (!ec_biscalar_mul_pair_normA_1x2x2x2(ker_chall, ker_chall + N4x2,
                                            &sig->mat_Bchall_can_to_B_chall,
                                            A_chall, A24_chall, B_chall_can))
        return false;

    // Reject if the signature did not give an isogeny between elliptic products.
    if (!theta_isogeny_chain_bary_vec(A_34, NULL, ker_chall, B_aux_can, A24_chall, sig->A24_aux,
                                      NULL, SQIsign_response_length, 0, false, true))
        return false;

    // Always the first split curve, by our (2^n, 2^n)-isogeny formulae.
    fp2_4x2x1_to_2x2x2(A_com, A_tmp, A_34);

    // recomputing the challenge vector
    scalar_t chk_chall;
    byte_t hash_bytes[NBYTES_PK + NBYTES_FP2];
    get_hash_bytes(hash_bytes, pk, A_com);
    hash_to_challenge(&chk_chall, hash_bytes, m, l);

    // performing the final check
    return mp_compare(sig->chall_coeff, chk_chall, NWORDS_ORDER) == 0;
}

bool verify(const unsigned char *m,
    unsigned long long mlen,
    const unsigned char *sig,
    unsigned long long siglen,
    const unsigned char *pk
)
{
    public_key_2x2x2_t pkt;
    signature_2x2x2_t sigt;

    public_key_from_bytes_2x2x2(&pkt, pk);
    signature_from_bytes_2x2x2(&sigt, sig);

    return protocols_verify(&sigt, &pkt, m, mlen);
}
