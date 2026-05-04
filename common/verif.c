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
}

static void j_inv_to_bytes_2x2x2(byte_t *enc, __m512i* j_inv)
{
    __m512i j[N4x2];
    uint64_t unvec[4*NWORDS64];

    fp2_frommont_2x2x2(j, j_inv);
    fp_limbs_to_64_4x2(unvec, j);
    memcpy(enc, (byte_t *) unvec, NBYTES_FP2);
}

static void get_hash_bytes(byte_t *enc, public_key_2x2x2_t* pk, __m512i* j_com)
{
    public_key_to_bytes_2x2x2(enc, pk);
    j_inv_to_bytes_2x2x2(enc + NBYTES_PK, j_com);
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
    multiple_mp_shiftl(aux, SQIsign_response_length + HD_extra_torsion - (int)sig->backtracking, NWORDS_ORDER);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            if (mp_compare(aux, sig->mat_Bchall_can_to_B_chall[i][j], NWORDS_ORDER) <= 0) {
                ret = false;
            }
        }
    }

    return ret;
}

// Compute the 2^n isogeny from the signature with kernel
// P + [chall_coeff]Q and store the codomain in E_chall
static bool compute_challenge_verify(__m512i* res_A24_chall, __m512i* res_A_chall, const signature_2x2x2_t *sig, public_key_2x2x2_t* pk)
{
    __m512i ker[N4x2], A24[N4x2], A[N4x2], bas_Epk[3*N4x2];
    // length of 2^n isogeny
    int len = TORSION_EVEN_POWER - sig->backtracking;

    fp2_copy_2x2x2(A, pk->A);
    fp2_copy_2x2x2(A24, pk->A24);

    // Compute basis for E_pk from hint
    ec_curve_to_basis_from_hint_SQI_1x2x2x2(bas_Epk, A, pk->hint);

    if (!ec_ladder_3pt_normA_1x2x2x2(ker, A, bas_Epk, sig->chall_coeff)) {
        return false;
    }
    ec_double_iter_1x2x2x2(ker, ker, A24, sig->backtracking);

    if (!isog_chain_1d_1x2x2x2(A24, A24, ker, len, NULL, 0, false)) {
        return false;
    }

    ec_normalize_curve_2x2x2(res_A24_chall, res_A_chall, A24);

    return true;
}

// same as matrix_application_even_basis() in id2iso.c, with some modifications:
// - this version works with a matrix of scalars (not ibz_t).
// - reduction modulo 2^f of matrix elements is removed here, because it is
//   assumed that the elements are already cannonical representatives modulo
//   2^f; this is ensured by calling check_canonical_basis_change_matrix() at
//   the beginning of protocols_verify().
static bool matrix_scalar_application_even_basis(__m512i* basis, __m512i* A24, __m512i* A, scalar_mtx_2x2_t *mat, int f)
{
    __m512i res_P[N4x2], res_Q[N4x2];

    scalar_t scalar0, scalar1;
    memset(scalar0, 0, NWORDS_ORDER * sizeof(digit_t));
    memset(scalar1, 0, NWORDS_ORDER * sizeof(digit_t));

    // For a matrix [[a, c], [b, d]] we compute:
    // first basis element R = [a]P + [b]Q
    // second basis element S = [c]P + [d]Q
    if(!ec_biscalar_mul_pair_normA_1x2x2x2(res_P, res_Q, mat, A, A24, basis)) // can't overwrite basis yet
        return false;

    // Their difference R - S = [a - c]P + [b - d]Q
    mp_sub(scalar0, (*mat)[0][0], (*mat)[0][1], NWORDS_ORDER);
    mp_mod_2exp(scalar0, f, NWORDS_ORDER);
    mp_sub(scalar1, (*mat)[1][0], (*mat)[1][1], NWORDS_ORDER);
    mp_mod_2exp(scalar1, f, NWORDS_ORDER);

    bool success = ec_biscalar_mul_normA_1x2x2x2(basis + 2*N4x2, scalar0, scalar1, A, A24, basis);
    fp2_copy_2x2x2(basis, res_P);
    fp2_copy_2x2x2(basis + N4x2, res_Q);
    return success;
}

// Compute the bases for the challenge and auxillary curve from
// the canonical bases. Challenge basis is reconstructed from the
// compressed scalars within the challenge.
static bool challenge_and_aux_basis_verify(
    __m512i* B_chall_can,
    __m512i* B_aux_can,
    __m512i* A24_chall,
    __m512i* A_chall,
    signature_2x2x2_t *sig,
    const int pow_dim2_deg_resp
)
{
    // recovering the canonical basis as TORSION_EVEN_POWER for consistency with signing
    ec_curve_to_basis_from_hint_2x2x2x2(B_chall_can, B_aux_can, A_chall, sig->A_aux, sig->hint_chall, sig->hint_aux);
    
    int n_chall = TORSION_EVEN_POWER - pow_dim2_deg_resp - HD_extra_torsion - sig->two_resp_length;
    int n_aux = TORSION_EVEN_POWER - pow_dim2_deg_resp - HD_extra_torsion;

    ec_double_basis_pair_1x2x2x2(B_aux_can, B_chall_can, B_aux_can, B_chall_can, sig->A24_aux, A24_chall, n_aux, n_chall);

#ifndef NDEBUG
    if (!test_basis_order_twof_1x2x2x2(B_chall_can, A24_chall, HD_extra_torsion + pow_dim2_deg_resp + sig->two_resp_length)) {
        printf("canonical basis has wrong order, expect something to fail\n");
        return false;
    }
#endif

    // applying the change matrix on the basis of E_chall
    return matrix_scalar_application_even_basis(B_chall_can,
                                                A24_chall,
                                                A_chall,
                                                &sig->mat_Bchall_can_to_B_chall,
                                                pow_dim2_deg_resp + HD_extra_torsion + sig->two_resp_length);
}

// When two_resp_length is non-zero, we must compute a small 2^n-isogeny
// updating E_chall as the codomain as well as push the basis on E_chall
// through this isogeny
static bool two_response_isogeny_verify(__m512i* A24_chall, __m512i* B_chall_can, const signature_2x2x2_t *sig, int pow_dim2_deg_resp)
{
    __m512i ker[N4x2];

    // choosing the right point for the small two_isogenies
    if (mp_is_even(sig->mat_Bchall_can_to_B_chall[0][0], NWORDS_ORDER) &&
        mp_is_even(sig->mat_Bchall_can_to_B_chall[1][0], NWORDS_ORDER)) {
        fp2_copy_2x2x2(ker, B_chall_can + N4x2); // ker = Q
    } else {
        fp2_copy_2x2x2(ker, B_chall_can); // ker = P
    }

    ec_double_iter_1x2x2x2(ker, ker, A24_chall, pow_dim2_deg_resp + HD_extra_torsion);

#ifndef NDEBUG
    if (!test_point_order_twof_1x2x2x2(ker, A24_chall, sig->two_resp_length)) {
        printf("kernel does not have order 2^(two_resp_length)\n");
        return false;
    }
#endif

    if (!isog_chain_1d_1x2x2x2(A24_chall, A24_chall, ker, sig->two_resp_length, B_chall_can, 3, true)) {
        return false;
    }

#ifndef NDEBUG
    if (!test_point_order_twof_1x2x2x2(B_chall_can, A24_chall, HD_extra_torsion + pow_dim2_deg_resp)) {
        printf("points[0] does not have order 2^(HD_extra_torsion + pow_dim2_deg_resp)\n");
        return false;
    }
    if (!test_point_order_twof_1x2x2x2(B_chall_can + N4x2, A24_chall, HD_extra_torsion + pow_dim2_deg_resp)) {
        printf("points[1] does not have order 2^(HD_extra_torsion + pow_dim2_deg_resp)\n");
        return false;
    }
    if (!test_point_order_twof_1x2x2x2(B_chall_can + 2*N4x2, A24_chall, HD_extra_torsion + pow_dim2_deg_resp)) {
        printf("points[2] does not have order 2^(HD_extra_torsion + pow_dim2_deg_resp)\n");
        return false;
    }
#endif
    return true;
}

// The commitment curve can be recovered from the codomain of the 2D
// isogeny built from the bases computed during verification.
static bool compute_commitment_curve_verify(
    __m512i* A_com,
    __m512i* B_chall_can,
    __m512i* B_aux_can,
    __m512i* A24_chall,
    signature_2x2x2_t* sig,
    int pow_dim2_deg_resp
)
{
    __m512i A_chall[N4x2], A_34[NUM_LIMBS];
    bool codomain_splits;

#ifndef NDEBUG

    // Check all the points are the correct order
    if (!test_basis_order_twof_1x2x2x2(B_chall_can, A24_chall, HD_extra_torsion + pow_dim2_deg_resp)) {
        printf("B_chall_can does not have order 2^(HD_extra_torsion + pow_dim2_deg_resp)\n");
        return false;
    }

    if (!test_basis_order_twof_1x2x2x2(B_aux_can, sig->A24_aux, HD_extra_torsion + pow_dim2_deg_resp)) {
        printf("B_aux_can does not have order 2^(HD_extra_torsion + pow_dim2_deg_resp)\n");
        return false;
    }
#endif
    // slightly redundant, A24_aux is already normalized
    theta_normalize_chain_input(A24_chall, sig->A24_aux, B_chall_can, B_aux_can);
    ec_normA24_to_normA_1x2x2x2(A_chall, A24_chall);
    
    // handling the special case where we don't need to perform any dim2 computation
    if (pow_dim2_deg_resp == 0) {
        codomain_splits = true;

        // We still need to check that E_chall is supersingular
        // This assumes that HD_extra_torsion == 2
        if (!test_basis_order_twof_1x2x2x2(B_chall_can, A24_chall, 2)) { // not 4 torsion
            return false;
        }
    } else {
        codomain_splits = theta_isogeny_chain_vec_verif(
            A_34, B_chall_can, B_aux_can, A_chall, sig->A_aux, A24_chall, sig->A24_aux,
            pow_dim2_deg_resp
        );
        // computing the commitment curve
        // its always the first one because of our (2^n,2^n)-isogeny formulae
        fp2_4x2x1_to_2x2x2(A_com, A_34, A_34);
    }

    return codomain_splits;
}

// SQIsign verification
bool protocols_verify(signature_2x2x2_t* sig, public_key_2x2x2_t* pk, const unsigned char* m, size_t l)
{
    __m512i A24_chall[N4x2], A_chall[N4x2], B_chall_can[3*N4x2], B_aux_can[3*N4x2], A_com[N4x2], j_com[N4x2];

    if (!check_canonical_basis_change_matrix(sig))
        return false;

    // Computation of the length of the dim 2 2^n isogeny
    int pow_dim2_deg_resp = SQIsign_response_length - (int)sig->two_resp_length - (int)sig->backtracking;

    // basic sanity test: checking that the response is not too long
    if (pow_dim2_deg_resp < 0)
        return false;
    // The dim 2 isogeny embeds a dim 1 isogeny of odd degree, so it can
    // never be of length 2.
    if (pow_dim2_deg_resp == 1)
        return false;

    if (!ec_curve_verify_normA_pair_2x2x2(pk->A, sig->A_aux)) {
        return false;
    }

    // computation of the challenge
    if (!compute_challenge_verify(A24_chall, A_chall, sig, pk)) {
        return false;
    }

    // Computation of the canonical bases for the challenge and aux curve
    if (!challenge_and_aux_basis_verify(B_chall_can, B_aux_can, A24_chall, A_chall, sig, pow_dim2_deg_resp)) {
        return false;
    }

    // When two_resp_length != 0 we need to compute a second, short 2^r-isogeny
    // After this, A24_chall is no longer normalized
    if (sig->two_resp_length > 0) {
        if (!two_response_isogeny_verify(A24_chall, B_chall_can, sig, pow_dim2_deg_resp)) {
            return false;
        }
    }

    // We can recover the commitment curve with a 2D isogeny
    // The supplied signature did not compute an isogeny between eliptic products
    // and so definitely is an invalid signature.
    if (!compute_commitment_curve_verify(A_com, B_chall_can, B_aux_can, A24_chall, sig, pow_dim2_deg_resp))
        return false;

    // j_com = j(E_com)
    ec_j_inv_2x2x2(j_com, A_com);

    // recomputing the challenge vector
    scalar_t chk_chall;
    byte_t hash_bytes[NBYTES_PK + NBYTES_FP2];
    get_hash_bytes(hash_bytes, pk, j_com);
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
