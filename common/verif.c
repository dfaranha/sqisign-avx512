#include "verif.h"
#include "mp.h"
#include "sqi_hd.h"
#include "sqi_encoded_sizes_lvl3.h"
#include "sqi_ec_params_lvl3.h"
#include <assert.h>
#include "fp_unvec_lvl3.h"
#include "fp_consts.h"
#include "fp2.h"
#include "ec.h"
#include "isog-1d.h"
#include "theta.h"
#include "sqi_test_extras.h"

// #include "bench.h"
// #include <inttypes.h>

// Check that the basis change matrix elements are canonical
// representatives modulo 2^(SQIsign_response_length + 2).
static bool
check_canonical_basis_change_matrix(const signature_t *sig)
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

// TODO: Change return type from int to bool


// Compute the 2^n isogeny from the signature with kernel
// P + [chall_coeff]Q and store the codomain in E_chall
static int
compute_challenge_verify(ec_curve_t *E_chall, const signature_t *sig, const ec_curve_t *Epk, const uint8_t hint_pk)
{
    ec_basis_t bas_Epk_unvec;
    ec_point_t ker_unvec;
    copy_curve(E_chall, Epk);

    // Set domain and length of 2^n isogeny
    int len = TORSION_EVEN_POWER - sig->backtracking;

    // Compute the basis from the supplied hint
    // uint64_t start4 = cpucycles();
    // for (int i = 0; i < 1000; i++) {
    if (!ec_curve_to_basis_2f_from_hint(&bas_Epk_unvec, E_chall, TORSION_EVEN_POWER, hint_pk)) // canonical
        return 0;
    // }
    // uint64_t end4 = cpucycles();
    // printf("basis %" PRIu64 "\n", (end4-start4)/1000);

    // ============== VECTORIZED ================================================================================

    __m512i ker[4], A24[4], A[4], bas_Epk[12];
    // uint64_t start2 = cpucycles();
    // for (int i = 0; i < 1000; i++) {
    ec_point_unvec_to_1x2x2x2(A24, &E_chall->A24);
    ec_point_t AC = {E_chall-> A, E_chall -> C};
    ec_point_unvec_to_1x2x2x2(A, &AC);
    ec_point_unvec_to_1x2x2x2(bas_Epk, &bas_Epk_unvec.P);
    ec_point_unvec_to_1x2x2x2(bas_Epk+4, &bas_Epk_unvec.Q);
    ec_point_unvec_to_1x2x2x2(bas_Epk+8, &bas_Epk_unvec.PmQ);
    // }
    // uint64_t end2 = cpucycles();
    // printf("conv to 1x2x2x2 %" PRIu64 "\n", (end2-start2)/1000);

    
    // recovering the exact challenge
    // uint64_t start1 = cpucycles();
    // for (int i = 0; i < 1000; i++) {

    if (!ec_ladder_3pt_normA_1x2x2x2(ker, A, bas_Epk, sig->chall_coeff)) {
        return 0;
    }

    // }
    // uint64_t end1 = cpucycles();
    // printf("3pt ladder %" PRIu64 "\n", (end1-start1)/1000);

    // Double the kernel until is has the correct order
    // uint64_t start3 = cpucycles();
    // for (int i = 0; i < 1000; i++) {


    ec_double_iter_1x2x2x2(ker, ker, A24, sig->backtracking);


    // }
    // uint64_t end3 = cpucycles();
    // printf("double backtrack %" PRIu64 "\n", (end3-start3)/1000);


    // Compute the codomain
    // __m512i A24_copy[4];
    // uint64_t start = cpucycles();
    // for (int i = 0; i < 1000; i++) {
    if (!isog_chain_1d_1x2x2x2(A24, A24, ker, len, NULL, 0, false)) {
        return 0;
    }
    // }
    // uint64_t end = cpucycles();
    // fp2_copy_2x2x2(A24, A24_copy);
    // printf("1D isog %" PRIu64 "\n", (end-start)/1000);

    ec_point_1x2x2x2_to_unvec(&E_chall->A24, A24);
    A24_to_AC(E_chall, &E_chall->A24);
    E_chall->is_A24_computed_and_normalized = false;

    // ============== END ================================================================================

    return 1;
}

// same as matrix_application_even_basis() in id2iso.c, with some modifications:
// - this version works with a matrix of scalars (not ibz_t).
// - reduction modulo 2^f of matrix elements is removed here, because it is
//   assumed that the elements are already cannonical representatives modulo
//   2^f; this is ensured by calling check_canonical_basis_change_matrix() at
//   the beginning of protocols_verify().
static int
matrix_scalar_application_even_basis(ec_basis_t *bas, const ec_curve_t *E, scalar_mtx_2x2_t *mat, int f)
{

    // ============== VECTORIZED ================================================================================
    __m512i res1[4], res2[4], A24[4], A[4], basis_vec[12];
    ec_point_t EA24, AC = {E->A, E->C};
    copy_point(&EA24, &E->A24);
    ec_point_unvec_to_1x2x2x2(A24, &EA24);
    ec_point_unvec_to_1x2x2x2(A, &AC);
    ec_point_unvec_to_1x2x2x2(basis_vec, &bas->P);
    ec_point_unvec_to_1x2x2x2(basis_vec + 4, &bas->Q);
    ec_point_unvec_to_1x2x2x2(basis_vec + 8, &bas->PmQ);

    scalar_t scalar0, scalar1;
    memset(scalar0, 0, NWORDS_ORDER * sizeof(digit_t));
    memset(scalar1, 0, NWORDS_ORDER * sizeof(digit_t));

    // For a matrix [[a, c], [b, d]] we compute:
    // first basis element R = [a]P + [b]Q
    // second basis element S = [c]P + [d]Q
    if(!ec_biscalar_mul_pair_normA_1x2x2x2(res1, res2, mat, A, A24, basis_vec))
        return 0;
    ec_point_1x2x2x2_to_unvec(&bas->P, res1);
    ec_point_1x2x2x2_to_unvec(&bas->Q, res2);

    // Their difference R - S = [a - c]P + [b - d]Q
    mp_sub(scalar0, (*mat)[0][0], (*mat)[0][1], NWORDS_ORDER);
    mp_mod_2exp(scalar0, f, NWORDS_ORDER);
    mp_sub(scalar1, (*mat)[1][0], (*mat)[1][1], NWORDS_ORDER);
    mp_mod_2exp(scalar1, f, NWORDS_ORDER);

    bool success = ec_biscalar_mul_normA_1x2x2x2(res1, scalar0, scalar1, A, A24, basis_vec);
    ec_point_1x2x2x2_to_unvec(&bas->PmQ, res1);
    return success ? 1 : 0;

    // ============== END ================================================================================
}

// Compute the bases for the challenge and auxillary curve from
// the canonical bases. Challenge basis is reconstructed from the
// compressed scalars within the challenge.
static int
challenge_and_aux_basis_verify(ec_basis_t *B_chall_can_unvec,
                               ec_basis_t *B_aux_can_unvec,
                               ec_curve_t *E_chall,
                               ec_curve_t *E_aux,
                               signature_t *sig,
                               const int pow_dim2_deg_resp)
{

    // recovering the canonical basis as TORSION_EVEN_POWER for consistency with signing
    if (!ec_curve_to_basis_2f_from_hint(B_chall_can_unvec, E_chall, TORSION_EVEN_POWER, sig->hint_chall))
        return 0;

    if (!ec_curve_to_basis_2f_from_hint(B_aux_can_unvec, E_aux, TORSION_EVEN_POWER, sig->hint_aux))
        return 0;

    // TODO: Pair these together

    // setting to the right order

    // We normalize E_chall to use normalized ladders
    // and E_aux for faster basis doubling
    __m512i A24_chall[N4x2], A24_aux[N4x2], A_chall[N4x2], B_chall_can[3*N4x2], B_aux_can[3*N4x2];
    ec_point_unvec_to_1x2x2x2(A24_chall, &E_chall->A24);
    ec_point_unvec_to_1x2x2x2(A24_aux, &E_aux->A24);
    ec_normalize_A24_2x2x2x2(A24_chall, A24_aux, A24_chall, A24_aux);
    ec_normA24_to_normA_1x2x2x2(A_chall, A24_chall);
    ec_point_t AC;
    ec_point_1x2x2x2_to_unvec(&AC, A_chall);
    fp2_copy(&E_chall->A, &AC.x);
    fp2_set_one(&E_chall->C);
    ec_point_1x2x2x2_to_unvec(&E_chall->A24, A24_chall);
    ec_point_1x2x2x2_to_unvec(&E_aux->A24, A24_aux);

    // ec_normalize_curve_and_A24(E_chall); // to use normalized ladders
    // ec_normalize_curve_and_A24(E_aux); // for basis doubling

    // ============== VECTORIZED ================================================================================

    
    ec_point_unvec_to_1x2x2x2(A24_chall, &E_chall->A24);
    ec_point_unvec_to_1x2x2x2(A24_aux, &E_aux->A24);
    ec_point_unvec_to_1x2x2x2(B_chall_can, &B_chall_can_unvec->P);
    ec_point_unvec_to_1x2x2x2(B_chall_can + 4, &B_chall_can_unvec->Q);
    ec_point_unvec_to_1x2x2x2(B_chall_can + 8, &B_chall_can_unvec->PmQ);
    ec_point_unvec_to_1x2x2x2(B_aux_can, &B_aux_can_unvec->P);
    ec_point_unvec_to_1x2x2x2(B_aux_can + 4, &B_aux_can_unvec->Q);
    ec_point_unvec_to_1x2x2x2(B_aux_can + 8, &B_aux_can_unvec->PmQ);

    int n_chall = TORSION_EVEN_POWER - pow_dim2_deg_resp - HD_extra_torsion - sig->two_resp_length;
    int n_aux = TORSION_EVEN_POWER - pow_dim2_deg_resp - HD_extra_torsion;

    ec_double_basis_pair_1x2x2x2(B_aux_can, B_chall_can, B_aux_can, B_chall_can, A24_aux, A24_chall, n_aux, n_chall);

#ifndef NDEBUG
    if (!test_basis_order_twof_1x2x2x2(B_chall_can, A24_chall, HD_extra_torsion + pow_dim2_deg_resp + sig->two_resp_length))
        printf("canonical basis has wrong order, expect something to fail\n");
#endif

    ec_point_1x2x2x2_to_unvec(&B_chall_can_unvec->P, B_chall_can);
    ec_point_1x2x2x2_to_unvec(&B_chall_can_unvec->Q, B_chall_can+4);
    ec_point_1x2x2x2_to_unvec(&B_chall_can_unvec->PmQ, B_chall_can+8);
    ec_point_1x2x2x2_to_unvec(&B_aux_can_unvec->P, B_aux_can);
    ec_point_1x2x2x2_to_unvec(&B_aux_can_unvec->Q, B_aux_can+4);
    ec_point_1x2x2x2_to_unvec(&B_aux_can_unvec->PmQ, B_aux_can+8);

    // ============== END ================================================================================

    // applying the change matrix on the basis of E_chall
    return matrix_scalar_application_even_basis(B_chall_can_unvec,
                                                E_chall,
                                                &sig->mat_Bchall_can_to_B_chall,
                                                pow_dim2_deg_resp + HD_extra_torsion + sig->two_resp_length);
}

// When two_resp_length is non-zero, we must compute a small 2^n-isogeny
// updating E_chall as the codomain as well as push the basis on E_chall
// through this isogeny
static int
two_response_isogeny_verify(ec_curve_t *E_chall, ec_basis_t *B_chall_can, const signature_t *sig, int pow_dim2_deg_resp)
{
    ec_point_t ker_unvec, points_unvec[3];

    // choosing the right point for the small two_isogenies
    if (mp_is_even(sig->mat_Bchall_can_to_B_chall[0][0], NWORDS_ORDER) &&
        mp_is_even(sig->mat_Bchall_can_to_B_chall[1][0], NWORDS_ORDER)) {
        copy_point(&ker_unvec, &B_chall_can->Q);
    } else {
        copy_point(&ker_unvec, &B_chall_can->P);
    }

    // ============== VECTORIZED ================================================================================
    __m512i A24[4], ker[4], points[12];
    ec_point_unvec_to_1x2x2x2(A24, &E_chall->A24);
    ec_point_unvec_to_1x2x2x2(ker, &ker_unvec);
    ec_point_unvec_to_1x2x2x2(points, &B_chall_can->P);
    ec_point_unvec_to_1x2x2x2(points+4, &B_chall_can->Q);
    ec_point_unvec_to_1x2x2x2(points+8, &B_chall_can->PmQ);

    ec_double_iter_1x2x2x2(ker, ker, A24, pow_dim2_deg_resp + HD_extra_torsion);

#ifndef NDEBUG
    if (!test_point_order_twof_1x2x2x2(ker, A24, sig->two_resp_length))
        printf("kernel does not have order 2^(two_resp_length)\n");
#endif

    if (!isog_chain_1d_1x2x2x2(A24, A24, ker, sig->two_resp_length, points, 3, true)) {
        return 0;
    }

#ifndef NDEBUG
    if (!test_point_order_twof_1x2x2x2(points, A24, HD_extra_torsion + pow_dim2_deg_resp))
        printf("points[0] does not have order 2^(HD_extra_torsion + pow_dim2_deg_resp)\n");
    if (!test_point_order_twof_1x2x2x2(points+4, A24, HD_extra_torsion + pow_dim2_deg_resp))
        printf("points[1] does not have order 2^(HD_extra_torsion + pow_dim2_deg_resp)\n");
    if (!test_point_order_twof_1x2x2x2(points+8, A24, HD_extra_torsion + pow_dim2_deg_resp))
        printf("points[2] does not have order 2^(HD_extra_torsion + pow_dim2_deg_resp)\n");
#endif

    ec_point_1x2x2x2_to_unvec(&E_chall->A24, A24);
    A24_to_AC(E_chall, &E_chall->A24);
    E_chall->is_A24_computed_and_normalized = false;
    ec_point_1x2x2x2_to_unvec(points_unvec, points);
    ec_point_1x2x2x2_to_unvec(points_unvec+1, points+4);
    ec_point_1x2x2x2_to_unvec(points_unvec+2, points+8);
    // ============== END ================================================================================

    copy_point(&B_chall_can->P, &points_unvec[0]);
    copy_point(&B_chall_can->Q, &points_unvec[1]);
    copy_point(&B_chall_can->PmQ, &points_unvec[2]);
    return 1;
}

// The commitment curve can be recovered from the codomain of the 2D
// isogeny built from the bases computed during verification.
static int
compute_commitment_curve_verify(ec_curve_t *E_com,
                                ec_basis_t *B_chall_can_unvec,
                                ec_basis_t *B_aux_can_unvec,
                                ec_curve_t *E_chall,
                                ec_curve_t *E_aux,
                                int pow_dim2_deg_resp)

{
    // ============== VECTORIZED ================================================================================
#ifndef NDEBUG
    __m512i A24_chall[4], A24_aux[4], B_chall_can[12], B_aux_can[12];
    ec_point_unvec_to_1x2x2x2(A24_chall, &E_chall->A24);
    ec_point_unvec_to_1x2x2x2(A24_aux, &E_aux->A24);
    ec_point_unvec_to_1x2x2x2(B_chall_can, &B_chall_can_unvec->P);
    ec_point_unvec_to_1x2x2x2(B_chall_can + 4, &B_chall_can_unvec->Q);
    ec_point_unvec_to_1x2x2x2(B_chall_can + 8, &B_chall_can_unvec->PmQ);
    ec_point_unvec_to_1x2x2x2(B_aux_can, &B_aux_can_unvec->P);
    ec_point_unvec_to_1x2x2x2(B_aux_can + 4, &B_aux_can_unvec->Q);
    ec_point_unvec_to_1x2x2x2(B_aux_can + 8, &B_aux_can_unvec->PmQ);


    // Check all the points are the correct order
    if (!test_basis_order_twof_1x2x2x2(B_chall_can, A24_chall, HD_extra_torsion + pow_dim2_deg_resp))
        printf("B_chall_can does not have order 2^(HD_extra_torsion + pow_dim2_deg_resp)\n");

    if (!test_basis_order_twof_1x2x2x2(B_aux_can, A24_aux, HD_extra_torsion + pow_dim2_deg_resp))
        printf("B_aux_can does not have order 2^(HD_extra_torsion + pow_dim2_deg_resp)\n");
#endif
    // ============== END ================================================================================

    // now compute the dim2 isogeny from Echall x E_aux -> E_com x E_aux'
    // of kernel B_chall_can x B_aux_can

    // first we set-up the kernel
    theta_couple_curve_t EchallxEaux;
    copy_curve(&EchallxEaux.E1, E_chall);
    copy_curve(&EchallxEaux.E2, E_aux);

    theta_kernel_couple_points_t dim_two_ker;
    copy_bases_to_kernel(&dim_two_ker, B_chall_can_unvec, B_aux_can_unvec);

    // computing the isogeny
    int codomain_splits;
    
    // handling the special case where we don't need to perform any dim2 computation
    if (pow_dim2_deg_resp == 0) {
        codomain_splits = 1;
        copy_curve(E_com, &EchallxEaux.E1);

        // We still need to check that E_chall is supersingular
        // This assumes that HD_extra_torsion == 2
        if (!ec_is_basis_four_torsion(B_chall_can_unvec, E_chall)) {
            return 0;
        }
    } else {
        //uint64_t start = cpucycles();
        theta_couple_curve_t codomain;
        ec_curve_init(&codomain.E1);
        ec_curve_init(&codomain.E2);

        //for (int i = 0; i < 1000; i++) {
        codomain_splits = theta_isogeny_chain_ver(
            &codomain, &EchallxEaux, &dim_two_ker, pow_dim2_deg_resp
        );
        //}

        // computing the commitment curve
        // its always the first one because of our (2^n,2^n)-isogeny formulae
        copy_curve(E_com, &codomain.E1);
        // uint64_t end = cpucycles();
        // printf("2D chain len %d runs in %" PRIu64 "\n", pow_dim2_deg_resp, (end-start)/1000);
    }
    return codomain_splits;
}

// SQIsign verification
bool
protocols_verify(signature_t *sig, const public_key_t *pk, const unsigned char *m, size_t l)
{
    // uint64_t start1 = cpucycles();

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

    // checking that we are given A-coefficients and no precomputation
    if(fp2_is_one(&pk->curve.C) != 0xFFFFFFFF || pk->curve.is_A24_computed_and_normalized)
        return false;

    // ============= Converting input to vectorized format ============================
    __m512i Epk_A24[8], Echl_A24[8];
    
    // check the public curve is valid
    if (!ec_curve_verify_A(&(pk->curve).A))
        return false;

    // Set auxiliary curve from the A-coefficient within the signature
    ec_curve_t E_aux;
    if (!ec_curve_init_from_A(&E_aux, &sig->E_aux_A))
        return false; // invalid curve

    // computation of the challenge
    //uint64_t start = cpucycles();
    ec_curve_t E_chall;
    // for (int i = 0; i < 1000; i++) {
    if (!compute_challenge_verify(&E_chall, sig, &pk->curve, pk->hint_pk)) {
        return false;
    }
    // }
    // uint64_t end = cpucycles();
    // printf("compute_challenge_verify %" PRIu64 "\n", (end-start)/1000);

    // Computation of the canonical bases for the challenge and aux curve
    ec_basis_t B_chall_can, B_aux_can;
    // start = cpucycles();
    // for (int i = 0; i < 1000; i++) {
    if (!challenge_and_aux_basis_verify(&B_chall_can, &B_aux_can, &E_chall, &E_aux, sig, pow_dim2_deg_resp)) {
        return false;
    }
    // }
    // end = cpucycles();
    // printf("challenge_and_aux_basis_verify %" PRIu64 "\n", (end-start)/1000);

    // When two_resp_length != 0 we need to compute a second, short 2^r-isogeny
    if (sig->two_resp_length > 0) {
        if (!two_response_isogeny_verify(&E_chall, &B_chall_can, sig, pow_dim2_deg_resp)) {
            return false;
        }
    }

    // We can recover the commitment curve with a 2D isogeny
    // The supplied signature did not compute an isogeny between eliptic products
    // and so definitely is an invalid signature.
    // start = cpucycles();
    ec_curve_t E_com;
    // for (int i = 0; i < 1000; i++) {
    if (!compute_commitment_curve_verify(&E_com, &B_chall_can, &B_aux_can, &E_chall, &E_aux, pow_dim2_deg_resp))
        return false;
    // }
    // end = cpucycles();
    // printf("compute_commitment_curve_verify %" PRIu64 "\n", (end-start)/1000);

    // recomputing the challenge vector
    scalar_t chk_chall;
    hash_to_challenge(&chk_chall, pk, &E_com, m, l);

    // uint64_t end1 = cpucycles();
    // printf("SQIsign verif runs in %" PRIu64 "\n", (end1-start1)/1000);

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
    public_key_t pkt = { 0 };
    signature_t sigt;

    public_key_from_bytes(&pkt, pk);
    signature_from_bytes(&sigt, sig);

    return protocols_verify(&sigt, &pkt, m, mlen);
}
