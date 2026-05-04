#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>

#include "utils.h"

#include "sqi_encoded_sizes.h"
#include "sqi_fp_constants.h"
#include "sqi_test_extras.h"

int cmp_u64(const void *v1, const void *v2)
{
    uint64_t x1 = *(const uint64_t *)v1;
    uint64_t x2 = *(const uint64_t *)v2;
    if (x1 < x2) {
        return -1;
    } else if (x1 == x2) {
        return 0;
    } else {
        return 1;
    }
}

void fp_random_test(fp_t *a)
{
    uint8_t tmp[FP_ENCODED_BYTES];

    rand_bytes(tmp, sizeof(tmp));

    fp_decode_reduce(a, tmp, sizeof(tmp));
}

void fp2_random_test(fp2_t *a)
{
    fp_random_test(&(a->re));
    fp_random_test(&(a->im));
}

void print_fp(const fp_t *a)
{
    fp_t coeffs;

    // Convert from Montgomery representation
    fp_encode(coeffs.arr, a);

    printf("{ ");
    for (int i = 0; i < NWORDS_FIELD; i++) {
        printf("0x%" PRIx64, coeffs.arr[i]);
        if (i != NWORDS_FIELD - 1) {
            printf(", ");
        }
    }
    printf(" }\n");
}

void print_fp2(const fp2_t *a) 
{
    print_fp(&a->re);
    print_fp(&a->im);
}

void ec_random_normalized_test(ec_point_t *P, const ec_curve_t *curve)
{
    fp2_set_one(&P->z);
    while (1) {
        fp2_random_test(&P->x);
        if (ec_test_is_on_curve(curve, P)) {
            break;
        }
    }
}

void ec_random_test(ec_point_t *P, const ec_curve_t *curve)
{
    ec_random_normalized_test(P, curve);
    fp2_random_test(&P->z);
    fp2_mul(&P->x, &P->x, &P->z);
}

void projective_difference_point(ec_point_t *PQ, const ec_point_t *P, const ec_point_t *Q, const ec_curve_t *curve)
{
    // Given P,Q in projective x-only, computes a deterministic choice for (P-Q)
    // Based on Proposition 3 of https://eprint.iacr.org/2017/518.pdf

    fp2_t Bxx, Bxz, Bzz, t0, t1;

    fp2_mul(&t0, &P->x, &Q->x);
    fp2_mul(&t1, &P->z, &Q->z);
    fp2_sub(&Bxx, &t0, &t1);
    fp2_sqr(&Bxx, &Bxx);
    fp2_mul(&Bxx, &Bxx, &curve->C); // C*(P.x*Q.x-P.z*Q.z)^2
    fp2_add(&Bxz, &t0, &t1);
    fp2_mul(&t0, &P->x, &Q->z);
    fp2_mul(&t1, &P->z, &Q->x);
    fp2_add(&Bzz, &t0, &t1);
    fp2_mul(&Bxz, &Bxz, &Bzz); // (P.x*Q.x+P.z*Q.z)(P.x*Q.z+P.z*Q.x)
    fp2_sub(&Bzz, &t0, &t1);
    fp2_sqr(&Bzz, &Bzz);
    fp2_mul(&Bzz, &Bzz, &curve->C); // C*(P.x*Q.z-P.z*Q.x)^2
    fp2_mul(&Bxz, &Bxz, &curve->C); // C*(P.x*Q.x+P.z*Q.z)(P.x*Q.z+P.z*Q.x)
    fp2_mul(&t0, &t0, &t1);
    fp2_mul(&t0, &t0, &curve->A);
    fp2_add(&t0, &t0, &t0);
    fp2_add(&Bxz, &Bxz, &t0); // C*(P.x*Q.x+P.z*Q.z)(P.x*Q.z+P.z*Q.x) + 2*A*P.x*Q.z*P.z*Q.x

    // Normalization: our squareroot always has the same sign as long as P.z, Q.z, and C
    // are in Fp and C is a square, so the B's should be scaled by C*C_bar^2*P.z_bar^2*Q.Z_bar^2
    fp_copy(&t0.re, &curve->C.re);
    fp_neg(&t0.im, &curve->C.im);
    fp2_sqr(&t0, &t0);
    fp2_mul(&t0, &t0, &curve->C);
    fp_copy(&t1.re, &P->z.re);
    fp_neg(&t1.im, &P->z.im);
    fp2_sqr(&t1, &t1);
    fp2_mul(&t0, &t0, &t1);
    fp_copy(&t1.re, &Q->z.re);
    fp_neg(&t1.im, &Q->z.im);
    fp2_sqr(&t1, &t1);
    fp2_mul(&t0, &t0, &t1);
    fp2_mul(&Bxx, &Bxx, &t0);
    fp2_mul(&Bxz, &Bxz, &t0);
    fp2_mul(&Bzz, &Bzz, &t0);

    // Solving quadratic equation
    fp2_sqr(&t0, &Bxz);
    fp2_mul(&t1, &Bxx, &Bzz);
    fp2_sub(&t0, &t0, &t1);
    fp2_sqrt(&t0);
    fp2_add(&PQ->x, &Bxz, &t0);
    fp2_copy(&PQ->z, &Bzz);
}