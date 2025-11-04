#include "sqi_ec.h"

uint32_t ec_is_zero(const ec_point_t *P)
{
    return fp2_is_zero(&P->z);
}

uint32_t ec_is_equal(const ec_point_t *P, const ec_point_t *Q)
{ // Evaluate if two points in Montgomery coordinates (X:Z) are equal
  // Returns 0xFFFFFFFF (true) if P=Q, 0 (false) otherwise
    fp2_t t0, t1;

    // Check if P, Q are the points at infinity
    uint32_t l_zero = ec_is_zero(P);
    uint32_t r_zero = ec_is_zero(Q);

    // Check if PX * QZ = QX * PZ
    fp2_mul(&t0, &P->x, &Q->z);
    fp2_mul(&t1, &P->z, &Q->x);
    uint32_t lr_equal = fp2_is_equal(&t0, &t1);

    // Points are equal if
    // - Both are zero, or
    // - neither are zero AND PX * QZ = QX * PZ
    return (l_zero & r_zero) | (~l_zero & ~r_zero * lr_equal);
}

void
xDBL(ec_point_t *Q, const ec_point_t *P, const ec_point_t *AC)
{ // Doubling of a Montgomery point in projective coordinates (X:Z). Computation of coefficient values A+2C and 4C
  // on-the-fly. 
  // Input: projective Montgomery x-coordinates P = (XP:ZP), where xP=XP/ZP, and Montgomery curve constants (A:C). 
  // Output: projective Montgomery x-coordinates Q <- 2*P = (XQ:ZQ) such that x(2P)=XQ/ZQ.
    fp2_t t0, t1, t2, t3;

    fp2_add(&t0, &P->x, &P->z);
    fp2_sqr(&t0, &t0);
    fp2_sub(&t1, &P->x, &P->z);
    fp2_sqr(&t1, &t1);
    fp2_sub(&t2, &t0, &t1);
    fp2_add(&t3, &AC->z, &AC->z);
    fp2_mul(&t1, &t1, &t3);
    fp2_add(&t1, &t1, &t1);
    fp2_mul(&Q->x, &t0, &t1);
    fp2_add(&t0, &t3, &AC->x);
    fp2_mul(&t0, &t0, &t2);
    fp2_add(&t0, &t0, &t1);
    fp2_mul(&Q->z, &t0, &t2);
}

void
xDBL_A24(ec_point_t *Q, const ec_point_t *P, const ec_point_t *A24, const bool A24_normalized)
{ // Doubling of a Montgomery point in projective coordinates (X:Z).
  // Input: projective Montgomery x-coordinates P = (XP:ZP), where xP=XP/ZP, and
  //        the Montgomery curve constants A24 = (A+2C:4C) (or A24 = (A+2C/4C:1) if normalized).
  // Output: projective Montgomery x-coordinates Q <- 2*P = (XQ:ZQ) such that x(2P)=XQ/ZQ.
    fp2_t t0, t1, t2;

    fp2_add(&t0, &P->x, &P->z);
    fp2_sqr(&t0, &t0);
    fp2_sub(&t1, &P->x, &P->z);
    fp2_sqr(&t1, &t1);
    fp2_sub(&t2, &t0, &t1);
    if (!A24_normalized)
        fp2_mul(&t1, &t1, &A24->z);
    fp2_mul(&Q->x, &t0, &t1);
    fp2_mul(&t0, &t2, &A24->x);
    fp2_add(&t0, &t0, &t1);
    fp2_mul(&Q->z, &t0, &t2);
}

void ec_dbl(ec_point_t *res, const ec_point_t *P, const ec_curve_t *curve)
{
    // If A24 = ((A+2)/4 : 1) we save multiplications
    if (curve->is_A24_computed_and_normalized) {
        assert(fp2_is_one(&curve->A24.z));
        xDBL_A24(res, P, &curve->A24, true);
    } else {
        // Otherwise we compute A24 on the fly for doubling
        xDBL(res, P, (const ec_point_t *)curve);
    }
}

void ec_dbl_iter(ec_point_t *res, int n, const ec_point_t *P, ec_curve_t *curve)
{
    if (n == 0) {
        copy_point(res, P);
        return;
    }

    // When the chain is long enough, we should normalise A24
    if (n > 50) {
        ec_curve_normalize_A24(curve);
    }

    // When A24 is normalized we can save some multiplications
    if (curve->is_A24_computed_and_normalized) {
        assert(fp2_is_one(&curve->A24.z));
        xDBL_A24(res, P, &curve->A24, true);
        for (int i = 0; i < n - 1; i++) {
            assert(fp2_is_one(&curve->A24.z));
            xDBL_A24(res, res, &curve->A24, true);
        }
    } else {
        // Otherwise we do normal doubling
        xDBL(res, P, (const ec_point_t *)curve);
        for (int i = 0; i < n - 1; i++) {
            xDBL(res, res, (const ec_point_t *)curve);
        }
    }
}

void
ec_normalize_point(ec_point_t *P)
{
    fp2_inv(&P->z);
    fp2_mul(&P->x, &P->x, &P->z);
    fp2_set_one(&(P->z));
}

void
ec_normalize_curve(ec_curve_t *E)
{
    fp2_inv(&E->C);
    fp2_mul(&E->A, &E->A, &E->C);
    fp2_set_one(&E->C);
}

void
ec_curve_normalize_A24(ec_curve_t *E)
{
    if (!E->is_A24_computed_and_normalized) {
        AC_to_A24(&E->A24, E);
        ec_normalize_point(&E->A24);
        E->is_A24_computed_and_normalized = true;
    }
    assert(fp2_is_one(&E->A24.z));
}