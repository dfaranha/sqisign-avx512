#include "sqi_ec.h"

uint32_t ec_recover_y(fp2_t *y, const fp2_t *Px, const ec_curve_t *curve)
{ // Recover y-coordinate of a point on the Montgomery curve y^2 = x^3 + Ax^2 + x
    fp2_add(y, Px, &curve->A); // A + x
    fp2_mul(y, y, Px);         // x^2 + Ax
    fp2_add_one(y, y);         // x^2 + Ax + 1
    fp2_mul(y, y, Px);         // x^3 + Ax^2 + x
    // This is required, because we do not yet know that our curves are
    // supersingular so our points live on the twist with B = 1.
    return fp2_sqrt_verify(y);
}

// Lifts a basis x(P), x(Q), x(P-Q) assuming the curve has (A/C : 1) and the point
// P = (X/Z : 1). For generic implementation see lift_basis()
uint32_t lift_basis_normalized(jac_point_t *P, jac_point_t *Q, ec_basis_t *B, ec_curve_t *E)
{
    assert(fp2_is_one(&B->P.z));
    assert(fp2_is_one(&E->C));

    fp2_copy(&P->x, &B->P.x);
    fp2_copy(&Q->x, &B->Q.x);
    fp2_copy(&Q->z, &B->Q.z);
    fp2_set_one(&P->z);
    uint32_t ret = ec_recover_y(&P->y, &P->x, E);

    // Algorithm of Okeya-Sakurai to recover y.Q in the montgomery model
    fp2_t v1, v2, v3, v4;
    fp2_mul(&v1, &P->x, &Q->z);
    fp2_add(&v2, &Q->x, &v1);
    fp2_sub(&v3, &Q->x, &v1);
    fp2_sqr(&v3, &v3);
    fp2_mul(&v3, &v3, &B->PmQ.x);
    fp2_add(&v1, &E->A, &E->A);
    fp2_mul(&v1, &v1, &Q->z);
    fp2_add(&v2, &v2, &v1);
    fp2_mul(&v4, &P->x, &Q->x);
    fp2_add(&v4, &v4, &Q->z);
    fp2_mul(&v2, &v2, &v4);
    fp2_mul(&v1, &v1, &Q->z);
    fp2_sub(&v2, &v2, &v1);
    fp2_mul(&v2, &v2, &B->PmQ.z);
    fp2_sub(&Q->y, &v3, &v2);
    fp2_add(&v1, &P->y, &P->y);
    fp2_mul(&v1, &v1, &Q->z);
    fp2_mul(&v1, &v1, &B->PmQ.z);
    fp2_mul(&Q->x, &Q->x, &v1);
    fp2_mul(&Q->z, &Q->z, &v1);

    // Transforming to a jacobian coordinate
    fp2_sqr(&v1, &Q->z);
    fp2_mul(&Q->y, &Q->y, &v1);
    fp2_mul(&Q->x, &Q->x, &Q->z);
    return ret;
}

uint32_t lift_basis(jac_point_t *P, jac_point_t *Q, ec_basis_t *B, ec_curve_t *E)
{
    // Normalise the curve E such that (A : C) is (A/C : 1)
    // and the point x(P) = (X/Z : 1).
    fp2_t inverses[2];
    fp2_copy(&inverses[0], &B->P.z);
    fp2_copy(&inverses[1], &E->C);

    fp2_batched_inv(inverses, 2);
    fp2_set_one(&B->P.z);
    fp2_set_one(&E->C);

    fp2_mul(&B->P.x, &B->P.x, &inverses[0]);
    fp2_mul(&E->A, &E->A, &inverses[1]);

    // Lift the basis to Jacobian points P, Q
    return lift_basis_normalized(P, Q, B, E);
}

// Given an x-coordinate, determines if this is a valid
// point on the curve. Assumes C=1.
static uint32_t is_on_curve(const fp2_t *x, const ec_curve_t *curve)
{
    assert(fp2_is_one(&curve->C));
    fp2_t t0;

    fp2_add(&t0, x, &curve->A); // x + (A/C)
    fp2_mul(&t0, &t0, x);       // x^2 + (A/C)*x
    fp2_add_one(&t0, &t0);      // x^2 + (A/C)*x + 1
    fp2_mul(&t0, &t0, x);       // x^3 + (A/C)*x^2 + x

    return fp2_is_square(&t0);
}