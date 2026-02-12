/** @file
 *
 * @authors Luca De Feo, Francisco RH
 *
 * @brief Elliptic curve stuff
 */

#ifndef SQI_EC_H
#define SQI_EC_H

#include "sqi_fp2.h"

#include <stdbool.h>

/** @brief Projective point on the Kummer line E/pm 1 in Montgomery coordinates
 *
 * @typedef ec_point_t
 *
 * @struct ec_point_t
 *
 * A projective point in (X:Z) or (X:Y:Z) coordinates (tbd).
 */
typedef struct ec_point_t
{
    fp2_t x;
    fp2_t z;
} ec_point_t;

/** @brief Addition components
 *
 * @typedef add_components_t
 *
 * @struct add_components_t
 *
 * 3 components u,v,w that define the (X:Z) coordinates of both
 * addition and substraction of two distinct points with
 * P+Q =(u-v:w) and P-Q = (u+v=w)
 */
typedef struct add_components_t
{
    fp2_t u;
    fp2_t v;
    fp2_t w;
} add_components_t;

/** @brief A basis of a torsion subgroup
 *
 * @typedef ec_basis_t
 *
 * @struct ec_basis_t
 *
 * A pair of points (or a triplet, tbd) forming a basis of a torsion subgroup.
 */
typedef struct ec_basis_t
{
    ec_point_t P;
    ec_point_t Q;
    ec_point_t PmQ;
} ec_basis_t;

/** @brief An elliptic curve
 *
 * @typedef ec_curve_t
 *
 * @struct ec_curve_t
 *
 * An elliptic curve in projective Montgomery form
 */
typedef struct ec_curve_t
{
    fp2_t A;
    fp2_t C;                             ///< cannot be 0
    ec_point_t A24;                      // the point (A+2 : 4C)
    bool is_A24_computed_and_normalized; // says if A24 has been computed and normalized
} ec_curve_t;

/** @}
 */
/** @defgroup ec_point_t Point operations
 * @{
 */

// Initalisation for curves and points
void ec_curve_init(ec_curve_t *E);
void ec_point_init(ec_point_t *P);

/**
 * @brief Point equality
 *
 * @param P a point
 * @param Q a point
 * @return 0xFFFFFFFF if equal, zero otherwise
 */
uint32_t ec_is_equal(const ec_point_t *P, const ec_point_t *Q);

/**
 * @brief Point equality
 *
 * @param P a point
 * @return 0xFFFFFFFF if point at infinity, zero otherwise
 */
uint32_t ec_is_zero(const ec_point_t *P);

// Copying points, bases and curves
static inline void
copy_point(ec_point_t *P, const ec_point_t *Q)
{
    fp2_copy(&P->x, &Q->x);
    fp2_copy(&P->z, &Q->z);
}

static inline void
copy_basis(ec_basis_t *B1, const ec_basis_t *B0)
{
    copy_point(&B1->P, &B0->P);
    copy_point(&B1->Q, &B0->Q);
    copy_point(&B1->PmQ, &B0->PmQ);
}

static inline void
copy_curve(ec_curve_t *E1, const ec_curve_t *E2)
{
    fp2_copy(&(E1->A), &(E2->A));
    fp2_copy(&(E1->C), &(E2->C));
    E1->is_A24_computed_and_normalized = E2->is_A24_computed_and_normalized;
    copy_point(&E1->A24, &E2->A24);
}

/**
 * @brief Given a curve E, compute (A+2 : 4C)
 *
 * @param A24 the value (A+2 : 4C) to return into
 * @param E a curve
 */
static inline void
AC_to_A24(ec_point_t *A24, const ec_curve_t *E)
{
    // Maybe we already have this computed
    if (E->is_A24_computed_and_normalized) {
        copy_point(A24, &E->A24);
        return;
    }

    // A24 = (A+2C : 4C)
    fp2_add(&A24->z, &E->C, &E->C);
    fp2_add(&A24->x, &E->A, &A24->z);
    fp2_add(&A24->z, &A24->z, &A24->z);
}

/**
 * @brief Point doubling
 *
 * @param res computed double of P
 * @param P a point
 * @param curve an elliptic curve
 */
void ec_dbl(ec_point_t *res, const ec_point_t *P, const ec_curve_t *curve);

/**
 * @brief Point iterated doubling
 *
 * @param res computed double of P
 * @param P a point
 * @param n the number of double
 * @param curve the curve on which P lays
 */
void ec_dbl_iter(ec_point_t *res, int n, const ec_point_t *P, ec_curve_t *curve);

void xDBL_A24(ec_point_t *Q, const ec_point_t *P, const ec_point_t *A24, const bool A24_normalized);

void xADD(ec_point_t *R, const ec_point_t *P, const ec_point_t *Q, const ec_point_t *PQ);

/**
 * @brief Reduce Z-coordinate of point in place
 *
 * @param P a point
 */
void ec_normalize_point(ec_point_t *P);

/**
 * @brief Reduce (A : C) to (A/C : 1) in place
 *
 * @param E a curve
 */
void ec_normalize_curve(ec_curve_t *E);

/**
 * @brief Reduce (A + 2 : 4C) to ((A+2)/4C : 1) in place
 *
 * @param E a curve
 */
void ec_curve_normalize_A24(ec_curve_t *E);

// Jacobian arithmetic

/** @brief Projective point in Montgomery coordinates
 *
 * @typedef jac_point_t
 *
 * @struct jac_point_t
 *
 * A projective point in (X:Y:Z) coordinates
 */
typedef struct jac_point_t
{
    fp2_t x;
    fp2_t y;
    fp2_t z;
} jac_point_t;

// Jacobian point init and copying
void jac_init(jac_point_t *P);
void copy_jac_point(jac_point_t *P, const jac_point_t *Q);

/**
 * @brief Test if two Jacobian points are equal
 *
 * @param P: a point
 * @param Q: a point
 *
 * @return 0xFFFFFFFF if they are equal, 0 otherwise
 */
uint32_t jac_is_equal(const jac_point_t *P, const jac_point_t *Q);

// Convert from Jacobian to x-only (just drop the Y-coordinate)
void jac_to_xz(ec_point_t *P, const jac_point_t *xyP);
// Convert from Jacobian coordinates in Montgomery model to Weierstrass
void jac_to_ws(jac_point_t *P, fp2_t *t, fp2_t *ao3, const jac_point_t *Q, const ec_curve_t *curve);
void jac_from_ws(jac_point_t *Q, const jac_point_t *P, const fp2_t *ao3, const ec_curve_t *curve);


void jac_neg(jac_point_t *Q, const jac_point_t *P);
void ADD(jac_point_t *R, const jac_point_t *P, const jac_point_t *Q, const ec_curve_t *AC);
void DBL(jac_point_t *Q, const jac_point_t *P, const ec_curve_t *AC);
void DBLW(jac_point_t *Q, fp2_t *u, const jac_point_t *P, const fp2_t *t);
void jac_to_xz_add_components(add_components_t *uvw, const jac_point_t *P, const jac_point_t *Q, const ec_curve_t *AC);

/**
 * @brief Given a basis in x-only, lift to a pair of Jacobian points
 *
 * @param P: a point
 * @param Q: a point
 * @param B: a basis
 * @param E: an elliptic curve
 * @param verify: if true, it checks if the lift was successful.
 *
 * @return 0xFFFFFFFF if there was no error, 0 otherwise
 */
uint32_t lift_basis(jac_point_t *P, jac_point_t *Q, ec_basis_t *B, ec_curve_t *E, const bool verify);

/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Test functions for printing and order checking, only used in debug mode
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

/**
 * @brief Check if a point (X : Z) has order exactly 2^t
 *
 * @param P: a point
 * @param E: an elliptic curve
 * @param t: an integer
 *
 * @return 0xFFFFFFFF if the order is correct, 0 otherwise
 */
static int test_point_order_twof(const ec_point_t *P, const ec_curve_t *E, int t)
{
    ec_point_t test;
    ec_curve_t curve;
    test = *P;
    copy_curve(&curve, E);

    if (ec_is_zero(&test))
        return 0;
    // Scale point by 2^(t-1)
    ec_dbl_iter(&test, t - 1, &test, &curve);
    // If it's zero now, it doesnt have order 2^t
    if (ec_is_zero(&test))
        return 0;
    // Ensure [2^t] P = 0
    ec_dbl(&test, &test, &curve);
    return ec_is_zero(&test);
}

/**
 * @brief Check if a [2^t] * (X : Z) = 0
 *
 * @param P: a point
 * @param E: an elliptic curve
 * @param t: an integer
 *
 * @return 0xFFFFFFFF if the order is correct, 0 otherwise
 */
static int test_point_torsion_twof(const ec_point_t *P, const ec_curve_t *E, int t)
{
    ec_point_t test;
    ec_curve_t curve;
    test = *P;
    copy_curve(&curve, E);

    // Scale point by 2^(t-1)
    ec_dbl_iter(&test, t, &test, &curve);
    // If it's zero now, it doesnt have order 2^t
    return ec_is_zero(&test);
}

/**
 * @brief Check if a Jacobian point (X : Y : Z) has order exactly 2^f
 *
 * @param P: a point
 * @param E: an elliptic curve
 * @param t: an integer
 *
 * @return 0xFFFFFFFF if the order is correct, 0 otherwise
 */
static int test_jac_order_twof(const jac_point_t *P, const ec_curve_t *E, int t)
{
    jac_point_t test;
    test = *P;
    if (fp2_is_zero(&test.z))
        return 0;
    for (int i = 0; i < t - 1; i++) {
        DBL(&test, &test, E);
    }
    if (fp2_is_zero(&test.z))
        return 0;
    DBL(&test, &test, E);
    return (fp2_is_zero(&test.z));
}

static void ec_montA_to_curve(ec_curve_t *E, const fp2_t *A)
{
    fp2_copy(&E->A, A);
    fp2_set_one(&E->C);
    E->is_A24_computed_and_normalized = false;
    ec_curve_normalize_A24(E);
}

static void ec_montAC_to_curve(ec_curve_t *E, const fp2_t *A, const fp2_t *C)
{
    fp2_copy(&E->A, A);
    fp2_copy(&E->C, C);
    E->is_A24_computed_and_normalized = false;
    ec_curve_normalize_A24(E);
    ec_normalize_curve(E);
}

static int ec_test_is_on_curve(const ec_curve_t *E, const ec_point_t *P) 
{
    fp2_t x, one, C_inv, tmp;

    // x = X/Z
    fp2_copy(&x, &P->z);
    fp2_inv(&x);
    fp2_mul(&x, &x, &P->x);   // x1 = X1 / Z1

    fp2_set_one(&one);
    fp2_copy(&C_inv, &E->C);
    fp2_inv(&C_inv);

    // check that Cy^2 = x^3 + Ax^2 + x
    fp2_add(&tmp, &x, &E->A);
    fp2_mul(&tmp, &tmp, &x);
    fp2_add(&tmp, &tmp, &one);
    fp2_mul(&tmp, &tmp, &x);
    fp2_mul(&tmp, &tmp, &C_inv);

    return fp2_is_square(&tmp) | fp2_is_zero(&P->z); // point at inf
}

static void ec_j_inv(fp2_t* j_inv, ec_curve_t* curve) 
{
    fp2_t nom, denom, tmp;

    fp2_sqr(&tmp, &curve->A);
    
    // nom = 3, denom = 4
    fp2_set_one(&nom);
    fp2_add(&denom, &nom, &nom);
    fp2_add(&denom, &denom, &denom);
    fp2_sub(&nom, &denom, &nom);

    // nom = A^2 - 3, denom = A^2 - 4
    fp2_sub(&nom, &tmp, &nom);
    fp2_sub(&denom, &tmp, &denom);

    // nom = 2^8 * (A^2 - 3)^3
    fp2_sqr(&tmp, &nom);
    fp2_mul(&nom, &nom, &tmp);
    for (int i = 0; i < 8; i++) {
        fp2_add(&nom, &nom, &nom);
    }

    // nom / denom
    fp2_inv(&denom);
    fp2_mul(j_inv, &nom, &denom);
}

#endif