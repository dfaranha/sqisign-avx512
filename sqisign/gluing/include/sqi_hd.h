/** @file
 *
 * @authors Antonin Leroux
 *
 * @brief The HD-isogenies algorithm required by the signature
 *
 */

#include "sqi_ec.h"
#include "sqi_fp2.h"

#include <stdbool.h>

#ifndef SQI_HD_H
#define SQI_HD_H

/** @defgroup hd_module Abelian surfaces and their isogenies
 * @{
 */

#define HD_extra_torsion 2

/** @defgroup hd_struct Data structures for dimension 2
 * @{
 */

/** @brief Type for couple point with XZ coordinates
 * @typedef theta_couple_point_t
 *
 * @struct theta_couple_point
 *
 * Structure for the couple point on an elliptic product
 * using XZ coordinates
 */
typedef struct theta_couple_point
{
    ec_point_t P1;
    ec_point_t P2;
} theta_couple_point_t;

/** @brief Type for three couple points T1, T2, T1-T2 with XZ coordinates
 * @typedef theta_kernel_couple_points_t
 *
 * @struct theta_kernel_couple_points
 *
 * Structure for a triple of theta couple points T1, T2 and T1 - T2
 */
typedef struct theta_kernel_couple_points
{
    theta_couple_point_t T1;
    theta_couple_point_t T2;
    theta_couple_point_t T1m2;
} theta_kernel_couple_points_t;

/** @brief Type for couple point with XYZ coordinates
 * @typedef theta_couple_jac_point_t
 *
 * @struct theta_couple_jac_point
 *
 * Structure for the couple point on an elliptic product
 * using XYZ coordinates
 */
typedef struct theta_couple_jac_point
{
    jac_point_t P1;
    jac_point_t P2;
} theta_couple_jac_point_t;

/** @brief Type for couple curve *
 * @typedef theta_couple_curve_t
 *
 * @struct theta_couple_curve
 *
 * the  theta_couple_curve structure
 */
typedef struct theta_couple_curve
{
    ec_curve_t E1;
    ec_curve_t E2;
} theta_couple_curve_t;

/** @brief Type for a product E1 x E2 with corresponding bases
 * @typedef theta_couple_curve_with_basis_t
 *
 * @struct theta_couple_curve_with_basis
 *
 * tType for a product E1 x E2 with corresponding bases Ei[2^n]
 */
typedef struct theta_couple_curve_with_basis
{
    ec_curve_t E1;
    ec_curve_t E2;
    ec_basis_t B1;
    ec_basis_t B2;
} theta_couple_curve_with_basis_t;

/** @brief Type for theta point *
 * @typedef theta_point_t
 *
 * @struct theta_point
 *
 * the  theta_point structure used
 */
typedef struct theta_point
{
    fp2_t x;
    fp2_t y;
    fp2_t z;
    fp2_t t;
} theta_point_t;

/** @brief Type for theta point with repeating components
 * @typedef theta_point_compact_t
 *
 * @struct theta_point_compact
 *
 * the  theta_point structure used for points with repeated components
 */
typedef struct theta_point_compact
{
    fp2_t x;
    fp2_t y;
} theta_point_compact_t;

/** @brief Type for theta structure *
 * @typedef theta_structure_t
 *
 * @struct theta_structure
 *
 * the  theta_structure structure used
 */
typedef struct theta_structure
{
    theta_point_t null_point;
    bool precomputation;

    // Eight precomputed values used for doubling and
    // (2,2)-isogenies.
    fp2_t XYZ0;
    fp2_t YZT0;
    fp2_t XZT0;
    fp2_t XYT0;

    fp2_t xyz0;
    fp2_t yzt0;
    fp2_t xzt0;
    fp2_t xyt0;
} theta_structure_t;

/** @brief A 2x2 matrix used for action by translation
 * @typedef translation_matrix_t
 *
 * @struct translation_matrix
 *
 * Structure to hold 4 fp2_t elements representing a 2x2 matrix used when computing
 * a compatible theta structure during gluing.
 */
typedef struct translation_matrix
{
    fp2_t g00;
    fp2_t g01;
    fp2_t g10;
    fp2_t g11;
} translation_matrix_t;

/** @brief A 4x4 matrix used for basis changes
 * @typedef basis_change_matrix_t
 *
 * @struct basis_change_matrix
 *
 * Structure to hold 16 elements representing a 4x4 matrix used for changing
 * the basis of a theta point.
 */
typedef struct basis_change_matrix
{
    fp2_t m[4][4];
} basis_change_matrix_t;

/** @brief Type for a symetric element induced by a 4 torsion point
 * @typedef symetric_element_t
 *
 * @struct symetric_element
 *
 * Structure for a projective symetric element induced by a 4 torsion points.
 * It consists in the element of the 2x2 matrix with index zero
 */
typedef struct symetric_element
{
    fp2_t s00;
    fp2_t s01;
    fp2_t s10;
} symetric_element_t;

/** @brief Type for a basis of symetric element induced by a 4 torsion basis
 * @typedef symetric_element_basis_t
 *
 * @struct symetric_element_basis
 *
 * Structure for a projective symetric element induced by a 4 torsion basis.
 * It consists in delta the determinant of all symetric elements and of 3
 * symetric ellements
 */
typedef struct symetric_element_basis
{
    fp2_t delta;
    symetric_element_t g1;
    symetric_element_t g2;
    symetric_element_t g3;
} symetric_element_basis_t;

/** @brief Element required to compute gluing
 * @typedef gluing_basis
 *
 * @struct gluing_basis_t
 *
 * Structure used to compute a gluing basis
 */
typedef struct gluing_basis
{

    theta_point_t square_dual_point;
    theta_point_compact_t components_01_23;
    theta_point_compact_t components_02_13;
    theta_point_compact_t components_03_12;
    bool sign_flip;

} gluing_basis_t;

/** @brief Type for gluing (2,2) theta isogeny *
 * @typedef theta_gluing_t
 *
 * @struct theta_gluing
 *
 * the  theta_gluing structure
 */
typedef struct theta_gluing
{

    theta_couple_curve_t domain;
    theta_couple_jac_point_t xyK1_8;
    theta_point_compact_t imageK1_8;
    gluing_basis_t basis;
    uint8_t position;
    theta_point_t precomputation;
    theta_point_t codomain;

} theta_gluing_t;

/** @brief Type for standard (2,2) theta isogeny *
 * @typedef theta_isogeny_t
 *
 * @struct theta_isogeny
 *
 * the  theta_isogeny structure
 */
typedef struct theta_isogeny
{
    theta_point_t T1_8;
    theta_point_t T2_8;
    bool hadamard_bool_1;
    bool hadamard_bool_2;
    theta_structure_t domain;
    theta_point_t precomputation;
    theta_structure_t codomain;
} theta_isogeny_t;

/** @brief Type for splitting isomorphism *
 * @typedef theta_splitting_t
 *
 * @struct theta_splitting
 *
 * the theta_splitting structure
 */
typedef struct theta_splitting
{
    basis_change_matrix_t M;
    theta_structure_t B;

} theta_splitting_t;

/**
 * @brief Given a bases B1 on E1 and B2 on E2 copies this to create a kernel
 *         on E1 x E2 as couple points T1, T2 and T1 - T2
 *
 * @param ker Output: a kernel for dim_two_isogenies (T1, T2, T1-T2)
 * @param B1 Input basis on E1
 * @param B2 Input basis on E2
 **/
void copy_bases_to_kernel(theta_kernel_couple_points_t *ker, const ec_basis_t *B1, const ec_basis_t *B2);

/**
 * @brief Compute the double of the theta couple point in on the elliptic product E12
 *
 * @param out Output: the theta_couple_point
 * @param in the theta couple point in the elliptic product
 * @param E1E2 an elliptic product
 * in = (P1,P2)
 * out = [2] (P1,P2)
 *
 */
void double_couple_point(theta_couple_point_t *out, const theta_couple_point_t *in, const theta_couple_curve_t *E1E2);

/**
 * @brief Compute the double of the theta couple point in on the elliptic product E12
 *
 * @param out Output: the theta_couple_point
 * @param in the theta couple point in the elliptic product
 * @param E1E2 an elliptic product
 * in = (P1,P2)
 * out = [2] (P1,P2)
 *
 */
void double_couple_jac_point(theta_couple_jac_point_t *out,
                             const theta_couple_jac_point_t *in,
                             const theta_couple_curve_t *E1E2);

/**
 * @brief Compute the iterated double of the theta couple jac point in on the elliptic product E12
 *
 * @param out Output: the theta_couple_jac_point
 * @param n : the number of iteration
 * @param in the theta couple jac point in the elliptic product
 * @param E1E2 an elliptic product
 * in  = (P1,P2)
 * out = [2^n] (P1,P2)
 *
 */
void double_couple_jac_point_iter(theta_couple_jac_point_t *out,
                                  unsigned n,
                                  const theta_couple_jac_point_t *in,
                                  const theta_couple_curve_t *E1E2);

/**
 * @brief A forgetful function which returns (X : Z) points given a pair of (X : Y : Z) points
 *
 * @param P Output: the theta_couple_point
 * @param xyP : the theta_couple_jac_point
 **/
void couple_jac_to_xz(theta_couple_point_t *P, const theta_couple_jac_point_t *xyP);

#endif