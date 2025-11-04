#include "sqi_gluing_isogenies.h"
#include <assert.h>

/**
 * @brief Given T1 and T2 a basis of E[4], compute the symetric ellements above T1, T2
 * and T1*T2 that can be used to compute the theta structure induced by T1 and T2.
 *
 * @param basis Output: the symetric ellement basis of T1, T2
 * @param T1_4 a point in E[4]
 * @param T2_4 a point in E[4]
 *
 */
static uint8_t
compute_symmetric_element(symetric_element_basis_t *basis,
                          const ec_point_t *T1_4,
                          const ec_point_t *T2_4)
{
    fp2_t r1, r2, s1, s2, tmp1, tmp2;
    uint8_t pos = 255u;

    fp2_add(&r1, &T1_4->x, &T1_4->z);
    fp2_sub(&s1, &T1_4->x, &T1_4->z);
    fp2_add(&r2, &T2_4->x, &T2_4->z);
    fp2_sub(&s2, &T2_4->x, &T2_4->z);

    pos = (fp2_is_zero(&s1) ? 0u : pos);
    pos = (fp2_is_zero(&r1) ? 1u : pos);
    pos = (fp2_is_zero(&s2) ? 2u : pos);
    pos = (fp2_is_zero(&r2) ? 3u : pos);

    fp2_mul(&s2, &r1, &s2);    // s2 = (x1 + z1)(x2 - z2)
    fp2_mul_i(&r1, &s2, 1); // r1 = i(x1 + z1)(x2 - z2)
    fp2_mul(&r2, &r2, &s1);    // r2 = (x1 - z1)(x2 + z2)

    fp2_add(&s1, &r1, &r2);
    fp2_sub(&s2, &r1, &r2);
    pos = (fp2_is_zero(&s1) ? 4u : pos);
    pos = (fp2_is_zero(&s2) ? 5u : pos);

    fp2_copy(&tmp1, &T1_4->x);
    fp2_copy(&tmp2, &T1_4->z);
    fp2_select(&tmp1, &tmp1, &T2_4->x, -(pos == 0 || pos == 1));
    fp2_select(&tmp2, &tmp2, &T2_4->z, -(pos == 0 || pos == 1));
    fp2_sqr(&s1, &tmp1);
    fp2_sqr(&s2, &tmp2);
    fp2_sub(&basis->delta, &s1, &s2); // delta = x2^2 - z2^2
    fp2_add(&r1, &s1, &s2);           // r1 = x2^2 +z2^2
    fp2_mul(&r2, &tmp1, &tmp2);
    fp2_add(&r2, &r2, &r2); // r2 = 2x2z2

    // g1 = (-r1,-r2),(r2,r1)
    fp2_neg(&basis->g1.s00, &r1);
    fp2_copy(&basis->g1.s10, &r2);
    fp2_neg(&basis->g1.s01, &r2);

    // g2 = (0,-d),(-d,0)
    fp2_set_zero(&basis->g2.s00);
    fp2_neg(&basis->g2.s10, &basis->delta);
    // g2 = (0,d),(d,0)
    fp2_select(&basis->g2.s10, &basis->g2.s10, &basis->delta, -(pos & 1u));
    fp2_copy(&basis->g2.s01, &basis->g2.s10);

    // g1 = (-r1,-r2), (r2,r1); g2 = (0,-d),(-d,0)
    fp2_cswap(&basis->g1.s00, &basis->g2.s00, -(pos == 0u || pos == 1u));
    fp2_cswap(&basis->g1.s10, &basis->g2.s10, -(pos == 0u || pos == 1u));
    fp2_cswap(&basis->g1.s01, &basis->g2.s01, -(pos == 0u || pos == 1u));

    // g3 = (ir2,ir1),(-ir1,-ir2)
    uint32_t flg1 = (pos == 0u || pos == 3u || pos == 4u);
    // g3 = (-ir2,-ir1),(ir1,ir2)
    uint32_t flg2 = (pos == 1u || pos == 2u || pos == 5u);
    fp2_mul_i(&basis->g3.s00, &r2, flg1);
    fp2_mul_i(&basis->g3.s10, &r1, flg2);
    fp2_mul_i(&basis->g3.s01, &r1, flg1);

    // g2 = (-ir2,-ir1),(ir1,ir2); g3 = (0,-d),(-d,0)
    fp2_cswap(&basis->g2.s00, &basis->g3.s00, -(pos == 4u || pos == 5u));
    fp2_cswap(&basis->g2.s10, &basis->g3.s10, -(pos == 4u || pos == 5u));
    fp2_cswap(&basis->g2.s01, &basis->g3.s01, -(pos == 4u || pos == 5u));

    return pos;
}

//--------------
// NOT TESTED

/** 
void
compute_barycentric_components(add_components_t *add, const ec_point_t *P, const ec_point_t *Q, const ec_point_t *PmQ)
{
    fp2_t temp1, temp2, temp3, temp4;
    fp2_sub(&temp1, &P->x, &P->z);
    fp2_add(&temp2, &Q->x, &Q->z);
    fp2_mul(&temp1, &temp1, &temp2);

    fp2_add(&temp2, &P->x, &P->z);
    fp2_sub(&temp3, &Q->x, &Q->z);
    fp2_mul(&temp2, &temp2, &temp3);

    fp2_add(&temp3, &temp1, &temp2); // temp3 = lambda =  (xP-zP)(xQ+zQ)+(xP+zP)(xQ-zQ)
    fp2_sub(&temp1, &temp1, &temp2); // temp1 = mu =(xP-zP)(xQ+zQ)-(xP+zP)(xQ-zQ)

    fp2_mul(&temp4, &temp3, &PmQ->z);
    fp2_sqr(&temp4, &temp4); // temp4 = (z_m*lambda)^2

    fp2_mul(&add->w, &temp1, &PmQ->x); // w = x_m*mu
    fp2_sqr(&temp2, &add->w);          // temp2 = (x_m*mu)^2

    fp2_add(&add->u, &temp4, &temp2);
    fp2_sub(&add->v, &temp4, &temp2);

    fp2_mul(&add->w, &add->w, &temp1);  // w = x_m*mu^2
    fp2_mul(&add->w, &add->w, &PmQ->z); // w = z_m*x_m*mu^2
    fp2_add(&add->w, &add->w, &add->w); // w = 2z_m*x_m*mu^2
}
**/


// Returns 1 if the basis is as expected and 0 otherwise
// We only expect this to fail for malformed signatures, so
// do not require this to run in constant time.
static int
verify_two_torsion(const theta_couple_point_t *K1_2,
                   const theta_couple_point_t *K2_2,
                   const theta_couple_curve_t *E12)
{
    // First check if any point in K1_2 or K2_2 is zero, if they are then the points did not have
    // order 8 when we started gluing
    if (ec_is_zero(&K1_2->P1) | ec_is_zero(&K1_2->P2) | ec_is_zero(&K2_2->P1) | ec_is_zero(&K2_2->P2)) {
        return 0;
    }

    // Now ensure that P1, Q1 and P2, Q2 are independent. For points of order two this means
    // that they're not the same
    if (ec_is_equal(&K1_2->P1, &K2_2->P1) | ec_is_equal(&K1_2->P2, &K2_2->P2)) {
        return 0;
    }

    // Finally, double points to ensure all points have order exactly 0
    theta_couple_point_t O1, O2;
    double_couple_point(&O1, K1_2, E12);
    double_couple_point(&O2, K2_2, E12);
    // If this check fails then the points had order 2*f for some f, and the kernel is malformed.
    if (!(ec_is_zero(&O1.P1) & ec_is_zero(&O1.P2) & ec_is_zero(&O2.P1) & ec_is_zero(&O2.P2))) {
        return 0;
    }

    return 1;
}

// Given the appropriate four torsion, computes the
// change of basis to compute the correct theta null
// point.
// Returns 0 if the order of K1_4 or K2_4 is not 4
static uint8_t
gluing_basis_compute(gluing_basis_t *basis,
                     const theta_couple_point_t *K1_4,
                     const theta_couple_point_t *K2_4)
{
    // Compute the four 2x2 matrices for the action by translation
    // on the four points:

    symetric_element_basis_t Basis_1, Basis_2;
    uint8_t posi, posi2;
    fp2_t delta12, tmp1, tmp2, tmp3, M00, M01, M02, M03, M10, M20, Msgl;
    bool sgl1, sgl2, sgl3, nsgl;

    posi = compute_symmetric_element(&Basis_1, &K1_4->P1, &K2_4->P1);
    posi2 = compute_symmetric_element(&Basis_2, &K1_4->P2, &K2_4->P2);
    posi = (posi == 255u ? posi : 3u * (posi >> 1u) + (posi2 >> 1u));

    sgl1 = (posi == 0u);
    sgl2 = (posi == 4u);
    sgl3 = (posi == 8u);
    nsgl = !(sgl1 | sgl2 | sgl3);

    fp2_mul(&delta12, &Basis_1.delta, &Basis_2.delta);

    // trace for the first row
    fp2_copy(&M00, &delta12);
    fp2_mul(&tmp1, &Basis_1.g1.s00, &Basis_2.g1.s00);
    fp2_add(&M00, &M00, &tmp1);
    fp2_mul(&tmp1, &Basis_1.g2.s00, &Basis_2.g2.s00);
    fp2_add(&M00, &M00, &tmp1);
    fp2_mul(&tmp1, &Basis_1.g3.s00, &Basis_2.g3.s00);
    fp2_sub(&M00, &M00, &tmp1);

    fp2_mul(&M01, &Basis_1.g1.s00, &Basis_2.g1.s10);
    fp2_mul(&tmp1, &Basis_1.g2.s00, &Basis_2.g2.s10);
    fp2_add(&M01, &M01, &tmp1);
    fp2_mul(&tmp1, &Basis_1.g3.s00, &Basis_2.g3.s10);
    fp2_sub(&M01, &M01, &tmp1);

    fp2_mul(&M02, &Basis_1.g1.s10, &Basis_2.g1.s00);
    fp2_mul(&tmp1, &Basis_1.g2.s10, &Basis_2.g2.s00);
    fp2_add(&M02, &M02, &tmp1);
    fp2_mul(&tmp1, &Basis_1.g3.s10, &Basis_2.g3.s00);
    fp2_sub(&M02, &M02, &tmp1);

    fp2_mul(&M03, &Basis_1.g1.s10, &Basis_2.g1.s10);
    fp2_mul(&tmp1, &Basis_1.g2.s10, &Basis_2.g2.s10);
    fp2_add(&M03, &M03, &tmp1);
    fp2_mul(&tmp1, &Basis_1.g3.s10, &Basis_2.g3.s10);
    fp2_sub(&M03, &M03, &tmp1);

    // Compute the action of (0,out.K2_4.P2) for the second row
    fp2_mul(&M10, &Basis_2.g2.s00, &M00);
    fp2_mul(&tmp1, &Basis_2.g2.s01, &M01);
    fp2_add(&M10, &M10, &tmp1);

    fp2_mul(&Msgl, &Basis_2.g2.s10, &M00);
    fp2_mul(&tmp1, &Basis_2.g2.s00, &M01);
    fp2_sub(&Msgl, &Msgl, &tmp1);

    // compute the action of (K1_4.P1,0) for the third row
    fp2_mul(&M20, &Basis_1.g1.s00, &M00);
    fp2_mul(&tmp1, &Basis_1.g1.s01, &M02);
    fp2_add(&M20, &M20, &tmp1);

    // In the case of sgl2, we know that b1 is always false.
    fp2_mul(&tmp2, &Basis_1.g1.s00, &M01);
    fp2_mul(&tmp1, &Basis_1.g1.s10, &M03);
    fp2_sub(&tmp2, &tmp2, &tmp1);
    fp2_select(&Msgl, &Msgl, &tmp2, -sgl2);

    // Scalar realignment
    fp2_mul(&M00, &M00, &delta12);
    fp2_mul(&M01, &M01, &delta12);
    fp2_mul(&M02, &M02, &delta12);
    fp2_mul(&M03, &M03, &delta12);
    fp2_mul(&M10, &M10, &Basis_1.delta);
    fp2_mul(&M20, &M20, &Basis_2.delta);

    fp2_copy(&tmp1, &Basis_1.delta);
    fp2_select(&tmp1, &tmp1, &Basis_2.delta, -sgl2);
    fp2_mul(&Msgl, &Msgl, &tmp1);

    // Compute the Hadamard transform
    fp2_sqr(&basis->square_dual_point.t, &M00);                                                     // a^2
    fp2_sqr(&basis->square_dual_point.y, &M10);                                                     // b^2
    fp2_sqr(&basis->square_dual_point.z, &M20);                                                     // c^2
    fp2_add(&basis->square_dual_point.x, &basis->square_dual_point.y, &basis->square_dual_point.z); // B_0 = (b^2 + c^2)
    fp2_sub(&basis->square_dual_point.y, &basis->square_dual_point.t, &basis->square_dual_point.y); // B_1 = (a^2 - b^2)
    fp2_sub(&basis->square_dual_point.z, &basis->square_dual_point.t, &basis->square_dual_point.z); // B_2 = (a^2 - c^2)
    fp2_set_zero(&basis->square_dual_point.t);

    fp2_copy(&tmp2, &M10);
    fp2_select(&tmp2, &tmp2, &M20, -sgl2);
    fp2_select(&tmp2, &tmp2, &M02, -nsgl);
    fp2_select(&tmp3, &Msgl, &M03, -nsgl);
    fp2_mul(&tmp1, &M00, &M01);                        // M00M01
    fp2_mul(&tmp2, &tmp2, &tmp3);                      // M20M21
    fp2_add(&basis->components_01_23.x, &tmp1, &tmp2); // M00M01 + M20M21
    fp2_sub(&basis->components_01_23.y, &tmp1, &tmp2); // M00M01 - M20M21

    // Check that M01 = M02. Used for constant timeness.
    fp2_mul(&tmp3, &M00, &M02);
    fp2_sub(&tmp1, &tmp1, &tmp3);
    basis->sign_flip = !(bool)(fp2_is_zero(&tmp1));
    basis->sign_flip = (nsgl ? false : basis->sign_flip);
    fp2_neg(&tmp1, &tmp2);
    fp2_select(&tmp2, &tmp2, &tmp1, -basis->sign_flip);
    // If nsgl, overwrite operand with different value.
    fp2_mul(&tmp1, &M01, &M03);
    fp2_neg(&tmp1, &tmp1);
    fp2_select(&tmp2, &tmp2, &tmp1, -nsgl);
    fp2_add(&basis->components_02_13.y, &tmp3, &tmp2);
    fp2_sub(&basis->components_02_13.x, &tmp3, &tmp2);

    fp2_mul(&tmp1, &M00, &M03);
    fp2_mul(&tmp2, &M01, &M02);
    fp2_neg(&tmp3, &tmp2);
    // These components are set to zero if branch is not taken.
    fp2_select(&tmp2, &tmp2, &tmp3, (sgl1 | sgl2 | sgl3));
    fp2_add(&basis->components_03_12.x, &tmp1, &tmp2);
    fp2_select(&tmp2, &tmp2, &tmp1, (sgl1 | sgl2 | sgl3));
    fp2_sub(&basis->components_03_12.y, &tmp1, &tmp2);

    return posi;
}

static void
compute_terms(theta_point_t *image,
              const fp2_t uuvv_p_ww[2],
              const fp2_t uuvv_m_ww[2],
              const fp2_t uw[2],
              const fp2_t *vw12,
              const gluing_basis_t *basis,
              const uint8_t pos) {
    fp2_t bas[4], ops[9], res[3], tmp1, tmp2, tmp3;

    fp2_copy(&bas[0], &basis->components_01_23.x);
    fp2_select(&bas[0], &bas[0], &basis->components_02_13.y, -(pos == 2));
    fp2_select(&bas[0], &bas[0], &basis->components_02_13.y, -(pos == 5));
    fp2_select(&bas[0], &bas[0], &basis->components_01_23.y, -(pos == 6));
    fp2_select(&bas[0], &bas[0], &basis->components_01_23.y, -(pos == 7));

    fp2_copy(&bas[1], &basis->components_01_23.y);
    fp2_select(&bas[1], &bas[1], &basis->components_02_13.y, -(pos == 1));
    fp2_select(&bas[1], &bas[1], &basis->components_01_23.x, -(pos == 2));
    fp2_select(&bas[1], &bas[1], &basis->components_01_23.x, -(pos == 6));
    fp2_select(&bas[1], &bas[1], &basis->components_02_13.y, -(pos == 7));

    fp2_copy(&bas[2], &basis->components_01_23.x);
    fp2_select(&bas[2], &bas[2], &basis->components_01_23.y, -(pos == 1));
    fp2_select(&bas[2], &bas[2], &basis->components_01_23.y, -(pos == 2));
    fp2_select(&bas[2], &bas[2], &basis->components_02_13.y, -(pos == 3));
    fp2_select(&bas[2], &bas[2], &basis->components_02_13.y, -(pos == 6));

    fp2_copy(&ops[0], &uuvv_p_ww[1]);
    fp2_select(&ops[0], &ops[0], &uuvv_p_ww[0], -(pos == 2));
    fp2_select(&ops[0], &ops[0], &uuvv_p_ww[0], -(pos == 5));

    fp2_copy(&ops[1], &uw[1]);
    fp2_select(&ops[1], &ops[1], &uw[0], -(pos == 2));
    fp2_select(&ops[1], &ops[1], &uw[0], -(pos == 5));

    fp2_copy(&ops[2], &uuvv_p_ww[0]);
    fp2_select(&ops[2], &ops[2], &uuvv_m_ww[1], -(pos == 2));
    fp2_select(&ops[2], &ops[2], &uuvv_m_ww[1], -(pos == 5));
    fp2_select(&ops[2], &ops[2], &uuvv_m_ww[0], -(pos == 6));
    fp2_select(&ops[2], &ops[2], &uuvv_m_ww[0], -(pos == 7));

    fp2_copy(&ops[3], &uuvv_p_ww[1]);
    fp2_select(&ops[3], &ops[3], &uuvv_p_ww[0], -(pos == 1));
    fp2_select(&ops[3], &ops[3], &uuvv_p_ww[0], -(pos == 7));

    fp2_copy(&ops[4], &uw[1]);
    fp2_select(&ops[4], &ops[4], &uw[0], -(pos == 1));
    fp2_select(&ops[4], &ops[4], &uw[0], -(pos == 7));

    fp2_copy(&ops[5], &uuvv_p_ww[0]);
    fp2_select(&ops[5], &ops[5], &uuvv_m_ww[1], -(pos == 1));
    fp2_select(&ops[5], &ops[5], &uuvv_m_ww[0], -(pos == 3));
    fp2_select(&ops[5], &ops[5], &uuvv_m_ww[0], -(pos == 5));
    fp2_select(&ops[5], &ops[5], &uuvv_m_ww[1], -(pos == 7));

    fp2_copy(&ops[6], &uuvv_p_ww[1]);
    fp2_select(&ops[6], &ops[6], &uuvv_p_ww[0], -(pos == 3));
    fp2_select(&ops[6], &ops[6], &uuvv_p_ww[0], -(pos == 6));

    fp2_copy(&ops[7], &uw[1]);
    fp2_select(&ops[7], &ops[7], &uw[0], -(pos == 3));
    fp2_select(&ops[7], &ops[7], &uw[0], -(pos == 6));

    fp2_copy(&ops[8], &uuvv_p_ww[0]);
    fp2_select(&ops[8], &ops[8], &uuvv_m_ww[0], -(pos == 1));
    fp2_select(&ops[8], &ops[8], &uuvv_m_ww[0], -(pos == 2));
    fp2_select(&ops[8], &ops[8], &uuvv_m_ww[1], -(pos == 3));
    fp2_select(&ops[8], &ops[8], &uuvv_m_ww[1], -(pos == 6));

    fp2_select(&bas[2], &bas[2], &basis->components_01_23.y, -(pos == 4u));
    fp2_copy(&bas[3], &basis->components_03_12.y);
    fp2_select(&bas[3], &bas[3], &basis->square_dual_point.z, -(pos == 0u));
    fp2_select(&bas[3], &bas[3], &basis->square_dual_point.y, -(pos == 4u));
    fp2_select(&bas[3], &bas[3], &basis->square_dual_point.x, -(pos == 8u));

    // Compute first element
    fp2_mul(&tmp1, &ops[0], &basis->square_dual_point.x);
    fp2_mul(&tmp2, &ops[1], &bas[0]);
    fp2_add(&tmp1, &tmp1, &tmp2);
    fp2_mul(&image->x, &ops[2], &tmp1);

    // Compute second element
    fp2_mul(&tmp1, &ops[3], &basis->square_dual_point.y);
    fp2_mul(&tmp2, &ops[4], &bas[1]);
    fp2_add(&tmp1, &tmp1, &tmp2);
    fp2_mul(&image->y, &ops[5], &tmp1);

    // Compute third element
    fp2_mul(&tmp1, &ops[6], &basis->square_dual_point.z);
    fp2_mul(&tmp2, &ops[7], &bas[2]);
    fp2_add(&tmp1, &tmp1, &tmp2);
    fp2_mul(&image->z, &ops[8], &tmp1);

    // Compute the last point.
    fp2_mul(&image->t, vw12, &bas[3]);

    // Compute the adjusting values. Start with the cases pos % 4 != 0.
    fp2_copy(&bas[0], &basis->components_02_13.x);
    fp2_copy(&bas[1], &basis->components_03_12.x);

    // Correct based on remaining values of pos.
    fp2_select(&bas[0], &bas[0], &basis->components_01_23.y, -(pos == 0u));
    fp2_select(&bas[1], &bas[1], &basis->square_dual_point.y, -(pos == 0u));
    fp2_select(&bas[0], &bas[0], &basis->components_01_23.y, -(pos == 4u));
    fp2_select(&bas[1], &bas[1], &basis->square_dual_point.z, -(pos == 4u));
    fp2_select(&bas[0], &bas[0], &basis->components_01_23.x, -(pos == 8u));
    fp2_select(&bas[1], &bas[1], &basis->square_dual_point.z, -(pos == 8u));

    fp2_mul(&tmp1, &uuvv_p_ww[1], &bas[0]);
    fp2_mul(&tmp2, &uw[1], &bas[1]);
    fp2_add(&tmp1, &tmp1, &tmp2);
    fp2_mul(&tmp1, &uw[0], &tmp1);
    fp2_copy(&res[0], &tmp1);

    fp2_select(&bas[0], &bas[0], &basis->components_01_23.x, -(pos == 0u));
    fp2_select(&bas[1], &bas[1], &basis->square_dual_point.x, -(pos == 0u));
    fp2_select(&bas[0], &bas[0], &basis->components_01_23.x, -(pos == 4u));
    fp2_select(&bas[1], &bas[1], &basis->square_dual_point.x, -(pos == 4u));
    fp2_select(&bas[0], &bas[0], &basis->components_01_23.y, -(pos == 8u));
    fp2_select(&bas[1], &bas[1], &basis->square_dual_point.y, -(pos == 8u));

    fp2_mul(&tmp1, &uuvv_p_ww[1], &bas[0]);
    fp2_mul(&tmp2, &uw[1], &bas[1]);
    fp2_add(&tmp1, &tmp1, &tmp2);
    fp2_mul(&tmp1, &uw[0], &tmp1);
    fp2_copy(&res[1], &tmp1);

    // Handle special cases when pos % 8 = 0.
    fp2_mul(&tmp1, &uuvv_m_ww[1], &uuvv_m_ww[0]);
    fp2_mul(&res[2], &tmp1, &bas[3]);

    fp2_add(&tmp1, &image->x, &res[0]);
    fp2_add(&tmp2, &image->y, &res[0]);
    fp2_add(&tmp3, &image->z, &res[0]);

    // Now apply the adjustments. We first handle the regular cases before.
    fp2_select(&image->x, &image->x, &tmp1, -(pos == 1));
    fp2_select(&image->x, &image->x, &tmp1, -(pos == 3));
    fp2_select(&image->y, &image->y, &tmp2, -(pos == 2));
    fp2_select(&image->y, &image->y, &tmp2, -(pos == 6));
    fp2_select(&image->z, &image->z, &tmp3, -(pos == 5));
    fp2_select(&image->z, &image->z, &tmp3, -(pos == 7));

    fp2_select(&image->x, &image->x, &tmp1, -(pos == 0));
    fp2_select(&image->x, &image->x, &tmp1, -(pos == 4));
    fp2_select(&image->y, &image->y, &tmp2, -(pos == 8));

    fp2_add(&tmp2, &image->y, &res[1]);
    fp2_add(&tmp3, &image->z, &res[1]);

    fp2_select(&image->y, &image->y, &tmp2, -(pos == 0));
    fp2_select(&image->z, &image->z, &tmp3, -(pos == 4));
    fp2_select(&image->z, &image->z, &tmp3, -(pos == 8));

    fp2_select(&image->z, &image->z, &res[2], -(pos == 0u));
    fp2_select(&image->y, &image->y, &res[2], -(pos == 4u));
    fp2_select(&image->x, &image->x, &res[2], -(pos == 8u));
}

static uint32_t
gluing_formula_general(theta_point_t *image,
                       const add_components_t *add_comp1,
                       const add_components_t *add_comp2,
                       const gluing_basis_t *basis,
                       const uint8_t pos)
{
    // Perform the desired precomputation
    fp2_t uuvv_p_ww[2], uuvv_m_ww[2], uw[2], vw12, tmp1, tmp2, tmp3;

    // handles the sign flip.
    fp2_neg(&tmp3, &add_comp1->w);
    fp2_select(&tmp3, &add_comp1->w, &tmp3, -basis->sign_flip);

    fp2_add(&uuvv_p_ww[0], &add_comp1->u, &add_comp1->v);
    fp2_sub(&tmp1, &add_comp1->u, &add_comp1->v);
    fp2_mul(&uuvv_p_ww[0], &uuvv_p_ww[0], &tmp1); // uuvv1 = u1^2 -v1^2

    fp2_add(&uuvv_p_ww[1], &add_comp2->u, &add_comp2->v);
    fp2_sub(&tmp1, &add_comp2->u, &add_comp2->v);
    fp2_mul(&uuvv_p_ww[1], &uuvv_p_ww[1], &tmp1); // uuvv2 = u2^2 -v2^2

    fp2_sqr(&tmp1, &add_comp1->w); // ww1 = w1^2
    fp2_sqr(&tmp2, &add_comp2->w); // ww2 = w1^2

    fp2_sub(&uuvv_m_ww[0], &uuvv_p_ww[0], &tmp1); // uuvv_m_ww1 = u1^2 - w1^2
    fp2_add(&uuvv_p_ww[0], &uuvv_p_ww[0], &tmp1); // uuvv_p_ww1 = u1^2 + w1^2
    fp2_sub(&uuvv_m_ww[1], &uuvv_p_ww[1], &tmp2); // uuvv_m_ww2 = u2^2 - w2^2
    fp2_add(&uuvv_p_ww[1], &uuvv_p_ww[1], &tmp2); // uuvv_p_ww2 = u2^2 + w2^2

    fp2_mul(&uw[0], &add_comp1->u, &tmp3);         // uw1 = ±u1w1
    fp2_add(&uw[0], &uw[0], &uw[0]);               // uw1 = 2u1w1
    fp2_mul(&uw[1], &add_comp2->u, &add_comp2->w); // uw2 = u2w2
    fp2_add(&uw[1], &uw[1], &uw[1]);               // uw2 = 2u2w2

    fp2_mul(&vw12, &add_comp1->v, &add_comp2->v);
    fp2_mul(&tmp1, &tmp3, &add_comp2->w);
    fp2_mul(&vw12, &vw12, &tmp1); // vw12 = ±v1v2w1w2
    fp2_add(&vw12, &vw12, &vw12); // vw12 = ±2v1v2w1w2
    fp2_add(&vw12, &vw12, &vw12); // vw12 = ±4v1v2w1w2

    compute_terms(image, uuvv_p_ww, uuvv_m_ww, uw, &vw12, basis, pos);

    return 1;
}

static uint32_t
gluing_formula_special(theta_point_t *image,
                       const theta_couple_point_t *P,
                       const gluing_basis_t *basis,
                       const uint8_t pos)
{
    // Perform the desired precomputation
    fp2_t uuvv_p_ww[2], uuvv_m_ww[2], uw[2], tmp1, tmp2, tmp3;

    // handles the sign flip.
    fp2_neg(&tmp3, &P->P1.z); // tpm3 = -w1
    fp2_select(&tmp3, &P->P1.z, &tmp3, -basis->sign_flip);

    fp2_sqr(&uuvv_p_ww[0], &P->P1.x); // uuvv_p_ww1 = u1^2
    fp2_sqr(&uuvv_p_ww[1], &P->P2.x); // uuvv_p_ww2 = u2^2
    fp2_sqr(&tmp1, &P->P1.z);         // ww1 = w1^2
    fp2_sqr(&tmp2, &P->P2.z);         // ww2 = w1^2

    fp2_sub(&uuvv_m_ww[0], &uuvv_p_ww[0], &tmp1); // uuvv_m_ww1 = u1^2 - w1^2
    fp2_add(&uuvv_p_ww[0], &uuvv_p_ww[0], &tmp1); // uuvv_p_ww1 = u1^2 + w1^2
    fp2_sub(&uuvv_m_ww[1], &uuvv_p_ww[1], &tmp2); // uuvv_m_ww2 = u2^2 - w2^2
    fp2_add(&uuvv_p_ww[1], &uuvv_p_ww[1], &tmp2); // uuvv_p_ww2 = u2^2 + w2^2

    fp2_mul(&uw[0], &P->P1.x, &tmp3);    // uw1 = ±u1w1
    fp2_add(&uw[0], &uw[0], &uw[0]);     // uw1 = ±2u1w1
    fp2_mul(&uw[1], &P->P2.x, &P->P2.z); // uw2 = u2w2
    fp2_add(&uw[1], &uw[1], &uw[1]);     // uw2 = 2u2w2

    fp2_set_zero(&tmp1);
    compute_terms(image, uuvv_p_ww, uuvv_m_ww, uw, &tmp1, basis, pos);

    return 1;
}

int
gluing_compute(theta_gluing_t *out,
               const theta_couple_curve_t *E12,
               const theta_couple_jac_point_t *xyK1_8,
               const theta_couple_jac_point_t *xyK2_8,
               bool verify)
{
    // Ensure that we have been given the eight torsion
#ifndef NDEBUG
    {
        int check = test_jac_order_twof(&xyK1_8->P1, &E12->E1, 3);
        if (!check)
            printf("xyK1_8->P1 does not have order 8\n");
        check = test_jac_order_twof(&xyK2_8->P1, &E12->E1, 3);
        if (!check)
            printf("xyK2_8->P1 does not have order 8\n");
        check = test_jac_order_twof(&xyK1_8->P2, &E12->E2, 3);
        if (!check)
            printf("xyK2_8->P1 does not have order 8\n");
        check = test_jac_order_twof(&xyK2_8->P2, &E12->E2, 3);
        if (!check)
            printf("xyK2_8->P2 does not have order 8\n");
    }
#endif

    out->xyK1_8 = *xyK1_8;
    out->domain = *E12;

    // Given points in E[8] x E[8] we need the four torsion below
    theta_couple_jac_point_t xyK1_4, xyK2_4;

    double_couple_jac_point(&xyK1_4, xyK1_8, E12);
    double_couple_jac_point(&xyK2_4, xyK2_8, E12);

    // Convert from (X:Y:Z) coordinates to (X:Z)
    theta_couple_point_t K1_8, K2_8;
    theta_couple_point_t K1_4, K2_4;

    couple_jac_to_xz(&K1_8, xyK1_8);
    couple_jac_to_xz(&K2_8, xyK2_8);
    couple_jac_to_xz(&K1_4, &xyK1_4);
    couple_jac_to_xz(&K2_4, &xyK2_4);

    // Compute points of order 2 from Ki_4
    theta_couple_point_t K1_2, K2_2;
    double_couple_point(&K1_2, &K1_4, E12);
    double_couple_point(&K2_2, &K2_4, E12);

    if (!verify_two_torsion(&K1_2, &K2_2, E12)) {
        return 0;
    }

    // Set the basis change matrix, if we have not been given a valid K[8] for this computation
    // gluing_change_of_basis will detect this and return 0
    out->position = gluing_basis_compute(&out->basis, &K1_4, &K2_4);

    if (out->position == 255u) {
        return 0;
    }

    // apply the base change to the kernel
    theta_point_t TT1, TT2;

    gluing_formula_special(&TT1, &K1_8, &out->basis, out->position);
    gluing_formula_special(&TT2, &K2_8, &out->basis, out->position);

    // If the kernel is well formed then TT1.t and TT2.t are zero
    // if they are not, we exit early as the signature we are validating
    // is probably malformed
    if (!(fp2_is_zero(&TT1.t) & fp2_is_zero(&TT2.t))) {
        printf("gluing failed TT1.t or TT2.t is not zero\n");
        return 0;
    }
    // Test our projective factors are non zero
    if (fp2_is_zero(&TT1.x) | fp2_is_zero(&TT2.x) | fp2_is_zero(&TT1.y) | fp2_is_zero(&TT2.z) | fp2_is_zero(&TT1.z))
        return 0; // invalid input

    // Projective factor: Ax
    fp2_mul(&out->codomain.x, &TT1.x, &TT2.x);
    fp2_mul(&out->codomain.y, &TT1.y, &TT2.x);
    fp2_mul(&out->codomain.z, &TT1.x, &TT2.z);
    fp2_set_zero(&out->codomain.t);
    // Projective factor: ABCxz
    fp2_mul(&out->precomputation.x, &TT1.y, &TT2.z);
    fp2_copy(&out->precomputation.y, &out->codomain.z);
    fp2_copy(&out->precomputation.z, &out->codomain.y);
    fp2_set_zero(&out->precomputation.t);

    // Compute the two components of phi(K1_8) = (x:x:y:y).
    fp2_mul(&out->imageK1_8.x, &TT1.x, &out->precomputation.x);
    fp2_mul(&out->imageK1_8.y, &TT1.z, &out->precomputation.z);

    // If K1_8 and K2_8 are our 8-torsion points, this ensures that the
    // 4-torsion points [2]K1_8 and [2]K2_8 are isotropic.
    if (verify) {
        fp2_t t1, t2;
        fp2_mul(&t1, &TT1.y, &out->precomputation.y);
        if (!fp2_is_equal(&out->imageK1_8.x, &t1))
            return 0;
        fp2_mul(&t1, &TT2.x, &out->precomputation.x);
        fp2_mul(&t2, &TT2.z, &out->precomputation.z);
        if (!fp2_is_equal(&t2, &t1))
            return 0;
    }

    // compute the final codomain
    hadamard(&out->codomain, &out->codomain);
    return 1;
}

// sub routine of the gluing eval
static void
gluing_eval_point(theta_point_t *image,
                  const theta_couple_jac_point_t *P,
                  const theta_gluing_t *phi)
{
    // theta_point_t T1;
    add_components_t add_comp1, add_comp2;

    // Compute the cross addition components of P1+Q1 and P2+Q2
    jac_to_xz_add_components(&add_comp1, &P->P1, &phi->xyK1_8.P1, &phi->domain.E1);
    jac_to_xz_add_components(&add_comp2, &P->P2, &phi->xyK1_8.P2, &phi->domain.E2);

    gluing_formula_general(image, &add_comp1, &add_comp2, &phi->basis, phi->position);

    // Compute (x, y, z, t)
    // As imageK1_8 = (x:x:y:y), its inverse is (y:y:x:x).
    fp2_mul(&image->x, &image->x, &phi->imageK1_8.y);
    fp2_mul(&image->y, &image->y, &phi->imageK1_8.y);
    fp2_mul(&image->z, &image->z, &phi->imageK1_8.x);
    fp2_mul(&image->t, &image->t, &phi->imageK1_8.x);

    hadamard(image, image);
}

void
gluing_eval_basis(theta_point_t *image1,
                  theta_point_t *image2,
                  const theta_couple_jac_point_t *xyT1,
                  const theta_couple_jac_point_t *xyT2,
                  const theta_gluing_t *phi)
{
    gluing_eval_point(image1, xyT1, phi);
    gluing_eval_point(image2, xyT2, phi);
}

// Same as gluing_eval_point but in the very special case where we already know that the point will
// have a zero coordinate at the place where the zero coordinate of the dual_theta_nullpoint would
// have made the computation difficult
int
gluing_eval_point_special_case(theta_point_t *image,
                               const theta_couple_point_t *P,
                               const theta_gluing_t *phi)
{
    gluing_formula_special(image, P, &phi->basis, phi->position);

    // Compute (x, y, z, t)
    fp2_mul(&image->x, &image->x, &phi->precomputation.x);
    fp2_mul(&image->y, &image->y, &phi->precomputation.y);
    fp2_mul(&image->z, &image->z, &phi->precomputation.z);
    // fp2_set_zero(&image->t);

    hadamard(image, image);
    return 1;
}

// To plug in with the AVX code.
void sqi_gluing_step(__m512i *OA, 
                     __m512i *P, 
                     __m512i *Q,
                     unsigned n,
                     theta_couple_curve_t *E12,
                     const theta_kernel_couple_points_t *ker,
                     bool extra_torsion,
                     bool verify,
                     bool randomize
) 
{
    theta_structure_t theta;

    // lift the basis
    theta_couple_jac_point_t xyT1, xyT2;

    ec_basis_t bas1 = { .P = ker->T1.P1, .Q = ker->T2.P1, .PmQ = ker->T1m2.P1 };
    ec_basis_t bas2 = { .P = ker->T1.P2, .Q = ker->T2.P2, .PmQ = ker->T1m2.P2 };
    if (!lift_basis(&xyT1.P1, &xyT2.P1, &bas1, &E12->E1))
        printf("lift basis E1 failed\n");
        return;
    if (!lift_basis(&xyT1.P2, &xyT2.P2, &bas2, &E12->E2))
        printf("lift basis E2 failed\n");
        return;

    const unsigned extra = HD_extra_torsion * extra_torsion;

#ifndef NDEBUG
    assert(extra == 0 || extra == 2); // only cases implemented
    if (!test_point_order_twof(&bas2.P, &E12->E2, n + extra))
        printf("bas2.P does not have correct order");

    if (!test_jac_order_twof(&xyT2.P2, &E12->E2, n + extra))
        printf("xyT2.P2 does not have correct order");
#endif

    int space = 1;
    for (unsigned i = 1; i < n; i *= 2)
        ++space;

    uint16_t todo[space];
    todo[0] = n - 2 + extra;

    int current = 0;

    // kernel points for the gluing isogeny
    theta_couple_jac_point_t jacQ1[space], jacQ2[space];
    jacQ1[0] = xyT1;
    jacQ2[0] = xyT2;
    while (todo[current] != 1) {
        assert(todo[current] >= 2);
        ++current;
        assert(current < space);
        // the gluing isogeny is quite a bit more expensive than the others,
        // so we adjust the usual splitting rule here a little bit: towards
        // the end of the doubling chain it will be cheaper to recompute the
        // doublings after evaluation than to push the intermediate points.
        const unsigned num_dbls = todo[current - 1] >= 16 ? todo[current - 1] / 2 : todo[current - 1] - 1;
        assert(num_dbls && num_dbls < todo[current - 1]);
        double_couple_jac_point_iter(&jacQ1[current], num_dbls, &jacQ1[current - 1], E12);
        double_couple_jac_point_iter(&jacQ2[current], num_dbls, &jacQ2[current - 1], E12);
        todo[current] = todo[current - 1] - num_dbls;
    }

    // kernel points for the remaining isogeny steps
    theta_point_t thetaQ1[space], thetaQ2[space];


    // the gluing step
    theta_gluing_t first_step;
    {
        assert(todo[current] == 1);

        // compute the gluing isogeny
        if (!gluing_compute(&first_step, E12, &jacQ1[current], &jacQ2[current], verify))
            printf("gluing compute failed\n");
            return;

        // push kernel points through gluing isogeny
        for (int j = 0; j < current; ++j) {
            gluing_eval_basis(&thetaQ1[j], &thetaQ2[j], &jacQ1[j], &jacQ2[j], &first_step);
            --todo[j];
        }

        --current;
    }


    fp2_t *codomain = (fp2_t*) &first_step.codomain;
    fp2_unvectorized_to_4x2x1(OA, codomain);

    /* WIP:
    *   You need to figure out if multiple points are output in this gluing.
    *   Extend your avx chain code to handle evaluation of additional points.
    *   Find KAT corresponding to what you have used for rest of chain, to give input to gluing.
    *   Might be a lot of compile issues due to missing functions for the gluing.
    */  


    // set-up the theta_structure for the first codomain
    theta.null_point = first_step.codomain;
}

