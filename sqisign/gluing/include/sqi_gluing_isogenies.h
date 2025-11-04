/** @file
 *
 * @authors Max Duparc
 *
 * @brief the gluing isogeny header
 */

#ifndef SQI_GLUING_ISOGENY_H
#define SQI_GLUING_ISOGENY_H

#include "fp2_lvl3.h"

#include "sqi_ec.h"
#include "sqi_fp2.h"
#include "sqi_theta_structure.h"
#include "sqi_hd.h"
 
 /** @internal
  * @ingroup hd_module
  * @defgroup Functions for gluing isogenies
  * @{
  */
 
 /**
  * @brief Compute the gluing isogeny from an elliptic product
  *
  * @param out Output: the theta_gluing
  * @param K1_8 a couple point
  * @param E12 an elliptic curve product
  * @param K2_8 a point in E2[8]
  *
  * out : E1xE2 -> A of kernel [4](K1_8,K2_8)
  * if the kernel supplied has the incorrect order, or gluing seems malformed,
  * returns 0, otherwise returns 1.
  */
 int gluing_compute(theta_gluing_t *out,
                    const theta_couple_curve_t *E12,
                    const theta_couple_jac_point_t *xyK1_8,
                    const theta_couple_jac_point_t *xyK2_8,
                    bool verify);
 
 /*
  * @brief Evaluate a gluing isogeny from an elliptic product on a basis
  *
  * @param image1 Output: the theta_point of the image of the first couple of points
  * @param image2 Output : the theta point of the image of the second couple of points
  * @param xyT1: A pair of points (X : Y : Z) on E1E2 to glue using phi
  * @param xyT2: A pair of points (X : Y : Z) on E1E2 to glue using phi
  * @param phi : a gluing isogeny E1 x E2 -> A
  *
  **/
 void gluing_eval_basis(theta_point_t *image1,
                        theta_point_t *image2,
                        const theta_couple_jac_point_t *xyT1,
                        const theta_couple_jac_point_t *xyT2,
                        const theta_gluing_t *phi);
 
 // Same as gluing_eval_point but in the very special case where we already know that the point will
 // have a zero coordinate at the place where the zero coordinate of the dual_theta_nullpoint would
 // have made the computation difficult
 int gluing_eval_point_special_case(theta_point_t *image,
                                    const theta_couple_point_t *P,
                                    const theta_gluing_t *phi);
 
 #endif
 

