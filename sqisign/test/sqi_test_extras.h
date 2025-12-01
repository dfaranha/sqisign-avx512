#ifndef SQI_TEST_EXTRAS_H
#define SQI_TEST_EXTRAS_H

#include "sqi_ec.h"

#define PASSED 0
#define FAILED 1

int cmp_u64(const void *v1, const void *v2);

void fp_random_test(fp_t *a);
void fp2_random_test(fp2_t *a);

void ec_random_normalized_test(ec_point_t *P, const ec_curve_t *curve);
void ec_random_test(ec_point_t *P, const ec_curve_t *curve);

void projective_difference_point(ec_point_t *PQ, const ec_point_t *P, const ec_point_t *Q, const ec_curve_t *curve);


#endif