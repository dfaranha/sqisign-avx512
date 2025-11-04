#ifndef SQI_THETA_STRUCTURE
#define SQI_THETA_STRUCTURE

#include "sqi_fp2.h"
#include "sqi_hd.h"

/**
 * @brief Perform the hadamard transform on a theta point
 *
 * @param out Output: the theta_point
 * @param in a theta point*
 * in = (x,y,z,t)
 * out = (x+y+z+t, x-y+z-t, x+y-z-t, x-y-z+t)
 *
 */
static inline void hadamard(theta_point_t *out, const theta_point_t *in)
{
    fp2_t t1, t2, t3, t4;

    // t1 = x + y
    fp2_add(&t1, &in->x, &in->y);
    // t2 = x - y
    fp2_sub(&t2, &in->x, &in->y);
    // t3 = z + t
    fp2_add(&t3, &in->z, &in->t);
    // t4 = z - t
    fp2_sub(&t4, &in->z, &in->t);

    fp2_add(&out->x, &t1, &t3);
    fp2_add(&out->y, &t2, &t4);
    fp2_sub(&out->z, &t1, &t3);
    fp2_sub(&out->t, &t2, &t4);
}

#endif