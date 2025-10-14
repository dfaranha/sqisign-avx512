/*
 * Marius: For simplicity, I use the randomness generator in common instead of the one in SQIsign.
 */

#include "test_utils.h"
#include <utils.h>

// Make n random-ish field elements (for tests only!).
void
fp_random_test(fp_t *a)
{
    uint8_t tmp[FP_ENCODED_BYTES];

    for(int i = 0; i < FP_ENCODED_BYTES; i++) {
        tmp[i] = (uint8_t) (rand_uint64() % 256);
    }

    fp_decode_reduce(a, tmp, sizeof(tmp));
}

void
fp2_random_test(fp2_t *a)
{
    fp_random_test(&(a->re));
    fp_random_test(&(a->im));
}

int
cmp_u64(const void *v1, const void *v2)
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