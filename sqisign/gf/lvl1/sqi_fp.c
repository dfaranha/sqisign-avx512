#include <assert.h>
#include "sqi_fp.h"
#include "sqi_fp_constants.h"

// p = 3 * 2^324 - 1; both arrays are read by sqi_fp_asm.S
const digit_t p[NWORDS_FIELD] = { 0xffffffffffffffff, 0xffffffffffffffff, 0xffffffffffffffff,
                                  0xffffffffffffffff, 0xffffffffffffffff, 0x000000000000002f };
const digit_t p2[NWORDS_FIELD] = { 0xfffffffffffffffe, 0xffffffffffffffff, 0xffffffffffffffff,
                                   0xffffffffffffffff, 0xffffffffffffffff, 0x000000000000005f };

void
fp_sqrt(fp_t *x)
{
    (void)gf3324_sqrt(x, x);
}

uint32_t
fp_is_square(const fp_t *a)
{
    // ls is (0, 1, -1) and we want fp_is_square
    // to return 0xFF..FF when ls is 1 or 0 and 0x00..00 otherwise
    int32_t ls = gf3324_legendre(a);
    return ~(uint32_t)(ls >> 1);
}

void
fp_inv(fp_t *x)
{
    (void)gf3324_invert(x, x);
}

void
fp_exp3div4(fp_t *a)
{
    gf3324_exp3div4(a);
}
