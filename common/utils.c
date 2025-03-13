#include <stdio.h>
#include <string.h>
#include "utils.h"

uint64_t rand_uint64() 
{
	uint64_t r = 0;
	for (int i=0; i<64; i++) {
		r = r*2 + rand()%2;
	}
	return r;
}

__m512i rand_512i() 
{
    return VSET(rand_uint64(), rand_uint64(), rand_uint64(), rand_uint64(), rand_uint64(), rand_uint64(), rand_uint64(), rand_uint64());
}

void print_512i(__m512i a) 
{
    uint64_t vals[8];
    memcpy(vals, &a, sizeof(vals));
    for(int i = 0; i < 8; i++) {
        printf("%d: %lx   ", i, vals[i]);
    }
    printf("\n");
}

void rand_fp_8x1(__m512i* res)
{
    __m512i mask, zero[8];
    mask = VSETALL(LIMB_MASK);
    fp_setzero_8x1(zero);

    for(int i = 0; i < 7; i++) {
        res[i] = VAND(rand_512i(), mask);
    }
    // Last limb is shorter
    mask = VSETALL(LAST_LIMB_MASK);
    res[7] = VAND(rand_512i(), mask);
    //Reduction
    fp_add_8x1(res, res, zero);
}