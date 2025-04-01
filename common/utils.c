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

void print_fp_8x1_index(__m512i* a, int index)
{
    uint64_t vals[8];
    for(int i = 0; i < 8; i++) {
        memcpy(vals, &a[i], sizeof(vals));
        printf("%d: %lx   ", i, vals[index]);
    }
    printf("\n");
}

void print_fp_4x2_index(__m512i* a, int index)
{
    uint64_t vals[8];
    for(int i = 0; i < 4; i++) {
        memcpy(vals, &a[i], sizeof(vals));
        printf("%d: %lx   ", i, vals[2*index]);
    }
    for(int i = 0; i < 4; i++) {
        memcpy(vals, &a[i], sizeof(vals));
        printf("%d: %lx   ", i+4, vals[2*index+1]);
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

void rand_fp_4x2(__m512i* res)
{
    __m512i mask, zero[4];
    mask = VSETALL(LIMB_MASK);
    fp_setzero_4x2(zero);

    for(int i = 0; i < 3; i++) {
        res[i] = VAND(rand_512i(), mask);
    }
    // Last limb is shorter
    mask = VSET(LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK, LIMB_MASK, LAST_LIMB_MASK);
    res[3] = VAND(rand_512i(), mask);
    //Reduction
    fp_add_4x2(res, res, zero);
}

void rand_fp2_8x1x1(__m512i* res)
{
    rand_fp_8x1(res);
    rand_fp_8x1(res+8);
}