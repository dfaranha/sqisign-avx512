#include <stdio.h>
#include <string.h>
#include "utils.h"

#include "fp2_lvl3.h"
#include "sqi_test_extras.h"

// TODO: Might not be cryptographically secure randomness.

void rand_bytes(uint8_t* bytes, int len)
{
    for (int i = 0; i < len; i++) {
        bytes[i] = 0;
        for (int j = 0; j < 8; j++) {
            bytes[i] *= 2;
            bytes[i] += rand()%2;
        }
    }
}

uint64_t rand_uint64() 
{
	uint64_t r = 0;
	for (int i=0; i<64; i++) {
		r *= 2;
        r += rand() % 2;
	}
	return r;
}

uint64_t rand_in_range(const uint64_t min, const uint64_t max, const int nbits)
{
    uint64_t r;
    do {
        r = 0;
        for (int i = 0; i < nbits; i++) {
            r = r*2 + rand()%2;
        }
    } while (r < min || r > max );
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
    printf("{ ");
    for (int i = 0; i < 8; i++) {
        printf("%lx", vals[i]);
        if(i < 7) {
            printf(", ");
        }
    }
    printf(" }\n");
}

void print_512i_array(__m512i* a, int n, char* name) {
    uint64_t vals[8];
    printf("%s = {\n", name);
    for (int i = 0; i < n; i++) {
        memcpy(vals, a+i, sizeof(vals));
        printf("    { ");
        for (int j = 0; j < 8; j++) {
            printf("0x%013lX", vals[j]);
            if(j < 7) {
                printf(", ");
            }
        }
        printf(" },\n");
    }
    printf("}\n");
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

void rand_fp2_4x1x2(__m512i* res)
{
    rand_fp_4x2(res);
    rand_fp_4x2(res+4);
}

void rand_fp2_4x2x1(__m512i* res)
{
    rand_fp_8x1(res);
}

void rand_fp2_2x2x2(__m512i* res)
{
    rand_fp_4x2(res);
}

void rand_fp2_2x4x1(__m512i* res)
{
    rand_fp_8x1(res);
    __m512i zero = VZERO();
    for(int i = 0; i < 8; i++) {
        res[i] = VBLEND(0x55, zero, res[i]);
    }
}

void print_fp2_2x2x2(__m512i* a)
{
    fp2_t unvec[2];
    fp2_2x4x1_to_unvectorized(unvec, a);
    for (int i = 0; i < 2; i++) {
        print_fp2(unvec+i);
    }
}

void print_fp2_2x4x1(__m512i* a) 
{
    fp2_t unvec[2];
    fp2_2x4x1_to_unvectorized(unvec, a);
    for(int i = 0; i < 2; i++) {
        print_fp2(unvec+i);
    }
}

void print_fp2_4x2x1(__m512i* a) 
{
    fp2_t unvec[4];
    fp2_4x2x1_to_unvectorized(unvec, a);
    for(int i = 0; i < 4; i++) {
        print_fp2(unvec+i);
    }
}

void print_fp2_8x1x1(__m512i* a) 
{
    fp2_t unvec[8];
    fp2_8x1x1_to_unvectorized(unvec, a);
    for(int i = 0; i < 8; i++) {
        print_fp2(unvec+i);
    }
}