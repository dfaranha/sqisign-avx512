#include <stdio.h>
#include <string.h>
#include "utils.h"
#include "fp_unvec.h"
#include "fp_consts.h"
#include "fp2.h"
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
    for(int i = 0; i < NUM_LIMBS; i++) {
        memcpy(vals, &a[i], sizeof(vals));
        printf("%d: %lx   ", i, vals[index]);
    }
    printf("\n");
}

void print_fp_4x2_index(__m512i* a, int index)
{
    uint64_t vals[8];
    for(int i = 0; i < N4x2; i++) {
        memcpy(vals, &a[i], sizeof(vals));
        printf("%d: %lx   ", i, vals[2*index]);
    }
    int rem4x2 = NUM_LIMBS % 2;
    for(int i = 0; i < N4x2-rem4x2; i++) {
        memcpy(vals, &a[i], sizeof(vals));
        printf("%d: %lx   ", i+N4x2, vals[2*index+1]);
    }
    printf("\n");
}

// void print_fp_2x4_index(__m512i* a, int index)
// {
//     uint64_t v0[8], v1[8];
//     memcpy(v0, &a[0], sizeof(v0));
//     memcpy(v1, &a[1], sizeof(v1));
//     for(int i = 0; i < 4; i++) {
//         printf("%d: %lx   ", 2*i, v0[4*index + i]);
//         printf("%d: %lx   ", 2*i+1, v1[4*index + i]);
//     }
//     printf("\n");
// }

void print_fp2_2x2x2(__m512i* a)
{
    fp2_t unvec[2];
    fp2_2x2x2_to_unvectorized(unvec, a);
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