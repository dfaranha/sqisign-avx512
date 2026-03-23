/**
* The randomness used is not cryptographically secure.
* These functions should be used for testing purposes only.
*/

#ifndef _UTILS_H
#define _UTILS_H

#include <stdint.h>
#include "fp_lvl3.h"
#include "intrinsics_avx512.h"

void rand_bytes(uint8_t* bytes, int len);
uint64_t rand_uint64();

// Samples uniformly from [min, max].
// nbits is the number of bits of max.
uint64_t rand_in_range(const uint64_t min, const uint64_t max, const int nbits);
__m512i rand_512i();
void rand_fp_8x1(__m512i* res);
void rand_fp_4x2(__m512i* res);
void rand_fp2_8x1x1(__m512i* res);
void rand_fp2_4x1x2(__m512i* res);
void rand_fp2_4x2x1(__m512i* res);
void rand_fp2_2x2x2(__m512i* res);
void rand_fp2_2x4x1(__m512i* res);

void print_512i(__m512i a);
void print_512i_array(__m512i* a, int n, char* name);
void print_fp_8x1_index(__m512i* a, int index);
void print_fp_4x2_index(__m512i* a, int index);
void print_fp2_2x4x1(__m512i* a);
void print_fp2_4x2x1(__m512i* a);
void print_fp2_8x1x1(__m512i* a);

#endif