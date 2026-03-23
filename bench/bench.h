#ifndef _BENCH_H
#define _BENCH_H

#include <stdint.h>

void bench_fp(int iterations);
void bench_fp2(int iterations);
void bench_ec(int iterations);
void bench_theta(int iterations);
void bench_verif();

// Comparison of uint64 for qsort
int cmp_uint64(const void *v1, const void *v2);
void print_bench_res(char * title, uint64_t cycles, uint64_t tmp);

static inline uint64_t cpucycles(void)
{
    uint32_t hi, lo;
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return ((uint64_t)lo) | ((uint64_t)hi << 32);
}

#endif