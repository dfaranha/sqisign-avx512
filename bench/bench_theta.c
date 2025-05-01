#include <stdio.h>
#include "bench.h"
#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "theta.h"
#include "utils.h"

#define N_RUNS 10

void bench_theta(int iterations)
{
    uint64_t cycles[N_RUNS];
    uint64_t start, end, tmp;
    __m512i a[16], b[16], c[16];
    rand_fp2_8x1x1(a);
    rand_fp2_8x1x1(b);
    rand_fp2_8x1x1(c);

    printf("Theta single benchmarks:\n");

    // Hadamard
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_had_single(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("Hadamard", cycles[(N_RUNS/2) -1], tmp);

    printf("Theta pair benchmarks:\n");

    // Hadamard
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_had_pair(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("Hadamard", cycles[(N_RUNS/2) -1], tmp);

    // Hadamard
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_had_pair_alt(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("Hadamard alt", cycles[(N_RUNS/2) -1], tmp);
}