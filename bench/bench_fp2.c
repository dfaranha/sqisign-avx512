#include <stdio.h>
#include "bench.h"
#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "utils.h"

#define N_RUNS 10

void bench_fp2(int iterations)
{
    uint64_t cycles[N_RUNS];
    uint64_t start, end, tmp;
    __m512i a[16], b[16], c[16];
    rand_fp2_8x1x1(a);
    rand_fp2_8x1x1(b);
    rand_fp2_8x1x1(c);

    printf("Fp2 lvl3 8x1x1 benchmarks:\n");

    // Addition
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_add_8x1x1(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("addition", cycles[(N_RUNS/2) -1], tmp);

    // Subtraction
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_sub_8x1x1(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("subtraction", cycles[(N_RUNS/2) -1], tmp);

    // Negation
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_neg_8x1x1(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("negation", cycles[(N_RUNS/2) -1], tmp);

    // Multiplication
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_mul_8x1x1(c, b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(c, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("multiplication", cycles[(N_RUNS/2) -1], tmp);

    // Squaring
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_sqr_8x1x1(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("squaring", cycles[(N_RUNS/2) -1], tmp);
}