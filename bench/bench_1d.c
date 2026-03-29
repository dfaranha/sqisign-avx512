#include <stdio.h>
#include "isog-1d.h"
#include "utils.h"
#include "bench.h"
#include "fp2_lvl3.h"

#define N_RUNS 10

void bench_1d_isog(int iterations)
{
    uint64_t cycles[N_RUNS];
    uint64_t start, end, tmp;
    __m512i A24[8], a[8], b[8], c[8], d[8], eval_data[16];

    printf("1d isog 2x2x2 benchmarks:\n");

    rand_fp2_2x2x2(a);
    rand_fp2_2x2x2(b);

    // Codomain
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            four_isog_codom_1x2x2x2(c, eval_data, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x2x2(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("codom", cycles[(N_RUNS/2) -1], tmp);

    // Eval
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            four_isog_eval_1x2x2x2(c, eval_data, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x2x2(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("eval", cycles[(N_RUNS/2) -1], tmp);

    printf("1d isog 2x4x1 benchmarks:\n");

    rand_fp2_2x4x1(a);
    rand_fp2_2x4x1(b);

    // Codomain
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            four_isog_codom_1x2x4x1(c, eval_data, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x4x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("codom", cycles[(N_RUNS/2) -1], tmp);

    // Eval
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            four_isog_eval_1x2x4x1(c, eval_data, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x4x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("eval", cycles[(N_RUNS/2) -1], tmp);
}