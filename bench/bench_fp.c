#include <stdio.h>
#include "bench.h"
#include "fp.h"
#include "fp_consts.h"
#include "utils.h"

#define N_RUNS 10

void bench_fp(int iterations) 
{
    int iterations_inv = 20;
    uint64_t cycles[N_RUNS];
    uint64_t start, end, tmp;
    __m512i a[NUM_LIMBS], b[NUM_LIMBS], c[NUM_LIMBS];
    rand_fp_8x1(a);
    rand_fp_8x1(b);
    rand_fp_8x1(c);

    printf("Fp lvl3 8x1 benchmarks:\n");

    // Addition
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_add_8x1(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_8x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("addition", cycles[(N_RUNS/2) -1], tmp);

    // Subtraction
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_sub_8x1(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_8x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("subtraction", cycles[(N_RUNS/2) -1], tmp);

    // Negation
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_neg_8x1(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_8x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("negation", cycles[(N_RUNS/2) -1], tmp);

    // Multiplication
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_mul_8x1(c, b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_8x1(c, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("multiplication", cycles[(N_RUNS/2) -1], tmp);

    // Squaring
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_sqr_8x1(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_8x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("squaring", cycles[(N_RUNS/2) -1], tmp);

    // Inv
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations_inv; j++) {
            fp_inv_8x1(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations_inv;
    }
    tmp = fp_is_equal_8x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("inversion", cycles[(N_RUNS/2) -1], tmp);

    printf("Fp lvl3 4x2 benchmarks:\n");

    // Addition
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_add_4x2(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_4x2(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("addition", cycles[(N_RUNS/2) -1], tmp);

    // Subtraction
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_sub_4x2(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_4x2(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("subtraction", cycles[(N_RUNS/2) -1], tmp);

    // Negation
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_neg_4x2(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_4x2(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("negation", cycles[(N_RUNS/2) -1], tmp);

    // Multiplication
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_mul_4x2(c, b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_4x2(c, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("multiplication", cycles[(N_RUNS/2) -1], tmp);

    // Squaring
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_sqr_4x2(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_4x2(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("squaring", cycles[(N_RUNS/2) -1], tmp);

    // Inv
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations_inv; j++) {
            fp_inv_4x2(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations_inv;
    }
    tmp = fp_is_equal_4x2(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("inversion", cycles[(N_RUNS/2) -1], tmp);

    printf("Fp lvl3 2x4 benchmarks:\n");

    // Multiplication
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_mul_2x4(c, b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_2x4(c, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("multiplication", cycles[(N_RUNS/2) -1], tmp);

    // Squaring
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_sqr_2x4(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_2x4(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("squaring", cycles[(N_RUNS/2) -1], tmp);

    // Square root
    iterations = 20;
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            fp_sqrt_2x4(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp_is_equal_2x4(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("sqrt", cycles[(N_RUNS/2) -1], tmp);

    // Inversion
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations_inv; j++) {
            fp_inv_2x4(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations_inv;
    }
    tmp = fp_is_equal_2x4(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("inv", cycles[(N_RUNS/2) -1], tmp);
}