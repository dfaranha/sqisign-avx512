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
    uint64_t arr[8*12];
    for(int j = 0; j < 8*12; j++) {
        arr[j] = rand_uint64();
    }

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

    printf("Fp2 lvl3 4x1x2 benchmarks:\n");

    // Addition
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_add_4x1x2(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x1x2(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("addition", cycles[(N_RUNS/2) -1], tmp);

    // Subtraction
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_sub_4x1x2(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x1x2(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("subtraction", cycles[(N_RUNS/2) -1], tmp);

    // Negation
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_neg_4x1x2(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x1x2(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("negation", cycles[(N_RUNS/2) -1], tmp);

    // Multiplication
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_mul_4x1x2(c, b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x1x2(c, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("multiplication", cycles[(N_RUNS/2) -1], tmp);

    // Squaring
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_sqr_4x1x2(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x1x2(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("squaring", cycles[(N_RUNS/2) -1], tmp);

    printf("Fp2 lvl3 4x2x1 benchmarks:\n");

    // Addition
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_add_4x2x1(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("addition", cycles[(N_RUNS/2) -1], tmp);

    // Subtraction
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_sub_4x2x1(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("subtraction", cycles[(N_RUNS/2) -1], tmp);

    // Negation
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_neg_4x2x1(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("negation", cycles[(N_RUNS/2) -1], tmp);

    // Multiplication
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_mul_4x2x1(c, b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(c, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("multiplication", cycles[(N_RUNS/2) -1], tmp);

    // Squaring
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_sqr_4x2x1(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("squaring", cycles[(N_RUNS/2) -1], tmp);

    printf("Fp2 lvl3 2x2x2 benchmarks:\n");

    // Addition
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_add_2x2x2(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x2x2(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("addition", cycles[(N_RUNS/2) -1], tmp);

    // Subtraction
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_sub_2x2x2(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x2x2(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("subtraction", cycles[(N_RUNS/2) -1], tmp);

    // Negation
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_neg_2x2x2(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x2x2(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("negation", cycles[(N_RUNS/2) -1], tmp);

    // Multiplication
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_mul_2x2x2(c, b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x2x2(c, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("multiplication", cycles[(N_RUNS/2) -1], tmp);

    // Squaring
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_sqr_2x2x2(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x2x2(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("squaring", cycles[(N_RUNS/2) -1], tmp);

    printf("Fp2 lvl3 2x4x1 benchmarks:\n");

    // Addition
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_add_2x4x1(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x4x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("addition", cycles[(N_RUNS/2) -1], tmp);

    // Subtraction
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_sub_2x4x1(b, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x4x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("subtraction", cycles[(N_RUNS/2) -1], tmp);

    // Negation
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_neg_2x4x1(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x4x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("negation", cycles[(N_RUNS/2) -1], tmp);

    // Multiplication
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_mul_2x4x1(c, b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x4x1(c, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("multiplication", cycles[(N_RUNS/2) -1], tmp);

    // Squaring
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_sqr_2x4x1(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x4x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("squaring", cycles[(N_RUNS/2) -1], tmp);

    printf("Fp2 lvl3 conversion benchmarks:\n");

    // unvec to 8x1x1
    fp2_t unvec[8];
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_unvectorized_to_8x1x1(c, unvec);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("8x1x1 <- unvectorized", cycles[(N_RUNS/2) -1], tmp);

    // 8x1x1 to unvec
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_8x1x1_to_unvectorized(unvec, c);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("8x1x1 -> unvectorized", cycles[(N_RUNS/2) -1], tmp);

    // unvec to 4x2x1
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_unvectorized_to_4x2x1(c, unvec);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("4x2x1 <- unvectorized", cycles[(N_RUNS/2) -1], tmp);

    // 4x2x1 to unvec
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_4x2x1_to_unvectorized(unvec, c);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("4x2x1 -> unvectorized", cycles[(N_RUNS/2) -1], tmp);

    // unvec to 2x4x1
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_unvectorized_to_2x4x1(c, unvec);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x4x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("2x4x1 <- unvectorized", cycles[(N_RUNS/2) -1], tmp);

    // 2x4x1 to unvec
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_2x4x1_to_unvectorized(unvec, c);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x4x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("2x4x1 -> unvectorized", cycles[(N_RUNS/2) -1], tmp);


    // 8x1x1 to 4x2x1
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_8x1x1_to_4x2x1(c, c+8, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("8x1x1 -> 4x2x1", cycles[(N_RUNS/2) -1], tmp);

    // 4x2x1 to 8x1x1
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_4x2x1_to_8x1x1(a, b, c);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("8x1x1 <- 4x2x1", cycles[(N_RUNS/2) -1], tmp);

    // 4x2x1 to 2x4x1
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_4x2x1_to_2x4x1(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("4x2x1 -> 2x4x1", cycles[(N_RUNS/2) -1], tmp);

    // 2x4x1 to 4x2x1
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            fp2_2x4x1_to_4x2x1(a, b, c);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(c, a);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("4x2x1 <- 2x4x1", cycles[(N_RUNS/2) -1], tmp);
}