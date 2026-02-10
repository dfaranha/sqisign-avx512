#include <stdio.h>
#include "bench.h"
#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "ec.h"
#include "theta.h"
#include "theta_gluing.h"
#include "theta_middle.h"
#include "theta_splitting.h"
#include "utils.h"
#include "KAT.h"

#define N_RUNS 10

void bench_theta(int iterations)
{
    uint64_t cycles[N_RUNS];
    uint64_t start, end, tmp;
    __m512i a[16], b[16], c[16], d[16], e[16], f[16];
    rand_fp2_8x1x1(a);
    rand_fp2_8x1x1(b);
    rand_fp2_8x1x1(c);
    rand_fp2_8x1x1(d);

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
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("had", cycles[(N_RUNS/2) -1], tmp);

    // mul
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_mul_single(c, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("mul", cycles[(N_RUNS/2) -1], tmp);

    // sqr 
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_sqr_single(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("sqr", cycles[(N_RUNS/2) -1], tmp);

    // double 
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_double_single(d, a, b, c);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("double", cycles[(N_RUNS/2) -1], tmp);

    // codomain
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_codomain_single(c, d, a, b, false, true);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("codomain (h1, h2 = 0, 1)", cycles[(N_RUNS/2) -1], tmp);

    // eval
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_eval_single(a, b, c, false, true);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("eval (h1, h2 = 0, 1)", cycles[(N_RUNS/2) -1], tmp);

    // chain
    fp2_tomont_4x2x1(KAT_OA, KAT_OA);
    fp2_tomont_4x2x1(KAT_KER_P1, KAT_KER_P1);
    fp2_tomont_4x2x1(KAT_KER_P2, KAT_KER_P2);

    int n = 188 - 1;
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        theta_isogeny_chain_no_gluing_single(a, b, KAT_OA, KAT_KER_P1, KAT_KER_P2, c, n, 0, true);
        end = cpucycles();
        cycles[i] = (end - start);
    }
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("chain n = 187, num_points = 0, 10 runs", cycles[(N_RUNS/2)-1], tmp);

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
    print_bench_res("had", cycles[(N_RUNS/2) -1], tmp);

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
    print_bench_res("had alt", cycles[(N_RUNS/2) -1], tmp);

    // mul
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_mul_pair(c, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("mul", cycles[(N_RUNS/2) -1], tmp);

    // sqr 
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_sqr_pair(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("sqr", cycles[(N_RUNS/2) -1], tmp);

    // double 
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_double_pair(d, a, b, c);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("double", cycles[(N_RUNS/2) -1], tmp);

    // eval
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int i = 0; i < iterations; i++) {
            theta_eval_pair(a, b, c, false, true);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("eval (h1, h2 = 0, 1)", cycles[(N_RUNS/2) -1], tmp);

    // chain
    __m512i KAT_KER[16];
    fp2_4x2x1_to_8x1x1(KAT_KER, KAT_KER_P1, KAT_KER_P2);

    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        theta_isogeny_chain_no_gluing_pair(a, b, KAT_OA, KAT_KER, c, n, 0, true);
        end = cpucycles();
        cycles[i] = (end - start);
    }
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("chain n = 187, num_points = 0, 10 runs", cycles[(N_RUNS/2)-1], tmp);

    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        ec_add_components_4x2x1_old(a, a+8, b, b+8, c, c+8, d, d+8, e, e+8);
        end = cpucycles();
        cycles[i] = (end - start);
    }
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("ec_add_components_4x2x1_old", cycles[(N_RUNS/2)-1], tmp);


    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        ec_add_components_4x2x1(a, a+8, b, b+8, c, c+8, d, d+8, e, e+8);
        end = cpucycles();
        cycles[i] = (end - start);
    }
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("ec_add_components_4x2x1", cycles[(N_RUNS/2)-1], tmp);
}