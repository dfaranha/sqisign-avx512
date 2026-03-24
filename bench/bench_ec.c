#include <stdio.h>
#include "bench.h"
#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "ec.h"

#include "sqi_test_extras.h"

#define N_RUNS 10

void bench_ec(int iterations)
{
    uint64_t cycles[N_RUNS];
    uint64_t start, end, tmp;
    __m512i A24_x4[8], a[8], b[8], c[8], d[8], Px_x4[8], Pz_x4[8];
    fp2_t A24_unvec[4];
    ec_point_t P[4], Q[4], R[4];

    // Working with fixed curve of j-invariant 1728
    ec_curve_t curve;
    ec_curve_init(&curve);
    fp2_set_small(&(curve.A), 0);
    fp2_set_small(&(curve.C), 1);
    ec_curve_normalize_A24(&curve);

    for (int i = 0; i < 4; i++) {
        fp2_copy(A24_unvec+i, &curve.A24.x);
    }
    fp2_unvectorized_to_4x2x1(A24_x4, A24_unvec);

    for (int i = 0; i < 4; i++) {
        ec_random_test(P+i, &curve);
    }
    ec_point_unvectorized_to_4x4x2x1(Px_x4, Pz_x4, P);

    printf("ec benchmarks\n");

    // ec_double_4x4x2x1
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            ec_double_4x4x2x1(a, b, A24_x4, Px_x4, Pz_x4);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, Px_x4);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("double 4x4x2x1", cycles[(N_RUNS/2) -1], tmp);

    // ec_double_1x2x4x1
    ec_point_unvec_to_1x2x4x1(a, P);
    ec_point_unvec_to_1x2x4x1(b, &curve.A24);
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            ec_double_1x2x4x1(c, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_2x4x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("double 1x2x4x1", cycles[(N_RUNS/2) -1], tmp);
}