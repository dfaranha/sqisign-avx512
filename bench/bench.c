#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "bench.h"

int cmp_uint64(const void *a, const void *b)
{
    uint64_t aa = *(const uint64_t *) a;
    uint64_t bb = *(const uint64_t *) b;
    if (aa < bb) {
        return -1;
    } else if (aa == bb) {
        return 0;
    } else {
        return 1;
    }
}

void print_bench_res(char * title, uint64_t cycles, uint64_t tmp)
{
    int len = strlen(title);
    if (len > 63) {
        printf("Benchmarking title too long: %s\n", title);
    }
    int dots_len = 64-len;
    char dots[dots_len+1];
    for(int i = 0; i < dots_len; i++) {
        dots[i] = '.';
    }
    dots[dots_len] = 0; //0-terminated

    printf("   %s runs in %s %" PRIu64 " cycles, (ignore: %lu)\n", title, dots, cycles, tmp);
}

int main()
{
    // === TODO: Improve how randomness is generated ===
    time_t t;
	srand((unsigned) time(&t));
    // =================================================

    int n_bench = 50;

    bench_fp(n_bench);
    bench_fp2(n_bench);
    // bench_ec(n_bench);
    // bench_1d_isog(n_bench);
    // bench_theta(n_bench);
    //bench_verif(20);

    return 0;
}
