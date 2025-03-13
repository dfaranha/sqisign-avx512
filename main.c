#include <stdio.h>
#include <time.h>

#include "fp_lvl3.h"
#include "utils.h"
#include "test.h"

int main()
{
    // === TODO: Improve how randomness is generated ===

    time_t t;
	srand((unsigned) time(&t));

    // =================================================

    __m512i a[8], b[8], c[8];
    rand_uint64();

    fp_add_8x1(a, b, c);

    int iterations = 1;
    run_tests(iterations);
    return 0;
}