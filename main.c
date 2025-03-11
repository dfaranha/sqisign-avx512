#include <stdio.h>
#include <fp_lvl3.h>
//#include "intrinsics_avx512.h"

int main()
{
    printf("Hello world\n");
    __m512i a, b, c;
    a = _mm512_setzero_si512();
    b = _mm512_setzero_si512();
    c = _mm512_setzero_si512();

    fp_add_1x8(a, b, c);
    return 0;
}