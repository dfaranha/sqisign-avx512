#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fp.h"
#include "fp_consts.h"
#include "test.h"
#include "utils.h"

int main()
{
    bool passed;
    
    // Initialize randomness
    time_t t;
    srand((unsigned) time(&t));
    
    passed = fp_8x1_tests();
    if(!passed) { return 0; }
    
    passed = fp_4x2_tests();
    if(!passed) { return 0; }
    
    passed = fp_2x4_tests();
    if(!passed) { return 0; }
}

