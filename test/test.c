#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "test.h"

void print_test_res(char * title, int passed)
{
    int len = strlen(title);
    if (len > 54) {
        printf("Test name too long: %s\n", title);
    } 
    int dots_len = 55-len;
    char dots[dots_len+1];
    for(int i = 0; i < dots_len; i++) {
        dots[i] = '.';
    }
    dots[dots_len] = 0; //0-terminated

    char * passed_txt = (passed == 0) ? "FAILED" : "PASSED";
    printf("   %s %s %s\n", title, dots, passed_txt);
}

int main()
{
    // === TODO: Improve how randomness is generated ===

    time_t t;
	srand((unsigned) time(&t));

    // =================================================

    int n_tests = 2000;
    bool passed;

    passed = test_fp(n_tests);
    if(!passed) { return 0; }

    passed = test_fp2(n_tests);
    if(!passed) { return 0; }

    // passed = test_fp_unvec_lvl3(n_tests);
    // if (!passed) { return 0; }

    // passed = test_ec(n_tests);
    // if (!passed) { return 0; }

    // passed = test_theta_gluing(n_tests);
    // if (!passed) { return 0; }

    // passed = test_theta_middle(n_tests);
    // if (!passed) { return 0; }

    // passed = test_theta_splitting(n_tests);
    // if (!passed) { return 0; }

    // passed = test_theta_KAT(n_tests);
    // if (!passed) { return 0; }

    passed = test_verification_kat();
    if (!passed) { return 0; }
    
    return 0;
}
