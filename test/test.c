#include <stdio.h>
#include <string.h>
#include "test.h"

void print_test_res(char * test_name, int passed)
{
    int len = strlen(test_name);
    if (len > 63) {
        printf("Test name too long: %s\n", test_name);
    }
    int dots_len = 64-len;
    char dots[dots_len+1];
    for(int i = 0; i < dots_len; i++) {
        dots[i] = '.';
    }
    dots[dots_len] = 0; //0-terminated

    char * passed_txt = (passed == 1) ? "PASSED" : "FAILED";
    printf("  %s %s %s\n", test_name, dots, passed_txt);
}

bool run_tests(int iterations)
{
    bool passed;
    passed = test_fp(iterations);
    if(!passed) {
        printf("test_fp FAILED\n");
        return false;
    }
    return true;
}