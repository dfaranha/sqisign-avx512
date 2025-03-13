#ifndef _TESTS_H
#define _TESTS_H

#include <stdbool.h>

bool test_fp(int iterations);

void print_test_res(char * test_name, int passed);
bool run_tests(int iterations);

#endif