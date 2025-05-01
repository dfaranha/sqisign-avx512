#ifndef _TESTS_H
#define _TESTS_H

#include <stdbool.h>

bool test_fp(int iterations);
bool test_fp2(int iterations);
bool test_theta(int iterations);

void print_test_res(char * title, int passed);

#endif