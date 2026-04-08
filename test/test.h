#ifndef _TESTS_H
#define _TESTS_H

#include <stdbool.h>

bool test_fp(int iterations);
bool test_fp2(int iterations);
bool test_fp_unvec_lvl3(int iterations);
bool test_ec(int iterations);
bool test_theta_KAT(int iterations);
bool test_theta_gluing(int iterations);
bool test_theta_middle(int iterations);
bool test_theta_splitting(int iterations);
int test_verification_kat();

void print_test_res(char * title, int passed);

#endif
