#ifndef TUTIL_H
#define TUTIL_H

#include <stddef.h>
#include <stdint.h>

#define digit_t uint64_t
#define sdigit_t int64_t
#define RADIX 64
#define LOG2RADIX 6
#define BSWAP_DIGIT(i) BSWAP64(i)

#endif