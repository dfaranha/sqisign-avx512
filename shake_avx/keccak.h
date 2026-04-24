/** 
 * Copyright (C) 2015 by Roberto Cabral 
 * https://github.com/rbCabral/SHA-3
 */ 

#include "immintrin.h"
#include "emmintrin.h"
#include <string.h>
#include <stdio.h>
#include <stdint.h>

#define BENCH 1000
#define ALIGN __attribute__ ((aligned (32)))

#define KECCAK256

typedef unsigned long long uInt256[4];
typedef unsigned long long uInt512[8];

// we only use the inner keccak function
void keccakF(__m512i* x, int rnd);