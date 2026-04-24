#ifndef SQI_VERIF_H
#define SQI_VERIF_H

#include "sqi_ec.h"
#include "sqi_fp_lvl3.h"
#include "sqi_fp2.h"
#include "sqi_fp_constants_lvl3.h"
#include "verif.h"

void hash_to_challenge(
    scalar_t* scalar,
    const byte_t *pk_com_enc,
    const unsigned char* message,
    size_t length
);

#endif