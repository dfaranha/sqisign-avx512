#ifndef KAT_SQIsign_lvl3_H
#define KAT_SQIsign_lvl3_H

#define PUBLICKEYBYTES 97
#define SIGBYTES 224
#define NUM_KATS 1

typedef struct KAT_SQIsign_t {
    unsigned char m[3300];
    unsigned long long mlen;
    unsigned char sig[SIGBYTES];
    unsigned long long siglen;
    unsigned char pk[PUBLICKEYBYTES];
} KAT_SQIsign_t;

extern KAT_SQIsign_t SQIsign_KATS_lvl3[NUM_KATS];

#endif