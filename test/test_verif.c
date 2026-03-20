#include <stdio.h>

#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "verif.h"
#include "test.h"

// lvl 3 constants
#define CRYPTO_SECRETKEYBYTES 529
#define CRYPTO_PUBLICKEYBYTES 97
#define CRYPTO_BYTES 224
#define SIGNATURE_BYTES CRYPTO_BYTES
#define CRYPTO_ALGNAME "SQIsign_lvl3"

#define MAX_MARKER_LEN         50

#define KAT_SUCCESS             0
#define KAT_FILE_OPEN_ERROR    -1
#define KAT_DATA_ERROR         -3
#define KAT_CRYPTO_FAILURE     -4
#define KAT_VERIFICATION_ERROR -5

static int      FindMarker(FILE *infile, const char *marker);
static int      ReadHex(FILE *infile, unsigned char *A, int Length, char *str);
static int      test_verification_kat_single(int cnt);

int test_verification_kat() {
    int rc;
    rc = test_verification_kat_single(-1);
    if (rc == KAT_SUCCESS) {
        printf("Known Answer Tests PASSED.\n");
    } else {
        printf("Known Answer Tests FAILED.\n");
    }
    return rc;
}

static int test_verification_kat_single(int cnt) {

    unsigned char       pk_rsp[CRYPTO_PUBLICKEYBYTES];
    unsigned char       *m, *m1, *sm_rsp;
    unsigned long long  mlen, smlen, mlen1;
    int                 count;
    int                 done;
    int                 ret_val;

    char                fn_rsp[64];
    FILE                *fp_rsp;

    sprintf(fn_rsp, "assets/PQCsignKAT_%d_%s.rsp", CRYPTO_SECRETKEYBYTES, CRYPTO_ALGNAME);
    if ( (fp_rsp = fopen(fn_rsp, "r")) == NULL ) {
        printf("Couldn't open <%s> for read\n", fn_rsp);
        return KAT_FILE_OPEN_ERROR;
    }

    done = 0;
    do {

        if ( FindMarker(fp_rsp, "count = ") ) {
            ret_val = fscanf(fp_rsp, "%d", &count);
        } else {
            done = 1;
            break;
        }

        if (cnt != -1 && cnt != count)
            continue;

        printf(".");

        if ( FindMarker(fp_rsp, "mlen = ") ) {
            ret_val = fscanf(fp_rsp, "%lld", &mlen);
        } else {
            printf("ERROR: unable to read 'mlen' from <%s>\n", fn_rsp);
            return KAT_DATA_ERROR;
        }

        sm_rsp = (unsigned char *)calloc(mlen + CRYPTO_BYTES, sizeof(unsigned char));

        if ( !ReadHex(fp_rsp, pk_rsp, CRYPTO_PUBLICKEYBYTES, "pk = ") ) {
            printf("ERROR: unable to read 'pk' from <%s>\n", fn_rsp);
            return KAT_DATA_ERROR;
        }

        if ( FindMarker(fp_rsp, "smlen = ") ) {
            ret_val = fscanf(fp_rsp, "%llu", &smlen);
        } else {
            printf("ERROR: unable to read 'smlen' from <%s>\n", fn_rsp);
            return KAT_DATA_ERROR;
        }

        if ( !ReadHex(fp_rsp, sm_rsp, smlen, "sm = ") ) {
            printf("ERROR: unable to read 'sm' from <%s>\n", fn_rsp);
            return KAT_DATA_ERROR;
        }

        // Extract m and sig from sm: sm = [sig (CRYPTO_BYTES) | m (mlen)]
        unsigned char *sig = sm_rsp;
        m = sm_rsp + SIGNATURE_BYTES;

        if ( (ret_val = verify(m, mlen, sm_rsp, CRYPTO_BYTES, pk_rsp)) == 0 ) {
            printf("crypto_sign_open returned <%d>\n", ret_val);
            return KAT_CRYPTO_FAILURE;
        }

        free(sm_rsp);

    } while ( !done );

    printf("\n");

    fclose(fp_rsp);

    return KAT_SUCCESS;
}


//
// ALLOW TO READ HEXADECIMAL ENTRY (KEYS, DATA, TEXT, etc.)
//
static int
FindMarker(FILE *infile, const char *marker) {
    char    line[MAX_MARKER_LEN];
    int     i, len;
    int curr_line;

    len = (int)strlen(marker);
    if ( len > MAX_MARKER_LEN - 1 ) {
        len = MAX_MARKER_LEN - 1;
    }

    for ( i = 0; i < len; i++ ) {
        curr_line = fgetc(infile);
        line[i] = curr_line;
        if (curr_line == EOF ) {
            return 0;
        }
    }
    line[len] = '\0';

    while ( 1 ) {
        if ( !strncmp(line, marker, len) ) {
            return 1;
        }

        for ( i = 0; i < len - 1; i++ ) {
            line[i] = line[i + 1];
        }
        curr_line = fgetc(infile);
        line[len - 1] = curr_line;
        if (curr_line == EOF ) {
            return 0;
        }
        line[len] = '\0';
    }

    // shouldn't get here
    return 0;
}

//
// ALLOW TO READ HEXADECIMAL ENTRY (KEYS, DATA, TEXT, etc.)
//
static int
ReadHex(FILE *infile, unsigned char *A, int Length, char *str) {
    int         i, ch, started;
    unsigned char   ich;

    if ( Length == 0 ) {
        A[0] = 0x00;
        return 1;
    }
    memset(A, 0x00, Length);
    started = 0;
    if ( FindMarker(infile, str) )
        while ( (ch = fgetc(infile)) != EOF ) {
            if ( !isxdigit(ch) ) {
                if ( !started ) {
                    if ( ch == '\n' ) {
                        break;
                    } else {
                        continue;
                    }
                } else {
                    break;
                }
            }
            started = 1;
            if ( (ch >= '0') && (ch <= '9') ) {
                ich = ch - '0';
            } else if ( (ch >= 'A') && (ch <= 'F') ) {
                ich = ch - 'A' + 10;
            } else if ( (ch >= 'a') && (ch <= 'f') ) {
                ich = ch - 'a' + 10;
            } else { // shouldn't ever get here
                ich = 0;
            }

            for ( i = 0; i < Length - 1; i++ ) {
                A[i] = (A[i] << 4) | (A[i + 1] >> 4);
            }
            A[Length - 1] = (A[Length - 1] << 4) | ich;
        } else {
        return 0;
    }

    return 1;
}
