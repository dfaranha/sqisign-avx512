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
    for (int i = 0; i < 100; i++) {
        rc = test_verification_kat_single(i);
        if (rc != KAT_SUCCESS) {
            printf("KAT failed at count %d\n", i);
            return rc;
        }
    }
    printf("Known Answer Tests PASSED. \n");
    return KAT_SUCCESS;
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


// bool test_verif_KAT()
// {
//     int passed = 1;

//     printf("SQIsign verification KATs:\n");

//     for (int k = 0; k < NUM_KATS; k++) {
//         KAT_SQIsign_t kat = SQIsign_KATS_lvl3[k];
//         passed = verify(kat.m, kat.mlen, kat.sig, kat.siglen, kat.pk);
//         // unsigned long long mlen = 33;
//         // unsigned long long siglen = 224;
//         // unsigned char m[33] = {
//         //     216, 28, 77, 141, 115, 79, 203, 251, 234, 222, 61, 63, 138, 3, 159, 170, 42, 44, 153, 87, 232, 53, 173, 85, 178, 46, 117, 191, 87, 187, 85, 106, 200,
//         // };
//         // unsigned char sig[224] = {
//         //     8, 104, 207, 191, 39, 91, 142, 123, 25, 191, 89, 125, 101, 141, 98, 204, 145, 59, 155, 41, 51, 227, 10, 41, 114, 136, 251, 230, 135, 246, 246, 184, 172, 138, 247, 170, 0, 127, 25, 19, 134, 187, 26, 32, 60, 221, 188, 43, 219, 66, 121, 45, 5, 218, 105, 164, 80, 112, 115, 209, 43, 11, 220, 71, 226, 179, 107, 196, 186, 69, 198, 135, 145, 145, 130, 129, 229, 120, 242, 220, 20, 41, 69, 4, 114, 109, 205, 76, 164, 196, 86, 95, 187, 137, 161, 40, 0, 4, 140, 123, 132, 116, 106, 44, 189, 130, 71, 36, 142, 36, 139, 112, 181, 26, 233, 25, 148, 149, 120, 87, 105, 42, 2, 141, 143, 92, 171, 171, 252, 145, 228, 191, 28, 93, 53, 2, 25, 160, 24, 156, 87, 222, 74, 119, 16, 210, 158, 3, 100, 199, 155, 33, 136, 68, 158, 192, 57, 115, 89, 67, 13, 89, 76, 123, 89, 128, 204, 103, 85, 25, 51, 169, 2, 211, 193, 31, 15, 189, 109, 195, 151, 17, 211, 225, 245, 1, 21, 158, 231, 251, 133, 206, 129, 180, 206, 36, 225, 1, 96, 6, 86, 125, 244, 105, 49, 93, 81, 62, 115, 246, 159, 99, 1, 102, 78, 100, 73, 175, 157, 206, 180, 0, 13, 21
//         // };
//         // unsigned char pk[97] = {
//         //     195, 35, 119, 214, 246, 215, 7, 41, 136, 74, 127, 104, 119, 239, 71, 145, 227, 93, 33, 247, 81, 163, 233, 109, 226, 63, 154, 122, 60, 1, 188, 216, 165, 241, 70, 220, 25, 228, 226, 172, 99, 0, 116, 87, 249, 125, 138, 64, 238, 132, 174, 231, 86, 76, 169, 167, 251, 230, 32, 15, 211, 229, 229, 89, 1, 191, 198, 14, 178, 92, 80, 211, 159, 92, 145, 201, 101, 16, 85, 107, 170, 34, 2, 141, 247, 99, 96, 132, 23, 33, 166, 1, 214, 94, 141, 15, 6,
//         // };
//         // passed = verify(m, mlen, sig, siglen, pk);
//         printf("passed %d", passed);
//         if (!passed) {
//             printf("   Failed KAT %d\n", k);
//             return false;
//         }
//     }
//     print_test_res("KATs", passed);

//     return true;
// }
