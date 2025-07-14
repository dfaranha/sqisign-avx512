#include <stdio.h>

#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "test.h"
#include "theta.h"
#include "utils.h"

bool test_theta(int iterations)
{
    __m512i a[16], b[16], c[16], d[16], e[16], f[16], zero[16], one[16], t;
    fp2_setzero_8x1x1(zero);
    fp2_setzero_8x1x1(one);
    one[0] = VSETALL(1);
    int passed = 1;

    printf("Theta single tests:\n");

    // Hadamard
    for (int i = 0; i < iterations; i++) {
        rand_fp2_4x2x1(a);
        rand_fp2_4x2x1(b);
        
        // H^2 is mult by 4
        theta_had_single(d, a);
        theta_had_single(e, d);
        fp2_add_4x2x1(f, a, a);
        fp2_add_4x2x1(f, f, f);

        if (fp2_is_equal_4x2x1(e, f) != 0xf) {
            passed = 0;
            break;
        }

        // In place
        fp2_copy_4x2x1(d, a);
        theta_had_single(d, d);
        theta_had_single(e, a);

        if (fp2_is_equal_4x2x1(d, e) != 0xf) {
            passed = 0;
            break;
        }

        // Linear H*a + H*b = H*(a+b)
        theta_had_single(d, a);
        theta_had_single(e, b);
        fp2_add_4x2x1(d, d, e);
        fp2_add_4x2x1(f, a, b);
        theta_had_single(f, f);

        if (fp2_is_equal_4x2x1(d, f) != 0xf) {
            passed = 0;
            break;
        }
    }
    print_test_res("Hadamard", passed);
    if (passed == 0) {
        return false;
    }

    // Isogeny chain KAT

    __m512i OA[8], P[8], Q[8];
    OA[0] = VSET(0x1B4711C9AD8C7, 0x7BA3443DE53A8, 0x6BB933FCE3592, 0x19A1C146791C0, 0x6246447442228, 0x590A2945EDCBC, 0x32B866A777EF3, 0x7708A64E81AD5);
    OA[1] = VSET(0x1A869DBC2E8FA, 0x4BD047070B32D, 0x5919409BD7452, 0x64BA66D7E00CA, 0x52B32A6633E7D, 0x0F22E02B91FF3, 0x1145CD45DC9D6, 0x280CFFFC66D8F);
    OA[2] = VSET(0x7478DA2EE534D, 0x2E408A7894D8A, 0x190AA9853FA2A, 0x337C03EC1C3F1, 0x75E6B768FF4B2, 0x267299E21625C, 0x1A7886BF59B90, 0x2BAE13559D8C3);
    OA[3] = VSET(0x5867F7FEB9DB1, 0x327A234E8F41A, 0x69D96257156E0, 0x4A9846674A5DE, 0x4800F53758EFA, 0x59910B3C96BF5, 0x59725F8FB4829, 0x71AF2E5551DB9);
    OA[4] = VSET(0x378622C14CCF0, 0x7FE683C927E4A, 0x65A996489D7CD, 0x1E13FB886BAEC, 0x14CD9F1BD6719, 0x5DF435AA9CBC9, 0x42F112A3271F6, 0x7C21AD69E086B);
    OA[5] = VSET(0x72D9228C3CD4F, 0x16F330EA1CEC2, 0x2F0C29D8902F0, 0x4D0DAF4D334ED, 0x64AAA7EF9DA29, 0x7D968087D21A8, 0x20DDAF3BF0FCA, 0x33B0FEEAE87D2);
    OA[6] = VSET(0x4733F7DDFB2C5, 0x4015385C1BE42, 0x72232378DC184, 0x51A8597246D6A, 0x13EE0287E757D, 0x3363803F40C04, 0x3EDD2E22C843C, 0x44F6A1556BB2D);
    OA[7] = VSET(0x0000000F0C326, 0x0000000DF76A4, 0x0000001182500, 0x00000017FFDF4, 0x000000160A35D, 0x0000001BB491C, 0x0000001880537, 0x000000053D06C);

    P[0] = VSET(0x775B1803F1072, 0x554F95575C615, 0x243E53BF562E5, 0x7ABC8877432F9, 0x53642AAC81761, 0x357775FCAA591, 0x20AE7E7BEBB6C, 0x20AC258AA58E3);
    P[1] = VSET(0x786D981FD8408, 0x272849C94D6D1, 0x30A3353E53A0F, 0x29BB9BE7F9A20, 0x184388E8C6741, 0x2C829D72D9BC3, 0x217F94E073FC1, 0x47F571A420408);
    P[2] = VSET(0x2DA5A747BC9D7, 0x2B53831B83F8B, 0x40474B14E5AF1, 0x4EF30248FB5B1, 0x1C4E6D84B6386, 0x10D0C83BB3B85, 0x6FA38C08903AA, 0x5B105AF21A0D8);
    P[3] = VSET(0x00C875AC96B87, 0x73AE5136CA9EC, 0x0B568B3534E22, 0x31438B32E90FD, 0x7ECECBC5D1311, 0x7F005148AC173, 0x7ACB2F3CAAE47, 0x6D4778622BA9E);
    P[4] = VSET(0x64C3F0CA2A372, 0x70842120E7379, 0x60E8EA52F1FF6, 0x4C61045D5DAF7, 0x20C18C818F08A, 0x0FF9BABB29D47, 0x33431CFE0F459, 0x64E48AF59942D);
    P[5] = VSET(0x655DB3815197B, 0x0040F20DF763E, 0x223B6509AFE8C, 0x641E49FA9EA87, 0x4665363421074, 0x76B4569F1D2BC, 0x04279DECA9EC3, 0x66609B84E9076);
    P[6] = VSET(0x70D80F0223680, 0x14BE9CECC251B, 0x6B87407E823CB, 0x69F015FBC6507, 0x21AE813D6CCED, 0x09A4D26AC49E2, 0x2052257F62AAA, 0x1E91C82B25D8B);
    P[7] = VSET(0x0000000D51932, 0x0000000F99540, 0x00000004D6107, 0x000000177B1B7, 0x00000013F735B, 0x00000013BD642, 0x000000145C4C9, 0x0000001E4B6DC);

    Q[0] = VSET(0x59D5A0019ECC8, 0x76AE18D8D1F38, 0x1C9798E027B61, 0x3C96A8446CCDA, 0x63F7A8435A8E5, 0x17BD275A2146E, 0x2E3E7FA88DD4F, 0x296F21B947D91);
    Q[1] = VSET(0x4F03BF379D9E5, 0x07B9A0532FC9D, 0x7996572EFBC35, 0x1D3D7AA738326, 0x19712025DF507, 0x0E88D0D44A2AE, 0x39DE788805ADA, 0x75DDE0A24EAE3);
    Q[2] = VSET(0x66B69DF74C707, 0x3D42AC1171EB6, 0x7F76720BD165D, 0x77F48D5B90325, 0x4039B52AC2A4A, 0x47C3C745E9AD2, 0x4D9C851CF6E2D, 0x5D4B070CC392C);
    Q[3] = VSET(0x4FD574D07B6A1, 0x52C8E818E5514, 0x14B160FA1E7EC, 0x3559A94D27118, 0x613942B9BA7EF, 0x39324E9886DEC, 0x05A84D9749E62, 0x5B874F9F9B7CA);
    Q[4] = VSET(0x6539126E20071, 0x10259D069667C, 0x75569E7E52C85, 0x6C0A940847ADC, 0x2831D7D09684E, 0x66ADA451EE6C4, 0x25D446EA5BCAC, 0x50DC8C1118B63);
    Q[5] = VSET(0x7CD53975A8F31, 0x3B4EF7FDEAE78, 0x11490A0ADBC58, 0x0486523BDC5AB, 0x7DBD05A7B00A1, 0x4CE2CA2F38B7B, 0x0363AE269CD35, 0x050EC4C692E35);
    Q[6] = VSET(0x1522E52555253, 0x65D8FF4968B75, 0x6E138205C001E, 0x2F42A441D01D9, 0x7D3DD1289B5A1, 0x382CDF99C076F, 0x4852BB30A78E2, 0x375FD102C3869);
    Q[7] = VSET(0x0000001AFB74A, 0x0000001336733, 0x0000000254FE6, 0x000000170C4D8, 0x000000192B0D3, 0x0000001FA3B62, 0x0000001288626, 0x00000013DE0F1);

    fp2_tomont_4x2x1(OA, OA);
    fp2_tomont_4x2x1(P, P);
    fp2_tomont_4x2x1(Q, Q);

    //We reduce by one as we have done the gluing step
    int n = 188 - 1;

    theta_isogeny_chain_single(OA, P, Q, n);

    printf("Theta pair tests:\n");

    // Hadamard
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_8x1x1(b);
        
        // H^2 is mult by 4
        theta_had_pair(d, a);
        theta_had_pair(e, d);
        fp2_add_8x1x1(f, a, a);
        fp2_add_8x1x1(f, f, f);

        if (fp2_is_equal_8x1x1(e, f) != 0xff) {
            passed = 0;
            break;
        }

        // In place
        fp2_copy_8x1x1(d, a);
        theta_had_pair(d, d);
        theta_had_pair(e, a);

        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }

        // Linear H*a + H*b = H*(a+b)
        theta_had_pair(d, a);
        theta_had_pair(e, b);
        fp2_add_8x1x1(d, d, e);
        fp2_add_8x1x1(f, a, b);
        theta_had_pair(f, f);

        if (fp2_is_equal_8x1x1(d, f) != 0xff) {
            passed = 0;
            break;
        }

        // Pair matches single
        theta_had_pair(d, a);
        fp2_8x1x1_to_4x2x1(e, a);
        theta_had_single(e, e);
        theta_had_single(e+8, e+8);
        fp2_4x2x1_to_8x1x1(f, e, e+8);

        if (fp2_is_equal_8x1x1(d, f) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("Hadamard", passed);
    if (passed == 0) {
        return false;
    }

    // mul
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);

        // Pair matches single
        theta_mul_pair(d, a, b);
        fp2_8x1x1_to_4x2x1(e, a);
        theta_mul_single(f, e, b);
        theta_mul_single(f+8, e+8, b);
        fp2_4x2x1_to_8x1x1(e, f, f+8);

        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("mul", passed);
    if (passed == 0) {
        return false;
    }

    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);
        rand_fp2_4x2x1(c);

        // Pair matches single
        theta_double_pair(d, a, b, c);
        fp2_8x1x1_to_4x2x1(e, a);
        theta_double_single(f, e, b, c);
        theta_double_single(f+8, e+8, b, c);
        fp2_4x2x1_to_8x1x1(e, f, f+8);
        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("double", passed);
    if (passed == 0) {
        return false;
    }

    // TODO: Test is not using an actual dual theta point
    // eval
    for (int i = 0; i < iterations; i++) {
        rand_fp2_8x1x1(a);
        rand_fp2_4x2x1(b);
        bool h1 = false;
        bool h2 = true;

        // Pair matches single
        theta_eval_pair(d, a, b, h1, h2);
        fp2_8x1x1_to_4x2x1(e, a);
        theta_eval_single(f, e, b, h1, h2);
        theta_eval_single(f+8, e+8, b, h1, h2);
        fp2_4x2x1_to_8x1x1(e, f, f+8);

        if (fp2_is_equal_8x1x1(d, e) != 0xff) {
            passed = 0;
            break;
        }
    }
    print_test_res("eval", passed);
    if (passed == 0) {
        return false;
    }

    return true;
}