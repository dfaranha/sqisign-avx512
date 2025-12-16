#include <stdio.h>

#include "fp_lvl3.h"
#include "fp2_lvl3.h"
#include "test.h"
#include "theta.h"
#include "theta_gluing.h"
#include "theta_middle.h"
#include "theta_splitting.h"
#include "utils.h"

#include "sqi_ec.h"

bool test_theta(int iterations)
{
    __m512i a[16], b[16], c[16], d[16], e[16], f[16], g[16];

    int passed = 1;

    printf("Theta chain tests:\n");

    // Isogeny chain KAT
    // TODO move cats to their own file, so they also can be used in bench
    __m512i OA[8], P[8], Q[8], codom[8];
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

    ec_curve_t E3_expected, E3_actual, E4_expected, E4_actual;
    ec_point_t eval_P1, eval_P2;
    fp2_t codom_unvec[4], eval_pts_unvec[4], A3_expected, A4_expected, j_expected, j_actual;
    fp_decode(&A3_expected.re, (const void *) ((uint64_t[6])
        {10867719576459308538U, 14426381232393286251U, 5948554069610288516U, 1853058860160698092U, 9225241635731980790U, 4588529309365208364U}
    ));
    fp_decode(&A3_expected.im, (const void *) ((uint64_t[6])
        {7630066809077324413U, 4807934117134237764U, 8466143812571798057U, 10358304445779752856U, 10545750972660984674U, 3604475042088292201U}
    ));
    fp_decode(&A4_expected.re, (const void *) ((uint64_t[6])
        {7297374626918444366U, 16063160519669627060U, 8936618591153282620U, 8542284924237654948U, 7528997634339566035U, 1037111423354581024U}
    ));
    fp_decode(&A4_expected.im, (const void *) ((uint64_t[6])
        {1492077505230043599U, 2736745017500914223U, 13160692708435571952U, 527388762590340581U, 17004613965867689856U, 3325318217617652149U}
    ));
    ec_montA_to_curve(&E3_expected, &A3_expected);
    ec_montA_to_curve(&E4_expected, &A4_expected);

    //We reduce by one as we have already done the gluing step
    int n = 188 - 1;
    int num_pts = 3;
    bool randomize = true;

    __m512i pts[num_pts*8];
    __m512i pts_copy[num_pts*8];
    __m512i eval_pts[num_pts*8];
    fp2_copy_4x2x1(pts, P);
    fp2_copy_4x2x1(pts+8, Q);
    fp2_copy_4x2x1(pts+16, Q);

    theta_isogeny_chain_no_gluing_single(codom, eval_pts, OA, P, Q, pts, n, num_pts, randomize);

    fp2_4x2x1_to_unvectorized(codom_unvec, codom);
    ec_montAC_to_curve(&E3_actual, codom_unvec, codom_unvec+2);
    ec_montAC_to_curve(&E4_actual, codom_unvec+1, codom_unvec+3);
    
    // non-randomized check:
    //int is_E3_correct = fp2_is_equal(&E3_expected.A, &E3_actual.A);
    //int is_E4_correct = fp2_is_equal(&E4_expected.A, &E4_actual.A);

    // randomized check:
    ec_j_inv(&j_expected, &E3_expected);
    ec_j_inv(&j_actual, &E3_actual);
    int is_E3_correct = fp2_is_equal(&j_expected, &j_actual);
    ec_j_inv(&j_expected, &E4_expected);
    ec_j_inv(&j_actual, &E4_actual);
    int is_E4_correct = fp2_is_equal(&j_expected, &j_actual);
    passed = is_E3_correct & is_E4_correct;

    // Check that evaluated points are on the curves
    for (int i = 0; i < num_pts && passed != 0; i++) {
        fp2_4x2x1_to_unvectorized(eval_pts_unvec, eval_pts + 8*i);
        eval_P1.x = eval_pts_unvec[0];
        eval_P2.x = eval_pts_unvec[1];
        eval_P1.z = eval_pts_unvec[2];
        eval_P2.z = eval_pts_unvec[3];

        int on_curve_E3 = ec_test_is_on_curve(&E3_actual, &eval_P1);
        int on_curve_E4 = ec_test_is_on_curve(&E4_actual, &eval_P2);

        int is_order4_P1 = test_point_order_twof(&eval_P1, &E3_actual, 2);
        int is_order4_P2 = test_point_order_twof(&eval_P2, &E4_actual, 2);
        int is_order4 = is_order4_P1 | is_order4_P2;
        passed = on_curve_E3 & on_curve_E4 & is_order4;
    }

    print_test_res("chain 4x2x1 KAT", passed);
    if (passed == 0) {
        return false;
    }

    // Isogeny chain KAT
    __m512i PQ[16];
    fp2_4x2x1_to_8x1x1(PQ, P, Q);
    fp2_4x2x1_to_8x1x1(pts, pts, pts+8);

    theta_isogeny_chain_no_gluing_pair(codom, eval_pts, OA, PQ, pts, n, num_pts, randomize);

    fp2_4x2x1_to_unvectorized(codom_unvec, codom);
    ec_montAC_to_curve(&E3_actual, codom_unvec, codom_unvec+2);
    ec_montAC_to_curve(&E4_actual, codom_unvec+1, codom_unvec+3);

    // non-randomized check:
    //is_E3_correct = fp2_is_equal(&E3_expected.A, &E3_actual.A);
    //is_E4_correct = fp2_is_equal(&E4_expected.A, &E4_actual.A);

    // randomized check:
    ec_j_inv(&j_expected, &E3_expected);
    ec_j_inv(&j_actual, &E3_actual);
    is_E3_correct = fp2_is_equal(&j_expected, &j_actual);
    ec_j_inv(&j_expected, &E4_expected);
    ec_j_inv(&j_actual, &E4_actual);
    is_E4_correct = fp2_is_equal(&j_expected, &j_actual);
    passed = is_E3_correct & is_E4_correct;

    // Check that evaluated points are on the curves:

    //Converting pairs back to two separate points.
    int num_pair_pts = num_pts / 2;
    for(int i = 0; i < num_pair_pts; i++) {
        fp2_8x1x1_to_4x2x1(eval_pts+16*i, eval_pts+16*i+8, eval_pts+16*i);
    }
    
    for (int i = 0; i < num_pts && passed != 0; i++) {
        fp2_4x2x1_to_unvectorized(eval_pts_unvec, eval_pts + 8*i);
        eval_P1.x = eval_pts_unvec[0];
        eval_P2.x = eval_pts_unvec[1];
        eval_P1.z = eval_pts_unvec[2];
        eval_P2.z = eval_pts_unvec[3];

        int on_curve_E3 = ec_test_is_on_curve(&E3_actual, &eval_P1);
        int on_curve_E4 = ec_test_is_on_curve(&E4_actual, &eval_P2);

        int is_order4_P1 = test_point_order_twof(&eval_P1, &E3_actual, 2);
        int is_order4_P2 = test_point_order_twof(&eval_P2, &E4_actual, 2);
        int is_order4 = is_order4_P1 | is_order4_P2;
        
        passed = on_curve_E3 & on_curve_E4 & is_order4;
    }

    print_test_res("chain 8x1x1 KAT", passed);
    if (passed == 0) {
        return false;
    }

    return true;
}

bool test_theta_KAT(int iterations)
{
    // ========== Setup ==================
    int length = 188;
    theta_couple_curve_t E12, E34_expected, E34_actual;
    theta_kernel_couple_points_t ker_extra, ker_exact;
    theta_couple_point_t input_pts_two[2], input_pts_three[3],
        res_pts_two[2], res_pts_three[3];
    fp2_t j_expected, j_actual;
    int passed = 1;

    ec_curve_init(&E12.E1);
    ec_curve_init(&E12.E2);
    ec_curve_init(&E34_expected.E1);
    ec_curve_init(&E34_expected.E2);

    fp_decode(&E12.E1.A.re, (const void *) ((uint64_t[6]) { 
        14772141750067297960U, 15497194067807589646U, 16152181089507688489U, 11760783556982355528U, 14985033237067221313U, 3980965958460185541U 
    }));
    fp_decode(&E12.E1.A.im, (const void *) ((uint64_t[6]) { 
        12664573960035771847U, 18153028621016581859U, 6945596057989715247U, 8733635196687419827U, 11146198237945641071U, 1934320287674005968U 
    }));
    fp_decode(&E12.E2.A.re, (const void *) ((uint64_t[6]) { 
        8903153739637942280U, 14727489198489190169U, 2957425914203028369U, 13328111210805823602U, 1376270801983474348U, 3151637089558903686U
    }));
    fp_decode(&E12.E2.A.im, (const void *) ((uint64_t[6]) {
        11847240010547413723U, 5178025954394730576U, 9783583908970607586U, 15920920559823786385U, 5534199853573548308U, 2927772670618944676U
    }));
    fp_decode(&E34_expected.E1.A.re, (const void *) ((uint64_t[6]) {
        10867719576459308538U, 14426381232393286251U, 5948554069610288516U, 1853058860160698092U, 9225241635731980790U, 4588529309365208364U
    }));
    fp_decode(&E34_expected.E1.A.im, (const void *) ((uint64_t[6]) {
        7630066809077324413U, 4807934117134237764U, 8466143812571798057U, 10358304445779752856U, 10545750972660984674U, 3604475042088292201U
    }));
    fp_decode(&E34_expected.E2.A.re, (const void *) ((uint64_t[6]) {
        7297374626918444366U, 16063160519669627060U, 8936618591153282620U, 8542284924237654948U, 7528997634339566035U, 1037111423354581024U
    }));
    fp_decode(&E34_expected.E2.A.im, (const void *) ((uint64_t[6]) {
        1492077505230043599U, 2736745017500914223U, 13160692708435571952U, 527388762590340581U, 17004613965867689856U, 3325318217617652149U
    }));
    fp_decode(&ker_extra.T1.P1.x.re, (const void *) ((uint64_t[6]) {
        13366702028363759326U, 761579417974681459U, 17689756108733811516U, 6329647123777029148U, 983018890354940375U, 1795279142226635474U
    }));
    fp_decode(&ker_extra.T1.P1.x.im, (const void *) ((uint64_t[6]) {
        10565314165628214632U, 17553840088349964772U, 12997417210223164999U, 18109885084621045011U, 2102998377533535836U, 2087788903899176838U
    }));
    fp_decode(&ker_extra.T1.P1.z.re, (const void *) ((uint64_t[6]) {
        11381609529241973170U, 10432418732008859666U, 719508680994577913U, 2272382642787187492U, 10945665385909613808U, 996740006660415122U
    }));
    fp_decode(&ker_extra.T1.P1.z.im, (const void *) ((uint64_t[6]) {
        2662910970183354532U, 12245874776217490583U, 8432600759547251633U, 16488009154928053586U, 14123866913009634971U, 946853326630560518U
    }));
    fp_decode(&ker_extra.T2.P1.x.re, (const void *) ((uint64_t[6]) {
        11212262540260969462U, 14553670370712682877U, 15014121696441355456U, 8542031148948179885U, 15786278263998617412U, 2260315967354622393U
    }));
    fp_decode(&ker_extra.T2.P1.x.im, (const void *) ((uint64_t[6]) {
        11915536837815516151U, 17462817145947734144U, 4551471595819947812U, 16077485286666381921U, 13519280559144078978U, 4262266575832993362U
    }));
    fp_decode(&ker_extra.T2.P1.z.re, (const void *) ((uint64_t[6]) {
        17240851651162568808U, 5862294420832380836U, 315049380557599025U, 1325150717198397104U, 4895224837204323169U, 527917906497852749U
    }));
    fp_decode(&ker_extra.T2.P1.z.im, (const void *) ((uint64_t[6]) {
        13462200890891550218U, 7201126103748793417U, 17544721750420029806U, 10723377278899051508U, 5617589786139969635U, 1422554973146496481U
    }));
    fp_decode(&ker_extra.T1m2.P1.x.re, (const void *) ((uint64_t[6]) {
        944526487515511498U, 4713688779741231978U, 7888663548198851175U, 15737038624069134607U, 11510806608612585393U, 2295235808726503653U
    }));
    fp_decode(&ker_extra.T1m2.P1.x.im, (const void *) ((uint64_t[6]) {
        10848198606355450974U, 4759316291395459547U, 1832950856166850680U, 1366389528915656239U, 13760785072520826239U, 3512649750253524103U
    }));
    fp_decode(&ker_extra.T1m2.P1.z.re, (const void *) ((uint64_t[6]) {
        4742783992192409018U, 2238594963689975320U, 15569365584285862538U, 7797468050884211271U, 16923776540166379173U, 416500226316062364U
    }));
    fp_decode(&ker_extra.T1m2.P1.z.im, (const void *) ((uint64_t[6]) {
        10950405929876078776U, 6996969671379197511U, 4769440988778232213U, 3388417155614612156U, 17486761379281689549U, 214712246056648784U
    }));
    fp_decode(&ker_extra.T1.P2.x.re, (const void *) ((uint64_t[6]) {
        2173794184159755161U, 6572561223267785736U, 5927591770197387481U, 3316473679015973462U, 4403975258828138864U, 1657378238725734343U
    }));
    fp_decode(&ker_extra.T1.P2.x.im, (const void *) ((uint64_t[6]) {
        4092216402223182058U, 9342283963803196352U, 12749667557371394690U, 8386546483735244856U, 12812973091093896960U, 1730075688850116482U
    }));
    fp_decode(&ker_extra.T1.P2.z.re, (const void *) ((uint64_t[6]) {
        3553678351579315118U, 14238864476911060610U, 5477669842191920686U, 18356484448831532952U, 10832088840282116531U, 3200778795052642851U
    }));
    fp_decode(&ker_extra.T1.P2.z.im, (const void *) ((uint64_t[6]) {
        7246259493455557795U, 7539957305796746191U, 3065750997145654282U, 6015217879957320095U, 14962352944568185081U, 4115662880306087396U
    }));
    fp_decode(&ker_extra.T2.P2.x.re, (const void *) ((uint64_t[6]) {
        6744329977273358840U, 11768027509370614159U, 6164254307727278721U, 5560875525830351828U, 9505168455850809731U, 3037151450548547731U
    }));
    fp_decode(&ker_extra.T2.P2.x.im, (const void *) ((uint64_t[6]) {
        1723298380839135088U, 4476241328224686439U, 6221284958633850947U, 12249935946569640711U, 6420419248514300742U, 2754845048584419359U
    }));
    fp_decode(&ker_extra.T2.P2.z.re, (const void *) ((uint64_t[6]) {
        10544403421480505248U, 3251321406652377035U, 13950880253345445605U, 17298466640535368108U, 6851875506469121671U, 3706981903672427793U
    }));
    fp_decode(&ker_extra.T2.P2.z.im, (const void *) ((uint64_t[6]) {
        13414774916337156249U, 11174841031226856949U, 15890442124425162722U, 16962103596829289419U, 1968690576738262052U, 1097525240117560900U
    }));
    fp_decode(&ker_extra.T1m2.P2.x.re, (const void *) ((uint64_t[6]) {
        7184381968770228615U, 2511403257099753494U, 677695194271644818U, 11299104508953438609U, 2835329907314358393U, 482423168525036602U
    }));
    fp_decode(&ker_extra.T1m2.P2.x.im, (const void *) ((uint64_t[6]) {
        18401562011917962245U, 8064755714889477782U, 7072649598930293479U, 18019624433376906936U, 11764218289701639104U, 21415800138262697U
    }));
    fp_decode(&ker_extra.T1m2.P2.z.re, (const void *) ((uint64_t[6]) {
        4466131040545729967U, 12040578765232403530U, 974824102318233694U, 12016595542357572161U, 5969760918017045741U, 1275052137654781948U
    }));
    fp_decode(&ker_extra.T1m2.P2.z.im, (const void *) ((uint64_t[6]) {
        5429109422436471438U, 12963434899314018524U, 5779997846008532278U, 5395786969854249455U, 799588212538382558U, 646804226983821729U
    }));

    ec_curve_normalize_A24(&E12.E1);
    ec_curve_normalize_A24(&E12.E2);
    ec_curve_normalize_A24(&E34_expected.E1);
    ec_curve_normalize_A24(&E34_expected.E2);

    // We might not be computing the same T1m2 as the script.
    double_couple_point(&ker_exact.T1, &ker_extra.T1, &E12);
    double_couple_point(&ker_exact.T1, &ker_exact.T1, &E12);
    double_couple_point(&ker_exact.T2, &ker_extra.T1, &E12);
    double_couple_point(&ker_exact.T2, &ker_exact.T2, &E12);
    double_couple_point(&ker_exact.T1m2, &ker_extra.T1m2, &E12);
    double_couple_point(&ker_exact.T1m2, &ker_exact.T1m2, &E12);

    // Checking points are of the right order
    assert(test_point_order_twof(&ker_exact.T1.P1, &E12.E1, length));
    assert(test_point_order_twof(&ker_exact.T1.P2, &E12.E2, length));
    assert(test_point_order_twof(&ker_exact.T2.P1, &E12.E1, length));
    assert(test_point_order_twof(&ker_exact.T2.P2, &E12.E2, length));
    assert(test_point_order_twof(&ker_exact.T1m2.P1, &E12.E1, length));
    assert(test_point_order_twof(&ker_exact.T1m2.P2, &E12.E2, length));
    assert(test_point_order_twof(&ker_extra.T1.P1, &E12.E1, length+2));
    assert(test_point_order_twof(&ker_extra.T1.P2, &E12.E2, length+2));
    assert(test_point_order_twof(&ker_extra.T2.P1, &E12.E1, length+2));
    assert(test_point_order_twof(&ker_extra.T2.P2, &E12.E2, length+2));
    assert(test_point_order_twof(&ker_extra.T1m2.P1, &E12.E1, length+2));
    assert(test_point_order_twof(&ker_extra.T1m2.P2, &E12.E2, length+2));

    // input_pts_two = [(xP1, zero), (zero, xP2)]
    copy_point(&input_pts_two[0].P1, &ker_extra.T1.P1);
    ec_point_init(&input_pts_two[0].P2);
    ec_point_init(&input_pts_two[1].P1);
    copy_point(&input_pts_two[1].P2, &ker_extra.T1.P2);
    
    // input_pts_three = [(xP1, zero), (xQ1, zero), (xPmQ1, zero)]
    copy_point(&input_pts_three[0].P1, &ker_extra.T1.P1);
    ec_point_init(&input_pts_three[0].P2);
    copy_point(&input_pts_three[1].P1, &ker_extra.T2.P1);
    ec_point_init(&input_pts_three[1].P2);
    copy_point(&input_pts_three[2].P1, &ker_extra.T1m2.P1);
    ec_point_init(&input_pts_three[2].P2);
    
    int num_pts = 2;
    bool randomize = false;

    printf("Theta chain KAT tests");

    // ========== Act ==================
    theta_isogeny_chain_no_torsion_singe(&E34_actual, res_pts_two,
        &E12, &ker_exact, input_pts_two, length, num_pts, randomize);

    // ========== Assert ==================
    ec_j_inv(&j_expected, &E34_expected.E1);
    ec_j_inv(&j_actual, &E34_actual.E1);
    int is_E3_correct = fp2_is_equal(&j_expected, &j_actual);
    ec_j_inv(&j_expected, &E34_expected.E2);
    ec_j_inv(&j_actual, &E34_actual.E2);
    int is_E4_correct = fp2_is_equal(&j_expected, &j_actual);
    int curves_correct = is_E3_correct & is_E4_correct;

    int is_zero_E3 = ec_is_zero(&res_pts_two[0].P2);
    int is_zero_E4 = ec_is_zero(&res_pts_two[1].P1);
    int is_zero = is_zero_E3 & is_zero_E4;

    int on_curve_E3 = ec_test_is_on_curve(&E34_actual.E1, &res_pts_two[0].P1);
    int on_curve_E4 = ec_test_is_on_curve(&E34_actual.E2, &res_pts_two[1].P2);
    int on_curve = on_curve_E3 & on_curve_E4;

    int is_order4_P1 = test_point_order_twof(&res_pts_two[0].P1, &E34_actual.E1, 2);
    int is_order4_P2 = test_point_order_twof(&res_pts_two[1].P2, &E34_actual.E1, 2);
    int is_order4 = is_order4_P1 & is_order4_P2;
    
    passed = curves_correct & is_zero & on_curve & is_order4;
    print_test_res("chain 4x2x1 no torsion", passed);
    if(passed == 0) {
        return false;
    }

    return true;
}