#include <stdio.h>
#include <assert.h>

#include "ec.h"
#include "fp.h"
#include "fp2.h"
#include "fp_consts.h"
#include "fp_unvec.h"
#include "test.h"
#include "theta.h"
#include "theta_gluing.h"
#include "theta_middle.h"
#include "theta_splitting.h"
#include "utils.h"

#include "sqi_ec.h"
#include "sqi_ec_params.h"
#include "sqi_test_extras.h"

// The KAT data below is lvl3-only: elsewhere fp_decode silently zeroes it,
// leaving every point zero, so skip the test instead.
#define THETA_KAT_IS_FOR_THIS_LEVEL (NWORDS_FIELD == 6 && TORSION_EVEN_POWER == 376)

#include "test_output.h" // defines TESTS, guarded by the macro above

bool test_theta_KAT(int iterations)
{
#if !THETA_KAT_IS_FOR_THIS_LEVEL
    (void) iterations;
    printf("Theta chain KAT tests\n");
    printf("   chain KAT data is lvl3-only ............................ SKIPPED\n");
    return true;
#else
    // ========== Setup ==================
    int length = 188;
    theta_couple_curve_t E12, E34_expected, E34_actual;
    theta_kernel_couple_points_t ker_extra, ker_exact;
    theta_couple_point_t input_pts_two[2], input_pts_three[3];
    fp2_t j3_expected, j4_expected, j_actual;
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
    fp_decode(&ker_exact.T1.P1.x.re, (const void *) ((uint64_t[6]) {
        0x6a147cd0a61681b6U, 0x3e5479637807aa6bU, 0x25de6cf07736c51dU, 0xeb1556797b6f28b9U, 0x9c35fa8d974a65dcU, 0x6a18428c23d0b2fU
    }));
    fp_decode(&ker_exact.T1.P1.x.im, (const void *) ((uint64_t[6]) {
        0x83cf48cd8e987f95U, 0xc2f689f726aa1ef6U, 0x3e0c2e7047642470U, 0x8bff522b3c572086U, 0x25ddf89092ec0666U, 0x1e529674aeeee3b7U
    }));
    fp_decode(&ker_exact.T1.P1.z.re, (const void *) ((uint64_t[6]) {
        0x1U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T1.P1.z.im, (const void *) ((uint64_t[6]) {
        0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T1.P2.x.re, (const void *) ((uint64_t[6]) {
       0x2a58af54e2e93617U, 0x7ab4282cfe9f548U, 0x7b7b9147649a0a1eU, 0x2b416b113facf9f5U, 0xa1953fb7f6a8bdd0U, 0xbf9ccc5953c0f08U
    }));
    fp_decode(&ker_exact.T1.P2.x.im, (const void *) ((uint64_t[6]) {
        0xc6db3f7c315d039bU, 0x5eeca6043c5fbbc3U, 0x423641aedd20fb7bU, 0x9e0c788145d43fccU, 0x4d43bec560cae293U, 0xeca99ef95298e1U
    }));
    fp_decode(&ker_exact.T1.P2.z.re, (const void *) ((uint64_t[6]) {
        0x1U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T1.P2.z.im, (const void *) ((uint64_t[6]) {
        0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T2.P1.x.re, (const void *) ((uint64_t[6]) {
       0x32f43922c3e58b77U, 0x913c201c019ff1d4U, 0x7722904db377f3c3U, 0x8664a4ebda378e9eU, 0x5952e83cf7d3b7fU, 0x287f42ac35ee7e8dU
    }));
    fp_decode(&ker_exact.T2.P1.x.im, (const void *) ((uint64_t[6]) {
        0x3f37b54537e30662U, 0xe0ab7f38c01bafc2U, 0x29a27ce1521ce5eaU, 0xe92afdca2ad77418U, 0xb46346fdb0f1060U, 0x39910bb315da0c5dU
    }));
    fp_decode(&ker_exact.T2.P1.z.re, (const void *) ((uint64_t[6]) {
        0x1U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T2.P1.z.im, (const void *) ((uint64_t[6]) {
        0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T2.P2.x.re, (const void *) ((uint64_t[6]) {
       0x6ebc3dc55840b866U, 0x65ad54afc8082230U, 0x86090ac035dc5ecdU, 0x4c3bc6996c76de68U, 0xaedb784feae56f46U, 0x379956d3aefc97fU
    }));
    fp_decode(&ker_exact.T2.P2.x.im, (const void *) ((uint64_t[6]) {
        0x18bba02aff15c119U, 0xc4b3d4cd3de10bcdU, 0xeaa09d4948ff1ba9U, 0xc7715e306beaaff4U, 0xdb69a1a200b3dba5U, 0xa4412e060bdc7c6U
    }));
    fp_decode(&ker_exact.T2.P2.z.re, (const void *) ((uint64_t[6]) {
        0x1U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T2.P2.z.im, (const void *) ((uint64_t[6]) {
        0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T1m2.P1.x.re, (const void *) ((uint64_t[6]) {
       0x6df2ad5ccf7aebf5U, 0x1b6552c8b5d84a9dU, 0xcd9a47befd1112fbU, 0xc699c8a5a3688af3U, 0x37580ae68fb05d88U, 0x419dd4ecb571614U
    }));
    fp_decode(&ker_exact.T1m2.P1.x.im, (const void *) ((uint64_t[6]) {
        0x5ca02ac0bb63761cU, 0x1daac1f5048d4b4fU, 0x2aee54e53002bdb8U, 0x909fc30857b71137U, 0xf352915fe308504eU, 0x25540be3422a42c0U
    }));
    fp_decode(&ker_exact.T1m2.P1.z.re, (const void *) ((uint64_t[6]) {
        0x1U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T1m2.P1.z.im, (const void *) ((uint64_t[6]) {
        0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T1m2.P2.x.re, (const void *) ((uint64_t[6]) {
       0x94a50bd2b64c56abU, 0xbf20fadc24518d2fU, 0xb7ef42cf89befd1aU, 0xe86a041a676772d7U, 0x6107a7fe97e8a8caU, 0x349969be0fc951faU
    }));
    fp_decode(&ker_exact.T1m2.P2.x.im, (const void *) ((uint64_t[6]) {
        0xe6367aaf4a3c4b00U, 0x612a8ad8d3e95ee7U, 0xaa34444a1cd0c9c0U, 0xe62a8bb162c9695cU, 0x641a580871ebbf69U, 0x23f4b82d648eb59eU
    }));
    fp_decode(&ker_exact.T1m2.P2.z.re, (const void *) ((uint64_t[6]) {
        0x1U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    fp_decode(&ker_exact.T1m2.P2.z.im, (const void *) ((uint64_t[6]) {
        0x0U, 0x0U, 0x0U, 0x0U, 0x0U, 0x0U
    }));
    
    ec_curve_normalize_A24(&E12.E1);
    ec_curve_normalize_A24(&E12.E2);
    ec_curve_normalize_A24(&E34_expected.E1);
    ec_curve_normalize_A24(&E34_expected.E2);
    ec_j_inv(&j3_expected, &E34_expected.E1);
    ec_j_inv(&j4_expected, &E34_expected.E2);

    // We might not be computing the same T1m2 as the script.
    // Therefore, I decided to copy ker_exact instead of computing it.
    // double_couple_point(&ker_exact.T1, &ker_extra.T1, &E12);
    // double_couple_point(&ker_exact.T2, &ker_extra.T1, &E12);
    // double_couple_point(&ker_exact.T1m2, &ker_extra.T1m2, &E12);
    // double_couple_point(&ker_exact.T1, &ker_exact.T1, &E12);
    // double_couple_point(&ker_exact.T2, &ker_exact.T2, &E12);
    // double_couple_point(&ker_exact.T1m2, &ker_exact.T1m2, &E12);

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

    theta_normalize_chain_input_unvec(&E12, &ker_extra);
    theta_normalize_chain_input_unvec(&E12, &ker_exact);

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
    
    printf("Theta chain KAT tests\n");

    bool randomize_vals[2] = { false, true };
    #define NUM_INPUT_LISTS 2
    theta_couple_point_t* input_points_vals[NUM_INPUT_LISTS] = { 
        input_pts_two, input_pts_three 
    };
    int num_pts_vals[NUM_INPUT_LISTS] = { 2, 3 };
    theta_couple_point_t res_pts[3]; 
    ec_point_t RmS1, RpS1, RmS2, RpS2;
    bool success;
    bool verify = false; // the verification chain below sets the flag itself

    // with torsion
    for (int ridx = 0; ridx < 2 && passed; ridx++) {
        bool randomize = randomize_vals[ridx];
        for (int pidx = 0; pidx < NUM_INPUT_LISTS && passed; pidx++) {
            theta_couple_point_t* input_pts = input_points_vals[pidx];
            int num_pts = num_pts_vals[pidx];

            passed = theta_isogeny_chain(&E34_actual, res_pts,
                &E12, &ker_extra, input_pts, length, num_pts, randomize, verify);

            if (!passed) { break; }

            ec_normalize_curve(&E34_actual.E1);
            ec_normalize_curve(&E34_actual.E2);

            if (randomize) {
                ec_j_inv(&j_actual, &E34_actual.E1);
                int j3_eq_j3_expected = fp2_is_equal(&j3_expected, &j_actual);
                ec_j_inv(&j_actual, &E34_actual.E2);
                int j4_eq_j4_expected = fp2_is_equal(&j4_expected, &j_actual);
                passed = j3_eq_j3_expected & j4_eq_j4_expected;
            } else {
                int E3_equal = fp2_is_equal(&E34_expected.E1.A, &E34_actual.E1.A);
                int E4_equal = fp2_is_equal(&E34_expected.E2.A, &E34_actual.E2.A);
                passed = E3_equal & E4_equal;
            }
            if(!passed) { break; }

            for (int i = 0; i < num_pts; i++) {
                int on_curve_E3 = ec_test_is_on_curve(&E34_actual.E1, &(res_pts+i)->P1);
                int on_curve_E4 = ec_test_is_on_curve(&E34_actual.E2, &(res_pts+i)->P2);
                passed &= on_curve_E3 & on_curve_E4;
            }
            if (!passed) { break; }

            projective_difference_point(&RmS1, &res_pts->P1, &(res_pts+1)->P1, &E34_actual.E1);
            projective_difference_point(&RmS2, &res_pts->P2, &(res_pts+1)->P2, &E34_actual.E2);
            xADD(&RpS1, &res_pts->P1, &(res_pts+1)->P1, &RmS1);
            xADD(&RpS2, &res_pts->P2, &(res_pts+1)->P2, &RmS2);

            if (pidx == 0) {
                int is_order4_RmS1 = test_point_order_twof(&RmS1, &E34_actual.E1, 2);
                int is_order4_RpS1 = test_point_order_twof(&RpS1, &E34_actual.E1, 2);
                int is_order4_1 = is_order4_RmS1 | is_order4_RpS1;
                int is_order4_RmS2 = test_point_order_twof(&RmS2, &E34_actual.E2, 2);
                int is_order4_RpS2 = test_point_order_twof(&RpS2, &E34_actual.E2, 2);
                int is_order4_2 = is_order4_RmS2 | is_order4_RpS2;
                
                int is_torsion4_RmS1 = test_point_torsion_twof(&RmS1, &E34_actual.E1, 2);
                int is_torsion4_RpS1 = test_point_torsion_twof(&RpS1, &E34_actual.E1, 2);
                int is_torsion4_1 = is_torsion4_RmS1 | is_torsion4_RpS1;
                int is_torsion4_RmS2 = test_point_torsion_twof(&RmS2, &E34_actual.E2, 2);
                int is_torsion4_RpS2 = test_point_torsion_twof(&RpS2, &E34_actual.E2, 2);
                int is_torsion4_2 = is_torsion4_RmS2 | is_torsion4_RpS2;

                passed = (is_order4_1 & is_torsion4_2) | (is_order4_2 & is_torsion4_1);
            } else {
                ec_point_t T1 = (res_pts+2)->P1;
                ec_point_t T2 = (res_pts+2)->P2;
                ec_normalize_point(&RmS1);
                ec_normalize_point(&RpS1);
                ec_normalize_point(&RmS2);
                ec_normalize_point(&RpS2);
                ec_normalize_point(&T1);
                ec_normalize_point(&T2);
                int is_diff1 = fp2_is_equal(&RmS1.x, &T1.x) | fp2_is_equal(&RpS1.x, &T1.x);
                int is_diff2 = fp2_is_equal(&RmS2.x, &T2.x) | fp2_is_equal(&RpS2.x, &T2.x);
                passed = is_diff1 & is_diff2;
            }
        }
    }
    print_test_res("chain with torsion", passed);
    if(passed == 0) {
        return false;
    }

    // 4x2x1 verification, packing the inputs as common/verif.c does.
    // theta_normalize_chain_input_unvec above gives the lift what it assumes.
    __m512i ker_basis1[3*N4x2], ker_basis2[3*N4x2], A_34[NUM_LIMBS];
    __m512i normA_1[N4x2], normA_2[N4x2], normA24_1[N4x2], normA24_2[N4x2];
    __m512i ker_x[NUM_LIMBS], ker_y[NUM_LIMBS], ker_z[NUM_LIMBS];
    ec_point_t AC;
    fp2_t A_34_unvec[4];

    ec_point_unvec_to_1x2x2x2(ker_basis1, &ker_extra.T1.P1);
    ec_point_unvec_to_1x2x2x2(ker_basis1 + N4x2, &ker_extra.T2.P1);
    ec_point_unvec_to_1x2x2x2(ker_basis1 + 2*N4x2, &ker_extra.T1m2.P1);
    ec_point_unvec_to_1x2x2x2(ker_basis2, &ker_extra.T1.P2);
    ec_point_unvec_to_1x2x2x2(ker_basis2 + N4x2, &ker_extra.T2.P2);
    ec_point_unvec_to_1x2x2x2(ker_basis2 + 2*N4x2, &ker_extra.T1m2.P2);

    fp2_copy(&AC.x, &E12.E1.A);
    fp2_set_one(&AC.z);
    ec_point_unvec_to_1x2x2x2(normA_1, &AC);
    fp2_copy(&AC.x, &E12.E2.A);
    fp2_set_one(&AC.z);
    ec_point_unvec_to_1x2x2x2(normA_2, &AC);
    ec_point_unvec_to_1x2x2x2(normA24_1, &E12.E1.A24);
    ec_point_unvec_to_1x2x2x2(normA24_2, &E12.E2.A24);

    success = lift_basis_pair_normalized_2x2x2(
        ker_x, ker_y, ker_z, ker_basis1, ker_basis2, normA_1, normA_2, true
    );
    success &= theta_isogeny_chain_vec_verif(
        A_34, ker_x, ker_y, ker_z, normA_1, normA_2, normA24_1, normA24_2, length
    );

    fp2_4x2x1_to_unvectorized(A_34_unvec, A_34);
    ec_curve_init(&E34_actual.E1);
    ec_curve_init(&E34_actual.E2);
    E34_actual.E1.A = A_34_unvec[0];
    E34_actual.E1.C = A_34_unvec[1];
    E34_actual.E2.A = A_34_unvec[2];
    E34_actual.E2.C = A_34_unvec[3];

    ec_normalize_curve(&E34_actual.E1);
    ec_normalize_curve(&E34_actual.E2);

    int E3_equal = fp2_is_equal(&E34_expected.E1.A, &E34_actual.E1.A);
    int E4_equal = fp2_is_equal(&E34_expected.E2.A, &E34_actual.E2.A);
    passed = success & E3_equal & E4_equal;

    print_test_res("chain verification", passed);
    if(passed == 0) {
        return false;
    }

    // 8x1x1 no torsion
    for (int ridx = 0; ridx < 2 && passed; ridx++) {
        bool randomize = randomize_vals[ridx];
        for (int pidx = 0; pidx < NUM_INPUT_LISTS && passed; pidx++) {
            theta_couple_point_t* input_pts = input_points_vals[pidx];
            int num_pts = num_pts_vals[pidx];

            passed = theta_isogeny_chain_no_torsion(&E34_actual, res_pts,
                &E12, &ker_exact, input_pts, length, num_pts, randomize, verify);

            if (!passed) { break; }

            ec_normalize_curve(&E34_actual.E1);
            ec_normalize_curve(&E34_actual.E2);
            
            ec_j_inv(&j_actual, &E34_actual.E1);
            int j3_eq_j3_expected = fp2_is_equal(&j3_expected, &j_actual);
            int j3_eq_j4_expected = fp2_is_equal(&j4_expected, &j_actual);

            ec_j_inv(&j_actual, &E34_actual.E2);
            int j4_eq_j3_expected = fp2_is_equal(&j3_expected, &j_actual);
            int j4_eq_j4_expected = fp2_is_equal(&j4_expected, &j_actual);

            int same = j3_eq_j3_expected & j4_eq_j4_expected;
            int swapped = j3_eq_j4_expected & j4_eq_j3_expected;
            passed = same | swapped;
            if (!passed) { break; }

            for (int i = 0; i < num_pts; i++) {
                int on_curve_E3 = ec_test_is_on_curve(&E34_actual.E1, &(res_pts+i)->P1);
                int on_curve_E4 = ec_test_is_on_curve(&E34_actual.E2, &(res_pts+i)->P2);
                passed &= on_curve_E3 & on_curve_E4;
            }
            if (!passed) { break; }

            projective_difference_point(&RmS1, &res_pts->P1, &(res_pts+1)->P1, &E34_actual.E1);
            projective_difference_point(&RmS2, &res_pts->P2, &(res_pts+1)->P2, &E34_actual.E2);
            xADD(&RpS1, &res_pts->P1, &(res_pts+1)->P1, &RmS1);
            xADD(&RpS2, &res_pts->P2, &(res_pts+1)->P2, &RmS2);

            if (pidx == 0) {
                int is_order4_RmS1 = test_point_order_twof(&RmS1, &E34_actual.E1, 2);
                int is_order4_RpS1 = test_point_order_twof(&RpS1, &E34_actual.E1, 2);
                int is_order4_1 = is_order4_RmS1 | is_order4_RpS1;
                int is_order4_RmS2 = test_point_order_twof(&RmS2, &E34_actual.E2, 2);
                int is_order4_RpS2 = test_point_order_twof(&RpS2, &E34_actual.E2, 2);
                int is_order4_2 = is_order4_RmS2 | is_order4_RpS2;
                
                int is_torsion4_RmS1 = test_point_torsion_twof(&RmS1, &E34_actual.E1, 2);
                int is_torsion4_RpS1 = test_point_torsion_twof(&RpS1, &E34_actual.E1, 2);
                int is_torsion4_1 = is_torsion4_RmS1 | is_torsion4_RpS1;
                int is_torsion4_RmS2 = test_point_torsion_twof(&RmS2, &E34_actual.E2, 2);
                int is_torsion4_RpS2 = test_point_torsion_twof(&RpS2, &E34_actual.E2, 2);
                int is_torsion4_2 = is_torsion4_RmS2 | is_torsion4_RpS2;

                passed = (is_order4_1 & is_torsion4_2) | (is_order4_2 & is_torsion4_1);
            } else {
                ec_point_t T1 = (res_pts+2)->P1;
                ec_point_t T2 = (res_pts+2)->P2;
                ec_normalize_point(&RmS1);
                ec_normalize_point(&RpS1);
                ec_normalize_point(&RmS2);
                ec_normalize_point(&RpS2);
                ec_normalize_point(&T1);
                ec_normalize_point(&T2);
                int is_diff1 = fp2_is_equal(&RmS1.x, &T1.x) | fp2_is_equal(&RpS1.x, &T1.x);
                int is_diff2 = fp2_is_equal(&RmS2.x, &T2.x) | fp2_is_equal(&RpS2.x, &T2.x);
                passed = is_diff1 & is_diff2;
            }
        }
    }
    print_test_res("chain no torsion", passed);
    if(passed == 0) {
        return false;
    }

    // TODO: Randomize, eval points
    // for (int i = 0; i < 10 && passed; i++) {
    //     dim_two_isogeny_test_t test = TESTS[i];
    //     bool randomize = false;
    //     int num_pts = 0;
    //     theta_couple_point_t input_pts[0];
    //     passed = theta_isogeny_chain_pair(&E34_actual, res_pts,
    //             &test.domain, &test.kernel, input_pts, test.length, num_pts, randomize, verify);
    //     if (!passed) { break; }
        
    //     ec_normalize_curve(&E34_actual.E1);
    //     ec_normalize_curve(&E34_actual.E2);
    //     passed = fp2_is_equal(&test.codomain.E1.A, &E34_actual.E1.A) 
    //         & fp2_is_equal(&test.codomain.E2.A, &E34_actual.E2.A);
    // }
    // print_test_res("chain 8x1x1 KAT", passed);
    // if(passed == 0) {
    //     return false;
    // }

    return true;
#endif
}
