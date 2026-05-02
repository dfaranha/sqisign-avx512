#include <stdio.h>
#include "bench.h"
#include "fp.h"
#include "fp2.h"
#include "fp_consts.h"
#include "ec.h"
#include "theta.h"
#include "theta_gluing.h"
#include "theta_middle.h"
#include "theta_splitting.h"
#include "utils.h"

#define N_RUNS 10

void bench_theta(int iterations)
{
    uint64_t cycles[N_RUNS];
    uint64_t start, end, tmp;
    __m512i a[2*NUM_LIMBS], b[2*NUM_LIMBS], c[2*NUM_LIMBS];
    __m512i d[2*NUM_LIMBS], e[2*NUM_LIMBS], f[2*NUM_LIMBS];
    rand_fp2_8x1x1(a);
    rand_fp2_8x1x1(b);
    rand_fp2_8x1x1(c);
    rand_fp2_8x1x1(d);

    printf("Theta single benchmarks:\n");

    // Hadamard
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_had_single(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("had", cycles[(N_RUNS/2) -1], tmp);

    // mul
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_mul_single(c, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("mul", cycles[(N_RUNS/2) -1], tmp);

    // sqr 
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_sqr_single(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("sqr", cycles[(N_RUNS/2) -1], tmp);

    // double 
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_double_single(d, a, b, c);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("double", cycles[(N_RUNS/2) -1], tmp);

    // codomain
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_codomain_single(c, d, a, b, false, true);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("codomain (h1, h2 = 0, 1)", cycles[(N_RUNS/2) -1], tmp);

    // eval
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_eval_single(a, b, c, false, true);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_4x2x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("eval (h1, h2 = 0, 1)", cycles[(N_RUNS/2) -1], tmp);

    printf("Theta pair benchmarks:\n");

    // Hadamard
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_had_pair(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("had", cycles[(N_RUNS/2) -1], tmp);

    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_had_pair_alt(b, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("had alt", cycles[(N_RUNS/2) -1], tmp);

    // mul
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_mul_pair(c, a, b);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("mul", cycles[(N_RUNS/2) -1], tmp);

    // sqr 
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_sqr_pair(c, a);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("sqr", cycles[(N_RUNS/2) -1], tmp);

    // double 
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_double_pair(d, a, b, c);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, c);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("double", cycles[(N_RUNS/2) -1], tmp);

    // eval
    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        for(int j = 0; j < iterations; j++) {
            theta_eval_pair(a, b, c, false, true);
        }
        end = cpucycles();
        cycles[i] = (end - start)/iterations;
    }
    tmp = fp2_is_equal_8x1x1(a, b);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("eval (h1, h2 = 0, 1)", cycles[(N_RUNS/2) -1], tmp);

    // TODO: We need Python script to generate KATs
    int length = 188;
    theta_couple_curve_t E12, E34;
    theta_kernel_couple_points_t ker_extra;
    ec_curve_init(&E12.E1);
    ec_curve_init(&E12.E2);
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

    for(int i = 0; i < N_RUNS; i++) {
        start = cpucycles();
        theta_isogeny_chain_ver(&E34, &E12, &ker_extra, length);
        end = cpucycles();
        cycles[i] = (end - start);
    }
    tmp = fp2_is_equal(&E34.E1.A, &E12.E1.A);
    qsort(cycles, N_RUNS, sizeof(uint64_t), cmp_uint64);
    print_bench_res("theta_isogeny_chain_ver", cycles[(N_RUNS/2)-1], tmp);
}