from file_writer import FileWriter
from gen_fp2 import (
    fp2_add_8x1x1, fp2_add_4x1x2, fp2_add_4x2x1, fp2_add_2x2x2, fp2_add_2x4x1,
    fp2_sub_8x1x1, fp2_sub_4x1x2, fp2_sub_4x2x1, fp2_sub_2x2x2, fp2_sub_2x4x1,
    fp2_neg_8x1x1, fp2_neg_4x1x2, fp2_neg_4x2x1, fp2_neg_2x2x2, fp2_neg_2x4x1,
    fp2_mixaddsub_8x1x1, fp2_mixaddsub_4x1x2, fp2_mixaddsub_4x2x1, fp2_mixaddsub_2x2x2, fp2_mixaddsub_2x4x1,
    fp2_mul_8x1x1, fp2_mul_4x1x2, fp2_mul_4x2x1, fp2_mul_2x2x2, fp2_mul_2x4x1,
    fp2_sqr_8x1x1, fp2_sqr_4x1x2, fp2_sqr_4x2x1, fp2_sqr_2x2x2, fp2_sqr_2x4x1,
    fp2_mulx2_4x2x1, fp2_sqrx2_4x2x1
)

def write_fp2_header_file(dir, pp, v, strat):
    path = f"{dir}/fp2.h"
    f = FileWriter(path)
    print(f"Writing {path}")
    
    f.line("#ifndef _FP2_H")
    f.line("#define _FP2_H")
    f.line()

    f.line("#include <stdint.h>")
    f.line("#include \"fp.h\"")
    f.line("#include \"intrinsics_avx512.h\"")
    f.line()

    f.comment([
        "Fp2 = Fp[X]/<X^2 + 1>",
        "Each element is of the form a_re + a_im * i, with a_re, a_im in Fp and i^2 = -1"
    ])
    f.line()

    f.banner("Conversions between representations")

    f.fun_comment([
        "Given (a, b, c, d, e, f, g, h) in 8x1x1,"
        "it computes (a, c, e, g) and (b, d, f, h) in 4x2x1."
    ])
    f.line(f"void fp2_8x1x1_to_4x2x1(__m512i* res1, __m512i* res2, __m512i* src);")
    f.line()

    f.fun_comment([
        "Given (a1, b1, c1, d1) and (a2, b2, c2, d2) in 4x2x1,",
        "it computes (a1, a2, b1, b2, c1, c2, d1, d2) in 8x1x1."
    ])
    f.line(f"void fp2_4x2x1_to_8x1x1(__m512i* res, __m512i* src1, __m512i* src2);")
    f.line()

    f.fun_comment([
        "Only extracts the first 4x2x1, i.e. from (a, b, c, d, e, f, g, h) in 8x1x1 it outputs (a, c, e, g) in 4x2x1."
    ])
    f.line(f"void fp2_8x1x1_to_first_4x2x1(__m512i* res, __m512i* src);")
    f.line()

    f.fun_comment([
        "Computes 4x2x1_to_8x1x1 with both inputs as src."
    ])
    f.line(f"void fp2_4x2x1_duplicate_to_8x1x1(__m512i* res, __m512i* src);")
    f.line()

    f.fun_comment([
        "Given (a, b, c, d) in 4x2x1 it computes (a, b) and (c, d) in 2x2x2."
    ])
    f.line(f"void fp2_4x2x1_to_2x2x2(__m512i* res1, __m512i* res2, __m512i* src);")
    f.line()

    f.fun_comment([
        "Given (a1, b1) and (a2, b2) in 2x2x2, it computes (a1, b1, a2, b2) in 4x2x1."
    ])
    f.line("void fp2_2x2x2_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2);")
    f.line()

    f.fun_comment([
        "Given (a, b, c, d) in 4x2x1 it computes (a, b) and (c, d) in 2x4x1."
    ])
    f.line("void fp2_4x2x1_to_2x4x1(__m512i* res1, __m512i* res2, __m512i* src);")
    f.line()

    f.fun_comment([
        "Given (a1, b1) and (a2, b2) in 2x4x1, it computes (a1, b1, a2, b2) in 4x2x1."
    ])
    f.line("void fp2_2x4x1_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2);")
    f.line()

    f.fun_comment([
        "Given (a, b) in 2x2x2 it computes (a, b) in 2x4x1."
    ])
    f.line("void fp2_2x2x2_to_2x4x1(__m512i* res, __m512i* src);")
    f.line()

    f.fun_comment([
        "Given (a, b) in 2x4x1 it computes (a, b) in 2x2x2."
    ])
    f.line("void fp2_2x4x1_to_2x2x2(__m512i* res, __m512i* src);")
    f.line()


    for fmt in ["8x1x1", "4x1x2", "4x2x1", "2x2x2", "2x4x1"]:
        f.banner(fmt)

        if fmt == "8x1x1":
            f.comment([
                "The 8x1x1 format stores eight Fp2 elements in 2 * NUM_LIMBS vectors.",
                "The first NUM_LIMBS vectors store the real Fp coefficient in the 8x1 format, the latter store the imaginary coefficient in 8x1."
            ])
        elif fmt == "4x1x2":
            f.comment([
                "The 4x1x2 format stores four Fp2 elements in 2 * N4x2 vectors.",
                "The first N4x2 vectors store the real Fp coefficient in the 4x2 format, and the latter store the imaginary coefficient in 4x2."
            ])
        elif fmt == "4x2x1":
            f.comment([
                "The 4x2x1 format stores four Fp2 elements a, b, c, d in NUM_LIMBS vectors.",
                "It stores (a_re, a_im, b_re, b_im, c_re, c_im, d_re, d_im) in 8x1."
            ])
        elif fmt == "2x2x2":
            f.comment([
                "The 2x2x2 format stores two Fp2 elements a and b in N4x2 vectors.",
                "It stores (a_re, a_im, b_re, b_im) in 4x2."
            ])
        elif fmt == "2x4x1":
            f.comment([
                "The 2x4x1 format stores two Fp2 elements a and b in NUM_LIMBS vectors.",
                "It stores (a_re, 0, a_im, 0, b_re, 0, b_im, 0) in 8x1."
            ])
        f.line()

        f.fun_comment("Sets all the Fp2 elements to 0.")
        f.line(f"void fp2_setzero_{fmt}(__m512i* res);")
        f.line()

        f.fun_comment("If bits[i] = 1, the i-th Fp2 element is set to 1 (in Montgomery repr.)")
        f.line(f"void fp2_set_binary_{fmt}(__m512i* res, __mmask8 bits);")
        f.line()

        f.fun_comment("Copies a to res")
        f.line(f"void fp2_copy_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment(f"Returns {fmt[0]}-bit mask where the i-th bit is 1 iff the i-th Fp2 elements are equal.")
        f.line(f"__mmask8 fp2_is_equal_{fmt}(__m512i* a, __m512i* b);")
        f.line()

        f.fun_comment(f"Returns {fmt[0]}-bit mask where the i-th bit is 1 iff the i-th Fp2 element is zero.")
        f.line(f"__mmask8 fp2_is_zero_{fmt}(__m512i* a);")
        f.line()

        f.fun_comment(f"Copies the i-th Fp2 element from a1 to res if the i-th bit of s is set, else it copies from a0.")
        f.line(f"void fp2_select_{fmt}(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);")
        f.line()

        f.fun_comment("res = a + b")
        if fmt == "8x1x1":
            fp2_add_8x1x1(f, pp, v, inline=True)
        elif fmt == "4x1x2":
            fp2_add_4x1x2(f, pp, v, inline=True)
        elif fmt == "4x2x1":
            fp2_add_4x2x1(f, pp, v, inline=True)
        elif fmt == "2x2x2":
            fp2_add_2x2x2(f, pp, v, inline=True)
        else:
            fp2_add_2x4x1(f, pp, v, inline=True)

        f.fun_comment("res = a - b")
        if fmt == "8x1x1":
            fp2_sub_8x1x1(f, pp, v, inline=True)
        elif fmt == "4x1x2":
            fp2_sub_4x1x2(f, pp, v, inline=True)
        elif fmt == "4x2x1":
            fp2_sub_4x2x1(f, pp, v, inline=True)
        elif fmt == "2x2x2":
            fp2_sub_2x2x2(f, pp, v, inline=True)
        else:
            fp2_sub_2x4x1(f, pp, v, inline=True)

        f.fun_comment("res = -a")
        if fmt == "8x1x1":
            fp2_neg_8x1x1(f, pp, v, inline=True)
        elif fmt == "4x1x2":
            fp2_neg_4x1x2(f, pp, v, inline=True)
        elif fmt == "4x2x1":
            fp2_neg_4x2x1(f, pp, v, inline=True)
        elif fmt == "2x2x2":
            fp2_neg_2x2x2(f, pp, v, inline=True)
        else:
            fp2_neg_2x4x1(f, pp, v, inline=True)

        if fmt in ["8x1x1", "4x1x2"]:
            f.fun_comment([
                "Mixed addition and subtraction.",
                "If add_mask_re[i]=1, re(res[i]) = re(a[i]) + re(b[i]), else re(res[i]) = re(a[i]) - re(b[i])",
                "Similarly, add_mask_im controls the operation applied to the imaginary coefficients."
            ])
            if fmt == "8x1x1":
                fp2_mixaddsub_8x1x1(f, pp, v, inline=True)
            else:
                fp2_mixaddsub_4x1x2(f, pp, v, inline=True)
        elif fmt in ["4x2x1", "2x2x2"]:
            f.fun_comment([
                "Mixed addition and subtraction. The operation applied is determined by add_mask.",
                "Bit 2*i and 2*i+1 determine the operation for the real and imaginary coefficient of the i-th Fp2 element."
            ])
            if fmt == "4x2x1":
                fp2_mixaddsub_4x2x1(f, pp, v, inline=True)
            else:
                fp2_mixaddsub_2x2x2(f, pp, v, inline=True)
            
        elif fmt == "2x4x1":
            f.fun_comment([
                "Mixed addition and subtraction. The operation applied is determined by the even index bits of add_mask.",
                "Bit 0 and 2 determine the operation applied to the real and imaginary coefficient of the first Fp2 element,"
                "bit 4 and 6 determine it for the second Fp2 element."
            ])
            fp2_mixaddsub_2x4x1(f, pp, v, inline=True)

        f.fun_comment("res = a * b")
        if fmt == "8x1x1":
            fp2_mul_8x1x1(f, pp, v, strat, inline=True)
        elif fmt == "4x1x2":
            fp2_mul_4x1x2(f, pp, v, strat, inline=True)
        elif fmt == "4x2x1":
            fp2_mul_4x2x1(f, pp, v, strat, inline=True)
        elif fmt == "2x2x2":
            fp2_mul_2x2x2(f, pp, v, strat, inline=True)
        else:
            fp2_mul_2x4x1(f, pp, v, inline=True)

        f.fun_comment("res = a^2")
        if fmt == "8x1x1":
            fp2_sqr_8x1x1(f, pp, v, strat, inline=True)
        elif fmt == "4x1x2":
            fp2_sqr_4x1x2(f, pp, v, strat, inline=True)
        elif fmt == "4x2x1":
            fp2_sqr_4x2x1(f, pp, v, strat, inline=True)
        elif fmt == "2x2x2":
            fp2_sqr_2x2x2(f, pp, v, strat, inline=True)
        else:
            fp2_sqr_2x4x1(f, pp, v, inline=True)

        if fmt == "4x2x1":
            f.fun_comment("Computes res1 = a1 * b1 and res2 = a2 * b2 using 8x1x1.")
            fp2_mulx2_4x2x1(f, pp, v, inline=True)

            f.fun_comment("Computes res1 = a1^2 and res2 = a2^2 using 8x1x1.")
            fp2_sqrx2_4x2x1(f, pp, v, inline=True)

        f.fun_comment("Multiplication by i.")
        f.line(f"void fp2_muli_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment("Complex conjugation. res = a_re - a_im * i")
        f.line(f"void fp2_conjugate_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment("res = a / 2")
        f.line(f"void fp2_div2_{fmt}(__m512i* res, __m512i* src);")
        f.line()

        f.fun_comment("res = a / 3")
        f.line(f"void fp2_div3_{fmt}(__m512i* res, __m512i* src);")
        f.line()

        f.fun_comment("res = a^-1")
        f.line(f"void fp2_inv_{fmt}(__m512i* res, __m512i* src);")
        f.line()

        f.fun_comment("res = sqrt(a)")
        f.line(f"void fp2_sqrt_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment("Converts a to the Montgomery representation. res = a * R")
        f.line(f"void fp2_tomont_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment("Converts a back from the Montgomery representation. res = a * R^-1")
        f.line(f"void fp2_frommont_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment("Samples a random Fp2 element.")
        f.line(f"void rand_fp2_{fmt}(__m512i* res);")
        f.line()

    f.line("#endif")