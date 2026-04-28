from file_writer import FileWriter

def write_fp_header_file(dir, pp, v, strat):
    path = f"{dir}/fp.h"
    f = FileWriter(path)
    print(f"Writing {path}")

    f.line("#ifndef _FP_H")
    f.line("#define _FP_H")
    f.line()

    f.line("#include <stdint.h>")
    f.line("#include \"intrinsics_avx512.h\"")
    f.line()

    f.banner("Conversions between Fp representations")

    f.fun_comment([
        "Given src = (a, b, c, d, e, f, g, h) in 8x1",
        "it computes res1 = (a, c, e, g) and res2 = (b, d, f, h) in 4x2."
    ])
    f.line("void fp_8x1_to_4x2(__m512i* res1, __m512i* res2, __m512i* src);")
    f.line()

    f.fun_comment([
        "Given src1 = (a1, b1, c1, d1) and src2 = (a2, b2, c2, d2) in 4x2,",
        "it computes res = (a1, a2, b1, b2, c1, c2, d1, d2) in 8x1."
    ])
    f.line("void fp_4x2_to_8x1(__m512i* res, __m512i* src1, __m512i* src2);")
    f.line()

    f.fun_comment([
        "Given src=(a, b, c, d) in 4x2, it computes res1 = (a, c) and res2 = (b, d) in 2x4.",
        "We choose this interleaving pattern for the Fp2 2x2x2 sqrt,",
        "where res1 = re and res2 = im."
    ])
    f.line("void fp_4x2_to_2x4(__m512i* res1, __m512i* res2, __m512i* src);")
    f.line()

    f.fun_comment([
        "Given src1 = (a1, b1) and src2 = (a2, b2) in 2x4,",
        "it computes res = (a1, a2, b1, b2) in 4x2."
    ])
    f.line("void fp_2x4_to_4x2(__m512i* res, __m512i* src1, __m512i* src2);")
    f.line()

    f.banner("Unvectorized Fp inversion")

    f.fun_comment([
        "Unvectorized inversion mod p using 64-bit arithmetic.",
        f"It uses an internal Montgomery representation with R_64 = 2^(64*{v.num_words64})."
    ])
    f.line("void fp_inv_64(uint64_t* res, const uint64_t* src);")
    f.line()

    for fmt in ["8x1", "4x2", "2x4"]:
        f.banner(fmt)

        if fmt == "8x1":
            f.comment([
                "In the 8x1 format, we store eight Fp elements in NUM_LIMBS vectors.",
                "The i-th vector stores the i-th limb of each vector."
            ])
        elif fmt == "4x2":
            f.comment([
                "In the 4x2 format, we store four Fp elements a, b, c, d in N4x2 vectors."
                "The i-th vector stores (a_i, a_(i + N4x2), b_i, b_(i + N4x2), c_i, c_(i + N4x2), d_i, d_(i + N4x2))."
            ])
        else:
            f.comment([
                "In the 2x4 format, we store two Fp element a and b in N2x4 vectors."
                "The i-th vector stores (a_i, a_(i + N2x4), a_(i + 2*N2x4), a_(i + 3*N2x4), b_i, b_(i + N2x4), b_(i + 2*N2x4), b_(i + 3*N2x4))."
            ])
        f.line()    

        f.fun_comment("Sets all the Fp elements to 0.")
        f.line(f"void fp_setzero_{fmt}(__m512i* res);")
        f.line()

        f.fun_comment("If bits[i] = 1, the i-th Fp element is set to 1 (in Montgomery repr.)")
        f.line(f"void fp_set_binary_{fmt}(__m512i* res, __mmask8 bits);")
        f.line()

        f.fun_comment("Copies a to res.")
        f.line(f"void fp_copy_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment("Sets each Fp element in res to a.", params=[("a", f"The {pp.limb_len}-bit limbs of an Fp element stored in a uint64_t array")], output=[("res", f"{fmt[0]} copies of a stored in {fmt}.")])
        f.line(f"void fp_duplicate_{fmt}(__m512i* res, const uint64_t* a);")
        f.line()

        f.fun_comment(f"Returns {fmt[0]}-bit mask where the i-th bit is 1 iff the i-th Fp elements are equal.")
        f.line(f"__mmask8 fp_is_equal_{fmt}(__m512i* a, __m512i* b);")
        f.line()

        f.fun_comment(f"Returns {fmt[0]}-bit mask where the i-th bit is 1 iff the i-th Fp element is zero.")
        f.line(f"__mmask8 fp_is_zero_{fmt}(__m512i* a);")
        f.line()

        f.fun_comment(f"Copies the i-th Fp element from a1 to res if the i-th bit of s is set, else it copies from a0.")
        f.line(f"void fp_select_{fmt}(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s);")
        f.line()

        f.fun_comment(f"res = a + b (mod p)")
        f.line(f"void fp_add_{fmt}(__m512i* res, __m512i* a, __m512i* b);")
        f.line()

        f.fun_comment(f"res = a - b (mod p)")
        f.line(f"void fp_sub_{fmt}(__m512i* res, __m512i* a, __m512i* b);")
        f.line()

        f.fun_comment("res = -a (mod p)")
        f.line(f"void fp_neg_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment([
            "Mixed addition and subtraction.",
            "If add_mask[i]=1, res[i] = a[i] + b[i] (mod p), else res[i] = a[i] - b[i] (mod p)."
        ])
        f.line(f"void fp_mixaddsub_{fmt}(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);")
        f.line()

        f.fun_comment("Reduces from [0, 2p) to [0, p)")
        f.line(f"void fp_normalize_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment("res = a * b (mod p)")
        f.line(f"void fp_mul_{fmt}(__m512i* res, __m512i* a, __m512i* b);")
        f.line()

        f.fun_comment("res = a^2 (mod p)")
        f.line(f"void fp_sqr_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment("res = a/2 (mod p)")
        f.line(f"void fp_div2_{fmt}(__m512i* res, __m512i* src);")
        f.line()

        f.fun_comment("res = a/3 (mod p)")
        f.line(f"void fp_div3_{fmt}(__m512i* res, __m512i* src);")
        f.line()

        f.fun_comment("res = a^(-1) (mod p). Input 0 yields 0.")
        f.line(f"void fp_inv_{fmt}(__m512i* res, __m512i* src);")
        f.line()

        f.fun_comment("res = sqrt(a) (mod p)")
        f.line(f"void fp_sqrt_{fmt}(__m512i* res, __m512i* src);")
        f.line()

        f.fun_comment("res = a^((p-3)/4) mod p")
        f.line(f"void fp_exp3div4_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment("Converts a to the Montgomery representation. res = a * R (mod p)")
        f.line(f"void fp_tomont_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment("Converts a back from the Montgomery representation. res = a * R^(-1) (mod p)")
        f.line(f"void fp_frommont_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        if fmt != "2x4":
            f.fun_comment("res = a + b, multiprecision addition, no correction, no carry propagation")
            f.line(f"void mp_add_{fmt}(__m512i* res, __m512i* a, __m512i* b);")
            f.line()

            if fmt == "8x1":
                f.fun_comment("res = res - a - b, double-length multiprecision subtraction, no correction, but propagates carry")
            else:
                f.fun_comment("res = res - a - b, double-length multiprecision subtraction, no correction or carry propagation")
            f.line(f"void mp_dblsubx2_{fmt}(__m512i* res, __m512i* a, __m512i* b);")
            f.line()

            carry = "with" if fmt == "8x1" else "without"
            lines = [f"res = res - a, double-length multiprecision subtraction, {carry} carry propagation."]
            if strat.R_ext:
                lines.append("We always add p*R to make the result non-negative.")
            else:
                lines.append("If the result is negative, we add p*R.")
            f.fun_comment(lines)
            f.line(f"void mp_subx2_{fmt}(__m512i* res, __m512i* a);")
            f.line()

            lines[0] = f"double-length multiprecision mixed add sub, {carry} carry propagation."
            f.fun_comment(lines)
            f.line(f"void mp_mixaddsubx2_{fmt}(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask);")
            f.line()

            mod = "2p" if strat.mod_2p else "p"
            f.fun_comment(f"(a + {mod}) - b, multiprecision with no correction {", but" if fmt == "8x1" else "or"} carry propagation.")
            f.line(f"void mp_add_p_sub_{fmt}(__m512i* res, __m512i* a, __m512i* b);")
            f.line()

        f.fun_comment(f"res = a * b, multiprecision multiplication.")
        f.line(f"void mp_mul_{fmt}(__m512i* res, __m512i* a, __m512i* b);")
        f.line()

        f.fun_comment("res = a (mod p), Montgomery reduction.")
        f.line(f"void rdc_{fmt}(__m512i* res, __m512i* a);")
        f.line()

        f.fun_comment(f"Sampled a random Fp {fmt} element.")
        f.line(f"void rand_fp_{fmt}(__m512i* res);")
        f.line()

        f.fun_comment([
            f"Converts from Fp {fmt} with {pp.limb_len}-bit limbs to an array of {fmt[0]}*{v.num_words64} 64-bit words.", 
            f"The first {v.num_words64} words encode the first Fp element, and so on.",
            "The function does not change the Montgomery representation."
        ])
        f.line(f"void fp_limbs_to_64_{fmt}(uint64_t* res, __m512i* src);")
        f.line()

        f.fun_comment([
            f"It takes as input an arry of {fmt[0]}*{v.num_words64} 64-bit words, where the first {v.num_words64} words encode the first Fp element, and so on.",
            f"It converts the input to an Fp 8x1 element."
        ])
        f.line(f"void fp_64_to_limbs_{fmt}(__m512i* res, uint64_t* unvec);")
        f.line()

        if fmt == "8x1":
            f.fun_comment("The same as fp_limbs_to_64_8x1, except that it only extract every other Fp element (index 0, 2, 4, 6).")
            f.line("void fp_limbs_to_64_8x1_only_even_idx(uint64_t* res, __m512i* src);")
            f.line()
            
            f.fun_comment([
                "The same as fp_64_to_limbs_8x1, except that it only takes an array of 4 Fp elements a, b, c, d in unvectorized form, ",
                "and converts it to an Fp 8x1 of the form (a, 0, b, 0, c, 0, d, 0)."
            ])
            f.line("void fp_64_to_limbs_8x1_only_even_idx(__m512i* res, uint64_t* unvec);")
            f.line()

    f.line("#endif")