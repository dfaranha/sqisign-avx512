from file_writer import FileWriter
from params import Fp2Mode

def write_fp2_include(f, pp, v):
    f.line("#include \"fp.h\"")
    f.line("#include \"fp_consts.h\"")
    f.line("#include \"fp2.h\"")
    f.line("#include \"utils.h\"")
    f.line()

def fp2_setzero_8x1x1(f, pp, v):
    f.start("void fp2_setzero_8x1x1(__m512i* res)")

    f.line("fp_setzero_8x1(res);")
    f.line(f"fp_setzero_8x1(res+{pp.num_limbs});")

    f.end()

def fp2_copy_8x1x1(f, pp, v):
    f.start("void fp2_copy_8x1x1(__m512i* res, __m512i* a)")
    
    f.line("fp_copy_8x1(res, a);")
    f.line(f"fp_copy_8x1(res+{pp.num_limbs}, a+{pp.num_limbs});")

    f.end()

def fp2_is_equal_8x1x1(f, pp, v):
    f.start("int fp2_is_equal_8x1x1(__m512i* a, __m512i* b)")

    f.line("int lo = fp_is_equal_8x1(a, b);")
    f.line(f"int hi = fp_is_equal_8x1(a+{pp.num_limbs}, b+{pp.num_limbs});")
    f.line("return lo & hi;")

    f.end()

def fp2_is_zero_8x1x1(f, pp, v):
    f.start("int fp2_is_zero_8x1x1(__m512i* a)")

    f.line("int lo = fp_is_zero_8x1(a);")
    f.line(f"int hi = fp_is_zero_8x1(a+{pp.num_limbs});")
    f.line(f"return lo & hi;")

    f.end()

def fp2_add_8x1x1(f, pp, v):
    f.start("void fp2_add_8x1x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line("fp_add_8x1(res, a, b);")
    f.line(f"fp_add_8x1(res+{pp.num_limbs}, a+{pp.num_limbs}, b+{pp.num_limbs});")

    f.end()

def fp2_sub_8x1x1(f, pp, v):
    f.start("void fp2_sub_8x1x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line("fp_sub_8x1(res, a, b);")
    f.line(f"fp_sub_8x1(res+{pp.num_limbs}, a+{pp.num_limbs}, b+{pp.num_limbs});")

    f.end()

def fp2_neg_8x1x1(f, pp, v):
    f.start("void fp2_neg_8x1x1(__m512i* res, __m512i* a)")

    f.line("fp_neg_8x1(res, a);")
    f.line(f"fp_neg_8x1(res+{pp.num_limbs}, a+{pp.num_limbs});")

    f.end()

def fp2_mixaddsub_8x1x1(f, pp, v):
    f.start("void fp2_mixaddsub_8x1x1(__m512i* res, __m512i* a, __m512i* b,\n    const __mmask8 add_mask_re, const __mmask8 add_mask_im)")

    f.line("fp_mixaddsub_8x1(res, a, b, add_mask_re);")
    f.line(f"fp_mixaddsub_8x1(res+{pp.num_limbs}, a+{pp.num_limbs}, b+{pp.num_limbs}, add_mask_im);")

    f.end()

def fp2_mul_8x1x1(f, pp, v, strat):
    f.start("void fp2_mul_8x1x1(__m512i* res, __m512i* a, __m512i* b)")

    if strat.mul_8x1 == Fp2Mode.reduce_everything:
        fp2_mul_8x1x1_standard(f, pp, v)
    else:
        fp2_mul_8x1x1_lazy(f, pp, v)

    f.end()

def fp2_mul_8x1x1_standard(f, pp, v):
    
    f.line(f"__m512i s1[{pp.num_limbs}], s2[{pp.num_limbs}], s3[{pp.num_limbs}];")
    f.line()

    # f.line(f"__m512i zero[{pp.num_limbs}];")
    # f.line(f"fp_setzero_8x1(zero);")
    # f.line()

    f.line(f"fp_add_8x1(s1, a, a + {pp.num_limbs});")
    f.line(f"fp_add_8x1(s2, b, b + {pp.num_limbs});")
    f.line(f"fp_mul_8x1(s3, s1, s2);")
    f.line(f"fp_mul_8x1(s1, a, b);")
    f.line(f"fp_mul_8x1(s2, a + {pp.num_limbs}, b + {pp.num_limbs});")

    # f.comment("Normalize to [0, p) — rdc output can be in [0, 2p)")
    # f.line(f"fp_add_8x1(s1, s1, zero);")
    # f.line(f"fp_add_8x1(s2, s2, zero);")
    # f.line(f"fp_add_8x1(s3, s3, zero);")

    f.line(f"fp_sub_8x1(s3, s3, s1);")
    f.line(f"fp_sub_8x1(res + {pp.num_limbs}, s3, s2);")
    f.line(f"fp_sub_8x1(res, s1, s2);")

def fp2_mul_8x1x1_lazy(f, pp, v):
    f.line(f"__m512i tt1[{2*pp.num_limbs}], tt2[{2*pp.num_limbs}], tt3[{2*pp.num_limbs}];")
    f.line()

    f.line(f"mp_add_8x1(tt3, a, a + {pp.num_limbs});")
    f.line(f"mp_add_8x1(tt3 + {pp.num_limbs}, b, b + {pp.num_limbs});")
    f.line(f"mp_mul_8x1_karat(tt1, a, b);")
    f.line(f"mp_mul_8x1_karat(tt2, a + {pp.num_limbs}, b + {pp.num_limbs});")
    f.line(f"mp_mul_8x1_karat(tt3, tt3, tt3 + {pp.num_limbs});")
    f.line(f"mp_dblsubx2_8x1(tt3, tt1, tt2);") # Always >=0
    f.line(f"mp_subx2_8x1(tt1, tt2);") # Can be negative
    f.line(f"rdc_8x1(res + {pp.num_limbs}, tt3);")
    f.line(f"rdc_8x1(res, tt1);")

def fp2_sqr_8x1x1(f, pp, v, strat):
    f.start("void fp2_sqr_8x1x1(__m512i* res, __m512i* a)")

    if strat.sqr_8x1 == Fp2Mode.reduce_everything:
        fp2_sqr_8x1x1_standard(f, pp, v)
    else:
        fp2_sqr_8x1x1_lazy(f, pp, v, strat)

    f.end()

def fp2_sqr_8x1x1_standard(f, pp, v):
    f.line(f"__m512i s1[{pp.num_limbs}], s2[{pp.num_limbs}], s3[{pp.num_limbs}];")
    f.line()

    f.line(f"fp_add_8x1(s1, a, a + {pp.num_limbs});")
    f.line(f"fp_sub_8x1(s2, a, a + {pp.num_limbs});")
    f.line(f"fp_add_8x1(s3, a, a);")
    f.line(f"fp_mul_8x1(res + {pp.num_limbs}, s3, a + {pp.num_limbs});")
    f.line(f"fp_mul_8x1(res, s1, s2);")

def fp2_sqr_8x1x1_lazy(f, pp, v, strat):
    f.line(f"__m512i s1[{pp.num_limbs}], s2[{pp.num_limbs}], s3[{pp.num_limbs}];")
    f.line()

    f.line(f"mp_add_8x1(s1, a, a + {pp.num_limbs});")
    if strat.sqr_8x1 == Fp2Mode.reduce:
        f.line(f"fp_sub_8x1(s2, a, a + {pp.num_limbs});")
    else:
        f.line(f"mp_add_p_sub_8x1(s2, a, a + {pp.num_limbs});")
    f.line(f"mp_add_8x1(s3, a, a);")
    f.line(f"fp_mul_8x1(res, s1, s2);")
    f.line(f"fp_mul_8x1(res + {pp.num_limbs}, s3, a + {pp.num_limbs});")

def fp2_muli_8x1x1(f, pp, v):
    f.start("void fp2_muli_8x1x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{pp.num_limbs}];")
    f.line()

    f.line(f"fp_neg_8x1(t, a + {pp.num_limbs});")
    f.line(f"fp_copy_8x1(res + {pp.num_limbs}, a);")
    f.line(f"fp_copy_8x1(res, t);")

    f.end()

def fp2_div3_8x1x1(f, pp, v):
    f.start("void fp2_div3_8x1x1(__m512i* res, __m512i* src)")

    f.line("fp_div3_8x1(res, src);")
    f.line(f"fp_div3_8x1(res + {pp.num_limbs}, src + {pp.num_limbs});")

    f.end()

def fp2_inv_8x1x1(f, pp, v):
    f.start("void fp2_inv_8x1x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i t0[{pp.num_limbs}], t1[{pp.num_limbs}];")

    f.line("fp_sqr_8x1(t0, a);")
    f.line(f"fp_sqr_8x1(t1, a + {pp.num_limbs});")
    f.line("fp_add_8x1(t0, t0, t1);")
    f.line("fp_inv_8x1(t0, t0);")
    f.line(f"fp_mul_8x1(t1, a + {pp.num_limbs}, t0);")
    f.line("fp_mul_8x1(res, a, t0);")
    f.line(f"fp_neg_8x1(res + {pp.num_limbs}, t1);")

    f.end()

def fp2_sqrt_8x1x1(f, pp, v):
    f.start("void fp2_sqrt_8x1x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i x0[{pp.num_limbs}], x1[{pp.num_limbs}], t0[{pp.num_limbs}], t1[{pp.num_limbs}];")
    f.line("__mmask8 is_zero;")
    f.line()

    f.comment("From \"Optimized One-Dimensional SQIsign Verification on Intel and")
    f.comment("Cortex-M4\" by Aardal et al: https://eprint.iacr.org/2024/1563")
    f.line()

    f.comment("x0 = \\delta = sqrt(a0^2 + a1^2)")
    f.line("fp_sqr_8x1(x0, a);")
    f.line(f"fp_sqr_8x1(x1, a + {pp.num_limbs});")
    f.line("fp_add_8x1(x0, x0, x1);")
    f.line("fp_sqrt_8x1(x0, x0);")
    f.comment("If a1 = 0, there is a risk of \\delta = -a0, which makes x0 = 0 below.")
    f.comment("In that case, we restore the value \\delta = a0.")
    f.line(f"is_zero = fp_is_zero_8x1(a + {pp.num_limbs});")
    f.line("fp_select_8x1(x0, x0, a, is_zero);")
    f.comment("x0 = \\delta + a0, t0 = 2 * x0.")
    f.line("fp_add_8x1(x0, x0, a);")
    f.line("fp_add_8x1(t0, x0, x0);")
    f.comment("x1 = t0^(p-3)/4.")
    f.line("fp_copy_8x1(x1, t0);")
    f.line("fp_exp3div4_8x1(x1, x1);")
    f.comment("x0 = x0 * x1, x1 = x1 * a1, t1 = (2x0)^2.")
    f.line("fp_mul_8x1(x0, x0, x1);")
    f.line(f"fp_mul_8x1(x1, x1, a + {pp.num_limbs});")
    f.line("fp_add_8x1(t1, x0, x0);")
    f.line("fp_sqr_8x1(t1, t1);")
    f.comment("If t1 = t0, return x0 + x1*i, otherwise x1 - x0*i.")
    f.line("fp_sub_8x1(t0, t0, t1);")
    f.line("is_zero = fp_is_zero_8x1(t0);")
    f.line("fp_neg_8x1(t1, x0);")
    f.line("fp_copy_8x1(t0, x1);")
    f.line("fp_select_8x1(res, t0, x0, is_zero);")
    f.line(f"fp_select_8x1(res + {pp.num_limbs}, t1, x1, is_zero);")

    f.end()

def fp2_tomont_8x1x1(f, pp, v):
    f.start("void fp2_tomont_8x1x1(__m512i* res, __m512i* a)")

    f.line(f"fp_tomont_8x1(res, a);")
    f.line(f"fp_tomont_8x1(res + {pp.num_limbs}, a + {pp.num_limbs});")

    f.end()

def fp2_frommont_8x1x1(f, pp, v):
    f.start("void fp2_frommont_8x1x1(__m512i*res, __m512i* a)")

    f.line("fp_frommont_8x1(res, a);")
    f.line(f"fp_frommont_8x1(res + {pp.num_limbs}, a + {pp.num_limbs});")

    f.end()

def fp2_rand_8x1x1(f, pp, v):
    f.start("void rand_fp2_8x1x1(__m512i* res)")

    f.line("rand_fp_8x1(res);")
    f.line(f"rand_fp_8x1(res+{pp.num_limbs});")
    
    f.end()

def fp2_8x1x1(f, pp, v, strat):
    f.banner("8x1x1")

    fp2_setzero_8x1x1(f, pp, v)
    fp2_copy_8x1x1(f, pp, v)
    fp2_is_equal_8x1x1(f, pp, v)
    fp2_is_zero_8x1x1(f, pp, v)
    fp2_add_8x1x1(f, pp, v)
    fp2_sub_8x1x1(f, pp, v)
    fp2_neg_8x1x1(f, pp, v)
    fp2_mixaddsub_8x1x1(f, pp, v)

    fp2_mul_8x1x1(f, pp, v, strat)
    fp2_sqr_8x1x1(f, pp, v, strat)

    fp2_muli_8x1x1(f, pp, v)
    fp2_div3_8x1x1(f, pp, v)
    fp2_inv_8x1x1(f, pp, v)
    fp2_sqrt_8x1x1(f, pp, v)

    fp2_tomont_8x1x1(f, pp, v)
    fp2_frommont_8x1x1(f, pp, v)

    fp2_rand_8x1x1(f, pp, v)

def fp2_setzero_4x1x2(f, pp, v):
    f.start("void fp2_setzero_4x1x2(__m512i* res)")

    f.line("fp_setzero_4x2(res);")
    f.line(f"fp_setzero_4x2(res+{pp.n42});")

    f.end()

def fp2_copy_4x1x2(f, pp, v):
    f.start("void fp2_copy_4x1x2(__m512i* res, __m512i* a)")
    
    f.line("fp_copy_4x2(res, a);")
    f.line(f"fp_copy_4x2(res+{pp.n42}, a+{pp.n42});")

    f.end()

def fp2_is_equal_4x1x2(f, pp, v):
    f.start("int fp2_is_equal_4x1x2(__m512i* a, __m512i* b)")

    f.line("int lo = fp_is_equal_4x2(a, b);")
    f.line(f"int hi = fp_is_equal_4x2(a+{pp.n42}, b+{pp.n42});")
    f.line("return lo & hi;")

    f.end()

def fp2_is_zero_4x1x2(f, pp, v):
    f.start("int fp2_is_zero_4x1x2(__m512i* a)")

    f.line("int lo = fp_is_zero_4x2(a);")
    f.line(f"int hi = fp_is_zero_4x2(a+{pp.n42});")
    f.line(f"return lo & hi;")

    f.end()

def fp2_add_4x1x2(f, pp, v):
    f.start("void fp2_add_4x1x2(__m512i* res, __m512i* a, __m512i* b)")

    f.line("fp_add_4x2(res, a, b);")
    f.line(f"fp_add_4x2(res+{pp.n42}, a+{pp.n42}, b+{pp.n42});")

    f.end()

def fp2_sub_4x1x2(f, pp, v):
    f.start("void fp2_sub_4x1x2(__m512i* res, __m512i* a, __m512i* b)")

    f.line("fp_sub_4x2(res, a, b);")
    f.line(f"fp_sub_4x2(res+{pp.n42}, a+{pp.n42}, b+{pp.n42});")

    f.end()

def fp2_neg_4x1x2(f, pp, v):
    f.start("void fp2_neg_4x1x2(__m512i* res, __m512i* a)")

    f.line("fp_neg_4x2(res, a);")
    f.line(f"fp_neg_4x2(res+{pp.n42}, a+{pp.n42});")

    f.end()

def fp2_mixaddsub_4x1x2(f, pp, v):
    f.start("void fp2_mixaddsub_4x1x2(__m512i* res, __m512i* a, __m512i* b,\n    const __mmask8 add_mask_re, const __mmask8 add_mask_im)")

    f.line("fp_mixaddsub_4x2(res, a, b, add_mask_re);")
    f.line(f"fp_mixaddsub_4x2(res+{pp.n42}, a+{pp.n42}, b+{pp.n42}, add_mask_im);")

    f.end()

def fp2_mul_4x1x2(f, pp, v, strat):
    f.start("void fp2_mul_4x1x2(__m512i* res, __m512i* a, __m512i* b)")

    fp2_mul_4x1x2_lazy(f, pp, v)

    f.end()

def fp2_mul_4x1x2_standard(f, pp, v):
    f.line(f"__m512i s1[{pp.n42}], s2[{pp.n42}], s3[{pp.n42}];")
    f.line()

    f.line(f"fp_add_4x2(s1, a, a + {pp.n42});")
    f.line(f"fp_add_4x2(s2, b, b + {pp.n42});")
    f.line(f"fp_mul_4x2(s3, s1, s2);")
    f.line(f"fp_mul_4x2(s1, a, b);")
    f.line(f"fp_mul_4x2(s2, a + {pp.n42}, b + {pp.n42});")
    f.line(f"fp_sub_4x2(s3, s3, s1);")
    f.line(f"fp_sub_4x2(res + {pp.n42}, s3, s2);")
    f.line(f"fp_sub_4x2(res, s1, s2);")

def fp2_mul_4x1x2_lazy(f, pp, v):
    lenx2 = pp.n42 + pp.num_limbs + 1
    f.line(f"__m512i t1[{lenx2}], t2[{lenx2}], t3[{lenx2}];")
    f.line()

    f.line(f"mp_add_4x2(t3, a, a + {pp.n42});")
    f.line(f"mp_add_4x2(t3 + {pp.n42}, b, b + {pp.n42});")
    f.line(f"mp_mul_4x2(t1, a, b);")
    f.line(f"mp_mul_4x2(t2, a+{pp.n42}, b+{pp.n42});")
    f.line(f"mp_mul_4x2(t3, t3, t3+{pp.n42});")
    f.line(f"mp_dblsubx2_4x2(t3, t1, t2);")
    f.line(f"mp_subx2_4x2(t1, t2);")
    f.line(f"rdc_4x2(res+{pp.n42}, t3);")
    f.line(f"rdc_4x2(res, t1);")

def fp2_sqr_4x1x2(f, pp, v, strat):
    f.start("void fp2_sqr_4x1x2(__m512i* res, __m512i* a)")

    fp2_sqr_4x1x2_lazy(f, pp, v)

    f.end()

def fp2_sqr_4x1x2_standard(f, pp, v):
    f.line(f"__m512i s1[{pp.n42}], s2[{pp.n42}], s3[{pp.n42}];")
    f.line()

    f.line(f"fp_add_4x2(s1, a, a + {pp.n42});")
    f.line(f"fp_sub_4x2(s2, a, a + {pp.n42});")
    f.line(f"fp_add_4x2(s3, a, a);")
    f.line(f"fp_mul_4x2(res + {pp.n42}, s3, a + {pp.n42});")
    f.line(f"fp_mul_4x2(res, s1, s2);")

def fp2_sqr_4x1x2_lazy(f, pp, v):
    f.line(f"__m512i s1[{pp.n42}], s2[{pp.n42}], s3[{pp.n42}];")
    f.line()

    f.line(f"mp_add_4x2(s1, a, a + {pp.n42});")
    f.line(f"mp_add_p_sub_4x2(s2, a, a + {pp.n42});")
    f.line(f"mp_add_4x2(s3, a, a);")
    f.line(f"fp_mul_4x2(res + {pp.n42}, s3, a + {pp.n42});")
    f.line(f"fp_mul_4x2(res, s1, s2);")


def fp2_muli_4x1x2(f, pp, v):
    f.start("void fp2_muli_4x1x2(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{pp.n42}];")
    f.line()

    f.line(f"fp_neg_4x2(t, a + {pp.n42});")
    f.line(f"fp_copy_4x2(res + {pp.n42}, a);")
    f.line(f"fp_copy_4x2(res, t);")

    f.end()

def fp2_div3_4x1x2(f, pp, v):
    f.start("void fp2_div3_4x1x2(__m512i* res, __m512i* src)")

    f.line("fp_div3_4x2(res, src);")
    f.line(f"fp_div3_4x2(res + {pp.n42}, src + {pp.n42});")

    f.end()

def fp2_inv_4x1x2(f, pp, v):
    f.start("void fp2_inv_4x1x2(__m512i* res, __m512i* a)")

    f.line(f"__m512i t0[{pp.n42}], t1[{pp.n42}];")

    f.line("fp_sqr_4x2(t0, a);")
    f.line(f"fp_sqr_4x2(t1, a + {pp.n42});")
    f.line("fp_add_4x2(t0, t0, t1);")
    f.line("fp_inv_4x2(t0, t0);")
    f.line(f"fp_mul_4x2(t1, a + {pp.n42}, t0);")
    f.line("fp_mul_4x2(res, a, t0);")
    f.line(f"fp_neg_4x2(res + {pp.n42}, t1);")

    f.end()

def fp2_sqrt_4x1x2(f, pp, v):
    f.start("void fp2_sqrt_4x1x2(__m512i* res, __m512i* a)")

    f.line(f"__m512i x0[{pp.n42}], x1[{pp.n42}], t0[{pp.n42}], t1[{pp.n42}];")
    f.line("__mmask8 is_zero;")
    f.line()

    f.comment("From \"Optimized One-Dimensional SQIsign Verification on Intel and")
    f.comment("Cortex-M4\" by Aardal et al: https://eprint.iacr.org/2024/1563")
    f.line()

    f.comment("x0 = \\delta = sqrt(a0^2 + a1^2)")
    f.line("fp_sqr_4x2(x0, a);")
    f.line(f"fp_sqr_4x2(x1, a + {pp.n42});")
    f.line("fp_add_4x2(x0, x0, x1);")
    f.line("fp_sqrt_4x2(x0, x0);")
    f.comment("If a1 = 0, there is a risk of \\delta = -a0, which makes x0 = 0 below.")
    f.comment("In that case, we restore the value \\delta = a0.")
    f.line(f"is_zero = fp_is_zero_4x2(a + {pp.n42});")
    f.line("fp_select_4x2(x0, x0, a, is_zero);")
    f.comment("x0 = \\delta + a0, t0 = 2 * x0.")
    f.line("fp_add_4x2(x0, x0, a);")
    f.line("fp_add_4x2(t0, x0, x0);")
    f.comment("x1 = t0^(p-3)/4.")
    f.line("fp_copy_4x2(x1, t0);")
    f.line("fp_exp3div4_4x2(x1, x1);")
    f.comment("x0 = x0 * x1, x1 = x1 * a1, t1 = (2x0)^2.")
    f.line("fp_mul_4x2(x0, x0, x1);")
    f.line(f"fp_mul_4x2(x1, x1, a + {pp.n42});")
    f.line("fp_add_4x2(t1, x0, x0);")
    f.line("fp_sqr_4x2(t1, t1);")
    f.comment("If t1 = t0, return x0 + x1*i, otherwise x1 - x0*i.")
    f.line("fp_sub_4x2(t0, t0, t1);")
    f.line("is_zero = fp_is_zero_4x2(t0);")
    f.line("fp_neg_4x2(t1, x0);")
    f.line("fp_copy_4x2(t0, x1);")
    f.line("fp_select_4x2(res, t0, x0, is_zero);")
    f.line(f"fp_select_4x2(res + {pp.n42}, t1, x1, is_zero);")

    f.end()

def fp2_tomont_4x1x2(f, pp, v):
    f.start("void fp2_tomont_4x1x2(__m512i* res, __m512i* a)")

    f.line(f"fp_tomont_4x2(res, a);")
    f.line(f"fp_tomont_4x2(res + {pp.n42}, a + {pp.n42});")

    f.end()

def fp2_frommont_4x1x2(f, pp, v):
    f.start("void fp2_frommont_4x1x2(__m512i*res, __m512i* a)")

    f.line("fp_frommont_4x2(res, a);")
    f.line(f"fp_frommont_4x2(res + {pp.n42}, a + {pp.n42});")

    f.end()

def fp2_rand_4x1x2(f, pp, v):
    f.start("void rand_fp2_4x1x2(__m512i* res)")
    
    f.line("rand_fp_4x2(res);")
    f.line(f"rand_fp_4x2(res + {pp.n42});")

    f.end()

def fp2_4x1x2(f, pp, v, strat):
    f.banner("4x1x2")

    fp2_setzero_4x1x2(f, pp, v)
    fp2_copy_4x1x2(f, pp, v)
    fp2_is_equal_4x1x2(f, pp, v)
    fp2_is_zero_4x1x2(f, pp, v)
    fp2_add_4x1x2(f, pp, v)
    fp2_sub_4x1x2(f, pp, v)
    fp2_neg_4x1x2(f, pp, v)
    fp2_mixaddsub_4x1x2(f, pp, v)

    fp2_mul_4x1x2(f, pp, v, strat)
    fp2_sqr_4x1x2(f, pp, v, strat)

    fp2_muli_4x1x2(f, pp, v)
    fp2_div3_4x1x2(f, pp, v)
    fp2_inv_4x1x2(f, pp, v)
    fp2_sqrt_4x1x2(f, pp, v)

    fp2_tomont_4x1x2(f, pp, v)
    fp2_frommont_4x1x2(f, pp, v)

    fp2_rand_4x1x2(f, pp, v)

def fp2_setzero_4x2x1(f, pp, v):
    f.start("void fp2_setzero_4x2x1(__m512i* res)")

    f.line("fp_setzero_8x1(res);")

    f.end()

def fp2_copy_4x2x1(f, pp, v):
    f.start("void fp2_copy_4x2x1(__m512i* res, __m512i* a)")
    
    f.line("fp_copy_8x1(res, a);")

    f.end()

def fp2_is_equal_4x2x1(f, pp, v):
    f.start("int fp2_is_equal_4x2x1(__m512i* a, __m512i* b)")

    f.line(f"int mask = fp_is_equal_8x1(a, b);")
    f.comment("a = b iff both halves are equal")  
    f.line("return (mask & (mask >> 1) & 0x1)")
    f.indent()
    f.line("| ((mask >> 1) & (mask >> 2) & 0x2)")
    f.line("| ((mask >> 2) & (mask >> 3) & 0x4)")
    f.line("| ((mask >> 3) & (mask >> 4) & 0x8);")
    f.unindent()

    f.end()

def fp2_is_zero_4x2x1(f, pp, v):
    f.start("int fp2_is_zero_4x2x1(__m512i* a)")

    f.line(f"int mask = fp_is_zero_8x1(a);")
    f.comment("a = 0 iff both halves are 0")  
    f.line("return (mask & (mask >> 1) & 0x1)")
    f.indent()
    f.line("| ((mask >> 1) & (mask >> 2) & 0x2)")
    f.line("| ((mask >> 2) & (mask >> 3) & 0x4)")
    f.line("| ((mask >> 3) & (mask >> 4) & 0x8);")
    f.unindent()

    f.end()

def fp2_add_4x2x1(f, pp, v):
    f.start("void fp2_add_4x2x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line("fp_add_8x1(res, a, b);")

    f.end()

def fp2_sub_4x2x1(f, pp, v):
    f.start("void fp2_sub_4x2x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line("fp_sub_8x1(res, a, b);")

    f.end()

def fp2_neg_4x2x1(f, pp, v):
    f.start("void fp2_neg_4x2x1(__m512i* res, __m512i* a)")

    f.line("fp_neg_8x1(res, a);")

    f.end()

def fp2_mixaddsub_4x2x1(f, pp, v):
    f.start("void fp2_mixaddsub_4x2x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")

    f.line("fp_mixaddsub_8x1(res, a, b, add_mask);")

    f.end()

def fp2_mul_4x2x1(f, pp, v, strat):
    f.start("void fp2_mul_4x2x1(__m512i* res, __m512i* a, __m512i* b)")

    fp2_mul_4x2x1_lazy(f, pp, v)

    f.end()

def fp2_mul_4x2x1_standard(f, pp, v):
    f.line(f"__m512i s1[{pp.num_limbs}], s2[{pp.num_limbs}], s3[{pp.num_limbs}];")
    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0")
    f.line("s2[i] = VSHUFFLE(a[i], 0xee); // a1, a1")
    f.line("s3[i] = VSHUFFLE(b[i], 0x4e); // b0, b1")

    f.unindent()
    f.line("}")

    f.line("fp_mul_8x1(s1, s1, b);")
    f.line("fp_mul_8x1(s2, s2, s3);")
    f.line("fp_mixaddsub_8x1(res, s1, s2, 0xaa);")

def fp2_mul_4x2x1_lazy(f, pp, v):
    f.line(f"__m512i t0[{2*pp.num_limbs}], t1[{2*pp.num_limbs}];")
    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("t0[i] = VSHUFFLE(a[i], 0x44); // a0, a0")
    f.line("t1[i] = VSHUFFLE(a[i], 0xee); // a1, a1")
    f.line(f"t1[i + {pp.num_limbs}] = VSHUFFLE(b[i], 0x4e); // b0, b1")

    f.unindent()
    f.line("}")

    f.line("mp_mul_8x1_karat(t0, t0, b);")
    f.line(f"mp_mul_8x1_karat(t1, t1, t1 + {pp.num_limbs});")
    f.line("mp_mixaddsubx2_8x1(t0, t0, t1, 0xaa);")
    f.line("rdc_8x1(res, t0);")

def fp2_sqr_4x2x1(f, pp, v, strat):
    f.start("void fp2_sqr_4x2x1(__m512i* res, __m512i* a)")

    if strat.sqr_8x1 == Fp2Mode.reduce_everything:
        fp2_sqr_4x2x1_standard(f, pp, v)
    else:
        fp2_sqr_4x2x1_lazy(f, pp, v, strat)

    f.end()

def fp2_sqr_4x2x1_standard(f, pp, v):
    f.line(f"__m512i s1[{pp.num_limbs}], s2[{pp.num_limbs}], s3[{pp.num_limbs}];")
    f.line("__mmask16 mask = 0b0011001100110011;")
    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0")
    f.line("s2[i] = VSHUFFLEMASKZ(mask, a[i], 0x0e); // 0, a1")
    f.line("s3[i] = VSHUFFLE(a[i], 0x4e); // a0, a1")

    f.unindent()
    f.line("}")

    f.line("fp_add_8x1(s1, s1, s3);")
    f.line("fp_sub_8x1(s2, a, s2);")
    f.line("fp_mul_8x1(res, s1, s2);")

def fp2_sqr_4x2x1_lazy(f, pp, v, strat):
    f.line(f"__m512i s1[{pp.num_limbs}], s2[{pp.num_limbs}], s3[{pp.num_limbs}];")
    f.line("__mmask16 mask = 0b0011001100110011;")
    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0")
    f.line("s2[i] = VSHUFFLEMASKZ(mask, a[i], 0x0e); // 0, a1")
    f.line("s3[i] = VSHUFFLE(a[i], 0x4e); // a0, a1")

    f.unindent()
    f.line("}")

    f.line("mp_add_8x1(s1, s1, s3);")
    if strat.sqr_8x1 == Fp2Mode.reduce:
        f.line(f"fp_sub_8x1(s2, a, s2);")
    else:
        f.line(f"mp_add_p_sub_8x1(s2, a, s2);")
    f.line("fp_mul_8x1(res, s1, s2);")

def fp2_mulx2_4x2x1(f, pp, v):
    f.comment("res1 = a1 * b1, res2 = a2 * b2")
    f.start("void fp2_mulx2_4x2x1(__m512i* res1, __m512i* res2, __m512i* a1, __m512i* b1, __m512i* a2, __m512i* b2)")

    f.line(f"__m512i t0[{2*pp.num_limbs}], t1[{2*pp.num_limbs}];")
    f.line()

    f.line("fp2_4x2x1_to_8x1x1(t0, a1, a2);")
    f.line("fp2_4x2x1_to_8x1x1(t1, b1, b2);")
    f.line("fp2_mul_8x1x1(t0, t0, t1);")
    f.line("fp2_8x1x1_to_4x2x1(res1, res2, t0);")

    f.end()

def fp2_sqrx2_4x2x1(f, pp, v):
    f.comment("res1 = a1^2, res2 = a2^a")
    f.start("void fp2_sqrx2_4x2x1(__m512i* res1, __m512i* res2, __m512i* a1, __m512i* a2)")

    f.line(f"__m512i t[{2*pp.num_limbs}];")
    f.line()

    f.line("fp2_4x2x1_to_8x1x1(t, a1, a2);")
    f.line("fp2_sqr_8x1x1(t, t);")
    f.line("fp2_8x1x1_to_4x2x1(res1, res2, t);")

    f.end()

def fp2_muli_4x2x1(f, pp, v):
    f.start("void fp2_muli_4x2x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{pp.num_limbs}];")
    f.line()

    f.line("fp_setzero_8x1(t);")
    f.line("fp_mixaddsub_8x1(t, t, a, 0x55);")

    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()
    f.line(f"res[i] = VSHUFFLE(t[i], 0x4e);")
    f.unindent()
    f.line("}")

    f.end()

def fp2_div3_4x2x1(f, pp, v):
    f.start("void fp2_div3_4x2x1(__m512i* res, __m512i* src)")

    f.line("fp_div3_8x1(res, src);")

    f.end()

def fp2_inv_4x2x1(f, pp, v):
    f.start("void fp2_inv_4x2x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i re[{pp.n42}], im[{pp.n42}], t0[{pp.n42}], t1[{pp.n42}];")
    f.line("fp_8x1_to_4x2(re, im, a);")
    f.line()

    f.line("fp_sqr_4x2(t0, re);")
    f.line("fp_sqr_4x2(t1, im);")
    f.line("fp_add_4x2(t0, t0, t1);")
    f.line("fp_inv_4x2(t0, t0);")
    f.line("fp_mul_4x2(re, re, t0);")
    f.line("fp_mul_4x2(im, im, t0);")
    f.line("fp_neg_4x2(im, im);")
    f.line()

    f.line("fp_4x2_to_8x1(res, re, im);")

    f.end()

def fp2_sqrt_4x2x1(f, pp, v):
    f.start("void fp2_sqrt_4x2x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i re[{pp.n42}], im[{pp.n42}], x0[{pp.n42}], x1[{pp.n42}], t0[{pp.n42}], t1[{pp.n42}];")
    f.line("__mmask8 is_zero;")
    f.line()

    f.comment("From \"Optimized One-Dimensional SQIsign Verification on Intel and")
    f.comment("Cortex-M4\" by Aardal et al: https://eprint.iacr.org/2024/1563")
    f.line()

    f.line("fp_8x1_to_4x2(re, im, a);")
    f.line()

    f.comment("x0 = \\delta = sqrt(a0^2 + a1^2)")
    f.line("fp_sqr_4x2(x0, re);")
    f.line("fp_sqr_4x2(x1, im);")
    f.line("fp_add_4x2(x0, x0, x1);")
    f.line("fp_sqrt_4x2(x0, x0);")
    f.comment("If a1 = 0, there is a risk of \\delta = -a0, which makes x0 = 0 below.")
    f.comment("In that case, we restore the value \\delta = a0.")
    f.line("is_zero = fp_is_zero_4x2(im);")
    f.line("fp_select_4x2(x0, x0, re, is_zero);")
    f.comment("x0 = \\delta + a0, t0 = 2 * x0.")
    f.line("fp_add_4x2(x0, x0, re);")
    f.line("fp_add_4x2(t0, x0, x0);")
    f.comment("x1 = t0^(p-3)/4.")
    f.line("fp_copy_4x2(x1, t0);")
    f.line("fp_exp3div4_4x2(x1, x1);")
    f.comment("x0 = x0 * x1, x1 = x1 * a1, t1 = (2x0)^2.")
    f.line("fp_mul_4x2(x0, x0, x1);")
    f.line("fp_mul_4x2(x1, x1, im);")
    f.line("fp_add_4x2(t1, x0, x0);")
    f.line("fp_sqr_4x2(t1, t1);")
    f.comment("If t1 = t0, return x0 + x1*i, otherwise x1 - x0*i.")
    f.line("fp_sub_4x2(t0, t0, t1);")
    f.line("is_zero = fp_is_zero_4x2(t0);")
    f.line("fp_neg_4x2(t1, x0);")
    f.line("fp_copy_4x2(t0, x1);")
    f.line("fp_select_4x2(re, t0, x0, is_zero);")
    f.line("fp_select_4x2(im, t1, x1, is_zero);")
    f.line()

    f.line("fp_4x2_to_8x1(res, re, im);")

    f.end()

def fp2_tomont_4x2x1(f, pp, v):
    f.start("void fp2_tomont_4x2x1(__m512i* res, __m512i* a)")

    f.line(f"fp_tomont_8x1(res, a);")

    f.end()

def fp2_frommont_4x2x1(f, pp, v):
    f.start("void fp2_frommont_4x2x1(__m512i*res, __m512i* a)")

    f.line("fp_frommont_8x1(res, a);")

    f.end()

def fp2_rand_4x2x1(f, pp, v):
    f.start("void rand_fp2_4x2x1(__m512i* res)")

    f.line("rand_fp_8x1(res);")

    f.end()

def fp2_4x2x1(f, pp, v, strat):
    f.banner("4x2x1")

    fp2_setzero_4x2x1(f, pp, v)
    fp2_copy_4x2x1(f, pp, v)
    fp2_is_equal_4x2x1(f, pp, v)
    fp2_is_zero_4x2x1(f, pp, v)
    fp2_add_4x2x1(f, pp, v)
    fp2_sub_4x2x1(f, pp, v)
    fp2_neg_4x2x1(f, pp, v)
    fp2_mixaddsub_4x2x1(f, pp, v)

    fp2_mul_4x2x1(f, pp, v, strat)
    fp2_sqr_4x2x1(f, pp, v, strat)
    fp2_mulx2_4x2x1(f, pp, v)
    fp2_sqrx2_4x2x1(f, pp, v)

    fp2_muli_4x2x1(f, pp, v)
    fp2_div3_4x2x1(f, pp, v)
    fp2_inv_4x2x1(f, pp, v)
    fp2_sqrt_4x2x1(f, pp, v)

    fp2_tomont_4x2x1(f, pp, v)
    fp2_frommont_4x2x1(f, pp, v)
    
    fp2_rand_4x2x1(f, pp, v)

def fp2_setzero_2x2x2(f, pp, v):
    f.start("void fp2_setzero_2x2x2(__m512i* res)")

    f.line("fp_setzero_4x2(res);")

    f.end()

def fp2_copy_2x2x2(f, pp, v):
    f.start("void fp2_copy_2x2x2(__m512i* res, __m512i* a)")
    
    f.line("fp_copy_4x2(res, a);")

    f.end()

def fp2_is_equal_2x2x2(f, pp, v):
    f.start("int fp2_is_equal_2x2x2(__m512i* a, __m512i* b)")

    f.line("int mask = fp_is_equal_4x2(a, b);")
    f.line("return (mask & (mask >> 1) & 0x1 ) | ((mask >> 1) & (mask >> 2) & 0x2); ")

    f.end()

def fp2_is_zero_2x2x2(f, pp, v):
    f.start("int fp2_is_zero_2x2x2(__m512i* a)")

    f.line("int mask = fp_is_zero_4x2(a);")
    f.line("return (mask & (mask >> 1) & 0x1 ) | ((mask >> 1) & (mask >> 2) & 0x2);")

    f.end()

def fp2_add_2x2x2(f, pp, v):
    f.start("void fp2_add_2x2x2(__m512i* res, __m512i* a, __m512i* b)")

    f.line("fp_add_4x2(res, a, b);")

    f.end()

def fp2_sub_2x2x2(f, pp, v):
    f.start("void fp2_sub_2x2x2(__m512i* res, __m512i* a, __m512i* b)")

    f.line("fp_sub_4x2(res, a, b);")

    f.end()

def fp2_neg_2x2x2(f, pp, v):
    f.start("void fp2_neg_2x2x2(__m512i* res, __m512i* a)")

    f.line("fp_neg_4x2(res, a);")

    f.end()

def fp2_mixaddsub_2x2x2(f, pp, v):
    f.start("void fp2_mixaddsub_2x2x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")

    f.line("fp_mixaddsub_4x2(res, a, b, add_mask);")

    f.end()

def fp2_mul_2x2x2(f, pp, v, strat):
    f.start("void fp2_mul_2x2x2(__m512i* res, __m512i* a, __m512i* b)")

    fp2_mul_2x2x2_lazy(f, pp, v)

    f.end()

def fp2_mul_2x2x2_standard(f, pp, v):
    f.line(f"__m512i s1[{pp.n42}], s2[{pp.n42}], s3[{pp.n42}];")
    f.line()

    f.line(f"for (int i = 0; i < {pp.n42}; i++) {{")
    f.indent()

    f.line("s1[i] = VPERM256(a[i], 0x44); // a0, a0")
    f.line("s2[i] = VPERM256(a[i], 0xee); // a1, a1")
    f.line("s3[i] = VPERM256(b[i], 0x4e); // b0, b1")

    f.unindent()
    f.line("}")

    f.line("fp_mul_4x2(s1, s1, b);")
    f.line("fp_mul_4x2(s2, s2, s3);")
    f.line("fp_mixaddsub_4x2(res, s1, s2, 0xcc);")

def fp2_mul_2x2x2_lazy(f, pp, v):
    lenx2 = pp.n42 + pp.num_limbs + 1
    f.line(f"__m512i t0[{lenx2}], t1[{lenx2}];")
    f.line()

    f.line(f"for (int i = 0; i < {pp.n42}; i++) {{")
    f.indent()

    f.line("t0[i] = VPERM256(a[i], 0x44); // a0, a0")
    f.line("t1[i] = VPERM256(a[i], 0xee); // a1, a1")
    f.line(f"t1[i + {pp.n42}] = VPERM256(b[i], 0x4e); // b0, b1")

    f.unindent()
    f.line("}")

    f.line("mp_mul_4x2(t0, t0, b);")
    f.line(f"mp_mul_4x2(t1, t1, t1+{pp.n42});")
    f.line("mp_mixaddsubx2_4x2(t0, t0, t1, 0xcc);")
    f.line("rdc_4x2(res, t0);")

def fp2_sqr_2x2x2(f, pp, v, strat):
    f.start("void fp2_sqr_2x2x2(__m512i* res, __m512i* a)")

    fp2_sqr_2x2x2_lazy(f, pp, v, strat)

    f.end()

def fp2_sqr_2x2x2_standard(f, pp, v):
    f.line(f"__m512i s1[{pp.n42}], s2[{pp.n42}];")
    f.line()

    f.line(f"for (int i = 0; i < {pp.n42}; i++) {{")
    f.indent()

    f.line("s1[i] = VPERM256(a[i], 0x44); // a0, a0")
    f.line("s2[i] = VPERM256(a[i], 0x4e); // a1, a0")

    f.unindent()
    f.line("}")

    f.line("fp_add_4x2(s1, s1, s2);")

    f.line(f"for (int i = 0; i < {pp.n42}; i++) {{")
    f.indent()

    f.line("s2[i] = VPERM256MASKZ(0x33, a[i], 0x4e); // a1, 0")

    f.unindent()
    f.line("}")
    f.line("fp_sub_4x2(s2, a, s2); // a0 - a1, a1")
    f.line("fp_mul_4x2(res, s1, s2);")

def fp2_sqr_2x2x2_lazy(f, pp, v, strat):
    f.line(f"__m512i s1[{pp.n42}], s2[{pp.n42}];")
    f.line()

    f.line(f"for (int i = 0; i < {pp.n42}; i++) {{")
    f.indent()

    f.line("s1[i] = VPERM256(a[i], 0x44); // a0, a0")
    f.line("s2[i] = VPERM256(a[i], 0x4e); // a1, a0")

    f.unindent()
    f.line("}")

    f.line("mp_add_4x2(s1, s1, s2);")

    f.line(f"for (int i = 0; i < {pp.n42}; i++) {{")
    f.indent()

    f.line("s2[i] = VPERM256MASKZ(0x33, a[i], 0x4e); // a1, 0")

    f.unindent()
    f.line("}")
    if strat.sqr_4x2 == Fp2Mode.reduce:
        f.line("fp_sub_4x2(s2, a, s2); // a0 - a1, a1")
    else:
        f.line("mp_add_p_sub_4x2(s2, a, s2); // a0 - a1, a1")
    f.line("fp_mul_4x2(res, s1, s2);")

def fp2_muli_2x2x2(f, pp, v):
    f.start("void fp2_muli_2x2x2(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{pp.n42}];")
    f.line()

    f.line("fp_setzero_4x2(t);")
    f.line("fp_mixaddsub_4x2(t, t, a, 0x33);")

    f.line(f"for (int i = 0; i < {pp.n42}; i++) {{")
    f.indent()
    f.line(f"res[i] = VPERM256(t[i], 0x4e);")
    f.unindent()
    f.line("}")

    f.end()

def fp2_div3_2x2x2(f, pp, v):
    f.start("void fp2_div3_2x2x2(__m512i* res, __m512i* src)")

    f.line("fp_div3_4x2(res, src);")

    f.end()

def fp2_inv_2x2x2(f, pp, v):
    f.start("void fp2_inv_2x2x2(__m512i* res, __m512i* a)")

    f.line(f"__m512i re[{pp.n24}], im[{pp.n24}], t0[{pp.n24}], t1[{pp.n24}];")
    f.line("fp_4x2_to_2x4(re, im, a);")
    f.line()

    f.line("fp_sqr_2x4(t0, re);")
    f.line("fp_sqr_2x4(t1, im);")
    f.line("fp_add_2x4(t0, t0, t1);")
    f.line("fp_inv_2x4(t0, t0);")
    f.line("fp_mul_2x4(re, re, t0);")
    f.line("fp_mul_2x4(im, im, t0);")
    f.line("fp_neg_2x4(im, im);")
    f.line()

    f.line("fp_2x4_to_4x2(res, re, im);")

    f.end()

def fp2_sqrt_2x2x2(f, pp, v):
    f.start("void fp2_sqrt_2x2x2(__m512i* res, __m512i* a)")

    f.line(f"__m512i re[{pp.n24}], im[{pp.n24}], x0[{pp.n24}], x1[{pp.n24}], t0[{pp.n24}], t1[{pp.n24}];")
    f.line("__mmask8 is_zero;")
    f.line()

    f.comment("From \"Optimized One-Dimensional SQIsign Verification on Intel and")
    f.comment("Cortex-M4\" by Aardal et al: https://eprint.iacr.org/2024/1563")
    f.line()

    f.line("fp_4x2_to_2x4(re, im, a);")
    f.line()

    f.comment("x0 = \\delta = sqrt(a0^2 + a1^2)")
    f.line("fp_sqr_2x4(x0, re);")
    f.line("fp_sqr_2x4(x1, im);")
    f.line("fp_add_2x4(x0, x0, x1);")
    f.line("fp_sqrt_2x4(x0, x0);")
    f.comment("If a1 = 0, there is a risk of \\delta = -a0, which makes x0 = 0 below.")
    f.comment("In that case, we restore the value \\delta = a0.")
    f.line("is_zero = fp_is_zero_2x4(im);")
    f.line("fp_select_2x4(x0, x0, re, is_zero);")
    f.comment("x0 = \\delta + a0, t0 = 2 * x0.")
    f.line("fp_add_2x4(x0, x0, re);")
    f.line("fp_add_2x4(t0, x0, x0);")
    f.comment("x1 = t0^(p-3)/4.")
    f.line("fp_copy_2x4(x1, t0);")
    f.line("fp_exp3div4_2x4(x1, x1);")
    f.comment("x0 = x0 * x1, x1 = x1 * a1, t1 = (2x0)^2.")
    f.line("fp_mul_2x4(x0, x0, x1);")
    f.line("fp_mul_2x4(x1, x1, im);")
    f.line("fp_add_2x4(t1, x0, x0);")
    f.line("fp_sqr_2x4(t1, t1);")
    f.comment("If t1 = t0, return x0 + x1*i, otherwise x1 - x0*i.")
    f.line("fp_sub_2x4(t0, t0, t1);")
    f.line("is_zero = fp_is_zero_2x4(t0);")
    f.line("fp_neg_2x4(t1, x0);")
    f.line("fp_copy_2x4(t0, x1);")
    f.line("fp_select_2x4(re, t0, x0, is_zero);")
    f.line("fp_select_2x4(im, t1, x1, is_zero);")
    f.line()

    f.line("fp_2x4_to_4x2(res, re, im);")

    f.end()

def fp2_tomont_2x2x2(f, pp, v):
    f.start("void fp2_tomont_2x2x2(__m512i* res, __m512i* a)")

    f.line(f"fp_tomont_4x2(res, a);")

    f.end()

def fp2_frommont_2x2x2(f, pp, v):
    f.start("void fp2_frommont_2x2x2(__m512i*res, __m512i* a)")

    f.line("fp_frommont_4x2(res, a);")

    f.end()

def fp2_rand_2x2x2(f, pp, v):
    f.start("void rand_fp2_2x2x2(__m512i* res)")

    f.line("rand_fp_4x2(res);")

    f.end()

def fp2_2x2x2(f, pp, v, strat):
    f.banner("2x2x2")

    fp2_setzero_2x2x2(f, pp, v)
    fp2_copy_2x2x2(f, pp, v)
    fp2_is_equal_2x2x2(f, pp, v)
    fp2_is_zero_2x2x2(f, pp, v)
    fp2_add_2x2x2(f, pp, v)
    fp2_sub_2x2x2(f, pp, v)
    fp2_neg_2x2x2(f, pp, v)
    fp2_mixaddsub_2x2x2(f, pp, v)

    fp2_mul_2x2x2(f, pp, v, strat)
    fp2_sqr_2x2x2(f, pp, v, strat)
    
    fp2_muli_2x2x2(f, pp, v)
    fp2_div3_2x2x2(f, pp, v)
    fp2_sqrt_2x2x2(f, pp, v)
    fp2_inv_2x2x2(f, pp, v)

    fp2_tomont_2x2x2(f, pp, v)
    fp2_frommont_2x2x2(f, pp, v)

    fp2_rand_2x2x2(f, pp, v)

def fp2_setzero_2x4x1(f, pp, v):
    f.start("void fp2_setzero_2x4x1(__m512i* res)")

    f.line("fp_setzero_8x1(res);")

    f.end()

def fp2_copy_2x4x1(f, pp, v):
    f.start("void fp2_copy_2x4x1(__m512i* res, __m512i* a)")
    
    f.line("fp_copy_8x1(res, a);")

    f.end()

def fp2_is_equal_2x4x1(f, pp, v):
    f.start("int fp2_is_equal_2x4x1(__m512i* a, __m512i* b)")

    f.line("int mask = fp_is_equal_8x1(a, b);")
    f.line("return (mask & (mask >> 2) & 0x1)")
    f.indent()
    f.line("| ((mask >> 3) & (mask >> 5) & 0x2);")
    f.unindent()

    f.end()

def fp2_is_zero_2x4x1(f, pp, v):
    f.start("int fp2_is_zero_2x4x1(__m512i* a)")

    f.line("int mask = fp_is_zero_8x1(a);")
    f.line("return (mask & (mask >> 2) & 0x1)")
    f.indent()
    f.line("| ((mask >> 3) & (mask >> 5) & 0x2);")
    f.unindent()

    f.end()

def fp2_add_2x4x1(f, pp, v):
    f.start("void fp2_add_2x4x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line("fp_add_8x1(res, a, b);")

    f.end()

def fp2_sub_2x4x1(f, pp, v):
    f.start("void fp2_sub_2x4x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line("fp_sub_8x1(res, a, b);")

    f.end()

def fp2_neg_2x4x1(f, pp, v):
    f.start("void fp2_neg_2x4x1(__m512i* res, __m512i* a)")

    f.line("fp_neg_8x1(res, a);")

    f.end()

def fp2_mixaddsub_2x4x1(f, pp, v):
    f.start("void fp2_mixaddsub_2x4x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")

    f.line("fp_mixaddsub_8x1(res, a, b, add_mask);")

    f.end()

def fp2_mul_2x4x1(f, pp, v):
    f.start("void fp2_mul_2x4x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i s1[{pp.num_limbs}], s2[{pp.num_limbs}];")
    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0, a1, a1")
    f.line("s2[i] = VPERM256(b[i], 0x28); // b0, b1, b1, b0")

    f.unindent()
    f.line("}")
    f.line("fp_mul_8x1(s1, s1, s2);")
    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("s2[i] = VPERM256MASKZ(0x55, s1[i], 0x32);")
    f.line("s1[i] = VPERM256MASKZ(0x55, s1[i], 0x10);")

    f.unindent()
    f.line("}")
    f.line("fp_mixaddsub_8x1(res, s1, s2, 0xcc);")

    f.end()

def fp2_sqr_2x4x1(f, pp, v):
    f.start("void fp2_sqr_2x4x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i s1[{pp.num_limbs}], s2[{pp.num_limbs}];")
    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("s1[i] = VSHUFFLE(a[i], 0x44); // a0, a0, a1, a1")
    f.line("s2[i] = VPERM256(a[i], 0x28); // a0, a1, a1, a0")

    f.unindent()
    f.line("}")
    f.line("fp_mul_8x1(s1, s1, s2);")
    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("s2[i] = VPERM256MASKZ(0x55, s1[i], 0x32);")
    f.line("s1[i] = VPERM256MASKZ(0x55, s1[i], 0x10);")

    f.unindent()
    f.line("}")
    f.line("fp_mixaddsub_8x1(res, s1, s2, 0xcc);")

    f.end()

def fp2_muli_2x4x1(f, pp, v):
    f.start("void fp2_muli_2x4x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{pp.num_limbs}];")
    f.line()

    f.line("fp_setzero_8x1(t);")
    f.line("fp_mixaddsub_8x1(t, t, a, 0x33);")

    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()
    f.line(f"res[i] = VPERM256(t[i], 0x4e);")
    f.unindent()
    f.line("}")

    f.end()

def fp2_div3_2x4x1(f, pp, v):
    f.start("void fp2_div3_2x4x1(__m512i* res, __m512i* src)")

    f.line("fp_div3_8x1(res, src);")

    f.end()

def fp2_inv_2x4x1(f, pp, v):
    f.start("void fp2_inv_2x4x1(__m512i* res, __m512i* src)")

    f.line(f"__m512i re[{pp.n42}], im[{pp.n42}], zero[{pp.n42}], t0[{pp.n24}], t1[{pp.n24}];")
    f.line()
    
    f.line("fp_8x1_to_4x2(re, zero, src); // re = (a_re, a_im, b_re, b_im), zero = (0, 0, 0, 0)")
    f.line("fp_4x2_to_2x4(re, im, re);")
    f.line()

    f.line("fp_sqr_2x4(t0, re);")
    f.line("fp_sqr_2x4(t1, im);")
    f.line("fp_add_2x4(t0, t0, t1);")
    f.line("fp_inv_2x4(t0, t0);")
    f.line("fp_mul_2x4(re, re, t0);")
    f.line("fp_mul_2x4(im, im, t0);")
    f.line("fp_neg_2x4(im, im);")
    f.line()

    f.line("fp_2x4_to_4x2(re, re, im);")
    f.line("fp_4x2_to_8x1(res, re, zero);")

    f.end()

def fp2_sqrt_2x4x1(f, pp, v):
    f.start("void fp2_sqrt_2x4x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i re[{pp.n42}], im[{pp.n24}], x0[{pp.n24}], x1[{pp.n24}], t0[{pp.n24}], t1[{pp.n24}];")
    f.line(f"__m512i zero[{pp.n42}];")
    f.line("__mmask8 is_zero;")
    f.line()

    f.comment("From \"Optimized One-Dimensional SQIsign Verification on Intel and")
    f.comment("Cortex-M4\" by Aardal et al: https://eprint.iacr.org/2024/1563")
    f.line()

    f.line("fp_8x1_to_4x2(re, zero, a);")
    f.line("fp_4x2_to_2x4(re, im, re);")
    f.line()

    f.comment("x0 = \\delta = sqrt(a0^2 + a1^2)")
    f.line("fp_sqr_2x4(x0, re);")
    f.line("fp_sqr_2x4(x1, im);")
    f.line("fp_add_2x4(x0, x0, x1);")
    f.line("fp_sqrt_2x4(x0, x0);")
    f.comment("If a1 = 0, there is a risk of \\delta = -a0, which makes x0 = 0 below.")
    f.comment("In that case, we restore the value \\delta = a0.")
    f.line("is_zero = fp_is_zero_2x4(im);")
    f.line("fp_select_2x4(x0, x0, re, is_zero);")
    f.comment("x0 = \\delta + a0, t0 = 2 * x0.")
    f.line("fp_add_2x4(x0, x0, re);")
    f.line("fp_add_2x4(t0, x0, x0);")
    f.comment("x1 = t0^(p-3)/4.")
    f.line("fp_copy_2x4(x1, t0);")
    f.line("fp_exp3div4_2x4(x1, x1);")
    f.comment("x0 = x0 * x1, x1 = x1 * a1, t1 = (2x0)^2.")
    f.line("fp_mul_2x4(x0, x0, x1);")
    f.line("fp_mul_2x4(x1, x1, im);")
    f.line("fp_add_2x4(t1, x0, x0);")
    f.line("fp_sqr_2x4(t1, t1);")
    f.comment("If t1 = t0, return x0 + x1*i, otherwise x1 - x0*i.")
    f.line("fp_sub_2x4(t0, t0, t1);")
    f.line("is_zero = fp_is_zero_2x4(t0);")
    f.line("fp_neg_2x4(t1, x0);")
    f.line("fp_copy_2x4(t0, x1);")
    f.line("fp_select_2x4(re, t0, x0, is_zero);")
    f.line("fp_select_2x4(im, t1, x1, is_zero);")
    f.line()

    
    f.line("fp_2x4_to_4x2(res, re, im);")
    f.line("fp_4x2_to_8x1(res, res, zero);")

    f.end()

def fp2_tomont_2x4x1(f, pp, v):
    f.start("void fp2_tomont_2x4x1(__m512i* res, __m512i* a)")

    f.line(f"fp_tomont_8x1(res, a);")

    f.end()

def fp2_frommont_2x4x1(f, pp, v):
    f.start("void fp2_frommont_2x4x1(__m512i*res, __m512i* a)")

    f.line("fp_frommont_8x1(res, a);")

    f.end()

def fp2_rand_2x4x1(f, pp, v):
    f.start("void rand_fp2_2x4x1(__m512i* res)")

    f.line("rand_fp_8x1(res);")
    f.line(f"for(int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()
    f.line("res[i] = VBLEND(0x55, VZERO(), res[i]);")
    f.unindent()
    f.line("}")

    f.end()

def fp2_2x4x1(f, pp, v):
    f.banner("2x4x1")

    fp2_setzero_2x4x1(f, pp, v)
    fp2_copy_2x4x1(f, pp, v)
    fp2_is_equal_2x4x1(f, pp, v)
    fp2_is_zero_2x4x1(f, pp, v)
    fp2_add_2x4x1(f, pp, v)
    fp2_sub_2x4x1(f, pp, v)
    fp2_neg_2x4x1(f, pp, v)
    fp2_mixaddsub_2x4x1(f, pp, v)

    fp2_mul_2x4x1(f, pp, v)
    fp2_sqr_2x4x1(f, pp, v)

    fp2_muli_2x4x1(f, pp, v)
    fp2_div3_2x4x1(f, pp, v)
    fp2_inv_2x4x1(f, pp, v)
    fp2_sqrt_2x4x1(f, pp, v)

    fp2_tomont_2x4x1(f, pp, v)
    fp2_frommont_2x4x1(f, pp, v)

    fp2_rand_2x4x1(f, pp, v)


def fp2_8x1x1_to_4x2x1(f, pp, v):
    f.start("void fp2_8x1x1_to_4x2x1(__m512i* res1, __m512i* res2, __m512i* src)")

    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line(f"__m512i re = src[i], im = src[i + {pp.num_limbs}];")
    f.line("res1[i] = VUNPACKLO(re, im);")
    f.line("res2[i] = VUNPACKHI(re, im);")

    f.unindent()
    f.line("}")

    f.end()

def fp2_8x1x1_to_first_4x2x1(f, pp, v):
    f.comment("Only extracts first 4x2x1")
    f.start("void fp2_8x1x1_to_first_4x2x1(__m512i* res, __m512i* src)")

    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line(f"res[i] = VUNPACKLO(src[i], src[i + {pp.num_limbs}]);")

    f.unindent()
    f.line("}")

    f.end()

def fp2_4x2x1_to_8x1x1(f, pp, v):
    f.start("void fp2_4x2x1_to_8x1x1(__m512i* res, __m512i* src1, __m512i* src2)")

    f.line("__m512i idx_re = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);")
    f.line("__m512i idx_im = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);")

    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line(f"__m512i s1 = src1[i], s2 = src2[i];")
    f.line("res[i] = VBLENDPERM(idx_re, s1, s2);")
    f.line(f"res[i+{pp.num_limbs}] = VBLENDPERM(idx_im, s1, s2);")

    f.unindent()
    f.line("}")

    f.end()

def fp2_4x2x1_duplicate_to_8x1x1(f, pp, v):
    f.start("void fp2_4x2x1_duplicate_to_8x1x1(__m512i* res, __m512i* src)")

    f.line("__m512i idx_re = _mm512_set_epi64(14, 6, 12, 4, 10, 2, 8, 0);")
    f.line("__m512i idx_im = _mm512_set_epi64(15, 7, 13, 5, 11, 3, 9, 1);")

    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("__m512i s = src[i];")
    f.line("res[i] = VBLENDPERM(idx_re, s, s);")
    f.line(f"res[i+{pp.num_limbs}] = VBLENDPERM(idx_im, s, s);")

    f.unindent()
    f.line("}")

    f.end()

def fp2_4x2x1_to_2x2x2(f, pp, v):
    f.start("void fp2_4x2x1_to_2x2x2(__m512i* res1, __m512i* res2, __m512i* src)")

    f.line("__m512i s, t;")
    f.line("__m512i idx1 = VSET(0, 8, 1, 9, 2, 10, 3, 11);")
    f.line("__m512i idx2 = VSET(4, 12, 5, 13, 6, 14, 7, 15);")

    f.line(f"for (int i = 0; i < {pp.n42 - pp.rem42}; i++) {{")
    f.indent()

    f.line(f"s = src[i], t = src[i+{pp.n42}];")
    f.line("res1[i] = VBLENDPERM(idx1, s, t);")
    f.line("res2[i] = VBLENDPERM(idx2, s, t);")

    f.unindent()
    f.line("}")

    if pp.rem42 == 1:
        f.line(f"s = src[{pp.n42-1}];")
        f.line(f"res1[{pp.n42-1}] = VBLENDPERM(idx1, s, VZERO());")
        f.line(f"res2[{pp.n42-1}] = VBLENDPERM(idx2, s, VZERO());")

    f.end()

def fp2_2x2x2_to_4x2x1(f, pp, v):
    f.start("void fp2_2x2x2_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2)")

    f.line("__m512i s, t;")
    f.line("__m512i idx1 = VSET(0, 2, 4, 6, 8, 10, 12, 14);")
    f.line("__m512i idx2 = VSET(1, 3, 5, 7, 9, 11, 13, 15);")

    f.line(f"for (int i = 0; i < {pp.n42 - pp.rem42}; i++) {{")
    f.indent()

    f.line("s = src1[i], t = src2[i];")
    f.line("res[i] = VBLENDPERM(idx1, s, t);")
    f.line(f"res[i+{pp.n42}] = VBLENDPERM(idx2, s, t);")

    f.unindent()
    f.line("}")

    if pp.rem42 == 1:
        f.line(f"res[{pp.n42-1}] = VBLENDPERM(idx1, src1[{pp.n42-1}], src2[{pp.n42-1}]);")

    f.end()

def fp2_4x2x1_to_2x4x1(f, pp, v):
    f.comment("Maps (a,b,c,d) in 4x2x1 to (a,b) and (c,d) in 2x4x1")
    f.start("void fp2_4x2x1_to_2x4x1(__m512i* res1, __m512i* res2, __m512i* src)")
    
    f.line("__m512i idx1 = VSET(0, 0, 1, 1, 2, 2, 3, 3);")
    f.line("__m512i idx2 = VSET(4, 4, 5, 5, 6, 6, 7, 7);")

    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("__m512i t = src[i];")
    f.line("res1[i] = VPERM512MASKZ(0x55, t, idx1);")
    f.line("res2[i] = VPERM512MASKZ(0x55, t, idx2);")

    f.unindent()
    f.line("}")

    f.end()

def fp2_2x4x1_to_4x2x1(f, pp, v):
    f.start("void fp2_2x4x1_to_4x2x1(__m512i* res, __m512i* src1, __m512i* src2)")

    f.line("__m512i idx = VSET(0, 2, 4, 6, 8, 10, 12, 14);")

    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()

    f.line("res[i] = VBLENDPERM(idx, src1[i], src2[i]);")

    f.unindent()
    f.line("}")

    f.end()

def fp2_2x2x2_to_2x4x1(f, pp, v):
    f.start("void fp2_2x2x2_to_2x4x1(__m512i* res, __m512i* src)")

    f.line("__m512i idx1 = VSET(0, 0, 2, 2, 4, 4, 6, 6);")
    f.line("__m512i idx2 = VSET(1, 1, 3, 3, 5, 5, 7, 7);")
    
    f.line(f"for (int i = 0; i < {pp.n42-pp.rem42}; i++) {{")
    f.indent()

    f.line("__m512i t = src[i];")
    f.line("res[i] = VPERM512MASKZ(0x55, t, idx1);")
    f.line(f"res[i+{pp.n42}] = VPERM512MASKZ(0x55, t, idx2);")

    f.unindent()
    f.line("}")

    if pp.rem42 == 1:
        f.line(f"res[{pp.n42-1}] = VPERM512MASKZ(0x55, src[{pp.n42-1}], idx1);")

    f.end()

def fp2_2x4x1_to_2x2x2(f, pp, v):
    f.start("void fp2_2x4x1_to_2x2x2(__m512i* res, __m512i* src)")

    f.line("__m512i idx = VSET(0, 8, 2, 10, 4, 12, 6, 14);")

    f.line(f"for (int i = 0; i < {pp.n42-pp.rem42}; i++) {{")
    f.indent()

    f.line(f"res[i] = VBLENDPERM(idx, src[i], src[i+{pp.n42}]);")

    f.unindent()
    f.line("}")

    if pp.rem42 == 1:
        f.line(f"res[{pp.n42-1}] = VBLENDPERM(idx, src[{pp.n42-1}], VZERO());")

    f.end()

def fp2_conversions(f, pp, v):
    f.banner("Conversions between Fp2 representations")

    fp2_8x1x1_to_4x2x1(f, pp, v)
    fp2_8x1x1_to_first_4x2x1(f, pp, v)
    fp2_4x2x1_to_8x1x1(f, pp, v)
    fp2_4x2x1_duplicate_to_8x1x1(f, pp, v)

    fp2_4x2x1_to_2x2x2(f, pp, v)
    fp2_2x2x2_to_4x2x1(f, pp, v)

    fp2_4x2x1_to_2x4x1(f, pp, v)
    fp2_2x4x1_to_4x2x1(f, pp, v)

    fp2_2x2x2_to_2x4x1(f, pp, v)
    fp2_2x4x1_to_2x2x2(f, pp, v)

def write_fp2_file(dir, pp, v, strat):
    path = f"{dir}/fp2.c"
    f = FileWriter(path)
    print(f"Writing {path}")

    write_fp2_include(f, pp, v)

    fp2_8x1x1(f, pp, v, strat)
    fp2_4x1x2(f, pp, v, strat)
    fp2_4x2x1(f, pp, v, strat)
    fp2_2x2x2(f, pp, v, strat)
    fp2_2x4x1(f, pp, v)
    fp2_conversions(f, pp, v)

