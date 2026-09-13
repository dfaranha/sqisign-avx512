from math import ceil
from gen_addchain import (write_dbl_and_add, write_fp_exp3_div4_naive, write_exp_function)
from params import MulNonnegMode

def fp_setzero_8x1(f, pp, v):
    f.start("void fp_setzero_8x1(__m512i* res)")
    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = VZERO();")
    f.end()

def fp_set_binary_8x1(f, pp, v):
    f.comment("If bits[i] = 1, the i-th Fp element is set to 1 in the Montgomery representation.")
    f.start("void fp_set_binary_8x1(__m512i* res, __mmask8 bits)")

    f.line("uint64_t m0 = -(bits & 0x1);")
    for i in range(1, 8):
        f.line(f"uint64_t m{i} = -((bits >> {i}) & 0x1);")
    f.line()
    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = VSET(R[{i}] & m0, R[{i}] & m1, R[{i}] & m2, R[{i}] & m3, R[{i}] & m4, R[{i}] & m5, R[{i}] & m6, R[{i}] & m7);")

    f.end()

def fp_copy_8x1(f, pp, v):
    f.start("void fp_copy_8x1(__m512i* res, __m512i* a)")
    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = a[{i}];")
    f.end()

def fp_duplicate_8x1(f, pp, v):
    f.start("void fp_duplicate_8x1(__m512i* res, const uint64_t* a)")
    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = VSETALL(a[{i}]);")
    f.end()

def fp_is_equal_8x1(f, pp, v, strat):
    f.start("__mmask8 fp_is_equal_8x1(__m512i* a, __m512i* b)")

    if strat.mod_2p:
        f.line(f"__m512i norm_a[{v.num_limbs}], norm_b[{v.num_limbs}];")
        f.line("fp_normalize_8x1(norm_a, a);")
        f.line("fp_normalize_8x1(norm_b, b);")
        f.line()
        var_a = "norm_a"
        var_b = "norm_b"
    else:
        var_a = "a"
        var_b = "b"

    f.line("__mmask8 acc = 0xff;")
    for i in range(pp.num_limbs):
        f.line(f"acc &= VCMPEQ({var_a}[{i}], {var_b}[{i}]);")
    f.line("return acc;")

    f.end()

def fp_is_zero_8x1(f, pp, v, strat):
    f.start("__mmask8 fp_is_zero_8x1(__m512i* a)")

    if strat.mod_2p:
        f.line(f"__m512i norm_a[{v.num_limbs}];")
    f.line(f"const __m512i zero = VZERO();")
    f.line()

    if strat.mod_2p:
        f.comment("Normalize a to [0, p)")
        f.line("fp_normalize_8x1(norm_a, a);")
        f.line()
        var_a = "norm_a"
    else:
        var_a = "a"

    f.line("__mmask8 acc = 0xff;")
    for i in range(pp.num_limbs):
        f.line(f"acc &= VCMPEQ({var_a}[{i}], zero);")
    f.line("return acc;")

    f.end()

def fp_select_8x1(f, pp, v):
    f.start("void fp_select_8x1(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)")

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = VBLEND(s, a0[{i}], a1[{i}]);")

    f.end()

def carry_prop_8x1(f, pp, v, len, tmp, mask, start_idx=0):
    f.comment("Carry propagation")
    for i in range(start_idx + 1, start_idx + len):
        f.line(f"{tmp}[{i}] = VADD({tmp}[{i}], VSRA({tmp}[{i-1}], {v.limb_len}));")
    for i in range(start_idx + len-1):
        f.line(f"{tmp}[{i}] = VAND({tmp}[{i}], {mask});")
    f.line()

def carry_prop_8x1_var(f, pp, v, len, tmp, mask, start_idx=0):
    f.comment("Carry propagation")
    for i in range(start_idx + 1, start_idx + len):
        f.line(f"{tmp}{i} = VADD({tmp}{i}, VSRA({tmp}{i-1}, {v.limb_len}));")
    for i in range(start_idx + len-1):
        f.line(f"{tmp}{i} = VAND({tmp}{i}, {mask});")
    f.line()

def carry_prop_8x1_swap(f, pp, v, len, tmp_arr, tmp_var, mask, start_idx=0):
    last_idx = start_idx + len -1
    assert pp.num_limbs >= 3

    f.comment("Carry propagation")
    f.line(f"{tmp_var}1 = VADD({tmp_arr}[1], VSRA({tmp_arr}[{0}], {v.limb_len}));")
    for i in range(start_idx + 2, last_idx):
        f.line(f"{tmp_var}{i} = VADD({tmp_arr}[{i}], VSRA({tmp_var}{i-1}, {v.limb_len}));")
    f.line(f"{tmp_arr}[{last_idx}] = VADD({tmp_arr}[{last_idx}], VSRA({tmp_var}{last_idx-1}, {v.limb_len}));")
    
    f.line(f"{tmp_arr}[0] = VAND({tmp_arr}[0], {mask});")
    for i in range(1, last_idx):
        f.line(f"{tmp_arr}[{i}] = VAND({tmp_var}{i}, {mask});")
    f.line()

def fp_add_8x1(f, pp, v, strat):
    f.start("void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b)")

    f.declare_single_vars(var_name="t", count=pp.num_limbs)
    f.line(f"__m512i sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    if strat.mod_2p:
        f.line(f"const __m512i p_bot = VSETALL(2);")
        f.line(f"const __m512i p_top = VSETALL(2*{v.p_plus_1_top});")
        p_txt = "2p"
    else:
        f.line(f"const __m512i p_bot = VSETALL(1);")
        f.line(f"const __m512i p_top = VSETALL({v.p_plus_1_top});")
        p_txt = "p"
    f.line()

    f.comment("Add limbs")
    for i in range(pp.num_limbs):
        f.line(f"t{i} = VADD(a[{i}], b[{i}]);")
    f.line()

    f.comment(f"Subtract {p_txt}, using special shape of p")
    f.line(f"t0 = VADD(t0, p_bot);")
    f.line(f"t{pp.num_limbs-1} = VSUB(t{pp.num_limbs-1}, p_top);")
    f.line()

    carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.comment(f"If result is negative, we undo the subtraction by {p_txt}")
    f.line(f"sign_mask = VSRA(t{pp.num_limbs-1}, 63);")
    f.line(f"t0 = VSUB(t0, VAND(p_bot, sign_mask));")
    f.line(f"t{pp.num_limbs-1} = VADD(t{pp.num_limbs-1}, VAND(p_top, sign_mask));")
    f.line()

    carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.single_vars_to_array(var_name="t", arr_name="res", count=pp.num_limbs)

    f.end()

def fp_sub_8x1(f, pp, v, strat):
    f.start("void fp_sub_8x1(__m512i* res, __m512i* a, __m512i* b)")

    f.declare_single_vars(var_name="t", count=pp.num_limbs)
    f.line(f"__m512i sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    if strat.mod_2p:
        f.line(f"const __m512i p_bot = VSETALL(2);")
        f.line(f"const __m512i p_top = VSETALL(2*{v.p_plus_1_top});")
        p_txt = "2p"
    else:
        f.line(f"const __m512i p_bot = VSETALL(1);")
        f.line(f"const __m512i p_top = VSETALL({v.p_plus_1_top});")
        p_txt = "p"
    f.line()

    f.comment("Subtract limbs")
    for i in range(pp.num_limbs):
        f.line(f"t{i} = VSUB(a[{i}], b[{i}]);")
    f.line()

    carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.comment(f"If result is negative, we add {p_txt}")
    f.line(f"sign_mask = VSRA(t{pp.num_limbs-1}, 63);")
    f.line(f"t0 = VSUB(t0, VAND(p_bot, sign_mask));")
    f.line(f"t{pp.num_limbs-1} = VADD(t{pp.num_limbs-1}, VAND(p_top, sign_mask));")
    f.line()

    carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.single_vars_to_array(var_name="t", arr_name="res", count=pp.num_limbs)

    f.end()

def fp_neg_8x1(f, pp, v, strat):
    f.start("void fp_neg_8x1(__m512i* res, __m512i* a)")

    f.declare_single_vars(var_name="t", count=pp.num_limbs)
    f.line(f"__m512i sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line("const __m512i zero = VZERO();")
    if strat.mod_2p:
        f.line(f"const __m512i p_bot = VSETALL(2);")
        f.line(f"const __m512i p_top = VSETALL(2*{v.p_plus_1_top});")
        p_txt = "2p"
    else:
        f.line(f"const __m512i p_bot = VSETALL(1);")
        f.line(f"const __m512i p_top = VSETALL({v.p_plus_1_top});")
        p_txt = "p"
    f.line()

    f.comment("0 - a")
    for i in range(pp.num_limbs):
        f.line(f"t{i} = VSUB(zero, a[{i}]);")
    f.line()

    carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.comment(f"If result is negative, we add {p_txt}")
    f.line(f"sign_mask = VSRA(t{pp.num_limbs-1}, 63);")
    f.line(f"t0 = VSUB(t0, VAND(p_bot, sign_mask));")
    f.line(f"t{pp.num_limbs-1} = VADD(t{pp.num_limbs-1}, VAND(p_top, sign_mask));")
    f.line()

    carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.single_vars_to_array(var_name="t", arr_name="res", count=pp.num_limbs)

    f.end()

def fp_mixaddsub_8x1(f, pp, v, strat):
    f.start("void fp_mixaddsub_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")

    f.declare_single_vars(var_name="t", count=pp.num_limbs)
    f.line(f"__m512i sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    if strat.mod_2p:
        f.line(f"const __m512i p_bot = VSETALL(2);")
        f.line(f"const __m512i p_top = VSETALL(2*{v.p_plus_1_top});")
        p_txt = "2p"
    else:
        f.line(f"const __m512i p_bot = VSETALL(1);")
        f.line(f"const __m512i p_top = VSETALL({v.p_plus_1_top});")
        p_txt = "p"
    f.line("const __mmask8 sub_mask = ~add_mask;")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"t{i} = VSUBMASK(a[{i}], sub_mask, a[{i}], b[{i}]);")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"t{i} = VADDMASK(t{i}, add_mask, a[{i}], b[{i}]);")
    f.line()

    f.comment(f"Subtract {p_txt}, using special shape of p")
    f.line(f"t0 = VADDMASK(t0, add_mask, t0, p_bot);")
    f.line(f"t{pp.num_limbs-1} = VSUBMASK(t{pp.num_limbs-1}, add_mask, t{pp.num_limbs-1}, p_top);")
    f.line()

    carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.comment(f"If the result is negative, we add {p_txt}")
    f.line(f"sign_mask = VSRA(t{pp.num_limbs-1}, 63);")
    f.line(f"t0 = VSUB(t0, VAND(p_bot, sign_mask));")
    f.line(f"t{pp.num_limbs-1} = VADD(t{pp.num_limbs-1}, VAND(p_top, sign_mask));")
    f.line()

    carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.single_vars_to_array(var_name="t", arr_name="res", count=pp.num_limbs)

    f.end()

def fp_normalize_8x1(f, pp, v):
    f.comment("Reduces from [0, 2p) to [0, p)")
    f.start("void fp_normalize_8x1(__m512i* res, __m512i* a)")

    f.declare_single_vars(var_name="t", count=pp.num_limbs)
    f.line(f"__m512i sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_bot = VSETALL(1);")
    f.line(f"const __m512i p_top = VSETALL({v.p_plus_1_top});")
    f.line()

    f.comment("Subtract p, using the special shape of p")
    f.line(f"t0 = VADD(a[0], p_bot);")
    for i in range(1, pp.num_limbs-1):
        f.line(f"t{i} = a[{i}];")
    f.line(f"t{pp.num_limbs-1} = VSUB(a[{pp.num_limbs-1}], p_top);")

    carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.comment("If the result is negative, we add p")
    f.line(f"sign_mask = VSRA(t{pp.num_limbs-1}, 63);")
    f.line(f"t0 = VSUB(t0, VAND(p_bot, sign_mask));")
    f.line(f"t{pp.num_limbs-1} = VADD(t{pp.num_limbs-1}, VAND(p_top, sign_mask));")

    carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.single_vars_to_array(var_name="t", arr_name="res", count=pp.num_limbs)

    f.end()

def fp_sqr_8x1(f, pp, v):
    f.start("void fp_sqr_8x1(__m512i* res, __m512i* a)")
    f.line("fp_mul_8x1(res, a, a);")
    f.end()

def fp_tomont_8x1(f, pp, v):
    f.start("void fp_tomont_8x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i vR2[{v.num_limbs}];")
    f.line("fp_duplicate_8x1(vR2, R2);")
    f.line("fp_mul_8x1(res, a, vR2);")

    f.end()

def fp_frommont_8x1(f, pp, v):
    f.start("void fp_frommont_8x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i one[{v.num_limbs}];")
    f.line("fp_setzero_8x1(one);")
    f.line("one[0] = VSETALL(1);")
    f.line("fp_mul_8x1(res, a, one);")

    f.end()

def mp_add_8x1(f, pp, v, strat):
    f.comment("Single-length multiprecision addition, no correction, no carry propagation")
    f.start("void mp_add_8x1(__m512i* res, __m512i* a, __m512i* b)")

    f.declare_single_vars(var_name="t", count=pp.num_limbs)
    if strat.carry_prop_karat:
        f.line(f"const __m512i limb_mask = VSETALL(LIMB_MASK);")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"t{i} = VADD(a[{i}], b[{i}]);")
    f.line()

    if strat.carry_prop_karat:
        carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.single_vars_to_array(var_name="t", arr_name="res", count=pp.num_limbs)

    f.end()

#TODO when 2 * last_limb_len <= limb_len, we can save one limb in the unreduced product, before propagating carry
def get_num_limbs_prod(pp):
    return 2 * pp.num_limbs

def mp_dblsubx2_8x1(f, pp, v):
    f.comment("Double-length multiprecision subtraction res = res - a - b, no correction")
    f.start("void mp_dblsubx2_8x1(__m512i* res, __m512i* a, __m512i* b)")

    num_limbs_prod = get_num_limbs_prod(pp)
    f.declare_single_vars(var_name="t", count=num_limbs_prod)
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line()

    for i in range(num_limbs_prod):
        f.line(f"t{i} = VSUB(res[{i}], a[{i}]);")
    f.line()

    for i in range(num_limbs_prod):
        f.line(f"t{i} = VSUB(t{i}, b[{i}]);")
    f.line()

    carry_prop_8x1_var(f, pp, v, num_limbs_prod, "t", "limb_mask")

    f.single_vars_to_array(var_name="t", arr_name="res", count=num_limbs_prod)

    f.end()

def mp_subx2_8x1(f, pp, v, strat):
    f.comment("Double-length multiprecision subtraction res = res - a")
    f.start("void mp_subx2_8x1(__m512i* res, __m512i* a)")

    num_limbs_prod = get_num_limbs_prod(pp)
    f.declare_single_vars(var_name="t", count=num_limbs_prod)
    if strat.mul_nonneg == MulNonnegMode.cond_add:
        f.line("__m512i sign_mask;")
    if strat.mul_nonneg == MulNonnegMode.add_2p_shifted:
        f.line("const __m512i p_bot = VSETALL(2);")
        f.line(f"const __m512i p_top = VSETALL(2*{v.p_plus_1_top});")
    elif strat.mul_nonneg == MulNonnegMode.add_4p_shifted:
        f.line("const __m512i p_bot = VSETALL(4);")
        f.line(f"const __m512i p_top = VSETALL(4*{v.p_plus_1_top});")
    else:
        f.line("const __m512i p_bot = VSETALL(1);")
        f.line(f"const __m512i p_top = VSETALL({v.p_plus_1_top});")
    
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line()

    for i in range(num_limbs_prod):
        f.line(f"t{i} = VSUB(res[{i}], a[{i}]);")
    f.line()

    match strat.mul_nonneg:
        case MulNonnegMode.cond_add:
            f.comment("Conditionally add p*R if negative")
        case MulNonnegMode.add_p_shifted:
            f.comment("We can get result in [0, p*R) by always adding p * 2^(num_limbs * limb_len)")
        case MulNonnegMode.add_2p_shifted:
            f.comment("We can get result in [0, p*R) by always adding 2p * 2^(num_limbs * limb_len)")
        case MulNonnegMode.add_4p_shifted:
            f.comment("We can get result in [0, p*R) by always adding 4p * 2^(num_limbs * limb_len)")

    if strat.mul_nonneg != MulNonnegMode.cond_add:
        f.line(f"t{pp.num_limbs} = VSUB(t{pp.num_limbs}, p_bot);")
        f.line(f"t{num_limbs_prod-1} = VADD(t{num_limbs_prod-1}, p_top);")
        f.line()

    carry_prop_8x1_var(f, pp, v, num_limbs_prod, "t", "limb_mask")

    if strat.mul_nonneg == MulNonnegMode.cond_add:
        # if result is negative, add p*R. 
        # Keeps output in [0, p*R), ensuring that Montgomery red gives result in [0, 2p)
        f.line(f"sign_mask = VSRA(t{num_limbs_prod-1}, 63);")
        f.line(f"t{pp.num_limbs} = VSUB(t{pp.num_limbs}, VAND(p_bot, sign_mask));")
        f.line(f"t{num_limbs_prod-1} = VADD(t{num_limbs_prod-1}, VAND(p_top, sign_mask));")

        carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask", pp.num_limbs)

    f.single_vars_to_array(var_name="t", arr_name="res", count=num_limbs_prod)

    f.end()

def mp_mixaddsubx2_8x1(f, pp, v, strat):
    f.start("void mp_mixaddsubx2_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")

    num_limbs_prod = get_num_limbs_prod(pp)
    f.declare_single_vars(var_name="t", count=num_limbs_prod)
    if strat.mul_nonneg == MulNonnegMode.cond_add:
        f.line("__m512i sign_mask;")
    if strat.mul_nonneg == MulNonnegMode.add_2p_shifted:
        f.line("const __m512i p_bot = VSETALL(2);")
        f.line(f"const __m512i p_top = VSETALL(2*{v.p_plus_1_top});")
    elif strat.mul_nonneg == MulNonnegMode.add_4p_shifted:
        f.line("const __m512i p_bot = VSETALL(4);")
        f.line(f"const __m512i p_top = VSETALL(4*{v.p_plus_1_top});")
    else:
        f.line("const __m512i p_bot = VSETALL(1);")
        f.line(f"const __m512i p_top = VSETALL({v.p_plus_1_top});")
    
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __mmask8 sub_mask = ~add_mask;")
    f.line()

    for i in range(num_limbs_prod):
        f.line(f"t{i} = VSUBMASK(a[{i}], sub_mask, a[{i}], b[{i}]);")
    f.line()

    match strat.mul_nonneg:
        case MulNonnegMode.cond_add:
            f.comment("Conditionally add p*R if negative")
        case MulNonnegMode.add_p_shifted:
            f.comment("We can get result in [0, p*R) by always adding p * 2^(num_limbs * limb_len)")
        case MulNonnegMode.add_2p_shifted:
            f.comment("We can get result in [0, p*R) by always adding 2p * 2^(num_limbs * limb_len)")
        case MulNonnegMode.add_4p_shifted:
            f.comment("We can get result in [0, p*R) by always adding 4p * 2^(num_limbs * limb_len)")

    if strat.mul_nonneg != MulNonnegMode.cond_add:
        f.line(f"t{pp.num_limbs} = VSUB(t{pp.num_limbs}, p_bot);")
        f.line(f"t{num_limbs_prod-1} = VADD(t{num_limbs_prod-1}, p_top);")
        f.line()

    for i in range(num_limbs_prod):
        f.line(f"t{i} = VADDMASK(t{i}, add_mask, a[{i}], b[{i}]);")
    f.line()

    carry_prop_8x1_var(f, pp, v, num_limbs_prod, "t", "limb_mask")

    if strat.mul_nonneg == MulNonnegMode.cond_add:
        # if result is negative, add p*R. 
        # Keeps output in [0, p*R), ensuring that Montgomery red gives result in [0, 2p)
        f.line(f"sign_mask = VSRA(t{num_limbs_prod-1}, 63);")
        f.line(f"t{pp.num_limbs} = VSUB(t{pp.num_limbs}, VAND(p_bot, sign_mask));")
        f.line(f"t{num_limbs_prod-1} = VADD(t{num_limbs_prod-1}, VAND(p_top, sign_mask));")

        carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask", pp.num_limbs)

    f.single_vars_to_array(var_name="t", arr_name="res", count=num_limbs_prod)

    f.end()

def mp_add_p_sub_8x1(f, pp, v, strat):
    plus_txt = "2p" if strat.mod_2p else "p"
    f.comment(f"Computes res = a + {plus_txt} - b without any carry propagation or conditional masks")
    f.start("void mp_add_p_sub_8x1(__m512i* res, __m512i* a, __m512i* b)")

    f.declare_single_vars(var_name="t", count=pp.num_limbs)
    m = 2 if strat.mod_2p else 1
    if strat.sqr8_faster_add_p_sub:
        f.line(f"const __m512i p_bot = VSETALL({m}*{v.limb_mask});")
        f.line(f"const __m512i p_top = VSETALL({m}*{v.p_plus_1_top}-{m});")
    else:
        f.line(f"const __m512i p_bot = VSETALL({m});")
        f.line(f"const __m512i p_top = VSETALL({m}*{v.p_plus_1_top});")
        f.line(f"const __m512i limb_mask = VSETALL(LIMB_MASK);")
    f.line()

    if strat.sqr8_faster_add_p_sub: # This is faster than carry, can be pipelined
        for i in range(pp.num_limbs-1):
            f.line(f"t{i} = VSUB(VADD(a[{i}], p_bot), b[{i}]);")
        f.line(f"t{pp.num_limbs-1} = VSUB(VADD(a[{pp.num_limbs-1}], p_top), b[{pp.num_limbs-1}]);")
    else:
        f.line(f"t0 = VSUB(VSUB(a[0], p_bot), b[0]);")
        for i in range(1, pp.num_limbs-1):
            f.line(f"t{i} = VSUB(a[{i}], b[{i}]);")
        f.line(f"t{pp.num_limbs-1} = VSUB(VADD(a[{pp.num_limbs-1}], p_top), b[{pp.num_limbs-1}]);")
    f.line()

    if not strat.sqr8_faster_add_p_sub:
        carry_prop_8x1_var(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.single_vars_to_array(var_name="t", arr_name="res", count=pp.num_limbs)

    f.end()

# Generalized to handle when the number of limbs is odd
def mp_mul_8x1(f, pp, v):
    f.start("void mp_mul_8x1(__m512i* res, __m512i* a, __m512i* b)")

    limbs_prod = get_num_limbs_prod(pp)
    low_limbs = ceil(pp.num_limbs/2)
    hi_limbs = pp.num_limbs - low_limbs
    low_limbs_prod = 2*low_limbs
    hi_limbs_prod = 2*hi_limbs
    f.array_to_single_vars(var_name="a", arr_name="a", count=pp.num_limbs)
    f.array_to_single_vars(var_name="b", arr_name="b", count=pp.num_limbs)
    f.declare_single_vars(var_name="t", count=limbs_prod)
    f.declare_single_vars(var_name="ta", count=low_limbs)
    f.declare_single_vars(var_name="tb", count=low_limbs)
    f.declare_single_vars(var_name="m", count=low_limbs_prod)
    f.line(f"const __m512i zero = VZERO();")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line()

    f.comment(f"a_low * b_low")
    for k in range(low_limbs_prod-1):
        if k < low_limbs:
            i_min = 0
            i_max = k
        else:
            i_min = k - (low_limbs-1)
            i_max = low_limbs - 1
        src_lo = "zero" if k == 0 else f"t{k}"

        for i in range(i_min, i_max+1):
            j = k - i
            src_hi = "zero" if i == i_min else f"t{k+1}"

            f.line(f"t{k} = VMULADDL({src_lo}, a{i}, b{j});")
            f.line(f"t{k+1} = VMULADDH({src_hi}, a{i}, b{j});")
        f.line(f"t{k+1} = VSL(t{k+1}, {v.limb_align});")
        f.line()
    
    f.comment(f"a_hi * b_hi")
    for k in range(hi_limbs_prod-1):
        if k < hi_limbs:
            i_min = 0
            i_max = k
        else:
            i_min = k - (hi_limbs-1)
            i_max = hi_limbs - 1
        src_lo = "zero" if k == 0 else f"t{k+low_limbs_prod}"

        for i in range(i_min, i_max+1):
            j = k - i
            src_hi = "zero" if i == i_min else f"t{k+1+low_limbs_prod}"

            f.line(f"t{k+low_limbs_prod} = VMULADDL({src_lo}, a{i+low_limbs}, b{j+low_limbs});")
            f.line(f"t{k+1+low_limbs_prod} = VMULADDH({src_hi}, a{i+low_limbs}, b{j+low_limbs});")
        f.line(f"t{k+1+low_limbs_prod} = VSL(t{k+1+low_limbs_prod}, {v.limb_align});")
        f.line()

    f.comment("ta = a_low + a_hi")
    for i in range(hi_limbs):
        f.line(f"ta{i} = VADD(a{i}, a{i+low_limbs});")
    if low_limbs > hi_limbs:
        f.line(f"ta{hi_limbs} = a{hi_limbs};")
    f.line()

    f.comment("tb = b_low + b_hi")
    for i in range(hi_limbs):
        f.line(f"tb{i} = VADD(b{i}, b{i+low_limbs});")
    if low_limbs > hi_limbs:
        f.line(f"tb{hi_limbs} = b{hi_limbs};")
    f.line()

    f.comment("m = ta * tb")
    for k in range(low_limbs_prod-1):
        if k < low_limbs:
            i_min = 0
            i_max = k
        else:
            i_min = k - (low_limbs-1)
            i_max = low_limbs - 1
        src_lo = "zero" if k == 0 else f"m{k}"

        for i in range(i_min, i_max+1):
            j = k - i
            src_hi = "zero" if i == i_min else f"m{k+1}"

            f.line(f"m{k} = VMULADDL({src_lo}, ta{i}, tb{j});")
            f.line(f"m{k+1} = VMULADDH({src_hi}, ta{i}, tb{j});")
        f.line(f"m{k+1} = VSL(m{k+1}, {v.limb_align});")
        f.line()

    f.comment("m = (ta * tb) - (a_low * b_low) - (a_hi * b_hi)")
    for i in range(hi_limbs_prod):
        f.line(f"m{i} = VSUB(m{i}, VADD(t{i}, t{i+low_limbs_prod}));")
    for i in range(hi_limbs_prod, low_limbs_prod):
        f.line(f"m{i} = VSUB(m{i}, t{i});")
    f.line()

    f.comment("t += m * 2^(limbs_low)")
    for i in range(low_limbs_prod):
        f.line(f"t{i+low_limbs} = VADD(t{i+low_limbs}, m{i});")
    f.line()

    f.single_vars_to_array(var_name="t", arr_name="res", count=limbs_prod)

    f.end()

def rdc_8x1_word_by_word(f, pp, v, strat):
    f.start("void rdc_8x1(__m512i* res, __m512i* a)")

    plus1 = " + 1" if strat.R_ext else ""
    f.comment(f"Montgomery reduction with R = 2^n mod p, with n = ((num_limbs{plus1}) * limb_len)")
    f.comment("low(a) = (a_0, a_1, ... , a_num_limbs)")
    f.comment("m = -p^(-1) mod 2^n = p + 2 = c * 2^e + 1")
    f.comment("The output of this reduction is (a + (low(a) * m mod 2^n )*p) / 2^n")
    f.line()


    f.array_to_single_vars(var_name="a", arr_name="a", count=2*pp.num_limbs, const=False)
    if strat.R_ext:
        f.line(f"__m512i a{2*pp.num_limbs} = VZERO();")
    f.line("__mmask8 msb_mask;")
    f.line(f"__m512i hi;")
    if not strat.mod_2p:
        f.line("__m512i sign_mask;")
        f.line(f"const __m512i bot = VSETALL(1);")
    f.line(f"const __m512i top = VSETALL({v.p_plus_1_top});")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i zero = VZERO();")
    f.line(f"const __m512i one = VSETALL(1);")
    f.line()

    for i in range(pp.num_limbs-1):
        f.line(f"a{i+pp.num_limbs-1} = VMULADDL(a{i+pp.num_limbs-1}, a{i}, top);")
        f.line(f"hi = VMULADDH(zero, a{i}, top);")
        f.line(f"hi = VSL(hi, {v.limb_align});")
        f.line(f"a{i+pp.num_limbs} = VADD(a{i+pp.num_limbs}, hi);")

    for i in range(pp.num_limbs-1, pp.num_limbs_R):
        f.line(f"a{i+1} = VADD(a{i+1}, VSRA(a{i}, {v.limb_len}));")
        f.line(f"a{i} = VAND(a{i}, limb_mask);")
        f.line(f"a{i+pp.num_limbs-1} = VMULADDL(a{i+pp.num_limbs-1}, a{i}, top);")
        f.line(f"hi = VMULADDH(zero, a{i}, top);")
        f.line(f"hi = VSL(hi, {v.limb_align});")
        f.line(f"a{i+pp.num_limbs} = VADD(a{i+pp.num_limbs}, hi);")

    carry_prop_8x1_var(f, pp, v, pp.num_limbs_R, "a", "limb_mask", start_idx=pp.num_limbs)

    start = pp.num_limbs_R
    end = pp.num_limbs_R + pp.num_limbs - 1
    f.comment(f"After dividing by 2^n, the result is in a{start}, ..., a{end}")

    if not strat.mod_2p:
        f.comment(f"We fully reduce the result down to [0, p).")

        f.comment("Subtract p, using special shape of p")
        f.line(f"a{start} = VADD(a{start}, bot);")
        f.line(f"a{end} = VSUB(a{end}, top);")
        f.line()

        carry_prop_8x1_var(f, pp, v, pp.num_limbs, "a", "limb_mask", start_idx=start)

        f.comment("If result is negative, we undo the subtraction by p")
        f.line(f"sign_mask = VSRA(a{end}, 63);")
        f.line(f"a{start} = VSUB(a{start}, VAND(bot, sign_mask));")
        f.line(f"a{end} = VADD(a{end}, VAND(top, sign_mask));")
        f.line()

        carry_prop_8x1_var(f, pp, v, pp.num_limbs, "a", "limb_mask", start_idx=start)
    
    f.single_vars_to_array(var_name="a", arr_name="res", count=pp.num_limbs, start_idx = start)

    f.end()

# c == 1 and e = 0 mod limb_len can not happen at the same time, for primes > 52 bits.
# Mersenne primes need e prime.
def rdc_8x1_oneshot(f, pp, v, strat):

    f.start("void rdc_8x1(__m512i* res, __m512i* a)")

    plus1 = " + 1" if strat.R_ext else ""
    f.comment(f"Montgomery reduction with R = 2^n mod p, with n = ((num_limbs{plus1}) * limb_len)")
    f.comment("low(a) = (a_0, a_1, ... , a_num_limbs)")
    f.comment("m = -p^(-1) mod 2^n = p + 2 = c * 2^e + 1")
    f.comment("The output of this reduction is (a + (low(a) * m mod 2^n )*p) / 2^n")
    f.line()

    nl = pp.num_limbs
    if strat.R_ext:
        n_t = nl + 2
        result_start = 2
    else:
        n_t = nl + 1
        result_start = 1

    f.array_to_single_vars(var_name="a", arr_name="a", count=2*nl)
    f.declare_single_vars(var_name="t", count=n_t)
    f.line("__mmask8 msb_mask;")
    if strat.R_ext:
        f.line(f"__m512i s{nl-1}, s{nl}, hi;")
    else:
        f.line(f"__m512i s{nl-1}, hi;")
    if not strat.mod_2p:
        f.line("__m512i sign_mask;")
        f.line(f"const __m512i bot = VSETALL(1);")
    f.line(f"const __m512i top = VSETALL({v.p_plus_1_top});")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i zero = VZERO();")
    f.line(f"const __m512i one = VSETALL(1);")
    f.line()

    # Compute s = low(a) * m mod 2^n
    if strat.R_ext:
        f.comment(f"s = low(a) * m mod 2^n, n = ({nl}+1)*{pp.limb_len}")
        f.comment(f"s0 = a0, ..., s{nl-2} = a{nl-2}")
        f.line(f"s{nl-1} = VMULADDL(a{nl-1}, a0, top);")
        f.line(f"s{nl} = VMULADDH(zero, a0, top);")
        f.line(f"s{nl} = VADD(a{nl}, VSL(s{nl}, {v.limb_align}));")
        f.line(f"s{nl} = VMULADDL(s{nl}, a{1}, top);")
        f.line(f"s{nl} = VADD(s{nl}, VSRA(s{nl-1}, {v.limb_len}));")
        f.line(f"s{nl-1} = VAND(s{nl-1}, limb_mask);")
        f.line(f"s{nl} = VAND(s{nl}, limb_mask);")
    else:
        f.comment(f"s = low(a) * m mod 2^n, n = {nl}*{pp.limb_len}")
        f.comment(f"s0 = a0, ..., s{nl-2} = a{nl-2}")
        f.line(f"s{nl-1} = VAND(VMULADDL(a{nl-1}, a0, top), limb_mask);")
    f.line()

    f.comment("Computing a + s*p as a + s*(p+1) - s.")
    f.comment("As the division by 2^n cancels the lower limbs, we don't need to compute - s.")

    # s * top: first nl-1 limbs of s are a[0..nl-2]
    f.line(f"t0 = VMULADDL(zero, a0, top);")
    f.line(f"hi = VMULADDH(zero, a0, top);")
    f.line(f"hi = VSL(hi, {v.limb_align});")
    f.line()

    for i in range(1, nl - 1):
        f.line(f"t{i} = VMULADDL(hi, a{i}, top);")
        f.line(f"hi = VMULADDH(zero, a{i}, top);")
        f.line(f"hi = VSL(hi, {v.limb_align});")
        f.line()

    f.line(f"t{nl-1} = VMULADDL(hi, s{nl-1}, top);")
    f.line(f"hi = VMULADDH(zero, s{nl-1}, top);")
    f.line(f"hi = VSL(hi, {v.limb_align});")
    f.line()

    if strat.R_ext:
        f.line(f"t{nl} = VMULADDL(hi, s{nl}, top);")
        f.line(f"t{nl+1} = VMULADDH(zero, s{nl}, top);")
        f.line(f"t{nl+1} = VSL(t{nl+1}, {v.limb_align});")
    else:
        f.line(f"t{nl} = hi;")
    f.line()

    f.comment("t += a (ignoring lower limbs)")
    for i in range(nl + 1):
        f.line(f"t{i} = VADD(t{i}, a{i + nl - 1});")
    f.line()

    carry_prop_8x1_var(f, pp, v, n_t, "t", "limb_mask")

    f.comment(f"After dividing by 2^n, the result is in t{result_start}, ..., t{n_t-1}")

    if not strat.mod_2p:
        f.comment(f"We fully reduce the result down to [0, p).")

        f.comment("Subtract p, using special shape of p")
        f.line(f"t{result_start} = VADD(t{result_start}, bot);")
        f.line(f"t{n_t-1} = VSUB(t{n_t-1}, top);")
        f.line()

        carry_prop_8x1_var(f, pp, v, nl, "t", "limb_mask", start_idx=result_start)

        f.comment("If result is negative, we undo the subtraction by p")
        f.line(f"sign_mask = VSRA(t{n_t-1}, 63);")
        f.line(f"t{result_start} = VSUB(t{result_start}, VAND(bot, sign_mask));")
        f.line(f"t{n_t-1} = VADD(t{n_t - 1}, VAND(top, sign_mask));")
        f.line()

        carry_prop_8x1_var(f, pp, v, nl, "t", "limb_mask", start_idx=result_start)
    
    f.single_vars_to_array(var_name="t", arr_name="res", count=nl, start_idx = result_start)

    f.end()

def rdc_8x1(f, pp, v, strat):
    rdc_8x1_word_by_word(f, pp, v, strat)

def fp_mul_8x1(f, pp, v):
    f.start("void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b)")

    limbs_prod = get_num_limbs_prod(pp)
    f.line(f"__m512i t[{limbs_prod}];")
    f.declare_single_vars(var_name="t", count=limbs_prod-1, start_idx=1)
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line()

    f.line(f"mp_mul_8x1(t, a, b);")
    f.line()

    carry_prop_8x1_swap(f, pp, v, limbs_prod, "t", "t", "limb_mask")

    f.line(f"rdc_8x1(res, t);")

    f.end()

# Computing a/2 mod p:
# First assume we reduce mod p.
# If a is even, we compute a >> 1.
# If a is odd, a >> 1 = (a-1)/2. In this case, we add (p+1)/2.
# (a-1)/2 + (p+1)/2 = (a + p)/2 <= (p-2 + p)/2 = p-1 (using that p-1 is even)
# If we reduce mod 2p:
# (2p-1 + p)/2 = (3p-1)/2 < 4p/2 = 2p, 
# so the same algorithm still works.
def fp_div2_8x1(f, pp, v):
    f.start("void fp_div2_8x1(__m512i* res, __m512i* src)")

    f.array_to_single_vars(var_name="a", arr_name="src", count=pp.num_limbs)
    f.declare_single_vars(var_name="t", count=pp.num_limbs)
    f.line(f"const __m512i one = VSETALL(1);")
    add_top_only = (pp.e-1) // pp.limb_len == pp.num_limbs - 1
    if add_top_only:
        f.line(f"const __m512i cnst = VSETALL({v.p_plus_1_div2_top});")
    else:
        f.line(f"const __m512i cnst_lo = VSETALL({v.p_plus_1_div2_top});")
        f.line(f"const __m512i cnst_hi = VSETALL({v.p_plus_1_div2_top_hi});")

    f.line(f"__m512i odd_mask;")
    f.line()

    f.comment("odd_mask = all 1s if a is odd, else 0")
    f.line(f"odd_mask = VSUB(VZERO(), VAND(a0, one));")
    f.line()

    f.comment("a >> 1")
    for i in range(pp.num_limbs - 1):
        f.line(f"t{i} = VOR(VSRL(a{i}, 1), VSL(VAND(a{i+1}, one), {pp.limb_len - 1}));")
    f.line(f"t{pp.num_limbs-1} = VSRL(a{pp.num_limbs-1}, 1);")
    f.line()

    # Assumes c is small
    f.comment("Conditionally add (p+1)/2 = c * 2^(e-1)")    
    if add_top_only:
        f.line(f"t{pp.num_limbs-1} = VADD(t{pp.num_limbs-1}, VAND(cnst, odd_mask));")
    else:
        f.line(f"t{pp.num_limbs-2} = VADD(t{pp.num_limbs-2}, VAND(cnst_lo, odd_mask));")
        f.line(f"t{pp.num_limbs-1} = VADD(t{pp.num_limbs-1}, VAND(cnst_hi, odd_mask));")

        # Carry propagation
        f.line(f"t{pp.num_limbs-1}= VADD(t{pp.num_limbs-1}, VSRL(t{pp.num_limbs-2}, {v.limb_len}));")
        f.line(f"t{pp.num_limbs-2} = VAND(t{pp.num_limbs-2}, VSETALL({v.limb_mask}));")
    f.line()

    f.single_vars_to_array(var_name="t", arr_name="res", count=pp.num_limbs)
    f.end()

def fp_div3_8x1(f, pp, v):
    f.start("void fp_div3_8x1(__m512i* res, __m512i* src)")

    f.line(f"__m512i t[{pp.num_limbs}];")
    f.line(f"fp_duplicate_8x1(t, {v.inv3});")
    f.line("fp_mul_8x1(res, src, t);")

    f.end()

def fp_inv_8x1(f, pp, v):
    f.comment("Batched inversion invoking a single call to fp_inv_64.")
    f.start("void fp_inv_8x1(__m512i* res, __m512i* a)")

    f.line(f"uint64_t r_64[{2*pp.num_words64}];")
    f.line(f"__m512i u[{pp.num_limbs}], t0[{pp.n42}], t1[{pp.n42}], s0[{pp.n24}], s1[{pp.n24}], r[{pp.n24}];")
    f.line("__m512i idx0, idx1;")
    f.line()
    f.line(f"fp_8x1_to_4x2(t0, t1, a);")
    f.comment("t0 = (a0a1, a2a3, a4a5, a6a7)")
    f.line(f"fp_mul_4x2(t0, t0, t1);")
    f.line("fp_4x2_to_2x4(s0, s1, t0);")
    f.comment("s0 = (a0a1a2a3, a4a5a6a7)")
    f.line("fp_mul_2x4(s0, s0, s1);")
    f.line("idx1 = VSET(4, 5, 6, 7, 0, 0, 0, 0);")
    f.line(f"for (int i = 0; i < {pp.n24}; i++) {{")
    f.indent()
    f.line("s1[i] = VPERM512(idx1, s0[i]);")
    f.unindent()
    f.line("}")
    f.comment("r = a0a1a2a3a4a5a6a7 (wastes a mul)")
    f.line("fp_mul_2x4(r, s0, s1);")
    f.line()

    f.comment("Unvectorized inversion of r")
    f.line("fp_limbs_to_64_2x4(r_64, r);")
    f.line("fp_inv_64(r_64, r_64);")
    f.line("fp_64_to_limbs_2x4(r, r_64);")
    f.line()
    
    f.comment("s0 = (inv_0123, inv_4567)")
    f.line("idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);")
    f.line("idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);")
    f.line(f"for (int i = 0; i < {pp.n24}; i++) {{")
    f.indent()
    f.line("s0[i] = VPERM512(idx0, s0[i]);")
    f.line("s1[i] = VPERM512(idx1, r[i]);")
    f.unindent()
    f.line("}")
    f.line("fp_mul_2x4(s0, s0, s1);")

    f.comment("t0 = (inv01, inv23, inv45, inv67)")
    f.line("fp_2x4_to_4x2(t1, s0, s0); // (inv_0123, inv_0123, inv_4567, inv_4567)")
    f.line("idx0 = VSET(2, 3, 0, 1, 6, 7, 4, 5);")
    f.line(f"for (int i = 0; i < {pp.n42}; i++) {{")
    f.indent()
    f.line("t0[i] = VPERM512(idx0, t0[i]);")
    f.unindent()
    f.line("}")
    f.line("fp_mul_4x2(t0, t0, t1);")

    f.comment("res = (inv0, inv1, inv2, ...)")
    f.line("fp_4x2_to_8x1(u, t0, t0);")
    f.line("idx0 = VSET(1, 0, 3, 2, 5, 4, 7, 6);")
    f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
    f.indent()
    f.line("res[i] = VPERM512(idx0, a[i]);")
    f.unindent()
    f.line("}")
    f.line("fp_mul_8x1(res, res, u);")

    f.end()

def fp_sqrt_8x1(f, pp, v):
    f.start("void fp_sqrt_8x1(__m512i* res, __m512i* a)")
    
    e_minus_2 = pp.e - 2
    f.comment(f"Candidate root is a^((p+1)/4), with (p+1)/4 = {pp.c}*2^{e_minus_2}")
    f.line(f"__m512i t[{pp.num_limbs}];")
    f.line()

    # Step 1: Compute a^c
    if pp.c > 1:
        write_dbl_and_add(f, pp, v, pp.c, fmt="8x1")       

    # Step 2: Compute ^(2^(e-2))
    if pp.c > 1:
        if e_minus_2 > 1: # current result in t
            f.line(f"for (int i = 0; i < {e_minus_2 - 1}; i++) {{")
            f.indent()
            f.line("fp_sqr_8x1(t, t);")
            f.unindent()
            f.line("}")
        
        if e_minus_2 > 0:
            f.line("fp_sqr_8x1(res, t);")
    else: # current result in a
        if e_minus_2 == 1:
            f.line("fp_sqr_8x1(res, a);")
        elif e_minus_2 > 1:
            f.line("fp_sqr_8x1(t, a);")
            if e_minus_2 > 2:
                f.line(f"for (int i = 0; i < {e_minus_2 - 2}; i++) {{")
                f.indent()
                f.line("fp_sqr_8x1(t, t);")
                f.unindent()
                f.line("}")
            f.line("fp_sqr_8x1(res, t);")
            
    f.end()

def fp_exp3div4_8x1(f, pp, v, strat, chain):
    fmt = "8x1"
    arr_size = pp.num_limbs
    if strat.gen_addchain:
        fun_name = "fp_exp3div4_8x1"
        write_exp_function(f, pp, v, chain, fun_name, fmt, arr_size)
    else:
        write_fp_exp3_div4_naive(f, pp, v, fmt, arr_size)

def fp_64_to_limbs_8x1(f, pp, v):
    f.comment(f"Assumes unvec is an array of 8 * {v.num_words64} 64-bit ints")
    f.start("void fp_64_to_limbs_8x1(__m512i* res, uint64_t* unvec)")

    f.line("uint64_t t[8];")
    f.line()

    n = pp.num_limbs
    leftover = 64
    i64 = 0
    for j in range(n):
        f.line("for (int i = 0; i < 8; i++) {")
        f.indent()
        plusi64 = "" if i64 == 0 else f" + {i64}"
        l = pp.last_limb_len if j == n-1 else pp.limb_len
        if leftover == 64:
            f.line(f"t[i] = unvec[{v.num_words64} * i{plusi64}] & {v.limb_mask};")
        elif leftover >= l:
           shift = 64 - leftover
           f.line(f"t[i] = (unvec[{v.num_words64} * i{plusi64}] >> {shift}) & {v.limb_mask};")
        else: #leftover < l
            rshift = 64 - leftover
            lshift = leftover
            rmask = (1 << leftover) - 1
            lmask = (1 << (l-leftover)) - 1
            f.line(f"t[i] = ((unvec[{v.num_words64} * i{plusi64}] >> {rshift}) & 0x{rmask:x}ULL) | ((unvec[{v.num_words64} * i + {i64+1}] & 0x{lmask:x}ULL) << {lshift});")
        
        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1
            
        f.unindent()
        f.line("}")
        f.line(f"res[{j}] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);")

    f.end()

def fp_64_to_limbs_8x1_only_even_idx(f, pp, v):
    f.comment(f"Assumes unvec is an array of 4 * {v.num_words64} 64-bit ints")
    f.start("void fp_64_to_limbs_8x1_only_even_idx(__m512i* res, uint64_t* unvec)")

    f.line("uint64_t t[4];")
    f.line()

    n = pp.num_limbs
    leftover = 64
    i64 = 0
    for j in range(n):
        f.line("for (int i = 0; i < 4; i++) {")
        f.indent()
        plusi64 = "" if i64 == 0 else f" + {i64}"
        l = pp.last_limb_len if j == n-1 else pp.limb_len
        if leftover == 64:
            f.line(f"t[i] = unvec[{v.num_words64} * i{plusi64}] & {v.limb_mask};")
        elif leftover >= l:
           shift = 64 - leftover
           f.line(f"t[i] = (unvec[{v.num_words64} * i{plusi64}] >> {shift}) & {v.limb_mask};")
        else: #leftover < l
            rshift = 64 - leftover
            lshift = leftover
            rmask = (1 << leftover) - 1
            lmask = (1 << (l-leftover)) - 1
            f.line(f"t[i] = ((unvec[{v.num_words64} * i{plusi64}] >> {rshift}) & 0x{rmask:x}ULL) | ((unvec[{v.num_words64} * i + {i64+1}] & 0x{lmask:x}ULL) << {lshift});")
        
        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1
            
        f.unindent()
        f.line("}")
        f.line(f"res[{j}] = VSET(t[0], 0, t[1], 0, t[2], 0, t[3], 0);")

    f.end()

def fp_limbs_to_64_8x1(f, pp, v, strat):
    f.start("void fp_limbs_to_64_8x1(uint64_t* res, __m512i* src)")

    f.line(f"uint64_t t[8];")
    if strat.mod_2p:
        f.line(f"__m512i s[{pp.num_limbs}];")
        f.line("fp_normalize_8x1(s, src);")
        src = "s"
    else:
        src = "src"
    f.line()

    n = pp.num_limbs
    i64 = 0
    leftover = 64
    for j in range(n):
        f.line(f"VSTORE(t, {src}[{j}]);")
        f.line("for (int i = 0; i < 8; i++) {")
        f.indent()

        plusi64 = "" if i64 == 0 else f" + {i64}"
        l = pp.last_limb_len if j == n-1 else pp.limb_len
        if leftover == 64:
            f.line(f"res[{v.num_words64} * i{plusi64}] = t[i] & {v.limb_mask};")
        elif leftover >= l:
            shift = 64 - leftover
            f.line(f"res[{v.num_words64} * i{plusi64}] |= (t[i] & {v.limb_mask}) << {shift};")
        else: #leftover < l
            lshift = 64 - leftover
            rshift = leftover
            lmask = (1 << leftover) - 1
            rmask = (1 << (l - leftover)) - 1
            f.line(f"res[{v.num_words64} * i{plusi64}] |= (t[i] & 0x{lmask:x}ULL) << {lshift};")
            f.line(f"res[{v.num_words64} * i + {i64 + 1}] = (t[i] >> {rshift}) & 0x{rmask:x}ULL;")

        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1

        f.unindent()
        f.line("}")

    f.end()

def fp_limbs_to_64_8x1_only_even_idx(f, pp, v, strat):
    f.start("void fp_limbs_to_64_8x1_only_even_idx(uint64_t* res, __m512i* src)")

    f.line(f"uint64_t t[8];")
    if strat.mod_2p:
        f.line(f"__m512i s[{pp.num_limbs}];")
        f.line("fp_normalize_8x1(s, src);")
        src = "s"
    else:
        src = "src"
    f.line()

    n = pp.num_limbs
    i64 = 0
    leftover = 64
    for j in range(n):
        f.line(f"VSTORE(t, {src}[{j}]);")
        f.line("for (int i = 0; i < 4; i++) {")
        f.indent()

        plusi64 = "" if i64 == 0 else f" + {i64}"
        l = pp.last_limb_len if j == n-1 else pp.limb_len
        if leftover == 64:
            f.line(f"res[{v.num_words64} * i{plusi64}] = t[2*i] & {v.limb_mask};")
        elif leftover >= l:
            shift = 64 - leftover
            f.line(f"res[{v.num_words64} * i{plusi64}] |= (t[2*i] & {v.limb_mask}) << {shift};")
        else: #leftover < l
            lshift = 64 - leftover
            rshift = leftover
            lmask = (1 << leftover) - 1
            rmask = (1 << (l - leftover)) - 1
            f.line(f"res[{v.num_words64} * i{plusi64}] |= (t[2*i] & 0x{lmask:x}ULL) << {lshift};")
            f.line(f"res[{v.num_words64} * i + {i64+1}] = (t[2*i] >> {rshift}) & 0x{rmask:x}ULL;")

        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1

        f.unindent()
        f.line("}")

    f.end()

def rand_fp_8x1(f, pp, v):
    f.start(f"void rand_fp_8x1(__m512i* res)")

    f.line(f"__m512i zero[{pp.num_limbs}];")
    f.line(f"__m512i mask = VSETALL({v.limb_mask});")
    f.line(f"fp_setzero_8x1(zero);")
    f.line()

    if pp.last_limb_len == pp.limb_len:
        f.line(f"for (int i = 0; i < {pp.num_limbs}; i++) {{")
        f.indent()
        f.line("res[i] = VAND(rand_512i(), mask);")
        f.unindent()
        f.line("}")
    else:
        f.line(f"for (int i = 0; i < {pp.num_limbs-1}; i++) {{")
        f.indent()
        f.line("res[i] = VAND(rand_512i(), mask);")
        f.unindent()
        f.line("}")

        f.comment("Last limb is shorter")
        f.line(f"mask = VSETALL({v.last_limb_mask});")
        f.line(f"res[{pp.num_limbs-1}] = VAND(rand_512i(), mask);")
        f.line()

    f.comment("Reduction")
    f.line("fp_add_8x1(res, res, zero);")
    f.end()

def write_fp_8x1(f, pp, v, strat, chain):
    f.banner("8x1")

    fp_setzero_8x1(f, pp, v)
    fp_set_binary_8x1(f, pp, v)
    fp_copy_8x1(f, pp, v)
    fp_duplicate_8x1(f, pp, v)
    fp_is_equal_8x1(f, pp, v, strat)
    fp_is_zero_8x1(f, pp, v, strat)
    fp_select_8x1(f, pp, v)

    fp_add_8x1(f, pp, v, strat)
    fp_sub_8x1(f, pp, v, strat)
    fp_neg_8x1(f, pp, v, strat)
    fp_mixaddsub_8x1(f, pp, v, strat)
    fp_normalize_8x1(f, pp, v)

    
    mp_add_8x1(f, pp, v, strat)
    mp_dblsubx2_8x1(f, pp, v)
    mp_subx2_8x1(f, pp, v, strat)
    mp_mixaddsubx2_8x1(f, pp, v, strat)
    mp_add_p_sub_8x1(f, pp, v, strat)

    mp_mul_8x1(f, pp, v)
    rdc_8x1(f, pp, v, strat)

    fp_mul_8x1(f, pp, v)
    fp_sqr_8x1(f, pp, v)
    fp_div2_8x1(f, pp, v)
    fp_div3_8x1(f, pp, v)
    fp_inv_8x1(f, pp, v)
    fp_sqrt_8x1(f, pp, v)
    fp_exp3div4_8x1(f, pp, v, strat, chain)

    fp_tomont_8x1(f, pp, v)
    fp_frommont_8x1(f, pp, v)

    fp_64_to_limbs_8x1(f, pp, v)
    fp_64_to_limbs_8x1_only_even_idx(f, pp, v)
    fp_limbs_to_64_8x1(f, pp, v, strat)
    fp_limbs_to_64_8x1_only_even_idx(f, pp, v, strat)
    
    rand_fp_8x1(f, pp, v)
