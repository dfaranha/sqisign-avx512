from math import ceil
from params import rdc_8x1_mode

def fp_setzero_8x1(f, pp, v):
    f.start("void fp_setzero_8x1(__m512i* res)")
    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = VZERO();")
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

def fp_is_equal_8x1(f, pp, v):
    f.start("int fp_is_equal_8x1(__m512i* a, __m512i* b)")

    f.line(f"__m512i zero[{v.num_limbs}], norm_a[{v.num_limbs}], norm_b[{v.num_limbs}];")
    f.line(f"__mmask8 acc;")
    f.line(f"fp_setzero_8x1(zero);")
    f.line()

    f.comment("Normalize both to [0, p-1] by adding zero")
    f.line("fp_add_8x1(norm_a, a, zero);")
    f.line("fp_add_8x1(norm_b, b, zero);")
    f.line()

    f.line("acc = 0xff;")
    for i in range(pp.num_limbs):
        f.line(f"acc &= VCMPEQ(norm_a[{i}], norm_b[{i}]);")
    f.line("return acc;")

    f.end()

def fp_is_zero_8x1(f, pp, v):
    f.start("int fp_is_zero_8x1(__m512i* a)")

    f.line(f"__m512i zero[{v.num_limbs}], norm_a[{v.num_limbs}];")
    f.line(f"__mmask8 acc;")
    f.line(f"fp_setzero_8x1(zero);")
    f.line()

    f.comment("Normalize a to [0, p-1] by adding zero")
    f.line("fp_add_8x1(norm_a, a, zero);")
    f.line()

    f.line("acc = 0xff;")
    for i in range(pp.num_limbs):
        f.line(f"acc &= VCMPEQ(norm_a[{i}], zero[{i}]);")
    f.line("return acc;")

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

def fp_add_8x1(f, pp, v):
    f.start("void fp_add_8x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{v.num_limbs}], sign_mask;")
    f.line(f"__m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"__m512i p_bot = VSETALL(1);")
    f.line(f"__m512i p_top = VSETALL({v.p_plus_1_top});")
    f.line()

    f.comment("Add limbs")
    for i in range(pp.num_limbs):
        f.line(f"t[{i}] = VADD(a[{i}], b[{i}]);")
    f.line()

    f.comment("Subtract p, using special shape of p")
    f.line(f"t[0] = VADD(t[0], p_bot);")
    f.line(f"t[{pp.num_limbs-1}] = VSUB(t[{pp.num_limbs-1}], p_top);")
    f.line()

    carry_prop_8x1(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.comment("If result is negative, we undo the subtraction by p")
    f.line(f"sign_mask = VSRA(t[{pp.num_limbs-1}], 63);")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line(f"t[{pp.num_limbs-1}] = VADD(t[{pp.num_limbs-1}], VAND(p_top, sign_mask));")
    f.line()

    carry_prop_8x1(f, pp, v, pp.num_limbs, "t", "limb_mask")

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_sub_8x1(f, pp, v):
    f.start("void fp_sub_8x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{v.num_limbs}], sign_mask;")
    f.line(f"__m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"__m512i p_bot = VSETALL(1);")
    f.line(f"__m512i p_top = VSETALL({v.p_plus_1_top});")
    f.line()

    f.comment("Subtract limbs")
    for i in range(pp.num_limbs):
        f.line(f"t[{i}] = VSUB(a[{i}], b[{i}]);")
    f.line()

    carry_prop_8x1(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.comment("If result is negative, we undo the subtraction by p")
    f.line(f"sign_mask = VSRA(t[{pp.num_limbs-1}], 63);")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line(f"t[{pp.num_limbs-1}] = VADD(t[{pp.num_limbs-1}], VAND(p_top, sign_mask));")
    f.line()

    carry_prop_8x1(f, pp, v, pp.num_limbs, "t", "limb_mask")

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_neg_8x1(f, pp, v):
    f.start("void fp_neg_8x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{v.num_limbs}], sign_mask;")
    f.line(f"__m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"__m512i p_bot = VSETALL(1);")
    f.line(f"__m512i p_top = VSETALL({v.p_plus_1_top});")
    f.line("__m512i zero = VZERO();")
    f.line()

    f.comment("0 - a")
    for i in range(pp.num_limbs):
        f.line(f"t[{i}] = VSUB(zero, a[{i}]);")
    f.line()

    carry_prop_8x1(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.comment("If result is negative, we undo the subtraction by p")
    f.line(f"sign_mask = VSRA(t[{pp.num_limbs-1}], 63);")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line(f"t[{pp.num_limbs-1}] = VADD(t[{pp.num_limbs-1}], VAND(p_top, sign_mask));")
    f.line()

    carry_prop_8x1(f, pp, v, pp.num_limbs, "t", "limb_mask")

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_mixaddsub_8x1(f, pp, v):
    f.start("void fp_mixaddsub_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")

    f.line(f"__m512i t[{v.num_limbs}], sign_mask;")
    f.line(f"__m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"__m512i p_bot = VSETALL(1);")
    f.line(f"__m512i p_top = VSETALL({v.p_plus_1_top});")
    f.line("const __mmask8 sub_mask = ~add_mask;")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"t[{i}] = VSUBMASK(a[{i}], sub_mask, a[{i}], b[{i}]);")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"t[{i}] = VADDMASK(t[{i}], add_mask, a[{i}], b[{i}]);")
    f.line()

    f.comment("Subtract p, using special shape of p")
    f.line(f"t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);")
    f.line(f"t[{pp.num_limbs-1}] = VSUBMASK(t[{pp.num_limbs-1}], add_mask, t[{pp.num_limbs-1}], p_top);")
    f.line()

    carry_prop_8x1(f, pp, v, pp.num_limbs, "t", "limb_mask")

    f.comment("If the result is negative, we add p")
    f.line(f"sign_mask = VSRA(t[{pp.num_limbs-1}], 63);")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line(f"t[{pp.num_limbs-1}] = VADD(t[{pp.num_limbs-1}], VAND(p_top, sign_mask));")
    f.line()

    carry_prop_8x1(f, pp, v, pp.num_limbs, "t", "limb_mask")

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = t[{i}];")

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

def mp_add_8x1(f, pp, v):
    f.comment("Single-length multiprecision addition, no correction, no carry propagation")
    f.start("void mp_add_8x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{v.num_limbs}];")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"t[{i}] = VADD(a[{i}], b[{i}]);")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_sub_8x1(f, pp, v):
    f.comment("Single-length multiprecision addition, no correction, no carry propagation")
    f.start("void mp_sub_8x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{v.num_limbs}];")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"t[{i}] = VSUB(a[{i}], b[{i}]);")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

#TODO when 2 * last_limb_len <= limb_len, we can save one limb in the unreduced product, before propagating carry
def get_num_limbs_prod(pp):
    return 2 * pp.num_limbs

def mp_dblsubx2_8x1(f, pp, v):
    f.comment("Double-length multiprecision subtraction res = res - a - b, no correction")
    f.start("void mp_dblsubx2_8x1(__m512i* res, __m512i* a, __m512i* b)")

    num_limbs_prod = get_num_limbs_prod(pp)
    f.line(f"__m512i t[2*{v.num_limbs}];")
    f.line(f"__m512i limb_mask = VSETALL({v.limb_mask});")
    f.line()

    for i in range(num_limbs_prod):
        f.line(f"t[{i}] = VSUB(res[{i}], a[{i}]);")
    f.line()

    for i in range(num_limbs_prod):
        f.line(f"t[{i}] = VSUB(t[{i}], b[{i}]);")
    f.line()

    carry_prop_8x1(f, pp, v, num_limbs_prod, "t", "limb_mask")

    for i in range(num_limbs_prod):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_subx2_8x1(f, pp, v):
    f.comment("Double-length multiprecision subtraction res = res - a")
    f.start("void mp_subx2_8x1(__m512i* res, __m512i* a)")

    num_limbs_prod = get_num_limbs_prod(pp)
    f.line(f"__m512i t[2*{v.num_limbs}];")
    f.line(f"__m512i limb_mask = VSETALL({v.limb_mask});")
    f.line()

    for i in range(num_limbs_prod):
        f.line(f"t[{i}] = VSUB(res[{i}], a[{i}]);")
    f.line()

    carry_prop_8x1(f, pp, v, num_limbs_prod, "t", "limb_mask")

    for i in range(num_limbs_prod):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_mixaddsubx2_8x1(f, pp, v):
    f.start("void mp_mixaddsubx2_8x1(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")

    num_limbs_prod = get_num_limbs_prod(pp)
    f.line(f"__m512i t[2*{v.num_limbs}];")
    f.line(f"__m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __mmask8 sub_mask = ~add_mask;")
    f.line()

    for i in range(num_limbs_prod):
        f.line(f"t[{i}] = VSUBMASK(a[{i}], sub_mask, a[{i}], b[{i}]);")
    f.line()

    for i in range(num_limbs_prod):
        f.line(f"t[{i}] = VADDMASK(t[{i}], add_mask, a[{i}], b[{i}]);")
    f.line()

    carry_prop_8x1(f, pp, v, num_limbs_prod, "t", "limb_mask")

    for i in range(num_limbs_prod):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_add_p_sub_8x1(f, pp, v):
    f.comment("Computes res = a + p - b without any carry propagation or conditional masks")
    f.start("void mp_add_p_sub_8x1(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{v.num_limbs}];")
    f.line(f"__m512i p_bot = VSETALL({v.limb_mask});")
    f.line(f"__m512i p_top = VSETALL({v.p_plus_1_top}-1);")
    f.line()

    f.comment("Add p")
    for i in range(pp.num_limbs-1):
        f.line(f"t[{i}] = VADD(a[{i}], p_bot);")
    f.line(f"t[{pp.num_limbs-1}] = VADD(a[{pp.num_limbs-1}], p_top);")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"t[{i}] = VSUB(t[{i}], b[{i}]);")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_dbl_8x1(f, pp, v):
    f.start("void mp_dbl_8x1(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{v.num_limbs}];")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"t[{i}] = VSL(a[{i}], 1);")
    f.line()

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

# Generalized to handle when the number of limbs is odd
def mp_mul_8x1_karat(f, pp, v):
    f.start("void mp_mul_8x1_karat(__m512i* res, __m512i* a, __m512i* b)")

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
        f.line(f"ta{hi_limbs} = a{hi_limbs}")
    f.line()

    f.comment("tb = b_low + b_hi")
    for i in range(hi_limbs):
        f.line(f"tb{i} = VADD(b{i}, b{i+low_limbs});")
    if low_limbs > hi_limbs:
        f.line(f"tb{hi_limbs} = b{hi_limbs}")
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

def rdc_8x1(f, pp, v, strat):
    f.start("void rdc_8x1(__m512i* res, __m512i* a)")

    plus1 = " + 1" if strat.fp2_lazy else ""
    f.comment(f"Montgomery reduction with R = 2^n mod p, with n = ((num_limbs{plus1}) * limb_len)")
    f.comment("low(a) = (a_0, a_1, ... , a_num_limbs)")
    f.comment("m = -p^(-1) mod 2^n = p + 2 = c * 2^e + 1")
    f.comment("The output of this reduction is (a + (low(a) * m mod 2^n )*p) / 2^n")
    f.line()

    # The logic is slightly different depending on whether:
    #   1. e % limb_len == 0
    #   2. c == 1
    #   3. c = 2^k +-1
    #
    # 1. and 2. can not happen simultaneously, at least not for cryptographically sized primes.
    # 2^e - 1 can only be prime when e itself is prime.
    #
    # Case 1 can only happen in certain cases.
    # Assume e = k * L
    # Let P = pbits, N = num_limbs and L = ceil(P/N),
    # By the definition of ceil, N * (L - 1) < P <= N * L.
    # Since c > 1, e < P, so k < N.
    # We need N * (L - 1) < P = k*L + c.nbits(), meaning c.nbits() > (N-k) * L - N.
    # But we also require that c is small, i.e. that c.nbits() < L.
    # (N-k) * L - N < L, implies that (N-k-1)*L < N.
    # If k < N-1, (N-k-1)*L > L, which implies L < N.
    # If k = N - 1, then c.nbits() > L - N.
    match strat.rdc_8x1:
        case rdc_8x1_mode.c1:  
            rdc_8x1_c1(f, pp, v, strat)
        case rdc_8x1_mode.shift0:
            rdc_8x1_shift0(f, pp, v, strat)
        case rdc_8x1_mode.ham2plus | rdc_8x1_mode.ham2minus:
            #TODO Finish
            #rdc_8x1_ham2(f, pp, v, strat)
            rdc_8x1_standard(f, pp, v, strat)
        case rdc_8x1_mode.shift0_ham2plus | rdc_8x1_mode.shift0_ham2minus:
            #rdc_8x1_shift0_ham2(f, pp, v, strat)
            rdc_8x1_standard(f, pp, v, strat)
        case _:
            rdc_8x1_standard(f, pp, v, strat)

    f.end()

#TODO Make changes done to standard to the other variants

# Assumes pp.e = 0 mod pp.limb_len
def rdc_8x1_shift0(f, pp, v, strat):
    f.line(f"__m512i t[{pp.num_limbs+2}], s{pp.num_limbs}, hi;")
    f.line("__mmask8 msb_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i top = VSETALL({v.p_plus_1_top});")
    f.line(f"const __m512i zero = VZERO();")
    f.line(f"const __m512i one = VSETALL(1);")
    f.line()

    f.comment("s = low(a) * m mod 2^n")
    f.comment(" = a0 * c * 2^(num_limbs * limb_len) + low(a) mod 2^n")
    f.comment(f"s0 = a0, ..., s{pp.num_limbs-1} = a{pp.num_limbs-1}")
    f.line(f"s{pp.num_limbs} = VAND(VMULADDL(a[{pp.num_limbs}], a[0], top), limb_mask);")
    f.line()

    f.comment("Computing a + s*p as a + s*(p+1) - s.")
    f.comment("As the division by 2^n cancels the lower limbs, we don't need to compute - s.")
    for i in range(pp.num_limbs):
        f.line(f"t[{i}] = VMULADDL(a[{pp.num_limbs+i}], a[{i}], top);")
        if i > 0:
            f.line(f"t[{i}] = VADD(t[{i}], hi);")
        f.line(f"hi = VMULADDH(zero, a[{i}], top);")
        f.line(f"hi = VSL(hi, LIMB_ALIGN);")
        f.line()

    f.line(f"t[{pp.num_limbs}] = VMULADDL(zero, s{pp.num_limbs}, top);")
    f.line(f"t[{pp.num_limbs}] = VADD(t[{pp.num_limbs}], hi);")
    f.line(f"hi = VMULADDH(zero, s{pp.num_limbs}, top);")
    f.line(f"t[{pp.num_limbs+1}] = VSL(hi, LIMB_ALIGN);")
    f.line()

    carry_prop_8x1(f, pp, v, pp.num_limbs+2, "t", "limb_mask")

    f.comment(f"After dividing by 2^n, the result is in t[2], ..., t[{pp.num_limbs+1}]")
    f.comment(f"The result can be {pp.pbits+1} bits.")
    f.comment(f"If so, we do a final subtraction by p to bring it down to {pp.pbits} bits.")

    shift = 64 - (pp.pbits + 1 - (pp.num_limbs - 1) * pp.limb_len)
    f.line(f"hi = VSL(t[{pp.num_limbs+1}], {shift});")
    f.line(f"msb_mask = VSIGN(hi);")
    f.line(f"t[{pp.num_limbs+1}] = VSUBMASK(t[{pp.num_limbs+1}], msb_mask, t[{pp.num_limbs+1}], top);")
    f.line(f"t[2] = VADDMASK(t[2], msb_mask, t[2], one);")
    f.line()

    if strat.rdc_final_sub_p:
        carry_prop_8x1(f, pp, v, pp.num_limbs, "t", "limb_mask", start_idx=2)
    else:
        f.comment(f"We do not propagate the carry here. The bottom limb can have {pp.limb_len+1} bits.")

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = t[{i+2}];")


# Assumes c = 1
def rdc_8x1_c1(f, pp, v, strat):

    f.line(f"__m512i t[{pp.num_limbs+2}], s{pp.num_limbs-1}, s{pp.num_limbs}, hi;")
    f.line("__mmask8 msb_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i top = VSETALL({v.p_plus_1_top});")
    f.line(f"const __m512i zero = VZERO();")
    f.line(f"const __m512i one = VSETALL(1);")
    f.line()

    f.comment("s = low(a) * m mod 2^n = (a0, a1) * m mod 2^n")
    f.comment(f"s0 = a0, ..., s{pp.num_limbs-2} = a{pp.num_limbs-2}")
    f.comment("(We simultaneously start on the next step)")
    lshift = pp.last_limb_len
    rshift = pp.limb_len - pp.last_limb_len

    f.line(f"t[0] = VADD(a[{pp.num_limbs-1}], VSL(a[0], {lshift}));")
    f.line(f"s{pp.num_limbs-1} = VAND(t[0], limb_mask);")
    f.line()

    f.line(f"t[1] = VADD(a[{pp.num_limbs}], VSL(a[1], {lshift}));")
    f.line(f"t[1] = VADD(t[1], VSRA(a[0], {rshift}));")
    f.line(f"s{pp.num_limbs} = VAND(t[1], limb_mask);")
    f.line()

    f.comment("Computing a + s*p as a + s*(p+1) - s.")
    f.comment("As the division by 2^n cancels the lower limbs, we don't need to compute - s.")

    for i in range(2, pp.num_limbs-1):
        f.line(f"t[{i}] = VADD(a[{i + pp.num_limbs-1}], VSL(a[{i}], {lshift}));")
        f.line(f"t[{i}] = VADD(t[{i}], VSRA(a[{i-1}], {rshift}));")
        f.line()

    f.line(f"t[{pp.num_limbs-1}] = VADD(a[{2*pp.num_limbs-2}], VSL(s{pp.num_limbs-1}, {lshift}));")
    f.line(f"t[{pp.num_limbs-1}] = VADD(t[{pp.num_limbs-1}], VSRA(a[{pp.num_limbs-2}], {rshift}));")
    f.line()

    f.line(f"t[{pp.num_limbs}] = VADD(a[{2*pp.num_limbs-1}], VSL(s{pp.num_limbs}, {lshift}));")
    f.line(f"t[{pp.num_limbs}] = VADD(t[{pp.num_limbs}], VSRA(s{pp.num_limbs-1}, {rshift}));")
    f.line()

    f.line(f"t[{pp.num_limbs+1}] = VSRA(s{pp.num_limbs}, {rshift});")

    carry_prop_8x1(f, pp, v, pp.num_limbs+2, "t", "limb_mask")

    f.comment(f"After dividing by 2^n, the result is in t[2], ..., t[{pp.num_limbs+1}]")
    f.comment(f"The result can be {pp.pbits+1} bits.")
    f.comment(f"If so, we do a final subtraction by p to bring it down to {pp.pbits} bits.")

    shift = 64 - (pp.pbits + 1 - (pp.num_limbs - 1) * pp.limb_len)
    f.line(f"hi = VSL(t[{pp.num_limbs+1}], {shift});")
    f.line(f"msb_mask = VSIGN(hi);")
    f.line(f"t[{pp.num_limbs+1}] = VSUBMASK(t[{pp.num_limbs+1}], msb_mask, t[{pp.num_limbs+1}], top);")
    f.line(f"t[2] = VADDMASK(t[2], msb_mask, t[2], one);")
    f.line()

    if strat.rdc_final_sub_p:
        carry_prop_8x1(f, pp, v, pp.num_limbs, "t", "limb_mask", start_idx=2)
    else:
        f.comment(f"We do not propagate the carry here. The bottom limb can have {pp.limb_len+1} bits.")

    for i in range(pp.num_limbs):
        f.line(f"res[{i}] = t[{i+2}];")

def rdc_8x1_ham2(f, pp, v, strat):
    """Optimized reduction for c = 2^k +- 1."""
    nl = pp.num_limbs
    L = pp.limb_len
    c = pp.c

    # Detect form
    k = None
    c_form = None
    for kk in range(1, 64):
        if c == (1 << kk) + 1:
            k = kk; c_form = 'plus'; break
        if c == (1 << kk) - 1:
            k = kk; c_form = 'minus'; break
    assert k is not None, f"c={c} is not 2^k ± 1"

    shift = pp.e % L  # bit offset within limb
    s1 = shift         # low shift
    s2 = shift + k     # high shift

    if strat.fp2_lazy:
        n_t = nl + 2
        result_start = 2
    else:
        n_t = nl + 1
        result_start = 1

    if strat.fp2_lazy:
        f.line(f"__m512i t[{n_t}], s{nl-1}, s{nl}, hi1, hi2;")
    else:
        f.line(f"__m512i t[{n_t}], s{nl-1}, hi1, hi2;")
    f.line("__mmask8 msb_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i top = VSETALL({v.p_plus_1_top});")
    f.line(f"const __m512i zero = VZERO();")
    f.line(f"const __m512i one = VSETALL(1);")
    f.line()

    # Helper: generate x * top = x << s2 ± x << s1
    # Returns (lo, hi) where lo is the low L bits and hi is the carry
    def emit_mul_top(dst_lo, dst_hi, src, accum_lo=None, accum_hi=None):
        """Generate code for dst = src * top, split into (lo, hi).
        If accum_lo/accum_hi given, add to those."""
        # x << s1: low part is (x << s1) & mask, high is x >> (L - s1)
        # x << s2: low part is (x << s2) & mask if s2 < L, else 0
        #          high is x >> (L - s2) if s2 < L, else x << (s2 - L)

        if s2 < L:
            # Both shifts produce bits in the low limb
            op = "VADD" if c_form == 'plus' else "VSUB"
            lo_expr = f"{op}(VSL({src}, {s2}), VSL({src}, {s1}))"
            f.line(f"__m512i {dst_lo}_raw = {lo_expr};")
            f.line(f"{dst_lo} = VAND({dst_lo}_raw, limb_mask);")

            hi_s1 = f"VSRA({src}, {L - s1})" if s1 > 0 else "zero"
            hi_s2 = f"VSRA({src}, {L - s2})"
            hi_expr = f"{op}({hi_s2}, {hi_s1})"
            # Also add the overflow from lo
            f.line(f"{dst_hi} = VADD({hi_expr}, VSRA({dst_lo}_raw, {L}));")
        elif s2 == L:
            # s2 shift puts everything in the high word
            f.line(f"{dst_lo} = VSL({src}, {s1});")
            hi_s1 = f"VSRA({src}, {L - s1})" if s1 > 0 else "zero"
            if c_form == 'plus':
                f.line(f"{dst_hi} = VADD({src}, {hi_s1});")
            else:
                f.line(f"{dst_hi} = VSUB({src}, {hi_s1});")
            f.line(f"__m512i {dst_lo}_overflow = VSRA({dst_lo}, {L});")
            f.line(f"{dst_lo} = VAND({dst_lo}, limb_mask);")
            f.line(f"{dst_hi} = VADD({dst_hi}, {dst_lo}_overflow);")
        else:
            # s2 > L: the s2 shift lands entirely in the high word
            s2_rem = s2 - L
            f.line(f"{dst_lo} = VSL({src}, {s1});")
            hi_s1 = f"VSRA({src}, {L - s1})" if s1 > 0 else "zero"
            if c_form == 'plus':
                f.line(f"{dst_hi} = VADD(VSL({src}, {s2_rem}), {hi_s1});")
            else:
                f.line(f"{dst_hi} = VSUB(VSL({src}, {s2_rem}), {hi_s1});")
            f.line(f"__m512i {dst_lo}_overflow = VSRA({dst_lo}, {L});")
            f.line(f"{dst_lo} = VAND({dst_lo}, limb_mask);")
            f.line(f"{dst_hi} = VADD({dst_hi}, {dst_lo}_overflow);")

    # --- Compute s ---
    # Use VMULADDL/VMULADDH for s since it's just 1-2 limb multiplies
    # and it keeps the code simpler
    if strat.fp2_lazy:
        f.comment(f"s = low(a) * ({c}*2^{pp.e} + 1) mod 2^(({nl}+1)*{L})")
        f.line(f"s{nl-1} = VMULADDL(a[{nl-1}], a[0], top);")
        f.line(f"s{nl} = VMULADDH(zero, a[0], top);")
        f.line(f"s{nl} = VADD(a[{nl}], VSL(s{nl}, {v.limb_align}));")
        f.line(f"s{nl} = VMULADDL(s{nl}, a[1], top);")
        f.line(f"s{nl} = VADD(s{nl}, VSRA(s{nl-1}, {L}));")
        f.line(f"s{nl-1} = VAND(s{nl-1}, limb_mask);")
        f.line(f"s{nl} = VAND(s{nl}, limb_mask);")
    else:
        f.comment(f"s = low(a) * ({c}*2^{pp.e} + 1) mod 2^({nl}*{L})")
        f.line(f"s{nl-1} = VAND(VMULADDL(a[{nl-1}], a[0], top), limb_mask);")
    f.line()

    # --- Compute s * (p+1) = s * c * 2^e using shifts ---
    f.comment(f"Compute s * (p+1) using shifts: c = 2^{k} {'+' if c_form == 'plus' else '-'} 1")
    op = "VADD" if c_form == 'plus' else "VSUB"

    # First limb: a[0]
    if s2 < L:
        f.line(f"t[0] = {op}(VSL(a[0], {s2}), VSL(a[0], {s1}));")
        if s1 > 0:
            f.line(f"hi1 = VSRA(a[0], {L - s1});")
        else:
            f.line(f"hi1 = zero;")
        f.line(f"hi2 = VSRA(a[0], {L - s2});")
        f.line(f"hi1 = VADD({op}(hi2, hi1), VSRA(t[0], {L}));")
        f.line(f"t[0] = VAND(t[0], limb_mask);")
    elif s2 == L:
        f.line(f"t[0] = VSL(a[0], {s1});")
        if s1 > 0:
            f.line(f"hi1 = {op}(a[0], VSRA(a[0], {L - s1}));")
        else:
            f.line(f"hi1 = {'VADD' if c_form == 'plus' else 'VSUB'}(a[0], zero);")
        f.line(f"hi1 = VADD(hi1, VSRA(t[0], {L}));")
        f.line(f"t[0] = VAND(t[0], limb_mask);")
    else:
        s2r = s2 - L
        f.line(f"t[0] = VSL(a[0], {s1});")
        if s1 > 0:
            f.line(f"hi1 = {op}(VSL(a[0], {s2r}), VSRA(a[0], {L - s1}));")
        else:
            f.line(f"hi1 = VSL(a[0], {s2r});")
        f.line(f"hi1 = VADD(hi1, VSRA(t[0], {L}));")
        f.line(f"t[0] = VAND(t[0], limb_mask);")
    f.line()

    # Middle limbs: a[1..nl-2]
    for i in range(1, nl - 1):
        src = f"a[{i}]"
        if s2 < L:
            f.line(f"t[{i}] = VADD(hi1, {op}(VSL({src}, {s2}), VSL({src}, {s1})));")
            if s1 > 0:
                f.line(f"hi1 = VSRA({src}, {L - s1});")
            else:
                f.line(f"hi1 = zero;")
            f.line(f"hi2 = VSRA({src}, {L - s2});")
            f.line(f"hi1 = VADD({op}(hi2, hi1), VSRA(t[{i}], {L}));")
        elif s2 == L:
            f.line(f"t[{i}] = VADD(hi1, VSL({src}, {s1}));")
            if s1 > 0:
                f.line(f"hi1 = {op}({src}, VSRA({src}, {L - s1}));")
            else:
                f.line(f"hi1 = {src};")
            f.line(f"hi1 = VADD(hi1, VSRA(t[{i}], {L}));")
        else:
            s2r = s2 - L
            f.line(f"t[{i}] = VADD(hi1, VSL({src}, {s1}));")
            if s1 > 0:
                f.line(f"hi1 = {op}(VSL({src}, {s2r}), VSRA({src}, {L - s1}));")
            else:
                f.line(f"hi1 = VSL({src}, {s2r});")
            f.line(f"hi1 = VADD(hi1, VSRA(t[{i}], {L}));")
        f.line(f"t[{i}] = VAND(t[{i}], limb_mask);")
        f.line()

    # s_{nl-1} limb
    src = f"s{nl-1}"
    if s2 < L:
        f.line(f"t[{nl-1}] = VADD(hi1, {op}(VSL({src}, {s2}), VSL({src}, {s1})));")
        if s1 > 0:
            f.line(f"hi1 = VSRA({src}, {L - s1});")
        else:
            f.line(f"hi1 = zero;")
        f.line(f"hi2 = VSRA({src}, {L - s2});")
        f.line(f"hi1 = VADD({op}(hi2, hi1), VSRA(t[{nl-1}], {L}));")
    elif s2 == L:
        f.line(f"t[{nl-1}] = VADD(hi1, VSL({src}, {s1}));")
        if s1 > 0:
            f.line(f"hi1 = {op}({src}, VSRA({src}, {L - s1}));")
        else:
            f.line(f"hi1 = {src};")
        f.line(f"hi1 = VADD(hi1, VSRA(t[{nl-1}], {L}));")
    else:
        s2r = s2 - L
        f.line(f"t[{nl-1}] = VADD(hi1, VSL({src}, {s1}));")
        if s1 > 0:
            f.line(f"hi1 = {op}(VSL({src}, {s2r}), VSRA({src}, {L - s1}));")
        else:
            f.line(f"hi1 = VSL({src}, {s2r});")
        f.line(f"hi1 = VADD(hi1, VSRA(t[{nl-1}], {L}));")
    f.line(f"t[{nl-1}] = VAND(t[{nl-1}], limb_mask);")
    f.line()

    # Top limb(s)
    if strat.fp2_lazy:
        src = f"s{nl}"
        if s2 < L:
            f.line(f"t[{nl}] = VADD(hi1, {op}(VSL({src}, {s2}), VSL({src}, {s1})));")
            if s1 > 0:
                f.line(f"hi1 = VSRA({src}, {L - s1});")
            else:
                f.line(f"hi1 = zero;")
            f.line(f"hi2 = VSRA({src}, {L - s2});")
            f.line(f"t[{nl+1}] = VADD({op}(hi2, hi1), VSRA(t[{nl}], {L}));")
        elif s2 == L:
            f.line(f"t[{nl}] = VADD(hi1, VSL({src}, {s1}));")
            if s1 > 0:
                f.line(f"t[{nl+1}] = {op}({src}, VSRA({src}, {L - s1}));")
            else:
                f.line(f"t[{nl+1}] = {src};")
            f.line(f"t[{nl+1}] = VADD(t[{nl+1}], VSRA(t[{nl}], {L}));")
        else:
            s2r = s2 - L
            f.line(f"t[{nl}] = VADD(hi1, VSL({src}, {s1}));")
            if s1 > 0:
                f.line(f"t[{nl+1}] = {op}(VSL({src}, {s2r}), VSRA({src}, {L - s1}));")
            else:
                f.line(f"t[{nl+1}] = VSL({src}, {s2r});")
            f.line(f"t[{nl+1}] = VADD(t[{nl+1}], VSRA(t[{nl}], {L}));")
        f.line(f"t[{nl}] = VAND(t[{nl}], limb_mask);")
    else:
        f.line(f"t[{nl}] = hi1;")
    f.line()

    # Add upper part of input
    f.comment("t += a (ignoring lower limbs)")
    for i in range(nl + 1):
        f.line(f"t[{i}] = VADD(t[{i}], a[{i + nl - 1}]);")
    f.line()

    carry_prop_8x1(f, pp, v, n_t, "t", "limb_mask")

    # Final conditional subtraction
    f.comment(f"After dividing by 2^n, the result is in t[{result_start}], ..., t[{n_t-1}]")
    f.comment(f"Conditionally subtract p")
    shift_final = 64 - (pp.pbits + 1 - (nl - 1) * L)
    f.line(f"hi1 = VSL(t[{n_t-1}], {shift_final});")
    f.line(f"msb_mask = VSIGN(hi1);")
    f.line(f"t[{n_t-1}] = VSUBMASK(t[{n_t-1}], msb_mask, t[{n_t-1}], top);")
    f.line(f"t[{result_start}] = VADDMASK(t[{result_start}], msb_mask, t[{result_start}], one);")
    f.line()

    if strat.rdc_final_sub_p:
        carry_prop_8x1(f, pp, v, nl, "t", "limb_mask", start_idx=result_start)
    else:
        f.comment(f"We do not propagate the carry here. The bottom limb can have {pp.limb_len+1} bits.")
    
    for i in range(nl):
        f.line(f"res[{i}] = t[{i + result_start}];")

def rdc_8x1_shift0_ham2(f, pp, v, strat):
    nl = pp.num_limbs
    L = pp.limb_len
    c = pp.c

    # Detect form
    k = None
    c_form = None
    for kk in range(1, 64):
        if c == (1 << kk) + 1:
            k = kk; c_form = 'plus'; break
        if c == (1 << kk) - 1:
            k = kk; c_form = 'minus'; break
    assert k is not None, f"c={c} is not 2^k ± 1"
    assert pp.e % L == 0, f"e={pp.e} is not a multiple of L={L}"
    assert k < L, f"k={k} must be less than L={L}"

    op = "VADD" if c_form == 'plus' else "VSUB"

    if strat.fp2_lazy:
        n_t = nl + 2
        result_start = 2
    else:
        n_t = nl + 1
        result_start = 1

    if strat.fp2_lazy:
        f.line(f"__m512i t[{n_t}], s{nl-1}, s{nl}, hi;")
    else:
        f.line(f"__m512i t[{n_t}], s{nl-1}, hi;")
    f.line("__mmask8 msb_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i top = VSETALL({v.p_plus_1_top});")
    f.line(f"const __m512i zero = VZERO();")
    f.line(f"const __m512i one = VSETALL(1);")
    f.line()

    # --- Compute s ---
    # Since e = (nl-1)*L, m = c*2^e + 1, and s = low(a)*m mod 2^n
    # s[0..nl-2] = a[0..nl-2], s[nl-1] = a[nl-1] + (a[0]*c) mod 2^L
    # For c = 2^k ± 1: a[0]*c = (a[0] << k) ± a[0]
    if strat.fp2_lazy:
        f.comment(f"s = low(a) * ({c}*2^{pp.e} + 1) mod 2^(({nl}+1)*{L})")
        f.line(f"s{nl-1} = {op}(VSL(a[0], {k}), a[0]);")
        f.line(f"s{nl-1} = VADD(a[{nl-1}], s{nl-1});")
        f.line(f"hi = VSRA(s{nl-1}, {L});")
        f.line(f"s{nl-1} = VAND(s{nl-1}, limb_mask);")
        # s[nl] = a[nl] + (a[1]*c + carry) mod 2^L
        f.line(f"s{nl} = {op}(VSL(a[1], {k}), a[1]);")
        f.line(f"s{nl} = VADD(s{nl}, VADD(a[{nl}], hi));")
        f.line(f"s{nl} = VAND(s{nl}, limb_mask);")
    else:
        f.comment(f"s = low(a) * ({c}*2^{pp.e} + 1) mod 2^({nl}*{L})")
        f.line(f"s{nl-1} = {op}(VSL(a[0], {k}), a[0]);")
        f.line(f"s{nl-1} = VAND(VADD(a[{nl-1}], s{nl-1}), limb_mask);")
    f.line()

    # --- Compute s * c * 2^e = s * c placed at limb nl-1 ---
    # Since e aligns and c = 2^k ± 1, for each limb x:
    #   x * c = (x << k) ± x
    #   lo = ((x << k) ± x) & LIMB_MASK
    #   hi = carry from lo + x >> (L-k)  [only from the shift term]

    f.comment(f"Compute s * (p+1) = s * {c} * 2^{pp.e}")
    f.comment(f"Since e aligns and c = 2^{k} {'+' if c_form == 'plus' else '-'} 1, use shifts")

    # First limb
    f.line(f"t[0] = {op}(VSL(a[0], {k}), a[0]);")
    f.line(f"hi = VSRA(a[0], {L - k});")
    f.line(f"hi = VADD(hi, VSRA(t[0], {L}));")
    f.line(f"t[0] = VAND(t[0], limb_mask);")
    f.line()

    # Middle limbs
    for i in range(1, nl - 1):
        f.line(f"t[{i}] = VADD(hi, {op}(VSL(a[{i}], {k}), a[{i}]));")
        f.line(f"hi = VSRA(a[{i}], {L - k});")
        f.line(f"hi = VADD(hi, VSRA(t[{i}], {L}));")
        f.line(f"t[{i}] = VAND(t[{i}], limb_mask);")
        f.line()

    # s_{nl-1} limb
    f.line(f"t[{nl-1}] = VADD(hi, {op}(VSL(s{nl-1}, {k}), s{nl-1}));")
    f.line(f"hi = VSRA(s{nl-1}, {L - k});")
    f.line(f"hi = VADD(hi, VSRA(t[{nl-1}], {L}));")
    f.line(f"t[{nl-1}] = VAND(t[{nl-1}], limb_mask);")
    f.line()

    if strat.fp2_lazy:
        f.line(f"t[{nl}] = VADD(hi, {op}(VSL(s{nl}, {k}), s{nl}));")
        f.line(f"t[{nl+1}] = VSRA(s{nl}, {L - k});")
        f.line(f"t[{nl+1}] = VADD(t[{nl+1}], VSRA(t[{nl}], {L}));")
        f.line(f"t[{nl}] = VAND(t[{nl}], limb_mask);")
    else:
        f.line(f"t[{nl}] = hi;")
    f.line()

    # Add upper part of input
    f.comment("t += a (ignoring lower limbs)")
    for i in range(nl + 1):
        f.line(f"t[{i}] = VADD(t[{i}], a[{i + nl - 1}]);")
    f.line()

    carry_prop_8x1(f, pp, v, n_t, "t", "limb_mask")

    # Final conditional subtraction
    f.comment(f"After dividing by 2^n, the result is in t[{result_start}], ..., t[{n_t-1}]")
    f.comment(f"Conditionally subtract p")
    shift_final = 64 - (pp.pbits + 1 - (nl - 1) * L)
    f.line(f"hi = VSL(t[{n_t-1}], {shift_final});")
    f.line(f"msb_mask = VSIGN(hi);")
    f.line(f"t[{n_t-1}] = VSUBMASK(t[{n_t-1}], msb_mask, t[{n_t-1}], top);")
    f.line(f"t[{result_start}] = VADDMASK(t[{result_start}], msb_mask, t[{result_start}], one);")
    f.line()

    if strat.rdc_final_sub_p:
        carry_prop_8x1(f, pp, v, nl, "t", "limb_mask", start_idx=result_start)
    else:
        f.comment(f"We do not propagate the carry here. The bottom limb can have {pp.limb_len+1} bits.")

    for i in range(nl):
        f.line(f"res[{i}] = t[{i + result_start}];")

def rdc_8x1_standard(f, pp, v, strat):
    nl = pp.num_limbs
    if strat.fp2_lazy:
        n_t = nl + 2
        result_start = 2
    else:
        n_t = nl + 1
        result_start = 1

    f.array_to_single_vars(var_name="a", arr_name="a", count=2*nl)
    f.declare_single_vars(var_name="t", count=n_t)
    if strat.fp2_lazy:
        f.line(f"__m512i s{nl-1}, s{nl}, hi;")
    else:
        f.line(f"__m512i s{nl-1}, hi;")
    f.line("__mmask8 msb_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i top = VSETALL({v.p_plus_1_top});")
    f.line(f"const __m512i zero = VZERO();")
    f.line(f"const __m512i one = VSETALL(1);")
    f.line()

    # Compute s = low(a) * m mod 2^n
    if strat.fp2_lazy:
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

    if strat.fp2_lazy:
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
    f.comment(f"The result can be {pp.pbits+1} bits.")

    if strat.rdc_final_sub_p:
        f.comment(f"If so, we do a final subtraction by p to bring it down to {pp.pbits} bits.")

        shift = 64 - (pp.pbits + 1 - (nl - 1) * pp.limb_len)
        f.line(f"hi = VSL(t{n_t-1}, {shift});")
        f.line(f"msb_mask = VSIGN(hi);")
        f.line(f"t{n_t-1} = VSUBMASK(t{n_t-1}, msb_mask, t{n_t-1}, top);")
        f.line(f"t{result_start} = VADDMASK(t{result_start}, msb_mask, t{result_start}, one);")
        f.line()

        carry_prop_8x1_var(f, pp, v, nl, "t", "limb_mask", start_idx=result_start)
    
    f.single_vars_to_array(var_name="t", arr_name="res", count=nl, start_idx = result_start)

def fp_mul_8x1(f, pp, v):
    f.start("void fp_mul_8x1(__m512i* res, __m512i* a, __m512i* b)")

    limbs_prod = get_num_limbs_prod(pp)
    f.line(f"__m512i t[{limbs_prod}];")
    f.declare_single_vars(var_name="t", count=limbs_prod-1, start_idx=1)
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line()

    f.line(f"mp_mul_8x1_karat(t, a, b);")
    f.line()

    carry_prop_8x1_swap(f, pp, v, limbs_prod, "t", "t", "limb_mask")

    f.line(f"rdc_8x1(res, t);")

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

def fp_64_to_limbs_8x1(f, pp, v):
    f.comment(f"Assumes unvec is an array of 8 * {v.num_words64} 64-bit ints")
    f.start("void fp_64_to_limbs_8x1(__m512i* res, uint64_t* unvec)")

    f.line("uint64_t t[8];")
    f.line()

    n = pp.num_limbs
    l = pp.limb_len
    leftover = 64
    i64 = 0
    for j in range(n):
        f.line("for (int i = 0; i < 8; i++) {")
        f.indent()
        plusi64 = "" if i64 == 0 else f" + {i64}"
        if leftover == 64:
            f.line(f"t[i] = unvec[{v.num_words64} * i{plusi64}] & {v.limb_mask};")
        elif leftover >= l or j == n-1:
           shift = 64 - leftover
           f.line(f"t[i] = (unvec[{v.num_words64} * i{plusi64}] >> {shift}) & {v.limb_mask};")
        else: #leftover < l and j < n-1
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
    l = pp.limb_len
    leftover = 64
    i64 = 0
    for j in range(n):
        f.line("for (int i = 0; i < 4; i++) {")
        f.indent()
        plusi64 = "" if i64 == 0 else f" + {i64}"
        if leftover == 64:
            f.line(f"t[i] = unvec[{v.num_words64} * i{plusi64}] & {v.limb_mask};")
        elif leftover >= l or j == n-1:
           shift = 64 - leftover
           f.line(f"t[i] = (unvec[{v.num_words64} * i{plusi64}] >> {shift}) & {v.limb_mask};")
        else: #leftover < l and j < n-1
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

#TODO: Assumes fully reduced, but after mul lowest limb can be one bit larger
def fp_limbs_to_64_8x1(f, pp, v):
    f.start("void fp_limbs_to_64_8x1(uint64_t* res, __m512i* src)")

    f.line("uint64_t t[8];")
    f.line()

    n = pp.num_limbs
    l = pp.limb_len
    i64 = 0
    leftover = 64
    for j in range(n):
        f.line(f"VSTORE(t, src[{j}]);")
        f.line("for (int i = 0; i < 8; i++) {")
        f.indent()

        plusi64 = "" if i64 == 0 else f" + {i64}"
        if leftover == 64:
            f.line(f"res[{v.num_words64} * i{plusi64}] = t[i] & {v.limb_mask};")
        elif leftover >= l or j == n - 1:
            shift = 64 - leftover
            f.line(f"res[{v.num_words64} * i{plusi64}] |= (t[i] & {v.limb_mask}) << {shift};")
        else: #leftover < l and j < n-1
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

#TODO: Assumes fully reduced, but after mul lowest limb can be one bit larger
def fp_limbs_to_64_8x1_only_even_idx(f, pp, v):
    f.start("void fp_limbs_to_64_8x1_only_even_idx(uint64_t* res, __m512i* src)")

    f.line("uint64_t t[8];")
    f.line()

    n = pp.num_limbs
    l = pp.limb_len
    i64 = 0
    leftover = 64
    for j in range(n):
        f.line(f"VSTORE(t, src[{j}]);")
        f.line("for (int i = 0; i < 4; i++) {")
        f.indent()

        plusi64 = "" if i64 == 0 else f" + {i64}"
        if leftover == 64:
            f.line(f"res[{v.num_words64} * i{plusi64}] = t[2*i] & {v.limb_mask};")
        elif leftover >= l or j == n - 1:
            shift = 64 - leftover
            f.line(f"res[{v.num_words64} * i{plusi64}] |= (t[2*i] & {v.limb_mask}) << {shift};")
        else: #leftover < l and j < n-1
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

def write_fp_8x1(f, pp, v, strat):
    f.banner("8x1")

    fp_setzero_8x1(f, pp, v)
    fp_copy_8x1(f, pp, v)
    fp_duplicate_8x1(f, pp, v)
    fp_is_equal_8x1(f, pp, v)
    fp_is_zero_8x1(f, pp, v)
    fp_add_8x1(f, pp, v)
    fp_sub_8x1(f, pp, v)
    fp_neg_8x1(f, pp, v)
    fp_mixaddsub_8x1(f, pp, v)

    mp_add_8x1(f, pp, v)
    mp_sub_8x1(f, pp, v)
    mp_dblsubx2_8x1(f, pp, v)
    mp_subx2_8x1(f, pp, v)
    mp_mixaddsubx2_8x1(f, pp, v)
    mp_add_p_sub_8x1(f, pp, v)
    mp_dbl_8x1(f, pp, v)
    mp_mul_8x1_karat(f, pp, v)
    rdc_8x1(f, pp, v, strat)

    fp_mul_8x1(f, pp, v)
    fp_sqr_8x1(f, pp, v)
    fp_div3_8x1(f, pp, v)
    fp_inv_8x1(f, pp, v)

    fp_tomont_8x1(f, pp, v)
    fp_frommont_8x1(f, pp, v)

    fp_64_to_limbs_8x1(f, pp, v)
    fp_64_to_limbs_8x1_only_even_idx(f, pp, v)
    fp_limbs_to_64_8x1(f, pp, v)
    fp_limbs_to_64_8x1_only_even_idx(f, pp, v)
    
    rand_fp_8x1(f, pp, v)
