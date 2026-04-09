from gen_addchain import (run_addchain, generate_exp_function)

def fp_setzero_2x4(f, pp, v):
    f.start("void fp_setzero_2x4(__m512i* res)")

    for i in range(pp.n24):
        f.line(f"res[{i}] = VZERO();")

    f.end()

def fp_copy_2x4(f, pp, v):
    f.start("void fp_copy_2x4(__m512i* res, __m512i* a)")
    for i in range(pp.n24):
        f.line(f"res[{i}] = a[{i}];")
    f.end()

def fp_duplicate_2x4(f, pp, v):
    f.start("void fp_duplicate_2x4(__m512i* res, const uint64_t* a)")
    for i in range(pp.n24):
        l1 = f"a[{i+pp.n24}]" if i+pp.n24 < pp.num_limbs else "0"
        l2 = f"a[{i+2*pp.n24}]" if i+2*pp.n24 < pp.num_limbs else "0"
        l3 = f"a[{i+3*pp.n24}]" if i+3*pp.n24 < pp.num_limbs else "0"
        f.line(f"res[{i}] = VSET(a[{i}], {l1}, {l2}, {l3}, a[{i}], {l1}, {l2}, {l3});")

    f.end()

def fp_is_equal_2x4(f, pp, v):
    f.start("int fp_is_equal_2x4(__m512i* a, __m512i* b)")

    f.line(f"__m512i zero[{pp.n24}], norm_a[{pp.n24}], norm_b[{pp.n24}];")
    f.line(f"fp_setzero_2x4(zero);")
    f.line("")

    f.comment("Normalize both to [0, p-1] by adding zero")
    f.line("fp_add_2x4(norm_a, a, zero);")
    f.line("fp_add_2x4(norm_b, b, zero);")
    f.line()

    f.line("__mmask8 acc = 0xff;")
    for i in range(pp.n24):
        f.line(f"acc &= VCMPEQ(norm_a[{i}], norm_b[{i}]);")
    f.line()

    f.line("return (acc & (acc >> 1) & (acc >> 2) & (acc >> 3) & 0x1)")
    f.indent()
    f.line("| ((acc >> 3) & (acc >> 4) & (acc >> 5) & (acc >> 6) & 0x2);")
    f.unindent()

    f.end()

def fp_is_zero_2x4(f, pp, v):
    f.start("int fp_is_zero_2x4(__m512i* a)")

    f.line(f"__m512i zero[{pp.n24}], norm_a[{pp.n24}];")
    f.line(f"fp_setzero_2x4(zero);")
    f.line("")

    f.comment("Normalize a to [0, p-1] by adding zero")
    f.line("fp_add_2x4(norm_a, a, zero);")
    f.line()

    f.line("__mmask8 acc = 0xff;")
    for i in range(pp.n24):
        f.line(f"acc &= VCMPEQ(norm_a[{i}], zero[{i}]);")
    f.line()

    f.line("return (acc & (acc >> 1) & (acc >> 2) & (acc >> 3) & 0x1)")
    f.indent()
    f.line("| ((acc >> 3) & (acc >> 4) & (acc >> 5) & (acc >> 6) & 0x2);")
    f.unindent()

    f.end()

def fp_select_2x4(f, pp, v):
    f.start("void fp_select_2x4(__m512i* res, __m512i* a0, __m512i* a1, const __mmask8 s)")

    f.line(f"__m512i t[{pp.n24}];")
    f.line("__mmask8 bit0 = s & 0x1, bit1 = (s >> 1) & 0x1;")
    f.line("__mmask8 mask = bit0 | (bit0 << 1) | (bit0 << 2) | (bit0 << 3)")
    f.indent()
    f.line("| (bit1 << 4) | (bit1 << 5) | (bit1 << 6) | (bit1 << 7);")
    f.unindent()
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VBLEND(mask, a0[{i}], a1[{i}]);")
    f.line()

    for i in range(pp.n24):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def carry_prop_2x4(f, pp, v, tmp_name, mask_name, carry_name, idx_shift_name):
    f.comment("Carry propagation (3 full cross-lane ripples to cross all 4 lanes)")
    for i in range(3):
        if i == 0:
            f.comment("Straight ripple through the vectors")
        for j in range(pp.n24-1):
            f.line(f"{carry_name} = VSRA({tmp_name}[{j}], {v.limb_len});")
            f.line(f"{tmp_name}[{j}] = VAND({tmp_name}[{j}], {mask_name});")
            f.line(f"{tmp_name}[{j+1}] = VADD({tmp_name}[{j+1}], {carry_name});")
        if i == 0:
            f.comment("Cross-lane ripple from the last vector back to the first")
        f.line(f"{carry_name} = VSRA({tmp_name}[{pp.n24-1}], {v.limb_len});")
        f.line(f"{tmp_name}[{pp.n24-1}] = VANDMASK({tmp_name}[{pp.n24-1}], 0x77, {tmp_name}[{pp.n24 - 1}], {mask_name});")
        f.line(f"{carry_name} = VPERM512MASKZ(0xee, {carry_name}, {idx_shift_name});")
        f.line(f"{tmp_name}[0] = VADD({tmp_name}[0], {carry_name});")
    f.comment(f"Final straight ripple to finish the exact (4 * n24) limb sequence")
    for j in range(pp.n24-1):
        f.line(f"{carry_name} = VSRA({tmp_name}[{j}], {v.limb_len});")
        f.line(f"{tmp_name}[{j}] = VAND({tmp_name}[{j}], {mask_name});")
        f.line(f"{tmp_name}[{j+1}] = VADD({tmp_name}[{j+1}], {carry_name});")
    f.line(f"{tmp_name}[{pp.n24-1}] = VANDMASK({tmp_name}[{pp.n24-1}], 0x77, {tmp_name}[{pp.n24-1}], limb_mask);")
    f.line()

# TODO: Does not work when num_limbs % 4 != 0
def fp_add_2x4(f, pp, v):
    f.start("void fp_add_2x4(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{pp.n24}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);")
    f.line(f"const __m512i p_top = VSET(0, 0, 0, {v.p_plus_1_top}, 0, 0, 0, {v.p_plus_1_top});") # wrong if num_limbs % 4 != 0, need to use correct top idx
    f.comment("Maps odd lanes to the next even lane (1->2, 3->4, 5->6)")
    f.line("const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    f.comment("Broadcasts the sign bits from lanes 3 and 7 to their respective elements")
    f.line("const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VADD(a[{i}], b[{i}]);")
    f.line()

    f.comment("Subtract p using the special shape of p")
    f.line(f"t[0] = VADD(t[0], p_bot);")
    f.line(f"t[{pp.n24-1}] = VSUB(t[{pp.n24-1}], p_top);")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    f.comment("If the result is negative, we add p")
    f.line(f"sign_mask = VSRA(t[{pp.n24-1}], 63);")
    f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
    f.line(f"t[{pp.n24-1}] = VADD(t[{pp.n24-1}], VAND(p_top, sign_mask));")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    for i in range(pp.n24):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_sub_2x4(f, pp, v):
    f.start("void fp_sub_2x4(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{pp.n24}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);")
    f.line(f"const __m512i p_top = VSET(0, 0, 0, {v.p_plus_1_top}, 0, 0, 0, {v.p_plus_1_top});") # wrong if num_limbs % 4 != 0, need to use correct top idx
    f.comment("Maps odd lanes to the next even lane (1->2, 3->4, 5->6)")
    f.line("const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    f.comment("Broadcasts the sign bits from lanes 3 and 7 to their respective elements")
    f.line("const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VSUB(a[{i}], b[{i}]);")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    f.comment("If the result is negative, we add p")
    f.line(f"sign_mask = VSRA(t[{pp.n24-1}], 63);")
    f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
    f.line(f"t[{pp.n24-1}] = VADD(t[{pp.n24-1}], VAND(p_top, sign_mask));")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    for i in range(pp.n24):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_neg_2x4(f, pp, v):
    f.start("void fp_neg_2x4(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{pp.n24}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);")
    f.line(f"const __m512i p_top = VSET(0, 0, 0, {v.p_plus_1_top}, 0, 0, 0, {v.p_plus_1_top});") # wrong if num_limbs % 4 != 0, need to use correct top idx
    f.comment("Maps odd lanes to the next even lane (1->2, 3->4, 5->6)")
    f.line("const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    f.comment("Broadcasts the sign bits from lanes 3 and 7 to their respective elements")
    f.line("const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);")
    f.line("const __m512i zero = VZERO();")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VSUB(zero, a[{i}]);")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    f.comment("If the result is negative, we add p")
    f.line(f"sign_mask = VSRA(t[{pp.n24-1}], 63);")
    f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
    f.line(f"t[{pp.n24-1}] = VADD(t[{pp.n24-1}], VAND(p_top, sign_mask));")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    for i in range(pp.n24):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_mixaddsub_2x4(f, pp, v):
    f.start("void fp_mixaddsub_2x4(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")

    f.line(f"__m512i t[{pp.n24}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);")
    f.line(f"const __m512i p_top = VSET(0, 0, 0, {v.p_plus_1_top}, 0, 0, 0, {v.p_plus_1_top});") # wrong if num_limbs % 4 != 0, need to use correct top idx
    f.comment("Maps odd lanes to the next even lane (1->2, 3->4, 5->6)")
    f.line("const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    f.comment("Broadcasts the sign bits from lanes 3 and 7 to their respective elements")
    f.line("const __m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);")
    f.line("const __mmask8 sub_mask = ~add_mask;")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VSUBMASK(a[{i}], sub_mask, a[{i}], b[{i}]);")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VADDMASK(t[{i}], add_mask, a[{i}], b[{i}]);")

    f.comment("Subtract p using the special shape of p")
    f.line(f"t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);")
    f.line(f"t[{pp.n24-1}] = VSUBMASK(t[{pp.n24-1}], add_mask, t[{pp.n24-1}], p_top);")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    f.comment("If the result is negative, we add p")
    f.line(f"sign_mask = VSRA(t[{pp.n24-1}], 63);")
    f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
    f.line(f"t[{pp.n24-1}] = VADD(t[{pp.n24-1}], VAND(p_top, sign_mask));")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    for i in range(pp.n24):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_mul_2x4(f, pp, v):
    f.start("void mp_mul_2x4(__m512i* res, __m512i* a, __m512i* b)")

    n_prod = pp.n24 + pp.num_limbs
    f.line(f"__m512i t[{n_prod}];")
    f.line(f"__m512i B[{pp.num_limbs}];")
    f.line(f"const __m512i zero = VZERO();")
    f.line()
    
    f.comment("Create a vector for each limb of b")
    for j in range(pp.num_limbs):
        b_idx = j % pp.n24
        lane_idx = j // pp.n24
        if lane_idx == 0:
            mask = 0
        elif lane_idx == 1:
            mask = 0x55
        elif lane_idx == 2:
            mask = 0xaa
        else: #lane_idx == 3
            mask = 0xff

        f.line(f"B[{j}] = VPERM256(b[{b_idx}], 0x{mask:x});")
    f.line()

    f.comment("Schoolbook")
    for k in range(n_prod - 1):
        i_min = k - pp.num_limbs + 1 if k >= pp.num_limbs else 0
        i_max = k if k < pp.n24 else pp.n24 - 1
        src_lo = "zero" if k == 0 else f"t[{k}]"
        
        for i in range(i_min, i_max+1):
            j = k - i
            src_hi = "zero" if i == i_min else f"t[{k+1}]"

            f.line(f"t[{k}] = VMULADDL({src_lo}, a[{i}], B[{j}]);")
            f.line(f"t[{k+1}] = VMULADDH({src_hi}, a[{i}], B[{j}]);")
        f.line(f"t[{k+1}] = VSL(t[{k+1}], {v.limb_align});")
        f.line()

    for i in range(n_prod):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

#TODO: Reduce the size of t?
def rdc_2x4(f, pp, v):
    f.comment(f"Modfies a. Its length must be {2*pp.num_limbs+1}.")
    f.start("void rdc_2x4(__m512i* res, __m512i* a)")

    N = pp.num_limbs
    n24 = pp.n24
    top_offset = N - 1 - n24

    f.line("__m512i carry, lo, hi, shuf;")
    f.line("const __m512i zero = VZERO();")
    f.line("const __m512i one = VSETALL(1);")
    f.line("const __m512i cnst = VSETALL((uint64_t)65 << 40);")
    f.line("const __m512i limb_mask = VSETALL(LIMB_MASK);")
    f.line("__mmask8 msb_mask;")
    f.line()

    f.comment("Initialize the upper accumulators to zero")
    for i in range(N + n24, 2 * N + 1):
        f.line(f"a[{i}] = zero;")
    f.line()

    f.comment("Move each limb one step to the left, for each half")
    f.line("const __m512i idx = VSET(1, 2, 3, 0, 5, 6, 7, 0);")
    f.line()

    f.comment("Generalized Montgomery reduction loop")
    for i in range(N + 1):
        f.line(f"carry = VSRA(a[{i}], LIMB_LEN);")
        f.line(f"a[{i}] = VAND(a[{i}], limb_mask);")
        f.line(f"a[{i+1}] = VADD(a[{i+1}], carry);")
        
        f.line(f"shuf = VSHUFFLEMASKZ(0x0c0c, a[{i}], 0x4e);")
        f.line(f"a[{i + top_offset}] = VMULADDL(a[{i + top_offset}], shuf, cnst);")
        f.line(f"hi = VMULADDH(zero, shuf, cnst);")
        f.line(f"a[{i + top_offset + 1}] = VADD(a[{i + top_offset + 1}], VSL(hi, {v.limb_align}));")
        
        f.line(f"a[{i + n24}] = VADD(a[{i + n24}], VPERM512MASKZ(0x77, a[{i}], idx));")
        f.line()

    f.comment("Propagate carries")
    for i in range(N + 1, 2 * N):
        f.line(f"carry = VSRA(a[{i}], LIMB_LEN);")
        f.line(f"a[{i}] = VAND(a[{i}], limb_mask);")
        f.line(f"a[{i+1}] = VADD(a[{i+1}], carry);")
        
        # We don't fold the track on the very last propagation step
        if i != 2 * N - 1:
            f.line(f"a[{i + n24}] = VADD(a[{i + n24}], VPERM512MASKZ(0x77, a[{i}], idx));")
        f.line()

    f.comment("Final reduction step to get the result down to the strict bit boundary")
    f.line(f"hi = VSL(a[{2 * N}], 16);")
    f.line("msb_mask = VSIGN(hi);")
    f.line("__mmask8 mask_top = msb_mask & 0x11;")
    f.line()
    f.line(f"a[{2 * N}] = VSUBMASK(a[{2 * N}], mask_top, a[{2 * N}], cnst);")
    f.line(f"a[{N + 1}] = VADDMASK(a[{N + 1}], mask_top, a[{N + 1}], one);")
    f.line(f"a[{2 * N}] = VAND(a[{2 * N}], limb_mask);")
    f.line()

    f.comment("Pack result from pos 0 of the accumulators back into 2x4 format")
    f.line("const __m512i idx_pack = VSET(0, 1, 8, 9, 4, 5, 12, 13);")
    for k in range(n24):
        base = N + 1 + k
        f.line(f"lo = VUNPACKLO(a[{base}], a[{base + n24}]);")
        f.line(f"hi = VUNPACKLO(a[{base + 2 * n24}], a[{base + 3 * n24}]);")
        f.line(f"res[{k}] = VBLENDPERM(idx_pack, lo, hi);")
        if k < n24 - 1:
            f.line()
    f.end()


def fp_mul_2x4(f, pp, v):
    f.start("void fp_mul_2x4(__m512i* res, __m512i* a, __m512i* b)")
    f.line(f"__m512i t[{2*pp.num_limbs+1}];")
    f.line()

    f.line("mp_mul_2x4(t, a, b);")
    f.line("rdc_2x4(res, t);")

    f.end()

def fp_sqr_2x4(f, pp, v):
    f.start("void fp_sqr_2x4(__m512i* res, __m512i* a)")
    f.line(f"__m512i t[{2*pp.num_limbs+1}];")
    f.line()

    f.line("mp_mul_2x4(t, a, a);")
    f.line("rdc_2x4(res, t);")

    f.end()

def fp_div3_2x4(f, pp, v):
    f.start("void fp_div3_2x4(__m512i* res, __m512i* src)")

    f.line(f"__m512i t[{pp.n24}];")
    f.line(f"fp_duplicate_2x4(t, {v.inv3});")
    f.line("fp_mul_2x4(res, src, t);")

    f.end()

def fp_inv_2x4(f, pp, v):
    f.comment("Batched inversion.")
    f.start("void fp_inv_2x4(__m512i* res, __m512i* a)")

    f.line(f"uint64_t r_64[{2*pp.num_words64}];")
    f.line(f"__m512i s[{pp.n24}], r[{pp.n24}];")
    f.line("__m512i idx0, idx1;")
    f.line()

    f.comment("r = a0a1 (wastes one mul)")
    f.line("idx1 = VSET(4, 5, 6, 7, 0, 0, 0, 0);")
    f.line(f"for (int i = 0; i < {pp.n24}; i++) {{")
    f.indent()
    f.line("r[i] = VPERM512(idx1, a[i]);")
    f.unindent()
    f.line("}")
    f.line("fp_mul_2x4(r, r, a);")
    f.line()

    f.comment("Computing r^-1 mod p")
    f.line("fp_limbs_to_64_2x4(r_64, r);")
    f.line("fp_inv_64(r_64, r_64);")
    f.line("fp_64_to_limbs_2x4(r, r_64);")
    f.line()

    f.comment("res = (inv_0, inv_1)")
    f.line("idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);")
    f.line("idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);")
    f.line(f"for (int i = 0; i < {pp.n24}; i++) {{")
    f.indent()
    f.line("s[i] = VPERM512(idx0, a[i]);")
    f.line("r[i] = VPERM512(idx1, r[i]);")
    f.unindent()
    f.line("}")
    f.line("fp_mul_2x4(res, r, s);")
    f.line()

    f.end()

# Computes t = a^k (or res = a^k if copy_to_res=True)
def dbl_and_add(f, pp, v, k, copy_to_res=False):
    assert k > 1
    k_bits = bin(k)[2:]
    sqr_count = 0
    first_op = True
    
    # Iterate through the bits
    # The msb is always 1, and corresponds to multiplying the accumulator 1 with a.
    # Hence, we can skip it by setting t = a.
    bits_to_process = k_bits[1:]
    for idx, bit in enumerate(bits_to_process):
        is_last_bit = (idx == len(bits_to_process) - 1)
        sqr_count += 1
        
        if bit == '1':
            # Perform the accumulated squarings. 
            if first_op:
                f.line("fp_sqr_2x4(t, a);")
                if sqr_count > 1:
                    f.line(f"for (int i = 1; i < {sqr_count}; i++) {{")
                    f.indent()
                    f.line("fp_sqr_2x4(t, t);")
                    f.unindent()
                    f.line("}")
                first_op = False
            else:
                if sqr_count == 1:
                    f.line("fp_sqr_2x4(t, t);")
                else:
                    f.line(f"for (int i = 0; i < {sqr_count}; i++) {{")
                    f.indent()
                    f.line("fp_sqr_2x4(t, t);")
                    f.unindent()
                    f.line("}")
            
            # If this is the very last bit of the entire exponent, route to res
            dst = "res" if (copy_to_res and is_last_bit) else "t"
            f.line(f"fp_mul_2x4({dst}, t, a);")
            sqr_count = 0

    # Handle trailing zeros in k
    if sqr_count > 0:
        if sqr_count == 1:
            src = "a" if first_op else "t"
            dst = "res" if copy_to_res else "t"
            f.line(f"fp_sqrt_2x4({dst}, {src});")
        elif sqr_count > 1 and not first_op and not copy_to_res:
            f.line(f"for (int i = 0; i < {sqr_count}; i++) {{")
            f.indent()
            f.line("fp_sqr_2x4(t, t);")
            f.unindent()
            f.line("}")
        elif sqr_count > 1 and first_op and not copy_to_res:
            f.line("fp_sqr_2x4(t, a);")
            f.line(f"for (int i = 1; i < {sqr_count}; i++) {{")
            f.indent()
            f.line("fp_sqr_2x4(t, t);")
            f.unindent()
            f.line("}")
        elif sqr_count == 2 and copy_to_res:
            src = "a" if first_op else "t"
            f.line(f"fp_sqr_2x4(t, {src});")
            f.line(f"fp_sqr_2x4(res, t);")
        elif sqr_count == 3 and first_op and copy_to_res:
            f.line(f"fp_sqr_2x4(t, a);")
            f.line(f"fp_sqr_2x4(t, t);")
            f.line(f"fp_sqr_2x4(res, t);")
        elif sqr_count > 2 and not first_op and copy_to_res:
            f.line(f"for (int i = 0; i < {sqr_count-1}; i++) {{")
            f.indent()
            f.line("fp_sqr_2x4(t, t);")
            f.unindent()
            f.line("}")
            f.line("fp_sqr_2x4(res, t);")
        else: #sqr_count > 3 and first_op and copy_to_res:
            f.line("fp_sqr_2x4(t, a);")
            f.line(f"for (int i = 1; i < {sqr_count-1}; i++) {{")
            f.indent()
            f.line("fp_sqr_2x4(t, t);")
            f.unindent()
            f.line("}")
            f.line("fp_sqr_2x4(res, t);")

def fp_sqrt_2x4(f, pp, v):
    f.start("void fp_sqrt_2x4(__m512i* res, __m512i* a)")
    
    e_minus_2 = pp.e - 2
    f.comment(f"Candidate root is a^((p+1)/4), with (p+1)/4 = {pp.c}*2^{e_minus_2}")
    f.line(f"__m512i t[{pp.n24}];")
    f.line()

    # Step 1: Compute a^c
    if pp.c > 1:
        dbl_and_add(f, pp, v, pp.c)       

    # Step 2: Compute ^(2^(e-2))
    if pp.c > 1:
        if e_minus_2 > 1: # current result in t
            f.line(f"for (int i = 0; i < {e_minus_2 - 1}; i++) {{")
            f.indent()
            f.line("fp_sqr_2x4(t, t);")
            f.unindent()
            f.line("}")
        
        if e_minus_2 > 0:
            f.line("fp_sqr_2x4(res, t);")
    else: # current result in a
        if e_minus_2 == 1:
            f.line("fp_sqr_2x4(res, a);")
        elif e_minus_2 > 1:
            f.line("fp_sqr_2x4(t, a);")
            if e_minus_2 > 2:
                f.line(f"for (int i = 0; i < {e_minus_2 - 2}; i++) {{")
                f.indent()
                f.line("fp_sqr_2x4(t, t);")
                f.unindent()
                f.line("}")
            f.line("fp_sqr_2x4(res, t);")
            
    f.end()

def fp_exp3div4_2x4(f, pp, v, strat):
    if strat.gen_addchain:
        fun_name = "fp_exp3div4_2x4"
        fmt = "2x4"
        arr_size = pp.n24
        expr_str = f"{pp.c}*2^{pp.e-2} - 1"
        chain = run_addchain(expr_str)
        generate_exp_function(f, pp, v, chain, fun_name, fmt, arr_size)
    else:
        fp_exp3_div4_2x4_naive(f, pp, v)

def fp_exp3_div4_2x4_naive(f, pp, v):
    f.start("void fp_exp3div4_2x4(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{pp.n24}];")

    exponent = (pp.p - 3) // 4
    assert exponent == pp.c * 2**374 - 1
    dbl_and_add(f, pp, v, exponent, copy_to_res=True)

    f.end()

def fp_tomont_2x4(f, pp, v):
    f.start("void fp_tomont_2x4(__m512i* res, __m512i* a)")

    f.line(f"__m512i vR2[{pp.n24}];")
    f.line("fp_duplicate_2x4(vR2, R2);")
    f.line("fp_mul_2x4(res, a, vR2);")

    f.end()

def fp_frommont_2x4(f, pp, v):
    f.start("void fp_frommont_2x4(__m512i* res, __m512i* a)")

    f.line(f"__m512i one[{pp.n24}];")
    f.line(f"fp_setzero_2x4(one);")
    f.line(f"one[0] = VSET(1, 0, 0, 0, 1, 0, 0, 0);")
    f.line(f"fp_mul_2x4(res, a, one);")

    f.end()

#TODO: Assumes fully reduced, but after mul lowest limb can be one bit larger
#TODO: Also assumes all vectors full.
def fp_64_to_limbs_2x4(f, pp, v):
    f.comment(f"Assumes unvec is an array of 2 * {v.num_words64} 64-bit ints")
    f.start("void fp_64_to_limbs_2x4(__m512i* res, uint64_t* unvec)")

    f.line("uint64_t t[8];")
    f.line()

    n = pp.n24
    l = pp.limb_len
    leftover = 64
    i64 = 0
    lines = [[] for _ in range(n)]
    for j in range(3*n):
        plusi64 = "" if i64 == 0 else f" + {i64}"
        t_idx = "4*i" if j < n else f"4*i + {j // n}"
        if leftover == 64:
            line = f"t[{t_idx}] = unvec[{v.num_words64} * i{plusi64}] & {v.limb_mask};"
        elif leftover >= l:
            shift = 64 - leftover
            line = f"t[{t_idx}] = (unvec[{v.num_words64} * i{plusi64}] >> {shift}) & {v.limb_mask};"
        else: #leftover < l
            rshift = 64 - leftover
            lshift = leftover
            rmask = (1 << leftover) - 1
            lmask = (1 << (l-leftover)) - 1
            line = f"t[{t_idx}] = ((unvec[{v.num_words64} * i{plusi64}] >> {rshift}) & 0x{rmask:x}ULL) | ((unvec[{v.num_words64} * i + {i64+1}] & 0x{lmask:x}ULL) << {lshift});"
        lines[j % n].append(line)

        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1

    for j in range(n):
        f.line("for (int i = 0; i < 2; i++) {")
        f.indent()
        for line in lines[j]:
            f.line(line)

        plusi64 = "" if i64 == 0 else f" + {i64}"
        if leftover == 64:
            f.line(f"t[4*i+3] = unvec[{v.num_words64} * i{plusi64}] & {v.limb_mask};")
        elif leftover >= l or j == n-1:
           shift = 64 - leftover
           f.line(f"t[4*i+3] = (unvec[{v.num_words64} * i{plusi64}] >> {shift}) & {v.limb_mask};")
        else: #leftover < l and j < n-1
            rshift = 64 - leftover
            lshift = leftover
            rmask = (1 << leftover) - 1
            lmask = (1 << (l-leftover)) - 1
            f.line(f"t[4*i+3] = ((unvec[{v.num_words64} * i{plusi64}] >> {rshift}) & 0x{rmask:x}ULL) | ((unvec[{v.num_words64} * i + {i64+1}] & 0x{lmask:x}ULL) << {lshift});")
        
        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1
            
        f.unindent()
        f.line("}")
        f.line(f"res[{j}] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);")

    f.end()

def fp_limbs_to_64_2x4(f, pp, v):
    f.start("void fp_limbs_to_64_2x4(uint64_t* res, __m512i* src)")

    f.line("uint64_t t[8];")
    f.line()

    n = pp.n24
    l = pp.limb_len
    i64 = 0
    leftover = 64
    indices = [[] for _ in range(n)]
    lines_l = [[] for _ in range(n)]
    lines_r = [[] for _ in range(n)]

    for j in range(3*n):
        idx = j % n
        t_idx = "4*i" if j < n else f"4*i + {j // n}"
        plusi64 = "" if i64 == 0 else f" + {i64}"
        if leftover == 64:
            indices[idx].append(i64)
            lines_l[idx].append(f"res[{v.num_words64} * i{plusi64}]")
            lines_r[idx].append(f"t[{t_idx}] & {v.limb_mask};")
        elif leftover >= l:
            shift = 64 - leftover
            indices[idx].append(i64)
            lines_l[idx].append(f"res[{v.num_words64} * i{plusi64}]")
            lines_r[idx].append(f"(t[{t_idx}] & {v.limb_mask}) << {shift};")
        else: #leftover < l
            lshift = 64 - leftover
            rshift = leftover
            lmask = (1 << leftover) - 1
            rmask = (1 << (l - leftover)) - 1
            indices[idx].append(i64)
            lines_l[idx].append(f"res[{v.num_words64} * i{plusi64}]")
            lines_r[idx].append(f"(t[{t_idx}] & 0x{lmask:x}ULL) << {lshift};")
            indices[idx].append(i64+1)
            lines_l[idx].append(f"res[{v.num_words64} * i + {i64+1}]")
            lines_r[idx].append(f"(t[{t_idx}] >> {rshift}) & 0x{rmask:x}ULL;")

        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1

    seen = set()

    for j in range(n):
        f.line(f"VSTORE(t, src[{j}]);")
        f.line("for (int i = 0; i < 2; i++) {")
        f.indent()
        for k in range(len(indices[j])):
            idx = indices[j][k]
            line_l = lines_l[j][k]
            line_r = lines_r[j][k]
            op = "|=" if idx in seen else "="
            f.line(f"{line_l} {op} {line_r}")
            seen.add(idx)

        plusi64 = "" if i64 == 0 else f" + {i64}"
        op = "|=" if i64 in seen else "="
        seen.add(i64)

        if leftover == 64:
            f.line(f"res[{v.num_words64} * i{plusi64}] {op} t[4*i+3] & {v.limb_mask};")
        elif leftover >= l or j == n-1:
            shift = 64 - leftover
            f.line(f"res[{v.num_words64} * i{plusi64}] {op} (t[4*i+3] & {v.limb_mask}) << {shift};")
        else: #leftover < l and j < n-1
            lshift = 64 - leftover
            rshift = leftover
            lmask = (1 << leftover) - 1
            rmask = (1 << (l - leftover)) - 1
            op_next = "|=" if (i64+1) in seen else "="
            seen.add(i64+1)
            f.line(f"res[{v.num_words64} * i{plusi64}] {op} (t[4*i+3] & 0x{lmask:x}ULL) << {lshift};")
            f.line(f"res[{v.num_words64} * i + {i64+1}] {op_next} (t[4*i+3] >> {rshift}) & 0x{rmask:x}ULL;")

        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1

        f.unindent()
        f.line("}")

    f.end()

#TODO: Like most of the impl, this is wrong if the last limb is not full
def rand_fp_2x4(f, pp, v):
    f.start(f"void rand_fp_2x4(__m512i* res)")

    f.line(f"__m512i zero[{pp.n24}];")
    f.line(f"__m512i mask = VSETALL({v.limb_mask});")
    f.line(f"fp_setzero_2x4(zero);")
    f.line()


    f.line(f"for (int i = 0; i < {pp.n24 -1}; i++) {{")
    f.indent()
    f.line("res[i] = VAND(rand_512i(), mask);")
    f.unindent()
    f.line("}")
    f.line()

    f.comment("Last limb is shorter")
    f.line(f"mask = VSET({v.limb_mask}, {v.last_limb_mask}, {v.limb_mask}, {v.last_limb_mask}, {v.limb_mask}, {v.last_limb_mask}, {v.limb_mask}, {v.last_limb_mask});")
    f.line(f"res[{pp.n24 - 1}] = VAND(rand_512i(), mask);")
    f.line()

    f.comment("Reduction")
    f.line("fp_add_2x4(res, res, zero);")
    f.end()

def write_fp_2x4(f, pp, v, strat):
    f.banner("2x4")

    fp_setzero_2x4(f, pp, v)
    fp_copy_2x4(f, pp, v)
    fp_duplicate_2x4(f, pp, v)
    fp_is_equal_2x4(f, pp, v)
    fp_is_zero_2x4(f, pp, v)
    fp_select_2x4(f, pp, v)

    fp_add_2x4(f, pp, v)
    fp_sub_2x4(f, pp, v)
    fp_neg_2x4(f, pp, v)
    fp_mixaddsub_2x4(f, pp, v)

    mp_mul_2x4(f, pp, v)
    rdc_2x4(f, pp, v)
    fp_mul_2x4(f, pp, v)
    fp_sqr_2x4(f, pp, v)
    fp_tomont_2x4(f, pp, v)
    fp_frommont_2x4(f, pp, v)

    fp_div3_2x4(f, pp, v)
    fp_inv_2x4(f, pp, v)
    fp_sqrt_2x4(f, pp, v)
    fp_exp3div4_2x4(f, pp, v, strat)

    fp_64_to_limbs_2x4(f, pp, v)
    fp_limbs_to_64_2x4(f, pp, v)

    rand_fp_2x4(f, pp, v)