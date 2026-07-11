from gen_addchain import (write_exp_function, write_dbl_and_add, write_fp_exp3_div4_naive)

def fp_setzero_2x4(f, pp, v):
    f.start("void fp_setzero_2x4(__m512i* res)")

    for i in range(pp.n24):
        f.line(f"res[{i}] = VZERO();")

    f.end()

def fp_set_binary_2x4(f, pp, v):
    f.comment("If bits[i] = 1, the i-th Fp element is set to 1 in the Montgomery representation.")
    f.start("void fp_set_binary_2x4(__m512i* res, __mmask8 bits)")

    f.line("uint64_t m0 = -(bits & 0x1);")
    f.line(f"uint64_t m1 = -((bits >> 1) & 0x1);")
    f.line()

    for i in range(pp.n24):
        l0 = f"R[{i}] & m0"
        l4 = f"R[{i}] & m1"
        l1 = f"R[{i + pp.n24}] & m0" if i + pp.n24 < pp.num_limbs else "0"
        l5 = f"R[{i + pp.n24}] & m1" if i + pp.n24 < pp.num_limbs else "0"
        l2 = f"R[{i + 2*pp.n24}] & m0" if i + 2*pp.n24 < pp.num_limbs else "0"
        l6 = f"R[{i + 2*pp.n24}] & m1" if i + 2*pp.n24 < pp.num_limbs else "0"
        l3 = f"R[{i + 3*pp.n24}] & m0" if i + 3*pp.n24 < pp.num_limbs else "0"
        l7 = f"R[{i + 3*pp.n24}] & m1" if i + 3*pp.n24 < pp.num_limbs else "0"
        f.line(f"res[{i}] = VSET({l0}, {l1}, {l2}, {l3}, {l4}, {l5}, {l6}, {l7});")
    
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

def fp_is_equal_2x4(f, pp, v, strat):
    f.start("__mmask8 fp_is_equal_2x4(__m512i* a, __m512i* b)")

    if strat.mod_2p:
        f.line(f"__m512i norm_a[{pp.n24}], norm_b[{pp.n24}];")
        f.line("fp_normalize_2x4(norm_a, a);")
        f.line("fp_normalize_2x4(norm_b, b);")
        f.line()
        var_a = "norm_a"
        var_b = "norm_b"
    else:
        var_a = "a"
        var_b = "b"

    f.line("__mmask8 acc = 0xff;")
    for i in range(pp.n24):
        valid = 0
        for lane in range(8):
            q = lane % 4
            limb_idx = i + q * pp.n24
            if limb_idx < pp.num_limbs:
                valid |= (1 << lane)

        if valid == 0xff:
            f.line(f"acc &= VCMPEQ({var_a}[{i}], {var_b}[{i}]);")
        else:
            ignore = 0xff ^ valid
            f.line(f"acc &= VCMPEQ({var_a}[{i}], {var_b}[{i}]) | 0x{ignore:02x};")
    f.line()

    f.line("return (acc & (acc >> 1) & (acc >> 2) & (acc >> 3) & 0x1)")
    f.indent()
    f.line("| ((acc >> 3) & (acc >> 4) & (acc >> 5) & (acc >> 6) & 0x2);")
    f.unindent()

    f.end()

def fp_is_zero_2x4(f, pp, v, strat):
    f.start("__mmask8 fp_is_zero_2x4(__m512i* a)")

    
    if strat.mod_2p:
        f.line(f"__m512i norm_a[{pp.n24}];")
    f.line("const __m512i zero = VZERO();")
    f.line()

    if strat.mod_2p:
        f.line("fp_normalize_2x4(norm_a, a);")
        f.line()
        var = "norm_a"
    else:
        var = "a"
    

    f.line("__mmask8 acc = 0xff;")
    for i in range(pp.n24):
        valid = 0
        for lane in range(8):
            q = lane % 4
            limb_idx = i + q * pp.n24
            if limb_idx < pp.num_limbs:
                valid |= (1 << lane)

        if valid == 0xff:
            f.line(f"acc &= VCMPEQ({var}[{i}], zero);")
        else:
            ignore = 0xff ^ valid
            f.line(f"acc &= VCMPEQ({var}[{i}], zero) | 0x{ignore:02x};")
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

def get_top_idx(pp):
    idx_top = (pp.num_limbs - 1) % pp.n24    # which vector holds the top limb
    limb_top = (pp.num_limbs - 1) // pp.n24   # idx within the vector
    return idx_top, limb_top

def write_p(f, v, mod_2p, limb_top):
    if mod_2p:
        f.line(f"const __m512i p_bot = VSET(2, 0, 0, 0, 2, 0, 0, 0);")
        top_val = f"2*{v.p_plus_1_top}"
    else:
        f.line(f"const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);")
        top_val = v.p_plus_1_top

    top_list = [(top_val if i % 4 == limb_top else "0")  for i in range(8)]
    top_str = ", ".join(top_list)
    f.line(f"const __m512i p_top = VSET({top_str});")

# Simple but unoptimized, doesn't account for where the top limb is.
# def carry_prop_2x4(f, pp, v, tmp_name, mask_name, carry_name, idx_shift_name):
#     f.comment("Carry propagation (3 full cross-lane ripples to cross all 4 lanes)")
#     for i in range(3):
#         if i == 0:
#             f.comment("Straight ripple through the vectors")
#         for j in range(pp.n24-1):
#             f.line(f"{carry_name} = VSRA({tmp_name}[{j}], {v.limb_len});")
#             f.line(f"{tmp_name}[{j}] = VAND({tmp_name}[{j}], {mask_name});")
#             f.line(f"{tmp_name}[{j+1}] = VADD({tmp_name}[{j+1}], {carry_name});")
#         if i == 0:
#             f.comment("Cross-lane ripple from the last vector back to the first")
#         f.line(f"{carry_name} = VSRA({tmp_name}[{pp.n24-1}], {v.limb_len});")
#         f.line(f"{tmp_name}[{pp.n24-1}] = VANDMASK({tmp_name}[{pp.n24-1}], 0x77, {tmp_name}[{pp.n24 - 1}], {mask_name});")
#         f.line(f"{carry_name} = VPERM512MASKZ(0xee, {carry_name}, {idx_shift_name});")
#         f.line(f"{tmp_name}[0] = VADD({tmp_name}[0], {carry_name});")
#     f.comment(f"Final straight ripple to finish the exact (4 * n24) limb sequence")
#     for j in range(pp.n24-1):
#         f.line(f"{carry_name} = VSRA({tmp_name}[{j}], {v.limb_len});")
#         f.line(f"{tmp_name}[{j}] = VAND({tmp_name}[{j}], {mask_name});")
#         f.line(f"{tmp_name}[{j+1}] = VADD({tmp_name}[{j+1}], {carry_name});")
#     f.line(f"{tmp_name}[{pp.n24-1}] = VANDMASK({tmp_name}[{pp.n24-1}], 0x77, {tmp_name}[{pp.n24-1}], limb_mask);")
#     f.line()

def carry_prop_2x4(f, pp, v, tmp_name, mask_name, carry_name, idx_shift_name):
    _, limb_top = get_top_idx(pp)
    n24 = pp.n24

    sign_preserve_mask = 0xff ^ ((1 << limb_top) | (1 << (limb_top + 4)))

    f.comment(f"Carry propagation ({limb_top} cross-lane ripples + final straight ripple)")
    for i in range(limb_top):
        for j in range(n24 - 1):
            f.line(f"{carry_name} = VSRA({tmp_name}[{j}], {v.limb_len});")
            f.line(f"{tmp_name}[{j}] = VAND({tmp_name}[{j}], {mask_name});")
            f.line(f"{tmp_name}[{j+1}] = VADD({tmp_name}[{j+1}], {carry_name});")
        f.line(f"{carry_name} = VSRA({tmp_name}[{n24-1}], {v.limb_len});")
        f.line(f"{tmp_name}[{n24-1}] = VANDMASK({tmp_name}[{n24-1}], 0x{sign_preserve_mask:02x}, {tmp_name}[{n24-1}], {mask_name});")
        f.line(f"{carry_name} = VPERM512MASKZ(0xee, {carry_name}, {idx_shift_name});")
        f.line(f"{tmp_name}[0] = VADD({tmp_name}[0], {carry_name});")

    f.comment("Final straight ripple")
    for j in range(n24 - 1):
        f.line(f"{carry_name} = VSRA({tmp_name}[{j}], {v.limb_len});")
        f.line(f"{tmp_name}[{j}] = VAND({tmp_name}[{j}], {mask_name});")
        f.line(f"{tmp_name}[{j+1}] = VADD({tmp_name}[{j+1}], {carry_name});")
    f.line(f"{tmp_name}[{n24-1}] = VANDMASK({tmp_name}[{n24-1}], 0x{sign_preserve_mask:02x}, {tmp_name}[{n24-1}], {mask_name});")
    f.line()

def fp_add_2x4(f, pp, v, strat):
    f.start("void fp_add_2x4(__m512i* res, __m512i* a, __m512i* b)")

    idx_top, limb_top = get_top_idx(pp)

    f.line(f"__m512i t[{pp.n24}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    write_p(f, v, strat.mod_2p, limb_top)
    p_txt = "2p" if strat.mod_2p else "p"
    f.comment("Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation")
    f.line(f"const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    f.comment("For duplicating sign bit in each half")
    f.line(f"const __m512i idx_sign = VSET({limb_top}, {limb_top}, {limb_top}, {limb_top}, {limb_top+4}, {limb_top+4}, {limb_top+4}, {limb_top+4});")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VADD(a[{i}], b[{i}]);")
    f.line()

    f.comment(f"Subtract {p_txt} using the special shape of p")
    f.line(f"t[0] = VADD(t[0], p_bot);")
    f.line(f"t[{idx_top}] = VSUB(t[{idx_top}], p_top);")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    f.comment(f"If the result is negative, add {p_txt} back")
    f.line(f"sign_mask = VSRA(t[{pp.n24-1}], 63);") # We carry the last ripple all the way to the end
    f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
    f.line(f"t[{idx_top}] = VADD(t[{idx_top}], VAND(p_top, sign_mask));")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    for i in range(pp.n24):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_sub_2x4(f, pp, v, strat):
    f.start("void fp_sub_2x4(__m512i* res, __m512i* a, __m512i* b)")

    idx_top, limb_top = get_top_idx(pp)

    f.line(f"__m512i t[{pp.n24}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    write_p(f, v, strat.mod_2p, limb_top)
    p_txt = "2p" if strat.mod_2p else "p"
    f.comment("Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation")
    f.line("const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    f.comment("For duplicating sign bit in each half")
    f.line(f"const __m512i idx_sign = VSET({limb_top}, {limb_top}, {limb_top}, {limb_top}, {limb_top+4}, {limb_top+4}, {limb_top+4}, {limb_top+4});")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VSUB(a[{i}], b[{i}]);")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    f.comment(f"If the result is negative, we add {p_txt}")
    f.line(f"sign_mask = VSRA(t[{pp.n24-1}], 63);")
    f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
    f.line(f"t[{idx_top}] = VADD(t[{idx_top}], VAND(p_top, sign_mask));")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    for i in range(pp.n24):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_neg_2x4(f, pp, v, strat):
    f.start("void fp_neg_2x4(__m512i* res, __m512i* a)")

    idx_top, limb_top = get_top_idx(pp)

    f.line(f"__m512i t[{pp.n24}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    write_p(f, v, strat.mod_2p, limb_top)
    p_txt = "2p" if strat.mod_2p else "p"
    f.comment("Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation")
    f.line("const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    f.comment("For duplicating sign bit in each half")
    f.line(f"const __m512i idx_sign = VSET({limb_top}, {limb_top}, {limb_top}, {limb_top}, {limb_top+4}, {limb_top+4}, {limb_top+4}, {limb_top+4});")
    f.line("const __m512i zero = VZERO();")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VSUB(zero, a[{i}]);")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    f.comment(f"If the result is negative, we add {p_txt}")
    f.line(f"sign_mask = VSRA(t[{pp.n24-1}], 63);")
    f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
    f.line(f"t[{idx_top}] = VADD(t[{idx_top}], VAND(p_top, sign_mask));")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    for i in range(pp.n24):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_mixaddsub_2x4(f, pp, v, strat):
    f.start("void fp_mixaddsub_2x4(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")

    idx_top, limb_top = get_top_idx(pp)

    f.line(f"__m512i t[{pp.n24}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    write_p(f, v, strat.mod_2p, limb_top)
    p_txt = "2p" if strat.mod_2p else "p"
    f.comment("Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation")
    f.line("const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    f.comment("For duplicating sign bit in each half")
    f.line(f"const __m512i idx_sign = VSET({limb_top}, {limb_top}, {limb_top}, {limb_top}, {limb_top+4}, {limb_top+4}, {limb_top+4}, {limb_top+4});")
    f.line("const __mmask8 sub_mask = ~add_mask;")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VSUBMASK(a[{i}], sub_mask, a[{i}], b[{i}]);")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VADDMASK(t[{i}], add_mask, a[{i}], b[{i}]);")

    f.comment(f"Subtract {p_txt} using the special shape of p")
    f.line(f"t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);")
    f.line(f"t[{idx_top}] = VSUBMASK(t[{idx_top}], add_mask, t[{idx_top}], p_top);")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    f.comment(f"If the result is negative, we add {p_txt}")
    f.line(f"sign_mask = VSRA(t[{pp.n24-1}], 63);")
    f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
    f.line(f"t[{idx_top}] = VADD(t[{idx_top}], VAND(p_top, sign_mask));")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    for i in range(pp.n24):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_normalize_2x4(f, pp, v):
    f.start("void fp_normalize_2x4(__m512i* res, __m512i* a)")

    idx_top, limb_top = get_top_idx(pp)

    f.line(f"__m512i t[{pp.n24}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")

    write_p(f, v, False, limb_top)

    f.comment("Maps odd lanes to the next even lane (1->2, 3->4, 5->6) for propagation")
    f.line(f"const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    f.comment("For duplicating sign bit in each half")
    f.line(f"const __m512i idx_sign = VSET({limb_top}, {limb_top}, {limb_top}, {limb_top}, {limb_top+4}, {limb_top+4}, {limb_top+4}, {limb_top+4});")
    f.line()

    f.comment("Subtract p using the special shape of p")
    f.line(f"t[0] = VADD(a[0], p_bot);")
    if idx_top == 0:
        f.line(f"t[0] = VSUB(t[0], p_top);")
        for i in range(1, pp.n24):
            f.line(f"t[{i}] = a[{i}];")
    else:
        for i in range(1, pp.n24):
            if i == idx_top:
                f.line(f"t[{i}] = VSUB(a[{i}], p_top);")
            else:
                f.line(f"t[{i}] = a[{i}];")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    f.comment("If the result is negative, add p back")
    f.line(f"sign_mask = VSRA(t[{pp.n24-1}], 63);") # We carry the last ripple all the way to the end
    f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
    f.line(f"t[{idx_top}] = VADD(t[{idx_top}], VAND(p_top, sign_mask));")
    f.line(f"t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    for i in range(pp.n24):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_mul_2x4(f, pp, v):
    f.start("void mp_mul_2x4(__m512i* res, __m512i* a, __m512i* b)")
    # uses accumulator format

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

def get_fold_mask(pp):
    """Compute the VPERM fold mask for rdc_2x4.
    The fold rotates (lane 0 <- 1, 1 <- 2, 2 <- 3, 3 <- 0).
    Lane 3 is always masked (wrap-around not needed).
    Lane 2 must also be masked if quarter 3 has no limbs,
    since it reads from lane 3 (garbage).
    """
    q3_limbs = max(0, pp.num_limbs - 3 * pp.n24)
    if q3_limbs == 0:
        return 0x33  # zero lanes 2,3,6,7
    else:
        return 0x77  # zero lanes 3,7 only

def get_rdc_2x4_size(pp):
    return pp.num_limbs_R + 4 * pp.n24

#Slightly faster, but so messy!
def rdc_2x4_old(f, pp, v, strat):
    f.comment(f"Modifies a. Its length must be at least {get_rdc_2x4_size(pp)}.")
    f.start("void rdc_2x4(__m512i* res, __m512i* a)")

    N = pp.num_limbs
    n24 = pp.n24
    top_offset = N - 1 - n24
    idx_top, limb_top = get_top_idx(pp)
    start_idx = pp.num_limbs_R

    f.line("__m512i carry, lo, hi, shuf;")
    f.line("const __m512i zero = VZERO();")
    f.line(f"const __m512i p_top_all = VSETALL({v.p_plus_1_top});")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    if not strat.mod_2p:
        f.line("__m512i sign_mask;")
        f.line(f"const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
        f.line(f"const __m512i idx_sign = VSET({limb_top}, {limb_top}, {limb_top}, {limb_top}, {limb_top+4}, {limb_top+4}, {limb_top+4}, {limb_top+4});")
    write_p(f, v, strat.mod_2p, limb_top)
    f.line()

    f.comment("Initialize the upper accumulators to zero")
    for i in range(N + n24, start_idx + 4 * n24):
        f.line(f"a[{i}] = zero;")
    f.line()

    fold_mask = get_fold_mask(pp)

    f.comment("Move each limb one step to the left, for each half")
    f.line("const __m512i idx = VSET(1, 2, 3, 0, 5, 6, 7, 0);")
    f.line()

    f.comment("Generalized Montgomery reduction loop")
    for i in range(start_idx):
        f.line(f"carry = VSRA(a[{i}], {v.limb_len});")
        f.line(f"a[{i}] = VAND(a[{i}], limb_mask);")
        f.line(f"a[{i+1}] = VADD(a[{i+1}], carry);")
        
        f.line(f"shuf = VSHUFFLEMASKZ(0x0c0c, a[{i}], 0x4e);")
        f.line(f"a[{i + top_offset}] = VMULADDL(a[{i + top_offset}], shuf, p_top_all);")
        f.line(f"hi = VMULADDH(zero, shuf, p_top_all);")
        f.line(f"a[{i + top_offset + 1}] = VADD(a[{i + top_offset + 1}], VSL(hi, {v.limb_align}));")
        
        f.line(f"a[{i + n24}] = VADD(a[{i + n24}], VPERM512MASKZ(0x{fold_mask:02x}, a[{i}], idx));")
        f.line()

    f.comment("Propagate carries")
    for i in range(start_idx, 2 * N):
        f.line(f"carry = VSRA(a[{i}], {v.limb_len});")
        f.line(f"a[{i}] = VAND(a[{i}], limb_mask);")
        f.line(f"a[{i+1}] = VADD(a[{i+1}], carry);")
        
        if i + n24 <= 2 * N:
            f.line(f"a[{i + n24}] = VADD(a[{i + n24}], VPERM512MASKZ(0x{fold_mask:02x}, a[{i}], idx));")
        f.line()

    f.comment("Pack result from pos 0 of the accumulators back into 2x4 format")
    f.line("const __m512i idx_pack = VSET(0, 1, 8, 9, 4, 5, 12, 13);")
    for k in range(n24):
        base = start_idx + k
        f.line(f"lo = VUNPACKLO(a[{base}], a[{base + n24}]);")
        f.line(f"hi = VUNPACKLO(a[{base + 2 * n24}], a[{base + 3 * n24}]);")
        f.line(f"res[{k}] = VBLENDPERM(idx_pack, lo, hi);")
        if k < n24 - 1:
            f.line()
    f.line()

    # Zero padding lanes to detect sign correctly.
    for k in range(n24):
        valid = 0
        for lane in range(8):
            q = lane % 4
            limb_idx = k + q * n24
            if limb_idx < N:
                valid |= (1 << lane)
        if valid != 0xff:
            f.line(f"res[{k}] = VBLEND(0x{valid:02x}, zero, res[{k}]);")
    f.line()

    if not strat.mod_2p:
        f.comment("Reduce from [0, 2p) to [0, p)")
        f.comment("Subtract p")
        f.line(f"res[0] = VADD(res[0], p_bot);")
        f.line(f"res[{idx_top}] = VSUB(res[{idx_top}], p_top);")
        f.line()

        carry_prop_2x4(f, pp, v, "res", "limb_mask", "carry", "idx_shift")

        f.comment("If negative, add p back")
        f.line(f"sign_mask = VSRA(res[{n24-1}], 63);")
        f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
        f.line(f"res[{idx_top}] = VADD(res[{idx_top}], VAND(p_top, sign_mask));")
        f.line(f"res[0] = VSUB(res[0], VAND(p_bot, sign_mask));")
        f.line()

        carry_prop_2x4(f, pp, v, "res", "limb_mask", "carry", "idx_shift")

    f.end()

# split into carry + fold stage and mult stage
def rdc_2x4_split(f, pp, v, strat):
    f.comment(f"Modifies a. Its length must be at least {pp.num_limbs_R + pp.n24}.")
    f.start("void rdc_2x4(__m512i* res, __m512i* a)")

    N = pp.num_limbs
    n24 = pp.n24
    idx_top, limb_top = get_top_idx(pp)
    start_idx = pp.num_limbs_R

    f.line("__m512i carry, lo, hi, shuf;")
    f.line("const __m512i zero = VZERO();")
    f.line(f"const __m512i p_top_all = VSETALL({v.p_plus_1_top});")
    f.line(f"const __m512i top_first = VSET({v.p_plus_1_top}, 0, 0, 0, {v.p_plus_1_top}, 0, 0, 0);")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    if not strat.mod_2p:
        f.line("__m512i sign_mask;")
        f.line(f"const __m512i idx_sign = VSET({limb_top}, {limb_top}, {limb_top}, {limb_top}, {limb_top+4}, {limb_top+4}, {limb_top+4}, {limb_top+4});")
    write_p(f, v, strat.mod_2p, limb_top)
    f.line()

    fold_mask = get_fold_mask(pp)

    f.comment("Move each limb one step to the left, for each half")
    f.line("const __m512i idx = VSET(1, 2, 3, 0, 5, 6, 7, 0);")
    f.line()

    if strat.R_ext:
        f.line(f"a[{N+n24}] = VZERO();")

    f.comment("Carry and fold")
    for i in range(start_idx):
        f.line(f"carry = VSRA(a[{i}], {v.limb_len});")
        f.line(f"a[{i}] = VAND(a[{i}], limb_mask);")
        f.line(f"a[{i+1}] = VADD(a[{i+1}], carry);")
        # fold:
        f.line(f"a[{i + n24}] = VADD(a[{i + n24}], VPERM512MASKZ(0x{fold_mask:02x}, a[{i}], idx));")
        f.line()

    f.comment("Aggregation in lane 0") #Only overlaps with first N limbs at i=0
    extra = 1 if strat.R_ext else 0
    for i in range(n24 + extra):
        target = i + N -1
        f.line(f"a[{target}] = VMULADDL(a[{target}], a[{i}], top_first);")
        f.line(f"hi = VMULADDH(zero, a[{i}], top_first);")
        f.line(f"a[{target + 1}] = VADD(a[{target + 1}], VSL(hi, {v.limb_align}));")

    f.comment("Carry propagation at affected later iterations")
    for i in range(N-1, pp.num_limbs_R):
        f.line(f"carry = VSRA(a[{i}], {v.limb_len});")
        f.line(f"a[{i}] = VAND(a[{i}], limb_mask);")
        f.line(f"a[{i+1}] = VADD(a[{i+1}], carry);")

    f.comment("Split aggregation")
    f.line(f"shuf = VSHUFFLEMASKZ(0x0c0c, a[{n24+extra}], 0x4e);")
    f.line(f"a[{n24+extra + N-1}] = VMULADDL(a[{n24+extra + N-1}], a[{n24+extra}], top_first);")
    f.line(f"hi = VMULADDH(zero, shuf, p_top_all);")
    f.line(f"a[{n24+extra + N-n24}] = VADD(a[{n24+extra + N-n24}], VSL(hi, {v.limb_align}));")

    if n24+extra +1 < pp.num_limbs_R:
        f.comment("Aggregation in second lane")
        max_second = min(2*n24+extra, pp.num_limbs_R)
        for i in range(n24 + extra + 1, max_second):
            target = i + N-1-n24
            f.line(f"shuf = VSHUFFLEMASKZ(0x0c0c, a[{i}], 0x4e);")
            f.line(f"a[{target}] = VMULADDL(a[{target}], shuf, p_top_all);")
            f.line(f"hi = VMULADDH(zero, shuf, p_top_all);")
            f.line(f"a[{target+1}] = VADD(a[{target+1}], VSL(hi, {v.limb_align}));")

    if 2*n24+extra < pp.num_limbs_R:
        f.comment("Split aggregation")
        f.line(f"lo = VSHUFFLEMASKZ(0x0c0c, a[{2*n24+extra}], 0x4e);") # second lane
        f.line(f"shuf = VPERM256MASKZ(0x44, a[{2*n24+extra}], 0x00);") # third lane
        f.line(f"a[{2*n24+extra + N-n24-1}] = VMULADDL(a[{2*n24+extra + N-n24-1}], lo, p_top_all);")
        f.line(f"hi = VMULADDH(zero, shuf, p_top_all);")
        f.line(f"a[{2*n24+extra + N-2*n24}] = VADD(a[{2*n24+extra + N-2*n24}], VSL(hi, {v.limb_align}));")

    if 2*n24+extra+1 < pp.num_limbs_R:
        max_third = min(3*n24+extra, pp.num_limbs_R) #N=5, n24=2, 3*n24 = 6, 2*n24+1 = 5
        f.comment("Aggregation in third lane")
        for i in range(2*n24+extra+1, max_third):
            target = i + N - 1 - 2*n24
            f.line(f"shuf = VPERM256MASKZ(0x44, a[{i}], 0x00);")
            f.line(f"a[{target}] = VMULADDL(a[{target}], shuf, p_top_all);")
            f.line(f"hi = VMULADDH(zero, shuf, p_top_all);")
            f.line(f"a[{target+1}] = VADD(a[{target+1}], VSL(hi, {v.limb_align}));")
    
    if 3*n24+extra < pp.num_limbs_R:
        f.comment("Split aggregation")
        f.line(f"lo = VPERM256MASKZ(0x44, a[{3*n24+extra}], 0x00);") # third lane
        f.line(f"shuf = VPERM256MASKZ(0x88, a[{3*n24+extra}], 0x00);") # fourth lane
        f.line(f"a[{3*n24+extra + N-1-2*n24}] = VMULADDL(a[{3*n24+extra + N-1-2*n24}], lo, p_top_all);")
        f.line(f"hi = VMULADDH(zero, shuf, p_top_all);")
        f.line(f"a[{3*n24+extra+ N-3*n24}] = VADD(a[{3*n24+extra + N-3*n24}], VSL(hi, {v.limb_align}));")
    
    if 3*n24+extra+1 < pp.num_limbs_R:
        f.comment("Aggregation in third lane")
        for i in range(3*n24+extra+1, pp.num_limbs_R):
            target = i + N - 1 - 3*n24
            f.line(f"shuf = VPERM256MASKZ(0x88, a[{i}], 0x00);")
            f.line(f"a[{target}] = VMULADDL(a[{target}], shuf, p_top_all);")
            f.line(f"hi = VMULADDH(zero, shuf, p_top_all);")
            f.line(f"a[{target+1}] = VADD(a[{target+1}], VSL(hi, {v.limb_align}));")

    for i in range(n24):
        f.line(f"res[{i}] = a[{i+pp.num_limbs_R}];")

    carry_prop_2x4(f, pp, v, "res", "limb_mask", "carry", "idx_shift")

    if not strat.mod_2p:
        f.comment("Reduce from [0, 2p) to [0, p)")
        f.comment("Subtract p")
        f.line(f"res[0] = VADD(res[0], p_bot);")
        f.line(f"res[{idx_top}] = VSUB(res[{idx_top}], p_top);")
        f.line()

        carry_prop_2x4(f, pp, v, "res", "limb_mask", "carry", "idx_shift")

        f.comment("If negative, add p back")
        f.line(f"sign_mask = VSRA(res[{n24-1}], 63);")
        f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
        f.line(f"res[{idx_top}] = VADD(res[{idx_top}], VAND(p_top, sign_mask));")
        f.line(f"res[0] = VSUB(res[0], VAND(p_bot, sign_mask));")
        f.line()

        carry_prop_2x4(f, pp, v, "res", "limb_mask", "carry", "idx_shift")

    f.end()

# WIP
# def rdc_2x4_fused(f, pp, v, strat):
#     f.comment(f"Modifies a. Its length must be at least {pp.num_limbs_R + pp.n24}.")
#     f.start("void rdc_2x4(__m512i* res, __m512i* a)")

#     N = pp.num_limbs
#     n24 = pp.n24
#     idx_top, limb_top = get_top_idx(pp)
#     start_idx = pp.num_limbs_R
#     top_offset = N - 1 - n24

#     f.line("__m512i carry, lo, hi, shuf;")
#     f.line("const __m512i zero = VZERO();")
#     f.line(f"const __m512i p_top_all = VSETALL({v.p_plus_1_top});")
#     f.line(f"const __m512i top_first = VSET({v.p_plus_1_top}, 0, 0, 0, {v.p_plus_1_top}, 0, 0, 0);")
#     f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
#     f.line(f"const __m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
#     if not strat.mod_2p:
#         f.line("__m512i sign_mask;")
#         f.line(f"const __m512i idx_sign = VSET({limb_top}, {limb_top}, {limb_top}, {limb_top}, {limb_top+4}, {limb_top+4}, {limb_top+4}, {limb_top+4});")
#     write_p(f, v, strat.mod_2p, limb_top)
#     f.line()

#     fold_mask = get_fold_mask(pp)

#     f.comment("Move each limb one step to the left, for each half")
#     f.line("const __m512i idx = VSET(1, 2, 3, 0, 5, 6, 7, 0);")
#     f.line()

#     if strat.R_ext:
#         f.line(f"a[{N+n24}] = VZERO();")

#     f.comment("Generalized Montgomery reduction loop")
#     for i in range(start_idx):
#         f.line(f"carry = VSRA(a[{i}], {v.limb_len});")
#         f.line(f"a[{i}] = VAND(a[{i}], limb_mask);")
#         f.line(f"a[{i+1}] = VADD(a[{i+1}], carry);")
        
#         f.line(f"shuf = VSHUFFLEMASKZ(0x0c0c, a[{i}], 0x4e);")
#         f.line(f"a[{i + top_offset}] = VMULADDL(a[{i + top_offset}], shuf, p_top_all);")
#         f.line(f"hi = VMULADDH(zero, shuf, p_top_all);")
#         f.line(f"a[{i + top_offset + 1}] = VADD(a[{i + top_offset + 1}], VSL(hi, {v.limb_align}));")
        
#         f.line(f"a[{i + n24}] = VADD(a[{i + n24}], VPERM512MASKZ(0x{fold_mask:02x}, a[{i}], idx));")
#         f.line()

#     for i in range(n24):
#         f.line(f"res[{i}] = a[{i+pp.num_limbs_R}];")

#     carry_prop_2x4(f, pp, v, "res", "limb_mask", "carry", "idx_shift")

#     if not strat.mod_2p:
#         f.comment("Reduce from [0, 2p) to [0, p)")
#         f.comment("Subtract p")
#         f.line(f"res[0] = VADD(res[0], p_bot);")
#         f.line(f"res[{idx_top}] = VSUB(res[{idx_top}], p_top);")
#         f.line()

#         carry_prop_2x4(f, pp, v, "res", "limb_mask", "carry", "idx_shift")

#         f.comment("If negative, add p back")
#         f.line(f"sign_mask = VSRA(res[{n24-1}], 63);")
#         f.line(f"sign_mask = VPERM512(idx_sign, sign_mask);")
#         f.line(f"res[{idx_top}] = VADD(res[{idx_top}], VAND(p_top, sign_mask));")
#         f.line(f"res[0] = VSUB(res[0], VAND(p_bot, sign_mask));")
#         f.line()

#         carry_prop_2x4(f, pp, v, "res", "limb_mask", "carry", "idx_shift")

#     f.end()

# Not designed for fp2 lazy red, correction step would need to handle input in [-p, 2p) like other fmts
def rdc_2x4(f, pp, v, strat):
    rdc_2x4_split(f, pp, v, strat)

def fp_mul_2x4(f, pp, v):
    f.start("void fp_mul_2x4(__m512i* res, __m512i* a, __m512i* b)")

    rdc_size = get_rdc_2x4_size(pp)
    f.line(f"__m512i t[{rdc_size}];")
    f.line()

    f.line("mp_mul_2x4(t, a, b);")
    f.line("rdc_2x4(res, t);")

    f.end()

def fp_sqr_2x4(f, pp, v):
    f.start("void fp_sqr_2x4(__m512i* res, __m512i* a)")

    rdc_size = get_rdc_2x4_size(pp)
    f.line(f"__m512i t[{rdc_size}];")
    f.line()

    f.line("mp_mul_2x4(t, a, a);")
    f.line("rdc_2x4(res, t);")

    f.end()

# Assumes num_limbs >= 4  and that c is small
def fp_div2_2x4(f, pp, v):
    f.start("void fp_div2_2x4(__m512i* res, __m512i* src)")

    f.array_to_single_vars(var_name="a", arr_name="src", count=pp.n24)
    f.declare_single_vars(var_name="t", count=pp.n24)
    f.line(f"const __m512i one = VSETALL(1);")

    # We case depending on the position of the two topmost limbs
    # (a0, a1, a2, a3)
    if pp.n24 == 1 and pp.rem24 == 0:
        fp_div2_2x4_n1_r0(f, pp, v)
    # (a0, a2, a4, -)
    # (a1, a3, -, -)
    elif pp.n24 == 2 and pp.rem24 == 1:
        fp_div2_2x4_n2_r1(f, pp, v)
    # Both in lane 2, lo at vector pp.n24 - 2 and hi at vector pp.n24 - 1
    # all vectors are of the form (a, b, c, -)
    elif (pp.n24 == 3 and pp.rem24 == 1) or (pp.n24 == 2 and pp.rem24 == 2):
        fp_div2_2x4_lo2_hi2(f, pp, v)
    # lo in lane 2 at last vector, hi in lane 3 at first vector
    elif (pp.n24 == 4 and pp.rem24 == 1) or (pp.n24 == 3 and pp.rem24 == 2) or (pp.n24 == 2 and pp.rem24 == 3):
        fp_div2_2x4_lo2_hi3(f, pp, v)
    # both in lane 3, last (4-rem24) % 4 vectors missing limb in lane 3
    else:
        fp_div2_2x4_lo3_hi3(f, pp, v)

    f.single_vars_to_array(var_name="t", arr_name="res", count=pp.n24)
    f.end()

# (a0, a1, a2, a3)
def fp_div2_2x4_n1_r0(f, pp, v):
    add_top_only = (pp.e-1) // pp.limb_len == pp.num_limbs - 1
    if add_top_only:
        f.line(f"const __m512i cnst = VSET(0, 0, 0, {v.p_plus_1_div2_top}, 0, 0, 0, {v.p_plus_1_div2_top});")
    else:
        f.line(f"const __m512i cnst = VSET(0, 0, {v.p_plus_1_div2_top}, {v.p_plus_1_div2_top_hi}, 0, 0, {v.p_plus_1_div2_top}, {v.p_plus_1_div2_top_hi});")
    f.line("__m512i odd_mask;")
    f.line()

    f.comment("odd_mask = all 1s if a is odd, else 0")
    f.line("odd_mask = VPERM256(a0, 0x00); // (a[0], a[0], a[0], a[0])")
    f.line(f"odd_mask = VSUB(VZERO(), VAND(odd_mask, one));")
    f.line()
    
    f.comment("a >> 1")
    f.line(f"t0 = VPERM256MASKZ(0x77, a0, 0x39); // (limb1, limb2, limb3, 0)")
    f.line(f"t0 = VSL(VAND(t0, one), {pp.limb_len-1});")
    f.line(f"t0 = VOR(t0, VSRL(a0, 1));")
    f.line()

    f.comment("Conditionally add (p+1)/2 = c * 2^(e-1)")
    f.line(f"t0 = VADD(t0, VAND(cnst, odd_mask));")

    #Carry propagation
    if not add_top_only:
        f.line(f"__m512i carry = VSRA(t0, {pp.limb_len});")
        f.line(f"t0 = VANDMASK(t0, 0x44, t0, VSETALL({v.limb_mask}));")
        f.line(f"carry = VPERM256MASKZ(0x88, carry, 0xaa);")
        f.line(f"t0 = VADD(t0, carry);")
    f.line()

# (a0, a2, a4, -)
# (a1, a3, -, -)
def fp_div2_2x4_n2_r1(f, pp, v):
    add_top_only = (pp.e-1) // pp.limb_len == pp.num_limbs - 1
    if add_top_only:
        f.line(f"const __m512i cnst = VSET(0, 0, {v.p_plus_1_div2_top}, 0, 0, 0, {v.p_plus_1_div2_top}, 0);")
    else:
        f.line(f"const __m512i cnst_lo = VSET(0, {v.p_plus_1_div2_top}, 0, 0, 0, {v.p_plus_1_div2_top}, 0, 0);")
        f.line(f"const __m512i cnst_hi = VSET(0, 0, {v.p_plus_1_div2_top_hi}, 0, 0, 0, {v.p_plus_1_div2_top_hi}, 0);")
    f.line("__m512i odd_mask;")
    f.line()

    f.comment("odd_mask = all 1s if a is odd, else 0")
    f.line("odd_mask = VPERM256(a0, 0x00); // (a[0], a[0], a[0], a[0])")
    f.line("odd_mask = VSUB(VZERO(), VAND(odd_mask, one));")
    f.line()
    
    f.comment("a >> 1")
    f.line("t0 = VSRL(a0, 1);")
    # transfer top bit from a1 to a0 and a3 to a2
    f.line(f"t0 = VADDMASK(t0, 0x33, t0, VSL(VAND(a1, one), {pp.limb_len-1}));")
    f.line("t1 = VSRL(a1, 1);")
    # transfer top bit from a2 to a1 and from a4 to a3
    f.line(f"t1 = VADDMASK(t1, 0x33, t1, VPERM256(VSL(VAND(a0, one), {pp.limb_len-1}), 0x99));")
    f.line()

    f.comment("Conditionally add (p+1)/2 = c * 2^(e-1)")
    if add_top_only:
        f.line(f"t0 = VADD(t0, VAND(cnst, odd_mask));")
    else:
        f.line(f"t1 = VADD(t1, VAND(cnst_lo, odd_mask));")
        f.line(f"t0 = VADD(t0, VAND(cnst_hi, odd_mask));")

    #Carry propagation
    if not add_top_only:
        f.line(f"__m512i carry = VSRL(t1, {pp.limb_len});")
        f.line(f"t1 = VANDMASK(t1, 0x22, t1, VSETALL({v.limb_mask}));")
        f.line(f"carry = VPERM256MASKZ(0x44, carry, 0x55);")
        f.line(f"t0 = VADD(t0, carry);")
    f.line()

# Both in lane 2, lo at vector pp.n24 - 2 and hi at vector pp.n24 - 1
# all vectors are of the form (a, b, c, -)
def fp_div2_2x4_lo2_hi2(f, pp, v):
    add_top_only = (pp.e-1) // pp.limb_len == pp.num_limbs - 1
    if add_top_only:
        f.line(f"const __m512i cnst = VSET(0, 0, {v.p_plus_1_div2_top}, 0, 0, 0, {v.p_plus_1_div2_top}, 0);")
    else:
        f.line(f"const __m512i cnst_lo = VSET(0, 0, {v.p_plus_1_div2_top}, 0, 0, 0, {v.p_plus_1_div2_top}, 0);")
        f.line(f"const __m512i cnst_hi = VSET(0, 0, {v.p_plus_1_div2_top_hi}, 0, 0, 0, {v.p_plus_1_div2_top_hi}, 0);")
    f.line("__m512i odd_mask;")
    f.line()

    f.comment("odd_mask = all 1s if a is odd, else 0")
    f.line("odd_mask = VPERM256(a0, 0x00); // (a[0], a[0], a[0], a[0])")
    f.line("odd_mask = VSUB(VZERO(), VAND(odd_mask, one));")
    f.line()
    
    f.comment("a >> 1")
    for i in range(pp.n24-1):
        f.line(f"t{i} = VSRL(a{i}, 1);")
        f.line(f"t{i} = VADDMASK(t{i}, 0x77, t{i}, VSL(VAND(a{i+1}, one), {pp.limb_len-1}));")
    # last vector
    f.line(f"t{pp.n24-1} = VSRL(a{pp.n24-1}, 1);")
    f.line(f"t{pp.n24-1} = VADDMASK(t{pp.n24-1}, 0x33, t{pp.n24-1}, VPERM256(VSL(VAND(a0, one), {pp.limb_len-1}), 0x99));")
    f.line()

    f.comment("Conditionally add (p+1)/2 = c * 2^(e-1)")
    if add_top_only:
        f.line(f"t{pp.n24-1} = VADD(t{pp.n24-1}, VAND(cnst, odd_mask));")
    else:
        f.line(f"t{pp.n24-2} = VADD(t{pp.n24-2}, VAND(cnst_lo, odd_mask));")
        f.line(f"t{pp.n24-1} = VADD(t{pp.n24-1}, VAND(cnst_hi, odd_mask));")

    #Carry propagation
    if not add_top_only:
        f.line(f"__m512i carry = VSRL(t{pp.n24-2}, {pp.limb_len});")
        f.line(f"t{pp.n24-2} = VANDMASK(t{pp.n24-2}, 0x44, t{pp.n24-2}, VSETALL({v.limb_mask}));")
        f.line(f"t{pp.n24-1} = VADD(t{pp.n24-1}, carry);")
    f.line()

# lo in lane 2 at last vector, hi in lane 3 at first vector
def fp_div2_2x4_lo2_hi3(f, pp, v):
    add_top_only = (pp.e-1) // pp.limb_len == pp.num_limbs - 1
    if add_top_only:
        f.line(f"const __m512i cnst = VSET(0, 0, 0, {v.p_plus_1_div2_top}, 0, 0, 0, {v.p_plus_1_div2_top});")
    else:
        f.line(f"const __m512i cnst_lo = VSET(0, 0, {v.p_plus_1_div2_top}, 0, 0, 0, {v.p_plus_1_div2_top}, 0);")
        f.line(f"const __m512i cnst_hi = VSET(0, 0, 0, {v.p_plus_1_div2_top_hi}, 0, 0, 0, {v.p_plus_1_div2_top_hi});")
    f.line("__m512i odd_mask;")
    f.line()

    f.comment("odd_mask = all 1s if a is odd, else 0")
    f.line("odd_mask = VPERM256(a0, 0x00); // (a[0], a[0], a[0], a[0])")
    f.line("odd_mask = VSUB(VZERO(), VAND(odd_mask, one));")
    f.line()
    
    f.comment("a >> 1")
    for i in range(pp.n24-1):
        f.line(f"t{i} = VSRL(a{i}, 1);")
        f.line(f"t{i} = VADDMASK(t{i}, 0x77, t{i}, VSL(VAND(a{i+1}, one), {pp.limb_len-1}));")
    # last vector
    f.line(f"t{pp.n24-1} = VSRL(a{pp.n24-1}, 1);")
    f.line(f"t{pp.n24-1} = VADDMASK(t{pp.n24-1}, 0x77, t{pp.n24-1}, VPERM256(VSL(VAND(a0, one), {pp.limb_len-1}), 0x39));")
    f.line()

    f.comment("Conditionally add (p+1)/2 = c * 2^(e-1)")
    if add_top_only:
        f.line(f"t0 = VADD(t0, VAND(cnst, odd_mask));")
    else:
        f.line(f"t{pp.n24-1} = VADD(t{pp.n24-1}, VAND(cnst_lo, odd_mask));")
        f.line(f"t0 = VADD(t0, VAND(cnst_hi, odd_mask));")

    #Carry propagation
    if not add_top_only:
        f.line(f"__m512i carry = VSRL(t{pp.n24-1}, {pp.limb_len});")
        f.line(f"t{pp.n24-1} = VANDMASK(t{pp.n24-1}, 0x44, t{pp.n24-1}, VSETALL({v.limb_mask}));")
        f.line(f"carry = VPERM256MASKZ(0x88, carry, 0xaa);")
        f.line(f"t0 = VADD(t0, carry);")
    f.line()

# both in lane 3, last (4-rem24) % 4 vectors missing limb in lane 3
def fp_div2_2x4_lo3_hi3(f, pp, v):
    add_top_only = (pp.e-1) // pp.limb_len == pp.num_limbs - 1
    if add_top_only:
        f.line(f"const __m512i cnst = VSET(0, 0, 0, {v.p_plus_1_div2_top}, 0, 0, 0, {v.p_plus_1_div2_top});")
    else:
        f.line(f"const __m512i cnst_lo = VSET(0, 0, 0, {v.p_plus_1_div2_top}, 0, 0, 0, {v.p_plus_1_div2_top});")
        f.line(f"const __m512i cnst_hi = VSET(0, 0, 0, {v.p_plus_1_div2_top_hi}, 0, 0, 0, {v.p_plus_1_div2_top_hi});")
    f.line("__m512i odd_mask;")
    f.line()

    f.comment("odd_mask = all 1s if a is odd, else 0")
    f.line("odd_mask = VPERM256(a0, 0x00); // (a[0], a[0], a[0], a[0])")
    f.line("odd_mask = VSUB(VZERO(), VAND(odd_mask, one));")
    f.line()
    
    f.comment("a >> 1")
    idx_top, _ = get_top_idx(pp)
    # Stage 1: next vector has all 4 limbs
    for i in range(idx_top):
        f.line(f"t{i} = VSRL(a{i}, 1);")
        f.line(f"t{i} = VADD(t{i}, VSL(VAND(a{i+1}, one), {pp.limb_len-1}));")
    # Stage 2: The top vector is the last with a limb in lane 3
    num_vec_with_lane3_zero = (4 - pp.rem24) % 4
    for i in range(num_vec_with_lane3_zero):
        j = i + idx_top
        f.line(f"t{j} = VSRL(a{j}, 1);")
        f.line(f"t{j} = VADDMASK(t{j}, 0x77, t{j}, VSL(VAND(a{j+1}, one), {pp.limb_len-1}));")
    # Stage 3: The last vector
    f.line(f"t{pp.n24-1} = VSRL(a{pp.n24-1}, 1);")
    f.line(f"t{pp.n24-1} = VADDMASK(t{pp.n24-1}, 0x77, t{pp.n24-1}, VPERM256(VSL(VAND(a0, one), {pp.limb_len-1}), 0x39));")
    f.line()

    f.comment("Conditionally add (p+1)/2 = c * 2^(e-1)")
    if add_top_only:
        f.line(f"t{idx_top} = VADD(t{idx_top}, VAND(cnst, odd_mask));")
    else:
        f.line(f"t{idx_top-1} = VADD(t{idx_top-1}, VAND(cnst_lo, odd_mask));")
        f.line(f"t{idx_top} = VADD(t{idx_top}, VAND(cnst_hi, odd_mask));")

    #Carry propagation
    if not add_top_only:
        f.line(f"__m512i carry = VSRL(t{idx_top-1}, {pp.limb_len});")
        f.line(f"t{idx_top-1} = VANDMASK(t{idx_top-1}, 0x88, t{idx_top-1}, VSETALL({v.limb_mask}));")
        f.line(f"t{idx_top} = VADD(t{idx_top}, carry);")
    f.line()

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

def fp_sqrt_2x4(f, pp, v):
    f.start("void fp_sqrt_2x4(__m512i* res, __m512i* a)")
    
    e_minus_2 = pp.e - 2
    f.comment(f"Candidate root is a^((p+1)/4), with (p+1)/4 = {pp.c}*2^{e_minus_2}")
    f.line(f"__m512i t[{pp.n24}];")
    f.line()

    # Step 1: Compute a^c
    if pp.c > 1:
        write_dbl_and_add(f, pp, v, pp.c, fmt="2x4")       

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

def fp_exp3div4_2x4(f, pp, v, strat, chain):
    fmt = "2x4"
    arr_size = pp.n24
    if strat.gen_addchain:
        fun_name = "fp_exp3div4_2x4"
        write_exp_function(f, pp, v, chain, fun_name, fmt, arr_size)
    else:
        write_fp_exp3_div4_naive(f, pp, v, fmt, arr_size)

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
        if j >= pp.num_limbs:
            lines[j % n].append(f"t[{t_idx}] = 0;")
            continue
        
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


        limb_idx = 3 * n + j
        if limb_idx >= pp.num_limbs:
            f.line("t[4*i+3] = 0;")
        else:
            plusi64 = "" if i64 == 0 else f" + {i64}"
            l = pp.last_limb_len if limb_idx == pp.num_limbs-1 else pp.limb_len
            if leftover == 64:
                f.line(f"t[4*i+3] = unvec[{v.num_words64} * i{plusi64}] & {v.limb_mask};")
            elif leftover >= l:
                shift = 64 - leftover
                f.line(f"t[4*i+3] = (unvec[{v.num_words64} * i{plusi64}] >> {shift}) & {v.limb_mask};")
            else: #leftover < l
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

def fp_limbs_to_64_2x4(f, pp, v, strat):
    f.start("void fp_limbs_to_64_2x4(uint64_t* res, __m512i* src)")

    f.line(f"uint64_t t[8];")
    if strat.mod_2p:
        f.line(f"__m512i s[{pp.n24}];")
        f.line("fp_normalize_2x4(s, src);")
        src = "s"
    else:
        src = "src"
    f.line()

    n = pp.n24
    l = pp.limb_len
    i64 = 0
    leftover = 64
    indices = [[] for _ in range(n)]
    lines_l = [[] for _ in range(n)]
    lines_r = [[] for _ in range(n)]

    for j in range(min(3*n, pp.num_limbs)):
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
        f.line(f"VSTORE(t, {src}[{j}]);")
        f.line("for (int i = 0; i < 2; i++) {")
        f.indent()
        for k in range(len(indices[j])):
            idx = indices[j][k]
            line_l = lines_l[j][k]
            line_r = lines_r[j][k]
            op = "|=" if idx in seen else "="
            f.line(f"{line_l} {op} {line_r}")
            seen.add(idx)

        limb_idx = 3*n + j
        if limb_idx < pp.num_limbs:
            plusi64 = "" if i64 == 0 else f" + {i64}"
            op = "|=" if i64 in seen else "="
            seen.add(i64)

            l = pp.last_limb_len if j == n-1 else pp.limb_len

            if leftover == 64:
                f.line(f"res[{v.num_words64} * i{plusi64}] {op} t[4*i+3] & {v.limb_mask};")
            elif leftover >= l:
                shift = 64 - leftover
                f.line(f"res[{v.num_words64} * i{plusi64}] {op} (t[4*i+3] & {v.limb_mask}) << {shift};")
            else: #leftover < l
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

def rand_fp_2x4(f, pp, v):
    f.start(f"void rand_fp_2x4(__m512i* res)")

    f.line(f"__m512i zero[{pp.n24}];")
    f.line(f"__m512i mask;")
    f.line(f"fp_setzero_2x4(zero);")
    f.line()

    def vset(vals):
        return f"VSET({', '.join(str(x) for x in vals)})"

    for i in range(pp.n24):
        mask_vals = []
        for lane in range(8):
            q = lane % 4
            limb_idx = i + q * pp.n24
            if limb_idx >= pp.num_limbs:
                mask_vals.append("0")
            elif limb_idx == pp.num_limbs - 1:
                mask_vals.append(v.last_limb_mask)
            else:
                mask_vals.append(v.limb_mask)

        f.line(f"mask = {vset(mask_vals)};")
        f.line(f"res[{i}] = VAND(rand_512i(), mask);")
    f.line()

    f.comment("Reduction")
    f.line("fp_add_2x4(res, res, zero);")
    f.end()

def write_fp_2x4(f, pp, v, strat, chain):
    f.banner("2x4")

    fp_setzero_2x4(f, pp, v)
    fp_set_binary_2x4(f, pp, v)
    fp_copy_2x4(f, pp, v)
    fp_duplicate_2x4(f, pp, v)
    fp_is_equal_2x4(f, pp, v, strat)
    fp_is_zero_2x4(f, pp, v, strat)
    fp_select_2x4(f, pp, v)

    fp_add_2x4(f, pp, v, strat)
    fp_sub_2x4(f, pp, v, strat)
    fp_neg_2x4(f, pp, v, strat)
    fp_mixaddsub_2x4(f, pp, v, strat)
    fp_normalize_2x4(f, pp, v)

    mp_mul_2x4(f, pp, v)
    rdc_2x4(f, pp, v, strat)
    fp_mul_2x4(f, pp, v)
    fp_sqr_2x4(f, pp, v)
    fp_tomont_2x4(f, pp, v)
    fp_frommont_2x4(f, pp, v)

    fp_div2_2x4(f, pp, v)
    fp_div3_2x4(f, pp, v)
    fp_inv_2x4(f, pp, v)
    fp_sqrt_2x4(f, pp, v)
    fp_exp3div4_2x4(f, pp, v, strat, chain)

    fp_64_to_limbs_2x4(f, pp, v)
    fp_limbs_to_64_2x4(f, pp, v, strat)

    rand_fp_2x4(f, pp, v)