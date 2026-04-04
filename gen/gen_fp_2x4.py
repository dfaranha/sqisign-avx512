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
        f.line(f"{tmp_name}[{pp.n24-1}] = VANDMASK({tmp_name}[{pp.n24-1}], 0x77, {tmp_name}[{pp.n24LIMBS_24 - 1}], {mask_name});")
        f.line(f"{carry_name} = VPERM512MASKZ(0xee, {carry_name}, {idx_shift_name});")
        f.line(f"{tmp_name}[0] = VADD({tmp_name}[0], {carry_name});")
    f.comment(f"Final straight ripple to finish the exact (4 * n24) limb sequence")
    for j in range(pp.n24-1):
        f.line(f"{carry_name} = VSRA({tmp_name}[{j}], {v.limb_len});")
        f.line(f"{tmp_name}[{j}] = VAND({tmp_name}[{j}], {mask_name});")
        f.line(f"{tmp_name}[{j+1}] = VADD({tmp_name}[{j+1}], {carry_name});")
    f.line()

# TODO: Does not work when num_limbs % 4 != 0
def fp_add_2x4(f, pp, v):
    f.start("void fp_add_2x4(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{pp.n24}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_bot = VSET(1, 0, 0, 0, 1, 0, 0, 0);")
    f.line(f"const __m512i p_top = VSET(0, 0, 0, {v.p_plus_1_top}, 0, 0, 0, {v.plus_1_top});") # wrong if num_limbs % 4 != 0, need to use correct top idx
    f.comment("Maps odd lanes to the next even lane (1->2, 3->4, 5->6)")
    f.line("__m512i idx_shift = VSET(0, 0, 1, 2, 0, 4, 5, 6);")
    f.comment("Broadcasts the sign bits from lanes 3 and 7 to their respective elements")
    f.line("__m512i idx_sign = VSET(3, 3, 3, 3, 7, 7, 7, 7);")
    f.line()

    for i in range(pp.n24):
        f.line(f"t[{i}] = VADD(a[{i}, b[{i}]]);")
    f.line()

    f.comment("Subtract p using the special shape of p")
    f.line(f"t[0] = VADD(t[0], p_bot);")
    f.line(f"t[{pp.n24-1}] = VSUB(t[{pp.n24-1}], p_top);")
    f.line()

    carry_prop_2x4(f, pp, v, "t", "limb_mask", "carry", "idx_shift")

    f.comment("If the result is negative, we add p")
    f.line(f"sign_mask = VSRA(t[{pp.n24-1}], 63);")
    f.line(f"sign_mask = VPERMV(idx_sign, sign_mask);")
    f.line(f"")
    f.line()

    f.end()

def write_fp_2x4(f, pp, v):
    f.banner("2x4")

    fp_setzero_2x4(f, pp, v)
    fp_copy_2x4(f, pp, v)
    fp_duplicate_2x4(f, pp, v)
    fp_is_equal_2x4(f, pp, v)
    fp_is_zero_2x4(f, pp, v)
    fp_select_2x4(f, pp, v)