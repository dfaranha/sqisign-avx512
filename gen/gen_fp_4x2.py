from math import ceil, floor

# 4x2 format stores an Fp element in n42 vectors.
# The i-th vector stores (a[i], a[i+n42], b[i], b[i+n42], c[i], c[i+n42], d[i], d[i+n42])
# If num_limbs is odd, the last limb is (a[n42-1], 0, b[n42-1], 0, c[n42-1], 0, d[n42-1], 0)

def fp_setzero_4x2(f, pp, v):
    f.start("void fp_setzero_4x2(__m512i* res)")
    for i in range(pp.n42):
        f.line(f"res[{i}] = VZERO();")
    f.end()

def fp_copy_4x2(f, pp, v):
    f.start("void fp_copy_4x2(__m512i* res, __m512i* a)")
    for i in range(pp.n42):
        f.line(f"res[{i}] = a[{i}];")
    f.end()

def fp_duplicate_4x2(f, pp, v):
    f.start("void fp_duplicate_4x2(__m512i* res, const uint64_t* a)")
    n_full = pp.n42 - pp.rem42
    for i in range(n_full):
        f.line(f"res[{i}] = VSET(a[{i}], a[{i+pp.n42}], a[{i}], a[{i+pp.n42}], a[{i}], a[{i+pp.n42}], a[{i}], a[{i+pp.n42}]);")

    if pp.rem42 == 1:
        f.line(f"res[{n_full}] = VSET(a[{n_full}], 0, a[{n_full}], 0, a[{n_full}], 0, a[{n_full}], 0);")
    f.end()

def fp_is_equal_4x2(f, pp, v):
    f.start("int fp_is_equal_4x2(__m512i* a, __m512i* b)")

    f.line(f"__m512i zero[{v.n42}], norm_a[{v.n42}], norm_b[{v.n42}];")
    f.line(f"__mmask8 acc;")
    f.line(f"fp_setzero_4x2(zero);")
    f.line()

    f.comment("Normalize both to [0, p-1] by adding zero")
    f.line("fp_add_4x2(norm_a, a, zero);")
    f.line("fp_add_4x2(norm_b, b, zero);")
    f.line()

    f.line("acc = 0xff;")
    for i in range(pp.n42):
        f.line(f"acc &= VCMPEQ(norm_a[{i}], norm_b[{i}]);")
    f.line()

    f.comment("a = b iff both halves are equal")  
    f.line("return (acc & (acc >> 1) & 0x1)")
    f.indent()
    f.line("| ((acc >> 1) & (acc >> 2) & 0x2)")
    f.line("| ((acc >> 2) & (acc >> 3) & 0x4)")
    f.line("| ((acc >> 3) & (acc >> 4) & 0x8);")
    f.unindent()

    f.end()

def fp_is_zero_4x2(f, pp, v):
    f.start("int fp_is_zero_4x2(__m512i* a)")

    f.line(f"__m512i zero[{v.n42}], norm_a[{v.n42}];")
    f.line(f"__mmask8 acc;")
    f.line(f"fp_setzero_4x2(zero);")
    f.line()

    f.comment("Normalize a to [0, p-1] by adding zero")
    f.line("fp_add_4x2(norm_a, a, zero);")
    f.line()

    f.line("acc = 0xff;")
    for i in range(pp.n42):
        f.line(f"acc &= VCMPEQ(norm_a[{i}], zero[{i}]);")
    f.line()

    f.comment("a = 0 iff both halves are 0")  
    f.line("return (acc & (acc >> 1) & 0x1)")
    f.indent()
    f.line("| ((acc >> 1) & (acc >> 2) & 0x2)")
    f.line("| ((acc >> 2) & (acc >> 3) & 0x4)")
    f.line("| ((acc >> 3) & (acc >> 4) & 0x8);")
    f.unindent()

    f.end()

def carry_prop_4x2(f, pp, v, tmp_name, mask_name, carry_name, start_idx=0):
    f.comment("Carry propagation")
    for i in range(1+start_idx, pp.n42+start_idx):
        f.line(f"{tmp_name}[{i}] = VADDMASK({tmp_name}[{i}], 0x55, {tmp_name}[{i}], VSRA({tmp_name}[{i-1}], {v.limb_len}));")
    f.line(f"{carry_name} = VSRA({tmp_name}[{pp.n42-1+start_idx}], {v.limb_len});")
    f.line(f"{carry_name} = VSHUFFLEMASKZ(0xcccc, {carry_name}, 0x4e);")
    f.line(f"{tmp_name}[{start_idx}] = VADDMASK({tmp_name}[{start_idx}], 0xaa, {tmp_name}[{start_idx}], {carry_name});")
    for i in range(1 + start_idx, pp.n42 - pp.rem42 + start_idx):
        f.line(f"{tmp_name}[{i}] = VADDMASK({tmp_name}[{i}], 0xaa, {tmp_name}[{i}], VSRA({tmp_name}[{i-1}], {v.limb_len}));")
    
    if pp.rem42 == 0:
        for i in range(start_idx, pp.n42-1+start_idx):
            f.line(f"{tmp_name}[{i}] = VAND({tmp_name}[{i}], {mask_name});")
        f.line(f"{tmp_name}[{pp.n42-1 + start_idx}] = VANDMASK({tmp_name}[{pp.n42-1 + start_idx}], 0x55, {tmp_name}[{pp.n42-1 + start_idx}], {mask_name});")
    else:
        for i in range(start_idx, pp.n42-2 + start_idx):
            f.line(f"{tmp_name}[{i}] = VAND({tmp_name}[{i}], {mask_name});")
        f.line(f"{tmp_name}[{pp.n42-2 + start_idx}] = VANDMASK({tmp_name}[{pp.n42-2 + start_idx}], 0x55, {tmp_name}[{pp.n42-2 + start_idx}], {mask_name});")
        f.line(f"{tmp_name}[{pp.n42-1 + start_idx}] = VANDMASK({tmp_name}[{pp.n42-1 + start_idx}], 0x55, {tmp_name}[{pp.n42-1 + start_idx}], {mask_name});")
    f.line()

def fp_add_4x2(f, pp, v):
    f.start("void fp_add_4x2(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{pp.n42}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);")
    f.line(f"const __m512i p_top = VSET(0, {v.p_plus_1_top}, 0, {v.p_plus_1_top}, 0, {v.p_plus_1_top}, 0, {v.p_plus_1_top});")
    f.line()

    f.comment("Add limbs")
    for i in range(pp.n42):
        f.line(f"t[{i}] = VADD(a[{i}], b[{i}]);")
    f.line()

    f.comment("Subtract p, using special shape of p")
    top_idx = pp.n42 - 1 - pp.rem42
    f.line("t[0] = VADD(t[0], p_bot);")
    f.line(f"t[{top_idx}] = VSUB(t[{top_idx}], p_top);")
    f.line()

    carry_prop_4x2(f, pp, v, "t", "limb_mask", "carry")

    f.comment("If result is negative, we undo the subtraction by p")
    f.comment("sign_mask = {0, sign_mask1, 0, sign_mask2, ...}")
    f.line(f"sign_mask = VSRA(t[{top_idx}], 63);")
    f.line(f"t[{top_idx}] = VADD(t[{top_idx}], VAND(sign_mask, p_top));")
    f.line(f"sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);")
    f.line("t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_4x2(f, pp, v, "t", "limb_mask", "carry")

    for i in range(pp.n42):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_sub_4x2(f, pp, v):
    f.start("void fp_sub_4x2(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{pp.n42}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);")
    f.line(f"const __m512i p_top = VSET(0, {v.p_plus_1_top}, 0, {v.p_plus_1_top}, 0, {v.p_plus_1_top}, 0, {v.p_plus_1_top});")
    f.line()

    f.comment("Subtract limbs")
    for i in range(pp.n42):
        f.line(f"t[{i}] = VSUB(a[{i}], b[{i}]);")
    f.line()

    carry_prop_4x2(f, pp, v, "t", "limb_mask", "carry")

    f.comment("If result is negative, we undo the subtraction by p")
    f.comment("sign_mask = {0, sign_mask1, 0, sign_mask2, ...}")
    top_idx = pp.n42 - 1 - pp.rem42
    f.line(f"sign_mask = VSRA(t[{top_idx}], 63);")
    f.line(f"t[{top_idx}] = VADD(t[{top_idx}], VAND(sign_mask, p_top));")
    f.line(f"sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);")
    f.line("t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_4x2(f, pp, v, "t", "limb_mask", "carry")

    for i in range(pp.n42):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_neg_4x2(f, pp, v):
    f.start("void fp_neg_4x2(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{pp.n42}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);")
    f.line(f"const __m512i p_top = VSET(0, {v.p_plus_1_top}, 0, {v.p_plus_1_top}, 0, {v.p_plus_1_top}, 0, {v.p_plus_1_top});")
    f.line(f"const __m512i zero = VZERO();")
    f.line()

    f.comment("Subtract limbs")
    for i in range(pp.n42):
        f.line(f"t[{i}] = VSUB(zero, a[{i}]);")
    f.line()

    carry_prop_4x2(f, pp, v, "t", "limb_mask", "carry")

    f.comment("If result is negative, we undo the subtraction by p")
    f.comment("sign_mask = {0, sign_mask1, 0, sign_mask2, ...}")
    top_idx = pp.n42 - 1 - pp.rem42
    f.line(f"sign_mask = VSRA(t[{top_idx}], 63);")
    f.line(f"t[{top_idx}] = VADD(t[{top_idx}], VAND(sign_mask, p_top));")
    f.line(f"sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);")
    f.line("t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_4x2(f, pp, v, "t", "limb_mask", "carry")

    for i in range(pp.n42):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def fp_mixaddsub_4x2(f, pp, v):
    f.start("void fp_mixaddsub_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")
    f.line(f"__m512i t[{pp.n42}], carry, sign_mask;")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_bot = VSET(1, 0, 1, 0, 1, 0, 1, 0);")
    f.line(f"const __m512i p_top = VSET(0, {v.p_plus_1_top}, 0, {v.p_plus_1_top}, 0, {v.p_plus_1_top}, 0, {v.p_plus_1_top});")
    f.line(f"const __mmask8 sub_mask = ~add_mask;")
    f.line()

    for i in range(pp.n42):
        f.line(f"t[{i}] = VSUBMASK(a[{i}], sub_mask, a[{i}], b[{i}]);")
    f.line()

    for i in range(pp.n42):
        f.line(f"t[{i}] = VADDMASK(t[{i}], add_mask, a[{i}], b[{i}]);")
    f.line()

    f.comment("a + b - p")
    top_idx = pp.n42 - 1 - pp.rem42
    f.line("t[0] = VADDMASK(t[0], add_mask, t[0], p_bot);")
    f.line(f"t[{top_idx}] = VSUBMASK(t[{top_idx}], add_mask, t[{top_idx}], p_top);")
    f.line()

    carry_prop_4x2(f, pp, v, "t", "limb_mask", "carry")

    f.comment("If result is negative, we undo the subtraction by p")
    f.comment("sign_mask = {0, sign_mask1, 0, sign_mask2, ...}")
    f.line(f"sign_mask = VSRA(t[{top_idx}], 63);")
    f.line(f"t[{top_idx}] = VADD(t[{top_idx}], VAND(sign_mask, p_top));")
    f.line(f"sign_mask = VSHUFFLEMASKZ(0xffff, sign_mask, 0x4e);")
    f.line("t[0] = VSUB(t[0], VAND(p_bot, sign_mask));")
    f.line()

    carry_prop_4x2(f, pp, v, "t", "limb_mask", "carry")

    for i in range(pp.n42):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_add_4x2(f, pp, v):
    f.comment("Single-length multiprecision addition, no correction, no carry propagation")
    f.start("void mp_add_4x2(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{pp.n42}];")
    f.line()

    for i in range(pp.n42):
        f.line(f"t[{i}] = VADD(a[{i}], b[{i}]);")
    f.line()

    for i in range(pp.n42):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_add_p_sub_4x2(f, pp, v):
    f.start("void mp_add_p_sub_4x2(__m512i* res, __m512i* a, __m512i* b)")

    f.line(f"__m512i t[{pp.n42}];")
    f.line(f"const __m512i p_bod = VSETALL({v.limb_mask});")
    f.line(f"const __m512i p_top = VSET({v.limb_mask}, {v.p_plus_1_top}-1, {v.limb_mask}, {v.p_plus_1_top}-1, {v.limb_mask}, {v.p_plus_1_top}-1, {v.limb_mask}, {v.p_plus_1_top}-1);")
    f.line()

    top_idx = pp.n42-1 - pp.rem42
    for i in range(top_idx):
        f.line(f"t[{i}] = VSUB(VADD(a[{i}], p_bod), b[{i}]);")
    f.line(f"t[{top_idx}] = VSUB(VADD(a[{top_idx}], p_top), b[{top_idx}]);")
    if pp.rem42 == 1:
        f.line(f"t[{top_idx+1}] = VADDMASK(a[{top_idx+1}], 0x55, a[{top_idx+1}], p_bod);")
        f.line(f"t[{top_idx+1}] = VSUBMASK(t[{top_idx+1}], 0x55, t[{top_idx+1}], b[{top_idx+1}]);")
    f.line()

    for i in range(pp.n42):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_mixaddsubx2_4x2(f, pp, v):
    f.start("void mp_mixaddsubx2_4x2(__m512i* res, __m512i* a, __m512i* b, const __mmask8 add_mask)")
    n_prod = 3*pp.n42
    f.line(f"__m512i t[{n_prod}];")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line("const __mmask8 sub_mask = ~add_mask;")
    f.line()

    for i in range(n_prod):
        f.line(f"t[{i}] = VSUBMASK(a[{i}], sub_mask, a[{i}], b[{i}]);")
    f.line()

    for i in range(n_prod):
        f.line(f"t[{i}] = VADDMASK(t[{i}], add_mask, a[{i}], b[{i}]);")
    f.line()

    for i in range(n_prod):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_dblsubx2_4x2(f, pp, v):
    f.comment("Double-length multiprecision subtraction res = res - a - b, no correction")
    f.start("void mp_dblsubx2_4x2(__m512i* res, __m512i* a, __m512i* b)")

    n_prod = 3*pp.n42
    f.line(f"__m512i t[{n_prod}];")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line()

    for i in range(n_prod):
        f.line(f"t[{i}] = VSUB(res[{i}], a[{i}]);")
    f.line()

    for i in range(n_prod):
        f.line(f"t[{i}] = VSUB(t[{i}], b[{i}]);")
    f.line()

    for i in range(n_prod):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_subx2_4x2(f, pp, v):
    f.comment("Double-length multiprecision subtraction res = res - a")
    f.start("void mp_subx2_4x2(__m512i* res, __m512i* a)")

    n_prod = 3*pp.n42
    f.line(f"__m512i t[{n_prod}];")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line()

    for i in range(n_prod):
        f.line(f"t[{i}] = VSUB(res[{i}], a[{i}]);")
    f.line()

    for i in range(n_prod):
        f.line(f"res[{i}] = t[{i}];")

    f.end()

def mp_mul_4x2(f, pp, v):
    f.comment("Algorithm 2 from \"Finite Field Arithmetic Using AVX-512 For Isogeny-Based Cryptography\" by Orisaka, Aranha and López")
    f.start("void mp_mul_4x2(__m512i* res, __m512i* a, __m512i* b)")

    #TODO: Might be suboptimal for odd number of limbs
    n_prod = pp.n42 + pp.num_limbs
    f.line(f"__m512i t[{n_prod}], sl[{pp.n42}], sh[{pp.n42-pp.rem42}];")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i zero = VZERO();")
    f.line()

    f.comment("Duplicating limbs of b with shuffles")
    f.comment(f"If b[0] = {{a_0, a_{pp.n42}, b_0, b_{pp.n42}, ...}} then sl0 = {{a_0, a_0, b_0, b_0, ...}}")
    for i in range(pp.n42):
        f.line(f"sl[{i}] = VSHUFFLE(b[{i}], 0x44);")
    f.comment(f"If b[0] = {{a_0, a_{pp.n42}, b_0, b_{pp.n42}, ...}} then sh0 = {{a_{pp.n42}, a_{pp.n42}, b_{pp.n42}, b_{pp.n42}, ...}}")
    for i in range(pp.n42-pp.rem42):
        f.line(f"sh[{i}] = VSHUFFLE(b[{i}], 0xee);")
    f.line()

    f.comment("Schoolbook")
    # i in [0, n42-1], j in [0, 2*n42-1-rem42]
    # sum is at most (n42-1) + (2*n42 - 1 - rem42) = 3*n42 -2 - rem42
    n_i_max = pp.n42 - 1
    n_j_max = pp.num_limbs - 1

    for k in range(pp.n42 + pp.num_limbs - 1):
        i_min = max(0, k - n_j_max)
        i_max = min(n_i_max, k)
        src_lo = "zero" if k == 0 else f"t[{k}]"

        for i in range(i_min, i_max+1):
            j = k - i
            src_hi = "zero" if i == i_min else f"t[{k+1}]"
            bj = f"sl[{j}]" if j < pp.n42 else f"sh[{j-pp.n42}]"

            f.line(f"t[{k}] = VMULADDL({src_lo}, a[{i}], {bj});")
            f.line(f"t[{k+1}] = VMULADDH({src_hi}, a[{i}], {bj});")
        f.line(f"t[{k+1}] = VSL(t[{k+1}], {v.limb_align});")
        f.line()
    
    for i in range(n_prod):
        f.line(f"res[{i}] = t[{i}];")

    #TODO this is only when R is larger to allow lazy red and the number of limbs is even
    f.line(f"res[{n_prod}] = zero;")

    f.end()

#TODO this does not yet handle odd limbs or a smaller R

def rdc_4x2(f, pp, v):
    f.comment("Algorithm 3 from \"Finite Field Arithmetic Using AVX-512 For Isogeny-Based Cryptography\" by Orisaka, Aranha and López")
    f.comment("Modifies a")
    f.start("void rdc_4x2(__m512i* res, __m512i* a)")

    f.line("__m512i carry, hi, shuf;")
    f.line(f"const __m512i top = VSETALL({v.p_plus_1_top});")
    f.line(f"const __m512i limb_mask = VSETALL({v.limb_mask});")
    f.line(f"const __m512i zero = VZERO();")
    f.line(f"const __m512i one = VSETALL(1);")
    f.line()

    f.comment("Montgomery reduction loop")
    top_idx = pp.n42 - 1 - pp.rem42
    #TODO it is only + 1 when R is larger to allow lazy red
    for i in range(pp.num_limbs+1):
        if i == 0:
            f.comment("using the fact that -p^{-1} mod 2^(limb_len) = 1 + c* 2^f mod 2^(limb_len) = 1 as f > limb_len")
        f.line(f"carry = VSRA(a[{i}], {v.limb_len});")
        f.line(f"a[{i}] = VAND(a[{i}], limb_mask);")
        f.line(f"a[{i+1}] = VADD(a[{i+1}], carry);")
        if i == 0:
            f.comment("a = a + u_i * p * 2^(limb_len * i)")
            f.comment("To multiply by p, we multiply by (p+1) and subtract one.")
            f.comment("However, since we will drop the lower limbs when we divide later, we don't need to subtract.")

        f.line(f"shuf = VSHUFFLEMASKZ(0xcccc, a[{i}], 0x4e);")
        f.line(f"a[{i+top_idx}] = VMULADDL(a[{i+top_idx}], shuf, top);")
        f.line(f"hi = VMULADDH(zero, shuf, top);")
        f.line(f"hi = VSL(hi, {v.limb_align});")
        f.line(f"a[{i+top_idx+1}] = VADD(a[{i+top_idx+1}], hi);")
        if i == 0:
            f.comment("Folding step")
        f.line(f"a[{i+4}] = VADDMASK(a[{i+4}], 0x55, a[{i+4}], VSHUFFLE(a[{i}], 0x4e));")
        f.line()

    f.comment(f"Result is now in a[{pp.num_limbs+1}], ..., a[{pp.num_limbs + pp.n42}]")
    start_idx = pp.num_limbs+1
    carry_prop_4x2(f, pp, v, "a", "limb_mask", "carry", start_idx)

    f.comment(f"The result can be {pp.pbits+1} bits.")
    f.comment(f"If so, we do a final subtraction by p to bring it down to {pp.pbits} bits.")
    shift = 64 - (pp.pbits + 1 - (pp.num_limbs - 1) * pp.limb_len)
    f.line(f"hi = VSL(a[{pp.n42 + pp.num_limbs}], {shift});")
    f.line("__mmask8 msb_mask = VSIGN(hi);")
    f.line("__mmask8 mask_odd  = msb_mask & 0xaa;")
    f.line("__mmask8 mask_even = mask_odd >> 1;")
    top_idx = start_idx + top_idx
    f.line(f"a[{top_idx}] = VSUBMASK(a[{top_idx}], mask_odd, a[{top_idx}], top);")
    f.line(f"a[{start_idx}]  = VADDMASK(a[{start_idx}], mask_even, a[{start_idx}], one);")
    f.line()

    f.comment("Output result: final carry propagation is left for the final result.")
    f.comment("               Bottom limb can have up to 49 bits (because of the +1).")
    for i in range(pp.n42):
        f.line(f"res[{i}] = a[{i+start_idx}];")

    f.end()

def fp_mul_4x2(f, pp, v):
    f.start("void fp_mul_4x2(__m512i* res, __m512i* a, __m512i* b)")

    # We need one extra limb for lazy red. When n42 is odd, there is a limb available in the last vec.
    # TODO: Depends on Montgomery constant used, may not need last limb if R = 2^(num_limbs * limb_len) mod p
    f.line(f"__m512i t[{pp.n42 + pp.num_limbs + 1}];")
    f.line()

    f.line("mp_mul_4x2(t, a, b);")
    f.line("rdc_4x2(res, t);")

    f.end()

def fp_sqr_4x2(f, pp, v):
    f.start("void fp_sqr_4x2(__m512i* res, __m512i* a)")
    f.line("fp_mul_4x2(res, a, a);")
    f.end()

def fp_tomont_4x2(f, pp, v):
    f.start("void fp_tomont_4x2(__m512i* res, __m512i* a)")

    f.line(f"__m512i vR2[{pp.n42}];")
    f.line("fp_duplicate_4x2(vR2, R2);")
    f.line("fp_mul_4x2(res, a, vR2);")

    f.end()

def fp_frommont_4x2(f, pp, v):
    f.start("void fp_frommont_4x2(__m512i* res, __m512i* a)")

    f.line(f"__m512i one[{pp.n42}];")
    f.line(f"fp_setzero_4x2(one);")
    f.line(f"one[0] = VSET(1, 0, 1, 0, 1, 0, 1, 0);")
    f.line(f"fp_mul_4x2(res, a, one);")

    f.end()

def fp_div3_4x2(f, pp, v):
    f.start("void fp_div3_4x2(__m512i* res, __m512i* src)")

    f.line(f"__m512i t[{pp.n42}];")
    f.line(f"fp_duplicate_4x2(t, {v.inv3});")
    f.line("fp_mul_4x2(res, src, t);")

    f.end()

def fp_inv_4x2(f, pp, v):
    f.comment("Batched inversion.")
    f.start("void fp_inv_4x2(__m512i* res, __m512i* a)")

    f.line(f"uint64_t r_64[{2*pp.num_words64}];")
    f.line(f"__m512i s0[{pp.n24}], s1[{pp.n24}], r[{pp.n24}];")
    f.line(f"__m512i t0[{pp.n42}], t1[{pp.n42}];")
    f.line("__m512i idx0, idx1;")
    f.line()

    f.comment("s0 = (a0*a1, a2*a3)")
    f.line("fp_4x2_to_2x4(s0, s1, a);")
    f.line("fp_mul_2x4(s0, s0, s1);")
    f.line()

    f.comment("r = a0a1a2a3 (wastes one mul)")
    f.line("idx1 = VSET(4, 5, 6, 7, 0, 0, 0, 0);")
    f.line(f"for (int i = 0; i < {pp.n24}; i++) {{")
    f.indent()
    f.line("s1[i] = VPERM512(idx1, s0[i]);")
    f.unindent()
    f.line("}")
    f.line("fp_mul_2x4(r, s0, s1);")
    f.line()

    f.comment("Computing r^-1 mod p")
    f.line("fp_limbs_to_64_2x4(r_64, r);")
    f.line("fp_inv_64(r_64, r_64);")
    f.line("fp_64_to_limbs_2x4(r, r_64);")
    f.line()

    f.comment("s0 = (inv_01, inv_23)")
    f.line("idx0 = VSET(4, 5, 6, 7, 0, 1, 2, 3);")
    f.line("idx1 = VSET(0, 1, 2, 3, 0, 1, 2, 3);")
    f.line(f"for (int i = 0; i < {pp.n24}; i++) {{")
    f.indent()
    f.line("s0[i] = VPERM512(idx0, s0[i]);")
    f.line("s1[i] = VPERM512(idx1, r[i]);")
    f.unindent()
    f.line("}")
    f.line("fp_mul_2x4(s0, s0, s1);")
    f.line()

    f.comment("res = (inv0, inv1, inv2, inv3)")
    f.line("fp_2x4_to_4x2(t0, s0, s0); // (inv_01, inv_01, inv_23, inv_23)")
    f.line("idx1 = VSET(2, 3, 0, 1, 6, 7, 4, 5);")
    f.line(f"for (int i = 0; i < {pp.n42}; i++) {{")
    f.indent()
    f.line("t1[i] = VPERM512(idx1, a[i]);")
    f.unindent()
    f.line("}")
    f.line("fp_mul_4x2(res, t0, t1);")

    f.end()

#TODO: Assumes fully reduced, but after mul lowest limb can be one bit larger
def fp_64_to_limbs_4x2(f, pp, v):
    f.comment(f"Assumes unvec is an array of 4 * {v.num_words64} 64-bit ints")
    f.start("void fp_64_to_limbs_4x2(__m512i* res, uint64_t* unvec)")

    f.line("uint64_t t[8];")
    f.line()

    n = pp.n42
    l = pp.limb_len
    leftover = 64
    i64 = 0
    lines = []
    for j in range(n):
        plusi64 = "" if i64 == 0 else f" + {i64}"
        if leftover == 64:
            line = f"t[2*i] = unvec[{v.num_words64} * i{plusi64}] & {v.limb_mask};"
        elif leftover >= l:
            shift = 64 - leftover
            line = f"t[2*i] = (unvec[{v.num_words64} * i{plusi64}] >> {shift}) & {v.limb_mask};"
        else: #leftover < l
            rshift = 64 - leftover
            lshift = leftover
            rmask = (1 << leftover) - 1
            lmask = (1 << (l-leftover)) - 1
            line = f"t[2*i] = ((unvec[{v.num_words64} * i{plusi64}] >> {rshift}) & 0x{rmask:x}ULL) | ((unvec[{v.num_words64} * i + {i64+1}] & 0x{lmask:x}ULL) << {lshift});"
        lines.append(line)
        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1

    top_idx = n - 1 - pp.rem42
    for j in range(n):
        f.line("for (int i = 0; i < 4; i++) {")
        f.indent()
        f.line(lines[j])

        plusi64 = "" if i64 == 0 else f" + {i64}"
        if j == top_idx + 1: #num_limbs odd
            f.line(f"t[2*i+1] = 0;")
        elif leftover == 64:
            f.line(f"t[2*i+1] = unvec[{v.num_words64} * i{plusi64}] & {v.limb_mask};")
        elif leftover >= l or j == top_idx:
           shift = 64 - leftover
           f.line(f"t[2*i+1] = (unvec[{v.num_words64} * i{plusi64}] >> {shift}) & {v.limb_mask};")
        else: #leftover < l and j < n-1
            rshift = 64 - leftover
            lshift = leftover
            rmask = (1 << leftover) - 1
            lmask = (1 << (l-leftover)) - 1
            f.line(f"t[2*i+1] = ((unvec[{v.num_words64} * i{plusi64}] >> {rshift}) & 0x{rmask:x}ULL) | ((unvec[{v.num_words64} * i + {i64+1}] & 0x{lmask:x}ULL) << {lshift});")
        
        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1
            
        f.unindent()
        f.line("}")
        f.line(f"res[{j}] = VSET(t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7]);")

    f.end()

def fp_limbs_to_64_4x2(f, pp, v):
    f.start("void fp_limbs_to_64_4x2(uint64_t* res, __m512i* src)")

    f.line("uint64_t t[8];")
    f.line()

    n = pp.n42
    l = pp.limb_len
    i64 = 0
    leftover = 64
    lines = []

    for j in range(n):
        plusi64 = "" if i64 == 0 else f" + {i64}"
        if leftover == 64:
            lines.append([f"res[{v.num_words64} * i{plusi64}] = t[2*i] & {v.limb_mask};"])
        elif leftover >= l:
            shift = 64 - leftover
            lines.append([f"res[{v.num_words64} * i{plusi64}] |= (t[2*i] & {v.limb_mask}) << {shift};"])
        else: #leftover < l and j < n-1
            lshift = 64 - leftover
            rshift = leftover
            lmask = (1 << leftover) - 1
            rmask = (1 << (l - leftover)) - 1
            # We should always do |= for the last, because l < 64
            op = "|=" if j == n-1 else "="
            lines.append([
                f"res[{v.num_words64} * i{plusi64}] |= (t[2*i] & 0x{lmask:x}ULL) << {lshift};",
                f"res[{v.num_words64} * i + {i64+1}] {op} (t[2*i] >> {rshift}) & 0x{rmask:x}ULL;"
            ])

        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1

    top_idx = n - 1 - pp.rem42
    for j in range(n):
        f.line(f"VSTORE(t, src[{j}]);")
        f.line("for (int i = 0; i < 4; i++) {")
        f.indent()
        for line in lines[j]:
            f.line(line)
        f.line()

        plusi64 = "" if i64 == 0 else f" + {i64}"
        if j == top_idx + 1:
            pass
        elif leftover == 64:
            f.line(f"res[{v.num_words64} * i{plusi64}] = t[2*i+1] & {v.limb_mask};")
        elif leftover >= l or j == top_idx:
            shift = 64 - leftover
            f.line(f"res[{v.num_words64} * i{plusi64}] |= (t[2*i+1] & {v.limb_mask}) << {shift};")
        else: #leftover < l and j < n-1
            lshift = 64 - leftover
            rshift = leftover
            lmask = (1 << leftover) - 1
            rmask = (1 << (l - leftover)) - 1
            f.line(f"res[{v.num_words64} * i{plusi64}] |= (t[2*i+1] & 0x{lmask:x}ULL) << {lshift};")
            f.line(f"res[{v.num_words64} * i + {i64+1}] = (t[2*i+1] >> {rshift}) & 0x{rmask:x}ULL;")

        leftover -= l
        if leftover <= 0:
            leftover += 64
            i64 += 1

        f.unindent()
        f.line("}")

    f.end()

def rand_fp_4x2(f, pp, v):
    f.start(f"void rand_fp_4x2(__m512i* res)")

    f.line(f"__m512i zero[{pp.n42}];")
    f.line(f"__m512i mask = VSETALL({v.limb_mask});")
    f.line(f"fp_setzero_4x2(zero);")
    f.line()

    if pp.last_limb_len == pp.limb_len:
        f.line(f"for (int i = 0; i < {pp.n42 - pp.rem42}; i++) {{")
        f.indent()
        f.line("res[i] = VAND(rand_512i(), mask);")
        f.unindent()
        f.line("}")
    else:
        f.line(f"for (int i = 0; i < {pp.n42 - pp.rem42 -1}; i++) {{")
        f.indent()
        f.line("res[i] = VAND(rand_512i(), mask);")
        f.unindent()
        f.line("}")
        f.line()

        f.comment("Last limb is shorter")
        f.line(f"mask = VSET({v.limb_mask}, {v.last_limb_mask}, {v.limb_mask}, {v.last_limb_mask}, {v.limb_mask}, {v.last_limb_mask}, {v.limb_mask}, {v.last_limb_mask});")
        f.line(f"res[{pp.n42 - pp.rem42 - 1}] = VAND(rand_512i(), mask);")
    f.line()

    if pp.rem42 == 1:
        f.line(f"mask = VSET({v.limb_mask}, 0, {v.limb_mask}, 0, {v.limb_mask}, 0, {v.limb_mask}, 0);")
        f.line(f"res[{pp.n42 - 1}] = VAND(rand_512i(), mask);")
        f.line()

    f.comment("Reduction")
    f.line("fp_add_4x2(res, res, zero);")
    f.end()


def write_fp_4x2(f, pp, v):
    f.banner("4x2")

    fp_setzero_4x2(f, pp, v)
    fp_copy_4x2(f, pp, v)
    fp_duplicate_4x2(f, pp, v)
    fp_is_equal_4x2(f, pp, v)
    fp_is_zero_4x2(f, pp, v)
    fp_add_4x2(f, pp, v)
    fp_sub_4x2(f, pp, v)
    fp_neg_4x2(f, pp, v)
    fp_mixaddsub_4x2(f, pp, v)

    mp_add_4x2(f, pp, v)
    mp_add_p_sub_4x2(f, pp, v)
    mp_mixaddsubx2_4x2(f, pp, v)
    mp_dblsubx2_4x2(f, pp, v)
    mp_subx2_4x2(f, pp, v)
    mp_mul_4x2(f, pp, v)
    rdc_4x2(f, pp, v)

    fp_mul_4x2(f, pp, v)
    fp_sqr_4x2(f, pp, v)
    fp_div3_4x2(f, pp, v)
    fp_inv_4x2(f, pp, v)

    fp_tomont_4x2(f, pp, v)
    fp_frommont_4x2(f, pp, v)

    fp_64_to_limbs_4x2(f, pp, v)
    fp_limbs_to_64_4x2(f, pp, v)

    rand_fp_4x2(f, pp, v)