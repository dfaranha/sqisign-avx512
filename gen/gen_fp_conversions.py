def fp_8x1_to_4x2(f, pp, v):
    f.comment("Given src = (a, b, c, d, e, f, g, h),")
    f.comment("it computes res1 = (a, c, e, g) and res2 = (b, d, f, h).")
    f.start("void fp_8x1_to_4x2(__m512i* res1, __m512i* res2, __m512i* src)")

    f.line("__m512i s, t;")
    f.line("const __m512i idx1 = VSET(0, 8, 2, 10, 4, 12, 6, 14);")
    f.line("const __m512i idx2 = VSET(1, 9, 3, 11, 5, 13, 7, 15);")
    f.line(f"for (int i = 0; i < {pp.n42-pp.rem42}; i++) {{")
    f.indent()
    
    f.line(f"s = src[i], t = src[i + {pp.n42}];")
    f.line("res1[i] = VBLENDPERM(idx1, s, t);")
    f.line("res2[i] = VBLENDPERM(idx2, s, t);")

    f.unindent()
    f.line("}")

    if pp.rem42 == 1:
        f.line(f"s = src[{pp.n42-1}];")
        f.line(f"res1[{pp.n42-1}] = VBLENDPERM(idx1, s, VZERO());")
        f.line(f"res2[{pp.n42-1}] = VBLENDPERM(idx2, s, VZERO());")

    f.end()

def fp_4x2_to_8x1(f, pp, v):
    f.comment("Given src1 = (a1, b1, c1, d1) and src2 = (a2, b2, c2, d2),")
    f.comment("it computes res = (a1, a2, b1, b2, c1, c2, d1, d2).")
    f.start("void fp_4x2_to_8x1(__m512i* res, __m512i* src1, __m512i* src2)")

    f.line("__m512i s, t;")
    f.line("const __m512i idx1 = VSET(0, 8, 2, 10, 4, 12, 6, 14);")
    f.line("const __m512i idx2 = VSET(1, 9, 3, 11, 5, 13, 7, 15);")
    f.line(f"for (int i = 0; i < {pp.n42-pp.rem42}; i++) {{")
    f.indent()
    
    f.line(f"s = src1[i], t = src2[i];")
    f.line("res[i] = VBLENDPERM(idx1, s, t);")
    f.line(f"res[i + {pp.n42}] = VBLENDPERM(idx2, s, t);")

    f.unindent()
    f.line("}")

    if pp.rem42 == 1:
        f.line(f"s = src1[{pp.n42-1}], t = src2[{pp.n42-1}];")
        f.line(f"res[{pp.n42-1}] = VBLENDPERM(idx1, s, t);")

    f.end()

def fp_4x2_to_2x4(f, pp, v):
    f.comment("Given src=(a, b, c, d), it computes res1 = (a, c) and res2 = (b, d).")
    f.comment("We choose this interleaving pattern for the Fp2 2x2x2 sqrt,")
    f.comment("where res1 = re and res2 = im.")
    f.start("void fp_4x2_to_2x4(__m512i* res1, __m512i* res2, __m512i* src)")

    if pp.n42 == 2 * pp.n24: #r24 = 0, 3
        fp_4x2_to_2x4_simple(f, pp, v)
    else: # n42 = 2 * n24 - 1
        fp_4x2_to_2x4_ugly(f, pp, v)

    f.end()

def fp_4x2_to_2x4_simple(f, pp, v):
    f.line("__m512i idx1 = VSET(0, 8, 1, 9, 4, 12, 5, 13);")
    f.line("__m512i idx2 = VSET(2, 10, 3, 11, 6, 14, 7, 15);")
    f.line(f"for (int i = 0; i < {pp.n24}; i++) {{")
    f.indent()

    f.line(f"__m512i s = src[i], t = src[i+{pp.n24}];")
    f.line("res1[i] = VBLENDPERM(idx1, s, t);")
    f.line("res2[i] = VBLENDPERM(idx2, s, t);")

    f.unindent()
    f.line("}")

def fp_4x2_to_2x4_ugly(f, pp, v):
    # Main pattern for i < n24-1:
        #   Lanes 0,1,4,5 (sublimb 0): VBLENDPERM from (src[i], src[i+n24])
        #   Lanes 2,3,6,7 (sublimb 1): VBLENDPERM from (src[i+1], src[i+n24+1])
        # Last vector (i = n24-1) has a different source arrangement.

        f.line("const __m512i idx_lo_ac = VSET(0, 8, 0, 0, 4, 12, 0, 0);")
        f.line("const __m512i idx_lo_bd = VSET(2, 10, 0, 0, 6, 14, 0, 0);")
        f.line("const __m512i idx_hi_ac = VSET(0, 0, 1, 9, 0, 0, 5, 13);")
        f.line("const __m512i idx_hi_bd = VSET(0, 0, 3, 11, 0, 0, 7, 15);")
        f.comment("For the single-source hi (lanes 2,6 only)")
        f.line("const __m512i perm_ac = VSET(0, 0, 1, 0, 0, 0, 5, 0);")
        f.line("const __m512i perm_bd = VSET(0, 0, 3, 0, 0, 0, 7, 0);")
        f.comment("For the last vector (src[n24-1] paired with src[0])")
        f.line(f"const __m512i idx_last_ac = VSET(0, 9, 0, 0, 4, 13, 0, 0);")
        f.line(f"const __m512i idx_last_bd = VSET(2, 11, 0, 0, 6, 15, 0, 0);")
        f.line(f"__m512i t[{pp.n42}], lo, hi;")
        f.line()

        for i in range(pp.n42):
            f.line(f"t[{i}] = src[{i}];")
        f.line()

        # Main loop: i = 0..n24-3 (all 4 source vectors available)
        for i in range(pp.n24 - 2):
            if i + pp.n24 + 1 < pp.n42:
                f.line(f"lo = VBLENDPERM(idx_lo_ac, t[{i}], t[{i+pp.n24}]);")
                f.line(f"hi = VBLENDPERM(idx_hi_ac, t[{i+1}], t[{i+pp.n24+1}]);")
                f.line(f"res1[{i}] = VBLEND(0xcc, lo, hi);")
                f.line(f"lo = VBLENDPERM(idx_lo_bd, t[{i}], t[{i+pp.n24}]);")
                f.line(f"hi = VBLENDPERM(idx_hi_bd, t[{i+1}], t[{i+pp.n24+1}]);")
                f.line(f"res2[{i}] = VBLEND(0xcc, lo, hi);")
            else:
                # t[i+n24+1] out of bounds, lane 3,7 padding
                f.line(f"lo = VBLENDPERM(idx_lo_ac, t[{i}], t[{i+pp.n24}]);")
                f.line(f"hi = VPERM512MASKZ(0x44, t[{i+1}], perm_ac);")
                f.line(f"res1[{i}] = VBLEND(0xcc, lo, hi);")
                f.line(f"lo = VBLENDPERM(idx_lo_bd, t[{i}], t[{i+pp.n24}]);")
                f.line(f"hi = VPERM512MASKZ(0x44, t[{i+1}], perm_bd);")
                f.line(f"res2[{i}] = VBLEND(0xcc, lo, hi);")
            f.line()

        # i = n24-2: src[i+n24+1] = src[2*n24-1] = src[n42] out of bounds
        i = pp.n24 - 2
        f.line(f"lo = VBLENDPERM(idx_lo_ac, t[{i}], t[{i+pp.n24}]);")
        f.line(f"hi = VPERM512MASKZ(0x44, t[{i+1}], perm_ac);")
        f.line(f"res1[{i}] = VBLEND(0xcc, lo, hi);")
        f.line(f"lo = VBLENDPERM(idx_lo_bd, t[{i}], t[{i+pp.n24}]);")
        f.line(f"hi = VPERM512MASKZ(0x44, t[{i+1}], perm_bd);")
        f.line(f"res2[{i}] = VBLEND(0xcc, lo, hi);")
        f.line()

        # i = n24-1: different source pairing (src[n24-1] with src[0])
        i = pp.n24 - 1
        f.line(f"lo = VBLENDPERM(idx_last_ac, t[{pp.n24-1}], t[0]);")
        f.line(f"hi = VPERM512MASKZ(0x44, t[{pp.n24}], perm_ac);")
        f.line(f"res1[{i}] = VBLEND(0xcc, lo, hi);")
        f.line(f"lo = VBLENDPERM(idx_last_bd, t[{pp.n24-1}], t[0]);")
        f.line(f"hi = VPERM512MASKZ(0x44, t[{pp.n24}], perm_bd);")
        f.line(f"res2[{i}] = VBLEND(0xcc, lo, hi);")


def fp_2x4_to_4x2(f, pp, v):
    f.start("void fp_2x4_to_4x2(__m512i* res, __m512i* src1, __m512i* src2)")

    if pp.n42 == 2 * pp.n24: #r24 = 0, 3
        fp_2x4_to_4x2_simple(f, pp, v)
    else: # n42 = 2 * n24 - 1
        fp_2x4_to_4x2_ugly(f, pp, v)

    f.end()

def fp_2x4_to_4x2_simple(f, pp, v):
    f.line("__m512i idx1 = VSET(0, 2, 8, 10, 4, 6, 12, 14);")
    f.line("__m512i idx2 = VSET(1, 3, 9, 11, 5, 7, 13, 15);")
    f.line(f"for (int i = 0; i < {pp.n24}; i++) {{")
    f.indent()

    f.line(f"__m512i s = src1[i], t = src2[i];")
    f.line("res[i] = VBLENDPERM(idx1, s, t);")
    f.line(f"res[i+{pp.n24}] = VBLENDPERM(idx2, s, t);")

    f.unindent()
    f.line("}")

def fp_2x4_to_4x2_ugly(f, pp, v):
    # For each output res[j]:
    #   even lanes (0,2,4,6): sublimb s0 = j // n24, from src1[j%n24], src2[j%n24]
    #   odd  lanes (1,3,5,7): sublimb s1 = (j+n42) // n24, from src1[(j+n42)%n24], src2[(j+n42)%n24]
    # idx_s = VSET(s, s, 8+s, 8+s, 4+s, 4+s, 12+s, 12+s)

    max_sub = max((j + pp.n42) // pp.n24 for j in range(pp.n42))
    for s in range(max_sub + 1):
        vals = [s, s, s+8, s+8, s+4, s+4, s+12, s+12]
        f.line(f"const __m512i idx_{s} = VSET({', '.join(str(x) for x in vals)});")
    f.line(f"__m512i t1[{pp.n24}], t2[{pp.n24}], even, odd;")
    f.line()

    for i in range(pp.n24):
        f.line(f"t1[{i}] = src1[{i}], t2[{i}] = src2[{i}];")
    f.line()

    for j in range(pp.n42):
        s0 = j // pp.n24
        m0 = j % pp.n24
        s1 = (j + pp.n42) // pp.n24
        m1 = (j + pp.n42) % pp.n24
        
        f.line(f"even = VBLENDPERM(idx_{s0}, t1[{m0}], t2[{m0}]);")
        f.line(f"odd  = VBLENDPERM(idx_{s1}, t1[{m1}], t2[{m1}]);")
        f.line(f"res[{j}] = VBLEND(0xaa, even, odd);")
        if j < pp.n42 - 1:
            f.line()

def write_fp_conversions(f, pp, v):
    f.banner("Conversions between Fp representations")

    fp_8x1_to_4x2(f, pp, v)
    fp_4x2_to_8x1(f, pp, v)

    fp_4x2_to_2x4(f, pp, v)
    fp_2x4_to_4x2(f, pp, v)