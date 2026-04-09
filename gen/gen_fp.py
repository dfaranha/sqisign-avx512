# Auto-generates the vectorized Fp arithmetic, for a generalized Mersenne prime p = c* 2^e -1, where c is a small odd cofactor.

from file_writer import FileWriter
from gen_fp_8x1 import write_fp_8x1
from gen_fp_4x2 import write_fp_4x2
from gen_fp_2x4 import write_fp_2x4
from gen_fp_inv import write_fp_inv_functions

def write_fp_include(f):
    f.line("#include <stdio.h>")
    f.line("#include <immintrin.h>")
    f.line()
    f.line("#include \"fp.h\"")
    f.line("#include \"fp_consts.h\"")
    f.line("#include \"utils.h\"")
    f.line()

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

#TODO: Guaranteed to fail when r24 = 1,2
def fp_4x2_to_2x4(f, pp, v):
    f.comment("Given src=(a, b, c, d), it computes res1 = (a, c) and res2 = (b, d).")
    f.comment("We choose this interleaving pattern for the Fp2 2x2x2 sqrt,")
    f.comment("where res1 = re and res2 = im.")
    f.start("void fp_4x2_to_2x4(__m512i* res1, __m512i* res2, __m512i* src)")

    f.line("__m512i idx1 = VSET(0, 8, 1, 9, 4, 12, 5, 13);")
    f.line("__m512i idx2 = VSET(2, 10, 3, 11, 6, 14, 7, 15);")
    f.line(f"for (int i = 0; i < {pp.n24}; i++) {{")
    f.indent()

    f.line(f"__m512i s = src[i], t = src[i+{pp.n24}];")
    f.line("res1[i] = VBLENDPERM(idx1, s, t);")
    f.line("res2[i] = VBLENDPERM(idx2, s, t);")

    f.unindent()
    f.line("}")

    f.end()

#TODO: Guaranteed to fail when r24 = 1,2
def fp_2x4_to_4x2(f, pp, v):
    f.start("void fp_2x4_to_4x2(__m512i* res, __m512i* src1, __m512i* src2)")

    f.line("__m512i idx1 = VSET(0, 2, 8, 10, 4, 6, 12, 14);")
    f.line("__m512i idx2 = VSET(1, 3, 9, 11, 5, 7, 13, 15);")
    f.line(f"for (int i = 0; i < {pp.n24}; i++) {{")
    f.indent()

    f.line(f"__m512i s = src1[i], t = src2[i];")
    f.line("res[i] = VBLENDPERM(idx1, s, t);")
    f.line(f"res[i+{pp.n24}] = VBLENDPERM(idx2, s, t);")

    f.unindent()
    f.line("}")

    f.end()

def fp_conversions(f, pp, v):
    f.banner("Conversions between Fp representations")

    fp_8x1_to_4x2(f, pp, v)
    fp_4x2_to_8x1(f, pp, v)

    fp_4x2_to_2x4(f, pp, v)
    fp_2x4_to_4x2(f, pp, v)

def write_fp_file(dir, pp, v, strat):
    path = f"{dir}/fp.c"
    f = FileWriter(path)
    
    write_fp_include(f)
    write_fp_8x1(f, pp, v, strat)
    write_fp_4x2(f, pp, v, strat)
    write_fp_2x4(f, pp, v, strat)
    fp_conversions(f, pp, v)
    write_fp_inv_functions(f, pp, v)
