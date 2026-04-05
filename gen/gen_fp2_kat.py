from sage.all import (GF, PolynomialRing)
from file_writer import FileWriter

def write_fp2_kat_include(f, pp, v):
    f.line("#include \"fp_consts.h\"")
    f.line("#include \"intrinsics_avx512.h\"")
    f.line()

def get_rand_Fp2_vec(Fp2, n):
    return [Fp2.random_element() for _ in range(n)]

def vec_to_int(vec):
    return [[ai.lift() for ai in a.polynomial().list()] for a in vec]

def gen_test_case(Fp2, n, num_operands, op):
    assert num_operands <= 2

    a = get_rand_Fp2_vec(Fp2, n)
    if num_operands == 1:
        c = [op(a[i]) for i in range(n)]
        return vec_to_int(a), vec_to_int(c)
    if num_operands == 2:
        b = get_rand_Fp2_vec(Fp2, n)
        c = [op(a[i], b[i]) for i in range(n)]
        return vec_to_int(a), vec_to_int(b), vec_to_int(c)

def gen_test_cases(f, pp, v, Fp2, n, op, op_name, format):
    num_operands = op.__code__.co_argcount
    assert num_operands <= 2
    A = []
    B = []
    C = []

    for _ in range(pp.num_kat):
        if num_operands == 1:
            a, c = gen_test_case(Fp2, n, 1, op)
        else:
            a, b, c = gen_test_case(Fp2, n, 2, op)
            B.append(b)
        A.append(a)
        C.append(c)
    
    f.fp2_array_2d(A, pp, v, format, arr_name=f"a_{op_name}_{format}", size1_name = v.num_kat_fp2)
    if num_operands == 2:
        f.fp2_array_2d(B, pp, v, format, arr_name=f"b_{op_name}_{format}", size1_name = v.num_kat_fp2)
    f.fp2_array_2d(C, pp, v, format, arr_name=f"c_{op_name}_{format}", size1_name = v.num_kat_fp2)

def fp2_kat(f, pp, v, Fp, Fp2, n, format):
    f.banner(format)

    f.comment("Addition")
    op = lambda x,y: x+y
    gen_test_cases(f, pp, v, Fp2, n, op, "add", format)

    f.comment("Subtraction")
    op = lambda x,y: x-y
    gen_test_cases(f, pp, v, Fp2, n, op, "sub", format)

    f.comment("Multiplication")
    op = lambda x,y: x*y/Fp(pp.R)
    gen_test_cases(f, pp, v, Fp2, n, op, "mul", format)

    f.comment("Squaring")
    op = lambda x: x**2 / Fp(pp.R)
    gen_test_cases(f, pp, v, Fp2, n, op, "sqr", format)

def write_fp2_kat_file(dir, pp, v):
    path = f"{dir}/fp2_kat.h"
    f = FileWriter(path)

    Fp = GF(pp.p)
    R = PolynomialRing(Fp, 'x')
    x = R.gen()
    Fp2 = GF(pp.p**2, name = 'i', modulus=x**2+1)

    write_fp2_kat_include(f, pp, v)

    f.line(f"#define {v.num_kat_fp2} {pp.num_kat}")
    f.line()

    fp2_kat(f, pp, v, Fp, Fp2, 8, "8x1x1")
    fp2_kat(f, pp, v, Fp, Fp2, 4, "4x1x2")
    fp2_kat(f, pp, v, Fp, Fp2, 4, "4x2x1")
    fp2_kat(f, pp, v, Fp, Fp2, 2, "2x2x2")
    fp2_kat(f, pp, v, Fp, Fp2, 2, "2x4x1")