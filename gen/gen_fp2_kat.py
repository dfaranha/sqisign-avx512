from file_writer import FileWriter
import random

def write_fp2_kat_include(f, pp, v):
    f.line("#include \"fp_consts.h\"")
    f.line("#include \"intrinsics_avx512.h\"")
    f.line()

def get_rand_Fp2_vec(p, n):
    return [(random.randrange(0, p), random.randrange(0, p)) for _ in range(n)]

def gen_test_case(p, n, num_operands, op):
    assert num_operands <= 2

    a = get_rand_Fp2_vec(p, n)
    if num_operands == 1:
        c = [op(a[i]) for i in range(n)]
        return a, c
    if num_operands == 2:
        b = get_rand_Fp2_vec(p, n)
        c = [op(a[i], b[i]) for i in range(n)]
        return a, b, c

def gen_test_cases(f, pp, v, n, op, op_name, format):
    num_operands = op.__code__.co_argcount
    assert num_operands <= 2
    A = []
    B = []
    C = []

    for _ in range(pp.num_kat):
        if num_operands == 1:
            a, c = gen_test_case(pp.p, n, 1, op)
        else:
            a, b, c = gen_test_case(pp.p, n, 2, op)
            B.append(b)
        A.append(a)
        C.append(c)
    
    f.fp2_array_2d(A, pp, v, format, arr_name=f"a_{op_name}_{format}", size1_name = v.num_kat_fp2)
    if num_operands == 2:
        f.fp2_array_2d(B, pp, v, format, arr_name=f"b_{op_name}_{format}", size1_name = v.num_kat_fp2)
    f.fp2_array_2d(C, pp, v, format, arr_name=f"c_{op_name}_{format}", size1_name = v.num_kat_fp2)

def fp2_kat(f, pp, v, n, format, R_inv):
    f.banner(format)

    f.comment("Addition")
    op = lambda x,y: ((x[0] + y[0]) % pp.p, (x[1] + y[1]) % pp.p)
    gen_test_cases(f, pp, v, n, op, "add", format)

    f.comment("Subtraction")
    op = lambda x,y: ((x[0] - y[0]) % pp.p, (x[1] - y[1]) % pp.p)
    gen_test_cases(f, pp, v, n, op, "sub", format)

    f.comment("Multiplication")
    op = lambda x,y: (((x[0]*y[0] - x[1]*y[1])*R_inv) % pp.p, ((x[0]*y[1] + x[1]*y[0])*R_inv) % pp.p)
    gen_test_cases(f, pp, v, n, op, "mul", format)

    f.comment("Squaring")
    op = lambda x: (((x[0]**2 - x[1]**2)*R_inv) % pp.p, (2*(x[0]*x[1])*R_inv) % pp.p)
    gen_test_cases(f, pp, v, n, op, "sqr", format)

def write_fp2_kat_file(dir, pp, v):
    path = f"{dir}/fp2_kat.h"
    f = FileWriter(path)
    print(f"Writing {path}")

    write_fp2_kat_include(f, pp, v)

    f.line(f"#define {v.num_kat_fp2} {pp.num_kat}")
    f.line()

    R_inv = pow(pp.R, -1, pp.p) #precomp
    fp2_kat(f, pp, v, 8, "8x1x1", R_inv)
    fp2_kat(f, pp, v, 4, "4x1x2", R_inv)
    fp2_kat(f, pp, v, 4, "4x2x1", R_inv)
    fp2_kat(f, pp, v, 2, "2x2x2", R_inv)
    fp2_kat(f, pp, v, 2, "2x4x1", R_inv)