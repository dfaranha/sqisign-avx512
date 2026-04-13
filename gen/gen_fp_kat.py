from file_writer import FileWriter
import random

def write_fp_kat_include(f, pp, v):
    f.line("#include \"fp_consts.h\"")
    f.line("#include \"intrinsics_avx512.h\"")
    f.line()

def get_rand_Fp_vec(p, n):
    return [random.randrange(0, p) for _ in range(n)]


def gen_test_case(p, n, num_operands, op):
    assert num_operands <= 2

    a = get_rand_Fp_vec(p, n)
    if num_operands == 1:
        c = [op(a[i]) for i in range(n)]
        return a, c
    if num_operands == 2:
        b = get_rand_Fp_vec(p, n)
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
    
    f.fp_array_2d(A, pp, v, f"a_{op_name}_{format}", size1_name = v.num_kat_fp)
    if num_operands == 2:
        f.fp_array_2d(B, pp, v, f"b_{op_name}_{format}", size1_name = v.num_kat_fp)
    f.fp_array_2d(C, pp, v, f"c_{op_name}_{format}", size1_name = v.num_kat_fp)

def fp_kat(f, pp, v, n, format, R_inv):
    f.banner(format)

    f.comment("Addition")
    op = lambda x,y: (x+y) % pp.p
    gen_test_cases(f, pp, v, n, op, "add", format)

    f.comment("Subtraction")
    op = lambda x,y: (x-y) % pp.p
    gen_test_cases(f, pp, v, n, op, "sub", format)

    f.comment("Multiplication")
    op = lambda x,y: (x*y * R_inv) % pp.p
    gen_test_cases(f, pp, v, n, op, "mul", format)

def write_fp_kat_file(dir, pp, v):
    
    path = f"{dir}/fp_kat.h"
    f = FileWriter(path)
    print(f"Writing {path}")

    write_fp_kat_include(f, pp, v)

    f.line(f"#define {v.num_kat_fp} {pp.num_kat}")
    f.line()

    R_inv = pow(pp.R, -1, pp.p) #precomp
    fp_kat(f, pp, v, 8, "8x1", R_inv)
    fp_kat(f, pp, v, 4, "4x2", R_inv)
    fp_kat(f, pp, v, 2, "2x4", R_inv)