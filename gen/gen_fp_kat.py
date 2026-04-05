from sage.all import GF
from file_writer import FileWriter

def write_fp_kat_include(f, pp, v):
    f.line("#include \"fp_consts.h\"")
    f.line("#include \"intrinsics_avx512.h\"")
    f.line()

def get_rand_Fp_vec(Fp, n):
    return [Fp.random_element() for _ in range(n)]

def vec_to_int(vec):
    return [a.lift() for a in vec]

def gen_test_case(Fp, n, num_operands, op):
    assert num_operands <= 2

    a = get_rand_Fp_vec(Fp, n)
    if num_operands == 1:
        c = [op(a[i]) for i in range(n)]
        return vec_to_int(a), vec_to_int(c)
    if num_operands == 2:
        b = get_rand_Fp_vec(Fp, n)
        c = [op(a[i], b[i]) for i in range(n)]
        return vec_to_int(a), vec_to_int(b), vec_to_int(c)
    
def gen_test_cases(f, pp, v, Fp, n, op, op_name, format):
    num_operands = op.__code__.co_argcount
    assert num_operands <= 2
    A = []
    B = []
    C = []

    for i in range(pp.num_kat):
        if num_operands == 1:
            a, c = gen_test_case(Fp, n, 1, op)
        else:
            a, b, c = gen_test_case(Fp, n, 2, op)
            B.append(b)
        A.append(a)
        C.append(c)

    if n == 8:
        size2_name = v.num_limbs
    elif n == 4:
        size2_name = v.n42
    else: # n == 2
        size2_name = v.n24
    
    f.vec_array_2d(A, pp.num_limbs, pp.limb_len, f"a_{op_name}_{format}", size1_name = v.num_kat, size2_name = size2_name)
    if num_operands == 2:
        f.vec_array_2d(B, pp.num_limbs, pp.limb_len, f"b_{op_name}_{format}", size1_name = v.num_kat, size2_name = size2_name)
    f.vec_array_2d(C, pp.num_limbs, pp.limb_len, f"c_{op_name}_{format}", size1_name = v.num_kat, size2_name = size2_name)

def fp_kat(f, pp, v, Fp, n, format):
    f.banner(format)

    f.comment("Addition")
    op = lambda x,y: x+y
    gen_test_cases(f, pp, v, Fp, n, op, "add", format)

    f.comment("Subtraction")
    op = lambda x,y: x-y
    gen_test_cases(f, pp, v, Fp, n, op, "sub", format)

    f.comment("Multiplication")
    op = lambda x,y: x*y
    gen_test_cases(f, pp, v, Fp, n, op, "mul", format)


def write_fp_kat_file(dir, pp, v):
    
    path = f"{dir}/fp_kat.h"
    f = FileWriter(path)

    Fp = GF(pp.p)

    write_fp_kat_include(f, pp, v)

    f.line(f"#define {v.num_kat} {pp.num_kat}")
    f.line()

    fp_kat(f, pp, v, Fp, 8, "8x1")
    fp_kat(f, pp, v, Fp, 4, "4x2")
    fp_kat(f, pp, v, Fp, 2, "2x4")