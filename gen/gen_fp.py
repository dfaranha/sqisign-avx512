# Auto-generates the vectorized Fp arithmetic, for a generalized Mersenne prime p = c* 2^e -1, where c is a small odd cofactor.

from file_writer import FileWriter
from gen_fp_8x1 import write_fp_8x1
from gen_fp_4x2 import write_fp_4x2
from gen_fp_2x4 import write_fp_2x4
from gen_fp_conversions import write_fp_conversions
from gen_fp_inv import write_fp_inv_functions

def write_fp_include(f):
    f.line("#include <stdio.h>")
    f.line("#include <immintrin.h>")
    f.line()
    f.line("#include \"fp.h\"")
    f.line("#include \"fp_consts.h\"")
    f.line("#include \"utils.h\"")
    f.line()

def write_fp_file(dir, pp, v, strat, chain):
    path = f"{dir}/fp.c"
    f = FileWriter(path)
    print(f"Writing {path}")

    write_fp_include(f)
    write_fp_8x1(f, pp, v, strat, chain)
    write_fp_4x2(f, pp, v, strat, chain)
    write_fp_2x4(f, pp, v, strat, chain)
    write_fp_conversions(f, pp, v)
    write_fp_inv_functions(f, pp, v)
