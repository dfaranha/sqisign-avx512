from sage.all import GF
from file_writer import FileWriter

def write_fp2_kat_include(f, pp, v):
    f.line("#include \"fp_consts.h\"")
    f.line("#include \"intrinsics_avx512.h\"")
    f.line()

def write_fp2_kat_file(dir, pp, v):
    
    path = f"{dir}/fp2_kat.h"
    f = FileWriter(path)

    Fp = GF(pp.p)

    write_fp2_kat_include(f, pp, v)