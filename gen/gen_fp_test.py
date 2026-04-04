from sage.all import GF
from file_writer import FileWriter
from params import (get_fp_params, get_param_names)
from gen_fp import int_to_limbs

def fp_8x1_tests(f, pp, v, Fp):
    pass

def write_fp_test_file(name, e, c):
    pp = get_fp_params(e, c)
    v = get_param_names()
    
    file_name = f"fp_{name}_test.c"
    f = FileWriter(file_name)

    Fp = GF(pp.p)

    fp_8x1_tests(f, pp, v, Fp)