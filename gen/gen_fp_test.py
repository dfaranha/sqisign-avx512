from sage.all import GF
from file_writer import FileWriter

def write_test_include(f, pp, v):
    f.line("#include <stdbool.h>")
    f.line("#include <stdio.h>")
    f.line("#include <stdlib.h>")
    f.line("#include <string.h>")
    f.line("#include <time.h>")
    f.line()
    f.line("#include \"fp.h\"")
    f.line("#include \"fp_consts.h\"")
    f.line("#include \"test.h\"")
    f.line("#include \"utils.h\"")
    f.line()

def fp_8x1_tests(f, pp, v, Fp):
    pass

def fp_4x2_tests(f, pp, v, Fp):
    pass

def fp_2x4_tests(f, pp, v, Fp):
    pass

def fp_test_main(f, pp, v, Fp): # should be executable with a main
    f.start("int main()")

    f.line("bool passed;")
    f.line()

    f.comment("Initialize randomness")
    f.line("time_t t;")
    f.line("srand((unsigned) time(&t));")
    f.line()

    f.line("passed = fp_8x1_tests();")
    f.line("if(!passed) { return 0; }")
    f.line()

    f.line("passed = fp_4x2_tests();")
    f.line("if(!passed) { return 0; }")
    f.line()

    f.line("passed = fp_2x4_tests();")
    f.line("if(!passed) { return 0; }")

    f.end()

def write_fp_test_file(dir, pp, v):
    
    path = f"{dir}/fp_test.c"
    f = FileWriter(path)

    Fp = GF(pp.p)

    write_test_include(f, pp, v)
    fp_8x1_tests(f, pp, v, Fp)
    fp_4x2_tests(f, pp, v, Fp)
    fp_2x4_tests(f, pp, v, Fp)
    fp_test_main(f, pp, v, Fp)