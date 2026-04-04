# Auto-generates the vectorized Fp arithmetic, for a generalized Mersenne prime p = c* 2^e -1, where c is a small odd cofactor.

from file_writer import FileWriter
from params import (get_fp_params, get_param_names)
from gen_fp_8x1 import write_fp_8x1
from gen_fp_4x2 import write_fp_4x2
from gen_fp_2x4 import write_fp_2x4

def get_limb_mask(pp):
    limb_mask = (1 << pp.limb_len) - 1
    last_limb_mask = (1 << pp.last_limb_len) - 1
    return limb_mask, last_limb_mask

def int_to_limbs(n, pp):
    if n == 0:
        return [0]
    limbs = []
    val = n
    i = 0
    while val > 0:
        # Mask out the bottom bits and append
        limbs.append((val & ((1 << pp.limb_len) - 1)))
        val >>= pp.limb_len
        i += 1
    return limbs

def write_fp_include(f):
    f.line("#include <stdio.h>")
    f.line("#include <stdint.h>")
    f.line()
    f.line("#include \"fp_lvl3.h\"")
    f.line("#include \"intrinsics_avx512.h\"")
    f.line("#include \"utils.h\"")
    f.line()

# Converts the integer n to a limb array, then writes it as a uint64_t array
def write_const_array(f, n, arr_name, pp, v):
    limbs = int_to_limbs(n, pp)
    limbs += (pp.num_limbs-len(limbs)) * [0]

    hex_limbs = [f"0x{l:x}" for l in limbs]
    
    f.line(f"static const uint64_t {arr_name}[{v.num_limbs}] = {{")

    limbs_per_line = 4
    f.indent()
    for i in range(0, pp.num_limbs, limbs_per_line):
        chunk = hex_limbs[i : i + limbs_per_line]
        line = ", ".join(chunk)
        
        suffix = "," if i + limbs_per_line < pp.num_limbs else ""
        f.line(f"{line}{suffix}")

    f.unindent()
    f.line("};")
    f.line()


def write_fp_consts(f, pp, v):
    f.line(f"#define {v.limb_len} {pp.limb_len}")
    f.line(f"#define {v.num_limbs} {pp.num_limbs}")
    f.line(f"#define {v.limb_len} 52 - {v.limb_len}")
    f.line(f"#define {v.num_limbs64} {pp.num_limbs64}")
    f.line()

    f.line(f"#define {v.n42} {pp.n42}")
    f.line(f"#define {v.n24} {pp.n24}")
    f.line()

    limb_mask, last_limb_mask = get_limb_mask(pp)
    f.line(f"#define {v.limb_mask} 0x{limb_mask:x}ULL")
    f.line(f"#define {v.last_limb_mask} 0x{last_limb_mask:x}ULL")
    f.line()

    p_plus_1_limbs = int_to_limbs(pp.p+1, pp)
    f.line(f"#define {v.p_plus_1_top} 0x{p_plus_1_limbs[-1]:x}ULL")
    f.line()
    
    f.comment(f"Montgomery constant {v.R} = 2^({pp.num_limbs+1}*{pp.limb_len}) mod p")
    write_const_array(f, pp.R, v.R, pp, v)
    f.comment(f"{v.R2} = {v.R}^2 mod p")
    write_const_array(f, pp.R2, v.R2, pp, v)

def write_fp_file(name, e, c):
    pp = get_fp_params(e, c)
    v = get_param_names()

    file_name = f"fp_{name}.c"
    f = FileWriter(file_name)
    
    write_fp_include(f)
    write_fp_consts(f, pp, v)
    write_fp_8x1(f, pp, v)
    write_fp_4x2(f, pp, v)
    write_fp_2x4(f, pp, v)