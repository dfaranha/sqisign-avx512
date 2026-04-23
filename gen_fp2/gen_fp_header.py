from file_writer import (FileWriter, int_to_limbs)
from params import (get_limb_mask, get_inv_params, get_div2_consts)

def write_fp_consts(f, pp, v):
    f.line(f"#define {v.limb_len} {pp.limb_len}")
    f.line(f"#define {v.num_limbs} {pp.num_limbs}")
    f.line(f"#define {v.limb_align} 52 - {v.limb_len}")
    f.line(f"#define {v.num_words64} {pp.num_words64}")
    f.line()

    f.line(f"#define {v.n42} {pp.n42}")
    f.line(f"#define {v.n24} {pp.n24}")
    f.line()

    limb_mask, last_limb_mask = get_limb_mask(pp)
    f.line(f"#define {v.limb_mask} 0x{limb_mask:x}ULL")
    f.line(f"#define {v.last_limb_mask} 0x{last_limb_mask:x}ULL")
    f.line()

    p_plus_1_limbs = int_to_limbs(pp.p+1, pp.limb_len)
    f.line(f"#define {v.p_plus_1_top} 0x{p_plus_1_limbs[-1]:x}ULL")

    lo, hi = get_div2_consts(pp.e, pp.c, pp.num_limbs, pp.limb_len)
    if hi is None:
        f.line(f"#define {v.p_plus_1_div2_top} 0x{lo:x}ULL")
    else:
        f.line(f"#define {v.p_plus_1_div2_top} 0x{lo:x}ULL")
        f.line(f"#define {v.p_plus_1_div2_top_hi} 0x{hi:x}ULL")
    f.line()
    
    f.comment(f"Montgomery constant {v.R} = 2^({pp.num_limbs}*{pp.limb_len}) mod p")
    f.limb_array_1d(pp.R, pp.num_limbs, pp.limb_len, v.R, v.num_limbs)
    f.comment(f"{v.R2} = {v.R}^2 mod p")
    f.limb_array_1d(pp.R2, pp.num_limbs, pp.limb_len, v.R2, v.num_limbs)

    f.comment("3^-1 * R mod p")
    inv3 = (pow(3, -1, pp.p) * pp.R) % pp.p
    f.limb_array_1d(inv3, pp.num_limbs, pp.limb_len, v.inv3, v.num_limbs)

    f.comment("Cofactor c of p = c*2^e - 1")
    c_nwords64 = (pp.c.bit_length()+63)//64
    f.line(f"#define {v.c_nbits} {pp.c.bit_length()}")
    f.line(f"#define {v.c_nwords64} {c_nwords64}")
    f.limb_array_1d(pp.c, c_nwords64, 64, v.c, v.c_nwords64)

    f.comment("p in 64-bit repr.")
    f.limb_array_1d(pp.p, pp.num_words64, 64, v.p_64, v.num_words64)
    f.comment("R in 64-bit repr.")
    f.limb_array_1d(pp.R, pp.num_words64, 64, v.R_64, v.num_words64)
    f.comment("Constant used to correct output of fp_inv64")
    _, _, C = get_inv_params(pp)
    f.limb_array_1d(C, pp.num_words64, 64, v.inv_const_64, v.num_words64)

def write_fp_header_file(dir, pp, v):
    path = f"{dir}/fp_consts.h"
    f = FileWriter(path)
    print(f"Writing {path}")

    f.line("#ifndef _FP_CONSTS_H")
    f.line("#define _FP_CONSTS_H")
    f.line()

    f.line("#include <stdint.h>")
    f.line()

    write_fp_consts(f, pp, v)

    f.line("#endif")