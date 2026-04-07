from file_writer import (FileWriter, int_to_limbs)
from params import (get_limb_mask, get_inv_params)

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
    f.line()
    
    f.comment(f"Montgomery constant {v.R} = 2^({pp.num_limbs+1}*{pp.limb_len}) mod p")
    f.limb_array_1d(pp.R, pp.num_limbs, pp.limb_len, v.R, v.num_limbs)
    f.comment(f"{v.R2} = {v.R}^2 mod p")
    f.limb_array_1d(pp.R2, pp.num_limbs, pp.limb_len, v.R2, v.num_limbs)
    f.comment("3^-1 * R mod p")
    inv3 = (pow(3, -1, pp.p) * pp.R) % pp.p
    f.limb_array_1d(inv3, pp.num_limbs, pp.limb_len, v.inv3, v.num_limbs)

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

    f.line("#ifndef _FP_CONSTS_H")
    f.line("#define _FP_CONSTS_H")
    f.line()

    f.line("#include <stdint.h>")
    f.line()

    write_fp_consts(f, pp, v)

    f.line("#endif")