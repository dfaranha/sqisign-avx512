from file_writer import (FileWriter, int_to_limbs)
from params import (get_limb_mask)

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

    p_plus_1_limbs = int_to_limbs(pp.p+1, pp.limb_len)
    f.line(f"#define {v.p_plus_1_top} 0x{p_plus_1_limbs[-1]:x}ULL")
    f.line()
    
    f.comment(f"Montgomery constant {v.R} = 2^({pp.num_limbs+1}*{pp.limb_len}) mod p")
    f.limb_array_1d(pp.R, pp.num_limbs, pp.limb_len, v.R, v.num_limbs)
    f.comment(f"{v.R2} = {v.R}^2 mod p")
    f.limb_array_1d(pp.R2, pp.num_limbs, pp.limb_len, v.R2, v.num_limbs)

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