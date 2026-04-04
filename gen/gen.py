import os
from gen_fp import write_fp_file
from gen_fp_header import write_fp_header_file
from gen_fp_test import write_fp_test_file
from params import (get_fp_params, get_param_names)


# TODO:
#   Move carry to rdc_8x1
#   when 2 * last_limb_len <= limb_len, we can save one limb in the unreduced product.
#   Add fp_div3_8x1
#   Test odd karatsuba
#   Test 4x2 odd limbs. Is the carry pattern safe if I skip the last limb?
#   Find out how to generate inversion code?
#   8x1 <-> 4x2 functions
#   Optimize 2x4 when num_limbs % 4 != 0
#   Optimize with addition chain for pow3div4
#   inv

### input ###
dir = "out"

# lvl3
e = 376
c = 65

# lvl5
# e = 500
# c = 27

# no shift
# e = 2601
# c = 609

# mersenne
# e = 521
# c = 1

#############

pp = get_fp_params(e, c)
v = get_param_names()

if not os.path.exists(dir):
    os.mkdir(dir)

write_fp_header_file(dir, pp, v)
write_fp_file(dir, pp, v)
write_fp_test_file(dir, pp, v)