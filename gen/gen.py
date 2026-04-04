from gen_fp import write_fp_file
from gen_fp_test import write_fp_test_file


# TODO:
#   Move carry to rdc_8x1
#   when 2 * last_limb_len <= limb_len, we can save one limb in the unreduced product.
#   Add fp_div3_8x1
#   Test odd karatsuba
#   Test 4x2 odd limbs. Is the carry pattern safe if I skip the last limb?
#   Find out how to generate inversion code?
#   8x1 <-> 4x2 functions
#   Optimize 2x4 when num_limbs % 4 != 0

### input ###

name = "lvl3"
e = 376
c = 65

# name = "lvl5"
# e = 500
# c = 27

# name = "noshift"
# e = 2601
# c = 609

# name = "mersenne"
# e = 521
# c = 1

#############


write_fp_file(name, e, c)
#write_fp_test_file(name, e, c)