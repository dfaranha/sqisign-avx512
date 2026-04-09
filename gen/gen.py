import os
from gen_fp import write_fp_file
from gen_fp2 import write_fp2_file
from gen_fp_header import write_fp_header_file
from gen_fp_kat import write_fp_kat_file
from gen_fp2_kat import write_fp2_kat_file
from params import (comp_params_and_strat, get_param_names)


# TODO:
#   Move carry to rdc_8x1
#   when 2 * last_limb_len <= limb_len, we can save one limb in the unreduced product.
#   Test odd karatsuba
#   Test 4x2 odd limbs. Is the carry pattern safe if I skip the last limb?
#   8x1 <-> 4x2 functions
#   unvectorized <-> vec (just convert to uint64_t array, then for sqisign define interface for converting to fp2_t)
#   Optimize 2x4 when num_limbs % 4 != 0
#   Optimize with addition chain for pow3div4
#   Optimize inv (twice as slow as lvl3 unvec)
#   We have R as a constant, so defining a set_binary function is trivial
#   Can we optimize Fp2 2x4x1 mul with lazy red?

# Idea: Create strategies at the start instead of having conditional logic on the shape of p deep in the code.
# Examples: R with num_limbs + 1 and lazy red in fp2. Don't subtract by p in Mont red.

# Maybe define one interface for fp2 8, 4 and 2 elements, hide implementation details (perms in high-level code)? Might take too much time, 
# also need example where one is not faster than other, low priority

# Have two sets of conversion functions, interleaved (like 8x1x1 to 4x2x1) and un-interleaved.
# More user friendly for a general purpose library.

### input ###
#dir = "out"
num_kat = 20
gen_addchain = True # Requires addchain to be installed

# lvl3
# e = 376
# c = 65

# lvl5
# e = 500
# c = 27

# no shift
e = 2601
c = 609

# mersenne
# e = 521
# c = 1

#############

pp, strat = comp_params_and_strat(e, c, num_kat, gen_addchain)
v = get_param_names()

# if not os.path.exists(dir):
#     os.mkdir(dir)

write_fp_header_file("../include", pp, v)
write_fp_file("../common", pp, v, strat)
write_fp2_file("../common", pp, v, strat)
write_fp_kat_file("../test", pp, v)
write_fp2_kat_file("../test", pp, v)
