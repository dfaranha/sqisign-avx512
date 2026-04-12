import os
from gen_addchain import run_addchain_exp3div4
from gen_fp import write_fp_file
from gen_fp2 import write_fp2_file
from gen_fp_header import write_fp_header_file
from gen_fp_kat import write_fp_kat_file
from gen_fp2_kat import write_fp2_kat_file
from params import (comp_params_and_strat, get_param_names)


# TODO:
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
#   Change 4x2 and 2x4 to use single vars instead of tmp arrays.

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

#lvl1
e = 248
c = 5

# lvl3
# e = 376
# c = 65

# lvl5
# e = 500
# c = 27

# no shift
# e = 2601
# c = 609

# mersenne
# e = 521
# c = 1

# limb_len = last_limb_len = 51: FAILS 2x4 mul a * (b * c) != (a * b) * c and KAT
# e = 398
# c = 893

# limb_len = 51, last_limb_len = 50
# e = 397
# c = 729

# limb_len: 50 last_limb_len: 50
# e = 395
# c = 19

# limb_len: 50 last_limb_len: 49
# e = 392
# c = 123

# limb_len: 50 last_limb_len: 48
# e = 338
# c = 683

# limb_len: 50 last_limb_len 47 
# e = 343
# c = 9

# limb_len: 50 last_limb_len 45
# e = 336
# c = 297


#############

pp, strat = comp_params_and_strat(e, c, num_kat, gen_addchain)
print(pp)
print(strat)
v = get_param_names()
chain = run_addchain_exp3div4(pp, strat)

# if not os.path.exists(dir):
#     os.mkdir(dir)
script_dir = os.path.dirname(os.path.abspath(__file__))
include_dir = os.path.join(script_dir, "..", "include")
common_dir = os.path.join(script_dir, "..", "common")
test_dir = os.path.join(script_dir, "..", "test")

write_fp_header_file(include_dir, pp, v)
write_fp_file(common_dir, pp, v, strat, chain)
write_fp2_file(common_dir, pp, v, strat)
write_fp_kat_file(test_dir, pp, v)
write_fp2_kat_file(test_dir, pp, v)
