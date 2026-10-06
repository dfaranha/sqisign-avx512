import os
import sys
from gen_addchain import run_addchain_exp3div4
from gen_fp import write_fp_file
from gen_fp2 import write_fp2_file
from gen_fp_header import write_fp_header_file
from gen_fp2_header import write_fp2_header_file
from gen_fp_consts_header import write_fp_consts_header
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

# TODO Check all cases for fp_div2_2x4

if len(sys.argv) > 1:
    arg = sys.argv[1]
    if arg == "lvl1":
        e = 324
        c = 3
    elif arg == "lvl3":
        e = 500
        c = 27
    elif arg == "lvl5":
        e = 664
        c = 17
else:
    # limb_len = 51, last_limb_len = 51: 
    # c = 1125899906842695
    # e = 153

    # limb_len = 51, last_limb_len = 50: 
    # c = 562949953421387
    # e = 204

    # limb_len = 51, last_limb_len = 49: 
    # c = 281474976710875
    # e = 153

    # limb_len = 50, last_limb_len = 50: 
    c = 562949953421499 
    e = 200

    # limb_len = 50, last_limb_len = 49: 
    # c = 281474976710717
    # e = 150

    # limb_len = 50, last_limb_len = 48: 
    # c = 140737488355547
    # e = 200

    # limb_len = 49, last_limb_len = 49: 
    # c = 281474976710841
    # e = 147

    # limb_len = 49, last_limb_len = 48: 
    # c = 140737488355515
    # e = 196

    # limb_len = 49, last_limb_len = 47: 
    # c = 70368744178131
    # e = 147

    # mersenne prime: 
    # c = 1
    # e = 607


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

write_fp_consts_header(include_dir, pp, v)
write_fp_header_file(include_dir, pp, v, strat)
write_fp2_header_file(include_dir, pp, v, strat)
write_fp_file(common_dir, pp, v, strat, chain)
write_fp2_file(common_dir, pp, v, strat)
write_fp_kat_file(test_dir, pp, v)
write_fp2_kat_file(test_dir, pp, v)
