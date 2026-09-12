from dataclasses import dataclass
from enum import Enum
from math import ceil

@dataclass
class FpParams:
    e: int
    c: int
    p: int
    pbits: int
    num_limbs: int
    limb_len: int
    last_limb_len: int
    num_words64: int
    n42: int
    rem42: int
    n24: int
    rem24: int
    num_limbs_R: int
    R: int
    R2: int
    num_kat: int

@dataclass
class FpParamNames:
    limb_len: str
    num_limbs: str
    limb_align: str
    limb_mask: str
    last_limb_mask: str
    num_words64: str
    n42: str
    n24: str
    p_plus_1_top: str
    p_plus_1_div2_top: str
    p_plus_1_div2_top_hi: str
    R: str
    R2: str
    inv3: str
    p_64: str
    R_64: str
    inv_const_64: str
    c: str
    c_nbits: str
    c_nwords64: str
    num_kat_fp: str
    num_kat_fp2: str

class Fp2Mode(Enum):
    lazy = 0
    lazy_carry = 1
    reduce = 2
    reduce_everything = 3

@dataclass
class Strat:
    gen_addchain: bool
    R_ext: bool
    mod_2p: bool
    mul8_faster_nonneg: bool
    carry_prop_karat: bool
    reduce_everything: bool
    sqr8_use_add_p_sub: bool
    sqr8_faster_add_p_sub: bool
    sqr4_faster_add_p_sub: bool
    mul_8x1: Fp2Mode
    sqr_8x1: Fp2Mode
    sqr_4x2: Fp2Mode

def get_p(e, c):
    p = c * 2**e -1
    pbits = p.bit_length()
    #assert p.is_prime()
    # script only designed for cryptographically sized primes
    assert pbits >= 51*4 

    # min_limb_len <= 39 when num_limbs >= 4
    assert c.bit_length() <= 39

    

    return p, pbits

def get_limb_params(pbits, e, c):
    num_limbs = ceil(pbits/51)
    num_words64 = ceil(pbits/64)

    limb_len = ceil(pbits/num_limbs)
    last_limb_len = pbits - limb_len*(num_limbs-1)

    # For simplicity we require that all but the last limb of c is limb_mask
    assert c == 1 or (c.bit_length() + (e % limb_len) <= last_limb_len)

    return num_limbs, limb_len, last_limb_len, num_words64

def get_len_42_24(num_limbs):
    n42 = ceil(num_limbs/2)
    n24 = ceil(num_limbs/4)
    rem42 = num_limbs % 2
    rem24 = num_limbs % 4
    return n42, n24, rem42, rem24

def get_strat(num_limbs, limb_len, last_limb_len, gen_addchain):
    #TODO:
    # mp_add_p_sub relies on all but top limb of p being limb_mask
    #   which is why we have the assert in get_limb_params. Generalize to the case c_bits + (e mod limb_len) > last_limb_len
    #  (mult by c, shift by e mod limb_limb_len) Find example with limb_len 51

    if limb_len <= 49:
        mul_8x1 = Fp2Mode.lazy
        sqr_8x1 = Fp2Mode.lazy
        sqr_4x2 = Fp2Mode.lazy
        R_ext = True
        mod_2p = True
    elif limb_len == 50 and last_limb_len <= 49: # 8x1 sqr needs carry when limb_ln = 50
        mul_8x1 = Fp2Mode.lazy
        sqr_8x1 = Fp2Mode.lazy_carry
        sqr_4x2 = Fp2Mode.lazy
        R_ext = True
        mod_2p = True
    elif limb_len == 50 and last_limb_len == 50: # 8x1 sqr needs mod p
        mul_8x1 = Fp2Mode.lazy
        sqr_8x1 = Fp2Mode.lazy_carry
        sqr_4x2 = Fp2Mode.lazy
        R_ext = True
        mod_2p = False
    elif last_limb_len <= 48: # 8x1 mul & 4x2 sqr requires carry
        mul_8x1 = Fp2Mode.lazy_carry
        sqr_8x1 = Fp2Mode.lazy_carry
        sqr_4x2 = Fp2Mode.lazy_carry
        R_ext = False
        #R_ext = num_limbs > 5 # Experimental data shows this is efficiency threshold
        mod_2p = True
    elif last_limb_len == 49: # 8x1 squaring must switch to mod p
        mul_8x1 = Fp2Mode.lazy_carry
        sqr_8x1 = Fp2Mode.lazy_carry
        sqr_4x2 = Fp2Mode.lazy_carry
        R_ext = False
        mod_2p = False
    elif last_limb_len == 50: # 8x1 squaring must reduce sub, need R_ext to keep square within [0, pR)
        mul_8x1 = Fp2Mode.lazy_carry
        sqr_8x1 = Fp2Mode.reduce
        sqr_4x2 = Fp2Mode.lazy_carry
        R_ext = True
        mod_2p = False
    else: #8x1x1 lazy mult & squaring doesn't work
        mul_8x1 = Fp2Mode.reduce_everything
        sqr_8x1 = Fp2Mode.reduce_everything
        sqr_4x2 = Fp2Mode.lazy_carry
        R_ext = True
        mod_2p = False

    # TODO: Can we save one limb in prod when last_limb_len < 2 * limb_len?
    return Strat(
        gen_addchain = gen_addchain,
        mul_8x1 = mul_8x1,
        sqr_8x1 = sqr_8x1,
        sqr_4x2 = sqr_4x2,
        R_ext = R_ext,
        mod_2p = mod_2p,
        carry_prop_karat = False,
        reduce_everything = False,
        mul8_faster_nonneg = False,
        sqr8_use_add_p_sub = False,
        sqr8_faster_add_p_sub = False,
        sqr4_faster_add_p_sub = False
    )

def get_strat_new(num_limbs, limb_len, last_limb_len, gen_addchain):

    # About to implement new strategies, adding this to swap out one by one and make debugging easier !!!!
    
    if limb_len <= 49:
        mul_8x1 = Fp2Mode.lazy
        sqr_8x1 = Fp2Mode.lazy
        sqr_4x2 = Fp2Mode.lazy
        R_ext = True
        mod_2p = True
    elif limb_len == 50 and last_limb_len <= 49: # 8x1 sqr needs carry when limb_ln = 50
        mul_8x1 = Fp2Mode.lazy
        sqr_8x1 = Fp2Mode.lazy_carry
        sqr_4x2 = Fp2Mode.lazy
        R_ext = True
        mod_2p = True
    elif limb_len == 50 and last_limb_len == 50: # 8x1 sqr needs mod p
        mul_8x1 = Fp2Mode.lazy
        sqr_8x1 = Fp2Mode.lazy_carry
        sqr_4x2 = Fp2Mode.lazy
        R_ext = True
        mod_2p = False
    elif last_limb_len <= 48: # 8x1 mul & 4x2 sqr requires carry
        mul_8x1 = Fp2Mode.lazy_carry
        sqr_8x1 = Fp2Mode.lazy_carry
        sqr_4x2 = Fp2Mode.lazy_carry
        R_ext = False
        #R_ext = num_limbs > 5 # Experimental data shows this is efficiency threshold
        mod_2p = True
    elif last_limb_len == 49: # 8x1 squaring must switch to mod p
        mul_8x1 = Fp2Mode.lazy_carry
        sqr_8x1 = Fp2Mode.lazy_carry
        sqr_4x2 = Fp2Mode.lazy_carry
        R_ext = False
        mod_2p = False
    elif last_limb_len == 50: # 8x1 squaring must reduce sub, need R_ext to keep square within [0, pR)
        mul_8x1 = Fp2Mode.lazy_carry
        sqr_8x1 = Fp2Mode.reduce
        sqr_4x2 = Fp2Mode.lazy_carry
        R_ext = True
        mod_2p = False
    else: #8x1x1 lazy mult & squaring doesn't work
        mul_8x1 = Fp2Mode.reduce_everything
        sqr_8x1 = Fp2Mode.reduce_everything
        sqr_4x2 = Fp2Mode.lazy_carry
        R_ext = True
        mod_2p = False

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # main strats to optimize 8x1x1 mul
    if limb_len <= 50:
        carry_prop_karat = False
        reduce_everything = False
        if last_limb_len <= limb_len-2:
            mod_2p = True
            R_ext = True
            mul8_faster_nonneg = True
        elif last_limb_len == limb_len-1 or limb_len < 50:
            mod_2p = True
            R_ext = True
            mul8_faster_nonneg = False
        else:
            mod_2p = False
            R_ext = True
            mul8_faster_nonneg = False
    else: # limb_len = 51
        carry_prop_karat = True # Also in fp2 sqr
        if last_limb_len <= 49:
            mod_2p = True
            R_ext = True
            mul8_faster_nonneg = True
            reduce_everything = False
        elif last_limb_len == 50:
            mod_2p = False
            R_ext = False
            mul8_faster_nonneg = False
            reduce_everything = False
        else: #last_limb_len = 51
            mod_2p = False
            R_ext = False
            mul8_faster_nonneg = False
            reduce_everything = True

    # additional strat for squaring
    sqr8_use_add_p_sub = R_ext
    if mod_2p:
        sqr8_faster_add_p_sub = (limb_len <= 49)
        sqr4_faster_add_p_sub = (limb_len <= 50)
    else:
        sqr8_faster_add_p_sub = (limb_len <= 50)
        sqr4_faster_add_p_sub = True

    return Strat(
        gen_addchain = gen_addchain,
        mul_8x1 = mul_8x1,
        sqr_8x1 = sqr_8x1,
        sqr_4x2 = sqr_4x2,
        R_ext = R_ext,
        mod_2p = mod_2p,
        carry_prop_karat = carry_prop_karat,
        reduce_everything = reduce_everything,
        mul8_faster_nonneg = mul8_faster_nonneg,
        sqr8_use_add_p_sub = sqr8_faster_add_p_sub,
        sqr8_faster_add_p_sub = sqr8_faster_add_p_sub,
        sqr4_faster_add_p_sub = sqr4_faster_add_p_sub
    )



def get_mont_const(p, num_limbs, limb_len, strat):
    num_limbs_R = num_limbs+1 if strat.R_ext else num_limbs
    R = pow(2, num_limbs_R * limb_len, p)
    R2 = pow(R, 2, p)
    return num_limbs_R, R, R2

def get_limb_mask(pp):
    limb_mask = (1 << pp.limb_len) - 1
    last_limb_mask = (1 << pp.last_limb_len) - 1
    return limb_mask, last_limb_mask

# Assumes c is small.
def get_div2_consts(e, c, num_limbs, limb_len):
    # Need top limb(s) of (p+1)/2 = c*2^(e-1)
    idx_start = (e-1)// limb_len
    val = c << ((e-1) % limb_len)
    if idx_start == num_limbs-1:
        lo = val
        hi = None
    else:
        lo = val & ((1 << limb_len) - 1)
        hi = val >> limb_len
    return lo, hi

def comp_params_and_strat(e, c, num_kat, gen_addchain):
    p, pbits = get_p(e, c)
    num_limbs, limb_len, last_limb_len, num_words64 = get_limb_params(pbits, e, c)
    n42, n24, rem42, rem24 = get_len_42_24(num_limbs)
    strat = get_strat(num_limbs, limb_len, last_limb_len, gen_addchain)
    num_limbs_R, R, R2 = get_mont_const(p, num_limbs, limb_len, strat)

    pp = FpParams(
        e = e, 
        c = c, 
        p = p, 
        pbits = pbits, 
        num_limbs = num_limbs, 
        limb_len = limb_len, 
        last_limb_len = last_limb_len, 
        num_words64 = num_words64, 
        n42 = n42,
        rem42 = rem42,
        n24 = n24, 
        rem24 = rem24,
        num_limbs_R = num_limbs_R,
        R = R, 
        R2 = R2,
        num_kat=num_kat
    )
    return pp, strat

def get_param_names():
    return FpParamNames(
        limb_len = "LIMB_LEN", 
        num_limbs = "NUM_LIMBS", 
        limb_align = "LIMB_ALIGN", 
        num_words64= "NWORDS64",
        n42 = "N4x2",
        n24 = "N2x4",
        limb_mask = "LIMB_MASK", 
        last_limb_mask = "LAST_LIMB_MASK",
        p_plus_1_top = "P_PLUS_1_TOP",
        p_plus_1_div2_top = "P_PLUS_1_DIV2_TOP",
        p_plus_1_div2_top_hi = "P_PLUS_1_DIV2_TOP_HI",
        R = "R",
        R2 = "R2",
        inv3 = "INV3",
        p_64 = "P_64",
        R_64 = "R_64",
        inv_const_64 = "INV_CONST_64",
        c = "COFACTOR",
        c_nbits = "COFACTOR_NBITS",
        c_nwords64 = "COFACTOR_NWORDS64",
        num_kat_fp="NUM_FP_KAT",
        num_kat_fp2="NUM_FP2_KAT"
    )

def get_fp_vec_params(num_fp, num_limbs):
    assert num_fp in [2, 4, 8]

    n42, n24, _, _ = get_len_42_24(num_limbs)

    if num_fp == 8:
        num_vec = num_limbs
        limbs_per_vec = 1
    elif num_fp == 4:
        num_vec = n42
        limbs_per_vec = 2
    else:
        num_vec = n24
        limbs_per_vec = 4
    return num_vec, limbs_per_vec

def get_fp_num_vec_name(v, nump_fp):
    assert nump_fp in [2, 4, 8]
    if nump_fp == 8:
        return v.num_limbs
    elif nump_fp == 4:
        return v.n42
    else:
        return v.n24
    
def get_fp2_num_vec_name(v, format):
    if format == "8x1x1":
        return f"2*{v.num_limbs}"
    elif format == "4x1x2":
        return f"2*{v.n42}"
    elif format == "4x2x1" or format == "2x4x1":
        return v.num_limbs
    else: # 2x4x1
        return v.n42
    
def get_inv_params(pp):
    N_outer = ceil((2 * pp.pbits - 62) / 31)
    N_scalar = 2 * pp.pbits - 1 - N_outer * 31
    if N_scalar < 0:
        N_scalar = 0
    N = N_outer * 31 + N_scalar
    R_64 = pow(2, 64 * pp.num_words64)
    inv2powN = pow(2, - N, pp.p)
    # constant needed to correct extended gcd result
    inv_const = (pp.R * R_64 * inv2powN) % pp.p  
    return N_outer, N_scalar, inv_const