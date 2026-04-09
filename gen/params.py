from dataclasses import dataclass
from enum import Enum
from math import ceil
from sage.all import ZZ

@dataclass
class FpParams:
    e: int
    c: int
    p: ZZ
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
    R: ZZ
    R2: ZZ
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
    R: str
    R2: str
    inv3: str
    p_64: str
    R_64: str
    inv_const_64: str
    num_kat_fp: str
    num_kat_fp2: str

class rdc_8x1_mode(Enum):
    c1 = 1
    shift0 = 2
    ham2plus = 3
    ham2minus = 4
    shift0_ham2plus = 5
    shift0_ham2minus = 6
    standard = 7

@dataclass
class Strat:
    gen_addchain: bool
    fp2_lazy: bool
    rdc_final_sub_p: bool
    rdc_8x1: rdc_8x1_mode
    ham2_k: int

def get_p(e, c):
    p = ZZ(c * 2**e -1)
    pbits = p.nbits()
    assert p.is_prime()
    # script only designed for cryptographically sized primes
    assert pbits >= 51*4 

    return p, pbits

def get_limb_params(pbits, e, c):
    num_limbs = ceil(pbits/51)
    num_words64 = ceil(pbits/64)

    # TODO: Can it sometimes be worth it to pick a larger limb size that divides pp.e?
    limb_len = ceil(pbits/num_limbs)
    last_limb_len = pbits - limb_len*(num_limbs-1)

    # For simplicity we require the cofactor c to be small
    # TODO: Do we want to handle larger c?
    assert ZZ(c).nbits() + (e % limb_len ) < 52

    return num_limbs, limb_len, last_limb_len, num_words64

def get_len_42_24(num_limbs):
    n42 = ceil(num_limbs/2)
    n24 = ceil(num_limbs/4)
    rem42 = num_limbs % 2
    rem24 = num_limbs % 4
    return n42, n24, rem42, rem24

def get_strat(e, c, limb_len, last_limb_len, gen_addchain):
    has_spare_bits = (limb_len - last_limb_len) > 0
    fp2_lazy = limb_len <= 50
    # last_limb_len + 1 <= limb_len, we can fit results in [0, 2p) without changes.
    rdc_final_sub_p = not (has_spare_bits or limb_len <= 49)

    # Picking rdc_8x1 strategy:
    # Detecting whether c = 2^k +- 1
    ham2_k = None
    c_form = None
    if c > 1:
        for kk in range(1, 64):
            if c == (1 << kk) + 1:
                ham_2k = kk; c_form = 1
            if c == (1 << kk) - 1:
                ham2_k = kk; c_form = -1

    if c == 1:
        rdc_8x1 = rdc_8x1_mode.c1
    if e % limb_len == 0:
        if c_form == 1:
            rdc_8x1 = rdc_8x1_mode.shift0_ham2plus
        elif c_form == -1:
            rdc_8x1 = rdc_8x1_mode.shift0_ham2minus
        else:
            rdc_8x1 = rdc_8x1_mode.shift0
    elif c_form == 1:
        rdc_8x1 = rdc_8x1_mode.ham2plus
    elif c_form == -1:
        rdc_8x1 = rdc_8x1_mode.ham2minus
    else:
        rdc_8x1 = rdc_8x1_mode.standard

    return Strat(
        gen_addchain = gen_addchain,
        fp2_lazy = fp2_lazy,
        rdc_final_sub_p = rdc_final_sub_p,
        rdc_8x1 = rdc_8x1,
        ham2_k = ham2_k
    )

# TODO: Optimize Montgomery constant for lazy reduction like 48-bit limb case    
def get_mont_const(p, num_limbs, limb_len, strat):
    num_limbs_R = num_limbs+1 if strat.fp2_lazy else num_limbs
    R = ZZ(pow(2, num_limbs_R * limb_len, p))
    R2 = ZZ(pow(R, 2, p))
    return num_limbs_R, R, R2

def get_limb_mask(pp):
    limb_mask = (1 << pp.limb_len) - 1
    last_limb_mask = (1 << pp.last_limb_len) - 1
    return limb_mask, last_limb_mask

def comp_params_and_strat(e, c, num_kat, gen_addchain):
    p, pbits = get_p(e, c)
    num_limbs, limb_len, last_limb_len, num_words64 = get_limb_params(pbits, e, c)
    n42, n24, rem42, rem24 = get_len_42_24(num_limbs)
    strat = get_strat(e, c, limb_len, last_limb_len, gen_addchain)
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
        R = "R",
        R2 = "R2",
        inv3 = "INV3",
        p_64 = "P_64",
        R_64 = "R_64",
        inv_const_64 = "INV_CONST_64",
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