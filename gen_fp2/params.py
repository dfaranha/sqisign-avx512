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

class MulNonnegMode(Enum):
    add_p_shifted = 1   # add p*2^(Nl)
    add_2p_shifted = 2  # add 2p*2^(Nl)
    add_4p_shifted = 3  # add 4p*2^(Nl)

class KaratsubaAddMode(Enum):
    lazy = 0
    carry_prop = 1
    reduce = 2

@dataclass
class Strat:
    gen_addchain: bool
    R_ext: bool
    mod_2p: bool
    mul_nonneg: MulNonnegMode
    karat_add: KaratsubaAddMode
    reduce_everything: bool
    sqr_use_add_p_sub: bool
    sqr8_faster_add_p_sub: bool
    sqr4_faster_add_p_sub: bool

def get_p(e, c):
    p = c * 2**e -1
    pbits = p.bit_length()
    #assert p.is_prime()
    # script only designed for cryptographically sized primes
    assert pbits > 51*3 # need at least 4 limbs

    # min_limb_len <= 39 when num_limbs >= 4
    # assert c.bit_length() <= 39

    return p, pbits

def get_len_42_24(num_limbs):
    n42 = ceil(num_limbs/2)
    n24 = ceil(num_limbs/4)
    rem42 = num_limbs % 2
    rem24 = num_limbs % 4
    return n42, n24, rem42, rem24

def get_strat(num_limbs, limb_len, last_limb_len, gen_addchain):

    # main strats to optimize 8x1x1 mul
    if limb_len <= 50:
        if last_limb_len <= limb_len-2:
            mod_2p = True
            R_ext = True
            mul_nonneg = MulNonnegMode.add_p_shifted
            karat_add = KaratsubaAddMode.lazy
            reduce_everything = False
        elif last_limb_len == limb_len-1:
            mod_2p = True
            R_ext = True
            mul_nonneg = MulNonnegMode.add_2p_shifted
            karat_add = KaratsubaAddMode.lazy
            reduce_everything = False
        elif limb_len < 50:
            mod_2p = True
            R_ext = True
            mul_nonneg = MulNonnegMode.add_4p_shifted
            karat_add = KaratsubaAddMode.lazy
            reduce_everything = False
        else:
            mod_2p = False
            R_ext = True
            mul_nonneg = MulNonnegMode.add_p_shifted
            karat_add = KaratsubaAddMode.lazy
            reduce_everything = False
    else: # limb_len = 51
        if last_limb_len <= 49:
            mod_2p = True
            R_ext = True
            mul_nonneg = MulNonnegMode.add_p_shifted
            karat_add = KaratsubaAddMode.carry_prop
            reduce_everything = False
        elif last_limb_len == 50:
            mod_2p = False
            R_ext = True
            mul_nonneg = MulNonnegMode.add_p_shifted
            karat_add = KaratsubaAddMode.carry_prop
            reduce_everything = False
        else: #last_limb_len = 51
            mod_2p = False
            R_ext = True
            mul_nonneg = MulNonnegMode.add_p_shifted
            karat_add = KaratsubaAddMode.reduce
            reduce_everything = True

    # Notes on what I'm doing:
    # Realizes that we can do mod_2p and most of lazy red for the "worst" cases
    # if we keep using R_ext = True and so the Karatsuba add fully reduced

    # For limb_len = last_limb_len = 51: 
    # By using R_ext = True, we can add p shifted, skipping cond add p
    # Furthermore, while we before did no lazy comp, I realized that we just need to
    # compute the Karatsuba addition properly, then the input to fp_mul_8x1 has 51 bit limbs.
    # After this, we can go way beyond the limb size, so the rest of the computation can be 
    # lazy like before. For my test prime, I went from 260 -> 187 cycles.

    # For limb_len = 51, last_limb_len = 50:
    # By doing the Karatsuba addition properly, we can keep inputs mod 2p.
    # This gave a speedup from 234 -> 212 cycles for my test prime.

    # For limb_len = last_limb_len = 50: 
    # with mod 2p, we get 51 bits in top, and then Karatsuba add gives 52.
    # However, by doing the addition properly, we once again get 51 bits in the top.
    # carry prop karat doesn't work here, as that would only increase the top by an additional bit
    # 228 -> 213
    
    # TODO:
    # remove cond_add mode? [x]
    # add mode for karatsuba add: lazy, carry, reduce [x]
    # fix 8x1x1 sqr too
    # other modes
    # update limb_len picker
    # remove old strat data
    # add SQIsign v3 primes
    # check Patrick optimization

    # additional strat for squaring
    sqr_use_add_p_sub = R_ext  #  TODO: Working on removing this
    if mod_2p:
        sqr8_faster_add_p_sub = (limb_len <= 49)
        sqr4_faster_add_p_sub = (limb_len <= 50)
    else:
        sqr8_faster_add_p_sub = (limb_len <= 50)
        sqr4_faster_add_p_sub = True

    return Strat(
        gen_addchain = gen_addchain,
        R_ext = R_ext,
        mod_2p = mod_2p,
        reduce_everything = reduce_everything,
        mul_nonneg = mul_nonneg,
        karat_add = karat_add,
        sqr_use_add_p_sub = sqr_use_add_p_sub,
        sqr8_faster_add_p_sub = sqr8_faster_add_p_sub,
        sqr4_faster_add_p_sub = sqr4_faster_add_p_sub
    )

def get_limb_params(pbits, e, c):
    num_limbs = ceil(pbits/51)
    num_words64 = ceil(pbits/64)

    min_limb_len = ceil(pbits/num_limbs)
    # We use the assumption that all but the top limb is all 1's.
    # e >= (N-1) ell => ell <= floor(e/(N-1))
    max_limb_len = min(51, e//(num_limbs-1))
    assert min_limb_len <= max_limb_len

    limb_len = find_best_limb_len_in_range(pbits, num_limbs, min_limb_len, max_limb_len)
    last_limb_len = get_last_limb_len(pbits, num_limbs, limb_len)

    return num_limbs, limb_len, last_limb_len, num_words64

def get_last_limb_len(pbits, num_limbs, limb_len):
    return pbits - limb_len*(num_limbs-1)

def find_best_limb_len_in_range(pbits, num_limbs, min_limb_len, max_limb_len):
    if min_limb_len == max_limb_len:
        return min_limb_len

    candidates = [ell for ell in range(min_limb_len, max_limb_len+1)]

    # Sorted by priority
    predicates = [
        lambda strat: strat.mod_2p,
        lambda strat: not strat.karat_add == KaratsubaAddMode.reduce,
        lambda strat: strat.karat_add == KaratsubaAddMode.lazy,
        lambda strat: strat.sqr_use_add_p_sub,
        lambda strat: strat.sqr8_faster_add_p_sub,
        lambda strat: strat.sqr4_faster_add_p_sub,
        lambda strat: strat.R_ext if num_limbs > 4 else not strat.R_ext,
        lambda strat: not strat.reduce_everything
    ]
    
    for pred in predicates:
        new_candidates = find_candidates_satisfying_predicate(pbits, num_limbs, candidates, pred)
        if len(new_candidates) == 1:
            return new_candidates[0]
        elif len(new_candidates) > 1:
            candidates = new_candidates
    return candidates[0]

def find_candidates_satisfying_predicate(pbits, num_limbs, candidates, pred):
    new_candidates = []
    for limb_len in candidates:
        last_limb_len = get_last_limb_len(pbits, num_limbs, limb_len)
        strat = get_strat(num_limbs, limb_len, last_limb_len, gen_addchain=None)
        if pred(strat):
            new_candidates.append(limb_len)
    return new_candidates

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