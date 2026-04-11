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
    R: str
    R2: str
    inv3: str
    p_64: str
    R_64: str
    inv_const_64: str
    num_kat_fp: str
    num_kat_fp2: str

class Fp2Mode(Enum):
    lazy_no_carry = 0
    lazy_carry = 1
    reduce_everything = 2

@dataclass
class Strat:
    gen_addchain: bool
    fp2_mode: Fp2Mode
    mod_2p_8x1: bool

def get_p(e, c):
    p = c * 2**e -1
    pbits = p.bit_length()
    #assert p.is_prime()
    # script only designed for cryptographically sized primes
    assert pbits >= 51*4 

    return p, pbits

def get_limb_params(pbits, e, c):
    num_limbs = ceil(pbits/51)
    num_words64 = ceil(pbits/64)

    limb_len = ceil(pbits/num_limbs)
    last_limb_len = pbits - limb_len*(num_limbs-1)

    # For simplicity we require the cofactor c to be small
    assert c.bit_length() + (e % limb_len ) < 52

    return num_limbs, limb_len, last_limb_len, num_words64

def get_len_42_24(num_limbs):
    n42 = ceil(num_limbs/2)
    n24 = ceil(num_limbs/4)
    rem42 = num_limbs % 2
    rem24 = num_limbs % 4
    return n42, n24, rem42, rem24

def get_strat(e, c, limb_len, last_limb_len, gen_addchain):
    # Considerations for Fp2 lazy multiplication:

    # 1. Montgomery reduction requires input in [0, p*R)
    # re = a_re*b_re - a_im*b_im
    # im = (a_re + a_im)*(b_re + b_im) - a_re*b_re - a_im*b_im = a_re*b_im + b_re*b_im
    # We need to ensure that when these are passed to rdc, they are in the range.
    # We use the standard choice R = 2^(num_limbs * limb_len).
    # Note that 2^(limb_len - last_limb) * p < R
    #
    # When a, b are reduced mod p: re in (-p^2, p^2) and im in [0, 2p^2)
    # To make re be positive, the only way is to conditionally add p*R, bringing it to [0, p*R] as p < R.
    # If last_limb_len < limb_len: 2p < R and im < 2p^2 < p*R.
    # If last_limb_len = limb_len: 2p > R. Observe that since p < R, 2p - R < p
    #   Then by conditionally subtracting im by p*R, it's new upper bound is 
    #   2p^2 - p*R = p*(2p - R) < p^2 < p*R.
    #   (Observe that this is faster than extending R to 2^((num_limbs+1)*limb_len) )
    #
    # When a,b are reduced mod 2p (i.e. we skip the last subtraction by p in the reduction):
    # re in [-4p^2, 4p^2] and im in [0, 8p^2).
    # If last_limb_len <= limb_len - 3: im is [0, p*R). By conditionally adding p*R to re, it is always in [0, pR).
    # If last_limb_len = limb_len -2, it is sufficient to conditionally subtract im by p*R, but at his point we migh as well reduce by p.
    # So we for last_limb_len >= limb_len-2, we recommend reducing mod p.
    # (Observe that even when we reduce mod 2p, the resulting limbs are at most limb_len bits)

    # 2. Fp multiplication strategy
    # With Fp Karatsuba, we compute (a_lo+a_hi)*(b_lo+b_hi), so input limbs must be 51 bits. This is why we never go above 51 bit limbs.
    # With Fp schoolbook, the input can have 52 bit limbs.
    # Depending on the Fp2 multiplication strategy, we need to ensure that we don't call Fp mul with too large limbs.
    #
    # Fp2 schoolbook with Fp schoolbook (2x2x2), Fp2 schoolbook with Fp Karatsuba (4x2x1, 2x4x1) and Fp2 Karatsuba with Fp schoolbook (4x1x2):
    # Karatsuba increases the limbs to limb_len + 1 <= 52 bits, which is below the AVX mul threshold.
    # Hence, these strategies work.
    #
    # Fp2 Karatsuba with Fp Karatsuba (8x1x1):
    # The standard lazy approach requires limb_len <= 50. Then a_re + a_im is 51 bits, so Fp Karatsuba works.
    # When limb_len = 51, we can propogate carries for a_re + a_im. The top limb grows by 1 bit, and the others are 51 bits.
    # If last_limb < 51: last_limb + carry is still at most 51 bits.
    # If last_limb = 51: last_limb + carry can be 52 bits. 
    #   We need a conditional subtraction by p and another propagation. Now potentially the computation of im can be negative, etc..
    #   In this case it is simply faster to do the standard non-lazy approach. 

    # If last_limb_len = 50, then mod 2p it becomes 51 bits.
    # With Fp2 8x1x1 karatsuba over Fp, a_re + a_im becomes 52 bits (propagating carry doesn't help).
    # mp_mul_8x1_karat requires <= 51 bits, so this doesn't work.
    mod_2p_8x1 = last_limb_len <= limb_len - 3
    subtract_im = last_limb_len == limb_len

    if limb_len <= 50:
        # We reduce with a larger R, ensuring that even with lazy red in Fp2,
        # each Fp result before reduction is <= p*R.
        # This ensures that Montgomery reduction produces a result in [0, 2p)
        
        # In 8x1x1 Fp2 Karatsuba, we compute (a_re + a_im)*(b_re + b_im) with mp_mul_8x1_karat,
        # which requires each input limb to be <= 51 bits.
        # Hence, in order to compute a_re + a_im with mp_add_8x1 with no carry propagation,
        # each limb must be <= 50.
        fp2_mode = Fp2Mode.lazy_no_carry
    elif last_limb_len < limb_len:
        # TODO: This is standard R analysis. Consolidate with rest.

        # When limb_len = 51, we use the standard Montgomery constant.
        # In order for Montgomery red to work, we need the input to be in [0, p*R).
        # Analyzing this when using lazy red:
        # t1 = a_re*b_re - a_im*b_im
        # t3 = (a_re + a_im)*(b_re + b_im) - a_re*b_re - a_im*b_im = a_re*b_im + b_re*b_im
        # (2^(limb_len - last_limb_len))*p < R
        # If we reduce mod 2p, t1 in [-4p^2, 4p^2], t3 in [0, 8p^2].
        # Looking at t3, we then need 8*p < R, i.e. last_limb_len <= limb_len - 3 = 48
        # If we reduce mod p, t1 in [-p^2, p^2] and t3 in [0, 2p^2].
        # last_limb_len <= limb_len is fine. 
        # After adding p*R to t1 when it is negative, it is within bounds.

        # Fp2 Karatsuba using Fp Karatsuba (8x1x1):
        # Since we use Fp karat, we need to ensure that all the lower limbs
        # of a_re + a_im are <= 51 bits. We do this for all but the top limb by propagating carry.
        # For the top limb to also be within the bound, we need last_limb_len to be one bit less, 
        # so that it can fit the carry after propagation.

        # Fp2 Schoolbook using Fp Karatsuba (4x2x1, 2x4x1):
        # Fp Karatsuba trivially works, since all limbs are <= 51

        # Fp2 Karatsuba using Fp Schoolbook (4x1x2)
        # After a_re + a_im, limbs grow to 52 bits, which fp schoolbook can handle,
        # even without us propagating carrys.
        
        # Fp2 Schoolbook using Fp Schoolbook (2x2x2)
        # No issues at all.

        fp2_mode = Fp2Mode.lazy_carry
    else:
        # Fp2 8x1x1 lazy reduction is only possi
        fp2_mode = Fp2Mode.reduce_everything

    # TODO: Can we save one limb in prod when last_limb_len < 2 * limb_len?

    return Strat(
        gen_addchain = gen_addchain,
        fp2_mode = fp2_mode,
        mod_2p_8x1 = mod_2p_8x1
    )
    
def get_mont_const(p, num_limbs, limb_len, strat):
    num_limbs_R = num_limbs+1 if strat.fp2_mode == Fp2Mode.lazy_no_carry else num_limbs
    R = pow(2, num_limbs_R * limb_len, p)
    R2 = pow(R, 2, p)
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