from dataclasses import dataclass
from sage.all import ZZ
from math import ceil

@dataclass
class FpParams:
    e: int
    c: int
    p: ZZ
    pbits: int
    num_limbs: int
    limb_len: int
    last_limb_len: int
    num_limbs64: int
    n42: int
    rem42: int
    n24: int
    rem24: int
    R: ZZ
    R2: ZZ

@dataclass
class FpParamNames:
    limb_len: str
    num_limbs: str
    limb_align: str
    limb_mask: str
    last_limb_mask: str
    num_limbs64: str
    n42: str
    n24: str
    p_plus_1_top: str
    R: str
    R2: str


def get_p(e, c):
    p = ZZ(c * 2**e -1)
    pbits = p.nbits()
    assert p.is_prime()
    # script only designed for cryptographically sized primes
    assert pbits >= 51*4 

    return p, pbits

def get_limb_params(pbits, e, c):
    num_limbs = ceil(pbits/51)
    num_limbs64 = ceil(pbits/64)

    # TODO: Can it sometimes be worth it to pick a larger limb size that divides pp.e?
    limb_len = ceil(pbits/num_limbs)
    last_limb_len = pbits - limb_len*(num_limbs-1)

    # For simplicity we require the cofactor c to be small
    # TODO: Do we want to handle larger c?
    assert ZZ(c).nbits() + (e % limb_len ) < 52

    return num_limbs, limb_len, last_limb_len, num_limbs64

# TODO: Optimize Montgomery constant for lazy reduction like 48-bit limb case    
def get_mont_const(p, num_limbs, limb_len):
    R = ZZ(pow(2, (num_limbs+1) * limb_len, p))
    R2 = ZZ(pow(R, 2, p))
    return R, R2

def get_fp_params(e, c):
    p, pbits = get_p(e, c)
    num_limbs, limb_len, last_limb_len, num_limbs64 = get_limb_params(pbits, e, c)
    R, R2 = get_mont_const(p, num_limbs, limb_len)

    return FpParams(
        e = e, 
        c = c, 
        p = p, 
        pbits = pbits, 
        num_limbs = num_limbs, 
        limb_len = limb_len, 
        last_limb_len = last_limb_len, 
        num_limbs64 = num_limbs64, 
        n42 = ceil(num_limbs/2),
        rem42 = num_limbs % 2,
        n24 = ceil(num_limbs/4), 
        rem24 = num_limbs % 4,
        R = R, 
        R2 = R2
    )

def get_param_names():
    return FpParamNames(
        limb_len = "LIMB_LEN", 
        num_limbs = "NUM_LIMBS", 
        limb_align = "LIMB_ALIGN", 
        num_limbs64= "NUM_LIMBS_64",
        n42 = "N4x2",
        n24 = "N2x4",
        limb_mask = "LIMB_MASK", 
        last_limb_mask = "LAST_LIMB_MASK",
        p_plus_1_top = "P_PLUS_1_TOP",
        R = "R",
        R2 = "R2"
    )