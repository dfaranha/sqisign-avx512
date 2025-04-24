from utilities.deterministic_sqrt import sqrt_fp2


def two_isogeny_codomain(XP, ZP):
    """
    Algorithm 8.20 of the SQISign spec.

    Compute the codomain of the 2-isogeny E -> E/<P> for P != (0 : 1)
    of order 2.
    """
    A24_prime = XP * XP
    C24_prime = ZP * ZP
    A24_prime = C24_prime - A24_prime
    return A24_prime, C24_prime


def two_isogeny_eval(XP, ZP, XQ, ZQ):
    """
    Algorithm 8.21 of the SQISign spec.

    Evaluate a point Q (XQ : ZQ) under the action of the 2-isogeny E -> E/<P>
    for P != (0 : 1) of order 2
    """
    t0 = XP + ZP
    t1 = XP - ZP
    t2 = XQ + ZQ
    t3 = XQ - ZQ

    t0 = t0 * t3
    t1 = t1 * t2
    t2 = t0 + t1
    t3 = t0 - t1

    XQ_prime = XQ * t2
    ZQ_prime = ZQ * t3

    return XQ_prime, ZQ_prime


def two_isogeny_codomain_singular(A24, C24):
    """
    Algorithm 8.22 of the SQISign spec.

    Compute the codomain of the 2-isogeny E -> E/<P> for P = (0 : 1)
    """
    t0 = A24 + A24
    t0 - t0 - C24
    t0 = t0 + t0
    t0 = t0 / C24
    c0 = t0
    A24_prime = t0 + t0
    t0 = t0 - 4
    t0 = sqrt_fp2(t0)
    c1 = -t0
    C24_prime = t0 + t0
    A24_prime = A24_prime + C24_prime
    C24_prime = C24_prime + C24_prime

    return (A24_prime, C24_prime), (c0, c1)


def two_isogeny_eval_singular(XP, ZP, c0, c1):
    """
    Algorithm 8.23 of the SQISign spec.

    Evaluate a point P (XP : ZP) under the action of the 2-isogeny E -> E/<P>
    """
    t0 = XP * ZP
    t1 = c0 * ZP
    t1 = t1 + XP
    t1 = t1 * XP
    XQ = ZP * ZP
    XQ = XQ + t1
    ZQ = t0 * c1

    return XQ, ZQ


def four_isogeny_codomain(XP, ZP):
    """
    Algorithm 8.24 of the SQISign spec.

    Compute the codomain of the 4-isogeny E -> E/<P> such that [2]P != (0 : 1)
    """
    c0 = ZP * ZP
    c1 = XP - ZP
    c2 = XP + ZP
    t0 = XP * XP
    t3 = c0 + t0
    t4 = c0 - t0
    A24_prime = t3 * t4
    C24_prime = c0 * c0
    c0 = c0 + c0
    c0 = c0 + c0

    return (A24_prime, C24_prime), (c0, c1, c2)


def four_isogeny_eval(XQ, ZQ, c0, c1, c2):
    """
    Algorithm 8.25 of the SQISign spec.

    Evaluate a point Q (XQ : ZQ) under the action of the 4-isogeny E -> E/<P>
    for [2]P != (0 : 1) using precomputed constants
    """
    t0 = XQ + ZQ
    t1 = XQ - ZQ
    XQ_prime = t0 * c1
    ZQ_prime = t1 * c2
    t0 = t0 * t1
    t0 = t0 * c0
    t1 = XQ_prime + ZQ_prime
    ZQ_prime = XQ_prime - ZQ_prime
    t1 = t1 * t1
    ZQ_prime = ZQ_prime * ZQ_prime
    XQ_prime = t0 + t1
    t0 = t0 - ZQ_prime
    XQ_prime = XQ_prime * t1
    ZQ_prime = ZQ_prime * t0

    return XQ_prime, ZQ_prime
