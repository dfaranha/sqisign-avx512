from sage.rings.integer_ring import ZZ

from elliptic_types import Curve, BasisX, PointX
from elliptic_arithmetic import x_mul
from utilities import sqrt_fp2


def normalize_curve_constants(E: Curve):
    """
    Normalise to ensure C and C24 are one
    """
    F = E.A.parent()
    return Curve(E.A / E.C, F.one(), E.A24 / E.C24, F.one())


def projective_difference(E: Curve, xP: PointX, xQ: PointX):
    """
    Algorithm 8.10 of the SQISign spec.

    Given (XP : ZP) and (XQ : ZQ) compute xPQ which is
    equal to x(P ± Q) with a deterministic choice
    """
    A = E.A
    assert E.C.is_one()
    XP, ZP = xP
    XQ, ZQ = xQ

    b_xx = (XP * XQ - ZP * ZQ) ** 2
    b_xz = (XP * XQ + ZP * ZQ) * (XP * ZQ + ZP * XQ) + 2 * A * XP * XQ * ZP * ZQ
    b_zz = (XP * ZQ - ZP * XQ) ** 2

    lam = (ZP.conjugate() * ZQ.conjugate()) ** 2
    b_xx = lam * b_xx
    b_xz = lam * b_xz
    b_zz = lam * b_zz

    delta = sqrt_fp2(b_xz**2 - b_xx * b_zz)
    xPQ = PointX(delta + b_xz, b_zz)

    return xPQ


def torsion_basis_from_hint(E: Curve, hint: ZZ, e: ZZ) -> BasisX:
    """
    Algorithm 2.2 of the SQISign spec.

    Compute a deterministic basis for E[2^e]  <P, Q> given a hint
    which is used to optimise the computation of x(P) and x(Q).
    """
    E = normalize_curve_constants(E)

    A = E.A
    assert E.C.is_one()
    Fp2 = A.parent()
    p = Fp2.characteristic()

    # Unpack the hint
    hint_A = hint & 1
    hint_P = hint // 2

    # debugging
    assert hint_A == A.is_square()

    # When hint_P is zero, we have an edge case where the hint did not
    # work and we need to find our own basis. This is rare, and doesn't
    # happen in the KATS so I'm not implementing it right now.
    #
    # We work projectively to avoid inverting Fp2([1, hint_P]) when hint_A
    # is true, and the point Q which is not above (0 : 0) has x coordinate
    # -(xP / zP + A), or projectively (-(xP + A*zP) : zP)
    if hint_P:
        if hint_A:
            xP, zP = -A, Fp2([1, hint_P])
            xQ, zQ = -(xP + A*zP), zP
        else:
            xP, zP = hint_P * A, Fp2.one()
            xQ, zQ = -(xP + A), Fp2.one()
    else:
        raise NotImplementedError

    # Clear the cofactor with x-only arithmetic
    P = PointX(xP, zP)
    Q = PointX(xQ, zQ)

    cofactor = (p + 1) // (2**e)
    P = x_mul(E, P, cofactor)
    Q = x_mul(E, Q, cofactor)

    # Ensure the relative sign matches specification
    # This is very sensitive to projective factors
    PmQ = projective_difference(E, P, Q)

    return BasisX(P.X, P.Z, PmQ.X, PmQ.Z, Q.X, Q.Z)
