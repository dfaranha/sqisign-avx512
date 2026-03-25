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

    # 4M parallel
    t0 = XP * XQ
    t1 = ZP * ZQ
    t4 = XP * ZQ
    t5 = ZP * XQ

    # 4 add parallel
    t3 = t0 + t1
    t7 = t4 + t5
    t2 = t0 - t1
    t6 = t4 - t5

    # 2M parallel (uses 2S)
    t8 = t3 * t7 
    t9 = t0 * t1
    b_xx = t2 ** 2
    b_zz = t6 ** 2

    # Needs t9, loose end mul on it's own?
    t10 = A * t9
    
    # Annoying add
    b_xz = t8 + t10 + t10

    # No need to mul, we have t11 from t1 directly with one fp negation
    # t11 = ZP.conjugate() * ZQ.conjugate()
    F = t1.parent()
    t11 = F([t1[0],  -t1[1]])
    assert t11 == ZP.conjugate() * ZQ.conjugate()

    # 4M parallel (uses 1S)
    lam = t11 ** 2
    b_xx = lam * b_xx
    b_xz = lam * b_xz
    b_zz = lam * b_zz

    # 2M parallel
    t12 = b_xz ** 2
    t13 = b_xx * b_zz

    # Annoying add
    delta = sqrt_fp2(t12 - t13)

    # Annoying add
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
