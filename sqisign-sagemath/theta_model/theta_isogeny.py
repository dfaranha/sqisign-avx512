from utilities.deterministic_sqrt import sqrt_fp2
from theta_model.theta_structures import ThetaPoint, ThetaNullPoint


def check_isotropic(P: ThetaPoint, Q: ThetaPoint, I: ThetaPoint):
    """
    Algorithm 8.32 of the SQISign spec.

    Return a boolean indicating whether the 4-torsion points are isotropic
    """
    xP, yP, zP, wP = P.coords()
    xQ, yQ, zQ, wQ = Q.coords()
    A_inv, B_inv, C_inv, D_inv = I.coords()

    t1 = xP * A_inv
    t2 = yP * B_inv
    if t1 != t2:
        return False

    t1 = zP * C_inv
    t2 = wP * D_inv
    if t1 != t2:
        return False

    t1 = xQ * A_inv
    t2 = zQ * C_inv
    if t1 != t2:
        return False

    t1 = yQ * B_inv
    t2 = wQ * D_inv
    if t1 != t2:
        return False

    return True


def generic_codomain_with_eight_torsion(
    T1: ThetaPoint,
    T2: ThetaPoint,
    h1: bool = False,
    h2: bool = True,
    verify: bool = True,
):
    """
    Algorithm 8.31 of the SQISign spec.

    Given points T1, T2 such that ker(Phi) = [4]T1 \oplus [4]T2

    Compute:
      - The inverse of the dual isogenous theta null point (A^-1 : B^-1 : C^-1 : D^-1)
      - The theta null point on B

    Cost 8S + 9M + 24a
    """
    # Compute H . S(Ti)
    if h1:
        xA, xB, yC, yD = T1.hadamard()
        xA, xB, yC, yD = T1._squaring_hadamard_transform(xA, xB, yC, yD)
        zA, wB, zC, wD = T2.hadamard()
        zA, wB, zC, wD = T2._squaring_hadamard_transform(zA, wB, zC, wD)
    else:
        xA, xB, yC, yD = T1.squaring_hadamard()
        zA, wB, zC, wD = T2.squaring_hadamard()

    if verify and any(x.is_zero() for x in (xA, xB, zA, wB, zC, wD)):
        raise ValueError("unexpected splitting detected")

    # Compute projectively A, B, C, D with factor xzwAB
    xAwB = xA * wB
    zAxB = zA * xB
    A = zA * xAwB
    B = wB * zAxB
    C = zC * xAwB
    D = wD * zAxB

    # Compute projectively 1/A, 1/B, 1/C, 1/D by computing ABCDxzw / A
    #  ABCDxzw / B, etc.
    zCwD = zC * wD
    A_inv = xB * zCwD
    B_inv = xA * zCwD
    C_inv = D
    D_inv = C

    # Set the inverse dual points as ThetaPoints
    OB_dual_inv = ThetaPoint(A_inv, B_inv, C_inv, D_inv)

    # For verification, we check whether P, Q as defined below
    # are isotropic
    if verify:
        P = ThetaPoint(xA, xB, yC, yD)
        Q = ThetaPoint(zA, wB, zC, wD)
        if not check_isotropic(P, Q, OB_dual_inv):
            raise ValueError("P and Q are not isotropic")

    # Compute the Hadamard transform on the dual to get the OB the theta null point
    if h2:
        a2, b2, c2, d2 = ThetaPoint._hadamard_transform(A, B, C, D)
    else:
        a2, b2, c2, d2 = A, B, C, D
    OB = ThetaNullPoint(a2, b2, c2, d2)

    return OB_dual_inv, OB


def generic_codomain_with_four_torsion(
    T1: ThetaPoint, OA: ThetaPoint, h1: bool = False, h2: bool = True
):
    """
    Algorithm 8.33 of the SQISign spec.

    Given points T1 such that [2]T1 in ker(Phi) and the domain OA

    Compute:
      - The dual isogenous theta null point (A : B : C : D) and
      - The inverse of the dual isogenous theta null point (A^-1 : B^-1 : C^-1 : D^-1)
      - The theta null point on B

    Cost 8S + 17M + 24a + 2 sqrt
    """
    # Compute H . S(Ti)
    if h1:
        xAB, t0, xCD, t1 = T1.hadamard()
        xAB, _, xCD, _ = T1._squaring_hadamard_transform(xAB, t0, xCD, t1)
        AA, BB, CC, DD = OA.hadamard()
        AA, BB, CC, DD = OA._squaring_hadamard_transform(AA, BB, CC, DD)
    else:
        xAB, _, xCD, _ = T1.squaring_hadamard()
        AA, BB, CC, DD = OA.squaring_hadamard()

    # Compute AB and AC from square roots
    AB = sqrt_fp2(AA * BB)
    AC = sqrt_fp2(AA * CC)

    # Compute dual coordinates of codomain
    B = AB * AC
    D_inv = B * xCD
    B = B * xAB
    D = xCD * AB * AA
    A = xAB * AA
    C = A * CC
    A = A * AC

    # Compute inverse elements
    A_inv = xAB * DD
    C_inv = A_inv * BB
    A_inv = A_inv * CC
    B_inv = A_inv * AB
    A_inv = A_inv * BB
    C_inv = C_inv * AC
    D_inv = D_inv * BB

    # Set inverse dual points as ThetaPoints for later evaluation
    OB_dual_inv = ThetaPoint(A_inv, B_inv, C_inv, D_inv)

    # Compute the Hadamard transform on the dual to get the OB the theta null point
    if h2:
        a2, b2, c2, d2 = T1._hadamard_transform(A, B, C, D)
    else:
        a2, b2, c2, d2 = A, B, C, D

    OB = ThetaNullPoint(a2, b2, c2, d2)

    return OB_dual_inv, OB


def generic_codomain(OA: ThetaPoint, h1: bool = False, h2: bool = True):
    """
    Algorithm 8.34 of the SQISign spec.

    Given the domain OA compute a (2, 2) isogeny

    Compute:
      - The dual isogenous theta null point (A : B : C : D) and
      - The inverse of the dual isogenous theta null point (A^-1 : B^-1 : C^-1 : D^-1)
      - The theta null point on B

    Cost 4S + 10M + 16a + 3 sqrt
    """
    # Compute H . S(OA)
    if h1:
        AA, BB, CC, DD = OA.hadamard()
        AA, BB, CC, DD = OA._squaring_hadamard_transform(AA, BB, CC, DD)
    else:
        AA, BB, CC, DD = OA.squaring_hadamard()

    # Compute dual coordinates of codomain with square roots
    A = AA
    B = sqrt_fp2(AA * BB)
    C = sqrt_fp2(AA * CC)
    D = sqrt_fp2(AA * DD)

    # Compute inverse elements projectively
    A_inv = CC * DD
    B_inv = A_inv * B
    A_inv = A_inv * BB
    C_inv = DD * BB * C
    D_inv = CC * BB * D

    # Set inverse dual points as ThetaPoints for later evaluation
    OB_dual_inv = ThetaPoint(A_inv, B_inv, C_inv, D_inv)

    # Compute the Hadamard transform on the dual to get the OB the theta null point
    if h2:
        a2, b2, c2, d2 = OA._hadamard_transform(A, B, C, D)
    else:
        a2, b2, c2, d2 = A, B, C, D
    OB = ThetaNullPoint(a2, b2, c2, d2)

    return OB_dual_inv, OB


def generic_eval(P: ThetaPoint, I: ThetaPoint, h1: bool = False, h2: bool = True):
    """
    Algorithm 8.35 of the SQISign spec.

    Given a theta point P on OA and the inverse of the dual null point
    (A^-1 : B^-1 : C^-1 : D^-1) for OB, compute the image of Phi: OA -> OB
    on P

    Cost: 4S + 4M + 16a
    """
    # Compute H . S (P)
    if h1:
        XP, YP, ZP, WP = P.hadamard()
        XP, YP, ZP, WP = P._squaring_hadamard_transform(XP, YP, ZP, WP)
    else:
        XP, YP, ZP, WP = P.squaring_hadamard()

    # Multiply by precomputed constants
    A_inv, B_inv, C_inv, D_inv = I.coords()
    XP = A_inv * XP
    YP = B_inv * YP
    ZP = C_inv * ZP
    WP = D_inv * WP

    if h2:
        im_xP, im_yP, im_zP, im_wP = P._hadamard_transform(XP, YP, ZP, WP)
    else:
        im_xP, im_yP, im_zP, im_wP = XP, YP, ZP, WP
    return ThetaPoint(im_xP, im_yP, im_zP, im_wP)
