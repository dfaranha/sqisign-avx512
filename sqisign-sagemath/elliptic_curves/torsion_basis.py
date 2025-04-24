from utilities.deterministic_sqrt import sqrt_fp2


def lift_basis_to_jacobian(basis, A):
    """
    Given an x-only basis, lift to <P, Q> in Jacobian coordinates
    """
    # Unpack points from basis
    xP, xQ, xPQ = basis

    # Unpack coordinates and normalise P = (PX : 1)
    PX = xP.x()
    QX, QZ = xQ.XZ()
    PQX, PQZ = xPQ.XZ()

    # compute PY with a square root
    PYPY = PX * (PX**2 + PX * A + 1)
    PY = sqrt_fp2(PYPY)
    P = (PX, PY, 1)

    # compute QY with Okeya-Sakurai algorithm
    v1 = PX * QZ
    v2 = QX + v1
    v3 = QX - v1
    v3 = v3 * v3
    v3 = v3 * PQX
    v1 = A + A
    v1 = v1 * QZ
    v2 = v2 + v1
    v4 = PX * QX
    v4 = v4 + QZ
    v2 = v2 * v4
    v1 = v1 * QZ
    v2 = v2 - v1
    v2 = v2 * PQZ
    QY = v3 - v2
    v1 = PY + PY
    v1 = v1 * QZ
    v1 = v1 * PQZ
    QX = QX * v1
    QZ = QZ * v1
    v1 = QZ * QZ
    QY = QY * v1
    QX = QX * QZ
    Q = (QX, QY, QZ)

    return P, Q


def projective_difference(xP, xQ, A):
    """
    Algorithm 8.10 of the SQISign spec.

    Given (XP : ZP) and (XQ : ZQ) compute xPQ which is
    equal to x(P ± Q) with a deterministic choice

    WARNING:

    Because of the deterministic nature of sqrt_fp2 this function
    depends on the projective factors of xP, xQ despite the design
    intention of this function.

    To get the matching projective_difference() to align with KAT
    values, we need to have the exact (X : Z) values as are computed
    in the C code.
    """
    L = xP.parent()
    XP, ZP = xP.XZ()
    XQ, ZQ = xQ.XZ()

    b_xx = (XP * XQ - ZP * ZQ) ** 2
    b_xz = (XP * XQ + ZP * ZQ) * (XP * ZQ + ZP * XQ) + 2 * A * XP * XQ * ZP * ZQ
    b_zz = (XP * ZQ - ZP * XQ) ** 2

    lam = (ZP.conjugate() * ZQ.conjugate()) ** 2
    b_xx = lam * b_xx
    b_xz = lam * b_xz
    b_zz = lam * b_zz

    delta = sqrt_fp2(b_xz**2 - b_xx * b_zz)
    xPQ = L([(delta + b_xz), b_zz])

    return xPQ

def _full_even_point_from_nqr(E):
    """
    Helper function which finds an element suitable
    as an x-coordinate with full even torsion when
    the Montgomery coefficient A is a square in F
    """
    A = E.a()
    Fp2 = E.base_ring()
    Fp = Fp2.base_ring()
    h = 0
    while True:
        h = h + 1
        if not Fp(1 + h**2).is_square():
            xP = -A /  Fp2([1, h])
            if E.is_x_coordinate(xP):
                return xP, h

def _full_even_point_from_A(E):
    """
    Helper function which finds an element suitable
    as an x-coordinate with full even torsion when
    the Montgomery coefficient A is not a square
    """
    A = E.a()
    h, xP = 0, 0
    while True:
        h = h + 1
        xP = xP + A
        if E.is_x_coordinate(xP):
            return xP, h

def torsion_basis_to_hint(E, e):
    """
    Algorithm 2.1 of the SQISign spec.

    Compute a deterministic basis for E[2^e]  <P, Q> together with a hint
    used for an optimised algorithm torsion_basis_from_hint used during
    verification
    """
    Fp2 = E.base_ring()
    p = Fp2.characteristic()
    A = E.a()

    hint_A = int(A.is_square())
    if hint_A:
        xP, hint_P = _full_even_point_from_nqr(E)
    else:
        xP, hint_P = _full_even_point_from_A(E)
    
    # Compute the second x-coord of a point of full even torsion not above (0 : 0)
    xPQ = - (xP + A)

    # Clear the cofactor to get points of order 2^e
    E.normalize_curve_constants()
    cofactor = (p + 1) // (2**e)
    xP, xPQ = cofactor * E(xP), cofactor * E(xPQ)
    
    # Ensure the relative sign matches specification
    # This is very sensitive to projective factors
    xQ = projective_difference(xP, xPQ, A)

    # We only allow 7-bit hints, in the unlikely case
    # that we needed this many tries, we fall back to
    # generating points on the fly during verification
    if hint_P > 127:
        hint_P = 0

    # Pack the hint into one 8-bit value
    hint = 2*hint_P + hint_A
    assert hint_A < 256

    return (xP, xQ, xPQ), hint

def torsion_basis_from_hint(E, hint, e):
    """
    Algorithm 2.2 of the SQISign spec.

    Compute a deterministic basis for E[2^e]  <P, Q> given a hint
    which is used to optimise the computation of x(P) and x(Q).
    """
    Fp2 = E.base_ring()
    p = Fp2.characteristic()
    A = E.a()

    hint_A = hint & 1
    hint_P = hint // 2

    # debugging
    assert hint_A == A.is_square()

    # When hint_P is zero, we have an edge case where the hint did not
    # work and we need to find our own basis
    if hint_P:
        if hint_A:
            xP = -A / Fp2([1, hint_P])
        else:
            xP = hint_P * A
    else:
        if hint_A:
            xP, _ = _full_even_point_from_nqr(E)
        else:
            xP, _ = _full_even_point_from_A(E)
    
    # Compute the second x-coord of a point of full even torsion not above (0 : 0)
    xQ = -(xP + A)

    # Clear the cofactor with x-only arithmetic
    E.normalize_curve_constants()
    cofactor = (p + 1) // (2**e)
    xP, xQ = cofactor * E(xP), cofactor * E(xQ)

    # Ensure the relative sign matches specification
    # This is very sensitive to projective factors
    xPQ = projective_difference(xP, xQ, A)

    return xP, xPQ, xQ
