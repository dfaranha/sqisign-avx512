from sage.all import ZZ

from elliptic_types import Curve, BasisX, PointX, PointJ
from avx_types import AVX_4x2x1, AVX_2x4x1, AVX_8x1x1
from avx_arithmetic import (
    avx_421_sqr,
    avx_421_mul,
    avx_421_add,
    avx_421_sub,
    avx_421_dbl,
    avx_421_pair_sqr,
    avx_421_pair_mul,
    avx_241_add_sub,
    avx_241_mul,
    avx_241_sqr,
    avx_241_add,
    avx_421_add_sub,
    avx_421_had,
    avx_241_to_421,
    avx_421_to_241,
    avx_241_to_811,
    avx_811_to_241,
    avx_811_add_sub,
    avx_811_had,
    avx_811_mul,
    avx_811_sqr,
)
from utilities import sqrt_fp2, batched_inversion

# ======================================================================
# 64 bit operations
# ======================================================================


def j_invariant(E: Curve):
    """
    Compute the j-invariant of the curve
    """
    A, C = E.A, E.C
    j_num = 256 * (A**2 - 3 * C**2) ** 3
    j_den = C**4 * (A**2 - 4 * C**2)
    return j_num / j_den


def x_diff_add(XP, ZP, XQ, ZQ, xPQ, zPQ):
    """
    Algorithm 8.4 of the SQISign spec.

    Montgomery differential addition

    Input:  projective coordinates P = (XP : ZP),
            Q=(XQ : ZQ), and their difference
            x(P-Q) = (xPQ : zPQ)
    Output: coordinates of sum P + Q = (XQP : ZQP)

    Cost: 4M + 2S + 6a
    """
    t0 = XP + ZP
    t1 = XP - ZP
    t2 = XQ + ZQ
    t3 = XQ - ZQ
    t0 = t0 * t3
    t1 = t1 * t2
    t2 = t0 + t1
    t3 = t0 - t1
    t2 = t2 * t2
    t3 = t3 * t3
    XPQ = zPQ * t2
    ZPQ = xPQ * t3

    return XPQ, ZPQ


def x_double(X, Z, A24, C24):
    """
    Algorithm 8.3 of the SQISign spec.

    Montgomery doubling with projective curve constant

    Input:  projective point P = (X:Z), curve constants (A24 : C24)
    Output: projective point [2]P = (X2 : Z2)

    Cost: 4M + 2S + 4a
    """
    t0 = X + Z
    t0 = t0 * t0
    t1 = X - Z
    t1 = t1 * t1
    t2 = t0 - t1
    t1 = t1 * C24
    X2 = t0 * t1
    t0 = t2 * A24
    t0 = t0 + t1
    Z2 = t0 * t2

    return X2, Z2


def x_double_n(E: Curve, P: PointX, n: int) -> PointX:
    """
    Compute [2^n]P using repeated doublings
    """
    PX, PZ = P
    A24, C24 = E.A24, E.C24
    for _ in range(n):
        PX, PZ = x_double(PX, PZ, A24, C24)
    return PointX(PX, PZ)


def x_double_add(XP, ZP, XQ, ZQ, xPQ, zPQ, A24, C24):
    """
    Algorithm 8.5 of the SQISign spec.

    Montgomery ladder for simultaneous doubling and differential
    addition

    Input: projective coordinates P=(XP:ZP) and Q=(XQ:ZQ),
           projective difference P-Q=(xPQ:zPQ) and
           curve constant A24/C24=(A+2C)/4C.
    Output: projective coordinates of 2P=(X2P:Z2P)
            and Q+P=(XQP:ZQP)

    Cost: 8M + 4S + 8a
    """

    t0 = XP + ZP
    t1 = XP - ZP
    X2P = t0 * t0
    t2 = XQ - ZQ
    XQP = XQ + ZQ
    t0 = t0 * t2
    Z2P = t1 * t1
    t1 = t1 * XQP
    t2 = X2P - Z2P
    Z2P = Z2P * C24
    X2P = X2P * Z2P
    XQP = A24 * t2
    ZQP = t0 - t1
    Z2P = XQP + Z2P
    XQP = t0 + t1
    Z2P = Z2P * t2
    ZQP = ZQP * ZQP
    XQP = XQP * XQP
    ZQP = xPQ * ZQP
    XQP = XQP * zPQ

    return X2P, Z2P, XQP, ZQP


def x_mul(E: Curve, P: PointX, m: ZZ) -> PointX:
    try:
        m = ZZ(m)
    except TypeError:
        raise TypeError(f"Cannot coerce input scalar {m = } to an integer")

    # If m is zero, return identity
    R = E.A.parent()
    if not m:
        return PointX(R.zero(), R.zero())

    # [m]P = [-m]P for x-only
    m = abs(m)

    # Initialise for loop
    XP, ZP = P
    X0, Z0 = R.one(), R.zero()
    X1, Z1 = XP, ZP

    # Extract out A24 and C24 from parent curve
    A24, C24 = E.A24, E.C24

    # Montgomery-ladder
    for bit in bin(m)[2:]:
        if bit == "1":
            X1, Z1, X0, Z0 = x_double_add(X1, Z1, X0, Z0, XP, ZP, A24, C24)
        else:
            X0, Z0, X1, Z1 = x_double_add(X0, Z0, X1, Z1, XP, ZP, A24, C24)

    return PointX(X0, Z0)


def mul_simple(A, A24, C24, X, Y, Z, m):
    """
    This needs to be parallel... but just for testing right now
    """
    R = X.parent()

    # Convert from Jacobian to x-only Montgomery
    xP, zP = X, Z * Z

    # Set up the ladder variables
    X0, Z0 = R.one(), R.zero()
    X1, Z1 = xP, zP

    # x-only Montgomery-ladder
    for bit in bin(m)[2:]:
        if bit == "1":
            X1, Z1, X0, Z0 = x_double_add(X1, Z1, X0, Z0, xP, zP, A24, C24)
        else:
            X0, Z0, X1, Z1 = x_double_add(X0, Z0, X1, Z1, xP, zP, A24, C24)

    # Use projective instead of Jacobian coordinates here
    PX = X * Z
    PY = Y
    PZ = Z * Z * Z

    # Recover the y-coordinate in projective coordinates
    t0 = (PX * X0) + (PZ * Z0)
    t1 = PX * Z0
    t2 = X0 * PZ
    t3 = PZ * Z0
    t4 = (A + A) * t3
    u = (t0 * (t1 + t2 + t4)) - t4 * t3
    v = (PY + PY) * t3 * Z1

    RX = X0 * v
    RY = (u * Z1) - ((t1 - t2) ** 2 * X1)
    RZ = Z0 * v

    # Recover the Jacobian coordinates
    RX *= RZ
    RY *= RZ**2

    return PointJ(RX, RY, RZ)


def add_simple(A, P, Q):
    """
    Given two Jacobian points P, Q with coordinates (X : Y : Z) and the Montgomery coefficient A
    computes R = P + Q with complete formulas

    TODO:

    This needs to be modified to be parallel for a pair of sums...

    Cost: 6S + 17M + 13a
    """
    # Unpack coordinates
    XP, YP, ZP = P
    XQ, YQ, ZQ = Q

    # Cases for P, Q zero handled at end of function to mimic constant time impl
    ctl_1 = ZP.is_zero()
    ctl_2 = ZQ.is_zero()

    # Compute the slope dy / dx for the case P != Q
    t0 = ZP * ZP
    t1 = ZQ * ZQ
    v1 = t1 * ZQ
    t2 = t0 * ZP
    v1 = v1 * YP
    t2 = t2 * YQ
    dy = t2 - v1
    u2 = t0 * XQ
    u1 = t1 * XP
    dx = u2 - u1

    # Compute the slope f'(x) for the case P = Q
    t1 = YP + YP
    t2 = A * XP
    t2 = t2 + t2
    t2 = t2 + t0
    t2 = t2 * t0
    t0 = XP * XP
    t2 = t2 + t0
    t2 = t2 + t0
    t2 = t2 + t0
    t2 = t2 * ZQ

    # If dx == dy == 0 we have P = Q and we use the slope from f'(x)
    # NOTE: for constant time impl, the swap here must be constant time
    ctl = dx.is_zero() and dy.is_zero()
    if ctl:
        dx, dy = t1, t2

    # Compute the new coordinates
    t0 = ZP * ZQ
    t1 = t0 * t0
    t2 = dx * dx
    t3 = dy * dy

    XR = A * t1
    XR = XR + u1
    XR = XR + u2
    XR = XR * t2
    XR = t3 - XR
    YR = u1 * t2
    YR = YR - XR
    YR = dy * YR
    t3 = dx * t2
    t3 = t3 * v1
    YR = YR - t3
    ZR = dx * t0

    # Set the point R with Jacobian coordinates
    R = PointJ(XR, YR, ZR)

    # Handle the cases where P or Q is zero
    # NOTE: for constant time impl, the swap here must be constant time
    if ctl_1:
        R = Q
    if ctl_2:
        R = P
    return R


def ladder_3_pt(E: Curve, basis: BasisX, m: ZZ):
    """
    Algorithm 8.7 of the SQISign spec.

    Compute xP + [m]xQ using x-only arithmetic.
    """
    try:
        m = ZZ(m)
    except TypeError:
        raise TypeError(f"Cannot coerce input scalar {m = } to an integer")

    # If m is zero, return xP
    if not m:
        return PointX(basis.PX, basis.PZ)

    # [m]P = [-m]P for x-only
    m = abs(m)

    # Converting parameters for projective DBLADD -> (A24:C24)=(A+2C:4C)
    A24, C24 = E.A24, E.C24

    # Extract out coordinates
    X0, Z0 = basis.PX, basis.PZ
    X1, Z1 = basis.QX, basis.QZ
    X2, Z2 = basis.PmQX, basis.PmQZ

    # Montgomery-ladder
    for bit in bin(m)[:1:-1]:
        if bit == "1":
            X1, Z1, X0, Z0 = x_double_add(X1, Z1, X0, Z0, X2, Z2, A24, C24)
        else:
            X1, Z1, X2, Z2 = x_double_add(X1, Z1, X2, Z2, X0, Z0, A24, C24)
    return PointX(X0, Z0)


def ladder_biscalar(E: Curve, basis: BasisX, a: ZZ, b: ZZ):
    s0 = a
    s1 = b

    A24, C24 = E.A24, E.C24
    X0, Z0 = basis.PX, basis.PZ
    X1, Z1 = basis.QX, basis.QZ
    XD, ZD = basis.PmQX, basis.PmQZ

    while s0 != 0:
        if s1 < s0:
            s0, s1 = s1, s0
            X0, Z0, X1, Z1 = X1, Z1, X0, Z0

        assert s1 >= s0

        if s1 <= 4 * s0:
            s1 = s1 - s0
            R0, S0 = x_diff_add(X1, Z1, X0, Z0, XD, ZD)
            X0, Z0, XD, ZD = R0, S0, X0, Z0

        elif s0 % 2 == s1 % 2:
            s1 = (s1 - s0) // 2
            R0, S0, R1, S1 = x_double_add(X1, Z1, X0, Z0, XD, ZD, A24, C24)
            X0, Z0, X1, Z1 = R1, S1, R0, S0

        elif s1 % 2 == 0:
            s1 = s1 // 2
            R0, S0, R1, S1 = x_double_add(X1, Z1, XD, ZD, X0, Z0, A24, C24)
            X1, Z1, XD, ZD = R0, S0, R1, S1
        else:
            assert s0 % 2 == 0
            s0 = s0 // 2
            R0, S0, R1, S1 = x_double_add(X0, Z0, XD, ZD, X1, Z1, A24, C24)
            X0, Z0, XD, ZD = R0, S0, R1, S1

    while s1 & 1 == 0:
        s1 //= 2
        X1, Z1 = x_double(X1, Z1, A24, C24)

    res = PointX(X1, Z1)
    return x_mul(E, res, s1)


def lift_normalised_basis_to_jacobian(E: Curve, basis: BasisX) -> tuple[PointJ, PointJ]:
    """
    Given an x-only basis, lift to <P, Q> in Jacobian coordinates
    """
    # Unpack coordinates and normalise P = (PX : 1)
    PX = basis.PX
    assert basis.PZ == 1
    QX, QZ = basis.QX, basis.QZ
    PQX, PQZ = basis.PmQX, basis.PmQZ

    # We need the Montgomery coefficient from the curve
    A = E.A
    assert E.C == 1

    # compute PY with a square root
    PYPY = PX * (PX**2 + PX * A + 1)
    PY = sqrt_fp2(PYPY)
    P = PointJ(PX, PY, 1)

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
    Q = PointJ(QX, QY, QZ)

    return P, Q


# ======================================================================
# AVX operations
# ======================================================================


def x_double_1x2x4x1(A24: AVX_2x4x1, P: AVX_2x4x1) -> AVX_2x4x1:
    """
    Double a single x-only point using 2x4x1 multiplications
    """
    T0 = AVX_2x4x1(P[0], P[0])
    T1 = AVX_2x4x1(P[1], P[1])
    t0_s0 = avx_241_add_sub(T0, T1)
    t1_s1 = avx_241_sqr(t0_s0)

    T0 = AVX_2x4x1(t1_s1[0], t1_s1[0])
    T1 = AVX_2x4x1(t1_s1[1], t1_s1[1])
    t0_s0 = avx_241_add_sub(T0, T1)

    T0 = AVX_2x4x1(A24[1], A24[0])
    T1 = AVX_2x4x1(t1_s1[1], t0_s0[1])
    t2_s2 = avx_241_mul(T0, T1)

    T0 = AVX_2x4x1(t2_s2[0], 0)
    T1 = AVX_2x4x1(t2_s2[1], 0)
    t3_00 = avx_241_add(T0, T1)

    T0 = AVX_2x4x1(t2_s2[0], t3_00[0])
    T1 = AVX_2x4x1(t1_s1[0], t0_s0[1])
    P_dbl = avx_241_mul(T0, T1)

    return P_dbl


def x_double_n_1x2x4x1(A24: AVX_2x4x1, P: AVX_2x4x1, n: int) -> AVX_2x4x1:
    """
    Repeated doubling of a point in AVX_2x4x1 representation
    """
    for _ in range(n):
        P = x_double_1x2x4x1(A24, P)
    return P


def x_double_4x4x2x1(A24: AVX_4x2x1, X: AVX_4x2x1, Z: AVX_4x2x1):
    """
    Simultaneously double four X-only points where the X-coordinates
    of P1, Q1 P2, Q2 on E1 x E2 are stored within X (and Z resp.)
    Here we assume C24 = 1
    """
    t0 = avx_421_add(X, Z)
    t1 = avx_421_sub(X, Z)
    t0, t1 = avx_421_pair_sqr(t0, t1)
    t2 = avx_421_sub(t0, t1)
    X2, t0 = avx_421_pair_mul(t0, t1, t2, A24)
    t0 = avx_421_add(t0, t1)
    Z2 = avx_421_mul(t0, t2)

    return X2, Z2


def x_basis_double_n_1x2x4x1(
    A24: AVX_4x2x1, B: tuple[AVX_2x4x1], n: int
) -> tuple[AVX_2x4x1]:
    """
    Compute [2^n](P, Q, PmQ) using repeated doublings
    """
    P, Q, PmQ = B
    # TODO: should we batch these together using 8x1x1 multiplications instead?
    for _ in range(n):
        P = x_double_1x2x4x1(A24, P)
        Q = x_double_1x2x4x1(A24, Q)
        PmQ = x_double_1x2x4x1(A24, PmQ)
    return P, Q, PmQ


def x_diff_add_1x2x4x1(P: AVX_2x4x1, Q: AVX_2x4x1, PmQ: AVX_2x4x1) -> AVX_2x4x1:
    T0 = AVX_2x4x1(P[0], P[0])
    T1 = AVX_2x4x1(P[1], P[1])
    t0_t1 = avx_241_add_sub(T0, T1)

    T0 = AVX_2x4x1(Q[0], Q[0])
    T1 = AVX_2x4x1(Q[1], Q[1])
    t2_t3 = avx_241_add_sub(T0, T1)

    T0 = AVX_2x4x1(t2_t3[1], t2_t3[0])
    t0_t1 = avx_241_mul(t0_t1, T0)

    T0 = AVX_2x4x1(t0_t1[0], t0_t1[0])
    T1 = AVX_2x4x1(t0_t1[1], t0_t1[1])
    t2_t3 = avx_241_add_sub(T0, T1)
    t2_t3 = avx_241_sqr(t2_t3)

    T0 = AVX_2x4x1(PmQ[1], PmQ[0])
    XPQ_ZPQ = avx_241_mul(t2_t3, T0)

    return XPQ_ZPQ


# this is an alt 4x2x1 which uses 2M 1S, but uses a normalised A instead of (A24 : C24)
# def x_double_add_1x2x4x1(
#     A24: AVX_2x4x1, P: AVX_2x4x1, Q: AVX_2x4x1, PmQ: AVX_2x4x1
# ) -> tuple[AVX_2x4x1]:
#     # Requires A and not A24
#     a24 = A24[0] / A24[1]
#     a = 4 * a24 - 2

#     X2, Z2 = P
#     X3, Z3 = Q
#     X1, Z1 = PmQ

#     A0 = X3 + Z3
#     B0 = X3 - Z3
#     C0 = X2 + Z2
#     D0 = X2 - Z2

#     A1 = A0 * D0
#     B1 = B0 * C0
#     C1 = C0 * C0
#     D1 = D0 * D0

#     A2 = A1 + B1
#     B2 = A1 - B1
#     C2 = C1 + D1
#     D2 = C1 - D1

#     A3 = A2**2
#     B3 = B2**2
#     C3 = C2**2
#     D3 = D2**2

#     A4 = A3 * Z1
#     B4 = B3 * X1
#     C4 = C2 * D2
#     D4 = D3 * a

#     X5 = A4
#     Z5 = B4
#     X4 = C3 - D3
#     Z4 = 2 * C4 + D4

#     return AVX_2x4x1(X4, Z4), AVX_2x4x1(X5, Z5)


def x_double_add_1x2x4x1(P_Q: AVX_4x2x1, PmQ_A24: AVX_4x2x1) -> AVX_4x2x1:
    # First double add sub
    T0 = AVX_4x2x1(P_Q[0], P_Q[2], P_Q[0], P_Q[2])
    T1 = AVX_4x2x1(P_Q[1], P_Q[3], P_Q[1], P_Q[3])
    S0 = avx_421_add_sub(T0, T1)

    # 4x2x1 mul with two squares hidden inside
    T0 = AVX_4x2x1(S0[0], S0[2], S0[0], S0[2])
    T1 = AVX_4x2x1(S0[0], S0[2], S0[3], S0[1])
    S1 = avx_421_mul(T0, T1)

    # Second double add sub
    T0 = AVX_4x2x1(S1[0], S1[2], S1[0], S1[2])
    T1 = AVX_4x2x1(S1[1], S1[3], S1[1], S1[3])
    S2 = avx_421_add_sub(T0, T1)

    # 4x2x1 mul with two squares hidden inside
    T0 = AVX_4x2x1(PmQ_A24[3], PmQ_A24[2], S2[1], S2[3])
    T1 = AVX_4x2x1(S1[1], S2[2], S2[1], S2[3])
    S3 = avx_421_mul(T0, T1)

    # Loose end addition which uses a 2x4x1 call
    # TODO: should we just do one 4x1x1 add with no conversion?
    t2_s2, _ = avx_421_to_241(S3)
    T0 = AVX_2x4x1(t2_s2[0], 0)
    T1 = AVX_2x4x1(t2_s2[1], 0)
    t3_00 = avx_241_add(T0, T1)

    # Convert back to 4x2x1 after the 2x4x1 addition
    t3_00_00_00 = AVX_4x2x1(t3_00[0], 0, 0, 0)

    # Final multiplication
    T0 = AVX_4x2x1(S3[0], t3_00_00_00[0], S3[2], S3[3])
    T1 = AVX_4x2x1(S1[0], S2[2], PmQ_A24[1], PmQ_A24[0])
    PP_PQ = avx_421_mul(T0, T1)

    return PP_PQ


# Requires normalized A coordinate 4* (A24 / C4) - 2
def x_double_add_normA_1x2x4x1(P_Q: AVX_4x2x1, PmQ_A: AVX_4x2x1) -> AVX_4x2x1:
    # First had
    T0 = AVX_4x2x1(P_Q[0], P_Q[0], P_Q[2], P_Q[2])
    T1 = AVX_4x2x1(P_Q[1], P_Q[1], P_Q[3], P_Q[3])
    S0 = avx_421_had(T0, T1)

    T0 = AVX_4x2x1(S0[0], S0[1], S0[1], S0[0])
    S1 = avx_421_mul(S0, T0)

    T0 = AVX_4x2x1(S1[0], S1[0], S1[2], S1[2])
    T1 = AVX_4x2x1(S1[1], S1[1], S1[3], S1[3])
    S2 = avx_421_had(T0, T1)

    S3 = avx_421_sqr(S2)

    T0 = AVX_4x2x1(S2[0], S3[1], S3[2], S3[3])
    T1 = AVX_4x2x1(S2[1], PmQ_A[2], PmQ_A[1], PmQ_A[0])
    S4 = avx_421_mul(T0, T1)

    # figure out how to do this efficiently in C
    PP_PQ = AVX_4x2x1(S3[0] - S3[1], 2 * S4[0] + S4[1], S4[2], S4[3])
    return PP_PQ


def x_double_add_pair_8x1x1(PQDA1: AVX_8x1x1, PQDA2: AVX_8x1x1):
    # PQDA contains the 8 Fp2 elements
    # (PX : PZ : QX : QZ : PmQZ : PmQZ : A24 : C24)

    # Redundant info: A24 is at the end of both
    assert PQDA1[6] == PQDA2[6]
    assert PQDA1[7] == PQDA2[7]

    # First quad add sub (+ + + + - - - -)
    T0 = AVX_8x1x1(
        PQDA1[0], PQDA1[2], PQDA2[0], PQDA2[2], PQDA1[0], PQDA1[2], PQDA2[0], PQDA2[2]
    )
    T1 = AVX_8x1x1(
        PQDA1[1], PQDA1[3], PQDA2[1], PQDA2[3], PQDA1[1], PQDA1[3], PQDA2[1], PQDA2[3]
    )
    S0 = avx_811_add_sub(T0, T1)

    # 8x2x1 mul with four squares hidden inside
    T0 = AVX_8x1x1(S0[0], S0[4], S0[0], S0[4], S0[2], S0[6], S0[2], S0[6])
    T1 = AVX_8x1x1(S0[0], S0[4], S0[5], S0[1], S0[2], S0[6], S0[7], S0[3])
    S1 = avx_811_mul(T0, T1)

    # Second quad add sub (+ + + + - - - -)
    T0 = AVX_8x1x1(S1[0], S1[2], S1[4], S1[6], S1[0], S1[2], S1[4], S1[6])
    T1 = AVX_8x1x1(S1[1], S1[3], S1[5], S1[7], S1[1], S1[3], S1[5], S1[7])
    S2 = avx_811_add_sub(T0, T1)

    # 8x1x1 mul with four squares hidden inside
    T0 = AVX_8x1x1(PQDA1[7], PQDA1[6], S2[1], S2[5], PQDA2[7], PQDA2[6], S2[3], S2[7])
    T1 = AVX_8x1x1(S1[1], S2[4], S2[1], S2[5], S1[5], S2[6], S2[3], S2[7])
    S3 = avx_811_mul(T0, T1)

    # Loose end addition which uses a 2x4x1 call
    # TODO: what's the fastest way to handle this?
    # maybe we should just do a single 8x1x1 add with
    # no conversion?
    A, _, C, _ = avx_811_to_241(S3)
    T0 = AVX_2x4x1(A[0], C[0])
    T1 = AVX_2x4x1(A[1], C[1])
    T0 = avx_241_add(T0, T1)
    t3 = avx_241_to_811(T0, T0, T0, T0)

    # Final multiplication
    T0 = AVX_8x1x1(S3[0], t3[0], S3[2], S3[3], S3[4], t3[1], S3[6], S3[7])
    T1 = AVX_8x1x1(S1[0], S2[4], PQDA1[5], PQDA1[4], S1[4], S2[6], PQDA2[5], PQDA2[4])
    R1_R2_S1_S2 = avx_811_mul(T0, T1)

    return R1_R2_S1_S2


# Requires normalized A coordinate 4* (A24 / C24) - 2
def x_double_add_pair_normA_8x1x1(PQDA1: AVX_8x1x1, PQDA2: AVX_8x1x1) -> AVX_8x1x1:
    # First had
    T0 = AVX_8x1x1(
        PQDA1[0], PQDA1[0], PQDA1[2], PQDA1[2], PQDA2[0], PQDA2[0], PQDA2[2], PQDA2[2]
    )
    T1 = AVX_8x1x1(
        PQDA1[1], PQDA1[1], PQDA1[3], PQDA1[3], PQDA2[1], PQDA2[1], PQDA2[3], PQDA2[3]
    )
    S0 = avx_811_had(T0, T1)

    T0 = AVX_8x1x1(S0[0], S0[1], S0[1], S0[0], S0[4], S0[5], S0[5], S0[4])
    S1 = avx_811_mul(S0, T0)

    T0 = AVX_8x1x1(S1[0], S1[0], S1[2], S1[2], S1[4], S1[4], S1[6], S1[6])
    T1 = AVX_8x1x1(S1[1], S1[1], S1[3], S1[3], S1[5], S1[5], S1[7], S1[7])
    S2 = avx_811_had(T0, T1)

    S3 = avx_811_sqr(S2)

    T0 = AVX_8x1x1(S2[0], S3[1], S3[2], S3[3], S2[4], S3[5], S3[6], S3[7])
    T1 = AVX_8x1x1(
        S2[1], PQDA1[6], PQDA1[5], PQDA1[4], S2[5], PQDA2[6], PQDA2[5], PQDA2[4]
    )
    S4 = avx_811_mul(T0, T1)

    # figure out how to do this efficiently in C
    R1_R2_S1_S2 = AVX_8x1x1(
        S3[0] - S3[1],
        2 * S4[0] + S4[1],
        S4[2],
        S4[3],
        S3[4] - S3[5],
        2 * S4[4] + S4[5],
        S4[6],
        S4[7],
    )
    return R1_R2_S1_S2


def x_mul_normA_1x2x4x1(A: AVX_2x4x1, P: AVX_2x4x1, m: ZZ) -> PointX:
    try:
        m = ZZ(m)
    except TypeError:
        raise TypeError(f"Cannot coerce input scalar {m = } to an integer")

    # If m is zero, return identity
    R = P[0].parent()
    if not m:
        return AVX_2x4x1(R.zero(), R.zero())

    # [m]P = [-m]P for x-only
    m = abs(m)

    # Initialise for loop
    xp = P
    x0 = AVX_2x4x1(R.one(), R.zero())
    x1 = P

    # TODO: memory management directly from P?
    X0_X1 = avx_241_to_421(x0, x1)
    XD_A = avx_241_to_421(xp, A)

    # Helper function for swapping P, Q inside an AVX4x2x1
    def swap(PQ):
        return AVX_4x2x1(PQ[2], PQ[3], PQ[0], PQ[1])

    # Montgomery-ladder, here we only ever have to swap x0, x1
    for bit in bin(m)[2:]:
        if bit == "1":
            X0_X1 = swap(X0_X1)

        X0_X1 = x_double_add_normA_1x2x4x1(X0_X1, XD_A)

        if bit == "1":
            X0_X1 = swap(X0_X1)

    x0, x1 = avx_421_to_241(X0_X1)
    return x0


def mul_pair(A_A24: AVX_2x4x1, P: PointJ, m0: ZZ, m1: ZZ) -> tuple[PointJ]:
    """
    Computes [m0] P and [m1] P given two scalars m0, m1 and a Jacobian
    point P = (X : Y : Z)
    """

    def swap_4x2x1(PQ: AVX_4x2x1) -> AVX_4x2x1:
        return AVX_4x2x1(PQ[2], PQ[3], PQ[0], PQ[1])

    def swap_8x1x1(PQDA: AVX_8x1x1) -> AVX_8x1x1:
        return AVX_8x1x1(
            PQDA[2], PQDA[3], PQDA[0], PQDA[1], PQDA[4], PQDA[5], PQDA[6], PQDA[7]
        )

    A, _ = A_A24
    X, Y, Z = P
    F = X.parent()

    # Convert from Jacobian to x-only Montgomery
    xP, zP = X, Z * Z

    # Set up the ladder variables
    X0, Z0 = F.one(), F.zero()
    X1, Z1 = xP, zP

    X2, Z2 = F.one(), F.zero()
    X3, Z3 = xP, zP

    # x-only Montgomery-ladder

    # We don't know that len(m0) = len(m1) so we do the longer one on its own
    # first... maybe there's a more elegant solution?
    bin_m0 = bin(m0)[2:]
    bin_m1 = bin(m1)[2:]
    l_m0 = len(bin_m0)
    l_m1 = len(bin_m1)

    if l_m0 > l_m1:
        top = l_m0 - l_m1
        top_bits, bin_m0 = bin_m0[:top], bin_m0[top:]

        P_Q = AVX_4x2x1(X0, Z0, X1, Z1)
        PmQ_A = AVX_4x2x1(xP, zP, A, 0)

        for i in range(top):
            if top_bits[i] == "1":
                P_Q = swap_4x2x1(P_Q)
            P_Q = x_double_add_normA_1x2x4x1(P_Q, PmQ_A)
            if top_bits[i] == "1":
                P_Q = swap_4x2x1(P_Q)

        X0, Z0, X1, Z1 = P_Q

    elif l_m1 > l_m0:
        top = l_m1 - l_m0
        top_bits, bin_m1 = bin_m1[:top], bin_m1[top:]

        P_Q = AVX_4x2x1(X2, Z2, X3, Z3)
        PmQ_A = AVX_4x2x1(xP, zP, A, 0)

        for i in range(top):
            if top_bits[i] == "1":
                P_Q = swap_4x2x1(P_Q)
            P_Q = x_double_add_normA_1x2x4x1(P_Q, PmQ_A)
            if top_bits[i] == "1":
                P_Q = swap_4x2x1(P_Q)

        X2, Z2, X3, Z3 = P_Q

    assert len(bin_m0) == len(bin_m1)

    P1_data = AVX_8x1x1(X0, Z0, X1, Z1, xP, zP, A, 0)
    P2_data = AVX_8x1x1(X2, Z2, X3, Z3, xP, zP, A, 0)

    for i, j in zip(bin_m0, bin_m1):
        if i == "1":
            P1_data = swap_8x1x1(P1_data)
        if j == "1":
            P2_data = swap_8x1x1(P2_data)

        result = x_double_add_pair_normA_8x1x1(P1_data, P2_data)

        P1_data = AVX_8x1x1(
            result[0],
            result[1],
            result[2],
            result[3],
            P1_data[4],
            P1_data[5],
            P1_data[6],
            P1_data[7],
        )
        P2_data = AVX_8x1x1(
            result[4],
            result[5],
            result[6],
            result[7],
            P2_data[4],
            P2_data[5],
            P2_data[6],
            P2_data[7],
        )

        if i == "1":
            P1_data = swap_8x1x1(P1_data)
        if j == "1":
            P2_data = swap_8x1x1(P2_data)

    X0, Z0, X1, Z1 = P1_data[0], P1_data[1], P1_data[2], P1_data[3]
    X2, Z2, X3, Z3 = P2_data[0], P2_data[1], P2_data[2], P2_data[3]

    # TODO: theres an easy 2-way parallel for the below, but maybe we can do better

    # Use projective instead of Jacobian coordinates here
    PX = X * Z
    PY = Y
    PZ = Z * zP

    # Recover the y-coordinate in projective coordinates
    tt = PX * X0
    t0 = PZ * Z0
    t1 = PX * Z0
    t2 = X0 * PZ
    t3 = PZ * Z0
    t0 = t0 + tt
    t4 = A + A
    t4 = t4 * t3
    t5 = t1 + t2
    t5 = t5 + t4
    t5 = t5 * t0
    t6 = t4 * t3
    u0 = t5 - t6
    v0 = PY + PY
    v0 = v0 * t3
    v0 = v0 * Z1

    ss = PX * X2
    s0 = PZ * Z2
    s1 = PX * Z2
    s2 = X2 * PZ
    s3 = PZ * Z2
    s0 = s0 + ss
    s4 = A + A
    s4 = s4 * s3
    s5 = s1 + s2
    s5 = s5 + s4
    s5 = s5 * s0
    s6 = s4 * s3
    u1 = s5 - s6
    v1 = PY + PY
    v1 = v1 * s3
    v1 = v1 * Z3

    # Projective points
    # TODO: i think we can save a mul or two here because of the Jacobian
    # conversion...

    RX = X0 * v0
    t0 = t1 - t2
    t0 = t0**2
    t0 = t0 * X1
    RY = u0 * Z1
    RY = RY - t0
    RZ = Z0 * v0

    SX = X2 * v1
    s0 = s1 - s2
    s0 = s0**2
    s0 = s0 * X3
    SY = u1 * Z3
    SY = SY - s0
    SZ = Z2 * v1

    # Recover the Jacobian coordinates
    t0 = RZ**2
    s0 = SZ**2
    RX *= RZ
    RY *= t0
    SX *= SZ
    SY *= s0

    R = PointJ(RX, RY, RZ)
    S = PointJ(SX, SY, SZ)

    return R, S


def ladder_3pt_1x2x4x1(A24: AVX_2x4x1, basis: tuple[AVX_2x4x1], m: ZZ) -> PointX:
    try:
        m = ZZ(m)
    except TypeError:
        raise TypeError(f"Cannot coerce input scalar {m = } to an integer")

    # [m]P = [-m]P for x-only
    m = abs(m)

    # Unpack the basis
    x0, x1, xd = basis

    # If m is zero, return xP
    if not m:
        return x0

    # Helper function for swapping P, Q inside an AVX4x2x1
    def swap(PQ):
        return [PQ[2], PQ[3], PQ[0], PQ[1]]

    X0_X1 = list(avx_241_to_421(x0, x1))
    XD_A24 = list(avx_241_to_421(xd, A24))

    # Montgomery-ladder
    for bit in bin(m)[:1:-1]:
        if bit == "1":
            # We can do this with simple swaps in x0, x1 only
            X1_X0 = swap(X0_X1)
            X1_X0 = x_double_add_1x2x4x1(X1_X0, XD_A24)
            X0_X1 = swap(X1_X0)
        else:
            # Here we use some tmp buffers to store x1, xd and x0 a24
            # but you could just read in a different order from
            # x_double_add_1x2x4x1 i suppose...
            X1_XD = (X0_X1[2], X0_X1[3], XD_A24[0], XD_A24[1])
            X0_A24 = (X0_X1[0], X0_X1[1], XD_A24[2], XD_A24[3])
            X1_XD = x_double_add_1x2x4x1(X1_XD, X0_A24)

            # Update X1 and XD from scratch vars
            X0_X1[2], X0_X1[3] = X1_XD[0], X1_XD[1]
            XD_A24[0], XD_A24[1] = X1_XD[2], X1_XD[3]

    x0, _ = avx_421_to_241(X0_X1)
    return x0


def ladder_biscalar_1x2x4x1(
    A: AVX_2x4x1, A24: AVX_2x4x1, basis: tuple[AVX_2x4x1], s0: ZZ, s1: ZZ
) -> AVX_2x4x1:
    # Parse out points from basis
    x0, x1, xd = basis

    X0_X1 = list(avx_241_to_421(x0, x1))
    XD_A = list(avx_241_to_421(xd, A))

    # Helper function for swapping P, Q inside an AVX4x2x1
    def swap(PQ):
        return [PQ[2], PQ[3], PQ[0], PQ[1]]

    while s0 != 0:
        if s1 < s0:
            s0, s1 = s1, s0
            X0_X1 = swap(X0_X1)

        assert s1 >= s0

        if s1 <= 4 * s0:
            s1 = s1 - s0
            # The xADD is a 2x4x1 so we need to convert here
            # Question: does conversion cost mean this is not
            # worth it and we should never do this stage?
            x0, x1 = avx_421_to_241(X0_X1)
            xd, _ = avx_421_to_241(XD_A)
            r0 = x_diff_add_1x2x4x1(x1, x0, xd)

            # Assign output back to 4x2x1 and update vars
            R0_X0 = avx_241_to_421(r0, x0)
            X0_X1[0], X0_X1[1] = R0_X0[0], R0_X0[1]
            XD_A[0], XD_A[1] = R0_X0[2], R0_X0[3]

        elif s0 % 2 == s1 % 2:
            s1 = (s1 - s0) // 2
            # Here we can do everything with swaps
            X1_X0 = swap(X0_X1)
            X1_X0 = x_double_add_normA_1x2x4x1(X1_X0, XD_A)
            X0_X1 = swap(X1_X0)

        elif s1 % 2 == 0:
            s1 = s1 // 2
            # Here we use scratch buffers for the input into
            # x_double_add_1x2x4x1

            X1_XD = [X0_X1[2], X0_X1[3], XD_A[0], XD_A[1]]
            X0_A = [X0_X1[0], X0_X1[1], XD_A[2], XD_A[3]]
            X1_XD = x_double_add_normA_1x2x4x1(X1_XD, X0_A)
            X0_X1[2], X0_X1[3] = X1_XD[0], X1_XD[1]
            XD_A[0], XD_A[1] = X1_XD[2], X1_XD[3]

        else:
            assert s0 % 2 == 0
            s0 = s0 // 2
            # Here we use scratch buffers for the input into
            # x_double_add_1x2x4x1

            X0_XD = [X0_X1[0], X0_X1[1], XD_A[0], XD_A[1]]
            X1_A = [X0_X1[2], X0_X1[3], XD_A[2], XD_A[3]]
            X0_XD = x_double_add_normA_1x2x4x1(X0_XD, X1_A)
            X0_X1[0], X0_X1[1] = X0_XD[0], X0_XD[1]
            XD_A[0], XD_A[1] = X0_XD[2], X0_XD[3]

    # For the end of the biscalar we double in 2x4x1 only then
    # within x_mul_1x2x4x1 we will convert back to 4x2x1 for more
    # xdbladds for the odd part of s1
    _, x1 = avx_421_to_241(X0_X1)

    while s1 & 1 == 0:
        s1 //= 2
        x1 = x_double_1x2x4x1(A24, x1)

    return x_mul_normA_1x2x4x1(A, x1, s1)


def ladder_double_biscalar_1x2x4x1(
    A: AVX_2x4x1,
    A24: AVX_2x4x1,
    basis: tuple[AVX_2x4x1],
    r0: ZZ,
    r1: ZZ,
    s0: ZZ,
    s1: ZZ,
) -> AVX_2x4x1:
    """
    Computes the Euclidean loop with 2 way parallelism to use 8x1x1 multiplication
    but needs to use ladder_biscalar_1x2x4x1 at the end as we can't guarantee s0 = r0
    at the same time

    As we want to call x_double_add_1x2x4x1 in pairs, we also skip the fibonacci step
    when s1 <= 4 * s0 as in this case we would have one xADD and one xDBLADD in the
    generic case which we can't do in parallel.
    """
    # Parse out points from basis, we need two copies as we have two distinct
    # ladders computed for R = [r0] P + [r1] Q and S = [s0] P + [s1] Q
    P, Q, PmQ = basis

    # For the Euclidean step of the algorithm we want to work with 8x1x1 data
    # These two blobs store the values of x0, x1, x0 - x1 as we go down the
    # Montgomery chain.
    X0_X1_XD_A = list(avx_241_to_811(P, Q, PmQ, A))
    Y0_Y1_YD_A = list(avx_241_to_811(P, Q, PmQ, A))

    # We may need less than this, but currently we need three additional
    # scratch buffers of 8x1x1. Two are for the inputs into xDBLADD_pair
    # and one is for the output. The output is then written into X0_X1_XD_A
    # and Y0_Y1_YD_A for the next step of the chain.
    #
    # P1_Q1_PQ1_A and P2_Q2_PQ2_A for the input
    # R1_R2_S1_S2 is the output which holds [2]P1, P1 + Q1, [2]P2, P2 + Q2
    P1_Q1_PQ1_A = X0_X1_XD_A[:]
    P2_Q2_PQ2_A = Y0_Y1_YD_A[:]

    # Depending on the bits of r0, r1 and s0 and s1 we need to have switch
    # cases for the input and output of the various functions
    r_case = None
    s_case = None

    while True:
        if r1 < r0:
            r0, r1 = r1, r0
            # Swap X0, X1
            X0_X1_XD_A[0], X0_X1_XD_A[1], X0_X1_XD_A[2], X0_X1_XD_A[3] = (
                X0_X1_XD_A[2],
                X0_X1_XD_A[3],
                X0_X1_XD_A[0],
                X0_X1_XD_A[1],
            )

        if s1 < s0:
            s0, s1 = s1, s0
            # Swap Y0, Y1
            Y0_Y1_YD_A[0], Y0_Y1_YD_A[1], Y0_Y1_YD_A[2], Y0_Y1_YD_A[3] = (
                Y0_Y1_YD_A[2],
                Y0_Y1_YD_A[3],
                Y0_Y1_YD_A[0],
                Y0_Y1_YD_A[1],
            )

        # Arithmetic for R
        if r0 % 2 == r1 % 2:
            r1 = (r1 - r0) // 2
            r_case = 0
            # P1, Q1, PQ1 = x1, x0, xd
            P1_Q1_PQ1_A[0], P1_Q1_PQ1_A[1] = X0_X1_XD_A[2], X0_X1_XD_A[3]
            P1_Q1_PQ1_A[2], P1_Q1_PQ1_A[3] = X0_X1_XD_A[0], X0_X1_XD_A[1]
            P1_Q1_PQ1_A[4], P1_Q1_PQ1_A[5] = X0_X1_XD_A[4], X0_X1_XD_A[5]

        elif r1 % 2 == 0:
            r1 = r1 // 2
            r_case = 1
            # P1, Q1, PQ1 = x1, xd, x0
            P1_Q1_PQ1_A[0], P1_Q1_PQ1_A[1] = X0_X1_XD_A[2], X0_X1_XD_A[3]
            P1_Q1_PQ1_A[2], P1_Q1_PQ1_A[3] = X0_X1_XD_A[4], X0_X1_XD_A[5]
            P1_Q1_PQ1_A[4], P1_Q1_PQ1_A[5] = X0_X1_XD_A[0], X0_X1_XD_A[1]

        else:
            r0 = r0 // 2
            r_case = 2
            # P1, Q1, PQ1 = x0, xd, x1
            P1_Q1_PQ1_A[0], P1_Q1_PQ1_A[1] = X0_X1_XD_A[0], X0_X1_XD_A[1]
            P1_Q1_PQ1_A[2], P1_Q1_PQ1_A[3] = X0_X1_XD_A[4], X0_X1_XD_A[5]
            P1_Q1_PQ1_A[4], P1_Q1_PQ1_A[5] = X0_X1_XD_A[2], X0_X1_XD_A[3]

        # Arithmetic for S
        if s0 % 2 == s1 % 2:
            s1 = (s1 - s0) // 2
            s_case = 0
            # P2, Q2, PQ2 = y1, y0, yd
            P2_Q2_PQ2_A[0], P2_Q2_PQ2_A[1] = Y0_Y1_YD_A[2], Y0_Y1_YD_A[3]
            P2_Q2_PQ2_A[2], P2_Q2_PQ2_A[3] = Y0_Y1_YD_A[0], Y0_Y1_YD_A[1]
            P2_Q2_PQ2_A[4], P2_Q2_PQ2_A[5] = Y0_Y1_YD_A[4], Y0_Y1_YD_A[5]

        elif s1 % 2 == 0:
            s1 = s1 // 2
            s_case = 1
            # P2, Q2, PQ2 = y1, yd, y0
            P2_Q2_PQ2_A[0], P2_Q2_PQ2_A[1] = Y0_Y1_YD_A[2], Y0_Y1_YD_A[3]
            P2_Q2_PQ2_A[2], P2_Q2_PQ2_A[3] = Y0_Y1_YD_A[4], Y0_Y1_YD_A[5]
            P2_Q2_PQ2_A[4], P2_Q2_PQ2_A[5] = Y0_Y1_YD_A[0], Y0_Y1_YD_A[1]
        else:
            s0 = s0 // 2
            s_case = 2
            # P2, Q2, PQ2 = y0, yd, y1
            P2_Q2_PQ2_A[0], P2_Q2_PQ2_A[1] = Y0_Y1_YD_A[0], Y0_Y1_YD_A[1]
            P2_Q2_PQ2_A[2], P2_Q2_PQ2_A[3] = Y0_Y1_YD_A[4], Y0_Y1_YD_A[5]
            P2_Q2_PQ2_A[4], P2_Q2_PQ2_A[5] = Y0_Y1_YD_A[2], Y0_Y1_YD_A[3]

        # R1, R2 = x_double_add_1x2x4x1(A24, P1, Q1, PQ1)
        # S1, S2 = x_double_add_1x2x4x1(A24, P2, Q2, PQ2)
        R1_R2_S1_S2 = x_double_add_pair_normA_8x1x1(P1_Q1_PQ1_A, P2_Q2_PQ2_A)

        if r_case == 0:
            # x0, x1 = R2, R1
            X0_X1_XD_A[0], X0_X1_XD_A[1] = R1_R2_S1_S2[2], R1_R2_S1_S2[3]
            X0_X1_XD_A[2], X0_X1_XD_A[3] = R1_R2_S1_S2[0], R1_R2_S1_S2[1]

        elif r_case == 1:
            # x1, xd = R1, R2
            X0_X1_XD_A[2], X0_X1_XD_A[3] = R1_R2_S1_S2[0], R1_R2_S1_S2[1]
            X0_X1_XD_A[4], X0_X1_XD_A[5] = R1_R2_S1_S2[2], R1_R2_S1_S2[3]
        else:
            # x0, xd = R1, R2
            X0_X1_XD_A[0], X0_X1_XD_A[1] = R1_R2_S1_S2[0], R1_R2_S1_S2[1]
            X0_X1_XD_A[4], X0_X1_XD_A[5] = R1_R2_S1_S2[2], R1_R2_S1_S2[3]

        if s_case == 0:
            # y0, y1 = S2, S1
            Y0_Y1_YD_A[0], Y0_Y1_YD_A[1] = R1_R2_S1_S2[6], R1_R2_S1_S2[7]
            Y0_Y1_YD_A[2], Y0_Y1_YD_A[3] = R1_R2_S1_S2[4], R1_R2_S1_S2[5]
        elif s_case == 1:
            # y1, yd = S1, S2
            Y0_Y1_YD_A[2], Y0_Y1_YD_A[3] = R1_R2_S1_S2[4], R1_R2_S1_S2[5]
            Y0_Y1_YD_A[4], Y0_Y1_YD_A[5] = R1_R2_S1_S2[6], R1_R2_S1_S2[7]
        else:
            # y0, yd = S1, S2
            Y0_Y1_YD_A[0], Y0_Y1_YD_A[1] = R1_R2_S1_S2[4], R1_R2_S1_S2[5]
            Y0_Y1_YD_A[4], Y0_Y1_YD_A[5] = R1_R2_S1_S2[6], R1_R2_S1_S2[7]

        # If either s0 or r0 is zero, we must break the chain
        if s0 == 0 or r0 == 0:
            break

    # We now know that either r0 or s0 is zero, but potentially not
    # both, so we finish off the rest of the chain using the single
    # biscalars (although one will simply be a doubling followed by
    # x_mul as the euclidean loop is finished)

    x0, x1, xd, _ = avx_811_to_241(X0_X1_XD_A)
    y0, y1, yd, _ = avx_811_to_241(Y0_Y1_YD_A)
    R = ladder_biscalar_1x2x4x1(A, A24, (x0, x1, xd), r0, r1)
    S = ladder_biscalar_1x2x4x1(A, A24, (y0, y1, yd), s0, s1)

    return R, S


def four_jac_to_ws(
    A12: AVX_4x2x1, X_M: AVX_4x2x1, Y_M: AVX_4x2x1, Z_M: AVX_4x2x1
) -> tuple[AVX_4x2x1]:
    """
    Convert from coordinates on a Montgomery curve to Weierstrass curve
    modified to work with four points in parallel
    """
    # We have an optimised fp division by three, maybe we can also
    # vectorise this?
    A12_over_3 = AVX_4x2x1(*[A / 3 for A in A12])

    # a = 1 - A^2 / 3
    one = AVX_4x2x1(1, 1, 1, 1)
    a, Z2 = avx_421_pair_mul(A12, A12_over_3, Z_M, Z_M)  # This is 1M, 1S
    a = avx_421_sub(one, a)
    Z4 = avx_421_sqr(Z2)

    # X = X + Z^2 * A/3
    # Y = Y
    # Z = Z
    # T = a * Z^4
    tmp, T = avx_421_pair_mul(Z2, A12_over_3, Z4, a)
    X = avx_421_add(X_M, tmp)
    Y = Y_M
    Z = Z_M

    return X, Y, Z, T, A12_over_3


def four_ws_to_jac(X, Y, Z, A12_over_3) -> tuple[PointJ]:
    """
    Convert from coordinates on a Weierstrass curve to Montgomery curve
    modified to work with four points in parallel
    """
    # X = P.X - P.Z**2 * A_over_3
    # Y = P.Y
    # Z = P.Z

    tmp = avx_421_sqr(Z)
    tmp = avx_421_mul(tmp, A12_over_3)
    X = avx_421_sub(X, tmp)

    return X, Y, Z


def four_double_ws(
    X: AVX_4x2x1, Y: AVX_4x2x1, Z: AVX_4x2x1, T: AVX_4x2x1
) -> tuple[AVX_4x2x1]:
    """
    Simultaneous point doubling of four points on a short weierstrass curve
    with modified Jacobian coordinates
    """
    XX, t0 = avx_421_pair_sqr(X, Y)

    A  = avx_421_dbl(t0)
    t1 = avx_421_dbl(XX)

    t2 = avx_421_add(X, A)
    t3 = avx_421_add(t1, XX)

    t2, AA = avx_421_pair_sqr(t2, A)

    t2 = avx_421_sub(t2, XX)
    U  = avx_421_dbl(AA)

    M = avx_421_add(t3, T)
    S = avx_421_sub(t2, AA)

    t0, t1 = avx_421_pair_mul(M, M, Y, Z)  # This is 1M, 1S

    Z3 = avx_421_dbl(t1)
    t2 = avx_421_dbl(S)

    X3 = avx_421_sub(t0, t2)

    t0 = avx_421_sub(S, X3)

    t0, t1 = avx_421_pair_mul(M, t0, U, T)

    Y3 = avx_421_sub(t0, U)
    T3 = avx_421_dbl(t1)

    return X3, Y3, Z3, T3


def four_double_ws_n(X: AVX_4x2x1, Y: AVX_4x2x1, Z: AVX_4x2x1, T: AVX_4x2x1, n: int):
    for _ in range(n):
        X, Y, Z, T = four_double_ws(X, Y, Z, T)
    return X, Y, Z, T


def four_add_components(A: AVX_4x2x1, P: tuple[AVX_4x2x1], Q: tuple[AVX_4x2x1]):
    """
    Algorithm 8.13 of the SQISign spec.

    Given distinct points P, Q and the Montgomery coefficient A compute u, v, w
    such that x(P + Q) = (u - v : w) and x(P - Q) = (u + x : w)

    Adapted to take in four points at the same time, where P is a tuple X, Y, Z
    where each is a AVX_4x2x1 with (PX1, QX1, PX2, QX2) etc.
    """
    # Unpack coordinates
    XP, YP, ZP = P
    XQ, YQ, ZQ = Q

    # Compute the slope dy / dx for the case P != Q
    t0, t1 = avx_421_pair_sqr(ZP, ZQ)
    t2, t3 = avx_421_pair_mul(t1, XP, t0, XQ)
    t4, t5 = avx_421_pair_mul(YP, ZQ, ZP, YQ)
    t4, t5 = avx_421_pair_mul(t4, t1, t5, t0)
    t0, t6 = avx_421_pair_mul(t0, t1, t4, t5)
    v = avx_421_dbl(t6)
    t4, t5 = avx_421_pair_sqr(t4, t5)
    t4 = avx_421_add(t4, t5)
    t5 = avx_421_add(t2, t3)
    t6 = avx_421_dbl(t3)
    t6 = avx_421_sub(t5, t6)
    t6, t1 = avx_421_pair_mul(t6, t6, A, t0)  # This is 1M, 1S
    t1 = avx_421_add(t5, t1)
    t1, w = avx_421_pair_mul(t1, t6, t6, t0)
    u = avx_421_sub(t4, t1)

    return (u, v, w)


def four_gluing_dbladd(
    A24: AVX_4x2x1, Xe: AVX_4x2x1, Ze: AVX_4x2x1, Xo: AVX_4x2x1, Zo: AVX_4x2x1, xD: AVX_4x2x1
):
    """
    One step of the gluing ladder on four points: given ([m]G : -) and
    ([m-1]G : -) with G = (xD : 1) the normalised difference, return
    ([2m]G : -) and ([2m-1]G : -).

    24M + 16S + 32a
    """
    t0 = avx_421_add(Xe, Ze)
    t1 = avx_421_sub(Xe, Ze)
    t2 = avx_421_sub(Xo, Zo)
    t3 = avx_421_add(Xo, Zo)

    # 4S
    Xe, Ze = avx_421_pair_sqr(t0, t1)

    # 4M
    t0, t1 = avx_421_pair_mul(t0, t2, t1, t3)
    t2 = avx_421_sub(Xe, Ze)

    # 4M
    Xe, t3 = avx_421_pair_mul(Xe, Ze, A24, t2)
    Ze = avx_421_add(Ze, t3)
    Xo = avx_421_add(t0, t1)
    Zo = avx_421_sub(t0, t1)

    # 4S
    Zo, Xo = avx_421_pair_sqr(Zo, Xo)

    # 4M
    Ze, Zo = avx_421_pair_mul(Ze, t2, Zo, xD)

    return Xe, Ze, Xo, Zo


def four_gluing_ladder(
    A24: AVX_4x2x1,
    Xe: AVX_4x2x1,
    Ze: AVX_4x2x1,
    Xo: AVX_4x2x1,
    Zo: AVX_4x2x1,
    xD: AVX_4x2x1,
    n: int,
):
    for _ in range(n):
        Xe, Ze, Xo, Zo = four_gluing_dbladd(A24, Xe, Ze, Xo, Zo, xD)
    return Xe, Ze, Xo, Zo


def four_bary_components(
    Xe: AVX_4x2x1,
    Ze: AVX_4x2x1,
    Xo: AVX_4x2x1,
    Zo: AVX_4x2x1,
    Dp: AVX_4x2x1,
    Dm: AVX_4x2x1,
):
    """
    Given P = ([m]G : -), its difference P - G = ([m-1]G : -) and the sum and
    difference Dp, Dm of the coordinates of G, compute u, v, w such that
    x(P + G) = (u - v : w) and x(P - G) = (u + v : w).

    Replaces four_add_components, which needs P in Jacobian coordinates.

    24M + 8S + 28a
    """
    t0 = avx_421_add(Xe, Ze)
    t1 = avx_421_sub(Xe, Ze)

    # 4M
    t0, t1 = avx_421_pair_mul(t0, Dm, t1, Dp)
    lam = avx_421_add(t1, t0)
    mu = avx_421_sub(t1, t0)

    # 4M
    t0, t1 = avx_421_pair_mul(Zo, lam, mu, Xo)

    # 4S
    t0, t3 = avx_421_pair_sqr(t0, t1)

    # 4M
    t2 = avx_421_mul(mu, Zo)
    u = avx_421_add(t3, t0)
    v = avx_421_sub(t3, t0)

    # 4M
    w = avx_421_dbl(avx_421_mul(t1, t2))

    return (u, v, w)


def lift_basis_to_jacobian_2x4x1(
    A: AVX_4x2x1, basis: tuple[AVX_2x4x1]
) -> tuple[AVX_2x4x1]:
    """
    Given an x-only basis, lift to <P, Q> in Jacobian coordinates
    """
    # First we need to normalise A and the first point in the basis
    xP, xQ, xPQ = basis
    inv_C, inv_PZ = batched_inversion(A[1], xP[1])
    a = A[0] * inv_C
    PX = xP[0] * inv_PZ

    # The other two points need not be normalised
    QX, QZ = xQ
    PQX, PQZ = xPQ

    # compute PY with a square root
    PYPY = PX * (PX**2 + PX * a + 1)
    PY = sqrt_fp2(PYPY)

    # compute QY with Okeya-Sakurai algorithm
    v1 = PX * QZ
    v2 = QX + v1
    v3 = QX - v1
    v3 = v3 * v3
    v3 = v3 * PQX
    v1 = a + a
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

    X = AVX_2x4x1(PX, QX)
    Y = AVX_2x4x1(PY, QY)
    Z = AVX_2x4x1(1, QZ)

    return a, X, Y, Z
