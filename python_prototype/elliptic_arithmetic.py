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
    avx_811_mul,
)
from utilities import sqrt_fp2

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

    #figure out how to do this efficiently in C
    PP_PQ = AVX_4x2x1(S3[0] - S3[1], 2*S4[0] + S4[1], S4[2], S4[3])
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


def x_mul_1x2x4x1(A24: AVX_2x4x1, P: AVX_2x4x1, m: ZZ) -> PointX:
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
    XD_A24 = avx_241_to_421(xp, A24)

    # Helper function for swapping P, Q inside an AVX4x2x1
    def swap(PQ):
        return AVX_4x2x1(PQ[2], PQ[3], PQ[0], PQ[1])

    # Montgomery-ladder, here we only ever have to swap x0, x1
    for bit in bin(m)[2:]:
        if bit == "1":
            X0_X1 = swap(X0_X1)

        X0_X1 = x_double_add_1x2x4x1(X0_X1, XD_A24)

        if bit == "1":
            X0_X1 = swap(X0_X1)

    x0, x1 = avx_421_to_241(X0_X1)

    return x0


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
    A24: AVX_2x4x1, basis: tuple[AVX_2x4x1], s0: ZZ, s1: ZZ
) -> AVX_2x4x1:
    # Parse out points from basis
    x0, x1, xd = basis

    X0_X1 = list(avx_241_to_421(x0, x1))
    XD_A24 = list(avx_241_to_421(xd, A24))

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
            xd, _ = avx_421_to_241(XD_A24)
            r0 = x_diff_add_1x2x4x1(x1, x0, xd)

            # Assign output back to 4x2x1 and update vars
            R0_X0 = avx_241_to_421(r0, x0)
            X0_X1[0], X0_X1[1] = R0_X0[0], R0_X0[1]
            XD_A24[0], XD_A24[1] = R0_X0[2], R0_X0[3]

        elif s0 % 2 == s1 % 2:
            s1 = (s1 - s0) // 2
            # Here we can do everything with swaps
            X1_X0 = swap(X0_X1)
            X1_X0 = x_double_add_1x2x4x1(X1_X0, XD_A24)
            X0_X1 = swap(X1_X0)

        elif s1 % 2 == 0:
            s1 = s1 // 2
            # Here we use scratch buffers for the input into
            # x_double_add_1x2x4x1

            X1_XD = [X0_X1[2], X0_X1[3], XD_A24[0], XD_A24[1]]
            X0_A24 = [X0_X1[0], X0_X1[1], XD_A24[2], XD_A24[3]]
            X1_XD = x_double_add_1x2x4x1(X1_XD, X0_A24)
            X0_X1[2], X0_X1[3] = X1_XD[0], X1_XD[1]
            XD_A24[0], XD_A24[1] = X1_XD[2], X1_XD[3]

        else:
            assert s0 % 2 == 0
            s0 = s0 // 2
            # Here we use scratch buffers for the input into
            # x_double_add_1x2x4x1

            X0_XD = [X0_X1[0], X0_X1[1], XD_A24[0], XD_A24[1]]
            X1_A24 = [X0_X1[2], X0_X1[3], XD_A24[2], XD_A24[3]]
            X0_XD = x_double_add_1x2x4x1(X0_XD, X1_A24)
            X0_X1[0], X0_X1[1] = X0_XD[0], X0_XD[1]
            XD_A24[0], XD_A24[1] = X0_XD[2], X0_XD[3]

    # For the end of the biscalar we double in 2x4x1 only then
    # within x_mul_1x2x4x1 we will convert back to 4x2x1 for more
    # xdbladds for the odd part of s1
    _, x1 = avx_421_to_241(X0_X1)

    while s1 & 1 == 0:
        s1 //= 2
        x1 = x_double_1x2x4x1(A24, x1)

    return x_mul_1x2x4x1(A24, x1, s1)


def ladder_double_biscalar_1x2x4x1(
    A24: AVX_2x4x1, basis: tuple[AVX_2x4x1], r0: ZZ, r1: ZZ, s0: ZZ, s1: ZZ
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
    X0_X1_XD_A24 = list(avx_241_to_811(P, Q, PmQ, A24))
    Y0_Y1_YD_A24 = list(avx_241_to_811(P, Q, PmQ, A24))

    # We may need less than this, but currently we need three additional
    # scratch buffers of 8x1x1. Two are for the inputs into xDBLADD_pair
    # and one is for the output. The output is then written into X0_X1_XD_A24
    # and Y0_Y1_YD_A24 for the next step of the chain.
    #
    # P1_Q1_PQ1_A24 and P2_Q2_PQ2_A24 for the input
    # R1_R2_S1_S2 is the output which holds [2]P1, P1 + Q1, [2]P2, P2 + Q2
    P1_Q1_PQ1_A24 = X0_X1_XD_A24[:]
    P2_Q2_PQ2_A24 = Y0_Y1_YD_A24[:]

    # Depending on the bits of r0, r1 and s0 and s1 we need to have switch
    # cases for the input and output of the various functions
    r_case = None
    s_case = None

    while True:
        if r1 < r0:
            r0, r1 = r1, r0
            # Swap X0, X1
            X0_X1_XD_A24[0], X0_X1_XD_A24[1], X0_X1_XD_A24[2], X0_X1_XD_A24[3] = (
                X0_X1_XD_A24[2],
                X0_X1_XD_A24[3],
                X0_X1_XD_A24[0],
                X0_X1_XD_A24[1],
            )

        if s1 < s0:
            s0, s1 = s1, s0
            # Swap Y0, Y1
            Y0_Y1_YD_A24[0], Y0_Y1_YD_A24[1], Y0_Y1_YD_A24[2], Y0_Y1_YD_A24[3] = (
                Y0_Y1_YD_A24[2],
                Y0_Y1_YD_A24[3],
                Y0_Y1_YD_A24[0],
                Y0_Y1_YD_A24[1],
            )

        # Arithmetic for R
        if r0 % 2 == r1 % 2:
            r1 = (r1 - r0) // 2
            r_case = 0
            # P1, Q1, PQ1 = x1, x0, xd
            P1_Q1_PQ1_A24[0], P1_Q1_PQ1_A24[1] = X0_X1_XD_A24[2], X0_X1_XD_A24[3]
            P1_Q1_PQ1_A24[2], P1_Q1_PQ1_A24[3] = X0_X1_XD_A24[0], X0_X1_XD_A24[1]
            P1_Q1_PQ1_A24[4], P1_Q1_PQ1_A24[5] = X0_X1_XD_A24[4], X0_X1_XD_A24[5]

        elif r1 % 2 == 0:
            r1 = r1 // 2
            r_case = 1
            # P1, Q1, PQ1 = x1, xd, x0
            P1_Q1_PQ1_A24[0], P1_Q1_PQ1_A24[1] = X0_X1_XD_A24[2], X0_X1_XD_A24[3]
            P1_Q1_PQ1_A24[2], P1_Q1_PQ1_A24[3] = X0_X1_XD_A24[4], X0_X1_XD_A24[5]
            P1_Q1_PQ1_A24[4], P1_Q1_PQ1_A24[5] = X0_X1_XD_A24[0], X0_X1_XD_A24[1]

        else:
            r0 = r0 // 2
            r_case = 2
            # P1, Q1, PQ1 = x0, xd, x1
            P1_Q1_PQ1_A24[0], P1_Q1_PQ1_A24[1] = X0_X1_XD_A24[0], X0_X1_XD_A24[1]
            P1_Q1_PQ1_A24[2], P1_Q1_PQ1_A24[3] = X0_X1_XD_A24[4], X0_X1_XD_A24[5]
            P1_Q1_PQ1_A24[4], P1_Q1_PQ1_A24[5] = X0_X1_XD_A24[2], X0_X1_XD_A24[3]

        # Arithmetic for S
        if s0 % 2 == s1 % 2:
            s1 = (s1 - s0) // 2
            s_case = 0
            # P2, Q2, PQ2 = y1, y0, yd
            P2_Q2_PQ2_A24[0], P2_Q2_PQ2_A24[1] = Y0_Y1_YD_A24[2], Y0_Y1_YD_A24[3]
            P2_Q2_PQ2_A24[2], P2_Q2_PQ2_A24[3] = Y0_Y1_YD_A24[0], Y0_Y1_YD_A24[1]
            P2_Q2_PQ2_A24[4], P2_Q2_PQ2_A24[5] = Y0_Y1_YD_A24[4], Y0_Y1_YD_A24[5]

        elif s1 % 2 == 0:
            s1 = s1 // 2
            s_case = 1
            # P2, Q2, PQ2 = y1, yd, y0
            P2_Q2_PQ2_A24[0], P2_Q2_PQ2_A24[1] = Y0_Y1_YD_A24[2], Y0_Y1_YD_A24[3]
            P2_Q2_PQ2_A24[2], P2_Q2_PQ2_A24[3] = Y0_Y1_YD_A24[4], Y0_Y1_YD_A24[5]
            P2_Q2_PQ2_A24[4], P2_Q2_PQ2_A24[5] = Y0_Y1_YD_A24[0], Y0_Y1_YD_A24[1]
        else:
            s0 = s0 // 2
            s_case = 2
            # P2, Q2, PQ2 = y0, yd, y1
            P2_Q2_PQ2_A24[0], P2_Q2_PQ2_A24[1] = Y0_Y1_YD_A24[0], Y0_Y1_YD_A24[1]
            P2_Q2_PQ2_A24[2], P2_Q2_PQ2_A24[3] = Y0_Y1_YD_A24[4], Y0_Y1_YD_A24[5]
            P2_Q2_PQ2_A24[4], P2_Q2_PQ2_A24[5] = Y0_Y1_YD_A24[2], Y0_Y1_YD_A24[3]

        # R1, R2 = x_double_add_1x2x4x1(A24, P1, Q1, PQ1)
        # S1, S2 = x_double_add_1x2x4x1(A24, P2, Q2, PQ2)
        R1_R2_S1_S2 = x_double_add_pair_8x1x1(P1_Q1_PQ1_A24, P2_Q2_PQ2_A24)

        if r_case == 0:
            # x0, x1 = R2, R1
            X0_X1_XD_A24[0], X0_X1_XD_A24[1] = R1_R2_S1_S2[2], R1_R2_S1_S2[3]
            X0_X1_XD_A24[2], X0_X1_XD_A24[3] = R1_R2_S1_S2[0], R1_R2_S1_S2[1]

        elif r_case == 1:
            # x1, xd = R1, R2
            X0_X1_XD_A24[2], X0_X1_XD_A24[3] = R1_R2_S1_S2[0], R1_R2_S1_S2[1]
            X0_X1_XD_A24[4], X0_X1_XD_A24[5] = R1_R2_S1_S2[2], R1_R2_S1_S2[3]
        else:
            # x0, xd = R1, R2
            X0_X1_XD_A24[0], X0_X1_XD_A24[1] = R1_R2_S1_S2[0], R1_R2_S1_S2[1]
            X0_X1_XD_A24[4], X0_X1_XD_A24[5] = R1_R2_S1_S2[2], R1_R2_S1_S2[3]

        if s_case == 0:
            # y0, y1 = S2, S1
            Y0_Y1_YD_A24[0], Y0_Y1_YD_A24[1] = R1_R2_S1_S2[6], R1_R2_S1_S2[7]
            Y0_Y1_YD_A24[2], Y0_Y1_YD_A24[3] = R1_R2_S1_S2[4], R1_R2_S1_S2[5]
        elif s_case == 1:
            # y1, yd = S1, S2
            Y0_Y1_YD_A24[2], Y0_Y1_YD_A24[3] = R1_R2_S1_S2[4], R1_R2_S1_S2[5]
            Y0_Y1_YD_A24[4], Y0_Y1_YD_A24[5] = R1_R2_S1_S2[6], R1_R2_S1_S2[7]
        else:
            # y0, yd = S1, S2
            Y0_Y1_YD_A24[0], Y0_Y1_YD_A24[1] = R1_R2_S1_S2[4], R1_R2_S1_S2[5]
            Y0_Y1_YD_A24[4], Y0_Y1_YD_A24[5] = R1_R2_S1_S2[6], R1_R2_S1_S2[7]

        # If either s0 or r0 is zero, we must break the chain
        if s0 == 0 or r0 == 0:
            break

    # We now know that either r0 or s0 is zero, but potentially not
    # both, so we finish off the rest of the chain using the single
    # biscalars (although one will simply be a doubling followed by
    # x_mul as the euclidean loop is finished)

    x0, x1, xd, _ = avx_811_to_241(X0_X1_XD_A24)
    y0, y1, yd, _ = avx_811_to_241(Y0_Y1_YD_A24)
    R = ladder_biscalar_1x2x4x1(A24, (x0, x1, xd), r0, r1)
    S = ladder_biscalar_1x2x4x1(A24, (y0, y1, yd), s0, s1)

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
    A = avx_421_dbl(t0)
    t0 = avx_421_add(X, A)
    t0, AA = avx_421_pair_sqr(t0, A)
    U = avx_421_dbl(AA)
    t0 = avx_421_sub(t0, XX)
    S = avx_421_sub(t0, AA)
    t0 = avx_421_dbl(XX)
    t0 = avx_421_add(t0, XX)
    M = avx_421_add(t0, T)
    t0, t1 = avx_421_pair_mul(M, M, Y, Z)  # This is 1M, 1S
    t2 = avx_421_dbl(S)
    X3 = avx_421_sub(t0, t2)
    t0 = avx_421_sub(S, X3)
    t0, t2 = avx_421_pair_mul(M, t0, U, T)
    Y3 = avx_421_sub(t0, U)
    Z3 = avx_421_dbl(t1)
    T3 = avx_421_dbl(t2)

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
