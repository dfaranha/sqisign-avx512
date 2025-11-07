from elliptic_types import Curve, BasisX, PointJ
from avx_types import AVX_4x2x1
from avx_arithmetic import avx_sqr, avx_pairwise_mul, avx_add, avx_sub, avx_dbl
from utilities import sqrt_fp2

# ======================================================================
# 64 bit operations
# ======================================================================


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


def four_double_x(A24: AVX_4x2x1, X: AVX_4x2x1, Z: AVX_4x2x1):
    """
    Simultaneously double four X-only points where the X-coordinates
    of P1, Q1 P2, Q2 on E1 x E2 are stored within X (and Z resp.)
    """
    t0 = avx_add(X, Z)
    t0 = avx_sqr(t0)
    t1 = avx_sub(X, Z)
    t1 = avx_sqr(t1)
    t2 = avx_sub(t0, t1)
    X2 = avx_pairwise_mul(t0, t1)
    t0 = avx_pairwise_mul(t2, A24)
    t0 = avx_add(t0, t1)
    Z2 = avx_pairwise_mul(t0, t2)

    return X2, Z2


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
    a = avx_pairwise_mul(A12, A12_over_3)
    a = avx_sub(one, a)

    # X = X + Z^2 * A/3
    # Y = Y
    # Z = Z
    # T = a * Z^4
    Z2 = avx_sqr(Z_M)
    Z4 = avx_sqr(Z2)

    tmp = avx_pairwise_mul(Z2, A12_over_3)
    X = avx_add(X_M, tmp)
    Y = Y_M
    Z = Z_M
    T = avx_pairwise_mul(Z4, a)

    return X, Y, Z, T, A12_over_3


def four_ws_to_jac(X, Y, Z, A12_over_3) -> tuple[PointJ]:
    """
    Convert from coordinates on a Weierstrass curve to Montgomery curve
    modified to work with four points in parallel
    """
    # X = P.X - P.Z**2 * A_over_3
    # Y = P.Y
    # Z = P.Z

    tmp = avx_sqr(Z)
    tmp = avx_pairwise_mul(tmp, A12_over_3)
    X = avx_sub(X, tmp)

    return X, Y, Z


def four_double_ws(
    X: AVX_4x2x1, Y: AVX_4x2x1, Z: AVX_4x2x1, T: AVX_4x2x1
) -> tuple[AVX_4x2x1]:
    """
    Simultaneous point doubling of four points on a short weierstrass curve
    with modified Jacobian coordinates
    """
    XX = avx_sqr(X)
    t0 = avx_sqr(Y)
    A = avx_dbl(t0)
    AA = avx_sqr(A)
    U = avx_dbl(AA)
    t0 = avx_add(X, A)
    t0 = avx_sqr(t0)
    t0 = avx_sub(t0, XX)
    S = avx_sub(t0, AA)
    t0 = avx_dbl(XX)
    t0 = avx_add(t0, XX)
    M = avx_add(t0, T)
    t0 = avx_sqr(M)
    t1 = avx_dbl(S)
    X3 = avx_sub(t0, t1)
    t0 = avx_sub(S, X3)
    t0 = avx_pairwise_mul(M, t0)
    Y3 = avx_sub(t0, U)
    t0 = avx_pairwise_mul(Y, Z)
    Z3 = avx_dbl(t0)
    t0 = avx_pairwise_mul(U, T)
    T3 = avx_dbl(t0)

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
    t0 = avx_sqr(ZP)
    t1 = avx_sqr(ZQ)
    t2 = avx_pairwise_mul(t1, XP)
    t3 = avx_pairwise_mul(t0, XQ)
    t4 = avx_pairwise_mul(YP, ZQ)
    t4 = avx_pairwise_mul(t4, t1)
    t5 = avx_pairwise_mul(ZP, YQ)
    t5 = avx_pairwise_mul(t5, t0)
    t0 = avx_pairwise_mul(t0, t1)
    t6 = avx_pairwise_mul(t4, t5)
    v = avx_dbl(t6)
    t4 = avx_sqr(t4)
    t5 = avx_sqr(t5)
    t4 = avx_add(t4, t5)
    t5 = avx_add(t2, t3)
    t6 = avx_dbl(t3)
    t6 = avx_sub(t5, t6)
    t6 = avx_sqr(t6)
    t1 = avx_pairwise_mul(A, t0)
    t1 = avx_add(t5, t1)
    t1 = avx_pairwise_mul(t1, t6)
    u = avx_sub(t4, t1)
    w = avx_pairwise_mul(t6, t0)

    return (u, v, w)
