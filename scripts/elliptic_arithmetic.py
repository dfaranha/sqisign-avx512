from elliptic_types import Curve, BasisX, PointJ
from avx_types import AVX_4x2x1
from avx_arithmetic import (
    avx_421_sqr,
    avx_421_mul,
    avx_421_add,
    avx_421_sub,
    avx_421_dbl,
    avx_421_pair_sqr,
    avx_421_pair_mul,
)
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
    t0 = avx_421_add(X, Z)
    t1 = avx_421_sub(X, Z)
    t0, t1 = avx_421_pair_sqr(t0, t1)
    t2 = avx_421_sub(t0, t1)
    X2, t0 = avx_421_pair_mul(t0, t1, t2, A24)
    t0 = avx_421_add(t0, t1)
    Z2 = avx_421_mul(t0, t2)

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
