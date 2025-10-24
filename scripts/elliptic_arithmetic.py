from elliptic_types import Curve, PointX, PointJ, PointW


def double_x(E: Curve, P: PointX) -> PointX:
    """
    Algorithm 8.3 of the SQISign spec.

    Montgomery doubling with projective curve constant

    Input:  projective point P = (X:Z), curve constants (A24 : C24)
    Output: projective point [2]P = (X2 : Z2)

    Cost: 4M + 2S + 4a
    """
    X, Z = P

    t0 = X + Z
    t0 = t0 * t0
    t1 = X - Z
    t1 = t1 * t1
    t2 = t0 - t1
    t1 = t1 * E.C24
    X2 = t0 * t1
    t0 = t2 * E.A24
    t0 = t0 + t1
    Z2 = t0 * t2

    return PointX(X2, Z2)


def double_jac(E: Curve, P: PointJ) -> PointJ:
    """
    Algorithm 8.11 of the SQISign spec.

    Given a Jacobian point P with coordinates (X : Y : Z) and the
    Montgomery coefficient A computes R = [2]P

    Cost: 6S + 6M + 14a
    """
    # Unpack coordinates
    XP, YP, ZP = P
    A = E.A / E.C  # TODO: remove inversion

    t0 = XP * XP
    t1 = t0 + t0
    t0 = t0 + t1
    t1 = ZP * ZP
    t2 = XP * A
    t2 = t2 + t2
    t2 = t1 + t2
    t2 = t1 * t2
    t2 = t0 + t2
    Z2P = YP * ZP
    Z2P = Z2P + Z2P
    t0 = Z2P * Z2P
    t0 = t0 * A
    t1 = YP * YP
    t1 = t1 + t1
    t3 = XP + XP
    t3 = t1 * t3
    X2P = t2 * t2
    X2P = X2P - t0
    X2P = X2P - t3
    X2P = X2P - t3
    Y2P = t3 - X2P
    Y2P = Y2P * t2
    t1 = t1 * t1
    Y2P = Y2P - t1
    Y2P = Y2P - t1

    return PointJ(X2P, Y2P, Z2P)


def jac_to_ws(E: Curve, P: PointJ) -> PointW:
    """
    Conversion from  Jacobian coordinates on a Montgomery curve
    to Modified Jacobian coordinates on a Weierstrass curve
    """
    assert E.C.is_one()

    A_over_3 = E.A / 3
    a = 1 - E.A * A_over_3

    Z2 = P.Z**2
    Z4 = Z2**2

    X = P.X + Z2 * A_over_3
    Y = P.Y
    Z = P.Z
    T = a * Z4

    return PointW(X, Y, Z, T), A_over_3


def ws_to_jac(P: PointW, A_over_3) -> PointJ:
    """
    Conversion from modified Jacobian coordinates on a Weierstrass curve
    to Jacobian coordinates on a Montgomery curve
    """
    X = P.X - P.Z**2 * A_over_3
    Y = P.Y
    Z = P.Z
    return PointJ(X, Y, Z)


def double_ws(P: PointW) -> PointW:
    """
    Point doubling on a short weierstrass curve with modified Jacobian coordinates
    """
    XX = P.X**2
    t0 = P.Y**2
    A = t0 + t0
    AA = A**2
    U = AA + AA
    t1 = P.X + A
    t2 = t1**2
    t3 = t2 - XX
    S = t3 - AA
    t4 = XX + XX
    t4 = t4 + XX
    M = t4 + P.T
    t5 = M**2
    t6 = S + S
    X3 = t5 - t6
    t7 = S - X3
    t8 = M * t7
    Y3 = t8 - U
    t9 = P.Y * P.Z
    Z3 = t9 + t9
    t10 = U * P.T
    T3 = t10 + t10

    return PointW(X3, Y3, Z3, T3)


def double_jac_n(E: Curve, P: PointJ, n: int):
    """
    Repeated Doubling of a Jacobian point
    """
    if n == 1:
        return double_jac(E, P)

    # Convert from Montgomery to Weierstrass
    # TODO: we compute a = 1 - A^2/3 twice as we do
    #       two points on the same curve, we could
    #       do better by doubling two points at the
    #       same time.
    Q, A_over_3 = jac_to_ws(E, P)  # 3M 2S
    for _ in range(n):
        Q = double_ws(Q)  # 3M 5S
    return ws_to_jac(Q, A_over_3)


def add_components(E: Curve, P: PointJ, Q: PointJ):
    """
    Algorithm 8.13 of the SQISign spec.

    Given distinct points P, Q and the Montgomery coefficient A compute u, v, w
    such that x(P + Q) = (u - v : w) and x(P - Q) = (u + x : w)
    """
    # Unpack coordinates
    XP, YP, ZP = P
    XQ, YQ, ZQ = Q
    A = E.A / E.C  # TODO: remove inversion

    # Compute the slope dy / dx for the case P != Q
    t0 = ZP * ZP
    t1 = ZQ * ZQ
    t2 = t1 * XP
    t3 = t0 * XQ
    t4 = YP * ZQ
    t4 = t4 * t1
    t5 = ZP * YQ
    t5 = t5 * t0
    t0 = t0 * t1
    t6 = t4 * t5
    v = t6 + t6
    t4 = t4 * t4
    t5 = t5 * t5
    t4 = t4 + t5
    t5 = t2 + t3
    t6 = t3 + t3
    t6 = t5 - t6
    t6 = t6 * t6
    t1 = A * t0
    t1 = t5 + t1
    t1 = t1 * t6
    u = t4 - t1
    w = t6 * t0

    return (u, v, w)
