def DBL(P, A):
    """
    Algorithm 8.11 of the SQISign spec.

    Given a Jacobian point P with coordinates (X : Y : Z) and the
    Montgomery coefficient A computes R = [2]P

    Cost: 6S + 6M + 14a
    """
    # Unpack coordinates
    XP, YP, ZP = P

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

    return (X2P, Y2P, Z2P)


def ADD_components(P, Q, A):
    """
    Algorithm 8.13 of the SQISign spec.

    Given distinct points P, Q and the Montgomery coefficient A compute u, v, w
    such that x(P + Q) = (u - v : w) and x(P - Q) = (u + x : w)
    """
    # Unpack coordinates
    XP, YP, ZP = P
    XQ, YQ, ZQ = Q

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
