from sage.all import proof, GF, EllipticCurve
from elliptic_curves.jacobian_arithmetic import DBL, ADD_components


def random_non_zero_element(F):
    while True:
        x = F.random_element()
        if not x.is_zero():
            return x


def randomize_point_to_jacobian(P):
    """
    Take a point and randomize the coordinates by a projective factor
    """
    F = P.base_ring()
    E = P.parent()
    XP, YP, ZP = P

    # Compute Jacobian coordinates from Projective coordinate
    ZP = random_non_zero_element(F) * ZP
    XP = XP * ZP**2
    YP = YP * ZP**3

    # Check everything works
    if not ZP.is_zero():
        assert P == E(XP / ZP**2, YP / ZP**3)
    else:
        assert P == E(0)

    return (XP, YP, ZP)


if __name__ == "__main__":
    proof.all(False)
    p = 19
    F = GF(p ^ 2)
    E = EllipticCurve(F, [0, 6, 0, 1, 0])
    A = E.a2()

    # Complete test for all points on a small order curve
    for P in E:
        P_jac = randomize_point_to_jacobian(P)

        # Check doubling formula for all P
        XR, YR, ZR = DBL(P_jac, A)
        if ZR.is_zero():
            assert 2 * P == E(0), "doubling failed!"
        else:
            R = E(XR / ZR**2, YR / ZR**3)
            assert R == 2 * P, "doubling failed!"

        # Check addition formula for all points P + Q
        for Q in E:
            u, v, w = ADD_components(P, Q, A)
            if not w.is_zero():
                assert (P + Q).x() == (u - v) / w
                assert (P - Q).x() == (u + v) / w
