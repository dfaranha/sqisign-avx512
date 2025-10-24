from elliptic_types import Curve, PointX, BasisX, PointJ
from theta_types import AVX_4x2x1, AVX_8x1x1
from theta_structure import (
    squaring,
    hadamard,
    pairwise_multiplication,
    pairwise_multiplication_3,
)
from utilities import apply_isomorphism

from elliptic_arithmetic import double_x, add_components
from utilities import batched_inversion, sqrt_fp2


def lift_basis_to_jacobian(E: Curve, basis: BasisX) -> tuple[PointJ, PointJ]:
    """
    Given an x-only basis, lift to <P, Q> in Jacobian coordinates
    """
    # Unpack coordinates and normalise P = (PX : 1)
    PX = basis.PX / basis.PZ
    QX, QZ = basis.QX, basis.QZ
    PQX, PQZ = basis.PmQX, basis.PmQZ

    # We need the Montgomery coefficient from the curve
    A = E.A / E.C  # TODO: remove inversion

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


def jacobian_to_xz(P: PointJ) -> PointX:
    """
    Convert a Jacobian point (X : Y : Z) to projective x-only (X : Z)
    """
    X, _, Z = P
    if Z.is_zero():
        return PointX(Z.parent().one(), Z)
    return PointX(X, Z * Z)


def gluing_kernel_to_x_only(
    P1: PointJ, P2: PointJ, Q1: PointJ, Q2: PointJ
) -> tuple[PointX, PointX, PointX, PointX]:
    """
    Helper function:

    Given the kernel ((P1, P2), (Q1, Q2)) map the Jacobian points
    onto the Kummer Line ready for codomain computation
    """
    xP1 = jacobian_to_xz(P1)
    xP2 = jacobian_to_xz(P2)
    xQ1 = jacobian_to_xz(Q1)
    xQ2 = jacobian_to_xz(Q2)

    return (xP1, xP2, xQ1, xQ2)


def _compute_matrix(X, Z, U, W, inv_det, inv_Z):
    """
    Helper function:

    Subroutine to compute the matrix given data from action_by_translation
    """
    d = U * Z * inv_det
    b = -W * Z * inv_det
    c = U * X * inv_det - X * inv_Z
    a = -d
    return (a, b, c, d)


def action_by_translation(
    E1: Curve, E2: Curve, P1_4: PointX, P2_4: PointX, Q1_4: PointX, Q2_4: PointX
):
    """
    Algorithm 8.36 of the SQISign spec.

    Compute the translation matrix induced by the generator of our order 4
    isotropic basis <(P1,P2)x(Q1,Q2)> in E_1 x E_2.

    Args:
        Pi, Qi are four torsion points on (E1 x E2)

    Returns:
        tuple: return the translation matrices corresponding to all input points
    """
    XP1, ZP1 = P1_4
    XQ1, ZQ1 = Q1_4
    UP1, WP1 = double_x(E1, P1_4)
    UQ1, WQ1 = double_x(E1, Q1_4)
    dP1 = WP1 * XP1 - UP1 * ZP1
    dQ1 = WQ1 * XQ1 - UQ1 * ZQ1

    XP2, ZP2 = P2_4
    XQ2, ZQ2 = Q2_4
    UP2, WP2 = double_x(E2, P2_4)
    UQ2, WQ2 = double_x(E2, Q2_4)
    dP2 = WP2 * XP2 - UP2 * ZP2
    dQ2 = WQ2 * XQ2 - UQ2 * ZQ2

    # invert all determinants and z-coords
    inverses = batched_inversion(*[dP1, dP2, dQ1, dQ2, ZP1, ZP2, ZQ1, ZQ2])

    # Compute matrices from points
    g1 = _compute_matrix(XP1, ZP1, UP1, WP1, inverses[0], inverses[4])
    g2 = _compute_matrix(XP2, ZP2, UP2, WP2, inverses[1], inverses[5])
    h1 = _compute_matrix(XQ1, ZQ1, UQ1, WQ1, inverses[2], inverses[6])
    h2 = _compute_matrix(XQ2, ZQ2, UQ2, WQ2, inverses[3], inverses[7])

    return g1, g2, h1, h2


def theta_change_of_basis(
    E1: Curve, E2: Curve, P1_4: PointX, P2_4: PointX, Q1_4: PointX, Q2_4: PointX
) -> tuple[AVX_8x1x1, AVX_8x1x1]:
    """
    Algorithm 8.37 of the SQIsign spec.

    Compute the change of matrix that transform the standard product
    theta structure into the theta structure induced by the 4-symplectic
    basis <(0,Q2_4),(P1_4,0)> + <(P1_4,P2_4),(Q1_4,Q2_4)>.

    Input:
        Pi, Qi are four torsion points on (E1 x E2)

    Returns:
        Matrix: the 4x4 change of basis matrix
    """
    g1, g2, h1, h2 = action_by_translation(E1, E2, P1_4, P2_4, Q1_4, Q2_4)

    t1 = g1[0] * h1[0] + g1[1] * h1[2]
    t2 = g1[2] * h1[0] + g1[3] * h1[2]
    t3 = g2[0] * h2[0] + g2[1] * h2[2]
    t4 = g2[2] * h2[0] + g2[3] * h2[2]

    a = g1[0] * g2[0] + h1[0] * h2[0] + t1 * t3 + 1
    b = g1[0] * g2[2] + h1[0] * h2[2] + t1 * t4
    c = g1[2] * g2[0] + h1[2] * h2[0] + t2 * t3
    d = g1[2] * g2[2] + h1[2] * h2[2] + t2 * t4

    a1 = h2[0] * a + h2[1] * b
    b1 = h2[2] * a + h2[3] * b
    c1 = h2[0] * c + h2[1] * d
    d1 = h2[2] * c + h2[3] * d

    a2 = g1[0] * a + g1[1] * c
    b2 = g1[0] * b + g1[1] * d
    c2 = g1[2] * a + g1[3] * c
    d2 = g1[2] * b + g1[3] * d

    a3 = g1[0] * a1 + g1[1] * c1
    b3 = g1[0] * b1 + g1[1] * d1
    c3 = g1[2] * a1 + g1[3] * c1
    d3 = g1[2] * b1 + g1[3] * d1

    return (
        AVX_8x1x1(a, b, c, d, a1, b1, c1, d1),
        AVX_8x1x1(a2, b2, c2, d2, a3, b3, c3, d3),
    )


def product_to_theta(P1: PointX, P2: PointX, N: tuple[AVX_8x1x1]) -> AVX_4x2x1:
    """
    Algorithm 8.38 of the SQISign spec.

    Given a couple of point P in E1xE2, compute their representative in the theta model.

    Args:
        P (tuple): A couple of elliptic points in E1xE2
        N (Matrix): the change of basis matrix describing our theta structure

    Returns:
        ThetaPoint: the image of theta structure
    """
    # extract X,Z coordinates on pairs of points
    X1, Z1 = P1
    X2, Z2 = P2

    # Pack into a ThetaPoint
    P = AVX_4x2x1(X1 * X2, X1 * Z2, Z1 * X2, Z1 * Z2)

    return apply_isomorphism(P, N)


def gluing_codomain(
    E1: Curve,
    E2: Curve,
    P1_8: PointJ,
    P2_8: PointJ,
    Q1_8: PointJ,
    Q2_8: PointJ,
) -> tuple[AVX_4x2x1, AVX_4x2x1, AVX_4x2x1, tuple[AVX_8x1x1, AVX_8x1x1]]:
    """
    Algorithm 8.39 of the SQIsign spec.
    """
    # Convert from the Jacobian points to x-only
    xP1_8, xP2_8, xQ1_8, xQ2_8 = gluing_kernel_to_x_only(P1_8, P2_8, Q1_8, Q2_8)

    # Double points to recover 4-torsion for change of basis
    xP1_4 = double_x(E1, xP1_8)
    xP2_4 = double_x(E2, xP2_8)
    xQ1_4 = double_x(E1, xQ1_8)
    xQ2_4 = double_x(E2, xQ2_8)

    # Compute the change of basis for our theta structure
    N = theta_change_of_basis(E1, E2, xP1_4, xP2_4, xQ1_4, xQ2_4)

    # Lift both 8-torsion points above the generator of our basis in the theta model.
    P1 = product_to_theta(xP1_8, xP2_8, N)
    P2 = product_to_theta(xQ1_8, xQ2_8, N)

    # compute their dual-image.
    P1 = squaring(P1)
    P2 = squaring(P2)

    P1 = hadamard(P1)
    P2 = hadamard(P2)

    # TODO: do parallel things below
    X1, Y1, Z1, W1 = P1
    X2, _, Z2, W2 = P2
    if not (W1.is_zero() and W2.is_zero()):
        raise ValueError("last coordinate is not zero")

    if any(x.is_zero() for x in [X1, X2, Y1, Z2]):
        raise ValueError("projective factors are zero")

    # ============================================================
    # Just like in the generic codomain there's a bit of ugliness
    # here and the below is my best effort to compute:
    #
    # A = X1 * X2
    # B = Y1 * X2
    # C = X1 * Z2

    # A_inv = Y1 * Z2
    # B_inv = C
    # C_inv = B

    # x = X1 * A_inv
    # y = Z1 * C_inv
    #
    # Which requires 6 multiplications and we can do it by doing
    # two calls to pairwise_multiplication with the last element
    # of t1 and t2 being 1 (so we only need 3/4 multiplications
    # from this function)
    # ============================================================

    t1 = AVX_4x2x1(X1, Y1, X1, 1)
    t2 = AVX_4x2x1(X2, X2, Z2, 1)

    A, B, C, _ = pairwise_multiplication_3(t1, t2)

    t1 = AVX_4x2x1(Y1, C, B, 1)
    t2 = AVX_4x2x1(Z2, Y1, Z1, 1)
    A_inv, x, y, _ = pairwise_multiplication_3(t1, t2)
    B_inv = C
    C_inv = B

    if (Y1 * B_inv != x) or (X2 * A_inv != Z2 * C_inv):
        raise ValueError("input kernel is not isotropic")

    OA = hadamard(AVX_4x2x1(A, B, C, 0))
    OA_dual_inv = AVX_4x2x1(A_inv, B_inv, C_inv, 0)
    J = AVX_4x2x1(x, x, y, y)

    return OA, OA_dual_inv, J, N


def gluing_eval(
    E1: Curve,
    E2: Curve,
    P1: PointJ,
    P2: PointJ,
    T1_8: PointJ,
    T2_8: PointJ,
    J: AVX_4x2x1,
    N: tuple[AVX_8x1x1, AVX_8x1x1],
) -> AVX_4x2x1:
    """
    Algorithm 8.40 of the SQIsign spec.
    """
    # Unpack x, y from the Theta Point
    # J = (x : x : y : y)
    assert J[0] == J[1]
    assert J[2] == J[3]
    x, y = J[0], J[2]

    # Add components
    u1, v1, w1 = add_components(E1, P1, T1_8)
    u2, v2, w2 = add_components(E2, P2, T2_8)

    # Perform S(N * U) and S(N * V)
    # We need the following 9 products which
    #
    # u1 * u2
    # u1 * v2
    # u1 * w2
    #
    # v1 * u2
    # v1 * v2
    # v1 * w2
    #
    # w1 * u2
    # w1 * v2
    # w1 * w2
    #
    # We also need to do two additions:
    # u1 * u2 + v1 * v2
    # v1 * u2 + u1 * v2
    #
    # If we use the normal AVX stuff this means 3 calls to
    # pairwise_multiplication and then addition as well.

    U = AVX_4x2x1(u1 * u2 + v1 * v2, u1 * w2, w1 * u2, w1 * w2)
    V = AVX_4x2x1(v1 * u2 + u1 * v2, v1 * w2, w1 * v2, 0)

    U = apply_isomorphism(U, N)
    V = apply_isomorphism(V, N)

    U = squaring(U)
    V = squaring(V)

    # Hadamard of component wise subtraction of U, V
    im_P = AVX_4x2x1(U.x - V.x, U.y - V.y, U.z - V.z, U.w - V.w)
    im_P = hadamard(im_P)

    # Scaling by (y : y : x : x)
    J_prime = AVX_4x2x1(y, y, x, x)
    im_P = pairwise_multiplication(im_P, J_prime)

    # Final Hadamard
    im_P = hadamard(im_P)

    return im_P
