from sage.all import Matrix

from elliptic_curves.kummer_line import KummerPoint
from utilities.batched_inversion import batched_inversion
from theta_model.theta_structures import ThetaPoint, ThetaNullPoint
from theta_model.utilities import apply_isomorphism
from elliptic_curves.jacobian_arithmetic import ADD_components


def _jacobian_to_xz(E, P):
    """
    Helper function:

    Convert a Jacobian point (X : Y : Z) to projective x-only (X : Z)
    """
    X, _, Z = P
    if Z.is_zero():
        return E.zero()
    return E(X, Z * Z)


def prepare_gluing_kernel(E1, E2, kernel):
    """
    Helper function:

    Given the kernel ((P1, P2), (Q1, Q2)) map the Jacobian points
    onto the Kummer Line ready for codomain computation
    """
    ((P1, P2), (Q1, Q2)) = kernel

    xP1 = _jacobian_to_xz(E1, P1)
    xP2 = _jacobian_to_xz(E2, P2)
    xQ1 = _jacobian_to_xz(E1, Q1)
    xQ2 = _jacobian_to_xz(E2, Q2)

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
    return Matrix(2, 2, [a, b, c, d])


def action_by_translation(P1_4, P2_4, Q1_4, Q2_4):
    """
    Algorithm 8.36 of the SQISign spec.

    Compute the translation matrix induced by the generator of our order 4
    isotropic basis <(P1,P2)x(Q1,Q2)> in E_1 x E_2.

    Args:
        Pi, Qi are four torsion points on (E1 x E2)

    Returns:
        tuple: return the translation matrices corresponding to all input points
    """
    XP1, ZP1 = P1_4.XZ()
    XQ1, ZQ1 = Q1_4.XZ()
    UP1, WP1 = P1_4.double().XZ()
    UQ1, WQ1 = Q1_4.double().XZ()
    dP1 = WP1 * XP1 - UP1 * ZP1
    dQ1 = WQ1 * XQ1 - UQ1 * ZQ1

    XP2, ZP2 = P2_4.XZ()
    XQ2, ZQ2 = Q2_4.XZ()
    UP2, WP2 = P2_4.double().XZ()
    UQ2, WQ2 = Q2_4.double().XZ()
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


def theta_change_of_basis(P1_4, P2_4, Q1_4, Q2_4):
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
    g1, g2, h1, h2 = action_by_translation(P1_4, P2_4, Q1_4, Q2_4)

    t1 = g1[0, 0] * h1[0, 0] + g1[0, 1] * h1[1, 0]
    t2 = g1[1, 0] * h1[0, 0] + g1[1, 1] * h1[1, 0]
    t3 = g2[0, 0] * h2[0, 0] + g2[0, 1] * h2[1, 0]
    t4 = g2[1, 0] * h2[0, 0] + g2[1, 1] * h2[1, 0]

    a = g1[0, 0] * g2[0, 0] + h1[0, 0] * h2[0, 0] + t1 * t3 + 1
    b = g1[0, 0] * g2[1, 0] + h1[0, 0] * h2[1, 0] + t1 * t4
    c = g1[1, 0] * g2[0, 0] + h1[1, 0] * h2[0, 0] + t2 * t3
    d = g1[1, 0] * g2[1, 0] + h1[1, 0] * h2[1, 0] + t2 * t4

    a1 = h2[0, 0] * a + h2[0, 1] * b
    b1 = h2[1, 0] * a + h2[1, 1] * b
    c1 = h2[0, 0] * c + h2[0, 1] * d
    d1 = h2[1, 0] * c + h2[1, 1] * d

    a2 = g1[0, 0] * a + g1[0, 1] * c
    b2 = g1[0, 0] * b + g1[0, 1] * d
    c2 = g1[1, 0] * a + g1[1, 1] * c
    d2 = g1[1, 0] * b + g1[1, 1] * d

    a3 = g1[0, 0] * a1 + g1[0, 1] * c1
    b3 = g1[0, 0] * b1 + g1[0, 1] * d1
    c3 = g1[1, 0] * a1 + g1[1, 1] * c1
    d3 = g1[1, 0] * b1 + g1[1, 1] * d1

    return Matrix([[a, b, c, d], [a1, b1, c1, d1], [a2, b2, c2, d2], [a3, b3, c3, d3]])


def product_to_theta(P, N):
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
    P1, P2 = P
    X1, Z1 = P1.XZ()
    X2, Z2 = P2.XZ()
    P = ThetaPoint(X1 * X2, X1 * Z2, Z1 * X2, Z1 * Z2)
    return apply_isomorphism(P, N)


def gluing_codomain(kernel: tuple[KummerPoint], verify=False):
    """
    Algorithm 8.39 of the SQIsign spec.

    Given a isotropic 8-basis above our gluing kernel in E1xE2, compute the codomain
    of our isogeny together with

    Args:
        kernel : x-only points ((P1, P2), (Q1, Q2)) on E1 x E1

    Returns:
        tuple:
            - “inverse” of the dual isogeneous theta null point
            - ThetaNullPoint corresponding to the codomain of our gluing isogeny,
            - dual image of the point (P1_8, P2_8)
            - change-of-basis matrix N
    """
    P1_8, P2_8, Q1_8, Q2_8 = kernel

    # Double points to recover 4-torsion for change of basis
    P1_4 = P1_8.double()
    P2_4 = P2_8.double()
    Q1_4 = Q1_8.double()
    Q2_4 = Q2_8.double()

    # Compute the change of basis for our theta structure
    N = theta_change_of_basis(P1_4, P2_4, Q1_4, Q2_4)

    # Lift both 8-torsion points above the generator of our basis in the theta model.
    P1 = product_to_theta((P1_8, P2_8), N)
    P2 = product_to_theta((Q1_8, Q2_8), N)

    # compute their dual-image.
    X1, Y1, Z1, W1 = P1.squaring_hadamard()
    X2, _, Z2, W2 = P2.squaring_hadamard()

    if verify and not (W1.is_zero() and W2.is_zero()):
        raise ValueError("last coordinate is not zero")

    if verify and any(x.is_zero() for x in [X1, X2, Y1, Z2]):
        raise ValueError("projective factors are zero")

    A = X1 * X2
    B = Y1 * X2
    C = X1 * Z2

    A_inv = Y1 * Z2
    B_inv = C
    C_inv = B

    x = X1 * A_inv
    y = Z1 * C_inv

    if verify and ((Y1 * B_inv != x) or (X2 * A_inv != Z2 * C_inv)):
        raise ValueError("input kernel is not isotropic")

    a2, b2, c2, d2 = ThetaPoint._hadamard_transform(A, B, C, 0)

    OA_dual_inv = ThetaPoint(A_inv, B_inv, C_inv, 0)
    J = ThetaPoint(x, x, y, y)
    OA = ThetaNullPoint(a2, b2, c2, d2)

    return OA_dual_inv, OA, J, N


def gluing_eval(domain, P, T_8, J, N):
    """
    Algorithm 8.40 of the SQIsign spec.

    Used to evaluate points though our gluing isogeny.

    Args:
        domain:                  : Montgomery coefficients A1 and A2
        P1  tuple(JacobianPoint) : A pair of points in E1 xE2
        T_8 (JacobianPoint)      : A pair of point in the 8-torsion point above the kernel of our gluing isogeny
        J (ThetaPoint)           : The dual image of T_8.
        N (Matrix)               : The change of basis matrix describing our theta structure

    Returns:
        ThetaPoint: the evaluation though the gluing isogeny of our point (P1, P2).
    """
    A1, A2 = domain
    P1, P2 = P
    T1_8, T2_8 = T_8

    # Unpack x, y from the Theta Point
    assert J[0] == J[1]
    assert J[2] == J[3]
    x, y = J[0], J[2]

    # Add components
    u1, v1, w1 = ADD_components(P1, T1_8, A1)
    u2, v2, w2 = ADD_components(P2, T2_8, A2)

    # Perform S(N * U) and S(N * V)
    U = ThetaPoint(u1 * u2 + v1 * v2, u1 * w2, w1 * u2, w1 * w2)
    V = ThetaPoint(v1 * u2 + u1 * v2, v1 * w2, w1 * v2, 0)

    U = apply_isomorphism(U, N)
    V = apply_isomorphism(V, N)

    U = ThetaPoint(U.squaring())
    V = ThetaPoint(V.squaring())

    # Hadamard of component wise subtraction of U, V
    X, Y, Z, W = ThetaPoint._hadamard_transform(
        U.x - V.x, U.y - V.y, U.z - V.z, U.w - V.w
    )

    # Scaling by (y : y : x : x)
    X, Y, Z, W = y * X, y * Y, x * Z, x * W

    # Final Hadamard
    im_x, im_y, im_z, im_w = ThetaPoint._hadamard_transform(X, Y, Z, W)

    return ThetaPoint(im_x, im_y, im_z, im_w)


def gluing_eval_special(P, I, N):
    """
    Algorithm 8.41 of the SQIsign spec.

    Used to evaluate points though our gluing isogeny.
        for couple points where one is zero.
    Args:
        P  (tuple[KummerPoint]): A pair of x-only points on E1 x E2
        I  (ThetaPoint)        : The “inverse” of the dual isogeneous theta null point
        N  (Matrix)            : The change of basis matrix describing our theta structure

    Returns:
        ThetaPoint: the evaluation though the gluing isogeny of our point (P1, P2).
    """
    P1, P2 = P[0], P[1]

    # Special method used to compute the image though our gluing isogeny
    # of couple points where one is the zero point.
    assert P1.is_zero() or P2.is_zero()

    # Map pair of points to theta point
    T = product_to_theta((P1, P2), N)

    # Compute H . S(T)
    X, Y, Z, W = T.squaring_hadamard()
    assert W.is_zero()

    # Scale by inverse element and compute Hadamard
    X, Y, Z, W = T._hadamard_transform(X * I.x, Y * I.y, Z * I.z, W)

    return ThetaPoint(X, Y, Z, W)
