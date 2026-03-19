from elliptic_types import PointX
from avx_types import AVX_4x2x1, AVX_8x1x1
from avx_arithmetic import (
    avx_421_sqr,
    avx_421_mul,
    avx_421_add,
    avx_421_sub,
    avx_421_neg,
    avx_811_mul,
    avx_811_sub,
    avx_421_pair_mul,
    avx_811_sqr,
    avx_421_all_zero,
    avx_421_any_zero,
    avx_421_eq,
)
from theta_structure import (
    hadamard,
    hadamard_pair,
)
from utilities import apply_isomorphism, apply_isomorphism_pair

from elliptic_arithmetic import x_double_4x4x2x1, four_add_components


def gluing_kernel_to_x_only(X: AVX_4x2x1, Z: AVX_4x2x1) -> tuple[AVX_4x2x1, AVX_4x2x1]:
    """
    Helper function:

    Given the kernel as the X, Y, Z coordinates of Jacobian coordinates
    return the x-only coordinates (X : Z)
    """
    return X, avx_421_sqr(Z)


def _compute_matrix(
    X: AVX_4x2x1,
    Z: AVX_4x2x1,
    U: AVX_4x2x1,
    W: AVX_4x2x1,
    det: AVX_4x2x1,
) -> tuple[AVX_4x2x1]:
    """
    Helper function:

    Subroutine to compute the matrix given data from action_by_translation
    Uses an alternative projective representation to avoid all inversions.

    With inverses precomputed we need five calls to "pairwise_mul" so there
    is an extra cost of a single avx_sqr call
    """
    # ============================================================================
    # For the ith coordinate of X, Z, U, W we compute ai, bi, ci, di as:
    # d = U * Z * inv_det
    # b = -W * Z * inv_det
    # c = U * X * inv_det - X * inv_Z
    # a = -d
    #
    # We want to work projectively here to remove the inversion.
    #
    # So let's write this with a (1 / det * Z) outside the front.
    # (1 / det * Z)  [- U * Z^2              -W * Z^2]
    #                [  U * X * Z - X * det   U * Z^2]
    # We can then return the matrix along with the projective factor.
    # ============================================================================

    C, pi = avx_421_pair_mul(U, Z, det, Z)
    C = avx_421_sub(C, det)
    C, Z_sqr = avx_421_pair_mul(X, C, Z, Z)  # This is 1M 1S
    D, B = avx_421_pair_mul(U, Z_sqr, W, Z_sqr)
    B = avx_421_neg(B)
    A = avx_421_neg(D)

    # TODO:
    # Memory management from the ai, bi, ci, di to the
    # matrix gi, hi, can be totally avoided as theta_change_of_basis
    # unpacks these anyway.
    a1, a2, a3, a4 = A
    b1, b2, b3, b4 = B
    c1, c2, c3, c4 = C
    d1, d2, d3, d4 = D

    g1 = AVX_4x2x1(a1, b1, c1, d1)
    h1 = AVX_4x2x1(a2, b2, c2, d2)
    g2 = AVX_4x2x1(a3, b3, c3, d3)
    h2 = AVX_4x2x1(a4, b4, c4, d4)

    return g1, g2, h1, h2, pi


def action_by_translation(
    A24: AVX_4x2x1, X_4: AVX_4x2x1, Z_4: AVX_4x2x1
) -> tuple[AVX_4x2x1]:
    """
    Algorithm 8.36 of the SQISign spec.

    Compute the translation matrix induced by the generator of our order 4
    isotropic basis <(P1,P2)x(Q1,Q2)> in E_1 x E_2.
    """
    U, W = x_double_4x4x2x1(A24, X_4, Z_4)
    t1, t2 = avx_421_pair_mul(W, X_4, U, Z_4)
    D = avx_421_sub(t1, t2)

    # Compute matrices from points
    g1, g2, h1, h2, proj = _compute_matrix(X_4, Z_4, U, W, D)

    return g1, g2, h1, h2, proj


def theta_change_of_basis(
    A24: AVX_4x2x1, X_4: AVX_4x2x1, Z_4: AVX_4x2x1
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

    Giac Notes:

    Computes N projectively which removes 21M + 1I from a batched inversion
    but costs 4S and 34 multiplications (so swaps one inversion for 13M + 1S)

    For AXV, 34 is not a multiple of four, so this adds 9 calls to avx_pairwise_mul
    and one call to avx_sqr.

    An interesting thing to explore is `avx_pairwise_mul` where one of the AVX blobs
    is four copies of the same element. This happens for 6 of the 9 total calls for
    avx_pairwise_mul.
    """
    # NOTE
    #
    # This function is long and complicated and parallelising computations has only
    # made it harder to follow. All inline comments trying to explain what is happening
    # mathematically have been abandoned and instead an attempt has been made to
    # simplify the discussion where AVX4x2x1 blobs are constructed and then used for
    # arithmetic.
    #
    # The implementation follows the logic of the following code:
    # https://github.com/GiacomoPope/isogeny_rs/blob/main/src/theta/theta_gluing.rs#L54
    # which is a decent overview of projective computation of the change of basis matrix.
    #
    # On notation:
    #
    # For the following I use T0, T1, etc to represent temp AVX4x2x1 values
    # which are essentially immediately discarded and replaced.
    #
    # Values named S0, S1, etc are instead values which we might need throughout
    # this function and in particular we select various fp2 values from these
    # AVX4x2x1 blobs to make additional Ti and Si values.
    #
    # As a result, this function will not be optimal with avoid stack allocation as
    # there will be points where Si values could no longer be user potentially, but
    # the choice has been made to absorb this loss to make it easier to follow.

    # Compute the action by translation on all four elements of E[4] simultaneously
    G1, G2, H1, H2, PI = action_by_translation(A24, X_4, Z_4)

    # Computation of S0
    T0 = AVX_4x2x1(PI[0], PI[2], PI[0], PI[1])
    T1 = AVX_4x2x1(PI[1], PI[3], PI[2], PI[3])
    S0 = avx_421_mul(T0, T1)

    # Computation of S1
    T0 = AVX_4x2x1(G1[0], G1[2], G2[0], G2[2])
    T1 = AVX_4x2x1(H1[0], H1[0], H2[0], H2[0])
    T2 = AVX_4x2x1(G1[1], G1[3], G2[1], G2[3])
    T3 = AVX_4x2x1(H1[2], H1[2], H2[2], H2[2])
    S1, T2 = avx_421_pair_mul(T0, T1, T2, T3)
    S1 = avx_421_add(S1, T2)

    # Computation of S2 (continued), S2, S3
    T0 = AVX_4x2x1(S0[1], S0[1], S0[0], S0[0])
    S2 = AVX_4x2x1(G1[0], G1[0], G1[2], G1[2])
    T1 = AVX_4x2x1(G2[0], G2[2], G2[0], G2[2])
    S1, S3 = avx_421_pair_mul(S1, T0, S2, T1)

    # Computation of S3 (continued), S4
    T0 = AVX_4x2x1(H1[0], H1[0], H1[2], H1[2])
    S4 = AVX_4x2x1(H2[0], H2[2], H2[0], H2[2])
    # TODO: every element of T2 is the same, can we use this?
    T1 = AVX_4x2x1(S0[3], S0[3], S0[3], S0[3])
    S3, T1 = avx_421_pair_mul(S3, T1, T0, S4)

    # Computation of S3 (continued), S5
    # TODO: every element of T0 is the same, can we use this?
    T0 = AVX_4x2x1(S0[2], S0[2], S0[2], S0[2])
    T2 = AVX_4x2x1(PI[0], S0[0], 1, 1)
    T3 = AVX_4x2x1(PI[3], S0[1], 1, 1)
    T1, S5 = avx_421_pair_mul(T1, T0, T2, T3)
    S3 = avx_421_add(S3, T1)

    # TODO: this could be optimised as we need only one Fp2 add?
    T0 = AVX_4x2x1(S5[1], 0, 0, 0)
    S3 = avx_421_add(S3, T0)

    # TODO: every element of T1 is the same, can we use this?
    T0 = AVX_4x2x1(S5[1], S5[1], S5[1], S5[1])
    T1 = AVX_4x2x1(S1[0], S1[0], S1[1], S1[1])
    T2 = AVX_4x2x1(S1[2], S1[3], S1[2], S1[3])
    S3, T4 = avx_421_pair_mul(S3, T0, T1, T2)
    S3 = avx_421_add(S3, T4)

    # Computation of S6
    S6 = AVX_4x2x1(S3[0], S3[0], S3[2], S3[2])
    T0 = AVX_4x2x1(S3[1], S3[1], S3[3], S3[3])
    T1 = AVX_4x2x1(H2[1], H2[3], H2[1], H2[3])
    S6, T2 = avx_421_pair_mul(S6, S4, T0, T1)
    S6 = avx_421_add(S6, T2)

    # Computation of S7, S8
    S7 = AVX_4x2x1(S3[0], S3[1], S3[0], S3[1])
    S8 = AVX_4x2x1(G1[1], G1[1], G1[3], G1[3])
    T0 = AVX_4x2x1(S3[2], S3[3], S3[2], S3[3])
    S7, T1 = avx_421_pair_mul(S7, S2, S8, T0)
    S7 = avx_421_add(S7, T1)

    # Computation of S9
    S9 = AVX_4x2x1(S6[0], S6[1], S6[0], S6[1])
    T0 = AVX_4x2x1(S6[2], S6[3], S6[2], S6[3])
    S9, T1 = avx_421_pair_mul(S9, S2, S8, T0)
    S9 = avx_421_add(S9, T1)

    # TODO: every element of T0 is the same, can we use this?
    T0 = AVX_4x2x1(S5[0], S5[0], S5[0], S5[0])
    S3 = avx_421_mul(S3, T0)

    T0 = AVX_4x2x1(PI[0], PI[0], PI[0], PI[0])
    T1 = AVX_4x2x1(PI[3], PI[3], PI[3], PI[3])
    S6, S7 = avx_421_pair_mul(S6, T0, S7, T1)

    return (
        AVX_8x1x1(S3[0], S3[1], S3[2], S3[3], S6[0], S6[1], S6[2], S6[3]),
        AVX_8x1x1(S7[0], S7[1], S7[2], S7[3], S9[0], S9[1], S9[2], S9[3]),
    )


def product_to_theta_pair(
    X: AVX_4x2x1, Z: AVX_4x2x1, N: tuple[AVX_8x1x1]
) -> tuple[AVX_4x2x1]:
    """
    Algorithm 8.38 of the SQISign spec.
    """
    # extract X,Z coordinates on pairs of points
    X1, X2, X3, X4 = X
    Z1, Z2, Z3, Z4 = Z

    t1 = AVX_8x1x1(X1, X1, Z1, Z1, X2, X2, Z2, Z2)
    t2 = AVX_8x1x1(X3, Z3, X3, Z3, X4, Z4, X4, Z4)
    P12 = avx_811_mul(t1, t2)

    P12 = apply_isomorphism_pair(P12, N)

    return P12


def gluing_codomain(
    A24_12: AVX_4x2x1, X_8: AVX_4x2x1, Z_8: AVX_4x2x1, verify: bool = False
) -> tuple[AVX_4x2x1, AVX_4x2x1, AVX_8x1x1, tuple[AVX_8x1x1, AVX_8x1x1]]:
    """
    Algorithm 8.39 of the SQIsign spec.
    """
    X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)

    # Double points to recover 4-torsion for change of basis
    X_4, Z_4 = x_double_4x4x2x1(A24_12, X_8, Z_8)

    # Compute the change of basis for our theta structure
    N = theta_change_of_basis(A24_12, X_4, Z_4)

    # Lift both 8-torsion points above the generator of our basis in the theta model.
    P12 = product_to_theta_pair(X_8, Z_8, N)

    # compute their dual-image.
    P12 = avx_811_sqr(P12)
    P12 = hadamard_pair(P12)

    X1, Y1, Z1, W1, X2, _, Z2, W2 = P12

    if verify:
        all_zero = AVX_4x2x1(W1, W2, W1, W2)
        no_zero = AVX_4x2x1(X1, X2, Y1, Z2)

        if not avx_421_all_zero(all_zero):
            raise ValueError("last coordinate is not zero")

        if avx_421_any_zero(no_zero):
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
    A, B, C, _ = avx_421_mul(t1, t2)

    t1 = AVX_4x2x1(Y1, C, B, 1)
    t2 = AVX_4x2x1(Z2, Y1, Z1, 1)
    A_inv, x, y, _ = avx_421_mul(t1, t2)
    B_inv = C
    C_inv = B

    # Ensure that Y1 * B_inv == x and X2 * A_inv == Z2 * C_inv
    if verify:
        T1 = AVX_4x2x1(Y1, X2, Z2, 1)
        T2 = AVX_4x2x1(B_inv, A_inv, C_inv, 1)
        T3 = avx_421_mul(T1, T2)

        T1 = AVX_4x2x1(T3[0], T3[1], 0, 0)
        T2 = AVX_4x2x1(x, T3[2], 0, 0)
        if not avx_421_eq(T1, T2):
            raise ValueError("input kernel is not isotropic")

    OA = hadamard(AVX_4x2x1(A, B, C, 0))
    OA_dual_inv = AVX_4x2x1(A_inv, B_inv, C_inv, 0)
    J_prime = AVX_8x1x1(y, y, x, x, y, y, x, x)

    return OA, OA_dual_inv, J_prime, N


def gluing_eval_pair(
    A_12: AVX_4x2x1,
    P: tuple[AVX_4x2x1],
    T8: tuple[AVX_4x2x1],
    J_prime: AVX_8x1x1,
    N: tuple[AVX_8x1x1, AVX_8x1x1],
) -> AVX_4x2x1:
    """
    Same as the above but accepts a pair of points on E1 x E2 as
    (P1, P2) and (Q1, Q2)
    """
    # Add components
    # u1, v1, w1 = add_components(E1, P1, T1_8)
    # u3, v3, w3 = add_components(E1, Q1, T1_8)
    # u2, v2, w2 = add_components(E2, P2, T2_8)
    # u4, v4, w4 = add_components(E2, Q2, T2_8)

    U, V, W = four_add_components(A_12, P, T8)

    # ===============================================================
    # We want to compute four additions and 9 multiplications as best
    # as we can...
    #
    # U1 = AVX_4x2x1(u1 * u2 + v1 * v2, u1 * w2, w1 * u2, w1 * w2)
    # U2 = AVX_4x2x1(u3 * u4 + v3 * v4, u3 * w4, w3 * u4, w3 * w4)
    # V1 = AVX_4x2x1(v1 * u2 + u1 * v2, v1 * w2, w1 * v2, 0)
    # V2 = AVX_4x2x1(v3 * u4 + u3 * v4, v3 * w4, w3 * v4, 0)
    # ===============================================================

    u1, u3, u2, u4 = U
    v1, v3, v2, v4 = V
    w1, w3, w2, w4 = W

    TT0 = AVX_8x1x1(u1, u1, w1, w1, u3, u3, w3, w3)
    TT1 = AVX_8x1x1(u2, w2, u2, w2, u4, w4, u4, w4)
    u1u2, u1w2, w1u2, w1w2, u3u4, u3w4, w3u4, w3w4 = avx_811_mul(TT0, TT1)

    TT0 = AVX_8x1x1(v1, u1, v1, w1, v3, u3, v3, w3)
    TT1 = AVX_8x1x1(u2, v2, w2, v2, u4, v4, w4, v4)
    v1u2, u1v2, v1w2, w1v2, v3u4, u3v4, v3w4, w3v4 = avx_811_mul(TT0, TT1)

    T0 = AVX_4x2x1(v1, v3, 0, 0)
    T1 = AVX_4x2x1(v2, v4, 0, 0)
    v1v2, v3v4, _, _ = avx_421_mul(T0, T1)  # Two redundant mul here.

    T0 = AVX_4x2x1(u1u2, u3u4, v1u2, v3u4)
    T1 = AVX_4x2x1(v1v2, v3v4, u1v2, u3v4)
    u1u2_v1v2, u3u4_v3v4, v1u2_u1v2, v3u4_u3v4 = avx_421_add(T0, T1)

    U12 = AVX_8x1x1(u1u2_v1v2, u1w2, w1u2, w1w2, u3u4_v3v4, u3w4, w3u4, w3w4)
    V12 = AVX_8x1x1(v1u2_u1v2, v1w2, w1v2, 0, v3u4_u3v4, v3w4, w3v4, 0)

    # Apply the isomorphism by N on U1, U2, V1, V2
    U12 = apply_isomorphism_pair(U12, N)
    V12 = apply_isomorphism_pair(V12, N)

    U12 = avx_811_sqr(U12)
    V12 = avx_811_sqr(V12)

    # Hadamard of component wise subtraction of U, V
    im_PQ = avx_811_sub(U12, V12)
    im_PQ = hadamard_pair(im_PQ)

    # Scaling by (y : y : x : x) | (y : y : x : x)
    im_PQ = avx_811_mul(im_PQ, J_prime)

    # Final Hadamard
    im_PQ = hadamard_pair(im_PQ)

    return im_PQ


def eval_product_to_theta(P1: PointX, P2: PointX, N: tuple[AVX_8x1x1]):
    """
    Given a pair of x-only points, compute the corresponding theta point
    NOTE: here we translate from 64 to 51 bit limbs.
    """
    X1, Z1 = P1
    X2, Z2 = P2
    P = AVX_4x2x1(X1, X1, Z1, Z1)
    T0 = AVX_4x2x1(X2, Z2, X2, Z2)
    P = avx_421_mul(P, T0)

    return apply_isomorphism(P, N)


def eval_product_to_theta_pair(
    P1: PointX, P2: PointX, Q1: PointX, Q2: PointX, N: tuple[AVX_8x1x1]
):
    """
    Given a pair of x-only points, compute the corresponding theta point
    NOTE: here we translate from 64 to 51 bit limbs.
    """
    X1, Z1 = P1
    X2, Z2 = P2
    X3, Z3 = Q1
    X4, Z4 = Q2

    PQ = AVX_8x1x1(X1, X1, Z1, Z1, X3, X3, Z3, Z3)
    T0 = AVX_8x1x1(X2, Z2, X2, Z2, X4, Z4, X4, Z4)
    PQ = avx_811_mul(PQ, T0)

    return apply_isomorphism_pair(PQ, N)


def gluing_eval_special(
    P1: PointX, P2: PointX, OA_inv: AVX_4x2x1, N: tuple[AVX_8x1x1]
) -> AVX_4x2x1:
    """
    Evaluation of couple points of the form (P1, 0) or (0, P2)
    which are the only form needed for SQIsign
    """
    assert P1.Z.is_zero() or P2.Z.is_zero()

    # Here we translate from x-only points with 64 bit limbs to a theta point
    # with 51 bit limbs
    T = eval_product_to_theta(P1, P2, N)
    T = avx_421_sqr(T)
    T = hadamard(T)
    assert T.w.is_zero()  # this is the special case we must have
    T = avx_421_mul(T, OA_inv)
    T = hadamard(T)

    return T


def gluing_eval_special_pair(
    P1: PointX,
    P2: PointX,
    Q1: PointX,
    Q2: PointX,
    OA_inv: AVX_8x1x1,
    N: tuple[AVX_8x1x1],
) -> AVX_4x2x1:
    """
    Evaluation of couple points of the form (P1, 0) or (0, P2)
    which are the only form needed for SQIsign
    """
    assert P1.Z.is_zero() or P2.Z.is_zero()
    assert Q1.Z.is_zero() or Q2.Z.is_zero()

    # Here we translate from x-only points with 64 bit limbs to a theta point
    # with 51 bit limbs
    PQ = eval_product_to_theta_pair(P1, P2, Q1, Q2, N)
    PQ = avx_811_sqr(PQ)
    PQ = hadamard_pair(PQ)
    assert PQ.w1.is_zero()  # this is the special case we must have
    assert PQ.w2.is_zero()  # this is the special case we must have

    PQ = avx_811_mul(PQ, OA_inv)
    PQ = hadamard_pair(PQ)

    return PQ
