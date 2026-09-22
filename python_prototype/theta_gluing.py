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
    avx_421_add_sub,
    avx_421_sub_add,
    avx_421_dbl,
    avx_421_had,
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


def compute_symmetric_elements(X_4: AVX_4x2x1, Z_4: AVX_4x2x1):
    """
    Compute the three symmetric elements above the 4-torsion basis of each curve
    of the elliptic product.

    Each element is packed as (g_1.s00, g_1.s01, g_2.s00, g_2.s01), where the
    subscript is the curve; the remaining entry s10 is +/- s01 and is recovered
    by the caller. The other outputs are (delta_1, r1_1, delta_2, r1_2),
    (r2_1, r2_2, delta_1 * delta_2, -) and the two conditional swap flags.

    8M + 4S + 32a
    """
    F = X_4[0].parent()
    zero = F.zero()
    zeta = F.gen()

    # ============================================================
    # Locate, for each curve, which of T1, T2, T1 + T2 lies above
    # the two-torsion point (0 : 1). The six possible answers are
    # encoded as pos in 0..5.
    # ============================================================

    # 8a
    R = avx_421_add(X_4, Z_4)  # (r1_1, r2_1, r1_2, r2_2)
    S = avx_421_sub(X_4, Z_4)  # (s1_1, s2_1, s1_2, s2_2)

    # 4M
    T0 = AVX_4x2x1(R[0], S[0], R[2], S[2])
    T1 = AVX_4x2x1(S[1], R[1], S[3], R[3])
    T2 = avx_421_mul(T0, T1)  # (P_1, Q_1, P_2, Q_2)

    # 4a
    T3 = AVX_4x2x1(zeta * T2[0], zeta * T2[0], zeta * T2[2], zeta * T2[2])
    T4 = AVX_4x2x1(T2[1], T2[1], T2[3], T2[3])
    U = avx_421_had(T3, T4)

    pos = []
    for c in range(2):
        p = 255
        if S[2 * c] == 0:
            p = 0
        if R[2 * c] == 0:
            p = 1
        if S[2 * c + 1] == 0:
            p = 2
        if R[2 * c + 1] == 0:
            p = 3
        if U[2 * c] == 0:
            p = 4
        if U[2 * c + 1] == 0:
            p = 5
        pos.append(p)

    swap12 = [(p >> 1) == 0 for p in pos]
    swap23 = [(p >> 1) == 2 for p in pos]

    # ============================================================
    # Every symmetric element is built from r1 = x^2 + z^2,
    # r2 = 2xz and delta = x^2 - z^2 of the torsion point which is
    # not the one above (0 : 1).
    # ============================================================

    # 4S
    e0 = 1 if swap12[0] else 0
    e1 = 3 if swap12[1] else 2
    XZ = AVX_4x2x1(X_4[e0], Z_4[e0], X_4[e1], Z_4[e1])
    SQ = avx_421_sqr(XZ)

    # 4a
    T5 = AVX_4x2x1(SQ[0], SQ[0], SQ[2], SQ[2])
    T6 = AVX_4x2x1(SQ[1], SQ[1], SQ[3], SQ[3])
    DR = avx_421_sub_add(T5, T6)  # (delta_1, r1_1, delta_2, r1_2)

    # 4M + 4a
    XZ2 = avx_421_dbl(XZ)
    T7 = AVX_4x2x1(XZ2[0], XZ2[2], DR[0], DR[0])
    T8 = AVX_4x2x1(XZ[1], XZ[3], DR[2], DR[2])
    T9 = avx_421_mul(T7, T8)  # (r2_1, r2_2, delta_1 * delta_2, -)

    RD = AVX_4x2x1(DR[1], T9[0], DR[3], T9[1])  # (r1_1, r2_1, r1_2, r2_2)

    # ============================================================
    # The three elements before the conditional swaps, in the order
    # (-r1, r2), (0, +/- delta) and (+/- i r2, -/+ i r1).
    # ============================================================
    nRD = avx_421_neg(RD)
    nDR = avx_421_neg(DR)
    iRD = AVX_4x2x1(*[zeta * a for a in RD])
    niRD = avx_421_neg(iRD)

    def sel(b, a0, a1):
        return a1 if b else a0

    odd = [p & 1 for p in pos]
    flg1 = [p in (0, 3, 4) for p in pos]
    flg2 = [p in (1, 2, 5) for p in pos]

    Ea = AVX_4x2x1(nRD[0], RD[1], nRD[2], RD[3])
    Eb = AVX_4x2x1(
        zero, sel(odd[0], nDR[0], DR[0]), zero, sel(odd[1], nDR[2], DR[2])
    )
    Ec = AVX_4x2x1(
        sel(flg1[0], niRD[1], iRD[1]),
        sel(flg2[0], niRD[0], iRD[0]),
        sel(flg1[1], niRD[3], iRD[3]),
        sel(flg2[1], niRD[2], iRD[2]),
    )

    def blend(A, B, flags):
        return AVX_4x2x1(
            sel(flags[0], A[0], B[0]),
            sel(flags[0], A[1], B[1]),
            sel(flags[1], A[2], B[2]),
            sel(flags[1], A[3], B[3]),
        )

    g1 = blend(Ea, Eb, swap12)
    g2 = blend(blend(Eb, Ec, swap23), Ea, swap12)
    g3 = blend(Ec, Eb, swap23)

    return g1, g2, g3, DR, T9, swap12, swap23


def theta_change_of_basis(X_4: AVX_4x2x1, Z_4: AVX_4x2x1) -> tuple[AVX_8x1x1, AVX_8x1x1]:
    """
    Compute the change of basis matrix that transforms the standard product
    theta structure into the theta structure induced by the 4-torsion basis
    <(P1_4, P2_4), (Q1_4, Q2_4)> of E1 x E2, from the symmetric elements
    above that basis.

    Input:
        X_4, Z_4 are the 4-torsion points (P1_4, Q1_4, P2_4, Q2_4).

    Returns:
        Matrix: the 4x4 change of basis matrix as two 8x1x1 blobs.

    8M + 4S + 32a for the symmetric elements
    48M + 24a for the matrix itself
    56M + 4S + 56a
    """
    F = X_4[0].parent()
    zero = F.zero()

    g1, g2, g3, DR, T9, swap12, swap23 = compute_symmetric_elements(X_4, Z_4)
    delta12 = T9[2]

    # The third entry of a symmetric element is s01 up to sign; the sign is
    # positive exactly for the element built from delta.
    g1a_s10 = g1[1] if swap12[0] else -g1[1]
    g2b_s10 = -g2[3] if (swap12[1] or swap23[1]) else g2[3]

    # First row: the product of the symmetric elements of the two curves.
    # 12M + 12a
    L1 = AVX_4x2x1(g1[0], g1[0], g1[1], g1[1])
    R1 = AVX_4x2x1(g1[2], g1[3], g1[2], g1[3])
    L2 = AVX_4x2x1(g2[0], g2[0], g2[1], g2[1])
    R2 = AVX_4x2x1(g2[2], g2[3], g2[2], g2[3])
    L3 = AVX_4x2x1(g3[0], g3[0], g3[1], g3[1])
    R3 = AVX_4x2x1(g3[2], g3[3], g3[2], g3[3])

    P1, P2 = avx_421_pair_mul(L1, R1, L2, R2)
    P3 = avx_421_mul(L3, R3)
    row0 = avx_421_sub(avx_421_add(P1, P2), P3)
    row0 = avx_421_add(row0, AVX_4x2x1(delta12, zero, zero, zero))

    # Second and third rows: act on the first row by g2 of the second curve
    # and by g1 of the first curve respectively.
    # 16M + 8a
    A1 = AVX_4x2x1(g2[2], g2[3], g2[2], g2[3])
    B1 = AVX_4x2x1(row0[0], row0[0], row0[2], row0[2])
    A2 = AVX_4x2x1(g2b_s10, g2[2], g2b_s10, g2[2])
    B2 = AVX_4x2x1(row0[1], row0[1], row0[3], row0[3])
    A3 = AVX_4x2x1(g1[0], g1[0], g1[1], g1[1])
    B3 = AVX_4x2x1(row0[0], row0[1], row0[0], row0[1])
    A4 = AVX_4x2x1(g1a_s10, g1a_s10, g1[0], g1[0])
    B4 = AVX_4x2x1(row0[2], row0[3], row0[2], row0[3])

    X1, X3 = avx_421_pair_mul(A1, B1, A3, B3)
    X2, X4 = avx_421_pair_mul(A2, B2, A4, B4)
    row1 = avx_421_had(X1, X2)
    row2 = avx_421_add_sub(X3, X4)

    # Fourth row: act on the second row by g1 of the first curve.
    # 8M + 4a
    B5 = AVX_4x2x1(row1[0], row1[1], row1[0], row1[1])
    B6 = AVX_4x2x1(row1[2], row1[3], row1[2], row1[3])
    X5, X6 = avx_421_pair_mul(A3, B5, A4, B6)
    row3 = avx_421_add_sub(X5, X6)

    # Realign the projective factors of the four rows.
    # 12M
    D12 = AVX_4x2x1(delta12, delta12, delta12, delta12)
    D1 = AVX_4x2x1(DR[0], DR[0], DR[0], DR[0])
    D2 = AVX_4x2x1(DR[2], DR[2], DR[2], DR[2])
    row0, row1 = avx_421_pair_mul(row0, D12, row1, D1)
    row2 = avx_421_mul(row2, D2)

    return AVX_8x1x1(*row0, *row1), AVX_8x1x1(*row2, *row3)


def product_to_theta_pair(
    X: AVX_4x2x1, Z: AVX_4x2x1, N: tuple[AVX_8x1x1]
) -> tuple[AVX_4x2x1]:
    """
    Algorithm 8.38 of the SQISign spec.

    40M + 24a
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

    24S + 116M + 132a
    """
    # 4S
    X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)

    # Double points to recover 4-torsion for change of basis
    # 8S + 8M + 16a
    X_4, Z_4 = x_double_4x4x2x1(A24_12, X_8, Z_8)

    # Compute the change of basis for our theta structure
    # 56M + 4S + 56a
    N = theta_change_of_basis(X_4, Z_4)

    # Lift both 8-torsion points above the generator of our basis in the theta model.
    # 40M + 24a
    P12 = product_to_theta_pair(X_8, Z_8, N)

    # compute their dual-image.
    # 8S + 16a
    P12 = avx_811_sqr(P12)
    P12 = hadamard_pair(P12)

    X1, Y1, Z1, W1, X2, _, Z2, W2 = P12

    if verify:
        # 8a
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

    # 4M
    t1 = AVX_4x2x1(X1, Y1, X1, 1)
    t2 = AVX_4x2x1(X2, X2, Z2, 1)
    A, B, C, _ = avx_421_mul(t1, t2)

    # 4M
    t1 = AVX_4x2x1(Y1, C, B, 1)
    t2 = AVX_4x2x1(Z2, Y1, Z1, 1)
    A_inv, x, y, _ = avx_421_mul(t1, t2)
    B_inv = C
    C_inv = B

    # Ensure that Y1 * B_inv == x and X2 * A_inv == Z2 * C_inv
    if verify:
        # 4M + 4a
        T1 = AVX_4x2x1(Y1, X2, Z2, 1)
        T2 = AVX_4x2x1(B_inv, A_inv, C_inv, 1)
        T3 = avx_421_mul(T1, T2)

        T1 = AVX_4x2x1(T3[0], T3[1], 0, 0)
        T2 = AVX_4x2x1(x, T3[2], 0, 0)
        if not avx_421_eq(T1, T2):
            raise ValueError("input kernel is not isotropic")

    # 8a
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

    32S + 138M + 124a
    """
    # Add components
    # u1, v1, w1 = add_components(E1, P1, T1_8)
    # u3, v3, w3 = add_components(E1, Q1, T1_8)
    # u2, v2, w2 = add_components(E2, P2, T2_8)
    # u4, v4, w4 = add_components(E2, Q2, T2_8)

    # 16S + 48M + 28a
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

    # 8M
    TT0 = AVX_8x1x1(u1, u1, w1, w1, u3, u3, w3, w3)
    TT1 = AVX_8x1x1(u2, w2, u2, w2, u4, w4, u4, w4)
    u1u2, u1w2, w1u2, w1w2, u3u4, u3w4, w3u4, w3w4 = avx_811_mul(TT0, TT1)

    # 8M
    TT0 = AVX_8x1x1(v1, u1, v1, w1, v3, u3, v3, w3)
    TT1 = AVX_8x1x1(u2, v2, w2, v2, u4, v4, w4, v4)
    v1u2, u1v2, v1w2, w1v2, v3u4, u3v4, v3w4, w3v4 = avx_811_mul(TT0, TT1)

    # 2M
    T0 = AVX_4x2x1(v1, v3, 0, 0)
    T1 = AVX_4x2x1(v2, v4, 0, 0)
    v1v2, v3v4, _, _ = avx_421_mul(T0, T1)  # Two redundant mul here.

    # 8a
    T0 = AVX_4x2x1(u1u2, u3u4, v1u2, v3u4)
    T1 = AVX_4x2x1(v1v2, v3v4, u1v2, u3v4)
    u1u2_v1v2, u3u4_v3v4, v1u2_u1v2, v3u4_u3v4 = avx_421_add(T0, T1)

    U12 = AVX_8x1x1(u1u2_v1v2, u1w2, w1u2, w1w2, u3u4_v3v4, u3w4, w3u4, w3w4)
    V12 = AVX_8x1x1(v1u2_u1v2, v1w2, w1v2, 0, v3u4_u3v4, v3w4, w3v4, 0)

    # Apply the isomorphism by N on U1, U2, V1, V2
    # 64 M + 48a
    U12 = apply_isomorphism_pair(U12, N)
    V12 = apply_isomorphism_pair(V12, N)

    # 16S
    U12 = avx_811_sqr(U12)
    V12 = avx_811_sqr(V12)

    # Hadamard of component wise subtraction of U, V
    # 24a
    im_PQ = avx_811_sub(U12, V12)
    im_PQ = hadamard_pair(im_PQ)

    # Scaling by (y : y : x : x) | (y : y : x : x)
    # 8M
    im_PQ = avx_811_mul(im_PQ, J_prime)

    # Final Hadamard
    # 16a
    im_PQ = hadamard_pair(im_PQ)

    return im_PQ


def eval_product_to_theta(P1: PointX, P2: PointX, N: tuple[AVX_8x1x1]):
    """
    Given a pair of x-only points, compute the corresponding theta point
    NOTE: here we translate from 64 to 51 bit limbs.

    36M + 24a
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

    40M + 24a
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
