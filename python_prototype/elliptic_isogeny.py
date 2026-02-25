from elliptic_types import Curve, PointX
from elliptic_arithmetic import x_double_1x2x4x1, x_double_n_1x2x4x1
from avx_types import AVX_2x4x1, AVX_8x1x1
from avx_arithmetic import (
    avx_241_sqr,
    avx_241_mul,
    avx_241_sub,
    avx_241_add_sub,
    avx_241_add,
    avx_241_to_811,
    avx_811_to_241,
    avx_811_add_sub,
    avx_811_sqr,
    avx_811_mul,
)


def curve_from_AC24(domain: AVX_2x4x1) -> Curve:
    """
    Helper function to compute a KummerLine element

    Given (A24 : C24) compute (A : C) and from this the Kummer Line
    """
    A24, C24 = domain

    A = A24 + A24
    A = A - C24
    A = A + A
    C = C24
    return Curve(A, C, A24, C24)


def two_isogeny_codomain(P: AVX_2x4x1) -> AVX_2x4x1:
    """
    Algorithm 8.20 of the SQISign spec.

    Compute the codomain of the 2-isogeny E -> E/<P> for P != (0 : 1)
    of order 2.
    """
    PP = avx_241_sqr(P)
    T0 = AVX_2x4x1(PP[1], PP[1])
    T1 = AVX_2x4x1(PP[0], 0)
    A24 = avx_241_sub(T0, T1)
    return A24


def two_isogeny_eval(P: AVX_2x4x1, Q: AVX_2x4x1) -> AVX_2x4x1:
    """
    Algorithm 8.21 of the SQISign spec.

    Evaluate a point Q (XQ : ZQ) under the action of the 2-isogeny E -> E/<P>
    for P != (0 : 1) of order 2
    """
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

    return avx_241_mul(Q, t2_t3)


def four_isogeny_codomain(P: AVX_2x4x1) -> tuple[AVX_2x4x1]:
    """
    Algorithm 8.24 of the SQISign spec.

    Compute the codomain of the 4-isogeny E -> E/<P> such that [2]P != (0 : 1)
    """

    T0 = AVX_2x4x1(P[0], P[0])
    T1 = AVX_2x4x1(P[1], P[1])
    c2_c1 = avx_241_add_sub(T0, T1)

    t0_t1 = avx_241_sqr(P)
    A24_C24 = avx_241_sqr(t0_t1)

    T0 = AVX_2x4x1(t0_t1[1], A24_C24[1])
    T1 = AVX_2x4x1(t0_t1[1], A24_C24[0])
    t1_A24 = avx_241_add_sub(T0, T1)

    T0 = AVX_2x4x1(t1_A24[0], 0)
    c0_00 = avx_241_add(T0, T0)

    A24_prime = AVX_2x4x1(t1_A24[1], A24_C24[1])
    C01 = AVX_2x4x1(c0_00[0], c2_c1[1])
    C2 = AVX_2x4x1(c2_c1[0], 0)

    return A24_prime, C01, C2


def four_isogeny_eval(Q: AVX_2x4x1, C01: AVX_2x4x1, C2: AVX_2x4x1) -> AVX_2x4x1:
    """
    Algorithm 8.25 of the SQISign spec.

    Evaluate a point Q (XQ : ZQ) under the action of the 4-isogeny E -> E/<P>
    for [2]P != (0 : 1) using precomputed constants
    """
    T0 = AVX_2x4x1(Q[0], Q[0])
    T1 = AVX_2x4x1(Q[1], Q[1])
    t0_t1 = avx_241_add_sub(T0, T1)

    T0 = AVX_2x4x1(t0_t1[0], t0_t1[0])
    T1 = AVX_2x4x1(C01[1], t0_t1[1])
    XQ_t0 = avx_241_mul(T0, T1)

    T0 = AVX_2x4x1(t0_t1[1], XQ_t0[1])
    T1 = AVX_2x4x1(C2[0], C01[0])
    ZQ_t0 = avx_241_mul(T0, T1)

    T0 = AVX_2x4x1(XQ_t0[0], XQ_t0[0])
    T1 = AVX_2x4x1(ZQ_t0[0], ZQ_t0[0])
    t1_ZQ = avx_241_add_sub(T0, T1)

    t1_ZQ = avx_241_sqr(t1_ZQ)

    T0 = AVX_2x4x1(ZQ_t0[1], ZQ_t0[1])
    T1 = AVX_2x4x1(t1_ZQ[0], t1_ZQ[1])
    XQ_t0 = avx_241_add_sub(T0, T1)

    T0 = AVX_2x4x1(XQ_t0[0], t1_ZQ[1])
    T1 = AVX_2x4x1(t1_ZQ[0], XQ_t0[1])
    XQ_ZQ = avx_241_mul(T0, T1)

    return XQ_ZQ


def four_isogeny_eval_quad(points: tuple[AVX_2x4x1], C01: AVX_2x4x1, C2: AVX_2x4x1):
    """
    Evaluate four points in parallel using 8-way vectorisation
    """
    # Pack the four points into a single 8x1x1 blob
    P_Q_R_S = avx_241_to_811(*points)

    # only need 3 of the 8 elements here but we'll have to convert to 8x1x1
    C012 = avx_241_to_811(C01, C2, C01, C2)

    # Note the ADD SUB uses (+ + + + - - - -)
    # Compute X + Z, X - Z for four points
    T0 = AVX_8x1x1(
        P_Q_R_S[0],
        P_Q_R_S[2],
        P_Q_R_S[4],
        P_Q_R_S[6],
        P_Q_R_S[0],
        P_Q_R_S[2],
        P_Q_R_S[4],
        P_Q_R_S[6],
    )
    T1 = AVX_8x1x1(
        P_Q_R_S[1],
        P_Q_R_S[3],
        P_Q_R_S[5],
        P_Q_R_S[7],
        P_Q_R_S[1],
        P_Q_R_S[3],
        P_Q_R_S[5],
        P_Q_R_S[7],
    )
    t0_t1 = avx_811_add_sub(T0, T1)

    # (X + Z) * C1, (X + Z) * (X - Z)
    T0 = AVX_8x1x1(
        t0_t1[0], t0_t1[0], t0_t1[1], t0_t1[1], t0_t1[2], t0_t1[2], t0_t1[3], t0_t1[3]
    )
    T1 = AVX_8x1x1(
        C012[1], t0_t1[4], C012[1], t0_t1[5], C012[1], t0_t1[6], C012[1], t0_t1[7]
    )
    XQ_t0 = avx_811_mul(T0, T1)

    # (X - Z) * C2 and (X + Z) * (X - Z) * C0
    T0 = AVX_8x1x1(
        t0_t1[4], XQ_t0[1], t0_t1[5], XQ_t0[3], t0_t1[6], XQ_t0[5], t0_t1[7], XQ_t0[7]
    )
    T1 = AVX_8x1x1(
        C012[2], C012[0], C012[2], C012[0], C012[2], C012[0], C012[2], C012[0]
    )
    ZQ_t0 = avx_811_mul(T0, T1)

    # Note the ADD SUB uses (+ + + + - - - -)
    T0 = AVX_8x1x1(
        XQ_t0[0], XQ_t0[2], XQ_t0[4], XQ_t0[6], XQ_t0[0], XQ_t0[2], XQ_t0[4], XQ_t0[6]
    )
    T1 = AVX_8x1x1(
        ZQ_t0[0], ZQ_t0[2], ZQ_t0[4], ZQ_t0[6], ZQ_t0[0], ZQ_t0[2], ZQ_t0[4], ZQ_t0[6]
    )
    t1_ZQ = avx_811_add_sub(T0, T1)
    t1_ZQ = avx_811_sqr(t1_ZQ)

    # Note the ADD SUB uses (+ + + + - - - -)
    T0 = AVX_8x1x1(
        ZQ_t0[1], ZQ_t0[3], ZQ_t0[5], ZQ_t0[7], ZQ_t0[1], ZQ_t0[3], ZQ_t0[5], ZQ_t0[7]
    )
    T1 = AVX_8x1x1(
        t1_ZQ[0], t1_ZQ[1], t1_ZQ[2], t1_ZQ[3], t1_ZQ[4], t1_ZQ[5], t1_ZQ[6], t1_ZQ[7]
    )
    XQ_t0 = avx_811_add_sub(T0, T1)

    T0 = AVX_8x1x1(
        XQ_t0[0], t1_ZQ[4], XQ_t0[1], t1_ZQ[5], XQ_t0[2], t1_ZQ[6], XQ_t0[3], t1_ZQ[7]
    )
    T1 = AVX_8x1x1(
        t1_ZQ[0], XQ_t0[4], t1_ZQ[1], XQ_t0[5], t1_ZQ[2], XQ_t0[6], t1_ZQ[3], XQ_t0[7]
    )
    im_P_Q_R_S = avx_811_mul(T0, T1)

    # Convert back to 2x4x1 points
    im_P, im_Q, im_R, im_S = avx_811_to_241(im_P_Q_R_S)

    return im_P, im_Q, im_R, im_S


def four_isogeny_eval_array(points: tuple[AVX_2x4x1], C01: AVX_2x4x1, C2: AVX_2x4x1):
    # We eval `n` points by evaluating four at a time with four_isogeny_eval_quad
    # and then the remaining `n mod 4` points with four_isogeny_eval
    chunks = len(points) // 4

    images = []
    for i in range(chunks):
        chunk = points[i * 4 : (i + 1) * 4]
        images.extend(four_isogeny_eval_quad(chunk, C01, C2))

    for Q in points[chunks * 4 :]:
        images.append(four_isogeny_eval(Q, C01, C2))

    return images


def two_isogeny_chain_1x2x4x1(
    domain: AVX_2x4x1, kernel: AVX_2x4x1, length: int, points: list[AVX_2x4x1] = []
):
    # Extract out constants from domain
    A24 = domain

    # Init strategy arrays
    space = length.bit_length()
    strat_points = [AVX_2x4x1(0, 0)] * space
    orders = [0] * space

    strat_points[0] = kernel
    orders[0] = length
    k = 0

    # Main loop: Compute with balanced strategies
    for j in range(length // 2):
        # Double strategy points to correct order
        while orders[k] != 2:
            k = k + 1
            n = 2 * (orders[k - 1] // 4) + (orders[k - 1] % 2)
            orders[k] = orders[k - 1] - n
            strat_points[k] = x_double_n_1x2x4x1(A24, strat_points[k - 1], n)

        # Compute a four isogeny with the kth point in the strategy
        ker = strat_points[k]

        # For the first step of the isogeny, we check our points have the correct order
        if j == 0:
            tmp = x_double_1x2x4x1(A24, ker)
            if tmp[1].is_zero():
                raise ValueError("1, chain failed: kernel has wrong order")
            tmp = x_double_1x2x4x1(A24, tmp)
            if not tmp[1].is_zero():
                raise ValueError("2, chain failed: kernel has wrong order")

        A24, C01, C2 = four_isogeny_codomain(ker)
        strat_points[:k] = four_isogeny_eval_array(strat_points[:k], C01, C2)
        orders = [m - 2 for m in orders]
        k = k - 1

        points = four_isogeny_eval_array(points, C01, C2)

    # Compute a 2-isogeny if the length is odd
    if length % 2:
        ker = strat_points[0]

        # Ensure K has order 2 and is not the point (0 : 1)
        tmp = x_double_1x2x4x1(A24, ker)
        if ker[1].is_zero() or not tmp[1].is_zero():
            raise ValueError("chain failed: kernel has wrong order")
        if ker[0].is_zero():
            raise ValueError("chain failed: unexpected singular 2-isogeny")

        A24 = two_isogeny_codomain(ker)
        points = [two_isogeny_eval(ker, Q) for Q in points]

    # Create the Kummer Line from the constants
    codomain = curve_from_AC24(A24)
    points = [PointX(XQ, ZQ) for (XQ, ZQ) in points]

    return codomain, points


def two_isogeny_chain_codomain_1x2x4x1(
    domain: AVX_2x4x1, kernel: AVX_2x4x1, length: int
):
    """
    Wrapper function which returns only the codomain of an isogeny chain
    """
    codomain, _ = two_isogeny_chain_1x2x4x1(domain, kernel, length)
    return codomain
