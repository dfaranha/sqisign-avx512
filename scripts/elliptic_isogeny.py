from elliptic_types import Curve, PointX
from elliptic_arithmetic import x_double
from utilities import sqrt_fp2


def curve_from_AC24(A24, C24):
    """
    Helper function to compute a KummerLine element

    Given (A24 : C24) compute (A : C) and from this the Kummer Line
    """
    A = A24 + A24
    A = A - C24
    A = A + A
    C = C24
    return Curve(A, C, A24, C24)


def two_isogeny_codomain(XP, ZP):
    """
    Algorithm 8.20 of the SQISign spec.

    Compute the codomain of the 2-isogeny E -> E/<P> for P != (0 : 1)
    of order 2.
    """
    A24_prime = XP * XP
    C24_prime = ZP * ZP
    A24_prime = C24_prime - A24_prime
    return A24_prime, C24_prime


def two_isogeny_eval(XP, ZP, XQ, ZQ):
    """
    Algorithm 8.21 of the SQISign spec.

    Evaluate a point Q (XQ : ZQ) under the action of the 2-isogeny E -> E/<P>
    for P != (0 : 1) of order 2
    """
    t0 = XP + ZP
    t1 = XP - ZP
    t2 = XQ + ZQ
    t3 = XQ - ZQ

    t0 = t0 * t3
    t1 = t1 * t2
    t2 = t0 + t1
    t3 = t0 - t1

    XQ_prime = XQ * t2
    ZQ_prime = ZQ * t3

    return XQ_prime, ZQ_prime


def two_isogeny_codomain_singular(A24, C24):
    """
    Algorithm 8.22 of the SQISign spec.

    Compute the codomain of the 2-isogeny E -> E/<P> for P = (0 : 1)
    """
    t0 = A24 + A24
    t0 - t0 - C24
    t0 = t0 + t0
    t0 = t0 / C24
    c0 = t0
    A24_prime = t0 + t0
    t0 = t0 - 4
    t0 = sqrt_fp2(t0)
    c1 = -t0
    C24_prime = t0 + t0
    A24_prime = A24_prime + C24_prime
    C24_prime = C24_prime + C24_prime

    return (A24_prime, C24_prime), (c0, c1)


def two_isogeny_eval_singular(XP, ZP, c0, c1):
    """
    Algorithm 8.23 of the SQISign spec.

    Evaluate a point P (XP : ZP) under the action of the 2-isogeny E -> E/<P>
    """
    t0 = XP * ZP
    t1 = c0 * ZP
    t1 = t1 + XP
    t1 = t1 * XP
    XQ = ZP * ZP
    XQ = XQ + t1
    ZQ = t0 * c1

    return XQ, ZQ


def four_isogeny_codomain(XP, ZP):
    """
    Algorithm 8.24 of the SQISign spec.

    Compute the codomain of the 4-isogeny E -> E/<P> such that [2]P != (0 : 1)
    """
    c0 = ZP * ZP
    c1 = XP - ZP
    c2 = XP + ZP
    t0 = XP * XP
    t3 = c0 + t0
    t4 = c0 - t0
    A24_prime = t3 * t4
    C24_prime = c0 * c0
    c0 = c0 + c0
    c0 = c0 + c0

    return (A24_prime, C24_prime), (c0, c1, c2)


def four_isogeny_eval(XQ, ZQ, c0, c1, c2):
    """
    Algorithm 8.25 of the SQISign spec.

    Evaluate a point Q (XQ : ZQ) under the action of the 4-isogeny E -> E/<P>
    for [2]P != (0 : 1) using precomputed constants
    """
    t0 = XQ + ZQ
    t1 = XQ - ZQ
    XQ_prime = t0 * c1
    ZQ_prime = t1 * c2
    t0 = t0 * t1
    t0 = t0 * c0
    t1 = XQ_prime + ZQ_prime
    ZQ_prime = XQ_prime - ZQ_prime
    t1 = t1 * t1
    ZQ_prime = ZQ_prime * ZQ_prime
    XQ_prime = t0 + t1
    t0 = t0 - ZQ_prime
    XQ_prime = XQ_prime * t1
    ZQ_prime = ZQ_prime * t0

    return XQ_prime, ZQ_prime


def two_isogeny_chain(
    domain: Curve, kernel: PointX, length: int, points: list[PointX] = []
):
    """
    Algorithm 8.26 of the SQISign spec.

    Compute a 2-isogeny chain and optionally evaluate this isogeny on a given list of points
    """
    # Extract out constants from domain and kernel
    A24, C24 = domain.A24, domain.C24
    XK, ZK = kernel

    # Init strategy arrays
    space = length.bit_length()
    strat_points = [PointX(0, 0)] * space
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
            PX, PZ = strat_points[k - 1]
            for _ in range(n):
                PX, PZ = x_double(PX, PZ, A24, C24)
            strat_points[k] = PointX(PX, PZ)

        # Compute a four isogeny with the kth point in the strategy
        XK, ZK = strat_points[k]

        # For the first step of the isogeny, we check our points have the correct order
        if j == 0:
            X_test, Z_test = x_double(XK, ZK, A24, C24)
            if Z_test.is_zero():
                raise ValueError("chain failed: kernel has wrong order")
            X_test, Z_test = x_double(X_test, Z_test, A24, C24)
            if not Z_test.is_zero():
                raise ValueError("chain failed: kernel has wrong order")

        (A24, C24), (c0, c1, c2) = four_isogeny_codomain(XK, ZK)
        strat_points = [
            four_isogeny_eval(XQ, ZQ, c0, c1, c2) for (XQ, ZQ) in strat_points
        ]
        orders = [m - 2 for m in orders]
        k = k - 1

        points = [four_isogeny_eval(XQ, ZQ, c0, c1, c2) for (XQ, ZQ) in points]

    # Compute a 2-isogeny if the length is odd
    if length % 2:
        XK, ZK = strat_points[0]

        # Ensure K has order 2 and is not the point (0 : 1)
        X_test, Z_test = x_double(XK, ZK, A24, C24)
        if ZK.is_zero() or not Z_test.is_zero():
            raise ValueError("chain failed: kernel has wrong order")
        if XK.is_zero():
            raise ValueError("chain failed: unexpected singular 2-isogeny")

        A24, C24 = two_isogeny_codomain(XK, ZK)
        points = [two_isogeny_eval(XK, ZK, XQ, ZQ) for (XQ, ZQ) in points]

    # Create the Kummer Line from the constants
    codomain = curve_from_AC24(A24, C24)
    points = [PointX(XQ, ZQ) for (XQ, ZQ) in points]

    return codomain, points


def two_isogeny_chain_codomain(domain: Curve, kernel: PointX, length: int):
    """
    Wrapper function which returns only the codomain of an isogeny chain
    """
    codomain, _ = two_isogeny_chain(domain, kernel, length)
    return codomain


def two_isogeny_chain_small(
    domain: Curve,
    kernel: PointX,
    length: int,
    points: list[Curve],
    allow_singular: bool = False,
):
    """
    Algorithm 8.27 of the SQISign spec.

    Compute a 2-isogeny chain with a naive approach and optionally evaluate
    this isogeny on a given list of points. Unlike the faster chain, this
    function also allows the computation of a so-called singular isogeny
    with kernel (0 : 1) which can occur during signing.
    """
    # Extract out constants from domain and kernel
    A24, C24 = domain.A24, domain.C24
    XK, ZK = kernel

    # Main loop, we compute the isogeny one step at a time
    for i in range(length):
        # Create point of order two for isogeny
        XP, ZP = XK, ZK
        for _ in range(length - i - 1):
            XP, ZP = x_double(XP, ZP, A24, C24)

        # For the first step, we ensure the point has full order and handle singular isogenies if allowed
        if i == 0:
            _, Z_test = x_double(XP, ZP, A24, C24)
            if not Z_test.is_zero():
                raise ValueError("kernel does not have order 2^length")

            # The point of order two is (0 : 1) which means we need singular isogenies
            if XP.is_zero():
                if not allow_singular:
                    raise ValueError("unexpected singular 2-isogeny")

                # Compute the codomain
                (A24, C24), (c0, c1) = two_isogeny_codomain_singular(A24, C24)

                # Push through the kernel and any other points
                XK, ZK = two_isogeny_eval_singular(XK, ZK, c0, c1)
                points = [
                    two_isogeny_eval_singular(XP, ZP, c0, c1) for (XP, ZP) in points
                ]

                # Go to the next step of the loop and skip the code below
                continue

        A24, C24 = two_isogeny_codomain(XP, ZP)
        XK, ZK = two_isogeny_eval(XP, ZP, XK, ZK)
        points = [two_isogeny_eval(XP, ZP, XQ, ZQ) for (XQ, ZQ) in points]

    codomain = curve_from_AC24(A24, C24)
    points = [PointX(XQ, ZQ) for (XQ, ZQ) in points]

    return codomain, points
