from elliptic_curves.even_isogeny import (
    two_isogeny_codomain,
    two_isogeny_eval,
    two_isogeny_codomain_singular,
    two_isogeny_eval_singular,
    four_isogeny_codomain,
    four_isogeny_eval,
)
from elliptic_curves.kummer_line import KummerLine, KummerPoint


def __kummer_line_from_AC24(A24, C24):
    """
    Helper function to compute a KummerLine element

    Given (A24 : C24) compute (A : C) and from this the Kummer Line
    """
    A = A24 + A24
    A = A - C24
    A = A + A
    C = C24

    Fq = A24.parent()
    return KummerLine(Fq, [A, C])


def two_isogeny_chain(
    domain: KummerLine, kernel: KummerPoint, length: int, points: list[KummerPoint] = []
):
    """
    Algorithm 8.26 of the SQISign spec.

    Compute a 2-isogeny chain and optionally evaluate this isogeny on a given list of points
    """
    # Extract out constants from domain and kernel
    A24, C24 = domain.AC24()
    XK, ZK = kernel.XZ()

    # Parse list of points to list of coefficients
    points = [P.XZ() for P in points]

    # Init strategy arrays
    space = length.bit_length()
    strat_points = [domain.zero().XZ()] * space
    orders = [0] * space

    strat_points[0] = (XK, ZK)
    orders[0] = length
    k = 0

    # Main loop: Compute with balanced strategies
    for j in range(length // 2):
        # Double strategy points to correct order
        while orders[k] != 2:
            k = k + 1
            n = 2 * (orders[k - 1] // 4) + (orders[k - 1] % 2)
            orders[k] = orders[k - 1] - n
            strat_points[k] = strat_points[k - 1]
            for _ in range(n):
                strat_points[k] = kernel.xDBL(*strat_points[k], A24, C24)

        # Compute a four isogeny with the kth point in the strategy
        XK, ZK = strat_points[k]

        # For the first step of the isogeny, we check our points have the correct order
        if j == 0:
            X_test, Z_test = kernel.xDBL(XK, ZK, A24, C24)
            if Z_test.is_zero():
                raise ValueError("chain failed: kernel has wrong order")
            X_test, Z_test = kernel.xDBL(X_test, Z_test, A24, C24)
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
        X_test, Z_test = kernel.xDBL(XK, ZK, A24, C24)
        if ZK.is_zero() or not Z_test.is_zero():
            raise ValueError("chain failed: kernel has wrong order")
        if XK.is_zero():
            raise ValueError("chain failed: unexpected singular 2-isogeny")

        A24, C24 = two_isogeny_codomain(XK, ZK)
        points = [two_isogeny_eval(XK, ZK, XQ, ZQ) for (XQ, ZQ) in points]

    # Create the Kummer Line from the constants
    codomain = __kummer_line_from_AC24(A24, C24)

    # Create points of type KummerPoint from the tuples
    points = [codomain(P) for P in points]

    return codomain, points


def two_isogeny_chain_codomain(domain: KummerLine, kernel: KummerPoint, length: int):
    """
    Wrapper function which returns only the codomain of an isogeny chain
    """
    codomain, _ = two_isogeny_chain(domain, kernel, length)
    return codomain


def two_isogeny_chain_small(
    domain: KummerLine,
    kernel: KummerPoint,
    length: int,
    points: list[KummerPoint],
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
    A24, C24 = domain.AC24()
    XK, ZK = kernel.XZ()

    # Parse list of points to list of coefficients
    points = [P.XZ() for P in points]

    # Main loop, we compute the isogeny one step at a time
    for i in range(length):
        # Create point of order two for isogeny
        XP, ZP = XK, ZK
        for _ in range(length - i - 1):
            XP, ZP = kernel.xDBL(XP, ZP, A24, C24)

        # For the first step, we ensure the point has full order and handle singular isogenies if allowed
        if i == 0:
            _, Z_test = kernel.xDBL(XP, ZP, A24, C24)
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

    # Create the Kummer Line from the constants
    codomain = __kummer_line_from_AC24(A24, C24)

    # Create points of type KummerPoint from the tuples
    points = [codomain(P) for P in points]

    return codomain, points
