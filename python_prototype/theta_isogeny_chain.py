from avx_types import AVX_4x2x1
from elliptic_types import Curve, BasisX, PointX

from avx_arithmetic import avx_811_to_421, avx_421_to_811
from elliptic_arithmetic import (
    lift_normalised_basis_to_jacobian,
    four_jac_to_ws,
    four_double_ws_n,
    four_ws_to_jac,
)
from theta_gluing import (
    gluing_codomain,
    gluing_eval_pair,
    gluing_eval_special,
    gluing_eval_special_pair,
)
from theta_superglue import (
    superglue_codomain,
    superglue_eval_pair,
)
from theta_isogeny import (
    theta_precompute,
    theta_double,
    theta_double_pair,
    theta_codomain_4x2x1,
    theta_codomain_8x1x1,
    theta_eval,
    theta_eval_pair,
    theta_codomain_with_four_torsion,
    theta_codomain_with_no_torsion,
)

from theta_splitting import (
    splitting_isomorphism_matrix,
    theta_to_product,
    theta_product_point_to_montgomery,
    randomized_null_point,
)
from utilities import batched_inversion, apply_isomorphism, apply_isomorphism_pair

# ================================================================================
# Helper functions for isogeny chains
# ================================================================================


def normalise_chain_input(
    domain: tuple[Curve], kernel: tuple[BasisX, BasisX]
) -> tuple[Curve, Curve, BasisX, BasisX]:
    """
    Given a domain E1 x E2 and a two x-only bases on these curves
    we normalize x(P1) and x(P2), and additionally the curve
    constants
    """
    E1, E2 = domain
    B1, B2 = kernel

    inv_P1_Z, inv_P2_Z, inv_E1_C, inv_E1_C24, inv_E2_C, inv_E2_C24 = batched_inversion(
        B1.PZ, B2.PZ, E1.C, E1.C24, E2.C, E2.C24
    )

    E1 = Curve(E1.A * inv_E1_C, 1, E1.A24 * inv_E1_C24, 1)
    E2 = Curve(E2.A * inv_E2_C, 1, E2.A24 * inv_E2_C24, 1)
    B1 = BasisX(B1.PX * inv_P1_Z, 1, B1.QX, B1.QZ, B1.PmQX, B1.PmQZ)
    B2 = BasisX(B2.PX * inv_P2_Z, 1, B2.QX, B2.QZ, B2.PmQX, B2.PmQZ)

    return E1, E2, B1, B2


def compute_product_doubles(
    A_12: AVX_4x2x1, strat_points: list[tuple[AVX_4x2x1]], orders: list[int], k: int
):
    # When we start doubling, only the first element is defined. We convert this
    # from the Montgomery to the Weierstrass model
    X_M, Y_M, Z_M = strat_points[0]
    X, Y, Z, U, A12_over_3 = four_jac_to_ws(A_12, X_M, Y_M, Z_M)

    # Now we store within strat_points the Weierstrass coordinates instead.
    # In C I guess this is slightly more annoying as we need different allocation?
    # In fact, we could just store X, Y, Z in here and have U as an extra AXV blob
    # which we keep somewhere in the stack simply within the loop, as we don't use
    # it for the conversion back in the end anyway.
    strat_points[0] = X, Y, Z, U

    # All doubling is performed in the Weierstrass model without conversions
    while orders[k] != 1:
        k = k + 1
        # TODO: need to balance this in the C code to find the optional
        # splitting of the chain, as eval is more expensive than doubling
        if orders[k - 1] >= 16:
            n = orders[k - 1] // 2
        else:
            n = orders[k - 1] - 1

        X, Y, Z, T = strat_points[k - 1]
        strat_points[k] = four_double_ws_n(X, Y, Z, T, n)
        orders[k] = orders[k - 1] - n

    # Now we must convert the Weierstrass model back to the Montgomery model
    # Which is done only on the points we have doubled so far.
    for i in range(k + 1):
        X, Y, Z, _ = strat_points[i]
        strat_points[i] = four_ws_to_jac(X, Y, Z, A12_over_3)

    return strat_points, orders, k


# ================================================================================
# Isogeny chains where we have the 2^n torsion only (requires sqrts)
# ================================================================================


def theta_isogeny_chain_4x2x1(
    domain: tuple[Curve],
    kernel: tuple[BasisX, BasisX],
    length: int,
    eval_points: list[tuple[PointX, PointX]] = [],
    randomise: bool = False,
):
    # We normalise points and the curve parameters the below functions stay with
    # 64 bit limbs to make use of the faster(?) sqrt we have available
    E1, E2, B1, B2 = normalise_chain_input(domain, kernel)
    P1, Q1 = lift_normalised_basis_to_jacobian(E1, B1)
    P2, Q2 = lift_normalised_basis_to_jacobian(E2, B2)

    # We are now in the position to convert from 64 to 51 bit limbs
    # For the curve we need to pack the Montgomery coefficient for
    # arithmetic.
    A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)

    # For the Jacobian points, we pack the coordinates into three blobs
    ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X)
    ker_Y = AVX_4x2x1(P1.Y, Q1.Y, P2.Y, Q2.Y)
    ker_Z = AVX_4x2x1(P1.Z, Q1.Z, P2.Z, Q2.Z)

    # Initalise the strategy arrays
    space = length.bit_length() + 1
    strat_points = [(None, None, None)] * space
    orders = [0] * space

    # Set the first values
    strat_points[0] = [ker_X, ker_Y, ker_Z]
    orders[0] = length - 2
    k = 0

    # Step One:
    # Perform repeated doubling of the Jacobian points and store intermediate results
    # to push through the isogeny.
    strat_points, orders, k = compute_product_doubles(A_12, strat_points, orders, k)

    # Compute the gluing codomain
    X_8, Y_8, Z_8 = strat_points[k]
    OA, OA_inv, J, N = gluing_codomain(A24_12, X_8, Z_8)

    # Package the point we use for translation
    T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    T8 = (T8_X, T8_Y, T8_Z)

    # Push kernel though the gluing isogeny
    for i in range(k):
        P = strat_points[i]
        imPQ = gluing_eval_pair(A_12, P, T8, J, N)
        imP, imQ = avx_811_to_421(imPQ)
        strat_points[i] = (imP, imQ)
        orders[i] = orders[i] - 1

    # Push other points through gluing isogeny
    eval_points = [gluing_eval_special(P1, P2, OA_inv, N) for (P1, P2) in eval_points]

    # Step Two:
    # Compute all but the last two steps of the isogeny without square roots
    k = k - 1
    j = 0
    while k >= 0 and orders[k] != 0:
        j = j + 1
        C1, C2 = theta_precompute(OA)

        # Perform the doubling
        while orders[k] != 1:
            k = k + 1
            n = orders[k - 1] // 2
            ker_P, ker_Q = strat_points[k - 1]
            for _ in range(n):
                ker_P = theta_double(ker_P, C1, C2)
                ker_Q = theta_double(ker_Q, C1, C2)
            strat_points[k] = (ker_P, ker_Q)
            orders[k] = orders[k - 1] - n

        # As we avoid the last two steps of the chain in this loop, h1 and h2
        # are fixed values
        h1, h2 = False, True

        # Compute the codomain from the kernel
        ker_P, ker_Q = strat_points[k]
        OA, inv = theta_codomain_4x2x1(ker_P, ker_Q, h1, h2)

        # Push kernel though the gluing isogeny
        for i in range(k):
            P, Q = strat_points[i]
            P = theta_eval(P, inv, h1, h2)
            Q = theta_eval(Q, inv, h1, h2)
            strat_points[i] = (P, Q)
            orders[i] = orders[i] - 1
        k = k - 1

        # Push the images through the kernel
        eval_points = [theta_eval(P, inv, h1, h2) for P in eval_points]

    # Step Three:
    # we now do the last two steps outside of the strategy. As we do not
    # have the torsion above we require square roots to compute the last
    # two codomains.
    if length > 3:
        P, _ = strat_points[0]
        ker = theta_eval(P, inv, h1, h2)
    else:
        ker, _ = strat_points[0]

    h1, h2 = False, False
    OA, inv = theta_codomain_with_four_torsion(ker, OA, h1, h2)
    eval_points = [theta_eval(P, inv, h1, h2) for P in eval_points]

    h1, h2 = True, False
    OA, inv = theta_codomain_with_no_torsion(OA, h1, h2)
    eval_points = [theta_eval(P, inv, h1, h2) for P in eval_points]

    # Step Four:
    # Finish the chain with an isomorphism to a theta point with a product
    # structure for efficient splitting

    # Compute the splitting isomorphism
    M = splitting_isomorphism_matrix(OA)

    # Apply the splitting isomorphism
    OA = apply_isomorphism(OA, M)
    eval_points = [apply_isomorphism(P, M) for P in eval_points]

    if randomise:
        OA, N = randomized_null_point(OA)
        eval_points = [apply_isomorphism(P, N) for P in eval_points]

    # Split the null point to the Montgomery coefficients
    (A1, C1), (A2, C2) = theta_to_product(OA)
    images = [theta_product_point_to_montgomery(P, OA) for P in eval_points]

    return (A1, C1), (A2, C2), images


def theta_isogeny_chain_8x1x1(
    domain: tuple[Curve],
    kernel: tuple[BasisX, BasisX],
    length: int,
    eval_points: list[tuple[PointX, PointX]] = [],
    randomise: bool = False,
):
    # We normalise points and the curve parameters the below functions stay with
    # 64 bit limbs to make use of the faster(?) sqrt we have available
    E1, E2, B1, B2 = normalise_chain_input(domain, kernel)
    P1, Q1 = lift_normalised_basis_to_jacobian(E1, B1)
    P2, Q2 = lift_normalised_basis_to_jacobian(E2, B2)

    # We are now in the position to convert from 64 to 51 bit limbs
    # For the curve we need to pack the Montgomery coefficient for
    # arithmetic.
    A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)

    # For the Jacobian points, we pack the coordinates into three blobs
    ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X)
    ker_Y = AVX_4x2x1(P1.Y, Q1.Y, P2.Y, Q2.Y)
    ker_Z = AVX_4x2x1(P1.Z, Q1.Z, P2.Z, Q2.Z)

    # Initalise the strategy arrays
    space = length.bit_length() + 1
    strat_points = [(None, None, None)] * space
    orders = [0] * space

    # Set the first values
    strat_points[0] = [ker_X, ker_Y, ker_Z]
    orders[0] = length - 2
    k = 0

    # If we can evaluate images in pairs we save time, so given the eval_points as input
    # we want to take pairs of inputs into eval_points_pairs and then if len(eval_points)
    # there'll be a single point in eval_point_single
    nk = len(eval_points)
    eval_pairs = [(eval_points[i], eval_points[i + 1]) for i in range(0, nk - 1, 2)]
    eval_single = [] if nk % 2 == 0 else [eval_points[-1]]

    # Step One:
    # Perform repeated doubling of the Jacobian points and store intermediate results
    # to push through the isogeny.
    strat_points, orders, k = compute_product_doubles(A_12, strat_points, orders, k)

    # Compute the gluing codomain
    X_8, Y_8, Z_8 = strat_points[k]
    OA, OA_inv, J, N = gluing_codomain(A24_12, X_8, Z_8)

    # Package the point we use for translation
    T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    T8 = (T8_X, T8_Y, T8_Z)

    # Push kernel though the gluing isogeny
    for i in range(k):
        P = strat_points[i]
        imPQ = gluing_eval_pair(A_12, P, T8, J, N)
        strat_points[i] = imPQ
        orders[i] = orders[i] - 1

    # Push other points through gluing isogeny
    OA_inv_inv = avx_421_to_811(OA_inv, OA_inv)
    eval_pairs = [
        gluing_eval_special_pair(P1, P2, Q1, Q2, OA_inv_inv, N)
        for ((P1, P2), (Q1, Q2)) in eval_pairs
    ]
    eval_single = [gluing_eval_special(P1, P2, OA_inv, N) for (P1, P2) in eval_single]

    # Step Two:
    # Compute all but the last two steps of the isogeny without square roots
    k = k - 1
    j = 0
    while k >= 0 and orders[k] != 0:
        j = j + 1
        C1, C2 = theta_precompute(OA, use_wide=True)

        # Perform the doubling
        while orders[k] != 1:
            k = k + 1
            n = orders[k - 1] // 2
            ker = strat_points[k - 1]
            for _ in range(n):
                ker = theta_double_pair(ker, C1, C2)
            strat_points[k] = ker
            orders[k] = orders[k - 1] - n

        # As we avoid the last two steps of the chain in this loop, h1 and h2
        # are fixed values
        h1, h2 = False, True

        # Compute the codomain from the kernel
        ker = strat_points[k]
        OA, inv = theta_codomain_8x1x1(ker, h1, h2)
        inv_inv = avx_421_to_811(inv, inv)

        # Push kernel though the gluing isogeny
        for i in range(k):
            P = strat_points[i]
            strat_points[i] = theta_eval_pair(P, inv_inv, h1, h2)
            orders[i] = orders[i] - 1
        k = k - 1

        # Push the images through the kernel
        eval_pairs = [theta_eval_pair(P, inv_inv, h1, h2) for P in eval_pairs]
        eval_single = [theta_eval(P, inv, h1, h2) for P in eval_single]

    # Step Three:
    # we now do the last two steps outside of the strategy. As we do not
    # have the torsion above we require square roots to compute the last
    # two codomains.
    if length > 3:
        P, _ = avx_811_to_421(strat_points[0])
        ker = theta_eval(P, inv, h1, h2)
    else:
        P, _ = avx_811_to_421(strat_points[0])

    h1, h2 = False, False
    OA, inv = theta_codomain_with_four_torsion(ker, OA, h1, h2)
    inv_inv = avx_421_to_811(inv, inv)
    eval_pairs = [theta_eval_pair(P, inv_inv, h1, h2) for P in eval_pairs]
    eval_single = [theta_eval(P, inv, h1, h2) for P in eval_single]

    h1, h2 = True, False
    OA, inv = theta_codomain_with_no_torsion(OA, h1, h2)
    inv_inv = avx_421_to_811(inv, inv)
    eval_pairs = [theta_eval_pair(P, inv_inv, h1, h2) for P in eval_pairs]
    eval_single = [theta_eval(P, inv, h1, h2) for P in eval_single]

    # Step Four:
    # Finish the chain with an isomorphism to a theta point with a product
    # structure for efficient splitting

    # Compute the splitting isomorphism
    M = splitting_isomorphism_matrix(OA)

    # Apply the splitting isomorphism
    OA = apply_isomorphism(OA, M)
    eval_pairs = [apply_isomorphism_pair(P, M) for P in eval_pairs]
    eval_single = [apply_isomorphism(P, M) for P in eval_single]

    if randomise:
        OA, N = randomized_null_point(OA)
        eval_pairs = [apply_isomorphism_pair(P, N) for P in eval_pairs]
        eval_single = [apply_isomorphism(P, N) for P in eval_single]

    # Recreate a list of 4x2x1 theta points to split.
    eval_points = [avx_811_to_421(P) for P in eval_pairs]
    eval_points = [item for tup in eval_points for item in tup]
    if eval_single:
        eval_points.append(eval_single[0])

    # Split the null point to the Montgomery coefficients
    (A1, C1), (A2, C2) = theta_to_product(OA)
    images = [theta_product_point_to_montgomery(P, OA) for P in eval_points]

    return (A1, C1), (A2, C2), images


# ================================================================================
# Isogeny chains where we have the 2^(n + 2) torsion
# ================================================================================


def theta_isogeny_chain_with_torsion_4x2x1(
    domain: tuple[Curve],
    kernel: tuple[BasisX, BasisX],
    length: int,
    eval_points: list[tuple[PointX, PointX]] = [],
    randomise: bool = False,
):
    # We normalise points and the curve parameters the below functions stay with
    # 64 bit limbs to make use of the faster(?) sqrt we have available
    E1, E2, B1, B2 = normalise_chain_input(domain, kernel)
    P1, Q1 = lift_normalised_basis_to_jacobian(E1, B1)
    P2, Q2 = lift_normalised_basis_to_jacobian(E2, B2)

    # We are now in the position to convert from 64 to 51 bit limbs
    # For the curve we need to pack the Montgomery coefficient for
    # arithmetic.
    A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)

    # For the Jacobian points, we pack the coordinates into three blobs
    ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X)
    ker_Y = AVX_4x2x1(P1.Y, Q1.Y, P2.Y, Q2.Y)
    ker_Z = AVX_4x2x1(P1.Z, Q1.Z, P2.Z, Q2.Z)

    # Initalise the strategy arrays
    space = length.bit_length() + 1
    strat_points = [(None, None, None)] * space
    orders = [0] * space

    # Set the first values
    strat_points[0] = [ker_X, ker_Y, ker_Z]
    orders[0] = length
    k = 0

    # Step One:
    # Perform repeated doubling of the Jacobian points and store intermediate results
    # to push through the isogeny.
    strat_points, orders, k = compute_product_doubles(A_12, strat_points, orders, k)

    # Compute the gluing codomain
    X_8, Y_8, Z_8 = strat_points[k]
    OA, OA_inv, J, N = gluing_codomain(A24_12, X_8, Z_8)

    # Package the point we use for translation
    T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    T8 = (T8_X, T8_Y, T8_Z)

    # Push kernel though the gluing isogeny
    for i in range(k):
        P = strat_points[i]
        imPQ = gluing_eval_pair(A_12, P, T8, J, N)
        imP, imQ = avx_811_to_421(imPQ)
        strat_points[i] = (imP, imQ)
        orders[i] = orders[i] - 1

    # Push other points through gluing isogeny
    eval_points = [gluing_eval_special(P1, P2, OA_inv, N) for (P1, P2) in eval_points]

    # Step Two: Main loop
    # Compute length - 1 (2, 2)-isogenies in the theta model
    k = k - 1
    j = 0
    while k >= 0 and orders[k] != 0:
        j = j + 1
        C1, C2 = theta_precompute(OA)

        # Perform the doubling
        while orders[k] != 1:
            k = k + 1
            n = orders[k - 1] // 2
            ker_P, ker_Q = strat_points[k - 1]
            for _ in range(n):
                ker_P = theta_double(ker_P, C1, C2)
                ker_Q = theta_double(ker_Q, C1, C2)
            strat_points[k] = (ker_P, ker_Q)
            orders[k] = orders[k - 1] - n

        # For the ultimate and penultimate steps we do extra / less Hadamards
        # to avoid zeros from the product structure
        if j == length - 1:
            h1, h2 = True, False
        elif j == length - 2:
            h1, h2 = False, False
        else:
            h1, h2 = False, True

        # Compute the codomain from the kernel
        ker_P, ker_Q = strat_points[k]
        OA, inv = theta_codomain_4x2x1(ker_P, ker_Q, h1, h2)

        # Push kernel though the gluing isogeny
        for i in range(k):
            P, Q = strat_points[i]
            P = theta_eval(P, inv, h1, h2)
            Q = theta_eval(Q, inv, h1, h2)
            strat_points[i] = (P, Q)
            orders[i] = orders[i] - 1
        k = k - 1

        # Push the images through the kernel
        eval_points = [theta_eval(P, inv, h1, h2) for P in eval_points]

    # Step Three:
    # Finish the chain with an isomorphism to a theta point with a product
    # structure for efficient splitting

    # Compute the splitting isomorphism
    M = splitting_isomorphism_matrix(OA)

    # Apply the splitting isomorphism
    OA = apply_isomorphism(OA, M)
    eval_points = [apply_isomorphism(P, M) for P in eval_points]

    if randomise:
        OA, N = randomized_null_point(OA)
        eval_points = [apply_isomorphism(P, N) for P in eval_points]

    # Split the null point to the Montgomery coefficients
    (A1, C1), (A2, C2) = theta_to_product(OA)
    images = [theta_product_point_to_montgomery(P, OA) for P in eval_points]

    return (A1, C1), (A2, C2), images


def theta_isogeny_chain_with_torsion_8x1x1(
    domain: tuple[Curve],
    kernel: tuple[BasisX, BasisX],
    length: int,
    eval_points: list[tuple[PointX, PointX]] = [],
    randomise: bool = False,
    verify: bool = False,
):
    # We normalise points and the curve parameters the below functions stay with
    # 64 bit limbs to make use of the faster(?) sqrt we have available
    E1, E2, B1, B2 = normalise_chain_input(domain, kernel)
    P1, Q1 = lift_normalised_basis_to_jacobian(E1, B1)
    P2, Q2 = lift_normalised_basis_to_jacobian(E2, B2)

    # We are now in the position to convert from 64 to 51 bit limbs
    # For the curve we need to pack the Montgomery coefficient for
    # arithmetic.
    A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)

    # For the Jacobian points, we pack the coordinates into three blobs
    ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X)
    ker_Y = AVX_4x2x1(P1.Y, Q1.Y, P2.Y, Q2.Y)
    ker_Z = AVX_4x2x1(P1.Z, Q1.Z, P2.Z, Q2.Z)

    # Initalise the strategy arrays
    space = length.bit_length() + 1
    strat_points = [(None, None, None)] * space
    orders = [0] * space

    # Set the first values
    strat_points[0] = [ker_X, ker_Y, ker_Z]
    orders[0] = length
    k = 0

    # If we can evaluate images in pairs we save time, so given the eval_points as input
    # we want to take pairs of inputs into eval_points_pairs and then if len(eval_points)
    # there'll be a single point in eval_point_single
    nk = len(eval_points)
    eval_pairs = [(eval_points[i], eval_points[i + 1]) for i in range(0, nk - 1, 2)]
    eval_single = [] if nk % 2 == 0 else [eval_points[-1]]

    # Step One:
    # Perform repeated doubling of the Jacobian points and store intermediate results
    # to push through the isogeny.
    strat_points, orders, k = compute_product_doubles(A_12, strat_points, orders, k)

    # Compute the gluing codomain
    X_8, Y_8, Z_8 = strat_points[k]
    OA, OA_inv, J, N = gluing_codomain(A24_12, X_8, Z_8)

    # Package the point we use for translation
    T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    T8 = (T8_X, T8_Y, T8_Z)

    # Push kernel though the gluing isogeny
    for i in range(k):
        P = strat_points[i]
        imP_Q = gluing_eval_pair(A_12, P, T8, J, N)
        strat_points[i] = imP_Q
        orders[i] = orders[i] - 1

    # Push other points through gluing isogeny
    OA_inv_inv = avx_421_to_811(OA_inv, OA_inv)
    eval_pairs = [
        gluing_eval_special_pair(P1, P2, Q1, Q2, OA_inv_inv, N)
        for ((P1, P2), (Q1, Q2)) in eval_pairs
    ]
    eval_single = [gluing_eval_special(P1, P2, OA_inv, N) for (P1, P2) in eval_single]

    # Step Two: Main loop
    # Compute length - 1 (2, 2)-isogenies in the theta model
    k = k - 1
    j = 0
    while k >= 0 and orders[k] != 0:
        j = j + 1
        C1, C2 = theta_precompute(OA, use_wide=True)

        # Perform the doubling
        while orders[k] != 1:
            k = k + 1
            n = orders[k - 1] // 2
            ker = strat_points[k - 1]
            for _ in range(n):
                ker = theta_double_pair(ker, C1, C2)
            strat_points[k] = ker
            orders[k] = orders[k - 1] - n

        # We cannot perform verification of codomain on the last step
        # instead we find errors from the splitting isomorphism
        if j == length - 1:
            codomain_verify = False
        else:
            codomain_verify = verify

        # For the ultimate and penultimate steps we do extra / less Hadamards
        # to avoid zeros from the product structure
        if j == length - 1:
            h1, h2 = True, False
        elif j == length - 2:
            h1, h2 = False, False
        else:
            h1, h2 = False, True

        # Compute the codomain from the kernel
        ker = strat_points[k]
        OA, inv = theta_codomain_8x1x1(ker, h1, h2, codomain_verify)
        inv_inv = avx_421_to_811(inv, inv)

        # Push kernel though the gluing isogeny
        for i in range(k):
            P = strat_points[i]
            strat_points[i] = theta_eval_pair(P, inv_inv, h1, h2)
            orders[i] = orders[i] - 1
        k = k - 1

        # Push the images through the kernel
        eval_pairs = [theta_eval_pair(P, inv_inv, h1, h2) for P in eval_pairs]
        eval_single = [theta_eval(P, inv, h1, h2) for P in eval_single]

    # Step Three:
    # Finish the chain with an isomorphism to a theta point with a product
    # structure for efficient splitting

    # Compute the splitting isomorphism
    M = splitting_isomorphism_matrix(OA)

    # Apply the splitting isomorphism
    OA = apply_isomorphism(OA, M)
    eval_pairs = [apply_isomorphism_pair(P, M) for P in eval_pairs]
    eval_single = [apply_isomorphism(P, M) for P in eval_single]

    if randomise:
        OA, N = randomized_null_point(OA)
        eval_pairs = [apply_isomorphism_pair(P, N) for P in eval_pairs]
        eval_single = [apply_isomorphism(P, N) for P in eval_single]

    # Split the null point to the Montgomery coefficients
    (A1, C1), (A2, C2) = theta_to_product(OA, verify)

    # Recreate a list of 4x2x1 theta points to split.
    eval_points = [avx_811_to_421(P) for P in eval_pairs]
    eval_points = [item for tup in eval_points for item in tup]
    if eval_single:
        eval_points.append(eval_single[0])
    images = [theta_product_point_to_montgomery(P, OA, verify) for P in eval_points]

    return (A1, C1), (A2, C2), images


# ================================================================================
# Isogeny chains where we have the 2^(n + 2) torsion and we verify the chain
# ================================================================================


def theta_isogeny_chain_verification_8x1x1(
    domain: tuple[Curve],
    kernel: tuple[BasisX, BasisX],
    length: int,
):
    """
    Specialised (2,2) chain which performs additional checks required
    by SQIsign and returns only the codomain and uses a variable time
    superglue for the gluing isogeny
    """

    # We normalise points and the curve parameters the below functions stay with
    # 64 bit limbs to make use of the faster(?) sqrt we have available
    E1, E2, B1, B2 = normalise_chain_input(domain, kernel)
    P1, Q1 = lift_normalised_basis_to_jacobian(E1, B1)
    P2, Q2 = lift_normalised_basis_to_jacobian(E2, B2)

    # We are now in the position to convert from 64 to 51 bit limbs
    # For the curve we need to pack the Montgomery coefficient for
    # arithmetic.
    A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)

    # For the Jacobian points, we pack the coordinates into three blobs
    ker_X = AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X)
    ker_Y = AVX_4x2x1(P1.Y, Q1.Y, P2.Y, Q2.Y)
    ker_Z = AVX_4x2x1(P1.Z, Q1.Z, P2.Z, Q2.Z)

    # Initalise the strategy arrays
    space = length.bit_length() + 1
    strat_points = [(None, None, None)] * space
    orders = [0] * space

    # Set the first values
    strat_points[0] = [ker_X, ker_Y, ker_Z]
    orders[0] = length
    k = 0

    # Step One:
    # Perform repeated doubling of the Jacobian points and store intermediate results
    # to push through the isogeny.
    strat_points, orders, k = compute_product_doubles(A_12, strat_points, orders, k)

    # Compute the gluing codomain
    X_8, Y_8, Z_8 = strat_points[k]

    OA, ab, precomputation = superglue_codomain(A24_12, X_8, Z_8)

    # Package the point we use for translation
    T8_X = AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2])
    T8_Y = AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2])
    T8_Z = AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2])
    T8 = (T8_X, T8_Y, T8_Z)

    # Push kernel though the gluing isogeny
    for i in range(k):
        P = strat_points[i]
        imP_Q = superglue_eval_pair(A_12, P, T8, ab, precomputation)
        strat_points[i] = imP_Q
        orders[i] = orders[i] - 1

    # Step Two: Main loop
    # Compute length - 1 (2, 2)-isogenies in the theta model
    k = k - 1
    j = 0
    while k >= 0 and orders[k] != 0:
        j = j + 1
        C1, C2 = theta_precompute(OA, use_wide=True)

        # Perform the doubling
        while orders[k] != 1:
            k = k + 1
            n = orders[k - 1] // 2
            ker = strat_points[k - 1]
            for _ in range(n):
                ker = theta_double_pair(ker, C1, C2)
            strat_points[k] = ker
            orders[k] = orders[k - 1] - n

        # We cannot perform verification of codomain on the last step
        # instead we find errors from the splitting isomorphism
        if j == length - 1:
            codomain_verify = False
        else:
            codomain_verify = True

        # For the ultimate and penultimate steps we do extra / less Hadamards
        # to avoid zeros from the product structure
        if j == length - 1:
            h1, h2 = True, False
        elif j == length - 2:
            h1, h2 = False, False
        else:
            h1, h2 = False, True

        # Compute the codomain from the kernel
        ker = strat_points[k]
        OA, inv = theta_codomain_8x1x1(ker, h1, h2, codomain_verify)
        inv_inv = avx_421_to_811(inv, inv)

        # Push kernel though the gluing isogeny
        for i in range(k):
            P = strat_points[i]
            strat_points[i] = theta_eval_pair(P, inv_inv, h1, h2)
            orders[i] = orders[i] - 1
        k = k - 1

    # Step Three:
    # Finish the chain with an isomorphism to a theta point with a product
    # structure for efficient splitting

    # Compute the splitting isomorphism
    M = splitting_isomorphism_matrix(OA)

    # Apply the splitting isomorphism
    OA = apply_isomorphism(OA, M)

    # Split the null point to the Montgomery coefficients
    (A1, C1), (A2, C2) = theta_to_product(OA, True)

    return (A1, C1), (A2, C2)
