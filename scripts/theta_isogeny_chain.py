from avx_types import AVX_8x1x1
from elliptic_types import Curve, PointJ, BasisX

from elliptic_arithmetic import four_double_n_jac
from theta_gluing import lift_basis_to_jacobian, gluing_codomain, gluing_eval
from theta_isogeny import (
    theta_precompute,
    theta_double,
    theta_double_pair,
    theta_codomain_4x2x1,
    theta_codomain_8x1x1,
    theta_eval,
    theta_eval_pair,
)

from theta_splitting import (
    splitting_isomorphism_matrix,
    theta_to_product,
    theta_product_point_to_montgomery,
)
from utilities import apply_isomorphism


def theta_isogeny_chain_4x2x1(
    domain: tuple[Curve],
    kernel: tuple[BasisX, BasisX],
    length: int,
    eval_points: list[tuple[PointJ, PointJ]] = [],
):
    # Extract out the domain and kernel data
    E1, E2 = domain
    B1, B2 = kernel

    # Lift the x-only basis to the Jacobian points
    P1, Q1 = lift_basis_to_jacobian(E1, B1)
    P2, Q2 = lift_basis_to_jacobian(E2, B2)

    # Initalise the strategy arrays
    space = length.bit_length() + 1
    strat_points = [
        (PointJ(0, 0, 0), PointJ(0, 0, 0), PointJ(0, 0, 0), PointJ(0, 0, 0))
    ] * space
    orders = [0] * space

    # Set the first values
    strat_points[0] = [P1, P2, Q1, Q2]
    orders[0] = length
    k = 0

    # Step One:
    # Perform repeated doubling of the Jacobian points and store intermediate results
    # to push through the isogeny. TODO: at the moment these points are PointJ, which
    # are arrays with 64-bit limb representations, but this doesn't need to be the case,
    # what I need to add is a AVX_PointJ type which we use here.
    while orders[k] != 1:
        k = k + 1
        # TODO: need to balance this in the C code to find the optional
        # splitting of the chain, as eval is more expensive than doubling
        if orders[k - 1] >= 16:
            n = orders[k - 1] // 2
        else:
            n = orders[k - 1] - 1

        iP1, iP2, iQ1, iQ2 = strat_points[k - 1]
        strat_points[k] = four_double_n_jac(E1, E2, iP1, iP2, iQ1, iQ2, n)
        orders[k] = orders[k - 1] - n

    # Compute the gluing codomain
    P1_8, P2_8, Q1_8, Q2_8 = strat_points[k]
    OA, _, J, N = gluing_codomain(E1, E2, P1_8, P2_8, Q1_8, Q2_8)

    # Push kernel though the gluing isogeny
    for i in range(k):
        P1, P2, Q1, Q2 = strat_points[i]
        # TODO: can we do something even faster here with a gluing_eval_pair
        #       using 8x1x1 blocks?
        imP = gluing_eval(E1, E2, P1, P2, P1_8, P2_8, J, N)
        imQ = gluing_eval(E1, E2, Q1, Q2, P1_8, P2_8, J, N)
        strat_points[i] = (imP, imQ)
        orders[i] = orders[i] - 1

    # Push other points through gluing isogeny
    eval_points = [
        gluing_eval(E1, E2, R1, R2, P1_8, P2_8, J, N) for (R1, R2) in eval_points
    ]

    # Main loop
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
        if i == length - 1:
            h1, h2 = True, False
        elif i == length - 2:
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

    # Compute the splitting isomorphism
    M = splitting_isomorphism_matrix(OA)

    # Apply the splitting isomorphism
    OA = apply_isomorphism(OA, M)
    eval_points = [apply_isomorphism(P, M) for P in eval_points]

    # Split the null point to the Montgomery coefficients
    (A1, C1), (A2, C2) = theta_to_product(OA)

    images = [theta_product_point_to_montgomery(P, OA) for P in eval_points]

    return (A1, C1), (A2, C2), images


def theta_isogeny_chain_8x1x1(
    domain: tuple[Curve],
    kernel: tuple[BasisX, BasisX],
    length: int,
    eval_points: list[tuple[PointJ, PointJ]] = [],
):
    # Extract out the domain and kernel data
    E1, E2 = domain
    B1, B2 = kernel

    # Lift the x-only basis to the Jacobian points
    P1, Q1 = lift_basis_to_jacobian(E1, B1)
    P2, Q2 = lift_basis_to_jacobian(E2, B2)

    # Initalise the strategy arrays
    space = length.bit_length() + 1
    strat_points = [
        (PointJ(0, 0, 0), PointJ(0, 0, 0), PointJ(0, 0, 0), PointJ(0, 0, 0))
    ] * space
    orders = [0] * space

    # Set the first values
    strat_points[0] = [P1, P2, Q1, Q2]
    orders[0] = length
    k = 0

    # Step One:
    # Perform repeated doubling of the Jacobian points and store intermediate results
    # to push through the isogeny. TODO: at the moment these points are PointJ, which
    # are arrays with 64-bit limb representations, but this doesn't need to be the case,
    # what I need to add is a AVX_PointJ type which we use here.
    while orders[k] != 1:
        k = k + 1
        # TODO: need to balance this in the C code to find the optional
        # splitting of the chain, as eval is more expensive than doubling
        if orders[k - 1] >= 16:
            n = orders[k - 1] // 2
        else:
            n = orders[k - 1] - 1

        iP1, iP2, iQ1, iQ2 = strat_points[k - 1]
        strat_points[k] = four_double_n_jac(E1, E2, iP1, iP2, iQ1, iQ2, n)
        orders[k] = orders[k - 1] - n

    # Compute the gluing codomain
    P1_8, P2_8, Q1_8, Q2_8 = strat_points[k]
    OA, _, J, N = gluing_codomain(E1, E2, P1_8, P2_8, Q1_8, Q2_8)

    # Push kernel though the gluing isogeny
    for i in range(k):
        P1, P2, Q1, Q2 = strat_points[i]
        # TODO: can we do something even faster here with a gluing_eval_pair
        #       using 8x1x1 blocks?
        imP = gluing_eval(E1, E2, P1, P2, P1_8, P2_8, J, N)
        imQ = gluing_eval(E1, E2, Q1, Q2, P1_8, P2_8, J, N)
        imP_Q = AVX_8x1x1(*imP, *imQ)
        strat_points[i] = imP_Q
        orders[i] = orders[i] - 1

    # Push other points through gluing isogeny
    eval_points = [
        gluing_eval(E1, E2, R1, R2, P1_8, P2_8, J, N) for (R1, R2) in eval_points
    ]

    # Main loop
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
            ker = strat_points[k - 1]
            for _ in range(n):
                ker = theta_double_pair(ker, C1, C2)
            strat_points[k] = ker
            orders[k] = orders[k - 1] - n

        # For the ultimate and penultimate steps we do extra / less Hadamards
        # to avoid zeros from the product structure
        if i == length - 1:
            h1, h2 = True, False
        elif i == length - 2:
            h1, h2 = False, False
        else:
            h1, h2 = False, True

        # Compute the codomain from the kernel
        ker = strat_points[k]
        OA, inv = theta_codomain_8x1x1(ker, h1, h2)

        # Push kernel though the gluing isogeny
        for i in range(k):
            P = strat_points[i]
            strat_points[i] = theta_eval_pair(P, inv, h1, h2)
            orders[i] = orders[i] - 1
        k = k - 1

        # Push the images through the kernel
        eval_points = [theta_eval(P, inv, h1, h2) for P in eval_points]

    # Compute the splitting isomorphism
    M = splitting_isomorphism_matrix(OA)

    # Apply the splitting isomorphism
    OA = apply_isomorphism(OA, M)
    eval_points = [apply_isomorphism(P, M) for P in eval_points]

    # Split the null point to the Montgomery coefficients
    (A1, C1), (A2, C2) = theta_to_product(OA)

    images = [theta_product_point_to_montgomery(P, OA) for P in eval_points]

    return (A1, C1), (A2, C2), images
