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
    n: int,
    eval_points: list[tuple[PointJ, PointJ]] = [],
):
    """
    Naive (2^n, 2^n) isogeny chain with no strategy using everything as 4x2x1 blocks
    """
    E1, E2 = domain
    B1, B2 = kernel

    # Lift the x-only basis to the Jacobian points
    P1, Q1 = lift_basis_to_jacobian(E1, B1)
    P2, Q2 = lift_basis_to_jacobian(E2, B2)

    # Obtain kernel points for gluing
    P1_8, P2_8, Q1_8, Q2_8 = four_double_n_jac(E1, E2, P1, P2, Q1, Q2, n - 1)

    # Compute the gluing codomain
    OA, _, J, N = gluing_codomain(E1, E2, P1_8, P2_8, Q1_8, Q2_8)

    # Push kernel though the gluing isogeny
    P = gluing_eval(E1, E2, P1, P2, P1_8, P2_8, J, N)
    Q = gluing_eval(E1, E2, Q1, Q2, P1_8, P2_8, J, N)

    # Push other points through gluing isogeny
    eval_points = [
        gluing_eval(E1, E2, R1, R2, P1_8, P2_8, J, N) for (R1, R2) in eval_points
    ]

    # Do n - 1 steps of the theta (2, 2)-isogeny
    for i in range(1, n):
        # Precompute constants for doubling
        C1, C2 = theta_precompute(OA)

        # Double the kernel points of order 2^(n+2) to get points of order 2^(1 + 2) = 8
        ker_P, ker_Q = P, Q
        for _ in range(n - i - 1):
            ker_P = theta_double(ker_P, C1, C2)
            ker_Q = theta_double(ker_Q, C1, C2)

        # For the ultimate and penultimate steps we do extra / less Hadamards
        # to avoid zeros from the product structure
        if i == n - 1:
            h1, h2 = True, False
        elif i == n - 2:
            h1, h2 = False, False
        else:
            h1, h2 = False, True

        # Compute the codomain from the kernel
        OA, inv = theta_codomain_4x2x1(ker_P, ker_Q, h1, h2)

        # Push the kernel through the images
        P = theta_eval(P, inv, h1, h2)
        Q = theta_eval(Q, inv, h1, h2)

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
    n: int,
    eval_points: list[tuple[PointJ, PointJ]] = [],
):
    """
    Naive (2^n, 2^n) isogeny chain with no strategy using 8x1x1 blocks when possible
    """
    E1, E2 = domain
    B1, B2 = kernel

    # Lift the x-only basis to the Jacobian points
    P1, Q1 = lift_basis_to_jacobian(E1, B1)
    P2, Q2 = lift_basis_to_jacobian(E2, B2)

    # Obtain kernel points for gluing
    P1_8, P2_8, Q1_8, Q2_8 = four_double_n_jac(E1, E2, P1, P2, Q1, Q2, n - 1)

    # Compute the gluing codomain
    OA, _, J, N = gluing_codomain(E1, E2, P1_8, P2_8, Q1_8, Q2_8)

    # Push kernel though the gluing isogeny
    P = gluing_eval(E1, E2, P1, P2, P1_8, P2_8, J, N)
    Q = gluing_eval(E1, E2, Q1, Q2, P1_8, P2_8, J, N)

    # TODO: look at using 8x1x1 within gluing_eval
    kernel = AVX_8x1x1(*P, *Q)

    # Push other points through gluing isogeny
    eval_points = [
        gluing_eval(E1, E2, R1, R2, P1_8, P2_8, J, N) for (R1, R2) in eval_points
    ]

    for i in range(1, n):
        # Precompute constants for doubling
        C1, C2 = theta_precompute(OA)

        # Double the kernel points of order 2^(n+2) to get points of order 2^(1 + 2) = 8
        ker_step = kernel
        for _ in range(n - i - 1):
            ker_step = theta_double_pair(ker_step, C1, C2)

        # For the ultimate and penultimate steps we do extra / less Hadamards
        # to avoid zeros from the product structure
        if i == n - 1:
            h1, h2 = True, False
        elif i == n - 2:
            h1, h2 = False, False
        else:
            h1, h2 = False, True

        # Compute the codomain from the kernel
        OA, inv = theta_codomain_8x1x1(ker_step, h1, h2)

        # Push the kernel through the images
        kernel = theta_eval_pair(kernel, inv, h1, h2)

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
