from typing import Tuple

from theta_types import AVX_8x1x1, AVX_4x2x1
from elliptic_types import Curve, PointJ, BasisX

from elliptic_arithmetic import double_jac_n
from theta_gluing import lift_basis_to_jacobian, gluing_codomain, gluing_eval
from theta_structure import (
    hadamard,
    hadamard_pair,
    squaring,
    squaring_pair,
    pairwise_multiplication,
    pairwise_multiplication_3,
    pairwise_multiplication_2,
    pairwise_pair,
)
from theta_splitting import (
    splitting_isomorphism_matrix,
    theta_to_product,
    theta_product_point_to_montgomery,
)
from utilities import apply_isomorphism


def theta_precompute(OA: AVX_4x2x1) -> Tuple[AVX_4x2x1, AVX_4x2x1]:
    """
    Precompute constants given a theta null point

    Cost: uses 3 4x2x1 multiplications instead of 12 Fp2 multiplications
          but requires additional memory management to form different AVX512
          blobs along the way
    """
    # Compute H . S(OA)
    OA_prime = squaring(OA)
    OA_prime = hadamard(OA_prime)

    # Memory management for 4x2x1 computations
    a, b, c, d = OA
    AA, BB, CC, DD = OA_prime
    t1 = AVX_4x2x1(AA, CC, a, c)
    t2 = AVX_4x2x1(BB, DD, b, d)
    C1 = AVX_4x2x1(BB, AA, DD, CC)
    C2 = AVX_4x2x1(b, a, d, c)

    # Precompute four multiplications AABB, CCDD, ab and cd
    t2 = pairwise_multiplication(t1, t2)

    # More memory management to repackage the first multiplication
    AABB, CCDD, ab, cd = t2
    t1 = AVX_4x2x1(CCDD, CCDD, AABB, AABB)
    t2 = AVX_4x2x1(cd, cd, ab, ab)

    # Compute the arithmetic constants more memory management and two more muls
    C1 = pairwise_multiplication(C1, t1)
    C2 = pairwise_multiplication(C2, t2)

    return C1, C2


def theta_double(P: AVX_4x2x1, C1: AVX_4x2x1, C2: AVX_4x2x1) -> AVX_4x2x1:
    """
    Double a point provided certain precomputed constants
    """
    P2 = squaring(P)
    P2 = hadamard(P2)
    P2 = squaring(P2)
    P2 = pairwise_multiplication(P2, C1)
    P2 = hadamard(P2)
    P2 = pairwise_multiplication(P2, C2)
    return P2


def theta_double_pair(P_Q: AVX_8x1x1, C1: AVX_4x2x1, C2: AVX_4x2x1) -> AVX_8x1x1:
    """
    Double a point provided certain precomputed constants
    """
    P2_Q2 = squaring_pair(P_Q)
    P2_Q2 = hadamard_pair(P2_Q2)
    P2_Q2 = squaring_pair(P2_Q2)
    P2_Q2 = pairwise_pair(P2_Q2, C1)
    P2_Q2 = hadamard_pair(P2_Q2)
    P2_Q2 = pairwise_pair(P2_Q2, C2)
    return P2_Q2


def theta_codomain_4x2x1(
    P: AVX_4x2x1, Q: AVX_4x2x1, h1: bool, h2: bool
) -> Tuple[AVX_4x2x1, AVX_4x2x1]:
    if h1:
        P = hadamard(P)
        Q = hadamard(Q)

    P = squaring(P)
    P = hadamard(P)

    Q = squaring(Q)
    Q = hadamard(Q)

    # =================================================================
    # TODO: what's the most AVX512 friendly way of this middle part...
    # xA, xB, yC, yD = ker_P
    # zA, wB, zC, wD = ker_Q
    # xAwB = xA * wB
    # zAxB = zA * xB
    # A = zA * xAwB
    # B = wB * zAxB
    # C = zC * xAwB
    # D = wD * zAxB

    # Compute projectively 1/A, 1/B, 1/C, 1/D by computing ABCDxzw / A
    # ABCDxzw / B, etc.
    # zCwD = zC * wD
    # A_inv = xB * zCwD
    # B_inv = xA * zCwD
    # C_inv = D
    # D_inv = C

    # OB = (A, B, C, D)
    # inv = (A_inv, B_inv, C_inv, D_inv)
    # =================================================================

    # Memory management of kernel data to make two temp blobs
    xA, xB, yC, yD = P
    zA, wB, zC, wD = Q
    t1 = AVX_4x2x1(xA, zA, zC, 1)
    t2 = AVX_4x2x1(wB, xB, wD, 1)

    # Here we only need three multiplications, we could make a special function
    t2 = pairwise_multiplication_3(t1, t2)

    # Memory management to get the precomputations we need
    xAwB, zAxB, zCwD, _ = t2
    t1 = AVX_4x2x1(xAwB, zAxB, xAwB, zAxB)

    # Compute the codomain
    OB = pairwise_multiplication(t1, Q)

    # Memory management to get the values needed for inverse
    _, _, C, D = OB
    inv = AVX_4x2x1(xB, xA, D, C)
    t1 = AVX_4x2x1(zCwD, zCwD, 1, 1)

    # Here we only need two multiplications, we could make a special function
    inv = pairwise_multiplication_2(inv, t1)

    if h2:
        OB = hadamard(OB)

    return OB, inv


def theta_codomain_8x1x1(
    ker: AVX_8x1x1, h1: bool, h2: bool
) -> Tuple[AVX_4x2x1, AVX_4x2x1]:
    if h1:
        ker = hadamard_pair(ker)
    ker = squaring_pair(ker)
    ker = hadamard_pair(ker)

    # Memory management of kernel data to make two temp blobs
    xA, xB, _, _, zA, wB, zC, wD = ker
    OB = AVX_4x2x1(zA, wB, zC, wD)
    t1 = AVX_4x2x1(xA, zA, zC, 1)
    t2 = AVX_4x2x1(wB, xB, wD, 1)

    # Here we only need three multiplications, we could make a special function
    t2 = pairwise_multiplication_3(t1, t2)

    # Memory management to get the precomputations we need
    xAwB, zAxB, zCwD, _ = t2
    t1 = AVX_4x2x1(xAwB, zAxB, xAwB, zAxB)

    # Compute the codomain
    OB = pairwise_multiplication(t1, OB)

    # Memory management to get the values needed for inverse
    A, B, C, D = OB
    inv = AVX_4x2x1(xB, xA, D, C)
    t1 = AVX_4x2x1(zCwD, zCwD, 1, 1)

    # Here we only need two multiplications, we could make a special function
    inv = pairwise_multiplication_2(inv, t1)

    if h2:
        OB = hadamard(OB)

    return OB, inv


def theta_eval(P: AVX_4x2x1, inv: AVX_4x2x1, h1: bool, h2: bool) -> AVX_4x2x1:
    """
    Evaluate a theta point through an isogeny given precomputed constants
    """
    if h1:
        P = hadamard(P)

    P = squaring(P)
    P = hadamard(P)
    P = pairwise_multiplication(P, inv)

    if h2:
        P = hadamard(P)

    return P


def theta_eval_pair(P_Q: AVX_8x1x1, inv: AVX_4x2x1, h1: bool, h2: bool):
    """
    Evaluate a theta point through an isogeny given precomputed constants
    """
    if h1:
        P_Q = hadamard_pair(P_Q)

    P_Q = squaring_pair(P_Q)
    P_Q = hadamard_pair(P_Q)
    P_Q = pairwise_pair(P_Q, inv)

    if h2:
        P_Q = hadamard_pair(P_Q)

    return P_Q


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
    P1_8 = double_jac_n(E1, P1, n - 1)
    P2_8 = double_jac_n(E2, P2, n - 1)
    Q1_8 = double_jac_n(E1, Q1, n - 1)
    Q2_8 = double_jac_n(E2, Q2, n - 1)

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
    P1_8 = double_jac_n(E1, P1, n - 1)
    P2_8 = double_jac_n(E2, P2, n - 1)
    Q1_8 = double_jac_n(E1, Q1, n - 1)
    Q2_8 = double_jac_n(E2, Q2, n - 1)

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
