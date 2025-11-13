from avx_types import AVX_4x2x1, AVX_8x1x1
from theta_structure import (
    squaring,
    hadamard,
    pairwise_multiplication,
    pairwise_multiplication_3,
    pairwise_multiplication_2,
    squaring_pair,
    hadamard_pair,
    pairwise_pair,
)
from utilities import sqrt_fp2


def theta_precompute(OA: AVX_4x2x1) -> tuple[AVX_4x2x1, AVX_4x2x1]:
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
) -> tuple[AVX_4x2x1, AVX_4x2x1]:
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


def theta_codomain_with_four_torsion(
    ker_P: AVX_4x2x1,
    OA: AVX_4x2x1,
    h1: bool,
    h2: bool,
) -> tuple[AVX_4x2x1, AVX_4x2x1]:
    if h1:
        T1 = hadamard(ker_P)
        T1 = squaring(T1)
        T1 = hadamard(T1)

        T2 = hadamard(OA)
        T2 = squaring(T2)
        T2 = hadamard(T2)
    else:
        T1 = squaring(ker_P)
        T1 = hadamard(T1)

        T2 = squaring(OA)
        T2 = hadamard(T2)

    xAB, _, xCD, _ = T1
    AA, BB, CC, DD = T2

    # TODO: here are two fp2 sqrt, do we want to do something
    #       in parallel here, or convert from 64-bit for the last
    #       two steps where we need sqrts?
    AB = sqrt_fp2(AA * BB)
    AC = sqrt_fp2(AA * CC)

    # NOTE: this uses 4 calls to pairwise_multiplication which
    #       is 16M but you can do it in 14M for serial computation.
    #       Good news is that the spec uses 15M instead of 14M, so
    #       there's an extra saving here we made :)
    #
    # x A^3 BC as a projective factor
    # A = xAB * AA * AC
    # B = AB * AC * xAB
    # C = xAB * AA * CC
    # D = xCD * AB * AA

    # x A^2 B^3 C^2 D^2 as a projective factor
    # A_inv = xAB * BB * CC * DD
    # B_inv = xAB * AB * CC * DD
    # C_inv = xAB * AC * BB * DD
    # D_inv = AB * BB * AC * xCD

    T0 = AVX_4x2x1(xAB, xAB, xCD, BB)
    T1 = AVX_4x2x1(AC, CC, AB, DD)
    t0, t1, t2, t3 = pairwise_multiplication(T0, T1)

    OB = AVX_4x2x1(AA, AB, AA, AA)
    T0 = AVX_4x2x1(t0, t0, t1, t2)
    OB = pairwise_multiplication(OB, T0)

    T0 = AVX_4x2x1(AB, BB, 0, 0)
    T1 = AVX_4x2x1(DD, AC, 0, 0)
    t4, t5, _, _ = pairwise_multiplication_2(T0, T1)

    inv = AVX_4x2x1(t1, t1, t0, t2)
    T0 = AVX_4x2x1(t3, t4, t3, t5)
    inv = pairwise_multiplication(inv, T0)

    if h2:
        OB = hadamard(OB)

    return OB, inv


def theta_codomain_with_no_torsion(
    OA: AVX_4x2x1,
    h1: bool,
    h2: bool,
) -> tuple[AVX_4x2x1, AVX_4x2x1]:
    if h1:
        T1 = hadamard(OA)
        T1 = squaring(T1)
        T1 = hadamard(T1)
    else:
        T1 = squaring(OA)
        T1 = hadamard(T1)

    # TODO: here are three fp2 sqrt, do we want to do something
    #       in parallel here, or convert from 64-bit for the last
    #       two steps where we need sqrts?
    AA, BB, CC, DD = T1
    AB = sqrt_fp2(AA * BB)
    AC = sqrt_fp2(AA * CC)
    AD = sqrt_fp2(AA * DD)
    OB = AVX_4x2x1(AA, AB, AC, AD)

    # This calls pairwise_multiplication twice for a total of
    # 8M where as in serial we only need 7, so 1 M is technically
    # redundant here.
    # A_inv = BB * CC * DD
    # B_inv = AB * CC * DD
    # C_inv = BB * AC * DD
    # D_inv = BB * CC * AD
    inv = AVX_4x2x1(BB, AB, BB, BB)
    T0 = AVX_4x2x1(CC, CC, AC, CC)
    inv = pairwise_multiplication(inv, T0)
    T0 = AVX_4x2x1(DD, DD, DD, AD)
    inv = pairwise_multiplication(inv, T0)

    if h2:
        OB = hadamard(OB)

    return OB, inv


def theta_codomain_8x1x1(
    ker: AVX_8x1x1, h1: bool, h2: bool
) -> tuple[AVX_4x2x1, AVX_4x2x1]:
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
