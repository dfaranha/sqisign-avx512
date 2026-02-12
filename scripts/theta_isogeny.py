from avx_types import AVX_4x2x1, AVX_8x1x1
from theta_structure import (
    hadamard,
    hadamard_pair,
)
from avx_arithmetic import (
    avx_421_sqr,
    avx_811_sqr,
    avx_811_mul,
    avx_421_mul,
    avx_421_pair_mul,
    avx_421_to_811,
    avx_811_any_zero,
    avx_421_eq,
)
from utilities import sqrt_fp2


def check_isotropic(P: AVX_4x2x1, Q: AVX_4x2x1, inv: AVX_4x2x1):
    """
    Algorithm 8.32 of the SQISign spec.

    Return a boolean indicating whether the 4-torsion points are isotropic
    """
    T1 = AVX_8x1x1(P[0], P[1], P[2], P[3], Q[0], Q[2], Q[1], Q[3])
    T2 = AVX_8x1x1(inv[0], inv[1], inv[2], inv[3], inv[0], inv[2], inv[1], inv[3])
    T1 = avx_811_mul(T1, T2)

    S1 = AVX_4x2x1(T1[0], T1[2], T1[4], T1[6])
    S2 = AVX_4x2x1(T1[1], T1[3], T1[5], T1[7])
    return avx_421_eq(S1, S2)


def theta_precompute(
    OA: AVX_4x2x1, use_wide=False
) -> tuple[AVX_4x2x1, AVX_4x2x1] | tuple[AVX_8x1x1, AVX_8x1x1]:
    """
    Precompute constants given a theta null point

    Cost: uses 3 4x2x1 multiplications instead of 12 Fp2 multiplications
          but requires additional memory management to form different AVX512
          blobs along the way
    """
    # Compute H . S(OA)
    OA_prime = avx_421_sqr(OA)
    OA_prime = hadamard(OA_prime)

    # Memory management for 4x2x1 computations
    a, b, c, d = OA
    AA, BB, CC, DD = OA_prime
    t1 = AVX_4x2x1(AA, CC, a, c)
    t2 = AVX_4x2x1(BB, DD, b, d)
    C1 = AVX_4x2x1(BB, AA, DD, CC)
    C2 = AVX_4x2x1(b, a, d, c)

    # Precompute four multiplications AABB, CCDD, ab and cd
    t2 = avx_421_mul(t1, t2)

    # More memory management to repackage the first multiplication
    AABB, CCDD, ab, cd = t2
    t1 = AVX_4x2x1(CCDD, CCDD, AABB, AABB)
    t2 = AVX_4x2x1(cd, cd, ab, ab)

    # Compute the arithmetic constants more memory management and two more muls
    C1, C2 = avx_421_pair_mul(C1, t1, C2, t2)

    # If we are expecting to multiply pairs of points, we convert C1, C2 to
    # 8x1x1 by repeating them
    if use_wide:
        C1 = avx_421_to_811(C1, C1)
        C2 = avx_421_to_811(C2, C2)

    return C1, C2


def theta_double(P: AVX_4x2x1, C1: AVX_4x2x1, C2: AVX_4x2x1) -> AVX_4x2x1:
    """
    Double a point provided certain precomputed constants
    """
    P2 = avx_421_sqr(P)
    P2 = hadamard(P2)
    P2 = avx_421_sqr(P2)
    P2 = avx_421_mul(P2, C1)
    P2 = hadamard(P2)
    P2 = avx_421_mul(P2, C2)
    return P2


def theta_double_pair(P_Q: AVX_8x1x1, C1: AVX_8x1x1, C2: AVX_8x1x1) -> AVX_8x1x1:
    """
    Double a point provided certain precomputed constants
    """
    P2_Q2 = avx_811_sqr(P_Q)
    P2_Q2 = hadamard_pair(P2_Q2)
    P2_Q2 = avx_811_sqr(P2_Q2)
    P2_Q2 = avx_811_mul(P2_Q2, C1)
    P2_Q2 = hadamard_pair(P2_Q2)
    P2_Q2 = avx_811_mul(P2_Q2, C2)
    return P2_Q2


def theta_codomain_4x2x1(
    P: AVX_4x2x1, Q: AVX_4x2x1, h1: bool, h2: bool
) -> tuple[AVX_4x2x1, AVX_4x2x1]:
    if h1:
        P = hadamard(P)
        Q = hadamard(Q)

    P = avx_421_sqr(P)
    P = hadamard(P)

    Q = avx_421_sqr(Q)
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
    t2 = avx_421_mul(t1, t2)

    # Memory management to get the precomputations we need
    xAwB, zAxB, zCwD, _ = t2
    t1 = AVX_4x2x1(xAwB, zAxB, xAwB, zAxB)

    # Compute the codomain
    OB = avx_421_mul(t1, Q)

    # Memory management to get the values needed for inverse
    _, _, C, D = OB
    inv = AVX_4x2x1(xB, xA, D, C)
    t1 = AVX_4x2x1(zCwD, zCwD, 1, 1)

    # Here we only need two multiplications, we could make a special function
    inv = avx_421_mul(inv, t1)

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
        ker_P = hadamard(ker_P)
        OA = hadamard(OA)

    T1 = avx_421_sqr(ker_P)
    T1 = hadamard(T1)

    T2 = avx_421_sqr(OA)
    T2 = hadamard(T2)

    xAB, _, xCD, _ = T1
    AA, BB, CC, DD = T2

    T0 = AVX_4x2x1(AA, AA, xAB, BB)
    T1 = AVX_4x2x1(BB, CC, CC, DD)
    AABB, AACC, xABCC, BBDD = avx_421_mul(T0, T1)

    # TODO: here are two fp2 sqrt, do we want to do something
    #       in parallel here, or convert from 64-bit for the last
    #       two steps where we need sqrts?
    AB = sqrt_fp2(AABB)
    AC = sqrt_fp2(AACC)

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

    T0 = AVX_4x2x1(xAB, AB, xCD, BB)
    T1 = AVX_4x2x1(AC, DD, AB, AC)
    xABAC, ABDD, xCDAB, BBAC = avx_421_mul(T0, T1)

    OB = AVX_4x2x1(AA, AB, AA, AA)
    T0 = AVX_4x2x1(xABAC, xABAC, xABCC, xCDAB)
    OB = avx_421_mul(OB, T0)

    inv = AVX_4x2x1(xABCC, xABCC, xABAC, xCDAB)
    T0 = AVX_4x2x1(BBDD, ABDD, BBDD, BBAC)
    inv = avx_421_mul(inv, T0)

    if h2:
        OB = hadamard(OB)

    return OB, inv


def theta_codomain_with_no_torsion(
    OA: AVX_4x2x1,
    h1: bool,
    h2: bool,
) -> tuple[AVX_4x2x1, AVX_4x2x1]:
    if h1:
        OA = hadamard(OA)
    T1 = avx_421_sqr(OA)
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
    inv = avx_421_mul(inv, T0)
    T0 = AVX_4x2x1(DD, DD, DD, AD)
    inv = avx_421_mul(inv, T0)

    if h2:
        OB = hadamard(OB)

    return OB, inv


def theta_codomain_8x1x1(
    ker: AVX_8x1x1, h1: bool, h2: bool, verify: bool = False
) -> tuple[AVX_4x2x1, AVX_4x2x1]:
    if h1:
        ker = hadamard_pair(ker)
    ker = avx_811_sqr(ker)
    ker = hadamard_pair(ker)

    # Memory management of kernel data to make two temp blobs
    xA, xB, yC, yD, zA, wB, zC, wD = ker
    OB = AVX_4x2x1(zA, wB, zC, wD)
    t1 = AVX_4x2x1(xA, zA, zC, 1)
    t2 = AVX_4x2x1(wB, xB, wD, 1)

    # Verification check: ensure there is no zero in the following data
    if verify:
        no_zero = AVX_8x1x1(xA, xB, zA, wB, zC, wD, 1, 1)
        if avx_811_any_zero(no_zero):
            raise ValueError("Unexpected splitting")

    # Here we only need three multiplications, we could make a special function
    t2 = avx_421_mul(t1, t2)

    # Memory management to get the precomputations we need
    xAwB, zAxB, zCwD, _ = t2
    t1 = AVX_4x2x1(xAwB, zAxB, xAwB, zAxB)
    inv = AVX_4x2x1(xB, xA, 1, 1)
    t2 = AVX_4x2x1(zCwD, zCwD, 1, 1)

    # Compute the codomain and computations for the inverse
    OB, inv = avx_421_pair_mul(t1, OB, inv, t2)

    # Memory management to get the values needed for inverse
    _, _, C, D = OB
    A_inv, B_inv, _, _ = inv
    inv = AVX_4x2x1(A_inv, B_inv, D, C)

    if verify:
        P = AVX_4x2x1(xA, xB, yC, yD)
        Q = AVX_4x2x1(zA, wB, zC, wD)
        if not check_isotropic(P, Q, inv):
            raise ValueError("P, Q are not isotropic")

    if h2:
        OB = hadamard(OB)

    return OB, inv


def theta_eval(P: AVX_4x2x1, inv: AVX_4x2x1, h1: bool, h2: bool) -> AVX_4x2x1:
    """
    Evaluate a theta point through an isogeny given precomputed constants
    """
    if h1:
        P = hadamard(P)

    P = avx_421_sqr(P)
    P = hadamard(P)
    P = avx_421_mul(P, inv)

    if h2:
        P = hadamard(P)

    return P


def theta_eval_pair(P_Q: AVX_8x1x1, inv: AVX_8x1x1, h1: bool, h2: bool):
    """
    Evaluate a theta point through an isogeny given precomputed constants
    """
    if h1:
        P_Q = hadamard_pair(P_Q)

    P_Q = avx_811_sqr(P_Q)
    P_Q = hadamard_pair(P_Q)
    P_Q = avx_811_mul(P_Q, inv)

    if h2:
        P_Q = hadamard_pair(P_Q)

    return P_Q
