from avx_types import AVX_4x2x1, AVX_8x1x1
from theta_structure import pairwise_pair


def deterministic_fp_root(a):
    """
    Ensure the square root matches deterministic output
    """
    if int(a) & 1:
        return -a
    return a


def deterministic_fp2_root(a):
    """
    Ensure the square root matches deterministic output
    """
    a0, a1 = a.list()

    a0_is_zero = a0.is_zero()
    a0_is_odd = int(a0) & 1
    a1_is_odd = int(a1) & 1

    # if a0 is odd, or a0 = 0 and a1 is odd, negate result
    if a0_is_odd or (a0_is_zero and a1_is_odd):
        return -a
    return a


def sqrt_fp(x):
    """
    Compute square-root using that p = 3 mod 4
    """
    p = x.parent().characteristic()
    assert p % 4 == 3
    r = x ** ((p + 1) // 4)
    return deterministic_fp_root(r)


def sqrt_fp2(x):
    """
    Algorithm 8.2 of the SQISign spec.

    Computes a square-root in Fp2 with modulus x^2 + 1
    """
    assert x.is_square(), f"Cannot take a sqrt as element {x} is not a square"

    F = x.parent()
    p = F.characteristic()

    a0, a1 = x.list()

    delta = sqrt_fp(a0**2 + a1**2)
    if a1.is_zero():
        delta = a0

    x0 = delta + a0
    t0 = 2 * x0
    x1 = t0 ** ((p - 3) // 4)

    x0 = x0 * x1
    x1 = x1 * a1
    t1 = (2 * x0) ** 2

    if t1 == t0:
        root = F([x0, x1])
    else:
        root = F([x1, -x0])

    assert root**2 == x
    return deterministic_fp2_root(root)


def internal_sum(A: AVX_8x1x1, B: AVX_8x1x1) -> AVX_4x2x1:
    """
    An internal helper function which should help with parallel
    matrix multiplication?
    """
    a0, b0, c0, d0, a1, b1, c1, d1 = A
    a2, b2, c2, d2, a3, b3, c3, d3 = B

    # Compute the sum of the first and last four elements
    # for each of the 8x1x1 blobs
    a = a0 + b0 + c0 + d0
    b = a1 + b1 + c1 + d1
    c = a2 + b2 + c2 + d2
    d = a3 + b3 + c3 + d3

    return AVX_4x2x1(a, b, c, d)


def apply_isomorphism(OA: AVX_4x2x1, M: tuple[AVX_8x1x1, AVX_8x1x1]) -> AVX_4x2x1:
    """
    Compute M * v = x for a 4x4 matrix M and vector x in a parallel way
    """
    # First two and last two rows of the 4x4 matrixs
    M01, M23 = M

    M01 = pairwise_pair(M01, OA)
    M23 = pairwise_pair(M23, OA)

    return internal_sum(M01, M23)
