from avx_types import AVX_4x2x1, AVX_8x1x1
from avx_arithmetic import avx_421_add, avx_811_add, avx_811_mul, avx_421_to_811


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
    a0, a1 = a.polynomial().list()

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

    a0, a1 = x.polynomial().list()

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


def batched_inversion(*input):
    """
    Given k values, compute the inverse using
    3(k-1) multiplications and one inversion
    """
    # for values a0, a1, ..., an compute
    # a0, a0a1, a0a1a2, ... a0...an using
    # (k-1) multiplications
    multiples = [input[0]]
    for ai in input[1:]:
        multiples.append(ai * multiples[-1])

    # Compute 1 / (a0a1a2...an)
    last_multiple = multiples[-1]
    inverses_multiples = [1 / last_multiple]

    # Compute (a0a1a2...an)^-1, (a0a1a2...a(n-1)) ... a0^-1
    # using k-1 multiplications
    for ai in reversed(input[1:]):
        inverses_multiples.append(ai * inverses_multiples[-1])

    # Reverse for easy ordering below
    # inverses_multiples = a0^-1, (a0a1)^-1 ...(a0a1...an)^-1
    inverses_multiples = inverses_multiples[::-1]
    inverses = [inverses_multiples[0]]

    # Compute the inverse of each element from multiples and
    # their inverses using k-1 multiplications
    k = len(input)
    for i in range(1, k):
        inverses.append(inverses_multiples[i] * multiples[i - 1])

    return inverses


def internal_sum(A: AVX_8x1x1, B: AVX_8x1x1) -> AVX_4x2x1:
    """
    An internal helper function which should help with parallel
    matrix multiplication?
    """
    a0, b0, c0, d0, a1, b1, c1, d1 = A
    a2, b2, c2, d2, a3, b3, c3, d3 = B

    # Compute the sum of the elements for each of the 8x1x1 blobs
    # a = a0 + b0 + c0 + d0
    # b = a1 + b1 + c1 + d1
    # c = a2 + b2 + c2 + d2
    # d = a3 + b3 + c3 + d3
    C = AVX_4x2x1(a0, a1, a2, a3)
    tmp = AVX_4x2x1(b0, b1, b2, b3)
    C = avx_421_add(C, tmp)
    tmp = AVX_4x2x1(c0, c1, c2, c3)
    C = avx_421_add(C, tmp)
    tmp = AVX_4x2x1(d0, d1, d2, d3)
    C = avx_421_add(C, tmp)

    return C


def apply_isomorphism(P: AVX_4x2x1, M: tuple[AVX_8x1x1, AVX_8x1x1]) -> AVX_4x2x1:
    """
    Compute M * v = x for a 4x4 matrix M and vector x in a parallel way
    """
    # First two and last two rows of the 4x4 matrices
    M01, M23 = M

    # Create an 8x1x1 from P | P
    PP = avx_421_to_811(P, P)

    # First and second row multiplied by P
    M01 = avx_811_mul(M01, PP)

    # Second and third row multiplied by P
    M23 = avx_811_mul(M23, PP)

    # Sum up multiplications to compute the four coordinates
    return internal_sum(M01, M23)


def internal_sum_pair(
    A: AVX_8x1x1, B: AVX_8x1x1, C: AVX_8x1x1, D: AVX_8x1x1
) -> AVX_8x1x1:
    """
    An internal helper function which should help with parallel
    matrix multiplication?
    """
    Pa0, Pb0, Pc0, Pd0, Qa1, Qb1, Qc1, Qd1 = A
    Qa0, Qb0, Qc0, Qd0, Pa1, Pb1, Pc1, Pd1 = B
    Pa2, Pb2, Pc2, Pd2, Qa3, Qb3, Qc3, Qd3 = C
    Qa2, Qb2, Qc2, Qd2, Pa3, Pb3, Pc3, Pd3 = D

    # Compute the sum of the elements for each of the 8x1x1 blobs
    # P:  a = Pa0 + Pb0 + Pc0 + Pd0
    #     b = Pa1 + Pb1 + Pc1 + Pd1
    #     c = Pa2 + Pb2 + Pc2 + Pd2
    #     d = Pa3 + Pb3 + Pc3 + Pd3
    #
    # Q:  a = Qa0 + Qb0 + Qc0 + Qd0
    #     b = Qa1 + Qb1 + Qc1 + Qd1
    #     c = Qa2 + Qb2 + Qc2 + Qd2
    #     d = Qa3 + Qb3 + Qc3 + Qd3
    E = AVX_8x1x1(Pa0, Pa1, Pa2, Pa3, Qa0, Qa1, Qa2, Qa3)

    tmp = AVX_8x1x1(Pb0, Pb1, Pb2, Pb3, Qb0, Qb1, Qb2, Qb3)
    E = avx_811_add(E, tmp)

    tmp = AVX_8x1x1(Pc0, Pc1, Pc2, Pc3, Qc0, Qc1, Qc2, Qc3)
    E = avx_811_add(E, tmp)

    tmp = AVX_8x1x1(Pd0, Pd1, Pd2, Pd3, Qd0, Qd1, Qd2, Qd3)
    E = avx_811_add(E, tmp)

    return E


def apply_isomorphism_pair(PQ: AVX_8x1x1, M: tuple[AVX_8x1x1, AVX_8x1x1]) -> AVX_4x2x1:
    """
    Compute M * v = x for a 4x4 matrix M and vector x in a parallel way
    """
    # First two and last two rows of the 4x4 matrices
    M01, M23 = M

    # Swap P || Q in a doubled point to Q || P
    QP = AVX_8x1x1(*PQ[4:], *PQ[:4])

    # Compute first row mul P, second row mul Q
    P0Q1 = avx_811_mul(M01, PQ)

    # Compute first row mul Q, second row mul P
    Q0P1 = avx_811_mul(M01, QP)

    # Compute third row mul P, fourth row mul Q
    P2Q3 = avx_811_mul(M23, PQ)

    # Compute third row mul Q, fourth row mul P
    Q2P3 = avx_811_mul(M23, QP)

    return internal_sum_pair(P0Q1, Q0P1, P2Q3, Q2P3)
