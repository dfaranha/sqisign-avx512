from typing import Tuple
from theta_structure import pairwise_pair
from theta_types import AVX_8x1x1, AVX_4x2x1

EVEN_INDICES = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 0),
    (1, 2),
    (2, 0),
    (2, 1),
    (3, 0),
    (3, 3),
]


def chi(i, j):
    """
    Helper function to compute chi(i, j) where we know
    values of i, j are restricted into the range [0, 3]
    """
    if (i, j) in EVEN_INDICES:
        return 1
    return -1


def get_index_splitting(OA: AVX_4x2x1):
    """
    Algorithm 8.42 of the SQIsign spec.

    Compute the index (i, j) such that U_i,j(0) = 0

    Used to determine which precomputed matrix to use for the splitting isomorphism

    TODO: can we use parallelism here?
    """
    index = None
    count = 0
    for i, j in EVEN_INDICES:
        U = sum(chi(i, k) * OA[k] * OA[j ^ k] for k in range(4))
        if U == 0:
            # Ensure we find exactly one zero value
            index = (i, j)
            count += 1
    if index is None or count != 1:
        raise ValueError("error in index splitting computation")
    return index


def splitting_isomorphism_matrix(OA: AVX_4x2x1):
    """
    Algorithm 8.43 of the SQIsign spec.

    Given a theta null point isomorphic to a product structure, compute
    an isomorphism represented as a 4x4 matrix whose action on OA gives
    the theta null point associated to the splitting structure

    If `random_bytes` is not None, it is expected to be a function such that
    `random_bytes(n) generates n random bytes deterministically.
    """
    # assumes p = 3 mod 4 and we're working in Fp2 with modulus x^2 + 1
    i = OA[0].parent().gen()
    assert i * i == -1

    # We represent each matrix as two AVX_8x1x1 blobs, one with the first two rows, and
    # the other as the last two rows.
    SPLITTING_MAPS = {
        (0, 0): (
            AVX_8x1x1(1, i, 1, i, 1, -i, -1, i),
            AVX_8x1x1(1, i, -1, -i, -1, i, -1, i),
        ),
        (1, 0): (
            AVX_8x1x1(1, 1, 1, 1, 1, -1, -1, 1),
            AVX_8x1x1(1, 1, -1, -1, -1, 1, -1, 1),
        ),
        (2, 0): (
            AVX_8x1x1(1, 1, 1, 1, 1, -1, 1, -1),
            AVX_8x1x1(1, -1, -1, 1, -1, -1, 1, 1),
        ),
        (3, 0): (
            AVX_8x1x1(1, 1, 1, 1, 1, -1, 1, -1),
            AVX_8x1x1(1, 1, -1, -1, -1, 1, 1, -1),
        ),
        (0, 1): (AVX_8x1x1(1, 0, 0, 0, 0, 0, 0, 1), AVX_8x1x1(0, 0, 1, 0, 0, -1, 0, 0)),
        (2, 1): (
            AVX_8x1x1(1, 1, 1, 1, 1, -1, 1, -1),
            AVX_8x1x1(1, -1, -1, 1, 1, 1, -1, -1),
        ),
        (0, 2): (AVX_8x1x1(1, 0, 0, 0, 0, 1, 0, 0), AVX_8x1x1(0, 0, 0, 1, 0, 0, -1, 0)),
        (1, 2): (AVX_8x1x1(1, 0, 0, 0, 0, 1, 0, 0), AVX_8x1x1(0, 0, 0, 1, 0, 0, 1, 0)),
        (0, 3): (AVX_8x1x1(1, 0, 0, 0, 0, 1, 0, 0), AVX_8x1x1(0, 0, 1, 0, 0, 0, 0, -1)),
        (3, 3): (AVX_8x1x1(1, 0, 0, 0, 0, 1, 0, 0), AVX_8x1x1(0, 0, 1, 0, 0, 0, 0, 1)),
    }
    splitting_index = get_index_splitting(OA)
    M = SPLITTING_MAPS[splitting_index]

    return M


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


def apply_isomorphism(OA: AVX_4x2x1, M: Tuple[AVX_8x1x1, AVX_8x1x1]) -> AVX_4x2x1:
    """
    Compute M * v = x for a 4x4 matrix M and vector x in a parallel way
    """
    # First two and last two rows of the 4x4 matrixs
    M01, M23 = M

    M01 = pairwise_pair(M01, OA)
    M23 = pairwise_pair(M23, OA)

    return internal_sum(M01, M23)


def theta_to_product(OA: AVX_4x2x1):
    """
    Algorithm 8.45 of the SQIsign spec.

    Compute Montgomery coefficients (Ai : Ci) of the product E1 x E2
    of Montgomery curves from a theta null point OA with compatible
    splitting structure

    TODO: can we use parallelism here?
    """
    a, b, c, d = OA
    if a * d != b * c:
        raise ValueError("OA does not come from a product theta structure")

    # TODO:
    # here we can do 6 Fp2 squares or two calls to squaring on OA
    # Fourth powers of null point coords
    x = a**4
    y = b**4
    z = c**4

    # A2 = -2(a^4 + b^4) / (a^4 - b^4)
    A2 = -2 * (x + y)
    C2 = x - y

    # A1 = -2(a^4 + c^4) / (a^4 - c^4)
    A1 = -2 * (x + z)
    C1 = x - z

    if C1.is_zero() or C2.is_zero():
        raise ValueError("failed to split to a product of elliptic curves")

    return (A1, C1), (A2, C2)


def theta_product_point_to_montgomery(P: AVX_4x2x1, OA: AVX_4x2x1):
    """
    Algorithm 8.46 of the SQIsign spec.

    Given a theta null point OA in a compatible form with splitting, compute
    P1, P2 in E1 x E2 on the corresponding product of Montgomery curves

    TODO: can we use parallelism here?
    """
    a, b, c, _ = OA
    x, y, z, _ = P

    X1 = a * z + c * x
    Z1 = a * z - c * x

    X2 = a * y + b * x
    Z2 = a * z - b * x

    return (X1, Z1), (X2, Z2)
