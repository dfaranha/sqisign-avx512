from sage.all import Matrix
from theta_model.theta_structures import ThetaPoint, ThetaNullPoint

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


def get_index_splitting(OA: ThetaNullPoint):
    """
    Algorithm 8.42 of the SQIsign spec.

    Compute the index (i, j) such that U_i,j(0) = 0

    Used to determine which precomputed matrix to use for the splitting isomorphism
    """
    index = None
    count = 0
    for i, j in EVEN_INDICES:
        U = sum(chi(i, k) * OA[k] * OA[j ^ k] for k in range(4))
        if U.is_zero():
            # Ensure we find exactly one zero value
            index = (i, j)
            count += 1
    if index is None or count != 1:
        raise ValueError("error in index splitting computation")
    return index


def splitting_isomorphism_matrix(OA: ThetaNullPoint, random_bytes=None):
    """
    Algorithm 8.43 of the SQIsign spec.

    Given a theta null point isomorphic to a product structure, compute
    an isomorphism represented as a 4x4 matrix whose action on OA gives
    the theta null point associated to the splitting structure

    If `random_bytes` is not None, it is expected to be a function such that
    `random_bytes(n) generates n random bytes deterministically.
    """
    # assumes p = 3 mod 4 and we're working in Fp2 with modulus x^2 + 1
    i = OA.base_ring().gen()
    assert i * i == -1

    SPLITTING_MAPS = {
        (0, 0): [1, i, 1, i, 1, -i, -1, i, 1, i, -1, -i, -1, i, -1, i],
        (1, 0): [1, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, -1, -1, 1, -1, 1],
        (2, 0): [1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1],
        (3, 0): [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1],
        (0, 1): [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, -1, 0, 0],
        (2, 1): [1, 1, 1, 1, 1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, -1],
        (0, 2): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, -1, 0],
        (1, 2): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        (0, 3): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1],
        (3, 3): [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    }
    splitting_index = get_index_splitting(OA)
    M = SPLITTING_MAPS[splitting_index]
    M = Matrix(OA.base_ring(), 4, 4, M)

    if random_bytes is not None:
        M = randomized_product(M, i, random_bytes)
    return M


def randomized_product(M, i, random_bytes):
    """
    Algorithm 8.44 of the SQIsign spec.

    Randomize the splitting matrix M by some isomorphism of the product curves
    """
    # assumes p = 3 mod 4 and we're working in Fp2 with modulus x^2 + 1
    assert i * i == -1
    RANDOMIZATION_MAPS = {
        [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
        [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
        [1, -1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, 1, 1],
        [-1, i, i, 1, i, -1, 1, i, i, 1, -1, i, 1, i, i, -1],
        [1, i, i, 1, i, 1, -1, i, i, -1, 1, i, -1, i, i, 1],
    }

    random_index = int.from_bytes(random_bytes(4), byteorder="little") % 6

    N = RANDOMIZATION_MAPS[random_index]
    N = Matrix(M.base_ring(), 4, 4, N)

    return N * M


def theta_to_product(OA: ThetaNullPoint):
    """
    Algorithm 8.45 of the SQIsign spec.

    Compute Montgomery coefficients (Ai : Ci) of the product E1 x E2
    of Montgomery curves from a theta null point OA with compatible
    splitting structure
    """
    a, b, c, d = OA.coords()
    if a * d != b * c:
        raise ValueError("OA does not come from a product theta structure")

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


def theta_product_point_to_montgomery(P: ThetaPoint, OA: ThetaNullPoint):
    """
    Algorithm 8.46 of the SQIsign spec.

    Given a theta null point OA in a compatible form with splitting, compute
    P1, P2 \in E1 x E2 on the corresponding product of Montgomery curves
    """
    a, b, c, _ = OA.coords()
    x, y, z, _ = P.coords()

    X1 = a * z + c * x
    Z1 = a * z - c * x

    X2 = a * y + b * x
    Z2 = a * z - b * x

    return (X1, Z1), (X2, Z2)
