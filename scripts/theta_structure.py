from theta_types import AVX_8x1x1, AVX_4x2x1


def hadamard(P: AVX_4x2x1) -> AVX_4x2x1:
    """
    The Hadamard transform
    """
    a = P.x + P.y + P.z + P.w
    b = P.x - P.y + P.z - P.w
    c = P.x + P.y - P.z - P.w
    d = P.x - P.y - P.z + P.w
    return AVX_4x2x1(a, b, c, d)


def hadamard_pair(P_Q: AVX_8x1x1) -> AVX_8x1x1:
    """
    The Hadamard transform on a pair of points
    """
    a1 = P_Q.x1 + P_Q.y1 + P_Q.z1 + P_Q.w1
    b1 = P_Q.x1 - P_Q.y1 + P_Q.z1 - P_Q.w1
    c1 = P_Q.x1 + P_Q.y1 - P_Q.z1 - P_Q.w1
    d1 = P_Q.x1 - P_Q.y1 - P_Q.z1 + P_Q.w1

    a2 = P_Q.x2 + P_Q.y2 + P_Q.z2 + P_Q.w2
    b2 = P_Q.x2 - P_Q.y2 + P_Q.z2 - P_Q.w2
    c2 = P_Q.x2 + P_Q.y2 - P_Q.z2 - P_Q.w2
    d2 = P_Q.x2 - P_Q.y2 - P_Q.z2 + P_Q.w2

    return AVX_8x1x1(a1, b1, c1, d1, a2, b2, c2, d2)


def squaring(P: AVX_4x2x1) -> AVX_4x2x1:
    """
    The squaring transform.
    """
    return AVX_4x2x1(*[x * x for x in P])


def squaring_pair(P_Q: AVX_8x1x1) -> AVX_8x1x1:
    """
    The squaring transform on a pair of points
    """
    return AVX_8x1x1(*[x * x for x in P_Q])


def pairwise_multiplication(P: AVX_4x2x1, Q: AVX_4x2x1) -> AVX_4x2x1:
    """
    Pairwise multiplication of two points
    """
    return AVX_4x2x1(P.x * Q.x, P.y * Q.y, P.z * Q.z, P.w * Q.w)


def pairwise_pair(P_Q: AVX_8x1x1, C: AVX_4x2x1) -> AVX_8x1x1:
    """
    Pairwise multiplication of two pairs of points
    """
    a1, b1, c1, d1 = P_Q.x1 * C.x, P_Q.y1 * C.y, P_Q.z1 * C.z, P_Q.w1 * C.w
    a2, b2, c2, d2 = P_Q.x2 * C.x, P_Q.y2 * C.y, P_Q.z2 * C.z, P_Q.w2 * C.w

    return AVX_8x1x1(a1, b1, c1, d1, a2, b2, c2, d2)
