from avx_types import AVX_8x1x1, AVX_4x2x1


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
