from avx_types import AVX_4x2x1, AVX_8x1x1


def avx_811_to_421(wide: AVX_8x1x1) -> tuple[AVX_4x2x1, AVX_4x2x1]:
    """
    Convert a pair of points within a AVX_8x1x1 to two AVX_4x2x1
    """
    A = AVX_4x2x1(*wide[:4])
    B = AVX_4x2x1(*wide[4:])
    return A, B


def avx_421_to_811(A: AVX_4x2x1, B: AVX_4x2x1) -> AVX_8x1x1:
    """
    Convert a pair of AVX_4x2x1 points to a single AVX_8x1x1
    """
    return AVX_8x1x1(*(A + B))


def avx_421_neg(A: AVX_4x2x1) -> AVX_4x2x1:
    """
    Neg components of an AVX blob
    """
    return AVX_4x2x1(*[-a for a in A])


def avx_421_add(A: AVX_4x2x1, B: AVX_4x2x1) -> AVX_4x2x1:
    """
    Add components of two AVX blobs
    """
    return AVX_4x2x1(*[a + b for a, b in zip(A, B)])


def avx_421_sub(A: AVX_4x2x1, B: AVX_4x2x1) -> AVX_4x2x1:
    """
    Sub components of two AVX blobs
    """
    return AVX_4x2x1(*[a - b for a, b in zip(A, B)])


def avx_421_dbl(
    A: AVX_4x2x1,
) -> AVX_4x2x1:
    """
    Double components of an AVX blob
    """
    return AVX_4x2x1(*[a + a for a in A])


def avx_421_sqr(P: AVX_4x2x1) -> AVX_4x2x1:
    """
    Square components of an AVX blob
    """
    return AVX_4x2x1(*[x * x for x in P])


def avx_421_mul(P: AVX_4x2x1, Q: AVX_4x2x1) -> AVX_4x2x1:
    """
    Pairwise multiplication of AVX blobs
    """
    return AVX_4x2x1(P.x * Q.x, P.y * Q.y, P.z * Q.z, P.w * Q.w)


def avx_811_add(A: AVX_8x1x1, B: AVX_8x1x1) -> AVX_8x1x1:
    """
    Sub components of two AVX_8x1x1 blobs
    """
    return AVX_8x1x1(*[a + b for a, b in zip(A, B)])


def avx_811_sub(A: AVX_8x1x1, B: AVX_8x1x1) -> AVX_8x1x1:
    """
    Sub components of two AVX_8x1x1 blobs
    """
    return AVX_8x1x1(*[a - b for a, b in zip(A, B)])


def avx_811_sqr(
    A: AVX_8x1x1,
) -> AVX_8x1x1:
    """
    Square components of an AVX_8x1x1 blob
    """
    return AVX_8x1x1(*[a * a for a in A])


def avx_811_mul(P: AVX_8x1x1, Q: AVX_8x1x1) -> AVX_8x1x1:
    """
    Pairwise multiplication of AVX_8x1x1 blobs
    """
    return AVX_8x1x1(*[a * b for a, b in zip(P, Q)])


def avx_421_pair_mul(A: AVX_4x2x1, B: AVX_4x2x1, C: AVX_4x2x1, D: AVX_4x2x1):
    """
    Compute the pairwise mul of A * B, C * D using 8x1x1 multiplications
    """
    P = avx_421_to_811(A, C)
    Q = avx_421_to_811(B, D)
    R = avx_811_mul(P, Q)
    return avx_811_to_421(R)


def avx_421_pair_sqr(A: AVX_4x2x1, B: AVX_4x2x1):
    """
    Compute the squares A**2, b**2 using 8x1x1 squaring
    """
    R = avx_421_to_811(A, B)
    R = avx_811_sqr(R)
    return avx_811_to_421(R)
