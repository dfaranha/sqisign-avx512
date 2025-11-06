from avx_types import AVX_4x2x1, AVX_8x1x1


def avx_neg(A: AVX_4x2x1) -> AVX_4x2x1:
    """
    Neg components of an AVX blob
    """
    return AVX_4x2x1(*[-a for a in A])


def avx_add(A: AVX_4x2x1, B: AVX_4x2x1) -> AVX_4x2x1:
    """
    Add components of two AVX blobs
    """
    return AVX_4x2x1(*[a + b for a, b in zip(A, B)])


def avx_sub(A: AVX_4x2x1, B: AVX_4x2x1) -> AVX_4x2x1:
    """
    Sub components of two AVX blobs
    """
    return AVX_4x2x1(*[a - b for a, b in zip(A, B)])


def avx_dbl(
    A: AVX_4x2x1,
) -> AVX_4x2x1:
    """
    Double components of an AVX blob
    """
    return AVX_4x2x1(*[a + a for a in A])


def avx_sqr(P: AVX_4x2x1) -> AVX_4x2x1:
    """
    Square components of an AVX blob
    """
    return AVX_4x2x1(*[x * x for x in P])


def avx_pairwise_mul(P: AVX_4x2x1, Q: AVX_4x2x1) -> AVX_4x2x1:
    """
    Pairwise multiplication of AVX blobs
    """
    return AVX_4x2x1(P.x * Q.x, P.y * Q.y, P.z * Q.z, P.w * Q.w)


def avx_wide_pairwise_mul(P: AVX_8x1x1, Q: AVX_8x1x1) -> AVX_8x1x1:
    """
    Pairwise multiplication of AVX_8x1x1 blobs
    """
    return AVX_8x1x1(*[a * b for a, b in zip(P, Q)])
