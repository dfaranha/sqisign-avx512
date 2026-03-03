from avx_types import AVX_2x4x1, AVX_4x2x1, AVX_8x1x1

# ======================================================
# Conversion to and from different AVX representations
# ======================================================


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


def avx_421_to_241(P: AVX_4x2x1) -> tuple[AVX_2x4x1]:
    """
    Convert a pair of points within a AVX_4x2x1 to two AVX_2x4x1
    """
    A = AVX_2x4x1(*P[:2])
    B = AVX_2x4x1(*P[2:])
    return A, B


def avx_241_to_421(A: AVX_2x4x1, B: AVX_2x4x1) -> AVX_4x2x1:
    """
    Convert a pair of AVX_4x2x1 points to a single AVX_4x2x1
    """
    return AVX_4x2x1(*(A + B))


def avx_241_to_811(A: AVX_2x4x1, B: AVX_2x4x1, C: AVX_2x4x1, D: AVX_2x4x1) -> AVX_8x1x1:
    """
    Convert four AVX_2x4x1 points to a single AVX_8x1x1
    """
    return AVX_8x1x1(*(A + B + C + D))


def avx_811_to_241(P: AVX_8x1x1) -> tuple[AVX_2x4x1]:
    """
    Convert one AVX_8x1x1 into four AVX_2x4x1 points
    """
    A = AVX_2x4x1(*P[:2])
    B = AVX_2x4x1(*P[2:4])
    C = AVX_2x4x1(*P[4:6])
    D = AVX_2x4x1(*P[6:])

    return A, B, C, D


# ======================================================
# AVX 2x4x1 Arithmetic
# ======================================================


def avx_241_add(A: AVX_2x4x1, B: AVX_2x4x1) -> AVX_2x4x1:
    """
    Add components of two AVX_2x4x1 blobs
    """
    return AVX_2x4x1(A[0] + B[0], A[1] + B[1])


def avx_241_sub(A: AVX_2x4x1, B: AVX_2x4x1) -> AVX_2x4x1:
    """
    Sub components of two AVX_2x4x1 blobs
    """
    return AVX_2x4x1(A[0] - B[0], A[1] - B[1])


def avx_241_add_sub(A: AVX_2x4x1, B: AVX_2x4x1) -> AVX_2x4x1:
    """
    Add the first and subtract the second components
    """
    return AVX_2x4x1(A[0] + B[0], A[1] - B[1])


def avx_241_mul(A: AVX_2x4x1, B: AVX_2x4x1) -> AVX_2x4x1:
    """
    Multiply components of two AVX_2x4x1 blobs
    """
    return AVX_2x4x1(A[0] * B[0], A[1] * B[1])


def avx_241_sqr(A: AVX_2x4x1) -> AVX_2x4x1:
    """
    Multiply components of two AVX_2x4x1 blobs
    """
    return AVX_2x4x1(A[0] * A[0], A[1] * A[1])


# ======================================================
# AVX 4x2x1 Arithmetic
# ======================================================


def avx_421_all_zero(A: AVX_4x2x1) -> bool:
    """
    Return True if all elements of A are zero and False otherwise
    """
    return all(a == 0 for a in A)


def avx_421_eq(A: AVX_4x2x1, B: AVX_4x2x1) -> bool:
    """
    Return True if A and B are equal, and False otherwise
    """
    T = avx_421_sub(A, B)
    return avx_421_all_zero(T)


def avx_421_any_zero(A: AVX_4x2x1) -> bool:
    """
    Return True if any elements of A are zero and False otherwise
    """
    return not all(a != 0 for a in A)


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


def avx_421_add_sub(A: AVX_2x4x1, B: AVX_2x4x1) -> AVX_2x4x1:
    """
    Add the first and subtract the second components
    """
    return AVX_4x2x1(A[0] + B[0], A[1] + B[1], A[2] - B[2], A[3] - B[3])


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


# ======================================================
# AVX 8x1x1 Arithmetic
# ======================================================


def avx_811_any_zero(A: AVX_8x1x1) -> bool:
    """
    Return True if any elements of A are zero and False otherwise
    """
    return not all(a != 0 for a in A)


def avx_811_add(A: AVX_8x1x1, B: AVX_8x1x1) -> AVX_8x1x1:
    """
    Add components of two AVX_8x1x1 blobs
    """
    return AVX_8x1x1(*[a + b for a, b in zip(A, B)])


def avx_811_sub(A: AVX_8x1x1, B: AVX_8x1x1) -> AVX_8x1x1:
    """
    Sub components of two AVX_8x1x1 blobs
    """
    return AVX_8x1x1(*[a - b for a, b in zip(A, B)])


def avx_811_add_sub(A: AVX_8x1x1, B: AVX_8x1x1) -> AVX_8x1x1:
    """
    Add the first four and subtract the last four components
    """
    return AVX_8x1x1(
        A[0] + B[0],
        A[1] + B[1],
        A[2] + B[2],
        A[3] + B[3],
        A[4] - B[4],
        A[5] - B[5],
        A[6] - B[6],
        A[7] - B[7],
    )


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
