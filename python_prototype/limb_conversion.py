from elliptic_types import BasisX, Curve
from avx_types import AVX_2x4x1


def basis_from_64_to_51_bit(B: BasisX) -> tuple[AVX_2x4x1]:
    return [
        AVX_2x4x1(B.PX, B.PZ),
        AVX_2x4x1(B.QX, B.QZ),
        AVX_2x4x1(B.PmQX, B.PmQZ),
    ]


def basis_from_51_to_64_bit(B: tuple[AVX_2x4x1]):
    PX, PZ = B[0]
    QX, QZ = B[1]
    PmQX, PmQZ = B[2]
    return BasisX(PX, PZ, QX, QZ, PmQX, PmQZ)


def curve_to_2x4x1_data(E: Curve) -> tuple[AVX_2x4x1]:
    A = AVX_2x4x1(E.A, E.C)
    A24 = AVX_2x4x1(E.A24, E.C24)
    return A, A24
