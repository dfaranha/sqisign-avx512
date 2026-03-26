from elliptic_types import PointX
from avx_types import AVX_4x2x1, AVX_8x1x1
from avx_arithmetic import (
    avx_421_sqr,
    avx_421_mul,
    avx_421_add,
    avx_421_sub,
    avx_421_neg,
    avx_811_mul,
    avx_811_sub,
    avx_421_pair_mul,
    avx_811_sqr,
    avx_421_all_zero,
    avx_421_any_zero,
    avx_421_eq,
    avx_421_to_811,
)
from theta_structure import (
    hadamard,
    hadamard_pair,
)
from elliptic_arithmetic import x_double_4x4x2x1, four_add_components


def gluing_kernel_to_x_only(X: AVX_4x2x1, Z: AVX_4x2x1) -> tuple[AVX_4x2x1, AVX_4x2x1]:
    """
    Helper function:

    Given the kernel as the X, Y, Z coordinates of Jacobian coordinates
    return the x-only coordinates (X : Z)
    """
    return X, avx_421_sqr(Z)


def compute_symmetric_position(X_4: AVX_4x2x1, Z_4: AVX_4x2x1):
    x1, x2, x3, x4 = X_4
    z1, z2, z3, z4 = Z_4

    # TODO: in SQIsign we know pos_2 will be 3 or 4 only, which
    # helps, but not all that much and I wont include this saving
    # for now (until I understand how this works)

    # The positions we will find
    pos_1 = None
    pos_2 = None

    # 8 parallel additions
    r1 = x1 + z1
    s1 = x1 - z1

    r2 = x2 + z2
    s2 = x2 - z2

    r3 = x3 + z3
    s3 = x3 - z3

    r4 = x4 + z4
    s4 = x4 - z4

    # Four parallel multiplications
    # TODO: mul by zeta is just rearranging memory and a
    # sign flip
    zeta = x1.parent().gen()
    t1 = zeta * r1 * s2
    t2 = r2 * s1
    t3 = zeta * r3 * s4
    t4 = r4 * s3

    # Four parallel additions
    u1 = t1 + t2
    u2 = t1 - t2
    u3 = t3 + t4
    u4 = t3 - t4

    # Set positions
    if s1 == 0:
        pos_1 = 0
    elif r1 == 0:
        pos_1 = 1
    elif s2 == 0:
        pos_1 = 2
    elif r2 == 0:
        pos_1 = 3
    elif u1 == 0:
        pos_1 = 4
    elif u2 == 0:
        pos_1 = 5
    else:
        raise ValueError("issue with four torsion")

    # Set positions
    if s3 == 0:
        pos_2 = 0
    elif r3 == 0:
        pos_2 = 1
    elif s4 == 0:
        pos_2 = 2
    elif r4 == 0:
        pos_2 = 3
    elif u3 == 0:
        pos_2 = 4
    elif u4 == 0:
        pos_2 = 5
    else:
        raise ValueError("issue with four torsion")

    return pos_1, pos_2


def compute_matrix_coefficients(x, z):
    # TODO: parallel computation, basic for now
    s1 = x**2
    s2 = z**2
    delta = s1 - s2
    r1 = s1 + s2
    r2 = x * z
    r2 = r2 + r2
    return r1, r2, delta


def compute_symmetric_elements(r1, r2, delta, pos):
    # TODO: mul by zeta is just sign flips and memory...
    F = r1.parent()
    zeta = F.gen()

    # TODO: make this parallel nicely...
    # TODO: pos 3, 4 and 5, 6 are just 1, 2 but ordered differently
    if pos == 0:
        g1 = [F.zero(), delta, delta]
        g2 = [r1, -r2, r2]
        g3 = [zeta * r2, -zeta * r1, zeta * r1]
    elif pos == 1:
        g1 = [F.zero(), -delta, -delta]
        g2 = [r1, -r2, r2]
        g3 = [-zeta * r2, zeta * r1, -zeta * r1]
    elif pos == 2:
        g1 = [r1, -r2, r2]
        g2 = [F.zero(), delta, delta]
        g3 = [-zeta * r2, zeta * r1, -zeta * r1]
    elif pos == 3:
        g1 = [r1, -r2, r2]
        g2 = [F.zero(), -delta, -delta]
        g3 = [zeta * r2, -zeta * r1, zeta * r1]
    elif pos == 4:
        g1 = [r1, -r2, r2]
        g2 = [-zeta * r2, zeta * r1, -zeta * r1]
        g3 = [F.zero(), -delta, -delta]
    elif pos == 5:
        g1 = [r1, -r2, r2]
        g2 = [zeta * r2, -zeta * r1, zeta * r1]
        g3 = [F.zero(), delta, delta]

    return g1, g2, g3


def compute_superglue_coefficients(X_4: AVX_4x2x1, Z_4: AVX_4x2x1):
    # Compute the positions of each element
    pos_1, pos_2 = compute_symmetric_position(X_4, Z_4)

    # TODO: do this in parallel for both points, just get it working
    # for now...
    x1, x2, x3, x4 = X_4
    z1, z2, z3, z4 = Z_4
    if pos_1 == 0 or pos_1 == 1:
        r1, r2, delta_1 = compute_matrix_coefficients(x2, z2)
    else:
        r1, r2, delta_1 = compute_matrix_coefficients(x1, z1)

    if pos_2 == 0 or pos_2 == 1:
        r3, r4, delta_2 = compute_matrix_coefficients(x4, z4)
    else:
        r3, r4, delta_2 = compute_matrix_coefficients(x3, z3)

    # TODO: this can be done in parallel, but there is no arithmetic, only
    # memory management
    g0_1, g1_1, g2_1 = compute_symmetric_elements(r1, r2, delta_1, pos_1)
    g0_2, g1_2, g2_2 = compute_symmetric_elements(r3, r4, delta_2, pos_2)

    # Combined projective factor and position
    delta = delta_1 * delta_2
    pos = 3 * (pos_1 >> 1) + (pos_2 >> 1)

    # Handle special cases for positions
    superglue_case = 0
    match pos:
        case 0:
            superglue_case = 1
        case 4:
            superglue_case = 2
        case 8:
            superglue_case = 3

    # Init first values
    a = delta
    b = 0
    c = 0
    d = 0

    # T1
    a += g0_1[0] * g0_2[0]
    b += g0_1[0] * g0_2[1]
    c += g0_1[1] * g0_2[0]
    d += g0_1[1] * g0_2[1]

    # T2
    a += g1_1[0] * g1_2[0]
    b += g1_1[0] * g1_2[1]
    c += g1_1[1] * g1_2[0]
    d += g1_1[1] * g1_2[1]

    # T1+T2
    a -= g2_1[0] * g2_2[0]
    b -= g2_1[0] * g2_2[1]
    c -= g2_1[1] * g2_2[0]
    d -= g2_1[1] * g2_2[1]

    # Now we act by (0, Q2)
    a1 = delta_1 * (g1_2[0] * a + g1_2[2] * b)
    if superglue_case == 1 or superglue_case == 3:
        sgl_val = delta_1 * (g1_2[1] * a - g1_2[0] * b)

    # Now we act by (P1, 0)
    a2 = delta_2 * (g0_1[0] * a + g0_1[2] * c)
    if superglue_case == 2:
        sgl_val = delta_2 * (g0_1[0] * b + g0_1[2] * d)

    # Projective factor
    a *= delta
    b *= delta
    c *= delta
    d *= delta

    # TODO: refactor
    dual_theta_null_square = [0, a1**2, a2**2, a**2]
    dual_theta_null_square[0] = dual_theta_null_square[1] + dual_theta_null_square[2]
    dual_theta_null_square[1] = dual_theta_null_square[3] - dual_theta_null_square[1]
    dual_theta_null_square[2] = dual_theta_null_square[3] - dual_theta_null_square[2]
    dual_theta_null_square[3] = 0

    match superglue_case:
        case 0:
            tmp1 = a * b
            tmp2 = c * d
            element_01 = [tmp1 + tmp2, tmp1 - tmp2]

            sign_flip = False

            tmp3 = a * c
            tmp2 = b * d
            element_02 = [tmp3 + tmp2, tmp3 - tmp2]

            tmp1 = a * d
            tmp3 = b * c
            element_03 = [tmp1 + tmp3, tmp1 - tmp3]
        case 1:
            tmp1 = a * b
            tmp2 = a1 * sgl_val
            element_01 = [tmp1 + tmp2, tmp1 - tmp2]

            tmp3 = a * c
            tmp1 -= tmp3
            sign_flip = tmp1 != 0

            # Not needed for case 1, 2, 3
            element_02, element_03 = None, None

        case 2:
            tmp1 = a * b
            tmp2 = a2 * sgl_val
            element_01 = [tmp1 + tmp2, tmp1 - tmp2]

            tmp3 = a * c
            tmp1 -= tmp3
            sign_flip = tmp1 != 0

            # Not needed for case 1, 2, 3
            element_02, element_03 = None, None

        case 3:
            tmp1 = a * b
            tmp2 = a1 * sgl_val
            element_01 = [tmp1 + tmp2, tmp1 - tmp2]

            tmp3 = a * c
            tmp1 -= tmp3
            sign_flip = tmp1 != 0

            # Not needed for case 1, 2, 3
            element_02, element_03 = None, None

    return pos, sign_flip, dual_theta_null_square, element_01, element_02, element_03


def superglue_evaluaton(
    barycentric_1,
    barycentric_2,
    position,
    sign_flip,
    dual_theta_null_square,
    element_01,
    element_02,
    element_03,
):
    tmp3 = barycentric_1[2]
    if sign_flip:
        tmp3 = -tmp3

    uuvv_p_ww1 = barycentric_1[0] + barycentric_1[1]
    tmp1 = barycentric_1[0] - barycentric_1[1]
    uuvv_p_ww1 = uuvv_p_ww1 * tmp1

    uuvv_p_ww2 = barycentric_2[0] + barycentric_2[1]
    tmp1 = barycentric_2[0] - barycentric_2[1]
    uuvv_p_ww2 = uuvv_p_ww2 * tmp1

    tmp1 = barycentric_1[2] ** 2
    tmp2 = barycentric_2[2] ** 2

    uuvv_m_ww1 = uuvv_p_ww1 - tmp1
    uuvv_p_ww1 = uuvv_p_ww1 + tmp1
    uuvv_m_ww2 = uuvv_p_ww2 - tmp2
    uuvv_p_ww2 = uuvv_p_ww2 + tmp2

    uw1 = barycentric_1[0] * tmp3
    uw1 = uw1 + uw1
    uw2 = barycentric_2[0] * barycentric_2[2]
    uw2 = uw2 + uw2

    vw12 = barycentric_1[1] * barycentric_2[1]
    tmp1 = tmp3 * barycentric_2[2]
    vw12 = vw12 * tmp1
    vw12 = vw12 + vw12
    vw12 = vw12 + vw12

    match position:
        case 1:
            # Compute first element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[0]
            tmp2 = uw2 * element_01[0]
            tmp1 = tmp1 + tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_02[0]
            tmp2 = uw2 * element_03[0]
            tmp1 = tmp1 + tmp2
            tmp1 = uw1 * tmp1
            X = tmp3 + tmp1

            # Compute second element
            tmp1 = uuvv_p_ww1 * dual_theta_null_square[1]
            tmp2 = uw1 * element_02[1]
            tmp1 = tmp1 * uuvv_m_ww2
            tmp2 = tmp2 * uuvv_m_ww2
            tmp1 = tmp1 + tmp2
            Y = 0 + tmp1

            # Compute third element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[2]
            tmp2 = uw2 * element_01[1]
            tmp1 = tmp1 * uuvv_m_ww1
            tmp2 = tmp2 * uuvv_m_ww1
            Z = tmp1 + tmp2

            # Compute the last point.
            T = vw12 * element_03[1]

            return (X, Y, Z, T)

        case 2:
            # Compute second element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[1]
            tmp2 = uw2 * element_01[0]
            tmp1 = tmp1 + tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_02[0]
            tmp2 = uw2 * element_03[0]
            tmp1 = tmp1 + tmp2
            tmp1 = uw1 * tmp1
            Y = tmp3 + tmp1

            # Compute first element
            tmp1 = uuvv_p_ww1 * dual_theta_null_square[0]
            tmp2 = uw1 * element_02[1]

            tmp1 = tmp1 * uuvv_m_ww2
            tmp2 = tmp2 * uuvv_m_ww2
            tmp1 = tmp1 + tmp2
            X = 0 + tmp1

            # Compute third element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[2]
            tmp2 = uw2 * element_01[1]
            tmp1 = tmp1 * uuvv_m_ww1
            tmp2 = tmp2 * uuvv_m_ww1
            Z = tmp1 + tmp2

            # Compute the last point.
            T = vw12 * element_03[1]

            return (X, Y, Z, T)

        case 3:
            # Compute first element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[0]
            tmp2 = uw2 * element_01[0]
            tmp1 = tmp1 + tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_02[0]
            tmp2 = uw2 * element_03[0]
            tmp1 = tmp1 + tmp2
            tmp1 = uw1 * tmp1
            X = tmp3 + tmp1

            # Compute third element
            tmp1 = uuvv_p_ww1 * dual_theta_null_square[2]
            tmp2 = uw1 * element_02[1]
            tmp1 = tmp1 * uuvv_m_ww2
            tmp2 = tmp2 * uuvv_m_ww2
            tmp1 = tmp1 + tmp2
            Z = 0 + tmp1

            # Compute second element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[1]
            tmp2 = uw2 * element_01[1]
            tmp1 = tmp1 * uuvv_m_ww1
            tmp2 = tmp2 * uuvv_m_ww1
            Y = tmp1 + tmp2

            # Compute the last point.
            T = vw12 * element_03[1]

            return (X, Y, Z, T)

        case 5:

            # Compute third element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[2]
            tmp2 = uw2 * element_01[0]
            tmp1 = tmp1 + tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_02[0]
            tmp2 = uw2 * element_03[0]
            tmp1 = tmp1 + tmp2
            tmp1 = uw1 * tmp1
            Z = tmp3 + tmp1

            # Compute first element
            tmp1 = uuvv_p_ww1 * dual_theta_null_square[0]
            tmp2 = uw1 * element_02[1]
            tmp1 = tmp1 * uuvv_m_ww2
            tmp2 = tmp2 * uuvv_m_ww2
            tmp1 = tmp1 + tmp2
            X = 0 + tmp1

            # Compute second element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[1]
            tmp2 = uw2 * element_01[1]
            tmp1 = tmp1 * uuvv_m_ww1
            tmp2 = tmp2 * uuvv_m_ww1
            Y = tmp1 + tmp2

            # Compute the last point.
            T = vw12 * element_03[1]

            return (X, Y, Z, T)

        case 6:

            # Compute second element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[1]
            tmp2 = uw2 * element_01[0]
            tmp1 = tmp1 + tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_02[0]
            tmp2 = uw2 * element_03[0]
            tmp1 = tmp1 + tmp2
            tmp1 = uw1 * tmp1
            Y = tmp3 + tmp1

            # Compute third element
            tmp1 = uuvv_p_ww1 * dual_theta_null_square[2]
            tmp2 = uw1 * element_02[1]
            tmp1 = tmp1 * uuvv_m_ww2
            tmp2 = tmp2 * uuvv_m_ww2
            tmp1 = tmp1 + tmp2
            Z = 0 + tmp1

            # Compute first element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[0]
            tmp2 = uw2 * element_01[1]
            tmp1 = tmp1 * uuvv_m_ww1
            tmp2 = tmp2 * uuvv_m_ww1
            X = tmp1 + tmp2

            # Compute the last point.
            T = vw12 * element_03[1]

            return (X, Y, Z, T)

        case 7:

            # Compute third element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[2]
            tmp2 = uw2 * element_01[0]
            tmp1 = tmp1 + tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_02[0]
            tmp2 = uw2 * element_03[0]
            tmp1 = tmp1 + tmp2
            tmp1 = uw1 * tmp1
            Z = tmp3 + tmp1

            # Compute second element
            tmp1 = uuvv_p_ww1 * dual_theta_null_square[1]
            tmp2 = uw1 * element_02[1]
            tmp1 = tmp1 * uuvv_m_ww2
            tmp2 = tmp2 * uuvv_m_ww2
            tmp1 = tmp1 + tmp2
            Y = 0 + tmp1

            # Compute first element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[0]
            tmp2 = uw2 * element_01[1]
            tmp1 = tmp1 * uuvv_m_ww1
            tmp2 = tmp2 * uuvv_m_ww1
            X = tmp1 + tmp2

            # Compute the last point.
            T = vw12 * element_03[1]

            return (X, Y, Z, T)

        case 0:
            # Compute first element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[0]
            tmp2 = uw2 * element_01[0]
            tmp1 = tmp1 + tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_01[1]
            tmp2 = uw2 * dual_theta_null_square[1]
            tmp1 = tmp1 + tmp2
            tmp1 = uw1 * tmp1
            X = tmp3 + tmp1

            # Compute second element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[1]
            tmp2 = uw2 * element_01[1]
            tmp1 = tmp1 + tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_01[0]
            tmp2 = uw2 * dual_theta_null_square[0]
            tmp1 = tmp1 + tmp2
            tmp1 = uw1 * tmp1
            Y = tmp3 + tmp1

            # Compute third element
            tmp1 = uuvv_m_ww2 * uuvv_m_ww1
            Z = tmp1 * dual_theta_null_square[2]

            # Compute the last point.
            T = vw12 * dual_theta_null_square[2]

            return (X, Y, Z, T)

        case 4:

            # Compute first element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[0]
            tmp2 = uw2 * element_01[0]
            tmp1 += tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_01[1]
            tmp2 = uw2 * dual_theta_null_square[2]
            tmp1 += tmp2
            tmp1 = uw1 * tmp1
            X = tmp3 + tmp1

            # Compute third element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[2]
            tmp2 = uw2 * element_01[1]
            tmp1 += tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_01[0]
            tmp2 = uw2 * dual_theta_null_square[0]
            tmp1 += tmp2
            tmp1 = uw1 * tmp1
            Z = tmp3 + tmp1

            # Compute third element
            tmp1 = uuvv_m_ww2 * uuvv_m_ww1
            Y = tmp1 * dual_theta_null_square[1]

            # Compute the last point.
            T = vw12 * dual_theta_null_square[1]

            return (X, Y, Z, T)

        case 8:
            # Compute second element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[1]
            tmp2 = uw2 * element_01[1]
            tmp1 += tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_01[0]
            tmp2 = uw2 * dual_theta_null_square[2]
            tmp1 += tmp2
            tmp1 = uw1 * tmp1
            Y = tmp3 + tmp1

            # Compute third element
            tmp1 = uuvv_p_ww2 * dual_theta_null_square[2]
            tmp2 = uw2 * element_01[0]
            tmp1 = tmp1 + tmp2
            tmp3 = uuvv_p_ww1 * tmp1

            tmp1 = uuvv_p_ww2 * element_01[1]
            tmp2 = uw2 * dual_theta_null_square[1]
            tmp1 = tmp1 + tmp2
            tmp1 = uw1 * tmp1
            Z = tmp3 + tmp1

            # Compute third element
            tmp1 = uuvv_m_ww2 * uuvv_m_ww1
            X = tmp1 * dual_theta_null_square[0]

            # Compute the last point.
            T = vw12 * dual_theta_null_square[0]

            return (X, Y, Z, T)


def superglue_precomputation(A24_12: AVX_4x2x1, X_8: AVX_4x2x1, Z_8: AVX_4x2x1):
    X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)

    # Double points to recover 4-torsion for change of basis
    X_4, Z_4 = x_double_4x4x2x1(A24_12, X_8, Z_8)

    # Compute all the precomputations needed for superglue
    # position, sign_flip, dual_theta_null_square, element_01, element_02, element_03
    return compute_superglue_coefficients(X_4, Z_4)


def superglue_codomain(X_8: AVX_4x2x1, Z_8: AVX_4x2x1, precomputation):
    # TODO: avoid computing this twice
    X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)

    # Unpack the insane precomputation
    position, sign_flip, dual_theta_null_square, element_01, element_02, element_03 = (
        precomputation
    )

    x1, x2, x3, x4 = X_8
    z1, z2, z3, z4 = Z_8

    bary_P1 = [x1, 0, z1]
    bary_P2 = [x3, 0, z3]
    bary_Q1 = [x2, 0, z2]
    bary_Q2 = [x4, 0, z4]

    xAxByCyD = superglue_evaluaton(
        bary_P1,
        bary_P2,
        position,
        sign_flip,
        dual_theta_null_square,
        element_01,
        element_02,
        element_03,
    )
    zAtBzYtD = superglue_evaluaton(
        bary_Q1,
        bary_Q2,
        position,
        sign_flip,
        dual_theta_null_square,
        element_01,
        element_02,
        element_03,
    )

    # Codomain coefficients
    ABCD = [0, 0, 0, 0]
    ABCD[0] = xAxByCyD[0] * zAtBzYtD[0]
    ABCD[1] = xAxByCyD[1] * zAtBzYtD[0]
    ABCD[2] = xAxByCyD[0] * zAtBzYtD[2]

    alpha_inv = xAxByCyD[1] * zAtBzYtD[2]

    a = xAxByCyD[0] * alpha_inv
    b = xAxByCyD[2] * ABCD[1]

    codomain = hadamard(AVX_4x2x1(*ABCD))

    return codomain, (a, b)


def superglue_eval_pair(
    A_12: AVX_4x2x1, P: tuple[AVX_4x2x1], T8: tuple[AVX_4x2x1], a, b, precomputation
):
    U, V, W = four_add_components(A_12, P, T8)
    u1, u3, u2, u4 = U
    v1, v3, v2, v4 = V
    w1, w3, w2, w4 = W

    # Unpack the insane precomputation
    position, sign_flip, dual_theta_null_square, element_01, element_02, element_03 = (
        precomputation
    )

    XaYaZbWb = superglue_evaluaton(
        (u1, v1, w1),
        (u2, v2, w2),
        position,
        sign_flip,
        dual_theta_null_square,
        element_01,
        element_02,
        element_03,
    )
    xyzt_1 = [XaYaZbWb[0] * b, XaYaZbWb[1] * b, XaYaZbWb[2] * a, XaYaZbWb[3] * a]
    xyzt_1 = hadamard(AVX_4x2x1(*xyzt_1))

    XaYaZbWb = superglue_evaluaton(
        (u3, v3, w3),
        (u4, v4, w4),
        position,
        sign_flip,
        dual_theta_null_square,
        element_01,
        element_02,
        element_03,
    )
    xyzt_2 = [XaYaZbWb[0] * b, XaYaZbWb[1] * b, XaYaZbWb[2] * a, XaYaZbWb[3] * a]
    xyzt_2 = hadamard(AVX_4x2x1(*xyzt_2))

    return avx_421_to_811(xyzt_1, xyzt_2)
