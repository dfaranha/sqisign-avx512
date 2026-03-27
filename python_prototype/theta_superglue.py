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
    """
    4M + 8a
    """
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

    # TODO: here we could check 4 of the 6 conditions
    # and exit early if we want, saving 1 4 way mul
    # and 1 4-way add

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

    def find_pos(s, r, s_other, r_other, u_plus, u_minus):
        if s == 0:
            return 0
        elif r == 0:
            return 1
        elif s_other == 0:
            return 2
        elif r_other == 0:
            return 3
        elif u_plus == 0:
            return 4
        elif u_minus == 0:
            return 5
        else:
            raise ValueError("No vanishing coordinate found in 4-torsion")

    pos_1 = find_pos(s1, r1, s2, r2, u1, u2)
    pos_2 = find_pos(s3, r3, s4, r4, u3, u4)

    return pos_1, pos_2


def compute_matrix_coefficients(x1, z1, x2, z2):
    """
    2M + 4S + 6a
    """
    # 2M + 4S parallel computations
    s1 = x1**2
    s2 = z1**2
    s3 = x2**2
    s4 = z2**2

    r2 = x1 * z1
    r4 = x2 * z2

    # 4s + 2s parallel computations
    r1 = s1 + s2
    r2 = r2 + r2
    r3 = s3 + s4
    r4 = r4 + r4

    delta_1 = s1 - s2
    delta_2 = s3 - s4

    return ((r1, r2, delta_1), (r3, r4, delta_2))


def compute_symmetric_elements(r1, r2, delta, pos):
    """
    Almost no arithmetic here, just memory management and a sign flip
    """
    # TODO: mul by zeta is just sign flips and memory...
    F = r1.parent()
    zero = F.zero()
    zeta = F.gen()
    ir1, ir2 = zeta * r1, zeta * r2

    # TODO: pos 3, 4 and 5, 6 are just 1, 2 but ordered differently
    if pos == 0:
        g1 = [zero, delta, delta]
        g2 = [r1, -r2, r2]
        g3 = [ir2, -ir1, ir1]
    elif pos == 1:
        g1 = [zero, -delta, -delta]
        g2 = [r1, -r2, r2]
        g3 = [-ir2, ir1, -ir1]
    elif pos == 2:
        g1 = [r1, -r2, r2]
        g2 = [zero, delta, delta]
        g3 = [-ir2, ir1, -ir1]
    elif pos == 3:
        g1 = [r1, -r2, r2]
        g2 = [zero, -delta, -delta]
        g3 = [ir2, -ir1, ir1]
    elif pos == 4:
        g1 = [r1, -r2, r2]
        g2 = [-ir2, ir1, -ir1]
        g3 = [zero, -delta, -delta]
    elif pos == 5:
        g1 = [r1, -r2, r2]
        g2 = [ir2, -ir1, ir1]
        g3 = [zero, delta, delta]

    return g1, g2, g3


def compute_superglue_coefficients(X_4: AVX_4x2x1, Z_4: AVX_4x2x1):
    # Compute the positions of each element
    pos_1, pos_2 = compute_symmetric_position(X_4, Z_4)

    # Memory management for what we send into compute_matrix_coefficients
    # based on pos_1 and pos_2
    x1, x2, x3, x4 = X_4
    z1, z2, z3, z4 = Z_4
    if pos_1 == 0 or pos_1 == 1:
        X1, Z1 = x2, z2
    else:
        X1, Z1 = x1, z1
    if pos_2 == 0 or pos_2 == 1:
        X2, Z2 = x4, z4
    else:
        X2, Z2 = x3, z3
    ((r1, r2, delta_1), (r3, r4, delta_2)) = compute_matrix_coefficients(X1, Z1, X2, Z2)

    # TODO: this can be done in parallel, but there is no arithmetic, only
    # memory management
    g0_1, g1_1, g2_1 = compute_symmetric_elements(r1, r2, delta_1, pos_1)
    g0_2, g1_2, g2_2 = compute_symmetric_elements(r3, r4, delta_2, pos_2)

    # Combined projective factor and position
    delta = delta_1 * delta_2

    # Handle special cases for positions
    position = 3 * (pos_1 >> 1) + (pos_2 >> 1)
    superglue_case = {0: 1, 4: 2, 8: 3}.get(position, 0)

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

    t0 = g1_2[0] * a
    t1 = g1_2[2] * b
    t2 = g0_1[0] * a
    t3 = g0_1[2] * c

    # 2 more M, annoying can be paired with optional 2 below
    # Now we act by (0, Q2)
    a1 = delta_1 * (t0 + t1)

    # Now we act by (P1, 0)
    a2 = delta_2 * (t2 + t3)

    # Special value needed for case 1, 2, 3
    # In SQIsign we are only in case 0 or 2
    sgl_val = 0
    if superglue_case in [1, 3]:  # Never happens in sqisign
        sgl_val = delta_1 * (g1_2[1] * a - g1_2[0] * b)
    elif superglue_case == 2:
        # 3 more M... annoying
        sgl_val = delta_2 * (g0_1[0] * b + g0_1[2] * d)

    # Projective factor, 4M is nice
    a *= delta
    b *= delta
    c *= delta
    d *= delta

    # Compute dual_theta_null_square for later
    a1sq, a2sq, asq = a1**2, a2**2, a**2
    dual_theta_null_square = [a1sq + a2sq, asq - a1sq, asq - a2sq, 0]

    match superglue_case:
        case 0:
            # Standard case: all four coordinates are nonzero
            t0 = a * b
            t1 = c * d
            t2 = a * c
            t3 = b * d
            t4 = a * d
            t5 = b * c

            element_01 = [t0 + t1, t0 - t1]
            element_02 = [t2 + t3, t2 - t3]
            element_03 = [t4 + t5, t4 - t5]
            sign_flip = False

        case 1 | 2 | 3:
            # Special case: one coordinate vanishes, sign_flip encodes
            # which square root branch to take
            alpha = a1 if superglue_case != 2 else a2

            t0 = a * b
            t1 = alpha * sgl_val
            t2 = a * b
            t3 = a * c

            element_01 = [t0 + t1, t0 - t1]
            sign_flip = (t2 - t3) != 0
            # element_02 and element_03 are not used in these cases
            element_02 = element_03 = None

    return (
        position,
        sign_flip,
        dual_theta_null_square,
        element_01,
        element_02,
        element_03,
    )


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
    # NOTE: this is always called in pairs, so we can make this
    # a pair function and do 8 way instead of 4 way as seen below.

    tmp3 = barycentric_1[2]
    if sign_flip:
        tmp3 = -tmp3

    # 4 adds
    uuvv_p_ww1 = barycentric_1[0] + barycentric_1[1]
    tmp1 = barycentric_1[0] - barycentric_1[1]
    uuvv_p_ww2 = barycentric_2[0] + barycentric_2[1]
    tmp2 = barycentric_2[0] - barycentric_2[1]

    # 2M + 2S
    uuvv_p_ww1 = uuvv_p_ww1 * tmp1
    uuvv_p_ww2 = uuvv_p_ww2 * tmp2
    tmp1 = barycentric_1[2] ** 2
    tmp2 = barycentric_2[2] ** 2

    # 4 adds
    uuvv_m_ww1 = uuvv_p_ww1 - tmp1
    uuvv_p_ww1 = uuvv_p_ww1 + tmp1
    uuvv_m_ww2 = uuvv_p_ww2 - tmp2
    uuvv_p_ww2 = uuvv_p_ww2 + tmp2

    # 4M
    uw1 = barycentric_1[0] * tmp3
    uw2 = barycentric_2[0] * barycentric_2[2]
    vw12 = barycentric_1[1] * barycentric_2[1]
    tmp1 = tmp3 * barycentric_2[2]

    vw12 = vw12 * tmp1  # Annoying one on its own

    uw1 = uw1 + uw1
    uw2 = uw2 + uw2
    vw12 = vw12 + vw12
    vw12 = vw12 + vw12

    # WARNING No refactoring from max has happened below.
    # SQIsign has only position 1, 4, 7 (to save time when writing code)
    match position:
        case 1:
            # 8 way multiplications with precomputations
            t0 = uuvv_p_ww2 * dual_theta_null_square[0]
            t1 = uw2 * element_01[0]
            t4 = uuvv_p_ww2 * element_02[0]
            t5 = uw2 * element_03[0]
            t8 = uuvv_p_ww1 * dual_theta_null_square[1]
            t9 = uw1 * element_02[1]
            t12 = uuvv_p_ww2 * dual_theta_null_square[2]
            t13 = uw2 * element_01[1]

            # Two adds
            t2 = t0 + t1
            t6 = t4 + t5

            # 7 multiplications
            t3 = uuvv_p_ww1 * t2
            t7 = uw1 * t6
            t10 = t8 * uuvv_m_ww2
            t11 = t9 * uuvv_m_ww2
            t14 = t12 * uuvv_m_ww1
            t15 = t13 * uuvv_m_ww1
            T = vw12 * element_03[1]

            # three adds
            X = t3 + t7
            Y = t10 + t11
            Z = t14 + t15

            return (X, Y, Z, T)

        case 4:
            # 8 way mul
            t0 = uuvv_p_ww2 * dual_theta_null_square[0]
            t1 = uw2 * element_01[0]
            t4 = uuvv_p_ww2 * element_01[1]
            t5 = uw2 * dual_theta_null_square[2]
            t8 = uuvv_p_ww2 * dual_theta_null_square[2]
            t9 = uw2 * element_01[1]
            t12 = uuvv_p_ww2 * element_01[0]
            t13 = uw2 * dual_theta_null_square[0]

            # 4 adds
            t2 = t0 + t1
            t6 = t4 + t5
            t10 = t8 + t9
            t14 = t12 + t13

            # 6 muls
            t3 = uuvv_p_ww1 * t2
            t7 = uw1 * t6
            t11 = uuvv_p_ww1 * t10
            t15 = uw1 * t14
            t16 = uuvv_m_ww2 * uuvv_m_ww1
            T = vw12 * dual_theta_null_square[1]

            # 3 adds
            X = t3 + t7
            Z = t11 + t15

            # 1 mul left over, can we refactor this?
            Y = t16 * dual_theta_null_square[1]

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

        # All other cases wont be touched by SQIsign

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


def superglue_codomain(
    A24_12: AVX_4x2x1,
    X_8: AVX_4x2x1,
    Z_8: AVX_4x2x1,
):
    # TODO: avoid computing this twice
    X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)

    # Double points to recover 4-torsion for change of basis
    X_4, Z_4 = x_double_4x4x2x1(A24_12, X_8, Z_8)

    # Unpack the insane precomputation
    precomputation = compute_superglue_coefficients(X_4, Z_4)
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

    return codomain, (a, b), precomputation


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
