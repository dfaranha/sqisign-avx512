from avx_types import AVX_2x4x1, AVX_4x2x1, AVX_8x1x1
from avx_arithmetic import (
    avx_241_add,
    avx_241_mul,
    avx_241_add_sub,
    avx_421_sqr,
    avx_421_mul,
    avx_421_add,
    avx_421_sub,
    avx_421_neg,
    avx_811_mul,
    avx_811_sub,
    avx_421_pair_mul,
    avx_811_sqr,
    avx_811_add_sub,
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


def compute_symmetric_position(X_4: AVX_4x2x1, Z_4: AVX_4x2x1) -> tuple[int]:
    """
    Compute the position from the four torsion. In SQIsign, the position of
    E2[4] is either 2 or 3, but for E1[4] the position can be any of the six
    possibilities.
    """
    x1, x2, x3, x4 = X_4
    z1, z2, z3, z4 = Z_4

    # The positions we will find
    pos_1 = None
    pos_2 = None

    # 8a
    T0 = AVX_8x1x1(x1, x2, x3, x4, x1, x2, x3, x4)
    T1 = AVX_8x1x1(z1, z2, z3, z4, z1, z2, z3, z4)
    T2 = avx_811_add_sub(T0, T1)
    r1, r2, r3, r4, s1, s2, s3, s4 = T2

    def find_pos_early(s, r, s_other, r_other):
        if s == 0:
            return 0
        elif r == 0:
            return 1
        elif s_other == 0:
            return 2
        elif r_other == 0:
            return 3
        else:
            return None

    # For SQIsign, pos_2 is always 2 or 3, but pos_1
    # is unbounded. We can return early here if we're
    # in the lucky 2/3 cases
    pos_1 = find_pos_early(s1, r1, s2, r2)
    pos_2 = find_pos_early(s3, r3, s4, r4)
    assert pos_2 in [2, 3]

    if pos_1 is not None:
        return pos_1, pos_2

    # If we've reached here, we need to do extra work to find pos_1
    # at a cost of 2M and 2a
    zeta = x1.parent().gen()
    ir1 = zeta * r1

    T0 = AVX_2x4x1(ir1, r2)
    T1 = AVX_2x4x1(s2, s1)
    T2 = avx_241_mul(T0, T1)

    t1, t2 = T2
    T0 = AVX_2x4x1(t1, t1)
    T1 = AVX_2x4x1(t2, t2)
    u1, u2 = avx_241_add_sub(T0, T1)

    # Find the position in the case where neither P or Q is above (0 : 1)
    if u1 == 0:
        pos_1 = 4
    elif u2 == 0:
        pos_1 = 5
    else:
        raise ValueError("No vanishing coordinate found in 4-torsion")

    return pos_1, pos_2


def compute_matrix_coefficients(four_torsion: AVX_4x2x1) -> AVX_8x1x1:
    """
    2M + 4S + 6a
    """
    x1, z1, x2, z2 = four_torsion

    # 8M (2M wasted)
    T0 = AVX_8x1x1(x1, z1, x2, z2, x1, x2, 0, 0)
    T1 = AVX_8x1x1(x1, z1, x2, z2, z1, z2, 0, 0)
    T2 = avx_811_mul(T0, T1)
    s1, s2, s3, s4, r2, r4, _, _ = T2

    # 8a (2a wasted)
    T0 = AVX_8x1x1(s1, r2, s3, r4, s1, s3, 0, 0)
    T1 = AVX_8x1x1(s2, r2, s4, r4, s2, s4, 0, 0)
    return avx_811_add_sub(T0, T1)


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
    """
    35M + 7S + 36a
    """
    # Compute the positions of each element
    # 2M + 10a
    pos_1, pos_2 = compute_symmetric_position(X_4, Z_4)

    # Memory management for what we send into compute_matrix_coefficients
    # based on pos_1 and pos_2
    assert pos_2 in [2, 3]
    x1, x2, x3, _ = X_4
    z1, z2, z3, _ = Z_4
    if pos_1 == 0 or pos_1 == 1:
        four_torsion = AVX_4x2x1(x2, z2, x3, z3)
    else:
        four_torsion = AVX_4x2x1(x1, z1, x3, z3)

    # Compute the unique elements which we use to file the symmetric elements
    # 2M + 4S + 6a
    coefficients = compute_matrix_coefficients(four_torsion)

    # TODO: this can be done in parallel, but there is no arithmetic, only
    # memory management so maybe it's not worth it
    r1, r2, r3, r4, delta_1, delta_2, _, _ = coefficients
    g0_1, g1_1, g2_1 = compute_symmetric_elements(r1, r2, delta_1, pos_1)
    g0_2, g1_2, g2_2 = compute_symmetric_elements(r3, r4, delta_2, pos_2)

    # Generically there's 9 elements here for the Q basis, but we actually only need 7
    assert g1_2[0] == 0
    assert g1_2[1] == g1_2[2]

    # Handle special cases for positions
    position = 3 * (pos_1 >> 1) + (pos_2 >> 1)
    assert position in [1, 4, 7]

    # 8M
    # a = delta + g1_1[0] * g1_2[0], g1_2[0] is 0 and we skip this to save a way mul
    a = g0_1[0] * g0_2[0]
    b = g0_1[0] * g0_2[1]
    c = g0_1[1] * g0_2[0]
    d = g0_1[1] * g0_2[1]
    delta = delta_1 * delta_2
    t0 = g1_1[0] * g1_2[1]
    t1 = g1_1[1] * g1_2[0]  # g1_2[0] is 0, we could skip this
    t2 = g1_1[1] * g1_2[1]

    # 4a
    a += delta
    b += t0
    c += t1
    d += t2

    # T1+T2
    # 4M
    r0 = g2_1[0] * g2_2[0]
    r1 = g2_1[0] * g2_2[1]
    r2 = g2_1[1] * g2_2[0]
    r3 = g2_1[1] * g2_2[1]

    # 4a
    a -= r0
    b -= r1
    c -= r2
    d -= r3

    # 4M
    t0 = g1_2[0] * a  # g1_2[0] is 0, we could skip this
    t1 = g1_2[2] * b
    t2 = g0_1[0] * a
    t3 = g0_1[2] * c

    # 2a
    t4 = t0 + t1
    t5 = t2 + t3

    # These first two are optional, but fit in a 4-way
    # 4M
    t6 = g0_1[0] * b
    t7 = g0_1[2] * d
    a1 = delta_1 * t4
    a2 = delta_2 * t5

    # This only needs to be done 1/3 of the time
    sgl_val = 0
    if position == 4:
        # Another 1a + 1M
        t8 = t6 + t7
        sgl_val = delta_2 * t8

    # 4M
    a *= delta
    b *= delta
    c *= delta
    d *= delta

    # Compute dual_theta_null_square for later
    # 3S
    a1sq = a1**2
    a2sq = a2**2
    asq = a**2

    # 3a
    A_inv = a1sq + a2sq
    B_inv = asq - a1sq
    C_inv = asq - a2sq

    dual_theta_null_square = [A_inv, B_inv, C_inv, 0]

    match position:
        case 1 | 7:
            # Standard case: all four coordinates are nonzero
            # 6M + 6a
            t0 = a * b
            t1 = c * d
            t2 = a * c
            t3 = b * d
            t4 = a * d
            t5 = b * c

            e0 = t0 + t1
            e2 = t2 + t3
            e4 = t4 + t5
            e1 = t0 - t1
            e3 = t2 - t3
            e5 = t4 - t5

            elements = [e0, e1, e2, e3, e4, e5]
            sign_flip = False

        case 4:
            # Special case: one coordinate vanishes, sign_flip encodes
            # which square root branch to take
            # 4M + 3a
            t0 = a * b
            t1 = a2 * sgl_val
            t2 = a * b
            t3 = a * c

            e0 = t0 + t1
            e1 = t0 - t1
            cc = t2 - t3

            elements = [e0, e1, 0, 0, 0, 0]
            sign_flip = cc != 0

        case _:
            raise ValueError("unsupported case for SQIsign verification")

    return (
        position,
        sign_flip,
        dual_theta_null_square,
        elements,
    )


def superglue_evaluation_on_position(data, position, dual_theta_null_square, elements):
    s0, s1, s2, s3, s4, s5, s6, s7 = data
    A_inv, B_inv, C_inv, _ = dual_theta_null_square

    # SQIsign has only position 1, 4, 7 (to save time when writing code)
    match position:
        # Special case where we need to handle the zero
        case 4:
            e0, e1, _, _, _, _ = elements

            # 8M
            T0 = AVX_8x1x1(s1, s5, s1, s5, s1, s5, s1, s5)
            T1 = AVX_8x1x1(A_inv, e0, e1, C_inv, C_inv, e1, e0, A_inv)
            T2 = avx_811_mul(T0, T1)

            # 4a
            T0 = AVX_4x2x1(T2[0], T2[2], T2[4], T2[6])
            T1 = AVX_4x2x1(T2[1], T2[3], T2[5], T2[7])
            T2 = avx_421_add(T0, T1)

            # 8M (2 wasted)
            T0 = AVX_8x1x1(s0, s4, s0, s4, s6, s7, 0, 0)
            T1 = AVX_8x1x1(T2[0], T2[1], T2[2], T2[3], B_inv, B_inv, 0, 0)
            T2 = avx_811_mul(T0, T1)

            # 4a
            T0 = AVX_2x4x1(T2[0], T2[2])
            T1 = AVX_2x4x1(T2[1], T2[3])
            T3 = avx_241_add(T0, T1)

            return AVX_4x2x1(T3[0], T2[5], T3[1], T2[4])

        # Cases 1 and 7 are nearly identical
        case 1 | 7:
            e0, e1, e2, e3, e4, e5 = elements

            # 8M
            T0 = AVX_8x1x1(s1, s5, s1, s5, s0, s4, s1, s5)
            # For positions 1, 7 only the role of A_inv and C_inv
            # and then X, Z are reversed.
            if position == 7:
                T1 = AVX_8x1x1(C_inv, e0, e2, e4, B_inv, e3, A_inv, e1)
            else:
                T1 = AVX_8x1x1(A_inv, e0, e2, e4, B_inv, e3, C_inv, e1)
            T2 = avx_811_mul(T0, T1)

            # 4a
            T3 = AVX_2x4x1(T2[0], T2[2])
            T4 = AVX_2x4x1(T2[1], T2[3])
            T5 = avx_241_add(T3, T4)

            # 8M (1 wasted)
            T6 = AVX_8x1x1(s0, s4, s3, s3, s2, s2, s6, 0)
            T7 = AVX_8x1x1(T5[0], T5[1], T2[4], T2[5], T2[6], T2[7], e5, 0)
            T8 = avx_811_mul(T6, T7)

            # 4a (1 wasted)
            T9 = AVX_4x2x1(T8[0], T8[2], T8[4], 0)
            Ta = AVX_4x2x1(T8[1], T8[3], T8[5], 0)
            Tb = avx_421_add(T9, Ta)

            # For positions 1, 7 only the role of A_inv and C_inv
            # and then X, Z are reversed.
            if position == 7:
                return AVX_4x2x1(Tb[2], Tb[1], Tb[0], T8[6])
            else:
                return AVX_4x2x1(Tb[0], Tb[1], Tb[2], T8[6])

        # All other cases wont be touched by SQIsign
        case _:
            raise ValueError("evaluation failed due to unsupported case")


def superglue_evaluaton_pair(
    U: AVX_4x2x1,
    V: AVX_4x2x1,
    W: AVX_4x2x1,
    position: int,
    sign_flip: bool,
    dual_theta_null_square,
    elements,
) -> tuple[AVX_4x2x1]:

    u1, u3, u2, u4 = U
    v1, v3, v2, v4 = V
    w1, w3, w2, w4 = W

    # undo this packing
    b1 = (u1, v1, w1)
    b2 = (u2, v2, w2)
    b3 = (u3, v3, w3)
    b4 = (u4, v4, w4)

    w1 = b1[2]
    w3 = b3[2]
    if sign_flip:
        w1 = -w1
        w3 = -w3

    # 8 adds
    # TODO: for codomain bi[1] is zero
    s0 = b1[0] + b1[1]
    s1 = b2[0] + b2[1]
    t0 = b1[0] - b1[1]
    t1 = b2[0] - b2[1]
    r0 = b3[0] + b3[1]
    r1 = b4[0] + b4[1]
    t5 = b3[0] - b3[1]
    t6 = b4[0] - b4[1]

    # 4M + 4S
    s0 = s0 * t0
    s1 = s1 * t1
    t2 = b1[2] ** 2
    t3 = b2[2] ** 2
    r0 = r0 * t5
    r1 = r1 * t6
    t7 = b3[2] ** 2
    t8 = b4[2] ** 2

    # 8 adds
    s2 = s0 - t2
    s0 = s0 + t2
    s3 = s1 - t3
    s1 = s1 + t3
    r2 = r0 - t7
    r0 = r0 + t7
    r3 = r1 - t8
    r1 = r1 + t8

    # 8M
    # TODO: for codomain bi[1] is zero
    s4 = b1[0] * w1
    s5 = b2[0] * b2[2]
    s6 = b1[1] * b2[1]
    t4 = b2[2] * w1
    r4 = b3[0] * w3
    r5 = b4[0] * b4[2]
    r6 = b3[1] * b4[1]
    t9 = b4[2] * w3

    # 4M (s7, r7 only used for case 4 though)
    s6 = s6 * t4
    s7 = s3 * s2
    r6 = r6 * t9
    r7 = r3 * r2

    # 6 adds
    s4 = s4 + s4
    s5 = s5 + s5
    s6 = s6 + s6
    r4 = r4 + r4
    r5 = r5 + r5
    r6 = r6 + r6

    # 2 adds
    s6 = s6 + s6
    r6 = r6 + r6

    # Package data to send to superglue_evaluation_on_position
    data_1 = s0, s1, s2, s3, s4, s5, s6, s7
    data_2 = r0, r1, r2, r3, r4, r5, r6, r7

    # This could also be batched, but when case is 1 | 7 we do a 8 way then 7 way mul
    # when case is 4 we do an 8 way then 6 way mul, we could only save time with batched
    # additions and the code would be so much harder to read.#
    P1 = superglue_evaluation_on_position(
        data_1, position, dual_theta_null_square, elements
    )
    P2 = superglue_evaluation_on_position(
        data_2, position, dual_theta_null_square, elements
    )

    return P1, P2


def superglue_codomain(
    A24_12: AVX_4x2x1,
    X_8: AVX_4x2x1,
    Z_8: AVX_4x2x1,
):
    """
    19S + 69M + 84a
    """
    # 4S
    X_8, Z_8 = gluing_kernel_to_x_only(X_8, Z_8)

    # Double points to recover 4-torsion for change of basis
    # 8S + 8M + 16a
    X_4, Z_4 = x_double_4x4x2x1(A24_12, X_8, Z_8)

    # 7S + 35M + 36a
    precomputation = compute_superglue_coefficients(X_4, Z_4)

    # Unpack the insane precomputation
    position, sign_flip, dual_theta_null_square, elements = precomputation

    x1, x2, x3, x4 = X_8
    z1, z2, z3, z4 = Z_8

    # Pack the (u, v, w) coordinates directly from the 8 torsion
    U = AVX_4x2x1(x1, x2, x3, x4)
    V = AVX_4x2x1(0, 0, 0, 0)
    W = AVX_4x2x1(z1, z2, z3, z4)

    # 20M + 24a
    A1, A2 = superglue_evaluaton_pair(
        U,
        V,
        W,
        position,
        sign_flip,
        dual_theta_null_square,
        elements,
    )

    # 4M
    T0 = AVX_4x2x1(A1[0], A1[1], A1[0], A1[1])
    T1 = AVX_4x2x1(A2[0], A2[0], A2[2], A2[2])
    T2 = avx_421_mul(T0, T1)

    ABCD = AVX_4x2x1(T2[0], T2[1], T2[2], 0)

    # 2M
    T0 = AVX_2x4x1(A1[0], A1[2])
    T1 = AVX_2x4x1(T2[3], ABCD[1])
    ab = avx_241_mul(T0, T1)

    # 8a
    codomain = hadamard(ABCD)

    return codomain, ab, precomputation


def superglue_eval_pair(
    A_12: AVX_4x2x1, P: tuple[AVX_4x2x1], T8: tuple[AVX_4x2x1], a, b, precomputation
):
    """
    16S + 76M + 68a
    """
    # 16S + 48M + 28a
    U, V, W = four_add_components(A_12, P, T8)

    # Unpack the insane precomputation
    position, sign_flip, dual_theta_null_square, elements = precomputation

    # 20M + 24a
    P1, P2 = superglue_evaluaton_pair(
        U,
        V,
        W,
        position,
        sign_flip,
        dual_theta_null_square,
        elements,
    )

    imP_Q = avx_421_to_811(P1, P2)
    J = AVX_8x1x1(b, b, a, a, b, b, a, a)

    # 8M
    imP_Q = avx_811_mul(imP_Q, J)

    # 16a
    imP_Q = hadamard_pair(imP_Q)
    return imP_Q
