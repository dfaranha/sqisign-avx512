from theta_model.gluing_isogeny import (
    gluing_codomain,
    gluing_eval_special,
    gluing_eval,
    prepare_gluing_kernel,
)
from theta_model.theta_isogeny import (
    generic_codomain_with_eight_torsion,
    generic_codomain_with_four_torsion,
    generic_codomain,
    generic_eval,
)
from theta_model.splitting_isomorphism import (
    splitting_isomorphism_matrix,
    theta_to_product,
    theta_product_point_to_montgomery,
)
from theta_model.utilities import apply_isomorphism
from elliptic_curves.kummer_line import KummerPoint, KummerLine
from elliptic_curves.jacobian_arithmetic import DBL
from elliptic_curves.torsion_basis import lift_basis_to_jacobian


def two_two_isogeny_chain(
    kernel: tuple[tuple[KummerPoint]],
    length: int,
    points: list[tuple[KummerPoint]] = [],
    randomize=False,
):
    """
    Algorithm 8.47 of the SQIsign spec.

    Compute a (2,2)-isogeny chain and optionally evaluate this isogeny on a given list of points

    Input:
        - kernel ((P1, P2), (Q1, Q2)): is represented by two x-only bases xP, xQ, xPQ on curves E1, E2
        - length (int): the length of the (2^n, 2^n)-isogeny chain
        - points [tuple(KummerPoint)]: pairs of KummerPoints on E1 x E2 to evaluate
        - randomize (random_bytes function): if not None, expects a function which returns random bytes
        - verify (bool): whether to perform additional arithmetic checks during isogeny computation
    """
    # Extract out the bases on E_chl and E_aux
    chl_basis, aux_basis = kernel

    # Montgomery coefficients of E_chl and E_aux needed for
    # Jacobian arithmetic
    E1 = chl_basis[0].parent()
    E2 = aux_basis[0].parent()
    (A1, A2) = (E1.a(), E2.a())

    # Lift xP, xQ, xPQ to points on the Jacobian for evaluation
    P_chl, Q_chl = lift_basis_to_jacobian(chl_basis, A1)
    P_aux, Q_aux = lift_basis_to_jacobian(aux_basis, A2)

    # Init strategy arrays
    space = length.bit_length() + 1
    strat_points = [(((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0)))] * space
    orders = [0] * space

    # Set first values of arrays
    strat_points[0] = ((P_chl, P_aux), (Q_chl, Q_aux))
    orders[0] = length - 2
    k = 0

    # Step One: Perform doubling on the Jacobian points, and use 8-torsion
    #           points to compute the gluing codomain. Push all stored
    #           Jacobian points through the gluing isogeny to obtain ThetaPoints

    # First build a list of doubled points on the Jacobian
    while orders[k] != 1:
        k = k + 1
        # Because gluing eval is expensive we modify how the optimized chain is computed
        if orders[k - 1] >= 16:
            n = orders[k - 1] // 2
        else:
            n = orders[k - 1] - 1

        # Double all four points n times
        ((P1, P2), (Q1, Q2)) = strat_points[k - 1]
        for _ in range(n):
            P1 = DBL(P1, A1)
            P2 = DBL(P2, A2)
            Q1 = DBL(Q1, A1)
            Q2 = DBL(Q2, A2)
        strat_points[k] = ((P1, P2), (Q1, Q2))
        orders[k] = orders[k - 1] - n

    # Compute the gluing isogeny
    # The kernel for codomain computation only needs data on the KummerLine
    # but for evaluation we need the points as Jacobian points
    gluing_kernel = strat_points[k]
    gluing_kernel_x_only = prepare_gluing_kernel(E1, E2, gluing_kernel)
    I, OA, J, N = gluing_codomain(gluing_kernel_x_only)

    # Push Jacobian points through the isogeny. Assumes input points are of
    # the form (P, 0) or (0, Q)
    points = [gluing_eval_special(P, I, N) for P in points]

    # Push the kernel through the isogeny
    T1_8 = gluing_kernel[0]
    for i in range(k):
        P, Q = strat_points[i]
        P_im = gluing_eval((A1, A2), P, T1_8, J, N)
        Q_im = gluing_eval((A1, A2), Q, T1_8, J, N)
        strat_points[i] = (P_im, Q_im)
        orders[i] = orders[i] - 1

    #
    # Main loop: compute (2,2) isogenies in the theta model with balanced strategy
    #            we can do all but the final two steps using the 8-torsion above
    k = k - 1
    j = 0
    while k >= 0 and orders[k] != 0:
        j = j + 1
        while orders[k] != 1:
            k = k + 1
            n = orders[k - 1] // 2
            # Perform n doublings of the kernel points (T1, T2)
            T1, T2 = strat_points[k - 1]
            for _ in range(n):
                T1 = OA.double(T1)
                T2 = OA.double(T2)
            strat_points[k] = (T1, T2)
            orders[k] = orders[k - 1] - n

        # Compute codomain with kernel in the 8-torsion
        I, OA = generic_codomain_with_eight_torsion(*strat_points[k])

        # Push the evaluation points through the isogeny
        points = [generic_eval(P, I) for P in points]

        # Push the kernel elements (T1, T2) through the isogeny
        assert orders[k] == 1
        for i in range(k):
            T1 = generic_eval(strat_points[i][0], I)
            T2 = generic_eval(strat_points[i][1], I)
            strat_points[i] = (T1, T2)
            orders[i] = orders[i] - 1
        k = k - 1

    # We need to push the kernel through the previous isogeny before continuing
    if length > 3:
        T1 = generic_eval(strat_points[0][0], I)
        T2 = generic_eval(strat_points[0][1], I)
        strat_points[0] = (T1, T2)

    # Compute the codomain using only the four torsion above and
    # push the image points through the isogeny
    I, OA = generic_codomain_with_four_torsion(strat_points[0][0], OA, False, False)
    points = [generic_eval(P, I, False, False) for P in points]

    # Compute the codomain with no torsion information and push
    # the image points through the isogeny
    I, OA = generic_codomain(OA, True, False)
    points = [generic_eval(P, I, True, False) for P in points]

    # Compute the splitting isomorphism
    M = splitting_isomorphism_matrix(OA, randomize)

    # Apply on the codomain and points to evaluate
    OA = apply_isomorphism(OA, M, null_point=True)
    points = [apply_isomorphism(P, M) for P in points]

    # Compute the Montgomery coefficients
    (A1, C1), (A2, C2) = theta_to_product(OA)
    Fq = E1.base_ring()
    E3 = KummerLine(Fq, [A1, C1])
    E4 = KummerLine(Fq, [A2, C2])

    # Compute the point coordinates on the KummerLines
    image_points = []
    for P in points:
        P, Q = theta_product_point_to_montgomery(P, OA)
        P, Q = E3(P), E4(Q)
        image_points.append((P, Q))

    return (E3, E4), image_points


def two_two_isogeny_chain_with_torsion(
    kernel: tuple[tuple[KummerPoint]],
    length: int,
    points: list[tuple[KummerPoint]] = [],
    randomize=None,
    verify=False,
):
    """
    Algorithm 8.48 of the SQIsign spec.

    Compute a (2,2)-isogeny chain and optionally evaluate this isogeny on a given list of points.
    Assumes that we have the E[2^(n + 2)] torsion available to optimise final steps of the isogeny

    Input:
        - kernel ((P1, P2), (Q1, Q2)): is represented by two x-only bases xP, xQ, xPQ on curves E1, E2
        - length (int): the length of the (2^n, 2^n)-isogeny chain
        - points [tuple(KummerPoint)]: pairs of KummerPoints on E1 x E2 to evaluate
        - randomize (random_bytes function): if not None, expects a function which returns random bytes
        - verify (bool): whether to perform additional arithmetic checks during isogeny computation
    """
    # Extract out the bases on E_chl and E_aux
    chl_basis, aux_basis = kernel

    print(f"{length = }")

    # Montgomery coefficients of E_chl and E_aux needed for
    # Jacobian arithmetic
    E1 = chl_basis[0].parent()
    E2 = aux_basis[0].parent()
    (A1, A2) = (E1.a(), E2.a())

    # Lift xP, xQ, xPQ to points on the Jacobian for evaluation
    P_chl, Q_chl = lift_basis_to_jacobian(chl_basis, A1)
    P_aux, Q_aux = lift_basis_to_jacobian(aux_basis, A2)

    # Init strategy arrays
    space = length.bit_length() + 1
    strat_points = [(((0, 0, 0), (0, 0, 0)), ((0, 0, 0), (0, 0, 0)))] * space
    orders = [0] * space

    # Set first values of arrays
    strat_points[0] = ((P_chl, P_aux), (Q_chl, Q_aux))
    orders[0] = length
    k = 0

    # Step One: Perform doubling on the Jacobian points, and use 8-torsion
    #           points to compute the gluing codomain. Push all stored
    #           Jacobian points through the gluing isogeny to obtain ThetaPoints

    # First build a list of doubled points on the Jacobian
    while orders[k] != 1:
        k = k + 1
        # Because gluing eval is expensive we modify how the optimized chain is computed
        if orders[k - 1] >= 16:
            n = orders[k - 1] // 2
        else:
            n = orders[k - 1] - 1

        # Double all four points n times
        ((P1, P2), (Q1, Q2)) = strat_points[k - 1]
        for _ in range(n):
            P1 = DBL(P1, A1)
            P2 = DBL(P2, A2)
            Q1 = DBL(Q1, A1)
            Q2 = DBL(Q2, A2)
        strat_points[k] = ((P1, P2), (Q1, Q2))
        orders[k] = orders[k - 1] - n

    # Compute the gluing isogeny
    # The kernel for codomain computation only needs data on the KummerLine
    # but for evaluation we need the points as Jacobian points
    gluing_kernel = strat_points[k]
    gluing_kernel_x_only = prepare_gluing_kernel(E1, E2, gluing_kernel)
    I, OA, J, N = gluing_codomain(gluing_kernel_x_only, verify=verify)

    # Push Jacobian points through the isogeny. Assumes input points are of
    # the form (P, 0) or (0, Q)
    points = [gluing_eval_special(P, I, N) for P in points]

    # Push the kernel through the isogeny
    T1_8 = gluing_kernel[0]
    for i in range(k):
        P, Q = strat_points[i]
        P_im = gluing_eval((A1, A2), P, T1_8, J, N)
        Q_im = gluing_eval((A1, A2), Q, T1_8, J, N)
        strat_points[i] = (P_im, Q_im)
        orders[i] = orders[i] - 1

    print("domain after gluing:")
    print(f"{OA = }")
    print()

    print("kernel after gluing:")
    print(f"{strat_points[0][0] = }")
    print(f"{strat_points[0][1] = }")
    print()

    #
    # Main loop: compute (2,2) isogenies in the theta model with balanced strategy
    #
    k = k - 1
    j = 0
    while k >= 0 and orders[k] != 0:
        j = j + 1
        while orders[k] != 1:
            k = k + 1
            n = orders[k - 1] // 2
            # Perform n doublings of the kernel points (T1, T2)
            T1, T2 = strat_points[k - 1]
            for _ in range(n):
                T1 = OA.double(T1)
                T2 = OA.double(T2)
            strat_points[k] = (T1, T2)
            orders[k] = orders[k - 1] - n

        # Set Hadamard booleans for codomain and evaluation, the last two steps
        # we need to be careful as arithmetic can break close to the product
        # For the last step
        if j == length - 1:
            # We cannot perform verification of codomain on the last step
            # instead we find errors from the splitting isomorphism
            verify = False
            h1, h2 = True, False
        # For the penultimate step
        elif j == length - 2:
            h1, h2 = False, False
        else:  # All other steps
            h1, h2 = False, True

        # Compute codomain with kernel in the 8-torsion
        I, OA = generic_codomain_with_eight_torsion(
            *strat_points[k], h1, h2, verify=verify
        )

        # Push the evaluation points through the isogeny
        points = [generic_eval(P, I, h1, h2) for P in points]

        # Push the kernel elements (T1, T2) through the isogeny
        assert orders[k] == 1
        for i in range(k):
            T1 = generic_eval(strat_points[i][0], I, h1, h2)
            T2 = generic_eval(strat_points[i][1], I, h1, h2)
            strat_points[i] = (T1, T2)
            orders[i] = orders[i] - 1
        k = k - 1

    print("codomain before isomorphism")
    print(f"{OA = }")
    print()

    # Compute the splitting isomorphism
    M = splitting_isomorphism_matrix(OA, randomize)

    # Apply on the codomain and points to evaluate
    OA = apply_isomorphism(OA, M, null_point=True)
    points = [apply_isomorphism(P, M) for P in points]

    # Compute the Montgomery coefficients
    (A1, C1), (A2, C2) = theta_to_product(OA)
    Fq = E1.base_ring()
    E3 = KummerLine(Fq, [A1, C1])
    E4 = KummerLine(Fq, [A2, C2])

    # Compute the point coordinates on the KummerLines
    image_points = []
    for P in points:
        P, Q = theta_product_point_to_montgomery(P, OA)
        P, Q = E3(P), E4(Q)
        image_points.append((P, Q))

    return (E3, E4), image_points


def two_two_isogeny_chain_verify(kernel: tuple[tuple[KummerPoint]], length: int):
    """
    Wrapper function for the two two chain used during verification where we
    never want to evaluate points and we want to have the additional verification checks
    """
    (E3, _), _ = two_two_isogeny_chain_with_torsion(kernel, length, verify=True)
    return E3
