# Compares the gluing that lifts the kernel to Jacobian coordinates with the
# barycentric one translated from the SQIsign v3 reference, and counts the
# vectorized operations each one issues.
#
# The kernel is the one a verification hands to the chain, taken from the KAT
# signature, so the prime is 27 * 2^500 - 1 at the default level 3.

import sys

from sage.all import EllipticCurve

import avx_arithmetic
import utilities
import sqisign as sqisign_module

from sqisign import SQIsign
from sqisign_parameters import SQISIGN_PARAMETERS, generate_finite_field
from test_sqisign import parse_kat_file

from avx_types import AVX_4x2x1
from elliptic_types import BasisX
from theta_isogeny_chain import (
    theta_isogeny_chain_4x2x1,
    theta_isogeny_chain_bary_4x2x1,
)


# ======================================================================
# Operation counting
# ======================================================================

# Number of fp2 elements held by each blob type
WIDTH = {"421": 4, "811": 8, "241": 2}

MUL = ["avx_421_mul", "avx_811_mul", "avx_241_mul"]
SQR = ["avx_421_sqr", "avx_811_sqr", "avx_241_sqr"]
ADD = [
    "avx_421_add", "avx_421_sub", "avx_421_add_sub", "avx_421_sub_add",
    "avx_421_had", "avx_421_dbl", "avx_421_neg",
    "avx_811_add", "avx_811_sub", "avx_811_add_sub", "avx_811_had",
    "avx_241_add", "avx_241_sub", "avx_241_add_sub",
]
OTHER = ["sqrt_fp2", "batched_inversion"]

counts = {}


def patch():
    """
    Wrap the leaf operations and swap them into every module that imported
    them by name. The pair and iterated helpers are left alone, they issue
    their work through the leaves.
    """
    wrapped = {}
    for name, kind in (
        [(n, "M") for n in MUL] + [(n, "S") for n in SQR] + [(n, "a") for n in ADD]
    ):
        fn = getattr(avx_arithmetic, name)
        lanes = WIDTH[name.split("_")[1]]

        def make(fn=fn, name=name, kind=kind, lanes=lanes):
            def wrapper(*args, **kwargs):
                counts[kind] += 1
                counts[kind + "_fp2"] += lanes
                return fn(*args, **kwargs)
            return wrapper

        wrapped[name] = make()
        setattr(avx_arithmetic, name, wrapped[name])

    for name in OTHER:
        fn = getattr(utilities, name)

        def make(fn=fn, name=name):
            def wrapper(*args, **kwargs):
                counts[name] += 1
                return fn(*args, **kwargs)
            return wrapper

        wrapped[name] = make()
        setattr(utilities, name, wrapped[name])

    for module in list(sys.modules.values()):
        if getattr(module, "__file__", None) is None:
            continue
        for name, new in wrapped.items():
            if name in module.__dict__:
                module.__dict__[name] = new


def reset():
    for key in ["M", "S", "a", "M_fp2", "S_fp2", "a_fp2"] + OTHER:
        counts[key] = 0


def report(name):
    print(f"{name}:")
    print(
        f"  instructions {counts['M']:6d}M {counts['S']:6d}S {counts['a']:6d}a"
    )
    print(
        f"  fp2 lanes    {counts['M_fp2']:6d}M {counts['S_fp2']:6d}S {counts['a_fp2']:6d}a"
    )
    print(f"  sqrt {counts['sqrt_fp2']}, batched inversions {counts['batched_inversion']}")


# ======================================================================
# The kernel of a (2,2) chain
# ======================================================================

# The level picks the prime, 27 * 2^500 - 1 at level 3
LEVEL = int(sys.argv[1]) if len(sys.argv) > 1 else 3


class Captured(Exception):
    pass


change_of_basis = {}


def capture_chain_input(level):
    """
    Run a KAT verification and stop it where it hands the kernel to the chain.
    """

    def capture(domain, kernel, length):
        raise Captured((domain, kernel, length))

    def lift(E, basis):
        # the challenge basis is the last one lifted, and is the one the
        # change of basis is applied to
        change_of_basis["basis"] = basis
        return original_lift(E, basis)

    def cob(A, A24, P, Q, M):
        change_of_basis["args"] = (A, A24, P, Q, M)
        return original_cob(A, A24, P, Q, M)

    original = sqisign_module.theta_isogeny_chain_verification_8x1x1
    original_lift = sqisign_module.lift_normalised_basis_to_jacobian
    original_cob = SQIsign.apply_change_of_basis
    sqisign_module.theta_isogeny_chain_verification_8x1x1 = capture
    sqisign_module.lift_normalised_basis_to_jacobian = lift
    SQIsign.apply_change_of_basis = staticmethod(cob)
    try:
        sqisign = SQIsign(level)
        name = f"../assets/PQCsignKAT_{sqisign.sk_len}_SQIsign_p{sqisign.f}_{sqisign.c}.rsp"
        test = parse_kat_file(name)[0]
        sqisign.verify(test["msg"], test["sm"][: -test["mlen"]], test["pk"])
        raise AssertionError("the verification never reached the chain")
    except Captured as captured:
        return captured.args[0]
    finally:
        sqisign_module.theta_isogeny_chain_verification_8x1x1 = original
        sqisign_module.lift_normalised_basis_to_jacobian = original_lift
        SQIsign.apply_change_of_basis = staticmethod(original_cob)


def jac_to_basis(F, EE, P, Q):
    """
    The chain is handed the kernel in Jacobian coordinates, as the change of
    basis on the challenge curve is done there. Recover the x-only basis both
    paths start from, which is what the biscalar ladder of the v3 reference
    returns.
    """
    P = EE(P.X / P.Z**2, P.Y / P.Z**3)
    Q = EE(Q.X / Q.Z**2, Q.Y / Q.Z**3)
    PmQ = P - Q
    return BasisX(P.xy()[0], F.one(), Q.xy()[0], F.one(), PmQ.xy()[0], F.one())


p, F = generate_finite_field(SQISIGN_PARAMETERS[LEVEL])
domain, kernel_jac, length = capture_chain_input(LEVEL)
E1, E2 = domain
EE1 = EllipticCurve(F, [0, E1.A / E1.C, 0, 1, 0])
EE2 = EllipticCurve(F, [0, E2.A / E2.C, 0, 1, 0])
kernel = (
    jac_to_basis(F, EE1, kernel_jac[0], kernel_jac[1]),
    jac_to_basis(F, EE2, kernel_jac[2], kernel_jac[3]),
)

c, f = SQISIGN_PARAMETERS[LEVEL]["cofactor"], SQISIGN_PARAMETERS[LEVEL]["f"]
print(f"level {LEVEL}, p = {c} * 2^{f} - 1, chain of length {length}")


# ======================================================================
# The first step of the chain, both ways
# ======================================================================

from avx_arithmetic import avx_811_to_421, avx_421_add, avx_421_sub
from elliptic_arithmetic import (
    lift_normalised_basis_to_jacobian,
    four_gluing_ladder,
)
from theta_structure import hadamard_pair
from theta_gluing import (
    gluing_codomain,
    gluing_codomain_x_only,
    gluing_eval_pair,
    gluing_eval_pair_bary,
    theta_invert_pair,
)
from theta_isogeny_chain import (
    normalise_chain_input,
    normalise_chain_input_bary,
    compute_product_doubles,
)


def strategy_split(order):
    return order // 2 if order >= 16 else order - 1


def step_one_jac(domain, kernel, steps):
    """
    Step one of theta_isogeny_chain_4x2x1: lift to the Jacobian, double in the
    Weierstrass model and push every stored point through the gluing with the
    eight torsion as the translation point.
    """
    E1, E2, B1, B2 = normalise_chain_input(domain, kernel)
    P1, Q1 = lift_normalised_basis_to_jacobian(E1, B1)
    P2, Q2 = lift_normalised_basis_to_jacobian(E2, B2)

    A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
    A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)

    space = steps.bit_length() + 2
    strat_points = [(None, None, None)] * space
    orders = [0] * space
    strat_points[0] = [
        AVX_4x2x1(P1.X, Q1.X, P2.X, Q2.X),
        AVX_4x2x1(P1.Y, Q1.Y, P2.Y, Q2.Y),
        AVX_4x2x1(P1.Z, Q1.Z, P2.Z, Q2.Z),
    ]
    orders[0] = steps

    strat_points, orders, k = compute_product_doubles(A_12, strat_points, orders, 0)

    X_8, Y_8, Z_8 = strat_points[k]
    OA, OA_inv, J, N = gluing_codomain(A24_12, X_8, Z_8)

    T8 = (
        AVX_4x2x1(X_8[0], X_8[0], X_8[2], X_8[2]),
        AVX_4x2x1(Y_8[0], Y_8[0], Y_8[2], Y_8[2]),
        AVX_4x2x1(Z_8[0], Z_8[0], Z_8[2], Z_8[2]),
    )

    images = [gluing_eval_pair(A_12, strat_points[i], T8, J, N) for i in range(k)]

    return OA, images


def step_one_bary(domain, kernel, steps):
    """
    Step one of theta_isogeny_chain_bary_4x2x1.
    """
    E1, E2, B1, B2 = normalise_chain_input_bary(domain, kernel)

    A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)
    xD = AVX_4x2x1(B1.PX, B1.QX, B2.PX, B2.QX)
    one = AVX_4x2x1(1, 1, 1, 1)
    G = (avx_421_add(xD, one), avx_421_sub(xD, one))

    space = steps.bit_length() + 2
    strat_points = [(None, None, None, None)] * space
    orders = [0] * space
    strat_points[0] = (xD, one, one, AVX_4x2x1(0, 0, 0, 0))
    orders[0] = steps

    k = 0
    while orders[k] != 1:
        k = k + 1
        n = strategy_split(orders[k - 1])
        strat_points[k] = four_gluing_ladder(A24_12, *strat_points[k - 1], xD, n)
        orders[k] = orders[k - 1] - n

    X_8, Z_8, _, _ = strat_points[k]
    OA, OA_inv, _, J_pair, N = gluing_codomain_x_only(A24_12, X_8, Z_8)

    imPQ = gluing_eval_pair_bary(strat_points[k], G, J_pair, N, False)
    J_prime = theta_invert_pair(imPQ)

    images = [hadamard_pair(imPQ)]
    images += [
        gluing_eval_pair_bary(strat_points[i], G, J_prime, N) for i in range(1, k)
    ]

    return OA, images


def projective_equal(P, Q):
    P = [x for x in P]
    Q = [x for x in Q]
    i = next(i for i, x in enumerate(P) if x != 0)
    if Q[i] == 0:
        return False
    lam = Q[i] / P[i]
    return all(lam * a == b for a, b in zip(P, Q))


patch()

reset()
OA_jac, images_jac = step_one_jac(domain, kernel, length)
report("step one, Jacobian")

reset()
OA_bary, images_bary = step_one_bary(domain, kernel, length)
report("step one, barycentric")

assert projective_equal(OA_jac, OA_bary), "codomains differ"
print("codomain theta null points agree")

assert len(images_jac) == len(images_bary)
for i, (A, B) in enumerate(zip(images_jac, images_bary)):
    assert projective_equal(A[:4], B[:4]), f"image {i} of the first generator differs"
    assert projective_equal(A[4:], B[4:]), f"image {i} of the second generator differs"
print(f"all {len(images_jac)} pairs of kernel images agree")


# ======================================================================
# The change of basis on the challenge curve
# ======================================================================

# The chain is fed by a change of basis, which the current code does on
# Jacobian points because the gluing needs them there. An x-only gluing lets
# it be a biscalar ladder instead, as in the v3 reference.
from limb_conversion import basis_from_64_to_51_bit
from elliptic_arithmetic import ladder_double_biscalar_1x2x4x1

A, A24, P, Q, M = change_of_basis["args"]
a00, a01, a10, a11 = M.list()
basis = basis_from_64_to_51_bit(change_of_basis["basis"])

reset()
SQIsign.apply_change_of_basis(A, A24, P, Q, M)
report("change of basis, Jacobian")

reset()
ladder_double_biscalar_1x2x4x1(A, A24, basis, a00, a10, a01, a11)
report("change of basis, x-only")


# ======================================================================
# The primitives that differ, one call each
# ======================================================================

from elliptic_arithmetic import (
    four_double_ws,
    four_gluing_dbladd,
    four_add_components,
    four_bary_components,
)

A24_12 = AVX_4x2x1(E1.A24, E1.A24, E2.A24, E2.A24)
A_12 = AVX_4x2x1(E1.A, E1.A, E2.A, E2.A)
R = AVX_4x2x1(kernel[0].PX, kernel[0].QX, kernel[1].PX, kernel[1].QX)
S = AVX_4x2x1(kernel[0].PmQX, kernel[0].QX, kernel[1].PmQX, kernel[1].QX)

for name, call in [
    ("four_double_ws       ", lambda: four_double_ws(R, S, R, S)),
    ("four_gluing_dbladd   ", lambda: four_gluing_dbladd(A24_12, R, S, R, S, R)),
    ("four_add_components  ", lambda: four_add_components(A_12, (R, S, R), (S, R, S))),
    ("four_bary_components ", lambda: four_bary_components(R, S, R, S, R, S)),
]:
    reset()
    call()
    print(
        f"{name} {counts['M']}M {counts['S']}S {counts['a']}a instructions,"
        f" {counts['M_fp2']}M {counts['S_fp2']}S {counts['a_fp2']}a fp2"
    )

# ======================================================================
# Whole chain, for the totals
# ======================================================================


def remove_extra_torsion(EE, B):
    """
    The chains taking a basis want a kernel of order exactly 2^length.
    """
    P = 4 * EE.lift_x(B.PX / B.PZ)
    Q = 4 * EE.lift_x(B.QX / B.QZ)
    PmQ = EE.lift_x(B.PmQX / B.PmQZ)
    if (P - Q).xy()[0] != (4 * PmQ).xy()[0]:
        Q = -Q
    PmQ = P - Q
    return BasisX(P.xy()[0], F.one(), Q.xy()[0], F.one(), PmQ.xy()[0], F.one())


exact_kernel = (
    remove_extra_torsion(EE1, kernel[0]),
    remove_extra_torsion(EE2, kernel[1]),
)

for algorithm in [theta_isogeny_chain_4x2x1, theta_isogeny_chain_bary_4x2x1]:
    reset()
    algorithm(domain, exact_kernel, length)
    report(algorithm.__name__)
