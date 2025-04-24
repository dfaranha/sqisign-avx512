# ============================================ #
#     Fast square root and quadratic roots     #
# ============================================ #


def deterministic_fp_root(a):
    """
    Ensure the square root matches deterministic output
    """
    if int(a) & 1:
        return -a
    return a


def deterministic_fp2_root(a):
    """
    Ensure the square root matches deterministic output
    """
    a0, a1 = a.list()

    a0_is_zero = a0.is_zero()
    a0_is_odd = int(a0) & 1
    a1_is_odd = int(a1) & 1

    # if a0 is odd, or a0 = 0 and a1 is odd, negate result
    if a0_is_odd or (a0_is_zero and a1_is_odd):
        return -a
    return a


def sqrt_fp(x):
    """
    Compute square-root using that p = 3 mod 4
    """
    p = x.parent().characteristic()
    assert p % 4 == 3
    r = x ** ((p + 1) // 4)
    return deterministic_fp_root(r)


def sqrt_fp2(x):
    """
    Algorithm 8.2 of the SQISign spec.

    Computes a square-root in Fp2 with modulus x^2 + 1
    """
    assert x.is_square(), f"Cannot take a sqrt as element {x} is not a square"

    F = x.parent()
    p = F.characteristic()

    a0, a1 = x.list()

    delta = sqrt_fp(a0**2 + a1**2)
    if a1.is_zero():
        delta = a0

    x0 = delta + a0
    t0 = 2 * x0
    x1 = t0 ** ((p - 3) // 4)

    x0 = x0 * x1
    x1 = x1 * a1
    t1 = (2 * x0) ** 2

    if t1 == t0:
        root = F([x0, x1])
    else:
        root = F([x1, -x0])

    assert root**2 == x
    return deterministic_fp2_root(root)
