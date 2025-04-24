def hadamard(P):
    """
    The Hadamard transform
    """
    x, y, z, w = P
    a = x + y + z + w
    b = x - y + z - w
    c = x + y - z - w
    d = x - y - z + w
    return (a, b, c, d)

def squaring(P):
    """
    The squaring transform.
    """
    x, y, z, w = P
    return (x*x, y*y, z*z, w*w)

def pairwise_multiplication(P, Q):
    """
    Pairwise multiplication of two points
    """
    xP, yP, zP, wP = P
    xQ, yQ, zQ, wQ = Q
    return (xP*xQ, yP*yQ, zP*zQ, wP*wQ)

def theta_precompute(OA):
    """
    Precompute constants given a theta null point
    """
    # Memory management for 4x2x1 computations
    a, b, c, d = OA
    ccaa = (c, c, a, a) # memory management in C needed
    ddbb = (d, d, b, b) # memory management in C needed
    badc = (b, a, d, c) # memory management in C needed

    # C2 = (bcd, acd, abd, abc)
    C2 = pairwise_multiplication(ccaa, ddbb)
    C2 = pairwise_multiplication(C2, badc)

    # compute H(S(OA))
    AA, BB, CC, DD = hadamard(squaring(OA))

    # Memory management for 4x2x1 computations
    CCAA = (CC, CC, AA, AA) # memory management in C needed
    DDBB = (DD, DD, BB, BB) # memory management in C needed
    BADC = (BB, AA, DD, CC) # memory management in C needed

    # C1 = (BCD, ACD, ABD, ABC)
    C1 = pairwise_multiplication(CCAA, DDBB)
    C1 = pairwise_multiplication(C1, BADC)

    return C1, C2


def theta_double(P, C1, C2):
    """
    Double a point provided certain precomputed constants
    """
    P2 = squaring(P)
    P2 = hadamard(P2)
    P2 = squaring(P2)
    P2 = pairwise_multiplication(P2, C1)
    P2 = hadamard(P2)
    P2 = pairwise_multiplication(P2, C2)
    return P2

def theta_codomain(P, Q, h1, h2):
    if h1:
        ker_P = hadamard(P)
        ker_Q = hadamard(Q)
    else:
        ker_P = P
        ker_Q = Q

    ker_P = squaring(ker_P)
    ker_P = hadamard(ker_P)

    ker_Q = squaring(ker_Q)
    ker_Q = hadamard(ker_Q)

    # =================================================================
    # TODO: what's the most AVX512 friendly way of this middle part...
    # xA, xB, yC, yD = ker_P
    # zA, wB, zC, wD = ker_Q
    # xAwB = xA * wB
    # zAxB = zA * xB
    # A = zA * xAwB
    # B = wB * zAxB
    # C = zC * xAwB
    # D = wD * zAxB

    # Compute projectively 1/A, 1/B, 1/C, 1/D by computing ABCDxzw / A
    # ABCDxzw / B, etc.
    # zCwD = zC * wD
    # A_inv = xB * zCwD
    # B_inv = xA * zCwD
    # C_inv = D
    # D_inv = C

    # OB = (A, B, C, D)
    # inv = (A_inv, B_inv, C_inv, D_inv)
    # =================================================================

    # =================================================================
    # First attempt at parallel codomain, could be improved(?)
    # =================================================================
    # Memory management of kernel data to make two temp blobs
    xA, xB, yC, yD = ker_P
    zA, wB, zC, wD = ker_Q
    t1 = (xA, zA, xA, zA)
    t2 = (wB, xB, wB, xB)

    # 2 calls to 4x2x1 multiplication instead of 6 Fp2 mul
    OB = pairwise_multiplication(t1, t2)
    OB = pairwise_multiplication(OB, ker_Q)

    # Now we need to compute the inv precomputation
    # Memory management of kernel data
    A, B, C, D = OB
    t1 = (zC, zC, 1, 1)
    t2 = (wD, wD, 1, 1)
    inv = (xB, xA, D, C)

    # 2 calls to 4x2x1 multiplication instead of 3 Fp2 mul
    # This seems bad and there must be something smarter we can do!!
    # We really need:
    # t1 = zC * wD
    # inv = (t1 * xB, t1 * xA, D, C)
    inv = pairwise_multiplication(inv, t1)
    inv = pairwise_multiplication(inv, t2)

    if h2:
        OB = hadamard(OB)

    return OB, inv

def theta_eval(P, inv, h1, h2):
    """
    Evaluate a theta point through an isogeny given precomputed constants
    """
    if h1:
        im_P = hadamard(P)
    else:
        im_P = P

    im_P = squaring(im_P)
    im_P = hadamard(im_P)
    im_P = pairwise_multiplication(im_P, inv)

    if h2:
        im_P = hadamard(im_P)

    return im_P

def theta_isogeny_chain(domain, kernel, n):
    """
    Naive (2^n, 2^n) isogeny chain with no strategy
    """
    OA = domain
    P, Q = kernel
    for i in range(n):
        # Precompute constants for doubling
        C1, C2 = theta_precompute(OA)

        # Double the kernel points of order 2^(n+2) to get points of order 2^(1 + 2) = 8
        ker_P, ker_Q = P, Q
        for _ in range(n - i - 1):
            ker_P = theta_double(ker_P, C1, C2)
            ker_Q = theta_double(ker_Q, C1, C2)

        # For the ultimate and penultimate steps we do extra / less Hadamards
        # to avoid zeros from the product structure
        if i == n - 1:
            h1, h2 = True, False
        elif i == n - 2:
            h1, h2 = False, False
        else:
            h1, h2 = False, True

        # Compute the codomain from the kernel
        OA, inv = theta_codomain(ker_P, ker_Q, h1, h2)

        # Push the kernel through the images
        P = theta_eval(P, inv, h1, h2)
        Q = theta_eval(Q, inv, h1, h2)

    return OA
