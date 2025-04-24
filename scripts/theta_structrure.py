def hadamard_transform(P):
    """
    The Hadamard transform
    """
    x, y, z, w = P
    a = x + y + z + w
    b = x - y + z - w
    c = x + y - z - w
    d = x - y - z + w
    return (a, b, c, d)

def hadmard_transform_rec(P):
    """
    The Hadamard transform, with recursive optimisation
    """
    x, y, z, w = P
    x, y = x + y, x - y
    z, w = z + w, z - w
    return (x + z, y + w, x - z, y - w)

def squaring_transform(P):
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
