def _cubical_dbl(A24, P_tilde):
    """
    Algorithm 8.14 of the SQISign spec.

    Compute the cubical double 2 P_tilde  
    """
    xP, zP = P_tilde
    a = (xP + zP) ** 2
    b = (xP - zP) ** 2
    c = a - b
    X2 = a * b 
    Z2 = c * (b + A24 * c)
    return X2, Z2
    
def _cubical_diff_add(P_tilde, Q_tilde, ixPQ):
    """
    Algorithm 8.15 of the SQISign spec.

    Compute the cubical differential addition P + Q
    """
    xP, zP = P_tilde
    xQ, zQ = Q_tilde
    a = xP + zP
    b = xP - zP
    c = xQ + zQ
    d = xQ - zQ
    X2 = (a * d + b* c) ** 2
    Z2 = (a * d - b* c) ** 2
    X2 = X2 * ixPQ
    return X2, Z2
           
def _cubical_ladder(A24, e, PQ_tilde, P_tilde, ixQ):
    """
    Algorithm 8.16 of the SQISign spec.

    Compute the cubical point 2^e P_tilde + Q_tilde

    Args:
        A24: the curve constant (A + 2) / 4
        e: exponent of the two-torsion
        PQ_tilde: the cubical point P + Q
        P_tilde: the cubical point P
        xQ: inverse of the x-coordinate of Q

    Returns:
        the cubical point 2^e P_tilde + Q_tilde
    """
    nPQ = PQ_tilde
    nP = P_tilde 
    for _ in range(1, e + 1):
        nPQ = _cubical_diff_add(nPQ, nP, ixQ)
        nP = _cubical_dbl(A24, nP)
    return nP, nPQ
        
def _cubical_translate(P_tilde, T_tilde):
    """
    Algorithm 8.17 of the SQISign spec.

    Compute the cubical point P + T tilde

    Args:
        P_tilde: the cubical point P
        T_tilde: the cubical point T, which is a 
        two-torsion point

    Returns:
        the cubical point P + T tilde  
    """
    xP, zP = P_tilde
    xT, zT = T_tilde
    
    X = xT * xP - zT * zP
    Z = zT * xP - xT * zP
    if zT == 0:
        Z = -Z
    if xT == 0:
        X = -X
    return X, Z
         
def _cubical_ratio(P1_tilde, P2_tilde):
    """
    Algorithm 8.18 of the SQISign spec.

    Computes the value lambda such that xP2 = lambda xP1 and zP2 = lambda zP1  
    """
    xP1, zP1 = P1_tilde
    xP2, zP2 = P2_tilde
    if xP1 == 0:
        return zP2 / zP1
    return xP2 / xP1    
    
def _tate_pairing(A24, e, xP, xQ, xPQ):
    """
    Algorithm 8.19 of the SQISign spec.

    Compute the value of the reduced Tate pairing t_(2^e)

    Args:
        A24: the curve constant (A + 2) / 4
        e: exponent of the two-torsion
        xP: normalized x(P) = (xP : 1)
        xQ: normalized x(Q) = (xQ : 1)
        xQ: normalized x(P+Q) = (xPQ : 1)

    Returns:
        the reduced Tate pairing t_(2^e)(P, Q)
    """
    p = A24.base_ring().characteristic()
    # We invert xQ to avoid repeated inversions in the ladder
    ixQ = 1 / xQ[0]
    
    nP, nPQ = _cubical_ladder(A24, e - 1, xPQ, xP, ixQ)
    O = _cubical_translate(nP, nP)
    Q_prime = _cubical_translate(nPQ, nP)

    lam = _cubical_ratio(xQ, Q_prime) / _cubical_ratio((1,0), O)
    
    # computing lam ^ (p-1)
    lam_p_1 = lam.conjugate() / lam
    return (lam_p_1) ** ((p+1) // (2**e))

def tate_pairing(E, e, P, Q, PQ):
    """
    Wrapper around _tate_paring which ensures the curve and
    points are correctly normalized
    """
    # Normalize (A24 : C24) to (A24 : 1)
    A24 = E.a24()

    # Normalize (X : Z) to (x : 1)
    P.normalize_point()
    Q.normalize_point()
    PQ.normalize_point()

    # For arithmetic we work with tuples (X : Z) instead of KummerPoint
    xP = P.XZ()
    xQ = Q.XZ()
    xPQ = PQ.XZ()

    return _tate_pairing(A24, e, xP, xQ, xPQ)
