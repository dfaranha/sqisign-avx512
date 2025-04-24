def apply_change_of_basis(basis, M, order):
    """
    Evaluate a 2x2 matrix A on the vector of elliptic
    curve points [P, Q] by acting with scalar multiplication
    """
    assert M.nrows() == M.ncols() == 2

    # Extract points from the basis
    xP, xQ, xPQ = basis

    # Extract out scalars
    a00, a01, a10, a11 = M.list()

    # Compute the new scaled basis
    xR = xP.ladder_biscalar(xQ, xPQ, a00, a10)
    xS = xP.ladder_biscalar(xQ, xPQ, a01, a11)

    # For the difference point we must be careful the result
    # is positive, so we work modulo the point order
    diff_m = (a00 - a01) % order
    diff_n = (a10 - a11) % order
    xRS = xP.ladder_biscalar(xQ, xPQ, diff_m, diff_n)

    return xR, xS, xRS
