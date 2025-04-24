from sage.all import Matrix
from theta_model.theta_structures import ThetaPoint, ThetaNullPoint


def apply_isomorphism(P: ThetaPoint, N: Matrix, null_point=False):
    """
    Applies change of theta coordinate to a point.

    Input:
        - P: point in level 2 theta-coordinates.
        - N: matrix of the level 2 coordinates base change (in row convention).

    Output: Point in level 2 theta'-coordinates after base change by N.
    """
    x, y, z, w = P.coords()
    X = N[0, 0] * x + N[0, 1] * y + N[0, 2] * z + N[0, 3] * w
    Y = N[1, 0] * x + N[1, 1] * y + N[1, 2] * z + N[1, 3] * w
    Z = N[2, 0] * x + N[2, 1] * y + N[2, 2] * z + N[2, 3] * w
    W = N[3, 0] * x + N[3, 1] * y + N[3, 2] * z + N[3, 3] * w
    if null_point:
        return ThetaNullPoint(X, Y, Z, W)
    return ThetaPoint(X, Y, Z, W)
