from typing import Optional
from sage.structure.element import get_coercion_model


class ThetaPoint:
    """
    Class representing a theta point with four projective coordinates
    """

    def __init__(self, *coords):
        if len(coords) == 1 and isinstance(coords[0], (list, tuple)):
            coords = coords[0]

        if len(coords) != 4:
            raise ValueError("theta point must be constructed from four coordinates")

        # Find a common parent of the coordinates
        cm = get_coercion_model()
        self._base_ring = cm.common_parent(*coords)

        # Parse point coordinates to common parent
        self.x, self.y, self.z, self.w = map(self._base_ring, coords)

    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        elif index == 2:
            return self.z
        elif index == 3:
            return self.w
        else:
            raise IndexError("Index out of range")

    def base_ring(self):
        """
        Return the base ring of the coordinates
        """
        return self._base_ring

    def coords(self):
        """
        Return the coordinates (x, y, z, w) as a tuple
        """
        return (self.x, self.y, self.z, self.w)

    def __str__(self):
        return f"{self.coords()}".replace(", ", " : ")

    def __repr__(self):
        return f"{self.coords()}".replace(", ", " : ")

    @staticmethod
    def _hadamard_transform(x, y, z, w):
        """
        The Hadamard transform acting on four elements.
        Note we use recursive formula to save 4a to the naive method

        Cost: 8a
        """
        x, y = x + y, x - y
        z, w = z + w, z - w
        return x + z, y + w, x - z, y - w

    def hadamard(self):
        """
        Compute the Hadamard operation on the point coordinates.
        Maps a point to its dual
        """
        x, y, z, w = self._hadamard_transform(*self.coords())
        return x, y, z, w

    @staticmethod
    def _squaring_transform(x, y, z, w):
        """
        The Squaring transform acting on four elements

        Cost: 4S
        """
        return (x * x, y * y, z * z, w * w)

    def squaring(self):
        """
        Compute the Hadamard operation on the point coordinates
        """
        x, y, z, w = self._squaring_transform(*self.coords())
        return x, y, z, w

    def _squaring_hadamard_transform(self, x, y, z, w):
        """
        Squaring followed by Hadamard transform

        Cost: 4S + 8a
        """
        xx, yy, zz, ww = self._squaring_transform(x, y, z, w)
        return self._hadamard_transform(xx, yy, zz, ww)

    def squaring_hadamard(self):
        """
        Compute the result of squaring followed by a Hadamard
        operation, used for doubling and isogenies
        """
        x, y, z, w = self._squaring_hadamard_transform(*self.coords())
        return x, y, z, w


class ThetaNullPoint(ThetaPoint):
    """
    Class representing a null theta point with four projective coordinates,
    child class of ThetaPoint and has additional functions made available
    for performing arithmetic with precomputed constants

    Optionally can be initalised with
    """

    def __init__(self, *coords, dual: Optional[ThetaPoint] = None):
        super().__init__(coords)

        # Precompute constants used for arithmetic
        self.__precompute_constants(dual)

    def __precompute_constants(self, dual=None):
        """
        Algorithm 8.29 of the SQISign spec.

        Precompute constants for arithmetic. When a dual NullPoint is known,
        we can save a few additions

        Cost: 4S + 12M + (8a)

        WARNING: using the dual input as specified seems to break this code
        """
        a, b, c, d = self.coords()
        if dual:
            AA, BB, CC, DD = self.squaring()
        else:
            AA, BB, CC, DD = self.squaring_hadamard()

        t1 = AA * BB
        t2 = CC * DD
        self.ABC = t1 * CC
        self.ABD = t1 * DD
        self.ACD = t2 * AA
        self.BCD = t2 * BB

        t1 = a * b
        t2 = c * d
        self.abc = t1 * c
        self.abd = t1 * d
        self.acd = t2 * a
        self.bcd = t2 * b

    def double(self, point: ThetaPoint):
        """
        Algorithm 8.30 of the SQISign spec.

        Compute the double of some ThetaPoint P

        Cost: 8S + 8M + 16a
        """
        # Compute S . H . S (x, y, z, w) of input point
        xP, yP, zP, wP = point.coords()
        X, Y, Z, W = self._squaring_hadamard_transform(xP, yP, zP, wP)
        X, Y, Z, W = self._squaring_transform(X, Y, Z, W)

        # Multiply by precomputed constants
        X = X * self.BCD
        Y = Y * self.ACD
        Z = Z * self.ABD
        W = W * self.ABC

        # Compute Hadamard
        X, Y, Z, W = self._hadamard_transform(X, Y, Z, W)

        # Multiply by precomputed constants
        X = X * self.bcd
        Y = Y * self.acd
        Z = Z * self.abd
        W = W * self.abc

        return ThetaPoint(X, Y, Z, W)
