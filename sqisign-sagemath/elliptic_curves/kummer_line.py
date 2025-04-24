"""
Implementation of the Kummer Line of Montgomery Curves and the corresponding
Kummer Points used for x-only Montgomery curve arithmetic

Adapted from:
https://github.com/GiacomoPope/KummerIsogeny
"""

from sage.all import cached_method, Integer, EllipticCurve

from sage.structure.element import RingElement
from sage.schemes.elliptic_curves.ell_generic import EllipticCurve_generic
from sage.schemes.elliptic_curves.ell_point import EllipticCurvePoint_field

from utilities.deterministic_sqrt import sqrt_fp2

# =================================================== #
#     Class for the Kummer Line x(x^2 + Ax + 1)       #
# =================================================== #


class KummerLine:
    def __init__(self, *args):
        self._curve = None

        # Allow the creation of the Kummer Line from an EllipticCurve
        if len(args) == 1:
            (curve,) = args
            if not isinstance(curve, EllipticCurve_generic):
                raise TypeError("not an elliptic curve")
            ainvs = curve.a_invariants()
            A, C = ainvs[1], 1
            if ainvs != (0, A, 0, 1, 0):
                raise ValueError("Must use Montgomery model")
            self._curve = curve
            self._base_ring = curve.base_ring()

        # Allow the creation of the Kummer line from a base field and coeffs.
        elif len(args) == 2:
            base_ring, curve_constants = args
            # Extract curve constants
            if isinstance(curve_constants, RingElement) or len(curve_constants) == 1:
                A = curve_constants
                C = 1
            elif len(curve_constants) == 2:
                A, C = curve_constants
            else:
                raise ValueError(
                    "The Montgomery coefficient must either be a single scalar a, or\
                    a tuple [A, C] representing a = A/C."
                )
            self._base_ring = base_ring
        else:
            raise ValueError(
                "A Kummer Line must be constructed from either a Montgomery curve, or\
                    a base field and tuple representing the coefficient A/C = [A, C]"
            )

        # init variables
        self._A = self._base_ring(A)
        self._C = self._base_ring(C)

        # Make sure the curve is not singular
        if (self._A**2 - 4 * self._C**2).is_zero():
            raise ValueError(
                f"Constants {curve_constants} do not define a Montgomery curve"
            )

        CC = self._C + self._C
        self._A24 = self._A + CC
        self._C24 = CC + CC

    def __eq__(self, other):
        """
        Test equality of two curves
        """
        if self.base_ring() != other.base_ring():
            return False
        return self._A * other._C == other._A * self._C

    def __repr__(self):
        """
        String representation of the class
        """
        if self.a():
            return f"Kummer line of the Montgomery curve y^2 = x^3 + {self.a()._coeff_repr()}*x^2 + x over {self.base_ring()}"
        else:
            return f"Kummer line of the Montgomery curve y^2 = x^3 + x over {self.base_ring()}"

    def __call__(self, *coords):
        """
        Create a Kummer Point with this Kummer Line as the parent
        """
        return KummerPoint(self, *coords)

    def base_ring(self):
        """
        Return the base ring of the Kummer Line
        """
        return self._base_ring

    def AC(self):
        """
        Return the Montgomery coefficient A as a tuple
        representing the projective form (A : C)
        """
        return self._A, self._C

    def AC24(self):
        """
        Return the Montgomery coefficient (A + 2)/4 as a tuple
        representing the projective form (A24 : C24)
        """
        return self._A24, self._C24

    def zero(self):
        """
        Return the identity point on the Kummer Line
        """
        return self(None)

    def curve(self):
        """
        Lift the Kummer Line to an elliptic curve as a
        SageMath EllipticCurve
        """
        if not self._curve:
            self._curve = self.montgomery_curve()
        return self._curve

    def normalize_curve_constants(self):
        """
        Normalise the projective constants such that
        C = C24 = 1
        """
        self._A = self._A / self._C
        self._C = self._base_ring.one()

        self._A24 = self._A24 / self._C24
        self._C24 = self._base_ring.one()

    def is_x_coordinate(self, x):
        """
        Return whether x is a point on the Kummer Line

        TODO: this could be optimized to use (A : C)
        """
        a = self.a()
        y2 = x**3 + a*x**2 + x
        return y2.is_square()
    
    def is_on_curve(self, X, Z):
        """
        Return whether (X : Z) is a point on the curve

        TODO: this could be optimized to use (X : Z)
        """
        x = X / Z
        return self.is_x_coordinate(x)

    @cached_method
    def montgomery_curve(self):
        """
        Compute the Montgomery Curve associated with the
        Kummer Line
        """
        F = self.base_ring()
        a = self.a()
        return EllipticCurve(F, [0, a, 0, 1, 0])

    @cached_method
    def short_weierstrass_curve(self):
        """
        Compute the Isomorphic curve in the short Weierstrass model
        associated with the Kummer Line
        """
        F = self.base_ring()
        A = self.a()

        A_sqr = A * A
        A_cube = A * A_sqr
        a = 1 - A_sqr / 3
        b = (2 * A_cube - 9 * A) / 27
        return EllipticCurve(F, [a, b])

    @cached_method
    def j_invariant(self):
        """
        Compute the j-invariant of the Kummer Line
        """
        j_num = 256 * (self._A**2 - 3 * self._C**2) ** 3
        j_den = self._C**4 * (self._A**2 - 4 * self._C**2)
        return j_num / j_den

    @cached_method
    def a(self):
        """
        Compute the Montgomery coefficient as a value
        in the base field
        """
        if self._C.is_one():
            return self._A
        return self._A / self._C
    
    @cached_method
    def a24(self):
        """
        Compute the constant (A + 2) / 4 coefficient as a value
        in the base field
        """
        if self._C24.is_one():
            return self._A24
        return self._A24 / self._C24


# ====================================================== #
#  Class for points on the Kummer Line x(x^2 + Ax + 1)   #
# ====================================================== #


class KummerPoint:
    def __init__(self, parent, *coords, check=False):
        # Ensure the parent is the right type
        if not isinstance(parent, KummerLine):
            raise TypeError("not a Montgomery Kummer line")
        R = parent.base_ring()

        # Handle the case when
        if len(coords) == 1:
            (coords,) = coords
            # Point at infinity
            if coords is None:
                coords = (R.one(), R.zero())
            # Handle the case where the input is a KummerPoint alread
            elif isinstance(coords, KummerPoint):
                if parent != coords.parent():
                    raise ValueError("points must be on the same curve")
                return coords
            # Construct point from P on an elliptic curve in Montgomery form
            elif isinstance(coords, EllipticCurvePoint_field):
                # Make sure point's parent curve matches with Kummer Line
                a = parent.a()
                assert coords.curve().a_invariants() == (0, a, 0, 1, 0)
                coords = coords[0], coords[2]
            # Construct from X coordinate only
            elif isinstance(coords, RingElement):
                coords = (coords, R.one())
        # Handle the case where X, Z coordinates are both given
        elif len(coords) == 2:
            coords = tuple(map(R, coords))
        else:
            raise ValueError("not a point on ℙ¹")

        self._base_ring = R
        self._parent = parent
        self._X, self._Z = coords
        if check:
            assert self._parent.is_on_curve(self._X, self._Z)

    def __repr__(self):
        return f"Kummer Point [{self._X} : {self._Z}] on {self._parent}"

    def __bool__(self):
        """
        A point represents False if it is the point at infinity and
        True otherwise
        """
        return bool(self._Z)

    def __eq__(self, other):
        """
        Equality of two Kummer Points
        """
        if not isinstance(other, KummerPoint):
            raise ValueError("Can only compare equality between to Kummer Points")
        if self._parent != other._parent:
            return False
        return self._X * other._Z == other._X * self._Z

    def is_zero(self):
        """
        A Kummer Point is considered Zero if it is the identity point
        on the parent curve
        """
        return self._Z.is_zero()

    def base_ring(self):
        """
        Get the base ring of the Kummer Point coordinates
        """
        return self._base_ring

    def parent(self):
        """
        Get the Kummer Line of which this point is constructed on
        """
        return self._parent
    
    def normalize_point(self):
        """
        Set point to be (X : 1) in place
        """
        self._X = self.x()
        self._Z = self.parent().base_ring().one()

    def XZ(self):
        """
        Return the projective (X : Z) coordinates of the point
        """
        return self._X, self._Z

    def x(self):
        r""" """
        if not self._Z:
            raise ValueError("The identity point has no valid x-coordinate")
        if self._Z.is_one():
            return self._X
        return self._X / self._Z

    @cached_method
    def curve_point(self):
        """
        Deterministically lift an x-coordinate
        taking the smallest y-coordinate as the
        chosen root.
        """
        # Get the Montgomery curve and constant A
        L = self.parent()
        E = L.curve()
        A = L.a()

        # Compute y2, assume x is a valid coordinate
        x = self.x()
        y2 = x * (x**2 + A * x + 1)
        y = sqrt_fp2(y2)
        return E(x, y)

    # =================================== #
    # Addition and multiplication helpers #
    # =================================== #

    @staticmethod
    def xDBL(X, Z, A24, C24):
        """
        Algorithm 8.3 of the SQISign spec.

        Montgomery doubling with projective curve constant

        Input:  projective point P = (X:Z), curve constants (A24 : C24)
        Output: projective point [2]P = (X2 : Z2)

        Cost: 4M + 2S + 4a
        """
        t0 = X + Z
        t0 = t0 * t0
        t1 = X - Z
        t1 = t1 * t1
        t2 = t0 - t1
        t1 = t1 * C24
        X2 = t0 * t1
        t0 = t2 * A24
        t0 = t0 + t1
        Z2 = t0 * t2

        return X2, Z2

    @staticmethod
    def xADD(XP, ZP, XQ, ZQ, xPQ, zPQ):
        """
        Algorithm 8.4 of the SQISign spec.

        Montgomery differential addition

        Input:  projective coordinates P = (XP : ZP),
                Q=(XQ : ZQ), and their difference
                x(P-Q) = (xPQ : zPQ)
        Output: coordinates of sum P + Q = (XQP : ZQP)

        Cost: 4M + 2S + 6a
        """
        t0 = XP + ZP
        t1 = XP - ZP
        t2 = XQ + ZQ
        t3 = XQ - ZQ
        t0 = t0 * t3
        t1 = t1 * t2
        t2 = t0 + t1
        t3 = t0 - t1
        t2 = t2 * t2
        t3 = t3 * t3
        XPQ = zPQ * t2
        ZPQ = xPQ * t3

        return XPQ, ZPQ

    @staticmethod
    def xDBLADD(XP, ZP, XQ, ZQ, xPQ, zPQ, A24, C24):
        """
        Algorithm 8.5 of the SQISign spec.

        Montgomery ladder for simultaneous doubling and differential
        addition

        Input: projective coordinates P=(XP:ZP) and Q=(XQ:ZQ),
               projective difference P-Q=(xPQ:zPQ) and
               curve constant A24/C24=(A+2C)/4C.
        Output: projective coordinates of 2P=(X2P:Z2P)
                and Q+P=(XQP:ZQP)

        Cost: 8M + 4S + 8a
        """

        t0 = XP + ZP
        t1 = XP - ZP
        X2P = t0 * t0
        t2 = XQ - ZQ
        XQP = XQ + ZQ
        t0 = t0 * t2
        Z2P = t1 * t1
        t1 = t1 * XQP
        t2 = X2P - Z2P
        Z2P = Z2P * C24
        X2P = X2P * Z2P
        XQP = A24 * t2
        ZQP = t0 - t1
        Z2P = XQP + Z2P
        XQP = t0 + t1
        Z2P = Z2P * t2
        ZQP = ZQP * ZQP
        XQP = XQP * XQP
        ZQP = xPQ * ZQP
        XQP = XQP * zPQ

        return X2P, Z2P, XQP, ZQP

    # =================================== #
    # Addition and multiplication methods #
    # =================================== #
    def _double(self):
        """
        Returns [2] self
        """
        X, Z = self.XZ()
        A24, C24 = self._parent.AC24()
        X2, Z2 = self.xDBL(X, Z, A24, C24)
        return self._parent((X2, Z2))

    def double(self):
        """
        Wrapper function which deals with the doubling of
        the identity

        Returns [2] * self
        """
        # Deal with identity
        if not self._Z:
            return self
        return self._double()

    def _add(self, Q, PQ):
        """
        Performs differential addition assuming
        P, Q and PQ are all not the point at
        infinity
        """
        XP, ZP = self.XZ()
        XQ, ZQ = Q.XZ()
        XPQ, ZPQ = PQ.XZ()

        X_new, Z_new = self.xADD(XP, ZP, XQ, ZQ, XPQ, ZPQ)
        return self._parent((X_new, Z_new))

    def add(self, Q, PQ):
        """
        Function to perform differential addition and
        handle the cases when P, Q or PQ are the points
        at infinity
        """
        # Adding O + Q = Q
        if not self._Z:
            return Q

        # Adding P + O = P
        if not Q._Z:
            return self

        # Difference is the identity
        # so P = Q and P+Q = [2]P
        if not PQ._Z:
            return self._double()

        return self._add(Q, PQ)

    def __mul__(self, m):
        """
        Algorithm 8.6 of the SQISign spec. (Ladder)

        Montgomery-ladder to compute [m]P

        Input: coordinates of P=(XP:ZP)
               scalar factor m, curve constants (A:C)
        Output: KummerPoint [m]P=(X0:Z0)
        """
        if not isinstance(m, (int, Integer)):
            try:
                m = Integer(m)
            except TypeError:
                raise TypeError(f"Cannot coerce input scalar {m = } to an integer")

        # If m is zero, return identity
        if not m:
            return self.parent().zero()

        # [m]P = [-m]P for x-only
        m = abs(m)

        # Extract base field and coefficients
        R = self.base_ring()
        XP, ZP = self.XZ()

        # Initialise for loop
        X0, Z0 = R.one(), R.zero()
        X1, Z1 = XP, ZP

        # Extract out A24 and C24 from parent curve
        A24, C24 = self.parent().AC24()

        # Montgomery-ladder
        for bit in bin(m)[2:]:
            if bit == "1":
                X1, Z1, X0, Z0 = self.xDBLADD(X1, Z1, X0, Z0, XP, ZP, A24, C24)
            else:
                X0, Z0, X1, Z1 = self.xDBLADD(X0, Z0, X1, Z1, XP, ZP, A24, C24)

        return self._parent(X0, Z0)

    def __rmul__(self, m):
        return self * m

    def __imul__(self, m):
        self = self * m
        return self

    def ladder_3_pt(xP, xQ, xPQ, m):
        """
        Algorithm 8.7 of the SQISign spec.

        Compute xP + [m]xQ using x-only arithmetic.

        Note: self = xP
        """
        if not isinstance(m, (int, Integer)):
            try:
                m = Integer(m)
            except TypeError:
                raise TypeError(f"Cannot coerce input scalar {m = } to an integer")

        # If m is zero, return xP
        if not m:
            return xP

        # [m]P = [-m]P for x-only
        m = abs(m)

        # Converting parameters for projective DBLADD -> (A24:C24)=(A+2C:4C)
        A24, C24 = xP.parent().AC24()

        # Extract out coordinates
        X0, Z0 = xP.XZ()
        X1, Z1 = xQ.XZ()
        X2, Z2 = xPQ.XZ()

        # Montgomery-ladder
        for bit in bin(m)[:1:-1]:
            if bit == "1":
                X1, Z1, X0, Z0 = xP.xDBLADD(X1, Z1, X0, Z0, X2, Z2, A24, C24)
            else:
                X1, Z1, X2, Z2 = xP.xDBLADD(X1, Z1, X2, Z2, X0, Z0, A24, C24)
        return xP._parent(X0, Z0)

    @staticmethod
    def __ladder_biscalar_recoding(m, n):
        """
        Helper function for the recoding stage of ladder_biscalar

        Given two scalars m, n of at most k bits precompute the values
        sigma = (sigma_0, sigma_1) in {0, 1} as well as the array of k
        bits denoted r
        """
        # First parse everything to Sage integer type
        try:
            m, n = Integer(m), Integer(n)
        except TypeError:
            raise TypeError(
                f"Cannot coerce input scalar {m = } or {n = } to an integer"
            )

        # max bit length of m and n for creation of r
        k = max(m.nbits(), n.nbits())

        # Recoding of the integers
        if not (m & 1) and (n & 1):
            s0, s1 = (1, 0)
        else:
            s0, s1 = (0, 1)

        # Ensure both scalars are odd
        m_prime, n_prime = m, n
        if not (m & 1):
            m_prime -= 1
        if not (n & 1):
            n_prime -= 1

        # Parse integers to arrays of bits (LSB first)
        b0 = m_prime.digits(2, padto=(k + 1))
        b1 = n_prime.digits(2, padto=(k + 1))
        b = [b0, b1]

        # Store array of 2*k bits made from bits of m, n
        r = [0] * 2 * k
        for i in range(k):
            r[2 * i] = b[s0][i] ^ b[s0][i + 1]
            r[2 * i + 1] = b[s1][i] ^ b[s1][i + 1]
            if r[2 * i + 1]:
                s0, s1 = s1, s0

        return s0, s1, r

    def ladder_biscalar(xP, xQ, xPQ, m, n):
        """
        Algorithm 8.8 of the SQISign spec.

        Given x(P), x(Q) and x(P - Q) together with two scalars m, n
        compute the point R = [m]P + [n]Q using x-only arithmetic

        NOTE: self is xP for this function
        """
        # Encode the scalars m, n into bits s0, s1 and bitarray r
        s0, s1, r = xP.__ladder_biscalar_recoding(m, n)
        k = len(r) // 2

        # We will access Ti, Ri for i in {0, 1, 2} in the algorithm
        L = xP.parent()
        T = [L.zero() for _ in range(3)]
        R = [L.zero() for _ in range(3)]

        # Init starting values T, R
        T[0], T[1] = xP, xQ
        R[1], R[2] = T[s0], T[s1]

        # Compute the difference points for T and R addition
        D1, D2 = R[1], R[2]
        R[2] = R[1].add(R[2], xPQ)
        F1, F2 = R[2], xPQ

        # Main ladder loop, compute [m]P + [n]Q
        for i in reversed(range(k)):
            r1, r2 = r[2 * i], r[2 * i + 1]
            h = r1 + r2

            # Compute new T values and swap diff points if needed
            T[0], T[1] = R[h % 2], R[2]
            T[0] = T[h // 2].double()
            T[1], T[2] = R[r2], R[r2 + 1]
            if r2:
                D1, D2 = D2, D1
            T[1] = T[1].add(T[2], D1)
            T[2] = R[0].add(R[2], F1)
            if h % 2:
                F1, F2 = F2, F1

            # Set R to be the T values and compute new T values
            R = T[:]

        # When m and n are both even we want R[0], if m and n are both odd
        # we want R[2], otherwise we want R[1]
        index = (m % 2) + (n % 2)
        return R[index]
