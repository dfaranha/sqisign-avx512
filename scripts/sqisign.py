from sage.matrix.constructor import Matrix
from sage.rings.integer_ring import ZZ

from hashlib import shake_256

from elliptic_types import Curve, BasisX, PointX
from elliptic_arithmetic import (
    j_invariant,
    ladder_3_pt,
    double_x_n,
    torsion_basis_from_hint,
    ladder_biscalar,
)

from elliptic_isogeny import two_isogeny_chain_codomain, two_isogeny_chain_small
from sqisign_parameters import SQISIGN_PARAMETERS, generate_finite_field
from theta_isogeny_chain import theta_isogeny_chain_verification_8x1x1


class SQIsign:
    """
    An implementation of the SQIsign digital signature algorithm.

    Input:
        security_level (int, optional): The NIST security level of the SQIsign algorithm. Defaults to 1.
        seed (bytes, optional): bytes to initialise a DRBG for deterministic signatures

    Methods:
        verify(message: bytes, signature: bytes, public_key: bytes) -> bool:
            Verify a signature for the given message and public key.
    """

    def __init__(self, security_level: int = 1):
        self._precompute_level_constants(security_level)

    def _precompute_level_constants(self, level):
        """
        Set all the internal constants for SQIsign based on
        the target security level
        """
        parameters = SQISIGN_PARAMETERS[level]
        self.security_bits = parameters["security_bits"]
        self.p, self.Fq = generate_finite_field(parameters)

        self.f = parameters["f"]
        assert self.f == (self.p + 1).valuation(2)

        self.hash_iterations = parameters["hash_iterations"]
        self.response_length = parameters["response_length"]

        self.fp_encoded_len = parameters["fp_encoded_len"]
        self.fp2_encoded_len = 2 * self.fp_encoded_len
        self.pk_len = parameters["pk_len"]
        self.sk_len = parameters["sk_len"]
        self.sig_len = parameters["sig_len"]

    # TODO: type of curve
    def _curve_from_bytes(self, curve_bytes: bytes) -> Curve:
        """
        Given bytes representing the Montgomery coefficient A
        compute the elliptic curve E/Fp2 : y^2 = x(x^2 + Ax + 1)
        """
        # Parse out the Montgomery coefficient
        A_re = ZZ.from_bytes(curve_bytes[: self.fp_encoded_len], byteorder="little")
        A_im = ZZ.from_bytes(curve_bytes[self.fp_encoded_len :], byteorder="little")
        A = self.Fq([A_re, A_im])
        return Curve(A, self.Fq.one(), (A + 2) / 4, self.Fq.one())

    # TODO: type of curve
    def _parse_public_key(self, public_key: bytes) -> tuple[Curve, int]:
        """
        Parse bytes interpreted as a SQIsign public key to extract
        an elliptic curve in Montgomery form and a hint < 128 used
        as a hint for torsion basis generation
        """
        # Ensure the lengths are as expected
        assert self.pk_len == self.fp2_encoded_len + 1
        assert self.pk_len == len(public_key)

        # Parse bytes into chunks
        curve_bytes, hint_byte = public_key[:-1], public_key[-1]

        # Parse the curve and the hint
        E_pk = self._curve_from_bytes(curve_bytes)
        hint_pk = ZZ(hint_byte)

        return E_pk, hint_pk

    def _parse_signature(
        self, signature: bytes
    ) -> tuple[Curve, ZZ, ZZ, Matrix, ZZ, ZZ, ZZ]:
        """
        Parse the SQIsign signature and extract the necessary components.
        """
        # Ensure the lengths are as expected
        scalar_n_bytes = (self.response_length + 9) // 8
        chl_n_bytes = self.security_bits // 8
        assert (
            self.sig_len
            == self.fp2_encoded_len + 2 + 4 * scalar_n_bytes + chl_n_bytes + 2
        )
        assert self.sig_len == len(signature), f"{self.sig_len = }, {len(signature) = }"

        # Auxiliary Curve bytes
        curve_bytes, signature = (
            signature[: 2 * self.fp_encoded_len],
            signature[2 * self.fp_encoded_len :],
        )
        E_aux = self._curve_from_bytes(curve_bytes)

        # backtracking and response length bytes
        backtracking_byte, signature = signature[0], signature[1:]
        two_resp_byte, signature = signature[0], signature[1:]
        n_bt = ZZ(backtracking_byte)
        r_rsp = ZZ(two_resp_byte)

        # Basis change coefficients
        aij_bytes, signature = (
            signature[: 4 * scalar_n_bytes],
            signature[4 * scalar_n_bytes :],
        )
        aij = [
            ZZ.from_bytes(aij_bytes[i : i + scalar_n_bytes], byteorder="little")
            for i in range(0, 4 * scalar_n_bytes, scalar_n_bytes)
        ]
        M_chl = Matrix(2, 2, aij)

        # challenge for kernel
        chl, signature = signature[:chl_n_bytes], signature[chl_n_bytes:]

        # bytes for basis hints
        hint_aux_byte = signature[0]
        hint_chl_byte = signature[1]
        hint_aux = ZZ(hint_aux_byte)
        hint_chl = ZZ(hint_chl_byte)

        return E_aux, n_bt, r_rsp, M_chl, chl, hint_aux, hint_chl

    def _iterative_shake256(self, b):
        """
        Repeatedly hash data with SHAKE256 to compute challenge bytes
        """
        hash_bytes = ((2 * self.security_bits) + 7) // 8
        scalar = shake_256(b).digest(hash_bytes)
        for _ in range(self.hash_iterations - 2):
            scalar = shake_256(scalar).digest(hash_bytes)
        hash_bytes = (self.security_bits + 7) // 8
        res = ZZ.from_bytes(shake_256(scalar).digest(hash_bytes), byteorder="little")
        res &= 2 ** (self.f - self.response_length) - 1
        return res.to_bytes(byteorder="little", length=hash_bytes)

    def _hash_challenge(self, E_pk: Curve, E_chl: Curve, message: bytes):
        """
        Pack bytes and then repeatedly hash to obtain challenge bytes
        """
        j_pk = j_invariant(E_pk)
        j_chl = j_invariant(E_chl)

        j_pk_bytes = j_pk[0].to_bytes(byteorder="little")
        j_pk_bytes += j_pk[1].to_bytes(byteorder="little")

        j_chl_bytes = j_chl[0].to_bytes(byteorder="little")
        j_chl_bytes += j_chl[1].to_bytes(byteorder="little")

        return self._iterative_shake256(j_pk_bytes + j_chl_bytes + message)

    def _compute_challenge_from_scalar(
        self, E_pk: Curve, basis_pk: BasisX, chl: ZZ, n_bt: ZZ
    ) -> Curve:
        """
        Given the challenge bytes, compute a scalar, then the kernel then the challenge codomain using a 2-isogeny chain
        """
        chl_scalar = ZZ.from_bytes(chl, byteorder="little")
        kernel = ladder_3_pt(E_pk, basis_pk, chl_scalar)
        kernel = double_x_n(E_pk, kernel, n_bt)

        return two_isogeny_chain_codomain(E_pk, kernel, self.f - n_bt)

    @staticmethod
    def _compute_short_two_isogeny(E: Curve, basis: PointX, M: Matrix, len: ZZ, r: ZZ):
        """
        Compute a short 2-isogeny chain given a basis and matrix from which to chose a kernel
        """
        # Extract points from basis
        xP, xQ, _ = basis

        # Compute the kernel depending on the change of basis M
        if M[0][0] % 2 == 0 and M[1][0] % 2 == 0:
            kernel = PointX(basis.QX, basis.QZ)
        else:
            kernel = PointX(basis.PX, basis.PZ)

        # Recover the kernel from repeated doubling
        ker = double_x_n(E, kernel, r + 2)

        # Compute the isogeny from the kernel and push through xP, xQ xPQ
        return two_isogeny_chain_small(E, ker, len, basis)

    @staticmethod
    def apply_change_of_basis(basis: BasisX, M: Matrix, order: ZZ):
        """
        Evaluate a 2x2 matrix A on the vector of elliptic
        curve points [P, Q] by acting with scalar multiplication
        """
        assert M.nrows() == M.ncols() == 2

        # Extract out scalars
        a00, a01, a10, a11 = M.list()

        # Compute the new scaled basis
        xR = ladder_biscalar(basis, a00, a10)
        xS = ladder_biscalar(basis, a01, a11)

        # For the difference point we must be careful the result
        # is positive, so we work modulo the point order
        diff_m = (a00 - a01) % order
        diff_n = (a10 - a11) % order
        xRS = ladder_biscalar(basis, diff_m, diff_n)

        return xR, xS, xRS

    def verify(self, message: bytes, signature: bytes, public_key: bytes) -> bool:
        """
        Algorithm 4.9 of the SQIsign spec.

        Verify a SQIsign signature

            Input:
                message (bytes): The message to be verified.
                signature (bytes): The signature to be verified.
                public_key (bytes): The public key to be used for verification.

            Output:
                bool: True if the signature is valid, False otherwise.
        """
        # Parse the public key from bytes
        E_pk, hint_pk = self.__parse_public_key(public_key)

        # Parse the signature from bytes
        E_aux, n_bt, r_rsp, M_chl, chl, hint_aux, hint_chl = self._parse_signature(
            signature
        )

        # NOTE: For the supersingularity testing rather than performing
        #       an explicit test we instead leave this as an implicit
        #       test that P, Q in E[2^n] have the full expected order,
        #       which is checked in our various isogeny chains

        # Set the modified response length from signature data
        e_rsp_prime = self.response_length - n_bt - r_rsp

        # Check the challenge basis change matrix has canonical elements
        chl_order = ZZ(2 ** (e_rsp_prime + r_rsp + 2))
        for m in M_chl.list():
            if m >= chl_order:
                return False

        # Create the torsion basis for the public key
        pk_basis = torsion_basis_from_hint(E_pk, hint_pk, self.f)

        # Compute the challenge curve with kernel 2^n_bt * (P_pk + [chl] Q_pk)
        try:
            E_chl = self._compute_challenge_from_scalar(E_pk, pk_basis, chl, n_bt)
        # For bad signatures, the chain may be maleformed, if an error is triggered,
        # the signature is rejected.
        except ValueError:
            return False

        # Create the torsion basis for the aux curve using the signature hint
        aux_basis = torsion_basis_from_hint(E_aux, hint_aux, self.f)
        aux_basis = [2 ** (self.f - e_rsp_prime - 2) * X for X in aux_basis]

        # Create the torsion basis for the challenge curve using the signature hint
        chl_basis = torsion_basis_from_hint(E_chl, hint_chl, self.f)
        chl_basis = [2 ** (self.f - e_rsp_prime - r_rsp - 2) * X for X in chl_basis]

        # Apply the change of basis matrix
        chl_basis = self.apply_change_of_basis(chl_basis, M_chl, chl_order)

        # Compute the short 2^r_rsp isogeny and push through points
        if r_rsp > 0:
            try:
                E_chl, chl_basis = self.__compute_short_two_isogeny(
                    E_chl, chl_basis, M_chl, r_rsp, e_rsp_prime
                )
            # For bad signatures, the chain may be malformed, if an error is triggered,
            # the signature is rejected.
            except ValueError:
                return False

        # In the very rare case, all of the isogeny was handled here and we skip the (2,2) chain
        if e_rsp_prime == 0:
            return chl == self._hash_challenge(E_pk, E_chl, message)

        # Compute the 2D isogeny
        # Because of our 2,2 chain with torsion above, the target curve is deterministically the first
        # curve E3
        try:
            E3, _ = theta_isogeny_chain_verification_8x1x1(
                (chl_basis, aux_basis), e_rsp_prime
            )
        except ValueError as e:
            print(f"{e = }")
            return False

        # Convert to Curve type... should do this in chain...
        A3, C3 = E3
        E3 = Curve(A3, C3, (A3 + 2 * C3), 4 * C3)

        # Compute the hash from the codomain
        chl_prime = self._hash_challenge(E_pk, E3, message)

        # The signature is valid if we recover chl from the hash of the codomain
        # of the (2,2)-chain computed above
        return chl_prime == chl
