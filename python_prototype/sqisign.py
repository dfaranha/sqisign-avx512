from sage.all import ZZ, Matrix

from hashlib import shake_256

from avx_types import AVX_2x4x1
from elliptic_types import Curve, BasisX, PointJ
from elliptic_arithmetic import (
    ladder_3pt_1x2x4x1,
    mul_pair,
    add_simple,
    lift_normalised_basis_to_jacobian,
)
from limb_conversion import (
    basis_from_64_to_51_bit,
    curve_to_2x4x1_data,
)
from torsion_basis import torsion_basis_from_hint
from elliptic_isogeny import (
    two_isogeny_chain_codomain_1x2x4x1,
)
from sqisign_parameters import SQISIGN_PARAMETERS, generate_finite_field
from theta_isogeny_chain import theta_isogeny_chain_verification_8x1x1
from utilities import batched_inversion


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
        self.c = parameters["cofactor"]
        assert self.f == (self.p + 1).valuation(2)

        self.response_len = parameters["response_len"]
        self.challenge_bytes = self.security_bits // 8
        self.response_bytes = (parameters["response_len"] + 9) // 8

        self.fp_encoded_len = parameters["fp_encoded_len"]
        self.fp2_encoded_len = 2 * self.fp_encoded_len
        self.pk_len = parameters["pk_len"]
        self.sk_len = parameters["sk_len"]
        self.sig_len = parameters["sig_len"]

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
        assert (
            self.sig_len
            == 2 + self.fp2_encoded_len + 4 * self.response_bytes + self.challenge_bytes
        )
        assert self.sig_len == len(signature), f"{self.sig_len = }, {len(signature) = }"

        # Auxiliary Curve bytes
        curve_bytes, signature = (
            signature[: 2 * self.fp_encoded_len],
            signature[2 * self.fp_encoded_len :],
        )
        E_aux = self._curve_from_bytes(curve_bytes)

        # Basis change coefficients
        aij_bytes, signature = (
            signature[: 4 * self.response_bytes],
            signature[4 * self.response_bytes :],
        )
        aij = [
            ZZ.from_bytes(aij_bytes[i : i + self.response_bytes], byteorder="little")
            for i in range(0, 4 * self.response_bytes, self.response_bytes)
        ]
        M_chl = Matrix(2, 2, aij)

        # challenge for kernel
        chl, signature = signature[:self.challenge_bytes], signature[self.challenge_bytes:]

        # bytes for basis hints
        hint_aux_byte = signature[0]
        hint_chl_byte = signature[1]
        hint_aux = ZZ(hint_aux_byte)
        hint_chl = ZZ(hint_chl_byte)

        return E_aux, M_chl, chl, hint_aux, hint_chl

    def _shake256(self, b):
        """
        Hash data with SHAKE256 to compute challenge bytes
        """
        return shake_256(b).digest(self.challenge_bytes)

    def _hash_challenge(self, pk: bytes, A_com, message: bytes):
        """
        Pack bytes and then hash to obtain challenge bytes
        """
        A_com_bytes_re = int(A_com[0]).to_bytes(self.fp_encoded_len, byteorder="little")
        A_com_bytes_im = int(A_com[1]).to_bytes(self.fp_encoded_len, byteorder="little")
        A_com_bytes = A_com_bytes_re + A_com_bytes_im
        return shake_256(b"SQI" + pk + A_com_bytes + message).digest(self.challenge_bytes)

    def _compute_challenge_from_scalar(
        self, A24: AVX_2x4x1, basis: tuple[AVX_2x4x1], chl: ZZ
    ) -> Curve:
        """
        Given the challenge bytes, compute a scalar, then the kernel then 
        the challenge codomain using a 2-isogeny chain. Our input here is
        the point k such that [4]k = ker(phi). This allows us to use the
        four torsion on the codomain to normalise the chain output without
        a square root. 
        """
        chl_scalar = ZZ.from_bytes(chl, byteorder="little")
        kernel = ladder_3pt_1x2x4x1(A24, basis, chl_scalar)
        return two_isogeny_chain_codomain_1x2x4x1(A24, kernel, self.security_bits)

    @staticmethod
    def normalise_chain_input(
        E1: Curve, E2: Curve, B1: BasisX, B2: BasisX
    ) -> tuple[Curve, Curve, BasisX, BasisX]:
        """
        Given a domain E1 x E2 and a two x-only bases on these curves
        we normalize x(P1) and x(P2), and additionally the curve
        constants
        """
        inv_P1_Z, inv_P2_Z, inv_E1_C, inv_E1_C24, inv_E2_C, inv_E2_C24 = (
            batched_inversion(B1.PZ, B2.PZ, E1.C, E1.C24, E2.C, E2.C24)
        )

        # TODO: i think at this point both curves are already normalized?
        assert E1.C == 1
        assert E2.C == 1
        assert E1.C24 == 1
        assert E2.C24 == 1

        E1 = Curve(E1.A * inv_E1_C, 1, E1.A24 * inv_E1_C24, 1)
        E2 = Curve(E2.A * inv_E2_C, 1, E2.A24 * inv_E2_C24, 1)
        B1 = BasisX(B1.PX * inv_P1_Z, 1, B1.QX, B1.QZ, B1.PmQX, B1.PmQZ)
        B2 = BasisX(B2.PX * inv_P2_Z, 1, B2.QX, B2.QZ, B2.PmQX, B2.PmQZ)

        return E1, E2, B1, B2

    @staticmethod
    def apply_change_of_basis(
        A: AVX_2x4x1, A24: AVX_2x4x1, P_chl: PointJ, Q_chl: PointJ, M: Matrix
    ):
        """
        Evaluate a 2x2 matrix A on the vector of elliptic curve points [P, Q]
        by acting with scalar multiplication.
        """
        # Extract out scalars
        a00, a01, a10, a11 = M.list()
        a, c = A
        a24, c24 = A24
        assert c == 1
        assert c24 == 1
        assert P_chl.Z == 1

        # TODO: how should this data be best packed for C... We only actually need a here
        curve_params = AVX_2x4x1(a, a24)

        # TODO
        # We can use 8-way arithmetic when we do xDBLADD on the main chain using
        # pairs of points, so we do pairwise multiplications, one to compute
        # [a00] * P and [a01] * P, then another to compute [a10] Q and [a11] Q.
        a00_P, a01_P = mul_pair(curve_params, P_chl, a00, a01)
        a10_Q, a11_Q = mul_pair(curve_params, Q_chl, a10, a11)

        # We then perform three more additions to compute:
        #
        # R = [a00] * P + [a10] Q
        # S = [a01] * P + [a11] Q
        # TODO: this should be parallelized easily too for a tiny saving
        R = add_simple(a, a00_P, a10_Q)
        S = add_simple(a, a01_P, a11_Q)

        return R, S

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
        E_pk, hint_pk = self._parse_public_key(public_key)

        # Parse the signature from bytes
        E_aux, M_chl, chl, hint_aux, hint_chl = self._parse_signature(
            signature
        )

        # NOTE: For the supersingularity testing rather than performing
        #       an explicit test we instead leave this as an implicit
        #       test that P, Q in E[2^n] have the full expected order,
        #       which is checked in our various isogeny chains

        # Check the challenge basis change matrix has canonical elements
        chl_order = ZZ(2 ** (self.response_len + 2))
        for m in M_chl.list():
            if m >= chl_order:
                return False

        # TODO: add the normalisation check here...

        # Create the torsion basis for the public key
        pk_basis = torsion_basis_from_hint(E_pk, hint_pk, self.security_bits + 2)

        # ================================================================================
        # NOTE: CONVERSION TO AND FROM 64 TO 51 BIT HERE.
        # Convert this basis from 64 bit to 51 bit before doubling and the change of basis
        # computation
        # ================================================================================

        _, A24_pk = curve_to_2x4x1_data(E_pk)
        pk_basis = basis_from_64_to_51_bit(pk_basis)

        # Compute the challenge curve with kernel [4] * (P_pk + [chl] Q_pk)
        try:
            E_chl = self._compute_challenge_from_scalar(A24_pk, pk_basis, chl)
        # For bad signatures, the chain may be malformed, if an error is triggered,
        # the signature is rejected.
        except ValueError as e:
            print(e)
            return False

        # ================================================================================

        # Create the torsion basis for the aux curve using the signature hint
        aux_basis = torsion_basis_from_hint(E_aux, hint_aux, self.response_len + 2)
        # Curve is now normalized when created
        # E_chl = normalize_curve_constants(E_chl)

        # Create the torsion basis for the challenge curve using the signature hint
        chl_basis = torsion_basis_from_hint(E_chl, hint_chl, self.response_len + 2)
        E_aux, E_chl, aux_basis, chl_basis = self.normalise_chain_input(
            E_aux, E_chl, aux_basis, chl_basis
        )
        A_chl, A24_chl = curve_to_2x4x1_data(E_chl)

        # TODO: new SQIsign v3 avoids Jacobians, uses biscalar ladders here and then changes how
        #       gluing is done to get everything working in x-only. However, we can use this older
        #       way with no issue.
        P_aux, Q_aux = lift_normalised_basis_to_jacobian(E_aux, aux_basis)
        P_chl, Q_chl = lift_normalised_basis_to_jacobian(E_chl, chl_basis)

        # Apply the change of basis matrix
        P_chl, Q_chl = self.apply_change_of_basis(A_chl, A24_chl, P_chl, Q_chl, M_chl)


        try:
            A3, _ = theta_isogeny_chain_verification_8x1x1(
                (E_chl, E_aux), (P_chl, Q_chl, P_aux, Q_aux), self.response_len
            )
        except ValueError as e:
            print(f"{e = }")
            return False

        # ==================================================================================
        # Compute the hash from the codomain
        chl_prime = self._hash_challenge(public_key, A3, message)

        # The signature is valid if we recover chl from the hash of the codomain
        # of the (2,2)-chain computed above
        return chl_prime == chl
