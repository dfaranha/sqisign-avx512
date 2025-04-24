from sage.all import GF, EllipticCurve, proof
from elliptic_curves.kummer_line import KummerLine
from elliptic_curves.torsion_basis import torsion_basis_from_hint, torsion_basis_to_hint

if __name__ == "__main__":
    proof.all(False)

    p = 5 * 2**248 - 1
    Fp2 = GF((p, 2), names="i", modulus=[1, 0, 1])
    E = EllipticCurve(Fp2, [0, 6, 0, 1, 0])
    E.set_order((p + 1)**2, num_checks = 0)

    for _ in range(10):
        E = E.isogeny(E.random_point(), algorithm="factored", model="montgomery").codomain()
        L = KummerLine(E)

        basis, hint = torsion_basis_to_hint(L, 248)
        assert basis == torsion_basis_from_hint(L, hint, 248)

        basis, hint = torsion_basis_to_hint(L, 128)
        assert basis == torsion_basis_from_hint(L, hint, 128)

    print("all tests passed")
