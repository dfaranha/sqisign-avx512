from sage.all import GF, EllipticCurve, proof, randint
from elliptic_curves.kummer_line import KummerLine
from elliptic_curves.torsion_basis import torsion_basis_to_hint
from elliptic_curves.tate_pairing import tate_pairing

def lift_basis_to_curve(P, Q, PmQ):
    # Lift points t curve
    P_lift = P.curve_point()
    Q_lift = Q.curve_point()

    # Ensure signs match
    if (P_lift - Q_lift).x() != PmQ.x():
        Q_lift = -Q_lift
    assert (P_lift - Q_lift).x() == PmQ.x()

    return P_lift, Q_lift

if __name__ == "__main__":
    proof.all(False)

    passed = 1

    p = 5 * 2**248 - 1
    Fp2 = GF((p, 2), names="i", modulus=[1, 0, 1])
    E = EllipticCurve(Fp2, [0, 6, 0, 1, 0])
    E.set_order((p + 1)**2, num_checks = 0)

    for _ in range(10):
        E = E.isogeny(E.random_point(), algorithm="factored", model="montgomery").codomain()
        L = KummerLine(E)
        
        basis, _ = torsion_basis_to_hint(L, 248)
        P, Q, PmQ = basis
        PpQ = P.add(Q, PmQ)
        
        k = randint(0, 20)
        e = 248 - k
        # Ensure points all have order e
        P, Q, PmQ, PpQ = [2**k * x for x in [P, Q, PmQ, PpQ]]        
        tPQ = tate_pairing(L, e, P, Q, PpQ)
        
        ###################################
        #       first sanity check:       #
        ###################################
        # the paring has the correct order
        tPQ_2e_1 = tPQ ** (2 ** (e-1))
        if tPQ_2e_1.is_one() and not (tPQ_2e_1**2).is_one():
            passed = 0
            print("the pairing has the wrong order")
            
        ####################################
        #       second sanity check:       #
        ####################################
        # the paring matches SageMath
        P_lift, Q_lift = lift_basis_to_curve(P, Q, PmQ)
        ePQ_sage = P_lift.tate_pairing(Q_lift, 2**e, 2)
        if tPQ != ePQ_sage:
            print("pairing result doe not match Sage")
            passed = 0

        ####################################
        #       third sanity check:       #
        ####################################
        # the paring is linear
        a1, a2, a3, a4 = [randint(0, 2**e) for _ in range(4)]
        while (a1 * a4 - a2 * a3) % 2 == 0:
            a1, a2, a3, a4 = [randint(0, 2**e) for _ in range(4)]
        
        P_new = a1  * P_lift + a2 * Q_lift
        Q_new = a3  * P_lift + a4 * Q_lift
        PQ_new = P_new + Q_new
        P, Q, PpQ = L(P_new), L(Q_new), L(PQ_new)
        
        tPQ_ = tate_pairing(L, e, P, Q, PpQ)
        det = a1 * a4 - a2 * a3
        if tPQ ** det != tPQ_:
            passed = 0
            print("the pairing is not linear")
            
        ###################################
        #       fourth sanity check:      #
        ###################################
        # the pairing is compatible with isogenies

        ker = (2 ** e) * E.random_point()
        while ker == E(0):
            ker = (2 ** e) * E.random_point() 
            
        phi = E.isogeny(ker, algorithm="factored", model="montgomery")
        d = phi.degree()
        P_prime = phi(P_lift)
        Q_prime = phi(Q_lift)
        PQ_prime = P_prime + Q_prime
        E_prime = phi.codomain()

        L_prime = KummerLine(E_prime)
        P, Q, PpQ = L_prime(P_prime), L_prime(Q_prime), L_prime(PQ_prime)
        tPQ_prime = tate_pairing(L_prime, e, P, Q, PpQ)
        
        if tPQ ** d != tPQ_prime:
            passed = 0
            print("the pairing is not compatible with isogenies")
    
    if passed:
        print("all tests passed")
