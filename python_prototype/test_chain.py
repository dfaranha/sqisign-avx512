from elliptic_types import Curve, BasisX, PointX
from theta_isogeny_chain import (
    theta_isogeny_chain_4x2x1,
    theta_isogeny_chain_8x1x1,
    theta_isogeny_chain_with_torsion_4x2x1,
    theta_isogeny_chain_with_torsion_8x1x1,
    theta_isogeny_chain_verification_8x1x1,
)

from sage.rings.finite_rings.finite_field_constructor import GF
from sage.all import EllipticCurve
from avx_types import AVX_4x2x1
from convert import print_fp2_4x2x1

def lift_to_curve(A, P):
    E = EllipticCurve(A.parent(), [0, A, 0, 1, 0])
    if P[1].is_zero():
        return E(0)
    return E.lift_x(P[0] / P[1])


def is_four_torsion(P):
    return (4 * P).is_zero()


def double_double_torsion(E, B):
    F = E.base_ring()

    P = E.lift_x(B.PX / B.PZ)
    Q = E.lift_x(B.QX / B.QZ)
    PmQ = E.lift_x(B.PmQX / B.PmQZ)

    if (P - Q).xy()[0] != PmQ.xy()[0]:
        Q = -Q
    assert (P - Q).xy()[0] == PmQ.xy()[0]

    P = 4 * P
    Q = 4 * Q
    PmQ = P - Q

    return BasisX(P.xy()[0], F.one(), Q.xy()[0], F.one(), PmQ.xy()[0], F.one())


def remove_extra_torsion(E1, E2, B1, B2):
    EE1 = EllipticCurve([0, E1.A / E1.C, 0, 1, 0])
    EE2 = EllipticCurve([0, E2.A / E2.C, 0, 1, 0])
    B1 = double_double_torsion(EE1, B1)
    B2 = double_double_torsion(EE2, B2)
    return B1, B2


p = 65 * 2**376 - 1
F = GF((p, 2), names="i", modulus=[1, 0, 1])
i = F.gen()

A24 = AVX_4x2x1(F(2)/F(4), F(2)/F(4), F(2)/F(4), F(2)/F(4))
print("A24")
print_fp2_4x2x1(A24)

# Data copy-pasted from SQIsign KAT tests
# fmt: off
length = 188

A1 = 4131683057790482110951497386599752287803860405993903650417773614549990742009360322364695736152114176383922791261639*i + 8503291677713416569822955266222010049348627477532219342395465583570033228745845284418181896141039121263649927888552
A2 = 6253684468565606625292677311255235609002017613670453471516513226370375649996976861958837858465107705058220117148379*i + 6731855965225326514093968679167145095444193536580881406291761977017399546433758259957980669587221923602436019152904

PX1 = 4459490032468168524897008902352486982101294016680352988345490436146770048247894268191553262901533410123797976849768*i + 3834692973655305067038900167832371227066214356498366362929847641217921568542306940026681618522656288090229600779998
PZ1 = 2022466430601464277909666074882451978292764642811747724640497785294010261461336449931744862871436710243977916647588*i + 2129023732410368271237422107320556937788396626630458065106810363820672919965178426284431218353189721451208563700146

QX1 = 9104146149618282411578396361168604587981273563896716100581751063117217693624749106192631175767891789425489875254263*i + 4828005603354504444331929397393403723414735997812983235834522793251784080645433934925902561480222257196415559961590
QZ1 = 3038558980525734857271384257758087239680777227073824555550714974602834586314099219806989217980693790580302976499210*i + 1127625804309920650540682466234044927316341338750599796120223064476446946090162191648898500384430733717705853007976

PmQX1 = 7502974328272350019592219958096764788373837721585676124355439059238397858554521750205034689832511600974789097049182*i + 4902593931821257462751327960900082276362295019480143791938524982320719221551918135109478531090251003326706698107594
PmQZ1 = 458622574030462352404592680192874211808920949968321727521041221465738892546912430129415942164565648098268884254904*i + 889639083869234281449360362939073105101744728617929361477896290701353191080591724172339666345378530454455737490874

PX2 = 3695419242545787010444036499119483356730414505199286781922980003948737375190217590298161288637262494972634586558698*i + 3540138431535599808396820470121233501577198392951299200254816522821833984879585997369489237072861642288981682355097
PZ2 = 8791002556554714953378244269304390290595740811366525305937555020729643833144372863858761336799674915491915448743075*i + 6836822011082996737355874605544625668534417938368333256860796481105055805533274177271078438444451341086442840488878

QX2 = 5884313309747229435247839530708587960050885725582284116048011783458225699339146927030256583602089640166049520035696*i + 6487316124500084985166113682882741299262804186710219995448070546730483373888500555851433082562169723105515745417720
QZ2 = 2344299684487094017758169285916720399543408430521513262182487050829176811348749390037305690289939889760454567313561*i + 7918065288637721663962886631407099377614344562807773614718593804590532289115712549138055733038018974652008260325280

PmQX2 = 45743871459202355848219626549813319978193892209309387587476227928388558327910784655862188487531456759290429262853*i + 1030449633797366614976422742972278339056828145036418125605574806472360201845589384462028665090803565509889747149191
PmQZ2 = 1381565443616288901952989095086949837819520497291027960167436120995138196630248310813049175345075535968296774750862*i + 2723494836153857916516909071240798739011532353535789611317809923200834298173608729220713243630239918282831400463791

A3_check = 7699111961201068875849057075735030638265049383663167904054184008166045870455306615734693307616905183192906059307645*i + 9801039118747212051005256821753249426960743982360077995689684836765623897948660990779357718377949839193289861524986
A4_check = 7102836603142932653148826753844949958196416261810681428739598437629750431518654746535245934059526424912893545183695*i + 2215256555090867642327107655771932802458346873251863327744315389546767808956358139976578364706094247219284838985038

# fmt: on

# Create the Curve types (Has A and A24)
E1 = Curve(A1, F.one(), A1 + 2, F(4))
E2 = Curve(A2, F.one(), A2 + 2, F(4))

# Pack the basis together
B1 = BasisX(PX1, PZ1, QX1, QZ1, PmQX1, PmQZ1)
B2 = BasisX(PX2, PZ2, QX2, QZ2, PmQX2, PmQZ2)

# Ensure the output from the chain matches
domain = (E1, E2)
extra_kernel = (B1, B2)
exact_kernel = remove_extra_torsion(E1, E2, B1, B2)

# Ensure the orders of the kernel match what we expect
EE1 = EllipticCurve([0, E1.A / E1.C, 0, 1, 0])
EE2 = EllipticCurve([0, E2.A / E2.C, 0, 1, 0])
assert EE1.lift_x(B1.PX / B1.PZ) * 2 ** (length) != EE1(0)
assert EE2.lift_x(B2.PX / B2.PZ) * 2 ** (length) != EE2(0)
assert EE1.lift_x(B1.PX / B1.PZ) * 2 ** (length + 2) == EE1(0)
assert EE2.lift_x(B2.PX / B2.PZ) * 2 ** (length + 2) == EE2(0)
_B1, _B2 = exact_kernel
assert EE1.lift_x(_B1.PX / _B1.PZ) * 2 ** (length) == EE1(0)
assert EE2.lift_x(_B2.PX / _B2.PZ) * 2 ** (length) == EE2(0)

xP1 = PointX(PX1, PZ1)
xQ1 = PointX(QX1, QZ1)
xPmQ1 = PointX(PmQX1, PmQZ1)
xP2 = PointX(PX2, PZ2)
zero = PointX(F.one(), F.zero())
input_images_two = [(xP1, zero), (zero, xP2)]
input_images_three = [(xP1, zero), (xQ1, zero), (xPmQ1, zero)]


for isogeny_algorithm in [
    theta_isogeny_chain_4x2x1,
    theta_isogeny_chain_8x1x1,
    theta_isogeny_chain_with_torsion_4x2x1,
    theta_isogeny_chain_with_torsion_8x1x1,
    theta_isogeny_chain_verification_8x1x1,
]:
    for randomise in [True, False]:
        for input_images in [input_images_two, input_images_three]:
            if "with_torsion" in isogeny_algorithm.__name__:
                kernel = extra_kernel
            else:
                kernel = exact_kernel

            # Handle different input and output for verification
            if "verification" in isogeny_algorithm.__name__:
                # We don't use randomise or images in verification
                if randomise or len(input_images) == 2:
                    continue
                (A3, C3), (A4, C4) = isogeny_algorithm(domain, extra_kernel, length)
                images = []
            else:
                (A3, C3), (A4, C4), images = isogeny_algorithm(
                    domain, kernel, length, input_images, randomise
                )

            # Ensure codomain matches
            if "verification" in isogeny_algorithm.__name__:
                E3 = EllipticCurve(F, [0, A3 / C3, 0, 1, 0])
                E3_check = EllipticCurve(F, [0, A3_check, 0, 1, 0])
                assert E3.j_invariant() == E3_check.j_invariant()

                E4 = EllipticCurve(F, [0, A4 / C4, 0, 1, 0])
                E4_check = EllipticCurve(F, [0, A4_check, 0, 1, 0])
                assert E4.j_invariant() == E4_check.j_invariant()
            elif "with_torsion" in isogeny_algorithm.__name__:
                if not randomise:
                    assert A3 / C3 == A3_check
                    assert A4 / C4 == A4_check
                else:
                    E3 = EllipticCurve(F, [0, A3 / C3, 0, 1, 0])
                    E3_check = EllipticCurve(F, [0, A3_check, 0, 1, 0])
                    assert E3.j_invariant() == E3_check.j_invariant()

                    E4 = EllipticCurve(F, [0, A4 / C4, 0, 1, 0])
                    E4_check = EllipticCurve(F, [0, A4_check, 0, 1, 0])
                    assert E4.j_invariant() == E4_check.j_invariant()
            else:
                # Sign ambiguity from sqrt in this case means we're only
                # isomorphic to the concrete test here, regardless of randomise
                E3 = EllipticCurve(F, [0, A3 / C3, 0, 1, 0])
                E3_check = EllipticCurve(F, [0, A3_check, 0, 1, 0])

                E4 = EllipticCurve(F, [0, A4 / C4, 0, 1, 0])
                E4_check = EllipticCurve(F, [0, A4_check, 0, 1, 0])

                j_check = [E3_check.j_invariant(), E4_check.j_invariant()]
                assert E3.j_invariant() in j_check
                assert E4.j_invariant() in j_check

            # No images to check
            if "verification" in isogeny_algorithm.__name__:
                pass

            elif len(input_images) == 2:
                (xP3, xP4), (xQ3, xQ4) = images

                P3 = lift_to_curve(A3 / C3, xP3)
                P4 = lift_to_curve(A4 / C4, xP4)
                Q3 = lift_to_curve(A3 / C3, xQ3)
                Q4 = lift_to_curve(A4 / C4, xQ4)

                assert is_four_torsion(P3 + Q3) or is_four_torsion(P3 - Q3)
                assert is_four_torsion(P4 + Q4) or is_four_torsion(P4 - Q4)

            elif len(input_images) == 3:
                (xP3, xP4), (xQ3, xQ4), (xR3, xR4) = images

                P3 = lift_to_curve(A3 / C3, xP3)
                P4 = lift_to_curve(A4 / C4, xP4)
                Q3 = lift_to_curve(A3 / C3, xQ3)
                Q4 = lift_to_curve(A4 / C4, xQ4)
                R3 = lift_to_curve(A3 / C3, xR3)
                R4 = lift_to_curve(A4 / C4, xR4)

                assert R3.xy()[0] in [(P3 - Q3).xy()[0], (P3 + Q3).xy()[0]]
                assert R4.xy()[0] in [(P4 - Q4).xy()[0], (P4 + Q4).xy()[0]]

            print(f"test {isogeny_algorithm.__name__} passed")

print("All tests passed")
