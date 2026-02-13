from sage.misc.randstate import set_random_seed
from sage.rings.finite_rings.finite_field_constructor import GF

from avx_types import AVX_4x2x1
from avx_arithmetic import avx_421_mul, avx_421_sqr
from convert import (
    theta_to_avx512_4_2_1,
    print_avx512_blobs,
    avx512_blobs_4_2_1_to_theta,
    theta_to_avx512_8_1_1,
    avx512_blobs_8_1_1_to_theta,
)
from theta_structure import hadamard

set_random_seed(0)

p = 65 * 2**376 - 1
F = GF((p, 2), names="i", modulus=[1, 0, 1])

# For now, use totally random points, later we'll do an isogeny chain
P = AVX_4x2x1(*[F.random_element() for _ in range(4)])
Q = AVX_4x2x1(*[F.random_element() for _ in range(4)])
C = AVX_4x2x1(*[F.random_element() for _ in range(4)])

# First let's us check the python functions do what we expect
# H(H(P)) should be projectively equal (up to a factor of 4)
P4 = hadamard(hadamard(P))
for ai, bi in zip(P, P4):
    assert 4 * ai == bi

# P * P should be the same as S(P)
PP = avx_421_mul(P, P)
PS = avx_421_sqr(P)
for ai, bi in zip(PP, PS):
    assert ai == bi

# Squaring should be done at the coordinate level
for ai, bi in zip(P, PS):
    assert ai**2 == bi

# Now generate values for the C code
P_had = hadamard(P)
P_sqr = avx_421_sqr(P)
Q_had = hadamard(Q)
Q_sqr = avx_421_sqr(Q)
PC = avx_421_mul(P, C)
QC = avx_421_mul(Q, C)

# Package four coordinates into n AVX blobs, where n is ceil(p.nbits() / 51)
P_avx512 = theta_to_avx512_4_2_1(P)
Q_avx512 = theta_to_avx512_4_2_1(Q)
P_had_avx512 = theta_to_avx512_4_2_1(P_had)
P_sqr_avx512 = theta_to_avx512_4_2_1(P_sqr)
PC_avx512 = theta_to_avx512_4_2_1(PC)

# Finally reverse decoding to make sure things match...
assert P == avx512_blobs_4_2_1_to_theta(F, P_avx512)
assert Q == avx512_blobs_4_2_1_to_theta(F, Q_avx512)
assert P_had == avx512_blobs_4_2_1_to_theta(F, P_had_avx512)
assert P_sqr == avx512_blobs_4_2_1_to_theta(F, P_sqr_avx512)
assert PC == avx512_blobs_4_2_1_to_theta(F, PC_avx512)

# Print output for C code
print("4x2x1 output:")
print_avx512_blobs(P_avx512, "P")
print_avx512_blobs(Q_avx512, "Q")
print_avx512_blobs(P_had_avx512, "P_had")
print_avx512_blobs(P_sqr_avx512, "Q_sqr")
print_avx512_blobs(PC_avx512, "PC")
print()

PQ_avx512 = theta_to_avx512_8_1_1(P, Q)
PQ_had_avx512 = theta_to_avx512_8_1_1(P_had, Q_had)
PQ_sqr_avx512 = theta_to_avx512_8_1_1(P_sqr, Q_sqr)
PCQC_avx512 = theta_to_avx512_8_1_1(PC, QC)

# Finally reverse decoding to make sure things match...
assert (P, Q) == avx512_blobs_8_1_1_to_theta(F, PQ_avx512)
assert (P_had, Q_had) == avx512_blobs_8_1_1_to_theta(F, PQ_had_avx512)
assert (P_sqr, Q_sqr) == avx512_blobs_8_1_1_to_theta(F, PQ_sqr_avx512)
assert (PC, QC) == avx512_blobs_8_1_1_to_theta(F, PCQC_avx512)

# Print output for C code
print("8x1x1 output:")
print_avx512_blobs(PQ_avx512, "PQ")
print_avx512_blobs(PQ_had_avx512, "PQ_had")
print_avx512_blobs(PQ_sqr_avx512, "PQ_sqr")
print_avx512_blobs(PCQC_avx512, "PCQC_avx512")
print()
