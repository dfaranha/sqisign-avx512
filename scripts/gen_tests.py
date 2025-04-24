from convert import theta_to_avx512_8_1_1, print_avx512_blobs
from theta_structrure import hadamard_transform, squaring_transform, pairwise_multiplication
from sage.all import GF, proof, set_random_seed

proof.all(False)
set_random_seed(0)

p = 65*2**376 - 1
F = GF(p**2, names="i", modulus=[1,0,1])

# For now, use totally random points, later we'll do an isogeny chain
P = [F.random_element() for _ in range(4)]
Q = [F.random_element() for _ in range(4)]

# First let's us check the python functions do what we expect
# H(H(P)) should be projectively equal (up to a factor of 4)
P4 = hadamard_transform(hadamard_transform(P))
for (ai, bi) in zip(P, P4):
    assert 4*ai == bi

# P * P should be the same as S(P)
PP = pairwise_multiplication(P, P)
PS = squaring_transform(P)
for (ai, bi) in zip(PP, PS):
    assert ai == bi

# Squaring should be done at the coordinate level
for (ai, bi) in zip(P, PS):
    assert ai**2 == bi


# Now generate values for the C code
P_had = hadamard_transform(P)
P_sqr = squaring_transform(P)
PQ = pairwise_multiplication(P, Q)

P_avx512 = theta_to_avx512_8_1_1(P)
Q_avx512 = theta_to_avx512_8_1_1(Q)
P_had_avx512 = theta_to_avx512_8_1_1(P_had)
P_sqr_avx512 = theta_to_avx512_8_1_1(P_sqr)
PQ_avx512 = theta_to_avx512_8_1_1(PQ)

print_avx512_blobs(P_avx512, "P")
print_avx512_blobs(Q_avx512, "Q")

print_avx512_blobs(P_had_avx512, "P_had")
print_avx512_blobs(P_sqr_avx512, "Q_sqr")
print_avx512_blobs(PQ_avx512, "PQ")
