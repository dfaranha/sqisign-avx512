from sympy import isprime
from params import get_limb_params

def get_pbits(num_limbs, limb_len, last_limb_len):
    return (num_limbs-1)*limb_len + last_limb_len

def find_prime_for_params(num_limbs, limb_len, last_limb_len):
    pbits = get_pbits(num_limbs, limb_len, last_limb_len)
    e = (num_limbs - 1) * limb_len

    # c must satisfy: p = c*2^e - 1 has exactly pbits bits
    # i.e. 2^(pbits-1) <= c*2^e - 1 < 2^pbits
    # so c = 2^(pbits-e-1) ... 2^(pbits-e) - 1, we want odd c
    c_min = (1 << (pbits - 1 - e)) + 1  # = ceil(2^(pbits-1) / 2^e) approximately
    c_max = (1 << (pbits - e)) - 1

    for c in range(c_min | 1, c_max + 1, 2):  # odd c only
        p = c * (1 << e) - 1
        if isprime(p):
            return p, c, e

    return None

def find_prime_for_case(num_limbs, limb_len, last_limb_len):
    p, c, e = find_prime_for_params(num_limbs, limb_len, last_limb_len)
    _num_limbs, _limb_len, _last_limb_len, _ = get_limb_params(get_pbits(num_limbs, limb_len, last_limb_len), e, c)
    assert _num_limbs == num_limbs and _limb_len == limb_len and _last_limb_len == last_limb_len
    print(f"limb_len = {limb_len}, last_limb_len = {last_limb_len}: c = {c}, e = {e}")

# cases for mul_strat
find_prime_for_case(num_limbs=4, limb_len=51, last_limb_len=51)
find_prime_for_case(num_limbs=5, limb_len=51, last_limb_len=50)
find_prime_for_case(num_limbs=4, limb_len=51, last_limb_len=49)
find_prime_for_case(num_limbs=5, limb_len=50, last_limb_len=50)
find_prime_for_case(num_limbs=4, limb_len=50, last_limb_len=49)
find_prime_for_case(num_limbs=5, limb_len=50, last_limb_len=48)
# last case limb_len = last_limb_len < 50
find_prime_for_case(num_limbs=4, limb_len=49, last_limb_len=49)
# two extra primes because why not
find_prime_for_case(num_limbs=5, limb_len=49, last_limb_len=48)
find_prime_for_case(num_limbs=4, limb_len=49, last_limb_len=47)

print(f"mersenne prime: c = 1, e = 607")

