from sage.rings.finite_rings.finite_field_constructor import GF
from sage.rings.integer_ring import ZZ

SQISIGN_PARAMETERS_LEVEL_1 = {
    "security_bits": 128,
    "cofactor": 5,
    "f": 248,
    "response_length": 126,
    "hash_iterations": 64,
    "fp_encoded_len": 32,
    "pk_len": 65,
    "sk_len": 353,
    "sig_len": 148,
}

SQISIGN_PARAMETERS_LEVEL_3 = {
    "security_bits": 192,
    "cofactor": 65,
    "f": 376,
    "response_length": 192,
    "hash_iterations": 256,
    "fp_encoded_len": 48,
    "pk_len": 97,
    "sk_len": 529,
    "sig_len": 224,
}

SQISIGN_PARAMETERS_LEVEL_5 = {
    "security_bits": 256,
    "cofactor": 27,
    "f": 500,
    "response_length": 253,
    "hash_iterations": 512,
    "fp_encoded_len": 64,
    "pk_len": 129,
    "sk_len": 701,
    "sig_len": 292,
}

SQISIGN_PARAMETERS = {
    1: SQISIGN_PARAMETERS_LEVEL_1,
    3: SQISIGN_PARAMETERS_LEVEL_3,
    5: SQISIGN_PARAMETERS_LEVEL_5,
}


def generate_finite_field(parameters):
    """
    Generate the finite field based on the NIST
    security level
    """
    f = parameters["f"]
    cofactor = parameters["cofactor"]

    # Create the extension field GF(p^2) with modulus
    # x^2 + 1
    p = ZZ(cofactor * 2**f - 1)
    assert p.is_prime()
    assert p % 4 == 3

    return p, GF((p, 2), names="i", modulus=[1, 0, 1])
