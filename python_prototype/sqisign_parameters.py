from sage.rings.finite_rings.finite_field_constructor import GF
from sage.rings.integer_ring import ZZ

SQISIGN_PARAMETERS_LEVEL_1 = {
    "security_bits": 128,
    "cofactor": 3,
    "f": 324,
    "response_len": 196,
    "fp_encoded_len": 41,
    "pk_len": 83,
    "sk_len": 270,
    "sig_len": 200,
}

SQISIGN_PARAMETERS_LEVEL_3 = {
    "security_bits": 192,
    "cofactor": 27,
    "f": 500,
    "response_len": 302,
    "fp_encoded_len": 64,
    "pk_len": 129,
    "sk_len": 417,
    "sig_len": 306,
}

SQISIGN_PARAMETERS_LEVEL_5 = {
    "security_bits": 256,
    "cofactor": 17,
    "f": 664,
    "response_len": 400,
    "fp_encoded_len": 84,
    "pk_len": 169,
    "sk_len": 549,
    "sig_len": 406,
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
