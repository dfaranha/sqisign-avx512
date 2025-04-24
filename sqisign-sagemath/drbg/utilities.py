
def xor_bytes(a, b):
    """
    Given two arrays of bytes return a xor b
    """
    assert len(a) == len(b)
    return bytes(x ^ y for x, y in zip(a, b))
