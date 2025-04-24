from math import ceil

def int_to_le(x, w, n=None):
    """
    Convert an integer into an array of w-bit words if n is not none,
    0-pad the list of limbs to have length n
    """
    mask = (1 << w) - 1
    limbs = []

    x = int(x)
    while x:
        limbs.append(x & mask)
        x >>= w

    # Optionally pad limbs to length n
    if not n is None:
        assert len(limbs) <= n
        pad = [0] * (n - len(limbs))
        limbs += pad

    return limbs

def fp2_to_le(x, w, n=None):
    """
    Convert an element of Fp2 into two arrays of w-bit words
    """
    re, im = x
    re_limbs = int_to_le(re, w, n)
    im_limbs = int_to_le(im, w, n)
    return re_limbs, im_limbs

def theta_to_avx512_4_2_1(P):
    """
    Given a theta point (a : b : c : d) as four elements in Fp2 convert this
    to a data blob suitible for the AVX512 operations
    """
    # Extract out the coordinates from the point
    a, b, c, d = P

    # For ease, we want all arrays to have the same number of words
    p = a.parent().characteristic()
    n = ceil(p.nbits() / 51)

    # First we convert each coordinate to le_limbs
    a_re, a_im = fp2_to_le(a, 51, n)
    b_re, b_im = fp2_to_le(b, 51, n)
    c_re, c_im = fp2_to_le(c, 51, n)
    d_re, d_im = fp2_to_le(d, 51, n)

    avx512_blobs = []
    # Now we create a blob of data where the first 8 51 bit values are the first word
    # of the real and imaginary parts of a, b, c, d and so on
    for i in range(n):
        avx512_blobs.append([a_re[i], a_im[i], b_re[i], b_im[i], c_re[i], c_im[i], d_re[i], d_im[i]])

    return avx512_blobs

def avx512_blobs_4_2_1_to_theta(F, avx_blobs):
    a_re, a_im = 0, 0
    b_re, b_im = 0, 0
    c_re, c_im = 0, 0
    d_re, d_im = 0, 0
    for (i, avx) in enumerate(avx_blobs):
        a_re += avx[0] * (1 << (51 * i))
        a_im += avx[1] * (1 << (51 * i))
        b_re += avx[2] * (1 << (51 * i))
        b_im += avx[3] * (1 << (51 * i))
        c_re += avx[4] * (1 << (51 * i))
        c_im += avx[5] * (1 << (51 * i))
        d_re += avx[6] * (1 << (51 * i))
        d_im += avx[7] * (1 << (51 * i))
    a = F([a_re, a_im])
    b = F([b_re, b_im])
    c = F([c_re, c_im])
    d = F([d_re, d_im])
    return (a, b, c, d)

def print_avx512(avx512, indent=True):
    output = ", ".join(f"0x{num:013X}" for num in avx512)
    output = "{ " + output + " },"
    if indent:
        output = "    " + output
    print(output)

def print_avx512_blobs(avx_blobs, name):
    print(f"{name} = {{")
    for avx in avx_blobs:
        print_avx512(avx)
    print("}")
