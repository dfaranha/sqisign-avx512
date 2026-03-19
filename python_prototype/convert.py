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
    if n is not None:
        assert len(limbs) <= n
        pad = [0] * (n - len(limbs))
        limbs += pad

    return limbs


def fp2_to_le(x, w, n=None):
    """
    Convert an element of Fp2 into two arrays of w-bit words
    """
    if isinstance(x, int):
        coeffs = [x, 0]
    else:
        coeffs = x.polynomial().coefficients(sparse=False)
    re_limbs = int_to_le(coeffs[0], w, n)
    im_limbs = int_to_le(coeffs[1], w, n)
    return re_limbs, im_limbs


def print_fp(x):
    limbs = int_to_le(x, 64)
    txt = "{ "
    for i in range(6):
        if i < len(limbs):
            val = limbs[i]
        else:
            val = 0
        txt += f"0x{val:x}"
        if i < 5:
            txt += ", "
    txt += " }"
    print(txt)


def print_fp2(x):
    if isinstance(x, int):
        print_fp(x)
        print_fp(0)
        return
    coeffs = x.polynomial().coefficients(sparse=False)
    for i in range(2):
        if i >= len(coeffs):
            print_fp(0)
        else:
            print_fp(coeffs[i])


def print_fp2_4x2x1(P):
    print_fp2(P.x)
    print_fp2(P.y)
    print_fp2(P.z)
    print_fp2(P.w)


def print_fp2_8x1x1(P):
    print_fp2(P.x1)
    print_fp2(P.y1)
    print_fp2(P.z1)
    print_fp2(P.w1)
    print_fp2(P.x2)
    print_fp2(P.y2)
    print_fp2(P.z2)
    print_fp2(P.w2)


def print_theta_pair(P):
    print_fp2(P.x1)
    print_fp2(P.x2)
    print_fp2(P.y1)
    print_fp2(P.y2)
    print_fp2(P.z1)
    print_fp2(P.z2)
    print_fp2(P.w1)
    print_fp2(P.w2)


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
        avx512_blobs.append(
            [a_re[i], a_im[i], b_re[i], b_im[i], c_re[i], c_im[i], d_re[i], d_im[i]]
        )

    return avx512_blobs


def theta_to_avx512_8_1_1(P, Q):
    """
    Given two theta points (a : b : c : d) and (a' : b' : c' : d') convert this
    to a data blob suitible for the AVX512 operations
    """
    # Extract out the coordinates from the point
    a, b, c, d = P
    x, y, z, w = Q

    # For ease, we want all arrays to have the same number of words
    p = a.parent().characteristic()
    n = ceil(p.nbits() / 51)

    # First we convert each coordinate to le_limbs
    a_re, a_im = fp2_to_le(a, 51, n)
    b_re, b_im = fp2_to_le(b, 51, n)
    c_re, c_im = fp2_to_le(c, 51, n)
    d_re, d_im = fp2_to_le(d, 51, n)
    x_re, x_im = fp2_to_le(x, 51, n)
    y_re, y_im = fp2_to_le(y, 51, n)
    z_re, z_im = fp2_to_le(z, 51, n)
    w_re, w_im = fp2_to_le(w, 51, n)

    avx512_blobs = []
    # Now we create a blob of data for the 8 real values then the 8 imaginary values
    for i in range(n):
        avx512_blobs.append(
            [a_re[i], b_re[i], c_re[i], d_re[i], x_re[i], y_re[i], z_re[i], w_re[i]]
        )
    for i in range(n):
        avx512_blobs.append(
            [a_im[i], b_im[i], c_im[i], d_im[i], x_im[i], y_im[i], z_im[i], w_im[i]]
        )

    return avx512_blobs


def avx512_blobs_4_2_1_to_theta(F, avx_blobs):
    a_re, a_im = 0, 0
    b_re, b_im = 0, 0
    c_re, c_im = 0, 0
    d_re, d_im = 0, 0
    for i, avx in enumerate(avx_blobs):
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


def avx512_blobs_8_1_1_to_theta(F, avx_blobs):
    a_re, b_re, c_re, d_re, x_re, y_re, z_re, w_re = 0, 0, 0, 0, 0, 0, 0, 0
    a_im, b_im, c_im, d_im, x_im, y_im, z_im, w_im = 0, 0, 0, 0, 0, 0, 0, 0

    # Parse the real values
    for i in range(8):
        avx = avx_blobs[i]
        a_re += avx[0] * (1 << (51 * i))
        b_re += avx[1] * (1 << (51 * i))
        c_re += avx[2] * (1 << (51 * i))
        d_re += avx[3] * (1 << (51 * i))
        x_re += avx[4] * (1 << (51 * i))
        y_re += avx[5] * (1 << (51 * i))
        z_re += avx[6] * (1 << (51 * i))
        w_re += avx[7] * (1 << (51 * i))

    # Parse the imaginary values
    for i in range(8):
        avx = avx_blobs[i + 8]
        a_im += avx[0] * (1 << (51 * i))
        b_im += avx[1] * (1 << (51 * i))
        c_im += avx[2] * (1 << (51 * i))
        d_im += avx[3] * (1 << (51 * i))
        x_im += avx[4] * (1 << (51 * i))
        y_im += avx[5] * (1 << (51 * i))
        z_im += avx[6] * (1 << (51 * i))
        w_im += avx[7] * (1 << (51 * i))

    # Convert to Fp2 elements
    a = F([a_re, a_im])
    b = F([b_re, b_im])
    c = F([c_re, c_im])
    d = F([d_re, d_im])
    x = F([x_re, x_im])
    y = F([y_re, y_im])
    z = F([z_re, z_im])
    w = F([w_re, w_im])

    return (a, b, c, d), (x, y, z, w)


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
