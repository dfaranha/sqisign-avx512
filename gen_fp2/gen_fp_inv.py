from file_writer import (int_to_limbs, limbs_to_hex)
from math import ceil
from params import get_inv_params

def inv_macros(f, pp, v):
    f.line("#define adc(cc, a, b, d) _addcarry_u64(cc, a, b, (unsigned long long *)(void *)d)")
    f.line("#define sbb(cc, a, b, d) _subborrow_u64(cc, a, b, (unsigned long long *)(void *)d)")
    f.line("#define lzcnt(x) ((x) == 0 ? 64 : (int64_t)__builtin_clzll(x))")
    f.line()

def sgnw(f, pp, v):
    f.comment("Outputs 0xffffffffffffffff if src is negative, else 0")
    f.start("uint64_t sgnw(const uint64_t src)")
    f.line("return (uint64_t)(*(int64_t *)&src >> 63);")
    f.end()

def is_zero64(f, pp, v):
    f.comment("Assumes input is fully reduced.")
    f.start("uint64_t is_zero64(const uint64_t* src)")

    f.line("uint64_t t = src[0];")
    for i in range(1, pp.num_words64):
        f.line(f"t |= src[{i}];")
    f.line()
    f.line("t = (t | -t) >> 63; // 0 if t = 0, 1 otherwise")
    f.line("return t-1;")

    f.end()

def select64(f, pp, v):
    f.comment("If mask == 0xffffffffffffffff, a1 is copied to res")
    f.start("void select64(uint64_t* res, const uint64_t* a0, const uint64_t* a1, const uint64_t mask)")
    
    for i in range(pp.num_words64):
        f.line(f"res[{i}] = a0[{i}] ^ (mask & (a0[{i}] ^ a1[{i}]));")

    f.end()

def umul_64(f, pp, v):
    f.comment("(lo, hi) = x*y as integers")
    f.start("void umul_64(uint64_t* lo, uint64_t* hi, const uint64_t x, const uint64_t y)")

    f.line("unsigned __int128 t;")
    f.line("t = (unsigned __int128)(x) * (unsigned __int128)(y);")
    f.line("*lo = (uint64_t) t;")
    f.line("*hi = (uint64_t)(t >> 64);")

    f.end()

def umul_add_64(f, pp, v):
    f.comment("(lo, hi) = x*y + z as integers")
    f.start("void umul_add_64(uint64_t* lo, uint64_t* hi, const uint64_t x, const uint64_t y, const uint64_t z)")

    f.line("unsigned __int128 t;")
    f.line("t = (unsigned __int128)(x) * (unsigned __int128)(y) + (unsigned __int128)(uint64_t)(z);")
    f.unindent()
    f.line("*lo = (uint64_t) t;")
    f.line("*hi = (uint64_t)(t >> 64);")

    f.end()

def umulx2_64(f, pp, v):
    f.comment("(lo, hi) = x1*y1 + x2*y2 as integers")
    f.start("void umulx2_64(uint64_t* lo, uint64_t* hi, const uint64_t x1, const uint64_t y1, const uint64_t x2, const uint64_t y2)")

    f.line("unsigned __int128 t;")
    f.line("t = (unsigned __int128)(x1) * (unsigned __int128)(y1) + (unsigned __int128)(x2) * (unsigned __int128)(y2);")
    f.line("*lo = (uint64_t) t;")
    f.line("*hi = (uint64_t)(t >> 64);")

    f.end()

def umulx2_add_64(f, pp, v):
    f.comment("(lo, hi) = x1*y1 + x2*y2 + z as integers")
    f.start("void umulx2_add_64(uint64_t* lo, uint64_t* hi, const uint64_t x1,\n    const uint64_t y1, const uint64_t x2, const uint64_t y2, const uint64_t z)")

    f.line("unsigned __int128 t;")
    f.line("t = (unsigned __int128)(x1) * (unsigned __int128)(y1) + (unsigned __int128)(x2) * (unsigned __int128)(y2)")
    f.indent()
    f.line("+ (unsigned __int128)(uint64_t)(z);")
    f.unindent()
    f.line("*lo = (uint64_t) t;")
    f.line("*hi = (uint64_t)(t >> 64);")

    f.end()

def neg64(f, pp, v):
    f.comment("Assumes input is fully reduced.")
    f.start("void neg64(uint64_t* res, const uint64_t* src)")

    f.line(f"uint64_t t[{pp.num_words64}];")
    f.line(f"unsigned char cc;")
    f.line(f"const uint64_t nz = ~is_zero64(src);")
    f.line()

    f.comment("p - a")
    p_words_hex = limbs_to_hex(int_to_limbs(pp.p, 64))
    f.line(f"cc = sbb(0, (uint64_t) {p_words_hex[0]}, src[0], t);")
    for i in range(1, pp.num_words64):
        f.line(f"cc = sbb(cc, (uint64_t) {p_words_hex[i]}, src[{i}], t+{i});")
    f.line()

    for i in range(pp.num_words64):
        f.line(f"res[{i}] = t[{i}] & nz;")

    f.end()

def lin64(f, pp, v):
    f.comment("res = f*u + g*v mod p")
    f.comment("Assumes f,g < 2^62")
    f.start("void lin64(uint64_t* res, const uint64_t* u, const uint64_t* v, uint64_t f, uint64_t g)")

    n = pp.num_words64
    f.line(f"uint64_t t[{n}], tu[{n}], tv[{n}], hi;")
    f.line("unsigned char cc;")
    f.line()

    f.comment("If f is negative, set f := abs(f) and negate u")
    f.line("uint64_t sf = sgnw(f);")
    f.line("f = (f ^ sf) - sf;")
    f.line("neg64(tu, u);")
    f.line("select64(tu, u, tu, sf);")
    f.line()

    f.comment("If g is negative, set g := abs(g) and negate v")
    f.line("uint64_t sg = sgnw(g);")
    f.line("g = (g ^ sg) - sg;")
    f.line("neg64(tv, v);")
    f.line("select64(tv, v, tv, sg);")

    f.comment("t = f*u + g*v over the integers")
    f.line("umulx2_64(t, &hi, tu[0], f, tv[0], g);")
    for i in range(1, n):
        f.line(f"umulx2_add_64(t+{i}, &hi, tu[{i}], f, tv[{i}], g, hi);")
    f.line()

    f.comment("Reduction")
    f.comment("t = L + 2^e H")
    f.comment("Since p = c* 2^e - 1, 2^e = 1/c mod p")
    f.comment("We fold the output as t = L + 2^e * (H mod c) + 2^e * floor(H/c)")
    f.line()
    # when f,g < 2^62 and u,v are fully reduced mod p, |f|*u + |g|*v < 2^63*p
    # L + 2^e * (H mod c) <= 2^e - 1 + 2^e * (c-1) = c*2^e - 1 = p
    # floor(H/c) <= 2^63
    # under the assumption that p > 2^64, the folded result is < 2p.
    split_word = pp.e // 64
    split_bit = pp.e % 64
    c_bits = int(pp.c).bit_length()
    #helper function for accessing correct variable
    def src(idx):
        if idx < n:
            return f"t[{idx}]"
        elif idx == n:
            return "hi"
        else:
            return "0"
    
    f.comment(f"Compute H in (h0, h1) and L in (t[0], ..., t[{split_word}]):")
    if split_bit == 0:
        f.line(f"uint64_t h0 = {src(split_word)};")
        f.line(f"uint64_t h1 = {src(split_word + 1)};")
    else:
        f.line(f"uint64_t h0 = ({src(split_word)} >> {split_bit})"
               f" | ({src(split_word + 1)} << {64 - split_bit});")
        h1_expr = f"({src(split_word + 1)} >> {split_bit})"
        if split_word + 2 <= n:
            h1_expr += f" | ({src(split_word + 2)} << {64 - split_bit})"
        f.line(f"uint64_t h1 = {h1_expr};")
        # clear top (64-split_bit) bits from t
        f.line(f"t[{split_word}] &= 0x{(1 << split_bit) - 1:016X}ULL;")
    f.line()

    f.comment("Divide H by c")
    f.line(f"__uint128_t H_full = ((__uint128_t)h1 << 64) | h0;")
    f.line(f"uint64_t quo = (uint64_t)(H_full / 0x{pp.c:x}ULL);")
    f.line(f"uint64_t rem = (uint64_t)(H_full % 0x{pp.c:x}ULL);")
    f.line()

    f.comment("t = L + quo + rem * 2^e")
    f.line(f"cc = adc(0, t[0], quo, &t[0]);")
    #Propagate carry
    for i in range(1, split_word):
        f.line(f"cc = adc(cc, t[{i}], 0, &t[{i}]);")

    # Handle rem and cc depending on splitting
    if split_bit == 0:
        f.line(f"t[{split_word}] = rem + (uint64_t)cc;")
        zero_start = split_word + 1
    elif split_bit + c_bits <= 64:
        f.line(f"adc(cc, t[{split_word}], rem << {split_bit}, &t[{split_word}]);")
        zero_start = split_word + 1
    else:
        f.line(f"cc = adc(cc, t[{split_word}], rem << {split_bit}, &t[{split_word}]);")
        f.line(f"t[{split_word + 1}] = (rem >> {64 - split_bit}) + (uint64_t)cc;")
        zero_start = split_word + 2

    for i in range(zero_start, n):
        f.line(f"t[{i}] = 0;")
    f.line()

    f.comment("Finally, if t > p, we subtract p to reduce fully.")
    p_words_hex = limbs_to_hex(int_to_limbs(pp.p, 64))
    f.line(f"cc = sbb(0, t[0], {p_words_hex[0]}ULL, tu);")
    for i in range(1, n):
        f.line(f"cc = sbb(cc, t[{i}], {p_words_hex[i]}ULL, tu + {i});")
    f.comment("If no borrow (t >= p), use tu; otherwise keep t")
    f.line("uint64_t mask = -(uint64_t)cc;")
    f.line("select64(res, tu, t, mask);")

    f.end()

def lindiv31abs_64(f, pp, v):
    n = pp.num_words64

    f.comment("d = abs(floor((a*fc + b*gc) / 2^31))")
    f.comment("Coefficients fc, gc are signed, |fc|,|gc| <= 2^31.")
    f.comment("Returns 0xFFFFFFFFFFFFFFFF if a*fc + b*gc < 0, else 0.")
    f.start(f"uint64_t lindiv31abs(uint64_t* d, const uint64_t* a, const uint64_t* b, uint64_t fc, uint64_t gc)")

    f.line(f"uint64_t ta[{n + 1}], tb[{n + 1}], t[{n + 1}], hi;")
    f.line("unsigned char cc;")
    f.line()

    f.comment("fc = abs(fc), conditionally negate a")
    f.line("uint64_t sf = sgnw(fc);")
    f.line("fc = (fc ^ sf) - sf;")
    f.line(f"cc = sbb(0, a[0] ^ sf, sf, &ta[0]);")
    for i in range(1, n):
        f.line(f"cc = sbb(cc, a[{i}] ^ sf, sf, &ta[{i}]);")
    f.line(f"(void)sbb(cc, 0, 0, &ta[{n}]);")
    f.line()

    f.comment("gc = abs(gc), conditionally negate b")
    f.line("uint64_t sg = sgnw(gc);")
    f.line("gc = (gc ^ sg) - sg;")
    f.line(f"cc = sbb(0, b[0] ^ sg, sg, &tb[0]);")
    for i in range(1, n):
        f.line(f"cc = sbb(cc, b[{i}] ^ sg, sg, &tb[{i}]);")
    f.line(f"(void)sbb(cc, 0, 0, &tb[{n}]);")
    f.line()

    f.comment("Compute ta*fc + tb*gc as a signed integer")
    f.line(f"umulx2_64(&t[0], &hi, ta[0], fc, tb[0], gc);")
    for i in range(1, n):
        f.line(f"umulx2_add_64(&t[{i}], &hi, ta[{i}], fc, tb[{i}], gc, hi);")
    f.line(f"t[{n}] = hi - (ta[{n}] & fc) - (tb[{n}] & gc);")
    f.line()

    f.comment("Right-shift by 31 bits")
    for i in range(n):
        f.line(f"t[{i}] = (t[{i}] >> 31) | (t[{i + 1}] << 33);")
    f.line()

    f.comment("If result is negative, negate it")
    f.line(f"uint64_t s = sgnw(t[{n}]);")
    f.line(f"cc = sbb(0, t[0] ^ s, s, &d[0]);")
    for i in range(1, n - 1):
        f.line(f"cc = sbb(cc, t[{i}] ^ s, s, &d[{i}]);")
    f.line(f"(void)sbb(cc, t[{n - 1}] ^ s, s, &d[{n - 1}]);")
    f.line()
    f.line("return s;")

    f.end()

def fp_mul_64_oneshot(f, pp, v):
    n = pp.num_words64
    split_word = pp.e // 64
    split_bit = pp.e % 64
    extra_bits = 64 * n - pp.e
    c_shifted = pp.c << split_bit

    p_hex = limbs_to_hex(int_to_limbs(pp.p, 64))

    f.comment(f"Montgomery multiplication: res = a * b / 2^{64*n} mod p")
    f.comment(f"p = {pp.c} * 2^{pp.e} - 1")
    f.start("void fp_mul_64(uint64_t* res, const uint64_t* a, const uint64_t* b)")

    f.line(f"const uint64_t P[{n}] = {{")
    f.indent()
    for i in range(n):
        comma = "," if i < n - 1 else ""
        f.line(f"{p_hex[i]}ULL{comma}")
    f.unindent()
    f.line("};")
    f.line()

    f.line(f"uint64_t e[{2 * n}], lo, hi;")
    f.line(f"uint64_t s[{n}], h[{n + 1}];")
    f.line("unsigned char cc;")
    f.line()

    f.comment("Schoolbook multiplication: e = a * b")
    for i in range(n):
        if i == 0:
            f.line(f"umul_64(&e[0], &hi, a[0], b[0]);")
            for j in range(1, n):
                f.line(f"umul_add_64(&e[{j}], &hi, a[0], b[{j}], hi);")
            f.line(f"e[{n}] = hi;")
        else:
            f.line(f"umul_64(&lo, &hi, a[{i}], b[0]);")
            f.line(f"cc = adc(0, e[{i}], lo, &e[{i}]);")
            for j in range(1, n):
                f.line(f"umul_add_64(&lo, &hi, a[{i}], b[{j}], hi);")
                f.line(f"cc = adc(cc, e[{i + j}], lo, &e[{i + j}]);")
            f.line(f"e[{i + n}] = hi + (uint64_t)cc;")
        f.line()

    f.comment(f"Montgomery reduction")
    f.comment(f"m = -1/p mod 2^(64 * n) = {pp.c} * 2^{pp.e} + 1) ")
    f.comment(f"s = low(e) * m mod 2^(64 * n)")
    f.line(f"for (int i = 0; i < {n}; i++) s[i] = e[i];")
    f.line(f"s[{split_word}] += (uint64_t)((e[0] & 0x{(1 << extra_bits) - 1:X}ULL) * {pp.c}ULL) << {split_bit};")
    f.line()

    f.comment(f"h = s * ({pp.c} << {split_bit})")
    f.line(f"umul_64(&h[0], &hi, s[0], {c_shifted}ULL);")
    for i in range(1, n):
        f.line(f"umul_add_64(&h[{i}], &hi, s[{i}], {c_shifted}ULL, hi);")
    f.line(f"h[{n}] = hi;")
    f.line()

    f.comment("e += s*p = s*c*2^f - s")
    f.comment("Step A: e -= s")
    f.line("cc = 0;")
    f.line(f"for (int i = 0; i < {n}; i++) cc = sbb(cc, e[i], s[i], &e[i]);")
    f.line(f"for (int i = {n}; i < {2 * n}; i++) cc = sbb(cc, e[i], 0, &e[i]);")
    f.line()

    f.comment(f"Step B: e += h at word position {split_word}")
    f.line("cc = 0;")
    f.line(f"for (int i = 0; i <= {n}; i++) cc = adc(cc, e[{split_word} + i], h[i], &e[{split_word} + i]);")
    remaining = 2 * n - (split_word + n + 1)
    if remaining > 0:
        f.line(f"for (int i = {split_word + n + 1}; i < {2 * n}; i++) cc = adc(cc, e[i], 0, &e[i]);")
    f.line()

    f.comment("Conditional subtraction of p for full reduction")
    f.line(f"uint64_t r[{n}];")
    f.line("cc = 0;")
    f.line(f"for (int i = 0; i < {n}; i++) cc = sbb(cc, e[{n} + i], P[i], &r[i]);")
    f.line("uint64_t mask = -(uint64_t)cc;")
    f.line(f"for (int i = 0; i < {n}; i++) res[i] = (e[{n} + i] & mask) | (r[i] & ~mask);")

    f.end()

def fp_mul_64(f, pp, v):
    n = pp.num_words64
    e_val = pp.e
    split_word = e_val // 64
    split_bit = e_val % 64
    C = int(pp.c)
    c_bits = C.bit_length()

    # p = -1 mod 2^64  <=>  e >= 64, which makes p' = -p^{-1} mod 2^64 = 1 (u = s[i]).
    assert e_val >= 64, "Montgomery-friendly word reduction requires e >= 64"
    # Short product u*c must be a single 64x64->128 multiply.
    assert c_bits <= 64, "reduction assumes c fits in one 64-bit word"

    p_hex = limbs_to_hex(int_to_limbs(pp.p, 64))

    f.comment(f"Montgomery multiplication: res = a * b / 2^{64*n} mod p")
    f.comment(f"p = {pp.c} * 2^{pp.e} - 1")
    f.start("void fp_mul_64(uint64_t* res, const uint64_t* a, const uint64_t* b)")

    f.line(f"const uint64_t P[{n}] = {{")
    f.indent()
    for i in range(n):
        comma = "," if i < n - 1 else ""
        f.line(f"{p_hex[i]}ULL{comma}")
    f.unindent()
    f.line("};")
    f.line()

    f.line(f"uint64_t e[{2 * n}], u, lo, hi;")
    f.line("unsigned char cc;")
    f.line()

    f.comment("Schoolbook multiplication: e = a * b")
    for i in range(n):
        if i == 0:
            f.line("umul_64(&e[0], &hi, a[0], b[0]);")
            for j in range(1, n):
                f.line(f"umul_add_64(&e[{j}], &hi, a[0], b[{j}], hi);")
            f.line(f"e[{n}] = hi;")
        else:
            f.line(f"umul_64(&lo, &hi, a[{i}], b[0]);")
            f.line(f"cc = adc(0, e[{i}], lo, &e[{i}]);")
            for j in range(1, n):
                f.line(f"umul_add_64(&lo, &hi, a[{i}], b[{j}], hi);")
                f.line(f"cc = adc(cc, e[{i + j}], lo, &e[{i + j}]);")
            f.line(f"e[{i + n}] = hi + (uint64_t)cc;")
        f.line()

    f.comment("Montgomery reduction word-by-word")
    top = 2 * n - 1
    for i in range(n):
        base = split_word + i
        f.line(f"u = e[{i}];")
        f.line(f"umul_64(&lo, &hi, u, {C}ULL);")

        # (hi:lo) << split_bit, spread across up to three words starting at base.
        if split_bit == 0:
            words = [(base, "lo"), (base + 1, "hi")]
        else:
            sb = split_bit
            words = [
                (base,     f"lo << {sb}"),
                (base + 1, f"(lo >> {64 - sb}) | (hi << {sb})"),
                (base + 2, f"hi >> {64 - sb}"),
            ]
        # Words landing at index >= 2n are zero so we drop them
        words = [(idx, expr) for (idx, expr) in words if idx <= top]

        last_idx = base
        for k, (idx, expr) in enumerate(words):
            carry_in = "0" if k == 0 else "cc"
            f.line(f"cc = adc({carry_in}, e[{idx}], {expr}, &e[{idx}]);")
            last_idx = idx
        # Ripple the remaining carry up to the top word.
        for k in range(last_idx + 1, top + 1):
            f.line(f"cc = adc(cc, e[{k}], 0, &e[{k}]);")
        f.line()

    f.comment("Conditional subtraction of p for full reduction")
    f.line(f"uint64_t r[{n}];")
    f.line("cc = 0;")
    f.line(f"for (int i = 0; i < {n}; i++) cc = sbb(cc, e[{n} + i], P[i], &r[i]);")
    f.line("uint64_t mask = -(uint64_t)cc;")
    f.line(f"for (int i = 0; i < {n}; i++) res[i] = (e[{n} + i] & mask) | (r[i] & ~mask);")

    f.end()

def fp_inv_64(f, pp, v):
    n = pp.num_words64
    p = int(pp.p)
    R = int(pp.R)

    f.comment("Implements inversion mod p using a constant-time extended binary GCD algorithm")
    f.comment("by Thomas Pornin: https://eprint.iacr.org/2020/972.")
    f.comment("This code is based on the version implented in the SQIsign submission.")
    f.comment(f"It asssumes the input is normalized and in the Montgomery representation with {v.R}.")
    f.start("void fp_inv_64(uint64_t* res, const uint64_t* src)")
    f.line(f"uint64_t a[{n}], b[{n}], u[{n}], v[{n}];")
    f.line("uint64_t xa, xb, f0, g0, f1, g1;")
    f.line()

    f.comment("Initialize a = src, b = p, u = 1 * R, v = 0")
    # Throughout, a = u*src mod p and b = v*src mod p.
    # Each iteration decreases nbits(a) + nbits(b) with 1 bit.
    # Output: b = gcd(src, p). If b = 1, v = src^-1 mod p (in Montgomery repr)

    f.line(f"for (int i = 0; i < {n}; i++) {{")
    f.indent()
    f.line("a[i] = src[i];")
    f.line(f"b[i] = {v.p_64}[i];")
    f.line(f"u[i] = {v.R_64}[i];")
    f.line("v[i] = 0;")
    f.unindent()
    f.line("}")
    f.line()


    # Outer loop with k = 32
    # Once nbits(a) + nbits(b) <= 62, both fit in a single uint64_t, 
    # so we can switch to a faster single-scalar loop.
    # We choose <= 62 instead of <= 64 because of the bounds on theinputs f, g for lin_64.
    
    N_outer, N_scalar, _ = get_inv_params(pp)
    f.comment(f"Main loop: {N_outer} outer iterations of k-1=31 inner halvings")
    f.line(f"for (int i = 0; i < {N_outer}; i++) {{")
    f.indent()

    # Approximation step
    f.comment("Get 64-bit approximations xa, xb of a, b")
    f.line("uint64_t tnzm = 0, tnza = 0, tnzb = 0, snza = 0, snzb = 0;")
    f.line("uint64_t found = 0;")
    f.line(f"for (int k = {n - 1}; k >= 1; k--) {{")
    f.indent()
    f.line("uint64_t mk = a[k] | b[k];")
    f.line("uint64_t nz = sgnw(mk | -mk) & ~found;")
    f.line("found |= nz;")
    f.line("tnzm |= mk & nz;")
    f.line("tnza |= a[k] & nz;")
    f.line("tnzb |= b[k] & nz;")
    f.line("snza |= a[k-1] & nz;")
    f.line("snzb |= b[k-1] & nz;")
    f.unindent()
    f.line("}")
    f.line()

    f.line("int64_t s = lzcnt(tnzm);")
    f.line("uint64_t sm = (uint64_t)((31 - s) >> 63);")
    f.line("tnza ^= sm & (tnza ^ ((tnza << 32) | (snza >> 32)));")
    f.line("tnzb ^= sm & (tnzb ^ ((tnzb << 32) | (snzb >> 32)));")
    f.line("s -= 32 & sm;")
    f.line("tnza <<= s;")
    f.line("tnzb <<= s;")
    f.line()

    f.line("uint64_t tzx = ~found;")
    f.line("tnza |= a[0] & tzx;")
    f.line("tnzb |= b[0] & tzx;")
    f.line("xa = (a[0] & 0x7FFFFFFF) | (tnza & 0xFFFFFFFF80000000);")
    f.line("xb = (b[0] & 0x7FFFFFFF) | (tnzb & 0xFFFFFFFF80000000);")
    f.line()

    # 31 inner iterations (packed fg)
    f.comment("31 inner iterations")
    f.line("uint64_t fg0 = (uint64_t)1;")
    f.line("uint64_t fg1 = (uint64_t)1 << 32;")
    f.line("for (int j = 0; j < 31; j++) {")
    f.indent()
    f.line("uint64_t a_odd, swap, t0, t1, t2;")
    f.line("unsigned char cc;")
    f.line("a_odd = -(xa & 1);")
    f.line("cc = sbb(0, xa, xb, &t0);")
    f.line("(void)sbb(cc, 0, 0, &swap);")
    f.line("swap &= a_odd;")
    f.line("t1 = swap & (xa ^ xb);")
    f.line("xa ^= t1;")
    f.line("xb ^= t1;")
    f.line("t2 = swap & (fg0 ^ fg1);")
    f.line("fg0 ^= t2;")
    f.line("fg1 ^= t2;")
    f.line("xa -= a_odd & xb;")
    f.line("fg0 -= a_odd & fg1;")
    f.line("xa >>= 1;")
    f.line("fg1 <<= 1;")
    f.unindent()
    f.line("}")
    f.line("fg0 += 0x7FFFFFFF7FFFFFFF;")
    f.line("fg1 += 0x7FFFFFFF7FFFFFFF;")
    f.line("f0 = (fg0 & 0xFFFFFFFF) - (uint64_t)0x7FFFFFFF;")
    f.line("g0 = (fg0 >> 32) - (uint64_t)0x7FFFFFFF;")
    f.line("f1 = (fg1 & 0xFFFFFFFF) - (uint64_t)0x7FFFFFFF;")
    f.line("g1 = (fg1 >> 32) - (uint64_t)0x7FFFFFFF;")
    f.line()

    # Update a, b, u, v
    f.comment("Update a, b, u, v")
    f.line(f"uint64_t na[{n}], nb[{n}], nu[{n}], nv[{n}];")
    f.line("uint64_t nega = lindiv31abs(na, a, b, f0, g0);")
    f.line("uint64_t negb = lindiv31abs(nb, a, b, f1, g1);")
    f.line("f0 = (f0 ^ nega) - nega;")
    f.line("g0 = (g0 ^ nega) - nega;")
    f.line("f1 = (f1 ^ negb) - negb;")
    f.line("g1 = (g1 ^ negb) - negb;")
    f.line("lin64(nu, u, v, f0, g0);")
    f.line("lin64(nv, u, v, f1, g1);")
    f.line(f"for (int k = 0; k < {n}; k++) {{")
    f.indent()
    f.line("a[k] = na[k]; b[k] = nb[k]; u[k] = nu[k]; v[k] = nv[k];")
    f.unindent()
    f.line("}")

    f.unindent()
    f.line("}")
    f.line()

    # Final scalar loop
    f.comment(f"Final scalar loop: {N_scalar} iterations")
    f.line("xa = a[0]; xb = b[0];")
    f.line("f0 = 1; g0 = 0; f1 = 0; g1 = 1;")
    f.line(f"for (int j = 0; j < {N_scalar}; j++) {{")
    f.indent()
    f.line("uint64_t a_odd, swap, t0, t1, t2, t3;")
    f.line("unsigned char cc;")
    f.line("a_odd = -(xa & 1);")
    f.line("cc = sbb(0, xa, xb, &t0);")
    f.line("(void)sbb(cc, 0, 0, &swap);")
    f.line("swap &= a_odd;")
    f.line("t1 = swap & (xa ^ xb);")
    f.line("xa ^= t1;")
    f.line("xb ^= t1;")
    f.line("t2 = swap & (f0 ^ f1);")
    f.line("f0 ^= t2;")
    f.line("f1 ^= t2;")
    f.line("t3 = swap & (g0 ^ g1);")
    f.line("g0 ^= t3;")
    f.line("g1 ^= t3;")
    f.line("xa -= a_odd & xb;")
    f.line("f0 -= a_odd & f1;")
    f.line("g0 -= a_odd & g1;")
    f.line("xa >>= 1;")
    f.line("f1 <<= 1;")
    f.line("g1 <<= 1;")
    f.unindent()
    f.line("}")
    f.line("lin64(v, u, v, f1, g1);")
    f.line()

    f.comment(f"Correction: Given the input src = x * R mod p,")
    f.comment(f" v now holds 2^num_iterations * x^-1 mod p")
    f.comment(f"We multiply by a precomputed constant to obtain x^-1 * R mod p")
    f.line(f"fp_mul_64(res, v, {v.inv_const_64});")

    f.end()


def write_fp_inv_functions(f, pp, v):
    f.banner("Unvectorized Fp inversion")

    inv_macros(f, pp, v)

    sgnw(f, pp, v)
    is_zero64(f, pp, v)
    select64(f, pp, v)
    umul_64(f, pp, v)
    umul_add_64(f, pp, v)
    umulx2_64(f, pp, v)
    umulx2_add_64(f, pp, v)
    neg64(f, pp, v)
    lin64(f, pp, v)
    lindiv31abs_64(f, pp, v)
    fp_mul_64(f, pp, v)
    fp_inv_64(f, pp, v)
