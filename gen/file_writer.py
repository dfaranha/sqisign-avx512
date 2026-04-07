from params import (get_fp_vec_params, get_fp_num_vec_name, get_fp2_num_vec_name)

def int_to_limbs(n, limb_len, num_limbs=None):
    if n == 0:
        limbs = [0]
    else:
        limbs = []
        val = n
        i = 0
        while val > 0:
            # Mask out the bottom bits and append
            limbs.append((val & ((1 << limb_len) - 1)))
            val >>= limb_len
            i += 1
    if num_limbs is not None:
        assert len(limbs) <= num_limbs
        limbs += (num_limbs-len(limbs)) * [0]
    return limbs

def limbs_to_hex(limbs):
    return [f"0x{l:x}" for l in limbs]

def fp_array_1d_inner(f, a_list, num_limbs, limb_len, last_comma = False):
    n, limbs_per_vec = get_fp_vec_params(len(a_list), num_limbs)

    # pad with 0s using n*k
    limbs = [limbs_to_hex(int_to_limbs(a, limb_len, n*limbs_per_vec)) for a in a_list]

    if limbs_per_vec == 1: 
        for i in range(n):
            suffix = "," if i < n-1 or last_comma else ""
            f.line(f"{{{limbs[0][i]}, {limbs[1][i]}, {limbs[2][i]}, {limbs[3][i]}, {limbs[4][i]}, {limbs[5][i]}, {limbs[6][i]}, {limbs[7][i]}}}{suffix}")
    elif limbs_per_vec == 2:
        for i in range(n):
            suffix = "," if i < n-1 or last_comma else ""
            f.line(f"{{{limbs[0][i]}, {limbs[0][i+n]}, {limbs[1][i]}, {limbs[1][i+n]}, {limbs[2][i]}, {limbs[2][i+n]}, {limbs[3][i]}, {limbs[3][i+n]}}}{suffix}")
    else:
        for i in range(n):
            suffix = "," if i < n-1 or last_comma else ""
            f.line(f"{{{limbs[0][i]}, {limbs[0][i+n]}, {limbs[0][i+2*n]}, {limbs[0][i+3*n]}, {limbs[1][i]}, {limbs[1][i+n]}, {limbs[1][i+2*n]}, {limbs[1][i+3*n]}}}{suffix}")

def interleave_lists(list_a, list_b):
    return [item for pair in zip(list_a, list_b) for item in pair]

def convert_fp2_list_to_fp1_lists(a_list, format):
    re = [a[0] for a in a_list]
    im = [a[1] for a in a_list]

    if format in ["8x1x1", "4x1x2"]:
        return [re, im]
    elif format in ["4x2x1", "2x2x2"]:
        return [interleave_lists(re, im)]
    else: # 2x4x1
        inter = interleave_lists(re, im)
        zero = [0] * 4
        return [interleave_lists(inter, zero)]


class FileWriter:
    def __init__(self, file_name):
        self.file_name = file_name
        self.file = open(file_name, "w")
        self.indent_lvl = 0

    def indent(self):
        self.indent_lvl += 1

    def unindent(self):
        if self.indent_lvl > 0:
            self.indent_lvl -= 1

    def line(self, str=""):
        tab = "    "
        self.file.write(f"{tab*self.indent_lvl}{str}\n")

    def comment(self, str):
        self.line("// " + str)

    def banner(self, str):
        banner_len = 100
        padding_len = (banner_len - len(str))//2
        self.comment("=" * banner_len)
        self.comment(" " * padding_len + str + " " * padding_len)
        self.comment("=" * banner_len)
        self.line()

    # start a new block
    def start(self, str=None):
        if str is not None:
            self.line(str)
        self.line("{")
        self.indent()

    # end the current block
    def end(self, suffix=None, newline=True):
        self.unindent()
        if suffix is None:
            self.line("}")
        else:
            self.line(f"}}{suffix}")
        if newline:
            self.line()

    # Converts the integer "a" to a limb array, then writes it as a uint64_t array
    def limb_array_1d(self, a, num_limbs, limb_len, arr_name=None, size_name=None, limbs_per_line=4):
        limbs = int_to_limbs(a, limb_len, num_limbs)
        hex_limbs = limbs_to_hex(limbs)

        if arr_name is None:
            self.start()
        elif size_name is None:
            self.start(f"static const uint64_t {arr_name}[{num_limbs}] =")
        else:
            self.start(f"static const uint64_t {arr_name}[{size_name}] =")
        for i in range(0, num_limbs, limbs_per_line):
            chunk = hex_limbs[i : i + limbs_per_line]
            line = ", ".join(chunk)
            
            suffix = "," if i + limbs_per_line < num_limbs else ""
            self.line(f"{line}{suffix}")
        self.end(";")

    def fp_array_1d(self, a_list, pp, v, arr_name=None, end_suffix=";", end_newline=True):
        size_name = get_fp_num_vec_name(v, len(a_list))
        
        if arr_name is None:
            self.start()
        else:
            self.start(f"static __m512i {arr_name}[{size_name}] =")

        fp_array_1d_inner(self, a_list, pp.num_limbs, pp.limb_len)
            
        self.end(suffix=end_suffix, newline=end_newline)


    def fp_array_2d(self, a_2d_list, pp, v, arr_name, size1_name):
        size1 = len(a_2d_list)
        size2_name = get_fp_num_vec_name(v, len(a_2d_list[0]))

        self.start(f"static __m512i {arr_name}[{size1_name}][{size2_name}] = ")
        for i in range(size1):
            end_suffix = "," if i < size1-1 else ""
            self.fp_array_1d(a_2d_list[i], pp, v, arr_name=None, end_suffix=end_suffix, end_newline=False)
        self.end(";")
        
    def fp2_array_1d(self, a_list, pp, v, format, arr_name=None, end_suffix=";", end_newline=True):
        fp_lists = convert_fp2_list_to_fp1_lists(a_list, format)
        size_name = get_fp2_num_vec_name(v, format)

        if arr_name is None:
            self.start()
        else:
            self.start(f"static __m512i {arr_name}[{size_name}] =")

        if len(fp_lists) == 1:
            fp_array_1d_inner(self, fp_lists[0], pp.num_limbs, pp.limb_len)
        else:
            fp_array_1d_inner(self, fp_lists[0], pp.num_limbs, pp.limb_len, last_comma=True)
            fp_array_1d_inner(self, fp_lists[1], pp.num_limbs, pp.limb_len, last_comma=False)

        self.end(suffix=end_suffix, newline=end_newline)


    def fp2_array_2d(self, a_2d_list, pp, v, format, arr_name, size1_name):
        size1 = len(a_2d_list)
        size2_name = get_fp2_num_vec_name(v, format)

        self.start(f"static __m512i {arr_name}[{size1_name}][{size2_name}] = ")
        for i in range(size1):
            end_suffix = "," if i < size1-1 else ""
            self.fp2_array_1d(a_2d_list[i], pp, v, format, arr_name=None, end_suffix=end_suffix, end_newline=False)
        self.end(";")