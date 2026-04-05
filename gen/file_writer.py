from math import ceil
from params import get_len_42_24

def int_to_limbs(n, limb_len, num_limbs=None):
    if n == 0:
        return [0]
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
    def end(self, suffix=None):
        self.unindent()
        if suffix is None:
            self.line("}")
        else:
            self.line(f"}}{suffix}")
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

    def vec_array_1d(self, a_list, num_limbs, limb_len, arr_name=None, end_suffix=";"):
        len_a = len(a_list)
        assert len_a in [2, 4, 8]
        n42, n24, rem42, rem24 = get_len_42_24(num_limbs)
        
        # n is the number of vectors, k the number of limbs per Fp elem per vector
        if len_a == 8: #8x1
            n = num_limbs
            k = 1
        elif len_a == 4: #4x2
            n = n42
            k = 2
        else: #2x4
            n = n24
            k = 4

        # pad with 0s using n*k
        limbs = [limbs_to_hex(int_to_limbs(a, limb_len, n*k)) for a in a_list]

        if arr_name is None:
            self.start()
        else:
            self.start(f"static const __m512i {arr_name}[{n}] =")

        if len_a == 8: 
            for i in range(n):
                suffix = "," if i < n-1 else ""
                self.line(f"VSET({limbs[0][i]}, {limbs[1][i]}, {limbs[2][i]}, {limbs[3][i]}, {limbs[4][i]}, {limbs[5][i]}, {limbs[6][i]}, {limbs[7][i]}){suffix}")
        elif len_a == 4:
            for i in range(n):
                suffix = "," if i < n-1 else ""
                self.line(f"VSET({limbs[0][i]}, {limbs[0][i+n]}, {limbs[1][i]}, {limbs[1][i+n]}, {limbs[2][i]}, {limbs[2][i+n]}, {limbs[3][i]}, {limbs[3][i+n]}){suffix}")
        else:
            for i in range(n):
                suffix = "," if i < n-1 else ""
                self.line(f"VSET({limbs[0][i]}, {limbs[0][i+n]}, {limbs[0][i+2*n]}, {limbs[0][i+3*n]}, {limbs[1][i]}, {limbs[1][i+n]}, {limbs[1][i+2*n]}, {limbs[1][i+3*n]}){suffix}")
            
        self.end(end_suffix)


    def vec_array_2d(self, a_2d_list, num_limbs, limb_len, arr_name, size1_name=None, size2_name=None):
        size1 = len(a_2d_list)
        size2 = len(a_2d_list[0])

        size1_name = f"{size1}" if size1_name is None else size1_name
        size2_name = f"{size2}" if size2_name is None else size2_name

        self.start(f"static const __m512i {arr_name}[{size1_name}][{size2_name}] = ")
        for i in range(size1):
            end_suffix = "," if i < size1-1 else ""
            self.vec_array_1d(a_2d_list[i], num_limbs, limb_len, arr_name=None, end_suffix=end_suffix)
        self.end(";")
        