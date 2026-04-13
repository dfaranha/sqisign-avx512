import subprocess

# ---------- AST ----------
class Leaf:
    def __init__(self, name): self.name = name
class Dbl:
    def __init__(self, child): self.child = child
class Shift:
    def __init__(self, child, n): self.child = child; self.n = n
class Add:
    def __init__(self, left, right): self.left = left; self.right = right

# ---------- Tokenizer ----------
def _tokenize(s):
    tokens = []
    i = 0
    while i < len(s):
        if s[i].isspace():
            i += 1
        elif s[i] in '()+*':
            tokens.append(s[i])
            i += 1
        elif s[i:i+2] == '<<':
            tokens.append('<<')
            i += 2
        elif s[i].isdigit() or s[i] == '_' or s[i].isalpha():
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == '_'):
                j += 1
            tokens.append(s[i:j])
            i = j
        else:
            raise ValueError(f"Unexpected character: {s[i]}")
    return tokens

# ---------- Parser ----------
class _ExprParser:
    def __init__(self, tokens):
        self.tokens = tokens
        self.pos = 0

    def peek(self):
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected=None):
        t = self.tokens[self.pos]
        if expected and t != expected:
            raise ValueError(f"Expected {expected}, got {t}")
        self.pos += 1
        return t

    def parse_expr(self):
        left = self.parse_shift_expr()
        while self.peek() == '+':
            self.consume('+')
            right = self.parse_shift_expr()
            left = Add(left, right)
        return left

    def parse_shift_expr(self):
        result = self.parse_unary_expr()
        while self.peek() == '<<':
            self.consume('<<')
            n = int(self.consume())
            result = Shift(result, n)
        return result

    def parse_unary_expr(self):
        if (self.peek() == '2' and
            self.pos + 1 < len(self.tokens) and
            self.tokens[self.pos + 1] == '*'):
            self.consume('2')
            self.consume('*')
            inner = self.parse_unary_expr()
            return Dbl(inner)
        return self.parse_atom()

    def parse_atom(self):
        if self.peek() == '(':
            self.consume('(')
            result = self.parse_expr()
            self.consume(')')
            return result
        return Leaf(self.consume())

# ---------- Linearize AST into primitive ops ----------
def _linearize(node, dst, ops):
    """Convert AST to sequence of (op, dst, src...) using dst as accumulator."""
    if isinstance(node, Leaf):
        if node.name != dst:
            ops.append(('copy', dst, node.name))
    elif isinstance(node, Dbl):
        _linearize(node.child, dst, ops)
        ops.append(('sqr', dst, dst))
    elif isinstance(node, Shift):
        _linearize(node.child, dst, ops)
        for _ in range(node.n):
            ops.append(('sqr', dst, dst))
    elif isinstance(node, Add):
        _linearize(node.left, dst, ops)
        if isinstance(node.right, Leaf):
            rname = node.right.name
        else:
            rname = '_tmp'
            _linearize(node.right, rname, ops)
        ops.append(('mul', dst, dst, rname))

# ---------- Optimization passes ----------
def _optimize_initial_copy(ops):
    """Replace copy(d,s) + sqr(d,d) -> sqr(d,s) and similar for mul."""
    result = []
    i = 0
    while i < len(ops):
        if i + 1 < len(ops) and ops[i][0] == 'copy':
            dst, src = ops[i][1], ops[i][2]
            nxt = ops[i + 1]
            if nxt[0] == 'sqr' and nxt[1] == dst and nxt[2] == dst:
                result.append(('sqr', dst, src))
                i += 2
                continue
            elif nxt[0] == 'mul' and nxt[1] == dst and nxt[2] == dst:
                result.append(('mul', dst, src, nxt[3]))
                i += 2
                continue
        result.append(ops[i])
        i += 1
    return result

def _collapse_squarings(ops):
    """Collapse consecutive sqr(v,v) into sqr_repeat(v, count)."""
    result = []
    i = 0
    while i < len(ops):
        if ops[i][0] == 'sqr' and ops[i][1] == ops[i][2]:
            var = ops[i][1]
            count = 0
            while i < len(ops) and ops[i] == ('sqr', var, var):
                count += 1
                i += 1
            result.append(('sqr_repeat', var, count))
        else:
            result.append(ops[i])
            i += 1
    return result

# ---------- Public API ----------
def run_addchain(expr_str):
    """Run addchain search on an expression and return optimized ops list.
    
    expr_str: expression like "65*2^374 - 1"
    Returns: list of tuples:
        ('copy', dst, src)
        ('sqr', dst, src)        -- dst may differ from src
        ('sqr_repeat', var, n)   -- square var in-place n times
        ('mul', dst, src1, src2)
    Variable '1' represents the function input.
    Variable '_result' represents the function output.
    """
    result = subprocess.run(
        ["addchain", "search", expr_str],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(f"addchain failed: {result.stderr}")

    all_ops = []
    for line in result.stdout.strip().split('\n'):
        line = line.strip()
        if line.startswith('addchain:'):
            continue
        if line.startswith('return'):
            var_name = '_result'
            expr_s = line[len('return'):].strip()
        elif '=' in line:
            parts = line.split('=', 1)
            var_name = parts[0].strip()
            expr_s = parts[1].strip()
        else:
            continue

        tokens = _tokenize(expr_s)
        parser = _ExprParser(tokens)
        ast = parser.parse_expr()
        _linearize(ast, var_name, all_ops)

    all_ops = _optimize_initial_copy(all_ops)
    all_ops = _collapse_squarings(all_ops)
    return all_ops

def run_addchain_exp3div4(pp, strat):
    if not strat.gen_addchain:
        return None
    
    print("Generating addchain")
    
    expr_str = f"{pp.c}*2^{pp.e-2} - 1"
    return run_addchain(expr_str)

def count_chain_ops(ops):
    """Return (n_squarings, n_multiplications) from an ops list."""
    n_sqr = 0
    n_mul = 0
    for op in ops:
        if op[0] == 'sqr':
            n_sqr += 1
        elif op[0] == 'sqr_repeat':
            n_sqr += op[2]
        elif op[0] == 'mul':
            n_mul += 1
    return n_sqr, n_mul

def write_exp_function(f, pp, v, ops, func_name, fmt, arr_size):
    """Generate a C exponentiation function from an addchain ops list.
    
    f:         file writer
    pp, v:     prime params and variable names
    ops:       list from run_addchain
    func_name: e.g. 'fp_exp3div4_2x4'
    fmt:       e.g. '2x4' — used to form fp_sqr_2x4 etc.
    arr_size:  number of __m512i per variable for this format
    """
    sqr_fn = f'fp_sqr_{fmt}'
    mul_fn = f'fp_mul_{fmt}'
    copy_fn = f'fp_copy_{fmt}'

    def c_name(var):
        if var == '1':
            return 'a'
        if var == '_result':
            return 'res'
        return f'v_{var}'

    # Collect all intermediate variables
    all_vars = set()
    for op in ops:
        for arg in op[1:]:
            if isinstance(arg, str):
                all_vars.add(arg)
    all_vars.discard('1')
    all_vars.discard('_result')

    n_sqr, n_mul = count_chain_ops(ops)
    f.comment(f"Cost: {n_sqr} squarings, {n_mul} multiplications")
    f.start(f'void {func_name}(__m512i* res, __m512i* a)')

    for var in sorted(all_vars):
        f.line(f'__m512i {c_name(var)}[{arr_size}];')
    f.line()

    for op in ops:
        if op[0] == 'copy':
            f.line(f'{copy_fn}({c_name(op[1])}, {c_name(op[2])});')
        elif op[0] == 'sqr':
            f.line(f'{sqr_fn}({c_name(op[1])}, {c_name(op[2])});')
        elif op[0] == 'sqr_repeat':
            n = op[2]
            cn = c_name(op[1])
            if n <= 3:
                for _ in range(n):
                    f.line(f'{sqr_fn}({cn}, {cn});')
            else:
                f.line(f'for (int i = 0; i < {n}; i++) {sqr_fn}({cn}, {cn});')
        elif op[0] == 'mul':
            f.line(f'{mul_fn}({c_name(op[1])}, {c_name(op[2])}, {c_name(op[3])});')

    f.end()


# Naive exp if addchain is turned off:
# Computes t = a^k (or res = a^k if copy_to_res=True)
def write_dbl_and_add(f, pp, v, k, fmt, copy_to_res=False):
    assert k > 1
    k_bits = bin(k)[2:]
    sqr_count = 0
    first_op = True
    
    # Iterate through the bits
    # The msb is always 1, and corresponds to multiplying the accumulator 1 with a.
    # Hence, we can skip it by setting t = a.
    bits_to_process = k_bits[1:]
    for idx, bit in enumerate(bits_to_process):
        is_last_bit = (idx == len(bits_to_process) - 1)
        sqr_count += 1
        
        if bit == '1':
            # Perform the accumulated squarings. 
            if first_op:
                f.line(f"fp_sqr_{fmt}(t, a);")
                if sqr_count > 1:
                    f.line(f"for (int i = 1; i < {sqr_count}; i++) {{")
                    f.indent()
                    f.line(f"fp_sqr_{fmt}(t, t);")
                    f.unindent()
                    f.line("}")
                first_op = False
            else:
                if sqr_count == 1:
                    f.line(f"fp_sqr_{fmt}(t, t);")
                else:
                    f.line(f"for (int i = 0; i < {sqr_count}; i++) {{")
                    f.indent()
                    f.line(f"fp_sqr_{fmt}(t, t);")
                    f.unindent()
                    f.line("}")
            
            # If this is the very last bit of the entire exponent, route to res
            dst = "res" if (copy_to_res and is_last_bit) else "t"
            f.line(f"fp_mul_{fmt}({dst}, t, a);")
            sqr_count = 0

    # Handle trailing zeros in k
    if sqr_count > 0:
        if sqr_count == 1:
            src = "a" if first_op else "t"
            dst = "res" if copy_to_res else "t"
            f.line(f"fp_sqrt_{fmt}({dst}, {src});")
        elif sqr_count > 1 and not first_op and not copy_to_res:
            f.line(f"for (int i = 0; i < {sqr_count}; i++) {{")
            f.indent()
            f.line(f"fp_sqr_{fmt}(t, t);")
            f.unindent()
            f.line("}")
        elif sqr_count > 1 and first_op and not copy_to_res:
            f.line(f"fp_sqr_{fmt}(t, a);")
            f.line(f"for (int i = 1; i < {sqr_count}; i++) {{")
            f.indent()
            f.line(f"fp_sqr_{fmt}(t, t);")
            f.unindent()
            f.line("}")
        elif sqr_count == 2 and copy_to_res:
            src = "a" if first_op else "t"
            f.line(f"fp_sqr_{fmt}(t, {src});")
            f.line(f"fp_sqr_{fmt}(res, t);")
        elif sqr_count == 3 and first_op and copy_to_res:
            f.line(f"fp_sqr_{fmt}(t, a);")
            f.line(f"fp_sqr_{fmt}(t, t);")
            f.line(f"fp_sqr_{fmt}(res, t);")
        elif sqr_count > 2 and not first_op and copy_to_res:
            f.line(f"for (int i = 0; i < {sqr_count-1}; i++) {{")
            f.indent()
            f.line(f"fp_sqr_{fmt}(t, t);")
            f.unindent()
            f.line("}")
            f.line(f"fp_sqr_{fmt}(res, t);")
        else: #sqr_count > 3 and first_op and copy_to_res:
            f.line(f"fp_sqr_{fmt}(t, a);")
            f.line(f"for (int i = 1; i < {sqr_count-1}; i++) {{")
            f.indent()
            f.line(f"fp_sqr_{fmt}(t, t);")
            f.unindent()
            f.line("}")
            f.line(f"fp_sqr_{fmt}(res, t);")

def write_fp_exp3_div4_naive(f, pp, v, fmt, arr_size):
    f.start(f"void fp_exp3div4_{fmt}(__m512i* res, __m512i* a)")

    f.line(f"__m512i t[{arr_size}];")

    exponent = (pp.p - 3) // 4
    assert exponent == pp.c * 2**(pp.e-2) - 1
    write_dbl_and_add(f, pp, v, exponent, fmt, copy_to_res=True)

    f.end()