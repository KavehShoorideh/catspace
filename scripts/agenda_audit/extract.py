"""Plan 75.1 audit, step 1: extract every branch, constant and comparison of a module as ATOMS (read-only; prints JSON lines).

An ITEM is one of:
  CONST    a module- or class-level assignment (each target), and each non-trivial default of a function parameter;
  BRANCH   the test of an if / elif / while / assert / conditional expression / comprehension filter, split at and / or / not into its ATOMS;
  COMPARE  a comparison outside any branch test (an assignment, a return, an argument, a lambda body);
  ORDER    a call to sorted / min / max / list.sort / heapq.heappush that orders by a key (a tie-break);
  LITERAL  a number other than 0 and 1 (or a chess.* / other-module UPPERCASE constant) outside the items above, docstrings and f-string text;
  KIND     a string literal naming a goal / claim / defect kind where a question, a claim or a defect is built (new_question, Claim, Defect, goal=, kind=).
"""
import ast
import json
import sys

PATH = sys.argv[1]
SRC = open(PATH).read()
LINES = SRC.splitlines()
TREE = ast.parse(SRC)

for node in ast.walk(TREE):
    for ch in ast.iter_child_nodes(node):
        ch._parent = node


def seg(n):
    s = ast.get_source_segment(SRC, n)
    if s is None:
        return ""
    return " ".join(s.split())


def func_of(n):
    p = getattr(n, "_parent", None)
    while p is not None:
        if isinstance(p, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return p.name
        if isinstance(p, ast.ClassDef):
            return "class " + p.name
        p = getattr(p, "_parent", None)
    return "<module>"


def in_text(n):
    """True when the node only shapes trace text: inside an f-string, or an argument of a .append/.extend/.insert on a list of lines / notes / gaps."""
    p = getattr(n, "_parent", None)
    cur = n
    while p is not None:
        if isinstance(p, ast.JoinedStr):
            return True
        if isinstance(p, ast.Call) and isinstance(p.func, ast.Attribute) and p.func.attr in ("append", "extend", "insert") and cur in p.args:
            tgt = seg(p.func.value)
            if any(w in tgt for w in ("lines", "notes", "gaps", "leaf_obs", "why", "out")):
                return True
        if isinstance(p, ast.BinOp) and isinstance(p.op, ast.Add) and (isinstance(p.left, (ast.JoinedStr, ast.Constant)) and isinstance(getattr(p.left, "value", None), str) or isinstance(p.left, ast.JoinedStr) or isinstance(p.right, ast.JoinedStr)):
            return True
        if isinstance(p, (ast.stmt,)):
            return False
        cur = p
        p = getattr(p, "_parent", None)
    return False


items = []
claimed = set()      # ids of nodes already inside an item


def mark(n):
    for x in ast.walk(n):
        claimed.add(id(x))


def atoms(test):
    """Split a test at and / or / not."""
    if isinstance(test, ast.BoolOp):
        out = []
        for v in test.values:
            out += atoms(v)
        return out
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        return atoms(test.operand)
    return [test]


def add(kind, n, text=None, extra=None):
    d = dict(file=PATH.split("/")[-1], line=n.lineno, end=getattr(n, "end_lineno", n.lineno), kind=kind, func=func_of(n), text=text if text is not None else seg(n), text_only=in_text(n))
    if extra:
        d.update(extra)
    items.append(d)


# pass 1: constants (module level and class level), function defaults
for st in TREE.body:
    if isinstance(st, (ast.Assign, ast.AnnAssign)):
        tg = st.targets if isinstance(st, ast.Assign) else [st.target]
        for t in tg:
            add("CONST", st, text=f"{seg(t)} = {seg(st.value)}")
        mark(st)
    if isinstance(st, ast.ClassDef):
        for s2 in st.body:
            if isinstance(s2, (ast.Assign, ast.AnnAssign)):
                tg = s2.targets if isinstance(s2, ast.Assign) else [s2.target]
                for t in tg:
                    add("CONST", s2, text=f"{st.name}.{seg(t)} = {seg(s2.value)}")
                mark(s2)
for n in ast.walk(TREE):
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = n.args.args + n.args.kwonlyargs
        defs = [None] * (len(n.args.args) - len(n.args.defaults)) + list(n.args.defaults) + list(n.args.kw_defaults)
        for a, d in zip(args, defs):
            if d is None:
                continue
            if isinstance(d, ast.Constant) and (d.value in (None, False, True, 0, "") or d.value == 1 and not isinstance(d.value, bool)):
                continue
            add("CONST", d, text=f"def {n.name}(... {a.arg}={seg(d)})")
        for d in n.args.defaults + [x for x in n.args.kw_defaults if x is not None]:
            mark(d)

# pass 2: branches
for n in ast.walk(TREE):
    tests = []
    if isinstance(n, (ast.If, ast.While)):
        tests = [n.test]
    elif isinstance(n, ast.Assert):
        tests = [n.test]
    elif isinstance(n, ast.IfExp):
        tests = [n.test]
    elif isinstance(n, ast.comprehension):
        tests = list(n.ifs)
    for t in tests:
        for a in atoms(t):
            if id(a) in claimed:
                continue
            add("BRANCH", a, extra=dict(stmt=type(n).__name__))
            mark(a)

# pass 2b: a short-circuit and / or outside a test is a branch too (its operands are atoms)
for n in ast.walk(TREE):
    if isinstance(n, ast.BoolOp) and id(n) not in claimed:
        for a in atoms(n):
            if id(a) in claimed:
                continue
            add("BRANCH", a, extra=dict(stmt="BoolOp"))
            mark(a)

# pass 3: comparisons outside branches
for n in ast.walk(TREE):
    if isinstance(n, ast.Compare) and id(n) not in claimed:
        add("COMPARE", n)
        mark(n)

# pass 4: orders (sorted / min / max / .sort / heappush with a key or a tuple)
for n in ast.walk(TREE):
    if not isinstance(n, ast.Call) or id(n) in claimed:
        continue
    f = n.func
    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
    key = next((k for k in n.keywords if k.arg == "key"), None)
    if name in ("sorted", "min", "max", "sort") and key is not None:
        add("ORDER", n, text=f"{name}(... key={seg(key.value)})")
        mark(key.value)
    elif name == "heappush":
        add("ORDER", n, text=seg(n))
        for a in n.args[1:]:
            mark(a)

# pass 5: kind literals
for n in ast.walk(TREE):
    if not isinstance(n, ast.Call):
        continue
    f = n.func
    name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""
    if name in ("new_question", "Claim", "Defect"):
        if n.args and isinstance(n.args[0], ast.Constant) and isinstance(n.args[0].value, str) and id(n.args[0]) not in claimed:
            add("KIND", n.args[0], text=f"{name}({n.args[0].value!r}, ...)")
            mark(n.args[0])
    for k in n.keywords:
        if k.arg in ("goal", "kind") and isinstance(k.value, ast.Constant) and isinstance(k.value.value, str) and id(k.value) not in claimed:
            add("KIND", k.value, text=f"{name}(... {k.arg}={k.value.value!r})")
            mark(k.value)

# pass 6: literals
docs = set()
for n in ast.walk(TREE):
    if isinstance(n, (ast.FunctionDef, ast.ClassDef, ast.Module, ast.AsyncFunctionDef)) and n.body and isinstance(n.body[0], ast.Expr) and isinstance(getattr(n.body[0], "value", None), ast.Constant):
        docs.add(id(n.body[0].value))
for n in ast.walk(TREE):
    if id(n) in claimed or id(n) in docs:
        continue
    if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool) and n.value not in (0, 1):
        p = getattr(n, "_parent", None)
        if isinstance(p, ast.UnaryOp) and isinstance(p.op, ast.USub):
            n2 = p
        else:
            n2 = n
        if in_text(n2) and not isinstance(getattr(n2, "_parent", None), ast.Subscript):
            pass
        # the statement it sits in, trimmed, gives the context
        st = n2
        while not isinstance(st, ast.stmt):
            st = st._parent
        add("LITERAL", n2, text=f"{seg(n2)}  in: {seg(st)[:160]}")
        mark(n2)
    elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.attr.isupper() and len(n.attr) > 1 and n.value.id not in ("self",):
        st = n
        while not isinstance(st, ast.stmt):
            st = st._parent
        add("LITERAL", n, text=f"{seg(n)}  in: {seg(st)[:160]}")
        mark(n)

items.sort(key=lambda d: (d["line"], d["kind"] != "CONST"))
for d in items:
    print(json.dumps(d))
