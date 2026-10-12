"""THE BASE VOCABULARY, the Python side (Kaveh 2026-10-12: "the base has to be deterministic and interpretable"). One table
-- each primitive with its type, its gloss, its Lean definition (Ideas.Base.Vocabulary) and its Rust evaluation
(vocab.rs) -- and what is built from it: terms, their S-expressions, their glosses, their Lean expressions, their syntax
graphs (for a graph network) and symbol sequences (for a transformer), and the enumeration smallest first.

No chess concept lives here; a term is a composition of these primitives and nothing else.
"""
from __future__ import annotations

import itertools

# symbol -> (argument types, result type, gloss template)
PRIMITIVES = {
    "W": ((), "sq", "our king"),
    "L": ((), "sq", "their king"),
    "P": ((), "sq", "our other piece"),
    "edge": ((), "set", "the edge squares"),
    "corner": ((), "set", "the corner squares"),
    "attacked": ((), "set", "the squares we attack"),
    "free": ((), "set", "the squares we do not attack"),
    "box": (("sq",), "set", "the box around {0}"),
    "rank": (("sq",), "set", "the rank of {0}"),
    "file": (("sq",), "set", "the file of {0}"),
    "line": (("sq", "sq"), "set", "the line through {0} and {1}"),
    "between": (("sq", "sq"), "set", "the squares between {0} and {1}"),
    "nearer": (("sq", "sq"), "set", "the squares nearer {0} than {1}"),
    "reach": (("sq", "set"), "set", "the squares {0} can walk to over {1}"),
    "and": (("set", "set"), "set", "{0} that are also {1}"),
    "not": (("set",), "set", "the squares outside {0}"),
    "count": (("set",), "num", "how many of {0}"),
    "dist": (("sq", "sq"), "num", "the king-step distance from {0} to {1}"),
    "sum": (("num", "num"), "num", "{0} plus {1}"),
}
SYMBOLS = list(PRIMITIVES)


def head(t):
    return t if isinstance(t, str) else t[0]


def sexpr(t) -> str:
    return t if isinstance(t, str) else "(" + " ".join(sexpr(x) for x in t) + ")"


def parse(s: str):
    toks = s.replace("(", " ( ").replace(")", " ) ").split()

    def at(i):
        if toks[i] != "(":
            return toks[i], i + 1
        h = toks[i + 1]; j = i + 2; args = []
        while toks[j] != ")":
            a, j = at(j); args.append(a)
        return (h, *args), j + 1
    t, _ = at(0)
    return t


def gloss(t, piece_name: str = "our other piece") -> str:
    """The term read out in the base's words (deterministic)."""
    if isinstance(t, str):
        return piece_name if t == "P" else PRIMITIVES[t][2]
    return PRIMITIVES[t[0]][2].format(*[gloss(a, piece_name) for a in t[1:]])


def size(t) -> int:
    return 1 if isinstance(t, str) else 1 + sum(size(a) for a in t[1:])


def lean(t) -> str:
    """The term as a Lean expression over Ideas.Vocabulary (parameters: the position `p`, our colour `c`, squares `W L P`)."""
    if isinstance(t, str):
        return {"W": "W", "L": "L", "P": "P", "edge": "edge", "corner": "corner",
                "attacked": "(attackedBy p c L)", "free": "(fun u => !(attackedBy p c L u))"}[t]
    h, *a = t
    x = [lean(y) for y in a]
    return {
        "box": lambda: f"(box {x[0]})", "rank": lambda: f"(rankOf {x[0]})", "file": lambda: f"(fileOf {x[0]})",
        "line": lambda: f"(lineThrough {x[0]} {x[1]})", "between": lambda: f"(between {x[0]} {x[1]})",
        "nearer": lambda: f"(nearer {x[0]} {x[1]})", "reach": lambda: f"(reach {x[0]} {x[1]})",
        "and": lambda: f"(fun u => {x[0]} u && {x[1]} u)", "not": lambda: f"(fun u => !({x[0]} u))",
        "count": lambda: f"(count {x[0]})", "dist": lambda: f"(dist {x[0]} {x[1]})", "sum": lambda: f"({x[0]} + {x[1]})",
    }[h]()


def graph(t):
    """The syntax tree as a graph: node symbols (indices into SYMBOLS), edges parent -> child with the argument slot."""
    nodes, edges = [], []

    def walk(u):
        i = len(nodes)
        nodes.append(SYMBOLS.index(head(u)))
        if not isinstance(u, str):
            for slot, a in enumerate(u[1:]):
                j = walk(a)
                edges.append((i, j, slot))
        return i
    walk(t)
    return nodes, edges


def tokens(t):
    """The term in prefix order, as symbol indices (a formal symbol sequence, not language)."""
    if isinstance(t, str):
        return [SYMBOLS.index(t)]
    return [SYMBOLS.index(t[0])] + [x for a in t[1:] for x in tokens(a)]


def enumerate_numbers(sqs=("W", "L", "P"), max_and: int = 2):
    """Number terms smallest first: the count of a set built from at most two base sets (and, not), or a distance."""
    base = ["edge", "corner", "attacked", "free"]
    base += [(h, s) for h in ("box", "rank", "file") for s in sqs]
    base += [(h, a, b) for h in ("line", "between", "nearer") for a in sqs for b in sqs if a != b]
    base += [("reach", s, "free") for s in sqs]
    sets = list(base) + [("not", a) for a in base]
    if max_and >= 2:
        sets += [("and", a, b) for a, b in itertools.combinations(base, 2)]
        sets += [("and", a, ("not", b)) for a in base for b in base if a != b]
    nums = [("count", s) for s in sets] + [("dist", a, b) for a, b in itertools.combinations(sqs, 2)]
    nums.sort(key=size)
    return nums
