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
    # their side (Kaveh 2026-10-12: "base vocab should be complete with words for their pieces"); at the end, so the
    # earlier symbols keep their indices
    "O": ((), "sq", "their other piece"),
    "hit": ((), "set", "the squares they attack"),
    "unhit": ((), "set", "the squares they do not attack"),
    "ours": ((), "set", "the squares of our pieces"),
    "theirs": ((), "set", "the squares of their pieces"),
    # the second batch (Kaveh 2026-10-12: "sets of squares ... along a diagonal or along a rank or a file from here to here
    # ... reachable by a knight, one hop or two hops ... the four squares on the center ... any particular square should
    # actually be able to be named, but ... the symmetry ... except for the last rank where there's promotion ... some per
    # piece vocabulary"; a time vocabulary later) -- each the mirror of a Lean definition in Ideas.Base.Vocabulary
    "lines": (("sq",), "set", "the rank and file through {0}"),
    "diagonals": (("sq",), "set", "the diagonals through {0}"),
    "segment": (("sq", "sq"), "set", "the squares from {0} to {1} along their line"),
    "knight1": (("sq",), "set", "the squares a knight on {0} reaches in one hop"),
    "knight2": (("sq",), "set", "the squares a knight on {0} reaches in two hops"),
    "center": ((), "set", "the four centre squares"),
    "attacks": (("sq",), "set", "the squares {0} attacks"),
    "cls": (("sq",), "set", "the squares like {0} under the board's symmetries"),
    # the square names (from our side), one per class of the board's eight symmetries -- since 2026-10-13 square words like
    # the other 54 below (cls still takes them)
    "a1": ((), "sq", "a1"), "b1": ((), "sq", "b1"), "c1": ((), "sq", "c1"), "d1": ((), "sq", "d1"),
    "b2": ((), "sq", "b2"), "c2": ((), "sq", "c2"), "d2": ((), "sq", "d2"), "c3": ((), "sq", "c3"), "d3": ((), "sq", "d3"),
    "d4": ((), "sq", "d4"),
    # a square's colour (Kaveh 2026-10-13: "go on both") -- at the end, the earlier symbols keep their indices
    "light": ((), "set", "the light squares"),
    "dark": ((), "set", "the dark squares"),
    "colour": (("sq",), "set", "the squares of {0}'s colour"),
    # EVERY SQUARE BY ITS NAME (Kaveh 2026-10-13: "you want to have a mirroring lemma separately, and then you want to have
    # the squares individually available"): the 64 squares named from our side (a8 is our promotion corner on the a-file),
    # no symmetry in the words -- the board's symmetries are a lemma of their own; and the square itself as a set
    "at": (("sq",), "set", "the square {0}"),
}
for _f in "abcdefgh":
    for _r in "12345678":
        PRIMITIVES.setdefault(_f + _r, ((), "sq", _f + _r))
SYMBOLS = list(PRIMITIVES)
SQUARE_CLASSES = ("a1", "b1", "c1", "d1", "b2", "c2", "d2", "c3", "d3", "d4")
SQUARE_NAMES = tuple(f + r for f in "abcdefgh" for r in "12345678")


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
    """The term as a Lean expression over Ideas.Vocabulary (parameters: the position `p`, our colour `c`, squares `W L P O`)."""
    if isinstance(t, str):
        if t in SQUARE_NAMES:                      # a square name -> the square so named from our side
            return f"(named c ((({ord(t[0]) - 97} : Fin 8)), (({int(t[1]) - 1} : Fin 8))))"
        if t in ("center", "light", "dark"):
            return t
        return {"W": "W", "L": "L", "P": "P", "O": "O", "edge": "edge", "corner": "corner",
                "attacked": "(attackedBy p c L)", "free": "(fun u => !(attackedBy p c L u))",
                "hit": "(attackedBy p (Color.opposite c) W)", "unhit": "(fun u => !(attackedBy p (Color.opposite c) W u))",
                "ours": "(piecesOf p c)", "theirs": "(piecesOf p (Color.opposite c))"}[t]
    h, *a = t
    if h == "cls":                                 # (the class word names its square itself, from side c)
        return f"(cls p c ((({ord(a[0][0]) - 97} : Fin 8)), (({int(a[0][1]) - 1} : Fin 8))))"
    x = [lean(y) for y in a]
    return {
        "box": lambda: f"(box {x[0]})", "rank": lambda: f"(rankOf {x[0]})", "file": lambda: f"(fileOf {x[0]})",
        "line": lambda: f"(lineThrough {x[0]} {x[1]})", "between": lambda: f"(between {x[0]} {x[1]})",
        "nearer": lambda: f"(nearer {x[0]} {x[1]})", "reach": lambda: f"(reach {x[0]} {x[1]})",
        "and": lambda: f"(fun u => {x[0]} u && {x[1]} u)", "not": lambda: f"(fun u => !({x[0]} u))",
        "count": lambda: f"(count {x[0]})", "dist": lambda: f"(dist {x[0]} {x[1]})", "sum": lambda: f"({x[0]} + {x[1]})",
        "lines": lambda: f"(lines {x[0]})", "diagonals": lambda: f"(diagonals {x[0]})", "segment": lambda: f"(segment {x[0]} {x[1]})",
        "knight1": lambda: f"(knight1 {x[0]})", "knight2": lambda: f"(knight2 {x[0]})", "attacks": lambda: f"(attacksFrom p {x[0]})",
        "colour": lambda: f"(colour {x[0]})", "at": lambda: f"(atSq {x[0]})",
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


def enumerate_numbers(sqs=("W", "L", "P", "O"), max_and: int = 2):
    """Number terms smallest first: the count of a set built from at most two base sets (and, not), or a distance."""
    base = ["edge", "corner", "attacked", "free", "hit", "unhit", "ours", "theirs", "center", "light", "dark"]
    base += [(h, s) for h in ("box", "rank", "file", "lines", "diagonals", "knight1", "knight2", "attacks", "colour") for s in sqs]
    base += [("segment", a, b) for a, b in itertools.combinations(sqs, 2)]
    base += [("cls", x) for x in SQUARE_CLASSES]
    base += [(h, a, b) for h in ("line", "between", "nearer") for a in sqs for b in sqs if a != b]
    base += [("reach", s, w) for s in sqs for w in ("free", "unhit")]
    sets = list(base) + [("not", a) for a in base]
    if max_and >= 2:
        sets += [("and", a, b) for a, b in itertools.combinations(base, 2)]
        sets += [("and", a, ("not", b)) for a in base for b in base if a != b]
    if max_and >= 2:
        # THE NAMED SQUARES (2026-10-13), only where they say something: a square in a base set (membership), a named rank
        # or file against a base set (with its complements); a piece's distance to a named square. The new sets alone are
        # constants (one square, eight squares) and two of them meet in a constant -- left out
        named = [("rank", f"a{r}") for r in "12345678"] + [("file", f"{f}1") for f in "abcdefgh"]
        sets += [("and", a, ("at", x)) for a in base for x in SQUARE_NAMES]
        sets += [y for a in base for r in named for y in (("and", a, r), ("and", a, ("not", r)), ("and", r, ("not", a)))]
    nums = [("count", s) for s in sets] + [("dist", a, b) for a, b in itertools.combinations(sqs, 2)]
    nums += [("dist", a, x) for a in sqs for x in SQUARE_NAMES]
    nums.sort(key=size)
    return nums
