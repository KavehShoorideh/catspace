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
    # RELATIVE GEOMETRY (Kaveh 2026-10-13: "the vocabulary needs to be complemented" -- whatever it took to see the cycle
    # lemmas, as words): a piece's file and rank from our side, and how far one piece stands from another along each
    # (plus 7: 7 is level) -- arrangements, their shifts, "ahead of", "the opposition" as compositions
    "fileof": (("sq",), "num", "the file of {0} (a = 0 ... h = 7)"),
    "rankof": (("sq",), "num", "the rank of {0} counted from our first rank (0 ... 7)"),
    "dfile": (("sq", "sq"), "num", "how many files {0} stands toward the h-file from {1}, plus 7 (7: the same file)"),
    "drank": (("sq", "sq"), "num", "how many ranks {0} stands ahead of {1} toward their side, plus 7 (7: the same rank)"),
}
for _f in "abcdefgh":
    for _r in "12345678":
        PRIMITIVES.setdefault(_f + _r, ((), "sq", _f + _r))
# THE VOCABULARY COMPLETED (Kaveh 2026-10-13: "Yes, please add all of these ... and also ... regular chess terms like pin ...
# in terms of simpler vocabulary so that we can kind of have compositional vocab") -- each the mirror of a Lean definition
# (Ideas.Base.Vocabulary) and of the Rust evaluator (vocab.rs); at the end, the earlier symbols keep their indices
_ORD = {2: "second", 3: "third", 4: "fourth", 5: "fifth"}
PRIMITIVES.update({
    # every piece named: strongest first (queen, rook, bishop, knight, pawn), pawns the most advanced first, then the lower square
    **{f"P{i}": ((), "sq", f"our {_ORD[i]} piece") for i in range(2, 6)},
    **{f"O{i}": ((), "sq", f"their {_ORD[i]} piece") for i in range(2, 6)},
    **{f"{side}{kind}": ((), "set", f"the squares of {'our' if side == 'our' else 'their'} {kind}")
       for side in ("our", "their") for kind in ("queens", "rooks", "bishops", "knights", "pawns")},
    # a pawn's direction (its owner's)
    "ahead": (("sq",), "set", "the squares in front of {0} on its file, toward the side it promotes on"),
    "front": (("sq",), "sq", "the square in front of {0}"),
    "promo": (("sq",), "sq", "the promotion square of {0}"),
    "pushes": (("sq",), "num", "how many ranks {0} has still to go"),
    "passer": (("sq",), "set", "the squares in front of {0} on its file and the two beside it"),
    # the square at an offset from a piece (from our side: files toward the h-file, ranks toward their side)
    "offset": (("sq", "int", "int"), "sq", "the square {1} files and {2} ranks from {0}"),
    # lines through pieces
    "beyond": (("sq", "sq"), "set", "the squares past {1} on the line from {0} through it"),
    "next": (("sq", "sq"), "sq", "the first piece past {1} on the line from {0} through it"),
    "xray": (("sq",), "set", "the squares {0} attacks through one piece in the way"),
    # control and exchanges
    "attackers": (("sq",), "num", "how many of our pieces attack {0}"),
    "defenders": (("sq",), "num", "how many of their pieces attack {0}"),
    "see": (("sq",), "num", "what the other side wins by the exchange on {0}"),
    "moves": (("sq",), "set", "the squares {0} can legally move to"),
    # the short window's tactics (exact; expensive -- layer 2)
    **{f"forcing_{side}_{v}": ((), "num", f"how few moves {'we need' if side == 'us' else 'they need, as if we passed,'} to force "
       + ("a mate" if v == "mate" else f"a gain of {v} or more") + " (0: not within 3)")
       for side in ("us", "them") for v in ("1", "3", "5", "9", "mate")},
    # material and the game's context; time
    "material_us": ((), "num", "our material"), "material_them": ((), "num", "their material"),
    "halfmoves": ((), "num", "the moves since the last capture or pawn move (the fifty-move clock, in plies)"),
    "seen": ((), "num", "how many times this game has seen the position"),
    "ago": (("int", "num"), "num", "{1}, {0} moves of ours ago"),
    # combinators
    "or": (("set", "set"), "set", "{0} or {1}"),
    "min": (("num", "num"), "num", "the smaller of {0} and {1}"),
    "max": (("num", "num"), "num", "the larger of {0} and {1}"),
    "diff": (("num", "num"), "num", "how far apart {0} and {1} are"),
    "eq": (("num", "num"), "num", "whether {0} equals {1} (1 or 0)"),
    # small whole numbers
    **{str(n): ((), "int", str(n)) for n in range(-2, 9)},
})

# THE CHESS TERMS -- each a composition of the words above (name -> (parameters, body, gloss)); never a new primitive: the
# evaluator (vocab.rs set_macros) and the Lean rendering unfold them into the base, the glosses keep their names
CHESS_TERMS = {
    "en_prise": (("X",), "(min (see X) 1)", "whether {0} can be won by the exchange on it (1 or 0)"),
    "pinned_ours": (("X",), "(min (min (count (and (at W) (xray (next W X)))) (count (and theirs (at (next W X))))) "
                            "(eq (count (and (between W X) (or ours theirs))) 0))",
                    "whether {0} is pinned to our king (1 or 0)"),
    "pinned_theirs": (("X",), "(min (min (count (and (at L) (xray (next L X)))) (count (and ours (at (next L X))))) "
                              "(eq (count (and (between L X) (or ours theirs))) 0))",
                      "whether {0} is pinned to their king (1 or 0)"),
    "targets_ours": (("X",), "(count (and (attacks X) (or theirs (at L))))", "how many of their pieces (their king too) {0} attacks"),
    "targets_theirs": (("X",), "(count (and (attacks X) (or ours (at W))))", "how many of our pieces (our king too) {0} attacks"),
    "skewer": (("X", "Y"), "(min (count (and (attacks X) (at Y))) (count (and (or theirs (at L)) (at (next X Y)))))",
               "whether {0} attacks {1} with a piece of theirs behind it (1 or 0)"),
    "discovery": (("X", "Y"), "(min (count (and (between Y L) (at X))) (count (and (xray Y) (at L))))",
                  "whether {0} stands between {1} and their king, {1} aiming through it (1 or 0)"),
    "passed_ours": (("X",), "(eq (count (and (passer X) theirpawns)) 0)", "whether {0} is a passed pawn, no pawn of theirs in front of it or beside (1 or 0)"),
    "passed_theirs": (("X",), "(eq (count (and (passer X) ourpawns)) 0)", "whether {0} is a passed pawn, no pawn of ours in front of it or beside (1 or 0)"),
    "outside_square": (("X",), "(eq (min (dist L (promo X)) (sum (pushes X) 1)) (sum (pushes X) 1))",
                       "whether their king is outside the square of {0} (1 or 0)"),
    "opposition": ((), "(max (min (eq (dfile W L) 7) (eq (diff (drank W L) 7) 2)) (min (eq (drank W L) 7) (eq (diff (dfile W L) 7) 2)))",
                   "whether the kings stand in opposition, one square between them on a file or rank (1 or 0)"),
    "king_ahead": (("X",), "(count (and (at W) (passer X)))", "whether our king stands in front of {0} (1 or 0)"),
    "on_key_square": (("X",), "(count (and (at W) (or (or (at (offset X -1 2)) (at (offset X 0 2))) (at (offset X 1 2)))))",
                      "whether our king stands on a key square of {0}, two ranks in front (1 or 0)"),
    "rook_behind": (("X",), "(count (and ourrooks (and (file X) (not (ahead X)))))", "how many of our rooks stand behind {0} on its file"),
    "trapped_theirs": (("X",), "(eq (count (and (moves X) free)) 0)", "whether {0} has no square to go to that we do not attack (1 or 0)"),
    "flights": ((), "(count (moves L))", "how many squares their king can move to"),
}
for _n, (_ps, _body, _g) in CHESS_TERMS.items():
    PRIMITIVES[_n] = (tuple("sq" for _ in _ps), "num", _g)
SYMBOLS = list(PRIMITIVES)
SQUARE_CLASSES = ("a1", "b1", "c1", "d1", "b2", "c2", "d2", "c3", "d3", "d4")
SQUARE_NAMES = tuple(f + r for f in "abcdefgh" for r in "12345678")


def install_terms() -> None:
    """The chess terms into the evaluator (vocab.rs set_macros): a term naming one is unfolded before it is read."""
    import catspace.incubator.analyst  # noqa: F401
    import analyst_core as A
    A.set_macros([(n, list(ps), body) for n, (ps, body, _) in CHESS_TERMS.items()])


def expand(t):
    """A term with its chess terms unfolded into the base words."""
    if isinstance(t, str):
        if t in CHESS_TERMS and not CHESS_TERMS[t][0]:
            return expand(parse(CHESS_TERMS[t][1]))
        return t
    h, *a = t
    a = [expand(x) for x in a]
    if h in CHESS_TERMS:
        ps, body, _ = CHESS_TERMS[h]
        env = dict(zip(ps, a))

        def sub(u):
            return env.get(u, u) if isinstance(u, str) else tuple(sub(v) for v in u)
        return expand(sub(parse(body)))
    return (h, *a)


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


def _sqtype(t) -> bool:
    return (PRIMITIVES.get(head(t), ((), "", ""))[1] == "sq")


def lean(t) -> str:
    """The term as a Lean expression over Ideas.Vocabulary (parameters: the position `p`, our colour `c`, squares `W L P O`,
    the game's line `hist`). A square word renders as an `OSq` (an optional square: a piece that is not there, a promotion
    square of nothing); the words over squares are lifted (`oset`, `onum`). Chess terms are unfolded into the base."""
    t = expand(t)
    if isinstance(t, str):
        if t in SQUARE_NAMES:                      # a square name -> the square so named from our side
            return f"(some (named c ((({ord(t[0]) - 97} : Fin 8)), (({int(t[1]) - 1} : Fin 8)))))"
        if t in ("W", "L", "P", "O"):
            return f"(some {t})"
        if t[0] in "PO" and t[1:].isdigit():
            return f"(nthPiece p {'c' if t[0] == 'P' else '(Color.opposite c)'} {t[1:]})"
        if PRIMITIVES.get(t, ((), "", ""))[1] == "int":
            return f"({t})"                         # (a numeral: Nat in number words; offsets cast it to Int)
        if t.startswith("forcing_"):
            _, side, v = t.split("_")
            vv = "mateGain" if v == "mate" else v
            return f"(forcingOf p {'c' if side == 'us' else '(Color.opposite c)'} {vv})"
        for side, col in (("our", "c"), ("their", "(Color.opposite c)")):
            for kind, k in (("queens", "queen"), ("rooks", "rook"), ("bishops", "bishop"), ("knights", "knight"), ("pawns", "pawn")):
                if t == f"{side}{kind}":
                    return f"(kindSet p {col} .{k})"
        return {"center": "center", "light": "light", "dark": "dark", "edge": "edge", "corner": "corner",
                "attacked": "(attackedBy p c L)", "free": "(fun u => !(attackedBy p c L u))",
                "hit": "(attackedBy p (Color.opposite c) W)", "unhit": "(fun u => !(attackedBy p (Color.opposite c) W u))",
                "ours": "(piecesOf p c)", "theirs": "(piecesOf p (Color.opposite c))",
                "material_us": "(materialOf p c)", "material_them": "(materialOf p (Color.opposite c))",
                "halfmoves": "p.halfmove", "seen": "(seenIn hist p)"}[t]
    h, *a = t
    if h == "cls":                                 # (the class word names its square itself, from side c)
        return f"(cls p c ((({ord(a[0][0]) - 97} : Fin 8)), (({int(a[0][1]) - 1} : Fin 8))))"
    if h == "ago":                                 # the term on the game's position k moves of ours back (p rebound)
        return f"(agoIn hist {a[0]} (fun p => {lean(a[1])}))"
    if h == "offset":
        return f"(offsetOf c {lean(a[0])} ({a[1]} : Int) ({a[2]} : Int))"
    x = [lean(y) for y in a]
    return {
        "box": lambda: f"(oset box {x[0]})", "rank": lambda: f"(oset rankOf {x[0]})", "file": lambda: f"(oset fileOf {x[0]})",
        "line": lambda: f"(oset2 lineThrough {x[0]} {x[1]})", "between": lambda: f"(oset2 between {x[0]} {x[1]})",
        "nearer": lambda: f"(oset2 nearer {x[0]} {x[1]})", "reach": lambda: f"(oreach {x[0]} {x[1]})",
        "and": lambda: f"(fun u => {x[0]} u && {x[1]} u)", "or": lambda: f"(fun u => {x[0]} u || {x[1]} u)",
        "not": lambda: f"(fun u => !({x[0]} u))",
        "count": lambda: f"(count {x[0]})", "dist": lambda: f"(onum2 dist {x[0]} {x[1]})", "sum": lambda: f"({x[0]} + {x[1]})",
        "lines": lambda: f"(oset lines {x[0]})", "diagonals": lambda: f"(oset diagonals {x[0]})",
        "segment": lambda: f"(oset2 segment {x[0]} {x[1]})",
        "knight1": lambda: f"(oset knight1 {x[0]})", "knight2": lambda: f"(oset knight2 {x[0]})",
        "attacks": lambda: f"(oset (attacksFrom p) {x[0]})",
        "colour": lambda: f"(oset colour {x[0]})", "at": lambda: f"(oset atSq {x[0]})",
        "fileof": lambda: f"(onum fileNum {x[0]})", "rankof": lambda: f"(onum (rankNum c) {x[0]})",
        "dfile": lambda: f"(onum2 dFile {x[0]} {x[1]})", "drank": lambda: f"(onum2 (dRank c) {x[0]} {x[1]})",
        "ahead": lambda: f"(oset (aheadOf p) {x[0]})", "passer": lambda: f"(oset (passerOf p) {x[0]})",
        "front": lambda: f"(frontOf p {x[0]})", "promo": lambda: f"(promoOf p {x[0]})",
        "pushes": lambda: f"(onum (pushesOf p) {x[0]})",
        "beyond": lambda: f"(oset2 beyondOf {x[0]} {x[1]})", "next": lambda: f"(nextOn p {x[0]} {x[1]})",
        "xray": lambda: f"(oset (xrayFrom p) {x[0]})",
        "attackers": lambda: f"(onum (attackerCount p c) {x[0]})",
        "defenders": lambda: f"(onum (attackerCount p (Color.opposite c)) {x[0]})",
        "see": lambda: f"(onum (seeOf p) {x[0]})", "moves": lambda: f"(oset (movesOf p) {x[0]})",
        "min": lambda: f"(min {x[0]} {x[1]})", "max": lambda: f"(max {x[0]} {x[1]})",
        "diff": lambda: f"(nDiff {x[0]} {x[1]})", "eq": lambda: f"(if {x[0]} = {x[1]} then 1 else 0)",
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


PIECES = ("W", "L", "P", "P2", "P3", "O", "O2", "O3")             # every piece a name (layer 1)
OTHERS = ("P", "P2", "P3", "O", "O2", "O3")


def layer1() -> list:
    """THE COMPLETED VOCABULARY's static words (Kaveh 2026-10-13), only where they say something: every piece's control,
    exchange and pawn words; distances to a piece's promotion square, front square and offsets; the new sets alone and
    against a few base sets (not every pair -- the oversearching the literature warns of); the chess terms on every piece."""
    kinds = [f"{s_}{k}" for s_ in ("our", "their") for k in ("queens", "rooks", "bishops", "knights", "pawns")]
    sets = list(kinds) + [(h, x) for h in ("ahead", "passer", "xray", "moves") for x in PIECES]
    sets += [("beyond", x, y) for x in PIECES for y in PIECES if x != y]
    few = ["attacked", "free", "hit", "unhit", "ours", "theirs", "edge", "center", ("box", "W"), ("box", "L")]
    out = [("count", x) for x in sets]
    out += [("count", ("and", x, y)) for x in sets for y in few]
    out += [("count", ("and", x, ("not", y))) for x in sets for y in few]
    out += [(h, x) for h in ("pushes", "attackers", "defenders", "see") for x in PIECES]
    out += [(h, (w, x)) for h in ("attackers", "defenders") for w in ("front", "promo") for x in OTHERS]
    out += [("dist", k, (w, x)) for k in ("W", "L") for w in ("front", "promo") for x in OTHERS]
    out += [("dist", k, ("offset", x, str(df), str(dr))) for k in ("W", "L") for x in OTHERS
            for df in range(-2, 3) for dr in range(-2, 3) if (df, dr) != (0, 0)]
    out += ["material_us", "material_them", "halfmoves", "seen"]
    for n, (ps, _, _) in CHESS_TERMS.items():
        if not ps:
            out.append(n)
        elif len(ps) == 1:
            out += [(n, x) for x in PIECES]
        else:
            out += [(n, x, y) for x in PIECES for y in PIECES if x != y]
    return out


def layer2() -> list:
    """The words that search or remember (expensive): the short window's tactics, and time."""
    out = [f"forcing_{side}_{v}" for side in ("us", "them") for v in ("1", "3", "5", "9", "mate")]
    out += [("ago", str(k), (h, "L")) for k in (1, 2, 3, 4) for h in ("fileof", "rankof")]
    return out


def enumerate_numbers(sqs=("W", "L", "P", "O"), max_and: int = 2, layers=(0, 1)):
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
    # the relative geometry: every piece's coordinates, every ordered pair's offsets
    nums += [(h, a) for h in ("fileof", "rankof") for a in sqs]
    nums += [(h, a, b) for h in ("dfile", "drank") for a in sqs for b in sqs if a != b]
    if 1 in layers:
        nums += layer1()
    if 2 in layers:
        nums += layer2()
    nums.sort(key=size)
    return nums
