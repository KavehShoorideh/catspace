"""GEOMETRIC LEMMAS, DISCOVERED WITHOUT AN LLM (Kaveh 2026-10-09: "how can we come up with these lemmas in a
mathematically sound way ... an algorithm where these lemmas can be written without the involvement of an LLM?";
the seed examples: a knight that hops from d7 can never guard d8; two pieces on opposite colours cannot be forked by a
knight).

STAGE 1 -- INVARIANTS OF A PIECE'S MOVES, by exact algebra over its move vectors (a move is a vector (df, dr)):
  modular   every linear form a*file + b*rank (mod m) that all of the piece's move vectors shift by the SAME constant c
            -- c = 0: preserved by every move; c != 0: every move shifts it by c (the knight's colour flip). Exhaustive
            over m <= 8: complete for linear forms with those moduli.
  monotone  every form a*file + b*rank (a, b in {-1, 0, 1}) that no move vector decreases (a pawn's rank)
  step      the most one move changes a distance (Chebyshev, Manhattan, knight distance): a lower bound on the moves
            needed to cover a distance
STAGE 2 -- EVERY IMPLICATION THAT HOLDS, over finite tables of configurations (Formal Concept Analysis, Ganter & Wille):
  rows are configurations (a piece moving s -> t with a square x; a piece on f with two targets x, y), columns are
  predicates (relations of squares, the Stage 1 invariants, attacks on an empty board). The canonical (Duquenne-Guigues)
  basis is computed with Ganter's NextClosure: SOUND (each implication holds on every row) and COMPLETE (every
  implication holding on the table follows from the basis) -- for the given predicates; the vocabulary is the ceiling.
  The rows are deduplicated first (implications depend only on the distinct rows).
"""
from __future__ import annotations

import itertools
import json
import time
from collections import deque
from math import gcd

from catspace.io import paths

SQ = [(f, r) for f in range(8) for r in range(8)]
KNIGHT = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
KING = [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1) if (a, b) != (0, 0)]
ROOK = [(d * k, 0) for d in (-1, 1) for k in range(1, 8)] + [(0, d * k) for d in (-1, 1) for k in range(1, 8)]
BISHOP = [(a * k, b * k) for a in (-1, 1) for b in (-1, 1) for k in range(1, 8)]
MOVES = {"knight": KNIGHT, "king": KING, "rook": ROOK, "bishop": BISHOP, "queen": ROOK + BISHOP,
         "pawn push (white)": [(0, 1), (0, 2)], "pawn capture (white)": [(1, 1), (-1, 1)]}


# ---------------------------------------------------------------- stage 1
def modular_invariants(vecs):
    out = []
    for m in range(2, 9):
        for a in range(m):
            for b in range(m):
                if (a, b) == (0, 0) or gcd(gcd(a, b), m) != 1:
                    continue                                        # primitive forms only
                cs = {(a * dx + b * dy) % m for dx, dy in vecs}
                if len(cs) == 1:
                    out.append((m, a, b, cs.pop()))
    # keep the ones not implied by a smaller modulus (m' | m with the same form reduced)
    keep = []
    for m, a, b, c in out:
        if any(m2 < m and m % m2 == 0 and (a % m2, b % m2) == (a2, b2) and c % m2 == c2 for m2, a2, b2, c2 in keep):
            continue
        keep.append((m, a, b, c))
    return keep


def monotone_forms(vecs):
    out = []
    for a, b in itertools.product((-1, 0, 1), repeat=2):
        if (a, b) == (0, 0):
            continue
        vals = [a * dx + b * dy for dx, dy in vecs]
        if min(vals) >= 0:
            out.append((a, b, "strictly increases" if min(vals) > 0 else "never decreases"))
    return out


def knight_distance():
    dist = {}
    for s in SQ:
        d = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for dx, dy in KNIGHT:
                v = (u[0] + dx, u[1] + dy)
                if 0 <= v[0] < 8 and 0 <= v[1] < 8 and v not in d:
                    d[v] = d[u] + 1; q.append(v)
        for t, k in d.items():
            dist[(s, t)] = k
    return dist


def exact_invariants(vecs):
    """integer forms a*file + b*rank (a, b in -2..2, primitive) that EVERY move changes by the same constant c, exactly"""
    out = []
    for a, b in itertools.product(range(-2, 3), repeat=2):
        if (a, b) == (0, 0) or gcd(abs(a), abs(b)) != 1 or (a, b) < (0, 0) and (-a, -b) in [(x, y) for x, y, _ in out]:
            continue
        cs = {a * dx + b * dy for dx, dy in vecs}
        if len(cs) == 1 and not any((-a, -b) == (x, y) for x, y, _ in out):
            c = cs.pop()
            if (a, b) < (0, 0) or (a == 0 and b < 0):                  # read it with the first coefficient positive
                a, b, c = -a, -b, -c
            if (a, b) not in [(x, y) for x, y, _ in out]:
                out.append((a, b, c))
    return out


def stage1() -> list:
    lemmas = []
    form = lambda a, b: " + ".join(x for x in ((f"{a}*file" if a != 1 else "file") if a else "", (f"{b}*rank" if b != 1 else "rank") if b else "") if x)
    for piece, vecs in MOVES.items():
        exact = exact_invariants(vecs)
        for a, b, c in exact:
            lemmas.append(dict(stage=1, piece=piece, kind="exact", a=a, b=b, shift=c,
                               text=f"{piece}: every move {'leaves' if c == 0 else f'adds {c} to'} {form(a, b)} {'unchanged' if c == 0 else ''}".rstrip()))
        for m, a, b, c in modular_invariants(vecs):
            if any((a * 1) % m == x % m and (b * 1) % m == y % m for x, y, _ in exact) or \
               any(all(((a * dx + b * dy) - (k * (x * dx + y * dy))) % m == 0 for dx, dy in vecs) for x, y, _ in exact for k in range(1, m)):
                continue                                              # implied by an exact invariant
            what = f"({form(a, b)}) mod {m}"
            lemmas.append(dict(stage=1, piece=piece, kind="modular", m=m, a=a, b=b, shift=c,
                               text=f"{piece}: every move {'preserves' if c == 0 else f'shifts by {c}'} {what}"))
        for a, b, how in monotone_forms(vecs):
            if len({a * dx + b * dy for dx, dy in vecs}) == 1:
                continue                                              # constant: an exact invariant, listed above
            lemmas.append(dict(stage=1, piece=piece, kind="monotone", a=a, b=b,
                               text=f"{piece}: every move {how} {form(a, b) if a >= 0 and b >= 0 else f'({a})*file + ({b})*rank'}"))
        cheb = max(max(abs(dx), abs(dy)) for dx, dy in vecs); man = max(abs(dx) + abs(dy) for dx, dy in vecs)
        lemmas.append(dict(stage=1, piece=piece, kind="step", text=f"{piece}: one move changes the Chebyshev distance by at most {cheb}, the Manhattan distance by at most {man}"))
    kd = knight_distance()
    par = all((kd[(s, t)] % 2) == ((s[0] + s[1] + t[0] + t[1]) % 2) for s in SQ for t in SQ)
    lemmas.append(dict(stage=1, piece="knight", kind="step",
                       text=f"knight: the number of moves between two squares has the parity of their colour difference (checked on all 4,096 pairs: {par}); the knight-distance table is exact (BFS)"))
    return lemmas


# ---------------------------------------------------------------- stage 2: Formal Concept Analysis
def colour(s):
    return (s[0] + s[1]) % 2


def on_line(a, b):
    return a != b and (a[0] == b[0] or a[1] == b[1] or abs(a[0] - b[0]) == abs(a[1] - b[1]))


def between(a, u, b):
    """u strictly between a and b on a rook or bishop line"""
    if not on_line(a, b) or u in (a, b):
        return False
    df = (b[0] > a[0]) - (b[0] < a[0]); dr = (b[1] > a[1]) - (b[1] < a[1])
    x, y = a[0] + df, a[1] + dr
    while (x, y) != b:
        if (x, y) == u:
            return True
        x += df; y += dr
    return False


def attacks(piece, s, x):
    """the piece on s attacks x on an EMPTY board"""
    d = (x[0] - s[0], x[1] - s[1])
    return d in MOVES[piece] if piece in MOVES else False


def orth_adj(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def diag_adj(a, b):
    return abs(a[0] - b[0]) == 1 and abs(a[1] - b[1]) == 1


def move_context():
    """rows: a piece moves s -> t (on an empty board), x any other square"""
    attrs = ["knight", "bishop", "rook", "queen", "king",
             "x orthogonally next to s", "x diagonally next to s", "x same colour as s", "x other colour than s",
             "x on a line through s", "t same colour as s", "t other colour than s",
             "s attacks x", "t attacks x", "t does NOT attack x",
             "t stays on the line s-x", "t LEAVES the line s-x",
             "keeps guarding x", "STOPS guarding x"]
    rows = set()
    for piece in ("knight", "bishop", "rook", "queen", "king"):
        for s in SQ:
            for d in MOVES[piece]:
                t = (s[0] + d[0], s[1] + d[1])
                if not (0 <= t[0] < 8 and 0 <= t[1] < 8):
                    continue
                for x in SQ:
                    if x in (s, t):
                        continue
                    a_s = attacks(piece, s, x); a_t = attacks(piece, t, x)
                    line = on_line(s, x)
                    row = {piece}
                    if orth_adj(x, s): row.add("x orthogonally next to s")
                    if diag_adj(x, s): row.add("x diagonally next to s")
                    row.add("x same colour as s" if colour(x) == colour(s) else "x other colour than s")
                    if line: row.add("x on a line through s")
                    row.add("t same colour as s" if colour(t) == colour(s) else "t other colour than s")
                    if a_s: row.add("s attacks x")
                    row.add("t attacks x" if a_t else "t does NOT attack x")
                    if line:
                        stays = between(s, t, x) or between(t, s, x) or between(s, x, t) or (on_line(t, x) and on_line(s, t) and
                                 ((s[0] - x[0]) * (t[1] - x[1]) == (s[1] - x[1]) * (t[0] - x[0])))
                        row.add("t stays on the line s-x" if stays else "t LEAVES the line s-x")
                    if a_s:
                        row.add("keeps guarding x" if a_t else "STOPS guarding x")
                    rows.add(frozenset(row))
    return attrs, rows


def fork_context():
    """rows: a piece on f, two other squares x, y"""
    attrs = ["knight", "bishop", "rook", "queen", "king", "x, y same colour", "x, y other colour",
             "x, y orthogonally adjacent", "x, y diagonally adjacent", "x, y on one line",
             "f attacks x", "f attacks y", "f attacks both (a fork)"]
    rows = set()
    for piece in ("knight", "bishop", "rook", "queen", "king"):
        for f in SQ:
            others = [s for s in SQ if s != f]
            for x, y in itertools.combinations(others, 2):
                ax, ay = attacks(piece, f, x), attacks(piece, f, y)
                row = {piece, "x, y same colour" if colour(x) == colour(y) else "x, y other colour"}
                if orth_adj(x, y): row.add("x, y orthogonally adjacent")
                if diag_adj(x, y): row.add("x, y diagonally adjacent")
                if on_line(x, y): row.add("x, y on one line")
                if ax: row.add("f attacks x")
                if ay: row.add("f attacks y")
                if ax and ay: row.add("f attacks both (a fork)")
                rows.add(frozenset(row))
    return attrs, rows


def canonical_basis(attrs, rows):
    """Ganter's NextClosure: the Duquenne-Guigues basis of the context (attributes as bits)."""
    n = len(attrs); idx = {a: i for i, a in enumerate(attrs)}
    objs = [sum(1 << idx[a] for a in r) for r in rows]
    full = (1 << n) - 1

    def closure(A):
        out = full
        for o in objs:
            if o & A == A:
                out &= o
        return out                                                   # (no object has A: every attribute)

    basis = []

    def lstar(X):                                                    # close X under the implications found so far
        changed = True
        while changed:
            changed = False
            for P, C in basis:
                if P & X == P and P != X and C & ~X:
                    X |= C; changed = True
        return X

    def next_set(A):
        for i in range(n - 1, -1, -1):
            bit = 1 << i
            if A & bit:
                A &= ~bit
            else:
                B = lstar(A | bit)
                if not (B & ~A) & (bit - 1):                          # nothing new below i
                    return B
        return None

    A = lstar(0)
    while A is not None:
        c = closure(A)
        if c != A:
            basis.append((A, c))
        if A == full:
            break
        A = next_set(A)
    names = lambda X: [attrs[i] for i in range(n) if X >> i & 1]
    imps = [(names(P), names(C & ~P)) for P, C in basis if C != full]
    never = [names(P) for P, C in basis if C == full]                 # premises no configuration has: exclusions
    return imps, never, (lambda prem: names(closure(sum(1 << idx[a] for a in prem))))


# the lemmas named on 2026-10-09 (Kaveh's knight on d7, the fork colours): each must FOLLOW from the computed basis
CHECKS = {
    "a piece moves s -> t, a square x": [
        (["knight", "x orthogonally next to s"], "t does NOT attack x", "a knight that hops from s never covers an orthogonal neighbour of s (d7 -> d8)"),
        (["knight", "x on a line through s"], "t LEAVES the line s-x", "a knight blocker always opens the line it stands on"),
        (["knight", "s attacks x"], "STOPS guarding x", "a knight that moves stops guarding every square it guarded"),
        (["bishop", "x other colour than s"], "t does NOT attack x", "a bishop never covers the other colour"),
    ],
    "a piece on f, two squares x and y": [
        (["knight", "f attacks both (a fork)"], "x, y same colour", "a knight forks only two pieces of the same colour"),
        (["knight", "x, y other colour", "f attacks both (a fork)"], None, "no knight forks two pieces of opposite colours (the premise never occurs)"),
    ],
}


def stage2():
    out = []
    for name, ctx in (("a piece moves s -> t, a square x", move_context), ("a piece on f, two squares x and y", fork_context)):
        t0 = time.time()
        attrs, rows = ctx()
        imps, never, close = canonical_basis(attrs, rows)
        checks = []
        for prem, concl, text in CHECKS.get(name, []):
            c = close(prem)
            if concl is None:
                holds = all(a in c for a in attrs)                     # the premise is unsatisfiable: it never occurs
            else:
                holds = concl in c and not all(a in c for a in attrs)  # follows, from a satisfiable premise
            checks.append((text, prem, concl, holds))
        out.append(dict(context=name, attributes=len(attrs), distinct_rows=len(rows), implications=imps, never=never,
                        checks=checks, secs=time.time() - t0))
    return out


def main():
    t0 = time.time()
    s1 = stage1()
    print(f"STAGE 1 -- invariants of each piece's moves ({len(s1)} statements, {time.time() - t0:.2f} s):")
    for l in s1:
        print("   " + l["text"])
    s2 = stage2()
    for c in s2:
        print(f"\nSTAGE 2 -- {c['context']}: {c['attributes']} predicates, {c['distinct_rows']} distinct rows, "
              f"canonical basis of {len(c['implications'])} implications ({c['secs']:.1f} s)")
        for text, prem, concl, holds in c["checks"]:
            print(f"   CHECK {'OK  ' if holds else 'FAIL'} {text}: {' & '.join(prem)} => {concl}")
        print(f"   never together ({len(c['never'])}): " + "; ".join(" & ".join(x) for x in c["never"][:8]) + (" ..." if len(c["never"]) > 8 else ""))
        short = sorted(c["implications"], key=lambda pc: (len(pc[0]), pc[0]))
        for P, C in short:
            if len(P) <= 3:
                print(f"   {' & '.join(P) or '(always)'}  =>  {' & '.join(C)}")
        print(f"   (+ {sum(1 for P, _ in short if len(P) > 3)} implications with longer premises in the json)")
    out = paths.derived("geo_lemmas.json")
    with open(out, "w") as fh:
        json.dump(dict(stage1=s1, stage2=s2), fh, indent=1)
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
