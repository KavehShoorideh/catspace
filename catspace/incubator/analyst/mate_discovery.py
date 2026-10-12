"""THE MATE DISCOVERER (Kaveh 2026-10-12: "the real question is, how can an algorithm come up with my reasoning using the
lemmas we have, defining useful lemmas along the way?").

Given only the material (our king plus pieces against a lone king), it reasons in the lone king's FRAME -- coordinates
relative to it, the 3x3 box around it -- never over whole boards:

  1 THE DEMAND of the goal (Lean Ideas.MateDemand): mate = every box square covered, the king's own square by a check.
  2 RELAXATION into a family: leave a side of the box open -- the king is then forced INTO that side: by CHECK (the rest of
    the box and its own square covered), or by ZUGZWANG (the rest covered, its own square not: it must move, and only
    there). The family's base case is the edge: with the open side off the board, the check variant is MATE and the
    zugzwang variant is STALEMATE (to avoid).
  3 ALLOCATION: placements of our pieces (relative to the lone king, a bounded window) that meet the relaxed demand --
    kings apart, a piece next to the lone king only if guarded (else it is taken), sliders blocked by our own pieces.
  4 GENERALIZATION: each piece's placements across the solutions described as a pattern -- a fixed offset, or a line
    (a row or column, at a distance) -- and whether the solutions are the product of those (independent roles).
  5 EXPLANATION: what each piece covers in every solution (its ROLE: the front row, the middle row, the check), and the
    roles no piece can take (our king never covers the middle row: it would touch).

The output is a derivation for a person and the statements the Lean proofs (Ideas/RookPush.lean ...) take up; the
enumeration over a window only SUGGESTS (Kaveh's rule: every proof in Lean, by reasoning).

  python -m catspace.incubator.analyst.mate_discovery --pieces KR
"""
from __future__ import annotations

import argparse
import itertools
from collections import Counter, defaultdict

BOX = [(x, y) for y in (1, 0, -1) for x in (-1, 0, 1)]          # back row y=+1, middle y=0, front y=-1
ROW = {1: "back", 0: "middle", -1: "front"}
DIRS = {"R": [(1, 0), (-1, 0), (0, 1), (0, -1)], "B": [(1, 1), (1, -1), (-1, 1), (-1, -1)]}
DIRS["Q"] = DIRS["R"] + DIRS["B"]
KNIGHT = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]


def attacks(kind: str, at, target, occupied) -> bool:
    """Does our piece of `kind` on `at` attack `target`, with `occupied` our other pieces (the lone king lifted)?"""
    if at == target:
        return False
    dx, dy = target[0] - at[0], target[1] - at[1]
    if kind == "K":
        return max(abs(dx), abs(dy)) == 1
    if kind == "N":
        return (dx, dy) in KNIGHT
    for ux, uy in DIRS[kind]:
        x, y = at
        while True:
            x, y = x + ux, y + uy
            if (x, y) == target:
                return True
            if (x, y) in occupied or abs(x) > 9 or abs(y) > 9:
                break
    return False


def coverage(pieces, placement):
    """The box squares covered by our pieces (a square holding one of ours counts only if another guards it)."""
    occ = set(placement)
    out = {}
    for sq in BOX:
        by = [i for i, (k, at) in enumerate(zip(pieces, placement)) if attacks(k, at, sq, occ - {at})]
        out[sq] = by
    return out


def solve(pieces: str, demand: set, check: bool, window: int = 4):
    """Placements meeting the demand: every demanded square covered; the king's own square covered iff `check`."""
    squares = [(x, y) for x in range(-window, window + 1) for y in range(-window, window + 1) if (x, y) != (0, 0)]
    sols = []
    for pl in itertools.product(squares, repeat=len(pieces)):
        if len(set(pl)) < len(pl):
            continue
        # kings apart; our pieces off the lone king's square
        if any(k == "K" and max(abs(p[0]), abs(p[1])) < 2 for k, p in zip(pieces, pl)):
            continue
        cov = coverage(pieces, pl)
        # a piece of ours next to the lone king must be guarded (else the king takes it -- a flight)
        ok = True
        for i, (k, p) in enumerate(zip(pieces, pl)):
            if max(abs(p[0]), abs(p[1])) == 1 and not cov.get(p):
                ok = False
        if not ok:
            continue
        if not all(cov[s] for s in demand if s != (0, 0)):
            continue
        if bool(cov[(0, 0)]) != check:
            continue
        sols.append((pl, cov))
    return sols


def describe(points):
    """A placement set as a pattern: a point, a row or a column (with the distances it takes), or a list."""
    pts = sorted(set(points))
    if len(pts) == 1:
        return f"fixed at {pts[0]}"
    ys = {p[1] for p in pts}; xs = {p[0] for p in pts}
    if len(ys) == 1:
        y = ys.pop()
        return f"on row y={y} ({ROW.get(y, 'outside the box')}), x in {sorted(xs)}"
    if len(xs) == 1:
        x = xs.pop()
        return f"on column x={x}, y in {sorted(ys)}"
    return f"{len(pts)} squares: {pts[:12]}{' ...' if len(pts) > 12 else ''}"


def roles(pieces, sols):
    """Per piece, the box rows it covers in every solution (its role)."""
    r = defaultdict(Counter)
    for pl, cov in sols:
        for i in range(len(pieces)):
            rows = tuple(sorted({ROW[s[1]] for s, by in cov.items() if i in by and s != (0, 0)} | ({"check"} if i in cov[(0, 0)] else set())))
            r[i][rows] += 1
    return r


GOALS = {
    "mate": (set(BOX), True),
    "retreat by check (back row open)": ({s for s in BOX if s[1] <= 0}, True),
    "retreat by zugzwang (back row open)": ({s for s in BOX if s[1] <= 0}, False),
    "corner-ward by check (back row + right column open)": ({s for s in BOX if s[1] <= 0 and s[0] <= 0}, True),
    "corner-ward by zugzwang (back row + right column open)": ({s for s in BOX if s[1] <= 0 and s[0] <= 0}, False),
}


def discover(pieces: str, window: int) -> None:
    print(f"OBS the discoverer, our pieces {'+'.join(pieces)} against a lone king (relative frame, window {window}); "
          f"a SUGGESTION from a bounded window -- the Lean proofs decide")
    # INSPIRATION FIRST (Kaveh 2026-10-12): the reasoned solutions whose pieces cover alike -- proposals, not proofs
    from .reasoned_library import retrieve
    for score, e in retrieve(pieces, 3):
        print(f"OBS   inspiration {score:.2f}: {e['id']} ({e['material']}) -- {e['relaxation']}; measure: {e['measure']}")
    for name, (demand, check) in GOALS.items():
        sols = solve(pieces, demand, check, window)
        if not sols:
            print(f"OBS   {name}: impossible in the window")
            continue
        print(f"OBS   {name}: {len(sols)} placements")
        proj = [[pl[i] for pl, _ in sols] for i in range(len(pieces))]
        for i, k in enumerate(pieces):
            print(f"OBS     our {k}: {describe(proj[i])}")
        prod = 1
        for i in range(len(pieces)):
            prod *= len(set(proj[i]))
        print(f"OBS     independent roles (the solutions are the product of the patterns): {prod == len(sols)}")
        for i, k in enumerate(pieces):
            for rws, n in roles(pieces, sols)[i].most_common(3):
                print(f"OBS     role of {k}: covers {', '.join(rws) or 'nothing'} ({n})")
    # THE ROLES NO PIECE CAN TAKE (the reverse assignment): our king, kept two away, never gives the check and covers at
    # most one middle-row square -- so the middle row (its two squares and the check) is never its role
    far = [p for p in itertools.product(range(-window, window + 1), repeat=2) if max(abs(p[0]), abs(p[1])) >= 2]
    check = [p for p in far if attacks("K", p, (0, 0), set())]
    most = max(sum(attacks("K", p, s, set()) for s in BOX if s[1] == 0 and s != (0, 0)) for p in far)
    print(f"OBS   our king two away: gives check from {len(check)} squares; covers at most {most} of the middle row's two "
          f"side squares -- the middle row is never its role (Kaveh: 'our king cannot cover the middle')")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="KR", help="our pieces besides... our king K first: KR, KQ, KBN")
    ap.add_argument("--window", type=int, default=4)
    a = ap.parse_args(argv)
    discover(a.pieces, a.window)


if __name__ == "__main__":
    main()
