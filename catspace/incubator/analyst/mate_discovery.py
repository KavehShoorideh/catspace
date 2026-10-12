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


# ---------------------------------------------------------------------------------------------------------------------
# THE OPERATORS MADE EXPLICIT (Kaveh 2026-10-12: "try it out" -- represent the reasoning so it is self-discovered): each
# operator takes the frames so far and adds a record (what it derived, from what); the trace is the plan and its
# explanation. Precedent (looked up): goal regression (REGRESS), difference reduction / rippling (MONOTONE), proof
# critics (SUPERSEDE), explanation-based generalization (the patterns from solutions).

def op_allocate(pieces, goal_name, window):
    demand, check = GOALS[goal_name]
    sols = solve(pieces, demand, check, window)
    if not sols:
        return None
    proj = {k: sorted({pl[i] for pl, _ in sols}) for i, k in enumerate(pieces)}
    indep = len(sols) == __import__("math").prod(len(v) for v in proj.values())
    rl = roles(pieces, sols)
    role = {k: rl[i].most_common(1)[0][0] for i, k in enumerate(pieces)}
    return dict(goal=goal_name, check=check, solutions=len(sols), squares=proj, independent=indep, role=role)


def line_of(points):
    """A placement set lying on one row (or column): ('row', y, xs) / ('col', x, ys), else None."""
    ys = {p[1] for p in points}; xs = {p[0] for p in points}
    if len(ys) == 1 and len(points) > 1:
        return ("row", next(iter(ys)), sorted(xs))
    if len(xs) == 1 and len(points) > 1:
        return ("col", next(iter(xs)), sorted(ys))
    return None


def op_hold(pieces, mover, rows, window):
    """REGRESS the step: before it our king is not yet in place, so ITS ROLE in the step (the box rows it covers) must be
    held by the moving piece alone -- allocate that demand to `mover` and find its squares (no check: the hold is
    quiet)."""
    front = {s for s in BOX if ROW[s[1]] in rows}
    squares = [(x, y) for x in range(-window, window + 1) for y in range(-window, window + 1) if max(abs(x), abs(y)) >= 2]
    out = []
    for at in squares:
        occ = set()
        if all(attacks(mover, at, s, occ) for s in front) and not attacks(mover, at, (0, 0), occ):
            out.append(at)
    return out


def op_monotone(pieces, wall_row_rel, samples=3000, seed=0):
    """MONOTONE on real boards (a sampled suggestion): with the rook on a rank below the lone king and our king off that
    rank, does any move of the lone king increase the ranks left beyond the rook (the king's side)?"""
    import random
    import chess
    rng = random.Random(seed)
    tested = broken = 0
    for _ in range(samples * 5):
        if tested >= samples:
            break
        rr, rf = rng.randrange(0, 7), rng.randrange(8)
        kr, kf = rng.randrange(rr + 1, 8), rng.randrange(8)
        wr, wf = rng.randrange(8), rng.randrange(8)
        if wr == rr or len({(rr, rf), (kr, kf), (wr, wf)}) < 3 or max(abs(kr - wr), abs(kf - wf)) < 2:
            continue
        b = chess.Board(None)
        b.set_piece_at(chess.square(rf, rr), chess.Piece(chess.ROOK, chess.WHITE))
        b.set_piece_at(chess.square(wf, wr), chess.Piece(chess.KING, chess.WHITE))
        b.set_piece_at(chess.square(kf, kr), chess.Piece(chess.KING, chess.BLACK))
        b.turn = chess.BLACK
        if not b.is_valid():
            continue
        tested += 1
        before = 7 - rr
        for m in b.legal_moves:
            if m.to_square != chess.square(rf, rr) and chess.square_rank(m.to_square) <= rr:
                broken += 1
                break
    return tested, broken


def plan(pieces: str, window: int) -> dict:
    """THE PLAN, derived by the operators from the material alone; returns the frame spec and prints the trace."""
    trace = []
    say = lambda op, txt: (trace.append((op, txt)), print(f"OBS   [{op}] {txt}"))
    print(f"OBS the plan for {'+'.join(pieces)} against a lone king, derived by the operators (window {window})")
    say("demand", "mate = every square of the king's 3x3 box covered, its own square by a check")
    mate = op_allocate(pieces, "mate", window)
    say("allocate", "mate in the open: " + ("possible" if mate else "impossible -- the goal needs the edge"))
    push = None
    for g in ("retreat by check (back row open)", "retreat by zugzwang (back row open)"):
        a = op_allocate(pieces, g, window)
        say("relax", f"{g}: " + (f"{a['solutions']} placements, independent roles {a['independent']}" if a else "impossible"))
        if a and a["independent"] and push is None:
            push = a
    if not push:
        say("fail", "no relaxation with independent roles -- the plan needs clustering by roles (not built)")
        return dict(pieces=pieces, trace=trace, frame=None)
    king = push["squares"]["K"]
    others = [k for k in pieces if k != "K"]
    mover = others[0]
    line = line_of(push["squares"][mover])
    say("allocate", f"the step ({push['goal']}): our king {king if len(king) > 1 else king[0]} covering {', '.join(push['role']['K'])}; "
        f"the {mover} {('on ' + line[0] + ' ' + str(line[1]) + ' at ' + str(line[2])) if line else push['squares'][mover]} covering {', '.join(push['role'][mover])}")
    say("invariance", f"the {mover}'s squares in the step form one class -- " + (f"its {line[0]} {line[1]}, any of {line[2]}: moving along it keeps "
        f"the step (a PASS move); every square of it is two or more from the king (safe)" if line else "no single line"))
    krows = [r for r in push["role"]["K"] if r != "check"]
    hold = op_hold(pieces, mover, krows, window)
    hline = line_of(hold)
    say("regress", f"before the step our king is not yet in place: its role ({', '.join(krows)} row) must be held by the {mover} alone -- "
        f"{('its ' + hline[0] + ' ' + str(hline[1]) + ' at ' + str(hline[2])) if hline else str(hold[:8])}: a WALL one row "
        f"in front of the king; the step is the {mover} moving up its column from the wall onto the king's row")
    t, br = op_monotone(pieces, -1)
    say("monotone", f"with the {mover} as a wall (our king off its line), the lone king never crosses it: {t} sampled "
        f"boards, {br} crossings -- rows beyond the wall never grow (a suggestion; the Lean cut lemma decides)")
    opp = king[0] if len(king) == 1 else None
    say("assemble", f"family: rows beyond the wall (each step moves the wall up one row); subgoal (regressed): our king "
        f"on {opp} relative to the lone king; measure = (rows beyond the wall, our king's distance to that square); "
        f"pass = the {mover} along the wall; base case: one row left -- the open side is off the board, the step is MATE")
    frame = dict(pieces=pieces, wall_piece=mover, opposition=opp, step=push["goal"],
                 measure=["rows beyond the wall", "our king's distance to the opposition square"],
                 pass_move=f"{mover} along the wall", safe=f"{mover} two or more files from the king")
    return dict(pieces=pieces, trace=trace, frame=frame)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--pieces", default="KR", help="our pieces, our king K first: KR, KQ, KBN")
    ap.add_argument("--window", type=int, default=4)
    ap.add_argument("--plan", action="store_true", help="derive the whole plan by the operators and write the frame spec")
    a = ap.parse_args(argv)
    if a.plan:
        import json
        from catspace.io import paths
        out = plan(a.pieces, a.window)
        with open(paths.derived(f"frame_{a.pieces}.json"), "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"OBS   frame spec -> {paths.derived(f'frame_{a.pieces}.json')}")
    else:
        discover(a.pieces, a.window)


if __name__ == "__main__":
    main()
