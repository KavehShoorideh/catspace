"""PLAN 79.10 D (SYSTEM_SPEC 8 "the whole is judged, not the piece"; piece: the wake): THE EQUAL CLOCK, for the measure. The old player does not spend the clock it is given (its wall time is about a third to
three fifths of it: it ends a reading when the depth is complete or the fuse is reached), while the wake spends its whole clock; so a comparison at the SAME CLOCK GIVEN is not a comparison at the same cost.
This module gives the old player's caller a WALL TARGET: `at_wall(call, target)` finds, by a few calls, the clock at which the old player's wall time is near the wake's, and returns that run. Nothing is tuned:
the clock is scaled by the ratio of the target to the wall just seen (the wall is nearly proportional to the clock), at most `tries` calls, kept inside [`floor` x target, `ceiling` x target]; the run whose wall is
nearest the target is the one returned, with every (clock, wall) tried, and `reached` False when it could not come within `tol` (the old player cannot spend more, or the clock bound was met): said, never hidden.

  at_wall(call, target) -> dict(row=<what call returned>, clock=, wall=, tried=[(clock, wall)], reached=bool)
`call(clock)` returns a dict with the key "wall" (seconds spent). Game-free; no engine is imported here.
"""
from __future__ import annotations

TOL = 0.15          # the wall counts as matched within this share of the target
TRIES = 6
FLOOR = 0.5         # the clock given never goes below this share of the target
CEILING = 8.0       # nor above this many times the target


def at_wall(call, target: float, *, tol: float = TOL, tries: int = TRIES, floor: float = FLOOR, ceiling: float = CEILING) -> dict:
    lo, hi = target * floor, target * ceiling
    clock = target
    tried: list = []
    best = None
    for _ in range(tries):
        row = call(clock)
        wall = float(row["wall"])
        tried.append((clock, wall))
        if best is None or abs(wall - target) < abs(best[2] - target):
            best = (row, clock, wall)
        if abs(wall - target) <= tol * target:
            break
        nxt = min(hi, max(lo, clock * target / max(wall, 1e-6)))
        if abs(nxt - clock) < 1e-9 or any(abs(nxt - c) < 1e-9 for c, _ in tried):
            break
        clock = nxt
    row, clock, wall = best
    return {"row": row, "clock": clock, "wall": wall, "tried": tried, "reached": abs(wall - target) <= tol * target}
