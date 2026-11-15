"""PLAN 78.49 stage A: does anything read NOW foretell a CONFIRMED threat move ARRIVING within the next plies (SYSTEM_SPEC section 5: go toward the tactics)?

Reads the per-position forcing reads kept by forcing_opportunity_measure.py (reads_fuse*.jsonl: per game, per ply, the mover's read `ours` and the other side's `theirs`).
A position is QUIET when neither side holds a CONFIRMED threat move now. For a quiet position at ply t and a horizon h (plies):
  ours arrives   = the mover holds a CONFIRMED threat move at one of its own turns t+2, t+4, ... <= t+h
  theirs arrives = the other side holds one at one of its own turns t+1, t+3, ... <= t+h
  target         = ours arrives - theirs arrives   (in -1, 0, 1)
Predictors read at t: the count of the mover's threat moves, of the other side's, their difference; the same for the NOT SETTLED ones only; a shuffle within the game as the control.
Printed: the base rates, and the rank correlation of each predictor with the target with a 90 pct interval by bootstrap over games. Odd game ids only (the test half of 78.48).

  .venv/bin/python -m catspace.incubator.analyst.discovery.forcing_arrival_measure --reads PATH [--draws N]
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from catspace.incubator.analyst.discovery.opportunity_measure import spearman

HORIZONS = (4, 8, 16)


def counts(read: "dict | None") -> "tuple[int, int, int]":
    """-> (threat moves, NOT SETTLED among them, CONFIRMED among them) of one read; an absent read counts nothing."""
    if read is None:
        return 0, 0, 0
    words = [t[0] for t in read["threats"]]
    return len(words), sum(1 for w in words if w in ("NOT SETTLED", "UNREAD")), sum(1 for w in words if w == "CONFIRMED")


def counts_of_game(plies: list) -> "tuple[list, list]":
    """(the mover's counts, the other side's counts) per position of a game, `counts` of each read; a position that is over counts nothing."""
    return [counts(p["ours"]) if p else (0, 0, 0) for p in plies], [counts(p["theirs"]) if p else (0, 0, 0) for p in plies]


def quiet_indices(plies: list, own: list, oth: list) -> list:
    """The positions of a game at which neither side holds a CONFIRMED threat move (the quiet ones); a position that is over is not."""
    return [t for t in range(len(plies)) if plies[t] is not None and not own[t][2] and not oth[t][2]]


def arrival_targets(own: list, n: int, t: int) -> list:
    """Per horizon, (the mover's CONFIRMED threat move arrives, the other side's arrives) as 0 or 1 for the quiet position t of a game of n positions; None where the horizon runs past the game."""
    targets = []
    for h in HORIZONS:
        if t + h >= n:
            targets.append(None)
            continue
        mine = any(own[u][2] for u in range(t + 2, t + h + 1, 2))
        theirs = any(own[u][2] for u in range(t + 1, t + h + 1, 2))
        targets.append((int(mine), int(theirs)))
    return targets


def rows_of_game(plies: list, rng: random.Random) -> list:
    """One row per quiet position: (predictors..., targets per horizon or None where the game ends before the horizon)."""
    own, oth = counts_of_game(plies)
    diffs = [own[t][0] - oth[t][0] for t in range(len(plies))]
    shuffled = diffs[:]
    rng.shuffle(shuffled)
    return [((own[t][0], oth[t][0], diffs[t], own[t][1] - oth[t][1], shuffled[t]), arrival_targets(own, len(plies), t)) for t in quiet_indices(plies, own, oth)]


NAMES = ("count of the mover's threat moves", "count of the other side's", "difference of the counts", "difference of the NOT SETTLED counts", "difference shuffled within the game")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--reads", required=True)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--tag", default="arrival")
    a = ap.parse_args(argv)
    rng = random.Random(7)
    games = []
    with open(a.reads) as f:
        for line in f:
            g = json.loads(line)
            if g["game"] % 2 == 1:
                games.append(rows_of_game(g["plies"], rng))
    print(f"OBS forcing arrival {a.tag}: test games {len(games)}, quiet positions {sum(len(g) for g in games)}")
    for hi, h in enumerate(HORIZONS):
        per_game = [[(r[0], r[1][hi]) for r in g if r[1][hi] is not None] for g in games]
        flat = [r for g in per_game for r in g]
        n = len(flat)
        mine = sum(r[1][0] for r in flat)
        theirs = sum(r[1][1] for r in flat)
        print(f"OBS forcing arrival {a.tag} h={h}: quiet positions with the horizon inside the game {n}; a CONFIRMED threat move arrives for the mover in {mine} ({100 * mine / n:.1f} pct), for the other side in {theirs} ({100 * theirs / n:.1f} pct)")

        def corr(sample: list, k: int, which: int) -> float:
            x = np.array([r[0][k] for r in sample], float)
            y = np.array([(r[1][0] - r[1][1]) if which == 2 else r[1][which] for r in sample], float)
            return spearman(x, y)

        brng = np.random.default_rng(11)
        draws = [brng.integers(0, len(per_game), len(per_game)) for _ in range(a.draws)]
        for k, name in enumerate(NAMES):
            for which, what in ((2, "ours less theirs"), (0, "ours arrives")):
                if which == 0 and k not in (0, 1):
                    continue
                v = corr(flat, k, which)
                bs = sorted(corr([r for i in d for r in per_game[i]], k, which) for d in draws)
                lo, hi_ = bs[int(0.05 * len(bs))], bs[int(0.95 * len(bs)) - 1]
                print(f"OBS forcing arrival {a.tag} h={h}: {name} against '{what}': rank correlation {v:+.4f} [{lo:+.4f}, {hi_:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
