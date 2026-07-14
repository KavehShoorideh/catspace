#!/usr/bin/env python
"""
experiments/gen_confirmatory_starts.py — mint the FROZEN confirmatory start set.

data_registry.json contract: confirmatory starts are generated FRESH with seed
777 only at confirmatory time and never reused. This script enforces the
contract mechanically: it refuses to overwrite an existing output (a consumed
confirmatory set is burned -- generate a new one with a new registered seed if
another confirmatory round is ever pre-registered), verifies every start is a
tablebase WIN for White (wdl=2, no cursed wins), and excludes any FEN present
in the TRAIN or EVAL sets.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.selfplay_generate import random_endgame_start
from experiments.value_fixed_point import TB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--material", default="krrkbp")
    ap.add_argument("--exclude", nargs="+",
                    default=["artifacts/experiments/krrkbp_win_starts.json",
                             "artifacts/experiments/krrkbp_test_n200.json",
                             "artifacts/experiments/krrkbp_fixed_set_n60.json"])
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--out",
                    default="artifacts/experiments/confirmatory_krrkbp_seed777_n120.json")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        sys.exit(f"REFUSING: {out} already exists -- a confirmatory set is single-use "
                 f"(data_registry.json). Pre-register a new seed for a new set.")

    taken = set()
    for p in args.exclude:
        if Path(p).exists():
            taken.update(json.loads(Path(p).read_text())["fens"])
    rng = np.random.default_rng(args.seed)
    tb = TB(args.syzygy_dir)
    fens: list[str] = []
    tried = 0
    while len(fens) < args.n and tried < 200_000:
        tried += 1
        b = random_endgame_start(rng, args.material)
        if b is None or b.turn != chess.WHITE:
            continue
        fen = b.fen()
        if fen in taken or fen in fens:
            continue
        w, _ = tb.wdl_dtz(b)
        if w == 2:                       # clean tablebase win for the mover (White)
            fens.append(fen)
    tb.close()
    if len(fens) < args.n:
        sys.exit(f"only found {len(fens)}/{args.n} verified wins after {tried} tries")
    out.write_text(json.dumps({"fens": fens, "seed": args.seed,
                               "material": args.material,
                               "verified": "syzygy wdl=2, White to move",
                               "excluded_sets": args.exclude}))
    print(f"-> {out}  ({len(fens)} fresh tb-verified wins, seed {args.seed}, "
          f"{tried} candidates tried)")


if __name__ == "__main__":
    main()
