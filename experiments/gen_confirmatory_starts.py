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

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.selfplay_generate import (KRRKBP_FIXED_START,
                                           openings_from_fixed_start)
from experiments.value_fixed_point import TB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--start-fen", default=KRRKBP_FIXED_START,
                    help="canonical start the confirmatory openings derive from "
                         "(2026-07-14: no random placements; must match the "
                         "distribution the candidate was trained/evaluated on)")
    ap.add_argument("--min-plies", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=10)
    ap.add_argument("--exclude", nargs="+",
                    default=["artifacts/experiments/krrkbp_fixed_train_n700.json",
                             "artifacts/experiments/krrkbp_fixed_test_n200.json"])
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
    # oversample, then drop anything colliding with train/eval sets
    pool = openings_from_fixed_start(rng, args.n + len(taken) + 200, tb,
                                     start_fen=args.start_fen,
                                     min_plies=args.min_plies, max_plies=args.max_plies)
    tb.close()
    fens = [f for f in pool if f not in taken][:args.n]
    if len(fens) < args.n:
        sys.exit(f"only found {len(fens)}/{args.n} fresh verified openings")
    out.write_text(json.dumps({"fens": fens, "seed": args.seed,
                               "start": args.start_fen,
                               "verified": "syzygy wdl=2, White to move, play-reachable",
                               "excluded_sets": args.exclude}))
    print(f"-> {out}  ({len(fens)} fresh tb-verified wins, seed {args.seed}, "
          f"{tried} candidates tried)")


if __name__ == "__main__":
    main()
