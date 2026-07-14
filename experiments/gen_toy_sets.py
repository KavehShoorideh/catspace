#!/usr/bin/env python
"""
experiments/gen_toy_sets.py — mint the fixed-start toy TRAIN/EVAL sets.

Kaveh 2026-07-14: no random start positions. The toy has ONE canonical start
(KRRKBP_FIXED_START, verified syzygy wdl=2); every train/eval position must be
REACHABLE from it by play. This mints:
  krrkbp_fixed_train_n700.json  -- rollout/table starts (TRAIN)
  krrkbp_fixed_test_n200.json   -- playout money-test starts (EVAL, disjoint)
Both are White-to-move, still tablebase-won openings sampled by random legal
play from the canonical start (2-10 plies). Registry roles are updated by the
caller (data_registry.json); the old random-placement sets become LEGACY.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.selfplay_generate import KRRKBP_FIXED_START, openings_from_fixed_start
from experiments.value_fixed_point import TB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-fen", default=KRRKBP_FIXED_START,
                    help="the canonical start every position derives from (must be a "
                         "tablebase win for the side to move; verified before minting)")
    ap.add_argument("--n-train", type=int, default=700)
    ap.add_argument("--n-test", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-plies", type=int, default=2)
    ap.add_argument("--max-plies", type=int, default=10)
    ap.add_argument("--syzygy-dir", default="data/syzygy")
    ap.add_argument("--out-dir", default="artifacts/experiments")
    args = ap.parse_args()

    import chess
    tb = TB(args.syzygy_dir)
    start = chess.Board(args.start_fen)
    w, _ = tb.wdl_dtz(start)
    assert start.is_valid() and w == 2, \
        f"start must be a clean tablebase win for the mover (wdl=2), got wdl={w}"
    rng = np.random.default_rng(args.seed)
    pool = openings_from_fixed_start(rng, args.n_train + args.n_test, tb,
                                     start_fen=args.start_fen,
                                     min_plies=args.min_plies, max_plies=args.max_plies)
    tb.close()
    if len(pool) < args.n_train + args.n_test:
        sys.exit(f"only {len(pool)} distinct openings found; want "
                 f"{args.n_train + args.n_test} -- raise --max-plies")
    order = np.random.default_rng(args.seed + 1).permutation(len(pool))
    train = [pool[i] for i in order[:args.n_train]]
    test = [pool[i] for i in order[args.n_train:args.n_train + args.n_test]]

    meta = dict(start=KRRKBP_FIXED_START, seed=args.seed,
                plies=[args.min_plies, args.max_plies],
                verified="syzygy wdl=2, White to move, reachable by play from the fixed start")
    out = Path(args.out_dir)
    (out / "krrkbp_fixed_train_n700.json").write_text(json.dumps(dict(fens=train, **meta)))
    (out / "krrkbp_fixed_test_n200.json").write_text(json.dumps(dict(fens=test, **meta)))
    print(f"-> {out}/krrkbp_fixed_train_n700.json ({len(train)}) and "
          f"krrkbp_fixed_test_n200.json ({len(test)}), disjoint, from {KRRKBP_FIXED_START!r}")


if __name__ == "__main__":
    main()
