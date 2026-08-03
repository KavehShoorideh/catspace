#!/usr/bin/env python
"""experiments/controlfield_gate2_wdl_decay.py -- gate 2 (known tactics), rerun
with the WDL-decay criterion (catspace/controlfield/wdl_decay.py) instead of the
control-field ascent cone. Kaveh's pivot (2026-08-02): don't use C at all here --
does the puzzle's solution move keep Stockfish's own win probability from
collapsing over a few moves of best play by both sides?
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.controlfield.wdl_decay import is_good_by_decay   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--k-moves", type=int, default=4)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--decay-tol", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()

    import pandas as pd
    import zstandard as zstd
    with open("data/lichess_db_puzzle.csv.zst", "rb") as f:
        data = zstd.ZstdDecompressor().stream_reader(f).read()
    df = pd.read_csv(io.BytesIO(data))
    theme_filter = {"mateIn2", "kingsideAttack"}
    mask = df["Themes"].fillna("").apply(lambda s: any(t in s.split() for t in theme_filter))
    pool = df[mask]
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(pool), min(args.n, len(pool)), replace=False)
    rows = pool.iloc[idx]

    eng = chess.engine.SimpleEngine.popen_uci(shutil.which("stockfish") or "/opt/homebrew/bin/stockfish")
    try:
        eng.configure({"UCI_ShowWDL": True})
    except Exception:
        pass

    tried, skipped, hits = 0, 0, 0
    for i, (_, row) in enumerate(rows.iterrows()):
        try:
            b = chess.Board(row["FEN"])
            moves = row["Moves"].split()
            if len(moves) < 2:
                skipped += 1
                continue
            b.push(chess.Move.from_uci(moves[0]))
            solution = chess.Move.from_uci(moves[1])
            if solution not in b.legal_moves or b.is_game_over():
                skipped += 1
                continue
            good, traj = is_good_by_decay(eng, b, solution, args.k_moves, args.depth, args.decay_tol)
            tried += 1
            hits += int(good)
        except Exception as e:
            skipped += 1
            print(f"  SKIP {row['PuzzleId']}: {type(e).__name__}: {e}", flush=True)
            continue
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(rows)} | hits so far {hits}/{tried} [{time.time() - t0:.0f}s]", flush=True)

    rate = hits / tried if tried else float("nan")
    print(f"VERDICT gate2-wdl-decay: tried={tried} skipped={skipped} hits={hits} "
          f"rate={rate:.1%} | (no control field used at all -- pure SF search) | "
          f"[{time.time() - t0:.0f}s]")
    eng.quit()


if __name__ == "__main__":
    main()
