#!/usr/bin/env python
"""
experiments/mine_mate_puzzles.py — permanent mate-in-N benchmark sets from the
Lichess puzzle database (Kaveh 2026-07-16: "these toy examples (mate in 1, 2,
3) will be useful later. You can mine lichess for these").

Streams lichess_db_puzzle.csv.zst (FEN,Moves,...,Themes), keeps puzzles themed
mateInN. Lichess convention: the stored FEN is the position BEFORE the
opponent's setup move; Moves[0] is that setup move, so the benchmark position
is FEN + Moves[0], side-to-move = the mating side. Each kept row is VERIFIED
by replaying the full solution line and checking it ends in checkmate with the
right number of our moves (engine-labeled upstream; this re-check is local
and exact for the given line -- it certifies "a mate-in-N line exists", the
theme label certifies forcedness).

Output: artifacts/experiments/mate_in_{N}_n{K}.json  {"fens": [...], "meta": ...}
EVAL-ONLY sets: never train on these (they are full-board, human-game
positions -- exactly the distribution the full-board committor trains on, so
training on them would leak the benchmark).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import subprocess
import sys
from pathlib import Path

import chess

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--puzzle-db", default="data/lichess_db_puzzle.csv.zst")
    ap.add_argument("--per-n", type=int, default=500)
    ap.add_argument("--max-n", type=int, default=3)
    ap.add_argument("--min-rating", type=int, default=800)
    ap.add_argument("--out-dir", default="artifacts/experiments")
    args = ap.parse_args()

    want = {n: [] for n in range(1, args.max_n + 1)}
    theme_of = {f"mateIn{n}": n for n in range(1, args.max_n + 1)}

    proc = subprocess.Popen(["zstd", "-dc", args.puzzle_db], stdout=subprocess.PIPE)
    reader = csv.reader(io.TextIOWrapper(proc.stdout, encoding="utf-8"))
    header = next(reader)
    col = {k: header.index(k) for k in ("FEN", "Moves", "Rating", "Themes")}
    scanned = kept = 0
    for row in reader:
        scanned += 1
        if all(len(v) >= args.per_n for v in want.values()):
            break
        themes = row[col["Themes"]].split()
        ns = [theme_of[t] for t in themes if t in theme_of]
        if len(ns) != 1 or len(want[ns[0]]) >= args.per_n:
            continue
        if int(row[col["Rating"]]) < args.min_rating:
            continue
        n = ns[0]
        moves = row[col["Moves"]].split()
        if len(moves) != 2 * n:          # setup move + 2n-1 solution plies
            continue
        b = chess.Board(row[col["FEN"]])
        try:
            for u in moves:
                b_prev = b.copy(stack=False)
                b.push(chess.Move.from_uci(u))
        except (ValueError, AssertionError):
            continue
        if not b.is_checkmate():
            continue
        start = chess.Board(row[col["FEN"]])
        start.push(chess.Move.from_uci(moves[0]))
        want[n].append(start.fen())
        kept += 1
    proc.terminate()

    for n, fens in want.items():
        out = Path(args.out_dir) / f"mate_in_{n}_n{len(fens)}.json"
        out.write_text(json.dumps(dict(
            fens=fens,
            meta=dict(source="lichess_db_puzzle.csv.zst", theme=f"mateIn{n}",
                      min_rating=args.min_rating, role="EVAL-ONLY benchmark",
                      note="position after the setup move; side to move mates "
                           f"in {n}; solution line verified to end in mate"))))
        print(f"VERDICT MATE_SET mate_in_{n}: {len(fens)} verified positions -> {out}")
    print(f"scanned {scanned} puzzles, kept {kept}")


if __name__ == "__main__":
    main()
