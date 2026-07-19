#!/usr/bin/env python
"""experiments/gen_forced_mate_data.py — curate FULL-BOARD forced-mate positions
from the Lichess puzzle DB (Kaveh 2026-07-19: "curate a set of forced mate
situations, in which the distance can be hinged to the number of moves it took
to mate"). A second mate-SURFACE data source beyond the <=6-piece tablebase
(gen_dtm_data.py): mate-in-N puzzles are full-board and NEAR mate, so they give
the composed hinge full-board near-mate ANCHORS -- the missing piece for
propagating the mate value back into middlegames/openings.

Each mateInN puzzle: FEN + Moves. Moves[0] is the opponent's setup move; after it
the SOLVER is to move with a forced mate in N. We store that position with
dtm = 2N-1 plies (the main-line ply distance) and result = +1 (white mates) /
-1 (black mates), matching gen_dtm_data's packed/meta/dtm layout so the two
datasets concatenate.

Usage:
  .venv/bin/python experiments/gen_forced_mate_data.py --cap 40000 \
    --out data/derived/forced_mate.npz
"""
from __future__ import annotations

import argparse
import csv
import io
import re
import sys
import time
from pathlib import Path

import chess
import numpy as np
import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import encode_meta, encode_packed

_MATE_RE = re.compile(r"mateIn(\d)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--puzzles", default="data/lichess_db_puzzle.csv.zst")
    ap.add_argument("--out", default="data/derived/forced_mate.npz")
    ap.add_argument("--cap", type=int, default=40000, help="max positions to keep")
    ap.add_argument("--max-n", type=int, default=5, help="keep mateIn1..max-n")
    args = ap.parse_args()

    packs, metas, dtms, results, ns = [], [], [], [], []
    kept = scanned = 0
    t0 = time.time()
    with open(args.puzzles, "rb") as f:
        reader = csv.reader(io.TextIOWrapper(
            zstandard.ZstdDecompressor().stream_reader(f), encoding="utf-8"))
        header = next(reader)
        fi, mi, ti = header.index("FEN"), header.index("Moves"), header.index("Themes")
        for row in reader:
            scanned += 1
            m = _MATE_RE.search(row[ti])
            if not m:
                continue
            n = int(m.group(1))
            if n < 1 or n > args.max_n:
                continue
            try:
                board = chess.Board(row[fi])
                mv = row[mi].split()
                if not mv:
                    continue
                board.push(chess.Move.from_uci(mv[0]))        # opponent's setup move
            except (ValueError, AssertionError):
                continue
            if board.is_game_over():
                continue
            solver_white = board.turn == chess.WHITE           # side to move = the mater
            packs.append(encode_packed(board)); metas.append(encode_meta(board))
            dtms.append(2 * n - 1)                              # main-line plies to mate
            results.append(1 if solver_white else -1)
            ns.append(n)
            kept += 1
            if kept % 5000 == 0:
                print(f"  kept {kept}/{args.cap} (scanned {scanned}, {kept/(time.time()-t0):.0f}/s)",
                      flush=True)
            if kept >= args.cap:
                break

    packed = np.stack(packs); meta = np.stack(metas)
    dtm = np.array(dtms, dtype=np.float32); result = np.array(results, dtype=np.int8)
    nmoves = np.array(ns, dtype=np.int8)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, packed=packed, meta=meta, dtm=dtm, result=result, nmoves=nmoves)
    w = int((result == 1).sum()); b = int((result == -1).sum())
    print(f"[stage] {kept} forced-mate positions ({scanned} scanned): {time.time()-t0:.1f}s")
    print(f"VERDICT FORCED_MATE n={kept} white_mates={w} black_mates={b} "
          f"dtm[min={dtm.min():.0f} med={np.median(dtm):.0f} max={dtm.max():.0f}] -> {args.out}")


if __name__ == "__main__":
    main()
