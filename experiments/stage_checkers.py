#!/usr/bin/env python
"""
experiments/stage_checkers.py — EVAL-ONLY named-stage detectors + validation.

Contract (JOURNAL 2026-07-15): these named concepts exist ONLY for our offline
verification of games; play/search never sees them. Stages (Kaveh):
  pin_pawn / pin_bishop      absolute pin of the black pawn/bishop
  double_attack              pawn AND bishop simultaneously attacked
  capture_pawn / capture_bishop
  king_edge / king_corner    black king driven to edge / corner
  midboard_trap              black king NOT on edge, zero safe squares, not in check
  mate_edge / mate_midboard  mate classified by king location (mate needn't be
                             on the edge -- tracked separately on purpose)

Validation mode: run over WON games from a rollout dump (tb-optimal White =
expert demonstrations -- every stage that's part of "how it's done" must fire
at a sane rate, or the checker is wrong).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import chess
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def stage_flags(b: chess.Board) -> dict:
    f = {}
    pawns = list(b.pieces(chess.PAWN, chess.BLACK))
    bishops = list(b.pieces(chess.BISHOP, chess.BLACK))
    f["pin_pawn"] = any(b.is_pinned(chess.BLACK, s) for s in pawns)
    f["pin_bishop"] = any(b.is_pinned(chess.BLACK, s) for s in bishops)
    f["double_attack"] = (bool(pawns) and bool(bishops)
                          and any(b.is_attacked_by(chess.WHITE, s) for s in pawns)
                          and any(b.is_attacked_by(chess.WHITE, s) for s in bishops))
    f["capture_pawn"] = not pawns
    f["capture_bishop"] = not bishops
    k = b.king(chess.BLACK)
    rank, file = chess.square_rank(k), chess.square_file(k)
    on_edge = rank in (0, 7) or file in (0, 7)
    f["king_edge"] = on_edge
    f["king_corner"] = rank in (0, 7) and file in (0, 7)
    safe = [s for s in b.attacks(k)
            if not b.is_attacked_by(chess.WHITE, s) and b.color_at(s) != chess.BLACK]
    mate = b.is_checkmate() and b.turn == chess.BLACK
    f["midboard_trap"] = (not on_edge) and not safe and not b.is_check() and not mate
    f["mate_edge"] = mate and on_edge
    f["mate_midboard"] = mate and not on_edge
    return f


STAGES = ["pin_pawn", "pin_bishop", "double_attack", "capture_pawn",
          "capture_bishop", "king_edge", "king_corner", "midboard_trap",
          "mate_edge", "mate_midboard"]


def annotate_game(fens: list) -> dict:
    """First ply each stage becomes true (None = never)."""
    first = {s: None for s in STAGES}
    for i, fen in enumerate(fens):
        fl = stage_flags(chess.Board(fen))
        for s in STAGES:
            if fl[s] and first[s] is None:
                first[s] = i
    return first


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", default="artifacts/experiments/rollout_dump_eps05.jsonl")
    ap.add_argument("--max-games", type=int, default=500)
    ap.add_argument("--won-only", action="store_true", default=True)
    args = ap.parse_args()

    firsts, n = {s: [] for s in STAGES}, 0
    for line in open(args.dump):
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if args.won_only and not rec["won"]:
            continue
        fens = [t[0] for t in rec["traj"]]
        # traj excludes the terminal board; reconstruct it by replay? The dump
        # stores pre-move states only, so mate flags come from the last state's
        # successors -- cheaper: skip mate stage here unless terminal stored.
        ann = annotate_game(fens)
        for s in STAGES:
            firsts[s].append(ann[s])
        n += 1
        if n >= args.max_games:
            break
    print(f"validated on {n} WON expert games ({args.dump})")
    print(f"{'stage':16s} {'fired':>6s} {'median first ply':>17s}")
    for s in STAGES:
        v = firsts[s]
        hit = [x for x in v if x is not None]
        med = f"{int(np.median(hit))}" if hit else "-"
        print(f"{s:16s} {len(hit)/max(n,1):6.1%} {med:>17s}")


if __name__ == "__main__":
    main()
