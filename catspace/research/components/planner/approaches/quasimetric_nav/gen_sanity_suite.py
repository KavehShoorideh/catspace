#!/usr/bin/env python
"""gen_sanity_suite.py -- build the BEHAVIORAL sanity suite (Kaveh 2026-08-10): positions that
test CONDUCT, not readouts. No standard suite tests this (Bratko-Kopec/STS are best-move
batteries for the winning side); the swindle mechanisms (perpetual, stalemate, repetition)
are documented phenomena with no test format -- so this is ours, STS-style: category, graded
move list, machine-verified ground truth.

Categories (all 6-7 pieces -- ABOVE the 5-piece tablebase, the field must act, not the lookup):
  resist    -- mover is lost whatever they play, but the mate distances differ by >= 6 plies.
               Credit = chosen move's mate distance vs the maximum (prolong the game).
  save-draw -- mover is losing/worse but >= 1 move FORCES a draw (repetition, stalemate,
               fortress-by-50-move, TB draw) while >= 1 move loses. Credit = drawing move.
  convert   -- mover has a forced mate; >= 2 plies spread between fastest and slowest mate.
               Credit = shortening (chosen mate distance vs the minimum).

Ground truth: deterministic Stockfish search (fixed nodes, 1 thread, Syzygy on) with mate
scores -- an INSTRUMENT, like the arena referee and the tablebase; per Kaveh's rule engine
evals stay out of training labels.

    .venv/bin/python -m ...gen_sanity_suite [--target 150] [--nodes 300000]
writes artifacts/experiments/sanity_suite.jsonl: {fen, cat, moves: {uci: grade 0..10}}
"""
from __future__ import annotations

import argparse
import json
import random

import chess
import chess.engine

from catspace.io import paths

PIECES = [chess.QUEEN, chess.ROOK, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN,
          chess.PAWN]


def random_board(rng, n_pieces):
    while True:
        b = chess.Board(None)
        sqs = rng.sample(range(64), n_pieces)
        b.set_piece_at(sqs[0], chess.Piece(chess.KING, True))
        b.set_piece_at(sqs[1], chess.Piece(chess.KING, False))
        # bias toward LOPSIDED material: the interesting categories live where one side is
        # much stronger (resist/save-draw for the weak mover, convert for the strong)
        strong = rng.random() < 0.5                     # which colour gets the material
        for k in range(2, n_pieces):
            pt = rng.choice(PIECES)
            sq = sqs[k]
            if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                pt = chess.KNIGHT
            col = strong if rng.random() < 0.8 else (not strong)
            b.set_piece_at(sq, chess.Piece(pt, col))
        b.turn = rng.random() < 0.5
        if b.is_valid() and not b.is_game_over(claim_draw=True):
            return b


def classify(board, eng, nodes):
    """analyse every legal move; return (cat, {uci: grade}) or None."""
    n = board.legal_moves.count()
    if n < 3 or n > 40:
        return None
    try:
        infos = eng.analyse(board, chess.engine.Limit(nodes=nodes), multipv=n)
    except Exception:
        return None
    entries = []                                        # (uci, kind, val) kind: mate+/mate-/cp
    for inf in infos:
        if "pv" not in inf or not inf["pv"]:
            continue
        sc = inf["score"].pov(board.turn)
        if sc.is_mate():
            m = sc.mate()
            entries.append((inf["pv"][0].uci(), "m+" if m > 0 else "m-", abs(m)))
        else:
            entries.append((inf["pv"][0].uci(), "cp", sc.score()))
    if len(entries) < 3:
        return None
    kinds = {k for _, k, _ in entries}
    # RESIST: every move mated-in; spread >= 3 moves (6 plies)
    if kinds == {"m-"}:
        dists = [v for _, _, v in entries]
        lo, hi = min(dists), max(dists)
        if hi - lo >= 3:
            grades = {u: round(10 * (v - lo) / (hi - lo), 1) for u, _, v in entries}
            return "resist", grades
    # SAVE-DRAW: >= 1 exact draw (cp == 0: repetition/stalemate/TB), >= 1 mated-in, no wins
    draws = [u for u, k, v in entries if k == "cp" and v == 0]
    mated = [u for u, k, _ in entries if k == "m-"]
    wins = [u for u, k, v in entries if k == "m+" or (k == "cp" and v > 150)]
    if draws and mated and not wins:
        grades = {u: (10.0 if u in draws else 0.0) for u, _, _ in entries}
        for u, k, v in entries:                         # losing-but-not-mated: partial credit
            if k == "cp" and v < 0:
                grades[u] = 3.0
        return "save-draw", grades
    # CONVERT: >= 1 mate-in for the mover, spread >= 2 moves; non-mating moves graded low
    mates = [(u, v) for u, k, v in entries if k == "m+"]
    if mates:
        lo = min(v for _, v in mates)
        hi = max(v for _, v in mates)
        if len(mates) >= 2 and hi - lo >= 2:
            grades = {}
            for u, k, v in entries:
                if k == "m+":
                    grades[u] = round(10 * (hi - v) / (hi - lo), 1)
                else:
                    grades[u] = 0.0
            return "convert", grades
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=150, help="positions per category")
    ap.add_argument("--nodes", type=int, default=300_000)
    ap.add_argument("--max-tries", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or paths.experiment("sanity_suite.jsonl")
    rng = random.Random(args.seed)
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    eng.configure({"Threads": 1, "Hash": 256, "SyzygyPath": str(paths.syzygy_dir())})
    counts = {"resist": 0, "save-draw": 0, "convert": 0}
    kept = []
    tries = 0
    while min(counts.values()) < args.target and tries < args.max_tries:
        tries += 1
        b = random_board(rng, rng.choice([6, 6, 7]))
        r = classify(b, eng, args.nodes)
        if r is None:
            continue
        cat, grades = r
        if counts[cat] >= args.target:
            continue
        counts[cat] += 1
        kept.append({"fen": b.fen(), "cat": cat, "moves": grades})
        if sum(counts.values()) % 25 == 0:
            print(f"[suite] {counts}  ({tries:,} tries)", flush=True)
    eng.quit()
    with open(out_path, "w") as f:
        for row in kept:
            f.write(json.dumps(row) + "\n")
    print(f"[suite] DONE {counts} -> {out_path}  ({tries:,} tries)", flush=True)


if __name__ == "__main__":
    main()
