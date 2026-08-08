#!/usr/bin/env python
"""gen_bestmove_labels.py -- engine best-move labels for GRADIENT SUPERVISION of the field.

Kaveh 2026-08-08: the field's discrete gradient at a node is the best-move arrow; rather than
model arrows instead of the field (which forfeits integrability), we supervise the FIELD's
gradient: d(best_child -> ref) must beat d(other_child -> ref). This script precomputes the
labels offline so training pays nothing per step.

For N sampled fit-split SF-store positions: shallow SF (SyzygyPath set, so endgame arrows are
tb-exact) gives the best move; one random OTHER legal move is stored as the contrast. Output NPZ:
row index, best-child (tok, glob), alt-child (tok, glob).
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import time

import chess
import chess.engine
import numpy as np

from catspace.io import paths


def _label_one(task):
    seed, fen, nodes, syzygy = task
    import chess, chess.engine, random
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
    b = chess.Board(fen)
    moves = list(b.legal_moves)
    if len(moves) < 2:
        return None
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    try:
        eng.configure({"SyzygyPath": syzygy, "Threads": 1, "Hash": 32})
        r = eng.play(b, chess.engine.Limit(nodes=nodes))
        if r.move is None:
            return None
        best = r.move
        rnd = random.Random(seed)
        others = [m for m in moves if m != best]
        alt = others[rnd.randrange(len(others))]
        outs = []
        for mv in (best, alt):
            b.push(mv)
            outs.append(tokenize(b))
            b.pop()
        return (outs[0][0], outs[0][1], outs[1][0], outs[1][1])
    finally:
        eng.quit()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=4000, help="store shape (must match training)")
    ap.add_argument("--n-piecedown", type=int, default=4000)
    ap.add_argument("--sf-only", action="store_true", default=True)
    ap.add_argument("--n-labels", type=int, default=30000)
    ap.add_argument("--nodes", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=paths.derived("bestmove_labels.npz"))
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T
    from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
        split_by_game)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.eval_dtz_gate import (
        row_to_board)
    t0 = time.time()
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, max_plies=400,
                 n_piecedown=args.n_piecedown, verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), args.seed)
    fit_games = np.flatnonzero(split == 0)
    game = tr.game_of_row()
    rows = np.flatnonzero(np.isin(game, fit_games))
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(rows, min(args.n_labels * 2, len(rows)), replace=False)

    tasks = []
    for r in rows:
        b = row_to_board(tr.tok[r], tr.glob[r])
        if b.is_valid() and not b.is_game_over():
            tasks.append((int(r), b.fen(), args.nodes, str(paths.syzygy_dir())))
        if len(tasks) >= args.n_labels:
            break
    print(f"[labels] {len(tasks):,} positions queued [{time.time()-t0:.0f}s]", flush=True)

    keep_rows, bt, bg, at_, ag = [], [], [], [], []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for (r, *_), res in zip(tasks, pool.imap(_label_one,
                                                [(t[0], t[1], t[2], t[3]) for t in tasks],
                                                chunksize=16)):
            if res is None:
                continue
            keep_rows.append(r)
            bt.append(res[0]); bg.append(res[1]); at_.append(res[2]); ag.append(res[3])
            if len(keep_rows) % 2000 == 0:
                print(f"[labels] {len(keep_rows):,} done "
                      f"[{(time.time()-t0)/len(keep_rows)*1000:.0f}ms/label]", flush=True)
    np.savez_compressed(args.out, row=np.array(keep_rows, np.int64),
                        best_tok=np.array(bt, np.uint8), best_glob=np.array(bg, np.uint8),
                        alt_tok=np.array(at_, np.uint8), alt_glob=np.array(ag, np.uint8))
    print(f"[labels] DONE {len(keep_rows):,} -> {args.out} [{(time.time()-t0)/60:.0f}m]")


if __name__ == "__main__":
    main()
