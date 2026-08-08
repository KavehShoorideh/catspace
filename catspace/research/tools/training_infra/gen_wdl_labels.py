#!/usr/bin/env python
"""gen_wdl_labels.py -- labels v2: per-move ENGINE WDL for every legal move (Kaveh 2026-08-08).

One MultiPV search with UCI_ShowWDL per position returns every legal move with its own
(win, draw, loss) per-mille triple -- cardinal, three-axis ground truth for the three-pole
graded listwise loss: d(child->LOSS) graded by our-win, d(child->DRAW) by draw, d(child->WIN)
by opponent-win. Runs as a resumable background worker: appends shards, dedupes by row, so
coverage of the store grows monotonically toward 'every point' as idle engine-time allows.

Output (appended shards merged on load): row, child tok/glob (ragged via offsets), per-child
wdl (n,3) int16 per-mille from the CHILD MOVER's POV computed from parent-POV search.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import time

import chess
import chess.engine
import numpy as np

from catspace.io import paths


def _label_one(task):
    row, fen, nodes, syzygy = task
    import chess, chess.engine
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize
    b = chess.Board(fen)
    n_moves = b.legal_moves.count()
    if n_moves < 2:
        return None
    eng = chess.engine.SimpleEngine.popen_uci("stockfish")
    try:
        eng.configure({"UCI_ShowWDL": True, "SyzygyPath": syzygy, "Threads": 1, "Hash": 32})
        infos = eng.analyse(b, chess.engine.Limit(nodes=nodes), multipv=min(n_moves, 64))
    except Exception:
        eng.quit(); return None
    eng.quit()
    toks, globs, wdls, mids = [], [], [], []
    for inf in infos:
        if "pv" not in inf or not inf["pv"] or "wdl" not in inf:
            continue
        mv = inf["pv"][0]
        w, d, l = tuple(inf["wdl"])          # parent-mover POV, per-mille
        b.push(mv)
        tk, gl = tokenize(b)
        b.pop()
        toks.append(tk); globs.append(gl)
        _promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
        mids.append((mv.from_square, mv.to_square, _promo.get(mv.promotion, 4)))
        # child-mover POV = opponent: their (w,d,l) is our (l,d,w)
        wdls.append((l, d, w))
    if len(toks) < 2:
        return None
    return (row, np.array(toks, np.uint8), np.array(globs, np.uint8),
            np.array(wdls, np.int16), np.array(mids, np.uint8))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--n-piecedown", type=int, default=4000)
    ap.add_argument("--n-labels", type=int, default=20000)
    ap.add_argument("--nodes", type=int, default=30000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shard", default=None, help="output shard; default auto-numbered")
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
    rng = np.random.default_rng(int(time.time()) % 100000)

    # resume: skip rows already labeled in existing shards
    done = set()
    ddir = paths.derived("wdl_labels")
    os.makedirs(ddir, exist_ok=True)
    for f in os.listdir(ddir):
        if f.endswith(".npz"):
            done.update(np.load(os.path.join(ddir, f))["row"].tolist())
    rows = np.array([r for r in rng.permutation(rows) if r not in done])[:args.n_labels]
    print(f"[wdl] {len(done):,} already labeled; queueing {len(rows):,} new", flush=True)

    tasks = []
    for r in rows:
        b = row_to_board(tr.tok[r], tr.glob[r])
        if b.is_valid() and not b.is_game_over():
            tasks.append((int(r), b.fen(), args.nodes, str(paths.syzygy_dir())))
    shard = args.shard or os.path.join(ddir, f"shard_{int(time.time())}.npz")
    R, TK, GL, WD, MI, OFF = [], [], [], [], [], [0]
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for res in pool.imap_unordered(_label_one, tasks, chunksize=8):
            if res is None:
                continue
            row, tk, gl, wd, mi = res
            R.append(row); TK.append(tk); GL.append(gl); WD.append(wd); MI.append(mi)
            OFF.append(OFF[-1] + len(tk))
            if len(R) % 1000 == 0:
                print(f"[wdl] {len(R):,}/{len(tasks):,} "
                      f"[{(time.time()-t0)/len(R)*1000:.0f}ms/pos]", flush=True)
    np.savez_compressed(shard, row=np.array(R, np.int64),
                        tok=np.concatenate(TK), glob=np.concatenate(GL),
                        wdl=np.concatenate(WD), mid=np.concatenate(MI),
                        off=np.array(OFF, np.int64))
    print(f"[wdl] DONE {len(R):,} positions -> {shard} [{(time.time()-t0)/60:.0f}m]")


if __name__ == "__main__":
    main()
