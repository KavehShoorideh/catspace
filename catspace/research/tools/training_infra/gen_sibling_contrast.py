#!/usr/bin/env python
"""gen_sibling_contrast.py -- cache (parent, played child, K random siblings) for contrastive training.

THE MEASURED PROBLEM THIS EXISTS TO FIX. Every learned component in the stack sits at chance when
asked to pick between sibling moves -- cef3's committor 49.3% on-track against 52.1% for a random
legal move, csg1's B block 43.3%, cqm1's geometry 44.2%, all beaten by a one-line material rule at
58.7-66.7%. And the cause is not capacity: 100% of siblings already receive DISTINCT concept
states, and the full state is 95% unique across the corpus.

The cause is the DATA. Every position in the corpus is one Stockfish played INTO. The model has
never seen "here are 30 children of this position, one is good and 29 are bad" -- it is asked to
rank siblings having never been shown a sibling comparison. Between-position accuracy (0.603 on
414-class resolution against a 0.275 base) is real and does not transfer, which is exactly what
that gap predicts.

THE LABEL IS FREE, and needs no engine in the loss -- a discipline this project has held to
deliberately. The move Stockfish actually played is, on average, better than a random legal
alternative. That is a ranking signal available from the corpus alone.

Cached offline because generating children inside the training loop means python-chess board
manipulation per step, which would dominate a 0.2 s/step budget. Stored as raw (tok, glob) so the
trainer just indexes.

    .venv/bin/python -m ...gen_sibling_contrast --n 400000 --k 7 --save
"""
from __future__ import annotations

import argparse
import time

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=400000, help="parent positions to expand")
    ap.add_argument("--k", type=int, default=7, help="random siblings kept per parent")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    import chess
    from catspace.research.tools.training_infra.concepts_reference import (
        TOK_PIECES, board_from_row)
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    gor = tr.game_of_row()
    rng = np.random.default_rng(args.seed)
    cand = np.flatnonzero((np.arange(n) + 1 < n) & (gor[np.clip(np.arange(n) + 1, 0, n - 1)] == gor))
    sel = cand[rng.choice(len(cand), min(args.n * 2, len(cand)), replace=False)]

    K = args.k
    p_row = np.zeros(args.n, np.int64)
    pos_tok = np.zeros((args.n, 64), np.uint8); pos_glb = np.zeros((args.n, 6), np.uint8)
    neg_tok = np.zeros((args.n, K, 64), np.uint8); neg_glb = np.zeros((args.n, K, 6), np.uint8)
    n_neg = np.zeros(args.n, np.int8)

    def tg(b):
        t = np.zeros(64, np.uint8)
        for q, p in b.piece_map().items():
            t[q] = TOK_PIECES.index((p.piece_type, p.color)) + 1
        g = np.array([int(b.turn), int(b.has_kingside_castling_rights(chess.WHITE)),
                      int(b.has_queenside_castling_rights(chess.WHITE)),
                      int(b.has_kingside_castling_rights(chess.BLACK)),
                      int(b.has_queenside_castling_rights(chess.BLACK)),
                      0 if b.ep_square is None else chess.square_file(b.ep_square) + 1], np.uint8)
        return t, g

    t0 = time.time(); w = 0
    for r in sel:
        if w >= args.n:
            break
        b = board_from_row(tr.tok[r], tr.glob[r])
        mv = list(b.legal_moves)
        if len(mv) < 3:
            continue
        nxt = tr.tok[r + 1]
        played, kids = None, []
        for m in mv:
            b.push(m); t, g = tg(b); b.pop()
            if played is None and np.array_equal(t, nxt):
                played = (t, g)
            else:
                kids.append((t, g))
        if played is None or not kids:
            continue
        p_row[w] = r
        pos_tok[w], pos_glb[w] = played
        idx = rng.choice(len(kids), min(K, len(kids)), replace=False)
        for j, ii in enumerate(idx):
            neg_tok[w, j], neg_glb[w, j] = kids[ii]
        n_neg[w] = len(idx)
        w += 1
        if w % 50000 == 0:
            el = time.time() - t0
            print(f"  {w:,}/{args.n:,}  {w/el:,.0f} pos/s  eta {(args.n-w)/(w/el)/60:.1f}m",
                  flush=True)

    print(f"\n  built {w:,} parents in {(time.time()-t0)/60:.1f}m | "
          f"mean siblings kept {n_neg[:w].mean():.1f}")
    if args.save:
        p = paths.derived(f"sibling_contrast_{K}_{n}.npz")
        np.savez_compressed(p, p_row=p_row[:w], pos_tok=pos_tok[:w], pos_glb=pos_glb[:w],
                            neg_tok=neg_tok[:w], neg_glb=neg_glb[:w], n_neg=n_neg[:w])
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
