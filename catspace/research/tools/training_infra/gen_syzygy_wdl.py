#!/usr/bin/env python
"""gen_syzygy_wdl.py -- EXACT win/draw/loss for every corpus position inside the tablebase.

Kaveh 2026-08-16: "eventually we will reach a tablebase five piece ending, and when we do we
will anchor it to the particular tablebase lookup... so it's gonna be one hundred percent,
either win or lost or drawn."

THIS IS THE ONLY GROUND TRUTH IN THE PROJECT. Every other value signal we have is an estimate:
the game's final result is one noisy sample of one playout, and Stockfish's WDL is an engine's
opinion at a node budget. Syzygy is the solved answer. We hold the complete 3/4/5-man set (145
tables: 5 + 30 + 110), which covers 287,977 corpus positions -- 6.19%. Those positions anchor
the value head, and the anchor propagates backwards through the dynamics to everything else.

WHITE POV, to match every other label in this repo. `probe_wdl` answers for the SIDE TO MOVE,
so black-to-move results are negated before being stored.

Emits per row:
    syz_wdl   int8   2 win / 1 draw / 0 loss (white POV), -1 = not in the tablebase
    syz_dtz   int16  distance-to-zeroing, -32768 where unavailable (kept for later; the
                     geometry may want a true ply metric, and DTZ is the exact one)

    .venv/bin/python -m ...gen_syzygy_wdl --games 4000 --n-piecedown 45906
"""
from __future__ import annotations

import argparse

import numpy as np

# inverse of jepa.tokenize's _PIECE: index -> (piece_type, is_white)
_INV = {1: (1, True), 2: (2, True), 3: (3, True), 4: (4, True), 5: (5, True), 6: (6, True),
        7: (1, False), 8: (2, False), 9: (3, False), 10: (4, False), 11: (5, False),
        12: (6, False)}


def board_from_tokens(tok, glob):
    """(64,) uint8 + (6,) uint8 -> chess.Board. Inverse of jepa.tokenize.

    Castling rights are DROPPED deliberately: Syzygy positions are defined without them, and a
    stale right in the glob would make probe_wdl raise. At <=5 men a genuine castling right is
    impossible anyway (it needs both a king and a rook unmoved, plus 3+ other men)."""
    import chess
    b = chess.Board(None)
    for sq in range(64):
        t = int(tok[sq])
        if t:
            pt, white = _INV[t]
            b.set_piece_at(sq, chess.Piece(pt, white))
    b.turn = bool(glob[0])
    b.castling_rights = 0
    ep = int(glob[5])
    b.ep_square = None          # ep only matters for a probe if a capture is pending; Syzygy
    return b                    # handles it via the position itself, and a stale ep can raise


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-men", type=int, default=5)
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    import chess, chess.syzygy
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    tok, glob = tr.tok, tr.glob
    n = len(tok)
    men = (tok != 0).sum(1)
    todo = np.flatnonzero(men <= args.max_men)
    print(f"[syz] {n:,} positions | {len(todo):,} at <={args.max_men} men "
          f"({100*len(todo)/n:.2f}%)", flush=True)

    wdl = np.full(n, -1, np.int8)
    dtz = np.full(n, -32768, np.int16)
    tb = chess.syzygy.open_tablebase(str(paths.syzygy_dir()))
    ok = fail = 0
    for j, r in enumerate(todo):
        b = board_from_tokens(tok[r], glob[r])
        try:
            w = tb.probe_wdl(b)                 # +2 win .. -2 loss, SIDE TO MOVE
            z = tb.probe_dtz(b)
        except Exception:
            fail += 1
            continue
        if not b.turn:                          # store white POV
            w = -w
        wdl[r] = 2 if w > 0 else (0 if w < 0 else 1)
        dtz[r] = int(np.clip(z, -32767, 32767))
        ok += 1
        if j and j % 50000 == 0:
            print(f"[syz]   {j:,}/{len(todo):,} probed ({fail} unprobeable)", flush=True)
    tb.close()

    print(f"[syz] probed {ok:,} | unprobeable {fail:,}", flush=True)
    lab = wdl >= 0
    if lab.any():
        b3 = np.bincount(wdl[lab], minlength=3)
        print(f"[syz] EXACT white-POV outcomes: LOSS {b3[0]:,} ({100*b3[0]/lab.sum():.1f}%)  "
              f"DRAW {b3[1]:,} ({100*b3[1]/lab.sum():.1f}%)  "
              f"WIN {b3[2]:,} ({100*b3[2]/lab.sum():.1f}%)", flush=True)
        # the point of an anchor is that it DISAGREES with the noisy signal often enough to
        # be worth having -- measure that rather than assume it
        out_w = tr.outcome_of_row_white()
        final = np.where(out_w == T.WIN, 2, np.where(out_w == T.DRAW, 1, 0))
        agree = (final[lab] == wdl[lab]).mean()
        print(f"[syz] agreement with the GAME'S FINAL RESULT: {100*agree:.1f}% "
              f"-- the remaining {100*(1-agree):.1f}% is where the played line left the "
              f"theoretical value, i.e. exactly what the anchor corrects", flush=True)

    if args.save:
        p = paths.derived(f"syzygy_wdl_{n}.npz")
        np.savez_compressed(p, syz_wdl=wdl, syz_dtz=dtz, men=men.astype(np.int16))
        print(f"[syz] wrote {p}", flush=True)


if __name__ == "__main__":
    main()
