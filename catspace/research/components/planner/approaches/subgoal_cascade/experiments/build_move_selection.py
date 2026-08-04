#!/usr/bin/env python
"""catspace/research/components/planner/approaches/subgoal_cascade/experiments/build_move_selection.py -- materialize (position, legal-move set, move PLAYED,
mover cohort) training rows for the opponent model, from the lichess shards. The played move
is recovered by matching each legal move's pushed position against the game's next stored row.
Positions with > --max-moves legal moves are dropped (rare). Output npz feeds
catspace/research/components/planner/approaches/opponent_model/experiments/train_opponent_model.py.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import chess
import numpy as np


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import elo_bin
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shards", default=paths.shards("lichess_db_standard_rated_2019-01.prefix1gb"))
    ap.add_argument("--n-rows", type=int, default=300_000)
    ap.add_argument("--max-moves", type=int, default=80)
    ap.add_argument("--out", default=paths.derived("move_selection_v1.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    L = args.max_moves

    PK, MT = [], []
    F, T, PC, CT = [], [], [], []            # per-row (L,) move descriptor arrays
    NM, PL, CO = [], [], []                  # n_moves, played_idx, cohort
    made = skipped = 0
    files = sorted(glob.glob(str(Path(args.shards) / "shard_*.npz")))
    rng.shuffle(files)
    for path in files:
        if made >= args.n_rows:
            break
        z = np.load(path)
        gid, pk, mt = z["game_id"], z["packed"], z["meta"]
        we, be = z["white_elo"], z["black_elo"]
        n = len(gid)
        order = rng.permutation(n - 1)
        for i in order:
            if made >= args.n_rows:
                break
            if gid[i] != gid[i + 1]:
                continue
            b = board_from_packed(pk[i], mt[i])
            moves = list(b.legal_moves)
            if not 1 <= len(moves) <= L:
                skipped += 1
                continue
            nxt = pk[i + 1]
            played = -1
            f_ = np.zeros(L, np.uint8); t_ = np.zeros(L, np.uint8)
            p_ = np.zeros(L, np.uint8); c_ = np.zeros(L, np.uint8)
            for j, m in enumerate(moves):
                f_[j], t_[j] = m.from_square, m.to_square
                p_[j] = b.piece_type_at(m.from_square) or 0
                cap = b.piece_type_at(m.to_square)
                c_[j] = cap or (1 if b.is_en_passant(m) else 0)
                if played < 0:
                    cb = b.copy(stack=False); cb.push(m)
                    if np.array_equal(encode_packed(cb), nxt):
                        played = j
            if played < 0:                    # next row not a child (game boundary quirk)
                skipped += 1
                continue
            elo = we[i] if b.turn == chess.WHITE else be[i]
            PK.append(pk[i]); MT.append(mt[i])
            F.append(f_); T.append(t_); PC.append(p_); CT.append(c_)
            NM.append(len(moves)); PL.append(played); CO.append(int(elo_bin(np.array([elo]))[0]))
            made += 1
            if made % 20_000 == 0:
                print(f"  {made}/{args.n_rows} rows ({skipped} skipped)  [{time.time()-t0:.0f}s]", flush=True)

    np.savez_compressed(args.out, packed=np.stack(PK), meta=np.stack(MT),
                        mv_from=np.stack(F), mv_to=np.stack(T), mv_piece=np.stack(PC),
                        mv_capt=np.stack(CT), n_moves=np.array(NM, np.uint8),
                        played=np.array(PL, np.uint8), cohort=np.array(CO, np.uint8))
    print(f"VERDICT MOVE_SELECTION rows={made} skipped={skipped} max_moves={L} -> {args.out}  "
          f"[{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
