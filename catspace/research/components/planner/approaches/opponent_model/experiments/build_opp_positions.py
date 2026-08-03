#!/usr/bin/env python
"""catspace/research/components/planner/approaches/opponent_model/experiments/build_opp_positions.py -- reconstruct the OPPONENT's decision points from the M2b
dense positions parquet (REACHABILITY_FOUNDATIONS §6.6: causal ẑ_opp(t) for the reach head).

No new data needed: for consecutive target-player rows (ply gap == 2) in a game,
    opponent position  = fen_t  + our played move
    opponent's move    = the unique legal move mapping that position to fen_{t+1}
Rows where the gap > 2 (cache subsampling) or the game ends after our move are skipped -- the
causal estimator tolerates missing observations.

Output parquet has the SAME schema as positions_dense.parquet with mover-swapped fields
(elo_self <-> elo_oppo, white negated, pidx=-1 anonymous, player_id = 10^12+game_id pseudo-id),
so catspace/research/components/planner/approaches/opponent_model/experiments/m2b_cache.py runs on it UNCHANGED (Maia candidates indexed in the mover frame).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import chess
import numpy as np
import pandas as pd
from catspace.io import paths



def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--positions", default=paths.derived("m2b/positions_dense.parquet"))
    ap.add_argument("--out", default=paths.derived("m2b/positions_dense_opp.parquet"))
    args = ap.parse_args()
    t0 = time.time()

    p = pd.read_parquet(args.positions).sort_values(["game_id", "ply"]).reset_index(drop=True)
    rows, n_gap, n_end, n_nomatch = [], 0, 0, 0
    for gid, g in p.groupby("game_id", sort=False):
        g = g.reset_index(drop=True)
        for j in range(len(g)):
            if j + 1 >= len(g):
                n_end += 1
                continue
            if int(g.ply[j + 1]) - int(g.ply[j]) != 2:
                n_gap += 1
                continue
            b = chess.Board(g.fen[j])
            b.push_uci(g.played[j])                       # opponent to move now
            nxt = chess.Board(g.fen[j + 1])
            mv = None
            for m in b.legal_moves:
                b.push(m)
                if b.board_fen() == nxt.board_fen() and b.turn == nxt.turn \
                        and b.castling_rights == nxt.castling_rights:
                    mv = m
                    b.pop()
                    break
                b.pop()
            if mv is None:
                n_nomatch += 1
                continue
            rows.append(dict(
                player_id=np.uint64(10**12 + int(gid)), pidx=-1, prov=False,
                elo_self=int(g.elo_oppo[j]), elo_oppo=int(g.elo_self[j]),
                white=not bool(g.white[j]), fen=b.fen(), played=mv.uci(),
                ply=int(g.ply[j]) + 1, game_id=int(gid), split=str(g.split[j])))
    out = pd.DataFrame(rows)
    print(f"AUDIT: {len(out):,} opponent decisions from {len(p):,} target rows | "
          f"skipped gap {n_gap:,} / game-end {n_end:,} / no-match {n_nomatch:,}")
    assert len(out) > 0.5 * len(p), "reconstruction lost >50% of rows -- inspect gaps"
    assert n_nomatch < 0.01 * len(p), f"too many no-match rows ({n_nomatch}) -- replay bug?"
    per_game = out.groupby("game_id").size()
    print(f"AUDIT: games {len(per_game)} | opp-moves/game med {per_game.median():.0f} "
          f"p90 {per_game.quantile(0.9):.0f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"wrote {args.out} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
