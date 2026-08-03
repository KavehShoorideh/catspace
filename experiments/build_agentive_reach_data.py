#!/usr/bin/env python
"""experiments/build_agentive_reach_data.py -- AGENTIVE first-hit labels from OUR OWN
steered games (Kaveh 2026-07-30, option a: close the descriptive-vs-agentive gap).

The v3 reach field answers "where do games between passive humans drift?"; navigation
asks "where does the game go when WE steer?". Six M5 reads showed plans realized in
2-4 plies against predictions of 7-11 -- the field is off-policy for its own navigator.
Fix: materialize first-hit labels from the M5 game corpus (catspace-m5 vs maia-1100)
and FINE-TUNE the reach head on them (train_reach_head.py --init).

Train==play discipline:
  * goals   = the 1024 TABLE regions the navigator queries (not the field's 256 bank);
  * hit rule = the navigator's own: phi-space nearest-centroid argmin (no eps ball);
  * rows    = OUR-move positions only, at the play-time context (elo 1800 vs 1100,
              z_self = 0, z_opp = 0 cold-start -> zopp companion file of zeros).

Output: --out npz in build_reach_data format (+ --out-zopp aligned zeros file).
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import chess.pgn
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgns", nargs="+",
                    default=["artifacts/experiments/m5_read100*.pgn"],
                    help="globs of OUR game PGNs (White or Black = catspace-m5)")
    ap.add_argument("--table", default="data/derived/reach/region_table_v4.npz",
                    help="region table whose 'regions' bank defines the goal set")
    ap.add_argument("--our-elo", type=float, default=1800.0)
    ap.add_argument("--opp-elo", type=float, default=1100.0)
    ap.add_argument("--out", default="data/derived/reach/agentive_v1.npz")
    ap.add_argument("--out-zopp", default="data/derived/reach/agentive_v1_zopp.npz")
    args = ap.parse_args()
    t0 = time.time()

    files = sorted({f for g in args.pgns for f in glob.glob(g)})
    print(f"[agentive] {len(files)} pgn files: {[Path(f).name for f in files]}")
    t = np.load(args.table, allow_pickle=True)
    bank = t["regions"].astype(np.float32)                     # (G,64) phi centroids
    G = len(bank)
    rf = ReachabilityField()

    rows_phi, rows_gid, rows_ply, rows_hit, rows_plies, rows_innow = [], [], [], [], [], []
    gid = 0
    from lczerolens import LczeroBoard
    for path in files:
        with open(path) as fh:
            while True:
                game = chess.pgn.read_game(fh)
                if game is None:
                    break
                we_white = game.headers.get("White", "") == "catspace-m5"
                we_black = game.headers.get("Black", "") == "catspace-m5"
                if not (we_white or we_black):
                    continue
                boards, our_move_idx = [], []
                b = LczeroBoard()
                for i, mv in enumerate(game.mainline_moves()):
                    if (b.turn == chess.WHITE) == we_white:
                        our_move_idx.append(len(boards))
                    boards.append(b.copy(stack=False))
                    b.push(mv)
                if len(boards) < 6 or not our_move_idx:
                    continue
                phis = np.concatenate([rf.phi(boards[i:i+512]).cpu().numpy()
                                       for i in range(0, len(boards), 512)])
                d2 = (phis*phis).sum(1)[:, None] + (bank*bank).sum(1)[None, :] \
                    - 2.0*phis@bank.T
                reg = d2.argmin(1)                              # navigator's hit rule
                T = len(reg)
                # first future index where each region occurs (backward sweep)
                first = np.full(G, -1, np.int64)
                firsts = np.zeros((T, G), np.int64)
                for i in range(T - 1, -1, -1):
                    firsts[i] = first
                    first[reg[i]] = i
                for i in our_move_idx:
                    fut = firsts[i]                             # first strictly-later hit
                    hit = (fut >= 0).astype(np.uint8)
                    pl = np.where(fut >= 0, fut - i, 0).astype(np.int16)
                    inn = np.zeros(G, np.uint8); inn[reg[i]] = 1
                    rows_phi.append(phis[i]); rows_gid.append(gid); rows_ply.append(i)
                    rows_hit.append(hit); rows_plies.append(pl); rows_innow.append(inn)
                gid += 1
    N = len(rows_phi)
    print(f"[agentive] {gid} games -> {N} our-move rows x {G} goals [{time.time()-t0:.0f}s]")
    hit = np.stack(rows_hit); plies = np.stack(rows_plies); in_now = np.stack(rows_innow)
    ok = in_now == 0
    print(f"AUDIT: hit rate (excl in_now) {hit[ok].mean():.4f} | "
          f"plies med {np.median(plies[hit.astype(bool) & ok]):.0f} | "
          f"per-goal hit min/med/max {hit.mean(0).min():.4f}/"
          f"{np.median(hit.mean(0)):.4f}/{hit.mean(0).max():.4f}")
    gida = np.asarray(rows_gid, np.int64)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, hit=hit, plies=plies, in_now=in_now, bank=bank, eps=np.float32(-1.0),
        phi=np.stack(rows_phi).astype(np.float32),
        pidx=np.full(N, -1, np.int32),
        elo_self=np.full(N, args.our_elo, np.float32),
        elo_oppo=np.full(N, args.opp_elo, np.float32),
        split=np.array(["train"] * N),
        game_id=gida, ply=np.asarray(rows_ply, np.int32),
        player_id=np.zeros(N, np.uint64),
        meta_cache="agentive:m5-selfgames", meta_goals=G, meta_seed=0,
        meta_eps_quantile=-1.0)
    np.savez_compressed(args.out_zopp, z_opp_t=np.zeros((N, 16), np.float32),
                        n_obs=np.zeros(N, np.int32), game_id=gida,
                        ply=np.asarray(rows_ply, np.int32))
    print(f"wrote {out} + {args.out_zopp} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
