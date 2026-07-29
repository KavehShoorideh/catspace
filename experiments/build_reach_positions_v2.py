#!/usr/bin/env python
"""experiments/build_reach_positions_v2.py -- SCALED positions for the reach-field power test
(JOURNAL 2026-07-28 decision: both-sequenced; field-z re-test trigger).

From data/records/player_games_rapid, take games of Z-TABLE players (the 2975 M2b styles),
replay, and emit BOTH SIDES' decision points (mover-frame rows, m2b_cache schema) -- unlike the
v1 dense cache (one side, ~21 rows/game, 3.7k games), this gives every position of every sampled
game. Adds `result_mover` (game outcome from the mover's POV) for the competing-risks WDL head.

Split discipline: by PLAYER (target-player id): ~10% of z-players held out entirely ('heldout'
= unseen-player split), plus game_id%10 carves eval games inside 'train' at train time (as v1).
Pooled replay (process pool over record shards); positions capped per game at --max-ply.
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def replay_chunk(task):
    rows_in, base_gid, max_ply = task
    import chess
    out = []
    for k, (pid, pelo, oelo, pwhite, result, moves, pidx) in enumerate(rows_in):
        gid = base_gid + k
        b = chess.Board()
        mv = moves.split()
        for ply, u in enumerate(mv):
            if ply >= max_ply:
                break
            mover_white = (ply % 2 == 0)
            mover_is_target = (mover_white == pwhite)
            fen = b.fen()
            try:
                m = chess.Move.from_uci(u)
                if not b.is_legal(m):
                    break
                b.push(m)
            except Exception:
                break
            r_mover = result if mover_is_target else -result
            out.append((
                np.uint64(pid), pidx if mover_is_target else -1, False,
                int(pelo if mover_is_target else oelo),
                int(oelo if mover_is_target else pelo),
                mover_white, fen, u, ply, gid, "", (r_mover + 1) / 2.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default="data/records/player_games_rapid")
    ap.add_argument("--zcache", default="data/derived/m2b/cache_3k.npz",
                    help="source of the player_id -> pidx mapping (the z-table membership)")
    ap.add_argument("--out", default="data/derived/m2b/positions_v2.parquet")
    ap.add_argument("--games", type=int, default=60000)
    ap.add_argument("--max-ply", type=int, default=120)
    ap.add_argument("--heldout-frac", type=float, default=0.10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)

    zc = dict(np.load(args.zcache, allow_pickle=True))
    zmap = {}
    for pid, px in zip(zc["player_id"], zc["pidx"]):
        if px >= 0:
            zmap[np.uint64(pid)] = int(px)
    print(f"z-table players: {len(zmap)}")

    shards = sorted(glob.glob(f"{args.records}/players_*.parquet"))
    frames = []
    for s in shards:
        p = pd.read_parquet(s, columns=["player_id", "player_elo", "opp_elo", "player_white",
                                        "result", "moves", "provisional"])
        p = p[~p.provisional]
        p = p[p.player_id.astype(np.uint64).isin(zmap.keys())]
        frames.append(p)
    g = pd.concat(frames, ignore_index=True)
    print(f"candidate games of z-players: {len(g):,}")
    if len(g) > args.games:
        g = g.sample(n=args.games, random_state=args.seed).reset_index(drop=True)

    # player-level heldout split
    players = g.player_id.unique()
    rng.shuffle(players)
    ho = set(players[: int(len(players) * args.heldout_frac)].tolist())
    print(f"players sampled: {len(players)} | heldout players: {len(ho)}")

    tasks, chunk, base = [], [], 0
    for _, r in g.iterrows():
        chunk.append((int(np.uint64(r.player_id)), r.player_elo, r.opp_elo, bool(r.player_white),
                      int(r.result), r.moves, zmap[np.uint64(r.player_id)]))
        if len(chunk) == 500:
            tasks.append((chunk, base, args.max_ply)); base += len(chunk); chunk = []
    if chunk:
        tasks.append((chunk, base, args.max_ply))

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, res in enumerate(ex.map(replay_chunk, tasks)):
            rows.extend(res)
            if (i + 1) % 20 == 0:
                print(f"  {(i+1)*500:,} games replayed, {len(rows):,} rows, "
                      f"{time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows, columns=["player_id", "pidx", "prov", "elo_self", "elo_oppo",
                                     "white", "fen", "played", "ply", "game_id", "split",
                                     "result_mover"])
    df["split"] = np.where(df.player_id.isin(list(ho)), "heldout", "train")
    # AUDITS (TESTING §2.14)
    print(f"AUDIT rows {len(df):,} | games {df.game_id.nunique():,} | "
          f"rows/game med {df.groupby('game_id').size().median():.0f}")
    for s in ("train", "heldout"):
        m = df[df.split == s]
        print(f"AUDIT [{s}]: rows {len(m):,} | players {m.player_id.nunique()} | "
              f"target-rows {(m.pidx>=0).mean():.3f} | elo med {m.elo_self.median():.0f} | "
              f"result_mover mean {m.result_mover.mean():.3f}")
    assert df.pidx.max() < 2975 and (df.pidx >= -1).all()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"wrote {args.out} ({Path(args.out).stat().st_size/1e6:.0f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
