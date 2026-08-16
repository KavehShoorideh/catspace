#!/usr/bin/env python
"""gen_game_transitions.py -- build data/derived/game_transitions_{N}.npz (2026-08-15).

WHY THIS EXISTS. Five scripts read this cache (train_reach_vit, concept_dynamics,
concept_interactions, eval_jepa, concept_state_field) and NONE of them created it -- it was
generated ad hoc once, for one corpus, and every consumer since has silently assumed it. Any
run with a different corpus size dies on FileNotFoundError, because the filename is keyed to
the ROW COUNT. This makes it reproducible for any corpus.

CONTENTS (the existing convention, matched exactly):
    par  (M,)   int64  parent ROW index into tr.tok
    mid  (M,3)  int64  [from_square, to_square, promo 0-4] of the move played at that row

ALIGNMENT, which is the only subtle part. `T.build` REPLAYS games and SKIPS any that fail, so
the loader's game list is not 1:1 with the trajectory's games. Rows are matched by `game_id`
rather than by position, and each game's ply i lives at row `start[g] + i`. A game contributes
`length[g] - 1` transitions; the final position has no successor.

    .venv/bin/python -m ...gen_game_transitions --games 4000
"""
from __future__ import annotations

import argparse

import numpy as np


def build(tr, games_rows, verbose=True):
    """-> (par, mid) for a Trajectories `tr` and the loader rows that produced it."""
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import move_ids
    import chess

    by_id = {int(g[0]): g[2] for g in games_rows}          # game_id -> ucis
    par_l, mid_l, missing, short = [], [], 0, 0
    for gi in range(len(tr.start)):
        gid = int(tr.game_id[gi])
        ucis = by_id.get(gid)
        if ucis is None:
            missing += 1
            continue
        s, n = int(tr.start[gi]), int(tr.length[gi])
        # n rows = n plies stored; transitions are between consecutive stored rows
        m = min(n - 1, len(ucis))
        if m <= 0:
            short += 1
            continue
        for i in range(m):
            try:
                mv = chess.Move.from_uci(ucis[i])
            except Exception:
                break
            f, t, p = move_ids(mv)
            par_l.append(s + i); mid_l.append((f, t, p))
    par = np.asarray(par_l, np.int64)
    mid = np.asarray(mid_l, np.int64).reshape(-1, 3)
    if verbose:
        print(f"[trans] {len(par):,} transitions from {len(tr.start):,} games "
              f"(unmatched {missing}, too-short {short})", flush=True)
    return par, mid


def ensure(tr, games_rows, verbose=True):
    """Load the cache for this corpus, building it if absent. -> dict(par=..., mid=...)."""
    from catspace.io import paths
    p = paths.derived(f"game_transitions_{len(tr.tok)}.npz")
    try:
        z = np.load(p)
        return {"par": z["par"], "mid": z["mid"]}
    except FileNotFoundError:
        if verbose:
            print(f"[trans] no cache for {len(tr.tok):,} rows -- building {p}", flush=True)
        par, mid = build(tr, games_rows, verbose)
        np.savez_compressed(p, par=par, mid=mid)
        return {"par": par, "mid": mid}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--n-piecedown", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    rows = T.load_sf_games(args.games, args.seed, max_plies=args.max_plies)
    d = ensure(tr, rows)
    print(f"[trans] ok: par {d['par'].shape} mid {d['mid'].shape}")


if __name__ == "__main__":
    main()
