#!/usr/bin/env python
"""catspace/research/components/planner/approaches/atlas_region_stats/experiments/gen_transition_data.py -- M2a data: training examples for the TRANSITION ESTIMATOR
T(phi(s), context) -> crossing risk. Streams raw lichess PGN (WITH per-move clocks), samples
positions where a player is to move, and records:
  phi        : the M1 reachability embedding phi(s) (catspace/field.py, frozen; T is a head over it)
  fen        : for the SF-labeling pass (objective committor before/after the ACTUAL move)
  clk_mover  : the mover's remaining clock ENTERING s (their previous move's post-clock)
  clk_opp    : opponent's remaining clock | base_s, inc_s : the time control
  elo_mover, elo_opp, ply
  move       : the ACTUAL move played (the realized decision whose crossing we label)
Single process (PGN stream is the bottleneck); phi computed batched on MPS. Output npz -> then
sf_label_transitions.py adds committor_before/after -> train_transition_estimator.py.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import chess
import numpy as np

from catspace.research.tools.chess_specific.chessdata.lichess import stream_filtered_games, GameFilter, _time_control_base_seconds
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", default=paths.raw("lichess_db_standard_rated_2019-01.pgn.zst"))
    ap.add_argument("--games", type=int, default=40000)
    ap.add_argument("--stride", type=int, default=6); ap.add_argument("--skip-open", type=int, default=8)
    ap.add_argument("--per-game", type=int, default=10); ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--phi-batch", type=int, default=4096)
    ap.add_argument("--out", default=paths.derived("transition_data.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed)
    from lczerolens import LczeroBoard
    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    import torch
    field = ReachabilityField()
    gf = GameFilter()

    boards_buf, meta = [], []                                # buffer boards; flush phi in batches
    phis = []
    def flush():
        if not boards_buf:
            return
        p = field.phi(boards_buf).to(torch.float16).cpu().numpy()
        phis.append(p); boards_buf.clear()

    fens, clkm, clko, bases, incs, elom, eloo, plies, moves, gids = ([] for _ in range(10))
    kept_games = 0
    for game in stream_filtered_games(args.pgn, gf, max_games=args.games):
        h = game.headers
        base = _time_control_base_seconds(h.get("TimeControl", "")) or 0
        inc = 0
        tc = h.get("TimeControl", "")
        if "+" in tc:
            try: inc = int(tc.split("+")[1])
            except Exception: inc = 0
        we = int(h.get("WhiteElo", 0) or 0); be = int(h.get("BlackElo", 0) or 0)
        nodes = list(game.mainline())
        ucis = [nd.move.uci() for nd in nodes]
        clocks = [nd.clock() for nd in nodes]                # remaining time AFTER move i
        n = len(ucis)
        if n < args.min_ply + 2:
            continue
        lb = LczeroBoard(); taken = 0; gid = kept_games
        for p in range(n - 1):                               # position BEFORE move p; mover plays ucis[p]
            on = p >= args.skip_open and (p - args.skip_open) % args.stride == 0 and taken < args.per_game
            if on and p >= args.min_ply and not lb.is_game_over():
                cm = clocks[p - 2] if p >= 2 and clocks[p - 2] is not None else float(base)
                co = clocks[p - 1] if p >= 1 and clocks[p - 1] is not None else float(base)
                mover_white = (p % 2 == 0)
                boards_buf.append(lb.copy())
                fens.append(lb.fen()); moves.append(ucis[p]); plies.append(p); gids.append(gid)
                clkm.append(cm); clko.append(co); bases.append(base); incs.append(inc)
                elom.append(we if mover_white else be); eloo.append(be if mover_white else we)
                taken += 1
                if len(boards_buf) >= args.phi_batch:
                    flush()
            try:
                lb.push(chess.Move.from_uci(ucis[p]))
            except Exception:
                break
        kept_games += 1
        if kept_games % 5000 == 0:
            print(f"  {kept_games} games, {len(fens)} positions [{time.time()-t0:.0f}s]", flush=True)
    flush()
    phi = np.concatenate(phis) if phis else np.zeros((0, field.head.iqe.components * field.head.iqe.k), np.float16)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, phi=phi, fen=np.array(fens, object),
                        move=np.array(moves, "U6"), ply=np.asarray(plies, np.int16),
                        clk_mover=np.asarray(clkm, np.float32), clk_opp=np.asarray(clko, np.float32),
                        base_s=np.asarray(bases, np.int32), inc_s=np.asarray(incs, np.int16),
                        elo_mover=np.asarray(elom, np.int16), elo_opp=np.asarray(eloo, np.int16),
                        game=np.asarray(gids, np.int32))
    print(f"\n=== {args.out}: {len(fens)} positions, {kept_games} games, phi{phi.shape} "
          f"[{time.time()-t0:.0f}s] ===")
    print(f"  clk_mover median {np.median(clkm):.0f}s | elo_mover median {int(np.median(elom))} | "
          f"time controls: base median {int(np.median(bases))}s")
    print("DONE gen_transition_data", flush=True)


if __name__ == "__main__":
    main()
