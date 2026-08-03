#!/usr/bin/env python
"""experiments/tactic_events.py -- E1 (INQUIRY_TACTICS.md): EXACT tactic events on toy
trajectories. A tactic event = a VETO LAPSE: a good target region that was DENIED from the
position before the defender's move becomes FORCEABLE after it. Defender = tb-optimal with
epsilon-random lapses injected (so events exist and are attributable); attacker = Stockfish
(clean lines). Per defender ply: sample candidate won-target regions from short random walks,
test region-forceability before vs after (forceable() DFS, Black tb-fixed). Reports events
per game, and the attribution table: P(flip | lapse move) vs P(flip | optimal move) --
Kaveh's definition ("an opportunity afforded by a mistake") measured exactly.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, tb_best_move
from experiments.measure_adversarial_veto import forceable, neighborhood_of, wdl_white


def candidate_regions(board, tb, rng, walks=120, j=4, cap=12):
    """Sample distinct won-target regions reachable within j plies (random walks)."""
    seen = {}
    for _ in range(walks):
        b = board.copy(stack=False)
        ok = True
        for _t in range(j):
            mv = list(b.legal_moves)
            if not mv:
                ok = False; break
            b.push(mv[int(rng.integers(len(mv)))])
        if not ok or b.is_game_over(claim_draw=True):
            continue
        if wdl_white(b, tb) == 2:
            seen.setdefault(b._transposition_key(), b.copy(stack=False))
    targets = list(seen.values())
    rng.shuffle(targets)
    return targets[:cap]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--n-games", type=int, default=10)
    ap.add_argument("--eps-black", type=float, default=0.25, help="defender lapse rate")
    ap.add_argument("--j", type=int, default=4)
    ap.add_argument("--targets", type=int, default=12)
    ap.add_argument("--max-plies", type=int, default=40)
    ap.add_argument("--sf-depth", type=int, default=12)
    ap.add_argument("--engine", default="stockfish")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed); tb = TB()
    eng = chess.engine.SimpleEngine.popen_uci(args.engine); eng.configure({"Threads": 1})

    dz = np.load(args.dtm_npz)
    P, M, dtm = np.asarray(dz["packed"]), np.asarray(dz["meta"]), np.asarray(dz["dtm"])
    cand = np.flatnonzero((dtm >= 12) & (dtm <= 30)); rng.shuffle(cand)

    flips_lapse = trials_lapse = flips_opt = trials_opt = 0
    events_total = 0; games = 0
    for ci in cand:
        if games >= args.n_games:
            break
        b = board_from_packed(P[ci], M[ci])
        if b.turn != chess.WHITE or b.is_game_over():
            continue
        games += 1
        for _ply in range(args.max_plies):
            if b.is_game_over(claim_draw=True):
                break
            if b.turn == chess.WHITE:
                info = eng.analyse(b, chess.engine.Limit(depth=args.sf_depth))
                if not info.get("pv"):
                    break
                b.push(info["pv"][0])
                continue
            # DEFENDER ply: measure region forceability BEFORE and AFTER the move
            targets = candidate_regions(b, tb, rng, j=args.j, cap=args.targets)
            pre = {g._transposition_key(): forceable(b, neighborhood_of(g), tb, args.j)
                   for g in targets}
            opt = tb_best_move(b, tb)
            lapse = rng.random() < args.eps_black
            mv = (list(b.legal_moves)[int(rng.integers(b.legal_moves.count()))] if lapse else opt)
            b.push(mv)
            if b.is_game_over(claim_draw=True):
                break
            flipped = 0
            for g in targets:
                k = g._transposition_key()
                if not pre[k] and forceable(b, neighborhood_of(g), tb, args.j):
                    flipped += 1                      # denied -> forceable = VETO LAPSE event
            events_total += flipped
            if lapse:
                trials_lapse += len(targets); flips_lapse += flipped
            else:
                trials_opt += len(targets); flips_opt += flipped
        print(f"  game {games}/{args.n_games} done  [{time.time()-t0:.0f}s]", flush=True)

    eng.quit(); tb.close()
    r_l = flips_lapse / max(trials_lapse, 1); r_o = flips_opt / max(trials_opt, 1)
    print(f"VERDICT TACTIC_EVENTS games={games} eps_black={args.eps_black}  "
          f"flip-rate after LAPSE {r_l:.3f} ({flips_lapse}/{trials_lapse}) vs after OPTIMAL "
          f"{r_o:.3f} ({flips_opt}/{trials_opt})  ratio {r_l/max(r_o,1e-9):.1f}x  "
          f"events_total={events_total}  [{time.time()-t0:.0f}s]", flush=True)
    print("  reading: ratio >> 1 = veto lapses (mistakes) CAUSE forceability flips -- the exact, "
          "measured form of 'a tactic is an opportunity afforded by a mistake'.", flush=True)


if __name__ == "__main__":
    main()
