"""Harness component: the game loop vs an engine opponent + the VERDICT lines.
Identical instrumentation/output to the pre-modular m5_mcts_probe (logs stay
comparable across the refactor)."""
from __future__ import annotations

import time
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np


def run_games(rf, planner, navigator, maia, args, rng, t0=None):
    """planner: prepare()d Planner; navigator: MCTSNavigator; maia: UCI engine."""
    t0 = t0 or time.time()
    W = D = L = 0; pgns = []; all_spells = []
    for g in range(args.games):
        from lczerolens import LczeroBoard
        board = LczeroBoard(); our_white = (g % 2 == 0)
        planner.new_game()
        for _ in range(args.opening_plies):
            ms = list(board.legal_moves)
            if not ms:
                break
            board.push(ms[rng.integers(0, len(ms))])
        ply = board.ply()
        while not board.is_game_over(claim_draw=True) and ply < args.max_plies:
            we_move = board.turn == (chess.WHITE if our_white else chess.BLACK)
            if we_move:
                phi_now = rf.phi([board]).cpu().numpy()[0]
                tid = planner.plan(phi_now, ply)
                mv = navigator.move(board, tid)
            else:
                mv = maia.play(board, chess.engine.Limit(nodes=1)).move
            if mv is None:
                break
            board.push(mv); ply += 1
        res = board.result(claim_draw=True)
        s = 0.5 if res == "1/2-1/2" else (1.0 if (res == "1-0") == our_white else 0.0)
        W += s == 1.0; D += s == 0.5; L += s == 0.0
        planner.finish_game(ply)
        all_spells.extend(planner.spells)
        gp = chess.pgn.Game.from_board(board)
        gp.headers["White"] = ("catspace-m5" if our_white else f"maia-{args.maia_elo}")
        gp.headers["Black"] = (f"maia-{args.maia_elo}" if our_white else "catspace-m5")
        gp.headers["Result"] = res
        pgns.append(str(gp))
        hits = sum(sp["outcome"] == "hit" for sp in planner.spells)
        print(f"  game {g+1}/{args.games} -> {res} (us {s}) | spells {len(planner.spells)} "
              f"hit {hits} | {time.time()-t0:.0f}s", flush=True)
    Path(args.save_pgn).write_text("\n\n".join(pgns))

    n = args.games
    print(f"VERDICT M5 score: {(W + 0.5 * D) / n:.3f} (W{W} D{D} L{L} of {n}) "
          f"vs maia-{args.maia_elo} [nodes={args.nodes}] "
          f"(shallow committor baseline context: 0.125)")
    if all_spells:
        hits = [sp for sp in all_spells if sp["outcome"] == "hit"]
        sw = sum(sp["outcome"] == "switch" for sp in all_spells)
        pp = np.array([sp["pred_plies"] for sp in hits], float)
        aa = np.array([sp["actual"] for sp in hits], float)
        cal = (f"plies pred median {np.median(pp):.1f} vs realized {np.median(aa):.1f}"
               if hits else "no hits")
        print(f"VERDICT M5 plans: {len(all_spells)} spells | hit {len(hits)} "
              f"({len(hits)/len(all_spells):.0%}) switch {sw} "
              f"({sw/len(all_spells):.0%}) | chain depth median "
              f"{np.median([sp['chain'] for sp in all_spells]):.0f} | {cal}")
        sws = [sp for sp in all_spells if sp["outcome"] == "switch"]
        if sws:
            mg = np.array([sp["margin"] for sp in sws])
            dc = np.array([sp["decay"] for sp in sws])
            print(f"VERDICT M5 switches: margin (challenger-incumbent, nats) median "
                  f"{np.median(mg):.3f} p90 {np.percentile(mg, 90):.3f} | incumbent decay "
                  f"since adoption median {np.median(dc):.3f} "
                  f"({np.mean(dc > np.median(mg)):.0%} decay-dominated = honest abandonment)")
    ev = np.array(navigator.evals, float)
    print(f"VERDICT M5 budget: {ev.mean():.0f} fresh evals/our-move "
          f"(median {np.median(ev):.0f}, n={len(ev)}) | cache {len(navigator.cache)} entries")
    return dict(score=(W + 0.5 * D) / n, W=W, D=D, L=L)
