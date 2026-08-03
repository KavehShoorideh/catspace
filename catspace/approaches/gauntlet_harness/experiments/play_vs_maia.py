#!/usr/bin/env python
"""catspace/approaches/gauntlet_harness/experiments/play_vs_maia.py -- put the trained full-board field into PLAY against MAIA (human-like
lc0 nets) and measure how it does (Kaveh: "how does the engine play against maia?").

The field player is COMMITTOR-GREEDY on field_fullgame_v3: for each legal move it evaluates the
committor c(s')=P(white win) of the resulting position and picks argmax if White / argmin if Black
(maximise MY win probability). This is the pure 1-ply VALUE policy on the field -- no search, no
opponent model yet (that is the z/T exploitation layer, next). It answers: does the committor, used
greedily, play sensible chess against a human-like opponent, and where does it break.

Maia = lc0 + maia-<elo>.pb.gz at nodes=1 (pure policy head = human-like move). Alternating colors,
optional random opening plies for diversity, W/D/L + score with a baseline, PGN saved.
"""
from __future__ import annotations

import argparse, sys, time
from pathlib import Path

import chess, chess.engine, chess.pgn
import numpy as np
import torch

from catspace.research.components.encoder.approaches.reachability_field.experiments.train_clock_field import ClockField
from catspace.research.tools.training_infra.train.scaffold import resolve_device


from catspace.research.components.planner.approaches.committor_value.src.committor import CommittorGreedy  # component home (refactor 2026-07-30)
from catspace.io import paths


def play_game(field, maia, field_is_white, opening_plies, max_plies, rng, maia_nodes, depth):
    from lczerolens import LczeroBoard
    board = LczeroBoard()
    # random opening plies (both sides) for diversity
    for _ in range(opening_plies):
        ms = list(board.legal_moves)
        if not ms: break
        board.push(ms[rng.integers(0, len(ms))])
    ply = board.ply()
    while not board.is_game_over(claim_draw=True) and ply < max_plies:
        if board.turn == (chess.WHITE if field_is_white else chess.BLACK):
            mv, _ = field.select(board, rng, depth=depth)
        else:
            mv = maia.play(board, chess.engine.Limit(nodes=maia_nodes)).move
        if mv is None: break
        board.push(mv); ply += 1
    res = board.result(claim_draw=True)
    return board, res


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.experiment("field_fullgame_v3_final.pt"))
    ap.add_argument("--maia-elo", type=int, default=1500)
    ap.add_argument("--maia-nodes", type=int, default=1)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--tau", type=float, default=0.0)
    ap.add_argument("--depth", type=int, default=1, help="1=committor-greedy, 2=2-ply search")
    ap.add_argument("--opp-tau", type=float, default=0.0, help="opponent fallibility temp (0=minimax, >0=expectimax)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-pgn", default=paths.experiment("field_v3_vs_maia.pgn"))
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = resolve_device(args.device); rng = np.random.default_rng(args.seed); t0 = time.time()
    field = CommittorGreedy(args.ckpt, dev, tau=args.tau, opp_tau=args.opp_tau)
    wpath = paths.engine(f"maia/maia-{args.maia_elo}.pb.gz")
    maia = chess.engine.SimpleEngine.popen_uci(["lc0", f"--weights={wpath}", "--backend=eigen"])
    print(f"[play] field_v3 (committor-greedy) vs maia-{args.maia_elo} (nodes={args.maia_nodes}) "
          f"| {args.games} games alternating colors depth={args.depth}", flush=True)

    W = D = L = 0; games_pgn = []
    for g in range(args.games):
        field_white = (g % 2 == 0)
        board, res = play_game(field, maia, field_white, args.opening_plies, args.max_plies, rng, args.maia_nodes, args.depth)
        # score from the FIELD's POV
        if res == "1/2-1/2": D += 1; s = 0.5
        elif (res == "1-0") == field_white: W += 1; s = 1.0
        else: L += 1; s = 0.0
        gp = chess.pgn.Game.from_board(board)
        gp.headers["White"] = "field_v3" if field_white else f"maia-{args.maia_elo}"
        gp.headers["Black"] = f"maia-{args.maia_elo}" if field_white else "field_v3"
        gp.headers["Result"] = res
        games_pgn.append(str(gp))
        print(f"  game {g+1}/{args.games} field={'W' if field_white else 'B'} -> {res} "
              f"(field {s}) | running W{W} D{D} L{L} [{time.time()-t0:.0f}s]", flush=True)
    maia.quit()
    n = W + D + L; score = (W + 0.5 * D) / n
    Path(args.save_pgn).parent.mkdir(parents=True, exist_ok=True)
    Path(args.save_pgn).write_text("\n\n".join(games_pgn))
    print(f"\nVERDICT field_v3 (committor-greedy, 1-ply) vs maia-{args.maia_elo}: "
          f"{W}W {D}D {L}L in {n} | SCORE {score:.3f} (0.5=even) | [{time.time()-t0:.0f}s]", flush=True)
    print(f"  PGN -> {args.save_pgn}")


if __name__ == "__main__":
    main()
