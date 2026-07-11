#!/usr/bin/env python
"""
experiments/arena_real.py — real-board arena: FBBoardPolicy (the trained cone,
greedy MIN readout) against random or Stockfish opponents, with alternating
colors, per-game seeded random opening plies (start diversification), and an
anytime-valid e-value verdict on decisive games (abtest.EValueTest).

FB plays with zMATE_W as white and zMATE_B as black (same policy, matching
goal). --save-pgn writes the games for the full-board viewer.

Honest framing: this field is imitation-bootstrapped from human games and
read out greedily with no search -- vs Stockfish (floor Elo 1320) losing is
the EXPECTED baseline; the roadmap's PI-refinement loop is what should move
it. vs random it should win decisively or something is wrong.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from catspace.abtest import EValueTest
from catspace.io.paths import derived_dir, generated_dir
from catspace.realboard import RandomBoardPolicy, play_board_game, record_to_pgn
from catspace.uci import UCIBoardPolicy


def make_opponent(spec: str):
    """'random' | 'sf:<elo>' | 'sf:skill=<k>' -> (policy or context manager, name)."""
    if spec == "random":
        return RandomBoardPolicy(), "random"
    if spec.startswith("sf:"):
        arg = spec[3:]
        if arg.startswith("skill="):
            return UCIBoardPolicy(skill=int(arg[6:]), movetime=0.02), spec
        return UCIBoardPolicy(elo=int(arg), movetime=0.05), spec
    raise SystemExit(f"unknown opponent spec {spec!r} (use random | sf:<elo> | sf:skill=<k>)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opponent", default="random")
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--depth", type=int, default=2, choices=(1, 2))
    ap.add_argument("--opening-plies", type=int, default=6)
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--elo-cond", type=int, default=1800, help="omega Elo bin FB assumes")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-pgn", default=None, help="write games as PGN (for the viewer)")
    args = ap.parse_args()

    import torch  # noqa: F401  (fail early with a clear message if .[nn] absent)
    from catspace.nn.fb import load_ckpt, pick_device
    from catspace.nn.policy_fb import FBBoardPolicy

    device = pick_device(args.device)
    fb, payload = load_ckpt(Path(args.ckpt) if args.ckpt else derived_dir() / "lichess_fb.pt", device)
    if "MATE_W" not in payload.get("zgoals", {}):
        raise SystemExit("checkpoint has no zgoals -- finish a train_lichess_fb.py run first")
    fb_white = FBBoardPolicy(fb, payload["zgoals"]["MATE_W"], depth=args.depth,
                             elo=args.elo_cond, device=device)
    fb_black = FBBoardPolicy(fb, payload["zgoals"]["MATE_B"], depth=args.depth,
                             elo=args.elo_cond, device=device)

    opponent, opp_name = make_opponent(args.opponent)
    print(f"FB(depth={args.depth}, elo_cond={args.elo_cond}) vs {opp_name}, "
          f"{args.games} games, opening_plies={args.opening_plies}, device={device}")

    records, fb_score = [], []
    test = EValueTest()

    def run_games():
        for i in range(args.games):
            rng = np.random.default_rng([args.seed, i])
            fb_is_white = i % 2 == 0
            white, black = (fb_white, opponent) if fb_is_white else (opponent, fb_black)
            rec = play_board_game(white, black, opening_plies=args.opening_plies,
                                  max_plies=args.max_plies, rng=rng)
            records.append((rec, fb_is_white))
            s = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5, "*": 0.5}[rec.result]
            s = s if fb_is_white else 1.0 - s
            fb_score.append(s)
            e = test.update(s - 0.5)                     # sign test on decisive games
            print(f"  game {i:03d}  FB as {'W' if fb_is_white else 'B'}  {rec.result:>7} "
                  f" plies={rec.n_plies:3d}  ({rec.termination})  e={e:.2f}", flush=True)

    if isinstance(opponent, UCIBoardPolicy):
        with opponent:
            run_games()
    else:
        run_games()

    score = np.array(fb_score)
    w, d, l = int((score == 1).sum()), int((score == 0.5).sum()), int((score == 0).sum())
    print(f"\nVERDICT FB vs {opp_name}: +{w} ={d} -{l}  score {score.mean():.3f}  "
          f"e={test.e:.2f} {'REJECT-H0(FB<=opp)' if test.reject_at(args.alpha) else 'no rejection'}")

    if args.save_pgn:
        path = Path(args.save_pgn) if args.save_pgn != "auto" else generated_dir() / "arena_real.pgn"
        with open(path, "w") as fh:
            for rec, fb_is_white in records:
                names = ("latentFB", opp_name) if fb_is_white else (opp_name, "latentFB")
                print(record_to_pgn(rec, *names), file=fh, end="\n\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
