#!/usr/bin/env python
"""eval_external.py -- catspace vs the OUTSIDE WORLD (Kaveh 2026-08-11: "evaluate against
stockfish... stockfish down material, so we can see: can we convert a winning position").

Two modes:
  --mode convert   engine gets WINNING starts (opponent a piece down, full-strength SF
                   defense): measures CONVERSION -- reaching the <=5-piece tablebase win zone
                   or delivering mate. The base case of playing well when ahead.
  --mode ladder    even starts vs weakened SF (skill levels / node caps): an absolute
                   strength anchor instead of self-relative arenas.

Engine plays with its real play config: cascade + time-capped iterative deepening.

    .venv/bin/python -m ...eval_external --ckpt <field.pt> --mode convert [--games 20]
"""
from __future__ import annotations

import argparse
import random
import time

import chess
import chess.engine

from catspace.io import paths
from catspace.research.components.planner.approaches.quasimetric_nav.kittychess import KittyChess


def engine_move(eng, b, budget=1.5, max_depth=6, wave=False, coherent=True, mc=False):
    if coherent or wave:
        rows = eng.search_coherent(b, budget=budget) if coherent \
            else eng.search_wave(b, budget=budget)
        if rows and mc:
            rows = eng.mc_tiebreak(b, rows)
        return rows[0]["mv"] if rows else None
    deadline = time.time() + budget
    best = None
    for d in range(1, max_depth + 1):
        try:
            rows = eng.search(b, depth=d, stop=lambda: time.time() > deadline)
        except Exception:
            break
        if time.time() > deadline:
            if best is None:
                best = rows
            break
        best = rows
    return best[0]["mv"] if best else None


def winning_starts(n, seed):
    """piece-down starts where the STRONG side is ours to play."""
    fens = []
    fp = paths.derived("piecedown_sfsf_all_v2.tsv")
    rng = random.Random(seed)
    lines = [l.split("\t") for l in open(fp) if l.count("\t") >= 3]
    rng.shuffle(lines)
    for parts in lines:
        res, fen = int(parts[1]), parts[2]
        if res == 0:
            continue
        b = chess.Board(fen)
        strong_white = res == 1                      # the side that won the source game
        fens.append((fen, strong_white))
        if len(fens) >= n:
            break
    return fens


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mode", choices=("convert", "ladder"), default="convert")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--budget", type=float, default=1.5)
    ap.add_argument("--sf-nodes", type=int, default=20000,
                    help="defender/ladder SF nodes per move")
    ap.add_argument("--skill", type=int, default=None,
                    help="ladder mode: SF Skill Level 0-20 (None = node cap only)")
    ap.add_argument("--wave", action="store_true", help="use the selective wave search")
    ap.add_argument("--coherent", action="store_true", default=True,
                    help="use the coherence-bounded search (sanity 10/10 at 1.48s)")
    ap.add_argument("--mc", action="store_true",
                    help="CPU rollout tiebreak among near-tied top moves (measured 82 vs 74.7 "
                         "pct keeps-the-win)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    eng = KittyChess(args.ckpt, args.device)
    sf = chess.engine.SimpleEngine.popen_uci("stockfish")
    conf = {"Threads": 1, "Hash": 128, "SyzygyPath": str(paths.syzygy_dir())}
    if args.skill is not None:
        conf["Skill Level"] = args.skill
    sf.configure(conf)

    rng = random.Random(args.seed)
    if args.mode == "convert":
        starts = winning_starts(args.games, args.seed)
    else:
        starts = []
        for _ in range(args.games):
            b = chess.Board()
            for _ in range(6):
                b.push(rng.choice(list(b.legal_moves)))
            starts.append((b.fen(), bool(rng.getrandbits(1))))

    score = conv = 0.0
    for gi, (fen, we_white) in enumerate(starts):
        b = chess.Board(fen)
        reached_tb_won = False
        while not b.is_game_over(claim_draw=True) and b.ply() < 350:
            ours = (b.turn == chess.WHITE) == we_white
            if ours:
                mv = engine_move(eng, b, budget=args.budget, wave=args.wave, coherent=args.coherent, mc=args.mc)
            else:
                mv = sf.play(b, chess.engine.Limit(nodes=args.sf_nodes)).move
            if mv is None:
                break
            b.push(mv)
            if args.mode == "convert" and not reached_tb_won \
                    and len(b.piece_map()) <= 5 and eng.tb is not None:
                try:
                    w, _ = eng.tb.wdl_dtz(b)
                    if w is not None:
                        mover_is_us = (b.turn == chess.WHITE) == we_white
                        if (w > 0) == mover_is_us and w != 0:
                            reached_tb_won = True
                except Exception:
                    pass
        out = b.outcome(claim_draw=True)
        r = 0.5 if out is None or out.winner is None else \
            (1.0 if (out.winner == chess.WHITE) == we_white else 0.0)
        score += r
        conv += float(reached_tb_won or r == 1.0)
        print(f"[ext] game {gi+1}/{len(starts)}: "
              f"{'won' if r == 1 else ('draw' if r == 0.5 else 'LOST')}"
              f"{' (tb-won zone reached)' if reached_tb_won else ''}  "
              f"running score {score}/{gi+1}", flush=True)
    sf.quit()
    n = len(starts)
    print(f"\n[ext] MODE {args.mode}  vs SF nodes={args.sf_nodes}"
          f"{' skill='+str(args.skill) if args.skill is not None else ''}")
    print(f"[ext] score {score}/{n} ({score/n:.0%})")
    if args.mode == "convert":
        print(f"[ext] conversion (won OR reached a tablebase-won <=5-piece position): "
              f"{conv/n:.0%}")


if __name__ == "__main__":
    main()
