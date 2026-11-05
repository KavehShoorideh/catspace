"""PLAN 78.3 THE MEASURE OF THE END-TO-END SYSTEM: full games of every perfect-information variant -- our engine (cs_engine.best_move: tactics `forcing`, don't
lose, planning `keep`) against a fixed opponent: a seeded RANDOM mover, and FAIRY-STOCKFISH at its lowest skill (an outside opponent and grader, never inside the
engine). The game is judged by our kernel (the outcome per role) and, for the game record's rule, threefold repetition by python-chess's board of the variant.
Colours alternate; one game per process (all cores). Per variant: wins / draws / losses against each opponent, plies, our moves PROVED, the moves where every
candidate lost, the time per move.

  .venv/bin/python scripts/plan78_games.py [--games 4] [--seconds 0.2] [--jobs 11] [--variants chess,atomic,...] [--fsf-skill 0]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine
import chess.variant

import cs_engine as CE
from catspace.io import paths

FSF = "/opt/homebrew/bin/fairy-stockfish"
BOARD = {"chess": chess.Board, "crazyhouse": chess.variant.CrazyhouseBoard, "atomic": chess.variant.AtomicBoard, "antichess": chess.variant.AntichessBoard,
         "kingofthehill": chess.variant.KingOfTheHillBoard, "3check": chess.variant.ThreeCheckBoard, "racingkings": chess.variant.RacingKingsBoard,
         "horde": chess.variant.HordeBoard}


def worth(variant: str = "chess") -> dict:
    """The game's own valuation from the store (plan 78.4: `valuation_of`); a game the store knows no valuation of gets none (nothing borrowed from chess)."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    v = LS.valuation_of(variant)
    return {k: int(x) for k, x in v.worth.items()} if v is not None else {}


def board_of(variant: str, start960: int) -> chess.Board:
    if variant == "chess960":
        return chess.Board.from_chess960_pos(start960)
    return BOARD[variant]()


def play(job: tuple) -> dict:
    variant, opponent, ours_white, seed, seconds, skill, w = job
    rng = random.Random(seed)
    start960 = rng.randrange(960) if variant == "chess960" else 518
    b = board_of(variant, start960)
    moves: list = []
    eng = None
    if opponent == "fairy-stockfish":
        eng = chess.engine.SimpleEngine.popen_uci(FSF)
        eng.configure({"Skill Level": skill})
    stats = Counter()
    t_ours = 0.0
    result = None
    try:
        while True:
            fen, mover, legal, outcome, _key = CE.state(variant, None, moves, start960)
            if outcome is not None:
                result = outcome
                break
            if b.is_repetition(3):
                result = [50, 50]
                stats["threefold"] += 1
                break
            ours = (mover == "white") == ours_white
            if ours:
                t = time.time()
                mv, proved, trace, nodes, _f = CE.best_move(variant, None, moves, seconds, w, start960)
                t_ours += time.time() - t
                stats["our moves"] += 1
                stats["proved"] += proved
                stats["every move lost"] += any("every move loses" in x for x in trace)
            elif opponent == "random":
                mv = rng.choice(sorted(legal))
            else:
                mv = eng.play(b, chess.engine.Limit(time=0.05)).move.uci()
            if mv not in legal:                                   # (both name castling alike: e1g1, and king-takes-rook in Chess960)
                result = None
                stats["illegal " + ("ours" if ours else "theirs")] += 1
                break
            b.push_uci(mv)
            moves.append(mv)
    finally:
        if eng:
            eng.quit()
    me = 0 if ours_white else 1
    score = None if result is None else ("win" if result[me] > result[1 - me] else "loss" if result[me] < result[1 - me] else "draw")
    return {"variant": variant, "opponent": opponent, "ours_white": ours_white, "seed": seed, "score": score, "plies": len(moves), "stats": dict(stats),
            "sec_per_move": t_ours / max(stats["our moves"], 1), "start960": start960}


def main(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--variants", default=",".join(CE.variants()))
    ap.add_argument("--opponents", default="random,fairy-stockfish")
    ap.add_argument("--fsf-skill", type=int, default=0)
    ap.add_argument("--tag", default="e2e")
    a = ap.parse_args(argv)
    jobs = [(v, o, g % 2 == 0, 7800 + g, a.seconds, a.fsf_skill, worth(v)) for v in a.variants.split(",") for o in a.opponents.split(",") for g in range(a.games)]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        res = pool.map(play, jobs, chunksize=1)
    out = Path(paths.experiment(f"plan78_games_{a.tag}.json"))
    out.write_text(json.dumps(res, indent=1))
    agg = defaultdict(Counter)
    for r in res:
        k = (r["variant"], r["opponent"])
        agg[k][r["score"] or "unfinished"] += 1
        agg[k]["plies"] += r["plies"]
        for s, n in r["stats"].items():
            agg[k][s] += n
        agg[k]["sec"] += r["sec_per_move"]
    print(f"OBS plan 78.3 games ({a.tag}): {len(res)} games, {a.seconds} s per move of ours, Fairy-Stockfish skill {a.fsf_skill} at 0.05 s, wall {time.time() - t0:.0f} s -> {out}")
    for (v, o), c in sorted(agg.items()):
        n = sum(c[x] for x in ("win", "draw", "loss", "unfinished"))
        print(f"   {v:14s} vs {o:16s} W {c['win']} D {c['draw']} L {c['loss']}" + (f" unfinished {c['unfinished']}" if c["unfinished"] else "") +
              f" | plies {c['plies'] / n:.0f}/game, our moves (all games) {c["our moves"]}, proved {c['proved']}, every-move-lost {c['every move lost']}, "
              f"threefold {c['threefold']}, {c['sec'] / n:.2f} s/move" + "".join(f", {k} {c[k]}" for k in c if k.startswith("illegal")))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
