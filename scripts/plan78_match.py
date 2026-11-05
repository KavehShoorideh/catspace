"""PLAN 78.7 A MATCH OF TWO ENGINE SETTINGS (the new engine against itself): A and B are cs_engine.best_move with different options (e.g. the density prior on / off),
the same judge (the store's valuation record in use), the same clock. Every game opens with seeded random legal moves (the games differ), the colours alternate; the
game is judged by the kernel's outcome and threefold repetition (python-chess's board: the record's rule). -> A's wins / draws / losses, with the score's standard
error, and the games' lengths. One game per process (all cores). Or A against Fairy-Stockfish at a skill (--opponent fsf --skill N).

  .venv/bin/python scripts/plan78_match.py [--games 40] [--seconds 0.5] [--a prior] [--b plain] [--random-plies 4] [--opponent fsf --skill 0]
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import chess
import chess.engine

import cs_engine as CE
from catspace.io import paths

SETTINGS = {"plain": {}, "prior": {"prior": True}, "plain60": {"_scale": 0.6}, "norecord": {"record": False}}      # plain60: the plain engine on 60 % of the clock (the prior's cost alone)
FSF = "/opt/homebrew/bin/fairy-stockfish"


def judge(name: str = "record") -> dict:
    """The judge both settings read: the valuation record in use ('record'), or a VALUATION entry named (a comparison)."""
    from catspace.incubator.analyst.discovery import lemma_store as LS
    return LS.judge_of("chess", None if name == "record" else name)


def play(job: tuple) -> dict:
    i, a_white, sa, sb, seconds, random_plies, opponent, skill, jd = job
    rng = random.Random(7800 + i)
    b = chess.Board()
    moves: list = []
    eng = None
    if opponent == "fsf":
        eng = chess.engine.SimpleEngine.popen_uci(FSF)
        eng.configure({"Skill Level": skill})
    stats = {"a_moves": 0, "a_prior": 0, "a_proved": 0}
    result = None
    try:
        while True:
            fen, mover, legal, outcome, _k = CE.state("chess", None, moves)
            if outcome is not None:
                result = outcome
                break
            if b.is_repetition(3):
                result = [50, 50]
                break
            if len(moves) < random_plies:
                mv = rng.choice(sorted(legal))
            else:
                a_to_move = (mover == "white") == a_white
                if a_to_move:
                    opt = dict(SETTINGS[sa])
                    mv, proved, trace, _n, _f = CE.best_move("chess", None, moves, seconds * opt.pop("_scale", 1.0), **jd, **opt)
                    stats["a_moves"] += 1
                    stats["a_proved"] += proved
                    stats["a_prior"] += any("planning density" in x for x in trace)
                elif opponent == "fsf":
                    mv = eng.play(b, chess.engine.Limit(time=0.05)).move.uci()
                else:
                    opt = dict(SETTINGS[sb])
                    mv, _p, _t, _n, _f = CE.best_move("chess", None, moves, seconds * opt.pop("_scale", 1.0), **jd, **opt)
            if mv not in legal:
                result = None
                break
            b.push_uci(mv)
            moves.append(mv)
    finally:
        if eng:
            eng.quit()
    me = 0 if a_white else 1
    score = None if result is None else (1.0 if result[me] > result[1 - me] else 0.0 if result[me] < result[1 - me] else 0.5)
    return {"game": i, "a_white": a_white, "score": score, "plies": len(moves), **stats, "moves": " ".join(moves)}


def main(argv: list) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=40)
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--a", default="prior")
    ap.add_argument("--b", default="plain")
    ap.add_argument("--random-plies", type=int, default=4)
    ap.add_argument("--opponent", default="self")
    ap.add_argument("--skill", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--judge", default="record")
    a = ap.parse_args(argv)
    jd = judge(a.judge)
    jobs = [(i, i % 2 == 0, a.a, a.b, a.seconds, a.random_plies, a.opponent, a.skill, jd) for i in range(a.games)]
    t0 = time.time()
    with Pool(a.jobs) as pool:
        res = pool.map(play, jobs, chunksize=1)
    Path(paths.experiment(f"plan78_match_{a.tag}.json")).write_text(json.dumps(res, indent=1))
    done = [r for r in res if r["score"] is not None]
    s = [r["score"] for r in done]
    m = sum(s) / max(len(s), 1)
    se = math.sqrt(sum((x - m) ** 2 for x in s) / max(len(s) - 1, 1) / max(len(s), 1))
    w, d, l = sum(x == 1 for x in s), sum(x == 0.5 for x in s), sum(x == 0 for x in s)
    opp = f"Fairy-Stockfish skill {a.skill}" if a.opponent == "fsf" else a.b
    print(f"OBS plan 78.7 match ({a.tag}): A = {a.a} against {opp}; {len(res)} games ({len(res) - len(done)} unfinished), {a.seconds} s a move, {a.random_plies} random opening "
          f"plies, the judge {jd.get('valuation')}; A scores {m:.3f} (se {se:.3f}): W {w} D {d} L {l}; plies {sum(r['plies'] for r in res) / len(res):.0f} a game; A's moves "
          f"{sum(r['a_moves'] for r in res)}, the prior chose {sum(r['a_prior'] for r in res)}, proved {sum(r['a_proved'] for r in res)}; wall {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
