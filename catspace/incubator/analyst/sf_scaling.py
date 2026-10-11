"""ELO AGAINST TIME, IN GENERAL (Kaveh 2026-10-10: "Elo versus time in general -- like Stockfish, how much time is worth how
much Elo"). Stockfish against itself at doubling node budgets (N vs 2N), Threads 1, from openings taken from the census
games (ply 10, both colours each), to the end or a ply cap (a draw); per doubling the score of the larger budget, the
Elo difference with a 95% interval, and the time a move costs at each budget. Pool over all cores, one engine per
worker."""
from __future__ import annotations

import argparse
import math
import os
import random
import time

import chess
import chess.engine

_ENG = None


def _game(args):
    global _ENG
    fen, n_white, n_black, gid = args
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
        _ENG.configure({"Threads": 1, "Hash": 16})
    b = chess.Board(fen)
    spent = {n_white: [0.0, 0], n_black: [0.0, 0]}
    while not b.is_game_over(claim_draw=True) and b.ply() < 400:
        n = n_white if b.turn else n_black
        t = time.perf_counter()
        r = _ENG.play(b, chess.engine.Limit(nodes=n), game=gid)
        spent[n][0] += time.perf_counter() - t; spent[n][1] += 1
        b.push(r.move)
    res = b.result(claim_draw=True)
    pts_white = {"1-0": 1.0, "0-1": 0.0}.get(res, 0.5)
    return n_white, n_black, pts_white, spent


def elo(s):
    s = min(max(s, 1e-3), 1 - 1e-3)
    return 400 * math.log10(s / (1 - s))


def main(argv=None):
    from multiprocessing import Pool
    from .tactic_census_measure import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--openings", type=int, default=100)
    ap.add_argument("--budgets", default="1000,2000,4000,8000,16000,32000,64000,128000")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 2)
    a = ap.parse_args(argv)
    rows = load("c300")
    games = sorted({r["game"] for r in rows})
    by = {(r["game"], r["ply"]): r["fen"] for r in rows}
    ops = [by[(g, 10)] for g in random.Random(1).sample(games, len(games)) if (g, 10) in by][: a.openings]
    budgets = [int(x) for x in a.budgets.split(",")]
    jobs = []
    gid = 0
    for lo, hi in zip(budgets, budgets[1:]):
        for f in ops:
            jobs.append((f, hi, lo, gid)); gid += 1
            jobs.append((f, lo, hi, gid)); gid += 1
    t0 = time.time()
    with Pool(a.workers) as p:
        res = p.map(_game, jobs, chunksize=1)
    print(f"TIME {time.time() - t0:.0f} s for {len(jobs)} games")
    per_move = {}
    for nw, nb, pw, spent in res:
        for n, (sec, k) in spent.items():
            per_move.setdefault(n, [0.0, 0]); per_move[n][0] += sec; per_move[n][1] += k
    print("OBS Stockfish, N nodes against 2N (Threads 1), 2 games per opening (both colours); the larger budget's score:")
    print(f"OBS   {'N -> 2N':>16s} {'games':>6s} {'W-D-L':>11s} {'score':>6s} {'Elo':>6s} {'95% interval':>16s} {'ms/move at 2N':>14s}")
    for lo, hi in zip(budgets, budgets[1:]):
        g = [(nw, nb, pw) for nw, nb, pw, _ in res if {nw, nb} == {lo, hi}]
        pts = [pw if nw == hi else 1 - pw for nw, nb, pw in g]
        n = len(pts); s = sum(pts) / n
        w = sum(x == 1 for x in pts); d = sum(x == 0.5 for x in pts); l = n - w - d
        var = sum((x - s) ** 2 for x in pts) / n
        se = math.sqrt(var / n)
        lo_e, hi_e = elo(s - 1.96 * se), elo(s + 1.96 * se)
        ms = 1000 * per_move[hi][0] / per_move[hi][1]
        print(f"OBS   {lo:>7d} -> {hi:<7d} {n:6d} {w:3d}-{d:3d}-{l:3d} {s:6.3f} {elo(s):+6.0f} [{lo_e:+5.0f}, {hi_e:+5.0f}] {ms:13.1f}")


if __name__ == "__main__":
    main()
