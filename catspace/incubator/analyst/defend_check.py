"""WHEN THE PLANNER DEFENDS INSTEAD OF TAKING A WIN -- checked against Stockfish (bug sweep, 2026-10-09).

After the review fixes (a check does not answer a threat; our move must expose nothing worth as much), the planner on
census positions found an immediate win but decided to defend more often than before. For every such position: our win,
their threat, the defence chosen, and Stockfish's evaluation of our winning move and of the defence (depth 12, mover's
view) -- which one was right.
"""
from __future__ import annotations

import os
from collections import Counter

import chess


def _one(fen):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    r = analyst_core.plan_trace(fen)
    return fen, r[1], r[5]


_ENG = None


def _sf(args):
    global _ENG
    fen, moves = args
    import chess.engine
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    b = chess.Board(fen)
    out = {}
    for u in moves:
        bb = b.copy(); bb.push(chess.Move.from_uci(u))
        sc = _ENG.analyse(bb, chess.engine.Limit(depth=12))["score"].pov(b.turn)
        out[u] = max(-1500, min(1500, sc.score(mate_score=10000)))
    return fen, out


def main():
    from multiprocessing import Pool
    from .tactic_census_measure import load
    fens = [r["fen"] for r in load("c300")]
    with Pool(10) as p:
        res = list(p.imap_unordered(_one, fens, chunksize=64))
    cases = [(fen, d, i) for fen, d, i in res if i and d and d[0] == "defend"]
    print(f"OBS {len(cases)} positions where the planner found an immediate win and decided to defend")
    with Pool(10) as p:
        sf = dict(p.imap_unordered(_sf, [(fen, [i[1], d[1]]) for fen, d, i in cases], chunksize=4))
    verdict = Counter()
    for fen, d, i in cases:
        w, df = sf[fen][i[1]], sf[fen][d[1]]
        k = "defence better (>= 50 cp)" if df >= w + 50 else ("win better (>= 50 cp)" if w >= df + 50 else "within 50 cp")
        verdict[k] += 1
    print(f"OBS Stockfish (depth 12, mover's view) on our winning move against the chosen defence: {dict(verdict)}")
    b = chess.Board
    for fen, d, i in cases[:8]:
        print(f"OBS   win {i[3]} ({sf[fen][i[1]]:+d}) vs {d[3]} ({sf[fen][d[1]]:+d})  {fen}")


if __name__ == "__main__":
    main()
