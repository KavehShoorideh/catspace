"""THE PROMOTION LEMMA, audited against Stockfish before it is trusted (bug sweep, 2026-10-10).

Every position (the census and the endgame puzzles) where the planner's decision is "makes an unstoppable passer": the
move played, then Stockfish (depth 18) from the opponent's side. A sound claim leaves the mover winning (>= +3 pawns or
a mate); every claim below is listed for digging.
"""
from __future__ import annotations

import json
import os
from collections import Counter

import chess

from catspace.io import paths


def _plan(fen):
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    r = analyst_core.plan_trace(fen)
    d = r[1]
    if d and d[0] == "" and "unstoppable passer" in d[3]:
        return fen, d[1], d[3], [l for l in r[0] if "promotion lemma" in l]
    return None


_ENG = None


def _sf(args):
    global _ENG
    fen, mv, why, tr = args
    import chess.engine
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    b = chess.Board(fen); us = b.turn
    b.push(chess.Move.from_uci(mv))
    sc = _ENG.analyse(b, chess.engine.Limit(depth=18))["score"].pov(us)
    return fen, mv, why, tr, sc.score(mate_score=10000)


def main():
    from multiprocessing import Pool
    from .tactic_census_measure import load
    fens = [r["fen"] for r in load("c300")]
    pz = paths.experiment("endgame_puzzles_60.jsonl")
    if os.path.exists(pz):
        fens += [json.loads(l)["fen"] for l in open(pz)]
    with Pool(10) as p:
        claims = [c for c in p.imap_unordered(_plan, fens, chunksize=64) if c]
    print(f"OBS promotion-lemma claims: {len(claims)} positions (census + endgame puzzles)")
    with Pool(10) as p:
        res = list(p.imap_unordered(_sf, claims, chunksize=2))
    v = Counter(); bad = []
    for fen, mv, why, tr, cp in res:
        k = "winning (>= +300 or mate)" if cp >= 300 else ("better (+100..+300)" if cp >= 100 else "NOT winning (< +100)")
        v[k] += 1
        if cp < 100:
            bad.append((cp, fen, mv, why))
    print(f"OBS Stockfish depth 18 after the claimed move: {dict(v)}")
    for cp, fen, mv, why in sorted(bad)[:12]:
        print(f"OBS   {cp:+5d}  {why}  {fen}")


if __name__ == "__main__":
    main()
