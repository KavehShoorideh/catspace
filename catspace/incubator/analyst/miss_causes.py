"""WHAT THE PLANNER DOES NOT SEE (Kaveh 2026-10-10: "you have the Stockfish continuation -- can't you just look at it and
see what the issue is?"). For each decision losing 150+ cp (decision_review.json), from Stockfish's lines (depth 18
here, both after the planner's move and from the position): MISSED MATE (we had one: its length, all checks or not),
ALLOWED MATE (our move walks into one: its length, all checks or not), else MATERIAL -- what their line after our move
wins and its first move's kind (a check, a capture, quiet), or what our best line wins that our move does not."""
import json
import os
from collections import Counter
from multiprocessing import Pool

import chess
import chess.engine

from catspace.io import paths

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
_ENG = None


def mat(b, c):
    return sum(VAL[p.piece_type] for p in b.piece_map().values() if p.color == c)


def line_info(b, pv, us):
    """(all of `us`'s moves checks?, material swing for us along the line, first move san, its kind)"""
    x = b.copy(); checks = True; first = None; kind = None
    m0 = mat(x, us) - mat(x, not us)
    for i, m in enumerate(pv):
        if m not in x.legal_moves:
            break
        san = x.san(m); cap = x.is_capture(m); mover = x.turn
        x.push(m)
        if i == 0:
            first, kind = san, ("check" if x.is_check() else "capture" if cap else "quiet")
        if mover == us and not x.is_check():
            checks = False
    return checks, (mat(x, us) - mat(x, not us)) - m0, first, kind


def one(r):
    global _ENG
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    b = chess.Board(r["fen"]); us = b.turn
    a = b.copy(); a.push_uci(r["move"])
    i0 = _ENG.analyse(b, chess.engine.Limit(depth=18)); i1 = _ENG.analyse(a, chess.engine.Limit(depth=18))
    s0, s1 = i0["score"].pov(us), i1["score"].pov(us)
    out = dict(fen=r["fen"], move=b.san(chess.Move.from_uci(r["move"])), kind=r["kind"], summary=r["summary"])
    if s1.is_mate() and s1.mate() > 0:
        out.update(cause="not a miss: our move mates too (depth 12 did not see it)", n=s1.mate(), line=a.variation_san(i1["pv"][:2 * s1.mate()]))
    elif s0.is_mate() and s0.mate() > 0:
        ch, _, first, k = line_info(b, i0["pv"], us)
        out.update(cause="missed our mate", n=s0.mate(), checks=ch, first=first, first_kind=k, line=b.variation_san(i0["pv"][:2 * s0.mate()]))
    elif s1.is_mate() and s1.mate() < 0:
        ch, _, first, k = line_info(a, i1["pv"], not us)
        out.update(cause="allowed their mate", n=-s1.mate(), checks=ch, first=first, first_kind=k, line=a.variation_san(i1["pv"][:2 * -s1.mate()]))
    else:
        _, swing_after, first, k = line_info(a, i1["pv"][:8], us)
        _, swing_best, bfirst, bk = line_info(b, i0["pv"][:9], us)
        if swing_best - (swing_after + (mat(a, us) - mat(a, not us)) - (mat(b, us) - mat(b, not us))) >= 2 and swing_after >= -1:
            out.update(cause="missed a bigger win", first=bfirst, first_kind=bk, gain=swing_best, line=b.variation_san(i0["pv"][:7]))
        elif swing_after <= -2:
            out.update(cause="our move loses material", first=first, first_kind=k, gain=swing_after, line=a.variation_san(i1["pv"][:7]))
        else:
            out.update(cause="positional / longer (no material in 8 plies)", first=first, first_kind=k, line=a.variation_san(i1["pv"][:7]),
                       best_line=b.variation_san(i0["pv"][:7]))
    out["loss"] = r["loss"]
    return out


if __name__ == "__main__":
    rows = json.load(open(paths.experiment("decision_review.json")))
    bad = sorted([r for r in rows if r["loss"] >= 150], key=lambda r: -r["loss"])
    with Pool(os.cpu_count()) as p:
        rs = p.map(one, bad, chunksize=2)
    json.dump(rs, open(paths.experiment("miss_causes.json"), "w"))
    print("OBS causes of the", len(rs), "decisions losing 150+:", dict(Counter(r["cause"] for r in rs).most_common()))
    for c in ("missed our mate", "allowed their mate"):
        g = [r for r in rs if r["cause"] == c]
        if g:
            print(f"OBS   {c}: n {len(g)}; length {dict(sorted(Counter(r['n'] for r in g).items()))}; all checks {sum(r['checks'] for r in g)}; "
                  f"by decision kind {dict(Counter(r['kind'] for r in g))}")
    for c in ("our move loses material", "missed a bigger win", "positional / longer (no material in 8 plies)"):
        g = [r for r in rs if r["cause"] == c]
        if g:
            print(f"OBS   {c}: n {len(g)}; first move kind {dict(Counter(r['first_kind'] for r in g))}; by decision kind {dict(Counter(r['kind'] for r in g))}")
