"""WHY THE SINGLE-THREAT LEMMA'S SURPRISES WIN (Kaveh 2026-10-10, on surprise 1: "let's go through them and find the
Stockfish evaluation ... and then reason our way through it. And if we're missing a lemma somewhere, let's bring it
in"). Per surprise: the escape the lemma took as quiet (from the planner's trace), Stockfish (depth 18) on the board
after it -- our refutation, its score, its line -- and the kind of the refutation's first move: a MATE line, a CHECK,
a FORK (a move after which we win two or more pieces by the static count), a CAPTURE, or QUIET. Pool over all cores."""
import json, os, re
from collections import Counter
from multiprocessing import Pool

import chess
import chess.engine

from catspace.io import paths
from catspace.incubator.analyst.lemma_reviewer import judge

_ENG = None


def one(c):
    global _ENG
    os.environ.setdefault("RAYON_NUM_THREADS", "1")
    import analyst_core
    if _ENG is None:
        _ENG = chess.engine.SimpleEngine.popen_uci("stockfish")
    b = chess.Board(c["fen"])
    tr = analyst_core.plan_trace(c["fen"])[0]
    line = next((l for l in tr if l.strip().startswith(c["san"] + ": one threat")), "")
    m = re.search(r"steps away to ([a-h][1-8])", line)
    if not m:
        return dict(c=c, escape=None)
    a = b.copy(); a.push_uci(c["move"])
    esc = next((x for x in a.legal_moves if x.from_square == chess.parse_square(c["target_sq"]) and chess.square_name(x.to_square) == m.group(1)), None)
    if esc is None:
        return dict(c=c, escape=None)
    q = a.copy(); q.push(esc)
    info = _ENG.analyse(q, chess.engine.Limit(depth=18))
    sc = info["score"].pov(b.turn); pv = info.get("pv", [])
    first = pv[0] if pv else None
    kind = "none"
    if sc.is_mate() and (sc.mate() or 0) > 0:
        kind = "mate line"
    elif first is not None:
        r = q.copy(); r.push(first)
        if r.is_check(): kind = "check"
        elif q.is_capture(first): kind = "capture"
        else:
            us = b.turn
            before = {s for s, p in q.piece_map().items() if p.color != us and p.piece_type != chess.PAWN and q.attackers(us, s)}
            after = {s for s, p in r.piece_map().items() if p.color != us and p.piece_type != chess.PAWN and r.attackers(us, s)}
            kind = "fork (2+ new attacks)" if len(after - before) >= 2 else "quiet"
    return dict(c=c, escape=a.san(esc), score=str(sc), kind=kind, line=q.variation_san(pv[:6]) if pv else "")


if __name__ == "__main__":
    d = json.load(open(paths.experiment("lemma_reviewer_single_threat.json")))
    w = sorted([c for c in d["cases"] if c["stratum"] != "passed" and judge(c)], key=lambda c: (not c["sf_tactic"], -c["gain"]))
    with Pool(os.cpu_count()) as p:
        rs = p.map(one, w, chunksize=1)
    print("OBS the refutation of the escape the lemma took as quiet, by kind:", dict(Counter(r.get("kind", "no escape in trace") for r in rs)))
    for i, r in enumerate(rs):
        c = r["c"]
        print(f"  {i + 1:2d}. {c['san']:6s} on {c['target_sq']}: escape {r.get('escape')} -> {r.get('kind')}, SF {r.get('score')}: {r.get('line', '')[:80]}")
    json.dump([{k: v for k, v in r.items() if k != "c"} | dict(fen=r["c"]["fen"], move=r["c"]["move"]) for r in rs],
              open(paths.experiment("single_threat_surprise_refutations.json"), "w"))
