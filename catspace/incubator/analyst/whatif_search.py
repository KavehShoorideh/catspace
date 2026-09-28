"""THE WHAT-IF SEARCH (Kaveh 2026-09-28: "these what-if questions are the search, but knowing which ones to consider and
in what priority is the main kind of learning"; chosen next: "answer edits, then search").

  1. ASK: the hypothetical edits of the side to move (`whatif.edits`: add an attacker, remove a defender, relocate a
     piece, guard), each answered statically by the frames.
  2. PRIORITIZE: a learned edit priority -- trained on training positions, an edit is good when a near-best move
     (Stockfish within 0.3) realizes it; the labels cost no search.
  3. ANSWER: for the top-k edits only, the moves that realize them are tested against the opponent's best reply
     (`lambda_frames.threat_after`: order 1 for a move that completes the edit, order 2 for one that starts it).
  4. CHOOSE: among the frames' tied moves, the move with the best answered value; when nothing survives the reply,
     the frames' first tie (so the search's own contribution is what is measured).

Measured on held-out positions (a hash split by position identity) against k -- the what-ifs asked -- for the learned
priority, the static gain and a random order: how often the choice is Stockfish's best, its mean loss, and the
answers computed (the cost).

    python -m catspace.incubator.analyst.whatif_search --n 1000
"""
from __future__ import annotations

import argparse
import hashlib
import random
import time

import chess
import numpy as np

from .gauge_verdicts import positions
from .engine import Engine

KINDS = ("add", "remove", "relocate", "guard")
EDIT_FEATURES = ["add", "remove", "relocate", "guard", "gain", "target_value", "attacker_value", "hops", "n_completes", "n_starts", "target_defended"]
VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
KS = (1, 2, 4, 8)


def _split(fen: str) -> str:
    return "test" if int(hashlib.sha1(("whatif" + fen).encode()).hexdigest()[:8], 16) % 3 == 0 else "train"


def _collect(fen: str):
    """-> (fen, ties, [(edit features, {move uci: 1 completes | 2 starts})]) for a position with tied moves."""
    import catspace.incubator.analyst.frame_player as FP
    from .whatif import edits, realizes
    from .frame_goals import counted_defenders
    FP.PLANS = "none"
    b = chess.Board(fen)
    c = FP.choose(b)
    if len(c["best"]) < 2:
        return fen, [], []
    ties = [chess.Move.from_uci(u) for u in c["best"]]
    out = []
    for e in edits(b):
        r = {}
        for m in ties:
            k = realizes(b, m, e)
            if k:
                r[m.uci()] = k
        if not r:
            continue                                              # no tied move makes it true: not a candidate here
        tp = b.piece_at(e.target)
        f = [e.kind == k for k in KINDS] + [e.gain, VAL[tp.piece_type] if tp else 0, e.arg if e.kind in ("add", "guard") else 0,
                                            e.hops if e.kind == "relocate" else 1, sum(v == 1 for v in r.values()), sum(v == 2 for v in r.values()),
                                            len(counted_defenders(b, e.target)) if tp else 0]
        out.append((f, r))
    return fen, c["best"], out


def _answer(args):
    """The answered value of each (move, order) pair of a held-out position: what survives the opponent's best reply."""
    fen, pairs = args
    from .lambda_frames import threat_after
    b = chess.Board(fen)
    return fen, {(u, o): threat_after(b, chess.Move.from_uci(u), o, 150) for u, o in pairs}


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--good-cp", type=int, default=30)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    import multiprocessing as mp
    boards = positions(a.n, 0)
    with mp.get_context("fork").Pool(a.workers) as pool:
        got = [g for g in pool.map(_collect, [b.fen() for b in boards], chunksize=2) if g[1] and g[2]]
    eng = Engine(threads=8, depth=a.depth)
    loss = {}; sf_best = {}
    try:
        for fen, ties, _ in got:
            b = chess.Board(fen)
            lines = eng.analyse(b, multipv=b.legal_moves.count(), depth=a.depth)
            cp = {l.move: l.cp for l in lines}; best = max(cp.values())
            loss[fen] = {u: best - cp.get(u, best - 1000) for u in ties}; sf_best[fen] = max(cp, key=cp.get)
    finally:
        eng.close()
    train = [g for g in got if _split(g[0]) == "train"]; test = [g for g in got if _split(g[0]) == "test"]
    X = np.array([f for fen, _, es in train for f, _ in es], dtype=float)
    y = np.array([any(loss[fen][u] <= a.good_cp for u in r) for fen, _, es in train for _, r in es])
    sc = StandardScaler().fit(X); model = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    print(f"[whatif_search] {len(got)} positions with realizable what-ifs ({len(train)} train, {len(test)} test); {len(X)} training edits "
          f"({y.mean():.2f} realized by a near-best move); {time.perf_counter() - t0:.0f}s")
    print("OBS edit priority weights (standardized): " + ", ".join(f"{n} {w:+.2f}" for n, w in sorted(zip(EDIT_FEATURES, model.coef_[0]), key=lambda t: -abs(t[1]))))
    rng = random.Random(0)
    orders = {"learned priority": lambda es: list(model.predict_proba(sc.transform(np.array([f for f, _ in es], dtype=float)))[:, 1]),
              "static gain": lambda es: [f[4] for f, _ in es],
              "random": lambda es: [rng.random() for _ in es]}
    ranked = {name: {fen: sorted(range(len(es)), key=lambda i, s=score(es): -s[i]) for fen, _, es in test} for name, score in orders.items()}
    need = {}                                                         # every (move, order) any ordering asks within k = max
    for name in orders:
        for fen, _, es in test:
            for i in ranked[name][fen][:max(KS)]:
                for u, k in es[i][1].items():
                    need.setdefault(fen, set()).add((u, 1 if k == 1 else 2))
    with mp.get_context("fork").Pool(a.workers) as pool:
        answered = dict(pool.map(_answer, [(fen, sorted(p)) for fen, p in need.items()], chunksize=1))
    print(f"[whatif_search] answered {sum(len(v) for v in answered.values())} (move, order) pairs on {len(answered)} held-out positions; {time.perf_counter() - t0:.0f}s")
    n = len(test)
    base = [ties[0] for _, ties, _ in test]
    print(f"OBS the frames' first tie (no what-if asked): first choice = Stockfish's best {sum(u == sf_best[f] for u, (f, _, _) in zip(base, test))}/{n}; "
          f"mean loss {np.mean([min(loss[f][u], 1000) for u, (f, _, _) in zip(base, test)]):.0f} cp")
    for name in orders:
        for k in KS:
            firsts = 0; losses = []; asked = []; answered_n = []; spoke = 0
            for fen, ties, es in test:
                vals = {}
                for i in ranked[name][fen][:k]:
                    for u, kk in es[i][1].items():
                        vals[u] = max(vals.get(u, 0.0), answered[fen].get((u, 1 if kk == 1 else 2), 0.0))
                asked.append(min(k, len(es))); answered_n.append(len(vals))
                top = max(vals.values(), default=0.0)
                pick = next(u for u in ties if vals.get(u, 0.0) == top) if top > 0 else ties[0]
                spoke += top > 0
                firsts += pick == sf_best[fen]; losses.append(min(loss[fen][pick], 1000))
            print(f"OBS {name:16s} k={k}: first choice = Stockfish's best {firsts}/{n}; mean loss {np.mean(losses):.0f} cp; an answered what-if decided "
                  f"{spoke}/{n}; what-ifs asked {np.mean(asked):.1f}, moves answered {np.mean(answered_n):.1f} per position")


if __name__ == "__main__":
    main()
