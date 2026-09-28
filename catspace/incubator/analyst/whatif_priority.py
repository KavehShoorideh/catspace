"""WHAT-IF PRIORITY, LEARNED (Kaveh 2026-09-28: "we always want to be asking a what-if question: what if there was a
knight here? what if there was another attacker here? what if we removed this attacker? ... these what-if questions
are the search, but knowing which ones to consider and in what priority is the main kind of learning ... put my idea
to the test").

The test. In every held-out position the frames leave a set of moves they cannot tell apart (their tied top set);
each such move is a what-if that can be made true now ("what if my knight stood on h5"). For each, the what-if is
asked in the frames' own terms -- its static worth (`frame_placement`: the moved piece's marginal worth in the frames
it would bear on, less where it stood), its realized threat (`lambda_frames.threat_after`: what survives the
opponent's best reply), and what it touches (the moved kind, mobility, attacks and defences gained, the kings'
distance, advance). A logistic model learns from training positions which what-ifs Stockfish's near-best moves are
made of; on held-out positions (a hash split by position identity) the learned priority orders the tied moves, and is
compared with the frames' first tie, a random tie, the static what-if, and the realized threat: how often the first
choice is Stockfish's best, its mean loss, and how early the best move comes up (search efficiency: the what-ifs one
must ask before reaching it).

    python -m catspace.incubator.analyst.whatif_priority --n 1000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time

import chess
import numpy as np

from catspace.io import paths

from .gauge_verdicts import positions
from .engine import Engine

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
FEATURES = ["static_worth", "static_here", "static_further", "worth_now", "realized_threat", "pawn", "knight", "bishop", "rook", "queen", "king",
            "check", "capture", "castle", "mobility_gain", "attacks_gain", "attacked_value_gain", "defends_gain", "enemy_king_closer",
            "own_king_closer", "advance"]


def _attacked(b: chess.Board, sq: int, side: chess.Color) -> tuple:
    enemy = [t for t in b.attacks(sq) if b.piece_at(t) is not None and b.piece_at(t).color != side and b.piece_at(t).piece_type != chess.KING]
    own = [t for t in b.attacks(sq) if b.piece_at(t) is not None and b.piece_at(t).color == side]
    return len(enemy), sum(VAL[b.piece_at(t).piece_type] for t in enemy), len(own)


def _mobility(b: chess.Board, sq: int) -> int:
    bb = b.copy(stack=False); bb.turn = b.piece_at(sq).color; bb.ep_square = None
    try:
        return sum(1 for m in bb.legal_moves if m.from_square == sq)
    except Exception:
        return 0


def features(b: chess.Board, m: chess.Move) -> list:
    """The what-if 'this piece on that square', asked in the frames' terms and in what it touches."""
    from .frame_placement import worth, _moved, placements, HALF
    from .lambda_frames import threat_after
    from .fates import passed
    side = b.turn; pc = b.piece_at(m.from_square)
    quiet = not (b.is_capture(m) or m.promotion or pc.piece_type == chess.KING)
    now = worth(b, m.from_square) if quiet else 0.0
    here = further = 0.0
    if quiet:
        at = _moved(b, m.from_square, m.to_square)
        here = worth(at, m.to_square)
        further = max((HALF * w for q, h, w, r in placements(at, m.to_square, 1)), default=0.0)
    realized = threat_after(b, m, 1, 300)
    nb = b.copy(stack=False); nb.push(m)
    pb = passed(nb) or nb
    a0 = _attacked(b, m.from_square, side); a1 = _attacked(pb, m.to_square, side) if pb.piece_at(m.to_square) is not None else (0, 0, 0)
    mob0 = _mobility(b, m.from_square); mob1 = _mobility(pb, m.to_square) if pb.piece_at(m.to_square) is not None else 0
    ek = b.king(not side); ok = b.king(side)
    dist = lambda s, k: chess.square_distance(s, k) if k is not None else 0
    adv = (chess.square_rank(m.to_square) - chess.square_rank(m.from_square)) * (1 if side == chess.WHITE else -1)
    kinds = [pc.piece_type == k for k in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)]
    return [max(here, further) - now, here, further, now, realized, *kinds, nb.is_check(), b.is_capture(m), b.is_castling(m), mob1 - mob0,
            a1[0] - a0[0], a1[1] - a0[1], a1[2] - a0[2], dist(m.from_square, ek) - dist(m.to_square, ek), dist(m.from_square, ok) - dist(m.to_square, ok), adv]


def _collect(fen: str):
    from .frame_player import choose
    import catspace.incubator.analyst.frame_player as FP
    FP.PLANS = "none"
    b = chess.Board(fen)
    c = choose(b)
    if len(c["best"]) < 2:
        return fen, []
    return fen, [(u, features(b, chess.Move.from_uci(u))) for u in c["best"]]


def _split(fen: str) -> str:
    return "test" if int(hashlib.sha1(("whatif" + fen).encode()).hexdigest()[:8], 16) % 3 == 0 else "train"


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--workers", type=int, default=10); ap.add_argument("--good-cp", type=int, default=30)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    boards = positions(a.n, 0)
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.workers) as pool:
        got = pool.map(_collect, [b.fen() for b in boards], chunksize=2)
    eng = Engine(threads=8, depth=a.depth)
    rows = []                                                         # (split, fen, [(uci, features, loss)])
    try:
        for fen, cands in got:
            if not cands:
                continue
            b = chess.Board(fen)
            lines = eng.analyse(b, multipv=b.legal_moves.count(), depth=a.depth)
            if not lines:
                continue
            cp = {l.move: l.cp for l in lines}; best = max(cp.values()); sf_best = max(cp, key=cp.get)
            rows.append((_split(fen), fen, [(u, f, best - cp.get(u, best - 1000)) for u, f in cands], sf_best))
    finally:
        eng.close()
    train = [r for r in rows if r[0] == "train"]; test = [r for r in rows if r[0] == "test"]
    X = np.array([f for r in train for _, f, _ in r[2]], dtype=float); y = np.array([loss <= a.good_cp for r in train for _, _, loss in r[2]])
    sc = StandardScaler().fit(X)
    model = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    print(f"[whatif_priority] {len(rows)} positions with tied what-ifs ({len(train)} train, {len(test)} test), "
          f"{len(X)} training what-ifs ({y.mean():.2f} near-best), {time.perf_counter() - t0:.0f}s")
    print("OBS learned weights (standardized): " + ", ".join(f"{n} {w:+.2f}" for n, w in sorted(zip(FEATURES, model.coef_[0]), key=lambda t: -abs(t[1]))))
    rng = random.Random(0)
    orders = {
        "frames' first tie": lambda cands: [0] * len(cands),
        "random tie": lambda cands: [rng.random() for _ in cands],
        "static what-if worth": lambda cands: [f[0] for _, f, _ in cands],
        "realized threat": lambda cands: [f[4] for _, f, _ in cands],
        "learned priority": lambda cands: list(model.predict_proba(sc.transform(np.array([f for _, f, _ in cands], dtype=float)))[:, 1]),
    }
    for name, score in orders.items():
        firsts = 0; losses = []; bad = 0; ranks = []
        for _, fen, cands, sf_best in test:
            s = score(cands)
            order = sorted(range(len(cands)), key=lambda i: (-s[i], i))    # ties keep the frames' order
            pick = cands[order[0]]
            firsts += pick[0] == sf_best; losses.append(min(pick[2], 1000)); bad += pick[2] >= 150
            if any(u == sf_best for u, _, _ in cands):
                ranks.append(1 + next(k for k, i in enumerate(order) if cands[i][0] == sf_best))
        n = len(test)
        print(f"OBS {name:22s}: first choice = Stockfish's best {firsts}/{n}; mean loss {np.mean(losses):.0f} cp; loses 1.5+ in {bad}/{n}; "
              f"where Stockfish's best is tied ({len(ranks)}), it comes up as what-if number {np.mean(ranks):.2f} on average "
              f"(top-1 {sum(r == 1 for r in ranks)}, top-3 {sum(r <= 3 for r in ranks)}; ties per position {np.mean([len(c) for _, _, c, _ in test]):.1f})")
    json.dump({"features": FEATURES, "coef": model.coef_[0].tolist(), "mean": sc.mean_.tolist(), "scale": sc.scale_.tolist()},
              open(paths.derived("whatif_priority.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
