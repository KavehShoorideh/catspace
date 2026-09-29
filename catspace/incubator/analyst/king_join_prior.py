"""THE KING-JOIN PRIOR (Kaveh 2026-09-28: "whenever in doubt, the prior should opt for the bigger frame ... if you're less
than 75% sure that the smaller frame is the right one, then just go for the bigger frame"; chosen: "learn the
king-join prior").

The counter-check rule joins a king's frame to any frame whose pieces can check that king; measured, joining changes
the frame's resolution in about a quarter of the cases, and a joined frame reads nearly the whole board (no reuse).
The prior predicts, from the frame and the king in frame terms, whether joining changes the resolution; the frame
joins when that probability is at least 25%. Trained on frames of training positions (and their children), scored on
held-out ones (a hash split by position identity): how many of the joins that matter it keeps (recall at 25%), and how
many joins it avoids.

    python -m catspace.incubator.analyst.king_join_prior --n 600          # train, score, save
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import chess
import numpy as np

from catspace.io import paths

from .gauge_verdicts import positions

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
FEATURES = ["own_king", "n_checks", "n_capture_checks", "checks_by_queen", "king_flights", "king_neighbours_attacked",
            "target_value", "frame_pieces", "queens_in_frame", "pinned_in_frame", "king_distance", "king_in_frame", "king_frame_pieces"]
# (the frame's value without the king was a feature -- weight -0.09 -- and cost a resolution per gate call: dropped 2026-09-28)
THRESHOLD = 0.25


def features(b: chess.Board, squares, members, c: chess.Color, checks: list, alone: float | None = None, kings=frozenset()) -> list:
    """The frame and the king of colour c, in frame terms."""
    from .fates import king_frame, resolve_frame
    k = b.king(c); other = not c
    t = b if b.turn != c else b                                        # checks are counted on the side that gives them
    flights = 0; attacked = 0
    for q in chess.SquareSet(chess.BB_KING_ATTACKS[k]):
        hit = b.is_attacked_by(other, q)
        attacked += hit
        pc = b.piece_at(q)
        flights += (not hit) and (pc is None or pc.color != c)
    targets = [q for q in squares if b.piece_at(q) is not None]
    tv = max((VAL[b.piece_at(q).piece_type] for q in targets), default=0)
    queens = sum(1 for q in members if b.piece_at(q) is not None and b.piece_at(q).piece_type == chess.QUEEN)
    pinned = sum(1 for q in members if b.piece_at(q) is not None and b.is_pinned(b.piece_at(q).color, q))
    dist = min((chess.square_distance(q, k) for q in squares), default=8)
    ks, km = king_frame(b, c)
    qchecks = sum(1 for m in checks if b.piece_at(m.from_square) is not None and b.piece_at(m.from_square).piece_type == chess.QUEEN)
    capchecks = sum(1 for m in checks if b.piece_at(m.to_square) is not None)
    return [c == b.turn, len(checks), capchecks, qchecks, flights, attacked, tv, len(members), queens, pinned, dist, k in members, len(km)]


def _rows(fen: str) -> list:
    """(features, joining changes the resolution) for every frame the counter-check rule would join, in the position
    and in six of its children."""
    from . import fates
    fates.KING_JOIN_GATE = None                                   # the rule's own joins, unfiltered
    from .fates import joint_base, counter_checks, king_frame, local_resolve, resolve_frame
    root = chess.Board(fen)
    boards = [root] + [root.copy() for _ in range(6)]
    for bb, m in zip(boards[1:], list(root.legal_moves)[:6]):
        bb.push(m)
    out = []
    for b in boards:
        if b.is_check() or b.is_game_over():
            continue
        for sq, pc in b.piece_map().items():
            if pc.color == b.turn or pc.piece_type == chess.KING or not b.attackers(b.turn, sq):
                continue
            squares, kings, members = joint_base(b, {sq}, set())
            if kings:
                continue                                          # a king already joined by a line the frame opens: not this rule's
            alone = resolve_frame(b, squares, frozenset(kings), members)[0]
            for c in (chess.WHITE, chess.BLACK):
                checks = counter_checks(b, members, c)
                if not checks:
                    continue
                ks, km = king_frame(b, c)
                joined = local_resolve(b, squares | ks, set(members) | km, {c})[0]
                out.append((features(b, squares, members, c, checks, alone), joined != alone))
    return out


def _split(fen: str) -> str:
    return "test" if int(hashlib.sha1(("kingjoin" + fen).encode()).hexdigest()[:8], 16) % 3 == 0 else "train"


def load():
    """The saved prior as a gate for fates.KING_JOIN_GATE: join when P(joining changes the resolution) >= THRESHOLD."""
    d = json.load(open(paths.derived("king_join_prior.json")))
    w = np.array(d["coef"]); b0 = d["intercept"]; mu = np.array(d["mean"]); sd = np.array(d["scale"])

    def gate(b, squares, kings, members, c, checks):
        x = (np.array(features(b, squares, members, c, checks, kings=kings), dtype=float) - mu) / sd
        return 1.0 / (1.0 + np.exp(-(x @ w + b0))) >= d["threshold"]
    return gate


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=600); ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    import multiprocessing as mp
    fens = [b.fen() for b in positions(a.n, 0)]
    with mp.get_context("fork").Pool(a.workers) as pool:
        got = pool.map(_rows, fens, chunksize=2)
    train = [r for f, rs in zip(fens, got) if _split(f) == "train" for r in rs]; test = [r for f, rs in zip(fens, got) if _split(f) == "test" for r in rs]
    X = np.array([f for f, _ in train], dtype=float); y = np.array([l for _, l in train])
    Xt = np.array([f for f, _ in test], dtype=float); yt = np.array([l for _, l in test])
    sc = StandardScaler().fit(X); model = LogisticRegression(max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    p = model.predict_proba(sc.transform(Xt))[:, 1]
    print(f"[king_join_prior] {len(train)} training frames ({y.mean():.2f} where joining changes the resolution), {len(test)} held out "
          f"({yt.mean():.2f}); {time.perf_counter() - t0:.0f}s")
    print("OBS weights (standardized): " + ", ".join(f"{n} {w:+.2f}" for n, w in sorted(zip(FEATURES, model.coef_[0]), key=lambda t: -abs(t[1]))))
    for th in (0.1, 0.25, 0.5):
        join = p >= th
        print(f"OBS threshold {th:.2f}: joins {join.mean():.0%} of the frames the rule would join; of the joins that change the resolution it keeps "
              f"{(join & yt).sum()}/{yt.sum()} ({(join & yt).sum() / max(1, yt.sum()):.0%}); joins avoided {(~join).sum()}/{len(yt)}")
    json.dump({"features": FEATURES, "coef": model.coef_[0].tolist(), "intercept": float(model.intercept_[0]), "mean": sc.mean_.tolist(),
               "scale": sc.scale_.tolist(), "threshold": THRESHOLD}, open(paths.derived("king_join_prior.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
