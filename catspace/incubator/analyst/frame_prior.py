"""THE FRAME-DETECTION PRIOR (Kaveh 2026-09-27): "use that labeling to train a frame detection prior, and then use that
frame detection prior to pick the right frame at a given moment". Given a position and a contested square, the
probability that each piece belongs to the square's frame, learned from the exchanges in games (`exchanges.py`):
per-piece features in the rules' own words -- attacks or defends the square directly, stands behind one or two
pieces on a line to it (x-ray), is one move from attacking it, attacks or defends a piece that attacks or defends
it, its kind, its distance, pinned -- and a logistic regression, so the prior reads as weights. Trained on the
training games, scored on the held-out games (game id % 5 == 0) against rules that pick frames without learning.

    python -m catspace.incubator.analyst.frame_prior          # train, score, save
"""
from __future__ import annotations

import argparse
import json
import pickle

import chess
import numpy as np

from catspace.io import paths

from .chess_domain import VAL

FEATURES = ["occupant", "same_side", "direct", "xray1", "xray1_behind_participant", "xray2", "one_move", "hits_participant", "backs_participant", "pinned",
            "pawn", "knight", "bishop", "rook", "queen", "king", "distance", "n_attackers", "n_defenders"]
LINE_MOVERS = {chess.BISHOP: "diag", chess.ROOK: "line", chess.QUEEN: "both"}


def _on_line(pt: int, p: int, s: int) -> bool:
    kind = LINE_MOVERS.get(pt)
    if kind is None:
        return False
    df, dr = chess.square_file(s) - chess.square_file(p), chess.square_rank(s) - chess.square_rank(p)
    diag = abs(df) == abs(dr) and df != 0; straight = (df == 0) != (dr == 0)
    return (kind == "diag" and diag) or (kind == "line" and straight) or (kind == "both" and (diag or straight))


def features(b: chess.Board, s: int) -> tuple:
    """-> (squares of the pieces, feature matrix) for every piece on the board, against the contested square s."""
    occ = b.piece_at(s)
    side = occ.color if occ else b.turn
    direct = {c: set(b.attackers(c, s)) for c in (chess.WHITE, chess.BLACK)}
    participants = direct[chess.WHITE] | direct[chess.BLACK] | ({s} if occ else set())
    sqs = []; rows = []
    for p, pc in b.piece_map().items():
        between = [x for x in chess.SquareSet(chess.between(p, s)) if b.piece_at(x) is not None]
        on_line = _on_line(pc.piece_type, p, s)
        is_direct = p in direct[pc.color]
        one = False
        if not is_direct and p != s:
            for t in b.attacks(p):
                if t == s or (b.piece_at(t) is not None and b.piece_at(t).color == pc.color):
                    continue
                nb = b.copy(stack=False); nb.remove_piece_at(p); nb.set_piece_at(t, pc)
                if s in nb.attacks(t):
                    one = True; break
        hits = any(q in participants and q != p and b.piece_at(q).color != pc.color for q in b.attacks(p))
        backs = any(q in participants and q != p and b.piece_at(q).color == pc.color for q in b.attacks(p))
        kinds = [pc.piece_type == k for k in (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING)]
        x1 = on_line and len(between) == 1 and p != s
        rows.append([p == s, pc.color == side, is_direct, x1, x1 and between[0] in participants, on_line and len(between) == 2 and p != s, one, hits, backs,
                     b.is_pinned(pc.color, p), *kinds, chess.square_distance(p, s) / 7.0, len(direct[not side]) / 4.0, len(direct[side]) / 4.0])
        sqs.append(p)
    return sqs, np.array(rows, dtype=float)


def dataset(exchanges: list) -> tuple:
    X = []; y = []; group = []
    for k, e in enumerate(exchanges):
        b = chess.Board(e["fen"]); s = e["squares"][0]
        members = {chess.parse_square(q) for q in e["pieces"]}
        sqs, F = features(b, s)
        X.append(F); y.extend(q in members for q in sqs); group.extend([k] * len(sqs))
    return np.vstack(X), np.array(y), np.array(group)


def rule_frames(b: chess.Board, s: int) -> dict:
    """Frames picked without learning: the square's direct attackers and defenders (with the occupant); the same with
    the x-rays behind them; the old contest rule (the attackers by worth, then the defenders, four pieces at most)."""
    from .chess_domain import contests
    direct = {s} | set(b.attackers(chess.WHITE, s)) | set(b.attackers(chess.BLACK, s))
    sqs, F = features(b, s)
    xr = direct | {q for q, f in zip(sqs, F) if f[FEATURES.index("xray1")] or f[FEATURES.index("xray2")]}
    c = next((set(ids) for _, _, ids, _ in contests(b) if ids[0] == s), {s})
    return dict(direct=direct, direct_xray=xr, contest4=c)


class Prior:
    def __init__(self, model, threshold: float = 0.5):
        self.model = model; self.threshold = threshold

    def frame(self, b: chess.Board, s: int) -> list:
        """The squares of the pieces the prior puts in the frame of s (the occupant always)."""
        sqs, F = features(b, s)
        pr = self.model.predict_proba(F)[:, 1]
        return [q for q, p in zip(sqs, pr) if p >= self.threshold or q == s]

    @staticmethod
    def load(path=None) -> "Prior":
        return pickle.load(open(path or paths.derived("frame_prior.pkl"), "rb"))


def frame_scores(pred: list, true: list) -> dict:
    tp = sum(len(p & t) for p, t in zip(pred, true)); fp = sum(len(p - t) for p, t in zip(pred, true)); fn = sum(len(t - p) for p, t in zip(pred, true))
    return dict(precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn), exact=sum(p == t for p, t in zip(pred, true)) / max(1, len(true)),
                size_err=float(np.mean([abs(len(p) - len(t)) for p, t in zip(pred, true)])))


def main(argv=None):
    from sklearn.linear_model import LogisticRegression
    ap = argparse.ArgumentParser(); ap.add_argument("--min-captures", type=int, default=1)
    a = ap.parse_args(argv)
    ex = [json.loads(l) for l in open(paths.derived("exchanges.jsonl"))]
    ex = [e for e in ex if len(e["captures"]) >= a.min_captures]
    train = [e for e in ex if e["game"] % 5 != 0]; test = [e for e in ex if e["game"] % 5 == 0]
    X, y, _ = dataset(train)
    model = LogisticRegression(max_iter=2000, class_weight=None).fit(X, y)
    prior = Prior(model)
    print(f"[frame_prior] trained on {len(train)} exchanges ({len(y)} pieces, {y.mean():.3f} in a frame); held out {len(test)} exchanges")
    print("OBS weights: " + ", ".join(f"{n} {w:+.2f}" for n, w in sorted(zip(FEATURES, model.coef_[0]), key=lambda t: -abs(t[1]))))
    true = []; preds = {"prior": [], "direct": [], "direct_xray": [], "contest4": []}; captured_found = {k: [0, 0] for k in preds}
    for e in test:
        b = chess.Board(e["fen"]); s = e["squares"][0]
        t = {chess.parse_square(q) for q in e["pieces"]}; true.append(t)
        cap = {chess.parse_square(q) for q, d in e["pieces"].items() if d.get("role") in ("captured", "capturer")}
        got = dict(prior=set(prior.frame(b, s)), **rule_frames(b, s))
        for k, v in got.items():
            preds[k].append(v); captured_found[k][0] += len(cap & v); captured_found[k][1] += len(cap)
    report = {}
    for k, v in preds.items():
        sc = frame_scores(v, true); sc["captors_found"] = captured_found[k][0] / max(1, captured_found[k][1]); report[k] = sc
        print(f"OBS {k:12s} precision {sc['precision']:.3f}  recall {sc['recall']:.3f}  exact frame {sc['exact']:.3f}  "
              f"size error {sc['size_err']:.2f}  captors and captured found {sc['captors_found']:.3f}")
    shown = 0
    for e, pr, t in zip(test, preds["prior"], true):                      # worked examples of the prior's mistakes
        if pr == t or shown >= 2 or len(t) < 4:
            continue
        shown += 1; b = chess.Board(e["fen"])
        print(f"\nEXAMPLE held-out game {e['game']} ply {e['start_ply']}: {' '.join(u for _, u, _ in e['captures'])} on {chess.square_name(e['squares'][0])}")
        print(b.unicode(borders=False, empty_square='·'))
        nm = lambda qs: ", ".join(f"{b.piece_at(q).symbol()}{chess.square_name(q)}" for q in sorted(qs)) or "none"
        print(f"   in the exchange but missed by the prior: {nm(t - pr)}; added by the prior but not in it: {nm(pr - t)}")
        for q in sorted((t - pr) | (pr - t)):
            d = e["pieces"].get(chess.square_name(q), {})
            print(f"   {b.piece_at(q).symbol()}{chess.square_name(q)}: label {d.get('link', 'not in the frame')} {'direct' if d.get('direct') else ('x-ray/later' if d else '')} {d.get('role', '')}")
    pickle.dump(prior, open(paths.derived("frame_prior.pkl"), "wb"))
    json.dump(report, open(paths.derived("frame_prior_report.json"), "w"), indent=1)
    print(f"[frame_prior] -> {paths.derived('frame_prior.pkl')}")


if __name__ == "__main__":
    main()
