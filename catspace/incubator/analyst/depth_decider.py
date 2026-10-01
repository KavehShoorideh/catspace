"""THE DEPTH DECIDER -- the tactical policy's stop-or-continue call, learned from the depth ladder (Kaveh 2026-09-29:
"deciding whether to continue or not should be the job of a tactical policy"; 2026-09-30: "go very deep on a lot of
positions, see which depth changed the result, use that as training labels, then predict what the next depth should
be" -- deep labels first, reinforcement learning after).

After each rung (the zone search deepening all candidates together: 4, 6, 8, ... plies) the decider reads what the
search knows so far -- the top set and whether it moved since the last rung, the gap to the second, the top value's
swing, the candidates out of budget, the nodes and their growth -- and the board's material and passed pawns, and
gives the chance that going on changes the decision. The LABEL (depth_ladder): the decision at this rung shares no
move with the decision at the ladder's deepest rung. In play it is the zone search's `stop` callback
(`TacticalSpec.stop`): stop when the chance is below the planner's threshold.

Train/play gap, kept small on purpose: the ladder deepens with no margin pruning and a large budget, the player prunes
candidates more than `margin` behind and has a smaller budget -- so no feature reads the candidates far behind (the
gap to the second is capped at the margin) and none counts the candidates still deepening.

    python -m catspace.incubator.analyst.depth_decider train          # the ladder files -> the model; held-out AUC by puzzle
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import chess

from catspace.io import paths

FEATURES = ["depth", "n_candidates", "top_size", "changed_last", "changes_before", "gap", "swing", "top", "mate",
            "out_of_budget", "log_nodes", "growth", "pieces", "pawns", "own_material", "opp_material", "own_passed",
            "opp_passed", "in_check"]
GAP_CAP = 3.0                                                       # the player's margin: nothing further behind is deepened in play
MODEL_NAME = "depth_decider.pkl"


def _values_at(history: dict, d: int) -> dict:
    """Each candidate at its last completed depth up to d (as rank() and depth_ladder.decisions read it); an empty
    history is a move that mates at once."""
    out = {}
    for u, h in history.items():
        if not h:
            out[u] = 1e6; continue
        done = [v for dd, v, _ in h if dd <= d and v is not None]
        if done:
            out[u] = done[-1]
    return out


def _top(vals: dict) -> set:
    if not vals:
        return set()
    t = max(vals.values())
    return {u for u, v in vals.items() if v == t}


def _passed(b: chess.Board, colour: chess.Color) -> int:
    n = 0
    for sq in b.pieces(chess.PAWN, colour):
        f, r = chess.square_file(sq), chess.square_rank(sq)
        ahead = range(r + 1, 8) if colour == chess.WHITE else range(0, r)
        if not any(b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not colour)
                   for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead):
            n += 1
    return n


def _material(b: chess.Board, colour: chess.Color) -> int:
    return sum(len(b.pieces(p, colour)) * v for p, v in ((chess.KNIGHT, 3), (chess.BISHOP, 3), (chess.ROOK, 5), (chess.QUEEN, 9)))


def features(b: chess.Board, d: int, history: dict, rungs: tuple) -> list:
    """What the search and the board say after rung d (history: candidate -> [(depth, value or None, nodes)])."""
    earlier = [r for r in rungs if r < d]
    vals = _values_at(history, d)
    top = _top(vals)
    tops = [_top(_values_at(history, r)) for r in earlier] + [top]
    changed_last = bool(earlier) and not (tops[-1] & tops[-2])
    changes_before = sum(1 for x, y in zip(tops[:-2], tops[1:-1]) if not (x & y))
    t = max(vals.values()) if vals else 0.0
    rest = [v for v in vals.values() if v < t]
    gap = min(GAP_CAP, t - max(rest)) if rest else GAP_CAP
    prev = _values_at(history, earlier[-1]) if earlier else vals
    tp = max(prev.values()) if prev else t
    clip = lambda x: max(-50.0, min(50.0, x))
    at = [(v, n) for h in history.values() for dd, v, n in h if dd == d]
    before = [n for h in history.values() for dd, v, n in h if earlier and dd == earlier[-1]]
    nodes = sum(n for _, n in at)
    me = b.turn
    return [d, len(history), len(top), changed_last, changes_before, gap, clip(t - tp) if abs(t) < 1e5 and abs(tp) < 1e5 else 0.0,
            clip(t), t >= 1e5, sum(1 for v, _ in at if v is None) / max(1, len(history)), math.log1p(nodes),
            math.log1p(nodes) - math.log1p(sum(before)) if before else 0.0,
            len(b.piece_map()), len(b.pieces(chess.PAWN, chess.WHITE)) + len(b.pieces(chess.PAWN, chess.BLACK)),
            _material(b, me), _material(b, not me), _passed(b, me), _passed(b, not me), b.is_check()]


def rung_nodes(history: dict, rungs: tuple, d: int) -> list:
    """The nodes each rung up to d spent (all candidates)."""
    return [sum(n for h in history.values() for dd, _, n in h if dd == r) for r in rungs if r <= d]


def next_cost(spent: list, first_growth: float) -> float:
    """The next rung's expected nodes: this rung's nodes times its growth over the last one (the first rung has no growth
    of its own: the data's median)."""
    here = spent[-1]
    if len(spent) == 1:
        return here * first_growth
    return here * (here / spent[-2] if spent[-2] > 0 else 1.0)


def first_growth(paths_in: list) -> dict:
    """The median growth from the first rung to the second, per first rung, over ladder files."""
    import statistics as st
    g = {}
    for path in paths_in:
        for line in open(path):
            rec = json.loads(line)
            rungs = tuple(sorted({d for h in rec["history"].values() for d, _, _ in h}))
            if len(rungs) > 1:
                a, b2 = rung_nodes(rec["history"], rungs, rungs[1])
                if a > 0:
                    g.setdefault(rungs[0], []).append(b2 / a)
    return {d: st.median(v) for d, v in g.items()}


def label(shallow: set, deep: set, kind: str = "any") -> bool:
    """Going on changed the decision: "any" -- the top set differs at all (a tie resolving counts: a tied top is an
    undecided search); "disjoint" -- the deep top shares no move with this rung's (2026-09-30: this first label ignored
    ties, and the decider trained on it matched fixed depth -- a negative made by the label; docs/BUGS.md)."""
    return not (shallow & deep) if kind == "disjoint" else shallow != deep


def rows(path: str, kind: str = "any") -> list:
    """The ladder file -> one row per position and rung below the deepest: features, label, puzzle id."""
    from .depth_ladder import decisions
    out = []
    for line in open(path):
        rec = json.loads(line)
        dec = decisions(rec)
        if not dec:
            continue
        rungs = tuple(sorted(dec))
        deep = dec[rungs[-1]]
        b = chess.Board(rec["fen"])
        for d in rungs[:-1]:
            out.append(dict(pid=rec["pid"], d=d, x=features(b, d, rec["history"], rungs), y=label(dec[d], deep, kind),
                            src=path.rsplit("/", 1)[-1]))
    return out


def _test(pid: str) -> bool:
    return int(hashlib.sha1(("dd" + pid).encode()).hexdigest()[:8], 16) % 4 == 0


def train(paths_in: list, kind: str = "any", log=print):
    import pickle
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    data = [r for p in paths_in for r in rows(p, kind)]
    X = np.array([r["x"] for r in data], dtype=float); y = np.array([r["y"] for r in data])
    test = np.array([_test(r["pid"]) for r in data])
    src = np.array([r["src"] for r in data])
    log(f"[depth_decider] {len(data)} rows (position x rung) from {len({r['pid'] for r in data})} puzzles; going on changes the "
        f"decision ({kind}) in {y.mean():.3f}; train {int((~test).sum())}, held out {int(test.sum())} (split by puzzle)")
    for s in sorted(set(src)):
        k = src == s
        log(f"OBS {s}: {int(k.sum())} rows, label rate {y[k].mean():.3f}; by rung: "
            + ", ".join(f"{d}: {y[k & (X[:, 0] == d)].mean():.3f} of {int((k & (X[:, 0] == d)).sum())}" for d in sorted(set(X[k, 0].astype(int)))))
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, class_weight="balanced").fit(X[~test], y[~test])
    p = model.predict_proba(X[test])[:, 1]
    single = {k: X[test][:, FEATURES.index(k)] for k in ("changed_last", "top_size", "gap", "depth", "pieces", "out_of_budget")}
    sign = {"gap": -1, "depth": -1, "pieces": -1}
    if len(set(y[test])) > 1:
        log(f"OBS held-out AUC for 'going on changes the decision': model {roc_auc_score(y[test], p):.3f}; single features: "
            + ", ".join(f"{k} {roc_auc_score(y[test], sign.get(k, 1) * v):.3f}" for k, v in single.items()) + "; a constant 0.500")
        for s in sorted(set(src)):
            k = src[test] == s
            if k.sum() and len(set(y[test][k])) > 1:
                log(f"OBS held-out AUC within {s}: model {roc_auc_score(y[test][k], p[k]):.3f} ({int(k.sum())} rows, {int(y[test][k].sum())} positive)")
    # the models for play: refit on every row once the held-out gauge is printed (play's puzzles are disjoint from these)
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, class_weight="balanced").fit(X, y)
    calibrated = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3).fit(X, y)
    pickle.dump(dict(model=model, calibrated=calibrated, first_growth=first_growth(paths_in), features=FEATURES, label=kind,
                     data=[p.rsplit("/", 1)[-1] for p in paths_in]), open(paths.derived(MODEL_NAME), "wb"))
    log(f"[depth_decider] saved {paths.derived(MODEL_NAME)} (label {kind}, all {len(data)} rows)")
    return model


def replay(paths_in: list, thresholds=(0.7, 0.5, 0.3, 0.15), prices=(1e-4, 3e-5, 1e-5, 3e-6, 1e-6), folds: int = 4,
           kind: str = "any", log=print) -> None:
    """The policies replayed on the ladder itself: stopping the ladder's search after rung d gives exactly its values up
    to d (all candidates deepen together), so each policy's decision and nodes per position are read off the records.
    Fixed depths against the decider at several thresholds; the decider's chance for a position comes from a model
    that never saw its puzzle (k folds by puzzle). Per ladder file: the solution among the top / alone on top, and the
    mean nodes (up to the stopping rung) -- the strength-per-node frontier.
    Two rules: a THRESHOLD on the chance (one for every rung), and a PRICE on search (Kaveh 2026-08-12: "reward = win -
    lambda * search effort"): go on when the calibrated chance of a change is worth the next rung's expected nodes,
    chance >= price * (this rung's nodes x its growth over the last rung) -- a rung costing five times more needs five
    times the chance."""
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from .depth_ladder import decisions
    recs = [dict(json.loads(l), src=p.rsplit("/", 1)[-1]) for p in paths_in for l in open(p)]
    fold = lambda pid: int(hashlib.sha1(("dd" + pid).encode()).hexdigest()[:8], 16) % folds
    pos = []
    for rec in recs:
        dec = decisions(rec)
        if not dec:
            continue
        rungs = tuple(sorted(dec)); b = chess.Board(rec["fen"])
        nodes = {d: sum(n for h in rec["history"].values() for dd, _, n in h if dd <= d) for d in rungs}
        X = [features(b, d, rec["history"], rungs) for d in rungs[:-1]]
        y = [label(dec[d], dec[rungs[-1]], kind) for d in rungs[:-1]]
        pos.append(dict(rec=rec, dec=dec, rungs=rungs, nodes=nodes, X=X, y=y, f=fold(rec["pid"])))
    for k in range(folds):                                            # each position's chances from the model of the other folds
        tr = [p for p in pos if p["f"] != k]
        Xt = np.array([x for p in tr for x in p["X"]], dtype=float); yt = np.array([v for p in tr for v in p["y"]])
        m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, class_weight="balanced").fit(Xt, yt)
        mc = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3).fit(Xt, yt)     # calibrated: for the price
        for p in pos:
            if p["f"] == k and p["X"]:
                p["p"] = list(m.predict_proba(np.array(p["X"], dtype=float))[:, 1])
                p["pc"] = list(mc.predict_proba(np.array(p["X"], dtype=float))[:, 1])
    fg = first_growth(paths_in)

    def cost(p, i):
        return next_cost(rung_nodes(p["rec"]["history"], p["rungs"], p["rungs"][i]), fg.get(p["rungs"][0], 1.0))
    for src in sorted({p["rec"]["src"] for p in pos}):
        ps = [p for p in pos if p["rec"]["src"] == src]
        rungs = sorted({d for p in ps for d in p["rungs"]})

        def score(stop_at):
            among = alone = 0; nodes = []
            for p in ps:
                d = stop_at(p)
                top = p["dec"][d]
                among += p["rec"]["sol"] in top; alone += top == {p["rec"]["sol"]}; nodes.append(p["nodes"][d])
            return among, alone, sum(nodes) / len(nodes)
        log(f"OBS replay {src} ({len(ps)} positions; the solution among the top / alone on top / mean nodes up to the stop):")
        for d in rungs:
            a, s, n = score(lambda p, d=d: max(r for r in p["rungs"] if r <= d) if any(r <= d for r in p["rungs"]) else p["rungs"][0])
            log(f"   fixed {d:>2} plies: {a} / {s} / {n:,.0f}")
        for t in thresholds:
            def stop_at(p, t=t):
                for d, pr in zip(p["rungs"][:-1], p.get("p", [])):
                    if pr < t:
                        return d
                return p["rungs"][-1]
            a, s, n = score(stop_at)
            where = {}
            for p in ps:
                d = stop_at(p); where[d] = where.get(d, 0) + 1
            log(f"   decider {t:<4g}: {a} / {s} / {n:,.0f}  (stops at " + ", ".join(f"{d}: {where[d]}" for d in sorted(where)) + ")")
        for lam in prices:
            def stop_at(p, lam=lam):
                for i, (d, pr) in enumerate(zip(p["rungs"][:-1], p.get("pc", []))):
                    if pr < lam * cost(p, i):
                        return d
                return p["rungs"][-1]
            a, s, n = score(stop_at)
            where = {}
            for p in ps:
                d = stop_at(p); where[d] = where.get(d, 0) + 1
            log(f"   priced {lam:<6g}: {a} / {s} / {n:,.0f}  (stops at " + ", ".join(f"{d}: {where[d]}" for d in sorted(where)) + ")")


_MODEL = {}


def stopper(b: chess.Board, rungs: tuple, threshold: float = 0.5, price: float | None = None, log: list | None = None):
    """The zone search's `stop` callback for the position b: after rung d, stop unless the chance that going on changes
    the decision is at least `threshold` -- or, with a `price` on search (the planner's), unless the calibrated chance is
    at least price x the next rung's expected nodes."""
    import pickle
    import numpy as np
    if "m" not in _MODEL:
        saved = pickle.load(open(paths.derived(MODEL_NAME), "rb"))
        _MODEL["m"] = saved["model"]; _MODEL["c"] = saved.get("calibrated"); _MODEL["g"] = saved.get("first_growth", {})

    def stop(d, got, history) -> bool:
        if d >= max(rungs):
            return True
        h = {(m.uci() if isinstance(m, chess.Move) else m): v for m, v in history.items()}
        for m, (v, _, dd) in got.items():                               # a mate at once: never searched, no history
            if dd == 1 and v >= 1e5:
                h[m.uci() if isinstance(m, chess.Move) else m] = []
        model = _MODEL["m"] if price is None else _MODEL["c"]
        p = float(model.predict_proba(np.array([features(b, d, h, rungs)], dtype=float))[:, 1][0])
        if log is not None:
            log.append((d, round(p, 3)))
        if price is None:
            return p < threshold
        return p < price * next_cost(rung_nodes(h, rungs, d), _MODEL["g"].get(min(rungs), 1.0))
    return stop


def main(argv=None):
    import os
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "1")                     # small fits: OpenMP threads beside busy search workers spin at
    sys.stdout.reconfigure(line_buffering=True)                       # barriers (2026-09-30: minutes, no progress, at 52% CPU)
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("--data", nargs="+", default=["depth_ladder_misses.jsonl", "depth_ladder_150_700.jsonl"])
    t.add_argument("--label", default="any", choices=["any", "disjoint"])
    r = sub.add_parser("replay"); r.add_argument("--data", nargs="+", default=["depth_ladder_misses.jsonl", "depth_ladder_150_700.jsonl"])
    r.add_argument("--thresholds", type=float, nargs="+", default=[0.7, 0.5, 0.3, 0.15])
    r.add_argument("--prices", type=float, nargs="+", default=[1e-4, 3e-5, 1e-5, 3e-6, 1e-6])
    r.add_argument("--label", default="any", choices=["any", "disjoint"])
    a = ap.parse_args(argv)
    if a.cmd == "replay":
        return replay([str(paths.derived(p)) for p in a.data], tuple(a.thresholds), tuple(a.prices), kind=a.label)
    train([str(paths.derived(p)) for p in a.data], a.label)


if __name__ == "__main__":
    main()
