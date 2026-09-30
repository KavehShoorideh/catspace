"""THE TACTICAL SEARCH POLICY, LEARNED (Kaveh 2026-09-29: "why limit to 4 moves? deciding whether to continue or not
should be the job of a tactical policy"; after a threshold rule on the king watches marked nearly every candidate:
"build the learned policy").

Which candidate deserves more search? The deeper search is the teacher (as in expert iteration): for every candidate
of a position, a SMALL zone search (4,000 nodes a depth, 4-6-8 plies) and a LARGE one (40,000 a depth, to 10 plies),
each on its own table so the labels are independent. The label: more search CHANGED the candidate -- its value by a
pawn or more, or its standing against the position's best (it overtakes the best, or the best falls). The features
are what the cheap phase knows: the move (kind, piece, what it takes, check), its values by depth and their swing,
the nodes it used and whether it ran out, its gap to the best, and whether it changes a king watch.

In play (frame_player BUDGET_POLICY = "learned"): phase 1, every candidate at the small budget; phase 2, the rest of
the same total shared by the predicted chance that more search changes it, the likely ones allowed to 10 plies.

    python -m catspace.incubator.analyst.search_policy collect --n 40 --offset 800     # training puzzles (fresh rows)
    python -m catspace.incubator.analyst.search_policy train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import chess

from catspace.io import paths

SMALL, LARGE = 4000, 40000
PHASE1_DEPTHS = (4, 6)                                              # the cheap phase in play (bug 2026-09-30: phase 1 to 8 plies cost as
MASK = ("v8", "nodes8", "swing68")                                  # much as the whole fixed search, leaving phase 2 nothing at equal
                                                                    # compute); the model is trained with the 8-ply features masked to match
FEATURES = ["capture", "check", "promotion", "quiet", "piece", "captured", "v4", "v6", "v8", "swing46", "swing68",
            "depth", "out_of_budget", "nodes4", "nodes6", "nodes8", "gap_to_best", "is_best", "n_candidates",
            "changes_watch"]


def _hist_at(hist: list, d: int):
    for dd, v, n in hist:
        if dd == d:
            return v, n
    return None, 0


def features(b: chess.Board, m: chess.Move, hist: list, value: float, depth: int, best: float, n: int, changes_watch: bool) -> list:
    v4, n4 = _hist_at(hist, 4); v6, n6 = _hist_at(hist, 6); v8, n8 = _hist_at(hist, 8)
    f = lambda x: -99.0 if x is None else max(-50.0, min(50.0, x))
    cap = b.piece_at(m.to_square)
    return [b.is_capture(m), b.gives_check(m), bool(m.promotion), not (b.is_capture(m) or b.gives_check(m) or m.promotion),
            b.piece_type_at(m.from_square), 0 if cap is None else cap.piece_type, f(v4), f(v6), f(v8),
            0.0 if v4 is None or v6 is None else f(v6 - v4), 0.0 if v6 is None or v8 is None else f(v8 - v6),
            depth, any(v is None for _, v, _ in hist), n4, n6, n8, f(best - value), value >= best, n, changes_watch]


def _candidates(b: chess.Board) -> list:
    import catspace.incubator.analyst.frame_player as FP
    stand = FP.passed(b)
    before, book = FP.joint_threats_book(stand, FP.square_frame) if stand is not None else ((0.0, None, []), None)
    one = FP._one_ply_values(b, before, book)
    ranked = sorted(one, key=lambda u: -one[u]); top1 = max(one.values())
    keep = set(ranked[:FP.CANDIDATES]) | {u for u in one if one[u] == top1} | {m.uci() for m in b.legal_moves if b.is_capture(m) or b.gives_check(m) or m.promotion}
    return [m for m in b.legal_moves if m.uci() in keep]


def _position_rows(b: chess.Board, pid: str, step: int) -> list:
    from .zone_search import ZoneSearch
    from .king_watch import _signature, watch
    from .fates import clear_caches
    clear_caches()
    moves = _candidates(b)
    if len(moves) < 2:
        return []
    small = ZoneSearch(SMALL); gs = small.rank(b, moves)
    best_s = max(v for v, _, _ in gs.values())
    before = {c: _signature(watch(b, c, operators=False), 3) for c in (chess.WHITE, chess.BLACK) if b.king(c) is not None}
    large = {}
    for m in moves:                                                   # each on its own table: independent labels
        zl = ZoneSearch(LARGE); large[m] = zl.rank(b, [m], depths=(4, 6, 8, 10))[m]
    best_l = max(v for v, _, _ in large.values())
    rows = []
    for m in moves:
        vs, _, ds = gs[m]; vl, _, dl = large[m]
        nb = b.copy(stack=False); nb.push(m)
        cw = b.gives_check(m) or any(nb.king(c) is None or _signature(watch(nb, c, operators=False), 3) != sig for c, sig in before.items())
        standing = (vs >= best_s) != (vl >= best_l)
        rows.append(dict(pid=pid, step=step, fen=b.fen(), move=m.uci(), x=features(b, m, small.history.get(m, []), vs, ds, best_s, len(moves), cw),
                         vs=vs, vl=vl, ds=ds, dl=dl, changed=abs(vl - vs) >= 1.0 or standing, standing=standing))
    return rows


def _puzzle_rows(row: dict) -> list:
    b = chess.Board(row["FEN"]); mv = row["Moves"].split(); b.push_uci(mv[0]); sol = mv[1:]
    out = []
    for i in range(0, len(sol), 2):
        if b.is_game_over():
            break
        out += _position_rows(b, row["PuzzleId"], i // 2)
        b.push_uci(sol[i])
        if i + 1 < len(sol):
            b.push_uci(sol[i + 1])
    return out


def collect(n: int = 40, offset: int = 800, workers: int = 10, log=print) -> str:
    import multiprocessing as mp
    from .frame_puzzles import fresh_rows
    t0 = time.perf_counter()
    rows = fresh_rows(n, 997, offset)
    out = paths.derived(f"search_policy_{n}_{offset}.jsonl")
    k = 0
    with mp.get_context("fork").Pool(workers) as pool, open(out, "w") as fh:
        for rs in pool.imap_unordered(_puzzle_rows, rows, chunksize=1):
            for r in rs:
                fh.write(json.dumps(r) + "\n"); k += 1
    log(f"[search_policy] {len(rows)} puzzles, {k} candidates -> {out} ({time.perf_counter() - t0:.0f}s)")
    return str(out)


def train(paths_in: list, label: str = "changed", log=print):
    import pickle
    import numpy as np
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    data = [json.loads(l) for p in paths_in for l in open(p)]
    test = np.array([int(hashlib.sha1(("sp" + d["pid"]).encode()).hexdigest()[:8], 16) % 4 == 0 for d in data])
    X = np.array([d["x"] for d in data], dtype=float); y = np.array([d["changed"] for d in data])
    for name in MASK:                                                 # what phase 1 in play cannot know
        X[:, FEATURES.index(name)] = -99.0 if name == "v8" else 0.0
    ys = np.array([d["standing"] for d in data])
    log(f"[search_policy] {len(data)} candidates from {len({d['pid'] for d in data})} puzzles; more search changes the candidate in "
        f"{y.mean():.3f} (its standing against the best in {ys.mean():.3f}); train {int((~test).sum())}, held out {int(test.sum())}")
    if label == "standing":                                           # the decision-relevant label: more search moves its standing
        y = ys
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced").fit(X[~test], y[~test])
    p = model.predict_proba(X[test])[:, 1]
    base = {name: X[test][:, FEATURES.index(name)] for name in ("swing46", "out_of_budget", "gap_to_best", "check", "changes_watch")}
    log(f"OBS held-out AUC for '{'its standing changes' if label == 'standing' else 'more search changes it'}': model {roc_auc_score(y[test], p):.3f}; single features: "
        + ", ".join(f"{k} {roc_auc_score(y[test], np.abs(v) if k == 'swing46' else (-v if k == 'gap_to_best' else v)):.3f}" for k, v in base.items()))
    if ys[test].sum():
        log(f"OBS held-out AUC for 'its standing changes': model {roc_auc_score(ys[test], p):.3f}")
    pickle.dump(model, open(paths.derived("search_policy.pkl"), "wb"))
    return model


_MODEL = {}


def allocate(b: chess.Board, moves: list, zs, weight_floor: float = 0.3, deep: int = 10) -> dict:
    """Phase 1 on `zs` (the small budget, all candidates to 8 plies), then the model's chance that more search changes
    each candidate -> alloc for phase 2: weight = floor + chance; the likelier half may go to `deep` plies."""
    import pickle
    import numpy as np
    from .king_watch import _signature, watch
    if "m" not in _MODEL:
        _MODEL["m"] = pickle.load(open(paths.derived("search_policy.pkl"), "rb"))
    got = zs.rank(b, moves, depths=PHASE1_DEPTHS)
    best = max(v for v, _, _ in got.values())
    before = {c: _signature(watch(b, c, operators=False), 3) for c in (chess.WHITE, chess.BLACK) if b.king(c) is not None}
    X = []
    for m in moves:
        nb = b.copy(stack=False); nb.push(m)
        cw = b.gives_check(m) or any(nb.king(c) is None or _signature(watch(nb, c, operators=False), 3) != sig for c, sig in before.items())
        v, _, d = got[m]
        X.append(features(b, m, zs.history.get(m, []), v, d, best, len(moves), cw))
    X = np.array(X, dtype=float)
    for name in MASK:
        X[:, FEATURES.index(name)] = -99.0 if name == "v8" else 0.0
    p = _MODEL["m"].predict_proba(X)[:, 1]
    # what matters is a change that can move the DECISION (2.7% of candidates in the first data -- too rare to learn from
    # 40 puzzles): the chance of change times closeness to the best (a pawn behind halves it)
    close = np.array([0.5 ** max(0.0, best - got[m][0]) for m in moves])
    voi = p * close
    cut = float(np.median(voi))
    return {m: (weight_floor + float(v), deep if v >= cut else 8) for m, v in zip(moves, voi)}


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--n", type=int, default=40); c.add_argument("--offset", type=int, default=800)
    c.add_argument("--workers", type=int, default=10)
    t = sub.add_parser("train"); t.add_argument("--data", nargs="+", default=None); t.add_argument("--label", default="changed", choices=["changed", "standing"])
    a = ap.parse_args(argv)
    if a.cmd == "collect":
        collect(a.n, a.offset, a.workers)
    else:
        train(a.data or [str(paths.derived("search_policy_40_800.jsonl"))], a.label)


if __name__ == "__main__":
    main()
