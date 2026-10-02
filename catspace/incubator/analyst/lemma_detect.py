"""THE HIGH-VALUE LEMMAS IN PLAY -- learned detectors (Kaveh 2026-10-01: high-value lemmas -- the pin, the back rank;
"start from a seed and ask what would have prevented this"; his picks: "lemmas in play", recognized by "learned
detectors" -- Stockfish only in training).

The FAMILIES are the groups of tactic_reasons (the 2,859 Lichess tactics grouped, without their themes, by HOW they
would have been prevented -- relations only, k-means k=16): the pin family (prevented by unpinning), the back-rank
family (by luft), ... Each family's LEMMA: its preventers (the conditions that must not hold) and its STAKE (the
median stake of its tactics, Stockfish's evaluation minus material, in pawns; a mate capped at 20).

In play there is no oracle: a DETECTOR reads a candidate move with our own relations only -- the same generic
relations as tactic_reasons, around the move as if it were a tactic's first move (its squares, the enemy pieces it then
hits, the enemy king): every non-king piece and every enemy reply, counted. Trained on the 2,859 tactics: the solution
move is a positive of its family, the other legal moves are 'none' (a multinomial logistic regression, 5 folds by
puzzle for the held-out numbers). A move's LEMMA VALUE = sum over families of P(family | move) x the family's stake;
its explanation = the most likely family's preventers ("a pin: unpinning would have saved it").

    python -m catspace.incubator.analyst.lemma_detect build      # families + every legal move's features (2,859 tactics)
    python -m catspace.incubator.analyst.lemma_detect train      # held-out ranks; the model for play
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pickle
import statistics as st
import time
from collections import Counter

import chess

from catspace.io import paths

SOURCE = "tactic_reasons_3000_200.jsonl"
FAMILIES = "tactic_families.json"
FEATURES = "lemma_detect_features.jsonl.gz"
MODEL = "lemma_detect.pkl"
K = 16
MATE_PAWNS = 20.0


def families(log=print) -> dict:
    """The relations-only preventer groups (tactic_reasons.groups at k=16, the same seed) with their lemmas."""
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import normalize
    from .tactic_reasons import _counts, _reason_only
    recs = [json.loads(l) for l in open(paths.derived(SOURCE))]
    recs = [r for r in recs if "skipped" not in r]
    pc = [_reason_only(_counts(r, False)) for r in recs]
    vocab = sorted({f for c in pc for f in c})
    raw = np.log1p(np.array([[c.get(f, 0) for f in vocab] for c in pc], dtype=float))
    lab = KMeans(K, n_init=10, random_state=0).fit_predict(normalize(raw))
    overall = raw.mean(0) + 1e-9
    fams = {}
    for g in sorted(set(lab)):
        m = lab == g
        lift = raw[m].mean(0) / overall; share = (raw[m] > 0).mean(0)
        top = [vocab[i] for i in np.argsort(-lift) if share[i] >= 0.4][:4]
        stakes = [min(r["stake"] / 100.0, MATE_PAWNS) for r, x in zip(recs, m) if x]
        th = Counter(t for r, x in zip(recs, m) if x for t in r["themes"])
        fams[int(g)] = dict(size=int(m.sum()), preventers=top, stake=st.median(stakes), themes=[t for t, _ in th.most_common(6)])
    out = dict(families=fams, label={r["pid"]: int(g) for r, g in zip(recs, lab)})
    json.dump(out, open(paths.derived(FAMILIES), "w"))
    for g, f in sorted(fams.items(), key=lambda kv: -kv[1]["size"]):
        log(f"   family {g:2d} ({f['size']:4d}; stake {f['stake']:.1f}): prevented by {', '.join(f['preventers'])} | themes {', '.join(f['themes'][:4])}")
    return out


def _key(b: chess.Board, m: chess.Move) -> dict:
    owner = b.turn; victim = not owner
    nb = b.copy(stack=False); nb.push(m)
    hits = {t for t in chess.SquareSet(nb.attacks_mask(m.to_square)) if b.piece_at(t) is not None and b.piece_at(t).color == victim}
    won = {m.to_square} if b.is_capture(m) else set()
    return dict(owner=owner, victim=victim, first=m, won=won, pieces={m.from_square} | hits | won, king=b.king(victim))


def move_features(b: chess.Board, m: chess.Move) -> dict:
    """Our own relations around the move as a tactic's first move (no oracle): counts over every non-king piece and
    every enemy reply (the enemy given the move), plus the move's own kind."""
    from .tactic_reasons import describe_defence, describe_piece
    key = _key(b, m); c = Counter()
    for sq, pc in b.piece_map().items():
        if pc.piece_type != chess.KING:
            c.update(describe_piece(b, sq, key))
    pb = b.copy(stack=False); pb.turn = key["victim"]; pb.ep_square = None
    if pb.is_valid():
        for r in pb.legal_moves:
            c.update(describe_defence(pb, r, key))
    pt = chess.piece_name(b.piece_type_at(m.from_square))
    c.update([f"move:{pt}"] + (["move:capture"] if b.is_capture(m) else []) + (["move:check"] if b.gives_check(m) else [])
             + (["move:promotion"] if m.promotion else []) + [f"move:hits_{len(key['pieces']) - 1}"])
    return dict(c)


def _position(args) -> list:
    pid, fen, sol, fam = args
    b = chess.Board(fen)
    return [dict(pid=pid, move=m.uci(), y=(fam if m.uci() == sol else -1), x=move_features(b, m)) for m in b.legal_moves]


def build(workers: int = 10, log=print) -> None:
    import multiprocessing as mp
    t0 = time.perf_counter()
    fam = families(log)
    recs = [json.loads(l) for l in open(paths.derived(SOURCE))]
    items = [(r["pid"], r["fen"], r["line"][0], fam["label"][r["pid"]]) for r in recs if "skipped" not in r]
    n = 0
    with mp.get_context("fork").Pool(workers) as pool, gzip.open(paths.derived(FEATURES), "wt") as fh:
        for rows in pool.imap_unordered(_position, items, chunksize=4):
            for r in rows:
                fh.write(json.dumps(r) + "\n"); n += 1
    log(f"[lemma_detect] {len(items)} tactics, {n} moves -> {paths.derived(FEATURES)} ({time.perf_counter() - t0:.0f}s)")


def _vec(rows, dv=None):
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    xs = [{k: float(np.log1p(v)) for k, v in r["x"].items()} for r in rows]
    if dv is None:
        dv = DictVectorizer(); X = dv.fit_transform(xs)
    else:
        X = dv.transform(xs)
    return X, dv


def _fit(X, y):
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced").fit(X, y)


def _value(model, X, stakes: dict):
    import numpy as np
    P = model.predict_proba(X)
    w = np.array([0.0 if c == -1 else stakes[int(c)] for c in model.classes_])
    return P @ w, P


def train(neg_per: int = 8, log=print) -> None:
    import numpy as np
    fam = json.load(open(paths.derived(FAMILIES)))
    stakes = {int(g): f["stake"] for g, f in fam["families"].items()}
    rows = [json.loads(l) for l in gzip.open(paths.derived(FEATURES), "rt")]
    by = {}
    for r in rows:
        by.setdefault(r["pid"], []).append(r)
    pids = sorted(by)
    fold = {p: int(hashlib.sha1(("ld" + p).encode()).hexdigest()[:8], 16) % 5 for p in pids}
    rng = np.random.default_rng(0)

    def sample(ps):                                                  # the solution + up to neg_per other moves per tactic
        out = []
        for p in ps:
            sol = [r for r in by[p] if r["y"] != -1]; neg = [r for r in by[p] if r["y"] == -1]
            out += sol + [neg[i] for i in rng.permutation(len(neg))[:neg_per]]
        return out
    ranks = {"lemma value": [], "P(any family)": [], "chance": []}
    for k in range(5):
        tr = [p for p in pids if fold[p] != k]; te = [p for p in pids if fold[p] == k]
        trs = sample(tr)
        X, dv = _vec(trs); model = _fit(X, [r["y"] for r in trs])
        for p in te:
            Xt, _ = _vec(by[p], dv)
            v, P = _value(model, Xt, stakes)
            anyf = 1.0 - P[:, list(model.classes_).index(-1)]
            i = next(j for j, r in enumerate(by[p]) if r["y"] != -1)
            for name, s in (("lemma value", v), ("P(any family)", anyf)):
                ranks[name].append(float((s > s[i]).sum() + ((s == s[i]).sum() + 1) / 2))
            ranks["chance"].append((len(by[p]) + 1) / 2)
    n = len(ranks["chance"])
    log(f"OBS lemma detectors, held out by puzzle (5 folds; {n} tactics, legal moves median {st.median(len(by[p]) for p in pids):g}): "
        "the solution's rank among all legal moves (mean; top-1 / top-3)")
    for name, rk in ranks.items():
        log(f"   {name:16s} mean {st.mean(rk):5.1f}  top-1 {sum(1 for x in rk if x <= 1):4d}  top-3 {sum(1 for x in rk if x <= 3):4d}")
    s_all = sample(pids)
    X, dv = _vec(s_all); model = _fit(X, [r["y"] for r in s_all])
    pickle.dump(dict(model=model, dv=dv, stakes=stakes, families=fam["families"]), open(paths.derived(MODEL), "wb"))
    log(f"[lemma_detect] model for play (all {len(pids)} tactics) -> {paths.derived(MODEL)}")


_M: dict = {}


def values(b: chess.Board) -> dict:
    """In play: every legal move's lemma value, its chance of starting any family's tactic, and its most likely family
    -> {uci: (value, P(any family), family or None)}."""
    import numpy as np
    if "m" not in _M:
        _M.update(pickle.load(open(paths.derived(MODEL), "rb")))
    moves = list(b.legal_moves)
    rows = [dict(x=move_features(b, m)) for m in moves]
    X, _ = _vec(rows, _M["dv"])
    v, P = _value(_M["model"], X, _M["stakes"])
    cls = list(_M["model"].classes_)
    none = cls.index(-1)
    out = {}
    for m, val, p in zip(moves, v, P):
        q = p.copy(); q[none] = -1.0; j = int(np.argmax(q))
        out[m.uci()] = (float(val), float(1.0 - p[none]), int(cls[j]))
    return out


def main(argv=None):
    import sys
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build"); b.add_argument("--workers", type=int, default=10)
    sub.add_parser("train")
    a = ap.parse_args(argv)
    if a.cmd == "build":
        build(a.workers)
    else:
        train()


if __name__ == "__main__":
    main()
