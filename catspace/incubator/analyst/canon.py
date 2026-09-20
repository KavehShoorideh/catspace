"""CANONICAL CONCEPTS (Kaveh 2026-09-19: "I want concepts to be canonical -- different runs must not produce
different concepts; rules-based when explaining them").

The vocabulary is fixed by the rules: the enumerated region relations (regions.py, slot-named as in root_tokens).
The SELECTION is deterministic: an L1 regularization path of a multinomial logistic model on the tablebase result
is a convex problem with a unique optimum, so the order in which relations enter is a property of the data, not
of a seed. Conjunctions of the leading singles join the vocabulary in a second pass (the composed concepts).
The concepts are then the selected relations in entry order, with their weights per result class -- the same list
from any run. Gauges: result accuracy beside the token network and the per-material majority; king-move ranking
beside chance; CROSS-RUN IDENTITY (two runs over shuffled rows) and BOOTSTRAP STABILITY (which concepts survive
resampling the data); the GAP to the network per material (where the vocabulary is missing something). The
explanation of two king moves is read off the canonical list in canonical order: the differing relations, and the
shortest prefix whose weighted sum flips the verdict."""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import chess
import numpy as np
from scipy import sparse

from .endgames import load as load_rows, split_of
from .root_tokens import feature_names, features_of, features_of_board
from .tokens import white_wdl

OUT = Path("data/derived/canon")
HELD = {"KRPvKP", "KQvKRP", "KBNvK"}
C_PATH = (0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0)


def build(rows, names, log=print):
    index = {n: i for i, n in enumerate(names)}
    t0 = time.perf_counter()
    X = np.stack([features_of(r, index) for r in rows]).astype(np.float32)
    keep = np.nonzero((X != 0).any(0))[0]                                  # relations that ever occur
    X = X[:, keep]; names = [names[i] for i in keep]
    y = np.array([white_wdl(r["wdl"], chess.Board(r["fen"]).turn) + 1 for r in rows])
    split = np.array([split_of(r, HELD) for r in rows]); mat = np.array([r["material"] for r in rows])
    log(f"[canon] {len(rows)} rows, {len(names)} relations that occur, built in {time.perf_counter() - t0:.0f}s")
    return X, y, split, mat, names


def add_conjunctions(X, names, top: list[int]):
    """Pairwise ANDs over the leading singles (count features taken as > 0)."""
    B = (X[:, top] > 0).astype(np.float32)
    cols, cn = [], []
    for i in range(len(top)):
        for j in range(i + 1, len(top)):
            c = B[:, i] * B[:, j]
            if 0.005 < c.mean() < 0.995:
                cols.append(c); cn.append(f"({names[top[i]]}) AND ({names[top[j]]})")
    if not cols:
        return X, names
    return np.concatenate([X, np.stack(cols, 1)], 1), names + cn


def path(X, y, names, tr, va, log=print, max_iter: int = 200):
    """The L1 path: the entry order of the relations and the accuracy at each C on the validation rows."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler(with_mean=False).fit(X[tr]); Xs = sparse.csr_matrix(sc.transform(X))
    order, seen, best = [], set(), None
    for C in C_PATH:
        clf = LogisticRegression(penalty="l1", solver="saga", C=C, max_iter=max_iter, tol=1e-3, random_state=0).fit(Xs[tr], y[tr])
        nz = np.nonzero(np.abs(clf.coef_).sum(0) > 1e-6)[0]
        new = [int(i) for i in nz if int(i) not in seen]
        # within one C, order the newcomers by their weight norm (deterministic given the solution)
        new.sort(key=lambda i: -float(np.abs(clf.coef_[:, i]).sum()))
        order += new; seen.update(new)
        acc = float(clf.score(Xs[va], y[va]))
        log(f"[canon] C {C:g}: {len(nz)} relations, val acc {acc:.3f}; new: " + "; ".join(names[i] for i in new[:5]) + (" ..." if len(new) > 5 else ""))
        if best is None or acc > best[0] + 0.002:
            best = (acc, C, clf)
    return order, best, sc


def fit(log=print, bootstrap: int = 5, conj_top: int = 30):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = load_rows(); names0 = feature_names()
    X, y, split, mat, names = build(rows, names0, log)
    tr, va, te, hd = (np.nonzero(split == s)[0] for s in ("train", "val", "test", "held"))
    log("[canon] pass 1: singles")
    order1, best1, _ = path(X, y, names, tr, va, log)
    top = order1[:conj_top]
    X2, names2 = add_conjunctions(X, names, top)
    log(f"[canon] pass 2: singles + {len(names2) - len(names)} conjunctions of the leading {len(top)}")
    order, best, sc = path(X2, y, names2, tr, va, log)
    acc, C, clf = best
    Xs = sparse.csr_matrix(sc.transform(X2))
    maj = {m: int(np.bincount(y[tr][mat[tr] == m], minlength=3).argmax()) for m in np.unique(mat[tr])}
    res = {}
    for name, ids in (("test", te), ("held", hd)):
        pred = clf.predict(Xs[ids])
        res[name] = dict(n=len(ids), acc=float((pred == y[ids]).mean()), majority=float(np.mean([maj.get(m, 1) == t for m, t in zip(mat[ids], y[ids])])),
                         own_majority=float(np.mean([np.bincount(y[ids][mat[ids] == m], minlength=3).max() / (mat[ids] == m).sum() for m in np.unique(mat[ids])])),
                         by_material={m: float((pred[mat[ids] == m] == y[ids][mat[ids] == m]).mean()) for m in np.unique(mat[ids])})
        log(f"[canon] {name}: acc {res[name]['acc']:.3f} (majority {res[name]['majority']:.3f}, own {res[name]['own_majority']:.3f}) at C {C:g} with {int((np.abs(clf.coef_).sum(0) > 1e-6).sum())} relations")
    # the gap to the token network, per material
    try:
        net = json.load(open("data/derived/checkpoints/tokens_sib/final.json"))
        for name in ("test", "held"):
            gaps = {m: (net[name]["by_material"][m]["acc"] - res[name]["by_material"][m]) for m in res[name]["by_material"] if m in net[name]["by_material"]}
            log(f"[canon] gap to the network on {name} (network minus canonical): " + "; ".join(f"{m} {g:+.2f}" for m, g in sorted(gaps.items(), key=lambda kv: -kv[1])[:8]))
            res[name]["gap_to_network"] = gaps
    except FileNotFoundError:
        pass
    # the canonical list: entry order with weights per class (loss, draw, win in White's view)
    W = clf.coef_ / sc.scale_[None, :]                                      # back to the raw feature scale
    concepts = [dict(rank=k, name=names2[i], w_loss=float(W[0, i]), w_draw=float(W[1, i]), w_win=float(W[2, i]), on=float((X2[:, i] > 0).mean())) for k, i in enumerate(order) if np.abs(W[:, i]).sum() > 1e-6]
    log(f"[canon] the canonical concepts, in entry order ({len(concepts)}):")
    for c in concepts[:25]:
        log(f"  {c['rank']:3d} {c['name'][:70]:<70} win {c['w_win']:+.2f} draw {c['w_draw']:+.2f} loss {c['w_loss']:+.2f} (on {c['on']:.2f})")
    # CROSS-RUN IDENTITY: the same fit over shuffled rows must give the same list
    perm = np.random.default_rng(7).permutation(len(tr))
    order_b, best_b, _ = path(X2, y, names2, tr[perm], va, log=lambda *a, **k: None)
    same = sum(a == b for a, b in zip(order, order_b)); 
    log(f"[canon] cross-run identity (rows shuffled): first {min(len(order), len(order_b))} entries agree at {same}/{min(len(order), len(order_b))} positions; sets equal: {set(order) == set(order_b)}")
    # BOOTSTRAP STABILITY: which of the leading concepts survive resampling the data
    survive = Counter(); k_top = 20
    for bi in range(bootstrap):
        rs = np.random.default_rng(100 + bi).choice(tr, len(tr), replace=True)
        o_b, _, _ = path(X2, y, names2, rs, va, log=lambda *a, **k: None)
        for i in o_b[:k_top]:
            survive[i] += 1
    stab = [(names2[i], survive[i] / bootstrap) for i in order[:k_top]]
    log(f"[canon] bootstrap stability of the leading {k_top} (share of {bootstrap} resamples in which each is again among the leading {k_top}): " + "; ".join(f"{n[:40]} {s:.1f}" for n, s in stab[:12]))
    json.dump(dict(C=C, concepts=concepts, results=res, order_names=[names2[i] for i in order], identity=dict(agree=same, sets_equal=bool(set(order) == set(order_b))), stability=stab), open(OUT / "canon.json", "w"), indent=1)
    import pickle
    pickle.dump(dict(clf=clf, scaler=sc, names=names2, base_names=names, kept=[n for n in names], top=top, names0=names0), open(OUT / "model.pkl", "wb"))
    return clf, sc, names2


def _features_full(board: chess.Board, m) -> np.ndarray:
    """A board -> the model's feature vector (base relations that occur, plus the conjunctions)."""
    index0 = {n: i for i, n in enumerate(m["names0"])}
    x0 = features_of_board(board, index0)
    base_idx = [index0[n] for n in m["base_names"]]
    x = x0[base_idx]
    top = m["top"]
    B = (x[top] > 0).astype(np.float32)
    conj = []
    names = m["names"]; nb = len(m["base_names"])
    for cn in names[nb:]:
        a, b = cn[1:-1].split(") AND (")
        conj.append(float((x[m["base_names"].index(a)] > 0) and (x[m["base_names"].index(b)] > 0)))
    return np.concatenate([x, np.array(conj, np.float32)]) if conj else x


def value_of(board: chess.Board, m) -> tuple[float, np.ndarray]:
    x = _features_full(board, m)
    p = m["clf"].predict_proba(m["scaler"].transform(x[None]))[0]
    return float(p[2] - p[0]), x                                            # White's view


def siblings(log=print, max_pairs: int = 3000):
    import pickle
    m = pickle.load(open(OUT / "model.pkl", "rb"))
    rows = load_rows()
    for sp in ("test", "held"):
        ids = [i for i, r in enumerate(rows) if r.get("siblings") and split_of(r, HELD) == sp]
        random.Random(0).shuffle(ids); ids = ids[:max_pairs]
        n = ok = 0
        for i in ids:
            r = rows[i]; b = chess.Board(r["fen"]); sib = r["siblings"]
            best = max(sib, key=lambda u: sib[u]); worse = [u for u in sib if np.sign(sib[u]) < np.sign(sib[best])]
            if not worse:
                continue
            vals = []
            for u in (best, random.Random(i).choice(worse)):
                nb = b.copy(stack=False); nb.push(chess.Move.from_uci(u)); v, _ = value_of(nb, m); vals.append(v * (1 if b.turn == chess.WHITE else -1))
            n += 1; ok += vals[0] > vals[1]
        log(f"[canon] siblings {sp}: {n} pairs; the canonical model ranks the better king move above the worse in {100 * ok / max(1, n):.1f}% (50% = chance)")


def explain(fen: str, log=print):
    import pickle
    import chess.syzygy
    m = pickle.load(open(OUT / "model.pkl", "rb")); names = m["names"]
    W = m["clf"].coef_ / m["scaler"].scale_[None, :]
    order = {n: k for k, n in enumerate(json.load(open(OUT / "canon.json"))["order_names"])}
    tb = chess.syzygy.open_tablebase("assets/tablebases/syzygy")
    b = chess.Board(fen); ksq = b.king(b.turn); sign = 1 if b.turn == chess.WHITE else -1
    kids = []
    for mv in b.legal_moves:
        if mv.from_square != ksq:
            continue
        nb = b.copy(stack=False); nb.push(mv); w = -tb.probe_wdl(nb) if not nb.is_game_over() else 0
        v, x = value_of(nb, m); kids.append((b.san(mv), w, v * sign, x))
    tb.close()
    log(f"[explain] {fen}: " + ", ".join(f"{s} {'wins' if w > 0 else 'draws' if w == 0 else 'loses'} (model {v:+.2f})" for s, w, v, _ in kids))
    kids.sort(key=lambda k: -k[1])
    good, bad = kids[0], kids[-1]
    if np.sign(good[1]) == np.sign(bad[1]):
        log("[explain] all king moves lead to the same result"); return
    d = good[3] - bad[3]
    contrib = (W[2] - W[0]) * d * sign                                       # the shift of (win - loss) toward the good move, per relation
    idx = [i for i in np.nonzero(d != 0)[0] if abs(contrib[i]) > 1e-9]
    idx.sort(key=lambda i: order.get(names[i], 10 ** 6))                       # CANONICAL ORDER
    log(f"[explain] {good[0]} over {bad[0]}: {len(idx)} relations differ; in canonical order, with their shift toward {good[0]}:")
    total = float(contrib[idx].sum()); run = 0.0; prefix = []
    for i in idx[:12]:
        run += contrib[i]; prefix.append(i)
        log(f"    #{order.get(names[i], -1):3d} {names[i][:80]:<80} {int(bad[3][i])} -> {int(good[3][i])}   shift {contrib[i]:+.2f}  (running {run:+.2f} of {total:+.2f})")
    k = next((j + 1 for j in range(len(idx)) if contrib[idx[:j + 1]].sum() >= 0.5 * total and contrib[idx[:j + 1]].sum() > 0), len(idx))
    log(f"[explain] the shortest canonical prefix carrying half the shift: {k} relation(s): " + "; ".join(names[i][:60] for i in idx[:k]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "siblings", "explain"]); ap.add_argument("--fen", default="4k3/1R6/4K3/4P3/8/8/8/5r2 b - - 0 58")
    ap.add_argument("--bootstrap", type=int, default=5)
    a = ap.parse_args(argv)
    if a.cmd == "fit":
        fit(bootstrap=a.bootstrap)
    elif a.cmd == "siblings":
        siblings()
    else:
        explain(a.fen)


if __name__ == "__main__":
    main()
