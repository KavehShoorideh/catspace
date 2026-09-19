"""ROOTING THE LEARNED TOKENS IN THE GRAPH OPERATIONS (Kaveh 2026-09-19). For each learned token: the smallest
expression over the enumerated region relations (regions.py) that predicts when it fires -- a shallow decision tree
over features named by piece slot and relation, e.g. "Bk->Wp: safe_meets_safe". A token explained by one relation
IS that concept; one needing a few is a composed concept; one no tree explains is reported as unrooted (a missing
relation or a shortcut). FAITHFULNESS: replace every token by its expression's output and re-predict the result
with the model's own head; the accuracy gap says how much the rooting lost. EXPLANATION: for two king moves with
different results, the tokens that differ between the children, rooted."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import chess
import numpy as np
import torch

from . import tokens as T
from .regions import PAIR_RELATIONS, UNARY, regions, relations
from .endgames import load as load_rows

OUT = Path("data/derived/tokens")
TYPE_CH = {chess.KING: "k", chess.QUEEN: "q", chess.ROOK: "r", chess.BISHOP: "b", chess.KNIGHT: "n", chess.PAWN: "p"}


def slots(board: chess.Board) -> dict[int, str]:
    """square -> slot name: W/B + type letter (+ index when the type repeats), by square order."""
    counts = Counter(); out = {}
    for sq in sorted(board.piece_map()):
        p = board.piece_at(sq); base = ("W" if p.color else "B") + TYPE_CH[p.piece_type]
        k = counts[base]; counts[base] += 1
        out[sq] = base + (str(k) if k else "")
    return out


def feature_names() -> list[str]:
    names = []
    kinds = ["Wk", "Wq", "Wr", "Wr1", "Wb", "Wb1", "Wn", "Wn1", "Wp", "Wp1", "Bk", "Bq", "Br", "Br1", "Bb", "Bb1", "Bn", "Bn1", "Bp", "Bp1"]
    for a in kinds:
        for b in kinds:
            if a != b:
                names += [f"{a}->{b}: {r}" for r in PAIR_RELATIONS]
    for a in kinds:
        names += [f"{a}: {u}" for u in UNARY]
    return names


def features_of(row: dict, index: dict[str, int]) -> np.ndarray:
    """From the row's stored relations (keyed by square names) to the slot-named feature vector."""
    b = chess.Board(row["fen"]); sl = slots(b)
    x = np.zeros(len(index), np.float32)
    for key, v in row["pairs"].items():
        a, c = key.split("-")
        sa, sc = sl.get(chess.parse_square(a)), sl.get(chess.parse_square(c))
        if sa is None or sc is None:
            continue
        for r, val in zip(PAIR_RELATIONS, v):
            i = index.get(f"{sa}->{sc}: {r}")
            if i is not None:
                x[i] = val
    for sq, v in row["unary"].items():
        sa = sl.get(chess.parse_square(sq))
        for u, val in zip(UNARY, v):
            i = index.get(f"{sa}: {u}")
            if i is not None:
                x[i] = val
    return x


def features_of_board(board: chess.Board, index: dict[str, int]) -> np.ndarray:
    regs = regions(board); rel = relations(board, regs)
    row = dict(fen=board.fen(), pairs={f"{chess.square_name(a)}-{chess.square_name(c)}": v for (a, c), v in rel["pairs"].items()}, unary={chess.square_name(s): v for s, v in rel["unary"].items()})
    return features_of(row, index)


@torch.no_grad()
def hard_tokens(model, data: T.Data, ids, device) -> np.ndarray:
    out = []
    for s in range(0, len(ids), 512):
        b = data.batch(ids[s:s + 512], device); o = model(b["sq"], b["side"], b["rel"])
        out.append((o["p"] > 0.5).cpu().numpy())
    return np.concatenate(out)


def tree_text(tree, names) -> str:
    from sklearn.tree import _tree
    t = tree.tree_
    def rec(node, depth):
        if t.feature[node] == _tree.TREE_UNDEFINED:
            v = t.value[node][0]; return f"-> {'ON' if v[-1] >= v[0] else 'off'} ({int(v.sum())})"
        name = names[t.feature[node]]; thr = t.threshold[node]
        return f"[{name} <= {thr:.1f}] ? ({rec(t.children_left[node], depth + 1)}) : ({rec(t.children_right[node], depth + 1)})"
    return rec(0, 0)


def root(ckpt: str, depth: int = 3, log=print):
    from sklearn.tree import DecisionTreeClassifier
    model, cfg, device, held = T.load_model(ckpt)
    rows = load_rows(); data = T.Data(rows, held, log=lambda *a, **k: None)
    names = feature_names(); index = {n: i for i, n in enumerate(names)}
    X = np.stack([features_of(r, index) for r in rows])
    tr, te, hd = data.ids("train"), data.ids("test"), data.ids("held")
    Ttr, Tte, Thd = hard_tokens(model, data, tr, device), hard_tokens(model, data, te, device), hard_tokens(model, data, hd, device)
    K = Ttr.shape[1]
    trees, report = {}, []
    log(f"[root] {ckpt}: {K} codes; {len(names)} relation features; train {len(tr)} test {len(te)} held {len(hd)}")
    for k in range(K):
        y = Ttr[:, k].astype(int)
        base = max(y.mean(), 1 - y.mean())
        if y.mean() < 0.01 or y.mean() > 0.99:
            report.append(dict(code=k, on=float(y.mean()), status="dead")); continue
        best = None
        for d in range(1, depth + 1):
            clf = DecisionTreeClassifier(max_depth=d, min_samples_leaf=50, random_state=0).fit(X[tr], y)
            acc_te = float((clf.predict(X[te]) == Tte[:, k]).mean()); base_te = max(Tte[:, k].mean(), 1 - Tte[:, k].mean())
            gain = acc_te - base_te
            if best is None or gain > best[0] + 0.01:
                best = (gain, d, clf, acc_te, base_te)
        gain, d, clf, acc_te, base_te = best
        acc_hd = float((clf.predict(X[hd]) == Thd[:, k]).mean()) if len(hd) else float("nan")
        status = "rooted" if gain >= 0.15 else "partly" if gain >= 0.05 else "unrooted"
        trees[k] = clf
        report.append(dict(code=k, on=float(y.mean()), status=status, depth=d, acc=acc_te, base=base_te, acc_held=acc_hd, expr=tree_text(clf, names)))
    for r in sorted(report, key=lambda r: -(r.get("acc", 0) - r.get("base", 0))):
        if r["status"] == "dead":
            continue
        log(f"  code {r['code']:2d} on {r['on']:.2f} {r['status']:8s} depth {r['depth']} acc {r['acc']:.3f} (base {r['base']:.3f}; held {r['acc_held']:.3f}): {r['expr'][:220]}")
    counts = Counter(r["status"] for r in report)
    log(f"[root] {dict(counts)}")
    # FAITHFULNESS: replace tokens by the trees' outputs, re-predict with the model's own head
    def faith(ids, Tk, name):
        Xr = X[ids]; R = Tk.copy()
        for k, clf in trees.items():
            R[:, k] = clf.predict(Xr).astype(bool)
        with torch.no_grad():
            a = model.wdl_from_tokens(torch.as_tensor(Tk, dtype=torch.float32, device=device))[:, :3].argmax(-1).cpu().numpy()
            b = model.wdl_from_tokens(torch.as_tensor(R, dtype=torch.float32, device=device))[:, :3].argmax(-1).cpu().numpy()
        y = data.wdl[ids]
        log(f"[root] faithfulness on {name}: result accuracy from the learned tokens {float((a == y).mean()):.3f}; from the ROOTED tokens {float((b == y).mean()):.3f}; tokens reproduced {float((R == Tk).mean()):.3f}")
    faith(te, Tte, "test"); faith(hd, Thd, "held materials")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(dict(report=report), open(OUT / "rooting.json", "w"), indent=1)
    import pickle
    pickle.dump(dict(trees=trees, names=names), open(OUT / "trees.pkl", "wb"))
    return trees, names, report


@torch.no_grad()
def explain(ckpt: str, fen: str, log=print):
    """Two (or more) king moves with different results: the tokens that differ between the children, rooted."""
    import pickle
    model, cfg, device, held = T.load_model(ckpt)
    tk = pickle.load(open(OUT / "trees.pkl", "rb")); trees, names = tk["trees"], tk["names"]
    rep = {r["code"]: r for r in json.load(open(OUT / "rooting.json"))["report"]}
    index = {n: i for i, n in enumerate(names)}
    import chess.syzygy
    tb = chess.syzygy.open_tablebase(str(Path("assets/tablebases/syzygy")))
    b = chess.Board(fen); ksq = b.king(b.turn)
    children = []
    for m in b.legal_moves:
        if m.from_square != ksq:
            continue
        nb = b.copy(stack=False); nb.push(m)
        w = -tb.probe_wdl(nb) if not nb.is_game_over() else 0
        sq, side, rel = T.encode(nb)
        o = model(torch.as_tensor(sq[None], device=device), torch.as_tensor([side], device=device), torch.as_tensor(rel[None], device=device).float())
        children.append((b.san(m), w, (o["p"][0] > 0.5).cpu().numpy(), features_of_board(nb, index)))
    tb.close()
    log(f"[explain] {fen}: " + ", ".join(f"{san} {'wins' if w > 0 else 'draws' if w == 0 else 'loses'}" for san, w, _, _ in children))
    by = defaultdict(list)
    for san, w, t, x in children:
        by[1 if w > 0 else 0 if w == 0 else -1].append((san, t, x))
    outcomes = sorted(by)
    if len(outcomes) < 2:
        log("[explain] all king moves lead to the same result"); return
    for i in range(len(outcomes)):
        for j in range(i + 1, len(outcomes)):
            (s1, t1, x1), (s2, t2, x2) = by[outcomes[i]][0], by[outcomes[j]][0]
            diff = np.nonzero(t1 != t2)[0]
            log(f"[explain] {s1} vs {s2}: {len(diff)} tokens differ")
            for k in diff:
                r = rep.get(int(k), {})
                which = s2 if t2[k] else s1
                log(f"    code {k} on after {which}: {r.get('status', '?')} -- {r.get('expr', '')[:200]}")
            # the relations that differ, in words (the rooted vocabulary directly)
            dx = np.nonzero(x1 != x2)[0]
            log(f"    relations that differ ({len(dx)}): " + "; ".join(f"{names[i]} {int(x1[i])}->{int(x2[i])}" for i in dx[:12]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["root", "explain"]); ap.add_argument("--ckpt", default="data/derived/checkpoints/tokens/last.pt")
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--fen", default="4k3/1R6/4K3/4P3/8/8/8/5r2 b - - 0 58")
    a = ap.parse_args(argv)
    if a.cmd == "root":
        root(a.ckpt, a.depth)
    else:
        explain(a.ckpt, a.fen)


if __name__ == "__main__":
    main()
