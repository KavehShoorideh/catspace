"""THE NEURAL GUIDE for the search over bodies (AlphaGeometry's role, Trinh et al. 2024: a learned proposer for the
step the symbolic engine cannot enumerate; here: which piece to bring into the frame next).

Labels: shap_bodies' per-piece mass share -- how much of a move's outcome each other piece carries (the exact Möbius
decomposition on the position's own pieces). Features per candidate piece: generic -- its type, the mover's type,
the domain's binary relations between the candidate and the mover BEFORE and AFTER the move, the candidate's unary
relations, and its rarity-weighted relation count (the heuristic the guide must beat). A small PyTorch MLP scores
each candidate; the body search ranks by the score. Gauges: ranking of the relevant pieces (AUC) against the
heuristic and against random; then bodies-to-settle in the body search.

    python -m catspace.incubator.analyst.body_guide train --labels data/derived/lemmas/shap_bodies_relative_labels.json
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn as nn

OUT = Path("data/derived/lemmas")
CKPT = OUT / "body_guide.pt"


def features(dom, board: chess.Board, move: chess.Move, cand: int, rels_bin, rels_un, types: list, weight: dict) -> np.ndarray:
    """One candidate piece's feature vector. Generic over the domain's relations."""
    mover = move.from_square
    f = []
    ctype = ("w" if board.piece_at(cand).color else "b") + board.piece_at(cand).symbol().upper()
    mtype = ("w" if board.piece_at(mover).color else "b") + board.piece_at(mover).symbol().upper()
    f += [1.0 if t == ctype else 0.0 for t in types]
    f += [1.0 if t == mtype else 0.0 for t in types]
    st = dom.state_of(board, [mover, cand])
    after_b = board.copy(stack=False); after_b.push(move)
    st2 = dom.state_of(after_b, [move.to_square, cand]) if after_b.piece_at(cand) is not None and cand != move.to_square else None
    for r in rels_bin:
        try:
            f += [float(r.fn(st, mover, cand)), float(r.fn(st, cand, mover))]
        except Exception:
            f += [0.0, 0.0]
        if st2 is not None:
            try:
                f += [float(r.fn(st2, move.to_square, cand)), float(r.fn(st2, cand, move.to_square))]
            except Exception:
                f += [0.0, 0.0]
        else:
            f += [0.0, 0.0]
    for r in rels_un:
        try:
            f.append(float(r.fn(st, cand)))
        except Exception:
            f.append(0.0)
    # the heuristic's own score, rarity-weighted relation count (so the guide can only add to it)
    h = 0.0
    for r in rels_bin:
        try:
            h += weight[r.name] * (bool(r.fn(st, mover, cand)) or bool(r.fn(st, cand, mover)))
            if st2 is not None:
                h += weight[r.name] * (bool(r.fn(st2, move.to_square, cand)) or bool(r.fn(st2, cand, move.to_square)))
        except Exception:
            pass
    f.append(h)
    f.append(float(cand == move.to_square))
    return np.array(f, dtype=np.float32)


def relation_weights(dom, board, rels_bin):
    from collections import Counter
    ents = [e for e, _ in dom.entities(board)]
    freq = Counter(); pairs = 0
    for i in ents:
        for j in ents:
            if i == j:
                continue
            pairs += 1
            st = dom.state_of(board, [i, j])
            for r in rels_bin:
                try:
                    if r.fn(st, i, j):
                        freq[r.name] += 1
                except Exception:
                    pass
    return {r.name: 1.0 - freq[r.name] / max(1, pairs) for r in rels_bin}


class Guide(nn.Module):
    def __init__(self, d_in: int, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def build(dom, records: list, log=print):
    rels_bin = [r for r in dom.relations if not r.on_action and r.arity == 2]
    rels_un = [r for r in dom.relations if not r.on_action and r.arity == 1]
    X = []; y = []; grp = []; heur = []
    t0 = time.perf_counter()
    for gi, rec in enumerate(records):
        b = chess.Board(rec["fen"]); mv = chess.Move.from_uci(rec["move"])
        w = relation_weights(dom, b, rels_bin)
        share = rec.get("share", {})
        for e, t in dom.entities(b):
            if e == mv.from_square or b.piece_at(e).piece_type == chess.KING:
                continue
            f = features(dom, b, mv, e, rels_bin, rels_un, dom.types, w)
            X.append(f); y.append(float(share.get(chess.square_name(e), 0.0))); grp.append(gi); heur.append(f[-2])
        if gi % 500 == 0 and gi:
            log(f"[guide] features {gi}/{len(records)} ({time.perf_counter() - t0:.0f}s)")
    return np.stack(X), np.array(y, dtype=np.float32), np.array(grp), np.array(heur, dtype=np.float32)


def auc_by_group(score, y, grp, thresh=0.05):
    """Mean over positions of the AUC of score for 'relevant' (share >= thresh) vs the rest; positions with both classes."""
    from sklearn.metrics import roc_auc_score
    aucs = []
    for g in np.unique(grp):
        m = grp == g
        lab = (y[m] >= thresh).astype(int)
        if lab.min() == lab.max():
            continue
        aucs.append(roc_auc_score(lab, score[m]))
    return float(np.mean(aucs)) if aucs else float("nan"), len(aucs)


def train(labels_path: Path, epochs: int = 30, seed: int = 0, log=print):
    from .chess_domain import domain
    dom = domain()
    data = json.load(open(labels_path))
    recs = [r for r in data["records"] if r.get("share")]
    rng = random.Random(seed); rng.shuffle(recs)
    # split by position identity (fen), not by row
    fens = sorted({r["fen"] for r in recs}); rng.shuffle(fens)
    test_f = set(fens[: len(fens) // 5])
    train_r = [r for r in recs if r["fen"] not in test_f]; test_r = [r for r in recs if r["fen"] in test_f]
    log(f"[guide] records train {len(train_r)} test {len(test_r)} (split by position)")
    Xtr, ytr, gtr, htr = build(dom, train_r, log); Xte, yte, gte, hte = build(dom, test_r, log)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    torch.manual_seed(seed)
    model = Guide(Xtr.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    xt = torch.tensor((Xtr - mu) / sd); yt = torch.tensor(ytr); xe = torch.tensor((Xte - mu) / sd)
    for ep in range(epochs):
        perm = torch.randperm(len(xt))
        tot = 0.0
        for i in range(0, len(xt), 256):
            idx = perm[i: i + 256]
            loss = nn.functional.mse_loss(model(xt[idx]), yt[idx])
            opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep % 10 == 0 or ep == epochs - 1:
            with torch.no_grad():
                pe = model(xe).numpy()
            a_g, n = auc_by_group(pe, yte, gte); a_h, _ = auc_by_group(hte, yte, gte); a_r, _ = auc_by_group(np.random.default_rng(seed).random(len(yte)), yte, gte)
            log(f"[guide] epoch {ep} train mse {tot / len(xt):.4f} | test AUC relevant-piece ranking: guide {a_g:.3f}  heuristic {a_h:.3f}  random {a_r:.3f}  (n positions {n})")
    torch.save(dict(state=model.state_dict(), mu=mu, sd=sd, d_in=Xtr.shape[1]), CKPT)
    log(f"[guide] -> {CKPT}")


class GuideRanker:
    """Ranks candidate bodies for body_search with the trained guide."""
    def __init__(self, dom, ckpt: Path = CKPT):
        d = torch.load(ckpt, weights_only=False)
        self.model = Guide(d["d_in"]); self.model.load_state_dict(d["state"]); self.model.eval()
        self.mu, self.sd = d["mu"], d["sd"]; self.dom = dom
        self.rels_bin = [r for r in dom.relations if not r.on_action and r.arity == 2]
        self.rels_un = [r for r in dom.relations if not r.on_action and r.arity == 1]

    def rank(self, board, move, candidates):
        w = relation_weights(self.dom, board, self.rels_bin)
        X = np.stack([features(self.dom, board, move, c, self.rels_bin, self.rels_un, self.dom.types, w) for c in candidates])
        with torch.no_grad():
            s = self.model(torch.tensor((X - self.mu) / self.sd)).numpy()
        return sorted(zip(s.tolist(), candidates), key=lambda x: (-x[0], x[1]))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train"]); ap.add_argument("--labels", default=str(OUT / "shap_bodies_relative_labels.json")); ap.add_argument("--epochs", type=int, default=30)
    a = ap.parse_args(argv)
    train(Path(a.labels), a.epochs)


if __name__ == "__main__":
    main()
