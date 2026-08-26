"""How much accuracy does an INTERPRETABLE readout of the concepts cost?

Kaveh 2026-08-25: "we have a bunch of transformer layers on top of the concept
bottleneck, but maybe we ought to have something more explainable -- a decision tree, a
random forest, or whatever the best practice is in interpretable regression -- and
predict WDL off that."

One property of this codebase settles most of the design space before any experiment.
The concepts are BINARY. A function of a binary variable takes only two values, so every
per-feature shape function collapses to f(x) = a + b*x. That makes a GAM, a Neural
Additive Model, and plain logistic regression THE SAME MODEL on these inputs. The whole
additive family is linear here. Anything that beats linear is therefore buying exactly
one thing: INTERACTIONS between concepts.

So the question is not tree-vs-forest-vs-regression, it is: how much do interactions
buy, and is the transformer earning its opacity? This measures that, on frozen concepts
from a trained checkpoint, against the trained transformer head as the incumbent:

  transformer head   the trained sf_v_full -- the incumbent, not interpretable
  logistic (dense)   every concept gets one weight: "concept k is worth w logits"
  logistic (L1)      the same, sparse -- how few concepts suffice?
  decision tree      depth-limited; reads as nested AND rules over concepts
  gradient boosting  a strong non-additive baseline; measures the interaction ceiling

Reported as 3-class WDL accuracy and log-loss on HELD-OUT rows, reproducing the
trainer's own holdout split so nothing is scored on data the encoder trained on.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import torch

from catspace.research.tools.npz_io import load_npz


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/gko8a_step8000.pt")
    ap.add_argument("--shards", default="data/derived/gko1_shards")
    ap.add_argument("--n_rows", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0, help="must match the run's --seed")
    ap.add_argument("--max_depth", type=int, default=5)
    ap.add_argument("--cache", default="artifacts/experiments/iwdl_cache.npz")
    a = ap.parse_args()

    from catspace.research.tools.training_infra.future_kind_labels import _engine
    import chess
    _ck = a.cache.replace(".npz", "_" + os.path.basename(a.ckpt).replace(".pt", "") + ".npz")
    if os.path.exists(_ck):
        _z = np.load(_ck)
        X, Y, ZA = _z["X"], _z["Y"], _z["ZA"]
        print(f"  [cache] {_ck}")
    else:
        X = None
    ke = _engine(a.ckpt, "mps" if torch.backends.mps.is_available() else "cpu")

    # Reproduce the trainer's holdout: it draws hold_rng.random(n) < 0.08 per shard in
    # sorted order, from default_rng(seed+3). Scoring on rows the encoder trained on
    # would flatter every readout equally but tell us nothing about generalisation.
    hold_rng = np.random.default_rng(a.seed + 3)
    if X is None:
     X, Y, ZA = [], [], []
     for f in sorted(glob.glob(os.path.join(a.shards, "shard_*.npz"))):
         z = load_npz(f)
         n = len(z["kind"])
         held = hold_rng.random(n) < 0.08
         if len(X) * 1 >= a.n_rows:
             break
         sf = (z["kind"] == 0) & held
         idx = np.nonzero(sf)[0]
         if len(idx) == 0:
             continue
         wdl, tgt, fen = z["wdl"], z["tgt"], z["fen"]
         for i in idx:
             if len(X) >= a.n_rows:
                 break
             t = np.where(np.isfinite(tgt[i]), tgt[i], -1e9)
             b = int(t.argmax())
             w = wdl[i, b].astype(np.float32)
             if not np.isfinite(w).all() or w.sum() <= 0:
                 continue
             try:
                 bd = chess.Board(str(fen[i]))
             except Exception:
                 continue
             if bd.is_game_over():
                 continue
             with torch.no_grad():
                 ke._encode(bd)
                 zq_i, hmv_i, nmv_i = ke._last
                 X.append(zq_i[0, :ke.field.d_state].cpu().numpy())
                 q_i, _, _ = ke.field.mv_vq(ke.field.mv_vq_proj(hmv_i))
                 m_i = torch.zeros(q_i.shape[:2], dtype=torch.bool, device=q_i.device)
                 m_i[:, :nmv_i] = True
                 ZA.append(ke.lift(zq_i, q_i * m_i[..., None])[0].cpu().numpy())
             Y.append(int(np.argmax(w)))          # 0=win 1=draw 2=loss, mover POV
     X, Y = np.asarray(X, np.float32), np.asarray(Y)
     ZA = np.asarray(ZA, np.float32)
     np.savez_compressed(_ck, X=X, Y=Y, ZA=ZA)
    if len(X) < 200:
        raise SystemExit(f"only {len(X)} usable held-out rows")
    ntr = int(0.7 * len(X))
    Xtr, Xte, Ytr, Yte = X[:ntr], X[ntr:], Y[:ntr], Y[ntr:]
    ZAte = ZA[ntr:]
    print(f"\n  {len(X)} held-out SF rows ({ntr} fit / {len(Xte)} score), "
          f"{X.shape[1]} concept dims | class balance "
          f"{np.bincount(Y, minlength=3) / len(Y)}")

    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, log_loss, recall_score

    rows = []

    def score(name, p, extra=""):
        pred = p.argmax(1)
        acc = accuracy_score(Yte, pred)
        # RAW ACCURACY IS NEARLY UNINFORMATIVE HERE. The WDL target is skewed
        # 63/14/23, so answering "win" every time scores ~0.63 and looks competent.
        # Balanced accuracy is the mean per-class recall, so the draw class -- 14% of
        # the data and the one a planner most needs -- can no longer be ignored for
        # free.
        rec = recall_score(Yte, pred, average=None, labels=[0, 1, 2], zero_division=0)
        bal = float(np.mean(rec))
        ll = log_loss(Yte, p, labels=[0, 1, 2])
        rows.append((name, acc, bal, ll, tuple(round(float(r), 3) for r in rec), extra))

    # the incumbent: the trained transformer head, on the same held-out rows
    with torch.no_grad():
        # the head's REAL input, built exactly as l_sv builds it in training
        za = torch.from_numpy(ZAte).to(ke.dev)
        pt = torch.softmax(ke.sf_v(za[:, ke._v_slice]), -1).float().cpu().numpy()
    score("transformer head (incumbent)", pt, "not interpretable")

    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(Xtr, Ytr)
    score("logistic, dense", lr.predict_proba(Xte),
          f"{X.shape[1]} weights, one per concept")

    l1 = LogisticRegression(max_iter=3000, penalty="l1", solver="saga", C=0.05)
    l1.fit(Xtr, Ytr)
    nz = int((np.abs(l1.coef_).max(0) > 1e-6).sum())
    score("logistic, L1 sparse", l1.predict_proba(Xte), f"{nz} concepts used")

    dt = DecisionTreeClassifier(max_depth=a.max_depth, random_state=0)
    dt.fit(Xtr, Ytr)
    score(f"decision tree (depth {a.max_depth})", dt.predict_proba(Xte),
          f"{dt.get_n_leaves()} leaves")

    gb = HistGradientBoostingClassifier(max_depth=6, max_iter=200, random_state=0)
    gb.fit(Xtr, Ytr)
    score("gradient boosting", gb.predict_proba(Xte), "interaction ceiling")

    print(f"\n  {'readout':<30} {'acc':>6} {'BALANCED':>9} {'logloss':>8}  "
          f"{'recall W/D/L':>20}  note")
    for nm, acc, bal, ll, rec, ex in rows:
        print(f"  {nm:<30} {acc:>6.3f} {bal:>9.3f} {ll:>8.3f}  "
              f"{str(rec):>20}  {ex}")
    cnt = np.bincount(Yte, minlength=3)
    base = cnt.max() / len(Yte)
    print(f"  {'always-predict-majority':<30} {base:>6.3f} {1/3:>9.3f} "
          f"{'--':>8}  {'(1,0,0)-like':>20}  baseline")
    print(f"  class balance in the test set: W {cnt[0]} / D {cnt[1]} / L {cnt[2]} "
          f"-- raw accuracy of {base:.3f} is free, balanced accuracy of 0.333 is free")
    lin = [r for r in rows if r[0] == "logistic, dense"][0]
    gbr = [r for r in rows if r[0] == "gradient boosting"][0]
    print(f"\n  INTERACTIONS ARE WORTH {gbr[2] - lin[2]:+.3f} BALANCED accuracy "
          f"(boosting {gbr[2]:.3f} vs additive/linear {lin[2]:.3f}); on raw accuracy "
          f"the same gap reads {gbr[1] - lin[1]:+.3f}, which the class skew inflates.")
    print(f"  Because the concepts are BINARY, 'logistic, dense' IS the best possible "
          f"additive\n  model -- a GAM or a Neural Additive Model would fit exactly the "
          f"same function.")


if __name__ == "__main__":
    main()
