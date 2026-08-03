#!/usr/bin/env python
"""experiments/koopman_dyn_probe.py -- "go the Koopman way" (Kaveh 2026-07-31): does
chess dynamics tolerate a LINEAR one-step operator in the trunk's own embedding
space? Koopman/DMDc form: phi(s') ~= A @ phi(s) + B @ u(a), A (d,d), B (d,72), fit by
closed-form ridge regression on REAL game transitions -- no gradient training loop,
reuses the frozen JEPA T1 encoder AND its own already-trained action-embedding
tables (from_emb/to_emb/promo_emb) for a fair, apples-to-apples comparison against
the existing NONLINEAR DynPredictor (a 2-layer MLP combiner over the same inputs).

This is the cheap, decision-relevant first cut before touching any architecture:
if a plain affine operator in the CURRENT frozen phi space already predicts s' about
as well as the trained nonlinear predictor, chess dynamics linearize more than
expected in this space and a real Koopman-constrained DynPredictor swap is worth
building. If linear prediction is much worse, that's real evidence against -- chess's
combinatorial/discrete tactics (a single move can flip many downstream facts
sharply) don't compress into a small global linear map, consistent with the risk
flagged when this was scoped.

Metric: R^2 (per-dimension, then macro-averaged) and cosine similarity between
predicted and true phi(s'), linear vs nonlinear, on a held-out (by game) split.

Usage:
  experiments/koopman_dyn_probe.py --n 8000
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labeled", default="data/derived/transition_data_labeled.npz")
    ap.add_argument("--ckpt", default="artifacts/experiments/jepa_t1_latest.pt")
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=1.0, help="ridge regularization")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/koopman_dyn_probe")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import JepaT1, tokenize, move_ids
    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    dev = resolve_device("auto")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model = JepaT1(**{k: ck["cfg"][k] for k in ("d", "layers", "n_class")}).to(dev)
    model.load_state_dict(ck["state_dict"]); model.eval()
    enc = model.enc
    for p in model.parameters():
        p.requires_grad_(False)

    d = np.load(args.labeled, allow_pickle=True)
    fen, mv_uci, game = d["fen"], d["move"], d["game"]
    idx = rng.choice(len(fen), min(args.n, len(fen)), replace=False)

    print(f"encoding {len(idx)} real (s, a, s') transitions with the frozen trunk ...")
    phi_s, phi_sp, acts, groups = [], [], [], []
    with torch.no_grad():
        for i in idx:
            board = chess.Board(fen[i])
            mv = chess.Move.from_uci(mv_uci[i])
            t0_, g0_ = tokenize(board)
            phi0 = enc(torch.as_tensor(t0_[None]).to(dev), torch.as_tensor(g0_[None]).to(dev))
            board2 = board.copy(); board2.push(mv)
            t1_, g1_ = tokenize(board2)
            phi1 = enc(torch.as_tensor(t1_[None]).to(dev), torch.as_tensor(g1_[None]).to(dev))
            phi_s.append(phi0[0].cpu().numpy()); phi_sp.append(phi1[0].cpu().numpy())
            acts.append(move_ids(mv)); groups.append(int(game[i]))
    phi_s = np.array(phi_s); phi_sp = np.array(phi_sp)
    acts = np.array(acts); groups = np.array(groups)

    # action embedding u(a): reuse the ALREADY-TRAINED tables from the checkpoint's
    # own nonlinear DynPredictor -- same action representation, only the combiner
    # (nonlinear MLP vs linear map) differs, so the comparison isolates that one axis.
    with torch.no_grad():
        a_t = torch.as_tensor(acts).to(dev)
        u = torch.cat([model.dyn.from_emb(a_t[:, 0]), model.dyn.to_emb(a_t[:, 1]),
                       model.dyn.promo_emb(a_t[:, 2])], -1).cpu().numpy()   # (N, 72)
        # nonlinear baseline: the trained DynPredictor's own one-step prediction
        phi_s_t = torch.as_tensor(phi_s).to(dev)
        mlp_pred = model.dyn(phi_s_t, a_t).cpu().numpy()

    X = np.concatenate([phi_s, u], axis=1)   # (N, d+72)
    Y = phi_sp

    gs = np.unique(groups)
    te_g = set(rng.choice(gs, max(1, int(len(gs) * args.test_frac)), replace=False).tolist())
    te = np.array([g in te_g for g in groups]); tr = ~te
    print(f"  {tr.sum()} train / {te.sum()} test transitions (held out by game)")

    from sklearn.linear_model import Ridge
    from sklearn.metrics import r2_score
    ridge = Ridge(alpha=args.alpha).fit(X[tr], Y[tr])
    Y_lin = ridge.predict(X[te])

    def cos(a, b):
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return (an * bn).sum(1)

    r2_lin = r2_score(Y[te], Y_lin)
    r2_mlp = r2_score(Y[te], mlp_pred[te])
    cos_lin = cos(Y[te], Y_lin).mean()
    cos_mlp = cos(Y[te], mlp_pred[te]).mean()
    # trivial baseline: predict s'==s (identity) -- most single moves change phi a
    # little, this bounds how much credit "any reasonable predictor" gets for free
    r2_id = r2_score(Y[te], phi_s[te])
    cos_id = cos(Y[te], phi_s[te]).mean()

    print(f"VERDICT koopman-dyn: n_test {te.sum()} | "
          f"KOOPMAN(linear-in-phi,affine-in-action) R2 {r2_lin:.3f} cos {cos_lin:.3f} | "
          f"NONLINEAR(trained DynPredictor MLP) R2 {r2_mlp:.3f} cos {cos_mlp:.3f} | "
          f"IDENTITY(phi(s') ~= phi(s)) R2 {r2_id:.3f} cos {cos_id:.3f} | "
          f"gap (nonlinear-linear) R2 {r2_mlp - r2_lin:+.3f}")
    if r2_lin > r2_id and (r2_mlp - r2_lin) < 0.05:
        verdict = "LINEARIZES WELL -- Koopman-constrained DynPredictor is worth building for real"
    elif r2_lin > r2_id:
        verdict = "PARTIAL -- linear captures real structure but leaves a nonlinear gap the MLP is using"
    else:
        verdict = "DOES NOT LINEARIZE -- barely beats the identity baseline, consistent with the combinatorial-tactics risk flagged when this was scoped"
    print(f"VERDICT koopman-dyn-reading: {verdict}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels_ = ["identity\n(no model)", "Koopman\n(linear)", "DynPredictor\n(nonlinear MLP)"]
    r2s = [r2_id, r2_lin, r2_mlp]
    coss = [cos_id, cos_lin, cos_mlp]
    xw = np.arange(3)
    ax2 = ax.twinx()
    ax.bar(xw - 0.18, r2s, width=0.36, color="#36c", label="R2")
    ax2.bar(xw + 0.18, coss, width=0.36, color="#c63", label="cosine sim")
    ax.set_xticks(xw); ax.set_xticklabels(labels_)
    ax.set_ylabel("R2 (one-step phi(s') prediction)", color="#36c")
    ax2.set_ylabel("cosine similarity", color="#c63")
    ax.set_title(f"one-step dynamics: linear vs nonlinear in frozen JEPA T1 phi (n_test={te.sum()})")
    fig.tight_layout()
    Path("artifacts/experiments").mkdir(exist_ok=True, parents=True)
    Path("docs/figures").mkdir(exist_ok=True, parents=True)
    fig.savefig(f"{args.out}.png", dpi=130)
    fig.savefig(f"docs/figures/{Path(args.out).name}.png", dpi=130)
    np.savez(f"{args.out}.npz", r2_lin=r2_lin, r2_mlp=r2_mlp, r2_id=r2_id,
             cos_lin=cos_lin, cos_mlp=cos_mlp, cos_id=cos_id)
    print(f"wrote {args.out}.png + docs/figures/{Path(args.out).name}.png + {args.out}.npz")
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
