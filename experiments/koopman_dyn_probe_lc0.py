#!/usr/bin/env python
"""experiments/koopman_dyn_probe_lc0.py -- same Koopman-linearization question as
koopman_dyn_probe.py, but against the biggest well-trained Leela net we actually
have on disk (data/engines/lc0/t1-512x15x8h.onnx, 15 layers/512-dim -- a real
distillate of a strong teacher, not our own possibly-undertrained JEPA T1). Kaveh's
suspicion: JEPA T1 may be too small/undertrained for the earlier linearization
result to mean much; check the same claim against a much stronger, independently
trained network.

Differences from the JEPA version (lc0 has no pretrained DynPredictor to compare
against, and no learned action-embedding tables to reuse):
  - phi(s) = mean-pooled per-square token at the trunk's final layer (order-
    invariant to lc0's POV board-flip, so no pov_square_index needed for a pool).
  - action features u(a) = raw one-hot(from_sq) ++ one-hot(to_sq) ++ one-hot(promo)
    (133-dim) -- standard DMDc control input, no embedding to learn/reuse.
  - nonlinear baseline = a FRESH MLPRegressor fit on the same (phi(s),u(a))->phi(s')
    data (fair self-trained comparison, since there's no existing trained predictor
    to borrow).

Usage:
  experiments/koopman_dyn_probe_lc0.py --n 4000
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
    ap.add_argument("--onnx", default="data/engines/lc0/t1-512x15x8h.onnx")
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--alpha", type=float, default=50.0,
                     help="ridge regularization -- needs to be much stronger than the "
                          "JEPA version (256-dim phi): here phi is 512-dim and X is "
                          "645-dim total, so n_train can be comparable to or smaller "
                          "than the feature count unless n is large")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts/experiments/koopman_dyn_probe_lc0")
    args = ap.parse_args()
    t0 = time.time()
    rng = np.random.default_rng(args.seed)

    from catspace.research.tools.training_infra.train.scaffold import resolve_device
    from lczerolens import LczeroModel, LczeroBoard
    dev = resolve_device("auto")
    trunk = LczeroModel.from_onnx_path(args.onnx).float().to(dev).eval()
    names = [n for n, _ in trunk.named_modules()
             if n and all(k not in n.lower() for k in ("policy", "value", "wdl", "output", "mlh"))]
    hook_name = names[-1]
    store = {}
    dict(trunk.named_modules())[hook_name].register_forward_hook(
        lambda mo, i, o: store.__setitem__("t", o))
    for p in trunk.parameters():
        p.requires_grad_(False)
    print(f"lc0 trunk loaded: {args.onnx}, hook={hook_name}")

    d = np.load(args.labeled, allow_pickle=True)
    fen, mv_uci, game = d["fen"], d["move"], d["game"]
    idx = rng.choice(len(fen), min(args.n, len(fen)), replace=False)
    boards_s, boards_sp, acts, groups = [], [], [], []
    for i in idx:
        b0 = chess.Board(fen[i]); mv = chess.Move.from_uci(mv_uci[i])
        b1 = b0.copy(); b1.push(mv)
        promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
        boards_s.append(b0); boards_sp.append(b1)
        acts.append((mv.from_square, mv.to_square, promo.get(mv.promotion, 4)))
        groups.append(int(game[i]))
    acts = np.array(acts); groups = np.array(groups)

    def encode(boards):
        out = []
        for i0 in range(0, len(boards), args.batch):
            chunk = boards[i0:i0 + args.batch]
            lc = [LczeroBoard(b.fen()) for b in chunk]
            x = torch.stack([bb.to_input_tensor() for bb in lc]).float().to(dev)
            with torch.no_grad():
                trunk(x)
                t = store["t"]; C = t.shape[-1]
                pooled = t.reshape(len(chunk), 64, C).mean(1)
            out.append(pooled.cpu().numpy())
            if (i0 // args.batch) % 20 == 0:
                print(f"    encoded {min(i0 + args.batch, len(boards))}/{len(boards)}", flush=True)
        return np.concatenate(out, 0)

    print(f"encoding {len(idx)} x2 real (s, a, s') transitions with lc0-big ...")
    phi_s = encode(boards_s)
    phi_sp = encode(boards_sp)

    def onehot(vals, n):
        z = np.zeros((len(vals), n), np.float32)
        z[np.arange(len(vals)), vals] = 1.0
        return z
    u = np.concatenate([onehot(acts[:, 0], 64), onehot(acts[:, 1], 64), onehot(acts[:, 2], 5)], 1)

    X = np.concatenate([phi_s, u], axis=1)
    Y = phi_sp
    gs = np.unique(groups)
    te_g = set(rng.choice(gs, max(1, int(len(gs) * args.test_frac)), replace=False).tolist())
    te = np.array([g in te_g for g in groups]); tr = ~te
    print(f"  {tr.sum()} train / {te.sum()} test transitions (held out by game)")

    from sklearn.linear_model import Ridge
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import r2_score
    from sklearn.preprocessing import StandardScaler
    # lc0's raw hook output isn't LayerNorm-bounded like JEPA's phi -- scale features
    # (fit on train only) so Ridge/MLP aren't dominated by a few large-magnitude dims,
    # and so the MLP doesn't blow up (it did, unscaled, at smoke scale: R2 in the
    # billions-negative -- a capacity/scale mismatch, not a real result).
    xsc = StandardScaler().fit(X[tr]); Xtr, Xte = xsc.transform(X[tr]), xsc.transform(X[te])
    ysc = StandardScaler().fit(Y[tr]); Ytr = ysc.transform(Y[tr])

    ridge = Ridge(alpha=args.alpha).fit(Xtr, Ytr)
    Y_lin = ysc.inverse_transform(ridge.predict(Xte))

    mlp = MLPRegressor(hidden_layer_sizes=(256,), max_iter=2000, early_stopping=True,
                        alpha=1e-2, random_state=args.seed).fit(Xtr, Ytr)
    Y_mlp = ysc.inverse_transform(mlp.predict(Xte))

    def cos(a, b):
        an = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
        bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return (an * bn).sum(1)

    r2 = lambda yt, yp: r2_score(yt, yp, multioutput="variance_weighted")
    r2_lin = r2(Y[te], Y_lin); cos_lin = cos(Y[te], Y_lin).mean()
    r2_mlp = r2(Y[te], Y_mlp); cos_mlp = cos(Y[te], Y_mlp).mean()
    r2_id = r2(Y[te], phi_s[te]); cos_id = cos(Y[te], phi_s[te]).mean()

    print(f"VERDICT koopman-dyn-lc0: n_test {te.sum()} | trunk t1-512x15x8h (real, "
          f"strong Leela distillate) | "
          f"KOOPMAN(linear) R2 {r2_lin:.3f} cos {cos_lin:.3f} | "
          f"MLP(fresh nonlinear, hidden=256) R2 {r2_mlp:.3f} cos {cos_mlp:.3f} | "
          f"IDENTITY R2 {r2_id:.3f} cos {cos_id:.3f} | gap (mlp-linear) R2 {r2_mlp - r2_lin:+.3f}")
    if r2_lin > r2_id and (r2_mlp - r2_lin) < 0.05:
        verdict = "LINEARIZES WELL even in the strong Leela net -- not a JEPA-T1-is-undertrained artifact"
    elif r2_lin > r2_id:
        verdict = "PARTIAL -- real linear structure, but a meaningful nonlinear gap remains"
    else:
        verdict = "DOES NOT LINEARIZE in the strong net either -- barely beats identity"
    print(f"VERDICT koopman-dyn-lc0-reading: {verdict}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5, 4.5))
    labels_ = ["identity\n(no model)", "Koopman\n(linear)", "MLP\n(fresh nonlinear)"]
    r2s = [r2_id, r2_lin, r2_mlp]; coss = [cos_id, cos_lin, cos_mlp]
    xw = np.arange(3)
    ax2 = ax.twinx()
    ax.bar(xw - 0.18, r2s, width=0.36, color="#36c", label="R2")
    ax2.bar(xw + 0.18, coss, width=0.36, color="#c63", label="cosine sim")
    ax.set_xticks(xw); ax.set_xticklabels(labels_)
    ax.set_ylabel("R2 (one-step phi(s') prediction)", color="#36c")
    ax2.set_ylabel("cosine similarity", color="#c63")
    ax.set_title(f"one-step dynamics: lc0 t1-512x15x8h (n_test={te.sum()})")
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
