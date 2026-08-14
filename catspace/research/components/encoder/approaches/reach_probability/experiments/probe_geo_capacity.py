#!/usr/bin/env python
"""probe_geo_capacity.py -- HOW MANY GEOMETRY HEADS IS ENOUGH? (Kaveh 2026-08-14: "three
feels like too few for the geometry. How do we know we have enough concepts?")

Answers it without a training run. The jqt6 geometry branch is a product quantizer: H heads,
64 codes each, over z_B. So simulate exactly that structure on a FROZEN checkpoint --
split z_B into H chunks, k-means 64 centroids per chunk, then ask how well the resulting
discrete code tuple reproduces the field's own readouts.

Reported per H:
    R2(y6)      how much of the six field readouts the codes explain
    E-resid     RMSE on expected score alone (the number a human would read)
    E-spread    median within-cell spread of E -- a cell spanning 0.3-0.7 is not a concept
    usage       distinct cells actually occupied (capacity you are really getting)
Saturation in R2/E-spread = enough heads. Still climbing = too few.

    .venv/bin/python -m ...probe_geo_capacity --ckpt artifacts/experiments/reach_jqt5_latest.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def kmeans(X, K, iters=25, seed=0):
    g = torch.Generator().manual_seed(seed)
    C = X[torch.randperm(len(X), generator=g)[:K]].clone()
    for _ in range(iters):
        d = torch.cdist(X, C)
        a = d.argmin(1)
        for k in range(K):
            m = a == k
            if m.any():
                C[k] = X[m].mean(0)
    return C, torch.cdist(X, C).argmin(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/reach_jqt5_latest.pt")
    ap.add_argument("--games", type=int, default=700)
    ap.add_argument("--heads", default="1,2,3,4,5,6,8")
    ap.add_argument("--codes", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
        load_net)
    dev = args.device
    model, _ = load_net(args.ckpt, dev)
    model.eval()
    tr = T.build(n_human=0, n_sf=4000, seed=0, cache=True, max_plies=400, n_piecedown=45906)
    g_of = tr.game_of_row()
    rng = np.random.default_rng(0)
    gsel = rng.choice(int(g_of.max()) + 1, args.games, replace=False)
    rws = np.flatnonzero(np.isin(g_of, gsel))
    rws = rws[rng.permutation(len(rws))[:40000]]

    P3 = model.poles.poles[:3]
    Z, Y = [], []
    with torch.no_grad():
        for a in range(0, len(rws), 4096):
            rr = rws[a:a + 4096]
            phi = model.backbone(
                torch.from_numpy(tr.tok[rr].astype(np.int64)).to(dev),
                torch.from_numpy(tr.glob[rr].astype(np.float32)).to(dev))
            zb = model.proj_b(phi).float()
            dA = torch.stack([model.dA(zb, P3[[k]].expand(len(zb), -1)) for k in range(3)], 1)
            dB = torch.stack([model.dB(zb, P3[[k]].expand(len(zb), -1)) for k in range(3)], 1)
            pr = torch.softmax(-dB / 5.0, 1)
            Z.append(zb.cpu()); Y.append(torch.cat([torch.log1p(dA.clamp(min=0)), pr], 1).cpu())
    Z = torch.cat(Z); Y = torch.cat(Y)
    E = (Y[:, 3] + 0.5 * Y[:, 4]).numpy()            # white expected score
    Yn = (Y - Y.mean(0)) / Y.std(0).clamp(min=1e-6)
    print(f"[geo-cap] {args.ckpt}\n[geo-cap] {len(Z):,} positions · z_B dim {Z.shape[1]} · "
          f"{args.codes} codes/head\n", flush=True)
    print(f"{'H':>2} {'bits':>5} {'R2(y6)':>8} {'E-resid':>8} {'E-spread':>9} "
          f"{'cells used':>11}")
    prev = None
    for H in [int(x) for x in args.heads.split(",")]:
        dims = Z.shape[1] // H
        codes = []
        for h in range(H):
            sl = Z[:, h * dims:(h + 1) * dims] if h < H - 1 else Z[:, h * dims:]
            _, a = kmeans(sl, args.codes, iters=20, seed=h)
            codes.append(a)
        C = torch.stack(codes, 1)
        # one-hot codes -> y6, ridge (the decoder the architecture would learn)
        X = torch.zeros(len(C), H * args.codes)
        for h in range(H):
            X[torch.arange(len(C)), h * args.codes + C[:, h]] = 1.0
        X = torch.cat([X, torch.ones(len(X), 1)], 1)
        W = torch.linalg.lstsq(X.T @ X + 1e-2 * torch.eye(X.shape[1]), X.T @ Yn).solution
        pred = X @ W
        r2 = float(1 - ((pred - Yn) ** 2).sum() / (Yn ** 2).sum())
        # E fidelity in native units, and the within-cell spread that decides legibility
        key = np.ascontiguousarray(C.numpy()).view(
            np.dtype((np.void, C.numpy().dtype.itemsize * H)))
        _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
        inv = np.asarray(inv).ravel()
        sp, res = [], []
        for c in np.flatnonzero(cnt >= 5):
            m = inv == c
            sp.append(float(E[m].std()))
            res.append(float(np.abs(E[m] - E[m].mean()).mean()))
        e_sp = float(np.median(sp)) if sp else float("nan")
        e_rs = float(np.mean(res)) if res else float("nan")
        bits = H * np.log2(args.codes)
        star = ""
        if prev is not None and r2 - prev < 0.01:
            star = "  <- saturated (+%.3f R2 for one more head)" % (r2 - prev)
        print(f"{H:>2} {bits:>5.0f} {r2:>8.3f} {e_rs:>8.4f} {e_sp:>9.4f} "
              f"{len(cnt):>11,}{star}")
        prev = r2
    print("\n[geo-cap] READ: R2 still climbing = too few heads. E-spread is the legibility "
          "number -- a cell spanning >0.05 in expected score is not one concept.")


if __name__ == "__main__":
    main()
