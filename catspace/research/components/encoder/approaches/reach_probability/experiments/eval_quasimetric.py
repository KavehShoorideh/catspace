#!/usr/bin/env python
"""eval_quasimetric.py -- does the learned geometry actually measure plies?

A quasimetric trained with QRL can look healthy on its own losses and still be useless as a
ruler, so every claim it needs to support is measured separately here, on HELD-OUT GAMES:

  1 CALIBRATION   adjacent (1-ply) distance should sit at the step cost. QRL's Theorem 3 says a
                  relaxed constraint recovers the true value only up to a known factor (1+eps),
                  so the reported number is divided by (1+eps) before being read as plies --
                  otherwise everything reads ~25% long and looks broken when it is correct.
  2 ODOMETER      d(z_t -> z_{t+k}) against k. Rank correlation is the honest summary; the
                  fitted slope says whether one unit really is one ply.
  3 ASYMMETRY     d(t -> t+k) must EXCEED d(t+k -> t). Chess is not reversible, and a symmetric
                  answer means the head has collapsed to a metric.
  4 AXIOMS        d(z,z) == 0 and the triangle inequality, checked on REAL triples rather than
                  trusted from the construction.
  5 GROUND TRUTH  syzygy DTZ is an exact ply ruler where it exists. Among tablebase positions in
                  the same game, the drop in DTZ is a true ply count, and the learned distance
                  should track it. This is the only external check available.

    .venv/bin/python -m ...eval_quasimetric --ckpt artifacts/experiments/cqm1_latest.pt --codes <npz>
"""
from __future__ import annotations

import argparse

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--codes", required=True)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--k-max", type=int, default=24)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from scipy.stats import spearmanr
    from catspace.research.components.encoder.approaches.reach_probability.experiments \
        .concept_quasimetric import ConceptQuasimetric
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    A = ck["args"] if isinstance(ck.get("args"), dict) else vars(ck["args"])
    z = np.load(args.codes, allow_pickle=True)
    g, s = z["g"].astype(np.int64), z["s"].astype(np.int64)
    dev = args.device
    g_book = torch.from_numpy(z["g_book"]).to(dev); s_book = torch.from_numpy(z["s_book"]).to(dev)
    n, H = g.shape
    d_state = H * g_book.shape[-1] + 64 * s_book.shape[-1]
    G = torch.from_numpy(g).to(dev); S = torch.from_numpy(s).to(dev)
    ar_h = torch.arange(H, device=dev)

    def state(rows):
        gi = G[rows]; si = S[rows]
        zq = torch.cat([g_book[ar_h[None, :], gi].reshape(len(rows), -1),
                        s_book[si].reshape(len(rows), -1)], -1)
        lo = zq.min(-1, keepdim=True).values; hi = zq.max(-1, keepdim=True).values
        return (zq - lo) / (hi - lo).clamp(min=1e-5)

    net = ConceptQuasimetric(d_state, d_iqe=A["d_iqe"], components=A["components"],
                             hidden=A["hidden"], leak_beta=A["leak_beta"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval()

    tr = T.build(n_human=0, n_sf=A["games"], seed=A["seed"], cache=True,
                 max_plies=A["max_plies"], n_piecedown=A["n_piecedown"])
    gor = tr.game_of_row()
    rs = np.random.default_rng(A["seed"]); perm = rs.permutation(len(tr.start))
    nt = max(1, int(0.1 * len(tr.start)))
    test_g = np.zeros(len(tr.start), bool); test_g[perm[:nt]] = True
    is_test = test_g[gor]

    rng = np.random.default_rng(0)
    cand = np.flatnonzero(is_test)
    cand = cand[cand + args.k_max < n]
    cand = cand[gor[cand + args.k_max] == gor[cand]]
    rows = cand[rng.choice(len(cand), min(args.n, len(cand)), replace=False)]
    R = torch.from_numpy(rows).to(dev)
    scale = 1.0 + args.eps

    def dist(a, b):
        out = []
        with torch.no_grad():
            for i in range(0, len(a), args.batch):
                out.append(net.dist(state(a[i:i + args.batch]), state(b[i:i + args.batch])))
        return torch.cat(out).cpu().numpy()

    print(f"\n  {args.ckpt.split('/')[-1]} step {ck.get('step')} | {len(rows):,} held-out rows")
    print(f"  distances divided by (1+eps)={scale:.2f} per QRL Theorem 3\n")

    d1 = dist(R, R + 1) / scale
    print(f"  1 CALIBRATION   adjacent d = {d1.mean():.3f} +/- {d1.std():.3f} "
          f"(target 1.000)   median {np.median(d1):.3f}")
    print(f"                  within 2x of target: {100*((d1>0.5)&(d1<2)).mean():.1f}% of pairs")

    ks = [1, 2, 3, 4, 6, 8, 12, 16, args.k_max]
    dk = {k: dist(R, R + k) / scale for k in ks}
    print(f"\n  2 ODOMETER      k:  " + "  ".join(f"{k:>5}" for k in ks))
    print(f"                  d:  " + "  ".join(f"{dk[k].mean():>5.2f}" for k in ks))
    kk = np.concatenate([np.full(len(rows), k) for k in ks])
    dd = np.concatenate([dk[k] for k in ks])
    rho = spearmanr(kk, dd).statistic
    slope = np.polyfit(kk, dd, 1)[0]
    print(f"                  rank correlation with k: {rho:+.3f}   fitted slope "
          f"{slope:.3f} units/ply")

    dback = dist(R + args.k_max, R) / scale
    fwd = dk[args.k_max]
    print(f"\n  3 ASYMMETRY     forward d(t -> t+{args.k_max}) {fwd.mean():.2f}   "
          f"reverse d(t+{args.k_max} -> t) {dback.mean():.2f}   "
          f"reverse/forward {dback.mean()/max(fwd.mean(),1e-9):.2f}x")
    print(f"                  reverse > forward on {100*(dback>fwd).mean():.1f}% of pairs "
          f"(un-capturing is impossible, so this should be near 100%)")

    dself = dist(R, R)
    a, b, c = R[:2000], R[2000:4000], R[4000:6000]
    m = min(len(a), len(b), len(c))
    dab = dist(a[:m], b[:m]); dbc = dist(b[:m], c[:m]); dac = dist(a[:m], c[:m])
    viol = (dac > dab + dbc + 1e-4)
    print(f"\n  4 AXIOMS        d(z,z) max {np.abs(dself).max():.2e}")
    print(f"                  triangle violations on real triples: {100*viol.mean():.3f}% "
          f"({int(viol.sum())} of {m})")

    # ---- 5 ground truth: syzygy DTZ ---------------------------------------------------
    try:
        sz = np.load(paths.derived(f"syzygy_wdl_{n}.npz"))
        dtz = sz["syz_dtz"].astype(np.int64)
        has = dtz > 0
        pairs = []
        for k in (2, 4, 8):
            cand2 = np.flatnonzero(is_test & has)
            cand2 = cand2[cand2 + k < n]
            cand2 = cand2[(gor[cand2 + k] == gor[cand2]) & has[cand2 + k]]
            if len(cand2) > 500:
                sel = cand2[rng.choice(len(cand2), min(3000, len(cand2)), replace=False)]
                dd2 = dist(torch.from_numpy(sel).to(dev),
                           torch.from_numpy(sel + k).to(dev)) / scale
                drop = (dtz[sel] - dtz[sel + k]).astype(float)
                pairs.append((k, len(sel), float(spearmanr(drop, dd2).statistic),
                              float(dd2.mean())))
        print(f"\n  5 GROUND TRUTH  syzygy DTZ, tablebase positions in held-out games")
        if pairs:
            for k, m2, r, dm in pairs:
                print(f"                  k={k:>2}  n={m2:>5,}  learned d {dm:5.2f}  "
                      f"rank corr with the true DTZ drop {r:+.3f}")
        else:
            print(f"                  too few tablebase pairs in held-out games to score")
    except FileNotFoundError:
        print(f"\n  5 GROUND TRUTH  syzygy_wdl_{n}.npz absent -- skipped")


if __name__ == "__main__":
    main()
