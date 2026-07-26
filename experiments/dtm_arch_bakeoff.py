#!/usr/bin/env python
"""experiments/dtm_arch_bakeoff.py -- DISTANCE-TO-MATE architecture bake-off (Kaveh
2026-07-25 correction: 'we don't want a policy target -- policy comes from the planner
one layer above. what we want is distance to mate accurately'). Keeps the quasimetric
philosophy: the field outputs distance-to-mate; the planner sits on top.

The middlegame failure, in distance terms: a middlegame reads ~20 from the mate bank when
the truth is 50-100+ plies -- the learned distance UNDERESTIMATES and SATURATES at long
range. There is NO middlegame DTM label (no tablebase >6 pieces), so the decisive testable
precondition is EXTRAPOLATION: train each backbone to predict DTM on SHORT distances only,
then measure whether it correctly orders LONGER distances it never trained on. A backbone
that extrapolates (far-slice spearman stays high) can represent long distance-to-mate; one
that saturates (far spearman collapses) is exactly the '~20 for everything far' failure.
Backbones: transformer (square tokens, Leela-style) vs CNN, at small dims. No policy head.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.nn.fb import pick_device
from experiments.arch_bakeoff import CNNBackbone, TransformerBackbone, eff_rank, tokens


class DTMNet(nn.Module):
    def __init__(self, backbone, d):
        super().__init__()
        self.bb = backbone
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, ids, stm):
        _, pooled = self.bb(ids, stm)
        return self.head(pooled)[:, 0], pooled


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backbone", choices=["xf", "cnn"], default="xf")
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--data", default="data/derived/dtm_endgame_v2.npz")
    ap.add_argument("--train-max-dtm", type=int, default=25,
                    help="train only on DTM <= this; test extrapolation beyond it")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device)
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)

    z = np.load(args.data)
    ids, stm = tokens(z["packed"], z["meta"])
    dtm = z["dtm"].astype(np.float32)
    ok = dtm > 0
    ids, stm, dtm = ids[ok], stm[ok], dtm[ok]
    n = len(dtm)
    T = args.train_max_dtm
    near = dtm <= T
    heldout = rng.random(n) < 0.1
    tr = np.flatnonzero(near & ~heldout)
    te_near = np.flatnonzero(near & heldout)
    te_far = np.flatnonzero(~near)                       # DTM > T: pure extrapolation
    tag = f"{args.backbone}-d{args.d}-L{args.layers}"
    print(f"[dtm-bakeoff {tag}] train(DTM<={T}) {len(tr)} | test-near {len(te_near)} "
          f"| test-far(DTM>{T}) {len(te_far)}", flush=True)

    bb = (TransformerBackbone(args.d, args.layers) if args.backbone == "xf"
          else CNNBackbone(args.d, args.layers))
    net = DTMNet(bb, args.d).to(dev)
    npar = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    tgt = torch.from_numpy(np.log1p(dtm)).to(dev)        # log target: DTM spans 1..200

    def feed(idx):
        return (torch.from_numpy(ids[idx]).to(dev), torch.from_numpy(stm[idx]).to(dev))

    for s in range(args.steps):
        bi = tr[rng.integers(0, len(tr), args.batch)]
        di, ds = feed(bi)
        pred, _ = net(di, ds)
        loss = F.huber_loss(pred, tgt[bi], delta=1.0)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 1000 == 0:
            print(f"  step {s} loss {float(loss):.4f} [{time.time()-t0:.0f}s]", flush=True)

    net.eval()
    def evalslice(idx):
        preds = []
        with torch.no_grad():
            for s in range(0, len(idx), 2048):
                di, ds = feed(idx[s:s + 2048])
                preds.append(net(di, ds)[0].cpu().numpy())
        p = np.concatenate(preds)
        true = dtm[idx]
        sp = spearman(p, true)
        mae = float(np.abs(np.expm1(p) - true).mean())
        return sp, mae
    sp_n, mae_n = evalslice(te_near)
    sp_f, mae_f = evalslice(te_far)
    # embedding effective rank on the FAR slice (long-distance positions)
    with torch.no_grad():
        di, ds = feed(te_far[:1500])
        _, pooled = net.bb(di, ds)
        er = eff_rank(pooled.cpu().numpy())
    print(f"VERDICT DTM {tag}: near[spearman {sp_n:+.3f} MAE {mae_n:.1f}] "
          f"FAR-extrap[spearman {sp_f:+.3f} MAE {mae_f:.1f}] "
          f"| far_eff_rank {er:.1f}/{args.d} | {npar/1e6:.2f}M [{time.time()-t0:.0f}s]",
          flush=True)


if __name__ == "__main__":
    main()
