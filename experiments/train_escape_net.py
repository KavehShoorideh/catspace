#!/usr/bin/env python
"""experiments/train_escape_net.py -- the LEARNED constraint head: board -> predicted black-king
escape volume. The concept scored 0.75 mate-rate as an ORACLE search value on the ladder
(vs 0.12 pure, 0.85 tb-oracle); this net is its play-legal replacement. Small CNN (DTMNet
arch), CPU-friendly. TRAINING_STANDARDS: MLflow, ckpt ladder, args-in-ckpt, held-out verdicts.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as tF

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.nn.features import feature_planes
from catspace.nn.fb import pick_device
from catspace.tracking import track_run
from experiments.train_dtm_cnn import DTMNet


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/escape_data_v1.npz")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scale", type=float, default=8.0)
    ap.add_argument("--out", default="data/derived/sep/escape_net_v1.pt")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from contextlib import ExitStack
    _stack = ExitStack()
    trk = _stack.enter_context(track_run("escape_net", args, run_name=Path(args.out).stem))
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    z = np.load(args.data)
    P, M, Y = z["packed"], z["meta"], z["escape"].astype(np.float32)
    n = len(Y)
    tr = np.flatnonzero(rng.random(n) < 0.9); te = np.setdiff1d(np.arange(n), tr)
    print(f"[data] {n} rows -> {len(tr)} train / {len(te)} held-out", flush=True)

    net = DTMNet(c=20).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for s in range(args.steps):
        idx = tr[rng.integers(0, len(tr), args.batch)]
        x = torch.from_numpy(feature_planes(P[idx], M[idx])).to(dev)
        y = torch.from_numpy(Y[idx] / args.scale).to(dev)
        net.train()
        loss = tF.smooth_l1_loss(net(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 500 == 0:
            print(f"  step {s} loss {float(loss.detach()):.4f}  [{time.time()-t0:.0f}s]", flush=True)
            trk.metrics(dict(loss=float(loss.detach())), step=s)
        if args.ckpt_every and s > 0 and s % args.ckpt_every == 0:
            op = Path(args.out)
            torch.save({"state": net.state_dict(), "c": 20, "scale": args.scale,
                        "target": "escape_volume", "args": vars(args)},
                       op.with_name(f"{op.stem}_step{s}{op.suffix}"))

    net.eval()
    from scipy.stats import spearmanr
    preds = []
    with torch.no_grad():
        for s in range(0, len(te), 1024):
            idx = te[s:s + 1024]
            preds.append(net(torch.from_numpy(feature_planes(P[idx], M[idx])).to(dev)).cpu().numpy())
    pred = np.concatenate(preds) * args.scale
    rho = spearmanr(pred, Y[te]).correlation
    mae = float(np.abs(pred - Y[te]).mean())
    print(f"VERDICT ESCAPE_NET steps={args.steps} held-out spearman={rho:+.3f} MAE={mae:.2f} squares "
          f"(target range {Y.min():.0f}-{Y.max():.0f})  [{time.time()-t0:.0f}s]", flush=True)
    trk.metrics(dict(heldout_spearman=float(rho), heldout_mae=mae))
    torch.save({"state": net.state_dict(), "c": 20, "scale": args.scale,
                "target": "escape_volume", "args": vars(args)}, args.out)
    print(f"saved {args.out}", flush=True)
    _stack.close()


if __name__ == "__main__":
    main()
