#!/usr/bin/env python
"""experiments/train_dtm_cnn.py -- a PLAIN CNN value net (no quasimetric) that regresses DTM from the board,
trained on the SAME data as the field (Kaveh 2026-07-21: "the test has got to be fair, against an MCTS
trained on the SAME data"). This is the fair baseline for the coarse-navigation experiment: field-guided MCTS
vs an MCTS whose value is this plain net -- both trained on dtm_endgame -- isolates whether the FIELD (and its
quasimetric structure) specifically helps, versus any learned value. Also reports the CNN's own spearman(d,DTM)
by piece count, i.e. is a plain regressor a better DTM predictor than the quasimetric field (0.53 @6-piece)?
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes
from catspace.nn.fb import pick_device
from catspace.tracking import track_run


class DTMNet(nn.Module):
    def __init__(self, c=20, w=96):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(c, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU())
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w), nn.ReLU(),
            nn.Conv2d(w, w, 3, padding=1), nn.BatchNorm2d(w)) for _ in range(4)])
        self.head = nn.Sequential(nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 1))

    def forward(self, x):
        h = self.stem(x)
        for b in self.blocks:
            h = torch.relu(h + b(h))
        return self.head(h.mean((2, 3)))[:, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--out", default="data/derived/sep/dtm_cnn.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from contextlib import ExitStack
    _stack = ExitStack()
    trk = _stack.enter_context(track_run("dtm_cnn", args, run_name=Path(args.out).stem))
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dz = np.load(args.dtm_npz); dtm = np.asarray(dz["dtm"]).astype(np.float32)
    P, M = np.asarray(dz["packed"]), np.asarray(dz["meta"])
    ok = np.flatnonzero(dtm > 0)
    planes = feature_planes(P[ok], M[ok]); dtm_ok = dtm[ok]
    pc = np.array([len(board_from_packed(P[i], M[i]).piece_map()) for i in ok])
    tr = np.flatnonzero(rng.random(len(ok)) < 0.85); te = np.setdiff1d(np.arange(len(ok)), tr)

    net = DTMNet(c=planes.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    for s in range(args.steps):
        idx = tr[rng.integers(0, len(tr), args.batch)]
        x = torch.from_numpy(planes[idx]).to(dev); y = torch.from_numpy(dtm_ok[idx] / args.scale).to(dev)
        net.train(); pred = net(x)
        loss = torch.nn.functional.smooth_l1_loss(pred, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 800 == 0:
            print(f"  step {s} loss {float(loss.detach()):.4f}", flush=True)
            trk.metrics(dict(loss=float(loss.detach())), step=s)
        if s > 0 and s % 1000 == 0:                       # step-suffixed checkpoint ladder
            op = Path(args.out)
            torch.save({"state": net.state_dict(), "c": planes.shape[1], "scale": args.scale},
                       op.with_name(f"{op.stem}_step{s}{op.suffix}"))
    net.eval()
    print(f"VERDICT DTM_CNN steps={args.steps} -- plain-CNN spearman(pred, DTM) by piece count (held-out):")
    with torch.no_grad():
        for k in (3, 4, 6):
            sel = te[pc[te] == k]
            if len(sel) < 50:
                continue
            sel = sel[rng.permutation(len(sel))[:2000]]
            p = net(torch.from_numpy(planes[sel]).to(dev)).cpu().numpy()
            rho = spearmanr(p, dtm_ok[sel]).correlation
            print(f"  {k}-piece: spearman={rho:+.3f}   (field distilled: 3p .88 4p .71 6p .53)")
            trk.metrics({f"spearman_{k}p": float(rho)})
    torch.save({"state": net.state_dict(), "c": planes.shape[1], "scale": args.scale,
                "args": vars(args)}, args.out)
    print(f"  saved {args.out}  [{time.time()-t0:.0f}s]")
    _stack.close()


if __name__ == "__main__":
    main()
