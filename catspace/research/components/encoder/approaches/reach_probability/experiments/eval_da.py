#!/usr/bin/env python
"""eval_da.py -- THE dA scoreboard (Kaveh 2026-08-08: "dA should include all the info we need;
dB should become redundant eventually"). One command, every number that defines "best dA":

  1. odometer     corr(dA -> own-outcome pole, remaining plies)          [length meaning]
  2. dA committor CE/top1 of softmax over dA to the 3 poles              [who-wins in A]
  3. dB committor same, for reference                                    [the incumbent]
  4. redundancy   top1 of dB given dA's prediction already made          [does B still add?]
  5. siblings     dA/dB keeps-the-win margin over random (wdl shards)    [move choice]

    .venv/bin/python -m ...eval_da --ckpt <ckpt.pt> [--n 2000]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from catspace.io import paths
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def fit_tau_ce(D, y):
    best = (None, float("inf"))
    for tau in np.geomspace(1e-3, 100, 300):
        ce = torch.nn.functional.cross_entropy(-D / tau, y).item()
        if ce < best[1]:
            best = (float(tau), ce)
    top1 = float(((-D / best[0]).argmax(1) == y).float().mean())
    return best[0], best[1], top1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    tr = T.build(n_human=0 if c.get("sf_only") else c["games"] // 2,
                 n_sf=c["games"] if c.get("sf_only") else c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    game = tr.game_of_row()
    y_all = tr.outcome_of_row_white()
    glen = np.repeat(tr.length, tr.length)
    remain = glen - tr.ply_of_row() - 1
    rows = np.flatnonzero(np.isin(game, np.flatnonzero(split == 1)) & (y_all >= 0))
    rng = np.random.default_rng(0)
    rows = rows[rng.choice(len(rows), min(args.n, len(rows)), replace=False)]
    y = torch.from_numpy(y_all[rows].astype(np.int64))
    pn = c["pole_names"]
    P = net.poles.poles.detach().float().to(args.device)
    pidx = [pn.index(k) for k in ("WIN", "DRAW", "LOSS")]
    with torch.no_grad():
        z = net.encode_q(torch.from_numpy(tr.tok[rows].astype(np.int64)).to(args.device),
                         torch.from_numpy(tr.glob[rows].astype(np.float32)).to(args.device))
        DA = torch.stack([net.dA(z, P[[k]].expand(len(z), -1)) for k in pidx], 1).float().cpu()
        DB = torch.stack([net.dB(z, P[[k]].expand(len(z), -1)) for k in pidx], 1).float().cpu()

    own = DA[torch.arange(len(y)), y].numpy()
    msk = remain[rows] < 100
    print(f"[1 odometer ] corr(dA->own pole, remaining) = "
          f"{np.corrcoef(own[msk], remain[rows][msk])[0, 1]:.3f}")
    ta, ca, t1a = fit_tau_ce(DA, y)
    tb, cb, t1b = fit_tau_ce(DB, y)
    print(f"[2 dA commit] tau {ta:7.3f}  CE {ca:.4f}  top1 {t1a:.3f}   (uniform 1.0986)")
    print(f"[3 dB commit] tau {tb:7.3f}  CE {cb:.4f}  top1 {t1b:.3f}")
    # 4: does dB still add information once dA has spoken? logistic stack: predict y from
    # dA logits alone vs dA+dB logits (simple ridge multinomial via torch).
    def stack_ce(feats):
        X = torch.cat(feats, 1)
        W = torch.zeros(X.shape[1], 3, requires_grad=True)
        b = torch.zeros(3, requires_grad=True)
        opt = torch.optim.LBFGS([W, b], max_iter=200)
        def cl():
            opt.zero_grad()
            l = torch.nn.functional.cross_entropy(X @ W + b, y) + 1e-3 * W.pow(2).sum()
            l.backward(); return l
        opt.step(cl)
        with torch.no_grad():
            ce = torch.nn.functional.cross_entropy(X @ W + b, y).item()
            t1 = float(((X @ W + b).argmax(1) == y).float().mean())
        return ce, t1
    cea, t1sa = stack_ce([-DA / ta])
    ceab, t1sab = stack_ce([-DA / ta, -DB / tb])
    print(f"[4 redundancy] stacked CE: dA alone {cea:.4f} (top1 {t1sa:.3f})  "
          f"dA+dB {ceab:.4f} (top1 {t1sab:.3f})  -> dB adds {cea - ceab:+.4f} nats")
    # 5: sibling keeps-the-win margins on the first wdl shard
    ddir = paths.derived("wdl_labels")
    sh = [os.path.join(ddir, f) for f in sorted(os.listdir(ddir)) if f.endswith(".npz")]
    if sh:
        pk = np.load(sh[0])
        psel = rng.choice(len(pk["row"]), 800, replace=False)
        keep = {"dA": 0, "dB": 0, "rand": 0}
        kn = 0
        for pi_ in psel:
            a, b2 = int(pk["off"][pi_]), int(pk["off"][pi_ + 1])
            if b2 - a < 3:
                continue
            prow = int(pk["row"][pi_])
            wtm = bool(tr.glob[prow, 0])
            pwin = pk["wdl"][a:b2, 2].astype(float) / 1000.0
            if pwin.max() < 0.75:
                continue
            ks = pwin >= min(0.5, pwin.max() - 0.05)
            with torch.no_grad():
                zc = net.encode_q(
                    torch.from_numpy(pk["tok"][a:b2].astype(np.int64)).to(args.device),
                    torch.from_numpy(pk["glob"][a:b2].astype(np.float32)).to(args.device))
                ow, ot = ("WIN", "LOSS") if wtm else ("LOSS", "WIN")
                dbw = net.dB(zc, P[[pn.index(ow)]].expand(len(zc), -1)).float().cpu().numpy()
                daw = net.dA(zc, P[[pn.index(ow)]].expand(len(zc), -1)).float().cpu().numpy()
                dal = net.dA(zc, P[[pn.index(ot)]].expand(len(zc), -1)).float().cpu().numpy()
            keep["dA"] += int(ks[int(np.argmax(dal - daw))])
            keep["dB"] += int(ks[int(np.argmin(dbw))])
            keep["rand"] += int(ks[int(rng.integers(0, b2 - a))])
            kn += 1
        print(f"[5 siblings ] keeps-the-win over {kn} winning positions: "
              f"dA {keep['dA']/kn:.1%}  dB {keep['dB']/kn:.1%}  random {keep['rand']/kn:.1%}")


if __name__ == "__main__":
    main()
