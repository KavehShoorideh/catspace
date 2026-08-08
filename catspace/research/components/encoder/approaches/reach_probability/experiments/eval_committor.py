#!/usr/bin/env python
"""eval_committor.py -- ABSOLUTE committor calibration, the gate that never existed.

2026-08-08: frozen pole readouts passed the whole battery because every gate measures
RELATIVE structure (routing, walls, ranking). This instrument asks the absolute question:
from the three pole distances at a position, how well is its actual game outcome predicted?
Reported on VAL rows, temperature fitted on the same rows (one scalar -- optimistic, but a
committor that can't beat uniform WITH a fitted temperature is dead beyond argument).

Label POV follows the checkpoint's train_args.basin_pov (white -> outcome_of_row_white).

    .venv/bin/python -m ...eval_committor --ckpt <ckpt.pt>

prints:  [committor] pov=... tau=...  CE ... (uniform 1.0986)  top1 ... (chance ...)
         per-class mean pole distances (the collapse signature is visible here)
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    pov = (c.get("train_args") or {}).get("basin_pov", "mover")
    tr = T.build(n_human=0 if c.get("sf_only") else c["games"] // 2,
                 n_sf=c["games"] if c.get("sf_only") else c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    game = tr.game_of_row()
    y_all = tr.outcome_of_row_white() if pov == "white" else tr.outcome_of_row()
    rows = np.flatnonzero(np.isin(game, np.flatnonzero(split == 1)) & (y_all >= 0))
    rng = np.random.default_rng(0)
    rows = rows[rng.choice(len(rows), min(args.n, len(rows)), replace=False)]
    y = torch.from_numpy(y_all[rows].astype(np.int64))

    dist = net.dB if getattr(net, "split_head", False) else net.iqe
    pn = c["pole_names"]
    P = net.poles.poles.detach().float().to(args.device)
    pidx = [pn.index(k) for k in ("WIN", "DRAW", "LOSS")]
    with torch.no_grad():
        z = net.encode_q(torch.from_numpy(tr.tok[rows].astype(np.int64)).to(args.device),
                         torch.from_numpy(tr.glob[rows].astype(np.float32)).to(args.device))
        D = torch.stack([dist(z, P[[k]].expand(len(z), -1)) for k in pidx], 1).float().cpu()

    best_tau, best_ce = None, float("inf")
    for tau in np.geomspace(1e-3, 100.0, 300):
        ce = torch.nn.functional.cross_entropy(-D / tau, y).item()
        if ce < best_ce:
            best_ce, best_tau = ce, float(tau)
    top1 = float(((-D / best_tau).argmax(1) == y).float().mean())
    chance = float(torch.bincount(y, minlength=3).max()) / len(y)
    print(f"[committor] pov={pov}  tau={best_tau:.3f}  CE {best_ce:.4f} (uniform 1.0986)  "
          f"top1 {top1:.3f} (majority-class {chance:.3f})  n={len(y)}")
    for k, name in enumerate(("W", "D", "L")):
        m = y == k
        if m.any():
            mm = D[m].mean(0)
            print(f"  true={name:1s} (n={int(m.sum()):5d})  mean d -> W {mm[0]:8.3f}  "
                  f"D {mm[1]:8.3f}  L {mm[2]:8.3f}   sigma {D[m].std(0).mean():.3f}")


if __name__ == "__main__":
    main()
