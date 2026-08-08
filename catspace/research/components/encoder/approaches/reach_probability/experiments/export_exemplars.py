#!/usr/bin/env python
"""export_exemplars.py -- save WHITE-POV terminal-exemplar embeddings next to a checkpoint.

2026-08-08: the walls-era default is --poles contrastive, whose committor is BY DESIGN a
readout against real terminal boards, not pole distances (net.poles holds untouched init
buffers -- reading them gave the frozen 45/65/139 eval bar). This exports the readout the
routing gate validated (median distance to terminal exemplars) so the engine and the analysis
board can use it: three distances, softmaxed = the bar.

Classes are WHITE-POV, from mover-POV terminal outcomes + side-to-move at the terminal:
    W (white wins) = mover lost & black to move  |  mover won & white to move
    D (draw)       = draw terminals
    L (black wins) = mover lost & white to move  |  mover won & black to move

    .venv/bin/python -m ...export_exemplars --ckpt <ckpt.pt> [--n 64]
writes <ckpt minus .pt>_exemplars.pt with {"W": (n,d), "D": ..., "L": ..., "meta": {...}}
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=64, help="exemplars per class")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    tr = T.build(n_human=0 if c.get("sf_only") else c["games"] // 2,
                 n_sf=c["games"] if c.get("sf_only") else c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    game = tr.game_of_row()
    t_rows, t_term = tr.terminal_rows()
    in_train = np.isin(game[t_rows], np.flatnonzero(split == 0))
    out_cls = T.TERM_OUTCOME[t_term]                    # 0 mover won / 1 draw / 2 mover lost
    wtm = tr.glob[t_rows, 0].astype(bool)

    masks = {"W": in_train & (((out_cls == 2) & ~wtm) | ((out_cls == 0) & wtm)),
             "D": in_train & (out_cls == 1),
             "L": in_train & (((out_cls == 2) & wtm) | ((out_cls == 0) & ~wtm))}
    rng = np.random.default_rng(0)
    out = {"meta": {"ckpt": args.ckpt, "n": args.n, "classes": "white-POV W/D/L"}}
    for k, m in masks.items():
        rows = t_rows[m]
        if len(rows) == 0:
            raise SystemExit(f"[exemplars] class {k} has NO terminal rows -- refusing to export")
        rows = rows[rng.choice(len(rows), min(args.n, len(rows)), replace=False)]
        with torch.no_grad():
            z = net.encode_q(torch.from_numpy(tr.tok[rows].astype(np.int64)).to(args.device),
                             torch.from_numpy(tr.glob[rows].astype(np.float32)).to(args.device))
        out[k] = z.detach().float().cpu()
        print(f"[exemplars] {k}: {len(rows)} terminals (of {int(m.sum())} available)")
    # TEMPERATURE CALIBRATION (standard temperature scaling, one scalar): exemplar distances
    # live on a ~0.3 scale here while the basin tau of 5 was chosen for pole distances --
    # softmax(-d/5) would read a dead 1/3 bar. Fit tau on VAL rows vs actual white-POV
    # outcomes by CE; tau is data, so it ships inside the sidecar, versioned with the model.
    val_rows = np.flatnonzero(np.isin(game, np.flatnonzero(split == 1)))
    y_white = tr.outcome_of_row_white()[val_rows]
    keep = y_white >= 0
    val_rows, y_white = val_rows[keep], y_white[keep]
    sel = rng.choice(len(val_rows), min(2000, len(val_rows)), replace=False)
    val_rows, y_white = val_rows[sel], y_white[sel]
    dist = net.dB if getattr(net, "split_head", False) else net.iqe
    with torch.no_grad():
        zv = net.encode_q(torch.from_numpy(tr.tok[val_rows].astype(np.int64)).to(args.device),
                          torch.from_numpy(tr.glob[val_rows].astype(np.float32)).to(args.device))
        cols = []
        for k in ("W", "D", "L"):
            E = out[k].to(args.device)
            dd = torch.stack([dist(zv, E[e].expand(len(zv), -1)) for e in range(len(E))], 1)
            cols.append(dd.median(1).values)
        Dv = torch.stack(cols, 1).float().cpu()          # (n, 3) class order W/D/L = y 0/1/2
    yt = torch.from_numpy(y_white.astype(np.int64))
    best_tau, best_ce = None, float("inf")
    for tau in np.geomspace(1e-3, 10.0, 200):
        ce = torch.nn.functional.cross_entropy(-Dv / tau, yt).item()
        if ce < best_ce:
            best_ce, best_tau = ce, float(tau)
    base_ce = float(np.log(3.0))
    acc = float(((-Dv / best_tau).argmax(1) == yt).float().mean())
    print(f"[calib] tau={best_tau:.4f}  val CE {best_ce:.4f} (uniform {base_ce:.4f})  "
          f"top1 {acc:.3f} on {len(yt)} rows")
    out["tau"] = best_tau
    out["meta"]["val_ce"] = best_ce
    out["meta"]["val_top1"] = acc
    path = args.ckpt[:-3] + "_exemplars.pt" if args.ckpt.endswith(".pt") else args.ckpt + "_exemplars.pt"
    torch.save(out, path)
    print(f"[exemplars] wrote {path}")


if __name__ == "__main__":
    main()
