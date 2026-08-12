#!/usr/bin/env python
"""student_distill.py -- speedup #5 (Kaveh 2026-08-11 'go on all 5'): a SMALL student network
(64-wide, 2-layer) trained to mimic the champion's six pole distances, for use as the search's
LEAF EVALUATOR. The big trunk stays the authority for the committor bar, concepts, and
training; the student exists to make depth cheap (the horizon autopsy: depth is the disease).

    (tok, glob) -> student trunk -> 6 outputs = [dA(W,D,L), dB(W,D,L)] of the TEACHER (frozen)
Gate: leaf-margin ORDERING agreement with the teacher on held-out move sets (the search only
consumes margins), plus wall-clock speedup.

    .venv/bin/python -m ...student_distill --ckpt <teacher.pt> [--steps 20000]
writes <teacher>_student.pt
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import JepaEncoder


class Student(nn.Module):
    def __init__(self, d=64, layers=2, heads=4):
        super().__init__()
        self.enc = JepaEncoder(d=d, layers=layers, heads=heads)
        self.out = nn.Sequential(nn.Linear(d, 128), nn.GELU(), nn.Linear(128, 6))

    def forward(self, tok, glob):
        return self.out(self.enc(tok, glob))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--rows", type=int, default=400_000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    pn = c["pole_names"]
    P = net.poles.poles.detach().float().to(args.device)
    pidx = [pn.index(k) for k in ("WIN", "DRAW", "LOSS")]
    tr = T.build(n_human=0, n_sf=c["games"], seed=c["traj_seed"], max_plies=c["max_plies"],
                 n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    game = tr.game_of_row()
    rng = np.random.default_rng(0)
    fit = np.flatnonzero(np.isin(game, np.flatnonzero(split == 0)))
    fit = fit[rng.choice(len(fit), min(args.rows, len(fit)), replace=False)]
    val = np.flatnonzero(np.isin(game, np.flatnonzero(split == 1)))
    val = val[rng.choice(len(val), 20000, replace=False)]

    def teacher(rows):
        Y = []
        for a in range(0, len(rows), 4096):
            rr = rows[a:a + 4096]
            with torch.no_grad():
                z = net.encode_q(torch.from_numpy(tr.tok[rr].astype(np.int64)).to(args.device),
                                 torch.from_numpy(tr.glob[rr].astype(np.float32)).to(args.device))
                DA = torch.stack([net.dA(z, P[[k]].expand(len(z), -1)) for k in pidx], 1)
                DB = torch.stack([net.dB(z, P[[k]].expand(len(z), -1)) for k in pidx], 1)
            Y.append(torch.cat([DA, DB], 1).float().cpu())
        return torch.cat(Y)

    print("[student] caching teacher targets...", flush=True)
    Yf, Yv = teacher(fit), teacher(val)
    mu, sd = Yf.mean(0), Yf.std(0).clamp(min=1e-6)
    stu = Student().to(args.device)
    n_par = sum(p_.numel() for p_ in stu.parameters())
    print(f"[student] {n_par/1e6:.2f}M params vs teacher ~1.2M-trunk+heads", flush=True)
    opt = torch.optim.Adam(stu.parameters(), lr=1e-3)
    for step in range(args.steps):
        sel = rng.integers(0, len(fit), args.batch)
        tok = torch.from_numpy(tr.tok[fit[sel]].astype(np.int64)).to(args.device)
        glob = torch.from_numpy(tr.glob[fit[sel]].astype(np.float32)).to(args.device)
        yh = stu(tok, glob)
        loss = torch.nn.functional.mse_loss(yh, ((Yf[sel] - mu) / sd).to(args.device))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % 4000 == 0:
            print(f"[student] step {step+1}: mse {float(loss):.4f}", flush=True)
    with torch.no_grad():
        YH = []
        for a in range(0, len(val), 4096):
            tok = torch.from_numpy(tr.tok[val[a:a+4096]].astype(np.int64)).to(args.device)
            glob = torch.from_numpy(tr.glob[val[a:a+4096]].astype(np.float32)).to(args.device)
            YH.append((stu(tok, glob).cpu() * sd + mu))
        YH = torch.cat(YH)
    for j, nm in enumerate(["dA_W", "dA_D", "dA_L", "dB_W", "dB_D", "dB_L"]):
        r = np.corrcoef(YH[:, j].numpy(), Yv[:, j].numpy())[0, 1]
        print(f"[student] {nm}: corr with teacher {r:.3f}")
    base = args.ckpt[:-3] if args.ckpt.endswith(".pt") else args.ckpt
    torch.save({"state_dict": stu.state_dict(), "mu": mu, "sd": sd}, base + "_student.pt")
    print(f"[student] saved {base}_student.pt")


if __name__ == "__main__":
    main()
