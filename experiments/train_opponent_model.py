#!/usr/bin/env python
"""experiments/train_opponent_model.py -- train the option-A opponent model
(catspace/nn/opponent.py) on move-selection rows (build_move_selection.py): masked CE over
legal moves, cohort = mover's Elo bin. VERDICTs: held-out NLL + top-1 accuracy overall and
per Elo bin (the skill gradient the model must express). TRAINING_STANDARDS: MLflow tracked,
step-suffixed checkpoint ladder, args in checkpoint.
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
from catspace.nn.opponent import OpponentModel
from catspace.tracking import track_run


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/derived/move_selection_v1.npz")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d-tok", type=int, default=128)
    ap.add_argument("--self-layers", type=int, default=2)
    ap.add_argument("--out", default="data/derived/sep/opponent_v1.pt")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    from contextlib import ExitStack
    _stack = ExitStack()
    trk = _stack.enter_context(track_run("opponent_model", args, run_name=Path(args.out).stem))
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    z = np.load(args.data)
    n = len(z["cohort"])
    tr = np.flatnonzero(rng.random(n) < 0.9); te = np.setdiff1d(np.arange(n), tr)
    print(f"[data] {n} rows -> {len(tr)} train / {len(te)} held-out", flush=True)

    net = OpponentModel(d_tok=args.d_tok, n_self_layers=args.self_layers, seed=args.seed).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)

    def batch(idx):
        pl = torch.from_numpy(feature_planes(z["packed"][idx], z["meta"][idx])).to(dev)
        t_ = lambda k, dt=torch.int64: torch.from_numpy(z[k][idx].astype(np.int64)).to(dev)
        return (pl, t_("mv_from"), t_("mv_to"), t_("mv_piece"), t_("mv_capt"),
                t_("n_moves"), t_("cohort"), t_("played"))

    for s in range(args.steps):
        idx = tr[rng.integers(0, len(tr), args.batch)]
        pl, f_, tt, pc, ct, nm, co, y = batch(idx)
        net.train()
        logits = net(pl, f_, tt, pc, ct, nm, co)
        loss = tF.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 200 == 0:
            print(f"  step {s} loss {float(loss.detach()):.4f}  [{time.time()-t0:.0f}s]", flush=True)
            trk.metrics(dict(loss=float(loss.detach())), step=s)
        if args.ckpt_every and s > 0 and s % args.ckpt_every == 0:
            op = Path(args.out)
            torch.save({"state": net.state_dict(), "config": net.config, "args": vars(args)},
                       op.with_name(f"{op.stem}_step{s}{op.suffix}"))

    # ---- held-out eval, per cohort
    net.eval()
    nll, acc, coh_all = [], [], []
    with torch.no_grad():
        for s in range(0, len(te), 512):
            idx = te[s:s + 512]
            pl, f_, tt, pc, ct, nm, co, y = batch(idx)
            logits = net(pl, f_, tt, pc, ct, nm, co)
            lp = tF.log_softmax(logits, dim=1)
            nll.append(-lp[torch.arange(len(y)), y].cpu().numpy())
            acc.append((logits.argmax(1) == y).cpu().numpy())
            coh_all.append(co.cpu().numpy())
    nll = np.concatenate(nll); acc = np.concatenate(acc); coh = np.concatenate(coh_all)
    print(f"VERDICT OPPONENT_V1 steps={args.steps} held-out NLL={nll.mean():.4f} "
          f"top1={acc.mean():.3f} (n={len(nll)})", flush=True)
    for cbin in sorted(set(coh)):
        m = coh == cbin
        if m.sum() >= 200:
            print(f"    elo-bin {cbin:2d}: NLL {nll[m].mean():.4f}  top1 {acc[m].mean():.3f}  (n={m.sum()})",
                  flush=True)
    trk.metrics(dict(heldout_nll=float(nll.mean()), heldout_top1=float(acc.mean())))
    torch.save({"state": net.state_dict(), "config": net.config, "args": vars(args)}, args.out)
    print(f"saved {args.out}  [{time.time()-t0:.0f}s]", flush=True)
    _stack.close()


if __name__ == "__main__":
    main()
