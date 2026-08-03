#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/distill_finetune.py -- can the field LEARN 6-piece DTM from targets? (Kaveh 2026-07-21:
"distillation is the way".) Extrapolation gave the field only spearman 0.21 on 6-piece (never trained there).
Here we DISTILL distance targets into the field -- regress d(F(s), MATE_W) -> DTM/scale on the endgame set
(all piece counts, so <=5 accuracy is preserved and 6-piece is added) -- and re-measure the 6-piece spearman.

If 6-piece alignment jumps toward the target quality, the architecture CAN represent 6-piece reachability; it
just needs targets extrapolation can't supply (validating distillation as the mechanism -- exact DTM here for
the toy; a search-backed teacher beyond the tablebase). If it stays low, the wall is the architecture/horizon.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("iqe_nucleus_gn.pt"))
    ap.add_argument("--dtm-npz", default=paths.derived("dtm_endgame.npz"))
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--scale", type=float, default=20.0)
    ap.add_argument("--out", default=paths.sep("nucleus_distilled.pt"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    fb, extra = load_ckpt(Path(args.field), dev); fb.train()
    zg = torch.load(args.field, map_location="cpu", weights_only=False)["zgoals"]
    MATE_W = (zg["MATE_W"].detach().float() if torch.is_tensor(zg["MATE_W"])
              else torch.tensor(np.asarray(zg["MATE_W"], np.float32))).to(dev)[None, :]
    om = torch.from_numpy(omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]).to(dev)

    dz = np.load(args.dtm_npz); dtm = np.asarray(dz["dtm"]).astype(np.float32)
    P, M = np.asarray(dz["packed"]), np.asarray(dz["meta"])
    ok = np.flatnonzero(dtm > 0)
    pc = np.array([len(board_from_packed(P[i], M[i]).piece_map()) for i in ok])
    planes_all = feature_planes(P[ok], M[ok])                    # precompute planes
    dtm_ok = dtm[ok]

    def dist(idx):
        t = torch.from_numpy(planes_all[idx]).to(dev)
        F = fb.embed_F(t, om[None, :].repeat(len(idx), 1))
        return fb.distance_matrix(F, MATE_W)[:, 0]

    def spear_by_pc():
        fb.eval(); out = {}
        with torch.no_grad():
            for k in (3, 4, 6):
                sel = np.flatnonzero(pc == k)
                if len(sel) < 50:
                    continue
                sel = sel[rng.permutation(len(sel))[:2000]]
                d = dist(sel).cpu().numpy()
                out[k] = spearmanr(d, dtm_ok[sel]).correlation
        fb.train(); return out

    before = spear_by_pc()
    opt = torch.optim.Adam(fb.parameters(), lr=args.lr)
    for s in range(args.steps):
        idx = rng.integers(0, len(ok), args.batch)
        d = dist(idx)
        tgt = torch.from_numpy(dtm_ok[idx] / args.scale).to(dev)
        loss = torch.nn.functional.smooth_l1_loss(d, tgt)
        opt.zero_grad(); loss.backward(); opt.step()
        if s % 300 == 0:
            print(f"  step {s} distill-loss {float(loss):.4f}", flush=True)
    after = spear_by_pc()
    print(f"VERDICT DISTILL_FINETUNE field={Path(args.field).stem} steps={args.steps} (spearman(d,DTM) by piece count)")
    for k in sorted(set(before) | set(after)):
        tag = "<=5 (in-dist)" if k <= 5 else "6-piece (was EXTRAPOLATION 0.21)"
        print(f"  {k}-piece: before {before.get(k, float('nan')):+.3f} -> after {after.get(k, float('nan')):+.3f}   [{tag}]")
    torch.save({"state_dict": fb.state_dict(), "config": extra.get("config", {}) if isinstance(extra, dict) else {},
                "step": -1, "zgoals": zg}, args.out)
    print(f"  saved {args.out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
