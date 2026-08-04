#!/usr/bin/env python
"""catspace/research/components/planner/approaches/reach_field/experiments/eval_agentive_lift.py -- did agentive fine-tuning improve the field's
prediction of OUR OWN games? BCE on held-out agentive rows (game_id%10==0, in_now
excluded): warm-start ckpt vs fine-tuned ckpt, paired bootstrap by game."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reachability_field.experiments.train_reach_head import ReachHead                      # noqa: E402
from catspace.io import paths


def bce_rows(ckpt, d, rows, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False)
    d_opp = 17 if any(k.startswith("state.0") and v.shape[1] > 84
                      for k, v in ck["state_dict"].items()) else 0
    m = ReachHead(d_phi=64, d_z=16, d_opp=d_opp).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval()
    bank = torch.as_tensor(d["bank"], dtype=torch.float32, device=dev)
    out = np.zeros(len(rows))
    with torch.no_grad():
        gh, _ = m.goal_embs(bank)
        for i0 in range(0, len(rows), 1024):
            r = rows[i0:i0+1024]
            f = torch.as_tensor(d["phi"][r], dtype=torch.float32, device=dev)
            B = len(r)
            zs = torch.zeros(B, 16, device=dev)
            ctx = [torch.stack([(torch.as_tensor(d["elo_self"][r], device=dev) - 1500) / 400,
                                (torch.as_tensor(d["elo_oppo"][r], device=dev) - 1500) / 400,
                                torch.ones(B, device=dev), torch.ones(B, device=dev)], -1)]
            if d_opp:
                ctx.append(torch.zeros(B, 17, device=dev))
            sh, _ = m.state_embs(f, zs, torch.cat(ctx, 1).float())
            logit = sh @ gh.T * m.scale + m.b_hit
            y = torch.as_tensor(d["hit"][r], dtype=torch.float32, device=dev)
            mask = torch.as_tensor(d["in_now"][r], device=dev) == 0
            nll = torch.nn.functional.binary_cross_entropy_with_logits(
                logit, y, reduction="none")
            out[i0:i0+1024] = ((nll * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=paths.reach("agentive_v1.npz"))
    ap.add_argument("--base", default=paths.experiment("reach_v3_full_latest.pt"))
    ap.add_argument("--tuned", default=paths.experiment("reach_v4_agentive_latest.pt"))
    args = ap.parse_args()
    dev = "cpu"
    d = dict(np.load(args.data, allow_pickle=True))
    gid = d["game_id"]
    rows = np.flatnonzero(gid % 10 == 0)
    print(f"eval rows: {len(rows)} (games {len(np.unique(gid[rows]))})")
    b0 = bce_rows(args.base, d, rows, dev)
    b1 = bce_rows(args.tuned, d, rows, dev)
    diff = b0 - b1                                     # >0 = tuned better
    games = gid[rows]
    ug = np.unique(games)
    boots = []
    rng = np.random.default_rng(0)
    for _ in range(2000):
        gs = rng.choice(ug, len(ug))
        boots.append(np.mean(np.concatenate([diff[games == g] for g in gs])))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"VERDICT agentive-lift: BCE base {b0.mean():.5f} -> tuned {b1.mean():.5f} | "
          f"lift {diff.mean():+.5f} nats/row CI[{lo:+.5f},{hi:+.5f}] "
          f"({'PASS' if lo > 0 else 'not significant'}, game-bootstrap)")


if __name__ == "__main__":
    main()
