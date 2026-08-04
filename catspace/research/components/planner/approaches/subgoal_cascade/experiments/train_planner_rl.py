#!/usr/bin/env python
"""catspace/research/components/planner/approaches/subgoal_cascade/experiments/train_planner_rl.py -- THE RL PLANSELECTOR v1 (Kaveh: 'the planner RL --
I want them in the loop too'). Contextual outcome model over plan choices: from every
recorded game, (probe observation at plan decision, plan taken, game outcome) tuples ->
tiny MLP P(win | obs, plan). Deployed by make_planner when a ckpt exists: plans are scored
by expected outcome, argmax wins (deterministic rules remain the fallback and the goal
machinery is unchanged -- the RL chooses the PLAN TYPE). Trains per improvement-loop round
on all accumulated data; epsilon-greedy exploration stays in the selector."""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from catspace.io import paths


from catspace.research.components.planner.approaches.subgoal_cascade.src.plan_rl import (
    OBS_KEYS, PLANS, PlanNet, featurize)   # shipping definitions live in src/


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-glob", default=paths.experiment("*results*.jsonl"))
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--out", default=paths.sep("planner_rl_r0.pt"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    X, A, Y = [], [], []
    for f in glob.glob(args.results_glob):
        for ln in Path(f).read_text().splitlines():
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            y = 1.0 if r.get("mate") else 0.0
            for snap in r.get("probes", []):
                plan = snap.get("plan", "direct").split("->")[0]
                if plan not in PLANS:
                    continue
                X.append(featurize(snap)); A.append(PLANS.index(plan)); Y.append(y)
    if len(X) < 50:
        print(f"VERDICT PLANNER_RL insufficient data (n={len(X)}) -- no ckpt", flush=True)
        return
    X = np.stack(X); A = np.array(A); Y = np.array(Y, np.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    Xn = (X - mu) / sd
    print(f"[data] {len(X)} (obs, plan, outcome) tuples; base rate {Y.mean():.2f}", flush=True)
    net = PlanNet(seed=args.seed)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(args.seed)
    oh = np.eye(len(PLANS), dtype=np.float32)
    for s in range(args.steps):
        bi = rng.integers(0, len(X), min(256, len(X)))
        logit = net(torch.from_numpy(Xn[bi]), torch.from_numpy(oh[A[bi]]))
        loss = nn.functional.binary_cross_entropy_with_logits(logit, torch.from_numpy(Y[bi]))
        opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.from_numpy(Xn), torch.from_numpy(oh[A]))).numpy()
    from scipy.stats import spearmanr
    sp = spearmanr(p, Y).correlation
    print(f"VERDICT PLANNER_RL n={len(X)} fit-spearman {sp:+.3f} "
          f"[{time.time()-t0:.0f}s]", flush=True)
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                "obs_keys": OBS_KEYS, "plans": PLANS}, args.out)
    print(f"saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
