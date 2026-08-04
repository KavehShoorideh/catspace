#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/reachability_field/experiments/test_field_fullgame.py -- TEST the trained full-board field (Kaveh: "then test it").
Evaluates a ClockField checkpoint on HELD-OUT VAL games (same game-level split as training) and
reports the metrics that matter for a metastability committor + quasimetric field:

  * COMMITTOR CALIBRATION -- bin the predicted committor into deciles; empirical win-rate per bin
    should track the bin's mean prediction (reliability curve). ECE = mean |pred - empirical|.
  * COMMITTOR-MAE + WIN/LOSS SEPARATION -- value calibration vs actual W/D/L; does c separate the
    win basin from the loss basin.
  * TABLEBASE AGREEMENT -- on <=7-piece val positions (exact WDL), committor vs ground truth.
  * MULTI-GOAL PAIR-ORDER (Spearman) + eff_rank(phi) -- geometry quality + collapse gate.
"""
from __future__ import annotations

import argparse, sys
from pathlib import Path

import numpy as np
import torch

from catspace.research.components.encoder.approaches.reachability_field.experiments.train_clock_field import ClockField
from catspace.research.components.encoder.approaches.reachability_field.experiments.train_field_fullgame import load_field_data
from catspace.research.components.encoder.approaches.reachability_field.experiments.arch_bakeoff import eff_rank
from catspace.research.tools.training_infra.train.scaffold import resolve_device
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.experiment("field_fullgame_latest.pt"))
    ap.add_argument("--data", default=paths.derived("field_fullgame.npz"))
    ap.add_argument("--val-frac", type=float, default=0.1); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    dev = resolve_device(args.device)

    payload = torch.load(args.ckpt, map_location=dev, weights_only=False)
    cfg = payload.get("cfg", {"d": 64, "ch": 128, "blocks": 8, "in_planes": 112})
    net = ClockField(cfg["d"], ch=cfg["ch"], blocks=cfg["blocks"], in_planes=cfg.get("in_planes", 112)).to(dev)
    net.load_state_dict(payload["state_dict"]); net.eval()
    D = load_field_data(args.data, val_frac=args.val_frac, seed=args.seed)
    va = D["val_idx"]; planes = D["planes"]; ending = D["ending"]; dtz = D["dtz"]
    print(f"[test-field] ckpt {args.ckpt} (step {payload.get('step')}) | val positions {len(va)}", flush=True)

    def fp(idx):
        return torch.from_numpy(planes[idx].astype(np.float32)).to(dev)

    with torch.no_grad():
        # committor over all val positions (batched)
        comm = np.concatenate([net.committor(fp(va[i:i+2048])).cpu().numpy() for i in range(0, len(va), 2048)])
    actual = np.where(ending[va] == 0, 1.0, np.where(ending[va] == 5, 0.0, 0.5)).astype(np.float32)
    win = (ending[va] == 0).astype(np.float32)                    # binary win indicator (committor target)
    mae = float(np.abs(comm - actual).mean())
    sep = float(comm[ending[va] == 0].mean() - comm[ending[va] == 5].mean())

    # CALIBRATION reliability curve (deciles) + ECE against the binary WIN outcome
    print("\nCOMMITTOR CALIBRATION (predicted P(win) vs empirical win-rate, val):")
    edges = np.quantile(comm, np.linspace(0, 1, 11))
    edges[-1] += 1e-6; ece = 0.0
    for b in range(10):
        m = (comm >= edges[b]) & (comm < edges[b + 1])
        if m.sum() < 5:
            continue
        pred, emp = comm[m].mean(), win[m].mean()
        ece += m.mean() * abs(pred - emp)
        bar = "#" * int(round(emp * 30))
        print(f"  bin{b} pred {pred:5.2f} emp {emp:5.2f} n{int(m.sum()):5d} {bar}")
    print(f"  ECE (win) {ece:.3f} | committor-MAE(W/D/L score) {mae:.3f} | win-loss sep {sep:+.3f}")

    # TABLEBASE agreement on <=7-piece val positions (exact ending grounded in Stage C)
    tb = va[(dtz[va] >= 0)]
    if len(tb):
        with torch.no_grad():
            ctb = np.concatenate([net.committor(fp(tb[i:i+2048])).cpu().numpy() for i in range(0, len(tb), 2048)])
        print(f"\nTABLEBASE-GROUNDED (<=7p won, n={len(tb)}): committor mean {ctb.mean():.2f} "
              f"(target ~1.0) | MAE {float(np.abs(ctb - 1.0).mean()):.3f}")

    # multi-goal pair-order + eff_rank on val
    from scipy.stats import spearmanr
    with torch.no_grad():
        if len(D["VMG_s"]):
            te = np.random.default_rng(0).integers(0, len(D["VMG_s"]), min(6000, len(D["VMG_s"])))
            dp = net.d_pair(fp(D["VMG_s"][te]), fp(D["VMG_g"][te])).cpu().numpy()
            po = float(spearmanr(dp, np.expm1(D["VMG_d"][te])).correlation)
        else:
            po = float("nan")
        er = float(eff_rank(net.phi(fp(va[np.random.default_rng(1).integers(0, len(va), min(3000, len(va)))])).cpu().numpy()))
    print(f"\nVERDICT TEST-FIELD: committor-ECE {ece:.3f} | committor-MAE {mae:.3f} | win-loss-sep {sep:+.3f} "
          f"| pair-order {po:+.3f} | eff_rank {er:.1f}", flush=True)


if __name__ == "__main__":
    main()
