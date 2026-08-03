#!/usr/bin/env python
"""experiments/precompute_trunk_wdl.py -- cache the FROZEN T1 trunk's 3-way WDL (win/draw/loss, side-to-
move POV) for every fen in the M2a data. These are the 3 outcome-basin targets (Win/Draw/Loss -- M0's
three basins) for the context-conditioned field's basin prototypes (train_cond_field.py). Tiny output
(N,3); avoids storing the 2.3 GB of raw trunk feature maps.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    data = "data/derived/transition_data_labeled.npz"
    out = "data/derived/m2a_trunk_wdl.npy"
    t0 = time.time()
    from lczerolens import LczeroBoard
    from catspace.research.components.encoder.approaches.reachability_field.src.field import ReachabilityField
    field = ReachabilityField()
    z = np.load(data, allow_pickle=True)
    fens = z["fen"]; N = len(fens)
    wdl = np.empty((N, 3), np.float32); B = 2048
    for s in range(0, N, B):
        chunk = fens[s:s + B]
        x = torch.stack([LczeroBoard(f).to_input_tensor() for f in chunk]).float().to(field.dev)
        with torch.no_grad():
            wdl[s:s + len(chunk)] = field.trunk(x)["wdl"].cpu().numpy()
        if s % (B * 8) == 0:
            print(f"  {s+len(chunk):,}/{N:,} [{time.time()-t0:.0f}s]", flush=True)
    np.save(out, wdl)
    am = wdl.argmax(1)
    print(f"\n=== {out}: {N:,} positions | basin mix W {np.mean(am==0):.1%} D {np.mean(am==1):.1%} "
          f"L {np.mean(am==2):.1%} [{time.time()-t0:.0f}s] ===")
    print("DONE precompute_trunk_wdl", flush=True)


if __name__ == "__main__":
    main()
