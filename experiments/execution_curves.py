#!/usr/bin/env python
"""experiments/execution_curves.py -- Kaveh 2026-07-23: "well-executed and poorly executed
tactics as rolled by simulation from a particular point." The banked shared-anchor ladder
rollouts (regimes: sf_full/sf_2000/sf_1700/sf_1400/maia_1900/1500/1100/random-vs-sf) give
each anchor an EXECUTION CURVE: outcome quality per capability tier. Steep curve = an
empirical TACTICAL position (converts only with skill -- the barrier, measured).

Outcome proxy (rules-only): material swing for the anchor's side-to-move from anchor to
rollout end (captures cashed minus material lost). VERDICTs: mean swing per regime (the
capability spread) + the steep-anchor count (sf_full converts >= +3, maia_1500 <= 0).
"""
from __future__ import annotations
import glob, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.tools.chess_specific.diagnostics import material_count
import chess

REG = {2: "sf_full", 3: "rand_vs_sf", 4: "sf_1400", 5: "sf_1700", 6: "sf_2000",
       8: "maia_1100", 9: "maia_1500", 10: "maia_1900"}

def main():
    t0 = time.time()
    swings = {r: [] for r in REG}
    per_anchor = {}
    files = sorted(glob.glob("data/shards/regime_rollouts_v1/shard_*.npz"))[:120]
    for f in files:
        z = np.load(f)
        gid, reg, aidx, ply = z["game_id"], z["regime"], z["anchor_idx"], z["ply"]
        P, M = z["packed"], z["meta"]
        change = np.flatnonzero(np.diff(gid)) + 1
        starts = np.concatenate([[0], change]); ends = np.concatenate([change, [len(gid)]])
        for s, e in zip(starts, ends):
            r = int(reg[s]); a = (f, int(aidx[s]))
            b0 = board_from_packed(P[s], M[s]); b1 = board_from_packed(P[e-1], M[e-1])
            mover = b0.turn
            swing = ((material_count(b1, mover) - material_count(b1, not mover))
                     - (material_count(b0, mover) - material_count(b0, not mover)))
            swings[r].append(swing)
            per_anchor.setdefault(a, {})[r] = swing
    print("VERDICT EXECUTION_CURVES  mean material swing (mover-POV) per capability tier:")
    for r in (2, 6, 5, 4, 10, 9, 8, 3):
        v = swings[r]
        if v:
            print(f"    {REG[r]:10s} n={len(v):6,}  mean {np.mean(v):+.2f}  frac>0 {np.mean(np.array(v)>0):.2f}")
    steep = sharp = 0
    for a, d in per_anchor.items():
        if 2 in d and 9 in d:
            sharp += 1
            if d[2] >= 3 and d[9] <= 0:
                steep += 1
    print(f"VERDICT STEEP_ANCHORS  {steep}/{sharp} anchors: SF-full converts (>=+3) while "
          f"maia-1500 does not (<=0) -- empirically TACTICAL positions  [{time.time()-t0:.0f}s]")

if __name__ == "__main__":
    main()
