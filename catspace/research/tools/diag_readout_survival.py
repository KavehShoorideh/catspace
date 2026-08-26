"""Is the future head's `chg` measuring one ply of chess, or the encoder drifting?

The trained run reported a 1-ply bit-change rate climbing 0.075 -> 0.587. A single move
cannot alter 59% of the concept bits, so one of two things is true: the concepts really
did become that volatile, or the EMA target simply fell behind the online encoder and
`chg` became a drift meter. Encoding parent and child with the SAME weights separates
them: same-encoder change is real dynamics, the gap to the EMA number is drift.
"""
import numpy as np, torch, chess
from catspace.research.tools.npz_io import load_npz
import glob, sys

ck = torch.load("artifacts/experiments/qat1_latest.pt", map_location="cpu", weights_only=False)
print("  checkpoint keys:", [k for k in ck.keys()][:8])
sd = ck.get("state_dict", ck.get("model", ck))
w = None
for k in sd:
    if k.endswith("readout.weight"):
        w = sd[k]; break
if w is None:
    print("  readout.weight not found; keys sample:", list(sd)[:12]); sys.exit(0)
print(f"\n  readout weight {tuple(w.shape)}")
D_STATE, L2, MV = 2528, 512, 32
col = w.abs().sum(0)
alive = (col > 0)
print(f"  live concept columns: {int(alive.sum())} of {w.shape[1]}")
fam = [("level-1 board+square+piece+edge", 0, D_STATE),
       ("level-2 quantized attention",     D_STATE, D_STATE + L2),
       ("MOVE concepts",                   D_STATE + L2, D_STATE + L2 + MV)]
print(f"\n  {'family':38s} {'alive':>10s} {'of':>6s}   {'share kept':>10s}")
for nm, a, b in fam:
    k = int(alive[a:b].sum()); n = b - a
    print(f"  {nm:38s} {k:>10d} {n:>6d}   {100*k/n:>9.1f}%")
mv_alive = int(alive[D_STATE+L2:].sum())
if mv_alive == 0:
    print("\n  >>> EVERY move concept was pruned: per-child ranking is structurally dead.")
else:
    print(f"\n  >>> {mv_alive}/32 move concepts survived, so ranking is still possible.")
