#!/usr/bin/env python
"""experiments/eval_field_dtm.py -- the DTM-ORDER measurement pass (Kaveh 2026-07-20).
Held-out evaluation of how well a field's distance-to-mate-pole ranks near-mate
positions by their true tablebase DTM. Reports OVERALL spearman and PER-MATERIAL
spearman (the ranking objective is within-material, so an overall number conflates
across-material separation with within-material order -- per-material is the honest
read). Also reports the eval-mode pawn-death one-way asymmetry. All in EVAL mode.

Usage:
  .venv/bin/python experiments/eval_field_dtm.py data/derived/sep/iqe_infinite_gn.pt
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch, chess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.data.encode import board_from_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt
from scipy.stats import spearmanr

dev = "cpu"
ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/derived/sep/iqe_infinite_gn.pt")
fb, pay = load_ckpt(ckpt, dev); fb.eval()
om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
nz = np.load("data/derived/lichess_nearmate.npz"); won = nz["dtm"] > 0
nmp, nmm, nmd = nz["packed"][won], nz["meta"][won], nz["dtm"][won].astype(np.float32)
matkey = np.array(["".join(sorted(p.symbol() for p in board_from_packed(nmp[i], nmm[i]).piece_map().values()))
                   for i in range(len(nmd))])
rng = np.random.default_rng(7)


def eF(pk, mt):
    return fb.embed_F(torch.from_numpy(feature_planes(pk, mt)),
                      torch.from_numpy(np.tile(om, (len(pk), 1))))


def eB(pk, mt):
    return fb.embed_B(torch.from_numpy(feature_planes(pk, mt)))


# mate pole = stored MATE_W if present, else mean-B of the 512 lowest-DTM positions
low = np.argsort(nmd)[:512]
zg = pay.get("zgoals") or {}
if "MATE_W" in zg:
    zmate = torch.as_tensor(zg["MATE_W"]).float().view(1, -1)
else:
    with torch.no_grad():
        zmate = eB(nmp[low], nmm[low]).mean(0, keepdim=True)

ev = rng.choice(len(nmd), min(3000, len(nmd)), replace=False)
with torch.no_grad():
    d = fb.distance_matrix(eF(nmp[ev], nmm[ev]), zmate)[:, 0].numpy()
overall = spearmanr(d, nmd[ev]).correlation

# per-material spearman for the largest classes (need spread in DTM within the class)
rows = []
for mk in sorted(set(matkey[ev]), key=lambda m: -np.sum(matkey[ev] == m)):
    sel = np.flatnonzero(matkey[ev] == mk)
    if len(sel) < 30 or np.unique(nmd[ev][sel]).size < 3:
        continue
    r = spearmanr(d[sel], nmd[ev][sel]).correlation
    rows.append((mk, len(sel), r))
    if len(rows) >= 10:
        break

# eval-mode pawn-death one-way asymmetry
pz = np.load("data/derived/pawndeath_pairs.npz")
pi = rng.choice(len(pz["p_packed"]), 300, replace=False)
with torch.no_grad():
    fwd = fb.distance_matrix(eF(pz["p_packed"][pi], pz["p_meta"][pi]),
                             eB(pz["c_packed"][pi], pz["c_meta"][pi])).diagonal().numpy()
    bwd = fb.distance_matrix(eF(pz["c_packed"][pi], pz["c_meta"][pi]),
                             eB(pz["p_packed"][pi], pz["p_meta"][pi])).diagonal().numpy()
asym = float(np.median(bwd / np.maximum(fwd, 1e-6)))

print(f"[ckpt] {ckpt}")
print(f"overall held-out spearman(d, DTM) = {overall:+.3f}   (n={len(ev)})")
print("per-material spearman (largest classes with DTM spread):")
for mk, n, r in rows:
    print(f"    {mk:<12} n={n:<5} spearman={r:+.3f}")
if rows:
    print(f"    per-material MEAN = {np.mean([r for *_, r in rows]):+.3f}  (median {np.median([r for *_, r in rows]):+.3f})")
print(f"pawn-death one-way asymmetry (EVAL) = {asym:.1f}x")
print(f"VERDICT FIELD_DTM overall={overall:+.3f} permat_mean={np.mean([r for *_, r in rows]) if rows else float('nan'):+.3f} pawndeath_asym={asym:.1f}x")
