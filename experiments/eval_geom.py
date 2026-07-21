#!/usr/bin/env python
"""experiments/eval_geom.py -- held-out evaluation of the geometry field (board-only).
Reports the metrics that MATTER for the planner:
  - per-material spearman(d, DTM): within-cluster distance-to-mate order (the planner's
    descent signal). Overall (across-material) spearman is reported too but is
    STRUCTURALLY limited: a single mate pole can't order separated clusters, so it is
    not the target.
  - reversible vs irreversible one-way asymmetry d(child->parent)/d(parent->child):
    the strata signal. Irreversible >> reversible = one-way structure present.

Usage: .venv/bin/python experiments/eval_geom.py data/derived/sep/iqe_geom.pt
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch, chess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt
from scipy.stats import spearmanr

ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/derived/sep/iqe_geom.pt")
fb, pay = load_ckpt(ckpt, "cpu"); fb.eval()
om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]


def bp(pk, mt):
    pl = feature_planes(pk, mt); pl[:, (18, 19)] = 0.0; return torch.from_numpy(pl)


def eF(pk, mt):
    return fb.embed_F(bp(pk, mt), torch.from_numpy(np.tile(om, (len(pk), 1))))


def eB(pk, mt):
    return fb.embed_B(bp(pk, mt))


nz = np.load("data/derived/lichess_nearmate.npz"); won = np.flatnonzero(nz["dtm"] > 0)
nmp, nmm, nmd = nz["packed"][won], nz["meta"][won], nz["dtm"][won].astype(np.float32)
mk = np.array(["".join(sorted(p.symbol() for p in board_from_packed(nmp[i], nmm[i]).piece_map().values()))
               for i in range(len(nmd))])
rng = np.random.default_rng(3); low = np.argsort(nmd)[:512]
zg = pay.get("zgoals") or {}
zmate = torch.as_tensor(zg["MATE_W"]).float().view(1, -1) if "MATE_W" in zg else eB(nmp[low], nmm[low]).mean(0, keepdim=True)
ev = rng.choice(len(nmd), 3000, replace=False)
with torch.no_grad():
    d = fb.distance_matrix(eF(nmp[ev], nmm[ev]), zmate)[:, 0].numpy()
overall = spearmanr(d, nmd[ev]).correlation
rows = []
for m in sorted(set(mk[ev]), key=lambda x: -np.sum(mk[ev] == x)):
    s = np.flatnonzero(mk[ev] == m)
    if len(s) < 30 or np.unique(nmd[ev][s]).size < 3:
        continue
    rows.append((m, len(s), spearmanr(d[s], nmd[ev][s]).correlation))
    if len(rows) >= 10:
        break


def ratio(P, C):
    pp = np.stack([encode_packed(b) for b in P]); pmt = np.stack([encode_meta(b) for b in P])
    cp = np.stack([encode_packed(b) for b in C]); cm = np.stack([encode_meta(b) for b in C])
    with torch.no_grad():
        f = fb.distance_matrix(eF(pp, pmt), eB(cp, cm)).diagonal().numpy()
        b = fb.distance_matrix(eF(cp, cm), eB(pp, pmt)).diagonal().numpy()
    return float(np.median(f)), float(np.median(b)), float(np.median(b / np.maximum(f, 1e-6)))


rp, rc, ip, ic = [], [], [], []
for j in rng.choice(won, 8000, replace=False):
    b = board_from_packed(nz["packed"][j], nz["meta"][j])
    if b.is_game_over():
        continue
    for m in b.legal_moves:
        c = b.copy(stack=False); c.push(m)
        if c.is_game_over():
            continue
        if b.is_irreversible(m) and len(ip) < 300:
            ip.append(b.copy(stack=False)); ic.append(c)
        elif not b.is_irreversible(m) and len(rp) < 300:
            rp.append(b.copy(stack=False)); rc.append(c)
        break
    if len(ip) >= 300 and len(rp) >= 300:
        break
rr = ratio(rp, rc); ir = ratio(ip, ic)
pmvals = [r for *_, r in rows]
print(f"[ckpt] {ckpt}")
print(f"held-out DTM:  overall={overall:+.3f} (structurally limited by clustering)  "
      f"per-material MEAN={np.mean(pmvals):+.3f} (median {np.median(pmvals):+.3f}, n_mat={len(rows)})")
for m, n, r in rows:
    print(f"    {m:<12} n={n:<5} spearman={r:+.3f}")
print(f"REVERSIBLE   edges: d_fwd={rr[0]:.2f} d_bwd={rr[1]:.2f}  asym={rr[2]:.2f}x")
print(f"IRREVERSIBLE edges: d_fwd={ir[0]:.2f} d_bwd={ir[1]:.2f}  asym={ir[2]:.2f}x")
print(f"VERDICT GEOM_EVAL permat={np.mean(pmvals):+.3f} overall={overall:+.3f} "
      f"oneway_separation={ir[2]/max(rr[2],1e-6):.2f}x (irrev {ir[2]:.2f}x / rev {rr[2]:.2f}x)")
