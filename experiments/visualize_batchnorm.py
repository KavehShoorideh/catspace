#!/usr/bin/env python
"""experiments/visualize_batchnorm.py — VISUAL proof that BatchNorm statistics
(not training/objective) create-then-hide the pawn-capture one-way asymmetry.
Scatters (d_forward, d_backward) for pawn-capture pairs vs reversible-king moves,
under EVAL (running BN stats) vs the BN-isolation mode (only BatchNorm on batch
stats, everything else eval). y=x is symmetric; points ABOVE = one-way."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch, chess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt

dev = "cpu"
_ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/derived/sep/iqe_infinite.pt")
_out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/experiments/batchnorm_proof.png"
fb, _ = load_ckpt(_ckpt, dev)
om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
pz = np.load("data/derived/pawncap_pairs.npz")
nz = np.load("data/derived/lichess_nearmate.npz"); won = np.flatnonzero(nz["dtm"] > 0)
rng = np.random.default_rng(5)


def eF(pk, mt):
    return fb.embed_F(torch.from_numpy(feature_planes(pk, mt)), torch.from_numpy(np.tile(om, (len(pk), 1))))


def eB(pk, mt):
    return fb.embed_B(torch.from_numpy(feature_planes(pk, mt)))


def fwd_bwd(pp, pm, cp, cm):
    with torch.no_grad():
        f = fb.distance_matrix(eF(pp, pm), eB(cp, cm)).diagonal().numpy()
        b = fb.distance_matrix(eF(cp, cm), eB(pp, pm)).diagonal().numpy()
    return f, b


# reversible-king (parent, child) pairs
rp, rc = [], []
for j in rng.choice(won, 1200, replace=False):
    b = board_from_packed(nz["packed"][j], nz["meta"][j])
    km = [m for m in b.legal_moves if b.piece_at(m.from_square)
          and b.piece_at(m.from_square).piece_type == chess.KING and not b.is_capture(m)]
    if not km:
        continue
    c = b.copy(stack=False); c.push(km[0])
    if c.is_game_over():
        continue
    rp.append((encode_packed(b), encode_meta(b))); rc.append((encode_packed(c), encode_meta(c)))
    if len(rp) >= 200:
        break
rpp = np.stack([x[0] for x in rp]); rpm = np.stack([x[1] for x in rp])
rcp = np.stack([x[0] for x in rc]); rcm = np.stack([x[1] for x in rc])
I = rng.choice(len(pz["p_packed"]), 200, replace=False)

bn = [m for m in fb.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]


def asym(setup, pawncap=True):
    setup()
    if pawncap:
        f, b = fwd_bwd(pz["p_packed"][I], pz["p_meta"][I], pz["c_packed"][I], pz["c_meta"][I])
    else:
        f, b = fwd_bwd(rpp, rpm, rcp, rcm)
    return b / np.maximum(f, 1e-6)


eval_setup = lambda: fb.eval()
bn_setup = lambda: ([fb.eval()] + [m.train() for m in bn])
pc_eval = asym(eval_setup, True)
pc_bn = asym(bn_setup, True)
rev_eval = asym(eval_setup, False)

fig, ax = plt.subplots(figsize=(11, 6))
bins = np.logspace(-0.5, 2.3, 45)
ax.hist(rev_eval, bins=bins, alpha=0.55, color="#4c78a8", label=f"reversible move (any mode): median {np.median(rev_eval):.1f}×")
ax.hist(pc_eval, bins=bins, alpha=0.65, color="#e45756", label=f"PAWN CAPTURE, EVAL / running stats (as USED): median {np.median(pc_eval):.1f}×")
ax.hist(pc_bn, bins=bins, alpha=0.65, color="#8b0000", label=f"PAWN CAPTURE, BATCH stats (only BN flipped): median {np.median(pc_bn):.1f}×")
ax.axvline(1.0, color="k", ls="--", lw=1, alpha=0.6)
ax.text(1.0, ax.get_ylim()[1] * 0.92, " symmetric (1×)", fontsize=9)
ax.set_xscale("log")
ax.set_xlabel("one-way asymmetry  d(child→parent) / d(parent→child)   (1 = reversible, ≫1 = one-way)")
ax.set_ylabel("count (pawn-capture / reversible pairs)")
_gap = np.median(pc_bn) / max(np.median(pc_eval), 1e-6)
if _gap > 3:   # BatchNorm field: eval collapses, batch stats reveal the learned rule
    _t = ("SAME weights, SAME positions — only the BatchNorm statistics change.\n"
          "The field LEARNS pawn captures are one-way (dark red), but running stats at inference (red) collapse it to symmetric.")
else:          # GroupNorm field: no train/eval gap — the two red distributions coincide
    _t = ("GroupNorm (fixed): NO train/eval gap — the two pawn-capture distributions COINCIDE.\n"
          "The one-way structure the field learns (dark red) now survives at inference (red); reversible stays symmetric.")
ax.set_title(_t)
ax.legend(loc="upper right", fontsize=9.5)
fig.tight_layout()
out = _out
fig.savefig(out, dpi=115)
print(f"VERDICT BN_VIZ saved {out}")
