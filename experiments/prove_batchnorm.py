#!/usr/bin/env python
"""experiments/prove_batchnorm.py — PROVE that BatchNorm (not training or the
objective) is why the learned pawn-capture one-way asymmetry vanishes at
inference (Kaveh 2026-07-20: capture this finding rigorously).

Design: ONE trained field (iqe_infinite), the SAME held-out pawn-capture pairs,
identical weights. We change ONLY the normalization mode and measure the one-way
asymmetry d(child->parent)/d(parent->child):
  (A) full EVAL  -- every module eval; BN uses fixed RUNNING stats.
  (B) full TRAIN -- every module train; BN uses per-BATCH stats.
  (C) eval EXCEPT BatchNorm set to train -- ISOLATES BN: everything else fixed,
      only BN switched to batch stats. If (C) ~ (B) >> (A), BN is the SOLE cause.
Plus: a reversible-king control (should be ~1 in all modes), and a census of
train/eval-sensitive modules (if BatchNorm is the only one, the entire train<->eval
gap IS BatchNorm, by construction)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch, chess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt

dev = "cpu"
_ckpt = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/derived/sep/iqe_infinite.pt")
fb, _ = load_ckpt(_ckpt, dev)
print(f"[ckpt] {_ckpt}")
om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
pz = np.load("data/derived/pawncap_pairs.npz")
nz = np.load("data/derived/lichess_nearmate.npz"); won = np.flatnonzero(nz["dtm"] > 0)
rng = np.random.default_rng(21)
I = rng.choice(len(pz["p_packed"]), 256, replace=False)          # held-out batch (>=2 for BN stats)


def eF(pk, mt):
    return fb.embed_F(torch.from_numpy(feature_planes(pk, mt)),
                      torch.from_numpy(np.tile(om, (len(pk), 1))))


def eB(pk, mt):
    return fb.embed_B(torch.from_numpy(feature_planes(pk, mt)))


def pawncap_asym():
    with torch.no_grad():
        fwd = fb.distance_matrix(eF(pz["p_packed"][I], pz["p_meta"][I]),
                                 eB(pz["c_packed"][I], pz["c_meta"][I])).diagonal().numpy()
        bwd = fb.distance_matrix(eF(pz["c_packed"][I], pz["c_meta"][I]),
                                 eB(pz["p_packed"][I], pz["p_meta"][I])).diagonal().numpy()
    return float(np.median(fwd)), float(np.median(bwd)), float(np.median(bwd / np.maximum(fwd, 1e-6)))


def reversible_asym():
    rr = []
    for j in rng.choice(won, 400, replace=False):
        b = board_from_packed(nz["packed"][j], nz["meta"][j])
        km = [m for m in b.legal_moves if b.piece_at(m.from_square)
              and b.piece_at(m.from_square).piece_type == chess.KING and not b.is_capture(m)]
        if not km:
            continue
        c = b.copy(stack=False); c.push(km[0])
        if c.is_game_over():
            continue
        with torch.no_grad():
            f = fb.distance_matrix(eF(encode_packed(b)[None], encode_meta(b)[None]),
                                   eB(encode_packed(c)[None], encode_meta(c)[None]))[0, 0]
            g = fb.distance_matrix(eF(encode_packed(c)[None], encode_meta(c)[None]),
                                   eB(encode_packed(b)[None], encode_meta(b)[None]))[0, 0]
        rr.append(float(g) / max(float(f), 1e-6))
        if len(rr) >= 128:
            break
    return float(np.median(rr))


# census of train/eval-sensitive modules
bn = [m for m in fb.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
dp = [m for m in fb.modules() if isinstance(m, torch.nn.Dropout)]
other = [type(m).__name__ for m in fb.modules()
         if isinstance(m, (torch.nn.modules.instancenorm._InstanceNorm,)) ]
print(f"train/eval-sensitive modules: BatchNorm={len(bn)}  Dropout={len(dp)}  other={other or 'none'}")

fb.eval();  A = pawncap_asym();  Ac = reversible_asym()
fb.train(); B = pawncap_asym()
fb.eval()
for m in bn:
    m.train()                                                    # only BN -> batch stats
C = pawncap_asym()
print(f"  (A) full EVAL  (BN running stats):        fwd={A[0]:.2f} bwd={A[1]:.2f} asym={A[2]:.1f}x")
print(f"  (B) full TRAIN (BN batch stats):          fwd={B[0]:.2f} bwd={B[1]:.2f} asym={B[2]:.1f}x")
print(f"  (C) EVAL except BN->train (isolate BN):   fwd={C[0]:.2f} bwd={C[1]:.2f} asym={C[2]:.1f}x")
print(f"  reversible-king control (full eval): asym={Ac:.2f}x  (should be ~1)")
print(f"VERDICT BN_PROOF eval={A[2]:.1f}x train={B[2]:.1f}x bn_only={C[2]:.1f}x "
      f"(C~B>>A + BatchNorm the sole train/eval module => BatchNorm is the cause)")
