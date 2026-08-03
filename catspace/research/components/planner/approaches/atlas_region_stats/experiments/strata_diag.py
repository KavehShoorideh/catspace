#!/usr/bin/env python
"""catspace/research/components/planner/approaches/atlas_region_stats/experiments/strata_diag.py — are STRATA forming? (Kaveh 2026-07-19: some moves
are one-way (no way back), some are reversible; this should appear as DIRECTED
asymmetry in the field.) For (parent, child) move pairs:
  d_fwd = d(F(parent), B(child))   (reach the child -- always ~1 ply)
  d_bwd = d(F(child), B(parent))   (reach the parent BACK)
REVERSIBLE move (king shuffle): parent reachable from child => d_bwd ~ d_fwd.
IRREVERSIBLE move (capture / pawn / promo): parent UNreachable => d_bwd >> d_fwd.
Strata present <=> irreversible asymmetry >> reversible asymmetry. Compares fields."""
from __future__ import annotations

import sys
from pathlib import Path

import chess
import numpy as np
import torch


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt
from catspace.io import paths

dev = "cpu"
om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
dz = np.load(paths.derived("dtm_endgame.npz"))
rng = np.random.default_rng(0)


def collect_pairs(n=300):
    """(parent, child, is_irrev) triples, balanced reversible/irreversible."""
    rev, irr = [], []
    tried = 0
    idx = rng.permutation(len(dz["dtm"]))
    for i in idx:
        if len(rev) >= n and len(irr) >= n:
            break
        tried += 1
        b = board_from_packed(dz["packed"][i], dz["meta"][i])
        for m in b.legal_moves:
            irrev = b.is_irreversible(m)
            c = b.copy(stack=False); c.push(m)
            if c.is_game_over():
                continue
            if irrev and len(irr) < n:
                irr.append((b, c))
            elif not irrev and len(rev) < n:
                rev.append((b, c))
    return rev, irr


def embed(fb, boards, side):
    pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
    pl = torch.from_numpy(feature_planes(pk, mt)).to(dev)
    with torch.no_grad():
        if side == "F":
            o = torch.from_numpy(np.tile(om, (len(boards), 1))).to(dev)
            return fb.embed_F(pl, o)
        return fb.embed_B(pl)


def asym(fb, pairs):
    parents = [p for p, c in pairs]; children = [c for p, c in pairs]
    Fp, Bp = embed(fb, parents, "F"), embed(fb, parents, "B")
    Fc, Bc = embed(fb, children, "F"), embed(fb, children, "B")
    with torch.no_grad():
        d_fwd = fb.distance_matrix(Fp, Bc).diagonal().cpu().numpy()   # parent -> child
        d_bwd = fb.distance_matrix(Fc, Bp).diagonal().cpu().numpy()   # child -> parent
    return d_fwd, d_bwd


def main():
    rev, irr = collect_pairs(300)
    print(f"pairs: reversible={len(rev)} irreversible={len(irr)}")
    for name, ckpt in [("incumbent", paths.sep("cert_base_full.pt")),
                       ("QRL (unreach-trained)", paths.sep("qrl_iqe_sn_full.pt")),
                       ("clustered", paths.sep("cert_base_cluster.pt"))]:
        if not Path(ckpt).exists():
            continue
        fb, _ = load_ckpt(Path(ckpt), dev); fb.eval()
        rf, rb = asym(fb, rev)
        irf, irb = asym(fb, irr)
        # asymmetry = backward / forward (>>1 => can't go back => stratum boundary)
        r_ratio = float(np.median(rb / np.maximum(rf, 1e-6)))
        i_ratio = float(np.median(irb / np.maximum(irf, 1e-6)))
        print(f"  {name:22s}: REVERSIBLE d_bwd/d_fwd={r_ratio:.2f}  "
              f"IRREVERSIBLE d_bwd/d_fwd={i_ratio:.2f}  "
              f"[fwd rev={np.median(rf):.2f} irr={np.median(irf):.2f}]")
        print(f"    VERDICT STRATA {name}: irrev_asym/rev_asym = {i_ratio/max(r_ratio,1e-6):.2f} "
              f"(>>1 => strata present; ~1 => field can't tell one-way from reversible)")


if __name__ == "__main__":
    main()
