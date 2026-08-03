#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/control_field_wdl/experiments/measure_veto_channels.py -- THE PUBLICATION GATE (Kaveh 2026-07-23): does the
multichannel field's learned channel divergence track the true adversarial veto?

Claim under test: d(F(s; regime=sf-optimal), B(g)) - d(F(s; regime=random), B(g)) -- the
learned gap between purposeful-play distance and drift distance -- correlates with EXACT
region-deniedness (the 87%/99% forceability ground truth, forceable() DFS vs tb-optimal
defense). If it does, the opponent's veto is readable off the map with NO tablebase at
play time -- the core claim of the LinkedIn note + interactive demo.

Support-honest design: probe positions come from the sf_cont endgame region (where BOTH
regime channels have training support via the mix); deniedness computed exactly there.
Per anchor: sample j-ply cooperative targets, label each {denied, forceable} exactly, and
test whether the per-target learned gap separates the two classes (AUC + spearman with
the anchor-level denied fraction).
"""
from __future__ import annotations

import argparse
import glob
import sys
import time
from pathlib import Path

import chess
import numpy as np


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.fields import FieldModel
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB
from catspace.research.components.encoder.approaches.control_field_wdl.experiments.measure_adversarial_veto import forceable, neighborhood_of, wdl_white
from catspace.io import paths


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("lichess_mc.pt"))
    ap.add_argument("--shards", default=paths.shards("sf_cont_endgame_v1"))
    ap.add_argument("--n-anchors", type=int, default=25)
    ap.add_argument("--targets-per-anchor", type=int, default=30)
    ap.add_argument("--j", type=int, default=4)
    ap.add_argument("--regime-opt", type=int, default=2)
    ap.add_argument("--regime-rand", type=int, default=1)
    ap.add_argument("--max-pieces", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); rng = np.random.default_rng(args.seed); tb = TB()
    fm = FieldModel(args.field, device=args.device, zero_board_only=False)

    def embF_regime(boards, regime):
        pk, mt = fm.pack(boards)
        om = np.tile(fm._om, (len(pk), 1)).astype(np.int64)
        om = np.concatenate([om, np.full((len(pk), 1), regime, np.int64)], axis=1)
        import torch
        with torch.no_grad():
            return fm.fb.embed_F(fm._planes(pk, mt), torch.from_numpy(om).to(fm.device)).cpu().numpy()

    # anchors: White-to-move, won, <= max-pieces, from the sf_cont region
    files = sorted(glob.glob(str(Path(args.shards) / "*.npz"))); rng.shuffle(files)
    anchors = []
    for f in files:
        z = np.load(f); P, M = z["packed"], z["meta"]
        for i in rng.permutation(len(P))[:4000]:
            b = board_from_packed(P[i], M[i])
            if (b.turn == chess.WHITE and not b.is_game_over()
                    and len(b.piece_map()) <= args.max_pieces and wdl_white(b, tb) == 2):
                anchors.append(b)
            if len(anchors) >= args.n_anchors:
                break
        if len(anchors) >= args.n_anchors:
            break
    print(f"[gate] {len(anchors)} won anchors (<= {args.max_pieces} pieces)  [{time.time()-t0:.0f}s]", flush=True)

    gaps, denied, frac_rows = [], [], []
    for a in anchors:
        # cooperative targets
        seen = {}
        for _ in range(200):
            b = a.copy(stack=False); ok = True
            for _t in range(args.j):
                mv = list(b.legal_moves)
                if not mv:
                    ok = False; break
                b.push(mv[int(rng.integers(len(mv)))])
            if ok and not b.is_game_over(claim_draw=True) and wdl_white(b, tb) == 2:
                seen.setdefault(b._transposition_key(), b.copy(stack=False))
        targets = list(seen.values()); rng.shuffle(targets)
        targets = targets[: args.targets_per_anchor]
        if len(targets) < 8:
            continue
        den = [0 if forceable(a, neighborhood_of(g), tb, args.j) else 1 for g in targets]
        Fo = embF_regime([a], args.regime_opt); Fr = embF_regime([a], args.regime_rand)
        Bg = fm.embed_B_boards(targets)
        import torch
        with torch.no_grad():
            d_o = fm.fb.distance_matrix(torch.from_numpy(Fo), torch.from_numpy(Bg))[0].cpu().numpy()
            d_r = fm.fb.distance_matrix(torch.from_numpy(Fr), torch.from_numpy(Bg))[0].cpu().numpy()
        g_ = d_o - d_r                      # learned veto gap per target
        gaps.extend(g_.tolist()); denied.extend(den)
        frac_rows.append((float(np.mean(g_)), float(np.mean(den))))

    gaps = np.array(gaps); denied = np.array(denied)
    from scipy.stats import spearmanr
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(denied, gaps) if 0 < denied.mean() < 1 else float("nan")
    # bootstrap CI (model-card convention: point estimates carry uncertainty)
    boots = []
    brng = np.random.default_rng(1)
    for _ in range(1000):
        idx = brng.integers(0, len(gaps), len(gaps))
        if 0 < denied[idx].mean() < 1:
            boots.append(roc_auc_score(denied[idx], gaps[idx]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (float("nan"),) * 2)
    rho_t = spearmanr(gaps, denied).correlation
    fr = np.array(frac_rows)
    rho_a = spearmanr(fr[:, 0], fr[:, 1]).correlation if len(fr) > 5 else float("nan")
    print(f"VERDICT VETO_CHANNELS field={Path(args.field).stem}  targets={len(gaps)} "
          f"denied-rate={denied.mean():.2f}  AUC(gap->denied)={auc:.3f} [95% CI {lo:.2f}-{hi:.2f}]  "
          f"spearman target-level {rho_t:+.3f} | anchor-level {rho_a:+.3f}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
    print("  gate reading: AUC >= 0.65 / anchor spearman >= +0.4 = the veto is READABLE off the "
          "learned map (publication gate PASSES); ~0.5 / ~0 = channels not yet separated.", flush=True)
    tb.close()


if __name__ == "__main__":
    main()
