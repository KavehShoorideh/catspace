#!/usr/bin/env python
"""catspace/research/components/encoder/approaches/concept_quantization/experiments/steer_concept.py -- is a concept CAV an ACTIONABLE subgoal gradient, or just a passive
correlate? (2026-07-21, per the concepts-as-subgoals literature synthesis; the actionability analogue of
Bush et al. ICLR 2025's causal steering test.)

If the value field is smooth in the concept direction, then hill-climbing the CAV projection over the legal
moves -- pick the move maximizing w_c . F(child) -- should CREATE the concept on the board more often than a
random legal move. That is exactly what "use the CAV as a subgoal" requires: the planner descends toward the
region and the region is actually reached. A passive correlate would rank moves no better than chance.

Protocol: train the CAV on a disjoint pool of F(s). Then over test positions, embed F(child) for every legal
move and score it by w_c . F(child). Restrict to DECISION POINTS where the children actually differ on the
ground-truth concept (some child has it, some doesn't) -- those are the moves where the gradient must be
right. Report:
  * move-level ROC-AUC of (w_c . F(child)) vs ground-truth concept(child)  -- does the CAV rank concept-
    creating moves above the rest?
  * hit-rate: concept prevalence of the ARGMAX-CAV move vs a RANDOM legal move (the actionable payoff).
Also runs the quasimetric dist-to-region reach cost as an alternative subgoal score, same metrics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.encoder.approaches.concept_quantization.experiments.concept_features import features as named_features
from catspace.io import paths


def embed_F(fb, boards, om_row, dev, bs=4096):
    """F(s) for a list of boards."""
    out = []
    for i in range(0, len(boards), bs):
        chunk = boards[i:i + bs]
        packed = np.stack([encode_packed(b) for b in chunk])
        meta = np.stack([encode_meta(b) for b in chunk])
        planes = torch.from_numpy(feature_planes(packed, meta)).to(dev)
        om = torch.from_numpy(np.tile(om_row, (len(chunk), 1))).to(dev)
        with torch.no_grad():
            out.append(fb.embed_F(planes, om).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("lichess_gn_iqeqrl_full.pt"))
    ap.add_argument("--shard", required=True)
    ap.add_argument("--concept", default="connected_rooks_w")
    ap.add_argument("--cav-n", type=int, default=10000, help="positions to train the CAV on (disjoint)")
    ap.add_argument("--test-n", type=int, default=2500, help="test positions to expand into child moves")
    ap.add_argument("--region-anchors", type=int, default=32)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    pool = np.flatnonzero(ply >= args.min_ply); pool = pool[rng.permutation(len(pool))]
    cav_idx, test_idx = pool[:args.cav_n], pool[args.cav_n:args.cav_n + args.test_n]

    # ---- train CAV on the disjoint pool ----
    cav_boards = [board_from_packed(P[i], M[i]) for i in cav_idx]
    Fc = embed_F(fb, cav_boards, om, dev)
    mu, sd = Fc.mean(0), Fc.std(0) + 1e-8
    ykey = args.concept
    yc = np.array([named_features(b)[ykey][0] for b in cav_boards], float)
    clf = LogisticRegression(max_iter=400).fit((Fc - mu) / sd, yc)
    w = clf.coef_[0].astype(np.float32); w = w / (np.linalg.norm(w) + 1e-9)
    # region anchors: B(g) of concept-positive pool rows (for the quasimetric reach score)
    with torch.no_grad():
        posb = [b for b, yy in zip(cav_boards, yc) if yy > 0.5]
        rng.shuffle(posb); posb = posb[:args.region_anchors]
        pk = np.stack([encode_packed(b) for b in posb]); mk = np.stack([encode_meta(b) for b in posb])
        Breg = fb.embed_B(torch.from_numpy(feature_planes(pk, mk)).to(dev))

    # ---- expand test positions into (parent, move, child) and score every child ----
    child_boards, owner, child_has = [], [], []
    kept_parents = 0
    for i in test_idx:
        b = board_from_packed(P[i], M[i])
        kids = []
        for m in b.legal_moves:
            c = b.copy(stack=False); c.push(m)
            kids.append(c)
        if len(kids) < 2:
            continue
        hv = np.array([named_features(c)[ykey][0] for c in kids], float)
        if hv.min() == hv.max():             # children all-same on the concept => no decision signal
            continue
        kept_parents += 1
        for c, h in zip(kids, hv):
            child_boards.append(c); owner.append(kept_parents - 1); child_has.append(h)
    owner = np.array(owner); child_has = np.array(child_has)
    Fk = embed_F(fb, child_boards, om, dev)
    cav_score = ((Fk - mu) / sd) @ w
    with torch.no_grad():
        dreg = fb.distance_matrix(torch.from_numpy(Fk).float().to(dev), Breg).min(dim=1).values.cpu().numpy()
    reach_score = -dreg

    def argmax_hitrate(score):
        hit = []
        for p in range(owner.max() + 1):
            sel = np.flatnonzero(owner == p)
            hit.append(child_has[sel[np.argmax(score[sel])]])
        return float(np.mean(hit))

    base_rate = float(child_has.mean())                       # random legal move
    print(f"VERDICT STEER_CONCEPT field={Path(args.field).stem} concept={ykey} "
          f"decision_positions={owner.max()+1} child_moves={len(child_has)}")
    print(f"  base rate (random legal move creates/keeps concept): {base_rate:.0%}")
    print(f"  {'subgoal score':16s} | {'move-AUC':>8s} | {'argmax hit-rate':>15s} | lift over random")
    for nm, sc in [("CAV proj", cav_score), ("quasimetric reach", reach_score)]:
        auc = roc_auc_score(child_has, sc); hr = argmax_hitrate(sc)
        print(f"  {nm:16s} | {auc:>8.3f} | {hr:>14.0%} | {hr/base_rate:>5.2f}x")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
