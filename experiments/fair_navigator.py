#!/usr/bin/env python
"""experiments/fair_navigator.py -- the FAIR coarse-navigation test (Kaveh 2026-07-21: compare against an MCTS
trained on the SAME data). From far-from-mate 6-piece KRRvKBP starts, reach the near-mate region (<=5 pieces)
with White=MCTS, Black=tablebase-optimal, across node budgets, for three MCTS leaf values:
  * FIELD    : -d(F, MATE_W) of the distilled quasimetric field
  * CNN      : a plain CNN value net (train_dtm_cnn.py) regressing DTM -- SAME data, no quasimetric  <- fair baseline
  * PURE     : constant value (MCTS + mate_stop only, no learned value)                              <- reference
If FIELD ~ CNN, the coarse-nav benefit is "any learned value", not the quasimetric; if FIELD > CNN, the
quasimetric structure helps; if FIELD < CNN, the field is a worse coarse value than a plain net.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import board_from_packed, encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device
from catspace.nn.mcts import MCTS
from experiments.train_dtm_cnn import DTMNet
from experiments.value_fixed_point import TB, tb_best_move


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/nucleus_distilled.pt")
    ap.add_argument("--cnn", default="data/derived/sep/dtm_cnn.pt")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--budgets", default="100,400")
    ap.add_argument("--target-pieces", type=int, default=5)
    ap.add_argument("--ply-cap", type=int, default=40)
    ap.add_argument("--min-dtm", type=int, default=30)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    zg = torch.load(args.field, map_location="cpu", weights_only=False)["zgoals"]
    MATE = (zg["MATE_W"].detach().float() if torch.is_tensor(zg["MATE_W"])
            else torch.tensor(np.asarray(zg["MATE_W"], np.float32))).to(dev)[None, :]
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    cd = torch.load(args.cnn, map_location=dev, weights_only=False)
    cnn = DTMNet(c=cd["c"]).to(dev); cnn.load_state_dict(cd["state"]); cnn.eval()
    tb = TB("data/syzygy")

    def planes_of(boards):
        return feature_planes(np.stack([encode_packed(b) for b in boards]),
                              np.stack([encode_meta(b) for b in boards]))

    def reach_field(boards):
        with torch.no_grad():
            F = fb.embed_F(torch.from_numpy(planes_of(boards)).to(dev),
                           torch.from_numpy(np.tile(om, (len(boards), 1))).to(dev))
            return -fb.distance_matrix(F, MATE)[:, 0].cpu().numpy()

    def reach_cnn(boards):
        with torch.no_grad():
            return -cnn(torch.from_numpy(planes_of(boards)).to(dev)).cpu().numpy()   # -pred_DTM = closer=higher

    def reach_pure(boards):
        return np.zeros(len(boards), dtype=np.float32)

    dz = np.load(args.dtm_npz); dtm = np.asarray(dz["dtm"]).astype(float)
    P, M = np.asarray(dz["packed"]), np.asarray(dz["meta"])
    starts = []
    for i in rng.permutation(len(P)):
        b = board_from_packed(P[i], M[i])
        if len(b.piece_map()) == 6 and dtm[i] >= args.min_dtm and b.turn == chess.WHITE and not b.is_game_over():
            starts.append(b)
        if len(starts) >= args.n:
            break

    def run(reach_fn, nodes):
        mcts = MCTS(reach_fn, max_nodes=nodes, mate_stop=True, pw_c=1.5, root_min_visits=10)
        plies, reached = [], 0
        for b0 in starts:
            b = b0.copy(stack=False); hit = None
            for p in range(args.ply_cap):
                if len(b.piece_map()) <= args.target_pieces:
                    hit = p; break
                if b.is_game_over(claim_draw=True):
                    break
                m = mcts.best_move(b) if b.turn == chess.WHITE else tb_best_move(b, tb)
                if m is None:
                    break
                b.push(m)
            if hit is not None:
                reached += 1; plies.append(hit)
        return reached / len(starts), (float(np.mean(plies)) if plies else float("nan"))

    print(f"VERDICT FAIR_NAVIGATOR n={len(starts)} target<={args.target_pieces}p (6-piece DTM>={args.min_dtm} starts)")
    print(f"  {'budget':>6s} | {'FIELD reach/plies':>18s} | {'CNN reach/plies':>18s} | {'PURE reach/plies':>18s}")
    for nb in [int(x) for x in args.budgets.split(",")]:
        rf, pf = run(reach_field, nb); rc, pc_ = run(reach_cnn, nb); rp, pp = run(reach_pure, nb)
        print(f"  {nb:>6d} | {rf:>6.0%}  {pf:>4.1f}p (ev {pf*nb:>6.0f}) | {rc:>6.0%}  {pc_:>4.1f}p (ev {pc_*nb:>6.0f}) | "
              f"{rp:>6.0%}  {pp:>4.1f}p (ev {pp*nb:>6.0f})")
    tb.close()
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
