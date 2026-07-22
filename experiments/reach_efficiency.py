#!/usr/bin/env python
"""experiments/reach_efficiency.py -- does the field make COARSE long-range navigation more EFFICIENT?
(Kaveh 2026-07-21: near mate, pure MCTS is dominant; the real question is whether the field gets us to the
near-mate region FASTER / with fewer node-evals from far away, where pure search is blind -- mate_stop only
fires near mate, so far out it has no gradient.)

From KRRvKBP (6-piece) starts, play toward the near-mate region (<= --target-pieces, where search+tablebase
take over) with White = MCTS, Black = tablebase-optimal. Compare, across node budgets:
  * field-guided : leaf value = -d(F, MATE_W) (the distilled field's coarse distance-to-mate gradient)
  * pure-search  : constant leaf value (MCTS + mate_stop only, no long-range gradient)
Metrics: reach-rate, plies-to-target, and TOTAL node-evals-to-target (plies x budget) = the compute Kaveh
cares about. If the field reaches the near-mate region in fewer evals -- especially at LOW budgets where pure
search is myopic -- the field earns its place as the coarse navigator.
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
from experiments.value_fixed_point import TB, tb_best_move


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/nucleus_distilled.pt")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--budgets", default="100,400,1600")
    ap.add_argument("--target-pieces", type=int, default=5)
    ap.add_argument("--ply-cap", type=int, default=40)
    ap.add_argument("--min-dtm", type=int, default=30, help="only far-from-mate 6-piece starts (DTM>=this)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    zg = torch.load(args.field, map_location="cpu", weights_only=False)["zgoals"]
    MATE = (zg["MATE_W"].detach().float() if torch.is_tensor(zg["MATE_W"])
            else torch.tensor(np.asarray(zg["MATE_W"], np.float32))).to(dev)[None, :]
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    tb = TB("data/syzygy")

    def reach_field(boards):
        pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
        with torch.no_grad():
            F = fb.embed_F(torch.from_numpy(feature_planes(pk, mt)).to(dev),
                           torch.from_numpy(np.tile(om, (len(boards), 1))).to(dev))
            return -fb.distance_matrix(F, MATE)[:, 0].cpu().numpy()

    def reach_pure(boards):
        return np.zeros(len(boards), dtype=np.float32)

    dz = np.load(args.dtm_npz); dtm = np.asarray(dz["dtm"]).astype(float)
    P, M = np.asarray(dz["packed"]), np.asarray(dz["meta"])
    starts = []
    for i in rng.permutation(len(P)):
        b = board_from_packed(P[i], M[i])
        if len(b.piece_map()) == 6 and dtm[i] >= args.min_dtm and b.turn == chess.WHITE and not b.is_game_over():
            starts.append((b, dtm[i]))
        if len(starts) >= args.n:
            break

    def run(reach_fn, nodes):
        mcts = MCTS(reach_fn, max_nodes=nodes, mate_stop=True, pw_c=1.5, root_min_visits=10)
        plies_list, reached = [], 0
        for b0, _ in starts:
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
            if len(b.piece_map()) <= args.target_pieces and hit is None:
                hit = args.ply_cap
            if hit is not None:
                reached += 1; plies_list.append(hit)
        rate = reached / len(starts)
        mp = float(np.mean(plies_list)) if plies_list else float("nan")
        return rate, mp

    print(f"VERDICT REACH_EFFICIENCY field={Path(args.field).stem} n={len(starts)} target<={args.target_pieces}p "
          f"(6-piece starts, DTM>={args.min_dtm}; White=MCTS, Black=tablebase-optimal)")
    print(f"  {'budget':>6s} | {'FIELD reach/plies/evals':>26s} | {'PURE reach/plies/evals':>26s}")
    for nb in [int(x) for x in args.budgets.split(",")]:
        rf, pf = run(reach_field, nb)
        rp, pp = run(reach_pure, nb)
        ef = pf * nb if pf == pf else float("nan"); ep = pp * nb if pp == pp else float("nan")
        print(f"  {nb:>6d} | {rf:>6.0%}  {pf:>4.1f}p  {ef:>8.0f} ev | {rp:>6.0%}  {pp:>4.1f}p  {ep:>8.0f} ev")
    tb.close()
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
