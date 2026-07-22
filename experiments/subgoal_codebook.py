#!/usr/bin/env python
"""experiments/subgoal_codebook.py -- rank concepts as planner subgoals by NAVIGABILITY x VALUE
(2026-07-21, Kaposi's call; the subgoal-density-prior plan, S = forceability x reachability x value).

For each named binary concept we measure two axes and rank:

  navigability -- can White STEER into the concept by climbing its CAV? Rollout from positions lacking the
    concept, opponent = base FB reach policy (depth 1), White greedy-1-ply on the CAV; reach-within-K rate
    vs White just playing the base policy toward mate. lift = cav_reach - base_reach. (Validated for
    connected_rooks at 28% vs 6% in concept_reach_rollout.py.)
  value -- is the concept WORTH reaching? Two signals on a held-out pool: (a) FIELD reach-to-MATE_W gap
    (mean score(F(s),MATE_W) for concept-positive minus concept-negative, z-scored) and (b) EXTERNAL game
    result gap (mean white-POV outcome, concept-positive minus negative).

A good subgoal is both navigable AND valuable. codebook_score = lift x max(field_value_z, 0).
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
from catspace.nn.policy_fb import FBBoardPolicy
from experiments.concept_features import features as named_features
from experiments.steer_concept import embed_F
from sklearn.linear_model import LogisticRegression

CONCEPTS = ["connected_rooks_w", "passed_pawn_w", "king_safe_w", "bishop_pair_w", "queens_on"]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--cav-n", type=int, default=9000)
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--plies", type=int, default=10)
    ap.add_argument("--region", type=int, default=0, help="unused; kept for parity")
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    fb, extra = load_ckpt(Path(args.field), dev); fb.eval()
    zg = torch.load(args.field, map_location="cpu", weights_only=False)["zgoals"]
    def _vec(x): return x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, np.float32)
    zW_np, zB_np = _vec(zg["MATE_W"]), _vec(zg["MATE_B"])
    zW = torch.tensor(zW_np, device=dev)
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]

    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    result = np.asarray(nz["result"]).astype(np.float32)             # game outcome
    pool = np.flatnonzero(ply >= args.min_ply); pool = pool[rng.permutation(len(pool))]
    cav_idx = pool[:args.cav_n]; rest = pool[args.cav_n:]
    # white-POV outcome in [0,1]; detect {-1,0,1} vs {0,.5,1}
    rv = result[cav_idx]; lo = np.nanmin(rv)
    win = (rv + 1) / 2 if lo < -0.5 else rv                          # -> white score in [0,1]

    cav_boards = [board_from_packed(P[i], M[i]) for i in cav_idx]
    Fc = embed_F(fb, cav_boards, om, dev)
    mu, sd = Fc.mean(0), Fc.std(0) + 1e-8
    with torch.no_grad():
        reach_mate = fb.score(torch.from_numpy(Fc).float().to(dev), zW).cpu().numpy()   # field value
    feats = [named_features(b) for b in cav_boards]

    opp = FBBoardPolicy(fb, zB_np, depth=1, device=dev)
    basepol_w = FBBoardPolicy(fb, zW_np, depth=1, device=dev)

    def white_move(board, w, strat):
        moves = list(board.legal_moves)
        if strat == "basepolicy":
            return basepol_w.move(board, rng)
        kids = [board.copy(stack=False) for _ in moves]
        for c, m in zip(kids, moves):
            c.push(m)
        Fk = embed_F(fb, kids, om, dev)
        return moves[int(np.argmax(((Fk - mu) / sd) @ w))]           # climb CAV

    print(f"VERDICT SUBGOAL_CODEBOOK field={Path(args.field).stem} games={args.games} plies={args.plies}")
    print(f"  {'concept':16s} {'base':>4s} | {'cav_reach':>9s} {'basepol':>7s} {'lift':>5s} | "
          f"{'value_z':>7s} {'result_gap':>10s} | {'SCORE':>6s}")
    rows = []
    for cname in CONCEPTS:
        y = np.array([f[cname][0] for f in feats], float)
        if not (0.05 < y.mean() < 0.95):
            continue
        w = LogisticRegression(max_iter=400).fit((Fc - mu) / sd, y).coef_[0].astype(np.float32)
        w = w / (np.linalg.norm(w) + 1e-9)
        # value: field reach-to-mate gap (z-scored by reach std) and external result gap
        pos, neg = y > 0.5, y <= 0.5
        val_z = (reach_mate[pos].mean() - reach_mate[neg].mean()) / (reach_mate.std() + 1e-9)
        res_gap = float(win[pos].mean() - win[neg].mean())
        # navigability: rollout from positions lacking the concept
        starts = []
        for i in rest:
            b = board_from_packed(P[i], M[i])
            if b.turn == chess.WHITE and b.legal_moves and named_features(b)[cname][0] <= 0.5 \
                    and not b.is_game_over():
                starts.append(b)
            if len(starts) >= args.games:
                break
        reach = {}
        for strat in ("cav", "basepolicy"):
            hits = 0
            for b0 in starts:
                b = b0.copy(stack=False)
                for p in range(args.plies):
                    if b.is_game_over():
                        break
                    if b.turn == chess.WHITE:
                        b.push(white_move(b, w, strat))
                        if named_features(b)[cname][0] > 0.5:
                            hits += 1; break
                    else:
                        b.push(opp.move(b, rng))
            reach[strat] = hits / max(1, len(starts))
        lift = reach["cav"] - reach["basepolicy"]
        score = lift * max(val_z, 0.0)
        rows.append((cname, y.mean(), reach["cav"], reach["basepolicy"], lift, val_z, res_gap, score))
        print(f"  {cname:16s} {y.mean():>3.0%} | {reach['cav']:>8.0%} {reach['basepolicy']:>6.0%} "
              f"{lift:>+4.0%} | {val_z:>+6.2f} {res_gap:>+9.2f} | {score:>6.3f}")
    print("  --- ranked subgoal codebook (navigability x value) ---")
    for r in sorted(rows, key=lambda r: -r[-1]):
        print(f"    {r[0]:16s} score {r[-1]:+.3f}  (lift {r[4]:+.0%}, value_z {r[5]:+.2f})")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
