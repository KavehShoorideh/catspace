#!/usr/bin/env python
"""experiments/concept_reach_rollout.py -- the linchpin "concept as a navigable subgoal" test
(2026-07-21, design chosen by Kaposi: base-FB opponent, greedy-1-ply white steering).

From positions where White does NOT have the concept (connected_rooks), roll the game out K plies with the
OPPONENT playing the base FB reach policy (a competent adversary) and WHITE choosing its move greedily on a
SUBGOAL score. If the value field is a navigable multi-step cost-to-go, greedily descending the quasimetric
reach-to-region should steer the game INTO the concept region more often than White just playing its normal
game. That is the claim that justifies wiring concept regions as planner subgoals; the 1-ply actionability
test (steer_concept.py) only weakly supported it (CAV move-AUC 0.658), and a cost-to-go must be judged over
multiple steps.

White strategies compared (opponent = base FB policy, depth 1, throughout):
  * reach2region -- min quasimetric d(F(child), B-region)   [the subgoal mechanism under test]
  * cav          -- max w_c . F(child)                        [the linear concept gradient]
  * basepolicy   -- base FB reach policy toward MATE_W        [control: White just plays the game]
  * random       -- random legal move                          [floor]
Reports reached-within-K rate and median plies-to-reach. If reach2region/cav >> basepolicy ~ random, the
concept is a navigable subgoal, not a byproduct of good play.
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

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import FBBoardPolicy
from experiments.concept_features import _connected_rooks
from experiments.steer_concept import embed_F
from experiments.concept_features import features as named_features
from sklearn.linear_model import LogisticRegression


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--cav-n", type=int, default=9000)
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--plies", type=int, default=10)
    ap.add_argument("--region-anchors", type=int, default=48)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    fb, extra = load_ckpt(Path(args.field), dev); fb.eval()
    zg = extra["zgoals"] if isinstance(extra, dict) and "zgoals" in extra else \
        torch.load(args.field, map_location="cpu", weights_only=False)["zgoals"]
    def _vec(x):
        return (x.detach().float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x, np.float32))
    zW_np, zB_np = _vec(zg["MATE_W"]), _vec(zg["MATE_B"])
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]

    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    pool = np.flatnonzero(ply >= args.min_ply); pool = pool[rng.permutation(len(pool))]

    # CAV + region anchors on a disjoint pool
    cav_idx = pool[:args.cav_n]
    cav_boards = [board_from_packed(P[i], M[i]) for i in cav_idx]
    Fc = embed_F(fb, cav_boards, om, dev)
    mu, sd = Fc.mean(0), Fc.std(0) + 1e-8
    yc = np.array([_connected_rooks(b, chess.WHITE) for b in cav_boards], float)
    w = LogisticRegression(max_iter=400).fit((Fc - mu) / sd, yc).coef_[0].astype(np.float32)
    w = w / (np.linalg.norm(w) + 1e-9)
    posb = [b for b, yy in zip(cav_boards, yc) if yy > 0.5]; rng.shuffle(posb)
    posb = posb[:args.region_anchors]
    pk = np.stack([encode_packed(b) for b in posb]); mk = np.stack([encode_meta(b) for b in posb])
    with torch.no_grad():
        Breg = fb.embed_B(torch.from_numpy(feature_planes(pk, mk)).to(dev))

    # start positions: White to move, White lacks connected rooks
    starts = []
    for i in pool[args.cav_n:]:
        b = board_from_packed(P[i], M[i])
        if b.turn == chess.WHITE and not _connected_rooks(b, chess.WHITE) and not b.is_game_over():
            starts.append(b)
        if len(starts) >= args.games:
            break

    opp = FBBoardPolicy(fb, zB_np, depth=1, device=dev)      # black = base FB reach policy
    basepol_w = FBBoardPolicy(fb, zW_np, depth=1, device=dev)

    def child_F(board, moves):
        kids = []
        for m in moves:
            c = board.copy(stack=False); c.push(m); kids.append(c)
        return embed_F(fb, kids, om, dev), kids

    def white_move(board, strat):
        moves = list(board.legal_moves)
        if strat == "random":
            return moves[rng.integers(len(moves))]
        if strat == "basepolicy":
            return basepol_w.move(board, rng)
        Fk, _ = child_F(board, moves)
        if strat == "cav":
            return moves[int(np.argmax(((Fk - mu) / sd) @ w))]
        if strat == "reach2region":
            with torch.no_grad():
                d = fb.distance_matrix(torch.from_numpy(Fk).float().to(dev), Breg).min(1).values.cpu().numpy()
            return moves[int(np.argmin(d))]
        raise ValueError(strat)

    strategies = ["reach2region", "cav", "basepolicy", "random"]
    reached = {s: [] for s in strategies}          # plies-to-reach or np.inf
    for s in strategies:
        for b0 in starts:
            b = b0.copy(stack=False); hit = np.inf
            for p in range(args.plies):
                if b.is_game_over():
                    break
                if b.turn == chess.WHITE:
                    m = white_move(b, s); b.push(m)
                    if _connected_rooks(b, chess.WHITE):
                        hit = p + 1; break
                else:
                    b.push(opp.move(b, rng))
            reached[s].append(hit)

    print(f"VERDICT CONCEPT_REACH field={Path(args.field).stem} concept=connected_rooks_w "
          f"games={len(starts)} plies={args.plies} opponent=base_FB_policy")
    print(f"  {'white strategy':14s} | {'reached<=K':>10s} | {'median plies':>12s}")
    for s in strategies:
        arr = np.array(reached[s]); rate = float(np.mean(np.isfinite(arr)))
        fin = arr[np.isfinite(arr)]; med = float(np.median(fin)) if len(fin) else float("nan")
        print(f"  {s:14s} | {rate:>10.0%} | {med:>12.1f}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
