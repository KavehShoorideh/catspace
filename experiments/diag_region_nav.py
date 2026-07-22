#!/usr/bin/env python
"""experiments/diag_region_nav.py -- WHY does quasimetric dist-to-region fail to navigate to the region
while the CAV succeeds? (Kaposi 2026-07-21.) Rule out a sign/direction bug, then test the "sampling +
whole-state" account directly.

Checks:
  (A) distance direction convention: for consecutive trajectory states (s -> s_next), is d(F(s),B(s_next))
      (forward, 1 move to reach) SMALLER than d(F(s_next),B(s)) (backward) and than d to a random state?
      If forward isn't smallest, distance_matrix(F,B) is not d(source->goal) and the rollout minimized the
      wrong thing.
  (B) sampling account: for positions with a legal move that CONNECTS the rooks, (1) is the field's distance
      to that connecting successor small in absolute terms (does the field know it's ~1 move)? (2) does the
      connecting move rank near the TOP among sibling moves under -dist-to-48-global-anchors, and under the
      CAV? If the CAV ranks the connecting move high but dist-to-region does not, that is the mechanism.
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
from experiments.concept_features import _connected_rooks
from experiments.steer_concept import embed_F
from sklearn.linear_model import LogisticRegression


def embed_B(fb, boards, dev):
    pk = np.stack([encode_packed(b) for b in boards]); mk = np.stack([encode_meta(b) for b in boards])
    with torch.no_grad():
        return fb.embed_B(torch.from_numpy(feature_planes(pk, mk)).to(dev))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/lichess_gn_iqeqrl_full.pt")
    ap.add_argument("--shard", required=True)
    ap.add_argument("--cav-n", type=int, default=8000)
    ap.add_argument("--anchors", type=int, default=48)
    ap.add_argument("--n-connect", type=int, default=250)
    ap.add_argument("--min-ply", type=int, default=8)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)

    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    nz = np.load(args.shard)
    P, M, ply = np.asarray(nz["packed"]), np.asarray(nz["meta"]), np.asarray(nz["ply"]).astype(int)
    gid = np.asarray(nz["game_id"])
    pool = np.flatnonzero(ply >= args.min_ply); pool = pool[rng.permutation(len(pool))]

    def F_of(boards): return embed_F(fb, boards, om, dev)

    # ---- (A) distance direction convention on real consecutive states ----
    fwd, bwd, rnd = [], [], []
    seen = 0
    order = np.argsort(gid, kind="stable")               # group same game together
    for k in range(len(order) - 1):
        i, j = order[k], order[k + 1]
        if gid[i] == gid[j] and ply[j] == ply[i] + 1:    # s -> s_next in same game
            s = board_from_packed(P[i], M[i]); sn = board_from_packed(P[j], M[j])
            u = board_from_packed(P[pool[seen % len(pool)]], M[pool[seen % len(pool)]])
            Fs = F_of([s, sn]); Bb = embed_B(fb, [sn, s, u], dev)
            with torch.no_grad():
                D = fb.distance_matrix(torch.from_numpy(Fs).float().to(dev), Bb).cpu().numpy()
            fwd.append(D[0, 0])          # d(F(s), B(s_next))  forward, 1 move
            bwd.append(D[1, 1])          # d(F(s_next), B(s))  backward
            rnd.append(D[0, 2])          # d(F(s), B(random))
            seen += 1
        if seen >= 400:
            break
    print(f"VERDICT DIAG_REGION_NAV field={Path(args.field).stem}")
    print(f"  (A) direction convention over {seen} consecutive (s->s_next) pairs, distance_matrix(F,B):")
    print(f"      d(F(s),B(s_next)) fwd/1-move = {np.mean(fwd):.2f}   d(F(s_next),B(s)) bwd = {np.mean(bwd):.2f}"
          f"   d(F(s),B(random)) = {np.mean(rnd):.2f}")
    print(f"      -> forward {'IS' if np.mean(fwd) < np.mean(bwd) and np.mean(fwd) < np.mean(rnd) else 'IS NOT'}"
          f" smallest (expected: forward smallest if distance_matrix = d(source->goal))")

    # ---- CAV + 48 global anchors ----
    cav_idx = pool[:args.cav_n]
    cav_boards = [board_from_packed(P[i], M[i]) for i in cav_idx]
    Fc = F_of(cav_boards); mu, sd = Fc.mean(0), Fc.std(0) + 1e-8
    yc = np.array([_connected_rooks(b, chess.WHITE) for b in cav_boards], float)
    w = LogisticRegression(max_iter=400).fit((Fc - mu) / sd, yc).coef_[0].astype(np.float32)
    w = w / (np.linalg.norm(w) + 1e-9)
    posb = [b for b, y in zip(cav_boards, yc) if y > 0.5]; rng.shuffle(posb)
    Breg = embed_B(fb, posb[:args.anchors], dev)

    # ---- (B) positions with a legal rook-connecting move ----
    d_connect, d_global, rank_reg, rank_cav, top1_reg, top1_cav = [], [], [], [], [], []
    got = 0
    for i in pool[args.cav_n:]:
        b = board_from_packed(P[i], M[i])
        if b.turn != chess.WHITE or _connected_rooks(b, chess.WHITE) or b.is_game_over():
            continue
        moves = list(b.legal_moves); kids = []
        for m in moves:
            c = b.copy(stack=False); c.push(m); kids.append(c)
        conn = [k for k, c in enumerate(kids) if _connected_rooks(c, chess.WHITE)]
        if not conn:
            continue
        Fk = F_of(kids)
        with torch.no_grad():
            dg = fb.distance_matrix(torch.from_numpy(Fk).float().to(dev), Breg).min(1).values.cpu().numpy()
        cav = ((Fk - mu) / sd) @ w
        # distance from the PARENT to its own connecting successor (should be ~1 move)
        Bconn = embed_B(fb, [kids[conn[0]]], dev)
        Fpar = F_of([b])
        with torch.no_grad():
            d_connect.append(float(fb.distance_matrix(torch.from_numpy(Fpar).float().to(dev), Bconn)[0, 0]))
        d_global.append(float(dg.min()))
        # rank of the best connecting child among siblings (1.0 = top). higher reach := -dg ; higher cav
        reg_score = -dg;
        best_conn = max(conn, key=lambda k: reg_score[k])
        rank_reg.append(float((reg_score <= reg_score[best_conn]).mean()))
        top1_reg.append(int(np.argmax(reg_score) in conn))
        best_conn_c = max(conn, key=lambda k: cav[k])
        rank_cav.append(float((cav <= cav[best_conn_c]).mean()))
        top1_cav.append(int(np.argmax(cav) in conn))
        got += 1
        if got >= args.n_connect:
            break
    print(f"  (B) {got} positions with a 1-move rook-connection available:")
    print(f"      field d(parent -> its own connecting successor) = {np.mean(d_connect):.2f}"
          f"   vs  min d(parent -> 48 global anchors) = {np.mean(d_global):.2f}")
    print(f"      connecting move's percentile rank among siblings:  dist-to-region {np.mean(rank_reg):.2f}"
          f"   |  CAV {np.mean(rank_cav):.2f}   (1.0 = ranked top)")
    print(f"      connecting move is the ARGMAX (top-1 pick):        dist-to-region {np.mean(top1_reg):.0%}"
          f"   |  CAV {np.mean(top1_cav):.0%}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
