#!/usr/bin/env python
"""experiments/conversion_field_mcts.py -- TABLEBASE-FREE, field-guided MCTS for the toy mate
(Kaveh 2026-07-21: "try the mate without leveraging the tablebase; MCTS guided by the model; clusters on B
as subgoals; get closer to those subgoals; use F to see which subgoals are reachable").

White's search uses ONLY the model (no tablebase leaf, no tablebase move oracle). Architecture:
  * subgoals = CLUSTERS on B (basins of the endgame bank).
  * mate target = the "exposed-cornered-king" basin -- identified GEOMETRICALLY (black king on edge AND few
    escape squares = cornered & exposed), so no tablebase; this is Kaveh's assumed concept, used not validated.
  * F(s)  says which subgoal is REACHABLE from here:  reach_i = min d(F(s), B(basin_i)).
  * B says which subgoal LEADS TO MATE:               mate_i  = min d(F(basin_i), B(mate_basin)).
  * pick the basin minimizing reach_i + lam*mate_i  (the F/B intersection), receding horizon.
  * a field-guided MCTS (value = progress toward the chosen subgoal cluster) executes toward it; mate_stop
    catches real checkmates. NO tablebase anywhere in White's decision.
Black defends tablebase-optimally (the environment/defender). Scored on the KRRvKBP toy: mate_rate, and how
cornered the black king ends up (progress), since a tablebase-free mate vs perfect defense is a hard bar.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.search.approaches.puct_mcts.src.mcts import MCTS
from experiments.value_fixed_point import TB, tb_best_move


def cornered_exposed(b):
    """Kaveh's concept: black king on the edge AND confined (few escapes) = cornered & exposed (mating zone)."""
    k = b.king(chess.BLACK)
    if k is None:
        return 0.0
    r, f = chess.square_rank(k), chess.square_file(k)
    on_edge = (r in (0, 7) or f in (0, 7))
    esc = 0
    for sq in b.attacks(k):
        p = b.piece_at(sq)
        if p is not None and p.color == chess.BLACK:
            continue
        if b.is_attacked_by(chess.WHITE, sq):
            continue
        esc += 1
    return float(on_edge and esc <= 2)


class FieldMCTS:
    def __init__(self, fb, dev, bank_pk, bank_mt, nodes, n_basins, lam, replan):
        self.fb, self.dev, self.lam, self.replan = fb, dev, lam, replan
        self.om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
        Bb = self._embB(bank_pk, bank_mt); Fb = self._embF(bank_pk, bank_mt)
        lab = KMeans(n_clusters=n_basins, n_init=4, random_state=0).fit_predict(Bb.cpu().numpy())
        corner = np.array([cornered_exposed(board_from_packed(bank_pk[i], bank_mt[i])) for i in range(len(bank_pk))])
        self.basins = []                                    # (B_members, mean_corner)
        for c in range(n_basins):
            m = np.flatnonzero(lab == c)
            if len(m):
                self.basins.append((Bb[torch.from_numpy(m).to(dev)], float(corner[m].mean()),
                                    Fb[torch.from_numpy(m).to(dev)]))
        mate_c = int(np.argmax([b[1] for b in self.basins]))
        self.mate_B = self.basins[mate_c][0]                # exposed-cornered-king cluster on B
        with torch.no_grad():                               # each basin's field distance to the mate cluster
            self.basin_mate = torch.stack([self.fb.distance_matrix(Fm, self.mate_B).min() for _, _, Fm in self.basins])
        self.mate_corner = self.basins[mate_c][1]
        self.subgoal_B = None; self.since = 10 ** 9
        self.mcts = MCTS(self._reach, max_nodes=nodes, mate_stop=True, pw_c=1.5, root_min_visits=10)

    def _embF(self, pk, mt):
        with torch.no_grad():
            return self.fb.embed_F(torch.from_numpy(feature_planes(pk, mt)).to(self.dev),
                                   torch.from_numpy(np.tile(self.om, (len(pk), 1))).to(self.dev))

    def _embB(self, pk, mt):
        with torch.no_grad():
            return self.fb.embed_B(torch.from_numpy(feature_planes(pk, mt)).to(self.dev))

    def _select(self, board):
        f = self._embF(encode_packed(board)[None], encode_meta(board)[None])   # (1,d)
        with torch.no_grad():
            reach = torch.stack([self.fb.distance_matrix(f, Bm)[0].min() for Bm, _, _ in self.basins])
            composed = reach + self.lam * self.basin_mate                       # F-reachable AND B-leads-to-mate
            self.subgoal_B = self.basins[int(composed.argmin())][0]
        self.since = 0

    def _reach(self, boards):
        if getattr(self, "pure_search", False):
            return np.zeros(len(boards), dtype=np.float32)               # constant value -> MCTS+mate_stop only
        f = self._embF(np.stack([encode_packed(b) for b in boards]),
                       np.stack([encode_meta(b) for b in boards]))
        with torch.no_grad():
            return -self.fb.distance_matrix(f, self.subgoal_B).min(1).values.cpu().numpy()  # progress toward subgoal

    def move(self, board, rng):
        if self.subgoal_B is None or self.since >= self.replan:
            self._select(board)
        self.since += 1
        return self.mcts.best_move(board)


def play(planner, start, tb, rng, ply_cap, white_random=False):
    b = start.copy(stack=False); best_corner = 0.0
    for _ in range(ply_cap):
        if b.is_game_over(claim_draw=True):
            break
        if b.turn == chess.WHITE:
            if white_random:                                # baseline control: is cornering just incidental?
                lm = list(b.legal_moves); m = lm[rng.integers(len(lm))]
            else:
                m = planner.move(b, rng)
        else:
            m = tb_best_move(b, tb)
        if m is None:
            break
        b.push(m); best_corner = max(best_corner, cornered_exposed(b))
    out = b.outcome(claim_draw=True)
    mated = 1.0 if (out and out.winner == chess.WHITE) else 0.0
    return mated, best_corner


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default="data/derived/sep/xfer_treat.pt")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--syzygy", default="data/syzygy")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--bank", type=int, default=1500)
    ap.add_argument("--n-basins", type=int, default=40)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--replan", type=int, default=4)
    ap.add_argument("--ply-cap", type=int, default=100)
    ap.add_argument("--white-random", action="store_true", help="baseline: White plays random (is cornering incidental?)")
    ap.add_argument("--pure-search", action="store_true", help="baseline: constant leaf value -> MCTS+mate_stop only (no field guidance)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    fb, _ = load_ckpt(Path(args.field), dev); fb.eval()
    tb = TB(args.syzygy)
    fens = json.loads(Path(args.fixed_set).read_text())["fens"]
    starts = [chess.Board(f) for f in fens[args.offset:args.offset + args.n]]
    dz = np.load(args.dtm_npz); idx = rng.permutation(len(dz["packed"]))[:args.bank]
    pl = FieldMCTS(fb, dev, dz["packed"][idx], dz["meta"][idx], args.nodes, args.n_basins, args.lam, args.replan)
    pl.pure_search = args.pure_search
    print(f"VERDICT FIELD_MCTS field={Path(args.field).stem} n={len(starts)} nodes={args.nodes} "
          f"basins={len(pl.basins)} mate_basin_corner={pl.mate_corner:.2f} (NO tablebase in White's search)", flush=True)
    start_corner = float(np.mean([cornered_exposed(s) for s in starts]))
    mates, corners = [], []
    for s in starts:
        mt, cr = play(pl, s, tb, rng, args.ply_cap, white_random=args.white_random); mates.append(mt); corners.append(cr)
    tb.close()
    tag = "RANDOM-White baseline" if args.white_random else "field-MCTS"
    print(f"  [{tag}] mate_rate {np.mean(mates):.3f}  |  cornered-king reached {np.mean(corners):.2f} "
          f"(start {start_corner:.2f})  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
