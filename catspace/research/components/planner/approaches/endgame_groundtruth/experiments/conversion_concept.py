#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/conversion_concept.py -- can a CAV-climb CONCEPT SUBGOAL improve KRRvKBP conversion?
(Kaveh 2026-07-21: "see if you can convert the toy examples.")

Prior threads found field-DISTANCE / composed-waypoint subgoal navigation FAILS to beat the incumbent
converter (conversion_subgoal.py, conversion_composed_ab.py) and that conversion is SEARCH-limited. Today's
diagnostic (diag_region_nav.py) showed WHY distance-to-region can't navigate to an attribute -- and that
climbing the concept CAV (the attribute direction itself) CAN (connected_rooks reach 28% vs 6%, 4.7x). So
this retries subgoal-guided conversion with the RIGHT primitive: blend a connected_rooks CAV-climb term into
the long/short planner's leaf value. connected_rooks is the decisive coordination structure in a two-rook
mate, so it is the natural subgoal here.

leaf = (1 - alpha) * v_base + alpha * tanh(cav_z)      (convex blend -> stays in [-1,1], real tablebase
terminals at +-1 still dominate; alpha=0 recovers the incumbent long/short planner exactly.)

The CAV is extracted IN-DOMAIN: endgame positions embedded with the SAME field and the SAME planes the
planner uses (BOARD_ONLY channels zeroed), labelled by connected_rooks(White). A/B on the fixed KRRvKBP
test set; opponent = tablebase-optimal.
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


from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed, encode_meta, encode_packed
from catspace.research.components.encoder.approaches.concept_quantization.experiments.concept_features import _connected_rooks
from catspace.research.components.planner.approaches.subgoal_cascade.experiments.planner_longshort import LongShortPlanner
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import tb_best_move
from sklearn.linear_model import LogisticRegression
from catspace.io import paths


class ConceptPlanner(LongShortPlanner):
    alpha = 0.0
    cav = None                                             # (w, mu, sd, z_mu, z_sd) or None

    def field_value(self, board):
        key = board._transposition_key()
        v = self._dcache.get(key)
        if v is None:
            F = self._embF(encode_packed(board)[None], encode_meta(board)[None])   # (1, d)
            with torch.no_grad():
                if self.quasi:
                    d = float(self.fb.distance_matrix(F, self.B_goal).min(1).values[0])
                    vbase = float(np.tanh((20.0 - d) / 20.0))
                else:
                    vbase = float(np.tanh(float((F @ self.B_goal.T).max())))
                if self.alpha and self.cav is not None:
                    w, mu, sd, zmu, zsd = self.cav
                    proj = float(((F[0] - mu) / sd) @ w)
                    cavz = float(np.tanh((proj - zmu) / (zsd + 1e-9)))
                    v = (1.0 - self.alpha) * vbase + self.alpha * cavz
                else:
                    v = vbase
            self._dcache[key] = v
        return v


def fit_cav(planner, dtm_npz, n, seed):
    """connected_rooks CAV in the planner's F-space (planes match: BOARD_ONLY zeroed via _embF)."""
    dz = np.load(dtm_npz)
    P, M = np.asarray(dz["packed"]), np.asarray(dz["meta"])
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(P))[:n]
    boards = [board_from_packed(P[i], M[i]) for i in idx]
    y = np.array([_connected_rooks(b, chess.WHITE) for b in boards], float)
    with torch.no_grad():
        F = planner._embF(P[idx], M[idx]).cpu().numpy()
    mu, sd = F.mean(0), F.std(0) + 1e-8
    if not (0.03 < y.mean() < 0.97):
        return None, y.mean()
    w = LogisticRegression(max_iter=400).fit((F - mu) / sd, y).coef_[0].astype(np.float32)
    w = w / (np.linalg.norm(w) + 1e-9)
    proj = ((F - mu) / sd) @ w
    dev = planner.dev
    cav = (torch.tensor(w, device=dev), torch.tensor(mu, device=dev, dtype=torch.float32),
           torch.tensor(sd, device=dev, dtype=torch.float32), float(proj.mean()), float(proj.std()))
    return cav, y.mean()


def play(planner, start, ply_cap):
    b = start.copy(stack=False)
    for _ in range(ply_cap):
        if b.is_game_over(claim_draw=True):
            return 1.0 if (b.is_checkmate() and b.turn == chess.BLACK) else 0.0
        m = planner.move(b) if b.turn == chess.WHITE else tb_best_move(b, planner.tb)
        if m is None:
            return 0.0
        b.push(m)
    return 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--field", default=paths.sep("iqe_nucleus_gn.pt"))
    ap.add_argument("--data", default=paths.derived("stratified_perfect.npz"))
    ap.add_argument("--dtm-npz", default=paths.derived("dtm_endgame.npz"))
    ap.add_argument("--fixed-set", default=paths.experiment("krrkbp_test_n200.json"))
    ap.add_argument("--syzygy", default=str(paths.syzygy_dir()))
    ap.add_argument("--frontier", type=int, default=5)
    ap.add_argument("--short-depth", type=int, default=3)
    ap.add_argument("--qdepth", type=int, default=0)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--offset", type=int, default=0, help="start slice offset into the fixed set")
    ap.add_argument("--ply-cap", type=int, default=80)
    ap.add_argument("--cav-n", type=int, default=6000)
    ap.add_argument("--alphas", default="0.0,0.3")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    _fens = json.loads(Path(args.fixed_set).read_text())["fens"]
    starts = [chess.Board(f) for f in _fens[args.offset:args.offset + args.n]]
    alphas = [float(a) for a in args.alphas.split(",")]

    planner = ConceptPlanner(args.field, args.data, args.syzygy, args.frontier,
                             args.short_depth, args.qdepth, device=args.device, seed=args.seed)
    cav, prev = fit_cav(planner, args.dtm_npz, args.cav_n, args.seed)
    print(f"[cav] connected_rooks base-rate {prev:.0%} in endgame domain, cav={'ok' if cav else 'DEGENERATE'} "
          f"({time.time()-t0:.0f}s)", flush=True)
    planner.cav = cav

    print(f"VERDICT CONVERSION_CONCEPT field={Path(args.field).stem} n={len(starts)} "
          f"short_depth={args.short_depth} frontier={args.frontier}")
    res = {}
    for a in alphas:
        planner.alpha = a; planner._dcache.clear()
        t1 = time.time()
        mates = np.array([play(planner, s, args.ply_cap) for s in starts])
        res[a] = mates.mean()
        print(f"  alpha={a:.2f}  mate_rate={mates.mean():.3f}  ({time.time()-t1:.0f}s)", flush=True)
    base = res[alphas[0]]
    for a in alphas[1:]:
        print(f"  DELTA alpha={a:.2f}: {res[a]-base:+.3f} vs alpha=0")
    planner.close()
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
