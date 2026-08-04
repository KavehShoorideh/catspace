#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/conversion_field_subgoal.py -- UNSUPERVISED field-subgoal conversion (Kaveh 2026-07-21:
"no supervised probes at all; find subgoals directly off the field and navigate to them by quasimetric
distance; use the pure field without the board-structure head").

Nothing supervised: no CAV probe, no committor/eval head, no board-structure head. Only the pure quasimetric
field's own distances. Subgoals are REACHABLE waypoints (from the endgame position bank) ranked purely by the
field:
    g* = argmin_g [ d(F(s), B(g))  +  lambda * d(F(g), MATE) ]      (reachable now AND itself near mate)
Navigate to g* by minimizing the quasimetric distance d(F(s'), B(g*)) in a short adversarial search, receding
horizon (re-select every --replan plies). The tablebase is used ONLY as the exact leaf at the frontier and
for true terminals (the environment), never to guide move choice.

Rationale (JOURNAL 2026-07-21 diag_region_nav): the quasimetric to a SINGLE concrete reachable target is
sharp (d=0.04 to a real 1-move successor); the earlier failure was min-over-scattered-whole-position anchors.
Navigating to ONE field-chosen reachable subgoal at a time is the regime where quasimetric navigation works.

A = base long/short planner (navigate straight to the MATE region -- the incumbent pure-field converter).
B = this field-subgoal planner (navigate through field-chosen reachable waypoints). Same field, same search.
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


from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.planner.approaches.subgoal_cascade.experiments.planner_longshort import LongShortPlanner
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import tb_best_move
from catspace.io import paths


class FieldSubgoalPlanner(LongShortPlanner):
    """LongShortPlanner that navigates to a field-chosen reachable subgoal instead of straight to mate."""
    def setup_subgoals(self, bank_pk, bank_mt, lam, replan, n_basins=40, bank_mat=None):
        """Cluster the B-bank into BASINS (regions on B). Each basin is a subgoal region; navigate NEAR it,
        never to an exact state. Mate is itself a cluster (self.B_goal), so basin->mate closeness is a
        region-to-region distance."""
        from sklearn.cluster import KMeans
        with torch.no_grad():
            self.bank_B = self._embB(bank_pk, bank_mt)                       # (K, d)
            bank_F = self._embF(bank_pk, bank_mt)                            # (K, d)
            dmate = self.fb.distance_matrix(bank_F, self.B_goal).min(1).values.cpu().numpy()  # each -> mate cluster
        lab = KMeans(n_clusters=n_basins, n_init=4, random_state=0).fit_predict(self.bank_B.cpu().numpy())
        self.basins = []                                                     # (member_B (m,d), basin dmate)
        for c in range(n_basins):
            mem = np.flatnonzero(lab == c)
            if len(mem) == 0:
                continue
            self.basins.append((self.bank_B[torch.from_numpy(mem).to(self.dev)],
                                float(np.quantile(dmate[mem], 0.25))))       # robust basin->mate closeness
        self.basin_dmate = torch.tensor([b[1] for b in self.basins], device=self.dev)
        self.lam = lam
        self.replan = replan
        self.nav_target = None
        self.since = 10 ** 9
        self._chosen = []                                                    # dmate of picked basins

    def _reselect(self, board):
        with torch.no_grad():
            f = self._embF(encode_packed(board)[None], encode_meta(board)[None])   # (1, d)
            # basin reachability = hop to the NEAREST member (near the region, not an exact state)
            reach = torch.stack([self.fb.distance_matrix(f, mem)[0].min() for mem, _ in self.basins])
            composed = reach + self.lam * self.basin_dmate                         # reachable-near AND near mate
            c = int(composed.argmin())
        self.nav_target = self.basins[c][0]                                        # (m, d) basin region
        self._chosen.append(float(self.basin_dmate[c]))
        self._dcache.clear()
        self.since = 0

    def field_value(self, board):
        target = self.nav_target if self.nav_target is not None else self.B_goal
        key = board._transposition_key()
        v = self._dcache.get(key)
        if v is None:
            F = self._embF(encode_packed(board)[None], encode_meta(board)[None])
            with torch.no_grad():
                if self.quasi:
                    d = float(self.fb.distance_matrix(F, target).min(1).values[0])
                    v = float(np.tanh((20.0 - d) / 20.0))
                else:
                    v = float(np.tanh(float((F @ target.T).max())))
            self._dcache[key] = v
        return v

    def move(self, board):
        if len(board.piece_map()) <= self.frontier:
            return tb_best_move(board, self.tb)
        if self.since >= self.replan:
            self._reselect(board)
        self.since += 1
        return super().move(board)


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
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--ply-cap", type=int, default=80)
    ap.add_argument("--bank", type=int, default=1200)
    ap.add_argument("--n-basins", type=int, default=40)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--replan", type=int, default=4)
    ap.add_argument("--mode", default="both", choices=["both", "base", "subgoal"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time()
    fens = json.loads(Path(args.fixed_set).read_text())["fens"]
    starts = [chess.Board(f) for f in fens[args.offset:args.offset + args.n]]
    rng = np.random.default_rng(args.seed)

    print(f"VERDICT FIELD_SUBGOAL field={Path(args.field).stem} n={len(starts)} bank={args.bank} "
          f"lam={args.lam} replan={args.replan} short_depth={args.short_depth}", flush=True)
    res = {}
    if args.mode in ("both", "base"):
        base = LongShortPlanner(args.field, args.data, args.syzygy, args.frontier,
                                args.short_depth, args.qdepth, device=args.device, seed=args.seed)
        t1 = time.time()
        a = np.array([play(base, s, args.ply_cap) for s in starts])
        res["A_base_to_mate"] = a.mean()
        print(f"  A  base (navigate straight to MATE)     mate_rate={a.mean():.3f}  ({time.time()-t1:.0f}s)", flush=True)
        base.close()
    if args.mode in ("both", "subgoal"):
        sg = FieldSubgoalPlanner(args.field, args.data, args.syzygy, args.frontier,
                                 args.short_depth, args.qdepth, device=args.device, seed=args.seed)
        dz = np.load(args.dtm_npz)
        idx = rng.permutation(len(dz["packed"]))[:args.bank]
        sg.setup_subgoals(dz["packed"][idx], dz["meta"][idx], args.lam, args.replan, args.n_basins)
        t1 = time.time()
        b = np.array([play(sg, s, args.ply_cap) for s in starts])
        res["B_field_subgoal"] = b.mean()
        ch = sg._chosen or [0.0]
        print(f"  B  field-subgoal navigation             mate_rate={b.mean():.3f}  ({time.time()-t1:.0f}s)", flush=True)
        print(f"     [{len(sg.basins)} basins; picked basin->mate dmate: median {np.median(ch):.2f} "
              f"range [{min(ch):.2f},{max(ch):.2f}]  (intermediate if > mate-cluster's own spread)]", flush=True)
        sg.close()
    if "A_base_to_mate" in res and "B_field_subgoal" in res:
        print(f"  DELTA subgoal - base: {res['B_field_subgoal'] - res['A_base_to_mate']:+.3f}")
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
