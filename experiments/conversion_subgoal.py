#!/usr/bin/env python
"""experiments/conversion_subgoal.py — a SUBGOAL PLANNER (Kaveh 2026-07-19:
"the planner is key. A proper planner with proper subgoals should efficiently
make it to the mate"). Flat value-navigation caps at ~0.55 because the field's
distance is mushy over long horizons. Instead, decompose:

  each move -> pick the composed-optimal STEPPING-STONE waypoint
               g* = argmin_g [ d(F(s), B(g)) + dtm(g) ]   (closest known position
               that is itself close to mate)
            -> navigate the SHORT hop toward g* (where the field IS reliable)
            -> re-plan (receding horizon), chaining subgoals to the mate.

The subgoal is a CONCRETE nearby target (KRRvKBP -> a KRRvK-ish position -> a
KRvK-ish position -> mate), so every hop is short and the field only has to be
locally accurate. A = incumbent committor (0.567), B = subgoal planner (same
field), tablebase defender.

Usage:
  .venv/bin/python experiments/conversion_subgoal.py --n 30 --nodes 400
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.eval_head import EvalHead
from catspace.nn.features import feature_planes, omega_ids
from catspace.nn.fb import load_ckpt, pick_device
from catspace.nn.mcts import MCTS
from catspace.nn.policy_fb import make_search_policy
from experiments.playout_ab import playout
from experiments.value_fixed_point import TB


class SubgoalPlanner:
    def __init__(self, fb, bank_B, bank_dtm, dev, nodes, replan_every=3, dtm_scale=6.0):
        self.fb, self.dev = fb, dev
        self.bank_B = bank_B                                   # (K, d) tensor
        self.bank_dtm = torch.from_numpy(bank_dtm).to(dev) / dtm_scale
        self.om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
        self.subgoal = None
        self.since = 999
        self.replan_every = replan_every
        self.mcts = MCTS(self._reach, max_nodes=nodes, mate_stop=True, pw_c=1.5, root_min_visits=10)

    def _embF(self, boards):
        pk = np.stack([encode_packed(b) for b in boards])
        mt = np.stack([encode_meta(b) for b in boards])
        pl = torch.from_numpy(feature_planes(pk, mt)).to(self.dev)
        o = torch.from_numpy(np.tile(self.om, (len(boards), 1))).to(self.dev)
        with torch.no_grad():
            return self.fb.embed_F(pl, o)

    def _select(self, board):
        f = self._embF([board])                               # (1, d)
        with torch.no_grad():
            d = self.fb.distance_matrix(f, self.bank_B)[0]     # (K,) hop to each waypoint
        composed = d + self.bank_dtm                           # d(s,g)+dtm(g)/scale
        self.subgoal = self.bank_B[int(composed.argmin())]     # best stepping stone

    def _reach(self, boards):
        f = self._embF(boards)
        with torch.no_grad():
            return -self.fb.distance_matrix(f, self.subgoal[None, :])[:, 0].cpu().numpy()

    def move(self, board, rng):
        if self.subgoal is None or self.since >= self.replan_every:
            self._select(board); self.since = 0
        self.since += 1
        return self.mcts.best_move(board)


def rate(pol, starts, tb, seed, max_plies):
    return np.array([playout(pol, chess.Board(f), tb, np.random.default_rng([seed, i]), max_plies)[0]
                     for i, f in enumerate(starts)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--bank", type=int, default=800)
    ap.add_argument("--replan-every", type=int, default=3)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = pick_device(args.device)
    starts = json.loads(Path(args.fixed_set).read_text())["fens"][:args.n]
    tb = TB("data/syzygy")
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.eval()

    hp = torch.load(args.phead, map_location=dev, weights_only=False)
    ph = EvalHead(d_in=hp["d_in"]).to(dev); ph.load_state_dict(hp["state"]); ph.eval()

    class Committor(torch.nn.Module):
        def forward(self, f):
            p = torch.softmax(ph(f), dim=1)
            return (-torch.log(p[:, 0].clamp_min(1e-6))).unsqueeze(-1)
    pol_a = make_search_policy("mcts", fb, pay["zgoals"]["MATE_W"], max_nodes=args.nodes,
                               device=dev, committor_head=Committor(), mate_stop=True,
                               pw_c=1.5, root_min_visits=10)

    # waypoint bank (force low-dtm anchors in), B-embedded on this field
    dz = np.load(args.dtm_npz)
    rng = np.random.default_rng(0)
    low = np.argsort(dz["dtm"])[:args.bank // 3]
    rest = rng.choice(len(dz["dtm"]), args.bank - len(low), replace=False)
    idx = np.concatenate([low, rest])
    with torch.no_grad():
        bank_B = fb.embed_B(torch.from_numpy(feature_planes(dz["packed"][idx], dz["meta"][idx])).to(dev))
    planner = SubgoalPlanner(fb, bank_B, dz["dtm"][idx].astype(np.float32), dev,
                             args.nodes, args.replan_every)

    t0 = time.time()
    a = rate(pol_a, starts, tb, args.seed, args.max_plies)
    print(f"[A committor]        mate-rate {a.mean():.3f}  ({time.time()-t0:.0f}s)")
    t0 = time.time()
    b = rate(planner, starts, tb, args.seed, args.max_plies)
    print(f"[B subgoal planner]  mate-rate {b.mean():.3f}  ({time.time()-t0:.0f}s)")
    tb.close()
    print(f"VERDICT SUBGOAL_CONVERSION n={len(starts)} nodes={args.nodes} bank={args.bank} "
          f"A_committor={a.mean():.3f} B_subgoal={b.mean():.3f} diff={b.mean()-a.mean():+.3f}")


if __name__ == "__main__":
    main()
