#!/usr/bin/env python
"""experiments/conversion_ensemble.py — committor + board-DTM ENSEMBLE value
(Kaveh 2026-07-19 autonomous). The failure diagnosis: the committor fails ~50/50
by (1) blundering material (material-blind) and (2) failing to mate a won
position (no mate gradient). The board-DTM net has a material+mate gradient but
is weak on KRRvKBP alone. Combine: reach = (committor W-L) - w*board_dtm -- the
committor keeps it winning/safe, the DTM adds the gradient toward mate. Both are
LEARNED (no hand-coded material guard). A = committor, B = ensemble.

Usage:
  .venv/bin/python experiments/conversion_ensemble.py --n 30 --nodes 400 --w 0.15
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
from experiments.train_board_dtm import BoardDTM
from experiments.value_fixed_point import TB


def rate(pol, starts, tb, seed, max_plies):
    return np.array([playout(pol, chess.Board(f), tb, np.random.default_rng([seed, i]), max_plies)[0]
                     for i, f in enumerate(starts)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--board-dtm", default="data/derived/sep/board_dtm.pt")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--w", type=float, default=0.15, help="board-DTM weight in the ensemble")
    ap.add_argument("--dtm-scale", type=float, default=20.0)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = pick_device(args.device)
    starts = json.loads(Path(args.fixed_set).read_text())["fens"][:args.n]
    tb = TB("data/syzygy")
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.eval()
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
    hp = torch.load(args.phead, map_location=dev, weights_only=False)
    ph = EvalHead(d_in=hp["d_in"]).to(dev); ph.load_state_dict(hp["state"]); ph.eval()
    bp = torch.load(args.board_dtm, map_location=dev, weights_only=False)
    net = BoardDTM(bp["channels"], bp["blocks"]).to(dev); net.load_state_dict(bp["state"]); net.eval()

    class Committor(torch.nn.Module):
        def forward(self, f):
            p = torch.softmax(ph(f), dim=1)
            return (-torch.log(p[:, 0].clamp_min(1e-6))).unsqueeze(-1)
    pol_a = make_search_policy("mcts", fb, pay["zgoals"]["MATE_W"], max_nodes=args.nodes,
                               device=dev, committor_head=Committor(), mate_stop=True,
                               pw_c=1.5, root_min_visits=10)

    class EnsemblePolicy:
        def __init__(self, nodes):
            self.mcts = MCTS(self._reach, max_nodes=nodes, mate_stop=True, pw_c=1.5, root_min_visits=10)

        def _reach(self, boards):
            pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
            pl = torch.from_numpy(feature_planes(pk, mt)).to(dev)
            o = torch.from_numpy(np.tile(om, (len(boards), 1))).to(dev)
            with torch.no_grad():
                f = fb.embed_F(pl, o)
                p = torch.softmax(ph(f), dim=1)
                committor = (p[:, 0] - p[:, 2]).cpu().numpy()             # white-POV W-L
                dtm = net(pl).cpu().numpy()                              # plies to white mate
            return committor - args.w * dtm / args.dtm_scale             # higher=better for white

        def move(self, board, rng):
            return self.mcts.best_move(board)

    t0 = time.time()
    a = rate(pol_a, starts, tb, args.seed, args.max_plies)
    print(f"[A committor]  mate-rate {a.mean():.3f}  ({time.time()-t0:.0f}s)")
    t0 = time.time()
    b = rate(EnsemblePolicy(args.nodes), starts, tb, args.seed, args.max_plies)
    print(f"[B ensemble]   mate-rate {b.mean():.3f}  ({time.time()-t0:.0f}s)")
    tb.close()
    print(f"VERDICT ENSEMBLE_CONVERSION n={len(starts)} nodes={args.nodes} w={args.w} "
          f"A_committor={a.mean():.3f} B_ensemble={b.mean():.3f} diff={b.mean()-a.mean():+.3f}")


if __name__ == "__main__":
    main()
