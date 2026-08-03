#!/usr/bin/env python
"""experiments/conversion_board_dtm.py — navigate a won endgame by the BOARD-DTM
net (Kaveh 2026-07-19 autonomous). The board net predicts DTM far better than the
DTM-poor F (krvk 0.93, krrvk 0.68, krrkbp 0.29). Since conversion SIMPLIFIES
toward the well-predicted sub-endgames, navigating by -DTM may beat the flat
committor. A = incumbent committor (0.567), B = board-DTM navigation.

Usage:
  .venv/bin/python experiments/conversion_board_dtm.py --n 30 --nodes 400
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

from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.eval_head import EvalHead
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.encoder import BoardEncoder  # noqa: F401 (used via BoardDTM)
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.search.approaches.puct_mcts.src.mcts import MCTS
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import make_search_policy
from experiments.playout_ab import playout
from experiments.train_board_dtm import BoardDTM
from experiments.value_fixed_point import TB


class BoardDTMPolicy:
    """Raw MCTS navigating by reach = -board_dtm(planes) (toward min DTM = mate)."""

    def __init__(self, net, dev, nodes):
        self.net, self.dev = net, dev
        self.mcts = MCTS(self._reach, max_nodes=nodes, mate_stop=True,
                         pw_c=1.5, root_min_visits=10)

    def _reach(self, boards):
        pk = np.stack([encode_packed(b) for b in boards])
        mt = np.stack([encode_meta(b) for b in boards])
        pl = torch.from_numpy(feature_planes(pk, mt)).to(self.dev)
        with torch.no_grad():
            return -self.net(pl).cpu().numpy()

    def move(self, board, rng):
        return self.mcts.best_move(board)


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

    bp = torch.load(args.board_dtm, map_location=dev, weights_only=False)
    net = BoardDTM(bp["channels"], bp["blocks"]).to(dev); net.load_state_dict(bp["state"]); net.eval()
    pol_b = BoardDTMPolicy(net, dev, args.nodes)

    t0 = time.time()
    a = rate(pol_a, starts, tb, args.seed, args.max_plies)
    print(f"[A committor]     mate-rate {a.mean():.3f}  ({time.time()-t0:.0f}s)")
    t0 = time.time()
    b = rate(pol_b, starts, tb, args.seed, args.max_plies)
    print(f"[B board-DTM nav] mate-rate {b.mean():.3f}  ({time.time()-t0:.0f}s)")
    tb.close()
    print(f"VERDICT BOARDDTM_CONVERSION n={len(starts)} nodes={args.nodes} "
          f"A_committor={a.mean():.3f} B_boarddtm={b.mean():.3f} diff={b.mean()-a.mean():+.3f}")


if __name__ == "__main__":
    main()
