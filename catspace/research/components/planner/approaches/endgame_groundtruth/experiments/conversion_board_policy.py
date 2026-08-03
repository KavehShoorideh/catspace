#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/conversion_board_policy.py — AlphaZero recipe: MCTS with the
BOARD-POLICY priors + committor value (Kaveh 2026-07-19 autonomous). A = incumbent
committor only (0.567). B = same committor value + board-policy priors guiding the
search. Does a tablebase-optimal-move prior improve conversion?

Usage:
  .venv/bin/python catspace/research/components/planner/approaches/endgame_groundtruth/experiments/conversion_board_policy.py --n 30 --nodes 400
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
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.eval_head import EvalHead
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import make_search_policy
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_head import legal_priors
from catspace.research.tools.stats_eval.playout_ab import playout
from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.train_board_policy import BoardPolicy
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import TB
from catspace.io import paths


def rate(pol, starts, tb, seed, max_plies):
    return np.array([playout(pol, chess.Board(f), tb, np.random.default_rng([seed, i]), max_plies)[0]
                     for i, f in enumerate(starts)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.sep("cert_base_full.pt"))
    ap.add_argument("--phead", default=paths.sep("cert_base_full_phead.pt"))
    ap.add_argument("--board-policy", default=paths.sep("board_policy.pt"))
    ap.add_argument("--fixed-set", default=paths.experiment("krrkbp_test_n200.json"))
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = pick_device(args.device)
    starts = json.loads(Path(args.fixed_set).read_text())["fens"][:args.n]
    tb = TB(str(paths.syzygy_dir()))
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.eval()
    z = pay["zgoals"]["MATE_W"]
    om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]

    hp = torch.load(args.phead, map_location=dev, weights_only=False)
    ph = EvalHead(d_in=hp["d_in"]).to(dev); ph.load_state_dict(hp["state"]); ph.eval()

    class Committor(torch.nn.Module):
        def forward(self, f):
            p = torch.softmax(ph(f), dim=1)
            return (-torch.log(p[:, 0].clamp_min(1e-6))).unsqueeze(-1)

    def embF(boards):
        pl = feature_planes(np.stack([encode_packed(b) for b in boards]),
                            np.stack([encode_meta(b) for b in boards]))
        o = np.tile(om, (len(boards), 1))
        with torch.no_grad():
            return fb.embed_F(torch.from_numpy(pl).to(dev), torch.from_numpy(o).to(dev))

    def val_fn(boards):
        with torch.no_grad():
            p = torch.softmax(ph(embF(boards)), dim=1).cpu().numpy()
        return p[:, 0] - p[:, 2]                            # white-POV committor value

    bp = torch.load(args.board_policy, map_location=dev, weights_only=False)
    pnet = BoardPolicy(bp["channels"], bp["blocks"]).to(dev); pnet.load_state_dict(bp["state"]); pnet.eval()

    def pol_fn(board):
        pl = feature_planes(encode_packed(board)[None], encode_meta(board)[None])
        with torch.no_grad():
            lg = pnet(torch.from_numpy(pl).to(dev)).cpu().numpy()[0]
        return legal_priors(lg, board)

    pol_a = make_search_policy("mcts", fb, z, max_nodes=args.nodes, device=dev,
                               committor_head=Committor(), mate_stop=True, pw_c=1.5, root_min_visits=10)
    pol_b = make_search_policy("mcts", fb, z, max_nodes=args.nodes, device=dev,
                               committor_head=Committor(), policy_fn=pol_fn, value_fn=val_fn,
                               mate_stop=True, pw_c=1.5, root_min_visits=10)

    t0 = time.time()
    a = rate(pol_a, starts, tb, args.seed, args.max_plies)
    print(f"[A committor only]        mate-rate {a.mean():.3f}  ({time.time()-t0:.0f}s)")
    t0 = time.time()
    b = rate(pol_b, starts, tb, args.seed, args.max_plies)
    print(f"[B committor + policy AZ] mate-rate {b.mean():.3f}  ({time.time()-t0:.0f}s)")
    tb.close()
    print(f"VERDICT POLICY_CONVERSION n={len(starts)} nodes={args.nodes} "
          f"A_committor={a.mean():.3f} B_policy_az={b.mean():.3f} diff={b.mean()-a.mean():+.3f}")


if __name__ == "__main__":
    main()
