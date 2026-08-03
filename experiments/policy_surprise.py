#!/usr/bin/env python
"""
experiments/policy_surprise.py — assess a policy head by its SURPRISE at the
MCTS result, not by human-move accuracy (Kaveh 2026-07-19: "keep the results of
the mcts and check surprise from the policy head"). This is AlphaZero's policy
target: a good prior predicts what the search converges to.

For a set of holdout positions it runs MCTS (VALUE-ONLY expansion, so the search
result is independent of the policy being graded), keeps the visit distribution
pi_search, and reports:
  KL(pi_search || pi_policy)   mean surprise (nats); lower = better prior
  top1_agree                    policy argmax == search argmax
  CE_bestmove                   -log pi_policy(search's best move)
  vs a uniform-policy baseline (the no-information reference).

Usage:
  .venv/bin/python experiments/policy_surprise.py --n 150 --nodes 800
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from catspace.research.tools.chess_specific.chessdata.encode import board_from_packed
from catspace.research.tools.chess_specific.chessdata.shards import sample_shard_rows
from catspace.io.paths import newest_shard_dir
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.eval_head import EvalHead
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import make_search_policy
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_head import PolicyHead, legal_priors, move_index


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--policy", default="data/derived/sep/cert_base_full_policy.pt")
    ap.add_argument("--n", type=int, default=150, help="holdout positions")
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cpu"
    shard_dir = newest_shard_dir()
    fb, pay = load_ckpt(Path(args.ckpt), dev); fb.eval()
    hp = torch.load(args.phead, map_location=dev, weights_only=False)
    ph = EvalHead(d_in=hp["d_in"]).to(dev); ph.load_state_dict(hp["state"]); ph.eval()
    pp = torch.load(args.policy, map_location=dev, weights_only=False)
    pol_head = PolicyHead(d_in=pp["d_in"], hidden=pp.get("hidden", 256)).to(dev)
    pol_head.load_state_dict(pp["state"]); pol_head.eval()
    omega = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]

    class Committor(torch.nn.Module):
        def forward(self, f):
            p = torch.softmax(ph(f), dim=1)
            return -torch.log(p[:, 0].clamp_min(1e-6)).unsqueeze(-1)

    # VALUE-ONLY search (no policy_fn): result is independent of the graded policy
    pol = make_search_policy("mcts", fb, pay["zgoals"]["MATE_W"], max_nodes=args.nodes,
                             device=dev, committor_head=Committor(), mate_stop=True,
                             pw_c=1.5, tactical_prior=0.25, root_min_visits=10)

    def policy_priors(board):
        from catspace.research.tools.chess_specific.chessdata.encode import encode_packed, encode_meta
        planes = feature_planes(encode_packed(board)[None], encode_meta(board)[None])
        with torch.no_grad():
            f = fb.embed_F(torch.from_numpy(planes).to(dev),
                           torch.from_numpy(np.tile(omega, (1, 1))).to(dev))
            lg = pol_head(f).cpu().numpy()[0]
        return legal_priors(lg, board)

    picks = sample_shard_rows(shard_dir, args.n * 3, seed=args.seed, holdout_only=True)
    kls, kls_u, agree, ce = [], [], [], []
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    done = 0
    for name, row in picks:
        if done >= args.n:
            break
        npz = np.load(shard_dir / name)
        board = board_from_packed(npz["packed"][row], npz["meta"][row])
        if board.is_game_over(claim_draw=True) or len(list(board.legal_moves)) < 2:
            continue
        root = pol.mcts.run(board)
        tot = sum(c.N for c in root.children)
        if tot < 4:
            continue
        pi = {c.move: c.N / tot for c in root.children}            # search visit dist
        pri = policy_priors(board)                                  # graded policy prior
        u = 1.0 / len(list(board.legal_moves))                      # uniform baseline
        kl = kl_u = 0.0
        for m, ps in pi.items():
            if ps <= 0:
                continue
            kl += ps * np.log(ps / max(pri.get(m, 1e-9), 1e-9))
            kl_u += ps * np.log(ps / u)
        best = max(pi, key=pi.get)
        kls.append(kl); kls_u.append(kl_u)
        agree.append(max(pri, key=pri.get) == best if pri else False)
        ce.append(-np.log(max(pri.get(best, 1e-9), 1e-9)))
        done += 1
    print(f"[stage] {done} positions @ {args.nodes}n: {time.time() - t0:.1f}s")
    print(f"VERDICT POLICY_SURPRISE KL(search||policy)={np.mean(kls):.3f} "
          f"(uniform baseline {np.mean(kls_u):.3f}; lower=better)  "
          f"top1_agree={np.mean(agree):.3f}  CE_bestmove={np.mean(ce):.3f}")


if __name__ == "__main__":
    main()
