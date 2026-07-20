#!/usr/bin/env python
"""experiments/conversion_dtmhead_ab.py — SAME field, committor vs DTM-head
readout (Kaveh 2026-07-19 autonomous). Isolates the READOUT: does navigating by
a decoupled DTM head (predicted distance-to-mate, which HAS a gradient) convert
better than the flat committor P(win)? Both play White (argmax hop search) on
the incumbent field vs a tablebase defender from the same KRRvKBP starts.

Usage:
  .venv/bin/python experiments/conversion_dtmhead_ab.py --n 30 --nodes 400
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

from catspace.nn.eval_head import EvalHead
from catspace.nn.fb import load_ckpt, pick_device
from catspace.nn.policy_fb import make_search_policy
from experiments.playout_ab import playout
from experiments.train_dtm_head import DTMHead
from experiments.value_fixed_point import TB


def rate(pol, starts, tb, seed, max_plies):
    out = []
    for i, fen in enumerate(starts):
        m, _ = playout(pol, chess.Board(fen), tb, np.random.default_rng([seed, i]), max_plies)
        out.append(m)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--dtmhead", default="data/derived/sep/cert_base_full_dtmhead.pt")
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
    z = pay["zgoals"]["MATE_W"]

    hp = torch.load(args.phead, map_location=dev, weights_only=False)
    ph = EvalHead(d_in=hp["d_in"]).to(dev); ph.load_state_dict(hp["state"]); ph.eval()

    class Committor(torch.nn.Module):
        def forward(self, f):
            p = torch.softmax(ph(f), dim=1)
            return (-torch.log(p[:, 0].clamp_min(1e-6))).unsqueeze(-1)

    dp = torch.load(args.dtmhead, map_location=dev, weights_only=False)
    dh = DTMHead(dp["d_in"], dp.get("hidden", 256)).to(dev); dh.load_state_dict(dp["state"]); dh.eval()

    class DTMValue(torch.nn.Module):
        def forward(self, f):
            return dh(f).unsqueeze(-1)                       # predicted DTM; reach=-DTM -> toward mate

    def pol_of(head):
        return make_search_policy("mcts", fb, z, max_nodes=args.nodes, device=dev,
                                  committor_head=head, mate_stop=True, pw_c=1.5, root_min_visits=10)

    t0 = time.time()
    a = rate(pol_of(Committor()), starts, tb, args.seed, args.max_plies)
    print(f"[A committor]  mate-rate {a.mean():.3f}  ({time.time()-t0:.0f}s)")
    t0 = time.time()
    b = rate(pol_of(DTMValue()), starts, tb, args.seed, args.max_plies)
    print(f"[B DTM head]   mate-rate {b.mean():.3f}  ({time.time()-t0:.0f}s)")
    tb.close()
    print(f"VERDICT DTMHEAD_CONVERSION n={len(starts)} nodes={args.nodes} "
          f"A_committor={a.mean():.3f} B_dtmhead={b.mean():.3f} diff={b.mean()-a.mean():+.3f}")


if __name__ == "__main__":
    main()
