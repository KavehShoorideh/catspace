#!/usr/bin/env python
"""experiments/conversion_composed_ab.py — does the COMPOSED RETRIEVAL readout
navigate a won endgame better than the incumbent? (Kaveh 2026-07-19, direction 1
after the DTM training hinge was disproven: use the composed distance as the
engine value at INFERENCE -- no new training.)

Side A: the INCUMBENT field + its committor readout (the current play).
Side B: the pure QRL field + the COMPOSED RETRIEVAL readout
        value(s) = -min_g[ d(F(s), B(g)) + d(g->mate) ] over a DTM + forced-mate
        waypoint bank (catspace/memory/retrieval.composed_distance).
Both play White (argmax hop search) vs a TABLEBASE-OPTIMAL defender from the same
KRRvKBP winning starts. Reports mate-rate A vs B (paired).

Usage:
  .venv/bin/python experiments/conversion_composed_ab.py --n 60 --nodes 400
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

from catspace.research.components.memory.approaches.vector_store_retrieval.src.retrieval import WaypointBank, composed_distance
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.eval_head import EvalHead
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import make_search_policy
from experiments.playout_ab import playout
from experiments.value_fixed_point import TB


def build_composed_bank(dtm_npz, fm_npz, n, seed=0):
    """Combined WHITE-WIN surface: tablebase DTM endgames (d=dtm) + full-board
    forced-mate white-wins (d=dtm plies). Forces low-dtm anchors in."""
    rng = np.random.default_rng(seed)
    pk, mt, dm = [], [], []
    dz = np.load(dtm_npz)
    k = n // 2
    low = np.argsort(dz["dtm"])[: k // 4]
    rest = rng.choice(len(dz["dtm"]), k - len(low), replace=False)
    idx = np.concatenate([low, rest])
    pk.append(dz["packed"][idx]); mt.append(dz["meta"][idx]); dm.append(dz["dtm"][idx])
    if Path(fm_npz).exists():
        fz = np.load(fm_npz)
        w = np.flatnonzero(fz["result"] == 1)
        j = rng.choice(w, min(n - k, len(w)), replace=False)
        pk.append(fz["packed"][j]); mt.append(fz["meta"][j]); dm.append(fz["dtm"][j])
    return WaypointBank(np.concatenate(pk), np.concatenate(mt),
                        np.concatenate(dm).astype(np.float32), "W-composed")


def side_incumbent(ckpt, phead, nodes, dev):
    fb, pay = load_ckpt(Path(ckpt), dev)
    hp = torch.load(phead, map_location=dev, weights_only=False)
    ph = EvalHead(d_in=hp["d_in"]).to(dev); ph.load_state_dict(hp["state"]); ph.eval()

    class Committor(torch.nn.Module):
        def forward(self, f):
            p = torch.softmax(ph(f), dim=1)
            return (-torch.log(p[:, 0].clamp_min(1e-6))).unsqueeze(-1)   # d to win surface
    return make_search_policy("mcts", fb, pay["zgoals"]["MATE_W"], max_nodes=nodes,
                              device=dev, committor_head=Committor(), mate_stop=True,
                              pw_c=1.5, root_min_visits=10)


def side_composed(ckpt, bank, k, nodes, dev):
    fb, pay = load_ckpt(Path(ckpt), dev)
    bank.refresh(fb, dev)

    class Composed(torch.nn.Module):
        def forward(self, f):
            return composed_distance(fb, f, bank, k=k).unsqueeze(-1)     # composed d to mate
    return make_search_policy("mcts", fb, pay["zgoals"]["MATE_W"], max_nodes=nodes,
                              device=dev, committor_head=Composed(), mate_stop=True,
                              pw_c=1.5, root_min_visits=10)


def rate(pol, starts, tb, seed, max_plies):
    mated = []
    for i, fen in enumerate(starts):
        rng = np.random.default_rng([seed, i])
        m, _ = playout(pol, chess.Board(fen), tb, rng, max_plies)
        mated.append(m)
    return np.array(mated)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt-a", default="data/derived/sep/cert_base_full.pt")
    ap.add_argument("--phead-a", default="data/derived/sep/cert_base_full_phead.pt")
    ap.add_argument("--ckpt-b", default="data/derived/sep/qrl_iqe_sn_full.pt")
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--forced-mate", default="data/derived/forced_mate.npz")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--nodes", type=int, default=400)
    ap.add_argument("--bank", type=int, default=1200)
    ap.add_argument("--k", type=int, default=24)
    ap.add_argument("--max-plies", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    import json
    starts = json.loads(Path(args.fixed_set).read_text())["fens"][:args.n]
    tb = TB("data/syzygy")
    dev = args.device
    t0 = time.time()
    a = rate(side_incumbent(args.ckpt_a, args.phead_a, args.nodes, dev), starts, tb, args.seed, args.max_plies)
    print(f"[A incumbent committor] mate-rate {a.mean():.3f}  ({time.time()-t0:.0f}s)")
    t0 = time.time()
    bank = build_composed_bank(args.dtm_npz, args.forced_mate, args.bank, args.seed)
    b = rate(side_composed(args.ckpt_b, bank, args.k, args.nodes, dev), starts, tb, args.seed, args.max_plies)
    print(f"[B QRL + composed retrieval] mate-rate {b.mean():.3f}  ({time.time()-t0:.0f}s)")
    tb.close()
    diff = b.mean() - a.mean()
    print(f"VERDICT COMPOSED_CONVERSION n={len(starts)} nodes={args.nodes} "
          f"A_incumbent={a.mean():.3f} B_qrl_composed={b.mean():.3f} diff={diff:+.3f}")


if __name__ == "__main__":
    main()
