#!/usr/bin/env python
"""experiments/m5_mcts_probe.py -- M5 probe: thin CLI over the modular stack in
catspace/probe (Kaveh 2026-07-30: end-to-end first, then modularize to iterate).

Components (each independently swappable; see catspace/probe/__init__.py):
  Encoder=frozen Leela trunk | ReachModel=--field ckpt | Atlas=--table |
  Planner=--planner (chute = threshold-free chain-of-chutes value iteration) |
  Navigator=MCTS with --leaf {reach,committor} backups, --order {tiered,none}
  descent, --opp-model maia2 priors | Harness=shared instruments + VERDICTs.

History of the algorithm choices (design docstrings live with the components):
chain-of-chutes + tiered order + value authority + agentive field -- see
JOURNAL.md 2026-07-29/30 and the read ladder (0.045 -> 0.095).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess.engine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.atlas import RegionAtlas, SubgoalRanker                   # noqa: E402
from catspace.encoder import ReachabilityField                          # noqa: E402
from catspace.harness import run_games                                  # noqa: E402
from catspace.navigator import MCTSNavigator                            # noqa: E402
from catspace.opponent import make_maia2_policy                         # noqa: E402
from catspace.planner import PLANNERS                                   # noqa: E402
from catspace.reach import RegionReach                                  # noqa: E402
from catspace.train.scaffold import resolve_device                      # noqa: E402
from catspace.value import CommittorGreedy                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--field", default="artifacts/experiments/reach_v3_full_latest.pt")
    ap.add_argument("--ckpt", default="artifacts/experiments/field_fullgame_v3_final.pt",
                    help="committor ckpt (tier-3 ordering / committor leaf); '' = none")
    ap.add_argument("--planner", default="chute", choices=sorted(PLANNERS))
    ap.add_argument("--leaf", default="reach", choices=["reach", "committor"],
                    help="MCTS backup value (committor = value authority)")
    ap.add_argument("--order", default="tiered", choices=["tiered", "none"],
                    help="descent order (none + committor leaf = the WDL ablation)")
    ap.add_argument("--opp-model", type=int, default=1,
                    help="1 = maia2 priors at opponent tree nodes; 0 = adversarial")
    ap.add_argument("--reach", default="data/derived/reach/reach_v3.npz")
    ap.add_argument("--table", default="data/derived/reach/region_table_v3.npz")
    ap.add_argument("--maia-elo", type=int, default=1100)
    ap.add_argument("--our-elo", type=float, default=1800.0)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--nodes", type=int, default=200, help="fresh-eval budget per our-move")
    ap.add_argument("--opening-plies", type=int, default=4)
    ap.add_argument("--max-plies", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="m5b")
    ap.add_argument("--save-pgn", default="artifacts/experiments/m5_probe.pgn")
    args = ap.parse_args()
    dev = resolve_device("auto"); rng = np.random.default_rng(args.seed); t0 = time.time()

    rf = ReachabilityField(device=str(dev))
    rk = SubgoalRanker(args.field, args.reach, args.table, device=str(dev))
    cg = CommittorGreedy(args.ckpt, dev) if args.ckpt else None
    opp_policy = None
    if args.opp_model:
        from maia2 import model as maia_model, inference as m2_inf
        m2 = maia_model.from_pretrained(type="rapid", device=str(dev))
        opp_policy = make_maia2_policy(m2, m2_inf, args.maia_elo, int(args.our_elo))
    print(f"  leaf: {args.leaf} | order: {args.order}"
          f" | opp nodes: {'maia2 priors' if opp_policy else 'adversarial softmax'}",
          flush=True)

    reach = RegionReach(rk, args.our_elo, float(args.maia_elo))
    atlas = RegionAtlas(rk, args.our_elo, float(args.maia_elo))
    planner = PLANNERS[args.planner](reach, atlas)
    planner.prepare()
    print(f"  {planner.graph_line()}", flush=True)
    navigator = MCTSNavigator(reach, rf, atlas, cg=cg, opp_policy=opp_policy,
                              leaf=args.leaf, order=args.order, nodes=args.nodes)
    maia = chess.engine.SimpleEngine.popen_uci(
        ["lc0", f"--weights=data/engines/maia/maia-{args.maia_elo}.pb.gz",
         "--backend=eigen"])
    run_games(rf, planner, navigator, maia, args, rng, t0)
    maia.quit()


if __name__ == "__main__":
    main()
