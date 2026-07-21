#!/usr/bin/env python
"""experiments/planner_search_demo.py -- exercise the pluggable low-level searches
(weighted_astar / minimax_astar / mcts) on reaching a subgoal region, using iqe_geom as
the heuristic. Validates the interface + shows optimistic vs adversarial behavior.

Start = a mid-DTM nucleus position; subgoal = the mate region (lowest-DTM B-embeddings).
Reports for each search: reached?, plies, nodes, and the field distance-to-mate at the
start vs the end of the returned path (did it make real progress?)."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, torch, chess
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.data.encode import board_from_packed
from catspace.nn.fb import load_ckpt, pick_device
from catspace.planner.search import Budget, get_search, SEARCH_REGISTRY
from catspace.planner.field_heuristic import make_field_goal

dev = pick_device("auto")
fb, _ = load_ckpt(Path("data/derived/sep/iqe_geom.pt"), dev); fb.eval()
nz = np.load("data/derived/lichess_nearmate.npz"); won = np.flatnonzero(nz["dtm"] > 0)
dtm = nz["dtm"].astype(np.float32)
rng = np.random.default_rng(0)

# start = a moderate-DTM won position, White-to-move
def matkey(i):
    return "".join(sorted(p.symbol() for p in board_from_packed(nz["packed"][i], nz["meta"][i]).piece_map().values()))
cand = won[np.abs(dtm[won] - 16) < 2]
start_i = cand[0]
start = board_from_packed(nz["packed"][start_i], nz["meta"][start_i])
smat = matkey(start_i)

# subgoal = a NEARBY within-material region (same material, ~6 plies closer to mate) --
# the regime where the field distance is reliable (+0.454 per-material). Pick same-material
# lower-DTM positions and keep those the field puts ~4-8 plies from the start.
same = won[np.array([matkey(i) == smat for i in won])]
lower = same[(dtm[same] < dtm[start_i] - 2) & (dtm[same] > dtm[start_i] - 10)]
from catspace.planner.field_heuristic import _planes
from catspace.nn.features import omega_ids
_om = omega_ids(np.array([1800]), np.array([1800]), np.array([300.0]))[0]
with torch.no_grad():
    Fs = fb.embed_F(torch.from_numpy(_planes([start])).to(dev),
                    torch.from_numpy(_om[None]).to(dev))
    Bl = fb.embed_B(torch.from_numpy(_planes([board_from_packed(nz["packed"][i], nz["meta"][i]) for i in lower])).to(dev))
    dstart = fb.distance_matrix(Fs, Bl)[0].cpu().numpy()
band = lower[(dstart > 3) & (dstart < 9)][:8]
if len(band) == 0:
    band = lower[np.argsort(dstart)[3:11]]
subgoal_boards = [board_from_packed(nz["packed"][i], nz["meta"][i]) for i in band]
goal = make_field_goal(fb, subgoal_boards, device=dev, reach_thresh=1.5, label=f"{smat}-nearer")
h0 = goal.h1(start)
print(f"START  DTM(tablebase)={int(dtm[start_i])}  field d->mate={h0:.2f}  {'W' if start.turn else 'B'}-to-move")
print(f"       fen: {start.fen()}\n")

configs = {
    "weighted_astar": dict(w=1.5),
    "minimax_astar": dict(max_depth=6),
    "mcts": dict(iterations=600, rollout_depth=8),
}
budget = Budget(max_nodes=1500, max_plies=20)
for name in ["weighted_astar", "minimax_astar", "mcts"]:
    s = get_search(name, **configs[name])
    res = s.search(start, goal, budget)
    end = chess.Board(res.trajectory[-1])
    h_end = goal.h1(end)
    moves = " ".join(m.uci() for m in res.path[:8]) + (" ..." if len(res.path) > 8 else "")
    print(f"[{name:14s}] reached={str(res.reached):5s} plies={res.cost if res.reached else '-':>4} "
          f"nodes={res.nodes:5d}  d->mate {h0:.2f} -> {h_end:.2f}  progress={h0-h_end:+.2f}")
    print(f"                 path: {moves}")
print(f"\nregistry (plug-and-play): {list(SEARCH_REGISTRY)}")
print("VERDICT PLANNER_SEARCH interface OK; 3 searches ran on one Goal/Budget")
