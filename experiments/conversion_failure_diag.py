#!/usr/bin/env python
"""Where does the committor conversion fail? (Kaveh 2026-07-19: grounds what a
proper subgoal must target.) Runs the incumbent committor from KRRvKBP starts vs
a tablebase defender; for the games it FAILS to mate within the budget, records
the final material configuration -- did it fail to SIMPLIFY (still has the pawn /
full material) or fail to MATE a simplified endgame (KRRvK / KRvK)?"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import chess, numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from catspace.nn.eval_head import EvalHead
from catspace.nn.fb import load_ckpt
from catspace.nn.policy_fb import make_search_policy
from experiments.value_fixed_point import TB, tb_best_move

dev = "cpu"
fb, pay = load_ckpt(Path("data/derived/sep/cert_base_full.pt"), dev); fb.eval()
hp = torch.load("data/derived/sep/cert_base_full_phead.pt", map_location=dev, weights_only=False)
ph = EvalHead(d_in=hp["d_in"]); ph.load_state_dict(hp["state"]); ph.eval()


class Committor(torch.nn.Module):
    def forward(self, f):
        p = torch.softmax(ph(f), 1)
        return (-torch.log(p[:, 0].clamp_min(1e-6))).unsqueeze(-1)


pol = make_search_policy("mcts", fb, pay["zgoals"]["MATE_W"], max_nodes=400, device=dev,
                         committor_head=Committor(), mate_stop=True, pw_c=1.5, root_min_visits=10)


def matkey(b):
    w, bl = [], []
    for _, pc in b.piece_map().items():
        (w if pc.color == chess.WHITE else bl).append(pc.symbol().upper())
    return "".join(sorted(w)) + " v " + "".join(sorted(bl))


N = int(sys.argv[1]) if len(sys.argv) > 1 else 25
starts = json.loads(Path("artifacts/experiments/krrkbp_test_n200.json").read_text())["fens"][:N]
tb = TB("data/syzygy")
fails, mated = Counter(), 0
for i, fen in enumerate(starts):
    b = chess.Board(fen); seen = set()
    for ply in range(120):
        if b.is_game_over(claim_draw=True):
            break
        m = pol.move(b, np.random.default_rng([0, i])) if b.turn == chess.WHITE \
            else tb_best_move(b, tb, seen)
        if b.turn == chess.BLACK:
            seen.add(b.board_fen())
        if m is None:
            break
        b.push(m)
    out = b.outcome(claim_draw=True)
    if out and out.winner == chess.WHITE:
        mated += 1
    else:
        fails[matkey(b)] += 1
    print(f"  game {i}: {'MATE' if (out and out.winner==chess.WHITE) else 'FAIL @ '+matkey(b)}", flush=True)
tb.close()
print(f"VERDICT FAILURE_DIAG committor mated {mated}/{N} = {mated/N:.3f}")
for k, v in fails.most_common():
    print(f"   {v:2d}x FAIL final-material  {k}")
