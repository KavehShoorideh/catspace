#!/usr/bin/env python
"""experiments/mate_with_search.py -- S2b (METASTABILITY_PLAN): does FIELD + shallow SEARCH
convert the endgame where greedy (0.4%) couldn't? Policy = planner, not greedy field (Kaveh's
design). Minimax to depth D with the trained mate-field as leaf value (checkmate=+inf,
draw/loss=-inf so search avoids stalemate and seeks mate); White plays the search move, Black
defends TABLEBASE-OPTIMALLY (hardest). Mate-rate by depth -- if it climbs with depth, the
field is a usable VALUE and the planner supplies the POLICY.

Leaf evals are batched (collect the frontier once, one field forward, then minimax backup).
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

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.fb import pick_device
from catspace.tb import TB, DEFAULT_SYZYGY, rollout_dtm, tb_best_move
from experiments.arch_bakeoff import tokens
from experiments.gen_dtm_data import random_class_start
from experiments.train_mate_field import MateField
from experiments.value_fixed_point import white_pov_value

MATE_V, DRAW_V = 1e9, -1e6


def load(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False); c = ck["cfg"]
    net = MateField(c["d"], c["d_bb"], c["blocks"], c["iqe_components"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval(); return net, ck


@torch.no_grad()
def eval_fens(fens, net, dev):
    boards = [chess.Board(f) for f in fens]
    pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
    ids, stm = tokens(pk, mt)
    d = net.d_to_mate(torch.from_numpy(ids.astype(np.int64)).to(dev),
                      torch.from_numpy(stm.astype(np.int64)).to(dev)).cpu().numpy()
    return {f: -float(x) for f, x in zip(fens, d)}          # value = -distance (want small d)


def search_move(board, depth, net, dev):
    leaves = []
    def collect(node, d):
        if node.is_checkmate() or node.is_game_over(claim_draw=True):
            return
        if d == 0:
            leaves.append(node.fen()); return
        for m in node.legal_moves:
            node.push(m); collect(node, d - 1); node.pop()
    collect(board, depth)
    memo = eval_fens(list(set(leaves)), net, dev) if leaves else {}

    def mm(node, d):
        if node.is_checkmate():
            return MATE_V if node.turn == chess.BLACK else -MATE_V   # black mated => White wins
        if node.is_game_over(claim_draw=True):
            return DRAW_V                                    # draw = threw the win
        if d == 0:
            return memo[node.fen()]
        vals = []
        for m in node.legal_moves:
            node.push(m); vals.append(mm(node, d - 1)); node.pop()
        return max(vals) if node.turn == chess.WHITE else min(vals)

    best, bv = None, -1e18
    for m in board.legal_moves:
        board.push(m); v = mm(board, depth - 1); board.pop()
        if v > bv:
            bv, best = v, m
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default="artifacts/experiments/mate_field_v1.pt")
    ap.add_argument("--depths", type=int, nargs="*", default=[1, 3, 5])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--classes", nargs="*", default=["KQvK", "KRvK", "KRRvK", "KBNvK"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    net, ck = load(args.ckpt, dev)
    tb = TB(str(DEFAULT_SYZYGY), cache_db=None)
    print(f"[mate+search] {args.ckpt} greedy-mate was {ck['metrics'].get('mate_rate')}%", flush=True)

    # fixed set of start positions (same across depths)
    starts = []
    while len(starts) < args.n:
        cls = args.classes[rng.integers(0, len(args.classes))]
        b = random_class_start(rng, cls)
        if b is None or b.turn != chess.WHITE or white_pov_value(b, tb) != 1.0:
            continue
        opt = rollout_dtm(b, tb)
        if opt and opt >= 1:
            starts.append((b.fen(), opt))

    for depth in args.depths:
        mated = 0
        for fen, opt in starts:
            b = chess.Board(fen); cap = 3 * opt + 6; ok = False
            while len(b.move_stack) < cap:
                if b.is_checkmate(): ok = True; break
                if b.is_game_over(claim_draw=True): break
                if b.turn == chess.WHITE:
                    m = search_move(b, depth, net, dev)
                    if m is None: break
                    b.push(m)
                else:
                    m = tb_best_move(b, tb, set())
                    if m is None: break
                    b.push(m)
            mated += int(ok)
        print(f"  depth {depth}: MATE-RATE {100*mated/len(starts):.1f}% ({mated}/{len(starts)}) "
              f"[{time.time()-t0:.0f}s]", flush=True)
    tb.close()
    print("DONE mate_with_search", flush=True)


if __name__ == "__main__":
    main()
