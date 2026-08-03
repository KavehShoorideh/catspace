#!/usr/bin/env python
"""catspace/research/tools/stats_eval/ab_convert.py -- deep alpha-beta conversion (S2e). Simple endgames mate in a
bounded number of plies, so a DEEP search finds the mate directly even with a coarse field
value (the field already separates won d~20 from draw d~468). This is the engine-style route:
deep search + value, and it's the same planner the midgame will use. Negamax alpha-beta with
per-node batched child eval for move ordering + depth-1 leaves, and a FEN value memo. White =
AB search on the field; Black = tablebase-optimal defense. Conversion rate by depth.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch


from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, DEFAULT_SYZYGY, rollout_dtm, tb_best_move
from catspace.research.components.encoder.approaches.reachability_field.experiments.arch_bakeoff import tokens
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import white_pov_value
from catspace.io import paths

BIG = 1e7


def load_any(ckpt, dev):
    ck = torch.load(ckpt, map_location=dev, weights_only=False); c = ck["cfg"]
    if ck.get("model") == "FullField":
        from catspace.research.components.encoder.approaches.reachability_field.experiments.train_field_full import FullField
        net = FullField(c["d"], c["d_bb"], c["blocks"], c["iqe_components"]).to(dev)
    else:
        from catspace.research.components.encoder.approaches.reachability_field.experiments.train_mate_field import MateField
        net = MateField(c["d"], c["d_bb"], c["blocks"], c["iqe_components"]).to(dev)
    net.load_state_dict(ck["state_dict"]); net.eval(); return net, ck


class Searcher:
    def __init__(self, net, dev):
        self.net = net; self.dev = dev; self.memo = {}
        self.dfn = getattr(net, "d_mate", None) or getattr(net, "d_to_mate")

    @torch.no_grad()
    def vbatch(self, boards):
        pk = np.stack([encode_packed(b) for b in boards]); mt = np.stack([encode_meta(b) for b in boards])
        ids, stm = tokens(pk, mt)
        d = self.dfn(torch.from_numpy(ids.astype(np.int64)).to(self.dev),
                     torch.from_numpy(stm.astype(np.int64)).to(self.dev)).cpu().numpy()
        return -d                                            # White value = -distance

    def leaf_terminal(self, b):
        if b.is_checkmate():
            return BIG if b.turn == chess.BLACK else -BIG
        if b.is_game_over(claim_draw=True):
            return -BIG                                      # draw = threw the win
        return None

    def negamax(self, board, depth, alpha, beta):
        t = self.leaf_terminal(board)
        if t is not None:
            return t
        moves = list(board.legal_moves)
        kids = []
        for m in moves:
            board.push(m); kids.append((m, board.copy(stack=False))); board.pop()
        # terminal children shortcut + batched value for ordering / depth-1 leaves
        tv = [self.leaf_terminal(c) for _, c in kids]
        need = [i for i, v in enumerate(tv) if v is None]
        if need:
            fv = self.vbatch([kids[i][1] for i in need])
            for k, i in enumerate(need):
                tv[i] = float(fv[k])
        order = sorted(range(len(kids)), key=lambda i: tv[i], reverse=board.turn == chess.WHITE)
        if depth <= 1:
            return max(tv) if board.turn == chess.WHITE else min(tv)
        if board.turn == chess.WHITE:
            v = -BIG - 1
            for i in order:
                v = max(v, self.negamax(kids[i][1], depth - 1, alpha, beta))
                alpha = max(alpha, v)
                if alpha >= beta:
                    break
            return v
        else:
            v = BIG + 1
            for i in order:
                v = min(v, self.negamax(kids[i][1], depth - 1, alpha, beta))
                beta = min(beta, v)
                if alpha >= beta:
                    break
            return v

    def best_move(self, board, depth):
        moves = list(board.legal_moves); best, bm = -1e18, None
        for m in moves:
            board.push(m); v = self.negamax(board, depth - 1, -1e18, 1e18); board.pop()
            if v > best:
                best, bm = v, m
        return bm


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.experiment("mate_field_v1.pt"))
    ap.add_argument("--depths", type=int, nargs="*", default=[4, 6, 8])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--classes", nargs="*", default=["KQvK", "KRvK"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    net, ck = load_any(args.ckpt, dev)
    tb = TB(str(DEFAULT_SYZYGY), cache_db=None)
    print(f"[ab-convert] {args.ckpt}  eff_rank {ck['metrics'].get('eff_rank')}", flush=True)

    starts = []
    from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.gen_dtm_data import random_class_start
    while len(starts) < args.n:
        cls = args.classes[rng.integers(0, len(args.classes))]
        b = random_class_start(rng, cls)
        if b is None or b.turn != chess.WHITE or white_pov_value(b, tb) != 1.0:
            continue
        opt = rollout_dtm(b, tb)
        if opt and opt >= 1:
            starts.append((b.fen(), opt))

    for depth in args.depths:
        se = Searcher(net, dev); mated = drew = 0
        for fen, opt in starts:
            b = chess.Board(fen); cap = 3 * opt + 10; ok = False
            while len(b.move_stack) < cap:
                if b.is_checkmate(): ok = True; break
                if b.is_game_over(claim_draw=True): break
                if b.turn == chess.WHITE:
                    m = se.best_move(b, depth)
                    if m is None: break
                    b.push(m)
                else:
                    m = tb_best_move(b, tb, set())
                    if m is None: break
                    b.push(m)
            if ok: mated += 1
            elif b.is_game_over(claim_draw=True) and not b.is_checkmate(): drew += 1
        print(f"  depth {depth}: CONVERT {100*mated/len(starts):.1f}% ({mated}/{len(starts)}) "
              f"drew {drew} [{time.time()-t0:.0f}s]", flush=True)
    tb.close(); print("DONE ab_convert", flush=True)


if __name__ == "__main__":
    main()
