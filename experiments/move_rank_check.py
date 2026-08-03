#!/usr/bin/env python
"""experiments/move_rank_check.py -- HOW BAD a move-prior is the field? (Kaveh 2026-07-21: the field IS the
MCTS prior -- softmax over child field-values -- and it underperforms a uniform prior, so its move-ranking
must be worse than chance.) For KRRvKBP positions, rank the legal moves by the field score of each child and
find where the TABLEBASE-OPTIMAL move lands.

  percentile of tb-best move in the field's ranking:  1.0 = field puts it on top, 0.5 = no better than a
  UNIFORM prior (random), < 0.5 = the field actively ranks the best move BELOW average (a harmful prior).
Two prior formulations: distance-to-SUBGOAL (what the field-MCTS used) and distance-to-MATE-region (direct).
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

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import load_ckpt, pick_device
from experiments.conversion_field_mcts import FieldMCTS
from experiments.value_fixed_point import TB, tb_best_move


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fields", nargs="+", default=["treat_5k=data/derived/sep/xfer_treat.pt",
                                                     "nucleus=data/derived/sep/iqe_nucleus_gn.pt",
                                                     "treat_20k=data/derived/sep/xfer_treat_20k.pt"])
    ap.add_argument("--dtm-npz", default="data/derived/dtm_endgame.npz")
    ap.add_argument("--fixed-set", default="artifacts/experiments/krrkbp_test_n200.json")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--bank", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    t0 = time.time(); dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    tb = TB("data/syzygy")
    dz = np.load(args.dtm_npz); idx = rng.permutation(len(dz["packed"]))[:args.bank]
    fens = json.loads(Path(args.fixed_set).read_text())["fens"][:args.n]

    def pctile(scores, best_i):
        order = np.argsort(-scores)                          # field's move ranking, best-first
        rank = int(np.flatnonzero(order == best_i)[0])       # 0 = field's top pick
        n = len(scores)
        return 1.0 - rank / (n - 1) if n > 1 else 1.0        # 1.0 = top, 0.5 ~ uniform

    print("VERDICT MOVE_RANK (percentile of tablebase-optimal move in the field's prior ordering; "
          "0.5=uniform, <0.5=harmful)")
    print(f"  {'field':12s} | {'subgoal-prior':>14s} | {'mate-region-prior':>17s}  (mean over positions, top-1 hit)")
    for spec in args.fields:
        label, ck = spec.split("=", 1)
        fb, _ = load_ckpt(Path(ck), dev); fb.eval()
        pl = FieldMCTS(fb, dev, dz["packed"][idx], dz["meta"][idx], 400, 40, 1.0, 4)
        sub_pc, mate_pc, sub_top1, mate_top1 = [], [], [], []
        for f in fens:
            b = chess.Board(f)
            if b.is_game_over() or len(list(b.legal_moves)) < 2:
                continue
            best = tb_best_move(b, tb)
            if best is None:
                continue
            moves = list(b.legal_moves); bi = moves.index(best)
            kids = [b.copy(stack=False) for _ in moves]
            for c, m in zip(kids, moves):
                c.push(m)
            f_child = pl._embF(np.stack([__import__("catspace.research.tools.chess_specific.chessdata.encode", fromlist=["encode_packed"]).encode_packed(c) for c in kids]),
                               np.stack([__import__("catspace.research.tools.chess_specific.chessdata.encode", fromlist=["encode_meta"]).encode_meta(c) for c in kids]))
            pl._select(b)
            with torch.no_grad():
                s_sub = -fb.distance_matrix(f_child, pl.subgoal_B).min(1).values.cpu().numpy()
                s_mate = -fb.distance_matrix(f_child, pl.mate_B).min(1).values.cpu().numpy()
            sub_pc.append(pctile(s_sub, bi)); mate_pc.append(pctile(s_mate, bi))
            sub_top1.append(int(np.argmax(s_sub) == bi)); mate_top1.append(int(np.argmax(s_mate) == bi))
        print(f"  {label:12s} | {np.mean(sub_pc):>8.2f} ({np.mean(sub_top1):>3.0%}) | "
              f"{np.mean(mate_pc):>10.2f} ({np.mean(mate_top1):>3.0%})   [{len(sub_pc)} positions]")
    tb.close()
    print(f"[done] {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
