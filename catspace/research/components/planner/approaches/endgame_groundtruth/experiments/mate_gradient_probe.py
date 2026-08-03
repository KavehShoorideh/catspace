#!/usr/bin/env python
"""catspace/research/components/planner/approaches/endgame_groundtruth/experiments/mate_gradient_probe.py -- VISUALISE the (missing) mating gradient (Kaveh
2026-07-26). For a few mate-in-1 positions: print the start FEN + its field distance-to-mate,
then EVERY legal move with its resulting FEN, the TRUE DTM (tablebase), and the FIELD's
distance-to-mate of the resulting position -- sorted by field distance. If a gradient existed,
the mating move (true DTM 0) would have the smallest field distance.

Also addresses the 'two distances' point: distance-to-mate is directional. For a White-winning
position there are TWO mate regions -- White-DELIVERS-mate (Black checkmated; should be ~small)
and White-GETS-mated (White checkmated; should be ~infinite, it can't happen). We report both
(two separate landmark banks) to see whether the field distinguishes them.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import chess
import numpy as np
import torch


from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB, DEFAULT_SYZYGY, rollout_dtm, rollout_line
from catspace.research.components.planner.approaches.endgame_groundtruth.experiments.gen_dtm_data import random_class_start
from catspace.research.components.encoder.approaches.reachability_field.experiments.mate_from_field import embed, load_field
from catspace.research.components.planner.approaches.committor_value.experiments.value_fixed_point import white_pov_value
from catspace.io import paths


def build_bank(net, tb, rng, classes, want_white_wins, size, dev):
    """Bank of terminal checkmate positions. want_white_wins=True => Black-checkmated
    (White delivered mate); False => White-checkmated (Black delivered mate)."""
    boards = []
    tries = 0
    while len(boards) < size and tries < size * 400:
        tries += 1
        cls = classes[rng.integers(0, len(classes))]
        b = random_class_start(rng, cls)
        if b is None:
            continue
        wv = white_pov_value(b, tb)
        if want_white_wins and wv != 1.0:
            continue
        if (not want_white_wins) and wv != 0.0:
            continue
        line = rollout_line(b, tb, cap=200)
        if line and line[-1].is_checkmate():
            boards.append(line[-1])
    return embed(net, boards, dev) if boards else None, len(boards)


@torch.no_grad()
def d_to_bank(net, boards, bank_emb, dev):
    e = embed(net, boards, dev)
    return net.iqe.pairwise(e, bank_emb).min(dim=1).values.cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=paths.experiment("quasimetric_shared_v1.pt"))
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--bank", type=int, default=512)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    from catspace.research.components.encoder.approaches.jepa_tokenizer.src.fb import pick_device
    dev = pick_device(args.device); rng = np.random.default_rng(args.seed)
    net, ck = load_field(args.ckpt, dev)
    tb = TB(str(DEFAULT_SYZYGY), cache_db=None)
    print(f"ckpt {args.ckpt}  pair-order {ck['metrics']['pair_spearman']:.3f}  "
          f"mate-via-min {ck['metrics']['mate_min_spearman']:.3f}\n")

    win_classes = ["KQvK", "KRvK", "KRRvK", "KBBvK", "KBNvK"]
    lose_classes = ["KvKQ", "KvKR", "KvKRR", "KvKBB", "KvKBN"]
    win_bank, nw = build_bank(net, tb, rng, win_classes, True, args.bank, dev)
    lose_bank, nl = build_bank(net, tb, rng, lose_classes, False, args.bank, dev)
    print(f"banks: White-delivers-mate {nw} pos | White-gets-mated {nl} pos\n")

    shown = 0
    while shown < args.examples:
        cls = win_classes[rng.integers(0, len(win_classes))]
        b = random_class_start(rng, cls)
        if b is None or b.turn != chess.WHITE or white_pov_value(b, tb) != 1.0:
            continue
        if rollout_dtm(b, tb) != 1:                       # mate-in-1 (1 ply)
            continue
        shown += 1
        d_start_win = float(d_to_bank(net, [b], win_bank, dev)[0])
        d_start_lose = float(d_to_bank(net, [b], lose_bank, dev)[0])
        print(f"=== Example {shown}: {b.fen()}")
        print(f"    ({cls}, White to move, true DTM = 1)")
        print(f"    START field distance:  to-WIN-mate {d_start_win:7.3f}   "
              f"to-GET-mated {d_start_lose:7.3f}")
        moves = list(b.legal_moves)
        kids, rows = [], []
        for m in moves:
            b.push(m); kids.append(b.copy(stack=False))
            dtm = 0 if b.is_checkmate() else rollout_dtm(b, tb)
            san = b.san(m) if False else None; b.pop()
            rows.append([b.san(m), m.uci(), dtm])
        dw = d_to_bank(net, kids, win_bank, dev)
        for r, x in zip(rows, dw):
            r.append(float(x))
        order = np.argsort([r[3] for r in rows])          # sort by field distance (ascending)
        print(f"    {'move':7} {'uci':7} {'trueDTM':>8} {'field_d_to_mate':>16}")
        for oi in order:
            san, uci, dtm, fd = rows[oi]
            tag = "  <== MATE (true best)" if dtm == 0 else ""
            dtm_s = "mate" if dtm == 0 else (str(dtm) if dtm is not None else "draw")
            print(f"    {san:7} {uci:7} {dtm_s:>8} {fd:16.3f}{tag}")
        # did the field rank the mating move first?
        best_field = rows[int(order[0])]
        print(f"    -> field's top move: {best_field[0]} (trueDTM "
              f"{'mate' if best_field[2]==0 else best_field[2]})  "
              f"{'CORRECT' if best_field[2]==0 else 'WRONG (not the mate)'}\n")
    tb.close()


if __name__ == "__main__":
    main()
