#!/usr/bin/env python
"""eval_boundary_routing.py -- THE deployment gate for the hybrid oracle (Kaveh 2026-08-07):
"at the tablebase we'll just do lookup" -- so the field's real job is ROUTING into favourable
TB entries, and this measures exactly that handoff.

Held-out positions with SIX pieces (one ply outside lookup coverage): every capture-move child
has <=5 pieces and is EXACTLY scoreable by Syzygy. Among the TB-scoreable children spanning at
least two distinct outcomes, the field ranks by d(child -> LOSS pole | cond) ascending (child's
mover is the opponent). Reported:

  pairwise-choice  -- over all scoreable child pairs with different TB value, how often the
                      field prefers the TB-better entry (random = 0.5). The primary number.
  top1-entry       -- how often the field's best scoreable child is a TB-best entry.
"""
from __future__ import annotations

import argparse

import chess
import numpy as np
import torch

from catspace.research.components.encoder.approaches.reach_probability.experiments.build_reach_pairs import (
    split_by_game)
from catspace.research.components.encoder.approaches.reach_probability.experiments.eval_dtz_gate import (
    row_to_board)
from catspace.research.components.encoder.approaches.reach_probability.experiments.interpret_reach import (
    load_net)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T
from catspace.research.components.planner.approaches.endgame_groundtruth.src.tb import TB
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import tokenize


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n-pos", type=int, default=800)
    ap.add_argument("--cond-elo", type=float, default=None)
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    net, pay = load_net(args.ckpt, args.device)
    c = pay["cfg"]
    tr = T.build(n_human=c["games"] // 2, n_sf=c["games"] // 2, seed=c["traj_seed"],
                 max_plies=c["max_plies"], n_piecedown=c.get("n_piecedown", 0), verbose=False)
    split = split_by_game(np.arange(len(tr)), (0.70, 0.15), c["traj_seed"])
    test = np.flatnonzero(split == 2)
    game, pc = tr.game_of_row(), tr.piece_count()
    rows = np.flatnonzero(np.isin(game, test) & (pc == 6))
    rng = np.random.default_rng(0)
    rng.shuffle(rows)

    def random_boards(n):
        """Random LEGAL 6-piece boards: unlimited, unseen-by-construction -- game-sourced 6-piece
        positions with >=2 differing capture children are too rare for statistical power (11 in
        the 4k-game test split)."""
        out = []
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT, chess.PAWN]
        while len(out) < n:
            b = chess.Board(None)
            sqs = rng.choice(64, 6, replace=False)
            b.set_piece_at(int(sqs[0]), chess.Piece(chess.KING, True))
            b.set_piece_at(int(sqs[1]), chess.Piece(chess.KING, False))
            for k in range(2, 6):
                pt = pieces[int(rng.integers(0, len(pieces)))]
                sq = int(sqs[k])
                if pt == chess.PAWN and chess.square_rank(sq) in (0, 7):
                    pt = chess.QUEEN
                b.set_piece_at(sq, chess.Piece(pt, bool(rng.integers(0, 2))))
            b.turn = bool(rng.integers(0, 2))
            if b.is_valid() and not b.is_game_over():
                out.append(b)
        return out

    def enc(tok_t, glob_t):
        if args.cond_elo is not None and getattr(net, "dual", False):
            cval = (args.cond_elo - 1500.0) / 500.0
            cond = torch.full((len(tok_t), net.qhead.proj_delta.in_features
                               - net.qhead.proj_base.in_features), cval, device=args.device)
            return net.encode_dual(tok_t, glob_t, cond)[1]
        return net.encode_q(tok_t, glob_t)

    iqe = net.qhead.iqe if getattr(net, "dual", False) else net.iqe
    pn = c["pole_names"]
    pL = net.poles.poles.detach().float()[pn.index("LOSS")]

    tb = TB()
    pair_ok = pair_n = 0
    top1 = 0.0
    n_done = 0
    boards = [row_to_board(tr.tok[r], tr.glob[r]) for r in rows[:2000]]
    boards = [b for b in boards if b.is_valid()] + random_boards(args.n_pos * 6)
    for b in boards:
        if n_done >= args.n_pos:
            break
        toks, globs, vals = [], [], []
        for mv in b.legal_moves:
            if not b.is_capture(mv):
                continue                      # only capture children drop to <=5 = scoreable
            b.push(mv)
            try:
                w, dz = tb.wdl_dtz(b)         # child's mover = opponent; lower w = better for us
            except Exception:
                w = None
            if w is not None:
                tk, gl = tokenize(b)
                toks.append(tk); globs.append(gl); vals.append(w)
            b.pop()
        if len(toks) < 2 or len(set(vals)) < 2:
            continue
        with torch.no_grad():
            z = enc(torch.from_numpy(np.array(toks).astype(np.int64)).to(args.device),
                    torch.from_numpy(np.array(globs).astype(np.float32)).to(args.device))
            d = iqe(z, pL.expand(len(z), -1).to(args.device)).float().cpu().numpy()
        vals = np.array(vals, float)
        for a in range(len(vals)):
            for bb in range(a + 1, len(vals)):
                if vals[a] == vals[bb]:
                    continue
                better = a if vals[a] < vals[bb] else bb    # lower opponent-wdl = better for us
                pref = a if d[a] < d[bb] else bb
                pair_ok += better == pref
                pair_n += 1
        top1 += float(vals[int(np.argmin(d))] == vals.min())
        n_done += 1
    tb.close()

    print(f"[route] {n_done} held-out 6-piece positions with >=2 TB-scoreable children of "
          f"differing value (cond-elo {args.cond_elo})")
    print(f"[route] pairwise-choice: {pair_ok/max(pair_n,1):.3f}  (random 0.500, n {pair_n})")
    print(f"[route] top1-entry: {top1/max(n_done,1):.3f}")


if __name__ == "__main__":
    main()
