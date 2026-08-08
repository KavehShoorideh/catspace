#!/usr/bin/env python
"""concept_ply_matrix.py -- concept row x ply column, floats = extent of concept at that ply
(Kaveh 2026-08-08). The measured layer of concept dynamics: computable predicates on boards, no
probes, no model -- ground truth for the concept-forecast head that will replace arm A's Gaussian
region readout. Outputs: corpus-average heatmap + one example game's matrix.
"""
from __future__ import annotations

import argparse

import chess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                            # noqa: E402
import numpy as np                                                         # noqa: E402

from catspace.io import paths                                              # noqa: E402
from catspace.research.components.encoder.approaches.reach_probability.experiments.eval_dtz_gate import (  # noqa: E402
    row_to_board)
from catspace.research.components.encoder.approaches.reach_probability.src import trajectories as T  # noqa: E402

PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def concepts_of(b: chess.Board) -> dict:
    """~18 computable concepts, each a float in [0, 1]-ish (extent, mover POV where signed)."""
    me, opp = b.turn, not b.turn
    my = {pt: len(b.pieces(pt, me)) for pt in PIECE_VAL}
    th = {pt: len(b.pieces(pt, opp)) for pt in PIECE_VAL}
    mat_me = sum(PIECE_VAL[p] * n for p, n in my.items())
    mat_th = sum(PIECE_VAL[p] * n for p, n in th.items())

    def passed(color):
        n = 0
        for sq in b.pieces(chess.PAWN, color):
            f, r = chess.square_file(sq), chess.square_rank(sq)
            ahead = range(r + 1, 8) if color else range(0, r)
            if not any(chess.square(ff, rr) in b.pieces(chess.PAWN, not color)
                       for ff in (f - 1, f, f + 1) if 0 <= ff <= 7 for rr in ahead):
                n += 1
        return n

    def open_files():
        pf = {chess.square_file(s) for s in b.pieces(chess.PAWN, True)} | \
             {chess.square_file(s) for s in b.pieces(chess.PAWN, False)}
        return 8 - len(pf)

    def king_ring_pressure(color):
        k = b.king(color)
        if k is None:
            return 0
        ring = chess.SquareSet(chess.BB_KING_ATTACKS[k])
        return sum(1 for sq in ring if b.is_attacked_by(not color, sq))

    return {
        "material advantage": np.clip((mat_me - mat_th) / 9 + 0.5, 0, 1),
        "queens on": (my[chess.QUEEN] + th[chess.QUEEN]) / 2,
        "heavy pieces on": (my[chess.ROOK] + th[chess.ROOK] + my[chess.QUEEN] + th[chess.QUEEN]) / 6,
        "minor pieces on": (my[chess.KNIGHT] + my[chess.BISHOP] + th[chess.KNIGHT] + th[chess.BISHOP]) / 8,
        "pawns on": (my[chess.PAWN] + th[chess.PAWN]) / 16,
        "bishop pair (mine)": float(my[chess.BISHOP] >= 2),
        "passed pawns (mine)": min(passed(me), 3) / 3,
        "passed pawns (theirs)": min(passed(opp), 3) / 3,
        "open files": open_files() / 8,
        "my king attacked ring": king_ring_pressure(me) / 8,
        "their king attacked ring": king_ring_pressure(opp) / 8,
        "in check": float(b.is_check()),
        "castling rights (mine)": (b.has_kingside_castling_rights(me)
                                   + b.has_queenside_castling_rights(me)) / 2,
        "castling rights (theirs)": (b.has_kingside_castling_rights(opp)
                                     + b.has_queenside_castling_rights(opp)) / 2,
        "capture available": float(any(b.is_capture(m) for m in b.legal_moves)),
        "mobility": min(b.legal_moves.count(), 40) / 40,
        "tb range (<=5 pieces)": float(len(b.piece_map()) <= 5),
        "endgame (non-pawn mat <=10)": float(mat_me + mat_th
                                             - (my[chess.PAWN] + th[chess.PAWN]) <= 10),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=4000)
    ap.add_argument("--n-piecedown", type=int, default=4000)
    ap.add_argument("--n-avg", type=int, default=300, help="games averaged for the corpus heatmap")
    ap.add_argument("--max-ply", type=int, default=120)
    ap.add_argument("--example-game", type=int, default=7)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tr = T.build(n_human=0, n_sf=args.games, seed=0, max_plies=400,
                 n_piecedown=args.n_piecedown, verbose=False)
    rng = np.random.default_rng(0)
    games = rng.choice(len(tr), args.n_avg, replace=False)
    names = list(concepts_of(chess.Board()).keys())
    C = len(names)
    acc = np.zeros((C, args.max_ply)); cnt = np.zeros(args.max_ply)
    ex = None
    for gi_i, gi in enumerate(games):
        st, ln = int(tr.start[gi]), int(tr.length[gi])
        M = np.full((C, args.max_ply), np.nan)
        for p in range(min(ln, args.max_ply)):
            b = row_to_board(tr.tok[st + p], tr.glob[st + p])
            if not b.is_valid():
                continue
            v = concepts_of(b)
            M[:, p] = [v[n] for n in names]
            acc[:, p] += M[:, p]; cnt[p] += 1
        if gi_i == args.example_game:
            ex = (int(gi), M.copy())
    avg = acc / np.maximum(cnt, 1)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for ax, (mat, title) in zip(axes, ((avg, f"corpus average ({args.n_avg} SF games)"),
                                       (ex[1], f"one game (id {ex[0]})"))):
        im = ax.imshow(mat, aspect="auto", cmap="magma", vmin=0, vmax=1,
                       interpolation="nearest")
        ax.set_yticks(range(C), names, fontsize=8)
        ax.set_xlabel("ply")
        ax.set_title(title)
    fig.colorbar(im, ax=axes, fraction=0.02, label="concept extent [0,1]")
    fig.suptitle("Concept x ply activation matrix -- the measured layer of concept dynamics\n"
                 "(these matrices are the training targets for the concept-forecast head)",
                 fontsize=12)
    out = args.out or paths.figure("concept_ply_matrix.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"[concepts] {C} concepts x {args.max_ply} plies, {args.n_avg} games -> {out}")


if __name__ == "__main__":
    main()
