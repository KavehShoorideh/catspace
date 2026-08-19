#!/usr/bin/env python
"""gen_move_graphs.py -- precompute square+move heterogeneous graphs for the GNN move scorer.

The GNN alternative (Kaveh 2026-08-19, from the sister design chat) exists to kill one measured
cost: scoring siblings through the trunk encodes EVERY child board separately -- 328 trunk passes
per step in cef6, 0.66 s/step, 3.7 h per run. Move-nodes score all legal moves in ONE forward
pass. This file precomputes the graphs once so train-time graph construction costs nothing.

PER PARENT (from the existing sibling-cache parent rows, so the split and the played-move
identity match every previous run):
    sq_int   (64, 8)  uint8   piece id 0-12 | attackers_w | attackers_b | pawn_att_w |
                              pawn_att_b | king_ring_w | king_ring_b | defended
    glob     (6,)     uint8   [stm, K, Q, k, q, ep_file+1]
    mv_int   (MV, 10) uint8   piece type 1-6 | from_sq | to_sq | is_capture |
                              captured_val 0-9 | promo 0-4 | gives_check | n_new_att |
                              is_castle | truncated
    mut      (MV, 24) uint8   squares NEWLY attacked by the moved piece after the move
                              (post-move attack set minus the same piece's pre-move set;
                              castling rook's attacks ignored -- stated limitation), 255 pad
    att_e    (224, 2) uint8   square->square attack edges (attacker sq, attacked sq), 255 pad
    n_mv, played_idx, n_att   uint8/uint16

    .venv/bin/python -m ...gen_move_graphs --save
"""
from __future__ import annotations

import argparse
import time

import numpy as np

MV_CAP = 48
MUT_CAP = 24
ATT_CAP = 224
VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sib-k", type=int, default=40, help="which sibling cache to take parents from")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    import chess
    from catspace.research.tools.training_infra.concepts_reference import TOK_PIECES, board_from_row
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    tr = T.build(n_human=0, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    sb = np.load(paths.derived(f"sibling_contrast_{args.sib_k}_{n}.npz"))
    p_row = sb["p_row"]
    N = len(p_row)

    SQ = np.zeros((N, 64, 8), np.uint8)
    GL = np.zeros((N, 6), np.uint8)
    MVI = np.zeros((N, MV_CAP, 10), np.uint8)
    MUT = np.full((N, MV_CAP, MUT_CAP), 255, np.uint8)
    ATT = np.full((N, ATT_CAP, 2), 255, np.uint8)
    NMV = np.zeros(N, np.uint8); PLY = np.zeros(N, np.uint8); NATT = np.zeros(N, np.uint16)
    trunc_mut = trunc_att = trunc_mv = 0

    t0 = time.time()
    for i, r in enumerate(p_row):
        b = board_from_row(tr.tok[r], tr.glob[r])
        GL[i] = tr.glob[r]
        kw, kb = b.king(chess.WHITE), b.king(chess.BLACK)
        ring_w = chess.SquareSet(chess.BB_KING_ATTACKS[kw]) if kw is not None else chess.SquareSet()
        ring_b = chess.SquareSet(chess.BB_KING_ATTACKS[kb]) if kb is not None else chess.SquareSet()
        ne = 0
        for sq in range(64):
            p = b.piece_at(sq)
            aw = b.attackers(chess.WHITE, sq); ab = b.attackers(chess.BLACK, sq)
            SQ[i, sq, 0] = 0 if p is None else TOK_PIECES.index((p.piece_type, p.color)) + 1
            SQ[i, sq, 1] = min(len(aw), 7); SQ[i, sq, 2] = min(len(ab), 7)
            SQ[i, sq, 3] = any(b.piece_type_at(a) == chess.PAWN for a in aw)
            SQ[i, sq, 4] = any(b.piece_type_at(a) == chess.PAWN for a in ab)
            SQ[i, sq, 5] = sq in ring_w; SQ[i, sq, 6] = sq in ring_b
            if p is not None:
                own = aw if p.color else ab
                SQ[i, sq, 7] = min(len(own), 7)
                for a in own:
                    if ne < ATT_CAP:
                        ATT[i, ne] = (a, sq); ne += 1
                    else:
                        trunc_att += 1
            # attack edges: every attacker -> this square (both colours), occupied or not
            for a in (aw | ab):
                if ne < ATT_CAP:
                    ATT[i, ne] = (a, sq); ne += 1
                else:
                    trunc_att += 1
        NATT[i] = ne

        nxt = tr.tok[r + 1]
        mv = list(b.legal_moves)
        if len(mv) > MV_CAP:
            trunc_mv += 1
            mv = mv[:MV_CAP]
        played = 0
        for j, m in enumerate(mv):
            pc = b.piece_at(m.from_square)
            cap = b.piece_at(m.to_square)
            is_ep = b.is_en_passant(m)
            pre = set(b.attacks(m.from_square))
            gc = b.gives_check(m)
            castle = b.is_castling(m)
            b.push(m)
            post = set(b.attacks(m.to_square))
            t = np.zeros(64, np.uint8)
            for q, pp in b.piece_map().items():
                t[q] = TOK_PIECES.index((pp.piece_type, pp.color)) + 1
            if np.array_equal(t, nxt):
                played = j
            b.pop()
            new = sorted(post - pre)
            if len(new) > MUT_CAP:
                trunc_mut += 1
                new = new[:MUT_CAP]
            MUT[i, j, :len(new)] = new
            MVI[i, j] = (pc.piece_type, m.from_square, m.to_square,
                         int(cap is not None or is_ep),
                         VAL[cap.piece_type] if cap else (1 if is_ep else 0),
                         {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
                          chess.QUEEN: 4}.get(m.promotion, 0),
                         int(gc), len(new), int(castle), 0)
        NMV[i] = len(mv); PLY[i] = played
        if (i + 1) % 50000 == 0:
            el = time.time() - t0
            print(f"  {i+1:,}/{N:,}  {(i+1)/el:,.0f} pos/s  eta {(N-i-1)/((i+1)/el)/60:.1f}m",
                  flush=True)

    el = time.time() - t0
    print(f"\n  {N:,} graphs in {el/60:.1f}m | truncated: moves {trunc_mv} "
          f"({100*trunc_mv/N:.2f}%), mutation sets {trunc_mut}, attack edges {trunc_att}")
    if args.save:
        p = paths.derived(f"move_graphs_{N}_{n}.npz")
        np.savez_compressed(p, sq=SQ, gl=GL, mv=MVI, mut=MUT, att=ATT,
                            n_mv=NMV, played=PLY, n_att=NATT, p_row=p_row)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
