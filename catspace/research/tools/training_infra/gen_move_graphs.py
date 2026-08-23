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
VAC_CAP = 24
DISC_CAP = 12
ATT_CAP = 224
VAL = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9, 6: 0}


def glob_of_board(b):
    """the corpus glob row [stm, K, Q, k, q, ep_file+1] for a LIVE board -- the inverse of
    concepts_reference.board_from_row's glob consumption, so live graphs match corpus graphs."""
    import chess
    ep = 0 if b.ep_square is None else chess.square_file(b.ep_square) + 1
    return np.array([int(b.turn),
                     int(b.has_kingside_castling_rights(chess.WHITE)),
                     int(b.has_queenside_castling_rights(chess.WHITE)),
                     int(b.has_kingside_castling_rights(chess.BLACK)),
                     int(b.has_queenside_castling_rights(chess.BLACK)), ep], np.uint8)


_SLOT_ORDER = {1: (8, 16), 2: (6, 8), 3: (4, 6), 4: (2, 4), 5: (1, 2), 6: (0, 1)}
# per side: K=0, Q=1, R=2-3, B=4-5, N=6-7, P=8-15 (by square order within type);
# promoted extras beyond standard counts fill vacant pawn slots. CANONICAL-BY-BOARD:
# identity is deterministic from the position alone (no history); twin pieces that
# cross in square order swap slots -- rare, documented, the v1 price of a
# board-only API (gcf5 piece concepts).


def piece_slots(b):
    """(64,) uint8: which of 32 piece slots occupies each square; 255 = empty."""
    import chess
    ps = np.full(64, 255, np.uint8)
    for color in (chess.WHITE, chess.BLACK):
        base = 0 if color else 16
        used = set()
        overflow = []
        for pt in (6, 5, 4, 3, 2, 1):
            lo, hi = _SLOT_ORDER[pt]
            sqs = sorted(b.pieces(pt, color))
            for i, sq in enumerate(sqs):
                slot = lo + i
                if slot < hi:
                    ps[sq] = base + slot
                    used.add(slot)
                else:
                    overflow.append(sq)
        free_pawn = [k for k in range(8, 16) if k not in used]
        for sq, slot in zip(overflow, free_pawn):
            ps[sq] = base + slot
    return ps


def parent_graph(b, nxt_tok=None):
    """all parent-side arrays for ONE position -- the single source of truth shared by this
    offline generator and the live policies (arena play must build the exact encoding training
    saw). `nxt_tok` (64,) enables played-move detection; live callers pass None (played=0,
    meaningless there). Returns (dict of arrays, dict of truncation counts)."""
    import chess
    from catspace.research.tools.training_infra.concepts_reference import TOK_PIECES
    sq = np.zeros((64, 8), np.uint8)
    mvi = np.zeros((MV_CAP, 10), np.uint8)
    mut = np.full((MV_CAP, MUT_CAP), 255, np.uint8)
    vac = np.full((MV_CAP, VAC_CAP), 255, np.uint8)
    dsc = np.full((MV_CAP, DISC_CAP), 255, np.uint8)
    att = np.full((ATT_CAP, 2), 255, np.uint8)
    pslot = piece_slots(b)
    tc = dict(mv=0, mut=0, vac=0, att=0)
    kw, kb = b.king(chess.WHITE), b.king(chess.BLACK)
    ring_w = chess.SquareSet(chess.BB_KING_ATTACKS[kw]) if kw is not None else chess.SquareSet()
    ring_b = chess.SquareSet(chess.BB_KING_ATTACKS[kb]) if kb is not None else chess.SquareSet()
    ne = 0
    for sqi in range(64):
        pc = b.piece_at(sqi)
        aw = b.attackers(chess.WHITE, sqi); ab = b.attackers(chess.BLACK, sqi)
        sq[sqi, 0] = 0 if pc is None else TOK_PIECES.index((pc.piece_type, pc.color)) + 1
        sq[sqi, 1] = min(len(aw), 7); sq[sqi, 2] = min(len(ab), 7)
        sq[sqi, 3] = any(b.piece_type_at(a) == chess.PAWN for a in aw)
        sq[sqi, 4] = any(b.piece_type_at(a) == chess.PAWN for a in ab)
        sq[sqi, 5] = sqi in ring_w; sq[sqi, 6] = sqi in ring_b
        if pc is not None:
            own = aw if pc.color else ab
            sq[sqi, 7] = min(len(own), 7)
            for a in own:
                if ne < ATT_CAP:
                    att[ne] = (a, sqi); ne += 1
                else:
                    tc["att"] += 1
        # attack edges: every attacker -> this square (both colours), occupied or not
        for a in (aw | ab):
            if ne < ATT_CAP:
                att[ne] = (a, sqi); ne += 1
            else:
                tc["att"] += 1

    mv = list(b.legal_moves)
    if len(mv) > MV_CAP:
        tc["mv"] += 1
        mv = mv[:MV_CAP]
    played = 0
    for j, m in enumerate(mv):
        pc = b.piece_at(m.from_square)
        cap = b.piece_at(m.to_square)
        is_ep = b.is_en_passant(m)
        pre = set(b.attacks(m.from_square))
        dying = set(b.attacks(m.to_square)) if cap is not None else set()
        gc = b.gives_check(m)
        castle = b.is_castling(m)
        b.push(m)
        post = set(b.attacks(m.to_square))
        if nxt_tok is not None:
            t = np.zeros(64, np.uint8)
            for q, pp in b.piece_map().items():
                t[q] = TOK_PIECES.index((pp.piece_type, pp.color)) + 1
            if np.array_equal(t, nxt_tok):
                played = j
        # discovered: squares attacked by pieces (either colour) whose line passes the
        # vacated square -- approximate as attacks-to targets gained via the empty from_sq
        disc = set()
        for col in (chess.WHITE, chess.BLACK):
            for a in b.attackers(col, m.from_square):
                disc |= set(b.attacks(a)) - {m.from_square}
        b.pop()
        new = sorted(post - pre)
        gone = sorted((pre - post) | dying)
        disc = sorted(disc - post - pre)[:DISC_CAP]
        if len(new) > MUT_CAP:
            tc["mut"] += 1
            new = new[:MUT_CAP]
        mut[j, :len(new)] = new
        if len(gone) > VAC_CAP:
            tc["vac"] += 1; gone = gone[:VAC_CAP]
        vac[j, :len(gone)] = gone
        dsc[j, :len(disc)] = disc
        mvi[j] = (pc.piece_type, m.from_square, m.to_square,
                  int(cap is not None or is_ep),
                  VAL[cap.piece_type] if cap else (1 if is_ep else 0),
                  {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3,
                   chess.QUEEN: 4}.get(m.promotion, 0),
                  int(gc), len(new), int(castle), 0)
    return (dict(sq=sq, pslot=pslot, mv=mvi, mut=mut, vac=vac, dsc=dsc, att=att,
                 n_mv=len(mv), played=played, n_att=ne), tc)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=20000)
    ap.add_argument("--n-human", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0,
                    help="sample N random non-terminal parent rows (for corpora without a "
                         "sibling cache, e.g. human games)")
    ap.add_argument("--n-piecedown", type=int, default=45906)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sib-k", type=int, default=40, help="which sibling cache to take parents from")
    ap.add_argument("--rows-npz", default="", help="take parent rows from this npz instead of "
                    "the sibling cache (e.g. the counterfactual file); rows are DEDUPLICATED "
                    "and terminal rows (no played child) dropped with a count")
    ap.add_argument("--rows-key", default="row")
    ap.add_argument("--tag", default="", help="artifact name tag: move_graphs_v2_<tag>_N_n.npz")
    ap.add_argument("--save", action="store_true")
    args = ap.parse_args()

    import chess
    from catspace.research.tools.training_infra.concepts_reference import TOK_PIECES, board_from_row
    from catspace.research.components.encoder.approaches.reach_probability.src import (
        trajectories as T)
    from catspace.io import paths

    tr = T.build(n_human=args.n_human, n_sf=args.games, seed=args.seed, cache=True,
                 max_plies=args.max_plies, n_piecedown=args.n_piecedown)
    n = len(tr.tok)
    gor = tr.game_of_row()
    g_end = np.zeros(n, np.int64)
    for gi in range(len(tr.start)):
        s0, ln = int(tr.start[gi]), int(tr.length[gi])
        g_end[s0:s0 + ln] = s0 + ln - 1
    if args.rows_npz:
        raw = np.load(args.rows_npz)[args.rows_key]
        p_row = np.unique(raw)
        term = p_row == g_end[p_row]
        if term.any():
            print(f"  dropping {int(term.sum())} terminal rows (no played child)")
            p_row = p_row[~term]
        print(f"  parents from {args.rows_npz}[{args.rows_key}]: {len(raw):,} rows -> "
              f"{len(p_row):,} unique non-terminal")
    elif args.sample:
        rng0 = np.random.default_rng(args.seed)
        cand = np.flatnonzero(np.arange(n) != g_end)      # non-terminal rows
        p_row = np.sort(rng0.choice(cand, min(args.sample, len(cand)), replace=False))
        print(f"  sampled {len(p_row):,} non-terminal parents")
    else:
        sb = np.load(paths.derived(f"sibling_contrast_{args.sib_k}_{n}.npz"))
        p_row = sb["p_row"]
    N = len(p_row)

    SQ = np.zeros((N, 64, 8), np.uint8)
    GL = np.zeros((N, 6), np.uint8)
    MVI = np.zeros((N, MV_CAP, 10), np.uint8)
    MUT = np.full((N, MV_CAP, MUT_CAP), 255, np.uint8)
    # THE DELETION HALF OF THE DERIVATIVE (Kaveh 2026-08-19: "some edges are added and some
    # edges are deleted -- there is a derivative in these edges"). v1 carried additions only.
    VAC = np.full((N, MV_CAP, VAC_CAP), 255, np.uint8)   # attacks the mover loses + the
                                                          # captured piece's dying attacks
    DSC = np.full((N, MV_CAP, DISC_CAP), 255, np.uint8)  # discovered: newly attacked through
                                                          # the vacated square, other pieces
    # slim CHILD block: the played child's square features + attack edges, for future-code
    # dynamics targets (no move nodes needed on the child side)
    CSQ = np.zeros((N, 64, 8), np.uint8); CGL = np.zeros((N, 6), np.uint8)
    CATT = np.full((N, ATT_CAP, 2), 255, np.uint8); CNATT = np.zeros(N, np.uint16)
    # t+8 slim block for the VAMP-2 slow-concept objective (lag-8 pairs; MSE persistence is
    # the refuted degenerate form). valid8=0 where the game ends first.
    C8SQ = np.zeros((N, 64, 8), np.uint8); C8GL = np.zeros((N, 6), np.uint8)
    C8ATT = np.full((N, ATT_CAP, 2), 255, np.uint8); C8NATT = np.zeros(N, np.uint16)
    VALID8 = np.zeros(N, np.uint8)
    OUTC = np.zeros(N, np.int8); P2E = np.zeros(N, np.uint16)
    ATT = np.full((N, ATT_CAP, 2), 255, np.uint8)
    NMV = np.zeros(N, np.uint8); PLY = np.zeros(N, np.uint8); NATT = np.zeros(N, np.uint16)
    trunc_mut = trunc_att = trunc_mv = trunc_vac = trunc_dsc = 0

    t0 = time.time()
    for i, r in enumerate(p_row):
        b = board_from_row(tr.tok[r], tr.glob[r])
        GL[i] = tr.glob[r]
        g, tc = parent_graph(b, nxt_tok=tr.tok[r + 1])
        SQ[i], MVI[i], MUT[i] = g["sq"], g["mv"], g["mut"]
        VAC[i], DSC[i], ATT[i] = g["vac"], g["dsc"], g["att"]
        NMV[i], PLY[i], NATT[i] = g["n_mv"], g["played"], g["n_att"]
        trunc_mv += tc["mv"]; trunc_mut += tc["mut"]
        trunc_vac += tc["vac"]; trunc_att += tc["att"]
        # child block: the played child's board
        cb = board_from_row(tr.tok[r + 1], tr.glob[r + 1])
        CGL[i] = tr.glob[r + 1]
        ne2 = 0
        for sq in range(64):
            pch = cb.piece_at(sq)
            aw2 = cb.attackers(chess.WHITE, sq); ab2 = cb.attackers(chess.BLACK, sq)
            CSQ[i, sq, 0] = 0 if pch is None else TOK_PIECES.index((pch.piece_type, pch.color)) + 1
            CSQ[i, sq, 1] = min(len(aw2), 7); CSQ[i, sq, 2] = min(len(ab2), 7)
            CSQ[i, sq, 3] = any(cb.piece_type_at(a) == chess.PAWN for a in aw2)
            CSQ[i, sq, 4] = any(cb.piece_type_at(a) == chess.PAWN for a in ab2)
            for a in (aw2 | ab2):
                if ne2 < ATT_CAP:
                    CATT[i, ne2] = (a, sq); ne2 += 1
        CNATT[i] = ne2
        if r + 8 <= g_end[r]:
            VALID8[i] = 1
            c8 = board_from_row(tr.tok[r + 8], tr.glob[r + 8])
            C8GL[i] = tr.glob[r + 8]
            ne3 = 0
            for sq in range(64):
                p8 = c8.piece_at(sq)
                aw3 = c8.attackers(chess.WHITE, sq); ab3 = c8.attackers(chess.BLACK, sq)
                C8SQ[i, sq, 0] = 0 if p8 is None else TOK_PIECES.index((p8.piece_type, p8.color)) + 1
                C8SQ[i, sq, 1] = min(len(aw3), 7); C8SQ[i, sq, 2] = min(len(ab3), 7)
                C8SQ[i, sq, 3] = any(c8.piece_type_at(a) == chess.PAWN for a in aw3)
                C8SQ[i, sq, 4] = any(c8.piece_type_at(a) == chess.PAWN for a in ab3)
                for a in (aw3 | ab3):
                    if ne3 < ATT_CAP:
                        C8ATT[i, ne3] = (a, sq); ne3 += 1
            C8NATT[i] = ne3
        OUTC[i] = tr.result[gor[r]] if tr.glob[r, 0] else -tr.result[gor[r]]
        P2E[i] = min(int(g_end[r] - r), 65535)
        if (i + 1) % 50000 == 0:
            el = time.time() - t0
            print(f"  {i+1:,}/{N:,}  {(i+1)/el:,.0f} pos/s  eta {(N-i-1)/((i+1)/el)/60:.1f}m",
                  flush=True)

    el = time.time() - t0
    print(f"\n  {N:,} graphs in {el/60:.1f}m | truncated: moves {trunc_mv} "
          f"({100*trunc_mv/N:.2f}%), mut {trunc_mut}, vac {trunc_vac}, attack edges {trunc_att}")
    if args.save:
        tag = f"_{args.tag}" if args.tag else ""
        p = paths.derived(f"move_graphs_v2{tag}_{N}_{n}.npz")
        np.savez_compressed(p, sq=SQ, gl=GL, mv=MVI, mut=MUT, vac=VAC, dsc=DSC, att=ATT,
                            n_mv=NMV, played=PLY, n_att=NATT, p_row=p_row,
                            c_sq=CSQ, c_gl=CGL, c_att=CATT, c_n_att=CNATT,
                            c8_sq=C8SQ, c8_gl=C8GL, c8_att=C8ATT, c8_n_att=C8NATT,
                            valid8=VALID8, outcome_mover=OUTC, plies_to_end=P2E)
        print(f"  wrote {p}")


if __name__ == "__main__":
    main()
