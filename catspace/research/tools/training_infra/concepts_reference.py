#!/usr/bin/env python
"""concepts_reference.py -- the python-chess ORACLE for the Rust probe basis. (Kaveh 2026-08-17)

"python-chess has functionality to probe the board in various ways... I want to build a version in
Rust... then test it on the corpus for accuracy."

This module is not a throwaway. It is the SPECIFICATION the Rust crate is differential-tested
against, so it is written to be read as a definition rather than to be fast. Every feature is
computed from python-chess's own probe API -- `attackers`, `attacks`, `pin`, `is_pinned`,
`checkers`, `legal_moves`, `SquareSet.between` -- and every dimension has a name in SQ_NAMES /
GLOB_NAMES. Those names are the whole point: a concept distilled into a program prints as a
weighted expression over them, which is what a direction in a learned trunk can never do.

WHY A NAMED BASIS AT ALL. Concepts today are directions in a 192-d JepaEncoder embedding. The
plan is to distil each one into a Rust program over these primitives, freeze it, and let the
network compensate -- until the trunk can be deleted and inference is pure Rust. Measured, the
trunk costs 2.347 ms/position single-threaded, i.e. ~2.6 s/move at 800 nodes; this basis targets
5-20 us.

TWO OUTPUTS, matching the two codebooks the model already has:
    sq_feat   (64, 28)   per square -- feeds the 64-square VQ
    glob_feat (100,)     position-level -- feeds the 8 global heads

SEE is IMPORTED, not redefined. `control._see_square` is the repo's existing swap-off algorithm
and is now the single definition; a second one would silently disagree (the module it lives in
notes python-chess has no SEE of its own). The Rust must match THAT algorithm, not "some SEE".

REPETITION AND HALFMOVE ARE NOT POSITION-LOCAL. They cannot be recovered from (tok, glob) and are
passed in from the corpus. Forgetting this produces features that disagree with python-chess on
exactly the rows where it matters, which is a nasty class of bug to chase later.

    .venv/bin/python -m catspace.research.tools.training_infra.concepts_reference        # tests
"""
from __future__ import annotations

import numpy as np

import chess

from catspace.research.components.encoder.approaches.control_field_wdl.src.control import (
    _see_square)

# jepa token vocab: 0 empty, 1-6 white PNBRQK, 7-12 black pnbrqk
TOK_PIECES = [(chess.PAWN, chess.WHITE), (chess.KNIGHT, chess.WHITE), (chess.BISHOP, chess.WHITE),
              (chess.ROOK, chess.WHITE), (chess.QUEEN, chess.WHITE), (chess.KING, chess.WHITE),
              (chess.PAWN, chess.BLACK), (chess.KNIGHT, chess.BLACK), (chess.BISHOP, chess.BLACK),
              (chess.ROOK, chess.BLACK), (chess.QUEEN, chess.BLACK), (chess.KING, chess.BLACK)]
PIECE_VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
             chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 10}

_SYM = "PNBRQKpnbrqk"
SQ_NAMES = (
    [f"occ_{s}" for s in _SYM]                       # 0-11  occupant one-hot
    + ["att_w", "att_b",                             # 12-13 attacker counts
       "least_att_w", "least_att_b",                 # 14-15 cheapest attacker's value
       "see",                                        # 16    swap-off value if captured here
       "pinned", "pinning",                          # 17-18 occupant is pinned / pins
       "xray_w", "xray_b",                           # 19-20 attackers through one blocker
       "pawn_att_w", "pawn_att_b",                   # 21-22 attacked by a pawn
       "hanging", "defenders",                       # 23-24 attacked & undefended / defender count
       "attacks_count",                              # 25    squares the occupant attacks
       "king_ring_w", "king_ring_b"]                 # 26-27 in a king's ring
)
N_SQ = len(SQ_NAMES)

GLOB_NAMES = (
    [f"n_{s}" for s in _SYM]                                        # 12 material counts
    + ["material_diff", "phase_nonpawn", "piece_count"]
    + [f"pawn_{k}_{c}_file{f}" for k in ("passed", "isolated", "doubled", "backward")
       for c in ("w", "b") for f in range(8)]                       # 64 pawn structure
    + [f"king_{k}_{c}" for k in ("ring_pressure", "shield", "escape", "open_files")
       for c in ("w", "b")]                                         # 8 king safety
    + ["stm_white", "castle_K", "castle_Q", "castle_k", "castle_q",
       "ep_file", "in_check", "checker_count"]                      # 8
    + ["tension", "legal_moves", "checks_available"]                # 3
    + ["repetition", "halfmove"]                                    # 2 -- from the corpus
)
N_GLOB = len(GLOB_NAMES)


# ------------------------------------------------------------------------------------------
def board_from_row(tok, glob) -> chess.Board:
    """(tok (64,) uint8, glob (6,) = [turn, K, Q, k, q, ep_file+1]) -> Board.

    The corpus stores a complete position this way, so neither FEN parsing nor game replay is
    needed -- Rust reads the same two arrays. `tok` is indexed by python-chess square index
    (a1=0 .. h8=63), which is what `gen_capture_labels` relies on when it reads a capture's
    victim off `tok[row, move.to_square]`.
    """
    b = chess.Board(None)
    for sq in range(64):
        t = int(tok[sq])
        if t:
            pt, col = TOK_PIECES[t - 1]
            b.set_piece_at(sq, chess.Piece(pt, col))
    b.turn = chess.WHITE if int(glob[0]) else chess.BLACK
    rights = 0
    if int(glob[1]): rights |= chess.BB_H1
    if int(glob[2]): rights |= chess.BB_A1
    if int(glob[3]): rights |= chess.BB_H8
    if int(glob[4]): rights |= chess.BB_A8
    b.castling_rights = rights
    ep = int(glob[5])
    # the ep TARGET square sits behind the pawn that just double-pushed: rank 6 when White is to
    # move (Black pushed), rank 3 when Black is to move.
    b.ep_square = chess.square(ep - 1, 5 if b.turn == chess.WHITE else 2) if ep else None
    return b


def sq_features(board: chess.Board) -> np.ndarray:
    """-> (64, N_SQ) float32, indexed by chess square index."""
    out = np.zeros((64, N_SQ), np.float32)
    occ = board.occupied
    kw, kb = board.king(chess.WHITE), board.king(chess.BLACK)
    ring_w = chess.SquareSet(chess.BB_KING_ATTACKS[kw]) if kw is not None else chess.SquareSet()
    ring_b = chess.SquareSet(chess.BB_KING_ATTACKS[kb]) if kb is not None else chess.SquareSet()
    for sq in range(64):
        p = board.piece_at(sq)
        if p is not None:
            out[sq, TOK_PIECES.index((p.piece_type, p.color))] = 1.0
        aw = board.attackers(chess.WHITE, sq)
        ab = board.attackers(chess.BLACK, sq)
        out[sq, 12] = len(aw)
        out[sq, 13] = len(ab)
        out[sq, 14] = min((PIECE_VAL[board.piece_type_at(a)] for a in aw), default=0)
        out[sq, 15] = min((PIECE_VAL[board.piece_type_at(a)] for a in ab), default=0)
        if p is not None:
            # SEE from the perspective of the side that does NOT own the piece -- "what does
            # capturing here win?" is only a question for the opponent.
            out[sq, 16] = _see_square(board, sq, not p.color)
            out[sq, 17] = float(board.is_pinned(p.color, sq))
            # PINNING: this piece holds an enemy piece against its king -- the mirror of `pinned`,
            # which python-chess only answers for the pinned side.
            pin_flag = 0.0
            if p.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                ek = board.king(not p.color)
                if ek is not None:
                    ray = chess.SquareSet.ray(sq, ek)
                    if ray and sq != ek:
                        between = chess.SquareSet.between(sq, ek) & occ
                        if len(between) == 1:
                            mid = list(between)[0]
                            mp = board.piece_at(mid)
                            if mp is not None and mp.color != p.color and \
                                    sq in board.attackers(p.color, mid):
                                pin_flag = 1.0
            out[sq, 18] = pin_flag
            out[sq, 23] = float(bool(board.attackers(not p.color, sq)) and
                                not bool(board.attackers(p.color, sq)))
            out[sq, 24] = len(board.attackers(p.color, sq))
            out[sq, 25] = len(board.attacks(sq))
        # X-RAY: attackers hidden behind exactly ONE blocker -- the queen on f3 aiming at f7
        # through a knight on f6. Computed with the `occupied` override, which is precisely what
        # that parameter is for: ask who would attack this square if NOTHING blocked, drop those
        # already attacking directly, and keep the ones with a single man in the way. Removing
        # the attackers from `occupied` (the obvious-looking version) is backwards -- it deletes
        # the very pieces whose sight-lines are in question.
        for col, direct, idx in ((chess.WHITE, aw, 19), (chess.BLACK, ab, 20)):
            unblocked = board.attackers_mask(col, sq, chess.BB_EMPTY)
            hidden = unblocked & ~int(direct.mask if hasattr(direct, "mask")
                                      else board.attackers_mask(col, sq))
            out[sq, idx] = sum(1 for s2 in chess.SquareSet(hidden)
                               if len(chess.SquareSet.between(s2, sq) & occ) == 1)
        out[sq, 21] = float(any(board.piece_type_at(a) == chess.PAWN for a in aw))
        out[sq, 22] = float(any(board.piece_type_at(a) == chess.PAWN for a in ab))
        out[sq, 26] = float(sq in ring_w)
        out[sq, 27] = float(sq in ring_b)
    return out


def _pawn_structure(board, color):
    """-> (passed, isolated, doubled, backward) each an 8-vector over files."""
    ours = board.pieces(chess.PAWN, color)
    theirs = board.pieces(chess.PAWN, not color)
    fwd = 1 if color == chess.WHITE else -1
    files_ours = [0] * 8
    for s in ours:
        files_ours[chess.square_file(s)] += 1
    passed = np.zeros(8, np.float32); iso = np.zeros(8, np.float32)
    dbl = np.zeros(8, np.float32); back = np.zeros(8, np.float32)
    for s in ours:
        f, r = chess.square_file(s), chess.square_rank(s)
        if not any(chess.square_file(t) in (f - 1, f, f + 1) and
                   (chess.square_rank(t) - r) * fwd > 0 for t in theirs):
            passed[f] = 1.0
        if not any(files_ours[g] for g in (f - 1, f + 1) if 0 <= g <= 7):
            iso[f] = 1.0
        # BACKWARD: no friendly pawn on an adjacent file at or behind this rank, so it cannot be
        # supported by one; the classic definition, and distinct from isolated.
        if not any(chess.square_file(t) in (f - 1, f + 1) and
                   (chess.square_rank(t) - r) * fwd <= 0 for t in ours if t != s):
            back[f] = 1.0
    for f in range(8):
        dbl[f] = max(0, files_ours[f] - 1)
    return passed, iso, dbl, back


def glob_features(board: chess.Board, repetition: float = 0.0,
                  halfmove: float = 0.0) -> np.ndarray:
    """-> (N_GLOB,) float32. `repetition`/`halfmove` come from the corpus: they are properties of
    the GAME, not of the position, and cannot be recovered from (tok, glob)."""
    out = np.zeros(N_GLOB, np.float32)
    i = 0
    counts = {}
    for k, (pt, col) in enumerate(TOK_PIECES):
        counts[(pt, col)] = len(board.pieces(pt, col))
        out[i + k] = counts[(pt, col)]
    i += 12
    matw = sum(PIECE_VAL[pt] * counts[(pt, chess.WHITE)] for pt in
               (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
    matb = sum(PIECE_VAL[pt] * counts[(pt, chess.BLACK)] for pt in
               (chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN))
    out[i] = matw - matb
    out[i + 1] = (matw + matb) - (counts[(chess.PAWN, chess.WHITE)]
                                  + counts[(chess.PAWN, chess.BLACK)])
    out[i + 2] = len(board.piece_map())
    i += 3
    for kind_idx in range(4):
        for c_idx, col in enumerate((chess.WHITE, chess.BLACK)):
            arr = _pawn_structure(board, col)[kind_idx]
            out[i:i + 8] = arr
            i += 8
    for kind in ("ring_pressure", "shield", "escape", "open_files"):
        for col in (chess.WHITE, chess.BLACK):
            k = board.king(col)
            if k is None:
                i += 1
                continue
            ring = chess.SquareSet(chess.BB_KING_ATTACKS[k])
            if kind == "ring_pressure":
                out[i] = sum(len(board.attackers(not col, s)) for s in ring)
            elif kind == "shield":
                out[i] = sum(1 for s in ring if (q := board.piece_at(s)) is not None
                             and q.color == col and q.piece_type == chess.PAWN)
            elif kind == "escape":
                out[i] = sum(1 for s in ring if board.piece_at(s) is None
                             and not board.attackers(not col, s))
            else:
                kf = chess.square_file(k)
                out[i] = sum(1 for f in {max(0, kf - 1), kf, min(7, kf + 1)}
                             if not any((q := board.piece_at(chess.square(f, r))) is not None
                                        and q.piece_type == chess.PAWN and q.color == col
                                        for r in range(8)))
            i += 1
    out[i] = float(board.turn == chess.WHITE)
    out[i + 1] = float(board.has_kingside_castling_rights(chess.WHITE))
    out[i + 2] = float(board.has_queenside_castling_rights(chess.WHITE))
    out[i + 3] = float(board.has_kingside_castling_rights(chess.BLACK))
    out[i + 4] = float(board.has_queenside_castling_rights(chess.BLACK))
    out[i + 5] = 0.0 if board.ep_square is None else chess.square_file(board.ep_square) + 1
    out[i + 6] = float(board.is_check())
    out[i + 7] = len(board.checkers())
    i += 8
    legal = list(board.legal_moves)
    out[i] = sum(1 for m in legal if board.is_capture(m))
    out[i + 1] = len(legal)
    out[i + 2] = sum(1 for m in legal if board.gives_check(m))
    i += 3
    out[i] = repetition
    out[i + 1] = halfmove
    return out


def extract(tok, glob, repetition=None, halfmove=None):
    """Batch oracle. -> (sq (N,64,N_SQ), gl (N,N_GLOB)) float32."""
    n = len(tok)
    sq = np.zeros((n, 64, N_SQ), np.float32)
    gl = np.zeros((n, N_GLOB), np.float32)
    for r in range(n):
        b = board_from_row(tok[r], glob[r])
        sq[r] = sq_features(b)
        gl[r] = glob_features(b, 0.0 if repetition is None else float(repetition[r]),
                              0.0 if halfmove is None else float(halfmove[r]))
    return sq, gl


# ------------------------------------------------------------------------------------------
def _tests():
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        cond = bool(cond); ok &= cond
        print(f"  {'OK  ' if cond else 'FAIL'} {name}{('  ' + extra) if extra else ''}")

    chk("SQ_NAMES has no duplicates", len(set(SQ_NAMES)) == len(SQ_NAMES), f"{N_SQ} dims")
    chk("GLOB_NAMES has no duplicates", len(set(GLOB_NAMES)) == len(GLOB_NAMES), f"{N_GLOB} dims")
    print(f"       total basis = 64*{N_SQ} + {N_GLOB} = {64*N_SQ + N_GLOB} named dims")

    b = chess.Board()
    s, g = sq_features(b), glob_features(b)
    chk("startpos shapes", s.shape == (64, N_SQ) and g.shape == (N_GLOB,))
    chk("startpos: e1 holds a white king",
        s[chess.E1, SQ_NAMES.index("occ_K")] == 1.0)
    chk("startpos: 20 legal moves, 0 captures, 0 checks",
        g[GLOB_NAMES.index("legal_moves")] == 20 and g[GLOB_NAMES.index("tension")] == 0
        and g[GLOB_NAMES.index("checks_available")] == 0)
    chk("startpos: material_diff 0, 32 pieces",
        g[GLOB_NAMES.index("material_diff")] == 0 and g[GLOB_NAMES.index("piece_count")] == 32)
    # d4 is attacked twice by White at the start (Qd1 through d2? no -- c3/e3 knights? no):
    # it is attacked by the c- and e-pawns... which have not moved. Assert against python-chess
    # itself rather than a remembered number.
    chk("att_w matches board.attackers everywhere",
        all(s[q, SQ_NAMES.index("att_w")] == len(b.attackers(chess.WHITE, q)) for q in range(64)))

    # a position with real structure: pins, x-ray, passers, checks
    f = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1"
    b2 = chess.Board(f)
    s2, g2 = sq_features(b2), glob_features(b2)
    # f7 is the textbook case for the x-ray feature: Bc4 hits it directly, while Qf3 is blocked
    # by the knight on f6 -- so a DIRECT count of 1 and an x-ray of 1 is the correct reading, and
    # a basis that reported 2 direct attackers would be wrong about the position.
    chk("f7: exactly one DIRECT attacker (Bc4; Qf3 is blocked by Nf6)",
        s2[chess.F7, SQ_NAMES.index("att_w")] == 1,
        f"got {s2[chess.F7, SQ_NAMES.index('att_w')]:.0f}")
    chk("f7: the blocked queen shows up as an X-RAY",
        s2[chess.F7, SQ_NAMES.index("xray_w")] >= 1,
        f"got {s2[chess.F7, SQ_NAMES.index('xray_w')]:.0f}")
    chk("hanging is never set on an empty square",
        all(s2[q, SQ_NAMES.index("hanging")] == 0 for q in range(64)
            if b2.piece_at(q) is None))
    chk("occupancy one-hot sums to 1 exactly on occupied squares",
        all(abs(s2[q, :12].sum() - (1.0 if b2.piece_at(q) else 0.0)) < 1e-6 for q in range(64)))
    chk("pinned agrees with board.is_pinned",
        all(s2[q, SQ_NAMES.index("pinned")] == float(b2.is_pinned(p.color, q))
            for q in range(64) if (p := b2.piece_at(q)) is not None))
    chk("xray is non-negative everywhere",
        (s2[:, SQ_NAMES.index("xray_w")] >= 0).all()
        and (s2[:, SQ_NAMES.index("xray_b")] >= 0).all())

    # ROUND TRIP: the corpus encoding must reconstruct the position exactly
    for fen in [chess.STARTING_FEN, f,
                "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
                "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"]:
        bb = chess.Board(fen)
        tok = np.zeros(64, np.uint8)
        for q, p in bb.piece_map().items():
            tok[q] = TOK_PIECES.index((p.piece_type, p.color)) + 1
        gl = np.array([int(bb.turn),
                       int(bb.has_kingside_castling_rights(chess.WHITE)),
                       int(bb.has_queenside_castling_rights(chess.WHITE)),
                       int(bb.has_kingside_castling_rights(chess.BLACK)),
                       int(bb.has_queenside_castling_rights(chess.BLACK)),
                       0 if bb.ep_square is None else chess.square_file(bb.ep_square) + 1],
                      np.uint8)
        rt = board_from_row(tok, gl)
        same = (rt.board_fen() == bb.board_fen() and rt.turn == bb.turn
                and rt.castling_rights == bb.castling_rights)
        chk(f"round-trip {fen.split(' ')[0][:24]}...", same,
            "" if same else f"\n        got {rt.fen()}\n        want {bb.fen()}")

    print("\nALL CONCEPTS-REFERENCE TESTS PASSED" if ok else "\nCONCEPTS-REFERENCE TESTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_tests())
