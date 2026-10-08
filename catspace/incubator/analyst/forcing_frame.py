"""THE FORCING FRAME: the exchange joined with every capture elsewhere, and (optionally) checks -- a frame on the board
whose moves are the forcing ones (Kaveh 2026-10-08: "the exchange should include intermezzos ... a higher value threat
somewhere else"; the replay's surprises: captures elsewhere carry 59% of the frame's error, checks in between 39%).

A state is (board, plies left, checks left). The side to move may EXIT (stand pat) unless it is in check; its moves are
its captures (any square, promotions included) and, while `checks left` > 0, its checking moves; in check, every legal
evasion. Mate is -MATE for the mated side; at the horizon the side to move stands pat (0) unless mated. Values are
material in pawns, negamax, searched by frame_search.FrameSearch (lemmas, priors, supports) like any frame. This is the
classical quiescence search with checks, seen as one frame among others: its moves are the board's forcing moves.
"""
from __future__ import annotations

import chess

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
MATE = 20.0


def double_threat(b: chess.Board) -> float:
    """THE DOUBLE-THREAT EXIT LEMMA (mined 2026-10-08 from unflagged blunders: Kxf1 Qxd4 -- the frame let White stand pat
    while the queen hit a1 and d-file at once): standing pat costs the opponent's SECOND-best threat. A threat = the
    opponent's static exchange on one of our pieces were we to pass (the null-move loss, read on the safe null move);
    one threat can be parried, two cannot both be. -> the cost (>= 0)."""
    from .chess_domain import passed
    from .exchange_law import see
    p = passed(b)
    if p is None:
        return 0.0
    vals = sorted((see(p, q)[0] for q, pc in p.piece_map().items()
                   if pc.color == b.turn and pc.piece_type != chess.KING and p.attackers(p.turn, q)), reverse=True)
    vals = [v for v in vals if v > 0]
    return float(vals[1]) if len(vals) >= 2 else 0.0


def _best_capture(b: chess.Board) -> int:
    """The side to move's best LEGAL capture by the static exchange (0 when none wins)."""
    from .exchange_law import see
    best = 0
    for x in b.legal_moves:
        if b.is_capture(x):
            t = b.piece_at(x.to_square)
            if t is not None and t.piece_type != chess.KING:
                best = max(best, see(b, x.to_square)[0])
    return best


def parry_cost(b: chess.Board) -> float:
    """THE PARRY LEMMA (mined 2026-10-08 from unflagged blunders: 119 were answered by a quiet move making ONE threat --
    a threat that cannot be parried): standing pat costs the least the opponent still wins by a capture after our best
    single reply. Read only when the opponent has a threat (their best capture on the safe null move); it subsumes the
    double threat (no single reply parries two), and catches trapped, pinned and overloaded targets. A reply that is
    itself a capture counts its own gain. -> the cost (>= 0)."""
    from .chess_domain import passed
    p = passed(b)
    if p is None or _best_capture(p) < 1:
        return 0.0
    best = None
    for m in b.legal_moves:
        g = 0
        if b.is_capture(m):
            t = b.piece_at(m.to_square)
            g = VAL[t.piece_type] if t is not None else 1
        nb = b.copy(stack=False); nb.push(m)
        if nb.is_checkmate():
            return 0.0
        if nb.is_check():                                  # a check postpones the threat, it does not parry it (bug
            continue                                       # 2026-10-08: Qa4+ read as a parry of the Qxd4 double attack)
        loss = _best_capture(nb) - g
        best = loss if best is None else min(best, loss)
        if best <= 0:
            return 0.0
    return float(max(0, best if best is not None else 0))


class ForcingFrame:
    def __init__(self, checks: bool = True, threats: bool | str = False):
        self.checks = checks
        self.threats = threats                           # False, True (the double-threat exit lemma) or "parry"

    def moves(self, st):
        b, plies, checks = st
        if plies <= 0:
            return []
        if b.is_check():
            return list(b.legal_moves)
        out = [m for m in b.legal_moves if b.is_capture(m) or m.promotion]
        if self.checks and checks > 0:
            out += [m for m in b.legal_moves if not b.is_capture(m) and not m.promotion and b.gives_check(m)]
        victim = lambda m: VAL[(b.piece_at(m.to_square) or chess.Piece(chess.PAWN, not b.turn)).piece_type] if b.is_capture(m) else 0
        return sorted(out, key=lambda m: (-victim(m), VAL[b.piece_at(m.from_square).piece_type]))   # most valuable victim first

    def gain(self, st, m):
        b = st[0]
        g = 0.0
        if b.is_capture(m):
            g += VAL[(b.piece_at(m.to_square) or chess.Piece(chess.PAWN, not b.turn)).piece_type]
        if m.promotion:
            g += VAL[m.promotion] - 1
        return g

    def next(self, st, m):
        b, plies, checks = st
        nb = b.copy(stack=False); nb.push(m)
        quiet_check = not b.is_capture(m) and not m.promotion and not b.is_check()
        return (nb, plies - 1, checks - (1 if quiet_check else 0))

    def exit(self, st):
        b, plies, _ = st
        if b.is_check():
            return -MATE if (plies <= 0 and b.is_checkmate()) or plies > 0 else 0.0   # in check: no standing pat
        if self.threats == "parry":
            return -parry_cost(b)
        return -double_threat(b) if self.threats else 0.0

    def key(self, st):
        b, plies, checks = st
        return (b.board_fen(), b.turn, b.castling_rights, b.ep_square, plies, checks)


def forcing_lemmas(threats: bool = False):
    from .frame_search import Lemma

    def stand(st):                                       # not in check, the side to move can always stand pat
        return None if st[0].is_check() else (0.0, float("inf"))
    if threats == "parry":                               # standing pat costs what no single reply parries
        return [Lemma("stand pat", "proved: the exit is a move of the frame (FrameSearch.exit)",
                      state=lambda st: None if st[0].is_check() else (-parry_cost(st[0]), float("inf")))]
    if threats:                                          # standing pat now costs the double threat: the bound moves with it
        return [Lemma("stand pat", "proved: the exit is a move of the frame (FrameSearch.exit)",
                      state=lambda st: None if st[0].is_check() else (-double_threat(st[0]), float("inf")))]
    return [Lemma("stand pat", "proved: the exit is a move of the frame (FrameSearch.exit)", state=stand)]


def capture_value(b: chess.Board, s: int, plies: int = 8, checks: int = 2, use_checks: bool = True, threats: bool = False) -> tuple:
    """The value for the side to move of capturing on s (its best legal capture there), in the forcing frame.
    -> (value, nodes)."""
    from .frame_search import FrameSearch
    fs = FrameSearch(ForcingFrame(use_checks, threats), forcing_lemmas(threats))
    fr = fs.frame; st = (b, plies, checks)
    best = None
    for m in b.legal_moves:
        if m.to_square == s and b.is_capture(m):
            v = fr.gain(st, m) - fs.value(fr.next(st, m))
            best = v if best is None else max(best, v)
    return (best if best is not None else 0.0), fs.nodes
