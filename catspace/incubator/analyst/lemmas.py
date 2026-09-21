"""FAMILY LEMMAS (Kaveh 2026-09-20, the Be6 position): a fact found in one search that closes a whole family of
replies in the next. "Any move of the bishop on e6 discovers check from the queen on e7 -- and fxe7 answers every
one of them, so no bishop move needs searching." The mechanism is generic: a FAMILY is a set of the opponent's
replies that share a structure (here: moves of one piece that open one line onto our king); a LEMMA is one answer
of ours that is legal and adequate after every member. The search then evaluates the family once, and the
explanation states the lemma instead of the members. Rules only: lines, checks, captures."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .facts import value


@dataclass
class Family:
    kind: str                      # "discovered_check"
    mover: int                     # the square of the piece whose moves make the family
    opener: int                    # the piece whose line the family opens (the checker)
    members: list = field(default_factory=list)      # the replies

    def text(self, board: chess.Board) -> str:
        p, q = board.piece_at(self.mover), board.piece_at(self.opener)
        return f"any move of the {chess.piece_name(p.piece_type)} on {chess.square_name(self.mover)} discovers check from the {chess.piece_name(q.piece_type)} on {chess.square_name(self.opener)}"


@dataclass
class Lemma:
    family: Family
    answer: chess.Move             # our move that answers every member
    covers: list = field(default_factory=list)      # the members it covers
    text: str = ""


def discovery_possible(board: chess.Board) -> bool:
    """The cheap gate: the side to move has a slider aligned with the enemy king with exactly one own piece between.
    Without it no reply can discover check, and the reply loop is skipped (on puzzles the ungated detection cost
    more than the lemmas saved, 2026-09-21)."""
    us = board.turn; ek = board.king(not us)
    if ek is None:
        return False
    for sq, p in board.piece_map().items():
        if p.color != us or p.piece_type not in (chess.QUEEN, chess.ROOK, chess.BISHOP):
            continue
        df, dr = chess.square_file(ek) - chess.square_file(sq), chess.square_rank(ek) - chess.square_rank(sq)
        straight = df == 0 or dr == 0; diag = abs(df) == abs(dr) and df != 0
        if not ((straight and p.piece_type != chess.BISHOP) or (diag and p.piece_type != chess.ROOK)):
            continue
        between = [t for t in chess.SquareSet.between(sq, ek) if board.piece_at(t) is not None]
        if len(between) == 1 and board.piece_at(between[0]).color == us:
            return True
    return False


def discovered_check_families(board: chess.Board) -> list[Family]:
    """The side to move's replies that give check by a piece OTHER than the one that moved (the moved piece opened
    its line), grouped by (mover, opener)."""
    if not discovery_possible(board):
        return []
    fams: dict[tuple[int, int], Family] = {}
    for r in board.legal_moves:
        nb = board.copy(stack=False); nb.push(r)
        if not nb.is_check():
            continue
        for chk in nb.checkers():
            if chk != r.to_square:                                   # the checker did not move: a discovery
                key = (r.from_square, chk)
                fams.setdefault(key, Family("discovered_check", r.from_square, chk)).members.append(r)
    return [f for f in fams.values() if len(f.members) >= 2]


def refute(board: chess.Board, fam: Family) -> Lemma | None:
    """One answer for the whole family: a capture of the opener, by a piece that is not the king, legal after every
    non-capturing member, and not a losing exchange (the opener is worth at least the capturer)."""
    us = not board.turn
    candidates = None
    for r in fam.members:
        nb = board.copy(stack=False); nb.push(r)
        answers = {a for a in nb.legal_moves if a.to_square == fam.opener and nb.piece_at(a.from_square).piece_type != chess.KING
                   and value(nb, fam.opener) >= value(nb, a.from_square)}
        # a member that itself captures may need its own accounting; the lemma covers the others and notes the capturers
        candidates = answers if candidates is None else (candidates & answers)
        if not candidates:
            return None
    ans = sorted(candidates, key=lambda a: value(board, a.from_square))[0]
    covered = [r for r in fam.members if not board.is_capture(r)]
    if not covered:
        return None
    lem = Lemma(fam, ans, covered)
    nb0 = board.copy(stack=False); nb0.push(covered[0])
    san = nb0.san(ans) if ans in nb0.legal_moves else ans.uci()
    lem.text = (f"{fam.text(board)}; {san} answers every one of them"
                f" ({len(covered)} replies searched once" + (f"; {len(fam.members) - len(covered)} that also capture searched on their own" if len(fam.members) > len(covered) else "") + ")")
    return lem


def lemmas_after(board: chess.Board) -> list[Lemma]:
    """The lemmas over the side to move's reply families, for the searcher who just moved."""
    out = []
    for fam in discovered_check_families(board):
        lem = refute(board, fam)
        if lem is not None:
            out.append(lem)
    return out
