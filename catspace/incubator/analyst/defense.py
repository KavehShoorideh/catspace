"""defense.py -- the EFFECTIVE DEFENDERS operator (Kaveh 2026-09-05: compose from the atoms so a rule shift keeps
the meaning). A piece that attacks a square is only a defender if it can actually recapture there and recapturing
does not cost more than it saves. Both halves are computed from the rules, not named:
  legal       simulate the cheapest capture, ask the rules whether the recapture is a legal move
              (absolute pins, a king recapturing into check, atomic adjacency, forced captures... whatever the variant says)
  economic    play the recapture and ask what the enemy can now win that it could not right after its capture (the
              removal test on the defender's origin, priced by SEE): a defender that unshields a bigger piece (a relative
              pin) or abandons another duty (an overload) does not count; what the capture itself uncovered is not
              charged to the recapturer
Every named four-piece fact -- pinned-and-attacked, hanging through a pin, overloaded -- is a READING of this trace,
and the trace explains itself: "its only defender, the knight on d4, cannot recapture: it is pinned to the rook on a1".
One level deep: exposures are priced with nominal defenders elsewhere (no overload chains)."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from . import rules as R
from .facts import see, value
from .memo import pin_line


@dataclass(frozen=True)
class Disqualified:
    square: int                 # the defender that does not count
    reason: str                 # "illegal" | "pinned" | "overloaded" | "loses_more"
    exposed: int | None         # the own piece the enemy could win once the defender recaptures
    cost: float                 # what that exposure is worth to the enemy


@dataclass
class Defense:
    target: int
    attackers: list[int]
    defenders: list[int]                        # nominal: own pieces attacking the square
    effective: list[int]                        # the ones that count
    disqualified: list[Disqualified] = field(default_factory=list)
    see: float = 0.0                            # SEE from the attacker's side with only the effective defenders: > 0 => the target falls

    @property
    def falls(self) -> bool:
        return self.see > 0


def _cheapest_legal_capture(board: chess.Board, sq: int, by: chess.Color, reads=None) -> chess.Move | None:
    b = board.copy(stack=False); b.turn = by
    cands = sorted(b.attackers(by, sq), key=lambda a: value(board, a))
    for a in cands:
        if reads is not None:
            reads.square(a); reads.legality(board, by, a)      # the capture's legality: the attacker's pin line, its king's checkers
        for mv in (chess.Move(a, sq), chess.Move(a, sq, promotion=chess.QUEEN)):
            if mv in b.legal_moves:
                return mv
    return None


SLIDERS = (chess.BISHOP, chess.ROOK, chess.QUEEN)


def exposure_candidates(board: chess.Board, dq: int, color: chess.Color) -> set[int]:
    """The own pieces a defender on dq could expose by leaving: those it defends, and those it shields from an
    enemy slider (dq lies between the slider and the piece). Nothing else can change when dq moves to the
    capture square, so nothing else is read -- which is what lets the concept store serve this across positions."""
    out = {t for t in board.attacks(dq) if board.piece_at(t) and board.piece_at(t).color == color}
    for a, ap in board.piece_map().items():
        if ap.color == color or ap.piece_type not in SLIDERS:
            continue
        ray = chess.ray(a, dq)
        if not ray or dq not in chess.SquareSet(ray):
            continue
        for t in chess.SquareSet(ray):
            tp = board.piece_at(t)
            if tp and tp.color == color and t != dq and dq in chess.SquareSet.between(a, t):
                out.add(t)
    return out


def _enemy_gains(board: chess.Board, color: chess.Color, squares, reads=None) -> dict[int, float]:
    """For the given pieces of `color`: what SEE says the enemy wins by starting on each."""
    out = {}
    for t in squares:
        piece = board.piece_at(t)
        if piece is None or piece.color != color:
            continue
        if reads is not None:
            reads.attacks_of(board, t)                  # whether it is attacked at all is a read
        if not board.attackers(not color, t):
            continue
        g = see(board, t, not color, reads=reads)
        if g > 0:
            out[t] = g
    return out


def defense(board: chess.Board, sq: int, memo=None, reads=None) -> Defense | None:
    """The defense of the piece on sq against the enemy's capture sequence, or None when nothing attacks it.
    With `memo`, served from the concept store when the pieces on its footprint are unchanged; `reads` lets a
    caller (reach) fold this computation's footprint into its own."""
    if memo is not None:
        return memo.get("defense", (sq,), board, lambda r: _defense(board, sq, r))
    return _defense(board, sq, reads)


def _defense(board: chess.Board, sq: int, reads=None) -> Defense | None:
    piece = board.piece_at(sq)
    if reads is not None:
        reads.attacks_of(board, sq)
    if piece is None:
        return None
    c = piece.color
    attackers = sorted(board.attackers(not c, sq), key=lambda a: value(board, a))
    if not attackers:
        return None
    k = R.royal_square(board, c)
    if k is not None:
        if reads is not None:
            reads.checkers(board, c)                           # a predicate, recomputed at lookup: not the king's whole region
        if board.is_attacked_by(not c, k):                     # the owner is in check: no static exchange exists until it is met
            return None
    defenders = sorted(board.attackers(c, sq))
    d = Defense(sq, attackers, defenders, [])
    cap = _cheapest_legal_capture(board, sq, not c, reads)
    if cap is None:                                            # no legal capture exists: nothing to defend against
        d.effective = list(defenders); d.see = 0.0
        return d
    after = board.copy(stack=False); after.turn = not c; after.push(cap)
    won = value(board, cap.from_square)                        # what a recapture wins back: the capturer
    for dq in defenders:
        if not after.piece_at(dq) or after.piece_at(dq).color != c:
            continue
        if reads is not None:                                  # the recapture's legality: dq's pin line, and the line the
            reads.add(pin_line(after, c, dq) | pin_line(board, c, cap.from_square))   # capturer's departure may open onto the king
            reads.square(dq)
        rec = next((m for m in (chess.Move(dq, sq), chess.Move(dq, sq, promotion=chess.QUEEN)) if m in after.legal_moves), None)
        if rec is None:
            d.disqualified.append(Disqualified(dq, "illegal", None, 0.0)); continue
        cands = exposure_candidates(after, dq, c)              # only what this defender's departure can change
        if reads is not None:
            for a, ap in after.piece_map().items():            # which enemy sliders align through dq is itself a read
                if ap.color != c and ap.piece_type in SLIDERS:
                    reads.square(a)
            reads.attacks_of(board, dq)
        base_gains = _enemy_gains(after, c, cands, reads)      # exposures the CAPTURE itself created (a discovered attack) are not the recapturer's fault
        b2 = after.copy(stack=False); b2.push(rec)
        gains = _enemy_gains(b2, c, cands, reads)
        # the exposure: an own piece the enemy can now win (or win more of) that it could not before the recapture,
        # the recapturing piece itself excluded (its fate on sq is the SEE continuation, not an exposure)
        best_sq, best = None, 0.0
        for t, g in gains.items():
            if t == sq:
                continue
            delta = g - base_gains.get(t, 0.0)
            if delta > best:
                best_sq, best = t, delta
        if best_sq is not None and best >= won:
            # name the reason from the geometry: beyond the defender's origin on a line from an enemy slider -> pinned;
            # a piece the defender was guarding -> overloaded; anything else -> loses more
            reason = "loses_more"
            if best_sq in board.attacks(dq):
                reason = "overloaded"
            elif chess.SquareSet.between(dq, best_sq) is not None and any(
                    dq in chess.SquareSet.between(a, best_sq) for a in board.attackers(not c, dq)
                    if board.piece_at(a) and board.piece_at(a).piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN)):
                reason = "pinned"
            d.disqualified.append(Disqualified(dq, reason, best_sq, best)); continue
        d.effective.append(dq)
    d.see = see(board, sq, not c, exclude={x.square for x in d.disqualified}, reads=reads)
    return d


def all_defenses(board: chess.Board, memo=None) -> dict[int, Defense]:
    out = {}
    for sq, piece in board.piece_map().items():
        if R.is_royal(board, sq):
            continue
        df = defense(board, sq, memo=memo)
        if df is not None:
            out[sq] = df
    return out
