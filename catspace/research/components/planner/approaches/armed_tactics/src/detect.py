"""catspace/armed/detect.py -- M7 DETECTION (see MILESTONES.md §M7, sequencing
override 2026-08-03): find candidate moves whose immediate committor gain
looks good but DECAYS over a few more plies of best play, identify the
SPECIFIC opponent move responsible for the biggest single-step decay, and
extract a concrete, cheaply-recheckable BLOCKING CONDITION from it -- the
payload of the M7 "armed-tactic record" (region/pattern, line, payoff,
blocking condition).

Reuses catspace.research.components.encoder.approaches.control_field_wdl.src.wdl_decay's validated SF-search decay check
(committor/best_move/_committor_or_terminal -- parked control-field work,
but that one utility is generic and proven, 99% on its own gate; this does
NOT un-park the ascent-cone/control-field thread itself, it only imports
three pure helper functions from it).

Blocking-condition representation, deliberately simple: a single fact --
"does color `defender_color` still attack `guarded_square`". Every M7 spec
example ("the Nf6 guards h7", "defender left, guard broken, pin released")
reduces to this one check: a capture-refutation's guarded_square is the
square our piece would be captured ON; a quiet-defensive-move refutation's
guarded_square is whatever critical square the new defender's attack set
shares with our own candidate piece's attack set (the two-piece overlap IS
the contested square in the common case). This is an approximation, stated
plainly: it will not correctly localize the guarded square for tactics whose
critical square isn't in either piece's immediate attack set (e.g. a
discovered-attack refutation) -- flagged as a known gap, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import chess.engine

from catspace.research.components.encoder.approaches.control_field_wdl.src.wdl_decay import committor, best_move, _committor_or_terminal


@dataclass
class BlockingCondition:
    """A single, cheaply-recheckable fact: does `defender_color` still attack
    `guarded_square` with at least `min_defenders` pieces? "Removed" = this
    flips False on a later board -- the spec's "defender left, guard broken,
    pin released" collapse to one predicate."""
    guarded_square: int
    defender_color: bool          # chess.WHITE / chess.BLACK
    min_defenders: int = 1
    source: str = "new_defender"  # "capture" | "new_defender" | "fallback" (diagnostic only)

    def is_active(self, board: chess.Board) -> bool:
        return len(board.attackers(self.defender_color, self.guarded_square)) >= self.min_defenders

    def describe(self) -> str:
        who = "White" if self.defender_color == chess.WHITE else "Black"
        return f"{who} still defends {chess.square_name(self.guarded_square)} ({self.source})"


@dataclass
class ArmedTacticCandidate:
    fen: str                        # the position where the candidate was found
    move: str                       # our candidate move, uci
    immediate_gain: float           # mover-POV committor gain from playing `move` alone
    trend: list                     # committor trajectory (mover POV), from wdl_decay
    blocking_ply: int | None        # index into `trend` where the biggest single-step drop occurs
    blocking_move: str | None       # the opponent move causing it, uci
    blocking: BlockingCondition | None
    payoff_if_unblocked: float      # trend[0] -- what's on offer if the blocker weren't there


def _trend_with_moves(eng: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move,
                       k_moves: int, depth: int):
    """Like wdl_decay.wdl_decay_trend, but also returns the actual moves played
    at each ply (that function only returns committor VALUES) so the specific
    blocking move can be identified, not just inferred from a number."""
    mover = board.turn
    b = board.copy()
    b.push(move)
    committors = [_committor_or_terminal(eng, b, depth, mover)]
    opp_moves, our_moves = [], []
    for _ in range(k_moves - 1):
        if b.is_game_over():
            break
        opp_m = best_move(eng, b, depth)
        b.push(opp_m)
        opp_moves.append(opp_m)
        if b.is_game_over():
            committors.append(_committor_or_terminal(eng, b, depth, mover))
            our_moves.append(None)
            break
        our_m = best_move(eng, b, depth)
        b.push(our_m)
        our_moves.append(our_m)
        committors.append(_committor_or_terminal(eng, b, depth, mover))
        if b.is_game_over():
            break
    return committors, opp_moves, our_moves


def _guarded_square_for_capture(board_before: chess.Board, blocking_move: chess.Move) -> int:
    """The refutation is a capture -- the checkable fact is "the defender can
    still capture/attack the square our piece occupies", i.e. the captured
    square itself."""
    if board_before.is_en_passant(blocking_move):
        return chess.square(chess.square_file(blocking_move.to_square),
                             chess.square_rank(blocking_move.from_square))
    return blocking_move.to_square


def _guarded_square_for_quiet_move(board_before: chess.Board, board_after: chess.Board,
                                    blocking_move: chess.Move,
                                    our_move: chess.Move | None) -> tuple[int, str]:
    """The refutation is a quiet defensive move (new guard, not a capture).
    Best-effort localization: the square the blocking piece newly attacks
    THAT our own candidate piece (from ITS CURRENT square, before it moves --
    `our_move.from_square`, still valid on `board_after` since only the
    defender has moved so far) also attacks -- the overlap is the contested
    square in the common case (both sides' pieces bear on the same spot).
    Falls back to the blocking move's own to-square (a coarser but still
    honestly-checkable fact: "the piece survives here") when no overlap is
    found -- e.g. discovered-attack refutations, a known gap."""
    defender_piece_attacks = board_after.attacks(blocking_move.to_square)
    if our_move is not None and board_after.piece_at(our_move.from_square):
        our_piece_attacks = board_after.attacks(our_move.from_square)
        overlap = defender_piece_attacks & our_piece_attacks
        if overlap:
            return next(iter(overlap)), "new_defender"
    return blocking_move.to_square, "fallback"


def classify_blocking_move(board_before: chess.Board, blocking_move: chess.Move,
                            our_next_move: chess.Move | None) -> BlockingCondition:
    """board_before is the position with the DEFENDER to move (about to play
    `blocking_move`); our_next_move is what we played immediately after it in
    the tracked line (may be None if the game ended)."""
    defender_color = board_before.turn
    if board_before.is_capture(blocking_move):
        sq = _guarded_square_for_capture(board_before, blocking_move)
        return BlockingCondition(guarded_square=sq, defender_color=defender_color, source="capture")
    board_after = board_before.copy(stack=False)
    board_after.push(blocking_move)
    sq, source = _guarded_square_for_quiet_move(board_before, board_after, blocking_move, our_next_move)
    return BlockingCondition(guarded_square=sq, defender_color=defender_color, source=source)


def find_armed_tactic_candidates(eng: chess.engine.SimpleEngine, board: chess.Board,
                                  k_candidates: int = 3, k_moves: int = 4, depth: int = 12,
                                  min_gain: float = 0.15, decay_tol: float = 0.05) -> list[ArmedTacticCandidate]:
    """For up to `k_candidates` of the mover's legal moves (ranked by immediate
    committor gain), check whether the gain DECAYS under `k_moves` plies of SF
    best play both sides (wdl_decay's mechanism). A candidate whose immediate
    gain clears `min_gain` but whose trend drops by more than `decay_tol`
    qualifies as an armed-tactic candidate: it "almost works" -- flag it with
    its specific blocking move/condition instead of discarding it."""
    mover = board.turn
    c_now = _committor_or_terminal(eng, board, depth, mover)
    legal = list(board.legal_moves)
    scored = []
    for m in legal:
        b2 = board.copy(stack=False); b2.push(m)
        c_after = _committor_or_terminal(eng, b2, depth, mover)
        scored.append((c_after - c_now, m))
    scored.sort(key=lambda x: -x[0])

    out = []
    for gain, move in scored[:k_candidates]:
        if gain < min_gain:
            continue
        trend, opp_moves, our_moves = _trend_with_moves(eng, board, move, k_moves, depth)
        if len(trend) < 2:
            continue
        drops = [trend[i] - trend[i + 1] for i in range(len(trend) - 1)]
        worst_i = int(max(range(len(drops)), key=lambda i: drops[i])) if drops else None
        net_trend = trend[-1] - trend[0]
        if worst_i is None or drops[worst_i] < decay_tol or net_trend >= -decay_tol:
            continue   # sound: no meaningful single-step decay, not "almost works but blocked"

        b_before = board.copy(stack=False)
        b_before.push(move)
        for i in range(worst_i):
            b_before.push(opp_moves[i]); b_before.push(our_moves[i])
        blocking_move = opp_moves[worst_i]
        our_next = our_moves[worst_i] if worst_i < len(our_moves) else None
        blocking = classify_blocking_move(b_before, blocking_move, our_next)

        out.append(ArmedTacticCandidate(
            fen=board.fen(), move=move.uci(), immediate_gain=round(gain, 4), trend=trend,
            blocking_ply=worst_i, blocking_move=blocking_move.uci(), blocking=blocking,
            payoff_if_unblocked=round(trend[0], 4)))
    return out
