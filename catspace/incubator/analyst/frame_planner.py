"""frame_planner.py -- THE PLANNER SIDE of the frame player (Kaveh 2026-09-30: "all the hard-coded limits on how deep to search should
become parameters we can tune or specify from the planner side ... the planner should be able to say don't spend a
lot of effort here, or do ... the tactical layer doesn't understand any plan, it understands frames"; "solving puzzles
without the planner is going to be problematic -- you need the planner at least to do a simple material count, to see
whether we ought to be trading or avoiding trades ... a good start; later we want our learned planner layer there").

A planner, given a position and the candidate moves, returns a PLAN: the TacticalSpec for the tactical layer (its
depths, budgets, effort per candidate, stop rule) with the planner's GOALS as squares for the root zones (the tactical
layer takes squares, not plans), and a PREFERENCE among moves the tactics value equally.

MaterialPlanner is the first, rule-based stand-in for the learned planner (planner.py, 2026-09-11, is the learned
planning layer -- regime codes, steering targets, a search budget from the expected sharpness -- to plug in here):
- goals: the pawn runs (passed pawns within `runner_steps` of promotion, either side, and the squares ahead of them --
  moved here from the zone search, where it was hard-coded) and, if `watch_zone`, the king watches' pursued members;
- the preference: by the material count -- ahead by `trade_margin` or more, prefer the moves that trade; as far behind,
  prefer the ones that do not. Off by default (`trades=False`): on 2026-09-15 Kaveh found trading when ahead does NOT
  pay in Lichess games (+-0.02) and set "never code the rule" -- so it is a switch to be measured, and the learned
  planner's job to set.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import chess

from .zone_search import TacticalSpec

VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def runners(b: chess.Board, steps: int = 3) -> set:
    """Passed pawns within `steps` of promotion, either side, with the squares ahead of them to the last rank."""
    out = set()
    for c in (chess.WHITE, chess.BLACK):
        for s in b.pieces(chess.PAWN, c):
            f, r = chess.square_file(s), chess.square_rank(s)
            togo = 7 - r if c == chess.WHITE else r
            if togo > steps:
                continue
            ahead = range(r + 1, 8) if c == chess.WHITE else range(r - 1, -1, -1)
            if any(b.piece_at(chess.square(ff, rr)) == chess.Piece(chess.PAWN, not c)
                   for ff in (f - 1, f, f + 1) if 0 <= ff < 8 for rr in ahead):
                continue
            out.add(s); out |= {chess.square(f, rr) for rr in ahead}
    return out


def material(b: chess.Board, side: chess.Color) -> float:
    return sum(VAL[p.piece_type] * (1 if p.color == side else -1) for p in b.piece_map().values())


@dataclass
class Plan:
    spec: TacticalSpec
    prefer: object = None               # prefer(board, move, line) -> score; used only among the moves the tactics tie
    notes: list = field(default_factory=list)


@dataclass
class MaterialPlanner:
    spec: TacticalSpec = field(default_factory=TacticalSpec)
    runner_steps: int | None = 3        # None: no pawn-run goals
    watch_zone: int = 0                 # the king watches' top-N pursued members as goals (0: none)
    trades: bool = False                # the material-count trade preference (measured; see the module note)
    trade_margin: float = 2.0
    jobs: bool = False                  # hand the tactics the position's job ledger (jobs.py: pins, guards, discoveries)
    imagine: int = 0                    # the top-k imagined frames' squares join every candidate's root zone (imagine.goals)
    job_kinds: tuple = ("shield", "guard", "block", "discovery")
    job_min: float = 0.5

    def plan(self, b: chess.Board, moves: list) -> Plan:
        hints = []
        notes = []
        if self.runner_steps is not None:
            steps = self.runner_steps
            hints.append(lambda bb, s=steps: runners(bb, s))
        if self.watch_zone:
            hints.append(lambda bb, n=self.watch_zone: _watch_squares(bb, n))
        if self.imagine:                                              # the goals are the ROOT's imagined frames: the same squares
            from .imagine import goals                                # whichever candidate is searched
            squares = goals(b, self.imagine)[1]
            hints.append(lambda bb, sq=squares: sq)
            notes.append(f"imagined goals: {len(squares)} squares")
        prefer = None
        if self.imagine and not self.trades:                          # among the moves the tactics tie: the one advancing more
            from .imagine import values                               # imagined frames (measured: the solution first in 482 of
            iv = values(b)                                            # 669 steps vs 456 for a random tie-break)
            prefer = lambda board, m, line, iv=iv: (lambda t: (1000.0 if t[0] else 0.0) + t[1])(iv.get(m.uci(), (False, 0.0)))
        if self.trades:
            lead = material(b, b.turn)
            if abs(lead) >= self.trade_margin:
                want = lead > 0                                   # ahead: trade; behind: keep the pieces on
                prefer = lambda board, m, line, want=want: 1.0 if _trades(board, m, line) == want else 0.0
                notes.append(f"material {lead:+g}: {'prefer' if want else 'avoid'} trades")
        knowledge = tuple(self.spec.knowledge)
        if self.jobs:
            from .jobs import ledger
            knowledge += tuple(j for j in ledger(b) if j.kind in self.job_kinds and j.value >= self.job_min)
        return Plan(replace(self.spec, hints=tuple(self.spec.hints) + tuple(hints), knowledge=knowledge), prefer, notes)


def _watch_squares(bb: chess.Board, n: int) -> set:
    from .king_watch import watch, policy
    out = set()
    for c in (chess.WHITE, chess.BLACK):
        if bb.king(c) is None:
            continue
        w = watch(bb, c, operators=False)
        pursued = policy(w, n)
        for m in pursued:
            out |= {m.square, m.check_from} | set(m.route) | set(m.blockers)
        ids = {m.square for m in pursued}
        out |= {x.square for x in w.conditions if x.member in ids} | set(w.flights)
    return out


def _trades(b: chess.Board, m: chess.Move, line: list) -> bool:
    """A trade: the move captures and the principal line's reply takes back on the same square, for about as much."""
    if not b.is_capture(m) or len(line) < 2 or not line[1]:
        return False
    nb = b.copy(stack=False); nb.push(m)
    r = line[1]
    if r not in nb.legal_moves or r.to_square != m.to_square:
        return False
    won = VAL[b.piece_type_at(m.to_square)] if b.piece_at(m.to_square) else 1
    lost = VAL[nb.piece_type_at(r.to_square)]
    return abs(won - lost) <= 1
