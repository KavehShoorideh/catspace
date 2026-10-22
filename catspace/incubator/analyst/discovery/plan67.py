"""plan 67 THE PLAN'S OWN WEAKNESSES (Lean `Ideas.Plan`; Kaveh 2026-10-22, dictated): roll OUR plan forward without the opponent, read the weak
moments it creates by itself, make each one THEIR goal with a deadline, and close the plan by arithmetic when no piece of theirs reaches any weak
moment in time (`Ideas.Plan.plan_sound`); otherwise the reachable weak moments are the plan's DEFECTS, and a support placement with no reachable
weak moment (the rook BEHIND the pawn) is a fix of ours (67.4).

The plan here is the promotion plan of one passer: its single pushes (never the double step: it allows the en-passant capture the arithmetic does not
read), with a piece of ours that blocks the path stepping aside just before the pawn needs its square. Their pieces stand still in the rollout; what
they could do in the meantime is the reach test, with the PROVED lower bounds (`Race.kdist` for the king, `Passer2.pdist` for a pawn, `Race.lb` for a
rook, bishop, queen or knight), blockers ignored (a blocker can only slow them: over-approximating their reach is the safe direction, `safe_mono`).

Weak moments listed (every square that could matter; more is safer):
  capture   -- a plan piece (the pawn; the promoted queen; a support piece) stands UNPROTECTED on a square after our move t: deadline t (the last move
               at which it still stands there; `standDeadline`).
  blockade  -- the square the pawn pushes to at our move t+1 must be empty when they reply to move t: deadline t.
  check     -- a square from which a piece of theirs of some kind would attack our king on the board after our move t (lines through the pawn included:
               a pin is on the same line): deadline t. A check costs the plan a tempo, so the arithmetic refuses it; the chase reads such replies.
A plan is CLOSED (claim) when no weak moment is reachable and they are never stalemated by our pushes alone. Only the arithmetic decides; nothing is
estimated. Every number in the trace is a count printed by this module.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import chess

from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery.race import kdist, lb, pdist

PLAN = os.environ.get("CATSPACE_PLAN", "1") != "0"                # plan 67 (ON; `analyse --no-plan` sets CATSPACE_PLAN=0, inherited by the workers: a comparison condition only)

_SLIDER_KINDS = (chess.ROOK, chess.BISHOP, chess.QUEEN, chess.KNIGHT)


@dataclass
class Weak:
    sq: int
    deadline: int
    kind: str                                                    # capture | blockade | check
    what: str                                                    # the piece or the reason, for the trace
    by: frozenset = frozenset()                                  # check: the piece types of theirs that would attack our king from this square (a king never checks)

    def text(self) -> str:
        return f"{G.name(self.sq)} by move {self.deadline} ({self.kind}: {self.what})"


@dataclass
class Reach:
    piece: str                                                   # 'K', 'R', ... of theirs
    frm: int
    weak: Weak
    need: float                                                  # the proved lower bound on their moves

    def text(self) -> str:
        return f"their {self.piece} on {G.name(self.frm)} reaches {self.weak.text()} in {int(self.need)}"


@dataclass
class Plan:
    pawn: int
    moves: list = field(default_factory=list)                    # our moves in order (chess.Move)
    boards: list = field(default_factory=list)                   # the rollout: boards[t] after our move t (boards[0] = the start), THEM to move
    weak: list = field(default_factory=list)
    reach: list = field(default_factory=list)
    stalemate_at: int | None = None                              # a rollout board where they have no legal move (a draw; the arithmetic refuses it)
    mate_at: int | None = None                                   # ... with their king in check: mate, the plan's end (no weak moment at or after it)
    their_race: float = float("inf")                             # the fewest pushes a pawn of theirs needs to its promotion rank (blockers ignored: a lower bound)
    why: str = ""

    @property
    def n(self) -> int:
        return len(self.moves)

    @property
    def closed(self) -> bool:
        """The promotion is PROVED by arithmetic: no weak moment of any kind reachable, no stalemate."""
        return bool(self.moves) and not self.reach and self.stalemate_at is None and self.their_race > self.n

    @property
    def immune(self) -> bool:
        """No CAPTURE or BLOCKADE weak moment reachable: the plan's own weaknesses are covered; only checks (forcing replies the chase reads) remain."""
        return bool(self.moves) and self.stalemate_at is None and not any(r.weak.kind in ("capture", "blockade") for r in self.reach)

    @property
    def own_reach(self) -> list:
        return [r for r in self.reach if r.weak.kind in ("capture", "blockade")]

    def line(self, board: chess.Board) -> str:
        b = board.copy(stack=False)
        out = []
        for m in self.moves:
            out.append(b.san(m))
            b = _our_move(b, m)
        return " ".join(out)


# ---------------------------------------------------------------------------------------------------------------- 67.1 the rollout

def _our_move(b: chess.Board, m: chess.Move) -> chess.Board:
    """The board after our move with the opponent PASSING (our move again)."""
    c = b.copy(stack=False)
    c.push(m)
    return G.with_turn(c, not c.turn)


def _path(me: bool, sq: int) -> list:
    step = 8 if me == chess.WHITE else -8
    out, s = [], sq + step
    while 0 <= s < 64:
        out.append(s)
        s += step
    return out


def _step_aside(b: chess.Board, blocker: int, path_rest: set) -> chess.Move | None:
    """A legal move of our blocker off the pawn's path (the first that keeps it on the board and off the path); None: it cannot step aside."""
    for m in b.legal_moves:
        if m.from_square == blocker and m.to_square not in path_rest:
            return m
    return None


def rollout(board: chess.Board, pawn: int) -> Plan:
    """67.1: the pushes of the pawn with our blockers stepping aside, the opponent passing. The plan stops where the rules stop it (a piece of theirs on the
    path: the plan ends there, that square a blockade weak moment at deadline 0 -- reached already)."""
    me = board.turn
    b = G.with_turn(board, me)
    plan = Plan(pawn=pawn, boards=[b])
    path = _path(me, pawn)
    sq = pawn
    for i, nxt in enumerate(path):
        occ = b.piece_at(nxt)
        if occ is not None and occ.color != me:
            plan.why = f"their {occ.symbol().upper()} stands on {G.name(nxt)}"
            break
        if occ is not None:
            m = _step_aside(b, nxt, set(path[i:]))
            if m is None:
                plan.why = f"our {occ.symbol().upper()} on {G.name(nxt)} cannot step aside"
                break
            plan.moves.append(m)
            b = _our_move(b, m)
            plan.boards.append(b)
        promo = chess.QUEEN if chess.square_rank(nxt) in (0, 7) else None
        push = chess.Move(sq, nxt, promotion=promo)
        if push not in b.legal_moves:
            plan.why = f"the push {G.name(sq)}-{G.name(nxt)} is not legal (a pin, a check)"
            break
        plan.moves.append(push)
        b = _our_move(b, push)
        plan.boards.append(b)
        sq = nxt
        if promo:
            break
    return plan


# ---------------------------------------------------------------------------------------------------------------- 67.2 the weak moments

def _plan_pieces(board: chess.Board, plan: Plan) -> set:
    """The squares of the pieces the plan moves (the pawn and every stepping-aside piece), on the start board."""
    return {m.from_square for m in plan.moves if board.piece_at(m.from_square) is not None} | {plan.pawn}


def _check_squares(b: chess.Board, me: bool) -> dict:
    """Every square from which a piece of theirs would attack our king, by the piece types that could (blockers ignored: the rays through our own pieces
    included, a pin lies on the same line). -> {square: frozenset of piece types}."""
    k = b.king(me)
    if k is None:
        return {}
    out = {}
    for s in chess.SQUARES:
        if s == k:
            continue
        df, dr = abs(chess.square_file(s) - chess.square_file(k)), abs(chess.square_rank(s) - chess.square_rank(k))
        kinds = set()
        if df == 0 or dr == 0:
            kinds |= {chess.ROOK, chess.QUEEN}
        if df == dr:
            kinds |= {chess.BISHOP, chess.QUEEN}
        if sorted((df, dr)) == [1, 2]:
            kinds.add(chess.KNIGHT)
        if df == 1 and (chess.square_rank(s) - chess.square_rank(k)) == (1 if me == chess.WHITE else -1):
            kinds.add(chess.PAWN)                                # a pawn of theirs checks from the square diagonally ahead of our king
        if kinds:
            out[s] = frozenset(kinds)
    return out


def weak_moments(board: chess.Board, plan: Plan) -> list:
    """67.2: the weak moments of the rollout (the deadline of a stand is the last move at which the piece still stands there, `Ideas.Plan.standDeadline`)."""
    me = board.turn
    n = plan.n
    out = []
    tracked = {}                                                 # square -> (arrive, piece symbol): where plan pieces stand, from when
    for s in _plan_pieces(board, plan):
        tracked[s] = (0, board.piece_at(s).symbol().upper())
    path = _path(me, plan.pawn)
    for t in range(0, n + 1):
        b = plan.boards[t]
        if t > 0:
            m = plan.moves[t - 1]
            arrive, sym = tracked.pop(m.from_square, (t, "?"))
            tracked[m.to_square] = (t, b.piece_at(m.to_square).symbol().upper())
        for s, (arrive, sym) in tracked.items():
            if sym == "K":
                continue                                         # the king is not captured; a check on it is the `check` kind
            if not b.is_attacked_by(me, s):                      # unprotected on this board: a capture at their reply t needs their piece on s by t
                out.append(Weak(s, t, "capture", f"our {sym} stands unprotected"))        # (the dedupe below keeps the LATEST t per square: `standDeadline`)
        if t < n and b.piece_type_at(plan.moves[t].from_square) == chess.PAWN and plan.moves[t].to_square in path:
            out.append(Weak(plan.moves[t].to_square, t, "blockade", "the pawn pushes there next"))
        for s, kinds in _check_squares(b, me).items():
            out.append(Weak(s, t, "check", "a check from there costs the plan a tempo", kinds))
        if t > 0 and plan.stalemate_at is None and plan.mate_at is None:
            theirs = G.with_turn(b, not me)
            if not any(True for _ in theirs.legal_moves):
                if theirs.is_check():
                    plan.mate_at = t                             # the plan ends in mate: no reply exists after it
                else:
                    plan.stalemate_at = t
    # one weak per (square, kind) at the latest deadline (the strongest requirement); the trace stays short
    best = {}
    for w in out:
        if plan.mate_at is not None and w.deadline >= plan.mate_at:
            continue                                             # no reply of theirs exists after the mate
        k = (w.sq, w.kind)
        if k not in best or best[k].deadline < w.deadline:
            best[k] = w
    plan.weak = sorted(best.values(), key=lambda w: (w.kind != "capture", w.kind != "blockade", w.deadline, w.sq))
    return plan.weak


# ---------------------------------------------------------------------------------------------------------------- 67.3 their goals: the reach

def _bound(piece: chess.Piece, frm: int, sq: int, kind: str) -> float:
    """The PROVED lower bound on the moves their piece needs to stand on `sq` (a capture lands on it; a blockade stands on it; a check is given from it)."""
    if piece.piece_type == chess.KING:
        return kdist(frm, sq)
    if piece.piece_type == chess.PAWN:
        if kind == "capture":                                    # a pawn captures diagonally: it must first stand on a square attacking `sq`
            fwd = 1 if piece.color == chess.WHITE else -1
            r = chess.square_rank(sq) - fwd
            best = float("inf")
            for df in (-1, 1):
                f = chess.square_file(sq) + df
                if 0 <= f < 8 and 0 <= r < 8:
                    best = min(best, pdist(piece.color, frm, chess.square(f, r)) + 1)
            return best
        return pdist(piece.color, frm, sq)
    if piece.piece_type == chess.ROOK:
        return rdist(frm, sq)
    if piece.piece_type == chess.BISHOP:
        return bdist(frm, sq)
    return lb(frm, sq)


def rdist(s: int, t: int) -> int:
    """Plan.rdist: a rook's lower bound: 0, 1 (the same file or rank), else 2."""
    if s == t:
        return 0
    return 1 if chess.square_file(s) == chess.square_file(t) or chess.square_rank(s) == chess.square_rank(t) else 2


def bdist(s: int, t: int) -> float:
    """Plan.bdist: a bishop's lower bound: 0, 1 (the same diagonal), 2 (the same colour), never (the other colour)."""
    if s == t:
        return 0
    df, dr = abs(chess.square_file(s) - chess.square_file(t)), abs(chess.square_rank(s) - chess.square_rank(t))
    if df == dr:
        return 1
    return 2 if (chess.square_file(s) + chess.square_rank(s)) % 2 == (chess.square_file(t) + chess.square_rank(t)) % 2 else float("inf")


def reach(board: chess.Board, plan: Plan) -> list:
    """67.3: the weak moments a piece of theirs reaches by its deadline (`deadline < bound` for all = safe, `Ideas.Plan.safe`)."""
    me = board.turn
    out = []
    for w in plan.weak:
        for s in chess.SQUARES:
            p = board.piece_at(s)
            if p is None or p.color == me:
                continue
            if w.kind == "check" and p.piece_type not in w.by:
                continue                                         # that piece cannot check from there (a king never does)
            need = _bound(p, s, w.sq, w.kind)
            if need <= w.deadline:
                out.append(Reach(p.symbol().upper(), s, w, need))
    plan.reach = out
    return out


def read(board: chess.Board, pawn: int) -> Plan:
    """The whole read: rollout, weak moments, reach. `plan.closed` = the promotion is proved by arithmetic (no reachable weak moment, no stalemate)."""
    plan = rollout(board, pawn)
    if plan.moves:
        weak_moments(board, plan)
        reach(board, plan)
        them = not board.turn
        promo_rank = 0 if them == chess.WHITE else 7
        for s in board.pieces(chess.PAWN, them):
            plan.their_race = min(plan.their_race, pdist(them, s, chess.square(chess.square_file(s), 7 if them == chess.WHITE else 0)))
    return plan


def trace(board: chess.Board, plan: Plan) -> str:
    """The trace line (every count is printed here)."""
    if not plan.moves:
        return f"plan: no plan for the pawn on {G.name(plan.pawn)}: {plan.why}"
    head = f"plan {plan.line(board)} ({plan.n} move{'s' if plan.n != 1 else ''}): {len(plan.weak)} weak moment(s)"
    if plan.stalemate_at is not None:
        return head + f"; they are stalemated after move {plan.stalemate_at}: refused"
    if plan.mate_at is not None:
        head += f" (mate at move {plan.mate_at})"
    if not plan.reach and plan.their_race <= plan.n:
        return head + f", none reachable, but a pawn of theirs promotes within {int(plan.their_race)}: the race word decides, not this plan"
    if not plan.reach:
        return head + ", none reachable: the promotion is proved by arithmetic"
    own = plan.own_reach
    checks = len(plan.reach) - len(own)
    if not own:
        return head + f", none of ours reachable (immune to capture and blockade); {checks} check square(s) reachable: the chase reads the checks"
    return head + f", {len(own)} of ours reachable: " + "; ".join(r.text() for r in own[:3]) + (f"; {checks} check square(s) reachable" if checks else "")


# ---------------------------------------------------------------------------------------------------------------- 67.4 the regression: a support placement

def support_fixes(board: chess.Board, pawn: int, limit_kinds: tuple = ("capture", "blockade")) -> list:
    """67.4: our quiet moves after which the plan (this move first, then the pushes) has NO reachable weak moment -- the support placed so that no step
    unprotects (the rook behind the pawn). -> [(move, san, the plan after it)], the shortest plans first. Only plans whose reachable weak moments are
    of the kinds in `limit_kinds` are worth a support move (a reachable CHECK square is not fixed by support: the chase reads checks)."""
    me = board.turn
    base = read(board, pawn)
    if not base.moves or base.immune:
        return []
    out = []
    for m in board.legal_moves:
        if m.from_square == pawn or board.is_capture(m) or board.gives_check(m):
            continue                                             # a capture or a check is a forcing route of its own; the support move is QUIET
        b2 = _our_move(board, m)
        if b2.king(me) is not None and b2.is_attacked_by(not me, b2.king(me)):
            continue
        p2 = read(b2, pawn)
        if p2.immune:
            p2.moves.insert(0, m)
            p2.boards.insert(0, G.with_turn(board, me))
            out.append((m, board.san(m), p2))
    out.sort(key=lambda t: (t[2].n, t[1]))
    return out
