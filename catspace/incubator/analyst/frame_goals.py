"""FRAME GOALS -- the graph of reasoning over frames (Kaveh 2026-09-28: "which attacker do we bring in? ... that
would invalidate any frame the knight touches ... we want to break that pawn chain ... where are the weak points of
that frame? where can we add attackers ... we are trying to disprove a theorem"; "graph of reasoning rather than
chain -- might need branching").

A GOAL is an enemy piece's frame to break. Its interface is the frames' own: the pieces that attack it and the
defenders that count (a pinned defender does not; `defense.defense`). The graph branches both ways:
  * OR over OPERATORS: bring one more attacker (a route: `reach.reach`), or take a defender off (any one move after
    which it no longer counts -- a pin, a block, an attack that ties it; found by trying the move, not by name);
  * AND inside a PLAN: operators combined, because one alone may not break it (c4 adds an attacker to d5 and Rh6
    pins its defender e6: either alone breaks nothing, both do -- Nxd5);
  * AND over the opponent's ANSWERS: every reply after our first move that keeps the frame whole;
  * the pieces LINK the goals: an answer that moves a piece off a plan of its own side gives that plan up (after
    Rg1 in Kaveh's position, Nh3 defends f2 and leaves White's d5 plan without its knight) -- the dilemma.

Everything is VERIFIED by the frames (`fates.joint_threats`), with the null move of lambda-search (Thomsen 2000): a
plan breaks its goal when, playing our moves with the opponent passing, the frames then win at least GAIN more than
before on a line through the target; an answer holds when, after it, the rest of the plan no longer does.

    python -m catspace.incubator.analyst.frame_goals --fen "<fen>"
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field

import chess

from .defense import defense
from .fates import captured_value, joint_threats, passed, square_frame
from .reach import reach

GAIN = 0.5                                                        # a frame breaks when the frames win at least this much more
MAX_MOVES = 3                                                     # our moves in a plan, at most
MAX_HOPS = 2                                                      # an attacker's route, at most


def _pinned_off(b: chess.Board, d: int, s: int) -> bool:
    pc = b.piece_at(d)
    return pc is not None and b.is_pinned(pc.color, d) and not (b.pin(pc.color, d) & chess.BB_SQUARES[s])


def counted_defenders(b: chess.Board, t: int) -> set:
    """The defenders of the piece on t that count: the ledger's effective ones when t is attacked, else its owner's
    pieces bearing on t that are not pinned off it."""
    if b.piece_at(t) is None:
        return set()                                              # the target has left t (castling moves a rook too)
    d = defense(b, t)
    if d is not None:
        return set(d.effective)
    owner = b.piece_at(t).color
    return {q for q in b.attackers(owner, t) if not _pinned_off(b, q, t)}


def _to_move(b: chess.Board, side: chess.Color) -> chess.Board | None:
    return b if b.turn == side else passed(b)


def _frames(b: chess.Board) -> tuple:
    v, sq, ln = joint_threats(b, square_frame)
    return v, ln


def _hits(line: list, t: int) -> bool:
    return any(chess.Move.from_uci(u).to_square == t for u in line)


@dataclass
class Op:
    kind: str                       # "attacker" | "removes"
    piece: int                      # the moving piece's square
    moves: tuple                    # our moves, in order
    text: str


@dataclass
class Answer:
    move: chess.Move
    san: str
    vacates: list                   # the answering side's own plans (Plan) this move gives up


@dataclass
class Plan:
    side: chess.Color
    target: int
    ops: list
    moves: list                     # our moves in order
    gain: float                     # what the frames win at the end, beyond what they won before
    line: list                      # the frames' winning line at the end (uci)
    answers: list = field(default_factory=list)   # the answers to its final threat (Answer)
    checked: bool = False           # through / answers computed
    through: bool = False
    stopped_by: list | None = None

    def san(self, b: chess.Board) -> str:
        bb = _to_move(b, self.side).copy(); out = []
        for m in self.moves:
            out.append(bb.san(m)); bb.push(m); bb = passed(bb) or bb
        return " ".join(out)


def operators(b: chess.Board, side: chess.Color, t: int) -> list:
    """The ways to change the frame of the enemy piece on t, each one operator: bring one more attacker by a route,
    or take one counted defender off by a single move (tried, not named)."""
    b0 = _to_move(b, side)
    if b0 is None or b0.piece_at(t) is None or b0.piece_at(t).color == side:
        return []
    out = []
    attacking = set(b0.attackers(side, t))
    one = set()
    for m in b0.legal_moves:                                      # every one-move way to bear on t
        pc = b0.piece_at(m.from_square)
        if b0.is_capture(m) or pc.piece_type == chess.KING or m.from_square in attacking:
            continue
        nb = b0.copy(stack=False); nb.push(m)
        if nb.is_check():
            continue
        if t in nb.attacks(m.to_square):
            d = defense(nb, m.to_square)
            if d is not None and d.falls:
                continue                                          # it would land where it falls (Rg2 in 7R/3b2r1/..: Nxg2), as reach requires
            one.add(m.from_square)
            out.append(Op("attacker", m.from_square, (m,), f"the {chess.piece_name(pc.piece_type)} on {chess.square_name(m.from_square)} comes to bear on {chess.square_name(t)} ({b0.san(m)})"))
    for p, pc in b0.piece_map().items():                          # two-move routes (reach) for the pieces with no safe one-move way
        if pc.color != side or pc.piece_type == chess.KING or p in attacking or p in one:
            continue
        path = reach(b0, p, t, MAX_HOPS)
        if path is None or path.hops < 2:
            continue
        bb = b0.copy(stack=False); moves = []; cur = p
        for to in path.squares:
            m = next((x for x in bb.legal_moves if x.from_square == cur and x.to_square == to), None)
            if m is None:
                moves = None; break
            moves.append(m); bb.push(m); bb = passed(bb)
            if bb is None:
                moves = None; break
            cur = to
        if moves:
            out.append(Op("attacker", p, tuple(moves), f"the {chess.piece_name(pc.piece_type)} on {chess.square_name(p)} comes to bear on {chess.square_name(t)}"))
    before = counted_defenders(b0, t)
    for m in b0.legal_moves:
        if b0.is_capture(m) or b0.piece_at(m.from_square).piece_type == chess.KING:
            continue
        nb = b0.copy(stack=False); nb.push(m)
        if nb.is_check():
            continue
        pb = passed(nb)
        if pb is None or pb.piece_at(t) is None:
            continue
        gone = before - counted_defenders(pb, t)
        if gone:
            d = min(gone)
            out.append(Op("removes", m.from_square, (m,), f"{b0.san(m)} takes the {chess.piece_name(b0.piece_at(d).piece_type)} on {chess.square_name(d)} off {chess.square_name(t)}"))
    return out


def verify(b: chess.Board, side: chess.Color, t: int, moves: list) -> tuple | None:
    """Play `moves` for `side` with the opponent passing; -> (gain beyond the frames' value before, the frames' line)
    when the frames then win at least GAIN more on a line through t, else None."""
    b0 = _to_move(b, side)
    if b0 is None or b0.piece_at(t) is None:
        return None
    base, _ = frame_on(b0, t)
    bb = b0
    for m in moves:
        if m not in bb.legal_moves:
            return None
        bb = bb.copy(stack=False); bb.push(m)
        if bb.is_check():
            return None                                           # a check is not a quiet step: the frames resolve checks
        bb = passed(bb)
        if bb is None:
            return None
    v, ln = frame_on(bb, t)
    if v - base >= GAIN:
        return v - base, ln
    return None


def frame_on(b: chess.Board, t: int) -> tuple:
    """The frame of the square t joined with what it pulls in (fates.joint_frame), resolved for the side to move:
    -> (value, line). A plan is judged on its own target's frame, not on the board's best frame (after Bd3 in
    8/5pk1/.., the best frame is Bxe2; the c4 frame's own resolution is Bxc4)."""
    from .fates import joint_frame, local_resolve
    if b.piece_at(t) is None or b.piece_at(t).color == b.turn:
        return 0.0, []
    key = (b._transposition_key(), t)
    if key not in _FRAME_ON:
        if len(_FRAME_ON) > 200000:
            _FRAME_ON.clear()
        sq, kings, members = joint_frame(b, {t}, set())
        _FRAME_ON[key] = local_resolve(b, sq, set(members), kings)
    return _FRAME_ON[key]


_FRAME_ON: dict = {}


def plans_for(b: chess.Board, side: chess.Color, targets=None) -> list:
    """Every plan of at most MAX_MOVES of our moves that breaks an enemy frame, one operator or two (both orders: the
    order decides which answers are in time), verified by the frames; smallest first."""
    b0 = _to_move(b, side)
    if b0 is None:
        return []
    out = []; seen = set()
    for t, pc in sorted(b0.piece_map().items()):
        if pc.color == side or pc.piece_type == chess.KING or (targets is not None and t not in targets):
            continue
        ops = operators(b0, side, t)
        combos = [[o] for o in ops] + [[o1, o2] for i, o1 in enumerate(ops) for o2 in ops[i + 1:] if o1.piece != o2.piece]
        for combo in combos:
            n = sum(len(o.moves) for o in combo)
            if n > MAX_MOVES:
                continue
            orders = [combo] if len(combo) == 1 else [combo, combo[::-1]]
            for order in orders:
                moves = [m for o in order for m in o.moves]
                key = (t, tuple(moves))
                if key in seen:
                    continue
                seen.add(key)
                got = verify(b0, side, t, moves)
                if got is None:
                    continue
                if len(combo) == 2 and any(verify(b0, side, t, list(o.moves)) for o in combo):
                    continue                                      # one operator alone does it: the pair adds nothing
                out.append(Plan(side, t, order, moves, got[0], got[1]))
    out.sort(key=lambda p: (len(p.moves), -p.gain))
    return out


def _zone(b0: chess.Board, plan: Plan) -> set:
    """The squares the plan's proof lives on (Cazenave's trace): the target, the pieces bearing on it, the plan's
    squares, and the lines between them. A reply that touches none of them cannot stop it."""
    t = plan.target
    z = {t} | set(b0.attackers(chess.WHITE, t)) | set(b0.attackers(chess.BLACK, t))
    for m in plan.moves:
        z |= {m.from_square, m.to_square}
    for u in plan.line[:4]:
        m = chess.Move.from_uci(u); z |= {m.from_square, m.to_square}
    zs = list(z)
    for i, x in enumerate(zs):
        for y in zs[i + 1:]:
            z |= set(chess.SquareSet(chess.between(x, y)))
    return z


def _replies(b: chess.Board, zone: set | None, t: int | None = None) -> list:
    """The replies that can matter to a plan on target t: those touching its zone, captures, checks, a move of the
    target, and any move that adds a counted defender to it (the frame's interface: a defender arriving from h3 or
    h2 touches no square of the proof, yet changes the count). Every other reply is the pass it was verified against."""
    if zone is None:
        return list(b.legal_moves)
    out = []
    base = None
    for a in b.legal_moves:
        if a.from_square in zone or a.to_square in zone or b.is_capture(a) or b.gives_check(a) or a.from_square == t:
            out.append(a); continue
        if t is not None and b.piece_at(t) is not None and b.piece_at(t).color == b.turn:
            if base is None:
                base = len(counted_defenders(b, t))
            ab = b.copy(stack=False); ab.push(a)
            if ab.piece_at(t) is None or len(counted_defenders(ab, t)) > base:
                out.append(a)
    return out


def _stops(b: chess.Board, plan: Plan, i: int, before: float, zone: set, first: bool, debt: float = 0.0) -> list:
    """AND node: the board after our move plan.moves[i], their turn. -> their replies that stop the plan (all of them
    at the first reply, for the report; the first found deeper, where only its existence matters). A reply stops it
    when it gives nothing away and either the frames no longer win on the target at the end, or our next move is no
    longer possible, or they have a stopping reply again after it; a check stops the fixed sequence."""
    out = []
    for a in _replies(b, None if first else zone, plan.target):
        ab = b.copy(stack=False); ab.push(a)
        if ab.is_check():
            out.append(a)
            if not first:
                return out
            continue
        took = debt + captured_value(b, a)                        # what their replies took on this line: our frames are net of it
        if _frames(ab)[0] - took >= before + GAIN and not (i == len(plan.moves) - 1):
            continue                                              # it gives something away (judged at the end otherwise)
        if i == len(plan.moves) - 1:
            v, ln = _frames(ab)
            stopped = v - took < before + GAIN                    # wherever the frames win it (f3 moves the target into exf3: no answer);
                                                                  # net of their capture (Qf5 Qxf5 gxf5 is a trade, not +9)
        else:
            m = plan.moves[i + 1]
            if m not in ab.legal_moves:
                stopped = True
            else:
                nb = ab.copy(stack=False); nb.push(m)
                stopped = nb.is_check() or bool(_stops(nb, plan, i + 1, before, zone, False, took))
        if stopped:
            out.append(a)
            if not first:
                return out
    return out


def goes_through(b: chess.Board, plan: Plan) -> tuple:
    """Does the plan go through against every defence? An AND/OR check with a reply after each of our moves; replies
    off the plan's zone are the pass the plan was verified against (Cazenave's trace). -> (True, None) or (False,
    the first stopping line found: [(ply index, their reply SAN)])."""
    b0 = _to_move(b, plan.side)
    before, _ = _frames(b0)
    nb = b0.copy(stack=False); nb.push(plan.moves[0])
    stop = _stops(nb, plan, 0, before, _zone(b0, plan), False)
    if not stop:
        return True, None
    return False, [nb.san(stop[0])]


def final_answers(b: chess.Board, plan: Plan) -> list:
    """Their answers to the plan's FINAL threat: our moves played with them passing up to the last one; after it, the
    replies that stop the frames winning on the target and give nothing away. Each names the plans of its own side
    (verifying on that board) that no longer verify after it -- what answering costs them (the dilemma)."""
    b0 = _to_move(b, plan.side)
    before, _ = _frames(b0)
    bb = b0
    for m in plan.moves[:-1]:
        bb = bb.copy(stack=False); bb.push(m); bb = passed(bb)
    nb = bb.copy(stack=False); nb.push(plan.moves[-1])
    stop = _stops(nb, plan, len(plan.moves) - 1, before, _zone(b0, plan), True)
    theirs = plans_for(nb, not plan.side) if stop else []
    out = []
    for a in stop:
        ab = nb.copy(stack=False); ab.push(a)
        vac = [q for q in theirs if not (q.moves and q.moves[0] == a) and verify(ab, q.side, q.target, q.moves) is None]
        for q in vac:
            if not q.checked:                                     # does the given-up plan go through? (only those cost anything)
                q.through, q.stopped_by = goes_through(nb, q); q.checked = True
        out.append(Answer(a, nb.san(a), vac))
    return out


def stopping_price(b: chess.Board, plan: Plan) -> float:
    """The cheapest price of stopping the plan's final threat, computed lazily: an answer that gives up nothing, or
    only plans that do not go through, costs 0 and ends the search; a given-up plan's own AND/OR check runs only when
    every answer so far gave something up. Same number as min(price(a) for a in final_answers(b, plan))."""
    b0 = _to_move(b, plan.side)
    before, _ = _frames(b0)
    bb = b0
    for m in plan.moves[:-1]:
        bb = bb.copy(stack=False); bb.push(m); bb = passed(bb)
    nb = bb.copy(stack=False); nb.push(plan.moves[-1])
    stop = _stops(nb, plan, len(plan.moves) - 1, before, _zone(b0, plan), True)
    if not stop:
        return 0.0
    theirs = plans_for(nb, not plan.side)
    if not theirs:
        return 0.0
    vacs = []
    for a in stop:
        ab = nb.copy(stack=False); ab.push(a)
        vac = [q for q in theirs if not (q.moves and q.moves[0] == a) and verify(ab, q.side, q.target, q.moves) is None]
        if not vac:
            return 0.0                                            # a free answer
        vacs.append(vac)
    best = None
    for vac in sorted(vacs, key=lambda v: max(q.gain for q in v)):
        cost = 0.0
        for q in sorted(vac, key=lambda q: -q.gain):
            if best is not None and q.gain <= cost:
                break
            if not q.checked:
                q.through, q.stopped_by = goes_through(nb, q); q.checked = True
            if q.through:
                cost = max(cost, q.gain); break                   # sorted by gain: the first that goes through is the price
        best = cost if best is None else min(best, cost)
        if best == 0.0:
            return 0.0
    return best or 0.0


def price(ans: Answer) -> float:
    """What an answer costs its side: the most a given-up plan would have won -- counting only plans that go through;
    a plan any reply stops was not worth keeping (White's Nh5 against a rook that just steps away)."""
    return max((q.gain for q in ans.vacates if q.through), default=0.0)


def joint_value(b: chess.Board, first: chess.Move, plans: list) -> float:
    """THE DILEMMA of one move that starts several one-move plans (a double attack; Kaveh: two frames one reply cannot
    both touch): after the move, each reply of theirs is followed by what our frames win; the value is their best
    reply's outcome, i.e. min over replies of what the frames still win beyond before. Only the replies that touch a
    plan's zone, capture, check, move a target or add a defender matter; the rest are the pass."""
    b0 = _to_move(b, plans[0].side)
    before, _ = _frames(b0)
    nb = b0.copy(stack=False); nb.push(first)
    zone = set().union(*(_zone(b0, p) for p in plans))
    replies = {a for p in plans for a in _replies(nb, zone, p.target)}
    worst = min((v for _, v in [(None, max(p.gain for p in plans))]), default=0.0)   # the pass: every plan goes through
    for a in replies:
        ab = nb.copy(stack=False); ab.push(a)
        v = _frames(ab)[0] - captured_value(nb, a) - before        # what our frames win after it, less what it took (checks: the king frame)
        worst = min(worst, v)
    return max(0.0, worst)


def score_moves(b: chess.Board, moves) -> dict:
    """The tactical plan layer's credit for the side to move's candidate moves: a move that starts a plan scores the
    plan's gain when it goes through, else the cheapest price of stopping its final threat (every answer gives up
    something); a move that starts one-move plans on two or more targets scores what no single reply saves (the
    double attack); other moves 0."""
    wanted = {m.uci() for m in moves}
    out = {u: 0.0 for u in wanted}
    ps = plans_for(b, b.turn)
    by_first = {}
    for p in ps:
        if len(p.moves) == 1:
            by_first.setdefault(p.moves[0].uci(), []).append(p)
    for u, group in by_first.items():
        if u in wanted and len({p.target for p in group}) > 1:
            out[u] = max(out[u], joint_value(b, group[0].moves[0], group))
    for p in ps:
        u = p.moves[0].uci()
        if u not in wanted:
            continue
        ok, _ = goes_through(b, p)
        v = p.gain if ok else stopping_price(b, p)
        out[u] = max(out[u], v)
    return out


def _still_now(now: chess.Board, plan: Plan, before: float):
    """A one-move plan: after the reply, with us to move, do the frames still win more than before on a line through
    the target?"""
    v, ln = _frames(now)
    return (v, ln) if v >= before + GAIN and _hits(ln, plan.target) else None


def reasoning(b: chess.Board, top: int = 6) -> dict:
    """Both sides' verified plans; for each: does it go through, and the answers to its final threat with their price."""
    sides = {c: plans_for(b, c) for c in (chess.WHITE, chess.BLACK)}
    for c, ps in sides.items():
        for p in ps[:top]:
            p.through, p.stopped_by = goes_through(b, p)
            p.answers = [] if p.through else final_answers(b, p); p.checked = True
    return sides


def describe(b: chess.Board, sides: dict, top: int = 6) -> list:
    lines = []
    for c in (b.turn, not b.turn):
        name = "White" if c else "Black"
        ps = sides[c]
        if not ps:
            lines.append(f"{name}: no frame breaks within {MAX_MOVES} moves"); continue
        for p in ps[:top]:
            tp = b.piece_at(p.target)
            bb = _to_move(b, c)
            head = f"{name}: {p.san(b)} breaks the {chess.piece_name(tp.piece_type)} on {chess.square_name(p.target)} (+{p.gain:g}: " \
                   f"then {' '.join(_line_san(bb, p.moves, p.line))}) -- {'; '.join(o.text for o in p.ops)}"
            lines.append(head)
            if p.checked:
                if p.through:
                    lines.append("    it goes through: no defence stops it")
                else:
                    lines.append(f"    stopped (first found: {' '.join(p.stopped_by)}); the answers to its final threat:")
                    for a in p.answers[:8]:
                        gives = sorted({f"{chess.square_name(q.target)} (+{q.gain:g}{'' if q.through else ', stoppable'})" for q in a.vacates})
                        lines.append(f"      {a.san}" + (f" -- gives up its plans against {', '.join(gives)}" if gives else " -- costs nothing"))
                    if len(p.answers) > 8:
                        lines.append(f"      ... {len(p.answers) - 8} more")
    return lines


def _line_san(b0: chess.Board, moves: list, line: list) -> list:
    bb = b0
    for m in moves:
        bb = bb.copy(stack=False); bb.push(m); bb = passed(bb)
    out = []
    for u in line:
        m = chess.Move.from_uci(u)
        if m not in bb.legal_moves:
            break
        out.append(bb.san(m)); bb = bb.copy(stack=False); bb.push(m)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--fen", required=True); ap.add_argument("--top", type=int, default=6)
    a = ap.parse_args(argv)
    b = chess.Board(a.fen); t0 = time.perf_counter()
    sides = reasoning(b, a.top)
    for l in describe(b, sides, a.top):
        print(l)
    print(f"[frame_goals] {sum(len(v) for v in sides.values())} plans, {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
