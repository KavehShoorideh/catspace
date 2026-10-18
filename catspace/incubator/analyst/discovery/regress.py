"""GOAL REGRESSION ON QUESTIONS (ANALYST_PLAN 52.1, 52.2): a QUESTION is a future state as a partial pattern ("our queen on h2, their king on g1"), asked as a
counterfactual of layer 1 (graph.py). Its answer is a DEFECT LIST read from the rules (for a mate: no check / the checker can be taken / a flight exists / an
interposition exists; for material: not attacked / defended and the exchange gives nothing). Each defect GENERATES sub-questions (guard the square: who x-rays
it, what blocks the x-ray; remove the blocker; cover the flight). TEMPO ARITHMETIC prunes: a future state with d defects needs d prep moves of ours and the
opponent answers each, so every prep move must be FORCING (a check; for material also a capture or threat worth at least the question's stake) or the fix
is useless; a question whose defects cannot all be fixed by forcing moves dies WITH THAT REASON. A ROUTE is our move that supplies a fix and is forcing; the
guarantee of a route is never read here: binder.py verifies it over every reply.

52.11: a fix is a move that does what the fix names and introduces no new defect; the disappearance of a defect for another reason is not a fix.

Deviation from 52.2 (tempo): a state with d defects needs d prep moves and the opponent gets d replies, one after each (not d-1: the reply after the last prep
move is the one that can undo the pattern before our final move), so each of the d prep moves must be forcing. One move may serve two defects.
Deviation from 52.2 (signatures): fixes_of / routes_of also take the question and the board (a fix is read against the board it is asked on).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from catspace.incubator.analyst.discovery import graph as G

STAKE_MATE = 100
STAKE_PROMOTE = 8                             # a queen less the pawn
PROMOTE_MAX_PUSHES = 3                        # farther pawns: the mate / material stakes dominate (52.6)


@dataclass
class Defect:
    kind: str                                 # 'no check' | 'checker takeable' | 'flight' | 'interposition' | 'not attacked' | 'defended'
    sq: int | None = None                     # the square the defect is about (the checker, the flight, the target)
    by: list = field(default_factory=list)    # squares of the pieces involved (takers, interposers, defenders)
    why: str = ""

    def key(self) -> tuple:
        return (self.kind, self.sq, tuple(self.by))


@dataclass
class Route:
    move: chess.Move
    san: str
    why: str                                  # 'check', 'capture 5', 'threat 9' ...
    trap: int | None = None                   # 52.10: the square of the piece this move leaves with no safe square (the binder's second move is the capture of it)
    fork: int | None = None                   # 52.12: the square of the piece of theirs this CHECK attacks with the moved piece (a fork: two threats, one reply)


@dataclass
class Question:
    id: int
    goal: str                                 # 'mate' | 'win' | 'save' | 'guard' | 'remove' | 'cover' | 'gap'
    stake: int
    pattern: dict                             # the partial future: piece symbol, at, frm, king / target squares
    defects: list = field(default_factory=list)
    fixes: list = field(default_factory=list)       # sub-questions, each with .defect_key naming the defect it fixes
    parent: int | None = None
    status: str = "open"                      # open | routed | verified | killed | gap
    reason: str = ""
    why: list = field(default_factory=list)   # strings for the trace
    routes: list = field(default_factory=list)
    unforced: list = field(default_factory=list)    # SAN of moves that would fix but are not forcing (the tempo reason)
    defect_key: tuple | None = None
    depth: int = 0
    text: str = ""


# ---------------------------------------------------------------------------------------------------------------- the placement (the pattern as a counterfactual)

def placement_of(q: Question) -> list:
    """The pattern's piece placements [(piece, at, removed_from)] (the root pattern of a mate question; fixes inherit their root's)."""
    p = q.pattern
    if "sym" not in p:
        return []
    return [(chess.Piece.from_symbol(p["sym"]), p["at"], p.get("frm"))] + [(chess.Piece.from_symbol(sym), at, frm) for sym, at, frm in p.get("more", [])]


def future_board(board: chess.Board, placement: list, them: bool) -> chess.Board:
    """The counterfactual future: the placements made, THEIR move."""
    cf = board
    for piece, at, frm in placement:
        cf = G.with_piece(cf, piece, at, frm)
    return G.with_turn(cf, them)


# ---------------------------------------------------------------------------------------------------------------- defects read from the rules

def mate_defects(board: chess.Board, us: str | bool, placement: list) -> list:
    """What stands between the pattern and a mate, read from the rules on the counterfactual (placements made, their move). [] = they have no legal move and are
    in check: the pattern IS a mate. Order: the checker takeable, flights, interpositions."""
    me = G.colour_of(us)
    return mate_defects_on(future_board(board, placement, not me), me)


def mate_defects_on(cf: chess.Board, me: bool) -> list:
    """The defects of an already-built future `cf` (their move): [] = they are mated."""
    ksq = cf.king(not me)
    if ksq is None:
        return [Defect("no check", None, [], "their king is not on the board")]
    checkers = list(cf.checkers())
    if not checkers:
        return [Defect("no check", ksq, [], f"nothing of ours attacks {G.name(ksq)}")]
    out = []
    legal = list(cf.legal_moves)
    for c in checkers:
        for m in legal:
            if m.to_square == c:
                pc = cf.piece_at(m.from_square)
                out.append(Defect("checker takeable", c, [m.from_square], f"{pc.symbol().upper()} on {G.name(m.from_square)} can take the checker on {G.name(c)}"))
    for m in legal:
        if m.from_square == ksq and m.to_square not in checkers:
            out.append(Defect("flight", m.to_square, [ksq], f"the king can go to {G.name(m.to_square)}"))
    if len(checkers) == 1:
        c = checkers[0]
        between = set(chess.SquareSet(chess.between(c, ksq)))
        by_sq: dict = {}
        for m in legal:
            if m.from_square != ksq and m.to_square in between:
                by_sq.setdefault(m.to_square, []).append(m.from_square)
        for sq, frm in sorted(by_sq.items()):
            out.append(Defect("interposition", sq, frm, f"a piece can step between on {G.name(sq)}"))
    return out


def _interposers_mated(cf: chess.Board, placement: list, defect: Defect, me: bool) -> tuple:
    """52.7: for EVERY piece that can step between on defect.sq, our checker (the root's placed piece) takes it there: is the result a mate (no defects)? -> (all
    mated, the notes for the trace)."""
    at = placement[0][1]
    notes, ok = [], True
    for frm in defect.by:
        c = cf.copy(stack=False)
        pc = c.piece_at(frm)
        step = chess.Move(frm, defect.sq)
        take = chess.Move(at, defect.sq)
        if step not in c.legal_moves:
            ok = False
            continue
        c.push(step)
        if take not in c.legal_moves:
            notes.append(f"{pc.symbol().upper()} {G.name(frm)}->{G.name(defect.sq)}: our {placement[0][0].symbol().upper()} cannot take it")
            ok = False
            continue
        san = c.san(take)
        c.push(take)
        left = mate_defects_on(c, me)
        notes.append(f"{pc.symbol().upper()} {G.name(frm)}->{G.name(defect.sq)}, {san}: " + ("the future is a mate" if not left else "still " + ", ".join(d.kind for d in left)))
        ok = ok and not left
    return ok, notes


def _recapture_mates(cf: chess.Board, placement: list, taker: int, sq: int, me: bool) -> str:
    """52.7: their NON-king piece on `taker` takes the checker on `sq`; is there a recapture of ours on `sq` after which they are mated? -> its SAN or ''."""
    c = cf.copy(stack=False)
    m = chess.Move(taker, sq)
    if m not in c.legal_moves:
        return ""
    c.push(m)
    for r in list(c.legal_moves):
        if r.to_square == sq:
            san = c.san(r)
            d = c.copy(stack=False)
            d.push(r)
            if not mate_defects_on(d, me):
                return san
    return ""


def material_defects(board: chess.Board, us: str | bool, target_sq: int) -> list:
    """Why we do not win the piece on `target_sq` now: [] when the exchange on it already wins for us (`see` from our side > 0); otherwise 'not attacked' (no
    piece of ours attacks it) or 'defended' (attacked, but it is guarded and the exchange gives nothing)."""
    me = G.colour_of(us)
    if G.see_for(board, target_sq, me) > 0:
        return []
    if not G.attackers_of(board, target_sq, me):
        return [Defect("not attacked", target_sq, [], f"nothing of ours attacks {G.name(target_sq)}")]
    d = G.defenders_of(board, target_sq)
    return [Defect("defended", target_sq, d, f"attacked, guarded {len(d)}x, the exchange gives nothing")]


# ---------------------------------------------------------------------------------------------------------------- forcing, routes

def threat_value(board_after: chess.Board, us: str | bool) -> int:
    """After our move (their turn) and a pass: the most we would win by the exchange on a piece of theirs (the Rust `see`, us on move)."""
    me = G.colour_of(us)
    c = G.with_turn(board_after, me)
    best = 0
    for s in chess.SQUARES:
        p = c.piece_at(s)
        if p and p.color != me and p.piece_type != chess.KING and c.is_attacked_by(me, s):
            best = max(best, G.see(c, s))
    return best


def forcing_why(board: chess.Board, m: chess.Move, stake: int, us: str | bool) -> str | None:
    """A move is FORCING when it is a check, or a capture / threat worth at least the stake of the question (a mate stakes 100: only a check is). -> why or None."""
    if board.gives_check(m):
        return "check"
    if stake >= STAKE_MATE:
        return None
    if board.is_capture(m):
        v = 1 if board.is_en_passant(m) else G.worth(board.piece_at(m.to_square))
        if v >= stake:
            return f"capture {v}"
    c = board.copy(stack=False)
    c.push(m)
    t = threat_value(c, us)
    return f"threat {t}" if t >= stake else None


def _moves(board: chess.Board, us: str | bool, skip_from: int | None = None) -> list:
    b = G.with_turn(board, G.colour_of(us))
    return [m for m in b.legal_moves if m.from_square != skip_from and not b.is_castling(m)]


def _goal_done(q: Question, cf: chess.Board, me: bool) -> bool:
    """52.11 (a): the POSITIVE condition of the fix's goal on the counterfactual `cf` (placements made, their move, after our prep move)."""
    p = q.pattern
    if q.goal == "cover":
        return cf.is_attacked_by(me, p["target"])
    if q.goal == "guard":                      # the checker's square is attacked by another piece of ours
        return any(a != p.get("frm") and a != p["target"] for a in cf.attackers(me, p["target"]))
    if q.goal == "remove" and p.get("blocker") is not None:     # the named blocker is gone from its square (it stepped away)
        pc = cf.piece_at(p["blocker"])
        return pc is None or pc.color != me
    return True


def _still(q: Question, board: chess.Board, us: str | bool, m: chess.Move, root: Question) -> bool:
    """Is the fix `q` NOT done by m? A fix is a move that does what the fix names and introduces no new defect; the disappearance of a defect for another
    reason is not a fix (52.11). For a mate root, on the counterfactual after m: (a) the goal's positive condition holds (cover: a piece of ours attacks the
    flight square; guard: the checker's square is attacked by another piece of ours; remove: the blocker has left) and (b) every defect after m was a defect
    before m, less the fixed one; 'no check' is never allowed."""
    c = board.copy(stack=False)
    c.push(m)
    me = G.colour_of(us)
    if root.goal == "mate":
        pl = placement_of(root)
        after = {d.key()[:2] for d in mate_defects(c, me, pl)}
        before = {d.key()[:2] for d in mate_defects(board, me, pl)} - {q.defect_key[:2]}
        if q.defect_key[:2] in after or any(k[0] == "no check" for k in after) or not after <= before:
            return True
        return not _goal_done(q, future_board(c, pl, not me), me)
    if root.goal == "win":
        if c.is_check():                       # our move checks: the board cannot be turned back to us (their king would be in check on our move), so the defect
            return not G.attackers_of(c, root.pattern["target"], me)       # is read as 'attacked or not'; the binder alone decides what the route is worth
        return bool(material_defects(G.with_turn(c, me), me, root.pattern["target"]))
    return True


def routes_of(q: Question, board: chess.Board, us: str | bool, root: Question | None = None) -> list:
    """Our moves that supply the fix `q` names AND are forcing, each re-read on the counterfactual after the move (the defect gone). 'guard' / 'cover' / 'attack':
    any move but the pattern piece's own and the blockers'; 'remove': the blocker's own moves. Moves that fix without forcing go to q.unforced."""
    root = root or q
    skip = root.pattern.get("frm")
    out, unforced = [], []
    for m in _moves(board, us, skip):
        is_blocker_move = m.from_square == q.pattern.get("blocker")
        if q.goal == "remove" and not is_blocker_move:
            continue
        if q.goal in ("guard", "cover") and m.from_square in root.pattern.get("blockers", ()):
            continue
        if _still(q, board, us, m, root):
            continue
        why = forcing_why(board, m, q.stake, us)
        if why:
            out.append(Route(m, board.san(m), why))
        else:
            unforced.append(board.san(m))
    q.unforced = unforced
    return out


# ---------------------------------------------------------------------------------------------------------------- fixes (sub-questions)

_ids = [0]


def new_question(goal: str, stake: int, pattern: dict, parent: Question | None = None, **kw) -> Question:
    _ids[0] += 1
    return Question(id=_ids[0], goal=goal, stake=stake, pattern=pattern, parent=parent.id if parent else None, depth=parent.depth + 1 if parent else 0, **kw)


def reset_ids() -> None:
    _ids[0] = 0


def fixes_of(defect: Defect, q: Question, board: chess.Board, us: str | bool) -> list:
    """The sub-questions a defect generates, with their routes read against `board` (the real position, us to move)."""
    me = G.colour_of(us)
    root = q
    out = []
    if q.goal == "mate":
        cf = future_board(board, placement_of(q), not me)
        if defect.kind == "checker takeable":
            for taker in defect.by:
                if cf.piece_at(taker).piece_type == chess.KING:
                    xs = G.xrays_to(cf, defect.sq, me)
                    for ps, blockers in xs:
                        for b in blockers:
                            root.pattern.setdefault("blockers", []).append(b)
                    g = new_question("guard", q.stake, dict(q.pattern, target=defect.sq), q, defect_key=defect.key(),
                                     text=f"guard {G.name(defect.sq)} (their king takes it: we must attack it)")
                    g.routes = routes_of(g, board, us, root)
                    for ps, blockers in xs:
                        pc = cf.piece_at(ps)
                        own = ", ".join(f"{'our' if cf.piece_at(b).color == me else 'their'} {cf.piece_at(b).symbol().upper()} on {G.name(b)}" for b in blockers)
                        g.why.append(f"{pc.symbol().upper()} on {G.name(ps)} x-rays {G.name(defect.sq)} through {own}")
                    for ps, blockers in xs:
                        for b in blockers:
                            blk = cf.piece_at(b)
                            if blk.color != me:
                                continue                     # their blocker would have to be captured: a capture is a route of its own (not in this slice)
                            r = new_question("remove", q.stake, dict(q.pattern, target=defect.sq, blocker=b, xray=ps), g, defect_key=defect.key(),
                                             text=f"remove {G.name(b)} ({blk.symbol().upper()}) with tempo")
                            r.routes = routes_of(r, board, us, root)
                            g.fixes.append(r)
                    out.append(g)
                else:
                    t = new_question("remove", q.stake, dict(q.pattern, target=defect.sq), q, defect_key=defect.key(),
                                     text=f"take or pin the {cf.piece_at(taker).symbol().upper()} on {G.name(taker)} that can take the checker")
                    t.routes = [Route(m, board.san(m), "capture") for m in _moves(board, us, root.pattern.get("frm"))
                                if m.to_square == taker and board.gives_check(m)]
                    rc = _recapture_mates(cf, placement_of(q), taker, defect.sq, me)
                    if rc and q.pattern.get("move"):             # 52.7: the taker takes the checker and our recapture is the mate: the root's own move is the route
                        mv = chess.Move.from_uci(q.pattern["move"])
                        t.routes.append(Route(mv, board.san(mv), f"recapture on {G.name(defect.sq)} with check"))
                        t.why.append(f"the {cf.piece_at(taker).symbol().upper()} on {G.name(taker)} takes the checker on {G.name(defect.sq)} and we recapture with check: {rc} -- the future is a mate")
                    out.append(t)
        elif defect.kind == "flight":
            c = new_question("cover", q.stake, dict(q.pattern, target=defect.sq), q, defect_key=defect.key(), text=f"cover the flight {G.name(defect.sq)}")
            c.routes = routes_of(c, board, us, root)
            out.append(c)
        elif defect.kind == "interposition":
            # 52.7: the interposer costs THEM the interposer: our checker takes it, with check again, and the future may be a mate. Forcing (a check); the
            # route is the root's own move, verified by the binder over every reply.
            t = new_question("remove", q.stake, dict(q.pattern, target=defect.sq), q, defect_key=defect.key(),
                             text=f"take the interposer on {G.name(defect.sq)} with check")
            ok, notes = _interposers_mated(cf, placement_of(q), defect, me)
            t.why += notes
            if ok and q.pattern.get("move"):
                mv = chess.Move.from_uci(q.pattern["move"])
                t.routes = [Route(mv, board.san(mv), f"takes the interposer on {G.name(defect.sq)} with check: the future is a mate")]
            out.append(t)
        else:
            out.append(new_question("gap", q.stake, dict(q.pattern), q, defect_key=defect.key(), status="gap",
                                    reason=f"no fix for '{defect.kind}' in this slice", text=defect.kind))
    elif q.goal == "win":
        if defect.kind == "not attacked":
            a = new_question("guard", q.stake, dict(q.pattern), q, defect_key=defect.key(), text=f"attack {G.name(defect.sq)} with tempo")
            a.goal = "cover"
            a.routes = routes_of(a, board, us, root)
            out.append(a)
        else:
            for d in defect.by:
                r = new_question("remove", q.stake, dict(q.pattern, blocker=d), q, defect_key=defect.key(), text=f"remove the defender on {G.name(d)} with tempo")
                r.routes = routes_of(r, board, us, root)
                out.append(r)
        out += trap_fix(q, defect, board, us)
    for f in out:
        q.fixes.append(f)
    return out


def trap_routes(board: chess.Board, us: str | bool, target: int) -> list:
    """52.10 / 52.12: our QUIET moves after which the piece of theirs on `target` has no safe square (`(count (mobility T))` = 0 on the board after the move, read
    with the piece's owner as the mover and NO CHECK on the board); the capture of the piece itself is not a trap. A check is never a trap (with their king in
    check every reply is forced, so every check would "trap" every piece: 02XyH, 01w7J, 08pSj, 0RhU6, 0DoFB): see `fork_routes`. Each carries `trap` = target."""
    me = G.colour_of(us)
    out = []
    for m in _moves(board, us):
        if m.to_square == target or board.gives_check(m):
            continue
        c = board.copy(stack=False)
        c.push(m)
        assert not c.is_check()                   # mobility is read with no check on the board
        if c.piece_at(target) is None or c.piece_at(target).color == me or G.mobility_of(c, target) != 0:
            continue
        how = "also attacked" if c.is_attacked_by(me, target) else "quiet"
        out.append(Route(m, board.san(m), f"traps the {c.piece_at(target).symbol().upper()} on {G.name(target)}: no safe square, {how}", trap=target))
    return out


def fork_routes(board: chess.Board, us: str | bool, target: int) -> list:
    """52.12: our CHECKS after which the moved piece also attacks the piece of theirs on `target` (Lean `forkOf`): two threats, one reply. Each carries `fork` = target."""
    out = []
    for m in _moves(board, us):
        if m.to_square == target or not board.gives_check(m):
            continue
        c = board.copy(stack=False)
        c.push(m)
        pc = c.piece_at(target)
        if pc is None or pc.color == c.piece_at(m.to_square).color or target not in c.attacks(m.to_square):
            continue
        out.append(Route(m, board.san(m), f"fork: check and attack the {pc.symbol().upper()} on {G.name(target)}", fork=target))
    return out


def trap_fix(q: Question, defect: Defect, board: chess.Board, us: str | bool) -> list:
    """52.10 / 52.12: the fixes `take its squares` (a piece that can run has safe squares now; the QUIET moves after which it has none) and `fork` (a check that
    also attacks it). A quiet trap costs a tempo (they get a free reply) unless the piece is also attacked; the binder reads both over every reply."""
    target = q.pattern["target"]
    pc = board.piece_at(target)
    if pc is None or pc.piece_type == chess.KING:
        return []
    out = []
    n = G.mobility_of(G.with_turn(board, pc.color), target)
    routes = trap_routes(board, us, target) if n else []
    if routes:
        t = new_question("cover", q.stake, dict(q.pattern), q, defect_key=defect.key(),
                         text=f"take its squares: the {pc.symbol().upper()} on {G.name(target)} can run (safe squares: {n}), the quiet moves after which it has none")
        t.routes = routes
        if any(r.why.endswith("quiet") for r in routes):
            t.why.append("a quiet trap costs a tempo (they get a free reply) unless the piece is also attacked; the depth-two schema reads it over every reply")
        out.append(t)
    forks = fork_routes(board, us, target)
    if forks:
        f = new_question("cover", q.stake, dict(q.pattern), q, defect_key=defect.key(),
                         text=f"fork: check and attack the {pc.symbol().upper()} on {G.name(target)}, one reply")
        f.routes = forks
        out.append(f)
    return out


def all_routes(q: Question) -> list:
    """The routes of a fix question and of its sub-questions."""
    out = list(q.routes)
    for f in q.fixes:
        out += all_routes(f)
    return out


def tempo_ok(q: Question) -> bool:
    """d defects need d prep moves; the opponent answers each, so each defect needs a fix with a FORCING route (a question with no defect is a one-move
    pattern: fine). False carries its reason in q.reason."""
    if not q.defects:
        return True
    dead = []
    for d in q.defects:
        fx = [f for f in q.fixes if f.defect_key == d.key()]
        if not any(all_routes(f) for f in fx):
            quiet = sorted({m for f in fx for m in _unforced_all(f)})
            dead.append(f"{d.kind}{' ' + G.name(d.sq) if d.sq is not None else ''}: "
                        + (f"only quiet fixes ({', '.join(quiet[:4])}), the opponent gets a free reply" if quiet else "no fix at all"))
    if dead:
        q.reason = "tempo: " + "; ".join(dead)
        return False
    return True


def _unforced_all(q: Question) -> list:
    out = list(q.unforced)
    for f in q.fixes:
        out += _unforced_all(f)
    return out


# ---------------------------------------------------------------------------------------------------------------- the opponent's view of a route

def opponent_defects(board: chess.Board, route: chess.Move, us: str | bool) -> list:
    """The opponent's view of the future after our route (Kaveh: their reply is a defect of OUR future): their SAFE checks (the checker cannot be taken by us)
    that attack a piece of ours we do not guard, or worth more than the checker. -> ['Qc4+ forks Be4', ...]."""
    me = G.colour_of(us)
    c = board.copy(stack=False)
    c.push(route)
    out = []
    for r in list(c.legal_moves):
        if not c.gives_check(r):
            continue
        d = c.copy(stack=False)
        d.push(r)
        sq = r.to_square
        if d.is_attacked_by(me, sq):
            continue                                   # we can take the checker: not a fork
        hit = []
        for t in d.attacks(sq):
            p = d.piece_at(t)
            if p and p.color == me and p.piece_type != chess.KING and (not d.is_attacked_by(me, t) or G.worth(p) > G.worth(d.piece_at(sq))):
                hit.append(f"{p.symbol().upper()}{G.name(t)}")
        if hit:
            out.append(f"{c.san(r)} forks {' '.join(hit)}")
    return out


# ---------------------------------------------------------------------------------------------------------------- the promotion goal (52.6)

def pawn_path(board: chess.Board, sq: int, pushes: int) -> list:
    """The squares in front of the pawn on `sq`, nearest first, the promotion square last (`ahead`, named)."""
    fwd = 1 if board.piece_at(sq).color == chess.WHITE else -1
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return [chess.square(f, r + fwd * i) for i in range(1, pushes + 1)]


def promote_defects(board: chess.Board, sq: int) -> tuple:
    """What stands between the pawn on `sq` (any colour) and a queen, read with the Rust words (`pushes`, `ahead`, `promo`, `attackers`, `defenders`, `dist`)
    from its owner's side: [Defect] and the facts {pushes, path, promo, king_dist, king_sq}. Kinds: 'blocked' (a piece on the path), 'path attacked' (a path
    square attacked by them more than we guard it), 'promo square attacked' (the same on the promotion square), 'stopper' (a piece of theirs that reaches the
    promotion square or the path within the pawn's pushes: the king by `dist`, a slider / knight by its static-board `reach`). No defect = unstoppable by the
    tempo arithmetic (the owner moves first; the other side's pieces move for free while the pawn runs)."""
    pc = board.piece_at(sq)
    own = pc.color
    n = G.word(board, ("pushes", G.name(sq)), own)
    path = pawn_path(board, sq, n)
    promo = path[-1]
    facts = dict(pushes=n, path=path, promo=promo, king_dist=None, king_sq=board.king(not own))
    out = []
    ahead_ours = G.word(board, ("count", ("and", ("ahead", G.name(sq)), "ours")), own)
    ahead_theirs = G.word(board, ("count", ("and", ("ahead", G.name(sq)), "theirs")), own)
    if ahead_ours + ahead_theirs:
        for t in path:
            b_ = board.piece_at(t)
            if b_ is not None:
                out.append(Defect("blocked", t, [t], f"{'our' if b_.color == own else 'their'} {b_.symbol().upper()} stands on {G.name(t)}"))
    for t in path:
        a = G.word(board, ("defenders", G.name(t)), own)            # theirs attacking t
        d = G.word(board, ("attackers", G.name(t)), own)            # ours attacking t
        if a > d:
            kind = "promo square attacked" if t == promo else "path attacked"
            by = [x for x in G.attackers_of(board, t, not own)]
            out.append(Defect(kind, t, by, f"{G.name(t)} is attacked by them {a}x and guarded by us {d}x"))
    ek = facts["king_sq"]
    if ek is not None:                                  # a king standing on the path is the blocker (already a defect), not a stopper
        kd = G.word(board, ("dist", "L", ("promo", G.name(sq))), own)
        facts["king_dist"] = kd
        if kd <= n and ek not in path:
            out.append(Defect("stopper", promo, [ek], f"their K on {G.name(ek)} reaches {G.name(promo)} in {kd}, the pawn needs {n}"))
    for s in chess.SQUARES:
        x = board.piece_at(s)
        if x is None or x.color == own or x.piece_type in (chess.KING, chess.PAWN):
            continue
        k = G.piece_moves_to(board, s, set(path), n)
        if k is not None:
            out.append(Defect("stopper", promo, [s], f"their {x.symbol().upper()} on {G.name(s)} reaches the path in {k} (static board), the pawn needs {n}"))
    return out, facts


def promote_fixes(board: chess.Board, sq: int, defect: Defect, q: "Question") -> "Question":
    """The sub-question a promotion defect generates, with its forcing routes (moves of the pawn's owner only; `board` has the owner on move)."""
    own = board.piece_at(sq).color
    push = chess.Move(sq, pawn_path(board, sq, 1)[0])
    if defect.kind == "blocked":
        f = new_question("remove", q.stake, dict(q.pattern, blocker=defect.sq), q, defect_key=defect.key(), text=f"capture or deflect the blocker on {G.name(defect.sq)}")
        f.routes = [Route(m, board.san(m), f"capture {G.worth(board.piece_at(defect.sq))}") for m in board.legal_moves if m.to_square == defect.sq]
    elif defect.kind in ("path attacked", "promo square attacked"):
        f = new_question("guard", q.stake, dict(q.pattern, target=defect.sq), q, defect_key=defect.key(), text=f"guard {G.name(defect.sq)} or take its attacker (or push past)")
        f.routes = [Route(m, board.san(m), f"capture {G.worth(board.piece_at(m.to_square))}") for m in board.legal_moves if m.to_square in defect.by]
    else:
        f = new_question("cover", q.stake, dict(q.pattern, target=defect.by[0]),
                         q, defect_key=defect.key(), text=f"cut the stopper on {G.name(defect.by[0])} (take it) or push with a check")
        f.routes = [Route(m, board.san(m), f"capture {G.worth(board.piece_at(m.to_square))}") for m in board.legal_moves if m.to_square == defect.by[0]]
        if push in board.legal_moves and board.gives_check(push):
            f.routes.append(Route(push, board.san(push), "push with check"))
    q.fixes.append(f)
    return f
