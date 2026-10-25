"""PLAN 62.1 / 62.2: THE PASSER RACE WORD AND THE FRAME'S SOUNDNESS CONDITION -- the Python mirror of `lean/ideas/Ideas/Race.lean`.

The case (RoR8t, `Kg4 Rf4 f5 h3 v Kg1 Rb8 a2 b3`): the 1-ply stopper defect says "their K on g1 reaches h1 in 1, the pawn needs 2" -- true by its definition and too weak. A king
that must stand on h1 is not a stopper once OUR king covers h2 and a SECOND passer deflects it.

The word, over a board with the owner of the passer to move and not in check, their pieces a king and pawns (the gate; a rook of theirs can check, and the Lean `race_sound`
holds for the stopper analysis `hheld`, not for checks):

* THE STAGE TEST (one passer A, `n` pushes, path `y1..yn`). Their king, `j` moves into the race after our j-th push, STOPS it iff it can capture the pawn on `y_j` (`kdist(K, y_j) <= j`
  and `y_j` is not defended by a piece of ours that stays) or stand on `y_{j+1}` (`kdist(K, y_{j+1}) <= j`, not DENIED). DENIED = a square a king cannot stand on:
  next to our king, attacked by a pawn or a piece of ours (`Race.no_stand_adjacent`, `no_stand_pawn`, `no_stand_attacked`: a legal move never leaves the mover's king
  attacked). Held by no stage: the word binds with `ours = n`.
* THE DEFLECTION (A is held, only a blockade holds it; a second passer B with `nB` pushes). We push A to its 7th rank (`nA - 1` moves), then B. From then on their king must stay
  on a HOLD SQUARE of A (`Race.holdSet`: on the promotion square, or next to it while it is undefended; not denied -- A on its 7th now denies the squares it attacks), else A
  promotes. B walks when no hold square of A is on or next to B's path (`Race.overloaded` for the promotion squares, plus the path squares a king could stand on). `ours = (nA - 1) + nB`.
* THEIRS. Their pawns need more than `ours - 1` moves to reach any bad square (the path, B's square, the squares that check our king, their promotion rank: the prototype's `pawnsFar`), so
  `ours <= theirs` (we have the move: Lean `raceBinds`).
* NO STALEMATE (`hlive`): their side keeps a reply: a pawn move that stays free for `ours` plies.

The frame (62.2): the word READS the passers and their paths, both kings, the pieces that deny, and their pawns' distances. Every other piece is EXCLUDED, and an exclusion is sound
only when (a) it cannot touch the read set within `ours` tempi (`Race.far_stays_off`: kings by `kdist`, pawns by `pdist`, the rest by `lb`) and (b) the claim holds against a pass
(`Race.frame_sound_pass`): the word counts only OUR tempi, never theirs, and the one-ply recheck below re-reads it after every legal reply to the first push.

`python -m catspace.incubator.analyst.discovery.race --validate N` checks every claim against an exhaustive race search (the bridge).
"""
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field

import chess

RACE = os.environ.get("CATSPACE_RACE", "1") != "0"     # plan 62.1 (ON; `analyse --no-race` sets CATSPACE_RACE=0, inherited by the workers: only a comparison condition)
DENY = os.environ.get("CATSPACE_DENY", "1") != "0"     # plan 62.1b (ON; `analyse --no-deny` is only a comparison condition)
LIFT = os.environ.get("CATSPACE_LIFT", "1") != "0"       # plan 62.1c (ON; `analyse --no-lift` is only a comparison condition): the stopper verdict judges only their KING when they have other pieces
FRAME = os.environ.get("CATSPACE_FRAME", "1") != "0"   # plan 62.2 (ON; `analyse --no-frame`): the frame's condition (b) gate; off = the word is served without the one-ply recheck
STATS = {"asked": 0, "bound": 0, "single": 0, "overload": 0, "frame_refused": 0}


# ---------------------------------------------------------------------------------------------------------------- the Lean words (Race.lean)

def kdist(a: int, b: int) -> int:
    return chess.square_distance(a, b)


def promo_tempi(pushes: int, delay: int = 0) -> int:
    """Race.promoTempi."""
    return pushes + delay


def race_binds(ours: int, theirs: int) -> bool:
    """Race.raceBinds: with the move, we promote on our `ours`-th move, they on their `theirs`-th, which comes after ours."""
    return ours <= theirs


def holds(x: int, s: int) -> bool:
    """Race.holds: a king on `s` stops a pawn about to promote on `x` (on it, or next to it)."""
    return kdist(s, x) <= 1


def hold_set(denied: int, x: int) -> int:
    """Race.holdSet as a bitboard: the squares within one of `x` that are not in the `denied` mask."""
    return chess.BB_KING_ATTACKS[x] & ~denied | (chess.BB_SQUARES[x] & ~denied)


def overloaded(denied: int, a: int, b: int) -> bool:
    """Race.overloaded: no square holds both promotion squares."""
    return not (hold_set(denied, a) & hold_set(denied, b))


def lb(s: int, t: int) -> int:
    """Race.lb: the lower bound on the moves a rook, bishop, queen or knight needs from `s` to `t`: 0, 1 (a line or a knight's move), else 2."""
    if s == t:
        return 0
    f, r = chess.square_file(s) - chess.square_file(t), chess.square_rank(s) - chess.square_rank(t)
    if f == 0 or r == 0 or abs(f) == abs(r) or sorted((abs(f), abs(r))) == [1, 2]:
        return 1
    return 2


def pdist(color: bool, s: int, t: int) -> float:
    """Passer2.pdist: a lower bound on the moves a pawn of `color` on `s` needs to reach `t` (inf: never). The double step is allowed from its start rank."""
    fwd = 1 if color == chess.WHITE else -1
    start = 1 if color == chess.WHITE else 6
    ahead = (chess.square_rank(t) - chess.square_rank(s)) * fwd
    df = abs(chess.square_file(t) - chess.square_file(s))
    if ahead < 0 or df > ahead:
        return float("inf")
    return max(df, ahead - (1 if chess.square_rank(s) == start else 0))


# ---------------------------------------------------------------------------------------------------------------- the word

@dataclass
class Race:
    pawn: int                                   # the passer the claim is named for (A)
    second: int | None                          # the deflecting passer (B), or None
    ours: int                                   # promoTempi: our moves to the first promotion
    theirs: float                               # their quickest pawn (inf: none)
    plan: list = field(default_factory=list)    # our moves, UCI, in order (the line)
    text: str = ""                              # the sentence for the trace
    frame: str = ""                             # the frame sentence (read set, excluded pieces and why)
    first: chess.Move | None = None

    def key(self) -> tuple:
        return (self.pawn, self.second, self.ours)


def _path(me: bool, sq: int, n: int) -> list:
    fwd = 1 if me else -1
    f, r = chess.square_file(sq), chess.square_rank(sq)
    return [chess.square(f, r + fwd * i) for i in range(1, n + 1)]


def _pushes(me: bool, sq: int) -> int:
    return abs((7 if me else 0) - chess.square_rank(sq))


def _name(sq: int) -> str:
    return chess.square_name(sq)


def _pawn_attacks(me: bool, sq: int) -> int:
    return chess.BB_PAWN_ATTACKS[me][sq]


def _stable_attacks(b: chess.Board, me: bool, ek: int, ok: int, T: int, plan_pawns: dict) -> int:
    """The squares we attack for the whole race: our king's, the pawns outside the plan, and the pieces their king cannot even reach in time (`kdist(K, p) > T`), the board
    read with their king taken off (a king may step along a line it blocks) and with the plan pawns at EVERY square they stand on (a slider is blocked by them there): the
    intersection over those boards. Used for `denied` (a king cannot stand on it) and for `defended` (a king cannot take on it)."""
    att = chess.BB_KING_ATTACKS[ok]
    outside = [s for s in chess.SquareSet(b.occupied_co[me]) if s != ok and s not in plan_pawns and not (b.pawns & chess.BB_SQUARES[s])]
    pawns_out = [s for s in chess.SquareSet(b.pawns & b.occupied_co[me]) if s not in plan_pawns]
    for s in pawns_out:
        if kdist(ek, s) > T:
            att |= b.attacks_mask(s)
    if outside:
        boards = []
        configs = [dict()]                                      # {pawn start: square it stands on}
        for p0, walk in plan_pawns.items():
            configs = [{**c, p0: sq} for c in configs for sq in [p0] + walk]
        base = b.copy(stack=False)
        base.remove_piece_at(ek)
        inter = None
        for c in configs[:64]:
            d = base.copy(stack=False)
            moved = {p0: d.piece_at(p0) for p0 in plan_pawns}
            for p0 in plan_pawns:
                d.remove_piece_at(p0)
            for p0, sq in c.items():
                d.set_piece_at(sq, moved[p0])
            m = 0
            for s in outside:
                if kdist(ek, s) > T:
                    m |= d.attacks_mask(s)
            inter = m if inter is None else inter & m
        att |= inter or 0
    return att


def _their_pawns_ok(b: chess.Board, me: bool, T: int, bad: set) -> tuple:
    """The prototype's `pawnsFar`: every pawn of theirs needs more than T - 1 moves to reach a bad square (or the bad square is off its file and it cannot leave the file in time: the
    cone form). -> (ok, the quickest pawn's moves to its promotion rank (inf when none), a note)."""
    them = not me
    quick = float("inf")
    for e in chess.SquareSet(b.pawns & b.occupied_co[them]):
        n = abs((0 if them == chess.BLACK else 7) - chess.square_rank(e))
        quick = min(quick, n)
    capturable = set(chess.SquareSet(b.occupied_co[me])) | set(bad)

    def stuck(e: int) -> bool:
        return all(pdist(them, e, d) > T - 1 for d in capturable if abs(chess.square_file(d) - chess.square_file(e)) == 1)

    for e in chess.SquareSet(b.pawns & b.occupied_co[them]):
        for t in bad:
            if not (pdist(them, e, t) > T - 1 or (chess.square_file(t) != chess.square_file(e) and stuck(e))):
                return False, quick, f"their P on {_name(e)} reaches {_name(t)} in {int(pdist(them, e, t))}"
    return True, quick, ""


def _spare_moves(b: chess.Board, me: bool, T: int) -> bool:
    """`hlive`: they keep a reply for T plies: the free squares ahead of their unpinned pawns (not blocked now; our passers move off the files they use or stay put, so a blocked
    square is counted as not free) number at least T, or the king has room (checked on the board: some neighbour of the king is not attacked and not ours)."""
    them = not me
    free = 0
    for e in chess.SquareSet(b.pawns & b.occupied_co[them]):
        if b.is_pinned(them, e):
            continue
        fwd = -1 if them == chess.BLACK else 1
        f, r = chess.square_file(e), chess.square_rank(e)
        k = 1
        while 0 <= r + fwd * k <= 7 and b.piece_at(chess.square(f, r + fwd * k)) is None and k <= 6:
            free += 1
            k += 1
    return free >= T


def _stage_held(me: bool, ek: int, path: list, rest: int, denied: int, defended: int, upto: int, block_upto: int | None = None) -> str | None:
    """The stage test for a passer standing on `rest` with path `path`: its j-th push lands on y_j (A stood on y_{j-1}, `rest` for j = 1, while their king made j - 1 moves). Their king HOLDS it at
    stage j when it can (a) take the pawn on y_j (or the queen that y_n becomes): stand, after j - 1 moves, on a square next to y_j that is not DENIED -- the squares our king and our pieces
    attack, and the squares the passer itself attacks from y_{j-1} (the two next to y_j on its rank: Race.no_stand_pawn) -- while y_j is not defended; or (b) stand on y_{j+1} after j moves
    (j + 1 <= block_upto), not denied. -> why it is held, or None."""
    n = len(path)
    for j in range(1, upto + 1):
        yj = path[j - 1]
        prev = rest if j == 1 else path[j - 2]
        stand_denied = denied | chess.BB_PAWN_ATTACKS[me][prev]
        if not (defended & chess.BB_SQUARES[yj]):
            for s in chess.SquareSet(chess.BB_KING_ATTACKS[yj]):
                if kdist(ek, s) <= j - 1 and not (stand_denied & chess.BB_SQUARES[s]):
                    return f"their K takes on {_name(yj)} after push {j} (from {_name(s)}, {kdist(ek, s)} away; {_name(yj)} is not guarded)"
        if j < n and j + 1 <= (n if block_upto is None else block_upto):
            nx = path[j]
            if kdist(ek, nx) <= j and not (denied & chess.BB_SQUARES[nx]):
                return f"their K stands on {_name(nx)} before push {j + 1} (it is {kdist(ek, nx)} away, {_name(nx)} is not covered)"
    return None


def _king_holds(board: chess.Board, me: bool, sq: int, with_second: bool) -> tuple | None:
    ek, ok = board.king(not me), board.king(me)
    n = _pushes(me, sq)
    if n < 1 or n > 6:
        return None
    path = _path(me, sq, n)
    if any(board.piece_at(t) is not None for t in path):
        return None
    stable = _stable_attacks(board, me, ek, ok, n, {sq: path})
    why = _stage_held(me, ek, path, sq, stable, stable, n)
    if why is None:
        return False, "their king cannot stop it: no stage of the walk is reached in time (squares it cannot stand on denied)"
    if with_second:
        d = _deflect(board, me, sq, path, n, ek, ok, why, verdict_only=True)
        if d:
            return False, f"{d}; the king held it alone: {why}"
    return True, why


def king_verdict(board: chess.Board, sq: int) -> tuple | None:
    """62.1c: THEIR KING AS A STOPPER of our pawn on `sq` (the owner to move), judged by the proved analysis only (Race.deny_sound, race_sound restricted to the king): the stage test with the squares it cannot stand
    and survive on denied, then the deflection by a second passer (this pawn as the one that walks, or as the one that rests). -> (held, why): held=False = the king is no stopper (the verdict may not kill the
    question for it); None = not applicable (not our pawn, path not clear, in check). Their OTHER pieces are not read here at all: they are replies, and the chase reads every reply (Race.lifted_is_gap)."""
    me = board.turn
    pc = board.piece_at(sq)
    if pc is None or pc.piece_type != chess.PAWN or pc.color != me or board.is_check():
        return None
    if board.king(not me) is None or board.king(me) is None:
        return None
    first = _king_holds(board, me, sq, True)
    if first is None or not first[0]:
        return first
    for q in chess.SquareSet(board.pawns & board.occupied_co[me]):
        if q == sq:
            continue
        o = _king_holds(board, me, q, True)
        if o is not None and not o[0]:
            return False, f"the passer on {_name(q)} is not held by the king ({o[1]}); the king that holds {_name(sq)} cannot also stop it"
    return first


def support_moves(board: chess.Board, me: bool, sq: int, legal: list) -> list:
    """62.1c: our moves (not the pawn's own) after which their king is NO LONGER a stopper of the pawn on `sq` by `king_verdict` (we have the move again: the pass reading): a cover (Kg3 takes h2 and g2 from
    the king), a second passer's support. Each is a ROUTE of the promote question; the chase reads every reply to it. -> [(move, why)]."""
    before = king_verdict(board, sq)
    if before is None or not before[0]:
        return []
    out = []
    for m in legal:
        if m.from_square == sq:
            continue
        after = board.copy(stack=False)
        after.push(m)
        if after.is_check():
            continue
        after.turn = me
        after.ep_square = None
        v = king_verdict(after, sq)
        if v is not None and not v[0]:
            out.append((m, f"makes their king no stopper: {v[1]}"))
    return out


def race_claim(board: chess.Board, sq: int, allow_second: bool = True, recheck: bool = True) -> Race | None:
    """THE RACE WORD for our pawn on `sq` (the owner to move, not in check): -> a `Race` when the word binds, else None. See the module docstring for the definition and the Lean theorems it mirrors."""
    if not RACE:
        return None
    STATS["asked"] += 1
    me = board.turn
    them = not me
    pc = board.piece_at(sq)
    if pc is None or pc.piece_type != chess.PAWN or pc.color != me or board.is_check():
        return None
    if board.occupied_co[them] & (board.rooks | board.bishops | board.queens | board.knights):
        return None                                    # the gate: a rook / queen / minor piece of theirs can check, fork, or arrive; `hheld` is not read from the rules for it
    ek, ok = board.king(them), board.king(me)
    if ek is None or ok is None:
        return None
    n = _pushes(me, sq)
    if n < 1 or n > 6:
        return None
    path = _path(me, sq, n)
    if any(board.piece_at(t) is not None for t in path):
        return None
    # --- the single passer: no stage holds it
    T = n
    stable = _stable_attacks(board, me, ek, ok, T, {sq: path})
    why = _stage_held(me, ek, path, sq, stable, stable, n)
    res = None
    if why is None:
        res = _bind(board, me, sq, None, T, path, [], ek, ok, text=None, why_free="their king cannot stop it: no stage of the walk is reached in time")
        if res is not None:
            STATS["single"] += 1
    elif allow_second:
        res = _deflect(board, me, sq, path, n, ek, ok, why)
        if res is not None:
            STATS["overload"] += 1
    if res is None:
        return None
    if FRAME and recheck and not _recheck(board, res):
        STATS["frame_refused"] += 1
        return None
    STATS["bound"] += 1
    return res


def _deflect(board: chess.Board, me: bool, sq: int, path: list, n: int, ek: int, ok: int, why_held: str, verdict_only: bool = False) -> "Race | str | None":
    best = None
    for sb in chess.SquareSet(board.pawns & board.occupied_co[me]):
        if sb == sq:
            continue
        nb = _pushes(me, sb)
        if nb < 1 or nb > 6:
            continue
        pathb = _path(me, sb, nb)
        if any(board.piece_at(t) is not None and t != ek for t in pathb):             # their king on B's path is allowed: it must leave to hold A (else A promotes)
            continue
        T = (n - 1) + nb
        # the stable denial/defence over the whole plan: A walks to its 7th, B walks
        walk_a = path[:n - 1]
        stable = _stable_attacks(board, me, ek, ok, T, {sq: walk_a, sb: pathb})
        rest = path[n - 2] if n >= 2 else sq                      # A rests here while B walks
        xa = path[-1]
        fwd = 1 if me else -1
        # B's start square attacks (a pawn): denied squares while A walks; A on its resting square attacks too
        denied_a = stable | _pawn_attacks(me, rest) | _pawn_attacks(me, sb)
        # A must reach `rest` unheld: stages 1..n-1 (the K may neither take it nor stand on y_{j+1} before)
        if n >= 2 and _stage_held(me, ek, path, sq, stable, stable, n - 1, n - 1) is not None:
            continue
        # A rests on `rest` while B walks: nobody takes it there, B's square is safe while A walks
        if not (stable & chess.BB_SQUARES[rest]) and kdist(ek, rest) <= T + 1:
            continue
        if not (stable & chess.BB_SQUARES[sb]) and kdist(ek, sb) <= T + 1:
            continue
        # the hold squares of A: the promotion square (a block) and, while it is unguarded, its neighbours
        undefended_x = not (stable & chess.BB_SQUARES[xa])
        h_a = (chess.BB_SQUARES[xa] & ~denied_a) | ((chess.BB_KING_ATTACKS[xa] & ~denied_a) if undefended_x else 0)
        # B walks: no hold square of A is on B's path (a king there blocks it) or next to B's promotion square while it is unguarded; the Lean `overloaded` for the promotion squares
        xb = pathb[-1]
        if not overloaded(denied_a, xa, xb) and undefended_x:
            pass                                                    # the Lean word over the promotion squares; the path test below is the conjunct that matters
        pb_mask = 0
        for t in pathb:
            pb_mask |= chess.BB_SQUARES[t]
        if h_a & pb_mask:
            continue
        undefended_xb = not (stable & chess.BB_SQUARES[xb])
        if undefended_xb and (h_a & chess.BB_KING_ATTACKS[xb]):
            continue
        if verdict_only:
            return f"deflected: the king that holds {_name(xa)} cannot hold {_name(xb)} (second passer {_name(sb)})"
        cand = _bind(board, me, sq, sb, T, path, pathb, ek, ok, text=None, why_free=f"the king that holds {_name(xa)} cannot hold {_name(xb)}", nA=n, nB=nb,
                     why_held=why_held, h_a=h_a, xa=xa, xb=xb)
        if cand is not None and (best is None or cand.ours < best.ours):
            best = cand
    return best


def _bind(board, me, sq, sb, T, path, pathb, ek, ok, text, why_free, nA=None, nB=None, why_held=None, h_a=0, xa=None, xb=None) -> Race | None:
    """The tail of the word: their pawns are slower, they keep a reply; build the Race with its line and frame text."""
    them = not me
    bad = set(path) | {chess.square(x, 0 if them == chess.BLACK else 7) for x in range(8)}
    if sb is not None:
        bad |= set(pathb) | {sb}
    cf = chess.square_file(ok)
    for df in (-1, 1):                                 # the squares from which their pawn would check our king
        f2, r2 = cf + df, chess.square_rank(ok) + (1 if them == chess.BLACK else -1)
        if 0 <= f2 < 8 and 0 <= r2 < 8:
            bad.add(chess.square(f2, r2))
    ok_, quick, note = _their_pawns_ok(board, me, T, bad)
    if not ok_:
        return None
    if not race_binds(T, quick if quick != float("inf") else 10 ** 6):
        return None
    if not _spare_moves(board, me, T) and not _king_room(board, me, T, [(sq, list(path))] + ([(sb, list(pathb))] if sb is not None else [])):
        return None
    fwd = 1 if me else -1
    san_first = None
    if sb is None:
        first = chess.Move(sq, path[0], promotion=chess.QUEEN if len(path) == 1 else None)
        plan = [first.uci()]
        line = f"{', '.join(_name(t) for t in path[:-1])}{', ' if len(path) > 1 else ''}{_name(path[-1])}=Q"
        text = (f"we promote in {T} ({_name(sq)} to {_name(path[-1])}; {why_free}); " + (f"they need {int(quick)}" if quick != float("inf") else "they have no passer"))
    else:
        if nA >= 2:
            first = chess.Move(sq, path[0])
        else:
            first = chess.Move(sb, pathb[0], promotion=chess.QUEEN if len(pathb) == 1 else None)
        plan = [first.uci()]
        line = (f"{_name(sq)} to {_name(path[-2]) if nA >= 2 else _name(sq)}, then {_name(sb)} to {_name(pathb[-1])}=Q (the K that holds {_name(xa)} is deflected)")
        text = (f"we promote in {T} ({_name(sq)} rests on {_name(path[-2]) if nA >= 2 else _name(sq)}, {_name(sb)} walks to {_name(pathb[-1])}; the K held {_name(sq)}: {why_held}; "
                f"{why_free}); " + (f"they need {int(quick)}" if quick != float("inf") else "they have no passer"))
    r = Race(pawn=sq, second=sb, ours=T, theirs=quick, plan=plan, text=text, first=first)
    r.frame = _frame_text(board, me, r, path, pathb, ek, ok)
    return r


def _queen_rays(pq: int) -> int:
    m = 0
    for t in chess.SQUARES:
        f, r = chess.square_file(t) - chess.square_file(pq), chess.square_rank(t) - chess.square_rank(pq)
        if t != pq and (f == 0 or r == 0 or abs(f) == abs(r)):
            m |= chess.BB_SQUARES[t]
    return m


def _king_room(board: chess.Board, me: bool, T: int, plans: list) -> bool:
    """`hlive` when they have no pawn moves to spare: neither a push on the way nor the new queen may STALEMATE them (the first oracle missed it: 8/5P1k with Kh5, f8=Q is stalemate; 7k/5K2/7P
    with g5-g6; k1K5/1P6 with b5-b6). `plans` = [(start, [squares it walks, the promotion square last])] for each pawn of the plan. For every combination of places the plan pawns can stand
    (a pawn on its promotion square is a QUEEN: the empty-board rays are a bound) and every square their king can stand on in T moves: it keeps a neighbour that is not attacked by anything
    of ours and not taken by their own piece -- unless its own square is attacked (that is check, and a mate is no stalemate)."""
    them = not me
    ek = board.king(them)
    c = board.copy(stack=False)
    c.remove_piece_at(ek)
    starts = {st for st, _w in plans}
    real = 0
    for st in starts:
        c.remove_piece_at(st)                                      # the walking pawns leave the lines of our sliders: counting the attacks WITHOUT them can only over-state what we attack
    for t in chess.SquareSet(c.occupied_co[me]):
        real |= c.attacks_mask(t)
    places = [[(st, False)] + [(w, i == len(walk) - 1) for i, w in enumerate(walk)] for st, walk in plans]
    combos = [[]]
    for pl in places:
        combos = [cb + [x] for cb in combos for x in pl]
    for cb in combos:
        att = real
        for sq_, is_q in cb:
            att |= _queen_rays(sq_) if is_q else chess.BB_PAWN_ATTACKS[me][sq_]
        for x in chess.SQUARES:
            if kdist(ek, x) > T:
                continue
            occ = board.piece_at(x)
            if occ is not None and occ.color == them and x != ek:
                continue
            free = [y for y in chess.SquareSet(chess.BB_KING_ATTACKS[x]) if not (att & chess.BB_SQUARES[y]) and (board.piece_at(y) is None or board.piece_at(y).color == me)]
            if not free and not (att & chess.BB_SQUARES[x]) and kdist(x, board.king(me)) > 1:
                return False
    return True


def ok_sq(board: chess.Board, me: bool) -> int:
    return board.king(me)


def _frame_text(board: chess.Board, me: bool, r: Race, path: list, pathb: list, ek: int, ok: int) -> str:
    """The frame the word read and the pieces it excluded, with the condition that makes the exclusion sound (Race.frame_sound_pass, Race.far_stays_off)."""
    read = {r.pawn, ek, ok} | set(path)
    if r.second is not None:
        read |= {r.second} | set(pathb)
    them = not me
    pieces = [(s, board.piece_at(s)) for s in sorted(chess.SquareSet(board.occupied_co[chess.WHITE] | board.occupied_co[chess.BLACK]))]
    read_txt = " ".join(f"{p.symbol().upper()}{_name(s)}" for s, p in pieces if s in read)
    exc = [(s, p) for s, p in pieces if s not in read]
    exc_txt = " ".join(f"{p.symbol().upper()}{_name(s)}" for s, p in exc)
    far = [min([pdist(them, s, t) for t in list(path) + list(pathb)] + [99]) for s, p in exc if p.color == them and p.piece_type == chess.PAWN]
    reason = f"their pawns are far (>= {int(min(far))} moves from the paths; the word needs more than {r.ours - 1})" if far else "none of theirs is outside"
    return (f"frame: {read_txt}; excluded {exc_txt or 'none'}: {reason}; the word counts only our tempi, so a pass of theirs changes nothing (the win is forced, not zugzwang)")


def _recheck(board: chess.Board, r: Race) -> bool:
    """The frame's condition (b), one ply: after our first move EVERY legal reply of theirs leaves a position on which the word binds again with one tempo fewer (read for the pawn we pushed,
    or for the other passer). A reply after which it does not is a parry the word does not cover, and the claim is refused. One board per reply."""
    if r.ours <= 1:
        return True
    me = board.turn
    b2 = board.copy(stack=False)
    b2.push(r.first)
    if not any(True for _ in b2.legal_moves):
        return False
    watch = [r.first.to_square] + [x for x in (r.pawn, r.second) if x is not None and x != r.first.from_square]
    for rep in list(b2.legal_moves):
        d = b2.copy(stack=False)
        d.push(rep)
        if d.is_check() or d.turn != me:
            return False
        ok = False
        for s in watch:
            pc = d.piece_at(s)
            if pc is None or pc.piece_type != chess.PAWN or pc.color != me:
                continue
            q = race_claim(d, s, allow_second=True, recheck=False)
            if q is not None and q.ours <= r.ours - 1:
                ok = True
                break
        if not ok:
            return False
    return True


def race_passers(board: chess.Board) -> dict:
    """{pawn square: ours} for every pawn of the side to move on which the word binds (the pass-claim reading: `board` has the owner to move)."""
    if not RACE:
        return {}
    out = {}
    for s in chess.SquareSet(board.pawns & board.occupied_co[board.turn]):
        q = race_claim(board, s)
        if q is not None:
            out[s] = q.ours
    return out


# ---------------------------------------------------------------------------------------------------------------- 62.1b: the stopper's PATH and its denial (Race.deny_sound)

def stopper_walk(board: chess.Board, s: int, targets: set, limit: int, deny: int = 0) -> tuple | None:
    """The shortest walk of the piece on `s` to a square of `targets` in at most `limit` moves over the squares NOT in the `deny` mask (static board: the other pieces stand; a square holding a piece
    of its own colour is not entered; a king may take a piece of ours that is not denied). -> (moves, [the squares it lands on, in order]) or None. This is the PATH of the stopper
    (Kaveh 2026-10-22: my intent is to promote; their intent is to stop it; how -- which piece, along which path), and `deny` is the set of squares it cannot STAND and SURVIVE on
    (Race.deny_sound: a walk that needs a denied square is not a walk).
    plan 72.7 step 3b: the walk read in Rust on the board's placement (`analyst_core.stopper_walk_bb`: the same walk, each step's squares in ascending order, as
    python-chess's SquareSet iterates them). The Python body is the test oracle (tests/test_plan72_7_step3.py)."""
    import analyst_core as _A
    tm = 0
    for t in targets:
        tm |= chess.BB_SQUARES[t]
    return _A.stopper_walk_bb((board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings, board.occupied_co[True], board.occupied_co[False]),
                              s, tm, max(limit, 0), deny)


def denial_mask(after: chess.Board, me: bool, stopper: int, skip: int) -> int:
    """The squares the stopper (their piece on `stopper`, `after` = the board after our move, their turn) cannot stand and survive on: for a KING every square we attack (their king
    removed from the board, our pawn `skip` -- the passer, which moves -- not counted): a legal move never leaves a king attacked (Race.no_stand_attacked); for another piece the squares
    our pawns (the passer excluded) attack, when the piece is worth more than a pawn (it is taken for a pawn).
    plan 72.7 step 3b: read in Rust on the board's placement (`analyst_core.denial_mask_bb`, the mask `deny_moves_bb` reads). The Python body is the test oracle
    (tests/test_plan72_7_step3.py)."""
    import analyst_core as _A
    return _A.denial_mask_bb((after.pawns, after.knights, after.bishops, after.rooks, after.queens, after.kings, after.occupied_co[True], after.occupied_co[False]),
                             me, stopper, skip)


def deny_moves(board: chess.Board, me: bool, sq: int, stopper: int, targets: set, n: int, legal: list) -> list:
    """62.1b: the moves of ours (not the passer's own) after which the stopper's walk to `targets` no longer exists within n + 1 moves (they get a reply before our first push) over the
    squares it can stand on -- DENY THE STOPPER'S PATH. -> [(move, why)] with the path it had named.
    plan 72.7 step 3b: read in Rust on the board's placement in ONE call (`analyst_core.deny_moves_bb`: python-chess's push on the placement -- castling's rook and en
    passant's pawn included --, the stopper's walk before and after, `denial_mask`, the stopper taking the piece that denies), the moves handed in python-chess's order
    and returned as the indices of the ones that deny. The Python body is the test oracle (tests/test_plan72_7_step3.py)."""
    import analyst_core as _A
    tm = 0
    for t in targets:
        tm |= chess.BB_SQUARES[t]
    got = _A.deny_moves_bb((board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings, board.occupied_co[True], board.occupied_co[False]),
                           me, sq, stopper, tm, max(n, 0), [m.uci() for m in legal])
    return [(legal[i], why) for i, why in got]


# ---------------------------------------------------------------------------------------------------------------- the bridge: an exhaustive race search

def wins_race(b: chess.Board, plies: int) -> bool:
    """We (the side to move) promote within `plies` plies whatever the opponent does AND the new queen survives their reply (value 8 = a queen for a pawn); a stalemated opponent does not count (the Lean `forcedWithinB`)."""
    us = b.turn

    def survives(bd: chess.Board) -> bool:
        """Value 8 is a queen for a pawn: the new queen must still stand after EVERY reply of theirs (a king that takes it undefended makes the promotion worth nothing); mate is fine, stalemate is not."""
        reps = list(bd.legal_moves)
        if not reps:
            return bd.is_checkmate()
        for r in reps:
            bd.push(r)
            ok = bool(bd.queens & bd.occupied_co[us])
            bd.pop()
            if not ok:
                return False
        return True

    def rec(bd: chess.Board, n: int, promoted: bool) -> bool:
        if promoted:
            return True
        if n == 0:
            return False
        moves = list(bd.legal_moves)
        if bd.turn == us:
            moves.sort(key=lambda m: (0 if m.promotion else 1, 0 if bd.piece_type_at(m.from_square) == chess.PAWN else 1))
            for m in moves:
                bd.push(m)
                if m.promotion:
                    ok = survives(bd)
                else:
                    ok = rec(bd, n - 1, False)
                bd.pop()
                if ok:
                    return True
            return False
        if not moves:
            return False
        for m in moves:
            bd.push(m)
            ok = rec(bd, n - 1, False)
            bd.pop()
            if not ok:
                return False
        return True

    return rec(b.copy(stack=True), plies, False)


def random_race_position(rng: random.Random, two_ours: bool = True, rook: bool = False) -> chess.Board | None:
    """Our king and one or two pawns (white to move) against a king and up to two pawns; sometimes a rook of ours. For the bridge."""
    b = chess.Board(None)
    k = rng.randrange(64)
    ek = rng.randrange(64)
    if kdist(k, ek) <= 1:
        return None
    npw = rng.randint(1, 2 if two_ours else 1)
    nbl = rng.randint(0, 2)
    squares = rng.sample([s for s in range(64) if s not in (k, ek)], npw + nbl + (1 if rook else 0))
    b.set_piece_at(k, chess.Piece(chess.KING, chess.WHITE))
    b.set_piece_at(ek, chess.Piece(chess.KING, chess.BLACK))
    for i, s in enumerate(squares):
        if i < npw:
            if chess.square_rank(s) in (0, 7):
                return None
            b.set_piece_at(s, chess.Piece(chess.PAWN, chess.WHITE))
        elif i < npw + nbl:
            if chess.square_rank(s) in (0, 7):
                return None
            b.set_piece_at(s, chess.Piece(chess.PAWN, chess.BLACK))
        else:
            b.set_piece_at(s, chess.Piece(chess.ROOK, chess.WHITE))
    b.turn = chess.WHITE
    if not b.is_valid() or b.is_check() or b.attackers_mask(chess.WHITE, ek):
        return None
    return b


def validate(n: int = 2000, seed: int = 1, log=print) -> dict:
    """The bridge: for random positions where the word binds, the exhaustive search must confirm that we promote within 2*ours - 1 plies. -> counts (a disagreement is listed)."""
    rng = random.Random(seed)
    t0 = time.monotonic()
    tried = bound = bad = single = over = 0
    rows = []
    while tried < n:
        b = random_race_position(rng, rook=rng.random() < 0.3)
        if b is None:
            continue
        tried += 1
        for s in chess.SquareSet(b.pawns & b.occupied_co[chess.WHITE]):
            q = race_claim(b, s)
            if q is None:
                continue
            bound += 1
            single += q.second is None
            over += q.second is not None
            if not wins_race(b, 2 * q.ours - 1):
                bad += 1
                rows.append((b.fen(), _name(s), q.ours, q.text))
                log(f"DISAGREE {b.fen()} pawn {_name(s)} ours {q.ours}: {q.text}")
            break
    log(f"OBS race bridge: {tried} positions, word bound on {bound} ({single} single, {over} deflection), exhaustive search disagrees on {bad}, {time.monotonic() - t0:.1f} s")
    return dict(tried=tried, bound=bound, bad=bad, rows=rows, single=single, over=over)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.validate:
        validate(a.validate, a.seed)
