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

import os
import re
from dataclasses import dataclass, field

import chess

import analyst_core as _A
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import race as RC

STAKE_MATE = 100
PASS_CLAIMS = [0]                             # 52.18 (3): forcing_why answers that were a claim bound after a pass (the OBS counter)
STAKE_PROMOTE = 8                             # a queen less the pawn
DEFECTS_RUST = True                           # 54 (B1): mate_defects / material_defects read by analyst_core (the Python path below stays; ValueError = a position shakmaty cannot hold -> Python)
ROUTES_RUST = True                            # 54 (B2): routes_of's `_still` filter and `forcing_why` for the whole candidate list in two Rust calls (the Python path stays)
TRAP_RUST = True                              # 54 (B4): trap_fix's three reads (safe squares, quiet trap routes, fork routes) in one Rust call (the Python path stays)
SKEWER_RUST = True                            # 54 (B4): skewer_routes / drive_onto_routes read by analyst_core (the Python path stays)
REGRESS_LOG = None                            # 54 (B): a list -> every fixes_of / routes_of / forcing_why call appends what it answered (scripts/compare_regress.py diffs these)
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
    reads: frozenset | None = None            # 52.12: the squares whose occupancy or attack counts this question read (None: not recorded, never inherited)
    claimed: tuple | None = None              # 52.13 (1): (net value, the Claim, its reads) once a claim verified this question; the reads = the question's plus the squares of the bound atoms


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
    """What stands between the pattern and a mate (`_mate_defects`); the call is logged when REGRESS_LOG is set."""
    out = _mate_defects(board, us, placement)
    if REGRESS_LOG is not None:
        REGRESS_LOG.append(("mate_defects", board.epd(), G.colour_of(us), [(p.symbol(), at, frm) for p, at, frm in placement], [(d.kind, d.sq, tuple(d.by), d.why) for d in out]))
    return out


def _mate_defects(board: chess.Board, us: str | bool, placement: list) -> list:
    """What stands between the pattern and a mate, read from the rules on the counterfactual (placements made, their move). [] = they have no legal move and are
    in check: the pattern IS a mate. Order: the checker takeable, flights, interpositions."""
    me = G.colour_of(us)
    if DEFECTS_RUST:
        try:
            return [Defect(k, sq, list(by), why) for k, sq, by, why in _A.mate_defects(G.fen_of(board), me, [(p.symbol(), at, frm) for p, at, frm in placement])]
        except ValueError:
            pass                                  # a counterfactual shakmaty cannot hold: the Python rules read it
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
    """Why we do not win the piece on `target_sq` now (`_material_defects`); the call is logged when REGRESS_LOG is set."""
    out = _material_defects(board, us, target_sq)
    if REGRESS_LOG is not None:
        REGRESS_LOG.append(("material_defects", board.epd(), G.colour_of(us), target_sq, [(d.kind, d.sq, tuple(d.by), d.why) for d in out]))
    return out


def _material_defects(board: chess.Board, us: str | bool, target_sq: int) -> list:
    """Why we do not win the piece on `target_sq` now: [] when the exchange on it already wins for us (`see` from our side > 0); otherwise 'not attacked' (no
    piece of ours attacks it) or 'defended' (attacked, but it is guarded and the exchange gives nothing)."""
    me = G.colour_of(us)
    if DEFECTS_RUST:
        try:
            return [Defect(k, sq, list(by), why) for k, sq, by, why in _A.material_defects(G.fen_of(board), me, target_sq)]
        except ValueError:
            pass
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


def exchange_read(board: chess.Board) -> str:
    """52.16 (3): the best static exchange of the side to move on `board`, READ: where it is, what it wins, the captures in order (the cheapest attacker first, each
    side going on while it has an attacker) and any attacker that stands BEHIND a piece of its own side on the line (an x-ray: the bishop that comes through the vacated
    square behind the queen). "Qxa8 wins 5: Qxa8 Qxa8 Bxa8, the Bxa8 behind the Q"."""
    me = board.turn
    best = (0, None)
    for s in chess.SQUARES:
        p = board.piece_at(s)
        if p is None or p.color == me or p.piece_type == chess.KING or not board.is_attacked_by(me, s):
            continue
        v = G.see(board, s)
        if v > best[0]:
            best = (v, s)
    v, sq = best
    if sq is None:
        return "no exchange"
    direct = set(board.attackers(me, sq)) | set(board.attackers(not me, sq))
    c = board.copy(stack=False)
    seq, behind = [], []
    while len(seq) < 8:
        atts = list(c.attackers(c.turn, sq))
        if not atts:
            break
        a = min(atts, key=lambda x: (G.worth(c.piece_at(x)) if c.piece_at(x).piece_type != chess.KING else 99, x))
        pc = c.piece_at(a)
        if pc.piece_type == chess.KING and any(True for _ in c.attackers(not c.turn, sq)):
            break
        letter = chess.FILE_NAMES[chess.square_file(a)] if pc.piece_type == chess.PAWN else pc.symbol().upper()
        txt = f"{letter}x{G.name(sq)}"
        seq.append(txt)
        if a not in direct:
            blockers = [x for x in chess.SquareSet(chess.between(a, sq)) if board.piece_at(x) is not None]
            if blockers:
                front = min(blockers, key=lambda x: chess.square_distance(x, sq))
                behind.append(f"the {txt} behind the {board.piece_at(front).symbol().upper()}")
        c.push(chess.Move(a, sq, promotion=chess.QUEEN if pc.piece_type == chess.PAWN and chess.square_rank(sq) in (0, 7) else None))
    return f"{seq[0]} wins {v}: {' '.join(seq)}" + (", " + ", ".join(behind) if behind else "")


def mate_threat(board: chess.Board, m: chess.Move, us: str | bool) -> bool:
    """52.12: after our (quiet) move m and a pass, we have a mate in one -- the rules' fact (every legal move of ours, one pushed): their every reply must meet it."""
    me = G.colour_of(us)
    c = board.copy(stack=False)
    c.push(m)
    if c.is_check():
        return False
    d = G.with_turn(c, me)
    for mv in list(d.legal_moves):
        if not d.gives_check(mv):                         # only a check can mate
            continue
        d.push(mv)
        mated = d.is_checkmate()
        d.pop()
        if mated:
            return True
    return False


def mate2_threat(board: chess.Board, m: chess.Move, us: str | bool) -> bool:
    """plan 59 tier 2: after our (quiet) move m and a pass, we have a FORCED MATE IN TWO, read by the proved word `mate_in_two` (Lean `Ideas.Mined.Schemas.mate_in_two`: a move x -> y after which
    every reply is met by a mate; bound by `selfmine.bind_mate2`, the premise read in Rust) -- the same word the binder's mate_in_two schema uses, on the board after m with the move passed to them."""
    from catspace.incubator.analyst.discovery import selfmine as SM
    me = G.colour_of(us)
    c = board.copy(stack=False)
    c.push(m)
    if c.is_check() or not any(True for _ in G.with_turn(c, me).legal_moves):
        return False
    return bool(SM.bind_mate2(G.with_turn(c, me).fen(), "w" if me else "b"))


def forcing_why(board: chess.Board, m: chess.Move, stake: int, us: str | bool, mate_threats: bool = False, notes: list | None = None, mate2: bool = False,
                delayed: bool = False, nosafe: bool = False) -> str | None:
    """The reason `m` is forcing (see `_forcing_why_py`), or None. 54 (B2): read by analyst_core when ROUTES_RUST (the same answer, the same notes). plan 59 tier 2 (`mate2`): a quiet move
    after which a pass leaves us a forced mate in two ('threat of mate in two', `mate2_threat`) is forcing too, whatever the stake (a mate question follows checks and mate threats only)."""
    got = _forcing_batch(board, [m], stake, us, mate_threats, notes) if ROUTES_RUST else None
    why = got[0] if got is not None else _forcing_why_py(board, m, stake, us, mate_threats, notes)
    if why is None and mate2 and mate_threats and mate2_threat(board, m, us):
        why = "threat of mate in two"
    if why is None and (delayed or nosafe) and mate_threats and not board.is_capture(m) and G.colour_of(us) == board.turn:
        if nosafe:                                                    # plan 61 A4 (Lean `Ideas.Forcing.noSafeMove`)
            why = no_safe_move(board, m, stake, us)
        if why is None and delayed:                                   # plan 61 A3 (Lean `Ideas.Forcing.delayedThreat`)
            why = delayed_threat(board, m, stake, us)
    if REGRESS_LOG is not None:
        REGRESS_LOG.append(("forcing", board.epd(), m.uci(), stake, mate_threats, why, tuple(notes) if notes is not None else None))
    return why


def _forcing_batch(board: chess.Board, moves: list, stake: int, us: str | bool, mate_threats: bool = False, notes: list | None = None) -> list | None:
    """54 (B2): `_forcing_why_py` for several moves of the side to move in one Rust call -> [why or None] (None overall: the Python path reads it)."""
    me = G.colour_of(us)
    if board.turn != me or not moves:
        return None
    try:
        res = _A.forcing_whys(G.fen_of(board), [m.uci() for m in moves], stake, me, mate_threats)
    except ValueError:
        return None
    out = []
    for m, (st, text, n1, n2, cap) in zip(moves, res):
        if st == "why":
            out.append(text)
        elif st == "pass":
            c = board.copy(stack=False)
            c.push(m)
            out.append(pass_claim(board, c, m, us, stake))
        elif st == "survives":
            c = board.copy(stack=False)
            c.push(m)
            out.append(f"threat {n1} survives {c.san(chess.Move.from_uci(cap))} (the {text} hangs, {n2})")
        elif st == "hangs":
            if notes is not None:
                c = board.copy(stack=False)
                c.push(m)
                notes.append(f"{board.san(m)}: not a threat: the {text} on {G.name(m.to_square)} hangs (they win {n1} by the exchange there)"
                             + (f" and the threat does not survive {c.san(chess.Move.from_uci(cap))} ({n2})" if cap else ""))
            out.append(None)
        else:
            out.append(None)
    return out


def _forcing_why_py(board: chess.Board, m: chess.Move, stake: int, us: str | bool, mate_threats: bool = False, notes: list | None = None) -> str | None:
    """A move is FORCING when it is a check, a threat of mate (52.12: after it, a pass leaves us a mate in one), or a capture / threat worth at least the stake of the
    question (a mate stakes 100: only a check or a mate threat is). 52.13 (2d): a QUIET move threatens only if the moved piece is not lost by the exchange on its own
    square (Lean `Ideas.Vocabulary.landsSafe`: after the move the exchange on its destination wins nothing for the other side; the Rust `see`, them on move): the
    threatening rook the queen takes is not a threat (08pSj's Rc6) -- the reason is appended to `notes`. -> why or None."""
    if board.gives_check(m):
        return "check"
    if mate_threats and mate_threat(board, m, us):          # (asked by the chase and the reach roots only: the other fixes keep their old forcing)
        return "threat of mate"
    if stake >= STAKE_MATE:
        return None
    if board.is_capture(m):
        v = 1 if board.is_en_passant(m) else G.worth(board.piece_at(m.to_square))
        if v >= stake:
            return f"capture {v}"
    c = board.copy(stack=False)
    c.push(m)
    t = threat_value(c, us)
    if t < stake:
        return pass_claim(board, c, m, us, stake)
    me = G.colour_of(us)
    lost = G.see_for(c, m.to_square, not me)
    if lost > 0:                                              # landsSafe fails: the exchange on the destination wins for them
        # 52.16 (3): a SACRIFICE THAT STILL THREATENS is forcing: the threat SURVIVES the capture of the moved piece -- after every capture of it by them, the most we would win by the
        # exchange on a piece of theirs, us to move, is still at least the stake (the discovery position's Nf6: Bxf6, then Qxa8 Qxa8 Bxa8 through the vacated e4). The worst capture for us decides.
        pc = c.piece_at(m.to_square)
        worst = None
        for cap in list(c.legal_moves):
            if cap.to_square != m.to_square:
                continue
            d = c.copy(stack=False)
            d.push(cap)
            t2 = 0 if d.is_check() else threat_value(d, us)           # in check we must answer it: no standing threat to read
            if worst is None or t2 < worst[0]:
                worst = (t2, c.san(cap))
        if worst is not None and worst[0] >= stake:
            return f"threat {worst[0]} survives {worst[1]} (the {pc.symbol().upper()} hangs, {lost})"
        if notes is not None:
            notes.append(f"{board.san(m)}: not a threat: the {pc.symbol().upper()} on {G.name(m.to_square)} hangs (they win {lost} by the exchange there)"
                         + (f" and the threat does not survive {worst[1]} ({worst[0]})" if worst is not None else ""))
        return None
    return f"threat {t}"


def pass_claim(board: chess.Board, after: chess.Board, m: chess.Move, us: str | bool, stake: int) -> str | None:
    """52.18 (3): THREAT-OF-X FORCING. A move is forcing when, after it, a PASS by them lets a claim of ours bind with no defect left -- the claim the move CREATES (it did not bind
    before the move). Here: a promotion by the passer lemma (`unstoppablePasser2_sound` binds for a pawn after the pass: 0Nv29's b6, "b7 then b8=Q"); the fork claim of a landing is
    item 4's. (A trap is not read here: it costs a free reply by its definition, 52.10.) -> the reason naming the claim, or None. The pass is the null move (`G.with_turn`)."""
    if stake > STAKE_PROMOTE or after.is_check():
        return None
    me = G.colour_of(us)
    if not any(board.pieces(chess.PAWN, me)):
        return None
    from catspace.incubator.analyst import lean_passer_v2_proto as PV
    passed = G.with_turn(after, me)
    now = dict(PV.proved_passers(passed, bishops=True))
    rnow = {sq: 2 * o - 1 for sq, o in RC.race_passers(passed).items() if sq not in now}          # plan 62.1: the race word's passers (Ideas.Race.race_sound), after the lemma's
    if not now and not rnow:
        return None
    before = dict(PV.proved_passers(G.with_turn(board, me), bishops=True))
    rbefore = RC.race_passers(G.with_turn(board, me)) if rnow else {}
    new = [sq for sq in now if not (sq in before or (sq == m.to_square and m.from_square in before))]
    rnew = [sq for sq in rnow if not (sq in before or sq in rbefore or (sq == m.to_square and (m.from_square in before or m.from_square in rbefore)))]
    if not new and not rnew:
        return None
    sq = min(new) if new else min(rnew)
    n = G.word(passed, ("pushes", G.name(sq)), me)
    path = [G.name(x) for x in pawn_path(passed, sq, n)]
    PASS_CLAIMS[0] += 1
    word = f"unstoppablePasser2_sound, within {now[sq]} plies" if new else f"race_sound, within {rnow[sq]} plies"
    return f"threat: {', '.join(path[:-1]) + (' then ' if len(path) > 1 else '')}{path[-1]}=Q binds after a pass ({word})"


# ---------------------------------------------------------------------------------------------------------------- plan 61 stage A: the forced chain's words

@dataclass
class Threat:
    """plan 61 A1: the threat a forcing move makes, as an object read back from its reason string and the board after the move.
    kind: check | mate | mate2 | capture | threat | pass-claim (| delayed | zugzwang: the A3 / A4 words, which no single reply parries)."""
    kind: str
    target: int | None                        # the square the threat is on: their king (check, mate), the captured piece, the piece the exchange wins, the promotion square
    value: int                                # what it wins (a mate: 100; a check: 0)
    word: str                                 # the proved word that binds it
    why: str                                  # the reason string it was read from


def _threat_square(after: chess.Board, us: str | bool) -> tuple:
    """After our move (their turn) and a pass: (the most we would win by the exchange on a piece of theirs, its square) -- `threat_value`, with the square."""
    me = G.colour_of(us)
    c = G.with_turn(after, me)
    best = (0, None)
    for s in chess.SQUARES:
        p = c.piece_at(s)
        if p and p.color != me and p.piece_type != chess.KING and c.is_attacked_by(me, s):
            v = G.see(c, s)
            if v > best[0]:
                best = (v, s)
    return best


def threat_of(board: chess.Board, m: chess.Move, why: str | None, us: str | bool | None = None) -> "Threat | None":
    """plan 61 A1: THE THREAT AS AN OBJECT. `why` is the reason `forcing_why` returned for `m` on `board` (the Rust forcing path gives the same string); the object is rebuilt in
    Python from the reason and the board after `m`. -> Threat or None (no reason)."""
    if not why:
        return None
    me = board.turn if us is None else G.colour_of(us)
    c = board.copy(stack=False)
    c.push(m)
    ksq = c.king(not me)
    if why == "check":
        return Threat("check", ksq, 0, "check", why)
    if why == "threat of mate":
        return Threat("mate", ksq, STAKE_MATE, "mate in one", why)
    if why == "threat of mate in two":
        return Threat("mate2", ksq, STAKE_MATE, "mate_in_two", why)
    if why.startswith("capture "):
        return Threat("capture", m.to_square, int(why.split()[1]), "capture", why)
    if why.startswith("threat:"):                                      # pass_claim: "threat: b7 then b8=Q binds after a pass (unstoppablePasser2_sound, ...)"
        mm = re.search(r"([a-h][1-8])=Q", why)
        return Threat("pass-claim", chess.parse_square(mm.group(1)) if mm else None, STAKE_PROMOTE, "unstoppablePasser2_sound", why)
    if why.startswith("threat "):                                      # "threat n" / "threat n survives X (the P hangs, k)"
        n = int(why.split()[1])
        v, sq = _threat_square(c, me)
        return Threat("threat", sq, n, "survives capture" if "survives" in why else "exchange", why)
    if why.startswith("delayed threat"):
        return Threat("delayed", None, 0, "delayedThreat", why)
    if why.startswith("no safe move"):
        return Threat("zugzwang", None, 0, "noSafeMove", why)
    return None


def _binds(d: chess.Board, th: Threat, stake: int, me: bool) -> int:
    """Does the threat still bind on `d` (the board after their reply, our move)? -> the value it binds at (0: it does not). Re-read on the reply's own board, never assumed."""
    if th.kind == "mate":
        for mv in list(d.legal_moves):
            if d.gives_check(mv):
                d.push(mv)
                mated = d.is_checkmate()
                d.pop()
                if mated:
                    return STAKE_MATE
        return 0
    if th.kind == "mate2":
        from catspace.incubator.analyst.discovery import selfmine as SM
        return STAKE_MATE if SM.bind_mate2(d.fen(), "w" if me else "b") else 0
    if th.kind == "threat":
        v = threat_value(d, me)
        return v if v >= stake else 0
    if th.kind == "pass-claim":
        from catspace.incubator.analyst import lean_passer_v2_proto as PV
        return STAKE_PROMOTE if th.target in dict(PV.proved_passers(G.with_turn(d, me), bishops=True)) else 0
    return 0                                                          # check, capture: already played, nothing stands to be parried; delayed / zugzwang: no single reply


def _counter(after: chess.Board, r: chess.Move, stake: int, them: bool) -> bool:
    """A reply that is itself forcing against us (Allis's counter-threat hole): a check, a capture worth the stake, or a threat of theirs by `forcing_why` with them on move."""
    if after.gives_check(r):
        return True
    if after.is_capture(r):
        v = 1 if after.is_en_passant(r) else G.worth(after.piece_at(r.to_square))
        if v >= stake:
            return True
    return forcing_why(after, r, stake, them) is not None


def parries(after: chess.Board, threat: Threat, stake: int, us: str | bool | None = None) -> tuple:
    """plan 61 A2 (Lean `Ideas.Forcing.parries` / `parryList` / `closedList`): `after` = the board after our forcing move (their turn). PARRIES are their legal replies after which the
    threat no longer binds (re-read on the reply's board: `_binds` < stake) plus every reply that is itself forcing against us (`_counter`); CLOSED are the rest, each with the value the
    threat still binds at on its own board. -> ([parry moves], [(closed move, value)]). A reply outside `parries` leaves the threat bound (`Ideas.Forcing.closed_binds`)."""
    me = (not after.turn) if us is None else G.colour_of(us)
    par, closed = [], []
    for r in list(after.legal_moves):
        d = after.copy(stack=False)
        d.push(r)
        v = _binds(d, threat, stake, me) if not d.is_check() else 0
        if v and not _counter(after, r, stake, after.turn):
            closed.append((r, v))
        else:
            par.append(r)
    return par, closed


_STANDING: dict = {}
G.CLEAR_HOOKS.append(_STANDING.clear)


def check_threat(board: chess.Board, x: chess.Move, stake: int, us: str | bool) -> str | None:
    """plan 63.1 (Lean `Ideas.Forcing.checkThreat`): a CHECK x that is also a THREAT. After x their king is in check, so no pass is legal; the pass test is read the way `pass_claim` /
    `threat_value` read it, on the board after x with US to move again (`G.with_turn`): a move of ours that then MATES (the discovered check Ng1+ that uncovers Qh7#: every reply that does not
    meet the mate loses), or an exchange worth at least the stake (stake < mate). A bare check with nothing behind it is None; a check that mates is 'check that mates'.
    -> the reason ('check that also threatens mate (Qh7#)' / 'check that also threatens +v') or None. Not a lambda-1 NEW test by itself: the caller filters moves that already stood before."""
    me = G.colour_of(us)
    if not board.gives_check(x):
        return None
    c = board.copy(stack=False)
    c.push(x)
    if c.is_checkmate():
        return "check that mates"
    d = G.with_turn(c, me)
    ksq = c.king(not me)
    for mv in list(d.legal_moves):
        if mv.to_square == ksq:
            continue
        if G.mates(d, mv):                                                # (pushes only a move that could give check)
            return f"check that also threatens mate ({G.with_turn(c, me).san(mv)})"
    if stake < STAKE_MATE:
        t = threat_value(c, me)
        if t >= stake:
            return f"check that also threatens +{t}"
    return None


def _follow_up(board: chess.Board, x: chess.Move, stake: int, me: bool) -> str | None:
    """plan 63.1: why `x` (our move on `board`) is a FOLLOW-UP for a delayed threat, or None: (a) a threat / capture worth the stake as in 61 A3 (a bare capture only if the exchange wins);
    (b) a THREAT OF MATE at any stake (a mate exceeds every stake: read with `mate_threats=True`); (c) a CHECK that is also a threat (`check_threat`). A bare check does not count at a material stake;
    at the mate stake a check counts as in 61 A3."""
    w = forcing_why(board, x, stake, me, mate_threats=True)
    if w is None:
        return None
    if w == "check":
        return "check" if stake >= STAKE_MATE else check_threat(board, x, stake, me)
    if w.startswith("capture") and stake < STAKE_MATE and G.see(board, x.to_square) <= 0:
        return None                                                      # (a bare capture counts only if the exchange on it wins)
    return w


def _standing_forcing(board: chess.Board, stake: int, me: bool) -> frozenset:
    """The UCIs of our moves that are follow-ups on `board` itself (what is already there before a quiet move is played): a delayed threat is made by the move, not found beside it."""
    key = (board.board_fen(), board.turn, board.castling_rights, board.ep_square, stake)
    hit = _STANDING.get(key)
    if hit is None:
        hit = _STANDING[key] = frozenset(x.uci() for x in board.legal_moves if _follow_up(board, x, stake, me))
    return hit


def delayed_threat(board: chess.Board, m: chess.Move, stake: int, us: str | bool) -> str | None:
    """plan 61 A3 (Lean `Ideas.Forcing.delayedThreat`; Thomsen's lambda-2) with plan 63.1: the quiet move `m` is DELAYED-FORCING iff after it, for EVERY legal reply of theirs, some NEW move of ours is a
    follow-up (`_follow_up`: a threat / capture worth the stake, a threat of mate at any stake, or a check that is also a threat -- Lean `checkThreat`). One bounded AND level: their replies x our
    candidates, the forcing test only, no recursion. -> "delayed threat: after every reply a new threat of ours (tightest: after Rg8+: h2 ...)" naming the reply that leaves the fewest, or None."""
    me = G.colour_of(us)
    c = board.copy(stack=False)
    c.push(m)
    replies = list(c.legal_moves)
    if c.is_check() or not replies:
        return None
    standing = _standing_forcing(board, stake, me)
    tight, via_check = None, None
    for r in replies:
        d = c.copy(stack=False)
        d.push(r)
        ours = [x for x in d.legal_moves if x.uci() not in standing]       # the threat must be NEW: a follow-up already there before `m` (a hanging queen) is not made by `m`
        hits = []
        for x in ours:
            w = _follow_up(d, x, stake, me)
            if w:
                hits.append((x, w))
        if not hits:
            return None
        if tight is None or len(hits) < tight[0]:
            tight = (len(hits), c.san(r), d.san(hits[0][0]), hits[0][1])
        if via_check is None and all(w.startswith("check that") for _x, w in hits):          # 63.1: the replies after which ONLY a check-that-is-a-threat follows are named
            via_check = f"after {c.san(r)}: {d.san(hits[0][0])} ({hits[0][1]})"
    return (f"delayed threat: after every reply a new threat of ours (tightest, after {tight[1]}: {tight[2]} {tight[3]}; {tight[0]} such move(s))"
            + (f"; by a check that is also a threat, {via_check}" if via_check else ""))


# ---------------------------------------------------------------------------------------------------------------- plan 63.2: the duty word (Lean Ideas.Duty)

DUTY = os.environ.get("CATSPACE_DUTY", "1") != "0"       # plan 63.2: a safe square reached by a move that abandons a duty >= the piece's worth is NOT safe (trap_fix, `take its squares`); analyse --no-duty sets CATSPACE_DUTY=0 (inherited by the spawned workers)
STAKE_DUTY_MATE = 100


def iter_safe_moves(board: chess.Board, sq: int):
    """`safe_moves`, lazily (the callers that need only the first safe square that is kept stop reading there)."""
    pc = board.piece_at(sq)
    p = G.with_turn(board, pc.color)
    for m in p.legal_moves:
        if m.from_square != sq or p.is_castling(m):
            continue
        a = p.copy(stack=False)
        a.push(m)
        if G.see_for(a, m.to_square, not pc.color) == 0:
            yield m


def safe_moves(board: chess.Board, sq: int) -> list:
    """The legal moves of the piece on `sq` (its owner to move, no e.p., castling is not a move of the piece) after which the exchange on the destination wins nothing for the other side: the
    Rust word `mobility` (vocab.rs, Lean `landsSafe`), listed. `len(safe_moves(b, s)) == G.mobility_of(b, s)`."""
    return list(iter_safe_moves(board, sq))


_MATE_AHEAD: dict = {}
G.CLEAR_HOOKS.append(_MATE_AHEAD.clear)


def _mate_ahead(d: chess.Board, me: bool) -> str | None:
    """On `d` with US to move: the way we MATE or threaten mate, by rule -- a move that checkmates, a check that also threatens mate (`check_threat`), a quiet move after which a pass leaves a mate
    in one (`mate_threat`; the same word the Rust forcing path reads, here asked in one batch). -> the move's SAN (with the mate it leads to) or None. (`_their_after_capture` reads a duty the same
    way: mate -> 100.) Memoised by position."""
    key = (d.board_fen(), d.turn, d.castling_rights, d.ep_square, me)
    if key in _MATE_AHEAD:
        return _MATE_AHEAD[key]
    moves = list(d.legal_moves)
    whys = _forcing_batch(d, moves, STAKE_MATE, me, True) if d.turn == me else None
    out = None
    if whys is None:
        whys = []
        for x in moves:
            whys.append("check" if d.gives_check(x) else ("threat of mate" if mate_threat(d, x, me) else None))
    for x, w in zip(moves, whys):                                      # (a quiet threat of mate is read already: the checks, which cost a scan each, only when there is none)
        if w == "threat of mate":
            out = f"{d.san(x)} (threat of mate)"
            break
    if out is None:
        for x, w in zip(moves, whys):
            if w == "check":
                t = check_threat(d, x, STAKE_MATE, me)
                if t:
                    out = f"{d.san(x)} ({t})"
                    break
    _MATE_AHEAD[key] = out
    return out


def duty_after(board: chess.Board, r: chess.Move, us: str | bool, ctx: int | None = None, need: int | None = None) -> tuple:
    """plan 63.2: what we win BY RULE after their move `r` on `board` (them to move), us to move next, beyond what we already won: (value, what). A move that checks us wins nothing (a counter-threat,
    not an abandonment); a mate (or a threat of mate) we did not have before -> 100; else the best static exchange of ours (`threat_value`) less what `r` itself captured and less the exchange already
    standing before the move. Lean `Ideas.Duty.dutyAfter`."""
    me = G.colour_of(us)
    gain = 1 if board.is_en_passant(r) else (G.worth(board.piece_at(r.to_square)) if board.is_capture(r) else 0)
    a = board.copy(stack=False)
    a.push(r)
    if a.is_check():
        return 0, "check"
    now_ex = ctx if ctx is not None else _best_exchange(G.with_turn(board, me), me)                  # what we already won before their move (read once per board by the callers)
    v = max(0, _best_exchange(a, me) - gain - now_ex)
    if need is not None and v >= need:                                    # the exchange alone already reaches the piece's worth: the mate need not be read
        return v, f"the exchange wins {v}"
    mate = _mate_ahead(a, me)
    if mate and _mate_ahead(G.with_turn(board, me), me) is None:           # a mate (or threat of mate) that did not stand before the move
        return STAKE_DUTY_MATE, mate
    return v, f"the exchange wins {v}" if v else "nothing"


def _best_exchange(b: chess.Board, me: bool) -> int:
    """The most we win by the exchange on a piece of theirs, us to move on `b` (= `threat_value` of a board read with us to move)."""
    return threat_value(b, me)


def duty(board: chess.Board, sq: int, us: str | bool) -> tuple:
    """plan 63.2 (Lean `Ideas.Duty.duty`): the DUTY of their piece on `sq` = the LEAST, over its legal moves, of what we win by rule after that move (`duty_after`). 'The piece is pinned to a threat.'
    -> (value, what, move SAN): the least move. A piece with no legal move has duty 0 (it is frozen, not bound). `dutyBound` = value >= the piece's worth (a mate: every worth)."""
    pc = board.piece_at(sq)
    p = G.with_turn(board, pc.color)
    best = None
    me = G.colour_of(us)
    now = G.with_turn(board, me)
    ctx = _best_exchange(now, me)
    for m in p.legal_moves:
        if m.from_square != sq or p.is_castling(m):
            continue
        v, what = duty_after(p, m, us, ctx)
        if best is None or v < best[0]:
            best = (v, what, p.san(m))
    return best if best is not None else (0, "no legal move", "")


def duty_bound(board: chess.Board, sq: int, us: str | bool) -> bool:
    """The piece is DUTY-BOUND: every move of it abandons a duty worth at least the piece (Lean `Ideas.Duty.dutyBound`)."""
    pc = board.piece_at(sq)
    if pc is None or pc.piece_type == chess.KING:
        return False
    v, _w, san = duty(board, sq, us)
    return bool(san) and v >= G.worth(pc)


_DUTY: dict = {}
G.CLEAR_HOOKS.append(_DUTY.clear)


def duty_safe_moves(board: chess.Board, sq: int, us: str | bool) -> tuple:
    """The safe squares of the piece on `sq` that do NOT abandon a duty >= its worth -> (kept moves, [(abandoning move SAN, value, what)]). Memoised by position (cleared with the other caches)."""
    key = (board.board_fen(), board.castling_rights, sq, G.colour_of(us))
    hit = _DUTY.get(key)
    if hit is not None:
        return hit
    pc = board.piece_at(sq)
    kept, cut = [], []
    p = G.with_turn(board, pc.color)
    now = G.with_turn(board, G.colour_of(us))
    ctx = _best_exchange(now, G.colour_of(us))        # what already stood before the move: read once per board, not once per move
    for m in safe_moves(board, sq):
        v, what = duty_after(p, m, us, ctx)
        if pc.piece_type != chess.KING and v >= G.worth(pc):
            cut.append((p.san(m), v, what))
        else:
            kept.append(m)
    _DUTY[key] = (kept, cut)
    return kept, cut


def can_run(board: chess.Board, sq: int, us: str | bool) -> tuple:
    """plan 63.2 (a): can the piece on `sq` RUN? True at the first safe square whose move does not abandon a duty >= its worth (a lazy `duty_safe_moves`: the question is whether any is left, so the
    reading stops there); else False with the list of the abandoning moves [(SAN, value, what)] -- the piece is pinned to a threat on every safe square. -> (bool, cut)."""
    pc = board.piece_at(sq)
    if pc.piece_type == chess.KING:
        return True, []
    key = ("run", board.board_fen(), board.castling_rights, sq, G.colour_of(us))
    hit = _DUTY.get(key)
    if hit is not None:
        return hit
    me = G.colour_of(us)
    p = G.with_turn(board, pc.color)
    now = G.with_turn(board, me)
    ctx = _best_exchange(now, me)
    cut = []
    out = None
    for m in iter_safe_moves(board, sq):
        v, what = duty_after(p, m, us, ctx, need=G.worth(pc))
        if v >= G.worth(pc):
            cut.append((p.san(m), v, what))
        else:
            out = (True, [])
            break
    if out is None:
        out = (False, cut)
    _DUTY[key] = out
    return out


def duty_text(board: chess.Board, sq: int, us: str | bool) -> str | None:
    """The sentence for the trace: 'their B on h4 is pinned to a threat: every safe move of it lets Ng1+ (check that also threatens mate (Qh7#))' or None when it is not duty-bound."""
    pc = board.piece_at(sq)
    if not duty_bound(board, sq, us):
        return None
    v, what, san = duty(board, sq, us)
    return f"the {pc.symbol().upper()} on {G.name(sq)} is pinned to a threat: any move of it ({san}, the least) lets us {what}"


# ---------------------------------------------------------------------------------------------------------------- plan 63.4: one move, several questions (explanation only)

def traps_set(board: chess.Board, m: chess.Move, us: str | bool, limit: int = 4) -> list:
    """plan 63.4: the replies to our move `m` that LOSE OUTRIGHT to a proved mate: after the reply we have a mate in one (a checkmating move) or a forced MATE IN TWO (`selfmine.bind_mate2`, the
    proved word `mate_in_two`, the move named). The opponent's error as a named omission (2026-09-15). Explanation only, no value. -> [(reply SAN, our move SAN, 'mate in one' / 'mate in two')],
    at most `limit`; none when `m` is itself a check (every reply is then forced)."""
    from catspace.incubator.analyst.discovery import selfmine as SM
    me = G.colour_of(us)
    c = board.copy(stack=False)
    c.push(m)
    out = []
    if c.is_check():
        return out
    for r in list(c.legal_moves):
        d = c.copy(stack=False)
        san = c.san(r)
        d.push(r)
        if d.is_check() or not any(True for _ in d.legal_moves):
            continue
        mate1 = next((d.san(x) for x in d.legal_moves if d.gives_check(x) and _mates(d, x)), None)
        if mate1:
            out.append((san, mate1, "mate in one"))
        else:
            got = SM.bind_mate2(d.fen(), "w" if me else "b")
            if got:
                (x, y), = got[0]["pairs"]
                out.append((san, d.san(chess.Move.from_uci(x + y)), "mate in two"))
        if len(out) >= limit:
            break
    return out


def _mates(d: chess.Board, x: chess.Move) -> bool:
    d.push(x)
    try:
        return d.is_checkmate()
    finally:
        d.pop()


def explain_move(board: chess.Board, m: chess.Move, answers: list, us: str | bool) -> str:
    """plan 63.4: the decision line for a move that is a route of SEVERAL root questions: 'Bf2: saves our B on e3; attacks their B on h4 (pinned to the mate on h7); sets ...Bxf2? Nxf2 (mate on h7)'.
    `answers` = the root questions' texts the move answers (the caller lists them from the claims whose first move is `m`); the pin is named by the duty word for a target piece of theirs ('win their B on h4');
    the traps set are `traps_set`. Explanation only: it changes no value and no choice."""
    me = G.colour_of(us)
    parts = []
    after = board.copy(stack=False)
    after.push(m)
    for t in answers:
        text = t
        mm = re.search(r"their ([KQRBNP]) on ([a-h][1-8])", t)
        if mm and (pc := after.piece_at(chess.parse_square(mm.group(2)))) is not None and pc.color != me and DUTY:
            dt = duty_text(after, chess.parse_square(mm.group(2)), me)
            if dt:
                text = f"{t} -- {dt}"
        parts.append(text)
    traps = traps_set(board, m, me)
    if traps:
        parts.append("traps set: " + ", ".join(f"...{san}? {x} ({what})" for san, x, what in traps))
    return f"{board.san(m)}: " + "; ".join(parts)


def no_safe_move(board: chess.Board, m: chess.Move, stake: int, us: str | bool) -> str | None:
    """plan 61 A4 (Lean `Ideas.Forcing.noSafeMove`; zugzwang as 'no safe move'): the quiet move `m` is forcing iff after it EVERY legal reply of theirs loses at least the stake by rule: after
    the reply, the best static exchange of ours on a piece of theirs (the piece that moved or the duty it left -- the stake was not reachable before the reply, `m` being not forcing) less
    what the reply itself captured. A reply that checks us, promotes, or loses less makes the move not forcing (a free tempo move). -> "no safe move: every reply loses >= stake (cheapest: ...)" or None."""
    if stake >= STAKE_MATE:
        return None
    me = G.colour_of(us)
    c = board.copy(stack=False)
    c.push(m)
    replies = list(c.legal_moves)
    if c.is_check() or not replies:
        return None
    cheapest = None
    for r in replies:
        d = c.copy(stack=False)
        gain = 1 if c.is_en_passant(r) else (G.worth(c.piece_at(r.to_square)) if c.is_capture(r) else 0)
        san = c.san(r)
        d.push(r)
        if d.is_check() or r.promotion:
            return None
        best, sq = _threat_square(G.with_turn(d, me), me)
        loss = best - gain
        if loss < stake:
            return None
        if cheapest is None or loss < cheapest[0]:
            cheapest = (loss, san, f"{d.san(next(x for x in d.legal_moves if x.to_square == sq))} wins {best}" if sq is not None and any(x.to_square == sq for x in d.legal_moves) else f"wins {best}")
    return f"no safe move: every reply loses >= {stake} ({board.san(m)}: cheapest {cheapest[1]} loses {cheapest[0]}: {cheapest[2]})"


# ---------------------------------------------------------------------------------------------------------------- plan 61 stage C: the cost of a route (Lean Ideas.Forcing.routeCost)

_COST: dict = {}
COST_CANDIDATES = 3       # the forcing moves (fewest replies first) whose parries `node_cost` counts: a bounded scan, the cost is an ORDER of reading, never a value
G.CLEAR_HOOKS.append(_COST.clear)


def node_cost(board: chess.Board, stake: int, us: str | bool) -> tuple:
    """plan 61 C1: THE COST OF AN UNREAD NODE (our move on `board`) = the number of the parries of OUR BEST forcing move there (best = the fewest parries: the cheapest to read in full); 0 when
    a move of ours mates (a leaf closed by rule) or when we have no forcing move (nothing to follow). One forcing scan and one `parries` call per forcing move. -> (cost, the move's SAN or None)."""
    me = G.colour_of(us)
    key = (board.board_fen(), board.turn, board.castling_rights, board.ep_square, stake, me)
    hit = _COST.get(key)
    if hit is not None:
        return hit
    b = G.with_turn(board, me) if board.turn != me else board
    best = (0, None)
    seen = False
    for m in b.legal_moves:
        if b.gives_check(m):
            c = b.copy(stack=False)
            c.push(m)
            if c.is_checkmate():
                _COST[key] = (0, b.san(m))
                return _COST[key]
    cands = []
    for m in b.legal_moves:
        why = forcing_why(b, m, stake, me, mate_threats=stake >= STAKE_MATE)
        if why:
            after = b.copy(stack=False)
            after.push(m)
            cands.append((after.legal_moves.count(), m.uci(), m, why, after))
    cands.sort(key=lambda t: t[:2])
    for _n, _u, m, why, after in cands[:COST_CANDIDATES]:      # (the parries are at most the legal replies: the forcing moves with the fewest replies are the ones read, a bounded scan)
        th = threat_of(b, m, why, me)
        n = _n if th is None else len(parries(after, th, stake, me)[0])
        if not seen or n < best[0]:
            best, seen = (n, b.san(m)), True
    _COST[key] = best
    return best


def route_cost(board: chess.Board, m: chess.Move, stake: int, us: str | bool, why: str | None = None, limit: int | None = None) -> tuple:
    """plan 61 C1 (Lean `Ideas.Forcing.routeCost`): cost(route) = 1 + the sum, over the parries of the route's threat, of the cost of the unread child (`node_cost` on the board after the parry).
    -> (cost, [(parry SAN, child cost)]). A route with no threat object counts every legal reply as a parry. `limit`: stop pricing once the cost is above it (the result is then a lower bound)."""
    me = G.colour_of(us)
    if why is None:
        why = forcing_why(board, m, stake, me, mate_threats=stake >= STAKE_MATE)
    th = threat_of(board, m, why, me)
    after = board.copy(stack=False)
    after.push(m)
    par = list(after.legal_moves) if th is None else parries(after, th, stake, me)[0]
    kids, total = [], 1
    for r in par:                                                       # (every parry priced exactly, as in Lean: an extrapolated cost is not a lower bound, and the judge cuts on a lower bound -- the discovery position lost Nf6 to it)
        d = after.copy(stack=False)
        san = after.san(r)
        d.push(r)
        kids.append((san, node_cost(d, stake, me)[0]))
        total += kids[-1][1]
        if limit is not None and total > limit:                          # 64.3 (speed only): the cost is already above what the judge would read (a LOWER BOUND, the route is cut anyway): the rest of the parries are not priced
            return total, kids
    return total, kids


MOVES_CACHE = True        # plan 54 (B5.3): `_moves` answers the same (pieces, castling rights, side, skip) once per root question (graph.clear_caches clears it)
_MOVES: dict = {}


def _moves(board: chess.Board, us: str | bool, skip_from: int | None = None) -> list:
    if MOVES_CACHE:
        # the counterfactual board has no history and no e.p. square (graph.with_turn), so its legal moves are a function of these fields alone
        key = (board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings, board.occupied_co[True], board.occupied_co[False],
               board.castling_rights, G.colour_of(us), skip_from)
        hit = _MOVES.get(key)
        if hit is None:
            b = G.with_turn(board, G.colour_of(us))
            hit = _MOVES[key] = tuple(m for m in b.legal_moves if m.from_square != skip_from and not b.is_castling(m))
        return list(hit)
    b = G.with_turn(board, G.colour_of(us))
    return [m for m in b.legal_moves if m.from_square != skip_from and not b.is_castling(m)]


G.CLEAR_HOOKS.append(_MOVES.clear)


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


def _still(q: Question, board: chess.Board, us: str | bool, m: chess.Move, root: Question, before: set | None = None) -> bool:
    """Is the fix `q` NOT done by m? A fix is a move that does what the fix names and introduces no new defect; the disappearance of a defect for another
    reason is not a fix (52.11). For a mate root, on the counterfactual after m: (a) the goal's positive condition holds (cover: a piece of ours attacks the
    flight square; guard: the checker's square is attacked by another piece of ours; remove: the blocker has left) and (b) every defect after m was a defect
    before m, less the fixed one; 'no check' is never allowed. `before` = the defects of `board` less the fixed one, computed once by the caller
    (52.20 (2): they were rescanned for every candidate move)."""
    c = board.copy(stack=False)
    c.push(m)
    me = G.colour_of(us)
    if root.goal == "mate":
        pl = placement_of(root)
        after = {d.key()[:2] for d in _mate_defects(c, me, pl)}
        if before is None:
            before = _defects_before(q, board, me, root)
        if q.defect_key[:2] in after or any(k[0] == "no check" for k in after) or not after <= before:
            return True
        return not _goal_done(q, future_board(c, pl, not me), me)
    if root.goal == "win":
        if c.is_check():                       # our move checks: the board cannot be turned back to us (their king would be in check on our move), so the defect
            return not G.attackers_of(c, root.pattern["target"], me)       # is read as 'attacked or not'; the binder alone decides what the route is worth
        return bool(_material_defects(G.with_turn(c, me), me, root.pattern["target"]))
    return True


def _defects_before(q: Question, board: chess.Board, me: bool, root: Question) -> set:
    """The mate root's defects on `board`, less the one `q` fixes: what `_still` compares the counterfactual after a candidate move against."""
    return {d.key()[:2] for d in mate_defects(board, me, placement_of(root))} - {q.defect_key[:2]}


def routes_of(q: Question, board: chess.Board, us: str | bool, root: Question | None = None) -> list:
    """Our moves that supply the fix `q` names AND are forcing, each re-read on the counterfactual after the move (the defect gone). 'guard' / 'cover' / 'attack':
    any move but the pattern piece's own and the blockers'; 'remove': the blocker's own moves. Moves that fix without forcing go to q.unforced."""
    root = root or q
    skip = root.pattern.get("frm")
    me = G.colour_of(us)
    before = _defects_before(q, board, me, root) if root.goal == "mate" else None       # 52.20 (2): once per route list
    cands = []
    for m in _moves(board, us, skip):
        is_blocker_move = m.from_square == q.pattern.get("blocker")
        if q.goal == "remove" and not is_blocker_move:
            continue
        if q.goal in ("guard", "cover") and m.from_square in root.pattern.get("blockers", ()):
            continue
        cands.append(m)
    got = _routes_rust(q, board, us, root, cands, before) if ROUTES_RUST else None
    if got is None:
        out, unforced = [], []
        for m in cands:
            if _still(q, board, us, m, root, before):
                continue
            why = _forcing_why_py(board, m, q.stake, us)
            if why:
                out.append(Route(m, board.san(m), why))
            else:
                unforced.append(board.san(m))
    else:
        out, unforced = got
    q.unforced = unforced
    if REGRESS_LOG is not None:
        REGRESS_LOG.append(("routes", q.text, [(r.san, r.why) for r in out], list(unforced)))
    return out


def _routes_rust(q: Question, board: chess.Board, us: str | bool, root: Question, cands: list, before: set | None):
    """54 (B2): the `_still` filter and the forcing reasons of the whole candidate list in two Rust calls -> (routes, unforced SAN) or None (the Python loop answers)."""
    me = G.colour_of(us)
    if board.turn != me or root.goal not in ("mate", "win") or not cands:
        return None
    if root.goal == "mate" and q.defect_key is None:
        return None
    p = q.pattern
    dk = tuple(q.defect_key[:2]) if root.goal == "mate" else ("", None)
    spec = (root.goal, q.goal, dk, [(pc.symbol(), at, frm) for pc, at, frm in placement_of(root)] if root.goal == "mate" else [],
            sorted(before, key=repr) if before is not None else [], p.get("target"), p.get("frm"), p.get("blocker"), root.pattern.get("target"))
    try:
        stills = _A.still_routes(G.fen_of(board), me, [m.uci() for m in cands], spec)
    except ValueError:
        return None
    keep = [m for m, (_, still) in zip(cands, stills) if not still]
    whys = _forcing_batch(board, keep, q.stake, us) if keep else []
    if whys is None:
        return None
    out, unforced = [], []
    for m, why in zip(keep, whys):
        if why:
            out.append(Route(m, board.san(m), why))
        else:
            unforced.append(board.san(m))
    return out, unforced


# ---------------------------------------------------------------------------------------------------------------- fixes (sub-questions)

_ids = [0]


def new_question(goal: str, stake: int, pattern: dict, parent: Question | None = None, **kw) -> Question:
    _ids[0] += 1
    return Question(id=_ids[0], goal=goal, stake=stake, pattern=pattern, parent=parent.id if parent else None, depth=parent.depth + 1 if parent else 0, **kw)


def reset_ids() -> None:
    _ids[0] = 0


def fixes_of(defect: Defect, q: Question, board: chess.Board, us: str | bool) -> list:
    """The sub-questions a defect generates, with their routes read against `board` (the real position, us to move)."""
    out = _fixes_of(defect, q, board, us)
    if REGRESS_LOG is not None:
        REGRESS_LOG.append(("fixes", defect.kind, defect.sq, [(f.goal, f.text, f.status, [(r.san, r.why) for r in f.routes], list(f.unforced), list(f.why)) for f in out]))
    return out


def _fixes_of(defect: Defect, q: Question, board: chess.Board, us: str | bool) -> list:
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
            # 52.22 (2): the twin of "remove the defender with tempo": ADD AN ATTACKER of the target with tempo -- a move after which the exchange on the target wins (a threat)
            a = new_question("cover", q.stake, dict(q.pattern), q, defect_key=defect.key(), text=f"add an attacker of {G.name(defect.sq)} with tempo")
            a.routes = routes_of(a, board, us, root)
            out.append(a)
        out += trap_fix(q, defect, board, us)
        out += discovery_fix(q, defect, board, us)
        out += skewer_fix(q, defect, board, us)
    for f in out:
        q.fixes.append(f)
    return out


SKEWER_FIX = True                                               # 52.23 (4) switch (bisect)


def _rust_routes(fn, board: chess.Board, us: str | bool, *args):
    """54 (B4): -> [Route] from an analyst_core route reader `fn(fen, us_white, *args)`, or None (the Python path answers: a position shakmaty cannot hold, or the wrong side to move)."""
    me = G.colour_of(us)
    if board.turn != me:
        return None
    try:
        got = fn(G.fen_of(board), me, *args)
    except ValueError:
        return None
    out = []
    for u, why in got:
        m = chess.Move.from_uci(u)
        out.append(Route(m, board.san(m), why))
    return out


def skewer_routes(board: chess.Board, us: str | bool, t: int) -> list:
    """52.23 (4): THE SKEWER AS A NAMED FIX of a win defect: a check by a slider of ours whose line runs through their king to the target `t`; the king must answer the check,
    and when it steps off the line the slider attacks `t` (Lean `Ideas.Skewer.skewer_attack`; the capture of the checker and the interposition are read by the chase as replies)."""
    got = _rust_routes(_A.skewer_routes, board, us, t) if SKEWER_RUST else None
    if got is not None:
        return got
    me = G.colour_of(us)
    b = G.with_turn(board, me)
    out = []
    for m in b.legal_moves:
        if b.is_castling(m) or m.to_square == t:
            continue
        c = b.copy(stack=False)
        c.push(m)
        if not c.is_check():
            continue
        s2 = G.skewer_after(c, me, t)
        if s2 == m.to_square:
            pc = board.piece_at(t)
            out.append(Route(m, b.san(m), f"check; skewer: through their K on {G.name(c.king(not me))} to the {pc.symbol().upper()} on {G.name(t)} (Ideas.Skewer.skewer_attack)"))
    return out


def drive_onto_routes(board: chess.Board, us: str | bool, t: int, stake: int) -> list:
    """52.23 (4b): DRIVE THEIR KING ONTO THE SKEWER LINE -- a check after which EVERY king reply stands on a line a slider of ours can then check along, through the king
    to `t` (a skewer route exists on the board after each king reply); the other replies (take the checker, interpose) are read by the chase. Two moves: the fix says `steps` 2."""
    got = _rust_routes(_A.drive_onto_routes, board, us, t, stake) if SKEWER_RUST else None
    if got is not None:
        return got
    me = G.colour_of(us)
    b = G.with_turn(board, me)
    out = []
    for m in b.legal_moves:
        if b.is_castling(m) or m.to_square == t:
            continue
        c = b.copy(stack=False)
        c.push(m)
        if not c.is_check() or G.skewer_after(c, me, t) == m.to_square:
            continue
        kings = [r for r in c.legal_moves if r.from_square == c.king(not me)]
        if not kings:
            continue
        ok = True
        for r in kings:
            d = c.copy(stack=False)
            d.push(r)
            if d.piece_at(t) is None or d.piece_at(t).color == me or not skewer_routes(d, us, t):
                ok = False
                break
        if ok:
            out.append(Route(m, b.san(m), f"check; drives their king onto a skewer line to {G.name(t)} (every king reply, then a check through it: Ideas.Skewer.skewer_attack)"))
    return out


def skewer_fix(q: Question, defect: Defect, board: chess.Board, us: str | bool) -> list:
    """52.23 (4): the sub-question `skewer t: a check through their king` of a win defect ('not attacked', 'defended') -- offered when such a check exists."""
    if defect.kind not in ("not attacked", "defended") or not SKEWER_FIX:
        return []
    t = defect.sq
    out = []
    rs = skewer_routes(board, us, t)
    rs2 = None
    if REGRESS_LOG is not None:
        rs2 = drive_onto_routes(board, us, t, q.stake)
        REGRESS_LOG.append(("skewer", t, [(r.move.uci(), r.san, r.why) for r in rs], [(r.move.uci(), r.san, r.why) for r in rs2]))
    if rs:
        f = new_question("cover", q.stake, dict(q.pattern), q, defect_key=defect.key(), text=f"skewer {G.name(t)}: a check through their king, the line runs on to it")
        f.routes = rs
        out.append(f)
    rs = rs2 if rs2 is not None else drive_onto_routes(board, us, t, q.stake)
    if rs:
        f = new_question("cover", q.stake, dict(q.pattern, steps=2), q, defect_key=defect.key(), text=f"drive their king onto a skewer line to {G.name(t)} (two moves: the check, then the skewer)")
        f.routes = rs
        out.append(f)
    return out


def discovery_routes(board: chess.Board, us: str | bool, t: int, stake: int) -> list:
    """52.18 (2): THE DISCOVERY AS A NAMED FIX. For the square t: every blocker of ours that stands in front of a rear slider of ours aimed at t (`G.discoverers`) and leaves the line with
    tempo (`forcing_why`: a check, a capture, or a threat that survives the capture of the moved piece) -- the vacated line raises the attacker count on t by one. Each route says so."""
    me = G.colour_of(us)
    b = G.with_turn(board, me)
    out = []
    for s1, s2, rest, almost in G.discoverers(b, me, t):
        line = set(chess.SquareSet(chess.between(s2, t))) | {t, s2}
        for m in b.legal_moves:
            if m.from_square != s1 or m.to_square in line or b.is_castling(m):
                continue
            why = forcing_why(b, m, stake, us)
            if why:
                out.append(Route(m, b.san(m), f"{why}; discovery: {G.discovery_text(b, s1, s2, rest, t)}"))
    return out


def discovery_fix(q: Question, defect: Defect, board: chess.Board, us: str | bool) -> list:
    """52.18 (2): the sub-question `discover an attack on t` of a win defect ('not attacked', 'defended') -- offered when a battery of ours is aimed at the target."""
    t = q.pattern.get("target")
    if t is None:
        return []
    routes = discovery_routes(board, us, t, q.stake)
    if not routes:
        return []
    f = new_question("cover", q.stake, dict(q.pattern), q, defect_key=defect.key(), text=f"discover an attack on {G.name(t)}: a battery of ours aims there, its front piece leaves with tempo")
    f.routes = routes
    return [f]


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


def _trap_rust(board: chess.Board, us: str | bool, target: int):
    """54 (B4): -> (n, trap routes, fork routes) from `analyst_core.trap_fix_routes`, or None (the Python path answers: a position shakmaty cannot hold, or the wrong side to move)."""
    me = G.colour_of(us)
    if board.turn != me:
        return None
    try:
        n, traps, forks = _A.trap_fix_routes(G.fen_of(board), me, target)
    except ValueError:
        return None
    t_routes = [Route(m, board.san(m), why, trap=target) for m, why in ((chess.Move.from_uci(u), w) for u, w in traps)]
    f_routes = [Route(m, board.san(m), why, fork=target) for m, why in ((chess.Move.from_uci(u), w) for u, w in forks)]
    return n, t_routes, f_routes


def trap_fix(q: Question, defect: Defect, board: chess.Board, us: str | bool) -> list:
    """52.10 / 52.12: the fixes `take its squares` (a piece that can run has safe squares now; the QUIET moves after which it has none) and `fork` (a check that
    also attacks it). A quiet trap costs a tempo (they get a free reply) unless the piece is also attacked; the binder reads both over every reply."""
    target = q.pattern["target"]
    pc = board.piece_at(target)
    if pc is None or pc.piece_type == chess.KING:
        return []
    got = _trap_rust(board, us, target) if TRAP_RUST else None
    cut = []
    if got is None:
        n = G.mobility_of(G.with_turn(board, pc.color), target)
        routes = trap_routes(board, us, target) if n else []
        if DUTY and n and routes:                                              # plan 63.2: the piece cannot run when every safe square is reached by a move that abandons a duty >= its worth
            run, cut = can_run(board, target, us)
            if not run:
                n, routes = 0, []
        forks = fork_routes(board, us, target)
    else:
        n, routes, forks = got
        if DUTY and n and routes:                                              # 63.2 after the Rust read (only when there are trap routes to drop: the duty can only remove the question): can the piece RUN? not if every safe square is reached by a move that abandons a duty >= its worth (the routes
            run, cut = can_run(board, target, us)                              # stay the Rust ones: a quiet move after which no square is safe even without the duty traps it a fortiori)
            if not run:
                n, routes = 0, []
    if REGRESS_LOG is not None:
        REGRESS_LOG.append(("trap", target, n or 0, [(r.san, r.why, r.trap, r.fork) for r in routes], [(r.san, r.why, r.trap, r.fork) for r in forks]))
    out = []
    if routes:
        t = new_question("cover", q.stake, dict(q.pattern), q, defect_key=defect.key(),
                         text=f"take its squares: the {pc.symbol().upper()} on {G.name(target)} can run (safe squares: {n}), the quiet moves after which it has none")
        t.routes = routes
        if any(r.why.endswith("quiet") for r in routes):
            t.why.append("a quiet trap costs a tempo (they get a free reply) unless the piece is also attacked; the depth-two schema reads it over every reply")
        out.append(t)
    if forks:
        f = new_question("cover", q.stake, dict(q.pattern), q, defect_key=defect.key(),
                         text=f"fork: check and attack the {pc.symbol().upper()} on {G.name(target)}, one reply")
        f.routes = forks
        out.append(f)
    return out


# ---------------------------------------------------------------------------------------------------------------- the reach (52.12)

def slide_region(sq: int, occupied: int) -> chess.SquareSet:
    """52.21 (1): the squares a queen on `sq` scans on a board with this occupancy: every ray up to and including its first blocker (a rook, a bishop or a queen on `sq` reads a part of it). What
    an attack, a pin or a check on `sq` can depend on: a square outside it cannot change what a piece on `sq` attacks or is pinned by."""
    return chess.SquareSet(chess.BB_DIAG_ATTACKS[sq][chess.BB_DIAG_MASKS[sq] & occupied] | chess.BB_RANK_ATTACKS[sq][chess.BB_RANK_MASKS[sq] & occupied]
                           | chess.BB_FILE_ATTACKS[sq][chess.BB_FILE_MASKS[sq] & occupied])


def reach_steps(board: chess.Board, us: str | bool, frm: int, target: int) -> list:
    """52.12: the first steps of the shortest (2-move) paths of the piece on `frm` to `target`: its legal moves to a square t from which it attacks `target`, read on
    the board with the piece lifted from `frm` (blocked squares as the rules say). [] when `target` is a one-move square or out of two moves."""
    return reach_steps_read(board, us, frm, target)[0]


def attack_lines(board: chess.Board, squares) -> frozenset:
    """52.21 (1): every square that could hold an attacker of one of `squares` on `board`, read for its occupant: the rays from each square up to and including the first blocker (a slider, a pawn or
    the king next to it) and the knight squares. What the attackers of a square, and so a defect read on a counterfactual `board`, depend on besides the square itself."""
    out = set()
    for x in squares:
        out.update(slide_region(x, board.occupied))
        out.update(chess.SquareSet(chess.BB_KNIGHT_ATTACKS[x]))
    return frozenset(out)


class ReachTable:
    """52.21 (1): the two-move reach of ONE piece, listed once: its legal moves, and for each landing square t the squares it attacks there on the board with the piece moved. The steps to any
    target are then a lookup (the scan asked the same legal moves and the same counterfactuals again for every target). `reads` = the squares the answers depended on, for their occupant only:
    the piece and what it attacks from `frm` (up to and including the first blocker: its moves), our king and, when the piece stands on a line through it, that line out to the first blocker
    beyond the piece (a pin: the move may not leave our king in check; our king is not in check on the board, so no other line matters), each landing square, and for every target asked, the
    squares between a landing square and the target when they are on a line for this piece type (the attack could be blocked there)."""

    def __init__(self, board: chess.Board, us: str | bool, frm: int):
        self.board, self.frm = board, frm
        self.pc = pc = board.piece_at(frm)
        reads = {frm}
        self.landings: list = []                                         # [(move, the squares attacked from its landing square)]
        self.one_move: set = set()                                       # the squares the piece reaches in one move
        if pc is not None and pc.piece_type not in (chess.KING, chess.PAWN):
            colour = G.colour_of(us)
            own = G.with_turn(board, colour)
            reads.update(chess.SquareSet(board.attacks_mask(frm)))
            ksq = board.king(colour)
            if ksq is not None:
                reads.add(ksq)
                line = chess.BB_RAYS[ksq][frm]
                if line:                                                 # the piece stands on a line through our king: what stands behind it on that line decides whether it is pinned
                    reads.update(chess.SquareSet(slide_region(ksq, board.occupied & ~chess.BB_SQUARES[frm]) & line))
            mine = [m for m in own.legal_moves if m.from_square == frm]          # 52.20 (3): the legal moves listed once
            self.one_move = {m.to_square for m in mine}
            for m in mine:
                if m.promotion or own.is_castling(m):
                    continue
                c = G.with_piece(board, pc, m.to_square, frm)
                reads.add(m.to_square)
                self.landings.append((m, c.attacks_mask(m.to_square)))
        self._reads = reads

    @property
    def reads(self) -> frozenset:
        return frozenset(self._reads)

    def steps(self, target: int) -> list:
        """The legal first steps from which the piece attacks `target` ([] when `target` is a one-move square or out of two moves, or holds a piece of the piece's own colour)."""
        pc = self.pc
        if pc is None or pc.piece_type in (chess.KING, chess.PAWN) or target in self.one_move:
            return []
        self._reads.add(target)
        o = self.board.piece_at(target)
        if o is not None and o.color == pc.color:
            return []
        out = []
        for m, at in self.landings:
            if self._line(m.to_square, target):
                self._reads.update(chess.SquareSet(chess.between(m.to_square, target)))
            if at & chess.BB_SQUARES[target]:
                out.append(m)
        return out

    def _line(self, t: int, target: int) -> bool:
        """Whether the piece type on `t` would attack `target` on an empty board (then the squares between decide)."""
        pt = self.pc.piece_type
        if pt == chess.KNIGHT:
            return False                                                 # nothing between: a knight attacks or not by geometry
        hit = 0
        if pt in (chess.BISHOP, chess.QUEEN):
            hit |= chess.BB_DIAG_ATTACKS[t][0]
        if pt in (chess.ROOK, chess.QUEEN):
            hit |= chess.BB_RANK_ATTACKS[t][0] | chess.BB_FILE_ATTACKS[t][0]
        return bool(hit & chess.BB_SQUARES[target])


def reach_steps_read(board: chess.Board, us: str | bool, frm: int, target: int) -> tuple:
    """`reach_steps` and the squares it read (an occupancy-only read set, see `ReachTable`). -> (the steps, the squares read)."""
    t = ReachTable(board, us, frm)
    steps = t.steps(target)
    return steps, t.reads


# ---------------------------------------------------------------------------------------------------------------- read sets (52.12)

def reads_of(q: Question, board: chess.Board) -> frozenset:
    """The squares whose occupancy or attack counts the question read on `board`: the placement squares, the target and its attackers / guards, the king and its
    flight squares, the squares of the defects and of the pieces involved, the line between the checker and the king, the squares of the fixes' routes."""
    p, out = q.pattern, set()
    for _piece, at, frm in placement_of(q):
        out.add(at)
        if frm is not None:
            out.add(frm)
    for key in ("king", "target", "blocker", "xray", "pawn", "frm", "at"):
        if p.get(key) is not None:
            out.add(p[key])
    for d in q.defects:
        if d.sq is not None:
            out.add(d.sq)
        out.update(d.by)
    if q.goal == "mate" and p.get("king") is not None:
        k = p["king"]
        out.update(chess.SquareSet(chess.BB_KING_ATTACKS[k]))
        if p.get("at") is not None:
            out.update(chess.SquareSet(chess.between(p["at"], k)))
    if q.goal == "win" and p.get("target") is not None:
        out.update(board.attackers(chess.WHITE, p["target"]))
        out.update(board.attackers(chess.BLACK, p["target"]))
    for r in all_routes(q):
        out.add(r.move.from_square)
        out.add(r.move.to_square)
    return frozenset(out)


def diff_squares(before: chess.Board, after: chess.Board) -> frozenset:
    """The squares whose piece or whose attacker counts (either colour) differ between two boards: what a move and its reply changed."""
    out = set()
    for s in chess.SQUARES:
        if (before.piece_at(s) != after.piece_at(s) or len(before.attackers(chess.WHITE, s)) != len(after.attackers(chess.WHITE, s))
                or len(before.attackers(chess.BLACK, s)) != len(after.attackers(chess.BLACK, s))):
            out.add(s)
    return frozenset(out)


def question_key(q: Question) -> tuple | None:
    """What identifies a root question across positions (the same goal asked about the same squares)."""
    p = q.pattern
    if q.goal == "mate":
        return ("mate", p.get("sym"), p.get("at"), p.get("frm"), p.get("king"), p.get("move"))
    if q.goal == "win":
        return ("win", p.get("target"))
    if q.goal == "promote":                                     # 52.16 (4): the promotion of the pawn on that square
        return ("promote", p.get("pawn"))
    return None


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


def promote_defects(board: chess.Board, sq: int, lift: bool = False) -> tuple:
    """What stands between the pawn on `sq` (any colour) and a queen, read with the Rust words (`pushes`, `ahead`, `promo`, `attackers`, `defenders`, `dist`)
    from its owner's side: [Defect] and the facts {pushes, path, promo, king_dist, king_sq}. Kinds: 'blocked' (a piece on the path), 'path attacked' (a path
    square attacked by them more than we guard it), 'promo square attacked' (the same on the promotion square), 'stopper' (a piece of theirs that reaches the
    promotion square or the path within the pawn's pushes: the king by `dist`, a slider / knight by its static-board `reach`). No defect = unstoppable by the
    tempo arithmetic (the owner moves first; the other side's pieces move for free while the pawn runs)."""
    pc = board.piece_at(sq)
    own = pc.color
    kv = None
    if lift and RC.LIFT and board.turn == own and (board.occupied_co[not own] & (board.rooks | board.bishops | board.queens | board.knights)):
        kv = RC.king_verdict(board, sq)                 # plan 62.1c: with a piece of theirs beyond king and pawns the verdict judges only the KING, by the proved analysis; the pieces are replies the chase reads
    n = G.word(board, ("pushes", G.name(sq)), own)
    path = pawn_path(board, sq, n)
    promo = path[-1]
    facts = dict(pushes=n, path=path, promo=promo, king_dist=None, king_sq=board.king(not own))
    out = []
    lifted = []
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
            if kv is not None and not kv[0] and set(by) <= {board.king(not own)}:
                lifted.append(f"{G.name(t)} is attacked only by their king, which is no stopper ({kv[1]})")
                continue
            out.append(Defect(kind, t, by, f"{G.name(t)} is attacked by them {a}x and guarded by us {d}x"))
    ek = facts["king_sq"]
    if kv is not None:
        facts["lifted"] = lifted
        if kv[0]:
            out.append(Defect("stopper", promo, [ek], f"their K stops it: {kv[1]}"))
        else:
            lifted.append(f"their K is no stopper: {kv[1]}")
        facts["king_dist"] = G.word(board, ("dist", "L", ("promo", G.name(sq))), own)
        return out, facts
    if ek is not None:                                  # a king standing on the path is the blocker (already a defect), not a stopper
        kd = G.word(board, ("dist", "L", ("promo", G.name(sq))), own)
        facts["king_dist"] = kd
        if kd <= n and ek not in path:
            w = RC.stopper_walk(board, ek, {promo}, n)               # 62.1b: the stopper's path, named
            out.append(Defect("stopper", promo, [ek], f"their K on {G.name(ek)} reaches {G.name(promo)} in {kd}, the pawn needs {n}" + (f" (path {' '.join(G.name(x) for x in w[1])})" if w else "")))
    for s in chess.SQUARES:
        x = board.piece_at(s)
        if x is None or x.color == own or x.piece_type in (chess.KING, chess.PAWN):
            continue
        k = G.piece_moves_to(board, s, set(path), n)
        if k is not None:
            w = RC.stopper_walk(board, s, set(path), n)
            out.append(Defect("stopper", promo, [s], f"their {x.symbol().upper()} on {G.name(s)} reaches the path in {k} (static board), the pawn needs {n}" + (f" (path {' '.join(G.name(x) for x in w[1])})" if w else "")))
    return out, facts


def drives_off(board: chess.Board, m: chess.Move, x: int, ksq: int) -> bool:
    """52.17 (3): THE KING DRIVEN OFF A SQUARE. `m` is a check after which EVERY legal reply is a move of their king (the one on `ksq`) to a square that does not touch `x`
    (not on it, not next to it): the square is left unattacked by the king whatever it does. 0LRTF: after Rh8+ Kd7, Rd8+ drives the king off c8 (Ke6, Ke7; Kxd8 is
    illegal, c7 guards d8); Rh8+ itself does not (Kd7 touches c8)."""
    if not board.gives_check(m):
        return False
    c = board.copy(stack=False)
    c.push(m)
    reps = list(c.legal_moves)
    return bool(reps) and all(r.from_square == ksq and chess.square_distance(r.to_square, x) >= 2 for r in reps)


def drive_off_routes(board: chess.Board, us: str | bool, x: int, ksq: int) -> list:
    """52.17 (3): our checks that drive their king off `x` (see `drives_off`); read set: their king's squares after the check, the checker's safety."""
    return [Route(m, board.san(m), f"drives the king off {G.name(x)}: every king reply leaves it unattacked by the king") for m in _moves(board, us) if drives_off(board, m, x, ksq)]


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
        if RC.LIFT and defect.by[0] == board.king(not own) and board.turn == own and (board.occupied_co[not own] & (board.rooks | board.bishops | board.queens | board.knights)):        # 62.1c: the king is the stopper by the proved analysis: the moves that make it none are routes
            have = {r.move for r in f.routes}
            for m, why in RC.support_moves(board, own, sq, list(board.legal_moves)):
                if m not in have:
                    f.routes.append(Route(m, board.san(m), why))
            if any(r.why.startswith("makes their king") for r in f.routes):
                f.text += " -- or make the king no stopper"
        if RC.DENY and board.turn == own:                          # 62.1b: DENY THE STOPPER'S PATH (a move after which it cannot stand and survive on a square of its path in time) addresses the defect
            n_ = abs((7 if own == chess.WHITE else 0) - chess.square_rank(sq))
            tg = {defect.sq} if defect.by[0] == board.king(not own) else set(pawn_path(board, sq, n_))
            have = {r.move for r in f.routes}
            for m, why in RC.deny_moves(board, own, sq, defect.by[0], tg, n_, list(board.legal_moves)):
                if m not in have:
                    f.routes.append(Route(m, board.san(m), why))
            if any("denies the path" in r.why for r in f.routes):
                f.text += " -- or deny its path"
    if defect.kind in ("path attacked", "promo square attacked"):             # 52.18 (2): a discovery that guards the square (a battery of ours aimed at it, its front piece leaving with tempo)
        have = {r.move for r in f.routes}
        extra = [r for r in discovery_routes(board, own, defect.sq, q.stake) if r.move not in have]
        f.routes += extra
        if extra:
            f.text += f" -- or discover a guard of {G.name(defect.sq)}"
    ek = board.king(not own)
    if ek is not None and ((defect.kind == "blocked" and defect.sq == ek) or (defect.kind in ("path attacked", "promo square attacked", "stopper") and ek in defect.by)):
        have = {r.move for r in f.routes}
        extra = [r for r in drive_off_routes(board, own, defect.sq, ek) if r.move not in have]
        f.routes += extra
        if extra:
            f.text += f" -- or drive their king off {G.name(defect.sq)} with a check"
    q.fixes.append(f)
    return f
