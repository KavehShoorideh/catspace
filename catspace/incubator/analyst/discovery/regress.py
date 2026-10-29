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

72.8 THE SKELETON GOALS (`skeleton_goals`, end of the module): the slow units are the fixed board, our fast units are imagined where the goal's defects need them,
priced by the roll forward. THE ARRANGEMENT IS A CANDIDATE GOAL; ITS PROOF IS THE CHASE. Stage 1 proves nothing new in Lean: an arrangement is a conjunction of
bearing predicates over a counterfactual board (Lean `Ideas.Vocabulary.attackableIn`; `Ideas.Reach.reachPrice`, price 0 = attackableIn by
`Ideas.Reach.reachPrice_zero_iff`; the mate defects are the hypotheses of `Ideas.checkmate_of_unmeetable_check` read as a list), and it adds routes, roots and a
policy feature only -- it never claims and never kills a question.
"""
from __future__ import annotations

import itertools
import os
import re
from dataclasses import dataclass, field
from typing import NamedTuple

import chess

import analyst_core as _A
from catspace.incubator.analyst.discovery import gdl_chess as GC
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import race as RC
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import regression as RGN      # plan 76.4a: THE REGRESSION -- the defects and the fix filter read off the spec
from catspace.incubator.analyst.discovery.game import CHESS as _GAME              # plan 76.2 (b): the rules read through the Game surface (the judge's parry kinds)

STAKE_MATE = LS.goal_kinds().terminal.stake    # plan 75.2 (a): the stake of the goal that ends the game, read from its GOAL entry (lemmas/mate.json: ONE definition)
PASS_CLAIMS = [0]                             # 52.18 (3): forcing_why answers that were a claim bound after a pass (the OBS counter)
STAKE_PROMOTE = LS.goal_kinds().promotion.stake      # plan 75.2 (a): a queen less the pawn, read from the promotion GOAL entry and the material VALUATION
# (plan 76.4a: DEFECTS_RUST left -- the mate and material defects are the regression's, regression.py over rules/chess.gdl; the hand tables and the plan 54 Rust
#  twins are test oracles, tests/oracles/regress_tables.py)
ROUTES_RUST = True                            # 54 (B2): `forcing_why` for the whole candidate list in one Rust call (the Python path stays); plan 76.4a: the fix filter is the regression's
TRAP_RUST = True                              # 54 (B4): trap_fix's three reads (safe squares, quiet trap routes, fork routes) in one Rust call (the Python path stays)
SKEWER_RUST = True                            # 54 (B4): skewer_routes / drive_onto_routes read by analyst_core (the Python path stays)
REGRESS_LOG = None                            # 54 (B): a list -> every fixes_of / routes_of / forcing_why call appends what it answered (scripts/compare_regress.py diffs these)
# (plan 75.2 (a): PROMOTE_MAX_PUSHES = 3 left for the promotion GOAL entry's `near`, lemmas/promotion.json)


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
    price: tuple | None = None                # 72.3: (material, tempi) of the join this move is the first step of (None: not a join)


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
    joins: list = field(default_factory=list)      # 72.3: the join sub-questions (`joins_of`): QUIET plans, kept OUT of `fixes` -- a join hands them free moves, so it is no forcing fix (tempo_ok, the chase and the fuse never see it)
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


# ---------------------------------------------------------------------------------------------------------------- defects read from the rules: THE REGRESSION (plan 76.4a)
# The defects of a mate question and of a material question are the regression's (discovery/regression.py: the GOAL entry's spec relation -- `mate` on the
# question's counterfactual, `capture` of the target with the exchange as its value -- regressed over rules/chess.gdl's clause bodies, its defects named by the
# store's DEFECT_FIX entries); the hand tables they replace (the audit's D1a / D1b: the defect kinds by name) are test oracles (tests/oracles/regress_tables.py),
# equal on every question of the guard positions (76.4a (b)).

_REG: list = [None]
_RGOALS: dict = {}                                # (goal, side, the question's key) -> its Goal (per root question)
_STORE: list = [None, None, None]                # (the store object, its defect names, its goal relations)


def regressor() -> "RGN.Regressor":
    """plan 76.4a: the process's regression (the spec, the chess binding); its cache and the goals cleared with every fresh root (graph.clear_caches)."""
    if _REG[0] is None:
        B = GC.Binding()
        _REG[0] = RGN.Regressor(RGN.Spec(B.prog, certified=GC.CERTIFIED), B)
    return _REG[0]


def _clear_regression() -> None:
    if _REG[0] is not None:
        _REG[0].clear()
    _RGOALS.clear()


G.CLEAR_HOOKS.append(_clear_regression)


def _store() -> tuple:
    st = LS.load()
    if _STORE[0] is not st:
        _STORE[0], _STORE[1], _STORE[2] = st, LS.defect_names(st), LS.gdl_goals(st)
    return _STORE[1], _STORE[2]


def _goal(kind: str, me: bool, target=None, placement: list = ()) -> "RGN.Goal":
    """The Goal of the store's GOAL entry `kind` for one question (its side; its target; its placements, the key a regression of the same question is validated by)."""
    key = (kind, me, target, tuple((p.symbol(), at, frm) for p, at, frm in placement))
    g = _RGOALS.get(key)
    if g is None:
        B = regressor().B
        g = _RGOALS[key] = RGN.goal_of(kind, _store()[1][kind], B, B.role(me), target=target, unit=LS.goal_kinds().valuation.unit, extra=key[3])
    return g


def _named(reg: "RGN.Regression", x, theirs: bool = False) -> list:
    """The regression's defects by the store's names, as the machinery's Defects (`theirs`: the goal is the other side's, read for the keeper)."""
    return [Defect(n.kind, n.sq, list(n.by), n.why) for n in RGN.name(reg, _store()[0], regressor().B, x, theirs=theirs)]


def threat_defects(board: chess.Board, us: str | bool, target: "int | None" = None) -> list:
    """plan 76.4a: THE DEFECTS OF A KEEP QUESTION -- the OTHER side's goal read on the board, named by the store's names bound `for` theirs: our king in check is
    their terminal goal's positive literal `check` holding (its attackers: one defect per checker); a unit of ours on `target` that hangs is their gain goal holding
    on it (their capturers, the exchange law's value for them)."""
    me = G.colour_of(us)
    if target is None:
        reg = regressor().regress(board, _goal("mate", not me))
    else:
        reg = regressor().regress(board, _goal("win", not me, target=target))
    return _named(reg, board, theirs=True)


def landings(defects: list) -> list:
    """plan 76.4a: the locations of the defects whose name reads a witness at its LANDING (the store's binding `sq` 'to': for the mate goal, the squares their checked
    unit can step to) -- read off the names' bindings, not their words."""
    by = {e["name"]: e["gdl"] for e in _store()[0]}
    return [d.sq for d in defects if by.get(d.kind, {}).get("sq") == "to"]


def mate_defects(board: chess.Board, us: str | bool, placement: list) -> list:
    """What stands between the pattern and a mate (`_mate_defects`); the call is logged when REGRESS_LOG is set."""
    out = _mate_defects(board, us, placement)
    if REGRESS_LOG is not None:
        REGRESS_LOG.append(("mate_defects", board.epd(), G.colour_of(us), [(p.symbol(), at, frm) for p, at, frm in placement], [(d.kind, d.sq, tuple(d.by), d.why) for d in out]))
    return out


def _mate_regression(board: chess.Board, me: bool, placement: list, near: "RGN.Regression | None" = None) -> tuple:
    """The regression of the mate goal on the question's counterfactual (the placements made, their move) -> (the regression, the counterfactual)."""
    cf = future_board(board, placement, not me)
    return regressor().regress(cf, _goal("mate", me, placement=placement), near=near), cf


def _mate_defects(board: chess.Board, us: str | bool, placement: list) -> list:
    """What stands between the pattern and a mate: the regression of the mate goal on the counterfactual (placements made, their move), named. [] = the pattern IS a
    mate (in check, no legal move). The names' order is the store's: the checker takeable, flights, interpositions (or 'no check' alone)."""
    me = G.colour_of(us)
    reg, cf = _mate_regression(board, me, placement)
    return _named(reg, cf)


def mate_defects_on(cf: chess.Board, me: bool) -> list:
    """The defects of an already-built future `cf` (their move), regressed: [] = they are mated."""
    return _named(regressor().regress(cf, _goal("mate", me)), cf)


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


def _material_defects(board: chess.Board, us: str | bool, target_sq: int, near: "RGN.Regression | None" = None) -> list:
    """Why we do not win the piece on `target_sq` now: the regression of the gain goal (capture of the target, us on move, the exchange law its value), named --
    [] when the capture's exchange already wins (at least the valuation's unit); otherwise 'not attacked' (the capture's literal false: nothing of ours attacks
    it) or 'defended' (the capture holds, its exchange short: the attacks of theirs on it read)."""
    me = G.colour_of(us)
    reg = regressor().regress(board, _goal("win", me, target=target_sq), near=near)
    return _named(reg, board)


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


def _game():
    """plan 75.4 (5): the Game instance (imported late: game.py imports graph, which this module's importers load first)."""
    from catspace.incubator.analyst.discovery import game as GM
    return GM.CHESS


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
        cap = _game().push_to(c, a, sq, LS.goal_kinds().valuation, legal=False)      # plan 75.4 (5): the kernel's capture (pseudo-legal: the static exchange; a promotion: the valuation's unit)
        if cap is None:
            break
        c.push(cap)
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
    every reply is met by a mate; bound by `selfmine.bind_mate2`, the premise read in Rust) -- the same word the binder's mate_in_two schema uses, on the board after m with the move passed to them.
    plan 72.7 step 3a: read in Rust on the board's handle in ONE call (`analyst_core.mate2_threat_pos`: the push, the pass, the binder's candidates and atoms -- words.rs); a board
    whose handle shakmaty does not hold is pushed here and read on the handle after the move. The Python body is the test oracle (tests/test_plan72_7_step3.py)."""
    from catspace.incubator.analyst.discovery import base_vocabulary as BV
    me = G.colour_of(us)
    BV.install_terms()
    h = G.pos_of(board)
    if h is not None:
        return _A.mate2_threat_pos(h, m.uci(), me)
    c = board.copy(stack=False)
    c.push(m)
    hc = G.pos_of(c)
    return _A.mate2_after_pos(hc, me) if hc is not None else False


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
        res = _A.forcing_whys_pos(G.handle(board), [m.uci() for m in moves], stake, me, mate_threats)
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
        return STAKE_MATE if SM.bind_mate2(None, "w" if me else "b", pos=G.pos_of(d)) else 0      # plan 72.7 step 3a: on d's handle (no FEN)
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


def duty_after(board: chess.Board, r: chess.Move, us: str | bool, ctx: int | None = None, need: int | None = None, passer: bool = False, mate: bool = True) -> tuple:
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
    mt = _mate_ahead(a, me) if mate else None                             # (mate=False: the GAP side's cheap duty -- stuck_threats -- reads material duties only; mates are their agenda's own claims)
    if mt and _mate_ahead(G.with_turn(board, me), me) is None:             # a mate (or threat of mate) that did not stand before the move
        return STAKE_DUTY_MATE, mt
    if passer:                                                             # plan 72.2 (b), read for the GAP side only (stuck_threats): the move frees a PASSER of ours -- its promote defects (the tempo arithmetic with the priced stopper) are none after the move and were not before: the duty is the promotion (not Lean `dutyAfter`: a word of the tempo arithmetic, never a proof)
        held = board.attacks(r.from_square) | chess.SquareSet([r.from_square])      # (the cost rule of 2026-10-23: a move can free a passer only if the moving piece stood ON its path or BORE on a path square; geometry first, the priced defects only then)
        for pw in chess.SquareSet(board.pawns & board.occupied_co[me]):
            if a.piece_at(pw) is None or not is_passer(board, pw):
                continue
            n_ = _pushes_of(board, pw, me)
            if n_ < 1 or not (held & chess.SquareSet(pawn_path(board, pw, n_))):
                continue
            st = promote_defects_cached(a, pw, priced=False)                 # the static read after the move first: pricing can only LIFT piece stoppers, so any other defect left settles it without a roll
            ek = a.king(not me)
            if any(d.kind != "stopper" or d.by == [ek] for d in st):
                continue
            if st and promote_defects_cached(a, pw):
                continue
            return STAKE_PROMOTE, f"frees our passer on {G.name(pw)} (unstoppable by the tempo arithmetic after it)"
    return v, f"the exchange wins {v}" if v else "nothing"


def _pushes_of(board: chess.Board, pw: int, own: bool) -> int:
    """The pushes to the promotion rank (7 - rank, or the rank, by colour): geometry only, as the Rust `pushes` (a blocker is the `blocked` defect, not fewer pushes); builds no FEN."""
    r = chess.square_rank(pw)
    return (7 - r) if own else r


_PD: dict = {}
G.CLEAR_HOOKS.append(_PD.clear)
G.CLEAR_HOOKS.append(_A.passer_cache_clear)                                   # plan 72.7 step 3c: the passer family's Rust memos (promote_defects_cached, can_run), cleared with `_PD`


def promote_defects_cached(board: chess.Board, sq: int, priced: "bool | None" = None) -> list:
    """`promote_defects` (no lift) memoised by position: the duty reads it for every candidate move of every unit."""
    key = (board.board_fen(), board.turn, sq, priced)
    hit = _PD.get(key)
    if hit is None:
        hit = _PD[key] = promote_defects(board, sq, priced=priced)[0]
    return hit


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


def can_run(board: chess.Board, sq: int, us: str | bool, passer: bool = False) -> tuple:
    """plan 63.2 (a): can the piece on `sq` RUN? True at the first safe square whose move does not abandon a duty >= its worth (a lazy `duty_safe_moves`: the question is whether any is left, so the
    reading stops there); else False with the list of the abandoning moves [(SAN, value, what)] -- the piece is pinned to a threat on every safe square. -> (bool, cut)."""
    pc = board.piece_at(sq)
    if pc.piece_type == chess.KING:
        return True, []
    key = ("run", board.board_fen(), board.castling_rights, sq, G.colour_of(us), passer)
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
        v, what = duty_after(p, m, us, ctx, need=G.worth(pc), passer=passer, mate=not passer)
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
            got = SM.bind_mate2(None, "w" if me else "b", pos=G.pos_of(d))                # plan 72.7 step 3a: on d's handle (no FEN)
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
    """plan 72.3: a JOIN route (the first step of a priced reach, `join_fixes`) pays its tempi beyond the first as unread nodes (each is a free move they get): + tempi - 1 on the plan 61 cost."""
    cost, kids = _route_cost(board, m, stake, us, why, limit)
    j = _JOIN.get((board.board_fen(), m.uci()))
    return (cost + max(0, j[1] - 1), kids) if j is not None else (cost, kids)


def _route_cost(board: chess.Board, m: chess.Move, stake: int, us: str | bool, why: str | None = None, limit: int | None = None) -> tuple:
    """plan 61 C1 (Lean `Ideas.Forcing.routeCost`): cost(route) = 1 + the sum, over the parries of the route's threat, of the cost of the unread child (`node_cost` on the board after the parry).
    -> (cost, [(parry SAN, child cost, what the parry is)]). A route with no threat object counts every legal reply as a parry. `limit`: stop pricing once the cost is above it (the result is then a lower bound).
    plan 76.2 (b): what each parry IS (a check, a capture, a quiet move: the judge prices it at its kind's rate, 72.9 (a)) is read by the Game surface (`Game.reply_kind`), not off its SAN."""
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
        kids.append((san, node_cost(d, stake, me)[0], _GAME.reply_kind(after, r)))
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


def _still(q: Question, board: chess.Board, us: str | bool, m: chess.Move, root: Question, before: set | None = None) -> bool:
    """Is the fix `q` NOT done by m? plan 76.4a: THE REGRESSION'S FIX TEST (regression.fixes) -- a fix is a move after which, in the goal's next state, the defect
    `q` names is gone and no defect is new (52.11: the disappearance of a defect is read by the spec itself -- the witness of the defect, a legal move of theirs, is no
    witness there -- not by a hand-written positive condition). A mate root: the regression of the mate goal on the counterfactual after m (the placements made,
    their move); every defect after m must have been a defect before m, less the fixed one ('no check' never was: it is new). A win root: the gain goal's
    regression after m (us on move; where the exchange law reads no position -- our move gave check -- its head alone decides, the binder reads the rest).
    `before` = the defects of `board` less the fixed one, computed once by the caller (52.20 (2)). The regression read on `board` is the one every candidate's
    is validated against first (regression.Regressor.regress: unchanged answers, the same regression). The hand filter this replaces is a test oracle
    (tests/oracles/regress_tables.py `still`); where they differ (76.4a (b)): a square covered along a line through their king is a fix here, the capture of the
    target itself is not."""
    c = board.copy(stack=False)
    c.push(m)
    me = G.colour_of(us)
    if root.goal == "mate":
        pl = placement_of(root)
        r0, _cf0 = _mate_regression(board, me, pl)
        reg, cf = _mate_regression(c, me, pl, near=r0)
        after = {d.key()[:2] for d in _named(reg, cf)}
        if before is None:
            before = _defects_before(q, board, me, root)
        return q.defect_key[:2] in after or not after <= before
    if root.goal == "win":
        t = root.pattern["target"]
        r0 = regressor().regress(board, _goal("win", me, target=t))
        return bool(_material_defects(c, me, t, near=r0))
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
    goal_sq = root.pattern.get("target") if root.goal == "win" else None
    keep = [m for m in cands if (goal_sq is not None and m.to_square == goal_sq) or not _still(q, board, us, m, root, before)]       # plan 76.4a: the regression's fix test
    #   (2026-10-28, after 76.4a's gate: THE GOAL'S OWN MOVE -- the capture of the target -- is a ROUTE of a win question whether or not it mends a defect: its merit is the exchange law's,
    #    read by the verifier; the hand oracle admitted it by accident (it read the exchange on our own unit there) and the regression rightly says it fixes nothing, but an agenda without a
    #    chase (theirs, side_depth 0) has no other way to the capture: test_64_0's their-claim-after-Qxa8 fell to 0 without it)
    whys = _forcing_batch(board, keep, q.stake, us) if ROUTES_RUST and keep else None       # 54 (B2): the forcing reasons in one Rust call (the Python twin where it answers None)
    if whys is None:
        whys = [_forcing_why_py(board, m, q.stake, us) for m in keep]
    out, unforced = [], []
    for m, why in zip(keep, whys):
        if why:
            out.append(Route(m, board.san(m), why))
        else:
            unforced.append(board.san(m))
    q.unforced = unforced
    if REGRESS_LOG is not None:
        REGRESS_LOG.append(("routes", q.text, [(r.san, r.why) for r in out], list(unforced)))
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
    """54 (B4): -> [Route] from an analyst_core route reader `fn(pos, us_white, *args)` (plan 72.7 step 1: the board's handle, `G.handle`), or None (the Python path answers: a
    position shakmaty cannot hold, or the wrong side to move)."""
    me = G.colour_of(us)
    if board.turn != me:
        return None
    try:
        got = fn(G.handle(board), me, *args)
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
    got = _rust_routes(_A.skewer_routes_pos, board, us, t) if SKEWER_RUST else None
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
    got = _rust_routes(_A.drive_onto_routes_pos, board, us, t, stake) if SKEWER_RUST else None
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
        n, traps, forks = _A.trap_fix_routes_pos(G.handle(board), me, target)
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


_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1))      # (file, rank) steps: the four orthogonal, then the four diagonal directions


def _ray(occupied: int, sq: int, df: int, dr: int) -> list:
    """The squares from `sq` (exclusive) in the direction (df, dr) up to and including the first occupied square."""
    out, f, r = [], chess.square_file(sq) + df, chess.square_rank(sq) + dr
    while 0 <= f < 8 and 0 <= r < 8:
        s = chess.square(f, r)
        out.append(s)
        if occupied >> s & 1:
            break
        f, r = f + df, r + dr
    return out


def move_diff(board: chess.Board, m: chess.Move) -> frozenset:
    """plan 72.10 (1) THE DIFF of the move `m` on `board` (in the chase: their reply on the board after our route). The TOUCHED squares (from, to, and every square whose piece the move changes:
    the castling rook's two squares, the pawn en passant takes); what a piece on a touched square attacks, before or after the move (the moved piece's new lines, the old lines of the piece
    that left or was taken); and every line THROUGH an empty touched square that a slider sees along, before or after the move -- the ray from the touched square up to and including its first
    piece, on the side away from the slider (the lines the move OPENED: the ray beyond `from`, a discovery as `Ideas.Battery` names it; and CLOSED: the ray beyond `to`).
    THE NEGLECT THEOREM (Lean `Ideas.Frame.chess_neglect` over `Game.step`, generic `Ideas.Frame.neglect_step`; the frame form `Ideas.Frame.chess_frame`): a word that reads the pieces ON its squares
    and the attacks ONTO them (`Ideas.FrameLemma`'s two inputs) and whose value differs after the move reads a square of `Ideas.Frame.chessDiff`, of which this set is a superset (each opened or
    closed line is taken whole). Tested against the brute force -- every square whose piece or whose attackers (either colour, as a set) changed -- on 50 random positions
    (tests/test_plan72_10.py). `analyst_core.Pos.diff` is not this: it compares attacker COUNTS, so a swap of one attacker for another on a square is missed."""
    after = board.copy(stack=False)
    after.push(m)
    touched = {m.from_square, m.to_square} | {s for s in chess.SquareSet(board.occupied | after.occupied) if board.piece_at(s) != after.piece_at(s)}
    out = set(touched)
    for X in (board, after):
        occ = X.occupied
        for s in touched:
            if occ >> s & 1:
                out.update(chess.SquareSet(X.attacks_mask(s)))
                continue
            for i, (df, dr) in enumerate(_DIRS):
                back = _ray(occ, s, -df, -dr)                     # the slider that would see THROUGH s comes from the other side
                if not back or not occ >> back[-1] & 1:
                    continue
                pc = X.piece_at(back[-1])
                if pc.piece_type == chess.QUEEN or pc.piece_type == (chess.ROOK if i < 4 else chess.BISHOP):
                    out.update(_ray(occ, s, df, dr))
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


def promote_defects(board: chess.Board, sq: int, lift: bool = False, priced: "bool | None" = None) -> tuple:
    """What stands between the pawn on `sq` (any colour) and a queen, read with the Rust words (`pushes`, `ahead`, `promo`, `attackers`, `defenders`, `dist`)
    from its owner's side: [Defect] and the facts {pushes, path, promo, king_dist, king_sq}. Kinds: 'blocked' (a piece on the path), 'path attacked' (a path
    square attacked by them more than we guard it), 'promo square attacked' (the same on the promotion square), 'stopper' (a piece of theirs that reaches the
    promotion square or the path within the pawn's pushes: the king by `dist`, a slider / knight by its static-board `reach`). No defect = unstoppable by the
    tempo arithmetic (the owner moves first; the other side's pieces move for free while the pawn runs).
    plan 72.7 step 3c: read in Rust on the board's placement (`analyst_core.promote_defects_bb`, words.rs: every word read as `G.word` reads it -- the compiled terms on the
    board given to the owner --, the king's and the pieces' walks, the priced roll; the defects and the facts in the order the Python word gave them). The king verdict of the
    lift (plan 62.1c) is race.py's and is read here. The Python body is the test oracle (tests/test_plan72_7_step3.py)."""
    pc = board.piece_at(sq)
    own = pc.color
    kv = None
    priced = PRICED_STOPPER if priced is None else priced                   # plan 72.2: the stoppers read by the priced reach (the duty's cheap pre-check reads them static)
    if lift and RC.LIFT and board.turn == own and (board.occupied_co[not own] & (board.rooks | board.bishops | board.queens | board.knights)):
        kv = RC.king_verdict(board, sq)                 # plan 62.1c: with a piece of theirs beyond king and pawns the verdict judges only the KING, by the proved analysis; the pieces are replies the chase reads
    G.BV.install_terms()
    defs, n, path, promo, kd, ksq, priced_notes, lifted = _A.promote_defects_bb(G.bbs(board), sq, kv, priced)
    out = [Defect(k, t, list(by), why) for k, t, by, why in defs]
    facts = dict(pushes=n, path=list(path), promo=promo, king_dist=kd, king_sq=ksq)
    if lifted is not None:
        facts["lifted"] = lifted
    if priced_notes:
        facts["priced"] = priced_notes                                         # (kept apart from `lifted`, plan 62.1c's king verdict: that key re-asks the question unlifted)
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


# ---------------------------------------------------------------------------------------------------------------- plan 72: THE PRICED REACH
PRICED_STOPPER = True                         # plan 72.2: promote_defects' piece stoppers must reach the path free by the exchange (reach_price material 0), not on the static board
REACH_K = 3                                   # plan 72.1: the fuse on the roll forward's depth (a count, named as one: 72.3 replaces it by the race's tempo horizon -- what their fastest counter-plan needs)


def is_passer(board: chess.Board, sq: int) -> bool:
    """plan 72.2 (a): a PASSED pawn -- no pawn of the other side ahead of it on its file or the adjacent files (the Rust region word `passer`, read directly)."""
    pc = board.piece_at(sq)
    if pc is None or pc.piece_type != chess.PAWN:
        return False
    f, r = chess.square_file(sq), chess.square_rank(sq)
    for s in chess.SquareSet(board.pawns & board.occupied_co[not pc.color]):
        if abs(chess.square_file(s) - f) <= 1 and ((chess.square_rank(s) > r) if pc.color else (chess.square_rank(s) < r)):
            return False
    return True


def reach_map(board: chess.Board, sq: int, k: int = REACH_K, job: "tuple | None" = None) -> dict:
    """plan 72.1 THE ROLL FORWARD OF ONE UNIT: every landing the unit on `sq` (either colour) can reach within `k` of ITS OWNER'S moves, the other side's QUIET moves passed and
    its PAYING CAPTURES assumed -- a landing before the last must be SAFE by the exchange (`landsSafe`; an unsafe one ends the roll: the unit is taken there), the last may be PRICED
    (what the other side wins by the exchange there). -> {landing: dict(material, tempi, path [UCI], board)} with the cheapest entry by (material, tempi) per landing; the board is the one
    after the landing (the other side to move: what the unit bears on there is read from it). Memoised by position (cleared with the other caches).
    `job` = (target, bearing): the caller reads only the landings that DO the job (bear on / land on the target), so a landing on the roll's LAST step that does not is neither priced nor kept -- the
    exchange is read where an answer can come from (the last step is the widest, and every landing there paid a priced read for nothing). The earlier steps are kept: they lead on.
    plan 72.7 step 3b: THE ROLL IN RUST (`analyst_core.reach_map_bb`, words.rs): the same roll on the board's placement -- the one unit's quiet legal moves generated by
    python-chess's rules and in its order (a board whose side not to move is in check, which the roll reaches after a quiet check, included), the price by the word
    `see` as `G.see_for` reads it, the dict's order (a landing's first entry fixes its place) -- memoised in Rust by the same key, cleared by the same hook. The entry's
    `board` is a `Landing`: the unit's attacks from its landing come from Rust, and the python-chess board after the landing is REPLAYED from the roll's start (the
    path's moves, each followed by the pass, as the roll pushed them) the first time anything else of it is read. The Python body is the test oracle
    (tests/test_plan72_7_step3.py)."""
    pc = board.piece_at(sq)
    if pc is None:
        return {}
    b8 = G.bbs(board)
    key = ("rmap", b8, sq, k, job)                                            # (the placement: what board_fen() named)
    hit = _REACH.get(key)
    if hit is not None:
        return hit
    start = G.with_turn(board, pc.color)
    out = {to: dict(material=material, tempi=tempi, path=path, board=Landing(start, path, to, att))
           for to, material, tempi, path, att in _A.reach_map_bb(b8, sq, max(k, 0), job)}
    _REACH[key] = out
    return out


class Landing:
    """plan 72.7 step 3b: THE BOARD AFTER A LANDING OF THE ROLL (`reach_map`'s entry `board`), on demand. `attacks(at)` at the landing is the unit's attack set read in
    Rust; anything else is read on the python-chess board after the landing, REPLAYED from the roll's start the first time it is needed -- the path's moves, each but
    the last followed by the pass (`G.with_turn`), as the roll pushed them -- so it is the board the roll built, field by field."""
    __slots__ = ("_start", "_path", "_to", "_att", "_b")

    def __init__(self, start: chess.Board, path: list, to: int, att: int):
        self._start, self._path, self._to, self._att, self._b = start, path, to, att, None

    def _real(self) -> chess.Board:
        if self._b is None:
            b, own = self._start, self._start.turn
            for u in self._path[:-1]:
                a = b.copy(stack=False)
                a.push(chess.Move.from_uci(u))
                b = G.with_turn(a, own)
            a = b.copy(stack=False)
            a.push(chess.Move.from_uci(self._path[-1]))
            self._b = a
        return self._b

    def attacks(self, sq: int) -> chess.SquareSet:
        return chess.SquareSet(self._att) if sq == self._to else self._real().attacks(sq)

    def __getattr__(self, name):
        return getattr(self._real(), name)


_REACH: dict = {}
G.CLEAR_HOOKS.append(_REACH.clear)
G.CLEAR_HOOKS.append(_A.reach_cache_clear)                                    # plan 72.7 step 3b: the roll's Rust memo (the same key), cleared with `_REACH`


def reach_price(board: chess.Board, sq: int, target: int, k: int = REACH_K, bearing: bool = True) -> dict | None:
    """plan 72.1 THE PRICED REACH (Kaveh 2026-10-23: "attackable in ... nothing is illegal in chess, it just costs you something; the natural form of cost is material and tempo").
    The unit on `sq` comes to BEAR on `target` (bearing=True: attacks it from where it lands; False: lands on it) within `k` of its owner's moves, by `reach_map`'s roll: the price is a
    PAIR (material, tempi), the cheapest by that order; `attackableIn` (Lean `Ideas.Vocabulary`, `Ideas.Reach.reachPrice_zero_iff`) is the price-0 case. -> dict(material, tempi,
    first (UCI), path, reads), tempi 0 and an empty path when the unit bears on the target already; None when no path within k. ONE unit rolls: a path another unit must open
    (Kaveh's two knights) is not read here. The answer is asymmetric by construction: no path, or a price above the stake, kills a join NOW (with its read set: a move of theirs that
    opens a line re-asks it); a payable price is a CANDIDATE line with no reply read.
    plan 72.7 step 3b: read in Rust on the board's placement (`analyst_core.reach_price_bb`: the roll with the job, the cheapest landing doing it -- the first in the
    roll's order on a tie -- and its read set). The Python body is the test oracle (tests/test_plan72_7_step3.py)."""
    got = _A.reach_price_bb(G.bbs(board), sq, target, max(k, 0), bearing)
    if got is None:
        return None
    material, tempi, first, path, reads = got
    return dict(material=material, tempi=tempi, first=first, path=path, reads=frozenset(reads))


def stuck_threats(board: chess.Board, us: str | bool, k: int = REACH_K) -> list:
    """plan 72.2 (b) THE UNIT THAT CANNOT RUN, as THEIR threat on a static leaf (`board`: us to move; the move is passed to them). For each unit of ours (not the king, not a pawn:
    a pawn never runs, its loss is a one-move word) not attacked already (then it hangs: the one-move words), two readings, by what the unit can do NOW (the cost rule of the
    gate of 2026-10-23: the roll forward of every unit of theirs at every leaf cost tens of seconds a run; a unit that can run is threatened only by an attack that also removes
    its escape, and that is ONE move of theirs):
      (1) the unit CAN run now: THEIR ONE-MOVE ATTACKS on it (their legal moves after which it is attacked, the mover's landing safe by the exchange) such that after the attack it
          cannot run (63.2 `can_run` with the passer duty: a safe square whose move frees a passer of theirs is no escape -- I7k2H's b3 against the bishop on c2);
      (2) the unit CANNOT run now: their cheapest `reach_price` to a square bearing on it within k at a material price below its worth (the stuck knight, the king's walk).
    Its PARRY by the same word with the sides swapped: a unit of ours that bears on it already, or (2 only) reaches a square bearing on it within tempi-1 at price 0, or a move of
    ours after which it can run. -> [dict(sq, worth, attacker, price, tempi, first, path, parry | None, reads)], the worst first. A parry in time is a CANDIDATE (the roll reads
    no reply), a missing parry is the leaf's GAP text, never a kill.
    plan 72.7 step 3c: read in Rust in ONE call on the board's handle (`analyst_core.stuck_threats_pos`, words.rs: can_run with the passer duty -- duty_after's passer branch,
    promote_defects static and priced --, their one-move attacks generated by python-chess's rules on the board given to them, where our king may stand in check, the roll for
    the unit that cannot run, the parry by reach_price and by our moves, the rows in the order the Python word sorted them). The Python body is the test oracle
    (tests/test_plan72_7_step3.py)."""
    G.BV.install_terms()
    return [dict(sq=s_, worth=w, attacker=a, price=p, tempi=t, first=f, path=list(path), parry=parry, reads=frozenset(reads))
            for s_, w, a, p, t, f, path, parry, reads in _A.stuck_threats_pos(G.handle(board), G.colour_of(us), max(k, 0))]
JOIN_FIXES = True                             # plan 72.3: a defect that needs a piece of ours at L also gets "bring U to bear on L" for each unit U, priced by `reach_price`
JOIN_KINDS = {"mate": ("flight", "checker takeable"), "win": ("not attacked", "defended")}      # the defect kinds read as "a piece of ours is needed bearing on L"
JOIN_KING = ("flight", "checker takeable", "path attacked", "promo square attacked")             # the guards the king can give
_JOIN: dict = {}                              # (board_fen, first step UCI) -> (material, tempi): read by `route_cost`
G.CLEAR_HOOKS.append(_JOIN.clear)
_COUNTER: dict = {}                           # (board_fen, turn, side, first step UCI, mu) -> counter_value: what their counter-plan buys is a fact of the POSITION, not of the question that asks it
G.CLEAR_HOOKS.append(_COUNTER.clear)


def counter_value(board: chess.Board, us: str | bool, step: chess.Move, mu: int) -> tuple:
    """Memoised by position: every inner agenda and every question that reaches the same square asks the same board, step and tempi again; see `_counter_value`."""
    key = (board.board_fen(), board.turn, G.colour_of(us), step.uci() if step is not None else None, mu)
    hit = _COUNTER.get(key)
    if hit is None:
        hit = _COUNTER[key] = _counter_value(board, us, step, mu)
    return hit


def _counter_value(board: chess.Board, us: str | bool, step: chess.Move, mu: int) -> tuple:
    """plan 72.3 THE STAKE RULE: what THEY buy with the free moves our join of `mu` tempi hands them. Our moves 1..mu are the join, our (mu+1)-th is the strike; they move after each, so
    they have mu-1 moves to bring a unit to bear before our strike and their strike is the mu-th (the race: our mu against their nu = their reach tempi + 1; `Game.race_sound`, Ideas.Game --
    here the roll is a CANDIDATE, only the chase proves; nothing is claimed of the race beyond the comparison nu <= mu = they strike first).
      mu == 1: the threat word with the move passed, APPROXIMATED by `threat_value` for them on the board after our step (what they win by the exchange on a piece of ours, their move),
               less the same on the board as it stands (what they win whatever we do: not the plan's cost).
      mu >= 2: their cheapest price-0 `reach_map` landing within mu-1 moves after which a unit of ours is won by the exchange (`G.see`), the largest such win; the board is ours as it stands
               (our join steps are not played: the approximation).
    -> (value, text or None)."""
    me = G.colour_of(us)
    if mu <= 1:
        a = board.copy(stack=False)
        a.push(step)
        base = threat_value(G.with_turn(board, not me), me)
        v = threat_value(a, not me)
        if v <= base:
            return 0, None
        return v - base, f"after {board.san(step)} they win {v - base} by the exchange with the move"
    best, text = 0, None
    for sq in chess.SQUARES:
        pc = board.piece_at(sq)
        if pc is None or pc.color == me or pc.piece_type == chess.KING:
            continue
        for at, e in reach_map(board, sq, mu - 1).items():
            if e["material"] != 0:
                continue
            c = G.with_turn(e["board"], not me)
            for v in c.attacks(at):
                tp = c.piece_at(v)
                if tp is None or tp.color != me or tp.piece_type == chess.KING:
                    continue
                w = G.see(c, v)
                if w > best:
                    best, text = w, f"their {pc.symbol().upper()} on {G.name(sq)} reaches our {tp.symbol().upper()} on {G.name(v)} in {e['tempi']}"
    return best, text


_JOIN_UNIT: dict = {}                         # (board_fen, side, unit square, L, k, stake) -> the unit's verdict: a fact of the position and the stake, read once however many inner agendas and questions ask it
G.CLEAR_HOOKS.append(_JOIN_UNIT.clear)


def _join_unit(b: chess.Board, me: bool, sq: int, L: int, kk: int, stake: int) -> "tuple | None":
    """One unit's answer to the join question: None (it cannot come to bear, or already does), ("why", the refusal) or ("route", first step UCI, label, (material, tempi)). Memoised by position:
    the question that asks it (and the inner agenda it sits in) changes nothing the answer reads, only the board, the unit, L, the fuse and the stake do."""
    if not _can_bear_in_principle(b.piece_at(sq), sq, L, kk):
        return None
    key = (b.board_fen(), me, sq, L, kk, stake)
    if key in _JOIN_UNIT:
        return _JOIN_UNIT[key]
    out = None
    r = reach_price(b, sq, L, kk)
    if r is not None and r["tempi"] != 0 and r["first"] is not None:
        pc = b.piece_at(sq)
        step = chess.Move.from_uci(r["first"])
        if stake - r["material"] - _counter_ceiling(b, me) >= 1:   # the stake clears the price by more than the most their counter-plan could ever win (our most valuable non-king unit): it cannot refuse the join, so it is not rolled
            counter, ctext = 0, None
        else:
            counter, ctext = counter_value(b, me, step, r["tempi"])
        nm = pc.symbol().upper() if pc.piece_type != chess.PAWN else "pawn"
        if stake - r["material"] - counter < 1:
            out = ("why", f"the {nm} cannot join in time: " + (ctext + ", " if ctext else "") + f"price {r['material']} + their {counter} against a stake of {stake}")
        else:
            label = f"bring the {nm} from {G.name(sq)} to bear on {G.name(L)} in {r['tempi']}" + (f" (via {G.name(step.to_square)})" if r["tempi"] > 1 else "") + f", price {r['material']}"
            out = ("route", r["first"], label, (r["material"], r["tempi"]))
    _JOIN_UNIT[key] = out
    return out


def _can_bear_in_principle(pc: chess.Piece, sq: int, L: int, k: int) -> bool:
    """The geometry of the piece, before any roll: whether the unit on `sq` could come to bear on L within k QUIET moves on an empty board. A bishop never leaves its colour complex, so it never
    bears on a square of the other colour; a king bears on L only from a square next to it and walks k steps; a pawn walks straight (the roll has no captures) and attacks the diagonal
    ahead, so it bears only on an adjacent file at a rank 1..k+2 ahead (k pushes, the first a double step, and the attack one rank further) -- unless it can promote within k, when it is a queen and the roll decides.
    A necessary condition only: rooks, queens and knights are always rolled."""
    pt = pc.piece_type
    if pt == chess.BISHOP:
        return (chess.square_file(sq) + chess.square_rank(sq)) % 2 == (chess.square_file(L) + chess.square_rank(L)) % 2
    if pt == chess.KING:
        return chess.square_distance(sq, L) <= k + 1
    if pt == chess.PAWN:
        d = 1 if pc.color else -1
        if (7 - chess.square_rank(sq) if pc.color else chess.square_rank(sq)) <= k:
            return True
        return abs(chess.square_file(sq) - chess.square_file(L)) == 1 and 1 <= (chess.square_rank(L) - chess.square_rank(sq)) * d <= k + 2
    return True


def _counter_ceiling(b: chess.Board, me: bool) -> int:
    """The most their counter-plan can ever win: the exchange on a unit of ours wins at most that unit's worth, so at most our most valuable non-king unit."""
    return max((G.worth(p) for p in b.piece_map().values() if p.color == me and p.piece_type != chess.KING), default=0)


def join_fixes(board: chess.Board, us: str | bool, defect: Defect, q: Question, k: int | None = None) -> "Question | None":
    """plan 72.3 THE JOIN QUESTION: can a unit of ours come to bear on the location L the defect needs a piece at (the flight to cover, the square to guard, the target to attack)?
    For each unit U of ours (the king only for the guards it can give, `JOIN_KING`; never the root pattern's mover or a blocker), `reach_price(board, U, L, k, bearing=True)`, k the pawn's
    pushes for a promotion defect, else the fuse `REACH_K`. A payable reach is a ROUTE whose move is the word's FIRST STEP ("bring the N from g1 to bear on f7 in 2 (via h3), price 0"), with
    (material, tempi) on the route and in `route_cost`. THE STAKE RULE (Kaveh 2026-10-23): the plan gains `q.stake`; it pays the reach's material and hands them `counter_value` (above);
    when stake - material - counter < 1 the fix is NOT PUSHED -- a kill of this FIX (a route not asked, the reason in the question's why), not of the goal, which keeps its other fixes.
    The bar is "worth doing at all" (net positive); the spec's wording "below the goal's stake" is read as that. -> a 'cover' sub-question (routes may be empty when every reach was refused),
    or None when no unit can reach L (a cut is a GAP, not a kill: nothing is said)."""
    me = G.colour_of(us)
    L = defect.sq
    if L is None:
        return None
    pat = q.pattern
    skip = {pat.get("frm"), pat.get("pawn")} | set(pat.get("blockers", ())) | ({pat["blocker"]} if pat.get("blocker") is not None else set())
    kk = REACH_K if k is None else k
    b = G.with_turn(board, me)
    routes, why = [], []
    for sq in chess.SQUARES:
        pc = b.piece_at(sq)
        if pc is None or pc.color != me or sq in skip or sq == L or (pc.piece_type == chess.KING and defect.kind not in JOIN_KING):
            continue
        verdict = _join_unit(b, me, sq, L, kk, q.stake)
        if verdict is None:
            continue
        if verdict[0] == "why":
            why.append(verdict[1])
            continue
        _, step_uci, label, price = verdict
        step = chess.Move.from_uci(step_uci)
        routes.append(Route(step, b.san(step), label, price=price))
        _JOIN[(b.board_fen(), step_uci)] = price
    if not routes and not why:
        return None
    f = new_question("cover", q.stake, dict(q.pattern, target=L), q, defect_key=defect.key(), text=f"bring a unit to bear on {G.name(L)} (the priced reach)")
    f.routes = routes
    f.why = [x.why for x in routes] + why
    return f


def joins_of(q: Question, board: chess.Board, us: str | bool, k: int | None = None) -> list:
    """plan 72.3: the join questions of the defects of `q` that have NO forcing fix (the ones the tempo arithmetic is about to kill: only quiet moves could supply the piece). Stored in `q.joins`,
    NOT in `q.fixes`: a join is a QUIET plan (the first step hands them a free move), so it is not a forcing route -- `tempo_ok`, `all_routes`, the chase and the question count never read it
    (the first form put the routes in `fixes` and 02XVK's inner questions went from <1000 to 4776, 0LRTF / 04Qx2 / 0BHbB changed their decisions: a quiet route is no fix for a tempo count).
    The agenda prints them under the question and the policy / the survey (72.5) may order by them; the stake rule has already refused the ones their counter-plan beats. -> the join questions."""
    if not JOIN_FIXES or q.joins:
        return q.joins
    kinds = JOIN_KINDS.get(q.goal, ()) + (("path attacked", "promo square attacked") if q.goal == "promote" else ())
    saved = _ids[0]                                                  # (the join questions take no id from the numbering the trace prints)
    for d in q.defects:
        if d.kind not in kinds or any(all_routes(f) for f in q.fixes if f.defect_key == d.key()):
            continue
        j = join_fixes(board, us, d, q, k=k)
        if j is not None:
            q.joins.append(j)
    _ids[0] = saved
    return q.joins

# ---------------------------------------------------------------------------------------------------------------- plan 72.5: THE SURVEY BEFORE THE QUESTIONS
SURVEY_LINES = 8                              # plan 72.5: the fuse on the survey's printed lines (a count, named as one); the table itself is whole


class SurveyRow(NamedTuple):
    """One row of the survey: the unit on `sq` (colour `owner`, python-chess `kind`) and the cheapest layer-2 reach (`reach_map`) to one JOB: job_kind 'zone' (a square of the enemy king's
    neighbourhood; mate), 'attack' (an enemy unit it comes to bear on; `job` is its square), 'path' (a square of a passer's way to promotion, either side's passer), 'free' (mobility: `count`
    landings that cost nothing by the exchange). `via` is the landing the unit bears from (None: it bears from where it stands); tempi 0 = already."""
    sq: int
    owner: bool
    kind: int
    job: "int | None"
    job_kind: str
    material: int
    tempi: int
    via: "int | None" = None
    count: int = 0


_SURVEY: dict = {}
G.CLEAR_HOOKS.append(_SURVEY.clear)


def passer_path(board: chess.Board, sq: int) -> list:
    """The squares ahead of the passed pawn on `sq` up to and including its promotion square."""
    pc = board.piece_at(sq)
    step = 8 if pc.color else -8
    out, s = [], sq + step
    while 0 <= s < 64:
        out.append(s)
        s += step
    return out


def survey(board: chess.Board, k: int = REACH_K) -> list:
    """plan 72.5 THE SURVEY (Kaveh 2026-10-23: "a quick pass before deciding on the right questions"): for every unit of BOTH sides the cheapest layer-2 reach to each job on the board within
    `k` of its owner's moves -- the enemy king's neighbourhood (zone), each enemy unit it could bear on (attack), the path squares of every passer (path) -- and how many landings are free
    (free). ONE `reach_map` roll per unit (memoised), nothing else rolls. -> [SurveyRow]. It orders, it never decides: the proof stays the chase's."""
    key = (board.board_fen(), k)
    hit = _SURVEY.get(key)
    if hit is not None:
        return hit
    pieces = board.piece_map()
    paths = {s: passer_path(board, s) for s, p in pieces.items() if is_passer(board, s)}
    rows = []
    for sq, pc in pieces.items():
        spots = [(sq, 0, 0, board.attacks(sq))]
        rmap = reach_map(board, sq, k)
        spots += [(at, e["tempi"], e["material"], e["board"].attacks(at)) for at, e in rmap.items() if at != sq]
        rows.append(SurveyRow(sq, pc.color, pc.piece_type, None, "free", 0, 0, None, sum(1 for at, e in rmap.items() if at != sq and e["material"] == 0)))

        def cheapest(want):                                       # (material, tempi, spot, the first job square) over the spots whose attacks (or standing square) meet `want`
            best = None
            for at, t, m, att in spots:
                hits = [x for x in sorted(want) if x in att or x == at]
                if hits and (best is None or (m, t) < best[:2]):
                    best = (m, t, at, hits[0])
            return best
        ek = board.king(not pc.color)
        if ek is not None and pc.piece_type != chess.KING:
            b = cheapest({ek} | set(chess.SquareSet(chess.BB_KING_ATTACKS[ek])))
            if b:
                rows.append(SurveyRow(sq, pc.color, pc.piece_type, b[3], "zone", b[0], b[1], None if b[2] == sq else b[2]))
        for tsq, tp in pieces.items():
            if tp.color != pc.color and tp.piece_type != chess.KING:
                b = cheapest({tsq})
                if b:
                    rows.append(SurveyRow(sq, pc.color, pc.piece_type, tsq, "attack", b[0], b[1], None if b[2] == sq else b[2]))
        for psq, path in paths.items():
            if psq != sq:
                b = cheapest(set(path))
                if b:
                    rows.append(SurveyRow(sq, pc.color, pc.piece_type, b[3], "path", b[0], b[1], None if b[2] == sq else b[2]))
    _SURVEY[key] = rows
    return rows


def _unit_name(board: chess.Board, sq: int, us: bool) -> str:
    pc = board.piece_at(sq)
    return f"{'our' if pc.color == us else 'their'} {pc.symbol().upper()}"


def survey_lines(board: chess.Board, us: bool, k: int = REACH_K) -> list:
    """The survey as plain words, one line per unit that has something to say, the most telling first, at most SURVEY_LINES:
    'survey: their K reaches c7 in 3, bearing on our N on b8' / 'survey: our N on b8 can join nothing within 3' / 'survey: our B on c2 holds their c-pawn's path'."""
    return [t for t, _r, _w in survey_items(board, us, k)]


def survey_items(board: chess.Board, us: bool, k: int = REACH_K) -> list:
    """plan 74.4: the lines `survey_lines` prints as OBSERVATIONS -- (the line, its READS: the squares it names (the unit's, the landing's, the job's, a held pawn's), its TERM: (the
    word, the unit's square, the job's square)). The same words, the same order, the same cap; the reads are what a goal's frame is compared with (the promotion rule)."""
    rows = survey(board, k)
    by: dict = {}
    for r in rows:
        by.setdefault(r.sq, []).append(r)
    out = []                                                       # (priority, tempi, text)
    for sq, rs in by.items():
        pc = board.piece_at(sq)
        name = f"{_unit_name(board, sq, us)} on {chess.square_name(sq)}"
        free = next(r.count for r in rs if r.job_kind == "free")
        paths = [r for r in rs if r.job_kind == "path" and r.tempi == 0]
        if paths and pc.piece_type != chess.PAWN:
            ps = min(paths, key=lambda r: r.job)
            pawn = next(s for s in range(64) if s != sq and is_passer(board, s) and ps.job in passer_path(board, s))
            owner = board.piece_at(pawn).color
            out.append((1, 0, f"{name} holds {'our' if owner == us else 'their'} {chess.FILE_NAMES[chess.square_file(pawn)]}-pawn's path", frozenset({sq, ps.job, pawn}), ("holds", sq, pawn)))
            continue
        att = [r for r in rs if r.job_kind == "attack" and r.tempi > 0 and r.via is not None and r.material < G.worth(board.piece_at(r.job)) and G.worth(board.piece_at(r.job)) > G.worth(pc)]
        if att:
            r = min(att, key=lambda r: (r.tempi, r.material, -G.worth(board.piece_at(r.job))))
            price = f", at a price of {r.material}" if r.material else ""
            out.append((0, r.tempi, f"{name} reaches {chess.square_name(r.via)} in {r.tempi}, bearing on {_unit_name(board, r.job, us)} on {chess.square_name(r.job)}{price}", frozenset({sq, r.via, r.job}), ("bears", sq, r.job)))
            continue
        if pc.piece_type not in (chess.KING, chess.PAWN) and not any(r.job_kind != "free" for r in rs):
            out.append((2, 0, f"{name} can join nothing within {k}" + ("" if free else " and has no free landing"), frozenset({sq}), ("idle", sq, None)))
            continue
        z = [r for r in rs if r.job_kind == "zone" and r.tempi > 0 and r.material == 0]
        if z:
            r = min(z, key=lambda r: r.tempi)
            out.append((3, r.tempi, f"{name} comes to bear on {'their' if pc.color == us else 'our'} king's zone in {r.tempi}", frozenset(x for x in (sq, r.via, r.job) if x is not None), ("zone", sq, r.job)))
    out.sort(key=lambda t: (t[0], t[1], t[2]))
    return [("survey: " + t, rd, term) for _p, _t, t, rd, term in out[:SURVEY_LINES]]


def survey_features(board: chess.Board, us: bool, targets: list, k: int = REACH_K) -> tuple:
    """plan 72.5 THE SURVEY AS FEATURES of the graph policy (feature set 'survey1'; an older model runs on the old width, see graph_policy). -> (unit_feats, question_feats):
    unit_feats[sq] = [a, b, c, d]: a the cheapest tempi to bear on the enemy king's zone, b the cheapest tempi to bear on an enemy unit of higher worth (both over k+1, 1.0 = none within k),
    c the free landings over 20 (capped), d 1 when the unit holds a passer's path now; question_feats[i] = [t]: the fewest tempi over our units to bear on / land on question i's target
    (over k+1, 1.0 = none or no target)."""
    rows = survey(board, k)
    pieces = board.piece_map()
    unit = {}
    for sq, pc in pieces.items():
        rs = [r for r in rows if r.sq == sq]
        a = min([r.tempi for r in rs if r.job_kind == "zone"], default=k + 1)
        b = min([r.tempi for r in rs if r.job_kind == "attack" and G.worth(pieces[r.job]) > G.worth(pc)], default=k + 1)
        c = sum(r.count for r in rs if r.job_kind == "free")
        d = any(r.job_kind == "path" and r.tempi == 0 for r in rs)
        unit[sq] = [a / (k + 1), b / (k + 1), min(c, 20) / 20, float(d)]
    qf = []
    for t in targets:
        best = k + 1
        if t is not None and 0 <= t < 64:
            for sq, pc in pieces.items():
                if pc.color == us and sq != t:
                    r = reach_price(board, sq, t, k)
                    if r is not None:
                        best = min(best, r["tempi"])
        qf.append([best / (k + 1)])
    return unit, qf


# ---------------------------------------------------------------------------------------------------------------- plan 72.8: THE SKELETON GOALS
# Kaveh 2026-10-23: "we ask too many questions because we do not know the goal". A goal is an ARRANGEMENT of our fast pieces against the FIXED board (the slow units: kings, pawns,
# the units that cannot run), read on counterfactual boards, priced by the roll forward (72.1), weighed by the stake rule (72.3). Stage 1: THE ARRANGEMENT IS A CANDIDATE GOAL; ITS
# PROOF IS THE CHASE. Nothing new is proved in Lean here (an arrangement is a conjunction of bearing predicates over a counterfactual board: `Ideas.Vocabulary.attackableIn`,
# `Ideas.Reach.reachPrice` / `reachPrice_zero_iff`, the mate defects = the failed hypotheses of `Ideas.checkmate_of_unmeetable_check`).
SKELETON_MAX_JOBS = 4                         # plan 72.8 (5), a NAMED FUSE: a goal whose skeleton defects name more than 4 jobs is dropped -- the arrangement is then a plan the chase cannot verify in the horizon (not a limit on the engine's words)
SKELETON_MAX_UNITS = 7                        # plan 72.8 (12), a NAMED FUSE: at most 7 fast units of ours enter the assignment (7*6*5*4*3 = 2520 tuples at most); with more (a promotion's extra queen) the 7 most valuable
SKELETON_LINES = 6                            # plan 72.8 (10): the trace prints at most 6 arrangement lines (a print cap; the list itself is whole)
_VERDICTS = ("reachable", "open", "refused", "unreachable")
_GOALS = ("mate", "win", "promote")


def _aligned(piece_type: int, sq: int, ksq: int) -> bool:
    """Whether a piece of this type on `sq` would attack `ksq` on an EMPTY board (every attack on a real board is one of these); agenda._aligned is this word."""
    if piece_type == chess.KNIGHT:
        return bool(chess.BB_KNIGHT_ATTACKS[sq] & chess.BB_SQUARES[ksq])
    hit = 0
    if piece_type in (chess.BISHOP, chess.QUEEN):
        hit |= chess.BB_DIAG_ATTACKS[sq][0]
    if piece_type in (chess.ROOK, chess.QUEEN):
        hit |= chess.BB_RANK_ATTACKS[sq][0] | chess.BB_FILE_ATTACKS[sq][0]
    return bool(hit & chess.BB_SQUARES[ksq])


@dataclass
class Job:
    """What an arrangement needs a unit of ours for: to BEAR on `square` (a flight to cover, the checker's square to guard, an interposition square, the target, a path square)."""
    square: int
    kind: str                                 # 'flight' | 'guard' | 'interposition' | 'attack' | 'path' | 'promo' | another defect kind
    why: str = ""


@dataclass
class Deflection:
    """A fast unit of theirs on `their_sq` that occupies or bears on `job_square` (or a defender / stopper the defects name): the job cannot be done as priced. Stage 1 records it (the
    arrangement is `open: deflection`); a deflection question is a LATER item."""
    their_sq: int
    job_square: int
    why: str = ""


@dataclass
class Arrangement:
    """plan 72.8 (9): one skeleton goal. `target`: 'king' for a mate, else the target square (the unit to win; the promotion square). `about`: the square a root question of the same goal names
    (the placement square, the target, the pawn). `assignment`: [(unit sq, Job or 'placement', material, tempi, path [UCI])]."""
    target: "int | str"
    goal: str
    placement: "tuple | None"                 # (unit sq, square) for a mate, else None
    jobs: list
    assignment: list
    step_asides: list                         # [(unit sq, text)]
    deflections: list
    material: int
    tempi: int
    stake: int
    counter: int
    verdict: str                              # reachable | open | refused | unreachable
    text: str = ""
    why: str = ""                             # the verdict's reason: the job not reached, the counter-plan's text
    goal_text: str = ""
    about: "int | None" = None
    syms: dict = field(default_factory=dict)  # square -> the unit's letter (ours assigned or stepping aside, theirs deflected): the text names them

    @property
    def net(self) -> int:
        """The stake rule's balance (72.3): stake - material - counter."""
        return self.stake - self.material - self.counter


def skeleton_partition(board: chess.Board) -> dict:
    """plan 72.8 (1) SLOW and FAST, either colour: {sq: reason} with reason 'king' / 'pawn' / 'no safe square' / 'duty-bound' for a SLOW unit and None for a FAST one. A unit is slow when it
    cannot run (`can_run(board, sq, the other side)`, 63.2) -- 'duty-bound' when it has safe squares but every one abandons a duty worth at least the unit (the cut list), 'no safe square' when
    it has none. (`duty_bound` needs no second read: a unit that can run has a safe move whose duty is below its worth, so it is never duty-bound -- the least over its moves is below its worth.)"""
    out = {}
    for sq, pc in board.piece_map().items():
        if pc.piece_type == chess.KING:
            out[sq] = "king"
        elif pc.piece_type == chess.PAWN:
            out[sq] = "pawn"
        else:
            run, cut = can_run(board, sq, not pc.color)
            out[sq] = None if run else ("duty-bound" if cut else "no safe square")
    return out


def _counterfactual(board: chess.Board, me: bool, drop) -> chess.Board:
    """A copy without the units on `drop`, `me` to move, no en passant, no castling (a counterfactual has no history). Never pushed on."""
    c = board.copy(stack=False)
    for s in drop:
        c.remove_piece_at(s)
    c.turn = me
    c.ep_square = None
    c.castling_rights = chess.BB_EMPTY
    return c


def skeleton_a(board: chess.Board, us: str | bool, part: dict | None = None) -> chess.Board:
    """plan 72.8 (2) SKELETON A: the board with OUR fast units removed (theirs all kept), us to move."""
    me = G.colour_of(us)
    part = skeleton_partition(board) if part is None else part
    return _counterfactual(board, me, [s for s, why in part.items() if why is None and board.piece_at(s).color == me])


def skeleton_b(board: chess.Board, us: str | bool, part: dict | None = None) -> chess.Board:
    """plan 72.8 (2) SKELETON B: skeleton A with THEIR fast units removed too (read only to find the deflections: their fast units are the ones B lacks)."""
    me = G.colour_of(us)
    part = skeleton_partition(board) if part is None else part
    return _counterfactual(board, me, [s for s, why in part.items() if why is None])


def _sym(board: chess.Board, sq: int) -> str:
    return board.piece_at(sq).symbol().upper()


def skeleton_line(board: chess.Board, us: str | bool, part: dict) -> str:
    """The trace line of the partition (ours): 'skeleton: fixed K g1, P a2 b2 f2 g2 h2, N f3 (no safe square), B e2 (duty-bound); fast Q d1, R a1, R f1'."""
    me = G.colour_of(us)
    ours = {s: w for s, w in part.items() if board.piece_at(s).color == me}
    order = lambda s: (-G.worth(board.piece_at(s)), s)
    fixed = [f"K {G.name(s)}" for s, w in ours.items() if w == "king"]
    pawns = sorted(s for s, w in ours.items() if w == "pawn")
    if pawns:
        fixed.append("P " + " ".join(G.name(s) for s in pawns))
    fixed += [f"{_sym(board, s)} {G.name(s)} ({ours[s]})" for s in sorted((s for s, w in ours.items() if w not in (None, "king", "pawn")), key=order)]
    fast = [f"{_sym(board, s)} {G.name(s)}" for s in sorted((s for s, w in ours.items() if w is None), key=order)]
    return f"skeleton: fixed {', '.join(fixed) or 'nothing'}; fast {', '.join(fast) or 'none'}"


def _path_squares(path: list, start: int, bear_on: "int | None" = None) -> set:
    """6c: the squares of a chosen path on the live board -- every landing and the squares between consecutive landings (from the start), and for a bearing job the line from the last
    landing (the start when the unit bears already) to the job square (on skeleton A that line was open only because our other fast units were lifted)."""
    out, at = set(), start
    for u in path:
        m = chess.Move.from_uci(u)
        out.update(chess.SquareSet(chess.between(m.from_square, m.to_square)))
        out.add(m.to_square)
        at = m.to_square
    if bear_on is not None:
        out.update(chess.SquareSet(chess.between(at, bear_on)))
    return out


def _mate_jobs(defects: list, s: int) -> list:
    """plan 72.8 (5) the jobs of a mate placement on `s`: flight -> bear on it; checker takeable -> bear on `s` (a guard); interposition -> bear on it; any other kind -> bear on its square.
    One job per square (two takers of the checker are one guard job)."""
    jobs, seen = [], set()
    for d in defects:
        if d.kind == "checker takeable":
            sq, kind = s, "guard"
        elif d.kind in ("flight", "interposition"):
            sq, kind = d.sq, d.kind
        else:
            sq, kind = d.sq, d.kind
        if sq is None or sq in seen:
            continue
        seen.add(sq)
        jobs.append(Job(sq, kind, d.why))
    return jobs


def _first_step(entry_path: list) -> "chess.Move | None":
    return chess.Move.from_uci(entry_path[0]) if entry_path else None


def skeleton_report(board: chess.Board, us: str | bool, k: int = REACH_K) -> dict:
    """plan 72.8, the whole stage-1 reading of one position: -> dict(line, partition, targets, dropped, arrangements). Memoised for the LAST position only (the passes of one root ask it
    again; a different board replaces it) -- a pure function of the board and the fuses."""
    me = G.colour_of(us)
    b = board if board.turn == me else G.with_turn(board, me)
    key = (b.board_fen(), me, b.castling_rights, k, SKELETON_MAX_JOBS, SKELETON_MAX_UNITS, PRICED_STOPPER, DUTY)
    if _SKEL.get("key") == key:
        return _SKEL["report"]
    rep = _skeleton_report(b, me, k)
    _SKEL.clear()
    _SKEL.update(key=key, report=rep)
    return rep


_SKEL: dict = {}


def skeleton_goals(board: chess.Board, us: str | bool, k: int = REACH_K) -> list:
    """plan 72.8 (9) THE SKELETON GOALS of the position (`us` to move): [Arrangement], reachable first by (stake - material - counter) descending, then open, then refused, then unreachable.
    Every arrangement is a CANDIDATE: it adds routes, roots and a feature in the agenda; it never claims and never kills (the chase proves the survivor)."""
    return skeleton_report(board, us, k)["arrangements"]


def _skeleton_report(b: chess.Board, me: bool, k: int) -> dict:
    them = not me
    part = skeleton_partition(b)
    skel_a = skeleton_a(b, me, part)
    skel_b = skeleton_b(b, me, part)
    fast = sorted((s for s, w in part.items() if w is None and b.piece_at(s).color == me), key=lambda s: (-G.worth(b.piece_at(s)), s))
    units = fast[:SKELETON_MAX_UNITS]
    their_fast = [s for s in chess.SQUARES if (p := skel_a.piece_at(s)) is not None and p.color == them and skel_b.piece_at(s) is None]
    alone = {u: G.with_piece(skel_a, b.piece_at(u), u) for u in units}        # 6a: each unit ALONE on skeleton A (our other fast units are off the board: a path they block is open)
    bears: dict = {}

    def bear(u: int, sq: int):                                               # 6a: the job's cost, reach_price(bearing=True) on skeleton A with the unit -> (material, tempi, path) or None
        kk = (u, sq)
        if kk not in bears:
            r = reach_price(alone[u], u, sq, k, bearing=True)
            bears[kk] = None if r is None else (r["material"], r["tempi"], list(r["path"]))
        return bears[kk]

    def land(u: int, s: int):                                                # 6a: the placement's cost, reach_price(bearing=False): the unit LANDS on s
        occ = skel_a.piece_at(s)
        if occ is None:
            e = reach_map(alone[u], u, k).get(s)
            if e is None:
                return None
            after, m, t, path = e["board"], e["material"], e["tempi"], list(e["path"])
        else:                                                                # (4) their unit on s: a capture removes it -- the roll is quiet (72.1), so the capture is the roll's last step:
            r = reach_price(alone[u], u, s, k - 1, bearing=True)             # the unit bears on s within k-1, then takes (one more tempo), priced by the exchange after the capture
            if r is None:
                return None
            at = chess.Move.from_uci(r["path"][-1]).to_square if r["path"] else u
            c = G.with_turn(G.with_piece(alone[u], b.piece_at(u), at, u) if at != u else alone[u], me)
            mv = chess.Move(at, s)
            if mv not in c.legal_moves:
                return None
            c.push(mv)
            after, m, t, path = c, G.see_for(c, s, them), r["tempi"] + 1, list(r["path"]) + [mv.uci()]
        ek_ = after.king(them)
        if m and not (set(after.attackers(them, s)) - {ek_}):               # the landing is the check: a capture by their KING alone is the defect 'checker takeable', which the guard job
            m = 0                                                            # of this arrangement pays (a guarded checker cannot be taken by the king); a capture by another unit stays priced
        return (m, t, path)

    arrs, dropped, targets = [], 0, 0
    ek = b.king(them)
    if ek is not None:                                                       # (3)(i) their king: goal `mate`, one arrangement per (piece kind, placement square)
        targets += 1
        places: dict = {}
        for u in units:
            pc = b.piece_at(u)
            for s in chess.SQUARES:
                if s == u or not _aligned(pc.piece_type, s, ek):
                    continue
                occ = skel_a.piece_at(s)
                if occ is not None and (occ.color == me or occ.piece_type == chess.KING):
                    continue
                places.setdefault((pc.piece_type, s), []).append(u)
        for (pt, s), cands in sorted(places.items(), key=lambda t: (t[0][1], -t[0][0])):
            pc = chess.Piece(pt, me)
            cf = G.with_piece(skel_a, pc, s)
            if ek not in cf.attacks(s):
                continue
            defects = mate_defects(cf, me, [(pc, s, None)])
            if any(d.kind == "no check" for d in defects):
                continue
            jobs = _mate_jobs(defects, s)
            if len(jobs) > SKELETON_MAX_JOBS:
                dropped += 1
                continue
            arrs.append(_price(b, me, k, "mate", "king", s, jobs, [], [], STAKE_MATE, units, their_fast, bear, fast=fast, place=(s, cands, land), ek=ek))
    for sq in sorted(part):                                                  # (3)(ii) each SLOW unit of theirs that is not a king or a pawn: goal `win`
        pc = b.piece_at(sq)
        if pc.color != them or part[sq] in (None, "king", "pawn"):
            continue
        targets += 1
        defects = material_defects(skel_a, me, sq)
        jobs = [Job(sq, "attack", d.why) for d in defects if d.kind == "not attacked"]
        defl = [Deflection(x, sq, f"their {_sym(b, x)} on {G.name(x)} defends {G.name(sq)}") for d in defects if d.kind == "defended" for x in d.by]
        if len(jobs) > SKELETON_MAX_JOBS:
            dropped += 1
            continue
        arrs.append(_price(b, me, k, "win", sq, sq, jobs, defl, [], G.worth(pc), units, their_fast, bear, fast=fast))
    for pw in sorted(chess.SquareSet(b.pawns & b.occupied_co[me])):          # (3)(iii) the promotion square of each passer of ours: goal `promote`
        n = _pushes_of(b, pw, me)
        if n < 1 or not is_passer(b, pw):
            continue
        targets += 1
        try:
            defects, facts = promote_defects(skel_a, pw)
        except ValueError:
            continue                                                         # (a skeleton the Rust words cannot read: no arrangement, nothing said)
        jobs, defl, asides = [], [], []
        for d in defects:
            if d.kind in ("path attacked", "promo square attacked"):
                if d.sq not in [j.square for j in jobs]:
                    jobs.append(Job(d.sq, "promo" if d.kind == "promo square attacked" else "path", d.why))
            elif d.kind == "stopper":
                defl += [Deflection(x, d.sq, d.why) for x in d.by]
            elif d.kind == "blocked":
                o = skel_a.piece_at(d.sq)
                if o is not None and o.color == me:
                    asides.append((d.sq, f"{_sym(b, d.sq)} {G.name(d.sq)} off the pawn's path"))
                else:
                    defl.append(Deflection(d.sq, d.sq, d.why))               # (their unit on the path: it must be driven off -- a deflection, not a job)
        if len(jobs) > SKELETON_MAX_JOBS:
            dropped += 1
            continue
        arrs.append(_price(b, me, k, "promote", facts["promo"], pw, jobs, defl, asides, STAKE_PROMOTE, units, their_fast, bear, fast=fast, pawn_path_=list(facts["path"])))
    arrs.sort(key=lambda a: (_VERDICTS.index(a.verdict), -a.net if a.verdict != "unreachable" else 0, a.tempi, _GOALS.index(a.goal), a.about if a.about is not None else -1, a.placement or ()))      # (9): by verdict, then the stake rule's balance; at an equal balance the fewer tempi first
    return dict(line=skeleton_line(b, me, part), partition=part, targets=targets, dropped=dropped, arrangements=arrs)


def _price(b: chess.Board, me: bool, k: int, goal: str, target, about: int, jobs: list, defl: list, asides: list, stake: int, units: list, their_fast: list, bear, fast: "list | None" = None,
           place: "tuple | None" = None, ek: "int | None" = None, pawn_path_: "list | None" = None) -> Arrangement:
    """plan 72.8 (6)-(8): the priced assignment, the step-asides, the deflections, the stake rule -> the Arrangement with its verdict and its text."""
    slots = []                                                               # per slot: {unit: (material, tempi, path)}
    if place is not None:
        s, cands, land = place
        slots.append({u: c for u in cands if (c := land(u, s)) is not None})
    for j in jobs:
        slots.append({u: c for u in units if (c := bear(u, j.square)) is not None})
    names = (["placement"] if place is not None else []) + jobs
    first_kind = _sym(b, place[1][0]) if place is not None else None
    goal_text = (f"mate on {G.name(place[0])} ({first_kind} {G.name(place[1][0])}->{G.name(place[0])}, their king on {G.name(ek)})" if goal == "mate" else
                 f"win their {_sym(b, about)} on {G.name(about)}" if goal == "win" else f"promote our P on {G.name(about)}")
    base = dict(target=target, goal=goal, jobs=jobs, deflections=list(defl), stake=stake, about=place[0] if place is not None else about, goal_text=goal_text)
    for i, sl in enumerate(slots):                                           # 6b: a slot no unit reaches within k: UNREACHABLE, the job named
        if not sl:
            what = f"the placement on {G.name(place[0])}" if names[i] == "placement" else f"{G.name(names[i].square)} ({names[i].kind})"
            a = Arrangement(placement=(place[1][0], place[0]) if place is not None else None, assignment=[], step_asides=[], material=0, tempi=0, counter=0, verdict="unreachable",
                            why=f"no unit of ours reaches {what} within {k}", **base)
            a.text = arrangement_line(a, goal_text)
            return a
    best = None
    for perm in itertools.permutations(units, len(slots)):                   # 6b: injective assignments, the least (material, tempi), ties by the squares' order
        tot_m = tot_t = 0
        ok = True
        for sl, u in zip(slots, perm):
            c = sl.get(u)
            if c is None:
                ok = False
                break
            tot_m += c[0]
            tot_t += c[1]
        if ok and (best is None or (tot_m, tot_t, perm) < best):
            best = (tot_m, tot_t, perm)
    if best is None:
        a = Arrangement(placement=(place[1][0], place[0]) if place is not None else None, assignment=[], step_asides=[], material=0, tempi=0, counter=0, verdict="unreachable",
                        why=f"too few units of ours for {len(slots)} slot(s)", **base)
        a.text = arrangement_line(a, goal_text)
        return a
    tot_m, tot_t, perm = best
    assignment = [(u, names[i], *slots[i][u]) for i, u in enumerate(perm)]
    placement = (perm[0], place[0]) if place is not None else None
    if place is not None:
        base["goal_text"] = goal_text = f"mate on {G.name(place[0])} ({_sym(b, perm[0])} {G.name(perm[0])}->{G.name(place[0])}, their king on {G.name(ek)})"
    moving = set(perm)                                                       # 6c: a fast unit of ours with no job standing on a chosen path steps aside (+1 tempo each; what it does then is not priced)
    free = set(units if fast is None else fast) - moving                    # (every FAST unit of ours with no job, the ones past the unit fuse included)
    blockers = []
    for u, what, m_, t_, path in assignment:
        sqs = _path_squares(path, u, None if what == "placement" else what.square)
        blockers += [x for x in sorted(sqs) if x in free and x not in blockers]
    if pawn_path_:                                                           # (the pawn's own path on the live board: our fast unit standing on it is the 'blocked' defect the skeleton hides)
        blockers += [x for x in pawn_path_ if x in free and x not in blockers]
    step_asides = list(asides) + [(x, f"{_sym(b, x)} {G.name(x)} off the path") for x in blockers]
    tempi = tot_t + len(step_asides)
    deflections = list(defl)                                                 # 7: their FAST units that occupy or bear on a job square (or the placement square) on the live board
    jsq = ([place[0]] if place is not None else []) + [j.square for j in jobs]
    for t in jsq:
        for f in their_fast:
            if place is not None and t == place[0] and f == place[0]:
                continue                                                     # (the occupant of the placement square: the capture there removes it)
            if f == t or t in b.attacks(f):
                if not any(d.their_sq == f and d.job_square == t for d in deflections):
                    deflections.append(Deflection(f, t, f"their {_sym(b, f)} on {G.name(f)} {'stands on' if f == t else 'bears on'} {G.name(t)}"))
    mu = min(tempi, k + 1)                                                   # 8: THE STAKE RULE (72.3) -- their counter-plan within mu-1 free moves (the roll's fuse k binds their roll as ours)
    counter, ctext = 0, None
    if mu > 0:
        step = None
        for u, what, m_, t_, path in assignment:
            mv = _first_step(path)
            if mv is not None and mv in b.legal_moves:
                step = mv
                break
        if mu == 1 and step is None:
            mu = 2                                                           # (the one step is not a move on the live board: their free move is read by the roll)
        counter, ctext = counter_value(b, me, step, mu)
    base["deflections"] = deflections
    base["syms"] = {x: _sym(b, x) for x in set(perm) | {x for x, _t in step_asides} | {d.their_sq for d in deflections} if b.piece_at(x) is not None}
    a = Arrangement(placement=placement, assignment=assignment, step_asides=step_asides, material=tot_m, tempi=tempi, counter=counter, verdict="reachable", **base)
    if a.net < 1:
        a.verdict, a.why = "refused", (ctext or f"price {tot_m} against a stake of {stake}")
    elif deflections:
        a.verdict, a.why = "open", "deflection"
    a.text = arrangement_line(a, goal_text)
    return a


def arrangement_line(a: Arrangement, goal_text: "str | None" = None) -> str:
    """plan 72.8 (10) THE TRACE TEXT: 'arrangement <goal text>: Q d1->h7 (2), B e2->d3 bears on h7 (1), R a1 step-aside (+1), deflect their Q from d1 (open); price (0, 4), counter 0 -> open:
    deflection'. The goal text is the root question's own text where one exists."""
    goal_text = goal_text or a.goal_text
    if a.verdict == "unreachable":
        jobs = ", ".join(f"{G.name(j.square)} ({j.kind})" for j in a.jobs) or "no job"
        return f"arrangement {goal_text}: jobs {jobs}; price (-, -), counter - -> unreachable: {a.why}"
    verdict = a.verdict if a.verdict == "reachable" else ("open: deflection" if a.verdict == "open" else f"refused: {a.why}")
    return f"arrangement {goal_text}: {arrangement_items(a)}; price ({a.material}, {a.tempi}), counter {a.counter} -> {verdict}"


def arrangement_value(q: Question) -> int:
    """plan 72.8 (11d) THE POLICY FEATURE: the matched arrangement's stake - material - counter (`q.pattern["arrangement"]`, set by the agenda for a reachable one), else 0."""
    a = q.pattern.get("arrangement")
    return (q.stake - a["material"] - a["counter"]) if a else 0


def arrangement_items(a: Arrangement) -> str:
    """The unit items of the trace text: 'Q d1->h7 (2)' a landing, 'B e2->d3 bears on h7 (1)' a job, 'R a1 step-aside (+1)', 'deflect their Q from d1 (open)'."""
    items = []
    for u, what, m_, t_, path in a.assignment:
        land = chess.Move.from_uci(path[-1]).to_square if path else u
        nm = a.syms.get(u, "?")
        if what == "placement":
            items.append(f"{nm} {G.name(u)}->{G.name(land)} ({t_})")
        elif path:
            items.append(f"{nm} {G.name(u)}->{G.name(land)} bears on {G.name(what.square)} ({t_})")
        else:
            items.append(f"{nm} {G.name(u)} bears on {G.name(what.square)} (0)")
    items += [f"{a.syms.get(x, '?')} {G.name(x)} step-aside (+1)" for x, _txt in a.step_asides]
    seen = []
    for d in a.deflections:
        if d.their_sq not in seen:
            seen.append(d.their_sq)
            items.append(f"deflect their {a.syms.get(d.their_sq, '?')} from {G.name(d.their_sq)} (open)")
    return ", ".join(items) or "nothing to arrange"


def arrangement_steps(board: chess.Board, a: Arrangement) -> list:
    """plan 72.8 (11b/c): the arrangement's FIRST STEPS that are moves on the live board -> [(move, label, (material, tempi))], in the assignment's order; the label is 72.3's
    ('bring the B from e2 to bear on h7 in 1, price 0'; a placement LANDS: 'bring the Q from d1 to h7 in 2 (via h5), price 0')."""
    out, have = [], set()
    legal = set(board.legal_moves)
    for u, what, m_, t_, path in a.assignment:
        mv = _first_step(path)
        if mv is None or mv not in legal or mv in have:
            continue
        have.add(mv)
        pc = board.piece_at(u)
        nm = pc.symbol().upper() if pc.piece_type != chess.PAWN else "pawn"
        dest = f"to {G.name(a.placement[1])}" if what == "placement" else f"to bear on {G.name(what.square)}"
        label = f"bring the {nm} from {G.name(u)} {dest} in {t_}" + (f" (via {G.name(mv.to_square)})" if t_ > 1 else "") + f", price {m_}"
        out.append((mv, label, (m_, t_)))
    return out


def arrangement_fix(board: chess.Board, us: str | bool, a: Arrangement, q: Question, text: "str | None" = None) -> tuple:
    """plan 72.8 (11b): the arrangement's first steps as FIX ROUTES of the root question it matches, through 72.3's join machinery (a priced Route, its tempi in `route_cost`): a 'cover'
    sub-question 'bring the units of the arrangement into place', route text the forcing word, then 72.3's label with the arrangement's line in brackets. ONLY A FORCING STEP IS A ROUTE
    (72.3 corrected, 2026-10-23 14:05: a quiet step hands them a free move, so it is no fix -- the chase, the fuse and the count never take one); the quiet first steps are returned apart,
    candidates the agenda prints. No defect key: the arrangement fixes the skeleton's defects, not one defect of the live board, so the tempo arithmetic of the live defects is untouched.
    -> (the sub-question or None when no step is forcing, [the quiet steps' SAN])."""
    me = G.colour_of(us)
    b = G.with_turn(board, me) if board.turn != me else board
    text = text or a.text
    forcing, quiet = [], []
    for mv, label, price in arrangement_steps(b, a):
        why = forcing_why(b, mv, q.stake, me, mate_threats=True)
        if why:
            forcing.append(Route(mv, b.san(mv), f"{why}; {label} [{text}]", price=price))
        else:
            quiet.append(b.san(mv))
    if not forcing:
        return None, quiet
    f = new_question("cover", q.stake, dict(q.pattern), q, text=f"bring the units of the arrangement into place ({a.goal_text})")
    f.routes, f.unforced = forcing, list(quiet)
    for r in forcing:
        _JOIN[(b.board_fen(), r.move.uci())] = r.price
    return f, quiet
