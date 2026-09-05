"""L4 -- DYNAMICS: how L3 relations change under moves.
  * EFFECTS of a move / a line: captures, relations created and destroyed, per ply.
  * THREATS via the null-move probe: what the opponent gains from a free tempo, ABOVE the tempo
    baseline (a free move is worth ~+100 cp even when nothing is on; measured 2026-09-04).
  * ZUGZWANG: passing would beat our best move.
No objective is consulted; engine numbers are carried through as measurements only."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from .facts import value
from .relations import Relation, relations
from .engine import Engine, Line

NOTABLE = ("hanging", "loose", "attacked_by_lower_value", "pin", "skewer", "xray", "fork",
           "discovered_attack", "battery", "overloaded", "attacks_king_zone",
           "disabled_attack", "disabled_defense", "pinned_and_attacked", "hanging_through_pin")
TEMPO_BASELINE_CP = 100
REAL_THREAT_MARGIN_CP = 30      # a threat must be worth this much beyond a bare tempo to be 'real'
ZUGZWANG_MARGIN_CP = 30


# which square a relation is ABOUT (its subject) -- used to tell how a relation ended
SUBJECT = {"hanging": 0, "loose": 0, "attacked_by_lower_value": 1, "pin": 1, "skewer": 1, "xray": 1,
           "fork": 0, "discovered_attack": 1, "battery": 1, "overloaded": 0, "disabled_attack": 1,
           "disabled_defense": 1, "pinned_and_attacked": 1, "hanging_through_pin": 0, "attacks_king_zone": 0}


@dataclass
class Effect:
    """What ONE move does, in L3 vocabulary. The effect side of Kaveh's two representations:
    L3 says a structure exists; here we say what a move did with it -- USED it (a discovered
    attack executed), RESOLVED it by capture (took the piece that could not flee), ESCAPED it
    (moved the subject out of it), CREATED or ENDED it."""
    ply: int
    san: str
    uci: str
    mover: bool                      # color that moved
    captured: tuple[str, int] | None # (piece name, exchange value)
    created: list[Relation] = field(default_factory=list)
    destroyed: list[Relation] = field(default_factory=list)   # ended by other means
    gives_check: bool = False
    uses: list[Relation] = field(default_factory=list)        # discovered attacks / batteries executed
    resolved: list[Relation] = field(default_factory=list)    # ended because the subject was captured
    escaped: list[Relation] = field(default_factory=list)     # ended because the subject moved away


_SYMVAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 100}
DESTROYED_NOTABLE = ("hanging", "pin", "fork", "skewer", "overloaded", "attacked_by_lower_value",
                     "disabled_attack", "disabled_defense", "pinned_and_attacked", "hanging_through_pin")


def _val(sym: str) -> int:
    return _SYMVAL.get(sym.lower(), 0) if sym != "-" else 0


def significant(r: Relation) -> bool:
    """A relation worth naming: line relations need a valuable far target, 'loose' needs a
    piece, king-zone counts are a summary statistic (kept in L3 for gauges, not in effects)."""
    if r.name in ("discovered_attack", "battery", "xray"):
        return "check" in r.tags or _val(r.pieces[2]) >= 3
    if r.name == "pin":      # a pinned pawn shielding a minor piece is not a fact a player names
        return "absolute" in r.tags or _val(r.pieces[1]) >= 3 or _val(r.pieces[2]) >= 9
    if r.name in ("disabled_attack", "disabled_defense"):   # a pinned knight unable to take a pawn is minor
        return _val(r.pieces[3]) >= 3 and _val(r.pieces[1]) >= 3
    if r.name == "pinned_and_attacked":                      # pawns do not "flee"; a pinned pawn is a pin fact
        return _val(r.pieces[1]) >= 3
    if r.name == "loose":
        return _val(r.pieces[0]) >= 3
    if r.name == "attacks_king_zone":
        return False
    return r.name in NOTABLE


def _notable(rels: list[Relation]) -> dict[tuple, Relation]:
    return {r.key: r for r in rels if significant(r)}


def move_effects(board: chess.Board, move: chess.Move, ply: int = 1) -> Effect:
    before = _notable(relations(board))
    cap = board.piece_at(move.to_square)
    if board.is_en_passant(move):
        cap = chess.Piece(chess.PAWN, not board.turn)
    captured = (chess.piece_name(cap.piece_type), value(board, move.to_square) if not board.is_en_passant(move) else 1) if cap else None
    san, mover, check = board.san(move), board.turn, board.gives_check(move)
    after_b = board.copy(stack=False)
    after_b.push(move)
    after = _notable(relations(after_b))
    created = [after[k] for k in after.keys() - before.keys()
               if not (after[k].name in ("hanging", "loose") and after[k].squares[0] == move.to_square)]
    gone = [before[k] for k in before.keys() - after.keys() if before[k].name in DESTROYED_NOTABLE]
    uses = [r for r in before.values() if r.name in ("discovered_attack", "battery") and r.squares[1] == move.from_square]
    resolved, escaped, ended = [], [], []
    for r in gone:
        subj = r.squares[SUBJECT.get(r.name, 0)]
        actor_cashes = (r.name in ("fork", "attacked_by_lower_value", "skewer", "pin", "xray")
                        and r.squares[0] == move.from_square and move.to_square in r.squares[1:])
        if actor_cashes:                       # the attacker executed its own threat
            uses.append(r)
        elif cap is not None and subj == move.to_square:
            resolved.append(r)
        elif subj == move.from_square:
            escaped.append(r)
        else:
            ended.append(r)
    return Effect(ply, san, move.uci(), mover, captured, created, ended, check, uses, resolved, escaped)


def line_effects(board: chess.Board, pv, max_plies: int = 6) -> list[Effect]:
    t, out = board.copy(stack=False), []
    for i, m in enumerate(pv[:max_plies], 1):
        m = chess.Move.from_uci(m) if isinstance(m, str) else m
        out.append(move_effects(t, m, i))
        t.push(m)
    return out


@dataclass
class Threat:
    line: Line                       # the opponent's line if we pass
    gain_cp: int                     # what the free tempo is worth to them, above baseline
    real: bool                       # gain above baseline -> must be answered
    first_effects: Effect | None     # what their first move does
    structure: Relation | None = None   # the L3 fact that names the threat (hanging piece, fork, ...), if any

    def san_line(self, board: chess.Board, n: int = 6) -> str:
        nb = board.copy(stack=False); nb.push(chess.Move.null())
        return self.line.san_line(nb, n)


STRUCTURAL = ("hanging", "attacked_by_lower_value", "pinned_and_attacked", "hanging_through_pin", "overloaded",
              "fork", "skewer", "discovered_attack", "battery", "disabled_defense")


def structural_threats(board: chess.Board) -> list[tuple[Relation, chess.Move | None]]:
    """L3 facts that are THREATS against the side to move, each with the opponent's natural executing
    move on the null-moved board (attacker takes victim), or None when the fact has no single move."""
    us = board.turn
    nb = board.copy(stack=False); nb.push(chess.Move.null())
    out = []
    for r in relations(board):
        if r.name not in STRUCTURAL or not significant(r):
            continue
        victim_side = r.name in ("hanging", "pinned_and_attacked", "hanging_through_pin", "overloaded", "disabled_defense")
        ours = (r.owner == us) if victim_side else (r.owner != us)
        if not ours:
            continue
        mv = None
        if r.name == "hanging":
            atk = list(board.attackers(not us, r.squares[0]))
            if atk:
                a = min(atk, key=lambda q: value(board, q)); mv = chess.Move(a, r.squares[0])
        elif r.name == "attacked_by_lower_value":
            mv = chess.Move(r.squares[0], r.squares[1])
        elif r.name == "pinned_and_attacked":
            mv = chess.Move(r.squares[3], r.squares[1])
        elif r.name == "hanging_through_pin":
            atk = list(board.attackers(not us, r.squares[0]))
            if atk:
                a = min(atk, key=lambda q: value(board, q)); mv = chess.Move(a, r.squares[0])
        elif r.name in ("fork",):
            t = max(r.squares[1:], key=lambda q: value(board, q)); mv = chess.Move(r.squares[0], t)
        elif r.name in ("skewer", "battery"):
            mv = chess.Move(r.squares[0], r.squares[1])
        if mv is not None and mv not in nb.legal_moves:
            promo = chess.Move(mv.from_square, mv.to_square, promotion=chess.QUEEN)
            mv = promo if promo in nb.legal_moves else None
        out.append((r, mv))
    return out


def threats(board: chess.Board, engine: Engine, *, multipv: int = 5, baseline: int = TEMPO_BASELINE_CP,
            now: Line | None = None, probe: list[Line] | None = None, structural: bool = True) -> list[Threat]:
    """Standing threats = the opponent's best lines given a free move (engine, above the tempo baseline)
    MERGED with the L3 structural facts against us (a hanging piece, a fork, ...), each verified by a
    restricted evaluation of its executing move on the null-moved board. Ranked by materialized gain.
    `now` / `probe` may be passed in to reuse lines the caller already has."""
    now = now or engine.best(board)
    probe = probe if probe is not None else engine.null_move_probe(board, multipv=multipv)
    if now is None or probe is None:
        return []
    if now.mate is not None and now.mate > 0:      # we are mating: the tempo arithmetic is meaningless
        return []
    nb = board.copy(stack=False); nb.push(chess.Move.null())
    out, seen = [], set()
    for l in probe:
        gain = l.cp + now.cp - baseline      # opp POV after pass + our POV now = tempo value
        eff = move_effects(nb, chess.Move.from_uci(l.move)) if l.pv else None
        out.append(Threat(l, gain, gain >= REAL_THREAT_MARGIN_CP or l.mate is not None, eff))
        seen.add(l.move)
    if structural:
        pdepth = max(6, engine.depth - 4)
        for r, mv in structural_threats(board):
            if mv is None:
                continue
            if mv.uci() in seen:                       # the engine already listed it: just name the structure
                for t in out:
                    if t.line.move == mv.uci() and t.structure is None:
                        t.structure = r
                continue
            l = engine.eval_move(nb, mv, depth=pdepth)
            if l is None:
                continue
            gain = l.cp + now.cp - baseline
            out.append(Threat(l, gain, gain >= REAL_THREAT_MARGIN_CP or l.mate is not None, move_effects(nb, mv), r))
            seen.add(mv.uci())
    out.sort(key=lambda t: (t.line.mate is not None and t.line.mate > 0, t.gain_cp), reverse=True)
    return out


@dataclass
class Zugzwang:
    flagged: bool
    best_cp: int
    pass_cp: int      # our eval if we could pass


def zugzwang(board: chess.Board, engine: Engine, margin: int = ZUGZWANG_MARGIN_CP,
             now: Line | None = None, probe: list[Line] | None = None) -> Zugzwang | None:
    now = now or engine.best(board)
    probe = probe if probe is not None else engine.null_move_probe(board, multipv=1)
    if now is None or not probe:
        return None
    pass_cp = -probe[0].cp
    return Zugzwang(pass_cp > now.cp + margin, now.cp, pass_cp)
