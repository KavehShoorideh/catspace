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
           "discovered_attack", "battery", "overloaded", "attacks_king_zone")
TEMPO_BASELINE_CP = 100
REAL_THREAT_MARGIN_CP = 50      # a threat must be worth this much beyond a bare tempo to be 'real'
ZUGZWANG_MARGIN_CP = 30


@dataclass
class Effect:
    ply: int
    san: str
    uci: str
    mover: bool                      # color that moved
    captured: tuple[str, int] | None # (piece name, exchange value)
    created: list[Relation] = field(default_factory=list)
    destroyed: list[Relation] = field(default_factory=list)
    gives_check: bool = False


_SYMVAL = {"p": 1, "n": 3, "b": 3, "r": 5, "q": 9, "k": 100}
DESTROYED_NOTABLE = ("hanging", "pin", "fork", "skewer", "overloaded", "attacked_by_lower_value")


def _val(sym: str) -> int:
    return _SYMVAL.get(sym.lower(), 0) if sym != "-" else 0


def significant(r: Relation) -> bool:
    """A relation worth naming: line relations need a valuable far target, 'loose' needs a
    piece, king-zone counts are a summary statistic (kept in L3 for gauges, not in effects)."""
    if r.name in ("discovered_attack", "battery", "xray"):
        return "check" in r.tags or _val(r.pieces[2]) >= 3
    if r.name == "pin":      # a pinned pawn shielding a minor piece is not a fact a player names
        return "absolute" in r.tags or _val(r.pieces[1]) >= 3 or _val(r.pieces[2]) >= 9
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
    created = [after[k] for k in after.keys() - before.keys()]
    destroyed = [before[k] for k in before.keys() - after.keys() if before[k].name in DESTROYED_NOTABLE]
    return Effect(ply, san, move.uci(), mover, captured, created, destroyed, check)


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

    def san_line(self, board: chess.Board, n: int = 6) -> str:
        nb = board.copy(stack=False); nb.push(chess.Move.null())
        return self.line.san_line(nb, n)


def threats(board: chess.Board, engine: Engine, *, multipv: int = 3, baseline: int = TEMPO_BASELINE_CP) -> list[Threat]:
    """Standing threats: the opponent's best lines given a free move, scored above the tempo baseline."""
    now = engine.best(board)
    probe = engine.null_move_probe(board, multipv=multipv)
    if now is None or not probe:
        return []
    nb = board.copy(stack=False); nb.push(chess.Move.null())
    out = []
    for l in probe:
        gain = l.cp + now.cp - baseline      # opp POV after pass + our POV now = tempo value
        eff = move_effects(nb, chess.Move.from_uci(l.move)) if l.pv else None
        out.append(Threat(l, gain, gain >= REAL_THREAT_MARGIN_CP or l.mate is not None, eff))
    return out


@dataclass
class Zugzwang:
    flagged: bool
    best_cp: int
    pass_cp: int      # our eval if we could pass


def zugzwang(board: chess.Board, engine: Engine, margin: int = ZUGZWANG_MARGIN_CP) -> Zugzwang | None:
    now = engine.best(board)
    probe = engine.null_move_probe(board, multipv=1)
    if now is None or not probe:
        return None
    pass_cp = -probe[0].cp
    return Zugzwang(pass_cp > now.cp + margin, now.cp, pass_cp)
