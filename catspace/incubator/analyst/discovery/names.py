"""PLAN 75.2 (c) THE NAMES TABLE (Kaveh 2026-10-27: "some chess vocab has to be provided by us because we want the engine to be able to talk in the same language as we do. We
don't want it to learn everything. But from those basic things -- attacking, moving and sliding pieces, winning and drawing -- everything else should be derived from the basic
rules"). Every word WE SUPPLY, bound to the RULE KERNEL's definition of it (the python-chess / Rust predicate the engine calls: `kernel`) and to its LEAN TWIN (a definition of
lean/ideas by its full name, present in docs/LEAN_INDEX.md: `lean`). A name is never bound to a derived fact: the unit worths are the material VALUATION entry of the lemma
store, the goal kinds its GOAL entries, the stops its STOP entries; the outcome's rungs (losing / level / winning) are the worklist's ladder.

The store's `reads` (movement, attack, tempo, material) refer to this table (`READS`: each base word is a name here); the agenda reads the names it needs from here (the unit
kinds by their kernel kinds, the royal unit, the king's step) and the trace prints the kernel's own spelling (square names, unit letters, move notation).

  python -m catspace.incubator.analyst.discovery.names      # OBS lines: the names, their kernel predicates, their Lean twins (present / missing in the index)
"""
from __future__ import annotations

from dataclasses import dataclass

import chess


@dataclass(frozen=True)
class Name:
    word: str            # the word as we say it (the trace's, the store's)
    kernel: str          # the predicate or constant of the rule kernel the engine calls for it (python-chess's, the Game protocol's, the Rust core's)
    lean: "str | None"   # its Lean twin: a definition of lean/ideas by its full name, present in docs/LEAN_INDEX.md (None: no twin yet -- a finding)
    means: str           # one plain sentence


NAMES: dict = {n.word: n for n in (
    # the unit kinds (the kernel's piece types; Lean names a kind through the squares of its units)
    Name("pawn", "chess.PAWN", "Ideas.Vocabulary.kindSet", "the unit that pushes forward and promotes on its last rank (Game.promotes)"),
    Name("knight", "chess.KNIGHT", "Ideas.Vocabulary.kindSet", "the unit that hops (Ideas.Vocabulary.knight1)"),
    Name("bishop", "chess.BISHOP", "Ideas.Vocabulary.kindSet", "the unit that slides on the diagonals"),
    Name("rook", "chess.ROOK", "Ideas.Vocabulary.kindSet", "the unit that slides on the ranks and files"),
    Name("queen", "chess.QUEEN", "Ideas.Vocabulary.kindSet", "the unit that slides on both"),
    Name("king", "chess.KING", "Ideas.Vocabulary.kindSet", "the ROYAL unit (Game.royal): its loss is the terminal, it is worth nothing in the valuation; it steps to the adjacent squares (Game.adjacent)"),
    # the board
    Name("square", "chess.SQUARES", "Ideas.Vocabulary.sqIndex", "a location of the board (Game.locations), named by the kernel's square name (Game.loc_name)"),
    Name("rank", "chess.square_rank", "Ideas.Vocabulary.rankOf", "the row of a square"),
    Name("file", "chess.square_file", "Ideas.Vocabulary.fileOf", "the column of a square"),
    # the rules' verbs
    Name("move", "chess.Board.legal_moves", "Ideas.Game.Game.step", "a legal move of the side to move (Game.moves, Game.apply)"),
    Name("attack", "chess.Board.is_attacked_by", "Ideas.Vocabulary.attackedBy", "a unit bears on a square (Game.attackers, Game.attacks_from)"),
    Name("slide", "chess.Board.attacks", "Ideas.Vocabulary.lineThrough", "a unit's move along a line of empty squares (the kernel's rays, blockers stop them)"),
    Name("capture", "chess.Board.is_capture", "Ideas.Vocabulary.seeOf", "a move onto a unit of the other side, which leaves the board (Game.is_capture)"),
    Name("check", "chess.Board.is_check", "Ideas.Vocabulary.checkOf", "the side to move's royal unit is attacked (Game.in_check, Game.gives_check)"),
    Name("promotion", "chess.Move.promotion", "Ideas.Vocabulary.promoOf", "a pawn on its last rank is replaced by another unit (Game.promotes)"),
    Name("tempo", "chess.Board.ply", "Ideas.Game.chess", "whose move it is and a count of moves (Game.turn, the board's ply count; the pass flips the side to move, Game.pass_)"),
    # the outcomes
    Name("mate", "chess.Board.is_checkmate", "Ideas.Game.Game.mated", "the mover is in check and has no legal move: the game ends (Game.terminal says 'mate')"),
    Name("stalemate", "chess.Board.is_stalemate", "Ideas.Game.Game.stalemated", "the mover has no legal move and is not in check: a draw (Game.terminal says 'stalemate')"),
    Name("draw", "chess.Board.is_insufficient_material", "Ideas.Vocabulary.deadDraw", "the game ends level (Game.terminal says 'draw'); the repetition and the fifty-move draws are NOT in the kernel's terminal (a finding of 75.2)"),
    Name("win", "chess.Board.is_checkmate", "Ideas.Vocabulary.gainGoal", "the other side is mated, or a gain held ahead (the ladder's winning rung)"),
    Name("lose", "chess.Board.is_checkmate", "Ideas.Game.Game.mated", "we are mated, or behind (the ladder's losing rung)"),
    # the count the stakes are in (the worths are the VALUATION entry's, never this table's)
    Name("material", "analyst_core.Pos.balance", "Ideas.Vocabulary.materialOf", "the side's units counted by the material VALUATION (Game.measure; the worths are the store's)"),
)}

READS = {"movement": "move", "attack": "attack", "tempo": "tempo", "material": "material"}      # the store's base reads (lemma_store.BASE) -> the names they are


KIND = {"p": chess.PAWN, "n": chess.KNIGHT, "b": chess.BISHOP, "r": chess.ROOK, "q": chess.QUEEN, "k": chess.KING}      # a unit kind's letter (the valuation's keys) -> the kernel's kind
ROYAL = chess.KING                       # the royal unit's kind (Game.royal)
PAWN = chess.PAWN                        # the unit that promotes (Game.promotes)


def kind(letter: str) -> int:
    """The kernel's kind of a unit letter (either case)."""
    return KIND[letter.lower()]


def royal(piece) -> bool:
    """Is the unit the royal one (the king)?"""
    return piece is not None and piece.piece_type == ROYAL


def pawn(piece) -> bool:
    """Is the unit the one that promotes (a pawn)?"""
    return piece is not None and piece.piece_type == PAWN


CHECK_MARKS = ("+", "#")                 # the kernel's notation (python-chess / the Rust core's SAN) marks a check with + (a mate with #) ...
CAPTURE_MARK = "x"                       # ... and a capture with x


def san_kind(san: str) -> str:
    """What a move written in the kernel's notation is: 'check', 'capture' or 'quiet' (its marks: CHECK_MARKS, CAPTURE_MARK)."""
    return "check" if san.endswith(CHECK_MARKS) else "capture" if CAPTURE_MARK in san else "quiet"


def move_number(token: str) -> bool:
    """Is a token of a written line the notation's move number ('12.', '3...'), not a move?"""
    return token[:1].isdigit()


def lean_index(path=None) -> dict:
    from catspace.incubator.analyst.discovery import lemma_store as LS
    return LS.lean_index() if path is None else LS.lean_index(path)


def problems(idx: "dict | None" = None) -> list:
    """What is wrong with the table: a kernel binding that does not resolve, a Lean twin absent from docs/LEAN_INDEX.md, a base read that is no name."""
    import importlib
    idx = lean_index() if idx is None else idx
    bad = []
    for n in NAMES.values():
        mod, _, rest = n.kernel.partition(".")
        try:
            obj = importlib.import_module(mod)
            for part in rest.split("."):
                obj = getattr(obj, part)
        except (ImportError, AttributeError) as x:
            bad.append(f"{n.word}: kernel {n.kernel} does not resolve ({x})")
        if n.lean is not None and n.lean not in idx:
            bad.append(f"{n.word}: Lean twin {n.lean} absent from docs/LEAN_INDEX.md")
    for b, w in READS.items():
        if w not in NAMES:
            bad.append(f"base read {b}: {w} is no name")
    return bad


def report(log=print) -> list:
    idx = lean_index()
    lines = [f"OBS names: {len(NAMES)} names; with a Lean twin present in the index {sum(1 for n in NAMES.values() if n.lean in idx)}, "
             f"without one {sum(1 for n in NAMES.values() if n.lean is None)}, twin missing from the index {sum(1 for n in NAMES.values() if n.lean is not None and n.lean not in idx)}"]
    for n in NAMES.values():
        lines.append(f"OBS name {n.word}: kernel {n.kernel}; Lean {n.lean or 'none'}" + ("" if n.lean is None or n.lean in idx else " (ABSENT from the index)"))
    for x in problems(idx):
        lines.append("  PROBLEM " + x)
    for x in lines:
        log(x)
    return lines


if __name__ == "__main__":
    report()
