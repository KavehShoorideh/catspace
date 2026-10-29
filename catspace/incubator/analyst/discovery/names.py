"""PLAN 75.2 (c) THE NAMES TABLE (Kaveh 2026-10-27: "some chess vocab has to be provided by us because we want the engine to be able to talk in the same language as we do. We
don't want it to learn everything. But from those basic things -- attacking, moving and sliding pieces, winning and drawing -- everything else should be derived from the basic
rules"). Every word WE SUPPLY, bound to the RULE KERNEL's definition of it (the python-chess / Rust predicate the engine calls: `kernel`) and to its LEAN TWIN (a definition of
lean/ideas by its full name, present in docs/LEAN_INDEX.md: `lean`). A name is never bound to a derived fact: the unit worths are the material VALUATION entry of the lemma
store, the goal kinds its GOAL entries, the stops its STOP entries; the outcome's rungs (losing / level / winning) are the worklist's ladder.

The store's `reads` (movement, attack, tempo, material) refer to this table (`READS`: each base word is a name here); the agenda reads the names it needs from here (the unit
kinds by their kernel kinds, the royal unit, the king's step) and the trace prints the kernel's own spelling (square names, unit letters, move notation).

PLAN 76.1 (a): the table's words are entries of THE ONTOLOGY SHELF (ontology.py, lemmas/ontology/: one name, its bindings -- the spec's relation, the Rust core's, the
Lean definition, the kernel predicate). This module is the thin accessor: `NAMES` is read from the shelf's entries that are words of the table (`names: true`), in the
table's order (`WORDS`); `kernel` is the entry's python binding, `lean` its Lean binding. A word of WORDS with no entry is an error at import.

  python -m catspace.incubator.analyst.discovery.names      # OBS lines: the names, their kernel predicates, their Lean twins (present / missing in the index)
"""
from __future__ import annotations

from dataclasses import dataclass

import chess

from catspace.incubator.analyst.discovery import ontology as ON


@dataclass(frozen=True)
class Name:
    word: str            # the word as we say it (the trace's, the store's)
    kernel: str          # the predicate or constant of the rule kernel the engine calls for it (python-chess's, the Game protocol's, the Rust core's)
    lean: "str | None"   # its Lean twin: a definition of lean/ideas by its full name, present in docs/LEAN_INDEX.md (None: no twin yet -- a finding)
    means: str           # one plain sentence


# the table's words in its order: the unit kinds (the kernel's piece types; Lean names a kind through the squares of its units); the board; the rules' verbs; the
# outcomes; the count the stakes are in (the worths are the VALUATION entry's, never this table's)
WORDS = ("pawn", "knight", "bishop", "rook", "queen", "king", "square", "rank", "file", "move", "attack", "slide", "capture", "check", "promotion", "tempo",
         "mate", "stalemate", "draw", "draw by repetition", "draw by the fifty-move rule", "win", "lose", "material")


def _names() -> dict:
    shelf = {e["name"]: e for e in ON.names_table()}
    missing = [w for w in WORDS if w not in shelf]
    if missing or len(shelf) != len(WORDS):
        raise ON.OntologyError(f"the names table and the shelf differ: words with no entry {missing}, entries not in the table {sorted(set(shelf) - set(WORDS))}")
    return {w: Name(w, shelf[w]["python"], shelf[w]["lean"], shelf[w]["means"]) for w in WORDS}


NAMES: dict = _names()

READS = {"movement": "move", "attack": "attack", "tempo": "tempo", "material": "material"}      # the store's base reads (lemma_store.BASE) -> the names they are
DRAW_BY = {"repetition": "draw by repetition", "fifty-move": "draw by the fifty-move rule"}      # plan 75.4 (1): Game.terminal's draws by rule (game.DRAWS_BY_RULE) -> the names they are
DRAW = NAMES["draw"].word                # plan 75.4 (1): the outcome every draw reaches (the name a draw claim carries as its kind and goal)
EVENT_CAPTURE = NAMES["capture"].word    # plan 75.4 (3): the fifty-move rule's events as the table names them (Game.clock_events): a capture ...
EVENT_PAWN = NAMES["pawn"].word          # ... and a move of the pawn


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


# plan 76.2 (b): what a move IS (a check, a capture, a quiet move) is read by the Game surface (`Game.reply_kind`: the kernel's `gives_check` and `is_capture`), no
# longer off the notation's marks ('+' / '#', 'x'): the notation is printed, never read for a rule (the marks' reading is kept as the test oracle of
# tests/test_rules_shelf.py).


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
        parts = n.kernel.split(".")
        try:
            obj, k = None, 0
            for i in range(len(parts), 0, -1):                  # the longest importable module of the dotted name (plan 75.4: a Game method of ours), then its attributes
                try:
                    obj, k = importlib.import_module(".".join(parts[:i])), i
                    break
                except ImportError:
                    continue
            if obj is None:
                raise ImportError(f"no module in {n.kernel}")
            for part in parts[k:]:
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
