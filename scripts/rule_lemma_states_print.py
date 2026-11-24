"""plan 79.44: writes the fixture states of the first game's lemmas, rules/lemmas/chess/states.txt (the fixtures of a lemma are the states its certificate reads besides the seeded games).

Positions come from the kernel certificates' own corpora (discovery/certify.py: the rule corners, the seeded random endgames, every 25th puzzle position) and from made positions
that bring the cases of the lemmas into the corpus (a unit held on a line, a king moving into an attack, en passant that opens a line). Each is written as the state's fluents of the
description (discovery/gdl_chess.py state_of), one state a line.

  .venv/bin/python scripts/rule_lemma_states_print.py            # writes the fixtures, prints OBS lines
"""
from __future__ import annotations

import chess

from catspace.incubator.analyst.discovery import certify as CF
from catspace.incubator.analyst.discovery import gdl_chess as GC
from catspace.incubator.analyst.discovery import rule_lemma_certify as RC
from catspace.io import paths

OUT = paths.REPO_ROOT / "rules" / "lemmas" / "chess" / "states.txt"

# made positions: (note, fen)
MADE = [
    ("a unit alone between the king and a rook on a file", "4k3/4r3/8/8/4N3/8/8/4K3 w - - 0 1"),
    ("a unit alone between the king and a bishop on a diagonal", "4k3/8/8/8/1b6/8/3N4/4K3 w - - 0 1"),
    ("two units between the king and a rook: no pin", "4k3/4r3/8/8/4N3/4B3/8/4K3 w - - 0 1"),
    ("a pinned rook that may slide along its line", "4k3/4r3/8/8/4R3/8/8/4K3 w - - 0 1"),
    ("the king steps next to a king and onto an attacked square", "8/8/8/8/8/2k5/8/K7 w - - 0 1"),
    ("the king may take a defended unit or an undefended one", "8/8/8/8/8/1pn5/r7/K7 w - - 0 1"),
    ("en passant opens a diagonal to the king", "b3k3/8/8/3pP3/8/8/8/7K w - d6 0 2"),
    ("en passant takes both units from the king's row", "4k3/8/8/KPp4r/8/8/8/8 w - c6 0 1"),
    ("en passant takes both units from the king's row, the other way", "4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 1"),
    ("en passant where the captured unit stands nearer the king than the capturer on the king's row", "4k3/8/8/KpP4r/8/8/8/8 w - b6 0 1"),     # 2026-11-24: plan 79.45 found that no state reached the last clause of the lemma of the cases
    ("a pawn pinned on a diagonal may take the pinner and may not leave", "4k3/8/8/8/8/2b5/3P4/4K3 w - - 0 1"),
    ("a pawn pinned on a file may push", "4k3/4r3/8/8/8/8/4P3/4K3 w - - 0 1"),
    ("castling with the rook attacked and the king safe", "r3k2r/8/8/8/8/8/6b1/R3K2R w KQkq - 0 1"),
    ("black to move, a pinned knight", "4k3/4n3/8/8/8/8/4R3/4K3 b - - 0 1"),
    ("a lone rook on its file, nothing else on the file", "4k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
    ("a lone rook on its file, a unit beside it", "4k3/8/8/8/8/8/1P6/R3K3 w - - 0 1"),
    ("two lone rooks of the two sides", "r3k3/8/8/8/8/8/8/R3K3 w - - 0 1"),
]


def fens() -> list:
    out = [("a rule corner", f) for f in CF.RULE_CORNERS] + list(MADE)
    out += [("a seeded endgame", f) for f in CF.endgame_fens(120)]
    out += [("a puzzle position", f) for f in CF.puzzle_fens()[:150]]
    return out


def main() -> int:
    lines = ["; the fixtures of the lemmas of this game (scripts/rule_lemma_states_print.py): one state a line, the fluents of the description without `true`; a line that starts with `;` is a note"]
    n = 0
    for note, fen in fens():
        b = chess.Board(fen)
        if not b.is_valid():
            continue
        lines.append(f"; {note}: {fen}")
        lines.append(RC.state_text(GC.state_of(b)))
        n += 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    print(f"OBS fixtures: {n} states written to {OUT.relative_to(paths.REPO_ROOT)} ({OUT.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
