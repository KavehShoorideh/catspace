"""PLAN 78.56: the game-free core of the Lean library (lean/ideas/Ideas/Core/).

The core imports no chess file, directly or through any file it imports; it holds no `sorry` and no new axiom; the old names of the moved results are still reachable (the
index lists them under their old full names, in the core file); and the board structure of stage C has its five properties stated once.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "ideas"
CORE = LEAN / "Ideas" / "Core"

IMPORT = re.compile(r"^import\s+(\S+)", re.M)


def _imports(path: Path) -> list:
    return IMPORT.findall(path.read_text())


def _module_path(mod: str) -> Path:
    return LEAN / (mod.replace(".", "/") + ".lean")


def _closure(start: list) -> set:
    """Every module imported, transitively, by the given project files (Mathlib and the like are leaves: not followed)."""
    seen, todo = set(), list(start)
    while todo:
        mod = todo.pop()
        if mod in seen:
            continue
        seen.add(mod)
        p = _module_path(mod)
        if p.exists():
            todo.extend(_imports(p))
    return seen


def test_the_core_files_exist():
    names = {p.name for p in CORE.glob("*.lean")}
    for n in ("ForcedByMeasure.lean", "Race.lean", "Game.lean", "Frame.lean", "Draws.lean", "Forcing.lean", "Plan.lean", "Board.lean", "Prelude.lean"):
        assert n in names, n


def test_the_core_import_closure_has_no_chess_file():
    """Every import of every core file, transitively, is a core file or Mathlib: no `Chess.*`, no `Ideas.*` outside `Ideas.Core.*`."""
    start = []
    for p in CORE.glob("*.lean"):
        start.extend(_imports(p))
    closure = _closure(start)
    assert closure, "the core imports nothing at all? the check is void"
    for mod in closure:
        assert mod.startswith("Ideas.Core.") or mod.startswith("Mathlib") or mod.startswith("Init") or mod.startswith("Std") or mod.startswith("Lean"), \
            f"the core reaches {mod}, which is not a core or a Mathlib file"
        assert not mod.startswith("Chess"), mod
    # and the core's own sources name no chess type
    for p in CORE.glob("*.lean"):
        text = p.read_text()
        code = re.sub(r"/-.*?-/", "", text, flags=re.S)           # docstrings and comments name chess; code does not
        for word in ("Position", "legalMoves", "applyMove", "Chess.", "pieceAttacks", "Square"):
            assert word not in code, f"{p.name} uses the chess word {word} in code"


def test_the_core_has_no_sorry_and_no_axiom():
    for p in CORE.glob("*.lean"):
        code = re.sub(r"/-.*?-/", "", p.read_text(), flags=re.S)
        code = re.sub(r"--.*", "", code)
        assert not re.search(r"\bsorry\b", code), p.name
        assert not re.search(r"^\s*(axiom|constant|opaque)\b", code, re.M), p.name
        assert "native_decide" not in code, p.name                                                # no enumeration over places in the core


def _index() -> dict:
    """{full name: file} read from docs/LEAN_INDEX.md (the generated index)."""
    out, cur = {}, None
    for ln in (ROOT / "docs" / "LEAN_INDEX.md").read_text().splitlines():
        if ln.startswith("## "):
            cur = ln[3:].strip()
        m = re.match(r"- \*\*(?:OPEN )?\w+ `([^`]+)`\*\*", ln)
        if m and cur:
            out[m.group(1)] = cur
    return out


def test_the_old_names_resolve_in_the_index_to_the_core_files():
    idx = _index()
    for old in ("Ideas.Forced.forced_by_measure", "Ideas.Race.race_sound", "Ideas.Race.frame_sound_pass", "Ideas.Race.far_stays_off", "Ideas.Race.far_not_bearing",
                "Ideas.Race.deny_sound", "Ideas.Plan.plan_sound", "Ideas.Frame.neglect", "Ideas.Game.forced_cycle_threefold", "Ideas.Forcing.claim_mono",
                "Ideas.Game.Game.race_sound"):
        assert old in idx, old
        assert "lean/ideas/Ideas/Core/" in idx[old], (old, idx[old])
    for new in ("Ideas.Core.pathDist_step", "Ideas.Core.Board.frame_unchanged", "Ideas.Core.Board.path_lemma", "Ideas.BoardInstance.chessBoard_laws"):
        assert new in idx, new


def test_the_board_structure_states_the_five_properties_once():
    text = (CORE / "Board.lean").read_text()
    for field in ("touch :", "bears_occ :", "line :", "footprint_refl :", "bears_footprint :", "between_footprint :", "goal_safe :", "dist_self :", "dist_step :"):
        assert len(re.findall(r"^\s*" + re.escape(field), text, re.M)) == 1, field
    inst = (LEAN / "Ideas" / "BoardInstance.lean").read_text()
    assert "theorem chessBoard_laws : chessBoard.Laws" in inst and "sorry" not in inst
