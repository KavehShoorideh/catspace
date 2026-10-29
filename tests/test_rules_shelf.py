"""plan 76.2 THE RULE SHELF: the generated spec (rules/chess.gdl -> lean/ideas/Ideas/Spec/Chess.lean) is a certified model of the rule kernel for moves, states,
terminals and goals -- `legal` proved equal to the kernel's `legalMove` (Ideas/Spec/LegalEquiv.lean, with NextEquiv's next state); what is not proved stands as a
CONJECTURE entry of the store with its obstacle."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from catspace.incubator.analyst.discovery import lemma_store as LS

ROOT = Path(__file__).resolve().parents[1]
LEAN = ROOT / "lean" / "ideas"

# the theorems of 76.2 (a): the moves, the end of the game
EQUIVALENCES = ("Ideas.LegalEquiv.legal_iff", "Ideas.LegalEquiv.legal_iff_reachable", "Ideas.LegalEquiv.legal_moveOf", "Ideas.LegalEquiv.attackBy_iff",
                "Ideas.LegalEquiv.check_iff", "Ideas.LegalEquiv.hasMove_iff", "Ideas.LegalEquiv.mate_iff", "Ideas.LegalEquiv.stalemate_iff",
                "Ideas.LegalEquiv.mate_iff_mated", "Ideas.LegalEquiv.stalemate_iff_stalemated", "Ideas.LegalEquiv.drawByFiftyMoves_iff",
                "Ideas.LegalEquiv.goal_win_iff", "Ideas.LegalEquiv.goal_loss_iff", "Ideas.LegalEquiv.terminal_sound", "Ideas.LegalEquiv.terminal_complete")


def _lake() -> str:
    lake = shutil.which("lake") or str(Path.home() / ".elan" / "bin" / "lake")
    if not Path(lake).exists():
        pytest.skip("no lake")
    return lake


def test_the_equivalences_are_indexed_theorems():
    """Every theorem 76.2 (a) states is a proved theorem of docs/LEAN_INDEX.md (the index rebuilt after the file)."""
    idx = LS.lean_index()
    missing = [t for t in EQUIVALENCES if idx.get(t) not in ("theorem", "lemma")]
    assert not missing, missing


def test_the_open_clause_is_a_conjecture():
    """What Lean does not prove of the terminal (the criterion's knight clause as a dead position) is a CONJECTURE entry naming its obstacle and the theorem it
    leaves open."""
    e = LS.load(force=True)["dead position (the knight clause)"]
    assert e["status"] == "CONJECTURE" and e["lean"] == []
    assert "lean/ideas/Ideas/Spec/LegalEquiv.lean:terminal_sound" in e["implementation"]
    assert "obstacle" in e["note"]


@pytest.mark.slow
def test_legal_equiv_builds_by_reasoning():
    """SLOW (a Lean build, ~15 s with the library built): LegalEquiv builds; no sorry, no native_decide, no decide tactic in it; the main theorems rest on
    the standard axioms only (no sorryAx, no Lean.ofReduceBool)."""
    lake = _lake()
    text = (LEAN / "Ideas" / "Spec" / "LegalEquiv.lean").read_text()
    body = text.split("-/", 1)[1]                                              # (the module docstring names the banned tactics)
    for banned in ("sorry", "native_decide", "by decide", "<;> decide", "interval_cases"):
        assert banned not in body, banned
    r = subprocess.run([lake, "build", "Ideas.Spec.LegalEquiv"], cwd=LEAN, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    with tempfile.NamedTemporaryFile("w", suffix=".lean", delete=False) as fh:
        fh.write("import Ideas.Spec.LegalEquiv\n" + "".join(f"#print axioms {t}\n" for t in EQUIVALENCES))
    r = subprocess.run([lake, "env", "lean", fh.name], cwd=LEAN, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    lines = [ln for ln in r.stdout.splitlines() if "depends on axioms" in ln]
    assert len(lines) == len(EQUIVALENCES), r.stdout
    for ln in lines:
        assert "sorryAx" not in ln and "ofReduceBool" not in ln, ln
