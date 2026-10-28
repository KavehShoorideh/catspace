"""plan 76.0 THE SPIKE: the rule spec (rules/chess.gdl) read by the GDL interpreter (discovery/gdl.py, gdl_chess.py) against the fast kernels; the surprises it
filed (rules/surprises/) replayed forever; the guard's `--jobs` identical to its serial run."""
from __future__ import annotations

import filecmp
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import chess
import pytest

import analyst_core as AC
from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import gdl_chess as GC

ROOT = Path(__file__).resolve().parents[1]


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"plan76_{name}", ROOT / "scripts" / f"{name}.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


CP = _script("crossplay")

FIXED = [
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",                      # castling both ways
    "4k3/8/8/r2pP2K/8/8/8/8 w - d6 0 2",                         # e.p. that would expose the king along the rank: illegal
    "4k3/8/8/4pP2/8/8/8/4K3 w - e6 0 2",                         # e.p. legal
    "1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1",                          # promotions with and without capture, all four units
    "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1",      # pos2 (a middlegame of the guard)
    "R5k1/5ppp/8/8/8/8/8/6K1 b - - 100 80",                      # mated with the clock at its limit
]


def test_spec_loads_and_starts_where_the_kernel_starts():
    prog = GC.program()
    assert prog.roles() == ["black", "white"]
    assert prog.init() == GC.state_of(chess.Board())
    r = GC.Reading(prog.init())
    assert r.legal == sorted(m.uci() for m in chess.Board().legal_moves) and not r.terminal and r.goals == {}


@pytest.mark.parametrize("fen", FIXED)
def test_interpreter_gives_the_kernels_moves_and_next_states(fen):
    b = chess.Board(fen)
    st = GC.state_of(b)
    r = GC.Reading(st)
    assert r.legal == sorted(m.uci() for m in b.legal_moves)
    assert r.legal == sorted(AC.Pos(b.fen()).legal_ucis())
    for m in b.legal_moves:
        c = b.copy(stack=False)
        c.push(m)
        assert GC.next_state(st, m.uci(), r.mover) == GC.state_of(c), m.uci()
    assert r.reasons == GC.kernel_reasons(b)


def test_every_filed_surprise_replays():
    """Every case of rules/surprises/: python-chess agrees on every reading; a case read as a KNOWN kernel difference (reported, not patched) still shows exactly
    its own kernel and class and nothing else; every other case agrees with both."""
    cases = CP.load_surprises()
    assert cases, "no surprise filed"
    for case in cases:
        assert case["status"] == "read" and case["wrong"] and case["fide"] and case["change"], case["file"]
        left = CP.replay(case)
        assert not [d for d in left if d[0] == "python"], (case["file"], left)
        if case["expect"] == "known":
            keys = {(k, c, s) for k, c, s, _d in left}
            assert (case["kernel"], case["class"], case["signature"]) in keys and {(k, c) for k, c, _s in keys} == {(case["kernel"], case["class"])}, case["file"]
        else:
            assert not left, (case["file"], left)


def test_the_crossplay_fires_on_a_wrong_spec():
    """The negative control: the spec without the knight promotion disagrees with the kernel where a pawn promotes (the comparison is not vacuous)."""
    text = GC.SPEC.read_text()
    assert text.count("(promotable knight)") == 1
    wrong = GDL.Program(text.replace("(promotable knight)", ""), "mutant")
    res = CP.compare("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", rust=False, prog=wrong)
    assert [d for d in res["disagree"] if d[:2] == ("python", "legal")]
    assert not CP.compare("1n2k3/P7/8/8/8/8/8/4K3 w - - 0 1", rust=False)["disagree"]


def test_snapshot_jobs_is_the_serial_run(tmp_path):
    """plan 76.0 (0): `trace_identity.py snapshot --jobs` writes the serial run's files byte for byte (two of the ten: W0VFO and pos2)."""
    env = dict(os.environ, PYTHONPATH=str(ROOT))
    script = str(ROOT / "scripts" / "trace_identity.py")
    a, b = tmp_path / "serial", tmp_path / "jobs"
    subprocess.run([sys.executable, script, "snapshot", str(a), "--sets", "ten:W0VFO,pos2"], check=True, capture_output=True, env=env)
    subprocess.run([sys.executable, script, "snapshot", str(b), "--sets", "ten:W0VFO,pos2", "--jobs", "2"], check=True, capture_output=True, env=env)
    names = sorted(os.listdir(a))
    assert len(names) == 10 and names == sorted(os.listdir(b))
    match, mismatch, errors = filecmp.cmpfiles(a, b, names, shallow=False)
    assert match == names and not mismatch and not errors
