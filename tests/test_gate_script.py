"""plan 72.6: scripts/gate.py -- the file lists and the lock (no engine runs here)."""
import importlib.util
import json
import os
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("gate", Path(__file__).resolve().parents[1] / "scripts" / "gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_72_6_engine_list_is_a_subset_of_full_and_names_the_core_files():
    have, missing = gate.engine_files()
    assert have and set(have) <= set(gate.full_files())
    assert len(have) + len(missing) == len(gate.ENGINE_NAMES)
    assert {"test_game", "test_agenda", "test_plan72"} <= {p.stem for p in have}


def test_72_6_lock_refuses_a_second_gate_and_clears_a_stale_one(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "LOCK", tmp_path / "gate.lock")
    assert gate.acquire("engine") is None                       # first gate takes it
    held = gate.acquire("full")                                  # second is refused, naming the first
    assert held and held["pid"] == os.getpid() and held["cmd"] == "engine" and held["start"]
    gate.release()
    assert not gate.LOCK.exists()
    gate.LOCK.write_text(json.dumps({"pid": 2 ** 22 + 12345, "cmd": "full", "start": "x"}))   # dead pid
    assert gate.read_lock() is None and not gate.LOCK.exists()  # stale lock removed
    assert gate.acquire("engine") is None
    gate.release()


def test_72_6_summary_parser_reads_pytest_lines():
    c = gate.parse_summary("..x\n3 passed, 1 failed, 2 xfailed in 4.20s\n")
    assert c == {"passed": 3, "failed": 1, "xfailed": 2, "errors": 0}
    assert gate.parse_summary("no summary here") is None
