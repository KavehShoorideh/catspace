"""plan 76.5 THE VALUATION AS ONE DEFINITION (docs/ANALYST_PLAN.md 76.5): (a) the Rust core takes the valuation at init -- `analyst_core.set_valuation(worths, unit,
royal)`, set from the lemma store's entry by `lemma_store.install_valuation` (the one place), and THE STORE IS LOADED ON FIRST USE: a word read before the call
loads the store, never a default (and a store that does not set the table raises); a different table changes the Rust words' readings (the negative control as
a test) and the store's puts them back; (b) every Python copy routed: the scan (scripts/plan76_5.py scan) finds no worth literal in the engine outside the two
VALUATION entries but its listed findings; Game.measure, BaseWords.worth, the Rust `see`, `balance` and `principal_lines`, and the routed modules' tables agree
with the store's entry on fixed positions. Count budgets only; no clock."""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import chess
import pytest

import analyst_core as AC
from catspace.incubator.analyst.discovery import game as GM
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import selfmine as SM
from catspace.incubator.analyst.discovery import words as WD

ROOT = Path(__file__).resolve().parents[1]
# fixed positions, white to move: (fen, the square of a capture, what the exchange on it is in worths -- a function of the valuation, never a number)
EXCHANGES = (
    ("4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1", chess.D5, lambda w: w["n"]),                      # exd5: an undefended knight
    ("4k3/8/2p5/3q4/8/4N3/8/4K3 w - - 0 1", chess.D5, lambda w: w["q"] - w["n"]),           # Nxd5 cxd5: the queen for the knight
    ("4k3/8/2p5/3n4/8/8/8/3RK3 w - - 0 1", chess.D5, lambda w: max(0, w["n"] - w["r"])),    # Rxd5 cxd5 loses: no capture (0)
    ("4k3/8/8/3r4/2P5/8/8/3QK3 w - - 0 1", chess.D5, lambda w: w["r"]),                     # cxd5: the pawn takes first, the queen behind it
)
FENS = [f for f, _s, _x in EXCHANGES] + ["r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1",
                                         "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1", "8/3Pk3/4P3/8/pp6/4KP2/P7/8 w - - 0 1"]


def _harness():
    spec = importlib.util.spec_from_file_location("plan76_5", ROOT / "scripts" / "plan76_5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _material(b: chess.Board, side: bool, w: dict) -> int:
    return sum(w[p.symbol().lower()] * (1 if p.color == side else -1) for p in b.piece_map().values())


def _other_table() -> dict:
    """The control's table: the MINED entry's params (a different valuation of the store, read -- not written here)."""
    return LS.load()["material (mined)"]["params"]


# ---------------------------------------------------------------------------------------------------------------- (a) the core takes the valuation at init

def _child(code: str) -> str:
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED="0")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=300)
    return r.stdout + r.stderr


def test_76_5_a_the_store_is_loaded_on_first_use():
    """A fresh process that imports only the core: no table is set; the first worth read (Pos.balance) loads the store's entry -- the one the GOAL entries count in."""
    out = _child("import analyst_core as AC\n"
                 "print('before', AC.valuation_state())\n"
                 "v = AC.Pos('4k3/8/2p5/3q4/8/4N3/8/4K3 w - - 0 1').balance(True) - 64\n"
                 "print('after', AC.valuation_state(), 'read', v)\n"
                 "print('table', sorted(AC.valuation()[0].items()), AC.valuation()[1], AC.valuation()[2])\n")
    print("OBS 76.5a first use:", " | ".join(out.strip().splitlines()))
    v = LS.valuation()
    assert "before (False, 0)" in out
    assert "after (True, 1)" in out
    assert f"read {_material(chess.Board(EXCHANGES[1][0]), chess.WHITE, v.worth)}" in out
    assert f"table {sorted(v.worth.items())} {v.unit} {v.royal}" in out


def test_76_5_a_a_store_that_does_not_set_the_table_raises():
    """Never a silent default: when the store's install does not set the table, the first worth read raises (a Rust panic: PanicException)."""
    out = _child("from catspace.incubator.analyst.discovery import lemma_store as LS\n"
                 "import analyst_core as AC\n"
                 "LS.install_valuation = lambda v=None: None\n"
                 "try:\n"
                 "    AC.Pos('4k3/8/8/3n4/4P3/8/8/4K3 w - - 0 1').balance(True)\n"
                 "    print('READ WITHOUT A TABLE')\n"
                 "except BaseException as e:\n"
                 "    print('raised', type(e).__name__, str(e)[:120])\n"
                 "print('state', AC.valuation_state())\n")
    print("OBS 76.5a no table:", " | ".join(out.strip().splitlines()[-2:]))
    assert "READ WITHOUT A TABLE" not in out and "raised PanicException" in out and "state (False, 0)" in out


def test_76_5_a_set_valuation_refuses_a_table_that_is_not_one():
    w = dict(LS.valuation().worth)
    for bad in ({k: x for k, x in w.items() if k != "q"}, dict(w, x=1), dict(w, q=-1)):
        with pytest.raises(ValueError):
            AC.set_valuation(bad, 1, "k")
    with pytest.raises(ValueError):
        AC.set_valuation(w, 0, "k")
    with pytest.raises(ValueError):
        AC.set_valuation(w, 1, "z")
    assert AC.valuation()[0] == w                                                    # (a refused table leaves the store's in place)


def _readings() -> dict:
    """The Rust words that weigh a unit, on the fixed positions: the balance, the exchange (`see`), the principal lines' credits."""
    out = {}
    for fen in FENS:
        p = AC.Pos(fen)
        out[fen] = (p.balance(True) - 64, [p.see(s) for f, s, _x in EXCHANGES if f == fen], p.principal_lines())
    return out


def test_76_5_a_a_different_table_changes_the_rust_words_and_the_store_puts_them_back():
    """THE NEGATIVE CONTROL AS A TEST: the core set to another table (the mined entry's) reads the balance and the exchange in that table's worths; the store's
    entry installed again reads them back exactly."""
    v = LS.valuation()
    LS.install_valuation()
    before = _readings()
    other = _other_table()
    try:
        AC.set_valuation(other["worth"], other["unit"], other["royal"])
        assert AC.valuation() == (other["worth"], other["unit"], other["royal"])
        changed = _readings()
        for fen, s, x in EXCHANGES:
            assert AC.Pos(fen).see(s) == x(other["worth"]), fen
            assert AC.Pos(fen).balance(True) - 64 == _material(chess.Board(fen), chess.WHITE, other["worth"]), fen
    finally:
        LS.install_valuation()
    after = _readings()
    moved = sum(before[f] != changed[f] for f in FENS)
    print(f"OBS 76.5a control: the core set to {other['worth']} (unit {other['unit']}) moved {moved} of {len(FENS)} positions' readings; "
          f"the store's {v.worth} put back: {sum(before[f] == after[f] for f in FENS)} of {len(FENS)} equal")
    assert moved >= len(FENS) - 1 and after == before
