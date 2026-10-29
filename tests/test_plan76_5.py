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
    r = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), env=env, capture_output=True, text=True, timeout=120)      # (a hang is a failure)
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


def test_76_5_a_a_parallel_section_as_the_first_reader_loads_the_store_first():
    """The first worth read of a fresh process inside a rayon section (vocab_values over several boards): the section loads the table on the GIL's thread before
    its workers read (2026-10-29: without it a worker waited for the GIL its caller held -- the gate's test_threat_learning hung)."""
    fens = [f for f, _s, _x in EXCHANGES]
    out = _child("import analyst_core as AC\n"
                 f"print('read', AC.vocab_values({fens!r}, 'balance'), AC.valuation_state())\n")
    print("OBS 76.5a parallel first use:", out.strip().splitlines()[-1])
    w = LS.valuation().worth
    assert f"read {[64 + _material(chess.Board(f), chess.WHITE, w) for f in fens]} (True, 1)" in out


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
    print("OBS 76.5a no table:", " | ".join(ln for ln in out.strip().splitlines() if ln.startswith(("raised", "state", "READ"))))
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


# ---------------------------------------------------------------------------------------------------------------- (b) every copy routed

def test_76_5_b_the_scan_finds_no_worth_literal_in_the_engine_outside_the_entries_but_its_findings():
    H = _harness()
    got = H.scan(log=lambda *_a: None)
    bare = [h for h in got["ENGINE"] if h[4] is None]
    print(f"OBS 76.5b scan: ENGINE hits {len(got['ENGINE'])} (each a listed FINDING), unexplained {len(bare)}; OUTSIDE {len(got['OUTSIDE'])}, TESTS {len(got['TESTS'])}, "
          f"LEAN {len(got['LEAN'])}")
    assert not bare, bare
    unit = [h for h in got["ENGINE"] if h[2] == "UNIT"]
    print(f"OBS 76.5b scan: exchanges read against 0 (the unit as a literal; listed, not routed) {len(unit)} -- ceiling {H.UNIT_FINDINGS_AT_76_5}")
    assert len(unit) <= H.UNIT_FINDINGS_AT_76_5
    # the scan sees a copy: a table of the hand worths, a pawn literal, a threshold (its own fixture lines, read as code)
    assert H.table_hit(H.pairs_of_line("VAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}")) == "bnpqr"
    assert H.table_hit(H.pairs_of_line("match r { Role::Pawn => 1, Role::Knight | Role::Bishop => 3, Role::Rook => 5, Role::Queen => 9 }")) == "bnpqr"
    assert H.table_hit(H.pairs_of_line("match r { Role::Pawn => 1, Role::Knight => 2, Role::Bishop => 3, Role::Rook => 4 }")) is None     # piece-type codes
    assert [c for c, _w in H.hits_of_line("let v = if m.is_en_passant() { 1 } else { worth(r) };")] == ["PAWN"]
    assert [c for c, _w in H.hits_of_line("if p.role == Role::King || kernel_value(p.role) < 3 { continue; }")] == ["THRESHOLD"]


def test_76_5_b_the_readers_agree_with_the_store_on_fixed_positions():
    """Game.measure, BaseWords.worth, graph.worth, selfmine's tables, the Rust balance / see / principal lines, and every routed module's table read the store's entry."""
    from catspace.incubator.analyst import lean_backrank_puzzles as LB
    from catspace.incubator.analyst import lean_passer_v2_proto as PV
    from catspace.incubator.analyst.discovery import ledger as LG
    from catspace.incubator.analyst.discovery import ledger_puzzles as LP
    from catspace.incubator.analyst.discovery import ruler as RU
    from catspace.incubator.analyst.discovery import surprise_loop as SL
    from catspace.incubator.analyst.discovery import table as TB
    from catspace.incubator.analyst.discovery import watch as WT
    LS.install_valuation()
    v = LS.valuation()
    w = v.worth
    base = WD.CHESS_WORDS.base
    for sym in w:
        for colour in (chess.WHITE, chess.BLACK):
            pc = chess.Piece.from_symbol(sym if colour == chess.BLACK else sym.upper())
            assert base.worth(pc) == G.worth(pc) == v.worth_of(pc) == SM.PIECE_WORTH[sym] == w[sym]
    by = v.by_kind()
    assert RU.VAL == by and TB._VALUE == LG.PVAL == LP.VAL == WT.VAL == v.by_kind(royal=False) and not hasattr(SM, "VALUE")
    assert LB.VALUE == by and PV.VALUE == {k: float(x) for k, x in by.items()} and RU.SEE_VAL[chess.KING] == RU.MATE == LS.goal_kinds().terminal.stake
    assert PV.QUEEN_GAIN == LS.goal_kinds().promotion.stake == max(w[k] for k in "qrbn") - v.pawn and RU.PAWN_WORTH == v.pawn == w["p"]
    for fen in FENS:
        b = chess.Board(fen)
        p = AC.Pos(fen)
        for side in (chess.WHITE, chess.BLACK):
            m = _material(b, side, w)
            assert GM.CHESS.measure(b, side) == m == p.balance(side) - 64 == RU.material(b, side) - RU.material(b, not side) == SM.balance(fen, side)
            assert TB._material(b, side) - TB._material(b, not side) == m and LG._bal(b) * (1 if side else -1) == m
        assert SL.lead_of(b)[0] == _material(b, chess.WHITE, w)
    for fen, s, x in EXCHANGES:
        b = chess.Board(fen)
        assert AC.Pos(fen).see(s) == x(w), fen
        cap = min((m for m in b.legal_moves if m.to_square == s), key=lambda m: w[b.piece_at(m.from_square).symbol().lower()])
        assert max(0, RU.see(b, cap)) == x(w), fen                                    # ruler's own swap list, its worths the store's
    print(f"OBS 76.5b readers: {len(FENS)} positions x 2 sides, {len(EXCHANGES)} exchanges -- Game.measure, BaseWords.worth, the Rust balance and see, ruler, table, "
          f"ledger, ledger_puzzles, watch, surprise_loop, selfmine, lean_backrank_puzzles, lean_passer_v2_proto read {v.name!r} {w}")


def test_76_5_b_a_unit_hangs_when_they_win_the_unit_of_a_gain():
    """plan 76.5 (d)'s fix: reasoned.py's hanging test (and its parries) read the store's unit -- the need of their win goal, which the regression's `hangs` defect reads."""
    from catspace.incubator.analyst.discovery import reasoned as RS
    assert RS.UNIT == LS.valuation().unit
    b = chess.Board("4k3/8/2p5/3q4/8/4N3/8/4K3 b - - 0 1")                               # their queen d5, attacked by our knight, defended by the c6 pawn: it hangs
    assert chess.D5 in RS.hanging_of_ours(b, "b")


# ---------------------------------------------------------------------------------------------------------------- (d) the experiment's switch; the Lean side

def test_76_5_d_the_two_valuation_entries_state_the_lean_obligation_and_cite_its_files():
    """Lean holds the worths as a VALUATION, not a rule: both entries' notes cite the files (pieceValue, valueAt, the vocabulary's worthOf / mateGain, the bound of
    TreeStep.contrib_le_three) and state the OBLIGATION -- the theorems hold for any table with the properties they use; no proof is rewritten."""
    store = LS.load()
    root = ROOT / "lean" / "ideas"
    for name in ("material", "material (mined)"):
        note = store[name]["note"]
        assert "OBLIGATION" in note and "not re-proved" in note, name
        for f in ("Ideas/Fork.lean", "Ideas/ExchangeLaw.lean", "Ideas/Base/Vocabulary.lean", "Ideas/TreeStep.lean"):
            assert f in note and (root / f).exists(), (name, f)
    assert "contrib_le_three does NOT" in store["material (mined)"]["note"]
    assert "theorem contrib_le_three" in (root / "Ideas" / "TreeStep.lean").read_text()
    assert "def mateGain : Int := 100000" in (root / "Ideas" / "Base" / "Vocabulary.lean").read_text()


def test_76_5_d_the_switch_is_a_run_time_choice_and_restores_byte_for_byte(tmp_path, monkeypatch):
    """The harness's switch (no flag in the code): the GOAL entries pointed at the mined entry and back, the files restored byte for byte -- on a copy of the store."""
    import json
    import shutil
    H = _harness()
    store = tmp_path / "lemmas"
    shutil.copytree(LS.DIR, store)
    saved = tmp_path / "saved"
    saved.mkdir()
    monkeypatch.setattr(H, "_store_dir", lambda: store)
    monkeypatch.setattr(H, "_saved_dir", lambda: saved)
    monkeypatch.setattr(H, "ROOT", tmp_path)
    before = {n: (store / f"{n}.json").read_bytes() for n in H.GOAL_FILES}
    H.swap("mined", log=lambda *_a: None)
    assert all(json.loads((store / f"{n}.json").read_text())["params"]["valuation"] == H.MINED for n in H.GOAL_FILES)
    monkeypatch.setattr(H.subprocess, "run", lambda *a, **k: type("R", (), {"returncode": 0})())      # (the git check is the real run's; here the bytes are compared)
    H.swap("hand", log=lambda *_a: None)
    assert {n: (store / f"{n}.json").read_bytes() for n in H.GOAL_FILES} == before
