"""plan 72.7 THE RUST BOUNDARY (docs/RUST_BOUNDARY.md). Step 0, the sharper guard: the read set of every claim beside the trace (risk 2) and the cross-checks of
risks 4 and 5 -- python-chess's legal-move order and SAN against the Rust `Pos` over the guard's case set and the corners (the full sample is
`scripts/trace_identity.py crosscheck`)."""
import importlib.util
from pathlib import Path

import chess
import pytest

import analyst_core as AC

SPEC = importlib.util.spec_from_file_location("trace_identity", Path(__file__).resolve().parents[1] / "scripts" / "trace_identity.py")
TI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TI)


def test_72_7_0_the_corners_are_legal_positions():
    """A corner python-chess calls invalid would be dropped by the walk without a word: each must be a legal position."""
    assert all(chess.Board(f).is_valid() for f in TI.CORNERS)


def test_72_7_0_legal_order_and_san_match_python_chess_on_the_guard_case_set():
    """Risks 4 and 5 over the guard's ten positions and the corners, each walked WALK plies: `Pos.legal_ucis()` is python-chess's list, `Pos.san` its names."""
    r = TI.crosscheck(TI.crosscheck_positions(cases_only=True))
    print(f"OBS 72.7 step 0 crosscheck (case set and corners, walk {TI.WALK}): positions {r['positions']}, moves {r['moves']}, SAN mismatches {len(r['san'])}, "
          f"order mismatches {len(r['order'])}")
    assert r["positions"] >= len(TI.CORNERS) and not r["unparsable"]
    assert r["san"] == [] and r["order"] == []


def test_72_7_0_the_corners_name_what_they_are_for():
    """The corners exercise the cases python-chess and shakmaty could disagree on: castling (with check), promotions (under-promotions, with capture), en passant (and
    a pinned one, not legal), disambiguation by file, by rank and by both, mate, an evasion by en passant."""
    names = {f: [AC.Pos(f).san(u) for u in AC.Pos(f).legal_ucis()] for f in TI.CORNERS}
    every = {s for v in names.values() for s in v}
    assert {"O-O", "O-O-O", "O-O+", "axb8=Q+", "axb8=N", "a8=R", "b8=N", "exd6", "cxd6", "N4d3", "Nbd3", "Qa4d4", "Ra8#", "Qxf7#", "Kxe2"} <= every
    assert names["8/8/8/K2pP2r/8/8/8/7k w - d6 0 2"] == ["Kb6", "Ka6", "Kb5", "Kb4", "Ka4", "e6"]          # exd6 e.p. would expose the king on the rank
    assert names["7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"] == []
    assert "exd3" in names["8/8/8/2k5/3Pp3/8/8/4K3 b - d3 0 1"]
    order = AC.Pos("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").legal_ucis()
    assert order[-2:] == ["e1g1", "e1c1"] and order.index("e1d1") < order.index("a1a8")      # castling after every piece move (the a1 rook's after the king's), the king side first
    assert order == [m.uci() for m in chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").legal_moves]


def test_72_7_0_illegal_input_raises_value_error():
    """Risk 7: no panic across the border -- an illegal FEN and an illegal move are a ValueError."""
    with pytest.raises(ValueError):
        AC.Pos("8/8/8/8/8/8/8/8 w - - 0 1")
    p = AC.Pos(chess.STARTING_FEN)
    for bad in ("e2e5", "e7e5", "zz", "", "e1g1"):
        with pytest.raises(ValueError):
            p.san(bad)
        with pytest.raises(ValueError):
            p.push(bad)


def test_72_7_0_the_read_set_dump_is_stable_and_names_every_claim():
    """Risk 2: the snapshot's read sets -- sorted square names, the root questions with their claims' reads, every fact of the store."""
    assert TI._sq(None) == "None" and TI._sq(frozenset()) == "{}"
    assert TI._sq(frozenset({chess.H8, chess.A1, chess.E4})) == "a1 e4 h8"
    trace, reads = TI._run("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "w")
    assert "DECISION Ra8# 100" in trace
    lines = reads.splitlines()
    assert lines[0] == "Q1 mate verified reads None | claimed a1 f7 g7 h7 a8 b8 c8 d8 e8 f8 g8 h8"
    assert any(l.startswith("#2 claim w ") and "reads a1 f7 g7 h7 a8 b8 c8 d8 e8 f8 g8 h8" in l for l in lines)
    assert TI._run("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1", "w")[1] == reads                                        # the same run, the same dump
