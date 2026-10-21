"""PLAN 64 (docs/ANALYST_PLAN.md, 'the diagnosis round: every sound switch ends ON'). One position each: the numbers are directional (OBS lines)."""
import chess

import analyst_core as AC
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.analyse import analyse
from catspace.incubator.analyst.discovery.state import state_of

POS2 = "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1"


def _pos2_after(*sans: str) -> str:
    b = chess.Board(POS2)
    for s in sans:
        b.push_san(s)
    return b.fen()


def test_64_0_principal_line_ends_at_our_last_capture():
    """After Nxf4 Rxf4 the exchange on f4 is Bxf4 exf4 Qxf4: two captures of ours, Qxf4 the last."""
    ag = Agenda(state_of(POS2, "w"))
    sq, caps = ag._principal_captures(AC.Pos(_pos2_after("Nxf4", "Rxf4")))
    print("OBS 64.0 principal captures after Nxf4 Rxf4:", [(s, last) for s, _b, last in caps])
    assert chess.square_name(sq) == "f4"
    assert [(s, last) for s, _b, last in caps] == [("Bxf4", False), ("Qxf4", True)]


def test_64_0_their_free_move_after_the_last_capture_is_read():
    """pos2: 1.Nxf4 Rxf4 2.Bxf4 exf4 3.Qxf4 Bg5 pins the queen to Kc1: the close by 'exchange +3' is NOT proved, the reply stays asked."""
    ag = Agenda(state_of(POS2, "w"))
    P2 = AC.Pos(_pos2_after("Nxf4", "Rxf4"))
    t2, cl2 = ag._their_after_capture(P2)
    print(f"OBS 64.0 their claim after our capture(s) on f4: {t2} after {cl2[0] if cl2 else None}")
    assert t2 >= 1 and cl2[0] == "Qxf4"
    assert ag._closed_by_rule(P2, -2) is None


def test_64_0_the_probe_finds_bg5():
    """The probe (data/derived/ledger_puzzles_probe_pos2.json): after 3.Qxf4 our own engine finds Bg5."""
    (p,) = PA.load_set("probe_pos2")
    r = analyse(p, quiet=True, memory=False, seconds=10)
    print("OBS 64.0 probe decision:", r["decision"], r["move"])
    assert str(r["move"]).startswith("h4g5")


def test_64_0_pos2_no_longer_decides_nxf4():
    """pos2 (data/derived/ledger_puzzles_kaveh_pos2.json): the false proof of Nxf4 +1 (the leaf of the binder's delayed_root_max claim is read by their agenda too) is gone."""
    (p,) = PA.load_set("kaveh_pos2")
    r = analyse(p, quiet=True, memory=False, seconds=10)
    print("OBS 64.0 pos2 decision:", r["decision"], r["move"])
    assert r["move"] != "h3f4"
