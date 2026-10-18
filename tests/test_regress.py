"""ANALYST_PLAN 52: layer-1 questions, goal regression defects, tempo arithmetic, the generic binder, the agenda on 01IFU / 000Sa."""
import chess
import pytest

from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.state import state_of

IFU = "r1b2rk1/pp3pp1/3b3p/3Bn3/P4q2/2P4P/1P2NPP1/RN1Q1RK1 b - - 2 17"
SA = "2Q2bk1/5p1p/p5p1/2p3P1/4B3/7P/qPr2P2/2K4R w - - 0 33"
QH2 = [(chess.Piece(chess.QUEEN, False), chess.H2, chess.F4)]


def kinds(ds):
    return [d.kind for d in ds]


def test_mate_defects_queen_on_h2_is_only_checker_takeable():
    assert kinds(R.mate_defects(chess.Board(IFU), "b", QH2)) == ["checker takeable"]


def test_without_e5_the_queen_on_h2_is_mate():
    assert R.mate_defects(G.without(chess.Board(IFU), chess.E5), "b", QH2) == []


def test_flights_of_the_g1_king_are_none_and_the_xray_is_found():
    cf = R.future_board(chess.Board(IFU), QH2, True)
    assert G.flights(cf, chess.G1) == []
    assert G.xrays_to(cf, chess.H2, False) == [(chess.D6, [chess.E5])]


def test_no_check_is_a_defect():
    ds = R.mate_defects(chess.Board(IFU), "b", [(chess.Piece(chess.QUEEN, False), chess.H4, chess.F4)])
    assert kinds(ds) == ["no check"]


def test_tempo_ok_false_for_two_defects_without_a_forcing_fix():
    d1, d2 = R.Defect("flight", chess.G8), R.Defect("flight", chess.H7)
    q = R.Question(1, "mate", 100, {}, defects=[d1, d2])
    f1 = R.Question(2, "cover", 100, {}, defect_key=d1.key(), unforced=["Ng4"])
    f2 = R.Question(3, "cover", 100, {}, defect_key=d2.key())
    q.fixes = [f1, f2]
    assert not R.tempo_ok(q) and q.reason.startswith("tempo")
    f1.routes = [R.Route(chess.Move.from_uci("e5f3"), "Nf3+", "check")]
    assert not R.tempo_ok(q)                                  # the second defect still has no forcing fix
    f2.routes = f1.routes
    assert R.tempo_ok(q)
    assert R.tempo_ok(R.Question(4, "mate", 100, {}))        # no defect: a one-move pattern


def test_bind_mate_in_two_on_01ifu_is_e5_f3_only():
    got = BD.bind(BD.schema("mate_in_two"), chess.Board(IFU), "b")
    assert [(b.assignment["x"], b.assignment["y"]) for b in got] == [("e5", "f3")]
    assert got[0].replies == ["Kh1", "Bxf3", "gxf3"] and got[0].theorem == "mate_in_two"


def test_verify_stops_at_a_mate_and_binds_nothing_for_a_quiet_move():
    b = chess.Board(IFU)
    assert BD.verify(chess.Move.from_uci("e5f3"), b, "b").value == 100
    assert BD.verify(chess.Move.from_uci("a7a6"), b, "b") is None


def test_agenda_01ifu_reads_the_battery():
    ag = AG.Agenda(state_of(IFU, "b"))
    ag.ask()
    t = ag.trace()
    assert "defect checker takeable" in t and "B on d6 x-rays h2 through our N on e5" in t and "Nf3+ (check)" in t
    d = ag.decision()
    assert d.what == "Nf3+" and d.lemma == "mate_in_two" and d.value == 100


def test_agenda_000sa_takes_with_the_bishop_not_the_king():
    ag = AG.Agenda(state_of(SA, "w"))
    ag.ask()
    d = ag.decision()
    assert d.what == "Bxc2" and "Kxc2" in ag.trace() and "forks Be4" in ag.trace()
