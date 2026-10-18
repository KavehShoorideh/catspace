"""ANALYST_PLAN 52: layer-1 questions, goal regression defects, tempo arithmetic, the generic binder, the agenda on 01IFU / 000Sa."""
import chess
import pytest

from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import graph as G
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.state import State, state_of

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
    t = ag.trace()
    assert d.what == "Bxc2" and "Kxc2" in t
    assert "Kxc2 demoted: bound value 2 less their Qc4+ (3, lemma delayed_balance_deep_alive_d1) = net -1, against Bxc2 net 2" in t
    assert "forks Be4" not in t                                       # the fork heuristic is only a fallback label now


def _theirs(fen):
    ag = AG.Agenda(State(fen=fen, us="b", pieces=[]), side_depth=0)
    ag.ask()
    return ag


def test_their_agenda_after_kxc2_claims_the_bishop_and_after_bxc2_nothing():
    b = chess.Board(SA)
    b.push_uci("c1c2")
    got = _theirs(b.fen()).state.claims
    assert any(c.value == 3 and c.lemma == "delayed_balance_deep_alive_d1" and c.move in ("a2c4", "a2a4") for c in got)
    c = chess.Board(SA)
    c.push_uci("e4c2")
    assert _theirs(c.fen()).state.claims == []


def test_the_pass_threat_of_a_quiet_two_move_fork_is_a_save_root():
    fen = "4k3/3p4/8/8/1n6/8/P7/R3K3 w - - 0 1"                      # the Ra1 does not hang; Nc2+ forks Ke1 and Ra1
    ag = AG.Agenda(state_of(fen, "w"))
    ag.ask()
    roots = [q for q in ag.all if "threat" in q.pattern]
    assert len(roots) == 1 and roots[0].goal == "save" and roots[0].stake == 5
    assert roots[0].text.startswith("save our R on a1: their threat Nc2+ wins 5") and ag.their_runs >= 1


def test_no_pass_threat_when_we_are_in_check():
    ag = AG.Agenda(state_of(SA, "w"))
    ag.ask()
    assert not [q for q in ag.all if "threat" in q.pattern]
