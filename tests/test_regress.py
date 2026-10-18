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
    assert "Kxc2 demoted: bound value 2 less their Qc4+ then Qxe4 (3, lemma delayed_balance_alive_d2) = net -1, against Bxc2 net 2" in t
    assert "forks Be4" not in t                                       # the fork heuristic is only a fallback label now


def _theirs(fen):
    ag = AG.Agenda(State(fen=fen, us="b", pieces=[]), side_depth=0)
    ag.ask()
    return ag


def test_their_agenda_after_kxc2_claims_the_bishop_and_after_bxc2_nothing():
    b = chess.Board(SA)
    b.push_uci("c1c2")
    got = _theirs(b.fen()).state.claims
    assert any(c.value == 3 and c.lemma == "delayed_balance_alive_d2" and c.what == "Qc4+ then Qxe4" and c.move == "a2c4" for c in got)
    assert not any(c.move == "a2a4" for c in got)
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


def test_qa4_check_does_not_bind_at_depth_two_but_qc4_does():
    b = chess.Board(SA)
    b.push_uci("c1c2")
    good, _ = BD.verify_tried(chess.Move.from_uci("a2c4"), b, "b")
    assert good.schema == "delayed_balance_d2" and good.san == "Qc4+ then Qxe4" and good.value == 3
    bad, tried = BD.verify_tried(chess.Move.from_uci("a2a4"), b, "b")                      # after Kd3 the king guards e4
    assert bad is None and any(n == "delayed_balance_d2" and "no gain" in note for n, _, note in tried)


# ---------------------------------------------------------------------------------------------------------------- 52.6: the promotion goal

RACE = "8/8/4PK2/8/8/2k5/8/8 w - - 0 1"            # white Kf6 Pe6 v black Kc3, white to move
RACE_STOPPED = "8/2k5/4PK2/8/8/8/8/8 w - - 0 1"    # the black king on c7 reaches e8 in 2, the pawn needs 2
RACE_BLOCKED = "8/4n3/4PK2/8/8/8/2k5/8 w - - 0 1"
RACE_GAP = "8/8/4PK2/8/8/8/1B6/7k w - - 0 1"       # our bishop: the bare-king form excludes it
THEIR_PASSER = "8/8/8/8/8/4p3/1K6/6k1 w - - 0 1"   # black e3 pawn, our king 3 from e1: forced
THEIR_PASSER_FAR = "8/8/8/8/8/K3p3/8/6k1 w - - 0 1"


def _ag(fen, side_depth=1):
    ag = AG.Agenda(state_of(fen, "w" if chess.Board(fen).turn else "b"), side_depth=side_depth)
    ag.ask()
    return ag


def test_unstoppable_pawn_fires_the_lemma_and_claims_eight():
    ag = _ag(RACE)
    d = ag.decision()
    assert d.kind == "promote" and d.what == "e7" and d.value == 8 and d.lemma == "unstoppablePasser2_sound"
    assert "no defect -- unstoppable by the tempo arithmetic (their K needs 5 to e8, the pawn 2)" in ag.trace()
    assert R.promote_defects(chess.Board(RACE), chess.E6)[0] == []


def test_stopped_pawn_is_killed_with_the_distance_reason():
    ag = _ag(RACE_STOPPED)
    assert ag.decision() is None
    assert "stopped: their K on c7 reaches e8 in 2, the pawn needs 2" in ag.trace()


def test_blocked_pawn_names_the_blocker():
    ds, facts = R.promote_defects(chess.Board(RACE_BLOCKED), chess.E6)
    assert [d.kind for d in ds][0] == "blocked" and ds[0].sq == chess.E7 and "their N" in ds[0].why
    assert "defect blocked (their N stands on e7)" in _ag(RACE_BLOCKED).trace()


def test_defects_say_unstoppable_but_the_lemma_does_not_fire_is_a_gap_naming_the_hypothesis():
    ag = _ag(RACE_GAP)
    assert ag.decision() is None
    assert any("noSliders: our B on b2" in g for g in ag.state.gaps)


def test_their_passer_is_our_stop_save_root_with_stake_eight_and_forced_note():
    ag = _ag(THEIR_PASSER)
    roots = [q for q in ag.all if "promoter" in q.pattern]
    assert len(roots) == 1 and roots[0].goal == "save" and roots[0].stake == 8
    assert roots[0].text.startswith("stop their P on e3 (promotes on e1)")
    assert "forced: every move of ours is spent stopping the pawn" in ag.trace()
    far = _ag(THEIR_PASSER_FAR)
    assert "no stopper in time" in far.trace() and any("no stopper in time" in g for g in far.state.gaps)


# ---- 52.7: the interposition that only delays; the non-king taker recaptured with check
XEN = "8/1p6/2p5/2P2K1k/8/8/pr6/R7 w - - 0 44"


def _mate_root(fen, uci, us):
    ag = AG.Agenda(state_of(fen, us))
    q = next(q for q in ag.all if q.goal == "mate" and q.pattern.get("move") == uci)
    return ag, q


def test_interposition_is_fixed_by_taking_the_interposer_with_check_and_the_agenda_decides_rh1():
    ag, q = _mate_root(XEN, "a1h1", "w")
    q.defects = R.mate_defects(chess.Board(XEN), "w", R.placement_of(q))
    assert kinds(q.defects) == ["interposition"]
    f = R.fixes_of(q.defects[0], q, chess.Board(XEN), "w")[0]
    assert f.goal == "remove" and [r.san for r in f.routes] == ["Rh1+"] and any("the future is a mate" in w for w in f.why)
    ag.ask()
    d = ag.decision()
    assert d.what == "Rh1+" and d.lemma == "mate_in_two" and "Rh2" in d.line


def test_non_king_taker_recaptured_with_a_mating_check_is_a_forcing_fix_but_the_other_defect_still_kills():
    fen = "6k1/5ppp/4n3/Q7/8/8/5PPP/3R2K1 w - - 0 1"      # Rd8+ Nxd8 Qxd8#, but Nf8 interposes: a second defect with no fix
    ag, q = _mate_root(fen, "d1d8", "w")
    q.defects = R.mate_defects(chess.Board(fen), "w", R.placement_of(q))
    assert kinds(q.defects) == ["checker takeable", "interposition"]
    fixes = [R.fixes_of(d, q, chess.Board(fen), "w")[0] for d in q.defects]
    assert [r.san for r in fixes[0].routes] == ["Rd8+"] and "Qxd8#" in " ".join(fixes[0].why)
    assert fixes[1].routes == [] and any("still" in w for w in fixes[1].why)      # the future after Nf8 Rxf8+ Kxf8 still has defects: dies with the reason
    assert not R.tempo_ok(q) and "interposition f8" in q.reason


# ---- 52.8: deciding by the smallest loss when nothing gains
WM = "2r1k3/pp3p2/1b1Np1p1/8/4P1pq/8/PP2QP2/3R1RK1 b - - 0 20"


def test_052wm_in_check_nothing_gains_so_the_smallest_loss_decides_kf8():
    ag = AG.Agenda(state_of(WM, "b"))
    ag.ask()
    d = ag.decision()
    t = ag.trace()
    assert d.what == "Kf8" and d.kind == "save" and d.lemma == "none" and d.value < 0
    assert "deciding by the smallest loss" in t and "decision Kf8: their best claim after it is" in t


def test_smallest_loss_never_in_their_agenda_and_never_for_a_win_root():
    ag = AG.Agenda(State(fen=WM, us="b", pieces=[]), side_depth=0)
    ag.ask()
    assert ag.decision() is None and "smallest loss" not in ag.trace()


def test_a_gain_claim_still_outranks_000sa_unchanged():
    ag = AG.Agenda(state_of(SA, "w"))
    ag.ask()
    d = ag.decision()
    assert d.what == "Bxc2" and d.lemma != "none" and d.value > 0 and "smallest loss" not in ag.trace()


# ---- 52.10 piece 1: the trapped piece -------------------------------------------------------------------------------------------------------------
TRAP = "4k3/8/2n5/8/8/8/8/3R1BK1 w - - 0 1"        # Bb5 pins the c6 knight (the king cannot guard it: d7 is covered): no safe square, nothing to capture the bishop


def test_a_quiet_trap_binds_the_depth_two_schema_with_the_trap_named_and_not_without_it():
    b = chess.Board(TRAP)
    mv = chess.Move.from_uci("f1b5")
    assert not b.gives_check(mv)
    s2 = BD.schema("delayed_balance_d2")
    got, _ = BD.verify_tried(mv, b, "w", schemas=[s2], trap=chess.C6)
    assert got is not None and got.value == 3 and got.san == "Bb5 then Bxc6 (the knight has no safe square)"
    assert (got.assignment["x2"], got.assignment["y2"]) == ("b5", "c6")
    none, tried = BD.verify_tried(mv, b, "w", schemas=[s2])           # no trap named, not a check: the depth-two schemas are not asked
    assert none is None and "not a check" in tried[0][2]


def test_the_win_question_for_a_piece_that_can_run_gets_the_take_its_squares_fix():
    b = chess.Board(TRAP)
    R.reset_ids()
    q = R.new_question("win", 3, dict(target=chess.C6))
    q.defects = R.material_defects(b, "w", chess.C6)
    assert G.mobility_of(G.with_turn(b, False), chess.C6) > 0
    fx = R.fixes_of(q.defects[0], q, b, "w")
    t = [f for f in fx if f.text.startswith("take its squares")]
    assert len(t) == 1 and "Bb5" in [r.san for r in t[0].routes]
    r = next(r for r in t[0].routes if r.san == "Bb5")
    assert r.trap == chess.C6 and r.why.endswith("also attacked")
