"""ANALYST_PLAN 52: layer-1 questions, goal regression defects, tempo arithmetic, the generic binder, the agenda on 01IFU / 000Sa."""
import analyst_core as AC
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

@pytest.fixture(autouse=True)
def _rules_not_the_table(monkeypatch):
    """These tests read the RULE machinery on small endings; plan 62.3's tablebase would close those boards first (the comparison condition `--no-table`)."""
    from catspace.incubator.analyst.discovery import table as _TB62
    monkeypatch.setattr(_TB62, "TABLE", False)



@pytest.fixture
def chain_items_off(monkeypatch):
    """plan 64.2: the 52.23 chain items are ON by default. The tests that pin a cost or a cache mechanism of an EARLIER plan (52.16 / 52.17) isolate it from them (the items roughly double the
    inner work: DEEP_WIN_CHASE on 0LRTF 41 -> 169 inner questions, 0 -> 3 fuse hits); the decisions with the items ON are asserted in tests/test_offbuilt64.py and the 90."""
    for k in ("NEED_AWARE_CLOSE", "EVASION_CHASE", "NEED_PROPAGATION", "GOAL_JUDGE", "DEEP_WIN_CHASE"):
        monkeypatch.setattr(AG, k, False)
    monkeypatch.setattr(AG, "CHASE_CLOCK", 99)


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


def test_stopped_pawn_is_killed_with_the_distance_reason(monkeypatch):
    from catspace.incubator.analyst.discovery import race as _RC62
    monkeypatch.setattr(_RC62, "RACE", False)          # plan 62.1: with the race word ON the chase proves this one (e7, then Kf7: won in 5 plies); the 1-ply kill is the comparison condition
    ag = _ag(RACE_STOPPED)
    assert "defect stopper (their K on c7 reaches e8 in 2, the pawn needs 2" in ag.trace()          # the distance reason is the defect the promote question reads
    d = ag.decision()                                  # plan 72.10 (2026-10-23): the race word is no longer needed for the proof -- the chase's inner agendas carry the promote goal and ask
    assert d is not None and d.proved and d.what == "Kf7" and d.value == 8                           # one question each (the pawn's own), so the root's fuse reaches Kf7, then e7, e8=Q: K+P v K, won


def test_blocked_pawn_names_the_blocker():
    ds, facts = R.promote_defects(chess.Board(RACE_BLOCKED), chess.E6)
    assert [d.kind for d in ds][0] == "blocked" and ds[0].sq == chess.E7 and "their N" in ds[0].why
    assert "defect blocked (their N stands on e7)" in _ag(RACE_BLOCKED).trace()


def test_defects_say_unstoppable_but_the_lemma_does_not_fire_is_a_gap_naming_the_hypothesis(monkeypatch):
    from catspace.incubator.analyst.discovery import race as _RC62
    from catspace.incubator.analyst.discovery import plan67 as _PL67
    monkeypatch.setattr(_RC62, "RACE", False)          # plan 62.1: with the race word ON this king-and-pawn board (our bishop) is claimed by Ideas.Race.race_sound
    monkeypatch.setattr(_PL67, "PLAN", False)          # plan 67: and with the plan arithmetic ON it is claimed by Ideas.Plan.plan_sound; this test reads the GAP text of the older lemma alone
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


# ---- 52.10 piece 2: the leaf read by their agenda --------------------------------------------------------------------------------------------------
UQ = "r1bqkbnr/3p1ppp/p3P3/1pp5/2n1P3/2N5/PPP1NPPP/R1BQK2R w KQkq - 0 8"


def test_the_leaf_read_keeps_000sa_bxc2_and_reads_the_word_the_way_the_word_reads_it():
    ag = AG.Agenda(state_of(SA, "w"))
    ag.ask()
    assert ag.decision().what == "Bxc2" and ag.decision().lemma == "delayed_balance_alive_d1"
    assert ag.leaf_stats["read"] >= 1 and ag.leaf_stats["down"] == 0 and ag.leaf_stats["untrusted"] == 0
    assert any(o.startswith("our Bxc2 claimed 2; at the leaf Bxc2 ") and o.endswith("-> kept") for o in ag.leaf_obs)


def test_a_claim_of_theirs_at_the_leaf_at_least_our_gain_there_downgrades_the_claim_to_a_gap():
    from catspace.incubator.analyst.discovery.state import Claim

    class Fake:
        def __init__(self):
            self.state = State(fen="", us="b", pieces=[], claims=[Claim("win", "Qxe4", 99, "delayed_balance_alive_d1", "line")])
            self.their_runs = 0
    ag = AG.Agenda(state_of(SA, "w"))
    real = ag._their_agenda
    leaf_fens = []

    def fake(board):
        leaf_fens.append(board.fen())
        return Fake()
    ag._their_agenda = fake
    ag.ask()
    assert leaf_fens and ag.leaf_stats["down"] >= 1
    assert any("refuted at the leaf: their Qxe4 wins 99 (lemma delayed_balance_alive_d1)" in g for g in ag.state.gaps)
    assert any(o.endswith("-> downgraded") for o in ag.leaf_obs)
    assert not any(c.lemma == "delayed_balance_alive_d1" for c in ag.state.claims)       # nothing is decided by a downgraded claim


# (was xfail strict until 52.13 (4): the shortlist reads their replies in an order that finds exf7+ , the puzzle's answer, and Qd5 is no longer decided)
def test_0uqov_qd5_is_not_decided():
    ag = AG.Agenda(state_of(UQ, "w"))
    ag.ask()
    d = ag.decision()
    assert d is None or d.what != "Qd5"


CT0 = "4rrk1/pp4pp/2p1b3/4q3/4BB2/1PPQ4/P4PP1/R4R1K b - - 0 22"


def test_52_11_a_move_that_blocks_our_own_check_or_does_not_cover_is_not_a_fix():
    b = chess.Board(CT0)
    root = R.new_question("mate", 100, dict(sym="q", at=chess.H5, frm=chess.E5, move="e5h5"))
    flight = [d for d in R.mate_defects(b, "b", R.placement_of(root)) if d.kind == "flight"][0]
    assert flight.sq == chess.G1
    c = R.new_question("cover", 100, dict(root.pattern, target=chess.G1), root, defect_key=flight.key())
    bh3 = chess.Move.from_uci("e6h3")
    assert R._still(c, b, "b", bh3, root)                         # the flight key vanishes (no check), but nothing covers g1: not a fix
    assert not R.routes_of(c, b, "b", root) and "Bh3" not in [s for s in c.unforced]


def test_52_11_escape_note_by_the_rules_equals_the_binder_reading():
    from catspace.incubator.analyst.discovery import selfmine as SM
    for fen, us in ((CT0, "b"), (IFU, "b")):
        b = chess.Board(fen)
        for m in list(b.legal_moves)[:12]:
            c = b.copy(stack=False); c.push(m)
            for rep in list(c.legal_moves)[:12]:
                d = c.copy(stack=False); d.push(rep)
                rules = any(AG.Agenda._mates(d, mv) for mv in d.legal_moves)
                assert rules == bool(SM.bind_mate(d.fen(), us))


# ---- 52.12: trap vs fork, the chase, read sets, the reach ----------------------------------------------------------------------------------------
XY = "r4rk1/pp2bppp/2b1pn2/4N3/3qN3/3B4/PP3PPP/R2QR1K1 w - - 0 14"
RH = "k7/1pp4p/pq3rp1/N7/4Q3/8/PP3PPP/4R1K1 b - - 1 26"
DOFB = "1r2k3/1bp2p2/p3p2Q/q5B1/3b4/P7/5PPP/2RR2K1 b - - 4 29"


def test_52_12_a_check_is_never_a_trap_and_a_fork_binds_on_02xyh():
    b = chess.Board(XY)
    for t in chess.SQUARES:
        pc = b.piece_at(t)
        if pc is not None and not pc.color and pc.piece_type != chess.KING:
            assert not any(b.gives_check(r.move) for r in R.trap_routes(b, "w", t)), "a check was read as a trap"      # mobility with their king in check proves nothing
    forks = R.fork_routes(b, "w", chess.H7)
    assert [r.san for r in forks] == ["Nxf6+"] and forks[0].fork == chess.H7
    got, _ = BD.verify_tried(forks[0].move, b, "w", fork=chess.H7)
    assert got is not None and got.san == "Nxf6+ then Bxh7" and got.note == "a fork: they must answer the check"
    ag = AG.Agenda(state_of(XY, "w"))
    ag.ask()
    assert "traps" not in ag.trace().split("AGENDA")[-1] or all("a check" not in ln for ln in ag.trace().splitlines() if "traps" in ln)


def test_52_12_the_balance_in_python_is_the_vocabulary_word():
    for fen in (XY, RH, DOFB, SA, IFU):
        b = chess.Board(fen)
        for owner in (True, False):
            w = G.word(b, "balance", owner)
            if w is not None:                                # (the word reads nothing when the other side's king would be left in check)
                assert G.balance(b, owner) == w


def test_52_12_the_chase_decides_0rhu6_through_the_forced_reply_and_the_inner_mate():
    ag = AG.Agenda(state_of(RH, "b"))
    ag.ask()
    d = ag.decision()
    t = ag.trace()
    assert d.what == "Qxf2+" and d.value >= 100 and d.lemma.startswith("chase") and "chase: make them move" in t
    assert "after Kh1: Qf1+" in d.line and "Rxf1 Rxf1#" in d.line


def test_52_12_a_question_whose_reads_the_move_and_reply_did_not_touch_is_inherited():
    b0 = chess.Board(XY)
    outer = AG.Agenda(state_of(XY, "w"), side_depth=0, ctx=AG.ChaseCtx())
    outer.ask()
    killed = [q for q in outer.all if q.status == "killed" and q.reads is not None]
    assert killed
    b1 = b0.copy()
    b1.push_uci("a2a3")
    b1.push_uci("a7a6")                                    # the far wing: nothing the killed questions read
    diff = R.diff_squares(b0, b1)
    assert diff and all(chess.square_file(s) < 2 for s in diff)
    inner = AG.Agenda(State(b1.fen(), "w", []), side_depth=0, chase_depth=1, ctx=AG.ChaseCtx(), inherit=outer._inherit_for(diff))
    inner.ask()
    assert inner.inherited > 0 and any(": inherited: nothing changed for the" in ln for ln in inner.lines)
    near = outer._inherit_for(frozenset(chess.SQUARES))      # the whole board changed: nothing is inherited
    again = AG.Agenda(State(b1.fen(), "w", []), side_depth=0, chase_depth=1, ctx=AG.ChaseCtx(), inherit=near)
    again.ask()
    assert again.inherited == 0


def test_52_12_0dofb_has_the_reach_root_via_d5_and_qd5_is_a_threat_of_mate():
    b = chess.Board(DOFB)
    ag = AG.Agenda(state_of(DOFB, "b"))
    roots = [q for q in ag.all if q.pattern.get("reach")]
    assert any(chess.square_name(q.pattern["at"]) == "g2" and q.defects[0].kind == "reach" and "d5" in q.defects[0].why for q in roots)
    assert R.forcing_why(b, chess.Move.from_uci("a5d5"), R.STAKE_MATE, "b", mate_threats=True) == "threat of mate"
    assert R.forcing_why(b, chess.Move.from_uci("a5d5"), R.STAKE_MATE, "b") is None      # the old forcing does not know it
    assert chess.Move.from_uci("a5d5") in R.reach_steps(b, "b", chess.A5, chess.G2)


def test_52_13_a_claim_is_inherited_when_the_diff_misses_its_reads_and_reopened_when_it_meets_them():
    b0 = chess.Board(IFU)
    outer = AG.Agenda(state_of(IFU, "b"), side_depth=0, ctx=AG.ChaseCtx())
    outer.ask()
    won = [q for q in outer.all if q.status == "verified" and q.claimed is not None]
    assert won and outer.decision().what == "Nf3+"
    b1 = b0.copy()
    b1.push_uci("a7a6")
    b1.push_uci("b2b3")                                   # the queen-side pawns: nothing the mate on h2 read
    diff = R.diff_squares(b0, b1)
    assert not (diff & won[0].claimed[2])
    inh = outer._inherit_for(diff)
    assert any(v["kind"] == "claim" for v in inh.values())
    inner = AG.Agenda(State(b1.fen(), "b", []), side_depth=0, chase_depth=1, ctx=AG.ChaseCtx(), inherit=inh)
    inner.ask()
    assert inner.ctx.inherited_claims > 0 and inner.decision().what == "Nf3+"
    assert inner.decision().line.startswith("inherited: nothing changed for the") and inner.asked < outer.asked
    meet = AG.Agenda(State(b1.fen(), "b", []), side_depth=0, chase_depth=1, ctx=AG.ChaseCtx(), inherit=outer._inherit_for(frozenset(chess.SQUARES)))
    meet.ask()
    assert meet.ctx.inherited_claims == 0


PSJ = "8/7k/3Q4/p1r3p1/8/2P1bB1P/PP3rP1/7K b - - 1 48"


def test_52_13_a_quiet_threat_by_a_piece_the_exchange_wins_is_not_a_threat():
    b = chess.Board(PSJ)
    m = chess.Move.from_uci("c5c6")
    c = b.copy()
    c.push(m)
    assert R.threat_value(c, "b") == 9                       # the old reading: the rook on c6 attacks the queen
    assert G.see_for(c, chess.C6, True) > 0                  # ... and the queen takes it (landsSafe fails)
    notes = []
    assert R.forcing_why(b, m, 9, "b", notes=notes) is None
    assert notes and "not a threat: the R on c6 hangs" in notes[0]


AGL8 = "8/1kp2p2/R2p1nr1/1B1Pp1q1/4Pn1p/2P2P2/1PQ3PN/7K b - - 0 33"


def test_52_13_0agl8_is_not_decided_nxg2_at_four_times_the_fuse(monkeypatch):
    orig = AG.Agenda.__init__

    def init(self, *a, **k):                              # the scratch reproduction: 4x the question budget, so that Nxg2 IS followed to its end
        orig(self, *a, **k)
        self.budget *= 4
        if self.ctx is not None and k.get("ctx") is None and self.chase_depth == 0:
            self.ctx.left = self.budget * AG.FUSE
    monkeypatch.setattr(AG.Agenda, "__init__", init)
    ag = AG.Agenda(state_of(AGL8, "b"))
    res = ag._chase_route(chess.Move.from_uci("f4g2"))        # Nxg2 itself, followed to its end
    assert res["killed"] and "stands only as a balance, their " in res["killed"] and "+ wins 100" in res["killed"] and res["killed"].endswith("no parry") and res["value"] == 0
    ag.ask()
    assert all(c.what != "Nxg2" for c in ag.state.claims) and ag.decision().what != "Nxg2"
    assert not hasattr(AG.Agenda, "_static_refuted")


def test_52_13_a_reply_that_leaves_us_a_mate_in_one_or_an_unanswered_exchange_is_closed_by_rule():
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    ag = AG.Agenda(state_of(fen, "w"))
    assert ag._closed_by_rule(AC.Pos(fen), 0) == (BD.MATE_VALUE, "mate in one")
    ag = AG.Agenda(state_of(AGL8, "b"))
    res = ag._chase_route(chess.Move.from_uci("g5g2"))        # Qxg2+ Qxg2: the knight takes the queen, nothing of theirs
    assert res["value"] == 1 and any("closed by rule: exchange +9 (" in p and "), nothing of theirs" in p for p in res["parts"])
    assert ag.ctx.closed >= 1


def test_52_13_pos_replies_diff_and_exchange_equal_python_chess_on_dev_20():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    n = 0
    for p in PA.load_set("dev_20"):
        b = chess.Board(p["fen"])
        P = AC.Pos(b.fen())
        assert P.balance(b.turn) == G.balance(b, b.turn) and P.fen() == b.fen()
        m = next(iter(b.legal_moves))                         # the first legal move; every reply of it
        c = b.copy()
        c.push(m)
        P1 = P.push(m.uci())
        assert P1.fen() == c.fen()
        rows = {r[0]: r for r in P1.replies()}
        assert set(rows) == {r.uci() for r in c.legal_moves}
        for r in c.legal_moves:
            d = c.copy()
            d.push(r)
            _u, dbal, chk, cap, rec, mate1 = rows[r.uci()]
            mates = any(d.gives_check(x) and AG.Agenda._mates(d, x) for x in list(d.legal_moves))
            assert (dbal, chk, cap, rec, mate1) == (G.balance(d, b.turn) - G.balance(c, b.turn), d.is_check(), c.is_capture(r), c.is_capture(r) and r.to_square == m.to_square, mates)
            P2 = P1.push(r.uci())
            assert set(P.diff(P2)) == set(R.diff_squares(b, d)) and P2.mate_in_one() == mates and P2.legal_count() == d.legal_moves.count()
            ex = max([G.see(d, s) for s in chess.SQUARES if d.piece_at(s) and d.piece_at(s).color != d.turn and d.piece_at(s).piece_type != chess.KING and d.is_attacked_by(d.turn, s)] or [0])
            assert P2.exchange() == ex
            n += 1
    assert n > 300


# ---- 52.14: their threat is a save root of the inner agenda on a static leaf ----------------------------------------------------------------------

VSUT_LEAF = "1r5k/5ppp/3R4/1r6/8/P7/6PP/1R4NK w - - 1 31"          # 0VSUT after Qxe5 Rxe5 Rxb1 Reb5: we are +3 (balance, us to move) and their Rxb5xb1 threatens our rook


def test_52_14_0vsut_qxe5_is_decided_with_the_inner_static_leaf_read_and_the_parry_printed():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0VSUT"][0]
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    assert ag.decision().what == "Qxe5"
    t = ag.trace()
    assert "static leaves read from their side" in t
    assert any("Rxb1 after Reb5" in x and "their Rxb1 wins 4 if we pass; parry: Rxb5" in x and "keeps +3" in x for x in ag.ctx.static_lines)
    assert ag.ctx.static_read > 0 and ag.ctx.static_parried > 0


def test_52_14_a_threat_is_parried_and_the_gain_kept_on_a_constructed_static_leaf(monkeypatch):
    b = chess.Board(VSUT_LEAF)
    outer = AG.Agenda(State(b.fen(), "w", []), side_depth=1)
    t, cl = outer._their_pass(b.fen())
    assert t == 4 and cl.what == "Rxb1"
    their = outer.ctx.their_ags[AG._pos_key(G.with_turn(b, False).fen())]
    inner = AG.Agenda(State(b.fen(), "w", []), side_depth=0, chase_depth=1, ctx=outer.ctx, min_stake=BD.MATE_VALUE, their=their, keep=3)
    inner.ask()
    (q, cl), = inner.threat_qs
    assert q.status == "verified" and q.claimed[1].what == "Rxb5" and q.claimed[1].value == 3       # the rook takes the attacker, they retake: the +3 stands
    assert inner.best is None                                                                       # the parry keeps the gain, it is not a gain of its own
    gone = AG.Agenda(State(b.fen(), "w", []), side_depth=0, chase_depth=1, ctx=outer.ctx, min_stake=BD.MATE_VALUE, their=their, keep=0)
    gone.ask()
    q0 = gone.threat_qs[0][0]                                                                        # plan 70.1 (2026-10-22): the one-ply keep family fails at keep 0 (net 0 < 1) and the save is CHASED:
    assert q0.status == "verified" and gone.best is not None and gone.best[2].what == "Rxb5" and gone.best[2].kind == "save"     # Rxb5 is a threat of mate, read to its end, a claim
    monkeypatch.setattr(AG, "CHASE_SAVES", False)                                                    # the switch off: the keep family alone, as before 70.1
    gone0 = AG.Agenda(State(b.fen(), "w", []), side_depth=0, chase_depth=1, ctx=outer.ctx, min_stake=BD.MATE_VALUE, their=their, keep=0)
    gone0.ask()
    assert gone0.threat_qs[0][0].status == "killed" and "no parry keeps the gain" in gone0.threat_qs[0][0].reason    # a gain of 0 is not kept by any one-ply parry (net 0 < 1)


# ---- 52.15 (1): in check, the checker is a win target -----------------------------------------------------------------------------------------------

def test_52_15_in_check_the_checker_is_a_win_root_beside_the_save_root():
    b = chess.Board("8/4k3/p2Pppp1/Kp5p/5P2/P5P1/1P5P/8 b - - 0 39")          # 0Zsoa after Rxd6 exd6+: the P on d6 gives check
    ag = _ag(b.fen(), side_depth=0)
    goals = [(q.goal, q.text) for q in ag.questions]
    assert ("win", "win their P on d6 (it gives check)") in goals and any(g == "save" and "in check" in t for g, t in goals)
    wq = [q for q in ag.all if q.pattern.get("checker")][0]
    assert wq.stake == 1 and [r.san for r in ag._direct_routes(wq)] == ["Kxd6"]       # the routes are the legal captures of the checker, the king's included
    assert "Q2 win their P on d6 (it gives check) (stake 1): the checker is taken by a legal capture" in ag.trace()


def test_52_15_0sjz3_rxd6_is_decided_through_the_checker_after_rxd6_rxd6():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0SJz3"][0]
    ag = AG.Agenda(State(p["fen"], "b", []))
    ag.ask()
    assert ag.decision().what == "Rxd6"
    inner = AG.Agenda(State("8/p2k4/1p1R4/6p1/1PP5/8/P7/3K4 b - - 0 36", "b", []), side_depth=0, chase_depth=1, ctx=ag.ctx, min_stake=1)
    inner.ask()
    assert inner.best is not None and inner.best[2].what == "Kxd6" and inner.best[2].value == 5


# ---- 52.15 (2): promote roots chase ----------------------------------------------------------------------------------------------------------------

def test_52_15_a_stopped_promotion_is_chased_through_the_pawn_offered_to_the_stopper_0nv29():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0Nv29"][0]
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    t = ag.trace()
    assert "b6 (threat: b7 then b8=Q binds after a pass" in t or "b6 (offers the pawn to the stopper: the B on a5 takes" in t      # 52.18 (3): b6 is now forcing by the claim a pass lets bind; before it was the stopper rule's route
    assert ag.ctx.agendas > 0                                                                  # every reply of theirs was followed by the whole agenda
    # 52.16 (1): after Bb4 the bishop is a stopper that does not stop (the other colour from Kd3, two moves from b8): the passer lemma with bishops claims b7-b8
    assert "after Bb4: b7 (b7, then b8=Q within 3 plies whatever they do (2 pushes); value 8, lemma unstoppablePasser2_sound)" in t
    assert ag.decision().what == "b6" and "unstoppablePasser2_sound" in ag.decision().lemma
    assert "enemyKingPawns" not in t
    inner = AG.Agenda(State("8/8/1P6/2P1k3/1b2P3/3K4/8/8 w - - 1 56", "w", []), side_depth=0, chase_depth=1, ctx=ag.ctx, min_stake=R.STAKE_PROMOTE)
    inner.ask()
    assert inner.best is not None and inner.best[2].kind == "promote" and inner.best[2].lemma == "unstoppablePasser2_sound"


def test_52_15_a_promote_route_is_claimed_when_the_pawn_cannot_be_stopped_on_every_reply():
    ag = AG.Agenda(State("7r/P4N2/8/8/8/4k3/8/2K5 w - - 0 1", "w", []))          # the R on h8 stops a7-a8; Nxh8 takes the stopper
    ag.ctx.left = 100000
    res = ag._chase_route(chess.Move.from_uci("f7h8"), promote=True)
    assert res["killed"] is None and res["gap"] is None and res["value"] == R.STAKE_PROMOTE
    assert len(res["parts"]) == 7 and all("-> 8" in x for x in res["parts"]) and ("unstoppablePasser2_sound" in res["lemmas"] or "plan_sound" in res["lemmas"])       # every reply of the lone king is answered by a promote claim (plan 67's arithmetic may close it first)


def test_52_15_the_blocker_of_theirs_is_a_stopper_of_the_promote_question_0gjec():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0GJec"][0]
    ag = AG.Agenda(State(p["fen"], "b", []))
    ag.ask()
    assert "Rb1 (threat on the stopper: the R on c1)" in ag.trace()                           # the chase asks the promote question through the attack on the blocker


# ---- 52.15 (3): the parry as a bound schema (the keep family) -----------------------------------------------------------------------------------------

def test_52_15_the_keep_family_reads_the_delayed_balance_theorem_with_v_zero_and_agrees_with_a_brute_force_tail():
    from catspace.incubator.analyst import lean_index
    from catspace.incubator.analyst.discovery import selfmine as SM
    d = [x for x in lean_index.build() if x["full"] == "Ideas.Mined.Schemas." + BD.KEEP.theorem][0]
    assert BD.KEEP.theorem == "delayed_balance_alive_d1" and not d["open"]
    assert "(hTB : (T : Int) ≥ (B : Int) + v)" in d["statement"] and "(hCT : C + 4 = T)" in d["statement"] and "T ≥ 68" not in d["statement"]      # general in the bound B and the gain v, no floor on the balance

    def material(c, own):
        return sum(G.worth(c.piece_at(s)) for s in chess.SQUARES if c.piece_at(s) and c.piece_at(s).color == own and c.piece_at(s).piece_type != chess.KING)

    def tail(b, m, own, C):                                    # least (most (least (min balance (material + C)))) after the named move, by the rules
        c = b.copy()
        c.push(m)
        out = []
        for r in c.legal_moves:
            d1 = c.copy()
            d1.push(r)
            best = None
            for m2 in d1.legal_moves:
                e = d1.copy()
                e.push(m2)
                worst = None
                for r2 in e.legal_moves:
                    f = e.copy()
                    f.push(r2)
                    v = min(G.balance(f, own), material(f, own) + C)
                    worst = v if worst is None else min(worst, v)
                assert worst is not None
                best = worst if best is None else max(best, worst)
            assert best is not None
            out.append(best)
        return min(out)
    b = chess.Board("4k3/8/8/3b4/8/2N5/P7/4K3 w - - 0 1")
    B = SM.val_at(b.fen(), "balance", "w")
    kept = [b.san(m) for m in b.legal_moves if BD.verify_keep(m, b, "w")[0] is not None]
    brute = [b.san(m) for m in b.legal_moves if not m.promotion and tail(b, m, True, B - 4) >= B]
    assert kept == brute and 0 < len(kept) < len(list(b.legal_moves))                      # some moves hold the balance, some do not, and the schema says the same as the rules
    bind, _ = BD.verify_keep(next(m for m in b.legal_moves if b.san(m) == kept[0]), b, "w")
    assert bind.value == 0 and bind.theorem == "delayed_balance_alive_d1" and bind.schema == "keep_d1"


def test_52_15_the_parry_claim_carries_the_lean_theorem_and_no_their_view_label_remains():
    b = chess.Board(VSUT_LEAF)
    outer = AG.Agenda(State(b.fen(), "w", []), side_depth=1)
    outer._their_pass(b.fen())
    their = outer.ctx.their_ags[AG._pos_key(G.with_turn(b, False).fen())]
    inner = AG.Agenda(State(b.fen(), "w", []), side_depth=0, chase_depth=1, ctx=outer.ctx, min_stake=BD.MATE_VALUE, their=their, keep=3)
    inner.ask()
    (q, cl), = inner.threat_qs
    assert q.claimed[1].what == "Rxb5" and q.claimed[1].lemma == "delayed_balance_alive_d1" and "their_view" not in inner.trace() + q.claimed[1].lemma
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0VSUT"][0]
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    assert any("parry: Rxb5 (lemma delayed_balance_alive_d1) keeps +3" in x for x in ag.ctx.static_lines) and "their_view" not in ag.trace()


# ---- 52.16 (2): the fuse order ----------------------------------------------------------------------------------------------------------------------

def test_52_16_the_fuse_is_charged_by_stake_order_and_a_later_root_is_not_starved_0lrtf_0gjec():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0LRTF"][0]
    ag = AG.Agenda(State(p["fen"], "w", []), seconds=30)                                        # 52.23: the budget is time (the count fuse of 240 cut Rh8+ Kd7 once the inner agendas chase their checking fixes)
    ag.ask()
    t = ag.trace()
    assert "fuse left -" not in t and ag.ctx.left >= 0                                         # before 52.16: -14 (239 questions on a route share of 120)
    assert "killed by reasoning (the measure they hold)" in t and not ag.state.gaps            # 52.17 (1): the mate roots no longer chase (their gaps, the fuse hits under the king's walk, are gone)
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0GJec"][0]
    ag = AG.Agenda(State(p["fen"], "b", []))
    ag.ask()
    t = ag.trace()
    assert "Rxd4: killed -- after Rxd4" in t and "fuse left -" not in t                       # Q4's own routes were all GAPs with nothing left (fuse left -678); now one is read to its kill
    assert any("Rb1 not followed past Rxb1" in g for g in ag.state.gaps)                       # the remaining gap: the queen-check chain after Rb1 Rxb1


def test_52_16_a_route_is_charged_at_most_its_share_and_the_later_route_keeps_its_own():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "08pSj"][0]
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    assert ag.decision().what == "Rf1+"                                                        # the frames open at the fuse still finish: 08pSj stays decided


def test_52_16_a_position_asked_again_under_other_move_counters_is_a_cycle():
    a = "2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 10 64"
    ag = AG.Agenda(State(a, "w", []))
    ag.ctx.left = 1000
    ag.ctx.active.add(AG._pos_key(a))
    r = ag._inner("2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 14 66", 1)                                 # the same board two moves later
    assert r["claim"] is None and r["asked"] == 0 and r["value"] == 0


# ---- 52.16 (3): a sacrifice that still threatens is forcing -------------------------------------------------------------------------------------------

DISCOVERY = "r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1"        # Kaveh's discovery position (SF d20: Nf6 +385, Nd6 +320)


def test_52_16_the_knight_sacrifice_that_still_threatens_is_forcing_and_the_hanging_rook_is_not():
    b = chess.Board(DISCOVERY)
    notes = []
    assert R.forcing_why(b, chess.Move.from_uci("e4f6"), 5, "w", notes=notes) == "threat 5 survives Bxf6 (the N hangs, 3)"          # the battery through the vacated e4 stands after the capture
    assert R.forcing_why(b, chess.Move.from_uci("e4d6"), 5, "w") == "threat 5 survives Bxd6 (the N hangs, 3)"
    # 08pSj: Rc6 hangs to the queen and nothing survives its capture
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "08pSj"][0]
    b2 = chess.Board(p["fen"])
    notes = []
    rc6 = next(m for m in b2.legal_moves if b2.san(m) == "Rc6")
    assert R.forcing_why(b2, rc6, 5, "w", notes=notes) is None
    assert any("Rc6: not a threat: the R on c6 hangs" in n and "does not survive Qxc6" in n for n in notes)


def test_52_16_the_discovery_position_decides_nf6_with_the_exchange_read(monkeypatch):
    monkeypatch.setattr(AG, "POOL", False)                                   # the TIERS' reading of this position (the trace words below are theirs); the pool's is LOOP TARGET 1 (test_offbuilt64)
    ag = AG.Agenda(State(fen=DISCOVERY, us="w", pieces=[]), seconds=30)      # 2026-10-20: a reply closes by the exchange only when their agenda AFTER our capture finds nothing (kcQil);
    #                                                                          2026-10-21 (64.0 sound close: Qxa8 Bxa8 Kxf7 counts): 30 s, it decides in ~9 s; at 15 s the judge cuts it
    #                                                                          (test_64_discovery_decides_within_15_s, strict xfail until the tiers' time allocation is fixed)
    #                                                                          Nf6's replies are now read rather than closed, and the count budget no longer reaches the decision
    ag.ask()
    d = ag.decision()
    assert d is not None and d.what in ("Nf6", "Nd6")                                           # either is right by SF d20 (Nf6 +385, Nd6 +320); the tie of equal claims is broken by the move's uci
    assert "Nf6 (threat 5 survives Bxf6 (the N hangs, 3))" in ag.trace()                         # the sacrifice that still threatens is a route
    assert "after Bxf6: Qxa8" in ag.trace()                                                     # ... and Qxa8 is read after their replies (64.0: no longer closed by 'exchange +5': Qxa8 Bxa8 then
    #                                                                                            gxf6 / Kxf7 wins the N back, a claim of theirs on a8 worth more than the exchange's credit)
    assert R.exchange_read(chess.Board("r3q2k/pQ1nb1pp/5Bn1/5bP1/3P4/2N1P3/PPP3B1/R1B3K1 w - - 0 2")) .startswith("Qxa8 wins 5: Qxa8 Qxa8 Bxa8")


# ---- 52.16 (4): failed promote chases ----------------------------------------------------------------------------------------------------------------

def test_52_16_a_reply_that_leaves_the_stopper_standing_ends_the_promote_route_02xvk(chain_items_off, monkeypatch):
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    monkeypatch.setattr(AG, "CHASE_SAVES", False)       # plan 72.10: the route memo is per goal, so the save question's KEEP chase (70.1) reads Bxg6 itself instead of being served the mate goal's kill
    #                                                     (made at target 1) and decides first (Bxg6, the puzzle's answer); the 52.16 (4) mechanism this test pins is read with the 70.1 chase off
    p = [x for x in PA.load_set("dev_20") if x["id"] == "02XVK"][0]
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    t = ag.trace()
    assert ag.ctx.promote_stands > 0 and "defect stands" in t                                  # a route ended at a reply that left the stopper standing, with no new fix
    assert "Qh7+: killed -- after Kxh7: the stopper defect stands" in t or "defect stands" in t
    assert ag.ctx.asked < 1000                                                                 # before: 3742 inner questions (8.2 s); the promote question was re-chased under every reply
    assert ag.best is None and (ag.decision() is None or not ag.decision().proved)             # unchanged: 02XVK has no proved claim (Bxg6 stays a GAP at the root's mate chase; plan 65.3 may offer an unproved unfinished line)


def test_52_16_a_stand_check_does_not_end_the_pawns_own_push_0nv29():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0Nv29"][0]
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    assert ag.decision().what == "b6" and ag.ctx.promote_stands == 0                           # b6 is the pawn's own push: the question changes, the stand check does not apply


# ---- 52.17 (1): the region, the wall, the measure they hold ---------------------------------------------------------------------------------------

LRTF = "2k5/2P4R/1K6/8/8/p7/8/2r5 w - - 6 62"
WALL = "8/6k1/4p3/3r4/8/8/8/K6R w - - 0 1"             # their rook on d5 (guarded by the e6 pawn) holds the whole d-file: our king on a1 is cut off from the king's side


_LEAN_INDEX: list = []


def _lean(name, module="Region"):
    if not _LEAN_INDEX:                                                  # the index build reads every Lean file (about 2 minutes): once per run
        from catspace.incubator.analyst import lean_index
        _LEAN_INDEX.extend(lean_index.build())
    d = [x for x in _LEAN_INDEX if x["full"] == f"Ideas.{module}." + name]
    assert d and not d[0]["open"], name
    return d[0]


def test_52_17_the_region_lemmas_are_proved_in_lean_with_the_statements_the_python_mirrors():
    assert "(hwall : ∀ y, ¬ stand c y) (ha : a < c)" in _lean("king_region_wall_file")["statement"]
    assert "rookOnFile" in _lean("rook_wall_file")["statement"] and "(hself : ¬ stand c r)" in _lean("rook_wall_file")["statement"]
    assert "kingHits wa wb (ka + i) (kb + j) ∨ rookLine ra rb (ka + i) (kb + j)" in _lean("no_rook_mate_in_open")["statement"]
    assert "¬ chebLe 1 wa wb ka kb" in _lean("evasion_step")["statement"]
    _lean("rookLine_queenLine")


def test_52_17_no_rook_mate_in_the_open_agrees_with_the_rules_on_every_placement():
    # the Lean statement `no_rook_mate_in_open`, instance by instance: kings apart, the lone king's box on the board -> the king and one rook never cover its whole box
    from catspace.incubator.analyst.discovery import region as RG
    for k in chess.SQUARES:
        if not RG.open_box(k):
            continue
        box = [s for s in chess.SquareSet(chess.BB_KING_ATTACKS[k])] + [k]
        for w in chess.SQUARES:
            if RG.cheb(w, k) < 2:
                continue
            kcov = set(chess.SquareSet(chess.BB_KING_ATTACKS[w]))
            for r in chess.SQUARES:
                if r in (w, k):
                    continue
                rcov = {s for s in box if s != r and (chess.square_file(s) == chess.square_file(r) or chess.square_rank(s) == chess.square_rank(r))}
                full = all(s in kcov or s in rcov for s in box)
                assert not full, (chess.square_name(k), chess.square_name(w), chess.square_name(r))


def test_52_17_the_evasion_step_holds_the_distance_on_every_king_move():
    from catspace.incubator.analyst.discovery import region as RG
    steps = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for w in chess.SQUARES:
        for k in chess.SQUARES:
            if RG.cheb(w, k) < 2:
                continue
            for w2 in chess.SquareSet(chess.BB_KING_ATTACKS[w]):
                d0 = RG.cheb(w, k)
                ok = any(max(abs(chess.square_file(w2) - chess.square_file(k) - dx), abs(chess.square_rank(w2) - chess.square_rank(k) - dy)) >= d0 for dx, dy in steps)
                assert ok, (chess.square_name(w), chess.square_name(k), chess.square_name(w2))


def test_52_17_a_rook_that_holds_a_file_end_to_end_is_a_wall_for_a_king_region():
    from catspace.incubator.analyst.discovery import region as RG
    b = chess.Board(WALL)
    reg = RG.region(b, chess.A1)
    assert max(chess.square_file(s) for s in reg.squares()) < 3 and chess.D5 in reg.reads and chess.E6 in reg.reads        # a-c files: the wall is the rook and the pawn that guards it
    assert reg.dist[chess.A1] == 0 and reg.dist[chess.B2] == 1 and reg.dist[chess.C4] == 3
    for fen in ("8/6k1/8/3r4/8/8/8/K6R w - - 0 1", "8/6k1/4p3/8/8/3r4/8/K6R w - - 0 1"):       # unguarded rook: the king takes it and crosses (d5 / d3 on the way)
        assert max(chess.square_file(s) for s in RG.region(chess.Board(fen), chess.A1).squares()) >= 3


def test_52_17_the_region_of_0lrtf_connects_through_c8_so_the_region_kill_does_not_fire():
    from catspace.incubator.analyst.discovery import region as RG
    b = chess.Board(LRTF)
    reg = RG.region(b, chess.B6)
    assert chess.C8 in reg.squares() and chess.D7 in reg.squares() and chess.C6 not in reg.squares()        # the c7 pawn shields c8 from the rook; c2..c6 are the wall
    assert RG.region_kill(b, True, chess.H7, [chess.D7]) is None
    assert (chess.B6, chess.C7) in RG.bound_pieces(b, True) and (chess.H7, chess.C7) in RG.bound_pieces(b, True)


def test_52_17_0lrtf_mate_roots_die_by_the_measure_they_hold_before_any_chase():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0LRTF"][0]
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    t = ag.trace()
    q1, q2 = t.index("Q1 mate on h8"), t.index("Q2 mate on d8")
    q3 = t.index("Q3 promote")
    assert "killed by reasoning (the measure they hold)" in t[q1:q2] and "killed by reasoning (the measure they hold)" in t[q2:q3]
    assert "chase:" not in t[q1:q3] and "Rh8+ Kd7 (the box of d7 is on the board, distance 2 held)" in t[q1:q2]       # no inner agenda under either
    assert "Ideas.Region.no_rook_mate_in_open" in t[q1:q2] and "our K on b6 is bound to the P on c7" in t[q1:q2] and "promotes in 2" in t[q1:q2]
    assert ag.ctx.reasoned >= 2


def test_52_17_the_wall_kills_the_mate_roots_when_our_king_is_cut_off():
    ag = AG.Agenda(State(WALL, "w", []))
    ag.ask()
    t = ag.trace()
    assert t.count("killed by reasoning (the wall)") == 2 and "the wall is their r on d5, their p on e6" in t
    assert "chase:" not in t[:t.index("Q3 win")]


# ---- 52.17 (2): a chase route must address a defect of its own goal ---------------------------------------------------------------------------------

def _inner_agenda(fen, depth=2):
    return AG.Agenda(State(fen, "w", []), side_depth=0, chase_depth=depth, ctx=AG.ChaseCtx())


def _win_q(ag, sq):
    q = [x for x in ag.all if x.goal == "win" and x.pattern["target"] == sq][0]
    q.defects = R.material_defects(ag.board, "w", sq)
    return q


def test_52_17_a_check_that_fixes_no_defect_of_the_goal_is_not_a_route_of_it_and_the_fork_the_discovery_and_the_root_are():
    a = _inner_agenda("6k1/p6p/8/8/8/8/8/4R1K1 w - - 0 1")
    ok, note = a._addresses(_win_q(a, chess.A7), chess.Move.from_uci("e1e8"), "check")
    assert not ok and note == "Re8+: not a route of win a7 (a check that fixes no defect: a7 stays unattacked)"            # the walk of checks
    b = _inner_agenda("2b3k1/p6p/8/8/8/8/8/4R1K1 w - - 0 1")
    assert b._addresses(_win_q(b, chess.C8), chess.Move.from_uci("e1e8"), "check")[0]                                    # the check also attacks the target: a fork
    c = _inner_agenda("6k1/p6p/8/8/6B1/8/8/4K1Q1 w - - 0 1")
    assert c._addresses(_win_q(c, chess.A7), chess.Move.from_uci("g4h3"), "check")[0]                                    # a DISCOVERED check: the queen's line opens (a new threat from the vacated line)
    assert a._addresses(_win_q(a, chess.A7), chess.Move.from_uci("e1e8"), "threat of mate")[0]                            # forcing by a claim of its own
    root = _inner_agenda("6k1/p6p/8/8/8/8/8/4R1K1 w - - 0 1", depth=0)
    assert root._addresses(_win_q(root, chess.A7), chess.Move.from_uci("e1e8"), "check")[0]                               # the root's own chase keeps every forcing move
    a.ctx.routes[(AG._pos_key(a.board.fen()), "e1e8")] = dict(gap="chase fuse")
    assert a._addresses(_win_q(a, chess.A7), chess.Move.from_uci("e1e8"), "check")[0]                                    # a GAP is never a kill: a route the fuse cut stays open for later goals


Q04_KEEPS = ("Be8+", "Bh5+", "Bf5+", "Be4+", "Bd3+", "Bc2+", "Bb1+")      # 04Qx2: the bishop's discovered checks off g6 (the queen on g4 checks along the g-file), each keeping the bishop


def test_52_17_the_dev_20_decisions_that_needed_a_discovery_or_a_resumed_gap_are_kept_04qx2_0bhbb():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    for pid, san in (("04Qx2", "Be8+"), ("0BHbB", "Qb8+")):
        p = [x for x in PA.load_set("dev_20") if x["id"] == pid][0]
        ag = AG.Agenda(State(p["fen"], "w", []))
        ag.ask()
        if pid == "04Qx2":                             # plan 72.10: the save question's keep chase (the last root, a pool of 7) now reads every discovered check of the bishop to its end, not Be8+ alone; all keep
            d = ag.decision()                          # the bishop (value 3, 70.1's currency) and the equal claims are broken by the move's uci (53 (3)'s last resort), so the decision is one of them
            assert d is not None and d.proved and d.value == 3 and d.what in Q04_KEEPS and ag.ctx.unaddressed > 0, (pid, d)
            assert "Be8+: after Qg5: material +0 / after Qg7: material +0 / after Kh8: material +0 => value 0" in ag.trace()      # the discovery is read to its end
            continue
        assert ag.decision() is not None and ag.decision().what == san and ag.ctx.unaddressed > 0, pid


# ---- 52.17 (3): the king driven off a square, and the promotion that is an exchange -----------------------------------------------------------------

def test_52_17_rh8_does_not_drive_the_king_off_c8_but_rd8_after_kd7_does(monkeypatch):
    from catspace.incubator.analyst.discovery import race as _RC62
    monkeypatch.setattr(_RC62, "DENY", False)          # plan 62.1b adds deny-the-path routes to the stopper fix; this test reads the drive-off routes alone
    b = chess.Board(LRTF)
    assert not R.drives_off(b, chess.Move.from_uci("h7h8"), chess.C8, chess.C8)                 # Kd7 still touches c8
    c = chess.Board(LRTF)
    c.push_san("Rh8+")
    c.push_san("Kd7")
    assert R.drives_off(c, chess.Move.from_uci("h8d8"), chess.C8, chess.D7)                     # Ke6, Ke7; Kxd8 is illegal (c7 guards d8)
    assert [r.san for r in R.drive_off_routes(c, "w", chess.C8, chess.D7)] == ["Rd8+"]
    q = R.new_question("promote", R.STAKE_PROMOTE, dict(pawn=chess.C7))
    f = R.promote_fixes(c, chess.C7, R.Defect("stopper", chess.C8, [chess.D7], "their K reaches c8"), q)
    assert [r.san for r in f.routes] == ["Rd8+"] and "drive their king off c8" in f.text


def test_52_17_0lrtf_is_decided_rh8_by_the_drive_off_step_and_the_exchange_on_the_promotion_square(chain_items_off):
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0LRTF"][0]
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    d = ag.decision()
    t = ag.trace()
    assert d is not None and d.what == "Rh8+" and d.value == 4
    assert "after Kd7: Rd8+ (Rd8+; after Ke6: c8=Q+ (c8=Q+; Rxc8 wins 4: Rxc8 Rxc8: the promotion is worth 8, less the 4 they win on the new queen = 4" in t
    assert ag.ctx.budget_hits == 0 and ag.ctx.asked < 100                                         # before: 254 inner questions and 27 fuse hits, no decision


def test_52_17_the_promotion_exchange_is_read_on_the_board_after_the_push_and_not_claimed_when_the_queen_stands():
    ag = AG.Agenda(State("R7/2P5/8/7k/8/8/p6K/2r5 w - - 0 1", "w", []), side_depth=0, chase_depth=2, ctx=AG.ChaseCtx())
    q = [x for x in ag.all if x.goal == "promote" and x.pattern["pawn"] == chess.C7][0]
    facts = dict(pushes=1, promo=chess.C8)
    cl = ag._promotion_exchange(q, chess.C7, facts)
    assert cl is not None and cl.kind == "win" and cl.value == 8 - G.see_for(chess.Board("R1Q5/8/8/7k/8/8/p6K/2r5 b - - 0 1"), chess.C8, False)
    free = AG.Agenda(State("8/2P5/8/7k/8/8/p6K/8 w - - 0 1", "w", []), side_depth=0, chase_depth=2, ctx=AG.ChaseCtx())
    q2 = [x for x in free.all if x.goal == "promote"][0]
    assert free._promotion_exchange(q2, chess.C7, dict(pushes=1, promo=chess.C8)) is None          # nothing of theirs takes on c8: the passer's business, no exchange claim


# ---- 52.17 (4): cheapest reply first ----------------------------------------------------------------------------------------------------------------

def test_52_17_the_cheap_pass_closes_what_the_rules_close_and_leaves_the_rest_pending_without_an_inner_agenda(chain_items_off):
    ag = AG.Agenda(State(LRTF, "w", []))
    m = chess.Move.from_uci("h7h8")
    key = (AG._pos_key(ag.board.fen()), "h7h8")
    res = ag._chase_route(m, cheap=True)
    assert res.get("pending") and key not in ctx_routes(ag) and ag.ctx.agendas == 0              # Kd7 needs an inner agenda: pending, none opened, nothing memoised
    full = ag._chase_route(m)
    assert not full.get("pending") and key in ctx_routes(ag) and ag.ctx.agendas > 0
    ag.ask()                                                                                       # a route an earlier reply already killed is killed by the cheap pass alone
    before = ag.ctx.agendas
    ag.ctx.routes.pop((AG._pos_key(ag.board.fen()), "h7d7"), None)
    killed = ag._chase_route(chess.Move.from_uci("h7d7"), cheap=True)
    assert killed["killed"] and "standing kill" in killed["killed"] and ag.ctx.agendas == before


def ctx_routes(ag):
    return ag.ctx.routes


# ---- 52.17 (5): the budget is time ----------------------------------------------------------------------------------------------------------------

def test_52_17_a_time_budget_cuts_the_chase_to_gaps_names_the_time_and_no_seconds_changes_nothing():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("dev_20") if x["id"] == "04Qx2"][0]
    timed = AG.Agenda(State(p["fen"], "w", []), seconds=0.05)
    timed.ask()
    t = timed.trace()
    assert "out of time" in t and any("out of time" in g for g in timed.state.gaps) and timed.ctx.time_outs == 1 and timed.ctx.hit
    assert "budget: 0.05 s, used " in t and "OUT OF TIME" in t and "questions" in t.split("budget: ")[1]
    assert timed.ctx.resume and all(r["fen"] and r["text"] and "mark" in r and "cut_left" in r for r in timed.ctx.resume)           # nothing cut by the clock is dropped: each GAP keeps its resume point
    mid = AG.Agenda(State(p["fen"], "w", []), seconds=0.5)                                         # cut in the middle of a route: the reply it stopped at and the memo's deadline mark (0.5 since 2026-10-23: at 0.3 the first inner agenda opened ~0.05 s before the cut at HEAD and ~0.08 s after it with plan 72's priced stopper -- the property is the mark, not the constant)
    mid.ask()
    assert any(r["route"] and r["reply"] for r in mid.ctx.resume)
    assert any(v.get("timed") and v.get("mark") is not None and "cut_left" in v for v in mid.ctx.routes.values() if v.get("gap"))
    assert timed.ctx.agendas <= 8                                                              # past the deadline no inner agenda is opened (5 -> 8 on 2026-10-20: replies a capture
    #                                                                                            # would have closed are now read when the capture leaves a duty, so a few more open in the 0.05 s)
    assert timed.decision() is None or timed.decision().value >= 1 or not timed.decision().proved      # the best bound claim so far decides; a guess is only ever the marked, unproved unfinished line (plan 65.3)
    plain = AG.Agenda(State(p["fen"], "w", []))
    plain.ask()
    assert plain.ctx.time_outs == 0 and "budget:" not in plain.trace() and "out of time" not in plain.trace()
    assert plain.decision().proved and plain.decision().what in Q04_KEEPS                         # seconds=None: today's behaviour (the same decision as the set run; 72.10: one of the equal keeps, see above)


# ---- 52.18 (1): the landing table ----------------------------------------------------------------------------------------------------------------

def test_52_18_the_landing_lemmas_are_proved_and_the_table_is_keyed_by_what_the_answer_read():
    for name in ("land_board_other", "land_agree", "target_of_checkmate", "land_board_to"):
        _lean(name, "Landing")
    assert "(h : ∀ s, S s → p.board s = p'.board s)" in _lean("land_agree", "Landing")["statement"]


def test_52_18_the_discovery_positions_nf6_landing_is_a_target_with_the_discovery_line_in_its_reads():
    from catspace.incubator.analyst.discovery import landing as LD
    b = chess.Board(DISCOVERY)
    t = LD.LandingTable()
    lnd, served = t.ask(b, True, chess.E4, chess.F6)
    assert not served and lnd.kind == "target" and lnd.claim == "exchange" and lnd.value == 5 and "threat 5 survives Bxf6" in lnd.via
    assert {chess.E4, chess.F6, chess.A8, chess.B7, chess.G2, chess.E8} <= lnd.reads                        # the knight, its square, the rook, the queen and the bishop behind it on the line, the queen it also hits
    again, served = t.ask(b, True, chess.E4, chess.F6)
    assert served and again is lnd
    assert t.known_about(chess.G2)[0] is lnd                                                                # "what is known about the bishop on g2": the landing that reads g2
    moved = b.copy()
    moved.push_san("Bf3")                                                                                   # the bishop moves: g2 is in the diff, the answer is asked again
    _l2, served = t.ask(G.with_turn(moved, True), True, chess.E4, chess.F6)
    assert not served


def test_52_18_a_mate_root_defect_list_is_served_from_the_table_when_nothing_it_read_changed():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("dev_20") if x["id"] == "0BHbB"][0]
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    assert ag.ctx.landing_served > 0 and "landing table:" in ag.trace() and "served from the table" in ag.trace() and "known about the Q on g3" in ag.trace()
    assert ag.decision().what == "Qb8+"


# ---- 52.18 (2): the battery and the discovery -------------------------------------------------------------------------------------------------------

BATT = "4q3/8/8/8/4N3/1pK1R3/8/1k6 w - - 0 1"        # Re3 behind Ne4 aimed at the queen on e8; Nd2+ discovers it (SF d18: only Nd2+ wins; mate score)


def test_52_18_the_battery_lemmas_are_proved_in_lean_with_the_statements_the_python_mirrors():
    for name in ("almostBattery_blocked", "discovered_attack", "battery_persists"):
        _lean(name, "Battery")
    assert "almostBatteryB p c s1 s2 t = true" in _lean("discovered_attack", "Battery")["statement"] and "p'.board s1 = none" in _lean("discovered_attack", "Battery")["statement"]
    _lean("batteryB", "Battery")
    _lean("almostBatteryB", "Battery")


def test_52_18_an_almost_battery_does_not_attack_and_its_front_piece_leaving_makes_the_rear_attack_on_random_positions():
    import random
    rng = random.Random(7)
    seen = 0
    for _ in range(120):
        b = chess.Board()
        for _ply in range(rng.randrange(8, 60)):
            ms = list(b.legal_moves)
            if not ms:
                break
            b.push(rng.choice(ms))
        for col in (chess.WHITE, chess.BLACK):
            for t in chess.SQUARES:
                for s1, s2, rest, almost in G.discoverers(b, col, t):
                    if not almost:
                        continue
                    seen += 1
                    assert G.almost_battery(b, col, s1, s2, t) and s2 not in b.attackers(col, t)               # Lean almostBattery_blocked
                    c = b.copy(stack=False)
                    c.remove_piece_at(s1)                                                                       # Lean discovered_attack: the front piece off the line
                    assert s2 in c.attackers(col, t)
    assert seen > 20


# (was xfail strict, LOOP TARGET 1 / plan 72.9: under the POOL at 30 s neither Nf6 nor Nd6 was PROVED. Plan 72.10, 2026-10-23: with the local inner agendas the pool proves Nf6 +1 in
#  two diagnostic runs (OBS: 19.2 s and 19.5 s of wall) and the strict xfail XPASSed in the gate, so the assertion stands as written.)
def test_52_18_the_discovery_position_names_the_discovery_and_the_x_ray_fix_is_offered_on_a_win_defect():
    ag = AG.Agenda(State(DISCOVERY, "w", []), seconds=30)                    # (see the 52.16 discovery test: a time budget since 2026-10-20; 30 s since 64.0's sound close)
    ag.ask()
    t = ag.trace()
    d = ag.decision()
    assert d is not None and d.proved and d.what in ("Nf6", "Nd6")          # (2026-10-23, 72.9: PROVED, as the xfail names the gap -- the unproved fallback flips between Nd6, Nf6 and Nc5 from run to run and XPASSed the strict xfail once)
    assert "discover an attack on a8" in t and "discovery: the N leaves e4, the B on g2 joins the attack on a8 (battery with the Q on b7)" in t
    b = chess.Board(BATT)
    q = AG.Agenda(State(BATT, "w", [])).all[0]
    ag2 = AG.Agenda(State(BATT, "w", []))
    win = [x for x in ag2.all if x.goal == "win" and x.pattern["target"] == chess.E8][0]
    win.defects = R.material_defects(b, "w", chess.E8)
    fixes = R.fixes_of(win.defects[0], win, b, "w")
    disc = [f for f in fixes if f.text.startswith("discover an attack on e8")]
    nd2 = [r for r in disc[0].routes if r.san == "Nd2+"] if disc else []
    assert nd2 and "the N leaves e4, the R on e3 joins the attack on e8" in nd2[0].why and nd2[0].why.startswith("check;")


# ---- 52.18 (3): threat-of-X forcing ------------------------------------------------------------------------------------------------------------------

def test_52_18_a_move_is_forcing_when_a_pass_lets_the_passer_lemma_bind_and_the_reason_names_the_claim_0nv29():
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    p = [x for x in PA.load_set("fresh_60_threats_event") if x["id"] == "0Nv29"][0]
    b = chess.Board(p["fen"])
    why = R.forcing_why(b, chess.Move.from_uci("b5b6"), 8, "w")
    assert why.startswith("threat: b7 then b8=Q binds after a pass (unstoppablePasser2_sound, within 3 plies)")
    assert R.forcing_why(b, chess.Move.from_uci("c5c6"), 8, "w") is None and R.forcing_why(b, chess.Move.from_uci("b5b6"), 9, "w") is None      # a claim worth 8 does not rescue a stake of 9
    assert R.forcing_why(b, chess.Move.from_uci("d3c4"), 8, "w") is None                                                                    # a king move that changes nothing binds nothing new
    n0 = R.PASS_CLAIMS[0]
    R.forcing_why(b, chess.Move.from_uci("b5b6"), 1, "w")
    assert R.PASS_CLAIMS[0] == n0 + 1
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    assert ag.decision().what == "b6"


# ---- 52.18 (4): the fork square as a root -------------------------------------------------------------------------------------------------------------

FORK2 = "8/8/8/8/K1R5/P6k/6r1/3N1r2 w - - 0 1"          # Ne3 forks the rooks on f1 and g2 (SF d18: the only move, +260 against -7)
FORK_DEFECT = "8/8/3N3r/4b3/2R5/5P2/4K1k1/8 w - - 0 1"  # Nf7 forks Rh6 and Be5 but Rh5 moves one and guards the other


def test_52_18_the_fork_lemmas_are_proved_in_lean():
    _lean("one_move_touches_one", "ForkSquare")
    assert "(a b : Square) (hab : a ≠ b)" in _lean("one_move_touches_one", "ForkSquare")["statement"]
    _lean("knight_fork_same_colour", "GeoLemmas")


def test_52_18_a_fork_root_is_asked_for_a_landing_that_is_safe_and_attacks_two_pieces_it_wins():
    ag = AG.Agenda(State(FORK2, "w", []))
    forks = [q for q in ag.all if q.goal == "fork"]
    assert [q.pattern["san"] for q in forks] == ["Ne3"] and forks[0].stake == 2 and "attacks the R on f1 (wins 5), the R on g2 (wins 2)" in forks[0].text
    ag._ask(forks[0])
    assert forks[0].status == "verified" and ag.best[2].what == "Ne3" and ag.best[0] == 2
    assert "least over their replies of balance + exchange is 2" in ag.best[2].line and "ForkSquare.one_move_touches_one" in ag.best[2].lemma
    full = AG.Agenda(State(FORK2, "w", []))
    full.ask()
    assert full.decision().what == "Ne3"                                                                       # the same decision whichever root finds it first


def test_52_18_a_reply_that_answers_both_threats_kills_the_fork_with_the_defect_named_and_a_knight_cannot_fork_two_colours():
    ag = AG.Agenda(State(FORK_DEFECT, "w", []))
    q = [x for x in ag.all if x.goal == "fork" and x.pattern["san"] == "Nf7"][0]
    ag._ask(q)
    assert q.status == "killed" and "Rh5 (a move of one attacked piece that answers the other)" in q.reason
    ag2 = AG.Agenda(State(DISCOVERY, "w", []))
    notes = "\n".join(ag2.notes)
    assert "by a knight: killed by reasoning -- a knight attacks only squares of one colour (Ideas.GeoLemmas.knight_fork_same_colour)" in notes and "reads {" in notes


# ---- 52.22 (1), (2): a reply closes by rule only when the goal is met; the attacker added with tempo ------------------------------------------------------------

Z0P = "2r2rk1/1p4bp/6p1/pP1qpp2/6PP/P1n1BQ2/2B2R2/5RK1 w - - 1 29"


def _z0p_after_bb3_rfd8():
    b = chess.Board(Z0P)
    for m in ("Qxd5+", "Nxd5"):
        b.push_san(m)
    before = b.copy()
    b.push_san("Bb3")
    b.push_san("Rfd8")
    return before, b


def test_52_22_2_the_defended_knight_is_fixed_by_adding_an_attacker_with_tempo():
    _, b = _z0p_after_bb3_rfd8()
    win = R.new_question("win", 3, dict(target=chess.D5))
    ds = R.material_defects(b, "w", chess.D5)
    assert kinds(ds) == ["defended"]
    fixes = R.fixes_of(ds[0], win, b, "w")
    add = [f for f in fixes if f.text == "add an attacker of d5 with tempo"]
    assert add and {"Rd2", "Rd1"} <= {r.san for r in add[0].routes}


def test_52_22_1_the_exchange_closes_a_reply_only_when_it_reaches_the_goal_or_no_forcing_fix_remains():
    before, after = _z0p_after_bb3_rfd8()
    P2 = AC.Pos(after.fen())
    ag = AG.Agenda(State(before.fen(), "w", []))
    win3 = R.new_question("win", 3, dict(target=chess.D5))
    win1 = R.new_question("win", 1, dict(target=chess.D5))
    ag._chase_q = win1
    assert ag._closed_by_rule(P2, 0) is not None and "exchange +1" in ag._closed_by_rule(P2, 0)[1]          # the goal worth 1 is met by the exchange: closed as before
    ag._chase_q = win3
    n0 = ag.ctx.unmet
    assert ag._closed_by_rule(P2, 0) is None and ag.ctx.unmet == n0 + 1                                       # the goal worth 3 is not: a forcing fix (Rd2) remains, the reply is asked
    assert "not closed by rule: the goal" in ag.ctx.open_lines[-1] and "Rd2" in ag.ctx.open_lines[-1]
    ag._chase_q = None
    assert ag._closed_by_rule(P2, 0) is not None                                                              # no chased win goal: the rule is unchanged


# ---- 52.22 (3): the refuter fix -----------------------------------------------------------------------------------------------------------------------

def test_52_22_3_the_refuter_is_a_word_proved_in_lean():
    st = _lean("refuter_unforces", "Refuter")["statement"]
    assert "Refutes" in st and "forcedAtLeaves" in st


def test_52_22_3_07z0p_is_decided_qxd5_by_the_refuter_fix_with_the_line_printed(chain_items_off, monkeypatch):
    monkeypatch.setattr(AG, "CHASE_SAVES", False)       # plan 72.10: with the route memo per goal the save question's KEEP chase (70.1) proves Qxd5+ +1 first (no longer served the mate goal's kill
    ag = AG.Agenda(state_of(Z0P, "w"))                  # made at target 1), so the refuter fix that decides it otherwise is read with the 70.1 chase off
    ag.ask()
    t = ag.trace()
    assert ag.decision().what == "Qxd5+"
    assert "Qxd5+: refuter fix -- Bb3 fails to Qxb3 (the refuter: the Q on d5); exchange it first -- Qxd5+ Nxd5: the N on d5 cannot make that move; Bb3 is asked again after it" in t
    assert ag.ctx.refuter_fixes >= 1


ZB = "2k5/1p6/p1P1p1p1/2P5/P2Pp1p1/4PrPq/2Q1RP1r/3RK3 b - - 0 34"      # zBouR (rand2_10): Rhxf2 cxb7+ Kb8 Rxf2 Qh1+ Ke2 Rxf2+ Kxf2 Qh2+ Ke1 Qxc2


def _after(fen, sans):
    b = chess.Board(fen)
    for s in sans.split():
        b.push_san(s)
    return b


def test_52_23_4_the_skewer_is_a_word_proved_in_lean():
    st = _lean("skewer_attack", "Skewer")["statement"]
    assert "skewerB" in st and "pieceAttacksB" in st
    st = _lean("skewer_blocked", "Skewer")["statement"]
    assert "= false" in st


def test_52_23_4_the_skewer_routes_of_zbour_are_the_rook_check_through_e2_and_the_queen_checks_through_f2():
    b = _after(ZB, "Rhxf2 cxb7+ Kb8 Rxf2 Qh1+ Ke2")
    rs = R.skewer_routes(b, b.turn, chess.C2)
    assert [r.san for r in rs] == ["Rxf2+"] and "through their K on e2 to the Q on c2" in rs[0].why
    b = _after(ZB, "Rhxf2 cxb7+ Kb8 Rxf2 Qh1+ Ke2 Rxf2+ Kxf2")
    assert sorted(r.san for r in R.skewer_routes(b, b.turn, chess.C2)) == ["Qg2+", "Qh2+"]
    c = b.copy(); c.push_san("Qh2+")
    assert G.skewer_after(c, not c.turn, chess.C2) == chess.H2
    assert chess.H2 not in c.attackers(not c.turn, chess.C2)      # skewer_blocked: the king is in the way
    c.push_san("Ke1")
    assert chess.H2 in c.attackers(c.turn, chess.C2)                                                                  # skewer_attack: the king stepped off the line


def test_52_23_4_a_rook_does_not_skewer_along_a_diagonal_and_a_blocked_line_is_no_skewer():
    b = chess.Board("4k3/8/8/8/8/8/8/R3K2q w - - 0 1")      # Ra1-e1? the rook on a1 cannot check e8; put the rook so that a check along a diagonal would be needed
    assert R.skewer_routes(b, b.turn, chess.H1) == []
    b = chess.Board("8/8/8/8/8/8/k1q5/R3K3 b - - 0 1")      # their K a2 and Q c2 on the rank; our R a1 checks a2 along the file, not the rank: no skewer on c2
    b.turn = True
    assert R.skewer_routes(b, True, chess.C2) == []
    b = chess.Board("8/8/8/8/8/8/1q6/k1nRK3 w - - 0 1")     # Rd1 checks along the rank? the N on c1 stands between d1 and a1: no check at all
    assert R.skewer_routes(b, True, chess.B2) == []


def test_52_23_3_an_inner_agenda_in_check_chases_its_evasions_once(monkeypatch):
    monkeypatch.setattr(AG, "EVASION_CHASE", True)               # the item is OFF by default (52.23: the chain items wait for the time regime and the prior)
    st = state_of(ZB, "b")
    root = AG.Agenda(st, seconds=60)
    b = _after(ZB, "Rhxf2 cxb7+")
    ag = AG.Agenda(State(fen=b.fen(), us="b", pieces=[]), side_depth=0, chase_depth=1, ctx=root.ctx, min_stake=1)
    q = R.new_question("save", R.STAKE_MATE, dict(checkers=[chess.B7]), text="save our king (in check)")
    assert ag.evaded is False                                   # the gate alone: an evading agenda chases, an evaded one does not chase a second check of theirs
    ag2 = AG.Agenda(State(fen=b.fen(), us="b", pieces=[]), side_depth=0, chase_depth=2, ctx=root.ctx, min_stake=1, evaded=True)
    assert ag2.evaded is True
    assert ag2._chase(q) is False and "a second check of theirs in a row" in ag2.lines[-1]
