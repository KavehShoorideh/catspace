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
    assert res["killed"] and "stands only as a balance, their Qa7+ wins 100" in res["killed"] and res["killed"].endswith("no parry") and res["value"] == 0
    ag.ask()
    assert all(c.what != "Nxg2" for c in ag.state.claims) and ag.decision().what != "Nxg2"
    assert not hasattr(AG.Agenda, "_static_refuted")


def test_52_13_a_reply_that_leaves_us_a_mate_in_one_or_an_unanswered_exchange_is_closed_by_rule():
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    ag = AG.Agenda(state_of(fen, "w"))
    assert ag._closed_by_rule(AC.Pos(fen), 0) == (BD.MATE_VALUE, "mate in one")
    ag = AG.Agenda(state_of(AGL8, "b"))
    res = ag._chase_route(chess.Move.from_uci("g5g2"))        # Qxg2+ Qxg2: the knight takes the queen, nothing of theirs
    assert res["value"] == 1 and any("closed by rule: exchange +9, nothing of theirs" in p for p in res["parts"])
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


def test_52_14_a_threat_is_parried_and_the_gain_kept_on_a_constructed_static_leaf():
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
    assert gone.threat_qs[0][0].status == "killed" and "no parry keeps the gain" in gone.threat_qs[0][0].reason      # a gain of 0 is not kept by any parry (net 0 < 1)


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
    assert "b6 (offers the pawn to the stopper: the B on a5 takes" in t                       # the route the stopper rule adds; no forcing_why route exists here
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
    assert len(res["parts"]) == 7 and all("-> 8" in x for x in res["parts"]) and "unstoppablePasser2_sound" in res["lemmas"]       # every reply of the lone king is answered by a promote claim


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
    ag = AG.Agenda(State(p["fen"], "w", []))
    ag.ask()
    t = ag.trace()
    assert "fuse: this root is charged 120 of the pool 240" in t                              # the highest stake is charged half, never all
    assert "fuse left -" not in t and ag.ctx.left >= 0                                         # before: -14 (239 questions on a route share of 120)
    assert "Rd7: killed -- after Kxd7" in t                                                    # the second route had a share of 1 before; now it is read to its kill
    assert any("Rh8+ not followed past Kd7" in g for g in ag.state.gaps)                       # the remaining gap, named: the walk of the king under our rook checks (no fuse closes it)
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
