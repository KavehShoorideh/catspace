"""PLAN 72.8 THE SKELETON GOALS (Kaveh 2026-10-23): the slow units (kings, pawns, the units that cannot run) are the fixed board; our fast units are imagined where the goal's
defects need them, on counterfactual boards (skeleton A: our fast units lifted; skeleton B: theirs too), priced by the roll forward (72.1), weighed by the stake rule (72.3).
An arrangement is a CANDIDATE: it adds routes, roots and a policy feature; it never claims and never kills. Constructed positions, count budgets only; numbers are directional (OBS)."""
import re

import chess

from catspace.incubator.analyst.discovery import agenda as AGM
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.graph import clear_caches
from catspace.incubator.analyst.discovery.state import state_of

DISCOVERY = "r3q2k/pQ1nb1pp/6n1/5bP1/3PN3/2N1P3/PPP3B1/R1B3K1 w - - 0 1"
POS2 = "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1"
PINNED = "k3r3/8/8/8/8/8/1P2N3/B3K2R w - - 0 1"            # the knight on e2 pinned to the king, the bishop on a1 behind its own pawn, the rook on h1 free
DUTY = "k2r4/8/8/8/8/8/3N4/3QK3 w - - 0 1"                 # the knight on d2 pinned to the queen: every safe square abandons it
TWO_JOBS = "7k/8/8/8/8/4N3/4B3/2KR4 w - - 0 1"             # Rd8+ leaves g7 and h7 to the king: Be2-d3 covers h7, Ne3-f5 covers g7
STEP_ASIDE = "7k/6pp/8/8/R7/8/8/Q5K1 w - - 0 1"            # Qa8 is mate, but our own rook stands on the a-file
DEFLECT = "7k/R5pp/8/8/6Q1/2q5/8/7K w - - 0 1"             # Qxg7 with the rook behind it, but their queen on c3 guards g7
REFUSE = "b3k2r/1p3p2/8/8/3N4/2K5/8/B7 w - - 0 1"          # their bishop on a8 is caged (a win target), our knight needs two moves; their rook hits our bishop on a1 in one


def _goals(fen):
    clear_caches()
    R._SKEL.clear()
    b = chess.Board(fen)
    return b, R.skeleton_goals(b, b.turn)


def _placed(arrs, unit, sq):
    return next(a for a in arrs if a.placement == (unit, sq))


def test_72_8_the_partition_a_pinned_knight_and_a_stuck_bishop_are_slow_a_free_rook_is_fast():
    b = chess.Board(PINNED)
    part = R.skeleton_partition(b)
    assert part[chess.E2] == "no safe square" and part[chess.A1] == "no safe square" and part[chess.H1] is None
    assert part[chess.E1] == "king" and part[chess.B2] == "pawn" and part[chess.E8] is None          # (their rook is fast too: either colour)
    assert R.skeleton_line(b, True, part) == "skeleton: fixed K e1, P b2, B a1 (no safe square), N e2 (no safe square); fast R h1"
    d = R.skeleton_partition(chess.Board(DUTY))
    assert d[chess.D2] == "duty-bound" and d[chess.D1] is None


def test_72_8_skeleton_a_removes_exactly_our_fast_units_and_b_theirs_too():
    b = chess.Board(DISCOVERY)
    part = R.skeleton_partition(b)
    ours = {s for s, w in part.items() if w is None and b.piece_at(s).color == chess.WHITE}
    theirs = {s for s, w in part.items() if w is None and b.piece_at(s).color == chess.BLACK}
    assert ours and theirs
    a, bb = R.skeleton_a(b, True, part), R.skeleton_b(b, True, part)
    assert set(a.piece_map()) == set(b.piece_map()) - ours and all(a.piece_at(s) == b.piece_at(s) for s in a.piece_map())
    assert set(bb.piece_map()) == set(b.piece_map()) - ours - theirs
    assert a.turn == chess.WHITE and bb.turn == chess.WHITE and a.ep_square is None and not a.castling_rights
    assert b.board_fen() == chess.Board(DISCOVERY).board_fen()                                        # the live board is never touched


def test_72_8_a_mate_arrangement_with_two_jobs_priced_0_3_is_reachable_and_its_steps_become_routes():
    b, arrs = _goals(TWO_JOBS)
    a = _placed(arrs, chess.D1, chess.D8)
    assert sorted((chess.square_name(j.square), j.kind) for j in a.jobs) == [("g7", "flight"), ("h7", "flight")]
    assert (a.material, a.tempi, a.counter, a.verdict) == (0, 3, 0, "reachable")
    assert a.text == "arrangement mate on d8 (R d1->d8, their king on h8): R d1->d8 (1), B e2->d3 bears on h7 (1), N e3->f5 bears on g7 (1); price (0, 3), counter 0 -> reachable"
    ag = AGM.Agenda(state_of(TWO_JOBS, "w"))                                                           # (11b) the one-move root on d8 carries it, and asks its first steps as a fix
    q = next(q for q in ag.all if q.goal == "mate" and q.pattern.get("at") == chess.D8 and q.pattern.get("frm") == chess.D1 and q.pattern.get("move"))
    assert (q.pattern["arrangement"]["material"], q.pattern["arrangement"]["tempi"], q.pattern["arrangement"]["counter"]) == (0, 3, 0) and R.arrangement_value(q) == 100
    f = ag._arrangement_fix(q)
    assert f is not None and f in q.fixes and [r.san for r in f.routes] == ["Rd8+"]                   # only a FORCING first step is a route (72.3 corrected: a quiet step is no fix)
    r = f.routes[0]
    assert r.price == (0, 1) and r.why.startswith("check; bring the R from d1 to d8 in 1, price 0 [arrangement mate on d8 (R d1->d8, their king on h8): ")
    assert f.unforced == ["Bd3", "Nf5"] and f"  arrangement Q{q.id}: quiet first step(s) Bd3, Nf5 -- candidates, not routes (a quiet step hands them a free move)" in ag.lines


def test_72_8_step_aside_our_own_rook_blocks_the_queens_path():
    b, arrs = _goals(STEP_ASIDE)
    a = _placed(arrs, chess.A1, chess.A8)
    assert [x for x, _t in a.step_asides] == [chess.A4] and (a.material, a.tempi) == (0, 2) and not a.jobs
    assert a.text.startswith("arrangement mate on a8 (Q a1->a8, their king on h8): Q a1->a8 (1), R a4 step-aside (+1); price (0, 2)")
    r = _placed(arrs, chess.A4, chess.A8)                                                              # the rook itself has nothing in its way
    assert r.step_asides == [] and r.tempi == 1


def test_72_8_a_deflection_their_queen_guards_the_mating_square_the_arrangement_is_open():
    b, arrs = _goals(DEFLECT)
    a = _placed(arrs, chess.G4, chess.G7)
    assert [(j.square, j.kind) for j in a.jobs] == [(chess.G7, "guard")] and a.assignment[1][0] == chess.A7      # the rook guards g7 already
    assert a.verdict == "open" and [(d.their_sq, d.job_square) for d in a.deflections] == [(chess.C3, chess.G7)]
    assert "deflect their Q from c3 (open)" in a.text and a.text.endswith("-> open: deflection")


def test_72_8_the_stake_rule_refuses_when_their_counter_plan_wins_more_in_the_tempi_we_spend():
    b, arrs = _goals(REFUSE)
    a = next(a for a in arrs if a.goal == "win" and a.target == chess.A8)
    assert (a.material, a.tempi, a.stake, a.counter, a.verdict) == (0, 2, 3, 3, "refused")
    assert a.text == "arrangement win their B on a8: N d4->c7 bears on a8 (2); price (0, 2), counter 3 -> refused: their R on h8 reaches our B on a1 in 1"
    ag = AGM.Agenda(state_of(REFUSE, "w"))                                                             # a refused arrangement adds no route and no root
    q = next(q for q in ag.all if q.goal == "win" and q.pattern.get("target") == chess.A8)
    assert "arrangement" not in q.pattern and ag._arrangement_fix(q) is None


def test_72_8_the_trace_lines_on_the_discovery_position_and_the_cost():
    ags = {}
    for name, fen in (("discovery", DISCOVERY), ("pos2", POS2)):
        R._SKEL.clear()
        ags[name] = ag = AGM.Agenda(state_of(fen, "w"))
        obs = ag.skeleton_obs(name)
        print(obs)
        assert re.match(rf"OBS skeleton goals {name}: targets \d+, arrangements \d+ \(reachable \d+, open \d+, refused \d+, unreachable \d+\), ms \d+$", obs)
        assert ag.skeleton_stats["ms"] < 750                                                         # (12) under 5 % of a 15 s run
    lines = ags["discovery"].lines
    i = next(i for i, l in enumerate(lines) if l.startswith("skeleton: "))
    assert lines[i] == "skeleton: fixed K g1, P a2 b2 c2 e3 d4 g5; fast Q b7, R a1, B c1, B g2, N c3, N e4"
    arr = [l for l in lines if l.startswith("arrangement ")]
    assert 0 < len(arr) <= R.SKELETON_LINES and lines[i + 1:i + 1 + len(arr)] == arr
    pat = r"arrangement (mate on|win their|promote our) .+: .+; price \((\d+, \d+|-, -)\), counter (\d+|-) -> (reachable|open: deflection|refused: .+|unreachable: .+)$"
    assert all(re.match(pat, l) for l in arr)
    assert arr[0].startswith("arrangement mate on g7 (Q b7->g7 by the arrangement, their king on h8): Q b7->g7 (3), N e4->h5 bears on g7 (2); price (0, 5)")
    their = AGM.Agenda(state_of(DISCOVERY, "w"), side_depth=0)                                        # their agendas (and the inner ones) print none
    assert not any(l.startswith(("skeleton:", "arrangement ")) for l in their.lines)
    ag = AGM.Agenda(state_of(TWO_JOBS, "w"))
    ag.ask()
    assert sum(1 for l in ag.trace().splitlines() if l.lstrip().startswith("skeleton:")) == 1           # once per root agenda


def test_72_8_a_new_root_only_for_a_reachable_arrangement_no_root_matched():
    R._SKEL.clear()
    AGM.SKELETON_GOALS = False
    try:
        base = AGM.Agenda(state_of(DISCOVERY, "w"))
    finally:
        AGM.SKELETON_GOALS = True
    assert not any(l.startswith(("skeleton:", "arrangement ")) for l in base.lines)
    ag = AGM.Agenda(state_of(DISCOVERY, "w"))
    old = {q.text for q in base.all}
    new = [q for q in ag.all if q.text not in old]
    arrs = R.skeleton_goals(ag.board, ag.me)
    had = {q.pattern.get("at") for q in base.all if q.goal == "mate"}
    want = {a.about for a in arrs if a.verdict == "reachable" and a.goal == "mate" and a.about not in had and R.arrangement_steps(ag.board, a)}
    assert new and {q.pattern["at"] for q in new} == want
    for q in new:
        assert q.goal == "mate" and "by the arrangement" in q.text and q.pattern["reach"] and q.defects[0].kind == "reach" and q.pattern["arrangement"]["tempi"] > 0
    shut = {a.about for a in arrs if a.verdict != "reachable"} - {a.about for a in arrs if a.verdict == "reachable"}
    assert shut and not shut & {q.pattern["at"] for q in new}                                         # open, refused, unreachable: nothing pushed
