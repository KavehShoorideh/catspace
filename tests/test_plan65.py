"""PLAN 65 (docs/ANALYST_PLAN.md, 'time is a priced choice across the tiers'). One position each: the numbers are directional (OBS lines)."""
import re
from pathlib import Path

import chess
import pytest

from catspace.incubator.analyst.discovery import agenda as AGM
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.agenda import Agenda, judge_next
from catspace.incubator.analyst.discovery.state import state_of

POS2 = "r1bqnr1k/1p1n2p1/p2p2P1/2pPp3/4Pp1b/2N1BP1N/PPP4Q/2KR1B2 w - - 0 1"
VDTX = "r5k1/7p/pp2q1p1/2p5/3pN3/PP1nb3/1B5Q/1K3R2 w - - 0 33"       # rand3_10 2VDtX: Nf6+ is a cost-8 check, Qxh7+ a cost-5 one
LEAN = Path(__file__).resolve().parents[1] / "lean/ideas/Ideas/Forcing.lean"


def _agenda(fen: str, seconds: float = 15.0) -> Agenda:
    ag = Agenda(state_of(fen, "w" if chess.Board(fen).turn else "b"), seconds=seconds)
    ag.ctx.held = lambda: False                                    # (what the tier driver sets)
    ag.ctx.judge_t, ag.ctx.judge_n = 1.0, 100                      # a measured rate: 0.01 s per cost unit
    return ag


def _q(ag: Agenda, stake: int = 100):
    return next(t[4] for t in ag.heap if t[4].stake == stake)


def test_65_1_judge_next_is_the_lean_definition():
    """Lean `Ideas.Forcing.judgeNext` (+ judgeNext_cut / _read / _mono / none_affordable_of_cheapest) is in the file, no sorry, and the Python mirror agrees on a grid."""
    src = LEAN.read_text()
    for name in ("def judgeNext", "theorem judgeNext_cut", "theorem judgeNext_read", "theorem judgeNext_mono", "theorem none_affordable_of_cheapest"):
        assert name in src
    assert "sorry" not in src
    for c in (None, 0, 3, 7):
        for k in range(0, 10):
            lean = True if c is None else k <= c                          # judgeNext none k = true; judgeNext (some c) k = decide (k <= c)
            assert judge_next(c, k) == lean
            if c is not None and c < k:
                assert not judge_next(c, k)                               # judgeNext_cut
            if c is not None and k <= c:
                assert judge_next(c, k)                                   # judgeNext_read
    assert judge_next(None, 10 ** 9)


def test_65_1_the_next_tiers_cheaper_route_cuts_this_tiers_dearer_one():
    """2VDtX: with a rate known, a finished earlier read of Qxh7+ at 0.001 s is a tier-1 route cheaper than Nf6+ (cost 8 x 0.01 s): Nf6+ is a GAP 'cut by the judge', read in tier 1.
    Held (the status already kept), or no rate, or the last tier: nothing competes."""
    ag = _agenda(VDTX)
    q = _q(ag)
    ag._chase_q = q
    b = ag.board
    nf6, qxh7 = b.parse_san("Nf6+"), b.parse_san("Qxh7+")
    ag.ctx.route_secs[(ag.pk, qxh7.uci(), q.stake)] = (0.001, True)
    res, tok = ag._judge_begin(nf6, False, False, None)
    print("OBS 65.1 cut:", res["gap"] if res else None)
    assert res is not None and res["gap"].startswith("out of time: Nf6+ cut by the judge (tier 1's Qxh7+ is cheaper") and ag.ctx.compete_cuts == 1
    assert ag.ctx.root_gaps[(ag.pk, nf6.uci(), q.stake)][1] == "Nf6+"      # the open route 65.2 reads
    # the status is kept: no tier opens, nothing competes
    ag2 = _agenda(VDTX)
    ag2._chase_q = _q(ag2)
    ag2.ctx.held = lambda: True
    ag2.ctx.route_secs[(ag2.pk, qxh7.uci(), q.stake)] = (0.001, True)
    res2, _ = ag2._judge_begin(nf6, False, False, None)
    assert res2 is None
    # the last tier has no next tier
    ag3 = _agenda(VDTX)
    ag3._chase_q = _q(ag3)
    ag3.ctx.tier = AGM.TIER_COUNT - 1
    ag3.ctx.route_secs[(ag3.pk, qxh7.uci(), q.stake)] = (0.001, True)
    assert ag3._judge_begin(nf6, False, False, None)[0] is None
    # no rate yet: no price in seconds, nothing competes
    ag4 = _agenda(VDTX)
    ag4.ctx.judge_t, ag4.ctx.judge_n = 0.0, 0
    ag4._chase_q = _q(ag4)
    assert ag4._competitor(10.0, 0.0) is None


def test_65_1_a_route_already_open_or_deferred_once_is_not_cut():
    """Tier k stops opening NEW root routes: a route this tier has already opened (4pC83's Nf3+, retried under later questions) is read on; a route the previous tier deferred is read now (deferred once)."""
    for how in ("opened", "exempt"):
        ag = _agenda(VDTX)
        q = _q(ag)
        ag._chase_q = q
        b = ag.board
        nf6, qxh7 = b.parse_san("Nf6+"), b.parse_san("Qxh7+")
        ag.ctx.route_secs[(ag.pk, qxh7.uci(), q.stake)] = (0.001, True)
        key = (ag.pk, nf6.uci(), q.stake)
        if how == "opened":
            ag.ctx.opened.add(key)
        else:
            ag.ctx.exempt = frozenset({key})
        res, tok = ag._judge_begin(nf6, False, False, None)
        print("OBS 65.1", how, "->", "read on" if res is None else res["gap"])
        assert res is None and tok is not None


def test_65_1_a_tie_is_read_and_the_cheapest_class_is_never_cut():
    """The route's own re-read in the next tier is priced the same as itself: never strictly cheaper (the cheapest route is read in its own tier)."""
    ag = _agenda(VDTX)
    q = _q(ag)
    ag._chase_q = q
    qxh7 = ag.board.parse_san("Qxh7+")
    res, tok = ag._judge_begin(qxh7, False, False, None)
    print("OBS 65.1 the cheapest route:", "read on" if res is None else res["gap"])
    assert res is None and tok is not None


def test_65_1_the_competitor_set_includes_the_wider_vocabulary():
    """Tier 0 -> 1 adds the delayed threat: on 07Z0P's root Re2 / Rd1 are candidates at stake 1 (not forcing in tier 0)."""
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    (p,) = [x for x in PA.load_set("dev_20") if x["id"] == "07Z0P"]
    b = chess.Board(p["fen"])
    tier0 = {m.uci() for m in R._moves(b, "w" if b.turn else "b") if R.forcing_why(b, m, 1, "w" if b.turn else "b", mate_threats=True)}
    tier1 = {m.uci() for m in R._moves(b, "w" if b.turn else "b") if R.forcing_why(b, m, 1, "w" if b.turn else "b", mate_threats=True, delayed=True, nosafe=True)}
    print("OBS 65.1 07Z0P stake 1: tier 0 routes", len(tier0), "tier 1 routes", len(tier1), "new", sorted(b.san(chess.Move.from_uci(u)) for u in tier1 - tier0))
    assert tier0 < tier1 and tier0


def test_65_2_nothing_affordable_stops_the_widening():
    """Lean none_affordable_of_cheapest: with a rate known, the cheapest open route (a GAP of tier k) priced above the time left ends the widening, with the TIERS line; one that fits does not; no rate, no stop."""
    ag = _agenda(VDTX)
    q = _q(ag)
    nf6, qxh7 = ag.board.parse_san("Nf6+").uci(), ag.board.parse_san("Qxh7+").uci()
    ag.ctx.root_gaps[(ag.pk, nf6, q.stake)] = (500, nf6)
    ag.ctx.root_gaps[(ag.pk, qxh7, q.stake)] = (800, qxh7)
    ok, line = ag._affordable(ag, 0, 1.0, 0.0 + __import__("time").monotonic())
    print("OBS 65.2", ok, line)
    assert not ok and line.startswith("TIERS: stopped at") and "(tier 0, Nf6+) is priced 500 x 0.0100 = 5.0 s > the 1.0 s left" in line
    assert ag._affordable(ag, 0, 10.0, 0.0)[0]                      # 5.0 s fits in 10 s
    ag.ctx.judge_t, ag.ctx.judge_n = 0.0, 0
    assert ag._affordable(ag, 0, 1.0, 0.0)[0]                       # no rate yet: nothing is priced, nothing is said


def test_65_2_the_wider_vocabulary_adds_open_routes():
    """07Z0P: tier 0 -> 1 adds Re2 and Rd1 (the delayed threat); with nothing else open they are priced and, being dear (all replies are parries), do not fit in a short time left."""
    from catspace.incubator.analyst.discovery import puzzle_apply as PA
    (p,) = [x for x in PA.load_set("dev_20") if x["id"] == "07Z0P"]
    ag = _agenda(p["fen"])
    q = next(t[4] for t in ag.heap if t[4].stake == 1)
    ag._chase_q = q
    ag._competitor(1.0, 0.01)                                        # fills the candidates of (position, stake 1, tier 0)
    ok, line = ag._affordable(ag, 0, 0.5, __import__("time").monotonic())
    print("OBS 65.2 07Z0P:", ok, line)
    assert not ok and "(tier 1, " in line


def test_65_3_pos2_decides_bf2_as_an_unfinished_line_and_says_so():
    """pos2 (count budget, so no clock): no claim of any kind. Nxf4, Bxc5, Bxf4 are refuted (killed); the unfinished lines are GAPs; Bf2 has the best floor and the lowest price among those tied; the decision is marked unproved."""
    ag = Agenda(state_of(POS2, "w"))
    ag.ask()
    d = ag.decision()
    trace = ag.trace()
    line = next(x for x in trace.splitlines() if x.startswith("no proved claim"))
    print("OBS 65.3", line)
    print("OBS 65.3 decision:", d.what, d.value, "proved" if d.proved else "unproved", d.word)
    assert d is not None and d.what == "Bf2" and d.proved is False and d.word == "unfinished line" and d.lemma == "unfinished line"
    assert "Bf2 is played as the best unfinished line" in line and "Nxf4" in line and "refuted" in line
    assert ag.best is None and ag.best_est is None
    killed = {x["san"] for x in ag.root_reads.values() if x["status"] == "killed"}
    assert "Nxf4" in killed and d.what not in killed                       # a kill is a proved loss: an unfinished line is preferred to it


def test_65_3_never_when_a_claim_exists():
    """2VDtX has a proved claim (Nf6+): no 'no proved claim' line, the claim is proved, nothing unfinished decides."""
    ag = Agenda(state_of(VDTX, "w"))
    ag.ask()
    d = ag.decision()
    print("OBS 65.3 2VDtX:", d.what, "proved" if d.proved else "unproved")
    assert d is not None and d.proved and d.word == ""
    assert "no proved claim" not in ag.trace()


def test_65_3_off_switch_leaves_no_decision():
    """The comparison condition (analyse --no-unproved): pos2 decides nothing, as plan 64 left it."""
    old = AGM.UNPROVED
    try:
        AGM.UNPROVED = False
        ag = Agenda(state_of(POS2, "w"))
        ag.ask()
        assert ag.decision() is None
    finally:
        AGM.UNPROVED = old
