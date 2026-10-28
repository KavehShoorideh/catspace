"""PLAN 59: status-driven widening. The STATUS word and the line on small positions; KEEPS THE STATUS; the tier DRIVER (stops at the first tier that keeps; decision = the best claim over the
tiers run); the tier settings read through the root's ChaseCtx; the mate-in-two threat of tier 2 (a proved word); and TIERS off identical on 0LRTF and 07Z0P (the full diff is
scripts/compare_tiers.py against a run made before plan 59)."""
import chess
import pytest

import catspace.incubator.analyst  # noqa: F401
from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import analyse as AN
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.state import Claim, state_of

LOSING = "4k3/8/8/8/8/8/8/3qK3 w - - 0 1"                 # white is down a queen (balance -9)
WINNING = "4k3/8/8/8/8/8/8/3QK3 w - - 0 1"                # white is up a queen with nothing to fear
LEVEL = "4k3/8/8/8/8/8/8/4K3 w - - 0 1"
MATE2 = "4R3/8/7Q/8/6K1/k7/8/8 w - - 0 1"                 # Rh8 (quiet) threatens a mate in two after a pass, not a mate in one


@pytest.fixture(autouse=True)
def _terms_and_flags():
    BV.install_terms()
    saved = AG.TIERS
    yield
    AG.TIERS = saved


def agenda(fen: str, **kw) -> "AG.Agenda":
    return AG.Agenda(state_of(fen, "w" if chess.Board(fen).turn else "b"), **kw)


def claim(what="Kd2", value=1) -> Claim:
    return Claim(kind="win", what=what, value=value, lemma="test", line="test line", move="e1d2")


@pytest.fixture(autouse=True)
def _tiers_not_pool(monkeypatch):
    """This file tests the TIERS (plan 59). Since 69.1 the POOL is the default driver under a clock and under TIERS=True at the count budget alike; the tiers are
    the fallback. Every test here runs with the pool off."""
    monkeypatch.setattr(AG, "POOL", False)


def test_status_word_and_line_on_small_positions():
    st = agenda(LEVEL).status()
    assert (st["balance"], st["threat"], st["word"], st["line"]) == (0, 0, "level", 0)
    st = agenda(WINNING).status()
    assert (st["balance"], st["threat"], st["word"], st["line"]) == (9, 0, "winning", 1)
    st = agenda(LOSING).status()
    assert (st["balance"], st["word"], st["line"]) == (-9, "losing", 0)


def test_status_subtracts_their_pass_threat():
    p = next(x for x in PA.load_set("rand22_5") if x["id"] == "hqGOF")            # balance -1; with the move passed to them they win 9 (printed by `analyse --tiers`, 2026-10-20)
    ag = agenda(p["fen"])
    ag.ask()
    st = ag.status()
    assert (st["balance"], st["threat"], st["s0"], st["word"], st["line"]) == (-1, 9, -10, "losing", 0)
    assert ag.status() == st                                 # (read once from the threat roots, no second run of their agenda)


def test_status_in_check_has_no_pass():
    st = agenda("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1").status()
    assert st["threat"] == 0


def test_keeps_the_status():
    ag = agenda(LOSING)
    st = ag.status()                                       # balance -9, the line is 0
    assert ag.keeps(st, None) == (False, None)             # no claim: not kept
    assert ag.keeps(st, (8, 0, claim(value=8))) == (False, 8)       # -9 + 8 = -1 < 0
    assert ag.keeps(st, (9, 0, claim(value=9))) == (True, 9)        # reaches level
    ag = agenda(WINNING)
    st = ag.status()                                       # +9: the line is 1 (stay ahead)
    assert ag.keeps(st, (-8, 0, claim(value=-8)))[0] is True        # 9 - 8 = 1
    assert ag.keeps(st, (-9, 0, claim(value=-9)))[0] is False       # 0 < 1


def test_status_line_text():
    ag = agenda(LOSING)
    line = ag.status_line(0, ag.status(), (1, 0, claim("d3", 1)), "tier 1 opened (x)")
    assert line.startswith("STATUS: losing (balance -9, their threat 0): the line is 'reach level' (0); tier 0's best claim d3 +1 gives -8: not kept; tier 1 opened (x)")


def _patch_tiers(monkeypatch, values):
    """Replace one run of a root agenda by a canned best claim per tier (None: no claim); record the tiers run."""
    ran = []

    def fake(self):
        k = self.ctx.tier
        ran.append(k)
        v = values[k]
        if v is not None:
            c = claim(f"m{k}", v)
            self.state.claims.append(c)
            self.best = (v, 0, c)
    monkeypatch.setattr(AG.Agenda, "_ask_once", fake)
    return ran


def test_driver_stops_at_the_first_tier_that_keeps(monkeypatch):
    AG.TIERS = True
    ran = _patch_tiers(monkeypatch, [1, 2, 10, 20])        # LOSING: -9 + v >= 0 first at tier 2
    ag = agenda(LOSING)
    ag.ask()
    assert ran == [0, 1, 2]
    assert ag.decision().what == "m2" and ag.tier_decided == 2
    assert [t["kept"] for t in ag.tier_info] == [False, False, True]
    assert any(l.startswith("tier 2 keeps the status: m2") for l in ag.lines)
    assert any("tier 1 opened" in l for l in ag.lines)


def test_driver_stops_at_tier_zero_when_it_keeps(monkeypatch):
    AG.TIERS = True
    ran = _patch_tiers(monkeypatch, [10, 20, 30, 40])
    ag = agenda(LOSING)
    ag.ask()
    assert ran == [0] and ag.decision().what == "m0"


def test_driver_runs_all_tiers_and_takes_the_best_claim(monkeypatch):
    AG.TIERS = True
    ran = _patch_tiers(monkeypatch, [1, None, 3, 2])        # never reaches level: the decision is the best over the tiers (tier 2's +3)
    ag = agenda(LOSING)
    ag.ask()
    assert ran == [0, 1, 2, 3]
    assert ag.decision().what == "m2" and ag.tier_decided == 2 and ag.decision().value == 3
    assert all(not t["kept"] for t in ag.tier_info)


def test_driver_off_runs_tier_zero_only(monkeypatch):
    AG.TIERS = False
    ran = _patch_tiers(monkeypatch, [1, 2, 3, 4])
    ag = agenda(LOSING)
    ag.ask()
    assert ran == [0] and not hasattr(ag, "tier_info")


def test_tier_is_a_setting_of_the_ctx_not_a_module_flag():
    flags = (AG.NEED_AWARE_CLOSE, AG.EVASION_CHASE, AG.NEED_PROPAGATION, AG.GOAL_JUDGE, AG.DEEP_WIN_CHASE, AG.CHASE_CLOCK)
    ags = [agenda(WINNING, tier=k) for k in range(4)]
    assert flags == (AG.NEED_AWARE_CLOSE, AG.EVASION_CHASE, AG.NEED_PROPAGATION, AG.GOAL_JUDGE, AG.DEEP_WIN_CHASE, AG.CHASE_CLOCK)      # (nothing process-wide moved)
    assert [a.ctx.tier for a in ags] == [0, 1, 2, 3]
    assert [a._t(False, 3) for a in ags] == [False, False, False, True]               # the chain items from tier 3
    assert [a._clock() for a in ags] == [AG.CHASE_CLOCK] * 3 + [4]
    assert all(a._t(True, 3) for a in ags)                                            # a module flag forces an item on at every tier
    base = ags[0].ctx.left
    assert [a.ctx.left for a in ags] == [base, base * 2, base * 4, base * 8]          # tier k's count budget = the base fuse x 2^k
    assert base == ags[0].budget * ags[0].stops["chase fuse"]["fuse"]                  # (plan 75.2 (b): the fuse is read from the chase fuse entry)


def test_tier_one_lifts_the_cut_and_tier_two_adds_the_mate_in_two_threat():
    b = chess.Board(MATE2)
    m = chess.Move.from_uci("e8h8")
    assert R.forcing_why(b, m, 100, "w", mate_threats=True) is None
    assert R.forcing_why(b, m, 100, "w", mate_threats=True, mate2=True) == "threat of mate in two"      # the proved word mate_in_two (selfmine.bind_mate2) after the move and a pass
    assert R.mate2_threat(b, m, "w") and not R.mate_threat(b, m, "w")
    other = chess.Move.from_uci("g4g3")
    assert not R.mate2_threat(b, other, "w") or R.forcing_why(b, other, 100, "w", mate_threats=True, mate2=True) is not None


def _decision(pid, set_name, **flags):
    saved = {k: getattr(AG, k) for k in flags}
    for k, v in flags.items():
        setattr(AG, k, v)
    try:
        p = next(x for x in PA.load_set(set_name) if x["id"] == pid)
        lines = []
        r = AN.analyse(p, quiet=True, memory=False, log=lines.append)
        return r, lines
    finally:
        for k, v in saved.items():
            setattr(AG, k, v)


@pytest.mark.parametrize("pid,set_name,dec", [("07Z0P", "dev_20", "Qxd5+"), ("0LRTF", "fresh_60_threats_event", "Rh8+")])
def test_tiers_off_is_identical_to_the_decisions_before_plan_59(pid, set_name, dec):
    off = dict(NEED_AWARE_CLOSE=False, EVASION_CHASE=False, NEED_PROPAGATION=False, GOAL_JUDGE=False, DEEP_WIN_CHASE=False, CHASE_CLOCK=99) if pid == "07Z0P" else {}      # plan 64.2: under the COUNT fuse 07Z0P needs the items off (its Qxd5+ is asserted under the time budget in tests/test_offbuilt64.py)
    r, lines = _decision(pid, set_name, TIERS=False, **off)
    assert r["decision"] == dec and "tiers" not in r
    assert not any(l.startswith("OBS tiers") for l in lines)
    r2, lines2 = _decision(pid, set_name, TIERS=True, **off)           # (these two keep the status at tier 0: the same decision, one tier)
    assert r2["decision"] == dec and r2["tiers"][-1]["tier"] >= 0


QCX = "r3kb1r/p1P2ppp/b3p3/8/Q3PP2/3qpN1P/1P4P1/RN2K2R b KQkq - 1 16"     # qCXKp (rand3_10): tier 0 reaches level quickly (Bb5)


def test_59_4_under_a_time_budget_a_reached_level_raises_the_line_and_widening_goes_on():
    import chess
    from catspace.incubator.analyst.discovery import agenda as AG
    from catspace.incubator.analyst.discovery.state import state_of
    side = "w" if chess.Board(QCX).turn else "b"
    ag = AG.Agenda(state_of(QCX, side), seconds=1.0)
    ag.ask()
    assert ag.status_info.get("raised") and ag.status_info["line"] == 1 and len(ag.tier_info) > 1
    assert any("the line rises to 'stay ahead'" in l for l in ag.lines)
    ag2 = AG.Agenda(state_of(QCX, side))                      # under the count budget nothing changes: no tiers, no raise
    ag2.ask()
    assert getattr(ag2, "tier_info", None) in (None, [])
