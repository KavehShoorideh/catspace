"""PLAN 68 (docs/ANALYST_PLAN.md: relevance to the status orders the root questions across the tiers; Lean `Ideas.Forcing.keepsStatus` / `relevant`).
Numbers are directional (OBS lines)."""
import itertools

from catspace.incubator.analyst.discovery import agenda as AGM
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import state_of

ESJQH = "8/1Rb3kp/p2q2p1/2p1pp2/pQ6/1bP4P/1P3PPK/8 w - - 0 35"       # rand4_10 eSJqH: status losing (balance -3, their threat 7), the line 'reach level': need 3


def _keeps(balance, value, line):
    return line <= balance + value


def test_68_1_a_claim_below_the_need_keeps_no_status_and_a_kept_claim_was_relevant():
    """Lean `irrelevant_keeps_nothing`, `relevant_of_keeps`, over a grid of balances, values and lines (a TEST may enumerate)."""
    for balance, value, line in itertools.product(range(-12, 13), range(-12, 13), (0, 1)):
        need = line - balance
        if value < need:
            assert not _keeps(balance, value, line)
        if _keeps(balance, value, line):
            assert need <= value                                        # relevant


def test_68_2_the_stake_one_questions_are_deferred_behind_the_tiers_and_asked_last():
    """eSJqH under a time budget: the seven 'win their P (worth 1)' roots (stake 1 < need 3) are deferred in every tier; the relevant questions (mate 100, save 9,
    win 9, win 3) are asked; the deferred pass asks the rest with the time left. The flag off asks them in every tier as before."""
    ag = Agenda(state_of(ESJQH, "w"), seconds=6)
    ag.ask()
    text = ag.trace()
    deferred = [l for l in text.splitlines() if "deferred: Q" in l and not l.startswith("[")]
    print("OBS 68 eSJqH tier 0 deferred:", len(deferred), "| tiers", getattr(ag, "tier_secs", None), "| decision", ag.decision().what if ag.decision() else None)
    assert len(deferred) == 7 and all("(stake 1 < the status's need 3)" in l for l in deferred)
    asked0 = [q.text for q in ag.order]                                  # tier 0's own order: no stake-1 question in it
    assert asked0 and not any("worth 1" in t for t in asked0)
    assert "DEFERRED PASS" in text or "DEFERRED:" in text
    assert ag.ctx.need == 3


def test_68_2_nothing_is_deferred_when_the_status_is_kept_or_without_tiers(monkeypatch):
    """A level position with no threat has need 0: every root question is relevant; the count budget (no tiers) sets no need at all."""
    ag = Agenda(state_of("r1bqkbnr/pppppppp/2n5/8/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 2", "w"), seconds=2)
    ag.ask()
    assert ag.ctx.need is not None and ag.ctx.need <= 0 and "deferred: Q" not in ag.trace()
    ag2 = Agenda(state_of(ESJQH, "w"))
    ag2.ask()
    assert ag2.ctx.need is None and "deferred: Q" not in ag2.trace()


def test_68_2_the_switch_off_asks_the_pawn_questions_in_tier_zero(monkeypatch):
    monkeypatch.setattr(AGM, "DEFER", False)
    ag = Agenda(state_of(ESJQH, "w"), seconds=3)
    ag.ask()
    assert any("worth 1" in q.text for q in ag.order) and "deferred: Q" not in ag.trace()
