"""PLAN 69 (docs/ANALYST_PLAN.md: learned parts order, proofs bound). What is built: the pool driver (OFF by default: the agent that sees the board and the claims is not built; the
linear prior is a stand-in behind `--pool`), the status features of a question, the ITEM log the agent will train on. Numbers are directional (OBS lines)."""
import json
import os

from catspace.incubator.analyst.discovery import agenda as AGM
from catspace.incubator.analyst.discovery import question_prior as QP
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import state_of

ESJQH = "8/1Rb3kp/p2q2p1/2p1pp2/pQ6/1bP4P/1P3PPK/8 w - - 0 35"


def test_69_1_the_pool_is_on_by_default_and_the_tiers_are_the_fallback(monkeypatch):
    assert AGM.POOL is True
    monkeypatch.setattr(AGM, "POOL", False)
    ag = Agenda(state_of(ESJQH, "w"), seconds=4)
    ag.ask()
    assert "POOL:" not in ag.trace() and ag.tier_info and ag.tier_info[0]["tier"] == 0           # the tiers (the fallback) run with the pool off
    from catspace.incubator.analyst.discovery import graph_policy as GP
    if QP.load() is None and GP.load() is None:
        return
    monkeypatch.setattr(AGM, "POOL", True)
    ag2 = Agenda(state_of(ESJQH, "w"), seconds=4)
    ag2.ask()
    t = ag2.trace()
    print("OBS 69.1 pool:", [l for l in t.splitlines() if l.startswith("POOL")][:3])
    assert "POOL: one pool of questions, every word available, no tier" in t and ag2.tier_info[0]["tier"] == "pool" and ag2.ctx.left > AGM.NO_FUSE // 2      # (no count fuse: what was spent came off NO_FUSE)


def test_69_1_the_status_is_a_feature_of_a_question():
    from catspace.incubator.analyst.discovery import regress as R
    import chess
    q = R.new_question("win", 3, dict(target=chess.B3), text="win their B on b3")
    b = chess.Board(ESJQH)
    f = QP.features(q, b, "w", 0, dict(line=0, balance=-3, threat=7))
    assert f["need"] == 3 and f["stake_minus_need"] == 0 and f["stake_reaches_need"] == 1 and f["their_threat"] == 7
    assert "need" not in QP.features(q, b, "w", 0)


def test_69_3_the_item_log_records_every_root_route_with_its_words_price_order_and_outcome():
    ag = Agenda(state_of(ESJQH, "w"), seconds=3)
    ag.ask()
    rows = ag.ctx.item_rows
    print("OBS 69.3 item rows:", len(rows), "outcomes", sorted({r["outcome"] for r in rows}), "example", rows[0] if rows else None)
    assert rows and all({"q", "stake", "route", "why", "order", "check", "capture", "outcome", "need"} <= set(r) for r in rows)
    assert any(r["check"] for r in rows) and any(r["outcome"] == "killed" for r in rows)
