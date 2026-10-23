"""PLAN 70.1 (the audit of 2026-10-22, 2TUzC): a SAVE question is CHASED with KEEP semantics. Numbers are directional (OBS lines)."""
import chess

from catspace.incubator.analyst.discovery import agenda as AGM
from catspace.incubator.analyst.discovery import binder as BD
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.agenda import Agenda
from catspace.incubator.analyst.discovery.state import State, state_of

TUZC = "r4rk1/1p1b3p/p3p1p1/3pb3/P1P5/B6P/5qP1/R2QRB1K w - - 2 20"            # 2TUzC (rand8_10): Rxe5 is Stockfish's move
AFTER_RF5 = "r5k1/1p1b3p/p3p1p1/3pRr2/P1P5/B6P/5qP1/R2Q1B1K w - - 1 21"       # after Rxe5 Rf5: Rxf5! keeps the balance (Qxf1+ Kh2 Rxf1 wins the queen)


def test_70_1_a_save_question_chased_for_keeping_has_target_zero_and_the_king_evasions_keep_theirs():
    ag = Agenda(state_of(TUZC, "w"))
    q = R.new_question("save", 5, dict(target=chess.E5), text="save our R on e5")
    ag._chase_q = q
    assert AGM.CHASE_SAVES and ag._keep_goal(q) and ag._target() == 0
    ag._chase_q = R.new_question("save", 100, dict(checkers=[chess.F2]), text="save our king")
    assert not ag._keep_goal(ag._chase_q) and ag._target() >= 1


def test_70_1_the_static_leaf_inner_read_claims_the_save_by_the_chase_where_the_keep_family_fails():
    """The inner agenda after Rxe5 Rf5 as the chase opens it (a mate-only need, keep +3): the one-ply keep family rejects Rxf5 (Qxf1+ looks like a lost bishop); the chase reads
    every reply and CLAIMS save Rxf5 -- so their threat 'Rxe5 wins 4' is parried and Rxe5 is no longer refuted at the root."""
    root = Agenda(state_of(TUZC, "w"), seconds=15)
    their = root._their_agenda(chess.Board(AFTER_RF5.replace(" w ", " b ")))
    assert any(c.what == "Rxe5" for c in their.state.claims)
    inner = Agenda(State(fen=AFTER_RF5, us="w", pieces=[]), side_depth=0, chase_depth=1, ctx=root.ctx, min_stake=BD.MATE_VALUE, their=their, keep=3, since=1)
    inner.ask()
    text = inner.trace()
    print("OBS 70.1 inner save:", [l for l in text.splitlines() if "CLAIM save" in l][:1])
    assert any(c.kind == "save" and c.what == "Rxf5" for c in inner.state.claims)
    assert all(q.status == "verified" for q, _cl in inner.threat_qs)


def test_70_1_off_the_keep_family_alone_kills_the_save(monkeypatch):
    monkeypatch.setattr(AGM, "CHASE_SAVES", False)
    root = Agenda(state_of(TUZC, "w"), seconds=15)
    their = root._their_agenda(chess.Board(AFTER_RF5.replace(" w ", " b ")))
    inner = Agenda(State(fen=AFTER_RF5, us="w", pieces=[]), side_depth=0, chase_depth=1, ctx=root.ctx, min_stake=BD.MATE_VALUE, their=their, keep=3, since=1)
    inner.ask()
    assert any(q.status == "killed" for q, _cl in inner.threat_qs) and not any(c.kind == "save" for c in inner.state.claims)
