"""PLAN 61 stage C: the cost of a route (C1), the cost judge (C2), the forcing resources (C3), the cycle terminal (C4).

The cost and the judge decide only WHERE TIME GOES (a cut is a GAP); the cycle claim is proved only for a position that repeats one of the line, and only when we are behind."""
import time

import chess
import pytest

import catspace.incubator.analyst  # noqa: F401
from catspace.incubator.analyst.discovery import agenda as AG
from catspace.incubator.analyst.discovery import base_vocabulary as BV
from catspace.incubator.analyst.discovery import question_prior as QP
from catspace.incubator.analyst.discovery import regress as R
from catspace.incubator.analyst.discovery.state import state_of

BACK_RANK = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"          # Ra8# : the route is a mate, no parry at all
BARE_KING = "7k/8/8/8/8/8/8/R5K1 w - - 0 1"              # Ra8+ : two parries (Kg7, Kh7)
PERPETUAL = "1Q6/2R2pp1/5k1p/pB6/P1pb4/6PP/5r2/7K b - - 0 41"     # lichess w91Hv: black is a rook for a queen down, Rf1+ Kg2 Rf2+ Kh1 is a perpetual


@pytest.fixture(autouse=True)
def _setup():
    BV.install_terms()


def test_cost_of_a_mating_route_is_one():
    b = chess.Board(BACK_RANK)
    cost, kids = R.route_cost(b, b.parse_san("Ra8#"), 1, "w")
    assert (cost, kids) == (1, [])


def test_cost_is_one_plus_the_children_by_hand():
    b = chess.Board(BARE_KING)
    cost, kids = R.route_cost(b, b.parse_san("Ra8+"), 1, "w")
    assert sorted(k for k, _c, _kind in kids) == ["Kg7", "Kh7"]          # the parries are the legal replies of a check
    assert [kind for _k, _c, kind in kids] == ["quiet", "quiet"]         # (plan 76.2 (b): what each parry is, read by the Game surface)
    assert cost == 1 + sum(c for _, c, _kind in kids)
    for san, c, _kind in kids:                                           # each child's cost is node_cost on its own board
        d = b.copy(); d.push_san("Ra8+"); d.push_san(san)
        assert c == R.node_cost(d, 1, "w")[0]


def test_node_cost_is_zero_when_we_mate_or_have_nothing_forcing():
    assert R.node_cost(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"), 1, "w")[0] == 0 or R.node_cost(chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"), 1, "w")[1].endswith("#")
    assert R.node_cost(chess.Board("4k3/8/8/8/8/8/8/4K3 w - - 0 1"), 1, "w") == (0, None)


def _agenda_with_rate(spn, left):
    ag = AG.Agenda(state_of(BARE_KING, "w"), seconds=100)
    ctx = ag.ctx
    ctx.judge_n, ctx.judge_t = 100, 100 * spn
    ctx.deadline = time.monotonic() + left
    ag._route_move, ag._reply_san = chess.Move.from_uci("a1a8"), "Kg7"
    return ag


def test_judge_continues_when_cost_times_rate_fits(monkeypatch):
    monkeypatch.setattr(AG, "COST_JUDGE", True)
    monkeypatch.setattr(R, "node_cost", lambda *a, **k: (10, "Ra7+"))
    ag = _agenda_with_rate(0.01, 5.0)                                     # 10 x 0.01 = 0.1 s <= 5 s
    assert ag._judge_node("7k/8/8/8/8/8/8/R5K1 w - - 0 1") is False
    assert ag.ctx.judge_cuts == 0


def test_judge_cuts_when_it_does_not_fit_and_says_so(monkeypatch):
    monkeypatch.setattr(AG, "COST_JUDGE", True)
    monkeypatch.setattr(R, "node_cost", lambda *a, **k: (500, "Ra7+"))
    ag = _agenda_with_rate(0.01, 1.0)                                     # 500 x 0.01 = 5 s > 1 s
    assert ag._judge_node("7k/8/8/8/8/8/8/R5K1 w - - 0 1") is True
    assert ag.ctx.judged and ag.ctx.judge_cuts == 1
    assert "GAP" in ag.ctx.c_lines[-1] and "5.0 s needed" in ag.ctx.c_lines[-1]
    assert ag._cut("Ra8+", "Kg7").startswith("out of time: Ra8+ cut by the judge")


def test_judge_reads_on_without_a_rate_and_is_inert_without_a_time_budget(monkeypatch):
    monkeypatch.setattr(AG, "COST_JUDGE", True)
    monkeypatch.setattr(R, "node_cost", lambda *a, **k: (10 ** 6, None))
    ag = _agenda_with_rate(0.01, 0.0001)
    ag.ctx.judge_n = 0                                                     # nothing timed yet
    assert ag._judge_node("7k/8/8/8/8/8/8/R5K1 w - - 0 1") is False
    ag2 = AG.Agenda(state_of(BARE_KING, "w"))                              # the count budget: the fuse stays
    assert ag2._judge_node("7k/8/8/8/8/8/8/R5K1 w - - 0 1") is False


def test_root_route_judge_prints_the_cost_and_cuts_to_a_gap(monkeypatch):
    monkeypatch.setattr(AG, "COST_JUDGE", True)
    ag = _agenda_with_rate(10.0, 1.0)                                      # 10 s per node: nothing of cost >= 1 fits in 1 s
    ag._chase_q = None
    res, tok = ag._judge_begin(chess.Move.from_uci("a1a8"), False, False, None)
    assert res is not None and res["killed"] is None and res["value"] == 0 and res["gap"].startswith("out of time")
    assert any("judge: Ra8+ cut (cost" in x and "parries" in x and "-> GAP" in x for x in ag.lines)


def test_forcing_resources_are_logged_counts():
    r = QP.forcing_resources(chess.Board("1k1r3r/1p1b4/1P4pp/p3p3/2Bq4/P1Q5/5RPP/R5K1 w - - 4 29"), "w")
    assert set(r) == {"forcing_moves", "bearing", "flights", "blockers"} and r["forcing_moves"] > 0 and r["blockers"] == 1


def _repeat_ctx(monkeypatch, cycle=True):
    monkeypatch.setattr(AG, "CYCLE", cycle)
    ag = AG.Agenda(state_of(PERPETUAL, "b"))
    return ag


def test_cycle_claim_is_a_draw_worth_minus_the_balance_when_behind(monkeypatch):
    ag = _repeat_ctx(monkeypatch)
    ag._route_move, ag._reply_san = chess.Move.from_uci("f2f1"), "Kg2"
    out = ag._inner(PERPETUAL, 1)                                           # the root position again: a repeat of the line's first
    assert out["claim"].kind == "draw" and out["claim"].lemma == "repetition" and out["claim"].proved
    assert out["claim"].line.startswith(AG.CYCLE_DETAIL + "; the position after")                         # plan 75.4 (2): the claim says what it is (a derived lemma, not the rule)
    assert out["value"] == 7 and out["claim"].value == -(-7)
    assert any(x.startswith("    closed: repetition (a forced cycle: the threefold rule is reachable; the position after") and "repeats the root position" in x for x in ag.ctx.c_lines)


def test_cycle_off_is_no_claim_as_before(monkeypatch):
    ag = _repeat_ctx(monkeypatch, cycle=False)
    out = ag._inner(PERPETUAL, 1)
    assert out["value"] == 0
    assert out["claim"] is None or out["claim"].kind != "draw"          # 72.10 (2026-10-23): an inner agenda now returns a KEPT save at 0 (Rf1+ holds the balance) as a claim;
    #                                                                     the pin here is the cycle word's: with CYCLE off no repetition / draw claim appears, and the value is 0


def test_cycle_gives_nothing_when_ahead(monkeypatch):
    monkeypatch.setattr(AG, "CYCLE", True)
    fen = "1Q6/2R2pp1/5k1p/pB6/P1pb4/6PP/5r2/7K w - - 0 41"                 # same board, white to move: white is up 7
    ag = AG.Agenda(state_of(fen, "w"))
    assert ag._inner(fen, 1)["claim"] is None


def test_game_history_is_a_repetition_too(monkeypatch):
    monkeypatch.setattr(AG, "CYCLE", True)
    fen = "1Q6/2R2pp1/5k1p/pB6/P1pb4/6PP/5r2/7K b - - 0 41"
    other = "1Q6/2R2pp1/5k1p/pB6/P1pb4/6PP/6r1/7K b - - 0 41"
    ag = AG.Agenda(state_of(other, "b"))
    ag.ctx.history.add(AG._pos_key(fen))
    out = ag._inner(fen, 1)
    assert out["claim"] is not None and out["claim"].kind == "draw"
    assert any("repeats the game history" in x for x in ag.ctx.c_lines)


def test_flag_defaults():
    assert AG.COST_JUDGE is True                       # plan 64.3: ON (the 90 at 10 s: no RIGHT lost once the chain stops are GAPs and the routes are priced by their measured seconds)
    assert AG.CYCLE is True                            # graded on the 90: no decision changed
    from catspace.incubator.analyst.discovery import analyse as AN
    assert "COST_JUDGE" in AN._FLAGS and "CYCLE" in AN._FLAGS
