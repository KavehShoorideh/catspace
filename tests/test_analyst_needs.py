"""needs.py: means-ends planning over needs -- the pinned knight, the second attacker, the covered square, the lure,
the race, and the honest give-up."""
import chess
import pytest

from catspace.incubator.analyst.memo import Memo
from catspace.incubator.analyst.needs import Planner, Need, plan_win, ideas_for
from catspace.incubator.analyst.reach import flip_conditions
from catspace.incubator.analyst.engine import Engine

RUY = "r1bqkbnr/ppp2ppp/2np4/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"      # Bb5 pins Nc6 (defended by b7 only)


@pytest.fixture(scope="module")
def eng():
    with Engine(depth=12) as e:
        yield e


def test_the_flip_cap_is_the_largest_attacker_that_flips():
    b = chess.Board(RUY)
    f = next(f for f in flip_conditions(b, chess.C6, chess.WHITE) if f.kind == "add_attacker")
    assert f.max_value == 9.0                                     # any second attacker wins the exchange (was wrongly 1: pawns only)


def test_the_pinned_knight_needs_a_second_attacker_which_needs_d4_which_needs_the_lure(eng):
    b = chess.Board(RUY)
    line, pl = plan_win(b, chess.WHITE, chess.C6, Memo(), eng)
    assert line.best is not None and "add the knight on d4" in line.best.why
    assert line.sans() == ["d4", "Nxd4", "Nxc6"]
    reach = line.best.subs[0]; safe = reach.best.subs[0]; deflect = safe.best.subs[0]
    assert reach.need.kind == "reach" and reach.need.square == chess.D4
    assert safe.need.kind == "safe" and deflect.need.kind == "deflect" and deflect.need.against == chess.E5
    assert deflect.best.reply == "exd4" and "lure on d4" in deflect.best.why
    assert line.best.gain == 1.0 and line.cost == 0.5                    # a pawn by the exchange; the lure may be declined
    assert line.race == "met" and line.answer.startswith("a6 in 1") and line.free_tempi == 1   # the bishop is kicked first
    assert line.status == "refuted" and "given up" in line.verification   # and the goal search agrees: not a forced win
    text = pl.tree_text(line)
    assert text[0].startswith("Win the knight on c6: Nxc6 -- add the knight on d4 as an attacker") and "they are in time" in text[0]
    assert any("Deflect the pawn on e5 from d4: d4 (then exd4)" in t for t in text)


def test_two_coverers_and_poisoned_lures_are_given_up_honestly():
    b = chess.Board("r1bqkbnr/pp3ppp/2np4/1Bp1p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5")   # c5 and e5 both cover d4
    line, pl = plan_win(b, chess.WHITE, chess.C6, Memo(), None, verify=False)
    assert line.best is None and line.reason == "no way that pays"
    assert not any("h4" in t for t in pl.tree_text(line))                  # h4 (Qxh4?? Rxh4) is poisoned: never offered as a lure


def test_the_idea_loop_proposes_the_pin_with_its_verdict():
    b = chess.Board("r1bqkbnr/ppp2ppp/2np4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")   # the bishop still on f1
    ideas = ideas_for(b, chess.WHITE, Memo())
    mv, reason, line = next((i for i in ideas if b.san(i[0]) == "Bb5"), (None, None, None))
    assert mv is not None and reason == "it is undeveloped" and line.need.square == chess.C6
    assert line.race == "met" and "a6" in line.answer
