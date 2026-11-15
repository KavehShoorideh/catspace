"""Plan 78.46: THE FORCING READ (rust/engine/tactics/src/sheet/forcing.rs, `SheetModel.forcing`): a threat move, every reply to it, the claims that stand after each reply; the value of the move is the
least over the replies; CONFIRMED / NOT SETTLED / NO CLAIM. Small fixed positions written here; the soundness measure's judge, the puzzle measure's second column, the source scan (the layer names no game
object). The measures over real games and puzzles are forcing_soundness_measure.py and puzzle_sheet_measure.py."""
import re
from pathlib import Path

import cs_engine as CE
import pytest

from catspace.incubator.analyst.discovery import forcing_soundness_measure as FS
from catspace.incubator.analyst.discovery import puzzle_sheet_measure as PM
from catspace.incubator.analyst.discovery import question_sheet as Q
from catspace.incubator.analyst.discovery import sheet_rust as SR

WORTH = {"k": 0, "q": 14, "r": 8, "b": 5, "n": 4, "p": 2}
GOAL = 1e9
ROOT = Path(__file__).resolve().parents[1]
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"

KNIGHT_FORK = "r3k3/8/8/3N4/8/8/8/4K3 w - - 0 1"              # Nc7 gives the sign and strikes the rook: no reply answers both
PAWN_FORK = "4k3/8/8/3n1n2/8/4P3/8/6K1 w - - 0 1"             # the pawn steps up and strikes both knights
STEP_AWAY = "4k3/8/8/3n4/8/8/8/R3K3 w - - 0 1"                # the rook strikes the knight; the knight steps away
GUARD = "4k3/8/2n5/1B2r3/8/8/4R3/4K3 w - - 0 1"               # the bishop takes the unit that guards the rook (with the sign)
GOAL_IN_ONE = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"
GOAL_IN_TWO = "4k3/8/8/8/8/8/1R6/R3K3 w - - 0 1"              # a quiet move that makes the goal unanswerable next move
REPLY_WINS_MORE = "4k3/8/4p3/3p4/8/8/8/3QK3 w - - 0 1"        # the taking of the guarded pawn loses the taker


@pytest.fixture(scope="module")
def model():
    env = Q.Env("chess")
    return SR.model(env, WORTH)


def _by_move(read: dict) -> dict:
    return {t["move"]: t for t in read["threats"]}


# ------------------------------------------------------------------------------------------------------------------------------ the read

def test_a_double_claim_no_reply_answers_is_confirmed_and_every_reply_is_read(model):
    for fen, move in ((KNIGHT_FORK, "d5c7"), (PAWN_FORK, "e3e4")):
        t = _by_move(SR.forcing(model, fen))[move]
        assert t["verdict"] == "CONFIRMED" and t["value"] > 0
        legal_replies = CE.state("chess", fen, [move])[2]
        assert sorted(r["reply"] for r in t["replies"]) == sorted(legal_replies)          # EVERY legal reply, none left out
        assert all(r["value"] > 0 and not r["open"] for r in t["replies"])
        assert len(t["claims"]) >= (1 if fen == KNIGHT_FORK else 2)                          # the claims the move makes: two units; for the sign (no pass) what stands is read on the replies
    fork = _by_move(SR.forcing(model, KNIGHT_FORK))["d5c7"]
    assert fork["value"] == 8.0 and all(r["stands"]["target"] == "r@a8" and r["stands"]["value"] == 8.0 for r in fork["replies"])


def test_a_single_claim_the_target_steps_away_from_is_no_claim(model):
    read = SR.forcing(model, STEP_AWAY)
    t = _by_move(read)["a1a5"]
    assert t["verdict"] == "NO CLAIM" and t["value"] <= 0
    assert t["replies"][-1]["reply"].startswith("d5") and t["replies"][-1]["value"] <= 0 and not t["replies"][-1]["open"]      # the reply that closed it: the knight moves
    assert read["confirmed"] == 0


def test_the_taking_of_a_guard_is_confirmed_by_what_it_wins(model):
    t = _by_move(SR.forcing(model, GUARD))["b5c6"]
    assert t["verdict"] == "CONFIRMED" and t["value"] >= 4.0 and t["took_worth"] == 4.0
    assert any(c["kind"] == "takes" and c["target"].startswith("n@c6") for c in t["claims"])
    assert {r["reply"] for r in t["replies"]} == set(CE.state("chess", GUARD, ["b5c6"])[2])


def test_the_goal_forced_in_one_and_in_two_is_confirmed_above_every_worth(model):
    one = _by_move(SR.forcing(model, GOAL_IN_ONE))
    assert one["a1a8"]["verdict"] == "CONFIRMED" and one["a1a8"]["value"] >= GOAL and one["a1a8"]["replies"] == []        # the move ends the game itself
    assert [m for m, t in one.items() if t["verdict"] == "CONFIRMED"] == ["a1a8"]
    two = _by_move(SR.forcing(model, GOAL_IN_TWO))
    for m in ("a1a7", "b2b7"):                                                                                  # a quiet move: every reply is met by the goal in one
        t = two[m]
        assert t["verdict"] == "CONFIRMED" and t["value"] >= GOAL and t["replies"] and all(r["stands"]["kind"] == "goal" for r in t["replies"])
        assert t["replies"] and sorted(r["reply"] for r in t["replies"]) == sorted(CE.state("chess", GOAL_IN_TWO, [m])[2])
    assert two["b2b8"]["verdict"] != "CONFIRMED"                                                               # the sign without the follow-up is no claim


def test_a_budget_that_cuts_is_not_a_confirmation_and_not_a_refutation(model):
    full = SR.forcing(model, KNIGHT_FORK)
    assert full["confirmed"] == 1 and not full["cut"]
    cut = SR.forcing(model, KNIGHT_FORK, fuse=3)
    assert cut["cut"] and cut["confirmed"] == 0 and cut["no_claim"] == 0                    # nothing was confirmed and nothing was called empty: the rest is unread or not settled
    assert cut["unread"] + cut["unsettled"] == cut["found"]
    assert cut["nodes"] <= cut["fuse"] + 1


def test_a_threat_move_whose_reply_wins_more_than_the_claim_is_no_claim(model):
    t = _by_move(SR.forcing(model, REPLY_WINS_MORE))["d1d5"]
    assert t["verdict"] == "NO CLAIM" and t["value"] < 0
    last = t["replies"][-1]
    assert last["reply"] == "e6d5" and last["took_worth"] == 14.0 and last["value"] < 0         # the reply took the queen for the pawn


def test_a_position_that_is_over_is_refused_and_the_answer_is_plain_data(model):
    with pytest.raises(ValueError):
        SR.forcing(model, "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1")
    read = SR.forcing(model, KNIGHT_FORK)
    assert {"threats", "legal", "found", "confirmed", "unsettled", "no_claim", "unread", "nodes", "fuse", "cut", "level"} <= set(read)
    t = read["threats"][0]
    assert {"move", "verdict", "value", "took", "took_worth", "claims", "replies", "level", "plies"} <= set(t)
    assert {"reply", "took", "took_worth", "stands", "deeper", "line", "value", "open", "ends"} <= set(t["replies"][0])


def test_the_sheet_and_the_move_question_are_not_changed_by_the_read(model):
    env = Q.Env("chess")
    params = Q.Params(compare_old=False, clock="present", cost_model=(0.002, 0.01, 0.03, 0.06, 0.1))             # a stated cost: the order is a function of the position
    before = SR.sheet(model, env, KNIGHT_FORK, params, max_steps=10)
    SR.forcing(model, KNIGHT_FORK)
    after = SR.sheet(model, env, KNIGHT_FORK, params, max_steps=10)
    for q in (before, after):
        q.pop("seconds", None)
        for x in q["questions"]:
            x.pop("seconds", None)
    assert [x["key"] for x in before["questions"]] == [x["key"] for x in after["questions"]]


# ------------------------------------------------------------------------------------------------------------------------------ the soundness measure's judge

def test_a_confirmed_claim_holds_against_the_ledger_search(model):
    for fen, move in ((KNIGHT_FORK, "d5c7"), (PAWN_FORK, "e3e4"), (GUARD, "b5c6")):
        t = _by_move(SR.forcing(model, fen))[move]
        chk = SR.forcing_check(model, fen, move, FS.needed_depth(t), 5_000_000)
        assert not chk["cut"]
        assert FS.judge_claim(t, chk, chk["win"])["status"] == "holds"


def test_a_claim_the_search_does_not_bear_out_is_seen_as_a_contradiction(model):
    """NEGATIVE CONTROL of the measure: the read's claim made too large (a reply left out would do it) is a contradiction by the judge."""
    t = dict(_by_move(SR.forcing(model, KNIGHT_FORK))["d5c7"])
    chk = SR.forcing_check(model, KNIGHT_FORK, "d5c7", FS.needed_depth(t), 5_000_000)
    t["value"] = t["value"] + 20.0
    assert FS.judge_claim(t, chk, chk["win"])["status"] == "contradicted"
    cut = dict(chk, cut=True)
    assert FS.judge_claim(t, cut, chk["win"])["status"] == "unchecked"                    # a search that is cut is no check


# ------------------------------------------------------------------------------------------------------------------------------ the puzzle measure's second column

def _read(verdicts):
    threats = []
    for move, verdict, value, replies in verdicts:
        threats.append({"move": move, "verdict": verdict, "value": value, "level": 0, "claims": [{"target": "n@c6", "kind": "takes", "value": 4.0}],
                        "replies": [{"reply": r, "value": v, "open": o, "line": [r], "took": []} for r, v, o in replies]})
    return {"threats": threats, "found": len(threats), "confirmed": sum(t["verdict"] == "CONFIRMED" for t in threats), "unsettled": 0, "no_claim": 0, "unread": 0, "nodes": 10, "cut": False, "level": 0}


def test_sees2_picks2_and_the_class_f_from_forcing_data():
    seen_none = {"sees": False, "pick": "z9z9", "picks": False}
    view = PM.forcing_view(_read([("a1a2", "CONFIRMED", 8.0, []), ("b1b2", "CONFIRMED", 4.0, []), ("c1c2", "NO CLAIM", -2.0, [("x", 3.0, False), ("e8e7", -2.0, False)])]), "b1b2")
    s2 = PM.seeing2(seen_none, view, "b1b2")
    assert s2["sees2"] and s2["by_forcing"] and s2["pick2"] == "a1a2" and not s2["picks2"]             # the highest value is chosen: not the answer
    assert PM.seeing2(seen_none, view, "a1a2")["picks2"]
    view2 = PM.forcing_view(_read([("c1c2", "NO CLAIM", -2.0, [("x", 3.0, False), ("e8e7", -2.0, False)])]), "c1c2")
    assert not PM.seeing2(seen_none, view2, "c1c2")["sees2"]
    assert PM.seeing2(seen_none, view2, "c1c2")["pick2"] == "z9z9"                                    # nothing confirmed: the sheet's own pick
    assert view2["answer"]["held_by"] == "e8e7" and PM.forcing_class(view2, "R", False) == "F"         # read and not confirmed: the reply that held it
    assert PM.forcing_class(None, "R", False) == "R" and PM.forcing_class(view, "R", True) == "SEEN2"
    assert PM.seeing2({"sees": True, "pick": "q", "picks": False}, None, "q")["sees2"]                # the sheet's own SEES is kept


# ------------------------------------------------------------------------------------------------------------------------------ the layer stays game-free

def test_the_new_layer_source_and_wrappers_name_no_game_object():
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', (ROOT / "rust" / "engine" / "tactics" / "src" / "sheet" / "forcing.rs").read_text())
    hit = re.search(CHESS_WORDS, text, flags=re.I)
    assert hit is None, hit.group(0) if hit else None
    for name in ("sheet_rust.py", "forcing_soundness_measure.py"):
        src = (ROOT / "catspace" / "incubator" / "analyst" / "discovery" / name).read_text()
        assert not re.search(r"\b(king|queen|rook|bishop|knight|pawn)\b", src, flags=re.I)
