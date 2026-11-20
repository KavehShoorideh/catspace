"""PLAN 79.10 THEIR GOALS AFTER OUR MOVE, THE NEXT GOAL OF A CHAIN, NAMES STRAIGHT TO THE SEARCH, THE EQUAL CLOCK (SYSTEM_SPEC 7 one way of asking for every goal and the stake, 8 the rule for choosing a move, 4, 11; piece: the wake;
rust/engine/decision/src/{after,rule,wake,log}.rs, catspace/incubator/analyst/discovery/equal_wall.py).
Parts A (their goals after each move), B (the next goal of a chain) and C (the names direct) are unit-tested in Rust (cargo test --release --offline -p cs_decision --test wake_after --test wake_rule --test wake_direct --test wake_whole). THIS file holds what belongs to Python: the wall for the
sources, nothing new on by default, the binding of the whole wake with the new pieces (the fact about our move in the log, the value untouched, the clock held), and the equal-wall helper of part D. No puzzle, no stored game, no match."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import cs_engine as CE
from catspace.incubator.analyst.discovery import equal_wall as EW
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "rust" / "engine" / "decision"
NAMED = [DECISION / "src" / n for n in ("after.rs", "wake.rs", "rule.rs", "log.rs")] + [ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "equal_wall.py"]
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
WORTH = {"p": 475, "n": 1216, "b": 1301, "r": 1945, "q": 3583}
LINE_BLACK = "k7/4n3/2p5/8/8/5B2/8/4R2K b - - 0 1"
DEVELOPED = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7"
OLD = ["carried", "frames", "threats", "goals", "two_threat_goals", "reasons_pick"]
NEW = OLD + ["after_move", "direct_names"]


def _text(fen):
    return CE.state("chess", fen, [])[0]


def _wake(model, goal, fen, secs, pieces, depth=64):
    return json.loads(model.wake_decide(_text(fen), [], secs, depth, WORTH, {}, goal, pieces, 0, 0, None, None, ([], [], (100, 25), 20), None, None))


def test_the_wall_the_sources_of_this_plan_name_no_game_object():
    for p in NAMED:
        low = p.read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", low), (p.name, re.search(r"\b[a-h][1-8]\b", low).group(0))
        for w in BANNED:
            assert not re.search(rf"\b{w}\b", low), (p.name, w)
        assert not re.search(r"ladder_oracle|stockfish|cs_learn|analyse|reqwest|anthropic", low), p.name


def test_nothing_new_is_on_by_default_and_the_pieces_are_named_in_the_binding():
    wake = (DECISION / "src" / "wake.rs").read_text()
    assert "pub after_move: bool" in wake and "pub direct_names: bool" in wake
    assert "pub fn old_player() -> Pieces { Pieces::default() }" in wake
    binding = (ROOT / "rust" / "engine" / "py" / "src" / "base_graph_py.rs").read_text()
    assert '"after_move" => pc.after_move = true' in binding and '"direct_names" => pc.direct_names = true' in binding
    assert "keep_read_by" in wake and "if p.direct_names" in wake, "the reading of the named lines stays, after the search"


def test_the_binding_with_the_new_pieces_logs_the_fact_about_our_move_and_the_value_is_untouched():
    model = GL.model("chess", "chess")
    goal = G2M.goal_kind_of("chess")
    old = _wake(model, goal, LINE_BLACK, 5.0, OLD, depth=4)
    new = _wake(model, goal, LINE_BLACK, 5.0, NEW, depth=4)
    assert (new["value"], new["depth"]) == (old["value"], old["depth"]), "their goals after our move, the next goal and the direct names add nothing to a value"
    moves = {m["move"]: m for m in new["log"]["moves"]}
    assert all(m["their_after"] is None for m in old["log"]["moves"])
    knight = moves["e7f5"]["their_after"]
    assert knight and "took out of the way" in knight["said"] and knight["names"]
    assert new["log"]["after_not_asked"] == []
    pieces = [s["piece"] for s in new["steps"]]
    assert any("after each move" in p for p in pieces) and any("given to the search directly" in p for p in pieces) and any("after the search" in p for p in pieces)
    assert not any("their lines read" in p for p in pieces), "no line of a named move is read before the search"


def test_the_whole_wake_with_the_new_pieces_keeps_its_clock_and_every_claim_has_the_reason_it_ended():
    model = GL.model("chess", "chess")
    goal = G2M.goal_kind_of("chess")
    for fen, secs in ((DEVELOPED, 0.3), (LINE_BLACK, 0.3)):
        t0 = time.perf_counter()
        out = _wake(model, goal, fen, secs, NEW)
        wall = time.perf_counter() - t0
        assert out["move"]
        assert wall <= secs * 1.25 + 0.05, (fen, wall)
        assert out["chain"] and all(c["ended"].strip() for c in out["chain"]), "no claim without the reason it ended"
        assert out["log"]["why"]


def test_the_binding_with_no_piece_on_is_still_the_old_player():
    model = GL.model("chess", "chess")
    goal = G2M.goal_kind_of("chess")
    kw = dict(worth=WORTH, ours_first=False, valuation="odds record", threads=1, ledger_share=1.0)
    old = CE.solve_equals_batch([("chess", LINE_BLACK, [])], 5.0, ledger_depth=4, **kw)[0]
    out = _wake(model, goal, LINE_BLACK, 5.0, [], depth=4)
    assert out["move"] == old[0] and out["value"] == old[8] and out["log"]["moves"] == []


def test_the_equal_wall_helper_finds_the_clock_whose_wall_matches_and_says_when_it_cannot():
    # an old player that spends 0.4 of the clock: to spend a wall of 1.0 it needs a clock of 2.5
    r = EW.at_wall(lambda c: {"wall": 0.4 * c, "move": "x"}, 1.0)
    assert r["reached"] and abs(r["wall"] - 1.0) <= 0.15 and abs(r["clock"] - 2.5) < 0.5, r
    # one that cannot spend more than 0.3 s however much it is given: not reached, said
    r2 = EW.at_wall(lambda c: {"wall": min(0.3, 0.4 * c), "move": "x"}, 1.0)
    assert not r2["reached"] and r2["wall"] <= 0.3 + 1e-9 and len(r2["tried"]) >= 2
    # the clock stays inside its bounds
    assert all(EW.FLOOR * 1.0 - 1e-9 <= c <= EW.CEILING * 1.0 + 1e-9 for c, _ in r2["tried"])
    # a wall already near the target needs one call
    r3 = EW.at_wall(lambda c: {"wall": 1.05, "move": "x"}, 1.0)
    assert r3["reached"] and len(r3["tried"]) == 1
