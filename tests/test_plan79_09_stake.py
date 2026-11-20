"""PLAN 79.09 WHICH GOALS REACH THE SEARCH BY STAKE, THE CLOCK HELD, THE REASONS (SYSTEM_SPEC 7 the stake orders goals and the budget, 8 the rule for choosing a move, 4 the four ways a line ends, 11; piece: the wake;
rust/engine/decision/src/{rule,wake,threats,goals,pick,frame_rows}.rs, rust/engine/tactics/src/{ledger,lines}.rs, rust/engine/tactics/src/sheet/{questions,frames,threat_frames}.rs).
Parts A (the stake and the chain-by-chain naming), B (the clock) and C (the reasons) are unit-tested in Rust (cargo test --release --offline -p cs_decision --test wake_rule --test wake_clock --test wake_reasons --test wake_whole; -p cs_tactics --test stop_at). THIS file holds what belongs to
Python: the wall for the sources this plan touched, nothing new on by default, the old player unchanged through the binding, and the binding of the whole wake under its clock: the wall time within the clock and every claim with the reason it ended. No puzzle, no stored game, no match."""
from __future__ import annotations

import json
import re
import time
import tomllib
from pathlib import Path

import cs_engine as CE
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M

ROOT = Path(__file__).resolve().parents[1]
DECISION = ROOT / "rust" / "engine" / "decision"
TACTICS = ROOT / "rust" / "engine" / "tactics"
SHEET = TACTICS / "src" / "sheet"
NAMED = [DECISION / "src" / n for n in ("wake.rs", "rule.rs", "pick.rs", "frame_rows.rs", "threats.rs", "goals.rs")] + [SHEET / "threat_frames.rs", TACTICS / "src" / "lines.rs", TACTICS / "src" / "ledger.rs"]
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
WORTH = {"p": 475, "n": 1216, "b": 1301, "r": 1945, "q": 3583}
LINE = "k7/4n3/2p5/8/8/5B2/8/4R2K w - - 0 1"
DEVELOPED = "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7"
PIECES = ["threats", "goals", "two_threat_goals", "reasons_pick"]


def _text(fen):
    return CE.state("chess", fen, [])[0]


def test_the_wall_the_sources_of_this_plan_name_no_game_object_and_reach_for_no_oracle():
    for p in NAMED:
        text = p.read_text()
        low = text.lower()
        assert not re.search(r"\b[a-h][1-8]\b", low), (p.name, re.search(r"\b[a-h][1-8]\b", low).group(0))
        if p.name in ("wake.rs", "rule.rs", "pick.rs", "frame_rows.rs", "threat_frames.rs"):
            for w in BANNED:
                assert not re.search(rf"\b{w}\b", low), (p.name, w)
        assert not re.search(r"ladder_oracle|stockfish|cs_learn|analyse|reqwest|anthropic", low), p.name
        assert "rules/learned" not in low and "data/derived" not in low, p.name
    cargo = tomllib.loads((DECISION / "Cargo.toml").read_text())
    assert "cs_learn" not in cargo.get("dependencies", {}), "no decision-path crate depends on cs_learn"


def test_the_stake_orders_the_fixes_named_and_nothing_special_to_the_goal_unit_is_written():
    rule = (DECISION / "src" / "rule.rs").read_text()
    wake = (DECISION / "src" / "wake.rs").read_text()
    assert "pub fn named_fixes" in rule and "pub fn stake_of" in rule and "pub fn chain_top" in rule
    assert "rule::named_fixes" in wake and "chosen_goal(" not in wake and "chosen_goal(" not in rule, "the nearest-goal rule is replaced, not kept beside the chains"
    # the goal kind is the largest stake by the record's rule (a parameter), not a word of any game; the goal unit's own flag is not read by the ordering
    body = rule[rule.index("pub fn named_fixes"):rule.index("pub fn fixes_among")]
    assert "goal_unit" not in body, "the chain of the goal unit comes first only because its stake is the largest"
    # goals, threats and estimates order what is read; none of them is added to a value
    assert "never add to the judge's value" in wake.replace("\n", " ") or "never add to the judge" in wake


def test_nothing_new_is_on_by_default_and_a_clock_is_a_parameter():
    wake = (DECISION / "src" / "wake.rs").read_text()
    assert "pieces: Pieces::old_player()" in wake and "pre_share: None" in wake
    assert "pub deadline: Option<Instant>" in (DECISION / "src" / "threats.rs").read_text()
    assert "deadline: None" in (DECISION / "src" / "threats.rs").read_text(), "no clock unless the caller gives one: the readings are as they were"


def test_the_binding_with_no_piece_on_is_still_the_old_player_move_and_value():
    kw = dict(worth=WORTH, ours_first=False, valuation="odds record", threads=1, ledger_share=1.0)
    old = CE.solve_equals_batch([("chess", LINE, [])], 5.0, ledger_depth=4, **kw)[0]
    model = GL.model("chess", "chess")
    goal = G2M.goal_kind_of("chess")
    out = json.loads(model.wake_decide(_text(LINE), [], 5.0, 4, WORTH, {}, goal, [], 0, 0, None, None, None, None))
    assert out["move"] == old[0] and out["value"] == old[8] and out["depth"] == old[7]
    assert out["log"]["moves"] == [] and len(out["steps"]) == 1


def test_the_whole_wake_through_the_binding_keeps_its_clock_and_every_claim_has_the_reason_it_ended():
    model = GL.model("chess", "chess")
    goal = G2M.goal_kind_of("chess")
    for fen, secs in ((DEVELOPED, 0.3), (LINE, 0.3)):
        t0 = time.perf_counter()
        out = json.loads(model.wake_decide(_text(fen), [], secs, 64, WORTH, {}, goal, PIECES, 0, 0, None, None, ([], [], (100, 25), 20), None, None))
        wall = time.perf_counter() - t0
        assert out["move"]
        assert wall <= secs * 1.25 + 0.05, (fen, wall)
        assert out["chain"] and all(c["ended"].strip() for c in out["chain"]), "no claim without the reason it ended"
        why = out["log"]["why"]
        assert why
        if why.startswith("answers the threat"):
            assert out["log"]["their_threat"], "an answer is said only where a threat of theirs stands"
        assert {s["step"] for s in out["steps"]} >= {3, 4, 5, 6}
        # what the steps before the search did not ask is said in their notes or the log, never hidden
        step3 = [s["note"] for s in out["steps"] if s["step"] == 3]
        assert step3 and all(isinstance(n, str) for n in step3)
