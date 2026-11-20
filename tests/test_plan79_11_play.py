"""PLAN 79.11 THE WHOLE IN PLAY, WITH THE LOG OF EVERY DECISION (SYSTEM_SPEC 8 the whole is judged, 7 the learned last layer needs the logged games, 4, 11; piece: the wake).
Part A (the tilt of a named move) is unit-tested in Rust (cargo test --release --offline -p cs_tactics --test named_fair). THIS file holds what belongs to Python: the wall for the sources (no game object named, no oracle called or loaded), the mover that only calls the binding and writes the log, the format of the log,
the carry kept by the binding's slot, the equal-wall scale, and the count of what differed from the search. No match is played here."""
from __future__ import annotations

import gzip
import json
import re
import subprocess
import sys
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import graph_layer2_measure as G2M
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import whole_play as WP

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")
WORTH = {"p": 475, "n": 1216, "b": 1301, "r": 1945, "q": 3583}


def test_the_wall_the_module_names_no_game_object_and_reaches_for_no_oracle():
    p = DIR / "whole_play.py"
    low = p.read_text().lower()
    assert not re.search(r"\b[a-h][1-8]\b", low), re.search(r"\b[a-h][1-8]\b", low).group(0)
    assert "sys.path.insert" not in low
    for w in BANNED:
        assert not re.search(rf"\b{w}\b", low), w
    assert not re.search(r"ladder_oracle|leaf_ask|stockfish|analyse|cs_learn|anthropic|\blo\.", low)


def test_the_module_does_not_load_the_oracle_wrapper_even_through_what_it_imports():
    code = ("import sys; import catspace.incubator.analyst.discovery.whole_play; "
            "bad = [m for m in sys.modules if m.endswith('ladder_oracle') or m.endswith('leaf_ask') or m.endswith('cut_count') or m.endswith('cut_mend') or m.endswith('best_count') or m.endswith('best_mend')]; "
            "print('LOADED', bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout + r.stderr


def test_there_is_no_python_wake_the_mover_calls_the_binding_and_writes_the_log():
    t = (DIR / "whole_play.py").read_text()
    assert "wake_decide(" in t and "self.slot" in t
    assert "solve_equals_batch" not in t and "solve_batch" not in t, "the old player is 78.67's own mover; no search is called from here"
    binding = (ROOT / "rust" / "engine" / "py" / "src" / "base_graph_py.rs").read_text()
    assert "carry_slot" in binding and "self.carry.borrow_mut().remove" in binding


def test_the_reason_of_a_decision_is_classed_from_the_sentence_the_log_gives():
    for why, want in (("the end of the game is forced in its line: x", "forced_end"), ("its own line wins by its own reading: net 5", "own_line_wins"), ("answers the threat of theirs on x, which stands; y", "answers_threat"),
                      ("a fix of the goal on x; no threat", "fixes_goal"), ("no threat of theirs and no goal decides it: the reasons' score picks", "estimate"), ("no threat of theirs and no goal decides it: the search's own first move", "search_only"), ("", "none")):
        assert WP.reason_of(why) == want, why


def _games(wake_wall, old_wall, seats=(0, 1)):
    def st(wall, dec, differs=0, **by):
        return {"wall": wall, "decisions": dec, "differs": differs, **{"by_" + k: v for k, v in by.items()}}
    out = []
    for colour in seats:
        s = [st(wake_wall * 10, 10, 2, own_line_wins=1, fixes_goal=1), st(old_wall * 10, 10)]
        out.append({"colour": colour, "stats": s if colour == 0 else s[::-1]})
    return out


def test_the_scale_of_equal_wall_is_the_wakes_mean_wall_over_the_old_players_and_the_counts_by_reason():
    g = _games(0.2, 0.08)
    assert abs(WP.mean_wall(g, 0) - 0.2) < 1e-9 and abs(WP.mean_wall(g, 1) - 0.08) < 1e-9
    assert abs(WP.scale_of(g) - 2.5) < 1e-9
    d = WP.differed(g, 0)
    assert d["decisions"] == 20 and d["differs"] == 4 and d["own_line_wins"] == 2 and d["fixes_goal"] == 2 and d["answers_threat"] == 0
    assert WP.differed(g, 1)["differs"] == 0


def test_the_specs_of_a_match_give_each_player_its_own_clock_the_same_record_and_the_logs_place():
    specs = WP.match_specs("wake", "old", "A", 2, 0.2, Path("/x/scratch"), Path("/x/logs"), None, 0.5, label="m1_wake_v_old")
    assert len(specs) == 4 and len({s["gid"] for s in specs}) == 4
    for s in specs:
        wake = [p for p in s["players"] if p["kind"] == WP.KIND_WAKE][0]
        old = [p for p in s["players"] if p["kind"] != WP.KIND_WAKE][0]
        assert "seconds" not in wake and old["seconds"] == 0.5 and wake["worth"] == old["worth"] and s["log_dir"] == "/x/logs"
    assert [s["colour"] for s in specs[:2]] == [0, 1] and specs[0]["opening"] == specs[1]["opening"]


class _Model:
    """a stand-in for the binding: a canned decision, the call recorded"""
    def __init__(self, move_of):
        self.calls, self.move_of = [], move_of

    def wake_decide(self, pos, history, seconds, depth, worth, half, goal, pieces, budget, named_budget, keys, fuses, chances, reasons, share, slot=None):
        legal = CE.state(OM.VARIANT, pos, [])[2]
        mv, first = self.move_of(legal)
        self.calls.append({"pieces": pieces, "history": history, "seconds": seconds, "slot": slot, "worth": worth})
        return json.dumps({"move": mv, "value": 3, "depth": 5, "nodes": 99, "equal": [first, mv] if mv != first else [first], "named_mine": [], "named_theirs": [], "chain": [], "steps": [],
                           "log": {"why": "a fix of the goal on x; no threat of theirs that stands is answered by it", "moves": [{"move": m} for m in legal]}})


class _World:
    def __init__(self, model):
        self.model, self.goal, self.keys, self.chances, self.reasons = model, 0, [], None, None


def test_the_mover_calls_the_binding_with_every_piece_and_its_slot_and_logs_one_line_a_decision(tmp_path, monkeypatch):
    model = _Model(lambda legal: (legal[1], legal[0]))
    monkeypatch.setattr(WP, "_WORLD", _World(model))
    WP.CTX.clear()
    WP.CTX.update(gid="g:A1:0", log_dir=str(tmp_path))
    spec = {"kind": WP.KIND_WAKE, "worth": dict(WORTH), "charge": False}
    m = WP.MoverWhole(spec, 0.2, "g:A1:0:1", str(tmp_path / "trace.jsonl"))
    text = CE.state(OM.VARIANT, None, [])[0]
    legal = CE.state(OM.VARIANT, text, [])[2]
    mv, depth = m.choose(None, [], legal)
    assert mv == legal[1] and depth == 5
    call = model.calls[0]
    assert call["pieces"] == WP.PIECES and "after_move" in call["pieces"] and "direct_names" in call["pieces"] and call["slot"] == "g:A1:0:1" and call["worth"] == WORTH and call["seconds"] == 0.2
    assert m.stats["decisions"] == 1 and m.stats["differs"] == 1 and m.stats["by_fixes_goal"] == 1 and m.stats["wall"] > 0
    m.choose(None, [legal[0]], CE.state(OM.VARIANT, None, [legal[0]])[2])
    assert len(model.calls[1]["history"]) == 1, "the keys of the positions before this one go to the search (the rules' draw)"
    lines = [json.loads(x) for x in gzip.open(tmp_path / "g_A1_0.jsonl.gz", "rt")]
    assert len(lines) == 2 and lines[0]["kind"] == "decision"
    d = lines[0]
    assert d["played"] == legal[1] and d["search_first"] == legal[0] and d["differs"] and d["reason"] == "fixes_goal" and d["legal"] == legal and d["seat"] == 1 and d["ply"] == 0 and d["wall"] > 0
    assert [x["move"] for x in d["wake"]["log"]["moves"]] == legal and d["position"] == text


def test_the_mover_of_the_other_kind_is_78_67s_unchanged():
    spec = {"kind": WP.KIND_OLD, "worth": dict(WORTH), "charge": False, "seconds": 0.5}
    m = WP.MoverWhole(spec, 0.2, "g:A1:0:0", "/nonexistent/trace.jsonl")
    assert m.seconds == 0.5 and m.log_file is None
    assert WP.KIND_OLD == "tb_control"


def test_the_binding_keeps_the_open_tactics_of_a_slot_between_two_decisions():
    model = GL.model("chess", "chess")
    goal = G2M.goal_kind_of("chess")
    text = CE.state("chess", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7", [])[0]
    pieces = ["carried", "frames"]

    def call(slot):
        return json.loads(model.wake_decide(text, [], 0.3, 3, WORTH, {}, goal, pieces, 0, 0, [], (3, 1500, 30000, 1500), None, None, None, slot))["log"]["notes"]
    first = call("s1")
    assert any("no last decision" in n for n in first)
    assert not any("no last decision" in n for n in call("s1")), "the second decision of the same slot is given the first's open tactics"
    assert any("no last decision" in n for n in call("s2")), "another slot has none"
    assert any("no last decision" in n for n in call(None)), "no slot: nothing is held and nothing is carried"
