"""PLAN 79.17 CAN WE MATE? THE MEASURE OF PROGRESS BROUGHT INTO THE NEW ENGINE (SYSTEM_SPEC 7 "the small endings as the test of means and ends, and a measure of progress", 8 the wall, 9 the oracle rule; ledger rows 13, 23, 24, 25, 35; pieces: the measure, sleep, the wake).
Rust holds the unit tests (cargo test --release --offline -p cs_tactics --test measure --test measure_regions; -p cs_decision --test wake_measure): the atoms and terms of cs_tactics::measure, October's question by the search of toward.rs, the wake's held goal.
THIS file holds what belongs to Python: the wall (the measure scripts name no game object and only the oracle of the measure reads the table), the atoms against October's Python readings through the binding, the oracle's defender, the miner's pieces, the means of a game's log, the binding taking the goal.
"""
from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import chess
import numpy as np
import pytest

from catspace.incubator.analyst.discovery import measure_miner_79_17 as MM
from catspace.incubator.analyst.discovery import measure_oracle_79_17 as MO
from catspace.incubator.analyst.discovery import whole_79_17 as W17

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
BANNED = ("fen", "san", "uci", "check", "mate", "promotion", "king", "queen", "rook", "bishop", "knight", "pawn", "castle", "castling", "chess")


def _table_present() -> bool:
    return (MO.table_dir() / "KRvK.rtbw").exists() and (MO.table_dir() / "KRRvK.rtbw").exists()


needs_table = pytest.mark.skipif(not _table_present(), reason="the endgame tables are an untracked input (assets/tablebases)")


def test_the_measure_scripts_name_no_game_object_and_only_the_oracle_of_the_measure_reads_the_table():
    for name in ("whole_79_17.py", "measure_miner_79_17.py"):
        low = (DIR / name).read_text().lower()
        assert not re.search(r"\b[a-h][1-8]\b", low), name
        assert "sys.path.insert" not in low
        for w in BANNED:
            assert not re.search(rf"\b{w}\b", low), (name, w)
        assert not re.search(r"ladder_oracle|leaf_ask|stockfish|analyse_pos|cs_learn|anthropic|syzygy|\blo\.", low), name
    users = [p.name for p in (ROOT / "catspace").rglob("*.py") if "measure_oracle_79_17" in p.read_text() and p.name not in ("measure_oracle_79_17.py",)]
    assert set(users) <= {"whole_79_17.py", "measure_miner_79_17.py", "whole_79_17_print.py", "measure_miner_79_17_print.py", "layer3_fixtures_print.py"}, users
    assert "syzygy" not in "".join((DIR / n).read_text() for n in ("whole_play.py", "whole_79_16.py", "whole_79_14.py")).lower()
    for crate in ("tactics", "decision"):
        assert "cs_learn" not in (ROOT / "rust" / "engine" / crate / "Cargo.toml").read_text()
    # no crate source reads a table, an oracle or the learning crate for the measure
    for p in (ROOT / "rust" / "engine" / "tactics" / "src" / "measure.rs", ROOT / "rust" / "engine" / "decision" / "src" / "wake.rs"):
        assert not re.search(r"syzygy|tablebase|oracle_wrapper", p.read_text().lower())


def test_the_wake_with_no_goal_given_does_not_know_the_oracle_and_the_old_player_is_unchanged():
    wake = (ROOT / "rust" / "engine" / "decision" / "src" / "wake.rs").read_text()
    assert "memory: crate::memory::WakeMemory::default()" in wake          # plan 79.19: no goal given is a memory that holds nothing (the field `measure` moved into the two memories; `hold` is None by default)
    assert "MeasureGoal" in wake


# ------------------------------------------------------------------------------------------------------------------------------ the oracle of the measure

@needs_table
def test_the_starts_are_won_positions_with_the_stronger_side_to_move_both_colours():
    for cls in MO.CLASSES:
        s = MO.sample_starts(cls, 12, 5)
        assert {x["attacker_white"] for x in s} == {True, False}
        for x in s:
            b = chess.Board(x["text"])
            assert b.turn == x["attacker_white"] and MO.wdl_for(b, x["attacker_white"]) == 2 and not b.is_game_over()
        assert sum(MO.coverage(s).values()) == 12


@needs_table
def test_the_defender_holds_longest_by_the_tables_distance_and_never_gives_the_win_to_a_draw():
    s = MO.sample_starts("one", 6, 9)
    for x in s:
        b = chess.Board(x["text"])
        moves = [next(iter(sorted(m.uci() for m in b.legal_moves)))]            # the stronger side plays its first move by the text
        b.push_uci(moves[0])
        if b.is_game_over():
            continue
        legal = [m.uci() for m in b.legal_moves]
        reply = MO.defender_reply(x["text"], moves, legal)
        tb = MO.tables()
        best = None
        for u in legal:
            b.push_uci(u)
            if not b.is_game_over():
                w, d = tb.probe_wdl(b), abs(tb.probe_dtz(b))
                if best is None or (w, -d) < best[0]:
                    best = ((w, -d), u)
            b.pop()
        if best is not None:
            assert reply in legal
            b.push_uci(reply)
            assert (tb.probe_wdl(b), -abs(tb.probe_dtz(b))) == best[0]


@needs_table
def test_a_game_is_classed_by_how_it_ended():
    start = "7k/R7/6K1/8/8/8/8/8 w - - 0 1"
    assert MO.ending_of(start, ["a7a8"], 100, False, True) == "won"
    assert MO.ending_of(start, [], None, True, True) == "cap"
    assert MO.ending_of("8/8/8/8/8/8/R7/K1k5 w - - 100 80", [], 50, False, True) == "quiet_rule"
    assert MO.ending_of("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", [], 50, False, True) == "stalemate"


@needs_table
def test_the_generated_games_split_by_whether_they_reach_the_end_and_the_table_set_is_kept_apart():
    good, bad = MO.generated_games("one", 6, 3)
    assert len(good) + len(bad) == 6
    for r in good:
        assert MO.attacker_won(r["text"], r["moves"], r["attacker_white"])
    for r in bad:
        assert not MO.attacker_won(r["text"], r["moves"], r["attacker_white"]) and MO.kept_the_win(r["text"], r["moves"], r["attacker_white"])
    t = MO.table_games("one", 4, 4)
    assert all(MO.attacker_won(r["text"], r["moves"], r["attacker_white"]) for r in t)


# ------------------------------------------------------------------------------------------------------------------------------ the miner's pieces

def test_the_terms_of_the_library_are_ordered_simplest_first_and_each_names_the_atoms_it_reads():
    atoms = ["a", "b", "c"]
    terms = MM.terms_of(atoms)
    assert [t[0] for t in terms[:3]] == ["a", "b", "c"]
    assert terms[3][2] == 2 and terms[3][3] != "tuple"
    assert any(t[3] == "tuple" and t[2] == 3 for t in terms)
    assert len({t[0] for t in terms}) == len(terms)


def test_test1_reads_the_share_of_moves_after_which_a_term_is_lower_by_set():
    atoms = ["x", "y"]
    good, bad = MM.Table(atoms), MM.Table(atoms)
    good.rows = [np.array([5, 1]), np.array([4, 1]), np.array([3, 1]), np.array([2, 1])]
    good.pairs = [(0, 1), (1, 2), (2, 3)]
    bad.rows = [np.array([5, 1]), np.array([6, 1]), np.array([5, 1]), np.array([6, 1])]
    bad.pairs = [(0, 1), (1, 2), (2, 3)]
    terms = MM.terms_of(atoms)
    s = MM.test1({"good": good, "bad": bad}, atoms, terms)
    assert s["x"] == {"good": 1.0, "bad": 1 / 3}
    assert s["y"] == {"good": 0.0, "bad": 0.0}
    assert s["[sum x y]"]["good"] == 1.0
    assert s["[tuple y x]"]["good"] == 1.0, "a tuple is compared in order: y does not change, x falls"
    assert s["[tuple x y]"]["bad"] == 1 / 3


# ------------------------------------------------------------------------------------------------------------------------------ the logs and the cells

def _game(plies: int, outcome, unfinished: bool, moves: list, attacker_white=True) -> dict:
    return {"gid": "c:one0:0", "start": "one0", "colour": 0, "outcome": outcome, "plies": plies, "unfinished": unfinished, "moves": moves, "stats": [{"depth_sum": 20, "decisions": 4, "wall": 0.4}, {"depth_sum": 0, "decisions": 4, "wall": 0.0}]}


@needs_table
def test_a_cell_counts_wins_endings_and_the_mean_plies():
    start = {"text": "7k/R7/6K1/8/8/8/8/8 w - - 0 1", "attacker_white": True, "dtz": 1, "wdl": 2, "class": "one"}
    s = W17.summarise([_game(1, 100, False, ["a7a8"])], [start], "one")
    assert s["games"] == 1 and s["wins"] == 1 and s["ended"] == {"won": 1} and s["mean_plies"] == 1
    line = W17.cell_line("t", s)
    assert "wins 1" in line and "ended by {'won': 1}" in line


def _log_row(reached: bool, toward: bool, dropped) -> dict:
    return {"kind": "decision", "wake": {"log": {"measure": {"reached": reached, "played_toward": toward, "dropped": dropped}}}}


def test_the_fall_of_the_measure_along_a_game_counts_drops_the_longest_run_and_why_a_goal_was_dropped(tmp_path):
    rows = [_log_row(False, True, None), _log_row(True, True, None), _log_row(False, False, None), _log_row(False, True, None), _log_row(False, True, None), _log_row(True, True, "no line of at most 3: x")]
    with gzip.open(tmp_path / "g1.jsonl.gz", "wt") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    d = W17.falls(tmp_path)
    assert d["games"] == 1 and d["drops"] == 2 and d["longest_run"] == 3 and d["toward_played"] == 5
    assert list(d["dropped_why"]) == ["no line of at most 3"]


# ------------------------------------------------------------------------------------------------------------------------------ the binding

@pytest.fixture(scope="module")
def model():
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    try:
        return W14.world().model
    except FileNotFoundError as e:                                                # the untracked inputs of the harness (a worktree without them)
        pytest.skip(str(e))


@needs_table
def test_the_atoms_agree_with_octobers_python_readings_on_won_positions(model):
    from catspace.incubator.analyst.discovery import technique_measures as TMS
    s = MO.sample_starts("one", 24, 11)
    side = {True: 0, False: 1}
    groups = sorted(map(tuple, model.measure_groups(s[0]["text"], side[s[0]["attacker_white"]])))
    kinds = model.kinds()
    lead = kinds.index(next(k for k in kinds if k == "king")) if "king" in kinds else None
    assert lead is not None
    atoms = [f"region(1,{lead})", f"edge(1,{lead})", f"corner(1,{lead})", f"dist(0,{lead},1,{lead})"] + [f"span(1,{lead},{f})" for f in range(4)]
    for x in s:
        b = chess.Board(x["text"])
        mine = model.measure_atoms([x["text"]], side[x["attacker_white"]], atoms)[0]
        oct = TMS.atom_values(b, x["attacker_white"])
        assert mine[0] == oct["region"], (x["text"], mine, oct)
        assert mine[1] == oct["edge_dist"] and mine[2] == oct["corner_dist"] and mine[3] == oct["kings_dist"], (x["text"], mine, oct)
        spans = mine[4:]
        assert any(spans[i] * spans[j] == oct["region_area"] for i in range(4) for j in range(i + 1, 4)), (x["text"], spans, oct["region_area"])
    assert groups, "the class has groups"


def test_the_binding_gives_the_goal_to_the_wake_and_takes_no_goal_as_before(model):
    from catspace.incubator.analyst.discovery import whole_79_14 as W14
    w = W14.world()
    text = "7k/8/8/8/8/8/R7/K7 w - - 0 1"
    kinds = model.kinds()
    lead = kinds.index("king")
    term = f"region(1,{lead})"
    args = (text, [], 0.5, 64, {"p": 475, "n": 1216, "b": 1301, "r": 1945, "q": 3583}, {}, "king", ["goals", "threats", "direct_names"], 0, 0, None, None, None, None)
    plain = json.loads(model.wake_decide(*args))
    assert plain["log"]["measure"] is None
    held = json.loads(model.wake_decide(*args, measure=(term, 2), carry_slot="t17"))
    m = held["log"]["measure"]
    assert m["term"] == term and m["d"] == 2 and m["first"] and "taken up" in m["said"]
    again = json.loads(model.wake_decide(*args, measure=(term, 2), carry_slot="t17"))
    assert again["log"]["measure"]["held_before"] is not None, "the goal is carried by the binding's slot from move to move"
    first, moves, game_end, cut, nodes = model.measure_lowers(text, 0, term, model.measure_value(text, 0, term), 2, 5.0)
    assert first == m["first"] and moves >= 1 and not cut and nodes > 0
