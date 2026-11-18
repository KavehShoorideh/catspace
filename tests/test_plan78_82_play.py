"""Plan 78.82: the reasons of 78.81 as the hint among the moves the search holds equal, in the match harness (catspace/incubator/analyst/discovery/symbols_play.py).
SYSTEM_SPEC 7 (the hint among equal moves), 8 (the wake), 9 (the wake's match).

The pick stays in the tie set over random tables; silence and ties go to the search's first move; the quiet-only mode returns the first move when a tied move takes a unit; the player's pick equals symbols' pick on built
data; a two-pair match runs end to end and its picks are a function of the position and the tie set; the layer names no chess object."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import symbols as SY
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.incubator.analyst.discovery import tiebreak_measure as TM

ROOT = Path(__file__).resolve().parents[1]
DISC = ROOT / "catspace" / "incubator" / "analyst" / "discovery"
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
HAVE = (SP.src_root() / "reasons.json").exists() and (SP.src_root() / "fits.pkl").exists()
needs_run = pytest.mark.skipif(not HAVE, reason="needs the 78.81 run's reasons.json and fits.pkl")


def stub_reader(n_reasons: int = 12, seed: int = 0, ctl: "dict | None" = None) -> SP.ReasonsReader:
    """A reader with built weights and no engine: the tables the engine would count are handed in by the test (monkeypatched deltas_of / controls_of)."""
    rd = SP.ReasonsReader.__new__(SP.ReasonsReader)
    rng = np.random.default_rng(seed)
    rd.reasons = [{"weight": float(w)} for w in rng.normal(size=n_reasons)]
    ev = [(j, "mover", "up") for j in range(n_reasons)]
    rd.weight = {e: rd.reasons[j]["weight"] for j, e in enumerate(ev)}
    rd.index = {e: j for j, e in enumerate(ev)}
    rd.order = list(ev)
    rd.ctl_weights = ctl if ctl is not None else {"takes": 1.2, "moves:a": 0.3, "moves:b": -0.2, "moves:none": 0.0}
    rd.counter, rd.facts = None, None
    return rd


def random_tables(rd: SP.ReasonsReader, tied: list, rng: np.random.Generator) -> dict:
    return {m: frozenset(e for e in rd.order if rng.random() < 0.3) for m in tied}


def patch(monkeypatch, rd: SP.ReasonsReader, table: dict, takes: dict, kinds: dict) -> None:
    monkeypatch.setattr(SP.SY, "deltas_of", lambda position, tied, counter: table)
    monkeypatch.setattr(rd, "controls_of", lambda moves, tied: (takes, kinds), raising=False)


# ------------------------------------------------------------------------------------------------------------------------------ the pick among the tied moves

@pytest.mark.parametrize("mode", ["reasons", "quiet", "withctl"])
def test_the_pick_is_always_one_of_the_tied_moves(monkeypatch, mode):
    rng = np.random.default_rng(5)
    for seed in range(60):
        rd = stub_reader(seed=seed)
        tied = [f"m{j}" for j in range(int(rng.integers(2, 12)))]
        table = random_tables(rd, tied, rng)
        takes = {m: bool(rng.random() < 0.2) for m in tied}
        kinds = {m: str(rng.choice(["a", "b", ""])) for m in tied}
        patch(monkeypatch, rd, table, takes, kinds)
        move, detail = rd.pick(["x"] * 12, tied, mode)
        assert move in tied
        assert set(detail["scores"]) == set(tied)


def test_silence_and_ties_go_to_the_search_s_first_move(monkeypatch):
    rd = stub_reader()
    tied = ["first", "b", "c", "d"]
    patch(monkeypatch, rd, {m: frozenset() for m in tied}, {m: False for m in tied}, {m: "" for m in tied})
    assert rd.pick(["x"], tied, "reasons")[0] == "first"                      # no reason carried: silent
    same = frozenset([rd.order[0], rd.order[3]])
    patch(monkeypatch, rd, {m: same for m in tied}, {m: False for m in tied}, {m: "" for m in tied})
    assert rd.pick(["x"], tied, "reasons")[0] == "first"                      # every move carries the same: a tie
    rd.weight[rd.order[0]] = 5.0
    patch(monkeypatch, rd, {"first": frozenset(), "b": frozenset([rd.order[0]]), "c": frozenset(), "d": frozenset([rd.order[0]])}, {m: False for m in tied}, {m: "" for m in tied})
    assert rd.pick(["x"], tied, "reasons")[0] == "b"                          # the first of the moves with the top score, not the last
    assert rd.pick(["x"], ["only"], "reasons")[0] == "only"                   # no tie to break


def test_the_quiet_mode_returns_the_first_move_when_a_tied_move_takes_a_unit(monkeypatch):
    rd = stub_reader()
    rd.weight[rd.order[1]] = 4.0
    tied = ["first", "b", "c"]
    table = {"first": frozenset(), "b": frozenset([rd.order[1]]), "c": frozenset()}
    patch(monkeypatch, rd, table, {"first": False, "b": False, "c": True}, {m: "" for m in tied})
    assert rd.pick(["x"], tied, "quiet")[0] == "first"                        # a tied move takes: no hint
    assert rd.pick(["x"], tied, "reasons")[0] == "b"                          # the same position, the reasons alone choose
    patch(monkeypatch, rd, table, {m: False for m in tied}, {m: "" for m in tied})
    assert rd.pick(["x"], tied, "quiet")[0] == "b"                            # none takes: the reasons choose


def test_the_controls_enter_only_in_their_own_mode(monkeypatch):
    rd = stub_reader(ctl={"takes": 10.0, "moves:none": 0.0})
    tied = ["first", "b", "c"]
    table = {"first": frozenset(), "b": frozenset([rd.order[0]]), "c": frozenset()}
    rd.weight[rd.order[0]] = 1.0
    patch(monkeypatch, rd, table, {"first": False, "b": False, "c": True}, {m: "" for m in tied})
    assert rd.pick(["x"], tied, "reasons")[0] == "b"
    assert rd.pick(["x"], tied, "withctl")[0] == "c"                          # the weight of taking is added in this mode only


def test_the_player_s_pick_equals_symbols_pick_on_built_data(monkeypatch):
    rng = np.random.default_rng(11)
    for seed in range(40):
        rd = stub_reader(seed=seed)
        tied = [f"m{j}" for j in range(int(rng.integers(2, 10)))]
        table = random_tables(rd, tied, rng)
        patch(monkeypatch, rd, table, {m: False for m in tied}, {m: "" for m in tied})
        pats = [{"column": e[0], "side": e[1], "dir": e[2], "weight": w} for e, w in rd.weight.items()]
        assert rd.pick(["x"], tied, "reasons")[0] == SY.symbol_pick({"moves": ["x"]}, tied, pats, None, table)


# ------------------------------------------------------------------------------------------------------------------------------ the player in a match

def test_the_reading_time_is_taken_from_the_charged_player_s_clock():
    spec = {"kind": "sy_reasons", "worth": {}, "charge": True}
    m = SP.MoverReasons(spec, 0.2, "g", "/nonexistent/trace")
    m.costs = [0.05, 0.05]
    assert abs(m.search_clock() - 0.15) < 1e-9
    plain = SP.MoverReasons(dict(spec, charge=False), 0.2, "g", "/nonexistent/trace")
    plain.costs = [0.05]
    assert plain.search_clock() == 0.2
    control = SP.MoverReasons({"kind": "tb_control", "worth": {}, "charge": True}, 0.2, "g", "/nonexistent/trace")
    control.costs = [0.05]
    assert control.search_clock() == 0.2                                      # the control is never charged


def test_every_player_is_named_and_maps_to_a_kind():
    assert set(SP.PLAYERS) >= {"control", "random", "reasons", "quiet", "withctl"}
    assert set(SP.MODE) == {"sy_reasons", "sy_quiet", "sy_withctl"}
    for p in ("reasons", "quiet", "withctl"):
        assert SP.KIND[p] in SP.MODE


@needs_run
def test_a_two_pair_match_runs_end_to_end_and_its_picks_are_a_function_of_the_position(tmp_path):
    specs = TM.match_specs("reasons", "control", "A", 2, 0.05, tmp_path / "scratch", False, SP.KIND)
    for s in specs:
        s["max_plies"] = 40
    games = TM.run_games(specs, 1, tmp_path / "g.jsonl", lambda s: None, play=SP.play82)
    assert [g["gid"] for g in games] == [s["gid"] for s in specs] and len(games) == 4
    assert all(g["plies"] > 0 and len(g["stats"]) == 2 for g in games)
    # a rerun skips the games already played (the same file), and the result is the same games
    again = TM.run_games(specs, 1, tmp_path / "g.jsonl", lambda s: None, play=SP.play82)
    assert [g["moves"] for g in again] == [g["moves"] for g in games]
    # the picks logged are the reader's pick for that position and tie set, again (the clock only decides which moves are tied)
    rd = SP.reader82()
    n = 0
    for g in games:
        st = g["stats"][TM.seat_of(g, 0)]
        assert st["decisions"] >= len(st["dlog"]) == st["ties"]
        for c in st["cases"]:
            assert c["pick"] in c["tied"] and c["first"] == c["tied"][0] and c["pick"] != c["first"]
            assert rd.pick(c["moves"], c["tied"], "reasons")[0] == c["pick"]
            n += 1
    assert n >= 1
    ctl = [g["stats"][TM.seat_of(g, 1)] for g in games]
    assert all(s["changed"] == 0 and not s["cases"] for s in ctl)             # the control plays the search's own move


@needs_run
def test_the_real_reader_loads_once_and_keeps_the_pick_in_the_tie_set():
    from catspace.incubator.analyst.discovery import symbols_measure as SM
    rd = SP.reader82()
    assert SP.reader82() is rd
    rows = [r for r in SM.jsonl(Path(TM.paths.experiment("symbols_78_80", "full", "sample.jsonl"))) if r["part"] == "confirm"][:12]
    for r in rows:
        for mode in ("reasons", "quiet", "withctl"):
            mv, _d = rd.pick(r["moves"], r["tied"], mode)
            assert mv in r["tied"]
        assert rd.pick(r["moves"], r["tied"], "reasons")[0] == rd.pick(r["moves"], r["tied"], "reasons")[0]


# ------------------------------------------------------------------------------------------------------------------------------ the scan

def test_the_player_names_no_chess_object_and_only_the_printing_script_spells_moves():
    hit = re.search(CHESS_WORDS, (DISC / "symbols_play.py").read_text(), flags=re.I)
    assert hit is None, repr(hit.group(0))
    assert "import chess" not in (DISC / "symbols_play.py").read_text()
    assert "import chess" in (DISC / "ladder_print.py").read_text()
    assert "play82" in (DISC / "ladder_print.py").read_text()
