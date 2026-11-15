"""Plan 78.44: the opportunity number along a game (catspace/incubator/analyst/discovery/opportunity_measure.py): the number and its baselines on hand-built briefs, the outcome at a horizon,
the shuffled control, the source scan (the script names no game object) and one small end-to-end run on a short fixed game."""
import random
import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import opportunity_measure as OM

ROOT = Path(__file__).resolve().parents[1]
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


class Table:
    """A stand-in for the chance table: the chance of a key by its exact words, else 0.5."""
    def __init__(self, cells):
        self.cells = cells

    def chance(self, key, horizon=4):
        return self.cells.get(tuple(key), 0.5)


def target(owner, worth, verdict, key=None, live0=False):
    side = "mine" if owner == "us" else "theirs"
    return (owner, "x", "p", worth, verdict, verdict in OM.DANGER_KINDS, 1.0, 2, key or [side, verdict, "1", "clear"], live0)


TABLE = Table({("theirs", "confirmed", "1", "clear"): 0.8, ("mine", "lost", "0", "clear"): 0.5, ("theirs", "live0"): 0.9, ("mine", "live0"): 0.6, ("theirs", "waiting", "2", "own"): 0.25})
BRIEF = {"mover": "a", "targets": [
    target("them", 8, "confirmed"),                            # open, danger: 0.8 x 8
    target("them", 4, "waiting", ["theirs", "waiting", "2", "own"]),   # open, not danger: 0.25 x 4
    target("them", 14, "refuted"),                             # not open
    target("us", 5, "lost", ["mine", "lost", "0", "clear"]),   # open, danger against me: 0.5 x 5
    target("us", 2, "closed2"),                                # not open
    target("them", 6, "refuted", live0=True),                  # level 0 read it live, a later level closed it: only in net0 (0.9 x 6)
    target("us", 3, "idle1", live0=True),                      # only in net0 (0.6 x 3)
    target("us", 0, "unread"),
]}


def test_gain_risk_net_and_the_baselines_from_a_hand_built_brief():
    n = OM.number_of(BRIEF, TABLE)
    assert n["gain"] == pytest.approx(0.8 * 8 + 0.25 * 4)
    assert n["risk"] == pytest.approx(0.5 * 5)
    assert n["net"] == pytest.approx(n["gain"] - n["risk"])
    assert n["net0"] == pytest.approx(0.9 * 6 - 0.6 * 3)
    assert n["count"] == 2 - 1                                  # two open claims against theirs, one against mine
    assert n["net_d"] == pytest.approx(0.8 * 8 - 0.5 * 5)       # the danger kinds only
    assert n["mine"] == 10 and n["theirs"] == 32 and n["open_any"] == 3


def test_a_kind_the_table_has_no_cell_for_takes_the_tables_own_value():
    b = {"mover": "a", "targets": [target("them", 10, "live1", ["theirs", "live1", "1", "clear"])]}
    assert OM.number_of(b, TABLE)["gain"] == pytest.approx(5.0)      # the stand-in's fallback 0.5; the real table's prefix fallback is the Python oracle's


def test_the_real_tables_prefix_fallback_is_used_for_an_unseen_key():
    from catspace.incubator.analyst.discovery import question_sheet as QS
    t = QS.Chances({("theirs", "confirmed", "1", "clear"): {4: (3, 4)}})
    root, seen = t.chance(("theirs",), 4), t.chance(("theirs", "confirmed", "1", "clear"), 4)
    assert t.chance(("theirs", "refuted", "9", "own"), 4) == pytest.approx(t.chance(("theirs",), 4)) and seen > root
    b = {"mover": "a", "targets": [target("them", 10, "confirmed", ["theirs", "confirmed", "1", "clear"])]}
    assert OM.number_of(b, t)["gain"] == pytest.approx(10 * seen)


def plies(rows):
    return [None if r is None else {"mover": r[0], "mine": r[1], "theirs": r[2]} for r in rows]


def test_the_outcome_at_a_horizon_is_the_movers_change_of_balance_and_its_two_halves():
    # ply 0: a to move, a has 10, b has 10.  ply 2: a to move again, a 9 and b 6 (a won 4, lost 1).  ply 3: b to move, b has 6, a has 9
    g = plies([("a", 10, 10), ("b", 10, 10), ("a", 9, 6), ("b", 6, 9), ("a", 9, 6)])
    assert OM.outcome_at(g, 0, 2) == (3, 4, 1)
    assert OM.outcome_at(g, 1, 2) == (-3, 1, 4)              # b to move at 1: at 3 b has 6 (lost 4) and a has 9 (b won 1)
    assert OM.outcome_at(g, 0, 1) == (0, 0, 0)               # the other mover at t+1 is read from the mover's side: same worths, no change
    assert OM.outcome_at(g, 2, 3) is None                    # no ply 5
    assert OM.outcome_at(plies([("a", 1, 1), None, ("a", 1, 1)]), 0, 1) is None     # a position that is over is no reading


def test_the_shuffled_control_changes_nothing_but_the_order():
    nets = [1.0, None, 2.5, -3.0, 0.0, 7.0]
    out = OM.shuffled_net(nets, random.Random(1))
    assert out[1] is None and sorted(x for x in out if x is not None) == sorted(x for x in nets if x is not None)
    assert OM.shuffled_net(nets, random.Random(1)) == out    # the seed fixes it
    assert out != nets


def test_the_table_of_a_game_has_the_outcome_where_the_horizon_exists():
    mk = lambda mover, me, them: {"mover": mover, "targets": [target("us", me, "idle1"), target("them", them, "confirmed")]}
    briefs = [[mk("a", 10, 10), mk("b", 10, 10), mk("a", 9, 6), mk("b", 6, 9), mk("a", 9, 6)] + [mk("a" if i % 2 == 0 else "b", 9, 6) for i in range(20)], [None]]
    a = OM.table_of(briefs, TABLE)
    assert len(a["ply"]) == 25 and set(a["game"]) == {0.0}
    assert a["out4"][0] == pytest.approx(3.0) and a["won4"][0] == pytest.approx(4.0) and a["lost4"][0] == pytest.approx(1.0)
    assert np.isnan(a["out4"][-1]) and np.isnan(a["out16"][-1]) and not np.isnan(a["out4"][0])
    assert np.nansum(a["shuffled"]) == pytest.approx(a["net"].sum())      # the same values, only permuted


def test_the_rank_correlation_and_the_bands():
    x = np.arange(100.0)
    assert OM.spearman(x, x * 2) == pytest.approx(1.0) and OM.spearman(x, -x) == pytest.approx(-1.0)
    a = {"net": x, "out4": x, "won4": np.maximum(x, 0), "lost4": np.zeros(100)}
    b = OM.bands(a, 4)
    assert [r[1] for r in b] == [20] * 5 and b[0][3] < b[-1][3]
    z = np.sin(x)
    assert OM.partial_spearman(x, x + 0.0, z) == pytest.approx(1.0)


def test_the_script_names_no_game_object_in_code_or_comments():
    text = (ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "opportunity_measure.py").read_text()
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    hit = re.search(CHESS_WORDS, text, flags=re.I)
    assert hit is None, hit.group(0) if hit else None


GAME_MOVES = ["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "g1f3", "c8g4", "h2h3", "g4f3", "d1f3", "e7e6", "f1b5", "c7c6", "b5c4", "g8f6"]


def test_an_end_to_end_reading_of_a_short_fixed_game(tmp_path):
    import json
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.io import paths
    (tmp_path / "games.jsonl").write_text(json.dumps({"variant": "chess", "game": 0, "start": None, "moves": GAME_MOVES, "outcome": None}) + "\n")
    table = QS.load_chances(paths.derived("question_chances", "fit1112.json"))
    briefs = OM.briefs_of("chess", OM.read_games(str(tmp_path)), 2)
    assert len(briefs) == 1 and len(briefs[0]) == len(GAME_MOVES) + 1
    assert all(len(t) == 10 and len(t[8]) == 4 for b in briefs for x in b if x for t in x["targets"])
    a = OM.table_of(briefs, table)
    assert len(a["ply"]) == len(GAME_MOVES) + 1
    assert np.isfinite(a["net"]).all() and np.isfinite(a["net0"]).all()
    assert (~np.isnan(a["out4"])).sum() == len(GAME_MOVES) + 1 - 4 and (~np.isnan(a["out16"])).sum() == 1
    assert a["out4"][0] == pytest.approx(a["won4"][0] - a["lost4"][0])
