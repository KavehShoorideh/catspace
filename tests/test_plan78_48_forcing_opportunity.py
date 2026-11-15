"""Plan 78.48: the opportunity number with the forcing read (catspace/incubator/analyst/discovery/forcing_opportunity_measure.py): the terms of a threat move, the fit of the chance cells, the split by
game, the fallback of a thin cell, the number, the test arrays, the source scan (the script names no game object) and a tiny end-to-end run on two short games."""
import json
import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import forcing_opportunity_measure as FO
from catspace.incubator.analyst.discovery import opportunity_measure as OM

ROOT = Path(__file__).resolve().parents[1]
CHESS_WORDS = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"


def threat(verdict, value, level, claims, took=0.0, plies=2):
    return (verdict, value, level, claims, took, plies)


def read(threats, goal_stake=20.0, cut=False):
    return {"threats": threats, "goal_stake": goal_stake, "cut": cut, "seconds": 0.01, "nodes": 5}


def ply(ours, theirs=None):
    return {"mover": "a", "ours": read(ours), "theirs": None if theirs is None else read(theirs)}


def test_the_verdict_the_kind_and_the_claimed_worth():
    assert FO.verdict_of("UNREAD") == "NOT SETTLED" and FO.verdict_of("CONFIRMED") == "CONFIRMED" and FO.verdict_of("NOT BORNE OUT") == "NOT BORNE OUT"
    assert FO.kind_of([("takes", 3.0), ("exchange", 2.0)]) == "exchange" and FO.kind_of([("sign", 0.0), ("takes", 1.0)]) == "takes" and FO.kind_of([]) == "none"
    assert FO.kind_of([("exchange", 5.0), ("goal", FO.GOAL)]) == "goal" and FO.kind_of([("takes", 1.0), ("ends", FO.GOAL)]) == "ends"
    assert FO.claimed_worth(threat("CONFIRMED", 4.0, 1, [("exchange", 9.0)]), 20.0) == 4.0                 # a confirmed move claims its floor
    assert FO.claimed_worth(threat("CONFIRMED", FO.GOAL, 0, [("goal", FO.GOAL)]), 20.0) == 20.0           # the goal at its stake
    assert FO.claimed_worth(threat("NO CLAIM", 0.0, 0, [("takes", 3.0), ("exchange", 5.0)]), 20.0) == 5.0
    assert FO.claimed_worth(threat("NOT SETTLED", 0.0, 0, [("sign", 0.0)]), 20.0) == 0.0
    assert FO.case_of(threat("CONFIRMED", 4.0, 2, [("takes", 4.0)]), 20.0) == (("CONFIRMED", "takes", "2"), 4.0)


def test_a_position_with_no_pass_has_empty_terms_for_the_other_side():
    c = FO.cases_of_ply(ply([threat("CONFIRMED", 3.0, 0, [("takes", 3.0)])]))
    assert c["has_pass"] is False and c["theirs"] == [] and c["confirmed"] == 1 and len(c["ours"]) == 1
    assert FO.cases_of_ply(None) == {}
    assert FO.cases_of_ply(ply([], [threat("NO CLAIM", 0.0, 1, [("takes", 2.0)])]))["has_pass"] is True


def test_the_split_by_game_is_the_parity_of_the_games_id():
    fit, test = FO.split_games([{"game": g} for g in (10, 11, 12, 13, 14)])
    assert fit == [0, 2, 4] and test == [1, 3]


def a_table(rows):
    """The 78.44-style table of one game: rows of (won4, lost4) for plies 0..; other horizons NaN."""
    n = len(rows)
    a = {"game": np.zeros(n), "ply": np.arange(n, dtype=float)}
    for h in FO.HORIZONS:
        a[f"won{h}"] = np.array([r[0] if h == 4 else np.nan for r in rows], dtype=float)
        a[f"lost{h}"] = np.array([r[1] if h == 4 else np.nan for r in rows], dtype=float)
    return a


def test_the_fit_counts_gained_at_least_the_claimed_worth_by_side():
    reads = [[ply([threat("CONFIRMED", 3.0, 0, [("takes", 3.0)])], [threat("NO CLAIM", 0.0, 0, [("takes", 2.0)])]),     # mover won 3 (>= 3: yes); other side's gain = lost 0 (no)
              ply([threat("CONFIRMED", 3.0, 0, [("takes", 3.0)])]),                                                    # won 2 < 3: no
              ply([threat("CONFIRMED", 3.0, 0, [("takes", 3.0)])]),                                                    # no outcome at 4 (NaN)
              None]]
    a = a_table([(3.0, 0.0), (2.0, 5.0), (np.nan, np.nan), (0.0, 0.0)])
    cells = FO.fit_cells(reads, a, [0])
    assert cells[("CONFIRMED", "takes", "0")][4] == [1, 2]
    assert cells[("NO CLAIM", "takes", "0")][4] == [0, 1]
    assert 8 not in cells[("CONFIRMED", "takes", "0")]                      # no outcome at 8: no case
    assert FO.base_rate(cells, 4) == (1, 3)
    assert FO.hit(0.0, 0.0) is False and FO.hit(0.0, 0.5) is True            # a claim of nothing needs something gained
    assert FO.hit(2.0, 2.0) and not FO.hit(2.0, 1.9)


def test_a_thin_cell_falls_back_to_the_coarser_cell_then_the_verdict_then_all():
    cells = {("CONFIRMED", "takes", "0"): {4: [9, 10]}, ("CONFIRMED", "takes", "1"): {4: [0, 1]}, ("CONFIRMED", "exchange", "0"): {4: [1, 10]}, ("NO CLAIM", "takes", "0"): {4: [5, 40]}}
    t = FO.table_of_cells(cells)
    seen_big, thin = t.chance(("CONFIRMED", "takes", "0"), 4), t.chance(("CONFIRMED", "takes", "1"), 4)
    parent = t.chance(("CONFIRMED", "takes"), 4)
    assert seen_big > parent > thin                                           # the cell of 1 case is pulled to its parent (verdict, kind), away from its own 0 of 1
    assert thin > 0.0
    assert t.chance(("CONFIRMED", "takes", "7"), 4) == pytest.approx(parent)  # a level never seen has the chance of (verdict, kind)
    assert t.chance(("CONFIRMED", "sign", "0"), 4) == pytest.approx(t.chance(("CONFIRMED",), 4))      # a kind never seen has the chance of the verdict
    assert t.chance(("NOT SETTLED", "takes", "0"), 4) == pytest.approx(t.chance((), 4))              # a verdict never seen has the chance of all
    lines = FO.cell_lines(cells, t, "x")
    assert any("cell CONFIRMED/takes/0: cases 10, gained 9" in s for s in lines) and any("base rate" in s for s in lines)


class Flat:
    def __init__(self, c):
        self.c = c

    def chance(self, key, horizon=4):
        return self.c.get(tuple(key), 0.5)


def test_the_number_is_the_best_threat_of_the_mover_less_the_best_of_the_other_side_and_the_sum_form():
    t = Flat({("CONFIRMED", "takes", "0"): 0.8, ("NOT SETTLED", "exchange", "1"): 0.25, ("NO CLAIM", "takes", "0"): 0.1})
    c = FO.cases_of_ply(ply([threat("CONFIRMED", 5.0, 0, [("takes", 5.0)]), threat("NOT SETTLED", 0.0, 1, [("exchange", 8.0)])], [threat("NO CLAIM", 0.0, 0, [("takes", 10.0)])]))
    n = FO.number_at(c, t, 4)
    assert n["f"] == pytest.approx(max(0.8 * 5, 0.25 * 8) - 0.1 * 10)
    assert n["fsum"] == pytest.approx(0.8 * 5 + 0.25 * 8 - 0.1 * 10)
    assert n["count"] == 2 - 1
    c0 = FO.cases_of_ply(ply([threat("CONFIRMED", 5.0, 0, [("takes", 5.0)])]))            # no pass: the other side's terms are empty
    assert FO.number_at(c0, t, 4)["f"] == pytest.approx(4.0)
    assert FO.number_at(FO.cases_of_ply(ply([])), t, 4) == {"f": 0.0, "fsum": 0.0, "count": 0.0}


def opp(n_games, plies_per_game):
    rows = [(g, t) for g in range(n_games) for t in range(plies_per_game)]
    a = {"game": np.array([r[0] for r in rows], float), "ply": np.array([r[1] for r in rows], float), "net": np.arange(len(rows), dtype=float), "net0": np.zeros(len(rows)),
         "count": np.zeros(len(rows)), "net_d": np.zeros(len(rows)), "gain": np.zeros(len(rows)), "risk": np.zeros(len(rows)), "shuffled": np.zeros(len(rows)), "open_any": np.zeros(len(rows))}
    for h in FO.HORIZONS:
        a[f"out{h}"] = np.arange(len(rows), dtype=float)
        a[f"won{h}"] = np.arange(len(rows), dtype=float)
        a[f"lost{h}"] = np.zeros(len(rows))
    return a


def test_the_test_arrays_hold_the_forcing_number_beside_the_78_44_number_and_the_shuffle_keeps_the_values():
    n_games, n = 4, 6
    reads = [[ply([threat("CONFIRMED", float(t), 0, [("takes", float(t))])], [threat("NO CLAIM", 0.0, 0, [("takes", 1.0)])] if t % 2 else None) for t in range(n)] for _g in range(n_games)]
    a_opp = opp(n_games, n)
    t = Flat({("CONFIRMED", "takes", "0"): 0.5, ("NO CLAIM", "takes", "0"): 0.5})
    a = FO.build(reads, a_opp, t, [1, 3])
    assert len(a["ply"]) == 2 * n and set(a["game"]) == {1.0, 3.0}
    assert a["f4"][2] == pytest.approx(0.5 * 2) and a["f4"][3] == pytest.approx(0.5 * 3 - 0.5 * 1)
    assert a["both4"][2] == pytest.approx(a["net"][2] + a["f4"][2]) and a["nconf"][0] == 1 and a["has_pass"][3] == 1 and a["has_pass"][2] == 0
    assert sorted(a["fshuf4"][:n]) == pytest.approx(sorted(a["f4"][:n])) and not np.array_equal(a["fshuf4"][:n], a["f4"][:n])
    lines, data = FO.report(a, 2, 20, "t")
    assert any("positions where the other side has no pass (its terms empty) 6" in s for s in lines)
    assert any("forcing number vs outcome, all test positions" in s for s in lines) and any("net0 = 0 and no CONFIRMED threat" in s for s in lines)
    rho = [c for c in data["correlations"] if c["name"] == "h4 forcing number vs outcome, all test positions"][0]["spearman"]
    assert rho > 0.3                                                           # here the number rises with the outcome by construction


def test_the_script_names_no_game_object_in_code_or_comments():
    text = (ROOT / "catspace" / "incubator" / "analyst" / "discovery" / "forcing_opportunity_measure.py").read_text()
    text = re.sub(r'"(?:\\.|[^"\\])*"', '""', text)
    hit = re.search(CHESS_WORDS, text, flags=re.I)
    assert hit is None, hit.group(0) if hit else None


GAMES = [["e2e4", "d7d5", "e4d5", "d8d5", "b1c3", "d5a5", "g1f3", "c8g4", "h2h3", "g4f3", "d1f3", "e7e6"],
         ["d2d4", "g8f6", "c2c4", "e7e6", "b1c3", "f8b4", "e2e3", "e8g8", "f1d3", "d7d5", "g1f3", "c7c5"]]


def test_an_end_to_end_run_on_two_short_games(tmp_path):
    from catspace.incubator.analyst.discovery import question_sheet as QS
    from catspace.io import paths
    records = [{"variant": "chess", "game": i, "start": None, "moves": m} for i, m in enumerate(GAMES)]
    res = FO.read_chunk("chess", records, 2, 2000)
    assert len(res) == 2 and all(len(r) == len(m) + 1 for r, m in zip(res, GAMES))
    p = res[0][0]
    assert set(p) == {"mover", "ours", "theirs"} and {"threats", "goal_stake", "cut", "seconds", "nodes"} <= set(p["ours"]) and p["ours"]["goal_stake"] > 0
    assert all(len(t) == 6 and t[0] in ("CONFIRMED", "NOT BORNE OUT", "NOT SETTLED", "NO CLAIM", "UNREAD") for x in res[0] if x for t in x["ours"]["threats"])
    ck = str(tmp_path / "reads.jsonl")
    got = FO.collect("chess", records, 2, 2000, ck, 1)
    again = FO.collect("chess", records, 2, 2000, ck, 1)                       # the second call reads nothing: every game is in the checkpoint
    assert json.loads(json.dumps(again)) == json.loads(json.dumps(got)) and len((tmp_path / "reads.jsonl").read_text().splitlines()) == 2
    table = QS.load_chances(paths.derived("question_chances", "fit1112.json"))
    a_opp = OM.table_of(OM.briefs_of("chess", records, 2), table)
    reads = [got[r["game"]] for r in records]
    lines, data = FO.run(records, "chess", a_opp, reads, 10, "tiny")
    assert any("games 2 (fit 1 even-numbered, test 1 odd-numbered)" in s for s in lines)
    assert any("spearman h4 78.44 number vs outcome, all test positions" in s for s in lines)
    assert data["cells"]
    pr = FO.probe("chess", records, 2, [2000, 500], "tiny")
    assert any("fuse 2000:" in s for s in pr) and any("differ between fuse 2000 and fuse 500" in s for s in pr)
