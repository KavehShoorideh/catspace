"""PLAN 78.26 WIN / DRAW / LOSS FROM ANY POSITION (cs_planning::wdl): the probability of a win, a draw, a loss from playouts pooled with a results table, how sharp the
position is, and the controller's rule -- sharp (or unknown): keep reasoning; quiet: the probabilities pick among the moves the exact reading leaves equal.
The positions are chess (the extension names them); the layer reads the Game trait only."""
import json
import re
import time
from pathlib import Path

import pytest

import cs_engine as CE
from catspace.incubator.analyst.discovery import lemma_store as LS
from catspace.incubator.analyst.discovery import wdl_table as WT

J = LS.judge_of("chess", "material")
BACK_RANK = "6k1/5ppp/8/8/8/8/8/R3K3 w - - 0 1"         # Ra8 mates: decisive, few units
LOCKED = "4k3/4p3/8/8/8/8/4P3/4K3 w - - 0 1"            # the pawns block each other: nothing to win or lose by force
KINGS = "8/8/4k3/8/8/4K3/8/8 w - - 0 1"                 # two kings only: the game is over by the rules' draw (no estimate to make)
MIDDLE = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
OPENING = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6"]
TRACE_MODE = re.compile(r"^wdl: (sharp|quiet|unknown) \(")


def test_78_26_a_decisive_position_is_sharp_with_a_high_score():
    r = CE.wdl("chess", BACK_RANK, [], seconds=0.2, **J)
    print("OBS 78.26 decisive:", {k: v for k, v in r.items() if k not in ("evidence", "moves")})
    assert r["sharp"] is True and r["score"] > 0.9 and r["win"] > 0.9 and r["sharpness"] > 0.9
    assert r["move"] == "a1a8" and r["moves"] and all(e.startswith("wdl: ") for e in r["evidence"])
    assert r["table_key"] is None and r["playouts"] > 0


def test_78_26_a_blocked_position_is_quiet_and_two_bare_kings_have_no_estimate():
    r = CE.wdl("chess", LOCKED, [], seconds=0.2)
    print("OBS 78.26 quiet:", {k: v for k, v in r.items() if k not in ("evidence", "moves")})
    assert r["sharp"] is False and r["draw"] > 0.8
    with pytest.raises(ValueError, match="over"):
        CE.wdl("chess", KINGS, [], seconds=0.1)


def test_78_26_the_probabilities_sum_to_one_and_the_score_agrees():
    for fen in (BACK_RANK, LOCKED, MIDDLE):
        r = CE.wdl("chess", fen, [], seconds=0.1, **J)
        assert abs(r["win"] + r["draw"] + r["loss"] - 1.0) < 1e-3
        assert abs(r["score"] - (r["win"] + r["draw"] / 2)) < 2e-3
        for m in r["moves"]:
            assert abs(m["win"] + m["draw"] + m["loss"] - 1.0) < 1e-3 and m["visits"] > 0


def test_78_26_among_returns_only_those_moves_and_picks_the_win():
    legal = CE.state("chess", BACK_RANK, [])[2]
    pick = ["a1a8", "a1a2", "a1a3"]
    r = CE.wdl("chess", BACK_RANK, [], seconds=0.3, among=pick, **J)
    print("OBS 78.26 among:", [(m["move"], round(m["score"], 3)) for m in r["moves"]])
    assert {m["move"] for m in r["moves"]} == set(pick) and set(pick) <= set(legal)
    assert r["move"] == "a1a8"
    with pytest.raises(ValueError, match="among"):
        CE.wdl("chess", BACK_RANK, [], among=["a1a9"])


def test_78_26_keys_are_the_exact_position_then_the_units(tmp_path):
    keys = CE.wdl_keys("chess", None, ["e2e4"])
    assert keys[0].startswith("exact:") and keys[1].startswith("units:") and keys[1].endswith(":black") and len(keys) == 2
    along = CE.wdl_keys_along("chess", None, OPENING)
    assert len(along) == len(OPENING) + 1 and along[1] == keys and all(len(a) == 2 for a in along)
    assert along[0][1].endswith(":white")


def test_78_26_a_table_built_from_records_is_found_and_changes_the_estimate(tmp_path):
    # twenty-five games that all end 0-100 (black wins): after 1. e4 the mover is Black, who won every one
    recs = [(None, OPENING[:4], [0, 100]) for _ in range(25)] + [(None, OPENING, None)]
    t = WT.build(recs, "chess")
    st = t["stats"]
    print("OBS 78.26 table:", st)
    assert t["format"] == "wdl-table-1" and st["games"] == 26 and st["skipped_unfinished"] == 1 and st["positions"] == 100
    key = CE.wdl_keys("chess", None, ["e2e4"])[0]
    assert t["rows"][key] == [25.0, 0.0, 0.0]                      # (Black's wins, from Black's view)
    path = tmp_path / "t.json"
    WT.write(t, str(path))
    plain = CE.wdl("chess", None, ["e2e4"], seconds=0.1)
    hit = CE.wdl("chess", None, ["e2e4"], seconds=0.1, table=str(path))
    print("OBS 78.26 table effect: loss", plain["loss"], "->", hit["loss"], "win", plain["win"], "->", hit["win"])
    assert hit["table_key"] == key and hit["table_counts"] == [25.0, 0.0, 0.0] and plain["table_key"] is None
    assert hit["win"] > plain["win"] and hit["sources"] == "playouts+table"
    assert any("table" in e and key in e for e in hit["evidence"])


def test_78_26_a_fractional_outcome_is_split_by_the_same_rule_and_few_games_drop_exact_rows():
    assert WT.split(75) == [0.5, 0.5, 0.0] and WT.split(0) == [0.0, 0.0, 1.0] and WT.split(50) == [0.0, 1.0, 0.0]
    t = WT.build([(None, ["e2e4", "e7e5"], [75, 25])], "chess", min_exact=2)
    assert not any(k.startswith("exact:") for k in t["rows"]) and t["stats"]["dropped_exact"] == 2
    assert t["rows"][CE.wdl_keys("chess", None, [])[1]] == [0.5, 0.5, 0.0]
    assert t["rows"][CE.wdl_keys("chess", None, ["e2e4"])[1]] == [0.0, 0.5, 0.5]    # (Black's view: its 25 is half a draw and half a loss)


def test_78_26_a_table_with_another_format_is_an_error(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"format": "wdl-table-0", "game": "chess", "provenance": "x", "rows": {}}))
    with pytest.raises(ValueError, match="format"):
        CE.wdl("chess", None, ["e2e4"], seconds=0.05, table=str(bad))
    with pytest.raises(ValueError, match="format"):
        CE.best_move("chess", None, ["e2e4"], seconds=0.1, wdl=True, wdl_table=str(bad))


def test_78_26_the_cli_builds_a_table_from_a_wake_log(tmp_path, capsys):
    log = tmp_path / "wake.jsonl"
    rows = [{"kind": "game", "start": None, "moves": OPENING[:4], "outcome": [100, 0], "unfinished": False} for _ in range(3)]
    rows += [{"kind": "game", "start": None, "moves": OPENING, "outcome": None, "unfinished": True}, {"kind": "position", "label": "x"}]
    log.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = tmp_path / "t.json"
    assert WT.main(["chess", "--log", str(log), "--out", str(out)]) == 0
    text = capsys.readouterr().out
    print(text)
    assert "OBS wdl_table chess: 4 games read, 1 skipped as unfinished, 12 positions" in text and "rows kept exact" in text and str(out) in text
    t = json.loads(out.read_text())
    assert t["provenance"] == "own games" and t["format"] == "wdl-table-1"
    assert WT.default_path("chess").endswith("wdl_table_chess.json") and Path(WT.default_path("chess")).is_absolute()
    assert CE.wdl("chess", None, ["e2e4"], seconds=0.05, table=str(out))["table_key"] is None   # (three games: under the table's minimum mass)


def test_78_26_off_the_decision_and_its_trace_are_what_they_were():
    mv, _proved, trace, _n, _fen = CE.best_move("chess", BACK_RANK, [], seconds=0.3, **J)
    mv2, _p2, trace2, _n2, _f2 = CE.best_move("chess", BACK_RANK, [], seconds=0.3, wdl=False, **J)
    assert mv == mv2 == "a1a8" and trace[0] == trace2[0]
    assert not any(t.startswith("wdl:") for t in trace2)
    legal = CE.state("chess", MIDDLE, [])[2]
    a = CE.best_move("chess", MIDDLE, [], seconds=0.2, **J)
    b = CE.best_move("chess", MIDDLE, [], seconds=0.2, wdl=False, wdl_table=None, **J)
    assert a[0] in legal and b[0] in legal and a[2][0] == b[2][0] and not any(t.startswith("wdl:") for t in b[2])


def test_78_26_on_the_trace_carries_the_wdl_lines_and_exactly_one_mode_line():
    for fen in (MIDDLE, LOCKED, "4k3/8/2n5/8/Q7/8/8/4K3 w - - 0 1"):
        mv, _proved, trace, nodes, _fen = CE.best_move("chess", fen, [], seconds=0.4, wdl=True, **J)
        print("OBS 78.26 decide with wdl:", fen, mv, "\n  " + "\n  ".join(t[:200] for t in trace))
        assert mv in CE.state("chess", fen, [])[2]
        modes = [t for t in trace if TRACE_MODE.match(t)]
        assert len(modes) == 1 and any(t.startswith("wdl: win ") for t in trace)
        assert trace.index(modes[0]) > min(i for i, t in enumerate(trace) if t.startswith("wdl: sharpness"))
        if modes[0].startswith("wdl: sharp") or modes[0].startswith("wdl: unknown"):
            assert any(t.startswith("reason:") for t in trace) and "keep reasoning" in modes[0]
        else:
            assert "the probabilities pick among the moves the reading leaves equal" in modes[0] and any("ledger search" in t for t in trace)


def test_78_26_in_a_quiet_position_the_probabilities_pick_among_the_moves_the_ledger_leaves_equal():
    # the start: nothing to win by force and many moves the exact reading cannot tell apart -- the estimate is quiet, the ledger reads its ties, the probabilities pick
    mv, proved, trace, _nodes, _fen = CE.best_move("chess", None, [], seconds=0.3, wdl=True, **J)
    print("OBS 78.26 quiet path:", mv, "\n  " + "\n  ".join(t[:160] for t in trace if t.startswith("wdl:"))[:1500])
    assert not proved and mv in CE.state("chess", None, [])[2]
    assert any(t.startswith("wdl: quiet (") for t in trace) and not any(t.startswith("reason:") for t in trace)
    head = [t for t in trace if t.startswith("wdl: the probabilities pick among ")]
    assert len(head) == 1
    k = int(head[0].split("pick among ")[1].split()[0])
    picked = [t.split()[2] for t in trace[trace.index(head[0]) + 1:] if t.startswith("wdl: move ")]
    assert k > 1 and len(picked) == k and picked[0] == mv


def test_78_26_the_playout_policy_is_named_and_a_wrong_name_is_an_error():
    careful = CE.wdl("chess", MIDDLE, [], seconds=0.05, **J)
    greedy = CE.wdl("chess", MIDDLE, [], seconds=0.05, rollout="greedy", **J)
    assert any("rollout careful" in t for t in careful["evidence"]) and any("rollout greedy" in t for t in greedy["evidence"])
    assert careful["n"] >= 30 and greedy["n"] > careful["n"]          # the two-ply look costs more a playout
    with pytest.raises(ValueError):
        CE.wdl("chess", MIDDLE, [], seconds=0.05, rollout="clever", **J)


def test_78_26_a_decision_with_wdl_keeps_to_its_clock():
    for fen in (MIDDLE, LOCKED):
        t0 = time.perf_counter()
        CE.best_move("chess", fen, [], seconds=0.5, wdl=True, **J)
        took = time.perf_counter() - t0
        print(f"OBS 78.26 clock: {took:.3f} s of 0.5 s")
        assert took < 0.5 + 0.15


def test_78_26_solve_batch_with_wdl_returns_rows_and_logs_the_steps(tmp_path):
    log = tmp_path / "trace.jsonl"
    path = tmp_path / "t.json"
    WT.write(WT.build([(None, OPENING[:4], [0, 100]) for _ in range(25)], "chess"), str(path))
    rows = CE.solve_batch([("chess", MIDDLE, []), ("chess", LOCKED, []), ("chess", None, OPENING[:2])], 0.3, wdl=True, wdl_table=str(path), trace_path=str(log), threads=2, **J)
    assert all(r[4] is None and r[0] is not None for r in rows)
    logged = [json.loads(x) for x in log.read_text().splitlines()]
    assert all(any(s["operator"] == "wdl" for s in row["steps"]) for row in logged)
    assert any(f["relation"] == "sharpness" for f in logged[0]["facts"])
    off = CE.solve_batch([("chess", MIDDLE, [])], 0.2, **J)
    assert off[0][4] is None


def test_78_26_the_layer_crates_name_no_chess_object():
    chess_words = r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b"
    p = Path(__file__).resolve().parents[1] / "rust" / "engine" / "planning" / "src" / "wdl.rs"
    assert re.search(chess_words, p.read_text(), flags=re.I) is None
