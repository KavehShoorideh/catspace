"""PLAN 78.59: the heads from human games. Small tests: the record round trip with two sources, the filters of the sample, the no-annotation check, a source scan.

Run: .venv/bin/python -m pytest -q tests/test_plan78_59_human_heads.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from catspace.incubator.analyst.discovery import estimator_fit as EF
from catspace.incubator.analyst.discovery import estimator_heads as EH
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import human_heads_result as HR

ROOT = Path(__file__).resolve().parents[1]
SCAN_WORDS = (r"\b(king|queen|rook|bishop|knight|pawn|castl\w*|promot\w*|en passant|check|mate|stalemate|uci|fen|san|chess\w*|shakmaty|[a-h][1-8])\b")
MODULES = ("human_heads_data", "human_heads_result", "human_heads_arrival", "human_heads_route")


def synthetic(n: int, w: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.integers(-3, 4, size=(n, len(w))).astype(float)
    z = np.column_stack([x @ w, np.zeros(n), -(x @ w)])
    p = np.exp(z - z.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(3, p=r) for r in p])
    return x, y


def record_with_two_sources():
    x, y = synthetic(3000, np.array([0.3, 0.8]), 1)
    m = EF.fit_lr(x, y, 10.0)
    selfplay = EF.model_record(m, 10.0)
    xr, yr = synthetic(3000, np.array([0.5, 0.1, 0.9]), 2)
    centre, scale = xr.mean(axis=0), xr.std(axis=0)
    mh = EF.fit_lr((xr - centre) / scale, yr, 10.0)
    human = HR.model_head(mh, 10.0, centre, scale, ["a", "b", "c"], {"rows": 3000}, "synthetic")
    rec = {"kinds": ["a", "b"], "heads": {"result": {"type": "probability", "model": selfplay, "source": "self-play only"}, "kind_baseline": {"type": "terms", "terms": [0.3, 0.8], "source": "self-play only"},
                                           "result_human_RK": human}}
    return rec, x, xr, mh, centre, scale


def test_the_record_round_trip_holds_two_sources_and_the_self_play_head_is_not_overwritten():
    rec, x, xr, mh, centre, scale = record_with_two_sources()
    before = json.dumps(rec["heads"]["result"], sort_keys=True)
    back = json.loads(json.dumps(rec))
    assert json.dumps(back["heads"]["result"], sort_keys=True) == before
    assert back["heads"]["result"]["source"] == "self-play only" and back["heads"]["result_human_RK"]["source"] == "human games"
    p_self = EH.head_predict(back, "result", x)
    p_human = HR.head_probs(back["heads"]["result_human_RK"], xr)
    direct = EF.probs(mh, (xr - centre) / scale)
    print(f"OBS 78.59 record round trip: largest difference between the record's human head and the model {float(np.abs(p_human - direct).max()):.2e}; sources {[h['source'] for h in back['heads'].values()]}")
    assert np.abs(p_human - direct).max() < 1e-9
    assert np.allclose(p_self.sum(axis=1), 1.0) and np.allclose(p_human.sum(axis=1), 1.0)


def test_adding_a_head_with_a_taken_name_is_refused_by_the_writer():
    text = (ROOT / "catspace/incubator/analyst/discovery/human_heads_result.py").read_text()
    assert "assert k not in rec[\"heads\"]" in text


def synthetic_shard(path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    ok = "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5 c2c3 g8f6 d2d4 e5d4 c3d4 c5b4 b1c3 d7d5 e4d5 f6d5 e1g1 c8e6 c1g5 h7h6"
    rows = [
        (1, 1500, 1600, 1, 40, "Normal", ok),                              # kept
        (2, 1500, 1600, -1, 40, "Time forfeit", ok),                       # dropped: not ended on the board
        (3, 1500, 1600, 1, 10, "Normal", ok),                              # dropped: too short
        (4, 1500, 1600, 1, 40, "Normal", ok + " { [%eval 0.3] }"),         # dropped: annotated
        (5, 1500, 1600, 1, 40, "Normal", ok + " { [%clk 0:02:59] }"),      # dropped: annotated
        (6, 1900, 1800, 0, 300, "Normal", ok),                             # dropped: too long
        (7, 1900, 1800, 0, 60, "Normal", ok),                              # kept, a draw
    ]
    t = pa.table({"game_id": [r[0] for r in rows], "white_elo": pa.array([r[1] for r in rows], pa.int16()), "black_elo": pa.array([r[2] for r in rows], pa.int16()),
                  "result": pa.array([r[3] for r in rows], pa.int8()), "n_plies": pa.array([r[4] for r in rows], pa.int32()), "termination": [r[5] for r in rows], "moves": [r[6] for r in rows]})
    pq.write_table(t, path)


def test_the_filters_keep_only_games_decided_on_the_board_and_unannotated(tmp_path, monkeypatch):
    p = tmp_path / "records_00000.parquet"
    synthetic_shard(p)
    monkeypatch.setattr(HD, "shard_path", lambda i: p)
    games, c = HD.sample_shard(0, 100, 10)
    print(f"OBS 78.59 filters: {c}; kept ids {[g['id'] for g in games]}")
    assert c["games"] == 7 and c["annotated"] == 2 and c["not_normal"] == 1 and c["out_of_size"] == 2 and c["qualifying"] == 2
    assert sorted(g["id"] for g in games) == ["0:1", "0:7"]
    by = {g["id"]: g for g in games}
    assert by["0:1"]["y"] == 0 and by["0:7"]["y"] == 1 and by["0:7"]["band"] == 3 and by["0:1"]["band"] == 2
    assert all(len(g["moves"]) >= 20 for g in games)
    assert sum(g["arrival"] for g in games) == 2


def test_the_annotation_check_drops_comments_and_evaluations():
    for good in ("e2e4 e7e5 g1f3", "e7e8q a1a2"):
        assert HD.MOVE_WORD.match(good)
    for bad in ("e2e4 { [%eval 0.2] } e7e5", "e2e4 e7e5 { [%clk 0:03:00] }", "e2e4 +0.3 e7e5", "e2e4 e7e5 1-0", "e4 e5", "e2e4  e7e5", ""):
        assert not HD.MOVE_WORD.match(bad), bad


def test_no_player_id_and_no_second_directory_are_read():
    text = (ROOT / "catspace/incubator/analyst/discovery/human_heads_data.py").read_text()
    code = re.sub(r'"""[\s\S]*?"""', "", text)
    assert "white_id" not in code and "black_id" not in code
    assert "COLUMNS = " in code and "white_id" not in HD.COLUMNS and "black_id" not in HD.COLUMNS
    for m in MODULES:
        src = (ROOT / f"catspace/incubator/analyst/discovery/{m}.py").read_text()
        stripped = re.sub(r'"""[\s\S]*?"""', "", src)
        assert not re.search(r"lichess_dir|data/lichess|stockfish|tablebase|puzzle", stripped, flags=re.I), m
        assert "rules/learned" not in stripped and "data/derived" not in stripped and "derived(" not in stripped, m


def test_the_sources_name_no_object_of_any_game():
    for m in MODULES:
        text = (ROOT / f"catspace/incubator/analyst/discovery/{m}.py").read_text()
        code = "\n".join(ln.split("#")[0] for ln in text.splitlines())
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        code = re.sub(r'"(?:\\.|[^"\\])*"', '""', code)
        found = re.findall(SCAN_WORDS, code, flags=re.I)
        assert not found, (m, found[:5])
