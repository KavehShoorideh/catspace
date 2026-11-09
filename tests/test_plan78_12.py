"""PLAN 78.12 THE SHARED INTERFACE AND THE BATCH SOLVER (docs/ANALYST_PLAN.md 78.12). The Rust side (Operator / Controller / Goal / Outcome, the fixed controller equal
to the 78.11 decision) is tested in rust/engine/decision/tests; here the Python entry: one call decides many positions, in order, with the lock released, and the log
says what each decision did."""
import json
import threading
import time
from pathlib import Path

import cs_engine as CE

MATE_IN_ONE = "6k1/5ppp/8/8/8/8/5PPP/R3K3 w - - 0 1"
WON_QUEEN = "4k3/8/2n5/8/Q7/8/8/4K3 w - - 0 1"
OPENING = ("chess", None, ["e2e4", "e7e5"])


def _items():
    return [("chess", MATE_IN_ONE, []), ("chess", WON_QUEEN, []), OPENING, ("kingofthehill", None, ["e2e4"]), ("3check", None, [])]


def test_78_12_one_call_decides_every_position_in_order():
    items = _items()
    rows = CE.solve_batch(items, 0.3, threads=2)
    assert len(rows) == len(items)
    mv, proved, nodes, seconds, err = rows[0]
    assert (mv, proved, err) == ("a1a8", True, None) and nodes > 0
    for (mv, proved, nodes, seconds, err), (v, fen, moves) in zip(rows, items):
        assert err is None and mv is not None and nodes > 0 and seconds > 0, (v, fen, moves)
    single = CE.best_move("chess", MATE_IN_ONE, [], 0.3)
    assert single[0] == rows[0][0] and single[1] is True


def test_78_12_a_bad_position_is_its_own_error_row_and_the_batch_goes_on():
    rows = CE.solve_batch([("chess", "not a fen", []), ("chess", MATE_IN_ONE, []), ("chess", None, ["e2e5"]), ("nosuchgame", None, [])], 0.2, threads=2)
    assert [r[4] is not None for r in rows] == [True, False, True, True]
    assert rows[1][0] == "a1a8"


def test_78_12_the_log_names_the_snapshot_and_every_step(tmp_path: Path):
    log = tmp_path / "batch.jsonl"
    rows = CE.solve_batch(_items(), 0.3, snapshot="store-test-1", trace_path=str(log), threads=2, prior=True)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    assert len(lines) == len(rows) == 5
    assert [r["index"] for r in lines] == list(range(5))
    assert all(r["snapshot"] == "store-test-1" and r["kind"] == "position" for r in lines)
    first = lines[0]
    assert first["move"] == "a1a8" and first["proved"] is True
    assert [s["operator"] for s in first["steps"]] == ["forcing"] and first["steps"][0]["status"] == "Proved"
    assert first["steps"][0]["goal"].startswith("win(") and first["steps"][0]["nodes"] > 0
    ops = [s["operator"] for s in lines[2]["steps"]]
    assert ops[:2] == ["forcing", "ledger"], ops
    print(f"OBS 78.12 the log: {len(lines)} rows, steps per row {[len(r['steps']) for r in lines]}, operators of the opening {ops}")
    CE.solve_batch(_items()[:1], 0.2, snapshot="store-test-2", trace_path=str(log))
    assert len(log.read_text().splitlines()) == 6                      # appended, never overwritten


def test_78_12_the_batch_spreads_over_threads_and_releases_the_lock():
    items = [("chess", None, ["e2e4", "e7e5"])] * 8
    t0 = time.perf_counter()
    CE.solve_batch(items, 0.4, threads=1)
    serial = time.perf_counter() - t0
    t0 = time.perf_counter()
    CE.solve_batch(items, 0.4, threads=4)
    parallel = time.perf_counter() - t0
    print(f"OBS 78.12 eight positions at 0.4 s each: one thread {serial:.2f} s, four threads {parallel:.2f} s")
    assert parallel < serial * 0.75
    ticks = []
    stop = threading.Event()

    def tick():
        while not stop.is_set():
            ticks.append(1)
            time.sleep(0.01)
    th = threading.Thread(target=tick)
    th.start()
    CE.solve_batch(items[:4], 0.4, threads=2)
    stop.set()
    th.join()
    print(f"OBS 78.12 a Python thread ticked {len(ticks)} times during a 0.8 s batch")
    assert len(ticks) > 20


def test_78_12_the_valuation_and_the_snapshot_are_the_callers():
    w = {"q": 9, "r": 5, "b": 3, "n": 3, "p": 1}
    rows = CE.solve_batch([("chess", WON_QUEEN, [])], 0.3, worth=w, valuation="test table", ours_first=True)
    assert rows[0][0] == "a4c6"                                         # the queen takes the knight
