"""gauge_budget.py -- the planner's signal INTO the tactics, measured: the search budget from the expected sharpness
(Kaveh 2026-09-11: "some signal from the tactics to the planner and back"; 2026-07-30: "deepen search only where the
plan is unreliable; the budget falls automatically as planning improves").

On held-out positions the rules ranking runs four ways: the FULL budget on every position, NORMAL and SHALLOW
constant budgets, and GATED -- the budget the planner picks from its predicted sharpness at +10. Reported per way:
agreement with the engine's best move (all, quiet, capture), silence, and seconds per position; for the gated way,
the split by level. The claim to test: gated keeps full's agreement at a fraction of full's cost.

    python -m catspace.incubator.analyst.gauge_budget --data data/derived/knowledge --n 300 --shard 0/6
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import chess

from .memo import Memo
from .rank import score_moves, move_kind
from .planner import Planner, budget_for, DEFAULT_CKPT
from . import knowledge_net as K

WAYS = {"full": dict(combo_cap=1500, budget=120, mate_budget=1500), "normal": dict(combo_cap=600, budget=60, mate_budget=800),
        "shallow": dict(combo_cap=200, budget=30, mate_budget=300)}


def run(data_dir: str, n: int, shard: str, ckpt: str, judge: int = 6, log=print):
    from .engine import Engine
    i, k = (int(x) for x in shard.split("/"))
    data = K.Data(data_dir, meta_only=True)
    ids = [int(x) for x in data.ids("test")][:n][i::k]
    pl = Planner(ckpt, data_dir)
    out = Path("data/derived/budget"); out.mkdir(parents=True, exist_ok=True)
    eng = Engine(depth=12, cache_path=str(out / f"engine_{i}.sqlite")); memo = Memo()
    rows = []; t0 = time.perf_counter()
    for nn, idx in enumerate(ids, 1):
        m = data.meta[idx]; board = chess.Board(m["fen"])
        if not m["best"]:
            continue
        best = chess.Move.from_uci(m["best"])
        row = dict(i=idx, fen=m["fen"], best=m["best"], best_kind=m["best_kind"])
        try:
            t = time.perf_counter(); plan = pl.plan(board, memo=memo); row["plan_s"] = time.perf_counter() - t
            row["sharp10"] = plan.sharp10; row["level"] = plan.budget["level"]
            ways = dict(WAYS); ways["gated"] = {k_: v for k_, v in plan.budget.items() if k_ != "level"}
            for way, params in ways.items():
                t = time.perf_counter()
                sc = score_moves(board, memo, eng, engine_depth=judge, **params)
                top = max(board.legal_moves, key=lambda mv: sc.get(mv, (0.0, ""))[0]); s = sc.get(top, (0.0, ""))[0]
                row[way] = dict(hit=bool(top == best and s > 0), silent=bool(s <= 0), seconds=time.perf_counter() - t)
        except Exception as ex:
            log(f"[budget] {idx} error {ex}", flush=True); continue
        rows.append(row)
        if nn % 5 == 0:
            log(f"[budget {shard}] {nn}/{len(ids)} positions, {time.perf_counter() - t0:.0f}s", flush=True)
    eng.close()
    with open(out / f"rows_{i}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    log(f"[budget {shard}] done {len(rows)} in {time.perf_counter() - t0:.0f}s", flush=True)


def report(dir_: str = "data/derived/budget") -> str:
    rows = []
    for f in sorted(Path(dir_).glob("rows_*.jsonl")):
        rows.extend(json.loads(l) for l in open(f))
    n = len(rows); out = [f"[budget] {n} positions (plan {sum(r['plan_s'] for r in rows) / max(1, n):.2f}s each)"]
    out.append(f"  {'way':<9}{'top-1':>7}{'quiet':>7}{'capture':>9}{'silent':>8}{'s/pos':>7}")
    for way in ("full", "normal", "shallow", "gated"):
        rs = [r for r in rows if way in r]
        q = [r for r in rs if r["best_kind"] == "quiet"]; c = [r for r in rs if r["best_kind"] == "capture"]
        out.append(f"  {way:<9}{sum(r[way]['hit'] for r in rs) / max(1, len(rs)):>7.1%}{sum(r[way]['hit'] for r in q) / max(1, len(q)):>7.1%}{sum(r[way]['hit'] for r in c) / max(1, len(c)):>9.1%}"
                   f"{sum(r[way]['silent'] for r in rs) / max(1, len(rs)):>8.1%}{sum(r[way]['seconds'] for r in rs) / max(1, len(rs)):>7.1f}")
    out.append("  gated, by the planner's level (full's hit rate on the same positions in brackets):")
    for lvl in ("deep", "normal", "shallow"):
        rs = [r for r in rows if r.get("level") == lvl]
        if rs:
            out.append(f"    {lvl:<8} n={len(rs):<4} sharp10 mean {sum(r['sharp10'] for r in rs) / len(rs):.2f}  gated {sum(r['gated']['hit'] for r in rs) / len(rs):.1%} [full {sum(r['full']['hit'] for r in rs) / len(rs):.1%}]  "
                       f"{sum(r['gated']['seconds'] for r in rs) / len(rs):.1f}s [full {sum(r['full']['seconds'] for r in rs) / len(rs):.1f}s]; engine best forcing {sum(1 for r in rs if r['best_kind'] != 'quiet') / len(rs):.0%}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="data/derived/knowledge"); ap.add_argument("--n", type=int, default=300); ap.add_argument("--shard", default="0/1")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT); ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if a.report:
        print(report())
    else:
        run(a.data, a.n, a.shard, a.ckpt)


if __name__ == "__main__":
    main()
