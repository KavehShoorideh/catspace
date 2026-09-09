"""gauge_epd.py -- the reasoning's ranking on the classic EPD test suites (Kaveh 2026-09-08: "do the puzzle benchmarks
the rule-based systems did so I can compare").

Reinfeld's Win at Chess (WAC, 300 tactical positions, 1958; PARADISE 1980 reported 89 of the first 100) and the
Bratko-Kopec test (24 positions, 12 tactical and 12 positional, 1982). Each position carries one or more best
moves ("bm"); a position is solved when our top-ranked move, one try, is one of them with a positive score.
Same ranking as the Lichess gauge (rank.score_moves), no engine in the ranking itself.

    python -m catspace.incubator.analyst.gauge_epd assets/suites/WAC.epd [--first 100] [--shard i/n --cache file]
"""
from __future__ import annotations

import argparse
import time
from collections import Counter, defaultdict

import chess

from .memo import Memo
from .rank import score_moves, move_kind


def load(path: str) -> list[tuple[str, chess.Board, list[chess.Move]]]:
    out = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            board, ops = chess.Board.from_epd(line)
        except Exception:
            continue
        bms = ops.get("bm") or []
        out.append((str(ops.get("id", len(out) + 1)), board, list(bms)))
    return out


def run(items, engine, judge_depth: int = 6, budget: int = 100, log=print):
    memo = Memo(); results = []; t0 = time.perf_counter()
    for k, (pid, b, bms) in enumerate(items, 1):
        try:
            sc = score_moves(b, memo, engine, budget=budget, engine_depth=judge_depth)
        except Exception as ex:
            log(f"[epd] {pid} error {ex}", flush=True); continue
        order = sorted(b.legal_moves, key=lambda m: -sc.get(m, (0.0, ""))[0])
        top = order[0]; top_s, top_why = sc.get(top, (0.0, ""))
        best_rank = min((order.index(m) + 1 for m in bms if m in order), default=0)
        named = any(sc.get(m, (0.0, ""))[0] > 0 for m in bms)
        results.append(dict(id=pid, solved=(top in bms and top_s > 0), matched=top in bms, named=named, silent=top_s <= 0, rank=best_rank,
                            kind=move_kind(b, bms[0]) if bms else "?", bm=" ".join(b.san(m) for m in bms), top=b.san(top), top_why=top_why, fen=b.fen()))
        if k % 10 == 0:
            log(f"[epd] {k}/{len(items)} positions, {time.perf_counter()-t0:.0f}s, solved {sum(r['solved'] for r in results)}", flush=True)
    return results, time.perf_counter() - t0


def report(results, dt, name: str) -> str:
    n = len(results); s = sum(r["solved"] for r in results)
    out = [f"[epd] {name}: {n} positions in {dt:.0f}s ({dt / max(1, n):.1f}s each)",
           f"solved (top-1, positive score) {s}/{n} ({s / n:.0%}); top move is a best move regardless of score {sum(r['matched'] for r in results)}/{n}; "
           f"best move in our top-3 {sum(1 for r in results if 0 < r['rank'] <= 3)}/{n}; named {sum(r['named'] for r in results)}/{n}; silent {sum(r['silent'] for r in results)}/{n}"]
    g = defaultdict(list)
    for r in results:
        g[r["kind"]].append(r)
    out.append("by the best move's kind: " + "; ".join(f"{k} {sum(r['solved'] for r in v)}/{len(v)}" for k, v in sorted(g.items(), key=lambda kv: -len(kv[1]))))
    wrong = [r for r in results if not r["solved"]]
    out.append("missed: " + ", ".join(f"{r['id']} ({r['bm']} | {r['top']}{' silent' if r['silent'] else ''})" for r in wrong))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("epd"); ap.add_argument("--first", type=int, default=0); ap.add_argument("--judge", type=int, default=6); ap.add_argument("--budget", type=int, default=100)
    ap.add_argument("--shard", default=""); ap.add_argument("--cache", default=""); ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    items = load(a.epd)
    if a.first:
        items = items[:a.first]
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/")); items = items[i::n]
    from .engine import Engine
    with Engine(depth=12, cache_path=a.cache or None) as eng:
        results, dt = run(items, eng, a.judge, a.budget)
    print(report(results, dt, a.epd.split("/")[-1]), flush=True)
    if a.json:
        import json
        json.dump(results, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
