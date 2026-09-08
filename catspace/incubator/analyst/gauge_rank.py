"""gauge_rank.py -- the reasoning's ranking against Lichess puzzles (Kaveh 2026-09-07: "a hundred random puzzles;
where are we strong, where weak").

A puzzle's FEN is the position before the opponent's setup move; the solution starts after it. We rank every
legal move by the reasoning alone (rank.score_moves) and compare with the solution's first move: solved (top-1),
named (the move has a positive score), its rank, and silence (no move scored). Split by theme, by the kind of the
solution move, by rating band, and by the reason our top move carried when we were wrong.

    python -m catspace.incubator.analyst.gauge_rank --n 100 [--max-rating 1800 --seed 3 --judge 6]
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import time
from collections import Counter, defaultdict

import chess
import zstandard

from ...io import paths
from .memo import Memo
from .rank import score_moves, move_kind


def sample(path: str, n: int, seed: int, max_rating: int, min_rating: int = 0, scan: int = 60000) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    with open(path, "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        for i, row in enumerate(reader):
            if i >= scan:
                break
            r = int(row["Rating"])
            if min_rating <= r <= max_rating:
                rows.append(row)
    rng.shuffle(rows)
    return rows[:n]


def run(rows: list[dict], engine, judge_depth: int = 6, budget: int = 120, log=print):
    memo = Memo()
    results = []
    t0 = time.perf_counter()
    for k, row in enumerate(rows, 1):
        b = chess.Board(row["FEN"])
        moves = row["Moves"].split()
        b.push_uci(moves[0])                                   # the setup move: the solver is now to move
        sol = chess.Move.from_uci(moves[1])
        try:
            sc = score_moves(b, memo, engine, budget=budget, engine_depth=judge_depth)
        except Exception as ex:
            log(f"[gauge] {row['PuzzleId']} error {ex}", flush=True)
            continue
        legal = list(b.legal_moves)
        order = sorted(legal, key=lambda m: -sc.get(m, (0.0, ""))[0])
        top = order[0]; top_s, top_why = sc.get(top, (0.0, ""))
        sol_s, sol_why = sc.get(sol, (0.0, ""))
        rank = order.index(sol) + 1 if sol in order else 0
        res = dict(id=row["PuzzleId"], rating=int(row["Rating"]), themes=row["Themes"].split(), kind=move_kind(b, sol),
                   solved=(top == sol and top_s > 0), named=sol_s > 0, rank=rank, silent=top_s <= 0,
                   sol=b.san(sol), sol_why=sol_why, top=b.san(top), top_why=top_why, fen=b.fen(), n_legal=len(legal))
        results.append(res)
        if k % 10 == 0:
            log(f"[gauge] {k}/{len(rows)} puzzles, {time.perf_counter()-t0:.0f}s, solved {sum(r['solved'] for r in results)}, named {sum(r['named'] for r in results)}", flush=True)
    return results, time.perf_counter() - t0


def report(results: list[dict], dt: float) -> str:
    n = len(results)
    out = [f"[gauge] {n} puzzles in {dt:.0f}s ({dt / max(1, n):.1f}s each)"]
    solved = sum(r["solved"] for r in results); named = sum(r["named"] for r in results); silent = sum(r["silent"] for r in results)
    top3 = sum(1 for r in results if 0 < r["rank"] <= 3 and not r["silent"])
    out.append(f"solved (top-1) {solved}/{n} ({solved / n:.0%}); in our top-3 {top3}/{n}; solution named at all {named}/{n} ({named / n:.0%}); silent {silent}/{n}")
    def table(title, key):
        groups = defaultdict(list)
        for r in results:
            for g in (key(r) if isinstance(key(r), (list, set, tuple)) else [key(r)]):
                groups[g].append(r)
        out.append(f"{title:<28}{'n':>5}{'solved':>8}{'named':>8}{'silent':>8}")
        for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            if len(rs) < 3:
                continue
            m = len(rs)
            out.append(f"{str(g):<28}{m:>5}{sum(r['solved'] for r in rs) / m:>8.0%}{sum(r['named'] for r in rs) / m:>8.0%}{sum(r['silent'] for r in rs) / m:>8.0%}")
    table("by solution move", lambda r: r["kind"])
    table("by rating band", lambda r: f"{(r['rating'] // 400) * 400}-{(r['rating'] // 400) * 400 + 399}")
    table("by theme", lambda r: r["themes"])
    # what we said instead, when wrong
    wrong = [r for r in results if not r["solved"]]
    why = Counter(("silent" if r["silent"] else r["top_why"].split(":")[0].split(";")[0]) for r in wrong)
    out.append("when wrong, our top move's reason: " + ", ".join(f"{w} {c}" for w, c in why.most_common(8)))
    out.append("examples (solution | ours):")
    for r in wrong[:12]:
        out.append(f"  {r['id']} r{r['rating']} {' '.join(r['themes'][:3])}: {r['sol']} ({r['sol_why'] or 'unnamed'}) | {r['top']} ({r['top_why'] or 'nothing'})")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--max-rating", type=int, default=1800); ap.add_argument("--min-rating", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3); ap.add_argument("--judge", type=int, default=6); ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--db", default=str(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst"))
    ap.add_argument("--json", default="")
    a = ap.parse_args(argv)
    from .engine import Engine
    rows = sample(a.db, a.n, a.seed, a.max_rating, a.min_rating)
    with Engine(depth=12) as eng:
        results, dt = run(rows, eng, a.judge, a.budget)
    print(report(results, dt), flush=True)
    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
