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


def run(rows: list[dict], engine, judge_depth: int = 6, budget: int = 120, full: bool = True, log=print, zone_first: bool = False):
    """Each puzzle once, no retries: the position after the setup move is ranked and the top move compared with the
    solution's first move; with `full`, the opponent's reply from the solution is played and the next move of ours
    is ranked in turn, to the end of the line -- a puzzle is FULLY solved only when every move of ours is found."""
    memo = Memo()
    results = []
    t0 = time.perf_counter()
    for k, row in enumerate(rows, 1):
        b = chess.Board(row["FEN"])
        moves = row["Moves"].split()
        b.push_uci(moves[0])                                   # the setup move: the solver is now to move
        sol = chess.Move.from_uci(moves[1])
        try:
            sc = score_moves(b, memo, engine, budget=budget, engine_depth=judge_depth, zone_first=zone_first)
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
        ours_total = (len(moves) - 1 + 1) // 2                  # our moves in the solution: indices 1, 3, 5, ...
        res["moves_needed"] = ours_total; res["moves_found"] = 1 if res["solved"] else 0; res["failed_at"] = None if res["solved"] else 1
        if full and res["solved"]:
            bb = b.copy(stack=False); bb.push(sol)
            i = 2
            while i + 1 <= len(moves) - 1:
                bb.push_uci(moves[i])                          # their reply from the solution
                nxt = chess.Move.from_uci(moves[i + 1])
                try:
                    sc2 = score_moves(bb, memo, engine, budget=budget, engine_depth=judge_depth, zone_first=zone_first)
                except Exception:
                    res["failed_at"] = res["moves_found"] + 1; break
                order2 = sorted(bb.legal_moves, key=lambda m: -sc2.get(m, (0.0, ""))[0])
                t2 = order2[0]; s2 = sc2.get(t2, (0.0, ""))[0]
                if t2 == nxt and s2 > 0:
                    res["moves_found"] += 1
                    bb.push(nxt); i += 2
                else:
                    res["failed_at"] = res["moves_found"] + 1
                    res["fail_sol"] = bb.san(nxt); res["fail_top"] = bb.san(t2); res["fail_why"] = sc2.get(t2, (0.0, ""))[1]
                    break
        res["fully"] = res["moves_found"] == res["moves_needed"]
        results.append(res)
        if k % 10 == 0:
            log(f"[gauge] {k}/{len(rows)} puzzles, {time.perf_counter()-t0:.0f}s, solved {sum(r['solved'] for r in results)}, fully {sum(r.get('fully', False) for r in results)}, named {sum(r['named'] for r in results)}", flush=True)
    return results, time.perf_counter() - t0


def report(results: list[dict], dt: float) -> str:
    n = len(results)
    out = [f"[gauge] {n} puzzles in {dt:.0f}s ({dt / max(1, n):.1f}s each)"]
    solved = sum(r["solved"] for r in results); named = sum(r["named"] for r in results); silent = sum(r["silent"] for r in results)
    top3 = sum(1 for r in results if 0 < r["rank"] <= 3 and not r["silent"])
    out.append(f"solved (top-1) {solved}/{n} ({solved / n:.0%}); in our top-3 {top3}/{n}; solution named at all {named}/{n} ({named / n:.0%}); silent {silent}/{n}")
    fully = sum(1 for r in results if r.get("fully"))
    multi = [r for r in results if r.get("moves_needed", 1) > 1]
    if multi:
        out.append(f"FULL LINE: fully solved {fully}/{n} ({fully / n:.0%}); puzzles with more than one move of ours {len(multi)}: fully solved "
                   f"{sum(1 for r in multi if r.get('fully'))}/{len(multi)}, first move found {sum(1 for r in multi if r['solved'])}/{len(multi)}; "
                   f"failures after a correct first move: " + ", ".join(f"at move {k}: {v}" for k, v in sorted(Counter(r['failed_at'] for r in results if r['solved'] and not r.get('fully')).items())))
        by_len = defaultdict(list)
        for r in results:
            by_len[r.get("moves_needed", 1)].append(r)
        out.append("by moves of ours in the solution: " + "; ".join(f"{k} moves: {sum(1 for r in v if r.get('fully'))}/{len(v)} fully" for k, v in sorted(by_len.items())))
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
    later = [r for r in results if r["solved"] and not r.get("fully")]
    if later:
        out.append("examples of a correct first move and a miss later in the line (solution | ours):")
        for r in later[:8]:
            out.append(f"  {r['id']} r{r['rating']} {' '.join(r['themes'][:3])}: move {r['failed_at']}: {r.get('fail_sol')} | {r.get('fail_top')} ({r.get('fail_why') or 'nothing'})")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100); ap.add_argument("--max-rating", type=int, default=1800); ap.add_argument("--min-rating", type=int, default=0)
    ap.add_argument("--seed", type=int, default=3); ap.add_argument("--judge", type=int, default=6); ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--db", default=str(paths.assets_dir() / "lichess" / "lichess_db_puzzle.csv.zst"))
    ap.add_argument("--json", default=""); ap.add_argument("--first-only", action="store_true", help="check the first move only")
    ap.add_argument("--shard", default="", help="i/n: this process takes every n-th puzzle starting at i (parallel runs)")
    ap.add_argument("--cache", default="", help="a private engine cache file for this process")
    ap.add_argument("--zone-first", action="store_true", help="the focus prior: targets in the hottest sub-game first, the rest at half budget")
    a = ap.parse_args(argv)
    from .engine import Engine
    rows = sample(a.db, a.n, a.seed, a.max_rating, a.min_rating)
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        rows = rows[i::n]
    with Engine(depth=12, cache_path=a.cache or None) as eng:
        results, dt = run(rows, eng, a.judge, a.budget, full=not a.first_only, zone_first=a.zone_first)
    print(report(results, dt), flush=True)
    if a.json:
        import json
        with open(a.json, "w") as f:
            json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
