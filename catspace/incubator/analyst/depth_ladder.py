"""THE DEPTH LADDER -- labels for the stop-or-continue decider (Kaveh 2026-09-30: "go very deep on a lot of positions,
see which depth changed the result, use that as training labels, then predict what the next depth should be" --
deep labels first, reinforcement learning after; and "learn from Stockfish how deep it goes" -- sf_depth: the misses
secure their gain near ply 16).

For every solver step of fresh training puzzles, the zone search deepens all the candidates together through the
ladder (4, 6, 8, 10, 12 plies by default), with no margin pruning (a late payer is searched too), a generous budget per
candidate per depth, and the same planner goals as in play; every candidate's value at every depth is kept (None where
its budget ran out). The DECISION at a depth is what the player would choose stopping there (each candidate at its
last completed depth, as rank() does). The LABEL at depth d: the decision there differs from the decision at the
deepest depth -- going on would have changed the answer.

    python -m catspace.incubator.analyst.depth_ladder collect --n 10 --offset 900        # a pilot
    python -m catspace.incubator.analyst.depth_ladder report --n 10 --offset 900
"""
from __future__ import annotations

import argparse
import json
import os
import time

import chess

from catspace.io import paths

LADDER = (4, 6, 8, 10, 12)
BUDGET = 150000


def _position(b: chess.Board, pid: str, step: int, ladder: tuple, budget: int) -> dict | None:
    from .fates import clear_caches
    from .frame_planner import MaterialPlanner
    from .frame_player import candidates as _candidates
    from .zone_search import ZoneSearch
    from dataclasses import replace
    clear_caches()
    moves = _candidates(b)
    if len(moves) < 2:
        return None
    plan = MaterialPlanner().plan(b, moves)
    spec = replace(plan.spec, depths=ladder, budget=budget, margin=1e9)
    zs = ZoneSearch(spec=spec)
    t0 = time.perf_counter()
    zs.rank(b, moves)
    return dict(pid=pid, step=step, fen=b.fen(), moves=[m.uci() for m in moves],
                history={m.uci(): [[d, v, n] for d, v, n in zs.history.get(m, [])] for m in moves},
                nodes=zs.total, secs=round(time.perf_counter() - t0, 1))


def _puzzle(args) -> list:
    row, ladder, budget = args
    b = chess.Board(row["FEN"]); mv = row["Moves"].split(); b.push_uci(mv[0]); sol = mv[1:]
    out = []
    for i in range(0, len(sol), 2):
        if b.is_game_over():
            break
        r = _position(b, row["PuzzleId"], i // 2, ladder, budget)
        if r is not None:
            r["sol"] = sol[i]; out.append(r)
        b.push_uci(sol[i])
        if i + 1 < len(sol):
            b.push_uci(sol[i + 1])
    return out


def collect(n: int, offset: int, workers: int = 10, resume: bool = False, ladder=LADDER, budget: int = BUDGET, log=print) -> str:
    import multiprocessing as mp
    from .frame_puzzles import fresh_rows
    t0 = time.perf_counter()
    rows = fresh_rows(n, 997, offset)
    out = paths.derived(f"depth_ladder_{n}_{offset}.jsonl")
    done = set()
    if resume and os.path.exists(out):
        good = []
        for line in open(out):
            try:
                d = json.loads(line); good.append(line if line.endswith("\n") else line + "\n"); done.add(d["pid"])
            except json.JSONDecodeError:
                break
        open(out, "w").writelines(good)
        rows = [r for r in rows if r["PuzzleId"] not in done]
        log(f"[depth_ladder] resume: {len(done)} puzzles kept, {len(rows)} to go")
    k = 0
    with mp.get_context("fork").Pool(workers) as pool, open(out, "a" if resume else "w") as fh:
        for rs in pool.imap_unordered(_puzzle, [(r, tuple(ladder), budget) for r in rows], chunksize=1):
            for r in rs:
                fh.write(json.dumps(r) + "\n"); k += 1
            fh.flush()
    log(f"[depth_ladder] {len(rows)} puzzles, {k} positions -> {out} ({time.perf_counter() - t0:.0f}s)")
    return str(out)


def _miss_position(args) -> list:
    fen, sol, pid, ladder, budget = args
    r = _position(chess.Board(fen), pid, 0, ladder, budget)
    if r is None:
        return []
    r["sol"] = sol
    return [r]


def collect_misses(dumps: list, ladder=LADDER, budget: int = BUDGET, workers: int = 10, tag: str = "misses", resume: bool = False, log=print) -> str:
    """The ladder on the HARD positions: the failing step of every puzzle the zone ledger missed in `dumps`
    (frame_puzzles dumps) -- where Stockfish needs nominal depth 7-10 (sf_depth) and a 'keep going' label can exist."""
    import multiprocessing as mp
    t0 = time.perf_counter()
    items = []
    for name in dumps:
        for r in json.load(open(paths.derived(name))):
            if r["solved"]:
                continue
            s = next((s for s in r["steps"] if not s["tied"]), None)
            if s is not None:
                items.append((s["fen"], s["sol"], r["id"], tuple(ladder), budget))
    out = paths.derived(f"depth_ladder_{tag}.jsonl")
    done = set()
    if resume and os.path.exists(out):
        done = {json.loads(l)["pid"] for l in open(out) if l.strip()}
        items = [x for x in items if x[2] not in done]
    k = 0
    with mp.get_context("fork").Pool(workers) as pool, open(out, "a" if resume else "w") as fh:
        for rs in pool.imap_unordered(_miss_position, items, chunksize=1):
            for r in rs:
                fh.write(json.dumps(r) + "\n"); k += 1
            fh.flush()
    log(f"[depth_ladder] {len(items)} missed positions, {k} written -> {out} ({time.perf_counter() - t0:.0f}s)")
    return str(out)


def decisions(rec: dict) -> dict:
    """The decision at each depth of the ladder: the top candidate(s) when every candidate stands at its last
    completed depth up to that depth. -> {depth: set of ucis}; also the depths every candidate completed."""
    depths = sorted({d for h in rec["history"].values() for d, _, _ in h})
    out = {}
    for d in depths:
        val = {}
        for u, h in rec["history"].items():
            if not h:                                                 # no history: the move mates at once and is never searched
                val[u] = 1e6; continue                                # (bug 2026-09-30: mates dropped out of every decision)
            done = [v for dd, v, _ in h if dd <= d and v is not None]
            if done:
                val[u] = done[-1]
        if val:
            top = max(val.values())
            out[d] = {u for u, v in val.items() if v == top}
    return out


def report(path: str, log=print) -> None:
    import statistics as st
    recs = [json.loads(l) for l in open(path)]
    log(f"[depth_ladder] {len(recs)} positions from {len({r['pid'] for r in recs})} puzzles; nodes a position: median "
        f"{st.median([r['nodes'] for r in recs]):.0f}, max {max(r['nodes'] for r in recs)}; seconds a position: median "
        f"{st.median([r['secs'] for r in recs]):.1f}, max {max(r['secs'] for r in recs):.1f}")
    ladder = sorted({d for r in recs for h in r["history"].values() for d, _, _ in h})
    reached = {d: sum(1 for r in recs for h in r["history"].values() if any(dd == d and v is not None for dd, v, _ in h)) for d in ladder}
    total = sum(len(r["history"]) for r in recs)
    log("OBS candidates that completed each depth: " + ", ".join(f"{d}: {reached[d]}/{total}" for d in ladder))
    for d in ladder[:-1]:
        changed = sol_right_now = sol_right_deep = 0
        for r in recs:
            dec = decisions(r)
            if d not in dec or not dec:
                continue
            deep = dec[max(dec)]
            changed += not (dec[d] & deep)
            sol_right_now += r["sol"] in dec[d]; sol_right_deep += r["sol"] in deep
        log(f"OBS stopping at {d} plies: the decision differs from the deepest one in {changed}/{len(recs)} positions; "
            f"the solution among the top at {d}: {sol_right_now}, at the deepest: {sol_right_deep}")


def main(argv=None):
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect"); c.add_argument("--n", type=int, default=10); c.add_argument("--offset", type=int, default=900)
    c.add_argument("--workers", type=int, default=10); c.add_argument("--resume", action="store_true")
    c.add_argument("--ladder", type=int, nargs="+", default=list(LADDER)); c.add_argument("--budget", type=int, default=BUDGET)
    r = sub.add_parser("report"); r.add_argument("--n", type=int, default=10); r.add_argument("--offset", type=int, default=900)
    r.add_argument("--path", default=None)
    mm = sub.add_parser("misses"); mm.add_argument("--dumps", nargs="+", required=True); mm.add_argument("--resume", action="store_true")
    mm.add_argument("--ladder", type=int, nargs="+", default=list(LADDER)); mm.add_argument("--budget", type=int, default=BUDGET)
    a = ap.parse_args(argv)
    if a.cmd == "misses":
        return collect_misses(a.dumps, tuple(a.ladder), a.budget, resume=a.resume)
    if a.cmd == "collect":
        collect(a.n, a.offset, a.workers, a.resume, tuple(a.ladder), a.budget)
    else:
        report(a.path or str(paths.derived(f"depth_ladder_{a.n}_{a.offset}.jsonl")))


if __name__ == "__main__":
    main()
