"""PLAN 78.43 STAGE C, STEP THREE, THE MEASURE OF A GAME IN ONE CALL: `SheetModel.game(start, moves, ...)` and `SheetModel.games(...)` (rust/engine/tactics/src/sheet/run.rs `play_game`, via
sheet_rust.game / sheet_rust.games) against, at every ply of every game of a games directory (games.jsonl: one JSON record per line with `variant`, `start`, `moves`),
  (1) the Python light sheet's own result (`read_position`, the oracle, under the same stated cost model as the whole-sheet measure) put in the same brief form, and
  (2) the per-position Rust sheet (`sheet_rust.sheet`, the whole-sheet measure's subject) put in the same brief form.
THE BRIEF of a position: the steps taken; per target (the non-quiet questions in the order of the sheet) the owner, kind, place, worth, the top verdict kind, whether it says danger, the stake and
the deepest level asked; and, for the side to move and for the other side, the ways the deepest level settled as [confirmed, waiting, refuted, not settled] (by the side that asks).
THE SECONDS: per game, in one process, the game in one call against the same plies one `sheet` call at a time (both Rust), and Python's `read_position` over the same plies; then the many-game
form across std threads (1, 2, 4 and all cores) on the whole directory.

  .venv/bin/python -m catspace.incubator.analyst.discovery.sheet_rust_game_measure --games-dir DIR [--games N] [--jobs N] [--threads 1,2,4,11] [--out FILE]
"""
from __future__ import annotations

import argparse
import collections
import multiprocessing
import sys
import time
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import question_sheet as QS
from catspace.incubator.analyst.discovery import sheet_rust as SR
from catspace.incubator.analyst.discovery import sheet_rust_measure as M
from catspace.incubator.analyst.discovery import sheet_rust_sheet_measure as SM

GAME = "chess"
_STATE: dict = {}


def configs() -> list:
    """[(name, Params fields)] compared: the clock present, and chance when its table exists."""
    return [(n, f) for n, f, a in SM.configs() if n in ("present", "chance")]


def _setup() -> dict:
    if not _STATE:
        _STATE.update(SM._setup())
    return _STATE


def brief_python(sheet) -> dict:
    """The brief of a Python sheet (the oracle)."""
    r = sheet.reading
    ways = {"us": [0, 0, 0, 0], "them": [0, 0, 0, 0]}
    targets = []
    for q in sheet.questions:
        if q.quiet:
            continue
        kind = QS.verdict_kind(q)
        targets.append((q.target.side, q.target.kind, QS.place_text(q.target.at), r.worth_of(q.target.kind), kind, kind in QS.DANGER, q.stake, q.findings[-1].level if q.findings else -1))
        deep = next((f for f in reversed(q.findings) if f.level >= 2 and f.detail.get("results")), None)
        if deep is None:
            continue
        for x in deep.detail["results"]:
            if deep.detail["mode"] == "frame":
                if x.get("way") is None:
                    continue
                i = {QS.CONFIRMED_V: 0, QS.WAITING_V: 1, QS.REFUTED_V: 2}.get(x["status"], 3)
            else:
                i = {"confirmed": 0, "refuted": 2}.get(x["status"], 3)
            ways[q.asker][i] += 1
    return {"ply": None, "mover": r.mover, "steps": len(sheet.steps), "targets": targets, "ways_us": ways["us"], "ways_them": ways["them"]}


def brief_rust(res: dict) -> dict:
    """The brief of the per-position Rust sheet (`sheet_rust.sheet`)."""
    ways = {"us": [0, 0, 0, 0], "them": [0, 0, 0, 0]}
    targets = []
    worth = {tuple(u[:3]): u[3] for u in res["units"]}
    for q in res["questions"]:
        if q["quiet"]:
            continue
        t = q["target"]
        targets.append((t[0], t[1], t[2], worth[tuple(t)], q["key"][1], q["danger"], q["stake"], q["findings"][-1]["level"] if q["findings"] else -1))
        deep = next((f for f in reversed(q["findings"]) if f["level"] >= 2 and f["detail"].get("results")), None)
        if deep is None:
            continue
        d = deep["detail"]
        for x in d["results"]:
            if d["mode"] == "frame":
                if x["way"] is None:
                    continue
                i = {"confirmed": 0, "waiting": 1, "refuted": 2}.get(x["status"], 3)
            else:
                i = {"confirmed": 0, "refuted": 2}.get(x[1], 3) if isinstance(x, (list, tuple)) else 3
            ways[q["asker"]][i] += 1
    return {"ply": None, "mover": res["mover"], "steps": len(res["steps"]), "targets": targets, "ways_us": ways["us"], "ways_them": ways["them"]}


def canon_game(x: "dict | None") -> "dict | None":
    if x is None:
        return None
    return {"mover": x["mover"], "steps": x["steps"], "targets": [(t[0], t[1], t[2], float(t[3]), t[4], bool(t[5]), float(t[6]), t[7]) for t in x["targets"]], "ways_us": list(x["ways_us"]), "ways_them": list(x["ways_them"])}


def run_game(spec: dict) -> dict:
    st = _setup()
    env, worth, model = st["env"], st["worth"], st["model"]
    g = spec["record"]
    v, start, moves = g.get("variant") or env.variant, g.get("start"), g["moves"]
    out = {"game": spec["index"], "plies": 0, "unreadable": 0, "python_error": 0, "rust_error": 0, "diff_py": 0, "diff_sheet": 0, "targets": 0, "ways": 0, "diffs": [], "t_game": collections.Counter(),
           "t_sheet": collections.Counter(), "t_py": collections.Counter(), "t_core": collections.Counter()}
    for name, fields in configs():
        params = SM.params_of(fields)
        positions = [CE.state(v, start, moves[:ply])[0] for ply in range(len(moves) + 1)]
        t0 = time.perf_counter()
        try:
            got = SR.game(model, start, moves, params)
        except Exception as e:
            out["rust_error"] += 1
            out["diffs"].append({"game": spec["index"], "ply": None, "config": name, "field": "rust_error", "py": None, "rs": f"{type(e).__name__}: {e}"[:200]})
            continue
        out["t_game"][name] += time.perf_counter() - t0
        t_sheet = t_py = 0.0
        for ply, pos in enumerate(positions):
            try:
                t1 = time.perf_counter()
                sheet = QS.read_position(env, worth, params, pos, None, light=True)
                t_py += time.perf_counter() - t1
            except ValueError:
                if got[ply] is not None:
                    out["diff_py"] += 1
                    out["diffs"].append({"game": spec["index"], "ply": ply, "config": name, "field": "readable", "py": None, "rs": "a summary"})
                if name == "present":
                    out["unreadable"] += 1
                continue
            t2 = time.perf_counter()
            res = SR.sheet(model, env, pos, params)
            t_sheet += time.perf_counter() - t2
            out["t_core"][name] += res["seconds"]
            if name == "present":
                out["plies"] += 1
            a, b, c = brief_python(sheet), brief_rust(res), canon_game(got[ply])
            a.pop("ply"), b.pop("ply")
            a, b = canon_game(a), canon_game(b)
            out["targets"] += len(a["targets"])
            out["ways"] += sum(a["ways_us"]) + sum(a["ways_them"])
            if c is None:
                out["diff_py"] += 1
                out["diffs"].append({"game": spec["index"], "ply": ply, "config": name, "field": "readable", "py": "a summary", "rs": None})
                continue
            for what, other, key in (("python", a, "diff_py"), ("sheet", b, "diff_sheet")):
                bad = [f for f in sorted(c) if other[f] != c[f]]
                if bad:
                    out[key] += 1
                    for f in bad:
                        out["diffs"].append({"game": spec["index"], "ply": ply, "config": name, "field": f"{what}.{f}", "py": repr(other[f])[:240], "rs": repr(c[f])[:240]})
        out["t_sheet"][name] += t_sheet
        out["t_py"][name] += t_py
    return out


def format_obs(results: list, wall: float, jobs: int, label: str) -> list:
    n = sum(r["plies"] for r in results)
    lines = [f"OBS sheet_rust_game {label}: games {len(results)}, positions {n}, unreadable (over) {sum(r['unreadable'] for r in results)}, python errors {sum(r['python_error'] for r in results)}, rust errors {sum(r['rust_error'] for r in results)}, jobs {jobs}, wall {wall:.1f} s; cost model {SM.COST_MODEL}"]
    lines.append(f"OBS sheet_rust_game {label}: targets compared {sum(r['targets'] for r in results)}, ways counted {sum(r['ways'] for r in results)}, positions whose brief differs from the Python sheet's {sum(r['diff_py'] for r in results)}, from the per-position Rust sheet's {sum(r['diff_sheet'] for r in results)}")
    for name, _f in configs():
        tg, ts, tp, tc = (sum(r[k][name] for r in results) for k in ("t_game", "t_sheet", "t_py", "t_core"))
        if n:
            lines.append(f"OBS sheet_rust_game {label} clock {name} seconds per game (jobs {jobs}, one process each): the game in one call {tg / len(results):.4f}, the same plies one sheet call at a time {ts / len(results):.4f} (engine {tc / len(results):.4f}), "
                         f"Python read_position {tp / len(results):.4f}; per position: {tg / n:.5f}, {ts / n:.5f}, {tp / n:.5f}")
    return lines


def threads_obs(specs: list, counts: list, label: str) -> list:
    """The many-game form across std threads on all the games: wall seconds per call (the clock present, the stated cost model), and whether every threaded result equals the one-thread result."""
    st = _setup()
    model = st["model"]
    params = SM.params_of({"clock": "present"})
    games = [(s["record"].get("start"), s["record"]["moves"]) for s in specs]
    out, base = [], None
    for k in counts:
        t0 = time.perf_counter()
        res = SR.games(model, games, k, params)
        dt = time.perf_counter() - t0
        canon = [[canon_game(x) for x in game] for game in res]
        if base is None:
            base = canon
        same = canon == base
        plies = sum(len(g) for g in res)
        out.append(f"OBS sheet_rust_game {label} threads {k}: {len(games)} games, {plies} positions in {dt:.2f} s wall ({dt / len(games):.3f} s per game, {1000 * dt / plies:.2f} ms per position); equal to the one-thread result: {same}")
    return out


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--games", type=int, default=None, help="the first N games")
    ap.add_argument("--jobs", type=int, default=multiprocessing.cpu_count())
    ap.add_argument("--threads", default="1,2,4,11", help="thread counts of the many-game form (empty: not run)")
    ap.add_argument("--max-diffs", type=int, default=40)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    specs = [{"index": i, "record": g} for i, g in enumerate(M._games(a.games_dir, a.games))]
    t0 = time.perf_counter()
    if a.jobs <= 1:
        results = [run_game(s) for s in specs]
    else:
        with multiprocessing.get_context("spawn").Pool(min(a.jobs, len(specs))) as p:
            results = list(p.imap_unordered(run_game, specs))
    results.sort(key=lambda r: r["game"])
    label = Path(a.games_dir).name
    lines = format_obs(results, time.perf_counter() - t0, min(a.jobs, len(specs)), label)
    diffs = [d for r in results for d in r["diffs"]]
    lines.append(f"OBS sheet_rust_game {label}: differences listed {len(diffs)} (printed up to --max-diffs {a.max_diffs})")
    for d in diffs[:a.max_diffs]:
        lines.append(f"DIFF game {d['game']} ply {d['ply']} | config {d['config']} | {d['field']} | python-or-sheet {d['py']} | game {d['rs']}")
    if a.threads:
        lines += threads_obs(specs, [int(x) for x in a.threads.split(",")], label)
    for ln in lines:
        print(ln, flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
