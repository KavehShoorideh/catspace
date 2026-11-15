"""PLAN 78.43 STAGE C, STEP TWO, THE MEASURE OF THE WHOLE SHEET: the Rust schedule (rust/engine/tactics/src/sheet/run.rs, via sheet_rust.sheet) against the Python light sheet (question_sheet.py,
the oracle: `read_position(env, worth, params, position, None, light=True)` = `run`, `finish_levels(1)`, `settle_flagged`), on every position of every game of a games directory (games.jsonl: one
JSON record per line with `variant`, `start`, `moves`).

WHERE MEASURED TIME ENTERS THE ORDER. In the Python schedule the only place is `step`: the cost of a step is its wall time, `_record` adds it to the running mean of the costs of its level
(`_Costs`), and the next question is chosen by stake x urgency over the expected cost of its level (the clocks `present` and `chance`; `plain` uses none), and `budget_cost` stops the run when
the summed cost reaches it. (`budget_seconds` and the pool are wall-clock too and are not ported.) The order is therefore not reproducible from run to run. HOW IT IS MADE DETERMINISTIC HERE:
`Params.cost_model`, a tuple of seconds per level (the last for every deeper level), replaces the measurement in `step` when it is set (the default, None, is the measured time, unchanged); the
Rust form takes the same tuple (`Costing::Fixed`). Both forms are compared under COST_MODEL below, with `max_steps` (the default 40) as the budget, and with `budget_cost` for one run.

For every position where the Python light sheet reads and every config (the clocks present and plain; chance when the table at data/derived/question_chances/fit1112.json exists; present under
`budget_cost`), the whole sheet is compared: the ORDER OF THE STEPS TAKEN (question, level, status, mode); every question in order (its unit, stake, rank, level reached, state, status, urgency,
best value, time spent, the key of its record, whether the verdict kind says danger, its stake bounds and stake at risk); each finding of each question (level, status, whether new, value, and
the whole detail: levels 0 and 1 as stage A compares them, the goal unit's weak places, the deeper levels as stage B compares them, and the time obstacle of every way).

  .venv/bin/python -m catspace.incubator.analyst.discovery.sheet_rust_sheet_measure --games-dir DIR [--games N] [--jobs N] [--out FILE]
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
from catspace.incubator.analyst.discovery import sheet_rust_deep_measure as D
from catspace.incubator.analyst.discovery import sheet_rust_measure as M
from catspace.io import paths

GAME = "chess"
COST_MODEL = (0.002, 0.01, 0.03, 0.06, 0.1)        # seconds a step costs at level 0, 1, 2, 3, 4 and deeper: stated, not measured
BUDGET_COST = 0.15                                  # the summed cost the `budget` config stops at (a few steps under COST_MODEL)
CHANCES = paths.derived("question_chances", "fit1112.json")
_STATE: dict = {}


def configs() -> list:
    """[(name, Params fields, run arguments)] compared at every position."""
    out = [("present", {"clock": "present"}, {}), ("plain", {"clock": "plain"}, {})]
    if Path(CHANCES).exists():
        out.append(("chance", {"clock": "chance", "chances": CHANCES}, {}))
    out.append(("budget", {"clock": "present"}, {"budget_cost": BUDGET_COST}))
    return out


def _setup() -> dict:
    if not _STATE:
        env = QS.Env(GAME)
        worth = QS.store_worth(GAME)
        _STATE.update(env=env, worth=worth, model=SR.model(env, worth))
    return _STATE


def params_of(fields: dict) -> "QS.Params":
    return QS.Params(compare_old=False, cost_model=COST_MODEL, **fields)


# ------------------------------------------------------------------------------------------------------------------------------------------------- the Python side, canonical

def _key(u) -> "tuple | None":
    return None if u is None else (u.side, u.kind, QS.place_text(u.at))


def _weak(ws) -> list:
    pu = lambda us: [(u.side, u.kind, QS.place_text(u.at)) for u in us]
    return [(w["place"], pu(w["guards"]), pu(w["strikers"]), w["need"], [(h, _key(u), a) for h, u, a in w["candidates"]]) for w in ws]


def _times(detail: dict) -> list:
    return [x.get("time") for x in detail.get("results", []) if x.get("way") is not None] if detail.get("mode") == "frame" else []


def canon_python(sheet, params) -> dict:
    r, qs = sheet.reading, sheet.questions
    ix = {id(q): i for i, q in enumerate(qs)}
    out = {"steps": [(ix[id(s.question)], s.level, s.status, s.mode.split(",")[0]) for s in sheet.steps], "questions": [], "findings": [], "stage_a": None, "weak": [], "time": [], "deep": []}
    rows = []
    for i, q in enumerate(qs):
        k = QS.question_key(q)
        out["questions"].append({"target": _key(q.target), "asker": q.asker, "stake": q.stake, "rank": q.rank, "goal": q.goal, "quiet": q.quiet, "level": q.level, "state": q.state, "status": q.status,
                                 "urgency": q.urgency, "spent": q.spent, "live_by_level1": q.live_by_level1, "closed_at": q.closed_at, "best_value": q.best_value, "waiting": bool(q.waiting),
                                 "key": list(k), "danger": QS.verdict_kind(q) in QS.DANGER, "bounds": tuple(QS.stake_bounds(q, r)), "at_risk": QS.stake_at_risk(q, r, params)})
        out["findings"].append([(f.level, f.status, f.new, None if f.value is None else float(f.value), f.detail.get("mode", "").split(",")[0]) for f in q.findings])
        if q.quiet:
            if q.findings:                                  # offered by the schedule but not stepped when the budget ran out
                f = q.findings[0]
                out["deep"].append((i, 0, {"unit": _key(f.detail["unit"]), "strikes": f.detail["strikes"], "places": f.detail["places"]}))
            continue
        if len(q.findings) >= 2:
            rows.append((q, q.findings[0].status, q.findings[0].detail, q.findings[1].status, q.findings[1].detail))
            if q.goal:
                out["weak"].append((i, _weak(q.findings[1].detail.get("weak", []))))
        for f in q.findings:
            if f.level >= 2:
                out["deep"].append((i, f.level, D.canon_finding_python(q, f)))
                out["time"].append((i, f.level, _times(f.detail)))
    out["stage_a"] = M.canon_python({"r": r, "rows": rows})
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------- the Rust side, canonical

def _unit_words(u: tuple) -> str:
    return f"{'my' if u[0] == 'us' else 'their'} {u[1]} on {u[2]}"


def _rtime(entry: dict) -> "str | None":
    t = entry.get("time")
    return None if t is None else f"{'their' if t[2] == 'them' else 'my'} way on {_unit_words(tuple(t[1]))} lands first (in {t[0]} moves)"


def canon_rust(res: dict) -> dict:
    qs = res["questions"]
    out = {"steps": [(s[0], s[1], s[2], s[3]) for s in res["steps"]], "questions": [], "findings": [], "stage_a": None, "weak": [], "time": [], "deep": []}
    order, l0s, l1s = [], [], []
    for i, q in enumerate(qs):
        out["questions"].append({"target": None if q["target"] is None else tuple(q["target"]), "asker": q["asker"], "stake": q["stake"], "rank": q["rank"], "goal": q["goal"], "quiet": q["quiet"],
                                 "level": q["level"], "state": q["state"], "status": q["status"], "urgency": q["urgency"], "spent": q["spent"], "live_by_level1": q["live_by_level1"],
                                 "closed_at": q["closed_at"], "best_value": q["best_value"], "waiting": q["waiting"], "key": list(q["key"]), "danger": q["danger"], "bounds": tuple(q["bounds"]),
                                 "at_risk": q["at_risk"]})
        out["findings"].append([(f["level"], f["status"], f["new"], None if f["value"] is None else float(f["value"]), f["mode"]) for f in q["findings"]])
        if q["quiet"]:
            if q["findings"]:
                d = q["findings"][0]["detail"]
                out["deep"].append((i, 0, {"unit": tuple(d["unit"]), "strikes": d["strikes"], "places": d["places"]}))
            continue
        if len(q["findings"]) >= 2:
            t = q["target"]
            order.append(tuple(t) + (q["stake"], q["goal"], q["rank"]))
            l0s.append(q["findings"][0]["detail"])
            l1s.append(q["findings"][1]["detail"])
            if q["goal"]:
                out["weak"].append((i, [(w[0], [tuple(u) for u in w[1]], [tuple(u) for u in w[2]], w[3], [(h, tuple(u), a) for h, u, a in w[4]]) for w in q["findings"][1]["detail"].get("weak", [])]))
        for f in q["findings"]:
            if f["level"] >= 2:
                out["deep"].append((i, f["level"], D.canon_finding_rust(f["detail"])))
                if f["detail"]["mode"] == "frame":
                    out["time"].append((i, f["level"], [_rtime(x) for x in f["detail"]["results"] if x["way"] is not None]))
                else:
                    out["time"].append((i, f["level"], []))
    out["stage_a"] = M.canon_rust({"units": res["units"], "order": order, "level0": l0s, "level1": l1s})
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------- comparing

def diff_sheets(a: dict, b: dict) -> list:
    """[(where, field, python, rust)] of every difference of one position under one config."""
    out = []
    if a["steps"] != b["steps"]:
        n = next((i for i, (x, y) in enumerate(zip(a["steps"], b["steps"])) if x != y), min(len(a["steps"]), len(b["steps"])))
        out.append(("steps", f"order (first difference at step {n}; {len(a['steps'])} against {len(b['steps'])} steps)", a["steps"][n:n + 3], b["steps"][n:n + 3]))
    if len(a["questions"]) != len(b["questions"]):
        out.append(("questions", "count", len(a["questions"]), len(b["questions"])))
        return out
    for i, (x, y) in enumerate(zip(a["questions"], b["questions"])):
        for f in sorted(x):
            if x[f] != y[f]:
                out.append((f"question {i} {x['target']}", f, x[f], y[f]))
    for i, (x, y) in enumerate(zip(a["findings"], b["findings"])):
        if x != y:
            out.append((f"question {i} {a['questions'][i]['target']}", "findings (level, status, new, value, mode)", x, y))
    for what, key, field, pv, rv in M.diff_position(a["stage_a"], b["stage_a"]):
        out.append((f"stage A {what} {key}", field, pv, rv))
    if a["weak"] != b["weak"]:
        out.append(("goal unit", "weak places", a["weak"], b["weak"]))
    if len(a["deep"]) != len(b["deep"]):
        out.append(("deeper findings", "count", len(a["deep"]), len(b["deep"])))
    else:
        for (i, lv, x), (_j, _lv, y) in zip(a["deep"], b["deep"]):
            for where, field, pv, rv in D.diff_finding(x, y) if "mode" in x else [("quiet", f, x[f], y[f]) for f in x if x[f] != y[f]]:
                out.append((f"question {i} {a['questions'][i]['target']} level {lv} {where}", field, pv, rv))
    if a["time"] != b["time"]:
        out.append(("time obstacle", "notes", [t for t in a["time"] if any(t[2])], [t for t in b["time"] if any(t[2])]))
    return out


def run_game(spec: dict) -> dict:
    st = _setup()
    env, worth, model = st["env"], st["worth"], st["model"]
    g = spec["record"]
    v, start, moves = g.get("variant") or env.variant, g.get("start"), g["moves"]
    cfgs = configs()
    out = {"game": spec["index"], "positions": 0, "unreadable": 0, "python_error": 0, "rust_error": 0, "t_py": collections.Counter(), "t_rs": collections.Counter(), "t_core": collections.Counter(),
           "runs": collections.Counter(), "run_diff": collections.Counter(), "field_diff": collections.Counter(), "diffs": [], "steps": collections.Counter(), "questions": 0, "findings": 0, "deep": 0, "ways": 0,
           "quiet": 0, "weak": 0, "time_notes": 0, "danger": collections.Counter(), "levels": collections.Counter(), "kinds": collections.Counter()}
    for ply in range(len(moves) + 1):
        pos = CE.state(v, start, moves[:ply])[0]
        for name, fields, args in cfgs:
            params = params_of(fields)
            try:
                t0 = time.perf_counter()
                sheet = QS.read_position(env, worth, params, pos, None, light=True, **({"max_steps": args["max_steps"]} if "max_steps" in args else {})) if "budget_cost" not in args else _budget_python(env, worth, params, pos, args["budget_cost"])
                t_py = time.perf_counter() - t0
            except ValueError:
                if name == cfgs[0][0]:
                    out["unreadable"] += 1
                continue
            except Exception as e:
                out["python_error"] += 1
                out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "config": name, "where": "python_error", "field": type(e).__name__, "py": str(e)[:160], "rs": None})
                continue
            try:
                t1 = time.perf_counter()
                res = SR.sheet(model, env, pos, params, max_steps=args.get("max_steps"), budget_cost=args.get("budget_cost"))
                t_rs = time.perf_counter() - t1
            except Exception as e:
                out["rust_error"] += 1
                out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "config": name, "where": "rust_error", "field": type(e).__name__, "py": None, "rs": str(e)[:240]})
                continue
            if name == cfgs[0][0]:
                out["positions"] += 1
            a, b = canon_python(sheet, params), canon_rust(res)
            d = diff_sheets(a, b)
            out["runs"][name] += 1
            out["t_py"][name] += t_py
            out["t_rs"][name] += t_rs
            out["t_core"][name] += res["seconds"]
            out["steps"][name] += len(a["steps"])
            out["questions"] += len(a["questions"])
            out["findings"] += sum(len(x) for x in a["findings"])
            out["deep"] += len(a["deep"])
            out["ways"] += sum(len(x["results"]) - 1 for _i, _l, x in a["deep"] if x.get("mode") == "frame")
            out["quiet"] += sum(1 for q in a["questions"] if q["quiet"])
            out["weak"] += sum(len(w) for _i, w in a["weak"])
            out["time_notes"] += sum(1 for _i, _l, t in a["time"] for n in t if n)
            for q in a["questions"]:
                out["kinds"][q["key"][1]] += 1
                out["danger"][q["danger"]] += 1
            if d:
                out["run_diff"][name] += 1
            for where, field, pv, rv in d:
                out["field_diff"][(name, where.split(" ")[0] if not where.startswith("stage") else "stage A", field.split(" ")[0])] += 1
                out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "config": name, "where": where, "field": field, "py": repr(pv)[:260], "rs": repr(rv)[:260]})
    return out


def _budget_python(env, worth, params, pos, budget_cost):
    """`read_position` under a `budget_cost` (the equivalent: run with the budget, then finish_levels(1), settle_flagged)."""
    sheet = QS.run(pos, worth, env=env, params=params, light=True, budget_cost=budget_cost)
    QS.finish_levels(sheet, 1, params)
    QS.settle_flagged(sheet, params)
    return sheet


def profile_game(spec: dict) -> dict:
    """WHERE THE RUST TIME GOES: the whole sheet in Rust under each clock with the MEASURED cost of each step (`cost_model` None), the seconds summed by kind of step (level and mode) and the
    time outside the steps (making the position and the reading, handing the data to Python)."""
    st = _setup()
    env, model = st["env"], st["model"]
    g = spec["record"]
    v, start, moves = g.get("variant") or env.variant, g.get("start"), g["moves"]
    out = {"game": spec["index"], "positions": 0, "by_step": collections.defaultdict(lambda: [0, 0.0]), "total": collections.Counter(), "inside": collections.Counter()}
    for ply in range(len(moves) + 1):
        pos = CE.state(v, start, moves[:ply])[0]
        for name, fields, args in configs():
            if "budget_cost" in args:
                continue
            params = QS.Params(compare_old=False, **fields)
            try:
                t0 = time.perf_counter()
                res = SR.sheet(model, env, pos, params)
                dt = time.perf_counter() - t0
            except ValueError:
                break
            for _qi, level, _status, mode, cost in res["steps"]:
                key = (name, f"level {level}" + (f" {mode}" if mode else ""))
                out["by_step"][key][0] += 1
                out["by_step"][key][1] += cost
            out["total"][name] += dt
            out["inside"][name] += res["seconds"]
            if name == "present":
                out["positions"] += 1
    out["by_step"] = dict(out["by_step"])
    return out


def format_profile(results: list, label: str) -> list:
    n = sum(r["positions"] for r in results)
    by: dict = {}
    for r in results:
        for k, (c, t) in r["by_step"].items():
            x = by.setdefault(k, [0, 0.0])
            x[0] += c
            x[1] += t
    tot = collections.Counter()
    ins = collections.Counter()
    for r in results:
        tot.update(r["total"])
        ins.update(r["inside"])
    lines = [f"OBS sheet_rust_sheet {label} profile: positions {n}, Rust only, measured step costs, one process per game, seconds per position by kind of step (steps per position in brackets)"]
    for name in sorted({k[0] for k in by}):
        whole = tot[name] / n if n else 0.0
        lines.append(f"OBS sheet_rust_sheet {label} profile clock {name}: whole sheet {whole:.5f} s per position (engine {ins[name] / n:.5f}); by kind of step:")
        rows = sorted(((k[1], c, t) for k, (c, t) in by.items() if k[0] == name), key=lambda x: -x[2])
        steps_total = sum(t for _k, _c, t in rows)
        for kind, c, t in rows:
            lines.append(f"OBS sheet_rust_sheet {label} profile {name}: {kind}: {t / n:.5f} s per position ({100 * t / max(steps_total, 1e-12):.1f}% of the steps), {c / n:.2f} steps per position, {1000 * t / c:.3f} ms per step")
        lines.append(f"OBS sheet_rust_sheet {label} profile {name}: outside the steps {(ins[name] - steps_total) / n:.5f} s per position (reading, the quiet question's bookkeeping)")
    return lines


def format_obs(results: list, wall: float, jobs: int, label: str) -> list:
    n = sum(r["positions"] for r in results)
    C = collections.Counter
    names = [c[0] for c in configs()]
    lines = [f"OBS sheet_rust_sheet {label}: games {len(results)}, positions {n}, unreadable (over) {sum(r['unreadable'] for r in results)}, python errors {sum(r['python_error'] for r in results)}, rust errors {sum(r['rust_error'] for r in results)}, jobs {jobs}, wall {wall:.1f} s; cost model {COST_MODEL}, budget_cost {BUDGET_COST}"]
    for nm in names:
        runs = sum(r["runs"][nm] for r in results)
        lines.append(f"OBS sheet_rust_sheet {label} clock {nm}: sheets compared {runs}, sheets that differ {sum(r['run_diff'][nm] for r in results)}, steps compared {sum(r['steps'][nm] for r in results)}")
    lines.append(f"OBS sheet_rust_sheet {label}: questions compared {sum(r['questions'] for r in results)}, findings {sum(r['findings'] for r in results)}, deeper findings {sum(r['deep'] for r in results)}, ways settled {sum(r['ways'] for r in results)}, "
                 f"quiet questions {sum(r['quiet'] for r in results)}, weak places {sum(r['weak'] for r in results)}, time obstacle notes {sum(r['time_notes'] for r in results)}")
    kinds = sum((r["kinds"] for r in results), C())
    lines.append(f"OBS sheet_rust_sheet {label}: verdict kinds of the questions (Python, all configs): " + ", ".join(f"{k} {c}" for k, c in sorted(kinds.items())) + f"; danger {sum((r['danger'] for r in results), C())[True]} of {sum(kinds.values())}")
    fd = sum((r["field_diff"] for r in results), C())
    lines.append(f"OBS sheet_rust_sheet {label}: differences per (config, part, field): " + (", ".join(f"{'.'.join(k)} {c}" for k, c in sorted(fd.items())) if fd else "none"))
    for nm in names:
        runs = sum(r["runs"][nm] for r in results)
        if runs:
            tp, tr, tc = (sum(r[k][nm] for r in results) / runs for k in ("t_py", "t_rs", "t_core"))
            lines.append(f"OBS sheet_rust_sheet {label} seconds per position, clock {nm} (jobs {jobs}): python {tp:.5f} (the whole read_position), rust {tr:.5f} (the whole sheet; the engine's own work {tc:.5f}), ratio {tp / tr if tr else float('nan'):.1f}x")
    return lines


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--games", type=int, default=None, help="the first N games")
    ap.add_argument("--jobs", type=int, default=multiprocessing.cpu_count())
    ap.add_argument("--max-diffs", type=int, default=40)
    ap.add_argument("--out", default=None)
    ap.add_argument("--profile", action="store_true", help="Rust only: the seconds of the whole sheet by kind of step (measured costs), no comparison")
    a = ap.parse_args(argv)
    specs = [{"index": i, "record": g} for i, g in enumerate(M._games(a.games_dir, a.games))]
    t0 = time.perf_counter()
    if a.profile:
        results = [profile_game(s) for s in specs] if a.jobs <= 1 else multiprocessing.get_context("spawn").Pool(min(a.jobs, len(specs))).map(profile_game, specs)
        for ln in format_profile(results, Path(a.games_dir).name):
            print(ln, flush=True)
        return 0
    if a.jobs <= 1:
        results = [run_game(s) for s in specs]
    else:
        with multiprocessing.get_context("spawn").Pool(min(a.jobs, len(specs))) as p:
            results = []
            for r in p.imap_unordered(run_game, specs):
                results.append(r)
                print(f"OBS progress: game {r['game']} done, {len(results)} of {len(specs)}, {time.perf_counter() - t0:.0f} s", flush=True)
    results.sort(key=lambda r: r["game"])
    label = Path(a.games_dir).name
    lines = format_obs(results, time.perf_counter() - t0, min(a.jobs, len(specs)), label)
    diffs = [d for r in results for d in r["diffs"]]
    lines.append(f"OBS sheet_rust_sheet {label}: differences listed {len(diffs)} (printed up to --max-diffs {a.max_diffs}; the cause of each is read by hand)")
    for d in diffs[:a.max_diffs]:
        lines.append(f"DIFF game {d['game']} ply {d['ply']} {d['fen']} | config {d['config']} | {d['where']} {d['field']} | python {d['py']} | rust {d['rs']}")
    for ln in lines:
        print(ln, flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
