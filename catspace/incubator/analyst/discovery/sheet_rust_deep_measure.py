"""PLAN 78.43 STAGE B, THE MEASURE OF THE DEEPER LEVELS: the Rust obstacles, frames and verdicts of a way (rust/engine/tactics/src/sheet: cf, frame, walk, settle; sheet_rust.settle) against the
Python light sheet (question_sheet.py, the oracle), on every position of every game of a games directory (games.jsonl: one JSON record per line with `variant`, `start`, `moves`).

For every position where the Python light sheet reads, `QS.read_position(env, worth, params, position, None, light=True)` is run (the schedule `run`, `finish_levels(1)`, `settle_flagged`). Then:
  (1) LEVELS 0 AND 1 AND THE STAKES: the stage A comparison, read from the findings of the sheet that was just run (units, the first order with the stakes, level 0, the routes and the verdict of
      level 1: status, ways, failed links and which link failed, links, reach, far units, the goal unit's held numbers);
  (2) THE DEEPER LEVELS, DRIVEN: the Python sheet's own steps of level 2 and on, in the order it took them, are replayed in the Rust engine (`sheet_rust.settle`): the Rust side is not given
      the Python answers, it is given WHICH question to deepen to WHICH level, in that order (the schedule is stage C; the order matters because the sheet keeps answers on its reading, keyed
      as the Python sheet keys them). For each step the findings are compared, field by field: the status of the finding and, for a unit that is not the goal unit, for the exchange as it
      stands and for every way settled (the ways are taken by the Rust side from its own level 1, the same ways must come out): the verdict kind, the value, the line, where it was read
      from, the route, the obstacles (which units, single or joint, where, whether dead, where a unit was moved aside), what each obstacle was asked and answered, the answers that stop it,
      what it waits on, the duties of the unit that travels, the first obstacles of the first route, the nodes used; for the goal unit: the ways and the verdicts of the search, or the
      census' tactics. The `level reached` is the number of steps replayed per question (by construction the same: the schedule is not ported).
The seconds per position of each side: the Python side is the whole `read_position` (the schedule, the graph and the bookkeeping the Rust side does not do); the Rust side is the stage A
read plus the settle of the same steps (the engine's own seconds in brackets, the rest is handing the data to Python).

  .venv/bin/python -m catspace.incubator.analyst.discovery.sheet_rust_deep_measure --games-dir DIR [--games N] [--jobs N] [--out FILE]
"""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing
import re
import sys
import time
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import question_sheet as QS
from catspace.incubator.analyst.discovery import sheet_rust as SR
from catspace.incubator.analyst.discovery import sheet_rust_measure as M

GAME = "chess"
_STATE: dict = {}
_ASIDE = re.compile(r"moved aside to (\S+?)(?:,| \(|$)")


def _setup() -> dict:
    if not _STATE:
        env = QS.Env(GAME)
        worth = QS.store_worth(GAME)
        params = QS.Params(compare_old=False)
        _STATE.update(env=env, worth=worth, params=params, model=SR.model(env, worth))
    return _STATE


# ------------------------------------------------------------------------------------------------------------------------------------------------- the Python side, canonical

def _pu(u) -> "tuple | None":
    return None if u is None else (u.side, u.kind, QS.place_text(u.at))


def _pway(w) -> "tuple | None":
    return None if w is None else (w[0], w[1].kind, QS.place_text(w[1].at), QS.place_text(w[2]), w[3])


def _pstop(x) -> tuple:
    return (x.kind == "hard", x.answer, x.after, x.step, x.reason, tuple(x.places), tuple(x.mover) if x.mover else None)


def _pduty(d) -> tuple:
    return (_pu(d["unit"]), float(d["stake"]), d["kind"], float(d["value"]))


def _pob(o) -> tuple:
    m = _ASIDE.search(o.note or "")
    return (o.kind, _pu(o.unit), o.place, o.hop, bool(o.dead), tuple(o.joint), m.group(1) if m else None)


def _pask(a: dict) -> tuple:
    return (a["verdict"], float(a["cost"]), a["more"], tuple(_pduty(d) for d in a.get("duties", [])))


def _pentry(x: dict) -> dict:
    rt = x.get("route")
    ws = x.get("without_stopper")
    return {
        "way": _pway(x.get("way")), "status": x["status"], "value": float(x.get("value", 0.0)), "line": tuple(x.get("line", ())), "source": x.get("source"), "open": bool(x.get("open", False)),
        "nodes": x.get("nodes") or 0, "move": x.get("move"), "route": None if rt is None else [QS.place_text(p) for p in rt.places],
        "obstacles": [(_pob(o), _pask(a)) for o, a in x.get("obstacles", [])],
        "first_obstacles": None if x.get("first_obstacles") is None else [(s, k, bool(d)) for s, k, d in x["first_obstacles"]],
        "stops": [_pstop(s) for s in x.get("stops", [])], "alts": [(c[0], list(c[1])) for c in x.get("alts", [])], "breaks": [_pstop(s) for s in x.get("breaks", []) if hasattr(s, "kind")],
        "duties": None if x.get("duties") is None else [_pduty(d) for d in x["duties"]], "held_back": bool(x.get("held_back", False)), "skipped": bool(x.get("skipped", False)),
        "dead_routes": x.get("dead_routes"), "unread_routes": x.get("unread_routes"),
        "without_stopper": None if ws is None else (_pu(ws["unit"]), ws["verdict"], float(ws["value"])),
        "tried": x.get("tried"), "cut": None if x.get("cut") is None else bool(x["cut"]), "routes_tried": x.get("routes_tried"),
    }


def canon_finding_python(q, f) -> dict:
    d = f.detail
    mode = d.get("mode", "")
    out = {"status": f.status, "mode": mode.split(",")[0]}
    if mode == "frame":
        out.update(results=[_pentry(x) for x in d["results"]], confirmed=d["confirmed"], refuted=d["refuted"], waiting=d["waiting"], open=d["open"], lost_where_it_stands=d["lost_where_it_stands"],
                   value=d.get("value"), line=None if d.get("line") is None else list(d["line"]), sources=list(d["sources"]))
        if "cornered" in d:
            out["cornering"] = {"cornered": d["cornered"], "lost_now": d["lost_now"], "places": [(p, bool(lost)) for p, lost in d["places"]], "open_exits": d["open_exits"]}
    elif mode == "search":
        res = [(x.get("move"), x["status"], float(x["value"]), x.get("reply")) for x in d["results"]]
        out.update(depth=d["depth"], ways=[_pway(x["way"]) for x in d["results"]], results=res, confirmed=d["confirmed"], refuted=d["refuted"], open=d["open"], value=d.get("value"),
                   line=(d["line"][0] if d.get("line") else None))
    else:
        out.update(depth=d["depth"], tactics=d["tactics"], value=d.get("value"), line=None if d.get("line") is None else list(d["line"]))
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------- the Rust side, canonical

def _runit(u) -> "tuple | None":
    return None if u is None else tuple(u)


def _rduty(d) -> tuple:
    return (_runit(d[0]), float(d[1]), d[2], float(d[3]))


def _rentry(x: dict) -> dict:
    return {
        "way": None if x["way"] is None else tuple(x["way"]), "status": x["status"], "value": float(x["value"]), "line": tuple(x["line"]), "source": x["source"], "open": bool(x["open"]),
        "nodes": x["nodes"] or 0, "move": x["move"], "route": x["route"],
        "obstacles": [((o[0], _runit(o[1]), o[2], o[3], bool(o[4]), tuple(o[5]), o[6]), (a[0], float(a[1]), a[2], tuple(_rduty(d) for d in a[3]))) for o, a in x["obstacles"]],
        "first_obstacles": None if x["first_obstacles"] is None else [tuple(f) for f in x["first_obstacles"]],
        "stops": [_rstop(s) for s in x["stops"]], "alts": [(k, list(p)) for k, p in x["alts"]], "breaks": [_rstop(s) for s in x["breaks"]],
        "duties": None if x["duties"] is None else [_rduty(d) for d in x["duties"]], "held_back": bool(x["held_back"]), "skipped": bool(x["skipped"]),
        "dead_routes": x["dead_routes"], "unread_routes": x["unread_routes"],
        "without_stopper": None if x["without_stopper"] is None else (_runit(x["without_stopper"][0]), x["without_stopper"][1], float(x["without_stopper"][2])),
        "tried": x["tried"], "cut": x["cut"], "routes_tried": x["routes_tried"],
    }


def _rstop(s) -> tuple:
    hard, answer, after, step, reason, places, mover = s
    return (bool(hard), answer, after, step, reason, tuple(places), tuple(mover) if mover else None)


def canon_finding_rust(d: dict) -> dict:
    mode = d["mode"]
    out = {"status": d["status"], "mode": mode}
    if mode == "frame":
        out.update(results=[_rentry(x) for x in d["results"]], confirmed=d["confirmed"], refuted=d["refuted"], waiting=d["waiting"], open=d["open"], lost_where_it_stands=d["lost_where_it_stands"],
                   value=d["value"], line=d["line"], sources=list(d["sources"]))
        if "cornering" in d:
            c = d["cornering"]
            out["cornering"] = {"cornered": c["cornered"], "lost_now": c["lost_now"], "places": [tuple(p) for p in c["places"]], "open_exits": c["open_exits"]}
    elif mode == "search":
        out.update(depth=d["depth"], ways=[tuple(w) for w in d["ways"]], results=[(m, s, float(v), r) for m, s, v, r in d["results"]], confirmed=d["confirmed"], refuted=d["refuted"], open=d["open"],
                   value=d["value"], line=d["line"])
    else:
        out.update(depth=d["depth"], tactics=d["tactics"], value=d["value"], line=d["line"])
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------- comparing

def diff_finding(a: dict, b: dict) -> list:
    """[(where, field, python, rust)] of every difference of one deeper step."""
    out = []
    for f in sorted(set(a) | set(b)):
        if f == "results" and a.get("mode") == "frame":
            ra, rb = a.get("results", []), b.get("results", [])
            if len(ra) != len(rb) or [x["way"] for x in ra] != [x["way"] for x in rb]:
                out.append(("results", "ways", [x["way"] for x in ra], [x["way"] for x in rb]))
                continue
            for i, (x, y) in enumerate(zip(ra, rb)):
                for g in sorted(set(x) | set(y)):
                    if x.get(g) != y.get(g):
                        out.append((f"way {i} {x['way']}", g, x.get(g), y.get(g)))
        elif a.get(f) != b.get(f):
            out.append(("finding", f, a.get(f), b.get(f)))
    return out


def deep_steps(sheet) -> list:
    """[(question, finding)] of the Python sheet's steps of level 2 and on (the quiet question left out), in the order the sheet took them."""
    seen = collections.Counter()
    out = []
    for s in sheet.steps:
        q = s.question
        if q.quiet:
            continue
        f = q.findings[seen[id(q)]]
        seen[id(q)] += 1
        if f.level >= 2:
            out.append((q, f))
    return out


def level_rows(sheet) -> list:
    """The rows of stage A's canonical form (question, level 0 status and detail, level 1 status and detail) from the findings of the sheet that was run."""
    return [(q, q.findings[0].status, q.findings[0].detail, q.findings[1].status, q.findings[1].detail) for q in sheet.questions if not q.quiet]


def compare_position(st: dict, pos: str) -> dict:
    """One position, both sides: {steps, stage_a: [differences of levels 0 and 1 and the stakes], deep: [(question, level, where, field, python, rust)]} (ValueError when the position is over)."""
    env, worth, params, model = st["env"], st["worth"], st["params"], st["model"]
    sheet = QS.read_position(env, worth, params, pos, None, light=True)
    deep = deep_steps(sheet)
    a, b = M.canon_python({"r": sheet.reading, "rows": level_rows(sheet)}), M.canon_rust(SR.read(model, env, pos, params))
    res = SR.settle(model, env, pos, [(q.target.side, q.target.kind, QS.place_text(q.target.at), f.level) for q, f in deep], params)
    out = {"steps": deep, "stage_a": M.diff_position(a, b), "deep": []}
    for (q, f), rs in zip(deep, res["steps"]):
        key = (q.target.side, q.target.kind, QS.place_text(q.target.at))
        for where, field, pv, rv in diff_finding(canon_finding_python(q, f), canon_finding_rust(rs)):
            out["deep"].append((key, f.level, where, field, pv, rv))
    return out


def run_game(spec: dict) -> dict:
    st = _setup()
    env, worth, params, model = st["env"], st["worth"], st["params"], st["model"]
    g = spec["record"]
    v, start, moves = g.get("variant") or env.variant, g.get("start"), g["moves"]
    out = {"game": spec["index"], "positions": 0, "unreadable": 0, "python_error": 0, "t_py": 0.0, "t_rs": 0.0, "t_core": 0.0, "diffs": [], "steps": collections.Counter(), "step_diff": collections.Counter(),
           "ways": 0, "way_diff": 0, "obstacles": 0, "obstacle_diff": 0, "field_diff": collections.Counter(), "a_items": 0, "a_diff": 0, "levels": collections.Counter(), "unit_steps": 0}
    for ply in range(len(moves) + 1):
        pos = CE.state(v, start, moves[:ply])[0]
        t0 = time.perf_counter()
        try:
            sheet = QS.read_position(env, worth, params, pos, None, light=True)
        except ValueError:
            out["unreadable"] += 1
            continue
        except Exception as e:
            out["python_error"] += 1
            out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "q": None, "level": None, "where": "python_error", "field": type(e).__name__, "py": str(e)[:120], "rs": None})
            continue
        t_py = time.perf_counter() - t0
        # the Python sheet's own steps of level 2 and on, in the order taken
        deep, rows = deep_steps(sheet), level_rows(sheet)
        t1 = time.perf_counter()
        try:
            res_a = SR.read(model, env, pos, params)
            req = [(q.target.side, q.target.kind, QS.place_text(q.target.at), f.level) for q, f in deep]
            res_b = SR.settle(model, env, pos, req, params)
        except Exception as e:
            out["positions"] += 1
            out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "q": None, "level": None, "where": "rust_error", "field": type(e).__name__, "py": None, "rs": str(e)[:200]})
            continue
        t_rs = time.perf_counter() - t1
        out["positions"] += 1
        out["t_py"] += t_py
        out["t_rs"] += t_rs
        out["t_core"] += res_a["seconds"] + res_b["seconds"]
        # (1) levels 0 and 1 and the stakes, from the sheet just run
        a, b = M.canon_python({"r": sheet.reading, "rows": rows}), M.canon_rust(res_a)
        out["a_items"] += len(a["level0"])
        bad = {k for what, k, _f, _p, _r in M.diff_position(a, b) for k in [k if k is not None else _f]}
        out["a_diff"] += len(bad)
        for what, key, field, pv, rv in M.diff_position(a, b):
            out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "q": key, "level": what, "where": "stage A", "field": field, "py": repr(pv)[:240], "rs": repr(rv)[:240]})
        # (2) the deeper steps
        for (q, f), rs in zip(deep, res_b["steps"]):
            key = (q.target.side, q.target.kind, QS.place_text(q.target.at))
            pa, pb = canon_finding_python(q, f), canon_finding_rust(rs)
            mode = pa["mode"]
            out["steps"][mode] += 1
            out["levels"][f.level] += 1
            diffs = diff_finding(pa, pb)
            if mode == "frame":
                for x, y in zip(pa["results"], pb["results"]):
                    out["ways"] += x["way"] is not None
                    out["obstacles"] += len(x["obstacles"])
                    out["way_diff"] += x["way"] is not None and x != y
                    out["obstacle_diff"] += x["obstacles"] != y["obstacles"]
            if diffs:
                out["step_diff"][mode] += 1
            for where, field, pv, rv in diffs:
                out["field_diff"][(mode, field)] += 1
                out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "q": key, "level": f.level, "where": where, "field": field, "py": repr(pv)[:240], "rs": repr(rv)[:240]})
    return out


def format_obs(results: list, wall: float, jobs: int) -> list:
    n = sum(r["positions"] for r in results)
    tp, tr, tc = sum(r["t_py"] for r in results), sum(r["t_rs"] for r in results), sum(r["t_core"] for r in results)
    steps = sum((r["steps"] for r in results), collections.Counter())
    sdiff = sum((r["step_diff"] for r in results), collections.Counter())
    fdiff = sum((r["field_diff"] for r in results), collections.Counter())
    lv = sum((r["levels"] for r in results), collections.Counter())
    lines = [f"OBS sheet_rust_deep: games {len(results)}, positions compared {n}, unreadable (over) {sum(r['unreadable'] for r in results)}, python errors {sum(r['python_error'] for r in results)}, jobs {jobs}, wall {wall:.1f} s"]
    lines.append(f"OBS sheet_rust_deep stage A fields (levels 0 and 1 and the stakes, per question): items compared {sum(r['a_items'] for r in results)}, items that differ {sum(r['a_diff'] for r in results)}")
    lines.append(f"OBS sheet_rust_deep deeper steps driven (level 2 and on, the Python sheet's own order): {sum(steps.values())} ({', '.join(f'{m} {c}' for m, c in sorted(steps.items()))}); by level " + ", ".join(f"{k}: {c}" for k, c in sorted(lv.items())))
    for mode in sorted(steps):
        lines.append(f"OBS sheet_rust_deep steps {mode}: compared {steps[mode]}, steps that differ {sdiff[mode]}")
    lines.append(f"OBS sheet_rust_deep ways compared (settled ways, the exchange as it stands not counted): {sum(r['ways'] for r in results)}, ways that differ in any field {sum(r['way_diff'] for r in results)}")
    lines.append(f"OBS sheet_rust_deep obstacles compared (on the ways' routes, with their follow-up answers): {sum(r['obstacles'] for r in results)}, entries whose obstacles differ {sum(r['obstacle_diff'] for r in results)}")
    lines.append("OBS sheet_rust_deep differences per field: " + (", ".join(f"{m}.{f} {c}" for (m, f), c in sorted(fdiff.items())) if fdiff else "none"))
    if n:
        lines.append(f"OBS sheet_rust_deep seconds per position (jobs {jobs}): python {tp / n:.5f} (the whole read_position), rust {tr / n:.5f} (read and settle of the same steps; the engine's own work {tc / n:.5f}), ratio {tp / tr if tr else float('nan'):.1f}x")
    return lines


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--games", type=int, default=None, help="the first N games")
    ap.add_argument("--jobs", type=int, default=multiprocessing.cpu_count())
    ap.add_argument("--max-diffs", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    specs = [{"index": i, "record": g} for i, g in enumerate(M._games(a.games_dir, a.games))]
    t0 = time.perf_counter()
    if a.jobs <= 1:
        results = [run_game(s) for s in specs]
    else:
        with multiprocessing.get_context("spawn").Pool(min(a.jobs, len(specs))) as p:
            results = []
            for r in p.imap_unordered(run_game, specs):
                results.append(r)
                print(f"OBS progress: game {r['game']} done, {len(results)} of {len(specs)}, {time.perf_counter() - t0:.0f} s", flush=True)
    results.sort(key=lambda r: r["game"])
    lines = format_obs(results, time.perf_counter() - t0, min(a.jobs, len(specs)))
    diffs = [d for r in results for d in r["diffs"]]
    lines.append(f"OBS sheet_rust_deep differences listed: {len(diffs)} (printed below up to --max-diffs {a.max_diffs}; the cause of each is read by hand, see the plan note)")
    for d in diffs[:a.max_diffs]:
        lines.append(f"DIFF game {d['game']} ply {d['ply']} {d['fen']} | q {d['q']} level {d['level']} | {d['where']} {d['field']} | python {d['py']} | rust {d['rs']}")
    for ln in lines:
        print(ln, flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
