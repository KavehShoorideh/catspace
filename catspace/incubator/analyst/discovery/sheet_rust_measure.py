"""PLAN 78.43 STAGE A, THE MEASURE: the Rust light reading and levels 0 and 1 by geometry (sheet_rust.py) against the Python light sheet (question_sheet.py, the oracle), on every position
of every game of a games directory (games.jsonl: one JSON record per line with `variant`, `start`, `moves`).

For every position where the Python light sheet reads without error, three things are compared:
  (1) the UNITS read: (side, kind, place, worth), and the first order of the questions (side, kind, place, stake, goal, rank);
  (2) the LEVEL 0 verdict of every target: status, the units that strike it, the units that guard it, the open exits, the exits (the goal unit's struck and cornered);
  (3) the SET of level 1 routes of every target: the ways and, for every way, the routes with their places, stands, places passed over and obstacles.
Level 1's verdict (status, ways, failed links, links, reach, far units, the goal unit's held numbers) is compared as a fourth line. Every difference is printed with its position.
The seconds per position of each side are for the same work: the Python side builds the light reading, the first order and the level 0 and level 1 of every question; the Rust side is
one call that returns the same.

  .venv/bin/python -m catspace.incubator.analyst.discovery.sheet_rust_measure --games-dir DIR [--games N] [--jobs N] [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import question_sheet as QS
from catspace.incubator.analyst.discovery import sheet_rust as SR

GAME = "chess"
_STATE: dict = {}


def _games(directory: str, n: "int | None") -> list:
    rows = [json.loads(x) for x in (Path(directory) / "games.jsonl").read_text().splitlines() if x.strip()]
    return rows if n is None else rows[:n]


def _setup() -> dict:
    if not _STATE:
        env = QS.Env(GAME)
        worth = QS.store_worth(GAME)
        params = QS.Params(compare_old=False)
        _STATE.update(env=env, worth=worth, params=params, model=SR.model(env, worth))
    return _STATE


# ------------------------------------------------------------------------------------------------------------------------------------------------- the Python side, canonical

def _route(rt) -> tuple:
    obs = tuple((o.kind, None if o.unit is None else (o.unit.side, o.unit.kind, QS.place_text(o.unit.at)), o.place, o.hop, o.dead) for o in rt.obstacles)
    return (rt.unit.kind, QS.place_text(rt.unit.at), tuple(QS.place_text(p) for p in rt.places), tuple((bool(ok), why) for ok, why in rt.stands), rt.what,
            tuple(tuple(o) for o in rt.over), obs, rt.goal)


def _way(w) -> tuple:
    return (w[0], w[1].kind, QS.place_text(w[1].at), QS.place_text(w[2]), w[3])


def python_side(position: str, st: "dict | None" = None) -> dict:
    st = st or _setup()
    env, worth, params = st["env"], st["worth"], st["params"]
    r = QS.Reading(env, position, worth, light=True)
    rows = []
    for q in QS.first_order(r, params):
        s0, _t0, d0 = QS.level0(q, r, params)
        s1, _t1, d1 = QS._level1(q, r, params)
        rows.append((q, s0, d0, s1, d1))
    return {"r": r, "rows": rows}


def canon_python(res: dict) -> dict:
    r, rows = res["r"], res["rows"]
    pu = lambda us: sorted((u.kind, QS.place_text(u.at)) for u in us)
    out = {"units": [(u.side, u.kind, QS.place_text(u.at), r.worth_of(u.kind)) for u in r.units], "order": [], "level0": {}, "level1": {}, "routes": {}}
    for q, s0, d0, s1, d1 in rows:
        t = q.target
        key = (t.side, t.kind, QS.place_text(t.at))
        out["order"].append(key + (q.stake, q.goal, q.rank))
        l0 = {"status": s0, "strikers": pu(d0["strikers"]), "guards": pu(d0["guards"]), "open_exits": d0["open_exits"], "exits": d0["exits"]}
        if "struck" in d0:
            l0.update(struck=d0["struck"], cornered=d0["cornered"])
        out["level0"][key] = l0
        out["routes"][key] = sorted((_way(w), sorted(_route(x) for x in rs)) for w, rs in d1.get("routes", {}).items())
        l1 = {"status": s1, "ways": sorted(_way(w) for w in d1.get("supply", [])), "failed": sorted((f[0], f[1], f[2]) for f in d1.get("failed", [])), "failed_at": d1.get("failed_at"),
              "links": d1.get("links"), "reach": d1.get("reach"), "reach_geo": sorted(d1.get("reach_geo", ())), "far": [(h, u.kind, QS.place_text(u.at)) for h, u in d1.get("far", [])],
              "too_far": d1.get("too_far")}
        if q.goal and "scenarios" in d1:
            l1.update(open_exits=d1.get("open_exits"), near=d1.get("near"), scenarios=d1.get("scenarios"), moves=d1.get("moves"))
        out["level1"][key] = l1
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------- the Rust side, canonical

def canon_rust(res: dict) -> dict:
    out = {"units": [tuple(u) for u in res["units"]], "order": [tuple(o) for o in res["order"]], "level0": {}, "level1": {}, "routes": {}}
    for o, l0, l1 in zip(res["order"], res["level0"], res["level1"]):
        key = (o[0], o[1], o[2])
        c0 = {"status": l0["status"], "strikers": sorted(l0["strikers"]), "guards": sorted(l0["guards"]), "open_exits": l0["open_exits"], "exits": l0["exits"]}
        if "struck" in l0:
            c0.update(struck=l0["struck"], cornered=l0["cornered"])
        out["level0"][key] = c0
        out["routes"][key] = sorted((tuple(w), sorted(_rust_route(x) for x in rs)) for w, rs in l1["routes"])
        c1 = {"status": l1["status"], "ways": sorted(tuple(w) for w in l1["ways"]), "failed": sorted(tuple(f) for f in l1["failed"]), "failed_at": l1["failed_at"],
              "links": l1["links"], "reach": l1["reach"], "reach_geo": sorted(l1["reach_geo"]), "far": [tuple(f) for f in l1["far"]], "too_far": l1["too_far"]}
        if "scenarios" in l1:
            c1.update(open_exits=l1["open_exits"], near=l1["near"], scenarios=l1["scenarios"], moves=l1["moves"])
        out["level1"][key] = c1
    return out


def _rust_route(x) -> tuple:
    kind, at, places, stands, what, over, obs, goal = x
    return (kind, at, tuple(places), tuple((bool(ok), why) for ok, why in stands), what, tuple(tuple(o) for o in over), tuple((k, None if u is None else tuple(u), p, h, d) for k, u, p, h, d in obs), goal)


# ------------------------------------------------------------------------------------------------------------------------------------------------- comparing

def diff_position(py: dict, rs: dict) -> list:
    """[(what, key, field, python value, rust value)] of every difference."""
    out = []
    if py["units"] != rs["units"]:
        out.append(("units", None, "units", py["units"], rs["units"]))
    if py["order"] != rs["order"]:
        out.append(("units", None, "order", py["order"], rs["order"]))
    for key in py["level0"]:
        a, b = py["level0"][key], rs["level0"].get(key)
        if a != b:
            for f in sorted(set(a) | set(b or {})):
                if a.get(f) != (b or {}).get(f):
                    out.append(("level0", key, f, a.get(f), (b or {}).get(f)))
        if py["routes"][key] != rs["routes"].get(key):
            out.append(("routes", key, "routes", py["routes"][key], rs["routes"].get(key)))
        a, b = py["level1"][key], rs["level1"].get(key)
        if a != b:
            for f in sorted(set(a) | set(b or {})):
                if a.get(f) != (b or {}).get(f):
                    out.append(("level1", key, f, a.get(f), (b or {}).get(f)))
    return out


def cause_of(what: str, field: str) -> str:
    return f"to be read from the values ({what}.{field})"


def run_game(spec: dict) -> dict:
    st = _setup()
    env, worth, params, model = st["env"], st["worth"], st["params"], st["model"]
    g = spec["record"]
    v, start, moves = g.get("variant") or env.variant, g.get("start"), g["moves"]
    out = {"game": spec["index"], "positions": 0, "unreadable": 0, "python_error": 0, "units": [0, 0], "level0": [0, 0], "routes": [0, 0], "level1": [0, 0], "t_py": 0.0, "t_rs": 0.0, "t_core": 0.0, "diffs": [], "ways": 0, "route_count": 0}
    for ply in range(len(moves) + 1):
        pos = CE.state(v, start, moves[:ply])[0]
        t0 = time.perf_counter()
        try:
            py = python_side(pos)
        except ValueError:
            out["unreadable"] += 1
            continue
        except Exception as e:                                                                                # a position the Python sheet does not read: not compared
            out["python_error"] += 1
            out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "what": "python_error", "key": None, "field": type(e).__name__, "py": str(e)[:120], "rs": None})
            continue
        t_py = time.perf_counter() - t0
        t1 = time.perf_counter()
        try:
            res = SR.read(model, env, pos, params)
        except Exception as e:
            out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "what": "rust_error", "key": None, "field": type(e).__name__, "py": None, "rs": str(e)[:120]})
            out["positions"] += 1
            continue
        t_rs = time.perf_counter() - t1
        a, b = canon_python(py), canon_rust(res)
        out["positions"] += 1
        out["t_py"] += t_py
        out["t_rs"] += t_rs
        out["t_core"] += res["seconds"]
        n_units, n_targets = len(a["units"]), len(a["level0"])
        out["units"][0] += n_units + len(a["order"])
        out["level0"][0] += n_targets
        out["routes"][0] += n_targets
        out["level1"][0] += n_targets
        out["ways"] += sum(len(x["ways"]) for x in a["level1"].values())
        out["route_count"] += sum(len(rs) for x in a["routes"].values() for _w, rs in x)
        bad = {"units": set(), "level0": set(), "routes": set(), "level1": set()}
        for what, key, field, pv, rv in diff_position(a, b):
            bad[what].add(key if key is not None else field)
            out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "what": what, "key": key, "field": field, "py": repr(pv)[:300], "rs": repr(rv)[:300]})
        out["units"][1] += len(bad["units"])
        out["level0"][1] += len(bad["level0"])
        out["routes"][1] += len(bad["routes"])
        out["level1"][1] += len(bad["level1"])
    return out


def _pct(a: int, b: int) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "n/a"


def format_obs(results: list, wall: float, jobs: int) -> list:
    n = sum(r["positions"] for r in results)
    tp, tr, tc = sum(r["t_py"] for r in results), sum(r["t_rs"] for r in results), sum(r["t_core"] for r in results)
    lines = [f"OBS sheet_rust: games {len(results)}, positions compared {n}, unreadable (over) {sum(r['unreadable'] for r in results)}, python errors {sum(r['python_error'] for r in results)}, jobs {jobs}, wall {wall:.1f} s"]
    for label, k in (("units (units and first order entries)", "units"), ("level 0 verdicts (per target)", "level0"), ("level 1 route sets (per target)", "routes"), ("level 1 verdicts (per target)", "level1")):
        a = sum(r[k][0] for r in results)
        d = sum(r[k][1] for r in results)
        lines.append(f"OBS sheet_rust {label}: items compared {a}, items that differ {d}")
    lines.append(f"OBS sheet_rust ways {sum(r['ways'] for r in results)}, routes {sum(r['route_count'] for r in results)} in the Python results compared")
    if n:
        lines.append(f"OBS sheet_rust seconds per position (the same work, jobs {jobs}): python {tp / n:.5f}, rust {tr / n:.5f} (of which the engine's own work {tc / n:.5f}, the rest is handing the data to Python), ratio {tp / tr if tr else float('nan'):.1f}x")
    return lines


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--games", type=int, default=None, help="the first N games")
    ap.add_argument("--jobs", type=int, default=multiprocessing.cpu_count())
    ap.add_argument("--max-diffs", type=int, default=60)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    specs = [{"index": i, "record": g} for i, g in enumerate(_games(a.games_dir, a.games))]
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
    lines.append(f"OBS sheet_rust differences listed: {len(diffs)} (every one printed below up to --max-diffs {a.max_diffs})")
    for d in diffs[:a.max_diffs]:
        lines.append(f"DIFF game {d['game']} ply {d['ply']} {d['fen']} | {d['what']} {d['key']} {d['field']} | python {d['py']} | rust {d['rs']} | cause: {cause_of(d['what'], d['field'])}")
    for ln in lines:
        print(ln, flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
