"""PLAN 78.43 STAGE C1, THE MEASURE OF THE QUESTION OF THE MOVE AND THE LOSS OF A UNIT: the Rust `SheetModel.move_sheet` / `unit_loss` (rust/engine/tactics/src/sheet/mv.rs, via sheet_rust.py)
against the Python light sheet (question_sheet.py, the oracle: `Reading(light=True)`, `move_sheet`, `unit_loss`, `charged_ply`, the defaults of `Params()`), on every position of every game of a
games directory (games.jsonl: one JSON record per line with `variant`, `start`, `moves`).

  (1) THE MOVE QUESTION: for every position where the Python light reading reads, for EVERY legal move: the verdict kind, the three answers (what it leaves, where it lands, what the other
      side then has: the words), what it took and the credit of a unit that arrives as another kind, the units it took, every threat (unit, place, part, kind, value, line, whether open, words),
      what was left open, the unit the verdict names, the bounds and the line and the reason; and the order the moves are asked in. Differences are counted per field.
  (2) THE LOSS OF A UNIT: for every ply of every game and every unit that stands there, at the two horizons (4 and 8 plies): `unit_loss` (status, balance, place, ply, run, arrival) and, for a
      lost unit, `charged_ply`. The events are the Python codec's own (`codec.events`); the Rust form takes them as plain data.
The seconds: per move and per position of each side, the Python side the `move_sheet` of a fresh reading (what the sheet itself pays), the Rust side the whole call (the engine's own seconds in
brackets, the rest is making the position and handing the data to Python).

  .venv/bin/python -m catspace.incubator.analyst.discovery.sheet_rust_move_measure --games-dir DIR [--games N] [--jobs N] [--out FILE]
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

GAME = "chess"
HORIZONS = (4, 8)
_STATE: dict = {}


def _setup() -> dict:
    if not _STATE:
        env = QS.Env(GAME)
        worth = QS.store_worth(GAME)
        params = QS.Params(compare_old=False)
        _STATE.update(env=env, worth=worth, params=params, model=SR.model(env, worth), worth_of=env.worth_fn(worth))
    return _STATE


# ------------------------------------------------------------------------------------------------------------------------------------------------- canonical forms of one move's answer

def _u(u) -> tuple:
    return (u.side, u.kind, QS.place_text(u.at))


def canon_python(res: dict) -> dict:
    n = res.get("named")
    return {
        "move": res["move"], "unit": _u(res["unit"]), "dest": res["dest"], "took": float(res["took"]), "credit": float(res["credit"]), "took_units": [_u(x) for x in res["took_units"]],
        "a": res["answers"]["a"], "b": res["answers"]["b"], "c": res["answers"]["c"],
        "threats": [(_u(t["unit"]), t["place"], t["part"], t["kind"], float(t["value"]), tuple(t["line"]), bool(t["open"]), t["words"]) for t in res["threats"]],
        "unsettled": [(_u(x["unit"]), x["part"], x["words"], float(x["stake"])) for x in res["unsettled"]],
        "verdict": res["verdict"], "named": None if n is None else (_u(n["unit"]), n["place"], n["part"], n["words"]),
        "low": float(res["low"]), "high": float(res["high"]), "line": tuple(res["line"]), "because": res["because"],
    }


def canon_rust(res: dict) -> dict:
    n = res["named"]
    return {
        "move": res["move"], "unit": tuple(res["unit"]), "dest": res["dest"], "took": float(res["took"]), "credit": float(res["credit"]), "took_units": [tuple(x) for x in res["took_units"]],
        "a": res["answers"][0], "b": res["answers"][1], "c": res["answers"][2],
        "threats": [(tuple(t[0]), t[1], t[2], t[3], float(t[4]), tuple(t[5]), bool(t[6]), t[7]) for t in res["threats"]],
        "unsettled": [(tuple(x[0]), x[1], x[2], float(x[3])) for x in res["unsettled"]],
        "verdict": res["verdict"], "named": None if n is None else (tuple(n[0]), n[1], n[2], n[3]),
        "low": float(res["low"]), "high": float(res["high"]), "line": tuple(res["line"]), "because": res["because"],
    }


def diff_moves(a: dict, b: dict) -> list:
    return [(f, a.get(f), b.get(f)) for f in sorted(set(a) | set(b)) if a.get(f) != b.get(f)]


# ------------------------------------------------------------------------------------------------------------------------------------------------- the loss of a unit

def compare_losses(st: dict, v: str, start: "str | None", moves: list, out: dict, spec: dict) -> None:
    env, model, worth_of = st["env"], st["model"], st["worth_of"]
    events = env.codec.events(start, moves)
    wire = SR.event_wire(events)
    by_ply, taken = env.codec.tracks(start, moves)
    for ply in range(len(moves) + 1):
        for place, (uid, _side, _kind) in by_ply[ply].items():
            for h in HORIZONS:
                t0 = time.perf_counter()
                py = QS.unit_loss(events, uid, ply, h, worth_of)
                t1 = time.perf_counter()
                rs = SR.unit_loss(model, wire, uid, ply, h)
                t2 = time.perf_counter()
                out["l_py"] += t1 - t0
                out["l_rs"] += t2 - t1
                out["loss_calls"] += 1
                if py is None and rs is None:
                    out["loss_none"] += 1
                    continue
                out["loss_found"] += 1
                if py is None or rs is None:
                    out["loss_diff_calls"] += 1
                    out["loss_diff"][("found", "presence")] += 1
                    out["diffs"].append({"game": spec["index"], "ply": ply, "fen": place, "move": f"unit {uid} h {h}", "field": "loss presence", "py": repr(py)[:200], "rs": repr(rs)[:200]})
                    continue
                py_c = QS.charged_ply(events, py) if py["status"] == QS.LOST else None
                pyc = (py["status"], float(py["balance"]), py["place"], py["ply"], list(py["run"]), py["arrival"], py_c)
                rsc = (rs["status"], float(rs["balance"]), rs["place"], rs["ply"], list(rs["run"]), rs["arrival"], rs["charged"])
                out["loss_status"][py["status"]] += 1
                if py["status"] == QS.LOST:
                    out["loss_charged"] += 1
                if pyc != rsc:
                    out["loss_diff_calls"] += 1
                    for name, x, y in zip(("status", "balance", "place", "ply", "run", "arrival", "charged"), pyc, rsc):
                        if x != y:
                            out["loss_diff"][("found", name)] += 1
                    out["diffs"].append({"game": spec["index"], "ply": ply, "fen": place, "move": f"unit {uid} h {h}", "field": "loss", "py": repr(pyc)[:200], "rs": repr(rsc)[:200]})


# ------------------------------------------------------------------------------------------------------------------------------------------------- one game

def run_game(spec: dict) -> dict:
    st = _setup()
    env, worth, params, model = st["env"], st["worth"], st["params"], st["model"]
    g = spec["record"]
    v, start, moves = g.get("variant") or env.variant, g.get("start"), g["moves"]
    out = {"game": spec["index"], "positions": 0, "unreadable": 0, "python_error": 0, "rust_error": 0, "moves": 0, "move_diff": 0, "field_diff": collections.Counter(), "diffs": [], "order_diff": 0,
           "t_py": 0.0, "t_rs": 0.0, "t_core": 0.0, "verdicts": collections.Counter(), "l_py": 0.0, "l_rs": 0.0, "loss_calls": 0, "loss_found": 0, "loss_none": 0, "loss_diff": collections.Counter(),
           "loss_status": collections.Counter(), "loss_charged": 0, "loss_diff_calls": 0}
    for ply in range(len(moves) + 1):
        pos = CE.state(v, start, moves[:ply])[0]
        try:
            r = QS.Reading(env, pos, worth, light=True)
        except ValueError:
            out["unreadable"] += 1
            continue
        try:
            t0 = time.perf_counter()
            ms = QS.move_sheet(env, r, params)
            t_py = time.perf_counter() - t0
        except Exception as e:
            out["python_error"] += 1
            out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "move": None, "field": "python_error", "py": f"{type(e).__name__}: {e}"[:200], "rs": None})
            continue
        try:
            t1 = time.perf_counter()
            res = SR.move_sheet(model, env, pos, params)
            t_rs = time.perf_counter() - t1
        except Exception as e:
            out["rust_error"] += 1
            out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "move": None, "field": "rust_error", "py": None, "rs": f"{type(e).__name__}: {e}"[:200]})
            continue
        out["positions"] += 1
        out["t_py"] += t_py
        out["t_rs"] += t_rs
        out["t_core"] += res["seconds"]
        if list(ms.order) != list(res["order"]):
            out["order_diff"] += 1
            out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "move": None, "field": "order", "py": repr(ms.order)[:200], "rs": repr(res["order"])[:200]})
        py_by = {x["move"]: canon_python(x) for x in ms.results}
        rs_by = {x["move"]: canon_rust(x) for x in res["results"]}
        for mv in sorted(set(py_by) | set(rs_by)):
            out["moves"] += 1
            a, b = py_by.get(mv), rs_by.get(mv)
            if a is not None:
                out["verdicts"][a["verdict"]] += 1
            if a is None or b is None:
                out["move_diff"] += 1
                out["field_diff"]["presence"] += 1
                out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "move": mv, "field": "presence", "py": a is not None, "rs": b is not None})
                continue
            d = diff_moves(a, b)
            if d:
                out["move_diff"] += 1
                for f, x, y in d:
                    out["field_diff"][f] += 1
                    out["diffs"].append({"game": spec["index"], "ply": ply, "fen": pos, "move": mv, "field": f, "py": repr(x)[:240], "rs": repr(y)[:240]})
    compare_losses(st, v, start, moves, out, spec)
    return out


def format_obs(results: list, wall: float, jobs: int, label: str) -> list:
    n = sum(r["positions"] for r in results)
    mv = sum(r["moves"] for r in results)
    tp, tr, tc = sum(r["t_py"] for r in results), sum(r["t_rs"] for r in results), sum(r["t_core"] for r in results)
    fd = sum((r["field_diff"] for r in results), collections.Counter())
    vd = sum((r["verdicts"] for r in results), collections.Counter())
    ld = sum((r["loss_diff"] for r in results), collections.Counter())
    ls = sum((r["loss_status"] for r in results), collections.Counter())
    lines = [f"OBS sheet_rust_move {label}: games {len(results)}, positions {n}, unreadable (over) {sum(r['unreadable'] for r in results)}, python errors {sum(r['python_error'] for r in results)}, rust errors {sum(r['rust_error'] for r in results)}, jobs {jobs}, wall {wall:.1f} s"]
    lines.append(f"OBS sheet_rust_move {label}: moves compared {mv}, moves that differ {sum(r['move_diff'] for r in results)}, positions whose order of moves differs {sum(r['order_diff'] for r in results)}; Python verdicts " + ", ".join(f"{k} {c}" for k, c in sorted(vd.items())))
    lines.append(f"OBS sheet_rust_move {label}: differences per field: " + (", ".join(f"{f} {c}" for f, c in sorted(fd.items())) if fd else "none"))
    if n and mv:
        lines.append(f"OBS sheet_rust_move {label}: seconds per move (jobs {jobs}): python {tp / mv:.6f}, rust {tr / mv:.6f} (engine {tc / mv:.6f}); per position: python {tp / n:.5f}, rust {tr / n:.5f} (engine {tc / n:.5f}); ratio {tp / tr if tr else float('nan'):.1f}x")
    lc = sum(r["loss_calls"] for r in results)
    lines.append(f"OBS sheet_rust_move {label}: unit_loss calls {lc} (every unit at every ply, horizons {HORIZONS}), taken within the horizon {sum(r['loss_found'] for r in results)} ({', '.join(f'{k} {c}' for k, c in sorted(ls.items()))}), "
                 f"charged plies read {sum(r['loss_charged'] for r in results)}, calls that differ {sum(r['loss_diff_calls'] for r in results)}, per field: "
                 + (", ".join(f"{f[1]} {c}" for f, c in sorted(ld.items())) if ld else "none"))
    if lc:
        lines.append(f"OBS sheet_rust_move {label}: seconds per unit_loss call (jobs {jobs}): python {sum(r['l_py'] for r in results) / lc:.7f}, rust {sum(r['l_rs'] for r in results) / lc:.7f} (the rust call includes handing the game's events over each time)")
    return lines


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--games-dir", required=True)
    ap.add_argument("--games", type=int, default=None, help="the first N games")
    ap.add_argument("--jobs", type=int, default=multiprocessing.cpu_count())
    ap.add_argument("--max-diffs", type=int, default=40)
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
    label = Path(a.games_dir).name
    lines = format_obs(results, time.perf_counter() - t0, min(a.jobs, len(specs)), label)
    diffs = [d for r in results for d in r["diffs"]]
    lines.append(f"OBS sheet_rust_move {label}: differences listed {len(diffs)} (printed up to --max-diffs {a.max_diffs})")
    for d in diffs[:a.max_diffs]:
        lines.append(f"DIFF game {d['game']} ply {d['ply']} {d['fen']} | move {d['move']} | {d['field']} | python {d['py']} | rust {d['rs']}")
    for ln in lines:
        print(ln, flush=True)
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
