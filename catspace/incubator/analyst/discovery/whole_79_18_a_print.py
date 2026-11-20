"""PLAN 79.18 PART A (SYSTEM_SPEC 7 the measure of progress, 8 the wake): THE CAUSE FIRST, a reading of logs and no engine change. For every game of the one-unit class that the wake with the held goal did not win (plan 79.17 part E, or the cells of 79.18 part D), print ply by ply the measure's value now,
the lowest held, whether it was lowered, how many plies since, what the question found (first move, of how many of our moves, cut by the clock), what was played, and where the first position repeats; then, for the game, the facts a cause is read from.
A `*_print.py` script: it may show positions and moves as the game writes them. Reads logs only; writes only under artifacts/experiments/whole_79_18/.

INTERFACE: unwon(cell_dir, matches_file) -> [gid]; game_rows(path) -> [decision dicts]; facts(rows) -> dict; print_game(...) -> list of lines.
  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_18_a_print --logs DIR [DIR ...] --out FILE
Called by: the command line, whole_79_18.py (part D reads its own cells with it).
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


def rows_of(path: Path) -> list:
    out = []
    with gzip.open(path, "rt") as f:
        for x in f:
            try:
                out.append(json.loads(x))
            except ValueError:
                continue
    return out


def unwon(log_dir: Path) -> list:
    """The games of a folder whose result row is not a win of the side the wake plays (the wake's seat is in the result row; a win is the outcome of its colour)."""
    out = []
    for p in sorted(Path(log_dir).glob("*.jsonl.gz")):
        rows = rows_of(p)
        res = [r for r in rows if r.get("kind") == "result"]
        if not res:
            continue
        r = res[-1]
        decs = [d for d in rows if d.get("kind") == "decision"]
        seat_white = decs[0]["position"].split()[1] == "w" if decs else True
        won = r["outcome"] is not None and r["outcome"] == (100 if seat_white else 0)
        if not won:
            out.append(p)
    return out


def key4(fen: str) -> str:
    return " ".join(fen.split()[:4])


def facts(rows: list) -> dict:
    decs = [d for d in rows if d.get("kind") == "decision"]
    ms = [(d, d["wake"]["log"].get("measure")) for d in decs]
    ms = [(d, m) for d, m in ms if m]
    seen: dict = {}
    first_repeat = None
    for i, d in enumerate(decs):
        k = key4(d["position"])
        if k in seen and first_repeat is None:
            first_repeat = (i, seen[k])
        seen.setdefault(k, i)
    drop_ix = next((i for i, (_, m) in enumerate(ms) if m["dropped"]), None)
    above = [i for i, (_, m) in enumerate(ms) if m["held_before"] is not None and m["value_now"] > m["held_before"]]
    toward_above = [i for i in above if ms[i][1]["played_toward"]]
    toward_not_reached_with_repeat = 0
    cut = sum(1 for _, m in ms if m["cut"])
    named_none = sum(1 for _, m in ms if m["first"] is None and not m["dropped"])
    after_drop = 0 if drop_ix is None else len(ms) - drop_ix - 1 + (0 if ms[drop_ix][1]["dropped"] else 1)
    return {"decisions": len(decs), "measured": len(ms), "first_repeat": first_repeat, "dropped_at": drop_ix, "dropped_why": (ms[drop_ix][1]["dropped"] if drop_ix is not None else None),
            "decisions_after_drop": after_drop, "decisions_value_above_lowest": len(above), "toward_played_while_above": len(toward_above), "cut": cut, "no_line_named_goal_held": named_none, "reached": sum(1 for _, m in ms if m["reached"]),
            "toward_played": sum(1 for _, m in ms if m["played_toward"]), "unused": toward_not_reached_with_repeat}


def print_game(p: Path, rows: list) -> list:
    res = [r for r in rows if r.get("kind") == "result"][-1]
    decs = [d for d in rows if d.get("kind") == "decision"]
    f = facts(rows)
    out = [f"GAME {p.name}: plies {res['plies']}, outcome {res['outcome']}, unfinished {res['unfinished']}; facts {json.dumps(f)}"]
    seen: dict = {}
    for i, d in enumerate(decs):
        m = d["wake"]["log"].get("measure")
        k = key4(d["position"])
        rep = f" REPEATS ply {seen[k]}" if k in seen else ""
        seen.setdefault(k, d["ply"])
        if not m:
            out.append(f"  ply {d['ply']:3d} {d['position']} played {d['played']} (no measure entry)")
            continue
        out.append(f"  ply {d['ply']:3d} {d['position']} now {m['value_now']} held_before {m['held_before']} reached {int(m['reached'])} since {m['since']} first {m['first']} moves {m['moves']} cut {int(m['cut'])} toward {int(m['played_toward'])} dropped {m['dropped'] and m['dropped'][:40]} "
                   f"played {d['played']} search_first {d['search_first']} depth {d['wake']['depth']} value {d['wake']['value']}{rep}")
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs", default=None)
    a = ap.parse_args(argv)
    lines = []
    obs = []
    for d in a.logs:
        lines.append(f"CELL {d}")
        paths = unwon(Path(d))
        lines.append(f"  unwon games: {len(paths)}")
        obs.append(f"OBS whole_79_18 partA cell {Path(d).name}: {len(paths)} games not won")
        for p in paths:
            rows = rows_of(p)
            lines += print_game(p, rows)
            f = facts(rows)
            res = [r for r in rows if r.get("kind") == "result"][-1]
            ended = "quiet-ply rule" if res["plies"] >= 100 else "repetition"
            obs.append(f"OBS whole_79_18 partA game {p.name.split('_one')[-1].split('_0.jsonl')[0]}: plies {res['plies']}, ended by {ended}, decisions {f['decisions']}, first repeat (decision, of) {f['first_repeat']}, goal dropped at decision {f['dropped_at']} ({(f['dropped_why'] or '-')[:48]}), "
                       f"decisions after the drop {f['decisions_after_drop']}, decisions with the value above the lowest held {f['decisions_value_above_lowest']} of which a move toward the goal was played {f['toward_played_while_above']}, decisions the clock cut the question {f['cut']}, falls {f['reached']}")
    if a.obs:
        with open(a.obs, "w") as fo:
            fo.write("\n".join(obs) + "\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(lines) + "\n")
    for ln in lines:
        if ln.startswith("GAME") or ln.startswith("CELL") or "unwon" in ln:
            print(ln)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
