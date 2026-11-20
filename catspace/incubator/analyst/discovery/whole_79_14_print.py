"""PLAN 79.14 PART E, THE PRINTS (SYSTEM_SPEC 8 the whole is judged, not the piece, the wake step by step; piece: the wake). A PRINTING SCRIPT: it may use the game's own words (the position text, the units) because it only prints what the engine logged. It judges nothing and writes no number the logs do not hold.
  steps TAG RUN [RUN ...]       the mean wall of each step of section 8 over the decisions of the logged games of the runs (the table of part A and B), with the steps that name parts of other steps (22, 23) apart, and the share of the rows marked carried.
  twenty TAG RUN [--count N]    twenty decisions of the logged games in which the move of the wake differs from the old player's at the same position (the old player at the clock of the equal wall), with the board, the reasons the wake logged and the old player's move and depth, to be read against the board.
INTERFACE: `step_lines(root, runs)` and `twenty_lines(...)` return the lines they print. Called by the command line.
  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_14_print steps full b_v_old_A b_v_old_B | twenty full b_v_old_A
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import whole_79_14 as W14


def decisions_of(log_dir: Path, limit: "int | None" = None) -> list:
    out = []
    for p in sorted(Path(log_dir).glob("*.jsonl.gz")):
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "decision":
                    out.append(r)
        if limit and len(out) >= limit:
            break
    return out


def step_lines(root: Path, runs: list, label: str) -> list:
    tot: dict = defaultdict(float)
    cnt: Counter = Counter()
    n = 0
    rows = carried = 0
    depth = 0
    for run in runs:
        for d in decisions_of(root / "logs" / run):
            n += 1
            depth += d["wake"]["depth"]
            for s in d["wake"]["steps"]:
                key = (s["step"], s["piece"] + (" [" + s["note"].split(":")[0] + "]" if s["step"] == 23 else ""))
                tot[key] += s["seconds"]
                cnt[key] += 1
            for r in d["wake"]["log"]["rows"]:
                rows += 1
                carried += bool(r["carried"])
    whole = sum(v for (st, _p), v in tot.items() if st == 0) or 1.0
    out = [f"OBS whole_79_14 steps {label} {'+'.join(runs)}: mean wall of each step over {n} decisions (seconds, the share of the whole decision, the decisions the step ran in); mean depth {depth / max(1, n):.2f}; rows of the threat lists logged {rows}, marked carried {carried}"]
    for (st, piece), v in sorted(tot.items()):
        out.append(f"OBS whole_79_14 step {st} {piece}: mean {v / max(1, n):.4f} s, {100 * v / whole:.1f} pct, ran in {cnt[(st, piece)]} decisions")
    pre = sum(v for (st, _p), v in tot.items() if st in (1, 2, 3, 4))
    out.append(f"OBS whole_79_14 steps {label}: steps 1 to 4 together {pre / max(1, n):.4f} s, {100 * pre / whole:.1f} pct of the whole decision")
    return out


def board_text(position: str) -> str:
    try:
        return str(chess.Board(position))
    except Exception:
        return position


def twenty_lines(root: Path, run: str, count: int, world, old_clock: float) -> list:
    out = [f"OBS whole_79_14 twenty {run}: decisions of the wake whose move differs from the old player's at the same position (the old player at {old_clock:.4f} s), the first {count} in game order from ply 10 on; the board is as the engine logged it (white to move is up the board, upper case white)"]
    k = 0
    for d in decisions_of(root / "logs" / run):
        if d["ply"] < 10 or k >= count:
            continue
        o = world.old(d["position"], old_clock)
        if o["move"] == d["played"]:
            continue
        k += 1
        w, lg = d["wake"], d["wake"]["log"]
        out += [f"=== {k}. game {d['gid']} ply {d['ply']}: the wake plays {d['played']}, the old player {o['move']} (depth {o['depth']}, value {o['value']}); the wake's search: depth {w['depth']}, value {w['value']}, held equal {' '.join(w['equal'][:8])}",
                board_text(d["position"]), f"    position: {d['position']}",
                f"    THE WAKE'S REASON: {lg['why']}", f"    their largest standing threat: {lg['their_threat'] or 'none'}; named for us: {' '.join(w['named_mine'][:8]) or '-'}; for them: {' '.join(w['named_theirs'][:6]) or '-'}"]
        for m in lg["moves"]:
            if m["mv"] in (d["played"], o["move"]):
                out.append(f"    move {m['mv']}: held equal {m['equal']}, named {m['named']}, value {m['value']}, serves {m['serves'][:2] or '-'}, answers {m['answers'][:2] or '-'}, chance {m['chance']}")
        out.append("    TIME BY STEP: " + "; ".join(f"{s['step']} {s['piece'].split(':')[0][:40]} {s['seconds']:.3f}" for s in w["steps"] if s["step"] < 20))
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=("steps", "twenty"))
    ap.add_argument("tag")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--label", default="")
    a = ap.parse_args(argv)
    root = W14.out_root(a.tag)
    if a.which == "steps":
        lines = step_lines(root, a.runs, a.label)
    else:
        import json as _j
        scale = _j.loads((root / "scale.json").read_text())["scale"]
        lines = twenty_lines(root, a.runs[0], a.count, W14.world(), scale * 0.2)
    text = "\n".join(lines)
    print(text)
    (root / f"{a.which}_{'_'.join(a.runs)}_print.txt").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
