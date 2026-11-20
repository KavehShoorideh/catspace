"""PLAN 79.14 PART E, THE 300 PUZZLES OF 79.09 AGAINST THE OLD PLAYER AT EQUAL WALL (SYSTEM_SPEC 8 the whole is judged, not the piece; piece: the wake). A PRINTING SCRIPT: it reads puzzle positions (the game's own text) and prints what the engine answered; it judges nothing and writes no number the run did not print.
The same 300 puzzles as 79.09 and 79.10 (`ledger_puzzles.dev_rows(314, offset=11)[14:]`), a puzzle is hit when the first move equals the listed answer. For each puzzle and each clock (0.2 s and 2.0 s): the old player at the clock given, and for each config of 79.14 (`whole_79_14.CONFIGS`:
b, bc) the wake of the config and the old player at the clock whose wall matches the wake's (`equal_wall.at_wall`, as 79.10). Run on a quiet machine, one run, 9 workers (a Pool from this real script file, one thread a worker).

  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_14_puzzles_print [--count 300] [--configs b,bc] [--tag full]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import random
import time

import chess

from catspace.incubator.analyst.discovery import equal_wall as EW
from catspace.incubator.analyst.discovery import ledger_puzzles as LP
from catspace.incubator.analyst.discovery import whole_79_14 as W14

CLOCKS = (0.2, 2.0)
_STATE: dict = {}


def init() -> None:
    _STATE["world"] = W14.world()
    _STATE["table"] = W14.table_for("full")


def job(item: tuple) -> dict:
    pid, text, answer, rating, themes = item
    w = _STATE["world"]
    out: dict = {"id": pid, "answer": answer, "rating": rating, "themes": themes}
    for secs in CLOCKS:
        o = w.old(text, secs)
        row: dict = {"old": o["move"], "old_wall": o["wall"], "old_depth": o.get("depth"), "arms": {}}
        for cfg in _STATE["configs"]:
            kw = {"goal_table": _STATE["table"]} if cfg != "b" else {}
            r = W14.Runner(w, W14.CONFIGS[cfg]).fresh(text, secs, **kw)
            e = EW.at_wall(lambda c: w.old(text, c), r["wall"])
            row["arms"][cfg] = {"wake": r["move"], "wall": r["wall"], "depth": r["depth"], "old_eq": e["row"]["move"], "old_eq_wall": e["wall"], "old_eq_depth": e["row"].get("depth"), "reached": e["reached"]}
        out[str(secs)] = row
    return out


def band(d: list) -> str:
    rng = random.Random(7)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(2000))
    return f"{100 * sum(d) / len(d):+.1f} pct [90 pct band {100 * bs[100]:+.1f}, {100 * bs[1900]:+.1f}]"


def lines_for(res: list, configs: list) -> list:
    out = []
    for secs in CLOCKS:
        R = [x[str(secs)] for x in res]
        ans = [x["answer"] for x in res]
        o = [int(r["old"] == a) for r, a in zip(R, ans)]
        out.append(f"OBS whole_79_14 puzzles {len(R)} clock {secs}: old player at the clock given: {sum(o)} ({100 * sum(o) / len(o):.1f} pct); mean wall {sum(r['old_wall'] for r in R) / len(R):.3f} s; mean depth {sum((r['old_depth'] or 0) for r in R) / len(R):.2f}")
        for cfg in configs:
            A = [r["arms"][cfg] for r in R]
            w = [int(a["wake"] == t) for a, t in zip(A, ans)]
            e = [int(a["old_eq"] == t) for a, t in zip(A, ans)]
            n = len(A)
            out.append(f"OBS whole_79_14 puzzles {n} clock {secs} config {cfg}: answers: wake {sum(w)} ({100 * sum(w) / n:.1f} pct), old player at equal wall {sum(e)} ({100 * sum(e) / n:.1f} pct); wake less old at equal wall {band([a - b for a, b in zip(w, e)])}; wake less old at the clock given {band([a - b for a, b in zip(w, o)])}; "
                       f"wake only {sum(1 for a, b in zip(w, e) if a > b)}, old only {sum(1 for a, b in zip(w, e) if b > a)}; mean wall: wake {sum(a['wall'] for a in A) / n:.3f} s, old at equal wall {sum(a['old_eq_wall'] for a in A) / n:.3f} s; equal wall reached in {sum(1 for a in A if a['reached'])}; mean depth: wake {sum(a['depth'] for a in A) / n:.2f}, old at equal wall {sum((a['old_eq_depth'] or 0) for a in A) / n:.2f}")
            R2 = [a for a in A if a["reached"]]
            if R2:
                idx = [i for i, a in enumerate(A) if a["reached"]]
                w2 = [w[i] for i in idx]
                e2 = [e[i] for i in idx]
                out.append(f"OBS whole_79_14 puzzles clock {secs} config {cfg}, only the {len(R2)} where the old player reached the wake's wall: wake {sum(w2)} ({100 * sum(w2) / len(w2):.1f} pct), old at equal wall {sum(e2)} ({100 * sum(e2) / len(e2):.1f} pct); wake less old {band([a - b for a, b in zip(w2, e2)])}")
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--count", type=int, default=300)
    ap.add_argument("--configs", default="b,bc")
    ap.add_argument("--tag", default="full")
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args(argv)
    root = W14.out_root(a.tag)
    configs = a.configs.split(",")
    _STATE["configs"] = configs
    rows = LP.dev_rows(14 + a.count, offset=11)[14:]
    items = []
    for r in rows:
        b = chess.Board(r["FEN"])
        sol = r["Moves"].split()
        b.push_uci(sol[0])
        items.append((r["PuzzleId"], b.fen(), sol[1], int(r["Rating"]), r["Themes"]))

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    load0, busy = TM.wait_quiet(say)
    say(f"OBS whole_79_14 puzzles: {len(items)} puzzles of 79.09 at the clocks {CLOCKS}, configs {configs}, workers {a.workers}, load average at start {load0:.2f}, cores busy {busy:.2f}")
    t0 = time.time()
    res: list = []
    with multiprocessing.get_context("spawn").Pool(a.workers, initializer=init_with, initargs=(configs,)) as pool:
        for k, x in enumerate(pool.imap_unordered(job, items), 1):
            res.append(x)
            if k % 50 == 0 or k == len(items):
                print(f"{k}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    (root / "puzzles.json").write_text(json.dumps(res))
    for x in lines_for(sorted(res, key=lambda r: r["id"]), configs):
        say(x)
    return 0


def init_with(configs: list) -> None:
    _STATE["configs"] = configs
    init()


if __name__ == "__main__":
    raise SystemExit(main())
