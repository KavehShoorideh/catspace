"""PLAN 79.16 PARTS A, E: THE SEARCH THAT DOES NOT STOP ON A FAILED FIRST MOVE, AND THE WHOLE JUDGED AT A CLOCK WHERE EVERY PIECE RUNS (SYSTEM_SPEC 8 the whole is judged not the piece and the rule for choosing a move, 7 the budget and point 2 the estimate per goal, 11 everything in the hot path fast; piece: the wake).
A sandbox: nothing is wired into any default of the engine, nothing is written to rules/learned or data/derived, no oracle is called, no player id is written; every output goes under artifacts/experiments/whole_79_16/ (through catspace.io.paths). Game-free: positions come from the decision logs of
79.14 (read only, from the main checkout); the wake is called through the binding and only its JSON is read; the harness is 79.15's (whole_79_15), 79.14's (whole_79_14) and 79.11's (whole_play). This module names no object of the game.

PART A (`parta`): the 79.15 part A decisions (the first move's probe 400 or more below the search's value in the logs of 79.14) are asked again with the wake at the clock of the matches with the reading on of a failed first move off (the wake of 79.15's mend) and on. Printed: how often the search asked its first move
again, how it ended (first_held, other_held, all_failed, cut), what it cost against what it was allowed, the depth answered, and how many lowered first moves the equal reading then found.
PART E: two configs of the wake (`CONFIGS`): `abc` the reading on (A), the clock to the search (B), the steps lean (C), the chance order off; `abcd` the same with the chance order on (D: the one order of fixes by stake times chance, behind the chance_order switch). Each is judged against the old player at the scale of equal
wall found afresh for the build and the clock (`sanity`), sets A and B (`judge`), settled by `tiebreak_measure.settled`.

INTERFACE (written before the bodies).
  CONFIGS, SEC_LONG                                        the configs; the long clock (2.0 s).
  asked16(folder) -> dict                                  the means a decision from the logs of one match: depth, frames BUILT (the steps' own count of frames built, not the questions asked of them), frames of the read, frame rows, frame questions, list rows and questions, the wall by step, and the reading on of a failed first move (how often, how it ended, its seconds and what it was allowed).
  cmd_parta / cmd_sanity / cmd_judge                       the stages (command line below).
  Called by: the command line, whole_79_16_print.py, tests/test_plan79_16.py.

  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_16 parta | sanity --config abc --seconds 0.2 [--pairs 10] | judge --config abc --set A --seconds 0.2 --pairs 100 [--min-pairs 60] | table
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import gzip
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from catspace.incubator.analyst.discovery import lowered_first as LF
from catspace.incubator.analyst.discovery import whole_79_14 as W14
from catspace.incubator.analyst.discovery import whole_79_15 as W15
from catspace.incubator.analyst.discovery import whole_play as WP
from catspace.io import paths

OUT = ("whole_79_16",)
READ_ON = "read_on"
BASE = list(W14.BASE_PIECES)
CONFIGS = {"abc": BASE + [READ_ON], "abcd": BASE + [W14.CHANCE_ORDER, READ_ON], "main": BASE + [W14.CHANCE_ORDER]}      # main: the wake as 79.15 built it plus the one order of D (no reading on): the control of part A
SEC_SHORT, SEC_LONG = 0.2, 2.0
_STATE: dict = {}


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def table() -> tuple:
    return W15.table()


def watchdog_for(seconds: float) -> float:
    """The seconds one game may take before its worker is ended: scaled to the clock (a game is about 60 decisions of each side)."""
    return max(300.0, 750.0 * seconds)


# ------------------------------------------------------------------------------------------------------------------------------ PART A

def init_a() -> None:
    _STATE["world"] = W14.world()
    _STATE["table"] = table()


def ask(text: str, seconds: float, pieces: list, **kw) -> dict:
    return W14.Runner(_STATE["world"], pieces).fresh(text, seconds, goal_table=_STATE["table"], **kw)


def summary_of(out: dict, first: str) -> dict:
    lg = out["log"]
    mv = {x["move"]: x for x in lg["moves"]}
    reads = lg.get("depth_reads") or []
    return {"depth": out["depth"], "value": out["value"], "search_first": out.get("first"), "first_lowered": lg.get("first_lowered"), "equal_from": lg.get("equal_from"), "failed_first": lg.get("failed_first"),
            "probe_of_logged_first": (mv.get(first) or {}).get("value"), "last_read_finished": reads[-1]["finished"] if reads else None, "wall": out["wall"], "played": out["move"],
            "tie": next((s["seconds"] for s in out["steps"] if s["step"] == 5), None)}


def a_job(job: dict) -> dict:
    res = {"job": job, "asks": {}}
    for name, pieces in (("without", CONFIGS["main"]), ("with", CONFIGS["abcd"])):
        res["asks"][name] = summary_of(ask(job["position"], job["seconds_clock"], pieces), job["first"])
    return res


def cmd_parta(a, say) -> None:
    jobs = W15.pick(list(W15.FOLDERS), a.count, a.gap)
    for j in jobs:
        j["seconds_clock"] = a.seconds
    say(f"OBS whole_79_16 part A: {len(jobs)} decisions of 79.15's part A (79.14's logs, the first move's probe {a.gap} or more below the search's value, at most two a game), each asked again once with the wake of 79.15 (no reading on) and once with the reading on, at {a.seconds} s, alone on a worker; the position's history and the carry are not given")
    import multiprocessing as mp
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(a.workers, initializer=init_a) as p:
        res = p.map(a_job, jobs, chunksize=1)
    root = out_root(a.tag)
    (root / "parta.json").write_text(json.dumps(res))
    n = len(res)
    w, x = [r["asks"]["without"] for r in res], [r["asks"]["with"] for r in res]
    say(f"OBS whole_79_16 part A: {n} decisions asked twice ({time.perf_counter() - t0:.0f} s)")
    ff = [y for y in x if y["failed_first"]]
    oc = Counter(y["failed_first"]["outcome"] for y in ff)
    say(f"OBS whole_79_16 part A: the search asked its first move again in {len(ff)} of {n} decisions; how it ended: {dict(oc)}; the seconds it took a decision where it asked {sum(y['failed_first']['seconds'] for y in ff) / max(1, len(ff)):.4f} (allowed {sum(y['failed_first']['allowed'] for y in ff) / max(1, len(ff)):.4f}); nodes {sum(y['failed_first']['nodes'] for y in ff) / max(1, len(ff)):.0f}")
    say(f"OBS whole_79_16 part A: the depth answered: without {sum(y['depth'] for y in w) / n:.2f}, with {sum(y['depth'] for y in x) / n:.2f}; the answer came from the depth the clock stopped in {sum(1 for y in ff if y['failed_first']['outcome'] in ('other_held', 'all_failed'))} decisions")
    lw, lx = sum(y["first_lowered"] is not None for y in w), sum(y["first_lowered"] is not None for y in x)
    say(f"OBS whole_79_16 part A: lowered first moves found by the equal reading: without {lw}, with {lx}; the move played differs between the two asks in {sum(u0['played'] != v0['played'] for u0, v0 in zip(w, x))} of {n}; the search's first move differs in {sum(u0['search_first'] != v0['search_first'] for u0, v0 in zip(w, x))}")
    say(f"OBS whole_79_16 part A: mean wall a decision: without {sum(y['wall'] for y in w) / n:.4f} s, with {sum(y['wall'] for y in x) / n:.4f} s (clock {a.seconds} s)")
    lows = [(r["asks"]["without"], r["asks"]["with"]) for r in res if r["asks"]["without"]["first_lowered"] is not None]
    say(f"OBS whole_79_16 part A: among the {len(lows)} decisions in which the wake without the reading on found a lowered first move: with it the first move is lowered in {sum(v0['first_lowered'] is not None for _, v0 in lows)}, the search's first move changed in {sum(u0['search_first'] != v0['search_first'] for u0, v0 in lows)}, the reading on was asked in {sum(bool(v0['failed_first']) for _, v0 in lows)}")


# ------------------------------------------------------------------------------------------------------------------------------ the logs

def _num(note: str, word: str) -> int:
    m = re.search(r"(\d+) " + re.escape(word), note)
    return int(m.group(1)) if m else 0


def asked16(folder) -> dict:
    """The means a decision from the logs of a folder: see the module doc."""
    n = 0
    tot: dict = defaultdict(float)
    oc: Counter = Counter()
    for r in LF.decisions(folder):
        w = r["wake"]
        n += 1
        tot["depth"] += w.get("depth") or 0
        tot["wall"] += r.get("wall", 0.0)
        for s in w["steps"]:
            note, piece = s["note"], s["piece"]
            if piece == "frames and the question read":
                tot["frames"] += _num(note, "frames,")
                tot["frame_rows"] += _num(note, "rows,")
                tot["frame_questions"] += _num(note, "questions,")
            elif piece == "part of the frame read: the frames built":
                tot["built"] += _num(note, "frames built")
            elif piece == "threat list of both sides":
                tot["list_rows_us"] += _num(note, "rows against us")
                tot["list_rows_them"] += _num(note, "against them")
                tot["list_questions"] += _num(note, "questions;")
            if s["step"] in (1, 2, 3, 4, 5, 6, 7, 0) and piece in ("frames and the question read", "threat list of both sides", "the whole decision", "search with the count of worths at its end, and the equal reading", "the choice among the moves held equal (the rule, the estimates)", "the lines of the named moves read for the claims, after the search"):
                tot["s_" + piece] += s["seconds"]
        ff = w["log"].get("failed_first")
        if ff:
            tot["ff"] += 1
            tot["ff_seconds"] += ff["seconds"]
            tot["ff_allowed"] += ff["allowed"]
            oc[ff["outcome"]] += 1
    d = {k: v / max(1, n) for k, v in tot.items() if k not in ("ff", "ff_seconds", "ff_allowed")}
    d["decisions"] = n
    d["ff_n"] = int(tot["ff"])
    d["ff_seconds"] = tot["ff_seconds"] / max(1, tot["ff"])
    d["ff_allowed"] = tot["ff_allowed"] / max(1, tot["ff"])
    d["ff_outcomes"] = dict(oc)
    return d


def asked_line(name: str, d: dict) -> str:
    return (f"OBS whole_79_16 {name}: asked a decision (means over {d['decisions']} decisions): depth {d['depth']:.2f}; frames BUILT {d.get('built', 0):.2f} (the step's own count of frames built; the read's frames {d.get('frames', 0):.2f}), frame rows {d.get('frame_rows', 0):.2f}, frame questions {d.get('frame_questions', 0):.1f}; "
            f"threat list rows {d.get('list_rows_us', 0):.2f} against us and {d.get('list_rows_them', 0):.2f} against them, its questions {d.get('list_questions', 0):.1f}; wall of the frame read {d.get('s_frames and the question read', 0):.4f} s, of the threat list {d.get('s_threat list of both sides', 0):.4f} s, "
            f"of the search with the equal reading {d.get('s_search with the count of worths at its end, and the equal reading', 0):.4f} s, of the choice {d.get('s_the choice among the moves held equal (the rule, the estimates)', 0):.4f} s, of the claims' lines {d.get('s_the lines of the named moves read for the claims, after the search', 0):.4f} s, of the whole decision {d.get('s_the whole decision', 0):.4f} s; "
            f"the search asked its first move again in {d['ff_n']} decisions ({100 * d['ff_n'] / max(1, d['decisions']):.1f} pct): {d['ff_outcomes']}, {d['ff_seconds']:.4f} s each of {d['ff_allowed']:.4f} s allowed")


# ------------------------------------------------------------------------------------------------------------------------------ PART E

def specs_of(config: str, set_name: str, pairs: int, seconds: float, scratch: Path, log_dir: Path, clock_b: float, name: str) -> list:
    return WP.match_specs("wake", "old", set_name, pairs, seconds, scratch, log_dir, clock_b=clock_b, label=name, table_a=table(), pieces_a=CONFIGS[config], pre_scale_a=1.0)


def cmd_sanity(a, say) -> float:
    os.environ["CATSPACE_WATCHDOG"] = str(watchdog_for(a.seconds))
    root = out_root(a.tag)
    name = f"sanity_{a.config}_{a.seconds}"
    log_dir = root / "logs" / name
    specs = WP.match_specs("wake", "old", "A", a.pairs, a.seconds, root / "scratch", log_dir, label=name, table_a=table(), pieces_a=CONFIGS[a.config], pre_scale_a=1.0)
    load0, busy = TM.wait_quiet(say)
    games = TM.run_games(specs, a.workers, root / "matches" / f"{name}.jsonl", say, play=WP.play11, stop=None)
    s = WP.scale_of(games)
    (root / f"scale_{a.config}_{a.seconds}.json").write_text(json.dumps({"scale": s, "seconds": a.seconds, "wake_wall": WP.mean_wall(games, 0), "old_wall": WP.mean_wall(games, 1), "games": len(games)}))
    say(f"OBS whole_79_16 {name}: ONE SCALE of equal wall for the config {a.config} at {a.seconds} s = {s:.4f} (the wake's mean wall a move {WP.mean_wall(games, 0):.4f} s over the old player's {WP.mean_wall(games, 1):.4f} s, {len(games)} games, load average at start {load0:.2f}); the old player is given {s * a.seconds:.4f} s for equal mean wall")
    say(asked_line(name, asked16(log_dir)))
    say(f"OBS whole_79_16 {name}: mean depth of the search a decision: wake {W14.depth_of(games, 0):.2f}, old player {W14.depth_of(games, 1):.2f}")
    return s


def cmd_judge(a, say) -> None:
    os.environ["CATSPACE_WATCHDOG"] = str(watchdog_for(a.seconds))
    root = out_root(a.tag)
    scale_ = json.loads((root / f"scale_{a.config}_{a.seconds}.json").read_text())["scale"]
    name = f"{a.config}_{a.seconds}_v_old_{a.set}"
    log_dir = root / "logs" / name
    specs = specs_of(a.config, a.set, a.pairs, a.seconds, root / "scratch", log_dir, scale_ * a.seconds, name)
    say(f"OBS whole_79_16 {name}: the wake config {a.config} ({' '.join(CONFIGS[a.config][-3:])}), against the old player at {scale_ * a.seconds:.4f} s (the scale {scale_:.4f}); clock {a.seconds} s, ply cap {TM.PLY_CAP}, early stop min_pairs={a.min_pairs}, workers {a.workers}, watchdog {watchdog_for(a.seconds):.0f} s a game")
    load0, busy = TM.wait_quiet(say)
    t0 = time.perf_counter()
    games = TM.run_games(specs, a.workers, root / "matches" / f"{name}.jsonl", say, play=WP.play11, stop=lambda g: TM.settled(g, min_pairs=a.min_pairs))
    wall = time.perf_counter() - t0
    for x in WP.lines_of(name, "wake", "old", a.set, games, load0, os.getloadavg()[0], busy, wall, a.seconds, a.workers, {"old": scale_ * a.seconds}, log_dir):
        say(x.replace("OBS whole ", "OBS whole_79_16 ", 1))
    say(f"OBS whole_79_16 {name}: mean depth of the search a decision: wake {W14.depth_of(games, 0):.2f}, old player {W14.depth_of(games, 1):.2f} (the wake's mean wall {WP.mean_wall(games, 0):.4f} s, the old player's {WP.mean_wall(games, 1):.4f} s)")
    say(LF.line(name, LF.count(log_dir)).replace("OBS lowered_first", "OBS whole_79_16 lowered_first", 1))
    say(asked_line(name, asked16(log_dir)))


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("parta", "sanity", "judge", "table"))
    ap.add_argument("--config", default="abc", choices=tuple(CONFIGS))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--min-pairs", type=int, default=60)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--gap", type=int, default=400)
    ap.add_argument("--seconds", type=float, default=SEC_SHORT)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    if a.stage == "table":
        t = json.loads(W15.table16_path().read_text())
        (root / "table16.json").write_text(json.dumps(t))
        say(f"OBS whole_79_16 table: the table of 79.14 C read from {W15.table16_path().parent}: the window of {t['k']} plies, {t['total'][0]} goals, {t['total'][1]} reached")
        return 0
    {"parta": cmd_parta, "sanity": cmd_sanity, "judge": cmd_judge}[a.stage](a, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
