"""PLAN 79.15 PARTS A, D, E: WHY THE PROBE AND THE SEARCH DISAGREE, THE SHARE OF THE STEPS BEFORE THE SEARCH JUDGED BY PLAY, AND THE WHOLE IN PLAY (SYSTEM_SPEC 8 the wall and the rule for choosing a move, 7 the budget and point 1 calculate what is near, 11; piece: the wake).
A sandbox: nothing is wired into any default of the engine, nothing is written to rules/learned or data/derived, no oracle is called, no player id is written; every output goes under artifacts/experiments/whole_79_15/ (through catspace.io.paths). Game-free: positions come from the decision logs of
79.14 (read only, from the main checkout); the wake is called through the binding and only its JSON is read; the harness is 79.14's (whole_79_14) and 79.11's (whole_play). This module names no object of the game.

PART A (`parta`): on `--count` logged decisions (79.14's logs) where the search's first move had a probe below the search's value by `--gap` or more, the position is asked again with the wake at the clock of the matches, at twice that clock and at eight times it, with the piece `diag_first` (a measure: after the
decision the first move is asked again with the late-move reduction and the null move off, and at the next two depths). Printed: the depth each reading was made at, whether the first move was a named one, whether the clock cut the next depth, whether the unreduced reading agrees with the probe or the search, and which of the two a longer reading agrees with.
PART D, E: the three SHARES of the steps before the search (as on main, a half of it, a quarter of it: the binding's `pre_share_scale`) are three configs of the same wake (`SHARES`), each with the wall kept (79.15 B) and the reason for a line on the goal unit (79.15 C) in the build, the chance order on with the table of 79.14, carry_graph off
(79.14: it is not equal to asking afresh). For each share the scale of equal wall is found afresh (`sanity`), then the match against the old player at that scale (`judge`), sets A and B, settled with min_pairs 60.

INTERFACE (written before the bodies).
  pick(dirs, count, gap, seed) -> [dict]                  the logged decisions to ask again: label, position, folder, first move, value, probe, gap, others_hold, named, depth; at most two a game.
  a_job(job) -> dict                                      module level for the pool: the asks of one decision.
  cmd_parta / cmd_sanity / cmd_judge                      the stages (command line below).
  SHARES                                                   {"s1": 1.0, "s2": 0.5, "s4": 0.25}: the name of a share and its scale of the binding's pre-share (the share of the clock the steps before the search may take; None: as on main).
  share_specs(...)                                         the games of one match of one share.
  Called by: the command line, whole_79_15_print.py, tests/test_plan79_15.py.

  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_15 parta [--count 100] [--gap 400] | sanity --share s1 | judge --share s1 --set A | table
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

from catspace.incubator.analyst.discovery import lowered_first as LF
from catspace.incubator.analyst.discovery import goal_table as GT
from catspace.incubator.analyst.discovery import whole_79_14 as W14
from catspace.incubator.analyst.discovery import whole_play as WP
from catspace.io import paths

OUT = ("whole_79_15",)
LOGS_79_14 = ("whole_79_14", "full", "logs")                     # the logged games of 79.14 (read only, in the main checkout)
FOLDERS = ("b_v_old_A", "b_v_old_B", "bc_v_old_A", "bc_v_old_B")
SHARES = {"s1": 1.0, "s2": 0.5, "s4": 0.25}
PIECES = list(W14.BASE_PIECES) + [W14.CHANCE_ORDER]               # every piece on (the pieces of 79.13 and the chance order of 79.14 C); carry_graph off (79.14: not equal to asking afresh)
CLOCKS_A = (1.0, 2.0, 8.0)                                        # the clock of the matches, twice it, eight times it (as multiples)
_STATE: dict = {}


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def logs_root() -> Path:
    return LF.main_checkout() / "artifacts" / "experiments" / Path(*LOGS_79_14)


def table16_path() -> Path:
    return LF.main_checkout() / "artifacts" / "experiments" / "whole_79_14" / "full" / "table16.json"


def table() -> tuple:
    """The table of the chance of a goal of 79.14 C (the window of 16 plies, the fit half of 79.13's rows), read from the main checkout and never refitted."""
    return GT.binding_form(json.loads(table16_path().read_text()))


# ------------------------------------------------------------------------------------------------------------------------------ PART A

def pick(dirs: list, count: int, gap: int, seed: int = 7915) -> list:
    """Logged decisions whose first move's probe is `gap` or more below the search's value, at most two a game, spread by a seeded shuffle."""
    rng = random.Random(seed)
    pool = []
    for d in dirs:
        for r in LF.decisions(logs_root() / d):
            x = LF.lowered_of(r)
            if x is not None and x["gap"] >= gap:
                pool.append((d, r, x))
    rng.shuffle(pool)
    per: Counter = Counter()
    out = []
    for d, r, x in pool:
        if per[r["_game"]] >= 2:
            continue
        per[r["_game"]] += 1
        out.append({"label": f"{r['_game']}:{r['ply']}", "folder": d, "position": r["position"], "first": x["first"], "value": x["value"], "probe": x["probe"], "gap": x["gap"], "others_hold": x["others_hold"], "named": x["named"], "depth": x["depth"], "played_first": x["played_first"]})
        if len(out) >= count:
            break
    return out


def init_a() -> None:
    _STATE["world"] = W14.world()
    _STATE["table"] = table()


def ask(text: str, seconds: float, folder: str, diag: bool, **kw) -> dict:
    cfg = "bc" if folder.startswith("bc") else "b"                 # the config the logged match was played with
    pieces = list(W14.CONFIGS[cfg]) + (["diag_first"] if diag else [])
    if cfg != "b":
        kw["goal_table"] = _STATE["table"]
    return W14.Runner(_STATE["world"], pieces).fresh(text, seconds, **kw)


def summary_of(out: dict, first: str) -> dict:
    lg = out["log"]
    mv = {x["move"]: x for x in lg["moves"]}
    reads = lg.get("depth_reads") or []
    return {"depth": out["depth"], "value": out["value"], "search_first": out.get("first"), "first_lowered": lg.get("first_lowered"), "equal_from": lg.get("equal_from"), "n_equal": len(out["equal"]),
            "probe_of_logged_first": (mv.get(first) or {}).get("value"), "named_logged_first": (mv.get(first) or {}).get("named"), "last_read_finished": reads[-1]["finished"] if reads else None, "last_read_depth": reads[-1]["depth"] if reads else None,
            "reads": [(x["depth"], x["finished"], round(x["seconds"], 4)) for x in reads[-3:]], "diag": lg.get("first_diag"), "wall": out["wall"], "played": out["move"]}


def a_job(job: dict) -> dict:
    res = {"job": job, "asks": {}}
    base = job["seconds_clock"]
    for k in CLOCKS_A:
        out = ask(job["position"], base * k, job["folder"], diag=True)
        res["asks"][str(k)] = summary_of(out, job["first"])
    out = ask(job["position"], base, job["folder"], diag=False)                   # the wake as built, asked as played (no measure after it): what it plays now
    res["mend"] = summary_of(out, job["first"])
    return res


def classify(job: dict, asks: dict) -> dict:
    """What the asks say about one decision (the words the OBS lines count)."""
    as_played = asks["1.0"]
    reproduced = as_played["search_first"] == job["first"] and as_played["first_lowered"] is not None
    d = as_played["diag"] or {}
    out = {"reproduced": reproduced, "cut_next": as_played["last_read_finished"] is False, "unreduced": d.get("unreduced_holds") if reproduced else None}
    deeper = (d.get("deeper") or [{}])[0].get("probe") if reproduced else None
    out["deeper1"] = None if deeper is None else ("probe" if deeper < d.get("value", 0) else "search")
    for k in ("2.0", "8.0"):
        a = asks[k]
        v = a["probe_of_logged_first"]
        if v is None:
            out["longer_" + k] = "unknown"
        else:
            out["longer_" + k] = "probe" if v < a["value"] else "search"
    return out


def cmd_parta(a, say) -> None:
    jobs = pick(list(FOLDERS), a.count, a.gap)
    clock = a.seconds
    for j in jobs:
        j["seconds_clock"] = clock
    say(f"OBS whole_79_15 part A: {len(jobs)} logged decisions (79.14's four matches, at most two a game) in which the search's first move had a probe {a.gap} or more below the search's value; each asked again with the wake at {clock} s (as played), {2 * clock} s (twice) and {8 * clock} s (eight times), alone on a worker, with the piece diag_first; the position's history is not given and the carry is not given")
    import multiprocessing as mp
    t0 = time.perf_counter()
    with mp.get_context("spawn").Pool(a.workers, initializer=init_a) as p:
        res = p.map(a_job, jobs, chunksize=1)
    root = out_root(a.tag)
    (root / "parta.json").write_text(json.dumps(res))
    for r in res:
        r["class"] = classify(r["job"], r["asks"])
    n = len(res)
    say(f"OBS whole_79_15 part A: {n} decisions asked ({time.perf_counter() - t0:.0f} s); the others hold at the logged reading in {sum(r['job']['others_hold'] for r in res)}; the first move was a named one in {sum(bool(r['job']['named']) for r in res)}; the first move was the move played in {sum(r['job']['played_first'] for r in res)}")
    base = {d: LF.named_rate(logs_root() / d) for d in FOLDERS}
    say("OBS whole_79_15 part A: the first move is a named one in " + ", ".join(f"{b[1]} of {b[0]} ({100 * b[1] / max(1, b[0]):.1f} pct) of all decisions of {d}" for d, b in base.items()) + " (the rate over all decisions, to put beside the share among the lowered ones above)")
    dh = Counter(r["job"]["depth"] for r in res)
    say("OBS whole_79_15 part A: the depth the search and the probe read at (the probe is asked at the depth the search finished): " + ", ".join(f"depth {k}: {v}" for k, v in sorted(dh.items())))
    rep = [r for r in res if r["class"]["reproduced"]]
    say(f"OBS whole_79_15 part A: asked again at the clock as played the same first move came out lowered again in {len(rep)} of {n} (the clock is not repeatable: a game's logged decision is one reading)")
    say(f"OBS whole_79_15 part A: of those {len(rep)}: the clock cut the next depth in {sum(r['class']['cut_next'] for r in rep)}; "
        f"the first move asked again with the late-move reduction and the null move off holds the search's value in {sum(r['class']['unreduced'] is True for r in rep)} (the probe was the reduced reading and is the one that differs), does not hold it in {sum(r['class']['unreduced'] is False for r in rep)} (the search's own reading of the line was the reduced one), cut {sum(r['class']['unreduced'] is None for r in rep)}")
    say(f"OBS whole_79_15 part A: of those {len(rep)}: the same first move asked at the next depth by the table kept stays below the value in {sum(r['class']['deeper1'] == 'probe' for r in rep)}, holds it in {sum(r['class']['deeper1'] == 'search' for r in rep)}, unknown {sum(r['class']['deeper1'] is None for r in rep)}")
    for k, nm in (("2.0", "twice as long"), ("8.0", "eight times as long")):
        c = Counter(r["class"]["longer_" + k] for r in res)
        cr = Counter(r["class"]["longer_" + k] for r in rep)
        say(f"OBS whole_79_15 part A: a reading {nm} (the depth {sum(r['asks'][k]['depth'] for r in res) / n:.2f} against {sum(r['asks']['1.0']['depth'] for r in res) / n:.2f} on the same decisions): the logged first move is below that reading's value (agrees with the probe) in {c['probe']}, holds it (agrees with the search) in {c['search']}, was not probed {c['unknown']}; among the {len(rep)} reproduced: probe {cr['probe']}, search {cr['search']}, unknown {cr['unknown']}")
    m = [r["mend"] for r in res]
    low = [x for x in m if x["first_lowered"] is not None]
    played_low = [x for x in low if x["played"] == x["search_first"] and x["equal_from"] in ("verified", "probe_only")]
    say(f"OBS whole_79_15 part A/B: the same {n} decisions asked once more with the wake as built (the wall kept), no measure after it: the search's first move came out lowered in {len(low)}; the candidate set was made by {dict(Counter(x['equal_from'] for x in low))}; the wake PLAYED the lowered first move in {sum(x['played'] == x['search_first'] for x in low)} of them (of those where another move held: {len(played_low)}); the move played differs from the search's first in {sum(x['played'] != x['search_first'] for x in m)} of {n} (the logged decisions played the first in {sum(r['job']['played_first'] for r in res)})")
    cs = Counter((r["job"]["named"], r["class"]["longer_2.0"]) for r in res)
    say("OBS whole_79_15 part A: by whether the first move was a named one (named, what twice as long says): " + ", ".join(f"{k}: {v}" for k, v in sorted(cs.items(), key=str)))


# ------------------------------------------------------------------------------------------------------------------------------ PARTS D, E

def share_specs(share: str, set_name: str, pairs: int, seconds: float, scratch: Path, log_dir: Path, scale_: float, name: str) -> list:
    """The games of one match of one share: the wake (every piece on, the table, the share's scale of the pre-share) against the old player at `scale_ x seconds`."""
    return WP.match_specs("wake", "old", set_name, pairs, seconds, scratch, log_dir, clock_b=scale_ * seconds, label=name, table_a=table(), pieces_a=PIECES, pre_scale_a=SHARES[share])


def cmd_sanity(a, say) -> float:
    """The scale of the equal wall of one share: the wake against the old player at one clock, 20 games (10 opening pairs of set A, no early stop)."""
    root = out_root(a.tag)
    log_dir = root / "logs" / f"sanity_{a.share}"
    specs = WP.match_specs("wake", "old", "A", 10, a.seconds, root / "scratch", log_dir, label=f"sanity_{a.share}", table_a=table(), pieces_a=PIECES, pre_scale_a=SHARES[a.share])
    load0, busy = TM.wait_quiet(say)
    games = TM.run_games(specs, a.workers, root / "matches" / f"sanity_{a.share}.jsonl", say, play=WP.play11, stop=None)
    s = WP.scale_of(games)
    (root / f"scale_{a.share}.json").write_text(json.dumps({"scale": s, "seconds": a.seconds, "wake_wall": WP.mean_wall(games, 0), "old_wall": WP.mean_wall(games, 1), "games": len(games)}))
    say(f"OBS whole_79_15 sanity {a.share}: ONE SCALE of equal wall for the share {SHARES[a.share]} of the pre-share = {s:.4f} (the wake's mean wall a move {WP.mean_wall(games, 0):.4f} s over the old player's {WP.mean_wall(games, 1):.4f} s at the clock {a.seconds} s, {len(games)} games, load average at start {load0:.2f}); the old player is given {s * a.seconds:.4f} s for equal mean wall")
    return s


def cmd_judge(a, say) -> None:
    root = out_root(a.tag)
    scale_ = json.loads((root / f"scale_{a.share}.json").read_text())["scale"]
    name = f"{a.share}_v_old_{a.set}"
    log_dir = root / "logs" / name
    specs = share_specs(a.share, a.set, a.pairs, a.seconds, root / "scratch", log_dir, scale_, name)
    say(f"OBS whole_79_15 {name}: the wake with every piece on, the wall kept and the reason for a line on the goal unit, the steps before the search at {SHARES[a.share]} of the share on main, against the old player at {scale_ * a.seconds:.4f} s (the scale {scale_:.4f}); clock {a.seconds} s, ply cap {TM.PLY_CAP}, early stop min_pairs={WP.MIN_PAIRS}, workers {a.workers}")
    load0, busy = TM.wait_quiet(say)
    t0 = time.perf_counter()
    games = TM.run_games(specs, a.workers, root / "matches" / f"{name}.jsonl", say, play=WP.play11, stop=lambda g: TM.settled(g, min_pairs=WP.MIN_PAIRS))
    wall = time.perf_counter() - t0
    for x in WP.lines_of(name, "wake", "old", a.set, games, load0, os.getloadavg()[0], busy, wall, a.seconds, a.workers, {"old": scale_ * a.seconds}, log_dir):
        say(x.replace("OBS whole ", "OBS whole_79_15 ", 1))
    say(f"OBS whole_79_15 {name}: mean depth of the search a decision: wake {W14.depth_of(games, 0):.2f}, old player {W14.depth_of(games, 1):.2f} (the wake's mean wall {WP.mean_wall(games, 0):.4f} s, the old player's {WP.mean_wall(games, 1):.4f} s)")
    say(LF.line(name, LF.count(log_dir)).replace("OBS lowered_first", "OBS whole_79_15 lowered_first", 1))
    d = LF.asked_of(log_dir)
    say(f"OBS whole_79_15 {name}: asked a decision (means over {d['decisions']} decisions): depth {d['depth']:.2f}; frames built {d.get('frames', 0):.1f}, frame rows {d.get('frame_rows', 0):.1f}, frame questions {d.get('frame_questions', 0):.1f}; threat list rows {d.get('list_rows_us', 0):.1f} against us and {d.get('list_rows_them', 0):.1f} against them, its questions {d.get('list_questions', 0):.1f}; "
        f"wall of the frame read {d.get('s_frames and the question read', 0):.4f} s, of the threat list {d.get('s_threat list of both sides', 0):.4f} s, of the search with the equal reading {d.get('s_search with the count of worths at its end, and the equal reading', 0):.4f} s, of the whole decision {d.get('s_the whole decision', 0):.4f} s")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("parta", "sanity", "judge", "table"))
    ap.add_argument("--share", default="s1", choices=tuple(SHARES))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--gap", type=int, default=400)
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    if a.stage == "table":
        t = json.loads(table16_path().read_text())
        (root / "table16.json").write_text(json.dumps(t))
        say(f"OBS whole_79_15 table: the table of 79.14 C read from {table16_path().parent} and kept here: the window of {t['k']} plies, {t['total'][0]} goals, {t['total'][1]} reached")
        return 0
    {"parta": cmd_parta, "sanity": cmd_sanity, "judge": cmd_judge}[a.stage](a, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
