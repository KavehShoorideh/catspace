"""PLAN 79.14 PARTS A, B, E: THE COST OF THE STEPS BEFORE THE SEARCH, THEIR CARRY, AND THE GAMES (SYSTEM_SPEC 8 the wake step by step and "a piece that costs too much is made cheaper, not turned off", 4 the order of questions, 7 points 1 and 2; piece: the wake).
A sandbox: nothing is wired into any default of the engine, nothing is written to rules/learned or data/derived, no oracle is called, no player id is written; every output goes under artifacts/experiments/whole_79_14/ (through catspace.io.paths).
Game-free: positions come from the decision logs of the earlier plans (read only); the wake is called through the binding and only its JSON is read. This module names no object of the game.

PART A (profile): `profile` runs the wake at the play clock on consecutive decisions of one seat of a logged game (the slot keeps what the first decision left open for the second) and prints the seconds of each step and of the parts of the frame read and of the threat list (the parts are
logged by the wake as steps 22 and 23), and how many rows the carry held.
PART B (carry test): `carrytest` asks the same second position twice at a generous clock, once with the first decision's carry and once afresh, and compares what the steps before the search produced (the rows of the threat list, the goals, the names). A difference is printed with its row.

INTERFACE (written before the bodies).
  pairs_from_logs(dirs, count, seed) -> [(position_a, position_b, label)]    consecutive decisions of one seat (two plies apart) from the logs of the earlier plans, spread over games and plies.
  Runner(world, pieces)                                       wraps `World.wake` with a slot: `decide(text, seconds, slot, **kw)` -> the decision (JSON), its wall; `fresh(text, seconds, **kw)` with no slot.
  step_table(decisions) -> [(step, piece, mean seconds, share of the whole decision, count)]    the table of the steps.
  profile_job(job) / carry_job(job)                           module level, for the pool.
  CONFIGS                                                     PART E: the whole with B (`b`), with B and C (`bc`: the table of the window of 16 plies, the named moves in the order of stake times chance), with B, C and mode Gain mended (`bcg`, only if D's tests pass).
  table16(tag) -> table                                       the table of the chance of a goal at the window of 16 plies counted on the fit half of 79.13's goal rows (goal_table.py), written to <out>/<tag>/table16.json.
  cmd_sanity / cmd_judge                                      the sanity match that finds the scale of the equal wall for this build, and a match of a config against the old player at that scale (sets A and B, settled with min_pairs 60).
  Called by: the command line, whole_79_14_print.py, tests/test_plan79_14.py.

  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_14 profile [--count N] [--seconds S] [--pieces carried,...] | carrytest [--count N] [--seconds S] [--workers W]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import glob
import gzip
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path

from catspace.incubator.analyst.discovery import goal_reach as GR
from catspace.incubator.analyst.discovery import goal_table as GT
from catspace.incubator.analyst.discovery import whole_79_10_print as W10
from catspace.incubator.analyst.discovery import whole_play as WP
from catspace.io import paths

OUT = ("whole_79_14",)
LOG_ROOT = ("whole_79_13", "full", "logs")                 # the logged games the pairs are taken from (read only)
BASE_PIECES = list(W10.PIECES)                              # the pieces of 79.13, the carry on (the binding keeps the last decision of a slot)
GRAPH_CARRY = "carry_graph"                                 # PLAN 79.14 B: the switch of the carry that asks again only what the graph says was touched; off by default
CHANCE_ORDER = "chance_order"                               # PLAN 79.14 C: the named moves in the order of stake times the table's chance; off by default
GAIN_MEND = "gain_mend"                                     # PLAN 79.14 D: the gain mended; off by default
TABLE_WINDOW = 16                                           # the plies of the event of the table of C (the plan's window)
CONFIGS = {"b": BASE_PIECES + [GRAPH_CARRY], "bc": BASE_PIECES + [GRAPH_CARRY, CHANCE_ORDER], "bcg": BASE_PIECES + [GRAPH_CARRY, CHANCE_ORDER, GAIN_MEND]}
SEED_TAG = "full"                                           # 79.13's goal rows (read only)
_WORLD = None


def out_root(tag: str) -> Path:
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


def world() -> "W10.World":
    global _WORLD
    if _WORLD is None:
        _WORLD = W10.World()
    return _WORLD


def pairs_from_logs(dirs: list, count: int, seed: int = 7914, min_ply: int = 8) -> list:
    """`count` pairs (position of a decision, position of the same seat's next decision, label), spread over the games and the plies; only pairs two plies apart, from ply `min_ply` on (the opening is the same everywhere)."""
    rng = random.Random(seed)
    files = []
    for d in dirs:
        files += sorted(glob.glob(str(Path(paths.experiment(*LOG_ROOT)) / d / "*.jsonl.gz")))
    rng.shuffle(files)
    out: list = []
    for f in files:
        decs = []
        with gzip.open(f, "rt") as fh:
            for line in fh:
                r = json.loads(line)
                if r["kind"] == "decision":
                    decs.append(r)
        cand = [(a, b) for a, b in zip(decs, decs[1:]) if b["ply"] - a["ply"] == 2 and a["ply"] >= min_ply]
        if cand:
            a, b = rng.choice(cand)
            out.append((a["position"], b["position"], f"{Path(f).stem}:{a['ply']}"))
        if len(out) >= count:
            break
    return out


class Runner:
    """The wake as the harness of 79.11 calls it (the same arguments), with a slot for the carry. `kw` goes to the binding (chooser, goal_table)."""

    def __init__(self, w: "W10.World", pieces: list):
        self.w, self.pieces = w, list(pieces)

    def call(self, text: str, seconds: float, slot: "str | None", **kw) -> dict:
        w = self.w
        t0 = time.perf_counter()
        out = json.loads(w.model.wake_decide(text, [], seconds, 64, dict(w.worth), {}, w.goal, self.pieces, 0, 0, w.keys, WP.fuses(), w.chances, w.reasons, TM.LEDGER_SHARE, slot, **kw))
        out["wall"] = time.perf_counter() - t0
        return out

    def decide(self, text: str, seconds: float, slot: str, **kw) -> dict:
        return self.call(text, seconds, slot, **kw)

    def fresh(self, text: str, seconds: float, **kw) -> dict:
        return self.call(text, seconds, None, **kw)


def step_table(decisions: list) -> list:
    tot: dict = defaultdict(float)
    cnt: dict = defaultdict(int)
    for d in decisions:
        for s in d["steps"]:
            key = (s["step"], s["piece"] + (" [" + s["note"].split(":")[0] + "]" if s["step"] == 23 else ""))            # a round of the threat list is named by its note
            tot[key] += s["seconds"]
            cnt[key] += 1
    whole = sum(v for (st, _p), v in tot.items() if st == 0) or 1.0
    n = max(1, len(decisions))
    return [(st, p, v / n, 100 * v / whole, cnt[(st, p)]) for (st, p), v in sorted(tot.items())]


def note_numbers(d: dict, piece: str, word: str) -> int:
    """The count written before `word` in the note of the step named `piece` (0 when the step or the word is not there)."""
    import re
    for s in d["steps"]:
        if s["piece"] == piece:
            m = re.search(r"(\d+) " + re.escape(word), s["note"])
            if m:
                return int(m.group(1))
    return 0


def profile_job(job: tuple) -> dict:
    a, b, label, seconds, pieces = job
    r = Runner(world(), pieces)
    slot = f"profile:{label}:{os.getpid()}"
    first = r.decide(a, seconds, slot)
    second = r.decide(b, seconds, slot)
    return {"label": label, "first": first, "second": second}


def carry_job(job: tuple) -> dict:
    """The second position asked with the carry of the first and asked afresh; what the steps before the search made, side by side."""
    a, b, label, seconds, pieces = job
    r = Runner(world(), pieces)
    slot = f"carry:{label}:{os.getpid()}"
    r.decide(a, seconds, slot)
    carried = r.decide(b, seconds, slot)
    fresh = r.fresh(b, seconds)
    again = r.fresh(b, seconds)                                                 # the control: the same position asked afresh twice (what the clock alone changes)
    return {"label": label, "position": b, "carried": carried, "fresh": fresh, "again": again}


def pre_search(d: dict) -> dict:
    """What the steps before the search produced: the rows of the threat list (unit, class, verdict, net, line, the responses' states), the goals (target, unmet facts, fixes' moves, things in the way) and the names given to the search."""
    lg = d["log"]
    rows = sorted((r["list"], r["unit"], r["class"], r["verdict"], r["net"], tuple(r["line"]), tuple(tuple(x) for x in r["responses"])) for r in lg["rows"])
    goals = sorted((g["side"], g["target"], g["source"], g["depth"], tuple(g["unmet"]), tuple(g["stoppers"]), tuple(tuple(map(str, f)) for f in g["fixes"]), g["in_the_way"]) for g in lg["goals"])
    return {"rows": rows, "goals": goals, "named_mine": list(d["named_mine"]), "named_theirs": list(d["named_theirs"]), "their_threat": lg["their_threat"]}


def differences(c: dict, f: dict) -> list:
    """The differences between two `pre_search` readings, as words."""
    out = []
    for k in ("rows", "goals", "named_mine", "named_theirs", "their_threat"):
        if c[k] != f[k]:
            if k in ("rows", "goals"):
                only_c, only_f = [x for x in c[k] if x not in f[k]], [x for x in f[k] if x not in c[k]]
                out.append(f"{k}: {len(only_c)} only with the carry, {len(only_f)} only asked afresh: {only_c[:2]} | {only_f[:2]}")
            else:
                out.append(f"{k}: carried {c[k]} against fresh {f[k]}")
    return out


def cmd_profile(a, say) -> None:
    pairs = pairs_from_logs(a.logs.split(","), a.count)
    pieces = a.pieces.split(",") if a.pieces else BASE_PIECES
    say(f"OBS whole_79_14 profile: {len(pairs)} pairs of consecutive decisions of one seat from the logs {a.logs} of 79.13, the wake at {a.seconds} s with the pieces {','.join(pieces)}, one process, nothing else running")
    world()
    res = [profile_job((x, y, lab, a.seconds, pieces)) for x, y, lab in pairs]
    for which in ("first", "second"):
        ds = [r[which] for r in res]
        say(f"OBS whole_79_14 profile {which} decision (the second has the carry of the first): {len(ds)} positions; mean depth {sum(d['depth'] for d in ds) / len(ds):.2f}; mean wall {sum(d['wall'] for d in ds) / len(ds):.4f} s")
        for st, p, m, sh, c in step_table(ds):
            say(f"OBS whole_79_14 profile {which} step {st} {p}: mean {m:.4f} s, {sh:.1f} pct of the whole decision, in {c} decisions")
    ds = [r["second"] for r in res]
    held = sum(note_numbers(d, "carried open tactics", "rows held") for d in ds)
    again = sum(note_numbers(d, "carried open tactics", "read again") for d in ds)
    say(f"OBS whole_79_14 profile carry: rows held {held}, read again {again} (over {len(ds)} second decisions)")
    counts = {w: sum(note_numbers(d, p, w) for d in ds) for p, w in (("part of the frame read: the seeds on the board", "seeds found"), ("part of the frame read: the frames built", "frames built"), ("part of the frame read: the claims' questions", "frames asked"), ("part of the frame read: the claims' questions", "carried"))}
    carried_note = sum(note_numbers(d, "frames and the question read", "seeds") for d in ds)
    say(f"OBS whole_79_14 profile frame read counts over {len(ds)} second decisions: " + ", ".join(f"{w} {n}" for w, n in counts.items()) + f"; seeds carried (from the step's note) {carried_note}")


def cmd_carrytest(a, say, pool=None) -> None:
    pairs = pairs_from_logs(a.logs.split(","), a.count)
    pieces = (a.pieces.split(",") if a.pieces else BASE_PIECES)
    jobs = [(x, y, lab, a.seconds, pieces) for x, y, lab in pairs]
    say(f"OBS whole_79_14 carrytest: {len(jobs)} positions, each asked with the carry of the decision two plies before and afresh, at {a.seconds} s (the steps before the search have {0.2 * a.seconds:.2f} s); pieces {','.join(pieces)}; workers {a.workers}")
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(a.workers) as p:
        res = p.map(carry_job, jobs, chunksize=1)
    same = diff = ctl = 0
    carried_rows = carried_frames = 0
    for r in res:
        c, f = pre_search(r["carried"]), pre_search(r["fresh"])
        if differences(f, pre_search(r["again"])):
            ctl += 1
        ds = differences(c, f)
        carried_rows += sum(1 for x in r["carried"]["log"]["rows"] if x["carried"])
        carried_frames += note_numbers(r["carried"], "part of the frame read: the claims' questions", "carried")
        if ds:
            diff += 1
            say(f"OBS whole_79_14 carrytest DIFFERS {r['label']}: " + " || ".join(ds))
        else:
            same += 1
    say(f"OBS whole_79_14 carrytest: {same} of {len(res)} positions gave the same rows, goals and names carried and afresh; {diff} differed; rows marked carried {carried_rows}; frames carried {carried_frames}; CONTROL (the same position asked afresh twice): {ctl} of {len(res)} differed")


def table16(src_tag: str = SEED_TAG, tag: str = "full") -> dict:
    """The table of the chance of a goal at the window of 16 plies, counted on the FIT half of the goal rows of 79.13 (the held-out half is never counted), written for the wake."""
    d = GT.load(src_tag)
    j = GR.KS.index(TABLE_WINDOW)
    table = GT.fit_table(d, j, GR.fit_mask(d))
    (out_root(tag) / "table16.json").write_text(json.dumps(table))
    return table


def table_for(tag: str = "full") -> tuple:
    return GT.binding_form(json.loads((out_root(tag) / "table16.json").read_text()))


def config_specs(config: str, set_name: str, pairs: int, seconds: float, scratch: Path, log_dir: Path, scale_: float, table: "tuple | None", name: str) -> list:
    """The games of one match: the wake of `config` (the table of 16 plies for bc and bcg, mode Gain for bcg) against the old player at `scale_ x seconds`."""
    chooser = {"mode": "gain", "weights": {}, "seed": 0} if config == "bcg" else None
    return WP.match_specs("wake", "old", set_name, pairs, seconds, scratch, log_dir, clock_b=scale_ * seconds, label=name, chooser_a=chooser, table_a=table if config != "b" else None, pieces_a=CONFIGS[config])


def cmd_sanity(a, say) -> float:
    """The scale of the equal wall for this build: the wake of config b against the old player at one clock, 20 games (10 opening pairs of set A, no early stop): the wake's mean wall a move over the old player's."""
    root = out_root(a.tag)
    log_dir = root / "logs" / "sanity"
    specs = WP.match_specs("wake", "old", "A", 10, a.seconds, root / "scratch", log_dir, label="sanity", pieces_a=CONFIGS["b"])
    load0, busy = TM.wait_quiet(say)
    games = TM.run_games(specs, a.workers, root / "matches" / "sanity.jsonl", say, play=WP.play11, stop=None)
    s = WP.scale_of(games)
    (root / "scale.json").write_text(json.dumps({"scale": s, "seconds": a.seconds, "wake_wall": WP.mean_wall(games, 0), "old_wall": WP.mean_wall(games, 1), "games": len(games)}))
    say(f"OBS whole_79_14 sanity: ONE SCALE of equal wall for this build = {s:.4f} (the wake's mean wall a move {WP.mean_wall(games, 0):.4f} s over the old player's {WP.mean_wall(games, 1):.4f} s at the clock {a.seconds} s, {len(games)} games, load average at start {load0:.2f}); the old player is given {s * a.seconds:.4f} s for equal mean wall")
    return s


def depth_of(games: list, who: int) -> float:
    return sum(g["stats"][TM.seat_of(g, who)].get("depth_sum", 0) for g in games) / max(1, sum(g["stats"][TM.seat_of(g, who)]["decisions"] for g in games))


def cmd_judge(a, say) -> None:
    root = out_root(a.tag)
    scale_ = json.loads((root / "scale.json").read_text())["scale"]
    table = table_for(a.tag) if a.config != "b" else None
    name = f"{a.config}_v_old_{a.set}"
    log_dir = root / "logs" / name
    specs = config_specs(a.config, a.set, a.pairs, a.seconds, root / "scratch", log_dir, scale_, table, name)
    extra = ", ".join(CONFIGS[a.config][len(BASE_PIECES):]) + (f" (the table of the window of {TABLE_WINDOW} plies)" if a.config != "b" else "") + (", mode Gain" if a.config == "bcg" else "")
    say(f"OBS whole_79_14 {name}: the wake with the pieces of 79.13 and {extra} against the old player at {scale_ * a.seconds:.4f} s (the scale {scale_:.4f} of the sanity match); clock {a.seconds} s, ply cap {TM.PLY_CAP}, early stop min_pairs={WP.MIN_PAIRS}, workers {a.workers}")
    load0, busy = TM.wait_quiet(say)
    t0 = time.perf_counter()
    games = TM.run_games(specs, a.workers, root / "matches" / f"{name}.jsonl", say, play=WP.play11, stop=lambda g: TM.settled(g, min_pairs=WP.MIN_PAIRS))
    wall = time.perf_counter() - t0
    for x in WP.lines_of(name, "wake", "old", a.set, games, load0, os.getloadavg()[0], busy, wall, a.seconds, a.workers, {"old": scale_ * a.seconds}, log_dir):
        say(x.replace("OBS whole ", "OBS whole_79_14 ", 1))
    say(f"OBS whole_79_14 {name}: mean depth of the search a decision: wake {depth_of(games, 0):.2f}, old player {depth_of(games, 1):.2f} (the wake's mean wall {WP.mean_wall(games, 0):.4f} s, the old player's {WP.mean_wall(games, 1):.4f} s)")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("profile", "carrytest", "table16", "sanity", "judge"))
    ap.add_argument("--config", default="b", choices=tuple(CONFIGS))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--pieces", default="")
    ap.add_argument("--logs", default="gain_v_old_A,gain_v_old_B")
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)

    def say(line: str) -> None:
        TM.emit([line], root / "obs.txt")
    if a.stage == "table16":
        tb = table16(tag=a.tag)
        say(f"OBS whole_79_14 table16: the table of the window of {tb['k']} plies counted on the fit half of 79.13's goal rows: {tb['total'][0]} goals, {tb['total'][1]} reached, {sum(len(lv['cells']) for lv in tb['levels'])} cells of at least {tb['floor']} goals")
        return 0
    {"profile": cmd_profile, "carrytest": cmd_carrytest, "sanity": cmd_sanity, "judge": cmd_judge}[a.stage](a, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
