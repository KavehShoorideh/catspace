"""PLAN 78.68 (first run), SYSTEM_SPEC 9: THE LADDER, RUNGS 1 TO 3, AS A MEASUREMENT, and the list of open surprises with its groups. A sandbox: no player is changed, nothing is written to rules/learned or data/derived;
every output goes under artifacts/experiments/ladder_78_68/. Game-free: the game is named only through odds_measure (78.61), the engine and the readers that exist; the oracle only through ladder_oracle.py.

INTERFACE (what this module takes, what it returns, who calls it).
  Tools                 the readers of OUR OWN reasoning the ladder (ladder.py) is handed: the search of 78.67 (`solve_equals_batch` with the generation-2 record as judge), the forcing read and the best line (the line read of
                        78.72), the heads as tie-break (the arrival number of 78.49, the trees and linear heads of 78.65, read as 78.67 reads them), the features of a position (the frame read's classes, the line's events, the
                        phase by the count of units, whether our move is a taking or an answer to a threat of the threat list). `move_types(moves, move)` gives the last of these for any move.
  sample                the positions: games of the 78.67 matches (the control player's seat among them) and held-out human games (the 78.59 split's games that no fit read), by band of the count of units; writes positions.jsonl.
  label                 the oracle's accepted moves for every position (ladder_oracle.label); writes answers.jsonl, which nothing fitted may read.
  run                   the ladder on every position on a Pool; writes results.jsonl and surprise_list.jsonl (the wall-checked rows) and groups.json.
  report                the OBS lines. `answer_features` (the oracle's move types, for a diagnostic of the groups only) is a separate file.
Called by: the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.ladder_measure sample|label|run|report|answer-features  [--n N] [--workers W] [--tag T] [--record NAME] [--limit N]

PLAN 78.76 (SYSTEM_SPEC 5 and 9): THE RECORD IS A PARAMETER. `Tools(record=None)` takes a measured record of the kinds (threat_concrete.Record.load: a name made by ladder_records.py, a file, or None for the generation-2 record of
78.61, the default as before) and judges EVERY reading the ladder does by it: the search (`solve_equals_batch`'s worth), the forcing read and the best line (`Lines`). `run` and `report` with `--record NAME` read the positions and the
answers of the 78.68 run and write under artifacts/experiments/ladder_78_76/NAME/ (a short run, `--tag short --limit N`, under NAME/short/). Not read by the record, by design: the heads' own inputs (the 78.62 kind record the
heads were fitted on, the frame read's stake), which are part of what the heads are. ladder_compare.py prints the extra OBS lines of a record's run and the table of rung changes between records.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import ladder as LD
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.incubator.analyst.discovery import tactical_state as TS
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.io import paths

OUT = ("ladder_78_68",)
OUT_RECORDS = ("ladder_78_76",)                                       # PLAN 78.76: the ladder under each measured record of the kinds
BANDS = (("few", 0, 8), ("mid", 9, 16), ("many", 17, 99))          # phase by the count of units (a stated cut of this run)
QUOTA = {"few": 0.25, "mid": 0.35, "many": 0.40}                      # the share of a source's positions asked of each band
MIN_PLY = 8                                                           # no position inside the random opening of the match games or the first moves of a game
PER_GAME = 2                                                          # positions from one game at most (an even spread over games)
SEED = 6868
LIMITS = dict(LD.LIMITS)


def out_root(tag: str = "full", record: "str | None" = None) -> Path:
    """The run's folder: the 78.68 one, or (a named record) ladder_78_76/<record>/ for the full run and ladder_78_76/<record>/<tag>/ for any other."""
    if record is None:
        return Path(paths.experiment(*OUT, tag))
    return Path(paths.experiment(*OUT_RECORDS, record)) if tag == "full" else Path(paths.experiment(*OUT_RECORDS, record, tag))


def band_of(units: int) -> str:
    return next(n for n, lo, hi in BANDS if lo <= units <= hi)


# ------------------------------------------------------------------------------------------------------------------------------ the positions

def unit_counts(layer1, moves: list) -> list:
    """The count of units on the board after each ply (0 to len(moves)), from the layer-1 table."""
    names, raw = layer1.position_table([(None, list(moves))], 1)
    w = 2 + 2 * len(names)
    t = np.frombuffer(raw, dtype=np.int16).reshape(len(moves) + 1, w).astype(np.int32)
    return t[:, 2:].sum(axis=1).tolist()


def own_games() -> list:
    out = []
    for p in sorted(Path(paths.experiment("tiebreak_78_67", "full", "matches")).glob("*_v_control_*.jsonl")):
        for ln in p.read_text().splitlines():
            if ln.strip():
                g = json.loads(ln)
                out.append((f"{p.stem}:{g['gid']}", g["moves"]))
    return out


def human_games() -> list:
    """The held-out games of the 78.59 file: those the heads were not fitted on and the arrival number was not fitted on. The ordinal in the file stands for the game; no player id is kept."""
    out = []
    for i, ln in enumerate(Path(paths.experiment("estimator_78_59", "full", "games.jsonl")).read_text().splitlines()):
        if ln.strip():
            g = json.loads(ln)
            if not g["fit"] and not g.get("arrival"):
                out.append((f"h{i}", g["moves"]))
    return out


def playable(moves: list) -> bool:
    _f, mover, legal, outcome = CE.state(OM.VARIANT, None, moves)[:4]
    return mover is not None and outcome is None and len(legal) >= 2


def sample(n: int, say) -> list:
    import cs_engine  # noqa: F401
    from catspace.incubator.analyst.discovery import graph_layer1 as GL
    from catspace.incubator.analyst.discovery import human_heads_data as HD
    layer1 = GL.model(HD.GAME, HD.VARIANT)
    rng = random.Random(SEED)
    pools: dict = {}
    for source, games in (("own", own_games()), ("human", human_games())):
        games = list(games)
        rng.shuffle(games)
        for gid, moves in games[:6000]:
            cnt = unit_counts(layer1, moves)
            by = defaultdict(list)
            for k in range(MIN_PLY, len(moves)):
                by[band_of(cnt[k])].append(k)
            for b, ks in by.items():
                k = rng.choice(ks)
                pools.setdefault((source, b), []).append((gid, k, cnt[k], moves))
        say(f"candidates ({source}): " + ", ".join(f"{b} {len(pools.get((source, b), []))}" for b, _l, _h in BANDS))
    take: dict = {}
    taken_games: Counter = Counter()
    half = n // 2
    want = {(s, b): int(round(half * QUOTA[b])) for s in ("own", "human") for b, _l, _h in BANDS}
    # the thin band of the human sample is filled from our own games, never from an engine's continuation
    for s in ("human", "own"):
        for b, _l, _h in BANDS:
            pool = pools.get((s, b), [])
            rng.shuffle(pool)
            got = []
            for gid, k, u, moves in pool:
                if len(got) >= want[(s, b)]:
                    break
                if taken_games[(s, gid)] >= PER_GAME or not playable(moves[:k]):
                    continue
                taken_games[(s, gid)] += 1
                got.append((gid, k, u, moves))
            take[(s, b)] = got
            if s == "human" and len(got) < want[(s, b)]:
                want[("own", b)] += want[(s, b)] - len(got)
    rows = []
    for (s, b), got in take.items():
        for gid, k, u, moves in got:
            rows.append({"id": f"{s}:{gid}:{k}", "source": s, "units": u, "band": b, "moves": moves[:k]})
    say(f"the sample: {len(rows)} positions; " + "; ".join(f"{s} {b}: {len(take[(s, b)])} (asked {want[(s, b)]})" for s in ("own", "human") for b, _l, _h in BANDS))
    return rows


# ------------------------------------------------------------------------------------------------------------------------------ the oracle step

def _label_one(job: tuple) -> tuple:
    from catspace.incubator.analyst.discovery import ladder_oracle as LO
    pid, moves = job
    return pid, LO.label(moves)["accepted"]


def label_all(rows: list, workers: int, out_file: Path, say) -> None:
    done = {json.loads(x)["id"] for x in out_file.read_text().splitlines() if x.strip()} if out_file.exists() else set()
    todo = [(r["id"], r["moves"]) for r in rows if r["id"] not in done]
    t0 = time.perf_counter()
    with open(out_file, "a") as f, multiprocessing.get_context("spawn").Pool(workers) as pool:
        for i, (pid, acc) in enumerate(pool.imap_unordered(_label_one, todo, chunksize=1)):
            f.write(json.dumps({"id": pid, "accepted": acc}) + "\n")
            f.flush()
    say(f"labelled {len(todo)} positions (the oracle: depth {__import__('catspace.incubator.analyst.discovery.ladder_oracle', fromlist=['x']).ORACLE_DEPTH}, one thread a worker) in {time.perf_counter() - t0:.0f} s; "
        f"the file holds only the accepted moves")


# ------------------------------------------------------------------------------------------------------------------------------ the tools

class Tools:
    """One per process. Every reader is one of our own; the oracle is not here."""

    def __init__(self, record: "str | None" = None):
        self.record = TC.Record.load(record)                         # None: the generation-2 record, whose worths are those OM.players_of(WORTH_GEN) reads (a test confirms they are equal)
        self.worth = dict(self.record.worth)
        self.lines = TC.Lines(self.record, threads=1)
        self.wake_move = None
        self.last_move = None
        self.clocks = list(LIMITS["ladder"])

    def search(self, position: dict, seconds: float) -> dict:
        moves = list(position["moves"])
        row = PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, None, moves)], seconds, worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1,
                                                 ledger_share=TM.LEDGER_SHARE)[0])
        if seconds == LIMITS["wake_seconds"]:
            self.wake_move = row["move"]
        self.last_move = row["move"]
        return {"move": row["move"], "equals": row["equals"], "nodes": row["nodes"], "depth": row["depth"], "proved": row["proved"]}

    def forced(self, position: dict) -> dict:
        r = self.lines.ask([(None, list(position["moves"]))], [(0, 4, 0, 1, 0, 0, 0)])[0]
        if not r.get("ok"):
            return {"first": None, "kind": "gap"}
        first = (r.get("line") or [None])[0] if r.get("forced") in ("proved", "suspected") else None
        return {"first": first, "kind": r.get("forced")}

    def heads(self, position: dict, equals: list) -> dict:
        moves = list(position["moves"])
        legal = CE.state(OM.VARIANT, None, moves)[2]
        eng = self.wake_move
        nums = TM.arrival_numbers(moves, legal)
        out = {"arrival": TM.pick_best(eng, {m: nums[legal.index(m)] for m in equals if legal.index(m) in nums})}
        got = TM.reader().numbers(moves, legal, list(equals), TM.PLY_CAP)
        for key in ("trees", "linear"):
            out[key] = TM.pick_best(eng, {m: v[key] for m, v in got.items()})
        return out

    def line(self, position: dict, scale: float) -> dict:
        i = self.clocks.index(scale) if scale in self.clocks else 0
        r = self.lines.ask([(None, list(position["moves"]))], [(0, 8, 0, 1, 0, 0, 0)], TC.DEPTH + 2 * i, TC.FUSE * 4 ** i)[0]
        rd = (r.get("reads") or [{}])[0] if r.get("ok") else {}
        return {"names": rd.get("names", []), "events": rd.get("events", []), "end": rd.get("end"), "depth": rd.get("depth"), "cut": rd.get("cut"), "nodes": rd.get("nodes", 0)}

    def features(self, position: dict, deep: dict) -> dict:
        moves = list(position["moves"])
        st, res = TM.reader().frames_part([(None, moves)])
        classes, units = set(), 0
        if len(st["ply"]):
            for u in st["units"]:
                classes.add(f"{'us' if u[0] else 'them'}_{TS.CLASS_WORDS[int(u[2])]}" if int(u[2]) < len(TS.CLASS_WORDS) else f"{'us' if u[0] else 'them'}_other")
            units = len(res["rows"][0]["board"])
        ev = deep.get("events") or []
        toks = [f"wins_{k}" for mine, k in ev if not mine] + [f"loses_{k}" for mine, k in ev if mine]
        end = {0: "settled", 1: "game won", 2: "game lost"}.get(deep.get("end"), "unknown")
        if deep.get("cut"):
            end = "cut"
        return {"frame": sorted(classes), "phase": [band_of(units)] if units else [], "line": sorted(set(toks)) or ["quiet"], "line_end": [end],
                "our_move": move_types(moves, self.last_move)}


def move_types(moves: list, move: "str | None") -> list:
    """What kind of move: a TAKING (the count of units falls), an ANSWER to a threat of the frame read's threat list (the move is a response of a row against the side to move), else QUIET. The step of a goal (78.71)
    is not read here (a stated limit of this run)."""
    if move is None:
        return []
    global _THREADS
    if _THREADS is None:
        _THREADS = TL.ThreatReader(1)
    res = _THREADS.read([(None, list(moves))], [[len(moves)]])
    if not res["rows"]:
        return []
    raw = res["rows"][0]
    before = len(raw["board"])
    st2, res2 = TM.reader().frames_part([(None, list(moves) + [move])])
    types = []
    if len(st2["ply"]) and len(res2["rows"][0]["board"]) < before:
        types.append("taking")
    names = {k: res[k] for k in ("kinds", "roles", "places")}
    pl = names["places"]
    answers = {f"{pl[a[0]]}{pl[a[1]]}" for r in TL.threat_rows(raw, names)["mover"] for a in r["response_ix"]}
    if any(move.startswith(a) for a in answers):
        types.append("responds_to_threat")
    return types or ["quiet"]


# ------------------------------------------------------------------------------------------------------------------------------ the run

_TOOLS = None
_THREADS = None


def _run_one(job: tuple) -> dict:
    global _TOOLS
    pid, source, band, units, moves, acc = job[:6]
    if _TOOLS is None:
        _TOOLS = Tools(job[6] if len(job) > 6 else None)
    t0 = time.perf_counter()
    try:
        r = LD.ladder({"moves": moves}, acc, _TOOLS, LIMITS)
        err = None
    except Exception as e:                                           # a position the readers cannot take is reported, never dropped quietly
        import traceback
        r, err = None, repr(e)[:200] + " | " + " <- ".join(f"{f.name}:{f.lineno}" for f in reversed(traceback.extract_tb(e.__traceback__)))[:300]
    return {"id": pid, "source": source, "band": band, "units": units, "moves": moves, "result": r, "error": err, "wall": time.perf_counter() - t0}


def run_all(rows: list, answers: dict, workers: int, root: Path, say, record: "str | None" = None) -> list:
    out_file = root / "results.jsonl"
    done = {}
    if out_file.exists():
        for ln in out_file.read_text().splitlines():
            if ln.strip():
                d = json.loads(ln)
                done[d["id"]] = d
    jobs = [(r["id"], r["source"], r["band"], r["units"], r["moves"], answers[r["id"]], record) for r in rows if r["id"] not in done]
    t0 = time.perf_counter()
    n0 = len(done)
    with open(out_file, "a") as f, multiprocessing.get_context("spawn").Pool(workers) as pool:
        for d in pool.imap_unordered(_run_one, jobs, chunksize=1):
            done[d["id"]] = d
            f.write(json.dumps(d) + "\n")
            f.flush()
            if len(done) % 10 == 0:
                el = time.perf_counter() - t0
                say(f"progress: {len(done)} of {len(rows)} positions, {el:.0f} s, about {el / max(1, len(done) - n0) * (len(rows) - len(done)):.0f} s left")
    return [done[r["id"]] for r in rows if r["id"] in done]


def write_list(results: list, root: Path) -> list:
    rows = [LD.surprise_row(d["id"], d["source"], d["moves"], d["result"]) for d in results if d["result"] and d["result"]["rung"] in (0, 3)]
    LD.write_surprise_list(root / "surprise_list.jsonl", rows)
    g = LD.groups(rows)
    (root / "groups.json").write_text(json.dumps(g))
    return g


# ------------------------------------------------------------------------------------------------------------------------------ the report

def report(results: list, groups: list, root: Path, say) -> None:
    ok = [d for d in results if d["result"]]
    err = [d for d in results if not d["result"]]
    say(f"positions {len(results)}; read {len(ok)}; the readers failed on {len(err)}" + (f" ({Counter(d['error'][:60] for d in err).most_common(2)})" if err else ""))
    for key in ("source", "band"):
        for v in sorted({d[key] for d in ok}):
            sub = [d for d in ok if d[key] == v]
            c = Counter(d["result"]["rung"] for d in sub)
            say(f"RUNGS by {key} {v}: n {len(sub)}: rung 1 {c[1]}, rung 2 {c[2]}, rung 3 {c[3]}, rung 0 {c[0]}")
    c = Counter(d["result"]["rung"] for d in ok)
    say(f"RUNGS all: n {len(ok)}: rung 1 {c[1]} ({100 * c[1] / max(1, len(ok)):.1f} pct), rung 2 {c[2]} ({100 * c[2] / max(1, len(ok)):.1f} pct), rung 3 {c[3]} ({100 * c[3] / max(1, len(ok)):.1f} pct), rung 0 {c[0]} ({100 * c[0] / max(1, len(ok)):.1f} pct)")
    for src in sorted({d["source"] for d in ok}):
        for b, _l, _h in BANDS:
            sub = [d for d in ok if d["source"] == src and d["band"] == b]
            if sub:
                c = Counter(d["result"]["rung"] for d in sub)
                say(f"RUNGS {src} x {b}: n {len(sub)}: 1 {c[1]}, 2 {c[2]}, 3 {c[3]}, 0 {c[0]}")
    for r in (1, 2, 3):
        used = [d["result"]["cost"][str(r)] if str(r) in d["result"]["cost"] else d["result"]["cost"][r] for d in ok]
        used = [u for u in used if u["queries"]]
        if used:
            say(f"COST rung {r}: {len(used)} positions reached it; per position mean {np.mean([u['seconds'] for u in used]):.2f} s, {np.mean([u['nodes'] for u in used]):.0f} nodes, {np.mean([u['queries'] for u in used]):.1f} queries; total {sum(u['seconds'] for u in used):.0f} s")
    el = [d for d in ok if d["result"]["tie"]["eligible"]]
    if el:
        say(f"RUNG 2 ties: {len(el)} positions where the wake's equal set held the accepted move among two or more equal moves and not all accepted; a random pick would hit {np.mean([d['result']['tie']['random_hit'] for d in el]):.3f} on average")
        for h in ("arrival", "trees", "linear"):
            k = sum(d["result"]["tie"]["hits"].get(h, False) for d in el)
            say(f"RUNG 2 head {h}: hit {k} of {len(el)} ({100 * k / len(el):.1f} pct)")
        k = sum(bool(d["result"]["tie"]["control_hit"]) for d in el)
        say(f"RUNG 2 the search's own first move: hit {k} of {len(el)} ({100 * k / len(el):.1f} pct)")
    r3 = [d for d in ok if d["result"]["rung"] == 3]
    if r3:
        say(f"RUNG 3 found at the clock: {dict(sorted(Counter(d['result']['found_at'] for d in r3).items()))}; ladder {LIMITS['ladder']} s")
    say(f"SURPRISE LIST: {sum(1 for d in ok if d['result']['rung'] in (0, 3))} rows (rungs 3 and 0); groups {len(groups)}")
    by_id = {d["id"]: d for d in ok}
    for g in groups[:12]:
        ex = ", ".join(g["ids"][:3])
        say(f"GROUP size {g['size']}: key {g['key']}; examples {ex}")


def record_line(rec: "TC.Record", say) -> None:
    """The one OBS line that names the record every reading of the run is judged by."""
    say(f"RECORD {rec.source}: worths {dict(sorted(rec.worth.items(), key=lambda kv: kv[1]))}; half widths {({k: round(v, 1) for k, v in sorted(rec.half.items())})}; the search, the forcing read and the best line all read this record")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sample", "label", "run", "report", "answer-features"))
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--record", default=None, help="PLAN 78.76: a measured record of the kinds (a name of ladder_records, a file, or gen2); outputs go to ladder_78_76/<record>/")
    ap.add_argument("--limit", type=int, default=0, help="PLAN 78.76: the first N positions only (a short run)")
    a = ap.parse_args(argv)
    root = out_root(a.tag, a.record)
    root.mkdir(parents=True, exist_ok=True)
    src = out_root("full")                                           # the positions and the answers are those of the 78.68 run, whatever the record

    def say(m: str) -> None:
        line = f"OBS ladder {a.stage}: {m}"
        print(line, flush=True)
        with open(root / "obs.txt", "a") as f:
            f.write(line + "\n")
    pos_file, ans_file = src / "positions.jsonl", src / "answers.jsonl"
    if a.record is not None and a.stage in ("sample", "label"):
        raise SystemExit("sample and label are the 78.68 run's: no --record")
    if a.stage == "sample":
        rows = sample(a.n, say)
        pos_file.write_text("".join(json.dumps(r) + "\n" for r in rows))
        return 0
    rows = [json.loads(x) for x in pos_file.read_text().splitlines() if x.strip()]
    if a.limit:
        rows = rows[:a.limit]
    if a.stage == "label":
        label_all(rows, a.workers, ans_file, say)
        return 0
    answers = {json.loads(x)["id"]: json.loads(x)["accepted"] for x in ans_file.read_text().splitlines() if x.strip()}
    if a.stage == "run":
        say(f"load average {os.getloadavg()[0]:.2f}; clocks: the wake's {LIMITS['wake_seconds']} s, then {LIMITS['ladder']}; {a.workers} workers")
        if a.record is not None:
            record_line(TC.Record.load(a.record), say)
        res = run_all(rows, answers, a.workers, root, say, a.record)
        report(res, write_list(res, root), root, say)
        if a.record is not None:
            from catspace.incubator.analyst.discovery import ladder_compare as LC
            LC.extra_report(res, say)
    elif a.stage == "report":
        res = [json.loads(x) for x in (root / "results.jsonl").read_text().splitlines() if x.strip()]
        if a.record is not None:
            record_line(TC.Record.load(a.record), say)
        report(res, write_list(res, root), root, say)
        if a.record is not None:
            from catspace.incubator.analyst.discovery import ladder_compare as LC
            LC.extra_report(res, say)
    else:
        lst = [json.loads(x) for x in (root / "surprise_list.jsonl").read_text().splitlines() if x.strip()]
        out = {}
        for r in lst:
            for m in answers[r["id"]]:
                out.setdefault(r["id"], {})[m] = move_types(r["moves"], m)
        (root / "answer_features.json").write_text(json.dumps(out))
        say(f"the oracle's moves of {len(out)} surprises read for their types (a diagnostic of the groups; nothing fitted reads this file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
