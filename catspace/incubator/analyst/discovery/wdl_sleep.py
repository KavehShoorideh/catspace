"""PLAN 78.27 THE SLEEP CYCLE OF THE WIN / DRAW / LOSS ESTIMATE. The wake decides with what the last sleep left it (the store, the results table, its own search and playouts) and logs
where it lacked the data or the lemma; the sleep works exactly there: it reads those gaps, reads the gap positions deeper with the engine's own playouts, settles unfinished games,
and files everything in the results table the next wake pools with its playouts. An outside engine may be a JUDGE of the readings (it chooses a reading setting and measures the
table before and after); NO number of it is written into the table or into anything the wake reads. Human games are not read.

Game-free: the game and variant are parameters. The one place a chess library appears is `UciWdlJudge` (training only).

  gaps(log, ...)             the wake's gaps, per decision and as a worklist
  WdlJudge / UciWdlJudge     a judge of readings (training only): wdl(variant, moves, start) -> (win, draw, loss) for the side to move
  ReadSetting, default_menu  the engine's own estimate settings the sleep may read with
  choose_reading             which setting tracks the judge best (the judge's numbers stay inside)
  settle, deepen             unfinished games to soft results; gap positions to deep rows
  calibrate                  the wake's own estimate against the judge, with a table or none
  sleep_wdl                  the whole sleep: gaps -> setting -> settle -> deepen -> table -> before / after

  .venv/bin/python -m catspace.incubator.analyst.discovery.wdl_sleep chess --log LOG [--log ...] [--prior-table P] [--judge-engine UCI] [--judge-depth 12] [--judge-cache FILE]
         [--store SNAPSHOT] [--out P] [--deep-seconds S] [--max-deep N] [--sample N]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import random
import sys
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import sleep as SL
from catspace.incubator.analyst.discovery import wdl_table as WT
from catspace.io import paths

ROLES = WT.ROLES
DEEP_EXACT_MASS = 200.0        # the table's own cap (rust TABLE_CAP): an exact deep row then weighs about as much as the wake's own playouts when pooled
COARSE_VOTE = 1.0              # one position's vote on a coarser key: a coarse key stands for many positions, so one reading must not outweigh the games under it
WAKE_SECONDS = 0.03            # the wake's own estimate clock (the controller's share of a decision)
WAKE_CUT = 8                   # the wake's own cut in plies before the leaf valuation
GAP_KINDS = ("no_table", "coarse_table", "unknown", "thin_pick", "no_lemma", "outcome_surprise")


# ------------------------------------------------------------------------------------------------------------------------------------ the gaps

def _segments(facts: list) -> list:
    """The facts of a row cut at each `wdl` fact: [estimate's facts, prefer's facts]."""
    segs: list = []
    for f in facts or []:
        if f.get("relation") == "wdl":
            segs.append([])
        if segs:
            segs[-1].append(f)
    return segs


def reading_of(row: dict) -> "dict | None":
    """The estimate a wake `position` row records (structured fields, not prose): {mode, win, draw, loss, mass, score, table ('none' | 'exact' | 'coarse'), table_mass, lemma,
    picked, margin} or None when the row has no wdl step. `picked`: the probabilities picked among moves; `margin`: best less second-best score of that pick (None if not picked)."""
    segs = _segments(row.get("facts"))
    if not segs:
        return None
    first = segs[0]
    w = next(f for f in first if f["relation"] == "wdl")["args"]
    sharp = next((f for f in first if f["relation"] == "sharpness"), None)
    hit = next((f for f in first if f["relation"] == "table_hit"), None)
    win, draw, loss = float(w[0]), float(w[1]), float(w[2])
    picked = any(s.get("operator") == "wdl" and str(s.get("goal", "")).startswith("prefer") for s in row.get("steps", []))
    margin = None
    if picked and len(segs) > 1:
        sc = sorted((float(f["args"][1]) + float(f["args"][2]) / 2 for f in segs[-1] if f["relation"] == "move_wdl"), reverse=True)
        margin = sc[0] - sc[1] if len(sc) > 1 else None
    return {"mode": sharp["args"][1] if sharp else "unknown", "win": win, "draw": draw, "loss": loss, "mass": float(w[3]), "score": win + draw / 2,
            "table": "none" if hit is None else ("exact" if hit["args"][0].startswith("exact:") else "coarse"),
            "table_mass": 0.0 if hit is None else sum(float(x) for x in hit["args"][1:4]),
            "lemma": any(f.get("relation") == "fired" for f in row.get("facts") or []), "picked": picked, "margin": margin}


def _as_list(x) -> list:
    return [x] if isinstance(x, (str, os.PathLike)) else list(x)


def decision_readings(log_paths) -> list:
    """Every decision of wake logs that has a wdl estimate: dicts {log, run, game, ply, start, moves, role, finished, outcome, reading}. Own data only."""
    out = []
    for n, path in enumerate(_as_list(log_paths)):
        decisions, games = SL.read_log(str(path))
        for (run, g, ply), d in sorted(decisions.items()):
            rd = reading_of(d["row"])
            game = games.get((run, g))
            if rd is None or game is None:
                continue
            done = not game.get("unfinished") and game.get("outcome") is not None
            out.append({"log": n, "run": run, "game": g, "ply": ply, "start": game.get("start"), "moves": list(game["moves"][:ply]), "role": d["role"],
                        "finished": done, "outcome": game["outcome"] if done else None, "reading": rd})
    return out


def gaps(log_path, *, pick_margin: float = 0.02, surprise: float = 0.5) -> dict:
    """The gaps of a wake log (or a list of logs) played with wdl on -> {decisions, counts {kind: n}, modes, table {none, exact, coarse}, mass_median, with_lemma, picked, worklist}.
    A worklist entry: {log, run, game, ply, start, moves (the prefix), kinds}: one per position with at least one gap."""
    counts = {k: 0 for k in GAP_KINDS}
    modes = {"sharp": 0, "quiet": 0, "unknown": 0}
    table = {"none": 0, "exact": 0, "coarse": 0}
    work, masses, lemma, picked, n = [], [], 0, 0, 0
    for d in decision_readings(log_path):
        r = d["reading"]
        n += 1
        modes[r["mode"]] = modes.get(r["mode"], 0) + 1
        table[r["table"]] += 1
        masses.append(r["mass"])
        lemma += r["lemma"]
        picked += r["picked"]
        kinds = []
        if r["table"] == "none":
            kinds.append("no_table")
        if r["table"] == "coarse":
            kinds.append("coarse_table")
        if r["mode"] == "unknown":
            kinds.append("unknown")
        if r["picked"] and r["margin"] is not None and r["margin"] < pick_margin:
            kinds.append("thin_pick")
        if not r["lemma"]:
            kinds.append("no_lemma")
        if d["finished"] and abs(r["score"] - d["outcome"][d["role"]] / 100.0) >= surprise:
            kinds.append("outcome_surprise")
        for k in kinds:
            counts[k] += 1
        if kinds:
            work.append({"log": d["log"], "run": d["run"], "game": d["game"], "ply": d["ply"], "start": d["start"], "moves": d["moves"], "kinds": kinds})
    masses.sort()
    return {"decisions": n, "counts": counts, "modes": modes, "table": table, "mass_median": masses[len(masses) // 2] if masses else None, "with_lemma": lemma, "picked": picked, "worklist": work}


# ------------------------------------------------------------------------------------------------------------------------------------ the judge of readings (training only)

MATE_CP = 10000          # the centipawn value a forced mate is read as (plan 78.37)


class WdlJudge:
    """The interface of a judge of readings: TRAINING ONLY. `wdl(variant, moves, start=None) -> (win, draw, loss)` probabilities for the side to move."""
    training_only = True

    def wdl(self, variant: str, moves: list, start: "str | None" = None) -> tuple:
        raise NotImplementedError

    def close(self) -> None:
        pass


class UciWdlJudge(WdlJudge):
    """THE ONE PLACE A CHESS LIBRARY APPEARS: a UCI engine that reports win / draw / loss (UCI_ShowWDL) through python-chess, answers remembered on disk by position so re-runs repeat
    (the pattern of judges.Cached). Always `close()` it (in `finally`): python-chess's engine thread can keep the process alive."""
    source = "uci-wdl"

    def __init__(self, path: str, depth: int = 12, cache: "str | None" = None, options: "dict | None" = None):
        self.path, self.depth, self.options = path, depth, dict(options or {})
        self.cache_path = Path(cache) if cache else None
        self.memory = json.loads(self.cache_path.read_text()) if self.cache_path and self.cache_path.exists() else {}
        self.asked = self.hits = self.unsaved = 0
        self._engine = None

    def _board(self, variant: str, moves: list, start):
        import chess
        import chess.variant
        board = (chess.variant.find_variant(variant)(start) if start else chess.variant.find_variant(variant)()) if variant != "chess" else (chess.Board(start) if start else chess.Board())
        for m in moves:
            board.push_uci(m)
        return board

    def wdl(self, variant: str, moves: list, start: "str | None" = None) -> tuple:
        key = json.dumps([variant, start, list(moves), self.depth])
        self.asked += 1
        if key in self.memory:
            self.hits += 1
            return tuple(self.memory[key])
        import chess.engine
        board = self._board(variant, list(moves), start)
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self._engine.configure({"UCI_ShowWDL": True, **self.options})
        info = self._engine.analyse(board, chess.engine.Limit(depth=self.depth))
        w = info.get("wdl")
        if w is None:
            raise RuntimeError(f"{self.path} reported no win / draw / loss (UCI_ShowWDL)")
        p = w.pov(board.turn)
        tot = float(p.wins + p.draws + p.losses) or 1.0
        out = (p.wins / tot, p.draws / tot, p.losses / tot)
        self.memory[key] = list(out)
        self.unsaved += 1
        return out

    def read(self, variant: str, start: "str | None", moves: list) -> dict:
        """PLAN 78.37 the surprise meter's reading: {"cp": int or None, "wdl": (win, draw, loss)} for the side to move, at the judge's depth. A forced mate is read as a
        large centipawn value (MATE_CP) with the sign of the side that mates. Remembered in the same file as `wdl`, under its own key."""
        key = json.dumps(["read", variant, start, list(moves), self.depth])
        self.asked += 1
        if key in self.memory:
            self.hits += 1
            e = self.memory[key]
            return {"cp": e["cp"], "wdl": tuple(e["wdl"])}
        import chess.engine
        board = self._board(variant, list(moves), start)
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self.path)
            self._engine.configure({"UCI_ShowWDL": True, **self.options})
        info = self._engine.analyse(board, chess.engine.Limit(depth=self.depth))
        w = info.get("wdl")
        if w is None:
            raise RuntimeError(f"{self.path} reported no win / draw / loss (UCI_ShowWDL)")
        p = w.pov(board.turn)
        tot = float(p.wins + p.draws + p.losses) or 1.0
        sc = info.get("score")
        cp = sc.pov(board.turn).score(mate_score=MATE_CP) if sc is not None else None
        out = {"cp": cp, "wdl": (p.wins / tot, p.draws / tot, p.losses / tot)}
        self.memory[key] = {"cp": cp, "wdl": list(out["wdl"])}
        self.unsaved += 1
        return out

    def save(self) -> None:
        if self.cache_path is None or not self.unsaved:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache_path.with_suffix(self.cache_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.memory))
        os.replace(tmp, self.cache_path)
        self.unsaved = 0

    def close(self) -> None:
        try:
            self.save()
        finally:
            eng, self._engine = self._engine, None
            if eng is not None:
                try:
                    eng.quit()
                except Exception:
                    eng.close()


# ------------------------------------------------------------------------------------------------------------------------------------ reading settings

@dataclasses.dataclass(frozen=True)
class ReadSetting:
    """One way the engine's own estimate is read: its clock, the cut in plies before the leaf valuation, the playout policy."""
    name: str
    seconds: float
    max_playout: int = WAKE_CUT
    rollout: str = "careful"


def default_menu(deep_seconds: float = 1.0) -> list:
    """The wake's own reading, the same on a long clock, a far cut, and playouts to the end."""
    return [ReadSetting("wake", WAKE_SECONDS, WAKE_CUT), ReadSetting("long", deep_seconds, WAKE_CUT), ReadSetting("far", deep_seconds, 40), ReadSetting("end", deep_seconds, 400)]


def read(setting: ReadSetting, variant: str, start, moves: list, *, judge_kw: "dict | None" = None, seed: int = 1, table: "str | None" = None) -> dict:
    """The engine's own estimate at the position after `moves` with `setting` (CE.wdl), optionally pooled with a table file."""
    return CE.wdl(variant, start, list(moves), seconds=setting.seconds, max_playout=setting.max_playout, rollout=setting.rollout, seed=seed, table=table, **(judge_kw or {}))


def expected(w: float, d: float) -> float:
    """The expected score of a (win, draw, loss) reading."""
    return w + d / 2


def pearson(a: list, b: list) -> "float | None":
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va, vb = sum((x - ma) ** 2 for x in a), sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5


def choose_reading(positions: list, judge, menu: list, *, variant: str, judge_kw: "dict | None" = None, sample: int = 60, seed: int = 1) -> dict:
    """Which reading setting to sleep with. `positions`: (start, moves). With a judge: a seeded sample is read with every menu entry and by the judge; each entry is scored by the mean
    absolute error of its expected score (win + draw / 2) against the judge's (and the correlation); the non-wake entry with the smallest error is chosen. The judge's numbers do not
    leave this function except as these aggregates. Without a judge: `long`. -> {setting, reason, scores {name: {n, mae, corr, seconds}}, sample}."""
    by = {s.name: s for s in menu}
    if judge is None:
        return {"setting": by.get("long", menu[-1]), "reason": "no judge of the readings was given", "scores": {}, "sample": 0}
    pos = list(positions)
    pos = pos if len(pos) <= sample else random.Random(seed).sample(pos, sample)
    mine: dict = {s.name: [] for s in menu}
    theirs: dict = {s.name: [] for s in menu}
    secs: dict = {s.name: 0.0 for s in menu}
    for start, moves in pos:
        j = expected(*judge.wdl(variant, list(moves), start)[:2])
        for s in menu:
            r = read(s, variant, start, moves, judge_kw=judge_kw, seed=seed)
            mine[s.name].append(expected(r["win"], r["draw"]))
            theirs[s.name].append(j)
            secs[s.name] += r.get("seconds") or 0.0
    scores = {}
    for s in menu:
        a, b = mine[s.name], theirs[s.name]
        n = len(a)
        scores[s.name] = {"n": n, "mae": sum(abs(x - y) for x, y in zip(a, b)) / n if n else None, "corr": pearson(a, b), "seconds": secs[s.name] / n if n else None}
    deep = [s for s in menu if s.name != "wake" and scores[s.name]["mae"] is not None]
    if not deep:
        return {"setting": by.get("long", menu[-1]), "reason": "the judge scored no position", "scores": scores, "sample": len(pos)}
    best = min(deep, key=lambda s: scores[s.name]["mae"])
    wake = scores.get("wake", {}).get("mae")
    reason = (f"{best.name} has the smallest mean absolute error of the expected score against the judge ({scores[best.name]['mae']:.3f}) among "
              f"{', '.join(s.name for s in deep)} on {len(pos)} positions" + (f"; the wake's own reading has {wake:.3f}" if wake is not None else ""))
    return {"setting": best, "reason": reason, "scores": scores, "sample": len(pos)}


# ------------------------------------------------------------------------------------------------------------------------------------ settle and deepen

def settle(unfinished_games: list, setting: ReadSetting, *, variant: str, judge_kw: "dict | None" = None, seed: int = 1, table: "str | None" = None, say=print) -> list:
    """Read the LAST position of each unfinished game (game rows with `start` and `moves`) with `setting`; the game then counts with that reading as its result.
    -> [{start, moves, mover, wdl [win, draw, loss] of that mover, settled True, setting}]; a game whose last position cannot be read (over, no legal move) is skipped."""
    out = []
    for g in unfinished_games:
        moves = list(g["moves"])
        try:
            mover = CE.state(variant, g.get("start"), moves)[1]
            r = read(setting, variant, g.get("start"), moves, judge_kw=judge_kw, seed=seed, table=table)
        except ValueError:
            continue
        out.append({"start": g.get("start"), "moves": moves, "mover": mover, "wdl": [r["win"], r["draw"], r["loss"]], "settled": True, "setting": setting.name})
    say(f"OBS wdl_sleep settle: {len(out)} of {len(unfinished_games)} unfinished games read with {setting.name} ({setting.seconds} s, cut {setting.max_playout})")
    return out


def deepen(worklist: list, setting: ReadSetting, *, variant: str, max_positions: int, judge_kw: "dict | None" = None, seed: int = 1, table: "str | None" = None, say=print) -> dict:
    """Read gap positions with `setting`, at most `max_positions`, spread over the games (`sleep.spread`; a position of more gap kinds first within its game, each exact position once).
    Rows: the position's exact key gets the reading's probabilities at DEEP_EXACT_MASS, each coarser key at COARSE_VOTE. -> {rows {key: [w, d, l]}, touched [worklist entries], read, mass}."""
    ordered = sorted(worklist, key=lambda e: -len(e["kinds"]))
    pick, seen = [], set()
    for e in SL.spread(ordered, lambda e: (e.get("log", 0), e["run"], e["game"])):
        if len(pick) >= max_positions:
            break
        keys = CE.wdl_keys(variant, e["start"], list(e["moves"]))
        if keys[0] in seen:
            continue
        seen.add(keys[0])
        pick.append((e, keys))
    rows: dict = {}
    touched = []
    for e, keys in pick:
        try:
            r = read(setting, variant, e["start"], e["moves"], judge_kw=judge_kw, seed=seed, table=table)
        except ValueError:
            continue
        p = [r["win"], r["draw"], r["loss"]]
        for i, k in enumerate(keys):
            m = DEEP_EXACT_MASS if i == 0 else COARSE_VOTE
            row = rows.setdefault(k, [0.0, 0.0, 0.0])
            for j in range(3):
                row[j] += p[j] * m
        touched.append(e)
    mass = sum(sum(v) for v in rows.values())
    say(f"OBS wdl_sleep deepen: {len(touched)} positions read with {setting.name} of {len(worklist)} gap positions, {len(rows)} rows touched, mass {mass:.1f}")
    return {"rows": rows, "touched": touched, "read": len(touched), "mass": mass}


# ------------------------------------------------------------------------------------------------------------------------------------ calibration

def calibrate(positions: list, judge, *, variant: str, table: "str | None" = None, seconds: float = WAKE_SECONDS, judge_kw: "dict | None" = None, seed: int = 1,
              max_playout: int = WAKE_CUT) -> dict:
    """The wake's own estimate (its clock, pooled with `table` or none) against the judge on `positions` ((start, moves)) ->
    {positions, mae, corr, calls {engine_sharp_judge_sharp, ...}, table_hits, unknown}. A judge call is sharp when its win + loss is at least 0.5. Used before and after on the SAME positions."""
    setting = ReadSetting("wake", seconds, max_playout)
    a, b = [], []
    calls = {"engine_quiet_judge_quiet": 0, "engine_quiet_judge_sharp": 0, "engine_sharp_judge_quiet": 0, "engine_sharp_judge_sharp": 0}
    hits = unknown = 0
    for start, moves in positions:
        r = read(setting, variant, start, moves, judge_kw=judge_kw, seed=seed, table=table)
        w, d, l = judge.wdl(variant, list(moves), start)
        a.append(expected(r["win"], r["draw"]))
        b.append(expected(w, d))
        calls[f"engine_{'sharp' if r['sharp'] else 'quiet'}_judge_{'sharp' if w + l >= 0.5 else 'quiet'}"] += 1
        hits += r.get("table_key") is not None
        unknown += r.get("sharp") is None
    n = len(a)
    return {"positions": n, "mae": sum(abs(x - y) for x, y in zip(a, b)) / n if n else None, "corr": pearson(a, b), "calls": calls, "table_hits": hits, "unknown": unknown}


# ------------------------------------------------------------------------------------------------------------------------------------ the sleep

def default_out(game: str) -> str:
    return paths.derived(f"wdl_table_{game}_sleep.json")


def judge_kw_of_snapshot(path: str) -> dict:
    """The valuation a snapshot file plays by, as CE.wdl takes it ({} when it carries none)."""
    j = json.load(open(path)).get("judge")
    return {} if not j else {"worth": {k: int(v) for k, v in j["worth"].items()}, "ours_first": bool(j["ours_first"]), "valuation": j["name"]}


def unfinished_of(log_paths) -> list:
    """The unfinished game rows (start, moves) of wake logs."""
    out = []
    for p in _as_list(log_paths):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                if r.get("kind") == "game" and (r.get("unfinished") or r.get("outcome") is None):
                    out.append({"start": r.get("start"), "moves": r.get("moves", [])})
    return out


def sleep_wdl(game: str, logs, *, variant: "str | None" = None, prior_table: "str | None" = None, judge=None, judge_kw: "dict | None" = None, out: "str | None" = None,
              deep_seconds: float = 1.0, max_deep: int = 200, sample: int = 60, seed: int = 1, max_settle: int = 50, setting: "ReadSetting | None" = None, say=print) -> dict:
    """The estimate's sleep over wake logs: gaps, the reading setting (chosen against the judge, or `setting` given, or `long`), settle, deepen, the table written to `out` (the prior
    one carried forward), then the wake's own estimate before (prior table) and after (new table) measured against the judge on positions the deep readings did NOT touch (held out)
    and on the ones they did. The judge is used for the choice and the measure only: it never enters the table. -> the report (with `table` the path)."""
    variant = variant or game
    logs = _as_list(logs)
    out = str(out or default_out(game))
    keep = None
    if prior_table and os.path.exists(out) and os.path.exists(prior_table) and os.path.samefile(out, prior_table):
        keep = f"{out}.prior"                                   # (the new table overwrites the prior one in place: the `before` measure reads this copy)
        Path(keep).write_bytes(Path(prior_table).read_bytes())
        prior_table = keep
    try:
        return _sleep_wdl(game, logs, variant, prior_table, judge, judge_kw, out, deep_seconds, max_deep, sample, seed, max_settle, setting, say)
    finally:
        if keep and os.path.exists(keep):
            os.remove(keep)


def _sleep_wdl(game, logs, variant, prior_table, judge, judge_kw, out, deep_seconds, max_deep, sample, seed, max_settle, setting, say) -> dict:
    gp = gaps(logs)
    say(f"OBS wdl_sleep gaps {game}: decisions {gp['decisions']}, " + ", ".join(f"{k} {v}" for k, v in gp["counts"].items()) + f", worklist {len(gp['worklist'])}")
    allpos = [(d["start"], d["moves"]) for d in decision_readings(logs)]
    menu = default_menu(deep_seconds)
    if setting is not None:
        choice = {"setting": setting, "reason": "given", "scores": {}, "sample": 0}
    else:
        choice = choose_reading(allpos, judge, menu, variant=variant, judge_kw=judge_kw, sample=sample, seed=seed)
    st = choice["setting"]
    say(f"OBS wdl_sleep reading {game}: {st.name} ({st.seconds} s, cut {st.max_playout}), reason: {choice['reason']}")
    for name, sc in choice["scores"].items():
        say(f"OBS wdl_sleep reading {game}: {name} positions {sc['n']}, mean abs error {sc['mae']}, correlation {sc['corr']}, seconds a reading {sc['seconds']}")
    unfinished = unfinished_of(logs)[:max_settle]
    settled = settle(unfinished, st, variant=variant, judge_kw=judge_kw, seed=seed, say=say)
    deep = deepen(gp["worklist"], st, variant=variant, max_positions=max_deep, judge_kw=judge_kw, seed=seed, say=say)
    finished = WT.records_from_wake_logs(logs)
    table = WT.build(finished, variant, settled=settled, deep_rows=deep["rows"], prior_table=prior_table)
    WT.write(table, out)
    ts = table["stats"]
    say(f"OBS wdl_sleep table {game}: games {ts['games']}, finished positions {ts['positions']}, settled games {ts['settled']['games']} ({ts['settled']['positions']} positions), deep rows {ts['deep']['rows']} "
        f"(mass {ts['deep']['mass']:.1f}), rows carried {ts['carried_rows']}, rows kept {ts['rows_kept']}, provenance {table['provenance']} -> {out}")
    rep = {"game": game, "gaps": {k: v for k, v in gp.items() if k != "worklist"}, "worklist": len(gp["worklist"]), "choice": {**choice, "setting": dataclasses.asdict(st)}, "settled": len(settled),
           "deep": {"read": deep["read"], "rows": len(deep["rows"]), "mass": deep["mass"]}, "stats": ts, "table": out, "calibration": None}
    if judge is not None:
        touched = {(e["log"], e["run"], e["game"], e["ply"]) for e in deep["touched"]}
        rng = random.Random(seed)
        seen_t, seen_h = {}, {}
        for d in decision_readings(logs):
            bucket = seen_t if (d["log"], d["run"], d["game"], d["ply"]) in touched else seen_h
            bucket.setdefault((d["start"], tuple(d["moves"])), d)
        sets = {}
        for name, b in (("held_out", seen_h), ("touched", seen_t)):
            keys = sorted(b, key=lambda k: (str(k[0]), k[1]))
            sets[name] = [(k[0], list(k[1])) for k in (keys if len(keys) <= sample else rng.sample(keys, sample))]
        cal = {}
        for name, pos in sets.items():
            for label, tab in (("before", prior_table), ("after", out)):
                c = calibrate(pos, judge, variant=variant, table=tab, judge_kw=judge_kw, seed=seed)
                cal[f"{name}_{label}"] = c
                say(f"OBS wdl_sleep calibrate {game} {name} {label}: positions {c['positions']}, mean abs error {c['mae']}, correlation {c['corr']}, calls {c['calls']}, table hits {c['table_hits']}")
        rep["calibration"] = cal
    else:
        say(f"OBS wdl_sleep calibrate {game}: no judge given, not measured")
    return rep


def main(argv: list) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--log", action="append", required=True)
    ap.add_argument("--prior-table", default=None)
    ap.add_argument("--judge-engine", default=None, help="a UCI engine that reports win / draw / loss: the judge of the readings (training only)")
    ap.add_argument("--judge-depth", type=int, default=12)
    ap.add_argument("--judge-cache", default=None)
    ap.add_argument("--store", default=None, help="a snapshot file: the valuation the wake played by")
    ap.add_argument("--out", default=None)
    ap.add_argument("--deep-seconds", type=float, default=1.0)
    ap.add_argument("--max-deep", type=int, default=200)
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(argv)
    judge = UciWdlJudge(a.judge_engine, a.judge_depth, a.judge_cache) if a.judge_engine else None
    try:
        rep = sleep_wdl(a.game, a.log, variant=a.variant, prior_table=a.prior_table, judge=judge, judge_kw=judge_kw_of_snapshot(a.store) if a.store else None, out=a.out,
                        deep_seconds=a.deep_seconds, max_deep=a.max_deep, sample=a.sample, seed=a.seed)
    finally:
        if judge is not None:
            judge.close()
    print(f"OBS wdl_sleep table written: {rep['table']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
