"""PLAN 78.37 stage 1 THE SURPRISE METER. On full games from the game's own start, two readings of every position (both from the mover's side) are put in one unit, the
expected result in [0, 1], and the moves across which one reading jumps and the other does not are counted. The rate of surprise on fresh games is the measure; the heaviest
surprises, written as a ranked list, are the food of the learning step. Nothing here is specific to a game: a position is (start, moves), the outside engine is reached only
through a `read` function (the one adapter that touches a game library is `wdl_sleep.UciWdlJudge.read`), and the engine's own part is `cs_engine`.

  .venv/bin/python -m catspace.incubator.analyst.discovery.surprise_meter <game> --games N --seconds 0.2 [--store SNAPSHOT] [--judge-engine PATH] [--judge-depth 12]
         [--seed S] [--max-plies 160] [--scale X] [--out DIR]

PIECES
  game sources    `games(n, seed) -> [{start, moves, outcome, ...}]`: `SelfPlaySource` (wake.play_self with a trace log: our readings come from the decisions that played the game,
                  nothing is decided twice; rows carry `ours`, the readings by ply) and `RecordsSource` (a JSON Lines file of the same rows; one `solve_batch` over all its positions
                  gives our readings). No human game file is read by anything here.
  our reading     `ours_reading(row)` from a trace row (a `position` row of the trace log; the label is `run:game:ply:role`).
  judge's reading `read(variant, start, moves) -> {"cp": int or None, "wdl": (win, draw, loss)}` for the side to move; `CachedReader` remembers answers on disk.
  common unit     `expected_judge` = win + draw / 2. `expected_ours`: proved win 1, a claim class win / draw / loss 1 / 0.5 / 0, a search value that is the ledger's proof of a win or
                  a loss 1 / 0, otherwise a logistic of the value with ONE scale, `fit_scale` (least squares against the judge's expected result over the positions read; the
                  fitted scale is printed and saved; pass it back with `scale=` to stay comparable).
  jump, surprise  `jump` (across one move, seen from the mover: the reading after the move is the other side's, so it is flipped), `classify` (MISSED, FALSE, OPPOSITE, or a
                  STANDING GAP counted apart), `measure` (the dict) and `format_obs` (lines beginning `OBS `), `ranked` and `write_ranked` (the list, heaviest first).

WHAT THE TRACE CARRIES (read from real rows of a short self-play run and from rust/engine/tactics/src/operators.rs, ledger.rs, rust/engine/decision/src/claim_op.rs)
  * The search's value is the `value` of the step whose `operator` is "ledger" (status "Read"; the forcing step has value None). It is an integer from the MOVER's side (the
    ledger search runs for the side to move; a unit the mover wins is positive), in the worth units of the judge in use: with no valuation record every unit counts 1 (the Count
    judge), with a store's record the record's worths by kind, and a record that weighs our own losses first (`ours_first`) multiplies them, so the unit is the store's own, not
    centipawns. A line that ends the game is the ledger's WIN constant plus or minus the distance, hence the saturation test `SATURATED` below.
  * "Proved" is the row's `proved` flag (a decision that played a proved forced win of the game). The trace carries no proved LOSS: a lost position reads as a search value only.
  * A claim step (operator "claim") speaks through the row's top-level `facts`: relation `root_class` with args [move index, class W / D / U / L, "game" or "claim" or "", value].
    The position's class is win when any root move is W; else, when no root move is U, draw when some root move is D, else loss when every root move is L; otherwise none spoke.
  * A decision with no ledger value (the clock cut the search before a line: status "Gap") and no proof and no claim class has no reading: its position is left out.
  * The first plies of a self-play game that are seeded random moves are not decisions: they have no reading and the moves across them are not measured.
  * The `side` argument of `measure` is the parity of the ply (0: moves made at plies 0, 2, 4 ...; 1: the others) or "ours", which uses each game's `ours_parity` where it has one.

The ranked list and the trace log are training data: they are written only under a directory the caller names; nothing is written into rules/ or a store.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import cs_engine as CE

from catspace.incubator.analyst.discovery import judges as J
from catspace.incubator.analyst.discovery import wake
from catspace.io import paths

SATURATED = (1 << 40) - 1000          # |ledger value| at or above this is a line that ends the game (ledger.rs: WIN = 1 << 40)
MISSED, FALSE, OPPOSITE = "MISSED", "FALSE", "OPPOSITE"
KINDS = (MISSED, FALSE, OPPOSITE)
CLAIM_E = {"win": 1.0, "draw": 0.5, "loss": 0.0}
DEFAULT_ENGINE = "/opt/homebrew/bin/stockfish"


# ------------------------------------------------------------------------------------------------------------------------------------ our reading

def claim_class(facts: list) -> "str | None":
    """The class a claim step gave a position, read off the row's facts (see the module docstring); None when no claim spoke."""
    cls = [f["args"][1] for f in facts or [] if f.get("relation") == "root_class" and len(f.get("args", [])) == 4]
    if not cls:
        return None
    if "W" in cls:
        return "win"
    if "U" in cls:
        return None
    if "D" in cls:
        return "draw"
    return "loss" if all(c == "L" for c in cls) else None


def ours_reading(row: dict) -> "dict | None":
    """{"value": the ledger step's value or None, "proved": bool, "claim": win / draw / loss / None} of a trace `position` row, from the mover's side; None when the row is an
    error or says nothing usable."""
    if row.get("kind") != "position" or row.get("error"):
        return None
    value = next((s["value"] for s in row.get("steps", []) if s.get("operator") == "ledger" and s.get("value") is not None), None)
    out = {"value": value, "proved": bool(row.get("proved")), "claim": claim_class(row.get("facts"))}
    return out if (value is not None or out["proved"] or out["claim"]) else None


def read_log(path: str) -> dict:
    """{(run, game, ply): reading} for every usable `position` row of a trace log (labels `run:game:ply:role`)."""
    out = {}
    for line in open(path):
        row = json.loads(line)
        if row.get("kind") != "position" or not row.get("label"):
            continue
        parts = row["label"].split(":")
        r = ours_reading(row)
        if r is not None and len(parts) == 4:
            out[(parts[0], int(parts[1]), int(parts[2]))] = r
    return out


# ------------------------------------------------------------------------------------------------------------------------------------ the common unit

def expected_judge(reading: dict) -> float:
    w, d, _l = reading["wdl"]
    return w + d / 2


def _logistic(v: float, scale: float) -> float:
    x = max(-60.0, min(60.0, v / scale))
    return 1.0 / (1.0 + math.exp(-x))


LOSS_WEIGHT = 1000          # ledger.rs LOSS: with an ours-first judge a unit of ours lost counts this many times in the search's value


def net_of(value, loss_weight: int = 1) -> float:
    """The search's value as a plain net gain. The value is a CHANGE from the position as it stands (units won less units lost along the line read), not a level; with an
    ours-first judge our losses are weighted `loss_weight` times, which is undone here."""
    if loss_weight <= 1 or value >= 0:
        return float(value)
    lost = (-value + loss_weight - 1) // loss_weight
    return float(value + loss_weight * lost - lost)


def level_of(reading: dict, loss_weight: int = 1) -> float:
    """OUR LEVEL of a position for the mover: the balance of the units on the board by the judge in use (`balance`, 0 when the reading carries none) plus the net gain the
    search reads from there."""
    return float(reading.get("balance") or 0) + net_of(reading["value"], loss_weight)


def add_balance(games: list, balance_of) -> None:
    """Give every reading of ours its `balance`: `balance_of(variant, start, moves) -> number` for the side to move."""
    for g in games:
        for p, r in g.get("ours", {}).items():
            r["balance"] = balance_of(g.get("variant"), g.get("start"), g["moves"][:p])


def codec_balance(game: str, worth: dict):
    """A `balance_of` from the game's position codec (`owned`: each unit's side and kind) and a worth table by kind: the mover's units less the other side's."""
    from catspace.incubator.analyst.discovery import curriculum as CU
    codec = CU.codec_of(game)

    def balance_of(variant, start, moves):
        position, mover = CE.state(variant, start, list(moves))[:2]
        return sum((1 if side == mover else -1) * worth.get(kind, 0) for side, kind in codec.owned(position))
    return balance_of


class _Ones(dict):
    """A worth table where every kind counts one (the judge of a decision made with no record)."""

    def get(self, key, default=None):
        return 1


def uses_value(reading: dict) -> bool:
    """Does the reading's expected result come from the logistic (no proof, no claim class, a value short of the ledger's game-ending lines)?"""
    return not reading["proved"] and not reading["claim"] and reading["value"] is not None and abs(reading["value"]) < SATURATED


def expected_ours(reading: dict, scale: float, loss_weight: int = 1) -> float:
    """Our reading's expected result in [0, 1] (see the module docstring): a proof, a claim class, a line that ends the game, else the logistic of OUR LEVEL."""
    if reading["proved"]:
        return 1.0
    if reading["claim"]:
        return CLAIM_E[reading["claim"]]
    v = reading["value"]
    if abs(v) >= SATURATED:
        return 1.0 if v > 0 else 0.0
    return _logistic(level_of(reading, loss_weight), scale)


def fit_scale(pairs: list) -> float:
    """The one scale of the logistic: the value whose curve is nearest (least squares) to the judge's expected result over `pairs` = [(search value, judge's expected result)].
    A grid over a wide log range, then a finer grid around the best; 1.0 with nothing to fit."""
    pairs = [(float(v), e) for v, e in pairs]
    if not pairs:
        return 1.0

    def err(s):
        return sum((_logistic(v, s) - e) ** 2 for v, e in pairs)
    grid = [10 ** (-2 + 6 * i / 120) for i in range(121)]
    best = min(grid, key=err)
    fine = [best * 10 ** (-0.05 + 0.1 * i / 40) for i in range(41)]
    return min(fine, key=err)


# ------------------------------------------------------------------------------------------------------------------------------------ jump and surprise

def jump(before: float, after: float) -> float:
    """The change of the mover's expected result across its move: `after` is the other side's reading, so it is flipped. Positive: the mover improved."""
    return (1.0 - after) - before


def classify(jump_ours: float, jump_judge: float, threshold: float = 0.15) -> "tuple | None":
    """(kind, weight) of a move, or None. MISSED: the judge jumps, ours does not (weight: the judge's jump). FALSE: ours jumps, the judge does not (weight: ours). OPPOSITE: both jump
    with opposite signs (weight: the judge's jump)."""
    o, j = abs(jump_ours) >= threshold, abs(jump_judge) >= threshold
    if j and not o:
        return MISSED, abs(jump_judge)
    if o and not j:
        return FALSE, abs(jump_ours)
    if o and j and (jump_ours > 0) != (jump_judge > 0):
        return OPPOSITE, abs(jump_judge)
    return None


def standing_gap(e_ours: float, e_judge: float, jump_ours: float, jump_judge: float, threshold: float = 0.15) -> bool:
    """The two expected results differ by the threshold or more and neither reading jumps."""
    return abs(e_ours - e_judge) >= threshold and abs(jump_ours) < threshold and abs(jump_judge) < threshold


# ------------------------------------------------------------------------------------------------------------------------------------ the judge's reading, remembered

class CachedReader:
    """Any `read(variant, start, moves)` remembered on disk (JSON, written atomically by `save`): the same position asked twice gets the same answer. `tag` names the judge so a
    cache of another judge or depth is not read. `asked` and `hits` count the questions and the ones answered from memory."""

    def __init__(self, reader, path: str, tag: str = "", every: int = 200):
        self.inner, self.path, self.tag, self.every = reader, Path(path), tag, every
        self.memory = json.loads(self.path.read_text()) if self.path.exists() else {}
        self.asked = self.hits = self.unsaved = 0

    def read(self, variant: str, start, moves: list) -> dict:
        key = json.dumps([self.tag, variant, start, list(moves)])
        self.asked += 1
        e = self.memory.get(key)
        if e is not None:
            self.hits += 1
            return {"cp": e["cp"], "wdl": tuple(e["wdl"])}
        r = self.inner.read(variant, start, list(moves))
        self.memory[key] = {"cp": r["cp"], "wdl": list(r["wdl"])}
        self.unsaved += 1
        if self.every and self.unsaved >= self.every:
            self.save()
        return {"cp": r["cp"], "wdl": tuple(r["wdl"])}

    def save(self) -> None:
        if not self.unsaved:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.memory))
        os.replace(tmp, self.path)
        self.unsaved = 0

    def close(self) -> None:
        try:
            self.save()
        finally:
            c = getattr(self.inner, "close", None)
            if callable(c):
                c()


# ------------------------------------------------------------------------------------------------------------------------------------ game sources

class SelfPlaySource:
    """The engine against itself (wake.play_self) with a trace log: `ours` of each game is read from that log, the decisions that played it."""

    def __init__(self, variant: str, *, seconds: float = 0.2, store: "str | None" = None, specs: "dict | None" = None, log: str, random_plies: int = 4,
                 max_plies: int = 160, threads: int = 0, run: str = "m"):
        self.variant, self.seconds, self.store, self.specs, self.log = variant, seconds, store, specs, log
        self.random_plies, self.max_plies, self.threads, self.run = random_plies, max_plies, threads, run
        self.snapshot = ""

    def games(self, n: int, seed: int) -> list:
        if os.path.exists(self.log):
            os.remove(self.log)
        Path(self.log).parent.mkdir(parents=True, exist_ok=True)
        run = f"{self.run}{seed}"
        rows = wake.play_self(self.variant, n, seconds=self.seconds, random_plies=self.random_plies, max_plies=self.max_plies, seed=seed, store=self.store,
                              specs=self.specs, threads=self.threads, log=self.log, run=run)
        ours = read_log(self.log)
        self.snapshot = rows[0]["snapshot"] if rows else ""
        return [{"variant": self.variant, "game": g["game"], "start": g["start"], "moves": g["moves"], "outcome": g["outcome"], "unfinished": g["unfinished"],
                 "ours": {p: r for (rn, gm, p), r in ours.items() if rn == run and gm == g["game"]}} for g in rows]


def solve_ours(rows: list, variant: str, *, seconds: float, store: "str | None", specs: "dict | None", log: str, threads: int = 0, run: str = "r") -> None:
    """Fill `ours` of rows that lack it: ONE `solve_batch` call over every non-final position of every game, with a trace log."""
    todo = [(i, p) for i, r in enumerate(rows) if "ours" not in r for p in range(len(r["moves"]))]
    for r in rows:
        r.setdefault("ours", {})
    if not todo:
        return
    if os.path.exists(log):
        os.remove(log)
    Path(log).parent.mkdir(parents=True, exist_ok=True)
    positions = [(variant, rows[i].get("start"), list(rows[i]["moves"][:p])) for i, p in todo]
    labels = [f"{run}:{i}:{p}:0" for i, p in todo]
    CE.solve_batch(positions, seconds, snapshot=run, trace_path=log, threads=threads, store=store, specs=specs or {}, labels=labels)
    ours = read_log(log)
    for i, p in todo:
        if (run, i, p) in ours:
            rows[i]["ours"][p] = ours[(run, i, p)]


class RecordsSource:
    """A JSON Lines file of game rows {start, moves, outcome}: our readings come from one `solve_batch` over all positions (`solve_ours`, called by the caller of `games`)."""

    def __init__(self, path: str, variant: str):
        self.path, self.variant = path, variant

    def games(self, n: int, seed: int) -> list:
        rows = [json.loads(x) for x in open(self.path) if x.strip()]
        rows = [{"variant": self.variant, "game": i, "start": r.get("start"), "moves": list(r["moves"]), "outcome": r.get("outcome")} for i, r in enumerate(rows)]
        return rows[:n]


# ------------------------------------------------------------------------------------------------------------------------------------ the measure

def collect(games: list, reader, side=None) -> list:
    """The moves the meter can measure: [{game, ply, parity, start, moves_before, move, variant, ours_before, ours_after, judge_before, judge_after}]. A move is measured when
    both the position before it and the one after it have a reading of ours, and the `side` filter (parity of the ply, or "ours") lets it through. The judge is asked for those
    positions only."""
    out = []
    for g in games:
        ours = g.get("ours", {})
        for p in range(len(g["moves"])):
            if p not in ours or (p + 1) not in ours:
                continue
            if side == "ours":
                if g.get("ours_parity") is not None and p % 2 != g["ours_parity"]:
                    continue
            elif side is not None and p % 2 != side:
                continue
            v, st = g.get("variant"), g.get("start")
            out.append({"game": g.get("game"), "ply": p, "parity": p % 2, "variant": v, "start": st, "moves_before": list(g["moves"][:p]), "move": g["moves"][p],
                        "ours_before": ours[p], "ours_after": ours[p + 1],
                        "judge_before": reader.read(v, st, g["moves"][:p]), "judge_after": reader.read(v, st, g["moves"][:p + 1])})
    return out


def evaluate(moves: list, *, scale: "float | None" = None, threshold: float = 0.15, cp_cap: int = 1000, loss_weight: int = 1) -> dict:
    """Everything the meter says about collected moves: the fitted (or given) scale, one record per move with the readings in the common unit, the jumps, the kind and the weight."""
    fitted = scale is None
    if fitted:
        seen, pairs = set(), []
        for m in moves:
            for k, j in (("ours_before", "judge_before"), ("ours_after", "judge_after")):
                if uses_value(m[k]):
                    key = (m["game"], m["ply"] + (k == "ours_after"))
                    if key not in seen:
                        seen.add(key)
                        pairs.append((level_of(m[k], loss_weight), expected_judge(m[j])))
        scale = fit_scale(pairs)
    recs = []
    for m in moves:
        eo0, eo1 = expected_ours(m["ours_before"], scale, loss_weight), expected_ours(m["ours_after"], scale, loss_weight)
        ej0, ej1 = expected_judge(m["judge_before"]), expected_judge(m["judge_after"])
        jo, jj = jump(eo0, eo1), jump(ej0, ej1)
        c = classify(jo, jj, threshold)
        cp0, cp1 = m["judge_before"]["cp"], m["judge_after"]["cp"]
        recs.append({**m, "e_ours_before": eo0, "e_ours_after": eo1, "e_judge_before": ej0, "e_judge_after": ej1, "jump_ours": jo, "jump_judge": jj,
                     "kind": c[0] if c else None, "weight": c[1] if c else 0.0, "gap": standing_gap(eo0, ej0, jo, jj, threshold),
                     "loss_cp": None if cp0 is None or cp1 is None else min(cp_cap, max(0, cp0 + cp1)), "loss_e": max(0.0, ej0 - (1.0 - ej1))})
    return {"scale": scale, "scale_fitted": fitted, "records": recs}


def measure(games: list, reader, *, scale: "float | None" = None, threshold: float = 0.15, cp_cap: int = 1000, side=None, clock: "float | None" = None,
            snapshot: str = "", judge: "dict | None" = None, loss_weight: int = 1) -> dict:
    """The measure of the games as a dict: positions read, surprises per 100 positions by kind, the weighted rate, standing gaps per 100, the mean loss of the mover's moves in the
    judge's centipawns (capped per move) and in expected result, with the clock, snapshot, judge and the scale. The records (for `ranked`) are under "records"."""
    ev = evaluate(collect(games, reader, side), scale=scale, threshold=threshold, cp_cap=cp_cap, loss_weight=loss_weight)
    recs = ev["records"]
    n = len(recs)
    per = (lambda x: 100.0 * x / n) if n else (lambda x: 0.0)
    by_kind = {k: per(sum(1 for r in recs if r["kind"] == k)) for k in KINDS}
    cps = [r["loss_cp"] for r in recs if r["loss_cp"] is not None]
    return {"positions": n, "games": len(games), "threshold": threshold, "cp_cap": cp_cap, "side": side, "by_kind": by_kind, "surprises_per_100": sum(by_kind.values()),
            "counts": {k: sum(1 for r in recs if r["kind"] == k) for k in KINDS},
            "weighted_per_100": per(sum(r["weight"] for r in recs)), "gaps_per_100": per(sum(1 for r in recs if r["gap"])),
            "mean_loss_cp": sum(cps) / len(cps) if cps else None, "mean_loss_e": sum(r["loss_e"] for r in recs) / n if n else None,
            "clock": clock, "snapshot": snapshot, "judge": judge or {}, "loss_weight": loss_weight, "scale": ev["scale"], "scale_fitted": ev["scale_fitted"], "records": recs}


def format_obs(m: dict) -> list:
    """The measure as printed lines, each beginning `OBS `."""
    f = lambda x, p=2: "none" if x is None else f"{x:.{p}f}"  # noqa: E731
    k = m["by_kind"]
    return [
        f"OBS surprise meter: {m['games']} games, {m['positions']} positions read, side {m['side']}, jump threshold {m['threshold']}",
        f"OBS surprises per 100 positions: MISSED {f(k[MISSED])}, FALSE {f(k[FALSE])}, OPPOSITE {f(k[OPPOSITE])}, all {f(m['surprises_per_100'])}",
        f"OBS weighted rate per 100 positions: {f(m['weighted_per_100'], 3)}; standing gaps per 100: {f(m['gaps_per_100'])}",
        f"OBS mean loss of the mover's moves: {f(m['mean_loss_cp'], 1)} judge centipawns (cap {m['cp_cap']} a move), {f(m['mean_loss_e'], 4)} expected result",
        f"OBS clock {m['clock']} s, snapshot {m['snapshot'] or 'none'}, judge {json.dumps(m['judge'], sort_keys=True)}",
        f"OBS scale {m['scale']:.6g} ({'fitted on these positions' if m['scale_fitted'] else 'given'}): pass it back with --scale to stay comparable",
    ]


def ranked(m: dict) -> list:
    """The surprises of a measure, heaviest first: rows with start, the moves up to the position, the move played, kind, weight, both readings before and after."""
    rows = [{"game": r["game"], "ply": r["ply"], "variant": r["variant"], "start": r["start"], "moves": r["moves_before"], "move": r["move"], "kind": r["kind"], "weight": r["weight"],
             "ours": {"before": r["ours_before"], "after": r["ours_after"], "e_before": r["e_ours_before"], "e_after": r["e_ours_after"], "jump": r["jump_ours"]},
             "judge": {"before": r["judge_before"], "after": r["judge_after"], "e_before": r["e_judge_before"], "e_after": r["e_judge_after"], "jump": r["jump_judge"]}}
            for r in m["records"] if r["kind"]]
    rows.sort(key=lambda x: -x["weight"])
    return rows


def write_ranked(m: dict, directory: str) -> dict:
    """Write `surprises.jsonl` (the ranked list) and `measure.json` (the measure without its records) under `directory`, which the caller names; -> the two paths."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    rp, mp = d / "surprises.jsonl", d / "measure.json"
    with open(rp, "w") as f:
        for r in ranked(m):
            f.write(json.dumps(r) + "\n")
    mp.write_text(json.dumps({k: v for k, v in m.items() if k != "records"}, indent=1, sort_keys=True))
    return {"ranked": str(rp), "measure": str(mp)}


# ------------------------------------------------------------------------------------------------------------------------------------ command line

def main(argv: list) -> int:
    from catspace.incubator.analyst.discovery import game_store as GS
    from catspace.incubator.analyst.discovery import wdl_sleep as WS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("game")
    ap.add_argument("--variant", default=None)
    ap.add_argument("--games", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=0.2)
    ap.add_argument("--store", default=None, help="a snapshot file (default: the game's HEAD, if any; 'none' for no store)")
    ap.add_argument("--judge-engine", default=shutil.which("stockfish") or DEFAULT_ENGINE)
    ap.add_argument("--judge-depth", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-plies", type=int, default=160)
    ap.add_argument("--random-plies", type=int, default=4)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--cp-cap", type=int, default=1000)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--records", default=None, help="a JSON Lines file of game rows instead of self-play")
    ap.add_argument("--judge-cache", default=None, help="the judge's remembered answers (default: judge_cache.json under --out)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    variant = a.variant or a.game
    tag = time.strftime("%Y%m%d-%H%M%S")
    out = a.out or str(paths.derived_dir() / f"surprise_{tag}")
    store = a.store
    if store is None and GS.head(a.game):
        store = str(GS.path_of(a.game, GS.head(a.game)))
    if store == "none":
        store = None
    specs = GS.spec_texts(a.game) if store else None
    log = str(Path(out) / "trace.jsonl")
    t0 = time.perf_counter()
    snapshot = CE.snapshot_info(store)["id"] if store else ""
    if a.records:
        src = RecordsSource(a.records, variant)
        games = src.games(a.games, a.seed)
        solve_ours(games, variant, seconds=a.seconds, store=store, specs=specs, log=log)
    else:
        src = SelfPlaySource(variant, seconds=a.seconds, store=store, specs=specs, log=log, random_plies=a.random_plies, max_plies=a.max_plies)
        games = src.games(a.games, a.seed)
    # OUR LEVEL: the balance of the units by the judge the decisions used (the snapshot's record; with none, every unit counts one and our losses come first)
    own = (json.loads(Path(store).read_bytes()).get("judge") if store else None) or {"worth": {}, "ours_first": True}
    worth = own["worth"]
    add_balance(games, codec_balance(a.game, worth) if worth else codec_balance(a.game, _Ones()))
    loss_weight = LOSS_WEIGHT if own.get("ours_first") else 1
    with open(Path(out) / "games.jsonl", "w") as f:
        for g in games:
            f.write(json.dumps({**g, "ours": {str(k): v for k, v in g["ours"].items()}}) + "\n")
    judge = WS.UciWdlJudge(a.judge_engine, depth=a.judge_depth)
    desc = {**J.describe(judge), "depth": a.judge_depth}
    reader = CachedReader(judge, a.judge_cache or str(Path(out) / "judge_cache.json"), tag=json.dumps(desc, sort_keys=True))
    try:
        m = measure(games, reader, scale=a.scale, threshold=a.threshold, cp_cap=a.cp_cap, clock=a.seconds, snapshot=snapshot, judge=desc, loss_weight=loss_weight)
    finally:
        reader.close()
    files = write_ranked(m, out)
    for line in format_obs(m):
        print(line)
    print(f"OBS surprise meter: {time.perf_counter() - t0:.1f} s, judge asked {reader.asked} (from the cache {reader.hits}) -> {files['ranked']}, {files['measure']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
