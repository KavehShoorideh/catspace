"""PLAN 78.93: THE VERIFIED CHOICE A SECOND TIME, AND WHAT THE LATE REDUCTION BUYS. SYSTEM_SPEC 5 (the tactical reasoner's ledger search: the choice it plays, its late-move reduction, the verification of the choice).
Game-free: the game is named only through the pieces that exist (odds_measure, tiebreak_measure, choice_play, symbols_play).

The players are the control of 78.67 (the search of `solve_equals_batch`, its first move among the moves it holds equal) and differ only in the switches of the search:
  now          both prunings on, the verification of the choice off (the engine as it is);
  verify       as now and the verification of the choice on, taking its time from the equal reading's slot (78.92's form, mode "slot");
  verify_main  as verify, the verification taking its time from the main search's clock (mode "main": the deepening stops with `CHOICE_MAIN_SHARE` of its clock left, the equal reading keeps its slot);
  late_off     as now and the late-move reduction off.

INTERFACE (what each piece takes, what it returns, who calls it). Written before the bodies.
  PLAYERS: {name: the player's switches as spec keys null_move / late_reduce / verify_choice / choice_mode}; MATCHES: the four matches as (a, b); `specs_of(a, b, set, pairs, seconds, scratch)` -> the game specs
  (the opening sets A to D of tiebreak_measure, both colours of each opening); `play93(spec)` plays one game (module level for the spawned Pool: the movers are choice_play.ChoiceMover, which counts nodes and what the verification did);
  `games_file(tag, a, b, set, seconds)`; `lines_of(...)` the OBS lines of one match; `pooled_lines(...)` the lines of several sets of one match with their bootstrap band over opening pairs.
  The hint's matches (the ALSO PRINT): `hint_specs(set, pairs, seconds, scratch)` the reasons player of 78.82 against the control, BOTH with the verification of the choice on (symbols_play.play82 plays them).
Stages (command line): overlap | match | hint | report. `match` plays one match on one set at one clock (the load wait of 78.67 first, 9 workers); `hint` the reasons-against-control match; `report` prints every match found (alone,
pooled over C and D, pooled with 78.92's A and B where the match is the same one) and the reading rules of the plan; the numbers are OBS lines of <tag>/obs.txt. Called by the command line only.

  .venv/bin/python -m catspace.incubator.analyst.discovery.choice_play93 overlap | match --a now --b verify --set C --pairs 100 [--seconds 0.2] --tag full | hint --set A | report --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import choice_play as CP
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.io import paths

OUT = ("reasoner_choice_78_93",)
SRC92 = ("reasoner_choice_78_92", "full")                    # 78.92's matches on the sets A and B (read only)
SRC85 = ("reasoner_debts_78_85", "full")                    # 78.85's reasons against control on A and B (read only)
PLAYERS = {"now": {"null_move": True, "late_reduce": True, "verify_choice": False},
           "verify": {"null_move": True, "late_reduce": True, "verify_choice": True, "choice_mode": "slot"},
           "verify_main": {"null_move": True, "late_reduce": True, "verify_choice": True, "choice_mode": "main"},
           "late_off": {"null_move": True, "late_reduce": False, "verify_choice": False}}
MATCHES = (("now", "verify"), ("now", "late_off"), ("verify", "late_off"), ("verify_main", "verify"))
SAME_IN_92 = {("now", "verify"): "now_v_verify", ("now", "late_off"): "now_v_late_off"}     # (the matches 78.92 played on A and B that are the same match)
LONG_CLOCK = 5 * TM.SECONDS                                  # five times the play clock
SETS = ("A", "B", "C", "D")


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


def clock_tag(seconds: float) -> str:
    return f"{seconds:g}s"


def games_file(tag: str, a: str, b: str, set_name: str, seconds: float, kind: str = "") -> Path:
    return out_root(tag) / "matches" / f"{kind}{a}_v_{b}_{set_name}_{clock_tag(seconds)}.jsonl"


def specs_of(a: str, b: str, set_name: str, pairs: int, seconds: float, scratch: Path) -> list:
    base = OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")
    pa = dict(base, kind=TM.KIND["control"], charge=False, **PLAYERS[a])
    pb = dict(base, kind=TM.KIND["control"], charge=False, **PLAYERS[b])
    out = []
    for seed, opening in TM.openings(set_name, pairs):
        for colour in (0, 1):
            out.append({"gid": f"choice93_{a}_v_{b}:{set_name}{seed}:{colour}", "start": f"{set_name}{seed}", "colour": colour, "text": None, "opening": opening, "players": [pa, pb] if colour == 0 else [pb, pa],
                        "seconds": seconds, "max_plies": TM.PLY_CAP, "scratch": str(scratch)})
    return out


def play93(spec: dict) -> dict:
    return CP.play_choice(spec)


def hint_specs(set_name: str, pairs: int, seconds: float, scratch: Path) -> list:
    from catspace.incubator.analyst.discovery import symbols_play as SP
    specs = TM.match_specs("reasons", "control", set_name, pairs, seconds, scratch, False, SP.KIND)
    for s in specs:
        for p in s["players"]:
            p["verify_choice"] = True
            p["choice_mode"] = "slot"
    for s in specs:
        s["gid"] = "hint93_" + s["gid"]
    return specs


def play_hint(spec: dict) -> dict:
    from catspace.incubator.analyst.discovery import symbols_play as SP
    return SP.play82(spec)


def score_lines(tag: str, games: list, a: str, b: str) -> "tuple[str, float, float, float]":
    sc = np.array([OM.white_score(g) if g["colour"] == 0 else 1 - OM.white_score(g) for g in games])
    w, d, l = int((sc == 1).sum()), int((sc == 0.5).sum()), int((sc == 0).sum())
    s, lo, hi = OM.score_band(OM.pair_scores(games))
    unf = sum(g["unfinished"] for g in games)
    return (f"games {len(games)}: {a} W {w} D {d} L {l}; score of {a} {s:.4f} [{lo:.4f}, {hi:.4f}] 90 pct band by bootstrap over opening pairs; unfinished {unf} ({100 * unf / len(games):.1f} pct, counted as draws)", s, lo, hi)


def lines_of(a: str, b: str, set_name: str, games: list, load: float, busy: float, wall: float, seconds: float, workers: int, head: str = "match") -> list:
    tag = f"OBS choice_play93 {head} {a} v {b} [set {set_name}, clock {seconds:g} s]"
    text, _s, _lo, _hi = score_lines(tag, games, a, b)
    out = [f"{tag}: {text}"]
    out += [f"{tag}: {x}" for x in CP.per_side(games, (a, b))]
    out.append(f"{tag}: load average (1 min) at start {load:.2f}, cores busy at the instant of start {busy:.2f}; wall {wall:.0f} s; ledger share {TM.LEDGER_SHARE}, ply cap {TM.PLY_CAP}, workers {workers}")
    return out


def pooled_lines(a: str, b: str, label: str, games: list, seconds: float, head: str = "pooled") -> list:
    tag = f"OBS choice_play93 {head} {a} v {b} [{label}, clock {seconds:g} s]"
    text, _s, _lo, _hi = score_lines(tag, games, a, b)
    return [f"{tag}: {text}"] + [f"{tag}: {x}" for x in CP.per_side(games, (a, b))]


def read_if(p: Path) -> "list | None":
    return TM.read_games(p) if p.exists() else None


def games_of_92(a: str, b: str, set_name: str) -> "list | None":
    stem = SAME_IN_92.get((a, b))
    return read_if(Path(paths.experiment(*SRC92, "matches", f"{stem}_{set_name}.jsonl"))) if stem else None


def report(tag: str, obs: Path) -> None:
    emit = lambda lines: TM.emit(lines, obs)            # noqa: E731
    emit(TM.overlap_lines())
    verdicts = {}
    for seconds in (TM.SECONDS, LONG_CLOCK):
        for a, b in MATCHES:
            alone: dict = {}
            for s in SETS:
                if s in ("A", "B"):
                    g = games_of_92(a, b, s) if seconds == TM.SECONDS else None
                    if g:
                        alone[s] = g
                        emit([f"OBS choice_play93 78.92 {a} v {b} [set {s}, clock {seconds:g} s] read from 78.92's files:"] + CP.per_side(g, (a, b)) + [score_lines("", g, a, b)[0]])
                    continue
                p = games_file(tag, a, b, s, seconds)
                if p.exists():
                    g = TM.read_games(p)
                    alone[s] = g
                    mp = p.with_suffix(".meta.json")
                    meta = json.loads(mp.read_text()) if mp.exists() else {}
                    emit(lines_of(a, b, s, g, meta.get("load", float("nan")), meta.get("busy", float("nan")), meta.get("wall", float("nan")), seconds, meta.get("workers", 9)))
            new = [g for s in ("C", "D") if s in alone for g in alone[s]]
            if len(new) and all(s in alone for s in ("C", "D")):
                emit(pooled_lines(a, b, "sets C and D", new, seconds))
                sc, lo, hi = score_lines("", new, a, b)[1:]
                verdicts[(a, b, seconds)] = (sc, lo, hi)
            every = [g for s in SETS if s in alone for g in alone[s]]
            if all(s in alone for s in SETS):
                emit(pooled_lines(a, b, "sets A, B (78.92) and C, D (78.93)", every, seconds))
                if (a, b, seconds) in verdicts:
                    verdicts[(a, b, seconds, "all")] = score_lines("", every, a, b)[1:]
    for key, v in sorted(verdicts.items(), key=str):
        a, b, seconds = key[0], key[1], key[2]
        where = "sets A to D" if len(key) == 4 else "sets C and D"
        emit([f"OBS choice_play93 READING {a} v {b} [{where}, clock {seconds:g} s]: score of {a} {v[0]:.4f}, lower end of its band {v[1]:.4f}, upper end {v[2]:.4f}; lower end above one half: {'YES' if v[1] > 0.5 else 'NO'}; upper end below one half: {'YES' if v[2] < 0.5 else 'NO'}"])
    # the hint's matches (the ALSO PRINT)
    hint = {}
    for s in ("A", "B"):
        p = games_file(tag, "reasons", "control", s, TM.SECONDS, "hint_")
        if p.exists():
            hint[s] = TM.read_games(p)
            mp = p.with_suffix(".meta.json")
            meta = json.loads(mp.read_text()) if mp.exists() else {}
            emit(lines_of("reasons", "control", s, hint[s], meta.get("load", float("nan")), meta.get("busy", float("nan")), meta.get("wall", float("nan")), TM.SECONDS, meta.get("workers", 9), head="hint, both verify,"))
            old = read_if(Path(paths.experiment(*SRC85, "matches", f"reasons_v_control_{s}.jsonl")))
            if old:
                emit([f"OBS choice_play93 hint beside it, 78.85's reasons v control with the verification off [set {s}]: " + score_lines("", old, "reasons", "control")[0]])
    if len(hint) == 2:
        allh = [g for gs in hint.values() for g in gs]
        emit(pooled_lines("reasons", "control", "sets A and B, both verify", allh, TM.SECONDS, head="hint pooled"))
        old = [g for s in ("A", "B") if (g := read_if(Path(paths.experiment(*SRC85, "matches", f"reasons_v_control_{s}.jsonl"))))]
        if len(old) == 2:
            emit([f"OBS choice_play93 hint pooled beside it, 78.85's with the verification off [sets A and B]: " + score_lines("", [g for gs in old for g in gs], "reasons", "control")[0]])


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("overlap", "match", "hint", "report"))
    ap.add_argument("--a", default="now", choices=tuple(PLAYERS))
    ap.add_argument("--b", default="verify", choices=tuple(PLAYERS))
    ap.add_argument("--set", default="C", choices=SETS)
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"
    if a.stage == "overlap":
        TM.emit(TM.overlap_lines(), obs)
    elif a.stage in ("match", "hint"):
        load, busy = TM.wait_quiet(lambda s: TM.emit([s], obs))
        t0 = time.perf_counter()
        if a.stage == "match":
            gf = games_file(a.tag, a.a, a.b, a.set, a.seconds)
            games = TM.run_games(specs_of(a.a, a.b, a.set, a.pairs, a.seconds, root / "scratch"), a.workers, gf, lambda s: TM.emit([s], obs), play=play93)
            names = (a.a, a.b)
        else:
            gf = games_file(a.tag, "reasons", "control", a.set, a.seconds, "hint_")
            games = TM.run_games(hint_specs(a.set, a.pairs, a.seconds, root / "scratch"), a.workers, gf, lambda s: TM.emit([s], obs), play=play_hint)
            names = ("reasons", "control")
        wall = time.perf_counter() - t0
        gf.with_suffix(".meta.json").write_text(json.dumps({"load": load, "busy": busy, "wall": wall, "workers": a.workers}))
        TM.emit(lines_of(names[0], names[1], a.set, games, load, busy, wall, a.seconds, a.workers, head="match" if a.stage == "match" else "hint, both verify,"), obs)
    else:
        report(a.tag, obs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
