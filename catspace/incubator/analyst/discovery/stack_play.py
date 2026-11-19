"""PLAN 79.00 PART ONE, THE THREE THINGS TOGETHER IN PLAY (SYSTEM_SPEC 4 the equal reading and the verified choice, 5 the measured records, 7 the pick among equal moves; piece: tactical reasoning, the match). A sandbox: nothing is wired into any player of the wake, the DEFAULT record and
the default switches of the engine are not changed anywhere (the record, the verification of the choice and the pick are PARAMETERS of the player's spec), nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/stack_79_00/<tag>/.
Game-free: the game is named only through the pieces that exist (odds_measure, tiebreak_measure, symbols_play, record_play). THIS MODULE AND WHAT IT IMPORTS NEVER LOAD THE ORACLE WRAPPER (the wall; tests/test_plan79_00_stack.py).

THE THREE THINGS. record: the judge's measured record of worths (78.99: "human" the human results' record, "odds" the record every match since 78.61 was played with); verify: the verification of the choice (78.92, 78.93: the engine's `verify_choice`, mode slot, passed to the
search call by tiebreak_measure.Mover67.choice_kw); pick: 78.82's pick among the moves the search holds equal (symbols_play: the reasons' pick inside the set the search hands back). A player is the three of them; what differs between two sides of a match is what the plan names, nothing else.

TWO CLOCKS. The harness gives one clock to a game (spec "seconds"); a player spec may carry its own "seconds", which the mover below uses in place of the game's. Nothing else of the mover changes.

INTERFACE (written before the bodies).
  PLAYERS[name] -> {"record", "pick", "verify", "seconds"?}   the named sides of the plan's matches.
  player_spec(name, charge) -> dict       {"kind", "worth", "charge", "verify_choice", "choice_mode", "seconds"?}.
  match_specs(a, b, set_name, pairs, seconds, scratch, charge_a, charge_b) -> [spec]    the games of one match, sets A and B of 78.67, colours alternated.
  play79(spec) -> game                    one game; module level so the spawned workers import it.
  stop_of(min_pairs) -> f(games) -> word|None    record_play.stop_of (the early stop is not asked before that many whole opening pairs).
  replacements(games, who) -> {...}       what the verification did on one side over the games (asked, held, replaced, failed, cut).
  ahead(rows) -> bool                     the rule of the plan's conditions: every set's 90 pct band lies above a half (decided before any match was played).
  Called by: the command line (`sanity`, `match`, `chain`, `report`).

  .venv/bin/python -m catspace.incubator.analyst.discovery.stack_play sanity | match --a human+v --b human --set A --tag full | chain --tag full | report --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.incubator.analyst.discovery import symbols_play as SP
from catspace.io import paths

OUT = ("stack_79_00",)
MIN_PAIRS = RP.MIN_PAIRS
CLOCK_FIVE = 5 * TM.SECONDS                          # the yardstick clock of match 4: five times the play clock
PLAYERS = {"odds": {"record": "odds", "pick": False, "verify": False},
           "human": {"record": "human", "pick": False, "verify": False},
           "human+v": {"record": "human", "pick": False, "verify": True},
           "human+pick": {"record": "human", "pick": True, "verify": False},
           "human+v+pick": {"record": "human", "pick": True, "verify": True},
           "odds@5x": {"record": "odds", "pick": False, "verify": False, "seconds": CLOCK_FIVE}}
MATCHES = {"sanity": ("human+v+pick", "human+v+pick"), "m1": ("human+v", "human"), "m2": ("human+v+pick", "human+pick"), "m3": ("human+v+pick", "odds"), "m4": ("human+v+pick", "odds@5x")}
VC = ("vc_asked", "vc_held", "vc_replaced", "vc_failed", "vc_cut")


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


def player_spec(name: str, charge: bool = False) -> dict:
    p = PLAYERS[name]
    spec = {"kind": RP.KINDS[bool(p["pick"])], "worth": RP.worth_of(p["record"]), "charge": bool(charge), "verify_choice": bool(p["verify"]), "choice_mode": "slot"}
    if "seconds" in p:
        spec["seconds"] = float(p["seconds"])
    return spec


def match_specs(a: str, b: str, set_name: str, pairs: int, seconds: float, scratch: Path, charge_a: bool = False, charge_b: bool = False) -> list:
    pa, pb = player_spec(a, charge_a), player_spec(b, charge_b)
    tag = ("_c" if charge_a else "") + ("_c" if charge_b else "")
    specs = []
    for seed, opening in TM.openings(set_name, pairs):
        for colour in (0, 1):
            specs.append({"gid": f"{a}_v_{b}{tag}:{set_name}{seed}:{colour}", "start": f"{set_name}{seed}", "colour": colour, "text": None, "opening": opening,
                          "players": [pa, pb] if colour == 0 else [pb, pa], "seconds": seconds, "max_plies": TM.PLY_CAP, "scratch": str(scratch)})
    return specs


class Mover79(SP.MoverReasons):
    """78.82's mover (it plays 78.67's search when the kind carries no pick) with its own clock when the player's spec names one."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, float(spec.get("seconds", seconds)), seed, trace)


def play79(spec: dict) -> dict:
    """One game with the movers above; the per-seat stats ride with the result."""
    OM.Mover = Mover79
    TM.REG.clear()
    g = OM.play_game(spec)
    g["stats"] = [dict(m.stats) for m in TM.REG]
    return g


def stop_of(min_pairs: int = MIN_PAIRS):
    return RP.stop_of(min_pairs)


def replacements(games: list, who: int) -> dict:
    """What the verification did on side `who` (0: a, 1: b) over the games: the decisions asked, held, replaced, failed (kept) and cut."""
    tot = {k: 0 for k in VC}
    dec = 0
    for g in games:
        st = g["stats"][TM.seat_of(g, who)]
        dec += st["decisions"]
        for k in VC:
            tot[k] += int(st.get(k, 0))
    tot["decisions"] = dec
    return tot


def ahead(rows: list) -> bool:
    """The plan's condition for a further match, decided before any match was played: in every set played, the 90 pct band of the score lies wholly above a half. `rows`: [(score, lo, hi)]."""
    return bool(rows) and all(lo > 0.5 for _s, lo, _h in rows)


def lines_of(name: str, a: str, b: str, set_name: str, games: list, load0: float, load1: float, busy: float, wall: float, seconds: float, workers: int, charged: str = "") -> list:
    plain = SP._plain(games)
    out = [x.replace("OBS tiebreak match", f"OBS stack {name}", 1) for x in TM.match_lines(a, b, set_name, plain, load0, busy, wall, seconds, workers)]
    for who, nm in ((0, a), (1, b)):
        r = replacements(games, who)
        if r["vc_asked"]:
            out.append(f"OBS stack {name} {a} v {b} [set {set_name}]: {nm}: the verification asked {r['vc_asked']} times in {r['decisions']} decisions: held {r['vc_held']}, REPLACED the choice {r['vc_replaced']} ({100 * r['vc_replaced'] / max(1, r['vc_asked']):.2f} pct of the verified decisions), failed {r['vc_failed']}, cut {r['vc_cut']}")
    out.append(f"OBS stack {name} {a} v {b} [set {set_name}]: load average (1 min) at the start {load0:.2f}, at the end {load1:.2f}; the pick is 78.82's, {'CHARGED to the clock of: ' + charged if charged else 'uncharged'}; clocks {a} {PLAYERS[a].get('seconds', seconds)} s, {b} {PLAYERS[b].get('seconds', seconds)} s a move")
    return out


def games_file(tag: str, name: str, a: str, b: str, set_name: str, charged: str = "") -> Path:
    return out_root(tag) / "matches" / f"{name}_{a}_v_{b}_{set_name}{'_charged' if charged else ''}.jsonl"


def cmd_match(name: str, a: str, b: str, set_name: str, pairs: int, seconds: float, workers: int, tag: str, say, stop: bool = True, charge_a: bool = False, charge_b: bool = False) -> list:
    root = out_root(tag)
    (root / "matches").mkdir(parents=True, exist_ok=True)
    charged = " and ".join(n for n, c in ((a, charge_a), (b, charge_b)) if c)
    say(f"OBS stack {name} {a} v {b} [set {set_name}]: the early stop is asked with min_pairs={MIN_PAIRS} (99 pct band on whole opening pairs, clear of a half or within 0.03 of it); sides: {a} {PLAYERS[a]}, {b} {PLAYERS[b]}; records {RP.RECORD_OF[PLAYERS[a]['record']]} and {RP.RECORD_OF[PLAYERS[b]['record']]}")
    load0, busy = TM.wait_quiet(say)
    t0 = time.perf_counter()
    games = TM.run_games(match_specs(a, b, set_name, pairs, seconds, root / "scratch", charge_a, charge_b), workers, games_file(tag, name, a, b, set_name, charged), say, play=play79, stop=stop_of() if stop else None)
    load1 = os.getloadavg()[0]
    wall = time.perf_counter() - t0
    for x in lines_of(name, a, b, set_name, games, load0, load1, busy, wall, seconds, workers, charged):
        say(x)
    return games


def band_of(games: list) -> tuple:
    return OM.score_band(OM.pair_scores(games))


def word_of(games: list) -> "str | None":
    return TM.settled(games, min_pairs=MIN_PAIRS)


def cmd_chain(tag: str, seconds: float, workers: int, say) -> None:
    """The plan's matches in order, one at a time: sanity; m1; m2 (sets A and B); m2 charged on the stack side when m2 is ahead; m3; m4 when m3 is clear. The conditions are `ahead` (decided before any match)."""
    done: dict = {}

    def two_sets(name: str, a: str, b: str, **kw) -> list:
        rows = []
        for s in ("A", "B"):
            g = cmd_match(name, a, b, s, 100, seconds, workers, tag, say, **kw)
            rows.append((band_of(g), word_of(g), len(g)))
        done[name] = [r[0] for r in rows]
        return done[name]
    cmd_match("sanity", *MATCHES["sanity"], "A", 20, seconds, workers, tag, say, stop=False)
    for name in ("m1", "m2", "m3"):
        two_sets(name, *MATCHES[name])
        if name == "m2" and ahead(done["m2"]):
            say("OBS stack m2: ahead uncharged in both sets (the 90 pct band wholly above a half), so match 2 is played once CHARGED for the side with the verified choice and the pick; the other side uncharged")
            two_sets("m2c", *MATCHES["m2"], charge_a=True)
        elif name == "m2":
            say("OBS stack m2: not ahead in both sets (the plan's condition for the charged match), the charged match is not played")
    if ahead(done["m3"]):
        two_sets("m4", *MATCHES["m4"])
    else:
        say("OBS stack m3: not clear of a half in both sets, match 4 (the yardstick of five times the clock) is not played, as the plan says")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("sanity", "match", "chain", "report"))
    ap.add_argument("--name", default="m")
    ap.add_argument("--a", default="human+v", choices=tuple(PLAYERS))
    ap.add_argument("--b", default="human", choices=tuple(PLAYERS))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--pairs", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--tag", default="full")
    ap.add_argument("--charge-a", action="store_true")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"

    def say(line: str) -> None:
        TM.emit([line], obs)
    if a.stage == "sanity":
        cmd_match("sanity", *MATCHES["sanity"], "A", 20, a.seconds, a.workers, a.tag, say, stop=False)
    elif a.stage == "match":
        cmd_match(a.name, a.a, a.b, a.set, a.pairs, a.seconds, a.workers, a.tag, say, charge_a=a.charge_a)
    elif a.stage == "chain":
        cmd_chain(a.tag, a.seconds, a.workers, say)
    else:
        for p in sorted((root / "matches").glob("*.jsonl")):
            print(p.name, len([x for x in p.read_text().splitlines() if x.strip()]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
