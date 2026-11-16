"""PLAN 78.64 A (and its amendment): THE LIST OF EVENTS IS READ FROM THE RULES. A measurement in a sandbox; nothing is wired into any player, nothing is written to rules/learned or data/derived.
Game-free: no event, unit, kind, side or fluent is named in this file; every word below is read at run time from the rules' description (roles, goal values, fluents), from the engine's graph (kinds,
sides, units) or from the unit tracker of 78.54 (the difference of two consecutive unit lists). No engine evaluation, no labeller: what happened on the board and how the game ended.

WHAT IS AN EVENT. Two sources, both derived:
  1. THE CENSUS. A move changes the list of units of a position. The tracker (unit_fates.steps_between, the mirror of sheet/fates.rs) says what became of each unit of the position before: it STAYED or
     MOVED (no event: the census is unchanged), it was TAKEN (a unit of the other side stands on its place; the event carries the side that took, the kind taken, the kind that took), it CHANGED KIND (gone, and
     a unit of the same side and another kind has appeared; the event carries the side, the kind it was and the kind it became), or it VANISHED (gone, no unit on its place, no unit appeared: a taking that leaves
     the victim's place, the tracker's stated limit). The sides and kinds are the engine's words.
  2. THE RULES' FLUENTS THAT CAN ONLY GO FALSE. A fluent of the description whose every `next` clause carries the fluent itself, with the same arguments, in its own body (it is kept, never made) can go
     false and never true again: read from the clauses by `persistent_fluents`. Its instances are the `init` facts of the description. The event is an instance going from true to false. Where the rules'
     reading is not reachable a fact is found on the games instead and printed as a CANDIDATE "found on games, not read from the rules" (`monotone_candidates`; printed, not fitted).
  3. THE END. The description's goal values (the constants of the `goal` clauses) and its roles give the ways a game can end: for each role, the highest value; and the value shared by all roles when none
     has the highest. The record of a game says which of them happened (the first mover's result against the engine's first mover).
Each event is read as a CLASS at several grains (kind taken and kind that took; kind taken; any), by the side it belongs to. The program prints every class with its count before any fit.

ONE DERIVED READING ON TOP: "a side wins a unit and KEEPS it" -- a taking followed by a settled position (a quiet row of 78.59) with the taker's side not having lost, in between, a unit that the human
record's order of kinds (heads.kind_baseline_human of 78.59, the same record as 78.62) puts at or above the unit it took. A taking with no settled position after it before the game ends is CUT (unresolved),
not counted either way.

  .venv/bin/python -m catspace.incubator.analyst.discovery.events_rules --out-dir O [--games N] [--state-games]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import game_store as GS
from catspace.incubator.analyst.discovery import gdl as GDL
from catspace.incubator.analyst.discovery import graph_layer1 as GL
from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import unit_fates as UF
from catspace.io import paths

MIN_EVENTS = 100                      # a class is fitted with at least this many events in the FIT games; fewer are printed as too rare (candidates for the endgame fill)
HEADS_DIR = ("estimator_78_59", "full")
GAME, VARIANT = HD.GAME, HD.VARIANT


# ------------------------------------------------------------------------------------------------------------------------------ what the rules say

def rule_clauses(text: str) -> list:
    return GDL.clauses(GDL.parse(text))


def persistent_fluents(clauses: list) -> list:
    """The fluents (relation names without the `next.` prefix) whose every `next` clause holds the fluent itself, with the same arguments, in its body: kept, never made."""
    heads: dict = {}
    for c in clauses:
        rel, args = c.head
        if rel.startswith("next."):
            f = rel[len("next."):]
            kept = any(l[0] == "pos" and l[1] == "true." + f and tuple(l[2]) == tuple(args) for l in c.body)
            heads[f] = heads.get(f, True) and kept
    return sorted(f for f, ok in heads.items() if ok)


def init_instances(clauses: list, fluent: str) -> list:
    return sorted(tuple(c.head[1]) for c in clauses if c.head[0] == "init." + fluent and not c.body)


def role_words(clauses: list) -> list:
    return [c.head[1][0] for c in clauses if c.head[0] == "role" and not c.body]


def goal_values(clauses: list) -> list:
    """The constants a role's goal can take, as numbers, from the heads of the `goal` clauses."""
    vals = set()
    for c in clauses:
        if c.head[0] == "goal":
            v = c.head[1][-1]
            if not str(v).startswith("?"):
                vals.add(int(v))
    return sorted(vals)


def end_outcomes(roles: list, values: list) -> list:
    """The ways a game ends, as (name, winner or None): each role at the highest goal value; or none (every role at the shared value)."""
    top = values[-1]
    return [(f"goal {top} for {r}", r) for r in roles] + [("no role at the highest goal value", None)]


# ------------------------------------------------------------------------------------------------------------------------------ the census events

def census_events(prev: list, nxt: list, mover: str) -> list:
    """The events of one move: tuples (type, actor side, victim side, word a, word b). prev, nxt: lists of (side, kind, place); `mover` the side that moved."""
    steps, _amb = UF.steps_between(prev, nxt)
    used = {s[1] for s in steps if s[0] == "alive"}
    free = [c for j, c in enumerate(nxt) if j not in used]
    out = []
    for i, s in enumerate(steps):
        o, k, at = prev[i]
        if s[0] == "taken":
            taker = [c for c in nxt if c[2] == at and c[0] != o][0]
            out.append(("taken", taker[0], o, k, taker[1]))
        elif s[0] == "other":
            same = [c for c in free if c[0] == o]
            if same:
                free.remove(same[0])
                out.append(("changed kind", o, o, k, same[0][1]))
            else:
                out.append(("vanished", mover, o, k, ""))
    return out


def classes_of(ev: tuple) -> list:
    """The class keys an event belongs to, finest first: (type, side, words...) with the side first so a head can be taken per side."""
    typ, actor, _victim, a, b = ev
    if typ == "taken":
        return [(typ, actor, a, b), (typ, actor, a), (typ, actor)]
    if typ == "changed kind":
        return [(typ, actor, a, b), (typ, actor)]
    return [(typ, actor, a), (typ, actor)]


# ------------------------------------------------------------------------------------------------------------------------------ one game

_W: dict = {}


def _init_worker(spec: str, vocab_json: str, fluents: list, worth: dict, roles: list, edge: dict) -> None:
    _W["model"] = GL.model(GAME, VARIANT)
    _W["spec"], _W["vocab"], _W["fluents"], _W["worth"], _W["roles"], _W["edge"] = spec, vocab_json, fluents, worth, roles, edge


def fluent_facts(moves: list, start: "str | None" = None) -> list:
    """Per position, the set of (fluent, args) of the persistent fluents that hold (the engine's fact layer read through the description, perspective off)."""
    import cs_engine as CE
    out = []
    want = set(_W["fluents"])
    for t in range(len(moves) + 1):
        fs = CE.facts(VARIANT, _W["spec"], _W["vocab"], start, moves=list(moves[:t]), perspective=False)
        out.append(frozenset((f[0], tuple(f[1:])) for f in fs if f[0] in want))
    return out


def read_game(job: tuple) -> dict:
    """The events of one game, step by step. job: (index, moves, quiet plies (positions that are settled rows)). Step t is the move from position t to t+1."""
    idx, moves, quiet, start = job
    model = _W["model"]
    roles = _W["roles"]
    _kinds, plies = model.unit_fates((start, moves), [])
    movers = [p[0] for p in plies]
    units = [[(o, k, at) for (o, k, at, _f) in p[1]] for p in plies]
    n = len(moves)
    ev_steps: dict = {}
    fine: list = []
    for t in range(n):
        m = movers[t]
        for ev in census_events(units[t], units[t + 1], m):
            fine.append((t, ev))
            for c in classes_of(ev):
                ev_steps.setdefault(c, []).append(t)
    alternates = all(movers[t] is not None and movers[t + 1] is not None and movers[t] != movers[t + 1] for t in range(n - 1)) and movers[0] is not None
    # fluents that can only go false
    facts = fluent_facts(moves, start) if _W["fluents"] else [frozenset()] * (n + 1)
    violations = 0
    off: list = []
    for t in range(n):
        for f in facts[t] - facts[t + 1]:
            off.append((t, f))
        violations += len(facts[t + 1] - facts[t])
    for t, (f, args) in off:
        sides = [a for a in args if a in roles]
        side = sides[0] if sides else ""
        rest = tuple(a for a in args if a not in roles)
        for c in ((side, f) + rest, (side, f)):
            ev_steps.setdefault(("went false",) + c, []).append(t)
    # wins a unit and keeps it: a taking, then the next settled position; the taker's side must not have lost, in between, a unit at or above the worth of the one it took
    worth = _W["worth"]
    quiet = sorted(q for q in quiet)
    kept: dict = {}
    cut: dict = {}
    for t, ev in fine:
        if ev[0] != "taken":
            continue
        _typ, actor, _v, kind_taken, _by = ev
        later = [q for q in quiet if q > t]
        keys = [("keeps", actor, kind_taken), ("keeps", actor)]
        if not later:
            for c in keys:
                cut.setdefault(c, []).append(t)
            continue
        q = later[0]
        gave_back = False
        for s in range(t + 1, q):
            for later_ev in census_events(units[s], units[s + 1], movers[s]):
                if later_ev[0] == "taken" and later_ev[2] == actor and worth.get(later_ev[3], 0) >= worth.get(kind_taken, 0):
                    gave_back = True
        if not gave_back:
            for c in keys:
                kept.setdefault(c, []).append(t)
    # census monotone candidates: per (side, kind) did the count ever rise
    rises: dict = {}
    for t in range(n):
        a, b = {}, {}
        for o, k, _at in units[t]:
            a[(o, k)] = a.get((o, k), 0) + 1
        for o, k, _at in units[t + 1]:
            b[(o, k)] = b.get((o, k), 0) + 1
        for key in set(a) | set(b):
            if b.get(key, 0) > a.get(key, 0):
                rises[key] = rises.get(key, 0) + 1
    # the edge distances of the units at the settled positions: per position, per side (the rules' role order), per kind (the engine's order), the sum of the distances of the units to the nearest edge
    roles = _W["roles"]
    edge = np.zeros((len(quiet), len(roles), len(_kinds)), np.int16)
    for j, q in enumerate(quiet):
        for o, k, at in units[q]:
            edge[j, roles.index(o), _kinds.index(k)] += _W["edge"][at]
    census = np.zeros((n + 1, len(roles), len(_kinds)), np.int8)
    for t in range(n + 1):
        for o, k, _at in units[t]:
            census[t, roles.index(o), _kinds.index(k)] += 1
    import cs_engine as CE
    over = CE.state(VARIANT, start, list(moves))[3] is not None                  # the record ends in a position the rules call over
    mover_q = np.array([roles.index(movers[q]) if movers[q] is not None else -1 for q in quiet], np.int8)
    return {"idx": idx, "n": n, "first": movers[0], "alternates": alternates, "edge": edge, "census": census, "over": over, "mover_q": mover_q, "kinds": list(_kinds), "events": ev_steps, "kept": kept, "cut": cut, "violations": violations, "rises": rises,
            "kinds_seen": sorted({k for p in units for _o, k, _a in p})}


# ------------------------------------------------------------------------------------------------------------------------------ the sample

def load_worth() -> dict:
    from catspace.incubator.analyst.discovery import tactical_state as TS
    return TS.human_kind_record()["values"]


def spec_and_vocab() -> "tuple[str, str]":
    return GS.spec_texts(GAME)[VARIANT], json.dumps(GS.vocabulary(GAME))


def game_table(ids: "list | None", limit: int) -> "tuple[list, dict]":
    """The games (dicts of the 78.59 file) to read and, per game number of that file, its settled plies (the quiet rows)."""
    heads = Path(paths.experiment(*HEADS_DIR))
    allg = [json.loads(x) for x in (heads / "games.jsonl").read_text().splitlines() if x.strip()]
    if ids is not None:
        keep = {i: k for k, i in enumerate(ids)}
        games = sorted((g for g in allg if g["id"] in keep), key=lambda g: keep[g["id"]])
    else:
        games = allg
    if limit:
        games = games[:limit]
    return allg, games


def quiet_plies(allg: list) -> dict:
    rows = HD.load_rows(Path(paths.experiment(*HEADS_DIR)) / "rows.npz")
    by: dict = {}
    for g, p in zip(rows["game"].tolist(), rows["ply"].tolist()):
        by.setdefault(g, []).append(p)
    return by


def read_games(games: list, allidx: dict, quiet: dict, threads: int) -> list:
    spec, vocab = spec_and_vocab()
    clauses = rule_clauses(spec)
    fluents = persistent_fluents(clauses)
    worth = load_worth()
    roles = role_words(clauses)
    jobs = [(i, g["moves"], quiet.get(allidx.get(g["id"], -1), g.get("quiet", [])), g.get("start")) for i, g in enumerate(games)]
    from catspace.incubator.analyst.discovery import tactical_tables as TT
    meta = json.loads(Path(paths.experiment("tactical_78_62", "state", "state.json")).read_text())
    edge = dict(zip(meta["places"], TT.edge_distances(meta["places"]).tolist()))
    with Pool(threads, initializer=_init_worker, initargs=(spec, vocab, fluents, worth, roles, edge)) as pool:
        return pool.map(read_game, jobs, chunksize=8)


def summarise(res: list, games: list, say, fit_flags: list) -> dict:
    """Print the derived list with its counts and return {class: [fit events, test events, games with it, all events]} as a dict."""
    spec, _vocab = spec_and_vocab()
    clauses = rule_clauses(spec)
    roles = role_words(clauses)
    values = goal_values(clauses)
    fl = persistent_fluents(clauses)
    say(f"read from the rules: roles {roles}; goal values {values}; fluents that are kept and never made: {fl}; instances: " + ", ".join(f"{f}: {init_instances(clauses, f)}" for f in fl))
    say(f"read from the rules: ways a game ends {[w for w, _r in end_outcomes(roles, values)]}")
    n = len(res)
    say("class fields: taken | the side that took | the kind taken | the kind that took; changed kind | side | kind it was | kind it became; vanished | side that moved | kind gone; went false | side | fluent | other arguments; keeps | side that took | kind taken; end | the role at the highest goal value (empty: none)")
    say(f"games {n}; steps {sum(r['n'] for r in res)}; games where the sides alternate on every step {sum(r['alternates'] for r in res)}; fluent instances that went from false to true after being false (violations of the kept-never-made reading) {sum(r['violations'] for r in res)}")
    cls: dict = {}
    for r, g in zip(res, games):
        for key, steps in list(r["events"].items()) + [(k, v) for k, v in r["kept"].items()]:
            row = cls.setdefault(key, [0, 0, 0, 0])
            row[0 if g["fit"] else 1] += len(steps)
            row[2] += 1
            row[3] += len(steps)
    # the end
    from collections import Counter
    c = Counter()
    for r, g in zip(res, games):
        y = g["y"]
        win = r["first"] if y == 0 else (None if y == 1 else [x for x in roles if x != r["first"]][0])
        c[win] += 1
        key = ("end", win or "")
        row = cls.setdefault(key, [0, 0, 0, 0])
        row[0 if g["fit"] else 1] += 1
        row[2] += 1
        row[3] += 1
    say("THE ENDS (the record's result; a game stopped before the rules' own end is one by resignation or agreement): " + ", ".join(f"{(k or 'no role at the highest value')}: {v}" for k, v in sorted(c.items(), key=lambda kv: str(kv[0]))))
    for depth_name, pick in (("COARSE (type, side)", lambda k: len(k) == 2), ("MID", lambda k: len(k) == 3), ("FINE", lambda k: len(k) >= 4)):
        say(f"CLASSES {depth_name}: class: events in the fit games / in the test games / games holding it")
        for k in sorted((k for k in cls if pick(k)), key=lambda k: -cls[k][3]):
            f, t, gm, _a = cls[k]
            fitted = "fitted" if f >= MIN_EVENTS else f"too rare to fit (under {MIN_EVENTS} events in the fit games): candidate for the endgame fill"
            say(f"  {' | '.join(str(x) for x in k)}: {f} / {t} / {gm}; {fitted}")
    tot = Counter()
    for r in res:
        for key, v in r["rises"].items():
            tot[key] += v
    rose = {k for r in res for k in r["rises"]}
    seen = {(o, k) for r in res for (o, k) in [(s, kk) for s in roles for kk in r["kinds_seen"]]}
    never = sorted(x for x in seen if x not in rose)
    say(f"CANDIDATES found on games, not read from the rules: (side, kind) counts that never rose from one position to the next in any of the {n} games (a count of units that, once lower, was not made higher again here): {never}")
    return cls


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--games", type=int, default=0, help="the first N games (0: all of the sample)")
    ap.add_argument("--state-games", action="store_true", help="the games of the 78.62 state rows (the 800) instead of the whole 78.59 file")
    ap.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    t0 = time.perf_counter()
    out = Path(a.out_dir or paths.experiment("events_78_64", "sample"))
    out.mkdir(parents=True, exist_ok=True)
    lines: list = []

    def say(msg: str) -> None:
        lines.append(f"OBS events: {msg}")
        print(lines[-1], flush=True)

    ids = None
    if a.state_games:
        ids = [g["id"] for g in json.loads(Path(paths.experiment("tactical_78_62", "state", "games.json")).read_text())]
    allg, games = game_table(ids, a.games)
    allidx = {g["id"]: i for i, g in enumerate(allg)}
    quiet = quiet_plies(allg)
    say(f"sample: {len(games)} games of the {len(allg)} of the 78.59 file; kinds' order for the derived reading from the human record of 78.59: {load_worth()}")
    res = read_games(games, allidx, quiet, a.threads)
    cls = summarise(res, games, say, [g["fit"] for g in games])
    import pickle
    (out / "games.pkl").write_bytes(pickle.dumps({"ids": [g["id"] for g in games], "res": res}))
    (out / "events.json").write_text(json.dumps({"ids": [g["id"] for g in games], "games": [{"n": r["n"], "first": r["first"], "events": [[list(k), v] for k, v in r["events"].items()], "kept": [[list(k), v] for k, v in r["kept"].items()],
                                                                                            "cut": [[list(k), v] for k, v in r["cut"].items()]} for r in res]}))
    say(f"wall {time.perf_counter() - t0:.1f} s; written {out / 'events.json'}")
    (out / "events_obs.txt").write_text("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
