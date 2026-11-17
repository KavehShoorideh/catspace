"""PLAN 78.71 B and C: THE CHECKS ON THE GOALS (a measure script; the proposer is subgoal_goals.py, the print and the game's edit path are subgoal_print.py). SYSTEM_SPEC 7 "Choosing a subgoal, and getting there" (the four
graded claims) and 4 (where a concrete answer can be reached it is reached). A measurement in a sandbox on rows that exist: the held-out games of plan 78.70 (the first N in its order; the frame-read rows and the stakes of the kinds
reading of 78.70 are used as they are), one run each, directions only; no player changed, nothing written to rules/learned or data/derived. Game-free: a kind, a role or a place is a word read at run time; the one object that
knows the game's own text is the edit path built by the printing script, handed in as `editor`. No oracle, no engine evaluation, no external model, no assumed worth (the judge is the measured record, as in threat_concrete).
Human games are read only through the existing row builders; no player id is written. Puzzles are not used.

INTERFACE (what this module takes, what it returns, who calls it).
  read(a)     for the first `--games` held-out games: for every settled row (a position) of the game, the goals of the side to move (subgoal_goals.goals), the fix of each applied alone as a what-if edit and the claim read
              again (B), and, from the game as it was played, whether and when the facts of each goal came true and what the side then took (C), and every event where a side takes a unit and keeps it. Saves a pickle of
              compact records (per game: goals with their outcomes, events with the facts that became true, the time each position took). Games are spread over `--procs` processes (a real file, OMP_NUM_THREADS=1 in each).
  grade(a)    from that pickle: A (the goals counted by source, by the fact unmet and its cause, with a fix or not), B (the three outcomes counted and printed; controls), C (the chance of reaching with calibration bins, how
              often a unit was then won, the hindsight of events against the proposer's first few, with the controls: goals drawn at random from the same list, the order by stake alone, ply held fixed by drawing inside the list of the
              same position), the endgame coverage of the sample.
  Called by: the command line only; nothing calls it.

  .venv/bin/python -m catspace.incubator.analyst.discovery.subgoal_measure read  [--games N] [--procs K] [--out-dir D]
  .venv/bin/python -m catspace.incubator.analyst.discovery.subgoal_measure grade [--out-dir D]

THE EVENT of C: a side takes a unit at its move and keeps it: the other side takes nothing at its next move nor at its move after the side's next (two replies): the unit is won for nothing within two replies. THE FACTS THAT
BECAME TRUE: the graph's facts of the goal (a unit of the side bears on the unit taken, a bearer may take, no unit of the other side bears on its place) at the position two plies and four plies before the taking, against at the
position of the taking, the unit followed by its carried identity through the moves played. THE GOAL AMONG THE FIRST FEW: a goal of the side at the earlier position whose target is that unit (and, for the full match, whose
testable unmet facts are among those that became true), in the proposer's first m by the chance of its key times the stake (the chance tabled from the other half of the games), or by the stake alone. WHEN A GOAL'S FACTS CAME TRUE
LATER: at a later position with the side to move (two, four, six, eight plies on) the unmet facts the graph can show all hold at once; a goal with an unmet fact the graph cannot show (a line read, not a relation: `kept`,
`cornered`) is counted apart as untested. No horizon is put in: the cumulative share by plies is printed and the time to the facts is a survival measure with the end of the game as its end.
"""
from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import cs_engine as CE

from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_units as SU
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_measure as TM
from catspace.io import paths

OUT = ("subgoals_78_74",)
SEED = 7171
BACKS = (2, 4)                                                       # plies back from the taking, the same side to move
LATER = (2, 4, 6, 8)                                                 # plies on at which the facts are looked for (the side to move)
TESTABLE = ("bears", "may_take", "unguarded")


def out_dir(p: "str | None" = None) -> Path:
    d = Path(p) if p else Path(paths.experiment(*OUT))
    d.mkdir(parents=True, exist_ok=True)
    return d


def make_say(out: Path, name: str):
    lines: list = []

    def say(m: str) -> None:
        lines.append(f"OBS goals74: {m}")
        print(lines[-1], flush=True)
        (out / name).write_text("\n".join(lines) + "\n")
    return say


# ------------------------------------------------------------------------------------------------------------------------------ the facts at a position of the game played

def facts_at(pf: "SG.PositionFacts", side: str, place: "str | None") -> dict:
    """The testable facts of a goal about the unit on `place` for `side` at a position (None: the unit is gone). `may_take` only where `side` is to move."""
    if place is None:
        return {}
    other = next((u[0] for u in pf.units.values() if u[0] != side), None)
    bearers = pf.bearers(place, side)
    out = {"bears": bool(bearers), "unguarded": not pf.bearers(place, other)}
    if pf.mover == side:
        out["may_take"] = any(place in pf.may.get(u, ()) for u in bearers)
    return out


def place_of(maps_row: dict, uid: str) -> "str | None":
    for p, u in maps_row.items():
        if u == uid:
            return p
    return None


# ------------------------------------------------------------------------------------------------------------------------------ one process

G: dict = {}


def init_worker() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    from catspace.incubator.analyst.discovery import subgoal_print as SP
    rec = TC.Record.load()
    G["rec"] = rec
    G["lines"] = TC.Lines(rec, threads=1)
    G["facts"] = SG.Facts(HD.VARIANT)
    G["editor"] = SP.GameEditor(HD.VARIANT)


def variants_of(g: dict) -> dict:
    """The edit lists B tries for a goal (PLAN 78.74 B): `fix` the one stopper per unmet fact the line showed (plan 78.71's fix), `set` the whole fix set (every stopper named, the goal kind left out, a unit once), `first` the first of the set
    (the single removal that leaves the fewest facts unmet) alone, `last` the last of the set alone (only where the set has two or more removable units: the order is told from its reverse)."""
    fix = [e for e in g["edits"] if e[0] != "move"]
    rem = [e for e in g["fix_set"] if e["removable"]]
    return {"fix": fix, "set": list(g["set_edits"]), "first": [rem[0]["edit"]] if rem else [], "last": [rem[-1]["edit"]] if len(rem) >= 2 else []}


def stage_b(goal_sets: list, texts: list, movers: list, names: dict, rng: random.Random) -> None:
    """B for every goal of every position at once: each variant of the fix applied as a what-if edit (the edit path rejects an illegal result: counted), the claim read again with the line search (a goal about the goal unit: the forced end read again);
    the same read of the position re-spelt with no edit (the goal worked without the fix if that works), and of a position with a control edit (a unit of the owner's taken out that is not a stopper). Fills goal['b']."""
    ed, lines, rec, facts = G["editor"], G["lines"], G["rec"], G["facts"]
    games, queries, plan = [], [], []
    for pi, gs in enumerate(goal_sets):
        text = texts[pi]
        mover = movers[pi]
        f0 = None
        for g in gs:
            vs = variants_of(g)
            if g["source"] in ("unread", "two_moves") or not (vs["fix"] or vs["set"]):
                g["b"] = {"state": "no fix proposed" if not (g["edits"] or g["set_edits"]) else "only the quiet first move"}
                continue
            owner_is_mover = names["roles"][mover] == g["owner"]
            taker_p = 1 if owner_is_mover else 0
            unit_goal = g["source"] == "goal_unit"
            place_ix = names["places"].index(g["target"][2])
            if f0 is None:
                f0 = facts.of(text)
            stoppers = {u[2] for u in g["stoppers"]} | {e[1] for lst in vs.values() for e in lst}
            pool = sorted(p for u, (s, k, p) in f0.units.items() if s == g["owner"] and p not in stoppers and p != g["target"][2] and k != facts.goal_name)
            ctl_edit = [("remove", rng.choice(pool))] if pool else None
            texts_v, why = {}, {}
            seen: dict = {}
            for name, edits in vs.items():
                if not edits:
                    continue
                key = tuple(edits)
                if key in seen:
                    texts_v[name], why[name] = texts_v[seen[key]], why[seen[key]]
                    continue
                seen[key] = name
                texts_v[name], why[name] = ed.apply(text, edits)
            null, _w = ed.build(text, [])
            ctl, _w2 = ed.build(text, ctl_edit) if ctl_edit else (None, "no unit to take out")
            g["b"] = {"state": "read", "why": why.get("fix") or why.get("set"), "variants": {}, "n_set": len(vs["set"]), "n_fix": len(vs["fix"]), "n_rem": sum(1 for e in g["fix_set"] if e["removable"])}
            todo = dict(texts_v)
            todo["null"], todo["control"] = null, ctl
            qmap: dict = {}
            for name, tx in todo.items():
                if tx is None:
                    g["b"]["variants"][name] = {"verdict": "REJECTED"}
                    continue
                if tx in qmap:
                    gi = qmap[tx]
                else:
                    gi = len(games)
                    qmap[tx] = gi
                    games.append((tx, []))
                    if unit_goal:
                        queries.append((gi, 4, taker_p, 1, 0, 0, 0))
                    else:
                        mode = 0 if any(e[0] == "relocate" for e in vs.get(name, [])) else (3 if g["two_moves"] or g["source"] == "two_move_not_ahead" else 0)
                        queries.append((gi, mode, taker_p, 1, place_ix, 0, 0))
                plan.append((g, name, tx, taker_p, place_ix, gi, unit_goal, queries[gi][1]))
    res = lines.ask(games, queries)
    for (g, name, tx, taker_p, place_ix, gi, unit_goal, mode) in plan:
        r = res[gi]
        if unit_goal:
            v = {"verdict": "CONFIRMED" if r["ok"] and r.get("forced") == "proved" else "NO FORCED END"}
            g["b"]["variants"][name] = v
            continue
        best = TC.best_taking(r["reads"], rec) if r["ok"] and r["reads"] else None
        v = {"verdict": "NO TAKING" if best is None else best["verdict"]}
        if best is not None:
            v["net"] = best["net"]
        if name == "set" and v["verdict"] != "CONFIRMED":
            names_line = (best["prefix"] + best["names"]) if best is not None and mode == 3 else (best["names"] if best is not None else [])
            row = {"place": g["target"][2], "place_ix": place_ix, "two_moves": mode == 3, "line": names_line, "verdict": v["verdict"], "stake": g["stake"], "net": v.get("net")}
            side = names["roles"].index(g["taker"])
            item = {"start": tx, "moves": [], "names": names}
            try:
                gb = SG.build_goal(item, side, row, "refuted" if v["verdict"] in ("REFUTED", "EXCHANGE") else "no_taking_legal", G["facts"], G["editor"], tx, g["taker"], g["owner"], names)
            except Exception:
                gb = None
            v["still"] = gb["key"] if gb else "not explained"
            v["still_n"] = gb["n_unmet"] if gb else None
        g["b"]["variants"][name] = v


def compact(g: dict) -> None:
    """Keep the goal's record small (a goal for every unit is many goals): what the grade reads, not the facts' units."""
    g["facts_c"] = [(x["name"], x["status"], x["cause"]["type"] if x["cause"] else None, len(x["cause"]["stoppers"]) if x["cause"] else 0, x["cause"]["basis"] if x["cause"] else None) for x in g["facts"]]
    g["n_fixes"] = len(g["fixes"])
    g["set_c"] = [(e["kind"], e["removable"], e["left"]) for e in g["fix_set"]]
    g["steps_c"] = [(s["step"], s["moves"]) for s in g["steps"]]
    g["n_stoppers"] = len(g["stoppers"])
    for k in ("row", "plies", "facts", "fixes", "fix_set", "steps", "stoppers", "edits", "set_edits", "line"):
        g.pop(k, None)


def do_game(payload: dict) -> dict:
    facts, lines, rec, ed = G["facts"], G["lines"], G["rec"], G["editor"]
    t_game = time.perf_counter()
    names, moves, gid = payload["names"], payload["moves"], payload["gid"]
    roles = names["roles"]
    text0 = CE.state(facts.variant, None, [])[0]
    plies = facts.replay(text0, moves)
    maps, texts = facts.trace
    n = len(moves)
    pf_cache: dict = {}

    def pf_at(q: int):
        if q not in pf_cache:
            pf_cache[q] = facts.of(texts[q])
        return pf_cache[q]
    items = [{"start": None, "moves": moves[:ply], "raw": raw, "names": names, "list": lst} for (ply, raw, lst, _st) in payload["positions"]]
    t0 = time.perf_counter()
    conc = TC.concrete_lists(items, lines) if items else []
    t_conc = time.perf_counter() - t0
    t0 = time.perf_counter()
    stake_fns = [SU.stake_closure(st, payload["kind_cols"]) for (_p, _r, _l, st) in payload["positions"]]
    info = SU.every_unit(items, [[it["raw"]["mover"]] for it in items], conc, facts, lines, stake_fns) if items else {"units": 0, "reads": 0}
    t_every = time.perf_counter() - t0
    out_pos, goal_sets, tx_list, movers = [], [], [], []
    for (ply, raw, lst, _st), item, c in zip(payload["positions"], items, conc):
        mover = raw["mover"]
        t1 = time.perf_counter()
        gl = SG.goals(item, mover, c, facts, ed)
        wk = SG.works(item, mover, c)
        t_goal = time.perf_counter() - t1
        goal_sets.append(gl)
        tx_list.append(texts[ply])
        movers.append(mover)
        n_other = sum(1 for u in maps[ply].values() if u.split(":")[1] != roles[mover])
        out_pos.append({"ply": ply, "mover": mover, "n_units": len(raw["board"]), "n_other": n_other, "works": len(wk), "t_goal": t_goal, "goals": gl, "n_list_rows": len(c["mover"]) + len(c["other"]) + len(c["exchange"]) + len(c["dropped_rows"]) - sum(1 for k in ("mover", "other", "exchange", "dropped_rows") for r in c[k] if r.get("every"))})
    rng = random.Random(f"{SEED}:{gid}")
    t2 = time.perf_counter()
    if payload["do_b"]:
        stage_b(goal_sets, tx_list, movers, names, rng)
    else:
        for gs in goal_sets:
            for g in gs:
                g["b"] = {"state": "not sampled"}
    t_b = time.perf_counter() - t2
    # ---- C: the facts of each goal at the later positions of the game as it was played
    for pos in out_pos:
        q, mover = pos["ply"], pos["mover"]
        side = roles[mover]
        for g in pos["goals"]:
            tuid = maps[q].get(g["target"][2])
            g["uid"] = tuid
            testable = [f["name"] for f in g["facts"] if f["status"] in ("unmet", "unread") and f["name"] in TESTABLE]
            untested = [f["name"] for f in g["facts"] if f["status"] in ("unmet", "unread") and f["name"] not in TESTABLE]
            g["testable"], g["untested"] = testable, untested
            reach, won_T, won_any = {}, {}, {}
            for k in LATER:
                q2 = q + k
                if q2 >= n:
                    continue
                tp = place_of(maps[q2], tuid)
                if tp is None:
                    reach[k] = None                                   # the unit is gone by then
                    continue
                fa = facts_at(pf_at(q2), side, tp)
                reach[k] = all(fa.get(x, False) for x in testable)
                mv = plies[q2]
                won_T[k] = bool(mv["took"] and mv["took"][0] == tuid)
                won_any[k] = bool(mv["took"])
            g["reach"], g["won_T"], g["won_any"] = reach, won_T, won_any
            # the unit taken by this side within the next eight plies at all (what happens whatever the facts)
            g["taken_by_side"] = any(plies[x]["side"] == side and plies[x]["took"] and plies[x]["took"][0] == tuid for x in range(q, min(n, q + 8)))
            compact(g)
    # ---- the events: a side takes a unit and keeps it
    events = []
    for p, mv in enumerate(plies):
        if mv["took"] is None:
            continue
        s = mv["side"]
        later = [plies[x] for x in (p + 1, p + 3) if x < n]
        kept = all(x["took"] is None for x in later)
        e = {"p": p, "side": s, "uid": mv["took"][0], "kept": kept, "became": {}}
        for k in BACKS:
            q = p - k
            if q < 0:
                continue
            tq, tp_ = place_of(maps[q], mv["took"][0]), place_of(maps[p], mv["took"][0])
            fq, fp = facts_at(pf_at(q), s, tq), facts_at(pf_at(p), s, tp_)
            e["became"][k] = {"before": fq, "at": fp}
        events.append(e)
    own = [plies[x]["took"] is not None for x in range(n)]
    return {"gid": gid, "n": n, "positions": out_pos, "events": events, "t_conc": t_conc, "t_every": t_every, "every_units": info["units"], "every_reads": info["reads"], "t_b": t_b, "wall": time.perf_counter() - t_game, "own_took": own,
            "edit": dict(ed.count), "edit_reasons": dict(ed.reasons), "do_b": payload["do_b"]}


# ------------------------------------------------------------------------------------------------------------------------------ read

def load_sample(games: int) -> tuple:
    from catspace.incubator.analyst.discovery import threat_print as TP
    from catspace.incubator.analyst.discovery import two_readings as TR
    z = __import__("pickle").loads((TP.paths_dir_78_70() / "raw.pkl").read_bytes())
    lists = TM.build_lists(z)
    d = TR.load(lambda m: None)
    return z, lists, d["allg"]


def cmd_read(a) -> int:
    t0 = time.perf_counter()
    out = out_dir(a.out_dir)
    say = make_say(out, "read_obs.txt")
    z, lists, allg = load_sample(a.games)
    games = z["games"][:a.games]
    by: dict = {}
    for j, g in enumerate(z["g"]):
        by.setdefault(int(g), []).append(j)
    payloads = []
    for gx, g in enumerate(games):
        gid = z["sample_ids"][z["games"].index(g)]
        js = sorted(by.get(int(g), []), key=lambda j: int(z["ply"][j]))
        payloads.append({"gid": gid, "g": int(g), "moves": allg[gid]["moves"], "names": z["names"], "kind_cols": z["kind_cols"], "do_b": gx < a.b_games,
                         "positions": [(int(z["ply"][j]), z["rows"][j], lists[j], z["stake"][j]) for j in js]})
    say(f"sample: the first {len(games)} held-out games of the 78.70 sample (the games the 78.59 split did not fit), their settled rows: {sum(len(p['positions']) for p in payloads)} positions; judge: the measured record; {a.procs} processes; B (the fix read again) on the first {min(a.b_games, len(games))} games")
    res = []
    ctx = get_context("spawn")
    with ctx.Pool(a.procs, initializer=init_worker) as pool:
        for i, r in enumerate(pool.imap(do_game, payloads), 1):
            res.append(r)
            if i % 5 == 0 or i == len(payloads):
                say(f"progress: {i} of {len(payloads)} games, {sum(len(x['positions']) for x in res)} positions, wall {time.perf_counter() - t0:.0f} s")
    pk = out / "goals.pkl"
    pk.write_bytes(pickle.dumps({"res": res, "names": z["names"], "games": [p["gid"] for p in payloads]}))
    say(f"done: {len(res)} games; saved {pk}; wall {time.perf_counter() - t0:.0f} s")
    return 0


# ------------------------------------------------------------------------------------------------------------------------------ grade

def pct(x: float) -> str:
    return f"{100.0 * x:.1f}"


def hyper_hit(L: int, h: int, m: int) -> float:
    """The chance that at least one of m goals drawn at random without replacement from L has the target, h of the L having it."""
    if h <= 0 or L <= 0:
        return 0.0
    m = min(m, L)
    if L - h < m:
        return 1.0
    return 1.0 - math.comb(L - h, m) / math.comb(L, m)


def table_of(goals: list, fold: int, folds: int = 2) -> "tuple[dict, dict, tuple]":
    """The chance of reaching within four plies (the testable facts all hold two own moves on) by the goal's key, from the games of the OTHER fold (cross-fitted): returns {key: (n, reached)}, {facts: (n, reached)}, (n, reached)."""
    bykey, byfacts, tot = {}, {}, [0, 0]
    for (gi, _p, g) in goals:
        if gi % folds == fold:
            continue
        r = g["reach"].get(4)
        if r is None or g["untested"]:
            continue
        for tab, k in ((bykey, g["key"]), (byfacts, "|".join(sorted(g["testable"])))):
            x = tab.setdefault(k, [0, 0])
            x[0] += 1
            x[1] += 1 if r else 0
        tot[0] += 1
        tot[1] += 1 if r else 0
    return bykey, byfacts, tuple(tot)


def chance_of(g: dict, tabs: tuple) -> float:
    bykey, byfacts, tot = tabs
    for tab, k in ((bykey, g["key"]), (byfacts, "|".join(sorted(g["testable"])))):
        x = tab.get(k)
        if x and x[0] >= 20:
            return (x[1] + 1) / (x[0] + 2)
    return (tot[1] + 1) / (tot[0] + 2)


def top_hit(keys: list, flags: list, m: int) -> float:
    """The chance that a goal with a flag (about the unit then won) is among the first m when the goals are ordered by `keys` (smaller first) and goals with equal keys are in a uniformly random order among themselves (a tie is not read as luck of the
    sort: the expectation over it). One flagged goal at most is the case here (a unit has one goal)."""
    best = 0.0
    for i, fl in enumerate(flags):
        if not fl:
            continue
        b = sum(1 for k in keys if k < keys[i])
        s = sum(1 for k in keys if k == keys[i])
        r = m - b
        best = max(best, 0.0 if r <= 0 else min(1.0, r / s))
    return best


def cmd_grade(a) -> int:
    out = out_dir(a.out_dir)
    say = make_say(out, "measure_obs.txt")
    z = pickle.loads((out / "goals.pkl").read_bytes())
    res, names = z["res"], z["names"]
    ng = len(res)
    positions = [(gi, r["gid"], pos) for gi, r in enumerate(res) for pos in r["positions"]]
    goals = [(gi, pos, g) for gi, _gid, pos in positions for g in pos["goals"]]
    say(f"sample: {ng} held-out games, {len(positions)} positions (settled rows), goals of the side to move {len(goals)}; one run, directions only")
    # ---- coverage
    bins = [(0, 5), (6, 12), (13, 20), (21, 99)]
    cov = [sum(1 for _g, _i, p in positions if lo <= p["n_units"] <= hi) for lo, hi in bins]
    say("ENDGAME COVERAGE of the sample (positions by number of units on the board; human games end early): " + "; ".join(f"{lo}-{hi if hi < 99 else 'more'}: {c} positions ({pct(c / max(1, len(positions)))} pct)" for (lo, hi), c in zip(bins, cov)))
    # ---- A
    bysrc: dict = {}
    for _gi, _p, g in goals:
        bysrc[g["source"]] = bysrc.get(g["source"], 0) + 1
    npos = max(1, len(positions))
    say("A GOALS by source: " + "; ".join(f"{k} {v}" for k, v in sorted(bysrc.items())) + f"; threats already working (not goals) {sum(p['works'] for _g, _i, p in positions)}; positions with at least one goal {sum(1 for _g, _i, p in positions if p['goals'])} of {len(positions)}; goals per position {len(goals) / npos:.1f}")
    say(f"A THE LINE READ FOR UNITS THE THREAT LIST HAD NO ROW FOR: units read {sum(r['every_units'] for r in res)}, line queries {sum(r['every_reads'] for r in res)}; rows of the threat list per position {sum(p['n_list_rows'] for _g, _i, p in positions) / npos:.1f}; units of the other side per position {sum(p['n_other'] for _g, _i, p in positions) / npos:.1f}")
    un: dict = {}
    basis: dict = {}
    for _gi, _p, g in goals:
        for (nm, st, ct, ns, bs) in g["facts_c"]:
            if st in ("unmet", "unread"):
                k = f"{nm}:{ct or '-'}"
                un[k] = un.get(k, 0) + 1
                basis[bs or "-"] = basis.get(bs or "-", 0) + 1
    say("A UNMET FACTS by fact and cause (a goal can have several): " + "; ".join(f"{k} {v}" for k, v in sorted(un.items(), key=lambda x: -x[1])))
    say("A UNMET FACTS by the basis of the claim (line: a line the engine read said so; graph: a relation of the position, no line read for it): " + "; ".join(f"{k} {v}" for k, v in sorted(basis.items())))
    nf = sum(1 for _gi, _p, g in goals if g["n_fixes"])
    say(f"A FIXES: goals with at least one fix proposed {nf}, with none {len(goals) - nf}; goals by number of unmet facts: " + "; ".join(f"{k} {sum(1 for _g, _p, g in goals if g['n_unmet'] == k)}" for k in range(0, 9)) + f"; with more {sum(1 for _g, _p, g in goals if g['n_unmet'] >= 9)}")
    sz = {}
    for _gi, _p, g in goals:
        k = min(5, g["n_stoppers"])
        sz[k] = sz.get(k, 0) + 1
    say("A THE FIX AS A SET: goals by number of units that stop it (5 = five or more): " + "; ".join(f"{k} {v}" for k, v in sorted(sz.items())) + f"; goals with a stopper of the goal kind (cannot be taken out) {sum(1 for _g, _p, g in goals if any(not e[1] for e in g['set_c'] if e[0] == 'remove'))}; goals whose set has no unit that can be taken out or brought {sum(1 for _g, _p, g in goals if not any(e[1] for e in g['set_c']))}")
    stp: dict = {}
    for _gi, _p, g in goals:
        for (nm, mv) in g["steps_c"]:
            stp[f"{nm}:{mv if mv is not None else 'more'}"] = stp.get(f"{nm}:{mv if mv is not None else 'more'}", 0) + 1
    say("A TWO STEPS where nothing of ours bears (the step and by how many moves of the graph; 'more' = within two steps of the place graph, not within two moves of the rules): " + "; ".join(f"{k} {v}" for k, v in sorted(stp.items())))
    tg = sorted(p["t_goal"] for _g, _i, p in positions)
    pw = sorted(r["wall"] / max(1, len(r["positions"])) for r in res)
    say(f"A TIME per position (one thread): the goals median {tg[len(tg) // 2]:.3f} s, mean {sum(tg) / len(tg):.3f} s, slowest {tg[-1]:.2f} s; the concrete read per position {sum(r['t_conc'] for r in res) / npos:.3f} s; the line read of every other unit per position {sum(r['t_every'] for r in res) / npos:.3f} s; the B reads per position {sum(r['t_b'] for r in res if r['do_b']) / max(1, sum(len(r['positions']) for r in res if r['do_b'])):.3f} s (games with B only); wall per position, median over games {pw[len(pw) // 2]:.3f} s, mean {sum(r['wall'] for r in res) / npos:.3f} s")
    gu = [g for _gi, _p, g in goals if g["source"] == "goal_unit"]
    say(f"A THE GOAL UNIT as a target: {len(gu)} goals; unmet facts per such goal {sum(g['n_unmet'] for g in gu) / max(1, len(gu)):.1f}; places it may move to that we do not cover per goal {sum(sum(1 for f in g['facts_c'] if f[0] == 'covered' and f[1] == 'unmet') for g in gu) / max(1, len(gu)):.2f}; goal unit borne on in {sum(1 for g in gu if any(f[0] == 'borne_on' and f[1] == 'met' for f in g['facts_c']))}")
    # ---- B
    bgoals = [(gi, p, g) for gi, p, g in goals if g["b"]["state"] != "not sampled"]
    say(f"B SAMPLE: the fix is read again on the first {sum(1 for r in res if r['do_b'])} games: {len(bgoals)} goals of {len(goals)}")
    outc: dict = {}
    bysrc_b: dict = {}
    ctl = [0, 0]
    still: dict = {}
    for _gi, _p, g in bgoals:
        b = g["b"]
        if b["state"] != "read":
            outc["no edit: " + b["state"]] = outc.get("no edit: " + b["state"], 0) + 1
            continue
        v = b["variants"]
        fx, nu = v.get("fix", {"verdict": "NO EDIT"})["verdict"], v["null"]["verdict"]
        co = v["control"]["verdict"]
        if fx == "REJECTED":
            o = "the fix edit was rejected (no legal position)"
        elif fx == "NO EDIT":
            o = "no fix edit (the old fix had no removal)"
        elif nu == "CONFIRMED":
            o = "works without the fix (the fix was not needed)"
        elif fx == "CONFIRMED":
            o = "works with the fix (diagnosis complete)"
        else:
            o = "still fails with the fix (something else also stops it)"
        outc[o] = outc.get(o, 0) + 1
        bysrc_b.setdefault(g["source"], {})
        bysrc_b[g["source"]][o] = bysrc_b[g["source"]].get(o, 0) + 1
    rd = [g for _gi, _p, g in bgoals if g["b"]["state"] == "read"]
    nnu = sum(1 for g in rd if g["b"]["variants"]["null"]["verdict"] == "CONFIRMED")
    say(f"B THE POSITION RE-SPELT WITH NO EDIT (the goal works without the fix): {nnu} of {len(rd)} goals read")
    say("B THE OLD FIX (one stopper per unmet fact, as plan 78.71), the claim read again, outcomes: " + "; ".join(f"{k}: {v}" for k, v in sorted(outc.items())))
    for s, d in sorted(bysrc_b.items()):
        say(f"B old fix by source [{s}]: " + "; ".join(f"{k}: {v}" for k, v in sorted(d.items())))

    def tally(name: str, sel: list) -> str:
        c = {"CONFIRMED": 0, "REJECTED": 0, "other": 0, "n": 0}
        for g in sel:
            v = g["b"]["variants"].get(name)
            if v is None:
                continue
            c["n"] += 1
            c["CONFIRMED" if v["verdict"] == "CONFIRMED" else ("REJECTED" if v["verdict"] == "REJECTED" else "other")] += 1
        return f"works {c['CONFIRMED']} of {c['n']} (rejected edits {c['REJECTED']}, still fails {c['other']})"
    legal = [g for g in rd if all(g["b"]["variants"].get(x, {"verdict": "x"})["verdict"] != "REJECTED" for x in ("set", "control"))]
    say(f"B THE FIX AS A SET (every stopper named: taken out, or the unit brought; units of the goal kind left out), all goals read: {tally('set', rd)}")
    say(f"B THE SET, goals where the set and the control were both legal ({len(legal)}): the set {tally('set', legal)}; CONTROL (a unit of the owner's that is not a stopper taken out instead): {tally('control', legal)}; the old fix alone: {tally('fix', legal)}")
    big = [g for g in rd if g["b"]["n_rem"] >= 2]
    say(f"B THE SET where it has two or more units ({len(big)} goals): the set {tally('set', big)}; the old fix {tally('fix', big)}; the first of the set alone {tally('first', big)}; the last of the set alone {tally('last', big)}")
    both = [g for g in big if g["b"]["variants"].get("first", {"verdict": "REJECTED"})["verdict"] != "REJECTED" and g["b"]["variants"].get("last", {"verdict": "REJECTED"})["verdict"] != "REJECTED"]
    f_only = sum(1 for g in both if g["b"]["variants"]["first"]["verdict"] == "CONFIRMED" and g["b"]["variants"]["last"]["verdict"] != "CONFIRMED")
    l_only = sum(1 for g in both if g["b"]["variants"]["last"]["verdict"] == "CONFIRMED" and g["b"]["variants"]["first"]["verdict"] != "CONFIRMED")
    say(f"B THE ORDER OF THE SET (both single removals legal, {len(both)} goals): the first of the set works alone and the last does not in {f_only}; the last works and the first does not in {l_only}")
    for s in sorted({g["source"] for g in rd}):
        sel = [g for g in rd if g["source"] == s]
        sl = [g for g in sel if g in legal]
        say(f"B by source [{s}] ({len(sel)} goals): the set {tally('set', sel)}; the old fix {tally('fix', sel)}; the control {tally('control', sl)} (legal pairs {len(sl)})")
    wf = [g for g in rd if g["b"]["variants"].get("set", {"verdict": "x"})["verdict"] == "CONFIRMED" and g["b"]["variants"]["null"]["verdict"] != "CONFIRMED" and g["source"] != "goal_unit"]
    pos_s = sum(1 for g in wf if g["stake"] is not None and g["stake"] > 0)
    say(f"B THE FIRST CLAIM (worth it) where the set works: the target's stake is above zero in {pos_s} of {len(wf)} goals ({sum(1 for g in wf if g['stake'] is None)} have no stake: a unit of the goal kind or outside the head); the net of the line after the fix is above the record's resolution in all of them by the verdict")
    for g in rd:
        v = g["b"]["variants"].get("set", {})
        if v.get("verdict") not in ("CONFIRMED", "REJECTED", None) and "still" in v:
            still[v["still"]] = still.get(v["still"], 0) + 1
    say("B STILL FAILS with the set, what else stops it (the key of the goal read again on the edited position): " + "; ".join(f"{k or '(no named cause)'}: {v}" for k, v in sorted(still.items(), key=lambda x: -x[1])[:12]))
    er: dict = {}
    for r in res:
        for k, v in r["edit_reasons"].items():
            er[k] = er.get(k, 0) + v
    ea = sum(r["edit"]["applied"] for r in res)
    eb = sum(r["edit"]["rejected"] for r in res)
    say(f"B THE EDIT PATH: edits applied {ea}, rejected {eb}; reasons: " + "; ".join(f"{k}: {v}" for k, v in sorted(er.items(), key=lambda x: -x[1])[:8]))
    # ---- C: reach
    tabs = [table_of(goals, f) for f in (0, 1)]
    pooled = []
    for gi, p, g in goals:
        r = g["reach"].get(4)
        if r is None or g["untested"]:
            continue
        pooled.append((chance_of(g, tabs[gi % 2]), 1 if r else 0, g))
    nt = sum(1 for _gi, _p, g in goals if g["untested"])
    say(f"C GOALS whose unmet facts the graph can show: {len(goals) - nt}; goals with an unmet fact the graph cannot show (a line read, the goal unit's facts): {nt} (counted apart, not in the chance of reaching)")
    for k in LATER:
        have = [g for _gi, _p, g in goals if not g["untested"] and g["reach"].get(k) is not None]
        say(f"C REACH by {k} plies on (cumulative share of goals, the unit still standing then): {sum(1 for g in have if g['reach'][k])} of {len(have)} ({pct(sum(1 for g in have if g['reach'][k]) / max(1, len(have)))} pct)")
    say("C THE TABLE of the chance of reaching within four plies, fitted on the other half of the games by the key of the goal (smoothed; tabled keys with at least 20 goals): " + "; ".join(f"{k or '-'}: {v[1]}/{v[0]}" for k, v in sorted(tabs[0][0].items(), key=lambda x: -x[1][0])[:8] if v[0] >= 20))
    edges = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        sel = [(pr, r) for pr, r, _g in pooled if lo <= pr < hi]
        if sel:
            say(f"C CALIBRATION predicted chance in [{lo:.2f}, {hi:.2f}): {len(sel)} goals, mean predicted {sum(x for x, _ in sel) / len(sel):.3f}, reached within four plies {sum(r for _, r in sel) / len(sel):.3f}")
    reached = [(g, k) for _gi, _p, g in goals if not g["untested"] for k in [min((k for k in LATER if g["reach"].get(k)), default=None)] if k is not None]
    notr = [g for _gi, _p, g in goals if not g["untested"] and any(g["reach"].get(k) is not None for k in LATER) and not any(g["reach"].get(k) for k in LATER)]
    wT = sum(1 for g, k in reached if g["won_T"].get(k))
    wA = sum(1 for g, k in reached if g["won_any"].get(k))
    base_all = [x for r in res for x in r["own_took"]]
    say(f"C THE UNIT WON: at the first later position where all the unmet facts the graph can show held ({len(reached)} goals) the side's own move took the target in {wT} ({pct(wT / max(1, len(reached)))} pct) and took some unit in {wA} ({pct(wA / max(1, len(reached)))} pct); base rate of any move of the game taking a unit {pct(sum(base_all) / max(1, len(base_all)))} pct; goals whose facts did not come true within eight plies (the unit still standing): {len(notr)}, of which the side took the target within eight plies of the position anyway {sum(1 for g in notr if g['taken_by_side'])}")
    # ---- C: hindsight
    by_pos = {(gi, p["ply"]): p for gi, _gid, p in positions}
    ev_tot = ev_kept = 0
    elig = {k: 0 for k in BACKS}
    avail = {k: 0 for k in BACKS}
    cover = {k: 0 for k in BACKS}
    names_r = ("P", "S", "N", "R")                                  # the proposer (unmet count, chance, stake), stake alone, unmet count alone, random from the list
    h = {k: {"n_list": 0, "full": 0.0, "fullR": 0.0, "U3": 0.0, "U1": 0.0, **{f"{r}{m}": 0.0 for r in names_r for m in (1, 3)}} for k in BACKS}
    sizeL = {k: [] for k in BACKS}
    for gi, r in enumerate(res):
        tabs_g = tabs[gi % 2]
        for e in r["events"]:
            ev_tot += 1
            if not e["kept"]:
                continue
            ev_kept += 1
            for k in BACKS:
                bk = e["became"].get(k)
                if bk is None:
                    continue
                q = e["p"] - k
                pos = by_pos.get((gi, q))
                became = {f for f in TESTABLE if f in bk["before"] and f in bk["at"] and not bk["before"][f] and bk["at"][f]}
                if not became:
                    continue
                elig[k] += 1
                if pos is None:
                    continue
                avail[k] += 1
                gl = pos["goals"]
                if not gl:
                    continue
                h[k]["n_list"] += 1
                sizeL[k].append(len(gl))
                top = max((g["stake"] for g in gl if g["stake"] is not None), default=1.0)
                stk = [max(0.0, g["stake"] if g["stake"] is not None else top) for g in gl]
                m1 = [g["uid"] == e["uid"] for g in gl]
                full = [g["uid"] == e["uid"] and set(g["testable"]) <= became and not g["untested"] for g in gl]
                cover[k] += 1 if any(m1) else 0
                keyP = [(g["n_unmet"], -chance_of(g, tabs_g), -stk[i]) for i, g in enumerate(gl)]
                keyS = [-x for x in stk]
                keyN = [g["n_unmet"] for g in gl]
                keyR = [0] * len(gl)
                for m in (1, 3):
                    h[k][f"P{m}"] += top_hit(keyP, m1, m)
                    h[k][f"S{m}"] += top_hit(keyS, m1, m)
                    h[k][f"N{m}"] += top_hit(keyN, m1, m)
                    h[k][f"R{m}"] += top_hit(keyR, m1, m)
                h[k]["full"] += top_hit(keyP, full, 3)
                h[k]["fullR"] += top_hit(keyR, full, 3)
                nu = max(pos["n_other"], len(gl))
                h[k]["U3"] += hyper_hit(nu, 1, 3)                   # three units of the other side drawn at random: a goal about the unit then won among them
                h[k]["U1"] += 1.0 / max(1, nu)
    say(f"C EVENTS (a side takes a unit and keeps it: nothing taken back in its two replies): {ev_kept} of {ev_tot} takings in the sample")
    for k in BACKS:
        x = h[k]
        n = max(1, x["n_list"])
        av = max(1, avail[k])
        say(f"C HINDSIGHT {k} plies back: events where some testable fact became true in between {elig[k]}; of them with the earlier position in the sample {avail[k]}; with at least one goal listed {x['n_list']} (mean goals listed {sum(sizeL[k]) / max(1, len(sizeL[k])):.1f}); a goal about the unit then won was listed in {cover[k]} ({pct(cover[k] / n)} pct of those with a list)")
        say(f"C HINDSIGHT {k} plies back, RECALL of the unit then won over the {avail[k]} events with the earlier position in the sample (an event with no list is a miss), first 1 / first 3: "
            f"PROPOSER (fewest unmet facts, then chance of the key, then stake) {pct(x['P1'] / av)} / {pct(x['P3'] / av)}; CONTROLS (exact; goals with equal keys in random order): goals drawn at random from the same list {pct(x['R1'] / av)} / {pct(x['R3'] / av)}; "
            f"units of the other side drawn at random {pct(x['U1'] / av)} / {pct(x['U3'] / av)}; stake alone {pct(x['S1'] / av)} / {pct(x['S3'] / av)}; unmet count alone {pct(x['N1'] / av)} / {pct(x['N3'] / av)}")
        say(f"C HINDSIGHT {k} plies back, over the {x['n_list']} events with a list (as in plan 78.71), first 1 / first 3: proposer {pct(x['P1'] / n)} / {pct(x['P3'] / n)}; random from the list {pct(x['R1'] / n)} / {pct(x['R3'] / n)}; stake alone {pct(x['S1'] / n)} / {pct(x['S3'] / n)}; "
            f"unmet count alone {pct(x['N1'] / n)} / {pct(x['N3'] / n)}; units drawn at random {pct(x['U1'] / n)} / {pct(x['U3'] / n)}; the full match (the goal's unmet facts are the ones that became true), proposer first 3: {pct(x['full'] / n)} pct against random {pct(x['fullR'] / n)} pct")
    return 0


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["read", "grade"])
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--b-games", type=int, default=10 ** 9, help="B (the fix read again) on the first N games only")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    return {"read": cmd_read, "grade": cmd_grade}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
