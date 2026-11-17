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
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_measure as TM
from catspace.io import paths

OUT = ("subgoals_78_71",)
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
        lines.append(f"OBS goals71: {m}")
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


def stage_b(items: list, goal_sets: list, texts: list, movers: list, names: dict, rng: random.Random) -> None:
    """B for every goal of every position at once: the fix applied alone as a what-if edit (the edit path rejects an illegal result: counted), the claim read again with the line search; the same read of the position re-spelt
    with no edit (the fix was not needed if that works), and of a position with a control edit (a unit of the owner's taken out that is not a stopper). Fills goal['b']."""
    ed, lines, rec, facts = G["editor"], G["lines"], G["rec"], G["facts"]
    games, queries, plan = [], [], []
    for pi, gs in enumerate(goal_sets):
        text = texts[pi]
        mover = movers[pi]
        for g in gs:
            g["b"] = {"state": "no edit"}
            removal = [e for e in g["edits"] if e[0] != "move"]
            if not removal or g["source"] in ("unread", "two_moves"):
                g["b"]["state"] = "no fix proposed" if not g["edits"] else "only the quiet first move"
                continue
            owner_is_mover = names["roles"][mover] == g["owner"]
            taker_p = 1 if owner_is_mover else 0
            place_ix = names["places"].index(g["target"][2])
            mode = 3 if g["two_moves"] or g["source"] == "two_move_not_ahead" else 0
            fixed, why = ed.apply(text, removal)
            null, _w = ed.build(text, [])
            f0 = facts.of(text)
            stoppers = {e[1] for e in removal}
            pool = sorted(p for u, (s, k, p) in f0.units.items() if s == g["owner"] and p not in stoppers and p != g["target"][2] and k != facts.goal_name)
            ctl_edit = [("remove", rng.choice(pool))] if pool else None
            ctl, _w2 = ed.build(text, ctl_edit) if ctl_edit else (None, "no unit to take out")
            every = list(removal)                                       # every stopper the unmet facts name (the fix names the one the line showed): the diagnosis may be right and the fix too small
            for f in g["facts"]:
                if f["status"] == "unmet" and f["cause"] and f["cause"]["type"] in ("guard", "blocked", "goal_unit_attacked"):
                    for u in f["cause"]["stoppers"]:
                        if ("remove", u[2]) not in every:
                            every.append(("remove", u[2]))
            alltx = ed.build(text, every)[0] if len(every) > len(removal) else None
            g["b"] = {"state": "read", "why": why, "variants": {}, "n_every": len(every), "n_fix": len(removal)}
            for name, tx in (("fix", fixed), ("null", null), ("control", ctl), ("all", alltx)):
                if name == "all" and alltx is None:
                    continue
                if tx is None:
                    g["b"]["variants"][name] = {"verdict": "REJECTED"}
                    continue
                gi = len(games)
                games.append((tx, []))
                queries.append((gi, mode, taker_p, 1, place_ix, 0, 0))
                plan.append((g, name, tx, taker_p, place_ix, mode, pi))
    res = lines.ask(games, queries)
    for (g, name, tx, taker_p, place_ix, mode, pi), r in zip(plan, res):
        best = TC.best_taking(r["reads"], rec) if r["ok"] and r["reads"] else None
        v = {"verdict": "NO TAKING" if best is None else best["verdict"]}
        if best is not None:
            v["net"] = best["net"]
            v["line"] = (best["prefix"] + best["names"]) if mode == 3 else best["names"]
        if name == "fix" and v["verdict"] != "CONFIRMED":
            row = {"place": g["target"][2], "place_ix": place_ix, "two_moves": mode == 3, "line": v.get("line", []), "verdict": v["verdict"], "stake": g["stake"], "net": v.get("net")}
            side = names["roles"].index(g["taker"])
            item = {"start": tx, "moves": [], "names": names}
            try:
                gb = SG.build_goal(item, side, row, "refuted" if v["verdict"] in ("REFUTED", "EXCHANGE") else "no_taking_legal", facts, G["editor"], tx, g["taker"], g["owner"], names)
            except Exception:
                gb = None
            v["still"] = gb["key"] if gb else "not explained"
        g["b"]["variants"][name] = v


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
    items = [{"start": None, "moves": moves[:ply], "raw": raw, "names": names, "list": lst} for (ply, raw, lst) in payload["positions"]]
    t0 = time.perf_counter()
    conc = TC.concrete_lists(items, lines) if items else []
    t_conc = time.perf_counter() - t0
    out_pos, goal_sets, tx_list, movers = [], [], [], []
    for (ply, raw, lst), item, c in zip(payload["positions"], items, conc):
        mover = raw["mover"]
        t1 = time.perf_counter()
        gl = SG.goals(item, mover, c, facts, ed)
        wk = SG.works(item, mover, c)
        t_goal = time.perf_counter() - t1
        goal_sets.append(gl)
        tx_list.append(texts[ply])
        movers.append(mover)
        n_other = sum(1 for u in maps[ply].values() if u.split(":")[1] != roles[mover])
        out_pos.append({"ply": ply, "mover": mover, "n_units": len(raw["board"]), "n_other": n_other, "works": len(wk), "t_goal": t_goal, "goals": gl})
    rng = random.Random(f"{SEED}:{gid}")
    t2 = time.perf_counter()
    stage_b([it for it in items], goal_sets, tx_list, movers, names, rng)
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
            for k in list(g.keys()):
                if k in ("row", "plies"):
                    del g[k]                                           # keep the record small: the print has its own run
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
    return {"gid": gid, "n": n, "positions": out_pos, "events": events, "t_conc": t_conc, "t_b": t_b, "wall": time.perf_counter() - t_game, "own_took": own, "edit": dict(ed.count), "edit_reasons": dict(ed.reasons)}


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
    for g in games:
        gid = z["sample_ids"][z["games"].index(g)]
        js = sorted(by.get(int(g), []), key=lambda j: int(z["ply"][j]))
        payloads.append({"gid": gid, "g": int(g), "moves": allg[gid]["moves"], "names": z["names"], "positions": [(int(z["ply"][j]), z["rows"][j], lists[j]) for j in js]})
    say(f"sample: the first {len(games)} held-out games of the 78.70 sample (the games the 78.59 split did not fit), their settled rows: {sum(len(p['positions']) for p in payloads)} positions; judge: the measured record; {a.procs} processes")
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
    say("A GOALS by source: " + "; ".join(f"{k} {v}" for k, v in sorted(bysrc.items())) + f"; threats already working (not goals) {sum(p['works'] for _g, _i, p in positions)}; positions with at least one goal {sum(1 for _g, _i, p in positions if p['goals'])} of {len(positions)}")
    un: dict = {}
    for _gi, _p, g in goals:
        for f in g["facts"]:
            if f["status"] in ("unmet", "unread"):
                k = f"{f['name']}:{f['cause']['type'] if f['cause'] else '-'}"
                un[k] = un.get(k, 0) + 1
    say("A UNMET FACTS by fact and cause (a goal can have several): " + "; ".join(f"{k} {v}" for k, v in sorted(un.items(), key=lambda x: -x[1])))
    nf = sum(1 for _gi, _p, g in goals if g["fixes"])
    say(f"A FIXES: goals with at least one fix proposed {nf}, with none {len(goals) - nf}; goals by number of unmet facts: " + "; ".join(f"{k} {sum(1 for _g, _p, g in goals if len(g['unmet']) == k)}" for k in range(0, 6)))
    tg = sorted(p["t_goal"] for _g, _i, p in positions)
    say(f"A TIME of the goals per position (one thread, the concrete read apart): median {tg[len(tg) // 2]:.3f} s, mean {sum(tg) / len(tg):.3f} s, slowest {tg[-1]:.2f} s; the concrete read per game median {sorted(r['t_conc'] for r in res)[ng // 2]:.1f} s, the B reads {sorted(r['t_b'] for r in res)[ng // 2]:.1f} s per game, wall per game median {sorted(r['wall'] for r in res)[ng // 2]:.1f} s")
    # ---- B
    outc: dict = {}
    bysrc_b: dict = {}
    ctl = [0, 0]
    still: dict = {}
    single = {}
    rej: dict = {}
    for _gi, _p, g in goals:
        b = g["b"]
        if b["state"] != "read":
            outc["no edit: " + b["state"]] = outc.get("no edit: " + b["state"], 0) + 1
            continue
        v = b["variants"]
        fx, nu, co = v["fix"]["verdict"], v["null"]["verdict"], v["control"]["verdict"]
        if fx == "REJECTED":
            o = "the fix edit was rejected (no legal position)"
            w = b.get("why") or ""
            rej[w[:70]] = rej.get(w[:70], 0) + 1
        elif nu == "CONFIRMED":
            o = "works without the fix (the fix was not needed)"
        elif fx == "CONFIRMED":
            o = "works with the fix (diagnosis complete)"
        else:
            o = "still fails with the fix (something else also stops it)"
            still[v["fix"].get("still", "?")] = still.get(v["fix"].get("still", "?"), 0) + 1
        outc[o] = outc.get(o, 0) + 1
        bysrc_b.setdefault(g["source"], {})
        bysrc_b[g["source"]][o] = bysrc_b[g["source"]].get(o, 0) + 1
        if co != "REJECTED" and fx != "REJECTED":
            ctl[0] += 1
            ctl[1] += 1 if co == "CONFIRMED" else 0
        k = "one fix" if len(g["fixes"]) == 1 else "several fixes"
        single.setdefault(k, {})
        single[k][o] = single[k].get(o, 0) + 1
    nnu = sum(1 for _gi, _p, g in goals if g["b"]["state"] == "read" and g["b"]["variants"]["null"]["verdict"] == "CONFIRMED")
    say(f"B THE POSITION RE-SPELT WITH NO EDIT (the goal works without the fix): {nnu} of {sum(1 for _gi, _p, g in goals if g['b']['state'] == 'read')} goals read")
    alls = [g for _gi, _p, g in goals if g["b"]["state"] == "read" and "all" in g["b"]["variants"]]
    a_ok = sum(1 for g in alls if g["b"]["variants"]["all"]["verdict"] == "CONFIRMED")
    a_still = [g for g in alls if g["b"]["variants"]["fix"]["verdict"] not in ("CONFIRMED", "REJECTED")]
    a_fixed = sum(1 for g in a_still if g["b"]["variants"]["all"]["verdict"] == "CONFIRMED")
    say(f"B WITH EVERY STOPPER the unmet facts name taken out (not only the one the line showed), for the {len(alls)} goals that name more than one: the goal works in {a_ok}; of the {len(a_still)} of them that still failed with the fix alone, it works in {a_fixed}")
    wf = [g for _gi, _p, g in goals if g["b"]["state"] == "read" and g["b"]["variants"]["fix"]["verdict"] == "CONFIRMED" and g["b"]["variants"]["null"]["verdict"] != "CONFIRMED"]
    pos_s = sum(1 for g in wf if g["stake"] is not None and g["stake"] > 0)
    say(f"B THE FIRST CLAIM (worth it) where the fix works: the target's stake is above zero in {pos_s} of {len(wf)} goals ({sum(1 for g in wf if g['stake'] is None)} have no stake: a unit of the goal kind or outside the head); the net of the line after the fix is above the record's resolution in all of them by the verdict")
    say("B THE FIX AS A WHAT-IF EDIT, the claim read again (our own reasoning only), the three outcomes: " + "; ".join(f"{k}: {v}" for k, v in sorted(outc.items())))
    for s, d in sorted(bysrc_b.items()):
        say(f"B by source [{s}]: " + "; ".join(f"{k}: {v}" for k, v in sorted(d.items())))
    for s, d in sorted(single.items()):
        say(f"B by number of fixes [{s}]: " + "; ".join(f"{k}: {v}" for k, v in sorted(d.items())))
    say(f"B CONTROL (a unit of the owner's that is not a stopper taken out instead): the goal works in {ctl[1]} of {ctl[0]} goals where both edits were legal")
    say("B STILL FAILS, what else stops it (the key of the goal read again on the edited position): " + "; ".join(f"{k or '(no named cause)'}: {v}" for k, v in sorted(still.items(), key=lambda x: -x[1])[:12]))
    er: dict = {}
    for r in res:
        for k, v in r["edit_reasons"].items():
            er[k] = er.get(k, 0) + v
    ea = sum(r["edit"]["applied"] for r in res)
    eb = sum(r["edit"]["rejected"] for r in res)
    say(f"B THE EDIT PATH: edits applied {ea}, rejected {eb}; reasons: " + "; ".join(f"{k}: {v}" for k, v in sorted(er.items(), key=lambda x: -x[1])[:8]))
    # ---- C: reach
    tabs = [table_of(goals, f) for f in (0, 1)]
    cal = []
    pooled = []
    for gi, p, g in goals:
        r = g["reach"].get(4)
        if r is None or g["untested"]:
            continue
        pr = chance_of(g, tabs[gi % 2])
        pooled.append((pr, 1 if r else 0, g))
    nt = sum(1 for _gi, _p, g in goals if g["untested"])
    say(f"C GOALS whose unmet facts the graph can show: {len(goals) - nt}; goals with an unmet fact the graph cannot show (a line read): {nt} (counted apart, not in the chance of reaching)")
    for k in LATER:
        have = [g for _gi, _p, g in goals if not g["untested"] and g["reach"].get(k) is not None]
        say(f"C REACH by {k} plies on (cumulative share of goals, the unit still standing then): {sum(1 for g in have if g['reach'][k])} of {len(have)} ({pct(sum(1 for g in have if g['reach'][k]) / max(1, len(have)))} pct)")
    say(f"C THE TABLE of the chance of reaching within four plies, fitted on the other half of the games by the key of the goal (smoothed; tabled keys with at least 20 goals): " + "; ".join(f"{k or '-'}: {v[1]}/{v[0]}" for k, v in sorted(tabs[0][0].items(), key=lambda x: -x[1][0])[:8] if v[0] >= 20))
    edges = [0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 1.01]
    for lo, hi in zip(edges, edges[1:]):
        sel = [(pr, r) for pr, r, _g in pooled if lo <= pr < hi]
        if sel:
            say(f"C CALIBRATION predicted chance in [{lo:.2f}, {hi:.2f}): {len(sel)} goals, mean predicted {sum(x for x, _ in sel) / len(sel):.3f}, reached within four plies {sum(r for _, r in sel) / len(sel):.3f}")
    # the unit won
    reached = [(g, k) for _gi, _p, g in goals if not g["untested"] for k in [min((k for k in LATER if g["reach"].get(k)), default=None)] if k is not None]
    notr = [g for _gi, _p, g in goals if not g["untested"] and any(g["reach"].get(k) is not None for k in LATER) and not any(g["reach"].get(k) for k in LATER)]
    wT = sum(1 for g, k in reached if g["won_T"].get(k))
    wA = sum(1 for g, k in reached if g["won_any"].get(k))
    base_all = [x for r in res for x in r["own_took"]]
    say(f"C THE UNIT WON: at the first later position where all the unmet facts the graph can show held ({len(reached)} goals) the side's own move took the target in {wT} ({pct(wT / max(1, len(reached)))} pct) and took some unit in {wA} ({pct(wA / max(1, len(reached)))} pct); base rate of any move of the game taking a unit {pct(sum(base_all) / max(1, len(base_all)))} pct; goals whose facts did not come true within eight plies (the unit still standing): {len(notr)}, of which the side took the target within eight plies of the position anyway {sum(1 for g in notr if g['taken_by_side'])}")
    # ---- C: hindsight
    rng = random.Random(SEED)
    by_pos = {(gi, p["ply"]): p for gi, _gid, p in positions}
    ev_tot = ev_kept = 0
    elig = {k: 0 for k in BACKS}
    avail = {k: 0 for k in BACKS}
    cover = {k: 0 for k in BACKS}
    hit = {k: {"P1": 0, "P3": 0, "S3": 0, "R1": 0.0, "R3": 0.0, "FULL3": 0, "R3full": 0.0, "any": 0, "U3": 0.0} for k in BACKS}
    sizeL = {k: [] for k in BACKS}
    for gi, r in enumerate(res):
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
                sizeL[k].append(len(gl))
                tabs_g = tabs[gi % 2]
                top = max((g["stake"] for g in gl if g["stake"] is not None), default=1.0)
                m1 = [g["uid"] == e["uid"] for g in gl]
                full = [g["uid"] == e["uid"] and set(g["testable"]) <= became and not g["untested"] for g in gl]
                cover[k] += 1 if any(m1) else 0
                P = sorted(range(len(gl)), key=lambda i: (-(chance_of(gl[i], tabs_g) * max(0.0, gl[i]["stake"] if gl[i]["stake"] is not None else top)), len(gl[i]["unmet"]), i))
                S = sorted(range(len(gl)), key=lambda i: (-(max(0.0, gl[i]["stake"] if gl[i]["stake"] is not None else top)), len(gl[i]["unmet"]), i))
                hit[k]["P1"] += 1 if m1[P[0]] else 0
                hit[k]["P3"] += 1 if any(m1[i] for i in P[:3]) else 0
                hit[k]["S3"] += 1 if any(m1[i] for i in S[:3]) else 0
                hit[k]["FULL3"] += 1 if any(full[i] for i in P[:3]) else 0
                hs, hf = sum(m1), sum(full)
                hit[k]["R1"] += hyper_hit(len(gl), hs, 1)
                hit[k]["R3"] += hyper_hit(len(gl), hs, 3)
                hit[k]["R3full"] += hyper_hit(len(gl), hf, 3)
                hit[k]["U3"] += hyper_hit(max(pos["n_other"], len(gl)), 1, 3)                  # a goal about a unit of the other side drawn at random, three of them
                hit[k]["any"] += 1
    say(f"C EVENTS (a side takes a unit and keeps it: nothing taken back in its two replies): {ev_kept} of {ev_tot} takings in the sample")
    for k in BACKS:
        h = hit[k]
        n = max(1, h["any"])
        say(f"C HINDSIGHT {k} plies back: events where some testable fact became true in between {elig[k]}; of them with the earlier position in the sample {avail[k]}; with at least one goal listed {h['any']} (mean goals listed {sum(sizeL[k]) / max(1, len(sizeL[k])):.1f}); a goal about the unit then won was listed in {cover[k]} ({pct(cover[k] / n)} pct)")
        say(f"C HINDSIGHT {k} plies back, among the proposer's first few (chance of the key times the stake) over those {h['any']} events: first 1 {h['P1']} ({pct(h['P1'] / n)} pct), first 3 {h['P3']} ({pct(h['P3'] / n)} pct); CONTROL goals drawn at random from the same list (exact): first 1 {pct(h['R1'] / n)} pct, first 3 {pct(h['R3'] / n)} pct; the order by stake alone, first 3: {pct(h['S3'] / n)} pct; the full match (the goal's unmet facts are the ones that became true), proposer first 3: {pct(h['FULL3'] / n)} pct against random {pct(h['R3full'] / n)} pct. Ply is held fixed: the control draws from the list of the same position")
        say(f"C HINDSIGHT {k} plies back, RECALL: of the {avail[k]} events with the earlier position in the sample, a goal about the unit then won was listed for {cover[k]} ({pct(cover[k] / max(1, avail[k]))} pct); CONTROL three goals about units of the other side drawn at random (one of {sum(p['n_other'] for p in by_pos.values()) / max(1, len(by_pos)):.1f} on average), first 3, over the {h['any']} events with a list: {pct(h['U3'] / n)} pct")
    return 0


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["read", "grade"])
    ap.add_argument("--games", type=int, default=50)
    ap.add_argument("--procs", type=int, default=10)
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    return {"read": cmd_read, "grade": cmd_grade}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
