"""PLAN 78.79 (extends 78.78 B, C, D): THE THREE SIGNALS (the other side's goals as ours to stop, the long count, the stoppers with a reply read) SCORED ON THE OPEN SURPRISES, each alone with the base-rate table, on the scored part
and, as a second look kept apart, on the other part of the split. (Below: plan 78.78's own header, still true of the 78.78 parts.) PLAN 78.78 B, C, D: THE SUBGOAL RULE SCORED ON THE OPEN SURPRISES. SYSTEM_SPEC 7 (a subgoal is a hint that orders the moves the search holds equal) and 9 (a rule is scored by the surprises it resolves less those it creates).
A sandbox: no player is changed, nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/subgoal_pick_78_79/<tag>/. Game-free: the game is named only through the pieces that
exist (ladder_measure, move_rule_play, odds_measure); the oracle's answers are read only in `score_all`, which hands them to deep_heads.score and hit_rate and to `ceiling` (the same wall as deep_heads: the picks never see them).

INTERFACE (what this module takes, what it returns, who calls it).
  prepare_position(position, tools, reader) -> prep
      The cheap rungs of the ladder as deep_heads.prepare reads them (the search at the wake's clock, the forcing read: "eq", "move", "forced", the keys deep_heads.score reads) and, where two or more moves are held equal, the
      goals of the side to move (move_rule_play.RuleReader: the threat list made concrete, a line read for every unit, subgoal_goals.goals), the count of every goal before and after every tied move (subgoal_pick.count_table),
      the picks of the two modes that need the position (`rule`, `goals`) and what each read cost. Reads NO answer.
  candidates(preps) -> [candidate]          the candidates of the plan (a pick among the tied moves of a prep; deep_heads.score's form {"name", "pick", "cost", "random"}).
  ceiling(preps, open_ids, answers, split) -> dict      THE CEILING of the approach: among the open surprises whose accepted move is among the tied moves, how often an accepted move lowers the count of a listed goal,
                                                        by the rank of the goal, and the same for a tied move that is not accepted (the base rate). Reads the answers: lives here, beside the score, and nothing a pick reads sees it.
  silence(preps, ids) -> dict               how many goals a position has, how many tied moves change a count, and where no tied move does (the hint is then silent).
  prepare_signals(moves, tied, reader, rd, prep)      PLAN 78.79: adds to a prep the three signals' tables (subgoal_pick: cost_to_theirs from `reader.read(..., side="other")` and the plain read after each tied move; long_table; the mode-9 line after each
                                            tied move and subgoal_pick.line_keeps_unit) and what each cost in seconds. Reads NO answer. Skipped for a reader without a text editor.
  base_rate(preps, open_ids, answers, split, signals) -> dict      THE BASE RATE of 78.78 for each signal alone: per tied move, the share of accepted tied moves that carry it and of tied moves that are not accepted that carry it, and per position.
                                            Reads the answers: beside the score, never seen by a pick.   signal_silence(preps, ids, signals): the positions where no tied move carries a signal, and how many moves do.
  main                                      run | report, a Pool over the cores (one thread a worker). Both parts of the split are prepared; the score is on the scored part, the base rate on both, labelled.
Called by: the command line only; ladder_print.py reads its pickle for the cases.

  .venv/bin/python -m catspace.incubator.analyst.discovery.subgoal_pick_measure run --tag short --limit 20 | run --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import pickle
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import deep_heads as DH
from catspace.incubator.analyst.discovery import ladder_measure as LM
from catspace.incubator.analyst.discovery import move_rule as MR
from catspace.incubator.analyst.discovery import move_rule_play as MP
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_pick as SP
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.io import paths

OUT = ("subgoal_pick_78_79",)
OLD = ("subgoal_pick_78_78", "full")                              # the twelve cases of 78.78 (case_rows.json)
SEED = 7878
DRAWS = 1000
SHOWN = 10                                                         # goals kept in words per position (for the cases)
_TOOLS = None
_READER = None


def out_root(tag: str) -> Path:
    return Path(paths.experiment(*OUT, tag))


def jsonl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


# ------------------------------------------------------------------------------------------------------------------------------ the preparation

def goal_info(g: dict) -> dict:
    """A goal as plain data: what the printing script and the ceiling need, no graph."""
    return {"target": list(g["target"]), "source": g["source"], "words": MR.goal_words(g), "unmet": list(g["unmet"]), "own_moves": g["own_moves"], "w": MR.weight_of(g),
            "stoppers": [list(u) for u in g.get("stoppers", [])], "steps": [{"step": s["step"], "unit": list(s["unit"]) if s.get("unit") else None, "route": list(s.get("route") or []), "moves": s.get("moves")} for s in g.get("steps") or []],
            "line": list(g.get("line") or [])[:4], "stake": g["stake"]}


def prepare_position(position: dict, tools, reader) -> dict:
    """See the module doc. `reader` is a move_rule_play.RuleReader (one per process)."""
    moves = list(position["moves"])
    t0 = time.perf_counter()
    s = tools.search(position, LM.LIMITS["wake_seconds"])
    eq = list(s["equals"])
    f = tools.forced(position)
    prep = {"eq": eq, "move": s["move"], "forced": f, "t_cheap": time.perf_counter() - t0, "read": None}
    if len(eq) < 2 or s["move"] is None:
        return prep
    tied = [s["move"]] + sorted(m for m in eq if m != s["move"])
    prep["tied"] = tied
    t1 = time.perf_counter()
    rd = reader.read(moves, (1, 2, 3))
    prep["t_read"] = time.perf_counter() - t1
    if rd is None:
        prep["read"] = "none"
        return prep
    prep["read"] = "ok"
    goals = rd["goals"]
    pos = rd["position"]
    t2 = time.perf_counter()
    table = SP.count_table(pos, tied, goals)
    prep["t_table"] = time.perf_counter() - t2
    prep["table"] = table
    prep["goals"] = [goal_info(g) for g in goals]
    prep["own_mismatch"] = sum(1 for g, b in zip(goals, table["before"]) if g["own_moves"] != b["own"])
    for mode in ("rule", "goals"):
        t3 = time.perf_counter()
        pick, reason, _h = MR.choose(pos, tied, rd["lists"], goals, None, (1, 2, 3) if mode == "rule" else (2,))
        prep["t_" + mode] = time.perf_counter() - t3
        prep["pick_" + mode] = pick
        prep["reason_" + mode] = str(reason)[:600]
        prep["branch_" + mode] = reason.branch
    prep["threats"] = len(MR.threat_rows(rd["lists"]))
    if hasattr(reader, "editor"):
        prepare_signals(moves, tied, reader, rd, prep)
    return prep


def works_targets(rd: dict) -> list:
    """The targets of the side to move of the read `rd` that already work there (subgoal_goals.works), as (role, kind, place); the word "goal_unit" when a forced end of the game stands for it."""
    item, c = rd["item"], rd["lists"]
    side = item["raw"]["mover"]
    names = item["names"]
    out: list = []
    for r in SG.works(item, side, c):
        place = r["place"] if isinstance(r["place"], str) else names["places"][r["place_ix"]]
        out.append((names["roles"][r["owner"]], r["kind"], place))
    if any(r.get("owner") == 1 - side for r in c.get("goal", ())):
        out.append("goal_unit")
    return out


def prepare_signals(moves: list, tied: list, reader, rd: dict, prep: dict) -> None:
    """See the module doc. The other side's goals at the position come from the reader run for the other side; after each tied move, from the plain read of the position after it (the side to move there is the other side)."""
    pos, goals = rd["position"], rd["goals"]
    t0 = time.perf_counter()
    prep["their"], prep["their_unread"] = None, 0
    try:
        root = reader.read(moves, (2,), side="other")
    except Exception:
        root = None
    prep["their_read"] = "ok" if root is not None else "none"
    if root is not None:
        after = {}
        for m in tied:
            try:
                r = reader.read(list(moves) + [m], (2,))
            except Exception:
                r = None
            after[m] = None if r is None else {"goals": r["goals"], "works": works_targets(r)}
        prep["their"] = SP.cost_to_theirs(pos, tied, root["goals"], reader.facts, after)
        prep["their_words"] = [MR.goal_words(g) for g in root["goals"][:SHOWN]]
        prep["their_n"] = len(root["goals"])
        prep["their_unread"] = sum(1 for m in tied if after[m] is None)
    prep["t_their"] = time.perf_counter() - t0
    t1 = time.perf_counter()
    prep["long"] = SP.long_table(pos, tied, goals, reader.facts, SP.LONG_LIMIT, reader.editor)
    prep["t_long"] = time.perf_counter() - t1
    t2 = time.perf_counter()
    res = reader.lines.ask([(None, list(moves) + [m]) for m in tied], [(i, 9, 0, 1, 0, 0, 0) for i in range(len(tied))], TC.DEPTH + 1, TC.FUSE)
    raw, why = {}, {}
    for m, r in zip(tied, res):
        rd9 = (r.get("reads") or [{}])[0] if r.get("ok") else {}
        raw[m] = SP.line_keeps_unit(pos, reader.facts, m, {"ok": r.get("ok"), "names": rd9.get("names", []), "settled": rd9.get("settled"), "cut": rd9.get("cut")})
        if raw[m] is None:
            why[m] = "not read" if not r.get("ok") else ("cut by the fuse" if rd9.get("cut") else ("no line" if not rd9.get("names") else "line not replayed"))
    prep["replied_raw"], prep["replied_why"] = raw, why
    prep["t_replied"] = time.perf_counter() - t2
    prep["table"] = dict(prep["table"], their=prep["their"], long=prep["long"], replied={m: v is True for m, v in raw.items()})


def _job(position: dict) -> tuple:
    global _TOOLS, _READER
    if _TOOLS is None:
        _TOOLS = LM.Tools()
        _READER = MP.reader75()
    try:
        return position["id"], prepare_position(position, _TOOLS, _READER), None
    except Exception as e:
        import traceback
        return position["id"], None, position["id"] + ": " + repr(e)[:200] + " | " + " <- ".join(f"{f.name}:{f.lineno}" for f in reversed(traceback.extract_tb(e.__traceback__)))[:300]


def prepare_all(positions: list, workers: int, root: Path, say) -> dict:
    f = root / "preps.pkl"
    preps = pickle.loads(f.read_bytes()) if f.exists() else {}
    todo = [p for p in positions if p["id"] not in preps]
    t0 = time.perf_counter()
    errs = []
    with multiprocessing.get_context("spawn").Pool(workers) as pool:
        for i, (pid, p, err) in enumerate(pool.imap_unordered(_job, todo, chunksize=1), 1):
            if p is not None:
                preps[pid] = p
            else:
                errs.append(err)
            if i % 20 == 0:
                say(f"progress: {i} of {len(todo)} positions, {time.perf_counter() - t0:.0f} s, load average {os.getloadavg()[0]:.2f}")
    f.write_bytes(pickle.dumps(preps))
    say(f"prepared {len(todo)} positions (both parts) in {time.perf_counter() - t0:.0f} s ({workers} workers, load average {os.getloadavg()[0]:.2f}); errors {len(errs)}" + (f" ({errs[:3]})" if errs else ""))
    return preps


# ------------------------------------------------------------------------------------------------------------------------------ the candidates

def _pick_of(name: str, mode: str, scale: str):
    def pick(pid: str, prep: dict) -> str:
        if len(prep["eq"]) < 2 or prep.get("read") != "ok":
            return prep["move"]
        if mode in ("rule", "goals"):
            return prep["pick_" + mode]
        return SP.goal_pick(None, prep["tied"], prep["goals"], prep["table"], mode, None, scale)
    return pick


def _cost_of(mode: str):
    def cost(prep: dict) -> float:
        if len(prep["eq"]) < 2 or prep.get("read") != "ok":
            return float(prep.get("t_read", 0.0))
        base = prep["t_read"] + (prep["t_" + mode] if mode in ("rule", "goals") else prep["t_table"])
        if mode in SP.NEW_MODES:
            if mode in ("defence", "defence_long", "sum"):
                base += prep.get("t_their", 0.0)
            if mode in ("long", "defence_long", "sum"):
                base += prep.get("t_long", 0.0)
            if mode in ("replied", "sum"):
                base += prep.get("t_replied", 0.0)
        return float(base)
    return cost


def candidates() -> list:
    out = [{"name": "control: the search's own first move", "pick": lambda pid, p: p["move"], "cost": lambda p: 0.0, "random": False},
           {"name": "control: a random pick (expected)", "pick": None, "cost": lambda p: 0.0, "random": True}]
    for name, mode, scale in (("1 the rule as built (threats, then the held or first goal)", "rule", "own"), ("2 goals only (step 2)", "goals", "own"),
                              ("3a goal 1: the move that lowers layer one's count most", "top1", "own"), ("3b goals 1-2: summed drop", "top2", "own"), ("3c goals 1-3: summed drop", "top3", "own"),
                              ("4 every goal: summed drop", "all", "own"),
                              ("x1 goals 1-3 by the stoppers still in the way (78.75's scale)", "top3", "w"), ("x2 every goal by the stoppers still in the way", "all", "w")):
        out.append({"name": name, "pick": _pick_of(name, mode, scale), "cost": _cost_of(mode), "random": False, "mode": mode, "scale": scale})
    for name, mode in (("d the defence only (what the move costs the other side's goals)", "defence"), ("l the long count only (own moves for a unit to bear, no cut at two)", "long"),
                       ("r the stoppers with the reply read only (the unit that moved is kept by the line)", "replied"), ("dl the defence first, then the long count", "defence_long"), ("s the sum of the three", "sum")):
        out.append({"name": name, "pick": _pick_of(name, mode, "own"), "cost": _cost_of(mode), "random": False, "mode": mode, "scale": "own", "new": True})
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the ceiling and the silence (the ceiling reads the answers)

def _lowers(prep: dict, move: str, scale: str, lo: int, hi: "int | None") -> bool:
    """`move` lowers the count of some goal with index in [lo, hi) (hi None: to the end)."""
    t = prep["table"]
    n = len(t["before"]) if hi is None else min(hi, len(t["before"]))
    return any(SP.goal_drop(t, move, scale, i) > 0 for i in range(lo, n))


def ceiling(preps: dict, open_ids: list, answers: dict, split: str = "scored") -> dict:
    """For the open surprises of the part `split` whose accepted move is among the tied moves, and the goals read ('ok' reads): by scale ("own": layer one's count, "w": the stoppers) and rank class (the first goal,
    the first three, any), over the positions: the share where SOME accepted tied move lowers a goal of the class, and where SOME tied move that is not accepted does; over the (position, move) pairs: the share of the
    accepted and of the not accepted tied moves that do. The not accepted moves are the base rate."""
    ids = [i for i in open_ids if DH.split_part(i) == split and i in preps and len(preps[i]["eq"]) >= 2 and set(answers[i]) & set(preps[i]["eq"])]
    out = {"n_open_in_tie": len(ids), "n_read": 0, "rows": {}}
    ids = [i for i in ids if preps[i].get("read") == "ok"]
    out["n_read"] = len(ids)
    classes = {"first goal": (0, 1), "first three goals": (0, 3), "any goal": (0, None)}
    for scale in ("own", "w"):
        for cname, (lo, hi) in classes.items():
            pos_acc = pos_non = pos_both = n_both = 0
            pr_acc = [0, 0]
            pr_non = [0, 0]
            for i in ids:
                p, acc = preps[i], set(answers[i])
                a = [m for m in p["tied"] if m in acc]
                nn = [m for m in p["tied"] if m not in acc]
                la = [_lowers(p, m, scale, lo, hi) for m in a]
                ln = [_lowers(p, m, scale, lo, hi) for m in nn]
                pos_acc += any(la)
                pr_acc[0] += sum(la)
                pr_acc[1] += len(la)
                pr_non[0] += sum(ln)
                pr_non[1] += len(ln)
                if nn:
                    n_both += 1
                    pos_non += any(ln)
                    pos_both += any(la)
            out["rows"][(scale, cname)] = {"pos_acc": pos_acc, "n": len(ids), "pos_acc_both": pos_both, "pos_non": pos_non, "n_both": n_both, "pairs_acc": pr_acc, "pairs_non": pr_non}
    return out


def silence(preps: dict, ids: list) -> dict:
    """Over the positions `ids` with two or more moves held equal: the goals a position has, the tied moves that change a count of some goal (either scale; and by scale), the positions where none does."""
    rows = [preps[i] for i in ids if i in preps and len(preps[i]["eq"]) >= 2]
    ok = [p for p in rows if p.get("read") == "ok"]
    ng = [len(p["goals"]) for p in ok]
    out = {"n_ties": len(rows), "n_read": len(ok), "goals": ng, "n_zero_goals": sum(1 for n in ng if n == 0), "tied": [len(p["tied"]) for p in ok]}
    for scale in ("any", "own", "w"):
        ch = []
        for p in ok:
            t = p["table"]
            c = 0
            for m in p["tied"]:
                moved = [(x["own"] != y["own"]) if scale == "own" else (x["w"] != y["w"]) if scale == "w" else (x != y) for x, y in zip(t["before"], t["after"][m])]
                c += any(moved)
            ch.append(c)
        out["changing_" + scale] = ch
        out["silent_" + scale] = sum(1 for c in ch if c == 0)
    out["goal_unit_only"] = sum(1 for p in ok if p["goals"] and all(g["source"] == "goal_unit" for g in p["goals"]))
    out["own_none"] = [sum(1 for b in p["table"]["before"] if b["own"] is None) for p in ok]
    out["own_total"] = [len(p["table"]["before"]) for p in ok]
    out["mismatch"] = sum(p["own_mismatch"] for p in ok)
    out["rule_branches"] = {}
    for p in ok:
        out["rule_branches"][p["branch_rule"]] = out["rule_branches"].get(p["branch_rule"], 0) + 1
    return out


# ------------------------------------------------------------------------------------------------------------------------------ PLAN 78.79: the base rate and the silence of each signal alone

def _has(prep: dict, key: str) -> bool:
    return prep.get("read") == "ok" and bool(prep["table"].get(key))


SIGNALS = {
    "the stoppers still in the way, as 78.78 read them (no reply read)": lambda p, m: SP.drop(p["table"], m, "w", None) > 0,
    "1 the defence: the move costs some goal of the other side": lambda p, m: _has(p, "their") and SP.defends(p["table"]["their"], m),
    "1b the defence, net: the costs over all goals of the other side sum above nothing (what the pick reads)": lambda p, m: _has(p, "their") and SP.defence_gain(p["table"]["their"], m) > 0,
    "1c the defence on the first three goals of the other side: some one of them costs": lambda p, m: _has(p, "their") and any(r["cost"] > 0 for r in (p["table"]["their"]["after"].get(m) or [])[:3]),
    "1' the move makes some goal of the other side easier (to avoid)": lambda p, m: _has(p, "their") and SP.eases(p["table"]["their"], m),
    "2 the long count: the move lowers the count of some goal": lambda p, m: _has(p, "long") and SP.long_drop(p["table"], m) > 0,
    "3 the stoppers with the reply read: the move takes stoppers away and its line keeps the unit": lambda p, m: _has(p, "replied") and SP.replied_drop(p["table"], m) > 0,
}


def base_rate(preps: dict, open_ids: list, answers: dict, split: str, signals: dict = SIGNALS) -> dict:
    """For the open surprises of the part `split` whose accepted move is among the tied moves and the goals were read: by signal, over (position, tied move) pairs the share of the accepted and of the not accepted moves that carry the signal;
    over positions the share where some accepted tied move carries it, and where some not accepted one does (among the positions that have both kinds)."""
    ids = [i for i in open_ids if DH.split_part(i) == split and i in preps and len(preps[i]["eq"]) >= 2 and set(answers[i]) & set(preps[i]["eq"]) and preps[i].get("read") == "ok"]
    out = {"n": len(ids), "rows": {}}
    for name, fn in signals.items():
        pa, pn = [0, 0], [0, 0]
        pos_acc = pos_non = pos_both = n_both = 0
        for i in ids:
            p, acc = preps[i], set(answers[i])
            la = [bool(fn(p, m)) for m in p["tied"] if m in acc]
            ln = [bool(fn(p, m)) for m in p["tied"] if m not in acc]
            pa[0] += sum(la)
            pa[1] += len(la)
            pn[0] += sum(ln)
            pn[1] += len(ln)
            pos_acc += any(la)
            if ln:
                n_both += 1
                pos_non += any(ln)
                pos_both += any(la)
        out["rows"][name] = {"pairs_acc": pa, "pairs_non": pn, "pos_acc": pos_acc, "pos_non": pos_non, "n_both": n_both, "pos_acc_both": pos_both}
    return out


def signal_silence(preps: dict, ids: list, signals: dict = SIGNALS) -> dict:
    """Over the positions of `ids` with two or more moves held equal and the goals read: by signal, how many positions have no tied move that carries it (the hint is then silent), and the carrying moves per position."""
    ok = [preps[i] for i in ids if i in preps and len(preps[i]["eq"]) >= 2 and preps[i].get("read") == "ok"]
    out = {"n_read": len(ok), "rows": {}}
    for name, fn in signals.items():
        c = [sum(bool(fn(p, m)) for m in p["tied"]) for p in ok]
        out["rows"][name] = {"silent": sum(1 for x in c if x == 0), "carrying": c}
    return out


# ------------------------------------------------------------------------------------------------------------------------------ the run

def score_all(preps: dict, root: Path, say) -> None:
    lr = LM.out_root("full")
    positions = {p["id"]: p for p in jsonl(lr / "positions.jsonl")}
    results = {d["id"]: d for d in jsonl(lr / "results.jsonl")}
    open_ids = [r["id"] for r in jsonl(lr / "surprise_list.jsonl")]
    resolved_ids = [i for i, d in results.items() if d["result"] and d["result"]["rung"] in (1, 2)]
    ans = {d["id"]: d["accepted"] for d in jsonl(lr / "answers.jsonl")}          # the answers file: read here, handed to score(), hit_rate() and ceiling() only
    scored = [i for i in positions if DH.split_part(i) == "scored" and i in preps]
    cands = candidates()
    table, hits = [], []
    for c in cands:
        table.append(DH.score(c, preps, open_ids, resolved_ids, ans, "scored"))
        hits.append(DH.hit_rate(c, preps, scored, ans, "scored"))
    for s, h, c in zip(table, hits, cands):
        lo_r, hi_r = DH.wilson(s["resolved"], s["n_open"])
        lo_c, hi_c = DH.wilson(s["created"], s["n_resolved_sample"])
        say(f"STAGE H {s['name']}: resolved {s['resolved']:.1f} of {s['n_open']} open surprises (share {s['resolved'] / max(1, s['n_open']):.3f} [{lo_r:.3f}, {hi_r:.3f}]); created {s['created']:.1f} of {s['n_resolved_sample']} "
            f"resolved positions (share {s['created'] / max(1, s['n_resolved_sample']):.3f} [{lo_c:.3f}, {hi_c:.3f}]); net {s['resolved'] - s['created']:+.1f}; hit share {100 * h['rate']:.1f} pct [{100 * h['lo']:.1f}, {100 * h['hi']:.1f}] "
            f"({h['hits']:.1f} of {h['n']} ties, 90 pct Wilson); cost {s['cost']:.3f} s a position; created over all {s['n_resolved_all']} resolved positions of the part: {s['created_all']:.1f}")
    ctrl = table[0]
    rng = np.random.default_rng(SEED)
    ia = rng.integers(0, len(ctrl["res_vec"]), (DRAWS, len(ctrl["res_vec"])))
    ib = rng.integers(0, max(1, len(ctrl["keep_vec"])), (DRAWS, len(ctrl["keep_vec"])))
    for s in table:
        d_open = np.array(s["res_vec"]) - np.array(ctrl["res_vec"])
        d_keep = np.array(s["keep_vec"]) - np.array(ctrl["keep_vec"])                 # created less: more kept is fewer created
        diff = d_open[ia].sum(axis=1) + d_keep[ib].sum(axis=1)
        lo, hi = np.percentile(diff, [5, 95])
        say(f"STAGE H net against the search's own first move, {s['name']}: {d_open.sum() + d_keep.sum():+.1f} positions [{lo:+.1f}, {hi:+.1f}] (90 pct bootstrap over positions, paired)")
    # the ceiling (reads the answers) and the silence
    ce = ceiling(preps, open_ids, ans, "scored")
    say(f"CEILING: of the open surprises of the scored part, {ce['n_open_in_tie']} have an accepted move among the moves held equal; the goals were read for {ce['n_read']} of them")
    for (scale, cname), r in ce["rows"].items():
        what = "layer one's count of our own moves" if scale == "own" else "the stoppers in the way"
        pa, pn = r["pairs_acc"], r["pairs_non"]
        say(f"CEILING [{what}] {cname}: an accepted tied move lowers it in {r['pos_acc']} of {r['n']} open positions ({100 * r['pos_acc'] / max(1, r['n']):.1f} pct); a tied move that is not accepted does in {r['pos_non']} of "
            f"the {r['n_both']} positions that have one ({100 * r['pos_non'] / max(1, r['n_both']):.1f} pct; the accepted one in the same {r['n_both']} positions: {r['pos_acc_both']}, {100 * r['pos_acc_both'] / max(1, r['n_both']):.1f} pct); "
            f"per tied move: accepted {pa[0]} of {pa[1]} ({100 * pa[0] / max(1, pa[1]):.1f} pct [{100 * DH.wilson(pa[0], pa[1])[0]:.1f}, {100 * DH.wilson(pa[0], pa[1])[1]:.1f}]), "
            f"not accepted {pn[0]} of {pn[1]} ({100 * pn[0] / max(1, pn[1]):.1f} pct [{100 * DH.wilson(pn[0], pn[1])[0]:.1f}, {100 * DH.wilson(pn[0], pn[1])[1]:.1f}]) (90 pct Wilson)")
    for label, ids in (("every scored position", scored), ("the open surprises", [i for i in open_ids if i in preps and DH.split_part(i) == "scored"]),
                       ("the open surprises with an accepted move among the tied", [i for i in open_ids if i in preps and DH.split_part(i) == "scored" and DH.tie_eligible(preps[i], set(ans[i]))])):
        sl = silence(preps, ids)
        if not sl["n_read"]:
            continue
        g, t = np.array(sl["goals"]), np.array(sl["tied"])
        say(f"SILENCE {label}: {sl['n_ties']} positions with two or more moves held equal, goals read for {sl['n_read']}; goals per position: mean {g.mean():.1f}, median {np.median(g):.0f}, most {g.max()}, none {sl['n_zero_goals']}, only the goal unit's goal {sl['goal_unit_only']}; "
            f"tied moves per position mean {t.mean():.1f}")
        for scale, nm in (("any", "either count"), ("own", "layer one's count"), ("w", "the stoppers")):
            c = np.array(sl["changing_" + scale])
            say(f"SILENCE {label} [{nm}]: tied moves that change the count of at least one goal per position: mean {c.mean():.2f}, median {np.median(c):.0f}; positions where no tied move changes any count (the hint is silent): "
                f"{sl['silent_' + scale]} of {sl['n_read']} ({100 * sl['silent_' + scale] / sl['n_read']:.1f} pct)")
        say(f"SILENCE {label}: goals with no layer-one count (None) before the move: {sum(sl['own_none'])} of {sum(sl['own_total'])}; the table's before-count differs from the goal's own_moves in {sl['mismatch']} goals; "
            f"the rule as built fired by branch {sl['rule_branches']}")
    report_79(preps, open_ids, ans, scored, say)
    # the cases: THE SAME TWELVE POSITIONS as plan 78.78 (its case_rows.json: set A the first six open surprises of the scored part with an accepted move among the tied, set B the first six resolved positions its best candidate broke)
    old = json.loads(Path(paths.experiment(*OLD, "case_rows.json")).read_text())
    ids = old["A"] + old["B"]
    best = max((s for s, c in zip(table, cands) if c.get("new")), key=lambda s: s["resolved"] - s["created"])
    say(f"BEST NEW CANDIDATE by net (printed for the reader; the cases are the twelve of 78.78, fixed before): {best['name']}")
    rows = {"A": old["A"], "B": old["B"], "moves": {i: positions[i]["moves"] for i in ids if i in positions},
            "picks": {i: {c["name"]: (preps[i]["move"] if c.get("random") else c["pick"](i, preps[i])) for c in cands if not c.get("random")} for i in ids if i in preps}}
    (root / "case_rows.json").write_text(json.dumps(rows))
    say(f"CASES fixed: set A {old['A']}; set B {old['B']} (those of plan 78.78)")


def pct(k: float, n: int) -> str:
    lo, hi = DH.wilson(k, n)
    return f"{k:.0f} of {n} ({100 * k / max(1, n):.1f} pct [{100 * lo:.1f}, {100 * hi:.1f}])"


def report_79(preps: dict, open_ids: list, ans: dict, scored: list, say) -> None:
    """The OBS lines of plan 78.79: the base rate of each signal alone on the scored part and, labelled apart, on the other part; the silence; the guards; the cost."""
    for split, label in (("scored", "SCORED PART"), ("fit", "SECOND LOOK, THE OTHER PART (nothing was fitted, so both are honest; kept apart)")):
        br = base_rate(preps, open_ids, ans, split)
        say(f"BASE RATE [{label}]: {br['n']} open surprises with an accepted move among the moves held equal and the goals read")
        for name, r in br["rows"].items():
            pa, pn = r["pairs_acc"], r["pairs_non"]
            say(f"BASE RATE [{label}] signal {name}: per tied move, accepted {pct(pa[0], pa[1])}, not accepted {pct(pn[0], pn[1])} (90 pct Wilson); per position, an accepted move carries it in {r['pos_acc']} of {br['n']} "
                f"({100 * r['pos_acc'] / max(1, br['n']):.1f} pct); in the {r['n_both']} positions with both kinds of move: an accepted one carries it in {r['pos_acc_both']} ({100 * r['pos_acc_both'] / max(1, r['n_both']):.1f} pct), "
                f"a not accepted one in {r['pos_non']} ({100 * r['pos_non'] / max(1, r['n_both']):.1f} pct)")
    for split, label in (("scored", "SCORED PART"), ("fit", "OTHER PART")):
        ids = [i for i in preps if DH.split_part(i) == split]
        sl = signal_silence(preps, ids)
        for name, r in sl["rows"].items():
            c = np.array(r["carrying"]) if r["carrying"] else np.zeros(1)
            say(f"SILENCE [{label}] signal {name}: of {sl['n_read']} positions with a tie and the goals read, no tied move carries it in {r['silent']} ({100 * r['silent'] / max(1, sl['n_read']):.1f} pct); tied moves carrying it per position: mean {c.mean():.2f}, median {np.median(c):.0f}")
        ok = [preps[i] for i in ids if preps[i].get("read") == "ok"]
        if ok:
            tt = np.array([p.get("t_their", 0.0) for p in ok])
            tl = np.array([p.get("t_long", 0.0) for p in ok])
            tr = np.array([p.get("t_replied", 0.0) for p in ok])
            allp = max(1, len(ids))
            say(f"COST [{label}] seconds per position with a tie read (mean / slowest): their goals and the defence {tt.mean():.2f} / {tt.max():.2f}; the long count {tl.mean():.2f} / {tl.max():.2f}; the reply read {tr.mean():.2f} / {tr.max():.2f}; "
                f"over all {len(ids)} positions of the part the means are {tt.sum() / allp:.2f}, {tl.sum() / allp:.2f}, {tr.sum() / allp:.2f}")
            unread = sum(p.get("their_unread", 0) for p in ok)
            nm = sum(len(p["tied"]) for p in ok)
            none_root = sum(1 for p in ok if p.get("their_read") != "ok")
            say(f"READS [{label}] the other side's goals at the position were not read (the game refused its pass) in {none_root} of {len(ok)} positions; the plain read after a tied move gave none in {unread} of {nm} moves; "
                f"the other side's goals per position: mean {np.mean([p.get('their_n', 0) for p in ok]):.1f}")
            lu = [p["replied_raw"] for p in ok if "replied_raw" in p]
            vals = [v for d in lu for v in d.values()]
            say(f"READS [{label}] the line after a tied move (mode 9): keeps the unit that moved {sum(v is True for v in vals)}, loses it {sum(v is False for v in vals)}, not counted {sum(v is None for v in vals)} of {len(vals)} moves (by reason: {dict(Counter(w for p in ok for w in p.get('replied_why', {}).values()))})")
            cuts = [p["long"]["cut"] for p in ok if p.get("long")]
            if cuts:
                tabs = sum(c["tables"] for c in cuts)
                say(f"GUARD [{label}] the long count's search (limit {SP.LONG_LIMIT} own moves a walk, {SP.LONG_FUSE} graph reads a position): {tabs} walk tables read; the limit of moves cut {sum(c['cut_depth'] for c in cuts)} ({100 * sum(c['cut_depth'] for c in cuts) / max(1, tabs):.1f} pct), "
                    f"the guard of reads hit {sum(c['cut_fuse'] for c in cuts)} ({100 * sum(c['cut_fuse'] for c in cuts) / max(1, tabs):.1f} pct); positions with our unit not to move after a tied move (no walk): {sum(c['unwalkable'] for c in cuts)}; graph reads {sum(c['reads'] for c in cuts)}; "
                    f"goals of bearing facts only with no count: the search was cut in {sum(c['none_cut'] for c in cuts)} cases (more walk might have found one), ended by itself in {sum(c['none_exhausted'] for c in cuts)}")
                lc = [(sum(1 for x in p["long"]["before"] if x is not None), len(p["long"]["before"])) for p in ok if p.get("long")]
                say(f"GUARD [{label}] goals with a long count (not None) before the move: {sum(a for a, _b in lc)} of {sum(b for _a, b in lc)}; goals whose unmet facts are all of bearing (the goals the count is defined for): {sum(p['long']['eligible'] for p in ok if p.get('long'))}; of layer one's own count (cut at two): {sum(sum(1 for x in p['table']['before'] if x['own'] is not None) for p in ok)} of {sum(len(p['table']['before']) for p in ok)}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("run", "report"))
    ap.add_argument("--tag", default="short")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)

    def say(m: str) -> None:
        line = f"OBS subgoal pick {a.stage}: {m}"
        print(line, flush=True)
        with open(root / "obs.txt", "a") as f:
            f.write(line + "\n")
    lr = LM.out_root("full")
    positions = {p["id"]: p for p in jsonl(lr / "positions.jsonl")}
    results = {d["id"]: d for d in jsonl(lr / "results.jsonl")}
    scored = [positions[i] for i in positions if DH.split_part(i) == "scored" and results.get(i, {}).get("result")]
    other = [positions[i] for i in positions if DH.split_part(i) != "scored" and results.get(i, {}).get("result")]
    if a.limit:
        scored = random.Random(SEED).sample(scored, a.limit)
        other = random.Random(SEED).sample(other, a.limit)
    say(f"THE SPLIT: the scored part of the ladder's 400 positions, {len(scored)} positions used, and the other part (the one the scored part excludes), {len(other)} positions used for the second look; load average "
        f"{os.getloadavg()[0]:.2f}; {a.workers} workers")
    if a.stage == "run":
        preps = prepare_all(scored + other, a.workers, root, say)
    else:
        preps = pickle.loads((root / "preps.pkl").read_bytes())
    score_all(preps, root, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
