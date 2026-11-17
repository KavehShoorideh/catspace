"""PLAN 78.75 C and D: THE RULE FOR CHOOSING A MOVE AS A TIE-BREAK IN THE MATCH HARNESS OF 78.67, and what its branches did. SYSTEM_SPEC section 8 (the rule for choosing a move) and section 7 (holding a subgoal).
A sandbox: nothing is wired into any player of the wake, nothing is written to rules/learned or data/derived; every output goes under artifacts/experiments/move_rule_78_75/. Game-free: the game is named only through
odds_measure (78.61), the engine and the existing readers.

INTERFACE (what this module takes, what it returns, who calls it).
  RuleReader.read(moves, steps)   the position after `moves`, read as the rule needs it: the threat list of both sides made concrete (threat_concrete), and for the side to move a line read for every unit of the other
                                  side and the goals (subgoal_units, subgoal_goals); `steps` says which of the rule's steps are on so that what a step off would not use is not read (a position read for the threats only is not read for
                                  goals; one read for the goals only drops the rows against the side to move). Returns {"lists", "goals", "position", "cost": seconds by part}.
  Mover75                         the movers of 78.67 plus the kinds `mr_rule`, `mr_threats`, `mr_goals`: the SAME search call as every player of 78.67 (`solve_equals_batch`, the moves held equal handed back, the search's own move
                                  first), then, for a position with two or more equal moves only, the rule of move_rule.py picks one of them. The held subgoal rides from move to move in the mover. The decision is logged.
  stake_heads()                   the kinds reading's small table the stake of a unit is read from (the 78.69 heads that read the kinds alone), saved once from the 78.69 run's own file.
  main                            stake-heads | short | match | report | stage-d.
Called by: nothing in play. The harness is tiebreak_measure.py (players, the settings of the matches) reused as it is; this file adds only the players.

  .venv/bin/python -m catspace.incubator.analyst.discovery.move_rule_play stake-heads | match --a rule --b control --set A --pairs 10 --tag short | report --tag T | stage-d --tag T
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import pickle
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import cs_engine as CE

from catspace.incubator.analyst.discovery import human_heads_data as HD
from catspace.incubator.analyst.discovery import move_rule as MR
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import subgoal_goals as SG
from catspace.incubator.analyst.discovery import subgoal_print as SP
from catspace.incubator.analyst.discovery import subgoal_units as SU
from catspace.incubator.analyst.discovery import subgoal_play_measure as PM
from catspace.incubator.analyst.discovery import threat_concrete as TC
from catspace.incubator.analyst.discovery import threat_list as TL
from catspace.incubator.analyst.discovery import threat_measure as TMS
from catspace.io import paths

OUT = ("move_rule_78_75",)
KIND = dict(TM.KIND, rule="mr_rule", threats="mr_threats", goals="mr_goals")
STEPS = {"mr_rule": (1, 2, 3), "mr_threats": (1, 3), "mr_goals": (2,)}
PLAYERS = TM.PLAYERS + ("rule", "threats", "goals")
REG: list = []
_RR = None


def out_root(tag: str = "full") -> Path:
    return Path(paths.experiment(*OUT, tag))


def stake_file() -> Path:
    return Path(paths.experiment(*OUT)) / "stake_heads.pkl"


# ------------------------------------------------------------------------------------------------------------------------------ the stake (the kinds reading of 78.69, kept small)

def stake_heads() -> Path:
    """Save the part of the 78.69 readings that the stake of a kind needs (the class `result`, the head on the kinds alone, each side): its linear models, its active flags and the order of the kinds."""
    d, R, _te = TMS.load_world(lambda m: None)
    keep = {"cols": list(d["rows"]["meta"]["kinds"]), "heads": {}, "models": {}}
    for rel in ("us", "them"):
        h = R.heads[(TMS.CLASS_NAME, rel)]
        keep["heads"][rel] = {"rel": h.rel, "active": list(h.active)}
        keep["models"][rel] = R.saved["models"][(TMS.CLASS_NAME, rel, "kinds")]
    f = stake_file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_bytes(pickle.dumps(keep))
    return f


class StakeReader:
    """The duck-typed `Readings` that `threat_measure.stake_of_kinds` reads through: only the class `result` on the kinds alone."""

    def __init__(self):
        k = pickle.loads(stake_file().read_bytes())
        self.cols = k["cols"]
        self.heads = {rel: SimpleNamespace(rel=v["rel"], active=v["active"]) for rel, v in k["heads"].items()}
        self.models = k["models"]
        self.remaining = None

    def read(self, cname: str, rel: str, which: str, X: np.ndarray, remaining: "np.ndarray | None" = None) -> np.ndarray:
        from catspace.incubator.analyst.discovery import frames_heads as FH
        return FH.side_chance(self.heads[rel], FH.table_linear(self.models[rel], X, np.arange(len(X))), remaining)


# ------------------------------------------------------------------------------------------------------------------------------ the reading of one position

class RuleReader:
    """One per process: the readers the rule's two lists are made by (the frame read with answers, the line reader with the measured record, the graph facts, the stake)."""

    def __init__(self):
        self.threads = TL.ThreatReader(1)
        self.record = TC.Record.load()
        self.lines = TC.Lines(self.record, threads=1)
        self.facts = SG.Facts(HD.VARIANT)
        self.editor = SP.GameEditor(HD.VARIANT)
        self.stake = StakeReader()
        self.cost = {"frame": 0.0, "stake": 0.0, "concrete": 0.0, "every": 0.0, "goals": 0.0, "n": 0}

    def read(self, moves: list, steps: tuple, cap: int = TM.PLY_CAP) -> "dict | None":
        t0 = time.perf_counter()
        res = self.threads.read([(None, list(moves))], [[len(moves)]])
        if not res["rows"]:
            return None
        raw = res["rows"][0]
        names = {k: res[k] for k in ("kinds", "roles", "places")}
        t1 = time.perf_counter()
        cols = self.stake.cols
        K = np.zeros((1, len(cols)), np.float32)
        for o, k, _p in raw["board"]:
            if names["kinds"][k] in cols:
                K[0, cols.index(names["kinds"][k])] += 1.0 if o == raw["mover"] else -1.0
        stake = TMS.stake_of_kinds(self.stake, K, np.array([max(1, cap - len(moves))], np.int64))[0]
        lst = TL.threat_rows(raw, names, SU.stake_closure(stake, cols))
        mover = raw["mover"]
        rk = TL.rank_high_of(lst["mover"] + lst["other"])
        for r in lst["mover"] + lst["other"]:
            r["big"] = bool(rk.get(id(r), False))
        t2 = time.perf_counter()
        use_threats, use_goals = bool({1, 3} & set(steps)), 2 in steps
        if not use_threats:
            lst["mover"] = []                                    # goals only: the rows against us are not read
        if not use_goals:
            lst["other"] = []                                    # threats only: the rows against them are not read
        item = {"start": None, "moves": list(moves), "raw": raw, "names": names, "list": lst}
        c = TC.concrete_lists([item], self.lines)[0]
        t3 = time.perf_counter()
        goals: list = []
        if use_goals:
            SU.every_unit([item], [[mover]], [c], self.facts, self.lines, [SU.stake_closure(stake, cols)])
            t4 = time.perf_counter()
            goals = SG.goals(item, mover, c, self.facts, self.editor)
            self.cost["every"] += t4 - t3
            self.cost["goals"] += time.perf_counter() - t4
        self.cost["frame"] += t1 - t0
        self.cost["stake"] += t2 - t1
        self.cost["concrete"] += t3 - t2
        self.cost["n"] += 1
        pos = MR.Position(self.facts, HD.VARIANT, None, list(moves), names["roles"], mover)
        return {"lists": c, "goals": goals, "position": pos, "item": item}


def reader75() -> RuleReader:
    global _RR
    if _RR is None:
        _RR = RuleReader()
    return _RR


# ------------------------------------------------------------------------------------------------------------------------------ the players

class Mover75(TM.Mover67):
    """The movers of 78.67 plus `mr_rule` (the rule), `mr_threats` (step 2 off) and `mr_goals` (steps 1 and 3 off). The reading is after the search and on the equal moves only, as the number of 78.67."""

    def __init__(self, spec: dict, seconds: float, seed: str, trace: str):
        super().__init__(spec, seconds, seed, trace)
        self.held = None
        self.rlog: list = []
        self.stats["rlog"] = self.rlog

    def search_clock(self) -> float:
        if self.charge and self.kind in STEPS:
            return TM.reading_clock(self.seconds, self.costs)
        return super().search_clock()

    def choose(self, text: "str | None", moves: list, legal: list) -> tuple:
        if self.kind not in STEPS:
            return super().choose(text, moves, legal)
        t0 = time.perf_counter()
        row = PM.equal_set(CE.solve_equals_batch([(OM.VARIANT, text, list(moves))], self.search_clock(), worth=dict(self.worth), ours_first=False, valuation="odds record", threads=1,
                                                 ledger_share=TM.LEDGER_SHARE)[0])
        t1 = time.perf_counter()
        mv, eq = row["move"], row["equals"]
        pick, st = mv, self.stats
        st["decisions"] += 1
        if mv is not None and len(eq) >= 2:
            st["ties"] += 1
            tied = [mv] + sorted(m for m in eq if m != mv)
            rd = reader75().read(list(moves), STEPS[self.kind])
            if rd is not None:
                pick, reason, held = MR.choose(rd["position"], tied, rd["lists"], rd["goals"], self.held, STEPS[self.kind])
                assert pick in tied
                self.held = held
                ranked = MR.rank_goals(rd["goals"])
                self.rlog.append({"ply": len(moves), "n_tied": len(tied), "branch": reason.branch, "pick": pick, "changed": int(pick != mv), "reason": str(reason)[:400],
                                  "held": (None if held is None else {k: (list(v) if isinstance(v, tuple) else v) for k, v in held.items()}),
                                  "threats": len(MR.threat_rows(rd["lists"])), "goals": [MR.compact(g) for g in ranked[:10]]})
            st["changed"] += pick != mv
            self.costs.append(time.perf_counter() - t1)
        st["log"].append((len(moves), len(eq), int(pick != mv)))
        t2 = time.perf_counter()
        st["search"] += t1 - t0
        st["read"] += t2 - t1
        st["wall"] += t2 - t0
        return pick, row["depth"]


def play75(spec: dict) -> dict:
    OM.Mover = Mover75
    TM.REG.clear()
    g = OM.play_game(spec)
    g["stats"] = [dict(m.stats) for m in TM.REG]
    return g


def run_games(specs: list, workers: int, out_file: Path, say=print) -> list:
    done = {}
    if out_file.exists():
        for line in out_file.read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                done[g["gid"]] = g
    todo = [s for s in specs if s["gid"] not in done]
    t0 = time.perf_counter()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    n_all, n0 = len(specs), len(done)
    with open(out_file, "a") as f:
        def keep(g: dict) -> None:
            done[g["gid"]] = g
            f.write(json.dumps(g) + "\n")
            f.flush()
            if len(done) % 10 == 0 or len(done) == n_all:
                el = time.perf_counter() - t0
                say(f"OBS progress {out_file.stem}: {len(done)} of {n_all} games, {el:.0f} s, about {el / max(1, len(done) - n0) * (n_all - len(done)):.0f} s left")
        if workers <= 1 or not todo:
            for s in todo:
                keep(play75(s))
        else:
            with multiprocessing.get_context("spawn").Pool(workers) as pool:
                for g in pool.imap_unordered(play75, todo, chunksize=1):
                    keep(g)
    return [done[s["gid"]] for s in specs]


def match_specs(a: str, b: str, set_name: str, pairs: int, seconds: float, scratch: Path, charge: bool = False) -> list:
    base = OM.players_of(TM.WORTH_GEN, OM.out_root() / "full")
    pa, pb = dict(base, kind=KIND[a], charge=charge), dict(base, kind=KIND[b], charge=charge)
    specs = []
    for seed, opening in TM.openings(set_name, pairs):
        for colour in (0, 1):
            specs.append({"gid": f"{a}_v_{b}:{set_name}{seed}:{colour}", "start": f"{set_name}{seed}", "colour": colour, "text": None, "opening": opening, "players": [pa, pb] if colour == 0 else [pb, pa],
                          "seconds": seconds, "max_plies": TM.PLY_CAP, "scratch": str(scratch)})
    return specs


def games_file(tag: str, a: str, b: str, set_name: str) -> Path:
    return out_root(tag) / "matches" / f"{a}_v_{b}_{set_name}.jsonl"


def lines_of(a: str, b: str, set_name: str, games: list, load: float, busy: float, wall: float, seconds: float, workers: int, charge: bool) -> list:
    """The OBS lines of 78.67's match, with the rule's logs taken off the stats (the logs are read by stage D)."""
    plain = [dict(g, stats=[{k: v for k, v in s.items() if k != "rlog"} for s in g["stats"]]) for g in games]
    out = TM.match_lines(a, b, set_name, plain, load, busy, wall, seconds, workers)
    return [x.replace("OBS tiebreak match", "OBS move_rule match" + (" CHARGED" if charge else " uncharged"), 1) for x in out]


# ------------------------------------------------------------------------------------------------------------------------------ stage D

def branch_lines(a: str, b: str, set_name: str, games: list) -> list:
    """How often each branch fired in the games of the rule's side, the share of the ties where it had a choice, and how long a goal was held."""
    names = {"1": "step 1: a kept answer to the highest threat", "1+held": "step 1: a kept answer that is also a step of the held goal", "2 held": "step 2: a step of the held goal",
             "2 other": "step 2: the held goal had no step, another goal's step", "2 first": "step 2: no goal held, the first-ranked goal's step", "3": "step 3: a big threat with no kept answer, refuting it held",
             "4": "step 4: nothing applies, the search's own move"}
    tot = {k: 0 for k in names}
    n_ties = n_dec = changed = 0
    ages: list = []
    run: dict = {}
    for g in games:
        for who, nm in ((0, a), (1, b)):
            if nm not in ("rule", "threats", "goals"):
                continue
            seat = g["colour"] if who == 0 else 1 - g["colour"]
            st = g["stats"][seat]
            n_dec += st["decisions"]
            cur = None
            for e in st.get("rlog", []):
                tot[e["branch"]] += 1
                n_ties += 1
                changed += e["changed"]
                h = e["held"]
                tgt = (h["origin"], tuple(h["target"])) if h else None
                if tgt != cur and cur is not None:
                    ages.append(run.pop(cur, 1))
                if tgt is not None:
                    run[tgt] = run.get(tgt, 0) + 1 if tgt == cur else 1
                cur = tgt
            if cur is not None:
                ages.append(run.pop(cur, 1))
    tag = f"{a} v {b} [set {set_name}]"
    out = [f"OBS move_rule branches {tag}: the rule's side: decisions {n_dec}, with two or more equal moves {n_ties} ({100 * n_ties / max(1, n_dec):.1f} pct of the decisions), choice differed from the search's own {changed} ({100 * changed / max(1, n_ties):.1f} pct of the ties)"]
    for k, nmx in names.items():
        out.append(f"OBS move_rule branches {tag}: {nmx}: {tot[k]} ({100 * tot[k] / max(1, n_ties):.1f} pct of the ties)")
    if ages:
        v = np.array(ages)
        out.append(f"OBS move_rule branches {tag}: a subgoal held for {len(v)} spells; decisions in a spell: mean {v.mean():.2f}, median {np.median(v):.0f}, longest {v.max()}, spells of one decision {int((v == 1).sum())}")
    else:
        out.append(f"OBS move_rule branches {tag}: no subgoal was held")
    return out


def goal_state(g: dict) -> dict:
    """A goal read back from its compact record, in the form move_rule's graph tests take."""
    return {"target": tuple(g["target"]), "taker": g["taker"], "owner": g["owner"], "stake": g.get("stake"), "source": g.get("source"),
            "facts": [{"name": f["name"], "status": f["status"], "vars": {k: tuple(v) if v else None for k, v in f["vars"].items()}, "cause": {"stoppers": [tuple(u) for u in f["stoppers"]]} if f["stoppers"] else None} for f in g["facts"]]}


def count_of(pf, side: str, kind: str) -> int:
    return sum(1 for (s, k, _p) in pf.units.values() if s == side and k == kind)


def outcome_of(goal: dict, facts: SG.Facts, moves: list, ply: int, horizon: int) -> "tuple | None":
    """After the decision at `ply`: (every checked unmet fact true within `horizon` plies, any, the target's side lost a unit of the target's kind within `horizon` plies, the facts that could be checked)."""
    n = len(moves)
    if ply >= n:
        return None
    gs = goal_state(goal)
    checked = [f for f in gs["facts"] if f["status"] in ("unmet", "unread") and f["name"] in MR.CHECKED]
    side, kind, _p = gs["target"]
    c0 = None
    all_true = any_true = False
    lost = False
    for t in range(ply + 1, min(n, ply + horizon) + 1):
        pf = facts.of(CE.state(HD.VARIANT, None, moves[:t])[0])
        if c0 is None:
            pf0 = facts.of(CE.state(HD.VARIANT, None, moves[:ply])[0])
            c0 = count_of(pf0, side, kind)
        if count_of(pf, side, kind) < c0:
            lost = True
        if checked:
            tr = [MR.fact_true(f, gs, pf) for f in checked]
            all_true = all_true or all(tr)
            any_true = any_true or any(tr)
    return (all_true if checked else None, any_true if checked else None, lost, len(checked))


def held_vs_not_lines(a: str, b: str, set_name: str, games: list, horizons: tuple = (2, 4, 6)) -> list:
    """Of the goals the rule's side held at a decision and the goals of the same decision it did not hold: how often the checked facts came true and the unit was then won, within the horizon."""
    facts = SG.Facts(HD.VARIANT)
    held_rows, other_rows, top3_rows = [], [], []
    for g in games:
        for who, nm in ((0, a), (1, b)):
            if nm not in ("rule", "threats", "goals"):
                continue
            seat = g["colour"] if who == 0 else 1 - g["colour"]
            for e in g["stats"][seat].get("rlog", []):
                h = e["held"]
                if not e["goals"]:
                    continue
                for rank, goal in enumerate(e["goals"]):
                    is_held = bool(h and h["origin"] == "goal" and tuple(h["target"]) == tuple(goal["target"]))
                    rec = (g["moves"], e["ply"], goal)
                    if is_held:
                        held_rows.append(rec)
                    else:
                        other_rows.append(rec)
                        if rank < 3:
                            top3_rows.append(rec)
    tag = f"{a} v {b} [set {set_name}]"
    out = []
    for hz in horizons:
        for label, rows in (("held", held_rows), ("not held (every other goal of those decisions)", other_rows), ("not held, among the first three ranked", top3_rows)):
            n = nf = nl = nc = 0
            for mv, ply, goal in rows:
                o = outcome_of(goal, facts, mv, ply, hz)
                if o is None:
                    continue
                n += 1
                nl += o[2]
                if o[0] is not None:
                    nc += 1
                    nf += o[0]
            out.append(f"OBS move_rule stage D {tag}: goals {label}, within {hz} plies of the decision: {n} goals; the checked unmet facts all true in {nf} of the {nc} with a fact the graph can tell ({100 * nf / max(1, nc):.1f} pct); "
                       f"a unit of the target's kind then lost by its side in {nl} ({100 * nl / max(1, n):.1f} pct)")
    return out


def emit(lines: list, out_file: "Path | None") -> None:
    TM.emit(lines, out_file)


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=("stake-heads", "match", "report", "stage-d"))
    ap.add_argument("--set", default="A", choices=("A", "B"))
    ap.add_argument("--a", default="rule", choices=PLAYERS)
    ap.add_argument("--b", default="control", choices=PLAYERS)
    ap.add_argument("--pairs", type=int, default=10)
    ap.add_argument("--seconds", type=float, default=TM.SECONDS)
    ap.add_argument("--workers", type=int, default=9)
    ap.add_argument("--charge", action="store_true", help="take the rule's reading out of the clock of the search")
    ap.add_argument("--tag", default="short")
    a = ap.parse_args(argv)
    root = out_root(a.tag)
    root.mkdir(parents=True, exist_ok=True)
    obs = root / "obs.txt"
    if a.stage == "stake-heads":
        print(stake_heads())
        return 0
    if a.stage == "match":
        load, busy = TM.wait_quiet(lambda s: emit([s], obs))
        t0 = time.perf_counter()
        games = run_games(match_specs(a.a, a.b, a.set, a.pairs, a.seconds, root / "scratch", a.charge), a.workers, games_file(a.tag, a.a, a.b, a.set), lambda s: emit([s], obs))
        emit(lines_of(a.a, a.b, a.set, games, load, busy, time.perf_counter() - t0, a.seconds, a.workers, a.charge), obs)
        emit(branch_lines(a.a, a.b, a.set, games), obs)
    elif a.stage == "stage-d":
        games = TM.read_games(games_file(a.tag, a.a, a.b, a.set))
        emit(held_vs_not_lines(a.a, a.b, a.set, games), obs)
    else:
        for p in sorted((root / "matches").glob("*.jsonl")):
            x, rest = p.stem.split("_v_")
            y, s = rest.rsplit("_", 1)
            emit(lines_of(x, y, s, TM.read_games(p), float("nan"), float("nan"), float("nan"), a.seconds, a.workers, a.charge) + branch_lines(x, y, s, TM.read_games(p)), None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
