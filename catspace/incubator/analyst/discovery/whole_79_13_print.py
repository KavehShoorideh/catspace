"""PLAN 79.13 PARTS C, D AND E, THE PRINTS (SYSTEM_SPEC 8 the whole is judged, 7 "Calculation and estimate, decided together" point 2; piece: the estimate per goal). A printing script: it may use the game's own words (the position text, the units) because it only prints what the engine logged. It judges nothing and writes no number the logs do not hold.
  sample TAG [--runs a,b]      PART D (1): the sample of the Explore games with the table: decisions, decisions with two or more equal moves, the spread of each input, and in particular the four gain inputs (share nonzero, mean, std within a set).
  cost TAG [--runs a,b]        PART C: the cost of reading OUR goals after each move of ours (from the notes of the step in the logs) and of the whole step, against the same step of 79.12's logs (read from them, quoted).
  twenty TAG RUN [--count N]   PART D (2) and (1): twenty decisions in which the last layer's move differs from the rule's pick, with the chances and the gains that decided (the goals each candidate serves, their chance, the fact about their goals after the move, the four gain inputs).
  shares TAG [--runs a,b] [--cap N]   PART E: the share of games that ended on a result, from the result lines of the logs.
INTERFACE: `games(log_dir)` is whole_79_12_print.games_of; every function returns the lines it prints. Called by the command line.
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import whole_79_12_print as P
from catspace.incubator.analyst.discovery import whole_79_13 as W13
from catspace.incubator.analyst.discovery import whole_79_12 as W12

GAIN = ("our_gain", "their_gain", "our_goal_gain", "their_goal_gain")


def sample_lines(tag: str, runs: list) -> list:
    n_games = n_dec = n_var = 0
    rows_all: list = []
    within: list = []
    names = None
    for run in runs:
        for decs, _res in P.games_of(W13.out_root(tag) / "logs" / run):
            n_games += 1
            for d in decs:
                n_dec += 1
                c = P.varied(d)
                if c is None:
                    continue
                n_var += 1
                names = c["inputs"]
                m = np.array([x["row"] for x in c["candidates"]])
                rows_all.append(m)
                within.append(m - m.mean(axis=0))
    t = f"OBS whole_79_13 sample {'+'.join(runs)}"
    out = [f"{t}: games {n_games}, decisions {n_dec}, decisions with two or more moves held equal {n_var} ({100 * n_var / max(1, n_dec):.1f} pct)"]
    if not rows_all:
        return out
    A, W = np.vstack(rows_all), np.vstack(within)
    out.append(f"{t}: candidate rows {len(A)}; spread of each input (mean, std over the rows, std of the part within a set, share of rows nonzero, share of sets in which it varies):")
    sets_vary = [np.abs(w).max(axis=0) > 1e-12 for w in within]
    for k, nm in enumerate(names):
        vary = np.mean([s[k] for s in sets_vary])
        out.append(f"OBS whole_79_13 sample input {nm}: mean {A[:, k].mean():.4f}, std {A[:, k].std():.4f}, within-set std {W[:, k].std():.4f}, nonzero {100 * np.mean(A[:, k] != 0):.1f} pct, varies within {100 * vary:.1f} pct of the sets")
    return out


def cost_lines(tag: str, runs: list) -> list:
    """The seconds of the step that reads their goals after each of our moves, and (from its note) the part spent on ours, over the decisions of the logs; the same step of 79.12's logs beside it."""
    def step_mean(root: Path, rs: list, with_ours: bool) -> tuple:
        tot = ours = 0.0
        n = n_ours = 0
        for run in rs:
            for decs, _res in P.games_of(root / "logs" / run):
                for d in decs:
                    for s in d["wake"]["steps"]:
                        if s["piece"] == "their goals after each move of ours":
                            tot += s["seconds"]
                            n += 1
                            m = re.search(r"ours read too, ([0-9.]+) s of it", s["note"])
                            if m:
                                ours += float(m.group(1))
                                n_ours += 1
        return tot / max(1, n), ours / max(1, n_ours), n, n_ours
    a = step_mean(W13.out_root(tag), runs, True)
    b = step_mean(W12.out_root("full"), ["train_self", "train_old"], False)
    t = "OBS whole_79_13 cost"
    return [f"{t}: the step 'their goals after each move of ours' with ours read too: mean {a[0]:.4f} s over {a[2]} decisions, of which ours {a[1]:.4f} s over {a[3]} decisions (79.13's logs {'+'.join(runs)})",
            f"{t}: the same step in 79.12's logs (theirs only): mean {b[0]:.4f} s over {b[2]} decisions"]


def shares_lines(tag: str, runs: list) -> list:
    out = []
    for run in runs:
        n = ended = 0
        plies = 0
        for _decs, res in P.games_of(W13.out_root(tag) / "logs" / run):
            if res is None:
                continue
            n += 1
            ended += not res.get("unfinished")
            plies += res["plies"]
        out.append(f"OBS whole_79_13 shares {run}: games {n}, ended on a result {ended} ({100 * ended / max(1, n):.1f} pct), mean plies {plies / max(1, n):.1f}")
    return out


def twenty(tag: str, run: str, count: int = 20) -> list:
    by: dict = defaultdict(list)
    for decs, _res in P.games_of(W13.out_root(tag) / "logs" / run):
        for d in decs:
            c = P.varied(d)
            if c and c["applied"] and c["played"] != c["fixed_pick"]:
                by[d["wake"]["log"]["why"].split(":")[0]].append((d, c))
    # the first twenty in game order, taken round robin over the kinds of sentence the log gave (one kind for a rule with one reason)
    picked: list = []
    classes = sorted(by)
    i = 0
    while len(picked) < count and any(by[k] for k in classes):
        k = classes[i % len(classes)]
        if by[k]:
            picked.append(by[k].pop(0))
        i += 1
    out = [f"OBS whole_79_13 twenty {run}: {sum(len(v) for v in by.values()) + len(picked)} decisions where the last layer's move differs from the rule's pick; {len(picked)} printed (the first in game order of each kind of reason: {classes})"]
    for n, (d, c) in enumerate(picked, 1):
        lg = d["wake"]["log"]
        mv = {m["move"]: m for m in lg["moves"]}
        out.append(f"--- decision {n}: game {d['gid']} ply {d['ply']} seat {d['seat']}\nposition: {d['position']}\nsearch's first move: {d['search_first']}; the rule would pick {c['fixed_pick']}; the last layer played {c['played']}; why: {lg['why']}\ntheir threat that stands: {lg.get('their_threat')}")
        for x in sorted(c["candidates"], key=lambda x: -x["sum"]):
            row = dict(zip(c["inputs"], x["row"]))
            g = ", ".join(f"{nm}={row[nm]:+.4g}" for nm in GAIN)
            ml = mv.get(x["move"], {})
            ta = (ml.get("their_after") or {}).get("said", "no fact read")
            mark = "PLAYED" if x["move"] == c["played"] else "rule's pick" if x["move"] == c["fixed_pick"] else ""
            out.append(f"   {x['move']} {mark}: {g}; answers_threat={row['answers_threat']:g}; serves {ml.get('serves') or '-'} with chance {ml.get('chance')}; {ta}")
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("sample", "cost", "twenty", "shares"))
    ap.add_argument("tag")
    ap.add_argument("--runs", default="train_self,train_old")
    ap.add_argument("--run", default="")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    runs = a.runs.split(",")
    lines = {"sample": lambda: sample_lines(a.tag, runs), "cost": lambda: cost_lines(a.tag, runs), "shares": lambda: shares_lines(a.tag, runs), "twenty": lambda: twenty(a.tag, a.run, a.count)}[a.what]()
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
