"""PLAN 79.12 PARTS C AND E, THE PRINTS (SYSTEM_SPEC 8 the whole is judged, 7 "Calculation and estimate, decided together"; piece: the wake's last layer). A printing script: it may use the game's own words (the position text, the pieces) because it only READS logs and prints.
  sample TAG [--runs a,b]   PART C: the sample of the varied games: games, decisions, decisions with two or more equal moves, the spread of each input over their candidates (over the whole rows and within a set), the propensity,
                            and the ENDGAME COVERAGE (the units left on the board at each such decision, in bins, and the share of games that reached the last bin).
  steps TAG [--runs a,b]    PART A (3): the mean wall of each step of section 8 over the decisions of the logs: where the ply went.
  twenty TAG RUN            PART E: twenty decisions in which the chooser's move differs from the fixed rule's, round robin over the reason classes, printed with the inputs and the weighted sums that decided.
INTERFACE: `decisions(log_dir)` yields (decision record, result record or None) for every game log of a folder; `units_on_board(position)`; `sample_lines(...)`; `step_lines(...)`; `twenty(...)`. Called by the command line and the tests.
"""
from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import whole_79_12 as W12
from catspace.io import paths

BINS = ((0, 6, "6 or fewer"), (7, 12, "7 to 12"), (13, 20, "13 to 20"), (21, 99, "21 or more"))      # units left on the board: the bins of the endgame coverage


def read_game(p: Path) -> tuple:
    """([decision records], result record or None) of one game log."""
    decs, res = [], None
    with gzip.open(p, "rt") as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "decision":
                decs.append(r)
            else:
                res = r
    return decs, res


def games_of(log_dir: Path) -> list:
    return [read_game(p) for p in sorted(Path(log_dir).glob("*.jsonl.gz"))]


def units_on_board(position: str) -> int:
    """The units on the board of a position text: the letters of its first field (a descriptive count of a SAMPLE for the coverage print; never an input of the chooser)."""
    return sum(1 for c in position.split()[0] if c.isalpha())


def mover_result(decision: dict, res: "dict | None") -> "float | None":
    """The game's result from the side of the mover of this decision (1 a win, 0.5 a draw or an unfinished game, 0 a loss); None where the game has no result line."""
    if res is None:
        return None
    white = 0.5 if res.get("outcome") is None else res["outcome"] / 100.0
    return white if decision["seat"] == 0 else 1.0 - white


def varied(decision: dict) -> "dict | None":
    """The chooser record of a decision with two or more candidates, else None."""
    c = decision["wake"]["log"].get("chooser")
    return c if c and len(c["candidates"]) >= 2 else None


def sample_lines(tag: str, runs: list) -> list:
    out: list = []
    names = None
    rows_all: list = []
    within: list = []
    props: list = []
    bins = Counter()
    n_games = n_dec = n_var = n_played_in = 0
    reached = 0
    for run in runs:
        for decs, res in games_of(W12.out_root(tag) / "logs" / run):
            n_games += 1
            last = False
            for d in decs:
                n_dec += 1
                c = varied(d)
                if c is None:
                    continue
                n_var += 1
                names = c["inputs"]
                m = np.array([x["row"] for x in c["candidates"]])
                rows_all.append(m)
                within.append(m - m.mean(axis=0))
                props.append(c["propensity"])
                n_played_in += c["played"] in [x["move"] for x in c["candidates"]]
                u = units_on_board(d["position"])
                for lo, hi, nm in BINS:
                    if lo <= u <= hi:
                        bins[nm] += 1
                        last = last or nm == BINS[0][2]
            reached += last
    out.append(f"OBS whole_79_12 sample {'+'.join(runs)}: games {n_games}, decisions {n_dec}, decisions with two or more moves held equal {n_var} ({100 * n_var / max(1, n_dec):.1f} pct); the move played is among the candidates in {n_played_in}; mean propensity {np.mean(props) if props else float('nan'):.4f}")
    if not rows_all:
        return out
    A = np.vstack(rows_all)
    W = np.vstack(within)
    out.append(f"OBS whole_79_12 sample {'+'.join(runs)}: candidate rows {len(A)}; spread of each input (mean, std over the rows, std of the part within a set, share of rows nonzero):")
    for k, nm in enumerate(names):
        out.append(f"OBS whole_79_12 sample input {nm}: mean {A[:, k].mean():.4f}, std {A[:, k].std():.4f}, within-set std {W[:, k].std():.4f}, nonzero {100 * np.mean(A[:, k] != 0):.1f} pct")
    tot = sum(bins.values())
    out.append(f"OBS whole_79_12 sample endgame coverage {'+'.join(runs)} (units left on the board at the varied decisions): " + ", ".join(f"{nm} {bins[nm]} ({100 * bins[nm] / max(1, tot):.1f} pct)" for _lo, _hi, nm in BINS)
               + f"; games with a varied decision at 6 units or fewer: {reached} of {n_games}")
    return out


def step_lines(tag: str, runs: list) -> list:
    tot: dict = defaultdict(float)
    cnt: Counter = Counter()
    n = 0
    for run in runs:
        for decs, _res in games_of(W12.out_root(tag) / "logs" / run):
            for d in decs:
                n += 1
                seen: dict = defaultdict(float)
                for s in d["wake"]["log"].get("step_seconds", []):
                    seen[(s["step"], s["piece"])] += s["seconds"]
                for k, v in seen.items():
                    tot[k] += v
                    cnt[k] += 1
    out = [f"OBS whole_79_12 steps {'+'.join(runs)}: mean wall of each step of section 8 over {n} decisions (seconds; the share of the whole decision; the decisions in which the step ran):"]
    whole = sum(v for (st, _p), v in tot.items() if st == 0) or 1.0
    for (st, piece), v in sorted(tot.items()):
        out.append(f"OBS whole_79_12 step {st} {piece}: mean {v / max(1, n):.4f} s, {100 * v / whole:.1f} pct of the whole decision, ran in {cnt[(st, piece)]} decisions")
    return out


def twenty(tag: str, run: str, count: int = 20) -> list:
    """Twenty decisions where the chooser's move differs from the fixed rule's pick, round robin over the reason classes of the log; each printed with the position, the candidates, their inputs and weighted sums, the weights that decided."""
    from catspace.incubator.analyst.discovery import whole_play as WP
    by: dict = defaultdict(list)
    for decs, res in games_of(W12.out_root(tag) / "logs" / run):
        for d in decs:
            c = varied(d)
            if c and c["applied"] and c["played"] != c["fixed_pick"]:
                by[WP.reason_of(d["wake"]["log"]["why"])].append((d, c))
    picked: list = []
    classes = sorted(by)
    i = 0
    while len(picked) < count and any(by[k] for k in classes):
        k = classes[i % len(classes)]
        if by[k]:
            picked.append(by[k].pop(0))
        i += 1
    out = [f"OBS whole_79_12 twenty {run}: {sum(len(v) for v in by.values()) + len(picked)} decisions where the chooser's move differs from the rule's pick; {len(picked)} printed, round robin over the classes {classes}"]
    for n, (d, c) in enumerate(picked, 1):
        out.append(f"--- decision {n}: game {d['gid']} ply {d['ply']} seat {d['seat']}\nposition: {d['position']}\nsearch's first move: {d['search_first']}; the rule would pick {c['fixed_pick']}; the chooser played {c['played']}; why: {d['wake']['log']['why']}")
        for x in sorted(c["candidates"], key=lambda x: -x["sum"]):
            nz = ", ".join(f"{nm}={v:g}" for nm, v in zip(c["inputs"], x["row"]) if v)
            out.append(f"   {x['move']}: weighted sum {x['sum']:.4f}; inputs {nz or 'all nought'}")
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("what", choices=("sample", "steps", "twenty"))
    ap.add_argument("tag")
    ap.add_argument("--runs", default="train_self,train_old")
    ap.add_argument("--run", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    runs = a.runs.split(",")
    lines = sample_lines(a.tag, runs) if a.what == "sample" else step_lines(a.tag, runs) if a.what == "steps" else twenty(a.tag, a.run)
    text = "\n".join(lines)
    print(text)
    if a.out:
        Path(a.out).write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
