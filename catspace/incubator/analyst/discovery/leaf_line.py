"""PLAN 79.01 PART TWO, THE VALUE AT THE END OF THE LINE (SYSTEM_SPEC 4 the reasoning ends in a leaf, 5 and 6 the value read where the reasoning ends and fitted on results, 12 the debt of plan 78.84; piece: the estimator at the leaf, a MEASURE). No engine change, no player changed, no switch, nothing written to
rules/learned or data/derived; outputs under artifacts/experiments/leaf_79_01/<tag>/ (line/, lines.jsonl). It never calls the oracle (the labels are stored by leaf_ask.py and 79.00).

THE ESTIMATOR, read through its existing interface (human_heads_result.head_probs over the record of plan 78.59, `estimator_78_59/full/estimator_heads_human.json`, head `result_human_K`): the probability of win, draw and loss for the side to move from the kinds ours less theirs alone (no rating, no statement, no concept: the rating is
not known at a leaf, and the kinds-only head is the one head that is the same function of any position), fitted on the quiet rows of HUMAN GAMES' RESULTS (8337 games' rows, a multinomial logistic fit, no engine value and no oracle in it). The chance of the side to move is the win share plus half the draw share; the chance FOR THE MOVER AT THE ROOT is that
chance when the root mover is to move at the leaf and one less that chance when the other side is. The record's source, data and date are printed with the first line of the run.

WHAT IS READ FOR AN EQUAL MOVE. The line the search's value comes from: equal_trace.Tracer.line (the engine's line read, mode 9: the line that starts with the move, the best replies, resolved to a settled end), under the stack's judge (the human-result record: stack_causes.StackTracer), at D = the depth of the position's run of 79.00 (stored `depth`); the units each
side has by kind at its settled end (the estimator's inputs) and whose turn it is there. AND the units after the move alone (one ply, the position the choosers of 78.94 to 78.96 read): `one_after`. The same two reads for the keeping moves as for the losing ones (79.00's trace has the losing ones only, so every equal move of a firm-room position is read here).

INTERFACE (written before the bodies).
  Chance()                                    per process: the head of the record. `.of(inputs, leaf_is_root) -> float` the chance for the root mover from {kind: [side to move's count, other's]}; `.sym(inputs) -> float` the same with the side to move's own bonus taken out (inputs as the side to move at the leaf sees them; the result is for that side); `.source` the record's words.
  items_of(tag, src) -> [(rec, moves)]        the firm room's positions (leaf_firm.firm_vec of 79.00's labels and the second asking, the margin of margin.json), in the order of the sample's hash (pure given the files; tested through firm_vec).
  Lines()                                     per process: the stack's tracer. `.read(moves, depth, tied) -> [row]` per equal move {"ok", "names", "settled", "value", "cut", "end", "end_is_root", "one", "one_is_root"} (an engine call; one thread).
  run(tag, src, workers, limit, say) / load(root)   the Pool over the cores (node budgets, untimed), one file a chunk; lines.jsonl.
  Called by: leaf_measure (the tables, the judged positions), leaf_debt is independent, ladder_print --leaf01 (reads lines.jsonl).

  .venv/bin/python -m catspace.incubator.analyst.discovery.leaf_line lines --tag short --limit 40 | --tag full [--workers K]
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import human_heads_result as HR
from catspace.incubator.analyst.discovery import leaf_ask as LA
from catspace.incubator.analyst.discovery import leaf_firm as LF
from catspace.incubator.analyst.discovery import odds_measure as OM
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.incubator.analyst.discovery import stack_causes as SC
from catspace.incubator.analyst.discovery import stack_room as SR
from catspace.io import paths

RECORD = ("estimator_78_59", "full", "estimator_heads_human.json")
HEAD = "result_human_K"
CHUNK = 6
NAMES_SHOWN = 14


class Chance:
    """See the module doc."""

    def __init__(self):
        rec = json.loads(Path(paths.experiment(*RECORD)).read_text())
        self.head = rec["heads"][HEAD]
        self.kinds = [k.split(":", 1)[1] for k in self.head["inputs"]]
        self.source = f"{self.head['source']}; head {HEAD}: {self.head['what']}; data {self.head['data']['rows']} quiet rows ({self.head['data']['fit_rows']} fit rows, split {self.head['data']['split']}); date {self.head['date']}"

    def side_to_move(self, inputs: dict) -> float:
        x = np.array([[inputs[k][0] - inputs[k][1] for k in self.kinds]], float)
        p = HR.head_probs(self.head, x)[0]
        return float(p[0] + 0.5 * p[1])

    def of(self, inputs: dict, leaf_is_root: bool) -> float:
        c = self.side_to_move(inputs)
        return c if leaf_is_root else 1.0 - c

    def sym(self, inputs: dict) -> float:
        """The same head with the side to move's own bonus taken out: the mean of the root mover's chance read as the side to move and of one less the other side's chance read as the side to move (the kinds alone, nobody's turn). Added after the judged positions (see leaf_firm.rule_scores)."""
        a = self.side_to_move(inputs)
        b = self.side_to_move({k: [v[1], v[0]] for k, v in inputs.items()})
        return 0.5 * (a + 1.0 - b)


def items_of(tag: str, src: str = LA.SRC) -> list:
    root = LF.out_root(tag)
    margin = LF.load_margin(tag)
    recs = {r["id"]: r for r in LA.src_records(src)}
    games = {g["ord"]: g for g in FL.universe()}
    out = []
    for a in LA.load(root):
        r = recs[a["id"]]
        if LF.is_firm(LF.firm_vec(r, a, margin)):
            out.append((r, games[r["ord"]]["moves"][:r["ply"]]))
    return sorted(out, key=lambda t: PE.order_key(t[0]["ord"], t[0]["ply"]))


class Lines(SC.StackTracer):
    """See the module doc."""

    def one_after(self, moves: list, move: str) -> dict:
        names, raw = self.layer1.position_table([(None, list(moves) + [move])], 1)
        nk = len(names)
        t = np.frombuffer(raw, dtype=np.int16).reshape(len(moves) + 2, 2 + 2 * nk)[-1]
        return {n: [int(t[2 + i]), int(t[2 + nk + i])] for i, n in enumerate(names)}

    def read(self, moves: list, depth: int, tied: list) -> list:
        out = []
        for m in tied:
            ln = self.line(moves, m, depth)
            if not ln.get("ok"):
                out.append({"ok": False})
                continue
            out.append({"ok": True, "names": ln["names"][:NAMES_SHOWN], "settled": ln["settled"], "value": ln["value"], "cut": ln["cut"], "end": ln["inputs"], "end_is_root": ln["leaf_mover"] == ln["root_mover"],
                        "one": self.one_after(moves, m), "one_is_root": False})
        return out


_LN = None


def _init_worker() -> None:
    global _LN
    _LN = Lines()


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "line" / f"c{idx:05d}.json"
    if f.exists() and "\"error\"" not in f.read_text():                  # a chunk with a file and no error is not redone
        return idx, 0.0
    t0 = time.perf_counter()
    out = []
    for rec, moves in items:
        try:
            rows = _LN.read(moves, rec["depth"], rec["tied"])
            out.append({"id": rec["id"], "rows": rows})
        except Exception as e:                                       # a position the reading cannot take is counted, never silently
            out.append({"id": rec["id"], "error": f"{type(e).__name__}: {str(e)[:150]}"})
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def load(root: Path) -> list:
    return [json.loads(x) for x in (root / "lines.jsonl").read_text().splitlines() if x.strip()]


def run(tag: str, src: str, workers: int, limit: int, say) -> None:
    root = LF.out_root(tag)
    (root / "line").mkdir(exist_ok=True)
    items = items_of(tag, src)
    ch = Chance()
    say(f"the estimator, read through human_heads_result.head_probs: {ch.source}; inputs the kinds {ch.kinds} ours less theirs of the side to move")
    say(f"firm-room positions: {len(items)} (margin {LF.load_margin(tag):g}); reading {min(limit, len(items)) if limit else len(items)}; the line at D (stored depth of 79.00's run) under the human-result record {RP.worth_of(SR.RECORD)}; node budgets alone; {workers} workers; load average {os.getloadavg()[0]:.2f}")
    if limit:
        items = items[:limit]
    jobs = [(a // CHUNK, items[a:a + CHUNK], str(root)) for a in range(0, len(items), CHUNK)]
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for _i, _dt in pool.imap_unordered(_job, [(i, it, r) for i, it, r in jobs], chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    out = []
    for f in sorted((root / "line").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "lines.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    bad = sum(1 for r in out if "error" in r)
    say(f"read {len(out)} of {len(items)} positions in {time.perf_counter() - t0:.0f} s wall ({bad} errors); lines not read (an equal move whose line could not be read): {sum(1 for r in out if 'rows' in r for x in r['rows'] if not x['ok'])}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["lines"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--src", default=LA.SRC)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    run(a.tag, a.src, a.workers, a.limit, LA.Say(LF.out_root(a.tag), a.cmd))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
