"""PLAN 79.00 PART THREE, WHAT IS LEFT IN THE NEW ROOM (SYSTEM_SPEC 4 the equal reading, 5 the value and its measured records, 12 the debt of plan 78.84; piece: tactical reasoning). A MEASURE and nothing else: no player is changed, no switch is turned on in play, nothing is written to rules/learned
or data/derived; every output goes under artifacts/experiments/stack_79_00/<tag>/ (trace/, trace.jsonl). It never calls the oracle: it reads the classes stack_room stored. Game-free: a move is a word, a unit a kind word the engine hands back.

WHAT IS TRACED. In the stack's room (stack_room labels: positions whose equal moves differ in keeping the class), the class-losing equal moves (the stored class after the move is lower than before it), at most CAP_MOVES of a position (in the order of the equal set); the keeper is the first class-keeping
equal move. Each is traced exactly as 78.98 traced its stuck moves (equal_trace.Tracer: the audit at D = the depth the position's play-clock run finished, by node budgets alone; the longer reading two plies on; the line each value comes from, its exchange and the units at its settled end), with ONE difference that
is the stack's: the judge is the HUMAN-RESULT record (the stack's judge), not the odds record. The causes are 78.98's, decided before any count (equal_trace.cause_of): not held; (b) the pruned value too low; (c) a loss past the depth; (a) the judge's value the same at every depth read; unread.
THE LINE OF A CAUSE-(a) MOVE (first that applies, decided here before any count): no unit given or taken at all; a unit of the record's weakest kind given for nothing; units of other kinds given for nothing; an exchange of unlike units (the same number each way, different kinds); an exchange of like units; other.
Independently: whether the keeper's line reaches the same exchange (the same units given and taken), the position's value at D under, at or over nothing, and what the record makes of the exchange.
Not the same population as 78.98 (there: the moves stuck at five and twenty-five times the clock and with the prunings off, under the odds record); here every class-losing equal move of the room up to the cap, so the two sets of numbers are read side by side and never merged.

INTERFACE (written before the bodies).
  items_of(recs, cap_moves, limit) -> [item]       the class-losing equal moves of the room as trace items {"id", "moves", "depth", "stuck", "keeper", "band", "tag"} (pure given the labels and the games; tested).
  StackTracer()                                    equal_trace.Tracer with the human record's worths as the judge. `.trace(moves, depth, stuck, keeper) -> dict`.
  line_kind(exchange, worth) -> str                the kind of a cause-(a) line above (pure; tested).
  run(tag, cap_moves, limit, workers, say)  /  cmd_count(tag, say)     the Pool over the cores (untimed: node budgets), one file a chunk; the counts by cause and by kind of line, by band, 90 pct bands by games.
  Called by: the command line, ladder_print --stack79 (the cases of part three are not printed: the twenty judged positions are part two's).

  .venv/bin/python -m catspace.incubator.analyst.discovery.stack_causes trace --tag short --limit 40 | trace --tag full | count --tag full
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse
import json
import multiprocessing
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from catspace.incubator.analyst.discovery import equal_measure as EM
from catspace.incubator.analyst.discovery import equal_trace as ET
from catspace.incubator.analyst.discovery import future_labels as FL
from catspace.incubator.analyst.discovery import prior_equal as PE
from catspace.incubator.analyst.discovery import prior_measure as PMS
from catspace.incubator.analyst.discovery import record_play as RP
from catspace.incubator.analyst.discovery import stack_room as SR

BANDS = ("few", "mid", "many")
CAP_MOVES = 2
KINDS = ("no taking", "weakest kind for nothing", "other kinds for nothing", "unlike exchange", "like exchange", "other")


def out_root(tag: str) -> Path:
    return SR.out_root(tag)


def items_of(recs: list, games: dict, cap_moves: int = CAP_MOVES, limit: int = 0) -> list:
    """See the module doc. The room: some equal move keeps the class, some does not."""
    items = []
    for r in recs:
        keep = [PE.keeps(r["cls"], c) for c in r["cm"]]
        if not (any(keep) and not all(keep)) or r.get("depth") is None:
            continue
        keeper = next(m for m, k in zip(r["tied"], keep) if k)
        moves = games[r["ord"]]["moves"][:r["ply"]]
        for m, k in [(m, k) for m, k in zip(r["tied"], keep) if not k][:cap_moves]:
            items.append({"id": r["id"], "moves": moves, "depth": r["depth"], "stuck": m, "keeper": keeper, "band": r["band"], "tag": "room"})
    return items[:limit] if limit else items


class StackTracer(ET.Tracer):
    """equal_trace.Tracer with the stack's judge (the human-result record)."""

    def __init__(self):
        super().__init__()
        self.worth = RP.worth_of(SR.RECORD)


_TR = None


def _init_worker() -> None:
    global _TR
    _TR = StackTracer()


def _job(args: tuple) -> tuple:
    idx, items, root = args
    f = Path(root) / "trace" / f"c{idx:05d}.json"
    if f.exists():
        return idx, 0.0
    t0 = time.perf_counter()
    out = []
    for it in items:
        try:
            r = _TR.trace(it["moves"], it["depth"], it["stuck"], it["keeper"])
        except Exception as e:
            r = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        out.append(dict(r, id=it["id"], band=it["band"], tag=it.get("tag")))
    f.write_text(json.dumps(out))
    return idx, time.perf_counter() - t0


def load(root: Path) -> list:
    return ET.load(root)


def run(tag: str, cap_moves: int, limit: int, workers: int, say) -> None:
    root = out_root(tag)
    (root / "trace").mkdir(exist_ok=True)
    games = {g["ord"]: g for g in FL.universe()}
    all_items = items_of(SR.load(root), games, cap_moves)
    items = all_items[:limit] if limit else all_items
    say(f"{len(all_items)} class-losing equal moves of the room (at most {cap_moves} a position); tracing {len(items)}; D the depth the position's play-clock run finished; node budgets alone (no clock); the judge the human-result record {RP.worth_of(SR.RECORD)}; {workers} workers of {os.cpu_count()} cores; load average {os.getloadavg()[0]:.2f}")
    jobs = [(a // ET.CHUNK, items[a:a + ET.CHUNK], str(root)) for a in range(0, len(items), ET.CHUNK)]
    t0 = time.perf_counter()
    done = 0
    with multiprocessing.get_context("spawn").Pool(workers, initializer=_init_worker) as pool:
        for _i, _dt in pool.imap_unordered(_job, jobs, chunksize=1):
            done += 1
            if done % 10 == 0 or done == len(jobs):
                say(f"progress: {done} of {len(jobs)} chunks; {time.perf_counter() - t0:.0f} s; load average {os.getloadavg()[0]:.2f}")
    out = []
    for f in sorted((root / "trace").glob("c*.json")):
        out += json.loads(f.read_text())
    (root / "trace.jsonl").write_text("\n".join(json.dumps(r) for r in out) + "\n")
    say(f"traced {len(out)} of {len(items)} in {time.perf_counter() - t0:.0f} s wall; load average at the end {os.getloadavg()[0]:.2f}")


def line_kind(ex: dict, worth: dict) -> str:
    """The kind of a line's exchange (the mover's units given and taken, kind words), the first that applies; see the module doc."""
    gave, took = list(ex["gave"]), list(ex["took"])
    if not gave and not took:
        return "no taking"
    weakest = min(worth, key=worth.get)
    if gave and not took:
        return "weakest kind for nothing" if all(k == weakest for k in gave) else "other kinds for nothing"
    if len(gave) == len(took) and gave:
        return "like exchange" if sorted(gave) == sorted(took) else "unlike exchange"
    return "other"


def cmd_count(tag: str, say) -> None:
    root = out_root(tag)
    rows = [r for r in load(root) if "error" not in r]
    errs = [r for r in load(root) if "error" in r]
    worth = RP.worth_of(SR.RECORD)
    order = " < ".join(k for k, _w in sorted(worth.items(), key=lambda kv: kv[1]))
    say(f"RULE as 78.98 (a root move is held equal iff its null-window answer at the finished depth D is at least the value of the position, margin zero) under the stack's judge, the human-result record {worth} (order by worth {order})")
    say(f"the traced class-losing equal moves of the room: {len(rows)} ({len(errs)} errors); one thread's seconds a move traced {np.mean([r['seconds'] for r in rows]):.1f}" if rows else "no rows")
    if not rows:
        return
    game = np.array([int(r["id"].split("_")[0]) for r in rows])
    band = np.array([r["band"] for r in rows])
    cause = np.array([r["cause"] for r in rows])
    for c in ET.CAUSES:
        say(f"cause {c}: all {PMS.fmt(PMS.share((cause == c).astype(float), game))} pct ({int((cause == c).sum())} of {len(rows)}); " + "; ".join(
            f"{b} {PMS.fmt(PMS.share((cause == c)[band == b].astype(float), game[band == b]))} ({int(((cause == c) & (band == b)).sum())} of {int((band == b).sum())})" for b in BANDS if int((band == b).sum()) > 0))
    held = cause != "not held"
    for c in ET.CAUSES[1:]:
        say(f"cause {c} among the moves the rule held at D in the trace ({int(held.sum())}): {PMS.fmt(PMS.share((cause == c)[held].astype(float), game[held]))} pct")
    b_rows = [r for r in rows if r["cause"] == "b"]
    if b_rows:
        say(f"cause b ({len(b_rows)} moves), the prunings-off reading at D puts the move under its own best in {sum(1 for r in b_rows if (r.get('deep') or {}).get('other_d') is not None and (r['deep'].get('best_d') is not None) and r['deep']['other_d'] < r['deep']['best_d'])}; "
            f"the exact value under v in {sum(1 for r in b_rows if r['stuck_row']['exact'] is not None and r['stuck_row']['exact'] < r['value'])}")
    a_rows = [r for r in rows if r["cause"] == "a" and r.get("stuck_line", {}).get("ok")]
    if a_rows:
        kinds = [line_kind(r["stuck_line"]["exchange"], worth) for r in a_rows]
        ag = np.array([int(r["id"].split("_")[0]) for r in a_rows])
        ab = np.array([r["band"] for r in a_rows])
        for k in KINDS:
            v = np.array([x == k for x in kinds], float)
            say(f"cause a, the line shows {k}: all {PMS.fmt(PMS.share(v, ag))} pct ({int(v.sum())} of {len(a_rows)}); " + "; ".join(f"{b} {int(v[ab == b].sum())} of {int((ab == b).sum())}" for b in BANDS))
        sig = Counter((" ".join(r["stuck_line"]["exchange"]["gave"]) or "-", " ".join(r["stuck_line"]["exchange"]["took"]) or "-") for r in a_rows)
        say("cause a, the units the mover gave and took along the line (gave | took: moves): " + "; ".join(f"{g} | {t}: {n}" for (g, t), n in sig.most_common(10)))
        later = [(r["keeper_line"]["exchange"] == r["stuck_line"]["exchange"]) and bool(r["stuck_line"]["exchange"]["gave"] or r["stuck_line"]["exchange"]["took"]) for r in a_rows if (r.get("keeper_line") or {}).get("ok")]
        say(f"cause a, the class-keeping equal move reaches the same exchange (the same units given and taken along its line): {sum(later)} of {len(later)} lines read ({100 * sum(later) / max(1, len(later)):.1f} pct)")
        say(f"cause a, the value of the position at D: under nothing in {sum(1 for r in a_rows if r['value'] < 0)}, nothing in {sum(1 for r in a_rows if r['value'] == 0)}, over nothing in {sum(1 for r in a_rows if r['value'] > 0)}; "
            f"the value of the line the stuck move's value comes from is the value of the position in {sum(1 for r in a_rows if r['stuck_line'].get('value') == r['value'])} of {len(a_rows)}")
        nets = Counter(EM.sign_word(EM.net_by_record(r["stuck_line"]["exchange"], worth)) for r in a_rows)
        say("cause a, what the human-result record makes of the exchange along the line: " + "; ".join(f"{w}: {n}" for w, n in nets.most_common()))
        unlike = [r for r in a_rows if len(r["stuck_line"]["exchange"]["gave"]) == len(r["stuck_line"]["exchange"]["took"]) > 0 and r["stuck_line"]["exchange"]["gave"] != r["stuck_line"]["exchange"]["took"]]
        say(f"cause a, exchanges of the same number of units of different kinds ({len(unlike)} moves): the record makes it a gain for the mover in {sum(1 for r in unlike if EM.net_by_record(r['stuck_line']['exchange'], worth) > 0)}, a loss in {sum(1 for r in unlike if EM.net_by_record(r['stuck_line']['exchange'], worth) < 0)}, nothing in {sum(1 for r in unlike if EM.net_by_record(r['stuck_line']['exchange'], worth) == 0)}")
    unread = [r for r in rows if r["cause"] == "unread"]
    if unread:
        say(f"unread ({len(unread)}): the node budget stopped the prunings-off reading at D+{ET.DEEP}")


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["trace", "count"])
    ap.add_argument("--tag", default="short")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cap-moves", type=int, default=CAP_MOVES)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args(argv)
    say = SR.Say(out_root(a.tag), a.cmd)
    if a.cmd == "trace":
        run(a.tag, a.cap_moves, a.limit, a.workers, say)
    else:
        cmd_count(a.tag, say)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
