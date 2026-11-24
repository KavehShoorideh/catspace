"""PLAN 79.51 PART A, WHAT THE WAKE SAYS TODAY ON THE FIXTURE (SYSTEM_SPEC 7 "Future prediction: one channel, two sub-layers"; piece: tactical reasoning, opportunity policy; ledger row 50). A PRINTING SCRIPT: it may use the game's own
words (the position text) because it only prints. It asks the INSTALLED binding (main's, read-only, nothing rebuilt) for the wake of config abcd of 79.16 (the same pieces, the same tables as the measure) at the position and the clocks
given (two single decisions, no match, no clock of games) and prints the decision's own log: the move, the reason, their largest standing threat, the threat list's rows, the goals with their unmet facts, the moves named and the candidate moves.

  .venv/bin/python -m catspace.incubator.analyst.discovery.future_79_51_today_print [--position TEXT] [--seconds 0.2,2.0]
INTERFACE: `today(position, seconds) -> list[str]` the lines it prints. Called by the command line.
"""
from __future__ import annotations

from catspace.incubator.analyst.discovery import tiebreak_measure as TM          # first: it sets one thread a process before numpy is loaded

import argparse

from catspace.incubator.analyst.discovery import whole_79_14 as W14
from catspace.incubator.analyst.discovery import whole_79_16 as W16

FIXTURE = "6k1/pr2q1pp/1p2pp2/1Qp1nr2/7B/6P1/PP2PP1P/R4RK1 b - - 0 1"


def today(position: str, seconds: float) -> list:
    r = W14.Runner(W14.world(), W16.CONFIGS["abcd"])
    d = r.fresh(position, seconds)
    log = d.get("log") or {}
    out = [f"OBS future_79_51 today: wake abcd at {seconds} s on {position}: move {d.get('move')}, value {d.get('value')}, depth {d.get('depth')}, nodes {d.get('nodes')}, wall {d.get('wall'):.3f} s",
           f"  reason: {log.get('why')}", f"  their largest standing threat: {log.get('their_threat')}", f"  chosen goal: {log.get('chosen_goal')}", f"  far goal: {log.get('far_goal')}"]
    for row in log.get("rows", []):
        out.append(f"  row [{row['list']}] {row['unit']} class {row['class']} verdict {row['verdict']} net {row['net']} line {' '.join(row['line'])}")
    for g in log.get("goals", []):
        out.append(f"  goal [{g['side']}] on {g['target']} source {g['source']} unmet {g['unmet']} stoppers {g['stoppers']} fixes {[(f['kind'], f['moves'][:6]) for f in g['fixes']]} chance {g['chance']}")
    for m in log.get("moves", [])[:12]:
        out.append(f"  move {m['move']}: value {m['value']} equal {m['equal']} named {m['named']} serves {m['serves']} answers {m['answers']} line {' '.join(m['line'][:6])}")
    out.append(f"  named mine {[k['move'] for k in log.get('named_mine', [])]} named theirs {[k['move'] for k in log.get('named_theirs', [])]} notes {log.get('notes')}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--position", default=FIXTURE)
    ap.add_argument("--seconds", default="0.2,2.0")
    a = ap.parse_args(argv)
    for sec in [float(x) for x in a.seconds.split(",")]:
        print("\n".join(today(a.position, sec)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
