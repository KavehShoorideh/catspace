"""PLAN 79.16 PART E, THE TWENTY AND THE TABLES (SYSTEM_SPEC 8 the whole is judged, not the piece; 1 and 11 the judge reads the printed reasoning against the board; piece: the wake). A PRINTING SCRIPT: it may use the game's own words (the position text, the units) because it only prints; it judges nothing and writes no number the logs do not hold.
  twenty TAG RUN --config abcd --seconds 2.0 [--count 20] [--seed N]   twenty decisions from twenty DIFFERENT games of the logged match RUN (seeded random order of a game's decisions from ply 10) in which the wake's move differs from the old player's at the same position (the old player at the clock of the equal wall,
                                             from `scale_<config>_<seconds>.json`); for each: the board, the wake's reason, its largest standing threat, the frames BUILT and the rows read in that decision (the steps' notes), the reading on of the first move, the moves' probes and what they serve, the time by step.
  tables TAG [--runs a,b,...]                the per-match lines of the logs (means a decision: depth, frames built, rows read, the wall by step, the reading on of a first move; the count of lowered first moves).
INTERFACE: `twenty_lines(root, run, count, world, old_clock, seed)` returns the lines it prints. Called by the command line.
  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_16_print twenty full abcd_2.0_v_old_A --config abcd --seconds 2.0
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from catspace.incubator.analyst.discovery import lowered_first as LF
from catspace.incubator.analyst.discovery import whole_79_14 as W14
from catspace.incubator.analyst.discovery import whole_79_15_print as P15
from catspace.incubator.analyst.discovery import whole_79_16 as W16


def decision_asked(w: dict) -> str:
    """What the steps of one decision asked, from its notes: frames built and read, rows, questions, the threat list's rows."""
    out = []
    for s in w["steps"]:
        if s["piece"] == "frames and the question read":
            out.append(f"frame read: {s['note']}")
        elif s["piece"] == "part of the frame read: the frames built":
            out.append(f"frames built: {s['note']}")
        elif s["piece"] == "threat list of both sides":
            out.append(f"threat list: {s['note'][:150]}")
    return "; ".join(out) or "no frame read"


def twenty_lines(root: Path, run: str, count: int, world, old_clock: float, seed: int = 7916, tries: int = 12) -> list:
    rng = random.Random(seed)
    games = P15.games_of(root / "logs" / run)
    rng.shuffle(games)
    out = [f"OBS whole_79_16 twenty {run}: one decision from each of {count} different games in which the wake's move differs from the old player's at the same position (the old player at {old_clock:.4f} s), drawn at random (seed {seed}) from the decisions from ply 10 on; "
           "the board is as the engine logged it (white to move is up the board, upper case white)"]
    k = 0
    for gid, decs in games:
        if k >= count:
            break
        cand = [d for d in decs if d["ply"] >= 10]
        rng.shuffle(cand)
        for d in cand[:tries]:
            o = world.old(d["position"], old_clock)
            if o["move"] == d["played"]:
                continue
            k += 1
            w, lg = d["wake"], d["wake"]["log"]
            ff = lg.get("failed_first")
            out += [f"=== {k}. game {d['gid']} ply {d['ply']}: the wake plays {d['played']}, the old player {o['move']} (depth {o['depth']}, value {o['value']}); the wake's search: depth {w['depth']}, value {w['value']}, its first move {w.get('first')}, held equal {w['equal'][:6]}, wall {d['wall']:.3f} s",
                    P15.board_text(d["position"]), f"    position: {d['position']}",
                    f"    THE WAKE'S REASON: {lg['why']}", f"    their largest standing threat: {lg['their_threat'] or 'none'}; named for us: {' '.join(w['named_mine'][:8]) or '-'}; for them: {' '.join(w['named_theirs'][:6]) or '-'}",
                    f"    WHAT THE STEPS ASKED: {decision_asked(w)}",
                    f"    the first move asked again: {ff if ff else 'no (it finished its depth and held, or the game was decided)'}",
                    f"    the equal reading: first move lowered {lg.get('first_lowered')}, unverified {lg.get('equal_unverified')}, set made from {lg.get('equal_from')}; goals read {len(lg.get('goals', []))}, threat rows {len(lg.get('rows', []))}, chosen goal {(lg.get('chosen_goal') or {}).get('target')} ({(lg.get('chosen_goal') or {}).get('unmet')})"]
            for m in lg["moves"]:
                if m["move"] in (d["played"], o["move"], w.get("first")):
                    out.append(f"    move {m['move']}: held equal {m['equal']}, named {m['named']}, probe {m['value']}, serves {m['serves'][:2] or '-'}, answers {m['answers'][:2] or '-'}, chance {m['chance']}; line {' '.join(m['line'][:6])}")
            out.append("    TIME BY STEP: " + "; ".join(f"{s['step']} {s['piece'].split(':')[0][:40]} {s['seconds']:.3f}" for s in w["steps"] if s["step"] < 20))
            break
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=("twenty", "tables"))
    ap.add_argument("tag")
    ap.add_argument("run", nargs="?", default="")
    ap.add_argument("--config", default="abcd")
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7916)
    ap.add_argument("--runs", default="")
    a = ap.parse_args(argv)
    root = W16.out_root(a.tag)
    if a.which == "twenty":
        scale = json.loads((root / f"scale_{a.config}_{a.seconds}.json").read_text())["scale"]
        text = "\n".join(twenty_lines(root, a.run, a.count, W14.world(), scale * a.seconds, a.seed))
        print(text)
        (root / f"twenty_{a.run}_print.txt").write_text(text)
        return 0
    lines = []
    for r in a.runs.split(","):
        if r:
            lines.append(W16.asked_line(r, W16.asked16(root / "logs" / r)))
            lines.append(LF.line(r, LF.count(root / "logs" / r)))
    print("\n".join(lines))
    (root / "tables_print.txt").write_text("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
