"""PLAN 79.15 PART E, THE TWENTY (SYSTEM_SPEC 8 the whole is judged, not the piece; 1 and 11 the judge reads the printed reasoning against the board; piece: the wake). A PRINTING SCRIPT: it may use the game's own words (the position text, the units) because it only prints what the engine logged.
It judges nothing and writes no number the logs do not hold.
  twenty TAG RUN [--count 20] [--seed N]     twenty decisions from twenty DIFFERENT games of the logged match RUN (79.14 drew all twenty from one game, so they were a look and not a sample): from each game, in a seeded random order of its decisions from ply 10 on, the first decision in which the wake's move differs
                                             from the old player's at the same position (the old player at the clock of the equal wall, from `scale_<share>.json`); the board, the reasons the wake logged (the reason of mode Gain included), the first move's probe and the moves held equal, the old player's move and depth.
INTERFACE: `twenty_lines(root, run, count, world, old_clock, seed)` returns the lines it prints. Called by the command line.
  .venv/bin/python -m catspace.incubator.analyst.discovery.whole_79_15_print twenty full s1_v_old_A --share s1
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import whole_79_14 as W14
from catspace.incubator.analyst.discovery import whole_79_15 as W15


def games_of(log_dir: Path) -> list:
    out = []
    for p in sorted(Path(log_dir).glob("*.jsonl.gz")):
        decs = []
        with gzip.open(p, "rt") as f:
            for line in f:
                r = json.loads(line)
                if r["kind"] == "decision":
                    decs.append(r)
        out.append((p.stem, decs))
    return out


def board_text(position: str) -> str:
    try:
        return str(chess.Board(position))
    except Exception:
        return position


def twenty_lines(root: Path, run: str, count: int, world, old_clock: float, seed: int = 7915, tries: int = 12) -> list:
    rng = random.Random(seed)
    games = games_of(root / "logs" / run)
    rng.shuffle(games)
    out = [f"OBS whole_79_15 twenty {run}: one decision from each of {count} different games in which the wake's move differs from the old player's at the same position (the old player at {old_clock:.4f} s), drawn at random (seed {seed}) from the decisions from ply 10 on; "
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
            out += [f"=== {k}. game {d['gid']} ply {d['ply']}: the wake plays {d['played']}, the old player {o['move']} (depth {o['depth']}, value {o['value']}); the wake's search: depth {w['depth']}, value {w['value']}, its first move {w.get('first')}, held equal ({lg.get('equal_from')}) {' '.join(w['equal'][:8])}",
                    board_text(d["position"]), f"    position: {d['position']}",
                    f"    THE WAKE'S REASON: {lg['why']}", f"    their largest standing threat: {lg['their_threat'] or 'none'}; named for us: {' '.join(w['named_mine'][:8]) or '-'}; for them: {' '.join(w['named_theirs'][:6]) or '-'}",
                    f"    the equal reading: first move lowered {lg.get('first_lowered')}, unverified {lg.get('equal_unverified')}; {'; '.join(n for n in lg['notes'] if 'lower' in n)[:240]}"]
            for m in lg["moves"]:
                if m["move"] in (d["played"], o["move"], w.get("first")):
                    out.append(f"    move {m['move']}: held equal {m['equal']}, named {m['named']}, probe {m['value']}, serves {m['serves'][:2] or '-'}, answers {m['answers'][:2] or '-'}, chance {m['chance']}")
            out.append("    TIME BY STEP: " + "; ".join(f"{s['step']} {s['piece'].split(':')[0][:40]} {s['seconds']:.3f}" for s in w["steps"] if s["step"] < 20))
            break
    return out


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("which", choices=("twenty",))
    ap.add_argument("tag")
    ap.add_argument("run")
    ap.add_argument("--share", default="s1")
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7915)
    a = ap.parse_args(argv)
    root = W15.out_root(a.tag)
    scale = json.loads((root / f"scale_{a.share}.json").read_text())["scale"]
    lines = twenty_lines(root, a.run, a.count, W14.world(), scale * 0.2, a.seed)
    text = "\n".join(lines)
    print(text)
    (root / f"twenty_{a.run}_print.txt").write_text(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
