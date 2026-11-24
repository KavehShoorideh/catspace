"""PLAN 79.51 PART H, THE TWO POSITION LISTS (SYSTEM_SPEC 7 "Future prediction: one channel, two sub-layers"; piece: tactical reasoning, opportunity policy; ledger row 50). A PRINTING SCRIPT: it
may use the game's own words (the position text) because it only writes lists of positions for the Rust counting test (rust/engine/decision/tests/future_count.rs). No oracle is called, no
player id is written.

  puzzles300.txt  the 300 puzzles of the measure (the same rows as 79.09 to 79.16: `ledger_puzzles.dev_rows(314, offset=11)[14:]`): the position the solver faces (the puzzle's first move played)
                  and the first move of the solution, tab separated: position, first move in the move text of the list, the puzzle id.
  games200.txt    200 positions of our own logged games (the decision logs of 79.16, config abcd at 0.2 s, sets A, read only from the folder given): one decision from each of 200 game files, drawn
                  at random (seed 7951) from the decisions from ply 10 on; tab separated: position, the game id, the ply.

  .venv/bin/python -m catspace.incubator.analyst.discovery.future_79_51_lists_print [--logs DIR]
INTERFACE: `main(argv)` writes the two files under artifacts/experiments/whole_79_51/ (through catspace.io.paths) and prints OBS lines with the counts. Called by the command line only.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

import chess

from catspace.incubator.analyst.discovery import ledger_puzzles as LP
from catspace.io import paths

LOGS = "abcd_0.2_v_old_A"


def puzzle_lines() -> list:
    out = []
    for row in LP.dev_rows(314, offset=11)[14:]:
        b = chess.Board(row["FEN"])
        mv = row["Moves"].split()
        b.push_uci(mv[0])
        out.append("\t".join([b.fen(), mv[1], row["PuzzleId"]]))
    return out


def game_lines(folder: Path, count: int, seed: int) -> list:
    rng = random.Random(seed)
    files = sorted(folder.glob("*.jsonl.gz"))
    rng.shuffle(files)
    out = []
    for f in files:
        rows = []
        with gzip.open(f, "rt") as fh:
            for line in fh:
                d = json.loads(line)
                if d.get("kind") == "decision" and d["ply"] >= 10:
                    rows.append(d)
        if not rows:
            continue
        d = rng.choice(rows)
        out.append("\t".join([d["position"], d["gid"], str(d["ply"])]))
        if len(out) >= count:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True, help="the folder of the 79.16 decision logs of config abcd at 0.2 s, set A")
    a = ap.parse_args(argv)
    root = Path(paths.experiment("whole_79_51"))
    root.mkdir(parents=True, exist_ok=True)
    pz = puzzle_lines()
    gm = game_lines(Path(a.logs), 200, 7951)
    (root / "puzzles300.txt").write_text("\n".join(pz) + "\n")
    (root / "games200.txt").write_text("\n".join(gm) + "\n")
    print(f"OBS future_79_51 lists: {len(pz)} puzzles written to {root / 'puzzles300.txt'}; {len(gm)} game positions written to {root / 'games200.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
