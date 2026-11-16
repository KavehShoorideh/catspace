"""PLAN 78.70 D (puzzles): the loader that turns a 78.45 puzzle set into game-free pairs for threat_measure.py: each puzzle becomes {"id", "start": the position text, "first": the solution's first move}. A small script,
NOT a layer, sheet, binding or measure file: it is the one place that knows the puzzle files' own field names (which are game words). The puzzles are used as a MEASURE only: nothing of them enters a fit or a head.

  .venv/bin/python -m catspace.incubator.analyst.discovery.threat_puzzle_pairs [--set dev_300] [--out-dir D]
"""
from __future__ import annotations

import argparse
import json

from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import threat_measure as TM


def main(argv: "list | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", default="dev_300")
    ap.add_argument("--out-dir", default=None)
    a = ap.parse_args(argv)
    pairs = [{"id": r["id"], "start": r["fen"], "first": r["answer"]} for r in PA.load_set(a.set)]
    out = TM.out_dir(a.out_dir)
    (out / "puzzle_pairs.json").write_text(json.dumps(pairs))
    print(f"OBS threats: {len(pairs)} puzzles of the set {a.set} written as pairs to {out / 'puzzle_pairs.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
