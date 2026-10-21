"""A RANDOM sample of puzzles from the whole lichess puzzle file (Kaveh 2026-10-20: "randomly sample puzzles from the set; don't stick to the
same puzzles over and over"). dev_20 / fresh_60 were fixed strided draws; this draws n rows uniformly by reservoir sampling under a seed and
writes `ledger_puzzles_rand<seed>_<n>.json` in the shape `puzzle_apply.load_set` reads (id, fen after the setup move, answer uci, value, themes,
rating).

Run: .venv/bin/python -m catspace.incubator.analyst.discovery.sample_puzzles --seed 1 --n 10
then: .venv/bin/python -m catspace.incubator.analyst.discovery.analyse --set rand1_10
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random

from catspace.io import paths
from catspace.incubator.analyst.discovery import ledger_puzzles as LP


def rows(n: int, seed: int, themes: "list | None" = None) -> list:
    """Uniform reservoir sample of n rows; with `themes`, only rows whose Themes column has every one of them (plan 61 C4: `perpetualCheck`)."""
    import zstandard
    rng = random.Random(seed)
    keep = []
    with open(LP._csv_path(), "rb") as fh:
        reader = csv.DictReader(io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8"))
        i = -1
        for row in reader:
            if themes and not set(themes) <= set((row.get("Themes") or "").split()):
                continue
            i += 1
            if len(keep) < n:
                keep.append(row)
            else:
                j = rng.randrange(i + 1)
                if j < n:
                    keep[j] = row
    return keep


def record(row: dict) -> dict:
    b, sol, themes = LP._setup(row)
    return {"id": row["PuzzleId"], "fen": b.fen(), "answer": sol[0], "value": LP._value(b, sol, themes), "themes": themes,
            "rating": int(row.get("Rating", 0) or 0), "solution": sol}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--themes", nargs="*", default=None, help="plan 61 C4: only puzzles that carry all of these lichess themes (the set is named rand<seed>_<n>_<themes>)")
    a = ap.parse_args(argv)
    recs = [record(r) for r in rows(a.n, a.seed, a.themes)]
    name = f"rand{a.seed}_{a.n}" + ("_" + "_".join(a.themes) if a.themes else "")
    out = paths.derived(f"ledger_puzzles_{name}.json")
    json.dump(recs, open(out, "w"), indent=1)
    print(f"OBS sample {name}: {len(recs)} puzzles -> {out}; ids " + " ".join(r['id'] for r in recs))
    print("OBS sample ratings: " + " ".join(str(r['rating']) for r in recs) + "; values: " + " ".join(str(r['value']) for r in recs))


if __name__ == "__main__":
    main()
