"""PLAN 78.76, SYSTEM_SPEC 5 ("learn from results", the odds games) and 9 (the ladder as the diagnosis): THE OTHER MEASURED RECORDS OF THE KINDS, written in the layout threat_concrete.Record.load reads. A sandbox: no player is
changed, nothing is written to rules/learned or data/derived. Game-free: a kind is a word read from the stored fit; no kind is named here. Nothing is typed in: every worth and every band is read from a stored fit file.

INTERFACE (what this module takes, what it returns, who calls it).
  SOURCES                       {record name: (the stored fit file, where its kind terms are found in it)}. The names are the ones `ladder_measure --record <name>` takes (`gen2` is the 78.61 record, which exists already).
  make(name) -> Path            reads the stored fit of `name` and writes artifacts/experiments/ladder_78_76/<name>/record.json: {"heads": {"odds_effect": {"kind_values": {kind: whole number},
                                "per_kind": {kind: {"one": {"term", "lo", "hi"}}}}}, "provenance": {...}}. The convention of the generation-2 record is kept: the worth is round(1000 x the term), a term being the log odds
                                of a win against a loss per extra unit of the kind; a kind's half width is 500 x (hi - lo) of its printed 90 pct band, in the worths' own integers. Only the ratios of the worths matter to the
                                search; the half widths set the resolution of the line read (Record.resolution). The goal kind carries no worth in any record (a rule, not a fit), as in the generation-2 record.
  Called by: the command line (`python -m ...ladder_records make`); ladder_measure asks for the file by name (threat_concrete.named).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from catspace.io import paths

SCALE = 1000
SOURCES = {
    "selfplay_78_58": (("estimator_78_58", "full", "estimator.json"), ("order_of_kinds",)),                                 # self-play only (plan 78.58): the order of kinds with its band
    "human_78_59": (("estimator_78_59", "full", "estimator_heads_human.json"), ("heads", "kind_baseline_human", "order_of_kinds")),   # human games only (plan 78.59): the kinds alone
}


def _dig(x, keys):
    for k in keys:
        x = x[k]
    return x


def make(name: str) -> Path:
    src, keys = SOURCES[name]
    f = Path(paths.experiment(*src))
    order = _dig(json.loads(f.read_text()), keys)
    values = {o["kind"]: int(round(SCALE * o["term"])) for o in order}
    per_kind = {o["kind"]: {"one": {"term": o["term"], "lo": o["lo"], "hi": o["hi"]}} for o in order}
    rec = {"estimator": f"the kind terms of {'/'.join(src)}", "heads": {"odds_effect": {"kind_values": values, "per_kind": per_kind}},
           "provenance": {"source": "/".join(src), "path_in_file": "/".join(keys), "scale": SCALE, "worth": "round(scale x term)", "half_width": "500 x (hi - lo) of the printed 90 pct band, in the worths' integers"}}
    out = Path(paths.experiment("ladder_78_76", name, "record.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1))
    halves = {k: round(500.0 * (v["one"]["hi"] - v["one"]["lo"]), 1) for k, v in per_kind.items()}
    print(f"OBS ladder records {name}: {f} -> {out}; worths {values}; half widths {halves}", flush=True)
    return out


def main(argv: "list | None" = None) -> int:
    a = list(sys.argv[1:] if argv is None else argv)
    if not a or a[0] != "make":
        print(__doc__)
        return 2
    for name in (a[1:] or sorted(SOURCES)):
        make(name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
