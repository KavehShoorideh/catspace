#!/usr/bin/env python
"""
experiments/experiment_leaderboard.py — read every experiment_report.py JSON
record under artifacts/experiments/, sort by timestamp, and print (+
optionally write) a structured comparison: each run's key metrics plus its
delta vs the IMMEDIATELY PREVIOUS run and vs the BEST run so far by
--metric. Pure JSON in, structured JSON/table out -- no viz needed to see
whether the last training change helped.

Any record whose candidate.leakage_audit.clean is not true is marked DIRTY
and excluded from the "best" ranking (but still shown, so a contaminated
run doesn't just silently disappear from the history).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catspace.io.paths import experiments_dir

METRIC_PATHS = {
    "arena_score": lambda r: r["candidate"]["arena_vs_opponent"]["score_mean"],
    "arena_e_value": lambda r: r["candidate"]["arena_vs_opponent"]["e_value"],
    "diff_slope_won": lambda r: r["candidate"]["reach_slopes"]["diff_slope_won"],
    "diff_slope_lost": lambda r: r["candidate"]["reach_slopes"]["diff_slope_lost"],
    "decompose_mean_gain": lambda r: r["candidate"].get("decompose", {}).get("mean_gain"),
    "decompose_frac_improved": lambda r: r["candidate"].get("decompose", {}).get("frac_improved"),
}


def load_reports(path: Path) -> list:
    reports = []
    for f in sorted(path.glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except json.JSONDecodeError:
            continue
        r["_file"] = f.name
        reports.append(r)
    reports.sort(key=lambda r: r.get("timestamp", ""))
    return reports


def is_clean(r: dict) -> bool:
    return bool(r.get("candidate", {}).get("leakage_audit", {}).get("clean"))


def metric_value(r: dict, metric: str):
    try:
        return METRIC_PATHS[metric](r)
    except (KeyError, TypeError):
        return None


def build_rows(reports: list, metric: str) -> list:
    rows = []
    best_so_far = None
    prev = None
    for r in reports:
        clean = is_clean(r)
        v = metric_value(r, metric) if clean else None
        row = dict(file=r["_file"], timestamp=r.get("timestamp"), tag=r.get("tag"),
                  step=r.get("candidate", {}).get("step"), clean=clean,
                  sha256=r.get("candidate", {}).get("sha256"),
                  **{m: metric_value(r, m) for m in METRIC_PATHS})
        if clean and v is not None:
            row["delta_vs_prev"] = (v - prev) if prev is not None else None
            row["delta_vs_best"] = (v - best_so_far) if best_so_far is not None else None
            if best_so_far is None or v > best_so_far:
                best_so_far = v
            prev = v
        else:
            row["delta_vs_prev"] = None
            row["delta_vs_best"] = None
        rows.append(row)
    return rows


def print_table(rows: list, metric: str) -> None:
    print(f"{'step':>7}  {'clean':>5}  {metric:>18}  {'Δprev':>9}  {'Δbest':>9}  {'tag/file'}")
    for row in rows:
        v = row.get(metric)
        vs = f"{v:.4f}" if isinstance(v, (int, float)) else "n/a"
        dp = row["delta_vs_prev"]
        dps = f"{dp:+.4f}" if isinstance(dp, (int, float)) else "-"
        db = row["delta_vs_best"]
        dbs = f"{db:+.4f}" if isinstance(db, (int, float)) else "-"
        label = row["tag"] or row["file"]
        print(f"{str(row['step']):>7}  {'Y' if row['clean'] else 'DIRTY':>5}  {vs:>18}  "
              f"{dps:>9}  {dbs:>9}  {label}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=None, help="default: artifacts/experiments")
    ap.add_argument("--metric", default="arena_score", choices=sorted(METRIC_PATHS))
    ap.add_argument("--out", default=None, help="optional path to also write the rows as JSON")
    args = ap.parse_args()

    path = Path(args.dir) if args.dir else experiments_dir()
    reports = load_reports(path)
    if not reports:
        print(f"no experiment reports under {path}")
        return

    rows = build_rows(reports, args.metric)
    print_table(rows, args.metric)

    n_dirty = sum(1 for r in rows if not r["clean"])
    if n_dirty:
        print(f"\n{n_dirty} DIRTY (leakage-audit-failed) run(s) excluded from best/prev deltas.")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
