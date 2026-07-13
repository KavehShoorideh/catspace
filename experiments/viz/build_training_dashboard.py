#!/usr/bin/env python
"""
experiments/viz/build_training_dashboard.py — parse train_lichess_fb.py logs
(artifacts/generated/logs/train_*.log) into an interactive loss/top1/top8/
throughput dashboard. Pure log parsing, no torch import -- safe to rerun
mid-training to refresh the picture.
"""
from __future__ import annotations

import argparse
import datetime
import glob
import re
import time
from pathlib import Path

from catspace.io.paths import generated_dir
from catspace.viz.build_html import build_html

TRAIN_RE = re.compile(r"^step (\d+)  loss ([\d.]+)  train_top1 ([\d.]+)  \(([\d.]+) it/s\)$")
VAL_RE = re.compile(r"^  VAL step (\d+)  loss ([\d.]+)  top1 ([\d.]+)  top8 ([\d.]+)$")
RESUME_RE = re.compile(r"resumed .* at step (\d+)")
VERDICT_RE = re.compile(r"^VERDICT .*$")
BATCH_RE = re.compile(r"batch=(\d+)")


def parse_log(path: Path) -> dict:
    text = path.read_text()
    train = {"step": [], "loss": [], "top1": [], "rate": []}
    val = {"step": [], "loss": [], "top1": [], "top8": []}
    resumes, verdicts = [], []
    batch = 512
    for line in text.splitlines():
        m = BATCH_RE.search(line)
        if m and "batch=" in line and line.strip().startswith("shards="):
            batch = int(m.group(1))
        m = TRAIN_RE.match(line)
        if m:
            train["step"].append(int(m.group(1)))
            train["loss"].append(float(m.group(2)))
            train["top1"].append(float(m.group(3)))
            train["rate"].append(float(m.group(4)))
            continue
        m = VAL_RE.match(line)
        if m:
            val["step"].append(int(m.group(1)))
            val["loss"].append(float(m.group(2)))
            val["top1"].append(float(m.group(3)))
            val["top8"].append(float(m.group(4)))
            continue
        m = RESUME_RE.search(line)
        if m:
            resumes.append(int(m.group(1)))
            continue
        if VERDICT_RE.match(line):
            verdicts.append(line.strip())
    return dict(train=train, val=val, resumes=resumes, verdicts=verdicts,
               chance={"top1": 1.0 / batch, "top8": 8.0 / batch})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", nargs="*", default=None,
                    help="log paths or globs; default artifacts/generated/logs/train_*.log")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    t0 = time.time()
    patterns = args.logs or [str(Path(__file__).resolve().parents[2] / "artifacts" /
                                 "generated" / "logs" / "train_*.log")]
    paths = sorted({Path(p) for pat in patterns for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"no logs matched {patterns}")

    runs = []
    for p in paths:
        parsed = parse_log(p)
        runs.append(dict(name=p.stem, **parsed))
        print(f"parsed {p.name}: {len(parsed['train']['step'])} train pts, "
              f"{len(parsed['val']['step'])} val pts, {len(parsed['verdicts'])} verdicts")

    data = dict(meta=dict(title="catspace — FB training dashboard",
                          built=datetime.datetime.now().isoformat(timespec="seconds")),
               runs=runs)

    out = Path(args.out) if args.out else generated_dir() / "training-dashboard.html"
    template = Path(__file__).resolve().parents[2] / "catspace" / "viz" / "templates" / "training_dashboard.html"
    build_html(template, data, out)
    print(f"wrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
