#!/usr/bin/env python
"""prune_checkpoints.py -- retention policy for training checkpoints (Kaveh 2026-08-14: "have
a way to kind of delete all checkpoints that we don't think we are ever gonna use").

Checkpointing every 100 steps over a 20k-step run is 200 files x ~15MB = 3GB per generation,
so the ladder needs a policy rather than a spring clean. DRY-RUN BY DEFAULT: nothing is
removed unless --apply is passed, and the plan is always printed first.

What is NEVER touched, regardless of flags:
  *_latest.pt          the generation's final weights
  sidecars             _jqt / _former / _pointer / _vq / _dyn / _seq / _human / _exemplars .pt
                       (the play stack and every probe load these by name)
  anything in --keep   explicit protection list

Ladder policy: keep the last --keep-last checkpoints, plus every --keep-every-th one as a
coarse history, drop the rest.

    .venv/bin/python -m ...prune_checkpoints                      # dry run, whole dir
    .venv/bin/python -m ...prune_checkpoints --gens jqt1,jqt2 --apply
"""
from __future__ import annotations

import argparse
import os
import re

SIDECAR = ("_jqt.pt", "_former.pt", "_pointer.pt", "_vq.pt", "_dyn.pt", "_seq.pt",
           "_human.pt", "_exemplars.pt", "_conceptmap.json")


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f}{u}"
        n /= 1024
    return f"{n:.1f}TB"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="artifacts/experiments")
    ap.add_argument("--gens", default="",
                    help="comma list of generation prefixes to prune (default: every ladder)")
    ap.add_argument("--keep-last", type=int, default=3,
                    help="most recent ladder checkpoints to keep per generation")
    ap.add_argument("--keep-every", type=int, default=2000,
                    help="also keep ladder checkpoints at multiples of this step")
    ap.add_argument("--keep", default="",
                    help="comma list of substrings that protect a file outright")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry run)")
    args = ap.parse_args()

    prot = [x for x in args.keep.split(",") if x]
    gens = [g for g in args.gens.split(",") if g]
    lad = re.compile(r"^(.*)_step(\d+)\.pt$")
    by_gen: dict[str, list[tuple[int, str, int]]] = {}
    for f in os.listdir(args.dir):
        m = lad.match(f)
        if not m:
            continue
        stem, step = m.group(1), int(m.group(2))
        if gens and not any(g in stem for g in gens):
            continue
        p = os.path.join(args.dir, f)
        by_gen.setdefault(stem, []).append((step, p, os.path.getsize(p)))

    total_del = total_keep = 0
    plan = []
    for stem, rows in sorted(by_gen.items()):
        rows.sort()
        keep_steps = {s for s, _, _ in rows[-args.keep_last:]}
        keep_steps |= {s for s, _, _ in rows if args.keep_every and s % args.keep_every == 0}
        for s, p, sz in rows:
            base = os.path.basename(p)
            if any(x in base for x in prot) or any(base.endswith(x) for x in SIDECAR):
                keep_steps.add(s)
            if s in keep_steps:
                total_keep += sz
            else:
                total_del += sz
                plan.append(p)
        print(f"  {stem:34s} {len(rows):3d} ladder  keep {len(keep_steps):2d}  "
              f"drop {len(rows)-len(keep_steps):3d}")

    print(f"\n[prune] would free {human(total_del)} · keeping {human(total_keep)} of ladders")
    print(f"[prune] NEVER touched: *_latest.pt and sidecars {', '.join(SIDECAR[:6])}...")
    if not args.apply:
        print("[prune] DRY RUN -- pass --apply to delete")
        return
    freed = 0
    for p in plan:
        try:
            freed += os.path.getsize(p)
            os.remove(p)
        except OSError as e:
            print(f"  skip {p}: {e}")
    print(f"[prune] deleted {len(plan)} files, freed {human(freed)}")


if __name__ == "__main__":
    main()
