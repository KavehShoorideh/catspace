"""PLAN 79.02: where the outputs of the plan go and how they are printed (SYSTEM_SPEC 11 engineering rules; no piece of the system, a helper of the measure modules). Imports nothing of the oracle: the match module (cut_play) takes its output folder and its OBS lines from here and from nothing that loads the oracle wrapper (the wall,
tests/test_plan79_02_cut.py).

INTERFACE.
  OUT                      the experiment folder's name under artifacts/experiments (catspace.io.paths).
  out_root(tag) -> Path    artifacts/experiments/cut_79_02/<tag>, made.
  Say(root, stage)         callable: prints "OBS cut <stage>: <msg>" and appends it to <root>/obs.txt.
  Called by: cut_count, cut_mend, cut_play.
"""
from __future__ import annotations

from pathlib import Path

OUT = ("cut_79_02",)


def out_root(tag: str) -> Path:
    from catspace.io import paths
    p = Path(paths.experiment(*OUT, tag))
    p.mkdir(parents=True, exist_ok=True)
    return p


class Say:
    def __init__(self, root: Path, stage: str):
        self.root, self.stage = root, stage

    def __call__(self, msg: str) -> None:
        line = f"OBS cut {self.stage}: {msg}"
        print(line, flush=True)
        with open(self.root / "obs.txt", "a") as f:
            f.write(line + "\n")
