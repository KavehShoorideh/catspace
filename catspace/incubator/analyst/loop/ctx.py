"""The context a stage runs in: its round, its directories, its configuration, the code version, and helpers to
read earlier stages' outputs and stamp provenance."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from .contracts import Provenance
from .store import read_json


@dataclass
class Ctx:
    round: int
    dir: Path                    # data/derived/loop/round_NNN
    record: Path                 # artifacts/loop/round_NNN
    config: dict
    code: str
    log_lines: list = field(default_factory=list)

    def load(self, stage: str) -> dict | None:
        """An earlier stage's output (None when it has not run)."""
        d = read_json(self.dir / f"{stage}.json")
        return None if d is None else d["output"]

    def prov(self, stage: str, sources=()) -> Provenance:
        return Provenance(self.round, stage, list(sources), self.code)

    def cfg(self, stage: str) -> dict:
        return dict(self.config.get(stage) or {})

    def log(self, msg: str) -> None:
        line = f"[loop r{self.round:03d} {time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_lines.append(line)
        with open(self.dir / "round.log", "a") as fh:
            fh.write(line + "\n")
