"""PLAN 78.4 KNOWLEDGE OF ONE INSTANCE OF A GAME (Kaveh 2026-11-04: "there's some knowledge I have about chess in general. There's some knowledge I have about this
chess game that I'm playing"). Knowledge of the GAME as a whole lives in the store (lemma_store: entries scoped to a game by `game`, chess when absent). Knowledge of an
INSTANCE -- one game played or watched: where it started, its moves, its outcome, and what was learned while it was played (facts: name -> value, with where they were
learned) -- lives here, as records under data/derived/instances/<game>/. The miners (cs_learn: the valuation miner ...) read instances and propose game knowledge;
nothing is written to the store unless asked (`learn --write`).
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from catspace.io import paths


@dataclass
class Instance:
    game: str
    start: str = ""                       # an opaque start (a FEN for a Chess960 start); empty: the game's own initial state
    moves: list = field(default_factory=list)
    outcome: list = field(default_factory=list)       # the goal per role (empty while unfinished)
    players: dict = field(default_factory=dict)       # role -> who played it (our engine's approach, an outside engine, a person)
    facts: dict = field(default_factory=dict)         # what was learned while it was played: name -> {value, ply}
    source: str = ""
    when: float = field(default_factory=time.time)

    def learn(self, name: str, value, ply: "int | None" = None) -> None:
        self.facts[name] = {"value": value, "ply": len(self.moves) if ply is None else ply}

    def record(self) -> tuple:
        """The miners' form: (start, moves, outcome)."""
        return (self.start, list(self.moves), list(self.outcome))


def folder(game: str) -> Path:
    p = Path(paths.derived("instances")) / game
    p.mkdir(parents=True, exist_ok=True)
    return p


def save(insts: list, game: str, name: str) -> Path:
    """Append instances to data/derived/instances/<game>/<name>.jsonl."""
    f = folder(game) / f"{name}.jsonl"
    with open(f, "a") as h:
        for i in insts:
            h.write(json.dumps(asdict(i)) + "\n")
    return f


def load(game: str, name: "str | None" = None) -> list:
    """The instances of a game (one file, or every file of its folder)."""
    out = []
    for f in sorted(folder(game).glob(f"{name or '*'}.jsonl")):
        for line in open(f):
            out.append(Instance(**json.loads(line)))
    return out
