"""catspace/engine/watchlist.py -- the TacticWatchlist layer (INQUIRY_TACTICS.md sec 6-7):
latent tactics sensed once, monitored cheaply per ply via watershed SHELLS, alarmed on a
shell CROSSING (the veto lapsed -> pounce). Sits beside SubgoalSelector in LayeredEngine;
on alarm the engine drops alpha and searches the strike.

The monitor is O(1) field-distance calls per watched basin per ply -- monitoring, not
search, is the human-efficiency mechanism (spidey sense)."""
from __future__ import annotations

from dataclasses import dataclass, field

import chess
import numpy as np

from catspace.interfaces import Region


@dataclass
class LatentTactic:
    """A sensed-but-not-yet-available tactic: the basin (post-execution advantage region,
    as a Region + B-bank), the strike that enters it, and the refutation-derived
    preconditions (free-form for now; the RL/precondition-vector seam)."""
    name: str
    basin: Region                          # basin.bank = (n,d) B-embeddings of exemplars
    strike_uci: str | None = None
    preconditions: list = field(default_factory=list)
    sensed_at_ply: int = 0
    last_shell: float = float("inf")
    alarmed: bool = False


class TacticWatchlist:
    """monitor(board) -> list of alarm events. Shell = field distance from F(board) into
    the basin bank (~ plies to entry, coarse). An ALARM fires when the shell crosses from
    above `alarm_shell` to at-or-below it (we slipped into the watershed / their veto
    lapsed). Hysteresis via re-arm only after leaving `rearm_shell`."""

    def __init__(self, fieldmodel, alarm_shell: float = 2.0, rearm_shell: float = 4.0):
        self.field = fieldmodel
        self.alarm_shell = float(alarm_shell)
        self.rearm_shell = float(rearm_shell)
        self.tactics: list[LatentTactic] = []

    def add(self, tactic: LatentTactic):
        assert tactic.basin.bank is not None and len(tactic.basin.bank), "basin needs a B-bank"
        self.tactics.append(tactic)

    def monitor(self, board: chess.Board, ply: int = 0) -> list[dict]:
        """One field call per watched tactic. Returns alarm events:
        {name, shell, prev_shell, ply}."""
        events = []
        if not self.tactics:
            return events
        F = self.field.embed_F_boards([board])
        for t in self.tactics:
            shell = float(self.field.d_to_bank(F, t.basin.bank)[0])
            if t.alarmed and shell > self.rearm_shell:
                t.alarmed = False                     # hysteresis re-arm
            if (not t.alarmed) and shell <= self.alarm_shell < t.last_shell:
                t.alarmed = True
                events.append(dict(name=t.name, shell=shell, prev_shell=t.last_shell, ply=ply))
            t.last_shell = shell
        return events

    def shells(self, board: chess.Board) -> dict:
        """Diagnostic read: {tactic name: current shell}."""
        if not self.tactics:
            return {}
        F = self.field.embed_F_boards([board])
        return {t.name: float(self.field.d_to_bank(F, t.basin.bank)[0]) for t in self.tactics}
