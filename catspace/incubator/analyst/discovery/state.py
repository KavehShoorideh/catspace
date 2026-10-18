"""LAYER 2 OF THE CORE (ANALYST_PLAN 52.1): the STATE of one position -- all our reasoning about it as relations in the order of attention (kings, then by value,
ours before theirs, pawns last), the CLAIMS each backed by a proved lemma, and the admitted GAPS. The rows are puzzle_apply.state_lines's (imported, not
retyped); the tokens are remember.py's signature tokens, so one object serves the stream and the memory store.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from catspace.incubator.analyst.discovery import puzzle_apply as PA
from catspace.incubator.analyst.discovery import remember as RM


@dataclass
class Claim:
    kind: str          # 'mate' | 'win' | 'save'
    what: str          # the move, in SAN, that the claim is about
    value: int         # the guaranteed gain (100 for a mate)
    lemma: str         # the proved theorem whose binding backs it (selfmine.schema_name)
    line: str          # the verified route and the replies it is quantified over
    move: str = ""     # the first move of the bound route, UCI (the binding's x and y)
    goal: str = ""     # 52.9: the root question's goal kind (mate | win | save | promote)
    fixes: tuple = ()  # 52.9: the kinds of the fix sub-questions on the claim's path


@dataclass
class State:
    fen: str
    us: str
    pieces: list                                           # state_lines' rows, in the order of attention (the header line first)
    claims: list = field(default_factory=list)
    gaps: list = field(default_factory=list)

    def tokens(self, threat: bool = True) -> set:
        """remember.py's signature (piece tokens, no squares; the threat token) -- the same object the memory store indexes."""
        thr = None
        if threat:
            thr = RM.threat_token(self.fen, self.us, check=chess.Board(self.fen).is_check())
        return RM.signature(RM.piece_tokens(self.fen, self.us), (), thr)

    def lines(self) -> list:
        return list(self.pieces)


def state_of(fen: str, us: str) -> State:
    return State(fen=fen, us=us, pieces=PA.state_lines(fen, us))
