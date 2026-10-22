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
    qid: int = -1      # plan 53: the id of the question this claim answered (the question log says which question decided)
    proved: bool = True        # plan 61 B1: True = every leaf under the claim is closed by a proved rule/lemma; an ESTIMATED claim (a quiet leaf read by its status) is False and never becomes True
    confidence: float = 1.0    # ... the PRODUCT of the confidences of the estimated leaves under it (an expectation, never a bound); 1.0 for a proved claim
    estimator: str = ""        # ... the estimator's name ("quiet-status"); "" for a proved claim
    proved_by: str = ""        # plan 62.3: "" = a proved word (a Lean theorem); "table" = closed by the tablebase (exact by exhaustion, unexplained) -- ranked after the words at equal value, before any estimate
    word: str = ""             # plan 65.3: "unfinished line" -- the best GAP route when no claim exists at all (proved=False); "" for every claim of the search


@dataclass
class State:
    fen: str
    us: str
    pieces: list                                           # state_lines' rows, in the order of attention (the header line first)
    claims: list = field(default_factory=list)
    gaps: list = field(default_factory=list)
    estimated: list = field(default_factory=list)          # plan 61 B2: the ESTIMATED claims (proved=False), kept apart from `claims` -- nothing that reads the proved claims ever sees them

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


def combine(replies: list) -> tuple:
    """plan 61 B1 (the propagation, one place): the label of a ROUTE from the labels of its replies, each (proved, confidence): proved iff every reply is proved, the confidence the PRODUCT of
    the replies' confidences (the expectation that every estimated leaf holds, never a bound). An estimate never becomes proved: one estimated reply makes the route estimated."""
    proved = all(p for p, _c in replies)
    conf = 1.0
    for _p, c in replies:
        conf *= c
    return proved, (1.0 if proved else conf)


def best_route_label(routes: list) -> tuple:
    """plan 61 B1: a QUESTION's claim takes its best route -- routes are (value, proved, confidence); the greatest value wins, and at equal value a proved route beats an estimated one, then the
    higher confidence. -> the chosen route, or None."""
    return max(routes, key=lambda r: (r[0], r[1], r[2])) if routes else None
