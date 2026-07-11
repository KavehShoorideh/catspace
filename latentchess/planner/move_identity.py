"""
planner/move_identity.py — the MoveIdentity protocol: what counts as "the same
move" for precondition tracking and wake triggers.

Syntactic identity ("Bxa4") is the execution-leaf granularity; region-pair
identity (token(s) -> token(s')) is the semantic, generalizing granularity a
plan actually reasons about. Both produce hashable keys so PlanMemory can index
listeners by key without caring which scheme is in play.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from latentchess.chain import TransitionChain


class MoveIdentity(Protocol):
    name: str

    def key(self, chain: TransitionChain, s: int, mid: int) -> tuple:
        """A hashable key identifying "this move" at this granularity. First
        element is always the scheme's short tag."""
        ...


class SyntacticIdentity:
    name = "syntactic"

    def key(self, chain: TransitionChain, s: int, mid: int) -> tuple:
        return ("syn", chain.move_names[mid])


class RegionPairIdentity:
    """Move identity = (origin region token, destination region token), where
    "destination" is the modal token among the move's live outcomes (ties
    broken by lowest token id via np.bincount().argmax(), deterministic)."""

    name = "region_pair"

    def __init__(self, tokens: np.ndarray):
        self.tokens = tokens

    def key(self, chain: TransitionChain, s: int, mid: int) -> tuple:
        outs = chain.outs_of(mid)
        live = outs[outs < chain.n_live]
        origin = int(self.tokens[s])
        if live.size == 0:
            return ("rgn", origin, "T", int(chain.move_kind[mid]))
        dest = int(np.bincount(self.tokens[live]).argmax())
        return ("rgn", origin, dest)


MOVE_IDENTITIES: dict[str, type] = {
    "syntactic": SyntacticIdentity,
    "region_pair": RegionPairIdentity,
}
