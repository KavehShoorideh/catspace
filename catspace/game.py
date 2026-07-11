"""game.py — playing single games and bulk rollouts against an Opponent."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from catspace.chain import TransitionChain, KIND_MATE, KIND_STALEMATE, KIND_WHITE_TERMINAL
from catspace.opponents import Opponent
from catspace.planner.policy import Policy


@dataclass
class GameRecord:
    start: int
    states: list        # live-state indices visited, one per white move made
    move_ids: list       # global move id chosen at each step (parallel to states)
    result: str          # "mate" | "draw" | "bwin" | "cap" (cap = hit ply cap, unresolved)
    final_kind: int | None = None   # chain.move_kind of the last move played (None if capped)


def play_game(chain: TransitionChain, white: Policy, black: Opponent, start: int,
              cap: int = 120, rng: np.random.Generator | None = None) -> GameRecord:
    rng = rng if rng is not None else np.random.default_rng()
    s = start
    states: list[int] = []
    move_ids: list[int] = []
    for _ in range(cap):
        states.append(s)
        mid = white.move_id(chain, s, rng)
        move_ids.append(mid)
        k = int(chain.move_kind[mid])
        if k == KIND_MATE:
            return GameRecord(start, states, move_ids, "mate", k)
        if k == KIND_STALEMATE:
            return GameRecord(start, states, move_ids, "draw", k)
        if k == KIND_WHITE_TERMINAL:
            return GameRecord(start, states, move_ids, "bwin", k)
        outs = chain.outs_of(mid)
        bi = black.reply_index(chain, mid, rng)
        nxt = int(outs[bi])
        if nxt >= chain.n_live:
            result = ("mate" if nxt == chain.terminals.mate
                      else "bwin" if nxt == chain.terminals.bwin
                      else "draw")
            return GameRecord(start, states, move_ids, result, k)
        s = nxt
    return GameRecord(start, states, move_ids, "cap", None)


def rollout_transitions(chain: TransitionChain, white: Policy, black: Opponent, starts,
                         cap: int = 120, rng: np.random.Generator | None = None):
    """Bulk rollout collecting raw (row, col) chain transitions for
    empirical_P estimation -- replaces the ~6 near-identical `sample_round`
    copies scattered across the original trainer scripts."""
    rng = rng if rng is not None else np.random.default_rng()
    rows: list[int] = []
    cols: list[int] = []
    n_mate = 0
    for s0 in starts:
        s = int(s0)
        for _ in range(cap):
            mid = white.move_id(chain, s, rng)
            k = int(chain.move_kind[mid])
            if k == KIND_MATE:
                nxt = chain.terminals.mate
            elif k == KIND_STALEMATE:
                nxt = chain.terminals.draw
            elif k == KIND_WHITE_TERMINAL:
                nxt = chain.terminals.bwin
            else:
                outs = chain.outs_of(mid)
                bi = black.reply_index(chain, mid, rng)
                nxt = int(outs[bi])
            rows.append(s); cols.append(nxt)
            if nxt == chain.terminals.mate:
                n_mate += 1
            if nxt >= chain.n_live:
                break
            s = nxt
    return rows, cols, n_mate
