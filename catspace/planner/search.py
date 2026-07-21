"""catspace/planner/search.py -- pluggable low-level SUBGOAL search (Kaveh 2026-07-20).

The high-level landmark planner (SoRB/ProQ-style) proposes a subgoal using the field's
COOPERATIVE distance. The opponent may not cooperate, so the low-level search finds the
REAL cost to the subgoal. Chess is a two-player game, so we offer a spectrum:

  WeightedAStar   single-agent OPTIMISTIC (adversary ignored) -- fast feasibility + path.
  MinimaxAStar    AO*/LAO*-style ADVERSARIAL best-first (you=min, opponent=max) with the
                  field heuristic -- the "adversarial Dijkstra with a learned heuristic".
  MCTS            adversarial SAMPLING (UCT) -- scales when the tree is large.

All share one interface (Goal + Budget -> SearchResult) and one heuristic (the field
distance-to-subgoal), so they are plug-and-play and easy to extend: subclass
LowLevelSearch and register with @register.

Pruning (Kaveh's "stop searching a direction that keeps getting farther, or is dominated
by another frontier node"): the best-first frontier does this by construction (f-ordered,
optional beam); MinimaxAStar also alpha-beta prunes.
"""
from __future__ import annotations

import heapq
import itertools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import chess


@dataclass
class Budget:
    max_nodes: int = 3000        # expansions before giving up
    max_plies: int = 40          # depth cap (half-moves)
    beam: int | None = None      # keep only the beam-best frontier nodes (None = full)


@dataclass
class Goal:
    """A subgoal region. `heuristic` is the field distance-to-subgoal in plies (lower =
    closer); it is BATCHED for GPU efficiency. `reached` tests membership."""
    heuristic: Callable[[list[chess.Board]], "list[float]"]   # batched: boards -> distances
    reached: Callable[[chess.Board], bool]
    label: str = ""

    def h1(self, board: chess.Board) -> float:
        return float(self.heuristic([board])[0])


@dataclass
class SearchResult:
    reached: bool
    cost: float                  # plies along the returned path
    path: list                   # list[chess.Move] from start
    trajectory: list             # list[str] FENs along the path (for viz/log)
    nodes: int
    algo: str
    log: list = field(default_factory=list)


SEARCH_REGISTRY: dict[str, type] = {}


def register(cls):
    SEARCH_REGISTRY[cls.name] = cls
    return cls


def get_search(name: str, **kw) -> "LowLevelSearch":
    if name not in SEARCH_REGISTRY:
        raise KeyError(f"unknown search '{name}'; have {list(SEARCH_REGISTRY)}")
    return SEARCH_REGISTRY[name](**kw)


class LowLevelSearch(ABC):
    name = "base"

    @abstractmethod
    def search(self, start: chess.Board, goal: Goal, budget: Budget) -> SearchResult:
        ...

    @staticmethod
    def _terminal_path(start, moves):
        b = start.copy(stack=False); traj = [b.fen()]
        for m in moves:
            b.push(m); traj.append(b.fen())
        return traj


@register
class MinimaxAStar(LowLevelSearch):
    """AO*/LAO*-flavored ADVERSARIAL search: the mover (us) minimizes plies-to-subgoal,
    the opponent maximizes it. Iterative-deepening bounded minimax with the field heuristic
    at the leaves, alpha-beta pruning, and a transposition/visited guard for cycles
    (repetition). Returns the principal variation and whether the subgoal is forcibly
    reachable within the depth budget -- the 'adversarial Dijkstra with a learned heuristic'."""
    name = "minimax_astar"

    def __init__(self, max_depth: int = 8, w: float = 1.0):
        self.max_depth = max_depth
        self.w = w
        self._nodes = 0

    def search(self, start, goal, budget):
        us = start.turn                                    # the side trying to reach the subgoal
        self._nodes = 0
        best_line: list = []
        for depth in range(2, self.max_depth + 1, 2):      # iterative deepening (our-ply pairs)
            self._nodes_cap = budget.max_nodes
            line: list = []
            val = self._mm(start, depth, -math.inf, math.inf, us, goal, line, seen=set())
            if line:
                best_line = line
            if self._nodes >= budget.max_nodes:
                break
        # re-walk to build path/trajectory + check reach
        b = start.copy(stack=False); path = []; traj = [b.fen()]; reached = goal.reached(b)
        for m in best_line:
            if m not in b.legal_moves:
                break
            b.push(m); path.append(m); traj.append(b.fen())
            if goal.reached(b):
                reached = True; break
        cost = len(path)
        return SearchResult(reached, cost if reached else math.inf, path, traj, self._nodes, self.name)

    def _mm(self, b, depth, alpha, beta, us, goal, line, seen):
        self._nodes += 1
        if goal.reached(b):
            return 0.0
        if depth == 0 or b.is_game_over() or self._nodes >= self._nodes_cap:
            return self.w * goal.h1(b)
        key = b._transposition_key() if hasattr(b, "_transposition_key") else b.fen()
        if key in seen:                                    # cycle (repetition) -> treat as far
            return self.w * goal.h1(b) + 1.0
        seen = seen | {key}
        legal = list(b.legal_moves)
        kids = [b.copy(stack=False) for _ in legal]
        for c, m in zip(kids, legal):
            c.push(m)
        # order children by field heuristic (best-first) for stronger alpha-beta
        hs = goal.heuristic(kids)
        order = sorted(range(len(legal)), key=lambda i: hs[i])
        our_move = (b.turn == us)
        best = math.inf if our_move else -math.inf
        best_child_line: list = []
        for i in order:
            child_line: list = []
            v = 1.0 + self._mm(kids[i], depth - 1, alpha, beta, us, goal, child_line, seen)
            if our_move:
                if v < best:
                    best = v; best_child_line = [legal[i]] + child_line
                beta = min(beta, best)
            else:
                if v > best:
                    best = v; best_child_line = [legal[i]] + child_line
                alpha = max(alpha, best)
            if beta <= alpha:
                break
        line[:] = best_child_line
        return best


@register
class MCTS(LowLevelSearch):
    """Adversarial UCT toward the subgoal. Reward at a leaf = 1 if the subgoal is reached
    else a shaped -field_distance (closer = higher). Our nodes maximize, opponent nodes
    minimize (negamax backup). Rollouts are field-greedy (descend the heuristic)."""
    name = "mcts"

    def __init__(self, iterations: int = 800, c: float = 1.4, rollout_depth: int = 10):
        self.iterations = iterations
        self.c = c
        self.rollout_depth = rollout_depth

    def search(self, start, goal, budget):
        us = start.turn
        root_fen = start.fen()
        # node: fen -> dict(N, W, children{move:fen}, expanded)
        N: dict = {}; W: dict = {}; children: dict = {}

        def val(board):                                     # leaf value in [~ -inf, 1], us-relative
            if goal.reached(board):
                return 1.0
            return -float(goal.h1(board)) / 10.0            # shaped by distance (plies/10)

        nodes = 0
        for _ in range(min(self.iterations, budget.max_nodes)):
            b = start.copy(stack=False)
            path_fen = [b.fen()]; path_move = []
            # ---- selection ----
            while b.fen() in children and not goal.reached(b) and not b.is_game_over() and len(path_move) < budget.max_plies:
                fen = b.fen(); kids = children[fen]
                logp = math.log(max(1, N.get(fen, 1)))
                sign = 1.0 if b.turn == us else -1.0        # opponent picks to minimize our value
                best_m, best_u = None, -math.inf
                for m, cf in kids.items():
                    q = (W.get(cf, 0.0) / N[cf]) if N.get(cf, 0) else 0.0
                    u = sign * q + self.c * math.sqrt(logp / (1 + N.get(cf, 0)))
                    if u > best_u:
                        best_u, best_m = u, m
                bm = chess.Move.from_uci(best_m)
                b.push(bm); path_move.append(bm); path_fen.append(b.fen())
            # ---- expansion ----
            if not goal.reached(b) and not b.is_game_over() and len(path_move) < budget.max_plies:
                fen = b.fen()
                if fen not in children:
                    children[fen] = {m.uci(): (lambda c: (c.push(m) or c.fen()))(b.copy(stack=False))
                                     for m in b.legal_moves}
                    for cf in children[fen].values():
                        N.setdefault(cf, 0); W.setdefault(cf, 0.0)
            # ---- rollout (field-greedy) ----
            r = b.copy(stack=False); d = 0
            while not goal.reached(r) and not r.is_game_over() and d < self.rollout_depth:
                legal = list(r.legal_moves)
                kids = [r.copy(stack=False) for _ in legal]
                for c, m in zip(kids, legal):
                    c.push(m)
                hs = goal.heuristic(kids)
                r = kids[int(min(range(len(legal)), key=lambda i: hs[i]))]  # greedy descend
                d += 1
            leaf = val(r)
            # ---- backprop ----
            for fen in path_fen:
                N[fen] = N.get(fen, 0) + 1
                W[fen] = W.get(fen, 0.0) + leaf
            nodes += 1

        # extract principal variation: from root, greedily follow most-visited child toward goal
        b = start.copy(stack=False); path = []; traj = [b.fen()]; reached = goal.reached(b)
        while b.fen() in children and len(path) < budget.max_plies:
            kids = children[b.fen()]
            m_uci = max(kids, key=lambda mu: N.get(kids[mu], 0))
            m = chess.Move.from_uci(m_uci)
            if m not in b.legal_moves:
                break
            b.push(m); path.append(m); traj.append(b.fen())
            if goal.reached(b):
                reached = True; break
        return SearchResult(reached, len(path) if reached else math.inf, path, traj, nodes, self.name)
