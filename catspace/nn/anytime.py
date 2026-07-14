"""
anytime.py — anytime path-to-mate search over the FB reach field.

Kaveh's search semantics (2026-07-14): "we want a path to mate, then when we
find one, we try for a better one." Two phases sharing one eval budget:

  SATISFICE: greedy best-first descent -- the frontier is ordered by the
    field's reach (the DIRECTION; depth-first when the field is confident),
    opponent replies are predicted by the same field (their best defense =
    the reply that minimizes our reach). The first mate line found becomes
    the INCUMBENT.
  OPTIMIZE (branch-and-bound): keep searching, but prune every node whose
    plies-so-far can no longer beat the incumbent's length -- an EXACT bound
    (g is just plies), needing no calibration of the heuristic. Each new
    mate line that is shorter replaces the incumbent. Budget exhausted ->
    play the incumbent's first move (or, with no incumbent, the first move
    toward the most reachable frontier node).

"Better" is currently SHORTER (plies); when a certainty-calibrated field
lands, the bound becomes the certainty cost plies + lam*(-ln P) -- same
algorithm, richer metric (see GLOSSARY: certainty geometry).

Like MCTS here, one budget unit = one network eval, so --nodes is matched
compute across all three readouts (beam / mcts / anytime). The opponent
model is OUR OWN field's 1-ply prediction -- a found "path" is a plan under
that model, not a proof against arbitrary defense; the policy re-searches
every move, so wrong predictions cost one replan, not the game.
"""
from __future__ import annotations

import heapq
import itertools

import chess
import numpy as np


class _PathNode:
    __slots__ = ("board", "g", "parent", "move")

    def __init__(self, board: chess.Board, g: int, parent, move):
        self.board = board          # White to move
        self.g = g                  # plies from the root
        self.parent = parent
        self.move = move            # WHITE move that left the parent

    def first_move(self) -> chess.Move:
        node = self
        while node.parent is not None and node.parent.parent is not None:
            node = node.parent
        return node.move


class AnytimePathSearch:
    """Best-first, field-directed, branch-and-bound path search (White POV:
    hunts MATE_W; the mirrored policy just passes the mirrored z)."""

    def __init__(self, reach_fn, max_nodes: int, beam: int = 4):
        assert max_nodes >= 1
        self.reach_fn = reach_fn
        self.max_nodes = max_nodes
        self.beam = beam            # White candidates per node that get a
                                    # predicted reply (reply prediction is the
                                    # expensive part: one eval per legal reply)
        self.evals_used = 0
        self.incumbent: _PathNode | None = None   # node whose board IS mate
        self.incumbent_plies: int = 10 ** 9

    def _reach(self, boards):
        self.evals_used += len(boards)
        return np.asarray(self.reach_fn(boards), dtype=float)

    def _expand(self, node: _PathNode):
        """One expansion: all White moves get a cheap 1-batch reach ranking;
        mates -> incumbent candidate; the top-`beam` non-terminal candidates
        get Black's reply predicted (their best defense = min our reach) and
        become frontier children. Returns list of (h, child), h = -reach of
        the child under their best defense (heap pops smallest = most
        reachable line first: the field is the DIRECTION)."""
        out = []
        mid = []                                   # (move, board-after-White)
        for m in node.board.legal_moves:
            b1 = node.board.copy(stack=False)
            b1.push(m)
            if b1.is_checkmate():
                if node.g + 1 < self.incumbent_plies:
                    self.incumbent = _PathNode(b1, node.g + 1, node, m)
                    self.incumbent_plies = node.g + 1
                continue
            if b1.is_game_over(claim_draw=True):   # draw line: dead
                continue
            mid.append((m, b1))
        # bound: a child line needs >= g+3 plies to mate (Wm, Bm, Wm-mate)
        if not mid or node.g + 3 >= self.incumbent_plies:
            return out
        r1 = self._reach([b for _, b in mid])      # cheap ranking, one batch
        order = np.argsort(-r1)[: self.beam]
        for i in order:
            m, b1 = mid[int(i)]
            live = []
            for rm in b1.legal_moves:
                b2 = b1.copy(stack=False)
                b2.push(rm)
                if not b2.is_game_over(claim_draw=True):
                    live.append(b2)
            if not live:                           # every reply ends the game: dead line
                continue
            r = self._reach(live)
            pick = int(np.argmin(r))               # their best defense per OUR field
            child = _PathNode(live[pick], node.g + 2, node, m)
            out.append((-float(r[pick]), child))
        return out

    def search(self, board: chess.Board) -> chess.Move:
        moves = list(board.legal_moves)
        if not moves:
            raise ValueError("no legal moves")
        self.evals_used = 0
        self.incumbent, self.incumbent_plies = None, 10 ** 9
        root = _PathNode(board.copy(stack=False), 0, None, None)
        tie = itertools.count()                    # heap tiebreak: FIFO, deterministic
        frontier: list = []
        best_frontier: _PathNode | None = None
        best_frontier_h = np.inf
        for h, child in self._expand(root):
            heapq.heappush(frontier, (h, next(tie), child))
        while frontier and self.evals_used < self.max_nodes:
            h, _, node = heapq.heappop(frontier)
            if node.g + 1 >= self.incumbent_plies:  # bound moved since push
                continue
            if h < best_frontier_h:
                best_frontier_h, best_frontier = h, node
            for h2, child in self._expand(node):
                heapq.heappush(frontier, (h2, next(tie), child))
        if self.incumbent is not None:
            return self.incumbent.first_move()
        if best_frontier is not None:
            return best_frontier.first_move()
        # every line was pruned/dead at the root: any legal move
        return moves[0]


class FBAnytimePolicy:
    """playout_ab-compatible wrapper (same construction contract as
    FBMCTSPolicy: single goal z or exemplar bank)."""

    def __init__(self, fb, z, max_nodes: int, elo: int = 1800,
                 clock: float = 300.0, device: str = "cpu"):
        from catspace.nn.mcts import FBMCTSPolicy
        # reuse FBMCTSPolicy's batched reach closure (encode/omega/bank logic)
        self._donor = FBMCTSPolicy(fb, z, max_nodes=1, elo=elo, clock=clock,
                                   device=device)
        self.search = AnytimePathSearch(self._donor.mcts.reach_fn, max_nodes)

    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        return self.search.search(board)
