"""MCTS WITH LEMMA PRIORS: the board-level search that moves the board forward (Kaveh 2026-09-23), on top of the
scheduler's answers. PUCT over the rules' moves; the PRIOR at a node comes from the resolved exchange (the capture
quiescence) plus the lemma verdicts the graph layer holds for that node's moves; the LEAF value is the resolved
exchange from the root side, squashed. Every expanded node registers its frame tasks with the scheduler at an
importance equal to its visit share, and the scheduler runs a slice of that work each simulation -- the asynchronous
analysis that shifts with the search. Counted in nodes.

    python -m catspace.incubator.analyst.player play --search mcts --sims 400 ...
"""
from __future__ import annotations

import math
import random
from collections import Counter

import chess

from . import nodes
from .player import MATE, Search, evaluate, mvv_lva


def squash(v: float) -> float:
    return math.tanh(v / 3.0)


class Node:
    __slots__ = ("board", "side", "children", "N", "W", "P", "expanded", "moves", "key")

    def __init__(self, board: chess.Board, side: chess.Color):
        self.board = board; self.side = side; self.children = {}; self.N = 0; self.W = 0.0; self.P = {}; self.expanded = False; self.moves = []
        self.key = board._transposition_key()

    @property
    def Q(self):
        return self.W / self.N if self.N else 0.0


class MCTS:
    def __init__(self, player, sims: int = 400, c_puct: float = 1.5, q_depth: int = 6, lemma_weight: float = 0.5, task_slice: int = 4, rng=None):
        self.pl = player; self.sims = sims; self.c = c_puct; self.q_depth = q_depth; self.lw = lemma_weight; self.task_slice = task_slice
        self.rng = rng or random.Random(0); self.search = Search(node_cap=10 ** 9, q_depth=q_depth); self.stats = Counter()

    def resolved(self, b: chess.Board, side: chess.Color) -> float:
        self.search.n = 0
        return self.search.quiesce(b, side, -MATE * 2, MATE * 2, self.q_depth)

    def priors(self, node: Node) -> dict:
        """Softmax over (resolved gain of each move from the mover's side) + lemma verdict bonus; the verdicts come
        from the player's graph layer (content-cached), asked for through the scheduler when one is attached."""
        b = node.board; mover = b.turn
        legal = list(b.legal_moves)
        node.moves = legal
        gains = {}
        for m in legal:
            b.push(m); gains[m] = self.resolved(b, mover); b.pop()
        gv = {}
        if self.pl.app is not None:
            gv = self.pl.graph_verdicts(b, legal)
        logits = {}
        for m in legal:
            k, l = gv.get(m, (0, 0))
            lemma = (1.0 if k > l else -1.0 if l > k else 0.0)
            logits[m] = gains[m] + self.lw * lemma + 0.05 * mvv_lva(b, m)
        mx = max(logits.values())
        ws = {m: math.exp(min(20.0, (v - mx))) for m, v in logits.items()}
        z = sum(ws.values())
        return {m: w / z for m, w in ws.items()}

    def expand(self, node: Node):
        node.P = self.priors(node); node.expanded = True; nodes.tick("gs")

    def select(self, node: Node):
        best, best_u = None, -1e9
        sq = math.sqrt(node.N + 1)
        for m, p in node.P.items():
            ch = node.children.get(m)
            q = -ch.Q if ch else 0.0                                          # child's value is from the child's mover: negate
            u = q + self.c * p * sq / (1 + (ch.N if ch else 0))
            if u > best_u:
                best, best_u = m, u
        return best

    def leaf_value(self, b: chess.Board, side: chess.Color) -> float:
        if b.is_game_over():
            return squash(evaluate(b, side))
        return squash(self.resolved(b, side))

    def run(self, board: chess.Board) -> dict:
        root = Node(board.copy(stack=False), board.turn)
        self.expand(root)
        for _ in range(self.sims):
            node = root; path = [root]
            while node.expanded and not node.board.is_game_over():
                m = self.select(node)
                ch = node.children.get(m)
                if ch is None:
                    nb = node.board.copy(stack=False); nb.push(m)
                    ch = Node(nb, nb.turn); node.children[m] = ch; path.append(ch)
                    break
                node = ch; path.append(node)
            leaf = path[-1]
            v = self.leaf_value(leaf.board, leaf.board.turn)                # from the leaf's mover
            if not leaf.expanded and not leaf.board.is_game_over() and leaf.N >= 1:
                self.expand(leaf)
            # backup: v is from the leaf mover's side; flip along the path
            for n_ in reversed(path):
                n_.N += 1
                n_.W += v if n_.board.turn == leaf.board.turn else -v
        best = max(root.P, key=lambda m: (root.children[m].N if m in root.children else 0, root.P[m]))
        visits = {m.uci(): (root.children[m].N if m in root.children else 0) for m in root.P}
        top = sorted(visits.items(), key=lambda kv: -kv[1])[:5]
        return dict(move=best, value=-root.children[best].Q if best in root.children else 0.0, visits=top, prior=root.P[best])
