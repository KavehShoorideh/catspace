"""catspace/engine/search.py -- MCTSSearch: the local searcher as a layer. Wraps
catspace.nn.mcts.MCTS with the (value, prior) sockets and returns a SearchOutcome with
real evals_used (strength-per-node accounting, DECISIONS sec 9)."""
from __future__ import annotations

import chess
import numpy as np

from catspace.engine.interfaces import SearchOutcome
from catspace.nn.mcts import MCTS


class MCTSSearch:
    def __init__(self, nodes: int = 400, mate_stop: bool = True, pw_c: float = 1.5,
                 root_min_visits: int = 10):
        self.nodes, self.mate_stop = nodes, mate_stop
        self.pw_c, self.root_min_visits = pw_c, root_min_visits

    def best_move(self, board: chess.Board, value=None, prior=None) -> SearchOutcome:
        """value: ValueModel | None (None -> constant 0 = pure search). prior: MovePrior |
        None. NOTE the AZ-style path (value consulted per expansion) requires BOTH; a value
        without a prior would be silently ignored by MCTS, so we default the prior to
        uniform whenever a value is supplied (the bug found 2026-07-22)."""
        reach = (lambda bs: np.zeros(len(bs), dtype=float))
        value_fn = (lambda bs: value.values(bs)) if value is not None else None
        policy_fn = (lambda b: prior.priors(b)) if prior is not None else None
        if value_fn is not None and policy_fn is None:
            policy_fn = lambda b: (lambda lm: {m: 1.0 / len(lm) for m in lm})(list(b.legal_moves))
        m = MCTS(reach, max_nodes=self.nodes, mate_stop=self.mate_stop, pw_c=self.pw_c,
                 root_min_visits=self.root_min_visits, policy_fn=policy_fn, value_fn=value_fn)
        root = m.run(board)
        white = board.turn == chess.WHITE
        best = max(root.children, key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None else c.Q)
                                                 * (1 if white else -1)))
        pv, node = [], best
        for _ in range(8):
            pv.append(node.move)
            if not node.children:
                break
            node = max(node.children, key=lambda c: c.N)
        return SearchOutcome(move=best.move, pv=pv, evals_used=m.evals_used,
                             stats=dict(root_children=len(root.children)))
