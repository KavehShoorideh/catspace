"""Navigator component: (board, target region) -> move, via reach-guided MCTS.

Swap points: leaf ('reach' | 'committor' backups), order ('tiered' | 'none'
descent), opp_policy (None | maia2 priors at opponent nodes). The per-move memo
shares the trunk forward between the value and ordering paths.
"""
from __future__ import annotations

import chess
import numpy as np

from catspace.search.mcts import MCTS
from catspace.predictor.reach.region import RegionReach


class MCTSNavigator:
    def __init__(self, reach: RegionReach, rf, atlas, cg=None, opp_policy=None,
                 leaf: str = "reach", order: str = "tiered", nodes: int = 200,
                 c_puct: float = 1.5, prior_tau: float = 0.5):
        assert leaf in ("reach", "committor") and order in ("tiered", "none")
        if leaf == "committor" or order == "tiered":
            assert cg is not None, "committor leaf / tiered order need a committor ckpt"
        self.reach = reach; self.rf = rf; self.atlas = atlas; self.cg = cg
        self.opp_policy = opp_policy; self.leaf = leaf; self.order = order
        self.nodes = nodes; self.c_puct = c_puct; self.prior_tau = prior_tau
        self.cache: dict = {}                       # outlives moves/games
        self.evals: list[int] = []                  # fresh evals per our-move

    def move(self, board, tid: int):
        our_white = board.turn == chess.WHITE
        memo: dict = {}

        def p_all(boards):
            rows = [memo.get(b.fen()) for b in boards]
            miss = [i for i, r in enumerate(rows) if r is None]
            if miss:
                phis = self.rf.phi([boards[i] for i in miss]).cpu().numpy()
                fresh = self.reach.heads(phis)[0]
                for j, i in enumerate(miss):
                    rows[i] = fresh[j]; memo[boards[i].fen()] = fresh[j]
            return np.stack(rows)

        if self.leaf == "committor":
            # VALUE AUTHORITY (Kaveh 2026-07-30 option b): backups = committor net
            # (prices material, knows conversion); plan keeps ordering/priors.
            def leaf_fn(boards):
                c = np.asarray(self.cg._committor(
                    [b.to_input_tensor().float().numpy() for b in boards]))
                return c - 0.5                               # White-POV already
        else:
            def leaf_fn(boards):
                pr = p_all(boards)[:, tid]
                return pr if our_white else -pr              # White-POV sign

        order_fn = None
        if self.cg is not None and self.order == "tiered":
            # TIERED ORDER (Kaveh 2026-07-29): (1) reach-to-chute, (2) -mated-mass
            # (geometric distance from my losing basin), (3) committor eval.
            def order_fn(boards, mover_white):
                P = p_all(boards)
                t1 = P[:, tid]
                t2 = -(P @ self.atlas.badness)               # our safety
                c = np.asarray(self.cg._committor(
                    [b.to_input_tensor().float().numpy() for b in boards]))
                w = np.stack([t1 if our_white else -t1,
                              t2 if our_white else -t2,
                              c - 0.5], 1)                   # White-POV tiers
                return w if mover_white else -w              # mover-POV

        ck = "c" if self.leaf == "committor" else tid        # committor is plan-free
        mcts = MCTS(leaf_fn, max_nodes=self.nodes, c_puct=self.c_puct,
                    prior_tau=self.prior_tau, cache=self.cache,
                    cache_key_fn=lambda b: f"{ck}|{b.fen()}",
                    pw_c=1.5 if order_fn else 0.0,
                    root_min_visits=10 if order_fn else 0,
                    mate_stop=order_fn is not None, order_fn=order_fn,
                    opp_policy_fn=self.opp_policy)
        mv = mcts.best_move(board)
        self.evals.append(mcts.evals_used)
        return mv


