"""
planner/decompose.py — meet-in-the-middle recursive decomposition over the FB
embedding (the M1.5 geodesic-midpoint generator, real-board flavor).

A hop s -> g is scored from both ends: reach(s, g) = score(F(s), z_g). When a
hop is not directly executable, candidate waypoints m from a WaypointPool are
scored by the BOTTLENECK of the two legs,

    score(m) = min( score(F(s), B(m)),  score(F(m), z_g) )

and the hop splits at the argmax — good waypoints are reachable from here AND
see the goal. `score` is pluggable (the `score_pairs` param): it defaults to
the raw dot product (cosine reach on L2-normalized embeddings — the original
behavior, and the second place the cosine-InfoNCE fix is load-bearing), but
quasimetric checkpoints MUST pass TorchFB.np_score_matrix instead, since a
raw dot never sees metric_scale/W (2026-07-12 review). Either way both legs
share one scale, which is what makes the min comparable.

Recursion stops per the agreed M1.5 give-up rules (the same rule vocabulary
plans.py::BlockReason reserves):
  no_midpoint         best waypoint doesn't beat the direct reach by min_gain
                      — the hop is HARD, not LONG; splitting can't help
  unlikely_territory  even the best split leaves the bottleneck below tau_floor
  dry_out             two successive splits each improved the bottleneck by
                      less than dry_gain — converging without arriving
  budget              depth cap (anytime: the tree so far is the answer)

The pool's F must be embedded under the PLANNER's omega (its own Elo/clock):
"can I route through m" is a question about the planning player, not about
whoever reached m in the source games. B is board-only, as always.

Executability here is reach >= tau_exec (calibrate as in plans.calibrate_tau);
MC-rollout verification of leaf hops ("a real path, verified not estimated")
is the demo/arena layer's job, pluggable via the executable_fn hook.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class WaypointPool:
    """Candidate waypoints: F under the planner's omega, board-only B, and a
    human-readable label (FEN) per row. Rows correspond across all three."""
    F: np.ndarray                    # (n, d) float32, L2-normalized
    B: np.ndarray                    # (n, d) float32, L2-normalized
    labels: list

    def __post_init__(self):
        assert self.F.shape == self.B.shape and len(self.labels) == len(self.F)

    def __len__(self) -> int:
        return len(self.F)


@dataclass
class HopNode:
    """One hop of the plan tree. Leaves are either executable or blocked;
    internal nodes carry the waypoint the hop split at."""
    reach: float                     # direct reach of this hop
    depth: int
    status: str                      # "executable"|"decomposed"|block rule
    waypoint: int | None = None      # pool index (internal nodes only)
    bottleneck: float | None = None  # min(leg1, leg2) at the chosen waypoint
    left: "HopNode | None" = None
    right: "HopNode | None" = None
    detail: str = ""

    def leaves(self) -> list["HopNode"]:
        if self.left is None:
            return [self]
        return self.left.leaves() + self.right.leaves()


@dataclass
class Decomposition:
    root: HopNode
    pool: WaypointPool
    executable: bool                 # every leaf hop is executable
    block_rule: str | None           # weakest leaf's rule when not executable
    waypoints: list = field(default_factory=list)   # pool indices, in play order

    @property
    def plan_bottleneck(self) -> float:
        return min(leaf.reach for leaf in self.root.leaves())

    def subgoal_labels(self) -> list:
        return [self.pool.labels[i] for i in self.waypoints]


def _dot_pairs(F: np.ndarray, B: np.ndarray) -> np.ndarray:
    return F @ B.T


def hop_reach(F_s: np.ndarray, z_g: np.ndarray,
              score_pairs: Callable[[np.ndarray, np.ndarray], np.ndarray] = _dot_pairs) -> float:
    return float(score_pairs(F_s[None, :], z_g[None, :])[0, 0])


def waypoint_scores(F_s: np.ndarray, z_g: np.ndarray, pool: WaypointPool,
                    score_pairs: Callable[[np.ndarray, np.ndarray], np.ndarray] = _dot_pairs
                    ) -> np.ndarray:
    """(n,) bottleneck score of routing s -> m -> g through each pool row.

    score_pairs(F: (n,d), B: (m,d)) -> (n,m) defaults to the raw dot
    product (the original cosine-reach behavior). For quasimetric
    checkpoints pass TorchFB.np_score_matrix instead -- the raw dot is
    mis-calibrated there (it never sees metric_scale/W; 2026-07-12
    review). Both legs stay on one shared scale either way, which is what
    keeps the min() comparable."""
    leg1 = score_pairs(F_s[None, :], pool.B)[0]     # reach of s toward each m
    leg2 = score_pairs(pool.F, z_g[None, :])[:, 0]  # reach of each m toward g
    return np.minimum(leg1, leg2)


def decompose(F_s: np.ndarray, z_g: np.ndarray, pool: WaypointPool,
              tau_exec: float, tau_floor: float,
              min_gain: float = 0.0, dry_gain: float = 0.02,
              max_depth: int = 3,
              executable_fn: Callable[[float], bool] | None = None,
              score_pairs: Callable[[np.ndarray, np.ndarray], np.ndarray] = _dot_pairs
              ) -> Decomposition:
    """Recursively split the hop F_s -> z_g at bottleneck-maximizing waypoints
    until every leaf is executable or a give-up rule fires. Anytime: the tree
    is returned whichever way it ends. Waypoints are consumed (a plan never
    routes through the same pool row twice)."""
    is_exec = executable_fn or (lambda r: r >= tau_exec)
    used = np.zeros(len(pool), dtype=bool)
    dry_run = {"n": 0}               # consecutive low-gain splits, shared DFS state

    def _rec(F_a: np.ndarray, z_b: np.ndarray, depth: int) -> HopNode:
        direct = hop_reach(F_a, z_b, score_pairs)
        if is_exec(direct):
            return HopNode(reach=direct, depth=depth, status="executable")
        if depth >= max_depth:
            return HopNode(reach=direct, depth=depth, status="budget",
                           detail=f"depth cap {max_depth}")

        scores = waypoint_scores(F_a, z_b, pool, score_pairs)
        scores[used] = -np.inf
        m = int(np.argmax(scores))
        bot = float(scores[m])
        gain = bot - direct

        if gain <= min_gain:
            return HopNode(reach=direct, depth=depth, status="no_midpoint",
                           waypoint=m, bottleneck=bot,
                           detail=f"best gain {gain:+.4f} <= {min_gain:+.4f} (hard, not long)")
        if bot < tau_floor:
            return HopNode(reach=direct, depth=depth, status="unlikely_territory",
                           waypoint=m, bottleneck=bot,
                           detail=f"bottleneck {bot:.4f} < floor {tau_floor:.4f}")
        if gain < dry_gain:
            dry_run["n"] += 1
            if dry_run["n"] >= 2:
                return HopNode(reach=direct, depth=depth, status="dry_out",
                               waypoint=m, bottleneck=bot,
                               detail=f"2 successive gains < {dry_gain:.4f}")
        else:
            dry_run["n"] = 0

        used[m] = True
        node = HopNode(reach=direct, depth=depth, status="decomposed",
                       waypoint=m, bottleneck=bot)
        node.left = _rec(F_a, pool.B[m], depth + 1)
        node.right = _rec(pool.F[m], z_b, depth + 1)
        return node

    root = _rec(np.asarray(F_s, dtype=np.float32), np.asarray(z_g, dtype=np.float32), 0)

    waypoints: list = []
    def _collect(n: HopNode) -> None:
        if n.left is not None:
            _collect(n.left)
            waypoints.append(n.waypoint)
            _collect(n.right)
    _collect(root)

    leaves = root.leaves()
    blocked = [n for n in leaves if n.status not in ("executable",)]
    block_rule = None
    if blocked:
        block_rule = min(blocked, key=lambda n: n.reach).status
    return Decomposition(root=root, pool=pool, executable=not blocked,
                         block_rule=block_rule, waypoints=waypoints)
