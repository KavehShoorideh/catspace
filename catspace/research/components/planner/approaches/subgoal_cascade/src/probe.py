"""
planner/probe.py — the probe() primitive (PLANNER_PROBE_DESIGN.md phase A).

A probe is a bounded MCTS run packaged as an EVALUATION, not a move choice:
the planner calls it to ask "what does a `budget`-eval search say about this
position?", gets back a structured ProbeResult, and may later `deepen()` the
same tree instead of re-searching from scratch. The MCTS instance supplied by
the caller carries the region/readout (committor to the W surface today; any
reach_fn tomorrow) and the opponent model (its min-node behavior), so this
layer adds no policy of its own.

Certification discipline (the 0.60->0.20 lesson, non-negotiable): the
[lo, hi] interval is GAME-TRUTH ONLY — derived from rules-terminal children
(mate / insufficient material / 50-move / path-aware threefold) at depth 1.
Certainty-resolved soft-terminals (network confidence) NEVER tighten it: a
node whose terminal_v came from the recognizer is excluded from the bound.
Scope: with White to move, lo = the best game-truth child (White may simply
play it), hi = MATE_V unless EVERY child is game-truth (then the interval
pins); Black to move is the mirror. No deeper proof propagation yet — a
subtree solved three plies down does not tighten the root bound. `value` is
the search's point estimate (root Q) and is NOT certified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import chess

from catspace.research.components.search.approaches.puct_mcts.src.mcts import MATE_V, MATED_V, MCTS, _Node, game_truth as _game_truth


@dataclass
class ProbeResult:
    value: float                       # root Q, White-POV, [-1, 1] — NOT certified
    best_move: chess.Move | None       # what the search would play
    lo: float                          # certified White-POV bounds (game truth
    hi: float                          # only; (-1, 1)-vacuous when nothing proven)
    hits: dict = field(default_factory=dict)   # visit-weighted leaf categories
    coherence: float = 1.0             # root coh_gamma (1.0 when coherence off)
    evals_spent: int = 0               # fresh NN evals this call
    cache_hits: int = 0
    visit_top2: tuple = (0, 0)         # (N_best, N_second) — decision stability
    tree: _Node | None = None          # root, for deepen()/reuse
    board_fen: str = ""

    @property
    def decided(self) -> bool:
        """Certified-decided: the interval is a point (nothing else to learn)."""
        return self.lo == self.hi


def _leaf_hits(root: _Node) -> dict:
    """Visit-weighted census of where simulations ENDED. Categories:
    mate_w / mate_b / draw (game truth), resolved (certainty soft-terminal),
    frontier (unexpanded — the field's word was taken)."""
    hits: dict[str, int] = {}
    stack = [root]
    while stack:
        n = stack.pop()
        if n.children:
            stack.extend(n.children)
            continue
        if n.N == 0:
            continue
        if _game_truth(n):
            cat = ("mate_w" if n.terminal_v > 0.5 else
                   "mate_b" if n.terminal_v < -0.5 else "draw")
        elif n.terminal_v is not None:
            cat = "resolved"
        else:
            cat = "frontier"
        hits[cat] = hits.get(cat, 0) + n.N
    return hits


def _certified(root: _Node) -> tuple[float, float]:
    """Depth-1 game-truth bounds (see module docstring for exact scope)."""
    if not root.children:
        v = root.terminal_v if root.terminal_v is not None else 0.0
        return (v, v)                       # stalemate/checkmate at the root
    white = root.board.turn == chess.WHITE
    truth = [c.terminal_v for c in root.children if _game_truth(c)]
    lo, hi = MATED_V, MATE_V
    if truth:
        if white:
            lo = max(truth)                 # White may simply play the best proven child
        else:
            hi = min(truth)
    if len(truth) == len(root.children):    # every reply proven => interval pins
        v = max(truth) if white else min(truth)
        lo = hi = v
    return (lo, hi)


def _best_child(root: _Node) -> _Node | None:
    """Mirror FBMCTSPolicy.move: game-truth mate first, else (N, mover-POV Q)."""
    if not root.children:
        return None
    white = root.board.turn == chess.WHITE
    for c in root.children:
        if _game_truth(c) and (c.terminal_v > 0.5 if white else c.terminal_v < -0.5):
            return c
    return max(root.children,
               key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None else c.Q)
                              * (1 if white else -1)))


def _summarize(mcts: MCTS, root: _Node, cache_hits0: int) -> ProbeResult:
    best = _best_child(root)
    lo, hi = _certified(root)
    visits = sorted((c.N for c in root.children), reverse=True)
    top2 = (visits[0] if visits else 0, visits[1] if len(visits) > 1 else 0)
    value = root.terminal_v if root.terminal_v is not None else root.Q
    return ProbeResult(value=float(value),
                       best_move=best.move if best is not None else None,
                       lo=float(lo), hi=float(hi),
                       hits=_leaf_hits(root), coherence=float(root.coh_gamma),
                       evals_spent=mcts.evals_used,
                       cache_hits=mcts.cache_hits - cache_hits0,   # per-call, like
                       visit_top2=top2, tree=root,                 # evals_spent
                       board_fen=root.board.fen())


def probe(mcts: MCTS, board: chess.Board, budget: int | None = None,
          reuse: ProbeResult | None = None) -> ProbeResult:
    """Run a bounded search from `board` and package the outcome. `budget`
    overrides mcts.max_nodes for this call only (fresh evals, cache hits
    free). `reuse` continues a previous probe's tree when its root matches —
    the deepening path (visit statistics and calibration carry over).

    The stability stop (mcts.decision_stop) is suspended for the duration: it
    is keyed to move-argmax stability, and a probe wants the VALUE refined —
    stopping when the move is settled would silently truncate exactly the
    estimate the planner is paying for. The certified mate-stop stays active
    (it pins the value too)."""
    old, old_stab = mcts.max_nodes, mcts.decision_stop
    if budget is not None:
        mcts.max_nodes = budget
    mcts.decision_stop = False
    hits0 = mcts.cache_hits
    try:
        root = mcts.run(board, reuse_root=reuse.tree if reuse is not None else None)
    finally:
        mcts.max_nodes, mcts.decision_stop = old, old_stab
    return _summarize(mcts, root, hits0)


def deepen(mcts: MCTS, board: chess.Board, result: ProbeResult,
           extra_budget: int) -> ProbeResult:
    """Spend `extra_budget` MORE fresh evals on an existing probe's tree."""
    return probe(mcts, board, budget=extra_budget, reuse=result)
