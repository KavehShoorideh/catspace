"""
nn/policy_fb.py — FBBoardPolicy: greedy readout of the TorchFB cone on real
boards. depth=1 scores own successors by F(s')@z; depth=2 applies the MIN
over opponent replies (README lesson 3: the readout's opponent model matters
as much as the field's) with every grandchild encoded in ONE batched forward.

Terminal ordering (README lesson 5, terminal scoring is load-bearing):
  I deliver mate  +1e9   >   any reach value   >   draw  -1e9   >   I get mated  -2e9
Draws score BAD because the policy is hunting its mate goal -- the toy
convention (draw at the 0.1% reach quantile) carried over.

Works for either color: pass the z that matches the side this policy plays
(zMATE_W when playing white, zMATE_B when playing black).
"""
from __future__ import annotations

import chess
import numpy as np
import torch

from catspace.data.encode import encode_meta, encode_packed
from catspace.nn.features import feature_planes, omega_ids

MATE_SCORE = 1e9
DRAW_SCORE = -1e9
MATED_SCORE = -2e9


def soft_min_bank(fb, f: torch.Tensor, Z: torch.Tensor, tau: float) -> torch.Tensor:
    """(n,) soft-min-distance region score of states f against an (m, d)
    exemplar bank Z: tau * (logsumexp(score/tau) - log(m)). Normalized so a
    bank of m identical exemplars scores exactly like that single goal.
    See FBSearchPolicy.__init__ for the play-tested rationale (hard max
    and plain centroid were both REJECTED at play, 2026-07-13)."""
    S = fb.score_matrix(f, Z)
    m = S.shape[1]
    return tau * (torch.logsumexp(S / tau, dim=1)
                  - torch.log(torch.tensor(float(m), device=S.device)))


class FBBoardPolicy:
    def __init__(self, fb, z, depth: int = 2, elo: int = 1800, clock: float = 300.0,
                 device: str = "cpu"):
        assert depth in (1, 2)
        self.fb = fb.to(device).eval()
        self.z = torch.as_tensor(z, dtype=torch.float32, device=device)
        self.depth = depth
        self.device = device
        self._omega_row = omega_ids(np.array([elo]), np.array([elo]), np.array([clock]))[0]

    @torch.no_grad()
    def _reach(self, boards: list[chess.Board]) -> np.ndarray:
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        planes = torch.from_numpy(feature_planes(packed, meta)).to(self.device)
        om = torch.from_numpy(np.tile(self._omega_row, (len(boards), 1))).to(self.device)
        f = self.fb.embed_F(planes, om)
        if self.z.dim() == 2:            # goal bank (see FBSearchPolicy.__init__)
            return soft_min_bank(self.fb, f, self.z, 0.1).cpu().numpy()
        return self.fb.score(f, self.z).cpu().numpy()

    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        return self.move_scored(board, rng)[0]

    def move_scored(self, board: chess.Board, rng: np.random.Generator
                    ) -> tuple[chess.Move, list[dict]]:
        """As move(), but also returns per-candidate dicts for viz/debugging:
        {uci, san, score, kind} where kind in {"mate","draw","mated","reach"};
        depth-2 reach candidates additionally carry {feared_uci, feared_san,
        feared_score} = the opponent reply that produced the min. Terminal
        sentinels (+-1e9/-2e9) stay in `score` for sorting; use `kind` to
        display them, not the raw number."""
        moves = list(board.legal_moves)
        scores = np.full(len(moves), -np.inf)
        pending: list[tuple[int, chess.Board]] = []      # depth-1 leaves
        pending2: list[tuple[int, chess.Board]] = []     # depth-2 leaves (i = my move idx)
        feared: dict[int, tuple[chess.Move, chess.Board]] = {}  # i -> (worst-so-far reply, board)
        kinds: dict[int, str] = {}

        for i, m in enumerate(moves):
            child = board.copy(stack=False)
            child.push(m)
            if child.is_checkmate():
                san = board.san(m)
                return m, [dict(uci=m.uci(), san=san, score=MATE_SCORE, kind="mate", chosen=True)]
            if child.is_game_over(claim_draw=True):
                scores[i] = DRAW_SCORE
                kinds[i] = "draw"
                continue
            if self.depth == 1:
                pending.append((i, child))
                kinds[i] = "reach"
                continue
            worst = np.inf
            replies = []
            for r in child.legal_moves:
                grand = child.copy(stack=False)
                grand.push(r)
                if grand.is_checkmate():
                    worst = MATED_SCORE                    # opponent mates me
                    feared[i] = (r, grand)
                    break
                if grand.is_game_over(claim_draw=True):
                    if DRAW_SCORE < worst:
                        worst = DRAW_SCORE
                        feared[i] = (r, grand)
                    continue
                replies.append((r, grand))
            if worst <= MATED_SCORE:
                scores[i] = MATED_SCORE
                kinds[i] = "mated"
            elif not replies:
                scores[i] = worst if np.isfinite(worst) else DRAW_SCORE
                kinds[i] = "draw"
            else:
                pending2.extend((i, r, g) for r, g in replies)
                scores[i] = worst                          # min over terminal replies so far
                kinds[i] = "reach"

        if pending:
            reach = self._reach([b for _, b in pending])
            for (i, _), v in zip(pending, reach):
                scores[i] = v

        # depth-2: track, per candidate move i, the (reply, board, reach) with
        # the LOWEST reach seen so far -- that reply is what "feared" means.
        pending2_by_i: dict[int, list[tuple[chess.Move, chess.Board]]] = {}
        for i, r, g in pending2:
            pending2_by_i.setdefault(i, []).append((r, g))
        if pending2:
            reach = self._reach([g for _, _, g in pending2])
            reach_by_i: dict[int, list[tuple[chess.Move, chess.Board, float]]] = {}
            for (i, r, g), v in zip(pending2, reach):
                reach_by_i.setdefault(i, []).append((r, g, float(v)))
            for i, entries in reach_by_i.items():
                r, g, v = min(entries, key=lambda e: e[2])
                prev = scores[i]
                new_score = min(prev, v) if np.isfinite(prev) else v
                if new_score != prev:
                    feared[i] = (r, g)
                scores[i] = new_score

        best = int(np.argmax(scores))
        moves_out = moves[best]

        cands = []
        for i, m in enumerate(moves):
            san = board.san(m)
            entry = dict(uci=m.uci(), san=san, score=float(scores[i]),
                        kind=kinds.get(i, "reach"), chosen=(i == best))
            if i in feared:
                r, g = feared[i]
                child = board.copy(stack=False); child.push(m)
                entry["feared_uci"] = r.uci()
                entry["feared_san"] = child.san(r)
                entry["feared_score"] = float(scores[i])
            cands.append(entry)
        cands.sort(key=lambda c: (-c["score"], not c["chosen"]))
        return moves_out, cands


class _SearchNode:
    """One position in an FBSearchPolicy game tree. `kind` is None for a
    position that still needs a reach evaluation (either an interior node
    with children, or a plain leaf); "mate_here"/"draw" mark terminals,
    which never get a reach call."""
    __slots__ = ("board", "move", "children", "score", "kind")

    def __init__(self, board: chess.Board, move: chess.Move):
        self.board = board
        self.move = move
        self.children: list["_SearchNode"] = []
        self.score: float | None = None
        self.kind: str | None = None


class FBSearchPolicy:
    """Beam-limited minimax over the TorchFB cone: F(s)@z is STILL the only
    learned signal (no retraining), but read out with genuine multi-ply
    lookahead instead of FBBoardPolicy's hardcoded depth-1/2 special cases.

    2026-07-11 rationale (see JOURNAL.md): after 4 rounds of tuning the
    embedding alone (LR schedule, 4x data, more steps) left arena score vs
    even the weakest Stockfish flat (~0.08-0.10, essentially "losing every
    game"), matching the literature and this repo's OWN documented
    expectation (arena_real.py's docstring: imitation-bootstrapped +
    no-search greedy readout losing to Stockfish is the EXPECTED baseline).
    Web research on combining learned value functions with search confirms
    deeper lookahead is the practical lever the literature identifies as
    actually closing this kind of gap, ahead of a full self-play/PI-
    refinement retraining loop.

    2026-07-11 update (Kaveh's call): search itself isn't the novelty here
    -- the point of a good PLAN is to need LESS search, not more. Rather
    than tune ply-depth, this now takes a fixed NODE BUDGET (`max_nodes`)
    modeled on Leela Chess Zero's own node economy: ~1500-2000 nodes/move
    is a reasonable "Leela actually playing" reference point (not its
    self-play floor of ~800, not its ~128k diminishing-returns ceiling),
    so ~150-200 nodes is our 10x-below-Leela target -- deliberately much
    less search than a strong engine uses, so any win margin has to come
    from the PLAN, not from out-searching the opponent. `max_nodes` and
    `beam` are BOTH externally-set constants for the current research
    phase (see JOURNAL.md) -- neither is tuned by the loop; depth is
    DERIVED per move from the real root branching (free -- it's just
    len(legal_moves)) to spend the fixed budget as deep as it reaches.

    Full alpha-beta pruning would need serial (one-at-a-time) leaf
    evaluation, losing the GPU-batching this codebase relies on throughout
    (FBBoardPolicy's depth-2, planner/decompose.py, etc.) -- so this
    implements plain (unpruned) minimax instead, with a BEAM cap on
    branching at every ply after the root (ranked by a cheap one-ply reach
    heuristic) to keep the tree small enough for ONE batched forward pass
    over every leaf. Root branching is never capped -- every legal move
    gets a real, fully-searched score, so the top-level choice never
    silently drops a candidate.

    Score convention matches FBBoardPolicy exactly: I deliver mate ->
    MATE_SCORE, draw -> DRAW_SCORE, I get mated -> MATED_SCORE, otherwise
    the raw F(s)@z reach (board.turn is itself an input feature, so reach
    is already turn-aware -- no negamax sign-flipping needed, matching how
    FBBoardPolicy's depth-2 already scores grandchildren directly).
    """

    def __init__(self, fb, z, max_nodes: int, beam: int = 4, elo: int = 1800,
                 clock: float = 300.0, device: str = "cpu"):
        assert max_nodes >= 1 and beam >= 1
        self.fb = fb.to(device).eval()
        # z may be a single goal embedding (d,) or a GOAL BANK (m, d): a set
        # of exemplar B-embeddings scored as a soft-min-distance REGION.
        # 2026-07-13 findings (JOURNAL.md): distance to any single mate
        # CENTROID is flat against true plies-to-mate (averaging exemplars
        # destroys the structure); nearest-exemplar distance correlates
        # (rho +0.17 -> +0.25) BUT a hard max-over-bank readout LOSES at
        # play on both checkpoints (e=65 and e=2.8e7, REJECT) -- it chases
        # whichever exemplar is closest each ply, destabilizing MOVE
        # ranking. bank_tau soft-min (normalized logsumexp) keeps region
        # structure while blending nearby exemplars: tau -> 0 recovers the
        # (rejected) hard max, larger tau smooths toward the (rejected)
        # centroid -- the play-tested middle is the point of the knob.
        self.z = torch.as_tensor(z, dtype=torch.float32, device=device)
        assert self.z.dim() in (1, 2)
        self.max_nodes = max_nodes
        self.beam = beam
        self.bank_tau = 0.1
        self.device = device
        self.last_depth_used: int | None = None    # set by move(), for introspection/testing
        self._omega_row = omega_ids(np.array([elo]), np.array([elo]), np.array([clock]))[0]

    def _bank_scores(self, f: "torch.Tensor") -> "torch.Tensor":
        return soft_min_bank(self.fb, f, self.z, self.bank_tau)

    def _depth_for_budget(self, root_branching: int) -> int:
        """Largest D such that a uniform (root_branching, beam) tree --
        root_branching + root_branching*beam + root_branching*beam^2 + ...
        -- stays within max_nodes. Always returns at least 1 (the root
        ply alone), even if that already exceeds the budget."""
        if root_branching <= 0:
            return 1
        total = level = root_branching
        depth = 1
        while True:
            next_level = level * self.beam
            next_total = total + next_level
            if next_total > self.max_nodes:
                return depth
            total, level, depth = next_total, next_level, depth + 1

    @torch.no_grad()
    def _reach_batch(self, boards: list[chess.Board]) -> np.ndarray:
        if not boards:
            return np.zeros(0, dtype=np.float32)
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        planes = torch.from_numpy(feature_planes(packed, meta)).to(self.device)
        om = torch.from_numpy(np.tile(self._omega_row, (len(boards), 1))).to(self.device)
        f = self.fb.embed_F(planes, om)
        if self.z.dim() == 2:            # goal bank: soft-min region readout
            return self._bank_scores(f).cpu().numpy()
        return self.fb.score(f, self.z).cpu().numpy()

    def _make_children(self, board: chess.Board) -> list["_SearchNode"]:
        nodes = []
        for m in board.legal_moves:
            child = board.copy(stack=False)
            child.push(m)
            node = _SearchNode(child, m)
            if child.is_checkmate():
                node.kind = "mate_here"
            elif child.is_game_over(claim_draw=True):
                node.kind = "draw"
            nodes.append(node)
        return nodes

    def _beam_cap(self, nodes: list["_SearchNode"]) -> list["_SearchNode"]:
        if len(nodes) <= self.beam:
            return nodes
        mates = [n for n in nodes if n.kind == "mate_here"]
        if len(mates) >= self.beam:
            return mates[: self.beam]
        scoreable = [n for n in nodes if n.kind is None]
        draws = [n for n in nodes if n.kind == "draw"]
        need = self.beam - len(mates)
        ranked = scoreable
        if scoreable:
            r = self._reach_batch([n.board for n in scoreable])
            ranked = [scoreable[i] for i in np.argsort(-r)]
        kept = ranked[:need]
        if len(kept) < need:
            kept = kept + draws[: need - len(kept)]
        return mates + kept

    def _expand(self, node: "_SearchNode", remaining_depth: int) -> None:
        if node.kind is not None or remaining_depth == 0:
            return
        children = self._make_children(node.board)
        if not children:
            node.kind = "draw"          # stalemate or no legal moves, not caught above
            return
        node.children = self._beam_cap(children)
        for c in node.children:
            self._expand(c, remaining_depth - 1)

    def _score(self, node: "_SearchNode", ply: int) -> float:
        if not node.children:
            if node.kind == "mate_here":
                # discount by ply so a FASTER mate strictly outscores a
                # slower one (and getting mated later strictly outscores
                # sooner) -- otherwise every mate within the horizon ties at
                # the flat sentinel and argmax may pick a needlessly slow
                # one. `ply` is at most `depth` (a handful), so this never
                # risks crossing into reach-score territory (~[-1, 1]).
                return (MATE_SCORE - ply) if ply % 2 == 1 else (MATED_SCORE + ply)
            if node.kind == "draw":
                return DRAW_SCORE
            return float(node.score)
        child_scores = [self._score(c, ply + 1) for c in node.children]
        return max(child_scores) if ply % 2 == 0 else min(child_scores)

    def _build_and_score(self, board: chess.Board) -> tuple[list["_SearchNode"], list[float]]:
        roots = self._make_children(board)
        if not roots:
            raise ValueError("no legal moves")
        depth = self._depth_for_budget(len(roots))
        self.last_depth_used = depth
        for r in roots:
            self._expand(r, depth - 1)

        # one batched reach call over every leaf that still needs a score
        # (interior placeholders never reach here: _expand only leaves
        # node.kind is None on nodes with no children, i.e. real leaves)
        def collect_leaves(node: "_SearchNode", out: list["_SearchNode"]) -> None:
            if not node.children:
                if node.kind is None:
                    out.append(node)
                return
            for c in node.children:
                collect_leaves(c, out)

        leaves: list["_SearchNode"] = []
        for r in roots:
            collect_leaves(r, leaves)
        if leaves:
            reach = self._reach_batch([n.board for n in leaves])
            for n, v in zip(leaves, reach):
                n.score = float(v)

        scores = [self._score(r, ply=1) for r in roots]
        return roots, scores

    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        roots, scores = self._build_and_score(board)
        best = int(np.argmax(scores))
        return roots[best].move

    def reliability(self, board: chess.Board, shallow_keep_frac: float = 0.5) -> float:
        """METHOD 1 sharpness sensor (2026-07-13, UNCERTAINTY_DESIGN.md): how
        much does DEEP search reorder the root moves versus a SHALLOW (1-ply
        reach) look? Among the shallow-PLAUSIBLE moves (top `shallow_keep_frac`
        by 1-ply reach -- this filters the obvious 1-ply blunders, which look
        bad shallowly AND deeply so they don't create disagreement anyway),
        low rank-agreement between shallow and deep = "thinking harder changed
        my mind here" = the model is UNRELIABLE / the position is SHARP. This is
        self-referential (no external label): it measures the engine's own
        instability, and it's reachability-native (both looks are F(.)@z).

        Returns a disagreement score in [0, 1]: 0 = quiet (deep agrees with the
        shallow ranking, extra search is wasted), 1 = maximally sharp (deep
        completely reorders the plausible moves)."""
        roots, deep = self._build_and_score(board)
        if len(roots) < 3:
            return 0.0
        shallow = self._reach_batch([r.board for r in roots])
        deep = np.asarray(deep, dtype=float)
        k = max(3, int(round(len(roots) * shallow_keep_frac)))
        keep = np.argsort(-shallow)[:k]                      # shallow-plausible moves
        sh, dp = shallow[keep], deep[keep]
        sr = np.argsort(np.argsort(sh)).astype(float)        # spearman = pearson on ranks
        dr = np.argsort(np.argsort(dp)).astype(float)
        if sr.std() < 1e-9 or dr.std() < 1e-9:
            return 0.0
        rho = float(np.corrcoef(sr, dr)[0, 1])
        return float((1.0 - rho) / 2.0)                      # [-1,1] rho -> [0,1] disagreement

    def plan(self, board: chess.Board, rng: np.random.Generator) -> tuple[chess.Move, chess.Board]:
        """Like move(), but also walks the PRINCIPAL VARIATION -- the
        sequence of backed-up-best children from the chosen root move down
        to a leaf -- and returns that leaf's board as a SUBGOAL: the
        position this search's own best-response line predicts play heads
        toward, several moves out. FBPlanPolicy commits to this subgoal
        instead of re-deriving a full-depth plan every ply."""
        roots, scores = self._build_and_score(board)
        best = int(np.argmax(scores))
        node = roots[best]
        ply = 1
        while node.children:
            child_scores = [self._score(c, ply + 1) for c in node.children]
            pick = int(np.argmax(child_scores) if ply % 2 == 0 else np.argmin(child_scores))
            node = node.children[pick]
            ply += 1
        return roots[best].move, node.board


class FBPlanPolicy:
    """Composes two FBSearchPolicy instances -- a DEEP planner and a cheap
    SHALLOW executor -- so the game doesn't re-run a full-depth search every
    ply when nothing about the plan has actually changed. Mirrors the toy-
    domain catspace/planner/plans.py PlanMemory/GreedyReach design (propose
    a plan, commit while it's ACTIVE, replan on drop/stall/achieved) but
    with exactly one always-active plan instead of a pool.

    2026-07-11 rationale (Kaveh): "the plan shouldn't change if the
    materials have just moved around the board without actually changing
    ... there's no need to re-search the plan space every time." The deep
    planner's own principal variation (FBSearchPolicy.plan()) stands in for
    planner/decompose.py's externally-sourced WaypointPool subgoal -- no
    separate waypoint search needed, since choosing the first move already
    computes one PV as a side effect.

    Replan triggers, checked each ply via reach-to-subgoal (F(s)@B(subgoal),
    both L2-normalized so this is a cosine similarity in [-1, 1]):
      - ACHIEVED: reach >= achieved_cos (subgoal effectively reached)
      - STALLED: max_plies_per_plan shallow moves played since the last plan
      - DROPPED: reach fell more than drop_delta below its value when the
        plan was made (something went wrong relative to the predicted line)
    Otherwise the cheap executor picks the move, holding the plan fixed.
    """

    def __init__(self, fb, z, plan_nodes: int = 2000, plan_beam: int = 4,
                 shallow_nodes: int = 60, shallow_beam: int = 3,
                 max_plies_per_plan: int = 6, drop_delta: float = 0.15,
                 achieved_cos: float = 0.95, elo: int = 1800, clock: float = 300.0,
                 device: str = "cpu"):
        self._planner = FBSearchPolicy(fb, z, max_nodes=plan_nodes, beam=plan_beam,
                                        elo=elo, clock=clock, device=device)
        self._executor = FBSearchPolicy(fb, z, max_nodes=shallow_nodes, beam=shallow_beam,
                                         elo=elo, clock=clock, device=device)
        self.max_plies_per_plan = max_plies_per_plan
        self.drop_delta = drop_delta
        self.achieved_cos = achieved_cos
        self.device = device
        self.subgoal_board: chess.Board | None = None   # public: viewer overlay reads this
        self._subgoal_z: torch.Tensor | None = None
        self._plies_since_plan = 0
        self._reach_at_plan: float | None = None
        self.plans_made = 0
        self.last_replanned = False       # introspection/testing
        self.last_replan_reason: str | None = None

    @torch.no_grad()
    def _reach_to(self, board: chess.Board, z: torch.Tensor) -> float:
        packed = encode_packed(board)[None]
        meta = encode_meta(board)[None]
        planes = torch.from_numpy(feature_planes(packed, meta)).to(self.device)
        om = torch.from_numpy(np.tile(self._executor._omega_row, (1, 1))).to(self.device)
        f = self._executor.fb.embed_F(planes, om)
        return float(self._executor.fb.score(f, z)[0].cpu().numpy())

    @torch.no_grad()
    def _embed_subgoal(self, subgoal_board: chess.Board) -> torch.Tensor:
        packed = encode_packed(subgoal_board)[None]
        meta = encode_meta(subgoal_board)[None]
        planes = torch.from_numpy(feature_planes(packed, meta)).to(self.device)
        return self._executor.fb.embed_B(planes)[0]

    def _replan(self, board: chess.Board, rng: np.random.Generator, reason: str) -> chess.Move:
        move, subgoal_board = self._planner.plan(board, rng)
        self.subgoal_board = subgoal_board
        self._subgoal_z = self._embed_subgoal(subgoal_board)
        self._executor.z = self._subgoal_z
        self._plies_since_plan = 0
        self._reach_at_plan = self._reach_to(board, self._subgoal_z)
        self.plans_made += 1
        self.last_replanned = True
        self.last_replan_reason = reason
        return move

    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        if self._subgoal_z is None:
            return self._replan(board, rng, "initial")

        reach_now = self._reach_to(board, self._subgoal_z)
        if reach_now >= self.achieved_cos:
            return self._replan(board, rng, "achieved")
        if self._plies_since_plan >= self.max_plies_per_plan:
            return self._replan(board, rng, "stalled")
        if self._reach_at_plan is not None and reach_now < self._reach_at_plan - self.drop_delta:
            return self._replan(board, rng, "dropped")

        self.last_replanned = False
        self.last_replan_reason = None
        self._plies_since_plan += 1
        return self._executor.move(board, rng)


class FBTwoHorizonPolicy(FBSearchPolicy):
    """Two-horizon readout (TWO_HORIZON_DESIGN.md). Same beam search as
    FBSearchPolicy, but scores positions with one of the two trained heads:

      mode="far"  : the FAR head's calibrated distance-to-goal (long-range
                    strategy). Identical machinery to FBSearchPolicy on the
                    far head/goal -- this IS the strategist.
      mode="near" : the NEAR head's cosine reach to a near goal (short-range
                    endgame precision). z_near may be a single centroid (d,)
                    or an exemplar BANK (m, d) scored by soft-max cosine
                    (nearest-exemplar, since centroids are flat -- 2026-07-13).

    Both modes drive the SAME search (beam + leaves), so each is a clean,
    single-variable ablation the pre-registered gate compares against the
    incumbent: does far fix ACPL (long-range calibration)? does near fix
    KRRvKBP conversion (endgame precision)? Near and far scores live on
    different scales, so a range-gated HANDOFF between them is deliberately
    deferred until the pure modes show which head earns which failure mode."""

    def __init__(self, fb, z_far, z_near, max_nodes: int, beam: int = 4,
                 mode: str = "far", elo: int = 1800, clock: float = 300.0,
                 device: str = "cpu"):
        assert getattr(fb, "two_horizon", False), "needs a two_horizon checkpoint"
        assert mode in ("far", "near")
        super().__init__(fb, z_far, max_nodes, beam, elo=elo, clock=clock, device=device)
        self.mode = mode
        self.z_near = torch.as_tensor(z_near, dtype=torch.float32, device=device)
        assert self.z_near.dim() in (1, 2)

    @torch.no_grad()
    def _reach_batch(self, boards: list[chess.Board]) -> np.ndarray:
        if self.mode == "far":
            return super()._reach_batch(boards)   # far head + z_far
        if not boards:
            return np.zeros(0, dtype=np.float32)
        packed = np.stack([encode_packed(b) for b in boards])
        meta = np.stack([encode_meta(b) for b in boards])
        planes = torch.from_numpy(feature_planes(packed, meta)).to(self.device)
        om = torch.from_numpy(np.tile(self._omega_row, (len(boards), 1))).to(self.device)
        f = self.fb.embed_F_near(planes, om)
        if self.z_near.dim() == 2:            # near exemplar bank -> soft-max cosine
            S = f @ self.z_near.T             # (n, m) cosine sims; higher = closer
            return (self.bank_tau * torch.logsumexp(S / self.bank_tau, dim=1)).cpu().numpy()
        return (f @ self.z_near).cpu().numpy()
