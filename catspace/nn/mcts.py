"""
mcts.py — production-grade PUCT MCTS readout over the FB reach field.

Replaces FBSearchPolicy's beam-minimax as the search layer (Kaveh, 2026-07-14):
same learned signal (F(s)@z reach, no retraining), read out with real
visit-guided search instead of a fixed-shape tree. AlphaZero-style PUCT with
the two adaptations a policy-net-less engine needs:

  * VALUE-ONLY expansion: there is no policy head, so expanding a node
    batch-evaluates ALL its children's reach in one GPU call; priors are a
    softmax over those child values (mover's perspective) and each child
    keeps its evaluated reach as a first-play value estimate. One expansion
    = one batched forward pass = len(children) node-budget units, so the
    node budget is directly comparable to FBSearchPolicy's leaf count.
  * SELF-CALIBRATING VALUE SQUASH: raw reach is an unbounded score (its
    scale differs per checkpoint), but PUCT's Q/U balance and terminal
    sentinels need a bounded scale. Each move() calibrates center/scale
    from the root children's reach and squashes with tanh into (-1, 1);
    terminals sit at/just outside the squash range: mate +1 (minus a per-ply
    discount so FASTER mates strictly dominate), mated -1, draw 0 -- neutral,
    which the White/Black sign-flip REQUIRES (a non-zero draw would read as a
    win for one side; 2026-07-17). Avoiding draws when winning is the draw-
    clearance term's job, not a distorted value.

Search values are ALWAYS White-POV (reach already conditions on side to
move); selection flips sign at Black-to-move nodes instead of negamaxing.

Deterministic by construction (no rollouts, no root noise by default):
argmax-visits at the root, ties broken by Q then move order -- required by
playout_ab's exact-paired methodology.
"""
from __future__ import annotations

import math

import chess
import numpy as np

# White-POV terminal values on a symmetric [-1, +1] scale (== the "win 2 / draw 1
# / loss 0" convention, centered). DRAW MUST be 0: the search flips sign for the
# side to move (_select_child: `q if white else -q`), so a non-zero draw is
# inconsistent -- at DRAW_V=-0.999 a draw read as +0.999 for Black (≈ a Black
# win) while reading ≈ a loss for White. 0 = neutral for BOTH, as a draw is.
# (2026-07-17, Kaveh: the old -0.999 collapsed draw onto loss and broke the
# White/Black symmetry; it also made steering-to-a-draw-when-losing impossible.)
MATE_V = 1.0
MATED_V = -1.0
DRAW_V = 0.0
PLY_DISCOUNT = 1e-4          # mate at depth k backs up MATE_V - k*PLY_DISCOUNT


class _Node:
    __slots__ = ("board", "move", "children", "P", "N", "W", "v_init",
                 "terminal_v", "parent", "rep_key", "coh_gamma", "raw_v", "cert")

    def __init__(self, board: chess.Board, move: chess.Move | None,
                 parent: "_Node | None" = None):
        self.board = board
        self.move = move
        self.parent = parent             # for path-aware threefold detection
        self.rep_key = board._transposition_key()   # repetition key (counter-free)
        self.children: list["_Node"] = []
        self.P = 0.0                     # prior (set by parent expansion)
        self.N = 0
        self.W = 0.0                     # sum of backed-up White-POV values
        self.v_init: float | None = None # squashed reach from parent's batch eval
        self.terminal_v: float | None = None
        self.raw_v: float | None = None  # UNsquashed reach (tree-reuse recalibration
                                         # must use raw units -- MATH_AUDIT A2)
        self.cert: tuple | None = None   # (white_pov_value, confidence) recognizer
                                         # output, cached so the coherence pass
                                         # doesn't re-run the network (MATH_AUDIT)
        # COHERENCE LENGTH (2026-07-16): per-node state-dependent backup discount
        # gamma=exp(-k*divergence), divergence = entropy of the child-value
        # distribution. 1.0 = fully forced (value flows up intact); <1 =
        # divergent (the best-case field value is trusted less the farther it
        # backs up through branchy territory). 1.0 until set in _expand.
        self.coh_gamma = 1.0

    @property
    def Q(self) -> float:
        if self.N > 0:
            return self.W / self.N
        return self.v_init if self.v_init is not None else 0.0


class MCTS:
    """Core tree search over a `reach_fn(boards) -> np.ndarray` oracle.
    Pure python-chess + numpy: unit-testable with a synthetic reach_fn."""

    def __init__(self, reach_fn, max_nodes: int, c_puct: float = 1.5,
                 prior_tau: float = 0.5, cache: dict | None = None,
                 rollout_on_flat: bool = False, flat_std: float = 0.05,
                 rollout_cap: int = 32, detect_threefold: bool = True,
                 coherence_k: float = 0.0,
                 certainty_fn=None, certainty_stop: float = 0.0,
                 cache_key_fn=None):
        assert max_nodes >= 1
        self.reach_fn = reach_fn
        self.detect_threefold = detect_threefold
        self.max_nodes = max_nodes
        self.c_puct = c_puct
        self.prior_tau = prior_tau
        # COHERENCE-LENGTH backup discount strength. 0 disables (exact prior
        # behavior). >0: a node's optimistic best-case field value is discounted
        # toward neutral as it backs up through DIVERGENT (high child-value
        # entropy) nodes, so the field is trusted deep only along FORCED lines.
        self.coherence_k = coherence_k
        # OBVIOUS-REGION soft-terminal: certainty_fn(boards) -> (white_pov_value,
        # confidence) from the committor/recognizer. A node whose confidence >=
        # certainty_stop is treated as RESOLVED (a low-complexity, high-P region
        # like "rook up") -- its value backs up and the search does NOT expand
        # below it, so it stops at obvious regions instead of searching to mate
        # (Kaveh 2026-07-17: phead-as-recognizer, the leaf-termination role).
        self.certainty_fn = certainty_fn
        self.certainty_stop = certainty_stop
        self.evals_used = 0              # budget = FRESH network evals only
        self.rep_history: dict = {}      # position-key counts of the game so far
        # exact eval cache (fen -> raw reach). Reach is a pure function of
        # position for a fixed field+goal, so cache hits are free budget --
        # measured 2026-07-15: 20/32/34% of a game's evals at 200/800/1600n
        # were repeats (transpositions + per-move tree rebuild). Pass a dict
        # that OUTLIVES the search to share across moves/games. NOTE: once a
        # fast MemoryField re-prices reach mid-game, key must include the
        # field version -- pure-slow-field readouts only, for now.
        self.cache = cache
        # cache key MUST include everything reach depends on: the augmented-state
        # rep count is injected by the policy's reach_fn, so a bare FEN key served
        # stale values across rep-counts (MATH_AUDIT: cache vs augmentation).
        self.cache_key_fn = cache_key_fn or (lambda b: b.fen())
        self.cache_hits = 0
        self.rollout_on_flat = rollout_on_flat   # classic-MCTS fallback: when
        self.flat_std = flat_std                 # the field is FLAT here (no
        self.rollout_cap = rollout_cap           # gradient) OR low-confidence
        self.rollouts_run = 0                    # (Kaveh), touch reality with a
        self.low_conf_fn = None                  # free random playout
        self._center = 0.0
        self._scale = 1.0

    # -- value calibration -------------------------------------------------
    def _squash(self, reach: np.ndarray) -> np.ndarray:
        return np.tanh((reach - self._center) / self._scale)

    def _calibrate(self, reach: np.ndarray) -> None:
        self._center = float(np.median(reach))
        self._scale = float(2.0 * reach.std() + 1e-3)

    # -- expansion ---------------------------------------------------------
    def _threefold(self, c: _Node) -> bool:
        """Total occurrences of c's position from the GAME root down the search
        path >= 3 (claimable/forced draw). rep_history counts the actual game
        so far (incl. the search root); ancestors strictly between root and c
        add the search's own repetitions; +1 for c itself."""
        total = self.rep_history.get(c.rep_key, 0) + 1
        anc = c.parent
        while anc is not None and anc.parent is not None:   # exclude root (in rep_history)
            if anc.rep_key == c.rep_key:
                total += 1
            anc = anc.parent
        return total >= 3

    def _expand(self, node: _Node, at_root: bool) -> float:
        """Create children, batch-eval their reach, set priors. Returns the
        White-POV value to back up for this simulation."""
        children = []
        # depth of `node` below the search root (root = 0): the per-ply mate
        # discount needs it -- a mate child at tree depth k+1 backs up
        # MATE_V - (k+1)*PLY_DISCOUNT so FASTER mates strictly dominate.
        # (MATH_AUDIT A1: the discount was previously a constant, making the
        # search indifferent among mate depths -- a conversion-shuffling cause.)
        depth = 0
        anc = node.parent
        while anc is not None:
            depth += 1
            anc = anc.parent
        for m in node.board.legal_moves:
            b2 = node.board.copy(stack=False)
            b2.push(m)
            c = _Node(b2, m, parent=node)
            if b2.is_checkmate():
                # the MOVER of m delivered mate; White-POV sign from who moved
                mate = MATE_V - (depth + 1) * PLY_DISCOUNT if node.board.turn == chess.WHITE \
                    else MATED_V + (depth + 1) * PLY_DISCOUNT
                c.terminal_v = mate
            elif b2.is_insufficient_material() or (b2.halfmove_clock >= 100):
                c.terminal_v = DRAW_V                # rules-exact, history-free draws
            elif self.detect_threefold and self._threefold(c):
                # path-aware threefold: the search's OWN lines can now see a
                # repetition forming (copy(stack=False) drops history, so
                # is_game_over could not -- this was the measured cause of the
                # toy shuffling into a draw the search never saw, 2026-07-16)
                c.terminal_v = DRAW_V
            children.append(c)
        if not children:                                  # stale/checkmated node
            node.terminal_v = DRAW_V if not node.board.is_checkmate() else (
                MATED_V if node.board.turn == chess.WHITE else MATE_V)
            return node.terminal_v

        # obvious-region soft-terminal: a confidently-resolved child (recognizer
        # certainty >= certainty_stop) is treated as terminal with its committor-
        # implied value -- the search stops there instead of recursing to mate.
        if self.certainty_fn is not None and self.certainty_stop > 0.0:
            cand = [c for c in children if c.terminal_v is None and c.cert is None]
            if cand:
                cvals, cconf = self.certainty_fn([c.board for c in cand])
                self.evals_used += len(cand)      # recognizer passes are real evals
                for c, v, cf in zip(cand, cvals, cconf):
                    c.cert = (float(v), float(cf))
                    if cf >= self.certainty_stop:
                        c.terminal_v = float(v)

        fresh = [c for c in children if c.terminal_v is None]
        if fresh:
            if self.cache is None:
                reach = np.asarray(self.reach_fn([c.board for c in fresh]), dtype=float)
                self.evals_used += len(fresh)
            else:
                keys = [self.cache_key_fn(c.board) for c in fresh]
                need = [i for i, k in enumerate(keys) if k not in self.cache]
                self.cache_hits += len(keys) - len(need)
                if need:
                    r = np.asarray(self.reach_fn([fresh[i].board for i in need]), dtype=float)
                    self.evals_used += len(need)
                    for i, v in zip(need, r):
                        self.cache[keys[i]] = float(v)
                reach = np.array([self.cache[k] for k in keys])
                if len(self.cache) > 2_000_000:      # crude memory bound
                    self.cache.clear()
            if at_root:
                self._calibrate(reach)
            sq = self._squash(reach)
            for c, v, rv in zip(fresh, sq, reach):
                c.v_init = float(v)
                c.raw_v = float(rv)
        vals = np.array([c.terminal_v if c.terminal_v is not None else c.v_init
                         for c in children])
        # priors: softmax over child values from the MOVER's perspective
        persp = vals if node.board.turn == chess.WHITE else -vals
        e = np.exp((persp - persp.max()) / self.prior_tau)
        pri = e / e.sum()
        for c, p in zip(children, pri):
            c.P = float(p)
        node.children = children
        # COHERENCE = P(we realize the outcome from here) (Kaveh 2026-07-17). The
        # backup trust factor gamma = exp(-k*(1 - P)): P~1 (a forced/won region,
        # committor confident) => gamma~1, value passes up INTACT -- a proven
        # mate with many legal moves is NOT discounted; P uncertain => discount.
        # Grounded in PROBABILITY-of-mate (the committor confidence), NOT move
        # count. Falls back to child-value entropy (a complexity proxy) only when
        # no committor is available. Proven-mate LINES have committor P~1 all the
        # way up, so their value reaches the root undiscounted.
        if self.coherence_k > 0.0:
            if self.certainty_fn is not None:
                if node.cert is None:
                    _, conf = self.certainty_fn([node.board])
                    self.evals_used += 1
                    node.cert = (0.0, float(conf[0]))
                node.coh_gamma = math.exp(-self.coherence_k * (1.0 - node.cert[1]))
            elif len(children) > 1:
                pp = pri[pri > 0.0]
                H = float(-(pp * np.log(pp)).sum())
                div = H / math.log(len(children))          # normalized entropy
                node.coh_gamma = math.exp(-self.coherence_k * div)
        boot = float(vals[int(np.argmax(persp))])
        if (self.rollout_on_flat and fresh
                and (float(np.std([c.v_init for c in fresh])) < self.flat_std
                     or (self.low_conf_fn is not None and self.low_conf_fn(node.board)))):
            # field flat here: one uniform-random playout to a terminal
            # (0 NN evals -- CPU only) restores game-truth to the backup
            self.rollouts_run += 1
            rb = fresh[int(np.argmax([c.v_init for c in fresh]))].board.copy(stack=False)
            rv = None
            for _ in range(self.rollout_cap):
                if rb.is_game_over(claim_draw=True):
                    out = rb.outcome(claim_draw=True)
                    rv = (MATE_V if out and out.winner == chess.WHITE
                          else MATED_V if out and out.winner == chess.BLACK else DRAW_V)
                    break
                ms = list(rb.legal_moves)
                rb.push(ms[np.random.default_rng(rb.ply()).integers(len(ms))])
            if rv is not None:
                return 0.5 * boot + 0.5 * rv
        return boot

    # -- selection ---------------------------------------------------------
    def _select_child(self, node: _Node) -> _Node:
        white = node.board.turn == chess.WHITE
        sqrt_n = math.sqrt(node.N)
        best, best_s = None, -np.inf
        for c in node.children:
            q = c.terminal_v if c.terminal_v is not None else c.Q
            s = (q if white else -q) + self.c_puct * c.P * sqrt_n / (1 + c.N)
            if s > best_s:
                best_s, best = s, c
        return best

    # -- main loop ---------------------------------------------------------
    def run(self, board: chess.Board, reuse_root: "_Node | None" = None) -> _Node:
        """Search until the eval budget is spent; return the root node.
        reuse_root: a subtree from a previous search whose board matches --
        its visit statistics carry over (tree reuse across moves)."""
        self.evals_used = 0
        # seed repetition history from the actual game so far (the board carries
        # its move stack), so the search can detect threefolds that COMPLETE
        # using positions already played before the search root
        self.rep_history = {}
        if board.move_stack:
            b = board.copy(stack=True)
            keys = [b._transposition_key()]
            while b.move_stack:
                b.pop()
                keys.append(b._transposition_key())
            for k in keys:
                self.rep_history[k] = self.rep_history.get(k, 0) + 1
        else:
            self.rep_history[board._transposition_key()] = 1
        if reuse_root is not None and reuse_root.board.fen() == board.fen():
            root = reuse_root
            if not root.children:
                root.N = max(root.N, 1)
                root.W += self._expand(root, at_root=True)
            else:
                # recalibrate on RAW reach (v_init is already-squashed output of
                # the PREVIOUS move's calibration; feeding it back mixed units --
                # MATH_AUDIT A2). Nodes lacking raw_v (old ckpts/terminals) skip.
                with_v = [c.raw_v for c in root.children if c.raw_v is not None]
                if with_v:
                    self._calibrate(np.array(with_v))
                    for c in root.children:
                        if c.raw_v is not None:
                            c.v_init = float(self._squash(np.array([c.raw_v]))[0])
        else:
            root = _Node(board.copy(stack=False), None)
            root.N = 1
            root.W = self._expand(root, at_root=True)
        # sims bound: budget is counted in NETWORK EVALS, and a simulation
        # that ends on a terminal consumes none -- when every reachable leaf
        # is terminal the eval budget alone would never be spent and the
        # loop would spin forever (2026-07-14: hung a 700-start generation
        # run 20 starts in). Terminal-only backups are also useless past a
        # point; cap total simulations at a generous multiple of the budget.
        sims, max_sims = 0, 32 * self.max_nodes
        while (self.evals_used < self.max_nodes and root.children
               and sims < max_sims):
            sims += 1
            node, path = root, [root]
            while node.children:
                node = self._select_child(node)
                path.append(node)
                if node.terminal_v is not None:
                    break
            if node.terminal_v is not None:
                v = node.terminal_v
            else:
                v = self._expand(node, at_root=False)
            # backup: leaf value flows to every ancestor. With coherence_k>0 it
            # is discounted toward neutral (0) by the compounding product of
            # coh_gamma over the nodes BELOW each ancestor -- a value earned
            # deep down a divergent line reaches the root attenuated, one earned
            # down a forced line reaches it intact (coh_gamma=1 => exact old
            # backup). Leaf-first so the product accumulates on the way up.
            v_run = v
            for n in reversed(path):
                n.N += 1
                n.W += v_run
                v_run = v_run * n.coh_gamma
        return root

    def best_move(self, board: chess.Board) -> chess.Move:
        root = self.run(board)
        if not root.children:
            raise ValueError("no legal moves")
        white = board.turn == chess.WHITE
        for c in root.children:                          # immediate mate: take it
            if c.terminal_v is not None and (c.terminal_v > 0.5 if white else c.terminal_v < -0.5):
                return c.move
        best, key = None, None
        for c in root.children:
            q = c.terminal_v if c.terminal_v is not None else c.Q
            k = (c.N, q if white else -q)
            if key is None or k > key:
                key, best = k, c
        return best.move


class FBMCTSPolicy:
    """playout_ab-compatible policy: MCTS readout of a TorchFB checkpoint.
    `z` is a single goal embedding (d,) or an exemplar bank (m, d) scored
    with the play-tested soft-min region readout (see policy_fb)."""

    def __init__(self, fb, z, max_nodes: int, c_puct: float = 1.5,
                 prior_tau: float = 0.5, elo: int = 1800, clock: float = 300.0,
                 device: str = "cpu", cache: bool = True, s_head=None,
                 g_sharp: float = 0.0, evidence: dict | None = None,
                 evidence_k: float = 4.0, rollout_on_flat: bool = False,
                 tree_reuse: bool = False, committor_head=None,
                 committor_dhead=None, clearance_beta: float = 0.0,
                 detect_threefold: bool = True, coherence_k: float = 0.0,
                 certainty_head=None, certainty_stop: float = 0.0):
        import torch
        from catspace.data.encode import encode_meta, encode_packed
        from catspace.nn.features import feature_planes, omega_ids
        from catspace.nn.policy_fb import soft_min_bank
        omega_row = omega_ids(np.array([elo]), np.array([elo]), np.array([clock]))[0]
        self.fb = fb.to(device).eval()
        self.z = torch.as_tensor(z, dtype=torch.float32, device=device)
        assert self.z.dim() in (1, 2)
        omega_row = omega_ids(np.array([elo]), np.array([elo]), np.array([clock]))[0]

        @torch.no_grad()
        def reach(boards):
            packed = np.stack([encode_packed(b) for b in boards])
            # augmented-state coordinate: repetition count from the GAME path
            # (search children are off-path -> 0; the game's own revisits mark
            # proximity to the threefold surface)
            meta = np.stack([encode_meta(b, rep=self.path_counts.get(b.board_fen(), 0))
                             for b in boards])
            planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
            om = torch.from_numpy(np.tile(omega_row, (len(boards), 1))).to(device)
            f = self.fb.embed_F(planes, om)
            if committor_head is not None:
                # committor readout (Kaveh 2026-07-15): the goal is a SURFACE,
                # not a pole -- reach = -d_W(s) = ln P(hit the mate boundary
                # first), a learned hitting probability with no goal vector.
                r = -committor_head(f).squeeze(-1)
                if committor_dhead is not None and clearance_beta != 0.0:
                    # draw-surface CLEARANCE (Kaveh 2026-07-16): where the win
                    # field is flat (the rim), distance from the draw basin
                    # breaks the tie -- progress reads better than shuffling
                    r = r + clearance_beta * committor_dhead(f).squeeze(-1)
                return r.cpu().numpy()
            if self.z.dim() == 2:
                r = soft_min_bank(self.fb, f, self.z, 0.1)
            else:
                r = self.fb.score(f, self.z)
            if s_head is not None and g_sharp != 0.0:
                # two-channel readout (2026-07-15): risk enters HERE, not in
                # the geometry -- reach discounted by the state's sharpness
                # times the fallibility weight (omega-dependent later)
                r = r - g_sharp * s_head(f).squeeze(-1)
            return r.cpu().numpy()

        # obvious-region recognizer: the phead's 3-way W/D/L softmax gives a
        # per-position resolved value (expected White-POV outcome) and a
        # confidence (peak class prob). certainty_stop turns high-confidence
        # regions into search-terminals (see MCTS.certainty_fn).
        # build whenever the head is present (not only for the soft-terminal):
        # coherence_k also consumes this confidence as its P(realize) signal.
        certainty = None
        if certainty_head is not None:
            @torch.no_grad()
            def certainty(boards):
                packed = np.stack([encode_packed(b) for b in boards])
                meta = np.stack([encode_meta(b, rep=self.path_counts.get(b.board_fen(), 0))
                                 for b in boards])
                planes = torch.from_numpy(feature_planes(packed, meta)).to(device)
                om = torch.from_numpy(np.tile(omega_row, (len(boards), 1))).to(device)
                f = self.fb.embed_F(planes, om)
                p = torch.softmax(certainty_head(f), dim=1)          # (n, 3) = W,D,L
                val = p[:, 0] * MATE_V + p[:, 1] * DRAW_V + p[:, 2] * MATED_V
                conf = p.max(dim=1).values
                return val.cpu().numpy(), conf.cpu().numpy()

        self.mcts = MCTS(reach, max_nodes=max_nodes, c_puct=c_puct,
                         prior_tau=prior_tau, cache={} if cache else None,
                         rollout_on_flat=rollout_on_flat,
                         detect_threefold=detect_threefold,
                         coherence_k=coherence_k,
                         certainty_fn=certainty, certainty_stop=certainty_stop,
                         cache_key_fn=lambda b: f"{b.fen()}|{self.path_counts.get(b.board_fen(), 0)}")
        self.evidence = evidence or {}
        self.evidence_k = evidence_k
        self.path_counts: dict = {}
        self.tree_reuse = tree_reuse
        self._carry: "object | None" = None

        if evidence is not None:
            base_reach = self.mcts.reach_fn

            @torch.no_grad()
            def blended(boards):
                r = np.asarray(base_reach(boards), dtype=float)
                # precision-weighted evidence blend: d_eff = (n*d_ev + k*d_field)
                # / (n + k); reach shifts by (d_field - d_eff). Live game-path
                # revisits are stall evidence (revisit = objectively no progress
                # in a must-progress conversion): d_ev -> horizon.
                packed = np.stack([encode_packed(b) for b in boards])
                meta = np.stack([encode_meta(b) for b in boards])
                pl = torch.from_numpy(feature_planes(packed, meta)).to(device)
                om = torch.from_numpy(np.tile(omega_row, (len(boards), 1))).to(device)
                f = self.fb.embed_F(pl, om)
                d_field = self.fb.distance_matrix(f, self.z[None, :])[:, 0].cpu().numpy()                     if self.z.dim() == 1 else None
                if d_field is None:
                    return r
                for i, b in enumerate(boards):
                    fen = b.fen()
                    n_ev, d_ev = self.evidence.get(fen, (0.0, 0.0))
                    rep = self.path_counts.get(b.board_fen(), 0)
                    if rep >= 1:
                        n_rep = 8.0 * rep
                        d_ev = (n_ev * d_ev + n_rep * 2.0) / (n_ev + n_rep)
                        n_ev = n_ev + n_rep
                    if n_ev > 0:
                        d_eff = (n_ev * d_ev + self.evidence_k * d_field[i]) / (n_ev + self.evidence_k)
                        r[i] += d_field[i] - d_eff
                return r
            self.mcts.reach_fn = blended
            # low-confidence proxy until a competence head ships: the field is
            # unvouched where no evidence exists near this state
            self.mcts.low_conf_fn = lambda b: self.evidence.get(b.fen(), (0.0, 0.0))[0] == 0

    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        self.path_counts[board.board_fen()] = self.path_counts.get(board.board_fen(), 0) + 1
        if not self.tree_reuse:
            return self.mcts.best_move(board)
        carry = None
        if self._carry is not None:
            for c in getattr(self._carry, "children", []):
                if c.board.fen() == board.fen():
                    carry = c
                    break
        root = self.mcts.run(board, reuse_root=carry)
        if not root.children:
            raise ValueError("no legal moves")
        white = board.turn == chess.WHITE
        best = None
        for c in root.children:
            if c.terminal_v is not None and (c.terminal_v > 0.5 if white else c.terminal_v < -0.5):
                best = c
                break
        if best is None:
            best = max(root.children,
                       key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None else c.Q)
                                      * (1 if white else -1)))
        self._carry = best
        return best.move
