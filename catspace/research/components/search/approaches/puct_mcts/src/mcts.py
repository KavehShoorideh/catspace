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


_PVAL = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5,
         chess.QUEEN: 9, chess.KING: 0}


def is_tactical_move(parent: chess.Board, move: chess.Move, child: chess.Board) -> bool:
    """RULE-DERIVED (no learned heuristic): the move gives check, captures,
    promotes, or the moved piece now attacks a strictly-higher-value or
    undefended enemy piece (a material threat). `child` is the board AFTER the
    move (expansion already built it). Used by the tactical prior (Kaveh
    2026-07-19: "MCTS should spend nodes on checks, captures, threats") --
    board truth injected into move ORDERING only, never into values."""
    if child.is_check() or move.promotion is not None or parent.is_capture(move):
        return True
    to = move.to_square
    mv = child.piece_at(to)
    if mv is None:                       # cannot happen for a legal move; guard
        return False
    mv_val = _PVAL[mv.piece_type]
    opp = child.turn                     # side to move after our move = the attacked side
    for sq in child.attacks(to):
        p = child.piece_at(sq)
        if p is not None and p.color == opp and p.piece_type != chess.KING:
            if _PVAL[p.piece_type] > mv_val or not child.is_attacked_by(opp, sq):
                return True              # attacks bigger piece, or an undefended one
    return False


def game_truth(node: "_Node") -> bool:
    """terminal_v present AND not planted by the certainty recognizer.
    Network confidence is never game truth (the 0.60->0.20 rule). Provenance
    is an explicit flag set at the planting site -- the earlier float-equality
    check (terminal_v == cert[0]) could misclassify a genuine rules terminal
    that also carried a coincidentally-equal cert (e.g. a stalemate child,
    cert-scanned before expansion, whose cert value was exactly DRAW_V):
    sound-but-loose, review 2026-07-18."""
    return node.terminal_v is not None and not node.cert_planted


class _Node:
    __slots__ = ("board", "move", "children", "P", "N", "W", "W2", "v_init",
                 "terminal_v", "parent", "rep_key", "coh_gamma", "raw_v", "cert",
                 "cert_planted", "pw_order", "tactical")

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
        self.W2 = 0.0                    # sum of squares (for the value CI)
        self.v_init: float | None = None # squashed reach from parent's batch eval
        self.terminal_v: float | None = None
        self.raw_v: float | None = None  # UNsquashed reach (tree-reuse recalibration
                                         # must use raw units -- MATH_AUDIT A2)
        self.cert: tuple | None = None   # (white_pov_value, confidence) recognizer
                                         # output, cached so the coherence pass
                                         # doesn't re-run the network (MATH_AUDIT)
        self.cert_planted = False        # terminal_v came from the recognizer,
                                         # NOT the rules -- never game truth
        self.pw_order: list | None = None  # children indices, mover-best first
                                         # (progressive-widening candidate order)
        self.tactical = False            # check/capture/promotion/material threat
                                         # (rule-derived; set by parent expansion)
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

    def value_ci(self, z: float = 1.96) -> tuple:
        """(mean, half-width) of the backed-up value (White-POV). Normal-approx
        CI on a bounded-[-1,1] mean; maximally uncertain (hw=1) below 2 samples.
        Terminal nodes carry no uncertainty -- callers use terminal_v (hw=0)."""
        if self.N < 2:
            return (self.Q, 1.0)
        mean = self.W / self.N
        var = max(0.0, self.W2 / self.N - mean * mean)
        return (mean, min(1.0, z * math.sqrt(var / self.N)))


class MCTS:
    """Core tree search over a `reach_fn(boards) -> np.ndarray` oracle.
    Pure python-chess + numpy: unit-testable with a synthetic reach_fn."""

    def __init__(self, reach_fn, max_nodes: int, c_puct: float = 1.5,
                 prior_tau: float = 0.5, cache: dict | None = None,
                 rollout_on_flat: bool = False, flat_std: float = 0.05,
                 rollout_cap: int = 32, detect_threefold: bool = True,
                 coherence_k: float = 0.0,
                 certainty_fn=None, certainty_stop: float = 0.0,
                 cache_key_fn=None, decision_stop: bool = False,
                 decision_check_every: int = 16, mate_stop: bool = False,
                 pw_c: float = 0.0, pw_alpha: float = 0.5, pw_min: int = 4,
                 tactical_prior: float = 0.0,
                 root_min_visits: int = 0, ci_z: float = 1.96,
                 policy_fn=None, value_fn=None, fpu_reduction: float = 0.25,
                 eval_cache: dict | None = None, batch_leaves: int = 1,
                 policy_batch_fn=None, order_fn=None, opp_policy_fn=None):
        assert max_nodes >= 1
        # EARLY-STOP LEVERS (2026-07-18, the planner energy objective
        # E[score] - c*compute): spend budget only while it can still change
        # the MOVE. Two independent flags (split after the v2 A/B):
        # - mate_stop: a game-truth immediate mate at the root ends the
        #   search after root expansion. CERTIFIED: with the game_truth gate
        #   in best_move, the readout is provably identical to the full-
        #   budget search (the mate short-circuit precedes visit-argmax).
        # - decision_stop: stop when the root visit gap exceeds the
        #   remaining budget in SIM units (remaining evals / measured
        #   evals-per-sim, x2 safety, capped by max_sims). HEURISTIC, not a
        #   theorem (terminal sims add visits at zero evals) -- and MEASURED
        #   HARMFUL at 800n: -0.040 conversion (e=4.38) for 6% energy;
        #   sims are too few (~36/move) for gap dominance. Kept for
        #   larger-sim regimes / tree-reuse retests only.
        self.decision_stop = decision_stop
        self.decision_check_every = decision_check_every
        self.mate_stop = mate_stop
        # PROGRESSIVE WIDENING (Kaveh 2026-07-19, interim until the plan-
        # alignment move prior orders moves): selection considers only the
        # top-K(N) children by mover-perspective value, K = max(pw_min,
        # ceil(pw_c * N^pw_alpha)) -- the canonical schedule (Coulom 2007 /
        # progressive unpruning). Value-only expansion pays branching-many
        # evals per node, so an unrestricted PUCT spreads the budget across
        # every legal reply before deepening ("7 visits at the root at 4000
        # nodes"); widening concentrates descent so depth scales with budget.
        # All children are still CREATED (rule-based mate/draw detection stays
        # exact -- a mate child has maximal mover value, so it is always inside
        # the window) and batch-evaluated once; only DESCENT is restricted.
        # pw_c=0 disables (exact prior behavior; training/eval default).
        self.pw_c = pw_c
        self.pw_alpha = pw_alpha
        self.pw_min = pw_min
        # TACTICAL PRIOR (Kaveh 2026-07-19: "spend nodes on checks, captures,
        # threats"): the FIELD is tactically blind (outcome-agnostic training),
        # so value-ordered priors can starve exactly the moves whose
        # consequences it misprices. Rule-derived tactical children (see
        # is_tactical_move) get (a) guaranteed membership in the progressive-
        # widening window and (b) a prior blend P = (1-w)*P_field + w*uniform
        # (tactical) -- progressive-bias style: ordering only, values untouched,
        # so a refuted tactic still loses on Q. 0 disables (exact prior
        # behavior). Interim until the plan-alignment prior; re-test on field
        # promotions (conditional-rejections rule).
        self.tactical_prior = tactical_prior
        # CI-DRIVEN ROOT EXPLORATION (Kaveh 2026-07-19: "every move needs a
        # minimum number of tries; keep trying until confidence in the badness
        # of a move > ~95%"). When root_min_visits>0, the ROOT (a) guarantees
        # every non-terminal move >= root_min_visits samples, then (b) allocates
        # by UCB/LCB best-arm identification: sample only moves whose UPPER value
        # bound still reaches the best move's LOWER bound (could-still-be-best);
        # a move whose whole CI sits below is ~(ci_z)-confidently worse and is
        # left alone. Deeper nodes keep PUCT (+ widening + tactical). 0 disables.
        self.root_min_visits = root_min_visits
        self.ci_z = ci_z
        # AZ-STYLE CHEAP EXPANSION (Kaveh 2026-07-19: policy head "only using the
        # field"). When policy_fn is given, expanding a node costs ONE eval: the
        # child priors come from policy_fn(node) and the node value from
        # value_fn(node); children are created UNEVALUATED (their Q comes from
        # backups, with first-play-urgency until visited). So the node budget
        # counts simulations, not branching*sims -- the fix for value-only
        # expansion flattening visits at low node counts. policy_fn(board) ->
        # {move: prior}; value_fn(boards) -> white-POV values in [-1, 1].
        self.policy_fn = policy_fn
        self.value_fn = value_fn
        # AZ-path acceleration (Kaveh 2026-07-23: "we won't get rid of MCTS"):
        # eval_cache: {transposition_key: (value, {uci: prior})} SHARED across MCTS
        # instances/moves/games (caller owns it; net evals on transpositions become free).
        # batch_leaves > 1: collect K leaves under virtual visits, evaluate values in ONE
        # value_fn batch (the dominant cost -- e.g. the 9.1M-param field at batch-1 CPU).
        self.eval_cache = eval_cache
        self.batch_leaves = max(1, int(batch_leaves))
        self.policy_batch_fn = policy_batch_fn
        # TIERED DESCENT ORDER (Kaveh 2026-07-29): order_fn(boards, mover_white) ->
        # (B, K) MOVER-POV tier scores, tier 0 first (e.g. reach-to-subgoal, then
        # -distance-to-my-mated-basin, then eval). Used ONLY for the progressive-
        # widening descent order: lexicographic, each non-final tier quantized at
        # its own per-expansion batch std (data-derived tolerance -- no hand
        # constants), final tier raw. Priors and backups untouched (progressive-
        # bias style, like tactical_prior). Costs len(children) evals per
        # expansion (the eval tier is a real net pass). None disables (exact
        # prior behavior). Requires pw_c > 0 to have any effect.
        self.order_fn = order_fn
        # OPPONENT-MODEL EXPANSION (MILESTONES §M5: "expansion weighted by the
        # OPPONENT MODEL -- expectimax over Maia/z policy, measured better than
        # minimax 0.125 vs 0.094"). opp_policy_fn(board) -> {move: prior} | None,
        # called at nodes where the OPPONENT is to move (turn != root turn):
        # child priors P come from the opponent's actual move distribution
        # instead of the adversarial value-softmax; Q/backups untouched, so
        # PUCT at opponent nodes becomes soft-expectimax (explore what they
        # PLAY, value what it does to us). None return = keep default priors
        # (poisoned/failed model call). Costs 1 eval per opponent expansion.
        self.opp_policy_fn = opp_policy_fn
        self._root_turn = None
        self.fpu_reduction = fpu_reduction
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
            if self.tactical_prior > 0.0:
                c.tactical = is_tactical_move(node.board, m, b2)
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

        if self.policy_fn is not None:
            # AZ-style: ONE policy eval -> all child priors; ONE value eval ->
            # the node's backup value. Children stay UNEVALUATED (FPU Q until
            # visited). ~1 eval/expansion instead of len(children).
            tkey = node.board._transposition_key() if self.eval_cache is not None else None
            if tkey is not None and tkey in self.eval_cache:      # transposition: FREE
                v, pri_uci = self.eval_cache[tkey]
                for c in children:
                    c.P = float(pri_uci.get(c.move.uci(), 1e-6))
                node.children = children
                return float(v)
            pri = self.policy_fn(node.board)
            for c in children:
                c.P = float(pri.get(c.move, 1e-6))
            node.children = children
            self.evals_used += 1
            v = float(self.value_fn([node.board])[0])
            if tkey is not None:
                self.eval_cache[tkey] = (v, {m.uci(): p for m, p in pri.items()})
            return v

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
                        c.cert_planted = True

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
        if self.tactical_prior > 0.0:
            tac = np.array([c.tactical for c in children], dtype=bool)
            if tac.any():
                w = self.tactical_prior
                pri = (1.0 - w) * pri + w * (tac / tac.sum())
        if (self.opp_policy_fn is not None and self._root_turn is not None
                and node.board.turn != self._root_turn):
            # OPPONENT node: priors = their actual move distribution (Maia/z),
            # replacing the adversarial value-softmax. Q/backups untouched.
            mp = self.opp_policy_fn(node.board)
            if mp:
                self.evals_used += 1
                om = np.array([mp.get(c.move, 0.0) for c in children], dtype=float)
                if om.sum() > 0.0:
                    pri = om / om.sum()
        for c, p in zip(children, pri):
            c.P = float(p)
        node.children = children
        if self.pw_c > 0.0:
            # candidate order for progressive widening: mover-best first. A
            # mate-for-the-mover child has maximal persp, so it is index 0.
            # Value recalibration (tree reuse) is monotone, so this order
            # stays valid without recomputation.
            if self.order_fn is not None:
                T = np.asarray(self.order_fn([c.board for c in children],
                                             node.board.turn == chess.WHITE), dtype=float)
                self.evals_used += len(children)     # the eval tier is a real net pass
                keys = []
                for k in range(T.shape[1] - 1):
                    sd = float(T[:, k].std())
                    keys.append(np.round(T[:, k] / (sd + 1e-9)))
                keys.append(T[:, -1])
                # np.lexsort: LAST key is primary; negate all for best-first
                order = [int(i) for i in np.lexsort(tuple(-k for k in reversed(keys)))]
                # rule-exact mover-winning terminals must stay in the window
                # regardless of their reach tier (a mate child can look reach-poor)
                tw = [i for i in order
                      if children[i].terminal_v is not None and persp[i] > 0.5]
                node.pw_order = tw + [i for i in order if i not in set(tw)]
            else:
                node.pw_order = [int(i) for i in np.argsort(-persp)]
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
        if self.root_min_visits > 0 and node.parent is None:
            live = [c for c in node.children if c.terminal_v is None]
            if live:
                # (a) floor: round-robin every non-terminal move up to the min --
                # but CAP the floor so it can't consume the whole budget at low
                # node counts (value-only expansion is ~nm evals/sim, so only
                # ~max_nodes/nm sims exist; flooring nm moves to f costs f*nm
                # sims). Keep the floor to ~70% of the budget so the UCB phase
                # still gets to concentrate -- else every move ends up ~equal.
                nm = len(live)
                eff = min(self.root_min_visits, max(1, int(0.7 * self.max_nodes / (nm * nm))))
                under = [c for c in live if c.N < eff]
                if under:
                    return min(under, key=lambda c: c.N)
            # (b) UCB/LCB best-arm: keep sampling the most optimistic move whose
            # upper bound still reaches the best lower bound; the rest are ~ci_z
            # confidently worse and get no more budget.
            bounds, best_lo = [], -np.inf
            for c in node.children:
                if c.terminal_v is not None:
                    persp = c.terminal_v if white else -c.terminal_v
                    lo = hi = persp
                else:
                    m, hw = c.value_ci(self.ci_z)
                    persp = m if white else -m
                    lo, hi = persp - hw, persp + hw
                bounds.append((c, lo, hi))
                best_lo = max(best_lo, lo)
            return max((c_hi for c_hi in ((c, hi) for c, lo, hi in bounds if hi >= best_lo)),
                       key=lambda t: t[1])[0]
        sqrt_n = math.sqrt(node.N)
        cand = node.children
        if self.pw_c > 0.0 and node.pw_order is not None:
            # progressive widening: descend only into the top-K(N) children --
            # plus every TACTICAL child (checks/captures/threats are never
            # excluded from the window; the field misprices exactly those)
            k = max(self.pw_min, math.ceil(self.pw_c * node.N ** self.pw_alpha))
            if k < len(cand):
                cand = [node.children[i] for i in node.pw_order[:k]]
                if self.tactical_prior > 0.0:
                    inside = set(node.pw_order[:k])
                    cand += [c for i, c in enumerate(node.children)
                             if c.tactical and i not in inside]
        # first-play-urgency: in AZ mode unvisited children have no v_init, so
        # estimate their (mover-frame) Q as the parent's value minus a reduction
        # -- discourages fanning out to every unvisited move (lc0 FPU).
        az = self.policy_fn is not None
        fpu = (node.Q if white else -node.Q) - self.fpu_reduction
        best, best_s = None, -np.inf
        for c in cand:
            if c.terminal_v is not None:
                qm = c.terminal_v if white else -c.terminal_v
            elif az and c.N == 0:
                qm = fpu
            else:
                qm = c.Q if white else -c.Q
            s = qm + self.c_puct * c.P * sqrt_n / (1 + c.N)
            if s > best_s:
                best_s, best = s, c
        return best

    # -- main loop ---------------------------------------------------------
    def run(self, board: chess.Board, reuse_root: "_Node | None" = None) -> _Node:
        """Search until the eval budget is spent; return the root node.
        reuse_root: a subtree from a previous search whose board matches --
        its visit statistics carry over (tree reuse across moves)."""
        self.evals_used = 0
        self._root_turn = board.turn         # opponent-model expansion needs "whose node"
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
            # reuse + threefold correctness (2026-07-25): _threefold flags are planted at
            # EXPANSION under the game history of that moment; the history has since grown
            # (it only grows, so stale flags stay valid) -- ADD flags for carried nodes
            # that are repetitions under the CURRENT history, else reuse blinds the search
            # to draws forming across moves (measured: threefold FAILs at mature banks).
            if self.detect_threefold:
                stack = list(root.children)
                while stack:
                    n = stack.pop()
                    if n.terminal_v is None and self._threefold(n):
                        n.terminal_v = DRAW_V
                    stack.extend(n.children)
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
        white = board.turn == chess.WHITE
        if self.mate_stop:
            for c in root.children:
                if game_truth(c) and (c.terminal_v > 0.5 if white else c.terminal_v < -0.5):
                    return root          # certified stop: immediate mate in hand
        sims, max_sims = 0, 32 * self.max_nodes
        if self.policy_fn is not None and self.value_fn is not None and self.batch_leaves > 1:
            return self._run_batched(root, max_sims)
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
                n.W2 += v_run * v_run          # for the per-move value CI
                v_run = v_run * n.coh_gamma
            if (self.decision_stop and sims % self.decision_check_every == 0
                    and len(root.children) > 1):
                vis = sorted((c.N for c in root.children), reverse=True)
                # gap must beat the remaining budget in SIM units, not eval
                # units: each sim adds ONE root visit but costs ~one expansion
                # batch of evals. v1 compared gap to remaining EVALS -- visits
                # accrue ~20x slower, so it fired only in the last few percent
                # (measured util 0.96 @800n, 2026-07-18). Terminal-path sims
                # cost 0 evals, so scale by 2x and cap with max_sims to stay
                # conservative (err toward stopping late).
                cost = self.evals_used / max(sims, 1)
                rem = min(max_sims - sims,
                          int(2.0 * (self.max_nodes - self.evals_used) / max(cost, 1e-9)))
                if vis[0] - vis[1] > rem:
                    break                # stability stop (heuristic, see __init__)
        return root

    def _backup(self, path, v):
        v_run = v
        for n in reversed(path):
            n.N += 1
            n.W += v_run
            n.W2 += v_run * v_run
            v_run = v_run * n.coh_gamma

    def _run_batched(self, root, max_sims):
        """AZ path with BATCHED leaf evaluation: collect up to batch_leaves distinct
        leaves under virtual visits (N inflation diversifies selection), evaluate all
        their values in ONE value_fn call (+ batched priors when policy_batch_fn is
        given), then backup and revert the virtual visits. Cache-aware: transposition
        hits skip the queue entirely. Same budget semantics (evals_used = net calls)."""
        sims = 0
        while (self.evals_used < self.max_nodes and root.children and sims < max_sims):
            pending = []                        # (node, path) needing net eval
            touched = []                        # paths holding virtual visits
            queued_ids = set()
            for _ in range(min(self.batch_leaves, self.max_nodes - self.evals_used)):
                if sims >= max_sims:
                    break
                node, path = root, [root]
                while node.children:
                    node = self._select_child(node)
                    path.append(node)
                    if node.terminal_v is not None:
                        break
                if node.terminal_v is not None:
                    sims += 1
                    self._backup(path, node.terminal_v)
                    continue
                if id(node) in queued_ids:      # selection converged on a queued leaf
                    break
                # cache probe before queueing: hits are free and backup immediately
                tkey = node.board._transposition_key() if self.eval_cache is not None else None
                if tkey is not None and tkey in self.eval_cache:
                    sims += 1
                    v = self._expand(node, at_root=False)   # applies cached priors, 0 evals
                    self._backup(path, v)
                    continue
                sims += 1
                queued_ids.add(id(node))
                for n in path:                  # virtual visit: diversify the next selection
                    n.N += 1
                touched.append(path)
                pending.append((node, path, tkey))
            if pending:
                boards = [n.board for n, _p, _k in pending]
                values = self.value_fn(boards)
                self.evals_used += len(boards)
                if self.policy_batch_fn is not None:
                    pris = self.policy_batch_fn(boards)
                else:
                    pris = [self.policy_fn(b) for b in boards]
                for (node, path, tkey), v, pri in zip(pending, values, pris):
                    self._attach_children_az(node, pri)
                    if tkey is not None:
                        self.eval_cache[tkey] = (float(v), {m.uci(): p for m, p in pri.items()})
                for path in touched:            # revert virtual visits (real backup re-adds)
                    for n in path:
                        n.N -= 1
                for (node, path, _k), v in zip(pending, values):
                    self._backup(path, float(v))
            elif not touched:
                # no evals and no terminals progressed? all-terminal tree -- bail via sims cap
                if all(c.terminal_v is not None for c in root.children):
                    break
        return root

    def _attach_children_az(self, node, pri):
        """Create node's children with priors from an already-computed policy dict
        (the batched path's replacement for _expand's AZ branch)."""
        depth, anc = 0, node
        while anc.parent is not None:
            depth += 1
            anc = anc.parent
        children = []
        for m in node.board.legal_moves:
            b2 = node.board.copy(stack=False)
            b2.push(m)
            c = _Node(b2, m, parent=node)
            if self.tactical_prior > 0.0:
                c.tactical = is_tactical_move(node.board, m, b2)
            if b2.is_checkmate():
                mate = MATE_V - (depth + 1) * PLY_DISCOUNT if node.board.turn == chess.WHITE \
                    else MATED_V + (depth + 1) * PLY_DISCOUNT
                c.terminal_v = mate
            elif b2.is_insufficient_material() or (b2.halfmove_clock >= 100):
                c.terminal_v = DRAW_V
            elif self.detect_threefold and self._threefold(c):
                c.terminal_v = DRAW_V
            c.P = float(pri.get(c.move, 1e-6))
            children.append(c)
        if children:
            node.children = children
        else:
            node.terminal_v = DRAW_V if not node.board.is_checkmate() else (
                MATED_V if node.board.turn == chess.WHITE else MATE_V)

    def best_move(self, board: chess.Board) -> chess.Move:
        root = self.run(board)
        if not root.children:
            raise ValueError("no legal moves")
        white = board.turn == chess.WHITE
        for c in root.children:                          # immediate mate: take it
            # game_truth gate REQUIRED (review 2026-07-18 HIGH): a cert-planted
            # terminal_v > 0.5 earlier in move order would otherwise be played
            # INSTEAD of a genuine mate-in-1 -- network confidence choosing the
            # move under a certified label
            if game_truth(c) and (c.terminal_v > 0.5 if white else c.terminal_v < -0.5):
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
                 certainty_head=None, certainty_stop: float = 0.0,
                 decision_stop: bool = False, mate_stop: bool = False,
                 pw_c: float = 0.0, pw_alpha: float = 0.5, pw_min: int = 4,
                 tactical_prior: float = 0.0, root_min_visits: int = 0, ci_z: float = 1.96,
                 policy_fn=None, value_fn=None, fpu_reduction: float = 0.25):
        import torch
        from catspace.research.tools.chess_specific.chessdata.encode import encode_meta, encode_packed
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.features import feature_planes, omega_ids
        from catspace.research.components.encoder.approaches.jepa_tokenizer.src.policy_fb import soft_min_bank
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
                         decision_stop=decision_stop, mate_stop=mate_stop,
                         pw_c=pw_c, pw_alpha=pw_alpha, pw_min=pw_min,
                         tactical_prior=tactical_prior,
                         root_min_visits=root_min_visits, ci_z=ci_z,
                         policy_fn=policy_fn, value_fn=value_fn, fpu_reduction=fpu_reduction,
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
            # game_truth gate: same HIGH as MCTS.best_move (review 2026-07-18)
            if game_truth(c) and (c.terminal_v > 0.5 if white else c.terminal_v < -0.5):
                best = c
                break
        if best is None:
            best = max(root.children,
                       key=lambda c: (c.N, (c.terminal_v if c.terminal_v is not None else c.Q)
                                      * (1 if white else -1)))
        self._carry = best
        return best.move


class FBPlanMCTSPolicy(FBMCTSPolicy):
    """Plan persistence on the WINNING substrate (2026-07-18 Pareto round):
    the beam-based FBPlanPolicy proved the tier-0 shape (p50=30 rows/move —
    a held plan makes most moves nearly free) but died on its substrate
    (0.150 conversion; beam replans cost ~3400 actual rows). This is the
    same persistence idea as a per-move BUDGET schedule on one committor
    MCTS with tree reuse:

      PLAN move (deep, plan_nodes):  a replan trigger fired — search fresh.
      EXEC move (cheap, exec_nodes): the game is following the plan — top up
        the CARRIED deep tree (tree_reuse) with a small budget.

    Triggers (FBPlanPolicy's set, re-grounded on this substrate; a first cut
    with N>=1 tree membership as "expected" replanned nearly every move —
    p50=802 at n=6 — because ~36 sims/move leave the defender's exact reply
    unvisited):
      surprise — the position is not even a NODE of the carried tree (the
                 carried parent was never expanded there; also fires on the
                 first move of a game)
      dropped  — the committor readout fell more than drop_delta below its
                 value when the plan was made (ln P(win) units; refutation
                 shows up in the VALUE, not in tree membership). Costs one
                 reach call per move, cache-absorbed.
      stalled  — plies_since_plan >= max_plies_per_plan (plans go stale)

    The mate-stop composes (proven move-identical); the stability stop stays
    off (measured harmful). Energy gate: hold mcts@plan_nodes strength at a
    fraction of its rows/move; graded by paired A/B + energy_baseline."""

    def __init__(self, fb, z, plan_nodes: int = 800, exec_nodes: int = 100,
                 max_plies_per_plan: int = 6, drop_delta: float = 0.5, **kw):
        kw.setdefault("mate_stop", True)
        super().__init__(fb, z, max_nodes=plan_nodes, tree_reuse=True, **kw)
        self.plan_nodes = plan_nodes
        self.exec_nodes = exec_nodes
        self.max_plies_per_plan = max_plies_per_plan
        self.drop_delta = drop_delta
        self._plies_since_plan = 0
        self._r_at_plan: float | None = None
        self.replans = 0
        self.last_trigger: str | None = None     # introspection/testing

    def _expected(self, board: chess.Board) -> bool:
        """Is this position a node of the carried tree at all? (Membership,
        not visits: children exist whenever their parent was expanded.)"""
        if self._carry is None:
            return False
        for c in getattr(self._carry, "children", []):
            if c.board.fen() == board.fen():
                return True
        return False

    def move(self, board: chess.Board, rng: np.random.Generator) -> chess.Move:
        r_now = float(np.asarray(self.mcts.reach_fn([board]))[0])
        trigger = None
        if not self._expected(board):
            trigger = "surprise"
        elif self._r_at_plan is not None and r_now < self._r_at_plan - self.drop_delta:
            trigger = "dropped"
        elif self._plies_since_plan >= self.max_plies_per_plan:
            trigger = "stalled"
        self.last_trigger = trigger
        if trigger is not None:
            self.mcts.max_nodes = self.plan_nodes
            self._plies_since_plan = 0
            self._r_at_plan = r_now
            self.replans += 1
        else:
            self.mcts.max_nodes = self.exec_nodes
            self._plies_since_plan += 1
        return super().move(board, rng)
