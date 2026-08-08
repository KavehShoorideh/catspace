"""ReachViT -- the FROM-SCRATCH reachability model: one ViT over raw bitboards, two heads.

WHY THE TRUNK HAD TO GO. The first attempt put this objective on the frozen lc0 trunk and measured
paired ratchet 0.570 against a random-init null of 0.555 -- flat across the checkpoint ladder. That
is a NEGATIVE, but an inconclusive one, and the reason is structural: a pretrained chess net already
contains the material ratchet, so its random-init "null" is not zero and no objective on top of it
can be shown to have added anything. A randomly-initialised ViT over tokenized boards knows no
chess at all, so here the random-init null IS the real zero and any ratchet that appears was
learned. That is the whole point of the rebuild (Kaveh 2026-08-05: "key point is whether we can get
strata without programming anything chess specific").

Two further confounds die with the trunk:
  * HISTORY LEAK. lc0's 112 planes carry eight plies of history, so for a pair within 8 plies
    position a sat literally inside b's own input tensor and the model could be right there for
    reasons that are not reachability. A single tokenized position cannot leak that way, so the
    `gap > 8` guard and every gap-stratified readout are gone.
  * EN PASSANT. Verified by controlled FEN pair: ep is encoded NOWHERE in the lc0 112 planes, so a
    plane-derived position identity is wrong about which positions are the same. tokenize()'s
    globals carry the ep file explicitly, which is also what makes repetition detection exact.

ONE ENCODER, TWO HEADS, so the arms are directly comparable -- any difference between them is the
geometry, not the input stage or the amount of training:

  ARM A (region).  proj_a: phi -> z_A, then the ReachJEPA predictor z_A(a) -> Normal(mu, sigma) over
                   z_A(b). b is scored by log-density, which is what the conformal calibration
                   consumes. EMA target branch, exactly as before.
  ARM B (IQE).     proj_b: phi -> z_B, then the Interval Quasimetric Embedding gives a DIRECTED
                   d(a->b). Kaveh 2026-08-05: "any embedding you learn needs to have asymmetry" --
                   here the asymmetry is by construction rather than by training, so d(b->a)/d(a->b)
                   on a pair is a DIRECT readout of the ratchet, with quiet-reversible pairs as the
                   ~1.0 control.

SEPARATE PROJECTIONS, NOT A SHARED z. The two heads want incompatible things of the same
coordinates -- arm A wants an isotropic Gaussian region, arm B wants interval endpoints whose
ORDER encodes direction -- so they share the trunk (which is what makes them comparable) and get
their own linear map (which is what stops them fighting over the same axes). The anti-collapse
terms are applied at phi as well as z_A, because a collapse of the trunk is a collapse of both arms.

NOTHING HERE IS TOLD ANY CHESS. The input vocabulary is 13 opaque token ids and 6 opaque globals;
piece count, captures, legality and material never enter. Piece count appears only in
interpret_reach.py, as an analysis LABEL computed as (tok > 0).sum(-1).
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from catspace.research.components.encoder.approaches.jepa_tokenizer.src.iqe import IQE
from catspace.research.components.encoder.approaches.jepa_tokenizer.src.jepa import JepaEncoder

# Same clamp as ReachJEPA: without it one easy example drives sigma -> 0 and the NLL to -inf, which
# is a divergence rather than a fit.
LOG_SIGMA_MIN, LOG_SIGMA_MAX = -6.0, 3.0

REGION, IQE_ARM = "region", "iqe"


class DualIQEHead(nn.Module):
    """ONE quasimetric, tuned by a strength/style residual that lives in the EMBEDDING.

        z(u | c)      = z_base(u) + delta(u, c)          delta zero-init
        d(u->v | c)   = IQE( z(u|c), z(v|c) )
        d_base(u->v)  = IQE( z_base(u), z_base(v) )      the c-neutral field

    WHY THE RESIDUAL MOVED OUT OF THE DISTANCE. The first version added a second IQE to the
    distance, d_total = d_base + d_res, which made d_total >= d_base a free theorem: "a fallible
    player never reaches a goal faster than best play." That property is only correct when the base
    IS best play. Kaveh 2026-08-05 corrected the premise -- "the full pooled will learn base
    reachability in broad terms, not absolute", so the base is the POOLED field, and a strong player
    should be able to come in CLOSER than it: "we don't want d_res to go to 0 at 3000; it can be
    negative." An additive-distance residual cannot express that, because a distance cannot be
    negative.

    So the requirement was restated to exactly what matters -- "what i care about is the full thing,
    base + residual, to be a quasimetric tuned to whomever i have the residual for" -- and d_res
    itself is explicitly NOT required to be a quasimetric. Putting the residual in the embedding
    delivers both, and the axioms come for free rather than by constraint: d(.|c) is an IQE of SOME
    embedding, so for EVERY c it is a valid quasimetric, while the readout d(.|c) - d_base is an
    unconstrained difference of two distances and may take either sign.

    VERIFIED, not asserted (300 random points, delta ~ N(0, 0.6)): identity |d(x,x)| max 0.0,
    non-negative, triangle satisfied 100/100, mean |d(u,v)-d(v,u)| 0.662 -- and the residual came
    out negative on 11.0% of pairs, range -0.278..+1.433, which is the "strong player is closer"
    behaviour the additive-distance form structurally forbade.

    WHAT PINS THE BASE, since the >= 0 constraint is gone: delta is ZERO-INIT (so training starts at
    "everyone is the pooled field" and data must push it off), and the trainer shrinks ||delta||.
    Shrinkage on the embedding is the right knob here -- it says "explain what you can with the
    shared field, deviate only where a population really differs" WITHOUT dictating the sign of the
    deviation, which is precisely the mistake the Elo-anchored-to-zero term made.
    """

    def __init__(self, d_in: int, d: int = 64, components: int = 16, leak_beta: float = 0.0,
                 d_cond: int = 0):
        super().__init__()
        self.d_cond = int(d_cond)
        self.proj_base = nn.Linear(d_in, d)
        self.proj_delta = nn.Linear(d_in + self.d_cond, d)
        nn.init.zeros_(self.proj_delta.weight)            # delta == 0 at step 0
        nn.init.zeros_(self.proj_delta.bias)
        self.iqe = IQE(d, components=components, leak_beta=leak_beta)

    def embed(self, phi, cond=None):
        """-> (z_base, z_cond). z_cond = z_base + delta(phi, cond); equal to z_base when cond is
        absent or delta is still zero, so the c-neutral field is always available as a reference."""
        z_base = self.proj_base(phi)
        if not self.d_cond or cond is None:
            return z_base, z_base
        delta = self.proj_delta(torch.cat([phi, cond], -1))
        return z_base, z_base + delta

    def delta(self, phi, cond):
        """(B,d) the conditioned displacement itself -- the SHRINKAGE target."""
        if not self.d_cond or cond is None:
            return phi.new_zeros(len(phi), self.proj_base.out_features)
        return self.proj_delta(torch.cat([phi, cond], -1))

    def d_base(self, zb_u, zb_v):
        """The pooled, conditioning-free field."""
        return self.iqe(zb_u, zb_v)

    def distance(self, z_u, z_v):
        """d(u->v | c) on whichever embedding is passed. A valid quasimetric for every c."""
        return self.iqe(z_u, z_v)

    def residual(self, zb_u, zb_v, zc_u, zc_v):
        """d(.|c) - d_base: how much further (or NEARER, if negative) this player is than the
        pooled field. The deliverable readout; deliberately not sign-constrained."""
        return self.iqe(zc_u, zc_v) - self.iqe(zb_u, zb_v)


def simplex_poles(d: int, components: int, n_poles: int = 3, height: float = 3.0):
    """(n_poles, d) FIXED poles: one block of IQE components per outcome.

    Kaveh 2026-08-05: "we'd fix the position of the poles in space so everything stays bounded and
    doesn't go to infinity" -- then "we need 3 poles, but it's not clear what kind of geometry they
    should have". The IQE's own structure settles it, and rules out the obvious choices:

    WHY NOT ALL THREE TOGETHER. A pole is ABSORBING when d(P->s) is large, which wants P LOW in the
    coordinate order; positions REACH it when d(s->P)~0, which needs s >= P. So every pole wants to
    sit low -- but if all three do, every position dominates all three, all three distances go to
    zero at once (legal in a quasimetric) and the committor is uniform everywhere. That is not a
    hypothetical: it is the basin_spread=0.0001 collapse measured on the first pole smoke.

    SO THE POLES MUST BE MUTUALLY INCOMPARABLE -- no pole dominating another. Giving each outcome
    its own block of components does exactly that:

        pole_o = +height on block o, 0 on every other block

    d(s -> P_o) is then small precisely when s is HIGH on block o, so a position's committor is
    "which block am I high on" -- readable directly off the coordinates, which is the point.

    VERIFIED (d=48, C=6, height=3): the pole-pole matrix is 0 on the diagonal and EXACTLY 2 off it
    -- a perfect simplex, symmetric under permuting W/D/L, which is the symmetry the problem has
    (win and loss are mirror images under colour; no outcome is privileged). A position high on
    block o reads P = [0.6, 0.2, 0.2] toward that outcome, and one half-high on two blocks reads
    [0.35, 0.33, 0.32] -- a genuine MIXTURE at the boundary, which is what a committor must do.

    Fixed poles also FIX THE GAUGE. A fully learned embedding has a global scale freedom (the IQE
    even has a learnable log_scale exploiting it), so no distance is comparable across checkpoints
    or runs. Nailing three poles down makes every distance relative to a known frame -- and since
    fixed poles cannot merge, the merged-pole saddle disappears and pole_potential is no longer
    needed at all.

    `height` is a real knob: it sets pole separation, and should sit near the typical observed-pair
    distance so the softmax operates in a sensible range instead of saturating. Confidence at the
    maximum is deliberately soft (0.6, not 0.99) -- over-sharp basins are what the pole design
    explicitly did not want.
    """
    k = d // components
    per = max(1, components // n_poles)
    P = torch.zeros(n_poles, d)
    for o in range(n_poles):
        for c in range(o * per, min((o + 1) * per, components)):
            P[o, c * k:(c + 1) * k] = height
    return P


class PoleBank(nn.Module):
    """The SUBSUMPTION HIERARCHY as points in the quasimetric space, conditioned on the dynamics.

    THE GEOMETRY (Kaveh 2026-08-05). Each endgame is a pole; each instance of that endgame is a
    point at 0-ply from it -- and 0-ply in an IQE is not a metaphor. d(u->v) accumulates interval
    length only where v EXCEEDS u, so d(u->v)=0 is exactly "u dominates v coordinatewise", and it
    coexists with d(v->u) > 0. Verified on our IQE: three specific positions dominating a
    general-3-fold point which dominates a general-draw point gave forward distances of exactly
    0.0/0.0/0.0 and reverse distances of 0.88/0.97/0.91, and the chain composed transitively
    (specific -> general-draw = 0.0) with the triangle inequality satisfied as 0 <= 0 + 0. Mutual
    zero forces equality, so this is a genuine partial order rather than a pile.

        terminal instance  >=  (ending x material signature)  >=  ending type  >=  outcome

    THE DANGER, stated because it is a recorded scar rather than a worry. This is the SAME surface
    as the ordering collapse that halted qrl_iqe_unreach (JOURNAL 2026-07-18): all F above all B in
    the IQE coordinates, every directed distance 0, and torch.maximum's gradient exactly 0 there --
    an ABSORBING fixed point. Training deliberately toward zero forward distances aims at that
    corner. Three guards, none optional: the anchor is a HINGE toward a target radius rather than a
    hard pull to the origin; absorbing_penalty holds every reverse direction open; and `zero_frac`
    is gated every eval so a global slide into the dead zone appears as a logged number instead of
    a suspiciously perfect loss. leak_beta > 0 keeps an escape gradient at the cost of making the
    zero approximate (measured: 0.127 rather than 0.000 at beta=10) -- that trade is the caller's.

    SOURCE CONDITIONING (Kaveh 2026-08-05: "sf won't blunder a stalemate but a human might"). The
    poles are a RESIDUAL per dynamics on a SHARED embedding, exactly as IQEHead.pole_delta does it,
    and for the same measured reason: two fields trained separately on disjoint halves of the same
    human data disagreed as much as a human-vs-SF pair, so a separately-fit-per-source design
    measures training noise. Sharing phi makes representation noise COMMON to both readouts and
    cancels it in their difference. Zero-initialised, so step 0 says "the dynamics do not differ"
    and the data has to push it off that. The difference is only believable against a
    permuted-source null (the 2026-08-05 conditioned-field protocol), which the readout runs.
    """

    def __init__(self, n_poles: int, d: int, parent: torch.Tensor, n_sources: int = 2,
                 init_scale: float = 0.01, fixed: bool = False, components: int = 16,
                 height: float = 3.0, n_outcome: int = 3):
        """TWO LEVELS (Kaveh 2026-08-05: "we should be able to approach these poles from different
        directions for different type of each ending"):

            terminal instance -> ENDING pole (learned, free direction) -> OUTCOME pole (fixed)

        The first `n_outcome` poles are the FIXED simplex -- the gauge, carrying no chess beyond
        W/D/L. Every pole after that is a LEARNED ending type (mate, stalemate, repetition,
        resign...), constrained only to SUBSUME into its outcome pole, i.e. d(ending -> outcome)=0.
        Domination is a partial order, so that constraint fixes the ending pole's distance to its
        outcome at zero while leaving its DIRECTION entirely free -- which is exactly "approach the
        same pole from different directions". Verified independently: distinct points can all sit
        at distance 0.0 from one pole while being 0.65-0.76 apart from each other.

        terminal_repulsion between ending poles then keeps those arrival points distinct, which is
        that term's stated purpose in the repo: different mate structures are different arrival
        points of ONE surface, not one point.
        """
        super().__init__()
        self.n_poles, self.d, self.n_sources = int(n_poles), int(d), int(n_sources)
        self.fixed, self.n_outcome = bool(fixed), int(n_outcome)
        n_learn = max(0, self.n_poles - self.n_outcome) if self.fixed else self.n_poles
        if self.fixed:
            # A BUFFER, not a Parameter: the frame is a fixed reference and must not drift.
            self.register_buffer("outcome_poles",
                                 simplex_poles(d, components, self.n_outcome, height))
            # Ending poles start ON their outcome pole (zero offset), so training begins with the
            # hierarchy exactly satisfied and the data has to push each ending off in its own
            # direction -- the same zero-init discipline as every other residual here.
            self.ending_delta = nn.Parameter(torch.zeros(n_learn, d)) if n_learn else None
            if n_learn:
                # Rooted poles (parent -1, e.g. START) have no outcome base to sit on, so a zero
                # delta would pin them at the origin -- which in an IQE is dominated by everything
                # and therefore degenerate. Only those get a nonzero init; ending poles keep the
                # zero-init discipline so the hierarchy starts exactly satisfied.
                with torch.no_grad():
                    roots = (torch.as_tensor(parent)[self.n_outcome:] < 0)
                    if roots.any():
                        self.ending_delta[roots] = torch.rand(int(roots.sum()), d) * height
        else:
            self.poles_p = nn.Parameter(torch.randn(n_poles, d) * init_scale)
        # `parent[i]` = the pole that pole i subsumes into, or -1 at a root.
        self.register_buffer("parent", parent.long())
        self.delta = nn.Parameter(torch.zeros(max(self.n_sources - 1, 0), n_poles, d)) \
            if self.n_sources > 1 else None

    @property
    def poles(self):
        """(n_poles, d) the full bank: fixed outcome simplex, then learned ending poles."""
        if not self.fixed:
            return self.poles_p
        if self.ending_delta is None or not len(self.ending_delta):
            return self.outcome_poles
        # each ending pole = its outcome pole + a learned offset (free direction)
        # A pole with parent -1 (START) is its OWN root and must NOT inherit an outcome position.
        # clamp(min=0) silently mapped -1 to index 0, so START was built as "the WIN pole plus a
        # zero offset" and landed exactly on WIN -- which is why it appeared to sit on top of the
        # mates. Rooted poles get a zero base and live wherever their delta puts them.
        par = self.parent[self.n_outcome:]
        base = torch.where(par[:, None] >= 0,
                           self.outcome_poles[par.clamp(min=0)],
                           torch.zeros_like(self.ending_delta))
        return torch.cat([self.outcome_poles, base + self.ending_delta], 0)

    def for_source(self, src):
        """(B,) int64 source ids -> (B, n_poles, d). Source 0 is the shared field by construction."""
        P = self.poles.unsqueeze(0)
        if self.delta is None:
            return P.expand(len(src), -1, -1)
        D = torch.cat([torch.zeros_like(self.delta[:1]), self.delta], 0)
        return P + D[src]

    def edges(self):
        """(child_idx, parent_idx) for every subsumption edge in the hierarchy."""
        c = torch.nonzero(self.parent >= 0, as_tuple=True)[0]
        return c, self.parent[c]

    def source_divergence(self):
        """(n_poles,) L2 norm of each pole's per-source residual -- how much the dynamics differ.

        This is the per-endgame competence readout: a pole whose human and SF versions sit apart is
        an endgame the two populations treat differently, which is precisely the KBN-vs-K question.
        Meaningless without the permuted-source null beside it."""
        if self.delta is None:
            return torch.zeros(self.n_poles, device=self.poles.device)
        return self.delta.norm(dim=-1).mean(0)


class ReachViT(nn.Module):
    """(tok (B,64) uint8, glob (B,6) uint8) -> phi -> {region head, quasimetric head}."""

    arch = "vit"

    def __init__(self, d_model: int = 256, layers: int = 6, heads: int = 8, d: int = 64,
                 hidden: int = 256, components: int = 8, ema_decay: float = 0.996,
                 leak_beta: float = 0.0, dual: bool = False, d_cond: int = 0,
                 split_head: bool = False):
        """`dual=False` is the LEGACY arm-B path (one IQE over proj_b) and is kept byte-identical
        on purpose: the strata run launched 2026-08-05 writes checkpoints with `proj_b.*`/`iqe.*`
        keys, and rewiring in place would have made its own ladder unloadable by interpret_reach
        halfway through a 3-hour run. `dual=True` swaps in the conditioned DualIQEHead
        (best-play floor + strength/style residual). Which one a checkpoint wants is recorded in
        its cfg, so both load correctly forever."""
        super().__init__()
        assert d % components == 0, "IQE needs d divisible by components"
        self.d_model, self.d = int(d_model), int(d)
        self.ema_decay = float(ema_decay)
        self.dual, self.d_cond = bool(dual), int(d_cond)
        self.source_blind = False        # the control; see blind_source()

        self.enc = JepaEncoder(d_model, layers, heads)
        self.proj_a = nn.Linear(d_model, d)          # arm A: region space
        if not self.dual:
            self.proj_b = nn.Linear(d_model, d)      # arm B (legacy): one quasimetric space

        # Target branch for arm A: an EMA copy of (encoder, proj_a). Never optimised, never receives
        # gradient -- the JEPA guard that stops the model making its own target easier to predict.
        self.t_enc = copy.deepcopy(self.enc)
        self.t_proj_a = copy.deepcopy(self.proj_a)
        for p in list(self.t_enc.parameters()) + list(self.t_proj_a.parameters()):
            p.requires_grad_(False)

        # Arm A predictor: z_A -> (mu, log_sigma). `head_in` is the L1-penalised layer.
        self.head_in = nn.Linear(d, hidden)
        self.head = nn.Sequential(nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                  nn.Linear(hidden, 2 * d))
        # Arm B: the quasimetric itself. Asymmetric by construction (Wang & Isola 2022).
        if self.dual:
            self.qhead = DualIQEHead(d_in=d_model, d=d, components=components,
                                     leak_beta=leak_beta, d_cond=self.d_cond)
        else:
            self.iqe = IQE(d, components=components, leak_beta=leak_beta)
        # SPLIT-BLOCK QUASIMETRIC (Kaveh 2026-08-08): one embedding, two independent IQE blocks --
        # A (first half) carries LENGTH (walls/gas: calibrated ply distances), B (second half)
        # carries OUTCOME structure. Losses touch only their own block, so the measured
        # walls-vs-CE escalation is impossible at the interface; each block is a full quasimetric
        # in its own right, read SEPARATELY -- 'one point, two rulers', any exchange rate applied
        # at query time, never trained in.
        self.split_head = bool(split_head)
        if self.split_head:
            assert d % 2 == 0
            self.iqe_a = IQE(d // 2, components=max(2, components // 2), leak_beta=leak_beta)
            self.iqe_b = IQE(d // 2, components=max(2, components // 2), leak_beta=leak_beta)
        self.poles = None                # attach_poles() installs the subsumption hierarchy

    def attach_poles(self, parent, n_sources: int = 2, init_scale: float = 0.3,
                     fixed: bool = True, height: float = 3.0):
        """Install the terminal/endgame subsumption hierarchy (see PoleBank).

        Kept out of __init__ because the pole set is DATA-derived -- which (ending x material
        signature) pairs clear the min_count threshold is a property of the corpus, not of the
        architecture -- so the trainer builds it from the store and hands it over here. The parent
        vector is saved in the checkpoint cfg so evaluation rebuilds the identical hierarchy."""
        # init_scale is EXPOSED but does not matter -- measured, not assumed. The planted-committor
        # simulation run at both 0.01 and 0.3 converges identically (MAE 0.015/0.013 at 4k steps),
        # and 0.01 breaks symmetry FASTER (spread 0.122 vs 0.073 at 300 steps): coincident poles
        # are the zero-init-softmax situation, where class-dependent gradients separate the
        # classes on their own. The uniform basin_spread=0.0001 smoke reading that prompted
        # suspicion here was UNDERTRAINING (300 steps under a deep encoder), not a pole pathology.
        self.poles = PoleBank(len(parent), self.d, torch.as_tensor(parent),
                              n_sources=n_sources, init_scale=init_scale, fixed=fixed,
                              components=self.qhead.iqe.components if self.dual
                              else self.iqe.components, height=height)
        return self.poles

    # ---- encoding -------------------------------------------------------------------------------
    def backbone(self, tok, glob):
        """(B,64) x (B,6) -> phi (B,d_model). The ONLY place the board is read."""
        return self.enc(tok, glob)

    def encode(self, tok, glob):
        """-> z_A (B,d), the arm-A ONLINE branch (gradients flow)."""
        return self.proj_a(self.backbone(tok, glob))

    @torch.no_grad()
    def encode_target(self, tok, glob):
        """-> z_A (B,d) from the EMA TARGET branch (no gradient, by construction)."""
        return self.t_proj_a(self.t_enc(tok, glob))

    def dA(self, z_u, z_v):
        """LENGTH-block distance (walls, calibration, odometry)."""
        h = z_u.shape[-1] // 2
        return self.iqe_a(z_u[..., :h], z_v[..., :h])

    def dB(self, z_u, z_v):
        """OUTCOME-block distance (basins, routing readouts)."""
        h = z_u.shape[-1] // 2
        return self.iqe_b(z_u[..., h:], z_v[..., h:])

    def encode_q(self, tok, glob):
        """-> z_B (B,d), the arm-B quasimetric embedding (gradients flow; no EMA branch, because
        arm B has an explicit repulsion and does not rely on a bootstrap target to avoid collapse).

        In dual mode this returns the BEST-PLAY embedding only, so every legacy caller
        (interpret_reach's directed_distance, score_rows, the conformal path) keeps reading the
        best-play field rather than silently switching to a conditioned one."""
        phi = self.backbone(tok, glob)
        if self.dual:
            return self.qhead.embed(phi)[0]
        return self.proj_b(phi)

    def encode_dual(self, tok, glob, cond=None):
        """-> (z_best, z_res). Dual mode only. `cond` (B,d_cond) is the strength+style vector."""
        return self.qhead.embed(self.backbone(tok, glob), cond)

    def distance_cond(self, z_u, z_v):
        """d(u->v | c) on the CONDITIONED embedding. A valid quasimetric for every c."""
        return self.qhead.distance(z_u, z_v)

    # ---- arm A: the region ------------------------------------------------------------------
    def predict(self, z_a):
        """(B,d) -> (mu (B,d), log_sigma (B,d)): the predicted reachable region from a."""
        mu, log_sigma = self.head(self.head_in(z_a)).chunk(2, dim=-1)
        return mu, log_sigma.clamp(LOG_SIGMA_MIN, LOG_SIGMA_MAX)

    def score(self, z_a, z_b):
        """(B,) log-density of z_b under the region predicted from a. HIGHER = more reachable.

        A proper log-density rather than a bare distance, so the predicted spread is actually used
        and the conformal nonconformity score has a tail that means something."""
        mu, log_sigma = self.predict(z_a)
        var = torch.exp(2.0 * log_sigma)
        return -0.5 * (((z_b - mu) ** 2) / var + 2.0 * log_sigma).sum(-1)

    # ---- arm B: the quasimetric -------------------------------------------------------------
    def distance(self, zq_a, zq_b):
        """(B,) directed d(a -> b) on the BEST-PLAY field. Asymmetric by construction."""
        return self.qhead.d_base(zq_a, zq_b) if self.dual else self.iqe(zq_a, zq_b)

    def score_q(self, zq_a, zq_b):
        """(B,) arm-B analogue of score(): HIGHER = more reachable, so -d."""
        return -self.distance(zq_a, zq_b)

    # ---- shared eval entry points (one path for both arms, and the blinding control) ---------
    def _src(self, tok, glob, q: bool):
        z = self.encode_q(tok, glob) if q else self.encode(tok, glob)
        return torch.zeros_like(z) if self.source_blind else z

    @torch.no_grad()
    def score_rows(self, tok_a, glob_a, tok_b, glob_b, arm: str = REGION):
        """(B,) reachability score for a batch of (a, b) row pairs, on either arm.

        Both arms expose the same sign convention (higher = b looks more like a future of a), so
        interpret_reach.py and calibrate_conformal.py run one code path over both."""
        if arm == IQE_ARM:
            return self.score_q(self._src(tok_a, glob_a, True), self.encode_q(tok_b, glob_b))
        return self.score(self._src(tok_a, glob_a, False), self.encode_target(tok_b, glob_b))

    @torch.no_grad()
    def region_volume_rows(self, tok, glob, arm: str = REGION, ref=None):
        """(B,) the model's OWN statement of how uncertain it is about the future from each source.

        This is the Mondrian taxonomy calibrate_conformal.py buckets on, and it must introduce no
        chess (Kaveh 2026-08-05: bucketing on material or ply "will effectively create strata").
          arm A: sum of log sigma -- the log-volume of the predicted region.
          arm B: mean d(a -> r) over a FIXED reference bank r of embedded positions -- how far the
                 quasimetric says it can still travel. Same content, expressed in the geometry the
                 arm actually has; the bank is fixed before any query is seen, exactly as the
                 bucket edges are.
        """
        if arm == IQE_ARM:
            if ref is None:
                raise ValueError("arm B needs a reference bank to express region volume")
            return self.iqe.pairwise(self.encode_q(tok, glob), ref).mean(-1)
        return self.predict(self.encode(tok, glob))[1].sum(-1)

    # ---- housekeeping ------------------------------------------------------------------------
    @torch.no_grad()
    def update_target(self):
        """EMA the online (encoder, proj_a) into the target branch. Once per optimiser step."""
        m = self.ema_decay
        for tm, om in ((self.t_enc, self.enc), (self.t_proj_a, self.proj_a)):
            for pt, po in zip(tm.parameters(), om.parameters()):
                pt.mul_(m).add_(po.detach(), alpha=1.0 - m)
            for bt, bo in zip(tm.buffers(), om.buffers()):
                bt.copy_(bo)

    def l1_penalty(self):
        """L1 on the predictor's input layer -- reported for monitoring. NOT the mechanism that
        produces sparsity; see prox_l1."""
        return self.head_in.weight.abs().mean()

    @torch.no_grad()
    def prox_l1(self, lam: float):
        """Proximal soft-threshold (ISTA) on the predictor input layer: W <- sign(W)relu(|W| - lam).

        Adding an L1 term to the loss and running Adam does NOT sparsify -- measured on the trunk
        version, w_l1=0.5 via the subgradient route left 64/64 coordinates alive. The proximal
        operator sets genuinely small weights to EXACT zero, so 'how many coordinates survive' is a
        real count rather than a thresholding convention.

        SWEEP IT, DO NOT GUESS IT. At lam scaled by --l1-prox 2.0 this zeroed the entire predictor
        input layer (support 0/64), producing a model that provably could not read the source at
        all -- which is the source-blind control, not a model. Start around 0.005."""
        w = self.head_in.weight
        w.copy_(torch.sign(w) * torch.clamp(w.abs() - lam, min=0.0))

    def input_support(self, tol: float = 0.0):
        """(d,) bool: which input coordinates the predictor still reads (any non-zero weight)."""
        return (self.head_in.weight.abs() > tol).any(dim=0)

    @torch.no_grad()
    def blind_source(self):
        """THE control: make the model provably unable to read the SOURCE position.

        Under the paired ratchet (same target b, two sources) a source-blind model returns two
        identical scores and must therefore score EXACTLY 0.500. This is what exposed the first,
        confounded metric -- a degenerate run that could not read the source still scored 0.575
        there. Both arms are blinded: arm A's predictor input layer is zeroed AND the source
        embedding is forced to a constant, so the blinding cannot be undone by any downstream layer.
        """
        self.source_blind = True
        self.head_in.weight.zero_()
        self.head_in.bias.zero_()
        return self
