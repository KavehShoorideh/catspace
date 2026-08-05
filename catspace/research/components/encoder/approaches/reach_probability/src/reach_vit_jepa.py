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
    """TWO IQE heads: best play, and human fallibility as a RESIDUAL on top of it.

    Kaveh 2026-08-05: "I want to LEARN the best play, and I want to LEARN the human likelihood of
    mistake directly as a residual on top of that perfect play" -- "so i want two different IQE
    heads".

        d_best(u->v)   = IQE_best(z_best(u),  z_best(v))      trained on SF-vs-SF (best play)
        d_human(u->v)  = d_best(u->v) + IQE_res(z_res(u), z_res(v))

    WHY ADDITION, AND WHAT IT BUYS -- this is an identification, so here are the criteria and the
    check (all verified numerically, not asserted):

      * THE SUM OF TWO QUASIMETRICS IS A QUASIMETRIC. d(x,x) = 0+0 = 0; non-negativity is closed
        under addition; and the triangle inequality adds side by side. Measured on 200 random
        triples: identity |d(x,x)| max 0.0, triangle satisfied 200/200, mean |d(u,v)-d(v,u)| 1.223
        (still genuinely asymmetric). So d_human is a valid quasimetric and every downstream thing
        that relies on the axioms -- the subsumption poles, the triangle-chaining, the IQE
        readouts -- remains valid for the human field, not just the base one.

      * d_human >= d_best BY CONSTRUCTION (verified exactly), because the residual is a distance
        and distances are non-negative. That is the right inductive bias and it is a theorem rather
        than a hope: a fallible player never reaches a goal FASTER than perfect play. A free-form
        residual head could learn a negative correction and quietly claim humans outplay Stockfish.

      * THE RESIDUAL IS THE DELIVERABLE. d_mistake(u->v) is, in distance units, how much further a
        human is from getting from u to v than best play is -- read per endgame, it is the
        "likelihood of mistake" this whole line exists to measure, and it is directionally
        resolved (d_mistake(u->v) need not equal d_mistake(v->u)), which matters because botching
        the conversion of a won endgame is not the same event as failing to hold a draw.

    ZERO-INIT: z_res's projection starts at zero, so every z_res is identical, every interval is
    empty, and d_mistake is EXACTLY 0 at step 0 (verified). The model therefore begins at "humans
    play perfectly" and the data has to push it off that -- the same discipline as the pole
    residual and the M2b style encoder.

    CONDITIONED, NOT BINARY (Kaveh 2026-08-05): "we could also treat the next stage as a
    strength+style conditioned residual, which will extract sf or human play as needed, on top of
    this; instead of supplanting this." So `d_cond > 0` makes the residual a function of a
    CONTINUOUS conditioning vector -- strength plus style z -- rather than of a human/SF flag:

        d(u->v | c) = d_best(u->v) + d_res(u->v ; c)

    Three things this buys over the binary split, and one thing to watch:
      * It generalises to PER-PLAYER. z is the M2b style residual, so "how badly does THIS opponent
        botch this endgame" is the same query with a different c -- the individual-z pathway the
        opponent work already established as the main one, rather than a population average.
      * SF and human stop being separate models. SF is simply the c where the residual is near
        zero; no branch, no second field, and the pooled base trained today is the warm start
        rather than something to discard.
      * The floor property SURVIVES because the residual is still an IQE, hence >= 0. So
        d(.|c) >= d_best for EVERY c, i.e. the base IS best play by construction and no
        conditioning can claim a player outruns it. The tempting alternative -- a free conditioned
        field with d_mistake read off as d(.|c_human) - d(.|c_SF) -- throws this away: a difference
        of two quasimetrics is neither sign-constrained nor a quasimetric.
      * TO WATCH: the base must actually be fitted to the STRONGEST play, not the pooled mean, or
        "floor" is a misnomer and every residual is measured against a half-human reference. That
        is what `detach_base` and `source` are for; c changes what the residual can express, not
        where the base sits.
    """

    def __init__(self, d_in: int, d: int = 64, components: int = 16, leak_beta: float = 0.0,
                 d_cond: int = 0):
        super().__init__()
        self.d_cond = int(d_cond)
        self.proj_best = nn.Linear(d_in, d)
        self.proj_res = nn.Linear(d_in + self.d_cond, d)
        nn.init.zeros_(self.proj_res.weight)              # d_mistake == 0 at step 0
        nn.init.zeros_(self.proj_res.bias)
        self.iqe_best = IQE(d, components=components, leak_beta=leak_beta)
        self.iqe_res = IQE(d, components=components, leak_beta=leak_beta)

    def embed(self, phi, cond=None):
        """-> (z_best, z_res). Two projections of ONE shared representation, which is what makes
        the residual a difference in geometry rather than a difference in encoder noise.

        `cond` (B, d_cond) is the strength+style vector. It enters ONLY the residual: the base must
        not see it, or best play would become conditional on who is playing, which is exactly the
        thing the floor is defined to be independent of."""
        if self.d_cond:
            if cond is None:
                cond = phi.new_zeros(len(phi), self.d_cond)
            return self.proj_best(phi), self.proj_res(torch.cat([phi, cond], -1))
        return self.proj_best(phi), self.proj_res(phi)

    def d_best(self, zb_u, zb_v):
        return self.iqe_best(zb_u, zb_v)

    def d_mistake(self, zr_u, zr_v):
        """The residual alone: how much further a human is than best play. >= 0 by construction."""
        return self.iqe_res(zr_u, zr_v)

    def d_human(self, zb_u, zb_v, zr_u, zr_v):
        return self.d_best(zb_u, zb_v) + self.d_mistake(zr_u, zr_v)

    def distance(self, zb_u, zb_v, zr_u=None, zr_v=None, source=None, detach_base=True):
        """Dispatch by dynamics: SF rows get d_best, human rows get d_human.

        `source` is the POLE_SRC id (0 = SF base, 1 = human residual), so a mixed batch trains both
        heads at once and the residual only ever sees human rows -- an SF row must never contribute
        gradient to the mistake head, or "the human penalty" would absorb engine data too.

        `detach_base` IS THE IDENTIFIABILITY CONSTRAINT, and without it the decomposition is not
        identified at all: human rows produce d_best + d_mistake, so gradient flows into BOTH and
        any amount of human error can be absorbed into d_best instead of the residual. The split
        would then be an arbitrary consequence of optimisation order rather than a measurement.
        Detaching the base on human rows pins the definition -- "best play" is what SF data says it
        is, full stop, and d_mistake is whatever is left over. This is the in-run form of the
        frozen-base-plus-residual discipline the M2b style encoder uses; the stricter version is to
        train the base to convergence on SF, freeze it, and fit the residual afterwards.
        """
        d = self.d_best(zb_u, zb_v)
        if zr_u is None:
            return d
        if self.d_cond:
            # CONDITIONED MODE: every row gets base + residual, engine included (Kaveh 2026-08-05:
            # "both best play (engine) and human play would need both base and residual terms").
            # There is no branch on population -- SF is just the conditioning point where the
            # residual should come out near zero, which makes "the base is best play" a CHECKABLE
            # prediction (see residual_magnitude) instead of an assumption baked in by masking.
            #
            # WHAT PINS THE BASE HERE, since gradient masking no longer does. d_res >= 0 alone is
            # satisfied by d_base = 0 with the residuals carrying everything, so the base is
            # identified only up to "some lower bound". SHRINKAGE on the residual is what makes it
            # the TIGHT one: penalise residual magnitude and the optimiser explains as much as it
            # can with the shared base, spending residual only where populations really differ.
            # The trainer must therefore carry a residual-shrinkage term; without it this whole
            # decomposition is unidentified and the base will drift toward zero.
            return d + self.d_mistake(zr_u, zr_v)
        if source is None:
            return d
        base = d.detach() if detach_base else d
        is_h = (source == 1)
        return torch.where(is_h, base + self.d_mistake(zr_u, zr_v), d)

    def residual_magnitude(self, zr_u, zr_v):
        """Mean d_res over the batch -- the SHRINKAGE target, and the diagnostic.

        As a loss term it is what identifies the base as the tight lower envelope rather than any
        lower bound. As a readout, its value AT THE SF CONDITIONING POINT is the falsifiable check
        on the whole design: if best play still needs a large residual, the base is not best play
        and every human 'mistake' number measured against it is inflated by that amount."""
        return self.d_mistake(zr_u, zr_v).mean()


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
                 init_scale: float = 0.01):
        super().__init__()
        self.n_poles, self.d, self.n_sources = int(n_poles), int(d), int(n_sources)
        self.poles = nn.Parameter(torch.randn(n_poles, d) * init_scale)
        # `parent[i]` = the pole that pole i subsumes into, or -1 at a root.
        self.register_buffer("parent", parent.long())
        self.delta = nn.Parameter(torch.zeros(max(self.n_sources - 1, 0), n_poles, d)) \
            if self.n_sources > 1 else None

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
                 leak_beta: float = 0.0, dual: bool = False, d_cond: int = 0):
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
        self.poles = None                # attach_poles() installs the subsumption hierarchy

    def attach_poles(self, parent, n_sources: int = 2):
        """Install the terminal/endgame subsumption hierarchy (see PoleBank).

        Kept out of __init__ because the pole set is DATA-derived -- which (ending x material
        signature) pairs clear the min_count threshold is a property of the corpus, not of the
        architecture -- so the trainer builds it from the store and hands it over here. The parent
        vector is saved in the checkpoint cfg so evaluation rebuilds the identical hierarchy."""
        self.poles = PoleBank(len(parent), self.d, torch.as_tensor(parent), n_sources=n_sources)
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

    def distance_cond(self, zb_u, zb_v, zr_u=None, zr_v=None):
        """d(u->v | c) = d_best + d_res. Dual mode only; >= d_best for every conditioning."""
        return self.qhead.distance(zb_u, zb_v, zr_u, zr_v)

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
        return self.qhead.d_best(zq_a, zq_b) if self.dual else self.iqe(zq_a, zq_b)

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
