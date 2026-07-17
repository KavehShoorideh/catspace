"""
nn/fb.py — TorchFB: the full-board Forward-Backward embedding. Two
BoardEncoders (F-side, B-side) + MLP heads; F is conditioned on omega
(white/black Elo bins + clock bucket -- README lesson 1: the cone is
opponent-conditioned), B is board-only (goals are positions, not regimes).

The InfoNCE loss follows cone/neural.py (logits = F @ B.T / tau,
one-directional cross-entropy over in-batch negatives) with ONE deliberate
deviation: embeddings are L2-NORMALIZED (cosine InfoNCE). The toy recipe's
unnormalized dot was only safe because one-hot toy encodings have constant
input norm; real boards lose material over a game, activation norms shrink
with ply, and unnormalized F.B inherits that decline -- measured as a -0.92
spearman(ply, reach-to-anything) on won AND lost games alike before
normalization (2026-07-11 diagnostic).

Real boards have no state indices, so TorchFB does NOT implement the chain
QuasimetricEmbedding protocol; the seam here is embed_F / embed_B / reach_z
(same shapes as EncodedNeuralFB's precomputed arrays).

2026-07-11 `quasimetric=True` mode (JOURNAL.md, MRN -- Liu/Feng/Liu/Stone,
AAAI 2023): score(f,g) = -d(f,g) + r(f,g) instead of a plain dot product.
`d` is a REAL metric by construction (Euclidean distance between
per-dimension-rescaled embeddings: non-negative, symmetric, and satisfies
the triangle inequality for ANY three points in R^d regardless of which
network produced them -- that's a property of the norm on the shared
codomain, not of embed_F/embed_B individually), so multi-hop plans compose
correctly by construction instead of only when training data happened to
show that exact hop. `r` is a small unconstrained bilinear residual for
whatever directed/non-metric structure remains (the literature's own
finding: an unconstrained network provably cannot represent a consistent
quasimetric on its own -- PQE, Wang & Isola -- hence needing `d` to be
metric BY CONSTRUCTION, not learned freeform). `metric_scale` inits to
all-ones and `W` inits to zero, so a fresh quasimetric run starts with
score == -||f-g||_2 on the SAME unit-normalized embeddings the plain
cosine mode uses (monotonic in cosine similarity, d^2 = 2-2cos on unit
vectors) -- a smooth, not-arbitrarily-different starting point, same
spirit as this file's existing normalization discipline. Config-gated
(old checkpoints have quasimetric=False and never see metric_scale/W, so
they remain byte-for-byte unaffected); when False, score()/score_matrix()
reduce to exactly the prior `f @ b.T` behavior.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from catspace.nn.encoder import BoardEncoder
from catspace.nn.features import N_CLOCK_BINS, N_ELO_BINS, N_PLANES


def _softplus_inv(y: float) -> float:
    """x such that softplus(x)=y -- to init a softplus-parametrized positive."""
    import math
    return math.log(math.expm1(y))


class _GradReverse(torch.autograd.Function):
    """Identity forward, sign-flipped gradient backward. Turns a min-loss step
    on the Lagrange multiplier into dual ASCENT (QRL, Wang et al. 2023): the
    multiplier grows exactly when the transition constraint is violated."""

    @staticmethod
    def forward(ctx, x):
        return x

    @staticmethod
    def backward(ctx, g):
        return -g


def grad_reverse(x: torch.Tensor) -> torch.Tensor:
    return _GradReverse.apply(x)


class TorchFB(nn.Module):
    def __init__(self, d: int = 64, channels: int = 64, blocks: int = 6,
                 enc_out: int = 256, dh: int = 512, omega_dim: int = 16,
                 tau: float = 0.1, seed: int = 0, quasimetric: bool = False,
                 two_horizon: bool = False, distributional: bool = False,
                 n_bins: int = 12, competence: bool = False,
                 outcome_poles: bool = False, concept_axes: int = 0,
                 iqe: bool = False, iqe_components: int = 8,
                 iqe_embed_scale: float = 50.0):
        torch.manual_seed(seed)          # one seed, sequential construction:
        super().__init__()               # encF and encB draw DIFFERENT inits
        if two_horizon or distributional or outcome_poles or iqe:
            quasimetric = True           # all keep the quasimetric d as distance
        self.config = dict(d=d, channels=channels, blocks=blocks, enc_out=enc_out,
                           dh=dh, omega_dim=omega_dim, tau=tau, seed=seed,
                           quasimetric=quasimetric, two_horizon=two_horizon,
                           distributional=distributional, n_bins=n_bins,
                           competence=competence, outcome_poles=outcome_poles,
                           concept_axes=concept_axes, iqe=iqe,
                           iqe_components=iqe_components,
                           iqe_embed_scale=iqe_embed_scale)
        self.encF = BoardEncoder(N_PLANES, channels, blocks, enc_out)
        self.encB = BoardEncoder(N_PLANES, channels, blocks, enc_out)
        self.emb_we = nn.Embedding(N_ELO_BINS, omega_dim)
        self.emb_be = nn.Embedding(N_ELO_BINS, omega_dim)
        self.emb_clk = nn.Embedding(N_CLOCK_BINS, omega_dim)
        self.headF = nn.Sequential(nn.Linear(enc_out + 3 * omega_dim, dh), nn.ReLU(),
                                   nn.Linear(dh, d))
        self.headB = nn.Sequential(nn.Linear(enc_out, dh), nn.ReLU(), nn.Linear(dh, d))
        self.tau = tau
        self.d = d
        self.quasimetric = quasimetric
        self.iqe = iqe
        self.iqe_embed_scale = iqe_embed_scale
        if iqe:
            # Interval Quasimetric Embedding: valid+universal quasimetric BY
            # CONSTRUCTION (merged paper). Replaces the MRN metric_scale/W --
            # IQE is already asymmetric, so score = -d_iqe (no bilinear residual).
            from catspace.nn.iqe import IQE
            self.iqe_head = IQE(d, components=iqe_components)
        elif quasimetric:
            self.metric_scale = nn.Parameter(torch.ones(d))
            self.W = nn.Parameter(torch.zeros(d, d))
        if quasimetric:
            # QRL Lagrange multiplier (Wang et al. 2023): softplus-parametrized,
            # dual-ascended via grad_reverse. Only exercised by qrl_loss; inert
            # (unused param) under the InfoNCE objective. Init 1.0 (responsive
            # from step 0) rather than the paper's 0.01: our unbounded encoder
            # lets the global push inflate the embedding -- and with it the
            # 1-ply-step distance -- faster than a 0.01-init multiplier on a
            # softplus (whose gradient is throttled by sigmoid(raw)~0.01) can
            # rein in. lambda gets its own higher LR in the trainer too.
            self.qrl_raw_lambda = nn.Parameter(torch.tensor(_softplus_inv(1.0)))
        # TWO-HORIZON (2026-07-13, TWO_HORIZON_DESIGN.md): the existing
        # headF/headB (+ quasimetric) ARE the FAR head -- calibrated
        # long-range distance-to-goal. The near head is purely additive:
        # extra MLP heads off the SAME shared encoder trunk (encF/encB),
        # cosine-scored, trained only on short-gap pairs for razor-sharp
        # endgame/close-range discrimination. Constructed LAST so a
        # two_horizon=False model's weights stay byte-identical.
        self.two_horizon = two_horizon
        if two_horizon:
            self.headF_near = nn.Sequential(nn.Linear(enc_out + 3 * omega_dim, dh),
                                            nn.ReLU(), nn.Linear(dh, d))
            self.headB_near = nn.Sequential(nn.Linear(enc_out, dh), nn.ReLU(),
                                            nn.Linear(dh, d))
        # DISTRIBUTIONAL head (2026-07-13, UNCERTAINTY_DESIGN.md option B): a
        # single head predicting distance-to-goal as a CATEGORICAL over ply-gap
        # bins (chess distance is bounded + integer, so fixed bins have no
        # edge-placement problem, and bimodality -- "3 ply or 30 ply depending
        # on the line" -- IS representable, unlike a Gaussian). The quasimetric
        # d stays the PLANNING distance (axioms preserved); the categorical's
        # ENTROPY rides on top as the auxiliary uncertainty/sharpness signal.
        # Constructed LAST -> distributional=False stays byte-identical.
        self.distributional = distributional
        if distributional:
            self.n_bins = n_bins
            # geometric ply-gap bin edges (ratio ~1.6): [2,3,5,8,13,20,33,52,84,134]
            edges = torch.round(1.6 ** torch.arange(1, n_bins, dtype=torch.float32))
            self.register_buffer("dist_bin_edges", edges)            # (n_bins-1,)
            self.cat_head = nn.Sequential(nn.Linear(2 * d, dh), nn.ReLU(),
                                          nn.Linear(dh, n_bins))
        # COMPETENCE head (2026-07-13, Kaveh's Method 2, training-integrated): a
        # small head predicting the model's OWN per-anchor retrieval error from
        # F(s). Trained jointly against the detached per-example loss, so it
        # learns "where do I fit poorly" for FREE, over the whole embedding
        # space, ALWAYS CURRENT (retrains with the model -- no stale offline kNN
        # map). This is the EPISTEMIC weakness signal (fit-error), complementing
        # Method 1's aleatoric search-disagreement. Constructed LAST ->
        # competence=False stays byte-identical; f is detached at its input so
        # the head can't distort the embedding to make its error predictable.
        self.competence = competence
        if competence:
            self.competence_head = nn.Sequential(nn.Linear(d, dh // 2), nn.ReLU(),
                                                 nn.Linear(dh // 2, 1))
        # OUTCOME POLES (2026-07-13, Kaveh: "add a loss that pushes the poles
        # apart; everything else pushed/pulled by the final side who won -- I
        # need HOPS, not euclidean distance"). Three learnable GOAL-space anchors
        # (loss / draw / win, indexed by result+1). The loss below repels the
        # three poles and, per state, hinges its QUASIMETRIC distance (= hops) so
        # its own outcome-pole is fewer hops away than the others by a margin.
        # This is the outcome-conditioned, self-consistent-by-result separation:
        # mutually-exclusive terminals become far-apart basins, and because it
        # rides on the ply-gap-calibrated d, the within-basin hop gradient
        # survives. Constructed LAST -> outcome_poles=False stays byte-identical.
        self.outcome_poles = outcome_poles
        if outcome_poles:
            self.poles = nn.Parameter(nn.functional.normalize(torch.randn(3, d), dim=1))
        # CONCEPT AXES (2026-07-14, Kaveh's multi-concept architecture): each concept
        # is a learnable unit DIRECTION in the shared space; a state's concept value
        # is the projection F(s)@u. Concept + opposite = the +- ends of ONE axis
        # (exclusive by construction, separated by a margin along it); DIFFERENT
        # concepts live on different axes so their regions overlap freely (a state
        # is a superposition of concepts). Slot 0 = the OUTCOME axis (near-White-mate
        # vs near-Black-mate), trained with a proximity-to-terminal-gated hinge that
        # constrains ONLY this projection -- the other d-1 dims stay free (the
        # pull-to-point pole collapse constrained ALL dims; that was the mistake).
        # Constructed LAST -> concept_axes=0 stays byte-identical.
        self.n_concept_axes = concept_axes
        if concept_axes > 0:
            self.concept_axes = nn.Parameter(
                nn.functional.normalize(torch.randn(concept_axes, d), dim=1))

    def competence_score(self, f: torch.Tensor) -> torch.Tensor:
        """(N,d) F-embeddings -> (N,) predicted per-anchor error = the engine's
        EPISTEMIC unreliability here (softplus, non-negative). Higher = the
        model fits this region poorly = search more."""
        return nn.functional.softplus(self.competence_head(f).squeeze(-1))

    def embed_F(self, planes: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        h = self.encF(planes)
        o = torch.cat([self.emb_we(omega[:, 0]), self.emb_be(omega[:, 1]),
                       self.emb_clk(omega[:, 2])], dim=1)
        e = self.headF(torch.cat([h, o], dim=1))
        # IQE needs FREE coordinate ranges at O(1): L2-normalizing to the unit
        # sphere crushes its interval-union geometry to near-uniform, and the
        # encoder's small-norm init (coord std ~0.08) leaves IQE distances too
        # flat to give InfoNCE a bootstrap gradient (diagnosed 2026-07-17). So
        # for IQE: no normalization, and a fixed scale to coord std ~O(1).
        return self.iqe_embed_scale * e if self.iqe else nn.functional.normalize(e, dim=1)

    def embed_B(self, planes: torch.Tensor) -> torch.Tensor:
        e = self.headB(self.encB(planes))
        return self.iqe_embed_scale * e if self.iqe else nn.functional.normalize(e, dim=1)

    def embed_F_near(self, planes: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        """FAR's shared trunk (encF), NEAR's head. Cosine-scored (near needs
        sharp local ranking, not the quasimetric geometry)."""
        h = self.encF(planes)
        o = torch.cat([self.emb_we(omega[:, 0]), self.emb_be(omega[:, 1]),
                       self.emb_clk(omega[:, 2])], dim=1)
        return nn.functional.normalize(self.headF_near(torch.cat([h, o], dim=1)), dim=1)

    def embed_B_near(self, planes: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.headB_near(self.encB(planes)), dim=1)

    def near_score(self, f_near: torch.Tensor, z_near: torch.Tensor) -> torch.Tensor:
        """(N,d) near-states against ONE near-goal (d,) -> (N,). Plain cosine
        (near is not quasimetric). Counterpart of `score` for the near head."""
        return f_near @ z_near

    # ---- distributional head (option B) ---------------------------------
    def dist_logits(self, f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(N,d) states x goal (d,) or (N,d) -> (N, n_bins) categorical logits
        over ply-gap distance bins. z broadcasts if it's a single goal."""
        if z.dim() == 1:
            z = z.expand(f.shape[0], -1)
        return self.cat_head(torch.cat([f, z], dim=1))

    def dist_entropy(self, f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(N,) Shannon entropy of the predicted distance distribution -- the
        auxiliary uncertainty/sharpness signal (wide = volatile/sharp, narrow =
        smooth). Uses torch.distributions.Categorical (FOSS)."""
        return torch.distributions.Categorical(logits=self.dist_logits(f, z)).entropy()

    def dist_bin_index(self, ply_gap: torch.Tensor) -> torch.Tensor:
        """observed ply-gap -> bin index in [0, n_bins-1] (the cross-entropy
        target). bucketize is right-inclusive on the geometric edges."""
        return torch.bucketize(ply_gap.to(self.dist_bin_edges.dtype), self.dist_bin_edges)

    def two_horizon_loss(self, planes_s: torch.Tensor, omega_s: torch.Tensor,
                         planes_g: torch.Tensor, ply_gap: torch.Tensor,
                         near_max: int = 8, far_min: int = 16, near_weight: float = 1.0,
                         ply_gap_weight: float = 0.05, ply_gap_scale: float = 50.0
                         ) -> tuple[torch.Tensor, torch.Tensor]:
        """Stratified two-horizon objective (TWO_HORIZON_DESIGN.md). Routes
        each (s, g) pair to a head by its real ply-gap:
          near head (cosine InfoNCE)  <- short-gap pairs (gap <= near_max)
          far head  (quasimetric + ply-gap calibration) <- long-gap pairs
                                        (gap >= far_min)
        Pairs in the dead zone (near_max < gap < far_min) are dropped so the
        two heads specialize on cleanly-separated ranges. Losses summed;
        each head's gradient flows only into its own params + the shared
        trunk. Returns (loss, near_top1) -- near_top1 is the short-horizon
        sharpness we must NOT lose (target ~0.97)."""
        gap = ply_gap
        loss = planes_s.new_zeros(())
        near_top1 = planes_s.new_zeros(())
        near_mask = gap <= near_max
        if int(near_mask.sum()) >= 2:
            idx = near_mask.nonzero(as_tuple=False).flatten()
            fN = self.embed_F_near(planes_s[idx], omega_s[idx])
            bN = self.embed_B_near(planes_g[idx])
            logits = (fN @ bN.T) / self.tau
            tgt = torch.arange(len(fN), device=logits.device)
            loss = loss + near_weight * nn.functional.cross_entropy(logits, tgt)
            near_top1 = (logits.argmax(dim=1) == tgt).float().mean()
        far_mask = gap >= far_min
        if int(far_mask.sum()) >= 2:
            idx = far_mask.nonzero(as_tuple=False).flatten()
            fF = self.embed_F(planes_s[idx], omega_s[idx])
            bF = self.embed_B(planes_g[idx])
            logits = self.score_matrix(fF, bF) / self.tau
            tgt = torch.arange(len(fF), device=logits.device)
            far_loss = nn.functional.cross_entropy(logits, tgt)
            d_true = self.distance_matrix(fF, bF).diagonal()
            far_loss = far_loss + ply_gap_weight * nn.functional.mse_loss(
                d_true, gap[idx].to(d_true.dtype) / ply_gap_scale)
            loss = loss + far_loss
        return loss, near_top1

    def loss_fn(self, planes_s: torch.Tensor, omega_s: torch.Tensor,
                planes_g: torch.Tensor, ply_gap: torch.Tensor | None = None,
                material_drop: torch.Tensor | None = None,
                ply_gap_weight: float = 0.05, ply_gap_scale: float = 50.0,
                asym_weight: float = 0.0, asym_margin: float = 0.2,
                asym_cap: int = 128, dist_weight: float = 0.5,
                competence_weight: float = 0.1, result: torch.Tensor | None = None,
                outcome_weight: float = 0.0, pole_tau: float = 1.0,
                pole_margin: float = 3.0, repel_weight: float = 0.0,
                repel_margin: float = 1.0, plies_to_end: torch.Tensor | None = None,
                axis_weight: float = 0.0, axis_margin: float = 1.0,
                axis_gate_plies: float = 8.0,
                horizon_k: float = 0.0) -> tuple[torch.Tensor, torch.Tensor]:
        """InfoNCE with in-batch negatives; returns (loss, top1 retrieval acc).

        horizon_k (>0, in plies; Kaveh 2026-07-16): bound the quasimetric at k
        plies -- calibrate the distance to min(k, ply_gap)/scale instead of
        ply_gap/scale. This is a valid capped quasimetric (min(k, a+b) <=
        min(k,a)+min(k,b), subadditivity preserved) and focuses capacity on the
        near-horizon structure planning actually descends (our retrieval is
        sharp to ~10 plies then cliffs -- so k~=10 makes the measured horizon
        explicit). Positions beyond k become the natural contrast class (they
        all sit at the horizon margin), which is where the hard-negative
        repulsion (nn/hard_negatives) plugs in.

        2026-07-12 ply-gap calibration (Kaveh: "if the future leads to a mate
        for me, that's a good future... maybe we have to search deeper"):
        in-batch retrieval only enforces RELATIVE ranking (is g_true closer
        than the other g's in this batch?) -- nothing calibrates the
        ABSOLUTE scale of the quasimetric distance to anything real, so
        "down material with no path back" and "down material but
        recoverable" can score identically as long as their RANKING within
        a batch happens to work out. `ply_gap` (real anchor->goal ply
        distance, already flowing through the pipeline unused) lets d(f,g)
        regress toward the ACTUAL number of plies real play took to get
        from s to g -- calibrating distance to mean roughly "moves of real
        play between here and there", for winning AND losing trajectories
        alike (losing ones are what teach the geometry of "no way back" --
        which is why the 2026-07-11 winner-POV training filter was removed
        the day this term landed), not just an uncalibrated relative score. Only meaningful in quasimetric mode (there is no
        `d` to calibrate otherwise); silently ignored when quasimetric=False
        so non-quasimetric callers don't need to change."""
        f = self.embed_F(planes_s, omega_s)
        b = self.embed_B(planes_g)
        logits = self.score_matrix(f, b) / self.tau
        target = torch.arange(len(f), device=logits.device)
        per_row = nn.functional.cross_entropy(logits, target, reduction="none")
        loss = per_row.mean()
        top1 = (logits.argmax(dim=1) == target).float().mean()
        if self.competence:
            # predict each anchor's OWN retrieval error from F(s) (detached, so
            # the head can't distort the embedding). At inference competence_score
            # is the epistemic "where do I fit poorly" signal (Method 2, native).
            pred_err = self.competence_score(f.detach())
            loss = loss + competence_weight * nn.functional.mse_loss(pred_err, per_row.detach())
        if self.quasimetric and ply_gap is not None:
            d_true = self.distance_matrix(f, b).diagonal()
            gap = ply_gap.to(d_true.dtype)
            if horizon_k > 0:
                gap = gap.clamp(max=horizon_k)          # capped (horizoned) quasimetric
            target_d = gap / ply_gap_scale
            loss = loss + ply_gap_weight * nn.functional.mse_loss(d_true, target_d)
        if self.distributional and ply_gap is not None:
            # categorical head: predict which ply-gap BIN the true goal falls in
            # (cross-entropy). Trains the distribution whose entropy is the
            # uncertainty signal; the quasimetric d above stays the distance.
            logits_bin = self.dist_logits(f, b)                     # (N, n_bins)
            target_bin = self.dist_bin_index(ply_gap)
            loss = loss + dist_weight * nn.functional.cross_entropy(logits_bin, target_bin)
        if self.quasimetric and asym_weight > 0 and material_drop is not None \
                and bool(material_drop.any()):
            # ASYMMETRY MARGIN (2026-07-13, JOURNAL.md): pairs whose material
            # strictly DECREASED anchor->goal crossed a capture, so the
            # reverse hop is impossible in real chess ("you can't un-capture
            # the rook"). Hinge the reverse distance d(F(g), B(s)) to exceed
            # the forward d(F(s), B(g)) by a margin -- derived purely from
            # trajectory direction in the data, no chess rules coded in.
            # (The fitness probe measured frac(reverse <= forward) at
            # 0.27-0.35 with the arrow of material barely learned -- want ~0.)
            # Capped subset: the reverse side needs its OWN extra F/B
            # forwards, so this term costs a second pass on <= asym_cap rows.
            rows = torch.nonzero(material_drop, as_tuple=False).flatten()[:asym_cap]
            f_rev = self.embed_F(planes_g[rows], omega_s[rows])   # goal as state
            b_rev = self.embed_B(planes_s[rows])                  # anchor as goal
            d_fwd = self.distance_matrix(f[rows], b[rows]).diagonal()
            d_rev = self.distance_matrix(f_rev, b_rev).diagonal()
            loss = loss + asym_weight * nn.functional.relu(
                asym_margin + d_fwd - d_rev).mean()
        if self.outcome_poles and outcome_weight > 0 and result is not None:
            # (a) push the three terminal poles apart; (b) SOFTLY sort each state
            # toward its own-outcome pole in HOPS (quasimetric d). result in
            # {-1 loss, 0 draw, +1 win} -> pole index {0,1,2}. The pull is a
            # temperature cross-entropy over -hops (a soft classifier), NOT a hard
            # margin: its gradient vanishes once the right pole is comfortably
            # nearest, so well-placed wins keep their internal DTM/hop gradient
            # instead of being crushed onto the pole (the hard-hinge failure mode).
            poles = nn.functional.normalize(self.poles, dim=1)        # (3, d) goal-space
            d_sp = self.distance_matrix(f, poles)                     # (N,3) state->pole hops
            idx = (result + 1).long().clamp(0, 2)
            pull = nn.functional.cross_entropy(-d_sp / pole_tau, idx)
            pp = torch.cdist(poles * self.metric_scale, poles * self.metric_scale)  # (3,3)
            repel = (nn.functional.relu(pole_margin - pp[0, 1])
                     + nn.functional.relu(pole_margin - pp[0, 2])
                     + nn.functional.relu(pole_margin - pp[1, 2])) / 3.0
            loss = loss + outcome_weight * (pull + repel)
        if repel_weight > 0 and result is not None and self.quasimetric:
            # CROSS-OUTCOME REPULSION (t-SNE-style, no attractor point): repel
            # anchor pairs with DIFFERENT final outcomes in HOPS up to a margin,
            # then stop (relu = the bounded/saturating role t-SNE's Student-t tail
            # plays). Within-outcome pairs are left to the reach/ply-gap attraction
            # -- so regions survive as extended blobs (their internal hop gradient
            # intact) while mutually-exclusive regions push apart. d(F(s_i),B(s_j))
            # = directed hops from anchor i to anchor j (B on anchors = extra pass).
            ab = self.embed_B(planes_s)
            dss = self.distance_matrix(f, ab)                    # (N,N) directed hops
            diff = (result[:, None] != result[None, :]).float()  # cross-outcome pairs
            denom = diff.sum().clamp_min(1.0)
            loss = loss + repel_weight * (nn.functional.relu(repel_margin - dss) * diff).sum() / denom
        if self.n_concept_axes > 0 and axis_weight > 0 and result is not None \
                and plies_to_end is not None:
            # OUTCOME CONCEPT AXIS (slot 0): hinge the projection F(s)@u to be
            # > +margin for states NEAR a White-win terminal and < -margin near a
            # Black-win one, gated by exp(-plies_to_end/gate) so only near-terminal
            # (~forced-region) states are pulled and the pull fades smoothly with
            # distance. Constrains ONE direction of d; everything orthogonal stays
            # free for other concepts (Kaveh: concept+opposite exclusive along the
            # axis, regions of different concepts may overlap).
            u = nn.functional.normalize(self.concept_axes[0], dim=0)
            proj = f @ u
            gate = torch.exp(-plies_to_end / axis_gate_plies) * (result != 0).float()
            loss = loss + axis_weight * (
                gate * nn.functional.relu(axis_margin - result * proj)).mean()
        return loss, top1

    def distance_matrix(self, f: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """All-pairs metric distance, (N,d) x (M,d) -> (N,M). Requires
        quasimetric=True (the metric only exists in that mode). Exposed
        separately from score_matrix so the triangle inequality can be
        tested directly on `d` alone -- `r` in score_matrix is an
        unconstrained residual that is NOT expected to respect it."""
        assert self.quasimetric, "distance_matrix requires quasimetric=True"
        if self.iqe:
            return self.iqe_head.pairwise(f, b)
        fs, bs = f * self.metric_scale, b * self.metric_scale
        d2 = ((fs * fs).sum(1, keepdim=True) + (bs * bs).sum(1)[None, :]
              - 2.0 * (fs @ bs.T))
        return torch.sqrt(d2.clamp_min(1e-9))

    def directed_distance(self, f: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Row-wise directed distance d(f_i -> b_i), (N,d)x(N,d) -> (N,). The
        per-pair counterpart of distance_matrix's diagonal (IQE computes it
        directly; MRN falls back to the matrix diagonal)."""
        assert self.quasimetric, "directed_distance requires quasimetric=True"
        if self.iqe:
            return self.iqe_head(f, b)
        return self.distance_matrix(f, b).diagonal()

    def qrl_loss(self, planes_s: torch.Tensor, omega_s: torch.Tensor,
                 planes_succ: torch.Tensor, planes_g: torch.Tensor,
                 valid: torch.Tensor, *, push_offset: float = 40.0,
                 softplus_beta: float = 0.1, step_cost: float = 1.0,
                 epsilon: float = 0.25):
        """Quasimetric-RL objective (Wang, Torralba, Isola, Zhang, ICML 2023),
        the loss IQE was DESIGNED for -- InfoNCE only enforces relative ranking
        and leaves the interval geometry collapsed ("d could remain arbitrarily
        small everywhere", their words; our exact plateau symptom).

        Two terms, no contrastive softmax:
          * LOCAL CONSTRAINT -- on real 1-ply transitions s->s', pin the
            best-case distance to the unit step: penalize d(s->s') > step_cost
            with a squared-hinge, dual-ascended by lambda toward tolerance eps.
            Multi-step (incl. long FORCED lines) distances then self-assemble by
            chaining unit steps through the triangle inequality -- NEVER capped.
          * GLOBAL PUSH -- on RANDOM (independent) state/goal pairs, spread
            distances toward push_offset via softplus(offset - d). offset is set
            WELL beyond the longest forcing line we care about (default 40 plies
            ~ 20 moves) so a genuinely-reachable long line (built by chaining to
            ~its true ply length) stays CLOSER than the unreachable random pairs
            -- the push is a saturating prior, not a horizon (Kaveh's constraint:
            never let a reachable long forced line look far)."""
        f_s = self.embed_F(planes_s, omega_s)
        b_succ = self.embed_B(planes_succ)
        b_g = self.embed_B(planes_g)
        # local: d(s -> s') <= step_cost on observed transitions
        d_step = self.directed_distance(f_s, b_succ)
        if valid.any():
            d_step_v = d_step[valid]
        else:
            d_step_v = d_step
        sq_dev = (d_step_v - step_cost).clamp_min(0.0).square().mean()
        lam = F.softplus(self.qrl_raw_lambda)
        constraint = (sq_dev - epsilon ** 2) * grad_reverse(lam)
        # global push: independent random state/goal pairs -> spread apart
        perm = torch.randperm(f_s.shape[0], device=f_s.device)
        d_rand = self.directed_distance(f_s, b_g[perm])
        push = F.softplus(push_offset - d_rand, beta=softplus_beta).mean()
        loss = push + constraint
        stats = {"push": float(push), "sq_dev": float(sq_dev), "lam": float(lam),
                 "d_step": float(d_step_v.mean()), "d_rand": float(d_rand.mean())}
        return loss, stats

    def score_matrix(self, f: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """All-pairs score, (N,d) x (M,d) -> (N,M). Plain dot product unless
        quasimetric=True, in which case -d(f,b)+r(f,b) (see module docstring)."""
        if not self.quasimetric:
            return f @ b.T
        if self.iqe:
            return -self.distance_matrix(f, b)          # IQE is already asymmetric
        return f @ self.W @ b.T - self.distance_matrix(f, b)

    def score(self, f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """(N,d) states against ONE fixed goal embedding z (d,) -> (N,).
        The search/readout-time counterpart of score_matrix -- used wherever
        the old code did `F(s) @ z` (a fixed zgoal like zMATE_W)."""
        if not self.quasimetric:
            return f @ z
        return self.score_matrix(f, z[None, :])[:, 0]

    @torch.no_grad()
    def np_score_matrix(self, F: "np.ndarray", B: "np.ndarray") -> "np.ndarray":
        """numpy adapter for score_matrix, shaped for planner/decompose.py's
        score_pairs hook: (n,d) x (m,d) float32 -> (n,m) float32. Exactly
        the dot product when quasimetric=False, so callers can pass it
        unconditionally."""
        device = next(self.parameters()).device
        f = torch.as_tensor(F, dtype=torch.float32, device=device)
        b = torch.as_tensor(B, dtype=torch.float32, device=device)
        return self.score_matrix(f, b).cpu().numpy()

    @staticmethod
    def reach_z(f: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return f @ z


def save_ckpt(fb: TorchFB, path, step: int = 0, opt: torch.optim.Optimizer | None = None,
              zgoals: dict | None = None, provenance: dict | None = None) -> None:
    """zgoals: name -> (d,) numpy/tensor goal vectors (e.g. MATE_W) travel
    with the model -- a field without its goals is not a planner artifact.
    provenance: catspace.audit.build_provenance(...) output, if the caller
    has one -- lets audit_checkpoint() certify this checkpoint never saw
    Stockfish-derived signal without re-deriving it from training logs."""
    payload = dict(state_dict=fb.state_dict(), config=fb.config, step=step,
                   zgoals={k: torch.as_tensor(v).cpu() for k, v in (zgoals or {}).items()},
                   provenance=provenance)
    if opt is not None:
        payload["opt_state"] = opt.state_dict()
    path = Path(path)                      # atomic: an interrupted save must not
    tmp = path.with_suffix(path.suffix + ".tmp")   # corrupt the previous checkpoint
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_ckpt(path, device: str = "cpu") -> tuple[TorchFB, dict]:
    """Returns (model, payload). payload keeps step/opt_state/zgoals.

    Input-plane growth compatibility: checkpoints trained before the
    repetition plane (N_PLANES 19 -> 20, 2026-07-15) have stem convs with 19
    in-channels; the new plane's weights are zero-padded, so the loaded model
    is bit-identical to the old one on rep=0 inputs (and inert on the new
    plane until trained)."""
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    fb = TorchFB(**payload["config"])
    state = payload["state_dict"]
    for k, ref in fb.state_dict().items():
        if (k in state and state[k].dim() == 4 and state[k].shape[1] < ref.shape[1]
                and state[k].shape[0] == ref.shape[0] and state[k].shape[2:] == ref.shape[2:]):
            pad = torch.zeros(ref.shape[0], ref.shape[1] - state[k].shape[1],
                              *ref.shape[2:], dtype=state[k].dtype, device=state[k].device)
            state[k] = torch.cat([state[k], pad], dim=1)
    fb.load_state_dict(state)
    fb.to(device)
    return fb, payload


def pick_device(arg: str = "auto") -> str:
    if arg != "auto":
        return arg
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
