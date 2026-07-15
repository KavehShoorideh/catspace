# The research journey: every hypothesis we broke on the way here

*Companion to [the state of the research](state_of_the_research.md). That
article is what survived; this one is the graveyard — disproven and
inconclusive hypotheses, in the order we believed them. Implementation bugs
are out of scope; these are ideas that were tested fairly and lost. Numbers
are verbatim from journaled script verdicts; the comparison machinery (paired
deterministic playouts, bootstrap CIs, anytime-valid e-values, pre-registered
confirmatories) and the claim→data→command reproduction map are described in
the companion article's closing sections and in the repo README's
"Reproducing the journaled results".*

The single most repeated event in this project — it happened at least seven
distinct times — is the **structure–play dissociation**: an intervention
improves a representational metric (calibration ρ, separability, curvature,
retrieval) and play does not move, or moves backward. If you take one thing
from this document, take that: in a planning system, the only arbiter is play
under the harness you'll actually deploy, and everything else is a compass at
best.

---

## Era 0 — the toy cone (5×5 boards)

The founding formulation: embed each state by its *future* — the discounted
successor measure ("the cone") factorized as F(s)·B(g) — and read concepts out
of the embedding. On exact 5×5 endgames this produced a genuinely strong
engine against random play (95.5% mate rate, 26× the random baseline).

**What broke:** the legibility gates. Rank-64 global reach error came in at
0.41 against a <5% gate, and only 1 of 3 hoped-for concept dimensions
correlated with ground truth. The lesson that survived is that per-dimension
concept alignment was the wrong operationalization (SVD dimensions are
rotation-arbitrary) — and, more usefully, that greedy planning only consumes
*local move ranking*, so global reach fidelity was never the load-bearing
quantity. An oracle-defined "DTM≤3 region" goal was also ablated away early:
the plain mate goal was never worse. Only when the toy engine was tested
against *optimal* defense did the deep result appear: the cone is
**opponent-conditioned** — an embedding trained on random play collapsed
against optimal play (recovered from 45% to 86% with a policy-iteration
opponent curriculum). Opponent-conditioning (ω) has been in every architecture
since.

## Era 1 — imitation + more training is enough (full board)

First real-data era: train the FB field on Lichess games, read it out
greedily, expect improvement with scale. Four consecutive rounds pulled
training levers — a learning-rate schedule, 5× more data, 2× more steps —
and arena score against the weakest Stockfish sat at **0.075 → 0.087 → 0.100 →
0.100**. Embedding-quality proxies improved; the metric that matters didn't.

**What broke:** the assumption that the bottleneck was in the weights. One
inference-time change — 3-ply beam search over the *same* embedding — jumped
the score to 0.250 (confirmed, e ≈ 1.2e4). Depth-4 then *regressed* to 0.200
(narrowing the beam to afford the ply cost more than the ply gained), closing
the "just search deeper" thread with a measured non-monotonicity that would
later matter (Era 5). Pure imitation has a documented ceiling; the readout was
the first-order term all along.

## Era 2 — the diagnostic endgame, and tactical blindness

Full-board losses were undiagnosable, so the project narrowed to KRRvKBP —
tablebase-verified won positions, one concept to learn ("keep the rooks where
the bishop can't touch them"). The field promptly hung a rook on move one, and
the ACPL probe generalized the finding: the policy blundered on the majority
of ordinary positions too. Root cause, confirmed by reading the training path:
the FB objective was **outcome-agnostic** — it predicted "which state came
later in this game" identically for wins and losses.

**Two hypotheses tested, one kept, one retired with honors:**

- *Winner-POV filtering* (train only on the winner's anchors): confirmed real
  (−22 cp vs a step-matched control, p = 0.0046) — the cheapest possible proof
  that outcome-conditioning matters — then deliberately **retired** once real
  mechanisms existed, because censoring losing trajectories deletes exactly
  the signal a calibrated distance needs ("down material with no way back"
  must be learnable as *far*).
- *Quasimetric architecture* (score = f·W·g − d(f,g) with d a true metric):
  kept; the optimal goal-conditioned value function is provably a quasimetric,
  and the structural axioms have held in every checkpoint since.

## Era 3 — the single-embedding compromise (rounds 13–18)

A long ladder of auxiliary-loss and data levers, almost all of which lost to
the incumbent at play. The pattern that emerged — short-horizon tactical
sharpness and long-horizon calibration **compete inside one embedding** — was
the era's real finding.

- **Self-play mix at 0.3–0.4 fraction:** dragged play for five straight rounds
  (its ε-noise games dulled short-horizon tactics). Found by an attribution
  ablation after three recipe changes had been made at once — the attribution
  debt was a knowing methodology cost, and it bit.
- **Endgame-start curriculum:** improved the underlying metric geometry
  (nearest-exemplar calibration +0.165 → +0.252), was *invisible* through the
  centroid readout the pre-registered gate had (wrongly) been defined
  against, and did not transfer to play. Dose-response was non-monotonic —
  a higher curriculum fraction scored worse.
- **Asymmetry hinge** ("you can't un-capture a rook"): taught the
  arrow-of-material almost for free (reverse-cheaper-than-forward errors 27% →
  3%) and moved the long-horizon retrieval tail outward — the only lever that
  ever did — but taxed k=1 retrieval (0.97 → 0.79/0.85), and short-horizon
  discrimination is what endgame play lives on: 19 losses from 60 won
  positions. Closed after two attempts; earmarked for re-entry only after the
  short horizon has other support.
- **Goal-as-region readout** (nearest/soft-min over a bank of real mate
  exemplars): the diagnosis behind it was correct — *averaging mate exemplars
  into any centroid destroys distance structure* (centroid calibration is
  flat; nearest-exemplar correlates) — but at play the bank lost decisively,
  three times (e.g. 0.433 vs 0.308, e = 65). Max/soft-min over heterogeneous
  exemplars injects goal-switching noise into move ranking; the centroid's
  blur is exactly what makes it a *stable* gradient. Positional calibration
  and move-ranking stability are different fitness axes.
- **The instrument itself dissociated:** the checkpoint with the best
  structural calibration of the era played 0.12 below the incumbent
  ([figure](figures/fig_proxy_vs_play.png)). The round-18 ablation
  (quasimetric + ply-gap, human data only) was promoted on *play*, and its
  calibration happened to be fine — but calibration never predicted the
  ordering.

## Era 4 — the sharpness detours

Kaveh's reframe: the tactical/positional split is not *depth*, it's local
**sharpness** of the value landscape. The reframe survived; every attempt to
detect sharpness statically died.

- **Two-horizon (ply-keyed near/far heads):** specialized exactly as designed
  at the representation level (near k=1 retrieval 0.98; far calibration
  +0.272, best of any checkpoint) — and the axis was still wrong, because a
  forcing line runs 20 plies deep while a quiet position is quiet at ply 2.
  Superseded, kept as a baseline.
- **Head-disagreement as a sharpness sensor:** rejected — ρ +0.079 vs the
  +0.202 of a plain score-spread baseline. The heads disagree about their
  *training distributions*, not about curvature.
- **Categorical distributional head:** its entropy came out *negatively*
  correlated with sharpness (−0.16 to −0.23 across three readouts); killed at
  the 15k-step short run by the pre-registered gate.
- **Then the ruler itself failed:** the tablebase sharpness benchmark was
  distance-confounded (ρ +0.39 with distance-to-mate). On a deconfounded
  criticality measure, **no static signal we emit detects sharpness** —
  and structural "tactical density" signals *anti*-correlate with endgame
  sharpness, because endgame sharpness is quiet precision, not melee.
- **Resolution:** sharpness is an invented concept whose only job is to
  allocate search — so define it self-referentially (where does deeper search
  change the model's mind?) and validate by play, not against a label. The
  label was retired as arbiter. Later, the multi-ε identification (main
  article, Finding 1) finally measured a real, plies-orthogonal sharpness —
  from rollout statistics, not from a static head.

## Era 5 — search-allocation and structure levers that didn't cash

- **Reliability-gated search alone:** both sensors worked (the kNN competence
  field generalized held-out at ρ +0.31; the shallow-vs-deep disagreement
  signal flags known-hard positions 0.24 vs 0.04), and gating still tied
  uniform search at matched compute (0.583 vs 0.600). Two honest reasons: the
  toy's difficulty is homogeneous (targeting has nothing to exploit), and
  concentrating search where the model is weak just concentrates *inert*
  search — deeper lookahead over a flat field is still flat. The signal's
  value is as the allocator of a closed data loop, not as a standalone gate.
- **W/D/L regions don't exist in the field** — and balanced outcome data
  doesn't create them (silhouette ≈ 0 before and after; win and loss
  intermixed even 4 plies from forced mate). Temporal attraction overwrites
  outcome structure. This is the deepest single diagnosis of why play
  plateaued: reach can't separate good from bad moves if wins and losses
  share neighborhoods.
- **Outcome poles:** a hard pull-to-pole loss *did* separate outcomes in hops
  (balanced accuracy 0.84) — and crushed play (0.54 → 0.30) by collapsing the
  win region's internal gradient. Gentle variants preserved play and the
  separation, and still only ever **tied** the incumbent. Restructuring
  geometry ≠ better moves.
- **The phantom V6:** an n=60 point estimate showed the gentle-pole variant
  +0.058 over the incumbent; the powered, paired, CI-carrying re-run said
  −0.005, e = 0.09. No variant was ever promoted on a point estimate again.
- **The era's exit was a measurement, not a variant:** the same weights
  converted 0.175 at 200 evals and 0.325 at 800. The whole variant ladder had
  been A/B'd in the **search-limited regime, where the embedding cannot
  matter** — every tie was uninformative by construction. Re-tested at
  saturation, the variants still tied, which finally made the negative
  conclusive rather than mis-measured. This reversed Era 1's own
  "node count is not a lever" conclusion — both were true *in their regimes*,
  and neither claim means anything without the regime attached.

## Era 6 — the certainty era's own casualties

Even the winning line shed hypotheses on the way:

- **The first certainty distill was null at play** (+0.142 field calibration,
  −0.025 play) — under-dosed data and the wrong regime, exactly the two
  factors later fixed by the scaling curve and the 800n ladder.
- **K=16's first significant look (+0.167) failed its confirmatory** (+0.050,
  ns) — a textbook winner's curse across four sequential looks, caught by the
  frozen-set protocol, one week before the same protocol *confirmed* the
  800n effect (+0.208). The protocol cuts both ways, which is why it's
  trustworthy.
- **More steps made play worse:** extending the promoted full-board run 155k →
  215k improved validation metrics and lost head-to-head at play (composed
  e ≈ 48 against). Retrieval is not planner quality; schedules overcook.
- **The closed loop's second lap shrank:** the loop compounds *data* quality
  strongly (own-play P̂ mean 0.14 → 0.34; within-won gradient +0.53 → +0.65)
  but its play return at round 2 was ~+0.08, thrice-repeated yet below the
  confirmatory's resolution. Most of the toy-distillable signal had already
  been banked into the base objective.
- **Two-channel v1 falsified** (main article, Finding 1): purifying the
  geometry to plies and re-adding risk at readout through a frozen sharpness
  probe cost ~0.2 conversion, CI-real at both deep rungs. The decomposition
  *finding* stands; that implementation died.

## The cross-cutting lessons

1. **Play is the only arbiter.** Representation metrics steered us well and
   ranked checkpoints wrong. Validate steering signals by whether using them
   improves play at matched compute.
2. **Name your regime.** "X doesn't matter" and "X is the bottleneck" were
   both measured for search depth — in different regimes. Every A/B has a
   budget attached; conclusions don't travel without it.
3. **Pre-register, freeze, confirm — and let it cut both ways.** The same
   protocol that killed our favorite result confirmed the real one.
4. **Point estimates on n=60 games are noise.** Paired designs, deterministic
   opponents, CIs, and anytime-valid e-values are what let sequential
   experimentation stay honest.
5. **Retire proxies once mechanisms exist.** Winner-POV proved
   outcome-conditioning and then was removed; the sharpness label proved the
   reframe and was retired as arbiter. A proxy that overstays becomes a
   confounder.
6. **Coverage is destiny for learned fields.** Every "the objective can't
   learn X" claim eventually decomposed into "no gradient ever reached X."
   Own-play data in the blind region was the only thing that ever put
   curvature there — and *accurate* curvature needed dense, correct outcome
   signal, not just any data.
