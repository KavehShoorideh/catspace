# Architecture Review — the full arrangement, verified against the literature

2026-07-17 (Fable). A top-to-bottom review of the catspace architecture as it
stands after this week's redesign, and a literature sweep checking each pillar
for known failure modes. Verdict up front: **the arrangement is sound, and its
soundness has a single organizing principle — one scalar field per question.**
Every failure we hit this week was one layer doing another layer's job; every
fix restored the separation.

## 0. The arrangement (one screen)

```
GEOMETRY   d_optimal(s,g)   IQE quasimetric, QRL-trained        min-sum, cooperative
VALUE      P(outcome|s)     committor phead (W/D/L surfaces)    measure-dependent, adversarial via minimax
TRUST      γ = e^{-k(1-P)}  coherence: how deep to believe d    derived from P, not from move-count
COMPOSE    minimax MCTS     turns cooperative d into forced v   the adversarial step no metric can do
DECIDE     planner (post-MVP)  internal {probe_region, set_plan} + game {make_move, offer_draw, resign}
DEFERRED   ω-conditioning   fallibility = complexity × skill    enters through P, nowhere else
```

Layer-separation is the thesis. The week's bugs, reread as violations of it:
phead@1.0 (value invading geometry → outcome-cluster collapse); DRAW_V=−0.999
(draw-avoidance *policy* invading the value *scale* → minimax asymmetry);
entropy-coherence (a complexity proxy standing in for probability → proven
mates discounted); the early horizon-cap (an executor concern injected into the
metric → long forcing lines blinded). All four fixes moved information back to
its home layer.

## 1. Geometry: IQE + QRL — SOUND, two caveats

**Literature.** The known limitation of QRL is exactly the one we identified
independently: its optimality guarantee holds for *deterministic* goal-reaching,
and in stochastic settings "distance = value" breaks (prior temporal distances
there even lose the triangle inequality). [TMD (arXiv 2509.20478)](https://arxiv.org/abs/2509.20478)
is built on this diagnosis and beats QRL ~3× on stochastic mazes.
**Why we're safe:** chess dynamics are deterministic given both players' moves —
our "stochasticity" is the opponent, and we *never read d as a value*. d is
geometry only; the adversarial value comes from minimax + committor. This is
precisely the layering the limitation demands.

**Our two-sided deviation is principled, not a hack.** QRL's one-sided
`d(s,s′) ≤ cost` is needed in general MDPs where an observed transition may be
dominated by a cheaper path. In a *unit-cost game graph* the true shortest-path
distance between distinct adjacent positions is exactly 1 (the observed move is
a witness; nothing shorter exists). So pinning `d(s,s′)=1` two-sided is the
exact constraint for our domain — and it is what forbids the ordering-collapse
attractor (all-F-above-all-B) that one-sided permits (observed: the 19k-step
late collapse).

**Caveat 1 — offset scale (MEDIUM, open).** The push lifts distances toward
offset=15 and saturates there; chained upper bounds can exceed 15 but nothing
*lifts* the 15–60 range. True reachable distances for long conversions
(opening → 6-piece endgame needs ≥ ~26 captures' worth of plies) exceed 15, so
long-range geometry may compress. Action: after stability is proven, sweep
offset {15, 30, 60} **under the final recipe** (two-sided + pool + PID +
phead 0.1 — the 40/128 collapses were all under the *old* recipe and say
nothing about large offsets now). Judge by spread + conversion.

**Caveat 2 — the human-play manifold.** The metric is trained on human-game
transitions, so d measures reachability *through human-plausible play*, not the
raw game graph. For planning realism that is arguably a feature; where it
matters, probes supply ground truth (§5).

**Contingency:** if QRL-IQE conversion disappoints at eval, TMD is the
literature's drop-in alternative trainer for exactly our setting (offline,
suboptimal data, quasimetric head, stop-grad + Bregman consistency) — a
shelf-ready pivot, not a redesign.

## 2. Value: the committor — SOUND, one required new check

**Literature.** The committor is the central object of transition path theory,
and neural committor estimation is an active field:
[deep-learning committors](https://www.researchgate.net/publication/335012771_Computing_committor_functions_for_the_study_of_rare_events_using_deep_learning),
[deep adaptive sampling on rare transition paths (2025)](https://arxiv.org/abs/2501.15522),
[committor-consistent transition pathways (Nature Comp. Sci. 2025)](https://www.nature.com/articles/s43588-025-00828-3).
Our phead trained on human outcomes is an **empirical committor under the
human-play measure** — the correct object for fallible play, with ω-conditioning
(player strength) as its natural refinement. Kaveh's formulation — coherence =
P(realize), degraded by "likelihood of our mistaking the position = complexity ×
our mistakes" — is exactly a measure-dependent committor.

**The rare-event problem maps onto us.** TPT's hard case is sampling rare
transitions; our phead is weakest exactly where games are rare (odd endgames).
The memory/tree-caching plan (probes sharpen the committor where we actually
searched) is the same medicine as adaptive sampling — search generates data
precisely in the undersampled regions.

**NEW REQUIRED CHECK — calibration (HIGH).** P now feeds three consumers:
coherence γ, the obvious-region soft-terminal, and (post-MVP) resign/draw. An
overconfident phead poisons all three *silently*. Calibration (reliability
curves on held-out outcomes; ECE) must become a standing health gate next to
effective rank. Until measured, treat `certainty_stop` conservatively.

## 3. Trust: coherence — SOUND, with a pleasing identity

The compounding backup discount is ∏ᵢ e^{−k(1−Pᵢ)} = e^{−k·Σ(1−Pᵢ)} ≈
(∏Pᵢ)^k for P near 1 — i.e. **the backed-up value is discounted by
approximately P(the whole line is realized)^k**. That is the same path-integral
that defines d_certainty = −ln P. Coherence, the certainty distance, and the
recognizer are one object (the committor) viewed at three ranges. Internal
consistency, not coincidence.

The literature's warning about state-dependent discounting
(AdaGamma's TD-error collapse) doesn't apply: our γ lives only in the *search
backup*, never in a TD bootstrap target, so the destabilizing feedback loop is
absent by construction.

**Value vs. the bridge — they are NOT the same layer** (Kaveh's question,
2026-07-17). Both derive from the committor, but value is **P evaluated at a
point** ("how good is the destination?" — a property of the *state*), while
coherence is **how P decays along a path** ("how far ahead is the map
trustworthy?" — a property of the *projection*). Two positions can share
P(win)=0.95 with opposite coherence: a KR-vs-K mate-in-12 (every defense stays
in the funnel — trust the field 12 deep, stop searching) vs. a sharp middlegame
win needing an only-move at ply 3 against four distinct defenses (trust ~3
plies, spend the whole budget there). Value cannot tell these apart; coherence
is exactly the number that can. And they are independent axes — a forced
perpetual has LOW value but MAXIMAL coherence. The planner's decision logic in
miniature:

| | high coherence | low coherence |
|---|---|---|
| **high P** | won & forced → bank it, stop (soft-terminal) | winning if navigated → **search here** |
| **low P** | dead draw/lost → offer draw / resign | murky → probe other regions |

Consumers differ accordingly: value feeds the leaf readout and outcome
decisions (reach, soft-terminal, resign/draw); coherence feeds the backup
discount and budget allocation. Amplitude vs. correlation length of one field.

## 4. Executor: minimax MCTS — SOUND, standard, one adopted practice

The soft-terminal is value-based truncation — what AlphaZero-family engines do
at every leaf; ours merely formalizes *stopping* at high confidence. For the
post-MVP resign/draw layer, adopt the documented
[AlphaGo mechanism](https://augmentingcognition.com/assets/Silver2017a.pdf):
resign threshold chosen to keep the false-positive rate **< 5%**, measured by
**disabling resignation in 10% of games**. This is literature-standard, cheap,
and exactly our e-value culture applied to resignation.

## 5. Decision: planner-as-prober — SOUND, with three anchors

1. **Rational metareasoning.** Kaveh's internal/game action split is
   [Russell & Wefald's computations-vs-external-actions formalism](https://people.eecs.berkeley.edu/~russell/research-bo.html),
   and probe selection by value-of-information is
   [Hay & Russell's metareasoning for MCTS](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2011/EECS-2011-119.pdf)
   / [Selecting Computations (UAI-12)](https://auai.org/uai2012/papers/123.pdf).
   Notably they prove the bandit/UCT frame is *wrong* for selecting
   computations — supporting our plan to *learn* the meta-policy (phase E)
   rather than hand-UCT over probes.
2. **SoRB.** [Search on the Replay Buffer](https://arxiv.org/abs/1906.05253) is
   the closest relative (planning over learned distances). Its documented
   fragility — distance overestimation / hallucinated shortcuts, patched with
   ensembles of distributional value functions — is exactly the failure our
   **probe-before-commit** removes: the planner never trusts the metric where a
   bounded minimax probe can check it. (Ensemble/distributional distance heads
   remain a cheap later upgrade.)
3. **Reachability games.** "Can I force this region" is formally **membership
   in the attractor set** of a two-player reachability game — computed by
   backward induction, [solvable in polynomial time](https://www.labri.fr/perso/anca/Games/graphgames.pdf),
   and *not* a shortest-path object. This is the theorem-shaped reason the
   quasimetric alone can never answer forceability (min-sum ≠ min-max
   fixpoint), and why the probe (a sampled, anytime attractor check) is a
   necessary primitive, not an implementation convenience. The committor≈1
   region under optimal play *is* the attractor.

## 6. Deferred: fallibility — aligned with the literature

Skill-conditioned human modeling is established
([Maia](https://arxiv.org/abs/2006.01855)-line); our plan (ω enters through the
committor's measure, nowhere else) keeps it a one-layer change. Under fallible
play the committor becomes graded and coherence shortens — the architecture
absorbs the whole extension through P.

## 7. Consolidated risk register

| # | Risk | Sev | Action |
|---|------|-----|--------|
| 1 | Phead calibration unmeasured; γ, soft-terminal, resign all consume P | HIGH | Reliability/ECE as standing gate; conservative certainty_stop until measured |
| 2 | Offset=15 may compress long-range geometry | MED | Sweep {15,30,60} under final recipe, judge by spread + conversion |
| 3 | Soft-terminal committor-value vs squashed-reach scale mismatch in selection | LOW | Reconcile scales (open review item c) |
| 4 | Metric ≠ raw game graph (human-play manifold) | LOW | Acknowledged; probes ground-truth it |
| 5 | QRL-IQE conversion could still disappoint | — | TMD is the shelf-ready alternative trainer |
| 6 | d misread as value by future code | — | Guarded by convention + this doc; keep readouts committor-based |

## 8. Bottom line

The arrangement is **sound and now literature-anchored at every level**: the
geometry sits inside QRL's validity domain *because* we never ask it for
adversarial values; the deviation we made (two-sided) is provably right for
unit-cost game graphs; the value layer is a committor with an active estimation
literature; the trust layer is a probability path-integral consistent with the
certainty distance; the executor's stopping rule and the resign design follow
engine-standard calibrated practice; and the planner's internal/game action
split independently rederived the rational-metareasoning formalism, with
attractor theory supplying the formal object probes compute. The one thing the
literature demands that we don't yet do is **calibrate the committor** — that
is the top new action item.
