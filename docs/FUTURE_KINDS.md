# Future kinds: subgoals without a quasimetric

Kaveh, 2026-08-24: *"I'm not sure quasimetric is the best tool for what I have in mind.
What I'm trying to predict is the kinds of futures available to me if I take a
particular move. If I trade the queens, what kind of endgame do I have? ... I want
concept states that kind of pool together. So what are some subgoals I can define, and
how do I define them if I don't use a quasimetric?"*

And the constraint that shapes the answer: *"not always will future concepts be
predictable from present concepts, so I expect that loss to remain high. The primary
objective should always be the Stockfish evaluation, but we do want a nudge -- if some
concepts can be predicted, predict those."*

Every claim below is measured in JOURNAL.md 2026-08-24 21:59 and reproducible with
`catspace/research/tools/training_infra/future_concepts.py`.

## What the measurement settled

1. **Concepts do pool by kind of future.** Queens-off futures separate at grouped
   holdout AUC **0.958** (null 0.508) from the discounted future occupancy, vs 0.678
   from the reached position alone. The doubt is resolved: the pooling is there.
2. **A mean cannot summarise them.** Within-class spread 25.5 vs between-class 9.4.
   Any head over futures must be distributional.
3. **Most concepts are individually unpredictable.** Median grouped R^2 **-0.110**;
   only 35% of 2528 dims beat R^2 > 0. Kaveh's expectation, quantified.
4. **Predictability does not sit where value sits.** corr(value-head weight, per-dim
   R^2) = **-0.001**. A fixed importance weight read off the value head buys nothing.
5. **Futures fall into kinds worth ~1.7 pawns** -- but globally those kinds track the
   POSITION (only 17% of positions reach more than one). The move's own contribution
   is currently near-invisible: within-position spearman **+0.084** for which future a
   sibling opens, against **+0.172** for the sibling's own value.

(5) is the same within-position blindness that stops the search ranking moves. It is
the deficiency, and it is what this objective is aimed at.

## The subgoal definition

> **A subgoal is a cluster in residual future-concept-occupancy space.**

Concretely: take Stockfish's PV for each legal move, form the discounted concept
occupancy of that line, subtract the position's mean over its own moves (so what
remains is what the MOVE changes, not what the position already was), and cluster.
Each cluster carries the mean Stockfish evaluation of the lines in it.

This is a definition with content, not a relabelling:
* **Discovered, not hand-coded.** No one writes down "trade queens" -- it falls out.
* **Grounded in a teacher, not in itself.** The labels come from Stockfish's actual
  continuations, so they are correct even while our concepts are too weak to predict
  them. This is distillation, not bootstrapping, and it is why the objective is safe
  to train against now rather than after the representation improves.
* **Priced.** Choosing among the kinds available at a position is worth ~1.74 pawns.
* **It replaces the quasimetric's job.** A quasimetric answers "how far to a named
  goal". Kaveh does not have named goals; he has "what kinds of future are open to
  me". A categorical over discovered kinds answers that question directly, and
  P(kind | s, move) is exactly the plan-selection distribution.

## The tool, and why its shape satisfies the constraint

A **categorical head over K future kinds**, per move, plus a value per kind.

Kaveh's constraint -- high loss is expected, Stockfish stays primary, nudge only what
is predictable -- is satisfied **structurally**, not by tuning a weight:

* **Predicting uniform costs exactly log K, and is always available.** So the model is
  never *forced* to pay more than log K for a position whose future is genuinely
  unpredictable: it can decline to predict rather than distort the encoder to fake
  predictability, and the gradient *vanishes* as the prediction approaches the true
  high-entropy posterior. Stated carefully, because the sloppier version — "CE is
  bounded by log K" — is **false** for an arbitrary model: `-log p(true)` is unbounded
  when the head is confidently wrong, and a randomly-initialised head measures slightly
  *above* the ceiling (1.86 against log 6 = 1.79). The bound is on what the model must
  accept, not on what a bad model can score. A per-dimension regression on future
  concepts offers no such option -- with 65% of dims unpredictable it would be dominated by the floor and
  would drag the encoder. **This is the whole reason to predict kinds rather than
  concept vectors.**
* **It is distributional and multimodal by construction**, which finding (2) requires.
* **Importance weighting comes free and ties to the primary objective.** Weight each
  position's CE by the spread of Stockfish value across the kinds actually reachable
  there. Positions where the choice of future is worth 1.7 pawns get gradient;
  positions where every future is equivalent get none. Finding (4) says a fixed weight
  from the value head would not do this -- the weight has to come from what the choice
  is WORTH, which is a per-position quantity.

### Guards, from things that have already gone wrong here

* **Targets from the EMA field, stop-gradient.** Any loss rewarding "future concepts
  predictable from present concepts" has a trivial solution: make concepts constant.
  The EMA field already exists for exactly this and must be the target encoder.
* **Centroids frozen after one fitting pass.** A clustering that moves with the
  encoder is a moving target -- the bootstrap spiral the loss-interaction review caught
  once already. Re-fit only on an explicit field promotion.
* **Cluster the residual, never the absolute.** Measured: absolute clustering gives a
  position recogniser (17% of positions offer any choice at all).
* **Standing health gates apply**: effective rank (collapse) and identity-R^2
  (temporal coherence) -- the stack-less distillation run killed ply-persistence
  while task metrics improved.

### What would falsify it

The criterion is a **matched contrast**, never an absolute number. Two checkpoints of
the same architecture, trained identically except for `--w-fkind`, measured on the SAME
positions and the SAME Stockfish futures (`future_concepts.py --pv-cache`). The auxiliary
earns its place only if within-position spearman for "which future does this sibling
open" is higher **with** it than **without**, by more than the standard error of the
difference.

Absolute thresholds were the original form and they were wrong: +0.084 and +0.172 came
from a different checkpoint at a different sample size, and re-measuring the control on
120 positions moved them to +0.051 and +0.089. Read against the stale numbers, gko8a's
+0.082 would have looked like it nearly cleared the bar while the control had moved just
as far. A threshold measured on one model is not a criterion for another.

### Where it pays off beyond training

The entropy of the predicted kind distribution is a per-move **plan-uncertainty**
readout. That is the missing input to the adaptive search budget already on the
roadmap: deepen search only where the plan is unreliable, and let the budget fall as
planning improves.

## Status

Design, measured but not built. gko7a is training (step 3000/8000); no run has been
launched against this, and the choice of whether to add it to the next training is
Kaveh's.

---

## Loss-interaction review (required before launch)

`l_fkind` against every term already in the stack. Recorded because the first time this
review was run it caught a bootstrap spiral, a wrong-space bug and a dead audit entry.

| against | interaction | verdict |
|---|---|---|
| `l_sf` (sharp head) | identical input (`_av_in`: quantised state + move code), different question -- value bin vs future kind. Measured, predictability and value-relevance are uncorrelated (-0.001), so the kind carries what the value does not | complementary; both pressure sibling discrimination |
| `l_sib` (sibling PL) | `l_sib` pressures the OPERATOR via rolled children, `l_fkind` pressures the ENCODER directly. Different parameters, same deficiency | complementary, and `l_fkind` should make `l_sib`'s job easier |
| `l_decor` | decorrelation could fight a future-kind signal that wants redundant coding across heads | low risk, `w_decor` is small; watch effective rank |
| LFQ entropy / commitment | `l_fkind` reads quantised concepts, so it pressures codes to distinguish futures -- should raise codebook usage, but its gradient passes the straight-through estimator | watch effective rank and code occupancy |
| depth cost (`w_depth`) | **genuine tension.** Future kinds may need deeper, more global features; the depth penalty prefers shallower reads | intended trade-off; watch `depth_profile()` for a shift toward late layers |
| operator losses | no shared parameters. Indirectly, sibling-discriminating concepts give the operator a better target space | expect neutral-to-positive on opR2 |
| EMA child targets | `l_fkind` moves the online field, so EMA targets drift | ordinary; the EMA lag exists for this |

**Scale.** The attainable ceiling is log K and the weight is normalised to mean 1, so
`l_fkind` lands in [0, 2.5] regardless of label coverage. At `w_fkind 0.2` that is at
most ~0.5 against `l_sf` ~3.5 and `l_maia` ~2.2 -- roughly a tenth of the objective,
which is the nudge Kaveh asked for and not a competitor to the Stockfish term.

## Correction to the guards above

The EMA/stop-gradient guard is **superseded, and by something stronger**. Labels are
precomputed integers, derived offline from a frozen reference field and Stockfish's own
continuations, then frozen on disk. The encoder has no path to make its own targets
easier, so the collapse solution that a predict-your-own-future objective invites does
not exist here at all. The frozen-centroid guard still applies and is now automatic:
the centroids live in `centroids.npz` and are never refit during a run.

The cost of that choice, stated plainly: the kinds are defined in the reference field's
geometry, so they are a distillation target rather than a self-consistent one. If the
new field's geometry diverges far from the reference, the labels stay *correct* (they
describe Stockfish's real continuations) but may stop being the most natural partition
of the new space. Refit on an explicit field promotion.

---

## Literature check (Kaveh: "check the literature so you don't chase your tail")

### The design has a name and a precedent

**Successor Options** (Ramesh, Tomar & Ravindran, IJCAI 2019) *is* this framework:
learn successor representations, cluster them, and take the cluster centres as
sub-goals — "the cluster centers translate to a set of sub-goals that have vastly
different successor states, meaning different sub-goals provide access to a different
region in state-space." Deep Successor Options extends it to function approximation by
clustering a *sample* of SF vectors, which is exactly what the `fit` stage does.

**CPC** (van den Oord et al., 2018) independently supports the central choice —
predicting a *kind* rather than a concept vector. CPC casts future prediction as
classification precisely because unimodal regression losses are a poor fit for
high-dimensional futures and waste capacity on unpredictable local detail. That is the
same argument the measurement made here (65% of concept dims have grouped R² < 0).

### Three deliberate deviations, now explicit

**1. State-action residual SR, not state SR.** They cluster ψ(s); we cluster the
position-centred residual of ψ(s, a). Forced by measurement: clustering the absolute
occupancy produced a position recogniser, with only 17% of positions reaching more than
one kind.

**2. The generating policy — the one the literature named for us.** Successor Options
is explicit that "the policy used to learn the SR determines a prior over state space.
As a result, the discovered sub-goals will lie in those states which are more likely to
be visited under π_SR", and they deliberately use a **uniform random** policy to avoid
that bias. We use Stockfish's principal variation, i.e. **optimal play by both sides**.

So our kinds describe *the futures reachable when both sides play well*, not the futures
a weak opponent might allow. For a planner choosing among its own options that is the
right prior. But it is a real commitment and it interacts with the opponent-modelling
line (z-conditioning): against an exploitable opponent the reachable set is different
and larger. Revisit when z-conditioning lands; the labels are cheap to regenerate under
a different generating policy.

**3. Degenerate clusters, handled differently.** They filter candidate states by SR
L1-norm (`SR_min < |ψ(s)|₁ < SR_max`) so that candidates are "neither fully developed
nor extremely sparse". We instead unit-normalise (cosine) and enforce equipartition with
Sinkhorn-Knopp, the SwAV / Self-Labelling remedy. Same failure, different lever.

### Adopted from the precedent, and what it bought

Successor Options selects sub-goals by **cosine similarity** to the cluster centre, not
Euclidean distance. The reason transfers: *which kind* of future a move opens is a
direction of concept change, *how far* it goes is a magnitude, and mixing the two is
what parks a large cluster at the origin. Measured on 686 real residual vectors:

| assignment | effective kinds | kinds/row | within-row value gap |
|---|---|---|---|
| euclidean, nearest-centroid | 4.8 / 6 | 4.21 | 2.38 pawns |
| euclidean + Sinkhorn | 6.0 / 6 | 4.49 | 2.37 pawns |
| cosine (Successor Options) | 5.8 / 6 | 4.56 | 2.44 pawns |
| **cosine + Sinkhorn** | **5.9 / 6** | **4.63** | **2.47 pawns** |

Cosine normalisation alone largely dissolves the imbalance, which confirms the central
blob was a magnitude artefact rather than structure. With it, K=8 reaches 7.5 effective
kinds where euclidean K=12 reached only 5.6 — so K=8 is defensible now and was not
before. The differences between the last three rows are small on 686 samples and are
not claimed as a result; cosine is adopted because it matches the precedent and is
consistently no worse, and K is confirmed on the production fit's own report.
