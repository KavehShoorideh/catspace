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

* **Cross-entropy over K kinds is bounded by log K.** A position whose future is
  genuinely unpredictable costs at most log K, and the gradient *vanishes* as the
  prediction approaches the true high-entropy posterior. The head is free to say "I
  don't know" and stop pulling. A per-dimension regression on future concepts has no
  such bound -- with 65% of dims unpredictable it would be dominated by the floor and
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

After joint training, if within-position spearman for "which future does this sibling
open" has not risen above **+0.084**, and sibling value ranking has not risen above
**+0.172**, the auxiliary is inert and should be removed rather than reweighted.

### Where it pays off beyond training

The entropy of the predicted kind distribution is a per-move **plan-uncertainty**
readout. That is the missing input to the adaptive search budget already on the
roadmap: deepen search only where the plan is unreliable, and let the budget fall as
planning improves.

## Status

Design, measured but not built. gko7a is training (step 3000/8000); no run has been
launched against this, and the choice of whether to add it to the next training is
Kaveh's.
