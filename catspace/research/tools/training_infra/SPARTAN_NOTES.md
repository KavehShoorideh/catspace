# SPARTAN vs the QGT transition — similarities, differences, borrowables

*Lei, Schölkopf, Posner: "SPARTAN: A Sparse Transformer World Model Attending to
What Matters", NeurIPS 2025 (arXiv 2411.06890v3). Read from the primary PDF
2026-08-28 (§1–4 verified directly; results-section numbers not re-verified here).
Written against `qgt_field.py` (GraphBitTransition) as of the qgt7 line.*

## What SPARTAN is

A transformer world model over **object-slot tokens** (from a pre-trained
object-centric encoder) whose central claim: **hard attention + sparsity
regularisation makes attention a faithful *local causal graph* of the dynamics.**
Mechanics, verified from §3:

- **Adjacency = sampled hard mask:** `A_ij ~ Bern(σ(q_i·k_j))`, Gumbel-softmax for
  gradients; adjacency masks standard scaled dot-product attention. State-dependent
  by construction (key–query).
- **Multi-layer cone via the path matrix:** `Ā = (A^L+I)···(A^1+I)`; token j is a
  local causal parent of i iff `Ā_ij ≥ 1` — their answer to "who can influence whom
  across L layers."
- **Training:** one-step next-slot prediction (MSE) + **L1 sparsity on the path
  matrix**, `min E[L(ŝ^{t+1}) + λ_s|Ā|]`, with λ scheduled by Lagrangian relaxation
  (they note mode-collapse sensitivity to λ).
- **Interventions:** a learnable codebook of *intervention tokens* appended to the
  token set; unknown interventions at test time are handled by **gradient descent in
  token space** to find the token that explains the observed shift.
- **Key argument (§3):** soft attention + thresholding is *insufficient* for causal
  discovery — nonzero soft weights on non-parents mean thresholds catch spurious
  edges or miss real ones; only masking + a sparsity price makes the best-loss model
  the one with exactly the true edges.
- **Evaluation:** Interventional Pong, CREATE (2D physics), and Traffic (Waymo Open
  Dataset, ego-motion); learnt graphs scored against ground-truth causal graphs
  (sims) and human-labelled graphs (Traffic). Claimed firsts: local causal structure
  beyond state-based settings; better edge identification, distractor robustness,
  few-shot adaptation.
- **Edges are UNTYPED** — presence/absence only (a scalar Bernoulli per pair).

## Similarities (they got there first — cite them)

1. **Attention as the learned interaction structure of a transition function**, with
   sparsity/hardness as the faithfulness mechanism — the qgt6 thesis, published.
2. **State-dependent graphs** recomputed from the current state each step (their
   key–query masks ≈ our GNS-style recompute-from-bits).
3. **A multi-hop influence-cone formalism** (their path matrix ≈ our `touched()`).
4. **External ground truth for the learned graph** (their sim/human causal graphs ≈
   our python-chess relation oracle).
5. **A learned token codebook conditioning the dynamics** (their intervention tokens
   ≈ our dyn move codes — different semantics, same mechanism family).

## Differences / our contributions (design-level; validation pending on qgt7+)

1. **Typed edges via a discrete question codebook.** SPARTAN's edges are binary. Our
   QA questions give every edge a *type* from a learned vocabulary — falsifiable per
   type against the oracle (`relation_alignment`), warm-startable per type from free
   domain labels (attacks). Richer object: not "who influences whom" but "who
   influences whom *how*."
2. **Fully discrete state.** They roll continuous slots with MSE; we roll ±1 concept
   bits with flip-weighted BCE at the 2-ply move unit (reply marginalised → per-bit
   flip probabilities). Interpretability constraint reaches the *state*, not only
   the graph.
3. **Exact frame axiom.** Their cone is stochastic (sampled masks, path counts);
   ours is a theorem of the forward pass — the no-input baseline subtraction makes
   an untouched square's delta *identically zero*, so "touched" is provable, not
   sampled.
4. **Localised action injection.** Moves enter at their from/to squares only;
   SPARTAN appends intervention tokens globally.
5. **Value-grounded, planning-facing training.** Their objective is reconstruction
   of next slots; ours is teacher bits + externally labelled value/regret through a
   Dreamer-separated readout, with the downstream consumer (search/planner) fixed in
   advance.

## Borrowables (queued candidates, not yet in)

- **Sparsity regularisation on the path matrix** (`λ_s|Ā|`, Lagrangian-scheduled):
  we currently have *no* sparsity price on attention — our locality is measured
  (`cov`), not enforced. Their regulariser is the natural sharpener if the cone
  stays sloppy. Mind their mode-collapse warning about λ.
- **Their soft-attention-insufficiency argument explains our H1.** The warm start
  supervises *pre-top-k logits* while routing uses the hard top-k — exactly the
  soft-vs-used gap they argue makes thresholded attention unfaithful. If H1 lands
  (auc flat), the fix their framing suggests: supervise the *selected* edges, or
  price unselected mass.
- **Test-time adaptation by gradient descent in token space.** Their
  unknown-intervention trick maps beautifully onto this project's opponent model:
  an opponent's style as an "intervention token" recovered online — a candidate
  mechanism for the M2c (Elo, z) filter's dynamics-side counterpart.
- **Few-shot adaptation as an evaluation axis** — a template our acceptance
  batteries don't yet have.

## Claim hygiene

Our novelty claims above hold against five literature sweeps (NRI/Gumbel structure
learning, Transformer-VQ/clustered attention, VQGraph/node tokenizers, SPARTAN) —
but a print-grade related-work pass must still walk SPARTAN's and NRI's forward
citations. And all "contributions" are design-level until a qgt run validates the
mechanism (typed-edge faithfulness probe > 2× density vs shuffle, per the locked
spec's acceptance).
