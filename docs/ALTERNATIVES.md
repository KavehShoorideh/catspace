# Alternatives to the quasimetric — the survey (2026-07-30)

**Why IQE failed here, precisely.** A quasimetric composes point-to-point:
d(a→c) ≤ d(a→b) + d(b→c) — "route via b." Against an opponent that inequality's
*operational meaning* dies: you cannot concatenate routes because the opponent
chooses which branch is actually traversed (the veto). Equivalently, the object
we needed is **sum-over-paths probability** under a policy pair, and a
min-over-paths object cannot represent it (the repo's measured verdict: z-lift
≈ 0 exactly as the theory predicts; replaced by the first-hit probability
field). The IQE machinery itself is healthy (probe verdicts 2026-07-30:
asymmetry ratio 1.79, monotonicity +0.915, 0% triangle violations) — it was
the *semantics* that didn't fit.

## Three legitimate escapes

### 1. Fix the opponent: hitting-time quasimetrics under a stochastic policy pair
The veto argument assumes an *adversary*. Against a **fixed stochastic
opponent** (ours: Maia/z at a rating — the whole thesis), play is a Markov
chain, and expected first-passage time genuinely satisfies the triangle
inequality by the strong Markov property: E_a[T_c] ≤ E_a[T_b] + E_b[T_c]
(run to b, then to c, is one way to hit c). So: **timing is a quasimetric;
probability is not.** An opponent-conditioned d_z(s→g) = E[plies to first hit
| policy pair z] is valid mathematics and could reuse the IQE parametrization
as a composition-enforcing inductive bias — the sibling of the hazard field's
E[plies | hit] head, with structure the MLP head lacks. Cheap to try: the IQE
code exists; train it on the checkpoint corpus timings.

### 2. Go up a level: the attractor lattice (sets, not points) — the veto-proof geometry
Point-to-point forcing does not compose; **set-level forcing does**: if from a
I can force the game into region B, and from *every* state of B force into C,
strategy-stitching gives a force from a into C. That is the attractor
computation of reachability games — transitive *despite* the veto, because the
opponent's branching is absorbed into the set. Transitive partial orders have
a mature embedding literature with calibrated probabilistic semantics:
[order embeddings](https://www.researchgate.net/publication/319770245_Order-Embeddings_of_Images_and_Language) (Vendrov),
[hyperbolic entailment cones](https://arxiv.org/abs/1804.01882) (Ganea),
[probabilistic box lattices](https://arxiv.org/abs/1805.06627) (Vilnis et al.) —
containment of boxes/cones ⇔ order, with P(⊆) calibrated. Sketch: embed each
state's **forcible set** as a box; "forced-reach g" ⇔ g's box ⊆ s's box;
composition is transitivity of ⊆, which survives the opponent. This is also
the natural *certificate geometry* for the ε-support/forced knob and M7's
armed tactics (the blocking condition = the face of the box that doesn't yet
contain g). Graded **plausibly novel** for games; nearest art is entailment
graphs, not game attractors.

### 3. Represent the sum over paths natively: (adversarial) flow networks
GFlowNets make the flow — the sum-over-trajectories object — the thing that is
learned; [Expected/Adversarial Flow Networks](https://arxiv.org/abs/2310.02779)
(ICLR 2024) extend this to stochastic environments and two-player zero-sum
games with equilibrium existence/uniqueness results, learning >80% optimal
moves in Connect-4 and beating AlphaZero in their tournaments
([code](https://github.com/GFNOrg/AdversarialFlowNetworks)). Our
P(first-reach g | s, policy pair) *is* a flow query with tempered opponent
edges. The fit: a principled alternative to the hazard energy if it stalls,
or to the planner's search if expectimax pricing proves miscalibrated.

## Also considered (briefly)

- **Game/bisimulation metrics** (de Alfaro, Henzinger, Majumdar, "Game
  relations and metrics"): true metrics for games — but they measure
  *behavioral similarity between states*, not distance-to-goal; the fit is
  atlas geometry (region assignment), not navigation.
- **Hyperbolic embeddings** for tree-like game DAGs: low-distortion trees, but
  the veto objection applies unchanged at point level; only useful combined
  with escape #2 (Ganea's cones are already hyperbolic).
- **Value-equivalent world models** (MuZero line): models need only be correct
  for planning-relevant quantities — the argument that *supports* the current
  hazard-energy JEPA (predict what the planner consumes, nothing else).
- **Successor measures / first-occupancy representations**: the current
  field's lineage; a properly *adversarial* successor representation appears
  open — escape #3 is its closest living relative.

## Standing recommendation

The JEPA/hazard line stays the best shot (the no-A/B rule). Shelved with
intent: **#1** (opponent-conditioned hitting-time IQE — a weekend, reuses
existing code, adds composition bias to timing) and **#2** (attractor-lattice
box embeddings — the veto-proof geometry, doubles as the forced-win
certificate machinery). **#3** is the named alternative if the hazard energy
or expectimax pricing fails its verdicts.
