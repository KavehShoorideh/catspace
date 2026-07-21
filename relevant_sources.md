# Relevant Sources — bottom-up, tablebase-grounded quasimetric extrapolation

*Curated 2026-07-20. Deep reads (full text) marked ●; lighter reads / abstract-level ○.
Each entry: what it does, the load-bearing detail, and **→ for us** (what it validates, what
we extend, or what it warns). The method these frame:*

> **Our method.** Learn an F/B **quasimetric reachability field** (IQE) whose strata boundary
> is **piece count** (captures — the only count-changing move). Ground it **bottom-up** on the
> exactly-solved tablebase strata (perfect-play labels), and **extrapolate one stratum above
> the solved frontier**, where the capture boundaries serve simultaneously as (a) curriculum
> levels, (b) planner subgoals, and (c) error-reset checkpoints (any position is ≤k captures
> from exact truth). The binding question is not compute but **how fast approximation error
> compounds per stratum above the exact frontier** (ε_n).

The pieces below all exist separately. The combination we could not find in the literature —
and which the newest quasimetric-planning paper (ProQ) names as its open limitation — is
**using an exactly-solved sub-problem library as the grounded frontier for quasimetric
extrapolation, with domain-structural boundaries doing triple duty.**

---

## 1. The quasimetric field (our L1 geometry) — established; we use the SOTA tools

### ● IQE — Interval Quasimetric Embeddings (Wang & Isola, 2211.15120)
The exact embedding our field uses. Latents reshaped to `k×l` (k components/heads, l dims each);
per component, distance = **length of the union of intervals** `⋃_j [u_ij, max(u_ij, v_ij)]`;
groups aggregated by **IQE-maxmean** `α·max_i d_i + (1−α)·mean_i d_i`. Key theory:
- **Thm 3.1** — IQE-maxmean *exactly* represents any finite quasimetric; IQE-sum to distortion `O(t·log²n)`.
- **Thm 3.3** — universal ε-approximation on compact spaces.
- **0–1 parameters** in the distance head (vs thousands for Deep Norm / MRN); latent positive
  homogeneity `d(αu,αv)=α·d(u,v)` → stable gradients.
- Empirically **best-in-class at predicting UNREACHABLE (d=∞) pairs** on web/random graphs.

**→ for us:** our `d=512, 32-component` field is exactly this. The unreachable-pair strength is
the theoretical basis for our **one-way strata** (a capture's reverse is ∞-ish): IQE is the
quasimetric family that represents "no way back" well, not just finite distances. Positive
homogeneity is relevant to the contract-vs-margin scale dynamics we see in training.

### ● QRL — Quasimetric RL (Wang, Torralba, Isola, Zhang, ICML 2023, 2304.01203)
The training objective our field descends from. Constrained optimization:
- **Local constraint** on observed transitions: `relu(d_θ(s,s') + r) ≤ ε` (pin the unit step).
- **Global objective**: `min_θ max_{λ≥0} −E[φ(d_θ(s,g))] + λ(E[relu(d_θ(s,s')+r)²] − ε²)` —
  push all pairs apart subject to the local constraint, via a Lagrange multiplier.
- **Thm 2**: optimizing over the full quasimetric space recovers `d* = −V*` (the optimal
  goal-reaching value) *almost surely*. **Thm 3**: with a universal approximator and the relaxed
  constraint, recovers `−V*` up to scale `(1+ε)` with error `O(√ε)`.
- **Contrastive/InfoNCE alone is insufficient**: it enforces neither how tightly adjacent states
  are pulled nor how far others are pushed, so distances can stay collapsed and it recovers only
  *on-policy* (not optimal) values.
- **Key assumption: deterministic dynamics.** Stochastic/partially-observed settings (their
  image tasks) violate it.

**→ for us:** our `L_pos` (d≈1) + push *is* QRL. Two direct implications: (1) **chess is
deterministic and fully observed**, so we satisfy QRL's core theoretical assumption *cleanly* —
the exact-recovery theorems apply to us in a way they don't to the image-based RL the paper
worried about. (2) The InfoNCE-collapse warning is precisely the failure we already hit
("distances stay small everywhere") and why we moved to the QRL-style push + targeted negatives.
Caveat inherited: recovery is contingent on **data coverage of `p_state`, `p_goal`** — our
sparsity concern, stated as a theorem assumption.

### ● ProQ — Projective Quasimetric Planning (2506.18847, 2025) — *our planner, independently arrived at*
IQE + QRL field, plus a planner that (a) spreads landmark keypoints by **Coulomb repulsion**
`ℒ_repel = λ ∑_{i≠j} 1/(d(z_i,z_j)+ε)`, (b) keeps them on-manifold with a **Lagrangian OOD
detector** `ℒ_ood = −λ ∑_k log ψ(z_k)` (Thm 4.1: calibrated inside reachable areas, repels
extrapolated regions), and (c) selects subgoals by **directional cost** `argmin_k [d(s,k) +
D*(k,g)]` with `D*` from Floyd–Warshall over the keypoint graph.
- **Stated limitation #3 (verbatim):** *"keypoint coverage remains limited to the convex hull of
  the offline dataset — unexplored areas cannot be reliably represented."* Also: OOD detection
  degrades for truly novel regions beyond the interpolation bounds.

**→ for us:** this maps one-to-one onto our field + landmark planner + reachability/committor
gate — strong external validation of the design. **Its open limitation is exactly our target:**
we replace "convex hull of the dataset" with "one capture above an exactly-solved stratum," and
the capture boundaries give hard error-resets the soft OOD detector cannot. *This is the clearest
statement of our contribution: we answer a named open problem in the current SOTA paper.*

### ○ Related quasimetric line
- **Offline GCRL with Quasimetric Representations** (2509.20478, 2025) — recent offline-GCRL
  built on quasimetric reps; confirms the line is active.
- **Hierarchical Quasimetric RL** (ICML-adjacent 2025) — hierarchy atop a quasimetric; relevant
  when we stack the planner's subgoal levels on the strata.
- **MRN** (Liu/Feng/Liu/Stone, AAAI 2023) — the residual-metric alternative we already ablated
  against IQE in `fb.py`.

---

## 2. Bottom-up stratification — this *is* Reverse Curriculum, and ours is a stronger form

### ● Reverse Curriculum Generation (Florensa et al., CoRL 2017, 1707.05300)
Train from start states increasingly far from the goal. A start is a **"good start"** when
`R_min < R(π_i, s_0) < R_max` (success probability in a band — "reaches the goal sometimes, not
always"). New starts generated by **SampleNearby**: Brownian random-action rollouts from mastered
starts. Maintains new+old start buffers (anti-forgetting); mastered starts (`R > R_max`) are
retired. Builds **"a tree of stabilized trajectories backwards from the goal."**
- **Assumption 1**: can reset to arbitrary start states. **Assumption 3**: random actions induce
  a communicating class linking all starts to the goal.

**→ for us:** our bottom-up-by-piece-count *is* reverse curriculum — near-goal (few pieces) is
easy, expand outward, bootstrap. **Where ours is strictly stronger:** (1) each curriculum level
is not merely "easier" but **exactly solved** (tablebase), not adaptively estimated by a reward
band; (2) the boundaries are **domain-structural** (captures), not generated by Brownian
sampling; (3) we trivially satisfy Assumption 1 (reset any board) and Assumption 3 (every game
reaches a terminal/capture by finiteness + the 50-move rule). Their "tree backward from the goal"
= our capture-DAG rooted at mate. The adaptive-difficulty idea (train where you're neither
failing nor mastered) is a lever we could add on top of the fixed strata schedule.

---

## 3. Tablebase-grounded / search-bootstrapped learning — known for *values*, not for a *metric*

### ○ Search Bootstrapping (Veness et al.) & engine+tablebase hybrids (LC0)
Learn an evaluation from deep-search-derived node values; engines already use tablebases as an
**exact terminal anchor** with a neural net above. So "exact frontier + learned extrapolation
above" is established *practice*.
**→ for us:** our negamax-into-tablebase 7p labeling is search-bootstrapping. **The difference:**
engines distill the tablebase into a **scalar eval or a leaf lookup during search**; we distill
it into a **composable quasimetric embedding** that supports subgoal planning and generalizes the
*geometry*, not the scalar. That's the novel use of the same ground truth.

### ● Acquisition of Chess Knowledge in AlphaZero (McGrath et al., PNAS 2022, 2111.09259)
Sparse linear probes (L1, over 16,384-d activations) produce **what-when-where plots**: material
is learned earliest and is trivially linear; **material *difference*** becomes linear only deeper
/ later; king-safety and mobility emerge **late**; the net encodes **single-ply opponent threats**
("could opponent mate in one") — opponent modeling is *in the network*, not deferred to search.
Regressed piece weights converge to classical values by ~128k steps. **Outlier positions cluster
in activation space (t-SNE)** due to representational similarity.
**→ for us:** direct support for (a) the **transfer thesis** — nets trained on chess acquire
human-interpretable, compositional structure; (b) our **UMAP clustering** as a legitimate lens
(they see the same representational clustering); (c) the idea that **value composes from
material + higher-order concepts**, which is what our field-plus-outcome factorization assumes.
The late emergence of king-safety/mobility hints the *harder* structure at higher piece counts is
exactly what will be slowest to learn — relevant to ε_n.

### ○ AlphaZero (Silver et al., 2017, 1712.01815)
Self-play value bootstrapping; the canonical evidence that value learning from
self-generated/curriculum data scales.

---

## 4. The tractability crux — OOD value extrapolation & compounding error (corroborates ε_n)

### ● GOAT — What is Essential for Unseen Goal Generalization of Offline GCRL? (2305.18882)
OOD **goal** generalization is bottlenecked by **advantage/value estimation error**;
**pessimism-based offline RL cannot generalize to OOD goals** (it avoids OOD actions, so
trajectories stay in-support); plain imitation overfits noise; **weighted imitation** is the
strong baseline. GOAT = uncertainty weighting (ensemble-std ≈ inverse density) + exponential
advantage weighting + data-selection thresholding + expectile regression.
- **Thm 3.1** — suboptimality bounded by `(2R_max/(1−γ)²)·[imitation loss + expert-gap +
  worst-case distribution shift d₁(T,S) + sample-complexity]`.
- Empirically **91.2% IID → 70.9% OOD**, with **performance cliffs at distribution boundaries**.

**→ for us:** the theorem's **`d₁(T,S)` (train→test distribution shift)** is precisely our
"distance above the exact frontier." The prescription — **uncertainty/density weighting + data
selection + expectile** — is the lever set for climbing strata. Expect **cliffs** at the frontier,
not smooth decay, unless anchoring holds. This is the strongest corroboration that **error, not
compute, is the binding constraint.**

### ○ Approximate Value Iteration error bounds (Munos) · Edge-of-Reach (2402.12527) · Compounding-error control
AVI/AMPI bounds quantify how Bellman error propagates; model-based rollouts compound error
`~γ^(k−t)·ε` per step; **Edge-of-Reach** shows failure specifically at states the model was never
grounded on. **"Learning to Combat Compounding-Error"** adaptively picks the **planning horizon
from a learned model-error function** — search deeper where the model is uncertain.
**→ for us:** formalizes "error compounds per stratum above exact"; **Edge-of-Reach = our quiet,
far-from-a-capture, under-sampled failure mode**; the adaptive-horizon result = **uncertainty-gated
search depth** (our competence/entropy head deciding when to search deeper). Concretely: below the
frontier ε=0 (exact); the reset-at-capture only fully protects the *first* stratum above exact.

---

## 5. Difficulty-axis / length extrapolation — does the metric climb a stratum? (mixed, favorable)

### ● Logical Extrapolation for Mazes (2410.03020)
"Logical extrapolation" = OOD shift along a difficulty axis. Weight-tied RNNs / implicit (DEQ)
nets extrapolate by **spending more test-time iterations**: DT-Net trained on 9×9 solves 99×99.
**But it fails on conceptually-different axes** (cyclic mazes with percolation p>0 — more
iterations don't help), and appears to learn *"the simplest algorithm that fits the training
data"* (matches deadend-filling 98.8% of the time, failing where that algorithm fails). Fixed-point
convergence is **not** necessary; **iterative refinement is the operative inductive bias.**
**→ for us:** positive evidence that **difficulty-axis (piece-count) extrapolation is possible**
with iterative computation + test-time compute (= deeper search above the frontier). The warning
is sharp: extrapolation holds **only along the same conceptual axis** — climbing to more pieces
should transfer *if the tactical structure is the same kind*, but **genuinely new motifs** at
higher piece counts (fortresses, new mating nets) are the "cyclic maze" case where it breaks.
Directly predicts *which* positions ε_n will blow up on.

### ● Tropical Attention (2505.17190) — why our *geometry* is on the favorable side
Softmax value functions **blur OOD**: combinatorial value functions are **piecewise-linear /
polyhedral**, but softmax carves the domain into "spherical caps" and dilutes as size grows.
The **tropical/max-plus semiring `(ℝ∪{−∞}, max, +)`** keeps operations piecewise-linear, is
**1-Lipschitz in the Hilbert projective metric**, and extrapolates: length 8 → 1024 on QuickSelect,
large value-OOD gains. The tropical Bellman recursion `d_v ← max_u (w_uv + d_u)` *is* shortest-path
relaxation (Floyd–Warshall is a benchmark).
**→ for us:** a real architectural argument. **IQE's distance is a min-over-intervals — the
min-plus (tropical dual) geometry** — and our composition is the **triangle inequality =
min-plus shortest path**. So choosing a **quasimetric over a smooth MLP value is exactly the
"sharp polyhedral geometry extrapolates OOD, smooth softmax blurs" prescription.** The paper stops
at max-plus and doesn't connect to min-plus/quasimetrics — that connection is ours to make and is
a citable justification for the whole L1 design.

### ○ Value Iteration Networks (1602.02867) · Value Propagation · XLVIN
Embed the VI computation as a differentiable module → generalizes planning to unseen domains;
supports the "bake in the iterative planning structure to extrapolate" thesis.

---

## 6. Landmark planning — SoRB is the canonical citation

### ● Search on the Replay Buffer (Eysenbach et al., NeurIPS 2019, 1906.05253)
Graph over buffer states; **goal-conditioned value = edge weights**; **Dijkstra** for waypoints.
Two reliability tricks: **distributional RL** (a catch-all "≥N steps" bin handles unreachable
pairs cleanly — no ill-defined ∞), and **ensembles** to kill **"wormholes"** (spurious "these far
states are actually close" shortcuts). **MaxDist** prunes implausible edges.
- **Limitation**: `|B|²` distance pairs can't all be reliable — many never occurred in training;
  graph search *exploits* wrong short predictions.

**→ for us:** our landmark planner is SoRB with a quasimetric. **Its two tricks are things we need
directly:** the distributional catch-all bin = our **categorical outcome head's** treatment of
unreachable/draw, and the ensemble anti-wormhole = our **committor/uncertainty gate** (a wormhole
is exactly a false cross-stratum shortcut our count-drop one-way term must forbid). The `|B|²`
unreliability is the OOD-distance risk our exact-frontier grounding is meant to bound.

---

## Synthesis

**What's established (we're standing on it):** the quasimetric field (IQE/QRL/ProQ), reverse
curriculum (Florensa), search-bootstrapping-from-tablebases (Veness, engine hybrids), landmark
planning (SoRB/ProQ), and the OOD-value-error theory (GOAT, AVI bounds, Edge-of-reach).

**What's novel (and targets a named open problem):** using an **exactly-solved sub-problem
library** (the tablebase strata) as a **grounded frontier** from which a **quasimetric field**
extrapolates upward, with **domain-structural boundaries (captures)** doing triple duty as
curriculum levels, subgoals, and **error-reset checkpoints**. ProQ's stated limitation (coverage
limited to the dataset's convex hull) is precisely what the exact-frontier + capture-reset
mechanism is designed to break.

**Architectural tailwinds the literature gives us:**
- Chess is **deterministic + fully observed** → QRL's exact-recovery theorems apply cleanly (they
  don't to the image-RL settings QRL itself worried about).
- IQE is **min-plus / polyhedral geometry**, which Tropical Attention shows is the *right* geometry
  for OOD/length extrapolation (smooth softmax value blurs; sharp quasimetric doesn't).
- Iterative test-time compute (deeper search above the frontier) is a proven extrapolation lever
  (maze paper), and **uncertainty-gated adaptive depth** is the prescribed way to spend it.

**The binding constraint (unanimous across the OOD strand): error, not compute.** ε_n — how fast
the field's error compounds per stratum above the exact frontier — decides how far this climbs.
The exact-frontier reset fully protects only the *first* stratum above exact; beyond that, leaves
are approximate and GOAT's `d₁(T,S)` shift term kicks in, with **cliffs at the boundary**, worst
on Edge-of-reach (quiet, far-from-capture, under-sampled) positions.

**Prescribed next experiment (from the literature, not just intuition):** measure ε_n vs stratum —
field-vs-tablebase where ground truth exists (≤ frontier), spot-check search above it — and pair
climbing with **uncertainty-gated search depth** (Combat-Compounding-Error) + a **ProQ-style OOD
gate** so the planner refuses to trust the field where it has left the grounded manifold.

---

### Read log
● full-text deep read: IQE (2211.15120), QRL (2304.01203), ProQ (2506.18847), Reverse Curriculum
(1707.05300), SoRB (1906.05253), GOAT/unseen-goal (2305.18882), McGrath (2111.09259), maze
extrapolation (2410.03020), Tropical Attention (2505.17190).
○ abstract/secondary: Hierarchical QRL, Offline GCRL-QM (2509.20478), MRN, AlphaZero (1712.01815),
Search Bootstrapping, AVI error bounds (Munos), Edge-of-Reach (2402.12527), Combat-Compounding-
Error, VIN (1602.02867), Conditions for Length Generalization (2311.16173).
