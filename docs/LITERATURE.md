# Literature base — exact formulations and hyperparameters

Extracted from primary sources (papers + reference implementations), not summaries, so we
never re-derive them. Every number here is quoted from the source; our own measurements are
marked **OURS**.

---

## 1. QRL — Optimal Goal-Reaching RL via Quasimetric Learning
Wang, Torralba, Isola, Zhang · ICML 2023 · [arXiv 2304.01203](https://arxiv.org/abs/2304.01203)
· repo [quasimetric-learning/quasimetric-rl](https://github.com/quasimetric-learning/quasimetric-rl)

### Objective (their Eq. 12)
```
min_θ max_{λ≥0}  −E_{s~p_state, g~p_goal}[ φ(d_θ(s,g)) ]
                 + λ ( E_{(s,a,s',r)}[ relu(d_θ(s,s') + r)² ] − ε² )
```
Maximise distances between pairs drawn from **independent marginals** (NOT same-trajectory —
same-trajectory sampling is what makes contrastive methods recover on-policy rather than
optimal values), subject to observed transitions costing at most their true cost.

**Why it recovers the geodesic:** among all functions satisfying the local constraints, the
*maximal* one is the shortest-path quasimetric. Pull the chain taut with no link longer than
one step.

### Exact defaults (`modules/quasimetric_critic/losses/local_constraint.py`)
```python
epsilon:                  0.25
step_cost:                1
init_lagrange_multiplier: 0.01     # held as softplus_inv, recovered via F.softplus
```
```python
sq_deviation = (dist - step_cost).relu().square().mean()
violation    = sq_deviation - epsilon ** 2      # <= 0 means satisfied
loss         = violation * lagrange_mult
```

### Learning rates (paper appendix) — THE RATIO IS THE POINT
| param | offline maze2d | online GCRL |
|---|---|---|
| λ (dual) | **0.01** | **0.01** |
| critic / model | 5e-4 | 1e-4 |
| policy | 3e-5 | 3e-5 |

λ runs at **~100:1** over the model. **OURS:** at model lr 3e-4, λ=0.01 is only 33:1 — three
times slower than theirs relative to what it must track, and jqt7 v2 drifted because of it.

### φ — the saturating transform on the maximised distance
```
φ(x) = −softplus(15 − x, β = 0.1)          # online Fetch, 50-step episodes
```
Slope is `σ(β·(knee − x))`: ~1 below the knee, → 0 above. Exists because naive maximisation
"tends to increase the weight norms of the late layers" so λ "needs to constantly catch up."
**Knee must be set to YOUR horizon.** **OURS:** chess games run 40–150 plies → knee 60.
**SCAR:** do NOT rescale β to the knee. The knee sets *where* the push stops; β sets *how
sharply* (transition width ≈ 1/β). β=0.03 left slope 0.71 at d=30 — the push never saturated
in our range and ran away.

### Recovery guarantee (Theorem 3)
With relaxed constraint `relu(d(s,s') + r) ≤ ε`, recovers `−V*` **up to a known scale (1+ε)**
→ distances read ~1.25× true and must be divided out before being interpreted as plies.

### Architecture
IQE-maxmean head, **64 components × 32** = 2048-d, behind a 256→1024→1024→2048 projector.
**OURS:** 48-d, 16 components — ~40× smaller. Untested whether this is a binding limit.

### Transition loss (their Eq. for L_transition)
```
L_transition = ½ ( d^z(ẑ', z')² + d^z(z', ẑ')² )
```
Next-latent prediction scored **by the learned quasimetric**, symmetrised — "empirically
superior to a simple regression loss on Z, whose scale is meaningless."
Weight 1 (offline), 0.1 (online state), **10 for image observations "since the dynamics
aren't fully deterministic"** — chess-with-an-opponent is that regime.
**OURS:** independently explains why our JEPA-MSE collapsed into persistence.

### Batch sizes
4096 (offline maze2d, 2e5 steps), 256 (online).

---

## 2. FSQ — Finite Scalar Quantization: VQ-VAE Made Simple
Mentzer et al., DeepMind · [arXiv 2309.15505](https://arxiv.org/abs/2309.15505)

### Quantization
```
bounding:      f(z_i) = ⌊L/2⌋ · tanh(z_i)        (with a shift when L is even)
straight-thru: round_ste(x) = x + sg(round(x) − x)
```
Implicit codebook = the product of per-dimension levels. **No learnable codebook**, so no
commitment loss, no EMA, no reseeding, no splitting, no entropy penalty. **Reconstruction
loss only.**

### Recommended levels (their Table 1)
| target size | 2⁸ | 2¹⁰ | 2¹² | 2¹⁴ | 2¹⁶ |
|---|---|---|---|---|---|
| levels ℒ | [8,6,5] | [8,5,5,5] | [7,5,5,5,5] | [8,8,8,6,5] | [8,8,8,5,5,5] |

Rule: **use Lᵢ ≥ 5 ∀i**. Dimensionality stays small (typically < 10).

### Codebook usage
FSQ ≈ **100%** (>99.5% at 2¹⁶). VQ: 81% at 1024; **0.78% without code splitting**.

### WHEN IT WINS — read before adopting
FSQ beats VQ **above ~2¹⁰ = 1024 codes**; below that "VQ marginally outperforms FSQ."
VQ's reconstruction FID bottoms out at 2¹¹ then degrades; FSQ keeps improving.
**OURS: our heads carry 16–64 codes — deep inside the regime where VQ is the better choice.**
FSQ's dimensions play the role our heads play, so adopting it pushes toward ~6 factors of 5–8
values, which is the OPPOSITE of the "more, thinner concepts" design (16 × 16) we chose.
Worth testing only because our pathology is *collapse*, which FSQ prevents by construction —
not because it is a free win.

### Library
`lucidrains/vector-quantize-pytorch` (already a dependency) provides `FSQ`, `ResidualFSQ`,
`GroupedResidualFSQ`, `LFQ`, `SimVQ`.
Caveat for our residual stack (global → square → piece): FSQ suffers **residual magnitude
decay** in multi-stage settings — later stages get exponentially weaker signal.
[Robust Residual FSQ, arXiv 2508.15860](https://arxiv.org/abs/2508.15860) fixes it with
learnable scaling + invertible layer norm.

---

## 3. IQE — Interval Quasimetric Embeddings
Wang & Isola · [arXiv 2211.15120](https://arxiv.org/abs/2211.15120) · the head we already use.
Satisfies all four criteria for quasimetric models; QRL uses IQE-maxmean.

---

## 4. Concepts in games — closest prior art

**Bridging the Human–AI Knowledge Gap** ([PNAS](https://www.pnas.org/doi/10.1073/pnas.2406675122),
[arXiv 2310.16410](https://arxiv.org/pdf/2310.16410)) — extracts concept vectors from AlphaZero
by convex optimization, using the policy-value net **and the MCTS tree**. Discovers "**dynamic
concepts that motivate a sequence of actions**" — the closest published analogue to our
trajectory layer. Concepts were taught to grandmasters.

**Codebook Features** ([arXiv 2310.17230](https://arxiv.org/pdf/2310.17230)) — vector
quantization applied at **each hidden layer** for interpretability and *control/steering*.
Closest published analogue to our multi-head concept design.

---

## 5. Adjacent quasimetric work (all continuous control, no board games found)
- ProQ, projective quasimetric planning with spread keypoints — [arXiv 2506.18847](https://arxiv.org/pdf/2506.18847)
- Eik-QRL, Eikonal-PDE continuous-time reformulation, hierarchical — [arXiv 2512.12046](https://arxiv.org/pdf/2512.12046)
- Multistep Quasimetric Learning — [arXiv 2511.07730](https://arxiv.org/pdf/2511.07730)
- Minimum Action Distance — [arXiv 2506.09276](https://arxiv.org/html/2506.09276)
- Offline GCRL with Quasimetric Representations — [arXiv 2509.20478](https://arxiv.org/abs/2509.20478)
- Benchmarks: maze2d, Fetch, MountainCar, [OGBench](https://arxiv.org/pdf/2410.20092)

**Gap:** a literature search found no quasimetric built over a discrete concept vocabulary,
and none applied to a turn-based board game.

**Retracted (2026-08-15):** I claimed the single-agent formalism was *structurally*
inapplicable to chess because an adversary picks half the plies. Kaveh's rebuttal is correct
and stands: our corpus is **Stockfish-vs-Stockfish**, so a witnessed path IS an
adversarially-realised path and the ceiling `d ≤ observed gap` is a valid bound under optimal
opposition. That is exactly the Option-B argument already encoded in our walls. The adversary
is in the data-generating process. The gap is that nobody has *done* it, not that it is invalid.
Residual nuance, acknowledged: which optimal line gets played is stochastic, and we do not
condition on opponent identity or full history.

---

## 6. MuZero — Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model
Schrittwieser et al. · [arXiv 1911.08265](https://arxiv.org/abs/1911.08265) · verified against the
PDF 2026-08-16, NOT from recall (this entry exists because I implemented K=5 from memory and
missed two of the four scaling choices).

### Training (Appendix G) — the exact wording
```
"During training, the MuZero network is unrolled for K hypothetical steps"
"In the experiments reported in this paper, we always unroll for K = 5 steps."
```
**Gradient is scaled in TWO separate places**, and they are easy to conflate:
```
"We scale the loss of each head by 1/K, where K is the number of unroll steps. This ensures
 that the total gradient has similar magnitude irrespective of how many steps we unroll for."

"We also scale the gradient at the start of the dynamics function by 1/2. This ensures that
 the total gradient applied to the dynamics function stays constant."
```
**Hidden state is RANGE-NORMALISED at every step:**
```
"To improve the learning process and bound the activations, we also scale the hidden state to
 the same range as the action input ([0,1]): s_scaled = (s - min(s)) / (max(s) - min(s))"
```
**OURS:** we had the 1/K only. The other two matter for us specifically: at unroll step 0 our
state is an actual codebook vector, at k>=1 it is a CONVEX MIXTURE of codebook vectors -- a
different distribution, which is exactly the off-manifold drift that made cdf1's free-running
rollout fall 0.590 -> 0.223 over 8 plies. Range normalisation puts both on the same footing.

**Board games bootstrap to the game's end, not n-step:**
```
"in board games without intermediate rewards, we omit the reward prediction loss. For board
 games we bootstrap directly to the end of the game, equivalent to predicting the final
 outcome; for Atari we bootstrap for n = 10 steps into the future."
```
Which is what our outcome term already does.

### Architecture (Appendix F)
```
"Both the representation and dynamics function use the same architecture as AlphaZero, but
 with 16 instead of 20 residual blocks. We use 3x3 kernels and 256 hidden planes."
```
**OURS:** 8.96M params against MuZero's ~40M+ (16 blocks x 256 planes). Correction to an earlier
estimate of mine that assumed AlphaZero's 20 blocks.

### Search and data (Appendices C, D)
```
"For board games, we use the same UCB constants, dirichlet exploration noise and the same 800
 simulations per search as in AlphaZero."
"the latest checkpoint of the network (updated every 1000 training steps) is used to play games"
"the training job keeps an in-memory replay buffer of the most recent 1 million games"
```
pUCT constants c1 = 1.25, c2 = 19652. Discount 0.997 (1 for board games, no intermediate rewards).

**OURS:** no search, no self-play, fixed 49,905-game corpus. The gap in TRAINING COMPUTE is
~500x (3.84M samples seen vs ~2B), far larger than the ~5x parameter gap.

---

## 2026-08-19 — read for the reachability pivot (full texts, arXiv PDFs)

### VAMPnets (Mardt, Pasquali, Wu, Noé — arXiv 1710.06012 / Nat. Comm. 9:5, 2018)

Read for: how to learn SLOW discrete states from trajectories. It answers with a
refutation of our current mechanism first:

* **The degenerate-objective warning (their §II.A, verbatim logic):** minimizing the
  least-squares change ||chi(x_{t+tau}) - K chi(x_t)||^2 is solved by chi = constant —
  "the model is completely uninformative as all dynamical information is lost."
  **cef5's slow-head MSE persistence penalty is exactly this objective.** Slowness must
  be bought as maximal time-lagged CORRELATION in a whitened space, not minimal change.
* **The objective:** VAMP-2 score R2 = ||C00^{-1/2} C01 C11^{-1/2}||_F^2 (+1 with the
  constant function added), with MEAN-FREE batch covariances
  C00=(T-1)^{-1} X̄X̄ᵀ, C01=(T-1)^{-1} X̄Ȳᵀ, C11=(T-1)^{-1} ȲȲᵀ over time-lagged pairs
  (x_t, x_{t+tau}); two identical network lobes; backprop through both (their Eqs 6-11).
  The C00^{-1/2}/C11^{-1/2} whitening forces decorrelated unit-variance features — it
  does structurally what our vicreg_covariance patch does, plus the slowness.
* **Softmax states:** output layer is a softmax over k states (chi_i >= 0, sum 1) — a
  FUZZY clustering to metastable states; K(tau) = C00^{-1} C01 is then the transition
  matrix of a fuzzy MSM. Hard state = argmax. k softmax outputs resolve k-1 timescales.
* **Lag time tau:** choose where implied timescales t_i(tau) = -tau / ln|lambda_i(tau)|
  plateau in tau; validate with Chapman-Kolmogorov K(n·tau) ≈ K(tau)^n. Their measured
  coupling: MORE output states -> SHORTER lag suffices (alanine: 2 states need tau=200ps,
  8 states tau=1ps). Too-long lag: slow process amplitude vanishes; too-short: everything
  looks slow and optimization sticks.
* **Training facts:** Adam on (x_t, x_{t+tau}) pair batches, 90/10 split scored by
  validation VAMP-2; pyramid layer sizing n_i/n_{i+1} = (n_in/n_out)^{1/d}, depth 4-7
  best, ReLU, dropout in early layers; optimisation is multi-modal — they run 100
  restarts and report <40% success at finding the rarest slow process. Multiple seeds
  are part of the method, not paranoia.
* **What replaces our churn% metric:** implied timescales in PLIES per slow state +
  the CK test. "Slow concept" becomes a measured relaxation time, not a percentage.

### C-learning (Eysenbach, Salakhutdinov, Levine — arXiv 2011.08909, ICLR 2021)
github.com/google-research/google-research/tree/master/c_learning

Read for: the canonical P(reach) estimator and its TD form. Our crf1 is their MC
variant with a hard window; the pieces we lack are:

* **The object:** gamma-discounted future-state density
  p+^pi(s+|s,a) = (1-gamma) * sum_Delta gamma^Delta * p(s_{t+Delta}=s+). GEOMETRIC
  horizon, not a hard H — sample Delta ~ Geom(1-gamma) along the trajectory for
  positives, random states for negatives; classifier C(F=1|s,a,s+); Bayes recovers the
  density f = C/(1-C) * p(s+), and POLICY decisions don't need the marginal p(s+).
* **Their Remarks 1-2 kill two tempting shortcuts:** exact-match Q is identically zero
  in stochastic/continuous settings, and hindsight-relabeled Q-learning learns neither
  a Q nor a density (bounded [0,1] while the density isn't). Supports classifier +
  calibration as the primary object (what crf1 already is).
* **TD / off-policy C-learning (Alg 2) — the within-parent machinery.** Recursion:
  p+(s+|s,a) = (1-gamma) p(s_{t+1}=s+|s,a) + gamma E_{s',a'~pi}[p+(s+|s',a')].
  Per transition (s,a,s') with random goal s+, three-term CE loss:
      (1-gamma) * CE(C(s,a,s'), y=1)              # next state as positive
    +            CE(C(s,a,s+), y=0)               # random goal as negative
    + gamma * stopgrad(w) * CE(C(s,a,s+), y=1)    # bootstrapped positive
  with w = C(s',a',s+) / (1 - C(s',a',s+)), a' ~ pi(s'). Equivalent soft target
  y = gamma*w / (gamma*w + 1). THIS evaluates a NEW policy's reach probabilities from
  OLD data — the flywheel-safe step before any self-play generation.
* **Goal-conditioned form (Alg 3):** condition classifier and policy on the commanded
  goal; policy objective is simply max E[log C(F=1|s,a,g)].
* **Their HER hypothesis, confirmed experimentally:** Q-learning-with-relabeling is
  sensitive to the goal-relabeling ratio with optimum ~ (1+gamma)/2, and UNDERESTIMATES
  the density; C-learning has no such knob.
* **Resonance with our measured failure:** their Fig 2 — a 1-step dynamics model
  UNROLLED to predict far futures degrades badly with horizon while the direct
  classifier stays accurate. Same lesson as our victim-chaining conjunction failure
  (joint 10.4%, derived signature 8.1%): predict the destination directly, never
  compose per-step predictions.

---

## 2026-08-19 — GNN best-practice audit for the graph concept field (full texts)

### GraphGPS (Rampasek et al., arXiv 2205.12454)

The canonical paper for exactly our architecture family (local MPNN + global attention per
layer). What it validates and what it corrects:

* **VALIDATED: the hybrid itself.** Their ablation: removing the MPNN is a "major drop" on
  every dataset; adding the Transformer module is beneficial-or-neutral everywhere. They argue
  AGAINST two-stage stacks (all-MPNN-then-attention, GraphTrans-style) -- early over-smoothing
  "irreparably" loses information -- which our per-layer interleaving already avoids.
* **CORRECTION: their canonical layer runs MPNN and attention IN PARALLEL on the same input,
  summed, then a 2-layer MLP** (X^{l+1} = MLP(MPNN(X) + Attn(X)), residuals on each). Ours is
  sequential (local, then attention on the refined output). Not ablated head-to-head by them,
  but the parallel-sum is the published recipe.
* **CORRECTION: positional/structural encodings are "one of the most important factors" for
  graph transformers.** RWSE -- diagonal of the m-step random-walk matrix as node features --
  gives consistent cheap gains (ZINC MAE 0.113 -> 0.070). Caveat for us: their graphs are
  ANONYMOUS (molecules); our square-nodes carry identity (file/rank + piece), so the 1-WL
  blindness argument is weaker here. Still, RWSE over the per-position ATTACK graph is a
  meaningful mobility/connectivity summary and cheap to precompute.
* Edge features feed the MPNN only, never the attention -- matches our design exactly.

### Graphormer (Ying et al., arXiv 2106.05234)

* **Spatial encoding as attention bias**: A_ij += b_phi(i,j), a learnable scalar per pair-
  relation bucket (they use shortest-path distance), per head, shared across layers. This is
  THE practice we lack: our global attention sees no board geometry at all beyond what leaks
  through features. The chess-native phi is not SPD but the square-pair relation itself:
  same-file / same-rank / same-diagonal / knight-hop / king-adjacent / else, plus
  move-node<->its-src/dst buckets. A static 64x64 bucket table, near-zero cost.
* **Centrality encoding** (degree embeddings added to node features): our attacker/defender
  counts per square ARE degree-in-the-attack-graph -- already satisfied.
* **Pre-LN** (LN before MHA and FFN) "has been unanimously adopted... leads to more effective
  optimization" -- ours is post-LN; switch.
* **Virtual node** = their graph readout; GPS's Fact 2 shows plain self-attention can emulate
  mean readout, so our pooled-mean global-concept readout is defensible as-is.

### Deltas adopted for gcf2 (bundled per the no-one-lever rule; gcf1 finishes as baseline)

1. GPS parallel form: MLP(MPNN(X) + Attn(X)) with residuals, replacing sequential.
2. Pre-LN throughout.
3. Chess spatial attention bias: learnable per-head scalar over static square-pair relation
   buckets (+ move-node buckets).
4. PNA-lite aggregation in relations: mean + max concatenated (mean-only discards existence
   information; PNA is GPS's strongest MPNN on several tasks).
5. RWSE-8 over the attack graph as 8 extra square features (precomputed at graph build).
6. Centrality: already covered by attacker/defender count features -- no change.

---

## TreeQN / ATreeC (Farquhar et al., ICLR 2018, 1710.11417) — read in full (pdf)

Differentiable tree unrolled INSIDE the network: encoder -> per-action residual transitions
z^a = z_env + tanh(W^a z_env) with z_env = z + tanh(W_env z) -- SHARED down the tree; states
UNIT-L2-NORMALIZED before every transition (their anti-drift trick); linear value per node;
soft-greedy backup b(x) = sum_i x_i softmax(x)_i so gradients flow down ALL paths (temp 1
sufficed, learned temp = no empirical gain); TD(lambda) mixing of INTERMEDIATE node values
into the final estimate ("encourages intermediate nodes to correspond to meaningful states,
reduces outlier impact"). Backprop end-to-end from the final Q through backup, values,
transitions, encoder. Ablations: depth 2-3 > 1; REWARD grounding helps; latent STATE
grounding (L2 to encoded true next state) does NOT help (and decoder-grounding also failed,
citing Oh et al.).

ADOPTED for gct: move-embedding-conditioned residual transition (4096 chess actions forbid
per-action matrices; our mv_emb replaces W^a), unit-norm before transitions, soft backup
with NEGAMAX parity (opponent plies: -b(-x)), lambda-mixing ~0.8, depth 3-4 beam 2-3.
STATE-grounding kept ONLY as the interpretability contract (dyn CE on real children), with
TreeQN's finding noted: do not expect performance from it.

## Gumbel MuZero (Danihelka et al., ICLR 2022) — via the mctx REFERENCE IMPLEMENTATION
## (openreview text is login-walled; formulas verified against
## google-deepmind/mctx/_src/policies.py, which we fetched)

Mechanism: root samples top-k actions WITHOUT replacement via Gumbel noise on prior logits
(argtop-k of g + logits); sequential halving spends the simulation budget over phases;
Q-values COMPLETED for unvisited actions by a mixed value estimate
(qtransform_completed_by_mix_value); the TRAINING TARGET is
    action_weights = softmax(prior_logits + sigma(completed_Q))
(verified verbatim in policies.py: softmax(root.prior_logits + completed_qvalues) after
invalid-action masking). Non-root selection is deterministic. The paper's claim (from the
abstract + reproductions): POLICY IMPROVEMENT GUARANTEED even at tiny budgets (n = 2-16
simulations) -- exactly the narrow-beam regime Kaveh specified.

ADOPTED for gct: the B-loss = CE(policy_head, softmax(prior + sigma*normQ_searched)) with
searched Q from our beam alpha-beta (real boards) on a per-batch subset; unsearched moves'
Q completed by the root value (mix form). sigma constants tuned, mctx defaults as start.

## TD-leaf (Baxter et al. 1999, KnightCap) — mechanism from the classic literature

Train the STATIC eval toward the backed-up minimax leaf value of a shallow search from the
same position: V(s) <- stopgrad(minimax_d(V)(s)) (gradient on the principal-variation leaf
in the original; root-consistency Huber in ours). The eval absorbs tactics below depth d,
which is what makes a NARROW beam safe. ADOPTED as the C-loss.

## Predictron (1612.08810) / VIN (1602.02867) — related, not adopted (per TreeQN's own
## comparison, which we read): Predictron is policy evaluation without control; VIN needs
## spatial full-state-space planning with convolutional dynamics -- neither matches a
## move-conditioned game tree.

## Quantized attention: prior art, and the one negative result that matters (2026-08-26)

Kaveh: "search some more the idea of quantized attention because I have a feeling this
gotta be somewhere out there... If it isn't out there in the way we want, then I want you
to search about WHY it's not out there. Have people tried it and has it failed?"

Answer: **every ingredient exists, the combination does not, and the motivation is the
real difference.** Three separate literatures, none of which is doing what we are doing.

### Line 1 — quantizing Q and/or K. Exists, mature, and is entirely about SPEED.

    Reformer (Kitaev+ 2020)          LSH-hash Q/K into buckets            O(n log n)
    Routing Transformer (Roy+ 2021)  k-means cluster queries AND keys     O(n^1.5)
    SMYRF (Daras+ 2020)              asymmetric clustering
    Transformer-VQ (Lin+ ICLR 2024)  VQ the KEYS, learned codebook        linear time
    VQ-TR, AVQ-Attention (2026)      VQ attention, adaptive code capacity

The machinery we want is therefore KNOWN TO TRAIN at scale -- reassuring for feasibility.
But every one of these papers reports perplexity and FLOPs. **None reports what the codes
MEAN.** Transformer-VQ is explicitly asymmetric and leaves QUERIES continuous. Routing
Transformer clusters queries but treats the cluster as a routing device, not a symbol.
[confidence: high on "the motivation is efficiency", medium on "no paper inspects code
semantics" -- established from surveys and abstracts, not by reading each full text.]

### Line 2 — quantizing for INTERPRETABILITY. Exists, but at the OUTPUT, not the query.

    Codebook Features (Tamkin+ 2023, arXiv 2310.17230)
        VQ bottleneck on attention-head OUTPUTS, before the residual stream, one codebook
        per head. Explicitly for interpretability and control. Codes fire on punctuation,
        syntax, lexical semantics, topics; better predictors of simple textual features
        than neurons. THIS IS ESSENTIALLY OUR LEVEL-2 OUTPUT QUANTISATION, already done.
    Discrete Key-Value Bottleneck (Trauble+ ICML 2023, arXiv 2207.11240)
        quantise to a key codebook, look up a value; for generalisation/continual learning.

### Line 3 — the critique, and the failures. This is the "has it been tried" answer.

  * **Hard attention DID fail, but for a reason that no longer applies.** Xu+ 2015 trained
    discrete attention with REINFORCE; it is non-differentiable, combinatorial, unstable,
    and soft attention beat it on image captioning. The field standardised on soft. But
    that failure was about the ESTIMATOR, not the idea -- straight-through + VQ (2017 on)
    is a different and far more stable tool, and is what we use. Reading the hard-attention
    literature as "discrete attention is a dead end" would be a misreading.
  * Jain & Wallace 2019, Wiegreffe & Pinter 2019: attention weights are not reliable
    explanations. Meister+ 2021: sparsity does not confer faithfulness.

### THE RESULT THAT SHOULD CHANGE OUR PRIORS

**"The Interpretability of Codebooks in Model-Based RL is Limited" (arXiv 2407.19532).**
Tested IRIS (VQ world model, 512 codes) on Crafter: **205,231 Grad-CAM heatmaps across 477
codes**, scoring each code's consistency as mean cosine similarity of its ResNet50 crop
embeddings. **The vast majority of codes sat only marginally above a random-crop baseline.**
Only a handful clustered cleanly. Their conclusion: **vector quantisation ALONE does not
produce interpretable codes; "latent semantic alignment is needed alongside
discretization".**

**Implication for us, stated before we see our own numbers:** we should EXPECT most of our
16 questions to be semantically inconsistent. Discretising the query buys a finite,
countable set of questions -- it does not buy meaning. Our `relation_alignment()` already
MEASURES alignment against python-chess ground truth but does not ENFORCE it; the paper's
prescription is precisely to add the enforcement.

Corroborating signal already in hand: in qat1, level-2 was the LEAST retained family under
magnitude pruning (**7.8%**, against 13.5% for level-1 and 59.4% for move concepts). If
quantized attention were producing high-value concepts, the readout would have kept them.

### WHAT IS ACTUALLY NEW HERE

  1. The QUERY is a discrete symbol from a small codebook and the code ID is the unit of
     interpretation -- "what is this token asking?" Efficiency papers quantise queries but
     never ask this; interpretability papers quantise outputs, not queries.
  2. Three constraints stacked at once: discrete question + top-k answer + quantized result.
  3. **External validation, which is plausibly WHY nobody has done this.** In NLP and vision
     there is no oracle for "what relation SHOULD head h encode", so the claim cannot be
     falsified. Chess has python-chess: attacks, defends, pins and blocks are computable
     ground truth. That is a structural advantage of the domain, not merely a different
     dataset -- it is what makes "question 5 means attacks" a testable statement.

Sources: arXiv 2001.04451 (Reformer), Routing Transformer, arXiv 2010.05315 (SMYRF),
arXiv 2309.16354 (Transformer-VQ), arXiv 2607.12789 (AVQ-Attention), arXiv 2310.17230
(Codebook Features), arXiv 2207.11240 (Discrete Key-Value Bottleneck), arXiv 2407.19532
(codebook interpretability is limited), Jain & Wallace 2019, Meister+ 2021.

## Aligning discrete codes to KNOWN chess concepts (2026-08-26) — Kaveh's proposal, and its lineage

Kaveh: "we take the known chess concepts and we penalize the discrete QKV codes that don't
align with these concepts... they should be expressible as an L1 combination of these
concepts, but the fewer concepts the better. Has it been done in the literature?"

**Yes -- every piece has, and one of them is in chess.** This is also precisely the
prescription of the negative result recorded above (arXiv 2407.19532: "latent semantic
alignment is needed alongside discretization").

### The four ingredients, all established

  **Network Dissection** (Bau+ CVPR 2017). Score each unit's alignment against a
  dictionary of labelled concepts (IoU). POST-HOC measurement, single concepts only.

  **Compositional Explanations of Neurons** (Mu & Andreas, NeurIPS 2020, arXiv 2006.14032).
  Explain a neuron as a LOGICAL FORMULA over primitive concepts (AND/OR/NOT), found by beam
  search, with the formula's complexity restricted. This IS Kaveh's "expressible as a
  combination, the fewer the better" -- with logic where he proposed L1. Still POST-HOC.
  A 2026 follow-up (arXiv 2511.20934) replaces the beam search with a guaranteed-optimal
  one, which matters because the composition space is enormous.

  **Concept Whitening** (Chen, Bei & Rudin, Nature MI 2020, arXiv 2002.01650). THE
  TRAINING-TIME MECHANISM. A drop-in replacement for batch norm that whitens the latent
  space and then ROTATES its axes to align with known concepts, so each concept's
  information flows through one designated node. Reported not to hurt predictive
  performance. This is exactly "penalise codes that don't align", built in rather than
  measured afterwards.

  **Concept Bottleneck Models** (Koh+ ICML 2020). Supervise intermediate units to BE known
  concepts, then predict from those units only.

### The chess precedent, which is close to what Kaveh described

**McGrath+, "Acquisition of Chess Knowledge in AlphaZero" (PNAS 2022, arXiv 2111.09259).**
They took **Stockfish's own evaluation-function components as the concept dictionary** and
trained **SPARSE LINEAR PROBES** from AlphaZero's activations to those concepts -- Kaveh's
proposal with the concept set drawn from Stockfish's eval terms rather than python-chess
predicates. Findings: material emerges early and then declines in importance while king
safety rises, "surprisingly human-like". But this is POST-HOC probing of a frozen network;
nothing was penalised during training.

### WHAT IS ACTUALLY OPEN

Searching did not turn up: **aligning DISCRETE (VQ/LFQ) codes to a known concept dictionary
AT TRAINING TIME with a sparsity penalty on how many concepts explain each code** -- still
less doing it to the QUERY codes of attention, i.e. constraining what a token may ASK. The
VQ-plus-concepts work that exists (arXiv 2602.02726, 2506.20040) is unsupervised concept
DISCOVERY, not alignment to a given dictionary.

So Kaveh's proposal is a new combination of proven parts, and our domain gives it an
advantage the vision work lacks: **python-chess computes the concept dictionary EXACTLY.**
Network Dissection's Broden labels are noisy and human-annotated; `is_pinned`,
`attackers`, `gives_check`, `is_capture` are ground truth.

### THE DESIGN TENSION, worth stating before building

Forcing codes to be sparse combinations of KNOWN concepts means we can only ever recover
KNOWN concepts -- the discovery upside is exactly what we would be regularising away, and
discovery is the point of the project ("mostly concepts that give the best moves"). Concept
Whitening handles this by aligning only SOME axes and leaving the rest free. Recommended
shape for us: align a designated SUBSET of code dims to the python-chess dictionary, leave
the remainder unconstrained, and report both -- the aligned part is auditable, the free part
is where anything new would show up. Partial alignment also gives a natural control: if the
free dims carry no extra predictive value, the known dictionary was sufficient, which is
itself a publishable finding.

We already have `relation_alignment()` in quantized_attention.py, which MEASURES
attention-vs-python-chess agreement. This line turns that metric into a loss.

Sources: arXiv 2002.01650 (Concept Whitening), arXiv 2006.14032 (Compositional Explanations
of Neurons), arXiv 2511.20934 (guaranteed-optimal follow-up), arXiv 2111.09259 (AlphaZero
chess concepts), Koh+ 2020 (Concept Bottleneck Models), Bau+ 2017 (Network Dissection),
arXiv 2407.19532 (VQ alone is not enough).

## Hierarchical quantization + quantized mixing for QGT (2026-08-27, Kaveh's ask)

RVQ (SoundStream/EnCodec/DAC; RQ-VAE): each codebook quantizes the RESIDUAL of the
previous -- coarse-to-fine hierarchy by construction. QGT amendment: levels are
RVQ-structured, giving (a) importance-ordered concepts (concept-budget = drop fine
levels), (b) per-position variable bitrate (adaptive search economics), (c) coarse-only
rolling (roll level-1, recompute residuals).

MoE discrete routing = quantized mixing generally: our question codebook is a router over
question-experts (cf. Mixture of Sparse Attention 2505.00315, expert-choice; Mixture of
Attention Heads 2210.05144). TRANSPLANT: MoE load-balancing auxiliaries are the proven
anti-collapse mechanism -- per-routing-decision, unlike LFQ's batch-global entropy, which
we measured failing per-position (the move-code degenerate state). Hierarchical routing
(global->sub-experts, 2508.07838) = auditable two-level mixing tree.

Learned graph coarsening (Graph-MoE lineage): quantized cluster assignment pooling 64
square tokens -> ~8 learned region super-nodes between levels; spatial hierarchy emerges,
domain-portable.

QGT design amendments adopted: RVQ-structured levels; MoE load-balancing on question
usage; optional square->region->board token hierarchy.

### Implementation-verified hyperparameters for QGT (read from sources, 2026-08-27)

ST-MoE (2202.08906, eq. 5-6 + Table 4, read from the PDF):
  router z-loss  L_z = (1/B) SUM_i (log SUM_j exp(x_j^i))^2,  c_z = 0.001
  total loss     L = L_CE + c_B*L_B + c_z*L_Z   (balance loss per Appendix A; Switch
                 lineage default c_B = 0.01)
  capacity factor train 1.25 / eval 2.0; router computed in float32.
  Footnote 4: z-loss ON ATTENTION LOGITS also stabilises without quality loss --
  directly applicable to QuantizedAttention's logits.
  Table 2/3 rulings: input-jitter and dropout stabilise but COST quality (skip);
  multiplicative parts (GEGLU/RMS scales) destabilise but buy quality (keep + z-loss).

vector_quantize_pytorch (installed, read from source):
  ResidualLFQ EXISTS with EnCodec structured quantizer dropout:
    quantize_dropout=True, quantize_dropout_multiple_of=4 (EnCodec's value, per the
    source comment), cutoff_index>=0 -- variable-bitrate concept hierarchy for free.
  LFQ defaults we already inherit: entropy_loss_weight=0.1, commitment_loss_weight=0.0,
  diversity_gamma=1.0; optional spherical=True (BSQ, 2406.07548).

QGT adoptions: question selector gets z-loss (c_z=1e-3) on question AND attention
logits + Switch balance loss (c_B=0.01, per-decision); levels = ResidualLFQ with
EnCodec dropout multiple-of-4; no jitter/dropout.
