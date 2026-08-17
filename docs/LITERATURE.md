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
