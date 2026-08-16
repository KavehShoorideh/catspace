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
