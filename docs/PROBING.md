# Probing the nets — playbook and tools

Standalone CLIs in `tools/`, all consuming one convention: a **representation
file** (npz with `emb` (N,d) + row-aligned label columns), produced by
`tools/embed.py` from any encoder checkpoint (JEPA T1 or the frozen lc0 trunk).
Every tool prints `VERDICT` lines (journal rule: numbers only from script output).

## The ladder (cheapest first)

1. **Label-free spectral health** — `tools/probe_rank.py`
   RankMe (entropy of the singular-value spectrum) tracks downstream linear-probe
   accuracy *without labels* and is the standard first look at a JE representation
   [Garrido et al. 2023]. LiDAR refines it with an LDA scatter under a surrogate
   label, discounting variance that carries no signal [Thilak et al. 2024]
   (`--lda-labels`). A cliff in the eigenspectrum or effective rank sliding toward
   1 is dimensional collapse [Jing et al. 2022] — this repo also asserts on it
   during training (the collapse gate).
2. **Frozen-feature probes** — `tools/probe_linear.py`
   The standard SSL protocol: encoder frozen, L2-normalized features, a *linear*
   model + cosine-kNN, and a **group-aware split** (`--group gid` — rows from one
   game never straddle the split). Always read scores against the printed
   majority/shuffle baselines. Linear-vs-kNN disagreement is itself informative
   (linear ≫ kNN: directions exist but geometry is not clustered; kNN ≫ linear:
   local structure without linear separability).
3. **Representation comparison** — `tools/probe_cka.py`
   Linear CKA [Kornblith et al. 2019] between row-aligned representation files:
   trained-vs-init (how far did training move), trained-vs-frozen-trunk (did the
   JEPA learn something the trunk lacks), checkpoint-vs-checkpoint (still moving?).
4. **Calibration** — `tools/probe_calibration.py`
   Reliability + count-weighted ECE for every probability the planner multiplies
   (R_g, P_fall, destination masses). Run it before trusting any planner
   arithmetic — the paper calls this "the figure the whole planner arithmetic
   rests on". Use `--quantile` when the mass is skewed to low p.
5. **Causal ablation** — `tools/ablate.py`
   The Zhang–Nanda caveats apply: conclusions depend on the corruption and the
   metric, so both are explicit in the output. `dims` mean-ablates embedding
   dimensions against a frozen probe (mean, not zero — zero-ablation overstates
   dependence). `board` is the minimal-pair primitive from the paper's T2 keep
   test: remove the piece, re-encode, and check the model's own risk/destination
   heads collapse. Recognition that survives removal of the structure it claims
   to recognize is not causal — delete the atom.

## Standing rules of interpretation

- **Proxy metrics don't certify usefulness.** The SAE/transcoder literature's
  core lesson (SAEBench-era): sparsity/fidelity/interpretability scores do not
  guarantee downstream value; automated-interp metrics can fail to distinguish
  trained from random transformers. Atoms and features earn their place through
  the causal test and downstream deltas, nothing softer.
- **One setup per claim.** Patching/ablation results are evidence *under the
  stated corruption and metric*, not global explanations.
- **Baselines ride along.** Majority class, shuffled labels, random-feature
  controls — a probe without its chance floor is not a number.

## Sources

- [RankMe (Garrido et al., 2023)](https://arxiv.org/abs/2210.02885)
- [LiDAR: sensing linear probing performance in JE architectures (Apple ML, 2024)](https://machinelearning.apple.com/research/sensing-linear-probing)
- [Understanding dimensional collapse (Jing, Vincent, LeCun, Tian, 2022)](https://openreview.net/forum?id=f3g5XpL9Kb)
- [Towards best practices of activation patching (Zhang & Nanda, 2023)](https://arxiv.org/abs/2309.16042)
- [Attribution patching at scale (Nanda)](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching)
- [reptrix: representation-quality metrics library](https://github.com/arnab39/reptrix)
- [CE-Bench / SAE evaluation caveats (2025)](https://arxiv.org/pdf/2509.00691)
- [Automated interp metrics fail trained-vs-random (2025)](https://arxiv.org/pdf/2501.17727)
- [Transcoders beat SAEs for interpretability (Paulo et al., 2025)](https://www.themoonlight.io/en/review/transcoders-beat-sparse-autoencoders-for-interpretability)

## Quasimetrics — `tools/probe_quasimetric.py`

What a good learned asymmetric distance must show (IQE/PQE/QRL literature — value
functions of goal-reaching MDPs *are* quasimetrics):
- **Asymmetry where the domain is irreversible**: chess one-step backward distance
  should exceed forward (captures/pawn pushes can't be undone). Verdict: bwd/fwd
  ratio distribution.
- **On-policy monotonicity**: d(s_t → s_final) shrinks along real games; per-game
  spearman vs plies-remaining.
- **Triangle inequality**: violation rate on in-game triples. IQE satisfies it by
  construction (a nonzero rate = implementation bug); unconstrained heads report
  their honest rate — compositionality of costs is what planning relies on.
First run on the T1-IQE field (15 games): ratio 1.79, monotonicity +0.915
(100% of games), violations 0.00%.

## Probability fields — `tools/probe_field.py` (+ `fig_hazard.py`)

Forecast-verification standards (Gneiting & Raftery 2007): score with an ENSEMBLE
of **proper scoring rules** (log score, Brier — each individually minimized by the
true distribution; they discriminate differently, no single best), report **skill**
vs the base-rate forecast, and follow "**maximize sharpness subject to
calibration**" — entropy/sharpness claims only count after `probe_calibration`
passes. `fig_hazard.py` renders the paper's Fig-4 identity chain (λ → S → f with
the explicit never-bar → R) from ANY hazard source, and prints the Jensen gap
γ = E[ρ^k] vs ρ^E[k] — the discount mistake the paper warns about, made visible.

## Figures

Every probe emits its figure via `--fig` (`tools/figlib.py` style: validated
4-hue palette, single-hue sequential, recessive grid, thin marks; t-SNE captions
state "neighbourhoods, not global distances"). `probe_map.py` renders embedding
maps; `probe_rank.py --fig` is the Fig-3c diagnostics panel (spectrum, cumulative
variance, rank measures) and accepts several representation files for side-by-side
training-trajectory comparisons.

### Additional sources
- [IQE (Wang & Isola 2022)](https://arxiv.org/abs/2211.15120) ·
  [PQE / learnability of quasimetrics](https://github.com/SsnL/poisson_quasimetric_embedding) ·
  [QRL overview](https://www.emergentmind.com/topics/quasimetric-reinforcement-learning-qrl) ·
  [triangle-inequality inductive bias](https://arxiv.org/pdf/2002.05825)
- [Gneiting & Raftery 2007 — proper scoring rules](https://mpra.ub.uni-muenchen.de/45186/1/MPRA_paper_45186.pdf) ·
  [proper scoring rules for estimation & evaluation (2025)](https://arxiv.org/html/2504.01781v1) ·
  [forecaster's dilemma — extreme events](https://arxiv.org/pdf/1512.09244)
