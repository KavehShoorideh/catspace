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
