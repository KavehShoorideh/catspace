# Is Region-Discovery Load-Bearing for the Latent Chess Planner? A Feasibility Analysis

## TL;DR
- **Unsupervised region discovery is a deferrable luxury, not a load-bearing component, for v1 of the planner.** The planning-utility payoff defensible today comes from (a) ~100–300 handcrafted, exactly-computable chess predicates learned into boxes, and (b) ω-conditioned kernel statistics estimated per-Elo-bin. Discovery earns its keep in exactly one place the classical vocabulary cannot name: **ω-conditioned "trap" regions** ("positions where 1400s blunder"), which is the strongest — and possibly only — pro-discovery argument.
- **Sparse autoencoders (SAEs) are the weakest of the three candidate-generators considered, but the criticisms mostly do NOT bite once features are reframed as candidate predicates fed to a kernel-leverage filter.** Non-canonicity, feature splitting, and dark matter are fatal for "true units of cognition" claims but neutral-to-helpful for a utility-filtered box lattice. The one criticism that survives the reframing is **feature absorption**, which silently corrupts predicate *extension* (the set of states a predicate fires on) and can fool the leverage filter — this needs an explicit defense.
- **The mathematics is sound and the data is (barely) sufficient at coarse/Elo-bin granularity on one GPU; it is decisively insufficient for per-individual kernels or large fine-grained kernels.** The scoping concern behind this analysis is correct about scope, not about impossibility: the pipeline is feasible if scoped to coarse regions × Elo-bins × within-stratum G-PCCA (Generalized Perron Cluster Cluster Analysis), and a handful of cheap kill-tests (days, not months) will settle every marginal item.

## Key Findings

1. **The deflationary baseline is strong.** McGrath, Kapishnikov, Tomašev, Pearce, Wattenberg, Hassabis, Kim, Paquet & Kramnik (PNAS 119(47):e2206625119, 2022, "Acquisition of Chess Knowledge in AlphaZero") showed a broad battery of human chess concepts — including Stockfish evaluation terms — are *linearly decodable* from an AlphaZero-style trunk; verbatim, "Linear probes applied to AlphaZero's internal state enable us to quantify when and where such concepts are represented in the network." This means most concept directions are obtainable by **supervised linear probing on exactly-labeled predicates** (python-chess computes passed pawns, open files, rook connectivity, king-safety proxies exactly and cheaply), skipping unsupervised discovery entirely for ω-independent structure.

2. **The SAE-critical literature (2024–2026) is now decisive on downstream-task underperformance.** Google DeepMind's mechanistic-interpretability team (Smith, Rajamanoharan, Conmy, McDougall, Kramár, Lieberum, Shah & Nanda, "Negative Results for SAEs On Downstream Tasks and Deprioritising SAE Research", DeepMind Safety Research, 26 Mar 2025) publicly deprioritized fundamental SAE research; verbatim: "Negative result: SAEs underperformed linear probes … we are deprioritising fundamental SAE research for the moment." Kantamneni et al. (ICML 2025, "Are Sparse Autoencoders Useful? A Case Study in Sparse Probing") found SAEs do not consistently beat simple baselines across data scarcity, class imbalance, label noise, and covariate shift.

3. **The chess/board-game evidence is the key empirical anchor and it is unflattering to SAEs.** Karvonen et al. 2024 (arXiv:2408.00113), Table 1: on the trained GPT, SAE board-reconstruction F1 was 0.85 (chess) / 0.95 (Othello) with coverage 0.48 / 0.52, versus **linear-probe reconstruction 0.98 / 0.99 and coverage 0.98 / 0.99** — probes beat SAEs on both metrics in both games. An independent replication found an SAE recovered only ~9 (later revised to ~33) of 180 probe-findable board-state features, and re-finds "simpler" center/corner features across seeds. Translation: SAE features on chess models *do* encode state properties (right type for regions), but they are an incomplete, seed-dependent subset of what supervised probes find.

4. **G-PCCA is the correct coarse-graining tool and it exists as maintained software.** Chess region-flow is non-reversible (a layered DAG across irreversibility strata), which breaks classical PCCA+. Reuter, Weber & Fackeldey's Generalized PCCA (G-PCCA, using real Schur vectors instead of eigenvectors) handles exactly non-reversible/non-equilibrium chains and is implemented in the maintained `pyGPCCA` package. The right decomposition is: **G-PCCA within-stratum (reversible-ish maneuvering) + explicit DAG layering across strata (irreversible captures/pawn advances).**

5. **Vector quantization (VQ) is the wrong primitive for a box lattice as-is, but multi-codebook variants rescue it.** Vanilla VQ partitions (no overlap, no nesting) and suffers codebook collapse. Product quantization (PQ) and residual VQ (RVQ) give the multi-granularity, overlap-capable structure a box lattice needs, and there is direct prior art for VQ-as-subgoals (Director's manager acts in a VQ-VAE discrete latent; OPAL, QPHIL, VQSkills).

6. **Data suffices for coarse per-Elo-bin kernels, not for per-individual anything.** The Lichess open database holds 7,949,495,674 standard rated games (verbatim from database.lichess.org; the Lichess End-of-Year Update 2025 states "over 7 billion rated standard games"), at roughly 100 million per month recently (Lichess forum growth data), with per-move clocks since 2017 and Elo labels. Multinomial arithmetic (~100 observed events per cell for 10% relative error) puts coarse kernels (R≈40 regions, 8 Elo bins, 4 clock buckets) within reach of ~1 month of data; per-individual kernels are off by 1–2 orders of magnitude and are infeasible.

## Details

### 1. SAE-criticism map with planning-consequence translation

The governing reframe: **interpretability is not the objective; planning utility is.** A discovered region is accepted iff conditioning on it reshapes the region-transition kernel toward outcome-relevant regions (kernel leverage) or improves reach-probability prediction / plan value. Human-legibility (via a behavioral label-verification layer that tests candidate English feature descriptions against model behavior) is a bonus filter, not the objective. Every criticism below is therefore evaluated twice: what it breaks *if the goal were interpretability*, and whether it still bites *for planning utility*.

| SAE criticism (source) | What it breaks IF the goal is interpretability | Does it bite for PLANNING UTILITY? |
|---|---|---|
| **Downstream underperformance vs linear probes** (Smith/Nanda et al. 2025; Kantamneni et al. ICML 2025) | Undermines SAEs as the primary lens | **Partially.** It means SAEs are unlikely to be the best candidate-generator, but they are not being asked to be a final probe — they are one of three generators feeding a utility filter. The consequence is *prioritization*: run supervised probes and difference-in-means first. |
| **Non-canonical units** (Paulo & Belrose 2025, "SAEs trained on the same data learn different features"; "Sparse Autoencoders Do Not Find Canonical Units of Analysis", ICLR 2025 — overlap as low as ~30% across seeds) | Fatal for "the true units of cognition" claims | **No — arguably irrelevant.** Candidate predicates need not be canonical; they need to survive the leverage filter and hold up on held-out confirmation. Seed-instability just means candidates should be pooled across seeds (a larger candidate pool is fine). |
| **Feature splitting** (Bricken et al. 2023; Leask et al. 2025) — one feature fragments into finer features as dictionary grows | Complicates the "atomic feature" story | **No — potentially helpful.** Splitting furnishes *multi-granularity* candidates, which is exactly what a box lattice wants (containment = implication, intersection = conjunction). Train SAEs at several widths deliberately. |
| **Feature absorption** (Chanin et al. 2024, "A is for Absorption") — a general feature stops firing on inputs absorbed by a specific child feature | Creates unintuitive gaps in feature extension | **YES — this is the one that survives.** Absorption corrupts predicate *extension*: a predicate C that "should" fire on a state silently doesn't. This distorts the conditional kernel P(next-region∣C) in a data-dependent way the leverage filter cannot distinguish from a genuine effect. Mitigation: define predicates by thresholded *decoder-direction projection* (a half-space), not by raw latent activation, and confirm extension against an exact python-chess predicate wherever one exists. |
| **Dead latents / high-frequency uninterpretable latents** (Smith et al. 2025 note auto-interp is misleading here) | Inflates auto-interp scores | **No.** Dead latents are dropped automatically (no leverage); high-frequency latents fail leverage or are near-constant. |
| **Dark matter / unexplained variance** (reconstruction never complete) | Limits completeness guarantees | **No.** Completeness is not required; what is required is *some* candidates that beat handcrafted leverage. Missing structure just caps the upside. |
| **Auto-interp / interpretability illusions** (Bolukbasi et al. "An Interpretability Illusion for BERT"; Makelov/Lange/Nanda subspace-patching illusion) — features look interpretable but aren't causal | Fatal for label trust | **No — sidestepped by design.** The English label carries no planning weight; labels are behaviorally verified *after* leverage selection, and leverage itself is a behavioral (kernel-level) test. |
| **Reconstruction–sparsity tradeoff / hyperparameter sensitivity** | Muddies "the" decomposition | **No.** Treated as a candidate-yield knob, not a correctness property. |

**Net:** the only criticism that must be actively engineered against is **absorption**, because it is the one that can inject a *spurious or masked* kernel-leverage signal. Everything else is either a prioritization signal (prefer probes) or is neutralized — sometimes even converted to an advantage — by the utility-filter framing.

### 2. Formalized discovery mathematics

**Setup.** Universal state embedding $x=\text{emb}(\text{FEN},\text{clocks})$. Discounted successor measure (the "cone") under own plan-latent $z$ and opponent embedding $\omega$:
$$\rho^{z,\omega}_\gamma(dg\mid x) \approx F_\omega(x,z)^\top B(g)\,\rho_0(dg),$$
the Touati–Ollivier forward–backward (FB) factorization (Touati & Ollivier 2021; Touati et al. 2023); $F,B:\to\mathbb{R}^d$ with rank-$d$ bottleneck. A **plan** is an option $\langle z,\beta_\text{term}\rangle$. Regions live in a **box lattice** (intersection = conjunction, containment = implication), stratified by irreversibility into a layered directed acyclic graph (DAG).

**Candidate-generation operators.** A candidate is a soft predicate $C:\mathcal{X}\to[0,1]$.
- **(a) SAE route.** Train SAE on backward space $B(g)$ (or frozen trunk activations). For latent $f$ with decoder direction $d_f$ and threshold $\tau_f$: $C_f(x)=\sigma\!\big(s(d_f^\top a(x)-\tau_f)\big)$ — i.e. a (soft) half-space. Boxes are intersections of half-spaces. Train at multiple widths to harvest feature-splitting as granularity levels.
- **(b) VQ route.** Product/residual VQ of $B$-space or plan-sketch latents; a codebook entry $c$ is a "kind of future" and $C_c(x)=\mathbb{1}[\,q(x)=c\,]$ (hard) or a soft assignment. Multiple codebooks (PQ) restore overlap/nesting a single codebook cannot give.
- **(c) MSM/G-PCCA route.** Fuzzy memberships $\chi_m(x)\in[0,1]$ from G-PCCA on the empirical region-kernel; the soft membership *is* a soft predicate and natively supports overlap.

**Region-transition kernel.** With regions/strata as nodes, $K_\omega(j\mid i,\text{clock})$ is the aggregate transition kernel, estimated per (i, j, Elo-bin $e$, clock-bin). Each row of the count matrix is a multinomial.

**Leverage functional (three definitions, cheap→expensive).**
1. **Conditional-kernel divergence (cheapest).** For stratum $s$ and opponent bin $\omega$,
$$\mathcal{L}_\text{KL}(C)=\sum_{j} u(j)\;D_{\mathrm{KL}}\!\Big(P(\text{next-region}=j\mid C,s,\omega)\,\big\|\,P(\text{next-region}=j\mid s,\omega)\Big),$$
with $u(j)$ an outcome-relevance weight (e.g. win-probability delta of region $j$). Estimable directly from counts; $O(R)$ per candidate.
2. **Reach-probability lift (mid).** $\Delta\text{AUC}$ or $\Delta$log-loss of a model of $P(\text{reach }G\mid x)$ with vs. without $C$ as a feature. Closer to the goal; needs a fitted predictor.
3. **Value-of-information in the planner (closest, most expensive).** $\text{VoI}(C)=\mathbb{E}\big[\mathcal{V}(z^\star\mid x,\text{regions}\cup\{C\})\big]-\mathbb{E}\big[\mathcal{V}(z^\star\mid x,\text{regions})\big]$, i.e. does adding $C$ to the region set improve the optimal-plan value against the ω-ensemble. Requires planner rollouts.

**Staged filter.** Screen thousands of candidates with $\mathcal{L}_\text{KL}$ (definition 1); promote the top few hundred to reach-probability lift (definition 2); confirm the final tens with planner VoI (definition 3) on held-out games. Definition 1 is cheapest to estimate; definition 3 is closest to the true objective.

**Statistics of the screen.**
- **Kernel estimation error.** Each kernel row is multinomial; the single-cell relative standard error is $\mathrm{RSE}(\hat p)=\sqrt{(1-p)/(Np)}$, so a target 10% relative error needs $Np\approx(1-p)/0.01\approx 100$ *observed events per cell* (standard binomial result, e.g. Casella & Berger, *Statistical Inference*; the identical $\sigma=\sqrt{p(1-p)/N}$ per-row formula appears in the MSM literature). Learning a whole row (k categories) to total-variation error $\varepsilon$ with confidence $1-\delta$ needs $n\ge\max(k/\varepsilon^2,\,(2/\varepsilon^2)\ln(2/\delta))$ samples (Canonne 2020, "A short note on learning discrete distributions", Thm 1: $\Theta((k+\log(1/\delta))/\varepsilon^2)$). Worked: k=10, ε=0.1 (TV), δ=0.05 ⇒ n≈1,000.
- **Detecting a leverage effect.** Comparing $P(\text{next}\mid C)$ vs $P(\text{next}\mid\neg C)$ over $R$ categories is a $\chi^2$ homogeneity test with Cohen's $w=\sqrt{\sum_i(p_{1i}-p_{0i})^2/p_{0i}}$; required $N=\lambda/w^2$ where $\lambda$ is the noncentrality for df $=R-1$, power $1-\beta$, level $\alpha$ (Cohen 1988, *Statistical Power Analysis for the Behavioral Sciences*, 2nd ed., Ch. 7; conventions small/medium/large $w$ = 0.1/0.3/0.5). For df $=4$ ($R=5$), $\alpha=0.05$, power $0.8$: $\lambda\approx 11.9$, so a *small* effect $w=0.1$ needs $N\approx 1{,}190$; a *medium* effect $w=0.3$ needs $N\approx 133$. Trap-detection effects are large ⇒ hundreds of transitions suffice per (C, cell).
- **Multiple testing under optional stopping.** Screening thousands of SAE/VQ candidates against a null of "no leverage" is a massive-multiplicity problem with arbitrary dependence (features are correlated). Use anytime-valid e-process machinery: form an e-process $E^{(k)}_t$ per candidate (a betting score against the no-leverage null accumulated as more transitions stream in), then apply **e-BH** (Wang & Ramdas 2022, "False Discovery Rate Control with E-values", *JRSS-B*), which controls the false discovery rate (FDR) under *arbitrary dependence with no correction*. Because e-processes permit optional stopping, use the **stopped e-BH** procedure (arXiv:2502.08539) so monitoring and per-candidate stopping are safe anytime — but respect the causal/global-filtration condition it identifies (no cross-stream information leakage), which holds here because streams are separated by (i, s, ω) cell.
- **Selection-induced bias (winner's curse).** Leverage estimates of *selected* candidates are upward-biased. Mandatory held-out confirmation: estimate leverage on a screening split, re-estimate the survivors' leverage on a disjoint confirmation split of games, and report only confirmation-split effect sizes.

### 3. MSM/G-PCCA transferability: from molecular dynamics to chess

**The mismatch.** The classical Markov State Model (MSM)/PCCA+ pipeline (Prinz, Wu, Sarich, Keller, Senne, Held, Chodera, Schütte & Noé 2011, *J. Chem. Phys.* 134:174105) assumes reversible-ish Markov dynamics, a meaningful stationary distribution, and a spectral gap defining metastable sets. Chess region-flow violates all four assumptions: it is (a) non-reversible (layered DAG), (b) non-stationary within a game, (c) has absorbing states (checkmate/draw), and (d) ω-dependent.

**The fix — and it is a real, cited fix.** Reuter, Weber & Fackeldey's **Generalized PCCA (G-PCCA)** (Reuter, Weber, Fackeldey, Röblitz & Garcia, *J. Chem. Theory Comput.* 14:3579, 2018; Reuter, Fackeldey & Weber, *J. Chem. Phys.* 150:174103, 2019) replaces the eigen-decomposition with a **real Schur decomposition**, explicitly handling non-reversible and non-equilibrium chains and cyclic/directed structure. It is implemented and maintained in `pyGPCCA` (msmdev/pyGPCCA). On the learning side, VAMP / VAMPnets (Mardt, Noé et al.; and state-free (non)-reversible VAMPnets, Chen–Sidky–Ferguson) provide the non-reversible/non-stationary Koopman-operator generalization for a learned coarse-graining rather than a spectral one.

**The correct decomposition (confirmed).**
- **Within-stratum:** maneuvering inside a fixed irreversibility layer (no captures/pawn moves) is approximately reversible and recurrent — this is the *sound* application of G-PCCA. Metastable sets = regions where within-region movement is fast/reversible and cross-region movement is slow.
- **Cross-stratum:** irreversible events (captures, pawn advances) define the **DAG layering** and should be modeled *by the DAG*, not by MSM spectral clustering. Do not ask one MSM to span strata; it will fail the Chapman–Kolmogorov / implied-timescale validation tests because the process is not Markovian across an absorbing boundary.
- **Data requirements are identical to §2's kernel estimation:** G-PCCA consumes the empirical region-kernel, so per-cell count requirements carry over unchanged. Poorly sampled transition cells (single-digit counts) are exactly what motivated uncertainty-aware coarse-graining (Bowman 2012, *J. Chem. Phys.* 137:134111); flag and exclude them.

### 4. Data-requirements roll-up (explicit arithmetic)

Assumptions: Lichess ≈ 100M standard rated games/month recently (7,949,495,674 total); a ~40-move game ≈ 80 ply; a game passes through ≈15–30 *distinct coarse regions* (region-crossing events), so ≈20 region-transitions/game; 8 Elo bins; 4 clock buckets; target 10% relative error ⇒ ~100 counts/cell.

| Pipeline stage | Quantity | Order-of-magnitude estimate | Feasible on one GPU box + laptop? |
|---|---|---|---|
| **SAE training on B-space / trunk activations** | positions | Karvonen-scale chess SAEs use millions of positions; a $d\!\sim\!512$–$4096$ dict on a small (Maia/lc0) trunk needs ~10–100M cached activation vectors | **Yes.** Caching 10M forward passes through a small residual net ≈ minutes–few hours; TopK-SAE training on cached activations ≈ a few GPU-hours. Days total. |
| **VQ / PQ training** | positions | comparable to SAE (10–100M) | **Yes**, similar budget. |
| **Coarse kernel** (R≈40, ~10 successors ⇒ ~400 nonzero cells × 8 Elo × 4 clock ≈ 12.8k cells) | transitions | 12.8k cells × 100 = 1.28M transitions ⇒ ÷20 ≈ **64k games**, but populate rarest cells ⇒ safety factor ~10 ⇒ ~0.5–1M games (≈ a few days of Lichess) | **Yes, comfortably** — even 1 month of one Elo bin (~12M games) over-covers. |
| **Fine kernel** (R≈1000, ~30 successors ⇒ ~30k nonzero cells × 8 × 4 ≈ 1M cells) | transitions | 1M × 100 = 100M ⇒ ÷20 ≈ **5M games** minimum; rare regions never reach 100 counts | **Marginal.** A few months of Lichess at Elo-bin granularity; expect a long tail of under-sampled cells requiring shrinkage/merging. |
| **Leverage screening + e-BH FDR** | candidates | thousands of candidates; each needs hundreds of transitions per (C, cell) for large trap effects (w≥0.3 ⇒ N≈133) | **Yes** for coarse; screening is cheap CPU/GPU. |
| **Per-individual ω kernel** | games per player | a coarse kernel needs ~10⁵ games; an active player has ~10³–10⁴ games total | **No** — off by 1–2 orders of magnitude. Per-Elo-bin (or per-small-cluster) is the correct granularity. |
| **Planner-utility A/B validation** | games of Maia-play/self-play per candidate set | to detect a small win-prob edge (~1–2%) at power 0.8 needs ~10³–10⁴ games per arm | **Marginal.** Feasible for a few candidate sets; not for thousands — hence the staged filter. |

### 5. Feasibility verdict

**(a) Clearly feasible solo (do this):**
- Per-Elo-bin **coarse** region-transition kernels (R≈30–50) with clock buckets.
- ~100–300 **handcrafted, exactly-computed chess predicates** (python-chess) learned into boxes in the frozen-trunk embedding + their kernel statistics.
- **Within-stratum G-PCCA** on coarse regions via `pyGPCCA`, with explicit DAG layering across strata.
- A **small SAE (d≈2k–4k) on B-space / frozen trunk** as a *candidate generator only*, with kernel-leverage filtering, e-BH FDR control, and held-out confirmation.
- The **FB cone / anchor-index (FAISS)** retrieval-decoding — orthogonal to discovery, independently feasible.

**(b) Marginal (gate behind kill-tests):**
- Fine-grained kernels (R≈10³): long tail of under-sampled cells; needs shrinkage and cell-merging.
- Large-dictionary SAEs (≥16k): the literature shows *width barely helps* probe/leverage performance (Kantamneni et al. found SAE width "unimportant" with near-zero slope), so the marginal candidate yield likely does not justify the cost.
- VQ route: only via **product/residual VQ**; single-codebook VQ is the wrong tool for a box lattice.

**(c) Infeasible / drop:**
- **Per-individual-ω kernels of any granularity** — data-starved by 1–2 orders of magnitude. Model individuals as a Bayesian offset on an Elo-bin (or behavioral-cluster) prior, never as their own kernel.
- SAEs as a *primary/canonical* feature basis or as a trusted-label source — the 2025 literature is decisive that they lose to linear probes on downstream tasks.
- Any expectation that discovery finds ω-*independent* structure better than handcrafting — it won't; that's a solved problem via probes.

**(d) The cheap kill-tests (days, single GPU/laptop):**
1. **Discovery-adds-nothing test.** Train a 4k-dict SAE on frozen lc0/Maia trunk activations over ~10M positions; screen every latent for kernel leverage against the leverage distribution of 200 handcrafted concepts. **Decision threshold:** if no SAE candidate beats the handcrafted set's 95th-percentile leverage (on the *confirmation* split), unsupervised discovery adds nothing at this scale — drop it for v1.
2. **Trap-region existence test.** Restrict to a low-Elo bin (e.g. 1200–1400); look for candidate predicates C with high $\mathcal{L}_\text{KL}$ toward loss-regions that are *absent or weak* in a high-Elo bin. **Threshold:** if ≥1 such ω-conditioned candidate survives e-BH at FDR 0.1 and held-out confirmation, discovery is load-bearing exactly here; if none, the pro-discovery argument collapses.
3. **Probe-vs-SAE bake-off.** For 20 handcrafted concepts with exact labels, compare box-in-embedding leverage from (i) difference-in-means/logistic probe vs (ii) best-matched SAE latent. **Threshold:** if probes match or beat SAEs on leverage (expected from the literature, and directly from Karvonen's 0.98 vs 0.85 reconstruction gap), fix probes as the default generator.
4. **G-PCCA validity test.** Run the Chapman–Kolmogorov / implied-timescale test within a single stratum vs. across strata. **Threshold:** if within-stratum passes and cross-stratum fails (expected), confirm the within/cross decomposition and never span strata with one MSM.
5. **Absorption audit.** For any selected SAE candidate that coincides with an exactly-computable concept, measure extension disagreement (false-negative rate of the SAE predicate vs the exact predicate). **Threshold:** if disagreement >~10%, replace the SAE predicate with the exact/probe predicate before it enters the kernel.

### 6. Prior-art flags and gaps

**Found (directly relevant prior art):**
- **SAEs on chess models:** Karvonen et al. 2024 (arXiv:2408.00113; ChessGPT/OthelloGPT board-reconstruction & coverage metrics, p-annealing); the "SAE finds only 9/180 board features" replication; Poupart 2024 **Contrastive SAEs for interpreting *planning* of chess agents** (lc0, `lczerolens`) — closest existing work to this document's intent; a 2026 "Tracing the Thought of a Grandmaster-level Chess-Playing Transformer" using sparse replacement layers/transcoders on lc0.
- **Human/ω-conditioned chess models:** Maia (McIlroy-Young, Sen, Kleinberg & Anderson, KDD 2020, arXiv:2006.01855 — per-rating nets; Maia 1900 predicts the exact human move 46–52% of the time, "Over half the time, Maia 1900 predicts the exact move a 1900-rated human played"), Maia-2 (Tang, Jiao, McIlroy-Young, Kleinberg, Sen & Anderson, NeurIPS 2024, arXiv:2409.20553 — skill-aware attention, "surpassing original Maia by almost 2 full percentage points" and reducing perplexity from 4.67 to 4.07 bits), Maia-3/Chessformer (2026), ChessMimic (2026, per-100-Elo transformers for move+clock+outcome with clock conditioning) — direct instantiations of the ω-ensemble / clock-conditioned π_ω.
- **VQ-as-subgoals/skills:** Director (Hafner et al. 2022, VQ-VAE discrete latent for the manager's subgoal space), OPAL, QPHIL (quantized planner for hierarchical implicit Q-learning), VQSkills, discrete diffusion skills — establishes VQ-latents-as-planning-subgoals as real prior art.
- **G-PCCA / non-reversible MSM:** Reuter/Weber/Fackeldey (2018/2019), `pyGPCCA`; VAMP/VAMPnets (Mardt & Noé) for the learned non-reversible generalization; Prinz et al. 2011 for MSM estimation/validation.
- **FDR machinery:** Wang & Ramdas 2022 (e-BH under arbitrary dependence), stopped e-BH (arXiv:2502.08539) for optional stopping.
- **FB representations:** Touati & Ollivier 2021, Touati et al. 2023, plus FB-ensemble uncertainty work (2025) matching the Bayesian ω-ensemble design.

**NOT found after multiple search rounds (explicit gaps — do not claim these exist):**
- **No application of MSM / PCCA+ / G-PCCA / Koopman coarse-graining to chess or to any discrete adversarial game.** Searches for "Markov state model chess", "Koopman operator games", "metastable states board games", "coarse-graining game dynamics" returned only molecular-dynamics, weather, epigenetic-landscape, and gene-regulatory applications. **MSM-on-chess-region-flow appears to be genuinely novel** — a contribution opportunity, but also means no precedent de-risks the transfer; the within/cross-stratum decomposition is a load-bearing assumption of this design and must be validated by kill-test #4.
- **No prior "kernel-leverage" concept-scoring functional** framed as conditional-kernel KL / reach-probability lift / planner VoI for chess region discovery was found; the closest analogues are TCAV-style concept scoring (Kim et al.) and McGrath's what-when-where probes, neither of which conditions on a *transition kernel*.
- **No work combining FB successor-measure factorization with opponent-conditioned region graphs** was found; FB is used for zero-shot RL, not for adversarial opponent-model-conditioned planning graphs.
- Matryoshka SAEs are *claimed* to mitigate absorption/splitting via nested dictionaries, but on the board-game metrics specifically the results "do not show significant differentiation" from ReLU/TopK/Gated in the low-L0 regime — so treat the absorption fix as **unverified for this domain**.

## Recommendations

**Stage 0 (this month) — build the deflationary baseline and the kill-test harness.** Compute ~100–300 exact python-chess predicates; learn their boxes in the frozen trunk embedding; estimate coarse per-Elo-bin kernels from ~1 month of Lichess. Stand up the leverage functional (definition 1) + e-BH + held-out confirmation. This is entirely in bucket (a) and is where planning utility actually comes from.

**Stage 1 (weeks) — run kill-tests #1–#4.** These settle every marginal item. Concretely: if kill-test #1 (discovery-adds-nothing) *fails to reject* (no SAE candidate beats the handcrafted 95th percentile), **defer all unsupervised discovery** and ship v1 on handcrafted + probed concepts. If kill-test #2 (trap-region) succeeds, **scope discovery narrowly to ω-conditioned regions only** — this is the one place it is load-bearing.

**Stage 2 (conditional) — only if Stage 1 says discovery pays.** Add a small (≤4k) B-space SAE and/or product-VQ as candidate generators, feed them through the *same* leverage/FDR/confirmation harness, and run absorption audits (kill-test #5) on every survivor. Use within-stratum G-PCCA to propose *dynamical* regions the SAE/probe routes miss.

**Thresholds that change the plan:**
- If per-Elo-bin coarse kernels show negligible ω-dependence in leverage, drop the ω-ensemble complexity for v1.
- If fine-kernel cells are >50% under-sampled after 3 months of data, abandon fine granularity and stay coarse.
- If SAE candidates never beat probes on leverage across two Elo bins, permanently retire the SAE route.

## Caveats
- **The MSM-on-chess transfer is unprecedented** (see gaps). The within/cross-stratum decomposition is a reasoned hypothesis, not a validated result; kill-test #4 is not optional.
- **Absorption is a genuine, unsolved threat** to any predicate defined by raw SAE latent activation; the decoder-projection + exact-predicate-confirmation defense is a mitigation, not a proof.
- **"~100 counts per cell" and the χ² sample sizes are derived from standard multinomial/power formulas, not from a chess-specific study.** A verbatim "hundreds of counts per state" rule was NOT located in the primary MSM sources (Prinz et al. 2011; Bowman/Pande/Noé 2014 book); it is a defensible derivation from the binomial RSE formula, not a cited theorem — state it that way.
- **Region-crossing rate (~20/game) is an assumption**, not a measured quantity; the coarse-kernel game counts scale inversely with it and should be re-derived once the empirical crossing rate is measured.
- **Matryoshka SAEs' absorption fix is unverified on board-game models**, so do not rely on it to rescue the SAE route.
- **Lichess ~100M/month is a recent figure**; older months are smaller, and per-move clocks exist only from 2017 onward, so clock-conditioned kernels draw on a smaller effective corpus than the full 7,949,495,674 games.
