# Milestone 1 Results — "Did we learn something?"
### 5×5 KRk: cone embedding learned from random play, with emergent plan tokens

**Setup.** KRk on a 5×5 board: 7,040 white-to-move states, all forcibly winning, max DTM 19 plies, exact ground truth for everything (retrograde DTM, exact transition kernel, exact successor measure). Learning signal: **random-play games only** — no labels, no concepts, no optimal play anywhere in training. Method: empirical transition matrix from sampled games → discounted successor measure M̂ → rank-d SVD factorization M̂ ≈ F·Bᵀ (the exact tabular analogue of a Forward-Backward representation). Plan target: the *region* G = {states with DTM ≤ 3} ∪ {mate} — "a mate somewhere in there" — via z_G = Σ_{g∈G} B(g). Engine: one-ply greedy cone-steering, score = E_black-reply[F(s′)·z_G].

## Headline results

| Policy | Mate rate (1,000 games vs random black, 100-ply cap) | Mean plies to mate |
|---|---|---|
| Random white (baseline) | 3.7% | 13.4 |
| Learned, 500 games (59% state coverage) | 50.9% | 10.2 |
| Learned, 2,000 games (96%) | 83.3% | 5.7 |
| Learned, 8,000 games (100%) | 92.5% | 5.1 |
| **Learned, 32,000 games** | **95.5%** | **5.1** |
| Exact-dynamics reference ("infinite data") | 94.1% | 4.9 |
| DTM-optimal ceiling (bug-fixed) | 100% | 3.6 |

**Ablation:** removing the mate-in-1 rule bonus changes nothing (95.5% → 95.5%). The learned reach field alone finds the mates. The learning curve is clean and saturates near the exact-dynamics reference.

**VQ plan tokens (K=32 k-means on cone shapes F):** all 32 codes used, usage perplexity 30.2 (no collapse). The codes self-organize by mate-distance without ever seeing DTM: cluster mean-DTM spans 1.0 → 8+ monotonically, with the three tightest clusters (mean DTM ≈ 1.0) being recognizable mating-pattern families and distinct mid-game maneuvering codes above them.

**Filmstrip (out/filmstrip.md):** a sampled game from DTM=13 reads as legible intent — plan token transitions track the phases, box area shrinks 6→3→4, reach score rises monotonically 0.06→0.34, mate on ply 7. "Watching it think" works at this scale.

## Gate accounting (honest version)

Pre-registered G-M1: (a) rank-64 relative reach error < 5% — **FAIL** (0.41; still 0.45 at d=128 after deflating the draw column). (b) ≥3 distinct embedding dims with |ρ|>0.5 against ground-truth concepts — **FAIL** (1 of 3; post-hoc linear probes on the learned embedding give moderate R²: rook-bk-distance 0.38, box-area 0.32, DTM 0.27). (c) engine ≥5× random baseline — **PASS at 26×**, near-ceiling, robust to ablation.

**Diagnosis of (a):** the gate metric was mis-registered. Global reach *values* genuinely need high rank (heavy spectral tail), but greedy planning only consumes *local move ranking* — comparing ~20 sibling successors — which low rank handles easily (the 95.5% engine is the proof). This is a real finding, not spin, and it sharpens Risk R1 for full chess: **rank-limited FB should be expected to support short-horizon/greedy steering but not long-horizon global value comparison**; deep plans will need the receding-horizon structure (concrete near field, coarse far field) rather than trusting far-field F·B values pointwise.

**Diagnosis of (b):** single-dim correlation was the wrong operationalization (SVD dims are rotation-arbitrary), but even the friendlier linear-probe audit says concept structure is *partial and distributed*, not clean, at this scale — while the VQ layer, which is the mechanism actually proposed for concept discovery, organized strongly and legibly. Tentative reading: discrete bottlenecks recover legible structure that raw linear axes don't display.

## What was demonstrated, plainly
1. **Learning is possible and useful:** from nothing but random games, a rank-64 factorized cone supports an engine at 95.5% mate rate (26× random, near the 100% optimal ceiling), with a clean data-scaling curve.
2. **Concepts emerge without being coded:** a discrete plan-token vocabulary self-organizes by mate-distance; ground-truth concepts are moderately linearly decodable; per-ply token streams read as coherent plans.
3. **Two pre-registered metrics failed for diagnosable reasons** — one mis-chosen metric (global-value error vs. the move-ranking planning actually uses), one partially real (concept emergence is present but weaker/more distributed than hoped at toy scale).

## Decision needed (why this comes back to you rather than proceeding)
Under the fail-fast rules, a formally failed gate can't be waved through unilaterally. The utility half of the thesis passed decisively; the legibility half passed partially (VQ strong, linear structure moderate). Options: **(i) proceed to Milestone 2** treating (a) as metric mis-registration (re-register the gate on move-ranking + calibrated region-reach) and (b) as "VQ is the concept mechanism, probes are diagnostics"; **(ii) one more toy iteration first** — add the actual VQ *bottleneck* into a learned (gradient-trained, not SVD) F/B to test whether forcing plans through discrete tokens *sharpens* concept structure, which is the architectural claim anyway; **(iii) stop** if partial legibility at toy scale doesn't clear your bar. My recommendation is (ii) — it directly tests the load-bearing architectural claim (bottleneck ⇒ concepts) that SVD, which has no bottleneck pressure, structurally cannot test — but the call is yours.

## Reproduction
`domain.py` (board, retrograde DTM), `learn.py` (chain, sampling, rSVD-FB), `experiment.py` (main run + gate), `diagnostics.py` + `sklearn_free.py` (post-hoc). Total runtime ≈ 4 minutes on one CPU core. Figures: rank_probe.png, learning_curve.png, concept_audit.png; filmstrip.md.
