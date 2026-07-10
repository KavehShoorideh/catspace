# Milestone 1 Results — v3 (CORRECTED)
### Supersedes RESULTS-v2. One claim retracted, three stronger findings added.

## Correction first

**Retracted:** RESULTS-v2's "6/6 mates against optimal play." Kaveh caught the tell: several games mated in *fewer plies than DTM*, which is mathematically impossible against optimal defense (DTM is defined by maximal delay). Root cause: a sign flip in the opponent code — "optimal" black was choosing the reply that *minimizes* white's distance-to-mate, i.e., actively running into the mating net. It did capture hanging rooks correctly (so the random-baseline draws were genuine), but between captures it cooperated.

**Unaffected:** the 95.5% vs random black, the learning curve, the VQ token structure, the concept probes — none of those used the buggy code.

## Corrected findings

### 1. The cone is opponent-conditioned — a cone learned under the wrong opponent fails
With true optimal defense (max-DTM black, captures hanging rooks), the original engine — whose cone was estimated from *random-play* dynamics — collapses: 0/6 in the viewer games; ~45% with an improved readout (immediate-mate taking, stalemate refusal) over 300 starts. Random white: 0% (rook captured within a few moves). The successor measure is a property of *both* players' dynamics; steering with the wrong opponent's cone is planning in the wrong geometry. This is the ω-thesis of the full formalization, reproduced from below in a 7,040-state toy — opponent conditioning is load-bearing, not an extension.

### 2. Policy iteration with an opponent curriculum fixes it: 45% → 86% vs true optimal defense
Six rounds; each round: white = ε-greedy on current reach scores, black = ε_b-optimal with ε_b annealed 1.0 → 0.0; transitions accumulate; cone re-estimated; scores refreshed.

| round | black ε | vs-OPTIMAL mate rate | white-moves / optimal |
|---|---|---|---|
| 0 (pure random data) | 1.00 | 0.453 | 1.19 |
| 1 | 0.50 | 0.613 | 1.12 |
| 2 | 0.25 | 0.647 | 1.10 |
| 3 | 0.10 | 0.700 | 1.10 |
| 4 | 0.00 | 0.817 | 1.11 |
| 5 | 0.00 | **0.863** | **1.11** |

Mates now arrive at ≥DTM always (e.g., start DTM 17 → mate in 9 white moves ≈ 17–18 plies), converting ~11% slower than perfect play. The annealing schedule matters because with a fully adversarial opponent from the start, random exploration never reaches mate — the curriculum keeps mate-region data flowing while the dynamics harden.

### 3. Neural FB generalizes to never-seen states (the requested MVP demonstration)
Protocol: 15% of states held out — they never appear in any training pair, in any role. Two MLPs (F, B; 77→256→256→32) trained by InfoNCE on geometric-horizon future pairs from the same filtered random-play experience given to the tabular learner.

| evaluation at HELD-OUT states | neural | tabular (same data) |
|---|---|---|
| reach-ranking spearman (holdout / train) | **0.46 / 0.41 — no gap** | 0.08 / −0.15 |
| engine from 400 held-out starts, vs random black | **99.8%** | 64.7% |
| same, vs optimal black | 9.5% | 16.0% |

Familiar concept family, unfamiliar exact position: the network ranks and plays from unseen states with zero degradation. The 9.5% vs optimal is expected — this cone was learned under random-black dynamics; finding 2 is the fix, and combining the two (neural FB inside the PI loop) is the natural Milestone 2 core.

### 4. A readout lesson that cost an afternoon: score conventions matter as much as the embedding
Two players consuming the *same* learned reach field differed 24% vs 99.8% mate-rate on uniform starts, entirely because one scored a rook-capture outcome as neutral (0.0) and the other as the 0.1% reach-quantile (catastrophic). The embedding knew hanging the rook was bad; the neutral readout never asked. With aligned conventions, the "weak hard tail" vanished: 100/100 from the hardest starts (box ≥ 15, DTM ≥ 13). Design consequence for the real system: the planner's outcome-scoring conventions must be pinned in one place and tested (this is the toy's version of the TerminalClassifier single-source-of-truth contract).

## Atlas (atlas.png)
Six panels over the neural embedding (PCA to 2D, 32% variance — labeled distortion caveat applies): colored by box area, DTM, VQ token, and reach with held-out states overplotted as triangles (they embed exactly where they belong — visual generalization); plus the requested boxing story: three games unboxed → boxed → mate as trajectories on the box-area map, and the per-move curves of box shrinking while reach rises (mates in 10–13 white moves from DTM 13).

## Interactive viewer (thought-viewer-vs-optimal.html — regenerated)
Six PI-engine games and three random-engine games, all against TRUE optimal defense. One honest PI failure included (hung rook from DTM 15, consistent with the 86% rate). Game lengths now satisfy plies-to-mate ≥ DTM everywhere.

## On "how many concepts can KRk support?"
The atlas uses K=16 (down from 32) — cluster coherence, not choice paralysis, should set K, and a domain this small honestly supports a vocabulary in the low tens. The worry that KRk is concept-poor is correct and is the argument for KRRk next: two rooks add ladder/lawnmower mates, cut coordination, rook sacrifice into the KRk stratum (a real irreversibility stratum drop) — a materially richer vocabulary on a still-exact domain.

## Files
Corrected/new this round: `exp_policy_iteration.py`, `gen_ui_data_pi.py` (fixed defense), `neural.py`, `exp_generalization.py`, `atlas.py`, `atlas.png`, regenerated `thought-viewer-vs-optimal.html`. RESULTS-v2 retained for the record; this document supersedes it.

---

# Addendum (v3.1): regions as areas, and the KRRk stratum

**Region maps replace dot clouds.** `region_map.png` (KRk): plan-token *territories* via grid-majority vote with boundaries and mean-DTM labels; the two attractors of one cone as fields — mate-flow (F·B[MATE], amber contours) vs draw-flow (F·B[DRAW], red contours) — with the mate doorstep (DTM≤2) and highest-draw-risk states marked; and the tug-of-war: from the same DTM-15 start vs optimal defense, the learned game (mate in 9) climbs the mate gradient while the random game (draw in 5) is dragged into the draw sink. The "opponent navigates toward draw, we navigate toward mate" picture, literally.

**KRRk (two rooks) — the richer domain, with real strata.** Union chain: 50,980 KRRk states + 7,040 KRk states; a rook capture is an irreversible stratum drop (a chute). All KRRk positions forcibly winning, max DTM 9 plies. Stratified DTM computed KRk-first, then KRRk retrograde with capture edges feeding KRk values — black's optimal defense automatically weighs "grab a rook iff the one-rook mate is slower."

**Training: curriculum from round 0, no pure-random diet** (per the directive that random self-play teaches too little). White ε-greedy on current scores vs black ε-optimal annealed 0.7→0:

| round | black ε | vs-OPTIMAL mate | moves/opt | stratum-drop rate |
|---|---|---|---|---|
| 0 | 0.70 | 0.540 | 1.17 | 0.447 |
| 2 | 0.20 | 0.933 | 1.14 | 0.067 |
| 4 | 0.00 | **0.977** | **1.11** | **0.023** |

The stratum-drop column is the emergent-concept headline: games where the engine loses a rook fall 45%→2.3% — rook safety learned without ever being named.

**`krrk_region_map.png`:** (A) both strata in one cone geometry — two distinct clouds with observed capture chutes drawn between them; (B) 24 token territories labeled by mean DTM; (C) the learned geometry vs a never-shown concept: two-rook box area as smooth filled contours; (D) tug-of-war from the same DTM-8 start — learned stays in the KRRk cloud and mates in 8, random hangs a rook, falls down the chute into the KRk cloud, and draws in 12. **`krrk_filmstrip.png`:** the learned game as rendered boards with per-move DTM/box/token.

On concept count: KRk honestly supports a low-tens vocabulary; KRRk's 24 territories separate cleanly by DTM and formation — the "more pieces → more concepts" intuition confirmed at the first opportunity.

---

# Addendum (v3.2): KRkn — the first two-sided domain

**K+R vs k+n on 5×5** (158,232 states + the KRk stratum): black finally has a piece, and with it counterplay — knight forks, pins, strategic rook hunting — and, for the first time, **genuine game-theoretic draws: 39.5% of positions cannot be won by perfect white** (max DTM in won positions: 43 plies; R-vs-N is a hard conversion, just like real chess). Strata: knight captured → KRk sub-game; rook captured → dead draw (k+n cannot mate).

**Bootstrapping failure and fix.** The KRRk curriculum failed here (17.7% conversion): the knight hunts the rook, so weak white almost never survives to mate and the cone never sees mate mass. Fix: **reverse-start curriculum** — 70% of training games start from won states with DTM ≤ cap, annealed 5 → all, so mate signal propagates backward (an oracle curriculum at toy scale; the full-scale analogue is starting from tablebase-adjacent wins).

| after 8 rounds vs OPTIMAL defense | value |
|---|---|
| conversion from won positions | 48.7% (from 17.7% pre-curriculum) |
| rook lost to the knight | 27.7% (round 0: 91.7%) |
| **wins that trade the knight first (via KRk)** | **91%** — "simplify when winning," never named |
| WIN/DRAW classification AUC of the learned reach field | 0.702 (chance 0.5; never labeled) |

**Readout-depth probe (the honest frontier).** Bellman backups of increasing depth on the *same* learned field: rook-loss collapses 28% → 5%, but conversion *drops* 49% → 27% — the field's safety signal propagates cleanly, its far-field mate signal is too noisy to support 43-ply conversions. Deeper search on a noisy field buys caution, not wins. **This is the empirical case, discovered rather than assumed, for the receding-horizon architecture** (concrete near field + coarse far field): a two-sided adversary with long conversion horizons is exactly where greedy readouts and naive deep backups both fail.

**Map (`krkn_region_map.png`):** (A) ground-truth won/drawn territories with the learned reach contours over them — the frontier the field partially discovers (AUC 0.70); (B) fork-danger territory (6.8% of states, computed as "black has a K+R fork available next move") — the first genuinely adversarial trap region, the toy ancestor of ω-conditioned trap concepts; (C) strata clouds with knight-capture chutes; (D) the same engine facing two truths — mating from a won start, and a drawn start where optimal black takes the rook because the position never was winnable.

**Rung ladder to date:** KRk (7k states, one-sided) → KRRk (51k, strata) → KRkn (158k, two-sided, real draws). Each rung surfaced one architectural lesson: opponent-conditioning (KRk), stratum structure + concept richness (KRRk), and now the necessity of far-field coarseness under long adversarial horizons (KRkn).

---

# Addendum (v3.3): shallow search, the goal region, and the three readout regimes

**Q: how is the "DTM≤3" goal programmed?** As a task-time query, not a training signal: G = {s : DTM(s) ≤ 3} ∪ {MATE}, z_G = Σ_{g∈G} B(g), engine scores moves by F(s′)·z_G. The field is oracle-free; the goal vector was not. `goal_region.png` shows where G lives on the map (the amber crescent at the mate edge of the won territory).

**Ablation verdict: the oracle region was never load-bearing.** z = B[MATE] alone — the pure absorbing state, zero oracle — matches or beats the oracle region at every depth (70.0% vs 67.7% conversion at 3-ply). **Adopted: the goal is henceforth B[MATE]; the oracle leak in the task spec is gone.**

**Q: does a simple ~3-ply search close the gap to exact DTM?** Results on KRkn won-starts vs optimal defense (minimax readout: MIN over replies, MAX over own moves, learned leaves; k backups ≈ (2k+1)-ply):

| search plies | conversion | exact-DTM games | moves/optimal | rook-lost |
|---|---|---|---|---|
| 1 (greedy, minimax reply model) | 0.685 | 0.345 | 1.145 | 0.045 |
| **3** | **0.700** | **0.385** | **1.108** | **0.003** |
| 5 | 0.033 | 0.033 | 1.000 | 0.000 |
| 7–13 | ≤0.030 | = conversion | 1.000 | 0.000 |

Three discoveries in one table:

1. **The readout's opponent model was worth 20 points by itself.** Training-time eval used MEAN over replies (the training mix) and scored 48.7%; switching the same field's readout to MIN (matching the actual minimax opponent) gives 68.5% at depth 1 and rook-loss 28%→4.5%. The ω-mismatch was at the *readout*, not only in the data — the formalization's max/expectation asymmetry, cashed out empirically.

2. **Shallow search helps; deep fixed-horizon minimax on a noisy field collapses pessimistically.** At ≥5 plies conversion crashes to ~3% (only in-horizon forced mates survive; note moves/optimal = 1.000 exactly — those are the *only* games it wins). Mechanism: repeated MIN over noisy leaves lets model-black steer every line to the field's pessimistic floor, so root moves become indistinguishable — the mirror image of the expectation-backup result in v3.2, where repeated MEAN smeared the mate signal into passivity. **Three regimes, now all measured: expectation-deep = safe but passive; minimax-deep = pessimistic collapse; shallow minimax (1–3 ply) on the learned field = best of both** (70% / 0.3% rook-loss / 1.11 tempo). This triple is precisely the argument that depth must be *selective* (near-field concrete, far-field coarse), not full-width.

3. **Exact-DTM play "in most cases": not yet.** 38.5% of conversions are tempo-perfect at 3-ply; the rest carry ~11% overhead. Closing the remainder needs a better field (neural generalization, more targeted data, higher rank) or selective tree search with the field as move-ordering — not more full-width depth, which the table shows is counterproductive. Caveat: the "search" here is global fixed-horizon backup (equivalent in value to per-position full-width search of that depth, but computed for all states at once); a selective alpha-beta was not tested.
