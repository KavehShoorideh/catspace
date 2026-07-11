# Research journal

Running lab notes, newest entry last. Each entry: what was done, wall-clock
timings, verdicts (copied verbatim from experiment output), and interpretation.

---

## 2026-07-11 — package rename; eval-head representation ablation (design)

**Rename.** Package `latentchess` -> `catspace` to match the repo (commit
`e58b99a`). 85 fast tests pass. Venv script shebangs still pointed at the old
repo directory name and were fixed in place.

**Question (Kaveh):** why do the eval heads read only F(s) — why not B(s) too?
Resolution: F-only is the hypothesis under test (FB theory says any reward's
value is linear in F; B enters on the *goal* side of the dot product), but
B-only and F++B probes are the natural controls. Also: the checkpoint's stored
`zgoals` give a **zero-label** eval readout F(s)@(zMATE_W - zMATE_B) — the
no-training floor that says how much eval the FB geometry already carries.

**Readout table (pre-registered):**

| comparison | reading |
|---|---|
| F >> B on DESC_AUC | value lives in forward/omega structure (hoped) |
| B ~ F | outcome info is static board features; forward training adds nothing to eval |
| B > F | red flag: InfoNCE training is *losing* outcome-relevant info |
| FB > F | F loses value info that B keeps |
| BASE ~ trained probes | geometry already carries eval; labels add little |

**Implementation.** `train_eval_heads.py --repr {F,B,FB}` + BASE_AUC/BASE_SPEAR
baseline verdicts; probes are unchanged 2-layer MLPs (d_in doubles for FB).
Note: the normative head trains on lichess `[%eval]` annotations already in the
shards — this experiment is NOT blocked on the Stockfish labeling run
(label_stockfish.py remains for coverage + the deep audit).

Data: `lichess_db_standard_rated_2019-01.prefix1gb` shards (12 shards, ~11.0M
rows, 1M rows/shard); model `data/derived/lichess_fb.pt` (cosine-InfoNCE fix,
gamma 0.98). Holdout = game_id % 50 == 0.

---

## 2026-07-11 — zgoals were never saved (interrupted-run bug); slopes recovered

While wiring the baseline: the checkpoint's `zgoals` dict was EMPTY. Cause:
`lichess_fb.pt` is the step-2000 PERIODIC save (train_lichess_fb.py:216); the
run was interrupted before the final save that attaches zgoals (line 223) —
so the post-cosine-fix REACH_SLOPE verdicts also never printed, and
policy_fb/arena_real would have crashed on first use. Rebuilt zgoals from the
existing checkpoint (no retrain): 2048 checkmate finals per side, 1.8s build,
atomic re-save (tmp + os.replace; noted save_ckpt itself is NOT atomic).

**Recovered verdicts** (reach_slope = mean per-game spearman(ply, F@z), 200
holdout games per condition, 9.0s CPU):

    REACH_SLOPE_WON=+0.713  REACH_SLOPE_LOST=+0.650   (z = zMATE_W)
    DIFF_SLOPE_WON=-0.201   DIFF_SLOPE_LOST=-0.365    (z = zMATE_DIFF)

Reading: the pre-fix pathology (-0.92 on both, norm-shrink artifact) is gone;
reach toward mate now RISES through games. But it rises for winners AND
losers — the shared "generic finality" component. Along MATE_DIFF the win/loss
separation exists (winners 0.16 less negative) yet both slopes are negative —
suspect a side-to-move artifact (MATE_W finals are all black-to-move-mated
positions, so the stm plane may dominate the direction). Open question; the
probe AUCs below are the cleaner measure of outcome signal.

---

## 2026-07-11 — eval-head ablation, first result (repr=F)

Timing: 3m49s/run (MPS, 2 epochs x 12 shards x 400k row cap; holdout report
included). 224,326 holdout rows, 19,891 with lichess [%eval].

    repr=F  VERDICT DESC_AUC=0.565 DESC_ACC3=0.354 (majority 0.480) NORM_SPEAR=0.236

Weak. The frozen F(s) probe barely separates won from lost games (AUC 0.565)
and correlates 0.24 with Stockfish winprob. Caveat before condemning the
representation: the FB model has only 2000 training steps. The B/FB controls
(rerun after a device bug in my BASE-baseline code: zdiff loaded on MPS vs
CPU embeddings; B's probes trained fine but its report crashed) will say
whether this is representation or training budget. DESC_ACC3 below majority
is threshold miscalibration of the fixed 0.45/0.55 cut, not extra signal loss
— AUC is the honest number.

---

## 2026-07-11 — M1.5 kickoff: meet-in-the-middle decomposer on real boards

New `catspace/planner/decompose.py`: recursive geodesic-midpoint decomposition
over the FB embedding. Hop s->g splits at the pool waypoint maximizing
min(F(s)@B(m), F(m)@z_g) — both legs cosines (the cosine fix is load-bearing
here too: it's what makes the two legs comparable inside the min). Give-up
rules exactly as agreed in the M1.5 design, reusing plans.py's BlockReason
vocabulary: no_midpoint (hard-not-long), unlikely_territory (floor), dry_out
(2 low-gain splits), budget (depth cap; anytime). Pool F is embedded under
the PLANNER's omega — "can I route through m" is about the planning player,
not whoever reached m in the source games. 9 tests on synthetic arc geometry
(unit circle => reach = cos(arc distance); waypoint must be the arc middle).

`experiments/decompose_demo.py` on real holdout positions (CPU, MPS busy):
20,800 rows embedded in 14s; decomposition itself 0.1 ms/start. First
calibration attempt was WRONG (compared ply to the global max ply, not each
game's own end => n_near_win=2). Fixed with a per-game np.maximum.at group-by
over one full shard: tau_exec = median reach of positions <=10 plies before
the end of won holdout games = 0.3596 (n=1542); tau_floor = q10 of start
reaches = 0.0954.

    VERDICT FRAC_IMPROVED=0.730 MEAN_GAIN=0.2871 FRAC_EXECUTABLE=1.000 MEAN_WAYPOINTS=0.73
    waypoint ply: mean 64.3 vs start ply mean 30.2 (pool mean 44.2)

---

## 2026-07-11 — eval-head ablation, full table

B and FB rerun after the zdiff device fix (B 3m48s, FB 6m52s — FB embeds both
encoders; F/B runs share one). Same 224,326-row holdout, 19,891 annotated.

| probe repr | DESC_AUC | NORM_SPEAR |
|---|---|---|
| F (omega-cond., headline) | 0.565 | 0.236 |
| B (board-only control)    | 0.570 | 0.254 |
| F++B (concat control)     | 0.579 | 0.248 |
| zero-label F@zMATE_DIFF   | 0.545 | 0.184 |

**Reading (pre-registered table, row "B ~ F"):** the outcome signal the
probes find is NOT forward/omega structure — the board-only B embedding
matches (marginally exceeds) F on both metrics, FB adds ~0.01 AUC over F, and
everything sits barely above the zero-label geometry readout. At 2000
training steps the field simply doesn't yet encode much position quality
anywhere: probes are reading residual static board features that both
encoders happen to retain. This is a budget finding, not (yet) a
representation verdict — the obvious next lever is a real training run
(20-50k steps instead of 2000) and a re-run of this exact ablation; the
comparison harness is now push-button (~4 min/arm on MPS).

Also of note: NORM_SPEAR > DESC-derived signal everywhere — Stockfish
winprob is an easier (less noisy) target than game results, as expected.

---

## M1.5 kickoff readings (decompose demo, continued)

Readings below. (1) the arc property shows up on real data — chosen waypoints sit
~34 plies later in games than the starts, i.e. the decomposer picks genuinely
intermediate, endgame-shaped stepping stones (e.g. ply-22 middlegame routed
via a ply-68 R+P endgame); (2) 73% of middlegame starts improve their
bottleneck by splitting, mean gain +0.29 in cosine reach; (3) ONE waypoint
always sufficed and no give-up rule ever fired — the 2000-step field is
generous (best min-leg through a 20k pool ~0.55 >> tau_exec). Executability
here is still reach>=tau, ESTIMATED not verified; the MC-rollout leaf check
("a real path") is the next layer and is where this gets kept honest.

---

## 2026-07-11 — the next jump: 30k-step training run + automated re-eval

Every finding today bottomed out at "the field has only 2000 steps," so the
jump is a real training run with the eval suite chained behind it. Before
launching unattended: made save_ckpt ATOMIC (tmp + os.replace) and attached
freshly-embedded zgoals to EVERY periodic save (collect_mate_finals once,
embed_zgoals per save) — the two halves of last night's interrupted-save bug.
Train script now also prints DIFF_SLOPE verdicts itself.

Smoke note (210-step fresh model, 256mb shards): DIFF_SLOPE_WON=-0.903,
LOST=-0.917 — a barely-trained field is pure generic finality, no outcome
separation. Step-2000 had separation 0.16; watch whether 30k widens it.

Pipeline (background, logs+timings in artifacts/generated/logs/):
train_lichess_fb --steps 30000 (resumes from 2000; 5.8 it/s on MPS => ~80
min) -> eval-head ablation repr=F/B/FB (~4 min each) -> decompose_demo.
Step-2000 checkpoint backed up as data/derived/lichess_fb_step2000.pt for
before/after comparisons.

Pre-registered expectations: VAL_TOP1 well above 0.024 (step-2000 value
unknown post-fix; chance 0.002); DESC_AUC meaningfully above 0.58 with F > B
emerging if the forward/omega story is right; DIFF_SLOPE won-lost separation
widening past 0.16; decompose give-up rules starting to fire as the field
sharpens (a sharper field should stop rating everything reachable).

---

## 2026-07-11 — 30k-step field: before/after (the budget hypothesis was right)

Pipeline timings: train 46m35s (5.8 it/s MPS, resumed 2000->30000), heads
3m41s / 3m45s / 6m52s (F/B/FB), decompose_demo 12s. Logs+times in
artifacts/generated/logs/.

    VERDICT VAL_TOP1=0.033 VAL_TOP8=0.179 (chance 0.0020)
    VERDICT REACH_SLOPE_WON=0.671 (n=200) REACH_SLOPE_LOST=0.587 (n=200)
    VERDICT DIFF_SLOPE_WON=0.174 DIFF_SLOPE_LOST=-0.080

| metric | step 2000 | step 30000 |
|---|---|---|
| DESC_AUC   F / B / FB | 0.565 / 0.570 / 0.579 | 0.625 / 0.596 / 0.636 |
| NORM_SPEAR F / B / FB | 0.236 / 0.254 / 0.248 | 0.482 / 0.376 / 0.516 |
| zero-label BASE (AUC / spear) | 0.545 / 0.184 | 0.598 / 0.369 |
| DIFF_SLOPE won / lost | -0.201 / -0.365 | +0.174 / -0.080 |
| decompose FRAC_IMPROVED / MEAN_GAIN | 0.730 / 0.287 | 0.825 / 0.430 |

Against the pre-registered expectations:
1. **F > B emerged** (AUC 0.625 vs 0.596; spearman 0.482 vs 0.376) — the
   ordering FLIPPED from step 2000. Outcome signal now lives in the
   forward/omega structure, not static board features. The F-only eval-head
   design is vindicated at this budget.
2. **DIFF_SLOPE separated with correct signs**: winners' outcome-direction
   reach rises (+0.174), losers' falls (-0.080); separation 0.254 vs 0.16
   and both-negative before. The stm-artifact worry is downgraded (a shared
   artifact wouldn't sign-split with more training).
3. **The zero-label readout (0.598 AUC) now beats step-2000's TRAINED probes**
   — the FB geometry itself is absorbing eval, exactly what the FB
   factorization promises. FB > F persists (+0.011 AUC, +0.034 spear): B
   still holds some complementary value info; worth re-checking at 100k.
4. **Give-up rules still never fire** in decompose (FRAC_IMPROVED up to
   0.825, MEAN_GAIN 0.43, one waypoint always suffices). Expectation 4 was
   WRONG, or the thresholds are what's generous: tau_exec (near-win median)
   dropped to 0.236 while best min-legs sit ~0.6+. The field did not stop
   rating everything reachable — reach>=tau executability saturates. This is
   now the clearest argument that the MC-ROLLOUT leaf verifier is the next
   necessary layer, not a nice-to-have: estimated feasibility has stopped
   being informative at the margin.

VAL_TOP1 0.033 = 16.9x chance (top8 11.4x chance/8). Loss still descending
at 30k — the curve says more budget helps; 100k+ is cheap (~2.6h) and the
suite is push-button. But the marginal information per hour now favors
building the rollout verifier first.
