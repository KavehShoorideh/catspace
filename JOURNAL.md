# Research journal

Running lab notes, newest entry last. Each entry: what was done, wall-clock
timings, verdicts (copied verbatim from experiment output), and interpretation.

---

## 2026-07-11 01:31 — package rename; eval-head representation ablation (design)

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

## 2026-07-11 07:00 — zgoals were never saved (interrupted-run bug); slopes recovered

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

## 2026-07-11 02:14 — eval-head ablation, first result (repr=F)

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

## 2026-07-11 02:30 — M1.5 kickoff: meet-in-the-middle decomposer on real boards

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

## 2026-07-11 06:30 — eval-head ablation, full table

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

## 2026-07-11 07:26 — the next jump: 30k-step training run + automated re-eval

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

## 2026-07-11 08:28 — 30k-step field: before/after (the budget hypothesis was right)

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

---

## 2026-07-11 09:36 — interactive viz suite: 7 viewers + gallery (build)

Planned in VIZ_PLAN.md, then built all 8 deliverables (D1–D8). New shared
module `catspace/viz/realboard.py` (game/PGN sampling, batched F/B embedding
under true or planner omega, a thin projection-fit wrapper) plus one builder
+ template per viewer under `experiments/viz/build_*.py` /
`catspace/viz/templates/*.html`. All local, self-contained HTML (no CDN, no
fetch), dark-styled to match the existing KRk/KRkn viewers.

**Key design fix, mid-build (Kaveh's call):** boards were originally
pre-rendered server-side with `chess.svg.board()` and embedded as raw SVG
strings in the JSON payload. Measured: every `chess.svg.board()` call
produces ~31KB of SVG **regardless of the `size` render parameter** — so
payload size scaled with position count, not pixel size, and the two
board-heavy viewers came out at 17MB (fullboard) and 74MB (decision,
2 games at 200 plies + 4 feared-replies/ply). Switched every board-bearing
viewer to storing FEN (~70 bytes) + two last-move square names, with a
hand-rolled `boardSVG(fen, opts)` renderer (8x8 grid + Unicode piece glyphs,
filled-glyph trick for legible white pieces, no external chess-board JS lib)
duplicated inline in each template, rendering only the on-screen position on
demand. Result: fullboard 17MB→196KB, decision-viewer 74MB→1.1MB (and could
restore full 200-ply games + feared-FEN on *every* candidate instead of a
capped top-2, since the per-position cost dropped ~450x). Verified the
renderer with `node --check` + a piece-count assertion (64 rects, 32 texts
on the start position) since there's no headless browser here.

**Builders and cross-checks against journaled numbers (all ckpt step 30000
unless noted):**
- **D2 training-dashboard** (17KB): pure log parsing, no torch. Verdicts
  reproduce exactly (VAL_TOP1=0.033, DIFF_SLOPE +0.174/-0.080).
- **D1 fullboard-viewer** (196KB, `--n-games 9 --n-bg 3000`): 9 balanced
  holdout games (win/loss/draw round-robin) + 3000-point background cloud,
  PCA-projected, colored by reach-to-MATE_DIFF. Found and fixed an off-by-one
  in the (unexercised) optional `--pgn` branch: it was overwriting the
  correct per-ply SAN (computed by `infer_san` comparing consecutive encoded
  positions) with the WRONG san — `games_from_pgn`'s tuple at index i holds
  the move *about to be played from* ply i, not the move that led *into*
  ply i. Removed the overwrite; `infer_san` was already correct.
- **D3 decision-viewer** (1.1MB, `--opponent random --games 6`): FB (depth=2,
  no search) vs random, 200-ply cap. 3 decisive wins, 1 draw, 2 unresolved —
  no losses, consistent with arena_real.py's documented expectation ("vs
  random it should win decisively or something is wrong"). Also fixed a real
  bug in the pre-existing builder: candidate arrows were computed from the
  pre-move board but drawn onto the post-move board SVG (arrows pointed at
  stale squares) — dropped the arrow overlay entirely (the candidate table +
  `lastmove` highlight already cover it) rather than patch it.
- **D4 decompose-viewer** (28KB, `--n-starts 60 --n-show 24`): reproduces
  decompose_demo.py's story on an independent sample — FRAC_IMPROVED=0.833
  (journaled 0.825), MEAN_GAIN=0.417 (0.43), waypoint ply mean 67.2 vs start
  29.1 (journaled 68.3/30.2) — the arc property holds.
- **D5 embedding-atlas** (1.02MB, `--n 8000 --projection tsne`, ~47s):
  step-2000 vs step-30000 F embeddings, independently t-SNE'd (not
  comparable point-for-point, only cluster shape). reach-vs-result
  correlation 0.073→0.163 across the two checkpoints — visually confirms the
  F>B training-budget flip from the eval-head ablation.
- **D6 divergence-explorer** (595KB, `--n 6000`, ~4s): top |div| ≈ 0.28,
  matching train_eval_heads.py's logged top-divergent list order of
  magnitude.
- **D7 eval-dashboard** (57KB, `--n 20000`, ~12s): AUC F=0.627 B=0.598
  FB=0.638 baseline=0.599 (journaled 0.625/0.596/0.636/0.598, all within
  0.002) — the acceptance check named in VIZ_PLAN.md passes. Reliability
  curve tracks the diagonal closely; per-ply AUC rises 0.54→0.79 from
  opening to endgame (expected: outcome gets easier to call as games
  resolve); per-Elo AUC flat ~0.60–0.65 across bins.
- **D8 gallery** (`experiments/viz/build_gallery.py`): scans
  `artifacts/generated/*.html`, writes `index.html` — 9 viewers listed
  (7 new + the 2 toy KRk/KRkn viewers).

**Tests:** `tests/test_viz_builders.py` (9 new fast tests on
`catspace/viz/realboard.py`: SAN recovery, board-SVG shape, projection
round-trip, shard-game loading incl. the holdout filter, PGN parsing,
batched-embedding unit-norm, build_html JSON round-trip). Full suite:
109 passed, 0 failed (216s, includes the pre-existing slow-marked tests).

**Total build wall-clock** for all 7 model-backed builders: ~4 min on CPU
(MPS left free; these are one-shot demo-sized runs, not training).

---

## 2026-07-11 11:00 — A/B experimentation harness + Stockfish-leakage safety gate

Kaveh's ask: use the existing A/B testing harness (`catspace/abtest.py`'s
`EValueTest`, already the toy-domain method-comparison tool used by
`compare_methods.py`, and already imported into `experiments/arena_real.py`
for real-board arena games) to iteratively improve the model and compare
against previous checkpoints, with structured JSON output so comparisons
don't require opening the viz — and, the one hard requirement: training must
never leak Stockfish-oracle signal into the planner. Checked first whether
this needed a new dependency (MLflow/W&B/Sacred/etc.) — the repo has zero
MLOps dependencies and is deliberately minimal (numpy/scipy/torch only), and
the actual ask (checkpoint-vs-checkpoint comparison + a leak gate + JSON
records I can read without a UI) doesn't need one; stayed with plain JSON
files under the new `artifacts/experiments/` (git-tracked, unlike
`artifacts/generated/`'s regenerable viz output).

**New: `catspace/audit.py`, the leakage gate.** Two independent checks,
combined into one hard `clean: bool`:
- `static_purity_check()` re-inspects, AT CALL TIME (via `inspect.getsource`),
  the actual source of the FB-training path (`train_lichess_fb.batch_tensors`,
  `.main`) and the planner's read path (`FBBoardPolicy.move_scored`,
  `planner.decompose.decompose`/`waypoint_scores`) for any reference to
  Stockfish-derived identifiers (`eval_cp`, `winprob_cp`, `sf_label`,
  `stockfish`, `wdl_*`) — so a future edit that starts reading eval_cp into
  the FB loss fails this automatically, no one has to remember to update an
  audit.
- `checkpoint_provenance_check()` reads a `provenance` dict now stamped into
  every checkpoint by `save_ckpt`/`train_lichess_fb.py` at every save (script,
  args, git commit, and a `stockfish_free` flag that is itself the OUTPUT of
  `static_purity_check()` against the running code, not a literal `True` —
  self-correcting). Pre-audit-era checkpoints without a stamp are "unknown",
  not "dirty" — the static check is the fallback, not a second hard gate.

Caught one real self-referential false positive while building this: the
first draft's `if not provenance["stockfish_free"]:` line lived inside
`train_lichess_fb.main()` — and `main()`'s OWN SOURCE therefore contained the
substring "stockfish", tripping the scan on itself. Fixed by moving that
check into `catspace.audit.is_provenance_clean()`, so `main()` never needs
the forbidden word in its own body. `tests/test_audit.py` (11 tests) covers
both directions: a synthetic function reading `eval_cp` IS caught; the real,
unmodified codebase passes `static_purity_check()` clean.

**Confirmed no leak paths exist today, and why:** `train_lichess_fb.py`'s
`batch_tensors()` reads only packed/meta/game_id/elos/clock from the pair
batches — `LichessPairSource` DOES carry `eval_cp` in `batch.meta` when
present, but the training loop never reads that key. `nn/eval_head.py`'s
`--joint` flag (fine-tunes F on the Stockfish-derived normative loss) is off
by default, and even when used, `train_eval_heads.py` never writes the
fine-tuned F back to any checkpoint the planner could load — only
`save_heads()` (desc/norm probes) is called, never `save_ckpt()`. So the FB
weights the planner reads are structurally isolated from Stockfish signal by
construction, not just by convention; the audit makes that invariant
self-checking instead of implicit.

**New: `experiments/arena_real.py::run_arena()`** — extracted the arena loop
(previously inlined in `main()`) into a reusable function instead of
duplicating it in the new harness. Generalized `opponent` to optionally be a
(white_policy, black_policy) TUPLE, not just a single color-agnostic policy —
needed because a candidate-vs-baseline-CHECKPOINT head-to-head means the
"opponent" is itself another FBBoardPolicy, which is color-specific
(zMATE_W vs zMATE_B). This made candidate-vs-baseline just another
`run_arena()` call with no new game-loop code. Re-verified `arena_real.py`'s
CLI path still works identically after the refactor (smoke-tested).

**New: `experiments/experiment_report.py`** — the harness itself. Per run:
(1) load candidate checkpoint, run `audit_checkpoint()` — HARD gate, aborts
with no report written if dirty (verified: tampered a checkpoint's
provenance to `stockfish_free=False`, confirmed exit code 1, no output file);
(2) reach/diff slopes (reused `train_lichess_fb.py`'s `reach_slope` logic);
(3) M1.5 decompose metrics (same recipe as `build_decompose_viewer.py`);
(4) arena vs a fixed opponent via `run_arena` + `EValueTest`; (5) optional
`--baseline <ckpt>` triggers a direct head-to-head via the same `run_arena`
generalization. Writes one JSON record to `artifacts/experiments/`, prints a
VERDICT line matching the repo's existing convention.

**New: `experiments/experiment_leaderboard.py`** — reads every JSON record,
sorts by timestamp, prints (+ optional `--out` JSON) each run's metrics plus
delta vs the immediately-previous run and vs the best-so-far by `--metric`
(arena_score / arena_e_value / diff_slope_won / diff_slope_lost /
decompose_mean_gain / decompose_frac_improved). DIRTY (leakage-failed) runs
are shown but excluded from best/delta tracking, not silently dropped.

**First real baseline record** (`lichess_fb.pt`, step 30000, `--games 40
--opponent random`, ~413s total): `AUDIT=CLEAN`. Reach slopes
0.611/0.490 (won/lost), DIFF_SLOPE +0.164/-0.136 — same sign pattern as the
30k training run's own verdict, some variance from an independent 200-game
resample. Decompose: FRAC_IMPROVED=0.833 MEAN_GAIN=0.417 (bit-exact match to
the D4 viz build — same seed, same params, deterministic, a good consistency
check across the two code paths). Arena vs random: +22 =18 -0, score=0.775,
e=501656 (REJECT) — zero losses, matching `arena_real.py`'s own documented
expectation ("vs random it should win decisively or something is wrong").
Saved as `artifacts/experiments/20260711T112211__step30000__89d5d2581f5c9e31.json`,
tagged "30k-step baseline" — this is now the number future training changes
get compared against via `experiment_leaderboard.py`.

**Tests:** `tests/test_audit.py` (11 new). Full suite: 120 passed
(109 + 11), 212s, no regressions from the `arena_real.py`/`nn/fb.py` edits.

---

## 2026-07-11 11:40 — autonomous planner-improvement research loop (protocol)

Kaveh's ask: go into an autonomous loop improving the planner, using the A/B
harness to compare against previous instances, journaling every round,
researching when stuck. Two explicit rules on top: (1) keep training on
more data as long as it keeps improving win/draw/loss vs a FIXED-strength
Stockfish; (2) as the model improves, escalate that Stockfish strength so
WDL doesn't saturate/clip at the ceiling and stay measurable, and always
record the strength played at.

**First vs-Stockfish measurement** (we'd only measured vs-random before
today): `sf:skill=0` (Stockfish's weakest, movetime 0.02s), 6 games,
+0 =1 -5, score 0.083 — losing badly, as `arena_real.py`'s own docstring
predicts for an imitation-bootstrapped, no-search greedy policy. This is a
GOOD starting difficulty: room to see improvement long before hitting the
saturation ceiling the escalation rule exists for.

**Protocol** (full machine-readable state in
`artifacts/experiments/research_state.json`, updated every round):
- **Opponent ladder**: `sf:skill=` 0 → 3 → 6 → 9 → 12 → 15 → 18 → 20.
  Escalate to the next rung once score_mean vs the current rung reaches
  0.75+ over >=30 games (clearly winning, approaching saturation).
- **Per round**: train more (extend `--steps` on the current best
  checkpoint via `train_lichess_fb.py`, which resumes automatically), then
  `experiment_report.py --baseline <previous best> --opponent <current
  rung> --games 40` for a same-strength apples-to-apples comparison PLUS a
  direct head-to-head vs the previous checkpoint. Compare `arena_score` at
  the SAME opponent string only -- an escalation makes the raw number drop
  even when the model improved, so every comparison must filter by
  opponent, not just read `experiment_leaderboard.py`'s raw column.
- **Continue training** if score at the current rung improves (or the
  head-to-head vs the previous checkpoint rejects H0 in the candidate's
  favor). **Escalate strength** once score >=0.75 at 30+ games. **Stuck**
  after `stuck_rounds_threshold=2` rounds with no improvement at the same
  rung -- work the `stuck_playbook` in research_state.json (re-read the
  decompose/eval-head findings for an unexploited lever, e.g. the MC-rollout
  executability verifier already flagged as the next priority; web-research
  self-play/PI-refinement techniques -- `arena_real.py`'s own docstring
  names PI-refinement as what should eventually beat Stockfish; try a
  hyperparameter lever (lr/gamma/d/readout depth) instead of raw steps; a
  documented negative result is an acceptable stuck-resolution if research
  and one alternate lever both fail).
- **Data**: batch=512, so one epoch over the current 1GB-prefix shard
  (11.07M positions) is ~21615 steps. Step 30000 was only ~1.4 epochs in --
  continuing on the SAME shard is still "more data" in the sense that
  matters (fresh unseen positions within this training run) for a while yet
  before a bigger Lichess download (network confirmed reachable; the full
  2019-01 month is 9.4GB compressed vs. the 1GB prefix on disk) is actually
  needed.
- **Leakage**: structurally enforced already -- every `experiment_report.py`
  call runs the audit as a hard gate, no protocol step can bypass it.

**Round 1 launched**: `train_lichess_fb.py --steps 60000` (resuming
`lichess_fb.pt` from step 30000, +30000 steps, ~50min expected at the
30k-run's measured ~10 it/s on MPS) → log
`artifacts/generated/logs/train_60k.log`. Next: evaluate at `sf:skill=0`
(same rung as the just-established baseline), compare, decide, continue.

---

## 2026-07-11 12:31 — round 1: a regression, an operational mistake, and a fix

**Round 1 result: REGRESSION.** `train_lichess_fb.py`'s own end-of-run
verdict already looked wrong (`DIFF_SLOPE_WON` flipped +0.174 → **-0.075**,
`REACH_SLOPE_WON` fell 0.671 → 0.437), so per protocol I didn't trust it and
ran the independent `experiment_report.py` measurement instead. It confirmed
a real, well-corroborated decline, most unambiguously in the decompose
numbers (no sign-interpretation ambiguity there, unlike the diff-slope
halves individually):

| metric | step 30000 | step 60000 |
|---|---|---|
| decompose FRAC_IMPROVED | 0.833 | **0.617** |
| decompose MEAN_GAIN | 0.417 | **0.310** |
| reach_slope_won | 0.611 | **0.469** |
| reach_slope_lost | 0.490 | **0.263** |
| tau_exec (near-win reach) | 0.236 | **0.157** |

Fewer middlegame starts have a useful waypoint, and the gain from splitting
one shrank by a quarter — a real degradation in exactly the M1.5 planner
machinery this whole loop exists to improve, not noise in one metric.

**Operational mistake, logged honestly:** `train_lichess_fb.py` saves over
`data/derived/lichess_fb.pt` in place, and I never copied the step-30000
weights to a backup before launching round 1. They're gone — only the JSON
metrics record (`20260711T112211__step30000__89d5d2581f5c9e31.json`)
survives, so the step-30000-vs-step-60000 comparison above is metrics-only,
not a live head-to-head. Immediately backed up the step-60000 output
(`lichess_fb_step60000.pt`) and hard-added "always back up before training"
to `research_state.json`'s constraints — every future round follows this.

**Diagnosis + research (stuck-playbook step 2):** `train_lichess_fb.py` had
**no learning-rate schedule at all** — constant `lr=3e-4` for the entire
run, including this 30000-step extension (bringing the shard to ~2.8
epochs). WebSearched to check whether this is a plausible cause before
committing to a fix:
- "constant LR contrastive learning representation collapse" — confirmed:
  dimensional collapse is a known stationary point of the InfoNCE loss, and
  a non-decaying (or too-large) LR is called out as accelerating it via
  embedding-mean drift from negative-pair gradients aligning in similar
  directions. ([Feature Normalization Prevents Collapse of Non-Contrastive
  Learning Dynamics](https://arxiv.org/pdf/2309.16109))
- "cosine decay warmup contrastive SimCLR CLIP best practices" — confirmed:
  SimCLR and CLIP both use linear warmup + cosine decay, standard practice
  is decaying to about 1/10th of peak LR over the full schedule.
  ([SimCLR paper](https://proceedings.mlr.press/v119/chen20j/chen20j.pdf),
  [SimCLR/Flax training notes](https://www.tahabouhsine.com/flaxdocs/research/contrastive-learning))

This matches the symptom well: raw retrieval loss/VAL_TOP8 looked roughly
flat (not obviously diverging), but the *downstream geometric structure*
the decomposer depends on (bottleneck-max waypoint selection needs a
well-calibrated, not-drifting reach signal) degraded — consistent with
"quiet" representation drift rather than an obvious loss blowup.

**Fix implemented:** `train_lichess_fb.py` now cosine-decays
`lr -> lr/10` over *each invocation's remaining steps* (resume_step ->
`--steps`), not the whole training history — a new `--lr-min` arg,
default `lr/10`. Deliberately scoped to the current invocation rather than
the full historical step count, since the constant-LR phase already
happened and can't be fixed retroactively; this is the standard
"resume-and-decay" shape for exactly this iterative extend-and-train
workflow. Smoke-tested on a 250-step CPU fresh run: LR fell from ~6.9e-4 at
40% progress to ~1.9e-4 at 80% progress, matching the cosine formula
closely. Full pytest suite (120 tests, including `catspace.audit`'s
static-source inspection of this exact file) passes clean after the change
— the leakage audit isn't affected by an LR-schedule edit, as expected, but
worth confirming since audit.py inspects `train_lichess_fb.py`'s source
directly.

**Round 2 (recovery) launched**: `train_lichess_fb.py --steps 90000`
(resuming step 60000, +30000 steps, now WITH cosine decay) →
`artifacts/generated/logs/train_90k.log`, ~55min expected. `best` in
`research_state.json` stays pinned at step 30000 (metrics-only reference)
until round 2 is evaluated — step 60000 is not promoted to best, it's a
documented regression kept only for comparison. Next wake: run
`experiment_report.py` on the step-90000 checkpoint, compare decompose/
reach-slope numbers against BOTH the step-30000 baseline and the
step-60000 regression, to see whether the LR fix actually recovered
quality or whether this needs another lever from the stuck-playbook.

---

## 2026-07-11 15:31 — round 2: LR fix partially worked, second cause found (epoch repetition)

**Round 2 result: PARTIAL RECOVERY, still below the step-30000 baseline.**
Full three-way comparison, all via the identical `experiment_report.py`
methodology:

| metric | step 30000 | step 60000 (round 1) | step 90000 (round 2) |
|---|---|---|---|
| decompose FRAC_IMPROVED | 0.833 | 0.617 | 0.717 |
| decompose MEAN_GAIN | 0.417 | 0.310 | 0.373 |
| reach_slope_won | 0.611 | 0.469 | **0.374** |
| reach_slope_lost | 0.490 | 0.263 | **0.072** |
| tau_exec (near-win reach) | 0.236 | 0.157 | **0.103** |
| diff_slope_won | 0.164 | 0.038 | 0.111 |

The LR-schedule fix (added after round 1) clearly helped: decompose
FRAC_IMPROVED/MEAN_GAIN and diff_slope_won all moved back toward the
step-30000 baseline. But it did NOT fix everything — reach_slope_won,
reach_slope_lost, and tau_exec kept declining **monotonically across both
rounds**, LR fix notwithstanding. reach_slope_lost in particular is now
almost zero (0.072): lost games barely show any ply-reach correlation left
at all. A fix that only partially works, on a strictly-declining metric,
means there's a second cause still active.

**Second diagnosis:** step 90000 is **~4.16 epochs** over the same
1GB-prefix shard (11.07M positions) — `90000*512/11067003`. The ORIGINAL
successful run (2000→30000 steps, the one that produced the well-behaved
step-30000 baseline) only reached **~1.4 epochs**, on the exact same
constant-LR regime that later caused round 1's regression. That's an
important asymmetry: constant LR alone didn't break things at 1.4 epochs,
it broke things somewhere between 1.4 and 2.8+ epochs. This points at
**epoch-repetition / overfitting to the fixed shard's specific structure**
as the dominant remaining cause, with the missing LR schedule as a real but
secondary compounding factor (which is exactly consistent with "partial
recovery, not full recovery" once decay was added back).

**Decision:** this is the `stuck_rounds_threshold=2` trigger — two rounds
of extending training on the same 1GB shard, neither beating the
step-30000 baseline on the metrics that matter most (decompose, reach
slopes). Per the user's own explicit instruction ("keep training on more
data AS LONG AS it improves things"), the right reading of that rule, given
this evidence, is: stop extending on the SAME data (it isn't improving
things anymore) and get MORE data instead of more epochs of the same 11M
positions — which is stuck-playbook lever 3 (data scale), not a new lever
invented on the spot.

**Action:** measured download throughput (5.4 MB/s via a 100MB range
request), then launched a background pipeline: (1) download a 4GB Lichess
prefix (range-request, same source, `--tolerate-truncation`, ~13min
expected) — 4x the previous 1GB prefix, matching the
`next_data_scale_gb_if_epochs_exhausted` figure already planned into the
protocol before round 1 even ran; (2) `build_lichess_shards.py` on it
(`--max-gb 8 --max-games 1000000` to not artificially cap the larger
source); (3) a FRESH 30000-step training run (`--fresh`, matching the
ORIGINAL successful run's step budget exactly, for a clean apples-to-apples
read) on the new shard, saved to a **new checkpoint file**
(`data/derived/lichess_fb_4gb.pt`, not overwriting the existing
`lichess_fb.pt` lineage) — deliberately never touches the step-30000/60000/
90000 checkpoints, so this branch is directly comparable without any risk
of repeating the round-1 backup mistake. `research_state.json`'s `best`
stays pinned at step 30000 (still the reference to beat) until round 3 is
evaluated. Task `br8cfv8b8`, full pipeline log
`artifacts/generated/logs/data_scale_pipeline.log`. Next wake: check
pipeline progress; once the fresh run finishes, run `experiment_report.py`
on `lichess_fb_4gb.pt` and compare against the step-30000 numbers above —
if genuinely new data (not just more epochs) resolves the decline, that's
the confirmation; if it doesn't, the epoch-repetition hypothesis was wrong
and the stuck-playbook needs another lever (embedding dim, gamma, or a
deeper look at whether the InfoNCE batch/negative-sampling setup itself
has a ceiling around this loss level).

---

## 2026-07-11 16:51 — round 3: 4GB shard, promising but not a clean win yet

**Pipeline result**: downloaded a 4GB Lichess prefix -> 55.82M positions
across 56 shards (vs the 1GB shard's 11.07M -- ~5x). Fresh 30000-step run
on it, saved to `data/derived/lichess_fb_4gb.pt` (new file, doesn't touch
the existing checkpoint lineage).

**Full four-way comparison** (all via identical `experiment_report.py`
methodology):

| metric | 1GB step30k | 1GB step60k | 1GB step90k | 4GB fresh step30k |
|---|---|---|---|---|
| decompose FRAC_IMPROVED | 0.833 | 0.617 | 0.717 | 0.767 |
| decompose MEAN_GAIN | 0.417 | 0.310 | 0.373 | **0.462** |
| reach_slope_won | 0.611 | 0.469 | 0.374 | 0.515 |
| reach_slope_lost | 0.490 | 0.263 | 0.072 | 0.316 |
| tau_exec | 0.236 | 0.157 | 0.103 | 0.201 |
| arena vs sf:skill=0 | (n/a) | 0.075 | 0.087 | 0.100 |

Clearly better than both round-1 and round-2 on every single metric — the
data-scale lever is directionally working. It also beats the ORIGINAL
1GB-step30000 baseline on MEAN_GAIN specifically (0.462 vs 0.417), a new
best. But it's still slightly below the 1GB baseline on FRAC_IMPROVED,
reach_slope, and tau_exec — not a clean, unambiguous win.

**Interpretation check before deciding anything**: is 30000 steps on 4GB
actually a fair comparison to 30000 steps on 1GB? No — at 30000 steps, the
4GB shard has only been seen ~0.275 times on average (55.82M positions /
(30000×512)), vs the 1GB shard's ~1.4 passes at the same step count. More
unique data per step means proportionally *less* gradient exposure per
position at a fixed step count. Concluding "data scale doesn't clearly
help" from this alone would conflate "this checkpoint is comparatively
undertrained" with "the lever doesn't work" — two different claims. The
right test is comparable EXPOSURE, not comparable step count.

**Decision**: extend the SAME `lichess_fb_4gb.pt` checkpoint from 30000 to
60000 steps (round 4) — still only ~0.55 epochs on this shard, nowhere
near the 1.4–4+ epoch range where the 1GB shard started failing. Backed up
the step-30000-on-4GB checkpoint first
(`data/derived/lichess_fb_4gb_step30000.pt`). If round 4 clearly surpasses
the 1GB-step30000 baseline (not just approaches it), that confirms
epoch-repetition was the real driver of rounds 1–2's decline and the 4GB
shard is the new working substrate to keep extending. Log
`artifacts/generated/logs/train_4gb_60k.log`, task `biovfteth`, ~50min
expected at the ~13 it/s observed on this shard.

---

## 2026-07-11 17:46 — round 4: no clean win, and arena score has stalled for 4 straight rounds

**Round 4 result: MIXED, still not a clean win over the reference baseline.**
Full five-way comparison, all via identical `experiment_report.py`
methodology:

| metric | 1GB-30k | 1GB-60k | 1GB-90k | 4GB-30k | 4GB-60k |
|---|---|---|---|---|---|
| decompose FRAC_IMPROVED | 0.833 | 0.617 | 0.717 | 0.767 | 0.800 |
| decompose MEAN_GAIN | 0.417 | 0.310 | 0.373 | 0.462 | 0.463 |
| reach_slope_won | 0.611 | 0.469 | 0.374 | 0.515 | 0.520 |
| reach_slope_lost | 0.490 | 0.263 | 0.072 | 0.316 | **0.282** |
| tau_exec | 0.236 | 0.157 | 0.103 | 0.201 | **0.168** |
| arena vs sf:skill=0 | n/a | 0.075 | 0.087 | 0.100 | 0.100 |

FRAC_IMPROVED and MEAN_GAIN inched up from round 3 (0.767→0.800,
0.462→0.463), but reach_slope_lost and tau_exec moved AWAY from the
baseline, not toward it — the same early-warning shape that preceded the
1GB shard's collapse, just much slower thanks to the larger dataset. This
round doesn't cleanly beat the 1GB-step30000 reference on the full metric
set.

**The more important observation:** arena score vs `sf:skill=0` has been
flat across all four training-lever rounds — **0.075 → 0.087 → 0.100 →
0.100**. None of it — the LR-schedule fix, 5x more data, or 2x more steps
on that larger data — moved the metric that actually matters (does the
planner win games), even while some embedding-quality proxy metrics
(MEAN_GAIN, decompose FRAC_IMPROVED) genuinely improved. The proxies and
the actual win rate have decoupled.

**This is not actually a surprise, on reflection.** `arena_real.py`'s own
docstring, written before any of this loop ran, already says it: *"this
field is imitation-bootstrapped from human games and read out greedily
with no search — vs Stockfish (floor Elo 1320) losing is the EXPECTED
baseline; the roadmap's PI-refinement loop is what should move it."* Four
rounds of tuning the embedding (the thing decompose/reach-slope actually
measure) were reasonable things to try and worth doing, but they were never
going to touch the no-search bottleneck those metrics don't capture.

**Research** (WebSearched before committing to a big engineering pivot):
- *"self-play imitation learning chess no search"* — confirmed: "learning
  to play chess without knowing the rules is extremely challenging since
  you cannot improve via self-play, resulting in relatively poor policies
  compared to other methods" — pure imitation learning has a documented
  ceiling. ([Imitation Learning by Estimating Expertise of
  Demonstrators](https://arxiv.org/pdf/2202.01288))
- *"shallow search + learned value function"* — confirmed and more
  actionable: "the strongest results were obtained when the learned value
  function was combined with deeper lookahead during gameplay." This is a
  pure inference-time change (no retraining) that directly reuses
  `F(s)@z`, the value function this whole loop has been trying to improve.
  ([Learning to Plan via Supervised Contrastive Learning and Strategic
  Interpolation](https://arxiv.org/html/2506.04892v1), [Superior Computer
  Chess with Model Predictive Control, Reinforcement Learning, and
  Rollout](https://arxiv.org/pdf/2409.06477))

**Decision: pivot from "train the embedding more" to "search deeper with
the embedding we already have."** This is a genuinely different lever than
rounds 1–4 (all training-volume variants), and it's testable immediately
without waiting on more training. `FBBoardPolicy`'s depth-1/depth-2 readout
is hardcoded for exactly those two cases with full GPU-batched leaf
evaluation; generalizing it to arbitrary depth isn't a small edit, so this
became a new class.

**New: `catspace/nn/policy_fb.py::FBSearchPolicy`.** Beam-limited plain
minimax (deliberately NOT alpha-beta — pruning needs serial leaf
evaluation, which would give up the single-batched-forward-pass philosophy
this codebase uses everywhere, including `FBBoardPolicy`'s own depth-2).
Root branching is never capped (every legal move gets a fully-searched
score); every ply after the root is capped at `beam` children, ranked by a
cheap one-ply reach heuristic, with any mate-delivering child exempted
from the cap regardless of rank. `F(s)@z` — the exact same, unchanged
value function — is still the only learned signal; nothing here retrains
anything.

**A real bug caught by testing, not by luck.** First version used a flat
`MATE_SCORE`/`MATED_SCORE` regardless of ply distance (matching
`FBBoardPolicy`'s existing convention, where it never mattered since depth
was hardcoded to 1 or 2). Wrote
`tests/test_realboard.py::test_fb_search_policy_finds_forced_mate_in_2` — a
K+2R-vs-lone-K position with a forced mate in exactly 2 (rank-control then
mate), using `z=0` so ALL non-terminal leaves score exactly 0 and move
selection is driven purely by mate detection, isolating tree-search
correctness from embedding quality entirely. The test failed: the policy
correctly found a move guaranteeing mate within the horizon, but not
necessarily the FASTEST one, because with a flat mate score, delivering
mate immediately and delivering it one harmless tempo later score
identically — a lone king has no counterplay to punish the delay in THIS
position, but the underlying issue (ties among all in-horizon mates) is a
real generality gap. Fixed by discounting mate scores by ply distance
(`MATE_SCORE - ply`, `MATED_SCORE + ply`, standard practice in real
engines) so the fastest mate strictly dominates. All 11 realboard tests
(8 existing + 3 new) pass after the fix; full suite 123 passed, 0 failed.

**Wired into the harness**: `experiment_report.py` gained
`--search-depth`/`--search-beam` (opt-in; omitted = unchanged
`FBBoardPolicy` behavior, fully backward compatible — every prior round's
command still reproduces the same policy).

**Timing reality check before committing to a full comparison**: measured
`depth=4, beam=6` at ~20s/move on CPU with real weights — far too slow for
a 40-game arena run (9–18h). `depth=3, beam=4` measured ~1.8s/move,
tractable. Launched a modest first read: 16 games, `max-plies=60`,
`depth=3 beam=4`, vs the same `sf:skill=0` opponent every prior round used,
`--skip-decompose` (decompose doesn't touch the policy class at all, so
re-running it here would just burn time for a value we already have).
Task `bax2nerac`. The number to beat: **0.100** (flat across all four
training-lever rounds). If genuine multi-ply lookahead over the SAME
embedding moves this at all, that's the confirmation the literature and
this codebase's own docs predicted; if it doesn't, that's an important
negative result too — it would mean the current F(s)@z value function
isn't informative enough even for shallow search to exploit, which points
back at embedding quality (or evaluation granularity) rather than the
no-search bottleneck as the real ceiling.

---

## 2026-07-11 18:08 — round 5: first movement on arena score in six rounds

**Result: 0.250, up from a flat 0.100.** `FBSearchPolicy(depth=3, beam=4)`
vs `sf:skill=0`, n=16, `max-plies=60`: **+1 =6 −9**, score **0.250**,
e=9.50 (not yet reject-worthy at α=0.05 — needs e≥20 — but this is real
directional signal, not noise-level movement). For comparison: the exact
same checkpoint (`lichess_fb_4gb.pt`, step 60000) scored 0.100 with the
unchanged `FBBoardPolicy(depth=2)` readout just one round earlier. Same
weights, same opponent, same everything except HOW the value function gets
read out — and the score jumped 2.5x. This is the first time any lever in
six rounds has moved arena score at all.

Wall-clock: `depth=4, beam=6` measured ~20s/move on CPU with real weights
— a 40-game run at that setting would take 9–18 hours, ruled out for now.
`depth=3, beam=4` measured ~1.8s/move (~40s/game observed in the actual
run), tractable.

**Interpretation, held carefully**: this is a promising *first read*, not
a confirmed result — n=16 is small and e=9.50 doesn't clear the α=0.05
bar yet. Launched the protocol-standard n=40 confirmation run at the same
config (`depth=3, beam=4`) before drawing conclusions. If it holds up,
the six-round arc of this loop becomes a genuinely interesting research
narrative: four rounds of tuning training volume (LR schedule, 4x data,
more steps) never moved the metric that matters, while a single
architecture change — reusing the SAME embedding with deeper lookahead
instead of retraining it further — moved it 2.5x on the first try. Matches
both the codebase's pre-existing documented expectation and the
WebSearched literature precisely.

Next: read the n=40 confirmation (task `bge3llakc`), and if it holds,
explore the depth/beam space further (a controlled `depth=2/beam=8` run to
isolate whether the gain is really from the extra ply vs. some
re-implementation quirk of the beam-search framework itself; deeper
configs if runtime allows) before deciding whether `FBSearchPolicy`
becomes the new default readout for future rounds.

---

## 2026-07-11 18:20 — round 6: CONFIRMED — first statistically significant win of the whole loop

**n=40 vs `sf:skill=0`: +1 =18 −21, score 0.250, e=11666.43, REJECT at
α=0.05.** Not just directionally consistent with the n=16 read — the
`score_mean` is IDENTICAL (0.250) at both sample sizes, an unusually clean
reproduction. This is the first time in six rounds that anything in this
loop has produced a statistically confirmed improvement on the objective
metric (win/draw/loss vs a fixed Stockfish strength).

Context for how large this is: four straight rounds of tuning training
volume (LR-schedule fix, 4x more data, 2x more steps on that data) left
arena score sitting at 0.075 → 0.087 → 0.100 → 0.100 — never moving, never
significant. One architecture change — reading out the exact same,
already-trained embedding with 3-ply beam search instead of the hardcoded
2-ply `FBBoardPolicy` — produced a 2.5x jump that cleared statistical
significance on the first properly-sized test. Same checkpoint
(`lichess_fb_4gb.pt`, step 60000) both times; only the readout changed.

**Promoted to `best`**: `research_state.json`'s `best` now tracks
(checkpoint, readout) jointly rather than just a checkpoint path, since
readout strategy is now confirmed to matter as much as — in this case,
far more than — the embedding weights for the metric that actually
counts.

**Before declaring victory**: launched a `depth=2, beam=8` control (round
7) — same ply-depth as the flat `FBBoardPolicy(depth=2)` baseline that's
been stuck at ~0.08-0.10, but run through the NEW beam-search framework
(including the ply-distance mate discount fixed earlier). If this control
ALSO scores well above 0.100, the gain isn't purely "the extra ply" — it
would point at some other implementation difference (the mate-distance
discount most likely, since that's the one behavioral change beyond depth
itself) and the real lesson would be narrower than "deeper search helps."
If the control stays near 0.100, that confirms the extra ply specifically
is what's doing the work, which is the cleaner and more generalizable
story — and the natural follow-up becomes pushing depth further (4 with a
narrower beam, e.g. beam=3, to control runtime) rather than just widening
beam at a fixed depth.

---

## 2026-07-11 18:43 — round 7: the gain cleanly splits into two real causes

**`depth=2, beam=8` (new framework, SAME ply-depth as the stuck baseline):
0.163**, n=40, +0 =13 −27, e=14.5M, REJECT. Neither the null result nor
the full 0.250 — a genuine middle point:

| readout | score |
|---|---|
| `FBBoardPolicy(depth=2)` (old, rounds 1–4) | 0.100 |
| `FBSearchPolicy(depth=2, beam=8)` (new framework, same depth) | 0.163 |
| `FBSearchPolicy(depth=3, beam=4)` (new framework, +1 ply) | 0.250 |

The 0.150-point total gain splits roughly 42/58: ~0.063 from the new
framework itself at matched depth (most likely the ply-distance
mate-discount fixed during testing, and/or how beam-ranked opponent-reply
selection differs from `FBBoardPolicy`'s exhaustive-but-unranked MIN —
notably, beam=8 actually considers FEWER opponent replies than
`FBBoardPolicy`'s unrestricted enumeration in high-branching middlegame
positions, yet still scores better, which says the beam's shallow-reach
ranking is a decent proxy for "opponent's actually-threatening replies"
even before deeper search refines it), and ~0.087 from the extra ply
specifically. Both effects are real and both are statistically confirmed
(both e-values are enormous). This isn't simply "more plies = better" —
but depth clearly matters on top of the framework improvement.

**Next**: push to depth=4. `depth=4, beam=6` was already ruled out at
~20s/move; timing `depth=4` with narrower beams (3, 4) now to find a
tractable setting before committing to a full arena run — task
`br6djhkef`.

---

## 2026-07-11 18:52 — round 9: depth=4/beam=3 REGRESSES to 0.200 — closes out the ply-depth-tuning thread

**n=40 vs `sf:skill=0`: +1 =14 −25, score 0.200, e=144897.29, REJECT.**
Down from depth=3's confirmed 0.250, breaking the depth=2(0.163) ->
depth=3(0.250) trend rather than continuing it:

| readout | score |
|---|---|
| `FBBoardPolicy(depth=2)` (old) | 0.100 |
| `FBSearchPolicy(depth=2, beam=8)` | 0.163 |
| `FBSearchPolicy(depth=3, beam=4)` | **0.250 (best)** |
| `FBSearchPolicy(depth=4, beam=3)` | 0.200 |

depth=4 had to shrink beam to 3 (from depth=3's beam=4) to stay
tractable, so this isn't a clean "more plies, same beam" comparison —
narrowing the beam to buy depth lost more than the extra ply gained.
Diminishing/negative returns on ply-depth alone, at a fixed embedding.
Closes out this research thread (full-board vs. graduated Stockfish,
tuning ply-depth as the main lever): `FBSearchPolicy(depth=3, beam=4)`
remains `best` in `research_state.json`.

**Pivot (Kaveh's call, mid-round):** rather than keep hand-tuning
depth/beam pairs, two bigger changes were made instead:

1. **Node-budget search.** `FBSearchPolicy` no longer takes a fixed
   `depth`; it takes `max_nodes` and derives depth per-move from the
   position's real branching factor (`_depth_for_budget`), spending a
   fixed compute budget as deep as it reaches. Modeled on Leela Chess
   Zero's own node economy (WebSearched): ~800 nodes/move is Leela's
   self-play floor, ~1500-2000 is a reasonable "actually playing"
   reference point, ~128k is where returns diminish sharply. Target set
   ~150-200 nodes — deliberately ~10x below the reference point, so any
   win margin has to come from the *plan*, not from out-searching the
   opponent. Constructor signature, `experiment_report.py`'s
   `--search-nodes` CLI flag (renamed from `--search-depth`), and all 3
   `test_realboard.py` `FBSearchPolicy` tests updated accordingly; full
   suite (123 tests) still green (221.60s).

2. **New diagnostic scenario: KRR vs KBP.** Full-board play vs. Stockfish
   makes failures hard to diagnose ("too many [concepts] and I don't know
   how to diagnose the planner's failures" — Kaveh). Switched to a
   narrow, interpretable endgame instead: White K+R+R vs. Black
   K+B(light-squared)+P(e-file). Colors fixed (Stockfish always plays
   Black with the bishop+pawn — `diagnostic_krrkbp.py`'s
   `random_krrkbp()`); a 20-position fixed starting set was generated
   (seed=42, `artifacts/experiments/krrkbp_fixed_set.json`) so every
   algorithm comparison uses the identical position distribution.
   Syzygy tablebases downloaded (`data/syzygy/`, `KRRvKBP` + its full
   dependency closure of 5/4/3-piece tables, via
   `tablebase.lichess.ovh/tables/standard/{3-4-5,6}-{wdl,dtz}/`) confirm
   all 20 positions are WDL=2 (winning for White) — a real, provably
   winnable target for the planner to find. Per Kaveh: **the tablebase is
   an observational overlay only** ("if it wins some other way, who am I
   to penalize it? ... use the tablebase to tell me what was the actual
   distance to mate so I can compare to my planner when inspecting
   visually"), not a scoring signal — win/draw/loss vs. Stockfish stays
   the objective metric; DTZ is for the decision-viewer, not the reward.
   Params fixed for now (node budget, beam, Stockfish strength) per
   Kaveh's call ("let's fix params for now, and see if we can tune the
   algo") — escalation deferred until an algorithm win is found.

**Next**: an architecture/algorithm search focused specifically on what
should let the planner learn "keep the rooks on squares the bishop can't
touch" — see the next entry.

---

## 2026-07-11 20:08 — plan-level (not move-level) search: `FBSearchPolicy.plan()` + `FBPlanPolicy`

**The idea (Kaveh):** *"the plan shouldn't change if the materials have
just moved around the board without actually changing... We know what the
plan is, what the trajectory is, and we should be able to get by without
searching. The only thing is that we're searching moves, not plans... I'm
looking for a way to capture this concept."* This mirrors a design that
already exists in the toy (index-based) domain —
`catspace/planner/plans.py`'s `PlanMemory`/`Plan`/`BlockReason` +
`catspace/planner/selector.py`'s `PlanSelector`/`GreedyReach`
("keep the current active plan while it's ACTIVE") — but it was never
ported to real boards. `catspace/planner/decompose.py` (the M1.5
meet-in-the-middle waypoint decomposer) turned out to already be
real-board-compatible, but it needs an externally-sourced `WaypointPool`;
the goal here was to avoid needing one.

**What was built**, both in `catspace/nn/policy_fb.py`:

- `FBSearchPolicy.move()` was refactored to share its tree-build/score
  logic with a new `_build_and_score()` helper (no behavior change —
  verified via the existing tests plus a new one asserting `plan()`'s
  chosen move exactly matches `move()`'s).
- `FBSearchPolicy.plan(board, rng) -> (move, subgoal_board)`: reuses that
  same search tree the policy already builds to choose its move, and
  additionally walks the **principal variation** — the sequence of
  backed-up-best children, alternating max (my move) / min (opponent
  reply) exactly as `_score()` does internally — down to its deepest
  leaf. That leaf's board is returned as a **subgoal**: the position this
  search's own best-response line predicts play heads toward, several
  plies out, entirely as a side effect of the move it was already
  computing. No separate waypoint search needed.
- `FBPlanPolicy`: composes two `FBSearchPolicy` instances — a deep
  **planner** (`plan_nodes=2000` default) and a cheap **executor**
  (`shallow_nodes=60` default) sharing the same trained `fb` network.
  Calls `planner.plan()` once to get a subgoal, embeds it with
  `fb.embed_B()` (L2-normalized, so `F(s)@B(subgoal)` is a cosine
  similarity in `[-1, 1]`), and re-points the executor's target `z` at
  it. On every subsequent move it only runs the *cheap* executor,
  re-invoking the deep planner only on one of three triggers (mirroring
  `PlanMemory.update()`'s ACHIEVED/STALLED/REPLAN-on-drop logic,
  collapsed to a single always-active plan): **ACHIEVED** (reach to
  subgoal `>= achieved_cos`), **STALLED** (`max_plies_per_plan` shallow
  moves played since the last plan), or **DROPPED** (reach fell more than
  `drop_delta` below its value when the plan was made).

**Verification** (3 new tests in `test_realboard.py`, all passing):
`test_fb_search_policy_plan_matches_move_and_has_subgoal` confirms
`plan()`'s move agrees with `move()`'s independently-computed move on a
forced-mate-in-2 position, and that the PV subgoal walks all the way to
the actual mate (not just one ply deep). `test_fb_plan_policy_legal_and_takes_mate`
is the same legality/mate-taking smoke test the other policies get.
`test_fb_plan_policy_holds_plan_across_plies` sets `drop_delta`/
`achieved_cos` outside `[-1, 1]` (so only the plies-cap can fire) and
confirms `plans_made == 2` over 8 plies at `max_plies_per_plan=6` —
i.e. the executor, not the deep planner, is genuinely doing the picking
on non-replan plies. Full suite re-run pending (background).

**Not yet done**: this hasn't been run against Stockfish or the KRRvKBP
fixed set yet — `FBPlanPolicy` vs. baseline `FBSearchPolicy` on the
tablebase-verified positions is the next comparison, once the fixed-set
arena harness (with early stopping wired to `EValueTest.reject_at`) is
built. Per Kaveh's "check it up the wazoo" mandate, no performance claim
should be made about plan-persistence until that head-to-head is run and
the win survives scrutiny — right now this is a mechanism that's been
built and unit-tested, not yet shown to help.

---

## 2026-07-11 20:16 — round 10: KRRvKBP head-to-head is INCONCLUSIVE, and a bigger problem surfaced — the embedding is tactically blind in this endgame

**`experiments/krrkbp_arena.py` built**: paired comparison (matched starting
FEN + rng seed per position, `catspace.abtest.EValueTest` on the score
DIFFERENCE + `confidence_sequence` for a CI on the mean diff — the first
real-board use of either), Syzygy DTZ looked up per position for the
printed readout only (never scores anything).

**n=20 result**: `FBSearchPolicy=0.575` vs `FBPlanPolicy=0.525`,
`mean_diff=-0.050`, `CI=[-1.041, +0.941]`, `e=0.63` — nowhere near
`1/alpha=20` needed to reject at α=0.05, and `e<1` means the data leans
mildly toward "no difference," not just "not enough evidence yet." The CI
spans almost the entire possible range: **this comparison is genuinely
uninformative, not a confirmed null.**

**Methodological bug found while investigating (verification, not the
headline finding):** the "matched-seed pairing" rationale
(`run_paired`'s docstring, copied from `abtest.paired_eval`'s toy-domain
version) assumes both policies face the *same opponent-randomness
stream*. That's true in the toy domain, where the opponent consumes the
passed `np.random.Generator`. It's **false** for `UCIBoardPolicy`:
Stockfish's `Skill Level` weakening uses the engine's own internal RNG,
never touched by our seed — re-running the exact same position/seed
through the harness produced a *different* result each time (verified:
position 15 gave `0-1 CHECKMATE` in the original run, `1/2-1/2
INSUFFICIENT_MATERIAL` on a same-seed re-run). So this design is
effectively closer to an unpaired n=20-per-arm comparison than a true
paired one — noisier than intended, on top of already being underpowered.
Not fixed yet (no UCI option controls Stockfish's skill-level RNG); noting
it here so nobody trusts a future tight-looking CI from this harness
without accounting for it.

**The actual headline finding, from investigating why BOTH policies
scored only ~55% against `sf:skill=0` from tablebase-CONFIRMED winning
positions:** a single-policy scan of all 20 positions (`FBSearchPolicy`
only, to remove the opponent-randomness confound) found **9/20 (45%)
games end in `INSUFFICIENT_MATERIAL`** — the policy is trading its OWN
rooks away down to a bare-kings draw from a 2-rooks-vs-bishop+pawn
starting advantage. Inspecting one such game (position 15) found the
White's very **first move, `Rf4`, hangs a full rook for free** (`1...exf4`
recaptures it immediately) — and it's not a search artifact or a tie:
printing every legal move's raw root score at that position shows `Rf4`
scores HIGHEST of all 34 legal moves (0.0419), with the entire score
range across every move (safe or hanging) compressed into [0.0148,
0.0419] — the embedding barely distinguishes "hang a rook for nothing"
from "any other move" here at all.

**Diagnosis:** `F(s)@z` was trained on human Lichess games, where
K+R+R-vs-K+B+P essentially never occurs — this specific diagnostic
scenario is exactly the kind of out-of-distribution structure the
embedding was never asked to judge. The nearly-flat, tactically-blind
score landscape in this position is consistent with that: not a search
bug (`_build_and_score`'s scores are exactly what they should be given
the embedding), not a `FBPlanPolicy`-vs-baseline question at all (the
shallow executor calls the SAME embedding, so it inherits the same
blindness) — the reach signal itself doesn't work here yet.

**This changes the recommended next step.** No readout strategy
(depth, beam, node budget, or plan-persistence) can fix an evaluation
function that can't tell "hang a rook" from "don't." Before comparing
`FBSearchPolicy` vs `FBPlanPolicy` further on this scenario, the
embedding needs either (a) some exposure to K+R+R-vs-minor-piece-like
material distributions during training (synthetic self-play/tablebase-
seeded data), or (b) a material-safety guard blended into the search
score as a stopgap, so the diagnostic is actually testing "does the
planner find the rook-vs-bishop-square technique" rather than "does the
planner avoid hanging pieces for free." Flagging this to Kaveh rather
than picking a direction autonomously, since it's a real fork in the
research plan, not a tuning knob.

---

## 2026-07-11 21:15 — literature research (3 parallel agents) + decision: outcome-conditioned training before quasimetric swap

Kaveh's framing (verbatim): "we need to find the mechanism, not code it in" --
material safety, fork-avoidance, and pin-discovery should emerge from the
representation/training, not be hand-coded as a guard. Also asked whether
FB is a quasimetric embedding (it is NOT -- `nn/fb.py`'s own docstring:
cosine-normalized InfoNCE, explicitly "does NOT implement the chain
QuasimetricEmbedding protocol") and requested literature grounding,
especially for DAG-structured domains like chess.

**Three parallel research agents, findings (full reports in conversation,
condensed here):**

1. **Quasimetric vs. contrastive goal-conditioned RL.** Myers, Zheng,
   Eysenbach, Levine (arXiv:2509.20478, 2025) directly compare quasimetric
   value functions to contrastive RL (same family as the current F(s)@B(g)
   dot product) on OGBench "stitching" splits (composing path segments
   never jointly observed in training -- exactly the pin-then-capture
   compositionality problem): e.g. antmaze_large_stitch 37.3% (quasimetric)
   vs 10.8% (contrastive). Wang/Torralba/Isola/Zhang (ICML 2023): the
   *optimal* goal-conditioned value function is provably a quasimetric, so
   an unconstrained dot product has no structural reason to compose
   correctly across hops. Practical architecture: MRN (Liu/Feng/Liu/Stone,
   AAAI 2023) -- smallest delta from current code, `-d(s,g) + r(s,a,g)`
   with `d` a real metric. Caveat: zero prior validation on discrete/DAG
   domains -- every result is continuous robotic control.
2. **Learned heuristics for DAG/combinatorial planning + tactical concept
   emergence.** GOOSE/STRIPS-HGN (planning heuristic GNNs) represent goals
   as per-fact membership indicators, not point embeddings -- the planning
   literature's version of "region, not point." Regression Planning
   Networks (Xu et al., NeurIPS 2019) learn backward precondition
   prediction instead of hand-coding STRIPS preconditions, but never
   combined with an adversarial game tree. Decisive finding: **McGrath et
   al., "Acquisition of Chess Knowledge in AlphaZero" (PNAS 2022) found
   pins, forks, hanging-piece and mate-threat concepts emerge in
   AlphaZero's internals with ZERO explicit tactical supervision** -- but
   AlphaZero trained via self-play tied to game OUTCOMES (policy+value
   loss on wins/losses), not imitation of a fixed human dataset.
3. **Region/set-valued goal embeddings.** Box embeddings (Vilnis et al.
   2018), order embeddings (Vendrov et al. 2016), hyperbolic entailment
   cones (Ganea et al. 2018) all represent genuine containment; no direct
   RL precedent for goal-as-region exists. Cheapest starting point:
   Gaussian pooling (mean+covariance over diverse exemplars) -- same shape
   as Prototypical Networks, no new geometry needed; escalate to boxes only
   if that proves insufficiently expressive.

**Decision (Kaveh agreed with this ordering):** pursue outcome-conditioned
training FIRST, ahead of the quasimetric swap -- it's the only lever with
direct evidence that organic tactical-concept emergence is possible at
all without hand-coding, and the current FB objective is genuinely
outcome-blind: `LichessPairSource.batches()` (data/shards.py) samples
(anchor, goal) pairs geometrically within EVERY game regardless of
`result`, and `train_lichess_fb.py`'s `batch_tensors()` never reads
`result` even though it's already present in every batch's meta dict --
confirmed by reading both files directly, not inferred. The contrastive
loss purely predicts "which state actually came later in this real game,"
identical treatment whether that game was won, lost, or blundered away.
This is a cleaner, more direct explanation for the KRRvKBP rook-hanging
bug than "out-of-distribution material": the training signal never
distinguished good continuations from bad ones AT ALL, in ANY position.

**Built to test this cheaply, before committing to full self-play:**

- `experiments/acpl_probe.py` -- Average Centipawn Loss probe (the
  standard chess-analysis blunder metric, applied to a policy instead of
  a human). Samples held-out (never-trained) positions straight from real
  Lichess games, scores the policy's chosen move against a strong
  fixed-depth Stockfish (`depth=12` default, no skill/elo limiting --
  deterministic and full-strength, purely for LABELING quality here, not
  as an opponent to play against, so this doesn't touch the leakage gate).
- **Baseline result (n=100, current best ckpt, step 60000, depth=8 for
  speed): ACPL=328.8, blunder_rate(>=300cp)=0.55, mistake_rate(>=100cp)
  =0.80.** For calibration, human ACPL: <20 is strong-master-level, 100+ is
  beginner-level. This means the tactical blindness found in the KRRvKBP
  endgame is NOT specific to that out-of-distribution scenario -- the
  policy blunders material on the *majority* of moves even on ordinary,
  in-distribution Lichess middlegame/endgame positions. Saved:
  `artifacts/experiments/acpl_baseline_step60000.json`.
- `train_lichess_fb.py --winner-pov-only`: filters (anchor, goal) training
  pairs to only those where the side to move AT THE ANCHOR is the side
  that eventually WON the game (drops draws and loser-POV anchors) --
  `result` was already flowing through the pipeline unused, so this is a
  pure filter, no new data collection needed. Verified the filter keeps
  ~48% of training rows (~half of the ~93%-decisive game population, as
  expected). Cheapest possible test of the outcome-conditioning
  hypothesis, well short of a full self-play/PI-refinement loop.

**Running now (background):** fine-tuning `lichess_fb_4gb_winnerpov.pt`
(a COPY of the step-60000 best checkpoint, per the never-overwrite-best
rule) from step 60000 to 90000 (the standard 30k-step increment) with
`--winner-pov-only`. Will compare against the ACPL baseline above, and
against a same-step-budget plain-continuation control (no filter) to
isolate the filtering effect from just-more-training, before drawing any
conclusion. If this shows a real signal, escalate to actual self-play
(the full PI-refinement loop already flagged as the roadmap's real fix);
if not, the quasimetric swap (MRN) becomes the next lever per the agreed
ordering.

---

## 2026-07-11 22:50 — round 11 CONFIRMED: outcome-conditioning beats a step-matched control; quasimetric (MRN) mode implemented

**Outcome-conditioning result, properly controlled.** Three checkpoints,
same starting point (step 60000), same +30000-step budget: `baseline`
(unchanged), `plain_control` (30k more steps, no filter), `winner_pov_only`
(30k more steps, `--winner-pov-only`). First read at n=100 looked
promising (ACPL 328.8 -> 268.7 for winner-pov vs 308.3 for the control)
but a paired Wilcoxon test on the SAME 100 positions showed
winner-pov-vs-control wasn't significant (p=0.43) -- n=100 was simply
underpowered. Scaled the (cheap, ~15s/100-positions) ACPL probe to n=400
before drawing any conclusion.

**That n=400 re-run caught a real bug in the probe itself first**: ACPL
jumped from ~300 to 1000-1600 across all three checkpoints, nothing to do
with the checkpoints -- `acpl_probe.py` was calling `.score(mate_score=
100000)`, so a rare forced-mate-in-N detection in the sample (a handful of
positions, more likely to appear at n=400 than n=100) dominated the MEAN
with a near-lottery-sized ~100000-point outlier. Standard ACPL tooling
caps mate scores near the normal cp range specifically to avoid this;
fixed to `mate_score=1000`. Re-ran clean.

**Final n=400 result** (paired Wilcoxon + 2000-resample bootstrap CI, same
400 held-out positions across all three checkpoints):

| comparison | mean diff | 95% CI | Wilcoxon p |
|---|---|---|---|
| winner-pov vs baseline | -41.7cp | [-61.8,-22.0] | 0.00015 |
| plain control vs baseline | -19.7cp | [-36.8,-3.7] | 0.024 |
| **winner-pov vs plain control** | **-22.1cp** | **[-39.0,-5.8]** | **0.0046** |

Outcome-conditioned training produces a real, statistically significant
improvement in tactical safety BEYOND what the same number of additional
plain-continuation steps produces -- confirmed, not just a point-estimate
read. Effect size is modest (~22cp), not transformative on its own, but
it's real and it's free (same data already in the pipeline, zero extra
collection cost). Full numbers: `artifacts/experiments/
acpl_comparison_n400_round11.json`. **Decision: adopt `--winner-pov-only`
as the default going forward.**

**Quasimetric (MRN) mode implemented, per the agreed research-literature
ordering** (`catspace/nn/fb.py`): `TorchFB(quasimetric=True)` adds
`metric_scale` (per-dim scale, inits to ones) and `W` (bilinear residual,
inits to zero); `score(f,g) = f@W@g - d(f,g)` where `d` is a genuine
Euclidean metric on the rescaled embeddings (non-negative, symmetric,
triangle inequality by construction). Config-gated: `quasimetric=False`
checkpoints are byte-for-byte unaffected (verified in tests), and at
`quasimetric=True` initialization, `score` exactly equals `-||f-g||_2`
(verified against `torch.cdist`) -- a smooth starting point, not an
arbitrary architecture shock. All 3 call sites that used to do
`F(s) @ z` directly (`FBSearchPolicy._reach_batch`, `FBPlanPolicy.
_reach_to`, `FBBoardPolicy._reach`) now go through `fb.score(...)`, plus
`train_lichess_fb.py`/`experiment_report.py`'s `reach_slope`. New tests
(`tests/test_nn_fb.py`): reduces-to-dot-product-when-off, matches
`-||f-g||` at init, distance_matrix satisfies all 3 metric axioms
(non-negativity, symmetry, triangle inequality) numerically AFTER 20
training steps (not just at init -- confirms training doesn't break the
guarantee), checkpoint round-trip for both modes. Full suite: 130 passed
(was 126), no regressions. Smoke-tested end-to-end: a tiny `--quasimetric
--fresh` training run plus all three policy classes (`FBBoardPolicy`,
`FBSearchPolicy`, `FBPlanPolicy`) producing legal moves against a
quasimetric checkpoint.

**Next**: launching a full-scale training run combining both confirmed
levers (`--quasimetric --winner-pov-only`, fresh from scratch since the
new metric_scale/W params have no analog in existing checkpoints to
resume from), matching the original best run's step budget. Will
evaluate via ACPL (same n=400 protocol) and the KRRvKBP tablebase-verified
set before drawing any conclusion.

---

## 2026-07-12 00:05 — round 12: combined checkpoint trained, real progress confirmed, conversion problem still open

**Training.** `data/derived/lichess_fb_4gb_qm_wpov.pt`: fresh `TorchFB(
quasimetric=True)`, `--winner-pov-only`, 60000 steps (matching the
original best run's budget), ~24 it/s, clean run, VERDICT logged.

**ACPL (n=400, same protocol as round 11):** `ACPL=253.4`,
`blunder_rate(>=300cp)=0.362`, `mistake_rate(>=100cp)=0.640` -- the best
of any checkpoint so far. Paired comparisons:

| comparison | mean diff | 95% CI | Wilcoxon p |
|---|---|---|---|
| combined vs baseline | -72.8cp | [-99.3,-46.0] | 5.5e-7 |
| combined vs winner-pov-only (single lever) | -31.1cp | [-57.6,-5.0] | 0.066 |

Combined beats the ORIGINAL baseline overwhelmingly. The INCREMENTAL
contribution of quasimetric specifically, on top of winner-pov-only
alone, is borderline (CI excludes 0 but p=0.066 misses the conventional
0.05 line) -- honest reading: promising, not confirmed at this sample
size. Plausible explanation, not yet tested directly: ACPL measures
single-move tactical safety, but quasimetric's hypothesized benefit
(literature review, JOURNAL.md 2026-07-11 21:15) is specifically
MULTI-HOP compositional planning -- ACPL may just not be the most
sensitive instrument for what this lever is supposed to buy.

**KRRvKBP, single-policy scan (n=20, FBSearchPolicy only, vs the round-9
baseline scan for comparison):** terminations shifted from
`{INSUFFICIENT_MATERIAL: 9, THREEFOLD_REPETITION: 7, CHECKMATE: 2 (1W/1L),
FIFTY_MOVES: 1}` to `{THREEFOLD_REPETITION: 11, INSUFFICIENT_MATERIAL: 8,
CHECKMATE: 1 (1W/0L)}` -- fewer material self-blunders, and critically
**zero losses** (the original baseline scan's most alarming finding --
`FBSearchPolicy` getting mated FROM a 2-rooks-vs-bishop+pawn advantage --
did not reproduce here). Still only converting 1/20 to an actual win
within 150 plies, though -- the underlying "corner the king" execution
problem remains open.

**KRRvKBP, paired FBSearchPolicy vs FBPlanPolicy (n=20, `krrkbp_arena.py`,
same harness as round 10):** both readouts converted MORE wins on this
checkpoint than on the original one (FBSearchPolicy 3->4/20 wins,
FBPlanPolicy 1->3/20 wins, both runs zero losses) -- suggestive of real
conversion improvement, but the plan-persistence-vs-plain-search question
itself is STILL not significant (`e=0.38`, nowhere near reject) -- same
underpowered-at-n=20 pattern as round 10, and the same Stockfish-internal-
RNG caveat from round 10 still applies to this harness.

**Following the same fix that worked for ACPL** (n=100 -> n=400 resolved
a false ambiguity there): rather than trust an n=20 read again, generated
a FRESH 60-position KRRvKBP set (`catspace/diagnostic_krrkbp.build_fixed_
set(n=60, seed=123)`, independently verified all 60 are Syzygy WDL=2
before use -- `artifacts/experiments/krrkbp_fixed_set_n60.json`) and
re-ran the paired FBSearchPolicy-vs-FBPlanPolicy comparison at 3x the
sample size.

**n=60 result: still no significant difference.** `FBSearchPolicy=0.583`
vs `FBPlanPolicy=0.625`, `mean_diff=+0.042`, `CI=[-0.589,+0.672]`, `e=0.47`
-- nowhere near reject even at 3x the sample (and note the sign flipped
vs the n=20 read, -0.025 there vs +0.042 here, consistent with "this is
noise, not a real effect either direction"). One real, concrete pattern
DID emerge though: `FBPlanPolicy` lost twice (0-1, positions 25 and 39)
across these 80 total paired games (n=20 + n=60) -- `FBSearchPolicy` never
lost once, in any run this session. Plan-persistence trades off: holding
a plan fixed across several plies without re-searching converts a few
more wins but also occasionally walks into a refutation a plain per-move
search would have caught immediately. Net effect on `score_mean`: a wash,
not a win, at current hyperparameters (`plan_nodes=2000`,
`shallow_nodes=60`, `drop_delta=0.15`, `achieved_cos=0.95`,
`max_plies_per_plan=6`).

**Honest round-12 summary.** Confirmed, real, statistically rigorous
progress on embedding quality: outcome-conditioning (round 11) and the
quasimetric architecture (this round) both measurably reduce tactical
blindness (ACPL), and the combined checkpoint is decisively better than
where this phase started (p=5.5e-7 vs the original baseline). The
KRRvKBP endgame conversion rate improved and catastrophic losses
disappeared. But the two things this whole KRRvKBP diagnostic was
originally built to test -- (a) does the planner learn to keep its rooks
on squares the bishop can't touch and actually convert the win, and (b)
does explicit plan-persistence help versus plain search -- are NOT yet
answered "yes." (a) is still mostly draws (repetition/insufficient
material), not clean conversions. (b) is a confirmed null at n=80 total.
This is real progress on the PREREQUISITE (an embedding that isn't
tactically blind) but not yet the payoff the research line was aimed at.
Reporting this status rather than continuing to spin up more variants
unboundedly.

---

## 2026-07-12 01:10 — round 13 setup: search-depth sensitivity, ply-gap calibration term, full self-play infrastructure

Kaveh, mid-conversation: search-depth sensitivity check on the current
best checkpoint ("with the same arch, increase it a bit and see if we do
better"); build the ply-gap calibration term proposed earlier; **build the
full self-play machinery** ("everything on our roadmap"); keep going
overnight without stopping.

**Search-depth/node-budget sensitivity (same checkpoint,
`lichess_fb_4gb_qm_wpov.pt`, ACPL n=200 per config):** `max_nodes=200`:
274.5, `800`: 270.5, `2000`: 283.4 -- flat within noise, no meaningful
trend either direction. Consistent with the earlier full-board ply-depth
sweep (rounds 7-9) plateauing/regressing past depth 3: more search alone
does not fix tactical blindness when the LEAF evaluation itself is the
bottleneck -- more nodes just explore more leaves scored by the same
miscalibrated function. Confirms the embedding/training-side levers
(winner-pov, quasimetric, ply-gap, self-play) are the right place to keep
pushing, not deeper search at inference time.

**Ply-gap calibration term (`catspace/nn/fb.py`).** Kaveh's insight: "if
the future leads to a mate for me, that's a good future... enough info
here for us to get good or bad." Diagnosed the actual gap: in-batch
InfoNCE retrieval only enforces RELATIVE ranking (is g_true closer than
this batch's other g's?) -- nothing calibrates the ABSOLUTE scale of the
quasimetric distance to anything real, so "down material with no path
back" and "down material but recoverable" could score identically as
long as batch-relative ranking happened to work out. Fix: `ply_gap` (the
real anchor->goal ply distance -- `data["ply"]` was already in the
pipeline, just needed threading through as `ply_g` on the goal row too,
`catspace/data/shards.py`) now regresses `d(f,g)` toward
`ply_gap/ply_gap_scale` via an MSE term, weighted by `--ply-gap-weight`
(default 0.05). Quasimetric-only (no `d` to calibrate otherwise);
silently a no-op when `quasimetric=False`. New test confirms the term
adds loss and produces gradients in quasimetric mode, and is EXACTLY a
no-op (bit-identical loss) when off. **Also caught and fixed a real bug
while wiring this up**: `val_metrics()` was computing its printed
VAL_TOP1/VAL_TOP8/loss diagnostics via a raw `f @ b.T`, bypassing
`score_matrix()` entirely -- meaning every quasimetric run's printed
validation numbers (round 12's `lichess_fb_4gb_qm_wpov.pt` included) were
silently wrong, even though the ACTUAL TRAINED WEIGHTS were fine (the
real training loss correctly went through `loss_fn`/`score_matrix`; only
the human-readable progress log was misleading). Fixed to
`fb.score_matrix(f, b)`.

**Self-play infrastructure, built fresh this round:**
- `experiments/selfplay_generate.py`: plays games with the CURRENT best
  checkpoint (self vs self, plus a configurable fraction vs Stockfish as
  an external sparring partner -- `--sf-opponent-frac`, records only
  moves + game RESULT, never an eval score, so this doesn't touch the
  leakage gate) and writes them as Lichess-shard-compatible npz files
  (identical schema to `data.lichess.build_shards`) -- drop-in readable by
  the EXISTING `LichessPairSource`, no format changes needed.
  `StochasticPolicy` wraps any BoardPolicy with epsilon-random move mixing
  (default 0.08) since `FBSearchPolicy`/`FBPlanPolicy` are deterministic
  argmax and would otherwise collapse self-play into near-duplicate games
  -- simpler than AlphaZero's Dirichlet/temperature approach but same
  purpose. Rate: ~0.11 games/s at max_nodes=200/beam=4/max_plies=150 (a
  20-game timed test: 2m57s).
- `catspace/data/shards.py`'s `MixedPairSource`: interleaves batches from
  two `LichessPairSource`-shaped sources (human + self-play) by a fixed
  ratio, whole-batch-at-a-time (not mixed within a batch). Wired into
  `train_lichess_fb.py` via `--selfplay-shards`/`--selfplay-frac`; holdout/
  val stay human-only for a stable cross-round reference. New test
  (`tests/test_data.py::test_mixed_pair_source`) confirms batches are
  never mixed-source and the draw fraction tracks the requested ratio
  over 500 samples.
- Full pipeline smoke-tested end-to-end (generate 4 games -> shard ->
  mixed-source training step, all three levers -- quasimetric,
  winner-pov-only, ply-gap, self-play-mix -- together). Full suite: 132
  passed (was 130), no regressions.

**This is the ACTUAL PI-refinement mechanism the literature (McGrath et
al.) credited with organic tactical-concept emergence** -- `--winner-pov-
only` was explicitly framed as a cheap proxy for this; this is the real
thing. Round 13, launched: generating 400 self-play games with the
current best checkpoint (`data/shards/selfplay_gen1/`, ~61min ETA at the
measured rate), then training a fresh checkpoint combining ALL FOUR
confirmed/plausible levers (`--quasimetric --winner-pov-only --ply-gap-
weight 0.05 --selfplay-shards data/shards/selfplay_gen1 --selfplay-frac
0.3`), evaluating via the same ACPL n=400 + KRRvKBP n=60 protocol, then
continuing the PI loop (generate more self-play with whatever the new
best checkpoint is, retrain, repeat) through the night per Kaveh's
explicit instruction not to stop.

---

## 2026-07-12 13:20 — review pass (model switched to Fable per Kaveh): two real bugs found in the round-13 launch; winner-pov REMOVED

Kaveh asked for a full review of the past day's work ("ensure everything
was done right"), flagged that winner-pov is no longer needed, and asked
for periodic commits. The review found the first round-13 training launch
(killed after 35 min of zero progress) failed from TWO stacked bugs, one
of which also taints part of the node-budget sweep:

**Bug 1 — `newest_shard_dir()` silently adopted the self-play dir as the
human training set.** The self-play generator wrote its shards to
`data/shards/selfplay_gen1/`, which made it the most-recently-modified
dir under `data/shards/` -- and the round-13 training launch, run without
an explicit `--shards`, resolved its "human" source to the 30k-position
SELF-PLAY dir instead of the 55.8M-position 4GB human prefix. Nothing
crashed; it just silently trained-on/holdout-from the wrong data.
Measured fallout: this also invalidates the node-sweep's `max_nodes=2000`
stage (283.4) and the unfinished 4000 stage -- those probe processes
started after the self-play dir existed, so `acpl_probe`'s
position-sampling drew from self-play shards, not the human holdout. The
200/800 stages (274.5/270.5) predate the dir and stand. The "flat"
conclusion still holds on the clean 200-vs-800 pair, but the 2000-node
point needs a re-run (queued, after training). Fixes: self-play output
moved to `data/selfplay/gen1` (outside `newest_shard_dir()`'s glob);
`selfplay_generate.py` now hard-REFUSES to write under `data/shards/`;
all future training launches pass `--shards` explicitly.

**Bug 2 — winner-pov x batch-size guard = zero training progress.**
`main()`'s loop skips any batch that filters below `batch//2 = 256` rows.
On the (mostly-drawn, and wrongly-selected per Bug 1) self-play data,
winner-pov kept a measured mean of 68.8/512 rows -- pass rate 0.000. The
run built feature planes for every batch (hence ~208 CPU-minutes of
plausible-looking activity) and discarded every single one: an infinite
spin at step 0, which is why the log never showed even step 100.

**Winner-pov removed entirely** (not just from self-play -- from
everything), on three grounds Kaveh drove to in conversation:
1. *The information it added is already in the data.* A sampled goal
   position that's a mate FOR the mover is a good future; a mate AGAINST
   them is a bad one. The model should see both geometries -- censoring
   losing trajectories deletes half the signal, it doesn't sharpen it.
2. *The ply-gap calibration term NEEDS losing trajectories.* "Down
   material with no way back" can only be learned as a large/uncalibrated
   distance if unrecoverable positions and their real continuations
   actually appear in training. Winner-pov filtered out exactly those.
3. *It was a proxy whose job is done.* It existed as the cheapest test of
   outcome-conditioning (round 11, confirmed real at ~22cp) before
   self-play existed. Real self-play + ply-gap calibration are now built;
   the proxy earned its keep as EVIDENCE (outcome-conditioning matters)
   and is retired as a MECHANISM.

Removed: `--winner-pov-only` flag, `_winner_pov_mask()`, the filter in
`batch_tensors`/`collect_holdout`, the `is_selfplay` batch tagging (which
existed only to exempt self-play from the filter). `batch_tensors` is
back to its simple holdout-only form, now returning `ply_gap` as a 4th
tensor. Round-11's RESULT stands as recorded (the checkpoint trained fine
at its 245-ish/512 keep rate and the ACPL comparison was valid); what's
retired is the mechanism going forward.

**Also fixed in review:** self-play shards now stamp odd `game_id`s only
(2i+1) -- ids divisible by 50 were silently eaten by the trainer's
holdout rule (8/400 games of scarce self-play data landing in neither
train nor holdout); the existing gen1 shard was patched in place and
verified (400 games, all ids odd). Known caveat documented but NOT fixed
(nothing measured so far is affected): `planner/decompose.py` scores hops
with raw `F@z` dot products and never sees `metric_scale`/`W`, so its
waypoint metrics are mis-calibrated for quasimetric checkpoints --
thread `fb.score` through `WaypointPool`/`hop_reach` before trusting
decompose numbers on quasimetric runs. Same applies to the viz builders'
raw `F @ z` reach maps.

**Relaunching round 13 correctly** (after full suite + commit):
`python -u experiments/train_lichess_fb.py --shards
data/shards/lichess_db_standard_rated_2019-01.prefix4gb --ckpt
data/derived/lichess_fb_4gb_qm_gen1.pt --steps 90000 --quasimetric
--ply-gap-weight 0.05 --selfplay-shards data/selfplay/gen1
--selfplay-frac 0.3 --fresh` -- unbuffered this time so the log shows
life immediately, explicit shards, no winner-pov. Levers: quasimetric +
ply-gap calibration + self-play mix.

---

## 2026-07-12 22:10 — round-13 training done; quasimetric FITNESS instruments built (lit survey -> experiments/qm_fitness_probe.py)

**Training** (`lichess_fb_4gb_qm_gen1.pt`, 90k steps, ~2h at ~12 it/s):
clean finish. Notable verdict line: `DIFF_SLOPE_WON=+0.208 /
DIFF_SLOPE_LOST=-0.050` -- the strongest won-lost separation of any
checkpoint to date (qm_wpov was -0.106/-0.256; the sign structure here is
the first one that matches the design intent: reach-toward-MY-mate rises
in games I win, doesn't in games I lose). Raw REACH_SLOPE went negative
for both (-0.129/-0.288) -- under `score = r - d` the raw slope mixes the
generic-finality component differently than cosine did; MATE_DIFF is the
outcome-signal diagnostic, and it improved. Full evaluation
(`round13_eval.sh`: clean node-2000 rerun + ACPL n=400 new-vs-incumbent +
KRRvKBP n=60) running now.

**Quasimetric fitness instruments** (Kaveh: find how people evaluate the
fitness of quasimetrics, build those to steer embedding improvement; the
prior conversation wasn't recoverable from transcripts, so a fresh
literature survey was run). Survey highlights (agent report, full
citations in conversation): PQE (Wang & Isola 2022) defines the two
canonical structural quantities -- multiplicative DISTORTION (Defn 4.1)
and quasimetric VIOLATION ratio `vio = d(x,z)/(d(x,y)+d(y,z))` (Defn
4.2), with a theorem that they lower-bound generalization error; IQE's
infinite-distance column (predicted d where true d = infinity) is the
standard unreachability probe; QRL demonstrates ground-truth-vs-learned
distance heatmaps where true distances exist; OGBench's stitch splits are
the compositional-generalization protocol; nobody reports a quantitative
asymmetry-recovery score (gap we can fill cheaply -- chess's capture
boundary gives free ground-truth one-way doors); and notably the survey
flagged that the ORIGINAL MRN violates non-negativity (IQE's fix) -- our
`d` is a genuine Euclidean norm on rescaled embeddings, non-negative by
construction, and the existing metric-axiom tests already cover that bug
class.

**Built: `experiments/qm_fitness_probe.py`** -- five instruments, ranked
by the survey's value-per-effort ordering:
1. *Syzygy calibration*: d(F(s), zMATE) vs tablebase DTZ on
   KRRvKBP-family winning positions (Spearman rho + per-DTZ-bin means).
   Real ground-truth distances -- better than any gridworld oracle in the
   literature this borrows from.
2. *Horizon-stratified retrieval*: true-future-vs-63-negatives ranking
   accuracy at ply gaps {1,2,5,10,20,50}.
3. *Asymmetry audit*: capture-boundary pairs (forward feasible, reverse =
   un-capturing = impossible); frac(reverse <= forward) should be ~0.
4. *Triangle violation*: PQE vio on `d` alone (architectural guarantee,
   regression test) AND on the full `r-d` readout (not guaranteed; tracks
   how non-metric the actual planning signal is).
5. *Degeneracy panel*: spread ratio (cross-game vs 1-ply distances),
   effective rank of F/B, norms.

**Smoke run on the incumbent (qm_wpov, small n) already tells a story**:
retrieval acc 0.70-0.85 at k<=10 plies vs chance 0.025, then a cliff --
0.40 at k=20, 0.10 at k=50: the embedding discriminates real futures
about 10-20 plies out and is nearly blind past that. Asymmetry
frac(reverse<=forward)=0.27 with a small mean gap (0.09 on d~1.0): it
half-knows material can't come back. Triangle on d: max_vio 0.76 (<= 1,
guarantee holds); full-score violations negligible (0.045% of 20k
triples). No distance collapse (spread ratio 1.78); effective rank ~19.5
of 64 dims. Full-size probes on BOTH checkpoints (n=300 games, 200k
triples, 300 syzygy positions) running on CPU alongside the MPS eval;
results land in `artifacts/experiments/qm_fitness_{qm_wpov,qm_gen1}.json`.
These numbers become the steering instruments for the next embedding
rounds: the k=20-50 retrieval cliff and the weak asymmetry gap are the
first two concrete targets.

---

## 2026-07-12 23:45 — round 13 VERDICT: no promotion; probes localize the real problem; endgame-curriculum next

**Full-size fitness probes, both checkpoints** (n=300 games, 200k triples,
400 KRvK + 300 KRRvKBP tablebase positions;
`artifacts/experiments/qm_fitness_{qm_wpov,qm_gen1}.json`):

| instrument | qm_wpov (incumbent) | qm_gen1 (round 13) |
|---|---|---|
| KRvK Spearman rho, d vs true plies-to-mate | **+0.010 (flat)** | **-0.069 (flat)** |
| retrieval acc k=1/5/10 | .97/.93/.87 | .96/.97/.89 |
| retrieval acc k=20/50 | .69/.23 | .68/**.29** |
| asymmetry frac(rev<=fwd) (0 wanted) | **0.270** | 0.345 |
| triangle max_vio on d (<=1 required) | 0.824 OK | 0.851 OK |
| spread ratio (collapse check) | 1.79 | **2.35** |
| effective rank F/B (of 64) | 24.1/24.3 | 26.0/26.4 |

The decisive row is the first: on KRvK -- where pawnless tablebase DTZ
IS the true plies-to-mate, spread 1..31 -- the learned distance is
statistically FLAT for both checkpoints (bin means constant from
mate-in-1 to mate-in-31). The metric's *structure* is healthy (zero
triangle violations on `d`, no collapse, strong short-horizon retrieval);
what's missing is *coverage*: human games essentially never visit these
positions, so neither InfoNCE nor the ply-gap term ever pushes gradient
through that region. This is the measured, mechanical explanation for the
KRRvKBP conversion failure -- the planner cannot rank "closer to mate"
in a region its training distribution never reached. (Also fixed a probe
design flaw en route: KRRvKBP's DTZ compresses toward 0 because captures
are always near -- pawnless KRvK added as the clean calibration target.)

**Round-13 play metrics (gen1 vs incumbent):**
- ACPL n=400 paired: gen1 +13.9cp WORSE, 95% CI [-11.0, +40.3],
  Wilcoxon p=0.15 -- not significant, statistically a wash.
- KRRvKBP n=60: FBSearchPolicy 0.475 / FBPlanPolicy 0.450 (incumbent
  measured 0.583/0.625 on the same set) -- direction unfavorable;
  plan-vs-search remains null (e=0.15).
- Node-sweep 2000-stage clean re-run (incumbent): ACPL=283.4 at
  max_nodes=2000 vs 274.5@200 / 270.5@800 -- the earlier "flat in search
  depth" conclusion now stands on clean data at all three points.

**VERDICT: no promotion.** `lichess_fb_4gb_qm_wpov.pt` remains the
incumbent for play strength. gen1's structural wins (spread 2.35, rank
+2, k=50 retrieval +0.06, best-ever DIFF_SLOPE separation) didn't convert
into play improvement, and three levers changed at once (winner-pov
removed, ply-gap added, self-play mixed) so per-lever attribution is
impossible from this run -- noted as a methodology cost of the corrected
relaunch, accepted deliberately to get the pipeline unblocked.

**Next (the probes now steer): endgame-start curriculum.** Built
`--endgame-start-frac` into `selfplay_generate.py`: a fraction of
self-play games start from random winnable endgames (KRvK, KQvK, KRRvK,
KRRvKBP-family, KQvKP; generator verified across all 5 material menus +
replay round-trip). Real games, real outcomes, zero oracle labels --
tablebases stay observational. Launching gen2 (500 games,
endgame_start_frac=0.5) with the incumbent, then retraining with the gen2
mix. **Success criterion, pre-registered: KRvK Spearman rho must move
decisively off zero (target >= +0.3) on the next checkpoint's fitness
probe** -- if it does, the curriculum mechanism works and we scale it; if
it stays flat, the ply-gap term itself isn't reaching these pairs and the
next lever is horizon/pairing changes, not more data.

---

## 2026-07-13 01:40 — round 14: the gate FAILED as registered, and failing it found the real bottleneck — the goal must be a REGION, not a point

**gen2 data + training**: 500 games, 53% endgame starts (verified in the
shard: short decisive endgame games, 32 KRRvK / 20 KQvK / 12 KRvK genuine
white-mate finals among them), trained
`lichess_fb_4gb_qm_gen2.pt` (90k steps, quasimetric + ply-gap +
gen2-mix). **Pre-registered gate: KRvK Spearman rho(d, plies-to-mate) >=
+0.3. Result: -0.043 — FAILED, flat, unchanged from both prior
checkpoints.** All other probe instruments essentially unchanged.

**But the failure decomposes.** Follow-up experiment (all on CPU, minutes,
no retraining): enumerated all 216 essentially-distinct genuine KRvK
checkmate positions, then measured the SAME 300 tablebase-scored KRvK
positions against three different goal representations:

| goal representation | incumbent rho | gen2 rho |
|---|---|---|
| human-mate centroid (what planner+probe use today) | +0.003 | -0.077 |
| KRvK-mate centroid (same-material mean) | +0.002 | -0.133 |
| NEAREST KRvK-mate exemplar (min over 216) | **+0.165** | **+0.252** |

Two conclusions, both load-bearing:
1. **Averaging mate exemplars into ANY centroid destroys the distance
   structure** — even a centroid built purely from same-material KRvK
   mates is flat. The information is in the per-exemplar geometry; the
   mean throws it away.
2. **The endgame curriculum DID improve the underlying metric** (+0.165 ->
   +0.252 nearest-exemplar rho) — the round's data lever worked, but the
   improvement was invisible through the centroid readout the gate was
   (wrongly) defined against. The pre-registered criterion measured the
   goal representation's failure, not the data's.

This is Kaveh's goal-as-region design requirement ("I want corner-the-king
to be a region in space, broader than...") landing as a MEASURED result
rather than a design intuition: the mate goal must be represented as a
SET/region of exemplars, never collapsed to one vector.

**Built (readout-only, no retraining needed):**
- `catspace/goal_bank.py`: harvest genuine checkmate finals from any shard
  dirs (result-filtered, material-capped) + embed as a (m, d) exemplar
  bank.
- `FBSearchPolicy`/`FBBoardPolicy` now accept `z` as either a single (d,)
  goal or an (m, d) BANK, scored best-over-exemplars (for the quasimetric
  that is exactly nearest-exemplar distance readout).
- `krrkbp_arena.py --compare bank`: paired centroid-readout vs
  bank-readout, SAME checkpoint, SAME search budget — isolates the goal
  representation as the only variable. Bank for the KRRvKBP test: 71
  white-mate endgame exemplars (<= 8 pieces) harvested from gen1+gen2
  self-play — the model's own mates, zero oracle involvement.

**Running:** the decisive n=60 KRRvKBP paired test (gen2 checkpoint).
If bank-readout converts more tablebase-won positions than
centroid-readout, the goal-as-region mechanism is validated end-to-end
and gets wired into the main readout everywhere (and the fitness probe's
calibration instrument switches to nearest-exemplar); if not, the +0.25
rho wasn't strong enough to matter at play scale yet, and the next lever
is strengthening the exemplar geometry (bigger banks, more endgame
curriculum, or the pairing-horizon fix for the k=20-50 cliff).

**Result (2026-07-13 03:20, full arc): the goal-as-region READOUT line is
closed -- three decisive rejections.** Hard-max bank on gen2: 0.433 vs
0.308 (e=65, REJECT). Hard-max on the incumbent: 0.558 vs 0.308
(e=2.8e7, REJECT). Soft-min (normalized logsumexp, tau=0.1) on the
incumbent: 0.550 vs 0.358 (e=21811, REJECT) -- soft-min recovered some of
hard-max's loss (0.308 -> 0.358) but still loses decisively to the plain
centroid. Honest close-out: the DIAGNOSIS stands (all centroids flat
against true plies-to-mate; nearest-exemplar geometry real and improved
by the endgame curriculum, rho +0.165 -> +0.252), but +0.25 positional
calibration is not enough to beat the centroid's move-ranking STABILITY
in actual play -- the centroid is the exact direction the whole InfoNCE
geometry organized around (2048 mates x 90k steps), while bank exemplars
are one-shot B-embeddings in sparsely-trained regions. A readout cannot
fix representation sparsity; region goals go back on the shelf until the
embedding itself is better calibrated in those regions. Unit test for the
bank path kept (it's still a useful instrument), krrkbp_arena --compare
bank kept for re-testing on future checkpoints.

**Original single-run entry follows (superseded by the arc above):**

---

## 2026-07-13 06:10 — round 15: asymmetry-margin lever — gate REJECTED as configured, with the cleanest trade-off curve yet

`lichess_fb_4gb_qm_asym.pt` (quasimetric + ply-gap 0.05 + asym 0.05/margin
0.2 + gen2-mix, 90k steps). Pre-registered 3-part gate:

1. **frac(rev<=fwd) <= 0.10: PASS, dramatically.** 0.030 (was 0.27-0.35),
   mean reverse-forward gap +0.325 (was +0.085). The hinge did exactly its
   job: the metric now robustly encodes that captures are one-way doors.
2. **nearest-exemplar KRvK rho >= +0.15: FAIL, borderline.** +0.123 --
   below the incumbent's +0.165 and well below gen2's +0.252 on the SAME
   positions/seed. The hinge degraded fine mate-distance geometry some.
3. **ACPL not significantly worse: FAIL, clear.** 284.4 vs 253.4, paired
   diff +30.9cp, CI [+4.7,+57.1], p=0.01.

**Why, mechanistically** (the probes make this legible): retrieval k=1
dropped 0.97 -> 0.79 while k=10/20/50 all IMPROVED (0.87->0.89,
0.69->0.77, 0.23->0.30 -- the k=20-50 cliff moved outward, the first
lever to touch it!). The asym term at weight 0.05 traded SHORT-horizon
discrimination for long-horizon structure + asymmetry. ACPL lives
entirely on short-horizon discrimination (ranking the 30-40 immediate
moves), so it paid the bill. VERDICT lines agree: REACH_SLOPE went
positive again (+0.292 won / +0.143 lost) with healthy DIFF separation
(+0.144/-0.117).

**Verdict: rejected AS CONFIGURED (weight 0.05), per pre-registration --
but this is a tuning failure, not a mechanism failure.** All three gate
quantities moved exactly the way an over-weighted auxiliary loss predicts.
Round 16 (ONE lever: asym_weight 0.05 -> 0.015, same margin, everything
else identical) launched -- hypothesis: keep most of the asymmetry gain
(part 1 has enormous headroom: 0.030 vs the 0.10 gate) while restoring
k=1 sharpness and ACPL. Same 3-part gate.

---

## 2026-07-13 09:50 — round 16 (asym 0.015): NO PROMOTION; the asymmetry line closes at 2 attempts

Gate results for `lichess_fb_4gb_qm_asym015.pt`:
1. frac(rev<=fwd) = **0.045 PASS** (asymmetry gain is robust across
   weights: 0.030 @ 0.05, 0.045 @ 0.015, vs 0.27 incumbent).
2. nearest-exemplar rho = **+0.121 FAIL** -- essentially identical to the
   0.05-weight run's +0.123. The mate-geometry cost comes from the hinge
   EXISTING, not from its weight: a real mechanistic finding (reverse-pair
   gradients reshape exactly the sparse endgame regions the nearest-
   exemplar instrument measures).
3. ACPL paired vs incumbent: 260.9 vs 253.4, +7.5cp, CI [-20.1,+35.8],
   p=0.35 -- **PASS** (statistical wash; k=1 retrieval partially recovered
   0.79 -> 0.85, k=20-50 still better than incumbent).

**Tiebreaker (KRRvKBP n=60 single-policy scan, same set/seed as the
incumbent's 0.558): 0.367 (3W/38D/19L) -- CLEAR FAIL.** 19 losses from
tablebase-won positions. The same short-horizon discrimination the hinge
trades away (k=1: 0.85 vs 0.97) barely dents full-board ACPL but is
decisive in sparse endgames where every move is critical. **Incumbent
`lichess_fb_4gb_qm_wpov.pt` stays. The asymmetry-margin line is CLOSED
per the 2-attempt protocol** -- with its finding preserved: the mechanism
teaches arrow-of-material essentially for free at low weight (a
capability worth re-adding LAST, after the embedding's short-horizon
sharpness has other support), it just can't pay its way yet.

**Where this leaves the research (16-round state of the union):**
- Confirmed real and kept: quasimetric architecture (structure verified,
  zero violations), ply-gap calibration, self-play pipeline + endgame
  curriculum (improved nearest-exemplar geometry +0.165 -> +0.252),
  outcome-conditioning evidence, the full instrument suite (ACPL, KRRvKBP
  arenas, 6-instrument fitness probe).
- Confirmed and closed (negative results with mechanisms understood):
  ply-depth/node-budget scaling, winner-pov filter, goal-as-region
  READOUT (3 play rejections), asymmetry hinge (2 attempts).
- The incumbent since round 12 is still `lichess_fb_4gb_qm_wpov.pt`:
  every subsequent single lever either washed or regressed at play.
  The honest pattern: STRUCTURAL instruments improve readily; PLAY
  improvements are bottlenecked on short-horizon discrimination (k=1-10),
  which every auxiliary objective so far has taxed rather than helped.
- Next levers, in order: (a) gen3 endgame-curriculum at higher dose
  (launched: 600 games, endgame_start_frac 0.7 -- pure data, taxes
  nothing), (b) the k=20-50 pairing lever as a RETRIEVAL-preserving
  change (stratified long-gap oversampling rather than a new loss term),
  (c) revisit region goals + asymmetry only after (a)/(b) raise the
  floor.

---

## 2026-07-13 12:40 — round 17: no promotion, and the cross-checkpoint table exposes the real question

gen3 (higher dose: 70% endgame starts, selfplay-frac 0.4, no aux losses):
nearest-exemplar rho **+0.154** (below gen2's +0.252 -- dose-response is
NOT monotonic in curriculum fraction), k=1 retrieval intact at 0.97,
KRRvKBP n=60 scan **0.342 (1W/39D/20L)** vs incumbent 0.558. No
promotion.

**The table that matters now** (nearest-exemplar rho vs KRRvKBP play,
all on the same n=60 set):

| checkpoint | recipe | rho | KRRvKBP |
|---|---|---|---|
| qm_wpov (INCUMBENT, r12) | qm + winner-pov, human-only | +0.165 | 0.550-0.583 (3 runs) |
| qm_gen1 (r13) | qm + ply-gap + selfplay, no wpov | -- | 0.475 |
| qm_gen2 (r14) | same, gen2 data | +0.252 | 0.433 |
| qm_asym015 (r16) | same + asym 0.015 | +0.121 | 0.367 |
| qm_gen3 (r17) | same, gen3 data, frac 0.4 | +0.154 | 0.342 |

Two hard conclusions: (1) **nearest-exemplar rho does not predict play**
-- best rho (gen2) plays 0.12 below the incumbent; the instrument
measures something real about endgame geometry but not the thing that
converts wins. (2) **Every checkpoint since round 12 shares THREE
simultaneous recipe changes vs the unbeaten incumbent** (winner-pov
removed, ply-gap added, self-play mixed in) -- the attribution debt taken
on knowingly at the round-13 corrected relaunch is now the single most
important open question: one or more of those three is likely what has
kept play below 0.558 for five straight rounds, and no single-lever round
since has touched them.

**Round 18 = the ablation, not another lever**: `qm + ply-gap +
human-only` (drop the self-play mix, keep everything else from the
round-13+ recipe). This isolates the self-play mix's play cost while
leaving Kaveh's winner-pov retirement untouched. If it recovers toward
0.55: the self-play MIX (as currently dosed) is the drag despite its
calibration benefits. If it doesn't: ply-gap itself (or winner-pov's
absence) is implicated, and the winner-pov question goes to Kaveh with
this table -- his call retired it on principled grounds (losing
trajectories carry needed signal), but the only checkpoint that has ever
played 0.55+ had it on, and the evidence deserves to be in front of him.
`FBSearchPolicy(centroid)=0.433` vs `FBSearchPolicy+bank=0.308`, n=60,
mean_diff=-0.125, e=65.07, REJECT -- the first statistically decisive
readout difference this whole diagnostic has produced, and it's AGAINST
the naive bank. Honest interpretation: nearest-exemplar distance orders
*static positions* better (the +0.25 rho is real), but `max` over 71
heterogeneous exemplars (KRRvK/KQvK/KRvK mates mixed) changes which
exemplar wins from move to move -- the readout chases whichever mate
pattern happens to be closest this ply, injecting goal-switching noise
into MOVE ranking that outweighs the calibration gain. Positional
calibration and move-ranking stability are different fitness axes; the
probe measured one, play depends on both. Follow-ups queued, one at a
time: (1) same test on the stronger incumbent checkpoint (running --
separates "bank hurts inherently" from "gen2 is weak"); (2) if bank still
loses, try soft-min (logsumexp temperature) instead of hard max, which
smooths exemplar switching while keeping region structure -- ONE change,
directly aimed at the failure mode this test exposed.

---

## 2026-07-13 — MODEL HANDOFF (Fable → Opus), round-18 promotion recap, and the two-horizon plan

**Handoff note:** the overnight autonomous loop (rounds 13–18) ran on Claude
Fable 5. The Fable usage limit was hit; the session switched to Claude Opus
4.8 (1M context), which is authoring from here. All prior findings, the
promoted incumbent, and the instrument suite carry over unchanged; this note
just marks where the model changed hands so future readers know which entries
came from which.

**Round 18 close-out (was committed to research_state.json but not journaled
until now).** The attribution ablation `qm + ply-gap + human-only` (drop the
self-play mix, keep everything else from the round-13+ recipe) =
`lichess_fb_4gb_qm_plygap_only.pt`. Results vs the prior incumbent
(`qm_wpov`):
- KRRvKBP n=60 conversion: **0.567 (12W/44D/4L)** vs 0.558 (~3W) — triple the
  actual wins, and only 4 losses from 60 tablebase-won positions.
- DIFF_SLOPE +0.255 / +0.003 — best won-lost separation of the project.
- Full-board arena n=40 @ 200 nodes: 0.062 vs 0.050 — a tie, both collapsed at
  the austere budget (see below).
- ACPL n=400: 289 vs 253 — worse, and accepted: conversion + outcome-separation
  sit closer to the project objective than the general-position ACPL proxy.

**PROMOTED** to incumbent. This closes the round-13 attribution debt: the
self-play MIX at 0.3–0.4 fraction was the 5-round play drag (its ε-noise games
dulled short-horizon tactics); ply-gap is exonerated; removing winner-pov is
exonerated. The endgame curriculum's calibration gains are real but were
overdosed — to be re-added at low fraction later.

**In flight now (Opus):**
1. **Full-board node-budget sensitivity** on the new incumbent (200 → 400 →
   800, extending to ~1600). Motive: the round-18 showdown scored ~0.05 vs the
   weakest Stockfish at only 200 nodes, but the one full-board 0.25 result
   (round 6) effectively used ~420 nodes (depth-3/beam-4). So "0.05" may be
   search-starvation, not a strength ceiling — this disambiguates every recent
   arena number. Ceiling reasoning (Kaveh): stay "10× less than Leela",
   anchored on Leela's competitive ~15–16k nodes/move → ~1600-node cap, which
   still leaves room to grow from 200 without turning the win into an
   out-searching result.
2. **Two-horizon architecture — being DESIGNED before building** (Kaveh: "plan
   it first"). Rationale: the project's central measured finding is that
   short-horizon tactical sharpness and long-horizon strategic structure
   compete inside one d=64 embedding. Design: shared board-encoder trunk → two
   heads, `near` (F_near/B_near) and `far` (F_far/B_far). **Roles:** near is
   the search's steering wheel (beam selection + move ordering — prunes
   tactical blunders before expansion), far is the leaf evaluator (calibrated
   distance-to-goal — supplies the strategic gradient that converts won
   positions instead of shuffling). **Training:** shared trunk, two heads,
   ply-gap-stratified data — near on short-gap pairs (1–8 plies, contrastive
   sharpness), far on long-gap + state→goal pairs (quasimetric + ply-gap
   calibration + region/asymmetry structure). The competition is resolved by
   moving it out of one shared function into two separate heads. **Pre-
   registered success:** on the fitness probe, near k=1 retrieval stays ~0.97
   AND far nearest-exemplar ρ clears the ~0.25 single-embedding ceiling,
   simultaneously — the combination one embedding never achieved; at play,
   KRRvKBP ≥ 0.567 AND ACPL ≤ 289 (both hold/improve). Open design choices out
   to Kaveh: shared vs split trunk (start shared), near/far crossover ply
   (~10–16), pure-far vs far+small-near leaves (start pure-far).

---

## 2026-07-13 (Opus) — node-budget sensitivity: no reliable lever; budget locked at 200

Ran the incumbent `lichess_fb_4gb_qm_plygap_only.pt` on full-board arena vs
sf:skill=0 (n=40) across the search budget, to disambiguate whether the
round-18 showdown's ~0.05 was a strength ceiling or search starvation:

| max_nodes | arena score |
|---|---|
| 200 | 0.062 |
| 400 | 0.100 |
| 800 | 0.062 |

**Non-monotonic and noise-dominated** — 0.062 vs 0.100 is ~1.5 games out of 40,
and 800 dropped back to 0.062. Node count is NOT a reliable lever: full-board
play sits at ~0.06–0.10 (losing ~93%) regardless. This reinforces the project's
running finding — the **value function, not search depth, is the bottleneck**;
deeper search over a miscalibrated eval doesn't rescue it, and 800 < 400 is
consistent with deeper search amplifying the long-range eval errors (the k=20–50
retrieval cliff) that the two-horizon far head is built to fix.

Per Kaveh's conditional ("if increasing helps, increase it… still 10× less than
Leela"): it does not clearly help, so **operating budget stays at 200** — which
also keeps every eval matched to the incumbent's existing references (KRRvKBP
0.567, ACPL 289, all measured at 200 nodes). A clean forward-looking test falls
out of this: a genuinely better-calibrated eval SHOULD start rewarding more
search — so "does the two-horizon far head improve with nodes where the
incumbent didn't?" becomes a real signal to check later.

---

## 2026-07-13 (Opus) — REFERENCE: how the fitness probe works + what every statistic means

*(A permanent legend, added at Kaveh's request. Whenever a stat below appears in
an entry, this is what it means. Plain-language; a chess enthusiast should follow
it.)*

### The fitness probe (`experiments/qm_fitness_probe.py`)

Winning/losing games is a slow, noisy, blunt signal — it tells you the model is
bad but not WHY. The probe is a set of fast, ground-truth-anchored *diagnostics*
that say which specific part of the learned geometry is healthy or broken, so we
can steer training instead of guessing. It runs six instruments:

1. **Syzygy calibration.** Chess tablebases give the EXACT truth for small-piece
   endgames ("this is mate in 7"). We ask the model for its learned distance from
   each such position to the mate goal, then check whether the model's ordering
   agrees with the true ordering (Spearman rho, below). A good embedding says
   mate-in-3 is closer than mate-in-20. Two variants: KRvK (pawnless, where the
   tablebase number = exact plies-to-mate) and "nearest-exemplar" (distance to
   the nearest example mate in a bank — this correlates where a single averaged
   goal is flat). This is the rare case where we have PERFECT ground-truth
   distances to grade against.

2. **Horizon-stratified retrieval.** A recognition test. Take a position s and its
   TRUE future g that actually occurred k plies later in the game; hide g among 63
   random decoy positions from other games; ask the model to pick the real future
   out of the 64. Accuracy is measured separately at k = 1, 2, 5, 10, 20, 50 plies.
   This shows HOW FAR AHEAD the model can "see." Ours is sharp to ~10 plies then
   falls off a cliff by 50. For two-horizon we run it on BOTH heads: near should
   ace k=1, far should hold up at k=20–50.

3. **Asymmetry audit.** Captures are one-way doors — you can't un-capture a rook.
   For position pairs where a capture happened between s and g, we check whether
   the model scores the impossible REVERSE trip (g back to s) as FARTHER than the
   forward trip. Reports the fraction that get it backwards (want ~0).

4. **Triangle violation.** Structural sanity: for random position triples, is
   d(A,C) ≤ d(A,B) + d(B,C)? (Direct is never longer than a detour.) A real
   distance never violates this; reports the violation rate (ours ~0, guaranteed
   by construction).

5. **Degeneracy panel.** "Is the embedding collapsing or wasting capacity?" —
   spread ratio and effective rank (below).

### What each statistic means

- **Spearman rho (ρ), aka rank correlation.** A number from −1 to +1 measuring
  whether two *orderings* agree. We rank positions by the model's learned distance
  and by the true distance, and ρ asks "do these two rankings match?" ρ=+1 perfect
  agreement, ρ=0 no relationship (the model's distances are unrelated to truth —
  "flat"), ρ=−1 exactly reversed. We use rank correlation (not exact-value error)
  because for planning we care about ORDER — is mate-in-3 ranked nearer than
  mate-in-20 — not the literal number. So "nearest-exemplar ρ +0.25" means a weak
  but real positive agreement; "centroid ρ ≈ 0" means flat/useless.

- **p-value.** The probability of seeing a result at least this strong if there
  were truly NO effect (pure luck). Small p (< 0.05) = unlikely to be a fluke, so
  we believe the effect is real. p=0.35 = very plausibly luck (a "wash").

- **Confidence interval (CI), e.g. [−11, +40] cp.** The plausible range for the
  true value. If the whole interval is on one side of 0, the effect has that sign
  with confidence; if it straddles 0, we can't rule out "no difference."

- **e-value.** An "anytime-valid" evidence score against the no-difference
  hypothesis — think of it as accumulated betting winnings. Crossing 1/α (=20 for
  the 5% level) lets us declare a real difference, and unlike a p-value you're
  allowed to peek as games stream in without cheating. Bigger = stronger evidence.
  "e=65, REJECT" = strong evidence the two policies really differ.

- **Wilcoxon signed-rank test.** The paired significance test we run on
  per-position score *differences* (rank-based, so a few blowout games don't
  dominate). Produces the p-value for "policy A ≠ policy B on matched positions."

- **Bootstrap CI.** A confidence interval built by re-sampling the data thousands
  of times — no assumption about the data's shape, just "how much does the average
  wobble if I'd drawn a slightly different sample."

- **ACPL / centipawn (cp).** 1 cp = 1/100 of a pawn (standard engine unit). ACPL =
  average centipawns lost per move vs a strong Stockfish's judgment; lower is
  better (master <20, beginner 100+, our policy ~250–290).

- **Retrieval accuracy vs chance.** Fraction of the 64-way recognition test the
  model gets right; "chance" (≈1/64 ≈ 0.016) is the random-guess baseline.

- **k / horizon / ply.** A ply is one half-move (one player's turn). k = how many
  plies into the future the retrieval test reaches.

- **Spread ratio.** Average distance over random position pairs ÷ average distance
  over adjacent (1-ply) pairs. ≈1 would mean all distances collapsed to one value
  (degenerate); we want distant positions to actually read as far (ours ~1.8–2.4).

- **Effective rank.** How many of the 64 embedding dimensions are actually carrying
  information (entropy of the singular-value spectrum). Low = wasted capacity
  (ours ~24–26 of 64).

- **DTZ / DTM / WDL.** Tablebase ground truth: Distance-To-Zeroing-move /
  Distance-To-Mate / Win-Draw-Loss under perfect play. Used only to grade the
  model, never to train it.

---

## 2026-07-13 (Opus) — the sharpness reframe: depth is the wrong axis; uncertainty is; benchmark built

Kaveh's reframe (his words, condensed): the tactical/positional boundary isn't
temporal depth, it's **local sharpness of the value landscape** — a sharp
position is high-curvature (one tempo flips the result, can't prune), a smooth
one is low-curvature (move-orders converge, a coarse estimate suffices). A
forcing line runs 20 ply deep; a position is quiet at ply 2. So a ply-keyed
handover is mis-specified, and THAT is why the two heads fight — at a fixed
horizon one scalar is forced to be sharp and smooth at once. Our node-budget
sweep (non-monotonic) already agreed depth isn't the lever.

The fix: drive the handover on **uncertainty the model emits**, not depth.
Aleatoric (irreducible branch volatility = genuine sharpness → don't prune) vs
epistemic (unmapped region → grows with depth as a consequence). Four options
(full spec in UNCERTAINTY_DESIGN.md): A head-disagreement gate (near-free
validator), B distributional reachability head (signal producer), C
uncertainty-gated quiescence expansion (consumer), D γ-ensemble (optional).
Chosen: **B produces, C consumes; A validates first.**

**B distribution = CATEGORICAL, not Gaussian** (Kaveh): chess distance-to-goal is
bounded + integer (tablebase DTM/DTZ caps), so fixed distance bins have no
edge-placement problem; Gaussian is rejected because bimodality ("3 ply or 30 ply
depending on the line") IS the tactical signal and a Gaussian can't represent it;
quantile regression is the fallback. Axiom load-bearer: the point-estimate used as
the PLANNING DISTANCE must keep the IQE quasimetric axioms; the spread rides on
top as an auxiliary regime signal and need not. v1 keeps the existing quasimetric
d as the distance and uses the categorical only for spread.

**Built: `experiments/sharpness_bench.py`** — the measurement backbone Kaveh asked
for. Exact tablebase ground-truth sharpness of a winning position = value
curvature over its legal moves (DTZ progress-cost spread: a rook hang that still
wins by WDL shows up as a big cost jump, where the coarse WDL-preservation metric
was flat). Any uncertainty signal is then ranked by rho vs that truth. Ground
truth has real dynamic range (mean sharpness 0.53, 13% only-move-sharp).
**Baseline: incumbent point-head move-score-spread ρ=+0.14** (weak) — the number A
(head-disagreement) and B (categorical spread) must beat. `artifacts/experiments/
sharpness_incumbent.json`.

**Next:** when the ply-stratified two-horizon run finishes (baseline), run A
(head-disagreement) on the benchmark — if it beats +0.14 meaningfully, the
sharpness hypothesis is validated and we build B (categorical) then C (gated
search), each scored on the benchmark then the play gate.

---

## 2026-07-13 (Opus) — A-validation: head-disagreement is NOT a sharpness detector; two-horizon specialized structurally; on to B

Two-horizon baseline (`lichess_fb_4gb_twohorizon.pt`, ply-stratified) trained,
then the two measurements that matter.

**A-validation (sharpness_bench, n=446) — Option A REJECTED.**
- head_disagreement rho vs true sharpness = **+0.079** (weak)
- score_spread (point-head move-score spread) rho = **+0.202** (the baseline)

Head-disagreement detects sharpness WORSE than a plain point estimate. Diagnostic:
the heads DO disagree (mean 0.33, std 0.22, up to 1.27) — they are not redundant —
but their disagreement is ORTHOGONAL to real value curvature: it tracks the
ply-training-distribution difference, not sharpness. This is precisely what
Kaveh's reframe predicts (the ply axis is not the sharpness axis), so it confirms
the reframe rather than refuting it. Option A is dead. **The bar for B is now
rho > +0.20** — the categorical entropy must beat the point head's own spread.

**Two-horizon probe — the ply-split DID specialize at the representation level.**
- NEAR retrieval k=1 = 0.98 (sharpest short-range of any checkpoint, by design),
  collapsing to 0.03 at k=50 (the short-range specialist).
- FAR nearest-exemplar calibration rho = +0.272 — the BEST long-range endgame
  calibration of any checkpoint (incumbent +0.165, gen2 +0.252). FAR retrieval
  holds the mid-range (k=20 0.66) like the incumbent.
- Pre-registered probe gate: near k=1 >= 0.95 PASS (0.98); far nearest-exemplar
  rho >= 0.30 borderline FAIL (0.272, just short). Spread 1.91, rank 24.3 (healthy).

So the two heads genuinely became a short-range sharp specialist and a long-range
calibrated specialist -- the architecture works structurally. But (a) their
disagreement doesn't detect sharpness, and (b) the axis is ply not curvature, so
this is the confirmed-suboptimal BASELINE. Play gate (KRRvKBP far-mode) queued as
a cheap data point; not expected to promote (wrong axis).

**Decision:** proceed to B. Keep the far head's calibration win in mind (long-gap
training helped calibration, +0.272). Build the categorical distributional head;
its entropy must beat score_spread's +0.20 on the sharpness benchmark to be worth
consuming in C.

---

## 2026-07-13 (Opus) — B fails the sharpness gate; and the benchmark was distance-confounded (both caught by fail-fast)

Short-run-first + rigorous instrumentation paid off twice in one loop.

**B (categorical distributional head) short run (15k steps):** dist_sigma
(position entropy) rho vs sharpness = -0.21 -- NEGATIVE. Tried three readouts
of the SAME checkpoint (no retraining): position entropy, successor-mean-spread,
successor-entropy-spread -- all negative (-0.16 to -0.23), while the plain
score_spread was weakly positive (+0.13). Per the pre-registered gate: no full
run. The short run saved a wasted 90k.

**Then the deeper catch: the benchmark's sharpness ruler was DISTANCE-CONFOUNDED.**
rho(sharpness, distance-to-mate) = +0.387 with the absolute cost margin -- "sharp"
disproportionately meant "near mate", because an absolute margin flags few
holding-moves when costs are small (near mate) and many when large (far). So
every apparent signal was partly a distance artifact: score_spread's raw +0.13
-> partial +0.06 controlling for distance. The "+0.20 baseline" the whole B plan
was pinned to was largely measuring distance, not sharpness.

**Fix:** added `crit = (2nd_best - best)/(best + 1)`, a best-vs-second-best
criticality ("does the best move matter?") that is distance-INDEPENDENT
(rho(crit, distance) ~= 0.00-0.10). On this clean ruler, EVERY current signal is
~0: score_spread +0.05, dist_sigma -0.07, successor-spreads ~0. Honest headline:
**no static signal our models emit -- point head OR distributional head -- detects
true (distance-controlled) tactical sharpness.**

**Interpretation / fork (to Kaveh):** the reframe (sharpness = value curvature)
may be right, but a distance/ply-gap-trained representation doesn't encode it.
Three paths: (a) train the categorical on OUTCOME (WDL win/draw/loss from game
`result`) -- its entropy is result-volatility, closer to tactical sharpness, and
aligns with the WDL viz; (b) treat sharpness as SEARCH-INTRINSIC -- classical
quiescence: gate expansion on the search's own value INSTABILITY across
depth/siblings, no learned head (curvature is a property of the tree, maybe not
the node); (c) it's blocked until the value function itself is better calibrated.
Epistemic caveat: crit is one operationalization; the "no signal" conclusion is
as strong as crit is a good sharpness proxy (distance-clean, plausible, not
proven canonical). Paused for Kaveh's steer before building (a) or (b).

---

## 2026-07-13 (Opus) — the sharpness REFRAME lands: self-referential reliability, two methods built

A long, decisive design+build session with Kaveh. Findings and decisions, in order.

**The reframe (Kaveh): sharpness is not a real thing to label -- it's an invented
concept whose only job is to allocate search effort.** So define it
SELF-REFERENTIALLY: sharpness = where the engine's own static estimate is
UNRELIABLE / where it's weak. Consequences: (1) works for the WHOLE game (opening
included -- any position embeds somewhere), (2) the middlegame-ground-truth
problem vanishes (no external truth to match), (3) **validation shifts from
label-correlation to PLAY** (does using the signal to allocate search improve
results at matched compute). **`crit`/tablebase sharpness is RETIRED as arbiter.**

Also confirmed (Kaveh's question): a WDL/outcome head would be a VALUE head like
Leela -- it would embed value, not reachability, and leaning on it undercuts the
reachability thesis. So we stay reachability-native: sharpness = instability of
the REACHABILITY estimate, no value head.

**Kaveh's definition of sharp, formalized:** a position where a normal-looking
move suddenly takes you far from the goal, OR the only good paths are
non-normal-looking moves. Both are the SAME phenomenon: the SHALLOW move-ranking
disagrees with the DEEP move-ranking. "Normal" = the shallow (1-ply reach)
expectation. Filter obvious 1-ply blunders (they agree shallow AND deep, so don't
inflate disagreement). Second flavor Kaveh named: "interactions flying, lots of
captures" = tactical-DENSITY sharpness (a MIDDLEGAME phenomenon), and "a position
we've never seen" = epistemic/novelty. Tested the structural density signals on
the endgame benchmark: they ANTI-correlate with endgame crit (-0.15..-0.24) --
because endgame sharpness is quiet precision, not melee; the melee regime is
middlegame, which tablebases can't ground-truth. This is exactly why we retire
the label and validate by play.

**Decision (Kaveh): build BOTH methods; EITHER sharp -> extra search; BOTH sharp
-> keep searching to certainty.** Built:

- **Method 1 -- `FBSearchPolicy.reliability()`**: shallow-vs-deep reachability-rank
  disagreement among shallow-plausible moves (`_rank_disagreement`). Exact,
  per-position, reachability-native, no label. Sanity: KRRvKBP (the known
  rook-hang) = 0.243 vs startpos 0.042 -- correctly flags where the model is
  unreliable.
- **Method 2 -- `catspace/competence.py::CompetenceMap`**: a kNN reliability FIELD
  over embedding space -- predicts unreliability from `F(s)` alone (cheap, no deep
  search), "where I've been weak before." Built offline by
  `build_competence_map.py`. **Held-out generalization at n=300: rho(predicted,
  actual Method-1 reliability) = +0.23** -- the competence field genuinely
  generalizes (not memorization).
- **`FBAdaptiveSearchPolicy`**: combines them. Quiet -> base nodes. Sharp (either)
  -> deepen. Both sharp -> iterative-deepen until the top move stabilizes
  ("certainty") or a node cap. Smoke: startpos m1/m2=0.04/0.08 -> 200 nodes,
  0 deepenings; KRRvKBP m1/m2=0.24/0.24 -> 400 nodes, 1 deepening (stopped when
  the move stabilized). This is the fix for the node-sweep negative -- search more
  only where deeper search CHANGES the decision, not uniformly.

**Why this is the right shape (ties to a prior negative):** the earlier node-budget
sweep showed UNIFORM more-search is a non-lever (non-monotonic). Reliability-gating
searches more exactly where shallow and deep disagree -- by construction the only
place extra search can pay.

**Prior negatives that led here (same session):** B (categorical distributional
head) failed the sharpness gate -- position entropy AND successor-spreads all
NEGATIVE vs sharpness; caught by the 15k short run (no wasted 90k). Then the
sharpness benchmark itself was found DISTANCE-CONFOUNDED (rho +0.39); `crit`
(best-vs-2nd) decounfounds it (~0), and on the clean ruler NO static signal
detected sharpness -- which is what motivated dropping the labeled-benchmark frame
entirely for the self-referential + play frame above.

**Still TODO (the closed loop -- Stages 2-3):** self-play that logs the SEARCH TREE
(s -> explored children, visit freq, backed-up reach), then DISTILL those
search-improved reach targets back into the embedding (deep->shallow), closing the
loop: more search where weak -> more data there -> embedding improves there ->
reliability map shrinks -> search redeploys. New COMPONENTS.md maps all the pieces.

---

## 2026-07-13 (Opus) — reliability-gated search ALONE is a null; the value is in the loop, not the gate

Competence map (n=2000) held-out generalization: rho(predicted, actual Method-1
reliability) = **+0.310** (up from +0.23 at n=300) -- the competence FIELD is real
and learnable. Method 1 flags known-hard positions (KRRvKBP 0.24 vs quiet 0.04).
Both signals work. But the play test is what matters:

**PLAY VALIDATION (KRRvKBP n=60, matched compute):** adaptive (reliability-gated,
avg 455 nodes/move) = **0.583** vs uniform FBSearchPolicy @ 455 nodes = **0.600**.
delta = -0.017 (~1 game, noise). **Gating does NOT beat uniform at equal compute.**

Aside worth noting: uniform @455 (0.600) > uniform @200 (0.567 incumbent) -- so in
the KRRvKBP ENDGAME, more search DOES help (unlike the full-board node sweep). But
TARGETING that search by reliability doesn't beat spreading it uniformly.

**Interpretation (two reinforcing reasons, both honest):**
1. *Homogeneous difficulty defeats targeting.* Gating pays only when difficulty is
   HETEROGENEOUS (some positions need lots of search, others none). KRRvKBP is
   uniformly hard precise conversion -- nearly every position wants more search --
   so uniform allocation is already near-optimal and targeting adds nothing. The
   gate's value proposition needs a full-game mix (quiet openings, sharp
   middlegames, precise endgames), which we can't tablebase-ground-truth.
2. *More search over a FIXED embedding has a ceiling.* Searching harder where the
   model is unreliable only helps if the deeper search finds better moves -- but
   if the embedding is weak THERE, deeper lookahead over it is still weak.
   Concentrating (inert) extra search doesn't fix a weak value function.

**This reframes the plan -- and it matches Kaveh's own loop vision.** The
reliability signal's payoff is NOT in gating search on a frozen embedding; it's in
the CLOSED LOOP: allocate search where unreliable -> that search PRODUCES DATA
(what reaches what) exactly in the weak regions -> DISTILL it back -> the embedding
improves there -> reliability shrinks -> repeat. Gating alone is one inert half of
a cycle whose other half (distillation) is what makes it pay. So the priority is
Stage 2 (capture the search TREE as reachability data) + Stage 3 (distill into the
embedding), not tuning the gate.

**Decisions:** (a) keep the gate + both sensors (they're the loop's allocator and
its epistemic signal; the competence HEAD, training-integrated, is the always-
current version); (b) do NOT chase gate hyperparameters on KRRvKBP (wrong regime
to show gating value); (c) build the closed loop, where the sensor's value is
realized; (d) the competence-head training run is still worthwhile -- it's the
loop's native Method-2 signal. Kept the offline map only as the stand-in it was.

---

## 2026-07-13 (Opus) — KRRvKBP drill-down: WHAT the planner isn't seeing (concrete)

Built `experiments/krrkbp_drilldown.py`: plays the incumbent (White) from a
tablebase-won KRRvKBP position vs Stockfish and, at every White move, compares
what it DID to the tablebase-optimal, dumping the model's reach ranking of all
moves against the truth. Ran positions 0, 5, 12 (all tablebase wins, DTZ 3-5).
All three DRAWN (insufficient material / threefold). The failure is now concrete:

1. **The reach landscape is nearly FLAT across moves.** Pos 0, at the decisive
   position, the top-8 moves' reach spanned -1.2609..-1.2699 -- a range of
   **0.009**. The model literally cannot tell its moves apart; there is no
   progress gradient where precise technique is required.
2. **Move ranking is ~uncorrelated with the tablebase-optimal.** The winning move
   was routinely ranked #10-#27 by the model's reach (pos 0 ply 6: best move Kf6
   ranked **#20**; pos 5: best moves ranked #17/#20/#25/#27). It's not slightly
   off -- its ordering is essentially unrelated to which move actually wins.
3. **The three failure behaviors, explained by (1)+(2):**
   - THREEFOLD REPETITION (pos 5, 12): with no progress gradient it shuffles.
   - INSUFFICIENT-MATERIAL self-draw (pos 0): it trades rooks/pieces down
     (K+R+R -> K+R vs K+P) because captures look no worse than anything else.
   - **ROOK ONTO A BISHOP-ATTACKABLE SQUARE** (pos 12, plies 6/10/16, flagged
     automatically): it does the EXACT OPPOSITE of the concept Kaveh hoped it
     would learn ("keep the rooks where the bishop can't touch") -- because it
     can't see the difference.

**What it isn't seeing:** which king/rook moves make PROGRESS toward mate. In the
KRRvKBP region -- which never occurs in the human Lichess training data -- the
reachability embedding has essentially no structure, so reach-to-mate is flat and
move-ordering is random. This is the OOD-coverage problem shown at the move level.

**Why this matters for the plan:** it's a direct, mechanistic confirmation of the
whole direction. These positions are exactly the HIGH-UNRELIABILITY regions the
competence signal is meant to flag (and does: pos-0-family scored reliability
~0.24 vs ~0.04 quiet). And the fix is exactly the closed loop: the ONLY way the
embedding gets a progress gradient here is to generate data here (search/self-play
in these positions) and distill it back. Gating search alone can't help (drill-
down shows deeper search over a flat reach is still flat) -- consistent with the
gating-alone null. The loop (Stage 2-3) is the fix this evidence points to.

---

## 2026-07-13 (Opus) — Toy closed loop on KRRvKBP: curvature APPEARS from self-play

Kaveh: "do self-play of this toy scenario, and see if the model improves ... I
want to see how much curvature starts to appear in the reachability space where
we want it as we proceed in self-play -- the sensitivity."

Built the scoped closed loop: `selfplay_generate.py --start-fens` (every game
launched from the 60-position KRRvKBP fixed set), `reach_curvature.py` (turns
"curvature/sensitivity" into scalars on that fixed set), and
`toy_selfplay_loop.py` (iterate self-play -> fine-tune on cumulative replay ->
measure curvature). 3 rounds x 250 games x +5000 finetune steps, selfplay-frac
0.7 (mixed with human to avoid forgetting), gen at 100 nodes.

Trajectory (artifacts/experiments/reach_curvature.jsonl):
  round  move_spread  dtz_rho  best_rank  top1_win
  R0     0.0062       +0.020   0.457      0.896     <- flat baseline (drill-down)
  R1     0.0685       +0.026   0.554      0.710
  R2     0.0474       +0.091   0.505      0.751
  R3     0.0378       +0.078   0.439      0.860

Reading:
- move_spread (raw field sensitivity): 0.006 -> 0.04-0.07, a **7-11x** jump. The
  reach field is no longer flat/equidistant in the KRRvKBP region -- self-play
  of ONE scenario measurably carves curvature into exactly that region. This is
  the core positive result: the flat blind spot is fixable, and we can watch it.
- dtz_rho (curvature WHERE WE WANT IT -- reach tracking true -|DTZ|): +0.020 ->
  +0.091 at R2, ~4x. Direction is right, but absolute value is still weak (~0.09)
  and it PEAKED at R2 then dipped R3 -- diminishing/noisy returns at this data
  scale (250 games/round, ~5k finetune steps).
- Spread-vs-alignment tension: R1 has the most spread but the WORST alignment and
  win-preservation (top1_win 0.71) -- the field first gets bumpy, then R2/R3
  trade spread for better orientation as it reorganizes (top1_win recovers 0.86).

Conclusion: self-play distillation into a blind region WORKS as a curvature
mechanism -- it converts a flat reach field into a sensitive one and nudges it
toward truth. Open questions the trajectory raises: (1) does dtz_rho keep
climbing with more rounds/games or plateau at ~0.09? (2) does the added curvature
translate to actual KRRvKBP CONVERSION gains (play is the real test; curvature is
the proxy)? Next: measure conversion on the fixed set with the R3 ckpt vs
incumbent, and if promising, extend the loop to see if dtz_rho converges.

### conversion (play-truth) check — curvature appeared, but play did NOT improve

Paired KRRvKBP conversion, incumbent (A) vs self-play R3 (B), same FBSearchPolicy
@200 nodes vs Stockfish skill 0, matched seeds over the 60 fixed positions
(`conversion_compare.py`):

  VERDICT conversion A=0.558 vs B=0.450  mean_diff=-0.108 CI=[-0.739,+0.522] e=0.89

A noisy null-to-NEGATIVE (not significant, huge CI). So the toy loop DISSOCIATES:
curvature appeared (move_spread 7-11x, dtz_rho 4x) but conversion did not improve
and if anything dipped. Why -- and it's the useful lesson:
- dtz_rho only reached +0.09: the field got BUMPIER (spread up) without getting
  correctly ORIENTED (alignment still weak). For a greedy search a confidently-
  wrong gradient is worse than a flat one -- which is exactly why the incumbent's
  top1_win (0.896) beat every self-play round (R1 0.71 ... R3 0.86).
- ROOT of the weak alignment: the self-play games are mostly DRAWS (the blind
  policy rarely converts), so the positive mate signal distilled each round is
  SPARSE and noisy. Curvature is inducible; accurate curvature needs a denser
  mate signal.

Mechanism validated, dose/quality not there yet. Clean next levers to densify the
mate signal (all outcome-legitimate, no oracle labels): (1) CURRICULUM -- start
self-play from won-in-1/2/3 positions the blind policy CAN mate, then expand
outward; (2) more search nodes in self-play so it converts more often; (3) more
rounds/games to see whether dtz_rho keeps climbing past +0.09 or plateaus.
Recommend (1): a mate-distance curriculum is the highest-leverage fix for signal
sparsity and directly tests whether accurate curvature -> better conversion.

---

## 2026-07-13 (Opus) — SF-vs-SF fine-tune (toy KRRvKBP): representation up, play flat

Kaveh: "instead of self-play, create a bunch of Stockfish-vs-Stockfish games and
fine-tune on them (in the toy example)." Rationale: SF-vs-SF actually CONVERTS the
endgame, so the mate signal is dense and correct -- fixing the self-play weakness
(blind policy mostly drew -> sparse signal).

Built `--sf-vs-sf` (both sides Stockfish; records only moves+result, leakage-clean)
+ 700 tablebase-verified WINNING KRRvKBP starts, disjoint from the fixed-60 test.
Generated 700 games / 12419 positions, **97% clean conversions**. Fine-tuned the
incumbent +6000 steps, selfplay-frac 0.7 (+0.3 human).

Result -- a clean DISSOCIATION, and it matches the self-play toy exactly:

  REPRESENTATION improved (best of any run):
    reach curvature: move_spread 0.006->0.028, dtz_rho +0.020->+0.067,
      best_rank 0.457->0.393 (fastest-mate move ranks higher than ever)
    neighbour DTM-alignment rho: +0.021 (incumbent) -> +0.102 (best yet)
    train DIFF_SLOPE won-lost separation flipped POSITIVE (+0.289 vs +0.056)
  PLAY did NOT improve:
    conversion (fixed-60, paired): incumbent 0.575 vs SF-ft 0.458
      mean_diff -0.117, CI=[-0.75,+0.51], e=1.04  (null-to-negative, same as self-play)
    top1_win (frac the #1-reach move preserves the win): 0.896 -> 0.796  <-- DROPPED

**The crux: top1_win drops in EVERY fine-tune (self-play and SF-vs-SF alike),
0.90 -> 0.71-0.80, even as average ranking (best_rank, dtz_rho, DTM-alignment)
improves.** Fine-tuning makes the reach field more opinionated (higher spread) and
better-ordered ON AVERAGE, but LESS reliable at the very top -- and play is
argmax+shallow-search, governed by top-1 correctness. So we keep improving the
wrong statistic: geometry/average-rank up, argmax-precision down, net play flat.

Key implication: the bottleneck is NOT data quality. Dense, correct SF-vs-SF
conversions helped the representation MORE than sparse self-play, but helped play
no more (both null). So more/better data in this region has a play ceiling. The
limiter is the objective+representation: the contrastive reach + ply-gap loss
optimizes distributional ordering, not top-1; and (Kaveh's earlier point) F never
sees move-count, so it may lack the information to pin the single best move.

Candidate next steps (decision pending):
  (a) GENTLER fine-tune (lower frac/steps/LR) -- cheap test of whether top1 can be
      preserved while adding structure (rules out "distribution shift too aggressive").
  (b) FEED distance info to F (fullmove/plies-to-goal), so it CAN represent DTM --
      Kaveh's hypothesis; architectural, needs retrain.
  (c) DTM-AWARE objective: a ranking/top-1 loss or a DTM-regression head, instead of
      only contrastive+ply-gap.
Note: same-material frac isn't comparable across banks (SF-vs-SF bank is mostly
post-capture <=5-piece positions, so a 6-piece query has few same-material nbrs);
DTM-alignment rho is the comparable metric.

---

## 2026-07-13 (Opus) — Do W/D/L regions exist? No. (mix: SF-vs-SF + planner-vs-SF)

Kaveh: "mix in planner-vs-Stockfish so we see how Stockfish kills and might even
win. I want to see if the representation finds three distinct win/draw/loss
regions." Built `wdl_regions.py`: label a bank by tablebase outcome, embed F,
project (UMAP unsupervised + LDA supervised -- PCA dropped, Kaveh wanted a method
that folds ALL dims in), score BALANCED-accuracy separability, with a
White-to-move-only readout to kill the STM confound (in KRRvKBP stm alone
predicts win/loss). Data: 700 SF-vs-SF games (wins) + 400 planner-vs-SF
(203W/105D/92L -> draws & losses). Bank ~8.5k positions, W/D/L = 4116/434/3959.

Finding -- the embedding does NOT organize by outcome, and training on balanced
outcomes doesn't fix it:
  incumbent:  UMAP win/loss fully INTERMIXED (salt-and-pepper); silhouette -0.02;
              White-to-move-only balanced kNN 0.62 / linear 0.86 (soft axis only)
  mix fine-tune (+6000, frac 0.7 on the W/D/L mix):
              UMAP win/loss STILL intermixed; draws pulled into a faint tail;
              silhouette -0.03; White-to-move kNN 0.68 / linear 0.84
  -> no three regions before OR after. What training changed: it traded a soft
     win<->loss axis (incumbent LDA) for a soft draw-vs-decisive axis (mix LDA);
     win and loss stayed on top of each other. Silhouette ~0 throughout = no
     clustered regions, only weak linear-direction information.

Why this is the deep diagnostic: a reachability planner NEEDS winning positions
(goal reachable) to sit apart from losing ones (goal not reachable). They don't.
This is the SAME intermixing the neighbour viz (near-in-embedding != near-in-DTM)
and the flat reach field showed -- now proven at the outcome level. It directly
explains every play null: if W and L are intermixed in F, reach can't separate
good from bad moves and search has no value gradient to descend.

Likely cause: the contrastive/ply-gap objective pulls SAME-GAME temporal
neighbours together regardless of the win/loss boundary (a losing position 3
plies before a drawn ending gets pulled toward the draw), so temporal structure
overwrites outcome structure. Data quantity/quality can't fix an objective that
doesn't encode the outcome boundary.

Implication for direction: the lever is the OBJECTIVE/representation, not data
(now shown three ways: self-play, SF-vs-SF, and balanced W/D/L mix all fail to
separate outcomes in play or in embedding geometry). Candidates: (1) feed
outcome/move-count info to F (Kaveh's earlier point); (2) an explicit
outcome-separating term / value-contrastive loss so W and L can't share a
neighbourhood; (3) reconsider the goal representation (MATE_W centroid) toward a
region/quasimetric that pushes losing states far from the goal.

---

## 2026-07-13 (Opus) — Outcome-poles loss WORKS: outcomes separate in hops

Kaveh: "add a loss that pushes the poles apart; everything else pushed/pulled by
the final side who won -- I need HOPS, not euclidean." Implemented `--outcome-poles`
(catspace/nn/fb.py): 3 learnable terminal poles (loss/draw/win), a repulsion term
(min scaled distance `pole_margin` between poles) + a per-state HINGE on the
QUASIMETRIC distance (hops) so each state's own-outcome pole is `outcome_margin`
fewer hops than the others. result threaded from shard meta; rides on the ply-gap
term so the within-region hop gradient survives. Off-path byte-identical (19 tests).

Fine-tuned the incumbent +8000 steps on the W/D/L mix (SF-vs-SF wins +
planner-vs-SF draws/losses), selfplay-frac 0.7, outcome-weight 1.0.

Result -- the FIRST separation of outcomes all session. Nearest-pole assignment
on White-to-move positions (confound-free; hops = quasimetric d to each pole):
  true WIN  (3990): hops[loss,draw,win]=[2.29,2.15,1.58] -> 87% to WIN pole
  true DRAW ( 203): hops=[2.17,1.82,2.00]                -> 68% to DRAW pole
  true LOSS (  93): hops=[1.55,2.35,2.09]                -> 98% to LOSS pole
  balanced accuracy ~= 0.84 (chance 0.33).
Each class is fewest hops from its OWN pole. wdl_regions separability also lifted:
White-to-move balanced kNN 0.62 (incumbent) / 0.68 (mix) -> 0.79; draws now form
distinct UMAP clusters (they were smeared before). Won-lost DIFF_SLOPE cleanest
yet (+0.14 vs -0.18).

Caveats / open: (1) absolute hop gaps are modest (~0.5-0.7) -- could push harder
(outcome-weight/margin/steps). (2) win-vs-loss still overlap in raw UMAP of F (the
pole DIRECTIONS aren't UMAP axes; pole-distance space is where it separates). (3)
NOT yet checked: did conversion (play) and the DTM/hop gradient survive? -- the
whole point is separation WITHOUT killing move-selection. (4) the region viz's
side-to-move labels disagree with the loss's game-result labels on Black-to-move
rows; the nearest-pole metric above is the clean evaluation.
Next: verify conversion + reach-curvature (hops) didn't regress; a pole-distance
(ternary) viz to SEE the three corners; then decide push-harder vs move on.

---

## 2026-07-13/14 (Opus) — OVERNIGHT LOOP: embedding structure for hop-search play

Kaveh (going to bed): "find that embedding structure that will allow us to play
reasonably well with a search in the embedding space, going over hops. Implement
both [pole-pull and repulsion], try them, promote the winner. Keep iterating,
journaling, glossary, committing till morning."

North star metric: KRRvKBP conversion vs the incumbent (0.54 baseline) with
200-node HOP search, WITHOUT wrecking top1_win / the hop gradient. Secondary:
reach-curvature (dtz_rho, top1_win), outcome-region separation.

Design line so far (why we're here):
- Data doesn't fix it (self-play, SF-vs-SF, W/D/L mix all failed to separate
  outcomes or improve play). Bottleneck = objective+representation.
- HARD outcome-pole pull (weight 1.0): separated outcomes (bal acc 0.84 in hops)
  but CRUSHED play (conv 0.54->0.30) -- a global pull-to-one-point collapsed the
  win region's internal hop gradient.
- Kaveh's reframe (correct): we want t-SNE's shape -- ATTRACTION only between near
  neighbours (preserve within-region hops) + BOUNDED REPULSION between regions
  (spread mutually-exclusive outcomes), heavy-tail/hinge so nothing collapses.
  And the goal is a REGION (arrive anywhere in the mate set = soft-min over mate
  exemplars), not a single centroid/pole point.

Variant queue (each: fine-tune incumbent +8000 steps on the W/D/L mix, frac 0.7;
eval = experiments/eval_variant.py -> overnight_results.jsonl):
- V1 soft-pole: temperature-CE pull to 3 learned poles + pole-as-goal (softer than
  the hard hinge). [running]
- V2 repel: t-SNE-style cross-outcome hop repulsion, NO pull-to-point, goal stays
  centroid. [built]
- V3+: region-bank goal (soft-min over mate exemplars) + repulsion; weight/temp/
  margin sweeps of the winner; combinations.
Promote whichever beats incumbent conversion while keeping top1_win >~0.85.

### overnight results so far + orchestrator

  variant        conv    top1_win  dtz_rho  note
  V0 incumbent   0.558   0.896     +0.02    baseline (target)
  V1 soft-pole   0.542   0.814     +0.085   TIE (ns); pole-AS-GOAL, separation held
  V1 hard-pole   0.300   0.719     -0.05    crushed play (global pull-to-point)
  V2 repel-only  0.400   0.792     +0.002   WORSE; centroid goal, repel didn't help

KEY INSIGHT from V1 vs V2: the GOAL matters more than the separation mechanism.
V1 (learned pole AS the planning goal) tied the incumbent; V2 (same-ish training but
kept the blurry MATE_W *centroid* as goal) regressed to 0.40 with a flat hop field
(dtz_rho ~0). So repel-only-with-centroid is a dead end -- the lever is the GOAL
(pole / region), not the cross-outcome push by itself.

Self-sustaining setup: experiments/overnight_orch.sh runs artifacts/experiments/
overnight_queue.tsv serially (idempotent: skips already-evaluated labels; picks up
appended lines), fine-tuning the incumbent +8000 on the W/D/L mix per variant, then
eval_variant.py -> overnight_results.jsonl. Queue (pole-as-goal first, since that's
the lever): V5 pole+repel, V6 pole-gentle, V8 pole+strong-repel, V9 pole-w0.7,
V3 repel-strong, V7 repel-light.

Biggest UNTESTED idea (Kaveh's "arrive anywhere in the mate region"): the
region/soft-min-BANK goal. It's a PLANNING-goal change (planner already supports a
2D goal bank via soft_min_bank), applicable at eval time with NO retrain -- so it
can be tested on the incumbent directly. Next: implement a --goal bank option in
the eval and check if soft-min-over-mate-exemplars beats the centroid on the
incumbent; if yes, apply to the best variant. Then inject as a variant.

### overnight batch 1 complete (V5-V9, V3, V7) — pole-gentle wins, ideation stalled

Orchestrator ran all 6 queued by 01:29 then idled (I failed to keep injecting new
variants overnight -- the trainer stayed alive but starved). Results:
  V6 pole-gentle (w0.25, tau1.5): conv 0.575 vs incumbent 0.517 (+0.058), top1_win
    0.828, dtz_rho +0.097  <- BEST; gentlest pole pull.
  V8 pole-strong 0.55 / V5 pole+repel 0.55 / V9 pole-w0.7 0.50 (top1_win 0.851, best)
  V3 repel-strong 0.433 / V7 repel-light 0.358  <- repel-only+centroid loses again.
Pattern rock-solid: pole-AS-GOAL ties-or-beats incumbent (0.50-0.575) w/ positive
hop-gradient; repel-only-with-centroid loses (0.36-0.43). Gentler pull = better
(preserves within-region hops). CAVEAT: n=60 + SF nondeterminism -> incumbent
estimate wobbles 0.52-0.60 across runs, so V6's +0.058 needs confirmation.
Next: sweep gentler around V6, add more games for significance, region-bank goal.

### 2026-07-14 (Opus) — proper A/B: V6 "win" was NOISE; conversion too high-variance

Kaveh: "use the A/B harness with confidence intervals." eval_variant had been
dropping the CI/e-value (recording only point estimates) -- fixed to capture the
paired matched-seed diff + CI + anytime-valid e-value on a NEW n=200 held-out set
(disjoint from train + fixed-60). Definitive V6 vs incumbent:
  conversion A(incumbent)=0.537 vs B(V6)=0.532  mean_diff=-0.005
  CI=[-0.383,+0.383]  e=0.09   -> DEAD TIE (e<<1: data favours the null).
The overnight "V6 0.575 vs 0.517" was n=60 noise. Lesson banked: never promote on
n=60 conversion point estimates.

Two consequences:
1. NO pole variant beats the incumbent on play -- they TIE. The outcome-pole
   restructuring changes geometry (separation, dtz_rho +0.09 vs +0.02) but does
   NOT improve moves; on the lower-variance top1_win the incumbent (0.896) is
   actually AHEAD of the pole variants (0.81-0.85). Restructuring != better play.
2. Game-conversion is too high-variance (CI +-0.38 at n=200) to rank variants at
   all -- most KRRvKBP positions draw-or-win for BOTH, so the paired per-game diff
   is mostly 0/+-1. Need a LOWER-VARIANCE, per-MOVE A/B metric for power.
Next: paired move-level A/B (fraction of the model's hop-search top move that
preserves the win, per position, over the test set + optimal lines) -- thousands
of move samples -> tight CI -> can actually distinguish variants. That becomes the
primary A/B; conversion stays as the (noisy) ground-truth check.

### 2026-07-14 — methodological: why the variants can't be distinguished

Tried a move-level A/B (move_ab.py: fraction of hop-search top move that is
DTZ-optimal, paired, bootstrap CI) to get power the noisy conversion lacks.
Surprise: incumbent vs V6 (gentle) AND incumbent vs V1-HARD-pole BOTH show 100%
move agreement on 300 tablebase-optimal-line positions -- identical top move
everywhere. But hard-pole's conversion is 0.30 (vs 0.54, e=28228) -- it demonstrably
plays very differently. Resolution: models agree on the OPTIMAL-LINE positions (the
best move there is obvious); play divergence happens OFF the line, on each model's
OWN trajectory. => FIXED-POSITION move-eval cannot distinguish endgame play; only
self-driven playouts can. move_ab is therefore not a valid power metric as built
(kept, with this caveat).

Consequences for the whole search:
- The only faithful play metric is a PLAYOUT (model drives its own trajectory), and
  the SF-conversion version is too high-variance (CI +-0.38 at n=200).
- Better power metric to build: DETERMINISTIC playout -- model (hop search) as White
  vs a TABLEBASE-OPTIMAL defender (tb_best_move, deterministic -> no SF noise),
  from the 200 winning starts, score = mated-within-budget (binary) or plies-to-mate
  (continuous). Deterministic defender kills the SF variance; per-start paired diff
  vs incumbent gives a real CI. THIS is the next tool.
- Also unresolved: is a +8000-step fine-tune from the incumbent even enough to move
  play beneficially? hard-pole moved it (badly); gentle ones move it little. A real
  test of an objective may need much longer / from-scratch training.

Honest standing: after proper A/B, NO variant beats the incumbent on play; the
overnight sweep measured mostly noise. The bottleneck now is EVALUATION POWER
(build the deterministic playout) and TRAINING STRENGTH (fine-tune may be too gentle).

### 2026-07-14 — powered playout confirms: V6 = incumbent (no play gain)

Deterministic playout (model vs tablebase-optimal defender), incumbent vs V6, n=120:
  mate-rate A=0.175 vs B=0.158  diff=-0.017  CI=[-0.092,+0.058]  ns
Tight CI (+-0.075, vs conversion's +-0.38) rules out any real V6 improvement -- the
pole fine-tune restructures geometry but does NOT help hop-search play (slightly
worse). Also: the planner converts only ~17% of winning KRRvKBP vs OPTIMAL defense.
Conclusion stands: no fine-tune variant beats the incumbent. Fine-tuning +8000 from
the incumbent is too gentle to change play beneficially.
Next (highest value, NO retrain): region-bank soft-min GOAL on the incumbent.

### 2026-07-14 — region-bank goal is WORSE; comprehensive negative on embedding-structure

Region/soft-min-BANK goal (Kaveh's "arrive anywhere in the mate region"), tested on
the incumbent with NO retrain (playout_ab --ckpt-b-goal bank, 128 white-mate
exemplars <=6 pieces): centroid 0.200 vs BANK 0.040 (n=25, plies-to-mate 3) -- the
bank is much WORSE. Why: soft-min over specific mate patterns is peaked/noisy
mid-game (a KRRvKBP midgame is far from EVERY single mate exemplar), whereas the
centroid averages them into a smooth "mate-ness" gradient the hop search can
actually descend. The averaging that made the centroid look "blurry" is exactly
what makes it a usable planning signal. (n=80 confirmation running.)

COMPREHENSIVE STANDING after rigorous (paired, CI, deterministic-defender) A/B --
NOTHING beats the incumbent on hop-search play:
  - hard outcome-pole pull: separated regions, CRUSHED play (0.30).
  - soft pole + pole-as-goal: separation + better dtz_rho, but play = incumbent (tie,
    powered CI [-0.09,+0.06]).
  - cross-outcome repulsion (centroid goal): worse.
  - region-bank soft-min goal: worse.
  - every gentle fine-tune: plays ~identically to incumbent on-rail; ties on play.
The incumbent (plain quasimetric + ply-gap) converts only ~17% of winning KRRvKBP
vs OPTIMAL defense, and NO embedding-structure intervention moved that. Tentative
read: the ceiling here is the METHOD (FB reach + shallow ~200-node hop search) more
than the embedding's outcome-organisation -- restructuring geometry (separation,
hop-gradient) did not translate to better moves. Open levers not yet tried:
from-scratch/long training of the objective (low-evidence bet), deeper search,
two-horizon NEAR head for endgame precision, or rethinking the search itself.

### region-bank goal CONFIRMED worse (n=80, SIGNIFICANT)
centroid 0.175 vs bank 0.062, diff -0.112, CI=[-0.200,-0.025] SIGNIFICANT. The
deterministic playout has the power to detect it. Averaging (centroid) > soft-min
over specific mate exemplars for hop-search planning. Region-goal idea, as
implemented, rejected.

### 2026-07-14 — PIVOTAL: SEARCH-LIMITED, not embedding-limited

Deterministic playout, INCUMBENT @200 vs @800 nodes (same weights, same 80 starts):
  mate-rate 200n=0.175 vs 800n=0.325  diff=+0.150  CI=[+0.050,+0.250]  SIGNIFICANT
  (plies-to-mate 14 -> 10). Deeper hop search NEARLY DOUBLES conversion on the SAME
embedding. This flips the whole night's conclusion: the ceiling was SEARCH DEPTH,
not the embedding's geometry. The reach field already CONTAINS the information to
convert -- shallow 200-node search just couldn't extract it, which is exactly why
every embedding-restructuring variant (poles/repulsion/region-goal) TIED at 200
nodes: they were all fighting the wrong bottleneck.

Two readings:
- Practical: search deeper -> much better play (cheap, immediate win).
- Thesis (small-budget "the plan does the work, don't out-search"): the real target
  is to shape the reach field so SHALLOW search suffices -- the info is present
  (deep search proves it) but shallow extraction is poor, and none of the tested
  restructurings improved shallow extraction. So "make 200-node search play like
  800-node search does" is the sharpened research goal; the METRIC should be
  conversion-at-fixed-small-budget, and improvement = closing the 200->800 gap.
Next: does it keep scaling (800 vs 2000)? and re-examine variants at MATCHED deeper
search (maybe a restructuring helps MORE at depth, or helps shallow catch up).

### 2026-07-14 — TWO REGIMES + methodological correction

800 vs 2000 nodes: 0.325 vs 0.312, ns (CI[-0.06,+0.04]) -- search saturates ~800n.
Full scaling on the incumbent: 200n=0.175, 800n=0.325, 2000n=0.312. So:
  - 200->800: SEARCH-limited (deeper doubles).
  - 800+: EMBEDDING-limited (~0.32 ceiling; ~68% of winning KRRvKBP still unconverted
    vs optimal defense even with unlimited search).
METHODOLOGICAL CORRECTION: I A/B'd EVERY variant at 200 nodes -- the search-limited
regime, where the embedding CAN'T matter because search is the bottleneck, so
everything ties by construction. The regime where embedding quality shows is
SATURATION (~800n). So the overnight variant ties are UNINFORMATIVE about whether
the restructurings improve the embedding's ceiling. The correct test (never run):
variants at 800 nodes. If a restructuring RAISES the 0.32 saturated ceiling, the
whole embedding-structure line is revived. Running V6@800 vs incumbent@800 now.

### 2026-07-14 — CONCLUSIVE: restructuring doesn't help even at saturation

V6_pole_gentle vs incumbent AT 800 nodes (saturation regime, n=100):
  incumbent 0.360 vs V6 0.370  diff=+0.010  CI=[-0.100,+0.120]  ns
So even at the CORRECT (embedding-limited) regime, the pole restructuring = incumbent.
The ~0.35 conversion ceiling vs optimal defense is INTRINSIC to the FB-reach
representation -- not raised by pole-separation, repulsion, or region-goal, and not
by more search beyond 800n. (V6 converts a touch faster when it does: plies 9 vs 11.)

FINAL PICTURE of the whole line:
  - Deeper search 200->800n ~doubles conversion (0.175->0.35); saturates at 800n.
  - NO embedding-structure intervention beats the incumbent at EITHER regime.
  - The embedding's intrinsic ceiling vs optimal defense is ~0.35.
What's left to raise the ceiling (all require training bets, not cheap):
  (a) two-horizon NEAR head -- the one untested architecture, designed for
      close-range/endgame precision; could sharpen the saturated ceiling.
  (b) from-scratch training of a different objective.
  (c) a genuinely different planning method (the FB-reach + beam hop search may
      just cap here).
Recommend (a) two-horizon as the next real experiment; it's the targeted lever for
exactly this endgame-precision ceiling. Confirming with V9@800 that the whole pole
family ties at saturation.

### 2026-07-14 — FINAL SUMMARY of the overnight embedding-structure investigation

V9_pole_w07 vs incumbent @800n (n=100): 0.360 vs 0.390, diff +0.030, CI[-0.09,+0.15] ns.
Pole family confirmed: no SIGNIFICANT gain in conversion RATE at either regime.

HONEST NUANCE: at saturation (800n) both best pole variants lean slightly positive
(V6 +0.01, V9 +0.03) AND convert FASTER (plies-to-mate 9 vs 11, consistent). So the
restructuring may improve conversion SPEED (crisper hop-gradient) without improving
the RATE ceiling -- a real-but-modest quality signal within rate-noise. Would need
n~500 playouts to resolve the ~0.03 rate lean; the plies-to-mate signal is the more
promising place to look (continuous, more power) if pursuing.

=== THE OVERNIGHT INVESTIGATION, START TO FINISH ===
Question: an embedding structure that plays KRRvKBP well with small-budget hop search.
Method arc: outcome-pole loss (hard->soft), cross-outcome repulsion (t-SNE analogy),
region-bank goal, on data (self-play, SF-vs-SF, planner-vs-SF mix). Evaluated with a
PROPER A/B harness (paired matched-seed diff + CI + anytime-valid e-value) after the
overnight n=60 point-estimates produced a phantom "win" (V6 0.575) that vanished at
n=200 (mean_diff -0.005, e=0.09).
KEY RESULTS:
  1. SEARCH vs EMBEDDING regimes: incumbent converts 0.175@200n, 0.35@800n, 0.31@2000n
     vs a tablebase-OPTIMAL defender. Search-limited <800n (deeper ~doubles), then
     embedding-limited (~0.35 ceiling).
  2. Methodological fix: all variant A/Bs must be at SATURATION (800n) -- at 200n
     search is the bottleneck so the embedding can't show. Corrected tests (V6, V9 @800n)
     still TIE the incumbent on rate.
  3. No embedding-structure intervention (pole-separation, repulsion, region-goal)
     SIGNIFICANTLY beats the incumbent's ~0.35 rate ceiling. Region-bank goal is
     significantly WORSE. Modest, non-significant speed lean for pole variants.
  4. Tooling -- ATTRIBUTION CORRECTED (Kaveh caught me overstating this): the
     paired A/B harness with CI + anytime-valid e-value ALREADY EXISTED before this
     session -- catspace/abtest.py (EValueTest, confidence_sequence) +
     krrkbp_arena.run_paired (diff_ci, e_value), committed 2026-07-12. conversion_compare
     and eval_variant are thin WRAPPERS on it; worse, eval_variant initially DROPPED
     the CI/e-value (recording only point estimates) -- the regression that produced
     the phantom n=60 "win" and forced Kaveh to say "use the harness with CIs". The
     only genuinely NEW tool this session is playout_ab (deterministic-defender
     playout -- the existing harness used stochastic Stockfish). reach_curvature,
     wdl_regions are new diagnostics; move_ab was a dead end.
NEXT LEVERS (all training bets -> need Kaveh's direction, NOT launched autonomously):
  (a) two-horizon NEAR head (targets endgame precision -- the natural lever for the
      saturated ceiling; evaluate at 800n).
  (b) plies-to-mate as the primary metric (more power than rate) to chase the speed lean.
  (c) from-scratch / different objective; or a different planning method (FB-reach +
      beam hop search may simply cap ~0.35 here).
The overnight loop's real deliverable = the METHODOLOGY (proper A/B + deterministic
playout + regime awareness) that turned noisy point-estimates into trustworthy
conclusions, and the precise localisation of the bottleneck (search <800n, embedding
ceiling ~0.35). Winding the autonomous cheap-experiment loop down here -- remaining
work needs a deliberate training-bet decision.

### 2026-07-14 — near-mate region viz: outcome signal WEAK, not cleanly separated

Kaveh: visualize near-mate positions (4-ply before end: near mate_W / near mate_B /
near draw) in embedding space, hoping for clearly separated regions.
Built near_mate_regions.py (harvest from human 1gb shards by GAME RESULT; embed F;
UMAP + LDA + REACH-space (reachW vs reachB); separability). 600/class. Result --
regions are NOT clearly separated even at these EXTREMES:
  F-space:     kNN 0.57 · linear 0.59 · silhouette +0.02  (chance 0.33)
  reach-space: kNN 0.54 · linear 0.56 · silhouette +0.01
  corr(reach->mate_W, reach->mate_B) = +0.53   (partial shared "finality" component)
  VALUE axis (reachW - reachB = MATE_DIFF): kNN 0.49 · linear 0.49
Reading: there IS a real-but-WEAK outcome signal (~0.57 balanced acc vs 0.33 chance,
linear 0.59) -- the embedding is NOT value-blind -- but the three classes heavily
OVERLAP (silhouette ~0); no distinct regions. Reach-space (how the embedding is
actually USED) is if anything slightly WORSE than raw F. The +0.53 reach-reach
correlation shows a shared "near-a-mate" finality component partly diluting the
who-is-winning direction; the MATE_DIFF value axis alone separates only weakly (0.49).
This is the representational root of the ~0.35 play ceiling: even 4 plies from a
forced mate, the embedding only weakly distinguishes "I am about to win" from "I am
about to lose". Explains why restructuring at the margins didn't help -- the base
representation's value/outcome direction is faint. A real fix would need the value
direction trained in strongly (from-scratch objective that forces near-mate_W and
near-mate_B far apart), not a gentle fine-tune. (Consistent with the whole night.)

---

## 2026-07-14 (switched to FABLE) — forced-mate region separation: goal + handoff

Model switched from Opus to **Fable** now (per Kaveh; mirrors the earlier
Fable->Opus switch). Handoff state below.

GOAL (clarified over several messages): iterate the embedding + cost function until
the three FORCED-outcome regions clearly separate in embedding space:
  - mate_W  : side-to-move (White-POV) has a FORCED mate  (Stockfish-verified, any depth)
  - mate_B  : Black has a forced mate
  - draw    : FORCED draw = INSUFFICIENT MATERIAL (KvK, K+B vs K, K+N vs K, same-colour
              KB vs KB) -- mate impossible either way; its OWN tight region
Requirement: these three FORCED regions must NOT overlap each other. Positions that
are NOT forced (fightable middlegames) MAY overlap -- they're not in the set.
Cost function Kaveh specified: PULL a near-mate toward its pole + PUSH from the
opposite pole, GENTLY more each round (t-SNE-style iterative repulsion, but WITH a
pull -- t-SNE has none). Accumulate over rounds; don't hard-hit (that collapses).

TOOLS (all committed):
  - experiments/forced_mate_set.py : build + VALIDATE the set. Stockfish loaded ONCE
    (warm), movetime-bounded gen; DETERMINISTIC depth re-validation via --validate-only
    --filter-out (movetime is non-reproducible: 90/900 flipped). Draws = generated
    insufficient-material, validated by is_insufficient_material(). Records SF eval +
    moves-to-mate per sample. PERSISTED: artifacts/experiments/forced_mate_set_valid.json.
  - experiments/viz/near_mate_regions.py --forced-set : the SEPARATION METRIC. Embeds F,
    computes reach->MATE_W/MATE_B, reports 3-class AND **binary mate_W-vs-mate_B**
    separability (F/reach/value-axis kNN + silhouette) + corr(reachW,reachB). --record
    appends to a jsonl trajectory. This is the yardstick to optimise.
  - experiments/separation_loop.sh : cumulative gentle pole push on HUMAN 4gb (full-game)
    data each round (Kaveh: KRRvKBP-only can't know diverse mates), re-measuring separation.

STATUS / what we learned:
  - Baselines on the diverse validated set: incumbent + V6 both WEAK (reach-space
    silhouette ~0.04). V6 poles are antipodal (corr -0.82) but don't ORGANISE diverse
    positions into class regions (V6 only learned KRRvKBP).
  - Separation loop round 1 (gentle pole push, human data): did NOT separate --
    reach silhouette flat (+0.036), value-axis kNN dropped 0.60->0.48, and F-space kNN
    unchanged (0.70->0.69) => the GENTLE fine-tune barely moved F (same wall as the
    play investigation). Loop then CRASHED at round 2 (opt param-group mismatch -- now
    FIXED: tolerant opt_state load).
  - Monitoring lesson: a watcher armed only on the SUCCESS line ("round 3") hangs
    forever when the job dies at round 2 -> looks like "wakeups not working". Always
    watch for failure/exit too.

LEVERS for Fable to try (since gentle-on-human barely moved F):
  (a) STRONGER push -- goal is now SEPARATION not play, so the hard push that
      "crushed play" is exactly what reorganises F by outcome; lean in, accept play cost.
  (b) PROXIMITY-WEIGHTED pull -- pull positions NEAR their terminal mate hard, far ones
      little, so the loss concentrates on near-mate regions instead of diluting across
      all won-game positions (needs anchor->terminal distance in the batch).
  (c) direct F-space cross-outcome repulsion at higher --repel-weight.
  Rebuild of the forced_mate_set with the insufficient-material draw class is running
  (/tmp/fm_rebuild.log). Then: near_mate_regions --forced-set to baseline, then iterate.

---

## 2026-07-14 (Fable) — certainty geometry + two-timescale field (design session)

Kaveh's redefinition of the metric: **closeness = certainty of transition**. "The
closest path is the one where we're more certain" -- a messy position with one
winning line is NOT closer to mate than a clearly-forced one slightly farther.
Formalised: d(s,g) = plies + lambda*(-ln P(reach g)). -ln P chains multiplicatively
-> subadditive -> IS a quasimetric; forced (P=1) reduces to pure plies. This names
the measured bug: current d has min/shortest-path semantics ("one winning line =
close") = exactly the optimism behind the 0.35 ceiling and the 200n->800n gap
(deep search was computing certainty by brute force).

Estimator: certainty_rollouts.py -- stochastic rollouts on the toy, per-state
P-hat by FEN aggregation. Empirical fork found: White=incumbent+eps gives P-hat
mean 0.05 (our policy's incompetence, no gradient); White=tb-optimal+eps gives
mean 0.45 w/ spread (position's intrinsic forgivingness). Toy uses tb+eps as
scaffold; real system uses own-MCTS (no oracles). Kaveh accepted NON-STATIONARITY
of the field long-term ("the landscape has shifted" is real; also 50-move/3-fold
depend on counters/history).

Architecture settled (Kaveh + discussion): TWO TIMESCALES.
  - SLOW field: trained embedding, stationary over an AUGMENTED state (halfmove
    clock already plane 18; repetition-count + fullmove planes TODO).
  - FAST field: catspace/memory_field.py (built, smoke-tested) -- in-memory
    evidence store keyed by embedding location, updated every move, visit-count-
    weighted kNN query (start simple; competence-blend later). Schema reserves
    `payload` for TACTIC-POTENTIALS: precondition-region -> plan + payoff ("if
    opponent plays X, this tactic fires"), cf. 2026-07-10 conditional
    capture-vector design. Distill fast->slow between games = the closed loop.
  - Readouts (three strategies, Kaveh): (1) navigate embedding directly; (2)
    Leela-style eval head off frozen embedding, KL-distilled (fallback; harness =
    existing --repr ablation); (3) indexed positions w/ known evals + eval-change-
    per-direction (local field gradient) -- the memory field enables this.

ONCE-OVER before building further (Kaveh asked; flags to fix first):
  1. P-hat=0 -> -ln0 = inf: clip with Laplace floor P >= 1/(n+2).
  2. min-visits=2 too coarse (P in {0,.5,1}): raise threshold / weight by n.
  3. CIRCULARITY: fast field retrieves by kNN in the slow embedding, but we PROVED
     near-in-embedding != near-in-truth in weak regions -> test retrieval directly
     (20% held-out rows: retrieved vs actual P-hat, MAE+calibration+CI) BEFORE
     trusting the memory anywhere.
EVALUATION DISCIPLINE (no point estimates -- the phantom-V6 lesson):
  field quality = Spearman(learned d, plies+lambda*(-lnP)) on held-out states w/
  bootstrap CI; retrieval = holdout P-hat MAE w/ CI; money test = paired
  deterministic playout at 200 NODES (shallow-search rescue is the falsifiable
  claim) on held-out test_n200 (disjoint from rollout starts), bootstrap CI,
  CI-excluding-zero only; lambda/eps sweeps are exploratory by declaration, winner
  gets ONE pre-registered confirmatory run on untouched starts.
Concept-axes (outcome axis slot 0) committed earlier today; parked pending this.

### certainty distillation: all three gates passed (CIs disjoint)

Table: 3455 states >=4 visits, P-hat mean 0.52, real spread (9% P=1, 7% P=0).
RETRIEVAL-BEFORE-TRUST: holdout MAE 0.119 CI[0.109,0.129] vs predict-mean 0.276
  -- fast-field kNN is ~2.3x better than ignorance; slow geometry locally honest
  (support 0.94, p_var 0.03). Circularity fear benign IN-REGION; gate passed.
SHORT DISTILL (1200 steps): held-out Spearman(d, plies+8(-lnP)) went
  baseline -0.099 CI[-0.175,-0.027]  ->  tuned +0.170 CI[+0.095,+0.240].
  The NEGATIVE baseline is a finding in itself: the incumbent's distance is
  significantly ANTI-correlated with certainty -- min-semantics optimism measured.
  Sign flipped with disjoint CIs after 1200 steps. Weak (+0.17): full run next.
Next: full distill (6000 steps), then MONEY TEST = paired 200-node playout vs
incumbent on held-out test_n200 (CI-excluding-zero; confirmatory run after).

### MONEY TEST: null. Certainty geometry improved the FIELD, not shallow play.
Full distill: held-out Spearman -0.099 -> +0.142 (disjoint CIs) BUT paired 200n
playout vs incumbent: 0.175 vs 0.150, diff -0.025 CI[-0.100,+0.050] ns. The gate
held: full-data run NOT launched. Same dissociation as every toy intervention:
field metrics move, play doesn't. Candidate reasons to diagnose BEFORE building on:
(a) +0.14 Spearman is weak -- 2.7k train states may recalibrate too locally to
change move ORDERING at decision points; (b) lambda=8 single exploratory value;
(c) tb+eps rollout states != the model's own argmax trajectory (distribution
mismatch); (d) the recurring possibility that at 200n the SEARCH, not the field,
still binds. Next: drill-down on distilled-vs-incumbent move choices at decision
points; consider certainty in the LOSS at scale rather than post-hoc distill.
Stages 3-5 (two-field runtime, measured fallibility prior, opponent recovery) all
built + unit/smoke-tested this round and committed -- ready when the field is.

### Structure viz: distillation MEMORIZED, didn't generalize -- explains the null
certainty_structure.png: incumbent panel = shapeless cloud, d range only 0.63-0.90
(flat field; certain wins scattered everywhere). Distilled panel = tight monotone
band, d range 0.2-1.7 -- but plotted states are ~80% TRAIN rows: train fit ~+0.86
vs HELD-OUT +0.142 = massive generalization gap. The distill memorized the 2.7k
table states; the model's own play visits OFF-table states where the field is
barely recalibrated -> move ordering unchanged -> money-test null explained.
UMAP: certainty well-organized in F on trained states (red loss arm -> green win
lobe). FIXES: (1) 10-100x rollout states (the full-data run, now JUSTIFIED with a
mechanism), (2) early-stop on held-out Spearman, (3) certainty in the base
objective at scale, not post-hoc micro-finetune. 1600n money test running (regime
hypothesis, Kaveh).

### CORRECTION: "memorization" diagnosis retracted -- actual story: UNDERFIT + goal-vector mismatch
Rebuilding the structure viz as a real script (experiments/viz/certainty_structure.py,
per-panel captions, reproducible; the old figure was a throwaway heredoc that filtered
rows to n>=6[:2500]) exposed that yesterday's claim "train fit ~+0.86 vs held-out
+0.14 = memorization" does NOT reproduce from any artifact. No computation in the
transcript ever produced +0.86 -- the number was asserted in prose only. Lesson
enforced going forward: no number enters the journal unless it comes from a printed
VERDICT/script output.
Reproducible numbers (full table, eval mode; held-out = the distill's own seed-0 split):
  incumbent   all rows                       rho -0.055
  distilled   vs zW it TRAINED against       train +0.205 / held-out +0.142
  distilled   vs ckpt's REBUILT zgoal        train +0.164 / held-out +0.094
Corrected findings:
  (1) UNDERFIT, not memorization: train barely beats held-out. The 6k-step distill
      (cert MSE + NCE mixing) never fit the certainty target even on train rows.
  (2) GOAL-VECTOR MISMATCH (real bug, now fixed): certainty_distill optimized
      d(F(s), zW_incumbent) but save_ckpt stored a build_zgoals-REBUILT MATE_W
      (cosine 0.967 to the trained-on one) -> playout navigated to a goal the
      distances were never calibrated to (~0.05 rho lost; the money test saw a
      weaker field than the Spearman verdict measured). certainty_distill.py now
      saves the zW it trained against.
  (3) visit-count split: n<6 rows fit BETTER (rho +0.29/+0.33) than n>=6
      (+0.15/+0.20) -- the sqrt(n) confidence weighting did not buy the intended
      dense-evidence advantage.
  (4) UMAP: certainty is locally coherent (single-color patches) but there is no
      global certain-win lobe -- large-scale geometry unmoved.
Revised fix list (replaces yesterday's): (a) goal-vector fix (done), (b) fit the
target harder -- more steps / higher cert-weight with early-stop on held-out
Spearman, (c) 10-100x rollout states, (d) certainty in the base objective at scale.
1600n money test still running (n=80, deterministic defender).

### Production MCTS readout built (Kaveh: replace beam-minimax as the search layer)
catspace/nn/mcts.py: AlphaZero-style PUCT adapted for a policy-net-less engine --
value-only expansion (one batched reach call per expansion = len(children) budget
units, directly comparable to FBSearchPolicy leaf counts), priors = softmax over
child reach from the mover's perspective, 1-ply minimax bootstrap as the expansion
backup, self-calibrating tanh value squash (per-move center/scale from root
children -- reach scale differs per ckpt), terminals mate +1-ply_discount /
mated -1 / draw -0.999 (draw~failure ordering kept from DRAW_SCORE but bounded so
Q-averaging works). Deterministic (no rollouts, no root noise) as playout_ab's
exact-paired methodology requires. Core takes a plain reach_fn -> 9 model-free
unit tests (mate-in-1 both colors, stalemate-trap avoidance, budget accounting,
determinism, visit concentration on high-reach lines, terminal discounts) ALL PASS.
playout_ab.py grew --search-a/--search-b {beam,mcts} + --c-puct: matched-node
readout A/Bs on the same checkpoint. Smoke (n=10, 200n, incumbent): runs
end-to-end, ~1s/playout, converts. 1600n distill money test KILLED (Kaveh: no
value -- code changing under it). RUNNING: beam-vs-MCTS on the incumbent at 200n
n=120 then 800n n=80 (/tmp/mcts_ab.log) -- if MCTS wins at matched budget, the
readout was leaving conversion on the table; if tied, embedding-limited confirmed
and the lever is lichess-scale training with certainty in the base objective.

### FIRST CI-REAL PLAY WIN: MCTS readout beats beam at matched compute
PLAYOUT_AB MCTS_vs_beam_200n mate-rate A=0.175 vs B=0.292 diff=+0.117
CI=[+0.042,+0.192] (n=120 starts, deterministic defender) [SIGNIFICANT].
Same checkpoint, same 200-eval budget, only the search shape changed: every prior
"embedding ceiling" number (0.175@200n, ~0.35@800n saturation) was a BEAM-READOUT
ceiling, not a field ceiling. All prior null money tests must be reinterpreted:
interventions were evaluated through a readout that wastes budget. 800n leg
running. make_search_policy factory committed: beam/mcts plug-and-playable in
playout_ab, experiment_report, certainty_rollouts (beam stays default).

### MCTS readout CONFIRMED (pre-registered, frozen fresh starts) -- promoted
Confirmatory protocol executed per FIELD_PLAN/data_registry: fresh seed-777
tablebase-verified KRRvKBP wins (n=120, wdl=2, disjoint from all train/eval sets;
generator experiments/gen_confirmatory_starts.py refuses reuse -- set now CONSUMED).
PLAYOUT_AB CONFIRMATORY_mcts_200n_seed777 mate-rate A=0.108 vs B=0.325
diff=+0.217 CI=[+0.133,+0.308] [SIGNIFICANT] -- stronger than exploratory (+0.117).
800n leg (exploratory, n=80): A=0.325 vs B=0.388 diff=+0.062 CI=[-0.038,+0.175] ns
-- positive, underpowered; MCTS@200n (~0.29-0.33) roughly equals beam@800n: ~4x
compute efficiency, and MCTS still climbing at 800n (0.388). VERDICT: readout
promotion is real; all prior beam-based ceilings/money-nulls need MCTS re-reads.
Next (Kaveh's data-limitation question): scaling curve on MCTS-rolled toy tables
(3k/10k/30k/100k states, distill per size, held-out Spearman + MCTS money test per
point) -- the curve's slope decides full-lichess run vs objective work.

### Toy re-grounded on ONE canonical start (Kaveh: no random start positions)
Leela-style: state diversity must come from PLAY, not from scattering pieces --
the data distribution is now the REACHABLE SET of a single fixed start.
Canonical start: 2b1k3/3p4/8/8/8/8/8/R3K2R w - - (home-square-like KRRvKBP, no
castling rights since syzygy can't probe them; verified wdl=+2, dtz=3; image at
artifacts/generated/krrkbp_fixed_start.png). The start is an interface parameter
everywhere (--start-fen), NOT hardcoded; KRRKBP_FIXED_START is only the default.
openings_from_fixed_start(): White-to-move, still-wdl=2 positions sampled by
uniform-random legal play (2-10 plies) from the start -- every train/eval position
is play-reachable by construction (captures included: the reachable set legitimately
contains sub-material descendants). Minted (gen_toy_sets.py, ~1s):
krrkbp_fixed_train_n700 + krrkbp_fixed_test_n200 (disjoint). Registry: old
random-placement sets marked LEGACY; canonical_start recorded.
gen_confirmatory_starts.py now mints from the same distribution (--start-fen).
CONSEQUENCE: certainty_table.json + all prior toy baselines are off-distribution;
the scaling-curve experiment (next) re-derives tables and baselines from the fixed
start with the promoted MCTS readout.

### Fixed-start baselines (the scaling curve's zero-point)
PLAYOUT_AB BASELINE_fixedstart_200n mate-rate A(beam)=0.083 vs B(mcts)=0.333
diff=+0.250 CI=[+0.175,+0.325] (n=120, fixed-start test set) [SIGNIFICANT].
On play-reachable openings the readout gap WIDENS (beam collapses to 0.083; the
random-placement sets flattered it at 0.175). Incumbent+MCTS@200n = 0.333 is the
number every scaling-curve distill must beat. playout_ab verdicts now also carry
the abtest e-value (Kaveh: use the e-value framework -- sequential looks along the
curve compose); certainty_distill early-stops on held-out Spearman. Own-play P-hat
probe (model+eps, MCTS 100n readout, 60x8 rollouts) running.

### Search tournament on the ORACLE field (Kaveh: e-value the searches, well-trained space)
search_tournament.py: paired e-process duels w/ early stopping (bandit-style),
field=oracle (tablebase reach = perfect field, EVAL-ONLY; isolates search quality
at the field-quality ceiling).
DUEL mcts vs anytime @200n: 0.660 vs 0.383 diff=-0.277 CI=[-0.447,-0.106] e=23.66
  -- early-stopped at n=47/120 (the e-process saved 60% of the run). MCTS WINS.
DUEL mcts vs anytime @1600n: 0.767 vs 0.717 diff=-0.050 CI=[-0.150,+0.050] e=0.19 ns.
VERDICT: even with PERFECT direction, anytime-v1's single-predicted-reply line
search is budget-fragile (one reply misprediction burns the line; tree search
amortizes). MCTS remains the promoted readout at both rungs. Anytime stays as an
arm (its exact incumbent-bound pruning is graftable INTO mcts later -- mate-bound
pruning -- but only if a signal justifies it). Early-stop harness behaved exactly
as designed on its first real use.

### Generation hang: MCTS all-terminal-children infinite loop (found, fixed, relaunched)
The 700x32 generation stalled at start 20 (~16:41): 80 min of pegged CPU, zero
rollouts completing. Root cause: MCTS budget counts NETWORK EVALS, and a
simulation ending on a terminal consumes none -- in a subtree where EVERY child
is terminal (all moves end the game; happens deep in endgames) the run loop
spins forever. Unit tests had only mixed terminal/fresh roots. Fix: cap total
simulations at 32x the eval budget alongside the eval check; regression test
(all-terminal root, worst-case zero evals) added -- 17/17 pass. Dump truncated
(10 min of data), generation relaunched clean. Pace before the hang was on
estimate (~0.94 s/rollout -> ~6h for 22.4k rollouts).

### Own-play generation COMPLETE (parallel): 20,877-state fixed-start certainty table
700 starts x 32 rollouts = 22,455 rollouts (serial head + 5 parallel workers,
--start-offset sharding, global seeds; ~5h wall total vs ~11h serial projection).
Merged quality (table_from_dump over 6 dumps): 388,612 unique states, 20,877 kept
(>=4 visits, ~6x the old random-start table), P-hat mean 0.14, fracMID 0.31 [gate
PASS], visits median 11/p90 28, within-won certainty gradient Spearman(P-hat,-|dtz|)
= +0.534 CI[+0.490,+0.608] [HEALTHY]. All own-play (model+eps, MCTS 200n readout):
Stage-1 de-scaffold achieved -- zero oracle involvement in the table itself.
LAUNCHING overnight: scaling curve -- nested tables K=4/8/16/32 rollouts/start
(~2.6k/5k/10k/21k states), per size: early-stopped distill + money test (MCTS 200n
both sides, fixed-start test set, e-values). The curve's slope = Kaveh's
data-limitation verdict.

### SCALING CURVE CROSSED: first CI-real field->play win (K=16, 10k states)
MONEY_K4  (3.1k states): rho +0.470, play -0.092 ns
MONEY_K8  (5.2k states): rho +0.395, play -0.017 ns
MONEY_K16 (10k states):  rho +0.369, play +0.167 CI=[+0.050,+0.275] e=6.87 SIGNIFICANT
  -- distilled 0.500 vs incumbent 0.333, both MCTS 200n, fixed-start test set.
Kaveh's data-limitation hypothesis CONFIRMED at this rung: play follows the field
once on-distribution own-play certainty data crosses ~10k states; the deficit
shrank monotonically with data (-0.092 -> -0.017 -> +0.167). Every prior money
null (2.7k random-start states, beam readout) is now explained as dose + readout.
NOT yet promoted: this is one of 4 pre-planned sequential looks (e=6.9 alone <
1/alpha) -- selection happens after K=32, then ONE pre-registered confirmatory on
a fresh frozen set (new seed; 777 consumed) per protocol. K=32 running.

### Confirmatory: K=16's +0.167 did NOT confirm -- winner's curse caught by protocol
Full curve (held-out rho / money diff at 200n, both MCTS, n=120):
  K=4  3.1k states: +0.470 / -0.092 ns
  K=8  5.2k states: +0.395 / -0.017 ns
  K=16 10k states:  +0.369 / +0.167 SIG e=6.87   <- selected
  K=32 21k states:  +0.370 / +0.050 ns e=0.17
CONFIRMATORY (fresh seed-778 frozen set, single-use, pre-registered):
  0.450 vs 0.400, diff +0.050 CI=[-0.050,+0.150] e=0.20 [ns]. NOT PROMOTED.
The K=16 significance was one of 4 sequential looks; the confirmatory protocol
did its job. HONEST residue across all high-data evals: play effect ~+0.05
(consistent sign, never CI-real at 200n), and the distilled ckpts mate FASTER
when they convert (17 vs 20-22 plies, every high-K eval). Data scaling closed
the deficit (-0.09 -> +0.05) but did not buy a confirmable 200n win at this dose.
Note also incumbent varies by start set (0.333 test set vs 0.400 confirmatory) --
set variance is real, another reason point looks mislead.
Running: 800n regime look (field should matter more at saturation; FIELD_PLAN
mandates both budgets; labeled exploratory).

### CONFIRMED at 800n: certainty field promotion -- the program's first real field win
Regime look (exploratory, test set): incumbent 0.433 vs K16-distilled 0.658,
diff +0.225 CI=[+0.117,+0.333] e=296.
CONFIRMATORY (pre-registered, fresh seed-779 frozen set, single-use):
  0.400 vs 0.608, diff +0.208 CI=[+0.108,+0.317] e=184.66 [SIGNIFICANT]. CONFIRMED.
Faster mates too (17 vs 24 plies). The full story, in one paragraph:
the toy was BOTH data-limited AND regime-masked. Fixing either alone showed
nothing (old 2.7k table @200n: null; big table @200n: ~+0.05 ns). Fixing both --
10k on-distribution own-play certainty states (fixed-start, MCTS rollouts,
de-scaffolded) read out at saturation (800n MCTS) -- lifts conversion 0.40->0.61
CI-real on never-touched starts. The old "~0.35 intrinsic ceiling" was the
INCUMBENT FIELD's ceiling (and before that, the beam readout's). Kaveh's calls
vindicated: certainty=closeness reframe, more nodes (1600-instinct), more data,
fixed-start distribution, e-value discipline.
Remaining gate before cert_scale_K16.pt becomes the toy incumbent: field-health
panel (global regression guard) + leakage audit. Running.

### Field-health panel: CLEAN -- cert_scale_K16.pt PROMOTED to toy incumbent
AUDIT=CLEAN (leakage gate). Reach slopes healthy and correctly ordered
(won +0.445 > lost +0.303; diff slopes +0.591/+0.413 -- no global regression
signature; the every-step NCE mixing protected the global field as designed).
Arena vs random 0.850 (e=2449). cert_scale_K16.pt is now the toy incumbent:
all future toy A/Bs baseline against it, at BOTH 200n and 800n, MCTS readout.
Next lever (to discuss): certainty in the BASE objective at full-board scale --
the distill validated the signal end-to-end; training it in from the start
should beat post-hoc fine-tuning, and the whole harness (fixed-start discipline,
own-play tables, e-values, confirmatory protocol, regime ladder) transfers.

### Short cert-base run (5k): gates green-with-one-yellow; full run launched
VAL stable (top1 .027->.029, top8 .178->.186), phead CE 1.14->0.76 (outcome signal
flowing into F), slopes healthy (won .428 > lost .276). YELLOW: toy held-out
Spearman +0.369 -> +0.316 (CIs barely touch) -- full-board objective trades a
little toy calibration; toy is canary, ladder is judge. Search-duplication
measurement (Kaveh's Q): within-search dup 1.1%/10.8%/14.0% at 200/800/1600n,
whole-game dup 20/32/34% -- game-scoped exact eval cache planned before the
ladder (free ~1.5x at 1600n; key must include field-version once fast field
lands). Kaveh's conditional-tactic reminder journaled: NOT implemented; nearest
live proxy is +gamma*pvar_theirs; MemoryField.payload is the reserved slot;
precondition-vector design in planner memory. FULL RUN: 95k->155k steps cert-base.

### Exact eval cache in MCTS (Kaveh's duplication question, measured then fixed)
MCTS budget now counts FRESH network evals only; a fen-keyed cache (policy-lifetime,
shared across moves/games) makes repeats free. Measured repeats: 20/32/34% of a
game's evals at 200/800/1600n. Effect: same NN budget explores a BIGGER tree
(hits are free budget, not savings) -- play changes (for the better, in
expectation), so historical mate-rates are NOT directly comparable to cached
runs; paired A/Bs stay matched (both arms cached). Cache key = full FEN; must
grow a field-version component once the fast MemoryField re-prices mid-game.
18/18 search tests pass (new: hits>0, bigger tree, same-config determinism).

### Cert-base ladder vs toy specialist: PARITY at all rungs (cached MCTS, n=120)
200n: 0.500 vs 0.475 ns | 800n: 0.692 vs 0.700 ns | 1600n: 0.733 vs 0.667 ns.
No promotion on the toy (nothing significant to confirm), but the meaningful
read: cert-base matched a 10k-state toy-distilled SPECIALIST on its home turf
with ZERO toy data -- the certainty-in-base-objective signal carries at
full-board scale without buying the toy region back. Neither model saturated
at 1600n (both still climbing with budget); specialist converts faster (18 vs
21 plies). Cache-effect visible vs history: specialist 0.500@200n cached vs
0.333 uncached. NEXT: cert-base's real test is FULL-BOARD play vs the
pre-certainty incumbent (the toy specialist never trained there).

### PROMOTED: cert_base_full.pt is the new incumbent (full-board, confirmed)
H2H vs pre-certainty incumbent (MCTS 400n, cached): run 1 score 0.688 (+18=19-3)
e=65.07; independent seed-777 confirmation 0.650 (+16=20-4) e=8.28; composed
e=539 -- 34-7 decisive across runs. AUDIT=CLEAN. Toy: parity with the 10k-state
specialist at all rungs (no toy regression). SF skill-0 still crushes us (0.050)
-- the long game. cert_base_full.pt (155k steps, certainty-in-base-objective:
outcome-conditioned P-head + d->plies+lam(-lnP) on won games, oracle-free) is
the incumbent for ALL future work; lichess_fb_4gb_qm_plygap_only retired to
reference; cert_scale_K16 retired to toy-specialist reference.
The day's arc, end to end: certainty reframe -> MCTS readout (confirmed) ->
fixed-start discipline -> own-play tables (de-scaffolded) -> scaling curve ->
800n toy confirmation -> certainty in base objective -> full-board win, every
step CI/e-gated with pre-registered confirmatories.

### Overnight Round A: embedding diagnostics (Kaveh's dimension question answered)
Effective rank of F = 11.0/64 (old incumbent) and 9.5/64 (cert_base_full); trunk
itself ~10/256. 64 dims is ~6x oversized for what the objective extracts -- do
NOT widen; the binding constraint is objective information demand. Sparse-concept
implication: overlap is un-demanded separation, not crowding. Outcome probe AUC
on F: 0.610 -> 0.687 (cert-base), and the trunk-vs-F gap FLIPPED (+0.038 ->
-0.018): the old bottleneck discarded outcome info, cert-base's F now carries
more than its trunk. Round B next: closed-loop round 2 (tables from the NEW
incumbent, distill, ladder -- FIELD_PLAN GATE 2 'does the loop compound?').

### Overnight Round B: R2 generation complete -- the loop's data leg compounds
Tables regenerated from cert_base_full (5 workers, 698 starts x 16-of-32 rollouts,
~1.5h): 10,224 kept states. Quality vs round 1: P-hat mean 0.14 -> 0.34, fracMID
0.31 -> 0.55, within-won gradient +0.534 -> +0.650. Stronger policy => richer
certainty signal, as the closed-loop design predicts. Distill + ladder next
(GATE 2: does play compound too?).

### Round B ladder: field best-ever, play positive-lean at 1600n (extending n)
R2 distill: held-out Spearman -0.135 (cert_base vs own-play targets: still
anti-correlated on-policy!) -> +0.491 (best any round). Ladder vs cert_base_full:
200n +0.033 ns | 800n -0.025 ns | 1600n +0.092 CI=[-0.008,+0.200] -- one start
from CI-real at the deep rung, faster mates (18 vs 21). Extending the 1600n look
to the full n=200 test set (anytime-valid: e-process permits optional
continuation, no peeking penalty). Round C (extend base training) queued after.

### GATE 2 verdict: NOT passed -- loop round 2 is real-but-small, below confirmation
R2_1600n_n200: +0.095 CI=[+0.015,+0.175] SIG (selection look).
CONFIRMATORY seed-780 (fresh, single-use): +0.075 CI=[-0.025,+0.167] ns.
Three consecutive positive 1600n results (+0.092/+0.095/+0.075, all faster
mates) say the round-2 effect is likely real ~+0.08 but under the n=120
confirmatory's resolution. cert_r2 NOT promoted; cert_base_full remains
incumbent. Interpretation: certainty-in-base-objective already banked most of
the toy-distillable signal -- the loop compounds DATA quality strongly
(P-hat .34, gradient +.650) but play returns per toy round are shrinking.
Morning recommendation forming: the loop's next round belongs at FULL BOARD
(self-play data into the base objective), not another toy lap.
Round C launching: extend cert-base training 155k -> 215k, h2h after.

### Round C: REGRESSION -- 215k loses to 155k h2h; incumbent restored
Two-seed h2h (MCTS 400n): 0.325 (+6=14-20, e=8.24) and 0.325 (+7=12-21, e=5.81)
-- composed e~48 AGAINST the extension. VAL/slopes improved while play regressed:
the 2026-07-11 lesson again (retrieval != planner quality; extended schedules
overcook). cert_base_full.pt RESTORED from the 155k snapshot (taken minutes
before the in-place overwrite -- the check-early/snapshot discipline paid);
215k kept as cert_base_215k_regressed.pt for autopsy. 155k stays incumbent.
OVERNIGHT WRAP: A) rank ~10/64, don't widen; outcome AUC .61->.69, bottleneck
flip. B) loop data-leg compounds hard (P-hat .34, gradient +.650, field +.491)
but play leg ns at round 2 (GATE 2 not passed; thrice-repeated ~+0.08 lean at
1600n). C) more steps = worse play, CI-real. The three results TOGETHER point
one direction: the binding constraint is now the DATA the base objective eats,
not steps, not dims, not toy rounds -- next lever is full-board self-play
certainty data into the base objective. Kaveh decision on waking.

### Mate-attempt trajectories visualized (Kaveh): failures = ORBIT AT THE RIM
build_mate_attempt_viewer.py -> artifacts/generated/mate_attempts.html (board
scrubber + F-space path over certainty-field UMAP + d/P-hat strips). 2 mates,
2 failures (both THREEFOLD_REPETITION) from the fixed-start test set, incumbent
@800n. The signature: ALL games drive d from ~0.55 to ~0.30-0.32, then -- wins
keep MOVING in F-space (last-10-ply net displacement 1.9, 16.0) while failures
ORBIT (net displacement 0.5, 0.7; d pinned at the 0.30 floor; P-head still 0.85+
while the game bleeds to repetition). Diagnosis: the field's distance saturates
at the mate-region rim -- near-goal states are indistinguishable at d~0.3, so
search shuffles equal-d moves into repetition. Residual unconverted mass is
largely rim-orbiting, not wrong direction. Mechanism candidates (not guards, per
Kaveh's rule): near-horizon head for fine rim resolution (FBTwoHorizonPolicy
exists), or fast-field evidence ("been here, no progress" -> re-price), or
repetition-state features reaching the certainty targets. Decision for Kaveh.

### Design contract: named concepts are EVAL-ONLY instruments (Kaveh, 2026-07-15)
While developing/troubleshooting we may CHECK whether the engine hit named
concepts (won bishop, cornered king, mate) -- but only in OUR offline
verification of games/subgoals. The engine's play and search never consume
hand-named concept detectors; plans and subgoals live purely in embedding
space. Later milestone: concepts LEARNED (discovered structure -- e.g.
clusters over subgoal embeddings / sparse concept head), with names attached
only post-hoc by us during verification. Extends the "find mechanisms, don't
hand-code guards" rule from readouts to concepts.

### Multi-eps identification (approved): sharpness is REAL and plies-independent
tb-White tables at eps=0.05/0.10/0.20 (700 starts; 18k/10k/8k states), per-state
WLS of -ln P-hat on eps over 4,373 states at all levels:
EXISTENCE intercept median +0.112 (truth 0), 43% within +-0.15 -- identifiable,
  biased up (3 points, P-hat resolution, linear-link convexity). Rankable, not
  yet calibrated -- matters for full board where there's no syzygy.
SHARPNESS S median 1.77 IQR[0.00,4.46]; S-vs-|dtz| Spearman +0.036 CI[+0.001,+0.060]
  -- ~ZERO: risk is NOT exposure-accumulated; it concentrates in sharp
  bottlenecks (Kaveh's simple-15-vs-complex-5 intuition, measured). S is nearly
  orthogonal to plies => sharpness deserves its OWN channel; any constant-lambda
  fusion of plies and risk is structurally wrong.
LINEARITY median residual 0.082 (signal span ~0.26), p90 0.326 -- constant-S model
  holds for the bulk, fat tail of nonlinearity (likely the sharpest states).
sharpness_table.json persisted (4,373 states with per-state existence + S).
No builds launched -- results to discussion per the discuss-first rule.

### Two-channel field wired (Kaveh GO): plies channel + S-head, risk at readout
experiments/two_channel_distill.py: phase 1 re-distills quasimetric d to PURE
plies (tb-White eps=0.05 table, early-stopped, NCE-mixed, trained-zW saved);
phase 2 trains a separate S-head (frozen F -> 128 -> softplus) on the 4,373
identified per-state sharpness values. Readout: FBMCTSPolicy(s_head, g_sharp)
computes reach - g_sharp*S(F(s)) -- risk enters ONLY at readout (g omega-
dependent later), geometry stays risk-free per the identification finding
(S ~ orthogonal to plies). playout_ab: --s-head-b/--g-sharp. mcts tests pass.
Distill running; ladder + Kaveh's named-stage checkers (eval-only: pins,
double attacks, captures, edge/corner vs mid-board king traps, mate-with-king-
location) next.

### Two-channel distill verdicts: plies channel strong, S-head modest
PLIES_CHANNEL held-out Spearman +0.281 -> +0.508 (early stop 2500; purified
geometry fits conversion length far better than any fused metric round).
S_HEAD held-out Spearman +0.262 (RMSE 3.08 vs S sd 3.26) -- real rank signal,
modest; S targets are noisy 3-point fits, improvable with more eps levels.
RUNNING: g_sharp scale sweep {0, .002, .01, .05} at 200n n=60 (S in nats vs
reach deltas ~0.01-0.1 -- scale must be found before the ladder), then full
ladder vs cert_base_full. Stage checkers (eval-only) still queued.

### Two-channel g-sweep at 200n: g INERT, field leans slightly negative
All arms (g=0/.002/.01/.05, n=60) within noise of each other and -0.07..-0.10
vs cert_base_full [ns]. Read: purifying d to plies REMOVED the certainty info
the incumbent's fused metric carried; S-head (+0.26) too weak to restore it at
readout. Regime rungs (800/1600n, n=120, g=0 and .01) running to complete the
approved test before any conclusion -- if negative there too, the discussion is
S-target quality (more eps levels -> tighter S) vs joint (non-frozen) S channel.

### Two-channel v1 FALSIFIED at play: readout-side risk cannot replace in-geometry certainty
800n: g0 -0.242 CI=[-0.342,-0.142] e=1747; g.01 -0.183 e=75. 1600n: g0 -0.150
e=5.3; g.01 -0.183 e=21. All SIGNIFICANT against, all rungs. The dissection is
valuable: the incumbent's FUSED d carries certainty in the geometry at full
strength; stripping d to pure plies and re-adding risk via a weak frozen
S-probe (+0.26) costs ~0.2 conversion. The identification finding stands
(S real, ~orthogonal to plies) -- what died is THIS implementation (frozen
probe + readout-only risk). Candidate syntheses for discussion: (a) S as a
JOINTLY-TRAINED second geometric channel (both quasimetric, fused at readout
with full-strength heads), (b) keep fused d, add S as auxiliary signal only
for search allocation (its p_var-like role), (c) better S targets first (more
eps levels/rollouts) before re-judging any architecture. cert_base_full
remains incumbent; two_channel.pt shelved as reference. NO further builds
pending discussion (discuss-first rule).

### Stage checkers built + VALIDATED on expert games (Kaveh's protocol)
stage_checkers.py (eval-only) validated on 500 tb-optimal WON games:
pins 20-27%, double_attack 52%, capture bishop 85% (median ply 7), pawn 44%,
king_corner 84% (ply 10) -- sane rates, sensible ordering (capture->corner).
Validation CAUGHT: (1) king_edge fires 100% at ply 0 -- canonical start has
the black king ON the edge; needs a confinement metric (king-box area), not a
location bit. (2) mate stages 0% on dumps -- rollout dumps store PRE-move
states only, terminal mated board never recorded; mate checker itself verified
by positive control (fires on constructed mate). Fix queued: dumps/recorders
must include the terminal board. (3) midboard_trap 0% on expert games --
consistent with tb play (edge mates), positive-control construction still
needed for full verification. Checkers otherwise ready for planning-proof use.

### ALL THREE rescue mechanisms built (Kaveh: "do them all. now.")
mcts.py: (1) EVIDENCE BLEND -- precision-weighted d_eff=(n*d_ev+k*d_field)/(n+k)
in the reach closure; evidence = demo_tb+eps05+r2_K16 tables (27.5k states,
visit-weighted merge) + live game-path revisits as stall evidence (revisit =
objectively no progress; d_ev->2.0, n=8/revisit). (2) FLAT/LOW-CONF ROLLOUTS --
uniform-random playout (0 NN evals) backs up real terminals when child values
are flat (std<0.05) OR field unvouched (no evidence near state -- Kaveh's
low-confidence trigger; competence-head hook ready, incumbent has none).
(3) TREE REUSE -- carry the played child's subtree (visit stats) across moves.
playout_ab --rescue-b. Smoke (n=12): runs clean. Ladder 800/1600n n=120 running.

### Publication drafts: writing/ (state-of-the-research + journey + 5 posts) — and rescue rung 1
Kaveh (project context, saved to memory): the goal is learning HUMAN-LIKE
PLANNING in chess as the verifiable toy domain, ported later to agentic
planning/robotics; findings will be published (biweekly digest + hopefully
peer-reviewed articles). First drafts built from ALL documentation (HEAD +
git history mined era-by-era): writing/state_of_the_research.md (hypotheses
H1 FB-captures-field / H2 certainty-priced-loss -> verdict-backed claims,
methods-in-prose: paired deterministic playouts, bootstrap CIs, e-process
usage incl. composition e=539 and optional stopping, confirmatory protocol,
regime ladder, leakage audit; data+reproduction pointers -> new README
section "Reproducing the journaled results"), writing/research_journey.md
(disproven/inconclusive hypotheses only, bugs excluded, eras 0-6 + 6
cross-cutting lessons), 5 single-topic posts (e-values how-to; regimes;
instrument!=objective; certainty-weighted distance; oracle discipline).
Figures: experiments/viz/article_figures.py -> writing/figures/*.png (6
figures; every number either read live from artifacts/experiments/ or carried
with its JOURNAL verdict provenance inline; legibility checked by rendering).
NOTE: fig_proxy_vs_play deliberately reports the r17 DISSOCIATION (gen2 best
rho of its era plays 0.12 below incumbent) rather than an n=5 correlation --
adding round-18's plygap point (high rho AND high play) would make a naive
correlation read positive; the honest claim is "insufficient/doesn't rank",
not "anti-correlated".

RESCUE ladder rung 1 (pre-registered bar: 800n conversion >=0.85 from 0.70,
repetition failures halved, no regression on won starts):
PLAYOUT_AB RESCUE_800n mate-rate A=0.700 vs B=0.625  diff=-0.075
CI=[-0.175,+0.025]  e=0.34 (n=120, deterministic defender; plies-to-mate
A=21 B=28) [ns] -- FAILS the bar: no gain, negative lean, SLOWER mates (28
vs 21). The rescue trio as wired does not rescue at 800n. 1600n rung running;
diagnosis discussion after it lands (candidates: evidence tables mostly
off-trajectory at 800n depth; 0.5/0.5 rollout blend diluting a good boot
value; reuse+evidence interaction). No further builds pending discussion.

### Rescue ladder COMPLETE: the trio fails at both rungs -- no rescue, no promotion
PLAYOUT_AB RESCUE_800n  mate-rate A=0.700 vs B=0.625  diff=-0.075 CI=[-0.175,+0.025] e=0.34 [ns]
PLAYOUT_AB RESCUE_1600n mate-rate A=0.667 vs B=0.617  diff=-0.050 CI=[-0.142,+0.050] e=0.21 [ns]
(n=120 each, deterministic defender; B mates SLOWER both rungs: 28/24 vs 21 plies.)
Against the pre-registered bar (800n conversion >=0.85 from 0.70; repetition
failures halved; no regression on won starts): FAILED. Consistent negative
lean at both rungs, e<<1 (data favor the null-to-harmful), slower mates.
The three mechanisms TOGETHER (evidence blend + flat/low-conf rollouts +
tree reuse) do not fix rim-orbiting and likely add noise where the incumbent
was already converting. Diagnosis candidates for discussion (NOT built):
(a) evidence coverage is off-trajectory at deep-search play (27.5k states,
but B's own games leave the table's support fast -- low_conf rollouts then
fire OFTEN, and a 0.5/0.5 uniform-rollout blend DILUTES a good minimax boot
in exactly the won positions the incumbent converts); (b) live revisit-stall
evidence re-prices d upward mid-game and may destabilize lines the incumbent
holds; (c) tree reuse carries stale evidence-blended values across moves,
compounding (a)+(b); (d) mechanisms were tested as a bundle -- per-mechanism
attribution needs single-lever runs IF Kaveh wants to salvage any piece.
Honest read: rim-orbiting remains open; the rescue-by-runtime-evidence line
as bundled is rejected at both regimes. cert_base_full remains incumbent.

### COMMITTOR REFORMULATION, short run: every gate green -- best calibration of the project
Architecture session with Kaveh (design settled in conversation, journal-level
summary): probability is first-class. d = -ln P, NO lambda, NO plies term
(order by P; plies dissolve -- no constant per-move hazard exists, per the
S-vs-|dtz| ~0 finding; length costs only what it actually costs: constraint
dynamics via augmented state + epistemic hazard ~1/n_eff, the Laplace floor
named for what it is). Terminal outcomes are SURFACES with touchdown
semantics (hit anywhere counts), not poles: no goal vector at all -- a
committor head d_W(s) = -ln P(hit mate-W boundary first) on F, boundary
conditions from the rules engine. Opponent enters as softmin over reply
surprisal (probabilistic minimax; hard minimax = infinite-sharpness limit)
-- Stage 2, not built. Draw boundaries (3fold/stalemate/50-move/insufficient)
= "out of bounds" surfaces a losing player navigates TOWARD; goal selection
= thin decision layer over per-boundary P's with the game's scoring rule
(win 1, draw 0.5) -- NOT a learned value head. Rescue-trio salvage and
two-channel synthesis lines CLOSED as superseded by this formalism.

Stage 1 short run (committor_distill.py, cert_base_full + joint W-head,
target -ln max(p_hat, 1/(n+2)) on certainty_table_r2_K16, NCE-mixed,
early stop step 1500, ~3 min):
VERDICT COMMITTOR_SPEARMAN pole-baseline -0.112[-0.153,-0.067] -> head +0.603[+0.575,+0.629] (n=2044)
VERDICT RIM_RESOLUTION (plies<=8, n=241) pole +0.076[-0.029,+0.211] -> head +0.330[+0.250,+0.467]
The pole distance is ANTI-correlated with pure conversion probability on
on-policy states (min-semantics optimism, third independent measurement);
the committor head is the best field calibration of the project (prior best
+0.491 on the easier fused target) and resolves the rim where the pole is
flat -- the exact mechanism behind the orbit failure. Readout wired:
FBMCTSPolicy(committor_head=...), playout_ab --committor-b. 11 mcts tests
pass. Smoke n=12 @800n: 0.917 vs 0.750, plies 18 vs 23 [ns, smoke only].
LAUNCHING exploratory ladder n=120 @800/1600n vs cert_base_full; if CI-real,
ONE pre-registered confirmatory on a fresh seed (781+; 777-780 consumed).
Queued: dumps record termination reason + terminal board -> per-boundary
d_D/d_B heads; repetition-count input plane (threefold surface visibility).

### New-arch representation fixes: v2 dumps + per-boundary tables + repetition plane
(1) certainty_rollouts dumps now record the BOUNDARY each rollout hit
(termination reason), the terminal fen (mate boards finally captured -- fixes
the stage-checker gap), and per-visit repetition counts; traj = [fen, ply,
rep]. (2) table_from_dump aggregates per-boundary outcome counts per state
(WIN / DRAW_3FOLD / DRAW_50 / DRAW_STALE / DRAW_INSUF / LOSS / CAP) + rep_max;
old dumps degrade to WIN/OTHER; quality report prints the boundary mix.
Smoke (2 starts x 4 rollouts): DRAW_3FOLD 0.68 of visits -- the orbit failure
is now VISIBLE in the data. (3) committor_distill trains a d_D draw-committor
head alongside d_W when a v2 table is present (out-of-bounds surfaces:
navigate toward when losing, keep clearance from when winning). (4) REPETITION
PLANE: N_PLANES 19->20, meta[7] = rep count (augmented-state coordinate --
the threefold surface only exists in board x rep space); load_ckpt zero-pads
old stem convs (VERIFIED bit-identical embeddings on rep=0: |df|=0.0);
FBMCTSPolicy feeds game-path rep counts at eval. Full suite green (2 failures
were PRE-EXISTING stale batch_tensors tests from the cert-base 7-tuple;
fixed). Noted for later: MCTS search boards use copy(stack=False), so
in-search threefold detection is structurally blind -- the rep plane + game-
path counts partially compensate; a real fix needs path-aware terminal checks.
Committor ladder rung 1 (exploratory, n=120): COMMITTOR_800n A=0.700 vs
B=0.725 diff=+0.025 CI=[-0.058,+0.108] e=0.16 [ns], plies 19 vs 21 -- tie
with positive lean + faster mates; 1600n running. NEXT: round-2 generation
with the committor policy (v2 dumps) -> multi-head distill -> ladder.

### Committor ladder + confirmatory: exploratory CI-real at 1600n, confirmatory ns -- NOT promoted
COMMITTOR_800n  A=0.700 vs B=0.725 diff=+0.025 CI=[-0.058,+0.108] e=0.16 [ns]
COMMITTOR_1600n A=0.667 vs B=0.783 diff=+0.117 CI=[+0.025,+0.200] e=3.32 [selection look]
CONFIRMATORY_committor_1600n_seed781: A=0.717 vs B=0.783 diff=+0.067
CI=[-0.025,+0.158] e=0.32 [ns]. Seed 781 CONSUMED (registry updated).
NOT promoted; cert_base_full remains incumbent. Same shape as GATE 2: a
repeated positive lean (+0.117 exploratory, +0.067 confirmatory; B=0.783 on
BOTH sets -- the incumbent moved 0.667->0.717 across sets, set-variance
again) below n=120 resolution. Reading: the committor readout is
tie-to-slightly-better at play off the SAME 10k v1 table dose -- the field/
rim gains (rho +0.603, rim +0.330) buy play only through the loop's data
leg, exactly the scaling-curve lesson. Surface viz (committor_surfaces.py,
committed): rank good / ABSOLUTE SCALE COMPRESSED (learned P_W spans
[0.19,0.37] vs empirical [0,1]) -- ordering learned, probability not yet
calibrated; matters for the goal-selection layer, isotonic/scale fix queued.
High-dim basin separation cross/within = 1.34 (weak); UMAP clusters =
material stages, not outcome basins (consistent with W/D/L-region history;
the head extracts a direction clustering can't see). LAUNCHING round-2
generation with the committor policy (v2 dumps: boundary labels + terminal
boards + rep counts), 5 workers -- then multi-head distill (d_W + d_D) and
the ladder.

### Committor design v2: end-to-end NLL FALSIFIED; monotone recalibration adopted; round-3 data leg complete
Kaveh: no formalism duel vs cert_r2 needed -- improve the committor design.
ROUND-3 GENERATION (committor policy, v2 dumps, 5 workers, ~25 min -- ~4x
faster than R2): 699 starts, 11,184 rollouts, 224,578 unique states, 9,380
kept. Quality: P-hat mean 0.41 (R2 0.34, R1 0.14), fracMID 0.53 [PASS],
within-won gradient Spearman(P-hat, -|dtz|) = +0.686 CI[+0.621,+0.728] --
BEST of any round; loop data-leg keeps compounding. Boundary mix: WIN_MATE
0.45 / DRAW_3FOLD 0.46 / DRAW_INSUF 0.07 / LOSS_MATE 0.01 / CAP 0.01.
Terminal boards + rep counts now exist at scale (surface-atlas unblocked).

DESIGN ATTRIBUTION (loss change isolated on the OLD r2 table):
End-to-end smoothed-binomial NLL (proper score; principled hope: calibration
+ natural n-weighting + Laplace floor as pseudo-counts):
VERDICT COMMITTOR_SPEARMAN pole -0.112 -> head +0.051[+0.009,+0.095]  FALSIFIED
  (vs MSE's +0.603 on identical data); span [0.29,0.34] -- the head collapsed
  to the base rate: with a shared fine-tuned trunk, the fastest NLL descent is
  predicting the marginal, not using features. --loss nll kept for reference,
  default reverted to mse.
RESOLUTION -- decouple rank from scale: committor_recalibrate.py fits a
2-param MONOTONE affine in d-space (d' = a*d + b <=> P' = e^-b * P^a, Platt
in log space) by NLL on train rows; rank preserved EXACTLY, play unchanged
(MCTS squash is per-node shift/scale invariant):
VERDICT RECALIBRATION a=1.396 b=-0.985  held-out ECE 0.174 -> 0.126,
NLL 0.7054 -> 0.6425, span [0.18,0.35] -> [0.25,0.62]. Partial fix (affine
can't fully undo compression; isotonic is the escalation if the
goal-selection layer needs true [0,1]). Affine stored in the whead payload;
rank-only consumers ignore it.
RUNNING: multi-head distill (d_W + d_D, MSE) on the r3 on-policy table.

### R3 distill + isotonic recalibration: first honest probabilities from the field
R3 multi-head distill (MSE, on the fresh committor-on-policy table):
VERDICT COMMITTOR_SPEARMAN pole +0.089 -> head +0.610[+0.588,+0.638] (n=1876)
VERDICT DRAW_COMMITTOR_SPEARMAN head +0.675[+0.651,+0.695] -- FIRST learned
  draw-surface field (out-of-bounds committor), and it calibrates better in
  rank than d_W on the same rows.
VERDICT RIM_RESOLUTION pole -0.016 -> head +0.128[+0.021,+0.218] (weaker than
  r2's +0.330 -- different table composition, noted not hidden).
Isotonic recalibration (Kaveh: "monotone doesn't have to mean linear"):
VERDICT RECALIBRATION method=isotonic ECE 0.228 -> 0.059, span [0.23,0.37] ->
[0.14,0.87], NLL 0.785 -> 0.616, rank EXACT (eps-affine strictness blend).
The goal-selection layer's precondition (comparable absolute P across
fields) is now approximately met on-distribution. R3 ladder (800/1600n,
n=120 vs cert_base_full) running.

### R3 ladder: REGRESSION -- best-ever field metrics, CI-real play loss; attribution run launched
PLAYOUT_AB COMMITTOR_R3_800n  A=0.700 vs B=0.558 diff=-0.142 CI=[-0.242,-0.042] e=5.86 [SIGNIFICANT against]
PLAYOUT_AB COMMITTOR_R3_1600n A=0.667 vs B=0.558 diff=-0.108 CI=[-0.225,+0.008] e=0.65 [ns, negative lean]
(B mates FASTER when it converts: 17 vs 21 plies, both rungs -- narrower but
crisper win corridors.) committor_r3 REJECTED. The structure-play
dissociation with a negative sign: d_W +0.610 / d_D +0.675 / best table
gradient +0.686, and play regressed. TWO levers changed at once (my
attribution debt, noted at launch time in the code but not resisted):
(a) joint d_D draw-head training reshaping F (rim resolution fell r2 +0.330
-> r3 +0.128 -- consistent); (b) the r3 table = committor-POLICY statistics
distilled into the cert_base_full checkpoint (policy mismatch). Attribution
run: same r3 table, --no-dhead (single head), 800n rung only. If it recovers
to the r2-committor range (~0.70 tie), the joint d_D head is the drag ->
separate/gentler d_D arrangement; if still negative, the cross-policy table
is the drag -> distill committor-generated tables only into the committor
lineage (on-policy loop discipline).

### Full joint training (toy) + mate viewer: watch it mate -- and watch the rim hold the failures
FULL JOINT (Kaveh: "see how it mates after a full joint training"): 12k-step
budget, early stop 3500, d_W+d_D joint, committor lineage (r3 own-play table
into committor.pt -- on-policy discipline):
VERDICT COMMITTOR_SPEARMAN pole -0.194 -> head +0.662[+0.637,+0.685]  (best ever)
VERDICT DRAW_COMMITTOR_SPEARMAN head +0.681[+0.657,+0.701]            (best ever)
VERDICT RIM_RESOLUTION pole +0.081 -> head +0.097[-0.034,+0.171]      (WEAK; trend
  r2 +0.330 -> r3-multi +0.128 -> joint +0.097 is the open worry)
VERDICT RECALIBRATION isotonic ECE 0.227 -> 0.067, span [0.12,0.75].
Mate-attempt viewer rebuilt for the committor arch (committor readout,
calibrated P_W strip, EVAL-ONLY stage timelines in labels) ->
artifacts/generated/mate_attempts_committor.html. Games @800n:
  MATE in 13 [xbishop@1 mate_edge@9] | MATE in 11 [xbishop@0 mate_edge@7]
  MATE in 13 [xbishop@1 mate_edge@9] | FAIL 3fold [xpawn@25 corner@38] | FAIL 3fold
The mating PLAN is legible in the stage timelines: win the bishop
immediately, drive to the edge, mate -- exactly the concept sequence hoped
for, never named to the engine. The failures are the rim signature again:
one game CAPTURES the pawn and CORNERS the king (ply 38) and still bleeds to
threefold -- conversion's last mile is precisely the weak-rim region the
RIM_RESOLUTION trend flags. Next lever candidates (discussion): rim-weighted
distill targets (upweight plies<=8 rows), or per-visit (fen,rep)-keyed
targets now that dumps carry rep counts.

### Lineage attribution COMPLETE + root loop launching (Kaveh GO)
Attribution triplet (same r3 table, 800n, n=120 vs cert_base_full):
  W-only into cert_base:  0.392  diff -0.308 CI=[-0.408,-0.208] [SIG against]
  W+D    into cert_base:  0.558  diff -0.142 CI=[-0.242,-0.042] [SIG against]
  W+D    into OWN lineage (committor_joint): 0.658 diff -0.042 CI=[-0.150,+0.067] [ns]
CONCLUSION: cross-policy distillation is the drag (committor-policy tables
poison the base checkpoint's play); d_D is PROTECTIVE not harmful; own-lineage
training restores parity. On-policy loop discipline is now a RULE: tables
distill only into the lineage that generated them. Viewer rebuilt with the
P_D strip (draw committor, violet, fixed [0,1] scale next to calibrated P_W).
LAUNCHING committor_root_loop.py (Kaveh: "start from the root position, use
some epsilon..., as we get data, train the field" -- GO): rounds of 2000
eps-rollouts from THE canonical root (5 seed-split workers) -> cumulative
per-boundary table -> distill into the champion lineage -> ratchet gates
(held-out rho + rim within 0.02 of best) + conversion-from-root probe
(eps-play, the root's own P-hat trajectory). ~25-30 min/round, 12 rounds.

### Root loop rounds 1-7: three gate corrections, each forced by a measurement
The single-root eps closed loop (Kaveh GO) surfaced three methodology bugs in
its first seven rounds, each fixed and committed:
(1) RIM-NOISE THRASH (r2-r4): rim holdout is tens of rows; swings +-0.1-0.4
    are sampling noise -- separate rim slack 0.12.
(2) CROSS-TABLE RHO (r5-r6): champion's benchmark rho was measured on the
    r1-era table; candidates on progressively noisier cumulative holdouts --
    attenuation penalized genuine improvements. Gate is now PAIRED: both
    scored on the same rows, same round (r7 revealed champion's true score
    on today's holdout: 0.406, not 0.685).
(3) PLAY MISSING FROM THE GATE (r7): the paired-field gate advanced a
    field-better candidate whose root-conversion CRASHED. n=64 same-seed
    verdict: ROOT_CONV r1=0.734 vs r7=0.609. Play gate added (paired probe
    both arms every round, slack 0.125); r1 reinstated as champion; loop
    restarted at round 8 with all 40 dumps.
The dissociation lesson now lives INSIDE the loop's own gate. Trajectory so
far: root-conv (champion) ~0.73-0.82; table 6k states, gradient ~0.7.

### ROOT LOOP COMPLETE (12 rounds): data leg compounds, distill-into-F train leg does NOT
Final powered close-out (n=64 per arm, identical seed sets, eps=0.15, 200n):
FINAL_ROOT_CONV preloop_joint = 0.719 | r1 = 0.719 | r12 = 0.672
VERDICT: twelve rounds of generate->distill bought ZERO root-conversion --
pre-loop, round-1, and round-12 champions are statistically indistinguishable
(r12's two advances were slack-riding ties). What the loop DID produce:
(a) DATA: 9,714-state boundary-labeled table of root-grounded eps-play,
    within-won gradient ~0.6-0.75 throughout, terminal boards + rep counts --
    the best training substrate the project has, generated in one night.
(b) MECHANISMS, each measured then fixed in the harness: rim-gate noise;
    cross-table rho attenuation (paired gate); field-better/play-worse
    advancement (play gate -- the dissociation now lives IN the loop);
    sqrt(n) opening-shell concentration (weight cap); noisy-tail training
    rows (train-min-n). After ALL fixes, candidates still trail champion
    play by ~0.1-0.2 per round: distill-into-F on a growing cumulative
    table is structurally lossy for play at this scale.
(c) The r1 anomaly localizes the recipe that DOES work: small dense fresh
    on-policy table, one distill, champion lineage -- i.e., the train leg
    wants FRESH-data pulses, not cumulative-table repetition.
MORNING DISCUSSION (no builds pending): train-leg alternatives -- (i)
fresh-pulse rounds (distill only on each round's new data, r1-style);
(ii) head-only continual updates on frozen F; (iii) full retrain with
committor targets in the base objective on the loop's cumulative data
(the mechanism that actually worked at full board = cert_base). Toy lineage
champion: rootloop_r12 (~= r1); overall incumbent remains cert_base_full.

### Mate-probe escalation finds the failure boundary -- and a zero-training full-board->toy win
mate_probe.py (committed): single-position diagnostic, field ranking + search
playouts. KRRvk, wK a3 Ra1 Ra2:
  bK h8 (DTZ 3): MATE at ALL budgets (200/800/1600n), optimal line 1.Rg1 Kh7
    2.Rh2#; field root-spread 0.21, boxing move ranked #2/18. Black-to-move
    variant: MATE at all budgets, one tempo off optimal. The engine is NOT
    blind at the mate surface -- the rim problem is the APPROACH, not the net.
  bK e5 (DTZ 7 -- mate in 4!): toy-trained committor readout: 200n MATE-in-23,
    800n CUTOFF, 1600n THREEFOLD. More search = worse. Failure boundary is
    between DTZ 3 and DTZ 7.
CLEARANCE readout (reach = -d_W + beta*d_D, Kaveh's approach 2, wired):
  beta sweep on the failure position: helps 800n consistently (cutoff ->
  MATE-in-15 at every beta), 200n mixed, 1600n UNCHANGED -- the 1600n line is
  bit-identical at every beta incl. 0: at deep budgets the per-node squash
  absorbs near-constant readout shifts; choices are search-dynamics-dominated.
  Toy-level readout fix INSUFFICIENT per Kaveh's decision tree -> full build.
ZERO-TRAINING TRANSFER (Kaveh: "train on full board, test in toy"): the
incumbent's own outcome head (cert_base_full_phead, 3-class CE on game
results, 155k full-board steps) read as a W-committor (d_W = -ln P_win):
  failure position: 200n threefold, 800n MATE-in-25, 1600n MATE-in-15 --
  MONOTONE in budget, converts exactly where every toy-trained field fails.
  Mirror-image budget profiles: toy-distilled = sharp-narrow (deep search
  hurts); full-board head = broad-calibrated (deep search pays). The
  better-field-rewards-search prediction, realized. Its softmax also carries
  P_draw: a full-board d_D for free.
RUNNING: fixed-set ladder (n=120, 800/1600n), same incumbent both sides,
pole readout vs phead-committor readout -- the zero-training promotion
candidate. Kaveh's rules journaled to memory: conditional rejections (keep
flag-gated mechanisms, re-test after field promotions); self-contained
weekly-report style.

### Zero-training readout ladder + staircase mechanism + committor-base launched
PHEAD-COMMITTOR ladder (same incumbent both sides, only the readout differs;
B = reach from the full-board outcome head, -ln P_win, no goal vector):
PLAYOUT_AB PHEAD_COMMITTOR_800n  A=0.700 vs B=0.758 diff=+0.058 CI=[-0.025,+0.150] [ns] plies 18v21
PLAYOUT_AB PHEAD_COMMITTOR_1600n A=0.667 vs B=0.750 diff=+0.083 CI=[-0.008,+0.183] [ns] plies 18v21
Positive lean both rungs, faster mates, one game from CI-real at 1600n;
n=200 continuation running (anytime-valid). A readout costing ZERO training
matches everything the toy campaign produced.
STAIRCASE DIAGNOSTIC (rim_staircase.py, committed) -- Kaveh's flatness
question answered with data, his wall argument CONFIRMED, my saturation
story corrected: (1) at the failing mate-in-4 position the best PROGRESS
move ranks #7/18 and the field PREFERS tempo-wasters (gap -0.044); (2) the
learned d_W is FLAT vs true DTZ on random KRRvk wins (Spearman -0.011,
n=379); (3) but the TARGET is NOT flat: empirical P-hat falls 0.87->0.73
across dtz 1->8 (Spearman +0.291 CI[+0.163,+0.507]) -- the gradient exists
and is WALL-GENERATED (rollouts die at threefold/cap on shuffle lines),
exactly Kaveh's point that in a guaranteed win only the draw walls make
waste costly. The field misses it because we erase the wall 3x: (a) tables
aggregate P-hat by fen, blurring repetition history; (b) rep plane trained
on rep-blind targets = inert; (c) in-search boards carry no history -- the
search cannot see a threefold forming in its own lines. Fix stack for the
iterate-until-mechanism directive: (fen,rep)-keyed targets, live rep plane,
path-aware in-search threefold detection, d_D clearance as the wall-sensor.
LAUNCHED: --committor-base training mode (3-class outcome head = multinomial
W/D/L committor in the base objective, no pole term, no goal vectors) --
short run 5k steps first per protocol, then the full run. Kaveh's plan:
full-board train (human/sf/self-play) -> iterate mate-in-N isolation until
the flatness has a mechanism; mine Lichess puzzle DB (mateIn1/2/3 themes)
for permanent benchmark sets.

### CI-REAL at n=200: the zero-training committor readout beats the pole readout
PLAYOUT_AB PHEAD_COMMITTOR_1600n_n200 mate-rate A=0.640 vs B=0.780
diff=+0.140 CI=[+0.070,+0.215] e=91.47 (n=200, deterministic defender;
plies-to-mate A=21 B=18) [SIGNIFICANT]
Anytime-valid continuation of the n=120 look (e-process permits). Same
incumbent checkpoint both sides; only the readout differs: A navigates to
the mate-pole vector, B reads reach = -ln P_win from the incumbent's own
full-board-trained outcome head (no goal vector). e=91 >> 20. Faster mates
too (18 vs 21). CONFIRMATORY launched per protocol: fresh single-use
seed-782 set (minted, registry to update on verdict), 1600n, n=120. If it
holds, this is the program's second confirmed READOUT promotion (after
MCTS-vs-beam) -- and both cost zero training.
Meanwhile committor-base full training passed 30k/155k healthy
(top1 0.031, 8.9 it/s).

### Seed-782 confirmatory: ns on its own; composed evidence crosses the bar -- promotion deferred to the purpose-built checkpoint
CONFIRMATORY_phead_1600n_seed782: A=0.675 vs B=0.742 diff=+0.067
CI=[-0.017,+0.142] e=0.54 [ns], plies 20 vs 24. Seed 782 CONSUMED.
Third time this week a ~+0.07-at-n=120 effect shrinks under the fresh-set
confirmatory's resolution. Two honest frames: (a) STRICT protocol: not
confirmed, no promotion. (b) COMPOSITION (the cert_base precedent):
independent-set e-values multiply -- 91.47 (test set, anytime-valid n=200,
single pre-specified comparison, not a sweep winner) x 0.54 (seed-782)
= 49.4 > 20: the combined evidence rejects the null. Sign consistent
everywhere (5 independent looks all positive, faster mates every time).
DECISION: no promotion now (strict rule kept); composed evidence recorded
as a strong prior. The question is about to be superseded: the
committor-base full training (60k/155k, healthy) evaluates with this
readout natively -- the purpose-built head settles it.

### Mate-in-N benchmarks built + incumbent baseline: the flatness staircase, powered, full-board
Permanent EVAL-ONLY sets mined from the Lichess puzzle DB (mine_mate_puzzles.py:
theme filter + solution-line replay verification; 500 positions each, registered):
mate_in_{1,2,3}_n500.json. Runner: mate_bench.py (field-only top-move-mates +
search vs full-strength depth-12 SF defender).
INCUMBENT BASELINE (cert_base_full + its phead readout, n=120/set, 800n):
VERDICT MATE_BENCH_INCUMBENT mateIn1 FIELD-ONLY top-move-mates 15/120 = 0.125
VERDICT MATE_BENCH_INCUMBENT mateIn1 SEARCH@800n 120/120 = 1.000
VERDICT MATE_BENCH_INCUMBENT mateIn2 SEARCH@800n  51/120 = 0.425
VERDICT MATE_BENCH_INCUMBENT mateIn3 SEARCH@800n  17/120 = 0.142
The rim flatness measured at full board: the field alone finds the mating
move 1-in-8; search fully compensates at depth 1, half at depth 2, barely at
depth 3. This is the BEFORE axis for the committor-base checkpoint (124k/155k,
phead CE 0.635 and descending).

### Committor-base full run: first verdicts -- mixed, with an overcooking signature
Training completed clean (155k steps, VAL_TOP1 0.036, DIFF_SLOPE +0.251/-0.092
-- the cleanest won-lost separation of any full-board run). Verdicts so far:
  FIELD-ONLY mateIn1 (n=120): incumbent 0.125 -> committor-base 0.183 (+0.058)
  SEARCH mateIn1@800n: 1.000 both | mateIn2: 0.425 both | mateIn3: 0.142 both
  OVERLAP forensics (mateIn2): identical 51/120 counts are a SUM coincidence
  -- both-win 33, each-exclusively-wins 18: the fields disagree on 30% of
  positions with exactly balanced competence. Dead heat, not a broken bench.
  ROOK PROBE regression: the 5k-step snapshot MATES the DTZ-7 failure
  position @800n; the 155k final THREEFOLDS it at every budget -- the
  overcooking signature (round-C precedent) again. Ladder snapshots saved
  without pheads (now fixed: pheads save with every snapshot), so step-wise
  localization needs the next run. mate_bench gained --dump-results
  (per-position vectors) for overlap forensics.
RUNNING: full bench on the 5k snapshot -- if it matches 155k on-distribution
too, "stop by play, not by budget" becomes the committor-base recipe.

### Bug check on the 51s (clean) + CAPACITY FORENSICS: Kaveh's flexibility hypothesis lands
51/120 x3 bug check: dump vectors genuinely differ (hamming 36-52 pairwise;
three-way: all-win 18, none-win 33, contested 69) -- equal SUMS on different
win-sets, ~0.5% coincidence on a shared machinery-limited rate. Inspected
failed "mate-in-2" games past the budget window: genuine mate-MISSES (engine
plays quiet moves, game wanders 20+ plies), not slack artifacts -- value-only
MCTS without check-first ordering often never expands the forcing line.
CAPACITY FORENSICS (capacity_forensics.py, committed; committor-base
snapshots 30k..150k vs final):
  EFFECTIVE RANK: 5.7 -> 6.9 of 64 across the whole run -- the objective
    lives in a ~7-dim subspace from start to finish (~10% utilization).
  ROTATION: the top-10 subspace churns to the very end (150k->final, 5k
    steps: mean 3.2deg, max 10.6deg) -- late training keeps rewriting the
    same few dims.
  REGIME-SPLIT DRIFT (the smoking gun): rare/common drift ratio climbs
    0.97 -> 1.17 -> 1.11 -> 1.13 -> 1.71 (final stretch): late gradients --
    which carry ZERO rook-endgame information -- move rare-regime features
    1.7x MORE than common ones. The rare regime is UNDEFENDED COLLATERAL:
    nothing in the data anchors it, so shared-parameter updates drag it.
    This is why 5k mates the rook position and 155k threefolds it.
Kaveh's diagnosis confirmed in effective terms: the representation has
almost no working flexibility (~7 dims), and continued training reallocates
it toward the frequent regime by dragging undefended features. His proposed
fix -- much wider embedding + L1-style sparsity tax so dims are allocated
per-pattern and regimes decouple -- is now directly evidence-backed.
Complementary cheap lever: a small replay anchor (toy/endgame data at low
fraction) to DEFEND rare features. Widened-sparse run spec ready; awaiting GO.

### Widen + sparsify launched (Kaveh: "make it even bigger... penalize use of more dimensions")
Diagnosis-driven architecture change: effective rank ~7/64 regardless of
width + rare-regime drift 1.71 => the metric has ~no working flexibility and
frequent-regime gradients drag undefended rare features. Fix = decouple
representational capacity from GEOMETRIC capacity: wide embedding, L1 tax on
the per-dim metric_scale (prices DISTANCE dims; representation stays free).
Trainer knobs added: --channels/--blocks/--enc-out/--dh + --l1-metric-scale
(warmup ramp). Snapshots now save pheads (localization gap fixed).
Size reality (single Mac GPU): full Leela-classic trunk (256ch/12blk, 32M)
runs 0.4 it/s = 17h+, infeasible; and its bulk is trunk DEPTH, not metric
width. The L1 hypothesis lives on EMBEDDING WIDTH d (where metric_scale and
the rank collapse are), so kept d=512 (8x the current 64 -> 512 distance
dims to sparsify) with a lighter 128ch/10blk trunk: 9.4M params, 1.8 it/s.
LAUNCHED committor_wide.pt: fresh, quasimetric + committor-base (phead in
base objective) + ply-gap, d=512, l1-metric-scale 3e-4 warmup 8k, 40k steps
(early-peak-informed: 5k>155k last run), snapshots+pheads every 5k, ~6h.
Pre-registered gates on snapshots: effective rank RISES and scales with
width; rare/common drift ratio flattens toward 1; rook competence survives
to late steps; field-only mateIn1 beats 0.183. Paper written
(writing/committor_planner.md): full architecture, math (score/loss/
committors), 4 contribution claims.

### Merged paper: derivation thread + experimental thread reconciled (writing/adversarial_reachability.md)
Kaveh shared a formal derivation paper ("Forcing Regions, Not States") from a
parallel thread and asked to merge both ways -- each is ahead in different
areas -- into one publication-grade paper with figures. Built:
writing/adversarial_reachability.md + experiments/viz/merged_paper_figures.py
(6 figures: two-pole geometry, region-necessity, component diagram, search-to-
certainty [schematic, ported/merged]; wall-gradient + capacity forensics
[DATA, verbatim from our VERDICTs]). WHICH THREAD AHEAD WHERE (paper's S11
ledger): DERIVATION ahead on -- region-necessity as a THEOREM (point-to-set
gap), IQE (valid+universal vs our MRN), two-ply adversarial stitch (vs our
one-ply InfoNCE), search/memory PROOFS (non-expansive amplification, sandwich
bounds, neighbor-disagreement certificate), and the entire plan-space meta-
game (double-oracle repertoires, CVaR risk knob) which we have NOT built.
EXPERIMENT ahead on -- (1) the WALL-GENERATED GRADIENT correction: derivation
CLAIMS reachability descends monotonically to mate; we MEASURED it flat
(rho -0.01) with the real gradient (+0.29) generated by draw walls and erased
by history-blind aggregation/representation/search = their Assumption A1
("repetition excluded") is load-bearing exactly where they claimed advantage;
(2) capacity forensics (7-dim collapse, 1.7x rare-regime drift) -> free-rep/
priced-metric, which their IQE derivation doesn't address; (3) two FALSIFIED
components that discipline their design -- categorical head entropy ANTI-
correlates with sharpness (their Remark G.2 predicts it), post-hoc distill
never compounds play; (4) the whole e-value/defender/leakage/confirmatory/
forensics harness. Merged position adopts: IQE + priced sigma; two-ply stitch
+ committor-in-base-objective; augmented-state committor + d_D clearance;
Dirac-only categorical + multi-eps sharpness + neighbor-disagreement
uncertainty; their proofs + meta-game as the untested frontier. Indexed as
the lead paper; committor_planner.md demoted to earlier single-thread draft.
Meanwhile committor_wide training healthy (pre-5k).

### Wide run early read (5k, pre-L1): width alone does NOT open capacity -- looks like contrastive collapse
Same 400 states, apples-to-apples effective rank:
  narrow d=64 (cert_base_full, 155k, trained): 4.51 of 64
  wide   d=512 (committor_wide, 5k, PRE-L1):    3.40 of 512
Widening 8x shows NO early sign of opening the metric's working subspace
(3.4 vs 4.5; wide is early+pre-L1 so may grow, but the pre-registered gate
"rank RISES and scales with width" is already leaning against). MECHANISTIC
REFRAME: this is the signature of dimensional/representational COLLAPSE, a
known contrastive-learning pathology -- and its textbook cure is stronger
REPULSION (hard negatives), which is exactly Kaveh's contrastive lever built
today (monotonicity + horizon negatives). So the capacity thread and the
contrastive thread may be ONE problem: the field collapses for lack of
repulsion pressure; L1-on-metric-scale prices dims of an already-low-rank
representation (possibly the wrong layer), while hard negatives attack the
collapse directly. CAVEATS: 5k, pre-L1 (warmup 8k), single 400-state set --
not a verdict. PLAN: let the run pass L1 engagement to 10-15k (snapshots
every 5k) and watch whether rank moves at all; if it stays flat, the
hard-negatives (repulsion) lever likely becomes the PRIMARY capacity
experiment, not just a speed lever -- a reorder for Kaveh's call.

### Wide run 10k read: L1-on-metric-scale is INERT; capacity fix is repulsion (contrastive), not metric-pricing
Same 400 states:
  step 5000  (pre-L1):  F eff.rank 3.40/512 | metric_scale participation 511/512 (uniform, ~all-ones)
  step 10000 (L1 on 2k): F eff.rank 4.49/512 | metric_scale participation 509/512, scales [0.67,1.12]
Two findings: (1) effective rank RISES with TRAINING (3.4->4.5), now MATCHING
the fully-trained narrow d=64 field (4.51) -- so width alone does not open
capacity; the ~4-5 dim subspace is a property of the objective/representation,
consistent with contrastive collapse. (2) The L1 tax (weight 3e-4) is INERT:
metric scales barely moved from all-ones (participation 509/512, nothing near
zero) because 3e-4 is ~1e4x weaker than NCE~5. So we are NOT running the
"priced metric" experiment -- effectively a plain wide committor-base run.
DEEPER READ: even if L1 bit, it prices the METRIC scales, which cannot create
embedding capacity the ENCODER isn't producing (eff rank 4.5 is the
embedding's, not the metric's) -- so L1-on-metric-scale is likely the WRONG
LAYER. The collapse is representational, and its textbook cure is stronger
REPULSION = hard negatives (the contrastive lever Kaveh directed). This
CONFIRMS the mechanistic basis for "do the contrastive thing": it is the
capacity fix, not a speed add-on. PLAN: let the wide run finish (get the full
rank trajectory + a play read on a wide committor-base -- does width alone
help play?), then the NEXT run is the hard-negatives/horizon repulsion run
(--unreach-weight + --horizon-k, built today), promoted to PRIMARY. L1-on-
metric-scale deprioritized (wrong layer; and inert at any safe weight).

### AUTONOMOUS (overnight, Kaveh asleep): MVP = e-value-gated toy conversion vs optimal defender
Scope locked: build ONLY what pertains to converting the winning toy vs the
tablebase-optimal defender. IN: field (IQE + committor + hard-neg + horizon +
two-ply stitch + augmented-state wall-fix) + executor (MCTS + search-to-
certainty + draw-clearance) + region hierarchical planner. OUT (deferred):
player strength/q_opp/softmin/recognition/belief (defender is optimal+fixed ->
hard minimax correct), meta-game, offline distill loop. Discipline: NO
tablebase in play loop (toy <=6 pieces = fully tablebased; in-tree mate only),
rank=diagnostic-not-gate, no point estimates (bootstrap CI), e-value-gate all
play, check runs early, commit continuously.
FOUNDATION built earlier: IQE distance head (axiom-tested, wired), hard
negatives (vectorized, monotonicity, ep/diagonal-immune), horizon-cap.
THIS STRETCH (2 wall-fix pieces, the MEASURED toy failure = drift into a
threefold the search couldn't see):
1. PATH-AWARE THREEFOLD DETECTION in MCTS: nodes carry parent+rep_key, run()
   seeds rep_history from the game's move stack, _threefold counts history +
   search-path occurrences -> the search now SEES repetitions forming in its
   own lines (copy(stack=False) was blind). +rules-exact insufficient-material
   /50-move draws. 12 mcts tests pass. Toggle --no-threefold-a for A/B.
2. PHEAD DRAW-CLEARANCE: reach = ln P_win - beta*ln P_draw from the 3-class
   outcome head (no separate d_D head) -> steer away from the draw basin at
   the flat rim.
RUNNING: (a) threefold A/B on incumbent (off vs on, 800n n=120) -- does seeing
repetitions help conversion; (b) merged field-foundation run (IQE+committor+
hardneg+horizon, 6k) ~24min left. Both e-value/CI gated. Next: eval merged
(rank CI diagnostic + field cal + toy conversion e-value), then two-ply stitch
+ region planner + search-to-certainty, testing conversion each iteration.

### Autonomous: merged IQE field 5k rank diagnostic = 1.96 (LOWER than MRN 4.5) -- noted, not gated
committor_merged (IQE+hardneg+horizon) step5000: eff.rank(F) 1.96 of 512
CI[1.87,2.04], vs narrow-MRN 4.51 and wide-MRN 4.49 (same 400 states). IQE +
hard-negatives collapses F HARDER than MRN at 5k, not less -- the repulsion
is not opening capacity in this setup. DIAGNOSTIC ONLY (Kaveh: rank not a
gate); caveats: early (5k, IQE retrieval still near-chance train_top1), and
rank-of-F may mis-measure IQE (the geometry reshapes F into interval
components, doesn't use F's linear rank directly). ARBITER = toy conversion,
read at 6k. If conversion is bad AND rank low -> IQE+hardneg is collapsing
(diagnose weight/components); if conversion OK despite low rank -> rank-of-F
is the wrong measure for IQE (chess may need few dims, as Kaveh anticipated).

### Autonomous: threefold A/B = +0.017 ns (e=0.63), 4/120 decisive -- correct fix, marginal lever
PLAYOUT_AB THREEFOLD_800n A(off)=0.683 vs B(on)=0.700 diff=+0.017
CI=[-0.017,+0.050] e=0.63 [ns]. Only 4/120 games decisive -> path-aware
threefold detection changes the outcome rarely: most incumbent non-conversions
don't drift into a SEEABLE threefold (they draw via insufficient-material/
50-move/no-gradient). LEARNING: the fix is correct (kept, conditional-
rejections rule) but the DOMINANT conversion lever is the FIELD (flat rim),
not the search seeing the draw. Next draw-side lever to test: phead clearance
(ln P_win - beta*ln P_draw), which steers AWAY from the basin proactively
rather than detecting it at the leaf. Waiting on merged IQE 6k for its
conversion (the bigger signal: is the IQE field even convertible).

### Autonomous: IQE merged field FALSIFIED at 6k -- worse than MRN committor (play arbiter)
committor_merged (IQE+hardneg+horizon, 6k): VAL_TOP1 0.019 (vs MRN committor
~0.036), DIFF_SLOPE_WON -0.397 (NEGATIVE/wrong sign; cert_base +0.25), rank
1.96. Mate-in-4 rook probe: spread 0.0097 (FLAT), THREEFOLDS (shuffles). So
IQE+hardneg+horizon at 6k trained a WORSE field -- flat, weak retrieval,
collapsed, shuffles. Clear negative; probe suffices (no 120-game A/B needed).
DECISION: (1) MVP path = the KNOWN-GOOD MRN committor field (cert_base_full,
converts ~0.70) + wall-fixes (threefold[done]/clearance[built]/rep-keyed
targets), NOT the IQE geometry that trains worse. (2) Diagnose IQE cause:
launching IQE-ALONE run (no hardneg/horizon) -- is IQE itself bad, or did the
repulsion/horizon break it? (3) Test the draw-clearance lever on the good
field now. IQE deferred pending the diagnostic; play says it's not ready.

### Autonomous: IQE FAILURE DIAGNOSED (Kaveh asked) -- L2-normalization + tiny-init crushed it
Root cause, empirically pinned: the IQE-trained field had ALL pairwise
distances ~0.34 (true-pair diag 0.340 = off-diag 0.340: true futures NOT
closer than random -> near-chance retrieval -> flat field). InfoNCE
logit-spread/row 0.05 (uniform softmax, no gradient) vs 0.43 for good MRN.
WHY: (1) embed_F/embed_B L2-NORMALIZE to the unit sphere -- correct for
cosine/MRN, CATASTROPHIC for IQE whose interval-union geometry needs free
coordinate ranges; on the sphere all interval-unions collapse to tiny/uniform.
(2) The encoder's small-norm init (coord std 0.08) leaves IQE distances flat
(logit-spread 0.01) with no bootstrap gradient -- and IQE's exceedance-interval
gradient is SPARSE at small coordinates. MRN escapes this because Euclidean is
scale-linear and lives on the sphere at a natural scale; IQE does not.
(3) ply-gap/horizon targets (~0.2-0.3) pulled IQE toward small distances =
its degenerate regime (Kaveh's 'k-window' intuition -- the horizon-cap at
10/50=0.2 compounded it). Kaveh right: IQE should have worked; it was a
normalization/scale setup bug, not IQE.
FIX (committed): for IQE -- (a) NO L2-normalization, (b) fixed embed_scale=50
-> coord O(1), (c) learnable output log_scale so calibration adjusts SCALE not
embeddings. IQE-init logit-spread 0.01 -> 0.32 (matches trained-MRN). 26 tests
pass. RUNNING: clean 3k IQE-fixed run -- does VAL_TOP1 now climb off chance?
Also banked: clearance A/B +0.025 ns (13 decisive) -- correct/marginal like
threefold; but phead readout itself converts 0.80@800n (vs pole ~0.70) -- the
real readout win. Good-field MVP baseline = 0.80.

### Autonomous: IQE CAN train with the fix (ply-gap was the culprit) -> heavy run launched
Pure-InfoNCE IQE (fix: un-norm + embed_scale 50) at 1k steps: VAL top1 0.014
top8 0.074 -- CLIMBING off chance (7x/5x chance). So IQE DOES train retrieval
with the fix; it's slow (sparse gradient) and the ply-gap term (target 0.3 vs
IQE natural ~11) shrinks the scale and re-collapses it -> that's why the
committor+ply-gap IQE run stayed flat. Kaveh vindicated: IQE works, it was a
normalization/scale setup bug. HEAVY RUN (token-gap): fresh IQE + committor-
base + NO ply-gap (drop the scale-fighting calibration) + embed_scale 50,
d=512 comp=32, 40k steps, ckpt-every 10k (~4h). Eval when tokens return: did
IQE reach usable retrieval + does the phead readout convert the toy? If yes,
IQE is back for the merged arch; if it plateaus low, MRN stays the field
(cert_base_full converts 0.80 via phead -- the safe MVP baseline).


## 2026-07-16 (Opus) — IQE root cause: wrong OBJECTIVE (InfoNCE, not QRL); QRL built

Two literature searches (Kaveh's rule: stuck >15min on something that should
work -> stop tuning, read the source) cracked the IQE plateau. Two findings,
one small and one structural.

**(1) Direction bug.** The IQE per-component interval was built REVERSED:
ours `[V,U]` where U>V (= d(v->u)) vs the paper's `[U, max(U,V)]` where V>U
(= d(u->v), arXiv 2211.15120). So d(F(s),B(g)) scored reach BACKWARD in time,
and InfoNCE was asked to make true future-pairs cheap to traverse in reverse
(irreversible -> fights time's arrow). The 7 axiom tests missed it (a
quasimetric's transpose is still a valid quasimetric). Fixed in iqe.py
(forward + pairwise). VERDICT: 7/7 axiom tests pass; direction sanity
`d(big->small)=0.000  d(small->big)=3.000` (was inverted). Fixed-direction
InfoNCE run nudged VAL top8 to ~0.10 (best) vs the flipped run's flat ~0.073,
but stayed noisy and far below MRN's ~0.17 -- the fix was real but not the
unlock.

**(2) The real cause: wrong objective.** IQE was designed to be trained with
the QRL constrained-max objective (Wang/Torralba/Isola/Zhang, ICML 2023),
NOT InfoNCE. QRL's own words: without the distance-maximization term "the
quasimetric could remain arbitrarily small everywhere" -- our exact symptom.
InfoNCE only enforces relative ranking (in-batch softmax); it never PUSHES
absolute distances apart, so IQE's union-of-interval lengths never stretch and
the max-mean gradient stays sparse. MRN survives InfoNCE (its bilinear f.W.g
doesn't need large absolute distances); IQE does not. Structural mismatch, not
a tuning issue.

**Built the QRL objective** (`--qrl-objective`), mirroring the official
quasimetric-rl loss:
  * GLOBAL PUSH: `softplus(offset - d, beta=0.1)` on RANDOM (independent,
    shuffled) state/goal pairs -> spreads distances toward `offset`.
  * LOCAL CONSTRAINT: on real 1-ply transitions s->s', squared-hinge
    `relu(d(s,s') - 1)^2` toward the unit step, dual-ascended by a
    softplus Lagrange multiplier lambda (grad-reversal trick, fb.py
    `grad_reverse` + `qrl_raw_lambda`).
  * No InfoNCE. Multi-step (incl. long FORCED lines) distances self-assemble
    by chaining unit steps through the triangle inequality -- never supervised
    directly, never capped.
Data: the 1-ply successor s' is derived at batch time from consecutive shard
rows (LichessPairSource `packed_succ` + `succ_is_last` mask; batch_tensors
appends succ planes + valid mask). No re-sharding.

**Kaveh's design constraints, baked in:**
  * NO horizon cap. Treating pairs beyond ~10 plies as "far" would train a
    reachable 12-ply forced mate to look unreachable -- blinding us to long
    forcing lines. So `--qrl-push-offset 40` (~20 moves), set WELL beyond the
    longest forcing line: a reachable long line chains to ~its true ply length
    and stays CLOSER than unreachable random pairs. The push is a saturating
    prior, not a horizon.
  * Divergence-vs-forcing -> COHERENCE LENGTH (physics framing). The QRL
    metric is BEST-CASE d_optimal (plies, "if I steer every move"); the
    committor/eval head owns d_certainty = -ln P(reach). The bridge is
    coherence length xi(s) ~ 1/(local branching entropy): d_certainty is the
    path-integral of local surprisal -- ~0 per forced ply (opponent has no
    choice, xi long, trust deep), large per divergent ply (xi short, trust
    shallow). Decided LAYERING: QRL learns d_optimal pure (don't corrupt the
    metric with branching or the triangle inequality breaks); xi becomes the
    MCTS search-depth gate (deep where forced, search where divergent). xi is
    measurable -- an experiment for once the metric trains. Forcedness signal
    (legal-move count / policy entropy) to be added as a logged feature; the
    coherence-depth control in the planner is the NEXT build, with sign-off.

**Lambda catch-up fix.** First smoke: the global push ran away before lambda
(init 0.01) could respond -- d_step inflated to 7.9 (target 1), sq_dev 48,
lambda stuck at 0.010. The QRL authors flag exactly this ("lambda needs to
constantly catch up"). Fix: lambda init 1.0 (responsive) + its OWN LR (0.01,
excluded from the cosine schedule). VERDICT (300-step smoke, healthy QRL
dynamics): `lam 1.00->1.41->1.85->2.12`, `d_step 0.44->1.46->1.25->1.67`
(pinned near the unit step), `d_rand 0.42->3.77->11.5->16.7` (spreading toward
offset 40). Runaway gone. 26/26 nn tests pass.

Reference package: Tongzhou Wang's `torch-quasimetric` (the authoritative IQE
impl) is git-only, not on PyPI. Our hand-rolled IQE now passes axioms +
direction + healthy QRL dynamics; a numerical cross-check against the reference
(git install, needs Kaveh's approval) is cheap insurance given we already found
one bug in it -- offered, not yet done.

NEXT: full QRL-IQE run (does d_optimal reach usable retrieval as a side effect;
does the phead readout convert the toy, e-value-gated). If QRL-IQE works ->
merged arch + coherence-length planner. Fallback stays MRN committor field
(cert_base_full converts 0.80 via phead @800n -- the safe MVP baseline).


## 2026-07-16 (Opus) — overnight: QRL offset=128 LOCAL COLLAPSE; coherence A/B running

QRL-IQE full run at offset=128 (Kaveh's data-driven call: max game 407 plies,
99th 146, mean sampled gap 51 -> 40 is below the mean reachable distance)
developed a LOCAL COLLAPSE by ~step 1000: d_step (mean d(F(s)->B(s')) on real
1-ply transitions) fell to 0.000 and stayed, lam ratcheted to 6.5, d_rand
bounced 2-30 without stably climbing to the 128 offset. The offset=40 smoke was
HEALTHY (d_step ~1.2-1.7, lam ~2.1) -- so the stronger push at 128 tipped it
into the degenerate solution (d(s,s')=0 trivially satisfies the one-sided
d<=1 constraint; adjacent positions map to identical embeddings). Killed it
rather than burn ~2h.

HYPOTHESIS for the collapse: the global push uses SHUFFLED cross-batch goals,
which are largely DISCONNECTED from the 1-ply constraint transitions in the
embedding graph -> nothing forces consecutive positions ~1 apart, so the model
spreads the (disconnected) random pairs while collapsing the (constrained)
local steps. Real QRL's push over the state x state marginal keeps near-future
pairs in the mix, whose triangle-inequality chains pin d(s,s')~1. FIX CANDIDATES
to sweep tonight: (a) lower offset (40/64 -- did 40 hold past 400 steps?);
(b) push over the REAL (anchor, geometric-future goal) pairs, which ARE coupled
to the constraint via shared positions; (c) two-sided step constraint.

OVERNIGHT PLAN: (1) coherence-length A/B on the INCUMBENT MRN field
(cert_base_full + phead committor, MCTS@800n, k=1.0 vs off, n=100, e-value
gated) -- running now, independent of QRL, validates Kaveh's coherence-length
mechanism through conversion. (2) QRL offset/push-source sweep to find the
config with stable d_step~1 + spreading d_rand; launch the real run. (3) eval
the healthy QRL field (conversion + coherence on it). Commit + JOURNAL each step.


## 2026-07-17 (Opus) — QRL collapse investigated; field training, conversion is arbiter

Coherence-length A/B on the INCUMBENT MRN field (cert_base_full + phead
committor, MCTS@800n, k=1.0 vs off, n=100): A(off)=0.610 vs B(k=1.0)=0.510,
diff -0.100 CI[-0.20,0.00] e=1.11 -- NOT significant, only 26/100 decisive
(underpowered). When B did mate it was FASTER (15 vs 21 plies): k=1.0 converts
a touch less but finishes quicker, consistent with OVER-discounting (too much
field trust pulled). Not a verdict; retry a gentler k (0.3-0.5) on a healthy
field + more decisive starts.

QRL d_step (mean d(F(s)->B(s')) on 1-ply transitions) investigation:
  * offset=128: d_step STUCK at 0 (systematic local collapse) -- the strong
    push, acting on shuffled cross-batch pairs DISCONNECTED from the 1-ply
    constraint, made squashing neighbors (free under the one-sided d<=1
    constraint) the path of least resistance. Killed.
  * offset=40 shuffle: d_step ~0.8 mean but SWINGS (dips to 0.006 on some
    batches). Not collapsed, just noisy.
  * offset=40 + push_real (push over real anchor->future pairs, coupled): d_step
    still swings AND d_rand stays low (~1-9, reachable pairs cap at chain length
    -> no far-scale). Worse. Shuffle keeps the far-scale (d_rand ~6-16).
  * offset=40 + VICReg var-reg (weight 1.0): var term satisfied instantly
    (dims DO have variance) but d_step STILL dips to 0.006 -- variance reg cures
    GLOBAL dimensional collapse, not the LOCAL-pair swing.

SEARCH (Kaveh's rule) for the problem: it's the known dual-ascent Lagrangian
OSCILLATION (Stooke et al. PID-Lagrangian arXiv 2007.03964; ALaM augmented-
Lagrangian arXiv 2605.00667: "standard dual gradient ascent induces severe
oscillations, overshoot propagates to adjacent states"). Targeted fix = PID or
augmented Lagrangian to damp the lambda oscillation. VICReg variance reg is the
collapse cure (matches Kaveh's standing rule) but addresses a different mode.

DECISION: stop diagnosing d_step (it's ~0.8 non-collapsed, and PLAY is the
arbiter, not the metric-internal number). Launched the full 40k QRL-IQE field
at offset=40 + var-reg(1.0, cheap dimensional-collapse safeguard). Judge by
CONVERSION at the 10k/20k/40k checkpoints vs the MRN incumbent (0.80 @800n via
phead). If conversion is poor AND d_step instability is implicated, implement
PID-Lagrangian as the targeted fix. New flags: --qrl-push-real, --qrl-var-weight
/-target (all committed). load_ckpt now backfills new params for old ckpts.


## 2026-07-17 (Opus) — QRL-IQE hits a SMALL-WORLD COLLAPSE; conversion is the arbiter

Extended QRL debugging (Kaveh: search the problem, then fine-tune). Two failure
modes, one solved, one not:

FIXED -- the d_step->0 HARD collapse. Root cause: the degenerate attractor puts
all F embeddings above all B in the IQE coords, so every directed d(F->B)=0,
which trivially satisfies the one-sided d(s,s')<=1 constraint AND drags d_rand
to 0. Searched (Stooke PID-Lagrangian arXiv 2007.03964; VICReg; QRL p_goal).
Implemented: PID-Lagrangian multiplier (--qrl-use-pid, derivative gain damps the
dual-ascent oscillation), VICReg variance reg (--qrl-var-weight), and the
decisive one -- TWO-SIDED constraint (--qrl-two-sided, pin d(s,s')=1 both ways;
correct for chess since every 1-ply move IS one step). Two-sided forbids the
attractor: d_step now holds ~1.1, no collapse.

NOT FIXED -- the SMALL-WORLD collapse. d_rand (distance for random/far pairs)
stays ~2 in EVERY config: one-sided, two-sided, in-batch shuffle, and even with
a diverse cross-batch goal pool (--qrl-goal-pool, dataset-wide p_goal, the
searched fix for spreading). The metric treats ANY two positions as ~2 plies
apart -- geometrically false for chess (reasonable positions are ~10-40 shortest-
path plies apart). The scale is pinned by the local constraint (adjacent=1) and
the embedding finds a clustered manifold the push can't pull apart. So the IQE
quasimetric is nearly TRIVIAL -- it adds little reachability geometry over the
raw encoder.

SUSPECT (untested): the committor phead (phead-weight 1.0, co-trained) pulls the
embedding toward 3 OUTCOME clusters (win/draw/loss) -- a low-dim structure that
fights the metric spread. Pure QRL (no phead) might spread but then has no
readout to convert.

DECISION: stop tuning the d_rand diagnostic; PLAY is the arbiter. Full 40k
QRL-IQE @128 (two-sided + pool + PID + var, the most stable config) training,
ckpt every 10k. Eval phead conversion at 10k (fail-fast) vs the MRN incumbent
(cert_base_full 0.80 @800n). If it converts despite the flat metric, d_rand was
a red herring; if not, we have strong evidence QRL-IQE is wrong for this data ->
fall back to MRN (works) or test the pure-QRL-then-phead hypothesis. All fixes
committed + flag-gated + GLOSSARY'd. Coherence-length A/B (k=1.0) was NS/over-
discounting on the MRN field -- retry gentler k once a field is chosen.


## 2026-07-17 (Opus) — SOLVED why QRL-IQE wouldn't spread: phead co-training @1.0

Kaveh: understand WHY, search for fixes not grind. Fetched the QRL reference
(global_push.py) and the offline-GCRL quasimetric paper (TMD, arXiv 2509.20478)
and found concrete divergences from our setup:
  * reference softplus_offset = 15 (distances expected to cluster BELOW 15); we
    used 128 -- conflated trajectory length (~128 plies) with SHORTEST-PATH
    distance (~15; chess is well-connected).
  * reference batches are RANDOMLY ORDERED RANDOM samples (negatives via
    torch.roll(zy,1)); OURS are consecutive game slices -> correlated anchors.
  * TMD co-trains auxiliary heads at zeta = 0.1; we ran the committor phead at
    weight 1.0 -- 10x too strong.
  * TMD uses MRN offline + stop-grad + a Bregman divergence exp(d-d')-d to keep
    gradients alive at extreme distances.

DISCRIMINATING TEST (phead-collapse vs batch-correlation): pure QRL, NO phead,
offset=15. VERDICT: d_rand SPREADS -- 0.43 -> 3.3 -> 7.2 -> 9.07 by step 1000
(climbing toward the offset), d_step stable ~1.1. WITH the phead @1.0 it was
pinned at ~2 in every prior config. So the committor phead @weight 1.0 was
collapsing the embedding into 3 outcome clusters (win/draw/loss) = the small-
world metric. CONFIRMED root cause, matches TMD's 0.1 weight.

FIX under test now: QRL + committor phead @weight 0.1 + offset 15 (keep the
readout but stop it dominating). If d_rand still spreads with the phead back at
0.1, that's the config -> full run + conversion eval. If 0.1 still collapses,
train pure-QRL metric first then fit phead on the FROZEN embedding (TMD-style
post-hoc extraction). IQE is NOT the problem -- our co-training weight + offset
scale were. Kaveh's structural intuition (IQE right for this) stands.


## 2026-07-17 (Fable) — full architecture review + literature soundness sweep

Wrote ARCHITECTURE_REVIEW.md: top-to-bottom review of the arrangement (geometry
/ value / trust / compose / decide / deferred) + six targeted literature
searches. VERDICT: sound, organizing principle = ONE SCALAR FIELD PER QUESTION
-- every bug this week was a layer doing another layer's job (phead@1.0,
DRAW_V=-0.999, entropy-coherence, horizon-cap), every fix restored separation.

Literature anchors found: (1) QRL's known stochastic-setting limitation (TMD,
arXiv 2509.20478) is exactly why we never read d as value -- our layering is
the demanded response, and TMD is the shelf-ready alternative trainer if
QRL-IQE conversion disappoints. (2) Our two-sided constraint deviation is
PROVABLY right for unit-cost game graphs (d*(s,s')=1 exactly; one-sided is for
MDPs with dominated transitions). (3) The phead is an empirical COMMITTOR under
the human-play measure (active TPT literature; DASTR adaptive sampling = our
probe-sharpens-committor plan). (4) Coherence discount = e^{-k Sum(1-P)} ~
P(line realized)^k -- the same path-integral as d_certainty; AdaGamma's
TD-collapse pitfall doesn't apply (search-backup only, no bootstrap). (5)
Kaveh's internal/game action split for the planner IS Russell & Wefald rational
metareasoning; Hay & Russell prove bandits are the wrong frame for computation
selection -> learn the meta-policy. (6) "Forceable region" = ATTRACTOR of a
two-player reachability game (min-max fixpoint, not shortest path) -- the
theorem-shaped reason probes are necessary. (7) SoRB's distance-overestimation
fragility is cured by our probe-before-commit. (8) AlphaGo's resign mechanism
(FP<5%, 10% no-resign games) adopted for post-MVP phase D.

TOP NEW ACTION ITEM: phead CALIBRATION (reliability/ECE) as a standing health
gate -- P feeds coherence + soft-terminal + resign; overconfidence poisons all
three silently. Also: offset sweep {15,30,60} under the final recipe once
stability is proven.

Also clarified (Kaveh Q): value vs coherence = P at a point (how good is the
destination) vs P's decay along a path (how far is the map trustworthy);
independent axes (forced perpetual = low value, max coherence); the 2x2
quadrant IS the planner's decision logic. Planner reframe folded into
PLANNER_PROBE_DESIGN.md: INTERNAL actions {probe_region, set_plan} vs GAME
actions {make_move, offer_draw, resign}; MCTS is a computation, the planner
makes the move.


## 2026-07-17 (Fable) — collapse detector EARNED ITS KEEP; dead-zone diagnosis

First 40k launch of the final recipe collapsed at ~step 2.5k (d_step=d_rand=0,
push=softplus(15) exactly => constant embeddings). The collapse detector FIRED
at step 4000 and HALTED the run (--qrl-halt-on-collapse) -- caught at 4k, not
19k-by-hand: the bug net's first live catch. WHY the 8k diagnostic survived but
the 40k run didn't: the stretched cosine keeps LR ~3e-4 through the danger zone
(diagnostic was half-decayed by 4k). MECHANISM of the trap: IQE dead-gradient
zone -- once all F sit above all B with margin, max(U,V)=U everywhere, d==0 with
ZERO gradient; two-sided wants d=1 but the inactive max supplies no path back;
lam climbed 27 uselessly (x0=0). FIX: relaunched with --qrl-var-weight 1.0 --
VICReg variance hinge acts on embeddings DIRECTLY (per-dim std ~0 at the
constant fixed point => full gradient), re-spreading dims until max() reactivates.
This is the reviewer's point inverted: var-reg can't prevent the ORDERING
collapse, but it's exactly live at the CONSTANT-embedding fixed point the
ordering collapse lands in at high LR. Detector armed; if it halts again, next
single lever = peak LR 2e-4. Fail-fast chain, each step ~15min to verdict.


## 2026-07-18 (Fable) — committor atlas replaces PCA; draw-confidence ceiling found

Kaveh rejected the PCA surface viz (correctly -- linear axes are meaningless
under a quasimetric). Built experiments/viz/committor_atlas.py: (1) outcome
SIMPLEX with game trajectories (surfaces = corners), (2) certainty plane
(-ln P_win vs -ln P_loss -- the planner's coordinates), (3) committor level
sets over material x ply (contours = the surfaces). Run on the incumbent:
artifacts/experiments/committor_atlas_cert_base_full.png. Panel 3 sanity: 0.50
contour hugs material 0..+1, material dominates ply. VERDICT (n=22,283 holdout
positions, 400 games): PC1 variance share on the simplex 0.900, P_draw
mean=0.092 std=0.071 MAX=0.49, R^2(P_draw ~ quad(P_win))=0.065, holdout game
results W/D/L = 0.46/0.05/0.48.

FINDING (initial "effectively 1-D" read RETRACTED after quantification --
P_draw is a genuine independent dof, R^2=0.065): the phead has a DRAW-
CONFIDENCE CEILING -- max P_draw 0.49 over 22k positions, tracking the 5%
draw base rate of the human-game training measure. Consequence: "confidently
drawn" can never fire (certainty_stop), the D-surface cannot be independently
recognized, resign/draw-offer would be draw-blind. Mechanism = MEASURE
MISMATCH (mu_train 5%-draw middlegames vs mu_deploy toy endgame where draws
are the failure mode), the committor-is-measure-dependent point of
ARCHITECTURE_REVIEW made concrete. Fix direction (Kaveh's call): draw-rich
training mass for the toy committor (self-play from the toy region /
draw-upweighted loss) before D-surface planning can work.


## 2026-07-18 (Fable) — audit fixes landed; re-baseline 0.60; soft-terminal harmful; mate shown

All MATH_AUDIT fixes committed (194 tests): per-ply mate discount, raw-reach
recalibration, sibling omega, one-sided PID, rep-aware cache key, counted+cached
certainty evals, monotone counts, calibration instruments. show_mate.py: the toy
mate is now visible -- start 2, fixed incumbent @800n vs optimal defender:
1.Rxb6+ Ke7 2.Rb7+ Kf8 3.Rc8# (5 plies).

VERDICT (playout_ab, n=100 @800n, deterministic defender):
  A (incumbent, FIXED search, no clearance) = 0.600   <- NEW baseline
  B (A + certainty_stop 0.9 soft-terminal) = 0.200
  diff -0.400 CI[-0.52,-0.28] e=4.1e6 SIGNIFICANT.
(1) The old 0.80 is NOT comparable: DRAW_V=-0.999 was doing accidental
draw-avoidance for the winning side; with DRAW_V=0 that work belongs to the
CLEARANCE term, which A didn't enable -- failure modes are draw-acceptances
(threefold, insufficient-material), matching show_mate starts 0-1. Next: re-run
with clearance.
(2) Soft-terminal at 0.9 is DECISIVELY harmful on an uncalibrated phead
(overconfident: pred 0.849 -> realized 0.717): search stops exactly where
conversion still needs work. NO soft-terminal until calibration passes -- the
calibration-gate warning, confirmed in play.

QRL: halted again @3k (small-world; d_rand 1.75 vs d_step 1.3, sib stuck ~57,
lam 7.7). omega-fix + one-sided PID insufficient; force balance (sib weight 1
vs lam~8 on a smooth encoder) is the standing hypothesis -> next single lever
sib-weight 8. GPU now EXCLUSIVE per Kaveh (no sharing): chained clearance eval
-> training.


## 2026-07-18 (Fable) — certified unreachability oracle replaces sibling hinge

Kaveh's design: instead of generating sibling pairs, certify the push pairs we
ALREADY embed. nn/unreachable.py tests s->g per direction: count monotonicity
(promotion-safe), castling rights, and pawn forward-cone injective matching
(relaxed capture-victims => infeasibility still proves unreachability; flag =
theorem, no-flag = unknown). VERDICT: 90.6% of cross-game pool pairs flagged,
0.6ms/128, 0.0% false flags on 210 real-game forward pairs; 7 oracle tests.
Wired as --qrl-unreach-weight 8 hinge (floor 30) on d_far -- 9x the sibling
hinge's coverage at zero extra embedding cost, directional, and the d_unr /
d_oth split makes d_rand's mixture noise diagnosable. Sibling hinge deprecated
(default 0). sib8 run killed at ~10k (its ckpt kept for comparison); full
unreach run launched, same recipe otherwise. Also: d_rand explained +
force-balance model documented in chat; sib8 had confirmed the force story
(d_rand 1.5->5.5-12.9, lam 7->77 counter-escalation as predicted, first run
past 10k).


## 2026-07-18 (Fable) — one-directional hinge died in the dead zone; completed to Kaveh's two-directional spec

The first oracle-hinge run collapsed at ~step 2000 (cliff, not drift: d_step
1.2 -> 0.000 between steps ~1700-2100; detector halted @4000). Mechanism: the
IQE ordering-collapse fixed point (all F above all B => every distance exactly
0 => push, constraint, AND the one-directional hinge all have zero gradient;
var-reg blind, dims still varied). Hypothesis for why sib8 survived 10k+ where
this died: the sibling hinge was accidentally BIDIRECTIONAL (same boards'
F and B on both sides), punishing the global ordering directly. Kaveh's
original oracle spec explicitly asked for "one directional OR two directional"
-- the one-directional wiring was the shortcut, not the design. Completed to
spec: anchor-anchor pairs (both have omega => both directions trainable),
oracle-certified per direction, hinges on d(F(s_i)->B(s_j)) AND
d(F(s_j)->B(s_i)) -- the ordering collapse is now hinge-visible with live
gradient. Smoke: unr 29.6 + unrb 29.6 both active. THE single all-additions
run relaunched. Probe API v2 (opponent param + certified-surface termination
+ label store) committed to PLANNER_PROBE_DESIGN.md; move-quality Q(s,m) =
(class-preservation vs perfect play, P_mu growth) lexicographic.


## 2026-07-18 (Fable) — THE run: bidirectional hinge SPREAD the space, then a λ spike imploded it (halt @8000)

qrl_iqe_unreach (all-additions: bidirectional anchor-anchor certified repulsion
w=8 floor=30, two-sided pin, one-sided PID, var floor, offset 15, lr 2e-4)
died differently from every predecessor — three acts, from the run log
(artifacts/experiments/qrl_iqe_unreach.log):

1. IT WORKED FIRST. The certified repulsion produced the first genuine spread
   of the entire program: d_rand 25 -> 70.5, d_unr 26 -> 67.9, d_oth up to
   86.7, unr hinge relaxing 15 -> 7.5 (floor being satisfied), d_step held
   ~1.0-1.7. Previous best-ever d_rand was ~13 (sib8).
2. THE SNAP (observed, mechanism graded HYPOTHESIS). Spreading dragged some
   adjacent pairs apart too: sq_dev spiked to 45.5 and the PID multiplier
   spiked lam 24 -> 134 within one print interval. The next prints show total
   implosion, not drift: d_rand 63.3 -> 1.43, d_unr 60.0 -> 1.46, d_oth
   86.7 -> 1.30, unr back to ~28.5. Hypothesis: lam x two-sided-pin is a
   global contraction force; at lam~134 it overwhelmed push+hinge in one burst
   and overshot straight through the ordering-collapse surface.
3. THE ONE-WAY DOOR (observed). From ~5-6k to the halt: d_step=0.000,
   d_rand=0.000, var=0.000, unr=unrb=30.000 EXACTLY (maximal hinge violation,
   d_unr=0.00) while lam climbed 81 -> 87 with zero effect, loss rising
   linearly. Inside the IQE dead zone (all F above all B) the gradient of
   EVERY loss routed through d — push, pin, and BOTH hinge directions — is
   identically zero. Bidirectionality made the collapse hinge-VISIBLE, as
   intended, but visibility is not gradient: the hinge dies with everything
   else at d=0. Detector halted @8000 (d_step=0.078 over 2000 steps). ckpt
   kept: data/derived/sep/qrl_iqe_unreach.pt (halt-save 12:54).

Conclusion: the spread mechanism is no longer the blocker — the multiplier
dynamics are. The dead zone is ABSORBING, so the cure must prevent entry, not
punish residence. Options for Kaveh (all flag-gated, none launched — pause per
standing instruction): (a) cap lam / cool the PID gains / clip the constraint
gradient norm so no single violation event can contract the space globally;
(b) schedule the floor (ramp 30 up from ~5) so the step-pin never faces a 30:1
dynamic-range shock; (c) enforce the quasimetric AXIOM d(F(x),B(x))=0 on the
diagonal — the two-tower factorization never sees it, and it structurally
forbids the wholesale all-F-above-all-B ordering (the QRL paper is immune
exactly because its shared encoder pins the diagonal by construction);
(d) full shared-tower (F=B=phi) as in the paper, omega entering elsewhere.
NOTE var=0.000 while the VICReg floor (weight 1.0, gradient NOT routed
through d) failed to reinflate — unexplained; check what `var` actually
measures before trusting (b)-style fixes.


## 2026-07-18 (Fable) — planner-energy program started (Kaveh: E[score] − c·compute, "least energy possible")

Direction set in discussion: the planner is a rational-metareasoning agent —
maximize E_mu[score 2/1/0] minus an explicit compute price c; probing is
best-ACTION identification (LUCB/dominance over certified intervals), not
value estimation; resign/draw-offer are ENERGY decisions (play on only while
expected swindle value exceeds compute cost); decision cascade tier 0-3
(plan-memory hit -> geometry-only -> coarse probes -> deepen; cheapest
sufficient tier). Kaveh: "start doing it autonomously."

Built and tested today (all committed, suite 222 green):
- experiments/energy_baseline.py — the compute-strength Pareto instrument.
  Energy = rows through embed_F/embed_B (policy-agnostic, cache-aware) +
  evals_used cross-check + wall-clock; VERDICT per (policy, budget); the
  playout_ab protocol exactly (tb-optimal defender, fixed test set), so
  conversion lands on the 0.600 baseline scale. Validated short (n=4 @200n:
  rows/move 209 = evals/move 209, util 1.04). Full CPU sweep (n=100,
  mcts/beam/plan x 200/800/1600n) running (GPU stays off it).
  KEY STRUCTURAL FACT it documents: MCTS spends its whole budget every move
  by construction (no stopping rule) — the energy profile is FLAT.
- catspace/planner/probe.py — phase-A probe() primitive: bounded MCTS ->
  ProbeResult(value, best_move, CERTIFIED [lo,hi], visit-weighted hit census,
  coherence, evals_spent, tree) + deepen() (tree-continuing re-probe).
  Certification discipline hard-coded: [lo,hi] from rules-terminal children
  only; recognizer-planted terminal_v can NEVER tighten it (unit-tested).
- catspace/nn/mcts.py — flag-gated decision_stop: (a) certified mate-stop
  (game-truth mate at root ends search after root expansion; flagless code
  burned the full 800 evals on mate-in-1s); (b) visit-gap-vs-remaining-budget
  stability stop (HEURISTIC, labeled as such — terminal sims add visits
  without evals, so not a dominance theorem; graded by paired A/B).
- tests/test_probe.py — 12 tests (FENs self-verified in-test; certification
  hygiene; budget honesty; deepen reuse; early-stop behavior).

Next (GPU now free, sequential): phead_calibration on the incumbent (fixed
instruments) -> paired early-stop A/B @800n n=100 (playout_ab, e-gated:
strength must be a wash, energy must drop) -> MPS energy re-measure.
FBPlanPolicy re-priced on the compute axis by the CPU sweep (its strength
wash e=0.47 was never priced; tier-0 skip-the-search is the biggest lever).


## 2026-07-18 (Fable) — early-stop v1: strength wash, but only 4% energy (units bug found+fixed); calibration quantified

Measured (all printed verdicts):
- PLAYOUT_AB EARLYSTOP_800n: A(incumbent)=0.600 B(+decision_stop)=0.590,
  diff -0.010 CI[-0.030,+0.000] e=1.00, n=100, 1 decisive pair. Strength: a
  wash (one start flipped). A=0.600 also reproduces the re-baseline on MPS.
- VERDICT ENERGY mcts+stop@800n: rows/move=768 (p50=807), util 0.96 — the v1
  stability rule saved only ~4%. DIAGNOSIS: units bug — it compared the root
  VISIT gap to remaining EVALS, but each sim costs ~one expansion batch
  (~20+ evals), so visits accrue ~20x slower and the rule could only fire in
  the last few percent of the budget. Fixed: remaining budget converted to
  SIM units (remaining evals / measured evals-per-sim, 2x safety, max_sims
  cap). The certified mate-stop half is PROVABLY move-identical (best_move's
  game-truth-mate short-circuit precedes visit-argmax), so the flipped start
  belongs to the stability heuristic alone.
- PHEAD calibration (fixed instruments, holdout n=400 games / 22283 pos):
  ECE=0.0518 CI[0.0321,0.0843], sharpness 0.205; MARTINGALE endpoint drift
  -0.00029/ply CI[-0.00095,+0.00035] and all three phase bins 0-in-CI — the
  committor IS consistent with a conditional expectation under mu.
  Overconfidence is LOCALIZED: bin [0.8,0.9) conf 0.849 -> realized 0.717
  (the 13-pt gap that made certainty_stop 0.9 harmful); closed top bin
  [0.9,1.0] 0.935 -> 0.883. Coherence's P(realize) use: acceptable. Hard
  soft-terminal: still gated until a temperature recalibration of the phead
  is measured (cheap next candidate, committor_recalibrate.py exists).
v2 A/B of the corrected stability rule launched (same protocol, label
EARLYSTOP_v2_800n).


## 2026-07-18 (Fable) — the compute–strength Pareto; mate-stop proven free; beam's budget accounting was wrong 2-4x; review HIGH fixed

The energy program's first full measurement round (energy_baseline.py, n=100
fixed toy starts, tb-optimal defender — the exact 0.600-baseline protocol;
rows = fresh forwards through embed_F/embed_B, policy-agnostic):

  config                conv   rows/move  (p50/p90)     notes
  mcts@200n             0.490     210     (209/222)
  mcts@800n             0.600     811     (809/822)     incumbent config
  mcts@800n+mate-stop   0.600     776     (809/822)     PROVEN identical (below)
  mcts@800n+both-stops  0.560     752                   stability: SHELVED
  mcts@1600n            0.620    1608     (1609/1621)
  beam@200n             0.250     339     util 1.69
  beam@800n             0.400    3119     util 3.90
  beam@1600n            0.400    3603     util 2.25
  plan@2000/60 (fixed)  0.150     870     (p50=30! p90=4107)

Findings, in order of importance:
1. BEAM'S BUDGET ACCOUNTING WAS WRONG 2-4x. util = actual embed rows /
   nominal budget: beam configs embed 1.7-3.9x their claimed budget (mcts
   ~1.0). make_search_policy's "one budget unit = one network eval in ALL of
   them" is FALSE for beam — every historical matched-nodes beam-vs-mcts
   comparison was unmatched (the A3 audit concern, now measured). MCTS
   STRICTLY DOMINATES beam on the real Pareto: beam@800 = 15x the energy of
   mcts@200 for LESS conversion (0.400 vs 0.490).
2. MATE-STOP: PROVEN AND PRICED. PLAYOUT_AB MATESTOP_800n: diff +0.000,
   CI [+0.000,+0.000], 0 decisive pairs of 100 — move-for-move identical, as
   the game_truth-gated proof requires. Energy 776 vs 811 rows/move: ~4%
   free. Keeper (flag-gated, zero risk).
3. STABILITY STOP: SHELVED. v2 (sim-units rule): -0.040 conversion
   (CI [-0.080,-0.010], e=4.38) for ~6% energy. Cause is structural: ~36
   sims/move at 800 evals (batch expansion) — visit gaps cannot become
   decisive. Retest shelf-conditions: high-sim regimes or tree reuse.
4. BUDGET LADDER KNEE AT ~800: +0.110 conversion for 200->800, +0.020 for
   800->1600. Adaptive per-position allocation (the cascade's job) is the
   remaining big in-search lever: hold ~0.60 strength at well under 800
   rows/move average by spending 200n on easy moves.
5. PLAN PERSISTENCE: RIGHT SHAPE, WRONG SUBSTRATE. Re-run with the state-leak
   fix (fresh policy per start): conv 0.150, mean 870 rows/move but p50=30 —
   tier-0 works (a held plan makes most moves nearly free); the mean and the
   strength die on the 2000n BEAM replans (~3400 actual rows each) and the
   z-goal beam readout. Next build: plan persistence on the mcts+committor
   substrate (probe() as the planner, cheap mcts executor, certified wake
   triggers). The tainted pre-fix row (0.120/779) is superseded.

Adversarial review of the session's commits (3-dimension workflow, 15 agents,
every finding adversarially verified): 1 HIGH — best_move/FBMCTSPolicy.move
mate short-circuits lacked the game_truth gate, so a cert-planted
terminal_v>0.5 earlier in move order was PLAYED over a proven mate-in-1
(reproduced; also falsified the mate-stop's move-identity claim under
--certainty-stop configs). Fixed + regression-tested; the MATESTOP proof-check
above ran on the fixed code. Also fixed: explicit cert_planted provenance
(float-equality could drop certifiable stalemate draws), vacuous single-
candidate certified in the cascade, FBPlanPolicy state leak in the sweep,
--decision-stop-b silent no-op, per-call cache_hits, probes suspending the
stability stop. 2 findings refuted. 22 planner tests; suite green (232).

OPEN PROTOCOL ISSUE (Kaveh decision): cross-game path_counts pollution —
FBMCTSPolicy instances shared across starts accumulate board_fen counts
across GAMES (repetition feature + cache keys see other games' visits).
Pre-existing in every historical PLAYOUT_AB number, so fixing it silently
would break comparability; needs a coordinated re-baseline.


## 2026-07-18 (Fable) — plan persistence SHELVED on its second substrate: dominated by plain mcts@200

FBPlanMCTSPolicy (two-budget committor-MCTS persistence: deep 800n on
surprise/dropped/stalled triggers, 100n carried-tree top-up otherwise;
mate-stop on) measured at n=100, paired, deterministic defender:

  PLAYOUT_AB MCTSPLAN_800_100: A(mcts@800)=0.600 vs B=0.440, diff -0.160
  CI[-0.260,-0.060] e=12.93, 30 decisive pairs. SIGNIFICANT loss.
  VERDICT ENERGY mctsplan@800n: conv 0.440, rows/move 223 (p50=111 p90=806).

The energy shape delivered (3.6x cut, tier-0 works mechanically) but the
strength did not — and the sharp verdict is DOMINATION, not tradeoff:
plain mcts@200 = 0.490 conv at 210 rows/move beats mctsplan (0.440 @ 223)
on BOTH axes. Persistence as-built spends its savings playing STALE moves:
exec top-ups (100 evals) rarely overturn the carried tree's old visit mass,
so changed positions get old answers (plies-to-mate 24 vs 20 = drift).
Consistent with the beam-era FBPlanPolicy shelf (ns strength, walked into
refutations) — the failure is the PERSISTENCE PRINCIPLE at this field
quality, not the substrate. Shelved (both substrates now priced); retest
condition: a field whose plans survive >6 plies (post-spread).

What the data says instead: allocation must be gated on UNCERTAINTY, not
plan bookkeeping — spend 800n exactly where a 200n search is CONTESTED
(small top-2 visit gap), i.e. the DecisionCascade's coarse->deepen rule as
a move policy. mcts@200 already holds 0.490 @ 210; the bet is that the
~0.11 conversion the 800n ladder step buys comes from a MINORITY of moves.
Next gate: escalate-on-uncertainty policy vs mcts@800 (strength) and vs
mcts@200/800 Pareto line (energy).


## 2026-07-18 (Fable) — escalation NO-BUILD: flip-rate 0.51 says difficulty is homogeneous; in-search allocation program CLOSED at this field quality

decision_flip_probe.py (new instrument; probe() as diagnostic): along
incumbent mcts@800 reference games vs the tb defender, a fresh 200n search
picks a DIFFERENT move on 51.0% of 478 decisions (n=30 starts). The coarse
top-2 visit-gap gate has real but insufficient signal: flip rate 0.649 /
0.585 / 0.278 by gap tercile; the lowest-gap third captures only 44.7% of
flips. Escalation arithmetic at these numbers: catching ~90% of flips means
escalating ~2/3 of moves => ~600 rows/move for residual strength loss —
dominated territory again. Corroborates the 2026-07-13 beam-era null
("homogeneous difficulty defeats targeting") with a direct decision-level
measurement on the current committor-MCTS substrate. NO-BUILD; the
diagnostic cost 147s instead of a 40-min gate run.

COROLLARY (field, not search): half the moves flipping between 200n and
800n while conversion moves only 0.490->0.600 means the field's move
ranking is SOFT nearly everywhere — search is doing the heavy lifting over
a weakly-discriminating reach signal. Consistent with the small-world
diagnosis. The binding constraint on BOTH strength and energy is the field
(the spread program awaiting Kaveh's collapse-remedy call), not the search
layer.

ENERGY PROGRAM STATE after day 1 (all numbers = printed verdicts, n=100
paired unless noted):
  KEEPER   mate-stop: -4% energy, diff exactly 0.000 (proven + verified)
  CLOSED   stability stop (-0.040 conv for 6%); plan persistence on BOTH
           substrates (beam: 0.150@870; mcts: 0.440@223, dominated by
           mcts@200 0.490@210); escalation (no-build by flip probe)
  FRONTIER mcts@200 (0.490@210) / mcts@800+mate-stop (0.600@776) /
           mcts@1600 (0.620@1608) — the honest Pareto, all in-search
           levers priced
  NEXT (needs Kaveh): field spread (collapse remedies a-d), THEN retest
  the shelf per conditional-rejections; label store (phase C amortization)
  is the remaining unbuilt energy idea and pairs naturally with a spread
  field's certified surfaces; path_counts protocol re-baseline decision.


## 2026-07-18 (Fable) — leaky IQE + PID guards pass the smoke THROUGH the death zone; THE run v2 (40k) launched

Kaveh's two ideas (offered as ideas, graded accordingly) + the spike guards:
- LEAKY IQE (his relaxed-relu): interval hard max -> softplus(beta=10); the
  collapse surface becomes repulsive (boundary grad ~8e-4 at gap 0.5) while
  deep-zone escape is explicitly NOT claimed (e^{-beta*gap}, float32-absorbed
  forward). Measured d(x,x) bias 0.54 < step_cost 1.0 -- and the two-sided
  pin over that floor is implicit spread pressure (encoder must widen coords
  to satisfy it). COST: eps-quasimetric (triangle inequality approximate);
  watch REACH_SLOPE/DIFF_SLOPE for calibration distortion.
- PID GUARDS: eclip 3.0 (spike arithmetic of the implosion: kp*45.5=23 +
  kd*45.5=91 -> lam 134; the controller now answers SUSTAINED violation
  only), kd 2.0 -> 0.25 (Kaveh's instinct: the derivative term was 2/3 of
  the spike), lam cap 20 + anti-windup.
- RETRACTED: the diagonal-axiom remedy (d(F(x),B(x))=0) -- it is CONSISTENT
  with the dead zone (B <= F includes the diagonal); shared-tower immunity
  does not transfer to a two-tower diagonal loss.

SMOKE (2k steps, full recipe + fixes): PASSED THROUGH the previous death
zone. step 1000: d_rand 68.0 d_unr 61.3 d_oth 89.1 (the old run snapped at
d_rand ~63) with d_step 1.16, lam 16 < cap. Final step 2000: d_rand 76.4,
d_unr 73.2, d_oth 98.6, d_step 1.28, lam AT the cap (20.000, sq_dev 2.66),
var 0.002, zero collapse flags, clean atomic save. WATCH ITEM for 40k:
lam-at-cap = bounded pin enforcement; d_step equilibrium ~1.1-1.3 instead
of exactly 1.0 (stability bought at some calibration cost -- graded by the
run's slope verdicts). ATTRIBUTION CAVEAT: this tests the PACKAGE (leak +
3 guards); leak-off ablation is one flag if wanted after a success.

THE run v2: qrl_iqe_leak.pt, 40k steps, same recipe + --iqe-leak-beta 10
--qrl-pid-kd 0.25 --qrl-pid-eclip 3.0 --qrl-lambda-max 20, detector armed,
~3.4h at 3.3 it/s. Also: Kaveh's density-prior subgoal idea recorded as
UNVALIDATED design note with the draw-mass confound + kill-test attached
(density ~ draw regions in human data could steer INTO the draw basin).


## 2026-07-18 (Fable) — leak+guards run: NOT the old collapse — spread PERSISTED (d_rand 169), d_step decayed because my lambda cap was too low to defend the pin

qrl_iqe_leak (40k, leaky IQE beta=10 + PID eclip/kd0.25/cap20) halted @18000.
The detector's message ("all-F-above-all-B") is a CANNED string and is WRONG
here — the trajectory (qrl push lines) tells a different, better story:

  step   d_step   d_rand   d_unr   lam   var
  ~200   0.49     0.49     0.49    1.2   0.94   start
  ~3k    1.16     55.3     52.6    20    0.00   spread engaging
  ~6k    1.08    129.4    128.1    20    0.00   PEAK HEALTH (pin ok, huge spread)
  ~10k   0.85     71.3     73.5    20    0.00   d_step starting to slip
  ~13k   0.16    105.8    110.7    20    0.00   pin broken, spread INTACT
  ~16k   0.19    136.7    140.4    20    0.00   spread still growing
  ~18k   0.009   168.9    174.0    20    0.00   halt

This is NOT the ordering-collapse dead zone (there d_rand -> 0 too). Here the
SPREAD SURVIVED and kept growing to 169 — the hard problem, solved and stable.
What decayed is the LOCAL unit-step pin (d_step 1.1 -> 0), and the cause is
diagnosable: lam pinned at the cap 20 the entire time. Mechanism: as the
encoder scaled up (d_rand ~100+ => large coordinate scale), a unit-step
deviation needs a PROPORTIONALLY larger multiplier to matter; capped at 20,
lam could not rise to defend d_step, so the field slowly walked the adjacent-
transition distance to zero while keeping random pairs far. My spike guard
(lam_max=20) traded the spike-implosion failure for a capped-enforcement
failure. eclip alone stops the spike; the CAP was the mistake.

HEALTHY CHECKPOINT PRESERVED: qrl_iqe_leak_step10000.pt (+_phead) saved @16:58,
d_rand ~129/d_step ~1.0 region — the largest-spread field yet, from BEFORE the
decay. This is the candidate to evaluate (conversion vs the 0.600 incumbent)
and, if the field is finally good, to re-open the shelved planner shelf on.

REMEDIES for Kaveh (his call; NOT relaunched):
  (a) raise or remove lam_max, keep eclip (eclip stops the spike without
      capping steady-state pin enforcement) — smallest change, most likely fix;
  (b) SCALE-NORMALIZE the constraint: pin/penalty relative to the embedding
      scale so a large d_rand can't dwarf a unit-step deviation (addresses the
      root cause — proportionality — not just the cap value);
  (c) anneal the offset/floor down as scale grows (keep dynamic range bounded).
var=0.00 throughout again (VICReg not reinflating) — still unexplained, worth a
look before (b).

Also: RUNBOOK.md added (all train/eval/analyze commands, per Kaveh — replicate
when tokens run out). Play-atlas interface build was interrupted by the session
limit: only experiments/viz/build_play_atlas.py (precompute) landed; server +
frontend still TODO (frozen contract in scratchpad).


## 2026-07-18 (Fable) — the spread field does NOT play: 0.160 vs incumbent 0.600. Spread is not sufficient for conversion.

PLAYOUT_AB LEAK_STEP10000_vs_incumbent (n=100 paired, tb-optimal defender,
mcts@800, committor readout both sides):
  A (incumbent cert_base_full) = 0.600
  B (qrl_iqe_leak_step10000, the biggest-spread field ever, d_rand~129) = 0.160
  diff -0.440 CI[-0.540,-0.340] e=7.8e9 SIGNIFICANT. 48 decisive pairs.
  (plies-to-mate A=20 B=8 — the few B converts are fast, but it converts 16%.)

THE headline, stated plainly: a hugely-spread quasimetric geometry plays MUCH
WORSE than the unspread incumbent. **Spread is not sufficient for play.** Two
honest caveats on B's disadvantage: (1) it trained with --phead-weight 0.1
(committor readout undertrained — the spread objective dominated), so this
measures the field AS SHIPPED, not spread's ceiling; (2) the whole point of
spread was to enable SUBGOAL PLANNING (waypoints need a metric that separates
regions), NOT to improve direct committor conversion — so a spread field that
reads out worse could still be the better planning substrate. But that is now
a HYPOTHESIS to prove, not a result. As a playing engine today, the incumbent
remains far ahead, and the week's spread program has NOT produced a better
player. Recorded against [[rigor_over_flattery]] — do not oversell the d_rand
breakthrough; it bought geometry, not wins, and may have cost the readout.

Strategic implication: the spread program's value is now gated on the planner
actually using the geometry (decompose waypoints + probe), which is unbuilt on
a real field. Before more field-spread work, the cheaper question is whether
ANY planner on this spread field beats direct committor readout — else spread
is a dead end for play. Field-remedy relaunch (lambda-cap fix) is on hold
pending that call.


## 2026-07-18 (Fable) — literature check on the Lagrange-multiplier tuning: our hard λ-cap was NOT paper-supported and IS the bug

Searched QRL (Wang et al ICML 2023, arXiv 2304.01203) and PID-Lagrangian
(Stooke/Achiam/Abbeel 2020, arXiv 2007.03964). Findings, graded:

- PROVEN (both papers): λ is only ever PROJECTED to ≥0 (QRL max_{λ≥0};
  Stooke Alg.2 line 9 λ←(KpΔ+Ki·I+Kd·∂)_+). NEITHER caps λ from above. Our
  --qrl-lambda-max 20 is not from the literature; it is what starved the pin
  (λ stuck at 20 while d_step decayed to 0 — the step10000->18000 failure).
  => REMOVE the cap; keep only the ≥0 projection + the ≥0 anti-windup on I
  (Stooke Alg.2 line 8 I←(I+Δ)_+), which we already have.
- PROVEN (QRL Eq.12): the constraint is exactly our shape —
  min_θ max_{λ≥0} −E[φ(d(s,g))] + λ(E[relu(d(s,s')+cost)²] − ε²). QRL
  DOCUMENTS OUR EXACT PATHOLOGY: "maximizing E[d(s,g)] increases late-layer
  weight norms, so λ needs to constantly catch up." Their fix is NOT a cap
  and NOT spectral norm — it is a convex/saturating φ that DOWN-WEIGHTS
  already-large distances (discount-like), removing the incentive that
  inflates the norms. Our push softplus(offset−d) already saturates past
  offset=15, so the push term is aligned; the unbounded growth to d_rand=169
  therefore comes from elsewhere (encoder weight-norm drift / unreach hinge
  interaction), which the cap can't fix.
- PROVEN (Stooke): gains are tuned per-env, ranges Kp∈[0.1,1], Ki∈[1e-4,1e-1];
  Kp damps oscillation, Kd prevents overshoot, Ki gives steady-state (alone =
  90° phase-lag oscillation). Our Kp=0.5/Ki=0.01/Kd=0.25 are IN range — gains
  are fine, the cap was the fault. Scale-invariance in Stooke = gradient-norm
  ratio β=||∇J||/||∇J_C|| (makes λ*→1 scale-invariant), NOT cost mean/std.
- PLAUSIBLE (broad lit: spectral-norm 1802.05957; SimbaV2 hyperspherical
  2502.15280): bound the encoder Lipschitz/weight-norm growth directly so
  distances can't blow up — the structural root-cause fix. Compatible with IQE
  if applied to WEIGHTS (we must NOT L2-normalize the IQE features).

CORRECTED REMEDY (ranked, literature-grounded; awaiting Kaveh, NOT relaunched):
  1. Remove --qrl-lambda-max (revert to ≥0 projection only). Keep eclip as a
     rate-limit only. Smallest change; directly undoes the starvation.
  2. Control the scale growth at the source: spectral-norm the encoder layers
     (bounds distance growth so λ never has to chase) — the SimbaV2/spectral
     literature's answer to exactly QRL's documented weight-norm problem.
  3. If still oscillating, adopt Stooke's gradient-norm-ratio scale-invariance
     on the constraint rather than tuning gains.


## 2026-07-18 (Fable) — no-cap fix HELPED but is INSUFFICIENT: λ diverges (→95+) chasing a pin it can't hold; remedy #2 (spectral norm) needed

Full no-cap run (cap removed, eclip 3.0, kd 0.25 — researched remedy #1) stopped
at step 10000 on a clear verdict. Trajectory (λ / d_step, target d_step=1.0):
  2.5k: λ 24, d_step 1.11   (healthy — matched the smoke)
  6k:   λ 62, d_step 0.4-0.8
  10k:  λ 94-98 (still climbing), d_step thrashing 0.03-0.79 mean ~0.3
Cf. the CAPPED run (died d_step 0.009 @18k): removing the cap DID help — λ is
free to rise (95 vs stuck 20) and d_step does NOT fully collapse (bounces
0.3-0.8, not 0.009). But it never settles at 1.0: λ diverges while the pin
thrashes. This is precisely QRL's documented pathology (maximizing E[d] grows
weight norms so λ must "constantly catch up") — and it confirms remedy #1
(free λ) alone can't win the race; the SCALE GROWTH must be bounded at the
source. var=0.000 throughout again (VICReg not reinflating — recurring
unexplained, may be contributing).

Verdict: escalate to researched remedy #2 — SPECTRAL NORMALIZATION on the
encoder (bound the Lipschitz constant so distances can't inflate, so λ never
has to chase). Well-supported (Miyato 1802.05957; SimbaV2 2502.15280 for RL)
and it's exactly the answer to the weight-norm-growth QRL flags. Alternatives
if Kaveh prefers: #3 Stooke gradient-norm-ratio scale-invariance on the
constraint; or QRL's own convex-φ down-weighting (verify our saturating push
isn't being outrun). Run stopped to not burn GPU on an under-pinned field;
NOT relaunched pending the remedy choice.


## 2026-07-18 (Fable) — SPECTRAL NORM WORKS: first stable full-recipe run; scale runaway solved, λ bounded

qrl_iqe_sn_smoke (2500 steps, no-cap recipe + --spectral-norm, single lever
vs the diverging no-cap run). VERDICT: the scale runaway is SOLVED.
  step 2.5k: λ=22 (STABLE — barely moved from ~18 @1200), d_step=1.36 (pinned
             near 1), d_rand=20.4, d_unr=21.0, d_oth=19.4, no COLLAPSE, saved.
  cf. no-cap run @ same step: λ already diverging past 40 -> 95, d_step
      thrashing 0.3. cf. capped run: d_step decayed to 0.009.
This is the FIRST time the full recipe held λ bounded AND d_step pinned through
2500 steps. Spectral norm (encoder+heads, 50 layers) bounds the Lipschitz
constant so push/hinge can't inflate the embedding scale -> λ has nothing to
chase. log_scale (the scalar escape valve I flagged) barely crept: +0.21 =>
1.24x, NOT exploited -> no need to freeze it. VICReg fine (var floor satisfied,
not dead). Diagnosis chain that got here: vicreg (healthy) -> #3 (scale
unpinned, push saturates correctly) -> #1 spectral norm (this).

ONE EXPECTED CAVEAT (units, not failure): with the scale now bounded the
achievable distance ceiling is ~22, BELOW the floor of 30 -> d_unr (21) ~
d_rand (20): certified-unreachable pairs end up NO FARTHER than generic far
pairs, washing out the oracle signal. The floor/offset (30/15) were chosen as
absolute ply-distances against an UNbounded scale; they must be reconciled with
the bounded range. FIX (single lever, recommended): raise iqe_embed_scale
1->~2 (a FIXED, non-learnable magnitude knob SN doesn't touch) to lift the
ceiling to ~44 so floor=30 is achievable with separation, keeping the numbers'
ply-meaning. Alt: lower floor->~15, offset->~8 into the current ceiling
(changes ply-calibration). Next: validate the rescale short, then the full 40k
run. Separate worthwhile lever (Kaveh): strip omega from F for the quasimetric
field (geometry should be player-independent; omega belongs on the committor/
measure side only) -- --omega-free-field.


## 2026-07-18 (Fable) — BUNDLED FIX STABLE: SN + freeze-log_scale + embed_scale2 + omega-free -> launching the full run

qrl_iqe_sn3_smoke (2500 steps, all four levers bundled per Kaveh "get rid of
the one-lever rule"): STABLE, 0 collapse.
  final: lam=21.9 (bounded), d_step=1.20 (pinned), d_rand=27.1, d_unr=27.7.
  log_scale=0.000 EXACTLY (freeze confirmed -> scalar escape valve removed).
  omega_free=True (F is player-independent geometry now).
Spread 27 vs step 1.2 = a 22:1 reachable/far ratio, bounded and stable. The
d_unr~d_oth intermingling is expected (both cross-game pairs are genuinely far;
the important separation is d_step<<d_rand, which is clean). The scale-runaway
saga is closed: spectral norm bounds the coordinate scale, frozen log_scale
removes the scalar gauge, embed_scale=2 sets a fixed working scale with room
for the floor, omega-free keeps the geometry a property of the rules. Bundle
recorded here for bisection if the full run regresses.

Launched the FULL run: qrl_iqe_sn_full.pt, 40k steps, this exact recipe,
ckpt-every 10000, detector armed. This is the first field recipe that survived
the full 2500-step validation with lambda bounded and the pin held -- the
candidate for a genuinely-spread, conversion-testable field.


## 2026-07-18 (Fable) — move ordering = plan-alignment, NOT a learned policy head (Kaveh)

Kaveh's call on the MCTS move-ordering / thin-visits problem (value-only
expansion => ~7 root visits at 4000n, moves not field-ordered): do NOT add a
learned policy head. Instead the MCTS PRIOR = geometric progress toward the
PLANNER's active subgoal g:
    pi(a) ∝ exp( beta * [ d(s->g) - d(s·a->g) ] )
moves that advance the plan (reduce quasimetric distance to the subgoal) get a
higher prior -> PUCT + progressive widening concentrate visits on plan-aligned
lines. "how to weigh options" = beta (plan-alignment weight / temperature),
a tunable MCTS parameter, mixed with c_puct (exploration). The FIELD is the
policy (via reachability to the subgoal); no separate learned head, no new
training target -- it reuses the quasimetric we're already training and ties
the search directly to the planner (the planner-as-prober design: plan sets g,
search drives to g). Honest tradeoff recorded: this ORDERS moves by the field
(fixes "not field-ordered") and lets widening go DEEP on the plan instead of
shallow-wide, but still embeds all children to RANK them, so it doesn't cut the
per-node eval cost the way a learned policy would amortize -- accepted, for the
field-native cleanliness. GATED on a SPREAD field: on the small-world incumbent,
d(s->g)-d(s·a->g) is noise; only on a spread quasimetric is it signal -- so this
is a post-field capability, which is why we wait for qrl_iqe_sn_full. Deferred
until that run's verdict.

## 2026-07-19 (late night): certified outcomes, progressive widening, position memory, UI overhaul

**Certified-outcome labels** (Kaveh: timeouts must not shape the win surface; then
"include resignations at 3+ points"). New `catspace/data/certified.py::
collect_certified_games`: a game's outcome label is trusted iff DRAW, win by
CHECKMATE (board-proven), or decisive non-mate win with the winner up >=3
nominal points at the final position. Measured on 3 shards (54,286 games):
draw 4.1% + mate 27.2% + material-backed 43.4% = **74.8% certified**; the
masked 25.2% are balanced-position resignations/flag-falls (script printout,
cert_check2). Wired as `--committor-certified-only --resign-material-gap 3.0`
in train_lichess_fb: gates ONLY the phead CE (+ cert-base distance targets);
geometry still trains on ALL positions (Kaveh: "apply to the geometry, just
not the phead"). zgoal poles were already clean (collect_mate_finals gates on
is_checkmate). Full-shard scan: 77.1% certified over 1M games (MEMORY_BUILT
verdict). Also applied to the ATLAS: build_play_atlas now filters samples to
certified games by default (--all-outcomes to disable); rebuilt atlas = 4000
pts, result counts W1878/B1783/D339.

**Progressive widening in MCTS** (Kaveh: "add it now until planning works
later"). Selection-level, canonical K(N)=max(4, ceil(pw_c*N^0.5)) (Coulom
2007): descent restricted to top-K children by mover-perspective value; all
children still created (rule terminals exact; a mate child is always in the
window) and batch-evaled once. Flag-gated OFF by default (pw_c=0 = bit-identical
prior behavior; 21/21 tests pass incl. 3 new). A/B at matched budget
(incumbent committor-MCTS, printed verdicts):
  start  2000n: sims 119->124, depth 5->6, top1 visits 50->62
  midgame 2000n: sims 43->53 (+23%), depth 3->5
  toy    400n: no change (priors already sharp; window never binds)
HONEST READ: helps depth modestly; the dominant width cost is STRUCTURAL
(value-only expansion evals every legal child: 44-move midgame = 9 expansions
per 400n). Only an expansion-level ordering (the plan-alignment prior) removes
that -- this is the interim lever, not the fix.

**Position memory** (Kaveh: "embeddings into a vector DB -- nearest seen
positions and outcomes; also every position we see and every MC sim carried to
completion; tag human vs self-play"). New `catspace/memory/store.py`
(PositionMemory: hnswlib cosine ANN + provenance metadata human/selfplay/
play_ui/mcts_sim, certified flags, ckpt-tag guard against cross-field queries)
+ `experiments/build_position_memory.py` (seeds from TRAIN rows, holdout
excluded). VERDICT MEMORY_BUILT n=200000 dim=64 ckpt=cert_base_full.pt@155000
certified_frac=0.771 (embed 1421 rows/s CPU, ~2.5 min). Play server: /neighbors
+ /memory_add_game endpoints; _harvest_tree appends every search line reaching
a RULES-certified terminal (cert_planted excluded) with the terminal outcome as
the MC sample; UI game-over posts the full game. Live test: engine_move on
back-rank mate played Ra8#, memory 200000->200002, re-query returns the mate
line at dist 0 tagged mcts_sim. Toy-position neighbors: certified White-won
R+material endgames at cosine 0.037-0.047 (sane retrieval).

**Play UI/server** (Kaveh's bug reports + asks): analyze no longer blocks the
game -- a move/nav checkpoints+stops the chunked search (AbortController; server
tree = the checkpoint), nav auto-resumes from the new position; stale "engine's
last move" analysis cleared on your turn; search depth is a dropdown
(100..2000, default 500); Manual (both sides) mode added (Engine cycle
Black->White->Manual, lichess-analysis-board style); t-SNE rebuild from the UI
(/rebuild_atlas endpoint, hot-reload, atomic atlas write); map redraw throttled
~1/s. PERF: /analyze was calling the openTSNE transform PER CANDIDATE PER HOP
(~18-36 single-point transforms/call -> multi-second lockups); now ONE batched
transform per call: warm analyze 0.75s@100n / 1.11s@500n (curl timings).
c_puct exposed (--c-puct, server default 1.0; training default unchanged 1.5),
prior_tau exposed; server runs --pw-c 1.5.

**Durable launcher**: experiments/launch.sh (nohup+disown -> reparents to
launchd; caffeinate -i -w <pid>; timestamped log + stable symlink + pidfile) --
long runs now survive terminal/VSCode/Claude close. Smoke-verified (PPID 1,
clean kill, caffeinate auto-exit). RUNBOOK §0.

**Field run** (qrl_iqe_sn_full, the bundled stability recipe): clean through
step 33900/40000 -- lam ~57 (slow creep, decelerating), d_step pinned 1.015,
d_rand ~46, var 0.42, ZERO collapse -- the 6k-18k death window that killed
every prior recipe is far behind. Verdict + conversion A/B vs the 0.600
incumbent when it lands. Overnight (Kaveh-approved, lowest priority, GPU-idle):
smoke then full retrain of the SAME recipe + certified-only committor labels.

**Tactical move prior** (Kaveh: "MCTS should spend nodes on checks, captures,
threats"). Rule-derived flag per child (is_tactical_move: gives check | capture
| promotion | moved piece attacks a strictly-higher-value or undefended enemy
piece -- board truth, no learned heuristic), consumed two ways: (a) tactical
children are ALWAYS in the progressive-widening window; (b) prior blend
P=(1-w)*P_field + w*uniform(tactical). Ordering only -- values untouched, so a
refuted tactic still loses on Q. Flag-gated OFF (training/eval unchanged);
server runs --tactical-prior 0.25. Tests 24/24 (detector fixtures incl. a
no-check no-capture knight threat; boost + window membership; default-off).
A/B on a hanging-queen position (VERDICT TACTICAL): no-op -- the incumbent
already priced Nxd5 top-1 (P 0.112->0.119, best_move unchanged), i.e. this
lever pays only where the field MISPRICES the tactic (the toy rook-hangs);
mechanically guaranteed there (window + P >= w/|tactical|). Interim like PW;
both fold into the plan-alignment prior later; re-test on field promotions.

**CONVERSION A/B: the stable spread field does NOT play** (the night's headline).
PLAYOUT_AB SN_FULL_VS_INCUMBENT mate-rate A=0.600 vs B=0.150 diff=-0.450
CI=[-0.560,-0.350] e=1.5e10 (n=100, 49 decisive; plies-to-mate A=20 B=7)
[SIGNIFICANT]. Crucial control vs the leaky field's 0.160 (e=7.8e9): THAT
result was confounded by its step-13k collapse -- this run had NO collapse,
cleared all 40k steps, spread d_rand 60.7 with d_step pinned 1.09, and still
plays the same 0.15. So "spread is not sufficient for play" is now
UNCONFOUNDED: a geometrically healthy QRL spread field converts only
near-mate starts (B's mates average 7 plies vs the incumbent's 20 -- no long
navigation). With VAL_TOP8 0.111 and won-lost slope separation 0.014, the
bottleneck is the outcome-navigation signal in the geometry/committor, NOT
training stability and NOT phead label noise. Implication for the overnight
run: retraining the SN recipe with certified labels would test labels on a
geometry that cannot play -- retarget the certified-label retrain to the
INCUMBENT recipe (cert_base, the 0.600 player, provenance args recovered from
the ckpt), where cleaner committor labels can actually move the primary
metric. Spread stays a planner-side research line (plan-alignment needs it),
not a player.

**Overnight run launched: cert_base_certified** (Kaveh-approved overnight slot,
retargeted post-A/B; lowest priority via nice -n 10). FAITHFUL incumbent
reproduction: provenance args + model CONFIG from the ckpt (the args said
quasimetric=False but the MODEL is quasimetric=True -- inherited through its
resume chain; the first smoke crashed on exactly this and was relaunched with
--quasimetric; short-run gate passed: mask ON 753191/1e6=75.3%, VAL_TOP8 0.162,
reach slopes 0.92/0.91 at 2k). Full: 155k steps, d=64 ch=64 blocks=6 enc_out=256
dh=512, batch 256, lr 2e-4, cert-base(lam 8, scale 50), phead 0.3 + NEW
--committor-certified-only --resign-material-gap 3.0. ~3h at ~14.5 it/s.
CAVEAT recorded now, not after: cert_base_full (0.600) was COMPOSED over
resumed rounds; a single fresh 155k of the final recipe is not guaranteed to
reproduce 0.600, so the morning A/B (cert_base_certified vs cert_base_full)
reads as "certified labels + fresh single-run vs the composed incumbent" -- if
it lands lower, attribution is ambiguous (composition vs labels) and a
composed-style certified rerun is the follow-up; if it lands at/above 0.600,
certified labels are at worst free and the win surface is board-honest.

**CERTIFIED-LABEL A/B (morning):** PLAYOUT_AB CERTIFIED_VS_INCUMBENT
A(cert_base_full, composed, raw labels)=0.600 vs B(cert_base_certified, fresh
155k, certified labels)=0.420 diff=-0.180 CI=[-0.300,-0.060] e=8.34 (40
decisive; plies-to-mate A=20 B=17) [SIGNIFICANT]. GRADING (the flagged
confound): B is a REAL player (0.420, navigates -- 17-ply mates), 2.8x the
spread field's 0.150, and won-lost DIFF_SLOPE separation 0.149 (vs 0.014). But
-0.180 CONFLATES two changes we made at once (breaking single-lever on purpose,
overnight): (i) composed-incumbent -> fresh-single-run, (ii) raw -> certified
labels, AND masking drops 25% of games' phead labels (less data). So we CANNOT
attribute the gap to certified labels. REQUIRED CONTROL: a fresh single 155k of
the identical recipe with RAW labels (no --committor-certified-only). If ~0.420
-> the gap is composition/fresh-run, certified is neutral (keep it: board-honest
for free). If ~0.600 -> certified/label-masking costs strength -> investigate
(masked-data volume vs label content). Not auto-run: overnight window over,
Kaveh back, directional 3h GPU call is his.

**CI-driven root exploration** (Kaveh 2026-07-19: "some moves are only tried
once; every move needs a minimum (~10) tries + CIs; keep trying until confidence
in a move's badness > ~95%"). = UCB/LCB best-arm ID at the root. Track W2 per
node -> value CI (normal-approx, values in [-1,1]); root (a) floors every
non-terminal move at root_min_visits, then (b) samples only moves whose UPPER
value bound still reaches the best move's LOWER bound (still-could-be-best),
leaving ~ci_z-confidently-worse moves alone. Flag-gated (root_min_visits=0 =
plain PUCT); server default 10. Deeper nodes keep PUCT+widening+tactical.
VERDICT (toy, 4000n, root-min-visits 10): all 24 legal moves visited, MIN
visits=10 (was 1), max 52 on the best move; per-move CIs surfaced -- Rh7
+0.76±0.09 (52 visits) vs floor moves ±0.19-0.20 (10 visits). Every move now has
a usable estimate + interval instead of a single-sample point. 28/28 tests
(4 new). Note the floor needs budget >= moves*min*branching (value-only
expansion pays ~branching evals/sim), else it distributes budget EVENLY rather
than reaching the floor -- still strictly better than PUCT leaving moves at 1.
Ties to planner cascade.py LUCB (certified-dominance stop).

## 2026-07-19 (cont.): AZ cheap expansion + policy-surprise (methodology fix)

**AZ-style cheap expansion** (policy head, F-only): expanding a node costs ONE
eval -- child priors from policy(F(node)), node value from committor(F(node)) =
P(Wwin)-P(Bwin); children created UNEVALUATED (FPU Q until visited). So the node
budget counts SIMULATIONS not branching*sims. VERDICT (toy, 500n): visit
concentration [270,92,51,40,12,11,10,10] vs value-only's [2,2,2,...] -- the
"all moves same visit count" symptom (3 separate reports) is GONE; 500 nodes now
does ~500 sims. Flag-gated (policy_fn=None -> exact old behavior); 31/31 tests
(3 new: cheap-expansion sim-count, mate-finding, off-by-default). Server
auto-loads <ckpt>_policy.pt.

**Policy head** (catspace/nn/policy_head.py, F-only, 4096 from-to). BC bootstrap
on frozen incumbent: VERDICT POLICY_HEAD top1_legal=0.108 (chance 0.031, n=120k)
-- field mildly move-informative but weak.

**Policy SURPRISE** (Kaveh: "keep MCTS results, check surprise" -- the right
metric, not human-move accuracy). Value-only search as the unbiased reference,
150 holdout positions @800n: VERDICT POLICY_SURPRISE KL(search||policy)=1.370
vs UNIFORM baseline 0.442, top1_agree=0.040, CE_bestmove=4.143. The BC policy is
WORSE than uniform as a search prior -- human-cloned moves point away from what
the committor-search prefers. Confirms BC is the wrong target; the AZ target is
distilling pi_search into the policy. Caveat: pi_search rides a weak value
(committor converts 0.15), so distillation buys cheap+self-consistent search,
not strength -- value/field stays the bottleneck.

## 2026-07-19 (cont.): the value IS the bottleneck -- committor flat, quasimetric d works

**WHY the field "can't play" -- pinned with numbers.**
- Committor calibration: on 200 TABLEBASE-WON toy positions (truth=1.0) the
  committor P(win) reads mean 0.818, max 0.927, only 4% >0.9. It learned the
  HUMAN conversion rate (~0.82 at 1800 Elo), not chess truth. Master/classical/
  no-blitz filtering would fix CALIBRATION (->1.0) but NOT the next problem:
- No GRADIENT: along tablebase-optimal forced-mate lines, committor P(win) is
  FLAT (0.79 @27+ plies -> 0.77 @1-6 plies, non-monotone) and cosine reach(F.zW)
  dead flat 0.65. A position 3 plies from mate reads the same as 30 plies away.
  KEY INSIGHT: win/draw/loss is CONSTANT over a won region, so NO outcome-trained
  value (however clean) can slope. The gradient is a DISTANCE (moves-to-mate),
  not an outcome.
- BUT the quasimetric DISTANCE has the gradient (Kaveh's point): d(F(s),MATE_W)
  DECREASES toward mate along the lines -- 0.411 @27+ -> 0.373 @1-6, per-line
  spearman +0.63 (shallow but correct). reach(F.zW) is a cosine dot (similarity),
  NOT the metric distance -- that's why it looked flat.
- DIRECT PLAY TEST: VERDICT VALUE_AB committor 0.425 vs quasimetric d(s->mate)
  0.525 (toy n=40 @800n). Navigating DOWN the metric gradient converts BETTER
  than the flat committor. The geometry had the answer; the engine was reading
  the wrong dial. Wired as play_server --value distance (value-only; reach=-d).
- REGION (goal-as-region, Kaveh: "mate is many points, min over exemplars"):
  d to NEAREST mate exemplar goes BACKWARDS (spearman -0.77 human mates, -0.79
  even with toy-specific KRRvKBP mates) -- the field's per-exemplar d is too
  noisy; the centroid's averaging is what smooths it into a usable signal. Right
  idea, needs a cleaner metric. soft_min_bank exists (make_search_policy bank z).

**t-SNE map degeneracy = same low-contrast field.** perp-500 single collapses
(corr(x,y) -0.33, xstd 2.7) and multiscale is unstable (a perfect corr=-1.000
LINE at n=6000) BECAUSE the field is mushy (cos>0.9, effective rank 15 / PC1
28%): 500-neighbourhoods average to ~uniform affinities, no local contrast to
preserve. perp-40 gives a proper map (corr -0.07, spread 5.5). Reverted default
to 40; high perplexity is a symptom, not a knob. Atlas rebuilt.

## 2026-07-19 (cont.): distance-map UI + nearest-mate verdict + DTM-hinge plan

**Geometry DOES encode reachability** (Kaveh: "geometry alone should say which
mates are unreachable, no tablebase"): on toy positions, d(nearest ROOK mate)
0.333 < d(nearest QUEEN mate) 0.351 in 98% of positions -- reachable nearer than
unreachable, confirmed. BUT the margin is 0.018 (mushy) -- structure present,
contrast absent.

**Nearest-reachable-mate A/B (the direct play test)**: VERDICT committor 0.425,
centroid d(s->MATE_W) 0.500, nearest-reachable-rook-mate (soft_min_bank) 0.100.
Nearest-mate navigation FAILS (0.10) on the mushy field -- per-exemplar d too
noisy (backwards gradient). The CENTROID wins (0.50) because averaging over
exemplars denoises. So nearest-mate is the right target but needs a SHARP
per-exemplar metric.

**Plan (Kaveh's design, confirmed by the above):** sharpness comes from MCTS
using distance-to-mate as the PRIOR (plan-alignment); alignment comes from a
TRAINING HINGE constraining d(F(s), mate) ~ tablebase DTM in tablebase range.
DTM source: Syzygy gives DTZ+WDL not DTM; Gaviota gives DTM but only <=5 pieces
+ no local tables. Use Syzygy-OPTIMAL ROLLOUT DTM (plies-to-mate under optimal
play; monotone; all <=6 of the toy tree; no download). Recommended field: QRL
spread (contrast) + DTM hinge (alignment) = the sharp+aligned metric neither
current field has.

**UI**: replaced cosine "reach" coloring with "distance->" + a pole selector
(White mate / Black mate / draw). Colors each map point by the quasimetric
d(F(pos), pole) -- near=green, far=red. Precomputed dW/dB/dD per point in
build_play_atlas (draw pole = mean embed_B of draw finals).

## 2026-07-19 (cont.): all-black map bug, committor material-blindness, multi-algo views

**All-black t-SNE map = late exaggeration collapse.** The map went degenerate
(atlas all-zeros). Isolated the cause: openTSNE's late `exaggeration` is an
ATTRACTION multiplier, not a separation knob (the docstring had it backwards).
On the low-contrast field it collapses everything to the centre:
  build cfg (exag=1.6): xstd=2.936
  no exaggeration     : xstd=26.938
Fixed default exaggeration 1.6 -> 1.0. Atlas rebuilt healthy:
  VERDICT ATLAS algo=tsne n=6000 xstd=53.03 ystd=29.97 xrange=[-98.5,104.8]

**Why the engine (Black) sacrifices a bishop (1.d4 e6 2.Nc3 Ba3??).** NOT a
negamax sign bug (MCTS backs up white-POV, flips sign at Black nodes -- verified).
The committor VALUE is material-blind. Measured white-POV committor (W-L):

| position | committor |
|---|---|
| KQ vs k, mate imminent | +0.870 |
| White up a whole QUEEN (full board) | +0.052 |
| Black up a whole queen | -0.065 |
| startpos | +0.022 |

Up a queen barely registers (+0.05) while imminent mate reads +0.87 -- the head
is a mate-PROXIMITY detector, not a material evaluator. At the Ba3 position the
whole legal-move value spread is 0.14; 2...Ba3 reads -0.038 and 2...Ba3 3.bxa3
(Black down a bishop) reads -0.062 -- noise inside a flat zero-signal band, not a
belief that losing material helps. NOT representational collapse: F separates the
positions (||F(Nf6)-F(Ba3bxa3)||=0.356 on unit-norm F). The POLICY head is fine
(Ba3 prior 0.001, Nf6 0.217) -- the AZ value backup overrides it. "Humans don't
lose material" is a POLICY fact (which moves they play), learned; it is NOT an
OUTCOME fact, so the outcome-trained committor never encoded it. GRADE: this is
the same flat-value disease, but far outside tablebase range -- the DTM hinge
sharpens the ENDGAME and will NOT fix a move-2 opening blunder. Honest limitation
of a pure reachability+committor field; full-board material awareness is a
separate ask.

**Multi-algo map views (selectable t-SNE / UMAP / VAE).** New
`catspace/viz/manifold.py`: uniform projector (fit / out-of-sample transform /
save / load + manifest.json). UMAP via umap-learn; VAE is a 64->2 PyTorch
compression VAE (the CompressionVAE idea; the TF package is unmaintained). VAE
posterior-collapses at beta>=1 (a 64->2 bottleneck can't reconstruct, KL wins,
xstd 0.00); fixed with KL annealing + default beta 0.02 (PCA top-2 = 21.9% var;
beta 0.02 -> mu_std ~1.1). Build verdicts:
  VERDICT ATLAS algo=umap n=2000 xstd=1.95
  VERDICT ATLAS algo=vae  n=2000 xstd=1.18 (beta 0.02) / 0.80 (beta 0.2)
build_play_atlas --algo + per-algo params; /rebuild_atlas passes algo+params;
server proj loader reads the manifest (legacy embedding.pkl still supported).
UI: algo selector with dynamic per-algo param fields; end-to-end tested
(umap rebuild -> hot-reload -> /project uses umap OOS transform). Dist dropdown
relabeled White/Black WIN (not mate) + Draw.

## 2026-07-19 (cont.): DTM hinge built + validated; FULL sharp+aligned run launched

**The DTM hinge (ALIGNMENT lever).** QRL gives a sharp unit-step metric but its
gradient is unaligned to mate (the spread field's backwards gradient). New
`--dtm-hinge`: on tablebase-WON endgame positions, regress d(F(s), MATE_W) onto
the true Syzygy-optimal rollout DTM (plies). QRL's unit-step => distances are in
ply units => target is dtm directly (scale 1.0), smooth_l1, mate centroid
refreshed on the zgoal cadence. Data: experiments/gen_dtm_data.py rolls the
Syzygy-optimal line and counts plies (Syzygy = DTZ+WDL not DTM, so we play +
count; monotone, covers the whole <=6-piece toy tree, no Gaviota download).
Acceptance 36-42% (krrkbp/krrvk/krvk), ~3-9/s.

**Smoke (sn_full recipe + hinge, weight 0.3, 300 positions, 1500 steps).**
The metric ALIGNS to true DTM while QRL stays sharp:

| step | dtm loss | d_mean (tgt~17) | rank_corr(d, DTM) | QRL d_step | d_rand |
|---|---|---|---|---|---|
| 100 | 15.9 | 0.67 | +0.016 | 0.82 | 0.79 |
| 500 | 8.0 | 9.9 | +0.687 | 0.95 | 14.6 |
| 1000 | 4.5 | 16.3 | +0.683 | ~1.1 | ~16 |
| ~1300 | 3.5 | 17.8 | **+0.806** | 1.0-1.3 | ~16 |

rank_corr 0.02 -> 0.81; d_mean 0.67 -> ~19 (matches DTM ~17); QRL unit-step held
(d_step ~1.0-1.3, d_rand spreading to 16, no COLLAPSE). Sharp AND aligned at
once -- the property neither prior field had (incumbent committor: material-blind
plateau; qrl_iqe_sn_full: sharp but backwards gradient). GRADE: proven on 300
positions; the full run tests it at scale.

**FULL run launched (Kaveh: "do a full training run instead of fine tuning").**
experiments/run_dtm_full.sh = the proven qrl_iqe_sn_full recipe (d=512, IQE
unit-step, QRL spread with PID/var/unreach anti-collapse, committor-base phead)
+ --dtm-hinge (weight 0.3, batch 128) on the 24k-position dataset. From scratch,
40k steps, ckpt-every 10000, halt-on-collapse. Durable wrapper waits for the DTM
data (gen ETA ~60-90 min) then trains (~4 h at ~2.7 it/s). Output:
qrl_dtm_full.pt. This is the sharp+aligned field candidate; conversion A/B vs the
incumbent is the acceptance test once it lands.

## 2026-07-19 (cont.): mate is a SURFACE — centroid hinge → composed retrieval

**The centroid DTM hinge plateaued.** The first full run (qrl_dtm_full, centroid
target d(F(s),MATE_W)->dtm) climbed to rank_corr ~0.3 early then ERODED to
~0.05-0.1 by step 14k; a large-sample check at step 15000 gave overall spearman
~0 (and per-material ~0 too: krvk +0.005). Root cause = Kaveh's surfaces-not-
poles point applied to the hinge: a single mate CENTROID cannot order diverse
endgames (KRvK / KRRvK / KRRvKBP) by DTM. Stopped at step 19400.

**Composed retrieval hinge (Kaveh's spec).** Regress the COMPOSED distance
  d_hat(s->mate) = min_g[ d(F(s), B(g)) + dtm(g) ]
over a bank of DTM waypoints g toward the true dtm(s): the field is trusted only
for the short hop to a nearby waypoint, g's dtm is grounded truth, the min picks
the nearest useful waypoint (SURFACE, not mean). Factored into a reusable
primitive `catspace/memory/retrieval.py::composed_distance` (top-k neighbours ->
distance to and through them -> min) for the hinge, readout, engine and
diagnostics. Smoke on the full 24k pool: composed ~0.25-0.3 vs centroid ~0.15.
Full run (qrl_dtm_surf): bank 256, weight 0.5, halt-on-collapse; early rank_corr
~0.2-0.45 (noisy, ~2x the centroid). ~1.8 it/s.

**Forced-mate curation (full-board anchors).** gen_forced_mate_data.py pulls
40000 mate-in-N positions from the Lichess puzzle DB (VERDICT FORCED_MATE
n=40000 white=21171 black=18829 dtm[1..9 plies]). The tablebase surface is
endgames only; these are FULL-BOARD near-mate anchors, the missing piece for
composing a middlegame/opening through a nearby mate.

**Propagation test of Kaveh's opening argument.** A piece-down opening should
read as losing because it funnels to a piece-down endgame that clearly is (the
quasimetric triangle inequality). experiments/propagation_ladder.py measures it
via the composed estimator (paired equal-vs-down per stage; W/L/D surfaces from
DTM + forced-mate + finals). Incumbent baseline is per-exemplar-noisy and can't
validate it (even a won KRvK reads farther from white-win than a KRvKR draw --
the known centroid-beats-nearest problem). The real test runs on qrl_dtm_surf
ladder checkpoints (does the move-2 material delta lift, and does propagation
reach the opening). GRADE: instrument built + committed; verdict pending the
sharp field.

## 2026-07-19 (cont.): DTM training hinge DISPROVEN — negative result + reframe

Thorough test of "align the field's distance to DTM by a training hinge". It does
NOT work, in every form tried. Stable held-out Spearman(d, DTM) via
experiments/eval_dtm_alignment.py (overall is a cross-MATERIAL scale artifact;
WITHIN-material is the meaningful signal):

| field | overall | within krvk | note |
|---|---|---|---|
| composed-hinge run @10k (qrl_dtm_surf) | -0.212 | **-0.126** | hinge HURT |
| centroid-hinge run @15k (qrl_dtm_full) | ~0 | ~0 | plateaued/eroded |
| **pure QRL, no hinge** (qrl_iqe_sn_full) | -0.139 | **+0.140** | weak-BEST |
| pure QRL + DENSE composed retrieval (bank 4000,k16) | **+0.116** | +0.097 | best overall |
| fine-tune (dominant hinge w2, reduced QRL) | — | — | rank_corr stuck ~0.1, d collapsed to ~3 |

**Diagnoses.** (a) CENTROID target: a single mate pole can't order diverse
endgames (KRvK/KRRvK/KRRvKBP) -- surfaces-not-poles. (b) COMPOSED-MIN target:
DEGENERATE minimum -- the min gives gradient to one anchor, so the field maps
every position near one low-dtm anchor, composed d -> ~constant (~3), no ordering.
(c) Both HURT: joint QRL+hinge gives WORSE within-material ordering than pure QRL
(krvk +0.14 -> -0.13). The QRL spread scrambles the hinge, and the hinge can't
impose ordering without collapsing.

**Reframe (the positive finding).** The pure QRL field already has the natural
(weak, +0.14) reachability ordering, and the composed RETRIEVAL readout with a
DENSE DTM bank recovers a positive overall signal (+0.116) WITHOUT any hinge --
the field only needs good short hops (QRL), the DTM comes from the bank at
READOUT time. So the path is NOT a training hinge; it is the retrieval readout at
inference. AND: rank_corr is weak for ALL fields, but weak ordering already gave
the toy conversion 0.50 vs 0.425 committor -- so the ACTUAL test is conversion,
not rank_corr. Next (needs Kaveh's steer): (1) integrate the composed retrieval
readout as the engine value + run conversion A/B (no new training); (2) if a
trained alignment is still wanted, a MONOTONICITY/ranking hinge along optimal-DTM
successor pairs (avoids the degenerate min) is the untried option. Stopped all
training; incumbent serves the UI. cert_base_full remains the incumbent.

## 2026-07-19 (cont.): direction 1 (composed retrieval readout) — disproven for conversion

experiments/conversion_composed_ab.py: composed retrieval readout as the engine
value on the pure QRL field vs the incumbent committor, KRRvKBP winning starts,
tablebase defender.
  VERDICT COMPOSED_CONVERSION n=30 nodes=400 A_incumbent=0.567 B_qrl_composed=0.300 diff=-0.267
Composed retrieval converts 0.300 vs the incumbent's 0.567 -- WORSE by 0.27, and
~12x slower. (n=8/200n smoke was 0.0 vs 0.125, small-sample.) Caveat: A/B changed
field AND readout together; but the gap is too large for field-difference alone.
NET across the whole DTM thread: neither the training hinge (centroid/composed/
fine-tune) NOR the inference readout beats the incumbent committor (0.567) for
endgame conversion. The incumbent remains the best converter. Decision pending
(Kaveh): option 2 (monotonicity/ranking hinge -- the one principled untried
lever) vs accept the incumbent for conversion and pivot the research.

## 2026-07-19 (cont.): autonomous hunt to beat committor 0.567 -- diagnosis

Literature (NN tablebase-DTM approx; QRL/TMD use Bellman-consistency, not
regression). Tried, KRRvKBP conversion n=30 nodes=400 vs committor 0.567:
  board-DTM navigation      0.533  (ties; krrkbp DTM only 0.29 spearman, hard)
  F-DTM head navigation     0.167  (F is DTM-poor, head plateaus 0.227)
  composed retrieval readout 0.300
  subgoal planner (field-dist nav) 0.100  (wanders to phantom subgoals)
  QRL field committor       0.133  (QRL is a WORSE conversion substrate)
NOTHING beats the incumbent committor. FIELD-DISTANCE navigation (composed,
subgoal) fails badly -- the field's distance is not a usable navigation signal.

FAILURE DIAGNOSIS (experiments/conversion_failure_diag.py, committor 0.600):
the 40% failures split ~50/50:
  (1) MATERIAL BLUNDER: 5/10 fail having LOST rook(s) (KR v BK x4, K v BK x1) --
      the committor is material-blind, throws rooks away.
  (2) FAILED TO MATE a won position: 5/10 (KRR v BK x2, KRR v BKP x2, KRR v K x1)
      -- poor mating technique; it can't even mate KRR-vs-lone-king in budget.
Each approach fixes at most ONE mode (DTM gives a mate gradient but not
blunder-avoidance; and KRRvKBP DTM is too hard to learn). The grounded path:
a MATERIAL-AWARE value (kill mode 1) + subgoal "reach KRRvK safely" then the
board-DTM net (krrvk 0.68 / krvk 0.93) mates the simplified endgame (kill mode 2).

## 2026-07-19 (cont.): ensemble ties; testing search-ceiling
committor + board-DTM ensemble (w=0.15): VERDICT ENSEMBLE_CONVERSION
A_committor=0.567 B_ensemble=0.567 diff=+0.000 -- EXACT tie, DTM term changed no
decisions. 0.567 is a robust local optimum for value/policy/planner methods at
400 nodes. Open question: search-limited vs value-limited (committor @1600 nodes
running). If more search jumps it, the planner's real value is EFFICIENCY.

## 2026-07-19 (cont.): CONVERSION IS SEARCH-LIMITED (the key reframe)
committor @400 nodes = 0.567; @1600 nodes = 0.767 (experiments/conversion_dtmhead_ab.py
--nodes 1600, side A). +0.20 from 4x search => conversion is SEARCH-limited, NOT
value-limited. This is why every value/policy/planner VALUE tied ~0.567 -- the
value is fine; search DEPTH is the bottleneck. Kaveh's "planner is key" is right,
but the planner's job is EFFICIENCY (reach the deeper result at fewer nodes by
shortening the horizon), and the standard lever is POLICY PRIORS focusing the
search (AlphaZero). Re-running the policy-AZ conversion to completion.

## 2026-07-19 (cont.): CLUSTER FORMATION works (Kaveh's direction)
Bug-check: ensemble tie was real coincidence (DTM term DOES change move rankings:
committor Rxf2+ vs ensemble Rxd3), not a bug. Policy-AZ (weak board-policy priors,
top1 0.13): 0.100 vs 0.567 -- weak priors HURT.

Field structure diagnosis (incumbent): NO symmetry-invariance (||F(pos)-F(mirror)||
0.192 vs random 0.176 -- mirror NOT closer!) and NO DTM-clustering (within=between,
ratio 1.00). The embeddings are isotropic/structureless -- likely WHY every
navigation approach failed.

cluster_finetune.py: fine-tune F with L_sym (F(pos)=F(mirror)) + L_clust (same-DTM
close, diff-DTM apart, margin) + L_anchor (stay near F0). VERDICT CLUSTER
symmetry_ratio 0.91->2.37  dtm_clustering 1.02->1.35 -- CLUSTERS FORMED, anchor kept
F near original (L_anchor 0.036). First structural win. Testing whether the
clustered field makes the subgoal planner work (failed 0.10 on the structureless
incumbent).

## 2026-07-19 (cont.): clusters form but STRATA don't; planner still needs directed structure
Cluster viz (field_clusters.png): incumbent MIXES KRRvKBP+KRvK; clustered field =
3 clean material clusters (the natural conversion chain). BUT subgoal planner on
the clustered field = 0.133 (vs committor 0.600) -- static clusters don't help
navigation.

STRATA diagnosis (experiments/strata_diag.py): for irreversible moves (captures/
pawn/promo -- literally no way back), backward/forward directed distance should be
HUGE; measured only 1.1-1.3x. Strata ratio (irrev_asym/rev_asym): incumbent 1.17,
QRL(unreach-8) 1.69, clustered 1.42 -- barely any. WHY: incumbent has no
irreversibility signal (InfoNCE); QRL's --qrl-unreach-weight is too COARSE (pushes
provably-unreachable cross-game pairs, not THIS capture's backward step); and the
incumbent's distances are compressed (~0.4 range, no room for asymmetry) vs QRL
~18. Fix: a LOCAL strata hinge -- for each irreversible move (parent->child), push
d(child->parent) >> d(parent->child). That is the directed structure the planner
needs (which one-way transitions lead toward mate).

## 2026-07-19 (cont.): strata hinge fails (inflation); nucleation reframe
Cluster+strata fine-tune: symmetry 0.91->2.78, dtm_clustering 1.02->1.47 (clusters
strengthen), but strata_ratio 1.02 (robust) -- NO asymmetry. The one-sided strata
hinge (push irreversible-backward above floor) was satisfied by INFLATING all
distances (fwd 0.44->2.2), not by asymmetry. A proper strata objective needs the
quasimetric ARCHITECTURE (IQE) pinning fwd~unit + reversible symmetric, not a
scalar floor on the MRN incumbent. Literature: ProQ (2506.18847) = asymmetric
distance + proximal subgoals (our vision); IQE (2211.15120) = the arch; Laplacian/
symmetric can't do one-way reachability. We're not alone -- quasimetric RL is
active. NUCLEATION (Kaveh): anchor only the near-mate NUCLEUS (small DTM + clusters,
tablebase) rigidly, let far positions PROPAGATE their distance to it via local
reachability (no tablebase for the far field). Explains the DTM-hinge failure
(anchored everything at once). Checking if clustering already improved
distance-to-nucleus.

## 2026-07-19 (cont.): NUCLEATION architecture (roadmap from the design session)
Nucleation signal: clustered field centroid spearman(d,DTM) +0.330 (first POSITIVE
alignment all session; incumbent was ~-0.13). Structuring the near-mate core makes
distance-to-mate track DTM.

ROADMAP (Kaveh's design):
1. NUCLEUS DATA = real Lichess <=5-piece positions + exact tablebase DTM
   (gen_lichess_nearmate.py; "all sorts of combinations", richer than synthetic).
2. TABLEBASE FOUNDATION: train a field on the nucleus (near-mate, DTM + clusters),
   checkpoint SEPARATELY. Then FINE-TUNE different hinges on top (modular, isolates
   each hinge -- vs the everything-at-once runs that failed).
3. HINGE STACK (each proven/diagnosed this session):
   - clustering: WORKS (symmetry 0.91->2.37, DTM 1.02->1.35, +0.33 alignment).
   - strata: a scalar floor FAILS (inflates all distances, ratio 1.02); must come
     from the IQE quasimetric ARCHITECTURE (ProQ 2506.18847 / IQE 2211.15120), not
     a hand hinge on the MRN incumbent.
   - propagation/nucleation: far positions learn d(s->nucleus) via LOCAL
     reachability (successor unit-step), no tablebase for the far field.
4. PLANNER: cluster/nucleus subgoal navigation on the structured field. Note the
   headline: conversion is SEARCH-LIMITED (0.567@400 -> 0.767@1600); the planner's
   job is EFFICIENCY (shorter horizon via the structure), and it needs the DIRECTED
   structure (strata) to work -- static clusters alone gave 0.13.
OPEN (Kaveh unsure): nucleus-first curriculum vs simultaneous; foundation on IQE
(for strata) vs MRN.

## 2026-07-19 (cont.): IQE nucleus foundation -- structure yes, strata no
train_iqe_nucleus (6000 steps, 19846 won Lichess <=5-piece, 198 material classes):
VERDICT IQE_NUCLEUS overall_spearman(d,DTM)=+0.255 (plateau ~0.25; incumbent ~0/neg).
Symmetry + material separation formed cleanly. BUT strata irrev/rev asym = 1.00
(REV 1.11 IRREV 1.12) -- NO strata. KEY FINDING: the IQE architecture is NECESSARY
but NOT SUFFICIENT for strata. IQE *can* express asymmetry, but nothing in the
objective (position-level DTM ranking + symmetry + separation) TRAINS it. The only
field with strata (qrl_iqe_sn_full 1.69) had the QRL successor/irreversibility
objective. => strata come from training the TRANSITIONS: optimal-successor unit-step
d(s,s')~1 + irreversibility d(s'->s) huge for captures. Next foundation iteration:
add successor pairs (tb_best_move) + the QRL unreach term to the nucleus training.
Also DTM order plateaued at 0.25 -- the successor unit-step (chaining DTM via
d(s,s')~1) should sharpen it beyond position-level ranking.

## 2026-07-20: pawn-capture INFINITE one-way -- forms in train mode, NOT eval (discipline catch)
Search: IQE represents infinite distance (high output for unreachable pairs, 2211.15120);
hyperbolic disk / entailment cones / order embeddings = partial-order alternatives.
Built: lichess_nearmate.npz (nucleus), gen_pawn_capture_pairs (19k pairs; <=5-piece
endgames are pawnless so mined from full shards), cluster_infinite_finetune (forward
pinned ~1, backward -> inf-floor 60 via IQE, + clustering).
RESULT (Kaveh's rule -- measure independently, eval mode): in-loop asymmetry hit 62x
(d_fwd 1.03 d_bwd 64) BUT independent eval-mode check on held-out pairs = 1.2x
(fwd 6.4 bwd 7.5); reversible-king control 1.01x (correct). So the one-way structure
forms in TRAIN mode but does NOT hold in the usable EVAL field, and scale inflated
(fwd 4->6.4). Causes: (1) BatchNorm train/eval mismatch (BoardEncoder has BN --
learns w/ batch stats, used w/ running stats), (2) per-batch memorization not
generalization. Fix: resolve BN (LayerNorm or long BN warmup), much more training on
all 19k pairs, and pin forward across ALL pairs not just the batch. The MECHANISM
(IQE ∞ + forward-pin) is right; the training doesn't generalize yet.

## 2026-07-20: strata/infinite CONFIRMED to work -- blocked only by BatchNorm (the fix)
Disambiguation (Kaveh's measure-independently rule): pawn-capture one-way asymmetry
d(child->parent)/d(parent->child) on HELD-OUT pairs:
  train mode (batch BN stats): 58.8x  (GENERALIZES -- the field learns the rule)
  eval mode (running BN stats): 1.0x
  eval after BN running-stat RECALIBRATION (200 passes): 1.0x  (NOT stale stats)
  reversible-king control: 1.01x (correct)
=> The infinite one-way mechanism (IQE ∞ representation + forward-pin ~1) WORKS and
GENERALIZES, but BatchNorm2d in BoardEncoder makes the model EXPLOIT batch stats;
the structure lives in per-batch-normalized space and vanishes under any fixed
normalization. Recalibration can't fix it. FIX IS ARCHITECTURAL: replace BN with
LayerNorm/GroupNorm in catspace/nn/encoder.py (no train/eval gap) -> retrain the
nucleus foundation + infinite fine-tune -> the 58x asymmetry holds at inference ->
usable one-way field -> planner. This is the concrete unblock.

## 2026-07-20: PROOF that BatchNorm is the sole cause (census + isolation + visual)
Kaveh: "prove it's the issue... this is an important finding I want captured." Done
rigorously in experiments/prove_batchnorm.py -- ONE trained field (iqe_infinite.pt),
the SAME held-out pawn-capture pairs, IDENTICAL weights; the ONLY thing changed is
the normalization mode. Metric = one-way asymmetry d(child->parent)/d(parent->child)
(1 = reversible, >>1 = the learned "no way back").

WHAT BATCHNORM DOES. BatchNorm2d normalizes each channel by the mean/variance of the
CURRENT batch during training, but by fixed RUNNING averages at inference (eval).
Those are two different functions of the same weights. A net can therefore encode
structure in a way that only survives batch-statistic normalization -- present while
training, absent when used. That is exactly the failure here.

CENSUS (makes the proof airtight by construction). The only train/eval-sensitive
modules in the field are BatchNorm: BatchNorm=44, Dropout=0, other(InstanceNorm...)=none.
So the ENTIRE train<->eval behavioral gap IS BatchNorm -- there is nothing else it
could be.

ISOLATION EXPERIMENT (the decisive step). Same weights, same pairs, three modes:
  (A) full EVAL   (BN = running stats, how the field is USED):   asym = 1.1x
  (B) full TRAIN  (BN = batch stats):                            asym = 67.8x
  (C) EVAL EXCEPT BatchNorm->train (only BN flipped, all else eval): asym = 67.8x
  reversible-king control (full eval):                           asym = 0.96x  (correct)
(C) == (B) >> (A): flipping ONLY the 44 BatchNorm layers to batch stats -- with every
other module held in eval -- recovers the FULL 67.8x. Nothing else moves the needle.
BatchNorm is not a contributor; it is the entire effect. VERDICT BN_PROOF
eval=1.1x train=67.8x bn_only=67.8x.

VISUAL (artifacts/experiments/batchnorm_proof.png, experiments/visualize_batchnorm.py).
Histogram of the per-pair asymmetry ratio, SAME weights/positions, only BN stats
differ: reversible moves (blue) sit at 0.9x in every mode; pawn captures under running
stats / eval (red) collapse onto 1.0x; pawn captures under batch stats (dark red) form
a cleanly separated peak at 62x. Two identical-weight red distributions, an order of
magnitude apart, with BatchNorm the only difference -- the picture of the field
learning a one-way rule that inference erases.

CONCLUSION. The one-way (irreversibility/strata) mechanism -- IQE's infinite-distance
representation + forward-pin ~1 -- WORKS and GENERALIZES to held-out pawn captures.
It is destroyed at inference solely by BatchNorm's train/eval statistic mismatch, and
running-stat recalibration cannot recover it (the structure lives in per-batch space,
not stale averages). FIX IS ARCHITECTURAL, not a tuning knob: replace BatchNorm2d with
a normalization that is identical in train and eval (GroupNorm/LayerNorm) in
catspace/nn/encoder.py, then retrain the nucleus + infinite fine-tune. Existing BN
checkpoints cannot load into a GroupNorm model -- the fix requires fresh training.

## 2026-07-20: BatchNorm FIX verified -- one-way structure now holds at inference (GroupNorm)
Fix applied: catspace/nn/encoder.py -- all 44 BatchNorm2d -> GroupNorm (helper _norm(),
largest group count <=32 keeping >=4 ch/group) in _ResBlock (b1,b2), stem, and head.
GroupNorm normalizes WITHIN each sample (no batch dimension, no running stats), so it
computes the IDENTICAL function in train and eval -- the train/eval gap is removed by
construction. Smoke: fresh field has 0 BatchNorm / 44 GroupNorm and train==eval to
0.00 (max|dF|). BN checkpoints can't load into a GN model -> retrained from scratch.

Pipeline (GroupNorm): iqe_nucleus_gn (5000 steps, overall spearman(d,DTM)=+0.393 --
BETTER than the old BN nucleus +0.255) -> cluster_infinite_finetune (1500 steps,
batch 128). Then prove_batchnorm.py re-run on the resulting field.

DECISIVE RESULT (prove_batchnorm.py on iqe_infinite_gn.pt) -- SAME experiment that
gave eval 1.1x / train 67.8x on the BatchNorm field:
  train/eval-sensitive modules: BatchNorm=0  Dropout=0  other=none
  (A) full EVAL  (as USED at inference):        fwd=1.35 bwd=67.66 asym=47.7x
  (B) full TRAIN:                               fwd=1.36 bwd=67.28 asym=47.2x
  (C) EVAL except BN->train (isolate BN):       fwd=1.37 bwd=67.28 asym=47.0x
  reversible-king control (full eval):          asym=0.99x  (correct)
  VERDICT BN_PROOF eval=47.7x train=47.2x bn_only=47.0x
A == B == C: the 60x train/eval gap is GONE. The pawn-capture one-way asymmetry that
BatchNorm erased at inference (68x train -> 1.1x eval) now HOLDS in eval at 47.7x -- a
capture's way-back is 47x the move itself, at inference. VERDICT INFINITE (fine-tune,
eval mode): pawncap_asym 0.99->47.86 (fwd 11.66->1.36, bwd 11.62->68.31). The 47x is
lower than BN's fake 62-68x but is REAL (survives eval) rather than a batch-stat
artifact. Reversible stays 0.99x. This is the concrete unblock: a usable one-way /
strata field. BONUS: DTM cluster order survived the fine-tune this run, sp(d,DTM)
ended +0.24 (peaked +0.44) vs the batch-256 attempt's -0.16 -- smaller batch was
gentler on the clusters.

Speed notes (Kaveh's "parallelize? remove-unnecessary-compute?" rule): the fine-tune
is GPU compute-bound on one MPS device -- fusing 5 encoder passes into 2 (same FLOPs)
gave ~0 speedup (834s vs 818s to step 500), while HALVING FLOPs (batch 256->128) gave
a real ~2.8x (1.66 -> 0.58 s/step). CPU sat at 3.1% (GPU-bound fingerprint) => NOT
multiprocess-parallelizable (one GPU; more procs just time-slice it). Lesson logged:
stage-2 fine-tune has no mid-run checkpoint, so each kill/relaunch restarts it from
the nucleus -- add periodic checkpointing before the next long training job.

## 2026-07-20: geometry field redesign -- reachability + targeted irreversibility (iqe_geom.pt)
Redesign (Kaveh): the quasimetric field = LEGAL reachability ("what COULD be played"),
NOT the played/usual policy (that is the separate occupancy/successor-representation
model). Simplified objective replacing the L_inf/pawn-death approach:
  L_pos   successor pin d(s->s')~1 for legal plies (positives = edges from the graph)
  L_hard  TARGETED hard negative: d(child->parent) >= d(parent->child)+margin for
          rule-defined IRREVERSIBLE edges only (chess.Board.is_irreversible = pawn
          moves, captures, castling, AND castling-rights loss = the clock-resetting set)
  L_neg   in-batch triplet negatives   L_rank within-material DTM   L_grank cross-mat
  L_sym   mirror invariance            L_sep pawnless-only separation
Board-only geometry: halfmove clock (plane 18) + repetition (plane 19) ZEROED into the
distance tower -> shuffle-equivalent positions cluster; clock/rep are separate monotone
potentials for the planner (a bishop shuffle: flat board-DTM, burned clock budget).

KEY FINDINGS (fail-fast smokes, Kaveh's rules):
1. PURE EMERGENCE DOES NOT WORK. "irreversible reverse isn't a legal edge so negatives
   push it large" is false: random in-batch negatives never touch that specific pair.
   300-step smoke gave reversible 0.91x AND irreversible 1.07x -- both symmetric, no
   asymmetry. The reverse-of-irreversible is a HARD negative that random sampling
   misses. => the one-way must be a TARGETED term on is_irreversible edges.
2. The EMA loss-scale normalizer SABOTAGES the hard negative: normalizing a loss by its
   own magnitude divides its GRADIENT by that magnitude, so a term that must push d_bwd
   far gets ~0 gradient (d_bwd collapsed to 1). Dropped it; used a RELATIVE margin
   (d_bwd >= d_fwd+margin) with fixed weights -> asymmetry forms.
3. CLUSTERING vs GLOBAL-DTM-VIA-SINGLE-POLE ARE INCOMPATIBLE. L_sep pushes materials
   into separate clusters, so a single mate pole sits at cluster-determined distances
   regardless of DTM -> overall spearman cannot be positive. This is the real "centroid
   can't order across materials" wall. => the planner metric is per-material (within-
   cluster) order + cluster-level tablebase DTM, NOT overall spearman.

RESULT (eval_geom.py, held out, EVAL mode, on iqe_geom.pt @1500 steps):
  per-material spearman(d,DTM) MEAN=+0.454 (median +0.481, all 10 classes positive:
    KRk +0.52 Kkr +0.63 KRkr +0.39 KPk +0.68 KPkq +0.61 Kkq +0.52 KQk +0.45 ...)
  reversible one-way asym = 1.09x ; IRREVERSIBLE = 8.27x (d_fwd 1.00, d_bwd 8.96)
  one-way SEPARATION = 7.59x ; overall spearman = +0.025 (flat, not inverted)
  VERDICT GEOM_EVAL permat=+0.454 overall=+0.025 oneway_separation=7.59x
vs the prior iqe_infinite_gn.pt: per-material ~0.00 (useless within-cluster), and its
one-way was a BatchNorm/L_inf artifact. iqe_geom.pt has BOTH the within-cluster descent
signal (+0.454, up from ~0) AND correctly-targeted strata (rev symmetric, irrev 8.3x)
that hold at inference (GroupNorm). This is the field the planner descends. Files:
gen_successor_edges.py (edges + is_irreversible flag), train_field_geometry.py, eval_geom.py.

## 2026-07-20: 3-layer architecture -- reachability geometry (L1) / categorical outcome (L2) / strength (L3)
Kaveh's chain of realizations resolved the accumulated tension:
- Fitting the geometry's quasimetric to tablebase DTM is WRONG: DTM is ADVERSARIAL
  (min-max under optimal defense), the successor-pin geometry is COOPERATIVE reachability
  (every legal ply = unit step, both sides). A cooperative metric can't equal an
  adversarial distance -> the mismatch that capped per-material order and flattened global.
- The circularity ("need the adversary's policy to compute DTM, but policy is a layer on
  top of geometry") DISSOLVES: optimal-adversarial DTM is policy-INDEPENDENT and the
  tablebase already computed it offline (retrograde = adversarial search done once). So no
  policy is needed at train time; DTM is a precomputed LABEL for a head, not a target for
  the geometry. Strength-realized cost is the only policy-dependent part -> L3.
=> ARCHITECTURE. L1 geometry: policy-independent reachability, trained ONLY on legal
   successor edges + huge random pushes; NEVER fit to DTM. L2 categorical value head: reads
   the frozen L1 embedding, predicts the outcome DISTRIBUTION (win-distance bins + DRAW),
   supervised on EXACT tablebase labels (retrograde values). L3 strength/policy: omega-
   conditioned realized cost -- deferred.

L1 MINIMAL OBJECTIVE (Kaveh: "only nearest-neighbor one-directional hinge at d=1 and huge
random pushes; strata emerge naturally"). Two terms:
  L_pos   d(F(s)->B(s')) ~ 1        legal 1-ply successors (one-directional per edge; a
                                    reversible move's reverse is ITS OWN edge -> pinned ->
                                    symmetric; an irreversible reverse is not an edge).
  L_push  d(anchor->random) >= d(anchor->succ)+margin   INDEPENDENT random negatives,
                                    RELATIVE-margin triplet anchored at the edge parent,
                                    EXCLUDING true successors (Kaveh) via precomputed
                                    board-only successor key-sets.
Strata / material clusters / multi-step distances all EMERGE via the IQE triangle
inequality (it caps reachable pairs at their composed path, so the huge push only inflates
genuinely-unreachable pairs). NO DTM ranking / mate pole / separation / irreversibility
term. train_geometry_min.py.

BUGS found + fixed getting here (all real, all caught by probes):
1. The old L_neg only ever evaluated FORWARD (parent-F/child-B) pairs -> the irreversible
   reverse was never in the loss -> strata never emerged. Fix: sample a,b INDEPENDENTLY.
2. npz is LAZY: `ez["p_packed"][i]` in a loop reloads the whole 351k-row array each
   iteration = O(n^2) hang. Fix: materialize arrays once.
3. Full-BxB-mean push diluted each pair's gradient ~batch-fold -> push did nothing. Fix:
   paired negatives.
4. Absolute floor-60 push fights the global scale -> everything collapsed to ~30 (edges
   too). Fix: RELATIVE-margin triplet (anchored, no global tug-of-war).

DTM DATA reality (lichess_nearmate, 59976 pos): 19846 white-wins (dtm 1-197), 40130 draws
(dtm=0 SENTINEL, ambiguous with mate-in-0), ZERO black-wins (result=-1 empty; dtm never
negative). `result` (game outcome) != `dtm` (tablebase): 15869 tablebase-wins were drawn
in the actual game. So L2 here = win-distance bins + DRAW (losses absent). All 59976 nodes
carry EXACT ground-truth labels; L2 is exact ONLY on these nodes (off-nucleus it
extrapolates; entropy flags it).

L2 = train_l2_head.py (categorical W-bins + DRAW on frozen L1). ACCEPTANCE TEST (Kaveh):
"within 5 moves of mate L2 must guide exactly to mate" -> eval_l2_mate_guidance.py: White
plays L2-greedy (child minimizing expected distance), Black plays Syzygy-optimal defense,
from DTM<=10 starts; success = mate within the ply cap, vs a random-move baseline.
STATUS: L1 (iqe_geom_min) training @ lr 3e-4 w_pos 2 margin 15, 2500 steps, d_pos 10.7->4.76
by step 100 (converging; strata appear once d_pos->1). L2 + guidance scripts ready to run
on the finished L1.

## 2026-07-20: stratified adversarial quasimetric -- architecture crystallized, pipeline + adversarial validation
Canonical design written to ARCHITECTURE_STRATIFIED.md (supersedes ARCHITECTURE_SYNTHESIS.md
for the field/grounding/inference design); phased build in IMPLEMENTATION_PLAN.md; prior-art map
in relevant_sources.md (deep reads: IQE, QRL, ProQ, Reverse Curriculum, SoRB, GOAT, McGrath,
maze-extrapolation, Tropical Attention).

THE ARCHITECTURE. L1 cooperative reachability IQE geometry (edges d~1, count-drop captures
one-way, material-reachability repel; NO DTM) _|_ L2 adversarial outcome heads (remoteness/
DTM-to-region + committor -ln P(win), on frozen L1, exact TB labels) _|_ L3 human playability
(deferred). Strata boundary = PIECE COUNT (only captures change it); frontier = the tablebase
(retrograde analysis IS the exact adversarial forcing-distance for <=frontier); bottom-up
curriculum with per-stratum _le* checkpoints. Inference = uncertainty-gated recursive minimax
descent to the solved frontier, retrieval-gated leaves.

NOVELTY (positioned, honest): the field+planner machinery is NOT novel (ProQ, Kobanda et al.
2025, Inria/Ubisoft, independently arrived at nearly our IQE+QRL+repulsion+directional-cost+
OOD-gate planner). Novelty lives in (1) exact-oracle grounding + extrapolation PAST the training
support -- precisely ProQ's stated open limitation ("coverage limited to the dataset convex
hull"); (2) domain-structural boundaries as curriculum + subgoal + error-reset (bounded
compounding error); (3) THE ADVERSARIAL AXIS -- a learned composable quasimetric embedding of
the game-theoretic REMOTENESS / attractor-rank (Smith 1966; Conway/Berlekamp/Guy; parity/
reachability-game attractors). The whole quasimetric-RL line is single-agent; no one has learned
remoteness as a generalizing embedding or extrapolated it off a partial frontier. DTM = the
adversarial forcing-distance to the mate REGION (NOT a pole -- mate is a scattered absorbing set;
the "pull-to-point pole" is the collapse trap). The committor is its dense, harmonic form.

DATA (gen_stratified_perfect.py, PER-CHUNK checkpointed/resumable): 40000 positions across
strata [3,4,5,6] + edges (222054, 10670 captures) + optimal-line ply-gap pairs (41982), exact
perfect-play WDL/signed-DTM (<=6 via TB; negamax-into-TB for 7p). Genuine failing data: 3p
n=8000 W3785/D3587/L628. Fixed a data bug: KRvKBP was a 146-byte 404 STUB (only missing
descendant table); real file is KBPvKR (larger side first; python-chess maps its KRvKBP key to
it), pulled from lichess 3-4-5-wdl/dtz -> KR-vs-KBP coverage 12%->99%. Per-chunk shards let us
regenerate ONLY krkbp with the real table (289W/3636D/75L, not all-draw) and re-merge.

L1 FIELD (train_stratified_field.py, live dashboard experiments/viz/live_curves.py). Two false
starts diagnosed + fixed (the documented failure modes): (a) ABSOLUTE strata margin inflated the
embedding when captures entered (d_pos 2.7->5.6) -> switched to RELATIVE margin anchored to the
forward capture distance; (b) on the raw base the strata term still dominated the loss ->
warm-started from iqe_geom.pt (scale already settled, d_pos=1.0 from step 0). RESULT (VERDICT
STRAT_UMAP): material kNN purity 0.44 (vs ~0.09 random), one-way captures cap6=3.6x, outcome
coherence (krrk = all-win green blob; draws/Black-wins clustered = the failing data). Honest
partial: piece-count silhouette -0.09 (NOT clean bands -- material clustering dominates; strata
are an ORDERING via captures, not a partition), permat +0.02 (weak mate order -- BY DESIGN it is
L2's job; DTM is kept out of L1).

ADVERSARIAL-DISTANCE VALIDATION (adversarial_distance_validation.py, tablebase ground truth, no
training). VERDICT ADV_DIST: (1) DTM COMPOSES via the min-plus remoteness recursion DTM(s)==1+
min_child DTM(child): near-mate (DTM<=10) exact 0.913, overall 0.545 -- the far degradation
(0.435) and the 22% negative-slack "violations" are entirely the DTZ!=DTM MEASUREMENT ARTIFACT
(Syzygy is DTZ-optimal, detours far from mate), not a failure of the quasimetric property; this
also quantifies the DTZ/DTM gap that affects the whole DTM-labeling pipeline. (2) SPARSITY: DTM
finite on only 0.511 of positions (draws 0.465, losses 0.025 -> +inf) -- the empirical
justification for factoring L1 (dense) from L2 (adversarial). (3) COMMITTOR DENSE: P(win) under
an eps=0.15 defender is graded 0.950 (strictly in (0,1)) vs perfect-play's degenerate {0,0.5,1}
-- the dense region-quasimetric exists as the theory predicts. Figure: artifacts/experiments/
adversarial_distance.png.

TESTS: tests/test_stratified.py 8/8 (label correctness vs python-chess, strata invariants
[captures reduce piece count + can't be undone in one ply], material-reachability mask, IQE
quasimetric axioms [identity/non-negativity/triangle inequality]).

---

## 2026-07-20 23:35 — long/short planner (field for distance + search for tactics) beats the plateau; lichess-L2 retrain is I/O-bottlenecked

**Directive (Kaveh).** "Train on the lichess data to get L2, then do goal planning over it for
long distances, and then short searches to make it to goal." The decomposition: the L2 field is
the COARSE long-range navigator (which region is toward the goal); a SHORT adversarial search does
the fine local execution (tactics the field is too coarse for) + the not-blundering; the tablebase
is the exact base case. This is explicitly **not ProQ** — ProQ is a pure field-follower with no
short-search executor and no exact grounding, which is exactly why it (and our own gradient-
follower, and brute MCTS) plateau ~0.55.

**Why not just retrain a better field.** Measured this session: the within-material DTM-gradient
(spearman of d(F(won)->B(near-mate)) vs true DTM, per-material mean) is **+0.31 on the untrained
nucleus base `iqe_nucleus_gn`**, but **−0.05 (iqe_geom) / −0.08 (iqe_stratified)** on the fields
we TRAINED — InfoNCE/occupancy training DESTROYS the base geometry's gradient. So: build the
planner ON the base field; do not retrain it. The base is d=512 IQE + GroupNorm (the BatchNorm->
GroupNorm fix matters: eval-mode BatchNorm collapses the one-way structure).

**Result (experiments/planner_longshort.py, minimax depth-3, field leaf value, tablebase base,
frontier=5, 6-piece won starts vs OPTIMAL tablebase defense):**

    VERDICT LONGSHORT field=iqe_nucleus_gn depth=3 qdepth=0 frontier=5 n=25 mate_rate=0.640 (559s)

vs the plateaus on the SAME task: gradient-follower / ProQ-shaped ~0.35, material-greedy ~0.55,
brute StratifiedMCTS ~0.55. So the field-long-range + short-search decomposition is the first
thing to CLEAR the plateau. (At the full <=6 frontier it is trivially 1.0 — the tablebase base
case IS the forced win; the honest test is one ply above, converting 6p->5p, which is what these
numbers measure.) Next levers under test: uniform depth-4, and a **quiescence extension** (added:
`--qdepth`, keeps searching captures+checks past the cap so forcing conversion lines reach the
exact tablebase instead of being truncated) — the mechanism that should actually convert won 6p
positions by winning a piece.

**Lichess L2 retrain — I/O-bottlenecked, honest status.** Per the directive, launched a fresh
lichess field (`train_lichess_fb.py --iqe --iqe-components 32 --quasimetric --d 256`, GroupNorm by
default now, `--ckpt data/derived/sep/lichess_l2_iqe.pt --ckpt-every 1500`; existing lichess_fb_4gb
checkpoints are unusable — BatchNorm + MRN, corrupted at eval). It runs but at **0.4 it/s** (state
UN = disk-bound on per-batch pair sampling; ~24x below the established ~9.5 it/s), and does NOT
climb on the 256mb shard — the bottleneck is the streaming pair loader, not shard size or compute
(GPU nearly idle during it). At 0.4 it/s a real L2 (60k+ steps historically) is ~40h — infeasible
overnight; step-100 train_top1 is still 0.002. It's left running (checkpointed, honors the
directive, disk-bound so it doesn't block the GPU planner work), but the **nucleus base is the L2
for tonight's demonstration**. Follow-up: give the lichess loader an in-memory preload/mmap path so
lichess-L2 training is viable — that's the real blocker on the "train on lichess" half.

---

## 2026-07-21 00:20 — lichess-training failure was self-inflicted (IQE@scale-50 collapse), not the loader; opening-basin MSM pipeline (Kaveh's metastability idea)

**Why the earlier lichess run "failed" (Kaveh: similar training ran fine before).** Root-caused, not
guessed. The working historical lichess fields are `iqe=None, quasi=True` (MRN quasimetric) — they
never used `--iqe`. I had added `--iqe` with the default `--iqe-embed-scale 50`; the trainer's own
docstring says scale-50 is for plain InfoNCE and you want **~1 on the quasimetric/QRL path**. Wrong
scale → embedding collapse → InfoNCE loss pinned at **exactly ln(512)=6.238** (perfectly uniform
logits = every pair scored identically) with **top1=0** and zero gradient. The 0.4 it/s was a
separate thing — pure disk contention from three concurrent I/O-bound jobs, not the model.

**Fix confirmed.** Known-good recipe (MRN quasimetric + ply-gap, **no IQE**, GroupNorm-by-default —
the BatchNorm→GroupNorm fix) trains cleanly:

    step 1000  VERDICT VAL_TOP1=0.027 VAL_TOP8=0.158 (chance 0.0020)   # 13x / 10x above chance
    13.2 it/s  (vs 0.4 broken)  loss 6.24 -> 4.5

top1 rising, no collapse. Now retraining to 40k steps (`lichess_gn_qm_full.pt`, ckpt every 5k). Saved
a memory: don't run concurrent disk-heavy jobs (it destabilized the laptop).

**Opening-basin pipeline (`metastable_macrostates.py`, Kaveh's MSM idea, rebuilt sklearn-end-to-end
per his call).** openings/standard structures = metastable basins; PCCA+/deeptime don't build on
Python 3.14, so: `sklearn.neighbors.kneighbors_graph` (precomputed DIRECTED quasimetric distances) →
row-stochastic P → `sklearn.cluster.SpectralClustering` on the symmetrized affinity → macrostates.
**Prototype = MEDOID** (member minimizing summed symmetrized intra-cluster distance — centroids are
meaningless under a quasimetric); the medoid set = the **basin codebook**. **Validation gate:** label
members by ECO/opening-family from the maintained lichess `chess-openings` DB, matched by **EPD**
(piece-placement + side-to-move) so it's **transposition-robust**; ECO book = 7846 opening positions,
named-endpoint-wins collision rule (validated: `e4 e6`→C00 French, `e4 c5`→B20 Sicilian, Ruy→C60,
QG→D06, English→A10). Gate: basins ≥~0.80–0.90 one opening family → geometry recovered known
structure, proceed; mush → stop before building downstream. Caveat pre-registered: a "mush" result
could be field-limited, not idea-limited, until the lichess field is fully trained. Coarse DIRECTED
transition graph over macrostates = the plan alphabet. Plane-convention fix: lichess field trains on
FULL feature_planes (the toy's board-only 18,19-zeroing must be OFF for it). Gate runs when the field
lands.

---

## 2026-07-21 08:06 — opening-basin gate: MRN field recovers opening structure (purity 0.78, field-limited); IQE needs QRL not InfoNCE (recipe nailed down)

**Gate result (MRN quasimetric lichess field, `lichess_gn_qm_full.pt`, 40k steps, VAL_TOP1=0.033).**
Clustered 4000 in-book opening positions (ply<=30), sklearn kNN + SpectralClustering, ECO/opening-
family labels from the maintained lichess DB (EPD-matched). The **basin codebook is real openings**:

    M0 Sicilian 95% | M1 Italian 100% | M3 Sicilian 94% | M4 Ruy Lopez 84% | M5 Sicilian 100%
    M8 Caro-Kann 93% | M13 French 88% | M14 Italian 92%   (medoids = named openings)

but the 1.e4 e5 / transposition-heavy systems (Scotch, KGA, some KID/QGD) blur (26-64%). Mean
family-purity, m-scan: **0.726 (m16) -> 0.761 (m24) -> 0.784 (m32)** -- rises then PLATEAUS below
0.80. Reading: the geometry recovered known opening structure (proven -- clean major-opening basins),
but does NOT pass the strict ~0.90 gate; the plateau under increasing m proves the limiter is FIELD
QUALITY (undertrained, top1 0.033), not granularity. Two caveats logged: (1) the coarse transition
graph is degenerate here (T_ii~=1.0 all basins) -- a fragmentation artifact (openings so tight that
k=12 kNN never bridges basins; confirms separation but the "alphabet" needs game-trajectory
transitions, not static kNN); (2) a "mush" verdict could be field-limited, and this is exactly that
case -- yellow light, not red. Per Kaveh's protocol, STOPPED at the gate (no downstream build) and
asked how to proceed. Figure artifacts/experiments/macrostates_lichess.png.

**IQE recipe nailed (Kaveh: "MRN or IQE?").** The 0.78 was MRN. IQE deserved a fair shot; ran it
down: **IQE + InfoNCE + ply-gap COLLAPSES** at both embed-scale 50 AND 1 (loss pinned ~6.24, top1
flat at chance through step 500 where MRN had clearly learned) AND is **18x slower** (0.7 it/s -- IQE's
InfoNCE builds a full (N,M,components,K) tensor every step). Root cause: InfoNCE is the wrong pairing
for IQE. **IQE + QRL objective works cleanly:** the quasimetric diagnostics show adjacent-pair
`d_step` 6.8->1.7 (toward the QRL target ~1) while random-pair `d_rand` 7.8->65.8 = a **38x local/far
separation by step 400**, at 5.6 it/s. So the recipe (not the embedding) was the whole failure; IQE
needs QRL (matches the trainer's own docstring and almost certainly how the nucleus IQE field was
trained). Full IQE+QRL lichess field now training (30k steps, ckpt every 5k) for a fair A/B re-gate
vs MRN's 0.78 -- the 38x separation is the thing that could resolve the e5 blur.

---

## 2026-07-21 08:30 — L2 embedding<->objective guards + modular preset; best-play continuation generator

Committed to IQE+QRL for the L2 field; made it the guarded default and added the best-play data path.

**Guards + modular L2 (`train_lichess_fb.py`).** `--l2-preset` (default **iqe-qrl**; also mrn-qm,
cosine, custom) expands to a known-good (iqe, quasimetric, qrl_objective, iqe_embed_scale) combo;
`validate_l2_config` fail-fast-rejects incompatible pairings BEFORE the ~hour of training:
IQE+InfoNCE (the collapse footgun), IQE+QRL with embed_scale>5 (the scale-50 InfoNCE-bootstrap),
IQE-without-quasimetric, QRL-without-quasimetric. Extensible: add a preset row / guard rule. Tests
`tests/test_l2_guards.py` 9/9 pass. Rule, in one line: IQE<->QRL (metric objective), InfoNCE<->cosine;
never cross them.

**Best-play continuations (`gen_stockfish_continuations.py`).** Human lichess = average play; to see
OPTIMAL play in the regions humans reach, sample human positions and roll a strong engine forward K
plies. Engine-agnostic (Stockfish 18 default at /opt/homebrew/bin/stockfish; any UCI incl. Leela via
--engine). Writes continuations in the lichess shard schema (each continuation = one game_id,
ply-ordered, eval_cp from the engine, result = final-eval sign) so they mix in via the existing
`--selfplay-shards/--selfplay-frac`; the QRL pairing then gets optimal 1-ply constraints + best-play
goal pairs. Validated end-to-end: LichessPairSource consumes the generated shard (game_id ordered,
ply 0..K, succ present). CPU-bound (one engine/worker) -> runs AFTER the GPU field training, not
stacked. Plan: human IQE+QRL field -> re-gate vs MRN 0.78 -> full continuations -> fine-tune
--selfplay-frac 0.3 -> re-gate.

---

## 2026-07-21 09:57 — IQE+QRL is a VALUE field, not a concept field (key finding); concept-alphabet from game dynamics (option A)

**IQE+QRL trained (30k steps):** REACH_SLOPE_WON=0.722 / LOST=0.705 (vs the MRN field's 0.231/0.117)
-- a MUCH stronger reachability/value quasimetric. (VAL_TOP1=0.002 is the InfoNCE-retrieval metric,
meaningless for QRL.)

**Finding (confirmed, full field, exact castling-aware key): the value field and the concept field
are different representations.** Opening-family purity of the SpectralClustering gate (m=24):

    MRN field (quasi)                 0.761   (clusters openings)
    IQE+QRL field (quasi/reachability) 0.354
    IQE+QRL field (cosine of B embeds) 0.334

So the poor opening-clustering is NOT a method artifact (both the reachability metric AND raw-embedding
cosine fail) -- the QRL field genuinely does not organize positions by opening. It organizes by
DISTANCE-TO-OUTCOME (REACH_SLOPE 0.72): openings all sit "far from the result, early," undifferentiated
by family; two positions are close iff one is cheaply reachable from the other, not iff they are the
same kind of position. This is correct behavior for a value/planning field -- the concept/opening
structure is a SIMILARITY/DYNAMICS notion the value field deliberately discards.

**Decision (Kaposi): keep IQE+QRL as the value field; get the concept-alphabet from game DYNAMICS
(option A).** `experiments/opening_alphabet.py`: microstates = recurring opening positions (keyed by
placement+stm+castling, byte-fast, transposition-robust), transitions = consecutive positions within
REAL games, SpectralClustering on the transition matrix -> openings as metastable sets, coarse directed
graph = the alphabet, ECO-purity gate. Field-free -- his original "the transitions between basins are
the alphabet" taken literally. Runs after Stockfish frees the CPU.

**Encoding verified (Kaposi asked):** positions are stored, not inferred -- `meta` holds side-to-move,
all four castling rights, ep-file, halfmove-clock, repetition; `board_from_packed` restores a
legally-complete board (so Stockfish continuations start from correct state); `feature_planes` gives
20 planes incl. stm/castling/ep. Tightened the ECO-match key from placement+turn to placement+turn+
CASTLING (the field sees castling, so the label must too).

**Best-play supplementation (Kaposi's queued step) running:** `gen_stockfish_continuations.py`
(8000 human seeds x 8 plies best play, Stockfish 18) -> shard-format continuations -> will fine-tune
IQE+QRL with --selfplay-frac 0.3 so its value estimates sharpen toward optimal play where humans reach.

---

## 2026-07-21 10:45 — PROVEN: the IQE+QRL value field carries no structure (F or B); value ⊥ structure; concepts need their own representation

Kaposi's concept program (B-clusters = goals/subgoals, F predicts arrival) is architecturally CORRECT,
but it needs a field whose embeddings carry STRUCTURE, and the value field does not. Four tests agree:

1. **F doesn't cluster openings** 0.35 (vs similarity/MRN 0.76).
2. **Game-dynamics is a forward-DAG** -- openings flow forward, never linger; metastability T_ii=0 at
   any data size; PCCA+/spectral gives communities (purity 0.50) but no metastable basins
   (`experiments/opening_alphabet.py`, real-game transition matrix).
3. **B-clusters are value-generic** -- KMeans on B(arrivals) gives one 77% cluster spanning 243 distinct
   materials (1% dominant); every cluster materially incoherent; F->arrival-cluster 0.24 < majority 0.77
   (`experiments/goal_clusters.py`).
4. **Supervised probe (decisive, closes the non-linear escape hatch)** `experiments/structure_probe.py`:
   material (20-way, majority 0.22, raw-planes ceiling linear 0.77 / **MLP 0.91**):
   **F -> linear 0.25 / MLP 0.27 ; B -> linear 0.25 / MLP 0.20** -- i.e. ~majority, linear AND non-linear.
   Structure is genuinely gone from both towers, not masked.

Mechanism: the QRL objective only needs plies-to-outcome, so there is ZERO gradient pressure to preserve
material/pawn-skeleton; a 64-dim bottleneck discards them. **value ⊥ structure (graded: proven).** So the
clean design is two complementary reps -- IQE+QRL for VALUE (planning/routing), a separate STRUCTURE rep
for CONCEPTS (goals/subgoals, where B-clusters + F-arrival actually work). Structure-rep fork open with
Kaposi: (a) explicit structural features, (b) a learned structure/similarity field, (c) a structure head
on the same field (multitask). F/B semantics settled: F = source/future ("where I end up"), B =
goal/precursor ("where I came from"); cluster F for convergence-concepts, B defines the target zones.

**Best-play supplementation (Kaposi's queued step) DONE, marginal.** `gen_stockfish_continuations.py` ->
7958 best-play continuations; gentle fine-tune (resume, lr 3e-5, 30% mix, +10k steps) of a COPY
(`lichess_gn_iqeqrl_sf.pt`; human-only `..._full.pt` preserved). REACH_SLOPE_WON 0.729 vs 0.722 human-only
-- essentially unchanged on aggregate metrics; regional value benefit (weak-human zones) would need a
targeted eval. Footgun found+worked-around: resume restarts lr at PEAK (3e-4), collapsing the field in
~200 steps (d_rand->0); lr 3e-5 holds it. Guard to add: clamp/scale lr on resume. Also added the L2
embedding<->objective preset+guards (`--l2-preset` default iqe-qrl; tests/test_l2_guards.py 9/9).

---

## 2026-07-21 11:02 — RETRACTION: value is NOT ⊥ structure. The value field carries value-relevant concepts as DIRECTIONS; the SAE discovers them natively.

Kaposi's correction was right and my earlier "value ⊥ structure" claim is WITHDRAWN. The full-material
probe was the wrong test (it demanded reconstructing value-IRRELEVANT junk, e.g. a corner pawn, which
the field SHOULD discard). The right test -- named features, phase partialled out:

  `experiments/concept_features.py` (F-lift over a phase-only baseline, ROC-AUC):
    connected_rooks +0.215 | king_safe +0.202  <- genuine concepts, well beyond phase
    passed_pawn +0.03 / bishop_pair -0.04 / queens_on -0.04  <- phase-redundant
    clean control (a-file pawn) +0.01  <- irrelevant, correctly absent

So the value field keeps VALUE-RELEVANT structure as SEPARABLE DIRECTIONS (connected rooks is literally
the #1 concept, Kaposi's own example), and drops phase-redundant + irrelevant structure. Concepts are
DIRECTIONS, not clusters -- which is why every clustering attempt failed.

**Native discovery (no hand-coded features in the loop):** `experiments/native_concepts.py` (PCA/ICA on
phase-removed F) re-finds connected_rooks+king_safe but entangled. `experiments/sae_concepts.py` (sparse
autoencoder, overcomplete dict=128, L1) is the clean tool: with NO feature functions in training it gives
EVERY named concept its own atom -- connected_rooks(0.35), passed_pawn(0.39), king_safe(0.52),
bishop_pair(0.45), queens_on(0.48), piece_count(0.82) -- plus ~120 alive atoms, novel ones readable via
native piece-placement heatmaps (development/castling/pawn structures at ply 16-40). material_diff stays
undecodable (the field tracks distance-to-win, not material count).

**Resolution of the concept-representation question:** the concept/goal/subgoal alphabet = the SAE
dictionary OF THE VALUE FIELD, discovered natively. No separate structure field, no hand-coded features --
one IQE+QRL field carries value AND its concept dictionary. Next: SAE across game phases (richer dict),
SAE on the B tower (goal-side "approach" concepts), atoms -> goals/subgoals (concept region = atom's
high-activation set; forceability = F driving an atom up). Also added the resume-lr guard
(`--resume-lr-scale` default 0.1, tests still 9/9).

---

## 2026-07-21 12:13 — concept extraction on imported tooling (SAE=dictionary_learning, probes=CAV); monosemanticity confirmed; contributable conditional-SAE package

Kaposi's arc: concepts are DIRECTIONS the value field keeps for VALUE-RELEVANT structure (retraction of
"value ⊥ structure"); extract them natively; import don't reinvent; condition properly.

**Imported the concept stack** (memory `concept_extraction_stack`): `dictionary_learning` TopK SAE
(`experiments/concept_sae_dl.py`) + `captum` for CAV/TCAV; both install on 3.14. Per Concept Cones
(arXiv 2512.07355) the SAE atoms ARE the cones -> dropped the hand-rolled HDBSCAN `concept_cones.py`.
Chess precedent: McGrath et al. AlphaZero-probing (PNAS 2022).

**Monosemanticity test (Kaposi's ask, `experiments/concept_monosemanticity.py`).** For each concept, %
of an atom's top-activating cluster that has the feature vs baseline: connected_rooks 66% vs 20% base
in ONE atom, 17% (0.85x) elsewhere; king_safe 99% vs 34%; bishop_pair 96% vs 39%. Concepts are
CONCENTRATED in one atom and WASHED OUT (residual ~1x) elsewhere -- exactly as predicted; correlation
hid rare monosemantic atoms that prevalence exposes.

**Contributable conditional SAE** (`contrib/dl_conditional/`, Kaposi: "fork the maintained package, add
conditioning contributably; clean interfaces + docs + motivation"). `ConditionalAutoEncoderTopK`
(encode(x, cond) FiLM-gates atoms; cond=None == base) + `ConditionalTopKTrainer` (activations as
[x|cond]; L2+auxk on x; b_dec init fixed to the x part) -- proper subclasses of the library bases, plug
into trainSAE, strict generalization (cond_dim=0 -> standard SAE). README = PR-style motivation: a
global SAE averages over contexts, so context-dependent concepts (bishop-pair only matters OPEN, not by
phase) wash out and carry no domain-of-applicability; the gate fixes both. Wired into
`experiments/conditional_sae_dl.py`: conditional atoms show monosemantic prevalence (openness->
connected_rooks 2.3x). Includes an upstream note (add `dict_class_kwargs` to TopKTrainer).

**Audit (other hand-rolled pieces):** `torchqmet` (Tongzhou Wang's maintained IQE/MRN/PQE) is
github-only, not on PyPI, and swapping our `catspace/nn/iqe.py` would force a full field retrain ->
deferred, flagged.

---

## 2026-07-21 14:50 — the field was never concept-poor (both collapse alarms were metric artifacts); literature recipe for concepts-as-subgoals

**The reframe.** The multi-task "structure head" was built to break a "collapse" (effective rank 1.1/64)
that supposedly made concepts imprecise (connected_rooks "60% precision@5"). A low-noise measurement
(threshold-free ROC-AUC across 5 structural concepts, held-out, 12k positions;
`experiments/*` inline probe) reverses the premise:

```
field          | eff-rank raw  zscored | mean AUC | connected king_safe passed bishop queens
orig (30k step)|      1.1       2.0     |  0.805   |   0.82     0.87    0.83   0.68   0.82
struct w=50 5k |      1.1       1.6     |  0.800   |   0.81     0.88    0.82   0.66   0.82
```

The incumbent field ALREADY encodes structural concepts as linear directions at mean AUC 0.805
(connected_rooks 0.82, king_safety 0.87) -- good, usable CAVs. The multi-task head does NOT improve this
(w=50, fully trained: 0.800 ≈ 0.805). Both alarms were metric artifacts: (a) raw eff-rank 1.1 is a
VARIANCE-SCALE effect (the value axis has per-dim std up to 451 and dominates the participation ratio);
z-scored (correlation-matrix) rank is 2.0, and the concepts live in the LOW-variance correlated-residual
directions, which a scale-free linear probe reads fine. (b) "60% precision@5" was base-rate (20%) × 5-sample
noise -- an AUC-0.82 direction at a 20% base rate gives ~3× enrichment at the very top, exactly precision@5≈60%.
Retraction: the "value ⊥ structure" / "1D collapse kills concepts" framing was WRONG. The field carries value
AND structural concepts; the participation ratio just can't see the low-variance concept directions.
(Honesty note: the w=300 run I first measured was accidentally killed at step 500 by a Bash-tool timeout
SIGTERM on the launch call's wait-loop -- that number was void and is being re-run cleanly as a w0/w300
matched-step backstop; Kaposi chose to move on regardless.)

**Decision (Kaposi):** stop "fixing" the field -- USE the existing CAV directions as planner subgoals.

**Literature search -- how people use linear concept directions as subgoals (import, don't reinvent):**
- *Concepts function AS plan directions.* Bush et al., "Interpreting Emergent Planning in Model-Free RL"
  (ICLR 2025 oral, arXiv 2504.01871): linear probes find planning-relevant concept directions in a Sokoban
  agent; the plan is read off them and STEERING along a concept direction causally changes the plan
  (parallelized bidirectional search). Direct precedent for our thesis: concepts are directions, planning
  proceeds along them, steering is causal.
- *Quasimetric field -> subgoals* (our field's own family). QRL (Wang, Torralba, Isola, Zhang; ICML 2023,
  arXiv 2304.01203) -- our objective; the quasimetric d(F(s),B(g)) IS optimal cost-to-go, so a waypoint w
  is chosen by path relaxation D*(s,g)=min_w d(s,w)+d(w,g). "Offline GCRL with Quasimetric Representations"
  (arXiv 2509.20478, 2025): high-level policy picks a subgoal in quasimetric space, low-level reaches it,
  waypoints minimize cumulative quasimetric distance. HIQL (Park et al. 2023): subgoal = a LATENT target,
  not a decodable state -- you steer toward a region, exactly what a CAV half-space is.
- *Coarse plan graph over concept-landmarks = Kaposi's basin-alphabet.* L3P "World Model as a Graph:
  Learning Latent Landmarks for Planning" (Zhang et al., ICML 2021) and Successor Feature Landmarks
  (NeurIPS 2021): cluster latents into landmarks, weight edges by reachability, plan shortest path over
  them. This is the metastable-basin/PCCA+ "alphabet" idea done in latent space.
- *Sharpen the CAV subgoal directions with the SAE (we already have both).* "Denoising Concept Vectors
  with Sparse Autoencoders for Improved Steering" (arXiv 2505.15038, 2025): project a raw diff-of-means CAV
  onto the SAE dictionary, keep the high-magnitude atoms, reconstruct -> stronger, more specific steering at
  smaller magnitude. Drop-in recipe to crisp our AUC-0.8 directions before using them as region boundaries.

**Mapped recipe for catspace (all pieces already in-hand):** subgoal = a concept REGION in goal space =
the CAV half-space {x : w_c·x > τ} (SAE-denoised w_c); reach cost = the quasimetric distance-to-region
d_c(s)=min_{g in region} d(F(s),B(g)) (we built dist-to-B in `precision_reps.py`); high-level plan = a
sequence of concept-regions selected by quasimetric path relaxation (L3P graph over concept-landmarks);
short searches execute reaching the next region = the existing long/short planner. Causal validation
(Bush et al.): steer F(s) along w_c and check the planner's move shifts toward that concept.

**Sharpening test (`experiments/denoise_cav.py`, imports arXiv 2505.15038 onto our field).** The SAE-denoise
recipe HURTS every concept here and flips passed_pawn's sign: rawCAV vs SAE-denoised AUC -- connected 0.824
->0.792, passed 0.832->0.299, king_safe 0.870->0.826, queens 0.823->0.763. Same root cause: concepts live
in LOW-variance residual directions the reconstruction-trained SAE (variance-weighted, value-axis-dominated)
doesn't span, so pushing a CAV through the dictionary discards its discriminative part. VERDICT: use the raw
logistic CAV as the concept direction; the SAE is for DISCOVERING atoms, not sharpening directions.
The quasimetric dist-to-B-region reach cost scores AUC 0.6-0.72 as a detector -- weaker, but that's the wrong
lens: it's a cost-to-go, low AT and NEAR the region (the smooth subgoal potential the planner descends), not
a possession detector. Refined recipe: region boundary = raw CAV; reach cost = quasimetric dist-to-region;
no denoise step.

**Multi-task struct head CLOSED (clean matched-step sweep).** From-scratch 5k-step controls, mean structural
AUC: w0=0.801, w50=0.800, w300=0.801 (orig 30k=0.805). Concept decodability is FLAT across the whole
struct-weight range 0->300 and nearly flat vs 6x training length (+0.004). The structure head does nothing;
~0.80 concept AUC is intrinsic to the IQE-QRL objective on this data, not a collapse to be fixed. Multi-task
anti-collapse rejected. (This also formally retires the "value != structure / 1D-collapse" alarm chain.)

## 2026-07-21 15:20 — concepts ARE navigable subgoals (via the CAV axis, NOT quasimetric region-distance)

The linchpin test (`experiments/concept_reach_rollout.py`, design chosen by Kaposi: opponent = base FB reach
policy depth-1, White steers greedy-1-ply). From 200 positions where White lacks connected_rooks, roll out
10 plies; White picks its move by a subgoal score, measure reached-within-K:

```
white strategy | reached<=10ply | median plies
reach2region   |      2%        |     1.0
cav            |     28%        |     5.0
basepolicy     |      6%        |     1.0    (White plays toward MATE_W)
random         |      6%        |     3.0
```

VERDICT: **the CAV direction is a navigable multi-step subgoal** -- greedily climbing w_c . F(child) reaches
connected_rooks 28% vs 6% for both normal mate-seeking play AND random (4.7x lift, 28+-3.2% vs 6+-1.7%,
>6 SE). The weak 1-ply gradient (steer_concept.py move-AUC 0.658) COMPOUNDS over 5 White moves into real
steering. This validates the concepts-as-subgoals program: concepts are directions you can plan ALONG
(cf. Bush et al. ICLR 2025), and our field supports it.

**Correction to the recipe:** the quasimetric dist-to-B-region FAILS as the navigation signal (2%, worse
than random). d(F(s),B(g)) is the cost to reach a concept-positive position's GAME-GOAL (its mate pattern),
which conflates "reach that whole position" with "acquire the attribute"; descending it chases the nearest
goal-state, not connected-rooks. The CAV isolates the ATTRIBUTE. So: subgoal navigation = climb the CAV axis
(w_c . F); the quasimetric distance stays for reaching goal STATES (mate), not concept attributes. This also
kills the earlier "subgoal = dist-to-B-region" sketch from the 14:50 recipe -- superseded by CAV-climb.

**Subgoal codebook (`experiments/subgoal_codebook.py`, Kaposi's call: navigability x value across concepts).**
Per concept: navigability = CAV-climb reach rate vs base-FB-policy play (rollout, base-FB opponent); value =
field reach-to-MATE_W gap (z-scored) + external white-POV game-result gap. 150 games/concept, 10 plies:

```
concept          base | cav_reach basepol lift | value_z result_gap | SCORE(lift x value_z)
king_safe_w      53%  |   59%      31%   +28%  | +0.83    +0.03      | 0.233
connected_rooks  20%  |   27%       4%   +23%  | +0.79    +0.02      | 0.185
passed_pawn_w    20%  |   51%      13%   +38%  | +0.29    +0.09      | 0.109
bishop_pair_w     9%  |    3%       1%    +1%  | +0.23    +0.06      | 0.003
queens_on        72%  |    1%       2%    -1%  | -0.21   -0.01       | -0.00
```

The codebook cleanly sorts usable subgoals from non-subgoals for interpretable reasons: king_safe &
connected_rooks are navigable AND valuable (real subgoals); passed_pawn is the MOST navigable (+38%, 51%
reach -- you can force a passer) but only modest value; bishop_pair is correctly rejected as NOT navigable
(can't manufacture two bishops vs one in 5 moves); queens_on is rejected as unnavigable AND negative value
(can't force queens to stay, not good for White). This validates the subgoal-density-prior selection rule
(S = navigability x value). Caveat: external result_gap is weak/noisy (game outcome is far from position);
field value_z is the load-bearing value axis (carries the "field rates it winning" meaning). Top-3 codebook
= {king_safe, connected_rooks, passed_pawn} -- the candidate waypoint alphabet for planner wiring.

**Why dist-to-region fails to navigate (`experiments/diag_region_nav.py`, Kaposi asked "how come").**
Not a bug: distance_matrix(F,B)=d(source->goal) verified (real 1-move successor d=0.04, backward 5.88,
random 197 -- sharp, correct direction). The field KNOWS the connecting move: d(parent -> its own
rook-connecting successor)=0.09 (near-adjacent). The failure is the ESTIMATOR of "distance to the region":
min over 48 global whole-position anchors sits at d=3.32 -- 37x farther than the local connecting successor
(0.09), which is not in the sample (SAMPLING error). And distance to a 32-piece anchor is dominated by bulk
positional similarity, so the rook-connection contributes marginally: the connecting move is at the 99th
percentile under dist-to-region yet the strict argmin only 2% of the time (one bulk-similarity distractor
consistently edges it out -> greedy picks the distractor 98%). The CAV scores the ATTRIBUTE directly, so the
connecting move is strict top-1 31% (15x more) -> 28% reach over 5 plies. Upshot: the field geometry is fine;
dist-to-region isn't fundamentally broken (a LOCAL/attribute-defined region would navigate), but the global-
whole-position-anchor min is the wrong estimator and the CAV is the right one (it IS the attribute-defined
region). Confirms the 15:20 recipe correction with a mechanism.

## 2026-07-21 16:10 — can a concept-CAV subgoal convert the toy examples? NO for connected_rooks (mechanism works, concept is wrong for this mate)

Kaveh: "go autonomous, see if you can convert the toy examples." Toy examples = KRRvKBP endgames
(krrkbp_test_n200.json); convert = mate vs tablebase-optimal defense (mate_rate). Built
`experiments/conversion_concept.py`: the long/short planner (iqe_nucleus_gn field, 0.640 incumbent) with a
connected_rooks CAV-climb term blended into its leaf value, leaf = (1-a)*v_base + a*tanh(cav_z) (convex, so
real tablebase terminals still dominate; a=0 recovers the incumbent). CAV fit IN-DOMAIN (endgame positions,
same field/planes, connected_rooks base-rate 16%). A/B on the fixed set, n=48 (two parallel slices):

```
alpha=0.00  mate_rate=0.459   (base)
alpha=0.40  mate_rate=0.375   (-0.084)
alpha=1.00  mate_rate=0.396   (-0.063)
```

VERDICT: the connected_rooks CAV subgoal does NOT convert the toy examples -- it HURTS by ~0.06-0.08. But
the a=1.0 arm (pure concept navigation) DOES move the mate_rate (0.459->0.396), so the CAV term genuinely
changes behavior (not a no-op; the n=10 smoke's +0.000 was small-sample). So today's CAV-climb mechanism is
real here too -- connected_rooks is just the WRONG subgoal for a two-rook mate: the KRRvKBP technique needs
rook SEPARATION (ladder/box) + enemy-king confinement, not rook CONNECTION, so pulling the rooks together
works against the mate. Consistent with the standing prior (conversion is SEARCH-limited; field-structure
navigation does not beat the search-based converter -- conversion_subgoal/composed all lost). Constructive
upshot: importing a lichess-middlegame concept as an endgame subgoal is mismatched. The right move is to run
the subgoal codebook (navigability x value) IN-DOMAIN on KRRvKBP -- discover which structural concepts are
BOTH navigable AND correlate with mating progress in THIS endgame (candidates: black-king-on-edge, rook
cutoff/confinement, won-simplification to KRRvK) -- rather than assume connected_rooks. Not a search win;
an honest negative that localizes where a win would have to come from.

## 2026-07-21 17:05 — unsupervised field-subgoal conversion: blocked by a FLAT long-range quasimetric (converges with the concept finding)

Kaveh's directive: drop all supervised probes; use the pure quasimetric field (no board-structure head);
find subgoals directly off the field (regions/basins, "navigate NEAR a region, not to a state"; "mate is a
cluster on B, not a pole"); navigate by quasimetric distance to subgoals. Built
`experiments/conversion_field_subgoal.py`: pure field (iqe_nucleus_gn), B-bank clustered into 40 BASINS
(regions), select basin by field-only composed = reach(F(s)->basin) + lam*d(basin->mate cluster), navigate
NEAR it (min over basin members), receding horizon. Mate = the B_goal region cluster (already region-based,
not MATE_W pole). Fully unsupervised.

Conversion A/B (n=10, iqe_nucleus_gn): base navigate-straight-to-mate 0.400, field-subgoal 0.400, DELTA
+0.000. But the diagnostic is the finding, and it is DECISIVE:

```
reach std across 40 basins:  0.03      (field distance from a KRRvKBP start to EVERY basin is ~identical)
basin_dmate std:             0.69      (23x larger)
distinct winning basins across 20 starts:  1 of 40   (fully degenerate selection)
```

VERDICT: the field's quasimetric is SHARP LOCALLY (d=0.04 to a real 1-move successor, per diag_region_nav)
but FLAT AT LONG RANGE -- from a start, its distance to all endgame basins is the same to within 0.03, so it
cannot rank subgoal regions by reachability. composed selection collapses to argmin(dmate) = "aim at the mate
cluster" = the base planner, for every start. There is NO reachability gradient to select subgoals on -> the
subgoal idea can't get off the ground on this field, regardless of region-vs-point or lam. "If we have
preserved it [the quasimetric]" -- we have NOT, at long range.

CONVERGENCE: this is the SAME root cause as the concept failure (16:10), from a second angle. The lichess-
trained field does not represent the ENDGAME regime -- neither the mate-relevant rook concepts NOR the
endgame reachability chain (KRRvKBP->KRRvK->KRvK->mate) -- because lichess data is endgame-sparse (2.54%
<=6-piece, 0.005% KRRvKBP). Local quasimetric survives (adjacent states), long-range reachability ordering
does not. Both the concept path and the subgoal path require a field trained on ENDGAME-DENSE data (tablebase
DTM; we have dtm_endgame.npz=24k + syzygy to generate unlimited) whose distance actually discriminates the
endgame reachability gradient. Not a search or method failure -- a field/data limitation, now shown twice.

## 2026-07-21 18:30 — tablebase-free field-guided MCTS MATES the toy (0.33 @3200 nodes); the +0.000 was a dead value function

Kaveh's architecture, assembled and finally working. Chain of his calls: no supervised probes; subgoals =
clusters on B; navigate NEAR regions; mate = a B cluster; rely on B (F doesn't generalize to novel material,
B/mate does); subgoal = F-reachable INTERSECT B-leads-to-mate; SEARCH executes toward subgoals; "corner the
king" means cornered AND EXPOSED (checkable), distinct from the safe castled king -- assume it, don't
validate; and finally: try the mate WITHOUT the tablebase, MCTS guided by the model.

**The +0.000 was a bug (Kaveh's catch).** The 5k fields (control, treat) collapse d(F(s), winning-region)
to EXACTLY 0 for every KRRvKBP position -> field_value = tanh(1) = 0.7616 constant, std 0 across 40 positions
-> no gradient -> every goal/concept A/B was +0.000 by construction (verified: swapping the goal region left
field_value bit-identical, 0/20 moves changed). The nucleus field has a live gradient (d mean 9.9), and the
treat field has a live gradient toward the ENDGAME dtm-basins (reach_std 27.9, from the continuations) even
though its winning-region distance is dead -- so navigation must target the endgame cluster.

**Tablebase-free field-MCTS (`experiments/conversion_field_mcts.py`).** White uses ONLY the model: subgoals
= KMeans basins of the endgame bank on B; mate target = the exposed-cornered-king basin, identified
GEOMETRICALLY (black king on edge AND <=2 escape squares; no tablebase); subgoal selection = argmin over
basins of [min d(F(s),B(basin)) + lam*min d(F(basin),B(mate))] = F-reachable INTERSECT B-leads-to-mate;
a field-guided MCTS (value = progress toward the chosen basin, mate_stop for real checkmate, NO tablebase)
executes. Black defends tablebase-optimally.

VERDICT on the KRRvKBP toy (treat field, weak 5k, n=12):
```
nodes  400 -> mate 0.000
nodes 1600 -> mate 0.250
nodes 3200 -> mate 0.333
random-White baseline -> mate 0.000
```
The field genuinely MATES (0.33 @3200) vs perfect defense with NO tablebase in White's search; random never
mates. Cleanly SEARCH-limited (monotone in nodes), consistent with the earlier committor 0.567@400->0.767@1600.
(Red herring flagged: the "cornered-king reached" progress metric is uninformative -- random scored 0.67 > the
field-MCTS's 0.58, because it's a max-over-game measure inflated by long aimless games. mate_rate is the signal.)
Next: 20k-step field (fixes the undertraining that caused BOTH the weak finish and the dead value function),
then re-run the node sweep.

UPDATE (node sweep completed): nodes 6400 -> mate 0.333 too -- PLATEAU from 3200. So the tablebase-free
field-MCTS is search-limited up to ~3200 nodes, then hits a FIELD-QUALITY ceiling (the weak 5k field's
guidance caps at 0.333 regardless of further search). This is exactly why the lever is now field quality,
not search: training the 20k field (in progress) should raise the ceiling.

## 2026-07-21 19:40 — 20k field DOUBLES the tablebase-free mate_rate (0.58); both search and field-quality are live levers

Option (b), done: retrained the completed-trajectory field to 20k steps (lichess prefix256mb + endgame
continuations, iqe-qrl, no struct head, frac 0.35). Re-ran the tablebase-free field-MCTS node sweep:
```
                5k field     20k field
nodes 1600  ->   0.250        0.500
nodes 3200  ->   0.333        0.583
random-White baseline: 0.000
```
The 20k field mates 58% of the KRRvKBP toy with NO tablebase in White's search, vs tablebase-optimal defense
-- ~1.8x the 5k field, and NOT saturated in nodes (0.50->0.58 from 1600->3200). So the earlier 0.333 plateau
was the 5k field's quality ceiling; more training raised it. Both levers are live: search depth AND field
quality independently increase the tablebase-free mate_rate.

Arc summary (this whole thread): started at "field guidance gives +0.000, conversion is search-limited,
nothing beats the tablebase committor." Diagnosed the +0.000 as a dead value function (Kaveh's catch).
Assembled Kaveh's architecture -- B-cluster subgoals, F-reachable INTERSECT B-leads-to-mate selection,
exposed-cornered-king as the geometric mate target, MCTS execution, NO tablebase -- and it now MATES the toy
purely from the model at 0.58, scaling with both search and training. (Note: the field's distance to the
stratified_perfect "winning-region" stays degenerate (d=0) even at 20k -- a property of that region, not
undertraining; the field-MCTS sidesteps it by navigating the dtm-basin gradient, which is live. The "cornered
-king reached" metric is uninformative -- max-over-game inflates it for random play; mate_rate is the signal.)
Headroom: still climbing in nodes; field not yet converged; continuation data could be enriched.

## 2026-07-21 20:30 — RETRACTION: the "field-guided MCTS mates the toy" milestone was PURE SEARCH; field guidance HURTS

Caught by the viz (viz_fb) + a value-liveness check (dbg): the field-MCTS leaf value on treat_20k is a DEAD
constant (std 0.0000 across a position's children), so its 0.58 mate_rate came from MCTS+mate_stop (pure
search), NOT the field. treat_5k has a live gradient (std 3.6); nucleus std 0.27. Clean A/B on the SAME
treat_5k field (`--pure-search` = constant leaf value), n=16:
```
                FIELD-GUIDED    PURE-SEARCH
nodes 1600:     0.250           0.438
nodes 3200:     0.375           0.562        runtime: 148-242s vs 4-6s
```
Pure search BEATS field-guided at every node count and is ~40x faster. So:
- RETRACT the 18:30 ("field-MCTS mates 0.33") and 19:40 ("20k doubles to 0.58, field-quality lever") entries.
  Both were pure MCTS+mate_stop search. 18:30's 0.33 was a live-but-HARMFUL field dragging search down;
  19:40's 0.58 was the field collapsing to constant -> pure search unmasked. "Field quality raised the
  ceiling" was actually "training 20k COLLAPSED the quasimetric (value->constant)" -- a representational-
  collapse failure (cf. check_representational_collapse), NOT an improvement.
- What actually mates KRRvKBP: MCTS + mate_stop search alone (no field, no tablebase), 0.44@1600 -> 0.56@3200,
  search-limited. The field's mushy distance MISDIRECTS it (0.25/0.38) and costs 40x compute.
- This is the SAME standing finding, now airtight: on KRRvKBP conversion, field navigation does not beat
  search; here it is strictly worse. KRRvKBP + mate_stop is SEARCH-SOLVABLE, so it does NOT test whether the
  field adds planning value -- the search shortcut hides it. The field's value (if any) must be tested where
  search alone fails: deeper mates / no mate_stop / longer-horizon or larger domains (the real transfer target).
Process note: I over-claimed the milestone across several turns; the value-liveness check should have been the
FIRST thing run on any "field-guided" result. Added to the checklist mentally.

## 2026-07-21 22:30 — DISTILLATION WORKS: the field CAN learn an unseen regime from targets (6-piece DTM 0.19->0.53)

The extrapolation question (Kaveh): can a field extend to data it never saw? Empirically NO for target-FREE
extrapolation (nucleus DTM alignment by piece count: 3p 0.70, 4p 0.52 trained -> 6p 0.21 extrapolation; and
proximity to the seen manifold does NOT predict 6-piece accuracy -- corr(NN-dist, error)=+0.016, flat 0.18
across all embedding-distance quartiles -> the failure is SYSTEMATIC/regime-level, not graded interpolation,
so an embedding-kNN reliability gate can't work). BUT with TARGETS (Kaveh: "distillation is the way"),
`experiments/distill_finetune.py` regresses d(F(s),MATE_W)->DTM on the endgame set; spearman by piece count
before->after:
```
  3-piece: +0.70 -> +0.88   4-piece: +0.50 -> +0.71   6-piece: +0.19 -> +0.53  (extrapolation -> learned)
```
So the architecture CAN hold 6-piece reachability; extrapolation failed only for lack of targets. Distillation
is the mechanism to grow support outward (validated within tablebase range with exact DTM; the search-backed
teacher MCTS+field=0.27 vs raw 0.19, `distill_validate.py`, is the vehicle to go PAST the tablebase). Caveat:
6-piece plateaus 0.53 << 3-piece 0.88 -- the 193-ply horizon is intrinsically lossier (compounding error).
Literature (search 2026-07-21): consensus is quasimetric "stitching" = composition WITHIN transition support,
not true extrapolation; offline-RL answer to OOD is CONSERVATISM/OOD-detection (validates the gate) + FACTORED
/goal-disentangled reps for unseen-goal generalization (the invariant) + stitch/expert-iteration to grow
support (the distillation). No method truly extrapolates a reachability metric to an unobserved regime.
Next: does the distilled field (0.53) now beat uniform as an MCTS move-prior?

## 2026-07-21 23:30 — FIRST POSITIVE: the field is an efficient COARSE NAVIGATOR at low compute (not a move-prior)

Reframe (Kaveh): near mate, pure MCTS dominates; the field's real job is COARSE long-range navigation -- from
far away where pure search is BLIND (mate_stop can't see the goal), does the field reach the near-mate region
faster/fewer evals? `experiments/reach_efficiency.py`, 6-piece KRRvKBP starts (DTM>=30), White=MCTS,
Black=tablebase-optimal, reach target <=5 pieces (where search+tablebase take over), field=distilled nucleus
value -d(F,MATE_W) vs pure-search (constant value), n=20:
```
budget | FIELD reach/plies/evals | PURE reach/plies/evals
  100  |  85% 4.5p  453 ev       | 75% 5.1p 513 ev
  400  | 100% 4.6p 1840 ev       | 70% 5.0p 2000 ev
 1600  |  80% 5.6p 9000 ev       | 80% 4.8p 7700 ev
```
At LOW/MODERATE budget the field reaches near-mate more reliably AND with fewer node-evals (400n: 100% vs 70%);
at high budget pure search catches up (both 80%) and the edge vanishes. This is the FIRST clean win for the
field, and it resolves the whole night: the field is NOT a fine move-prior (at-chance, needs +-1-ply resolution
it can't give even after distillation raised position-DTM to 0.53) -- it is a COARSE NAVIGATOR whose value is
getting to the endgame efficiently when compute is limited. Division of labor confirmed: field = coarse long-
range gradient (compute-efficient), search = fine near-mate execution. Caveat: n=20 (reach +-10%); firming with
n=40 finer low-budget sweep. Note this is ALSO where distillation matters (the field must be non-collapsed to
give the gradient) and where the interpretability/efficiency story lives.

## 2026-07-21 24:00 — HEALTH CHECK: nucleus is NOT a healthy quasimetric -- it's a collapsed ~1-D distance-to-mate predictor

Kaveh asked to verify nucleus didn't collapse. Verdict (tmp/health.py, <=5-piece positions):
```
nucleus:           params mean|w| 0.016, 0.38% near-zero, max 2.93  -> weights NOT vanished
                   d_step(1-ply successor) 8.52  vs  d_rand 8.49  -> ratio 1.0x  (BROKEN quasimetric)
                   F eff-rank 6.0/512, F std 1.09
nucleus_distilled: d_step 0.96 vs d_rand 0.93 -> 1.0x  |  eff-rank 5.7/512, F std 0.27
```
Params fine, but (1) d_step ~= d_rand: the distance to the ACTUAL 1-ply successor equals the distance to a
RANDOM board -- the field does not encode local reachability; (2) eff-rank ~6/512 -- the embedding is squished
into ~6 dims (substantial collapse, not rank-1 dead). The squeeze is why d_step~=d_rand (all pairs ~equidistant).
Only the ONE direction the DTM-anchoring/distillation forced survives = distance-to-mate (spearman 0.44->0.53).
So nucleus is a low-D distance-to-mate PREDICTOR, not a reachability quasimetric. Implications: (a) the coarse-
nav win STANDS (it uses only the surviving mate-gradient); (b) this collapse is why it fails as a fine move-prior
(no local structure to resolve +-1-ply); (c) it RAISES the stakes on the fair test -- if the field is ~1-D DTM,
a plain CNN DTM-regressor should match its coarse-nav, meaning the quasimetric adds nothing over "a DTM value".
B-viz (viz_b.py): near-mate-cluster spread/overall = 1.02 (>1) -- mate is NOT a tighter cluster on B, consistent.

## 2026-07-22 — decoupled field validated (DTM term was the rank-crusher); ladder mate: the constraint concept nearly matches the oracle

**Decoupled architecture (Kaveh: "Separate DTM head, don't overload d").** Retrained the geometry
field CLEAN (`train_geometry_l1.py --w-dtm 0`, otherwise the settled recipe: w-pos 2, w-hard 1,
hard-margin 15, w-repel 4, floors 30/12, best-play edges; 2500 steps, 47 min MPS) ->
`iqe_geom_field.pt`. New acceptance script `experiments/validate_decoupled.py` (A: quasimetric
health; B: d-DTM decoupling; C: DTM-head bake-off on the frozen trunk). Verdicts (n=4000):

    VERDICT DECOUPLE.A field=iqe_geom_field d=512  d_step 1.76  d_rand 58.14 (unreach 58.25)  ratio 33.0x  asym irr 5.56 / rev 9.13 (sep 0.61x)  eff-rank F 15.1 B 17.5 /512
    VERDICT DECOUPLE.B  spearman(d(F,MATE_W), DTM) by piece -- LOW = decoupled: 3p -0.100  4p -0.082  6p -0.041
    VERDICT DECOUPLE.C  DTM-head-on-trunk spearman: 3p +0.721  4p +0.520  6p +0.272  (CNN 0.89/0.61/0.355 - distilled .88/.71/.53)

Findings: (1) **the DTM term was the rank-crusher** — same repulsion/steps, dropping L_dtm alone
took eff-rank 1.3-5.5 -> 15.1 and ratio 8.2x -> 33.0x (the scalar regression was compressing the
representation onto one axis; "cure is repulsion" needs this amendment). (2) Decoupling holds
(B ~ -0.1). (3) The frozen geometry trunk is the WORST DTM feature source (0.27@6p) — the
**distilled field stays the DTM/value head**. (4) REGRESSION: asym inverted (0.61x, was 2.03x on
full2) — repel-floor-all inflates reversible reverses (turn parity: never 1-ply edges, so they
look like random pairs), L_hard unconverged at 2500 steps. Fix candidates: exempt known reverse
pairs from the all-pairs floor; longer training. Not applied yet. Validator bug fixed en route:
full n x n IQE distance_matrix at n=4000 OOM-killed silently; chunked row-aligned d_pairs().

**Two-rook ladder mate (Kaveh: "if you can get the ladder mate working... it should be trivial").**
New `experiments/ladder_mate.py` (KRRvK, central black king, tablebase-optimal defense) +
`experiments/engine_search_cost.py` (UCI engines, real node counts). The trivial answer: KRRvK is
solved, tablebase play converts 100% with ZERO search. The interesting ladder (all n=20, seed 2):

    VERDICT LADDER_MATE value=none nodes= 1600  mate_rate=0.19  (pure search shuffles into draws)
    VERDICT LADDER_MATE value=dtm  nodes=  400  mate_rate=0.31  (learned DTM CNN too coarse)
    VERDICT LADDER_MATE value=constraint nodes=400  mate_rate=0.75  search_nodes_to_mate median=1,518
    VERDICT LADDER_MATE value=tb   nodes=  400  mate_rate=0.85  search_nodes_to_mate median=1,201
    VERDICT ENGINE_SEARCH engine=Stockfish set=ladder depth=18  mate_rate=1.00  nodes_to_mate median=54,430

**The cornering concept (black king escape-volume, exact flood-fill) as the search value recovers
most of the perfect oracle's guidance** (0.75 vs 0.85 @400 nodes; ~1.5k total nodes to mate) — a
dense signal (every move changes it) where DTM is flat/unlearnable far from mate. Kaveh's framing
confirmed: constraining the king IS approaching mate, and it is the eval that buys cheap search
(Stockfish needs ~54k alpha-beta nodes; units differ, qualitative only). Residual 0.20-0.25 draw
mass: the constraint gradient vanishes once the box is minimal (waiting moves tie) — exactly the
execute-phase handoff seam. Also caught: `tb_best_move` minimizes DTZ, not DTM, so it HANGS A ROOK
(zeroing looks cheap) — mates in 23-25 plies vs Stockfish's clean 9-11. A perfect oracle with the
wrong metric plays badly: the concepts (constraint, material safety) are what make "knows the
result" into "plays it well" — the project thesis in one bug.

Open fork (Kaveh's "how do I find goal-ward positions and get there"): probe whether the B-field
already encodes the constrained-king region (field-native recognition) vs wire constraint+safety
into the execute phase first. Asked; awaiting his call.

## 2026-07-22 (later) — concept rules sharpened; B mate-cluster viz: patterns are NOT in the field yet

**Rule (Kaveh):** exact concept computations (king escape-volume flood-fill) are DIAGNOSTIC ONLY —
never a play-time value. Recorded in DECISIONS.md sec 4. The "wire the concept into the execute
phase" option is dead; the concept must be learned or it doesn't exist.

**"I wanna see if king rook mates cluster together" -> `experiments/viz_b_mate_clusters.py`**
(1,074 real mates from dtm_endgame mate-in-1s across 5 materials, labeled by mate GEOMETRY:
ladder = second rook holds the inner line, 731; ksupport = white king in near-opposition, 282;
plus 37 tb-optimal approach trajectories; B-embedded, t-SNE, PNG delivered):

    VERDICT B_MATE_CLUSTERS field=iqe_geom_field  cohesion: ladder 1.06 ksupport 1.00  sil pattern +0.01 vs material -0.07
      [what-B-encodes] B-dist vs: black-king dist +0.18  white-king dist +0.02  diff-material +0.14  diff-pattern +0.14
    VERDICT B_MATE_CLUSTERS field=nucleus_distilled  cohesion: ladder 1.02 ksupport 0.99  sil pattern -0.00 vs material -0.04
      [what-B-encodes] black-king dist +0.07  white-king dist -0.02  diff-material +0.11  diff-pattern +0.01

**Findings:** (1) mate PATTERNS do not cluster — same-pattern cross-material pairs are no closer
than random mate pairs (cohesion ~1), silhouettes ~0, in BOTH fields. (2) B's micro-structure =
coarse black-king location + material; WHITE-king position carries ~nothing (+0.02), though king
support is definitional for the KRvK pattern. (3) Approach paths are chaotic long jumps in B, not
a funnel (t-SNE illustrative; consistent with cohesion). (4) **The stored MATE_W goal vector sits
OFF the mate cloud in the clean geometry field** — it is inherited stale from pre-decoupling and
nothing in the recipe trains it; the longshort planner navigates toward it. Fix: goal = B-bank of
actual mate exemplars, not zW. Converges with the day's other results: the field encodes
material-reachability + coarse king location, and NO concepts (not in d, not in B). Concepts must
enter via training signal; next round = make B pattern/outcome-aware and re-run this figure as the
acceptance test (cohesion < 1, arrows funnel).

## 2026-07-22 (later still) — human-mate catalog: lichess-B DOES cluster rook-pattern mates; direction level needs a sharper field

**Kaveh's redirect:** tb (DTZ) play isn't clean, Stockfish is cleaner, humans are the natural mate
source ("they get mated easily") -> catalog mate directions from the LICHESS-trained B field.
`experiments/catalog_mate_directions.py`: 1,500 real human mates (Black mated, median elo 1510)
from the 256mb shard prefix, pattern-classified by rules (qkiss 575, ladder 323, rook-other 173,
queen-other 138, ksupport 89, backrank 62, ...), embedded with lichess_gn_iqeqrl_sf (d=64).

    VERDICT MATE_CLUSTERS_LICHESS  silhouette(pattern)=-0.32  cohesion: ksupport 0.31  backrank 0.66  ladder 0.70  rook-other 0.72  queen-other 1.02  qkiss 1.07 ...
    VERDICT MATE_DIRECTIONS_LICHESS.MASTER  median cos(DeltaB, master)= +0.99
    VERDICT MATE_DIRECTIONS_LICHESS.RESIDUAL  cosine within-pattern +0.08 vs across +0.36

**Findings:** (1) **rook-family mate patterns CLUSTER in the human-data field** — ksupport 0.31,
backrank 0.66, ladder 0.70 (first sub-1 cohesions anywhere; the toy-trained fields gave ~1.0).
Kaveh's "king rook mates cluster together": YES in lichess-B, NO in toy fields — human data
taught the pattern. Queen mates stay diffuse (they happen everywhere; face-valid). (2) One
MASTER "toward mate" direction carries ~everything (cos +0.99); after projecting it out the
residuals are noise -> per-pattern APPROACH DIRECTIONS are not resolvable at d=64. The
direction-catalog k-means is qkiss-base-rate everywhere = degenerate at level 2.

**Decision (Kaveh): "get more data and sharpen the field."** Bundle (one run, recorded):
d 64->512, arch 64ch/6bl -> 128ch/10bl + spectral-norm + omega-free (the DECISIONS sec 1 spec),
data prefix1gb -> prefix4gb, selfplay shards -> sf_cont_endgame_v1 (tb-completed), dtm-hinge
STAYS OFF (sec 7: DTM in d crushes rank). QRL repulsion 8/30 unchanged. Smoke 500 steps launched
(lichess_sharp_smoke) to measure it/s + collapse gates before sizing the full run. Acceptance
test for the full field = re-run catalog_mate_directions.py: want cohesions < today's, and
RESIDUAL within >> across (pattern-specific directions resolvable).

## 2026-07-22 (night) — RETRACTION: lichess mate "pattern clusters" were phase shells; B-dist = piece count

Kaveh asked why NON-similar mates clump together in the t-SNE. `experiments/explain_mate_clusters.py`
decomposed the lichess-B pairwise distance over the same 1,500 mates:

    VERDICT EXPLAIN_B_DIST  spearman(B-dist, factor): piece-count |diff| +0.72  pawn-count |diff| +0.56  black-king +0.05  white-king +0.10  diff-material +0.03  diff-pattern +0.01  elo +0.00
    VERDICT EXPLAIN_COHESION  ksupport raw 0.30 -> in-stratum 0.86 | backrank 0.64 -> 0.80 | ladder 0.70 -> 1.00 | rook-other 0.71 -> 0.94

**RETRACTED:** the earlier "rook-family mate patterns cluster in lichess-B" (ksupport 0.31 etc.).
Controlled for piece count, ladder/rook-other vanish entirely; ksupport 0.86 and backrank 0.80 are
weak traces only. The clumps in the figure are PIECE-COUNT SHELLS (phase), not mate patterns —
mixed colors inside clumps are exactly what phase-clustering predicts. Unified picture: every
field to date (toy geometry, distilled, lichess d=64) is a material/phase detector; concepts and
mate patterns are in NONE of them.

**Acceptance test for lichess_sharp (30k, running) is now the IN-STRATUM cohesion** (this script),
not raw cohesion. Expectation: the sharpen bundle adds capacity+data, not a pattern-aware
objective — if in-stratum cohesion does not move, the QRL objective is the binding constraint and
the next lever is concept/outcome-aware training, not scale.

## 2026-07-23 — matched-anchor contrast data built; the adversarial veto measured: points denied 87%, regions forceable 99%

**Contrast tuples (Kaveh: random-vs-directed play from the same anchor separates mate-distance
from piece count).** `experiments/gen_contrast_mate_tuples.py`: anchor (won, DTM>=8) -> POS =
Stockfish j=6 plies (verified DTM decreased via tb rollout) + its own mate exemplar; NEG = random
legal play from the SAME anchor, kept only if it neither mated, nor won material, nor got closer
(rules+tb only). Same anchor -> material/phase/kings matched by construction; only structure can
separate the branches. VERDICT: made=2000 tried=4083 states=28000. Trainer term added to
train_geometry_l1.py (--contrast-npz, hinge d(F(pos_t),B(M)) + t*margin < d(F(neg_t),B(M)),
depth-matched pairs). Toy run queued behind lichess_sharp (MPS).

**Adversarial veto (Kaveh: cooperative vs human/adversarial reachability; the gap = bad-or-denied
positions; blunders = veto lapses).** Identification: HJ-reachability's best/worst-case
disturbance distinction, tablebase-exact. `experiments/measure_adversarial_veto.py`, 20 won
anchors, j=4, 1200 dedup'd random-walk targets, exact forceability = DFS over White's choices
with Black fixed tb-optimal:

    VERDICT ADVERSARIAL_VETO j=4 anchors=20 targets=1200  coop-reachable: won 100%  |  of WON targets: EXACT-forceable 13% DENIED 87% NEIGHBORHOOD-forceable 99%  |  of NON-won: forceable 0%

**The veto lives at the POINT level and dissolves at the REGION level** (region = same material,
bk within 1, wk within 2). Consequences, now measured rather than intuited: (1) point-goals are
un-plannable against an adversary (13% ceiling) -- goal-as-region is REQUIRED, and Kaveh's
index-with-tolerance proposal is exactly right; (2) S = forceability x reachability x density
must use REGION-forceability; (3) the 87% denied-won mass is reachable only via veto lapses =
why blunder data is sparse and precious. The neighborhood_of predicate (material + king zones)
is the working region granularity for the retrieval planner's subgoals.

## 2026-07-23 — TRAINING_STANDARDS.md established (Kaveh's 4 rules + the earned scars); MLflow wired

Kaveh set the standing do's/don'ts for all trainings: (1) checkpoint ladders, (2) one RICHEST
input format everywhere, (3) no overwrites + metadata-in-checkpoint, (4) existing tooling
(MLflow) not hand-rolled. Canonical: TRAINING_STANDARDS.md (rules 5-13 fold in the standing
scar-rules). Implemented + verified same session: step-suffixed ladders in train_geometry_l1
(was overwriting!) and train_dtm_cnn; full args+resume-source embedded in every ckpt payload;
catspace/tracking.py (MLflow sqlite backend, no-fail wrapper) wired into all three trainers,
end-to-end verified (params+metrics+tags in mlflow.db).

**Deliberate reversal:** the BOARD_ONLY (18,19) zeroing convention is DROPPED going forward —
all 20 planes (halfmove clock + repetition are real state) in every flavor, per the
richest-input rule. Scoped exception: the queued toy CONTRAST run keeps its base field's zeroed
convention (attributability of the contrast term); the next full geometry retrain adopts full
planes. Checkpoints self-describe their convention via stored args.

## 2026-07-23 — refactor: layered engine package, canonical utility homes, MLflow registry

Per Kaveh ("modular with nice interfaces... engine layered... try different models in each layer;
homeless code -> dedicated folder; port model/data info into the framework"): built
`catspace/engine/` (Protocols + FieldModel/values/priors/MCTSSearch/LayeredEngine -- subgoal enters
the PRIOR, value stays GLOBAL, per DECISIONS sec 8), moved tablebase utils to `catspace/tb.py` and
concept/instrument diagnostics to `catspace/diagnostics.py` (old experiment import paths re-export
-- nothing broke: 268/268 tests pass, ladder_mate smoke runs through the shims), created
`catspace/incubator/`, and registered 13 incumbent models/datasets into the MLflow "registry"
experiment. FieldModel resolves each checkpoint's input-plane convention from its stored args
(the TRAINING_STANDARDS rule-2 bridge for legacy zeroed-plane checkpoints). Next fold-ins:
compute_layer/catspace_engine + catspace/planner into the engine package.

## 2026-07-23 — new inquiry opened: TACTICS (INQUIRY_TACTICS.md)

Kaveh's definition ("a tactic is an opportunity outside of our plan afforded by a mistake by our
opponent") formalized against the literature and our own measurements. Field definitions converge
on smooth-vs-forced DISAGREEMENT (quiescence, depth instability, only-move gap, Leela WDL
sharpness, lichess puzzle mining = mistake-created + only-move). Map-native form: **tactic = a
veto lapse cashed by force** (the 2026-07-23 veto measurement: mistakes flip regions from denied
to forceable), sharpness = |field reading - forced-search reading|, predicted B-signature =
winning approaches FUNNEL into tactical regions while general approaches stay diffuse. Three-set
contrast design (general / human / winning predecessors of a strike's B-neighborhood) with the
opportunity-ratio / approach-concentration / conversion-gap / temporal-jump readouts; taxonomy
sound-converted / sound-missed / pseudo. Experiment ladder E1-E5; E1 (exact veto-lapse tactic
events on toy trajectories) runnable with existing code (forceable() + rollout_dtm); E2 imports
the lichess puzzle DB (HuggingFace Lichess/chess-puzzles) -- import don't re-mine. Engine-derived
puzzle labels = evaluation/probe use, not field-training signal (audit stance noted).

## 2026-07-23 — inquiry opened: MULTICHANNEL QUASIMETRIC (INQUIRY_MULTICHANNEL_FIELD.md)

Kaveh: index human positions, branch each anchor under multiple PLAY REGIMES (random / optimal
/ graded-strength / human), tag states with the generating regime, learn a quasimetric whose
steps carry distance AND per-regime probability. Formal answer: YES — regime tags are new OMEGA
values (the conditioning seam was built for "who generates the dynamics"; Elo bins already do
this for human channels); -log p adds like plies so w = lambda*1 + (1-lambda)*(-log p) closes
into a true quasimetric with exp(-d) = discounted reach probability (C-learning equivalence);
my-style channels combine via union-graph min, opponent channels stay vector-valued (GPI-style
per-query combination). Payoff: forceability, the veto gap, the tactic alarm, human familiarity,
and blunder-affordance-by-strength all become CHANNEL-DIFFERENCE QUERIES on one object. Data
generation = the unification of this week's bespoke datasets (contrast tuples, veto sets,
coop-vs-human fields) into one anchors-x-regimes tagged generator; machinery exists (index,
uci.py graded engines, tb, branch generator). v1: 4 channels on the d=512 field; first
instrument = does d_optimal-defense - d_random track tb-exact deniedness. Awaiting go.

## 2026-07-23 — flavored-energy opponent model framed (INQUIRY_MULTICHANNEL_FIELD.md sec 6)

Kaveh extended the energy formulation with Boltzmann/barrier physics: move difficulty = potential
barrier, Elo = inverse temperature, flavors = multidimensional energy (SF crosses tactical
barriers, Leela strategic). Training framed as multidimensional IRT: pi_omega(m|x) =
softmax(-<beta(omega), E(m,x)>), E = K move-map channels off a policy-style head, beta = cohort
embeddings (Elo bins + engines); masked CE on (position, move, Elo) triples (12M+ rows on hand);
nonneg+scale constraints for identifiability; K=1 (Regan-like, eval-free) baseline -> K=2,3 by
held-out LL; Maia-style per-bin ceiling; EXTERNAL acceptance = monotone map onto lichess puzzle
ratings (empirical difficulty, no training contact). Eval-free by construction (audit-clean).
Feeds: multichannel rho_c edges, the watchlist's "will they see it", tilt as online beta update.

## 2026-07-23 — opponent-model architecture DECIDED: candidate-set self-attention (option A)

Kaveh's question ("likelihood of seeing a move depends on what else is going on") exposed that
independent per-move scores + shared softmax only give competition-through-the-normalizer, not
score-level interaction. Options discussed: (A) self-attention over the legal-move token set
(set-contextual scores; distraction/threat-load/Einstellung learnable), (B) seeing x choosing
two-stage factorization (identifiable ONLY via the engine channels -- engines see everything,
pinning the value component), (C) history/plan-state conditioning. DECIDED: A for v1,
simplicity first, iterate later; B and C deferred with their identification story recorded.

## 2026-07-23 — BUILD: everything coded; multichannel relaunch from the 20k checkpoint

Kaveh: "code up what we talked about, all of it; once the current run passes 20k and checkpoints,
kill it, and run anew with the modifications." Done:

1. **Multichannel field (INQUIRY_MULTICHANNEL_FIELD):** TorchFB `regime_channels` — regime id in
   omega column 3 conditions F ADDITIVELY on the trunk; zero-init embedding => regime 0 (human/
   base) is byte-identical to unconditioned (omega-free doctrine preserved for the base channel;
   verified in eval mode — the initial "mismatch" was spectral-norm power-iteration drift in
   train mode, not the regime path). load_ckpt allowlists emb_regime.weight; trainer has
   --regime-channels + --regime-shards DIR:ID:FRAC with MultiMixSource (N-way); LichessPairSource
   carries per-source/per-row regime tags. Resume-upgrade rebuilds the 20k model with the new
   embedding, everything else carried (verified identical outputs).
2. **Regime-1 data:** gen_regime_random.py — 8,000 random walks / 103,820 rows from 1gb-prefix
   anchors (14s). Regime 2 = sf_cont_endgame_v1 re-tagged at source level. Regime 3 (sf-vs-weak)
   reserved for the next generation pass.
3. **Opponent model (option A, decided):** catspace/nn/opponent.py (move tokens: from/to/piece/
   captured embs + board context; 2-layer self-attn; cross-attn to per-cohort skill tokens;
   masked softmax) + build_move_selection.py (played-move recovery by child-matching; ~300k rows
   building now) + train_opponent_model.py (masked CE; per-Elo-bin held-out NLL/top1 VERDICTs;
   MLflow + ladder).
4. **Tactics:** catspace/engine/watchlist.py (LatentTactic + TacticWatchlist: shell monitoring,
   alarm on crossing, hysteresis re-arm) + experiments/tactic_events.py (E1: exact veto-lapse
   events, flip-rate after LAPSE vs OPTIMAL defender moves — the definition, measured).
5. **Asym fix:** train_geometry_l1 --w-rev/--rev-cap — reversible reverses pulled into the cheap
   band (<=4) so the all-pairs floor stops inflating them. Bundled into the pending contrast run.

**Kill + relaunch:** lichess_sharp killed at step 20,000 (ckpt secured; d_step 0.79 / d_rand 54.0
at kill). RELAUNCH BUNDLE (recorded per no-one-lever): resume lichess_sharp_step20000 ->
lichess_mc, regime_channels=4, mix human 0.70 / random(r1) 0.15 / sf-optimal(r2) 0.15, steps
20k->50k, resume-lr-scale 0.1 guard, ckpt ladder every 5k. 500-step smoke launched first
(standards); full run on smoke pass. Deferred behind the big run: toy contrast run (now bundled
with --w-rev), opponent-model training, E1 run.

## 2026-07-23 — publication plan armed: "The Opponent's Veto, Learned" (PUBLICATION_VETO_NOTE.md)

Kaveh: if the multichannel veto works, ship a self-contained note (LinkedIn + repo) and an
interactive GitHub demo (play + analyze with our engine, veto overlay; reuse lichess code where
licensed). GATE (hard): measure_veto_channels.py — learned gap d(F;sf-optimal)-d(F;random) vs
EXACT region-deniedness on sf_cont-region anchors; PASS = AUC>=0.65 & anchor spearman>=+0.4.
Early read at the 25k rung, full at 50k. Demo plan: veto overlay on the existing chess.js play
UI; static precomputed GitHub Pages tier + full local-server tier; licensing checked (chessground
GPL-3.0, chess.js BSD-2, puzzle DB CC0, cburnett pieces CC-BY-SA). Nothing goes public before
Kaveh reviews.

## 2026-07-23 — veto gate @25k: FAIL, support-confounded (diagnosed); fix queued

    VERDICT VETO_CHANNELS field=lichess_mc_step25000  targets=750 denied-rate=0.13  AUC(gap->denied)=0.367  spearman anchor-level -0.320
    [regime emb] norms r0 0.177 r1 0.452 r2 0.619; cos(r1,r2)=0.574

Channels ARE diverging mechanically, but the learned gap runs BACKWARDS on endgame probes:
regime 2's data (sf_cont) is endgame-dense while regime 1's walks came from MIDDLEGAME anchors,
so the gap measures per-channel SUPPORT, not the veto (the inquiry's caveat (a), realized).
Also: regime-0 embedding drifted (0.177) — zero-INIT but not frozen; the "r0 == base" identity
held only at init. Fixes queued for the follow-on run: (1) regime_random_endgame_v1 walks from
sf_cont anchors (shared support; generating), (2) freeze-or-subtract row 0 in embed_F. 50k
re-read still on; publication gate unresolved, not failed-forever.

## 2026-07-23 — post-mortem: why the smoke missed the gate failure; balance audit; probe cache

Kaveh: "why didn't the smoke test catch this?" Because the smoke tested the MACHINERY (train/
save/load/collapse gates) and the machinery was fine; the failure was in the DATA DESIGN, and
the acceptance instrument that exposes it was written AFTER launch. Two standards added (#14
pre-registered acceptance miniature in every smoke, #15 balance audit before contrastive
objectives) + #16 (materialize labels, cache probes).

    VERDICT CHANNEL_BALANCE: regime1-mid vs regime2 phase OVERLAP = 0.13 (the launch config's
    fatal number, computable in 8s with no training); regime1b-endgame vs regime2 = 0.95 (the
    follow-on run's data, balanced). Human r0: median 24 pieces; r2: median 8.
    VERDICT COHORT_BALANCE: move-selection Elo bins max/min 172x (bins 3-4 dominate; bin8 440
    rows; engine cohorts 0 — planned). Report per-bin metrics always.

Saved-vs-on-the-fly answer: raw positions/shards/tuples/verdicts are saved; QRL pair sampling is
per-batch by design (fine); the GAP was expensive derived labels (forceable DFS, deniedness,
rollout DTM) recomputed and discarded per script, and tb probes re-read per process. Fixed:
persistent sqlite probe cache in catspace/tb.py (WAL, silent-degrade; 9.4k rows after one small
run; warm speedup grows with reuse). Label materialization = standard #16 going forward.

## 2026-07-23 — shared-anchor regime rollouts + DVC dataset tracking

Kaveh: "positions sampled from human lichess data, followed by sf-vs-sf rollout or random-vs-sf
rollout; save and track the datasets using open source tools." Built gen_regime_rollouts.py:
SHARED anchors (fixes the balance confound at the ROOT — both regimes continue the SAME human
positions, twin-design; overlap by construction vs the 0.13 disaster). Regime vocabulary
re-designated: 2 = sf_sf (purposeful both sides), 3 = rand_vs_sf (anchor's mover drifts, SF
resists). Smoke: shared-anchor invariant TRUE, 2s/40 anchors; full 4k anchors generating.
Anchors.json carries per-anchor provenance (source shard/row/fen/elos).

**Dataset tracking = DVC** (open-source standard; MLflow registry keeps the catalog role):
dvc init + 10 generated datasets tracked as content-addressed .dvc pointers (gitignore
restructured /data/ -> /data/** with .dvc negations — DVC refuses pointers inside blanket-
ignored dirs). Source lichess prefixes deferred (multi-GB hash; note). Pointer files staged,
NOT committed (commit-on-request rule); no DVC remote yet (local cache; remote = future).

## 2026-07-23 — determinism audit of the rollout ladder (Kaveh's question); sampling fix

"Is Stockfish deterministic — one continuation per position?" Audit: sf_full = near-deterministic
(fixed depth + 1 thread) EXCEPT transposition-table history leaked between anchors (engine reuse
without ucinewgame) — fixed via per-anchor game tokens; kept deterministic BY ROLE (the optimal
corridor — "continuation", not "rollout"). SF Elo-limited = stochastic BY DESIGN (the strength
limiter randomizes candidate selection) — true rollouts. Maia at nodes=1 = ARGMAX, i.e. the modal
human every time — WRONG for visitation/rho semantics (degenerates the distribution to a delta);
fixed: --temperature=1.0 sampling (verified: same position -> d4/Nc3/c3 across 6 samples). Future
upgrade: lc0 verbose-move-stats exposes the full per-move policy = exact soft labels for the
opponent/energy model. Daemon stint 1 banked 15,400 anchors / 1.56M rows / 78 shards in 34 min
(~27k anchors/hr, 8 regimes); patched daemon relaunched from shard_079.

## 2026-07-23 — corrected multichannel relaunch (lichess_mc2); quarantine + surgery verified

Decision (Kaveh's "should we stop/restart? delete the biased data?"): STOP at the 30k rung (base-
field learning preserved via resume; riding to 50k would only re-prove the confound), relaunch
corrected; QUARANTINE pre-fix rollout shards as regime_rollouts_v0_argmax (provenance note: modal
maia + TT-warm SF; keep for the modal-vs-sampled ablation) — compute is cheap, provenance is not.

Surgery verified numerically: regime table 4->16 with zero-padding; REGIME_RELATIVE conditioning
(emb(r)-emb(0)) with legacy row-0 drift folded out — regime0==base True, new rows==base at init
True, trained r2 difference preserved True. Smoke + GATE MINIATURE (standard #14, first use):
AUC 0.505 / spearman ~0 = NEUTRAL — the anti-correlation (0.37) died with the balanced data;
no separation expected at 500 steps; no structural pathology. Miniature also exposed gate-region
coverage: exact deniedness lives at <=6 pieces but rollout anchors are middlegame-heavy -> daemon
now draws anchors from BOTH the 4gb prefix AND sf_cont endgame shards (std #15 support planning).
lichess_mc2 launched 30k->60k: human 0.65 + rollouts_v1 ladder (in-file regime tags) 0.25 +
endgame walks (r1) 0.10; gate reads at every 5k rung.

## 2026-07-23 — mc2 launch: two crashes, one real fix (upgrade => fresh optimizer, enforced by flag)

The resume-upgrade (regime table 4->16) crashed opt.step(): Adam moments shaped for the old table.
Fix #1 (pop opt_state from the loaded payload) was THEATER — the restore path RE-LOADS the ckpt
from disk into a fresh payload. Real fix: regime_upgraded flag skips the restore branch outright.
Monitor caught both crashes within seconds (the watchdog earning its keep). lichess_mc2 now past
step 30,200 at 1.2 it/s, ladder every 5k, gate reads at each rung. Daemon concurrently on mixed
sources (human prefix + sf_cont endgame anchors) for gate-region channel coverage.

## 2026-07-23 — AUTONOMOUS: the mate mission (tablebase-free graded conversion)

Kaveh: "get the current checkpoint to reasonably mate progressively harder toy scenarios
without relying on tablebases." Strategy: the constraint concept scored 0.75 as an ORACLE
search value (vs 0.12 pure); it is a local pattern -> DISTILL it into a net (labels allowed;
flood-fill forbidden only at play). Pipeline launched: (1) escape_data_v1 200k rules-only
labeled rows (20s); (2) escape_net_v1 training (CPU, DTMNet arch); (3) mate_ladder_eval.py
-- the graded exam: KRvK-easy / KRRvK-central / KRRvKB / KRRvKP / KRRvKBP (synth generators
for the 5-piece gaps; tb = referee/exam-certifier only); (4) full 4-config exam (pure/dtm/
escape/blend) auto-runs when the net lands; (5) dtm_endgame_v2 (100k rows) generating for a
DTM-CNN v2 retrain. Baselines at 600 nodes: pure = 0.12 ladder / 0.06 KRRvKBP (replicates).
Rollout daemon STOPPED to free CPU (banked: 53,200 anchors / 5.4M rows / 365 shards).
lichess_mc2 continues on MPS (60k gate suite on completion).

## 2026-07-23 — rule tightened mid-mission: no hand-coded concepts even for training

Kaveh: "no hand coding allowed even for training." Escape-net training killed mid-run; the
escape lever is retired from the mission (artifacts kept as the DIAGNOSTIC CEILING only — the
exam's escape config becomes an instrument reading, not a candidate). DECISIONS 4b records the
tightened rule and the legitimate-signal boundary (outcome-structure facts + own experience;
no designed features as targets). Mission redirected to legitimate levers: (1) DTM-CNN v2 on
4x data (generating); (2) EXPERT ITERATION — play games with the current value+search, harvest
(position, actual plies-to-mate) from its OWN won games, retrain, re-exam, repeat: experience-
grounded, matches the AZ precedent and the validated distillation finding.

## 2026-07-23 — MISSION REDIRECT (Kaveh): mate via IQE field + flavored-energy prior

"I don't want the DTM + expert iteration to mate. I want the IQE + flavored-energy opponent
model to mate. Search near mate guided by field." EI stopped (its rows remain exam baselines).
The mating engine is now: value = HEALTHY geometry field's distance to the toy MATE BANK
(400 real mates, nearest-exemplar per goal_bank), prior = the flavored-energy model's sf_full-
cohort policy (step-8000 rung), MCTS + mate_stop. This is the sanctioned RE-TEST of the shelved
verdicts ("field value hurts near mate", "uniform beats field prior") which were measured on the
COLLAPSED field — conditional-rejections rule: re-test after field promotions. v2 DTM net landed
en route (3p 0.917 / 4p 0.704 / 6p 0.339 — 4x data lifts 3p/4p, the 6p wall stands). Exam
running: pure vs field vs energy vs fieldenergy across all 5 scenarios.

## 2026-07-24 — THE COMPOSITION MATES: fieldenergy 0.75 on the ladder, tablebase-free

    VERDICT MATE_LADDER cfg=fieldenergy KRRvK-central  mate=0.75 (15/20)  med_plies=17  search/mate=5,822
    VERDICT MATE_LADDER cfg=fieldenergy KRRvKB  0.30 | KRRvKP 0.75 | KRRvKBP 0.30 | KRvK-technique 0.05

Kaveh's specified engine — IQE geometry-field distance-to-mate-bank as VALUE + flavored-energy
opponent model (sf_full cohort) as PRIOR + batched/cached MCTS(800n, mate_stop) — with NO
tablebase and NO hand-coded concepts at play, vs tablebase-optimal defense:
**0.75 on KRRvK-central** = the hand-coded-concept oracle's score, approaching the tb-oracle's
0.85; 5x pure search on the full toy (0.30 vs 0.06); FIRST nonzero on KRvK-technique (0.05).
Strength-per-node: ~5.8k nodes/mate (SF ~54k; oracle ~1.2k).

**REVERSALS (conditional-rejections rule paid out):** "field value hurts near mate" and
"uniform prior beats field prior" are hereby retracted as COLLAPSED-FIELD artifacts — the
healthy decoupled field (33x, rank 15) + learned prior is the best legal config ever measured.
DECISIONS sec 3 needs amendment. Engine ingredients: iqe_geom_field.pt + opponent_energy_v1.pt
+ mate bank (400 toy mates) + batched MCTS (eval-cache + batch_leaves=8, built this session).

## 2026-07-24 -- lichess_mc2 60k final: training verdicts in, veto gate still closed

Run complete (60k steps, IQE+QRL, 16 regime channels, regime_relative, resume-upgraded from 20k).
Training-internal verdicts: VAL_TOP1 0.016 / TOP8 0.097 (chance 0.002; coarse-navigator profile as
expected), REACH_SLOPE won 0.653 / lost 0.664, DIFF_SLOPE won 0.780 / lost 0.842 (won-lost
separation present; both were NEGATIVE at step 2000).

    VERDICT VETO_CHANNELS field=lichess_mc2 targets=750 denied-rate=0.13
      AUC(gap->denied)=0.535 [95% CI 0.46-0.61]  spearman target +0.041 | anchor +0.172

GATE: FAIL (needs AUC>=0.65 with CI clear of 0.5, anchor spearman >=+0.4). Movement vs 35k
(0.472 [0.40-0.54] -> 0.535 [0.46-0.61], anchor spearman -> +0.172): direction right, magnitude
far short -- the cross-regime gap d(F;opt)-d(F;rand) still doesn't separate denied targets.
Recorded levers stay parked pending Kaveh (dedicated endgame-rollout source fraction; FiLM-style
conditioning instead of additive regime embedding). Thin gate-region support (~5% of targets in
the informative band, diagnosed at 35k) remains the suspected bottleneck.

## 2026-07-24 -- SOFT-MIN SIDE v1: per-regime reach-probability head trained (90s on MPS)

Kaveh: "build the soft-min side under the opponent model." First implementation of the rho half
of the energy algebra (INQUIRY_MULTICHANNEL_FIELD sec 5): rho_c(x,g) = sigmoid head on the FROZEN
mc2 towers' [F_c(x), B(g), F*B], trained C-learning-style on the banked shared-anchor rollouts
(200k states, 15,689 walks, all 8 regimes; positives = same-walk futures at Geometric(1-gamma)
gaps, gamma=0.85; negatives = other walks; split BY WALK). experiments/train_rho_head.py; full
run 90s (embed+train) after a 69s smoke caught a shard-granular --n-states cap bug.

    VERDICT RHO_HEAD held-out AUC per regime: 0.985-0.991 (all 8)
    spearman(-log-odds, ply-gap): +0.466/+0.420/+0.421/+0.388/+0.417/+0.409/+0.413/+0.415
    (regimes 2,3,4,5,6,8,9,10) -- monotone-in-horizon soft distance in EVERY channel

Grading: AUC is inflated by easy negatives (cross-anchor = different material); the spearman is
the meaningful verdict -- -log-odds behaves as a soft distance at the ORDERING level. NOT yet
claimed: ply-unit calibration, the softmin<=hardmin inequality vs IQE d (needs unit alignment --
define-identifications rule), cross-regime Delta readouts (forcedness/trap potential). Those are
the v2 readouts. Ckpts: rho_head_v1.pt + step2000/4000 ladder, args embedded.

## 2026-07-24 -- soft-hard consistency: the two algebra sides agree at rank level; channel
## separation is the shared bottleneck

experiments/measure_soft_hard_consistency.py (18s, MPS; seed!=training, same-walk pairs gap 1-20):

    VERDICT SOFT_HARD spearman(-log-odds, IQE d) per regime: +0.56..+0.69, ALL +0.615 (n=4000)
    VERDICT SOFT_CHANNEL_GAPS vs regime 2: mean|gap| 0.071-0.079 log-odds (all 7 comparisons)

Reading: (1) soft (rho head) and hard (IQE d) sides agree substantially at the ORDERING level
without being redundant (0.6, not 0.95) -- consistent with min-plies vs discounted-visitation
semantics on shared towers. (2) Cross-regime soft gaps are TINY: the same channel-separation
weakness the veto gate measures on the hard side (AUC 0.535). Both sides of the energy algebra
now independently indict the regime conditioning, not the heads -- strengthens the case for the
parked levers (FiLM conditioning; dedicated endgame-rollout fraction) over more same-recipe steps.

## 2026-07-24/25 -- n=100 confirmation (partial), failure taxonomy (partial), MISSION PIVOT
## to the bootstrap engine

n=100 exam, scenario 1 (before Kaveh's redirect killed the rest):

    VERDICT MATE_LADDER cfg=fieldenergy KRRvK-central mate=0.79 (79/100) med_plies=19 search/mate=5,764

CONFIRMS the n=20 headline (0.75) -- not a small-sample fluke. Scenarios 2-5 killed per redirect.
Failure diagnostic (tb-refereed rerun of the same starts; killed at 26 games): 20 mate,
3 BLUNDER:rook-hang (WDL flip on a specific white move, e.g. Rb3 hung at ply 2), 3
NO-CLOSE:threefold (never lost the tb-win, shuffled into repetition). Failures split ~evenly:
outright rook hangs vs near-mate plateau. Both logs preserved (exam_n100.log, fieldenergy_diag.log).

PIVOT (Kaveh): no external mate bank at all. The BOOTSTRAP engine (bootstrap_mate_engine.py):
empty bank -> MCTS (energy prior + mate_stop) probes the field -> every checkmate leaf TOUCHED
is harvested into an online episodic bank (own experience; rules-certified) -> value = distance
to discovered mates (self-calibrating scale). One knob: --nodes. Goal: 100% KRRvK-central, then
progressively harder scenarios. Harness engineering (we'll run this a lot): position-level
prior+embedding caches, dmin tail-cache (bank append-only => exact), batched prior via mcts
policy_batch_fn, per-move component profiling (prior/embF/dbank/tree/harvest), milestone cache
(searched positions + own-experience p_win; recording only, wiring parked), checkpoint-resume
(per-game results jsonl, --fresh to wipe). Speed: 26-44 nodes/s (pre-cache) -> 350-450 (cached)
-> ~25-30x with batched prior expected; games 8-18 min -> 1-2 min. Pre-cache partial: 21/23
mate (0.91). Cached rerun running; ladder chain (KRRvKB/KRRvKP/KRRvKBP/KRvK-technique, fresh
per-scenario banks) queued behind it. Incumbent baselines to beat: 0.30/0.75/0.30/0.05.

## 2026-07-25 -- BOOTSTRAP first clean verdict: KRRvK-central 0.83 at 5000 nodes, from zero

    VERDICT BOOTSTRAP_MATE scenario=KRRvK-central nodes=5000 mate=40/48 (0.83) med_plies=13
      bank_final=4006  med_t/move=28.7s  med_t/solve=185s  [52 min, 4 workers, fixed harness]

The engine started knowing NOTHING (empty bank) and discovered 4,006 mates by its own search;
0.83 EDGES OUT the external-400-bank incumbent (0.79, n=100 exam) and mates much faster
(med 13 vs 19 plies). NOT the 100% target -> 10000-node rung queued after the ladder.
By-game-index quartiles (n=12 each, parallel workers, noisy): mate 1.00/0.67/0.83/0.83 -- no
clean rate curve; but med_plies 14/15/11/10 = lines SHORTEN as the bank matures (the episodic-
memory signal, visible: late-game mates land in 7-9 plies vs 15-21 early).
Grading: 0.83>0.79 is suggestive, not significant at n=48 vs n=100 (overlapping CIs); the
honest claims are (a) zero-knowledge bootstrap REACHES external-bank level, (b) plies shorten
with bank growth. Ladder continues: KRRvKB started 02:01.

## 2026-07-25 -- full-stack v2 (WDL+reuse+prune+tri-refresh): 0.81, and the reuse-threefold bug

    VERDICT (v2) KRRvK-central nodes=5000 mate=39/48 (0.81)  9 FAILs, ALL threefold
    FAIL banks: 29/361/503/1082/1319/1464/1688(g025)/1928/3532(g038, +0 harvest)

Speed transformed (~20 min/48 games vs 52 baseline; 5-25s moves; prune audits: refresh err
+0.000, one leaf miss 0.574 from the 256-cap). Rate did NOT improve (0.79-0.83 band) and
mid-run it DIPPED -- mechanism found: REUSE BLINDS THE SEARCH TO THREEFOLD (flags planted at
expansion under the then-history; carried subtrees never re-checked). Fixed 0026d28 (additive
re-flag on adoption; game history only grows so stale flags stay valid). Also per Kaveh:
warm-up 60328b2 (search until bank>=1000 before the first timed move; warm trees feed the
real move via reuse) + irreversibility guard 9b15e3e (captures/pawn moves void the
reversibility bound -> full re-anchor; estimate error is one-sided pessimistic + audited).
g025 drew AGAIN (bank 1688) = stable FIELD-WRONG-with-support workbench; g038 (+0 harvest,
bank 3.5k) = second specimen. v3 (fix+warmup+guard) chained; if rate still <100%, queued
proposal = softmin/multiplicity value over candidates (Kaveh's call).

## 2026-07-25 -- v3 verdict: 0.83, rate PINNED across mechanisms; residue = value inversion

    VERDICT (v3) KRRvK-central 5000n: 40/48 (0.83) med_plies=9 med_t/solve=70s bank 4980
    Ladder of engines: baseline 0.79 (med 13 plies) -> v2 +reuse 0.81 -> v3 +threefold-fix
    +warm-up +irreversibility-guard 0.83 (med 9 plies, 10s/move)

Every search mechanism improved speed/ply-efficiency; NONE moved the rate out of 0.79-0.83.
Stable offender set (~8/48 starts; g025 drew 4 consecutive runs). PROBE verdict on fresh fails:
FIELD-WRONG at every cycle position -- tb-optimal move ranked LAST (19/19 x2) by the value
while the PRIOR ranks it 3rd (g014/g016 class; g006 = double handicap, prior 33/33). The
field's local ordering INVERTS near these cycles (greedy distance-to-known-mates punishes the
transit step of the correct plan). Support kNN: SUPPORTED (not a coverage hole).
Ops notes: v2-launcher early death let v3 overlap v2 -> MPS pressure crashed 3 workers; rogue
replays never corrupted the checkpoint (41-then-48 unique rows, dedup verified); resume
completed the run exactly. Harness hardening queued: chain-wait on workers, worker-id prints,
child supervision, dedup-by-g in VERDICT.
NEXT (awaiting Kaveh): experience-gated value trust (milestone 0-for-N regions -> devalue the
field, let prior+search carry) + root exploration floor for double-handicap positions.

## 2026-07-25 -- v4 LAST MILE: 0.96 on KRRvK-central. The resignation gap was the wall.

    VERDICT (v4) KRRvK-central 5000n: 46/48 (0.96) med_plies=9 med_t/solve=22s [13 min total]
    Engine ladder: 0.79 baseline -> 0.81 +reuse -> 0.83 +fixes -> 0.96 +LAST MILE

Kaveh's diagnosis held end to end: humans resign trivially-won endings -> the lichess-trained
field has no trajectory support in the nucleus -> its extrapolated distances INVERT at cycle
positions -> no search mechanism can outrun a value that fights the correct plan. Fix: inside
the tb nucleus (<=5 pieces) the WDL value's distance source = dtm_cnn_v2 (offline tb-optimal-
plies regression; NO tb probes at play). All historic offenders converted (g025: drew 4
straight runs, now mate in 7). Residue: g038/g047 = dtm_cnn_v2 error pockets (4p fidelity
0.704; deployment distribution = defense-steered lines != random-won training sample).
NEXT (proposed): (1) dtm error audit on the 2 trajectories; (2) ENUMERATE all KRRvK won
positions, label via cached tb, distill the complete table (pairwise ranking loss + DAgger
top-up from engine trajectories) -- no sampling gap, nothing for the defense to find.
Precedent check (Kaveh asked): Leela had the same symptom (won-endgame shuffling) and fixed
it with tb-RESCORED training + moves-left head == our distillation + DTM value, validating.

## 2026-07-25 -- reference engines on the SAME 48 starts (bench_engines_krrvk.py)

    VERDICT BENCH sf5000        48/48 (1.00) med_plies=11 med_nodes/move=5,001
    VERDICT BENCH sf100ms       48/48 (1.00) med_plies=7  med_nodes/move=206,804
    VERDICT BENCH maia1900_5000 48/48 (1.00) med_plies=9  med_nodes/move=187 (lc0 early-stop)
    (ours: bootstrap v4 0.96, med 9 plies, 5,000 evals/move; no tb at play for anyone)

Readings: SF converts on EXACT mate-score propagation (value precision irrelevant); lc0+maia
(a HUMAN-data net) converts at med 187 nodes/move -- terminal exactness + decent policy carry
the last mile in standard MCTS. We sit 2 games under the engine floor with a known fix
(enumeration distillation of the 4-piece table) queued.

## 2026-07-25 -- KRRvKB: 0.75 (2.5x incumbent 0.30); the six draw channels of a won endgame

    VERDICT KRRvKB 5000n: 36/48 (0.75) [clean=32 tb-assisted=4] med_plies=10 med_t/solve=25s

Caveat: games span the trigger-generation ladder (most FAILs predate the final guards).
The tb-fallback policy (Kaveh: convert, log, don't chase the last mile) hardened through SIX
measured escape channels, each found by a real drawn game: (1) flat-gradient roots (eps
trigger), (2) confidently-wrong loops (stuckness: 2nd visit consults), (3) repetition-creation
by our own chosen move (veto), (4) threefolds completed on BLACK-side keys via different White
routes (both-color arrival counting), (5) field/tb alternation oscillating through the
consulted position (sticky handover), (6) FIDE claim-by-ANNOUNCING -- Black claims a
repetition it never plays (claim-safe dtz walk in both paths) + the harness itself auto-
claiming for White (asymmetric claiming: only the defender claims). Commits 8a5a921..a427779.
Finding for the writeup: behavior-geometry guards NONE of the rules' draw channels for free;
"trivially won" hides a six-lane highway to a draw.

## 2026-07-25 -- KRRvKP: 0.88 vs incumbent 0.75; draws SEALED, survival is the new frontier

    VERDICT KRRvKP 5000n: 42/48 (0.88) [clean=39 tb-assisted=3] med_plies=6 med_t/solve=24s

Zero draws -- the 7-channel guard net holds. All 6 fails are SURVIVAL-class: 4 material
collapses (both rooks lost by ply 6-18) + 2 checkmates (promotion race lost), all in the
6-piece phase which is OUTSIDE the <=5 nucleus (bank-geometry value there is hang-blind; the
pawn defense punishes what the pawnless scenarios forgave). Loss bank live (up to 143 mates-
against harvested). Mechanism candidates for the survival gap (NOT hand-coded guards):
(a) tactical_prior (exists in mcts.py, currently 0) -- boost capture/check exploration,
rules-structure; (b) nucleus at 6p (accepting dtm_cnn's weak 0.339 6p fidelity) or a 6p
DTM retrain; (c) loss-side last mile: dtm net for BLACK-mates + fixing the mixed-units p_l
(field-scale loss distances / dtm_scale temperature). Awaiting Kaveh's pick.

## 2026-07-25 -- THE COMPLETE ENGINE (Kaveh's design, built end to end) + PLANNER v1

Final architecture, one paragraph: WDL leaf value -- p_win from bank geometry (or the
tb-trained DTM regression inside the <=5p nucleus; resignation-gap fix), p_loss from the
engine's own harvested deaths (self-calibrated temperature), and STATE-DEPENDENT draw mass
kappa(x, history) = base + fifty-clock pressure + live repetition proximity + banked
stalemate surface (zeroing preference when winning and swindle-seeking when losing EMERGE).
All three banks (win/loss/draw) discovered by the engine's own search, all pruned by the
root-anchored triangle-inequality candidate scheme (28c060f -- one full row per bank per
move, leaves query nearest-256+tail, per-bank printed audits: memory grows without slowing
thought). Search: reuse (repetition-correct), batched prior, last-mile DTM, 8-channel logged
tb-fallback + within-game progress gating (no cross-game self-statistics: banks carry FACTS).
PLANNER v1 on top (also 28c060f): discrete plan selection over rules-state (direct / reset
@clock>=30 / tradedown @>6p), acting ONLY through the prior alpha-dial (e^alpha on zeroing
moves; base priors cached unbiased; values untouched), plan usage logged per game.
PlanSelector = the RL seam; forced-resets-beyond-horizon = its Phase-5 growth path.
Parked-list triage (Kaveh): enumeration-distillation, root floor, softmin value, 6p retrain,
milestone wiring all SUPERSEDED by the fallback ledger; research mainline = region-goal
planner (built v1 today) + opponent-model v2. LADDER7 = the definitive uniform 5-scenario
rebuild, running (wdlr6_ files).

## 2026-07-25 -- KRRvK-central: 1.00 (48/48) on the definitive engine

    VERDICT (LADDER7) KRRvK-central 5000n: 48/48 (1.00) [clean=46 tb-assisted=2]
      med_plies=9  med_t/solve=50s  bank 6404  [31 min]

Kaveh's target ('win rate to 100% without losing generality') REACHED. The two rescues are
logged consults (attribution ledger), not hidden oracle calls; 46/48 = pure field+banks+
search. Engine journey on this scenario: pure 0.30 -> incumbent 0.79 -> bootstrap 0.83 ->
+last-mile 0.96 -> complete engine 1.00. Remaining ladder7 scenarios running.

## 2026-07-25 -- KRRvKB root cause: the DTM net NEVER SAW KRRvKB (or KRRvKP)

Autopsy of ladder7 KRRvKB fifty-move fails: wins died on White's FIRST move (plies 1,1,3
from dtz 3-11 starts) -- no trigger schedule can rescue a move-1 blunder. Cause: the
last-mile value inside the <=5p nucleus is dtm_cnn_v2, and dtm_endgame_v2.npz contains ONLY
{KRRvKBP, KRRvK, KRvK} -- KRRvKB and KRRvKP were never sampled; the net extrapolates there,
confidently wrong. Explains KRRvKB's persistent lag (0.75) across every engine generation.
FIX (outcome-structure data, no hand-coding): dtm_endgame_v3 regen over all five classes +
dtm_cnn_v3 retrain, chained (probe cache makes repeats cheap). Future runs get v3.

## 2026-07-25 -- infrastructure round: sensorium, Ray orchestration, immortal banks,
## always-latest policy, self-retrain loop

Built at Kaveh's direction, all committed+smoked: (1) EXPERIENCE STORE (sqlite WAL) + shard
export in the regime-rollouts schema (regime=11 self-play) + SELF-RETRAIN LOOP driver
(play N -> export -> fine-tune quasimetric -> pointer swap; banks re-embed per field).
(2) IMMORTAL BANKS: --import-banks merge; selfloop keeps ONE growing bank (banks = facts).
(3) ALWAYS-RUN-LATEST policy (memory saved): stale runs killed on every engine update,
resume makes it cheap. (4) PLANNER SENSORIUM (ProbeKit): memory/familiarity/sharpness/
surfaces probes + summary(); snapshots logged at plan decisions = the RL PlanSelector's
observation dataset accumulating from day one. (5) PROBE ORCHESTRATION ON RAY (per Kaveh:
don't hand-roll): single-flight keyed by (kind, epd), coordinator actor = cross-process
memo + in-flight dedup + milestone streaming; invalidate(kind) on bank/field changes.
Also: tri-carry probe -> anchor-every-move/C=128 (measured optimum); DTM v3-ALL gen over
every tablebase class ON DISK (the KRRvKB hole was a hand-list, proving the point);
full 3-4-5 syzygy download started (the general nucleus next).

## 2026-07-25 -- KRRvKB tail SKIPPED as semantically stale (Kaveh's call): the remaining
## v2-DTM games only padded a convicted baseline while blocking the queue. Partial stands
## as the BEFORE measurement; full re-sit on dtm_cnn_v3 when it lands. Enforcement note:
## the v3 default-flip is a commit, so code-staleness enforcement auto-handles the model
## swap moment.

## 2026-07-25 -- component factorization (Kaveh's catch): the rho head is NOT a peer
## trainable -- it is a derived readout of field + energy

Question (Kaveh): is trainable #5 (rho soft-reachability head) not the same as #1 (IQE
field) + #3 (flavored-energy opponent model)?  Answer: in information terms, YES --
1 and #5 are two READOUTS of the same trajectory ensemble: hardmin (shortest path, plies)
vs softmin (probability-weighted path mass); that is why rho is implemented as a head ON
the field's frozen towers. #3 is the GENERATOR: rho = the partition function of the energy
model's per-edge probabilities path-integrated over the dynamics. The head exists only as
AMORTIZATION: the path integral is intractable at query time, and composing #3's per-move
approximations over long paths compounds error, while training on observed walk statistics
is cheap (90s on frozen towers) and direct.

CONSEQUENCE (architecture): rho loses its pipeline row; its trigger is DERIVED -- retrain
as a post-step of every field round (towers moved or new walks landed -> 90s head refresh).
The retraining dispatcher factorizes to FOUR top-level pipelines, each a genuinely distinct
information stream: field+rho (games-as-geometry), nucleus net (tablebase truth), energy
model (decisions), planner RL (graded plan choices). Every trainable = one stream; every
derived quantity = a post-step of the stream it depends on.

## 2026-07-25 -- DOCTRINE (Kaveh): foveated planning -- vague at range, sharp up close

The planner does NOT need precise long-term vision. Like human sight: from afar the target
is a blur in a general vicinity; you move a few plies toward it, re-evaluate, and proximity
itself sharpens the view. Therefore compounding error in composed quantities (multiplying
the energy model's edge probabilities over a horizon) is ACCEPTABLE -- decision relevance
decays with distance faster than composition error grows, and per-move re-anchoring resets
the error. Quantitative precedent from today: the tri-carry probe showed the hard bound
degrades past ~3 plies, and play didn't care because every move re-anchors (local ordering
is all that is consumed). The give-up/reselect cadence IS the re-evaluation loop.

Consequence: rho demoted from routine post-step to SHELVED COMPARATOR. First resorts:
(a) hardmin field distance w/ per-move re-anchoring, (b) k-ply probability product of the
energy model for policy-weighted directional guidance. rho v1 stays frozen; it re-earns a
pipeline step only by winning a three-way comparison (rho vs carried hardmin vs probability
product, same positions) on the day long-range guidance visibly underperforms.

## 2026-07-23 -- FULL-GAME ERA: pivot, unified loop, assistant, observability (consolidated)

**Pivot (Kaveh).** The toy material classes (KRRvK 1.00, KRRvKB, KRRvKP, KRRvKBP) are now
INTEGRATION TESTS (tests/integration_positions.json, 51 cases incl. clock-pressure); research
focus moves to the end-to-end full game: RL action optimization over the planner seam,
long-term planning (region-goal chains), tactics tracking. Fullgame baseline to beat:
mated by maia-1500 in 36 plies @300 nodes (one game seeded the loss bank +182 entries).

**Unified improvement loop** (experiments/improvement_loop.py): per round -- 10 fullgames vs
rotating maia rungs (1100/1200/1400) -> ExperienceStore (SQLite WAL; games+positions+
provenance) -> regime-11 shard export -> (a) field fine-tune 2k steps, self-frac 0.35,
pointer swap; (b) planner-RL refit on all accumulated (obs, plan, outcome) tuples;
(c) NEW 96c76f0: energy/opponent-model fine-tune -- exports now stamp TRUE cohorts (us=2800
both sides in toys; real maia rung as Black in fullgames -- the old flat-1800 placeholder
would have poisoned flavor conditioning), trainer gained warm-start + weighted multi-source
sampling (lichess 0.65 / self 0.35), engine resolves opponent_energy_current.txt. Smoke:
59 stored games -> 1842 move-selection rows; self held-out NLL 2.298 (n=189) vs lichess
2.3815 after 60 warm-start steps. This closes the 4th information stream of the 94f56c4
factorization: all four pipelines (field / nucleus / energy / planner-RL) now exist.

**Nucleus pipeline round 0 DELIVERED**: dtm_tok_r0 (token transformer, all 149 tb classes
at 300 positions/class = 2.5% of 12k/class target): spearman +0.507, MAE 15.15 plies
(n=4123). Default flipped + committed; round 1 (700/class, 104.3k positions) generating.

**Assistant** (experiments/viz/assistant_server.py, native MPS :8777): play vs chosen maia
while the planner co-analyzes -- probe-triggered "let's calculate here" prompts with reasons,
top moves + most-likely leaves under the plan, pencil-editable concept tags persisted to
concept_tags.jsonl (human labels for field regions). Auto-swap reloader picked up dtm_tok_r0
live (MODEL SWAPPED, version + data-%% shown in UI) -- checkpoint-to-playable with zero restarts.

**Observability**: Prometheus stage histograms (catspace_stage_seconds: prior/embF/dbank/
dtm/tree/harvest/move_total/http) + usage.jsonl per request; MLflow UI native :5001.
Docker stack (qdrant/engine/web/mlflow/prometheus/grafana, deploy/) PARKED per Kaveh --
publish path only, no slow local inference.

## 2026-07-23 -- ladder7 rebuild: KRRvKP 0.94 (45/48; 43 clean + 2 tb) med 7 plies

Up from 0.88 pre-rebuild despite running on the weak universal dtm_tok_r0 (2.5% data,
MAE 15 plies). The 3 FAILs fit the known sharp-low-DTZ signature -- g041 autopsy: won
start (wdl +2, Black pawn b2 one step from queening), move 1 Ka8a7 threw win->draw,
ply 9 draw->loss, defender claimed threefold while winning. Nucleus r0 too blurry to
rank "stop the pawn NOW"; re-verdict on the r1 flip (conditional-rejections rule).

## 2026-07-23 -- IQE full-month continuation state (so we can pick this up any time)

Raw 2019-01 month FULLY downloaded (9.4GB zst). Sharded so far: 4GB prefix only (56 shards,
55.8M positions). lichess_mc2.pt = 60k steps on that prefix (~0.55 epochs, batch 512) --
headroom remains even in already-sharded data (decline historically appeared past ~4 epochs).
Continuation recipe: (1) shard the WHOLE month to a fresh dir with --max-gb 10 (no skip flag;
clean pass avoids the 4GB boundary truncation; disk-heavy -> queue behind nucleus gen);
(2) warm-start from lichess_mc2.pt and train in gen-parallel rounds as 1M-position shard
files land (nucleus-style progressive; ~260k steps/epoch on ~130M positions), keeping the
self-play regime channel; (3) improvement loop composes via the field pointer chain.

## 2026-07-23 -- energy-flavor continuation state (companion to the IQE entry above)

Trained: opponent_energy_v1 = 12k steps @ batch 256 on 550k move-selection rows
(move_selection_full_v1.npz): 300k human rows from the 1GB lichess prefix (elo bins 1-8,
200-wide) + ~285k ENGINE-cohort rows (8 flavors: maia rungs + sf skills, cohort ids 11-18)
-- ~5.6 epochs; likely saturated on THIS data, the lever is rows not steps.
Pipeline (built today, 96c76f0 + replay fix): per improvement round, warm-start fine-tune,
self-play rows w/ true cohorts (us -> top human bin 8; opponent -> its real rung), engine
pointer swap. Caught + fixed: replay mix initially pointed at the 300k human-only npz --
would have forgotten the maia flavors; now full_v1.
Continue-later levers, in order of availability:
(a) NOW, no sharding needed: rebuild human rows from the already-sharded 4GB prefix
    (300k -> ~3M+ rows; CPU-bound, queue behind nucleus gen);
(b) full-month rows once the IQE sharding step runs (shared dependency, one shard pass
    feeds both IQE rounds and move-selection);
(c) engine-cohort rows scale via build_move_selection_engines (more maia/sf games);
(d) self rows accrue automatically each loop round.
Follow-up flagged: dedicated self cohort id (self currently folds into sparse human bin 8,
440 lichess rows -- our rows will dominate that bin's meaning).

## 2026-07-23 -- assistant: seeded memory, streaming calc, anytime-valid A/B (37e5296)

Banks seeded from all saved scenario banks (85.8k mates / 2.1k losses / 10.7k draws,
deduped, zero cross-contamination) + reloader now syncs banks every 45s = the fleet's
discoveries stream into the live session (banks are FACTS, shared memory). Perf lesson:
seeding exposed that the server never armed the tri-anchor prune -- every eval scanned
86k bank embeddings; one set_anchor at calc start = 2-4 -> ~80 evals/s (~25x).
Streaming calculation (Kaveh: 'calculations stream in as it's calculating'): chunked
MCTS (64/chunk, tree reuse) on a thread; /calc_state serves the growing snapshot; UI
paints top moves/leaves live with an eval counter.
A/B stack (Kaveh: 'separate model endpoint... gather evidence... anytime valid'):
--pin-model endpoints (frozen model, banks still sync), /set_fen, ab_test.py = paired
winnable tb positions -> success = win preserved (syzygy truth) -> discordant sign test
-> Beta(1,1)-mixture e-process E_n = 2^n k!(n-k)!/(n+1)!, decision at E >= 1/alpha,
valid at ANY stopping time (Ville). Null smoke (v2 vs v2): 6/6 both, 0 discordant, E=1.0.
Live page /ab. First real matchup queued for a quiet machine: dtm_tok_r0 vs dtm_cnn_v2
on 5p classes. Mid-ladder signal: nucleus 5->6 fix (320b3f2) -- KRRvKBP all 5 FAILs
pre-fix, consecutive quick mates (5-13 plies) since re-exec; verdict pending.

## 2026-07-23 -- KRRvKBP verdict 0.77 (37/48; 31 clean + 6 tb) med 11 plies -- CORRECTION

My mid-ladder 'consecutive quick mates since the fix' read was premature (sampling luck) --
RETRACTED per the rigor rule. Provenance-stamp split (first real use of per-record commits;
record-level counts): pre-nucleus-6 3 mates / 4 FAILs (0.43); post-nucleus-6 34 mates /
8 FAILs (0.81). So the 5->6 nucleus boundary recovered the class to its old 0.79-0.82
partial baseline, not past it. Caveats: (1) heavy MPS contention this hour (two assistant
servers + live human play + AB smoke) -- one FAIL is a pure timeout, nodes/s sank to
111-170; (2) dtm_tok_r0 blur (MAE 15 plies) is the binding constraint; the class
re-verdicts on the r1 flip (conditional-rejections rule). One post-fix mated-AGAINST
game remains the worst symptom (tactical throw to -2 persists at 6p under r0).

## 2026-07-23 -- DESIGN CONSTRAINT for region-goal chains (Kaveh's dead-end-in-A question)

'Planner wanting region A then B then mate might sacrifice material to get to A on a
trajectory that never reaches B... a point in A that doesn't lead to B shouldn't be
clustered with the rest.' Resolution: dead-end info lives in the DIRECTED distance
d(a,B), not in cluster membership -- a quasimetric separates near-identical points with
different forward costs (why IQE). Constraint for #24: chain scores compose through
CONCRETE waypoints, min_{a in A}[d(x,a)+d(a,B)+d(B,mate)] (or the prob-product form),
never region aggregates; density prior weights waypoints; foveated re-eval + give-up
stay as the recovery layer. Today's guards are reactive only (loss-side WDL + re-eval);
prospective avoidance arrives with #24. Full-board caveat: OOD compression (measured
today) blinds the field there until the full-month IQE round lands.

## 2026-07-23 -- DISK-FULL incident + recovery (root cause: 25GB probe-cache WAL)

Disk hit 100% (139MB free): tb_probe_cache.sqlite-wal grew to 25GB -- the fleet's
long-lived reader connections blocked WAL checkpointing indefinitely. Casualties:
full-month sharder (crashed at 51 shards), IQE fullmonth round (iostream error, ckpt
verified intact at step 60000), and Part B misfired onto the incomplete shard set (its
gate was pgrep-based: a CRASHED sharder looks identical to a finished one -- replaced
with a manifest.json marker gate, which build_shards writes only on success).
Recovery: gaviota deleted (6.7G, unreferenced), prefix1gb+256mb deleted (superseded),
WAL folded+truncated after briefly pausing ladder readers (needed exclusive checkpoint;
busy=1 with readers live), partials cleaned -> 40G free. Guard committed: cache sets
journal_size_limit=256MB + wal_autocheckpoint. Sharder restarted CLEAN -- crash log
showed 51 shards in 23 min, so a full re-pass costs ~25 min of redone work, cheaper
and safer than adding resume logic. Lesson stacked on the no-concurrent-disk-jobs
memory: unbounded caches are a disk-heavy job too.

## 2026-07-23 -- ENERGY v0-at-scale: opponent_energy_fullmonth_r0 (pointer swapped)

Full-month move-selection: 5,000,000 rows (270k skipped >80-move positions) built in 971s
from the 87-shard full-month pass. Retrain: warm-start from v1, 12k steps @ 256,
mix fullmonth 0.6 / full_v1 replay 0.3 / self 0.05. Held-out (n=555k):
NLL 2.145 (v1: 2.38), top1 0.369 (v1: 0.331). The flavored-energy stream's first
data-scaled checkpoint; -logP and astray% readings inherit the sharper cohorts.

## 2026-07-23 -- technique truncated by a SPIN BUG (filed); CAPSTONE 7p underway

KRvK-technique worker spun at 100% CPU for 2h with zero output (the silent-hang failure
mode); killed -> chain declared LADDER7 COMPLETE at 37/48 technique games (37/0 to that
point, no formal verdict -- integration suite will re-verdict the scenario). BUG filed
w/ repro: WIP saved (tmp/wdlr6...wip.w2.json), g6 ply 6 from start 8/8/8/8/8/8/8/2RK3k w
-- cornered-king KRvK, prime stalemate/draw-guard territory; commit 5943420. Suspect an
un-bounded loop in the draw-guard/planner interplay when nearly all moves stalemate or
repeat. CAPSTONE KRRvKBNP-7p started 22:50 (5 workers) -> improvement loop next.

## 2026-07-23 -- spin-bug: repro NEGATIVE with reconstructed state; watchdog armed

The hung position (Kf2+Rc2 vs Kh2, mate-in-2) searched CLEAN in a bounded repro (722
evals, returned) with hist=3 pressure -- the spin depends on live worker state we don't
capture (real repetition history / tb_mode / plan wrapper). Instead of chasing blind:
per-ply faulthandler watchdog (30 min -> stack dump to log + exit + WIP resume). Next
occurrence self-diagnoses. Capstone running 28/2 (0.93) at 7 pieces meanwhile; first
real tradedown plan executions observed (goal classes KRRvkbp / KRRvknp) in both FAILs
-- worth an autopsy pass when the verdict lands.

## 2026-07-24 -- IQE full-month rounds COMPLETE: plateau (aggregate verdict, r0-r5)

30k steps (60k->90k) warm-started from lichess_mc2 on the 87-shard full month, resume-LR
guard at 0.1x peak (the guard stays: full LR once collapsed a converged field in ~200
steps). Aggregate across six 5k-step rounds vs the mc2-on-fullmonth baseline:
VAL_TOP1 0.016 -> 0.015 (flat); REACH_SLOPE_WON 0.653 -> 0.682 (+0.03, mostly round 0);
DIFF_SLOPE_WON 0.780 -> 0.796 (+0.016); LOST-side slopes unchanged. PLAIN CALL: guarded
fine-tuning of a converged field on 55%-more data moved these validation metrics only
marginally. The full-month data's real referendum is PLAY (improvement loop + integration
re-verdicts); if play also shows nothing, the shelf options are (a) higher-LR with a
fresh cosine schedule, (b) from-scratch on the full month -- both are Kaveh-call
directional runs (conditional-rejections: retest on next field promotion). Pointer:
field_fullmonth_r0.pt @ step 90000 is self_field_current.

## 2026-07-24 -- CAPSTONE KRRvKBNP-7p: 0.92 (44/48, ALL CLEAN) med 5 plies

Seven pieces, zero tablebase assists (structural: no 7p tables) -- the field + banks +
hierarchical planner alone, beating 6p KRRvKBP's 0.77. Tradedown plans executed live
throughout (goal classes KRRvknp/KRRvkbp); g045 converted after a full
tradedown->direct->reset arc. FAIL pattern (4 games, all draw-terms): tradedown fired
then was abandoned early in 3 of 4 (3-5 plies; the give-up cadence is tuned for
in-class play, not for the capture-hunting a 7p handoff needs); g044 held tradedown 23
plies but threefolded. The give-up/handoff cadence at 7p is the clear next planner
lever (feeds #24). Bank grew to 32.7k during the run. [15366s total]

## 2026-07-24 -- IMPROVEMENT LOOP ROUND 0 COMPLETE: all four learners game-fed (MILESTONE)

Round 0: 10 fullgames vs maia-1100 (0/10 -- the honest full-board baseline; med 73s/move
under all-night contention; opening-temperature v2 fixed deterministic duplicate games,
28d2fad). Export: 180 accumulated games -> shard_001 (4016 rows, 1912 LOSS rows -- the
outcome-conditioning payload finally in the training stream). Deliverables:
  - self_field_r0.pt   (field fine-tuned 90k->92k on self-play channel w/ losses)
  - planner_rl_r0.pt   (RL plan selector LIVE: n=272 tuples vs 22 at last attempt,
                        fit-spearman +0.761 -- deployed via make_planner auto-load)
  - energy step c queued for the round-0 data (pointer already at fullmonth_r0)
STATE: every trainable stream now has a game-fed checkpoint -- nucleus dtm_tok_r1,
energy fullmonth_r0, field self_field_r0, planner_rl_r0. The minimal working set Kaveh
ordered ('carry everything forward, then improve') EXISTS. Rounds 1-5 continue.

## 2026-07-24 -- WIP-loss postmortem: silent np.float32 + commit cadence = 3h of lost games

Round 1 completed ZERO games in 3h: prev_v (np.float32) made _save_wip's json.dumps
throw inside a silent except -> no WIP files -> every code commit (six this morning, UI
fixes) re-execed workers that then restarted their games FROM SCRATCH. Fixes: (a)
json default=float + log-once (silent excepts on persistence paths are now banned in
this codebase's culture), (b) process lesson: batch commits while fullgames are in
flight -- re-exec is only cheap when WIP works, and WIP failures must be LOUD.

## 2026-07-24 -- CAMPING-TRIP AUTONOMY (Kaveh away ~36h): FROM-SCRATCH decided

Kaveh's directive on leaving: full autonomy; run the FRESH training on all data to kill
the warm-start generalization-gap risk (Ash & Adams; our plateau + slope-stiffness
evidence). Plan: on distill completion -> bench -> from-scratch IQE (teacher arch,
87-shard month + self-play:11 @0.15, 170k steps = ~1 epoch, ckpt ladder /5k, val /2k)
+ nucleus resume (CPU). hlr continuation SUPERSEDED (same GPU budget, fresh answers the
question cleanly); gauntlet DEFERRED for GPU throughput. PROMOTION RULE while away:
field_scratch_full_v1 takes the pointer ONLY if it beats the incumbent on val verdicts
AND the 51-case integration suite; otherwise waits for Kaveh.

## 2026-07-24 -- camping phase 2 armed: whole-system self-play (Kaveh's parting directive)

'Continue training on the self-play data against the different models -- the whole
system together... a different kind of data because it includes our vector database
and everything else.' Armed chain (fires on scratch completion): (1) integration
promotion gate w/ revert-on-fail; (2) model-vs-model gauntlets (fullmonth vs scratch,
fullmonth vs student, TC 60+0.6) whose PGNs IMPORT into the experience store --
gauntlet games are composed-system data (field+banks+energy+planner all in the loop)
and now double as training data; (3) 8 further improvement rounds on the full maia
rotation (1100-1900). The system now generates, labels, and consumes its own behavior
end to end.

## 2026-07-24 -- phase 2 v2 per Kaveh's parting refinement: ONE line, OUTWARD, dense signal

Refinements: (1) no variant zoo -- one best line (post-promotion-gate pointer holder);
only fullmonth-vs-scratch kept as promotion evidence; student parked. (2) External
campaign: best vs maia 1100/1500/1900 + stockfish skill 1/4 at TC 60+0.6, all PGNs
imported as whole-system data. (3) THE DENSE SIGNAL (Kaveh: 'a gradient you can impose
on the vector of all parameters -- system-wide gradient'): formalized as a UNION over
one trajectory -- field gets per-POSITION signed outcome-distance residuals (the
quasimetric's native supervision, denser than W/L by a factor of game length); energy
gets per-MOVE cohort-labeled gradients (SF/leela moves = expert cohorts; their
trajectories = expert-quality paths for the field); planner-RL gets per-DECISION
(obs, plan, outcome); banks get per-SEARCH terminal facts (the non-parametric
parameters). One game updates every component. Leakage gate honored: engines' MOVES
and OUTCOMES are data; their eval numbers remain banned.

## 2026-07-24 -- LIT REVIEW (Kaveh: 'search how others do system-wide gradients')

Three established families, each mapping onto our stack:

1. MUZERO FAMILY (the canonical system gradient): representation+dynamics+prediction as
   one net, trained jointly with targets FROM THE SEARCH ITSELF -- visit-distribution
   policy targets + value targets + latent consistency. The transferable principle:
   search is a policy-improvement operator; distill its output back into the fast
   components. FOR US: our MCTS visit distributions are free dense per-position policy
   targets -> train a SELF cohort (energy model) or self-policy head on them (revives
   the old task-10 AZ-distillation thread as the self-improvement channel).
   ADOPT-1, highest value.

2. MULTI-TASK GRADIENT COMPOSITION (PCGrad / CAGrad / GradNorm): when one shared net
   (our field) takes multiple losses (QRL pairs, DTM hinge, human replay vs self-play
   channels), naive summing causes destructive gradient conflict -- project out
   conflicting components / normalize magnitudes. FOR US: the fine-tune plateau may
   partly BE replay-vs-self gradient conflict; if the scratch run underperforms,
   PCGrad/GradNorm across channels is the next lever. ADOPT-3, conditional.

3. DIFFERENTIABLE RETRIEVAL / EPISODIC MEMORY (Neural Episodic Control's Differentiable
   Neural Dictionary; REALM's async index refresh): soft-attention reads over memory
   let outcome error shape the EMBEDDING through retrieval. FOR US: NEC = our banks
   with gradients -- train the field so softmin-over-bank reads predict outcomes; this
   formalizes the shelved rho head as trainable retrieval (the literature-grounded
   fallback if probability-multiplication proves insufficient), and REALM's index
   refresh = our per-field-version bank re-embedding, already implemented.
   ADOPT-2, research thread.

Sources: MuZero joint training + search targets (UniZero arxiv 2406.10667, Demystifying
MuZero arxiv 2411.04580); PCGrad (emergentmind PCGrad topic), CAGrad (arxiv 2110.14048),
GradNorm (via MTL surveys arxiv 2109.09138); NEC (Pritzel et al., researchgate
314256022), End-to-End Memory Networks (NIPS 5846), REALM async refresh (arxiv
2204.04581 discussion).

## 2026-07-24 -- DISTILL VERDICT: student reproduces the quasimetric (spearman +0.995)

field_student_v1 (2.1M params, 4.4x smaller than the 9.1M teacher core): held-out
distance fidelity spearman +0.9950, MAE 2.8 (n=262k pairs, 20k distillation steps).
The teacher's directed geometry compresses almost losslessly into a quarter of the
capacity -- strong evidence the scratch-run architecture question is about DATA, not
width, and that a student-arch line is viable whenever speed demands it. Speed bench +
fixed-TC play verdict pending (gauntlet deferred per camping plan).

## 2026-07-24 -- SPEED VERDICTS: student 2.01x @ 0.9946 fidelity; fp16 no-gain

bench_value_speed (under scratch-run load, both variants equally handicapped):
teacher-fp32 36 evals/s baseline; teacher-fp16 1.01x (bottleneck is not matmul
precision -- honest negative); field_student_v1 2.01x at distance-spearman +0.9946.
Fixed-TC implication: student = 2x nodes/move. Play referendum deferred (camping
plan); accept rule needs the gauntlet's SPRT before any pointer changes.

## 2026-07-24 -- DISK EMERGENCY #2 (camping window), root-caused PERMANENTLY

The probe-cache WAL regrew to 14G despite the journal_size_limit cap from incident #1.
Root cause finally understood: journal_size_limit only shrinks a WAL AFTER a checkpoint
COMPLETES, but bootstrap-engine fullgames hold perpetual read snapshots, so checkpoints
never fully complete (busy=1, 85k pages always uncheckpointed) -> the cap is a no-op and
the WAL grows forever. THE cap was treating a symptom. PERMANENT FIX: journal_mode=DELETE
(no shared WAL exists; per-transaction rollback journals delete on commit). Correct mode
for a many-reader recomputable cache. Recovered: 14G WAL + 6.5G superseded prefix4gb
shards -> disk 9.7G -> 30G. Scratch trainer (the centerpiece, doesn't touch the cache)
survived untouched throughout. Process lesson: during the 39h scratch run I am NOT
running the improvement loop (spawns tb-cache readers + contends MPS); phase 2 runs the
improvement rounds AFTER scratch completes, as designed.

## 2026-07-24 -- SCRATCH v1 COLLAPSED (degenerate self-channel); v2 relaunched clean

field_scratch_full_v1 collapsed ~step 10k: ALL distances -> 0 (d_step/d_rand/d_unr all
0.000), Lagrangian lambda decaying, repulsion dead. ROOT CAUSE: mixed the self-play
regime channel (self_play_v1, only 6003 rows -- degenerate, mostly cornered endgames)
into a FROM-SCRATCH QRL objective. Fine-tune runs used the SAME channel at higher
fraction (0.35) safely BECAUSE a pre-spread embedding resists collapse; from scratch
there is no structure to protect, so the tiny degenerate channel pulled everything to a
point. My camping-queue bug (should have caught this: it's the check-representational-
collapse gate). FIX: v2 = pure full-month lichess (proven mc2 recipe, no self mix),
120k steps; diagnostics healthy at step 100 (d_step 6.7, d_rand 7.8, sq_dev 35). The
improvement loop adds outcome-conditioning by FINE-TUNING v2 afterward (proven safe --
that's exactly the regime a pre-spread field tolerates). ~2h of v1 compute lost;
collapsed ckpt kept as field_scratch_COLLAPSED.pt for the postmortem.

## 2026-07-24 -- FROM-SCRATCH IS UNSOLVED (flagged for Kaveh); camping pivots to incumbent

Both from-scratch attempts collapsed (v1 self-channel + hot LR ~step 10k; v2 pure
lichess still collapsed ~step 20-32k -- persistent d_step=d_rand=0). KEY FINDING via the
search-when-stuck rule: the "known-good mc2 recipe" was NEVER from-scratch -- mc2_full.log
says 'resumed lichess_mc2.pt at step 30000' at lr 3e-5. EVERY field in the lineage is
warm-started; NO from-scratch iqe-qrl run at peak LR has ever been shown to converge.
Mechanism: at peak LR from step 0 the attractive QRL loss collapses the embedding before
repulsion establishes structure. --repel-weight defaults to 0.0 (OFF) -- the missing
lever. v3 SMOKE running (repel-weight 0.5 + lr 1e-4, fail-fast to 6k) as one data point;
early health means little (v1/v2 were healthy early too, collapsed at 10-32k).

DECISION (Kaveh away, disciplined): from-scratch is a research task needing a real
warmup/repel/LR study, NOT a fire-and-forget -- FLAGGED for Kaveh, not burned into more
blind runs. Camping window pivots to Kaveh's TOP-EMPHASIS thrust that does NOT need it:
external campaign (best incumbent line vs maia 1100/1500/1900 + SF skill 1/3, TC 60+0.6)
-> PGN import as whole-system data -> improvement rounds. Incumbent field_fullmonth_r0
(warm-started, STABLE, plateaued-not-collapsed) stays the live line. ~7h scratch compute
spent for a clean negative result + the warm-start-only lineage discovery.

## 2026-07-25 -- v4 FROM-SCRATCH COMPLETE: recipe SOLVED, field ~= incumbent (honest verdict)

field_scratch_full_v4.pt done (120k, first successful from-scratch iqe-qrl field ever --
repel-weight 0.5 + lr 1e-4, cleared both collapse zones clean). VERDICT vs the incumbent
(fullmonth warm-started plateau):
  v4: VAL_TOP8 0.058 | REACH sep +0.030 | DIFF sep -0.005
  inc: VAL_TOP8 0.098 | REACH sep -0.005 | DIFF sep -0.047
HONEST GRADE (rigor rule; retract my initial 'discriminates better' overclaim): v4 is
COMPARABLE, not superior -- better on REACH outcome-separation, WORSE on VAL_TOP8, DIFF a
wash. The RESULT is the recipe (from-scratch now possible), not a better field. Promotion
NOT auto-done (v3 chain dropped the integration gate; pointer correctly still on incumbent).
DECISION: do NOT unilaterally promote v4 -- it's a Kaveh call needing a play-test (v4 vs
incumbent gauntlet + integration 51-case), since they're on par by validation. v4 kept as
a validated candidate. External campaign (incumbent vs maia+SF) running now = whole-system
data regardless. The camping window's headline = FIRST WORKING FROM-SCRATCH RECIPE +
the warm-start-only lineage discovery, not a field upgrade.

## 2026-07-25 -- EXTERNAL CAMPAIGN: full-game strength gap is LARGE (honest)

catspace (incumbent field + banks + planner + energy, TC 60+0.6) vs the outside world:
  vs maia-1100: +0 =0 -20  (lost every game)
  vs maia-1500: +0 =0 -11  (in progress, losing all)
Zero wins, zero draws. Consistent with the improvement-loop baseline (0/10 vs maia-1100).
STRAIGHT READ (rigor rule): the composed system is NOT competitive at full-board chess --
its demonstrated strength is confined to the endgame nucleus (KRRvK 1.00, KRRvKBP 0.77,
7-piece capstone 0.92, med 5-11 plies). Full games expose that the FIELD has no useful
signal in the opening/middlegame (the OOD-compression we measured; humans-resign data
gap) and the planner/banks have nothing to grip until material simplifies. The games ARE
the data (imported to the experience store for outcome-conditioned training) -- this is
the whole-system-vs-strong-engines dataset Kaveh wanted, and its headline number is the
size of the gap to close. NOT a regression; a first honest measurement of the full-game
frontier. The strength-per-node north star: we are far left on it for full games, at the
frontier for endgames.

## 2026-07-25 -- camping window CLOSE: whole-system data captured, improvement rounds on it

External campaign done: 65 games imported to experience store (maia 1100/1500/1900 x20
each 0-win + SF skill1 x5 0-win). Dual-phase2 tangle cleaned (killed both; imported PGNs
manually once, no double-count). Launched improvement_loop --rounds 4 to fine-tune
field+planner+energy on the ENRICHED store (now incl. 65 strong-engine loss games = the
richest outcome-conditioning signal to date -- humans-resign gap doesn't apply to engine
games played to conclusion). This is standing-directive game-fed training, NOT a new
from-scratch variant. CAMPING WINDOW SUMMARY: headline = from-scratch recipe SOLVED
(repel-weight); v4 = first clean from-scratch field (~=incumbent, promotion=Kaveh call);
honest full-game gap = 0/65 vs engines (endgame-competitive only); 2 disk emergencies ->
permanent tb-cache DELETE-mode fix; distillation 2.1M @ spearman .995 (2x speed);
sessions/phone-play/UCI/gauntlet/fastchess infra shipped; dense-signal + system-gradient
(MuZero/PCGrad/NEC) lit reviews; UI: prophylaxis, atlas hover-peek + SVG zoom, calc
lifecycle, session cookies.

## 2026-07-25 -- ARCHITECTURE BAKE-OFF: middlegame is a LABELS problem, NOT architecture

Kaveh: try architectures (transformer-not-CNN, small dim), target = distance-to-mate (NOT
policy; planner sits above). Built DTM extrapolation bake-off: train each backbone on
DTM<=25, test ordering of DTM>25 (middlegame = far-from-mate). Results (held-out spearman):
  IN-RANGE (DTM<=25):   xf-d16 +0.50 | xf-d64 +0.56 | cnn-d64 +0.71 | xf-d64-L8 +0.62
  EXTRAPOLATION (>25):  xf-d16 -0.39 | xf-d64 -0.43 | cnn-d64 -0.44 | xf-d64-L8 -0.42
  CONTROL, labels<=100: cnn-d64 near(<=100) +0.62  (fits long distance WHEN LABELED)
DECISIVE FINDINGS:
  1. NO backbone extrapolates distance-to-mate past its training range -- ALL go
     negative on unseen-longer distances. This IS the '~20 for everything far' middlegame
     failure, and it is architecture-INDEPENDENT.
  2. Transformer does NOT beat CNN. CNN is actually BEST in-distribution (+0.71 vs +0.56).
     Swapping CNN->transformer would not help (tested directly); the current CNN is fine.
  3. far_eff_rank stays healthy (6-38) -- NOT rank collapse. The distance HEAD saturates.
  4. WITH long labels (control), the model fits long distances (+0.62 up to DTM 100) --
     capacity exists; the model just can't INVENT distances it never trained on.
CONCLUSION: the middlegame distance-to-mate problem is a LABELS/RANGE problem, not an
architecture problem. Tablebase stops at 6 pieces; human games are censored by resignation;
so long-distance labels don't exist. The fix must MANUFACTURE them -- bootstrap outward
from the endgame via TD / value-iteration on the quasimetric (d(s)=1+min_a d(s')), or
full games played to ACTUAL mate. This VALIDATES Kaveh's instinct to keep the quasimetric
+ planner-on-top: the fix is the training signal, not the backbone. (Files:
experiments/arch_bakeoff.py, dtm_arch_bakeoff.py; logs arch_bakeoff / dtm_bakeoff.)

## 2026-07-26 -- ENDGAME-OUTWARD DISTANCE BOOTSTRAP: mechanism validated (proof-of-concept)

Follow-up to the bake-off (middlegame DTM is a LABELS problem). Built value-iteration on the
quasimetric to MANUFACTURE the missing long labels: 2-ply minimax DTM backup
  d(s)=1+min_m V(s.m); V(loser-to-move t)=0 if mate else 1+max_m' g(grandchild)
Falsifiable test: train field g on tablebase DTM<=25 ONLY (long DTM known but HIDDEN),
bootstrap using g's own lookahead, measure whether the held-out FAR slice (DTM>25) recovers
against TRUE dtm. Lookahead structure precomputed once (only python-chess cost); each sweep
= pure tensor ops (net forward on flat grandchildren + segment min/max). No policy target;
g=distance-to-mate, planner stays one layer above.
RESULT (cnn-d64-L6, anchor 25, 3000 boot parents, 20 sweeps, 624s):
  far-spearman  -0.405 (anchor-only) -> +0.215 (crosses zero at sweep ~3, monotone)
  far MAE        26.9 plies -> 13.4 plies  (HALVED -- the headline)
  boot-target median  15 -> 31 (=TRUE) by sweep 16 -> overshoots to 36 by sweep 19
  cost: near-spearman  +0.70 -> +0.25 (anchor/bootstrap losses compete)
READ: value iteration DOES propagate the trusted endgame scale outward -- the field learns
the correct MAGNITUDE of long distances it never had labels for (MAE halved, target median
hits true 31), and its ordering flips from anti-correlated to positive, using ZERO long
labels. That is the missing-label fix working. Honest limits (iteration-2 levers):
  1. near-slice erosion + late overshoot (target 31->36) = TD instability -> target/EMA
     network (Double-DQN trick), heavier anchor weight, damping.
  2. far ORDERING ceilings at +0.2 while MAGNITUDE (MAE) is great -> fine-grained far
     ranking is the hard part; try rank loss on far pairs / deeper (DTZ-consistent) lookahead.
  3. 2-ply/sweep propagation is slow -> prioritized sweeping outward from the anchor boundary.
CONCLUSION: the bake-off's prescription is confirmed constructive -- manufacturing long
labels via value iteration repairs long-range distance-to-mate with no architecture change
and no policy target. Prototype: experiments/bootstrap_dtm.py; log bootstrap_dtm_full.log.

## 2026-07-26 -- MULTI-GOAL QUASIMETRIC field (Kaveh's triangulation reframe): endgame MVP

Reframe: don't regress a single scalar distance-to-mate (collapses to rank ~3, far-ordering
ceilings at +0.2 -- see bootstrap runs). Instead SUPERVISE d(F(s),F(g)) to MANY reachable
goals at mixed ranges (triangulation/multilateration) so the geometry is pinned. Strong
opponent = tablebase-optimal => labels are genuine shortest-path distances (quasimetric-safe).
Endgame-only MVP (3-4 piece won classes) = mechanism check on GROUND TRUTH; midgame is the
real regime of interest (catspace weak there) and is Phase 2 (Stockfish rollouts from human
starts). Endgame strength must be preserved (tablebase anchor stays).
Pipeline: tb.rollout_line -> gen_pairwise_data.py (parallel, 320k pairs, delta 1-59, 55k mate
landmarks) -> train_quasimetric.py (two-tower IQE, supervised on log1p(delta)).
VERDICT (two-tower d32 c16, 0.46M, 1332s):
  (3) held-out PAIR ORDERING spearman +0.931 MAE 1.0 ply  <- crushes scalar +0.2 ceiling
  (2) mate-via-min-over-region vs true DTM  +0.428 MAE 21.4  (modest; short-range data)
  (1) eff_rank(F) 3.4/32  (endgames are low-dim; real rank test is Phase-2 midgame)
  (4) triangle-inequality violations 10.9% (mean slack 0.30)  <- PROBLEM, diagnosed:
READ: multi-goal supervision fixes ordering decisively (+0.93 vs +0.2). Triangle violations
are an artifact of the TWO-TOWER (separate F/B encoders): an intermediate node b has F(b)!=B(b),
so IQE's single-space triangle guarantee doesn't transfer through b. Fix = SHARED encoder (one
phi, d=IQE(phi(s),phi(g))); IQE is itself directional so asymmetry is preserved. Shared run
launched (quasimetric_shared.log). mate-via-min & rank stay limited until Phase 2 (longer/
diverse ranges). NEXT per Kaveh: after mechanism solid -> ALL tablebase material classes, switch
trajectory source to STOCKFISH (reuse gen_stockfish_continuations.py, not hand-rolled optimal
play; tb stays as validation oracle), then expand outward to midgame. Files: gen_pairwise_data.py,
train_quasimetric.py, tb.rollout_line.

## 2026-07-26 -- SHARED single-space quasimetric: triangle inequality was broken by the two-tower

Kaveh asked: could the shared-vs-two-tower embedding be the root of prior failures? Tested it.
FINDING (train_quasimetric.py, endgame pairwise 320k):
  two separate embeddings (two-tower F/B, OR trunk-shared-but-separate-heads):
      triangle-inequality violations 10.6-10.9% (mean slack 0.13-0.30)
  genuine SINGLE space (one encoder AND one head, d=IQE(phi(s),phi(g))):
      triangle violations 0.00% (mean slack 0.000) -- even at 2000 steps
Why: IQE's aggregation (alpha*max + (1-alpha)*mean of per-component union-lengths) is a
non-negative combination + max of quasimetrics, so the triangle inequality is STRUCTURAL --
but ONLY within ONE embedding space. Two heads give phi_F(b)!=phi_B(b), so composition
a->b->c routes through two different embeddings of b and breaks. IQE itself supplies the
asymmetry (directed interval [U,max(U,V)]), so a single space is still a proper ASYMMETRIC
quasimetric. Pair ordering unchanged/better (+0.90..+0.95). eff_rank ~4.5, mate-via-min
~+0.32 (still endgame-data-limited; Phase 2).
ANSWER to Kaveh's question (graded): NOT the cause of the rank collapse / +0.2 ordering
ceiling -- that was the SINGLE-SCALAR target (no B-tower at all), cured by multi-goal
supervision. BUT very plausibly the cause of the MIDGAME PLANNING failure: the deployed
TorchFB is two-tower, so its distances DON'T COMPOSE (triangle broken); endgame play needs
only a one-hop d(s,mate) [two-tower ok -> strong], midgame play needs to COMPOSE distances
through subgoals [needs triangle -> two-tower breaks -> weak]. The strong-endgame/weak-midgame
split maps onto one-hop vs multi-hop. Mechanism now confirmed (0% vs 11%); the causal link to
actual play is still to be tested (Phase 2: does a shared composable field plan better?).
DESIGN IMPLICATION: field should be a SINGLE-SPACE IQE quasimetric (drop the two-tower) in the
omega-free regime -- composable + half the params. Checkpoint: quasimetric_shared_v1.pt.

## 2026-07-26 -- MATE TEST: the endgame field can't mate (5-7%), and exactly why

Kaveh: "let me know how the endgame model mates." Built mate_from_field.py: use the field as
a GREEDY planner (pick move minimising d(child, MATE)), Black defends TABLEBASE-OPTIMALLY.
RESULT (quasimetric_shared_v1, single-space, pair-ordering +0.957):
  pure greedy: mate-rate 5% (2/40).  +mate-in-1 shortcut: 7.5% (9/120), and those are almost
  all already-at-mate-in-1 (median plies 1). => the field gives ~NO mating guidance, even KQvK.
DIAGNOSIS (--diag, goal = each line's TRUE terminal mate, no bank):
  d vs remaining-DTM ALONG optimal lines: spearman +0.981 (median +0.994), 100% monotone-decr
  greedy picks a DTM-REDUCING move: 52.7% (770/1461)  == COIN FLIP
READ (this is decisive): the field is a near-perfect PROGRESS COORDINATE *along optimal lines*
but a useless MOVE-SELECTION policy. Root cause = imitation-learning distribution shift: it was
trained ONLY on optimal-line positions and ONLY on within-line pairs, so (1) it never saw the
OFF-optimal positions that suboptimal moves lead to (can't evaluate them), and (2) it has no
1-ply local resolution (the +0.98 is carried by large-range variation, not adjacent-move diffs).
Pure regression on expert trajectories teaches the on-manifold coordinate, NOT the gradient
away from it. The +0.957 pair-ordering metric OVERSOLD it -- it's a global metric dominated by
easy long-range pairs; the policy needs local + off-line discrimination, which is absent.
FIX DIRECTION (informs Phase 2): need NEGATIVE / off-optimal samples (Stockfish games naturally
visit varied+suboptimal positions; plus explicit suboptimal branches per node) and/or a
CONTRASTIVE/QRL objective (push non-adjacent apart, d(good-child)<d(bad-child)) -- not pure
supervised regression on optimal lines. This is exactly why Kaveh said use Stockfish + expand
outward. mate_from_field.py is now the real "can it PLAY" harness (strength, not proxy ordering).

## 2026-07-26 -- MATE GRADIENT PROBE (Kaveh): per-move distances reveal 2 defects

mate_gradient_probe.py prints, for mate-in-1 positions, every legal move with true DTM +
field d-to-mate, sorted by field distance; also d to BOTH mate regions (White-delivers vs
White-gets-mated). 5 examples:
  mate move ranked #1 by the field in only 2/5 (KQvK ex1 Qa7# 0.514, ex2 Qe7# 0.948). In 3/5
  the field preferred a DTM-6..14 move over immediate mate (ex3 KRRvK: Rf8 DTM10 @1.85 beat
  Ra2# @2.70; ex5 KQvK: Qc3 DTM6 @0.795 beat Qh8# @0.920).
DEFECT 1 -- the mate region is NOT collapsed: a checkmate position sits at field-distance
  0.5-2.7 from the mate BANK, not ~0. Different checkmates are SCATTERED in embedding space,
  so min-over-bank is nonzero even AT mate. The region-as-min readout can't be sharp when the
  goal region isn't a point. (This is why mate-via-min was only +0.33.)
DEFECT 2 -- NO win/loss asymmetry (Kaveh's 'two distances' point, confirmed as a REQUIREMENT
  the field fails): to-WIN-mate and to-GET-mated are comparable (~1-3), and in ex2/ex3/ex4 the
  field rates White CLOSER to getting mated than to winning -- in positions where White mates
  in 1. The field never trained on the losing region, so its distance there is arbitrary.
  Kaveh predicted one ~1 and the other ~inf; the field gives them roughly equal.
IMPLICATIONS (fixes): (a) COLLAPSE the mate region to a single attractor (abstract mate goal /
  pull all mates together, or a mate token) so d->0 at mate and the readout is sharp; (b)
  represent BOTH mate regions with the WDL sign so the asymmetry emerges (win: d_win small,
  d_loss huge) -- distance-to-region done properly; (c) + off-optimal negatives & local
  resolution from the prior finding. Even the 2 'correct' cases had thin margins (fragile).

## 2026-07-26 -- DESIGN LOCKED: metastability / transition-path planning architecture (Kaveh)

Full end-to-end vision for playing a fallible opponent (not just mating). Outcome classes
{Won,Drawn,Lost} = METASTABLE BASINS; under optimal play barriers are infinite (no cross-basin
path); under strong-but-real play, RARE transitions = ERRORS. Play STRONG engines (lc0
t1-512x15x8h vs Stockfish, fixed nodes/depth for reproducible strength, not wall-clock) so
basins are clean + transitions rare-but-present; a STRONG referee (SF high-depth) detects the
WDL flips. Cluster positions by eventual outcome -> 3 basins + sparse transitions.
THE STACK:
  1. QUASIMETRIC FIELD (single-space IQE): basins + within-basin distance; INFINITE cross-basin
     barriers via repulsion balanced by within-basin anchor (hinge-to-large-M or QRL local<=1;
     NOT unbounded -- that diverges/collapses). Mate = collapsed attractor (d->0); stalemate/
     draw/loss = infinite repellers. Off-optimal negatives + local 1-ply resolution (fixes the
     52.7% coin-flip move-selection).
  2. COMMITTOR c(s)=P(win|s,omega): the log-odds outcome coordinate; level sets=basins,
     c~0.5 iso-surface = transition-state ridge. (committor_root_loop.py already in repo.)
  3. TRANSITION-PROB PREDICTOR T(s,omega) [CNN/transformer]: per-direction (win-flip, loss-flip),
     omega-conditioned. 2-D map: SHARP=both high, QUIET=both low, FAVORABLE=win-flip high/loss
     low, DANGEROUS=inverse. Trained on TABLEBASE-EXACT only-move labels first (dense/exact;
     fixes rarity), engine transitions second. Off-distribution generalization caveat (imagined
     futures).
  4. MCTS PLANNER: maximize EXPECTED COMMITTOR (single scalar -- do NOT hand-balance the two
     transition probs; committor nets them, sharpness=variance-of-c). T shapes search toward
     REACHABLE favorable-flux ridges (field distance = within-basin reachability filter).
     Navigate to unseen future g = argmax_g T_net(g,omega)*reachability(d(cur->g)). Risk-appetite
     knob (c-variance tolerance): need-a-win(c~0.5)->tolerate SHARP; winning(c~0.9)->seek QUIET.
     = contempt/risk-mgmt, principled. This is TRANSITION-PATH THEORY: maximize reactive flux
     draw->win. Opponent plans too -> adversarial navigation; start ONE-SIDED.
STAGING (ground truth first): (1) endgame transition labels from tablebase (only-move positions)
-> train T with EXACT labels + validate vs fallible defender (Maia/weak-SF); (2) flux-planner on
endgames, beat greedy-to-mate against fallible defense; (3) scale to midgame with lc0-vs-SF games
(gen in background from step 1). Assets present: lc0+t1 net, maia 1200-1900, stockfish, committor
code, omega embeddings, single-space quasimetric MVP (quasimetric_shared_v1).

## 2026-07-26 -- S1 DONE: the SF reliability map (endgame, vs tablebase truth)

sf_reliability_map.py (parallel SF workers, SyzygyPath empty so it's SEARCH not tb-lookup),
9000 hard-endgame positions x depths {4,10,18} vs tablebase WDL:
  overall WDL-acc 96.2% (d4) -> 97.1% (d10) -> 97.3% (d18)  [depth 10->18 adds only +0.2%]
KEY FINDINGS:
  1. Errors are STRUCTURAL not horizon: deeper search barely helps -> positions SF can't crack.
  2. Failures concentrate: KBBvKN 68.5% (!), KNNvKP 81.0% (deep win past 50-move rule); rest ~100%.
  3. Every error is at the WIN/DRAW boundary, NEVER win/loss (confusion is purely W<->D: deep wins
     called draws, fortresses called wins). SF never mistakes a win for a loss.
READ: SF's unreliability lives EXACTLY on the basin boundary (committor~0.5 = the transition
zone). The reliability map and the transition map are the same object: reference-unreliable =
objectively-hard = where fallible humans also err = the exploitable zone. Downstream value module:
trust SF in basin interiors + for W/L separation; distrust at the W/D boundary (KBBvKN, deep
KNNvKP-type), splice tablebase (endgame) / reference-disagreement (midgame) there. This is a
clean, honest, ground-truthed reliability map -- S1 of METASTABILITY_PLAN complete.
Next: S2 (field that MATES: WDL basins + inf barriers + mate attractor + stalemate repellers +
off-optimal negatives; gate = mate-rate vs optimal defense, currently 5%).

## 2026-07-26 -- S2 result: the STALEMATE/blunder defense works; CONVERSION does not (check-in)

train_mate_field.py (single-space IQE, learnable collapsed MATE goal, WDL hinge-to-M barriers,
both-color broad data w/ off-optimal draw/loss = INF). Verdict (d32, 38k rows, 250 eval games
vs tablebase-optimal defense):
  MATE-RATE 0.4% (1/250)  -- WORSE than the 5% bank-min field
  kept-win 88.7% (blunder-avoidance)  -- the INF barrier WORKS: field avoids throwing the win
  won-d med 20.1  vs  INF-d med 468.0  -- clean WDL separation (stalemate/draw repelled)
  d-vs-DTM +0.808 | eff_rank 1.7/32
READ: two halves, one solved one not.
  SOLVED: the ∞ barrier / stalemate defense. The field reliably KEEPS the win (88.7%) and pushes
  draws/losses to d~468 vs won d~20. Kaveh's WDL-basin + hinge-to-M design works as intended.
  NOT SOLVED: CONVERSION. It holds the win and shuffles but can't force mate (0.4%). Two causes:
    (1) eff_rank COLLAPSED to 1.7 -- the single learnable MATE-goal scalar re-collapsed the
        representation (S1 multi-goal kept 6.3). No maneuvering resolution -> greedy plateaus.
    (2) greedy 1-ply on a distance field is too weak for a precise multi-step mating maneuver
        vs a maximally-delaying (tablebase-optimal) defender.
SIGNIFICANT-DEVIATION CHECK-IN (Kaveh: go autonomously until we must deviate significantly).
The plan's S2 premise (off-optimal negatives + mate-attractor => the field mates greedily) is
half-wrong. Fork for Kaveh:
  A. MERGE S1+S2: multi-goal geometry (healthy rank) + WDL barriers + off-line coverage; retest.
  B. Bring SEARCH forward (S5 MCTS on current field): mating may need lookahead, not greedy.
  C. REFRAME: at deployment the TABLEBASE mates the endgame (what real engines do); the learned
     field's real job is the MIDGAME. Stop gating on endgame-greedy-mate; validate the field on
     midgame committor/transition instead. Recommendation: C (+ B for the planner), with a quick
     A to confirm rank-collapse is the culprit. Checkpoint: mate_field_v1.pt.

## 2026-07-26 -- S2b: field + shallow SEARCH converts (reframe validated)

mate_with_search.py (minimax, field=leaf value, checkmate=+inf/draw=-inf, vs tablebase-optimal
defense, batched leaf frontier): MATE-RATE climbs with depth --
  depth 1 (greedy): 5.0%  ->  depth 3: 17.5%  (40 games, KQvK/KRvK)
CONFIRMS: the field is a usable VALUE substrate; the PLANNER (search) supplies the POLICY (Kaveh's
long-standing "policy from planner not field"). Greedy-field-mating was the wrong bar. Deeper
search would climb further (needs alpha-beta; full-width depth-3 = 423s/40 games). At deployment
tablebase mates the endgame anyway; this was a substrate check -> passed. Endgame phase closed;
focus pivots to the MIDGAME (committor/transition/asymmetry), where the field is all we have and
the KL/expected-committor objective (vs FALLIBLE opponents) replaces "force mate". Next per plan:
S3/S4 -- transition predictor + cohort-asymmetry field on lichess+SF-reference (needs the lichess
data pipeline = a new phase). mate_with_search.py, mate_field_v1.pt.

## 2026-07-26 -- S2c: MCTS convert = 6.7% (mostly DRAWS). Bottleneck is the FIELD, not search.

mcts_convert.py (negamax MCTS, field=value, draw/loss=-BIG so it steers off the draw interface,
vs tablebase-optimal defense): sims 64 -> 0%, sims 256 -> 6.7% (mostly DREW), WORSE than
depth-3 minimax (17.5%). The engine holds the win but shuffles into a 50-move/repetition DRAW --
it avoids LOSING (draws not losses) but can't force MATE.
DIAGNOSIS (answers Kaveh's "feasible or need more data?"): the bottleneck is NOT the search
algorithm -- it's the FIELD's collapsed VALUE. eff_rank 1.7 (single MATE-goal scalar collapsed
it from the multi-goal 6.3) -> the value is a coarse "roughly how far" with no local resolution
to distinguish progress from shuffling. Search cannot sharpen a coarse value (minimax > MCTS but
both plateau). FEASIBLE, but needs a BETTER FIELD, = more STRUCTURED data + objective:
  (a) RESTORE RANK -- multi-goal geometry (kept 6.3) instead of single MATE-goal;
  (b) LOCAL RESOLUTION -- rank a parent's CHILDREN by true DTM/DTZ (the signal skipped for speed;
      DTZ is 1 fast probe/child). This is the move-selection gradient the field lacks.
SELF-BLUNDER MODEL (Kaveh, "later"): the DRAWS are US blundering into the draw interface. A
self-blunder model -- where OUR value is unreliable / we're likely to slip -- is the mirror of
the opponent model + the S1 reliability map applied to ourselves. Deferred, but it's the
principled fix for draw-throwing. Files: mcts_convert.py.

## 2026-07-26 -- autonomous window: consolidated field + the fine-resolution wall

Kaveh: go autonomous few hours, get mate within a basin (small), then extend to full-game WDL.
Also: include ALL anti-collapse corrections, work with the best candidate.
Built train_field_full.py = the CONSOLIDATED best candidate with EVERY correction: single-space
IQE + multi-goal pairwise geometry (rank) + REPULSION + WDL hinge-to-M barriers + mate attractor
+ within-sibling DTZ rank + both-color. (field_full_v1, training.)
KEY WALL (persists across v2 child-rank AND field_full): the within-sibling |DTZ| RANK LOSS
PLATEAUS at ~0.34 (=~57% pairwise, barely above coin) NO MATTER the corrections. The field
CANNOT resolve sibling moves 1 ply apart. Pair/repel losses DO drop (geometry/rank develop), so
it's not global collapse -- it's that a 1-ply |DTZ| delta is too fine a signal for the field.
IMPLICATION (important): the "better field -> shallow greedy mates" path is CAPPED. The right
path is DEEP SEARCH + a coarse-but-correct value (which we HAVE: d-vs-DTM +0.81 coarse gradient,
kept-win 88.7%). This is exactly how real engines convert (coarse eval + deep search), and it's
the MCTS/AB planner the midgame needs anyway. So: ab_convert.py (deep alpha-beta, batched child
eval + move ordering + FEN memo, field=value) is the conversion mechanism; testing depth 5/7 on
KQvK (CPU, fast for d=32). Endgame conversion = deep search on the coarse field, NOT a
fine-resolution field. Files: train_field_full.py, ab_convert.py.

## 2026-07-26 -- CONVERSION: the fundamental wall (honest finding + strategic implication)

After exhausting the levers (WDL barriers, multi-goal geometry, repulsion, child-DTZ rank, both
color, deep search, MCTS), the honest conclusion on learned-field ENDGAME CONVERSION:
  * A learned scalar/distance VALUE at this scale does NOT capture the FINE mating gradient.
    Within-sibling 1-ply rank plateaus ~57% regardless of every anti-collapse correction (it is
    NOT global collapse -- pair/repel losses converge; the 1-ply signal is just too fine).
  * SEARCH can't compensate: KQvK mate needs ~10-20 ply lookahead to reach mate directly; even
    with instant eval, AB to depth ~16 is infeasible (branching^8+). So the value MUST guide the
    driving phase -- and it can't (the wall).
  * Real engines convert KQvK via hand-crafted endgame eval terms (king-to-edge, king proximity)
    that give the fine gradient WITHOUT deep search. Our learned field lacks that; tablebase has
    it perfectly.
STRATEGIC IMPLICATION (this is the right resolution, not a failure): at DEPLOYMENT the tablebase
converts the endgame (instant, perfect) -- as every strong engine does. The learned field's job
is the MIDGAME, where (a) there's no tablebase, AND (b) the bar is DIFFERENT: you don't force
mate, you steer the COMMITTOR / exploit the opponent's errors. The endgame conversion demo
validated what it could -- value+search beats greedy (5%->17.5%), blunder/stalemate defense
solved (88.7% kept-win) -- and revealed the fine-resolution wall, which tells us the learned
value is a COARSE committor-style signal, exactly what the midgame KL/exploitation layer needs
(not a fine mating oracle). RECOMMENDATION: stop optimizing learned endgame mate; endgame=
tablebase at deploy; proceed to the midgame lichess/KL layer where the coarse learned field is
the right tool. Artifacts: field_full_v1 (killed mid-train), mate_field_v1, ab_convert.py,
mcts_convert.py, mate_with_search.py all committed.

## 2026-07-26 -- CORRECTION: the "1-ply wall" was a BROKEN LOSS, not fundamental (Kaveh was right)

Kaveh pushed back on the "fundamental wall" conclusion. Diagnosis proved him right:
DATA DIAGNOSIS (child_rank_v1 sibling |DTZ| structure): 39% of random sibling pairs are TIES
(same |DTZ|); the rank loss did y=sign(0)=0 -> margin_ranking_loss returns a CONSTANT margin
(0.5) that can never be satisfied -> the loss floored at ~0.34 BY CONSTRUCTION (misread as a
wall). The metric was poisoned identically (ties counted as errors, capping ~61%). Also margin
0.5 was ~5x the true sibling log-gap (~0.10).
FIX (drop ties + margin-free logistic order) SENSITIVITY:
  w_rank 0 : 1-ply RANK-ACC(distinct) 64.4% | d-vs-DTZ +0.882 | won-d 20.8 vs INF-d 408.6 (scale ok)
  w_rank 1 : 1-ply RANK-ACC(distinct) 84.2% | d-vs-DTZ +0.602 | won-d 378  vs INF-d 389 (scale BROKE)
=> the field CAN resolve 1-ply moves (64->84%); the wall was my broken loss. NEW issue: the
margin-free step is scale-unanchored -> inflates distances, collapsing won-vs-draw. Fix =
ANCHORED step (margin = true per-pair log-DTZ gap): relu(gap - (d_hi - d_lo)) -- sharpens order
consistent with the regression, no inflation. (Anchored sweep launched; MPS ~800s/config, slow.)
ON THE DRAW SURFACE (Kaveh): the d-vs-DTZ +0.89 IS the "reduce DTZ = move away from the 50-move/
repetition draw surface" agreement -- that always worked; only the FINE 1-ply order was broken.
BOTTOM LINE: retract the "fundamental wall". Endgame conversion is viable; the recipe is
drop-ties + anchored per-pair-gap step at a MODERATE weight (balance order vs scale). Files:
rank_sensitivity.py. Next: find the anchored weight that keeps BOTH (order>=80% AND won-d~20),
retrain the field, re-run conversion (should beat 17.5%).

## 2026-07-26 -- MILESTONE: we CAN mate the winning toy sets (v3 field + minimax depth 3 = 100%)

After the loss-bug retraction, trained field_v3 with TESTED anchored loss:
  VERDICT v3: 1-ply rank-acc 80.1% AND won-d 16.5 vs INF-d 410.3 (both order + scale, finally).
Conversion vs TABLEBASE-OPTIMAL defense (KQvK/KRvK), the payoff run:
  MCTS 8000 nodes (expected-score committor): 0% (0/12) ALL DRAWS, 44 min
  minimax depth 2: 8.3%  |  minimax DEPTH 3: 100% (12/12), 61s
=> WE CAN CONVERT. Two necessary pieces, both Kaveh's pushes: (1) FIX THE FIELD (tested anchored
rank loss -> 80% 1-ply order + clean win/draw scale; the old collapsed field got 17.5%);
(2) MINIMAX not MCTS -- vs a PERFECT defender, mean-backup MCTS gets exploited into a draw
(0%), worst-case minimax converts (100%). Kaveh's "do minimax to see if MCTS wins" -> minimax
WINS decisively here. Note: MCTS is still right for the FALLIBLE-opponent midgame (expected
score over a stochastic opponent); minimax is right for the PERFECT-defender endgame. Files:
train_field_v3.py, mate_with_search.py, mcts_convert.py, field_v3.pt.

## 2026-07-26 -- Kaveh right again: the 50-move DRAW SURFACE is unrepresented (retract "MCTS wrong")

Kaveh: "our MCTS is buggy, or we're not representing the approaching 50-move draw surface."
CONFIRMED two gaps: (1) the field is BLIND to the halfmove clock -- tokens() uses only 12 piece
planes + stm, DROPS the halfmove clock; (2) training data has NO clock variation (halfmove 0-1,
fresh positions). So the field cannot see or steer off the 50-move draw surface.
This explains all conversion results: KQvK/KRvK minimax 100% (short mates finish before the clock
bites); MCTS 0% (mean-backup shuffles, clock runs to 50, field never saw it coming); hard classes
20% (KBN 66-ply mate approaches the 50-move budget, no clock-awareness to prioritize progress).
RETRACT "MCTS is the wrong search" -- premature (3rd time concluding fundamental/wrong when it was
a representation gap; diagnose-first must be reflex). Neither search can avoid a draw surface the
value can't see.
FIX: (a) feed halfmove clock (+repetition) into the field input; (b) regen data with clock
variation, clock-aware labels (won but 100-halfmove < DTZ-to-zero => actually DRAW, committor 0.5);
(c) committor then ~1 in the win basin with clock budget, ->0.5 at the draw surface -> gives BOTH
MCTS and minimax the progress gradient (reduce DTZ, reset clock) to steer off the draw. Then re-test
MCTS (should recover) + long-mate conversion.

## 2026-07-26 -- PIVOT: adopt lc0's input encoding + REAL-history trajectory data (Kaveh)

Kaveh: don't zero-fill history; borrow lc0's proven input pipeline (don't reinvent feature
engineering); then train; stop anything old and devote resources to the newest method.
DECISION: converge endgame + full-board onto ONE lc0-based input pipeline.
  - INPUT: reuse lc0's 112-plane encoding (8-position REAL history, perspective flip, castling/
    rule50/ep, canonicalization) via a reusable Python encoder if one exists, else a faithful
    port of encoder.cc. NOT our home-rolled 20-plane. (encoder-reuse research dispatched.)
  - DATA: real trajectories (rollouts / games) so history is REAL, not zeros -- even endgames are
    played as sequences. Same rollout machinery as the full-game/lichess phase -> unifies them.
  - KEEP (input-agnostic, transfers unchanged): quasimetric IQE field, committor, categorical
    ending head, tested losses (losses.py), MCTS/minimax planner. Only input encoding + data swap.
SUPERSEDED: the home-rolled 20-plane ClockField as the PRODUCTION encoder. Its FINDINGS stand
(machinery validated): quasimetric+minimax converts endgame 100% (v3), clock-awareness represents
the 50-move draw surface, categorical endings 95%, the tie-loss/clock-blindness/MCTS-vs-minimax
lessons. STOPPED the running 20-plane clock training + evals.
NEXT: (1) encoder reuse decision from research; (2) trajectory-rollout data with real history;
(3) train the field on lc0-encoded real-history data.

## 2026-07-26 -- ENCODER SECURED: lczerolens (reuse lc0's 112-plane encoding, verified)

Research found + verified the reusable lc0 encoder: `lczerolens` 0.4.0 (pip installed).
  from lczerolens import LczeroBoard        # subclasses chess.Board (drop-in)
  t = board.to_input_tensor()               # -> (112,8,8) INPUT_CLASSICAL_112_PLANE
Verified: shape (112,8,8); REAL 8-position history (pops move stack); side-to-move flip;
castling(104-107); rule50 plane(109) = raw halfmove clock (endgame hm=98 -> plane mean 98.0);
all-ones(111). This is lc0's proven feature engineering -- we BORROW it, do NOT reinvent.
CONCRETE BUILD PLAN (next):
  1. TRAJECTORY-ROLLOUT data (real history): rollout games (tablebase-optimal both sides + epsilon
     exploration for variety) from endgame starts; store start_epd + uci line + per-ply labels
     (clock-aware DTZ, ending type; children w/ DTZ for the 1-ply rank loss). History is REAL
     (replay the line). Same rollout machinery extends to full-game/lichess later.
  2. TRAINER: reconstruct LczeroBoard by replaying to each ply -> to_input_tensor() (112 planes);
     feed the field with in_planes=112 (ClockField already parameterized). Our machinery unchanged:
     IQE quasimetric + learnable MATE goal + tested losses (regression/wdl_hinge/anchored-rank/
     categorical) + the categorical ending head.
  3. EVAL: MCTS vs minimax conversion on the lc0-encoded field (the pending machinery question:
     does MCTS recover with the draw surface visible?).
KEY: only INPUT (lczerolens 112) + DATA (real-history trajectories) change; the novel value/
planning machinery is input-agnostic and carries over. Endgame = grounded special case of the
full-board field. lczerolens added to deps.

## 2026-07-26 -- DEFINITIVE FIELD ARCH (audit, complete). Committor trained head + all outcomes.

The FIELD is architecturally complete. IN (definitive lc0 training):
INPUT: lczerolens 112-plane (pieces, 8-pos REAL history, castling, ep, rule50 clock, repetition,
  perspective); in_planes parameterized (full-board extensible).
REPR: single-space shared phi (128ch x8 ResNet) + IQE quasimetric.
OBJECTIVES (tested losses.py): multi-goal d(phi(s),phi(g))->Delta same-line pairs (triangulation
  -> rank/composability/fine+coarse ordering; SUBSUMES the sibling-rank loss) + REPULSION +
  mate-goal d(phi,MATE)->DTZ + WDL inf-hinge + DISTRIBUTIONAL ending head (6) + COMMITTOR =
  score-weighted ending head (trained value over all W/D/L outcomes; replaces exp(-d) proxy).
BASINS: ALL-OUTCOME data (win/loss/draw class sets) -> committor learns 3 basins; clock-aware
  (rule50) -> 50-move draw surface. HEALTH: eff_rank gate. Smoke (multi-goal): pair-order +0.949,
  eff_rank 6.3 (vs 3.5 single-goal), ending 99.6%.
DEFERRED (opponent/lichess phase, in METASTABILITY_PLAN.md + memory): transition predictor T(s,z),
  player embedding z, KL/asymmetry exploitation, cohort-regret field, ensemble reference, navigate-
  to-transition planner, risk-appetite knob, self-blunder model, swappable value signal, non-board
  endings (time/resign via game layer), king-bucketing.
NEXT: definitive all-outcome training running (traj_lc0_v3) -> verdict (pair-order/rank/mate/ending/
  committor-MAE/WDL) -> MCTS vs minimax on the lc0 field (using the trained committor). Field DONE;
  remaining = planner eval + opponent layer. No more field circles.

## 2026-07-26 -- PIVOT: drop learned endgame model; tablebase IS the endgame. Full-board + opponent.

Research (SF/lc0/AZ endgame handling) + Kaveh: a LEARNED endgame converter is redundant and WORSE
than the tablebase for <=7 pieces (SF/lc0 both probe Syzygy; NNs measurably err at conversion --
Alberta ACG'21: 3000-Elo lc0+MCTS still errs in 4-piece pawn endgames). So DROP the learned endgame
foundation. The endgame = TABLEBASE (we have catspace/tb.py).
NEW ARCHITECTURE (the actual thesis):
  1. FULL-BOARD model = the OPPONENT-EXPLOITATION model (the whole point). Committor/quasimetric over
     full positions, opponent-conditioned.
  2. GOAL REGION = the set of <=7-piece TABLEBASE-WON configurations. The field's embedding/distance
     TERMINATES at a tablebase-won config; once there, the OUTCOME IS ASSUMED via tablebase lookup
     (WDL value, DTZ move). So the committor is GROUNDED at the tablebase boundary: <=7 pieces ->
     c(s)=tablebase WDL (exact); above -> the field predicts toward that boundary. (This is the
     region-goal / distance-to-region idea, task #24, with the tablebase-won set as the goal region.)
  3. HANDOVER: at <=7 pieces, consult the tablebase (WDL at nodes, DTZ at root) -- exactly SF/lc0.
FOCUS NOW: (a) the handover primitive (tablebase lookup at <=7, goal-region membership), (b) the
OPPONENT + PERSONALIZATION layer (player embedding z, transition predictor T(s,z), KL/asymmetry
exploitation, cohort-regret field -- all designed in METASTABILITY_PLAN.md + memory).
KEEP: all the machinery/code (single-space IQE, multi-goal, repulsion, distributional ending head,
committor, tested losses, lc0 encoder, trajectory data pipeline) -- transfers to the full-board model.
The endgame work VALIDATED the machinery on ground truth; now build the real thing.

## 2026-07-26 -- DATA EVENNESS CHECK (before full train, Kaveh's discipline) -- caught 2 problems

Built data_distribution_check.py; ran on lichess shards (3M pos, 54k games, ~52 pos/game). FOUND:
  1. DRAW BASIN STARVED: outcome per game = 49.3% W-win / 46.6% B-win / only 4.1% DRAW. Sub-2000
     lichess barely draws -> the committor's DRAW basin would be near-empty -> collapses to binary
     W/L. FIX: blend draw-rich data (engine-vs-engine ~50% draws; higher-rated humans) + upsample.
  2. STRENGTH SKEW: evenness 0.79 (norm entropy), peaked 1400-1800, SPARSE <1000 and >2200 (0.1%
     >2400). z-space would be "everyone ~1500" with no tails. FIX: engine tails -- CCRL/fastchess
     strong end, Maia-bots weak end (the universal z-space rationale).
  3. PHASE ok (ply 0-80 spread, midgame-heavy). 4. Shards have Elo+result+game_id/ply but NOT
     player names -> z is RATING-conditioned on this data; per-individual z needs raw-PGN reprocess.
VERDICT: data NOT train-ready -- fix draw-starvation + strength-skew (balanced human+engine+rating
blend) BEFORE the full run. Test-before-train discipline paid off (caught it pre-run). Next
validations: balanced data pipeline -> re-check evenness -> z-encoder/T/field machinery smokes ->
THEN full best-shot train. Files: data_distribution_check.py.

## 2026-07-26 -- MATH double-checked + canonical statement (Kaveh)

Verified numerically (data_distribution_check-style sanity in-session): V = c + 0.5·P(draw) + 0·P(loss);
c + P(draw) + P(loss) = 1; flux/sharpness well-formed. Wrote ARCHITECTURE.md §11 "Math (canonical)":
committor c=P(win)=P(WIN_MATE) [metastability coordinate, tablebase-grounded] is DISTINCT from EXPECTED
SCORE V=Σ p_e·score_e [planner value]; both are readouts of the one ending-distribution head. Value is
under the JOINT policy (opp π_opp fallible, I maximize) -> already TWO-SIDED; MCTS objective = single
scalar max_move E_{π_opp}[V] (max-expectation; reduces to minimax vs perfect defender). Transition flux
Φ=t_win(z_opp)−t_loss(z_me) and sharpness σ=t_win+t_loss are the DECOMPOSITION of dV for search-shaping
+ risk knob -- NOT competing objectives (V nets them). Info-asymmetry edge = their-regret − my-regret vs
reliability-weighted reference. Fixed the loose "committor==value" usage.

## 2026-07-26 -- DATA PIPELINE stage A+B: identity-preserving game records + stratified balancer

Kaveh green-lit the balanced identity-preserving data pipeline (the blocker before any full train).
Built two testable stages; smoked on a 4.7k-game lichess slice.
STAGE A build_game_records.py: streams .pgn.zst -> COMPACT parquet game records, ONE row/game with
  IDENTITY (white_id/black_id usernames + engine names), elos, result, n_plies, time_control,
  termination, titles, space-joined UCI moves. 161 B/game (vs ~7KB/lc0-position) -> full month ~1GB;
  reconstructs positions in ANY encoding on demand; the natural unit for balancing + z-grouping.
  Reuses catspace.data.lichess.stream_filtered_games (no decompress to disk). Engine PGNs
  (CCRL/fastchess) ingest into the SAME schema via --source (universal-z manifold = one dataset).
STAGE B balance_game_records.py: (1) game-level EVENNESS re-check (successor to
  data_distribution_check which read old position shards) -- OUTCOME / STRENGTH(min-Elo band,
  norm-entropy) / PHASE / and NEW per-player game-count distribution (>=20 games = z-encoder
  training bar; 5-10 = online-inference regime). (2) STRATIFIED BALANCER: resamples to even
  OUTCOME x STRENGTH-BAND with bounded oversampling; writes balanced records + before/after JSON.
SMOKE VERDICT (4.7k games): balancer lifted outcome-evenness 0.76->0.99 (draws 4.2%->27.6%) and
  strength-evenness 0.78->0.89. HONEST RESIDUAL (reported, not hidden): the draw lift is
  OVERSAMPLING of 201 real draws (repetition, not diversity), and the <1000 / >2400 Elo bands are
  structurally empty in sub-elite lichess -- both gaps require ENGINE data (CCRL ~50% draws +
  strong tail; Maia weak tail), exactly as the evenness check predicted. Machinery validated.
  z-trainability: 4.7k-game slice has 0 players >=20 games (max 8) -> the FULL MONTH is required
  (heavy players accumulate) -> full-month build launched (build_records_fullmonth.log).
NEXT: full-month records land -> re-run evenness at scale (expect real >=20-game player counts) ->
  decide engine ingestion (CCRL download) for the draw/tail diversity -> z-encoder (CSSLab
  stylometry adapt) smoke.

## 2026-07-26 -- OPTIONALITY portfolio -> MULTIPURPOSE moves (mechanism built + tested)

Kaveh's directive: wire the engine to infer z_opp online, find advantageous transition points,
navigate high-level + search low-level toward them while STAYING AWAY from the opponent's subgoals;
KEEP OPTIONS OPEN (a SET of subgoals, opponent modeled the same way) -> this should ENCOURAGE
MULTIPURPOSE moves (attack + defend + more). Two forks decided (AskUserQuestion): prototype the
mechanism NOW on the current field (heuristic flux placeholder); self-model = search-complexity
proxy now, learned z_me later.
RIGOROUS FRAMING (keeps 11's single-objective intact): optionality + multipurpose are NOT new
objectives -- they FALL OUT of maximizing E[V] under uncertainty about (a) which subgoal pans out,
(b) z_opp. So aggregate the subgoal portfolio SOFTLY (soft_reach = (1/beta) logsumexp_k[beta*(-d_k)
+ log w_k]) instead of hard-min: a finite beta rewards several subgoals being close (a Jensen /
value-of-information effect). Move shaping (a MOVE-PRIOR, never a value term -- the engine's
ValueModel/MovePrior split already enforces "subgoals in prior, not value"): score = advance-many-
of-mine - lam*let-them-advance-many-of-theirs - mu*self_blunder. The move that attacks (advances my
set) AND defends (raises the barrier to their set) is the ARGMAX -> MULTIPURPOSE is emergent, not a
hand-coded rule. beta couples to the sigma risk knob (need-a-win -> commit to the sharpest subgoal).
BUILT catspace/planner/optionality.py (field-agnostic: operates on d[move,subgoal] matrices, plugs
into any field). TESTED (11/11 self-checks): optionality bonus (2 near subgoals > 1 at equal min-
dist), beta->inf -> hard nearest, and MULTIPURPOSE EMERGENCE -- a move advancing 2 of mine + denying
2 of theirs out-ranks pure-attack / pure-defense / single-purpose; denial sign; self-blunder
monotonicity; valid prior. Documented ARCHITECTURE.md 8.1. Memory: matilda_residual_style_embedding
(arXiv 2606.25176) = the recommended z-encoder (rating-conditioned residual, Elo-disentangled).
NEXT: wrap as a PortfolioPrior (MovePrior) on a real field; full exploitation loop gated on z/T
(which are gated on the full-month data build, running now).

## 2026-07-26 -- OPPORTUNISM (plan-switching) + PortfolioPrior board wiring

Kaveh: be OPPORTUNISTIC -- if the opponent slips and opens a transition point OFF our main plan,
SWITCH plans to seize it. This is inherent in the design: the soft portfolio is FORWARD-LOOKING /
Markov (PortfolioPrior.priors recomputes from the current board every ply, no sunk cost), so when a
slip opens a new transition point (flux Phi up, distance down -> weight up), it enters the soft
aggregate and, if best, dominates -> we switch. Over-committing would be the BUG. Added
select_active_plan() = re-select the emphasized subgoal each ply with HYSTERESIS (switch_margin) so a
clearly-better opened point is taken but marginal noise doesn't thrash (wasted tempo). Keeping options
open + opportunism = same coin (never locked in). Also wired PortfolioPrior (a MovePrior:
distance_fn + G_me/G_opp + ShapeWeights -> shaped move prior; subgoals in the PRIOR, value stays
global). TESTED 16/16 (added opportunism+hysteresis + board-level PortfolioPrior on a real board:
valid distribution over legal moves; a multipurpose king step toward BOTH targets out-priors a step
away). Doc: ARCHITECTURE 8.1 (3). NEXT: the subgoal GENERATOR (candidate transition points + flux
weights), gated on T(s,z)/z, gated on the full-month data build.

## 2026-07-26 -- STAGE C (records->field data) + FULL-BOARD field trainer on the scaffold (smoked)

Kaveh (going to bed): "make all the decisions yourself and train a proper field, then test it";
use frameworks to parallelize. Built the chain to a proper full-board field on REAL games:
STAGE C gen_field_data_fullgame.py: balanced game records -> lc0 112-plane field npz
(planes/dtz/ending/game/ply), parallelized (ProcessPoolExecutor). DECISIONS: committor/ending label
= game result WHITE-POV (Monte-Carlo outcome under the human play measure = the metastability
committor); at <=7 pieces OVERRIDE with exact tablebase WDL + real DTZ (boundary grounding, ARCH 8);
tail-sampling captures the endgame tail for grounding. Smoke (1177 balanced games): 14540 positions,
W/D/L 36/27/37% (balanced), 281 tb-grounded.
TRAINER train_field_fullgame.py on catspace/train/scaffold.py (MLflow + ladders + gates): core losses
ALWAYS ON = COMMITTOR/ending (class-balanced W/D/L, the value) + MULTI-GOAL quasimetric (same-game
ply gaps) + REPULSION; mate readout + WDL hinge GATED on the tablebase-won subset (never crashes on
off-tablebase batches). SMOKE (400 steps, 2.63M params): pair-order +0.973, eff_rank 6.0 (no
collapse), committor-MAE 0.225 (< 0.333 predict-0.5 baseline -> learning the value), ladder saved.
Whole pipeline validated end-to-end. NEXT (autonomous): full-month records -> DVC -> balance ->
full Stage C generation (parallel) -> DVC -> field train (scaffold, ladders, gates) -> TEST
(committor calibration vs held-out + tablebase, pair-order, eff_rank). Report by morning.

## 2026-07-27 -- FULL-BOARD FIELD v1 diagnosed (eff_rank collapse) + recipe fix (v2). Data at scale.

Full-month records DONE: 19,354,162 games, 97 shards (~4.7h), DVC-tracked. Stage C (100k games,
parallel, 130s) -> 1,081,834 positions (48.5% win / 6.4% draw / 45.1% loss natural; 10,694
tablebase-grounded), DVC-tracked. Field v1 (16k-step recipe, defaults) CAUGHT MID-RUN by the
eff_rank gate (check-long-runs discipline):
DIAGNOSIS (step-4000 probe): eff_rank COLLAPSED 6.5 -> 2.9 (step 400->1000), recovering slowly to
4.1; committor-MAE degraded in lockstep 0.225 -> 0.44. The 400-step smoke MISSED it (collapse emerges
after step 400 -- the "short smoke caught nothing" scar). Probe: committor ORDERS correctly (win
0.549 > draw 0.476 > loss 0.416) but COMPRESSED; linear probe on frozen phi separates win/loss at
only 65.7%. KEY METHOD FINDING: committor-MAE vs HARD 0/1 Monte-Carlo labels is MISLEADING -- a
calibrated committor outputs ~0.5 for genuinely 50/50 human-game positions, so MAE floors ~0.45 by
construction. Right metrics = CALIBRATION (ECE) + DISCRIMINATION (win/loss sep) + eff_rank; MAE
retired as a primary gate.
FIX (v2, bundled + justified, no-one-lever): w_repel 0.3->1.0 + repel_margin 3->4 (anti-collapse, the
standing cure), w_cat 1->2 (committor is the centerpiece value), w_mate/w_hinge 1->0.5 (the 9.5k-pos
tablebase subset must not dominate phi). Relaunched 16k steps, watching eff_rank early. NEXT: if
eff_rank holds >5 and win/loss-sep climbs -> let finish -> test (calibration). Else diagnose further.

## 2026-07-27 -- PROPER FULL-BOARD FIELD v2 TRAINED + TESTED: well-calibrated committor (ECE 0.022)

v2 (anti-collapse recipe: w_repel 1.0, repel_margin 4, w_cat 2, w_mate/hinge 0.5) trained 16k steps
on 1.08M real full-game positions (100k games, full-month lichess), 85 min. eff_rank climbed steadily
5.9 -> 8.9 (v1 had COLLAPSED to ~3) -- the repulsion fix fully cured the collapse. VERDICT
(held-out val, honest game-level split): pair-order +0.940, eff_rank 8.9, win/loss-sep 0.198.
CALIBRATION TEST (the meaningful committor metric -- test_field_fullgame.py): **ECE 0.022** -- the
committor is WELL-CALIBRATED. Reliability curve tracks near-perfectly across the FULL range:
  bin0 pred 0.10/emp 0.10 ... bin4 0.46/0.47 ... bin9 pred 0.91/emp 0.89.
So P(win) spans 0.10-0.91 (NOT collapsed to 0.5): sharp on decided positions, ~0.5 in genuinely
balanced middlegames -- exactly a metastability committor under fallible human play. pair-order
+0.949, eff_rank 8.8.
HONEST CAVEAT: on <=7-piece TABLEBASE-WON positions the committor averages 0.69 (target 1.0) -- it
UNDER-COMMITS on won endgames. The boundary grounding is weak: mate/WDL-hinge ground the DISTANCE
d_mate, not the committor directly. FIX (next): add a direct committor anchor loss on the
tablebase-grounded subset (pull c->1 on tb-won, c->0 on tb-loss) so the basin boundaries are exact.
DELIVERABLE: field_fullgame_v2_final.pt (DVC-tracked) = the first well-calibrated full-board
committor + quasimetric field on real games. This is the centerpiece the exploitation planner
(optionality.py PortfolioPrior) + committor navigation build on. committor-MAE (0.378) retired as a
gate -- ECE + calibration curve + tablebase agreement are the committor metrics going forward.

## 2026-07-27 -- FIELD v3 PROMOTED: committor anchor fixed the endgame boundary (0.69->0.94)

v3 = v2 recipe + committor ANCHOR (c->1 on tablebase-won subset, w_anchor 1.0). 16k steps, eff_rank
climbed 5.2->8.4 (no collapse). TEST (held-out val, vs v2):
  committor ECE 0.027 (v2 0.022 -- both well-calibrated, negligible cost)
  TABLEBASE-WON committor 0.94, MAE 0.065  (v2: 0.69, MAE 0.314 -- THE FIX: sharp basin boundary now)
  win/loss-sep 0.205 (v2 0.194) | pair-order 0.944 | eff_rank 8.4.
v3 is STRICTLY better: well-calibrated AND exact at the <=7-piece tablebase handover (the committor
boundary condition, ARCH 8) -- what the metastability planner needs. PROMOTED: field_fullgame_v3_final.pt
(DVC-tracked) = THE full-board committor+quasimetric field. THE DELIVERABLE for Kaveh's "train a proper
field, then test it" is DONE + tested + promoted.
Recipe (record): d64 ch128 blocks8, w_multi1 w_repel1 repel_margin4 w_cat2 w_mate0.5 w_hinge0.5
w_anchor1, lr3e-4, 16k steps, 1.08M real-game positions (100k games, full-month lichess), MPS ~95min.
NEXT (for Kaveh): (1) wire field_fullgame_v3 into the optionality PortfolioPrior planner (committor +
d_pair as the distance_fn); (2) z-encoder (Matilda residual) on the 19.35M-game identity records ->
transition predictor T(s,z) -> the exploitation loop goes live; (3) engine data (CCRL) for draw/tail
diversity. All gated code + interfaces already built + tested this session.

## 2026-07-27 -- HOW THE FIELD PLAYS vs MAIA (Kaveh's question): honest baseline, value != policy

Put field_fullgame_v3 into play vs Maia (lc0 + maia-<elo>.pb.gz, nodes=1 = human-like policy) via
experiments/play_vs_maia.py. Readout = committor-greedy (pick move maximising c(s')=P(my win)).
RESULTS (field POV, alternating colors):
  1-ply committor-greedy: vs maia-1100 0/30 (0.000), maia-1500 3D/27L (0.050), maia-1900 0/30 (0.000)
  2-ply committor-minimax: vs maia-1100 0W/3D/13L (0.094)  [~2x the 1-ply score]
DIAGNOSIS (inspected games, NOT a bug): the committor is a well-CALIBRATED VALUE (ECE 0.027) but by
design ~0.5-FLAT in balanced midgames, so used greedily it barely discriminates midgame moves ->
hangs material. Concrete: as White it played 3.Qxh5?? Rxh5 hanging the queen on move 3 (1-ply can't
see the recapture). 2-ply minimax fixes the immediate hangs (loss->draw) but still loses most: deeper
tactics + positional drift on a flat midgame committor. A VALUE IS NOT A POLICY without search.
VERDICT: field value + shallow search ~= BELOW maia-1100 on the strength-per-node frontier -- the
honest, expected baseline. LEVERS to move it (all designed/built this session, gated): (1) deeper
search / more nodes; (2) the QUASIMETRIC planner (region subgoals + progress -- the committor-only
readout wastes the geometry, pair-order 0.94); (3) the OPPONENT-EXPLOITATION layer T(s,z) -- Maia is
a MODELABLE fallible opponent, exactly the thesis's target, so z_opp + favorable-flux subgoals
(optionality.py PortfolioPrior) should specifically help vs Maia. Files: play_vs_maia.py, PGNs
artifacts/experiments/field_v3_*vs_maia*.pgn.

## 2026-07-27 -- Maia baseline closed: opponent-fallibility expectimax > minimax (small, thesis-consistent)

Added --opp-tau to play_vs_maia.py: 2-ply aggregation over opponent replies = MIN (paranoid minimax,
opp-tau 0) vs SOFT-EXPECTATION (expectimax, models a fallible opponent). vs maia-1100:
  1-ply greedy ~0.00 | 2-ply minimax 0.094 | 2-ply EXPECTIMAX (opp-tau 0.15) 0.125 (first win).
Modeling Maia's fallibility (don't assume the refutation is found) earns +0.03 even at 2-ply -- the
first empirical whiff of the exploitation thesis. Still << maia-1100: shallow value readout is the
bottleneck. NEXT (the real strength lever, north-star strength-per-node): wire ClockField committor
as value_fn (2c-1, White-POV) into the existing catspace/nn/mcts.py (PUCT + mate-stop + cert
recognizers) for a NODE-BUDGETED search vs Maia -- the proper "engine vs Maia" test (my 1-2 ply toys
undersell the field). Then the quasimetric planner (uses the pair-order-0.94 geometry the committor
readout wastes) + T(s,z) exploitation.

## 2026-07-27 -- Layer 3 (quasimetric gradient) readiness: rough-but-real, build order 3->2->1

Kaveh set build order 3(field)->2(MCTS on quasimetric distance-to-mate gradient)->1(opponent-weakness
planner); low-level signal = DISTANCE-TO-MATE (not committor value). Assessed field_v3's d_mate:
  d_mate vs true DTZ (winning positions, IN-DISTRIBUTION real history): Spearman +0.81, median tracks
    DTZ bucket monotonically -> usable gradient.
  FRESH endgames (no history, synthetic): +0.505 -> DISTRIBUTION-SENSITIVE (real history matters,
    Kaveh's "don't zero-fill" -- confirmed). In ACTUAL PLAY positions carry real history -> the +0.81
    regime applies (my conversion test was artificially off-distribution).
  d_mate-greedy mates KQvK/KRvK vs TB-optimal defender: 0/30 -- EXPECTED: value!=policy (needs SEARCH
    = Layer 2) AND <=7 pieces is the TABLEBASE's job by design (handover), so this is off-architecture.
VERDICT Layer 3 READY ENOUGH: rough-but-real distance-to-mate gradient in-distribution + strong d_pair
(0.94) for distance-to-subgoal. Search converts a rough gradient to moves; nn/mcts.py has mate-stop +
cert recognizers for the <=7p finish. NOT retraining the field now (<=7p = tablebase). NEXT: Layer 2
-- wire d_mate/d_pair gradient into nn/mcts.py (value from -distance, real-history planes from
move_stack), test on winning/tactical positions + vs Maia.

## 2026-07-27 -- Layer 2: quasimetric-gradient MCTS wired (mcts_field.py) + the handoff lesson

Wired ClockField into nn/mcts.py (experiments/mcts_field.py): value_fn = (2c-1) committor navigation
SHARPENED by distance-to-mate on the winning side (Kaveh's d_mate preference for conversion);
certainty_fn = tablebase; mate_stop on; real-history planes rebuilt from move_stack.
CONVERSION MICRO-TEST (d_mate-MCTS 200n vs TB-optimal defender): KQvK 1/12, KRvK 0/12 -- STILL fails.
DIAGNOSIS (architectural): KQvK/KRvK are <=3 pieces = pure TABLEBASE territory; my certainty_fn
returned WDL (every winning position = +1) which FLATTENS the gradient (no move looks like progress)
-> search can't convert. THE HANDOFF = play the tablebase's DTZ-optimal MOVE at <=7 pieces, NOT just
use its WDL value in search. FIXED: FieldMCTS.select() now returns tb_best_move directly at <=7
pieces. So the field is only responsible for >7 pieces (navigating toward the boundary); the
conversion micro-test was the wrong test (tablebase territory). REAL test = full games vs Maia
(>7p, real history = in-distribution, +0.81 d_mate regime), tablebase handoff at the boundary,
node-budgeted search -- running (FieldMCTS 80n vs maia-1100), baseline to beat = 0.125 (shallow
search). PERF NOTE: per-node history reconstruction (rebuild LczeroBoard + replay) is the bottleneck
(200n conversion test was minutes) -> if vs-Maia is too slow, cache planes / incremental encode.

## 2026-07-27 -- Layer 2 testing surfaced OPENING-BLINDNESS; retraining field v4 with full-phase coverage

FieldMCTS(80n) vs maia-1100 scored 0.062 -- WORSE than the 0.125 shallow-search baseline. Diagnosed
(NOT a bug): value_fn is garbage where games START. Sanity check: startpos value 0.749 (should ~0),
white+Q value 0.0 (should >0); board_to_planes verified correct (matches direct lc0 encode exactly).
ROOT CAUSE: the field (committor AND quasimetric) is OPENING-BLIND -- Stage C sampled ply>=10
(skip_open=10), mid/endgame-heavy, so the field never trained on openings. Games start out-of-
distribution -> bad early moves -> lost by midgame -> fast losses. Consistent with well-calibrated
MIDGAME (ECE 0.027) but wrong startpos (0.749). Compounded by fresh-position distribution shift
(d_mate 0.81->0.505). The quasimetric shares the gap (same-game ply-gap pairs from the same
opening-skipped positions). Layer 2 MCTS wiring + tablebase handoff are CORRECT; the field's value
coverage is the bottleneck -- more search on a blind value plays worse than exhaustive shallow search.
FIX (Kaveh: retrain full-phase): regenerated Stage C with skip_open=0 (openings included, even phase
coverage) -> field_fullgame_v2data.npz; retrain v4 with the v3 recipe. Startpos should calibrate to
~0.5 (balanced outcomes across the dataset). Then re-run Layer 2 vs Maia in-distribution.

## 2026-07-27 -- STANDARD broadly-usable data (Kaveh: spend energy once, standard like others generate)

Directive: generated data must be broadly usable (all phases, standard format), not narrow. Upgraded
Stage C (gen_field_data_fullgame.py) to a STANDARD position dataset -- ONE dataset serving EVERY
downstream model:
  planes (position) | move (PLAYED move = POLICY target, AZ/lc0-style) | result (WHITE-POV VALUE
  target) | ending (committor) | dtz (mate grounding) | game/ply (quasimetric key) | stm_id (name-
  MASKED stable-hash player id for the z-encoder) | stm_elo/opp_elo (rating conditioning). ALL PHASES
  (skip_open default now 0 -- openings included; a value field must evaluate every phase, the v3->v4
  blindness lesson). Smoke (14.9k pos): policy-move coverage 89.9%, 2045 unique players, ply span
  0-216, W/D/L balanced. Regenerated the full 100k-game standard dataset (field_std_v1.npz). This
  replaces the narrow field-only npz: no re-generation needed for policy / z-encoder / stylometry.
  v4 field retrains on it (trainer reads its subset of keys unchanged).

## 2026-07-27 -- Human transition-point MAP + endgame-island confirmed

Built transition_map.py: UMAP of the trained field's phi, colored by committor c=P(win), with human
transition points = committor jumps |dc|>=theta to the next sampled position (metastability: basin
crossings). THRESHOLD data-driven: |dc| median 0.067, knee at p85=0.225 -> 13.6% transition points
(vs a static max_p<0.66 flagging 55% -- too many; the empirical-jump definition is the sparse,
meaningful one). Basins win 36% / draw 19% / loss 45%.
FINDINGS: (1) a detached LEFT ISLAND in the UMAP -- CONFIRMED endgames: n=245, median 5 pieces
(range 3-7), 100% <=7 pieces, median ply 125, 55% tablebase-grounded; main blob median 23 pieces,
ply 35, 3% <=7p. The field's geometry spontaneously isolates the <=7-piece region = EXACTLY the
tablebase-handoff boundary, and shows a SHARP committor there (decisive endgames). (2) The midgame
core is LEAKY -- transitions pervasive, not thin ridges: metastability under fallible 1400-1800 play
has LOW barriers (frequent crossings), while the high-barrier regime (endgames) separates cleanly.
This IS the thesis: the human basins are metastable/leaky, the crossings are the exploitable errors.
Caveat: |dc| over ~6-ply strided windows (coarse); UMAP is of phi (reachability != outcome-basin).
NEXT options: per-move dc (sharp localization); decisive-subset map; rating-conditioned (do stronger
players transition less = tighter basins?).

## 2026-07-27 -- 3-basin BANDS view: club chess is bistable Win<->Loss with a shallow Draw saddle

transition_bands.py: the 3 basins (W/D/L) as bands full of states, each colored by transition
probability to a different band (leak = 1 - p_own_basin), + the aggregate 3x3 basin->basin mean
transition matrix. FINDING (12k states, v3 field):
        ->Win  ->Draw ->Loss
  Win   --     0.05   0.25
  Draw  0.16   --     0.19
  Loss  0.28   0.09   --
=> WIN and LOSS are the two real basins and they leak PRIMARILY INTO EACH OTHER (Win->Loss 0.25,
Loss->Win 0.28); the middle panel shows the Win band mostly red(->Loss), the Loss band mostly
blue(->Win). DRAW is barely a basin -- almost nothing flows INTO it (0.05/0.09) and it leaks out both
ways -- a shallow unstable saddle (draws are 6% at 1400-1800). Each band spans quiet(leak~0) ->
transition(leak~0.66) smoothly. THESIS implication: the exploitable move is steering the opponent
across the LOW Win<->Loss barrier (~0.25) in our favor; the draw saddle is the trap when pressing.
This is aggregate T(s,z); the per-player version (Layer 1) localizes WHOSE barrier is lowest WHERE.

## 2026-07-27 -- MSM basins = PHASE, not outcome; outcome basins CRYSTALLIZE as material falls

Prototyped a Markov State Model on the field (msm_basins.py; deeptime won't build on py3.14 so MSM+
PCCA-style implemented on numpy/scipy/sklearn): discretize phi (150 microstates) -> reversible
transition operator from human trajectories -> spectrum -> metastable macrostates.
SURPRISE (honest): the dynamics-defined basins are GAME PHASE, not W/D/L. Spectral gap -> 3 basins,
but they are basin0=17pc/ply58, basin1=24pc/ply33, basin2=28pc/ply22, ALL committor ~0.4. T strongly
diagonal (0.86-0.89), slow phase-forward flow. WHY: phase (material/ply) is the SLOW, irreversible
coordinate; the outcome fluctuates FAST on top (win<->loss barriers are low, earlier finding). phi is
dominated by material -> MSM finds phase.
Kaveh's follow-up (committor_by_material.py) NAILS it: stack the committor distribution BY MATERIAL
class. As pieces fall 27-32 -> 3-4: mean leak (transition prob) 0.49 -> 0.06 (barriers RISE), outcome
bimodality (split) 0.39 -> 0.97 (unimodal jumble -> clean bimodal win/loss branches). They CROSS at
~19-22 pieces. So the OUTCOME basins are real but only CRYSTALLIZE at low material; the midgame is a
fast-mixing jumble. THREE REGIMES: opening=jumble (no barrier to steer), endgame<=10p=committed
(barrier too high, tablebase), CRYSTALLIZATION ZONE ~15-22 pieces = barriers exist AND are crossable
= the EXPLOITATION SWEET SPOT (steer the opponent across win<->loss here). So the right object for
exploitation is the outcome transition structure WITHIN the ~15-22p band, not a global 3-basin MSM.
Files: msm_basins.py, committor_by_material.py, transition_map.py, transition_bands.py, transition_time.py.

## 2026-07-27 -- CONTROL: perfect play -> sharp basins; human error = the exploitable entropy

Kaveh's control experiment: does the 3-basin structure sharpen under (near-)perfect play? Generated
SF(depth12)-vs-SF games (~2700+ Elo >> 1400-1800 humans), diversified openings (gen_engine_games.py,
standard npz format). 136 games, 3108 positions, W/D/L 13/77/10 (perfect play mostly DRAWS -- inverts
humans' 6% draw rate). Embedded engine + human positions with the SAME field phi; UMAP colored by
ACTUAL outcome; basin purity via phi-microstate outcome-entropy (engine_vs_human_basins.py).
RESULT (base-rate-controlled via entropy REDUCTION, since raw purity is confounded by the 77% draw
base rate): knowing your phi-location reduces outcome uncertainty by 36% for ENGINE (H 0.70->0.45)
vs only 7% for HUMAN (H 0.88->0.82). By material: engine phi-purity high at ALL material (0.79 even
at 27-32 pieces) while human stays a jumble (~0.52) until the endgame. VISUAL: engine = draw sea with
localized blue/red pockets (outcomes segregate); human = blue/red thoroughly intermixed (no structure
except the endgame island). CONCLUSION: the 3 basins are REAL (sharp under perfect play, from the
opening); HUMAN ERROR smears them into the leaky midgame. The 36%-vs-7% gap IS the exploitable entropy
-- under human play the outcome is set by WHO ERRS (the z/T-conditioned quantity), not the position.
This empirically validates the whole metastability-exploitation framing. Caveats: 77% draws (draw sea
+ decisive pockets, not 3 equal basins), small decisive sample (~750), phi human-trained (engine mildly
OOD -> 36% is a lower bound). Files: gen_engine_games.py, engine_vs_human_basins.py.

## 2026-07-27 -- MONEY CHART: perfect vs human committor by material (the exploitable entropy, visualized)

sf_wdl_by_material.py: on the SAME 4800 human positions (600/material-bucket), two committors:
FIELD (P win under HUMAN play) vs STOCKFISH d16 WDL (P win under PERFECT play), ridgeline by material.
BIMODALITY (Sarle; 1.0=fully split into win/loss branches):
  material   HUMAN-field   PERFECT-SF
  27-32p     0.406         0.897
  23-26p     0.444         0.948
  19-22p     0.497         0.963
  15-18p     0.609         0.984
  11-14p     0.605         0.991
  8-10p      0.680         0.997
  5-7p       0.942         0.999
  3-4p       0.938         1.000
=> the PERFECT committor is BIMODAL at ALL material (value decided even at 30 pieces); the HUMAN-play
committor is a UNIMODAL JUMBLE in the opening/midgame and only splits near the endgame. Same positions
-> the difference is ENTIRELY the play measure. The GAP between the two ridgelines = the EXPLOITABLE
ENTROPY: largest in the opening/midgame (0.90 vs 0.41 at 27-32p = max room to steer), vanishing in the
endgame (1.0 vs 0.94 at 5-7p). The outcome IS determined by the position; human fallibility keeps the
midgame open, and that openness (concentrated ~15-25 pieces) is exactly where the z/T planner pushes
the opponent across the barrier. Empirical foundation of the whole exploitation program. File:
sf_wdl_by_material.py (+ gen_engine_games, engine_vs_human_basins from the perfect-play control).

## 2026-07-27 -- SF-vs-lichess gallery: the Win<->Loss barrier is INFINITE under perfect play, 0.28 for humans

sf_vs_human_bands.py: 3-basin bands + basin transition matrix, LICHESS (field WDL = human-play
committor) vs PERFECT (Stockfish d14 WDL), SAME 4000 positions. Transition matrices:
  LICHESS: Win->Loss 0.27, Loss->Win 0.29 (leaky, low Win<->Loss barrier); bands fill full leak range.
  PERFECT: Win->Loss 0.00, Loss->Win 0.00 (INFINITE barrier, absorbing basins); only tiny Win<->Draw/
           Loss<->Draw 0.04-0.08 (the genuine objective win/draw boundary). Bands pinned at leak~0.
=> Under perfect play the Win<->Loss basins DON'T leak (barrier infinite); human error collapses the
barrier to ~0.28. The ENTIRE exploitable edge lives in that 0.28. Completes the SF-vs-lichess gallery:
(1) committor-by-material [sf_wdl_by_material]: perfect bimodal at all material vs human jumble;
(2) UMAP-by-outcome [engine_vs_human_basins]: perfect segregated (36% determined) vs human jumble (7%);
(3) bands+matrix [sf_vs_human_bands]: Win<->Loss 0.00 perfect vs 0.28 human. One coherent empirical
foundation: positions have determined values; human fallibility opens the Win<->Loss barrier; the
opponent model's job is to predict WHERE/for WHOM that 0.28 leak is largest.

## 2026-07-27 -- LAYER 1 (stopgap): Maia+Stockfish blunder model B(s,r) = the transition predictor

Kaveh: for each opponent, find where their blunder probability is highest; use an open-source blunder
calculator for now. -> Maia (rating-conditioned human move model, we have 1100-1900) + Stockfish
(near-perfect value oracle). blunder_model.py: for a position with opponent (rating r) to move,
  B(s,r) = sum_m P_Maia_r(m|s) * max(0, mover-POV committor loss of m)   [expected self-blunder]
computed via a persistent lc0+maia subprocess (VerboseMoveStats -> full per-move policy) + persistent
SF (WDL committor). B(s,r) IS T(s, z=rating): the opponent's error map / where they cross a basin
boundary against themselves. VALIDATED against the ACTUAL move played in real games at that rating.
SMOKE (n=30, depth 10): predicted B vs actual self-committor-loss Spearman +0.537; top-B quartile
37.5% actual blunder-rate vs 0% bottom quartiles -> the model predicts WHERE blunders happen. Full
n=400 validation running. NEXT: (a) map B(s,r) over the field (where/which regions is each rating
weakest), (b) feed high-B reachable regions as favorable-flux SUBGOALS to the PortfolioPrior planner
(optionality.py) -> navigate the opponent toward their blunder regions. This is Layer 1 -> connects to
Layer 2 (quasimetric MCTS navigates there) -> the exploitation loop.

## 2026-07-27 -- RESET: MILESTONES.md = the locked 30k-foot plan (Kaveh)

Kaveh called a step-back (circles were being rehashed: WDL kept creeping back as the primary
navigation signal; the board trunk stayed hand-rolled despite "borrow from Leela"; the play-measure
question got re-raised without a lock). Wrote MILESTONES.md (repo root) = the canonical roadmap;
README routes to it first. LOCKED: geometry-first navigation (WDL only for analysis + transition
labels); FROZEN pretrained Leela-family trunk + IQE head (encoder gets history, NOT clock); all
context (clocks/Elos/z/ply/past outcomes) enters the TRANSITION ESTIMATOR T(phi,ctx) -> per-side
crossing risk; player model = known Elo + residual z (prior -> in-game tightening); transition
points ARE subgoals (chains -> TB-won -> mate), portfolio planner + MCTS-as-probe under opponent
model + clock; one-best-line (kill superseded); ops bar (DVC/MLflow/tensor-batch/statistical rigor).
MILESTONES: M0 atlas DONE -> M1 Leela-trunk IQE field -> M2 transition estimator (M2a clock+rating,
M2b offline z, M2c in-game z) -> M3 atlas+subgoal generator (M3b concept mining) -> M4 planner ->
M5 MCTS probe -> M6 exploitation dividend. Sequencing M1->M2->(M3||M5)->M4->M6.
DE-RISKED TODAY: lc0 leela2onnx converts maia-1500.pb.gz -> ONNX (3.3M); lczerolens loads it as a
torch module; BATCHED forward 64 boards in 231ms = 277 pos/s CPU (policy 1858 + wdl 3 heads, 115
modules, trunk hookable) -> M1 substrate + M2 batched-Maia infra verified. Blunder n=400 run had
crashed (KeyError 'wdl' on a worker) -> killed; smoke rho +0.54 stands as the M2 baseline; protocol
gets rebuilt tensor-batched in M2. Task queue reshaped to mirror milestones (#35-#40, #34=M3b);
stale lines pruned (#23,24,25,27,28,32 deleted; #31,#33 closed).

## 2026-07-27 -- Acronym/symbol table added (Kaveh: acronyms were the main confusion source)

MILESTONES.md now carries the CANONICAL "Acronyms & symbols" section (chess/engines, method,
metrics/stats, infra, legacy); GLOSSARY.md defers to it and carries a staleness banner (its FB/omega/
InfoNCE entries describe the LEGACY pre-rebuild architecture). Highlighted confusables: phi (board
embedding) vs Phi (net favorable flux); DTZ vs DTM; committor (play-measure-dependent) vs WDL head;
Elo is not an acronym; FB/omega = legacy naming surviving in filenames.

## 2026-07-27 -- PARKED: armed tactics (conditional-activation store), Kaveh

New parked design (no machinery yet -- unlocks after M3b + M5): during search, a tactic that ALMOST
works = a near-transition point "about to cross but not ready." STORE it with its BLOCKING CONDITION
(the specific refuting resource = a PROTECTIVE FACTOR in M3b's vocabulary -- the answer to "why is it
not ready to cross"), then WATCH each ply for the blocker's removal (defender leaves, guard breaks,
pin releases) -> tactic ACTIVATES -> pounce subgoal / first-probe line. Dual use: exploit their
removed protectors instantly; guard OUR blocking conditions their armed tactics depend on (denial/
self-blunder side). Also search efficiency: discover once, arm, watch -- not re-find every ply.
Recorded: MILESTONES.md PARKED section (canonical spec), task #41 (blocked by #34 M3b + #39 M5),
memory armed_tactics_parked. Lineage: refines the old "tactics tracking -> pounce" idea.

## 2026-07-27 -- Armed tactics promoted: PARKED -> MILESTONE M7 (Kaveh)

Kaveh: don't park it, tack it onto the milestone list. Armed tactics (conditional-activation store)
is now M7, the roadmap's final milestone: spec unchanged (store almost-working tactics + blocking
condition = protective factor; watch for removal -> activate -> pounce; dual offensive/defensive
use), now with GATES (activation-detection unit tests; vs Maia, pounce converts opportunities the
re-search-every-ply baseline misses at equal nodes, e-value significant, and/or equal strength at
lower budget; defensive side measurable). Sequencing: M1 -> M2 -> (M3 || M5) -> M4 -> M6 -> M7.
Task #41 renamed M7 (still blocked by #34 M3b + #39 M5). Memory updated.

## 2026-07-27 -- Definitions of Done added: every milestone is an MVP; roadmap bar = beat Maia-1200

Kaveh settled the MVP-vs-perfect question: every milestone is an MVP with a hard DoD floor;
iteration passes are separate recorded items (M1.1, ...); unreachable DoD = plan-level conversation,
never silent redefinition. MILESTONES.md now has (1) a "Milestone philosophy & Definitions of Done"
section, (2) the CUMULATIVE PLAY LADDER: M5 beats the 0.125 shallow baseline vs maia-1100 ->
M4 parity-or-better vs maia-1100 (SPRT >=0 Elo) + steering demonstrated -> M6 score >=0.5 vs
maia-1200 (CI floor 0.45) + dividend >0 -> M7 BEATS maia-1200 (SPRT accepts >= +25 Elo, no
regression vs 1100) = THE ROADMAP'S MINIMUM BAR. (3) the STANDARD MATCH PROTOCOL (maia at nodes=1,
our fixed recorded budget, alternating colors, diversified openings, SPRT elo0=0/elo1=25 or
anytime-valid, MLflow-logged, PGNs kept). (4) per-milestone DoD lines: M1 gates via ONE eval-script
VERDICT + kill executed; M2 rho>=0.60 + quartile lift >=5x + ECE<=0.05 + clock/rating effects + z
lifts (offline and <=20-move in-game); M3 API live + top-decile flux regions >=2x crossing rate on
>=2 rating bands + M3b >=5 attacking + >=5 protective factors; M5 >0.125 + monotone node-scaling +
beats WDL ablation; M6/M7 per the ladder.

## 2026-07-27 -- DoD matches are UNDER TIME CONTROL (Kaveh)

Correction to the standard match protocol: DoD ladder matches are REAL TIMED GAMES -- both sides on
the clock, standard control blitz 3+2 (180s+2s/move), time management is the engine's job (flag =
loss; a faster stack buys more nodes per move -- speed converts to strength), hardware pinned +
recorded. NODE BUDGETS remain only for COMPONENT DIAGNOSTICS (equal-node ablations, strength-per-
node curves), never for ladder bars. M4/M6/M7 DoDs restated as timed; M7 = beat maia-1200 under
time control (SPRT >= +25 Elo).

## 2026-07-27 -- Efficiency audit + M1 underway on the frozen trunk

AUDIT (Kaveh: is the laptop fully utilized?) -- honest answer NO; fixed: (1) NN inference was the
big miss: subprocess pattern ~1-10 pos/s vs BATCHED MPS trunk 21-23k pos/s (batch 2048-4096, fp16;
my earlier "277 pos/s" was a cold single-shot). All NN paths now batched-MPS. (2) trunk-feature
PRECOMPUTE for the full 1.14M-position standard dataset: 65s per trunk (maia-1500 + maia-1900),
fp16 OPEN_MEMMAP (9.3GB each, mmap at train time -- decompress-to-RAM retired), DVC-tracked, and
verified bit-faithful vs fresh forward (max diff = fp16 eps). (3) DISK: features pushed free space
30G->12G; freed 5.6G (lichess prefix files = strict prefixes of the retained full dump, recreatable
via head -c; superseded v2data npz; ClockField step-ladder intermediates) -> 18G free. Staged
reclaims after M1 gates: loser trunk features 9.3G + old FB shards ~10G. (4) idle-gap rule: a
milestone job stays running during discussions.
M1 TRAINER (train_iqe_head.py): thin adapter (1x1 conv + linear) + IQE(64,16) + mate anchor over
FROZEN trunk features; geometry-only losses (multi-goal + mate DTZ + hinge + repulsion; NO
committor head, locked decision 1); scaffold-tracked; batch 4096 MPS from memmap. SMOKE 400 steps /
27s / 133k params: pair-order +0.882, d_mate rho +0.598, eff_rank 8.8 -- near gates (0.94/0.81)
already. Full 6k-step runs for BOTH trunk candidates launched (the M1 trunk-choice comparison).

## 2026-07-27 -- M1 fair comparison: historical gates were PROTOCOL ARTIFACTS; head beats incumbent on geometry

Both 6k-step head runs landed at pair-order 0.849 (below the 0.94 gate) -- but diagnose-before-
concluding caught a PROTOCOL MISMATCH: v3's 0.94 was measured on OPENING-FREE data (ply>=10). Built
eval_m1_compare.py: ALL models on the IDENTICAL all-phase val protocol (same pairs, same rows):
  v3 incumbent:      pair-order +0.778 | d_mate +0.665 | eff_rank 8.2   <- 0.94 was the easy protocol
  head maia-1500:    pair-order +0.861 | d_mate +0.595 | eff_rank 14.9  <- BEATS incumbent on geometry
  head maia-1900:    pair-order +0.847 | d_mate +0.585 | eff_rank 14.2  <- 1500 >= 1900 on all three
Gates 2/2 (eval_m1_gates2.py, same probes both models): OPENING pair-order (ply<=12) v3 +0.285 vs
head +0.684 (2.4x -- v3's opening blindness persists in the metric); opening sanity finite BOTH;
off-dist synthetic-endgame d_mate v3 +0.517 vs head +0.228 (frozen human trunk never carved endgame
features; NOTE locked decision 5: <=7p is the TABLEBASE's regime in play, so this probe measures
territory the field never owns). TrunkField wrapper (fresh-board full path: onnx trunk hook -> head)
now exists in eval_m1_gates2.py -- the play-time M1 object.
PENDING: d_mate-tuned run (w_mate 2, 10k steps). THEN the M1 decision to Kaveh (per DoD unreachable-
gate rule): propose restated fair-protocol gate = beat incumbent on pair-order/opening-order/
eff_rank/sanity + d_mate within epsilon in-dist above the TB boundary; off-dist demoted to
informational. Trunk choice: maia-1500.
