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
  4. Reusable tooling built: eval_variant (CI+e-value), n=200 held-out set,
     conversion_compare, move_ab (+ its own null-result caveat), playout_ab
     (deterministic-defender, the metric with real power), reach_curvature, wdl_regions.
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
