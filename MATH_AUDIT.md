# Math Audit — does the code execute the math we claim?

2026-07-18. Method: 8 parallel module auditors (IQE, QRL loss, FB-rest, MCTS,
negatives/monotone, trainer+shards, playout/policy, calibration/eval-head) over
claims in docstrings, GLOSSARY.md, and ARCHITECTURE_REVIEW.md; every
non-MATCHES finding sent to an adversarial verifier instructed to refute it.
16 verifications completed (15 confirmed, 1 refuted); 27 were lost to a session
limit — **all HIGH findings below were then re-verified by hand** (code read +
numeric check); MEDs marked (v) verified, (h) hand-checked, (u) unverified.

Grading: DIVERGES = code does different math than claimed · WEAKER = valid but
weaker than the words · UNDOCUMENTED = load-bearing math with no stated claim.

## A. Confirmed HIGH

**A1. Mate-depth discount does not exist — "faster mates strictly dominate" is
false.** [mcts.py:49](catspace/nn/mcts.py#L49) claims `MATE_V − k·PLY_DISCOUNT`
at depth k; [mcts.py:162](catspace/nn/mcts.py#L162) applies a **constant** one
unit of `PLY_DISCOUNT` at mate-child creation, with no k anywhere in the file.
Every mate deeper than 1 ply backs up the identical value 0.9999, so the search
is indifferent between mate-in-2 and mate-in-9. *Consequence: a standing
candidate mechanism for the observed conversion shuffling.* Fix: depth-aware
discount at expansion (thread path depth into `_expand`).

**A2. Tree-reuse recalibration mixes units.** [mcts.py:297](catspace/nn/mcts.py#L297)
feeds `_calibrate` the reused root children's `v_init` — already-squashed tanh
outputs of the *previous* move's calibration — then `_squash` applies the
resulting center/scale to **raw** reach of fresh evals. Center/scale computed
in squashed space, applied in raw space. Gated on `tree_reuse=True` (off in
most evals to date). Fix: cache raw reach on nodes, calibrate on raw.

**A3. "Matched compute" A/Bs are not matched.** [policy_fb.py:269](catspace/nn/policy_fb.py#L269)
(`_depth_for_budget`) budgets beam search in **hypothetical uniform tree
nodes** (`root_b + root_b·beam + …`), while MCTS/anytime count **actual fresh
network evals** ([mcts.py:111](catspace/nn/mcts.py#L111)). The cross-module
claim at [policy_fb.py:661](catspace/nn/policy_fb.py#L661) ("one budget unit =
one network eval in ALL") is false; every historical beam-vs-MCTS comparison at
matched `--nodes` was biased. Fix: count real evals in FBSearchPolicy, or
re-state the claim and re-baseline affected A/Bs.

**A4. Sibling repulsion embeds pairs under the wrong ω.**
[train_lichess_fb.py:645](experiments/train_lichess_fb.py#L645) assumes pair
row t corresponds to `_js[t]`; but `irreversible_sibling_pairs`
([hard_negatives.py:127](catspace/nn/hard_negatives.py#L127)) **permutes
internally and skips** pair-less boards, so `om_s = core[1][_js[:n_s]]` assigns
arbitrary other anchors' elo/clock context to each pair. Pairs remain valid and
holdout-safe; only the F-conditioning context is scrambled (noise, not leak).
Fix: return source indices from the generator; index ω with them.

**A5. `monotone_coords` material coordinate is not monotone.**
[monotone_coords.py:42](catspace/nn/monotone_coords.py#L42) uses point values
[1,3,3,5,9]; **promotion increases it** (verified: `a8=Q` → 1.0→9.0), violating
the module's central "only ever SHRINK" claim. Coordinates 1–2 (pawn budget,
castling rights) are correct. Not wired in — no run affected. Fix: use piece
*count* (promotion-invariant), or value pawns ≥ 9.

## B. Confirmed MED

- **(v) Martingale estimator telescopes.** [phead_calibration.py:142](experiments/phead_calibration.py#L142):
  `mean(diff(P))` = `(P_T − P_0)/(n−1)` exactly — an **endpoints-only** drift
  test, not the claimed "per-ply harmonicity residual." Valid for the null
  E[P_T−P_0]=0, but blind to intra-game structure. Fix: per-ply conditional
  residuals (binned by ply/phase) or squared-residual statistics.
- **(v) ECE drops P=1.0.** [phead_calibration.py:133](experiments/phead_calibration.py#L133):
  last bin `[0.9, 1.0)` is open — float32 softmax saturates to exactly 1.0
  (verified), so the *most overconfident* predictions fall in **no bin**. Fix:
  close the top bin.
- **(v) ECE effective sample size overstated.** All positions of a game share
  one Bernoulli outcome; printed n≈per-position counts overstate evidence by
  ~mean game length. Fix: per-game bootstrap CI on ECE (same machinery as the
  drift CI).
- **(h) PID derivative term is two-sided.** [fb.py:526](catspace/nn/fb.py#L526):
  Stooke's Algorithm 2 uses the one-sided `(ΔJ)₊` so D-control only opposes
  *rising* cost; ours lets a good step slam λ toward 0, inviting
  violation-rebound cycling — a candidate contributor to the observed λ
  ratcheting/oscillation. Fix: clamp `deriv` at 0.
- **(h) Push "saturating prior" is weaker than claimed.** [fb.py:495](catspace/nn/fb.py#L495):
  per-pair push gradient = `sigmoid(0.1·(offset−d))` ≈ 0.92–0.98 across the
  entire reachable regime (d=1…15) — near-*constant* force, saturating only
  beyond the offset. Nothing in the softplus keeps reachable pairs closer; that
  job is done entirely by the λ-weighted constraint. Force-balance implication
  for the small-world plateau: uniform push (weight ~1) vs constraint at λ≈15
  compresses the spread equilibrium.
- **(h) Empty-valid batches still update λ.** [fb.py:514](catspace/nn/fb.py#L514):
  θ-penalty correctly skipped, but the fake `sq_dev=0` still drives dual decay
  / PID state. Fix: skip multiplier updates too.
- **(u→h) Uncounted + duplicated certainty evals.** [mcts.py:111](catspace/nn/mcts.py#L111):
  with `certainty_fn` set, each expansion runs up to two uncached, *unbudgeted*
  network passes (children at :185, self at :231) — breaks eval-budget
  fairness for any A/B with `--certainty-stop`. Fix: count + cache on node.
- **(u→h) Eval cache ignores the augmented state.** [mcts.py:113](catspace/nn/mcts.py#L113):
  cache key is bare FEN, but reach injects `rep=path_counts` (and evidence
  blends) — the cache can serve stale values across rep-counts, partially
  defeating the threefold-surface augmentation. Fix: include rep in the key.
- **(v) `--rescue-b` silently discards committor/phead/certainty config.**
  [playout_ab.py:132](experiments/playout_ab.py#L132): `kw = dict(...)`
  reassigns instead of `kw.update(...)`.
- **(h) Goal pool is a ~16-batch sliding window, not "dataset-wide p_goal".**
  [train_lichess_fb.py:566](experiments/train_lichess_fb.py#L566): shards are
  consumed sequentially, so the pool holds temporally-local games. Still ≫
  in-batch shuffle; claim overstated. Fix option: reservoir sampling.
- **(v) Dead draw assertion in tests.** [test_mcts.py:80](tests/test_mcts.py#L80):
  filters `terminal_v == DRAW_V` then asserts `< 0` — vacuous since DRAW_V=0.
  No live test pins draw terminals to 0.
- **(v) The Black-mate regression test doesn't pin the bug it cites.**
  [test_mcts.py:229](tests/test_mcts.py#L229): position has **no draw child**,
  so the old buggy predicate also passes it. Needs a draw-valued terminal
  ordered before the mate.
- **(h) zgoal mate-finals include holdout games.** `collect_mate_finals`
  ([train_lichess_fb.py:181](experiments/train_lichess_fb.py#L181)) applies no
  holdout mask; holdout finals (~1/50) enter the zgoal centroids used in
  training targets. Softens the "never trained on" docstring claim.
- **(h) λ's dedicated LR silently reverts on resume.** [train_lichess_fb.py:495](experiments/train_lichess_fb.py#L495):
  `opt.load_state_dict` restores the checkpoint's group LR; cosine loop
  overwrites main groups but deliberately skips the λ group — so a resumed run
  ignores a changed `--qrl-lambda-lr`.
- **(h) Provenance stamp is stale.** [train_lichess_fb.py:546](experiments/train_lichess_fb.py#L546):
  `data_columns_used` omits `result`/`ply`/`plies_to_end` — the outcome-
  conditioning fix means training *does* consume `result`; the audit stamp
  still claims otherwise.

## C. LOW / doc-drift (fix in one sweep)

fb.py:20 (score is "MRN-spirit" — symmetric metric + free bilinear residual —
not MRN's `d_sym + max-relu d_asym`; matters for citations) ·
fb.py:163 (bin-edge comment lists wrong sequence; true edges
[2,3,4,7,10,17,27,43,69,110,176]) · fb.py:186 (stale hard-margin comment) ·
fb.py:447 (IQE branch returns a quasimetric; docstring says "metric") ·
fb.py:613 (`reach_z` stays a dot product in quasimetric mode — undocumented) ·
GLOSSARY sq_dev/d_rand entries lag the two-sided/goal-pool recipe ·
shards.py:181 (mid-game shard flush breaks last-row-of-game for ~1 game/shard)
· eval_head.py:34 (`torch.manual_seed` in `__init__` clobbers global RNG;
desc/norm head pairs with equal seeds get identical init) · eval_head.py:54
(out-of-domain result leaves uninitialized targets) · playout_ab.py:48
(`b.ply()` correct only because the fixed sets are ply-normalized) ·
policy_fb.py:256 (soft-min "centroid" limit exact only in dot-product mode) ·
hard_negatives.py:138 (deterministic first-6 candidate scan = yield bias, not
unsoundness) · test_hard_negatives.py (no executable guard for certificate
monotonicity or monotone_coords) · test_mcts.py:186 (single-legal-move γ=1
branch uncovered) · iqe.py:15 (α ∈ (0,1) not [0,1] — refuted as cosmetic by
verifier; noted for precision).

## D. Verified MATCHES (the load-bearing positives)

- IQE `_union_length` **proven** = Lebesgue measure of the union (sort +
  shifted-cummax argument; brute-force over 200 randomized cases incl. ties,
  zero-width, duplicates; robust even to r<l inputs).
- IQE direction is the paper convention post-fix; `forward` ≡ `pairwise` to
  exactly 0.0 (training and eval cannot disagree); maxmean + `exp(log_scale)`
  preserve all axioms (20k-triple adversarial search: min slack +8.07).
- The test suite **is** re-flip-proof for IQE direction (dominance test fails
  deterministically on a forward flip; seed-4 diagonal test on a pairwise-only
  flip). Residual gap: argument-order swaps at *call sites*.
- `grad_reverse` dual ascent sign, push sign, two-sided pin, and holdout masks
  on all batch-data loss paths (goal pool, siblings, unreach) verified correct.

## E. Implications for the live small-world problem

Three audited facts bear directly on the d_rand≈1.4 plateau: (1) the push
force is ~uniform ≈0.95/pair over the reachable regime (B) while the two-sided
constraint opposes it at λ≈15 — the equilibrium compresses; (2) the goal pool
is temporally local (B), weakening the far-pair supply; (3) the PID's
two-sided derivative (B) permits the λ ratcheting observed in every run. Fix
order: PID one-sided clamp → sibling ω alignment → reservoir goal pool →
re-run; if the plateau persists, the force balance itself (push weight vs λ
cap) is the next lever, now with the actual gradient numbers in hand.
