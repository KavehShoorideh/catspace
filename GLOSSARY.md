# Catspace Glossary

Reference for the embedding and evaluation concepts used throughout the codebase.

---

## Embedding concepts

### Forward–Backward (FB) embedding
The core representation learned via `TorchFB` (catspace/nn/fb.py). Two convolutional board encoders produce:
- **F(s)**: the forward embedding of a position s — "what's reachable from here?" under optimal play.
- **B(g)**: the backward embedding of a goal position g — "how did we get here?" from the goal's perspective.

The key invariant: **reach(s, g) = F(s) · B(g)**, a dot product in shared embedding space. High reach means g is plausibly reachable from s.

The forward encoder is **omega-conditioned** (depends on player strength and time); the backward encoder sees only the board.

### Omega (ω) conditioning
Player-strength and time conditioning applied to the forward encoder. Omega includes:
- **White Elo bin** (one of 10 buckets: 800–1000, 1000–1200, ..., 2000+)
- **Black Elo bin** (same)
- **Clock bucket** (time control: bullet, blitz, rapid, classical)

The reachable cone depends on who's playing — strong players explore different positions than weak players. So F(s|ω) is "positions reachable by players like these."

### Cosine InfoNCE loss
Contrastive self-supervised learning objective (catspace/nn/fb.py, experiments/train_lichess_fb.py). Given a game state s and a later position g from the same game:
- **Positive pair**: (F(s), B(g)) should have high dot product.
- **Negative pairs**: (F(s), B(g')) for g' from other games should have low dot product.
- **Cosine variant**: embeddings are L2-normalized before the dot product, so reach is a cosine similarity [−1, 1]. This is essential on real boards where activation norms shrink with material loss; unnormalized dot products inherit that decline.

Training on Lichess games (seeded sampling across 11M positions; see LichessPairSource).

### Reach (or reach value)
The cosine similarity `F(s) @ B(g)` after L2 normalization, ranging [−1, 1]. Higher reach means positions s and g are more "aligned" in the embedding space — g is more plausibly reachable from s under optimal play for the given ω.

Not to be confused with "reachability" in chess (whether a position is legally reachable); reach is a learned score from the embedding.

### Goal vectors (z-vectors, zgoals)
Aggregate backward embeddings: summary vectors for outcome positions, stored in the checkpoint.

- **z_MATE_W**: mean B over all checkmate positions where white won (result = +1).
- **z_MATE_B**: mean B over all checkmate positions where black won (result = −1).
- **z_MATE_DIFF**: z_MATE_W − z_MATE_B, the outcome *direction*. Used as the objective in planning and for the zero-label eval baseline.

Computed once per shard scan; re-embedded at every periodic checkpoint save (catspace/nn/fb.py).

---

## Eval concepts

### Descriptive head
A 2-layer MLP that reads F(s) and predicts the result of the game (white win / draw / loss) as a 3-way softmax. Trained on actual game outcomes from Lichess. Answers "what actually happens from here among players like these?"

`EvalHead(n_out=3)` in catspace/nn/eval_head.py. Output is converted to expected score via P(W) + 0.5·P(D).

### Normative head
A 2-layer MLP that reads F(s) and predicts the Stockfish eval as a sigmoid. Trained on [%eval] annotations in Lichess games. Answers "what should happen under best play?"

`EvalHead(n_out=1)` in catspace/nn/eval_head.py.

### Divergence (descriptive vs normative)
The difference between what humans do and what the engine says they should do. Large positive divergence = "humans overperform the eval" (trap positions). Large negative divergence = "humans underperform the eval" (missed wins).

Used to identify positions where the human-eval mismatch is largest (JOURNAL.md, top divergent positions list).

### Frozen probe (vs joint fine-tuning)
By default, F is **detached** during eval-head training — the gradient doesn't flow back to the FB model. This treats the pre-trained representation as fixed and only trains the 2-layer linear probe. `--joint` flag allows fine-tuning both (experimental; off by default until the frozen probe's signal is understood).

---

## Ablation & baseline concepts

### DESC_AUC (Descriptive AUC)
Area-under-curve of the descriptive head's expected-score prediction, separating positions from white-won games from positions from white-lost games (on holdout data). Ranges [0, 1]; 0.5 = random; 1.0 = perfect.

Measures: does the frozen F(s) embedding capture enough outcome signal to discriminate won from lost?

**Calibration at 30k steps:** 0.625 (F), 0.596 (B), 0.636 (F⊕B). The F > B flip shows outcome signal lives in forward structure, not static board features.

### NORM_SPEAR (Normative Spearman)
Spearman rank correlation between the normative head's prediction and Stockfish winprob (on annotated holdout positions). Ranges [−1, 1]; 0 = no correlation; 1 = perfect agreement.

Measures: does the frozen F(s) agree with the engine's eval?

**At 30k steps:** 0.482 (F), 0.376 (B), 0.516 (F⊕B).

### BASE_AUC, BASE_SPEAR (zero-label baseline)
The score of F(s) · z_MATE_DIFF *without any probe training* — pure geometry. Reported as:
- BASE_AUC: AUC of the zero-label readout on won/lost separation.
- BASE_SPEAR: Spearman correlation with Stockfish winprob.

**At 30k steps:** 0.598 (AUC), 0.369 (Spear). When BASE approaches the trained probe values, it means the embedding's geometric structure is doing most of the work; the probe adds only marginal refinement.

### Representation ablation (--repr F / B / FB)
Three versions of the eval-head experiment:
- **F**: probe reads F(s) only (the hypothesis under test).
- **B**: probe reads B(s) only (control: if B ≈ F, outcome info is static board features).
- **FB**: probe reads [F(s), B(s)] concatenated (control: if FB > F, F is losing value-relevant info that B keeps).

**Reading at 30k:** F > B emerged (DESC_AUC 0.625 vs 0.596), confirming outcome signal lives in forward structure, not board-only features.

---

## Decomposition concepts

### Reach slope (REACH_SLOPE, DIFF_SLOPE)
Per-game Spearman rank correlation between ply count and reach value, averaged over 200 holdout games. Measures whether reach monotonically increases / decreases through a game.

- **REACH_SLOPE_WON / REACH_SLOPE_LOST**: correlation with reach toward z_MATE_W. At 30k: +0.671 won / +0.587 lost — both rise through games.
- **DIFF_SLOPE_WON / DIFF_SLOPE_LOST**: correlation with reach toward z_MATE_DIFF (the outcome direction). At 30k: +0.174 won / −0.080 lost — correct separation, winners drift positive.

Used to validate that the embedding captures trajectory-level structure (not just static positions).

### Decomposer (meet-in-the-middle waypoint planning)
Recursive algorithm in catspace/planner/decompose.py. For a hop s → g:
1. Score each pool waypoint m by the bottleneck: min(F(s)·B(m), F(m)·z_g).
2. Split at the argmax waypoint if it improves the reach.
3. Recurse on both legs until every leaf is executable or a give-up rule fires.

**Pool** is a set of candidate waypoints (positions from Lichess holdout games, embedded under the planner's own omega).

### Give-up rules (decomposer termination)
Stop splitting when:
- **no_midpoint**: best waypoint's gain ≤ min_gain — the hop is hard, not long; splitting won't help.
- **unlikely_territory**: bottleneck reaches below tau_floor — we're in territory the field considers very unlikely.
- **dry_out**: two consecutive splits each improved by < dry_gain — diminishing returns; we're converging without arriving.
- **budget**: depth cap reached (anytime algorithm: return the tree so far).

### FRAC_IMPROVED (decomposer metric)
Fraction of start positions where the best waypoint beats the direct reach. At 30k: 0.825 — 82.5% of middlegames have a usable stepping-stone.

### MEAN_GAIN (decomposer metric)
Average improvement in bottleneck reach (Δ in cosine units, [0, 1]) when splitting vs. going direct. At 30k: 0.43 — substantial; median start improves by 43 percentage points.

### Arc property
On synthetic unit-circle geometry, the geodesic-midpoint waypoint is exactly the arc middle. On real boards, "arc property" = chosen waypoints sit plausibly between starts and goals in game progression. At 30k: waypoint ply mean 68.3 vs start ply mean 30.2 — genuine endgame stepping-stones.

### tau_exec, tau_floor (decomposer thresholds)
- **tau_exec**: min reach threshold for a hop to be considered "executable." Calibrated as the median reach of positions ≤10 plies before mate in won holdout games. At 30k: 0.2364.
- **tau_floor**: lower bound; positions with reach < floor are "unlikely territory." Calibrated as q10 of all start reaches. At 30k: −0.1762.

---

## Data & training concepts

### Step / batch / epoch / pair (the training loop)
The units the training logs count in, smallest to largest:
- **Pair** — one data point: (position s, a position g that occurred later in the
  same game). What the model actually learns from ("is g reachable from s?").
- **Batch** — 512 pairs (our `--batch 512`) processed together in one GPU pass.
- **Step** (= *iteration*) — one batch → compute the loss → **backpropagate** →
  nudge every model weight ONCE. So **1 step = 1 batch = 1 weight update**. NOT a
  single data point (that's a pair) and NOT a full data pass (that's an epoch).
- **Epoch** — one full pass over the whole dataset. We sample pairs continuously
  rather than in clean epochs; for scale, 90,000 steps × 512 ≈ 46M pair-draws ≈
  0.8 of a pass over the 55.8M-position 4GB shard.
- **it/s** — iterations (steps) per second; sets wall-clock (90k ÷ ~15/s ≈ 100 min).
- **Learning rate (lr)** — how big each weight nudge is; starts ~3e-4 and cosine-
  decays toward ~0 (coarse adjustments early, fine ones late).
- **Training budget (--steps)** — total weight updates; more = more learning until
  it plateaus/overfits. 90k is the standard for comparability across checkpoints.

A log line `step 72000 loss 4.16 train_top1 0.44 lr 7.0e-05 (15 it/s)` = 72,000
weight updates done; the batch's error was 4.16; on that batch the model ranked
the true future #1 for 44% of pairs; each nudge is currently size 7e-5; running
at 15 updates/second.

### Holdout (held-out test set)
Positions from games where `game_id % 50 == 0`. Never seen during training. Used for validation and eval-head training. ~20% of the data; split deterministically by game ID.

### LichessPairSource (data loader)
Streams (s, g) pairs from Lichess shards: a board s and a later position g from the same game, with g sampled geometrically (gamma-distributed ply distance, default gamma=0.98). Covers 11M positions across 12 1GB shards of rated standard games from 2019-01.

### Shard (data storage)
One .npz file with arrays: packed (bitboard + metadata), meta (turn/castling/etc), ply, clock, eval_cp (Stockfish eval), result (±1 or 0), white_elo, black_elo, game_id. Built by catspace/data/shards.py; one shard ≈ 1M positions ≈ 119MB.

### Lichess [%eval] annotations
Optional Stockfish evaluation in the PGN comment, parsed as eval_cp (centipawns). Used to train the normative head. Not every position has an eval; finite annotation rate (~19k / 224k positions in the 1GB holdout).

---

## Quasimetric & distance concepts

*(Added rounds 11–18. The project moved from a plain cosine "reach" score to a
genuine distance geometry.)*

### Quasimetric embedding
A learned **distance** `d(s, g)` — "how many moves of real play from position s
to goal g" — that obeys the geometry of true travel distances. The key property
is the **triangle inequality**: `d(s, g) ≤ d(s, m) + d(m, g)` for any middle
position m. Going straight can never cost more than going via a detour. Plain
cosine reach (the original FB score) has no reason to respect this, so multi-step
plans ("pin the pawn, *then* win the knight") don't compose reliably. A
quasimetric is built so they do. "Quasi" because chess distance is **asymmetric**
— you can reach a won endgame from the middlegame, but never travel back (you
can't un-capture a piece), so `d(s, g) ≠ d(g, s)`.

Enabled by `TorchFB(quasimetric=True)`. Contrast with the default cosine dot
product.

### Score = r − d (residual minus distance)
The quasimetric readout, MRN-style (Metric Residual Network, Liu et al. 2023):
`score(s, g) = r(s, g) − d(s, g)`. Here `d` is a true, symmetric metric
(triangle inequality guaranteed by construction), and `r` is a small
unconstrained correction term that soaks up the leftover *asymmetric* structure.
Higher score = closer to the goal. When `quasimetric=False`, score collapses back
to the plain dot product, so old checkpoints behave identically.

### Triangle inequality / triangle violation
The property `d(s,g) ≤ d(s,m) + d(m,g)`. A **violation** is a triple where it
fails. Reported as `violation ratio = d(s,g) / (d(s,m) + d(m,g))`; ≤ 1 means no
violation. Used as a health check: our metric `d` has zero violations (guaranteed
by its construction), confirming the distance geometry is structurally sound.

### Ply-gap calibration
A training term that pins the *absolute scale* of the distance to something real.
The contrastive loss alone only cares about *ranking* (is the true future closer
than the wrong ones?) — nothing forces `d = 5` to mean "5 moves away." Ply-gap
calibration adds a penalty that regresses `d(s, g)` toward the **actual number of
plies** between s and g in the real game (a "ply" is one half-move). Without it,
"down a rook with no way back" and "down a rook but recoverable" can score
identically; with it, the first reads as genuinely *far*. Flag:
`--ply-gap-weight`.

### Asymmetry margin / one-way door
A training term teaching that captures are **one-way doors**. For any training
pair where material *dropped* between s and g (a capture happened), the reverse
trip is impossible in real chess. The margin loss pushes `d(reverse) > d(forward)
+ margin`. Derived purely from the direction of real games — no chess rules coded
in. It successfully drove the "arrow of material" error from 27% down to 3%, but
was **shelved** (rounds 15–16): it taxed short-term tactical sharpness, which
endgame play can't spare. Flag: `--asym-weight`.

---

## Goal representation

*(The "corner the king is a region, not a point" thread, round 14.)*

### Goal centroid (and why it's flat)
The default goal vector `z_MATE_W` is the **average** of many checkmate positions.
Measured against exact tablebase distance-to-mate, distance to *any* single
centroid is **flat** — it can't tell mate-in-1 from mate-in-30. Averaging many
mate positions into one point destroys the geometry that distinguishes them.

### Goal-as-region / goal bank
The fix in principle: represent a goal like "checkmate" as a **set** of example
positions (a *bank*), not one averaged point. A `goal bank` is a matrix of many
individual mate-position embeddings (`catspace/goal_bank.py` harvests real
checkmate positions from game data). Scoring a position against a bank uses the
**nearest** exemplar, not the average.

### Nearest-exemplar distance
Distance to the *closest* position in a goal bank. Unlike the centroid, this
**does** correlate with true distance-to-mate (Spearman ρ +0.17, rising to +0.25
after endgame-curriculum training) — evidence that the region idea captures real
structure. **But**: swapping it into actual play *lost* decisively (the readout
keeps chasing whichever mate pattern happens to be nearest this move, which
destabilizes move ranking). Conclusion: the region idea is right, but a readout
can't rescue a representation that's under-trained in those regions.

### Soft-min (logsumexp) aggregation
A smoothed alternative to picking the single nearest exemplar: a temperature-
controlled blend (`bank_tau`) that weights nearby exemplars together instead of
hard-switching. Meant to cure the goal-switching instability above. It recovered
some ground but still lost at play. Both hard-nearest and soft-min are kept as
instruments for future checkpoints.

---

## Search & readout

### Node budget (max_nodes)
How the planner spends its "thinking." Instead of a fixed lookahead depth, the
search is given a **budget of positions to examine** (`max_nodes`, e.g. 200) and
derives its depth per-move from how many legal moves the position has. Modeled on
Leela Chess Zero's node economy (~1500–2000 is "actually playing"; we
deliberately run ~200, roughly 10× below that, so any win has to come from a
*better plan*, not out-searching the opponent). Replaced the earlier fixed-ply
depth setting.

### FBSearchPolicy
The main planner: beam-limited minimax (look-ahead search) that scores leaf
positions with the FB embedding instead of a hand-written evaluation. "Beam"
means it only keeps the top-few candidate moves at each ply to keep the tree
small enough to score in one batched GPU pass. This is the readout that first
lifted arena score from 0.10 to 0.25.

### FBPlanPolicy (plan persistence)
An experimental policy that **commits to a subgoal** for several moves instead of
re-planning every move — the "if the pieces just shuffled around but the plan
hasn't changed, don't re-think from scratch" idea. It picks a target position from
its own search's principal variation, then plays cheaply toward it, only
re-planning when progress stalls, the subgoal is reached, or reach drops sharply.
Tested at n=80 games: no significant advantage over plain search yet (planning
over a not-yet-calibrated value function just follows noise). Kept for re-testing.

---

## Self-play & data curriculum

*(Rounds 13–18. How training data is generated beyond human Lichess games.)*

### Self-play (PI-refinement)
The model plays games against **itself** (and partly Stockfish); those fresh games
become new training data. "PI" = policy iteration — improve, generate new games
with the improved model, improve again. This is the mechanism the AlphaZero line
credits for tactical concepts (forks, pins) emerging with *no* explicit
supervision. Generated by `experiments/selfplay_generate.py`, written in the same
shard format as human data so it plugs into training unchanged.

### Self-play fraction (--selfplay-frac)
The share of training that comes from self-play games vs human games. `0.3` = 30%
self-play. Implemented by `MixedPairSource`, which draws each training batch
wholesale from one pool or the other by a weighted coin flip. **Finding**: at
0.3–0.4 this *hurt* play (see ε-noise) and was removed from the promoted
checkpoint.

### ε-noise (epsilon-greedy, StochasticPolicy)
The planner is deterministic — same position, same move — so self-play against
itself would produce identical, useless games. ε-noise injects a **random legal
move with probability ε** (default 8%) to force game variety. The cost: those
random moves are noise; too many noisy self-play games in the training mix dulled
short-term tactical sharpness (more blunders). This is *why* the self-play mix was
"the play drag."

### Stockfish sparring fraction (--sf-opponent-frac)
Fraction of self-play games (default 30%) where one side is Stockfish instead of
the model — external grounding so the model doesn't just reinforce its own bad
habits. Records only the **moves and the game result**, never Stockfish's
evaluation numbers, so the no-leakage rule (see Leakage gate) is preserved.

### Endgame-start curriculum (--endgame-start-frac)
Fraction of self-play games that begin from a random **winnable endgame**
(K+R vs K, K+Q vs K, K+R+R vs K, etc.) instead of the opening. Targets exactly
the sparse regions human games skip. Measurably improved endgame distance
calibration, though the improvement didn't (yet) translate to better play.

### Winner-POV filtering
An early, cheap proxy for outcome-conditioned training (round 11): keep only
training pairs where the side to move eventually **won** the game — "learn what
winning play looks like, not just what real play looks like." It measurably
helped (+22 centipawns), confirming outcome signal matters, then was **retired**
(Kaveh's call): it discards losing trajectories, which carry exactly the
"bad-future" signal the ply-gap and asymmetry terms need. Evidence kept, mechanism
replaced by real self-play.

---

## Diagnostic instruments

*(The measurement suite built rounds 11–14. "You can't improve what you can't
measure.")*

### ACPL (Average Centipawn Loss) probe
The standard chess-analysis blunder metric, applied to the policy. For each move,
compare a strong Stockfish's evaluation *before* vs *after* the move (from the
mover's side); the drop is the "loss" in centipawns (100 = one pawn). Averaged
over many held-out positions. Human reference: a master loses <20 per move, a
beginner 100+. Our policy runs ~250–330 — it blunders material on a majority of
moves, confirming tactical blindness is broad, not endgame-specific. Also reports
**blunder rate** (moves losing ≥300cp) and **mistake rate** (≥100cp). Stockfish
here is only a *grader*, never a training signal.

### KRRvKBP diagnostic
A narrow, interpretable test position family: White **K+R+R** vs Black
**K+B+P** (king, bishop, pawn). A materially winning endgame that never occurs in
human games, chosen because failures are *diagnosable* ("did it keep the rooks
where the bishop can't reach?") unlike full-board play. A fixed set of tablebase-
verified winning positions (`krrkbp_fixed_set*.json`) is played out vs Stockfish;
the score is win/draw/loss conversion. This is the project's primary planning
benchmark.

### Syzygy tablebases (DTZ, DTM, WDL)
Precomputed **perfect-play databases** for positions with few pieces (≤7). Give
exact ground truth:
- **WDL**: Win / Draw / Loss under perfect play (+2 = winning).
- **DTZ**: Distance To Zeroing move (a capture or pawn move) — used as a distance-
  to-mate proxy.
- **DTM**: Distance To Mate — exact number of moves to forced checkmate.

The chess equivalent of a gridworld's known shortest paths. Used **observationally
only** — to grade how well the learned distance tracks reality (calibration), and
to overlay optimal play in viewers — never as a training label.

### Fitness probe (quasimetric health check)
`experiments/qm_fitness_probe.py` — six instruments measuring whether the learned
distance is *good*, beyond win/loss:
- **Nearest-exemplar Spearman ρ**: does learned distance-to-mate rank-correlate
  with true tablebase distance? (The calibration number that tracks embedding
  progress; centroids stay flat near 0, nearest-exemplar rises to +0.25.)
- **Horizon-stratified retrieval**: can it pick the true future position out of
  63 decoys, at gaps of 1, 2, 5, 10, 20, 50 plies? Reveals *how far ahead* the
  embedding can see. Ours is sharp to ~10 plies (~90%) then falls off a cliff by
  50 (~25%).
- **Asymmetry audit**: fraction of capture-pairs where the (impossible) reverse
  trip scores *closer* than forward — should be ~0.
- **Triangle violation**: structural sanity of the metric (see above).
- **Degeneracy panel**: spread ratio + effective rank (see below).

### Spread ratio / effective rank (degeneracy checks)
Two "is my embedding collapsing?" checks. **Spread ratio** = average distance over
random position pairs ÷ average distance over adjacent (1-ply) pairs; ≈ 1 would
mean all distances collapsed to the same value (degenerate). **Effective rank** =
how many of the embedding's dimensions are actually being used (entropy of the
singular-value spectrum); low rank means wasted capacity. Ours are healthy
(spread ~1.8–2.4, rank ~24–26 of 64).

---

## Methodology terms

### E-value / anytime-valid test (EValueTest)
A statistically rigorous way to call a winner from a sequence of games **without**
p-hacking by peeking. An e-value is a betting score against the "no difference"
hypothesis; crossing `1/α` (e.g. 20 for α=0.05) lets you stop early and reject,
with the false-positive rate provably controlled *at every point you might have
looked*. Reported on every arena run (`catspace/abtest.py`).

### Paired comparison (matched-seed)
Comparing two policies by playing them from the **same positions with the same
random seeds**, then testing the per-position *difference* (Wilcoxon signed-rank +
bootstrap confidence interval). Cancels out position difficulty and luck,
isolating the policy effect. Caveat found: Stockfish's own internal randomness
isn't controlled by our seed, so the pairing is weaker than ideal.

### Incumbent / promotion / pre-registered gate
Research-loop discipline. The **incumbent** is the current best checkpoint; a new
one is **promoted** only if it beats the incumbent on a **pre-registered gate** —
success criteria written down *before* seeing results (e.g. "asymmetry error must
drop below 0.10 AND ACPL must not worsen"). Prevents rationalizing a loss into a
win after the fact. One **lever** (one variable) is changed per round so effects
are attributable.

### Short-horizon vs long-horizon discrimination
The project's central measured tension. **Short-horizon** = ranking the ~30 legal
moves right in front of you (tactics; what ACPL and endgame conversion need).
**Long-horizon** = judging positions many moves out (strategy; the k=20–50
retrieval, calibration, region structure). The finding across 18 rounds: these
**compete** inside one small embedding — nearly every change that improved
long-horizon structure taxed short-horizon sharpness, and play punished that.
This is the identified next target (a two-horizon architecture).

---

## Experiment & logging

### JOURNAL.md
Running lab notes at the repo root. Every experiment entry: what was done, wall-clock timings, VERDICT lines verbatim from the output, and interpretation. Newest entry last. (See research_workflow_journal_timing memory.)

### Artifacts
Checkpoints, logs, and generated outputs under `artifacts/`:
- `data/derived/lichess_fb.pt`: main trained FB model.
- `data/derived/lichess_fb_step2000.pt`: step-2000 backup for before/after.
- `data/derived/eval_heads.pt`: trained descriptive + normative probes.
- `artifacts/generated/logs/`: training and eval logs with timings.

### VAL_TOP1, VAL_TOP8
Validation set metrics during training (catspace/nn/fb.py, val_metrics):
- **TOP1**: fraction of (s, g) pairs where B(g) is the highest-ranked goal among all holdout goals (top-1 recall).
- **TOP8**: fraction where B(g) is in the top 8 (top-8 recall).
- **Chance baseline** (for batch size B): 1/B and 8/B respectively.

At 30k: VAL_TOP1 = 0.033 (16.9× chance), VAL_TOP8 = 0.179 (11.4× chance). The model is significantly better than random but still learning.

---

## Naming & abbreviations

- **FB**: forward–backward (the embedding model).
- **MPS**: Metal Performance Shaders (Apple GPU acceleration).
- **Elo bin**: one of 10 strength buckets; used in omega.
- **DTM / DTZ / WDL**: distance-to-mate / distance-to-zeroing-move / win-draw-loss (Syzygy tablebase ground truth).
- **Cosine reach**: L2-normalized dot product (the original reach metric for real boards).
- **Descriptor vs normative**: game outcomes vs engine evals; two different eval heads.
- **Frozen probe**: F is fixed; only the linear head trains.
- **Waypoint**: an intermediate position in a decomposed plan.
- **ACPL**: average centipawn loss (per-move blunder metric).
- **Quasimetric**: a learned, triangle-inequality-respecting, asymmetric distance.
- **MRN**: metric residual network — the `score = r − d` construction.
- **Ply**: one half-move (one player's turn); "ply-gap" = plies between two positions.
- **PI**: policy iteration (the self-play improve-generate-improve loop).
- **ε-noise**: random-move injection (probability ε) for self-play game diversity.
- **KRRvKBP**: the K+R+R vs K+B+P endgame diagnostic.
- **Incumbent**: the current best checkpoint; a new one must beat it to be promoted.
- **Lever**: one variable changed per experiment round, for clean attribution.
- **Centipawn (cp)**: 1/100 of a pawn, the standard engine evaluation unit.

---

## See also

- **ARCHITECTURE.md**: high-level system design and invariants.
- **README.md**: project overview and lessons learned.
- **JOURNAL.md**: timestamped results, timings, and interpretation.

## Outcome-structured embedding (2026-07-13/14)

- **Outcome poles.** Three learnable goal-space anchors (win / draw / loss) the
  training can push apart, so mutually-exclusive terminals become far-apart
  basins. A *single* pole is one point -- pulling every winning state to it
  collapses the region (kills the within-region hop gradient), which is why the
  hard-pull version tanked play.
- **Pull-to-point vs repulsion (t-SNE analogy).** t-SNE keeps clusters as extended
  blobs by using ATTRACTION only between near neighbours + BOUNDED REPULSION
  between everything (Student-t heavy tail). Our reach/ply-gap term is the
  attraction; **cross-outcome repulsion** (`--repel-weight`) is the bounded
  push between different-outcome states (relu hinge = the saturating tail). No
  attractor point => regions survive while exclusive regions separate.
- **Goal-as-region / soft-min bank.** The planner's goal is the SET of winning
  terminals ("arrive anywhere in the mate region"), scored by soft-min hops to a
  bank of real mate exemplars -- not the blurry average (centroid) of them. The
  `MATE_W` **centroid** was `mean(B(checkmate finals))`: a passive prototype the
  planner steered toward; weak because mates are structurally diverse.
- **On-policy value vs optimal (V^pi vs V*).** The result-label of a state is a
  single Monte-Carlo sample of its ON-POLICY hitting value (blunders count); the
  self-consistent "spring"/Bellman fixed point is the low-variance version. The
  quasimetric (triangle inequality = shortest path = min = optimal) intrinsically
  targets V*, so its distances are the OPTIMAL hops-to-goal; the gap V*-V^pi is
  the competence/difficulty signal.

## Search vs embedding limits (2026-07-14)
- **Search-limited vs embedding-limited.** Two regimes of a planner's play. Below a
  node budget (~800 here) conversion rises with more hop search (SEARCH-limited); above
  it, more search doesn't help and play is capped by the reach field's quality
  (EMBEDDING-limited). Consequence: to A/B whether an embedding CHANGE helps, you must
  evaluate at SATURATION (embedding-limited regime) -- at a search-limited budget every
  embedding ties because search, not the embedding, is the bottleneck.
- **Intrinsic ceiling.** The saturated conversion rate (~0.35 for FB-reach on KRRvKBP vs
  optimal defense) -- the best the representation can guide, independent of search depth.
- **Deterministic-defender playout.** Play metric where the model (hop search) faces a
  tablebase-OPTIMAL, deterministic opponent -> no engine variance -> a paired diff with
  real statistical power, and it captures self-driven play divergence that fixed-position
  move-eval cannot (see playout_ab.py).

## Certainty geometry & the two-timescale field (2026-07-14)
- **Certainty-weighted distance.** d(s,g) = plies + lambda*(-ln P(reach g)): a messy
  position (one winning line among chaos) is FARTHER from mate than a slightly
  longer forced win. -ln P chains multiplicatively, so it satisfies the triangle
  inequality -- certainty and hops unify in one quasimetric. Fixes min-semantics
  optimism ("one winning line = close").
- **Slow field / fast field.** Slow = the trained embedding (stationary geometry,
  updated only by retraining). Fast = an in-memory evidence store (memory_field.py)
  keyed by embedding location, updated every move with search/rollout statistics,
  queried by visit-weighted kNN. Fast evidence is periodically distilled into the
  slow field (the closed loop). "The landscape has shifted" = a fast-field write.
- **Tactic-potential (planned).** A memory-field row whose key is a PRECONDITION
  region ("if opponent plays X the state lands here") and whose payload is a
  plan/tactic + payoff -- conditional knowledge stored in the same store.

## Symbols (one place, mapped to concepts)

| symbol | meaning |
|---|---|
| **P** | true probability that a state reaches a goal (e.g. mate_W) under the plausible-play distribution (our side fallible, opponent resisting) |
| **P̂ ("P-hat")** | the *estimate* of P from data -- the hat always means "estimated from samples". Here: wins/visits over N stochastic rollouts through that state (certainty_rollouts.py). P̂=1 -> every rollout converted (forced-feeling); P̂≈0.5 -> coin-flip messy |
| **F(s)** | forward embedding of a *state* s ("where I am"; conditioned on ω) |
| **B(g)** | backward embedding of a *goal/future position* g ("where I want to be") |
| **d(f, b)** | the quasimetric distance between embeddings -- calibrated to mean "plies of real play"; certainty-weighted target: plies + λ·(−ln P) |
| **z, zgoal** | a fixed goal vector the planner navigates toward (e.g. MATE_W = mean B over checkmate finals; or a learned pole) |
| **λ (lambda)** | exchange rate between certainty and distance: how many plies one nat of −ln P is worth (λ=8: a coin-flip position costs ~5.5 extra plies) |
| **ε (epsilon)** | per-move slip probability in rollouts/self-play -- the "fallibility temperature" of the plausible-play distribution |
| **γ (gamma)** | geometric horizon for sampling goal pairs in training (how far ahead goals are drawn) |
| **ω (omega)** | conditioning context for F: Elo bins of both players + clock bucket (whose dynamics generate the futures) |
| **ρ (rho)** | Spearman rank correlation (our standard "does X track Y" statistic) |
| **k / mate-in-k** | moves (not plies) until forced mate, from Stockfish's mate score |
| **V\*, V^π** | optimal value vs on-policy value: V* assumes best play (tablebase truth); V^π averages over a policy π's actual (fallible) play. The gap V*−V^π is the competence signal |
| **u** | a concept axis: learnable unit direction in embedding space; F(s)·u = the state's value for that concept |
| **τ (tau)** | softmax temperature (InfoNCE tau; pole-tau for the soft pole pull) |
