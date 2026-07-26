# Metastability / Transition-Path Planning — Plan

Status: active (2026-07-26). Companion to JOURNAL.md (chronology) and the memory file
`metastability-planning-architecture`. This is the coherent plan; execute top-down, ground
truth first, keep dimensions low until effective rank saturates.

## 0. Thesis

Play a *fallible* opponent, not a perfect one. Outcome classes {Won, Drawn, Lost} are
**metastable basins**; under optimal play the barriers are infinite, under real play the
barriers are high-but-finite and every crossing is an **error**. Winning = steer the game
toward reachable positions where the opponent is likely to make the outcome-flipping error
and we are not. The edge is an **information asymmetry**: we know better than they do where
the transition zones are. North star: strength-per-node (concepts + shallow search ≈ brute
force), so the deep structure is amortized into learned fields and inference is a shallow
MCTS.

## 1. Architecture (the stack)

1. **Quasimetric field** — single-space IQE `d(φ(s), φ(g))` (composable: triangle inequality
   holds structurally in ONE space; the two-tower broke it → 10.9% violations vs 0.00%).
   Within-basin distances finite; **cross-basin barriers ∞** via repulsion balanced by a
   within-basin anchor (hinge-to-large-M or QRL local≤1 — NOT unbounded, which diverges and
   collapses the geometry). Mate = collapsed attractor (d→0); stalemate/draw/loss = ∞
   repellers. Needs **off-optimal negatives + local 1-ply resolution** (field trained only on
   optimal lines picks a distance-reducing move just 52.7% of the time — a coin flip; the
   +0.957 pair-ordering metric oversold it).
2. **Committor** `c(s) = P(win | s, play)` — the outcome coordinate; c≈0.5 iso-surface = the
   transition-state ridge. **Outcome-defined, therefore oracle-free** (see §3).
3. **Transition-probability predictor** `T(s, ω)` (CNN/transformer, low-dim) — per-direction
   over the six inter-basin crossings, ω-conditioned. 2-D texture: SHARP = both flip probs
   high, QUIET = both low, FAVORABLE = win-flip high / loss-flip low, DANGEROUS = inverse.
4. **MCTS planner** — maximize **expected SCORE** (W=1, D=½, L=0): the committor generalized
   over all three basins. Subsumes losing→max L→D swindle, winning→avoid W→D, drawn→max D→W.
   One scalar backed up in the tree (do NOT hand-balance the flip probs; expected-score nets
   them, sharpness = its variance). `T` shapes search toward *reachable* favorable-flux ridges
   (field distance = within-basin reachability filter). Navigate to unseen `g` =
   argmax_g score-gain(g)·reachability(d(cur→g)). **Risk-appetite knob** (score variance
   tolerated): need-a-win (c≈0.5) → tolerate SHARP; winning → seek QUIET. = contempt,
   principled. This is transition-path theory: maximize reactive flux toward the better basin.

## 2. Opponent model & information asymmetry

Opponent has latent type `θ` (skill, style, prep). Hold a prior `P(θ)`, sharpen the posterior
`P(θ | moves)` online (static θ for now; tilt/time → drifting θ_t via particle filter later).
Opponent is bounded-rational: `π_θ(a|s) = softmax(V_θ(s·a)/τ)` on their *distorted* value
`V_θ`. Three value functions: `V_ref` (strong reference, §3), `V_me` (ours — sharp in prep
regions, can locally beat the reference), `V_θ` (theirs). The exploitable signal at a node is
the opponent's **regret under a reliable reference**, `max_a V_ref(s·a) − E_{a~π_θ}[V_ref(s·a)]`,
restricted to where `V_ref` is trustworthy AND our estimate beats theirs:
`edge = (their deviation from consensus) − (our deviation)`. Steer toward reachable
high-`edge`, low-perceived-risk states (traps they can't see). Disagreement ≠ edge — being
*more right* is (calibrate `V_me`; overconfident prep springs the trap on ourselves).

## 3. Epistemics — no oracle, and the SF reliability map

There is no true midgame `V*` (chess unsolved > 7 pieces). Stockfish is a strong but fallible
**reference**, not truth. Operative principle: **whoever plays closer to the reference wins**
(a validated strength proxy — checkable). So the framework is **relative**: exploit *relative*
proximity to the reference, no absolute truth needed. Ceiling: the proxy self-limits as our
strength approaches the reference's ("can't surpass the teacher by imitation").

**What does not break: the committor is outcome-defined, hence oracle-free.** SF is only a
cheap, dense *estimator* of it. Value-signal hierarchy (weak→strong oracle): single engine →
ensemble → tablebase (TRUE, ≤7 pieces) → actual outcomes / self-play (TRUE, sparse, noisy).
Migration is a value-source upgrade, not a redesign: **A** (now, weak) SF; **B** (approaching)
ensemble + reference-disagreement handling + blend outcome targets where reference is unsure;
**C** (matching/exceeding) self-play/outcome-grounded (AZ escape from the teacher), engines →
sparring partners, tablebase → hard anchor.

**The keystone: the SF reliability map.** Calibrate SF against tablebase truth in the endgame
(where truth exists) to learn *where SF is trustworthy* — as a function of position features,
search depth, eval margin. Reference-disagreement (SF vs lc0) extends a trust signal into the
midgame (agreement = trustworthy; disagreement = genuinely sharp / do not charge the human with
an error). Everything downstream weights the reference by this map. **Build the value signal as
a swappable, ensemble-able, uncertainty-aware module from day one** so A→B→C is a config change.

## 4. Data

**Lichess** for fallibility + prep asymmetries + `π_θ` (real, specific, cohort-dependent human
error — richer than a smoothed engine). Naturally stochastic (many humans → distribution of
continuations) → transition *probabilities* are estimable for free. **Engine ensemble
(SF + lc0)** = the reference / `V_ref` labeler. **Tablebase** = truth anchor (≤7 pieces).
Maia (1200–1900, on disk) = a ready-made `π_θ` baseline. Assets present: lc0 + `t1-512x15x8h`,
Maia nets, Stockfish, committor code (`committor_root_loop.py`), ω embeddings, single-space
quasimetric MVP (`quasimetric_shared_v1.pt`).

## 5. Staged execution (ground truth first)

- **S1 — SF reliability map (endgame).** SF eval vs tablebase WDL/DTM across classes, depths,
  margins → where is SF trustworthy? Parallel SF workers; tensor calibration. *(executing now)*
- **S2 — field that MATES.** DONE/REFRAMED (2026-07-26). WDL basins + ∞ hinge-to-M barriers
  SOLVED the stalemate/blunder defense (kept-win 88.7%, won-d 20 vs draw-d 468); field is a good
  VALUE (d-vs-DTM +0.81) -- but greedy can't CONVERT (mate 0.4%) and a single MATE-goal collapses
  rank (1.7). REFRAME (Kaveh): policy comes from the PLANNER not greedy field; at deployment the
  TABLEBASE mates the endgame (as real engines do); the learned field's real job is the MIDGAME.
- **S2b — field + shallow search.** Minimax (field=leaf value, checkmate=+inf, draw/loss=-inf)
  vs tablebase-optimal defense; mate-rate by depth. Validates value+planner=policy.
- **S3 — transition predictor `T(s,ω)`** on tablebase-exact only-move labels; validate it
  predicts where a fallible defender (Maia) actually errs.
- **S4 — cohort deviation / asymmetry field** on lichess + engine-ensemble reference (weighted
  by the S1 reliability map): does exploitable regret concentrate in nameable structures?
- **S5 — flux MCTS planner** maximizing expected score; beat greedy-to-mate vs a fallible
  defender; then scale to midgame.

## 6. Engineering standards

Parallelize (process pools for engine/tablebase; batch everything on device). Tensor ops over
python loops. **Low dimensions until effective rank saturates, then increase.** Effective-rank
health gate on every training run. Checkpoints with metadata, no overwrites; MLflow not
hand-rolled (see TRAINING_STANDARDS.md). Short validation run → commit → full run.
