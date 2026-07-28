# Session handoff — catspace M2→M3 (2026-07-27)

Load this into a new session to continue. Companion doc: **`REACHABILITY_STRUCTURE_BRIEF.md`** (the
literature-search brief for the live open question). Canonical plan: **`MILESTONES.md`**. Running log:
**`JOURNAL.md`** (this session's entries at the end).

---

## TL;DR — where we are

- **M2 (transition estimator + player model) is BUILT and COMMITTED** — per-player style `z`, the online
  `(Elo,z)` estimator, and the crossing-risk primitive. All verdicts are committed script outputs.
- **M3 (atlas/subgoal) started, then triggered a foundational rethink of the reachability metric** that is
  the **live open thread**: reachability should be a **policy-conditioned probability** (Successor
  Representation / contrastive-RL family, retrieval-factored `⟨φ(s|z), ψ(g)⟩`) — **NOT** the IQE
  quasimetric. Next action: **literature search from a fresh session** using
  `REACHABILITY_STRUCTURE_BRIEF.md` (web-search budget was exhausted here), then build the contrastive
  reachability head.

---

## What was built & committed this session

- **M2b — per-player style residual `z`** (commit `8a6b60b`). Matilda-style residual over a *frozen Maia-2*
  human-move base + frozen `φ`. Finding: the **direct additive `z` overfits** (net −0.042 nats vs raw
  Maia on held-out players) but *discriminates* the player. **Kaveh's fix = infer-then-condition**: use the
  recovered `z` only to *retrieve* the k≈50 nearest *clean* training-player styles (Elo-banded) and predict
  with their blend → **beats raw Maia +0.006–0.009 nats AND is player-specific** (+0.005–0.010 vs a
  rating-matched wrong player's `z`), 525 held-out players / 62.9k moves. **`μ=0`** (raw Maia = the
  universal rating prior). Files: `catspace/style/{model,recover,dataio}.py`, `experiments/m2b_*.py`,
  `stats.paired_nll_ci`. Includes a **resumable crash-safe shard cache** (atomic temp-then-rename shards,
  disk pre-flight, resume rebuilds only missing shards).
- **M2c — online `(Elo, z)` estimator** (commit `e2b0fe1`, `catspace/style/estimator.py`). One filter per
  opponent, fed by their own moves (history-prior + live, **recency-weighted** for drift), uniform for all
  game counts (handles zero-games and <20-games — verified). **Elo**: known→tight, unknown→**estimated
  from moves** via Maia's 11 rating buckets (coarse: Elo-MAE 142 @ 40 moves vs 205 no-info); retrieval band
  widens with Elo uncertainty. Break-even: **identity from ~10 observed moves; beats the prior from ~40–80
  (cold)**; Elo-banded retrieval ±100 → +0.009.
- **Crossing-risk primitive** (commits `8155c06`, `676877c`, `catspace/transition.py`). Expected objective
  committor swing under a move-model: `risk(s|model) = Σ_m p_model(m)·max(0, committor(s)−committor(s·m))`.
  **REFEREE = Stockfish** (Kaveh corrected an initial trunk-WDL version — SF is the locked oracle). Weaker
  opponent crosses ~1.4–3× more (Maia-1100 vs 1900; ρ≈0.64 vs realized SF crossings). One primitive, both
  sides: opponent move-model → exploit flux; **our** move-model → self-blunder / denial term.
- **M3 groundwork** (commit `709c9d9`, `m3_*.py`). Flux gate: the fast `T` predictor finds crossing
  *location* strongly (top-decile 4.7–4.9× base rate) but its **ranking is rating-invariant** (Spearman
  0.95) — crossing *location* is position-driven, strength scales *magnitude*. Strength-gradient probe
  confirms it.
- **`reachability_target` + constraint tests** (in `experiments/losses.py`). The probability-adjusted
  target `log1p(n_moves / P(path|z))` with executable invariant tests (forced→1, 1/1000→1000, monotone,
  additive, floored) — built at Kaveh's request so the constraints can't silently break.

Uncommitted experimental scripts from the reachability thread: `experiments/train_cond_*.py`,
`build_cond_*.py`, `train_cond_reach2.py`, `precompute_trunk_wdl.py` (kept for reference; the thread's
conclusion is that this IQE-conditioning approach is the wrong structure — see below).

---

## Big design decisions this session (recorded, dated, in MILESTONES + memory)

- **`μ = 0`** — raw Maia IS the universal rating prior; a casual-pool prior imposed on active players is a
  population mismatch (and had a `μ/Δ` gauge bug). Opponent base-rate (measured): game-weighted **81% of
  opponents have ≥40 games**, so the individual-`z` pathway is the MAIN one. *(memory:
  opponent_base_rate_and_mu_zero)*
- **Infer-then-condition** — the recovered `z` is a *retriever*, not an additive predictor. *(memory:
  infer_then_condition_z)*
- **`z` may carry strength + structure-competence** — no purity firewall; exploitable per-player signal is
  the goal. Only validity checks kept (held-out players, wrong-`z` placebo, identity-init). *(memory:
  style_z_allows_strength)*
- **Stockfish is the referee** for committor/basins (locked plan; not the trunk WDL).
- **Decision-3 amendment (proposed, then EMPIRICALLY WALKED BACK)** — we tried to merge `d` and `T` into a
  single context-conditioned IQE head. It failed (next section); the merge is **not** adopted.

---

## The reachability-structure thread (THE LIVE CRUX)

We tried to make the reachability field itself `z`-conditioned (FiLM adapter over frozen `φ` + the original
IQE quasimetric objective; then the probability-adjusted target `log1p(n_moves/P(path|z))` with `P` from
the estimator's move probs along observed paths). **`z`-lift ≈ 0 on every target** (raw ply-gap =
player-independent; multi-step path-surprisal = not recoverable from the endpoints the field sees; 1-step =
degenerate). We audited the code, built constraint tests, checked `z` isn't degenerate (eff_rank 15.87/16).

**Conclusion (robust across 3 target formulations + a math argument):**
- `φ` encodes **structural** reachability; the `z`-dependent transition **probability** lives in the
  **policy** (the estimator's move distribution), not the metric geometry.
- The **IQE guarantees the triangle inequality = MIN / shortest-path**. Probabilistic reachability is a
  **SUM / expected-value** object (Kaveh: "ply count × probability, summed" = expected passage time =
  **MFPT**; and `P(reach)` = a hitting probability). A MIN structure **cannot** represent it. Wrong
  operator, not wrong `z`.
- **No absolute base exists** (Stockfish isn't perfect; data is human) — reachability is intrinsically
  probabilistic, policy-conditioned, two-player (opponent *veto*). Human trajectory data **is** the
  reachability distribution.
- **WDL and expected-plies are the same object, two readouts** of the policy's absorbing Markov chain
  (fundamental matrix `N=(I−Q)⁻¹`: absorption prob = WDL; expected steps = MFPT). Kemeny & Snell.

**What Kaveh actually wants:** given a **memory bank of points** `{gᵢ}`, compute **`P(reach gᵢ | s, z)`** for
each (WDL = the outcome-basin special case) — retrieval-indexed. **The literature fit is Successor
Representation / Contrastive RL** (`⟨φ(s|z), ψ(g)⟩ ≈ P(reach g)`, trained contrastively on human
trajectories). Details + authors + search queries in `REACHABILITY_STRUCTURE_BRIEF.md`.

---

## NEXT STEPS (pick up here, in order)

1. **Literature search from a fresh session** (this one's web budget = 0/200 left) using
   `REACHABILITY_STRUCTURE_BRIEF.md` — verify the Successor-Representation / Contrastive-RL framing, lc0
   MLH, absorbing-chain math, and the novelty (opponent-/strength-conditioned goal-reaching probability).
2. **Prototype the contrastive successor-features reachability head**: `⟨φ(s|z), ψ(g)⟩ ≈ P(reach g|s,z)`,
   trained contrastively on human `(state, future-state-in-same-trajectory)` pairs vs random negatives,
   `z`-conditioned, served via the existing **`d_to_goal`/`d_to_region` vector-DB retrieval primitive**
   (already built). WDL, basins, subgoals, armed-tactic triggers all become one query: "reach this point."
3. Keep the IQE/quasimetric **only** if the planner separately needs a shortest-*forcing*-route (MIN)
   object — a different question from reachability-as-probability.
4. Then resume M3 (atlas/subgoal generator) on the probability field × the crossing/flux primitive.

---

## Assets & bookkeeping

- **Data:** `data/records/player_games_rapid` (6k individual + 40k provisional, single-TC rapid);
  `data/derived/m2b/{cache_3k, cache_dense, positions_*}`; `cond_reach_data.npz`, `cond_flux_data.npz`
  (masked `z` + trunk-WDL basins); `transition_data_labeled.npz` (M2a, SF committor labels);
  `m2a_trunk_wdl.npy`. **Model:** `artifacts/experiments/m2b_style_3k.pt`.
- **Memory files** (`~/.claude/projects/-Users-kav-code-remote-github-catspace/memory/`):
  style_z_allows_strength, opponent_base_rate_and_mu_zero, infer_then_condition_z,
  matilda_residual_style_embedding, milestones_locked_roadmap (+ others).
- **Loose ends:** full 6.9k-player `z` run parked (3k settled the science); recency/drift validation needs
  **multi-month timestamped** data (single month has ~no drift); leftover procs to reap
  (`m3_primitive_bands` Stockfish engines, an `assistant_server.py` on a port); disk at ~96% (19G free).
- **Milestone status:** M0 done, M1 done, **M2a/b/c MET**, M3 in progress (blocked on the reachability
  structure above), M4–M8 pending.

---

*The through-line to carry forward: reachability is a **policy-conditioned probability** —
`⟨φ(s|z), ψ(g)⟩` in the Successor-Representation / Contrastive-RL family, learned from human trajectories —
**not** a shortest-path quasimetric. WDL is its outcome-basin special case; the crossing-risk primitive is
a 1-step slice of it. Verify the literature, then build the contrastive reachability head.*
